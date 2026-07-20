"""GRPO++ + SPRM Training Script — Phase 4a.

Trains Qwen2.5-Coder-7B-Instruct with GRPO++ using SPRM deviation scores as a
step-level penalty on top of SWE-bench task success reward.

Reward: r = task_success - α × mean(SPRM_waste_per_tool_call)

task_success can be measured in two ways:
  1. Docker execution (default): runs patch in SWE-bench Docker harness
  2. Opus judge (--use-opus-judge): Claude Opus 4.8 reads patch + issue

Architecture: External agent rollout → reward scoring → TRL GRPOTrainer.

  CRITICAL DESIGN NOTE: Qwen generates raw diffs (no tool calls) by default.
  Raw diffs give SPRM nothing to score → waste=1.0 for all → zero reward variance
  → compact filter discards all groups → GRPO learns nothing.

  Fix: RolloutAgent (rollout_agent.py) gives Qwen bash + str_replace tools and
  runs a proper multi-turn agent loop. The reward function receives agent
  trajectories (ToolStep list) rather than raw model completions.

  The reward function is called by TRL as normal, but ignores TRL's `completions`
  argument and instead runs fresh agent rollouts using the current model weights.
  This is the standard "external rollout" pattern for tool-use GRPO.

GRPO++ modifications (vs vanilla GRPO):
  - No KL penalty (beta=0)
  - No reward std normalization (Dr. GRPO style)
  - Compact filtering: skip groups where all rewards identical
  - Decoupled clip bounds (DAPO style)

Usage (50-task PoC with Opus judge — no Docker needed):
    python train_grpo_sprm.py \\
        --model Qwen/Qwen2.5-Coder-7B-Instruct \\
        --sprm-checkpoint ../oracle_trajectory/sprm_model \\
        --num-tasks 50 --num-generations 4 --alpha 0.1 \\
        --use-opus-judge --output-dir ./grpo_swe_7b_poc

Usage (with Docker):
    python train_grpo_sprm.py \\
        --model Qwen/Qwen2.5-Coder-7B-Instruct \\
        --sprm-checkpoint ../oracle_trajectory/sprm_model \\
        --num-tasks 50 --num-generations 4 --alpha 0.1 \\
        --output-dir ./grpo_swe_7b_poc

Requirements:
    pip install trl>=0.22.0 swebench transformers torch bitsandbytes peft
    For Opus judge: pip install anthropic; set ANTHROPIC_API_KEY
    For Docker: Docker daemon running with SWE-bench images pre-built
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# rollout_agent is in the same directory
sys.path.insert(0, str(Path(__file__).parent))
from rollout_agent import RolloutAgent, RepoSandbox, Trajectory, run_rollouts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

ALPHA_DEFAULT = 0.1   # SPRM penalty weight
SWEBENCH_SPLIT = "princeton-nlp/SWE-bench_Verified"

# SPRM label mapping (must match train_sprm.py)
CLASS_NAMES = ["necessary", "doom_loop", "over_read", "other_waste"]
WASTE_CLASSES = {1, 2, 3}  # everything except necessary

# Qwen tool call patterns (model may use either format)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
PATCH_RE = re.compile(r"```(?:diff|patch)\n(.*?)```", re.DOTALL)

# ── SPRM Scorer ───────────────────────────────────────────────────────────────

class SPRMScorer:
    """Wraps the trained DeBERTa SPRM to score individual tool call steps."""

    def __init__(self, checkpoint_path: str):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        p = str(Path(checkpoint_path).resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(p, use_fast=False, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(p, local_files_only=True)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        log.info(f"SPRM loaded from {checkpoint_path} on {self.device}")

    def score_step(self, tool_name: str, context: str = "", prev_labels: list[str] | None = None) -> dict:
        """Score a single tool call. Returns per-class probs + deviation score."""
        prev_str = " → ".join(prev_labels[-5:]) if prev_labels else "START"
        text = (
            f"CONTEXT: {context[:200]}\n"
            f"PREVIOUS: {prev_str}\n"
            f"CURRENT_TOOL: {tool_name}\n"
            f"HINT: "
        )
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        pred_class = int(torch.argmax(logits, dim=-1).item())
        waste_prob = sum(probs[i] for i in WASTE_CLASSES)
        return {
            "tool": tool_name,
            "pred_class": CLASS_NAMES[pred_class],
            "probs": {c: probs[i] for i, c in enumerate(CLASS_NAMES)},
            "necessary_prob": probs[0],
            "waste_prob": waste_prob,
        }

    def score_trajectory(self, tool_calls: list[str], context: str = "") -> list[dict]:
        """Score a full list of tool call names. Returns per-step dicts."""
        results = []
        prev_labels = []
        for tool in tool_calls:
            s = self.score_step(tool, context=context, prev_labels=prev_labels)
            results.append(s)
            prev_labels.append(s["pred_class"])
        return results

    def mean_waste(self, tool_calls: list[str], context: str = "") -> float:
        """Mean waste probability across all tool calls in trajectory."""
        if not tool_calls:
            return 1.0  # penalize empty trajectories
        scores = self.score_trajectory(tool_calls, context=context)
        return sum(s["waste_prob"] for s in scores) / len(scores)


# ── Opus 4.8 Judge ────────────────────────────────────────────────────────────

OPUS_JUDGE_MODEL = "claude-opus-4-8"

OPUS_JUDGE_SYSTEM = """You are an expert software engineering evaluator.
Given a GitHub issue description and a patch (unified diff), judge the patch quality.

Respond with EXACTLY one of:
  RESOLVED       — patch correctly and completely fixes the issue
  PARTIALLY      — patch addresses the right area or logic but is incomplete or has minor errors
  NOT_RESOLVED   — patch is wrong, irrelevant, or would not help fix the issue

Nothing else. No explanation."""

OPUS_JUDGE_USER_TMPL = """Issue:
{issue}

Patch:
```diff
{patch}
```

Judge this patch (RESOLVED / PARTIALLY / NOT_RESOLVED):"""


class OpusJudge:
    """Uses Claude Opus 4.8 as a reward judge instead of Docker execution."""

    def __init__(self, api_key: str | None = None, timeout: float = 60.0):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self.timeout = timeout
        self._call_count = 0
        self._total_cost = 0.0
        log.info(f"OpusJudge initialized with model {OPUS_JUDGE_MODEL}")

    def judge(self, issue: str, patch: str) -> float:
        """Return 1.0 if Opus thinks the patch resolves the issue, else 0.0."""
        if not patch.strip():
            return 0.0

        user_msg = OPUS_JUDGE_USER_TMPL.format(
            issue=issue[:3000],   # cap to avoid huge prompts
            patch=patch[:4000],
        )
        try:
            resp = self._client.messages.create(
                model=OPUS_JUDGE_MODEL,
                max_tokens=16,
                system=OPUS_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = resp.content[0].text.strip() if resp.content else ""
            text_upper = text.upper()
            if text_upper.startswith("RESOLVED"):
                score = 1.0
            elif text_upper.startswith("PARTIALLY"):
                score = 0.3
            else:
                score = 0.0
            # Track cost ($15/M in, $75/M out for Opus 4.8)
            self._call_count += 1
            cost = resp.usage.input_tokens * 15e-6 + resp.usage.output_tokens * 75e-6
            self._total_cost += cost
            log.info(f"Opus judge: {text!r} → score={score} (${cost:.4f})")
            return score
        except Exception as e:
            log.warning(f"Opus judge API error: {e}")
            return 0.0

    def summary(self) -> dict:
        return {
            "judge_calls": self._call_count,
            "total_judge_cost_usd": round(self._total_cost, 4),
        }


# ── SWE-bench Docker Executor ─────────────────────────────────────────────────

class SWEBenchExecutor:
    """Runs a patch against a SWE-bench instance in Docker and returns pass/fail."""

    def __init__(self, timeout: int = 300, log_dir: Optional[str] = None):
        self.timeout = timeout
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self._verify_docker()

    def _verify_docker(self):
        try:
            subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=10)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Docker daemon not running. Start Docker before running GRPO training.")

    def run(self, instance_id: str, patch: str, run_id: str = "grpo") -> bool:
        """Apply patch to instance, run tests, return True if resolved."""
        try:
            import docker
            from swebench.harness.run_evaluation import run_instance
            from swebench.harness.test_spec.test_spec import make_test_spec
            from datasets import load_dataset

            swebench = load_dataset(SWEBENCH_SPLIT, split="test")
            task_map = {r["instance_id"]: r for r in swebench}

            if instance_id not in task_map:
                log.warning(f"Instance {instance_id} not found in SWE-bench Verified")
                return False

            instance = task_map[instance_id]
            test_spec = make_test_spec(instance)
            client = docker.from_env()

            pred = {
                "instance_id": instance_id,
                "model_name_or_path": "grpo-qwen-7b",
                "model_patch": patch,
            }

            result = run_instance(
                test_spec=test_spec,
                pred=pred,
                rm_image=False,
                force_rebuild=False,
                client=client,
                run_id=run_id,
                timeout=self.timeout,
            )

            resolved = bool(result.get("resolved", False))

            if self.log_dir:
                log_path = self.log_dir / f"{instance_id}_{run_id}.json"
                log_path.write_text(json.dumps({"instance_id": instance_id,
                                                "resolved": resolved,
                                                "run_id": run_id}, indent=2))
            return resolved

        except Exception as e:
            log.warning(f"Docker eval failed for {instance_id}: {e}")
            return False


# ── Agent Reward Wrapper ──────────────────────────────────────────────────────

class AgentRewardWrapper:
    """Runs the RolloutAgent to collect real tool-call trajectories for reward scoring.

    TRL's GRPOTrainer generates completions internally, but those completions are
    raw diffs (no tool calls) because the model isn't given tools during generation.
    This wrapper intercepts reward function calls and re-runs the model as a proper
    tool-calling agent to produce trajectories that SPRM can score.

    The wrapper caches one shared RepoSandbox per instance to avoid re-cloning.
    """

    def __init__(
        self,
        model,
        tokenizer,
        instance_map: dict[str, dict],
        max_steps: int = 12,
        max_new_tokens: int = 1024,
        github_token: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.instance_map = instance_map    # instance_id → full SWE-bench record
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN") or ""

    def rollout(self, instance_id: str) -> Trajectory:
        """Run one agent rollout for the given instance. Returns Trajectory."""
        instance = self.instance_map.get(instance_id)
        if instance is None:
            log.warning(f"Instance {instance_id} not in instance_map — skipping rollout")
            from rollout_agent import Trajectory as T
            return T(instance_id=instance_id, messages=[], steps=[], patch="",
                     submitted=False, truncated=False, elapsed_s=0.0)
        agent = RolloutAgent(
            model=self.model,
            tokenizer=self.tokenizer,
            instance=instance,
            max_steps=self.max_steps,
            max_new_tokens=self.max_new_tokens,
            github_token=self.github_token,
        )
        return agent.run()


# ── Trajectory Parser ─────────────────────────────────────────────────────────

def parse_tool_calls(trajectory_text: str) -> list[str]:
    """Extract tool names from a trajectory (Qwen may use XML or JSON blocks)."""
    names = []
    for m in TOOL_CALL_RE.finditer(trajectory_text):
        try:
            obj = json.loads(m.group(1).strip())
            name = obj.get("name", "")
            if name:
                names.append(name)
        except (json.JSONDecodeError, AttributeError):
            pass
    if names:
        return names
    # Fallback: JSON code blocks
    for m in JSON_BLOCK_RE.finditer(trajectory_text):
        try:
            obj = json.loads(m.group(1).strip())
            name = obj.get("name", "")
            if name:
                names.append(name)
        except (json.JSONDecodeError, AttributeError):
            pass
    return names


def extract_patch(trajectory_text: str) -> str:
    """Extract the final unified diff patch from a trajectory."""
    patches = PATCH_RE.findall(trajectory_text)
    return patches[-1] if patches else ""


def extract_context(prompt: str) -> str:
    """Pull problem statement for SPRM context."""
    return prompt[:300]


# ── Reward Function ────────────────────────────────────────────────────────────

def build_reward_fn_opus(
    sprm: SPRMScorer,
    judge: "OpusJudge",
    alpha: float,
    issue_map: dict[str, str],
    agent_wrapper: Optional["AgentRewardWrapper"] = None,
) -> callable:
    """Reward function using Opus 4.8 judge instead of Docker.

    When agent_wrapper is provided (recommended), runs a fresh agent rollout
    per completion to get real tool-call trajectories that SPRM can score.
    Without agent_wrapper, falls back to parsing TRL's raw completions
    (which have no tool calls → waste=1.0 → zero variance → no learning).

    Reward: r = opus_judge(patch, issue) − α × mean(SPRM_waste)
    """

    def reward_fn(
        prompts: list,
        completions: list,
        instance_id: list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        rewards = []
        for i, (prompt, traj) in enumerate(zip(prompts, completions)):
            iid = instance_id[i] if instance_id else f"task_{i}"

            if agent_wrapper is not None:
                # Run proper tool-calling agent to get real trajectory
                trajectory = agent_wrapper.rollout(iid)
                tool_names = [s.tool_name for s in trajectory.steps]
                patch = trajectory.patch
                context = iid
            else:
                # Fallback: parse TRL completion (no tool calls → all waste)
                if isinstance(traj, list):
                    traj_text = " ".join(
                        m.get("content", "") if isinstance(m, dict) else str(m)
                        for m in traj
                    )
                else:
                    traj_text = str(traj)
                context = extract_context(prompt if isinstance(prompt, str) else str(prompt))
                tool_names = parse_tool_calls(traj_text) or ["edit_file"]
                patch = extract_patch(traj_text)

            mean_waste = sprm.mean_waste(tool_names, context=context)
            issue = issue_map.get(iid, iid)
            task_success = judge.judge(issue=issue, patch=patch)

            r = task_success - alpha * mean_waste
            rewards.append(r)

            log.info(
                f"[{iid}] tools={len(tool_names)} waste={mean_waste:.3f} "
                f"patch={'yes' if patch else 'no'} success={task_success} r={r:.3f}"
            )

        return rewards

    return reward_fn


def build_reward_fn(
    sprm: SPRMScorer,
    executor: SWEBenchExecutor,
    alpha: float,
    agent_wrapper: Optional["AgentRewardWrapper"] = None,
) -> callable:
    """Reward function using Docker execution for task success.

    When agent_wrapper is provided, runs a proper tool-calling agent rollout
    to get trajectories that SPRM can score. The patch from the agent is then
    evaluated via Docker.

    Reward: r = docker_success(patch) − α × mean(SPRM_waste)
    """

    def reward_fn(
        prompts: list,
        completions: list,
        instance_id: list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        rewards = []
        for i, (prompt, traj) in enumerate(zip(prompts, completions)):
            iid = instance_id[i] if instance_id else f"task_{i}"

            if agent_wrapper is not None:
                trajectory = agent_wrapper.rollout(iid)
                tool_names = [s.tool_name for s in trajectory.steps]
                patch = trajectory.patch
            else:
                if isinstance(traj, list):
                    traj_text = " ".join(
                        m.get("content", "") if isinstance(m, dict) else str(m)
                        for m in traj
                    )
                else:
                    traj_text = str(traj)
                context = extract_context(prompt if isinstance(prompt, str) else str(prompt))
                tool_names = parse_tool_calls(traj_text)
                patch = extract_patch(traj_text)

            mean_waste = sprm.mean_waste(
                tool_names,
                context=extract_context(prompt if isinstance(prompt, str) else str(prompt)),
            )

            if patch:
                run_id = f"grpo_{int(time.time())}_{i}"
                task_success = float(executor.run(iid, patch, run_id=run_id))
            else:
                task_success = 0.0
                log.debug(f"No patch found for {iid} — task_success=0")

            r = task_success - alpha * mean_waste
            rewards.append(r)

            log.info(
                f"[{iid}] tools={len(tool_names)} waste={mean_waste:.3f} "
                f"patch={'yes' if patch else 'no'} success={task_success} r={r:.3f}"
            )

        return rewards

    return reward_fn


# ── Dataset Preparation ───────────────────────────────────────────────────────

def build_swebench_dataset(
    num_tasks: int = 50,
    seed: int = 42,
) -> tuple[Dataset, dict[str, dict]]:
    """Load SWE-bench Verified, subsample, return (dataset, instance_map).

    instance_map: {instance_id → raw SWE-bench record} needed by AgentRewardWrapper.
    The dataset prompt is minimal (just the issue) — the agent reads files itself.
    """
    from datasets import load_dataset as hf_load
    import random

    random.seed(seed)
    swebench = hf_load(SWEBENCH_SPLIT, split="test")
    instances = list(swebench)
    random.shuffle(instances)
    instances = instances[:num_tasks]

    rows = []
    instance_map: dict[str, dict] = {}

    for inst in instances:
        iid = inst["instance_id"]
        problem = inst["problem_statement"].replace("\r\n", "\n").replace("\r", "\n")
        instance_map[iid] = dict(inst)  # full record for AgentRewardWrapper

        # Minimal prompt — agent explores repo itself using bash/str_replace tools
        prompt = [
            {"role": "user", "content": (
                f"Repository: {inst['repo']}\n\n"
                f"Issue:\n{problem[:3000]}\n\n"
                f"Fix this issue by exploring the repository and making the necessary edits."
            )},
        ]
        rows.append({
            "prompt": prompt,
            "instance_id": iid,
            "problem_statement": problem,
        })

    log.info(f"Built dataset: {len(rows)} SWE-bench tasks")
    return Dataset.from_list(rows), instance_map


# ── GRPO++ Config ─────────────────────────────────────────────────────────────

def build_grpo_config(args) -> "GRPOConfig":
    from trl import GRPOConfig

    num_tasks = getattr(args, "num_tasks", 50)
    # max_steps must be explicit — TRL can't infer length from IterableDataset
    max_steps = max(1, (num_tasks * args.epochs) // 8)

    return GRPOConfig(
        output_dir=args.output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        beta=0.0,                              # no KL penalty (GRPO++)
        use_vllm=args.use_vllm,
        vllm_mode="colocate" if args.use_vllm else None,
        vllm_gpu_memory_utilization=0.35,      # reduced to leave room for 4-bit model
        log_completions=True,
        logging_steps=1,
        save_steps=10,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="wandb" if args.use_wandb else "none",
    )


# ── Custom Trainer (GRPO++ patches) ───────────────────────────────────────────

def make_grpo_plus_plus_trainer(model_name, config, reward_fn, dataset, tokenizer):
    """Build GRPOTrainer with Dr. GRPO and compact filtering patches."""
    from trl import GRPOTrainer

    class GRPOPlusPlusTrainer(GRPOTrainer):
        """GRPO++ = GRPOTrainer + no reward std normalization + compact filtering."""

        def _compute_advantages(self, rewards, groups):
            # Dr. GRPO: skip std normalization to avoid length/variance bias
            # rewards: list of floats per group
            advantages = []
            for group_rewards in groups:
                if len(set(group_rewards)) <= 1:
                    # Compact filtering: all identical rewards → skip this group
                    advantages.append([0.0] * len(group_rewards))
                    continue
                mean_r = sum(group_rewards) / len(group_rewards)
                # No division by std (Dr. GRPO)
                advantages.append([r - mean_r for r in group_rewards])
            return advantages

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    log.info(f"Loading {model_name} in 4-bit + LoRA (QLoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainer = GRPOPlusPlusTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    return trainer


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GRPO++ + SPRM training on SWE-bench")
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--sprm-checkpoint",
                   default="../oracle_trajectory/sprm_model",
                   help="Path to trained SPRM DeBERTa checkpoint")
    p.add_argument("--num-tasks", type=int, default=50,
                   help="Number of SWE-bench tasks for PoC (full=500)")
    p.add_argument("--num-generations", type=int, default=4,
                   help="G in GRPO (rollouts per task). Use 4 for PoC, 8 for full")
    p.add_argument("--alpha", type=float, default=ALPHA_DEFAULT,
                   help="SPRM penalty weight in reward = task_success - α × waste")
    p.add_argument("--alpha-sweep", nargs="+", type=float, default=None,
                   help="Run α ablation sweep, e.g. --alpha-sweep 0.1 0.3 0.5")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-completion-length", type=int, default=2048)
    p.add_argument("--docker-timeout", type=int, default=300,
                   help="Seconds per Docker eval (default 300s = 5 min)")
    p.add_argument("--output-dir", default="./grpo_swe_7b_poc")
    p.add_argument("--use-vllm", action="store_true",
                   help="Use vLLM for fast generation (recommended on A100)")
    p.add_argument("--no-docker", action="store_true",
                   help="Skip Docker eval (use task_success=0 for pipeline testing)")
    p.add_argument("--use-opus-judge", action="store_true",
                   help="Use Claude Opus 4.8 as reward judge instead of Docker "
                        "(requires ANTHROPIC_API_KEY; ~$0.01/call, much faster than Docker)")
    p.add_argument("--log-dir", default="./grpo_logs",
                   help="Directory for per-instance Docker eval logs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-wandb", action="store_true",
                   help="Enable W&B logging (set WANDB_API_KEY env var)")
    p.add_argument("--wandb-project", default="sprm-grpo",
                   help="W&B project name")
    p.add_argument("--wandb-run-name", default=None,
                   help="W&B run name (default: auto)")
    return p.parse_args()


def main():
    args = parse_args()

    # W&B setup
    if args.use_wandb:
        import os
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_run_name:
            os.environ["WANDB_RUN_NAME"] = args.wandb_run_name
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                   config=vars(args))
        log.info(f"W&B logging enabled — project: {args.wandb_project}")

    # Resolve SPRM checkpoint path
    sprm_path = Path(args.sprm_checkpoint)
    if not sprm_path.is_absolute():
        sprm_path = Path(__file__).parent / sprm_path
    sprm_path = sprm_path.resolve()
    if not sprm_path.exists():
        raise FileNotFoundError(
            f"SPRM checkpoint not found: {sprm_path}\n"
            f"Run Phase 2 training first: experiments/oracle_trajectory/train_sprm.py"
        )

    # Load SPRM
    log.info(f"Loading SPRM from {sprm_path}")
    sprm = SPRMScorer(str(sprm_path))

    # Set up task success evaluator
    judge = None
    executor = None

    if getattr(args, "use_opus_judge", False):
        log.info("Using Claude Opus 4.8 as reward judge (no Docker needed)")
        judge = OpusJudge()
    elif args.no_docker:
        log.warning("--no-docker: task_success will always be 0 (pipeline test mode)")
    else:
        log.info("Initializing SWE-bench Docker executor")
        executor = SWEBenchExecutor(timeout=args.docker_timeout, log_dir=args.log_dir)

    # Build dataset + instance_map (agent needs full records to clone repos)
    dataset, instance_map = build_swebench_dataset(
        num_tasks=args.num_tasks,
        seed=args.seed,
    )

    # Build issue_map for Opus judge (instance_id → problem_statement)
    issue_map: dict[str, str] = {
        iid: rec.get("problem_statement", iid)
        for iid, rec in instance_map.items()
    }

    # Alpha sweep or single run
    alphas = args.alpha_sweep if args.alpha_sweep else [args.alpha]

    for alpha in alphas:
        run_output = args.output_dir
        if len(alphas) > 1:
            run_output = f"{args.output_dir}_alpha{alpha}"
        log.info(f"=== Starting GRPO++ run: α={alpha}, output={run_output} ===")

        # Load tokenizer (needed for agent rollouts + trainer)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        # Build GRPO config
        run_args = argparse.Namespace(**vars(args))
        run_args.output_dir = run_output
        config = build_grpo_config(run_args)

        # Build trainer (loads model in 4-bit QLoRA)
        trainer = make_grpo_plus_plus_trainer(
            model_name=args.model,
            config=config,
            reward_fn=lambda p, c, **kw: [0.0] * len(p),  # placeholder, replaced below
            dataset=dataset,
            tokenizer=tokenizer,
        )

        # Build AgentRewardWrapper now that model is loaded inside trainer
        agent_wrapper = AgentRewardWrapper(
            model=trainer.model,
            tokenizer=tokenizer,
            instance_map=instance_map,
            max_steps=15,
            max_new_tokens=args.max_completion_length,
        )

        # Build real reward function with agent wrapper
        if judge is not None:
            reward_fn = build_reward_fn_opus(
                sprm, judge, alpha=alpha, issue_map=issue_map,
                agent_wrapper=agent_wrapper,
            )
        elif executor is None:
            # --no-docker mode: SPRM-only penalty, no task success signal
            def reward_fn(prompts, completions, instance_id=None, **kw):
                rewards = []
                for i, (prompt, _) in enumerate(zip(prompts, completions)):
                    iid = instance_id[i] if instance_id else f"task_{i}"
                    traj = agent_wrapper.rollout(iid)
                    tool_names = [s.tool_name for s in traj.steps] or ["edit_file"]
                    waste = sprm.mean_waste(tool_names, context=iid)
                    r = 0.0 - alpha * waste
                    rewards.append(r)
                    log.info(f"[{iid}] tools={len(tool_names)} waste={waste:.3f} r={r:.3f}")
                return rewards
        else:
            reward_fn = build_reward_fn(
                sprm, executor, alpha=alpha, agent_wrapper=agent_wrapper,
            )

        # Inject real reward function into trainer
        trainer.reward_funcs = [reward_fn]

        log.info(f"Starting training — {args.num_tasks} tasks, G={args.num_generations}, α={alpha}")
        trainer.train()
        trainer.save_model(run_output)
        log.info(f"Model saved to {run_output}")

    if judge is not None:
        s = judge.summary()
        log.info(
            f"Opus judge summary: {s['judge_calls']} calls, "
            f"total cost ${s['total_judge_cost_usd']:.4f}"
        )

    log.info("Phase 4a GRPO++ training complete.")


if __name__ == "__main__":
    main()
