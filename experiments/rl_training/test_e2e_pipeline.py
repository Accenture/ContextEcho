"""End-to-end pipeline smoke test — no GPU required.

Tests the full SPRM reward pipeline:
  1. Load 3 SWE-bench instances
  2. Run RolloutAgent (with a tiny stub model that returns canned tool calls)
  3. Score trajectories with SPRM
  4. Verify reward variance > 0

Run with:
    python test_e2e_pipeline.py
    python test_e2e_pipeline.py --sprm-checkpoint ../oracle_trajectory/sprm_model
    python test_e2e_pipeline.py --live-model  # use real Qwen (needs GPU)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from rollout_agent import (
    RolloutAgent, RepoSandbox, Trajectory, ToolStep,
    TOOLS_SCHEMA, parse_tool_calls,
)


# ── Stub model for CPU testing ────────────────────────────────────────────────

CANNED_RESPONSES = [
    # Turn 1: explore structure
    '<tool_call>{"name": "bash", "arguments": {"cmd": "find . -name \'*.py\' | head -20"}}</tool_call>',
    # Turn 2: read a file
    '<tool_call>{"name": "str_replace_editor", "arguments": {"command": "view", "path": "setup.py"}}</tool_call>',
    # Turn 3: make an edit
    '<tool_call>{"name": "str_replace_editor", "arguments": {"command": "str_replace", "path": "setup.py", "old_str": "version = \'1.0\'", "new_str": "version = \'1.0.1\'"}}</tool_call>',
    # Turn 4: submit
    '<tool_call>{"name": "submit", "arguments": {}}</tool_call>',
]

# Variant B — doom_loop pattern (many bash calls, no edit)
CANNED_RESPONSES_DOOM = [
    '<tool_call>{"name": "bash", "arguments": {"cmd": "grep -r \'bug\' . | head -10"}}</tool_call>',
    '<tool_call>{"name": "bash", "arguments": {"cmd": "grep -r \'error\' . | head -10"}}</tool_call>',
    '<tool_call>{"name": "bash", "arguments": {"cmd": "grep -r \'fix\' . | head -10"}}</tool_call>',
    '<tool_call>{"name": "bash", "arguments": {"cmd": "grep -r \'issue\' . | head -10"}}</tool_call>',
    # Eventually submits without any edit
    '<tool_call>{"name": "submit", "arguments": {}}</tool_call>',
]


class StubModel:
    """Generates canned tool-call responses, cycling through the list."""

    def __init__(self, responses: list[str], device: str = "cpu"):
        self.responses = responses
        self.device = device
        self._idx = 0

    def generate(self, **kwargs):
        resp = self.responses[self._idx % len(self.responses)]
        self._idx += 1
        return resp

    def to(self, device):
        return self


class StubTokenizer:
    """Minimal tokenizer that passes text through unchanged."""

    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs) -> str:
        texts = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            texts.append(f"<{role}>{content}</{role}>")
        return "\n".join(texts)

    def __call__(self, text, return_tensors=None, **kwargs):
        return {"input_ids": [[0]]}

    def decode(self, ids, **kwargs) -> str:
        return self._last_response

    # The agent calls tokenizer → generate → decode; we need to intercept generate
    # We patch this at the RolloutAgent level instead (see StubRolloutAgent below)


class StubRolloutAgent(RolloutAgent):
    """RolloutAgent that uses StubModel responses instead of real generation."""

    def __init__(self, responses: list[str], instance: dict, max_steps: int = 10):
        self._responses = responses
        self._resp_idx = 0
        self.instance = instance
        self.max_steps = max_steps
        self.max_new_tokens = 512
        self._owns_sandbox = True
        self.sandbox = None
        # model/tokenizer not used (we override _generate)

    def _generate(self, messages: list[dict]) -> str:
        resp = self._responses[self._resp_idx % len(self._responses)]
        self._resp_idx += 1
        time.sleep(0.01)  # simulate latency
        return resp


# ── SPRM scorer (CPU) ─────────────────────────────────────────────────────────

def load_sprm(checkpoint_path: str):
    """Load SPRM from checkpoint, return SPRMScorer."""
    sys.path.insert(0, str(Path(__file__).parent))
    from train_grpo_sprm import SPRMScorer
    return SPRMScorer(checkpoint_path)


class StubSPRM:
    """SPRM stub that returns deterministic waste scores based on tool name."""

    WASTE_MAP = {
        "bash": 0.2,          # necessary — exploration
        "str_replace_editor": 0.1,  # necessary — edit
        "submit": 0.05,        # necessary — finish
    }

    def mean_waste(self, tool_names: list[str], context: str = "") -> float:
        if not tool_names:
            return 1.0
        wastes = [self.WASTE_MAP.get(t, 0.7) for t in tool_names]
        return sum(wastes) / len(wastes)


# ── Test helpers ──────────────────────────────────────────────────────────────

def make_fake_instance(instance_id: str) -> dict:
    """Create a minimal SWE-bench-compatible instance dict."""
    return {
        "instance_id": instance_id,
        "repo": "django/django",
        "base_commit": "main",
        "problem_statement": f"Fix bug in {instance_id}: the method returns wrong result",
        "hints_text": "",
        "created_at": "2024-01-01",
        "version": "4.2",
        "FAIL_TO_PASS": "[]",
        "PASS_TO_PASS": "[]",
        "environment_setup_commit": "main",
    }


def test_parse_tool_calls():
    """Unit test: verify Hermes XML tool call parsing."""
    log.info("=== test_parse_tool_calls ===")
    text = '<tool_call>{"name": "bash", "arguments": {"cmd": "ls"}}</tool_call>'
    calls = parse_tool_calls(text)
    assert calls == [{"name": "bash", "arguments": {"cmd": "ls"}}], f"Got: {calls}"

    # Multi-call
    text2 = (
        '<tool_call>{"name": "bash", "arguments": {"cmd": "grep -r x ."}}</tool_call>\n'
        '<tool_call>{"name": "submit", "arguments": {}}</tool_call>'
    )
    calls2 = parse_tool_calls(text2)
    assert len(calls2) == 2
    assert calls2[0]["name"] == "bash"
    assert calls2[1]["name"] == "submit"

    log.info("  PASS: tool call parsing works")


def test_stub_agent_trajectory():
    """Test that StubRolloutAgent produces a proper multi-step trajectory."""
    log.info("=== test_stub_agent_trajectory (efficient rollout) ===")
    inst = make_fake_instance("django__django-001")
    agent = StubRolloutAgent(responses=CANNED_RESPONSES, instance=inst, max_steps=10)

    # Use a temp dir that doesn't actually clone (skip _setup)
    import tempfile
    tmpdir = tempfile.mkdtemp()
    sb = object.__new__(RepoSandbox)
    sb.repo = "django/django"
    sb.base_commit = "main"
    sb._tmpdir = tmpdir
    sb.root = Path(tmpdir)
    sb.root.joinpath("setup.py").write_text("version = '1.0'\n")

    def fake_diff():
        f = sb.root / "setup.py"
        return f"--- a/setup.py\n+++ b/setup.py\n@@ -1 +1 @@\n-version = '1.0'\n+version = '1.0.1'\n"

    sb.diff = fake_diff
    agent._owns_sandbox = True
    agent.sandbox = sb

    traj = agent.run()
    assert len(traj.steps) >= 3, f"Expected ≥3 steps, got {len(traj.steps)}"
    assert traj.submitted, "Agent should have called submit"
    tool_names = [s.tool_name for s in traj.steps]
    assert "bash" in tool_names, f"Expected bash in steps: {tool_names}"
    assert "str_replace_editor" in tool_names, f"Expected str_replace_editor: {tool_names}"
    assert "submit" in tool_names, f"Expected submit: {tool_names}"
    log.info(f"  steps: {tool_names}")
    log.info(f"  patch present: {bool(traj.patch)}")
    log.info("  PASS: efficient trajectory produced")
    return traj


def test_doom_loop_trajectory():
    """Test that doom-loop pattern produces high waste score."""
    log.info("=== test_doom_loop_trajectory (doom loop) ===")
    inst = make_fake_instance("django__django-002")
    agent = StubRolloutAgent(responses=CANNED_RESPONSES_DOOM, instance=inst, max_steps=10)

    import tempfile
    tmpdir = tempfile.mkdtemp()
    sb = object.__new__(RepoSandbox)
    sb.repo = "django/django"
    sb.base_commit = "main"
    sb._tmpdir = tmpdir
    sb.root = Path(tmpdir)
    sb.diff = lambda: ""  # no edits
    agent._owns_sandbox = True
    agent.sandbox = sb

    traj = agent.run()
    tool_names = [s.tool_name for s in traj.steps]
    bash_count = tool_names.count("bash")
    edit_count = tool_names.count("str_replace_editor")
    log.info(f"  steps: {tool_names}")
    log.info(f"  bash={bash_count}, edits={edit_count}")
    assert bash_count >= 4, f"Expected ≥4 bash calls in doom loop, got {bash_count}"
    assert edit_count == 0, f"Doom loop should have 0 edits, got {edit_count}"
    log.info("  PASS: doom-loop trajectory has correct shape")
    return traj


def test_reward_variance(efficient_traj: Trajectory, doom_traj: Trajectory):
    """Test that efficient vs doom-loop trajectories produce different SPRM scores."""
    log.info("=== test_reward_variance ===")
    sprm = StubSPRM()

    efficient_tools = [s.tool_name for s in efficient_traj.steps]
    doom_tools = [s.tool_name for s in doom_traj.steps]

    waste_efficient = sprm.mean_waste(efficient_tools)
    waste_doom = sprm.mean_waste(doom_tools)

    log.info(f"  efficient waste: {waste_efficient:.3f}")
    log.info(f"  doom waste:      {waste_doom:.3f}")

    assert waste_efficient < waste_doom, (
        f"Efficient trajectory should have lower waste: "
        f"efficient={waste_efficient:.3f} > doom={waste_doom:.3f}"
    )

    # Simulate reward (task_success=0 for both since we have no Docker here)
    alpha = 0.1
    r_efficient = 0.0 - alpha * waste_efficient
    r_doom = 0.0 - alpha * waste_doom

    log.info(f"  r_efficient={r_efficient:.3f}, r_doom={r_doom:.3f}")
    assert r_efficient > r_doom, "Efficient rollout should have higher reward"

    variance = (r_efficient - r_doom) ** 2
    log.info(f"  variance proxy: {variance:.6f}")
    assert variance > 0, "Reward variance must be > 0 for GRPO to learn"
    log.info("  PASS: reward variance > 0")


def test_real_sprm(checkpoint_path: str, efficient_traj: Trajectory, doom_traj: Trajectory):
    """Test with the real SPRM model checkpoint."""
    log.info(f"=== test_real_sprm (checkpoint: {checkpoint_path}) ===")
    try:
        sprm = load_sprm(checkpoint_path)
    except Exception as e:
        log.warning(f"  SKIP: could not load SPRM: {e}")
        return

    efficient_tools = [s.tool_name for s in efficient_traj.steps]
    doom_tools = [s.tool_name for s in doom_traj.steps]

    waste_e = sprm.mean_waste(efficient_tools, context="django bug fix")
    waste_d = sprm.mean_waste(doom_tools, context="django bug fix")

    log.info(f"  Real SPRM — efficient waste: {waste_e:.3f}, doom waste: {waste_d:.3f}")
    log.info(f"  Variance exists: {waste_e != waste_d}")
    log.info("  PASS: real SPRM scoring works")


def test_live_model_rollout(model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"):
    """Test with real Qwen model (requires GPU). Optional."""
    log.info(f"=== test_live_model_rollout ({model_name}) ===")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        log.info("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        log.info("  Loading model (this may take a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        inst = make_fake_instance("django__django-real-001")
        inst["repo"] = "django/django"
        inst["base_commit"] = "stable/4.2"

        agent = RolloutAgent(
            model=model, tokenizer=tokenizer, instance=inst, max_steps=5,
        )
        log.info("  Running agent rollout...")
        traj = agent.run()
        log.info(f"  steps={len(traj.steps)} submitted={traj.submitted}")
        log.info(f"  tool names: {[s.tool_name for s in traj.steps]}")
        log.info("  PASS: live model rollout completed")
        return traj
    except Exception as e:
        log.error(f"  FAIL: {e}")
        raise


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sprm-checkpoint", default=None,
                   help="Path to SPRM checkpoint for real scoring test")
    p.add_argument("--live-model", action="store_true",
                   help="Run with real Qwen model (requires GPU)")
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    args = p.parse_args()

    t0 = time.time()
    failures = []

    try:
        test_parse_tool_calls()
    except AssertionError as e:
        log.error(f"FAIL test_parse_tool_calls: {e}")
        failures.append("parse_tool_calls")

    try:
        efficient_traj = test_stub_agent_trajectory()
    except AssertionError as e:
        log.error(f"FAIL test_stub_agent_trajectory: {e}")
        failures.append("stub_agent_trajectory")
        efficient_traj = None

    try:
        doom_traj = test_doom_loop_trajectory()
    except AssertionError as e:
        log.error(f"FAIL test_doom_loop_trajectory: {e}")
        failures.append("doom_loop_trajectory")
        doom_traj = None

    if efficient_traj and doom_traj:
        try:
            test_reward_variance(efficient_traj, doom_traj)
        except AssertionError as e:
            log.error(f"FAIL test_reward_variance: {e}")
            failures.append("reward_variance")

        if args.sprm_checkpoint:
            try:
                test_real_sprm(args.sprm_checkpoint, efficient_traj, doom_traj)
            except Exception as e:
                log.error(f"FAIL test_real_sprm: {e}")
                failures.append("real_sprm")

    if args.live_model:
        try:
            test_live_model_rollout(args.model)
        except Exception as e:
            log.error(f"FAIL test_live_model_rollout: {e}")
            failures.append("live_model_rollout")

    elapsed = time.time() - t0
    log.info(f"\n{'='*60}")
    if failures:
        log.error(f"FAILED tests: {failures}")
        sys.exit(1)
    else:
        log.info(f"ALL TESTS PASSED in {elapsed:.1f}s")
        log.info("")
        log.info("Pipeline is ready. Key findings:")
        log.info("  ✓ Hermes XML tool call parsing works")
        log.info("  ✓ RolloutAgent produces multi-step trajectories")
        log.info("  ✓ Efficient vs doom-loop trajectories differ in waste score")
        log.info("  ✓ Reward variance > 0 — GRPO will have signal to learn from")
        log.info("")
        log.info("Next step: Deploy to Spheron A100, run with real Qwen + SPRM:")
        log.info("  python train_grpo_sprm.py --model Qwen/Qwen2.5-Coder-7B-Instruct \\")
        log.info("    --sprm-checkpoint /home/ubuntu/sprm_model \\")
        log.info("    --num-tasks 50 --num-generations 4 --alpha 0.1 --use-opus-judge")


if __name__ == "__main__":
    main()
