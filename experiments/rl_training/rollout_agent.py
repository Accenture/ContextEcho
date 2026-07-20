"""Minimal tool-calling agent loop for SWE-bench rollouts.

Qwen2.5-Coder-7B runs as a proper agent with bash + str_replace tools.
Each rollout produces a multi-turn trajectory with tool calls that SPRM can score.

Architecture (inspired by mini-SWE-agent):
- Tool: bash  → grep, find, cat, python -m pytest (read-only, no internet)
- Tool: str_replace_editor → view (read file) + str_replace (patch file)
- Tool: submit → emit final diff and end episode

The agent operates on a TEMPORARY COPY of the repo (checked out at base_commit
from the tarball cached by SWE-bench). No GitHub API needed for file access —
the harness already has the repo locally when building Docker images.

For GRPO rollouts (before Docker eval), we use a lightweight sandbox:
- Clone the repo at base_commit into a temp dir using git
- Run bash commands inside that temp dir (no Docker needed for reads)
- str_replace edits files in the temp dir
- submit diffs the temp dir against base_commit → unified diff → send to Docker

Usage:
    agent = RolloutAgent(model, tokenizer, instance, max_steps=15)
    trajectory = agent.run()  # returns Trajectory namedtuple
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Tool schemas (injected into Qwen system prompt) ───────────────────────────

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a bash command in the repository root. "
                "Use for: grep -r, find, cat, head, python -m pytest. "
                "Do NOT modify files with bash — use str_replace_editor for edits. "
                "Output is truncated at 4000 chars."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Bash command to run"}
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": (
                "View or edit a file. "
                "To view: {\"command\": \"view\", \"path\": \"src/foo.py\"}. "
                "To edit: {\"command\": \"str_replace\", \"path\": \"src/foo.py\", "
                "\"old_str\": \"exact text to replace\", \"new_str\": \"replacement text\"}. "
                "old_str must match exactly (including whitespace)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "enum": ["view", "str_replace"]},
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                },
                "required": ["command", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "Submit your fix when you are done. "
                "Call this after applying all edits. "
                "No arguments needed — it will automatically diff your changes."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

TOOLS_JSON = "\n".join(json.dumps(t) for t in TOOLS_SCHEMA)

SYSTEM_PROMPT = f"""You are an expert software engineer fixing a GitHub issue.

You have access to the repository and can read files, search for code, and make edits.
Work methodically: read the relevant code first, understand the bug, then fix it.

# Tools

<tools>
{TOOLS_JSON}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags.

When you have fixed the issue, call submit() to finish.
"""

# ── Tool call parsing ─────────────────────────────────────────────────────────

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
# Qwen sometimes uses ```json code blocks instead of <tool_call> XML
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _normalize_tool_call(obj: dict) -> Optional[dict]:
    """Normalize a parsed JSON object into {name, arguments} form.

    Qwen may output "name" or "function" as the key for the tool name.
    """
    name = obj.get("name") or obj.get("function") or obj.get("tool")
    args = obj.get("arguments") or obj.get("parameters") or obj.get("input") or {}
    if not name:
        return None
    return {"name": name, "arguments": args}


def parse_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from model output.

    Handles multiple formats Qwen may use:
    1. <tool_call>{"name": ..., "arguments": ...}</tool_call>
    2. ```json {"name": ...} ``` or ```json {"function": ...} ```
    """
    calls = []
    # Try XML format first
    for m in TOOL_CALL_RE.finditer(text):
        try:
            obj = _normalize_tool_call(json.loads(m.group(1).strip()))
            if obj:
                calls.append(obj)
        except json.JSONDecodeError:
            pass
    if calls:
        return calls
    # Fallback: JSON code blocks
    for m in JSON_BLOCK_RE.finditer(text):
        try:
            obj = _normalize_tool_call(json.loads(m.group(1).strip()))
            if obj:
                calls.append(obj)
        except json.JSONDecodeError:
            pass
    return calls


# ── Trajectory dataclass ──────────────────────────────────────────────────────

@dataclass
class ToolStep:
    tool_name: str
    arguments: dict
    result: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Trajectory:
    instance_id: str
    messages: list[dict]          # full conversation for replay/logging
    steps: list[ToolStep]         # structured tool call sequence for SPRM
    patch: str                    # final unified diff (empty if submit not called)
    submitted: bool               # whether the agent called submit()
    truncated: bool               # whether max_steps was hit
    elapsed_s: float


# ── Sandbox: GitHub API-based virtual filesystem (no git clone) ───────────────

class RepoSandbox:
    """Virtual repo sandbox using GitHub raw content API.

    Many SWE-bench base_commits are not reachable from any public git ref
    (they're on release branches). raw.githubusercontent.com works for ANY
    commit SHA, so we fetch files on-demand and cache them locally.
    Edits apply to the local cache; diff() computes unified diff vs original.
    """

    GH_RAW = "https://raw.githubusercontent.com"
    GH_API = "https://api.github.com"

    def __init__(self, repo: str, base_commit: str, workdir: Optional[str] = None,
                 github_token: Optional[str] = None):
        self.repo = repo
        self.base_commit = base_commit
        self._tmpdir = tempfile.mkdtemp(prefix="sprm_sandbox_") if workdir is None else workdir
        self.root = Path(self._tmpdir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._token = github_token or ""
        self._original: dict[str, str] = {}   # path → original fetched content
        self._tree: Optional[list[str]] = None
        log.debug(f"Sandbox ready for {repo}@{base_commit[:8]}")

    def _gh_raw(self, path: str) -> Optional[str]:
        import urllib.request
        url = f"{self.GH_RAW}/{self.repo}/{self.base_commit}/{path}"
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "sprm-agent")
        if self._token:
            req.add_header("Authorization", f"token {self._token}")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception:
            return None

    def _gh_tree(self) -> list[str]:
        import urllib.request, json
        if self._tree is not None:
            return self._tree
        url = f"{self.GH_API}/repos/{self.repo}/git/trees/{self.base_commit}?recursive=1"
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "sprm-agent")
        req.add_header("Accept", "application/vnd.github.v3+json")
        if self._token:
            req.add_header("Authorization", f"token {self._token}")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                self._tree = [
                    item["path"] for item in data.get("tree", [])
                    if item.get("type") == "blob"
                ]
        except Exception:
            self._tree = []
        return self._tree

    def _ensure_local(self, path: str) -> Optional[Path]:
        """Fetch file from GitHub if not already local."""
        local = self.root / path
        if local.exists():
            return local
        content = self._gh_raw(path)
        if content is None:
            return None
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(content, encoding="utf-8")
        self._original[path] = content
        return local

    def bash(self, cmd: str, timeout: int = 30) -> str:
        """Run bash in the partially-populated repo root.
        Pre-fetches any .py files referenced in the command.
        Falls back to GitHub tree listing for find/ls with empty results.
        """
        py_refs = re.findall(r"[\w/.-]+\.py", cmd)
        for ref in py_refs[:3]:
            self._ensure_local(ref.lstrip("/"))
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=str(self.root),
                capture_output=True, text=True, timeout=timeout,
            )
            out = result.stdout + result.stderr
            if not out.strip() and any(k in cmd for k in ("find", "ls", "grep -r")):
                tree = self._gh_tree()
                py_files = [p for p in tree if p.endswith(".py")][:30]
                out = "\n".join(py_files) + "\n[from GitHub tree — use str_replace_editor view to read files]"
            if len(out) > 4000:
                out = out[:4000] + "\n[output truncated]"
            return out or "(no output)"
        except subprocess.TimeoutExpired:
            return f"[bash timeout after {timeout}s]"
        except Exception as e:
            return f"[bash error: {e}]"

    def view(self, path: str) -> str:
        """Read a file — fetches from GitHub if not local."""
        local = self._ensure_local(path)
        if local is None:
            tree = self._gh_tree()
            similar = [p for p in tree if path.split("/")[-1] in p][:5]
            hint = f"\nSimilar files: {similar}" if similar else ""
            return f"[file not found: {path}]{hint}"
        try:
            content = local.read_text(errors="replace")
            if len(content) > 8000:
                content = content[:8000] + "\n[truncated]"
            return content
        except Exception as e:
            return f"[read error: {e}]"

    def str_replace(self, path: str, old_str: str, new_str: str) -> str:
        """Edit a file in the local cache."""
        local = self._ensure_local(path)
        if local is None:
            return f"[file not found: {path}]"
        try:
            content = local.read_text(errors="replace")
            if old_str not in content:
                lines = content.splitlines()
                snippet = "\n".join(lines[:30])
                return f"[str_replace failed: old_str not found in {path}]\nFile starts with:\n{snippet}"
            count = content.count(old_str)
            if count > 1:
                return f"[str_replace failed: old_str appears {count} times in {path}. Make it more specific.]"
            new_content = content.replace(old_str, new_str, 1)
            local.write_text(new_content, encoding="utf-8")
            return f"[edit applied to {path}]"
        except Exception as e:
            return f"[str_replace error: {e}]"

    def diff(self) -> str:
        """Unified diff of all edited files vs original fetched content."""
        import difflib
        diffs = []
        for path, original in self._original.items():
            local = self.root / path
            if not local.exists():
                continue
            current = local.read_text(errors="replace")
            if current == original:
                continue
            diff_lines = list(difflib.unified_diff(
                original.splitlines(keepends=True),
                current.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            ))
            diffs.extend(diff_lines)
        return "".join(diffs)

    def cleanup(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)


# ── Main agent loop ───────────────────────────────────────────────────────────

class RolloutAgent:
    """Runs Qwen as a tool-calling agent on a SWE-bench instance."""

    def __init__(
        self,
        model,
        tokenizer,
        instance: dict,
        max_steps: int = 15,
        max_new_tokens: int = 1024,
        sandbox: Optional[RepoSandbox] = None,
        github_token: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.instance = instance
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self._github_token = github_token
        self._owns_sandbox = sandbox is None
        self.sandbox = sandbox

    def _generate(self, messages: list[dict]) -> str:
        """Run one generation step."""
        import torch
        from contextlib import contextmanager

        @contextmanager
        def _eval_mode(m):
            """Switch model to eval mode for inference, restore after."""
            training = m.training
            m.eval()
            try:
                yield m
            finally:
                if training:
                    m.train()

        # Do NOT pass tools= here — it overrides our system prompt's format
        # instructions and causes the model to use JSON code blocks instead
        # of <tool_call> XML. We handle tool format via SYSTEM_PROMPT directly.
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with _eval_mode(self.model), torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=1.0,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

    def _dispatch(self, name: str, args: dict) -> str:
        """Execute a tool call and return result string."""
        sb = self.sandbox
        assert sb is not None, "sandbox must be set before dispatching"
        if name == "bash":
            return sb.bash(args.get("cmd", ""))
        elif name == "str_replace_editor":
            cmd = args.get("command", "view")
            path = args.get("path", "")
            if cmd == "view":
                return sb.view(path)
            elif cmd == "str_replace":
                return sb.str_replace(
                    path,
                    args.get("old_str", ""),
                    args.get("new_str", ""),
                )
            return f"[unknown str_replace_editor command: {cmd}]"
        elif name == "submit":
            return "[submit acknowledged]"
        return f"[unknown tool: {name}]"

    def run(self) -> Trajectory:
        """Run the agent loop. Returns a Trajectory."""
        t0 = time.time()
        iid = self.instance["instance_id"]
        problem = self.instance["problem_statement"].replace("\r\n", "\n").replace("\r", "\n")
        repo = self.instance["repo"]
        commit = self.instance["base_commit"]

        # Set up sandbox (API-based, no git clone needed)
        if self.sandbox is None:
            self.sandbox = RepoSandbox(
                repo, commit, github_token=self._github_token
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Repository: {repo}\n\n"
                f"Issue:\n{problem[:3000]}\n\n"
                f"Please fix this issue. Start by exploring the repository structure."
            )},
        ]

        steps: list[ToolStep] = []
        submitted = False
        truncated = False
        patch = ""

        for step_num in range(self.max_steps):
            # Generate
            try:
                response = self._generate(messages)
            except Exception as e:
                log.warning(f"[{iid}] Generation error at step {step_num}: {e}")
                break

            messages.append({"role": "assistant", "content": response})

            # Parse tool calls
            calls = parse_tool_calls(response)
            if not calls:
                # Model gave final answer with no tool call — done
                log.debug(f"[{iid}] Step {step_num}: no tool call, ending")
                break

            # Execute first tool call (one per turn)
            call = calls[0]
            name = call.get("name", "unknown")
            args = call.get("arguments", {})

            result = self._dispatch(name, args)
            steps.append(ToolStep(tool_name=name, arguments=args, result=result))

            log.debug(f"[{iid}] Step {step_num}: {name}({list(args.keys())}) → {result[:80]!r}")

            if name == "submit":
                submitted = True
                patch = self.sandbox.diff()
                break

            # Feed result back
            messages.append({
                "role": "user",
                "content": f"<tool_response>\n{result}\n</tool_response>",
            })
        else:
            truncated = True
            # Even if max_steps hit, grab whatever diff exists
            patch = self.sandbox.diff()

        # If submitted but no diff yet (shouldn't happen), get it
        if submitted and not patch:
            patch = self.sandbox.diff()

        if self._owns_sandbox and self.sandbox is not None:
            self.sandbox.cleanup()
            self.sandbox = None

        return Trajectory(
            instance_id=iid,
            messages=messages,
            steps=steps,
            patch=patch,
            submitted=submitted,
            truncated=truncated,
            elapsed_s=round(time.time() - t0, 1),
        )


# ── Batch rollout (for GRPO: G rollouts per task) ────────────────────────────

def run_rollouts(
    model,
    tokenizer,
    instance: dict,
    num_rollouts: int = 4,
    max_steps: int = 15,
    max_new_tokens: int = 1024,
    github_token: Optional[str] = None,
) -> list[Trajectory]:
    """Run G rollouts for one SWE-bench instance.

    Each rollout gets its own fresh sandbox (cheap with API-based sandbox —
    no git clone, just a temp dir that fetches files on demand).
    """
    iid = instance["instance_id"]
    trajectories = []
    for i in range(num_rollouts):
        agent = RolloutAgent(
            model=model, tokenizer=tokenizer, instance=instance,
            max_steps=max_steps, max_new_tokens=max_new_tokens,
            github_token=github_token,
        )
        traj = agent.run()
        trajectories.append(traj)
        log.info(
            f"[{iid}] rollout {i+1}/{num_rollouts}: "
            f"steps={len(traj.steps)} submitted={traj.submitted} "
            f"patch_chars={len(traj.patch)} ({traj.elapsed_s}s)"
        )
    return trajectories
