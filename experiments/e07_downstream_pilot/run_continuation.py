"""Coding-session continuation downstream runner.

Per DRAFT PREREG_AMENDMENT_DOWNSTREAM_CODING.md. For each of n=25 cut points
in the donated transcript, runs scratch + recent3K arms against Sonnet 4.6
and records the model's proposed next tool call.

Output layout (PII-sensitive paths/args; see .gitignore):
  data_archive/downstream_coding/claude-sonnet-4-6/cutpoint-<i>/
    inputs.json              (PII: scratch+recent3K prompt material — gitignored)
    scratch_response.json    (PII: model output may echo paths — gitignored)
    recent3K_response.json   (PII: same — gitignored)
    metrics.json             (REDACTED: tool name match, similarity score — committed)

Usage:
  set -a && source ../.env && set +a && \\
    python -u scripts/run_downstream_continuation.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.clients import TargetClient  # type: ignore
from harness.cost import CostTracker  # type: ignore  # noqa: F401  # re-exported for callers

CUTPOINTS_PATH = REPO_ROOT / "data_archive" / "downstream_coding" / "CUTPOINTS.json"
TRANSCRIPT_PATH = REPO_ROOT / "data" / "session_raw_transcript.jsonl"
OUT_BASE = REPO_ROOT / "data_archive" / "downstream_coding"
TARGET_SAFE = "claude-sonnet-4-6"
MODEL_ID = "claude-sonnet-4-6"  # native Anthropic SDK uses bare model name

ACK_MESSAGE = "Acknowledged. How can I help continue this work?"

SYSTEM_PROMPT = (
    "You are an AI coding assistant operating like Claude Code: a tool-using "
    "agent that helps with software-engineering tasks via shell, file, and web "
    "tools. Given the conversation so far, propose the SINGLE next action you "
    "would take. Respond with exactly one tool call. Do not respond with text "
    "alone — emit a tool_use."
)

# Locked tool schema = the exact tool set Claude Code exposes. Subset that
# covers >99% of the donated transcript's tool_uses.
TOOLS = [
    {
        "name": "Bash",
        "description": "Run a shell command in a persistent zsh session.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "description": {"type": "string"},
                "run_in_background": {"type": "boolean"},
                "timeout": {"type": "number"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "Read",
        "description": "Read a file from the local filesystem.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "Edit",
        "description": "Performs exact string replacements in files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "Write",
        "description": "Writes a file to the local filesystem.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "WebFetch",
        "description": "Fetches a URL and extracts content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "prompt": {"type": "string"},
            },
            "required": ["url", "prompt"],
        },
    },
    {
        "name": "TaskUpdate",
        "description": "Update task list with progress.",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "status": {"type": "string"},
            },
            "required": ["id", "status"],
        },
    },
    {
        "name": "Grep",
        "description": "Search file contents with regex.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "Glob",
        "description": "Find files matching glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
            },
            "required": ["pattern"],
        },
    },
]


def load_transcript_indexed() -> list[dict]:
    """Load transcript line-by-line, indexed."""
    rows = []
    with TRANSCRIPT_PATH.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append(None)
    return rows


def extract_recent3K(rows: list[dict], cut_idx: int, target_chars: int = 3000) -> str:
    """Build a recent3K text block ending at cut_idx by concatenating
    backwards through messages (user + assistant) until ~target_chars."""
    parts = []
    total = 0
    i = cut_idx - 1
    while i >= 0 and total < target_chars * 2:  # walk back, take more than 3K then trim
        d = rows[i]
        i -= 1
        if not d:
            continue
        t = d.get("type")
        if t not in ("user", "assistant"):
            continue
        msg = d.get("message", {}) or {}
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            tparts = []
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "text":
                        tparts.append(c.get("text", ""))
                    elif c.get("type") == "tool_use":
                        tparts.append(f"[tool_use {c.get('name')}: {json.dumps(c.get('input', {}))[:200]}]")
                    elif c.get("type") == "tool_result":
                        tr = c.get("content", "")
                        if isinstance(tr, list):
                            tr = "".join(x.get("text", "") for x in tr if isinstance(x, dict))
                        tparts.append(f"[tool_result: {str(tr)[:200]}]")
            text = "\n".join(tparts)
        else:
            text = ""
        if not text.strip():
            continue
        role = "USER" if t == "user" else "ASSISTANT"
        parts.append(f"--- {role} ---\n{text}")
        total += len(text)
    parts.reverse()
    full = "\n\n".join(parts)
    if len(full) > target_chars:
        full = full[-target_chars:]
    return full


def get_immediate_context_at(rows: list[dict], cut_idx: int) -> str:
    """Build the 'immediate context' framing for the scratch arm.

    In a long Claude Code session, there's rarely a clean 'user prompt'
    immediately before a tool_use — most preceding 'user' rows are
    tool_results from prior tool calls. The realistic scratch baseline
    is: 'the model has just received the most recent tool_result and
    is being asked to propose the next action.' We construct this by:

      1. Walking back from cut_idx
      2. Taking the most recent tool_result (treated as terminal observation)
      3. Taking the most recent assistant text (treated as agent's last reasoning)
      4. Taking the most recent text-bearing user msg if present in last 200 rows

    This gives the model immediate situational awareness without the full
    3K-char history that the recent3K arm carries.
    """
    last_tool_result = None
    last_assistant_text = None
    last_user_text = None

    for offset in range(1, 200):
        r_idx = cut_idx - offset
        if r_idx < 0:
            break
        d = rows[r_idx]
        if not d:
            continue
        t = d.get("type")
        msg = d.get("message", {}) or {}
        content = msg.get("content")

        if t == "user" and isinstance(content, list):
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "tool_result" and last_tool_result is None:
                    tr = c.get("content", "")
                    if isinstance(tr, list):
                        tr = "".join(x.get("text", "") for x in tr if isinstance(x, dict))
                    last_tool_result = str(tr)[:1500]
                elif c.get("type") == "text" and last_user_text is None:
                    if c.get("text", "").strip():
                        last_user_text = c["text"][:1000]
        elif t == "user" and isinstance(content, str) and last_user_text is None:
            if content.strip():
                last_user_text = content[:1000]
        elif t == "assistant" and isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text" and last_assistant_text is None:
                    if c.get("text", "").strip():
                        last_assistant_text = c["text"][:1000]

        if last_tool_result and last_assistant_text:
            break

    parts = []
    if last_user_text:
        parts.append(f"[Most recent user instruction]\n{last_user_text}")
    if last_assistant_text:
        parts.append(f"[Your last reasoning]\n{last_assistant_text}")
    if last_tool_result:
        parts.append(f"[Most recent tool result]\n{last_tool_result}")
    parts.append("Propose your next single tool call.")
    return "\n\n".join(parts)


def jaccard_args(a: dict, b: dict) -> float:
    """Argument-JSON similarity as Jaccard over flattened key=value tokens."""
    def flatten(d, prefix=""):
        toks = set()
        for k, v in (d or {}).items():
            if isinstance(v, dict):
                toks |= flatten(v, f"{prefix}{k}.")
            elif isinstance(v, list):
                for i, x in enumerate(v):
                    toks.add(f"{prefix}{k}[{i}]={x}")
            else:
                # tokenize string values into 4-grams to give partial credit
                s = str(v)
                if len(s) > 50:
                    toks |= {f"{prefix}{k}~{s[i:i + 4]}" for i in range(0, len(s) - 3, 4)}
                else:
                    toks.add(f"{prefix}{k}={s}")
        return toks
    A = flatten(a)
    B = flatten(b)
    if not A and not B:
        return 1.0
    return len(A & B) / max(len(A | B), 1)


def run_one_cut(client: TargetClient, cut: dict, recent3K_text: str,
                user_msg: str, out_dir: Path) -> dict:
    """Run scratch + recent3K arms, score, save metrics."""
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        try:
            return json.loads(metrics_path.read_text())
        except Exception:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    gt_tool = cut["ground_truth_tool"]
    gt_args = cut["ground_truth_args"] or {}

    # Scratch arm
    scratch_messages = [
        {"role": "user", "content": [{"type": "text", "text": user_msg}]}
    ]
    t0 = time.perf_counter()
    scratch_resp = client.step(
        system_prompt=SYSTEM_PROMPT,
        messages=scratch_messages,
        tools=TOOLS,
        max_tokens=4096,
    )
    scratch_sec = time.perf_counter() - t0

    # Recent3K arm
    recent3K_messages = [
        {"role": "user", "content": [{"type": "text", "text": recent3K_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": ACK_MESSAGE}]},
        {"role": "user", "content": [{"type": "text", "text": user_msg}]},
    ]
    t0 = time.perf_counter()
    recent3K_resp = client.step(
        system_prompt=SYSTEM_PROMPT,
        messages=recent3K_messages,
        tools=TOOLS,
        max_tokens=4096,
    )
    recent3K_sec = time.perf_counter() - t0

    # Save full responses (PII; gitignored)
    (out_dir / "inputs.json").write_text(json.dumps({
        "cut_index": cut["cut_index"],
        "user_msg_excerpt": user_msg[:500],
        "recent3K_excerpt": recent3K_text[:500],
        "recent3K_len": len(recent3K_text),
    }, indent=2))
    (out_dir / "scratch_response.json").write_text(json.dumps({
        "text": scratch_resp.text,
        "tool_calls": [{"name": t.name, "input": t.input} for t in scratch_resp.tool_calls],
        "stop_reason": scratch_resp.stop_reason,
        "usage": scratch_resp.raw_usage,
        "wall_clock_sec": scratch_sec,
    }, indent=2, default=str))
    (out_dir / "recent3K_response.json").write_text(json.dumps({
        "text": recent3K_resp.text,
        "tool_calls": [{"name": t.name, "input": t.input} for t in recent3K_resp.tool_calls],
        "stop_reason": recent3K_resp.stop_reason,
        "usage": recent3K_resp.raw_usage,
        "wall_clock_sec": recent3K_sec,
    }, indent=2, default=str))

    # Score (REDACTED metrics — committed)
    scratch_tool = scratch_resp.tool_calls[0].name if scratch_resp.tool_calls else None
    recent3K_tool = recent3K_resp.tool_calls[0].name if recent3K_resp.tool_calls else None
    scratch_args = dict(scratch_resp.tool_calls[0].input) if scratch_resp.tool_calls else {}
    recent3K_args = dict(recent3K_resp.tool_calls[0].input) if recent3K_resp.tool_calls else {}

    metrics = {
        "cut_index": cut["cut_index"],
        "ground_truth_tool": gt_tool,
        "scratch_tool": scratch_tool,
        "recent3K_tool": recent3K_tool,
        "M1_scratch_match": (scratch_tool == gt_tool) if scratch_tool else False,
        "M1_recent3K_match": (recent3K_tool == gt_tool) if recent3K_tool else False,
        "M2_scratch_arg_sim": jaccard_args(scratch_args, gt_args) if scratch_tool == gt_tool else None,
        "M2_recent3K_arg_sim": jaccard_args(recent3K_args, gt_args) if recent3K_tool == gt_tool else None,
        "scratch_wall_sec": scratch_sec,
        "recent3K_wall_sec": recent3K_sec,
        "scratch_input_tokens": scratch_resp.raw_usage.get("input_tokens"),
        "scratch_output_tokens": scratch_resp.raw_usage.get("output_tokens"),
        "recent3K_input_tokens": recent3K_resp.raw_usage.get("input_tokens"),
        "recent3K_output_tokens": recent3K_resp.raw_usage.get("output_tokens"),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def main() -> int:
    if not CUTPOINTS_PATH.exists():
        sys.exit(f"Run scripts/select_cutpoints.py first: {CUTPOINTS_PATH} missing")
    if not TRANSCRIPT_PATH.exists():
        sys.exit(f"Transcript missing: {TRANSCRIPT_PATH}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY")

    cuts = json.loads(CUTPOINTS_PATH.read_text())["cutpoints"]
    print(f"Loaded {len(cuts)} cutpoints")

    print("Loading transcript...")
    rows = load_transcript_indexed()
    print(f"  {len(rows)} lines")

    cost_csv = OUT_BASE / TARGET_SAFE / "cost_log.csv"
    cost_csv.parent.mkdir(parents=True, exist_ok=True)
    cost = CostTracker(cost_csv)
    client = TargetClient("anthropic", MODEL_ID, cost_tracker=cost,
                          session_id="downstream_continuation")

    target_dir = OUT_BASE / TARGET_SAFE
    target_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    n_done = 0
    for i, cut in enumerate(cuts):
        out_dir = target_dir / f"cutpoint-{i:02d}"
        if (out_dir / "metrics.json").exists():
            print(f"  [skip {i}] cached")
            n_done += 1
            continue
        recent3K = extract_recent3K(rows, cut["cut_index"])
        user_msg = get_immediate_context_at(rows, cut["cut_index"])
        print(f"\n=== cut {i} (idx={cut['cut_index']}, gt={cut['ground_truth_tool']}) ===")
        print(f"  user_msg_len={len(user_msg)}  recent3K_len={len(recent3K)}")
        try:
            m = run_one_cut(client, cut, recent3K, user_msg, out_dir)
            print(f"  scratch tool={m['scratch_tool']} match={m['M1_scratch_match']}  "
                  f"recent3K tool={m['recent3K_tool']} match={m['M1_recent3K_match']}")
            n_done += 1
        except Exception as e:
            print(f"  [ERROR] {e}")

    elapsed = time.time() - started
    print(f"\n=== DONE: {n_done}/{len(cuts)} cells, {elapsed:.0f}s wall ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
