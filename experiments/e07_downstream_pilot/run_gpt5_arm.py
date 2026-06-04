"""4th arm: GPT-5-derived recent3K control for the downstream cut-points.

Per user critique 2026-04-30: the filler3K control was useful (ruled out
length-only confound) but it doesn't isolate persona drift from session-
context utility. The proper drift control varies the FLAVOR of recent
context while holding LENGTH and CONTENT-TYPE (real coding session)
constant.

Design:
  - Claude3K arm = the original recent3K (Claude-flavored coding session)
  - GPT5_3K arm  = recent3K-shaped slice from the GPT-5 synthetic session
                   used in Q2/Gap1 family-specificity test, length-matched
                   to the same target_chars (3000) used in Claude3K

Both are real coding-session content; they differ only in donor flavor.

Decision rule:
  Claude3K vs GPT5_3K significant → real persona-drift effect on continuation
                                      direction tells us if Claude-flavor is
                                      uniquely helpful OR uniquely harmful
  not significant                  → drift signal at register level (Q2)
                                      does NOT propagate to downstream tool
                                      args
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_downstream_continuation import (  # type: ignore
    CUTPOINTS_PATH, OUT_BASE, ACK_MESSAGE, SYSTEM_PROMPT, TOOLS,
    load_transcript_indexed, get_immediate_context_at, jaccard_args,
    MODEL_ID, TARGET_SAFE,
)
from harness.clients import TargetClient  # type: ignore
from harness.cost import CostTracker  # type: ignore

GPT5_TRANSCRIPT = REPO_ROOT / "data" / "openai_gpt-5_debug_and_fix_baseline_seed301_0952d536c9c9" / "transcript.jsonl"
TARGET_CHARS = 3000


def _flatten_content(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get('type', '')
        if btype == 'text':
            t = block.get('text', '')
            if t:
                parts.append(t)
        elif btype == 'tool_use':
            name = block.get('name', '')
            inp = block.get('input', {})
            if isinstance(inp, dict):
                arg_str = ', '.join(f'{k}={v!r}' for k, v in inp.items())
                parts.append(f"[tool_use {name}: {arg_str}]")
            else:
                parts.append(f"[tool_use {name}]")
        elif btype == 'tool_result':
            r = block.get('content', '')
            if isinstance(r, list):
                r = "".join(x.get('text', '') for x in r if isinstance(x, dict))
            parts.append(f"[tool_result: {str(r)[:300]}]")
    return "\n".join(parts)


def extract_gpt5_recent3K() -> str:
    parts = []
    with GPT5_TRANSCRIPT.open() as f:
        for line in f:
            d = json.loads(line)
            role = d.get("role")
            if role not in ("user", "assistant", "tool_result"):
                continue
            text = _flatten_content(d.get("content", ""))
            if text and text.strip():
                parts.append(f"--- {role.upper()} ---\n{text}")
    full = "\n\n".join(parts)
    return full[-TARGET_CHARS:]


def run_gpt5_arm(client, cut, gpt5_3K_text, user_msg, out_dir: Path) -> dict:
    metrics_path = out_dir / "metrics_gpt5_3K.json"
    if metrics_path.exists():
        try:
            return json.loads(metrics_path.read_text())
        except Exception:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    gt_tool = cut["ground_truth_tool"]
    gt_args = cut["ground_truth_args"] or {}

    msgs = [
        {"role": "user", "content": [{"type": "text", "text": gpt5_3K_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": ACK_MESSAGE}]},
        {"role": "user", "content": [{"type": "text", "text": user_msg}]},
    ]
    t0 = time.perf_counter()
    resp = client.step(
        system_prompt=SYSTEM_PROMPT,
        messages=msgs,
        tools=TOOLS,
        max_tokens=4096,
    )
    elapsed = time.perf_counter() - t0

    (out_dir / "gpt5_3K_response.json").write_text(json.dumps({
        "text": resp.text,
        "tool_calls": [{"name": t.name, "input": t.input} for t in resp.tool_calls],
        "stop_reason": resp.stop_reason,
        "usage": resp.raw_usage,
        "wall_clock_sec": elapsed,
    }, indent=2, default=str))

    g_tool = resp.tool_calls[0].name if resp.tool_calls else None
    g_args = dict(resp.tool_calls[0].input) if resp.tool_calls else {}

    metrics = {
        "cut_index": cut["cut_index"],
        "ground_truth_tool": gt_tool,
        "gpt5_3K_tool": g_tool,
        "M1_gpt5_3K_match": (g_tool == gt_tool) if g_tool else False,
        "M2_gpt5_3K_arg_sim": jaccard_args(g_args, gt_args) if g_tool == gt_tool else None,
        "gpt5_3K_wall_sec": elapsed,
        "gpt5_3K_input_tokens": resp.raw_usage.get("input_tokens"),
        "gpt5_3K_output_tokens": resp.raw_usage.get("output_tokens"),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def main() -> int:
    if not CUTPOINTS_PATH.exists():
        sys.exit("Run scripts/select_cutpoints.py first")
    if not GPT5_TRANSCRIPT.exists():
        sys.exit(f"GPT-5 transcript missing: {GPT5_TRANSCRIPT}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY")

    cuts = json.loads(CUTPOINTS_PATH.read_text())["cutpoints"]
    print(f"Loaded {len(cuts)} cutpoints")

    print("Extracting GPT-5-derived recent3K from synthetic session...")
    gpt5_3K = extract_gpt5_recent3K()
    print(f"  GPT-5 recent3K length: {len(gpt5_3K)} chars (target {TARGET_CHARS})")
    print(f"  preview: {gpt5_3K[:200]!r}")

    print("Loading donated transcript for cut-point user_msg extraction...")
    rows = load_transcript_indexed()

    cost_csv = OUT_BASE / TARGET_SAFE / "cost_log_gpt5.csv"
    cost_csv.parent.mkdir(parents=True, exist_ok=True)
    cost = CostTracker(cost_csv)
    client = TargetClient("anthropic", MODEL_ID, cost_tracker=cost,
                          session_id="downstream_gpt5_control")

    target_dir = OUT_BASE / TARGET_SAFE
    started = time.time()
    n_done = 0
    for i, cut in enumerate(cuts):
        out_dir = target_dir / f"cutpoint-{i:02d}"
        if (out_dir / "metrics_gpt5_3K.json").exists():
            print(f"  [skip {i}] cached")
            n_done += 1
            continue
        user_msg = get_immediate_context_at(rows, cut["cut_index"])
        print(f"\n=== cut {i} (idx={cut['cut_index']}, gt={cut['ground_truth_tool']}) ===")
        try:
            m = run_gpt5_arm(client, cut, gpt5_3K, user_msg, out_dir)
            print(f"  gpt5_3K tool={m['gpt5_3K_tool']} match={m['M1_gpt5_3K_match']}")
            n_done += 1
        except Exception as e:
            print(f"  [ERROR] {e}")

    elapsed = time.time() - started
    print(f"\n=== DONE: {n_done}/{len(cuts)} gpt5_3K cells, {elapsed:.0f}s wall ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
