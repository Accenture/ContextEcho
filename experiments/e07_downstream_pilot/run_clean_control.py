"""Clean downstream test with filler3K control arm.

Per user critique 2026-04-30: the original scratch-vs-recent3K design
confounds 'recent3K causes drift' with 'recent3K provides useful context'.
The arms differ in BOTH content (Claude-flavored vs nothing) AND length
(0 vs 3000 chars).

This runner adds a third arm: filler3K — generic Lorem-ipsum + pangram
content of identical length (3000 chars) to recent3K. The recent3K-vs-
filler3K contrast is now CLEAN: equal length and structure, varying only
flavor. This is the test of persona-drift's downstream cost.

Decision rule:
  filler3K vs scratch       → effect of having any prior context
  recent3K vs scratch       → confounded effect (context + flavor)
  recent3K vs filler3K      → CLEAN effect of Claude-flavor priming alone
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
    load_transcript_indexed, extract_recent3K, get_immediate_context_at,
    jaccard_args, MODEL_ID, TARGET_SAFE,
)
from harness.clients import TargetClient  # type: ignore
from harness.cost import CostTracker  # type: ignore


FILLER_TEMPLATE = (
    "The following is filler placeholder content for an experimental "
    "control. Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "The quick brown fox jumps over the lazy dog. Pack my box with five "
    "dozen liquor jugs. The rain in Spain falls mainly on the plain. "
    "How vexingly quick daft zebras jump. The five boxing wizards jump "
    "quickly. Sphinx of black quartz, judge my vow. Two driven jocks "
    "help fax my big quiz. Cwm fjord bank glyphs vext quiz. "
)


def make_filler(target_chars: int) -> str:
    chunks = []
    total = 0
    while total < target_chars:
        chunks.append(FILLER_TEMPLATE)
        total += len(FILLER_TEMPLATE)
    return "".join(chunks)[:target_chars]


def run_filler3K_arm(client, cut, filler3K_text, user_msg, out_dir: Path) -> dict:
    """Run only the filler3K arm. Other arms (scratch/recent3K) already cached."""
    metrics_path = out_dir / "metrics_filler3K.json"
    if metrics_path.exists():
        try:
            return json.loads(metrics_path.read_text())
        except Exception:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    gt_tool = cut["ground_truth_tool"]
    gt_args = cut["ground_truth_args"] or {}

    filler_messages = [
        {"role": "user", "content": [{"type": "text", "text": filler3K_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": ACK_MESSAGE}]},
        {"role": "user", "content": [{"type": "text", "text": user_msg}]},
    ]
    t0 = time.perf_counter()
    resp = client.step(
        system_prompt=SYSTEM_PROMPT,
        messages=filler_messages,
        tools=TOOLS,
        max_tokens=4096,
    )
    elapsed = time.perf_counter() - t0

    # PII-bearing
    (out_dir / "filler3K_response.json").write_text(json.dumps({
        "text": resp.text,
        "tool_calls": [{"name": t.name, "input": t.input} for t in resp.tool_calls],
        "stop_reason": resp.stop_reason,
        "usage": resp.raw_usage,
        "wall_clock_sec": elapsed,
    }, indent=2, default=str))

    f_tool = resp.tool_calls[0].name if resp.tool_calls else None
    f_args = dict(resp.tool_calls[0].input) if resp.tool_calls else {}

    metrics = {
        "cut_index": cut["cut_index"],
        "ground_truth_tool": gt_tool,
        "filler3K_tool": f_tool,
        "M1_filler3K_match": (f_tool == gt_tool) if f_tool else False,
        "M2_filler3K_arg_sim": jaccard_args(f_args, gt_args) if f_tool == gt_tool else None,
        "filler3K_wall_sec": elapsed,
        "filler3K_input_tokens": resp.raw_usage.get("input_tokens"),
        "filler3K_output_tokens": resp.raw_usage.get("output_tokens"),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def main() -> int:
    if not CUTPOINTS_PATH.exists():
        sys.exit("Run scripts/select_cutpoints.py first")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY")

    cuts = json.loads(CUTPOINTS_PATH.read_text())["cutpoints"]
    print(f"Loaded {len(cuts)} cutpoints")

    print("Loading transcript...")
    rows = load_transcript_indexed()

    cost_csv = OUT_BASE / TARGET_SAFE / "cost_log_filler.csv"
    cost_csv.parent.mkdir(parents=True, exist_ok=True)
    cost = CostTracker(cost_csv)
    client = TargetClient("anthropic", MODEL_ID, cost_tracker=cost,
                          session_id="downstream_filler_control")

    target_dir = OUT_BASE / TARGET_SAFE
    started = time.time()
    n_done = 0
    for i, cut in enumerate(cuts):
        out_dir = target_dir / f"cutpoint-{i:02d}"
        if (out_dir / "metrics_filler3K.json").exists():
            print(f"  [skip {i}] cached")
            n_done += 1
            continue
        # Build filler that exactly matches the recent3K length used originally.
        recent3K_len = len(extract_recent3K(rows, cut["cut_index"]))
        filler3K = make_filler(recent3K_len)
        user_msg = get_immediate_context_at(rows, cut["cut_index"])
        print(f"\n=== cut {i} (idx={cut['cut_index']}, gt={cut['ground_truth_tool']}) "
              f"filler_len={len(filler3K)} ===")
        try:
            m = run_filler3K_arm(client, cut, filler3K, user_msg, out_dir)
            print(f"  filler3K tool={m['filler3K_tool']} match={m['M1_filler3K_match']}")
            n_done += 1
        except Exception as e:
            print(f"  [ERROR] {e}")

    elapsed = time.time() - started
    print(f"\n=== DONE: {n_done}/{len(cuts)} filler3K cells, {elapsed:.0f}s wall ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
