"""Dual-surface anchor pilot — STRESSOR side.

For each candidate × target × position × paraphrase:
  prefix → anchor_framing → ack → (demo turns if any) → stressor

Output: results/dual_surface_pilot/<candidate>/stressors/<target>/<position>/v{NN}/cell.json
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
from harness.cost import CostTracker  # type: ignore
from experiments.e08_cross_compaction.run import (  # type: ignore
    POSITIONS, S2_VARIANTS, TARGETS,
    extract_prefix_at_turn, find_turn_to_line_index, load_transcript,
)
from experiments.e11_instruction_override.run import SYSTEM_PROMPT  # type: ignore
from experiments.e04_path_y.dual_surface_candidates import CANDIDATES  # type: ignore

TRANSCRIPT_PATH = REPO_ROOT / "archive" / "private" / "sessions_raw" / "session_raw_transcript.jsonl"
_TRANSCRIPT_PATH_PUBLIC = REPO_ROOT / "data" / "sessions" / "session_raw_transcript.jsonl"
if not TRANSCRIPT_PATH.exists() and _TRANSCRIPT_PATH_PUBLIC.exists():
    TRANSCRIPT_PATH = _TRANSCRIPT_PATH_PUBLIC

OUT_BASE = REPO_ROOT / "results" / "dual_surface_pilot"
PILOT_POSITIONS = ["P0_start", "P3_post_C3", "P5_pre_C6"]


def run_one(client, prefix: str, stressor: str, build_msgs, out_path: Path) -> dict:
    if out_path.exists():
        try:
            return json.loads(out_path.read_text())
        except Exception:
            pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    msgs = build_msgs(prefix, stressor)
    t0 = time.perf_counter()
    resp = client.step(system_prompt=SYSTEM_PROMPT, messages=msgs, tools=[],
                      max_tokens=1024)
    elapsed = time.perf_counter() - t0
    text = resp.text
    metrics = {
        "response_text": text, "response_len": len(text),
        "wall_clock_sec": elapsed,
        "input_tokens": resp.raw_usage.get("input_tokens"),
        "output_tokens": resp.raw_usage.get("output_tokens"),
    }
    out_path.write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def main() -> int:
    if not TRANSCRIPT_PATH.exists():
        sys.exit(f"Transcript missing: {TRANSCRIPT_PATH}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY")

    print(f"Using transcript: {TRANSCRIPT_PATH}")
    rows = load_transcript()
    turn_to_line, total = find_turn_to_line_index(TRANSCRIPT_PATH)
    print(f"  {total} real turns indexed")

    pos_to_turn = {label: turn for turn, label in POSITIONS}
    prefixes = {}
    for label in PILOT_POSITIONS:
        turn = pos_to_turn[label]
        prefixes[label] = extract_prefix_at_turn(rows, turn_to_line, turn,
                                                  max_chars=30000)
        print(f"  {label}: prefix len = {len(prefixes[label])} chars")

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    started = time.time()
    n_total = (len(CANDIDATES) * len(TARGETS) * len(PILOT_POSITIONS)
               * len(S2_VARIANTS))
    n_done = 0

    for cand_name, build_msgs in CANDIDATES:
        print(f"\n{'='*60}\nCandidate: {cand_name}\n{'='*60}")
        for model_id, target_safe in TARGETS:
            cost_csv = OUT_BASE / cand_name / "stressors" / f"{target_safe}_cost.csv"
            cost = CostTracker(cost_csv)
            client = TargetClient("anthropic", model_id, cost_tracker=cost,
                                  session_id=f"dualstr_{cand_name}_{target_safe}")
            for pos_label in PILOT_POSITIONS:
                prefix = prefixes[pos_label]
                for v_idx, stressor in enumerate(S2_VARIANTS):
                    out_path = (OUT_BASE / cand_name / "stressors" / target_safe /
                                pos_label / f"v{v_idx:02d}" / "cell.json")
                    if out_path.exists():
                        n_done += 1
                        continue
                    try:
                        m = run_one(client, prefix, stressor, build_msgs, out_path)
                        n_done += 1
                        if v_idx == 0:
                            print(f"  [{cand_name} {target_safe} {pos_label} v0] "
                                  f"len={m['response_len']} "
                                  f"resp={m['response_text'][:70]!r}")
                    except Exception as e:
                        print(f"  [ERROR {cand_name} {target_safe} {pos_label}]: {e}")

    elapsed = time.time() - started
    print(f"\n=== STRESSORS DONE: {n_done}/{n_total} cells, {elapsed:.0f}s ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
