"""Expand A_COMBINED (V0 identity + V2 demo) to all 12 trajectory positions
on 4 Anthropic targets, S2 stressor surface.

Lets us redraw Fig 1(b) with A as the mitigation line (apples-to-apples vs
the existing V2 mitigation line).

Total cells: 4 targets × 12 positions × 10 S2 paraphrases = 480 cells.
~$8, ~30 min wall.

Output: results/anchor_variants/A_COMBINED/<target>/<position>/v{NN}/cell.json
"""
from __future__ import annotations

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
from experiments.e04_path_y.run_anchor_variants import run_one  # type: ignore
from experiments.e04_path_y.dual_surface_candidates import build_messages_a  # type: ignore

TRANSCRIPT_PATH = REPO_ROOT / "archive" / "private" / "sessions_raw" / "session_raw_transcript.jsonl"
_TRANSCRIPT_PATH_PUBLIC = REPO_ROOT / "data" / "sessions" / "session_raw_transcript.jsonl"
if not TRANSCRIPT_PATH.exists() and _TRANSCRIPT_PATH_PUBLIC.exists():
    TRANSCRIPT_PATH = _TRANSCRIPT_PATH_PUBLIC

OUT_BASE = REPO_ROOT / "results" / "anchor_variants"
VARIANT_NAME = "A_COMBINED"
ALL_POSITIONS = [label for _turn, label in POSITIONS]


def main() -> int:
    if not TRANSCRIPT_PATH.exists():
        sys.exit(f"Transcript missing: {TRANSCRIPT_PATH}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY")

    print(f"Using transcript: {TRANSCRIPT_PATH}")
    print(f"Positions: {len(ALL_POSITIONS)}; targets: {len(TARGETS)}; "
          f"paraphrases: {len(S2_VARIANTS)}")
    print(f"Total cells: {len(ALL_POSITIONS) * len(TARGETS) * len(S2_VARIANTS)}")

    rows = load_transcript()
    turn_to_line, total = find_turn_to_line_index(TRANSCRIPT_PATH)
    print(f"  {total} real turns indexed")

    pos_to_turn = {label: turn for turn, label in POSITIONS}
    prefixes = {}
    for label in ALL_POSITIONS:
        turn = pos_to_turn[label]
        prefixes[label] = extract_prefix_at_turn(rows, turn_to_line, turn,
                                                  max_chars=30000)

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    started = time.time()
    n_total = len(TARGETS) * len(ALL_POSITIONS) * len(S2_VARIANTS)
    n_done = 0

    for model_id, target_safe in TARGETS:
        cost_csv = OUT_BASE / VARIANT_NAME / f"{target_safe}_cost.csv"
        cost = CostTracker(cost_csv)
        client = TargetClient("anthropic", model_id, cost_tracker=cost,
                              session_id=f"anchor_{VARIANT_NAME}_{target_safe}")
        print(f"\n=== {target_safe} ===")
        for pos_label in ALL_POSITIONS:
            prefix = prefixes[pos_label]
            for v_idx, stressor in enumerate(S2_VARIANTS):
                out_path = (OUT_BASE / VARIANT_NAME / target_safe /
                            pos_label / f"v{v_idx:02d}" / "cell.json")
                if out_path.exists():
                    n_done += 1
                    continue
                try:
                    m = run_one(client, prefix, stressor, VARIANT_NAME,
                                build_messages_a, SYSTEM_PROMPT, out_path)
                    n_done += 1
                    if v_idx == 0:
                        print(f"  [{target_safe} {pos_label} v0] "
                              f"len={m['response_len']} resp={m['response_text'][:80]!r}")
                except Exception as e:
                    print(f"  [ERROR {target_safe} {pos_label} v{v_idx}]: {e}")

    elapsed = time.time() - started
    print(f"\n=== ALL DONE: {n_done}/{n_total} cells, {elapsed:.0f}s wall ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
