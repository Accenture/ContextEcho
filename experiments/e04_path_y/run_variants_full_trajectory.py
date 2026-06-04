"""Expand V1 and V3 from the 3-position pilot to all 12 trajectory positions.

V0 and V2 are already complete on the full trajectory:
  V0: results/cross_compaction_pathy/  (480 cells)
  V2: results/anchor_variants/V2_IN_CONTEXT/  (480 cells)

This script fills in V1_BEHAVIORAL and V3_SYSTEM_INJECT at all 12 positions,
reusing the existing pilot cells for P0/P3/P5 (idempotent skip).

Total cells: 4 targets × 12 positions × 10 paraphrases × 2 variants = 960.
Existing 240 pilot cells reused; 720 new. ~$16, ~1 hr.
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
from experiments.e04_path_y.run_anchor_variants import (  # type: ignore
    build_messages_v1, build_messages_v3, run_one, ANCHOR_V0_TEXT,
)

TRANSCRIPT_PATH = REPO_ROOT / "archive" / "private" / "sessions_raw" / "session_raw_transcript.jsonl"
_TRANSCRIPT_PATH_PUBLIC = REPO_ROOT / "data" / "sessions" / "session_raw_transcript.jsonl"
if not TRANSCRIPT_PATH.exists() and _TRANSCRIPT_PATH_PUBLIC.exists():
    TRANSCRIPT_PATH = _TRANSCRIPT_PATH_PUBLIC

OUT_BASE = REPO_ROOT / "results" / "anchor_variants"
ALL_POSITIONS = [label for _turn, label in POSITIONS]

VARIANTS_TO_RUN = [
    ("V1_BEHAVIORAL",   build_messages_v1, SYSTEM_PROMPT),
    ("V3_SYSTEM_INJECT", build_messages_v3,
     SYSTEM_PROMPT + "\n\n" + ANCHOR_V0_TEXT),
]


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
    for label in ALL_POSITIONS:
        turn = pos_to_turn[label]
        prefixes[label] = extract_prefix_at_turn(rows, turn_to_line, turn,
                                                  max_chars=30000)

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    started = time.time()
    n_total = (len(VARIANTS_TO_RUN) * len(TARGETS) * len(ALL_POSITIONS)
               * len(S2_VARIANTS))
    n_done = 0

    for variant_name, build_msgs, system_prompt in VARIANTS_TO_RUN:
        print(f"\n{'='*60}\nVariant: {variant_name}\n{'='*60}")
        for model_id, target_safe in TARGETS:
            cost_csv = OUT_BASE / variant_name / f"{target_safe}_cost.csv"
            cost = CostTracker(cost_csv)
            client = TargetClient("anthropic", model_id, cost_tracker=cost,
                                  session_id=f"anchor_{variant_name}_{target_safe}")
            print(f"\n--- {target_safe} ---")
            for pos_label in ALL_POSITIONS:
                prefix = prefixes[pos_label]
                for v_idx, stressor in enumerate(S2_VARIANTS):
                    out_path = (OUT_BASE / variant_name / target_safe /
                                pos_label / f"v{v_idx:02d}" / "cell.json")
                    if out_path.exists():
                        n_done += 1
                        continue
                    try:
                        m = run_one(client, prefix, stressor, variant_name,
                                    build_msgs, system_prompt, out_path)
                        n_done += 1
                        if v_idx == 0:
                            print(f"  [{variant_name} {target_safe} {pos_label} v0] "
                                  f"len={m['response_len']} "
                                  f"resp={m['response_text'][:70]!r}")
                    except Exception as e:
                        print(f"  [ERROR {variant_name} {target_safe} "
                              f"{pos_label} v{v_idx}]: {e}")

    elapsed = time.time() - started
    print(f"\n=== ALL DONE: {n_done}/{n_total} cells, {elapsed:.0f}s wall ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
