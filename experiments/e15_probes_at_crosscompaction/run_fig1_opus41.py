"""Fig 1 motivation data: 25 probes × 12 positions × 2 arms × Sonnet 4.6.

Existing data already has 5 C-probes at all 12 positions and 25 probes
at P1/P3/P5. The idempotent skip means this run only fills in the
remaining 360 cells (9 missing positions × 20 missing probes × 2 arms).

~$3, ~30 min wall.

Output: results/probes_at_crosscompaction/claude-opus-4-1/<position>/
        {claude_session,filler}/<probe_id>.json
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
from harness.judge import Judge  # type: ignore
from harness.probes import ALL_PROBES, PROBE_FRAMING  # type: ignore
from experiments.e08_cross_compaction.run import (  # type: ignore
    POSITIONS, extract_prefix_at_turn, find_turn_to_line_index, load_transcript,
)
from experiments.e07_downstream_pilot.run_clean_control import make_filler  # type: ignore
from experiments.e15_probes_at_crosscompaction.run import (  # type: ignore
    run_one_probe, score_one,
)

TRANSCRIPT_PATH = REPO_ROOT / "archive" / "private" / "sessions_raw" / "session_raw_transcript.jsonl"
_TRANSCRIPT_PATH_PUBLIC = REPO_ROOT / "data" / "sessions" / "session_raw_transcript.jsonl"
if not TRANSCRIPT_PATH.exists() and _TRANSCRIPT_PATH_PUBLIC.exists():
    TRANSCRIPT_PATH = _TRANSCRIPT_PATH_PUBLIC

OUT_BASE = REPO_ROOT / "results" / "probes_at_crosscompaction"
TARGET_MODEL = "claude-opus-4-1"
TARGET_SAFE = "claude-opus-4-1"

ALL_POSITIONS = [label for _turn, label in POSITIONS]


def main() -> int:
    if not TRANSCRIPT_PATH.exists():
        sys.exit(f"Transcript missing: {TRANSCRIPT_PATH}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY")

    print(f"Using transcript: {TRANSCRIPT_PATH}")
    print(f"Target model:     {TARGET_MODEL}")
    print(f"Positions:        {len(ALL_POSITIONS)}")
    print(f"Probes:           {len(ALL_PROBES)}")
    print(f"Target cell count (after merge): "
          f"{len(ALL_POSITIONS) * len(ALL_PROBES) * 2}")

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
    target_dir = OUT_BASE / TARGET_SAFE
    target_dir.mkdir(parents=True, exist_ok=True)

    cost_csv = OUT_BASE / f"{TARGET_SAFE}_cost.csv"
    cost = CostTracker(cost_csv)
    client = TargetClient("anthropic", TARGET_MODEL, cost_tracker=cost,
                          session_id="fig1_opus41")

    judge_cost = CostTracker(OUT_BASE / "_judge_cost.csv")
    judge = Judge(provider="anthropic", model_id="claude-opus-4-1",
                  cost_tracker=judge_cost, session_id="fig1_opus41_judge")

    started = time.time()
    n_total = len(ALL_POSITIONS) * len(ALL_PROBES) * 2
    n_done = 0
    n_run = 0  # actually-fired (not skipped)

    for pos_label in ALL_POSITIONS:
        prefix = prefixes[pos_label]
        filler_arm = make_filler(len(prefix)) if prefix else ""
        for probe in ALL_PROBES:
            framed = f"{PROBE_FRAMING}\n\n{probe.text}"
            for arm_name, arm_prefix in (("claude_session", prefix),
                                          ("filler", filler_arm)):
                cell_path = (target_dir / pos_label / arm_name /
                              f"{probe.id}.json")
                if cell_path.exists():
                    n_done += 1
                    continue
                try:
                    gen = run_one_probe(client, arm_prefix, framed, cell_path)
                    judged = score_one(judge, probe.text, gen["response_text"])
                    merged = {
                        "probe_id": probe.id,
                        "probe_category": probe.category,
                        "probe_text": probe.text,
                        "arm": arm_name,
                        "position": pos_label,
                        "target": TARGET_SAFE,
                        **gen, **judged,
                    }
                    cell_path.parent.mkdir(parents=True, exist_ok=True)
                    cell_path.write_text(json.dumps(merged, indent=2,
                                                     default=str))
                    n_done += 1
                    n_run += 1
                except Exception as e:
                    print(f"  [ERR {pos_label} {arm_name} {probe.id}]: {e}")
        elapsed = time.time() - started
        print(f"  {pos_label} done: cum {n_done}/{n_total} (new={n_run}), "
              f"elapsed {elapsed:.0f}s", flush=True)

    elapsed = time.time() - started
    print(f"\n=== ALL DONE: {n_done}/{n_total} cells ({n_run} new), "
          f"{elapsed:.0f}s wall ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
