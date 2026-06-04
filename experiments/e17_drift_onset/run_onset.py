"""Drift-onset experiment: dense pre-C1 turn sweep on Sonnet 4.5.

Question: at what turn does drift first appear? Our 12-position grid leaves
the entire pre-C1 regime (turns 1..1338) covered by only P0_start (turn 100)
and P1_pre_C1 (turn 1300). This runner densely samples the 0..1500 range.

Output: results/drift_onset/<TARGET_SAFE>/T<NNNN>/{claude_session,filler}/<probe_id>.json

Cells: 8 turns x 5 coding-self probes x 2 arms = 80.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from harness.clients import TargetClient  # type: ignore
from harness.cost import CostTracker  # type: ignore
from harness.judge import Judge  # type: ignore
from harness.probes import ALL_PROBES, PROBE_FRAMING  # type: ignore
from experiments.e08_cross_compaction.run import (  # type: ignore
    extract_prefix_at_turn, find_turn_to_line_index, load_transcript,
)
from experiments.e07_downstream_pilot.run_clean_control import make_filler  # type: ignore
from experiments.e15_probes_at_crosscompaction.run import (  # type: ignore
    run_one_probe, score_one,
)

TRANSCRIPT_PATH = REPO_ROOT / "archive" / "private" / "sessions_raw" / "session_raw_transcript.jsonl"
_TRANSCRIPT_PATH_PUBLIC = REPO_ROOT / "data" / "sessions" / "session_raw_transcript.jsonl"
if not TRANSCRIPT_PATH.exists():
    TRANSCRIPT_PATH = _TRANSCRIPT_PATH_PUBLIC

OUT_BASE = REPO_ROOT / "results" / "drift_onset"

PROVIDER = os.environ.get("PROVIDER", "anthropic")
MODEL_ID = os.environ.get("MODEL_ID", "claude-sonnet-4-5")
TARGET_SAFE = os.environ.get("TARGET_SAFE", "claude-sonnet-4-5")
# Number of replicates per (position, probe, arm) cell. Default 1 = single
# cell at C01.json. With N=5 we additionally write C01_v01.json..C01_v04.json
# (skipping the bare-name case to keep prior runs untouched), totaling 5
# observations per cell.
N_REPLICATES = int(os.environ.get("N_REPLICATES", "1"))

# Log-spaced turn positions in the pre-C1 regime (C1 is at turn 1338).
# 8 points cover three decades: 1, 5, 25, 100, 250, 500, 1000, 1500.
ONSET_TURNS = [1, 5, 25, 100, 250, 500, 1000, 1500]

# Coding-self subset only (matches body's primary metric).
CODING_PROBE_IDS = {"C01", "C02", "C03", "C04", "C05"}
PROBES = [p for p in ALL_PROBES if p.id in CODING_PROBE_IDS]


def main() -> int:
    if not TRANSCRIPT_PATH.exists():
        sys.exit(f"Transcript missing: {TRANSCRIPT_PATH}")

    print(f"Transcript:    {TRANSCRIPT_PATH}")
    print(f"Provider:      {PROVIDER}")
    print(f"Target model:  {MODEL_ID}")
    print(f"Target safe:   {TARGET_SAFE}")
    print(f"Turns:         {ONSET_TURNS}")
    print(f"Probes:        {[p.id for p in PROBES]}")
    print(f"Cell count:    {len(ONSET_TURNS) * len(PROBES) * 2}")

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    judge_cost = CostTracker(OUT_BASE / "_judge_cost.csv")
    judge = Judge(provider="anthropic", model_id="claude-sonnet-4-6",
                  cost_tracker=judge_cost, session_id="e17_onset_judge")

    cost_csv = OUT_BASE / f"{TARGET_SAFE}_cost.csv"
    cost = CostTracker(cost_csv)
    client = TargetClient(provider=PROVIDER, model_id=MODEL_ID,  # type: ignore[arg-type]
                          cost_tracker=cost,
                          session_id=f"e17_onset_{TARGET_SAFE}")
    target_dir = OUT_BASE / TARGET_SAFE
    target_dir.mkdir(parents=True, exist_ok=True)

    rows = load_transcript()
    turn_to_line, total = find_turn_to_line_index(TRANSCRIPT_PATH)
    print(f"  {total} real turns indexed")

    # Pre-extract all prefixes
    prefixes = {}
    for turn in ONSET_TURNS:
        p = extract_prefix_at_turn(rows, turn_to_line, turn, max_chars=30000)
        prefixes[turn] = p
        print(f"  turn={turn:>4}  prefix_len={len(p):>6} chars")

    n_total = len(ONSET_TURNS) * len(PROBES) * 2 * N_REPLICATES
    n_done = 0
    n_run = 0
    started = time.perf_counter()

    for turn in ONSET_TURNS:
        prefix = prefixes[turn]
        filler_arm = make_filler(len(prefix)) if prefix else ""
        pos_label = f"T{turn:04d}"

        for probe in PROBES:
            framed = f"{PROBE_FRAMING}\n\n{probe.text}"
            for arm_name, arm_prefix in (("claude_session", prefix),
                                          ("filler", filler_arm)):
                for rep_idx in range(N_REPLICATES):
                    if rep_idx == 0:
                        cell_name = f"{probe.id}.json"
                    else:
                        cell_name = f"{probe.id}_v{rep_idx:02d}.json"
                    cell_path = target_dir / pos_label / arm_name / cell_name
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
                            "turn": turn,
                            "target": TARGET_SAFE,
                            "provider": PROVIDER,
                            "model_id": MODEL_ID,
                            "replicate": rep_idx,
                            **gen,
                            **judged,
                        }
                        cell_path.parent.mkdir(parents=True, exist_ok=True)
                        cell_path.write_text(json.dumps(merged, indent=2))
                        n_run += 1
                        n_done += 1
                    except Exception as e:
                        print(f"  ERROR {pos_label} {arm_name} {probe.id} v{rep_idx}: {e}")

        elapsed = int(time.perf_counter() - started)
        print(f"  {pos_label} done: cum {n_done}/{n_total} (new={n_run}), {elapsed}s")

    print(f"\nDONE — {n_done}/{n_total} cells, {n_run} new, "
          f"{int(time.perf_counter() - started)}s wall.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
