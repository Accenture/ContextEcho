"""Cross-org variant of run_fig1_*.py — runs 25 probes × 12 positions × 2 arms
on a non-Anthropic target via the existing TargetClient (now extended for
together provider).

Usage (env-driven):
  PROVIDER=openai    MODEL_ID=gpt-5                       TARGET_SAFE=gpt-5         python3 ...run_fig1_xtarget.py
  PROVIDER=together  MODEL_ID=deepseek-ai/DeepSeek-V3     TARGET_SAFE=deepseek-v3   python3 ...run_fig1_xtarget.py

Output: results/probes_at_crosscompaction/<TARGET_SAFE>/<position>/{claude_session,filler}/<probe_id>.json
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

PROVIDER = os.environ.get("PROVIDER", "")
MODEL_ID = os.environ.get("MODEL_ID", "")
TARGET_SAFE = os.environ.get("TARGET_SAFE", "")
# Comma-separated subset of position labels; empty = all 12.
POSITIONS_OVERRIDE = os.environ.get("POSITIONS_OVERRIDE", "")

_FULL_POSITIONS = [label for _turn, label in POSITIONS]
if POSITIONS_OVERRIDE:
    _wanted = [p.strip() for p in POSITIONS_OVERRIDE.split(",") if p.strip()]
    ALL_POSITIONS = [p for p in _FULL_POSITIONS if p in _wanted]
    if not ALL_POSITIONS:
        sys.exit(f"POSITIONS_OVERRIDE matched no known positions; got {_wanted}")
else:
    ALL_POSITIONS = _FULL_POSITIONS


def main() -> int:
    if not (PROVIDER and MODEL_ID and TARGET_SAFE):
        sys.exit("Set PROVIDER, MODEL_ID, TARGET_SAFE env vars.")
    if not TRANSCRIPT_PATH.exists():
        sys.exit(f"Transcript missing: {TRANSCRIPT_PATH}")

    print(f"Using transcript: {TRANSCRIPT_PATH}")
    print(f"Provider:         {PROVIDER}")
    print(f"Target model:     {MODEL_ID}")
    print(f"Target safe name: {TARGET_SAFE}")
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
        p = extract_prefix_at_turn(rows, turn_to_line, turn, max_chars=30000)
        prefixes[label] = p

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    judge_cost = CostTracker(OUT_BASE / "_judge_cost.csv")
    judge = Judge(provider="anthropic", model_id="claude-sonnet-4-6",
                  cost_tracker=judge_cost, session_id="e15_probes_judge_xtarget")

    cost_csv = OUT_BASE / f"{TARGET_SAFE}_cost.csv"
    cost = CostTracker(cost_csv)
    client = TargetClient(provider=PROVIDER, model_id=MODEL_ID,  # type: ignore[arg-type]
                          cost_tracker=cost,
                          session_id=f"e15_probes_{TARGET_SAFE}")
    target_dir = OUT_BASE / TARGET_SAFE
    target_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(ALL_POSITIONS) * len(ALL_PROBES) * 2
    n_done = 0
    n_run = 0
    started = time.perf_counter()

    for pos_label in ALL_POSITIONS:
        prefix = prefixes[pos_label]
        filler_arm = make_filler(len(prefix)) if prefix else ""

        for probe in ALL_PROBES:
            framed = f"{PROBE_FRAMING}\n\n{probe.text}"
            for arm_name, arm_prefix in (("claude_session", prefix),
                                          ("filler", filler_arm)):
                cell_path = (target_dir / pos_label / arm_name / f"{probe.id}.json")
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
                        "provider": PROVIDER,
                        "model_id": MODEL_ID,
                        **gen,
                        **judged,
                    }
                    cell_path.parent.mkdir(parents=True, exist_ok=True)
                    cell_path.write_text(json.dumps(merged, indent=2))
                    n_run += 1
                    n_done += 1
                except Exception as e:
                    print(f"  ERROR {pos_label} {arm_name} {probe.id}: {e}")

        elapsed = int(time.perf_counter() - started)
        print(f"  {pos_label} done: cum {n_done}/{n_total} (new={n_run}), elapsed {elapsed}s")

    print(f"\nDONE — {n_done}/{n_total} cells, {n_run} new, "
          f"{int(time.perf_counter() - started)}s wall.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
