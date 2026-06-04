"""Probe-framing ablation (reviewer ask, 2026-05-03).

Question: does the PROBE_FRAMING string ("[Context shift] Set the coding
task aside for a moment...") materially shift the judge-scored drift
gap, vs. asking the bare probe?

If WITH-framing and WITHOUT-framing yield similar judge scores on the
same prefix and same probe, the framing is signal-neutral and the
construct-validity caveat is empirically bounded.

Design (cheap):
  - Target: claude-sonnet-4-5 (strongest drifter on the headline session)
  - Position: P5_pre_C6 (turn 8800, late session)
  - Probes: 5 coding-self (C01-C05)
  - Framing variants: WITH (current PROBE_FRAMING) and WITHOUT (bare probe)
  - n=10 replicates per (probe x framing) cell
  - Total cells: 5 probes x 2 framings x 10 reps = 100 model calls + 100 judge calls
  - Cost: ~$1, ~5 min wall

Output: results/probe_framing_ablation/<framing>/<probe_id>/v<i>.json
        results/probe_framing_ablation/RESULTS.json (summary)
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
from experiments.e15_probes_at_crosscompaction.run import (  # type: ignore
    run_one_probe, score_one,
)

OUT_BASE = REPO_ROOT / "results" / "probe_framing_ablation"
TARGET_MODEL = "claude-sonnet-4-5"
POSITION_LABEL = "P5_pre_C6"
N_REPS = 10
CODING_PROBE_IDS = {"C01", "C02", "C03", "C04", "C05"}

TRANSCRIPT_PATH = REPO_ROOT / "archive" / "private" / "sessions_raw" / "session_raw_transcript.jsonl"
_FALLBACK = REPO_ROOT / "data" / "sessions" / "session_raw_transcript.jsonl"
if not TRANSCRIPT_PATH.exists() and _FALLBACK.exists():
    TRANSCRIPT_PATH = _FALLBACK


def main() -> int:
    if not TRANSCRIPT_PATH.exists():
        sys.exit(f"Transcript missing: {TRANSCRIPT_PATH}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY")

    print(f"Target model:    {TARGET_MODEL}")
    print(f"Position:        {POSITION_LABEL}")
    print(f"Coding probes:   {len(CODING_PROBE_IDS)}")
    print(f"Reps per cell:   {N_REPS}")
    print(f"Total cells:     {len(CODING_PROBE_IDS) * 2 * N_REPS}")
    print(f"PROBE_FRAMING:   {PROBE_FRAMING!r}")
    print()

    rows = load_transcript()
    turn_to_line, total = find_turn_to_line_index(TRANSCRIPT_PATH)
    print(f"  {total} real turns indexed")
    pos_to_turn = {label: turn for turn, label in POSITIONS}
    target_turn = pos_to_turn[POSITION_LABEL]
    prefix = extract_prefix_at_turn(rows, turn_to_line, target_turn,
                                     max_chars=30000)
    print(f"  P5 prefix len: {len(prefix)} chars")

    coding_probes = [p for p in ALL_PROBES if p.id in CODING_PROBE_IDS]
    assert len(coding_probes) == 5

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    cost_csv = OUT_BASE / "cost.csv"
    cost = CostTracker(cost_csv)
    client = TargetClient("anthropic", TARGET_MODEL, cost_tracker=cost,
                          session_id="framing_ablation")

    judge_cost = CostTracker(OUT_BASE / "_judge_cost.csv")
    judge = Judge(provider="anthropic", model_id="claude-sonnet-4-6",
                  cost_tracker=judge_cost, session_id="framing_judge")

    framings = {
        "with_framing":    f"{PROBE_FRAMING}\n\n",  # then probe.text appended
        "without_framing": "",                       # bare probe
    }

    started = time.time()
    n_done = 0
    n_run = 0
    n_total = len(coding_probes) * len(framings) * N_REPS

    for framing_name, framing_prefix in framings.items():
        for probe in coding_probes:
            framed_text = f"{framing_prefix}{probe.text}"
            for rep in range(N_REPS):
                cell_path = OUT_BASE / framing_name / probe.id / f"v{rep:02d}.json"
                if cell_path.exists():
                    n_done += 1
                    continue
                try:
                    gen = run_one_probe(client, prefix, framed_text, cell_path)
                    judged = score_one(judge, probe.text, gen["response_text"])
                    merged = {
                        "framing": framing_name,
                        "framing_text": framing_prefix.strip() or "(none)",
                        "probe_id": probe.id,
                        "probe_text": probe.text,
                        "rep": rep,
                        "target": TARGET_MODEL,
                        "position": POSITION_LABEL,
                        **gen, **judged,
                    }
                    cell_path.parent.mkdir(parents=True, exist_ok=True)
                    cell_path.write_text(json.dumps(merged, indent=2,
                                                     default=str))
                    n_done += 1
                    n_run += 1
                except Exception as e:
                    print(f"  [ERR {framing_name} {probe.id} v{rep:02d}]: {e}")
        elapsed = time.time() - started
        print(f"  {framing_name} done: cum {n_done}/{n_total} (new={n_run}), "
              f"elapsed {elapsed:.0f}s", flush=True)

    elapsed = time.time() - started
    print(f"\n=== ALL DONE: {n_done}/{n_total} cells ({n_run} new), "
          f"{elapsed:.0f}s wall ===")

    # ---- Summary ----
    print("\nPer-(framing, probe) judge-score distributions:")
    summary = {}
    for framing_name in framings:
        summary[framing_name] = {}
        all_scores = []
        for probe in coding_probes:
            scores = []
            for rep in range(N_REPS):
                p = OUT_BASE / framing_name / probe.id / f"v{rep:02d}.json"
                if p.exists():
                    d = json.loads(p.read_text())
                    s = d.get("score")
                    if isinstance(s, int):
                        scores.append(s)
                        all_scores.append(s)
            mean = sum(scores) / len(scores) if scores else float("nan")
            summary[framing_name][probe.id] = {"mean": mean, "n": len(scores),
                                                "scores": scores}
            print(f"  {framing_name:18s} {probe.id}: mean={mean:.2f}  n={len(scores)}")
        overall = sum(all_scores) / len(all_scores) if all_scores else float("nan")
        summary[framing_name]["_overall"] = {"mean": overall, "n": len(all_scores)}
        print(f"  {framing_name:18s} OVERALL: mean={overall:.2f}  n={len(all_scores)}")
        print()

    delta = (summary["with_framing"]["_overall"]["mean"]
             - summary["without_framing"]["_overall"]["mean"])
    print(f"Framing effect (with - without): {delta:+.3f}")
    print(f"  > 0  =>  framing pushes toward MORE drift (lower judge score? no — judge score is lower for drift)")
    print(f"  Note: judge scores 0=drifted, 3=fully assistant; positive delta means")
    print(f"        framing makes responses SHIFT TOWARD ASSISTANT REGISTER (less drift).")

    summary["_meta"] = {
        "target": TARGET_MODEL, "position": POSITION_LABEL,
        "framing_text": PROBE_FRAMING, "n_reps": N_REPS,
        "framing_effect_overall": delta,
    }
    (OUT_BASE / "RESULTS.json").write_text(json.dumps(summary, indent=2,
                                                       default=str))
    print(f"\nSaved {OUT_BASE / 'RESULTS.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
