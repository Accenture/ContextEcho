"""Extend e15 probe trajectory to all 12 positions × 3 arms.

Existing data:
  - probes_at_crosscompaction/<target>/{P1_pre_C1,P3_post_C3,P5_pre_C6}/
    {claude_session,filler}/<probe>.json   (3 positions, claude+filler arms)
  - generalization_test/A_COMBINED/CODING_PROBES/<target>/{P0,P3,P5}/
    <probe>.json                            (3 positions, A anchor arm)

We need a 12-position trajectory for the probe figure. Only run CODING_PROBES
(C01–C05; same probes A has data for), so n=5 probes per cell.

Missing positions (9 per target, both arms):
  no-anchor side: P0_start, P2_post_C1, P_pre_C2, P_post_C2, P_pre_C3,
                  P_pre_C4, P_post_C4, P_pre_C5, P4_post_C5
  A-anchor side:  P1_pre_C1, P2_post_C1, P_pre_C2, P_post_C2, P_pre_C3,
                  P_pre_C4, P_post_C4, P_pre_C5, P4_post_C5

Total cells:
  no-anchor (claude_session + filler): 9 positions × 4 targets × 5 probes × 2 arms = 360
  A-anchor:                            9 positions × 4 targets × 5 probes      = 180
  + already-present cells skipped via idempotent on-disk check.

Cells in this run: ~540. ~$10, ~40 min wall.

Output:
  results/probes_at_crosscompaction/<target>/<position>/{claude_session,filler}/<probe>.json
  results/generalization_test/A_COMBINED/CODING_PROBES/<target>/<position>/<probe>.json
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
from harness.probes import CODING_PROBES, PROBE_FRAMING  # type: ignore
from experiments.e08_cross_compaction.run import (  # type: ignore
    POSITIONS, TARGETS,
    extract_prefix_at_turn, find_turn_to_line_index, load_transcript,
)
from experiments.e07_downstream_pilot.run_clean_control import make_filler  # type: ignore
from experiments.e15_probes_at_crosscompaction.run import (  # type: ignore
    SYSTEM_PROMPT, run_one_probe, score_one,
)
from experiments.e04_path_y.dual_surface_candidates import build_messages_a  # type: ignore

TRANSCRIPT_PATH = REPO_ROOT / "archive" / "private" / "sessions_raw" / "session_raw_transcript.jsonl"
_TRANSCRIPT_PATH_PUBLIC = REPO_ROOT / "data" / "sessions" / "session_raw_transcript.jsonl"
if not TRANSCRIPT_PATH.exists() and _TRANSCRIPT_PATH_PUBLIC.exists():
    TRANSCRIPT_PATH = _TRANSCRIPT_PATH_PUBLIC

NO_ANCHOR_OUT = REPO_ROOT / "results" / "probes_at_crosscompaction"
A_OUT = REPO_ROOT / "results" / "generalization_test" / "A_COMBINED" / "CODING_PROBES"

ALL_POSITIONS = [label for _turn, label in POSITIONS]


def run_a_anchor_probe(client, prefix: str, probe_text: str,
                        out_path: Path) -> dict:
    """Generate model response under A anchor at this position."""
    if out_path.exists():
        try:
            return json.loads(out_path.read_text())
        except Exception:
            pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    framed = f"{PROBE_FRAMING}\n\n{probe_text}"
    msgs = build_messages_a(prefix, framed)
    t0 = time.perf_counter()
    resp = client.step(system_prompt=SYSTEM_PROMPT, messages=msgs, tools=[],
                      max_tokens=600)
    elapsed = time.perf_counter() - t0
    return {
        "response_text": resp.text,
        "response_len": len(resp.text),
        "wall_clock_sec": elapsed,
        "input_tokens": resp.raw_usage.get("input_tokens"),
        "output_tokens": resp.raw_usage.get("output_tokens"),
    }


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

    NO_ANCHOR_OUT.mkdir(parents=True, exist_ok=True)
    A_OUT.mkdir(parents=True, exist_ok=True)

    judge_cost = CostTracker(NO_ANCHOR_OUT / "_judge_cost.csv")
    judge = Judge(provider="anthropic", model_id="claude-sonnet-4-6",
                  cost_tracker=judge_cost, session_id="e15_12pos_judge")

    started = time.time()
    n_done = 0
    n_total = (len(ALL_POSITIONS) * len(TARGETS) * len(CODING_PROBES) * 3)
    print(f"Total cells (across 3 arms): {n_total}")

    for model_id, target_safe in TARGETS:
        cost_csv = NO_ANCHOR_OUT / f"{target_safe}_cost.csv"
        cost = CostTracker(cost_csv)
        client = TargetClient("anthropic", model_id, cost_tracker=cost,
                              session_id=f"e15_12pos_{target_safe}")
        print(f"\n=== {target_safe} ===")

        for pos_label in ALL_POSITIONS:
            prefix = prefixes[pos_label]
            filler_arm = make_filler(len(prefix)) if prefix else ""

            for probe in CODING_PROBES:
                framed = f"{PROBE_FRAMING}\n\n{probe.text}"

                # ----- no-anchor: claude_session arm -----
                cs_path = (NO_ANCHOR_OUT / target_safe / pos_label /
                            "claude_session" / f"{probe.id}.json")
                if not cs_path.exists():
                    try:
                        gen = run_one_probe(client, prefix, framed, cs_path)
                        judged = score_one(judge, probe.text, gen["response_text"])
                        merged = {
                            "probe_id": probe.id, "probe_category": probe.category,
                            "probe_text": probe.text, "arm": "claude_session",
                            "position": pos_label, "target": target_safe,
                            **gen, **judged,
                        }
                        cs_path.parent.mkdir(parents=True, exist_ok=True)
                        cs_path.write_text(json.dumps(merged, indent=2, default=str))
                    except Exception as e:
                        print(f"  [ERR claude_session {pos_label} {probe.id}]: {e}")
                n_done += 1

                # ----- no-anchor: filler arm -----
                fi_path = (NO_ANCHOR_OUT / target_safe / pos_label /
                            "filler" / f"{probe.id}.json")
                if not fi_path.exists():
                    try:
                        gen = run_one_probe(client, filler_arm, framed, fi_path)
                        judged = score_one(judge, probe.text, gen["response_text"])
                        merged = {
                            "probe_id": probe.id, "probe_category": probe.category,
                            "probe_text": probe.text, "arm": "filler",
                            "position": pos_label, "target": target_safe,
                            **gen, **judged,
                        }
                        fi_path.parent.mkdir(parents=True, exist_ok=True)
                        fi_path.write_text(json.dumps(merged, indent=2, default=str))
                    except Exception as e:
                        print(f"  [ERR filler {pos_label} {probe.id}]: {e}")
                n_done += 1

                # ----- A-anchor arm -----
                a_path = A_OUT / target_safe / pos_label / f"{probe.id}.json"
                if not a_path.exists():
                    try:
                        gen = run_a_anchor_probe(client, prefix, probe.text, a_path)
                        judged = score_one(judge, probe.text, gen["response_text"])
                        merged = {
                            "probe_id": probe.id, "probe_text": probe.text,
                            "arm": "A_COMBINED", "position": pos_label,
                            "target": target_safe, **gen, **judged,
                        }
                        a_path.parent.mkdir(parents=True, exist_ok=True)
                        a_path.write_text(json.dumps(merged, indent=2, default=str))
                    except Exception as e:
                        print(f"  [ERR A_anchor {pos_label} {probe.id}]: {e}")
                n_done += 1

            print(f"  {target_safe} {pos_label}: cum {n_done}/{n_total}, "
                  f"elapsed {time.time()-started:.0f}s")

    elapsed = time.time() - started
    print(f"\n=== ALL DONE: {n_done}/{n_total} cells, {elapsed:.0f}s wall ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
