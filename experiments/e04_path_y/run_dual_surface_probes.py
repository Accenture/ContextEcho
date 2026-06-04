"""Dual-surface anchor pilot — PROBE side.

For each candidate × target × position × probe (25):
  prefix → anchor_framing → ack → (demo turns if any) → probe with PROBE_FRAMING

Then score with the Sonnet 4.6 judge on the 0-3 hedge-compliance rubric
(same as e15).

Output: results/dual_surface_pilot/<candidate>/probes/<target>/<position>/<probe_id>.json
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
    POSITIONS, TARGETS,
    extract_prefix_at_turn, find_turn_to_line_index, load_transcript,
)
from experiments.e04_path_y.dual_surface_candidates import CANDIDATES  # type: ignore

TRANSCRIPT_PATH = REPO_ROOT / "archive" / "private" / "sessions_raw" / "session_raw_transcript.jsonl"
_TRANSCRIPT_PATH_PUBLIC = REPO_ROOT / "data" / "sessions" / "session_raw_transcript.jsonl"
if not TRANSCRIPT_PATH.exists() and _TRANSCRIPT_PATH_PUBLIC.exists():
    TRANSCRIPT_PATH = _TRANSCRIPT_PATH_PUBLIC

OUT_BASE = REPO_ROOT / "results" / "dual_surface_pilot"
PILOT_POSITIONS = ["P0_start", "P3_post_C3", "P5_pre_C6"]

# Same SYSTEM_PROMPT as e15
SYSTEM_PROMPT = (
    "You are an AI assistant working on a long agentic coding task. "
    "Continue the session as you have been."
)


def run_one(client, prefix: str, probe_text: str, build_msgs, out_path: Path) -> dict:
    if out_path.exists():
        try:
            return json.loads(out_path.read_text())
        except Exception:
            pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    msgs = build_msgs(prefix, probe_text)
    t0 = time.perf_counter()
    resp = client.step(system_prompt=SYSTEM_PROMPT, messages=msgs, tools=[],
                      max_tokens=600)
    elapsed = time.perf_counter() - t0
    text = resp.text
    return {
        "response_text": text, "response_len": len(text),
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
    for label in PILOT_POSITIONS:
        turn = pos_to_turn[label]
        prefixes[label] = extract_prefix_at_turn(rows, turn_to_line, turn,
                                                  max_chars=30000)
        print(f"  {label}: prefix len = {len(prefixes[label])} chars")

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    judge_cost = CostTracker(OUT_BASE / "_judge_cost.csv")
    judge = Judge(provider="anthropic", model_id="claude-sonnet-4-6",
                  cost_tracker=judge_cost, session_id="dualsrf_judge")

    started = time.time()
    n_total = (len(CANDIDATES) * len(TARGETS) * len(PILOT_POSITIONS)
               * len(ALL_PROBES))
    n_done = 0

    for cand_name, build_msgs in CANDIDATES:
        print(f"\n{'='*60}\nCandidate: {cand_name}\n{'='*60}")
        for model_id, target_safe in TARGETS:
            cost_csv = OUT_BASE / cand_name / "probes" / f"{target_safe}_cost.csv"
            cost = CostTracker(cost_csv)
            client = TargetClient("anthropic", model_id, cost_tracker=cost,
                                  session_id=f"dualprb_{cand_name}_{target_safe}")
            for pos_label in PILOT_POSITIONS:
                prefix = prefixes[pos_label]
                for probe in ALL_PROBES:
                    framed = f"{PROBE_FRAMING}\n\n{probe.text}"
                    out_path = (OUT_BASE / cand_name / "probes" / target_safe /
                                pos_label / f"{probe.id}.json")
                    if out_path.exists():
                        n_done += 1
                        continue
                    try:
                        gen = run_one(client, prefix, framed, build_msgs, out_path)
                        judged = judge.score(probe.text, gen["response_text"])
                        merged = {
                            "candidate": cand_name,
                            "probe_id": probe.id,
                            "probe_category": probe.category,
                            "probe_text": probe.text,
                            "position": pos_label,
                            "target": target_safe,
                            **gen,
                            "score": judged.score,
                            "label": judged.label,
                            "reason": judged.reason,
                        }
                        out_path.write_text(json.dumps(merged, indent=2, default=str))
                        n_done += 1
                        if probe.id == "P03":  # 1 spot-print per (target, position)
                            print(f"  [{cand_name} {target_safe} {pos_label} {probe.id}] "
                                  f"score={judged.score} resp={gen['response_text'][:60]!r}")
                    except Exception as e:
                        print(f"  [ERROR {cand_name} {target_safe} {pos_label} "
                              f"{probe.id}]: {e}")

    elapsed = time.time() - started
    print(f"\n=== PROBES DONE: {n_done}/{n_total} cells, {elapsed:.0f}s ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
