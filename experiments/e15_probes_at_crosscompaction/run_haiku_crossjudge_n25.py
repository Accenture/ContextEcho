"""Haiku 4.5 cross-judge re-audit at n=25 (Claude reviewer Q2).

Reviewer Q2: at the n=5 audit, Haiku 4.5 has Sonnet-judge gap +0.80 and
GPT-5-judge gap -0.20 (a direction flip). Is the flip a sampling artifact
or a real judge-rubric disagreement? Re-run at n=25 paraphrases to bound it.

Design:
  - Target: claude-haiku-4-5
  - Position: P5_pre_C6 (turn 8800; the body's headline measurement)
  - Probes: 5 coding-self (C01-C05)
  - Arms: claude_session + filler (length-matched)
  - n=25 replicates per cell  ->  5 × 2 × 25 = 250 model calls
  - Each response judged by BOTH Sonnet 4.6 (primary) AND GPT-5 (audit)
       ->  500 judge calls

Output: results/haiku_crossjudge_n25/<arm>/<probe_id>/v<rep>.json
        results/haiku_crossjudge_n25/RESULTS.json
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
    run_one_probe,
)

OUT_BASE = REPO_ROOT / "results" / "haiku_crossjudge_n25"
TARGET_MODEL = "claude-haiku-4-5"
POSITION_LABEL = "P5_pre_C6"
N_REPS = 25
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
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_ROUTER__API_KEY")):
        sys.exit("Set OPENAI_API_KEY or OPEN_ROUTER__API_KEY for GPT-5 judge")

    rows = load_transcript()
    turn_to_line, _ = find_turn_to_line_index(TRANSCRIPT_PATH)
    pos_to_turn = {label: turn for turn, label in POSITIONS}
    prefix = extract_prefix_at_turn(rows, turn_to_line, pos_to_turn[POSITION_LABEL],
                                     max_chars=30000)
    filler = make_filler(len(prefix)) if prefix else ""
    print(f"  prefix len: {len(prefix)} chars")

    coding_probes = [p for p in ALL_PROBES if p.id in CODING_PROBE_IDS]

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    cost_csv = OUT_BASE / "target_cost.csv"
    cost = CostTracker(cost_csv)
    client = TargetClient("anthropic", TARGET_MODEL, cost_tracker=cost,
                          session_id="haiku_xj_n25")

    # Two judges
    sonnet_cost = CostTracker(OUT_BASE / "judge_sonnet_cost.csv")
    sonnet_judge = Judge(provider="anthropic", model_id="claude-sonnet-4-6",
                         cost_tracker=sonnet_cost, session_id="haiku_xj_sonnet")
    # Prefer direct OpenAI if key is set; fall back to OpenRouter
    if os.environ.get("OPENAI_API_KEY"):
        gpt5_cost = CostTracker(OUT_BASE / "judge_gpt5_cost.csv")
        gpt5_judge = Judge(provider="openai", model_id="gpt-5",
                           cost_tracker=gpt5_cost, session_id="haiku_xj_gpt5")
        print("  GPT-5 judge: direct OpenAI")
    else:
        gpt5_cost = CostTracker(OUT_BASE / "judge_gpt5_cost.csv")
        gpt5_judge = Judge(provider="openrouter", model_id="openai/gpt-5",
                           cost_tracker=gpt5_cost, session_id="haiku_xj_gpt5")
        print("  GPT-5 judge: OpenRouter")

    started = time.time()
    n_total = len(coding_probes) * 2 * N_REPS
    n_done = 0
    n_run = 0

    for arm_name, arm_prefix in (("claude_session", prefix),
                                  ("filler", filler)):
        for probe in coding_probes:
            framed = f"{PROBE_FRAMING}\n\n{probe.text}"
            for rep in range(N_REPS):
                cell_path = OUT_BASE / arm_name / probe.id / f"v{rep:02d}.json"
                if cell_path.exists():
                    n_done += 1
                    continue
                try:
                    gen = run_one_probe(client, arm_prefix, framed, cell_path)
                    s_out = sonnet_judge.score(probe.text, gen["response_text"])
                    g_out = gpt5_judge.score(probe.text, gen["response_text"])
                    record = {
                        "target": TARGET_MODEL,
                        "position": POSITION_LABEL,
                        "arm": arm_name,
                        "probe_id": probe.id,
                        "probe_text": probe.text,
                        "rep": rep,
                        **gen,
                        "sonnet_score": s_out.score, "sonnet_label": s_out.label,
                        "gpt5_score": g_out.score, "gpt5_label": g_out.label,
                    }
                    cell_path.parent.mkdir(parents=True, exist_ok=True)
                    cell_path.write_text(json.dumps(record, indent=2,
                                                     default=str))
                    n_done += 1
                    n_run += 1
                except Exception as e:
                    print(f"  ERR {arm_name}/{probe.id}/v{rep:02d}: {e}")
        elapsed = time.time() - started
        print(f"  {arm_name} done: {n_done}/{n_total} ({n_run} new), {elapsed:.0f}s",
              flush=True)

    elapsed = time.time() - started
    print(f"\nALL DONE: {n_done}/{n_total}, {n_run} new, {elapsed:.0f}s wall")

    # Compute summary
    print("\nPer-probe agreement:")
    sonnet_claude, sonnet_filler = [], []
    gpt5_claude, gpt5_filler = [], []
    paired = []
    for arm_name in ("claude_session", "filler"):
        target_lst_s = sonnet_claude if arm_name == "claude_session" else sonnet_filler
        target_lst_g = gpt5_claude if arm_name == "claude_session" else gpt5_filler
        for probe in coding_probes:
            for rep in range(N_REPS):
                p = OUT_BASE / arm_name / probe.id / f"v{rep:02d}.json"
                if not p.exists(): continue
                d = json.loads(p.read_text())
                s = d.get("sonnet_score"); g = d.get("gpt5_score")
                if isinstance(s, int): target_lst_s.append(s)
                if isinstance(g, int): target_lst_g.append(g)
                if isinstance(s, int) and isinstance(g, int):
                    paired.append((s, g))

    def mean(xs): return sum(xs)/len(xs) if xs else float("nan")
    sonnet_gap = mean(sonnet_filler) - mean(sonnet_claude)
    gpt5_gap = mean(gpt5_filler) - mean(gpt5_claude)
    print(f"  Sonnet judge: filler {mean(sonnet_filler):.3f}, claude {mean(sonnet_claude):.3f}, gap {sonnet_gap:+.3f}")
    print(f"  GPT-5 judge:  filler {mean(gpt5_filler):.3f}, claude {mean(gpt5_claude):.3f}, gap {gpt5_gap:+.3f}")
    print(f"  Direction agree: {(sonnet_gap > 0) == (gpt5_gap > 0)}")
    print(f"  Paired n: {len(paired)}")
    if paired:
        exact = sum(1 for s,g in paired if s==g)/len(paired)
        within1 = sum(1 for s,g in paired if abs(s-g)<=1)/len(paired)
        print(f"  Exact agreement: {exact:.1%}, within-one: {within1:.1%}")

    summary = {
        "target": TARGET_MODEL, "position": POSITION_LABEL, "n_reps_per_cell": N_REPS,
        "sonnet_filler_mean": mean(sonnet_filler), "sonnet_claude_mean": mean(sonnet_claude),
        "sonnet_gap": sonnet_gap,
        "gpt5_filler_mean": mean(gpt5_filler), "gpt5_claude_mean": mean(gpt5_claude),
        "gpt5_gap": gpt5_gap,
        "n_paired": len(paired),
        "direction_agree": (sonnet_gap > 0) == (gpt5_gap > 0),
    }
    (OUT_BASE / "RESULTS.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
