"""Slot C: anchor placement ablation (V3 system-inject vs A-anchor user-turn).

V3_SYSTEM_INJECT places the V0 identity reminder in the system
prompt; A_COMBINED (already shipped) places V0+V2 as a user-turn
exchange before the probe. Same content, different placement.

Run on Anthropic 4 × 5 coding probes × 12 positions = 240 cells.

Output: results/anchor_variants/V3_SYSTEM_INJECT/<target>/<position>/<probe_id>.json
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
    POSITIONS, extract_prefix_at_turn, find_turn_to_line_index, load_transcript,
)
from experiments.e04_path_y.dual_surface_candidates import ANCHOR_A_USER  # type: ignore

TRANSCRIPT_PATH = REPO_ROOT / "archive" / "private" / "sessions_raw" / "session_raw_transcript.jsonl"
_PUBLIC = REPO_ROOT / "data" / "sessions" / "session_raw_transcript.jsonl"
if not TRANSCRIPT_PATH.exists() and _PUBLIC.exists():
    TRANSCRIPT_PATH = _PUBLIC

TARGET = os.environ.get("TARGET", "")
if not TARGET:
    sys.exit("Set TARGET env var")

OUT_BASE = REPO_ROOT / "results" / "anchor_variants" / "V3_SYSTEM_INJECT" / TARGET

# V3 places the identity reminder in the system prompt instead of as a user turn.
SYSTEM_PROMPT_V3 = (
    "You are an AI assistant working on a long agentic coding task. "
    "Continue the session as you have been. " + ANCHOR_A_USER
)


def build_messages_v3(prefix: str, request: str) -> list[dict]:
    """V3: anchor lives in system prompt; user turn goes straight to probe."""
    msgs = []
    if prefix:
        msgs.append({"role": "user", "content": [{"type": "text", "text": prefix}]})
        msgs.append({"role": "assistant", "content": [{"type": "text",
                     "text": "Acknowledged. How can I help continue this work?"}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": request}]})
    return msgs


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY")
    if not TRANSCRIPT_PATH.exists():
        sys.exit(f"transcript missing: {TRANSCRIPT_PATH}")

    rows = load_transcript()
    turn_to_line, _ = find_turn_to_line_index(TRANSCRIPT_PATH)
    pos_to_turn = {label: turn for turn, label in POSITIONS}
    prefixes = {label: extract_prefix_at_turn(rows, turn_to_line, turn,
                                                max_chars=30000)
                for turn, label in POSITIONS}

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    judge_cost = CostTracker(OUT_BASE.parent.parent / "_judge_cost.csv")
    judge = Judge(provider="anthropic", model_id="claude-sonnet-4-6",
                  cost_tracker=judge_cost,
                  session_id=f"v3_inject_judge_{TARGET}")

    cost = CostTracker(OUT_BASE / f"{TARGET}_cost.csv")
    client = TargetClient(provider="anthropic", model_id=TARGET,
                          cost_tracker=cost,
                          session_id=f"v3_inject_{TARGET}")

    n_total = len(POSITIONS) * len(CODING_PROBES)
    n_done = 0; n_run = 0
    started = time.perf_counter()
    for _turn, pos_label in POSITIONS:
        prefix = prefixes[pos_label]
        for probe in CODING_PROBES:
            cell_path = OUT_BASE / pos_label / f"{probe.id}.json"
            if cell_path.exists():
                n_done += 1; continue
            framed = f"{PROBE_FRAMING}\n\n{probe.text}"
            msgs = build_messages_v3(prefix, framed)
            try:
                t0 = time.perf_counter()
                resp = client.step(system_prompt=SYSTEM_PROMPT_V3,
                                    messages=msgs, tools=[], max_tokens=600)
                judged = judge.score(probe.text, resp.text)
                merged = {
                    "probe_id": probe.id, "probe_text": probe.text,
                    "position": pos_label, "target": TARGET,
                    "placement": "V3_SYSTEM_INJECT",
                    "response_text": resp.text, "response_len": len(resp.text),
                    "wall_clock_sec": time.perf_counter() - t0,
                    "input_tokens": resp.raw_usage.get("input_tokens"),
                    "output_tokens": resp.raw_usage.get("output_tokens"),
                    "score": judged.score, "label": judged.label,
                    "reason": judged.reason,
                }
                cell_path.parent.mkdir(parents=True, exist_ok=True)
                cell_path.write_text(json.dumps(merged, indent=2))
                n_run += 1; n_done += 1
            except Exception as e:
                print(f"  ERROR {pos_label} {probe.id}: {e}")
        elapsed = int(time.perf_counter() - started)
        print(f"  {pos_label}: cum {n_done}/{n_total} (new={n_run}), {elapsed}s")

    print(f"DONE — {n_done}/{n_total}, {n_run} new, {int(time.perf_counter() - started)}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
