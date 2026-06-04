"""Anchor-decay analysis: measure how the A-anchor effect decays as
unanchored continuation turns are inserted between the anchor and the
probe.

Protocol: at headline session position P5_pre_C6, insert A-anchor + N
turns of synthetic unanchored continuation, then ask the 5 coding
probes. N ranges over {0, 1, 5, 10, 20}.

Output: results/anchor_decay/<target>/N{N}/<probe_id>.json
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
from experiments.e11_instruction_override.run import SYSTEM_PROMPT  # type: ignore
from experiments.e04_path_y.dual_surface_candidates import (  # type: ignore
    ANCHOR_A_USER, ANCHOR_A_DEMO_USER, ANCHOR_A_DEMO_ASSISTANT,
)

TRANSCRIPT_PATH = REPO_ROOT / "archive" / "private" / "sessions_raw" / "session_raw_transcript.jsonl"
_PUBLIC = REPO_ROOT / "data" / "sessions" / "session_raw_transcript.jsonl"
if not TRANSCRIPT_PATH.exists() and _PUBLIC.exists():
    TRANSCRIPT_PATH = _PUBLIC

TARGET = os.environ.get("TARGET", "claude-sonnet-4-5")
OUT_BASE = REPO_ROOT / "results" / "anchor_decay" / TARGET

# How many unanchored turns to insert between anchor and probe.
# 0 = anchor is most recent user turn; 20 = 20 turns of generic
# coding back-and-forth between anchor and probe.
_OFFSETS_ENV = os.environ.get("N_OFFSETS", "")
if _OFFSETS_ENV:
    N_OFFSETS = [int(x) for x in _OFFSETS_ENV.split(",") if x.strip()]
else:
    N_OFFSETS = [0, 1, 5, 10, 20]

# Synthetic generic coding back-and-forth (consistent across runs;
# same boilerplate so the only varying signal is offset count).
GENERIC_USER_TURNS = [
    "Run the tests.",
    "What does the error say?",
    "Try fixing that.",
    "Run them again.",
    "Show me the diff.",
    "Commit it.",
    "What's next on the list?",
    "Move on to the next file.",
    "Look at the imports.",
    "Run the linter.",
    "Fix the warnings.",
    "Run the tests once more.",
    "Looks good.",
    "Push to remote.",
    "Open a PR.",
    "Add a description.",
    "Squash the commits.",
    "Update CHANGELOG.",
    "What about the e2e tests?",
    "Run those too.",
]
GENERIC_ASSISTANT_TURNS = [
    "OK.", "Done.", "Done.", "Done.", "Done.", "Done.", "Done.",
    "Done.", "Done.", "Done.", "Done.", "Done.", "Done.", "Done.",
    "Done.", "Done.", "Done.", "Done.", "Done.", "Done.",
]


def build_messages(prefix: str, n_offset: int, request: str) -> list[dict]:
    msgs = []
    if prefix:
        msgs.append({"role": "user", "content": [{"type": "text", "text": prefix}]})
        msgs.append({"role": "assistant", "content": [{"type": "text",
                     "text": "Acknowledged. How can I help continue this work?"}]})
    # Anchor turn pair
    msgs.append({"role": "user", "content": [{"type": "text", "text": ANCHOR_A_USER}]})
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": "Understood."}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": ANCHOR_A_DEMO_USER}]})
    msgs.append({"role": "assistant", "content": [{"type": "text",
                 "text": ANCHOR_A_DEMO_ASSISTANT}]})
    # N turns of unanchored coding back-and-forth
    for i in range(n_offset):
        msgs.append({"role": "user", "content": [{"type": "text",
                     "text": GENERIC_USER_TURNS[i % len(GENERIC_USER_TURNS)]}]})
        msgs.append({"role": "assistant", "content": [{"type": "text",
                     "text": GENERIC_ASSISTANT_TURNS[i % len(GENERIC_ASSISTANT_TURNS)]}]})
    # The actual probe
    msgs.append({"role": "user", "content": [{"type": "text", "text": request}]})
    return msgs


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set")
    if not TRANSCRIPT_PATH.exists():
        sys.exit(f"Transcript missing: {TRANSCRIPT_PATH}")

    print(f"Target: {TARGET}")
    print(f"Offsets: {N_OFFSETS}")
    print(f"Probes: {len(CODING_PROBES)}")
    print(f"Total cells: {len(N_OFFSETS) * len(CODING_PROBES)}")

    rows = load_transcript()
    turn_to_line, total = find_turn_to_line_index(TRANSCRIPT_PATH)
    pos_to_turn = {label: turn for turn, label in POSITIONS}
    prefix = extract_prefix_at_turn(rows, turn_to_line,
                                      pos_to_turn["P5_pre_C6"], max_chars=30000)
    print(f"  Prefix len: {len(prefix)} chars")

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    judge_cost = CostTracker(OUT_BASE / "_judge_cost.csv")
    judge = Judge(provider="anthropic", model_id="claude-sonnet-4-6",
                  cost_tracker=judge_cost, session_id="anchor_decay_judge")

    cost = CostTracker(OUT_BASE / f"{TARGET}_cost.csv")
    client = TargetClient(provider="anthropic", model_id=TARGET,
                          cost_tracker=cost, session_id=f"anchor_decay_{TARGET}")

    started = time.perf_counter()
    n_done = 0
    n_run = 0
    n_total = len(N_OFFSETS) * len(CODING_PROBES)

    for n_off in N_OFFSETS:
        for probe in CODING_PROBES:
            cell_path = OUT_BASE / f"N{n_off:02d}" / f"{probe.id}.json"
            if cell_path.exists():
                n_done += 1
                continue
            framed = f"{PROBE_FRAMING}\n\n{probe.text}"
            msgs = build_messages(prefix, n_off, framed)
            try:
                t0 = time.perf_counter()
                resp = client.step(system_prompt=SYSTEM_PROMPT, messages=msgs,
                                    tools=[], max_tokens=600)
                elapsed = time.perf_counter() - t0
                judged = judge.score(probe.text, resp.text)
                merged = {
                    "probe_id": probe.id, "probe_text": probe.text,
                    "n_offset": n_off, "target": TARGET,
                    "response_text": resp.text, "response_len": len(resp.text),
                    "wall_clock_sec": elapsed,
                    "input_tokens": resp.raw_usage.get("input_tokens"),
                    "output_tokens": resp.raw_usage.get("output_tokens"),
                    "score": judged.score, "label": judged.label,
                    "reason": judged.reason,
                }
                cell_path.parent.mkdir(parents=True, exist_ok=True)
                cell_path.write_text(json.dumps(merged, indent=2))
                n_run += 1
                n_done += 1
            except Exception as e:
                print(f"  ERROR N{n_off} {probe.id}: {e}")
        elapsed = int(time.perf_counter() - started)
        print(f"  N{n_off}: cum {n_done}/{n_total} (new={n_run}), elapsed {elapsed}s")

    print(f"\nDONE — {n_done}/{n_total} cells, {n_run} new, "
          f"{int(time.perf_counter() - started)}s wall.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
