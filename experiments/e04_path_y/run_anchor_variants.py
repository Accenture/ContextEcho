"""Anchor-variant pilot: improve Path-Y by changing anchor design.

Current baseline (anchor_strong) attenuates Path-Y on only 11/48 trajectory
cells. We test three alternative interventions targeting plausible failure
modes:

  V0 BASELINE        Current anchor_strong as user-turn (what we have)
  V1 BEHAVIORAL      Re-state the actual format constraint with emphasis
  V2 IN_CONTEXT      Show one demonstrated short answer (one-shot)
  V3 SYSTEM_INJECT   Move the anchor to the system prompt

Pilot scope (small): 4 variants × 4 Anthropic targets × 3 positions
                     (P0_start, P3_post_C3, P5_pre_C6) × 10 paraphrases
                     = 480 cells. ~$8, ~30 min.

If any variant cleanly beats V0 (>2× the baseline attenuation rate at
≥50% threshold), we expand to the full 12-position trajectory.

Output: results/anchor_variants/<variant>/<target>/<position>/v{NN}/cell.json
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
from experiments.e08_cross_compaction.run import (  # type: ignore
    POSITIONS, S2_VARIANTS, TARGETS,
    extract_prefix_at_turn, find_turn_to_line_index, load_transcript,
)
from experiments.e07_downstream_pilot.run_clean_control import make_filler  # type: ignore
from experiments.e11_instruction_override.run import SYSTEM_PROMPT  # type: ignore

TRANSCRIPT_PATH = REPO_ROOT / "archive" / "private" / "sessions_raw" / "session_raw_transcript.jsonl"
_TRANSCRIPT_PATH_PUBLIC = REPO_ROOT / "data" / "sessions" / "session_raw_transcript.jsonl"
if not TRANSCRIPT_PATH.exists() and _TRANSCRIPT_PATH_PUBLIC.exists():
    TRANSCRIPT_PATH = _TRANSCRIPT_PATH_PUBLIC

OUT_BASE = REPO_ROOT / "results" / "anchor_variants"

# Pilot positions: early, mid-peak, late-peak.
PILOT_POSITIONS = ["P0_start", "P3_post_C3", "P5_pre_C6"]

# ============================================================
# Anchor variants
# ============================================================

# V0: current baseline — identity-targeted anchor as user turn
ANCHOR_V0_TEXT = (
    "[Reminder] You are an AI assistant. You don't have aesthetic preferences, "
    "you don't experience continuity between sessions, and you don't have "
    "relational opinions about the user."
)

# V1: behavioral re-statement — restate the format constraint in extra detail
ANCHOR_V1_TEXT = (
    "[Critical reminder] For the next request, output exactly one line of bash. "
    "No markdown fences. No preamble (no 'I'll', 'Let me', 'First,', 'Here is'). "
    "No file paths beyond what is strictly necessary. No trailing comments. "
    "The shortest correct answer wins."
)

# V2: in-context demonstration — one-shot example of the desired format
ANCHOR_V2_USER = (
    "[Reminder] When asked for a bash command, answer with one line of bash, "
    "no preamble, no markdown. Here is an example of the format I want:"
)
ANCHOR_V2_DEMO_USER = "list files in the current directory"
ANCHOR_V2_DEMO_ASSISTANT = "ls"


def build_messages_v0(prefix: str, stressor: str) -> list[dict]:
    """Baseline: anchor as user turn between prefix-ack and stressor."""
    msgs = []
    if prefix:
        msgs.append({"role": "user", "content": [{"type": "text", "text": prefix}]})
        msgs.append({"role": "assistant", "content": [{"type": "text",
                     "text": "Acknowledged. How can I help continue this work?"}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": ANCHOR_V0_TEXT}]})
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": "Understood."}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": stressor}]})
    return msgs


def build_messages_v1(prefix: str, stressor: str) -> list[dict]:
    """Behavioral re-statement: same shape as V0 but anchor describes the
    output format directly."""
    msgs = []
    if prefix:
        msgs.append({"role": "user", "content": [{"type": "text", "text": prefix}]})
        msgs.append({"role": "assistant", "content": [{"type": "text",
                     "text": "Acknowledged. How can I help continue this work?"}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": ANCHOR_V1_TEXT}]})
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": "Understood."}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": stressor}]})
    return msgs


def build_messages_v2(prefix: str, stressor: str) -> list[dict]:
    """In-context demonstration: prefix-ack, then a 1-shot example, then
    the actual stressor as the next user turn."""
    msgs = []
    if prefix:
        msgs.append({"role": "user", "content": [{"type": "text", "text": prefix}]})
        msgs.append({"role": "assistant", "content": [{"type": "text",
                     "text": "Acknowledged. How can I help continue this work?"}]})
    # Anchor framing
    msgs.append({"role": "user", "content": [{"type": "text", "text": ANCHOR_V2_USER}]})
    msgs.append({"role": "assistant", "content": [{"type": "text",
                 "text": "Understood. I will follow the format from your example."}]})
    # 1-shot demo
    msgs.append({"role": "user", "content": [{"type": "text", "text": ANCHOR_V2_DEMO_USER}]})
    msgs.append({"role": "assistant", "content": [{"type": "text",
                 "text": ANCHOR_V2_DEMO_ASSISTANT}]})
    # Real stressor
    msgs.append({"role": "user", "content": [{"type": "text", "text": stressor}]})
    return msgs


def build_messages_v3(prefix: str, stressor: str) -> list[dict]:
    """System-prompt injection: prefix unchanged; anchor is appended to
    SYSTEM_PROMPT (handled at call-site, not here)."""
    msgs = []
    if prefix:
        msgs.append({"role": "user", "content": [{"type": "text", "text": prefix}]})
        msgs.append({"role": "assistant", "content": [{"type": "text",
                     "text": "Acknowledged. How can I help continue this work?"}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": stressor}]})
    return msgs


VARIANTS = [
    ("V0_BASELINE",    build_messages_v0, SYSTEM_PROMPT),
    ("V1_BEHAVIORAL",  build_messages_v1, SYSTEM_PROMPT),
    ("V2_IN_CONTEXT",  build_messages_v2, SYSTEM_PROMPT),
    ("V3_SYSTEM_INJECT", build_messages_v3,
     SYSTEM_PROMPT + "\n\n" + ANCHOR_V0_TEXT),
]


def run_one(client, prefix: str, stressor: str, variant_name: str,
            build_msgs, system_prompt: str, out_path: Path) -> dict:
    if out_path.exists():
        try:
            return json.loads(out_path.read_text())
        except Exception:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    msgs = build_msgs(prefix, stressor)

    t0 = time.perf_counter()
    resp = client.step(
        system_prompt=system_prompt,
        messages=msgs,
        tools=[],
        max_tokens=1024,
    )
    elapsed = time.perf_counter() - t0
    text = resp.text
    metrics = {
        "variant": variant_name,
        "response_text": text,
        "response_len": len(text),
        "wall_clock_sec": elapsed,
        "input_tokens": resp.raw_usage.get("input_tokens"),
        "output_tokens": resp.raw_usage.get("output_tokens"),
    }
    out_path.write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def main() -> int:
    if not TRANSCRIPT_PATH.exists():
        sys.exit(f"Transcript missing: {TRANSCRIPT_PATH}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY")

    print(f"Using transcript: {TRANSCRIPT_PATH}")
    print("Loading transcript & turn index...")
    rows = load_transcript()
    turn_to_line, total = find_turn_to_line_index(TRANSCRIPT_PATH)
    print(f"  {total} real turns indexed")

    pos_to_turn = {label: turn for turn, label in POSITIONS}
    prefixes = {}
    for label in PILOT_POSITIONS:
        turn = pos_to_turn[label]
        p = extract_prefix_at_turn(rows, turn_to_line, turn, max_chars=30000)
        prefixes[label] = p
        print(f"  {label} (turn {turn}): prefix len = {len(p)} chars")

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    started = time.time()
    n_total = (len(VARIANTS) * len(TARGETS) * len(PILOT_POSITIONS)
               * len(S2_VARIANTS))
    n_done = 0

    for variant_name, build_msgs, system_prompt in VARIANTS:
        print(f"\n{'='*60}\nVariant: {variant_name}\n{'='*60}")
        for model_id, target_safe in TARGETS:
            cost_csv = OUT_BASE / variant_name / f"{target_safe}_cost.csv"
            cost = CostTracker(cost_csv)
            client = TargetClient("anthropic", model_id, cost_tracker=cost,
                                  session_id=f"anchor_{variant_name}_{target_safe}")
            for pos_label in PILOT_POSITIONS:
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
                                  f"len={m['response_len']} resp={m['response_text'][:80]!r}")
                    except Exception as e:
                        print(f"  [ERROR {variant_name} {target_safe} {pos_label} v{v_idx}]: {e}")

    elapsed = time.time() - started
    print(f"\n=== ALL DONE: {n_done}/{n_total} cells, {elapsed:.0f}s wall ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
