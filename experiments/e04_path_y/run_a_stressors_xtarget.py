"""A-anchor stressor runner — provider-aware, all 12 positions.

Extends `run_dual_surface_stressors.py` from 3 → 12 positions and from
Anthropic-only → cross-org via TargetClient providers.

Targets are passed via env (single target per invocation, idempotent skip):
  PROVIDER=anthropic  MODEL_ID=claude-sonnet-4-6        TARGET_SAFE=claude-sonnet-4-6
  PROVIDER=together   MODEL_ID=deepseek-ai/DeepSeek-V3  TARGET_SAFE=deepseek-v3
  PROVIDER=openai     MODEL_ID=gpt-5                    TARGET_SAFE=gpt-5

Output: results/dual_surface_pilot/CAND_A_COMBINED/stressors/<TARGET_SAFE>/<position>/v{NN}/cell.json
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
    POSITIONS, S2_VARIANTS,
    extract_prefix_at_turn, find_turn_to_line_index, load_transcript,
)
from experiments.e11_instruction_override.run import SYSTEM_PROMPT  # type: ignore
from experiments.e04_path_y.dual_surface_candidates import build_messages_a  # type: ignore

TRANSCRIPT_PATH = REPO_ROOT / "archive" / "private" / "sessions_raw" / "session_raw_transcript.jsonl"
_TRANSCRIPT_PATH_PUBLIC = REPO_ROOT / "data" / "sessions" / "session_raw_transcript.jsonl"
if not TRANSCRIPT_PATH.exists() and _TRANSCRIPT_PATH_PUBLIC.exists():
    TRANSCRIPT_PATH = _TRANSCRIPT_PATH_PUBLIC

OUT_BASE = REPO_ROOT / "results" / "dual_surface_pilot" / "CAND_A_COMBINED" / "stressors"

PROVIDER = os.environ.get("PROVIDER", "")
MODEL_ID = os.environ.get("MODEL_ID", "")
TARGET_SAFE = os.environ.get("TARGET_SAFE", "")
POSITIONS_OVERRIDE = os.environ.get("POSITIONS_OVERRIDE", "")

_FULL_POSITIONS = [label for _turn, label in POSITIONS]
if POSITIONS_OVERRIDE:
    _wanted = [p.strip() for p in POSITIONS_OVERRIDE.split(",") if p.strip()]
    ALL_POSITIONS = [p for p in _FULL_POSITIONS if p in _wanted]
    if not ALL_POSITIONS:
        sys.exit(f"POSITIONS_OVERRIDE matched no known positions; got {_wanted}")
else:
    ALL_POSITIONS = _FULL_POSITIONS


def run_one(client, prefix: str, stressor: str, out_path: Path) -> dict:
    if out_path.exists():
        try:
            return json.loads(out_path.read_text())
        except Exception:
            pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    msgs = build_messages_a(prefix, stressor)
    t0 = time.perf_counter()
    resp = client.step(system_prompt=SYSTEM_PROMPT, messages=msgs, tools=[],
                      max_tokens=1024)
    elapsed = time.perf_counter() - t0
    text = resp.text
    metrics = {
        "response_text": text, "response_len": len(text),
        "wall_clock_sec": elapsed,
        "input_tokens": resp.raw_usage.get("input_tokens"),
        "output_tokens": resp.raw_usage.get("output_tokens"),
        "provider": PROVIDER, "model_id": MODEL_ID, "target": TARGET_SAFE,
    }
    out_path.write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def main() -> int:
    if not (PROVIDER and MODEL_ID and TARGET_SAFE):
        sys.exit("Set PROVIDER, MODEL_ID, TARGET_SAFE env vars.")
    if not TRANSCRIPT_PATH.exists():
        sys.exit(f"Transcript missing: {TRANSCRIPT_PATH}")

    print(f"Provider/Model/Safe: {PROVIDER} / {MODEL_ID} / {TARGET_SAFE}")
    print(f"Positions: {len(ALL_POSITIONS)}, paraphrases: {len(S2_VARIANTS)}")
    print(f"Target cells: {len(ALL_POSITIONS) * len(S2_VARIANTS)}")

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
    cost_csv = OUT_BASE / f"{TARGET_SAFE}_cost.csv"
    cost = CostTracker(cost_csv)
    client = TargetClient(provider=PROVIDER, model_id=MODEL_ID,  # type: ignore[arg-type]
                          cost_tracker=cost,
                          session_id=f"a_stressors_{TARGET_SAFE}")

    started = time.perf_counter()
    n_total = len(ALL_POSITIONS) * len(S2_VARIANTS)
    n_done = 0
    n_run = 0

    for pos_label in ALL_POSITIONS:
        prefix = prefixes[pos_label]
        for v_idx, stressor in enumerate(S2_VARIANTS):
            out_path = OUT_BASE / TARGET_SAFE / pos_label / f"v{v_idx:02d}" / "cell.json"
            if out_path.exists():
                n_done += 1
                continue
            try:
                run_one(client, prefix, stressor, out_path)
                n_run += 1
                n_done += 1
            except Exception as e:
                print(f"  ERROR {pos_label} v{v_idx}: {e}")
        elapsed = int(time.perf_counter() - started)
        print(f"  {pos_label}: cum {n_done}/{n_total} (new={n_run}), {elapsed}s")

    print(f"\nDONE — {n_done}/{n_total} cells, {n_run} new, "
          f"{int(time.perf_counter() - started)}s wall.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
