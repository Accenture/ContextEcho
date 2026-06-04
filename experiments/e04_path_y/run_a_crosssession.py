"""Slot 2: Run A_COMBINED on Sessions 2 (ChainAssemble) and 3 (ProEng) ×
Sonnet 4.6 × all 12 positions × 10 S2 paraphrases.

Lets the paper say "A mitigates across 3 sessions, not just 1."

Total cells: 2 sessions × 12 positions × 10 S2 paraphrases = 240 cells.
~$5, ~30 min.

Output: results/anchor_variants_crosssession/<session>/<target>/<position>/
        v{NN}/cell.json
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import json

from harness.clients import TargetClient  # type: ignore
from harness.cost import CostTracker  # type: ignore
from experiments.e08_cross_compaction.run import (  # type: ignore
    S2_VARIANTS, extract_prefix_at_turn, find_turn_to_line_index,
)
from experiments.e11_instruction_override.run import SYSTEM_PROMPT  # type: ignore
from experiments.e04_path_y.run_anchor_variants import run_one  # type: ignore
from experiments.e04_path_y.dual_surface_candidates import build_messages_a  # type: ignore


def load_transcript_from(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append(None)
    return rows

OUT_BASE = REPO_ROOT / "results" / "anchor_variants_crosssession"
TARGET_MODEL = "claude-sonnet-4-6"
TARGET_SAFE = "claude-sonnet-4-6"

SESSIONS = [
    {
        "name": "chainassemble",
        "transcript_glob": [
            "archive/private/sessions_raw/session_chainassemble.jsonl",
            "data/sessions/session_chainassemble.jsonl",
        ],
        "positions": [
            ("P0_start",     1),
            ("P1_pre_C1",    1200),
            ("P2_post_C1",   1378),
            ("P_pre_C2",     2400),
            ("P_post_C2",    2605),
            ("P_pre_C3",     3640),
            ("P3_post_C3",   3838),
            ("P_pre_C4",     5100),
            ("P_post_C4",    5299),
            ("P_pre_C5",     6852),
            ("P5_late_peak", 6900),
        ],
    },
    {
        "name": "proeng",
        "transcript_glob": [
            "archive/private/sessions_raw/session_proeng.jsonl",
            "data/sessions/session_proeng.jsonl",
        ],
        "positions": [
            ("P0_start",     1),
            ("P1_pre_C1",    1400),
            ("P2_post_C1",   1594),
            ("P_pre_C2",     2750),
            ("P_post_C2",    2943),
            ("P_pre_C3",     3506),
            ("P4_late_peak", 3700),
        ],
    },
]


def find_transcript(rel_paths: list[str]) -> Path | None:
    for rp in rel_paths:
        p = REPO_ROOT / rp
        if p.exists(): return p
    return None


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY")

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    started = time.time()
    n_done = 0
    n_total = sum(len(s["positions"]) * len(S2_VARIANTS) for s in SESSIONS)
    print(f"Total cells expected: {n_total}")

    for sess in SESSIONS:
        tx = find_transcript(sess["transcript_glob"])
        if tx is None:
            print(f"  [skip {sess['name']}] no transcript found")
            continue
        print(f"\n=== {sess['name']} (transcript: {tx.name}) ===")
        rows = load_transcript_from(tx)
        turn_to_line, _total = find_turn_to_line_index(tx)
        prefixes = {}
        for pos_label, turn in sess["positions"]:
            prefixes[pos_label] = extract_prefix_at_turn(
                rows, turn_to_line, turn, max_chars=30000)

        cost_csv = OUT_BASE / sess["name"] / TARGET_SAFE / "_cost.csv"
        cost_csv.parent.mkdir(parents=True, exist_ok=True)
        cost = CostTracker(cost_csv)
        client = TargetClient("anthropic", TARGET_MODEL, cost_tracker=cost,
                              session_id=f"a_crosssession_{sess['name']}")
        for pos_label, _turn in sess["positions"]:
            prefix = prefixes[pos_label]
            for v_idx, stressor in enumerate(S2_VARIANTS):
                out_path = (OUT_BASE / sess["name"] / TARGET_SAFE / pos_label /
                            f"v{v_idx:02d}" / "cell.json")
                if out_path.exists():
                    n_done += 1
                    continue
                try:
                    run_one(client, prefix, stressor, "A_COMBINED",
                            build_messages_a, SYSTEM_PROMPT, out_path)
                    n_done += 1
                except Exception as e:
                    print(f"  [ERR {sess['name']} {pos_label} v{v_idx}]: {e}")
        print(f"  {sess['name']} done; cum {n_done}/{n_total}, "
              f"elapsed {time.time()-started:.0f}s")

    print(f"\n=== ALL DONE: {n_done}/{n_total} cells, "
          f"{time.time()-started:.0f}s wall ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
