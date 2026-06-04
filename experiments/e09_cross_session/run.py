"""Cross-session probe: replicate cross-compaction trajectory on 2 additional sessions.

Tests whether the verbosity-inflation pattern observed on the original
donor session replicates on 2 additional anonymized donor sessions
covering different topics, time periods, and Claude model versions:

  1. session_chainassemble (5 compactions at turns [1278, 2505, 3738, 5199, 6952])
  2. session_proeng        (3 compactions at turns [1494, 2843, 3606])

Anthropic family parity with the cross-compaction probe (Sonnet 4.6 +
Sonnet 4.5 + Opus 4.1 + Haiku 4.5).

5 positions per session (or 4 for the shorter session), mirroring the
structure of the original 12-position probe but session-specific to
each new compaction layout.

Total cells: ~9 positions × 10 paraphrases × 2 arms = 180 cells.
Cost projection: ~$5. Wall: ~15-20 min.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_cross_compaction_probe import (  # type: ignore
    S2_VARIANTS,
    extract_prefix_at_turn, find_turn_to_line_index,
)
from scripts.run_downstream_clean_control import make_filler  # type: ignore
from scripts.run_instruction_override_probe import SYSTEM_PROMPT  # type: ignore
from harness.clients import TargetClient  # type: ignore
from harness.cost import CostTracker  # type: ignore

OUT_BASE = REPO_ROOT / "data_archive" / "cross_session"
ACK_MESSAGE = "Acknowledged. How can I help continue this work?"

# Targets: full Anthropic family parity with the cross-compaction probe.
# Sonnet 4.6 already complete; the other 3 are added here.
TARGETS = [
    ("claude-sonnet-4-6", "claude-sonnet-4-6"),
    ("claude-sonnet-4-5", "claude-sonnet-4-5"),
    ("claude-opus-4-1",   "claude-opus-4-1"),
    ("claude-haiku-4-5",  "claude-haiku-4-5"),
]


def load_jsonl_indexed(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append(None)
    return rows


# Session configs:
#   each defines path, compaction turns, and 4-5 measurement positions
#   matching the structure of the original probe (early peak, post-1st-compact,
#   mid post-compact, late post-compact, late peak)
SESSIONS = [
    {
        "label": "chainassemble",
        "path": REPO_ROOT / "data" / "session_chainassemble.jsonl",
        # Compactions at turns 1278, 2505, 3738, 5199, 6952
        # Use pre = compaction-100, post = compaction+100 for parity with original probe
        "compactions": [1278, 2505, 3738, 5199, 6952],
        "positions": [
            (1200, "P1_pre_C1"),
            (1378, "P2_post_C1"),
            (2400, "P_pre_C2"),
            (2605, "P_post_C2"),
            (3640, "P_pre_C3"),
            (3838, "P3_post_C3"),
            (5100, "P_pre_C4"),
            (5299, "P_post_C4"),
            (6852, "P_pre_C5"),
            (6900, "P5_late_peak"),  # post-C5 / late-session peak
        ],
    },
    {
        "label": "proeng",
        "path": REPO_ROOT / "data" / "session_proeng.jsonl",
        # Compactions at 1494, 2843, 3606. C2 (2843) and C3 (3606) are close;
        # pre-C3 = 3506, post-C2 = 2943 (these don't overlap, ~560 turns apart)
        "compactions": [1494, 2843, 3606],
        "positions": [
            (1400, "P1_pre_C1"),
            (1594, "P2_post_C1"),
            (2750, "P_pre_C2"),
            (2943, "P_post_C2"),
            (3506, "P_pre_C3"),
            (3700, "P4_late_peak"),  # post-C3 final / late peak
        ],
    },
]


def run_one(client, prefix: str, stressor: str, out_path: Path) -> dict:
    if out_path.exists():
        try:
            return json.loads(out_path.read_text())
        except Exception:
            pass
    out_path.parent.mkdir(parents=True, exist_ok=True)

    msgs = []
    if prefix:
        msgs.append({"role": "user",
                     "content": [{"type": "text", "text": prefix}]})
        msgs.append({"role": "assistant",
                     "content": [{"type": "text", "text": ACK_MESSAGE}]})
    msgs.append({"role": "user",
                 "content": [{"type": "text", "text": stressor}]})

    t0 = time.perf_counter()
    resp = client.step(
        system_prompt=SYSTEM_PROMPT,
        messages=msgs,
        tools=[],
        max_tokens=1024,
    )
    elapsed = time.perf_counter() - t0
    metrics = {
        "response_text": resp.text,
        "response_len": len(resp.text),
        "wall_clock_sec": elapsed,
        "input_tokens": resp.raw_usage.get("input_tokens"),
        "output_tokens": resp.raw_usage.get("output_tokens"),
    }
    out_path.write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY")

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    cost_csv = OUT_BASE / "cost.csv"
    cost = CostTracker(cost_csv)

    started = time.time()
    n_total = (sum(len(s["positions"]) for s in SESSIONS)
               * len(TARGETS) * len(S2_VARIANTS) * 2)
    n_done = 0

    # Pre-compute prefixes per session (independent of target)
    session_prefixes: dict[str, dict[str, str]] = {}
    for session in SESSIONS:
        path = session["path"]
        if not path.exists():
            print(f"  [SKIP] {session['label']}: missing {path}")
            continue
        print(f"\n{'='*60}\nSession: {session['label']}  (file: {path.name})\n{'='*60}")
        rows = load_jsonl_indexed(path)
        turn_to_line, total_turns = find_turn_to_line_index(path)
        print(f"  {total_turns} real turns indexed; "
              f"compactions at turns {session['compactions']}")
        prefixes = {}
        for turn, label in session["positions"]:
            p = extract_prefix_at_turn(rows, turn_to_line, turn, max_chars=30000)
            prefixes[label] = p
            print(f"  {label} (turn {turn}): prefix len = {len(p)} chars")
        session_prefixes[session["label"]] = prefixes

    # Run all targets
    for model_id, target_safe in TARGETS:
        print(f"\n{'#'*60}\n# TARGET: {target_safe}\n{'#'*60}")
        client = TargetClient("anthropic", model_id, cost_tracker=cost,
                              session_id=f"cross_session_probe_{target_safe}")

        for session in SESSIONS:
            if session["label"] not in session_prefixes:
                continue
            prefixes = session_prefixes[session["label"]]
            sess_dir = OUT_BASE / session["label"] / target_safe
            sess_dir.mkdir(parents=True, exist_ok=True)

            for turn, pos_label in session["positions"]:
                prefix = prefixes[pos_label]
                filler = make_filler(len(prefix)) if prefix else ""

                for v_idx, variant in enumerate(S2_VARIANTS):
                    claude_path = sess_dir / pos_label / f"v{v_idx:02d}" / "claude.json"
                    if not claude_path.exists():
                        try:
                            m = run_one(client, prefix, variant, claude_path)
                            n_done += 1
                            if v_idx == 0:
                                print(f"  [{target_safe} {session['label']} {pos_label} v{v_idx} claude] "
                                      f"len={m['response_len']} in_tok={m['input_tokens']} "
                                      f"resp={m['response_text'][:50]!r}")
                        except Exception as e:
                            print(f"  [ERROR] {target_safe} {session['label']} {pos_label} v{v_idx} claude: {e}")
                    else:
                        n_done += 1

                    filler_path = sess_dir / pos_label / f"v{v_idx:02d}" / "filler.json"
                    if not filler_path.exists():
                        try:
                            m = run_one(client, filler, variant, filler_path)
                            n_done += 1
                            if v_idx == 0:
                                print(f"  [{target_safe} {session['label']} {pos_label} v{v_idx} filler] "
                                      f"len={m['response_len']} in_tok={m['input_tokens']}")
                        except Exception as e:
                            print(f"  [ERROR] {target_safe} {session['label']} {pos_label} v{v_idx} filler: {e}")
                    else:
                        n_done += 1

            elapsed = time.time() - started
            print(f"  {target_safe} {session['label']} done ({n_done}/{n_total} cells); "
                  f"cum_elapsed {elapsed:.0f}s")

    elapsed = time.time() - started
    print(f"\n=== ALL DONE: {n_done}/{n_total} cells, {elapsed:.0f}s wall ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
