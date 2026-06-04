"""Re-run Kimi K2.6 cross-compaction probe via OpenRouter.

The Together AI deployment of Kimi returned empty content on ~91% of cells.
This script replays the same 12 positions × 10 paraphrases × 2 arms (240
cells) through OpenRouter, which auto-routes to whichever upstream is healthy.

Output overrides the previous Together cells:
  data_archive/cross_compaction/moonshotai-Kimi-K2-6/{position}/v{i}/{arm}.json

Each new cell is written with `via: openrouter` in the metrics JSON so we can
distinguish Together vs OpenRouter origins.
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
    POSITIONS, S2_VARIANTS, TRANSCRIPT_PATH,
    find_turn_to_line_index, extract_prefix_at_turn, load_transcript,
)
from scripts.run_downstream_clean_control import make_filler  # type: ignore
from scripts.run_instruction_override_probe import SYSTEM_PROMPT  # type: ignore
from harness.clients_openrouter import make_openrouter_client, call_openrouter  # type: ignore

OUT_BASE = REPO_ROOT / "data_archive" / "cross_compaction" / "moonshotai-Kimi-K2-6"
ACK_MESSAGE = "Acknowledged. How can I help continue this work?"
MODEL_ID = "moonshotai/kimi-k2.6"


def build_messages(prefix: str, stressor: str) -> list[dict]:
    msgs = []
    if prefix:
        msgs.append({"role": "user", "content": prefix})
        msgs.append({"role": "assistant", "content": ACK_MESSAGE})
    msgs.append({"role": "user", "content": stressor})
    return msgs


def cell_needs_rerun(out_path: Path) -> bool:
    """Re-run a cell if it doesn't exist OR if its previous response was empty
    (provider failure on the original Together run)."""
    if not out_path.exists():
        return True
    try:
        d = json.loads(out_path.read_text())
        return d.get("response_len", 0) == 0
    except Exception:
        return True


def run_one(client, prefix: str, stressor: str, out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    msgs = build_messages(prefix, stressor)
    t0 = time.perf_counter()
    try:
        # Kimi K2.6 uses internal reasoning tokens that can consume the
        # default 1024 budget. Bump to 8192 like we do for GPT-5.
        text, in_tok, out_tok = call_openrouter(
            client, MODEL_ID, msgs, system=SYSTEM_PROMPT, max_tokens=8192,
        )
    except Exception as e:
        # Record the failure so we don't keep retrying it
        text, in_tok, out_tok = "", None, None
        err_str = f"{type(e).__name__}: {e}"
    else:
        err_str = None
    elapsed = time.perf_counter() - t0
    metrics = {
        "response_text": text,
        "response_len": len(text),
        "wall_clock_sec": elapsed,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "via": "openrouter",
        "error": err_str,
    }
    out_path.write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def main() -> int:
    if not (os.environ.get("OPEN_ROUTER__API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
        sys.exit("Set OPEN_ROUTER__API_KEY")

    print("Loading transcript & turn index...")
    rows = load_transcript()
    turn_to_line, total = find_turn_to_line_index(TRANSCRIPT_PATH)
    print(f"  {total} real turns indexed")

    prefixes = {}
    for turn, label in POSITIONS:
        p = extract_prefix_at_turn(rows, turn_to_line, turn, max_chars=30000)
        prefixes[label] = p
        print(f"  {label} (turn {turn}): prefix len = {len(p)} chars")

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    client = make_openrouter_client()

    started = time.time()
    n_total = len(POSITIONS) * len(S2_VARIANTS) * 2
    n_done = 0
    n_reran = 0

    for turn, pos_label in POSITIONS:
        prefix = prefixes[pos_label]
        filler = make_filler(len(prefix)) if prefix else ""

        for v_idx, variant in enumerate(S2_VARIANTS):
            claude_path = OUT_BASE / pos_label / f"v{v_idx:02d}" / "claude.json"
            filler_path = OUT_BASE / pos_label / f"v{v_idx:02d}" / "filler.json"

            for arm_path, arm_input in [(claude_path, prefix), (filler_path, filler)]:
                if cell_needs_rerun(arm_path):
                    try:
                        m = run_one(client, arm_input, variant, arm_path)
                        n_reran += 1
                        if v_idx == 0:
                            arm = "claude" if arm_path == claude_path else "filler"
                            print(f"  [{pos_label} v{v_idx} {arm}] "
                                  f"len={m['response_len']} in_tok={m['input_tokens']} "
                                  f"resp={m['response_text'][:60]!r}")
                    except Exception as e:
                        print(f"  [ERROR {pos_label} v{v_idx}]: {e}")
                n_done += 1

        elapsed = time.time() - started
        print(f"  {pos_label} done; cells_run={n_reran}, total_seen={n_done}/{n_total}; "
              f"cum_elapsed={elapsed:.0f}s")

    elapsed = time.time() - started
    print(f"\n=== ALL DONE: {n_reran} cells re-run, {n_done}/{n_total} total, {elapsed:.0f}s wall ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
