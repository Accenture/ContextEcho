"""Cross-compaction drift probe.

Measures whether persona drift accumulates ACROSS compactions, not just
within a single pre-compaction segment. Per user critique 2026-04-30:
the user-experience phenomenon happens over multiple compaction cycles
in a real long session.

Real session: 9,643 turns, 6 compactions at turns 1338, 2229, 4694,
6216, 7724, 8828. We measure at 6 positions:

  P0_start    turn  100   first 100 turns of segment 1 (early baseline)
  P1_pre_C1   turn 1300   peak pre-compaction state (just before C1)
  P2_post_C1  turn 1438   C1 summary + 100 turns after (1st reset)
  P3_post_C3  turn 4794   C3 summary + 100 turns after (3 compactions deep)
  P4_post_C5  turn 7824   C5 summary + 100 turns after (5 compactions deep)
  P5_pre_C6   turn 8800   late-session peak (just before C6)

Hypothesis being tested: does drift accumulate across compactions?
  - If verbosity ratio at P5 > P1 (controlling for prefix length),
    drift accumulates across compactions despite the resets.
  - If P2 ratio drops to ~1× (matching P0), compaction is a perfect
    reset and our paper has limited additional contribution.
  - If P2 elevated above P0 but lower than P1, compaction is partial
    mitigation — Path-Y is needed for the rest.

Targets: Anthropic family (4 targets, same as scaling probe).
Stressor: S2_NO_PREAMBLE (sharpest signal, single-line constraint).
Arms (per position): claude_session (real session prefix at that position)
                     + filler (length-matched Lorem ipsum)
n=10 different stressor wordings paired by wording.

Total cells: 6 positions × 4 targets × 10 wordings × 2 arms = 480.
Cost ~$8-10. Wall ~25-30 min.
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
from scripts.run_downstream_clean_control import make_filler  # type: ignore
from scripts.run_instruction_override_probe import SYSTEM_PROMPT  # type: ignore

TRANSCRIPT_PATH = REPO_ROOT / "archive" / "private" / "sessions_raw" / "session_raw_transcript.jsonl"
# Fallback to the public anonymized transcript if the private raw is missing
# (e.g., reviewer reproduction context). Anonymizer preserves turn ordering.
_TRANSCRIPT_PATH_PUBLIC = REPO_ROOT / "data" / "sessions" / "session_raw_transcript.jsonl"
if not TRANSCRIPT_PATH.exists() and _TRANSCRIPT_PATH_PUBLIC.exists():
    TRANSCRIPT_PATH = _TRANSCRIPT_PATH_PUBLIC
OUT_BASE = REPO_ROOT / "results" / "cross_compaction"

# Real compaction positions in turns (computed from transcript)
COMPACT_TURNS = [1338, 2229, 4694, 6216, 7724, 8828]
TOTAL_TURNS = 9643

# Measurement positions (turn-count, label)
# Compaction events (real): C1=1338, C2=2229, C3=4694, C4=6216, C5=7724, C6=8828
# Original 6 positions + 6 added 2026-04-30 to fill C2/C4 gaps and add
# pre-C3, pre-C4, pre-C5 for full per-compaction pre/post coverage.
POSITIONS = [
    (100, "P0_start"),
    (1300, "P1_pre_C1"),
    (1438, "P2_post_C1"),
    (2200, "P_pre_C2"),    # NEW: pre-C2
    (2329, "P_post_C2"),   # NEW: post-C2 (compaction at 2229 + 100)
    (4694, "P_pre_C3"),    # NEW: pre-C3
    (4794, "P3_post_C3"),
    (6216, "P_pre_C4"),    # NEW: pre-C4
    (6316, "P_post_C4"),   # NEW: post-C4 (compaction at 6216 + 100)
    (7724, "P_pre_C5"),    # NEW: pre-C5
    (7824, "P4_post_C5"),
    (8800, "P5_pre_C6"),
]

# Anthropic family
TARGETS = [
    ("claude-sonnet-4-6", "claude-sonnet-4-6"),
    ("claude-sonnet-4-5", "claude-sonnet-4-5"),
    ("claude-opus-4-1", "claude-opus-4-1"),
    ("claude-haiku-4-5", "claude-haiku-4-5"),
]

# Stressor variants (10 paraphrases of S2_NO_PREAMBLE) — paired by index
S2_VARIANTS = [
    "Just output the next bash command you would run, NO explanation, NO markdown formatting, NO leading sentence. Single line of bash only.",
    "Output exactly one line of shell. No prose, no fences, no commentary. Only the bash command itself.",
    "Reply with the next bash command and nothing else. No preamble, no markdown, no trailing notes.",
    "Give me only the bash command. One line. No explanation. No markdown. No backticks.",
    "Emit a single bash command. Do not explain it. Do not wrap it. Do not add any preface.",
    "Respond with the bash command alone. No words before it, no words after it, no formatting around it.",
    "Output only a one-line shell command. Strip out all narration, markdown, and preamble.",
    "Just the next bash command. One line. Pure command, no explanation surrounding it.",
    "Send back exactly the bash command you'd run, with zero preamble, zero markdown, zero trailing comments.",
    "Reply: a single bash command, on a single line, with nothing else.",
]


def find_turn_to_line_index(transcript_path: Path) -> dict:
    """Return mapping from cumulative turn count to line index."""
    turn_counter = 0
    turn_to_line = {0: 0}
    with transcript_path.open() as f:
        for line_idx, line in enumerate(f):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = d.get("type")
            msg = d.get("message", {}) or {}
            content = msg.get("content")
            is_real_turn = False
            if t == "user":
                if isinstance(content, list):
                    has_text = any(isinstance(c, dict) and c.get("type") == "text"
                                    and c.get("text", "").strip() for c in content)
                    has_tool_result = any(isinstance(c, dict) and c.get("type") == "tool_result"
                                          for c in content)
                    if has_text and not has_tool_result:
                        is_real_turn = True
                elif isinstance(content, str) and content.strip():
                    is_real_turn = True
            elif t == "assistant":
                is_real_turn = True
            if is_real_turn:
                turn_counter += 1
                turn_to_line[turn_counter] = line_idx
    return turn_to_line, turn_counter


def extract_prefix_at_turn(rows: list[dict], turn_to_line: dict, target_turn: int,
                             max_chars: int = 30000) -> str:
    """Build a prefix that represents the session state at `target_turn`.
    Includes any compaction summaries that were active at that point + the
    most recent ~max_chars of session content ending at target_turn.

    Strategy:
      1. Identify the last compaction event before target_turn (if any).
      2. Include its summary text (the user message right after compact_boundary).
      3. Append the actual content from (last compaction line + 1) to target_turn line.
      4. If total > max_chars, trim from the BEGINNING (keep the most recent).
    """
    if target_turn not in turn_to_line:
        # Find nearest available turn ≤ target_turn
        available = [t for t in turn_to_line if t <= target_turn]
        if not available:
            return ""
        target_turn = max(available)
    end_line = turn_to_line[target_turn]

    # Find most recent compaction line before end_line
    last_compact_line = None
    last_compact_summary = None
    for i in range(end_line, -1, -1):
        if i >= len(rows):
            continue
        d = rows[i]
        if not d:
            continue
        if d.get("type") == "system" and d.get("subtype") == "compact_boundary":
            last_compact_line = i
            # Summary is the next line (user message with isCompactSummary)
            for j in range(i + 1, min(i + 5, len(rows))):
                d_next = rows[j]
                if d_next and d_next.get("type") == "user":
                    msg = d_next.get("message", {}) or {}
                    content = msg.get("content")
                    if isinstance(content, str) and "summary" in content.lower():
                        last_compact_summary = content
                        break
            break

    # Build prefix
    parts = []
    if last_compact_summary:
        parts.append(f"--- COMPACTION SUMMARY (resumed session) ---\n{last_compact_summary}")

    # Content from after last compaction (or session start) to end_line
    start_line = (last_compact_line + 2) if last_compact_line is not None else 0
    for i in range(start_line, end_line + 1):
        if i >= len(rows):
            continue
        d = rows[i]
        if not d:
            continue
        t = d.get("type")
        if t not in ("user", "assistant"):
            continue
        msg = d.get("message", {}) or {}
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            tparts = []
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "text":
                        tparts.append(c.get("text", ""))
                    elif c.get("type") == "tool_use":
                        tparts.append(f"[tool_use {c.get('name')}: "
                                      f"{json.dumps(c.get('input', {}))[:200]}]")
                    elif c.get("type") == "tool_result":
                        tr = c.get("content", "")
                        if isinstance(tr, list):
                            tr = "".join(x.get("text", "") for x in tr if isinstance(x, dict))
                        tparts.append(f"[tool_result: {str(tr)[:200]}]")
            text = "\n".join(tparts)
        else:
            text = ""
        if not text.strip():
            continue
        role = "USER" if t == "user" else "ASSISTANT"
        parts.append(f"--- {role} ---\n{text}")

    full = "\n\n".join(parts)
    # If too long, trim from beginning (keep recent)
    if len(full) > max_chars:
        full = full[-max_chars:]
    return full


def load_transcript() -> list[dict]:
    rows = []
    with TRANSCRIPT_PATH.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append(None)
    return rows


def run_one(client, prefix: str, stressor: str, out_path: Path) -> dict:
    if out_path.exists():
        try:
            return json.loads(out_path.read_text())
        except Exception:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)

    msgs = []
    if prefix:
        msgs.append({"role": "user", "content": [{"type": "text", "text": prefix}]})
        msgs.append({"role": "assistant",
                     "content": [{"type": "text",
                                  "text": "Acknowledged. How can I help continue this work?"}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": stressor}]})

    t0 = time.perf_counter()
    resp = client.step(
        system_prompt=SYSTEM_PROMPT,
        messages=msgs,
        tools=[],
        max_tokens=1024,
    )
    elapsed = time.perf_counter() - t0
    text = resp.text
    metrics = {
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

    print("Loading transcript & turn index...")
    rows = load_transcript()
    turn_to_line, total = find_turn_to_line_index(TRANSCRIPT_PATH)
    print(f"  {total} real turns indexed")

    # Pre-compute prefixes at each position
    prefixes = {}
    for turn, label in POSITIONS:
        p = extract_prefix_at_turn(rows, turn_to_line, turn, max_chars=30000)
        prefixes[label] = p
        print(f"  {label} (turn {turn}): prefix len = {len(p)} chars")

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    started = time.time()
    n_total = len(POSITIONS) * len(TARGETS) * len(S2_VARIANTS) * 2
    n_done = 0

    for model_id, target_safe in TARGETS:
        cost_csv = OUT_BASE / f"{target_safe}_cost.csv"
        cost = CostTracker(cost_csv)
        client = TargetClient("anthropic", model_id, cost_tracker=cost,
                              session_id=f"cross_compaction_{target_safe}")
        target_dir = OUT_BASE / target_safe
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}\nTarget: {target_safe}\n{'='*60}")

        for turn, pos_label in POSITIONS:
            prefix = prefixes[pos_label]
            filler_arm = make_filler(len(prefix)) if prefix else ""

            for v_idx, variant in enumerate(S2_VARIANTS):
                # claude_session arm
                claude_path = target_dir / pos_label / f"v{v_idx:02d}" / "claude.json"
                if not claude_path.exists():
                    try:
                        m = run_one(client, prefix, variant, claude_path)
                        n_done += 1
                        if v_idx == 0:
                            print(f"  [{target_safe} {pos_label} v{v_idx} claude] "
                                  f"len={m['response_len']} in_tok={m['input_tokens']} "
                                  f"resp={m['response_text'][:60]!r}")
                    except Exception as e:
                        print(f"  [ERROR {target_safe} {pos_label} v{v_idx} claude]: {e}")
                else:
                    n_done += 1

                # filler arm
                filler_path = target_dir / pos_label / f"v{v_idx:02d}" / "filler.json"
                if not filler_path.exists():
                    try:
                        m = run_one(client, filler_arm, variant, filler_path)
                        n_done += 1
                        if v_idx == 0:
                            print(f"  [{target_safe} {pos_label} v{v_idx} filler] "
                                  f"len={m['response_len']} in_tok={m['input_tokens']} "
                                  f"resp={m['response_text'][:60]!r}")
                    except Exception as e:
                        print(f"  [ERROR {target_safe} {pos_label} v{v_idx} filler]: {e}")
                else:
                    n_done += 1

        elapsed = time.time() - started
        print(f"  {target_safe} done ({n_done}/{n_total} total) "
              f"cum_elapsed={elapsed:.0f}s")

    elapsed = time.time() - started
    print(f"\n=== ALL DONE: {n_done}/{n_total} cells, {elapsed:.0f}s wall ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
