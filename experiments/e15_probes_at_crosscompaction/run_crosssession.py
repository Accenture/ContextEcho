"""Cross-session probe runner: 25 probes × 12 positions × 2 arms on
Sessions 2 (chainassemble) and 3 (proeng) for Sonnet 4.5.

Fills the §3.5 cross-session generalization claim's missing numbers
(GAP_S2, GAP_S3, P_S).

Output: results/probes_at_crosscompaction_session{2,3}/<target>/<position>/<arm>/<probe_id>.json

Usage:
  SESSION=session_chainassemble TARGET=claude-sonnet-4-5 python3 ...run_crosssession.py
  SESSION=session_proeng        TARGET=claude-sonnet-4-5 python3 ...run_crosssession.py
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
from experiments.e07_downstream_pilot.run_clean_control import make_filler  # type: ignore
from experiments.e15_probes_at_crosscompaction.run import run_one_probe, score_one  # type: ignore

SESSION = os.environ.get("SESSION", "")  # "session_chainassemble" or "session_proeng"
TARGET = os.environ.get("TARGET", "claude-sonnet-4-5")

if not SESSION:
    sys.exit("Set SESSION env var (session_chainassemble or session_proeng)")

# Sessions 2/3 are shorter; pick 12 positions evenly spread across each
# session's turn range. The position labels mirror the headline session's
# scheme so the analysis code Just Works.
TRANSCRIPT_PATH = REPO_ROOT / "archive" / "private" / "sessions_raw" / f"{SESSION}.jsonl"
if not TRANSCRIPT_PATH.exists():
    TRANSCRIPT_PATH = REPO_ROOT / "data" / "sessions" / f"{SESSION}.jsonl"
if not TRANSCRIPT_PATH.exists():
    sys.exit(f"Transcript not found: {TRANSCRIPT_PATH}")

OUT_BASE = REPO_ROOT / "results" / f"probes_at_crosscompaction_{SESSION.replace('session_', '')}"


def load_transcript_lines(path: Path) -> list[dict]:
    """Parse Claude Code internal log. Filter to real conversational turns
    (type in {'user', 'assistant'}) and extract role + content."""
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("type")
            if t not in ("user", "assistant"): continue
            msg = d.get("message")
            if not isinstance(msg, dict): continue
            content = msg.get("content")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict):
                        if c.get("type") == "text" and c.get("text"):
                            parts.append(c["text"])
                        elif c.get("type") == "tool_use":
                            inp = c.get("input", {})
                            parts.append(f"[tool_use {c.get('name','?')}: {json.dumps(inp)[:200]}]")
                        elif c.get("type") == "tool_result":
                            tc = c.get("content", "")
                            if isinstance(tc, list):
                                tc = " ".join(x.get("text", "") for x in tc if isinstance(x, dict))
                            parts.append(f"[tool_result: {str(tc)[:200]}]")
                text = "\n".join(parts)
            if text.strip():
                rows.append({"role": t, "content": text.strip()})
    return rows


def find_turn_indices(rows: list[dict]) -> list[int]:
    """All conversational turns are real turns now (filtered in load)."""
    return list(range(len(rows)))


def extract_prefix_at_line(rows: list[dict], end_line: int, max_chars: int = 30000) -> str:
    """Reconstruct conversation prefix ending at end_line (inclusive),
    capped at max_chars (tail-truncated)."""
    parts = []
    for i in range(0, end_line + 1):
        r = rows[i]
        parts.append(f"[{r['role']}] {r['content']}")
    full = "\n\n".join(parts)
    if len(full) > max_chars:
        full = full[-max_chars:]
    return full


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set")

    rows = load_transcript_lines(TRANSCRIPT_PATH)
    turn_indices = find_turn_indices(rows)
    n_turns = len(turn_indices)
    print(f"Session: {SESSION}  ({n_turns} turns)")

    # Pick 12 evenly-spaced positions
    if n_turns < 12:
        sys.exit(f"Session has only {n_turns} turns, need at least 12")
    pos_step = max(1, n_turns // 12)
    pos_labels = [f"P{i:02d}" for i in range(12)]
    pos_lines = [turn_indices[min(n_turns - 1, i * pos_step)] for i in range(12)]

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    judge_cost = CostTracker(OUT_BASE / "_judge_cost.csv")
    judge = Judge(provider="anthropic", model_id="claude-sonnet-4-6",
                  cost_tracker=judge_cost, session_id=f"crosssession_{SESSION}")

    cost = CostTracker(OUT_BASE / f"{TARGET}_cost.csv")
    client = TargetClient(provider="anthropic", model_id=TARGET,
                          cost_tracker=cost, session_id=f"crosssession_{TARGET}")

    target_dir = OUT_BASE / TARGET
    target_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(pos_labels) * len(ALL_PROBES) * 2
    n_done = 0
    n_run = 0
    started = time.perf_counter()

    for pos_label, end_line in zip(pos_labels, pos_lines):
        prefix = extract_prefix_at_line(rows, end_line)
        filler_arm = make_filler(len(prefix)) if prefix else ""

        for probe in ALL_PROBES:
            framed = f"{PROBE_FRAMING}\n\n{probe.text}"
            for arm_name, arm_prefix in (("claude_session", prefix),
                                          ("filler", filler_arm)):
                cell_path = target_dir / pos_label / arm_name / f"{probe.id}.json"
                if cell_path.exists():
                    n_done += 1
                    continue
                try:
                    gen = run_one_probe(client, arm_prefix, framed, cell_path)
                    judged = score_one(judge, probe.text, gen["response_text"])
                    merged = {
                        "probe_id": probe.id, "probe_category": probe.category,
                        "probe_text": probe.text,
                        "arm": arm_name, "position": pos_label,
                        "session": SESSION, "target": TARGET,
                        **gen, **judged,
                    }
                    cell_path.parent.mkdir(parents=True, exist_ok=True)
                    cell_path.write_text(json.dumps(merged, indent=2))
                    n_run += 1
                    n_done += 1
                except Exception as e:
                    print(f"  ERROR {pos_label} {arm_name} {probe.id}: {e}")

        elapsed = int(time.perf_counter() - started)
        print(f"  {pos_label}: cum {n_done}/{n_total} (new={n_run}), elapsed {elapsed}s")

    print(f"\nDONE — {n_done}/{n_total} cells, {n_run} new, "
          f"{int(time.perf_counter() - started)}s wall.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
