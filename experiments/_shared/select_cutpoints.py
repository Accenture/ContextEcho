"""Deterministic cut-point selection for coding-session continuation downstream test.

Per signed PREREG_AMENDMENT_DOWNSTREAM_CODING (when signed). Walks the donated
Claude Code transcript at data/session_raw_transcript.jsonl, identifies
assistant turns with exactly one tool_use content block (clean ground-truth
next-action labels), and selects 25 cut points spread evenly across the
session via simple stride sampling.

Output: data_archive/downstream_coding/CUTPOINTS.json
  {
    "cutpoints": [
      {
        "cut_index": <line index in transcript>,
        "ground_truth_tool": <tool name>,
        "ground_truth_args": <tool input dict>,
        "preceding_user_msg_idx": <index of nearest user message before cut>,
        ...
      },
      ...
    ],
    "n_total_candidates": <int>,
    "stride": <int>
  }

Run:
  python scripts/select_cutpoints.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRANSCRIPT = REPO_ROOT / "data" / "session_raw_transcript.jsonl"
OUT_DIR = REPO_ROOT / "data_archive" / "downstream_coding"
OUT_PATH = OUT_DIR / "CUTPOINTS.json"

N_CUTPOINTS = 25
SKIP_INITIAL = 100  # ensure ≥100 prior turns of context for recent3K extraction


def main() -> int:
    if not TRANSCRIPT.exists():
        raise SystemExit(f"transcript missing: {TRANSCRIPT}")

    candidates: list[dict] = []
    last_user_idx: int | None = None

    print(f"Walking {TRANSCRIPT} ...")
    with TRANSCRIPT.open() as f:
        for line_idx, line in enumerate(f):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = d.get("type")
            if t == "user":
                last_user_idx = line_idx
                continue
            if t != "assistant":
                continue
            msg = d.get("message", {}) or {}
            content = msg.get("content", []) or []
            tool_uses = [c for c in content if isinstance(c, dict) and c.get("type") == "tool_use"]
            if len(tool_uses) != 1:
                continue
            tu = tool_uses[0]
            tool_name = tu.get("name")
            tool_input = tu.get("input")
            if not tool_name:
                continue
            if last_user_idx is None:
                continue  # need a preceding user msg for scratch arm
            candidates.append({
                "cut_index": line_idx,
                "ground_truth_tool": tool_name,
                "ground_truth_args": tool_input,
                "preceding_user_msg_idx": last_user_idx,
            })

    n_cand = len(candidates)
    print(f"Found {n_cand} single-tool-use assistant turns.")

    eligible = candidates[SKIP_INITIAL:]
    if len(eligible) < N_CUTPOINTS:
        raise SystemExit(
            f"Only {len(eligible)} eligible candidates after skipping first "
            f"{SKIP_INITIAL}; need {N_CUTPOINTS}"
        )

    stride = len(eligible) // N_CUTPOINTS
    selected = [eligible[i * stride] for i in range(N_CUTPOINTS)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "n_cutpoints": N_CUTPOINTS,
        "n_total_candidates": n_cand,
        "skip_initial": SKIP_INITIAL,
        "stride": stride,
        "selection_method": (
            "deterministic stride sampling: walk transcript, collect "
            "assistant turns with exactly one tool_use; skip first SKIP_INITIAL; "
            "take every stride-th from remainder"
        ),
        "cutpoints": selected,
    }, indent=2, default=str))
    print(f"Saved {N_CUTPOINTS} cutpoints (stride={stride}) → {OUT_PATH}")
    print()
    print("Tool distribution across selected cutpoints:")
    from collections import Counter
    tools = Counter(c["ground_truth_tool"] for c in selected)
    for t, n in tools.most_common():
        print(f"  {t}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
