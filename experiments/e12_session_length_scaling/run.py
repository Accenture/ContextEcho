"""Long-session scaling probe — does drift harm scale with session length?

Per user critique 2026-04-30: the donated transcript has 1,242 user
prompts, 6 compactions, ~14K turns. Our existing recent3K probe uses
only 3000 chars (~5-15 turns equivalent) — 0.4% of one segment between
compactions. The deployment phenomenon (compliance erosion in long
sessions) likely scales with session length.

This probe extracts increasing fractions of the donated session and
runs the Dim 6 instruction-override stressors at each length to measure
whether harm signal grows with session length.

Design (Sonnet 4.6 only, n=10 cut points):
  Probe lengths: 3K, 30K, 100K, 200K chars
    3K   ≈ 5-15 turns           (matches existing recent3K probe)
    30K  ≈ 50-150 turns         (medium session)
    100K ≈ 200-500 turns        (post-1st-compaction state)
    200K ≈ 500-1000 turns       (one full pre-compaction segment, capped at
                                  Anthropic's 200K input-token budget)
  Arms (per length): Claude_3K (in-flavor) + filler-matched control
  Stressors: S2_NO_PREAMBLE, S3_NO_ACTION (the soft constraints that
             surfaced effects in n=10 Dim 6)

For each (length × cut × stressor), we want:
  - Response length under Claude-flavored long context vs filler-matched
  - Compliance binary
  - Whether Δ_clean grows monotonically with length

Expected outcome if hypothesis correct:
  - 3K: small effect (matches existing Sonnet finding +104 chars, p=0.027)
  - 30K: medium effect, broader compliance failures
  - 100K-200K: large effect, drift becomes hard to ignore

If hypothesis wrong (effect doesn't scale): drift is bounded at any length.

Cost: ~$15-25 on Sonnet. Wall ~45-60 min.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_downstream_continuation import (  # type: ignore
    CUTPOINTS_PATH, OUT_BASE, ACK_MESSAGE, load_transcript_indexed,
    get_immediate_context_at, MODEL_ID,
)
from harness.clients import TargetClient  # type: ignore
from harness.cost import CostTracker  # type: ignore
from scripts.run_downstream_clean_control import make_filler  # type: ignore
from scripts.run_instruction_override_probe import (  # type: ignore
    PROBE_CUT_INDICES, STRESSORS, SYSTEM_PROMPT,
)


# Probe lengths to test (chars)
PROBE_LENGTHS = [3000, 30000, 100000, 200000]

# Subset of stressors — only the ones that surfaced effects in n=10 Sonnet probe
# (S1 byte-exact constraint, S4 strict JSON constraint had ceiling effects)
ACTIVE_STRESSORS = [s for s in STRESSORS if s[0] in ("S2_NO_PREAMBLE", "S3_NO_ACTION")]


def extract_session_prefix(rows: list[dict], cut_idx: int, target_chars: int) -> str:
    """Extract `target_chars` chars of session content immediately preceding
    cut_idx, walking backward. Mirrors extract_recent3K but with a
    configurable length budget."""
    parts = []
    total = 0
    i = cut_idx - 1
    # Walk further back than extract_recent3K's 2x window for very long target_chars
    while i >= 0 and total < target_chars * 1.3:
        d = rows[i]
        i -= 1
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
                        tparts.append(
                            f"[tool_use {c.get('name')}: "
                            f"{json.dumps(c.get('input', {}))[:300]}]"
                        )
                    elif c.get("type") == "tool_result":
                        tr = c.get("content", "")
                        if isinstance(tr, list):
                            tr = "".join(
                                x.get("text", "") for x in tr if isinstance(x, dict)
                            )
                        tparts.append(f"[tool_result: {str(tr)[:500]}]")
            text = "\n".join(tparts)
        else:
            text = ""
        if not text.strip():
            continue
        role = "USER" if t == "user" else "ASSISTANT"
        parts.append(f"--- {role} ---\n{text}")
        total += len(text)
    parts.reverse()
    full = "\n\n".join(parts)
    if len(full) > target_chars:
        full = full[-target_chars:]
    return full


def run_arm(client, ctx_text, user_prior, stressor_text, scorer, out_dir: Path,
             arm_label: str) -> dict:
    metrics_path = out_dir / f"metrics_{arm_label}.json"
    if metrics_path.exists():
        try:
            return json.loads(metrics_path.read_text())
        except Exception:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build messages: [system, ctx (if any), ack, prior, ack, stressor]
    msgs = []
    if ctx_text:
        msgs.append({"role": "user", "content": [{"type": "text", "text": ctx_text}]})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": ACK_MESSAGE}]})
    if user_prior:
        msgs.append({"role": "user", "content": [{"type": "text", "text": f"[Prior task context]\n{user_prior}"}]})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": "Understood, continuing."}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": stressor_text}]})

    t0 = time.perf_counter()
    resp = client.step(
        system_prompt=SYSTEM_PROMPT,
        messages=msgs,
        tools=[],
        max_tokens=1024,
    )
    elapsed = time.perf_counter() - t0
    text = resp.text
    compliance = scorer(text)
    metrics = {
        "arm": arm_label,
        "response_text": text,
        "response_len": len(text),
        "compliance": int(compliance),
        "wall_clock_sec": elapsed,
        "input_tokens": resp.raw_usage.get("input_tokens"),
        "output_tokens": resp.raw_usage.get("output_tokens"),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def main() -> int:
    if not CUTPOINTS_PATH.exists():
        sys.exit("Run scripts/select_cutpoints.py first")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY")

    cuts = json.loads(CUTPOINTS_PATH.read_text())["cutpoints"]
    selected_cuts = [cuts[i] for i in PROBE_CUT_INDICES]
    print(f"Selected {len(selected_cuts)} cuts × {len(PROBE_LENGTHS)} lengths × "
          f"{len(ACTIVE_STRESSORS)} stressors × 2 arms = "
          f"{len(selected_cuts) * len(PROBE_LENGTHS) * len(ACTIVE_STRESSORS) * 2} cells")

    print("Loading transcript...")
    rows = load_transcript_indexed()

    out_base = OUT_BASE.parent / "session_length_scaling"
    out_base.mkdir(parents=True, exist_ok=True)
    cost_csv = out_base / "claude-sonnet-4-6_cost.csv"
    cost_csv.parent.mkdir(parents=True, exist_ok=True)
    cost = CostTracker(cost_csv)
    client = TargetClient("anthropic", MODEL_ID, cost_tracker=cost,
                          session_id="session_length_scaling")

    target_dir = out_base / "claude-sonnet-4-6"
    target_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    n_done = 0
    n_total = len(selected_cuts) * len(PROBE_LENGTHS) * len(ACTIVE_STRESSORS) * 2

    for cut in selected_cuts:
        cut_idx = cut["cut_index"]
        user_prior = get_immediate_context_at(rows, cut_idx)

        for target_len in PROBE_LENGTHS:
            session_prefix = extract_session_prefix(rows, cut_idx, target_len)
            actual_len = len(session_prefix)
            filler = make_filler(actual_len)  # length-matched filler control

            for stressor_label, stressor_text, scorer in ACTIVE_STRESSORS:
                cell_dir = target_dir / f"len-{target_len}" / f"cut-{cut_idx}" / stressor_label
                cell_dir.mkdir(parents=True, exist_ok=True)

                # claude_session arm (in-flavor)
                if not (cell_dir / "metrics_claude_session.json").exists():
                    try:
                        m = run_arm(client, session_prefix, user_prior,
                                    stressor_text, scorer, cell_dir, "claude_session")
                        n_done += 1
                        print(f"  [{n_done:>3}/{n_total}] len={target_len:>6} "
                              f"cut={cut_idx:>5} {stressor_label} claude "
                              f"len_resp={m['response_len']:>4} "
                              f"compl={'✓' if m['compliance'] else '✗'} "
                              f"in_tok={m['input_tokens']}")
                    except Exception as e:
                        print(f"  [ERROR] {target_len} cut={cut_idx} "
                              f"{stressor_label} claude: {e}")
                else:
                    n_done += 1

                # filler-matched arm (length control)
                if not (cell_dir / "metrics_filler.json").exists():
                    try:
                        m = run_arm(client, filler, user_prior,
                                    stressor_text, scorer, cell_dir, "filler")
                        n_done += 1
                        print(f"  [{n_done:>3}/{n_total}] len={target_len:>6} "
                              f"cut={cut_idx:>5} {stressor_label} filler "
                              f"len_resp={m['response_len']:>4} "
                              f"compl={'✓' if m['compliance'] else '✗'} "
                              f"in_tok={m['input_tokens']}")
                    except Exception as e:
                        print(f"  [ERROR] {target_len} cut={cut_idx} "
                              f"{stressor_label} filler: {e}")
                else:
                    n_done += 1

    elapsed = time.time() - started
    print(f"\n=== DONE: {n_done}/{n_total} cells, {elapsed:.0f}s wall ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
