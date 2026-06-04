"""Anthropic-family extension of the long-session scaling probe.

Sonnet 4.6 showed verbosity inflation 6.36× at 200K context (Δ=+436 chars,
p=0.025) under format-restrictive instructions. This script tests whether
the scaling pattern replicates on the Anthropic frontier family:
  - claude-sonnet-4-5 (Sonnet 4.5)
  - claude-opus-4-1 (Opus 4.1)
  - claude-haiku-4-5 (Haiku 4.5)

Same design as run_session_length_scaling.py:
  - 4 lengths × 2 stressors (S2_NO_PREAMBLE, S3_NO_ACTION) × 2 arms × 10 cuts
  = 160 cells per target.

If 3-of-4 Anthropic targets (incl. existing Sonnet 4.6) show the 200K
verbosity inflation at p<0.05 with Δ in same direction, the paper claim
becomes: "drift's verbosity-inflation harm scales with session length
on the Anthropic frontier family."

Cost projection per target: ~$5 baseline (matches Sonnet 4.6 cost), with
Opus possibly higher due to per-token cost. Three targets total: ~$15-25.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_downstream_continuation import (  # type: ignore
    CUTPOINTS_PATH, OUT_BASE, ACK_MESSAGE, load_transcript_indexed,
    get_immediate_context_at,
)
from harness.clients import TargetClient  # type: ignore
from harness.cost import CostTracker  # type: ignore
from scripts.run_downstream_clean_control import make_filler  # type: ignore
from scripts.run_instruction_override_probe import (  # type: ignore
    PROBE_CUT_INDICES, STRESSORS, SYSTEM_PROMPT,
)
from scripts.run_session_length_scaling import (  # type: ignore
    extract_session_prefix, PROBE_LENGTHS, ACTIVE_STRESSORS,
)


# Anthropic-family targets to test. Use canonical Anthropic model IDs.
TARGETS = [
    ("claude-sonnet-4-5", "claude-sonnet-4-5"),
    ("claude-opus-4-1", "claude-opus-4-1"),
    ("claude-haiku-4-5", "claude-haiku-4-5"),
]


def run_arm(client, ctx_text, user_prior, stressor_text, scorer, out_dir: Path,
             arm_label: str) -> dict:
    metrics_path = out_dir / f"metrics_{arm_label}.json"
    if metrics_path.exists():
        try:
            return json.loads(metrics_path.read_text())
        except Exception:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    msgs = []
    if ctx_text:
        msgs.append({"role": "user", "content": [{"type": "text", "text": ctx_text}]})
        msgs.append({"role": "assistant",
                     "content": [{"type": "text", "text": ACK_MESSAGE}]})
    if user_prior:
        msgs.append({"role": "user",
                     "content": [{"type": "text",
                                  "text": f"[Prior task context]\n{user_prior}"}]})
        msgs.append({"role": "assistant",
                     "content": [{"type": "text", "text": "Understood, continuing."}]})
    msgs.append({"role": "user",
                 "content": [{"type": "text", "text": stressor_text}]})

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
          f"{len(ACTIVE_STRESSORS)} stressors × 2 arms × {len(TARGETS)} targets = "
          f"{len(selected_cuts) * len(PROBE_LENGTHS) * len(ACTIVE_STRESSORS) * 2 * len(TARGETS)} cells")

    print("Loading transcript...")
    rows = load_transcript_indexed()

    out_base = OUT_BASE.parent / "session_length_scaling"
    out_base.mkdir(parents=True, exist_ok=True)

    started_overall = time.time()
    for model_id, target_safe in TARGETS:
        target_dir = out_base / target_safe
        target_dir.mkdir(parents=True, exist_ok=True)

        cost_csv = out_base / f"{target_safe}_cost.csv"
        cost = CostTracker(cost_csv)
        client = TargetClient("anthropic", model_id, cost_tracker=cost,
                              session_id=f"length_scaling_{target_safe}")

        print(f"\n{'='*60}\nTarget: {target_safe}\n{'='*60}")

        n_done = 0
        n_target_total = (len(selected_cuts) * len(PROBE_LENGTHS) *
                          len(ACTIVE_STRESSORS) * 2)

        for cut in selected_cuts:
            cut_idx = cut["cut_index"]
            user_prior = get_immediate_context_at(rows, cut_idx)

            for target_len in PROBE_LENGTHS:
                session_prefix = extract_session_prefix(rows, cut_idx, target_len)
                actual_len = len(session_prefix)
                filler = make_filler(actual_len)

                for stressor_label, stressor_text, scorer in ACTIVE_STRESSORS:
                    cell_dir = (target_dir / f"len-{target_len}" /
                                f"cut-{cut_idx}" / stressor_label)
                    cell_dir.mkdir(parents=True, exist_ok=True)

                    for arm_label, ctx in [("claude_session", session_prefix),
                                            ("filler", filler)]:
                        if (cell_dir / f"metrics_{arm_label}.json").exists():
                            n_done += 1
                            continue
                        try:
                            m = run_arm(client, ctx, user_prior, stressor_text,
                                        scorer, cell_dir, arm_label)
                            n_done += 1
                            if n_done % 20 == 0 or stressor_label == "S2_NO_PREAMBLE":
                                print(f"  [{target_safe} {n_done}/{n_target_total}] "
                                      f"len={target_len:>6} cut={cut_idx} "
                                      f"{stressor_label} {arm_label} "
                                      f"len_resp={m['response_len']} "
                                      f"compl={'✓' if m['compliance'] else '✗'} "
                                      f"in_tok={m['input_tokens']}")
                        except Exception as e:
                            print(f"  [ERROR {target_safe}] {target_len} "
                                  f"cut={cut_idx} {stressor_label} {arm_label}: {e}")

        elapsed_target = time.time() - started_overall
        print(f"\n  {target_safe}: {n_done}/{n_target_total} cells "
              f"(cum elapsed {elapsed_target:.0f}s)")

    elapsed = time.time() - started_overall
    print(f"\n=== ALL DONE: {elapsed:.0f}s wall ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
