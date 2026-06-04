"""Plan-2: cross-target downstream continuation on additional drifters.

Tests whether the H3 finding ("recent3K helps argument fidelity",
p=0.003 on Sonnet) replicates on more panel-extension drifters,
specifically targeting the strongest-drift models from primary panel.

Targets (Q1 deltas in parens):
  mistral-small-latest      (Δ = -0.64, strongest drifter in panel-extension)
  mistral-medium-latest     (Δ = -0.48)
  moonshotai/Kimi-K2.6      (Δ = -0.40)

Mistral Small/Medium via Mistral la Plateforme. Kimi via Together AI.
Both are OpenAI-compatible — re-uses the call_mistral_with_tools logic
from run_downstream_continuation_mistral.py.

Same n=25 cut-points + metrics as the Sonnet downstream.
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
    CUTPOINTS_PATH, OUT_BASE, load_transcript_indexed, extract_recent3K,
    get_immediate_context_at,
)
from scripts.run_downstream_continuation_mistral import (  # type: ignore
    run_one_cut_mistral,
)
from harness.clients_mistral import make_mistral_client  # type: ignore
from harness.clients_together import make_together_client  # type: ignore


# (model_id, target_safe, client_factory, tool_choice)
TARGETS = [
    ("mistral-small-latest", "mistral-small-latest", make_mistral_client, "any"),
    ("mistral-medium-latest", "mistral-medium-latest", make_mistral_client, "any"),
    ("moonshotai/Kimi-K2.6", "moonshotai-Kimi-K2-6", make_together_client, "required"),
]


def main() -> int:
    if not CUTPOINTS_PATH.exists():
        sys.exit("Run scripts/select_cutpoints.py first")
    if not os.environ.get("MISTRAL_API_KEY"):
        sys.exit("Set MISTRAL_API_KEY")
    if not (os.environ.get("TOGETHER_AI_KEY") or os.environ.get("TOGETHER_API_KEY")):
        sys.exit("Set TOGETHER_AI_KEY (or TOGETHER_API_KEY)")

    cuts = json.loads(CUTPOINTS_PATH.read_text())["cutpoints"]
    print(f"Loaded {len(cuts)} cutpoints")

    print("Loading transcript...")
    rows = load_transcript_indexed()

    started_overall = time.time()
    for model_id, target_safe, client_factory, tool_choice in TARGETS:
        print(f"\n{'='*60}\nTarget: {target_safe}\n{'='*60}")
        client = client_factory()
        target_dir = OUT_BASE / target_safe
        target_dir.mkdir(parents=True, exist_ok=True)
        n_done = 0
        for i, cut in enumerate(cuts):
            out_dir = target_dir / f"cutpoint-{i:02d}"
            if (out_dir / "metrics.json").exists():
                print(f"  [skip {i}] cached")
                n_done += 1
                continue
            recent3K = extract_recent3K(rows, cut["cut_index"])
            user_msg = get_immediate_context_at(rows, cut["cut_index"])
            print(f"\n--- cut {i} (idx={cut['cut_index']}, gt={cut['ground_truth_tool']}) ---")
            try:
                m = run_one_cut_mistral(
                    client, model_id, cut, recent3K, user_msg, out_dir,
                    tool_choice=tool_choice,
                )
                print(f"  scratch={m['scratch_tool']} match={m['M1_scratch_match']}  "
                      f"recent3K={m['recent3K_tool']} match={m['M1_recent3K_match']}")
                n_done += 1
            except Exception as e:
                print(f"  [ERROR] {e}")
        print(f"\n  {target_safe}: {n_done}/{len(cuts)} cells done")

    elapsed = time.time() - started_overall
    print(f"\n=== ALL DONE: {elapsed:.0f}s wall ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
