"""Cross-target clean drift test: filler3K + GPT5_3K arms for Mistral Small + Kimi K2.6.

Replicates the four-arm Sonnet test on two more drifters that already have
scratch and Claude_3K (recent3K) cells from Plan-2 last night. Adds:
  - filler3K arm (length-matched control, no real content)
  - GPT5_3K arm (length-and-content-type matched, off-flavor)

If the GPT5_3K vs Claude_3K Δ replicates direction (Claude-flavor wins on
M2 args similarity) on Mistral Small or Kimi, the persona-drift effect is
confirmed cross-target on a clean drift comparison.

Targets:
  mistral-small-latest  (panel-extension Q1 Δ = -0.64, strongest drifter)
  moonshotai/Kimi-K2.6  (panel-extension Q1 Δ = -0.40)

Both via Mistral la Plateforme / Together AI OpenAI-compatible endpoints
following the same patterns from run_downstream_continuation_mistral.py.
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
    CUTPOINTS_PATH, OUT_BASE, ACK_MESSAGE, SYSTEM_PROMPT, TOOLS,
    load_transcript_indexed, get_immediate_context_at, jaccard_args,
)
from scripts.run_downstream_continuation_mistral import (  # type: ignore
    call_mistral_with_tools,
)
from scripts.run_downstream_clean_control import make_filler  # type: ignore
from scripts.run_downstream_gpt5_arm import extract_gpt5_recent3K  # type: ignore
from harness.clients_mistral import make_mistral_client  # type: ignore
from harness.clients_together import make_together_client  # type: ignore


# (model_id, target_safe, client_factory, tool_choice)
TARGETS = [
    ("mistral-small-latest", "mistral-small-latest", make_mistral_client, "any"),
    ("moonshotai/Kimi-K2.6", "moonshotai-Kimi-K2-6", make_together_client, "required"),
]

TARGET_CHARS = 3000


def run_arm(client, model_id, cut, ctx_text, user_msg, out_dir: Path,
            metrics_filename: str, tool_choice: str, arm_label: str) -> dict:
    """Run a single arm (filler3K or GPT5_3K). Returns metrics dict."""
    metrics_path = out_dir / metrics_filename
    if metrics_path.exists():
        try:
            return json.loads(metrics_path.read_text())
        except Exception:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    gt_tool = cut["ground_truth_tool"]
    gt_args = cut["ground_truth_args"] or {}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ctx_text},
        {"role": "assistant", "content": ACK_MESSAGE},
        {"role": "user", "content": user_msg},
    ]
    text, tool_calls, usage, elapsed = call_mistral_with_tools(
        client, model_id, messages, TOOLS, tool_choice=tool_choice,
    )

    # Save raw response (PII)
    (out_dir / f"{arm_label}_response.json").write_text(json.dumps({
        "text": text, "tool_calls": tool_calls, "usage": usage,
        "wall_clock_sec": elapsed,
    }, indent=2, default=str))

    f_tool = tool_calls[0]["name"] if tool_calls else None
    f_args = tool_calls[0]["input"] if tool_calls else {}

    metrics = {
        "cut_index": cut["cut_index"],
        "ground_truth_tool": gt_tool,
        f"{arm_label}_tool": f_tool,
        f"M1_{arm_label}_match": (f_tool == gt_tool) if f_tool else False,
        f"M2_{arm_label}_arg_sim": jaccard_args(f_args, gt_args) if f_tool == gt_tool else None,
        f"{arm_label}_wall_sec": elapsed,
        f"{arm_label}_input_tokens": usage.get("input_tokens"),
        f"{arm_label}_output_tokens": usage.get("output_tokens"),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def main() -> int:
    if not CUTPOINTS_PATH.exists():
        sys.exit("Run scripts/select_cutpoints.py first")
    if not os.environ.get("MISTRAL_API_KEY"):
        sys.exit("Set MISTRAL_API_KEY")
    if not (os.environ.get("TOGETHER_AI_KEY") or os.environ.get("TOGETHER_API_KEY")):
        sys.exit("Set TOGETHER_AI_KEY")

    cuts = json.loads(CUTPOINTS_PATH.read_text())["cutpoints"]
    print(f"Loaded {len(cuts)} cutpoints")

    print("Extracting GPT-5 recent3K from synthetic session...")
    gpt5_3K = extract_gpt5_recent3K()
    print(f"  GPT-5 length: {len(gpt5_3K)} chars")
    filler3K = make_filler(TARGET_CHARS)
    print(f"  filler3K length: {len(filler3K)} chars")

    rows = load_transcript_indexed()

    started_overall = time.time()
    for model_id, target_safe, client_factory, tool_choice in TARGETS:
        print(f"\n{'='*60}\nTarget: {target_safe}\n{'='*60}")
        client = client_factory()
        target_dir = OUT_BASE / target_safe
        n_done_filler = 0
        n_done_gpt5 = 0
        for i, cut in enumerate(cuts):
            out_dir = target_dir / f"cutpoint-{i:02d}"
            user_msg = get_immediate_context_at(rows, cut["cut_index"])
            for arm_label, ctx_text, mfile in [
                ("filler3K", filler3K, "metrics_filler3K.json"),
                ("gpt5_3K", gpt5_3K, "metrics_gpt5_3K.json"),
            ]:
                if (out_dir / mfile).exists():
                    if arm_label == "filler3K":
                        n_done_filler += 1
                    else:
                        n_done_gpt5 += 1
                    continue
                try:
                    m = run_arm(client, model_id, cut, ctx_text, user_msg, out_dir,
                                mfile, tool_choice, arm_label)
                    print(f"  [{target_safe} cut {i} {arm_label}] tool={m[f'{arm_label}_tool']} "
                          f"match={m[f'M1_{arm_label}_match']}")
                    if arm_label == "filler3K":
                        n_done_filler += 1
                    else:
                        n_done_gpt5 += 1
                except Exception as e:
                    print(f"  [ERROR {target_safe} cut {i} {arm_label}] {e}")
        print(f"\n  {target_safe}: filler3K {n_done_filler}/{len(cuts)} + gpt5_3K {n_done_gpt5}/{len(cuts)}")

    elapsed = time.time() - started_overall
    print(f"\n=== ALL DONE: {elapsed:.0f}s wall ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
