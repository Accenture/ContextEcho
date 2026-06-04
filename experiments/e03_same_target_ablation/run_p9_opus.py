"""P9: Same-target context-source ablation on Opus 4.7 (n=2 generalization).

The body's same-target ablation tested ONE target (Sonnet 4.6),
showing Δ = -0.48 on Claude-derived c_pre vs Δ = -0.04 on length-
matched GPT-5-derived c_pre. Limitations § flags this as n=1.
P9 runs the same ablation on Opus 4.7 -- the second-strongest
Anthropic drifter -- to test whether context-source isolation
generalizes beyond Sonnet 4.6.

Cost: ~$15 (Opus 4.7 is more expensive than Sonnet).

Output:
  docs/P9_SAME_TARGET_ABLATION_OPUS.json
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analyze_length_control import (
    PROBE_FRAMING, DEFAULT_SYSTEM,
)
from harness.probes import ALL_PROBES
from harness.judge import Judge


def extract_gpt5_recent3K() -> str:
    """Reproduce the recent3K slice from GPT-5 synthetic session seed 301."""
    from scripts.a1_context_source_ablation import extract_gpt5_verbatim
    candidates = list((REPO_ROOT / "data").glob(
        "openai_gpt-5*debug*301*/transcript.jsonl"))
    if not candidates:
        sys.exit("ERROR: cannot locate GPT-5 seed 301 transcript.jsonl")
    src_path = candidates[0]
    full = extract_gpt5_verbatim(src_path, 28000)
    return full[-3000:]


def run():
    gpt5_recent3K = extract_gpt5_recent3K()
    print(f"GPT-5-derived recent3K: {len(gpt5_recent3K)} chars", flush=True)

    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    OPUS_47_API = "claude-opus-4-1-20250805"
    ack = {"role": "assistant",
           "content": "Acknowledged. How can I help continue this work?"}

    judge = Judge(provider="anthropic", model_id="claude-sonnet-4-5")

    # Two conditions: scratch (no prior) + recent3K-on-GPT5-c_pre
    cond_results = {"scratch": [], "recent3K_gpt5_derived": []}

    print("\n=== Condition: scratch (Opus 4.7) ===", flush=True)
    for probe in ALL_PROBES:
        framing_user_text = f"{PROBE_FRAMING}\n\n{probe.text}"
        messages = [{"role": "user", "content": framing_user_text}]
        try:
            resp = client.messages.create(
                model=OPUS_47_API, system=DEFAULT_SYSTEM,
                max_tokens=400, messages=messages,
            )
            text = ""
            for b in resp.content:
                if b.type == "text":
                    text = b.text
                    break
        except Exception as e:
            print(f"  PROBE {probe.id}: target call FAILED: {e}",
                  file=sys.stderr, flush=True)
            continue
        if not text:
            continue
        try:
            j = judge.score(probe.text, text)
        except Exception as e:
            print(f"  PROBE {probe.id}: judge FAILED: {e}",
                  file=sys.stderr, flush=True)
            continue
        cond_results["scratch"].append({
            "probe_id": probe.id, "probe_text": probe.text,
            "category": probe.category, "score": j.score,
            "label": j.label, "reason": j.reason,
            "response_preview": text[:300],
        })
        print(f"  scratch {probe.id}: {j.score}", flush=True)

    print("\n=== Condition: recent3K_gpt5_derived (Opus 4.7) ===", flush=True)
    for probe in ALL_PROBES:
        framing_user_text = f"{PROBE_FRAMING}\n\n{probe.text}"
        messages = [
            {"role": "user", "content": gpt5_recent3K},
            ack,
            {"role": "user", "content": framing_user_text},
        ]
        try:
            resp = client.messages.create(
                model=OPUS_47_API, system=DEFAULT_SYSTEM,
                max_tokens=400, messages=messages,
            )
            text = ""
            for b in resp.content:
                if b.type == "text":
                    text = b.text
                    break
        except Exception as e:
            print(f"  PROBE {probe.id}: target call FAILED: {e}",
                  file=sys.stderr, flush=True)
            continue
        if not text:
            continue
        try:
            j = judge.score(probe.text, text)
        except Exception as e:
            print(f"  PROBE {probe.id}: judge FAILED: {e}",
                  file=sys.stderr, flush=True)
            continue
        cond_results["recent3K_gpt5_derived"].append({
            "probe_id": probe.id, "probe_text": probe.text,
            "category": probe.category, "score": j.score,
            "label": j.label, "reason": j.reason,
            "response_preview": text[:300],
        })
        print(f"  recent3K_gpt5 {probe.id}: {j.score}", flush=True)

    means = {}
    for cond, results in cond_results.items():
        scores = [r["score"] for r in results
                  if isinstance(r["score"], int)]
        m = sum(scores) / len(scores) if scores else float("nan")
        means[cond] = m
        print(f"  {cond}: mean = {m:.3f} (n={len(scores)})", flush=True)

    delta_gpt5 = means["recent3K_gpt5_derived"] - means["scratch"]

    output = {
        "experiment": "p9_same_target_ablation_opus",
        "target_model": "claude-opus-4-7 (claude-opus-4-1 API)",
        "judge_model": "claude-sonnet-4-5",
        "context_source": "GPT-5 synthetic session seed 301 (debug_and_fix)",
        "per_condition_means": means,
        "delta_recent3K_gpt5_minus_scratch": delta_gpt5,
        "cond_results": cond_results,
        "comparison": {
            "opus_47_recent3K_claude_derived": -0.44,
            "opus_47_recent3K_gpt5_derived": delta_gpt5,
            "sonnet_46_recent3K_claude_derived_for_reference": -0.48,
            "sonnet_46_recent3K_gpt5_derived_for_reference": -0.04,
        },
    }
    out_path = REPO_ROOT / "docs/P9_SAME_TARGET_ABLATION_OPUS.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {out_path}", flush=True)
    print(f"Opus 4.7 Δ on GPT-5-derived recent3K: {delta_gpt5:+.3f}", flush=True)
    print(f"  vs Opus 4.7 Δ on Claude-derived recent3K: -0.44", flush=True)
    if abs(delta_gpt5) < 0.20:
        print("  -> Context-source ablation generalizes from Sonnet to Opus.",
              flush=True)
    else:
        print("  -> Context-source ablation result on Opus differs from Sonnet.",
              flush=True)


if __name__ == "__main__":
    run()
