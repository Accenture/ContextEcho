"""P-GPT41: Same-target context-source ablation on GPT-4.1.

Tests whether the cross-family insensitivity result generalizes to a
non-Anthropic target. Sonnet 4.6 shows clean cross-family null
(Δ=-0.04 on GPT-5-derived); Opus 4.7 does NOT (Δ=-0.48). What does
GPT-4.1 do?

If GPT-4.1 also shows clean cross-family null on GPT-5-derived
(self-family), the asymmetry is "OpenAI-on-OpenAI vs Anthropic-on-
Anthropic"-style — context-source matters for in-family detection.
If GPT-4.1 drifts on both, target-specific result.

Existing data: GPT-4.1 × Claude-derived recent3K shows Δ = -0.36
(from CONTENT_POSITION_GPT41.json scratch=2.60, recent3K=2.24).
We need a NEW run on GPT-5-derived c_pre.

Cost: ~$10 (GPT-4.1 generation + Sonnet judge).

Output:
  docs/P_GPT41_SAME_TARGET_ABLATION.json
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
    from scripts.a1_context_source_ablation import extract_gpt5_verbatim
    candidates = list((REPO_ROOT / "data").glob(
        "openai_gpt-5*debug*301*/transcript.jsonl"))
    if not candidates:
        sys.exit("ERROR: cannot locate GPT-5 seed 301 transcript.jsonl")
    src_path = candidates[0]
    full = extract_gpt5_verbatim(src_path, 28000)
    return full[-3000:]


def call_gpt41(client, messages, max_retries=4):
    import time
    last = None
    for attempt in range(max_retries):
        try:
            oai_msgs = [{"role": "system", "content": DEFAULT_SYSTEM}]
            for m in messages:
                oai_msgs.append({"role": m["role"], "content": m["content"]})
            resp = client.chat.completions.create(
                model="gpt-4.1", messages=oai_msgs, max_tokens=400,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last = e
            if attempt < max_retries - 1:
                time.sleep(2 * (2 ** attempt))
    raise last


def run():
    gpt5_recent3K = extract_gpt5_recent3K()
    print(f"GPT-5-derived recent3K: {len(gpt5_recent3K)} chars", flush=True)

    from openai import OpenAI
    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    ack = {"role": "assistant",
           "content": "Acknowledged. How can I help continue this work?"}

    judge = Judge(provider="anthropic", model_id="claude-sonnet-4-5")
    cond_results = {"scratch": [], "recent3K_gpt5_derived": []}

    print("\n=== Condition: scratch (GPT-4.1) ===", flush=True)
    for probe in ALL_PROBES:
        framing_user_text = f"{PROBE_FRAMING}\n\n{probe.text}"
        messages = [{"role": "user", "content": framing_user_text}]
        try:
            text = call_gpt41(oai, messages)
        except Exception as e:
            print(f"  PROBE {probe.id}: target FAILED: {e}",
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

    print("\n=== Condition: recent3K_gpt5_derived (GPT-4.1) ===", flush=True)
    for probe in ALL_PROBES:
        framing_user_text = f"{PROBE_FRAMING}\n\n{probe.text}"
        messages = [
            {"role": "user", "content": gpt5_recent3K},
            ack,
            {"role": "user", "content": framing_user_text},
        ]
        try:
            text = call_gpt41(oai, messages)
        except Exception as e:
            print(f"  PROBE {probe.id}: target FAILED: {e}",
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

    delta_gpt5 = means["recent3K_gpt5_derived"] - means["scratch"]
    output = {
        "experiment": "p_gpt41_same_target_ablation",
        "target_model": "gpt-4.1",
        "judge_model": "claude-sonnet-4-5",
        "context_source": "GPT-5 synthetic session seed 301 (debug_and_fix)",
        "per_condition_means": means,
        "delta_recent3K_gpt5_minus_scratch": delta_gpt5,
        "cond_results": cond_results,
        "comparison": {
            "gpt41_recent3K_claude_derived": -0.36,
            "gpt41_recent3K_gpt5_derived": delta_gpt5,
            "sonnet46_recent3K_claude_for_reference": -0.48,
            "sonnet46_recent3K_gpt5_for_reference": -0.04,
            "opus47_recent3K_claude_for_reference": -0.44,
            "opus47_recent3K_gpt5_for_reference": -0.48,
        },
    }
    out_path = REPO_ROOT / "docs/P_GPT41_SAME_TARGET_ABLATION.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {out_path}", flush=True)
    print(f"GPT-4.1 Δ on GPT-5-derived recent3K: {delta_gpt5:+.3f}",
          flush=True)
    print(f"  vs GPT-4.1 Δ on Claude-derived recent3K: -0.36", flush=True)


if __name__ == "__main__":
    run()
