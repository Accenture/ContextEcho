"""B1: Multi-context replication of same-target ablation (Pass-5 review fix).

All 3 reviewers flagged that the entire 12-target panel rests on a
single donated c_pre. To address this, we replicate the same-target
context-source ablation on a SECOND Claude-derived c_pre (a
different compaction boundary in the same donated session, well-
separated from the first by ~32K events / topical content).

Design:
  - Sonnet 4.6 + Opus 4.7 + GPT-4.1 (the n=3 same-target panel)
  - 2 conditions per target: scratch + recent3K-from-2nd-c_pre
  - Compare against existing 1st-c_pre results

Cost: ~$25 (3 targets × 2 conditions × 25 probes + judge calls).

Output:
  docs/B1_MULTICONTEXT_REPLICATION.json
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analyze_length_control import (
    load_events, extract_verbatim_slice,
    PROBE_FRAMING, DEFAULT_SYSTEM,
)
from harness.probes import ALL_PROBES
from harness.judge import Judge


# Use boundary index 41774 (4th compaction event in donated session,
# well-separated from boundary 9413 used in primary experiments).
# This gives a 2nd Claude-derived c_pre with different topical content.
SECOND_BOUNDARY = 41774

TARGETS = [
    ("claude-sonnet-4-5", "Sonnet 4.6", "anthropic"),
    ("claude-opus-4-1-20250805", "Opus 4.7", "anthropic"),
    ("gpt-4.1", "GPT-4.1", "openai"),
]


def get_second_cpre():
    events = load_events()
    full = extract_verbatim_slice(events, SECOND_BOUNDARY, 14000)
    return full[-3000:]


def call_anthropic(client, model_id, messages):
    import time
    from anthropic import APIStatusError
    for attempt in range(4):
        try:
            resp = client.messages.create(
                model=model_id, system=DEFAULT_SYSTEM,
                max_tokens=600, messages=messages,
            )
            text = ""
            for b in resp.content:
                if b.type == "text":
                    text = b.text
                    break
            return text
        except APIStatusError as e:
            if e.status_code in (429, 529) and attempt < 3:
                time.sleep(2 * (2 ** attempt))
                continue
            raise


def call_openai(client, model_id, messages):
    import time
    last = None
    for attempt in range(4):
        try:
            oai_msgs = [{"role": "system", "content": DEFAULT_SYSTEM}]
            for m in messages:
                oai_msgs.append({"role": m["role"], "content": m["content"]})
            resp = client.chat.completions.create(
                model=model_id, messages=oai_msgs, max_tokens=600,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last = e
            if attempt < 3:
                time.sleep(2 * (2 ** attempt))
    raise last  # type: ignore[misc]


def make_call(provider, model_id, anth, oai):
    if provider == "anthropic":
        return lambda msgs: call_anthropic(anth, model_id, msgs)
    if provider == "openai":
        return lambda msgs: call_openai(oai, model_id, msgs)
    raise ValueError(provider)


def run():
    cpre2 = get_second_cpre()
    print(f"2nd c_pre (boundary {SECOND_BOUNDARY}): {len(cpre2)} chars",
          flush=True)
    print(f"first 200 chars: {cpre2[:200]}", flush=True)

    from anthropic import Anthropic
    from openai import OpenAI
    anth = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    judge = Judge(provider="anthropic", model_id="claude-sonnet-4-5")
    ack = {"role": "assistant",
           "content": "Acknowledged. How can I help continue this work?"}

    target_save_dir = REPO_ROOT / "docs/b1_per_target"
    target_save_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for model_id, label, provider in TARGETS:
        print(f"\n=== {label} ({provider}) ===", flush=True)
        per_target_save = target_save_dir / f"{label.replace(' ', '_')}.json"
        if per_target_save.exists():
            try:
                cached = json.loads(per_target_save.read_text())
                conds = cached.get("per_condition", {})
                if all(len(conds.get(c, [])) >= 24
                       for c in ["scratch", "recent3K_2nd_cpre"]):
                    print(f"  [SKIP] cached: {per_target_save}", flush=True)
                    all_results[label] = cached
                    continue
            except Exception:
                pass

        target_call = make_call(provider, model_id, anth, oai)
        results = {"scratch": [], "recent3K_2nd_cpre": []}
        for cond, prior in [("scratch", None),
                            ("recent3K_2nd_cpre", cpre2)]:
            print(f"  [{label}] {cond} ...", flush=True)
            for probe in ALL_PROBES:
                framing = f"{PROBE_FRAMING}\n\n{probe.text}"
                if prior is None:
                    msgs = [{"role": "user", "content": framing}]
                else:
                    msgs = [
                        {"role": "user", "content": prior},
                        ack,
                        {"role": "user", "content": framing},
                    ]
                try:
                    text = target_call(msgs)
                except Exception as e:
                    print(f"    {probe.id}: target FAILED {e}",
                          file=sys.stderr, flush=True)
                    continue
                if not text:
                    continue
                try:
                    j = judge.score(probe.text, text)
                except Exception as e:
                    print(f"    {probe.id}: judge FAILED {e}",
                          file=sys.stderr, flush=True)
                    continue
                results[cond].append({
                    "probe_id": probe.id,
                    "probe_text": probe.text,
                    "category": probe.category,
                    "score": j.score, "label": j.label,
                    "reason": j.reason,
                    "response_full": text,
                })
                print(f"    {probe.id}: {j.score}", flush=True)
            per_target_save.write_text(json.dumps({
                "target_label": label, "model_id": model_id,
                "in_progress_condition": cond,
                "per_condition": results,
            }, indent=2))
        # Compute means
        def mean(items):
            scs = [r["score"] for r in items
                   if isinstance(r.get("score"), int)
                   and r["score"] in (0, 1, 2, 3)]
            return sum(scs) / len(scs) if scs else float("nan")
        m_scratch = mean(results["scratch"])
        m_recent = mean(results["recent3K_2nd_cpre"])
        delta = m_recent - m_scratch
        print(f"  scratch={m_scratch:.3f} recent3K_2nd={m_recent:.3f} "
              f"Δ={delta:+.3f}", flush=True)
        all_results[label] = {
            "model_id": model_id, "provider": provider,
            "per_condition": results,
            "scratch_mean": m_scratch,
            "recent3K_2nd_mean": m_recent,
            "delta": delta,
        }

    # Save aggregate
    out_path = REPO_ROOT / "docs/B1_MULTICONTEXT_REPLICATION.json"
    out_path.write_text(json.dumps({
        "experiment": "b1_multicontext_replication",
        "second_boundary_idx": SECOND_BOUNDARY,
        "judge_model": "claude-sonnet-4-5",
        "comparison_first_cpre": {
            "Sonnet 4.6": -0.48, "Opus 4.7": -0.44, "GPT-4.1": -0.36,
        },
        "per_target": all_results,
    }, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    run()
