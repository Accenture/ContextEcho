"""A2: Re-judge full responses under GPT-5 (Pass-5 review fix).

Claude's Pass-5 review (W2): the cross-judge collapse on the
12-model panel is confounded by the fact that GPT-5 judge scored
300-char response previews while Sonnet judge scored full responses.
Re-generate full responses for the 7 originally-drifting targets'
scratch + recent3K cells, then re-judge under both Sonnet and GPT-5
on full responses.

Targets (all originally drifting under Sonnet judge):
  Sonnet 4.5, Sonnet 4.6, Opus 4.6, Opus 4.7, GPT-4.1, Qwen 3 235B,
  DeepSeek V3.

Conditions: scratch + recent3K only (the headline contrast).
n=25 probes per cell. 7 targets × 2 conditions × 25 = 350 generations,
+ 350 Sonnet judge calls + 350 GPT-5 judge calls.

Cost: ~$30 estimated.

Output:
  docs/A2_CROSS_JUDGE_FULL_RESPONSES.json
  docs/a2_per_target/<target>.json (incremental)
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analyze_length_control import (
    load_events, extract_verbatim_slice,
    PROBE_FRAMING, DEFAULT_SYSTEM,
)
from harness.probes import ALL_PROBES
from harness.judge import Judge


# 7 originally drifting targets (Sonnet judge): API model id, label, provider
TARGETS = [
    # Together first (faster API typically)
    ("Qwen/Qwen3-235B-A22B-Instruct-2507-tput", "Qwen 3 235B", "together"),
    ("deepseek-ai/DeepSeek-V3", "DeepSeek V3", "together"),
    # OpenAI
    ("gpt-4.1", "GPT-4.1", "openai"),
    # Anthropic (slowest)
    ("claude-sonnet-4-5", "Sonnet 4.6", "anthropic"),
    ("claude-sonnet-4-5", "Sonnet 4.5", "anthropic"),  # same API model id
    ("claude-opus-4-1-20250805", "Opus 4.7", "anthropic"),
    ("claude-opus-4-1-20250805", "Opus 4.6", "anthropic"),
]


def find_compaction_boundary(events):
    for i, e in enumerate(events):
        msg = e.get("message", {})
        c = msg.get("content", "")
        if isinstance(c, str) and c.startswith("This session is being continued"):
            return i
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    if b.get("text", "").startswith("This session is being continued"):
                        return i
    return None


def get_recent3K():
    events = load_events()
    boundary = find_compaction_boundary(events)
    if boundary is None:
        sys.exit("ERROR: no compaction boundary")
    full = extract_verbatim_slice(events, boundary, 14000)
    return full[-3000:]


def build_messages(prior, probe):
    framing = f"{PROBE_FRAMING}\n\n{probe.text}"
    if prior is None:
        return [{"role": "user", "content": framing}]
    ack = {"role": "assistant",
           "content": "Acknowledged. How can I help continue this work?"}
    return [
        {"role": "user", "content": prior},
        ack,
        {"role": "user", "content": framing},
    ]


def call_anthropic(client, model_id, messages, max_retries=4):
    from anthropic import APIStatusError
    for attempt in range(max_retries):
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
            if e.status_code in (429, 529) and attempt < max_retries - 1:
                time.sleep(2 * (2 ** attempt))
                continue
            raise


def call_openai(client, model_id, messages, max_retries=4):
    last = None
    for attempt in range(max_retries):
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
            if attempt < max_retries - 1:
                time.sleep(2 * (2 ** attempt))
    raise last  # type: ignore[misc]


def call_together(client, model_id, messages, max_retries=4):
    last = None
    for attempt in range(max_retries):
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
            if attempt < max_retries - 1:
                time.sleep(2 * (2 ** attempt))
    raise last  # type: ignore[misc]


def make_target_call(provider, model_id, anth, oai, tg):
    if provider == "anthropic":
        return lambda msgs: call_anthropic(anth, model_id, msgs)
    if provider == "openai":
        return lambda msgs: call_openai(oai, model_id, msgs)
    if provider == "together":
        return lambda msgs: call_together(tg, model_id, msgs)
    raise ValueError(provider)


def run_target(target_label, target_call, prior, sonnet_judge, gpt5_judge, save_path):
    out = {"scratch": [], "recent3K": []}
    for cond, prior_for_cond in [("scratch", None), ("recent3K", prior)]:
        print(f"\n  [{target_label}] {cond} ...", flush=True)
        for probe in ALL_PROBES:
            messages = build_messages(prior_for_cond, probe)
            try:
                response = target_call(messages)
            except Exception as e:
                print(f"    {probe.id}: target FAILED {e}",
                      file=sys.stderr, flush=True)
                continue
            if not response:
                continue
            # Score under both judges
            try:
                sj = sonnet_judge.score(probe.text, response)
                gj = gpt5_judge.score(probe.text, response)
            except Exception as e:
                print(f"    {probe.id}: judge FAILED {e}",
                      file=sys.stderr, flush=True)
                continue
            out[cond].append({
                "probe_id": probe.id,
                "probe_text": probe.text,
                "category": probe.category,
                "response_full": response,
                "response_len": len(response),
                "sonnet_score": sj.score,
                "sonnet_label": sj.label,
                "gpt5_score": gj.score,
                "gpt5_label": gj.label,
            })
            print(f"    {probe.id}: sonnet={sj.score} gpt5={gj.score} "
                  f"len={len(response)}", flush=True)
        # Incremental save after each condition
        save_path.write_text(json.dumps({
            "target_label": target_label,
            "in_progress_condition": cond,
            "per_condition": out,
        }, indent=2))
    return out


def main():
    prior = get_recent3K()
    print(f"recent3K: {len(prior)} chars", flush=True)

    from anthropic import Anthropic
    from openai import OpenAI
    anth = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tg = OpenAI(api_key=os.environ["TOGETHER_AI_KEY"],
                base_url="https://api.together.xyz/v1")

    sonnet_judge = Judge(provider="anthropic", model_id="claude-sonnet-4-5")
    gpt5_judge = Judge(provider="openai", model_id="gpt-5")

    target_save_dir = REPO_ROOT / "docs/a2_per_target"
    target_save_dir.mkdir(parents=True, exist_ok=True)
    out_path = REPO_ROOT / "docs/A2_CROSS_JUDGE_FULL_RESPONSES.json"

    all_results = {}
    for model_id, target_label, provider in TARGETS:
        print(f"\n=== {target_label} ({provider}, {model_id}) ===", flush=True)
        per_target_save = target_save_dir / f"{target_label.replace(' ', '_')}.json"
        # Resume support
        if per_target_save.exists():
            try:
                cached = json.loads(per_target_save.read_text())
                conds = cached.get("per_condition", {})
                if (len(conds.get("scratch", [])) >= 24
                        and len(conds.get("recent3K", [])) >= 24):
                    print(f"  [SKIP] cached complete: {per_target_save}",
                          flush=True)
                    all_results[target_label] = {
                        "model_id": model_id, "provider": provider,
                        "per_condition": conds,
                    }
                    continue
            except Exception:
                pass

        target_call = make_target_call(provider, model_id, anth, oai, tg)
        try:
            r = run_target(target_label, target_call, prior,
                           sonnet_judge, gpt5_judge, per_target_save)
            all_results[target_label] = {
                "model_id": model_id, "provider": provider,
                "per_condition": r,
            }
        except Exception as e:
            print(f"FAILED on {target_label}: {e}",
                  file=sys.stderr, flush=True)
            all_results[target_label] = {"error": str(e)}
        # Save aggregate after each target
        out_path.write_text(json.dumps({
            "experiment": "a2_cross_judge_full_responses",
            "targets": [(t[0], t[1], t[2]) for t in TARGETS],
            "judge_models": ["claude-sonnet-4-5", "gpt-5"],
            "per_target": all_results,
        }, indent=2))

    # Print summary
    print("\n=== Summary: cross-judge replication on FULL responses ===",
          flush=True)
    print(f"{'Target':<14} | {'sonnet Δ':>10} | {'gpt5 Δ':>10}", flush=True)
    print("-" * 40, flush=True)
    for tl, td in all_results.items():
        if "error" in td:
            print(f"{tl:<14} | error", flush=True)
            continue
        pc = td["per_condition"]
        s_scratch = [c["sonnet_score"] for c in pc.get("scratch", [])
                     if isinstance(c.get("sonnet_score"), int)
                     and c["sonnet_score"] in (0, 1, 2, 3)]
        s_recent = [c["sonnet_score"] for c in pc.get("recent3K", [])
                    if isinstance(c.get("sonnet_score"), int)
                    and c["sonnet_score"] in (0, 1, 2, 3)]
        g_scratch = [c["gpt5_score"] for c in pc.get("scratch", [])
                     if isinstance(c.get("gpt5_score"), int)
                     and c["gpt5_score"] in (0, 1, 2, 3)]
        g_recent = [c["gpt5_score"] for c in pc.get("recent3K", [])
                    if isinstance(c.get("gpt5_score"), int)
                    and c["gpt5_score"] in (0, 1, 2, 3)]
        s_delta = (sum(s_recent) / len(s_recent)
                   - sum(s_scratch) / len(s_scratch)) if s_scratch and s_recent else float("nan")
        g_delta = (sum(g_recent) / len(g_recent)
                   - sum(g_scratch) / len(g_scratch)) if g_scratch and g_recent else float("nan")
        print(f"{tl:<14} | {s_delta:>+10.2f} | {g_delta:>+10.2f}", flush=True)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
