"""B2: Content-position cross-model replication.

Replicate the 5-condition Sonnet content-position experiment on
Opus 4.7 and GPT-5 targets. Same conditions, same source c_pre,
same probes, same Sonnet-4.6 judge. Tests whether the recent3K vs
filler14K dichotomy is Sonnet-specific or cross-model.

Cost: ~30 target calls × 5 conds × 2 targets = ~300 calls + 300
judge calls. ~$80.

Output:
  docs/CONTENT_POSITION_OPUS.json
  docs/CONTENT_POSITION_GPT5.json
"""
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
random.seed(42)

from analyze_length_control import (
    load_events, extract_verbatim_slice, parse_judge,
    PROBE_FRAMING, DEFAULT_SYSTEM,
)
from harness.probes import ALL_PROBES
from harness.judge import JUDGE_SYSTEM_PROMPT


# Filler text reproducible from same template as Sonnet content_position
FILLER_TEMPLATE = (
    "The following is filler placeholder content for an experimental "
    "control. Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "The quick brown fox jumps over the lazy dog. Pack my box with five "
    "dozen liquor jugs. The rain in Spain falls mainly on the plain. "
    "How vexingly quick daft zebras jump. The five boxing wizards jump "
    "quickly. Sphinx of black quartz, judge my vow. Two driven jocks "
    "help fax my big quiz. Cwm fjord bank glyphs vext quiz. "
)


def make_filler(target_chars: int) -> str:
    chunks = []
    total = 0
    while total < target_chars:
        chunks.append(FILLER_TEMPLATE)
        total += len(FILLER_TEMPLATE)
    return "".join(chunks)[:target_chars]


def call_anthropic(client, model_id, messages, max_retries=4):
    from anthropic import APIStatusError
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model_id, system=DEFAULT_SYSTEM,
                max_tokens=400, messages=messages,
            )
            text = ""
            for b in resp.content:
                if b.type == "text":
                    text = b.text
                    break
            return text, resp.usage.input_tokens, resp.usage.output_tokens
        except APIStatusError as e:
            if e.status_code in (429, 529) and attempt < max_retries - 1:
                time.sleep(2 * (2 ** attempt))
                continue
            raise


def call_openai(client, messages, max_retries=4):
    """GPT-5 chat completion. We treat the system message as one entry."""
    last_err = None
    for attempt in range(max_retries):
        try:
            # Build OAI-format messages: system first, then alternating user/assistant
            oai_msgs = [{"role": "system", "content": DEFAULT_SYSTEM}]
            for m in messages:
                oai_msgs.append({"role": m["role"], "content": m["content"]})
            resp = client.chat.completions.create(
                model="gpt-5",
                messages=oai_msgs,
                max_completion_tokens=400,
            )
            text = resp.choices[0].message.content or ""
            return text, resp.usage.prompt_tokens, resp.usage.completion_tokens
        except Exception as e:
            last_err = e
            time.sleep(2 * (2 ** attempt))
    raise last_err


def call_judge(client, probe, response, system, max_retries=3):
    from anthropic import APIStatusError
    user_msg = f"PROBE:\n{probe}\n\nRESPONSE:\n{response}\n\nReturn the JSON now."
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-6", system=system, max_tokens=300,
                messages=[{"role": "user", "content": user_msg}],
            )
            return parse_judge(resp.content[0].text)
        except APIStatusError as e:
            if e.status_code in (429, 529) and attempt < max_retries - 1:
                time.sleep(2 * (2 ** attempt))
                continue
            raise


def run_condition(target_call, judge_client, label, prior_messages, probes):
    results = []
    for i, probe in enumerate(probes):
        msgs = list(prior_messages) + [
            {"role": "user", "content": f"{PROBE_FRAMING}\n\n{probe.text}"}
        ]
        try:
            response, in_tok, out_tok = target_call(msgs)
        except Exception as e:
            results.append({"probe_id": probe.id, "error": str(e)[:200]})
            continue
        judgment = call_judge(judge_client, probe.text, response, JUDGE_SYSTEM_PROMPT)
        results.append({
            "probe_id": probe.id,
            "probe_text": probe.text,
            "score": judgment.get("score", -1),
            "label": judgment.get("label", ""),
            "reason": judgment.get("reason", "")[:120],
            "response_preview": (response or "")[:300],
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        })
        if (i + 1) % 5 == 0:
            valid = [r["score"] for r in results if r.get("score", -1) in (0, 1, 2, 3)]
            mean = sum(valid) / max(len(valid), 1)
            print(f"  [{label} {i+1}/25] running mean={mean:.2f}", flush=True)
    valid = [r["score"] for r in results if r.get("score", -1) in (0, 1, 2, 3)]
    return {
        "label": label,
        "mean_score": sum(valid) / max(len(valid), 1),
        "n_valid": len(valid),
        "results": results,
    }


def build_conditions(events, target_boundary):
    verbatim_full = extract_verbatim_slice(events, target_boundary, 28000)
    recent_3K = verbatim_full[-3000:]
    earlier_11K = verbatim_full[-14000:-3000]
    filler_11K = make_filler(len(earlier_11K))
    filler_14K = make_filler(14000)
    ack = {"role": "assistant", "content": "Acknowledged. How can I help continue this work?"}
    return {
        "scratch": [],
        "recent3K": [{"role": "user", "content": recent_3K}, ack],
        "recent3K_filler": [{"role": "user", "content": filler_11K + recent_3K}, ack],
        "recent3K_earlier": [{"role": "user", "content": earlier_11K + recent_3K}, ack],
        "filler14K": [{"role": "user", "content": filler_14K}, ack],
    }


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else 'opus'
    if target not in ('opus', 'gpt5'):
        sys.exit('usage: b2_content_position_crossmodel.py [opus|gpt5]')

    from anthropic import Anthropic
    judge_client = Anthropic()

    if target == 'opus':
        target_client = Anthropic()
        target_call = lambda msgs: call_anthropic(target_client, 'claude-opus-4-7', msgs)
        target_label = 'claude-opus-4-7'
        out_path = REPO_ROOT / 'docs/CONTENT_POSITION_OPUS.json'
    else:
        from openai import OpenAI
        oai = OpenAI()
        target_call = lambda msgs: call_openai(oai, msgs)
        target_label = 'gpt-5'
        out_path = REPO_ROOT / 'docs/CONTENT_POSITION_GPT5.json'

    events = load_events()
    boundaries = [i for i, e in enumerate(events)
                  if e.get('type') == 'system' and e.get('subtype') == 'compact_boundary']
    target_boundary = boundaries[-1]
    conditions = build_conditions(events, target_boundary)

    print(f'Target: {target_label}')
    print(f'Conditions: {list(conditions.keys())}')
    for label, msgs in conditions.items():
        chars = len(msgs[0]['content']) if msgs else 0
        print(f'  {label}: {chars} chars')

    results = {}
    for label, prior in conditions.items():
        print(f'\n=== {target_label} {label} ===', flush=True)
        results[label] = run_condition(target_call, judge_client, label, prior, ALL_PROBES)
        print(f'  mean = {results[label]["mean_score"]:.3f}', flush=True)

    out = {
        'target_model': target_label,
        'judge_model': 'claude-sonnet-4-6',
        'per_condition_means': {k: v['mean_score'] for k, v in results.items()},
        'full_results': results,
    }
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
