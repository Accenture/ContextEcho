"""Non-Anthropic targets extension — expands cross-organization sample
to address the n=1-per-org asymmetry.

Available serverless targets confirmed 2026-04-25:
  Together.ai:
    - meta-llama/Llama-3.3-70B-Instruct-Turbo (DONE earlier)
    - Qwen/Qwen3-235B-A22B-Instruct-2507-tput (running concurrently)
    - deepseek-ai/DeepSeek-V3 (NEW — Chinese non-Qwen)
  OpenAI:
    - gpt-5 (DONE earlier)
    - gpt-4o (NEW — different OpenAI generation)
    - gpt-4o-mini (smaller OpenAI)
    - gpt-4.1 (newer OpenAI generation)

Usage:
  python3 scripts/non_anthropic_extension.py [target]
where target is one of: deepseek_v3, gpt4o, gpt4o_mini, gpt41

Each runs the full 5-condition content-position protocol with the
same Sonnet 4.6 judge, on the SAME Claude-derived c_pre as the
B2 cross-model experiment.

Cost per target: ~$0.10-3 depending on provider.
Output: docs/CONTENT_POSITION_<TARGET>.json
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


def call_together(client, model_id: str, messages, max_retries=4):
    last_err = None
    for attempt in range(max_retries):
        try:
            oai_msgs = [{"role": "system", "content": DEFAULT_SYSTEM}]
            for m in messages:
                oai_msgs.append({"role": m["role"], "content": m["content"]})
            resp = client.chat.completions.create(
                model=model_id, messages=oai_msgs,
                max_tokens=400, temperature=0.7, timeout=60,
            )
            text = resp.choices[0].message.content or ""
            return text, resp.usage.prompt_tokens, resp.usage.completion_tokens
        except Exception as e:
            last_err = e
            time.sleep(2 * (2 ** attempt))
    raise last_err


def call_openai(client, model_id: str, messages, max_retries=4):
    last_err = None
    for attempt in range(max_retries):
        try:
            oai_msgs = [{"role": "system", "content": DEFAULT_SYSTEM}]
            for m in messages:
                oai_msgs.append({"role": m["role"], "content": m["content"]})
            resp = client.chat.completions.create(
                model=model_id, messages=oai_msgs,
                max_completion_tokens=4096, timeout=60,
            )
            text = resp.choices[0].message.content or ""
            return text, resp.usage.prompt_tokens, resp.usage.completion_tokens
        except Exception as e:
            last_err = e
            time.sleep(2 * (2 ** attempt))
    raise last_err


def call_judge(client_anthropic, probe, response, system, max_retries=3):
    from anthropic import APIStatusError
    user_msg = f"PROBE:\n{probe}\n\nRESPONSE:\n{response}\n\nReturn the JSON now."
    for attempt in range(max_retries):
        try:
            resp = client_anthropic.messages.create(
                model="claude-sonnet-4-6", system=system, max_tokens=300,
                messages=[{"role": "user", "content": user_msg}],
            )
            return parse_judge(resp.content[0].text)
        except APIStatusError as e:
            if e.status_code in (429, 529) and attempt < max_retries - 1:
                time.sleep(2 * (2 ** attempt))
                continue
            raise


def run_condition(call_fn, judge_client, label, prior_messages, probes):
    results = []
    for i, probe in enumerate(probes):
        msgs = list(prior_messages) + [
            {"role": "user", "content": f"{PROBE_FRAMING}\n\n{probe.text}"}
        ]
        try:
            response, in_tok, out_tok = call_fn(msgs)
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


# (provider, model_id, output_label)
TARGET_MAP = {
    'deepseek_v3':   ('together', 'deepseek-ai/DeepSeek-V3', 'CONTENT_POSITION_DEEPSEEK_V3'),
    'gpt4o':         ('openai',   'gpt-4o',                  'CONTENT_POSITION_GPT4O'),
    'gpt4o_mini':    ('openai',   'gpt-4o-mini',             'CONTENT_POSITION_GPT4O_MINI'),
    'gpt41':         ('openai',   'gpt-4.1',                 'CONTENT_POSITION_GPT41'),
}


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else 'deepseek_v3'
    if target not in TARGET_MAP:
        sys.exit(f'usage: non_anthropic_extension.py [{"|".join(TARGET_MAP)}]')
    provider, model_id, out_label = TARGET_MAP[target]
    out_path = REPO_ROOT / f'docs/{out_label}.json'

    if not os.environ.get('ANTHROPIC_API_KEY'):
        sys.exit('Set ANTHROPIC_API_KEY (judge)')
    from anthropic import Anthropic
    judge_client = Anthropic()

    if provider == 'together':
        if not os.environ.get('TOGETHER_API_KEY'):
            sys.exit('Set TOGETHER_API_KEY')
        from openai import OpenAI
        client = OpenAI(api_key=os.environ['TOGETHER_API_KEY'],
                        base_url='https://api.together.xyz/v1')
        call_fn = lambda msgs: call_together(client, model_id, msgs)
    else:  # openai
        if not os.environ.get('OPENAI_API_KEY'):
            sys.exit('Set OPENAI_API_KEY')
        from openai import OpenAI
        client = OpenAI()
        call_fn = lambda msgs: call_openai(client, model_id, msgs)

    events = load_events()
    boundaries = [i for i, e in enumerate(events)
                  if e.get('type') == 'system' and e.get('subtype') == 'compact_boundary']
    target_boundary = boundaries[-1]
    conditions = build_conditions(events, target_boundary)

    print(f'Target: {model_id} via {provider}')
    print(f'Output: {out_path}')
    for label, msgs in conditions.items():
        chars = len(msgs[0]['content']) if msgs else 0
        print(f'  {label}: {chars} chars')

    results = {}
    for label, prior in conditions.items():
        print(f'\n=== {model_id} {label} ===', flush=True)
        results[label] = run_condition(call_fn, judge_client, label, prior, ALL_PROBES)
        print(f'  mean = {results[label]["mean_score"]:.3f}', flush=True)

    out = {
        'target_model': model_id,
        'judge_model': 'claude-sonnet-4-6',
        'experiment': 'non_anthropic_extension',
        'provider': provider,
        'per_condition_means': {k: v['mean_score'] for k, v in results.items()},
        'full_results': results,
    }
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
