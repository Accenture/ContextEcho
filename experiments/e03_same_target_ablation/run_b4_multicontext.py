"""B4: Multi-context content-position on Sonnet.

Repeat the 5-condition content-position experiment on 3 different
recent3K slices (different positions in the donated session, or
different compaction events). Tests whether the recent3K vs filler14K
dichotomy is robust to choice of context.

The original Sonnet content-position used:
  recent_3K = verbatim_full[-3000:]  (last 3K of the
                                      ~28K verbatim_full slice)

For B4 we add 2 more "recent" anchors at different positions in the
same donated session:
  context_2: recent_3K = verbatim_full[-9000:-6000]  (3K window
             starting 6K before the end)
  context_3: recent_3K = verbatim_full[-15000:-12000]  (3K window
             starting 12K before the end)

For each context, we run the same 5-condition experiment as the
original B7. We then compare per-context recent3K Δ to verify the
recent-content-drives-drift finding generalizes, and verify
filler14K is null on each context.

Cost: ~25 target × 5 conds × 2 new contexts = ~250 calls + judge.
~$60.

Output: docs/B4_MULTICONTEXT_CTX{2,3}.json (CTX1 = the existing
docs/CONTENT_POSITION_SONNET.json).
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


def call_sonnet(client, messages, max_retries=4):
    from anthropic import APIStatusError
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model='claude-sonnet-4-6', system=DEFAULT_SYSTEM,
                max_tokens=400, messages=messages,
            )
            text = ''
            for b in resp.content:
                if b.type == 'text':
                    text = b.text
                    break
            return text, resp.usage.input_tokens, resp.usage.output_tokens
        except APIStatusError as e:
            if e.status_code in (429, 529) and attempt < max_retries - 1:
                time.sleep(2 * (2 ** attempt))
                continue
            raise


def call_judge(client, probe, response, system, max_retries=3):
    from anthropic import APIStatusError
    user_msg = f'PROBE:\n{probe}\n\nRESPONSE:\n{response}\n\nReturn the JSON now.'
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model='claude-sonnet-4-6', system=system, max_tokens=300,
                messages=[{'role': 'user', 'content': user_msg}],
            )
            return parse_judge(resp.content[0].text)
        except APIStatusError as e:
            if e.status_code in (429, 529) and attempt < max_retries - 1:
                time.sleep(2 * (2 ** attempt))
                continue
            raise


def run_condition(client, label, prior_messages, probes):
    results = []
    for i, probe in enumerate(probes):
        msgs = list(prior_messages) + [
            {'role': 'user', 'content': f'{PROBE_FRAMING}\n\n{probe.text}'}
        ]
        try:
            response, in_tok, out_tok = call_sonnet(client, msgs)
        except Exception as e:
            results.append({'probe_id': probe.id, 'error': str(e)[:200]})
            continue
        judgment = call_judge(client, probe.text, response, JUDGE_SYSTEM_PROMPT)
        results.append({
            'probe_id': probe.id,
            'probe_text': probe.text,
            'score': judgment.get('score', -1),
            'response_preview': (response or '')[:300],
            'input_tokens': in_tok,
            'output_tokens': out_tok,
        })
        if (i + 1) % 5 == 0:
            valid = [r['score'] for r in results if r.get('score', -1) in (0, 1, 2, 3)]
            mean = sum(valid) / max(len(valid), 1)
            print(f'  [{label} {i+1}/25] running mean={mean:.2f}', flush=True)
    valid = [r['score'] for r in results if r.get('score', -1) in (0, 1, 2, 3)]
    return {
        'label': label,
        'mean_score': sum(valid) / max(len(valid), 1),
        'n_valid': len(valid),
        'results': results,
    }


def build_conditions(verbatim_full, recent_offset_end, label_suffix):
    """Build the 5 conditions for a given recent_3K window position.

    recent_offset_end: end-position of the recent 3K window relative to
    end of verbatim_full. 0 = the original (last 3K). +6000 = window
    ending 6K before end.
    """
    recent_3K = verbatim_full[-(recent_offset_end + 3000):
                              -recent_offset_end if recent_offset_end > 0 else None]
    earlier_11K = verbatim_full[-(recent_offset_end + 14000):
                                -(recent_offset_end + 3000)]
    filler_11K = make_filler(len(earlier_11K))
    filler_14K = make_filler(14000)
    ack = {'role': 'assistant', 'content': 'Acknowledged. How can I help continue this work?'}
    return {
        f'scratch_{label_suffix}': [],
        f'recent3K_{label_suffix}': [{'role': 'user', 'content': recent_3K}, ack],
        f'recent3K_filler_{label_suffix}': [{'role': 'user', 'content': filler_11K + recent_3K}, ack],
        f'recent3K_earlier_{label_suffix}': [{'role': 'user', 'content': earlier_11K + recent_3K}, ack],
        f'filler14K_{label_suffix}': [{'role': 'user', 'content': filler_14K}, ack],
    }


def main():
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        sys.exit('Set ANTHROPIC_API_KEY')
    from anthropic import Anthropic
    client = Anthropic()

    events = load_events()
    boundaries = [i for i, e in enumerate(events)
                  if e.get('type') == 'system' and e.get('subtype') == 'compact_boundary']
    target_boundary = boundaries[-1]
    verbatim_full = extract_verbatim_slice(events, target_boundary, 32000)
    print(f'verbatim_full: {len(verbatim_full)} chars')

    # Two new context windows at different positions
    contexts = [
        ('ctx2', 6000),   # recent 3K window from -9000 to -6000
        ('ctx3', 12000),  # recent 3K window from -15000 to -12000
    ]

    for ctx_label, offset in contexts:
        print(f'\n#### Context {ctx_label} (offset {offset} from end) ####', flush=True)
        conditions = build_conditions(verbatim_full, offset, ctx_label)
        for cond_label, msgs in conditions.items():
            chars = len(msgs[0]['content']) if msgs else 0
            print(f'  {cond_label}: {chars} chars')

        results = {}
        for cond_label, prior in conditions.items():
            print(f'\n=== {cond_label} ===', flush=True)
            results[cond_label] = run_condition(client, cond_label, prior, ALL_PROBES)
            print(f'  mean = {results[cond_label]["mean_score"]:.3f}', flush=True)

        # Strip suffix for canonical key names so analysis can re-use existing tooling
        canonical = {}
        for k, v in results.items():
            canonical_key = k.rsplit(f'_{ctx_label}', 1)[0]
            canonical[canonical_key] = v
        out = {
            'target_model': 'claude-sonnet-4-6',
            'judge_model': 'claude-sonnet-4-6',
            'context_label': ctx_label,
            'recent_offset_end': offset,
            'per_condition_means': {k: v['mean_score'] for k, v in canonical.items()},
            'full_results': canonical,
        }
        out_path = REPO_ROOT / f'docs/B4_MULTICONTEXT_{ctx_label.upper()}.json'
        out_path.write_text(json.dumps(out, indent=2, default=str))
        print(f'  Wrote {out_path}', flush=True)


if __name__ == '__main__':
    main()
