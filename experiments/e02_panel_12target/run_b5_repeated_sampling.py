"""B5: Repeated sampling on Sonnet length-control.

Re-run the 4-condition Sonnet length-control with k=3 samples per
(probe, condition) cell. Estimates the within-cell sampling variance
to bound how much of the original n=25 single-sample analysis
captured probe variance vs sample variance.

Original cost was ~25 calls/cond × 4 conds = 100 target calls.
With k=3: 25 × 3 × 4 = 300 target calls + 300 judge calls. ~$30.

Conditions: same as original length-control (scratch, summary,
verbatim_slice, shuffled_slice). Source context: same c_pre as the
original Sonnet run (same compaction event). Judge: Sonnet 4.6
(matching original).

For each (probe, condition), we collect 3 independent samples
under provider-default decoding. We then:
  - Compute within-cell SD of the 3 judge scores per (probe, condition)
  - Compute mean Δ vs scratch using:
      (a) mean of 3 samples per probe-condition (low-variance estimator)
      (b) just the first sample (matches original n=25 analysis)
  - Compare effect size and 95% bootstrap CI between (a) and (b)
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

OUT_JSON = REPO_ROOT / 'docs/B5_REPEATED_SAMPLING.json'

from analyze_length_control import (
    load_events, extract_verbatim_slice, extract_summary, parse_judge,
    PROBE_FRAMING, DEFAULT_SYSTEM,
)
from harness.probes import ALL_PROBES
from harness.judge import JUDGE_SYSTEM_PROMPT


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


def run_repeated(client, label, prior_messages, probes, k=3):
    """For each probe, collect k independent samples + k judge calls."""
    out = []
    for i, probe in enumerate(probes):
        msgs = list(prior_messages) + [
            {'role': 'user', 'content': f'{PROBE_FRAMING}\n\n{probe.text}'}
        ]
        samples = []
        for sample_i in range(k):
            try:
                response, in_tok, out_tok = call_sonnet(client, msgs)
            except Exception as e:
                samples.append({'error': str(e)[:200]})
                continue
            judgment = call_judge(client, probe.text, response, JUDGE_SYSTEM_PROMPT)
            samples.append({
                'sample_idx': sample_i,
                'score': judgment.get('score', -1),
                'response_preview': (response or '')[:200],
            })
        out.append({
            'probe_id': probe.id,
            'probe_text': probe.text,
            'samples': samples,
        })
        valid_scores_so_far = []
        for r in out:
            for s in r['samples']:
                if s.get('score', -1) in (0, 1, 2, 3):
                    valid_scores_so_far.append(s['score'])
        if (i + 1) % 5 == 0:
            mean_so_far = sum(valid_scores_so_far) / max(len(valid_scores_so_far), 1)
            print(f'  [{label} probe {i+1}/25] running mean = {mean_so_far:.2f}', flush=True)
    return out


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
    verbatim_full = extract_verbatim_slice(events, target_boundary, 28000)
    # Match the original Sonnet length-control: ~13.7K char slice
    verbatim_slice = verbatim_full[-13700:]
    # Shuffled: chunk on 200-char boundaries and shuffle deterministically
    rng = random.Random(42)
    chunks = [verbatim_slice[i:i+200] for i in range(0, len(verbatim_slice), 200)]
    rng.shuffle(chunks)
    shuffled_slice = ''.join(chunks)
    # Summary: extract from the source compaction event (same as original script)
    summary_text = extract_summary(events, target_boundary)
    # Truncate to match the original 13.7K-char target
    if summary_text and len(summary_text) > 13700:
        summary_text = summary_text[:13700]
    print(f'verbatim_slice: {len(verbatim_slice)} chars')
    print(f'shuffled_slice: {len(shuffled_slice)} chars')
    print(f'summary_text: {len(summary_text or "")} chars')

    ack = {'role': 'assistant', 'content': 'Acknowledged. How can I help continue this work?'}
    conditions = {
        'scratch': [],
        'summary': [{'role': 'user', 'content': summary_text}, ack],
        'verbatim_slice': [{'role': 'user', 'content': verbatim_slice}, ack],
        'shuffled_slice': [{'role': 'user', 'content': shuffled_slice}, ack],
    }

    all_results = {}
    for label, prior in conditions.items():
        print(f'\n=== {label} (k=3) ===', flush=True)
        all_results[label] = run_repeated(client, label, prior, ALL_PROBES, k=3)

    # Save raw
    OUT_JSON.write_text(json.dumps({
        'target_model': 'claude-sonnet-4-6',
        'judge_model': 'claude-sonnet-4-6',
        'k_samples': 3,
        'conditions': all_results,
    }, indent=2, default=str))
    print(f'\nWrote {OUT_JSON}')


if __name__ == '__main__':
    main()
