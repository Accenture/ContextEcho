"""Direction A1: Context-source ablation.

Tests whether the recency-content phenomenon on Anthropic models is
specific to Claude-derived contexts or generalizes to ANY agentic
coding context. Reuses the synthetic GPT-5-target session data we
already have to build a "GPT-5-derived c_pre", then runs the
content-position protocol with Sonnet 4.6 as target.

If Sonnet 4.6 drifts on a GPT-5-derived context: the phenomenon is
"Anthropic models drift on agentic-coding contexts in general"
(broader claim). If Sonnet 4.6 doesn't drift: the phenomenon is
"Anthropic models drift on Anthropic-derived contexts only"
(narrower but more interesting claim about cross-family register
recognition).

Source: data/openai_gpt-5_debug_and_fix_baseline_seed301 (40-turn
GPT-5 session). We extract a verbatim slice ending at the last
turn (analogous to the Claude Code compact_boundary), and run the
same 5-condition content-position experiment.

Output: docs/A1_CONTEXT_SOURCE_GPT5_DERIVED.json
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

from analyze_length_control import parse_judge, PROBE_FRAMING, DEFAULT_SYSTEM
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


def _flatten_content(content) -> str:
    """The GPT-5 harness stores content as either a string or a list of
    structured blocks (tool_use, tool_result). Flatten to a single string
    that preserves the substantive text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get('type', '')
            if btype == 'text':
                t = block.get('text', '')
                if t:
                    parts.append(t)
            elif btype == 'tool_use':
                name = block.get('name', '')
                inp = block.get('input', {})
                # Render tool call as a readable string
                if isinstance(inp, dict):
                    arg_str = ', '.join(f'{k}={v!r}' for k, v in inp.items())
                else:
                    arg_str = str(inp)
                parts.append(f'<tool_use {name}({arg_str})>')
            elif btype == 'tool_result':
                inner = block.get('content', '')
                if isinstance(inner, str):
                    parts.append(f'<tool_result: {inner}>')
                elif isinstance(inner, list):
                    inner_text = ' '.join(b.get('text', str(b)) if isinstance(b, dict) else str(b) for b in inner)
                    parts.append(f'<tool_result: {inner_text}>')
                else:
                    parts.append(f'<tool_result: {inner}>')
        return ' '.join(parts)
    return ''


def extract_gpt5_verbatim(jsonl_path: Path, target_chars: int = 28000) -> str:
    """Extract a verbatim transcript slice from a GPT-5 synthetic session.

    The GPT-5 harness emits one JSONL row per turn with role in
    {'system', 'user', 'assistant', 'tool_result'}. content is either a
    string (user/system/textual assistant) or a list of structured blocks
    (assistant with tool_use, tool_result with results). We flatten lists
    via _flatten_content and concatenate all role-tagged turns, then
    return the LAST target_chars characters.
    """
    parts = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            role = d.get('role')
            if role not in ('user', 'assistant', 'tool_result'):
                continue
            content_str = _flatten_content(d.get('content'))
            if content_str.strip():
                parts.append(f'[{role}]: {content_str}')
    full = '\n\n'.join(parts)
    return full[-target_chars:]


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


def main():
    if not os.environ.get('ANTHROPIC_API_KEY'):
        sys.exit('Set ANTHROPIC_API_KEY')
    from anthropic import Anthropic
    client = Anthropic()

    # Use seed 301 (40-turn GPT-5 session, 451KB transcript, longest)
    src_path = REPO_ROOT / 'data/openai_gpt-5_debug_and_fix_baseline_seed301_0952d536c9c9/transcript.jsonl'
    if not src_path.exists():
        sys.exit(f'GPT-5 session not found: {src_path}')

    verbatim_full = extract_gpt5_verbatim(src_path, 28000)
    print(f'GPT-5 session verbatim_full: {len(verbatim_full)} chars (last 28K of seed 301)')

    recent_3K = verbatim_full[-3000:]
    earlier_11K = verbatim_full[-14000:-3000]
    filler_11K = make_filler(len(earlier_11K))
    filler_14K = make_filler(14000)
    ack = {'role': 'assistant', 'content': 'Acknowledged. How can I help continue this work?'}

    conditions = {
        'scratch': [],
        'recent3K': [{'role': 'user', 'content': recent_3K}, ack],
        'recent3K_filler': [{'role': 'user', 'content': filler_11K + recent_3K}, ack],
        'recent3K_earlier': [{'role': 'user', 'content': earlier_11K + recent_3K}, ack],
        'filler14K': [{'role': 'user', 'content': filler_14K}, ack],
    }

    print('Conditions:')
    for label, msgs in conditions.items():
        chars = len(msgs[0]['content']) if msgs else 0
        print(f'  {label}: {chars} chars')

    results = {}
    for label, prior in conditions.items():
        print(f'\n=== claude-sonnet-4-6 ON GPT-5-DERIVED {label} ===', flush=True)
        results[label] = run_condition(client, label, prior, ALL_PROBES)
        print(f'  mean = {results[label]["mean_score"]:.3f}', flush=True)

    out = {
        'target_model': 'claude-sonnet-4-6',
        'judge_model': 'claude-sonnet-4-6',
        'context_source': 'GPT-5 synthetic session seed 301 (debug_and_fix)',
        'experiment': 'A1_context_source_ablation',
        'description': 'Sonnet target on a GPT-5-derived c_pre (rather than the Claude-derived c_pre used in B2-Sonnet/B4)',
        'per_condition_means': {k: v['mean_score'] for k, v in results.items()},
        'full_results': results,
    }
    out_path = REPO_ROOT / 'docs/A1_CONTEXT_SOURCE_GPT5_DERIVED.json'
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
