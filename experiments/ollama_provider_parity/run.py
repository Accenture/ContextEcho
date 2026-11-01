"""Compare the native Ollama adapter with direct chat requests."""

import argparse
import json
import os
from pathlib import Path

import httpx

from harness.clients import TargetClient


def normalized(response):
    return {'text': response.text, 'tools': [{'name': call.name, 'arguments': call.input}
            for call in response.tool_calls], 'stop_reason': response.stop_reason, **response.raw_usage}


def direct_result(response):
    message = response['message']
    calls = message.get('tool_calls', [])
    return {'text': message['content'], 'tools': [{'name': call['function']['name'],
            'arguments': call['function']['arguments']} for call in calls],
            'stop_reason': 'tool_calls' if calls else response['done_reason'],
            'input_tokens': response['prompt_eval_count'], 'output_tokens': response['eval_count']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--budgets', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads((Path(__file__).parents[1] / 'local_provider_parity/cases.json').read_text())
    adapter = TargetClient('ollama', args.model)
    direct = httpx.Client(base_url=os.environ['OLLAMA_BASE_URL'].rstrip('/') + '/', timeout=600)
    version = direct.get('api/version').json()['version']
    rows = json.loads(args.output.read_text())['comparisons'] if args.output.exists() else []
    completed = {(r['case'], r['max_tokens']) for r in rows}

    def compare(label, system, messages, native, tools, budget):
        actual = adapter.step(system, messages, tools, budget)
        response = direct.post('api/chat', json={'model': args.model,
            'messages': [{'role': 'system', 'content': system}] + native,
            'tools': [{'type': 'function', 'function': {'name': t['name'],
                       'description': t['description'], 'parameters': t['input_schema']}} for t in tools],
            'think': False, 'stream': False, 'options': {'temperature': 0, 'num_predict': budget}})
        response.raise_for_status()
        observed, expected = normalized(actual), direct_result(response.json())
        record = {'case': label, 'max_tokens': budget, 'adapter': observed, 'direct': expected,
                  'equal': observed == expected, 'within_budget': observed['output_tokens'] <= budget}
        rows.append(record)
        args.output.write_text(json.dumps({'model': args.model, 'ollama_version': version,
            'thinking': False, 'temperature': 0, 'comparisons': rows}, indent=2, ensure_ascii=False) + '\n')
        print(label, budget, record['equal'], flush=True)
        return actual

    for case in cases:
        native = [{'role': m['role'], 'content': '\n'.join(b['text'] for b in m['content'])}
                  for m in case['messages']]
        for budget in args.budgets:
            if (case['id'], budget) not in completed:
                compare(case['id'], case['system'], case['messages'], native, [], budget)
    budget = max(args.budgets)
    schema = {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
              'required': ['a', 'b']}
    tools = [{'name': 'add', 'description': 'Add two integers.', 'input_schema': schema}]
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Use add to compute 19 plus 23.'}]}]
    native = [{'role': 'user', 'content': 'Use add to compute 19 plus 23.'}]
    if ('tool_request', budget) not in completed:
        compare('tool_request', 'Use the supplied tool for arithmetic.', messages, native, tools, budget)
    messages += [{'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'sum-call',
                  'name': 'add', 'input': {'a': 19, 'b': 23}}]},
                 {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'sum-call',
                  'content': '42'}, {'type': 'text', 'text': 'Return only the result.'}]}]
    native += [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'sum-call',
                'function': {'name': 'add', 'arguments': {'a': 19, 'b': 23}}}]},
               {'role': 'tool', 'content': '42', 'tool_name': 'add', 'tool_call_id': 'sum-call'},
               {'role': 'user', 'content': 'Return only the result.'}]
    if ('tool_result', budget) not in completed:
        compare('tool_result', 'Use the supplied tool for arithmetic.', messages, native, tools, budget)
    assert all(r['equal'] and r['within_budget'] for r in rows)


if __name__ == '__main__':
    main()
