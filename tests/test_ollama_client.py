import json
from unittest.mock import Mock

import httpx
import pytest

from harness.clients import TargetClient


@pytest.fixture
def endpoint(monkeypatch):
    requests = []
    response = {'message': {'role': 'assistant', 'content': '5'}, 'done': True,
                'done_reason': 'stop', 'prompt_eval_count': 28, 'eval_count': 2}

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=response)

    monkeypatch.setattr('harness.clients.make_ollama_client', lambda: httpx.Client(
        base_url='http://localhost:18539/', transport=httpx.MockTransport(handle)))
    return requests, response


@pytest.mark.parametrize('budget', [16, 4097])
def test_native_text_and_output_budget(endpoint, budget):
    requests, _ = endpoint
    tracker = Mock()
    response = TargetClient('ollama', 'research-model', tracker, 'case').step('Be concise.', [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'A'}, {'type': 'text', 'text': 'B'}]},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Ready.'}]},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Add 2 and 3.'}]},
    ], [], max_tokens=budget)
    assert str(requests[0].url) == 'http://localhost:18539/api/chat'
    body = json.loads(requests[0].content)
    assert body == {'model': 'research-model', 'stream': False, 'think': False,
                    'options': {'temperature': 0, 'num_predict': budget}, 'tools': [],
                    'messages': [{'role': 'system', 'content': 'Be concise.'},
                                 {'role': 'user', 'content': 'A\nB'},
                                 {'role': 'assistant', 'content': 'Ready.'},
                                 {'role': 'user', 'content': 'Add 2 and 3.'}]}
    assert response.text == '5' and response.stop_reason == 'stop'
    assert response.raw_usage == {'input_tokens': 28, 'output_tokens': 2}
    tracker.log.assert_called_once_with(session_id='case', role='target', model_id='research-model',
                                        input_tokens=28, output_tokens=2)


def test_parallel_tool_results_preserve_names_and_arguments(endpoint):
    requests, response = endpoint
    response['message'] = {'role': 'assistant', 'content': '', 'thinking': 'private reasoning',
                           'tool_calls': [{'function': {'name': 'add', 'arguments': {'a': 3, 'b': 4}}},
                                          {'function': {'name': 'multiply', 'arguments': {'a': 3, 'b': 4}}}]}
    tool = {'name': 'add', 'description': 'Add integers.',
            'input_schema': {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}}}
    client = TargetClient('ollama', 'research-model')
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Compute.'}]}]
    first = client.step('', messages, [tool], 64)
    assert first.text == '' and first.stop_reason == 'tool_calls'
    assert len({call.id for call in first.tool_calls}) == 2
    messages.extend([
        {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': c.id, 'name': c.name,
                                          'input': c.input} for c in first.tool_calls]},
        {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': first.tool_calls[0].id,
                                     'content': '7'}, {'type': 'tool_result',
                                     'tool_use_id': first.tool_calls[1].id, 'content': '12'}]},
    ])
    second = client.step('', messages, [tool], 64)
    body = json.loads(requests[1].content)
    assert body['messages'][-2:] == [
        {'role': 'tool', 'content': '7', 'tool_name': 'add', 'tool_call_id': first.tool_calls[0].id},
        {'role': 'tool', 'content': '12', 'tool_name': 'multiply', 'tool_call_id': first.tool_calls[1].id}]
    assert body['messages'][2]['tool_calls'][0]['function']['arguments'] == {'a': 3, 'b': 4}
    assert {call.id for call in first.tool_calls}.isdisjoint(call.id for call in second.tool_calls)
    assert body['tools'][0]['function']['parameters'] == tool['input_schema']


def test_truncated_generation_is_reported(endpoint):
    _, response = endpoint
    response['done_reason'] = 'length'
    assert TargetClient('ollama', 'research-model').step('', [], [], 1).stop_reason == 'length'


def test_missing_usage_is_not_scored_as_zero(endpoint):
    _, response = endpoint
    del response['eval_count']
    with pytest.raises(KeyError, match='eval_count'):
        TargetClient('ollama', 'research-model').step('', [], [], 16)
