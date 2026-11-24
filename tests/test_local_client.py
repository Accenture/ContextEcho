import json
from unittest.mock import Mock

import httpx
import pytest
from openai import OpenAI

from harness.clients import TargetClient


@pytest.fixture
def endpoint(monkeypatch):
    requests = []
    response = {
        "id": "local-response", "object": "chat.completion", "created": 0,
        "model": "local-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "42"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 19, "completion_tokens": 2, "total_tokens": 21},
    }

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=response)

    def make_client(**kwargs):
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))

    monkeypatch.setenv("LOCAL_OPENAI_BASE_URL", "http://localhost:18536/v1")
    monkeypatch.delenv("LOCAL_OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("harness.clients_local.OpenAI", make_client)
    return requests, response


@pytest.mark.parametrize("model", ["Qwen/Qwen3-4B", "gpt-5-local", "o3-local"])
@pytest.mark.parametrize("budget", [16, 4097])
def test_local_request_preserves_budget_and_blocks(endpoint, model, budget):
    requests, _ = endpoint
    cost = Mock()
    client = TargetClient("local", model, cost, "parity")
    answer = client.step("Use concise answers.", [
        {"role": "user", "content": [{"type": "text", "text": "Part A"},
                                      {"type": "text", "text": "Part B"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Ready."}]},
        {"role": "user", "content": [{"type": "text", "text": "Continue."}]},
    ], [], max_tokens=budget)
    body = json.loads(requests[0].content)
    assert str(requests[0].url) == "http://localhost:18536/v1/chat/completions"
    assert body == {"model": model, "max_tokens": budget, "temperature": 0.0,
                    "messages": [{"role": "system", "content": "Use concise answers."},
                                 {"role": "user", "content": "Part A\nPart B"},
                                 {"role": "assistant", "content": "Ready."},
                                 {"role": "user", "content": "Continue."}]}
    assert answer.text == "42"
    assert answer.stop_reason == "stop"
    assert answer.raw_usage == {"input_tokens": 19, "output_tokens": 2}
    cost.log.assert_called_once_with(session_id="parity", role="target", model_id=model,
                                     input_tokens=19, output_tokens=2)


def test_local_endpoint_is_required(monkeypatch):
    monkeypatch.delenv("LOCAL_OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://unrelated.invalid/v1")
    with pytest.raises(KeyError, match="LOCAL_OPENAI_BASE_URL"):
        TargetClient("local", "local-model")


def test_local_authentication_is_separate(endpoint, monkeypatch):
    requests, _ = endpoint
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-test-value")
    monkeypatch.setenv("LOCAL_OPENAI_API_KEY", "endpoint-test-value")
    TargetClient("local", "local-model").step("", [], [], max_tokens=16)
    assert requests[0].headers["authorization"] == "Bearer endpoint-test-value"


def test_local_tools_round_trip(endpoint):
    requests, response = endpoint
    response["choices"][0].update(finish_reason="tool_calls", message={
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "next-call", "type": "function",
                        "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'}}],
    })
    tool = {"name": "add", "description": "Add two integers.",
            "input_schema": {"type": "object", "properties": {"a": {"type": "integer"},
                                                               "b": {"type": "integer"}}}}
    result = TargetClient("local", "local-model").step("", [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "previous-call",
                                            "name": "add", "input": {"a": 1, "b": 2}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "previous-call",
                                      "content": "3"}, {"type": "text", "text": "Now add 2 and 3."}]},
    ], [tool], max_tokens=32)
    body = json.loads(requests[0].content)
    assert body["tools"] == [{"type": "function", "function": {
        "name": "add", "description": tool["description"], "parameters": tool["input_schema"]}}]
    assert body["messages"][1]["tool_calls"][0]["id"] == "previous-call"
    assert json.loads(body["messages"][1]["tool_calls"][0]["function"]["arguments"]) == {"a": 1, "b": 2}
    assert body["messages"][2] == {"role": "tool", "tool_call_id": "previous-call", "content": "3"}
    assert body["messages"][3] == {"role": "user", "content": "Now add 2 and 3."}
    assert result.tool_calls[0].id == "next-call"
    assert result.tool_calls[0].name == "add"
    assert result.tool_calls[0].input == {"a": 2, "b": 3}
    assert result.stop_reason == "tool_calls"


def test_local_does_not_invent_missing_token_usage(endpoint):
    _, response = endpoint
    response["usage"] = None
    with pytest.raises(AttributeError):
        TargetClient("local", "local-model").step("", [], [], max_tokens=16)
