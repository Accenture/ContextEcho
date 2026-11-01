"""Translate ContextEcho conversations to Ollama's native chat protocol."""

import os

import httpx


def make_ollama_client():
    return httpx.Client(base_url=os.environ['OLLAMA_BASE_URL'].rstrip('/') + '/', timeout=600)


def chat_payload(model, system_prompt, messages, tools, max_tokens):
    native = [{'role': 'system', 'content': system_prompt}]
    names = {}
    for message in messages:
        blocks = message['content']
        text = '\n'.join(block['text'] for block in blocks if block['type'] == 'text')
        calls = [block for block in blocks if block['type'] == 'tool_use']
        results = [block for block in blocks if block['type'] == 'tool_result']
        for result in results:
            native.append({'role': 'tool', 'content': result['content'],
                           'tool_name': names[result['tool_use_id']],
                           'tool_call_id': result['tool_use_id']})
        if text or calls:
            item = {'role': message['role'], 'content': text}
            if calls:
                item['tool_calls'] = [{'id': call['id'], 'function': {
                    'name': call['name'], 'arguments': call['input']}} for call in calls]
                names.update({call['id']: call['name'] for call in calls})
            native.append(item)
    return {'model': model, 'messages': native, 'stream': False, 'think': False,
            'options': {'temperature': 0, 'num_predict': max_tokens},
            'tools': [{'type': 'function', 'function': {
                'name': tool['name'], 'description': tool['description'],
                'parameters': tool['input_schema']}} for tool in tools]}


def native_chat(client, payload):
    response = client.post('api/chat', json=payload)
    response.raise_for_status()
    return response.json()
