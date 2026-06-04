"""Path A: Joint behavioral + activation-space measurement on open-weights
drifting and non-drifting targets.

Reuses Lu et al.'s `assistant_axis` package
(github.com/safety-research/assistant-axis):
  - ProbingModel for HF model loading + activation extraction
  - load_axis() to load pre-computed Assistant Axis from HuggingFace
  - project() to compute activation projection onto the axis at target_layer

For each (model, condition) combination in our 5-condition content-position
protocol, we:
  1. Set up the conversation (system prompt + prior context messages
     matching the conditions in scripts/open_weights_content_position.py)
  2. For each of 25 probes: append probe + framing as user turn
  3. Generate model response
  4. Extract residual-stream activations at target_layer for the
     LAST TOKEN of the response (the model's emitted persona signal)
  5. Project onto Assistant Axis
  6. Score the response with our judge for behavioral Δ

This script is intended to run on Modal/RunPod GPU. See infra/modal_path_a.py
for the Modal entrypoint.

Outputs (per model):
  docs/PATH_A_<MODEL>.json with per-(probe, condition) cells:
    {probe_id, condition, behavioral_score, axis_projection,
     response_text, input_tokens, target_layer}

Targets:
  - Qwen/Qwen3-32B  (drifter; we observed Δ=-0.80 behavioral via API)
  - meta-llama/Llama-3.3-70B-Instruct (non-drifter; Δ=+0.08 via API)

Usage (on GPU):
  python3 scripts/path_a_joint_behavioral_activation.py qwen
  python3 scripts/path_a_joint_behavioral_activation.py llama
"""
from __future__ import annotations
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
random.seed(42)

# Reuse our project's source-context construction
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


def build_conditions():
    events = load_events()
    boundaries = [i for i, e in enumerate(events)
                  if e.get('type') == 'system' and e.get('subtype') == 'compact_boundary']
    target_boundary = boundaries[-1]
    verbatim_full = extract_verbatim_slice(events, target_boundary, 28000)
    recent_3K = verbatim_full[-3000:]
    earlier_11K = verbatim_full[-14000:-3000]
    filler_11K = make_filler(len(earlier_11K))
    filler_14K = make_filler(14000)
    ack_text = "Acknowledged. How can I help continue this work?"
    return {
        'scratch':           [],
        'recent3K':          [('user', recent_3K), ('assistant', ack_text)],
        'recent3K_filler':   [('user', filler_11K + recent_3K), ('assistant', ack_text)],
        'recent3K_earlier':  [('user', earlier_11K + recent_3K), ('assistant', ack_text)],
        'filler14K':         [('user', filler_14K), ('assistant', ack_text)],
    }


TARGETS = {
    'qwen':   ('Qwen/Qwen3-32B', 'PATH_A_QWEN3_32B'),
    'llama':  ('meta-llama/Llama-3.3-70B-Instruct', 'PATH_A_LLAMA33_70B'),
}


def run_target(target_key: str):
    """Main: load model + axis, run protocol, save per-cell data."""
    from assistant_axis import get_config, load_axis, project
    from assistant_axis.internals import ProbingModel
    from huggingface_hub import hf_hub_download
    import torch

    model_name, out_label = TARGETS[target_key]
    out_path = REPO_ROOT / f'docs/{out_label}.json'

    print(f'[Path A] Target: {model_name}')
    print(f'[Path A] Output: {out_path}')

    # 1. Load model
    print('[Path A] Loading ProbingModel...', flush=True)
    pm = ProbingModel(model_name, dtype=torch.bfloat16)
    config = get_config(model_name)
    target_layer = config['target_layer']
    print(f'[Path A] target_layer = {target_layer}, total_layers = {config["total_layers"]}')

    # 2. Load pre-computed Assistant Axis from HuggingFace
    print('[Path A] Downloading pre-computed axis from HuggingFace...', flush=True)
    axis_filename = {
        'Qwen/Qwen3-32B':                    'qwen-3-32b/assistant_axis.pt',
        'meta-llama/Llama-3.3-70B-Instruct': 'llama-3.3-70b/assistant_axis.pt',
    }[model_name]
    axis_path = hf_hub_download(
        repo_id='lu-christina/assistant-axis-vectors',
        filename=axis_filename,
        repo_type='dataset',
    )
    axis = load_axis(axis_path)
    print(f'[Path A] axis shape = {tuple(axis.shape)}')

    # 3. Build the 5 conditions (matches scripts/open_weights_content_position.py)
    conditions = build_conditions()

    # 4. Build judge client (still Sonnet 4.6 for behavioral score; runs via Anthropic API)
    if not os.environ.get('ANTHROPIC_API_KEY'):
        sys.exit('Set ANTHROPIC_API_KEY (judge runs via Anthropic API, not on this GPU)')
    from anthropic import Anthropic
    judge_client = Anthropic()

    # 5. For each (condition, probe): generate response + extract activation + project
    all_results = {}
    for cond_label, prior in conditions.items():
        print(f'\n[Path A] === {cond_label} ===', flush=True)
        cond_results = []
        for i, probe in enumerate(ALL_PROBES):
            # Build conversation messages
            messages = [{'role': 'system', 'content': DEFAULT_SYSTEM}]
            for role, text in prior:
                messages.append({'role': role, 'content': text})
            messages.append({'role': 'user',
                             'content': f'{PROBE_FRAMING}\n\n{probe.text}'})

            # Generate response with activation hook on target_layer
            response_text, activation_at_last_token = generate_with_activation(
                pm, messages, target_layer, max_new_tokens=400,
            )

            # Project onto axis
            axis_projection = project(activation_at_last_token, axis, target_layer)

            # Score behaviorally with Sonnet judge
            judgment = call_judge(judge_client, probe.text, response_text)
            score = judgment.get('score', -1)

            cond_results.append({
                'probe_id': probe.id,
                'probe_text': probe.text,
                'score': score,
                'label': judgment.get('label', ''),
                'reason': judgment.get('reason', '')[:120],
                'response_preview': response_text[:300],
                'axis_projection': float(axis_projection),
                'target_layer': target_layer,
            })
            if (i + 1) % 5 == 0:
                valid = [r['score'] for r in cond_results if r.get('score', -1) in (0, 1, 2, 3)]
                proj_vals = [r['axis_projection'] for r in cond_results]
                print(f'  [{cond_label} {i+1}/25] '
                      f'mean_score={sum(valid)/max(len(valid),1):.2f}, '
                      f'mean_proj={sum(proj_vals)/len(proj_vals):+.3f}',
                      flush=True)

        all_results[cond_label] = {
            'label': cond_label,
            'results': cond_results,
            'mean_score': (sum(r['score'] for r in cond_results
                               if r.get('score', -1) in (0, 1, 2, 3))
                           / max(sum(1 for r in cond_results
                                    if r.get('score', -1) in (0, 1, 2, 3)), 1)),
            'mean_axis_projection': (sum(r['axis_projection'] for r in cond_results)
                                     / max(len(cond_results), 1)),
        }

    out = {
        'target_model': model_name,
        'judge_model': 'claude-sonnet-4-6',
        'experiment': 'path_a_joint_behavioral_activation',
        'target_layer': target_layer,
        'axis_source': f'huggingface://lu-christina/assistant-axis-vectors/{axis_filename}',
        'per_condition': all_results,
    }
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f'\n[Path A] Wrote {out_path}')


def generate_with_activation(pm, messages, target_layer: int, max_new_tokens: int = 400):
    """Generate a response and return (response_text, activation_at_last_response_token).

    Hook the residual-stream output of target_layer. We capture the activation at
    the last generated token (after the response is complete). This is the
    model's final persona signal in residual space."""
    import torch

    # Format conversation
    formatted = pm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = pm.tokenizer(formatted, return_tensors='pt').to(pm.model.device)
    input_len = inputs['input_ids'].shape[1]

    # Storage for hooked activation
    captured_activation = {}
    layer_module = pm.model.model.layers[target_layer]

    def hook(module, args, output):
        # output is typically a tuple; first element is hidden states (B, T, H)
        hidden = output[0] if isinstance(output, tuple) else output
        # Save the LAST token of the current forward (will be overwritten each
        # forward; final value will be the last generated-token activation)
        captured_activation['hidden'] = hidden[0, -1, :].detach().cpu().float()

    handle = layer_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            output = pm.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=pm.tokenizer.pad_token_id or pm.tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    # Decode the response (everything after input prompt)
    response_ids = output[0, input_len:]
    response_text = pm.tokenizer.decode(response_ids, skip_special_tokens=True)

    return response_text, captured_activation['hidden']


def call_judge(client, probe, response, max_retries=3):
    """Sonnet 4.6 judge."""
    from anthropic import APIStatusError
    user_msg = (f'PROBE:\n{probe}\n\nRESPONSE:\n{response}\n\n'
                f'Return the JSON now.')
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model='claude-sonnet-4-6', system=JUDGE_SYSTEM_PROMPT,
                max_tokens=300,
                messages=[{'role': 'user', 'content': user_msg}],
            )
            return parse_judge(resp.content[0].text)
        except APIStatusError as e:
            if e.status_code in (429, 529) and attempt < max_retries - 1:
                time.sleep(2 * (2 ** attempt))
                continue
            raise


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else 'qwen'
    if target not in TARGETS:
        sys.exit(f'usage: path_a_joint_behavioral_activation.py [{"|".join(TARGETS)}]')
    run_target(target)


if __name__ == '__main__':
    main()
