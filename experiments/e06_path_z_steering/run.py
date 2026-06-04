"""Path Z: Activation-steering causal test on Qwen 3 32B (locked 2026-04-25).

Pre-registration: docs/PREREG_STEERING.md
SHA-256: f213da3b11fda8ba558e3250e638f0b86c78b385deb7a86e8516a34e40432778

For each (alpha_coef, condition, probe) cell:
  1. Build conversation per Path A protocol (recent3K or scratch).
  2. Register a forward hook on layer 32 that ADDS alpha * v to the
     residual stream output, where
       v = -(p_recent3K - p_scratch) * a_hat = +8.65 * a_hat
     and a_hat is the unit-normalized Assistant Axis at layer 32 (from
     Lu et al.'s released axis file, slice [target_layer, :]).
  3. Generate target response with steering applied.
  4. After generation, capture POST-steering activation at last token
     (sanity check: should be approximately at scratch baseline at alpha=1.0).
  5. Project post-steering activation onto axis (verifies steering moved
     the projection in the expected direction).
  6. Score response with Sonnet 4.6 judge.

Conditions:
  alpha=0    on recent3K  (sanity: should reproduce Path A recent3K)
  alpha=0.5  on recent3K  (half-strength)
  alpha=1.0  on recent3K  (full subtraction; primary test)
  alpha=1.5  on recent3K  (over-correction)
  scratch    no steering  (behavioral baseline reference)

Outputs:
  docs/PATH_Z_QWEN3_32B.json
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

from analyze_length_control import (
    load_events, extract_verbatim_slice, parse_judge,
    PROBE_FRAMING, DEFAULT_SYSTEM,
)
from harness.probes import ALL_PROBES
from harness.judge import JUDGE_SYSTEM_PROMPT


FILLER_TEMPLATE = (
    "The following is filler placeholder content for an experimental "
    "control. Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "The quick brown fox jumps over the lazy dog. "
)


def make_filler(target_chars: int) -> str:
    chunks = []
    total = 0
    while total < target_chars:
        chunks.append(FILLER_TEMPLATE)
        total += len(FILLER_TEMPLATE)
    return "".join(chunks)[:target_chars]


def build_conditions():
    """Return {scratch: [], recent3K: [...]} same as Path A."""
    events = load_events()
    boundaries = [i for i, e in enumerate(events)
                  if e.get('type') == 'system' and e.get('subtype') == 'compact_boundary']
    target_boundary = boundaries[-1]
    verbatim_full = extract_verbatim_slice(events, target_boundary, 28000)
    recent_3K = verbatim_full[-3000:]
    ack_text = "Acknowledged. How can I help continue this work?"
    return {
        'scratch':  [],
        'recent3K': [('user', recent_3K), ('assistant', ack_text)],
    }


# Steering grid (LOCKED in pre-reg)
ALPHA_GRID = [0.0, 0.5, 1.0, 1.5]
# Steering vector magnitude from Path A Qwen 3 32B (LOCKED at pre-reg time):
#   p_recent3K - p_scratch = -19.77 - (-11.12) = -8.65
#   v = -(p_recent3K - p_scratch) * a_hat = +8.65 * a_hat
STEER_MAG = 8.65


def run_path_z():
    """Main: load Qwen 3 32B + axis, run protocol with steering hook."""
    from assistant_axis import get_config, load_axis, project
    from assistant_axis.internals import ProbingModel
    from huggingface_hub import hf_hub_download
    import torch

    model_name = 'Qwen/Qwen3-32B'
    out_path = REPO_ROOT / 'docs/PATH_Z_QWEN3_32B.json'

    print(f'[Path Z] Target: {model_name}', flush=True)
    print(f'[Path Z] Output: {out_path}', flush=True)
    print(f'[Path Z] Alpha grid: {ALPHA_GRID}', flush=True)
    print(f'[Path Z] Steering magnitude (locked): {STEER_MAG}', flush=True)

    # Load model
    print('[Path Z] Loading ProbingModel...', flush=True)
    pm = ProbingModel(model_name, dtype=torch.bfloat16)
    config = get_config(model_name)
    target_layer = config['target_layer']
    print(f'[Path Z] target_layer = {target_layer}', flush=True)

    # Load axis
    print('[Path Z] Downloading pre-computed axis...', flush=True)
    axis_path = hf_hub_download(
        repo_id='lu-christina/assistant-axis-vectors',
        filename='qwen-3-32b/assistant_axis.pt',
        repo_type='dataset',
    )
    axis = load_axis(axis_path)  # shape: (n_layers, hidden_dim)
    print(f'[Path Z] axis shape = {tuple(axis.shape)}', flush=True)

    # Compute unit-normalized axis at target layer
    a_layer = axis[target_layer].to(pm.model.device).to(torch.bfloat16)
    a_hat = a_layer / a_layer.norm()
    print(f'[Path Z] axis at layer {target_layer}: norm = {a_layer.norm():.4f}, '
          f'hidden_dim = {a_hat.shape[0]}', flush=True)

    # Steering vector v = +STEER_MAG * a_hat
    v = STEER_MAG * a_hat  # in bf16 on device

    # Conditions
    conditions = build_conditions()

    # Judge client
    if not os.environ.get('ANTHROPIC_API_KEY'):
        sys.exit('Set ANTHROPIC_API_KEY for Sonnet judge')
    from anthropic import Anthropic
    judge_client = Anthropic()

    all_results = {}

    # Experiment grid: scratch (alpha=0, baseline) + 4 alphas on recent3K
    cells = [('scratch', 0.0)] + [('recent3K', a) for a in ALPHA_GRID]

    for cond_label, alpha in cells:
        cell_label = f'{cond_label}_alpha{alpha}'
        print(f'\n[Path Z] === {cell_label} ===', flush=True)
        prior = conditions[cond_label]
        cond_results = []

        for i, probe in enumerate(ALL_PROBES):
            messages = [{'role': 'system', 'content': DEFAULT_SYSTEM}]
            for role, text in prior:
                messages.append({'role': role, 'content': text})
            messages.append({'role': 'user',
                             'content': f'{PROBE_FRAMING}\n\n{probe.text}'})

            response_text, post_steer_activation = generate_with_steering(
                pm, messages, target_layer, v, alpha, max_new_tokens=400,
            )
            post_steer_proj = project(post_steer_activation, axis, target_layer)

            # Score
            try:
                judgment = call_judge(judge_client, probe.text, response_text)
                score = judgment.get('score', -1)
                label = judgment.get('label', '')
                reason = judgment.get('reason', '')[:120]
            except Exception as e:
                score, label, reason = -1, '', f'judge_err:{str(e)[:80]}'

            cond_results.append({
                'probe_id': probe.id,
                'probe_text': probe.text,
                'score': score,
                'label': label,
                'reason': reason,
                'response_preview': (response_text or '')[:300],
                'post_steer_projection': float(post_steer_proj),
                'alpha': alpha,
                'condition': cond_label,
            })
            if (i + 1) % 5 == 0:
                valid = [r['score'] for r in cond_results
                         if r.get('score', -1) in (0, 1, 2, 3)]
                proj_vals = [r['post_steer_projection'] for r in cond_results]
                print(f'  [{cell_label} {i+1}/25] '
                      f'mean_score={sum(valid)/max(len(valid),1):.2f}, '
                      f'mean_proj={sum(proj_vals)/len(proj_vals):+.3f}',
                      flush=True)

        all_results[cell_label] = {
            'condition': cond_label,
            'alpha': alpha,
            'results': cond_results,
            'mean_score': (sum(r['score'] for r in cond_results
                               if r.get('score', -1) in (0, 1, 2, 3))
                           / max(sum(1 for r in cond_results
                                    if r.get('score', -1) in (0, 1, 2, 3)), 1)),
            'mean_post_steer_projection':
                sum(r['post_steer_projection'] for r in cond_results)
                / max(len(cond_results), 1),
        }

    out = {
        'target_model': model_name,
        'judge_model': 'claude-sonnet-4-6',
        'experiment': 'path_z_steering',
        'prereg_sha256': 'f213da3b11fda8ba558e3250e638f0b86c78b385deb7a86e8516a34e40432778',
        'target_layer': target_layer,
        'steering_magnitude': STEER_MAG,
        'alpha_grid': ALPHA_GRID,
        'per_cell': all_results,
    }
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f'\n[Path Z] Wrote {out_path}', flush=True)


def generate_with_steering(pm, messages, target_layer: int, v, alpha: float,
                           max_new_tokens: int = 400):
    """Generate a response with steering hook adding alpha*v to layer output.

    The hook fires on EVERY forward pass (every generation step), adding the
    fixed steering vector to the last-token residual stream output.
    """
    import torch

    formatted = pm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = pm.tokenizer(formatted, return_tensors='pt').to(pm.model.device)
    input_len = inputs['input_ids'].shape[1]

    captured_activation = {}
    layer_module = pm.model.model.layers[target_layer]

    def steering_hook(module, args, output):
        """Forward hook: ADD alpha*v to the layer's hidden state output.

        Modifies the residual stream in-place (well, returns a modified copy
        as the new output). Also captures the post-steering last-token
        activation for sanity logging.
        """
        # output is typically (hidden_states, ...) tuple
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = ()

        if alpha != 0.0:
            # Add steering: hidden has shape (B, T, H); add alpha*v broadcast
            # to all tokens. v has shape (H,), broadcasts naturally.
            hidden = hidden + (alpha * v).to(hidden.dtype).to(hidden.device)

        # Capture post-steering activation at last token (for sanity check)
        captured_activation['hidden'] = hidden[0, -1, :].detach().cpu().float()

        if rest:
            return (hidden,) + rest
        return hidden

    handle = layer_module.register_forward_hook(steering_hook)
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

    response_ids = output[0, input_len:]
    response_text = pm.tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text, captured_activation['hidden']


def call_judge(client, probe, response, max_retries=3):
    from anthropic import APIStatusError
    user_msg = f'PROBE:\n{probe}\n\nRESPONSE:\n{response}\n\nReturn the JSON now.'
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


if __name__ == '__main__':
    run_path_z()
