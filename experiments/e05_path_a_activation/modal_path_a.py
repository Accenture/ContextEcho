"""Modal entrypoint for Path A — joint behavioral + activation experiments
on Qwen 3 32B + Llama 3.3 70B using Lu et al.'s assistant-axis package.

Usage from local Mac:
  modal run infra/modal_path_a.py::run_qwen
  modal run infra/modal_path_a.py::run_llama
  modal run infra/modal_path_a.py::smoke_test_qwen

Cost estimate (H100 SXM at $4.56/hr):
  smoke_test:    ~10 min (model load + 5 axis-projection calls)  -> ~$0.80
  run_qwen:      ~3-4 hours (5 conds × 25 probes + axis loads)   -> ~$15-20
  run_llama:     ~5-6 hours (70B is slower)                       -> ~$25-30
"""
import modal


# Container image: PyTorch + transformers + Lu et al.'s assistant-axis package
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.46.0",
        "accelerate>=1.0.0",
        "huggingface_hub",
        "anthropic>=0.40.0",
        "sentencepiece",
    )
    # Install Lu et al.'s assistant-axis from GitHub (master branch)
    .run_commands(
        "git clone https://github.com/safety-research/assistant-axis.git /opt/assistant-axis",
        "cd /opt/assistant-axis && pip install -e .",
    )
    # Add our project files. Ignore data/ (212 MB of legacy Tier-1 seed
    # dirs) and paper/ (PDF + LaTeX, irrelevant to GPU). Then add the
    # single transcript file we actually need for build_conditions().
    .add_local_dir(
        local_path="<REPO_ROOT>",
        remote_path="/root/persona_drift",
        ignore=["data/**", "paper/**", "**/__pycache__/**", ".git/**"],
    )
    .add_local_file(
        local_path="<REPO_ROOT>/data/session_raw_transcript.jsonl",
        remote_path="/root/persona_drift/data/session_raw_transcript.jsonl",
    )
)

app = modal.App("persona-drift-path-a", image=image)

# Secrets: Anthropic key for the Sonnet judge calls; HF token for gated weights
SECRETS = [
    modal.Secret.from_name("anthropic-api-key"),
    modal.Secret.from_name("huggingface-token"),
]

# H100 SXM 80GB; 1 GPU is enough for Qwen 32B (with bf16); Llama 70B may need 2
GPU_QWEN = "H100"
GPU_LLAMA = "H100:2"  # 2x H100 to fit Llama 70B in bf16


@app.function(image=image, gpu=GPU_QWEN, secrets=SECRETS, timeout=60 * 60 * 6)
def smoke_test_qwen():
    """Quick verification: load Qwen 3 32B, load axis, project on 5 known
    persona prompts. Expected: 'Default Assistant' prompt should yield
    higher axis projection than 'Pirate' prompt."""
    import os, sys
    sys.path.insert(0, "/root/persona_drift")
    sys.path.insert(0, "/opt/assistant-axis")
    from assistant_axis import get_config, load_axis, project
    from assistant_axis.internals import ProbingModel
    from huggingface_hub import hf_hub_download
    import torch

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    model_name = "Qwen/Qwen3-32B"
    print(f"[smoke] Loading {model_name}...", flush=True)
    pm = ProbingModel(model_name, dtype=torch.bfloat16)
    config = get_config(model_name)
    target_layer = config["target_layer"]
    print(f"[smoke] target_layer = {target_layer}", flush=True)

    print("[smoke] Downloading pre-computed axis...", flush=True)
    axis_path = hf_hub_download(
        repo_id="lu-christina/assistant-axis-vectors",
        filename="qwen-3-32b/assistant_axis.pt",
        repo_type="dataset",
    )
    axis = load_axis(axis_path)
    print(f"[smoke] axis shape = {tuple(axis.shape)}", flush=True)

    # 5 test prompts spanning persona space
    test_prompts = [
        ("default",  "You are a helpful AI assistant."),
        ("pirate",   "You are a salty pirate captain."),
        ("therapist","You are a warm therapist who listens deeply."),
        ("default2", "You are an AI assistant created to help with tasks."),
        ("villain",  "You are a brilliant villain plotting world domination."),
    ]
    for name, sys_prompt in test_prompts:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "Briefly describe yourself in one sentence."},
        ]
        formatted = pm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = pm.tokenizer(formatted, return_tensors="pt").to(pm.model.device)
        captured = {}
        layer_module = pm.model.model.layers[target_layer]

        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["hidden"] = hidden[0, -1, :].detach().cpu().float()

        handle = layer_module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                pm.model(**inputs)  # one forward pass; we want activation at last input token
        finally:
            handle.remove()

        proj = project(captured["hidden"], axis, target_layer)
        print(f"[smoke]   {name:12s}: axis_projection = {proj:+.4f}", flush=True)

    print("[smoke] Done. If 'default' projections > 'pirate'/'villain', pipeline is healthy.", flush=True)


@app.function(image=image, gpu=GPU_QWEN, secrets=SECRETS, timeout=60 * 60 * 8)
def run_qwen():
    """Full Path A on Qwen 3 32B: 5 conditions × 25 probes with activation
    projection at the target layer for each (condition, probe) cell."""
    import os, sys, json
    sys.path.insert(0, "/root/persona_drift")
    os.chdir("/root/persona_drift")
    # Run our local script via subprocess-like exec
    import subprocess
    result = subprocess.run(
        ["python3", "scripts/path_a_joint_behavioral_activation.py", "qwen"],
        capture_output=True, text=True, env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/root/persona_drift:/opt/assistant-axis",
        },
    )
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr, flush=True)
        raise RuntimeError("path_a_joint_behavioral_activation.py qwen failed")
    # Read back the result and return it (Modal will pass back to local for saving)
    with open("/root/persona_drift/docs/PATH_A_QWEN3_32B.json") as f:
        return json.load(f)


@app.function(image=image, gpu=GPU_LLAMA, secrets=SECRETS, timeout=60 * 60 * 10)
def run_llama():
    """Full Path A on Llama 3.3 70B Instruct."""
    import os, sys, json
    sys.path.insert(0, "/root/persona_drift")
    os.chdir("/root/persona_drift")
    import subprocess
    result = subprocess.run(
        ["python3", "scripts/path_a_joint_behavioral_activation.py", "llama"],
        capture_output=True, text=True, env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/root/persona_drift:/opt/assistant-axis",
        },
    )
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr, flush=True)
        raise RuntimeError("path_a_joint_behavioral_activation.py llama failed")
    with open("/root/persona_drift/docs/PATH_A_LLAMA33_70B.json") as f:
        return json.load(f)


@app.function(image=image, gpu=GPU_QWEN, secrets=SECRETS, timeout=60 * 60 * 4)
def run_qwen_steering():
    """Path Z: activation-steering causal test on Qwen 3 32B.
    5 conditions (scratch + 4 alpha values on recent3K) × 25 probes."""
    import os, sys, json
    sys.path.insert(0, "/root/persona_drift")
    os.chdir("/root/persona_drift")
    import subprocess
    result = subprocess.run(
        ["python3", "scripts/path_z_steering.py"],
        capture_output=True, text=True, env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/root/persona_drift:/opt/assistant-axis",
        },
    )
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr, flush=True)
        raise RuntimeError("path_z_steering.py failed")
    with open("/root/persona_drift/docs/PATH_Z_QWEN3_32B.json") as f:
        return json.load(f)


@app.local_entrypoint()
def smoke():
    """Run smoke test from local: `modal run infra/modal_path_a.py::smoke`"""
    smoke_test_qwen.remote()


@app.local_entrypoint()
def qwen():
    """Full Qwen 3 32B run: `modal run infra/modal_path_a.py::qwen`"""
    result = run_qwen.remote()
    # Save the returned JSON to local docs/
    from pathlib import Path
    out_path = Path(__file__).resolve().parent.parent / "docs/PATH_A_QWEN3_32B.json"
    out_path.write_text(__import__("json").dumps(result, indent=2, default=str))
    print(f"Saved to {out_path}")


@app.local_entrypoint()
def llama():
    """Full Llama 3.3 70B run: `modal run infra/modal_path_a.py::llama`"""
    result = run_llama.remote()
    from pathlib import Path
    out_path = Path(__file__).resolve().parent.parent / "docs/PATH_A_LLAMA33_70B.json"
    out_path.write_text(__import__("json").dumps(result, indent=2, default=str))
    print(f"Saved to {out_path}")


@app.local_entrypoint()
def qwen_steering():
    """Path Z steering test: `modal run infra/modal_path_a.py::qwen_steering`"""
    result = run_qwen_steering.remote()
    from pathlib import Path
    out_path = Path(__file__).resolve().parent.parent / "docs/PATH_Z_QWEN3_32B.json"
    out_path.write_text(__import__("json").dumps(result, indent=2, default=str))
    print(f"Saved to {out_path}")
