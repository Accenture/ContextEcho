#!/usr/bin/env bash
# One-shot Spheron A100 setup + GRPO++ training launch.
#
# Usage: bash setup_spheron.sh [--opus-judge | --docker]
#
# Default: uses Opus judge (no Docker prebuild needed, cheaper to debug).
# With --docker: pre-builds SWE-bench images and runs full Docker eval.
#
# This script handles:
#   1. Python dependency install (HF + pip cache on /workspace)
#   2. HuggingFace model download (Qwen 7B)
#   3. SPRM checkpoint (already at /home/ubuntu/sprm_model if uploaded)
#   4. Optional SWE-bench Docker image prebuild
#   5. GRPO++ training launch

set -euo pipefail

MODE="${1:-}"
WORK_DIR="/ephemeral"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Spheron GRPO++ setup ==="
echo "Mode: ${MODE:-opus-judge (default)}"
echo "Script dir: $SCRIPT_DIR"
echo ""

# ── 1. Move caches to /workspace (persists across reboots, avoids disk full) ──
mkdir -p "$WORK_DIR/pip_cache" "$WORK_DIR/hf_cache"
export PIP_CACHE_DIR="$WORK_DIR/pip_cache"
export HF_HOME="$WORK_DIR/hf_cache"
export TRANSFORMERS_CACHE="$WORK_DIR/hf_cache"

# ── 2. Install deps ───────────────────────────────────────────────────────────
echo "Installing Python dependencies..."

# Core ML stack
pip install --cache-dir "$WORK_DIR/pip_cache" -q \
    torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install --cache-dir "$WORK_DIR/pip_cache" -q \
    transformers>=4.45.0 \
    trl>=0.22.0 \
    peft>=0.14.0 \
    bitsandbytes>=0.44.0 \
    accelerate>=1.0.0 \
    datasets>=3.0.0

# Tokenizer extras Qwen needs
pip install --cache-dir "$WORK_DIR/pip_cache" -q \
    tiktoken sentencepiece

# Jinja2 must be >= 3.1.0 for transformers chat template
pip install --cache-dir "$WORK_DIR/pip_cache" -q "jinja2>=3.1.0"

# SWE-bench (for dataset + Docker harness if needed)
pip install --cache-dir "$WORK_DIR/pip_cache" -q swebench

# Opus judge
pip install --cache-dir "$WORK_DIR/pip_cache" -q anthropic

echo "Dependencies installed."

# ── 3. Verify CUDA ────────────────────────────────────────────────────────────
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# ── 4. Pre-download Qwen model weights ───────────────────────────────────────
echo "Downloading Qwen2.5-Coder-7B-Instruct..."
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'Qwen/Qwen2.5-Coder-7B-Instruct'
print('  Downloading tokenizer...')
AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print('  Downloading model weights (this takes ~5 min on A100)...')
AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=True)
print('  Model cached.')
"

# ── 5. Verify SPRM checkpoint ─────────────────────────────────────────────────
SPRM_PATH="/home/ubuntu/sprm_model"
if [ ! -f "$SPRM_PATH/model.safetensors" ]; then
    echo "ERROR: SPRM checkpoint not found at $SPRM_PATH"
    echo "Upload it first: scp -r experiments/oracle_trajectory/sprm_model ubuntu@<ip>:/home/ubuntu/sprm_model"
    exit 1
fi
echo "SPRM checkpoint found at $SPRM_PATH"

# Test SPRM loads
python3 -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from train_grpo_sprm import SPRMScorer
s = SPRMScorer('$SPRM_PATH')
w = s.mean_waste(['bash', 'str_replace_editor', 'submit'])
print(f'  SPRM test: waste={w:.3f} (should be < 0.5 for efficient tools)')
"

# ── 6. Run end-to-end smoke test ─────────────────────────────────────────────
echo "Running pipeline smoke test..."
cd "$SCRIPT_DIR"
python3 test_e2e_pipeline.py --sprm-checkpoint "$SPRM_PATH"

# ── 7. Optional: SWE-bench Docker prebuild ────────────────────────────────────
if [ "$MODE" = "--docker" ]; then
    echo ""
    echo "=== Docker prebuild mode ==="
    echo "Building SWE-bench environment images (18 images, ~2 min each = ~36 min)..."
    python3 -c "
import subprocess, json
from datasets import load_dataset
from swebench.harness.run_evaluation import main as run_eval

ds = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
instances = list(ds)[:50]

# Write predictions with dummy patch to trigger Docker image prebuild
preds = [{'instance_id': r['instance_id'], 'model_patch': '# dummy', 'model_name_or_path': 'prebuild'} for r in instances]

import tempfile, json
with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
    json.dump(preds, f)
    pred_path = f.name

run_eval([
    '--dataset_name', 'princeton-nlp/SWE-bench_Verified',
    '--predictions_path', pred_path,
    '--max_workers', '4',
    '--cache_level', 'env',
    '--run_id', 'prebuild',
])
print('Docker prebuild complete.')
"
fi

# ── 8. Launch GRPO++ training ─────────────────────────────────────────────────
echo ""
echo "=== Launching GRPO++ training ==="

LAUNCH_ARGS="--model Qwen/Qwen2.5-Coder-7B-Instruct \
    --sprm-checkpoint $SPRM_PATH \
    --num-tasks 50 \
    --num-generations 4 \
    --alpha 0.1 \
    --epochs 1 \
    --max-completion-length 2048 \
    --output-dir $WORK_DIR/grpo_swe_7b_poc \
    --log-dir $WORK_DIR/grpo_logs \
    --seed 42"

if [ "$MODE" = "--docker" ]; then
    echo "Training with Docker execution..."
    python3 "$SCRIPT_DIR/train_grpo_sprm.py" $LAUNCH_ARGS
else
    echo "Training with Opus judge..."
    if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        echo "ERROR: ANTHROPIC_API_KEY not set. Export it first."
        exit 1
    fi
    python3 "$SCRIPT_DIR/train_grpo_sprm.py" $LAUNCH_ARGS --use-opus-judge
fi
