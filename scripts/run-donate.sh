#!/usr/bin/env bash
set -euo pipefail

SPEC="${CONTEXTECHO_DONATE_SPEC:-git+https://github.com/Accenture/ContextEcho.git}"
CONTEXTECHO_RELAY_URL="${CONTEXTECHO_RELAY_URL:-https://contextecho2026-context-echo-donation-relay.hf.space}"
CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/contextecho-donate"
UV_VENV="$CACHE_ROOT/uv-venv"
UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv-cache}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$CACHE_ROOT/uv-python}"
export CONTEXTECHO_RELAY_URL
export UV_CACHE_DIR
export UV_PYTHON_INSTALL_DIR

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required to run the ContextEcho donation wizard." >&2
  echo "Install Python 3, then rerun this command." >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  UV_RUN=(uv)
else
  echo "[ContextEcho] uv not found; creating a private bootstrap environment..."
  python3 -m venv "$UV_VENV"
  "$UV_VENV/bin/python" -m pip install --upgrade pip uv
  UV_RUN=("$UV_VENV/bin/uv")
fi

echo "[ContextEcho] starting local donation wizard..."
"${UV_RUN[@]}" tool run --refresh --python 3.11 --managed-python --from "$SPEC" contextecho-donate
