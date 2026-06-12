#!/usr/bin/env bash
set -euo pipefail

SPEC="${CONTEXTECHO_DONATE_SPEC:-git+https://github.com/Accenture/ContextEcho.git}"
CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/contextecho-donate"
PIPX_VENV="$CACHE_ROOT/pipx-venv"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required to run the ContextEcho donation wizard." >&2
  echo "Install Python 3, then rerun this command." >&2
  exit 1
fi

if python3 -m pipx --version >/dev/null 2>&1; then
  PIPX_RUN=(python3 -m pipx)
elif command -v pipx >/dev/null 2>&1; then
  PIPX_RUN=(pipx)
else
  echo "[ContextEcho] pipx not found; creating a private bootstrap environment..."
  python3 -m venv "$PIPX_VENV"
  "$PIPX_VENV/bin/python" -m pip install --upgrade pip pipx
  PIPX_RUN=("$PIPX_VENV/bin/python" -m pipx)
fi

echo "[ContextEcho] starting local donation wizard..."
"${PIPX_RUN[@]}" run --no-cache --spec "$SPEC" contextecho-donate
