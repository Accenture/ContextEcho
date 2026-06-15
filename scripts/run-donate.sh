#!/usr/bin/env bash
set -euo pipefail

SPEC="${CONTEXTECHO_DONATE_SPEC:-git+https://github.com/Accenture/ContextEcho.git}"
CONTEXTECHO_RELAY_URL="${CONTEXTECHO_RELAY_URL:-https://contextecho2026-context-echo-donation-relay.hf.space}"

host_os="$(uname -s)"
host_arch="$(uname -m)"
cache_suffix=""

python_arch() {
  "$1" -c 'import platform; print(platform.machine())' 2>/dev/null || true
}

binary_arch() {
  file "$1" 2>/dev/null || true
}

choose_python() {
  if [[ "$host_os" == "Darwin" && "$host_arch" == "arm64" ]]; then
    for candidate in /opt/homebrew/bin/python3 /usr/bin/python3 "$(command -v python3 2>/dev/null || true)"; do
      [[ -n "$candidate" && -x "$candidate" ]] || continue
      case "$(python_arch "$candidate")" in
        arm64|arm64e) printf '%s\n' "$candidate"; return 0 ;;
      esac
    done
  fi

  command -v python3 2>/dev/null || true
}

python3_bin="$(choose_python)"
if [[ -z "$python3_bin" ]]; then
  echo "ERROR: python3 is required to run the ContextEcho donation wizard." >&2
  echo "Install Python 3, then rerun this command." >&2
  exit 1
fi

if [[ "$host_os" == "Darwin" && "$host_arch" == "arm64" ]]; then
  py_arch="$(python_arch "$python3_bin")"
  if [[ "$py_arch" == "arm64" || "$py_arch" == "arm64e" ]]; then
    cache_suffix="-arm64"
    export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
  fi
fi

CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/contextecho-donate${cache_suffix}"
UV_VENV="$CACHE_ROOT/uv-venv"
UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv-cache}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$CACHE_ROOT/uv-python}"
export CONTEXTECHO_RELAY_URL
export UV_CACHE_DIR
export UV_PYTHON_INSTALL_DIR

use_system_uv=0
if command -v uv >/dev/null 2>&1; then
  uv_bin="$(command -v uv)"
  use_system_uv=1
  if [[ "$host_os" == "Darwin" && "$host_arch" == "arm64" ]]; then
    uv_desc="$(binary_arch "$uv_bin")"
    if [[ "$uv_desc" == *"x86_64"* && "$uv_desc" != *"arm64"* ]]; then
      use_system_uv=0
      echo "[ContextEcho] ignoring x86_64 uv on Apple Silicon; using native ARM bootstrap."
    fi
  fi
fi

if [[ "$use_system_uv" == "1" ]]; then
  UV_RUN=(uv)
else
  echo "[ContextEcho] uv not found for this architecture; creating a private bootstrap environment..."
  "$python3_bin" -m venv "$UV_VENV"
  "$UV_VENV/bin/python" -m pip install --upgrade pip uv
  UV_RUN=("$UV_VENV/bin/uv")
fi

echo "[ContextEcho] starting local donation wizard..."
"${UV_RUN[@]}" run --refresh --no-project --python 3.11 --managed-python --with "$SPEC" contextecho-donate "$@"
