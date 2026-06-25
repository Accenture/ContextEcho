#!/usr/bin/env bash
set -u

SPEC="${CONTEXTECHO_DONATE_SPEC:-git+https://github.com/Accenture/ContextEcho.git}"
CONTEXTECHO_RELAY_URL="${CONTEXTECHO_RELAY_URL:-https://contextecho2026-context-echo-donation-relay.hf.space}"
if [[ -n "${CONTEXTECHO_DONATE_PYTHON:-}" ]]; then
  CONTEXTECHO_DONATE_PYTHONS="$CONTEXTECHO_DONATE_PYTHON"
else
  CONTEXTECHO_DONATE_PYTHONS="${CONTEXTECHO_DONATE_PYTHONS:-3.12 3.11 3.13 3.10}"
fi

host_os="$(uname -s)"
host_arch="$(uname -m)"
cache_suffix=""

python_arch() {
  "$1" -c 'import platform; print(platform.machine())' 2>/dev/null || true
}

binary_arch() {
  file "$1" 2>/dev/null || true
}

explain_failure() {
  rc="$1"
  echo "" >&2
  echo "[ContextEcho] The local donation wizard did not start." >&2
  echo "[ContextEcho] What to try next:" >&2
  echo "  1. Check that this machine can reach GitHub and PyPI." >&2
  echo "  2. If your company blocks Python installs, ask IT to allow one of: ${CONTEXTECHO_DONATE_PYTHONS}." >&2
  echo "  3. On Apple Silicon, make sure Python and uv are arm64, not x86_64/Rosetta." >&2
  echo "  4. Rerun the same command; ContextEcho reuses its private cache." >&2
  echo "" >&2
  echo "[ContextEcho] Debug details:" >&2
  echo "  OS/arch: ${host_os}/${host_arch}" >&2
  echo "  python: ${python3_bin:-not found}" >&2
  echo "  wizard python candidates: ${CONTEXTECHO_DONATE_PYTHONS}" >&2
  if [[ -n "${python3_bin:-}" ]]; then
    echo "  python arch: $(python_arch "$python3_bin")" >&2
  fi
  echo "  cache: ${CACHE_ROOT:-not created}" >&2
  echo "  exit code: $rc" >&2
}

choose_python() {
  if [[ "$host_os" == "Darwin" && "$host_arch" == "arm64" ]]; then
    for candidate in /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.10 /opt/homebrew/bin/python3 /usr/bin/python3 "$(command -v python3.12 2>/dev/null || true)" "$(command -v python3.11 2>/dev/null || true)" "$(command -v python3.13 2>/dev/null || true)" "$(command -v python3.10 2>/dev/null || true)" "$(command -v python3 2>/dev/null || true)"; do
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

if ! "$python3_bin" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 8) else 1)
PY
then
  echo "ERROR: ContextEcho needs Python 3.8+ to bootstrap the local wizard." >&2
  echo "Found: $("$python3_bin" --version 2>&1)" >&2
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
  rc=$?
  if [[ "$rc" != "0" ]]; then
    explain_failure "$rc"
    exit "$rc"
  fi
  "$UV_VENV/bin/python" -m pip install --upgrade pip uv
  rc=$?
  if [[ "$rc" != "0" ]]; then
    explain_failure "$rc"
    exit "$rc"
  fi
  UV_RUN=("$UV_VENV/bin/uv")
fi

echo "[ContextEcho] starting local donation wizard..."
echo "[ContextEcho] raw sessions stay on this machine; the browser wizard will open automatically."

last_rc=1
for py_version in $CONTEXTECHO_DONATE_PYTHONS; do
  echo "[ContextEcho] trying Python ${py_version} for the local wizard..."
  UV_ARGS=(run --no-project --python "$py_version" --with "$SPEC")
  if [[ "${CONTEXTECHO_DONATE_REFRESH:-1}" != "0" ]]; then
    UV_ARGS=(run --refresh --no-project --python "$py_version" --with "$SPEC")
  fi
  "${UV_RUN[@]}" "${UV_ARGS[@]}" contextecho-donate "$@"
  rc=$?
  if [[ "$rc" == "0" ]]; then
    exit 0
  fi
  if [[ "$rc" == "130" ]]; then
    exit "$rc"
  fi
  last_rc="$rc"
  echo "[ContextEcho] Python ${py_version} did not start the wizard; trying the next supported runtime if available." >&2
done

explain_failure "$last_rc"
exit "$last_rc"
