#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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
    for candidate in /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3 "$(command -v python3 2>/dev/null || true)"; do
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
  echo "ERROR: python3 is required to bootstrap the ContextEcho maintainer intake." >&2
  echo "Install Python 3, then rerun this command." >&2
  exit 1
fi

if [[ "$host_os" == "Darwin" && "$host_arch" == "arm64" ]]; then
  py_arch="$(python_arch "$python3_bin")"
  if [[ "$py_arch" == "arm64" || "$py_arch" == "arm64e" ]]; then
    cache_suffix="-arm64"
    export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
  fi
fi

CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/contextecho-maintainer${cache_suffix}"
UV_VENV="$CACHE_ROOT/uv-venv"
UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv-cache}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$CACHE_ROOT/uv-python}"
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

PYTHON_VERSION="${CONTEXTECHO_PYTHON_VERSION:-3.14}"
VENV="${CONTEXTECHO_MAINTAINER_VENV:-.venv}"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]] || ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 14) else 1)' >/dev/null 2>&1; then
  echo "[ContextEcho] creating maintainer environment with Python $PYTHON_VERSION..."
  "${UV_RUN[@]}" venv --python "$PYTHON_VERSION" --managed-python "$VENV"
fi

echo "[ContextEcho] installing maintainer dependencies..."
"${UV_RUN[@]}" pip install --python "$PY" -r requirements-maintainer.txt

intake_args=(--run-quick --promote)
if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
  intake_args+=(--skip-download)
fi
if [[ "${INCLUDE_PROMOTED:-0}" == "1" ]]; then
  intake_args+=(--include-promoted)
fi
if [[ "${INCLUDE_REVIEWED:-0}" == "1" ]]; then
  intake_args+=(--include-reviewed)
fi
if [[ "${INCLUDE_DUPLICATES:-0}" == "1" ]]; then
  intake_args+=(--include-duplicates)
fi
if [[ -n "${DATASET_ROOT:-}" ]]; then
  intake_args+=(--dataset-root "$DATASET_ROOT")
fi
if [[ -n "${STAGING_DIR:-}" ]]; then
  intake_args+=(--staging-dir "$STAGING_DIR")
fi

echo "[ContextEcho] running full maintainer intake: download, duplicate/lineage check, review, quick validation, promote..."
"$PY" scripts/intake_donations.py "${intake_args[@]}"

backfill_args=()
if [[ -n "${DATASET_ROOT:-}" ]]; then
  backfill_args+=(--dataset-root "$DATASET_ROOT")
fi
if [[ -n "${STAGING_DIR:-}" ]]; then
  backfill_args+=(--staging-dir "$STAGING_DIR")
fi

echo "[ContextEcho] backfilling quick validation for promoted sessions missing validation output..."
"$PY" scripts/backfill_promoted_validation.py "${backfill_args[@]}"

echo "[ContextEcho] updating public release metadata..."
"$PY" scripts/update_project_stats.py --allow-offline
"$PY" scripts/update_contributors.py
"$PY" scripts/update_dataset_card.py

echo "[ContextEcho] checking public release metadata..."
"$PY" scripts/update_contributors.py --check
"$PY" scripts/update_dataset_card.py --check

echo "[ContextEcho] maintainer intake complete."
