#!/usr/bin/env bash

# Source this file after selecting a Python environment and an NVMe scratch
# directory. It never selects GPUs, loads modules, or resets PATH/compilers.

_TUTTI_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${TUTTI_PROFILE_ROOT:?Set TUTTI_PROFILE_ROOT to a scratch directory on NVMe}"
if [[ "$TUTTI_PROFILE_ROOT" != /* ]]; then
    echo "TUTTI_PROFILE_ROOT must be an absolute path" >&2
    return 1
fi
_TUTTI_PROFILE_ROOT="$TUTTI_PROFILE_ROOT"
_TUTTI_PYTHON="${TUTTI_PYTHON:-${PYTHON_BIN:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}}"
if [[ -z "$_TUTTI_PYTHON" ]]; then
    _TUTTI_PYTHON="$(command -v python3 || true)"
fi
if ! command -v "$_TUTTI_PYTHON" >/dev/null 2>&1; then
    echo "Python interpreter not found: $_TUTTI_PYTHON" >&2
    return 1
fi
export TUTTI_PROFILE_ROOT
export TMPDIR="$_TUTTI_PROFILE_ROOT/tmp"
export XDG_CACHE_HOME="$_TUTTI_PROFILE_ROOT/cache/xdg"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$VLLM_CACHE_ROOT/flashinfer_autotune_cache"
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME"
export FLASHINFER_CUBIN_DIR="$XDG_CACHE_HOME/flashinfer-cubins"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"

# NVTX ranges are what make a Tutti nsys report readable: without them the
# storage work is an undifferentiated band of CUDA API calls.
export TUTTI_NVTX="${TUTTI_NVTX:-1}"

# Three separate source roots, all required:
#   repo root          the `tutti` Python package
#   csrc/python/src    the compiled `tutti_runtime` extension (control plane)
#   csrc/kv_transfer   the compiled `tutti_kv_transfer` extension (data plane)
# Omitting the last one fails only inside the workers, as
# "No module named 'tutti_kv_transfer'" behind a generic
# "Engine core initialization failed".
export PYTHONPATH="$_TUTTI_REPO/csrc/python/src:$_TUTTI_REPO/csrc/kv_transfer:$_TUTTI_REPO${PYTHONPATH:+:$PYTHONPATH}"
export TUTTI_PYTHON="$_TUTTI_PYTHON"

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" \
  "$TUTTI_PROFILE_ROOT/logs" "$TUTTI_PROFILE_ROOT/reports"

unset _TUTTI_PROFILE_ROOT _TUTTI_PYTHON _TUTTI_REPO
