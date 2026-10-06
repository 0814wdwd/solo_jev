#!/usr/bin/env bash
# Source-checkout bootstrap; all model and server dependencies stay isolated.
set -euo pipefail
JEV_DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export JEV_DEPLOY_DIR
export JEV_VENV="${JEV_VENV:-$JEV_DEPLOY_DIR/.venv}"
export JEV_MODEL_DIR="${JEV_MODEL_DIR:-$JEV_DEPLOY_DIR/models/JEV-9B}"
export JEV_RUNTIME_DIR="${JEV_RUNTIME_DIR:-$JEV_DEPLOY_DIR/runtime}"
# Reuse the server interpreter even in SSH shells whose PATH lacks Python.
if [[ -z "${JEV_PYTHON:-}" ]]; then
    if [[ -x "$JEV_VENV/bin/python" ]]; then
        JEV_PYTHON="$JEV_VENV/bin/python"
    else
        JEV_PYTHON=python3
    fi
fi

usage() {
    cat <<'HELP'
Usage: bash deploy/jev9b.sh {up|setup|start|stop|status|doctor|help}

  up      Install isolated dependencies, fetch the pinned model, start and wait.
  setup   Install/download only; repeated runs reuse the environment and files.
  start   Start an already installed server and wait for the decision adapter.
  stop    Stop only the process group started by this checkout.
  status  Show the owned process and decision-model health.
  doctor  Show GPU, runtime, model and local endpoint readiness (no mutation).

Linux + NVIDIA CUDA GPU required. Tested: RTX 4090 24 GB, driver 595.71.05.
Reserve about 35 GB disk for weights and the isolated runtime. Model download
needs Hugging Face access; standard HF_TOKEN/HTTPS_PROXY settings are respected.

Paths: JEV_VENV, JEV_MODEL_DIR, JEV_RUNTIME_DIR; Python: JEV_PYTHON (3.10+).
Server: JEV_HOST (127.0.0.1), JEV_PORT (8000), JEV_START_TIMEOUT (600 seconds),
JEV_MAX_MODEL_LEN (16384), JEV_MAX_NUM_SEQS (4), JEV_BATCH_TOKENS (2048),
JEV_GPU_MEMORY_UTILIZATION (0.90), JEV_EAGER (1).
Do not bind an unauthenticated endpoint to a public network.
HELP
}

setup() {
    command -v nvidia-smi >/dev/null || { echo 'nvidia-smi is required on the GPU host.' >&2; exit 1; }
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
    "$JEV_PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Python 3.10+ is required")'
    mkdir -p "$JEV_RUNTIME_DIR"
    if [[ ! -x "$JEV_VENV/bin/python" ]]; then
        "$JEV_PYTHON" -m venv "$JEV_VENV"
        "$JEV_VENV/bin/python" -m pip install --upgrade pip
    fi
    "$JEV_VENV/bin/python" -m pip install -r "$JEV_DEPLOY_DIR/requirements-gpu.txt"
    "$JEV_VENV/bin/python" "$JEV_DEPLOY_DIR/fix_vllm_warmup.py"
    "$JEV_VENV/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable; check the NVIDIA driver and installed PyTorch build"; print("CUDA ready:", torch.cuda.get_device_name(0))'
    "$JEV_VENV/bin/python" -m pip install "$JEV_DEPLOY_DIR/..[demo,hub]"
    "$JEV_VENV/bin/python" "$JEV_DEPLOY_DIR/download.py"
    "$JEV_VENV/bin/python" -m pip freeze > "$JEV_RUNTIME_DIR/runtime-lock.txt"
    printf 'Setup complete. Start: bash %q start\n' "$JEV_DEPLOY_DIR/jev9b.sh"
}

case "${1:-help}" in
    up)
        if "$JEV_PYTHON" "$JEV_DEPLOY_DIR/control.py" status >/dev/null 2>&1; then
            "$JEV_PYTHON" "$JEV_DEPLOY_DIR/control.py" start
        else
            setup
            "$JEV_PYTHON" "$JEV_DEPLOY_DIR/control.py" start
        fi
        ;;
    setup) setup ;;
    start|stop|status|doctor) "$JEV_PYTHON" "$JEV_DEPLOY_DIR/control.py" "$1" ;;
    help|-h|--help) usage ;;
    *) usage >&2; exit 2 ;;
esac
