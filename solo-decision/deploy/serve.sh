#!/usr/bin/env bash
# Same serving configuration used for the measured layout comparisons.
set -euo pipefail
JEV_DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JEV_VENV="${JEV_VENV:-$JEV_DEPLOY_DIR/.venv}"
JEV_MODEL_DIR="${JEV_MODEL_DIR:-$JEV_DEPLOY_DIR/models/JEV-9B}"
if [[ ! -x "$JEV_VENV/bin/vllm" || ! -f "$JEV_MODEL_DIR/adapter_vllm/adapter_model.safetensors" ]]; then
    printf 'Run bash deploy/jev9b.sh setup, or set JEV_VENV and JEV_MODEL_DIR.\n' >&2
    exit 1
fi
JEV_EXTRA_ARGS=()
if [[ "${JEV_EAGER:-1}" == 1 ]]; then
    JEV_EXTRA_ARGS+=(--enforce-eager)
fi
exec "$JEV_VENV/bin/vllm" serve "$JEV_MODEL_DIR" \
    --served-model-name autotrust/JEV-9B \
    --host "${JEV_HOST:-127.0.0.1}" --port "${JEV_PORT:-8000}" \
    --dtype bfloat16 --tensor-parallel-size 1 \
    --enable-lora --max-loras 1 --max-lora-rank 32 \
    --lora-modules "jev-decision=$JEV_MODEL_DIR/adapter_vllm" \
    --logprobs-mode processed_logprobs --max-logprobs 16 \
    --enable-prompt-tokens-details --disable-uvicorn-access-log \
    --enable-prefix-caching --mamba-cache-mode align \
    --max-model-len "${JEV_MAX_MODEL_LEN:-16384}" \
    --max-num-seqs "${JEV_MAX_NUM_SEQS:-4}" \
    --max-num-batched-tokens "${JEV_BATCH_TOKENS:-2048}" \
    --gpu-memory-utilization "${JEV_GPU_MEMORY_UTILIZATION:-0.90}" \
    "${JEV_EXTRA_ARGS[@]}" "$@"
