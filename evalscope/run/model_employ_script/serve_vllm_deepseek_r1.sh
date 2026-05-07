#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-/inspire/hdd/project/ai4education/public/Models/deepseek-ai/DeepSeek-R1-0528}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-r1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8011}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
DTYPE="${DTYPE:-auto}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model path not found: $MODEL_PATH" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ "$DISABLE_CUSTOM_ALL_REDUCE" == "1" ]]; then
  EXTRA_ARGS+=(--disable-custom-all-reduce)
fi
if [[ -n "${VLLM_ARGS:-}" ]]; then
  EXTRA_ARGS+=(${VLLM_ARGS})
fi

PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --dtype "$DTYPE" \
  "${EXTRA_ARGS[@]}"
