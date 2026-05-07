#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-1}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-1}"
export NCCL_WORK_FIFO_BYTES="${NCCL_WORK_FIFO_BYTES:-262144}"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-262144}"
export NCCL_LL_BUFFSIZE="${NCCL_LL_BUFFSIZE:-131072}"
export NCCL_LL128_BUFFSIZE="${NCCL_LL128_BUFFSIZE:-131072}"

MODEL_PATH="${MODEL_PATH:-/inspire/hdd/project/ai4education/public/Models/Qwen/Qwen3.5-122B-A10B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-122b-a10b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8038}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
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
