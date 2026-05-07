#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/inspire/hdd/project/ai4education/qianhong-p-qianhong"
PYTHON_BIN="$ROOT_DIR/zzb/conda/envs/evalscope/bin/python"
RUNNER="$ROOT_DIR/benchmark/evalscope/run/run_parallel_eval.py"
WORK_DIR="$ROOT_DIR/benchmark/outputs/exploration"
MODELS="qwen3-14b"

cd "$ROOT_DIR"

"$PYTHON_BIN" "$RUNNER" \
  --models "$MODELS" \
  --datasets neocoder \
  --work-dir "$WORK_DIR" \
  --run-name neocoder_full_mt30000_v1 \
  --max-tokens 30000 \
  --eval-batch-size 4 \
  --max-parallel 1

"$PYTHON_BIN" "$RUNNER" \
  --models "$MODELS" \
  --datasets creative_math \
  --work-dir "$WORK_DIR" \
  --run-name creative_math_full_mt30000_v1 \
  --max-tokens 30000 \
  --eval-batch-size 1 \
  --max-parallel 1 \
  --dataset-args '{"creative_math":{"extra_params":{"evaluation_mode":"full"}}}'

"$PYTHON_BIN" "$RUNNER" \
  --models "$MODELS" \
  --datasets cs4 \
  --work-dir "$WORK_DIR" \
  --run-name cs4 \
  --max-tokens 30000 \
  --eval-batch-size 1 \
  --max-parallel 1 \
  --dataset-args '{"cs4":{"extra_params":{"evaluation_mode":"full","judge_max_tokens":8192}}}'
