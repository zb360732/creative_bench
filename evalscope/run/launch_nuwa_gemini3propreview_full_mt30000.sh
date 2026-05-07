#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/inspire/hdd/project/ai4education/qianhong-p-qianhong"
PYTHON_BIN="$ROOT_DIR/zzb/conda/envs/evalscope/bin/python"
RUNNER="$ROOT_DIR/benchmark/evalscope/run/run_parallel_eval.py"
WORK_DIR="$ROOT_DIR/benchmark/outputs/exploration"
MODELS="nuwa-gemini-3-pro-preview"
NO_PROXY_HOSTS="ai-notebook-inspire.sii.edu.cn,notebook-inspire.sii.edu.cn,api.nuwaapi.com,localhost,127.0.0.1"

export NO_PROXY="${NO_PROXY_HOSTS}${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"

cd "$ROOT_DIR"

"$PYTHON_BIN" "$RUNNER" \
  --models "$MODELS" \
  --datasets neocoder \
  --work-dir "$WORK_DIR" \
  --run-name neocoder_nuwa_gemini3propreview_full_mt30000_v1 \
  --max-tokens 30000 \
  --eval-batch-size 4 \
  --max-parallel 1

"$PYTHON_BIN" "$RUNNER" \
  --models "$MODELS" \
  --datasets creative_math \
  --work-dir "$WORK_DIR" \
  --run-name creative_math_nuwa_gemini3propreview_full_mt30000_v1 \
  --max-tokens 30000 \
  --eval-batch-size 1 \
  --max-parallel 1 \
  --dataset-args '{"creative_math":{"extra_params":{"evaluation_mode":"full"}}}'

"$PYTHON_BIN" "$RUNNER" \
  --models "$MODELS" \
  --datasets cs4 \
  --work-dir "$WORK_DIR" \
  --run-name cs4_nuwa_gemini3propreview \
  --max-tokens 30000 \
  --eval-batch-size 1 \
  --max-parallel 1 \
  --dataset-args '{"cs4":{"extra_params":{"evaluation_mode":"full","judge_max_tokens":8192}}}'
