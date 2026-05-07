#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/project/ai4education/qianhong-p-qianhong"
EVAL_PY="$ROOT/zzb/conda/envs/evalscope/bin/python"
RUNNER="$ROOT/benchmark/evalscope/run/run_parallel_eval.py"
MODELS_JSON="$ROOT/benchmark/evalscope/run/models.json"
WORK_DIR="$ROOT/benchmark/outputs/exploration"
MODELS="${MODELS:-olmo2-7b,olmo2-13b}"
OLMO_CREATIVEMATH_MAX_TOKENS="${OLMO_CREATIVEMATH_MAX_TOKENS:-1024}"
RUN_NAME="${RUN_NAME:-creative_math_newdeploy_olmo_mt1024_v3}"

cd "$ROOT"

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  "$EVAL_PY" "$RUNNER" \
    --models-json "$MODELS_JSON" \
    --models "$MODELS" \
    --datasets creative_math \
    --work-dir "$WORK_DIR" \
    --run-name "$RUN_NAME" \
    --max-tokens "$OLMO_CREATIVEMATH_MAX_TOKENS" \
    --temperature 0.0 \
    --eval-batch-size 1 \
    --max-parallel 2 \
    --dataset-args '{"creative_math":{"extra_params":{"evaluation_mode":"full"}}}' \
    --log-file "$WORK_DIR/$RUN_NAME/run.log"

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  "$EVAL_PY" benchmark/evalscope/run/run_scripts/collect_newdeploy_results.py

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  "$EVAL_PY" benchmark/outputs/exploration/refresh_exploration_metrics_matrix.py

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  "$EVAL_PY" benchmark/outputs/exploration/generate_metrics_matrix_transposed.py

echo "[OK] OLMo creative_math rerun finished."
