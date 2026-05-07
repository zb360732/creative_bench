#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/project/ai4education/qianhong-p-qianhong"
PY="$ROOT/zzb/conda/envs/evalscope/bin/python"
RUNNER="$ROOT/benchmark/evalscope/run/run_parallel_eval.py"
MODELS_JSON="$ROOT/benchmark/evalscope/run/models.json"
WORK_DIR="$ROOT/benchmark/outputs/exploration"

TS="$(date -u +%Y%m%d_%H%M%S)"
LAUNCH_DIR="$WORK_DIR/newdeploy_launch/$TS"
mkdir -p "$LAUNCH_DIR"

launch() {
  local run_name="$1"
  shift

  mkdir -p "$WORK_DIR/$run_name"

  # run_parallel_eval.py will tee logs into --log-file; we silence stdout/stderr to keep tty clean.
  nohup setsid \
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    "$PY" "$RUNNER" \
      --models-json "$MODELS_JSON" \
      --work-dir "$WORK_DIR" \
      --run-name "$run_name" \
      --log-file "$WORK_DIR/$run_name/run.log" \
      "$@" \
    </dev/null >/dev/null 2>&1 &

  echo "$!" > "$LAUNCH_DIR/$run_name.pid"
  echo "[LAUNCH] $run_name pid=$!"
}

LONG_MODELS="llama3.1-8b,mistral-7b,mistral-24b,qwen2.5-32b"
OLMO_MODELS="olmo2-7b,olmo2-13b"
OLMO_NEOCODER_MAX_TOKENS="${OLMO_NEOCODER_MAX_TOKENS:-2048}"
OLMO_CREATIVEMATH_MAX_TOKENS="${OLMO_CREATIVEMATH_MAX_TOKENS:-1024}"
OLMO_CS4_MAX_TOKENS="${OLMO_CS4_MAX_TOKENS:-2048}"
OLMO_CS4_JUDGE_MAX_TOKENS="${OLMO_CS4_JUDGE_MAX_TOKENS:-2048}"

# Long-context group (match previous mt30000 setup; cs4 judge_max_tokens=8192).
launch "neocoder_newdeploy_long_mt30000_v1" \
  --models "$LONG_MODELS" \
  --datasets neocoder \
  --limit none \
  --max-tokens 30000 \
  --temperature 0.0 \
  --eval-batch-size 2 \
  --max-parallel 2

launch "creative_math_newdeploy_long_mt30000_v1" \
  --models "$LONG_MODELS" \
  --datasets creative_math \
  --limit none \
  --max-tokens 30000 \
  --temperature 0.0 \
  --eval-batch-size 1 \
  --max-parallel 4 \
  --dataset-args '{"creative_math":{"extra_params":{"evaluation_mode":"full"}}}'

launch "cs4_newdeploy_long_mt30000_v1" \
  --models "$LONG_MODELS" \
  --datasets cs4 \
  --limit none \
  --max-tokens 30000 \
  --temperature 0.0 \
  --eval-batch-size 1 \
  --max-parallel 4 \
  --dataset-args '{"cs4":{"extra_params":{"evaluation_mode":"full","judge_max_tokens":8192}}}'

# OLMo group: keep creative_math lower than the other datasets because some prompts exceed 2200 tokens.
launch "neocoder_newdeploy_olmo_mt4096_v1" \
  --models "$OLMO_MODELS" \
  --datasets neocoder \
  --limit none \
  --max-tokens "$OLMO_NEOCODER_MAX_TOKENS" \
  --temperature 0.0 \
  --eval-batch-size 1 \
  --max-parallel 2

launch "creative_math_newdeploy_olmo_mt1024_v3" \
  --models "$OLMO_MODELS" \
  --datasets creative_math \
  --limit none \
  --max-tokens "$OLMO_CREATIVEMATH_MAX_TOKENS" \
  --temperature 0.0 \
  --eval-batch-size 1 \
  --max-parallel 2 \
  --dataset-args '{"creative_math":{"extra_params":{"evaluation_mode":"full"}}}'

launch "cs4_newdeploy_olmo_mt4096_v1" \
  --models "$OLMO_MODELS" \
  --datasets cs4 \
  --limit none \
  --max-tokens "$OLMO_CS4_MAX_TOKENS" \
  --temperature 0.0 \
  --eval-batch-size 1 \
  --max-parallel 2 \
  --dataset-args "{\"cs4\":{\"extra_params\":{\"evaluation_mode\":\"full\",\"judge_max_tokens\":${OLMO_CS4_JUDGE_MAX_TOKENS}}}}"

echo "[OK] Launch records: $LAUNCH_DIR"
