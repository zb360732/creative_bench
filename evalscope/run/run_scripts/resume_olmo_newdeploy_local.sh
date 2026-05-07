#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/project/ai4education/qianhong-p-qianhong"
SERVICE_PY="$ROOT/zzb/conda/envs/verl-agent-proagent/bin/python"
EVAL_PY="$ROOT/zzb/conda/envs/evalscope/bin/python"
RUNNER="$ROOT/benchmark/evalscope/run/run_parallel_eval.py"
WORK_DIR="$ROOT/benchmark/outputs/exploration"
MODELS_JSON="/tmp/olmo_newdeploy_local_models.json"
OLMO_NEOCODER_MAX_TOKENS="${OLMO_NEOCODER_MAX_TOKENS:-2048}"
OLMO_CREATIVEMATH_MAX_TOKENS="${OLMO_CREATIVEMATH_MAX_TOKENS:-1024}"
OLMO_CS4_MAX_TOKENS="${OLMO_CS4_MAX_TOKENS:-2048}"
OLMO_CS4_JUDGE_MAX_TOKENS="${OLMO_CS4_JUDGE_MAX_TOKENS:-2048}"

OLMO7_GPU="${OLMO7_GPU:-0}"
OLMO13_GPU="${OLMO13_GPU:-1}"
OLMO7_PORT="${OLMO7_PORT:-8034}"
OLMO13_PORT="${OLMO13_PORT:-8024}"
START_SERVICES="${START_SERVICES:-1}"

cat > "$MODELS_JSON" <<JSON
{
  "models": [
    {
      "name": "olmo2-7b",
      "model": "olmo2-7b",
      "api_url": "http://127.0.0.1:${OLMO7_PORT}/v1",
      "api_key": "EMPTY"
    },
    {
      "name": "olmo2-13b",
      "model": "olmo2-13b",
      "api_url": "http://127.0.0.1:${OLMO13_PORT}/v1",
      "api_key": "EMPTY"
    }
  ]
}
JSON

if [[ "$START_SERVICES" == "1" ]]; then
  nohup setsid env -u PYTHONPATH \
    CUDA_VISIBLE_DEVICES="$OLMO7_GPU" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$SERVICE_PY" -m vllm.entrypoints.openai.api_server \
      --model /inspire/hdd/project/ai4education/public/Models/OLMo/OLMo-2-7B-Instruct \
      --served-model-name olmo2-7b \
      --host 0.0.0.0 \
      --port "$OLMO7_PORT" \
      --gpu-memory-utilization 0.8 \
      --max-model-len 4096 \
      --tensor-parallel-size 1 \
      --max-num-seqs 16 \
      --max-num-batched-tokens 4096 \
      --disable-custom-all-reduce \
    > /tmp/olmo2-7b_${OLMO7_PORT}.log 2>&1 &

  nohup setsid env -u PYTHONPATH \
    CUDA_VISIBLE_DEVICES="$OLMO13_GPU" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$SERVICE_PY" -m vllm.entrypoints.openai.api_server \
      --model /inspire/hdd/project/ai4education/public/Models/OLMo/OLMo-2-13B-Instruct \
      --served-model-name olmo2-13b \
      --host 0.0.0.0 \
      --port "$OLMO13_PORT" \
      --gpu-memory-utilization 0.9 \
      --max-model-len 4096 \
      --tensor-parallel-size 1 \
      --max-num-seqs 16 \
      --max-num-batched-tokens 4096 \
      --disable-custom-all-reduce \
    > /tmp/olmo2-13b_${OLMO13_PORT}.log 2>&1 &

  echo "[INFO] Started OLMo services. Logs:"
  echo "  /tmp/olmo2-7b_${OLMO7_PORT}.log"
  echo "  /tmp/olmo2-13b_${OLMO13_PORT}.log"
  echo "[INFO] Waiting 20s for services to warm up..."
  sleep 20
fi

cd "$ROOT"

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  "$EVAL_PY" "$RUNNER" \
    --models-json "$MODELS_JSON" \
    --models "olmo2-7b,olmo2-13b" \
    --datasets neocoder \
    --work-dir "$WORK_DIR" \
    --run-name neocoder_newdeploy_olmo_mt4096_v1 \
    --max-tokens "$OLMO_NEOCODER_MAX_TOKENS" \
    --temperature 0.0 \
    --eval-batch-size 1 \
    --max-parallel 2 \
    --log-file "$WORK_DIR/neocoder_newdeploy_olmo_mt4096_v1/run.log"

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  "$EVAL_PY" "$RUNNER" \
    --models-json "$MODELS_JSON" \
    --models "olmo2-7b,olmo2-13b" \
    --datasets creative_math \
    --work-dir "$WORK_DIR" \
    --run-name creative_math_newdeploy_olmo_mt1024_v3 \
    --max-tokens "$OLMO_CREATIVEMATH_MAX_TOKENS" \
    --temperature 0.0 \
    --eval-batch-size 1 \
    --max-parallel 2 \
    --dataset-args '{"creative_math":{"extra_params":{"evaluation_mode":"full"}}}' \
    --log-file "$WORK_DIR/creative_math_newdeploy_olmo_mt1024_v3/run.log"

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  "$EVAL_PY" "$RUNNER" \
    --models-json "$MODELS_JSON" \
    --models "olmo2-7b,olmo2-13b" \
    --datasets cs4 \
    --work-dir "$WORK_DIR" \
    --run-name cs4_newdeploy_olmo_mt4096_v1 \
    --max-tokens "$OLMO_CS4_MAX_TOKENS" \
    --temperature 0.0 \
    --eval-batch-size 1 \
    --max-parallel 2 \
    --dataset-args "{\"cs4\":{\"extra_params\":{\"evaluation_mode\":\"full\",\"judge_max_tokens\":${OLMO_CS4_JUDGE_MAX_TOKENS}}}}" \
    --log-file "$WORK_DIR/cs4_newdeploy_olmo_mt4096_v1/run.log"

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  "$EVAL_PY" benchmark/evalscope/run/run_scripts/collect_newdeploy_results.py

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  "$EVAL_PY" benchmark/outputs/exploration/refresh_exploration_metrics_matrix.py

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  "$EVAL_PY" benchmark/outputs/exploration/generate_metrics_matrix_transposed.py

echo "[OK] OLMo benchmark resume finished."
