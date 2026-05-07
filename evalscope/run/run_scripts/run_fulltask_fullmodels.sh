#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

timestamp="$(date -u +%Y%m%d_%H%M%S)"
work_root="/inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/outputs/fulltask"
run_name="fulltask_fullmodels_${timestamp}"
run_dir="${work_root}/${run_name}"
models_json="${MODELS_JSON:-run/models2.json}"
eval_batch_size="${EVAL_BATCH_SIZE:-16}"
judge_worker_num="${JUDGE_WORKER_NUM:-4}"
limit="${LIMIT:-None}"
request_timeout="${REQUEST_TIMEOUT:-600}"
max_tokens="${MAX_TOKENS:-8192}"
temperature="${TEMPERATURE:-0.7}"
judge_temperature="${JUDGE_TEMPERATURE:-0.2}"
batch_mode="${BATCH_MODE:-off}"
max_parallel="${MAX_PARALLEL:-8}"

mkdir -p "${run_dir}"
cd "${repo_root}"

nohup setsid python run/run_parallel_eval.py \
  --models-json "${models_json}" \
  --datasets dat,bats,rat,metaphor,aut,creative_math,cs4,neocoder,transformation \
  --limit "${limit}" \
  --max-tokens "${max_tokens}" \
  --temperature "${temperature}" \
  --request-timeout "${request_timeout}" \
  --eval-batch-size "${eval_batch_size}" \
  --judge-worker-num "${judge_worker_num}" \
  --batch-mode "${batch_mode}" \
  --max-parallel "${max_parallel}" \
  --work-dir "${work_root}" \
  --run-name "${run_name}" \
  --dataset-args "{\"creative_math\":{\"extra_params\":{\"evaluation_mode\":\"full\",\"judge_temperature\":${judge_temperature}}},\"cs4\":{\"extra_params\":{\"evaluation_mode\":\"full\",\"judge_max_tokens\":8192,\"judge_temperature\":${judge_temperature}}},\"transformation\":{\"extra_params\":{\"evaluation_mode\":\"llm_judge\",\"judge_max_tokens\":4096,\"judge_temperature\":${judge_temperature},\"judge_max_retries\":4,\"judge_sleep_seconds\":1.0}}}" \
  --log-file "${run_dir}/run.log" \
  </dev/null >/dev/null 2>&1 &

echo "$!" > "${run_dir}/run.pid"

echo "Started run in ${run_dir} (pid $!) using ${models_json}"
