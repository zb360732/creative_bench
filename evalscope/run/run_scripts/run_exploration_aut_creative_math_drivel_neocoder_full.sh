#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

timestamp="$(date -u +%Y%m%d_%H%M%S)"
work_root="/inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/outputs/exploration"
run_name="aut_creative_math_drivel_neocoder_full_${timestamp}"
run_dir="${work_root}/${run_name}"

mkdir -p "${run_dir}"
cd "${repo_root}"

nohup setsid python run/run_parallel_eval.py \
  --models-json run/models.json \
  --datasets aut,creative_math,drivel_writing,neocoder \
  --limit none \
  --max-tokens 30000 \
  --temperature 0.0 \
  --eval-batch-size 1 \
  --max-parallel 5 \
  --work-dir "${work_root}" \
  --run-name "${run_name}" \
  --dataset-args '{"creative_math":{"extra_params":{"evaluation_mode":"full"}}}' \
  --log-file "${run_dir}/run.log" \
  </dev/null >/dev/null 2>&1 &

echo "$!" > "${run_dir}/run.pid"

echo "Started run in ${run_dir} (pid $!)"
