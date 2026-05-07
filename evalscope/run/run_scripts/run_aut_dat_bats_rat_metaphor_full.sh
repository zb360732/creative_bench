#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

timestamp="$(date -u +%Y%m%d_%H%M%S)"
work_root="/inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/outputs/combination"
run_name="dat_bats_rat_metaphor_full_${timestamp}"
run_dir="${work_root}/${run_name}"

mkdir -p "${run_dir}"
cd "${repo_root}"

nohup setsid python run/run_parallel_eval.py \
  --models-json run/models.json \
  --datasets dat,bats,rat,metaphor \
  --limit none \
  --max-tokens 30000 \
  --temperature 0.0 \
  --eval-batch-size 4 \
  --work-dir "${work_root}" \
  --run-name "${run_name}" \
  --log-file "${run_dir}/run.log" \
  </dev/null >/dev/null 2>&1 &

echo "$!" > "${run_dir}/run.pid"

echo "Started run in ${run_dir} (pid $!)"
