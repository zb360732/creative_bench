#!/usr/bin/env bash
set -euo pipefail

mkdir -p benchmark/outputs/transformation/problem_method_full
nohup python benchmark/evalscope/run/run_parallel_eval.py \
  --models-json benchmark/evalscope/run/models.json \
  --datasets problem_method \
  --limit none \
  --max-tokens 1024 \
  --temperature 0.7 \
  --eval-batch-size 1 \
  --max-parallel 5 \
  --work-dir benchmark/outputs/transformation \
  --run-name problem_method_full \
  --dataset-args '{"problem_method":{"extra_params":{"evaluation_mode":"full"}}}' \
  > benchmark/outputs/transformation/problem_method_full/run.log 2>&1 &
