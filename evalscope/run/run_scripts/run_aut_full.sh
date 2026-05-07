#!/usr/bin/env bash
set -euo pipefail
export NLTK_DATA=/inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/evalscope/dataprocess/nltk_data
# Guard against meta-tensor defaults leaking from other loads.
unset TORCH_DEFAULT_DEVICE TRANSFORMERS_DEVICE_MAP TRANSFORMERS_LOW_CPU_MEM_USAGE
unset ACCELERATE_USE_CPU ACCELERATE_USE_MPS_DEVICE
export TORCH_DEFAULT_DEVICE=cpu
export TRANSFORMERS_DEVICE_MAP=
export TRANSFORMERS_LOW_CPU_MEM_USAGE=0
export EVALSCOPE_AUT_METRICS_DEVICE=cpu

python /inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/evalscope/run/run_parallel_eval.py \
  --models-json /inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/evalscope/run/models2.json \
  --datasets aut,dat,bats,rat,metaphor \
  --limit none \
  --work-dir /inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/outputs/combination \
  --run-name aut_dat_bats_rat_metaphor_full_models2 \
  --max-tokens 30000 \
  --temperature 0.0 \
  --eval-batch-size 4 \
  --no-skip-done
