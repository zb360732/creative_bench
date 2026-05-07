#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-benchmark/outputs/combination/dat_bats_rat_metaphor_full}"

python benchmark/evalscope/run/summarize_reports.py \
  --run-dir "${RUN_DIR}" \
  --out-name scores_summary
