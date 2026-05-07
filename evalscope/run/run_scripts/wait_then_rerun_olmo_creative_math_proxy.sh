#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/project/ai4education/qianhong-p-qianhong"
WAIT_PID="${WAIT_PID:-2146354}"
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"

while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep "$CHECK_INTERVAL"
done

cd "$ROOT"
bash benchmark/evalscope/run/run_scripts/rerun_olmo_creative_math_proxy.sh
