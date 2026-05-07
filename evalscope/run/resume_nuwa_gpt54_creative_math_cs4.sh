#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/inspire/hdd/project/ai4education/qianhong-p-qianhong"
PYTHON_BIN="$ROOT_DIR/zzb/conda/envs/evalscope/bin/python"
export PYTHONPATH="$ROOT_DIR/benchmark/evalscope${PYTHONPATH:+:$PYTHONPATH}"
NO_PROXY_HOSTS="ai-notebook-inspire.sii.edu.cn,notebook-inspire.sii.edu.cn,api.nuwaapi.com,localhost,127.0.0.1"
export NO_PROXY="${NO_PROXY_HOSTS}${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import json

from evalscope.config import TaskConfig
from evalscope.run import run_task

root = Path("/inspire/hdd/project/ai4education/qianhong-p-qianhong").resolve()
models = json.loads((root / "benchmark/evalscope/run/models.json").read_text(encoding="utf-8"))
entry = next(model for model in models["models"] if model["name"] == "nuwa-gpt-5.4")

common = dict(
    model=entry["model"],
    model_id=entry["name"],
    api_url=entry["api_url"],
    api_key=entry["api_key"],
    seed=42,
    generation_config={"max_tokens": 30000, "temperature": 0.0},
    eval_batch_size=1,
    no_timestamp=True,
)

creative_cache = str(root / "benchmark/outputs/exploration/creative_math_nuwa_gpt54_full_mt30000_v1/nuwa-gpt-5.4")
print(f"[RUN] Resuming creative_math from {creative_cache}", flush=True)
run_task(TaskConfig(
    datasets=["creative_math"],
    dataset_args={"creative_math": {"extra_params": {"evaluation_mode": "full"}}},
    use_cache=creative_cache,
    **common,
))

cs4_work_dir = str(root / "benchmark/outputs/exploration/cs4_nuwa_gpt54/nuwa-gpt-5.4")
print(f"[RUN] Starting cs4 at {cs4_work_dir}", flush=True)
run_task(TaskConfig(
    datasets=["cs4"],
    dataset_args={"cs4": {"extra_params": {"evaluation_mode": "full", "judge_max_tokens": 8192}}},
    work_dir=cs4_work_dir,
    **common,
))

print("[OK] creative_math and cs4 completed", flush=True)
PY
