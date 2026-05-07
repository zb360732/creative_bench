# Fulltask Run Guide

This note documents the recommended commands for running the full benchmark suite and resuming from an existing run directory.

## Datasets

The full task suite used here is:

- `dat`
- `bats`
- `rat`
- `metaphor`
- `aut`
- `creative_math`
- `cs4`
- `drivel_writing`
- `neocoder`
- `transformation`

`creative_math`, `cs4`, and `transformation` read judge definitions from `run/llm_judge.json`. If that file contains 3 models, all 3 are used automatically as judges.

## Quick Start

Use the wrapper script for a standard full run:

```bash
cd /inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/evalscope

LIMIT=none \
EVAL_BATCH_SIZE=32 \
JUDGE_WORKER_NUM=32 \
REQUEST_TIMEOUT=600 \
bash run/run_scripts/run_fulltask_fullmodels.sh
```

Behavior:

- output root: `/inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/outputs/fulltask`
- run name: `fulltask_fullmodels_<utc_timestamp>`
- log file: `<run_dir>/run.log`
- pid file: `<run_dir>/run.pid`

## Full Command

If you want to run the orchestrator directly instead of using the wrapper script:

```bash
cd /inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/evalscope

python run/run_parallel_eval.py \
  --models-json run/models2.json \
  --datasets dat,bats,rat,metaphor,aut,creative_math,cs4,drivel_writing,neocoder,transformation \
  --limit none \
  --max-tokens 8192 \
  --temperature 0.0 \
  --request-timeout 600 \
  --eval-batch-size 32 \
  --judge-worker-num 32 \
  --batch-mode off \
  --max-parallel 5 \
  --work-dir /inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/outputs/fulltask \
  --run-name fulltask_fullmodels_YYYYMMDD_HHMMSS \
  --dataset-args '{"creative_math":{"extra_params":{"evaluation_mode":"full"}},"cs4":{"extra_params":{"evaluation_mode":"full","judge_max_tokens":8192}},"transformation":{"extra_params":{"evaluation_mode":"llm_judge","judge_max_tokens":4096}}}' \
  --log-file /inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/outputs/fulltask/fulltask_fullmodels_YYYYMMDD_HHMMSS/run.log
```

## Resume From Existing Run

To resume into an existing output directory, reuse the same `--work-dir` and `--run-name`.

Example:

```bash
cd /inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/evalscope

python run/run_parallel_eval.py \
  --models-json run/models2.json \
  --datasets dat,bats,rat,metaphor,aut,creative_math,cs4,drivel_writing,neocoder,transformation \
  --limit none \
  --max-tokens 8192 \
  --temperature 0.0 \
  --request-timeout 600 \
  --eval-batch-size 32 \
  --judge-worker-num 32 \
  --batch-mode off \
  --max-parallel 5 \
  --work-dir /inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/outputs/fulltask \
  --run-name fulltask_fullmodels_20260406_063414 \
  --dataset-args '{"creative_math":{"extra_params":{"evaluation_mode":"full"}},"cs4":{"extra_params":{"evaluation_mode":"full","judge_max_tokens":8192}},"transformation":{"extra_params":{"evaluation_mode":"llm_judge","judge_max_tokens":4096}}}' \
  --log-file /inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/outputs/fulltask/fulltask_fullmodels_20260406_063414/resume.log \
  --no-skip-done
```

Important details:

- `--no-skip-done` disables model-level skipping in the orchestrator.
- sample-level reuse still happens through `predictions/*.jsonl` and `reviews/*.jsonl` cache files.
- already completed samples are skipped automatically.
- missing samples are appended to the existing cache files.
- missing reports are regenerated from the current cache state.

## Resume One Model Or A Few Tasks

You can limit the resume scope if you only need to fill a subset.

Example: resume only `deepseek` on `creative_math` and `neocoder`.

```bash
python run/run_parallel_eval.py \
  --models-json run/models2.json \
  --models deepseek \
  --datasets creative_math,neocoder \
  --limit none \
  --max-tokens 8192 \
  --temperature 0.0 \
  --request-timeout 600 \
  --eval-batch-size 32 \
  --judge-worker-num 32 \
  --batch-mode off \
  --max-parallel 1 \
  --work-dir /inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/outputs/fulltask \
  --run-name fulltask_fullmodels_20260406_063414 \
  --dataset-args '{"creative_math":{"extra_params":{"evaluation_mode":"full"}}}' \
  --no-skip-done
```

## Recommended Checks

Check whether the main process is still alive:

```bash
ps -eo pid,ppid,stat,etime,cmd | rg 'run_parallel_eval.py|fulltask_fullmodels_'
```

Check model-level reports:

```bash
find /inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/outputs/fulltask/fulltask_fullmodels_20260406_063414 -maxdepth 4 -type f -path '*/reports/*/*.json' | sort
```

Check sample-level cache sizes:

```bash
python - <<'PY'
from pathlib import Path
base = Path('/inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/outputs/fulltask/fulltask_fullmodels_20260406_063414')
models = ['deepseek','qwen2.5-7b','qwen3-8b','qwen2.5-72b']
expected = {
    'dat_default': 1,
    'bats_sampled': 4000,
    'rat_default': 144,
    'metaphor_default': 2953,
    'aut_default': 73,
    'creative_math_default': 605,
    'cs4_default': 250,
    'drivel_writing_narrative-writing-english': 600,
    'neocoder_default': 1194,
    'transformation_default': 1308,
}
for model in models:
    for kind in ['predictions', 'reviews']:
        root = base / model / kind / model
        for name, exp in expected.items():
            path = root / f'{name}.jsonl'
            got = sum(1 for _ in path.open('r', encoding='utf-8')) if path.exists() else -1
            if got != exp:
                print(model, kind, name, got, exp)
PY
```

If the script prints nothing, the sample-level cache is complete for all listed tasks.

## Notes

- `run/run_scripts/run_fulltask_fullmodels.sh` supports `LIMIT`, `EVAL_BATCH_SIZE`, `JUDGE_WORKER_NUM`, and `REQUEST_TIMEOUT` via environment variables.
- `run/run_parallel_eval.py` supports `--request-timeout` and passes it into generation config.
- `creative_math`, `cs4`, and `transformation` support multi-judge evaluation from `run/llm_judge.json`.
- `transformation` uses an LLM judge path and relies heavily on review cache reuse during resume.
- if you want sample-level补跑, keep the same run directory and do not delete `predictions/` or `reviews/`.
