# Creative Benchmark

Creative Benchmark is a benchmark workspace for evaluating LLM creativity across
combination, exploration, and transformation tasks. The repository combines a
customized EvalScope-based runner with TriSkill prompt-enhancement experiments,
task suites, scripts, datasets, and analysis utilities.

## Repository Layout

- `evalscope/`: EvalScope-based benchmark framework and creative benchmark tasks.
- `evalscope/run/`: Main run orchestration scripts, model configs, suite configs,
  summary tools, and operational notes.
- `enhance/`: TriSkill enhancement layer for prompt-only and multi-skill
  creativity elicitation experiments.
- `plots/`: Generated figures kept for reporting.
- `outputs/`, model caches, temporary files, and local credentials are ignored by
  Git and should be regenerated locally.

## Benchmark Suites

Suites are defined in `evalscope/run/task_suites.json`.

| Suite | Tasks |
| --- | --- |
| `combination` | `dat`, `bats`, `rat`, `metaphor` |
| `exploration` | `aut`, `creative_math`, `drivel_writing`, `neocoder` |
| `transformation` | `transformation` |
| `fulltask` | all current tasks, including `cs4` |

## Setup

The EvalScope package requires Python 3.10 or newer.

```bash
cd evalscope
python -m pip install -e .
```

Optional dependency groups are listed in `evalscope/pyproject.toml` and
`evalscope/requirements/`. Install only the extras needed for the target
evaluation backend.

## Model And Judge Configuration

Main run scripts read model and judge settings from JSON files under
`evalscope/run/`.

- `models2.json`: example model list with `api_key` set to `EMPTY`.
- `models.json`: local model configuration, ignored by Git.
- `llm_judge.json`: local judge model configuration, ignored by Git.

Create local copies or edit the ignored local files on the machine that runs the
benchmark. Do not commit real API keys.

## Running Benchmarks

Use the suite wrapper from the `evalscope` directory:

```bash
cd evalscope
python run/run_suite.py --suite combination
python run/run_suite.py --suite exploration
python run/run_suite.py --suite transformation
python run/run_suite.py --suite fulltask
```

For full control, call the parallel runner directly:

```bash
cd evalscope
python run/run_parallel_eval.py \
  --models-json run/models2.json \
  --datasets dat,bats,rat,metaphor \
  --limit none \
  --max-tokens 30000 \
  --temperature 0.0
```

Operational details for full runs and resume behavior are documented in
`evalscope/run/README_FULLTASK_RUN.md`.

## TriSkill Enhancement

TriSkill lives in `enhance/` and can be used without modifying EvalScope.

Print a task plan:

```bash
python enhance/triskill_cli.py --task rat --plan
```

Wrap a prompt:

```bash
printf 'Find a single word that connects cottage/swiss/cake.' \
  | python enhance/triskill_cli.py --task rat
```

Run full LLM-backed TriSkill execution with an OpenAI-compatible endpoint:

```bash
export TRISKILL_API_URL=http://localhost:8000/v1
export TRISKILL_MODEL=your-model-name
export TRISKILL_API_KEY=EMPTY

python enhance/triskill_cli.py \
  --task rat \
  --input evalscope/evalscope/benchmarks/rat/data/rat.json \
  --output /tmp/rat_triskill_full.jsonl \
  --limit 5 \
  --method triskill_full \
  --use-env-llm
```

See `enhance/README.md` and `enhance/PAPER_PIPELINE.md` for more details.

## Outputs And Large Files

Evaluation outputs are intentionally excluded from version control. By default,
suite configs write outputs under benchmark output directories such as
`benchmark/outputs/<suite>`.

One benchmark data file,
`evalscope/evalscope/benchmarks/bats/data/bats_full.json`, is larger than
GitHub's recommended 50 MB threshold but below the 100 MB hard limit. Consider
Git LFS if this file grows further.

## Notes For Contributors

- Keep source changes focused and avoid committing generated outputs.
- Do not commit private endpoint credentials, tokens, or local model caches.
- Prefer adding new task definitions through the existing suite/config structure.
- Use `evalscope/run/README_CLEANUP.md` for the current cleanup and organization
  notes.
