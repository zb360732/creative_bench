# TriSkill Enhance Layer

This directory implements `solution.md` as a standalone prompt enhancement layer for the creativity benchmark.

It does not modify `evalscope`. The evaluator can keep using its existing adapters and answer parsers; TriSkill only wraps the visible task prompt with internal workflow instructions and preserves the final answer schema.

## Components

- `triskill/profiles.py`: task profiling and raw metric to canonical metric mapping.
- `triskill/skills.py`: skill registry and task/metric-conditioned skill selection.
- `triskill/core.py`: workflow composer and benchmark-safe prompt wrapper.
- `triskill/state.py`: `ElicitationState` plus scoring-field filtering.
- `triskill/executor.py`: MVP prompt-only workflow executor and experiment artifact builder.
- `triskill/normalizer.py`: final answer normalization helpers for adapter-compatible formats.
- `triskill/dataset.py`: JSON/JSONL dataset enhancement utilities.
- `triskill_cli.py`: command line interface.

## Usage

Print the selected workflow and skills:

```bash
python enhance/triskill_cli.py --task rat --plan
```

Wrap a prompt from stdin:

```bash
printf 'Find a single word that connects cottage/swiss/cake.' \
  | python enhance/triskill_cli.py --task rat
```

Wrap a prompt from an argument:

```bash
python enhance/triskill_cli.py \
  --task aut \
  --prompt 'What are some creative uses for a brick? Return JSON in <answer> tags.'
```

Create JSONL artifacts from an evalscope-style dataset without copying gold fields into the prompt:

```bash
python enhance/triskill_cli.py \
  --task rat \
  --input evalscope/evalscope/benchmarks/rat/data/rat.json \
  --output /tmp/rat_triskill_artifacts.jsonl \
  --limit 5
```

Run the full multi-skill execution path with an OpenAI-compatible endpoint:

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

Supported methods:

- `triskill_prompt_only`: build enhanced prompts and logs without calling an LLM.
- `triskill_full`: execute each selected skill as an LLM JSON step, then normalize the final answer.
- `triskill_without_verifier`: ablation removing verifier/check/audit skills.
- `triskill_wrong_skill_assignment`: diagnostic mismatched-skill ablation.
- `direct`, `generic_creativity_prompt`, `cot_structured`: baselines that require `--use-env-llm`.

Each artifact contains:

- `safe_item`: visible fields only, with gold/reference fields removed.
- `original_prompt`: original benchmark prompt.
- `enhanced_prompt`: prompt to send to the model.
- `skills`, `budgets`, `canonical_metrics`, `skill_trace`: experiment logging fields.
- `warnings`: excluded scoring-only fields or other safety notes.
- `final_answer`, `num_llm_calls`, `output_length`: populated by full execution methods.

Convert artifacts into a simple prediction file for external scoring scripts:

```bash
python enhance/triskill_cli.py \
  --task rat \
  --input /tmp/rat_triskill_full.jsonl \
  --output /tmp/rat_predictions.json \
  --to-predictions
```

Summarize experiment artifacts:

```bash
python enhance/triskill_cli.py \
  --task rat \
  --input /tmp/rat_triskill_full.jsonl,/tmp/rat_direct.jsonl \
  --output /tmp/rat_summary.json \
  --analyze
```

The analysis summary includes parse success rate, output length, LLM call counts, profile-shift grouping, and Transformation failure-mode counts when applicable.

Programmatic use:

```python
from triskill import enhance_prompt

prompt = enhance_prompt(
    "Please provide 10 words that are semantically distant from each other.",
    task_name="dat",
)
```

## Safety Contract

TriSkill only uses visible task fields: task text, visible choices, visible constraints, task name, output schema, and metric objectives. Hidden references, gold answers, scoring-only candidate lists, test-set statistics, and post-hoc scores must not be passed into prompts.

Final answers must remain short and adapter-parseable, usually inside the original `<answer>...</answer>` block.

Known scoring-only fields are filtered by default, including `answer`, `target`, `target_words`, `candidate_answers`, `reference`, `gold`, and `solutions`.
