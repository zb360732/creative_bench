# TriSkill Paper Pipeline

This is the paper-grade external pipeline for a benchmark + creativity elicitation framework paper. It intentionally does not modify `evalscope` scoring code.

## Reproducible Workflow

1. Create a run manifest.

```bash
python enhance/triskill_cli.py --task rat --make-manifest --output enhance/runs/manifest.jsonl
```

2. Run a method to produce artifacts.

```bash
python enhance/triskill_cli.py \
  --task rat \
  --input evalscope/evalscope/benchmarks/rat/data/rat.json \
  --output enhance/runs/rat_triskill_full.jsonl \
  --method triskill_full \
  --use-env-llm
```

3. Audit artifacts for required fields and gold leakage.

```bash
python enhance/triskill_cli.py \
  --task rat \
  --input enhance/runs/rat_triskill_full.jsonl \
  --output enhance/runs/rat_triskill_full.audit.json \
  --audit
```

4. Export predictions for external scoring.

```bash
python enhance/triskill_cli.py \
  --task rat \
  --input enhance/runs/rat_triskill_full.jsonl \
  --output enhance/runs/rat_triskill_full.predictions.json \
  --to-predictions
```

5. After scoring, join scores back to artifacts.

```bash
python enhance/triskill_cli.py \
  --task rat \
  --input enhance/runs/rat_triskill_full.jsonl \
  --scores enhance/runs/rat_triskill_full.scores.jsonl \
  --output enhance/runs/rat_triskill_full.scored.jsonl \
  --join-scores
```

6. Generate paper tables.

```bash
python enhance/triskill_cli.py \
  --task rat \
  --input enhance/runs/rat_direct.scored.jsonl,enhance/runs/rat_triskill_full.scored.jsonl \
  --output enhance/runs/rat_summary.json \
  --scored-summary \
  --primary-score score \
  --baseline-method direct
```

## Required Paper Baselines

- `direct`
- `generic_creativity_prompt`
- `cot_structured`
- `triskill_full`
- `triskill_without_verifier`
- `triskill_wrong_skill_assignment`

Optional later baselines:

- `multi_sample`
- `self_refine`
- `direct_long`
- `budget_matched`

## Claims This Pipeline Can Support

- Task-level improvement after score join.
- Level-level creativity profile shift.
- Novelty/appropriateness trade-off when scores include canonical groups.
- Length and budget controls via `output_length` and `num_llm_calls`.
- Transformation failure-mode reduction via diagnostics.
- Gold leakage audit via `safe_item` and artifact checks.

## What Still Requires Real Experiments

The code creates the full reproducible pipeline, but paper claims require actual model runs and evalscope/external scorer outputs. Do not claim score gains until scored JSONL files are generated and summarized.
