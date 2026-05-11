# Evidence Ledger

This file is the Stage 0 evidence ledger for the benchmark + TriSkill paper.
Every paper result should be traceable to one row or table in this document
before it appears in the manuscript.

Current repository commit when this ledger was started: `40fe6a0`.

## Status Labels

- `final`: suitable for main paper tables if the claim scope matches the run.
- `final-candidate`: likely usable, but should be rechecked before final
  manuscript freeze.
- `diagnostic`: useful for method iteration or qualitative discussion, not a
  main-table result unless explicitly labeled.
- `replay`: metric-compatible offline replay over cached artifacts; useful for
  analysis but should be labeled unless rerun through the standard evaluator.
- `superseded`: valid historical run that has been replaced by a later method or
  evaluator setup.
- `rejected`: negative or abandoned iteration; useful for failure analysis only.
- `not-comparable`: result uses incompatible evaluation code, task cardinality,
  or judge setup.

## Global Evidence Sources

| ID | Source | Status | Use |
|---|---|---|---|
| `merged_evalscope_direct` | `/inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/outputs/huizong/merged_evalscope/summary.json` | `final-candidate` | Direct benchmark baseline across available models. |
| `merged_evalscope_coverage` | `/inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/outputs/huizong/merged_evalscope/coverage.json` | `final-candidate` | Coverage audit for merged direct outputs. |
| `alias_merge_log` | `/inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/outputs/huizong/merged_evalscope/alias_merge_log.jsonl` | `final-candidate` | Documents alias merge, especially `deepseek-chat` -> `deepseek-v3.2`. |
| `enhance_worklog` | `enhance/WORKLOG.md` | `final-candidate` | Method iteration chronology and rejected-run rationale. |
| `paper_materials` | `enhance/PAPER_MATERIALS.md` | `final-candidate` | Paper vocabulary, claims, caveats, and result notes. |

Merged direct baseline coverage currently contains 17 models. Full 9-task
coverage is present for:

- `deepseek-r1-distill-qwen-1.5b`
- `deepseek-r1-distill-qwen-7b`
- `deepseek-r1-distill-qwen-14b`
- `deepseek-r1-distill-qwen-32b`
- `qwen3.5-4b`
- `qwen3.5-9b`
- `qwen3.5-27b`
- `qwen3.5-35b-a3b`
- `qwen3.5-122b-a10b`

Several closed-source or large models have partial coverage only. This must be
reported as incomplete coverage unless additional runs are completed.

## Main Exact-Metric Combinational Evidence

Run IDs:

| ID | Path | Method | Models | Tasks | Limit | Status |
|---|---|---|---|---|---|---|
| `comb_direct_full_deepseek_qwen` | `outputs/combination_validation/models2_combination_full_general_entitytype_nolimit_direct/summary.json` | direct | 1.5B, 7B, 14B, 32B DeepSeek-R1-Distill-Qwen | DAT, BATS, RAT, Metaphor | full | `final` |
| `comb_triskill_full_deepseek_qwen` | `outputs/combination_validation/models2_combination_full_general_entitytype_nolimit_triskill_full/summary.json` | TriSkill full | 1.5B, 7B, 14B, 32B DeepSeek-R1-Distill-Qwen | DAT, BATS, RAT, Metaphor | full | `final` |
| `comb_limit50_direct` | `outputs/combination_validation/models2_combination_limit50_direct/summary.json` | direct | models2 | DAT, BATS, RAT, Metaphor | 50 | `diagnostic` |
| `comb_limit50_triskill_general_entitytype` | `outputs/combination_validation/models2_combination_limit50_general_entitytype_triskill_full/summary.json` | TriSkill full | models2 | DAT, BATS, RAT, Metaphor | 50 | `diagnostic` |

Main full combinational scores:

| Model | DAT | BATS | RAT | Metaphor |
|---|---:|---:|---:|---:|
| deepseek-r1-distill-qwen-1.5b | 0.0000 -> 4.4571 | 0.1827 -> 0.2150 | 0.0069 -> 0.0139 | 0.1368 -> 0.1649 |
| deepseek-r1-distill-qwen-7b | 5.4625 -> 3.8459 | 0.4083 -> 0.4425 | 0.0208 -> 0.0694 | 0.3129 -> 0.3319 |
| deepseek-r1-distill-qwen-14b | 4.2710 -> 4.7337 | 0.5633 -> 0.5655 | 0.2569 -> 0.2569 | 0.4456 -> 0.4450 |
| deepseek-r1-distill-qwen-32b | 3.6811 -> 4.7337 | 0.5785 -> 0.5787 | 0.2847 -> 0.3194 | 0.4521 -> 0.4677 |

Paper-safe claim:

- Full exact-metric combinational evidence supports TriSkill as a
  profile-shifting method.
- The 1.5B and 32B models improve on all four combinational tasks.
- The 7B model improves on BATS, RAT, and Metaphor but regresses on DAT.
- The 14B model is mostly flat, with DAT/BATS slight gains, RAT unchanged, and
  Metaphor essentially unchanged/slightly down.
- BATS/RAT/Metaphor are convergent exact-match scoring interfaces, so expected
  gains are smaller than on open-ended divergent outputs.
- DAT should be interpreted cautiously because its full evaluation has `n=1`.

## qwen3.5-9b Direct Baseline and Current Full TriSkill Evidence

Run IDs:

| ID | Path | Method | Model | Tasks | Limit | Status |
|---|---|---|---|---|---|---|
| `qwen35_9b_direct_merged` | `/inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/outputs/huizong/merged_evalscope/summary.json` | direct | qwen3.5-9b | 9 tasks | merged full baseline | `final-candidate` |
| `qwen35_9b_triskill_9tasks_full_current` | `outputs/fulltask_validation/qwen35_9b_9tasks_full_triskill_current_triskill_full/summary.json` | TriSkill full | qwen3.5-9b | 9 tasks | full | `final-candidate` with caveats |

Scores:

| Task | Direct merged baseline | TriSkill current full | n direct | n TriSkill | Status |
|---|---:|---:|---:|---:|---|
| DAT | 6.8994 | 3.9620 | 1 | 1 | `superseded` by latest DAT selector diagnostics |
| BATS | 0.5232 | 0.5200 | 4000 | 4000 | `superseded` by BATS relation-consensus replay |
| RAT | 0.2153 | 0.2917 | 144 | 144 | `final-candidate` |
| Metaphor | 0.4470 | 0.4826 | 2953 | 2953 | `final-candidate` |
| AUT | 30.0000 | 40.3699 | 73 | 73 | `final-candidate` |
| CreativeMath | 0.4083 | 0.8760 | 605 | 605 | `final-candidate` |
| CS4 | 0.1909 | 0.3600 | 220 | 250 | `final-candidate`, note sample-count mismatch |
| NeoCoder | 0.2245 | 0.4556 | 1194 | 1194 | `final-candidate` |
| Transformation | 0.0000 | 0.9870 | 1 | 1308 | `not-comparable` because the user changed transformation eval code/setup |

Paper-safe claim:

- qwen3.5-9b currently provides the broadest TriSkill evidence, but it is not
  yet symmetric across models.
- For main-paper claims, avoid saying all 9 tasks improved because DAT/BATS are
  superseded and Transformation is not comparable to the merged baseline.
- The strongest qwen3.5-9b full-run claims are RAT, Metaphor, AUT,
  CreativeMath, CS4, and NeoCoder gains, subject to final verification of
  direct baseline compatibility.
- Transformation should be excluded from direct-vs-TriSkill comparison until
  the evaluator setup is made comparable.

## Latest qwen3.5-9b DAT/BATS Diagnostics

Run IDs:

| ID | Path | Method | Model | Tasks | Limit | Status |
|---|---|---|---|---|---|---|
| `qwen35_9b_dat_bats_iter3_embedspread_limit20` | `outputs/fulltask_validation/qwen35_9b_dat_bats_iter3_embedspread_limit20_triskill_full/summary.json` | TriSkill full | qwen3.5-9b | DAT, BATS | DAT full effectively n=1; BATS limit20 | `diagnostic` |
| `qwen35_9b_bats_iter4_consensus_limit20` | `outputs/fulltask_validation/qwen35_9b_bats_iter4_consensus_limit20_triskill_full/summary.json` | TriSkill full | qwen3.5-9b | BATS | 20 | `diagnostic` |
| `qwen35_9b_bats_iter4_consensus_full_replay` | `outputs/fulltask_validation/qwen35_9b_bats_iter4_consensus_full_replay/summary.json` | offline replay | qwen3.5-9b | BATS | 4000 cached artifacts | `replay` |

Scores:

| Task / run | Direct reference | TriSkill / replay | n | Status |
|---|---:|---:|---:|---|
| DAT iter3 embed-spread | 6.8994 | 7.4730 | 1 | `diagnostic`; supports latest DAT selector, but n=1 |
| BATS iter4 limit20 | 0.5232 | 0.9500 | 20 | `diagnostic`; too small for main claim |
| BATS iter4 full replay | 0.5232 | 0.5300 | 4000 | `replay`; useful evidence, should be rerun through standard evaluator for final table |

Paper-safe claim:

- Latest DAT/BATS changes appear promising for qwen3.5-9b.
- BATS full replay improves over the merged direct baseline, but it should be
  labeled replay-only unless rerun as a standard full evaluation.
- DAT remains a one-item semantic-diversity indicator.

## qwen3.5-9b Exploratory Full Evidence

Run IDs:

| ID | Path | Method | Model | Tasks | Limit | Status |
|---|---|---|---|---|---|---|
| `qwen35_9b_exploration_full_direct` | `outputs/exploration_validation/qwen35_9b_exploration_full_strict_output_gates_direct/summary.json` | direct | qwen3.5-9b | AUT, CreativeMath, CS4, NeoCoder | full | `superseded/diagnostic` |
| `qwen35_9b_exploration_full_triskill_strict_gates` | `outputs/exploration_validation/qwen35_9b_exploration_full_strict_output_gates_global_sched_triskill_full/summary.json` | TriSkill full | qwen3.5-9b | AUT, CreativeMath, CS4, NeoCoder | full | `superseded/diagnostic` |

Scores:

| Task | Direct full | TriSkill full | n direct | n TriSkill | Interpretation |
|---|---:|---:|---:|---:|---|
| AUT | 37.3288 | 10.8493 | 73 | 73 | negative; early workflow over-constrained list generation |
| CreativeMath | 0.6793 | 0.8860 | 605 | 605 | positive |
| CS4 | 0.3760 | 0.3927 | 250 | 247 | small positive, note sample-count mismatch |
| NeoCoder | 0.1198 | 0.0687 | 1194 | 1194 | negative; code correctness degraded |

Paper-safe claim:

- This run is valuable mainly as failure/boundary evidence.
- It should not be used as the final qwen3.5-9b exploratory result because
  later generic AUT and NeoCoder modules changed the method.
- It supports the narrative that exploratory workflow pressure can improve some
  modalities while harming fluency or executable correctness.

## qwen3.5-9b Exploratory Limit-20 Accepted Diagnostics

Run IDs:

| ID | Path | Method | Model | Tasks | Limit | Status |
|---|---|---|---|---|---|---|
| `qwen35_9b_exploration_limit20_forbidprompt_livingaut_direct` | `outputs/exploration_validation/qwen35_9b_exploration_limit20_forbidprompt_livingaut_direct/summary.json` | direct | qwen3.5-9b | AUT, CreativeMath, CS4, NeoCoder | 20 per task, CS4 aggregates 100 rows | `diagnostic` |
| `qwen35_9b_exploration_limit20_balanced_autpolish_triskill` | `outputs/exploration_validation/qwen35_9b_exploration_limit20_balanced_autpolish_triskill_full/summary.json` | TriSkill full | qwen3.5-9b | AUT, CreativeMath, CS4, NeoCoder | 20 per task, CS4 aggregates 100 rows | `diagnostic` |
| `qwen35_9b_exploration_limit20_map_forloop_gate_triskill` | `outputs/exploration_validation/qwen35_9b_exploration_limit20_map_forloop_gate_triskill_full/summary.json` | TriSkill full | qwen3.5-9b | AUT, CreativeMath, CS4, NeoCoder | 20 per task, CS4 aggregates 100 rows | `diagnostic` |
| `qwen35_9b_neocoder_limit20_map_forloop_gate_triskill` | `outputs/exploration_validation/qwen35_9b_neocoder_limit20_map_forloop_gate_triskill_full/summary.json` | TriSkill full | qwen3.5-9b | NeoCoder | 20 | `diagnostic` |

Accepted diagnostic comparison:

| Task | Direct limit20 | TriSkill limit20 | n direct | n TriSkill | Status |
|---|---:|---:|---:|---:|---|
| AUT | 41.6500 | 44.6500 | 20 | 20 | positive diagnostic |
| CreativeMath | 0.8000 | 1.0000 | 20 | 20 | positive diagnostic |
| CS4 | 1.0000 | 1.0000 | 100 | 100 | tied ceiling diagnostic |
| NeoCoder | 0.0000 | 0.4500 | 20 | 20 | positive diagnostic in focused run |

Additional map/filter four-task run:

| Task | TriSkill map/filter gate | n | Note |
|---|---:|---:|---|
| AUT | 43.5000 | 20 | Positive over direct 41.65, below accepted balanced_autpolish 44.65 |
| CreativeMath | 1.0000 | 20 | Positive/tied with accepted |
| CS4 | 1.0000 | 100 | Ceiling |
| NeoCoder | 0.4000 | 20 | Positive over direct 0.00, below focused NeoCoder 0.45 |

Paper-safe claim:

- Limit-20 qwen3.5-9b diagnostics show a broadly positive profile shift for the
  current exploratory modules, but they are not full-run evidence.
- Use them for design validation, ablation discussion, or motivation for a full
  rerun.

## Transformation Evidence

Run IDs:

| ID | Path | Method | Models | Task | Limit | Status |
|---|---|---|---|---|---|---|
| `transformation_limit5_direct` | `outputs/transformation_validation/models2_transformation_limit5_fulljudge2_operator_general_direct/summary.json` | direct | 1.5B, 7B, 14B, 32B DeepSeek-R1-Distill-Qwen | Transformation | 5 | `diagnostic` |
| `transformation_limit5_openanchor_triskill` | `outputs/transformation_validation/models2_transformation_limit5_fulljudge2_openanchor_triskill_full/summary.json` | TriSkill full | 1.5B, 7B, 14B, 32B DeepSeek-R1-Distill-Qwen | Transformation | 5 | `diagnostic` |

Scores:

| Model | Direct | TriSkill open-anchor | n | Status |
|---|---:|---:|---:|---|
| deepseek-r1-distill-qwen-1.5b | 0.0000 | 0.0000 | 5 | no gain |
| deepseek-r1-distill-qwen-7b | 0.2000 | 0.6000 | 5 | positive diagnostic |
| deepseek-r1-distill-qwen-14b | 1.0000 | 0.8000 | 5 | negative diagnostic from high direct baseline |
| deepseek-r1-distill-qwen-32b | 0.2000 | 0.6000 | 5 | positive diagnostic |

Paper-safe claim:

- Transformation results are promising but low-sample and judge-dependent.
- They support the method design narrative, not a definitive empirical claim.
- Full transformation evaluation requires a stable evaluator and comparable
  direct baseline.

## Rejected / Archived Iterations

Archived paths under `outputs/_archive_20260510_low_value_runs/` and
`outputs/_archive_20260511_low_value_runs/` should not support positive claims.
They may support failure-mode analysis if named explicitly.

Important rejected examples:

| Run group | Path pattern | Reason |
|---|---|---|
| early symbolic/task-specific combinational anchors | `outputs/_archive_20260510_low_value_runs/combination_validation/*anchor*` | Rejected because user required generic modules without task-specific directed tuning. |
| AUT over-filter iterations | `outputs/_archive_20260511_low_value_runs/aut_overfilter_iterations/*` | Rejected because extra safety/role filters reduced AUT fluency/score. |
| NeoCoder over-strict final gate | `outputs/_archive_20260511_low_value_runs/neocoder_overstrict_final_gate/*` | Rejected because correctness dropped too much. |
| NeoCoder constraint prompt oversteer | `outputs/_archive_20260511_low_value_runs/neocoder_constraint_prompt_oversteer/*` | Rejected because module-wide constraint reminders over-steered generation. |

## Current Evidence Gaps

These gaps should be resolved before final manuscript tables:

- Full TriSkill evaluation has not been run across the closed-source / large
  model set in `merged_evalscope`.
- Latest DAT/BATS qwen3.5-9b selectors need a standard full run, not only
  limit20 and BATS replay, before entering a main table.
- qwen3.5-9b exploratory current modules need a full rerun if we want to claim
  full-task improvements for AUT/CreativeMath/CS4/NeoCoder.
- Transformation results need comparable evaluator setup after the user's
  transformation-code change.
- Some direct merged baselines have partial task coverage; paper tables must
  either filter to complete models or mark missing cells.
- LLM-judge metrics must be separated from exact/deterministic metrics.

## Current Paper Claim Boundary

Allowed now:

- TriSkill is a leakage-free, definition-guided, metric-conditioned elicitation
  framework.
- Full combinational exact-metric results show model-dependent profile shifts,
  with full four-task gains for 1.5B and 32B.
- qwen3.5-9b current evidence suggests strong potential across several tasks,
  but the latest method needs clean full reruns for main-table claims.
- Negative and superseded runs demonstrate meaningful trade-offs, especially in
  exploratory generation and code correctness.

Not allowed yet:

- "TriSkill improves all tasks on all models."
- "TriSkill has been fully validated on all closed-source models."
- "Transformation full results are directly comparable to the merged direct
  baseline."
- "BATS latest full result is final standard-evaluator evidence" unless the
  replay is rerun through the official evaluation pipeline.
