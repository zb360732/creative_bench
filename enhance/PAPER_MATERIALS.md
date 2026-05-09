# TriSkill Paper Materials

This document maintains paper-ready material collected during the benchmark and
elicitation work. It is intentionally separate from `solution.md`: `solution.md`
defines the target framework, while this file records claims, evidence,
wording, results, caveats, and future paper assets.

## Paper Positioning

Working title:

```text
TriSkill: Definition-Guided and Metric-Conditioned Creativity Elicitation for Large Language Models
```

Core framing:

```text
We study creativity elicitation, not only creativity measurement. Given a fixed
LLM and a creativity benchmark, TriSkill asks whether a definition-guided and
metric-conditioned test-time workflow can systematically shift the model's
observed creativity profile without training, benchmark modification, or access
to hidden references.
```

One-sentence method summary:

```text
TriSkill profiles each creativity task by its creativity definition, maps
heterogeneous raw benchmark metrics into canonical creativity objectives,
composes reusable inference-time skills, and executes a leakage-free workflow
that generates, verifies, selects, and normalizes candidate outputs.
```

Short contribution list:

- Creativity elicitation as measurable profile shift rather than static prompt-only evaluation.
- Definition-guided routing across combinational, exploratory, and transformational creativity.
- Metric-conditioned skill composition through canonical creativity metrics.
- Leakage-free test-time workflow that uses only visible task input, output schema, task type, and metric semantics.
- Empirical evidence on combinational creativity tasks with full-run results for `models2.json`.

## Method Vocabulary

Recommended names:

- Framework: `TriSkill`
- Full method: `TriSkill full`
- Paper method name: `Definition-Guided and Metric-Conditioned Creativity Elicitation`
- Evaluation view: `creativity profile shift`
- Main workflow components:
  - `Task Profiler`
  - `Metric Abstraction Layer`
  - `Workflow Router`
  - `Skill Composer`
  - `Workflow Executor`
  - `Verifier / Selector`
  - `Output Normalizer`

Central slogan:

```text
Definition chooses the workflow.
Metrics choose the skills.
Skills guide inference.
The benchmark measures the profile shift.
```

## Generalized Operator Library

The module library should be described as definition-derived operators, not as
benchmark-task prompts. A task profile selects a subset of these reusable
operators using public creativity type and canonical metric objectives.

Combinational operators:

| Operator | Role |
|---|---|
| `unit_extraction` | Extract recombinable units: words, concepts, entities, properties, relations, contexts, and constraints. |
| `relation_property_abstraction` | Abstract the relation, property, role, or semantic axis connecting units. |
| `candidate_recombination` | Generate new combinations by transferring relations, bridging units, mapping properties, or assembling distant semantic units. |
| `constraint_preservation` | Preserve hard constraints after recombination: entity type, relation direction, part of speech, context, grammar, feasibility, and schema. |
| `combination_verification` | Verify that the recombined candidate meaningfully connects the required units. |
| `diversity_filtering` | Select a semantically diverse subset when the task rewards divergent combinations. |
| `output_normalization` | Convert the selected candidate into the benchmark-required final format. |

Exploratory operators:

| Operator | Role |
|---|---|
| `constraint_space_mapping` | Map hard constraints, soft preferences, forbidden outputs, format, and creative freedom. |
| `exploration_axis_expansion` | Expand distinct strategy axes or search directions. |
| `candidate_generation` | Produce multiple candidates across axes. |
| `coverage_balancing` | Preserve category, mechanism, or solution-structure coverage. |
| `novelty_transformation` | Push candidates toward novelty without violating feasibility. |
| `semantic_deduplication` | Remove duplicate or near-duplicate ideas. |
| `feasibility_evaluation` | Reject infeasible, irrelevant, incoherent, unsafe, or constraint-breaking candidates. |
| `portfolio_selection` | Select the best candidate under correctness, novelty, flexibility, and appropriateness trade-offs. |

Transformational operators:

| Operator | Role |
|---|---|
| `rule_change_extraction` | Extract changed rules and goals. |
| `legacy_assumption_mapping` | Identify old assumptions, mechanisms, interfaces, measurements, and language. |
| `breakage_propagation` | Trace how changed rules break the legacy system. |
| `primitive_induction` | Introduce new variables, roles, primitives, interfaces, or validity conditions. |
| `system_reconstruction` | Rebuild the system around new primitives rather than patching the old one. |
| `performance_reanchoring` | Restore performance criteria under the new rule world. |
| `norm_interface_establishment` | Define new standards, interfaces, training language, validation, and coordination norms. |
| `residue_audit` | Remove hidden dependence on invalid old-world assumptions. |
| `goal_coverage_verification` | Verify that stated goals are covered by concrete mechanisms. |

Important framing:

```text
The operator library is derived from creativity definitions. Benchmark tasks do
not receive bespoke prompts; they instantiate profiles that choose reusable
operators from the library.
```

## Task Taxonomy

Current implemented taxonomy:

- `combinational`: DAT, BATS, RAT, Metaphor
- `exploratory`: AUT, CreativeMath, CS4, NeoCoder
- `transformational`: Transformation-style rule/system rebuilding tasks

The current validation protocol has two layers: full combinational validation
for stable non-LLM-judge metrics, and `limit=5` LLM-judge validation for
exploratory/transformational tasks using the original benchmark judge pipeline.

Combinational subtypes:

| Task | Subtype | Cognitive operation | Metric character |
|---|---|---|---|
| DAT | divergent semantic recombination | assemble semantically distant words | open-ended semantic distance |
| BATS | analogical relation transfer | infer A:B relation and apply to C:? | single-answer accuracy |
| RAT | remote associative bridging | find one bridge word across three clues | single-answer accuracy |
| Metaphor | cross-domain property mapping | map source-domain property into context | single-answer accuracy |

Important paper nuance:

```text
BATS, RAT, and Metaphor are combinational in their cognitive operation, but
convergent in their scoring interface. Their final score rewards exact answer
selection, not the diversity or quality of intermediate combinations.
```

This explains why gains are expected to be smaller than on open-ended
divergent tasks such as DAT.

## Current Implementation

Code location:

- `enhance/triskill/profiles.py`: task profiles, raw metric aliases, canonical metrics, budgets, skill lists
- `enhance/triskill/executor.py`: workflow execution, direct seed, skill trace, artifact assembly
- `enhance/triskill/runtime_skills.py`: runtime skill prompts, verifier behavior, candidate/selection update
- `enhance/triskill/normalizer.py`: final benchmark-compatible answer normalization
- `enhance/run_combination_validation.py`: direct vs TriSkill validation driver over evalscope

Current combinational workflow behavior:

- DAT:
  - operators: unit extraction, candidate recombination, diversity filtering, constraint preservation, output normalization
  - objective: maximize semantic diversity while preserving lexical and format validity
- BATS:
  - operators: unit extraction, relation/property abstraction, candidate recombination, combination verification, constraint preservation, output normalization
  - objective: preserve relation direction, entity type, and abstraction level
- RAT:
  - operators: unit extraction, relation/property abstraction, candidate recombination, combination verification, constraint preservation, output normalization
  - objective: find one bridge satisfying all visible clues
- Metaphor:
  - operators: unit extraction, relation/property abstraction, candidate recombination, combination verification, constraint preservation, output normalization
  - objective: preserve metaphorical fit and context fit

Current conservative design choice:

```text
For BATS and Metaphor, direct-seed anchoring is used to avoid replacing a
correct direct answer with a weaker workflow candidate. This protects accuracy
but limits the maximum improvement size.
```

Open-ended task safeguard:

```text
For solution/story/code/reconstruction tasks, TriSkill first records a direct
answer anchor, then runs the workflow, and finally synthesizes a benchmark-ready
answer using the direct answer as a fidelity anchor. Workflow artifacts are
allowed to improve novelty, coverage, or flexibility only if they do not reduce
correctness, coherence, feasibility, or constraint satisfaction.
```

This safeguard is definition-derived rather than benchmark-answer-specific: it
addresses the generic failure mode of open-ended creative elicitation where
search and transformation can increase exploration while degrading the base
quality of the final answer.

## LLM-Judge Validation Snapshot

Configuration:

- Models: `/root/benchmark/evalscope/run/models2.json`
- Sample size: `limit=5`
- Parallelism: `--max-parallel 64`, `--eval-batch-size 64`
- Judge config: local `llm_judge2` endpoint injected by `enhance/run_evalscope_with_judge.py`
- Evalscope source: unchanged

qwen3.5-9b focused iteration after strict output gates:

| Task | Direct | TriSkill strict gates | Main interpretation |
|---|---:|---:|---|
| AUT fluency | 38.2 | 24.0 | Still below the direct multi-round AUT baseline, but recovered strongly from earlier placeholder-contaminated TriSkill runs (`7 -> 17 -> 24`). |
| AUT originality | 59.77 | 53.49 | Near-direct originality after raw-use filtering; remaining loss is mostly lower fluency and applicability. |
| CreativeMath novelty | 0.60 | 0.80 | Positive novelty shift while correctness and appropriateness remain `1.0`. |
| CreativeMath originality | 0.48 | 0.62 | Positive originality shift, though fine-grained novel-unknown does not improve beyond direct in the strict-gate run. |
| CS4 fluency/score | 0.32 | 0.44 | Main CS4 score improves, but story-quality and constraint-coverage metrics remain below direct. |
| NeoCoder follow/fluency | 0.60 / 0.60 | 1.00 / 1.00 | Format and constraint-following improve, but correctness remains `0.0`; current code modality control is not enough for executable correctness. |

Mechanistic note:

```text
Strict output gates improved the paper-useful hygiene of exploratory workflows:
AUT no longer treats schema or planning text as uses, CS4 better extracts story
drafts from planning shells, and NeoCoder distinguishes format following from
actual executable correctness. The resulting profile shift is positive on
CreativeMath novelty/originality and CS4 fluency, but not a uniform gain.
```

Full qwen3.5-9b exploratory comparison after strict output gates:

| Task | Metric view | Direct | TriSkill full | Delta | Paper interpretation |
|---|---|---:|---:|---:|---|
| AUT | main score / fluency | 37.3288 | 10.8493 | -26.4795 | Negative full-run result; the workflow over-constrains list generation and loses fluency/diversity despite stronger applicability. |
| AUT | applicability | 0.7532 | 3.0285 | +2.2753 | The generated uses are judged more applicable when present, but there are too few valid/diverse uses. |
| CreativeMath | main score / correctness | 0.6793 | 0.8860 | +0.2067 | Strongest full-run exploratory gain; fidelity anchoring improves correctness and appropriateness. |
| CreativeMath | novelty | 0.3504 | 0.2810 | -0.0694 | Correctness gain trades off against novelty/originality. |
| CS4 | main score / fluency | 0.3760 | 0.3927 | +0.0167 | Small main-score gain only; not enough to claim broad story-quality improvement. |
| CS4 | quality metrics | mixed | lower | negative | Grammar, coherence, likability, flexibility, appropriateness, novelty, and QUC all drop. |
| NeoCoder | follow / fluency | 0.8040 | 0.8802 | +0.0762 | Formatting and constraint-following improve. |
| NeoCoder | correctness / score | 0.1198 | 0.0687 | -0.0511 | Correctness loss dominates; code generation needs stronger executable validation. |

Full-run conclusion:

```text
For qwen3.5-9b exploratory tasks, TriSkill should be reported as a profile
shift rather than a general improvement method. It improves CreativeMath
correctness/appropriateness and slightly improves CS4's main fluency score, but
it currently hurts AUT divergent output volume/diversity and NeoCoder executable
correctness. This result is useful because it identifies the boundary between
creative search pressure and final-answer quality preservation.
```

AUT with corrected output normalization:

| Model | Main result |
|---|---|
| 1.5B | Full gain on all five AUT metrics: fluency `17.8 -> 20.4`, elaboration `4.50 -> 5.68`, flexibility `11.84 -> 18.31`, originality `8.60 -> 58.61`, applicability `0.595 -> 0.623`. |
| 7B | Improves flexibility, originality, and elaboration; fluency/applicability drop. |
| 14B/32B | Originality or elaboration may improve, but fluency/applicability regress. |

CreativeMath after open-ended fidelity anchor:

| Model | Main result |
|---|---|
| 1.5B | Correctness/appropriateness/overall improve `0.2 -> 1.0`; novelty remains zero. |
| 7B | Correctness/appropriateness/overall improve `0.6 -> 1.0`; novelty/originality hold. |
| 32B | Correctness/appropriateness/overall improve `0.6 -> 1.0`; novelty/originality drop. |
| 14B | Overall and correctness hold at `1.0`; novelty/originality drop. |

Transformation after open-ended fidelity anchor:

| Model | Main result |
|---|---|
| 7B | Full gain on all primary metrics: fluency `0.2 -> 0.6`, novelty `0.4 -> 0.9`, appropriateness `0.5 -> 1.2`, flexibility `0.0909 -> 0.6364`. |
| 32B | Full gain on all primary metrics: fluency `0.2 -> 0.6`, novelty `0.4 -> 1.0`, appropriateness `0.6 -> 1.6`, flexibility `0.1818 -> 0.7273`. |
| 1.5B | All metrics remain zero. |
| 14B | Regresses from a strong direct baseline. |

NeoCoder:

| Model | Main result |
|---|---|
| 7B | Follow-constraints and fluency improve `0.8 -> 1.0`; correctness remains zero. |
| 1.5B/14B | Mostly unchanged at zero correctness. |
| 32B | Follow-constraints/fluency improve but correctness drops `0.2 -> 0.0`. |

CS4:

```text
CS4 remains the clearest negative result. Openanchor reduces the previous
catastrophic quality collapse but still often trades coherence, fluency, and
QUC for modest flexibility/grammar gains. This is useful paper evidence that
not all creative objectives benefit from generic elicitation; story-level
constraint satisfaction appears to need stronger global coherence control.
```

Paper-safe interpretation:

```text
TriSkill produces measurable creativity-profile shifts rather than uniform
dominance. The strongest positive story is model- and task-family-specific:
small models benefit on AUT, medium/large models benefit on transformational
reconstruction, and open-ended math benefits mainly in correctness/appropriateness.
CS4 exposes the trade-off between exploration pressure and narrative coherence.
```

## Leakage-Free Protocol

The method must not use:

- gold answers
- hidden references
- scoring-only candidate answers
- post-hoc scores
- test-set aggregate statistics
- task-specific answer maps or symbolic lookup tables

Allowed inputs:

- visible task prompt
- visible choices or constraints, if provided by the benchmark prompt
- task name
- creativity level
- output schema
- metric semantics
- canonical metric objectives

Important history:

```text
Earlier small-sample attempts included task-specific symbolic/lexical adapters.
These were removed after the no-leakage/no-task-specific-tuning constraint was
clarified. The retained method is the general workflow only.
```

This should be mentioned in internal reproducibility notes, not necessarily in
the main paper unless discussing method hygiene.

## Full Combinational Validation

Run configuration:

```bash
python enhance/run_combination_validation.py \
  --skip-direct \
  --limit none \
  --work-dir /root/benchmark/outputs/combination_validation \
  --run-name models2_combination_full_general_entitytype_nolimit \
  --max-tokens 2048 \
  --request-timeout 180 \
  --max-parallel 64 \
  --eval-batch-size 64 \
  --judge-worker-num 1 \
  --triskill-method triskill_full
```

Model list:

```text
/root/benchmark/evalscope/run/models2.json
```

Artifact paths:

- Direct summary: `outputs/combination_validation/models2_combination_full_general_entitytype_nolimit_direct/summary.json`
- TriSkill summary: `outputs/combination_validation/models2_combination_full_general_entitytype_nolimit_triskill_full/summary.json`

Full-run results:

| Model | Task | Direct | TriSkill | Delta |
|---|---:|---:|---:|---:|
| deepseek-r1-distill-qwen-1.5b | DAT | 0.0000 | 4.4571 | +4.4571 |
| deepseek-r1-distill-qwen-1.5b | BATS | 0.1827 | 0.2150 | +0.0323 |
| deepseek-r1-distill-qwen-1.5b | RAT | 0.0069 | 0.0139 | +0.0070 |
| deepseek-r1-distill-qwen-1.5b | Metaphor | 0.1368 | 0.1649 | +0.0281 |
| deepseek-r1-distill-qwen-14b | DAT | 4.2710 | 4.7337 | +0.4627 |
| deepseek-r1-distill-qwen-14b | BATS | 0.5633 | 0.5655 | +0.0022 |
| deepseek-r1-distill-qwen-14b | RAT | 0.2569 | 0.2569 | +0.0000 |
| deepseek-r1-distill-qwen-14b | Metaphor | 0.4456 | 0.4450 | -0.0006 |
| deepseek-r1-distill-qwen-32b | DAT | 3.6811 | 4.7337 | +1.0526 |
| deepseek-r1-distill-qwen-32b | BATS | 0.5785 | 0.5787 | +0.0002 |
| deepseek-r1-distill-qwen-32b | RAT | 0.2847 | 0.3194 | +0.0347 |
| deepseek-r1-distill-qwen-32b | Metaphor | 0.4521 | 0.4677 | +0.0156 |
| deepseek-r1-distill-qwen-7b | DAT | 5.4625 | 3.8459 | -1.6166 |
| deepseek-r1-distill-qwen-7b | BATS | 0.4083 | 0.4425 | +0.0342 |
| deepseek-r1-distill-qwen-7b | RAT | 0.0208 | 0.0694 | +0.0486 |
| deepseek-r1-distill-qwen-7b | Metaphor | 0.3129 | 0.3319 | +0.0190 |

Useful result statements:

- `32B` improves on all four combinational tasks.
- `1.5B` also improves on all four combinational tasks.
- `7B` improves on three convergent tasks but regresses on DAT.
- `14B` is mostly flat, with tiny positive/negative changes.
- DAT gains are larger, but DAT full has only one item, so it should be treated as profile-shift evidence rather than a robust aggregate alone.
- BATS/RAT/Metaphor gains are smaller because they are exact-match single-answer tasks.

Recommended paper phrasing:

```text
On the full combinational benchmark, TriSkill yields consistent gains for the
strongest evaluated model across divergent association, analogical transfer,
remote association, and metaphorical mapping. The improvement is largest on DAT
and smaller on convergent single-answer tasks, reflecting the fact that exact
answer accuracy only partially captures the intermediate combinational process.
```

Avoid claiming:

- large universal gains across all models
- solved combinational creativity
- statistically strong DAT conclusion from a single DAT item
- improvements from hidden answer access
- task-specific prompt hacking

## Why Gains Are Small

Paper-useful explanation:

```text
The magnitude of improvement is constrained by the scoring interface. Three of
the four combinational tasks are convergent single-word tasks. Although their
cognitive process requires relation abstraction, associative bridging, or
cross-domain mapping, their metrics reward only final exact-match accuracy. As a
result, TriSkill can improve relation checking and candidate selection, but it
cannot receive credit for plausible intermediate candidates that do not match
the benchmark reference.
```

Mechanistic reasons:

- BATS/RAT/Metaphor require selecting one exact word, so verifier quality is the bottleneck.
- The method cannot use hidden gold answers, so selection relies on model self-evaluation.
- Direct baselines are already non-trivial for larger models, leaving limited headroom.
- Conservative direct-seed anchoring reduces harmful replacements but also reduces possible upside.
- Some improvements are diluted by exact-match lexical variation.

## Failure and Risk Notes

Known risks:

- Self-verification can select fluent but wrong candidates.
- Open-ended and convergent combinational objectives can conflict.
- DAT score can be sensitive to a single final word list because the dataset contains one test item.
- Exact-match tasks under-credit useful but non-reference creative associations.
- Runtime cost is high because full workflow uses multiple LLM calls per sample.

Observed runtime issue and fix:

```text
During full validation, `--max-parallel 64` did not translate to 64 backend
running requests because the scheduler originally divided parallelism across
task-level cache jobs. Resume runs with only one incomplete cache could still
underuse the backend, and partially complete caches made the observed running
count much smaller than requested. The validation driver now builds a global
queue of unfinished sample rows and appends rows under per-cache locks, so
`--max-parallel 64` is spent on unfinished samples rather than task containers.
```

Commit:

```text
8869382 Improve full validation resume scheduling
```

## Validation and Version Record

Relevant commits:

- `be317ee Optimize general TriSkill combinational workflow`
- `cf92743 Allow full combination validation runs`
- `8869382 Improve full validation resume scheduling`

Validation checks:

```bash
python -m py_compile enhance/run_combination_validation.py
PYTHONPATH=enhance python -m unittest discover -s enhance/tests -p 'test_*.py'
git diff --name-only -- evalscope
```

Observed validation status:

- `25` unit tests passed.
- `evalscope` had no source diff.
- Full direct and TriSkill summaries completed with status `ok` for all four models.

## Candidate Figures and Tables

Potential tables:

- Table 1: Creativity taxonomy and task mapping.
- Table 2: Raw metric to canonical metric to skill mapping.
- Table 3: Full combinational validation results.
- Table 4: Model-wise win/tie/loss summary.
- Table 5: Ablation results once run.

Potential figures:

- TriSkill architecture diagram:
  `Task Profiler -> Metric Abstraction -> Workflow Router -> Skill Composer -> Executor -> Verifier -> Normalizer`
- Creativity profile shift radar chart for the selected main model.
- Divergent vs convergent combinational task comparison.
- Runtime/cost breakdown by task.

## Needed Next Paper Assets

High-priority:

- Run ablations on the main model:
  - without verifier
  - without direct seed
  - prompt-only TriSkill
  - generic creativity prompt
  - budget-matched multi-sample direct
- Add candidate coverage diagnostics:
  - whether the final correct answer appeared in candidate pool
  - whether verifier selected it
  - how often normalization failed
- Add qualitative examples:
  - one DAT improvement example
  - one RAT bridge improvement example
  - one Metaphor improvement example
  - one failure where verifier picks a plausible but wrong word

Medium-priority:

- Report cost in `num_llm_calls` and wall time.
- Compare profile shift across model sizes.
- Separate divergent and convergent combinational subsets in analysis.
- Add confidence intervals or bootstrap for BATS/RAT/Metaphor.

Low-priority:

- Extend validation to exploratory and transformational tasks once the LLM-judge pipeline is reliable.
- Add human-readable trace examples from `skill_trace`.

## Draft Abstract Fragment

```text
Large language models are commonly evaluated on creativity benchmarks using the
raw benchmark prompt, but this static protocol does not distinguish latent
creative capacity from the ability of a prompt to elicit it. We propose
TriSkill, a definition-guided and metric-conditioned test-time elicitation
framework. TriSkill routes each task according to a creativity definition,
maps heterogeneous benchmark metrics into canonical creativity objectives, and
composes reusable inference-time skills such as semantic expansion, relation
abstraction, associative bridge search, metaphorical property mapping, and
constraint-aware verification. The framework is leakage-free: it uses only
visible task inputs, output schemas, and metric semantics, without hidden
answers or benchmark-specific answer maps. On combinational creativity tasks,
TriSkill induces measurable profile shifts, with the strongest evaluated model
improving across divergent association, analogical transfer, remote association,
and metaphorical mapping.
```

## Draft Limitations Fragment

```text
Our current evidence is strongest for combinational creativity and should not
be interpreted as universal improvement across all creativity forms. In
addition, three of the four combinational tasks use exact-match single-answer
accuracy, which under-represents the quality of intermediate creative
associations and makes verifier selection the dominant bottleneck. DAT provides
an open-ended divergent signal but contains only one full-test item in the
current benchmark. Future work should add broader open-ended tasks, stronger
budget-matched baselines, and candidate-level diagnostics.
```
