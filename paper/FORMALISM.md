# Formalism

This document defines the paper's core objects and protocol. It is written so
that sections can later be moved into the manuscript with minimal rewriting.

## 1. Notation Summary

| Symbol | Meaning |
|---|---|
| `M` | A fixed language model being evaluated. |
| `B` | A creativity benchmark consisting of multiple tasks. |
| `T` | A benchmark task, such as DAT, BATS, AUT, or Transformation. |
| `x` | A task instance visible to the model or elicitation method. |
| `v(x)` | The visible view of an instance: prompt, public fields, schema, and visible constraints. |
| `h(x)` | Hidden scoring information: answers, references, labels, scoring-only candidates, or post-hoc scores. |
| `Y_T` | The output space or answer schema for task `T`. |
| `m_T` | The raw benchmark metric or metric vector for task `T`. |
| `c_T` | The canonical creativity objective vector derived from `m_T`. |
| `l_T` | The creativity level of task `T`: combinational, exploratory, or transformational. |
| `P_T` | The task profile used by TriSkill. |
| `S` | The reusable skill/operator library. |
| `W_T` | The workflow selected for task `T`. |
| `E` | An elicitation method that maps visible input to an output. |
| `y` | The final model answer submitted to the benchmark evaluator. |
| `R(M, E, T)` | The measured result of model `M` under elicitation method `E` on task `T`. |
| `Delta(M, E, T)` | The profile shift between direct prompting and elicitation method `E` on task `T`. |

## 2. Creativity Task Instance

**Definition 1: Creativity task instance.**

A creativity task instance is a tuple:

```text
x = (p, a, s, q)
```

where:

- `p` is the natural-language prompt or problem statement;
- `a` is a set of public auxiliary fields, such as visible words, categories,
  constraints, rules, or goals;
- `s` is the required output schema;
- `q` is hidden scoring information used by the benchmark evaluator.

The model-facing part of the instance is:

```text
v(x) = (p, a_visible, s)
```

The evaluator-only part is:

```text
h(x) = q
```

TriSkill and direct prompting may use `v(x)` but must not use `h(x)`.

Examples of visible information:

- task prompt;
- public input fields such as `word_a`, `word_b`, `word_c`;
- visible constraints;
- output schema;
- task name or task type;
- public metric semantics, such as "semantic diversity" or "relation
  validity."

Examples of hidden information:

- `answer`;
- `answers`;
- `target`;
- `target_words`;
- `candidate_answers` when used for scoring;
- `reference`;
- `gold`;
- `label`;
- `solution`;
- post-hoc scores;
- test-set statistics.

## 3. Leakage-Free Protocol

**Definition 2: Leakage-free elicitation protocol.**

An elicitation method is leakage-free for a benchmark task if, for every
instance `x`, the method's input is restricted to:

```text
v(x), task name, output schema, visible constraints, and public metric semantics
```

and never includes:

```text
h(x), hidden references, gold answers, scoring-only candidate lists,
test-set statistics, or post-hoc scores.
```

In implementation terms, TriSkill enforces this protocol through a visible-field
filter. The current hidden-field blacklist includes:

```text
answer, answers, reference, references, gold, gold_answer, target, targets,
label, labels, candidate_answers, correct_answer, target_words, solution,
solutions
```

The visible-field allowlist includes public fields such as:

```text
id, item_id, query, question, prompt, input, category, category_name,
task_name, word_a, word_b, word_c, direction, relation_type, metaphor_word,
novelty, item, constraints, rules, goals, metadata
```

**Protocol statement for the paper.**

TriSkill does not modify the benchmark evaluator and does not access hidden
answers, references, labels, scoring-only candidate lists, or test-set
statistics. It transforms only the visible prompt, visible fields, output schema,
task type, visible constraints, and public metric semantics into a final answer.

## 4. Creativity Levels

**Definition 3: Creativity level.**

Each benchmark task is assigned to one of three creativity levels:

```text
l_T in {combinational, exploratory, transformational}
```

The current task taxonomy is:

| Level | Task family | Current tasks | Central operation |
|---|---|---|---|
| Combinational | Combine or bridge existing units | DAT, BATS, RAT, Metaphor | Extract units, abstract relation/property, recombine, verify fit. |
| Exploratory | Search within or expand a constrained possibility space | AUT, CreativeMath, CS4, NeoCoder | Map constraints, explore axes, generate candidates, evaluate feasibility. |
| Transformational | Rebuild a system after rule or assumption changes | Transformation | Extract rule changes, remove legacy assumptions, induce primitives, reconstruct system. |

This taxonomy is not a claim that every task is purely one type. It is a
workflow assignment: the level determines which family of operators TriSkill
uses.

## 5. Raw Metrics and Canonical Objectives

Benchmarks often expose heterogeneous task metrics: exact-match accuracy,
semantic distance, novelty, fluency, correctness, constraint satisfaction,
execution validity, and LLM-judge scores. TriSkill does not treat these raw
metrics as interchangeable scalar scores. It maps each task's raw metrics into
canonical creativity objectives.

**Definition 4: Canonical objective mapping.**

For each task `T`, let:

```text
m_T = (m_1, ..., m_k)
```

be the raw benchmark metric vector. TriSkill defines a public mapping:

```text
g: m_T -> c_T
```

where `c_T` is a vector of canonical objectives such as:

```text
semantic_diversity
relation_validity
associative_bridge
metaphorical_fit
fluency
flexibility
novelty
appropriateness
constraint_satisfaction
coherence
execution_validity
rule_utilization
system_reconstruction
performance_restoration
norm_establishment
old_assumption_removal
goal_coverage
format_validity
```

The mapping uses public metric semantics, not hidden labels. For example:

| Raw metric | Canonical objectives |
|---|---|
| `dat_semantic_distance` | `semantic_diversity`, `lexical_validity`, `format_validity` |
| `bats_accuracy` | `relation_validity`, `lexical_validity`, `format_validity` |
| `rat_accuracy` | `associative_bridge`, `relation_validity`, `lexical_validity`, `format_validity` |
| `metaphor_accuracy` | `metaphorical_fit`, `relation_validity`, `lexical_validity`, `format_validity` |
| `aut_fluency` | `fluency` |
| `aut_flexibility` | `flexibility` |
| `aut_originality` | `novelty` |
| `correctness` | `appropriateness`, `execution_validity` |
| `constraint_satisfaction` | `constraint_satisfaction` |
| `rule_coverage` | `rule_utilization` |
| `goal_coverage` | `goal_coverage` |

Metric conditioning means that public metric semantics influence workflow and
skill selection. It does not mean that TriSkill sees ground-truth labels,
references, or scores for the evaluated instance.

## 6. Task Profile

**Definition 5: Task profile.**

A TriSkill task profile is:

```text
P_T = (T, l_T, m_T, c_T, s_T, W_T, b_T)
```

where:

- `T` is the task name;
- `l_T` is the creativity level;
- `m_T` is the raw metric vector;
- `c_T` is the canonical objective vector;
- `s_T` is the output schema;
- `W_T` is the selected workflow;
- `b_T` is the inference budget, such as candidate count, sampling temperature,
  or max tokens.

Task profiles are schema and metric adapters. They are not answer maps. A
profile can specify that BATS requires a `target` field or that AUT requires a
`uses` list, but it cannot specify hidden answer values.

## 7. Skill Operator Library

**Definition 6: Skill operator.**

A skill operator is a reusable inference-time transformation:

```text
s_i: state -> state'
```

where `state` contains the visible task input, task profile, prior artifacts,
candidate pool, and audit trace. A skill may generate candidates, abstract
relations, evaluate feasibility, or normalize outputs, but it must not read
hidden scoring fields.

The current operator library is grouped by creativity level.

Combinational operators:

| Operator | Role |
|---|---|
| `unit_extraction` | Extract recombinable units, relations, properties, contexts, and output fields. |
| `relation_property_abstraction` | Abstract the relation, property, role, or semantic axis connecting units. |
| `candidate_recombination` | Generate recombinations by transferring relations, bridging units, or mapping properties. |
| `constraint_preservation` | Preserve relation direction, entity type, grammar, feasibility, and schema. |
| `combination_verification` | Verify whether a candidate meaningfully connects all required units. |
| `diversity_filtering` | Remove near-duplicates and preserve distinct semantic domains. |
| `output_normalization` | Convert selected artifacts into the benchmark answer format. |

Exploratory operators:

| Operator | Role |
|---|---|
| `constraint_space_mapping` | Separate hard constraints, soft preferences, forbidden outputs, and creative degrees of freedom. |
| `exploration_axis_expansion` | Expand distinct strategy axes or search directions. |
| `candidate_generation` | Produce multiple candidates across axes. |
| `semantic_deduplication` | Remove duplicate or near-duplicate ideas. |
| `coverage_balancing` | Preserve category, mechanism, or solution-structure coverage. |
| `novelty_transformation` | Increase novelty without breaking feasibility. |
| `feasibility_evaluation` | Reject irrelevant, infeasible, incoherent, unsafe, or constraint-breaking candidates. |
| `coherence_check` | Check causal flow, consistency, completeness, and contradictions. |
| `execution_verification` | Use symbolic or executable checks when visible examples or executable formats are available. |
| `portfolio_selection` | Select a final candidate under correctness, novelty, flexibility, clarity, and format constraints. |

Transformational operators:

| Operator | Role |
|---|---|
| `rule_change_extraction` | Extract changed rules and goals. |
| `legacy_assumption_mapping` | Identify assumptions, mechanisms, interfaces, measurements, and language inherited from the old system. |
| `breakage_propagation` | Trace how changed rules break legacy modules, interfaces, and coordination routines. |
| `primitive_induction` | Introduce new variables, roles, primitives, interfaces, or validity conditions. |
| `system_reconstruction` | Rebuild the core architecture around the new primitives. |
| `performance_reanchoring` | Restore performance, reliability, safety, or interpretability under the new rules. |
| `norm_interface_establishment` | Define new standards, records, training language, validation, and coordination norms. |
| `residue_audit` | Remove hidden dependence on invalid old-world assumptions. |
| `goal_coverage_verification` | Verify that stated goals are covered by concrete mechanisms. |

## 8. Workflow Selection

**Definition 7: Workflow.**

A workflow is an ordered sequence of skill operators:

```text
W_T = (s_1, s_2, ..., s_n)
```

selected from the operator library according to:

```text
W_T = select(l_T, c_T, s_T)
```

where `l_T` is the creativity level, `c_T` is the canonical objective vector,
and `s_T` is the output schema.

Examples:

```text
DAT:
unit_extraction -> candidate_recombination -> diversity_filtering
-> constraint_preservation -> output_normalization

BATS/RAT/Metaphor:
unit_extraction -> relation_property_abstraction -> candidate_recombination
-> combination_verification -> constraint_preservation -> output_normalization

AUT:
constraint_space_mapping -> candidate_generation -> coverage_balancing
-> novelty_transformation -> semantic_deduplication -> feasibility_evaluation
-> output_normalization

Transformation:
rule_change_extraction -> legacy_assumption_mapping -> breakage_propagation
-> primitive_induction -> system_reconstruction -> performance_reanchoring
-> norm_interface_establishment -> residue_audit -> goal_coverage_verification
-> output_normalization
```

This selection is task-conditioned but not instance-answer-conditioned. It
depends on public task structure and metric semantics, not on hidden labels.

## 9. Elicitation Method

**Definition 8: Elicitation method.**

An elicitation method is a test-time procedure:

```text
E(M, v(x), P_T) -> y in Y_T
```

that queries model `M` one or more times and returns a final answer `y` in the
benchmark-required output space `Y_T`.

Direct prompting is a baseline elicitation method:

```text
E_direct(M, v(x), P_T) = M(v(x))
```

TriSkill is a structured elicitation method:

```text
E_TriSkill(M, v(x), P_T) =
  normalize(select(execute(W_T, M, v(x), P_T)))
```

where:

- `execute` runs the selected workflow and records artifacts;
- `select` chooses a final candidate using verifier, consensus, portfolio, or
  modality-specific selection logic;
- `normalize` converts the selected candidate into the exact benchmark schema.

TriSkill is allowed to use multiple model calls at test time. It does not update
model weights and does not use training data or hidden references from the
evaluated instance.

## 10. Runtime State and Artifacts

At runtime, TriSkill maintains a state object:

```text
z = (v(x), P_T, artifacts, candidates, selected_candidate, skill_trace)
```

where:

- `artifacts` are intermediate JSON-like results produced by skills;
- `candidates` are possible answers or answer components;
- `selected_candidate` is the current best candidate;
- `skill_trace` records the sequence of skills and warnings.

The final answer submitted to the benchmark is:

```text
y = normalize(selected_candidate, s_T)
```

Intermediate artifacts are not submitted as the final benchmark response unless
the output schema explicitly requires them.

## 11. Creativity Profile and Profile Shift

**Definition 9: Creativity profile.**

For a model `M`, elicitation method `E`, and benchmark task set `B`, the
measured creativity profile is the vector:

```text
R(M, E, B) = (R(M, E, T_1), ..., R(M, E, T_n))
```

where each `R(M, E, T_i)` is the benchmark score or metric vector for task
`T_i`.

Because tasks have heterogeneous metrics, this vector should not be collapsed
into a single universal creativity score unless a justified aggregation is
specified. The paper's primary object is the profile itself.

**Definition 10: Creativity profile shift.**

Given direct prompting and a test-time elicitation method `E`, the measured
profile shift is:

```text
Delta(M, E, T) = R(M, E, T) - R(M, E_direct, T)
```

For a task set:

```text
Delta(M, E, B) =
  (Delta(M, E, T_1), ..., Delta(M, E, T_n))
```

Positive, zero, and negative shifts can occur on different tasks or metrics.
This is expected: creativity elicitation may improve diversity or novelty while
reducing exact-match validity, fluency, or executable correctness.

## 12. Result Interpretation Rules

The formalism implies the following result rules:

1. Report results as task-level or metric-level profile shifts, not as a single
   unqualified creativity score.
2. Separate exact/deterministic metrics from LLM-judge metrics.
3. Separate full-run evidence from diagnostic limit runs.
4. Label replay results as replay unless rerun through the standard evaluator.
5. Treat negative shifts as part of the profile, not as data to hide.
6. Do not compare scores across incompatible evaluator versions, sample counts,
   or judge configurations.

## 13. Method Component Classification

Every TriSkill component should be classified before it is described in the
paper:

| Class | Definition | Acceptable? | Examples |
|---|---|---|---|
| Generic operator | Derived from creativity mechanism and reusable across tasks. | Yes | `unit_extraction`, `candidate_recombination`, `primitive_induction` |
| Schema adapter | Uses visible task interface or output format. | Yes | BATS `target` field, AUT `uses` list |
| Metric-conditioned selector | Uses public metric semantics but no hidden labels. | Yes, with disclosure | DAT semantic dispersion, BATS relation consensus |
| Modality adapter | Handles output modality such as code, story, math, or word list. | Yes, with disclosure | NeoCoder visible-constraint parser, AUT portfolio renderer |
| Task hack | Uses hidden answers, test statistics, fixed answer maps, or sample-specific rules. | No | Country-capital answer map, hidden candidate answer lookup |

The paper should explicitly state that TriSkill contains schema and modality
adapters. The defensible novelty is not full task agnosticism; it is the
combination of shared creativity operators, public metric conditioning, and a
leakage-free information boundary.

## 14. Draft-Ready Paragraphs

### Problem Setup

We consider a creativity benchmark as a set of tasks whose instances contain a
visible prompt, public auxiliary fields, an output schema, and hidden
evaluator-only scoring information. A test-time elicitation method receives only
the visible view of each instance and returns a final answer in the required
schema. It must not access hidden answers, reference outputs, scoring-only
candidate lists, test-set statistics, or post-hoc scores.

### TriSkill Overview

TriSkill profiles each task by its creativity level, output schema, and public
metric semantics. It maps raw task metrics to canonical creativity objectives,
selects a workflow from a reusable operator library, executes that workflow
using the fixed model at test time, and normalizes the selected candidate into
the benchmark-required final format. The method is metric-conditioned because
public metric semantics influence the selected operators; it is leakage-free
because hidden labels and references are excluded from both prompts and
selectors.

### Profile Shift

Rather than reducing creativity to a single scalar score, we evaluate the
profile shift induced by an elicitation method. For each model and task, we
compare the score vector under TriSkill with the score vector under direct
prompting. This view allows gains, losses, and trade-offs to coexist, making it
possible to analyze when test-time creativity workflows improve diversity,
relation validity, or reconstruction quality, and when they harm fluency,
exact-match validity, or executable correctness.

## 15. Open Formalism Tasks

Before final manuscript freeze:

- Add citations for the three-level creativity taxonomy.
- Decide whether to introduce a normalized score only for visualization, not as
  the main metric.
- Add an appendix table with the full raw-to-canonical metric mapping.
- Add a precise statement of LLM-judge metrics and evaluator versions.
- Add notation for multi-metric tasks where `R(M, E, T)` is a vector rather than
  a scalar.
