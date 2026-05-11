# Paper Workflow and Quality Standard

This document is the operating protocol for writing the benchmark + TriSkill
paper. It defines the standard we will use before drafting, while drafting, and
before submission. The goal is not to write a plausible paper quickly; the goal
is to build a defensible A-conference paper whose claims are supported by
evidence, whose method is clearly novel, and whose limitations are explicit.

## 1. Target Standard

The paper should be written as a rigorous NLP/AI evaluation-and-method paper.
Every major claim must pass four tests:

1. **Conceptual clarity**: the paper must define what is being measured or
   elicited, why existing work does not already answer it, and where our task
   taxonomy fits in creativity theory.
2. **Methodological defensibility**: TriSkill must be described as a
   definition-guided, metric-conditioned, leakage-free inference framework, not
   as task-specific prompt engineering.
3. **Empirical traceability**: every number in the paper must be recoverable
   from an output path, config, commit, model list, and evaluation command.
4. **Claim discipline**: the paper may claim profile shifts and verified gains
   where supported; it must not claim universal improvement, solved creativity,
   or fully task-agnostic prompting.

The expected reader should understand the paper as:

```text
Creativity benchmarks should evaluate multiple creativity mechanisms.
TriSkill uses creativity definitions and metric objectives to elicit different
creative behaviors at test time without training or hidden-answer access.
The observed model profile shifts show both opportunities and boundaries of
test-time creativity elicitation.
```

## 2. Core Thesis

Working thesis:

```text
Creativity in LLM evaluation should not be treated as a single scalar ability.
We organize benchmark tasks around combinational, exploratory, and
transformational creativity, then introduce TriSkill, a leakage-free test-time
elicitation framework that maps visible task structure and metric semantics into
reusable skill workflows. TriSkill changes measured creativity profiles, with
strongest evidence on combinational and selected exploratory/transformational
settings, while also exposing clear trade-offs between novelty, validity,
fluency, and executable correctness.
```

This thesis is intentionally stronger than "we prompt better" and weaker than
"we improve every task." It leaves room for negative results and still makes a
clear contribution.

## 3. Non-Negotiable Claim Rules

Do not write any claim that violates these rules.

- Do not claim TriSkill uses no task information. Correct phrasing:
  `TriSkill uses visible task schema, task type, output format, visible
  constraints, and canonical metric objectives`.
- Do not claim task-specific prompts are absent. Correct phrasing:
  `Tasks instantiate lightweight schema adapters over a shared operator
  library`.
- Do not claim no metric conditioning. Correct phrasing:
  `Metric semantics select and prioritize skills; hidden labels and test
  statistics are never used`.
- Do not claim full dominance unless a complete run supports it for the named
  model and task group.
- Do not overclaim DAT because the full DAT split may contain one effective
  item; report it as a semantic-diversity profile indicator.
- Do not present `limit=5` or `limit=20` judge runs as final definitive
  evidence; label them as diagnostic or focused validation unless full runs are
  available.
- Do not hide negative results. A-conference papers are stronger when they
  explain boundaries rather than only select wins.

## 4. Paper Workstream

The paper will be built in eight stages. A later stage should not start until
the previous stage has produced its required artifacts.

### Stage 0: Evidence Inventory

Purpose: know exactly what data, code, and claims we already have.

Inputs:

- `enhance/PAPER_MATERIALS.md`
- `enhance/WORKLOG.md`
- `paper_context/00_project_understanding.md`
- `paper_context/乱七八糟的素材.md`
- `outputs/**/summary.json`
- merged historical outputs under the user-provided `huizong` directory

Required artifacts:

- `paper/EVIDENCE_LEDGER.md`
- one row per experiment result used in the paper
- each row must include task group, model, method, limit/full, config, path,
  commit if known, metric names, and claim status

Stop condition:

- Every planned table number can be traced to a path.
- Every incomplete/missing model is explicitly labeled.

### Stage 1: Literature and Positioning Review

Purpose: establish novelty against creativity evaluation, creativity theory,
LLM prompting, test-time inference, and benchmark leakage concerns.

Required literature buckets:

- Creativity theory: combinational, exploratory, transformational creativity;
  divergent thinking; analogical/remote association; creative problem solving.
- LLM creativity evaluation: existing creativity benchmarks and evaluation
  protocols.
- Prompting and test-time inference: self-consistency, tree/search-style
  reasoning, self-refine, verifier/reranker methods, multi-agent or workflow
  prompting where relevant.
- Evaluation methodology: metric validity, LLM-as-judge risks, contamination
  and leakage controls.
- Related benchmarks/tasks: DAT, AUT, RAT, analogy, metaphor, story generation,
  math creativity, code creativity, transformation/system-reconstruction tasks.

Required artifacts:

- `paper/RELATED_WORK_NOTES.md`
- `paper/REFERENCES.bib`
- `paper/CLAIM_MATRIX.md`

Quality bar:

- For each related-work bucket, identify what prior work does, what it does not
  do, and exactly where our benchmark or TriSkill differs.
- Do not cite papers only for decoration. Every citation must support a specific
  contrast or design choice.
- Mark uncertain references as `NEEDS_VERIFICATION` until checked from primary
  sources.

Stop condition:

- We can state the paper's novelty in one paragraph without relying on vague
  words like "comprehensive" or "effective."

### Stage 2: Formal Problem Definition

Purpose: make the benchmark and elicitation setup precise.

Definitions to write:

- Creativity task instance: visible input, hidden scoring fields, output schema,
  metric set.
- Creativity level: combinational, exploratory, transformational.
- Canonical metric objective: relation validity, semantic diversity, novelty,
  flexibility, appropriateness, execution validity, rule utilization, system
  reconstruction, etc.
- Elicitation method: a test-time procedure that transforms visible input into
  final answer without training and without hidden labels.
- Leakage-free protocol: allowed and disallowed information.
- Creativity profile shift: vector-valued metric change under fixed model and
  benchmark.

Required artifacts:

- `paper/FORMALISM.md`
- final notation table for the paper

Stop condition:

- A reviewer can tell the difference between benchmark construction, direct
  prompting, and TriSkill elicitation.

### Stage 3: Method Specification

Purpose: describe TriSkill as a method, not as a list of prompts.

Method sections:

- Task Profiler
- Metric Abstraction Layer
- Workflow Router
- Skill Operator Library
- Runtime Executor
- Verifier / Selector
- Output Normalizer
- Leakage Guard

Required figures:

- Architecture diagram: input task -> profiler -> canonical objectives ->
  workflow router -> skill executor -> final answer -> benchmark evaluator.
- Operator library diagram: combinational / exploratory / transformational
  workflows and their shared operators.
- Leakage boundary diagram: visible inputs inside boundary, hidden labels and
  scoring-only fields outside boundary.

Required artifacts:

- `paper/METHOD_SPEC.md`
- `paper/FIGURE_PLAN.md`

Quality bar:

- Every task-specific adapter must be described as schema/metric conditioning,
  not answer tuning.
- DAT embedding selection, BATS relation consensus, AUT portfolio rendering,
  and NeoCoder constraint parsing must be explicitly framed as generic
  selectors or modality adapters.

Stop condition:

- The method could be reimplemented from the paper without reading code.

### Stage 4: Experimental Protocol

Purpose: make results credible and reproducible.

Protocol fields required for every experiment:

- model name and config path
- method: direct or TriSkill variant
- task list
- sample limit or full
- parallelism
- request parameters, especially `enable_thinking=false` when using Qwen-style
  thinking models
- judge config and URL if LLM-judge metrics are used
- output directory
- code commit
- whether the result is final, diagnostic, rejected, or replay-only

Required tables:

- Task taxonomy and metrics.
- Main model/task results.
- Per-task deltas.
- Ablation table for TriSkill components where available.
- Cost/runtime table if feasible.

Required artifacts:

- `paper/EXPERIMENT_PROTOCOL.md`
- `paper/TABLE_PLAN.md`
- scripts or notebooks that generate paper tables from output summaries

Quality bar:

- No hand-copied number should enter the manuscript unless it is also in the
  evidence ledger.
- Replay-only results must be labeled as replay-only unless rerun through the
  standard evaluator.
- LLM-judge results must be separated from exact or deterministic metrics.

Stop condition:

- A reader could rerun or audit every reported number.

### Stage 5: Result Interpretation

Purpose: turn numbers into scientific claims.

Required analyses:

- Which creativity levels improve, for which models, and under which metrics?
- Which tasks show trade-offs rather than gains?
- Why do convergent exact-match combinational tasks show smaller gains than
  open-ended divergent tasks?
- What does TriSkill reveal about creativity as a profile rather than a scalar?
- Where does workflow exploration harm fidelity, fluency, or executable
  correctness?

Required artifacts:

- `paper/RESULTS_ANALYSIS.md`
- `paper/FAILURE_MODES.md`

Quality bar:

- Interpret every negative result.
- Do not cherry-pick models without explaining model coverage.
- Separate empirical findings from speculation.

Stop condition:

- The results section can answer a skeptical reviewer's question: "Why should I
  believe this is a method and not prompt hacking?"

### Stage 6: Manuscript Draft

Purpose: write the paper in conference style.

Recommended section order:

1. Abstract
2. Introduction
3. Benchmark and Creativity Taxonomy
4. TriSkill Method
5. Leakage-Free Evaluation Protocol
6. Experiments
7. Results
8. Analysis and Ablations
9. Limitations
10. Related Work
11. Conclusion

Drafting rule:

- Start with section skeletons and bullet claims.
- Then write paragraphs.
- Then add figures and tables.
- Then tighten language.

Required artifacts:

- LaTeX source under `paper/`
- `paper/main.tex`
- `paper/sections/*.tex`
- `paper/figures/`
- `paper/tables/`
- `paper/REFERENCES.bib`

Quality bar:

- Every paragraph should serve one of four purposes: define, motivate, prove,
  or interpret.
- Avoid vague adjectives unless supported by data.
- Keep limitations concrete, not apologetic.

Stop condition:

- First full compile succeeds and all placeholders are tracked.

### Stage 7: Internal Review and Revision

Purpose: stress-test the paper before submission.

Review passes:

- Novelty pass: what is new compared with prior benchmarks and prompting
  methods?
- Leakage pass: can any prompt, selector, or normalizer be accused of using
  hidden answers?
- Evidence pass: do all results match ledger paths?
- Negative-result pass: are limitations and trade-offs honestly explained?
- Figure/table pass: are visuals self-contained and publication-quality?
- Reproducibility pass: can someone rerun the reported pipeline?
- Writing pass: is the abstract/introduction crisp enough for a reviewer to
  understand in five minutes?

Required artifacts:

- `paper/REVIEW_CHECKLIST.md`
- `paper/REBUTTAL_RISKS.md`

Stop condition:

- No major claim remains unsupported.

## 5. Figure and Table Standard

Every figure/table must have a point. It should answer one review question.

Minimum planned figures:

- **Figure 1: Benchmark + TriSkill overview**. Shows task taxonomy, metric
  abstraction, workflow execution, and benchmark measurement.
- **Figure 2: Operator library**. Shows combinational, exploratory, and
  transformational workflows as reusable operator sequences.
- **Figure 3: Leakage-free boundary**. Shows allowed visible information and
  forbidden scoring-only information.
- **Figure 4: Creativity profile shift**. Radar/bar view of direct vs TriSkill
  for a representative model, likely qwen3.5-9b when evidence is final.

Minimum planned tables:

- **Table 1: Task taxonomy**. Tasks, creativity level, cognitive operation,
  output schema, metrics, scoring type.
- **Table 2: Canonical metric mapping**. Raw benchmark metrics to canonical
  objectives.
- **Table 3: Main results**. Direct vs TriSkill by model/task, separated by
  exact/deterministic and LLM-judge metrics.
- **Table 4: Ablations**. Direct seed, skill workflow, selector, normalizer,
  modality gates where available.
- **Table 5: Failure modes**. Representative failures and corresponding
  workflow boundaries.

Visual quality rules:

- No screenshot tables.
- No manually edited numbers in figures.
- Every plotted value must come from a script or ledger entry.
- Figures should be grayscale-readable and colorblind-safe.
- Captions must state the takeaway, not just describe axes.

## 6. Experiment Evidence Standard

Each result has one of four statuses:

- `final`: full or accepted run suitable for main table.
- `diagnostic`: limit run or iteration used to guide design.
- `replay`: metric-compatible replay over existing prediction artifacts, useful
  for analysis but labeled if not produced by the standard evaluator.
- `rejected`: run that informed design but should not support positive claims.

A result can enter the main paper only if:

- its status is `final`, or it is explicitly labeled as diagnostic in an
  analysis subsection;
- its output path exists;
- its model and request parameters are known;
- it did not modify `evalscope` source code;
- it follows the leakage-free protocol;
- it is not superseded by a later accepted run.

## 7. Method Novelty Standard

The paper should not sell TriSkill as "just prompt engineering." The novelty
must be argued through structure:

- Creativity definition determines workflow family.
- Raw heterogeneous metrics are mapped to canonical objectives.
- Canonical objectives select reusable operators.
- Operators produce inspectable artifacts.
- Verifier/selector and normalizer convert artifacts into benchmark-compatible
  answers.
- Leakage guards define the information boundary.
- Evaluation measures profile shift rather than a single scalar win.

The most paper-worthy parts are:

- the three-level creativity taxonomy as an evaluation lens;
- the canonical metric abstraction layer;
- the reusable operator library;
- the leakage-free workflow contract;
- the empirical profile-shift analysis, including failures.

## 8. Overfitting and Leakage Audit Standard

Every method component must be classified before inclusion:

- **Generic operator**: derived from creativity definition and reusable across
  tasks.
- **Schema adapter**: depends on visible output format or task interface.
- **Metric-conditioned selector**: uses public metric semantics but no hidden
  labels.
- **Modality adapter**: depends on output modality such as code, story, math, or
  word list.
- **Unacceptable task hack**: uses hidden answers, test statistics, fixed answer
  maps, or sample-specific rules.

Current known audit points:

- DAT semantic dispersion selector is metric-conditioned, not a hidden oracle.
- DAT fallback word list is an engineering fallback and should not be
  emphasized as a core method.
- BATS relation consensus is a generic analogy selector; copy-input rejection
  should be described as lexical-validity handling.
- AUT portfolio rendering is an exploratory/list-output modality adapter.
- NeoCoder visible-constraint parsing is a code-modality adapter based only on
  visible prompt constraints.

No method component should be described in the paper until it has this
classification.

## 9. Writing Style Standard

Use precise, modest, defensible language.

Preferred phrasing:

- "TriSkill elicits measurable creativity profile shifts."
- "We observe gains on..."
- "The result suggests..."
- "This failure case indicates..."
- "The method uses visible schema and metric semantics, not hidden references."

Avoid:

- "solves creativity"
- "universally improves"
- "fully task-agnostic"
- "human-level creativity"
- "guarantees"
- "proves"
- "dramatically" unless the statistic justifies it

Every section should make one clear point:

- Introduction: why profile-based creativity elicitation matters.
- Taxonomy: why tasks differ by creativity mechanism.
- Method: how definitions and metrics become workflows.
- Experiments: how we test without leakage.
- Results: where profile shifts occur and where they do not.
- Analysis: why the pattern is scientifically meaningful.

## 10. Immediate Next Actions

1. Build `paper/EVIDENCE_LEDGER.md` from current outputs and historical merged
   summaries.
2. Build `paper/RELATED_WORK_NOTES.md` with verified primary-source citations.
3. Decide the main claim scope: benchmark paper, elicitation-method paper, or
   combined benchmark + elicitation paper.
4. Draft `paper/FORMALISM.md` so later writing does not drift into vague
   terminology.
5. Create the LaTeX skeleton only after the evidence ledger and method
   formalism are stable.

## 11. Current Claim Scope

Given current evidence, the safest near-term paper scope is:

```text
Benchmark + elicitation framework paper.
The benchmark defines and measures multiple creativity mechanisms.
TriSkill is an accompanying leakage-free elicitation framework that demonstrates
how test-time workflows can shift measured creativity profiles.
```

This is stronger than a benchmark-only paper and safer than claiming TriSkill is
a universally improving algorithm.
