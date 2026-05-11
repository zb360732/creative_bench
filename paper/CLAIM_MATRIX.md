# Claim Matrix

This file maps paper claims to evidence and risk controls. A claim should not
enter the manuscript unless it appears here with an evidence link and a clear
status.

## Claim Status

- `ready-main`: can be used in the main paper with current evidence.
- `ready-limited`: can be used, but only with explicit scope or caveat.
- `diagnostic-only`: can be used for analysis, motivation, or limitations, not
  as a main result.
- `needs-rerun`: plausible claim, but requires a cleaner or fuller experiment.
- `blocked`: should not be used until the underlying evidence changes.

## Evidence Abbreviations

- `EL:comb_full`: `paper/EVIDENCE_LEDGER.md`, Main Exact-Metric
  Combinational Evidence.
- `EL:qwen_full`: `paper/EVIDENCE_LEDGER.md`, qwen3.5-9b Direct Baseline and
  Current Full TriSkill Evidence.
- `EL:dat_bats_latest`: `paper/EVIDENCE_LEDGER.md`, Latest qwen3.5-9b DAT/BATS
  Diagnostics.
- `EL:explore_full`: `paper/EVIDENCE_LEDGER.md`, qwen3.5-9b Exploratory Full
  Evidence.
- `EL:explore_l20`: `paper/EVIDENCE_LEDGER.md`, qwen3.5-9b Exploratory
  Limit-20 Accepted Diagnostics.
- `EL:trans_l5`: `paper/EVIDENCE_LEDGER.md`, Transformation Evidence.
- `WF`: `paper/PAPER_WORKFLOW.md`.
- `PM`: `enhance/PAPER_MATERIALS.md`.
- `CODE`: implementation under `enhance/triskill/`.

## A. Core Paper Positioning Claims

| ID | Claim | Status | Evidence | Manuscript Placement | Required Wording / Caveat |
|---|---|---|---|---|---|
| C-A1 | Creativity evaluation should be treated as a multi-dimensional profile rather than a single scalar score. | `ready-main` | `PM`, `WF`, task taxonomy in `CODE`, direct metrics in `merged_evalscope_direct` | Abstract, Introduction, Benchmark/Taxonomy | Frame as a paper lens and benchmark design principle, not as a solved theory of creativity. |
| C-A2 | The benchmark organizes tasks around combinational, exploratory, and transformational creativity mechanisms. | `ready-main` after related-work citation check | `PM`, `CODE:profiles.py`, `paper_context/00_project_understanding.md` | Introduction, Taxonomy | Needs primary citations for the three creativity categories before final submission. |
| C-A3 | We study creativity elicitation, not only creativity measurement: the same fixed model can show different measured creativity profiles under different test-time workflows. | `ready-limited` | `EL:comb_full`, `EL:qwen_full`, `EL:explore_l20` | Abstract, Introduction, Results | Say "can show" and "profile shifts"; do not say universal improvement. |
| C-A4 | The safest paper scope is a benchmark + elicitation framework, not a benchmark-only paper and not a universal prompting algorithm paper. | `ready-main` as internal writing stance | `WF`, `PM`, `EL` | Introduction framing, Discussion | This is a framing decision. It does not need to be stated exactly this way in the paper. |

## B. Method Claims

| ID | Claim | Status | Evidence | Manuscript Placement | Risk Control |
|---|---|---|---|---|---|
| C-B1 | TriSkill is a definition-guided, metric-conditioned, leakage-free test-time elicitation framework. | `ready-main` | `CODE:profiles.py`, `CODE:skills.py`, `CODE:core.py`, `CODE:state.py`, `EL` | Abstract, Method | Must define "metric-conditioned" as public metric semantics, not label access. |
| C-B2 | TriSkill maps raw heterogeneous benchmark metrics into canonical creativity objectives. | `ready-main` | `CODE:profiles.py` raw-to-canonical mappings; `PM` operator vocabulary | Method | Add a table mapping raw metrics to canonical objectives. |
| C-B3 | TriSkill composes reusable operators rather than sample-specific answer rules. | `ready-main` | `CODE:skills.py`, `CODE:task_prompts.py`, `enhance/WORKLOG.md` removal of task-specific adapters | Method, Leakage Protocol | Must acknowledge lightweight schema adapters. Do not claim no task-specific guidance. |
| C-B4 | The leakage guard excludes hidden scoring fields such as `answer`, `target_words`, `candidate_answers`, `reference`, and `gold`. | `ready-main` | `CODE:state.py`, hidden-gold tests in `enhance/tests/test_triskill.py` | Method, Evaluation Protocol | Mention this is an implementation and protocol guard; also ensure experiment artifacts comply. |
| C-B5 | The shared runtime template constrains every skill call to visible prompt, filtered visible fields, prior artifacts, skill instruction, and compact JSON output. | `ready-main` | `CODE:runtime_skills.py`, `CODE:core.py` | Method | Useful to distinguish framework from ad hoc prompt writing. |
| C-B6 | Task profiles are schema/metric adapters over the shared operator library. | `ready-main` | `CODE:profiles.py`, `CODE:task_prompts.py`, `PM` | Method, Limitations | Essential caveat: not fully task-agnostic. |
| C-B7 | DAT semantic dispersion selection is a generic metric-conditioned selector over a visible candidate pool. | `ready-limited` | `CODE:runtime_skills.py`, `EL:dat_bats_latest` | Method or Appendix | Must disclose embedding/semantic-distance alignment and fallback word-list caveat. |
| C-B8 | BATS relation consensus is a generic analogy selector over visible workflow artifacts, not a hidden-answer lookup. | `ready-limited` | `CODE:runtime_skills.py`, hidden-gold trap test, `EL:dat_bats_latest` | Method, Analysis | Main result needs standard full rerun; current full evidence is replay. |
| C-B9 | AUT portfolio rendering is an exploratory/list-output modality adapter that recovers feasible diverse uses from sparse or contaminated artifacts. | `ready-limited` | `CODE:runtime_skills.py`, `EL:explore_l20`, `enhance/WORKLOG.md` | Method, Analysis | Needs full rerun before main performance claim. |
| C-B10 | NeoCoder constraint parsing is a code-modality adapter based only on visible prompt constraints and visible sample I/O. | `ready-limited` | `CODE:runtime_skills.py`, tests, `EL:explore_l20` | Method, Analysis | Frame as visible-constraint preservation, not hidden-test optimization. |
| C-B11 | Transformation operators implement rule-change extraction, legacy-assumption removal, primitive induction, and system reconstruction. | `ready-main` as method claim | `CODE:skills.py`, `CODE:task_prompts.py`, `PM`, `paper_context/乱七八糟的素材.md` | Method, Taxonomy | Empirical strength is diagnostic only; method description is still valid. |

## C. Main Empirical Claims

| ID | Claim | Status | Evidence | Manuscript Placement | Required Caveat |
|---|---|---|---|---|---|
| C-C1 | On full exact-metric combinational tasks, TriSkill shifts model creativity profiles across DAT, BATS, RAT, and Metaphor. | `ready-main` | `EL:comb_full` | Results main table | Use "profile shift"; not "uniform improvement." |
| C-C2 | In full combinational evaluation, DeepSeek-R1-Distill-Qwen-1.5B improves on all four combinational tasks under TriSkill. | `ready-main` | `EL:comb_full` | Results | DAT n=1 caveat. |
| C-C3 | In full combinational evaluation, DeepSeek-R1-Distill-Qwen-32B improves on all four combinational tasks under TriSkill. | `ready-main` | `EL:comb_full` | Results | Gains on BATS are very small; report exact numbers. |
| C-C4 | DeepSeek-R1-Distill-Qwen-7B improves on BATS, RAT, and Metaphor but regresses on DAT. | `ready-main` | `EL:comb_full` | Results, Analysis | Use as evidence of model/task-dependent trade-offs. |
| C-C5 | DeepSeek-R1-Distill-Qwen-14B is mostly flat under TriSkill in full combinational evaluation. | `ready-main` | `EL:comb_full` | Results, Analysis | Do not overinterpret tiny BATS gain or Metaphor drop. |
| C-C6 | qwen3.5-9b current full TriSkill improves RAT, Metaphor, AUT, CreativeMath, CS4, and NeoCoder over the merged direct baseline. | `ready-limited` | `EL:qwen_full` | Secondary Results or Appendix | Requires final verification of baseline compatibility; CS4 sample-count mismatch must be noted. |
| C-C7 | qwen3.5-9b latest DAT selector improves DAT over merged direct baseline. | `diagnostic-only` | `EL:dat_bats_latest` | Analysis/Ablation | n=1; do not use as strong main claim. |
| C-C8 | qwen3.5-9b BATS relation-consensus selector improves BATS over merged direct baseline on full 4000-sample replay. | `diagnostic-only` / `needs-rerun` | `EL:dat_bats_latest` | Analysis/Ablation | Label as replay until standard full evaluator rerun. |
| C-C9 | Current qwen3.5-9b exploratory limit-20 diagnostics show positive shifts on AUT, CreativeMath, and NeoCoder, with CS4 at ceiling. | `diagnostic-only` | `EL:explore_l20` | Analysis/Ablation | Limit20 is not final full evidence. |
| C-C10 | Early qwen3.5-9b exploratory full run improved CreativeMath and slightly CS4 but hurt AUT and NeoCoder. | `diagnostic-only` | `EL:explore_full` | Failure Analysis | Mark as superseded by later modules; use for boundary discussion only. |
| C-C11 | Transformation limit5 diagnostics improve for 7B and 32B, remain flat for 1.5B, and regress for 14B. | `diagnostic-only` | `EL:trans_l5` | Analysis or Appendix | Low-sample, judge-dependent, not final. |

## D. Boundary and Negative Claims

| ID | Claim | Status | Evidence | Manuscript Placement | Why It Matters |
|---|---|---|---|---|---|
| C-D1 | TriSkill is not a universally improving method; gains are model-, task-, and metric-dependent. | `ready-main` | `EL:comb_full`, `EL:explore_full`, `EL:trans_l5` | Discussion, Limitations | This protects credibility and explains mixed results. |
| C-D2 | Convergent exact-match tasks under-credit intermediate combinational quality, so gains on BATS/RAT/Metaphor are expected to be smaller. | `ready-limited` | `PM`, `EL:comb_full` | Analysis | Needs careful phrasing as interpretation, not proven mechanism. |
| C-D3 | Workflow exploration can improve novelty or coverage while harming fluency, correctness, or executable validity. | `ready-limited` | `EL:explore_full`, rejected iterations in `EL` and `enhance/WORKLOG.md` | Analysis, Limitations | This is central for profile-shift framing. |
| C-D4 | Overly strict safety or constraint prompts can over-steer generation and reduce task quality. | `diagnostic-only` | `EL` rejected AUT/NeoCoder iterations, `enhance/WORKLOG.md` | Failure Modes or Appendix | Useful but should not occupy main contribution space. |
| C-D5 | Latest evidence gaps require more full runs before claiming broad closed-source model validation. | `ready-main` | `EL:Current Evidence Gaps` | Limitations, Experiment Coverage | Prevents overclaiming. |
| C-D6 | Transformation current results are not comparable to the merged direct baseline because the evaluator setup changed. | `ready-main` | `EL:qwen_full`, user note, `EL:Current Evidence Gaps` | Limitations / Results footnote | Must be explicit if any transformation number is shown. |

## E. Claims That Are Not Allowed Yet

| Forbidden Claim | Status | Why Blocked | What Would Be Needed |
|---|---|---|---|
| TriSkill improves all tasks on all models. | `blocked` | Contradicted by full combinational 7B/14B and exploratory full diagnostics. | Broad full runs showing consistent improvement, or revised scoped claim. |
| TriSkill is fully task-agnostic. | `blocked` | Method uses task profiles and schema adapters. | Do not pursue; correct claim is shared operators plus lightweight adapters. |
| TriSkill does not use metric information. | `blocked` | Metric conditioning is part of the method. | Correct claim: uses public metric semantics, not hidden labels. |
| Latest qwen3.5-9b BATS full gain is final standard-evaluator evidence. | `blocked` | Current full gain is replay-only. | Rerun full BATS with latest code through standard evaluator. |
| qwen3.5-9b Transformation improves over direct baseline. | `blocked` | Baseline and TriSkill counts/evaluator setup are not comparable. | Stable comparable direct and TriSkill transformation full or matched-limit run. |
| Closed-source models validate TriSkill. | `blocked` | Direct baselines exist partially; TriSkill runs have not been run across closed-source models. | Run TriSkill on selected closed-source models or restrict claim to direct benchmark coverage. |

## F. Main-Text Claim Plan

Recommended main-text claims for the first complete draft:

1. **Motivation claim**: Creativity should be evaluated as a profile across
   combinational, exploratory, and transformational mechanisms.
2. **Framework claim**: TriSkill operationalizes this view as a leakage-free,
   definition-guided, metric-conditioned test-time elicitation workflow.
3. **Method claim**: TriSkill composes reusable operators selected by canonical
   metric objectives and visible task schema.
4. **Primary empirical claim**: Full exact-metric combinational evaluation shows
   model-dependent profile shifts, including all-four-task gains for 1.5B and
   32B.
5. **Secondary empirical claim**: qwen3.5-9b evidence suggests broader gains
   across several tasks, but the latest method requires clean full reruns for
   main-table claims.
6. **Boundary claim**: Elicitation is not uniformly beneficial; workflow search
   can trade off novelty, fluency, exact-match validity, and executable
   correctness.

## G. Next Evidence Actions

Highest-value experiments before final paper tables:

1. Rerun qwen3.5-9b full BATS with the latest relation-consensus selector through
   the standard evaluator.
2. Rerun qwen3.5-9b full DAT/BATS/RAT/Metaphor with the latest combinational
   code, if time permits.
3. Rerun qwen3.5-9b full exploratory tasks with the latest AUT and NeoCoder
   modules to replace old superseded full evidence.
4. Produce matched direct/TriSkill transformation results under the same stable
   evaluator.
5. Decide whether any closed-source model will run TriSkill; if not, frame
   closed-source models as benchmark direct baselines only.

## H. Writing Guardrails

Use these phrases:

- "profile shift"
- "model-dependent gains"
- "metric-conditioned without hidden labels"
- "shared operator library with lightweight schema adapters"
- "diagnostic limit-20 evidence"
- "replay-only full BATS analysis"

Avoid these phrases unless evidence changes:

- "universally improves"
- "fully task-agnostic"
- "validated on all models"
- "solves creativity"
- "guaranteed leakage-free" without specifying the implemented field filter and
  protocol boundary
