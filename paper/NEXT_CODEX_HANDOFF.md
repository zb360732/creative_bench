# Next Codex Handoff

This document is for starting a fresh Codex session after context pressure. It
summarizes the current state, constraints, completed paper-planning artifacts,
and the next concrete task.

## Current Workspace

- Repository root: `/root/benchmark`
- Current branch: `enhance/combinational-workflow-optimization`
- Latest relevant commits:
  - `d10ccb7 Add paper formalism`
  - `e13ce07 Add paper claim matrix`
  - `cd8beb0 Create paper evidence ledger`
  - `40fe6a0 Add paper writing workflow protocol`
  - `328fd59 Audit genericity of BATS workflow`
- Remote branch has been pushed through `d10ccb7`.

Current dirty/untracked files that should not be casually modified or committed:

```text
M  evalscope/run/models2.json
?? evalscope/run/llm_judge2.json
?? paper/acl-style-files-master.zip
?? paper/acl-style-files-master/
?? paper_context/
```

The user has been editing/using some of these. Do not revert or delete them.

## Global Project Rules

- Do not modify `/root/benchmark/evalscope` source code. If evaluation wrapping
  is needed, implement it under `/root/benchmark/enhance`.
- Use `apply_patch` for manual file edits. Do not invoke `apply_patch` through
  shell commands.
- Keep the worktree clean and commit only files relevant to the current task.
- Track paper work in `paper/WORKLOG.md`.
- Be strict about leakage: no hidden answers, references, scoring-only
  candidate lists, target words, post-hoc scores, or test-set statistics may be
  used by TriSkill.
- The paper should target A-conference quality with careful claims, evidence,
  and limitations.

## User's Current Goal

The user wants to start writing the paper for a benchmark + creativity
elicitation framework. The paper should be rigorous, high-standard, and
defensible for an A-level conference.

The current paper story:

```text
Creativity should be evaluated as a profile across combinational, exploratory,
and transformational mechanisms. TriSkill is a leakage-free, definition-guided,
metric-conditioned test-time elicitation framework that maps visible task
structure and public metric semantics into reusable skill workflows. It can
shift measured creativity profiles, with strongest current evidence on
combinational tasks and qwen3.5-9b diagnostics/full candidates, while also
showing clear trade-offs and failure modes.
```

## Completed Paper Artifacts

### `paper/PAPER_WORKFLOW.md`

Defines the full paper workflow and quality standard. Important points:

- Claims must pass conceptual clarity, methodological defensibility, empirical
  traceability, and claim discipline.
- The paper should not claim universal improvement or full task agnosticism.
- Stages:
  1. Evidence inventory
  2. Literature and positioning review
  3. Formal problem definition
  4. Method specification
  5. Experimental protocol
  6. Result interpretation
  7. Manuscript draft
  8. Internal review

### `paper/EVIDENCE_LEDGER.md`

Records current evidence and classifies results as:

- `final`
- `final-candidate`
- `diagnostic`
- `replay`
- `superseded`
- `rejected`
- `not-comparable`

Key evidence:

- Full exact-metric combinational direct vs TriSkill for
  DeepSeek-R1-Distill-Qwen 1.5B/7B/14B/32B:
  - 1.5B and 32B improve on all four combinational tasks.
  - 7B improves BATS/RAT/Metaphor but regresses DAT.
  - 14B is mostly flat.
  - DAT has `n=1`, so treat carefully.
- qwen3.5-9b merged direct baseline vs current full TriSkill:
  - RAT, Metaphor, AUT, CreativeMath, CS4, NeoCoder show positive full-candidate
    evidence.
  - DAT/BATS current full are superseded by newer selector diagnostics.
  - Transformation is not comparable because the user changed transformation
    evaluation code/setup.
- Latest qwen3.5-9b DAT/BATS:
  - DAT latest diagnostic: `6.8994 -> 7.4730`, `n=1`.
  - BATS limit20 diagnostic: `0.5232 -> 0.9500`, `n=20`.
  - BATS full replay: `0.5232 -> 0.5300`, `n=4000`, replay-only.
- qwen3.5-9b exploratory limit20 diagnostics:
  - AUT `41.65 -> 44.65`
  - CreativeMath `0.80 -> 1.00`
  - CS4 `1.00 -> 1.00`
  - NeoCoder `0.00 -> 0.45`
- Transformation limit5 diagnostics:
  - 7B and 32B improve.
  - 1.5B no gain.
  - 14B regresses from high direct baseline.

### `paper/CLAIM_MATRIX.md`

Maps each candidate paper claim to evidence and risk controls.

Safe main claims now:

- Creativity evaluation should be profile-based, not a single scalar.
- TriSkill is definition-guided, metric-conditioned, and leakage-free.
- TriSkill uses shared operators plus lightweight schema/modality adapters.
- Full exact-metric combinational evaluation shows model-dependent profile
  shifts, including all-four-task gains for 1.5B and 32B.
- TriSkill is not universally improving; gains are model/task/metric dependent.

Blocked claims:

- TriSkill improves all tasks on all models.
- TriSkill is fully task-agnostic.
- TriSkill does not use metric information.
- Latest qwen3.5-9b BATS full gain is final standard-evaluator evidence.
- qwen3.5-9b Transformation improves over direct baseline.
- Closed-source models validate TriSkill.

### `paper/FORMALISM.md`

Defines:

- task instance `x = (p, a, s, q)`
- visible view `v(x)`
- hidden scoring information `h(x)`
- leakage-free protocol
- creativity levels
- canonical objective mapping `g: m_T -> c_T`
- task profile `P_T`
- skill operator library
- workflow selection `W_T = select(l_T, c_T, s_T)`
- direct prompting and TriSkill elicitation:

```text
E_direct(M, v(x), P_T) = M(v(x))

E_TriSkill(M, v(x), P_T) =
  normalize(select(execute(W_T, M, v(x), P_T)))
```

- creativity profile shift:

```text
Delta(M, E, T) = R(M, E, T) - R(M, E_direct, T)
```

## Current Task for New Session

Continue Stage 1: comprehensive related-work research.

The user explicitly said:

```text
开始调研吧，记得调研的一定要全面，各种方式都试一下
```

The previous session began searching but did not write research notes yet. Start
fresh and create:

- `paper/RELATED_WORK_NOTES.md`
- `paper/REFERENCES.bib`

Also update:

- `paper/WORKLOG.md`

## Research Buckets to Cover

Be comprehensive and prefer primary sources. Use web search because citations
and bibliographic details must be current and accurate.

### 1. Creativity theory

Must cover:

- Margaret Boden's combinational, exploratory, and transformational creativity.
- Divergent thinking.
- Remote association / associative theory of creativity.
- Creativity as novelty + appropriateness/usefulness.

Likely sources to verify:

- Boden, *The Creative Mind: Myths and Mechanisms*.
- Boden, "Creativity and Artificial Intelligence", Artificial Intelligence,
  1998.
- Guilford on creativity/divergent thinking.
- Mednick, "The associative basis of the creative process", Psychological
  Review, 1962.
- Runco / Torrance / Amabile if needed for novelty/usefulness framing.

### 2. Task/source literature

Must cover:

- DAT: Divergent Association Task.
- AUT: Alternative Uses Task.
- RAT: Remote Associates Test.
- BATS: Bigger Analogy Test Set.
- Metaphor/paraphrase or lexical substitution task background.
- Creative math evaluation if a primary benchmark/source can be found.
- Story generation constraints / creative story evaluation if relevant.
- Code creativity / constrained code generation if relevant.
- Transformation/system-reconstruction tasks: likely needs framing from
  transformational creativity and counterfactual/system reasoning rather than an
  established single benchmark.

Likely sources to verify:

- Olson et al., Divergent Association Task, PNAS 2021.
- Guilford / Christensen et al. for Alternative Uses.
- Mednick for RAT.
- Gladkova, Drozd, Matsuoka, BATS.

### 3. LLM creativity evaluation

Find and verify recent work on:

- LLMs on divergent thinking tasks.
- GPT/LLM performance on AUT/DAT/RAT/creative writing.
- Surveys or benchmarks of LLM creativity.
- Concerns that LLM creativity evaluation is task/metric dependent.

Likely search queries:

- "Putting GPT-3's Creativity to the Alternative Uses Test"
- "Divergent Association Task large language models creativity"
- "Assessing creativity of large language models"
- "LLM creativity benchmark novelty appropriateness fluency flexibility"

### 4. Test-time inference / prompting workflows

Must cover as related but distinct:

- Chain-of-thought and self-consistency.
- Tree of Thoughts / search over reasoning paths.
- Self-Refine / iterative feedback.
- Reflexion or verifier/reranker methods if relevant.
- Analogical prompting if relevant.

Likely sources:

- Wang et al., Self-Consistency Improves Chain of Thought Reasoning in Language
  Models.
- Yao et al., Tree of Thoughts.
- Madaan et al., Self-Refine.
- Shinn et al., Reflexion.

Positioning:

- These methods improve reasoning/problem solving generally.
- TriSkill differs by using creativity taxonomy + canonical creativity
  objectives + leakage-free benchmark protocol.

### 5. LLM-as-judge and evaluation reliability

Must cover:

- MT-Bench / Chatbot Arena / LLM-as-judge.
- Biases in LLM evaluators: position bias, verbosity bias, self-enhancement,
  model preference, inconsistency.
- Need to separate exact metrics from LLM-judge metrics.

Likely sources:

- Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.
- Papers on LLM evaluator bias/fairness.

### 6. Data contamination / benchmark leakage

Must cover:

- Benchmark contamination in LLM evaluation.
- Hidden label leakage and evaluation protocol concerns.
- Why TriSkill's information boundary matters.

Likely sources:

- Work on data contamination / test set leakage in LLM benchmark evaluation.
- Papers about benchmark memorization or contamination detection.

## How to Write `RELATED_WORK_NOTES.md`

Use this structure:

```markdown
# Related Work Notes

## Research Questions

## 1. Creativity Theory
### Key Takeaways
### Sources
### How This Positions Our Paper

## 2. Creativity Tasks and Metrics
...

## 3. LLM Creativity Evaluation
...

## 4. Test-Time Inference and Prompt Workflows
...

## 5. LLM-as-Judge and Evaluation Reliability
...

## 6. Leakage and Data Contamination
...

## Claim Support Map

## Open Citation Gaps
```

For each source, include:

- citation key;
- bibliographic details;
- what the paper says;
- how we use it;
- what not to overclaim from it;
- verification status: `verified-primary`, `needs-primary`, or
  `secondary-only`.

## How to Write `REFERENCES.bib`

Use BibTeX entries with stable citation keys. Include DOI/arXiv URL when known.
Prefer official publisher pages, arXiv pages, ACL Anthology, PNAS, APA, or
author/publication pages.

Example key style:

```bibtex
@article{boden1998creativityAI,
  ...
}
```

Do not include references that were not actually checked or are too vague. If a
source is uncertain, keep it in `RELATED_WORK_NOTES.md` as `needs-primary`
rather than adding a fake BibTeX entry.

## Worklog Requirement

Append an entry to `paper/WORKLOG.md` when:

- research starts;
- first source batch is verified;
- notes and bib are created;
- any uncertainty or missing primary source is identified.

Use the established format:

```markdown
## [YYYY-MM-DD HH:MM:SS UTC] <event>
- Action:
- Evidence:
- Artifacts:
- Error/fix:
- Current status:
```

## Important Claim Discipline During Research

Do not bend citations to fit our method. The goal is to make the paper harder to
attack.

Preferred positioning:

```text
Prior work studies creativity theory, creativity tasks, LLM creativity
evaluation, and general test-time reasoning workflows. Our paper connects these
threads by turning creativity definitions and public metric semantics into a
leakage-free elicitation workflow and evaluating the resulting profile shifts.
```

Avoid:

- saying nobody has evaluated LLM creativity;
- saying no one has used prompting for creativity;
- claiming TriSkill is the first workflow prompt method;
- claiming creativity has a universally accepted taxonomy;
- claiming LLM-as-judge metrics are ground truth.

## Current Best Next Step

Start with primary-source research and write `paper/RELATED_WORK_NOTES.md`.
Suggested first search batch:

- Boden 1998 creativity and AI.
- Boden *The Creative Mind*.
- Olson et al. 2021 DAT.
- Mednick 1962 RAT.
- Guilford / Alternative Uses.
- Gladkova et al. 2016 BATS.
- Wang et al. 2022 Self-Consistency.
- Yao et al. 2023 Tree of Thoughts.
- Madaan et al. 2023 Self-Refine.
- Zheng et al. 2023 LLM-as-judge.
- data contamination / benchmark leakage in LLM evaluation.

After this, expand to LLM creativity evaluation surveys and recent benchmark
papers.
