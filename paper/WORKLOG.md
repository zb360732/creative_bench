# Paper Worklog

This log records paper-writing decisions, evidence, produced artifacts, and
remaining risks. It is separate from implementation logs under `enhance/`.

## [2026-05-11 18:16:51 UTC] Start paper protocol
- Action: Started the paper-writing process by inspecting `paper/`, `paper_context/`, `enhance/PAPER_MATERIALS.md`, and the implementation worklog.
- Evidence: `paper/` currently contains ACL style files; `paper_context/00_project_understanding.md` contains the current project narrative and known experimental caveats; `enhance/PAPER_MATERIALS.md` contains method vocabulary, operator taxonomy, and result summaries.
- Artifacts: Planning to create `paper/PAPER_WORKFLOW.md` as the canonical writing and quality-control protocol.
- Error/fix: None.
- Current status: Defining a high-standard, reproducible paper workflow before drafting the manuscript.

## [2026-05-11 18:20:00 UTC] Create paper workflow protocol
- Action: Added the paper-writing protocol covering target standard, claim rules, staged workflow, figure/table requirements, evidence standards, novelty framing, leakage audit, writing style, and immediate next actions.
- Evidence: `paper/PAPER_WORKFLOW.md` now defines eight stages from evidence inventory through internal review, with required artifacts and stop conditions for each stage.
- Artifacts: `paper/PAPER_WORKFLOW.md`, `paper/WORKLOG.md`.
- Error/fix: None.
- Current status: Ready to start Stage 0 evidence inventory before drafting manuscript text.

## [2026-05-11 18:26:47 UTC] Create evidence ledger
- Action: Scanned local `outputs/**/summary.json` files and the merged direct baseline under `huizong/merged_evalscope`, then created the first paper evidence ledger.
- Evidence: The ledger records full combinational exact-metric direct vs TriSkill runs, qwen3.5-9b merged direct baseline vs current full TriSkill, latest DAT/BATS diagnostics, exploratory full/limit20 diagnostics, transformation limit5 diagnostics, rejected iterations, and current evidence gaps.
- Artifacts: `paper/EVIDENCE_LEDGER.md`.
- Error/fix: `jq` was unavailable in the environment, so JSON was inspected with one-off read-only Python commands; paper files were still written via `apply_patch`.
- Current status: Stage 0 has a first-pass ledger. Next step is to turn this into table-generation scripts or a stricter `CLAIM_MATRIX.md`.

## [2026-05-11 18:31:25 UTC] Create claim matrix
- Action: Created a claim matrix that maps candidate paper claims to evidence status, manuscript placement, risk controls, and required follow-up experiments.
- Evidence: `paper/CLAIM_MATRIX.md` separates core positioning, method, main empirical, boundary/negative, and forbidden claims. It explicitly blocks universal-improvement, fully task-agnostic, closed-source TriSkill validation, replay-as-final, and incomparable transformation claims.
- Artifacts: `paper/CLAIM_MATRIX.md`.
- Error/fix: None.
- Current status: Paper claims now have a gating document. Next step is Stage 1 related-work review or Stage 2 formalism, depending on whether we want to lock conceptual framing or notation first.

## [2026-05-11 18:39:13 UTC] Draft formalism
- Action: Added a formalism document defining task instances, visible/hidden information, leakage-free elicitation, creativity levels, canonical objective mapping, task profiles, operators, workflows, elicitation methods, runtime artifacts, and creativity profile shift.
- Evidence: `paper/FORMALISM.md` includes a notation table, ten definitions, component classification, result interpretation rules, and draft-ready paragraphs for the problem setup, TriSkill overview, and profile shift.
- Artifacts: `paper/FORMALISM.md`.
- Error/fix: None.
- Current status: Stage 2 now has a first-pass formal definition layer. Remaining tasks are citations for the taxonomy and exact evaluator/judge notation.

## [2026-05-11 18:59:57 UTC] Prepare fresh-session handoff
- Action: Created a handoff document for continuing the paper work in a new Codex session after context pressure.
- Evidence: `paper/NEXT_CODEX_HANDOFF.md` summarizes branch state, dirty files not to touch, completed paper artifacts, current claim boundaries, and the next Stage 1 research task with required literature buckets.
- Artifacts: `paper/NEXT_CODEX_HANDOFF.md`.
- Error/fix: The previous context was approaching model-window limits during literature search, so the handoff captures the exact next task before continuing.
- Current status: New sessions can continue by reading `paper/NEXT_CODEX_HANDOFF.md` first.
