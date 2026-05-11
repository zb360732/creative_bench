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
