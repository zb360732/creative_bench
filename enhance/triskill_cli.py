#!/usr/bin/env python3
"""Command line entry point for TriSkill prompt enhancement."""

from __future__ import annotations

import argparse
import sys

from triskill import TriSkillEnhancer, profile_task
from triskill.analysis import write_summary
from triskill.dataset import enhance_dataset_file
from triskill.evalscope_bridge import artifacts_to_predictions
from triskill.llm import OpenAICompatibleLLM
from triskill.paper_pipeline import audit_artifacts, create_experiment_manifest, join_scores, write_scored_summary
from triskill.runner import run_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhance a creativity benchmark prompt with TriSkill instructions.")
    parser.add_argument("--task", required=True, help="Benchmark task name, e.g. dat, rat, aut, bats, metaphor, cs4.")
    parser.add_argument("--metric", action="append", default=None, help="Raw metric name. Can be repeated.")
    parser.add_argument("--prompt", default=None, help="Prompt text. If omitted, read from stdin.")
    parser.add_argument("--input", default=None, help="Input JSON/JSONL dataset to enhance.")
    parser.add_argument("--output", default=None, help="Output JSONL artifact path for --input mode.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max records for --input mode.")
    parser.add_argument("--method", default="triskill_prompt_only", help="direct, generic_creativity_prompt, cot_structured, triskill_prompt_only, triskill_full, triskill_without_verifier, triskill_wrong_skill_assignment.")
    parser.add_argument("--use-env-llm", action="store_true", help="Use OpenAI-compatible env vars TRISKILL_API_URL/TRISKILL_MODEL for methods that call an LLM.")
    parser.add_argument("--schema", default=None, help="Override final answer schema instruction.")
    parser.add_argument("--constraint", action="append", default=None, help="Visible constraint. Can be repeated.")
    parser.add_argument("--plan", action="store_true", help="Print the TriSkill plan JSON instead of the enhanced prompt.")
    parser.add_argument("--with-plan", action="store_true", help="Print the plan JSON and enhanced prompt JSON payload.")
    parser.add_argument("--analyze", action="store_true", help="Analyze one or more JSONL artifact files passed via --input comma-separated paths.")
    parser.add_argument("--to-predictions", action="store_true", help="Convert artifact JSONL from --input to prediction JSON/JSONL at --output.")
    parser.add_argument("--include-prompts", action="store_true", help="Include prompts in --to-predictions output.")
    parser.add_argument("--make-manifest", action="store_true", help="Write a paper experiment manifest to --output.")
    parser.add_argument("--audit", action="store_true", help="Audit artifact JSONL from --input and write report to --output.")
    parser.add_argument("--scores", default=None, help="Score JSON/JSONL path for --join-scores.")
    parser.add_argument("--join-scores", action="store_true", help="Join artifact JSONL from --input with --scores into --output.")
    parser.add_argument("--scored-summary", action="store_true", help="Summarize scored JSONL files from comma-separated --input into --output.")
    parser.add_argument("--primary-score", default="score", help="Primary score key for --scored-summary.")
    parser.add_argument("--baseline-method", default="direct", help="Baseline method for score deltas.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.make_manifest:
        if not args.output:
            raise SystemExit("--make-manifest requires --output")
        rows = create_experiment_manifest(args.output, limit=args.limit)
        print(f"wrote {len(rows)} manifest rows to {args.output}")
        return 0
    if args.audit:
        if not args.input or not args.output:
            raise SystemExit("--audit requires --input and --output")
        report = audit_artifacts(args.input)
        import json
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote audit report to {args.output}")
        return 0
    if args.join_scores:
        if not args.input or not args.output or not args.scores:
            raise SystemExit("--join-scores requires --input, --scores, and --output")
        rows = join_scores(args.input, args.scores, args.output)
        print(f"wrote {len(rows)} scored rows to {args.output}")
        return 0
    if args.scored_summary:
        if not args.input or not args.output:
            raise SystemExit("--scored-summary requires --input and --output")
        paths = [part.strip() for part in args.input.split(",") if part.strip()]
        write_scored_summary(paths, args.output, primary_score=args.primary_score, baseline_method=args.baseline_method)
        print(f"wrote scored summary to {args.output}")
        return 0
    if args.analyze:
        if not args.input or not args.output:
            raise SystemExit("--analyze requires --input and --output")
        paths = [part.strip() for part in args.input.split(",") if part.strip()]
        write_summary(paths, args.output)
        print(f"wrote analysis summary to {args.output}")
        return 0
    if args.to_predictions:
        if not args.input or not args.output:
            raise SystemExit("--to-predictions requires --input and --output")
        rows = artifacts_to_predictions(args.input, args.output, include_prompts=args.include_prompts)
        print(f"wrote {len(rows)} predictions to {args.output}")
        return 0
    if args.input:
        if not args.output:
            raise SystemExit("--output is required when --input is used")
        if args.method == "triskill_prompt_only" and not args.use_env_llm:
            rows = enhance_dataset_file(args.task, args.input, args.output, limit=args.limit)
        else:
            llm = OpenAICompatibleLLM.from_env() if args.use_env_llm else None
            rows = run_dataset(args.task, args.input, args.output, method=args.method, llm=llm, limit=args.limit)
        print(f"wrote {len(rows)} enhanced artifacts to {args.output}")
        return 0

    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    profile = profile_task(
        task_name=args.task,
        raw_metrics=args.metric,
        output_schema=args.schema,
        visible_constraints=args.constraint,
    )
    enhancer = TriSkillEnhancer(profile)
    if args.with_plan:
        print(enhancer.as_json(prompt))
    elif args.plan:
        print(enhancer.as_json(None))
    else:
        print(enhancer.enhance(prompt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
