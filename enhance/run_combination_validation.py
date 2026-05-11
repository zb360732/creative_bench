#!/usr/bin/env python3
"""Run paired combinational-task validation for direct vs TriSkill prompts.

The script keeps evalscope untouched.  It writes TriSkill model outputs into
evalscope's prediction-cache format, then lets evalscope reuse those cached
predictions to run the original benchmark reviews and reports.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
from threading import Lock
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
ENHANCE_DIR = ROOT / "enhance"
EVALSCOPE_DIR = ROOT / "evalscope"
DEFAULT_TASKS = ("dat", "bats", "rat", "metaphor")
DEFAULT_DATASETS = {
    "dat": EVALSCOPE_DIR / "evalscope/benchmarks/dat/data/dat.json",
    "bats": EVALSCOPE_DIR / "evalscope/benchmarks/bats/data/bats_sampled.json",
    "rat": EVALSCOPE_DIR / "evalscope/benchmarks/rat/data/rat.json",
    "metaphor": EVALSCOPE_DIR / "evalscope/benchmarks/metaphor/data/metaphor.json",
    "aut": EVALSCOPE_DIR / "custom_eval/text/qa/exploration/aut.json",
    "creative_math": EVALSCOPE_DIR / "dataprocess/exploration/CreativeMath/data/subset.json",
    "cs4": EVALSCOPE_DIR / "dataprocess/exploration/cs4_benchmark/CS4_dataset/Story-based Base Stories.csv",
    "neocoder": EVALSCOPE_DIR / "dataprocess/exploration/NeoCoder/datasets/CodeForce/NeoCoder/NeoCoder.json",
    "transformation": EVALSCOPE_DIR / "dataprocess/transformation/generated/final_runs/transformation_eval_1235_all.json",
}
CS4_SUBSETS = ("constraints_7", "constraints_15", "constraints_23", "constraints_31", "constraints_39")

sys.path.insert(0, str(ENHANCE_DIR))
from triskill.core import enhance_prompt  # noqa: E402
from triskill.executor import run_triskill  # noqa: E402
from triskill.llm import OpenAICompatibleLLM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate direct vs TriSkill on creativity benchmark tasks.")
    parser.add_argument("--models-json", default=str(EVALSCOPE_DIR / "run/models2.json"))
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated tasks, e.g. dat,bats,rat,metaphor or aut,creative_math,cs4,neocoder.",
    )
    parser.add_argument(
        "--limit",
        type=_parse_limit,
        default=20,
        help="Max samples per task. Use none/null/full/all for the full dataset.",
    )
    parser.add_argument("--work-dir", default=str(ROOT / "outputs/combination_validation"))
    parser.add_argument("--run-name", default="models2_combination_limit20")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--judge-worker-num", type=int, default=1)
    parser.add_argument(
        "--dataset-args",
        default="",
        help="JSON string forwarded to evalscope --dataset-args without modifying evalscope code.",
    )
    parser.add_argument(
        "--judge-config",
        default="",
        help="Optional judge config JSON used by an enhance-side wrapper, e.g. evalscope/run/llm_judge2.json.",
    )
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--skip-triskill-generate", action="store_true")
    parser.add_argument("--skip-triskill-review", action="store_true")
    parser.add_argument(
        "--triskill-method",
        choices=("triskill_prompt_only", "triskill_full"),
        default="triskill_prompt_only",
        help="TriSkill variant to validate. triskill_full executes explicit workflow skills.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = _parse_tasks(args.tasks)
    work_root = Path(args.work_dir).expanduser().resolve()
    run_root = work_root / args.run_name
    direct_name = "direct"
    triskill_name = args.triskill_method
    run_root.mkdir(parents=True, exist_ok=True)

    metadata = {
        "tasks": list(tasks),
        "limit": args.limit,
        "models_json": str(Path(args.models_json).expanduser().resolve()),
        "judge_config": str(Path(args.judge_config).expanduser().resolve()) if str(args.judge_config).strip() else "",
        "work_root": str(work_root),
        "run_name": args.run_name,
        "method": triskill_name,
        "note": "TriSkill predictions are generated in enhance/ and reviewed by unmodified evalscope.",
    }
    (run_root / "validation_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.skip_direct:
        _run_evalscope(
            args=args,
            run_name=f"{args.run_name}_{direct_name}",
            datasets=tasks,
            limit=args.limit,
            work_root=work_root,
        )

    if not args.skip_triskill_generate:
        models = _load_models(Path(args.models_json))
        _generate_triskill_prediction_caches(
            models=models,
            run_dir=work_root / f"{args.run_name}_{triskill_name}",
            limit=args.limit,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            request_timeout=args.request_timeout,
            max_parallel=args.max_parallel,
            method=args.triskill_method,
            tasks=tasks,
        )

    if not args.skip_triskill_review:
        _run_evalscope(
            args=args,
            run_name=f"{args.run_name}_{triskill_name}",
            datasets=tasks,
            limit=args.limit,
            work_root=work_root,
        )

    print(f"[OK] Combination validation artifacts under: {work_root}")
    return 0


def _parse_tasks(value: str) -> tuple[str, ...]:
    tasks = tuple(task.strip().lower() for task in str(value).split(",") if task.strip())
    if not tasks:
        raise argparse.ArgumentTypeError("--tasks must contain at least one task")
    missing = [task for task in tasks if task not in DEFAULT_DATASETS]
    if missing:
        raise argparse.ArgumentTypeError(f"Unsupported task(s): {missing}")
    return tasks


def _parse_limit(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text in {"", "none", "null", "full", "all"}:
        return None
    limit = int(text)
    if limit < 0:
        return None
    if limit == 0:
        raise argparse.ArgumentTypeError("--limit must be positive, or use 'none' for the full dataset")
    return limit


def _limit_for_evalscope(limit: int | None) -> str:
    return "none" if limit is None else str(limit)


def _run_evalscope(args: argparse.Namespace, run_name: str, datasets: tuple[str, ...], limit: int | None, work_root: Path) -> None:
    runner = ["run/run_parallel_eval.py"]
    if str(args.judge_config).strip():
        runner = [
            str(ENHANCE_DIR / "run_evalscope_with_judge.py"),
            "--judge-config",
            str(Path(args.judge_config).expanduser().resolve()),
        ]
    cmd = [
        sys.executable,
        *runner,
        "--models-json",
        str(Path(args.models_json).expanduser().resolve()),
        "--datasets",
        ",".join(datasets),
        "--limit",
        _limit_for_evalscope(limit),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--request-timeout",
        str(args.request_timeout),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--judge-worker-num",
        str(args.judge_worker_num),
        "--batch-mode",
        "off",
        "--max-parallel",
        str(args.max_parallel),
        "--work-dir",
        str(work_root),
        "--run-name",
        run_name,
    ]
    if str(args.dataset_args).strip():
        cmd.extend(["--dataset-args", str(args.dataset_args)])
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=EVALSCOPE_DIR, check=True)


def _load_models(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    return [row for row in payload.get("models", []) if not _is_embedding(row)]


def _is_embedding(entry: dict[str, Any]) -> bool:
    text = " ".join(str(entry.get(key, "")).lower() for key in ("name", "model", "kind", "model_task"))
    return "embed" in text or "embedding" in text


def _generate_triskill_prediction_caches(
    models: list[dict[str, Any]],
    run_dir: Path,
    limit: int | None,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    max_parallel: int,
    method: str,
    tasks: tuple[str, ...],
) -> None:
    states: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], int, dict[str, Any]]] = []
    for model in models:
        model_id = str(model.get("name") or model.get("model"))
        pred_dir = run_dir / model_id / "predictions" / model_id
        pred_dir.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            for subset, rows in _load_task_splits(task, limit=limit):
                cache_path = pred_dir / f"{task}_{subset}.jsonl"
                expected_ids = _expected_row_ids(rows)
                if _cache_is_complete(cache_path, expected_ids):
                    print(f"[SKIP] Existing TriSkill cache: {cache_path}")
                    continue
                state = _cache_state(
                    cache_path=cache_path,
                    rows=rows,
                    model_entry=model,
                    model_id=model_id,
                    task=task,
                    subset=subset,
                )
                states.append(state)
                pending.extend(
                    (state, idx, record)
                    for idx, record in enumerate(rows)
                    if int(record.get("_sample_id", idx)) not in state["existing_by_id"]
                )

    if not pending:
        return

    workers = max(1, min(max_parallel, len(pending)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _build_and_append_cache_row,
                state=state,
                idx=idx,
                record=record,
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=request_timeout,
                method=method,
            )
            for state, idx, record in pending
        ]
        for future in as_completed(futures):
            future.result()

    for state in states:
        _finalize_cache_state(state)


def _cache_state(
    cache_path: Path,
    rows: list[dict[str, Any]],
    model_entry: dict[str, Any],
    model_id: str,
    task: str,
    subset: str,
) -> dict[str, Any]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = _dedupe_rows(_read_jsonl(cache_path))
    return {
        "cache_path": cache_path,
        "json_path": cache_path.with_suffix(".json"),
        "rows": rows,
        "expected_ids": _expected_row_ids(rows),
        "existing_by_id": {_row_id(row): row for row in existing_rows},
        "model_entry": model_entry,
        "model_id": model_id,
        "task": task,
        "subset": subset,
        "lock": Lock(),
    }


def _build_and_append_cache_row(
    state: dict[str, Any],
    idx: int,
    record: dict[str, Any],
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    method: str,
) -> None:
    row = _build_prediction_cache_row(
        idx=idx,
        record=record,
        task=state["task"],
        model_entry=state["model_entry"],
        model_id=state["model_id"],
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
        method=method,
    )
    sample_id = _row_id(row)
    with state["lock"]:
        state["existing_by_id"][sample_id] = row
        with state["cache_path"].open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
    print(f"[DONE] {state['model_id']} {state['task']} {idx + 1}/{len(state['rows'])}")


def _finalize_cache_state(state: dict[str, Any]) -> None:
    existing_by_id = state["existing_by_id"]
    ordered_rows = [existing_by_id[row_id] for row_id in state["expected_ids"] if row_id in existing_by_id]
    _rewrite_jsonl(state["cache_path"], ordered_rows)
    state["json_path"].write_text(json.dumps(ordered_rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_prediction_cache_row(
    idx: int,
    record: dict[str, Any],
    task: str,
    model_entry: dict[str, Any],
    model_id: str,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    method: str,
) -> dict[str, Any]:
    sample_id = int(record.get("_sample_id", idx))
    if method == "triskill_full":
        content, artifact = _run_full_workflow(
            task=task,
            record=record,
            model_entry=model_entry,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
        )
        metadata = {**_metadata(task, record, method=method), "artifact": artifact}
    else:
        base_prompt = _adapter_prompt(task, record)
        prompt = _final_prompt(task, enhance_prompt(base_prompt, task))
        content = _chat_completion(
            model_entry=model_entry,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=request_timeout,
        )
        metadata = _metadata(task, record, method=method)
    return _cache_row(
        idx=sample_id,
        model_name=str(model_entry.get("model") or model_id),
        model_id=model_id,
        content=content,
        metadata=metadata,
        method=method,
    )


def _load_task_splits(task: str, limit: int | None) -> list[tuple[str, list[dict[str, Any]]]]:
    if task == "creative_math":
        return [("default", _load_creative_math_records(limit))]
    if task == "cs4":
        return [(subset, _limit_records(_load_cs4_records(subset), limit)) for subset in CS4_SUBSETS]
    if task == "neocoder":
        return [("default", _load_neocoder_records(limit))]
    if task == "transformation":
        return [("default", _limit_records(_load_transformation_records(), limit))]
    records = json.loads(DEFAULT_DATASETS[task].read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Unsupported dataset format for {task}: {DEFAULT_DATASETS[task]}")
    return [("sampled" if task == "bats" else "default", _limit_records(records, limit))]


def _limit_records(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    return records if limit is None else records[:limit]


def _load_creative_math_records(limit: int | None) -> list[dict[str, Any]]:
    data = json.loads(DEFAULT_DATASETS["creative_math"].read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    sample_id = 0
    for problem_record in data:
        problem = str(problem_record.get("problem", ""))
        solutions = problem_record.get("solutions", {})
        if not isinstance(solutions, dict):
            continue
        n = len(solutions)
        solution_list = [str(solutions[str(i)]) for i in range(1, n + 1) if str(i) in solutions]
        for k in range(1, len(solution_list) + 1):
            prompt = _creative_math_prompt(problem, solution_list, k)
            records.append(
                {
                    "_sample_id": sample_id,
                    "query": prompt,
                    "problem": problem,
                    "solutions": solutions,
                    "k": k,
                    "n": n,
                    "problem_id": problem_record.get("problem_id", sample_id),
                    "competition": problem_record.get("competition", ""),
                    "difficulty": problem_record.get("difficulty", 0.0),
                    "competition_id": problem_record.get("competition_id", ""),
                }
            )
            sample_id += 1
    return _limit_records(records, limit)


def _creative_math_prompt(problem: str, solutions: list[str], k: int) -> str:
    reference_solutions = "\n\n".join(
        f"Solution {idx + 1}:\n{solution}" for idx, solution in enumerate(solutions[:k])
    )
    return f"""Criteria for evaluating the difference between two mathematical solutions include:
i). If the methods used to arrive at the solutions are fundamentally different, such as algebraic manipulation versus geometric reasoning, they can be considered distinct;
ii). Even if the final results are the same, if the intermediate steps or processes involved in reaching those solutions vary significantly, the solutions can be considered different;
iii). If two solutions rely on different assumptions or conditions, they are likely to be distinct;
iv). A solution might generalize to a broader class of problems, while another might be specific to certain conditions. In such cases, they are considered distinct;
v). If one solution is significantly simpler or more complex than the other, they can be regarded as essentially different, even if they lead to the same result.

Given the following mathematical problem:
{problem}

And some typical solutions:
{reference_solutions}

Please output a novel solution distinct from the given ones for this math problem."""


def _load_cs4_records(subset: str) -> list[dict[str, Any]]:
    constraint_num = int(subset.split("_", 1)[1])
    records: list[dict[str, Any]] = []
    with DEFAULT_DATASETS["cs4"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            if int(float(row.get("Number_of_Constraints") or 0)) != constraint_num:
                continue
            prompt = f"""Story Instruction: {row.get('Instruction', '')}

BaseStory:
{row.get('BaseStory', '')}

Task: Now revise the given BaseStory to satisfy the following constraints within 500 words:
{row.get('SelectedConstraints', '')}"""
            records.append(
                {
                    "_sample_id": idx,
                    "query": prompt,
                    "instruction": row.get("Instruction", ""),
                    "base_story": row.get("BaseStory", ""),
                    "selected_constraints": row.get("SelectedConstraints", ""),
                    "number_of_constraints": constraint_num,
                    "direction": row.get("Direction", ""),
                    "constraint_level": constraint_num,
                }
            )
    return records


def _load_neocoder_records(limit: int | None) -> list[dict[str, Any]]:
    data = json.loads(DEFAULT_DATASETS["neocoder"].read_text(encoding="utf-8"))
    template = """You are a Python code generator, only return the import and python function. Input will be an very detailed description of task, output will be the code.
The input will be from command line, and the output will be printed to the console as well. Your result will be solely a function named solve(), and do not call this function in your code.
Make sure the code is free of bug and can pass the test cases provided. You can use any library you want. The test cases are provided in the code. Do not call the solve() function in your code.

**IMPORTANT: Input Format**
- The input will be read from stdin using input() function
- You need to read multiple lines of input by calling input() multiple times
- The first line typically contains the number of test cases (t)
- For each test case, you need to read the required number of lines
- Example:
  - First line: t = int(input())
  - For each test case: read the required lines using input()
- DO NOT use input().split() only once and expect to get all data. You must call input() multiple times to read each line.

**IMPORTANT: Output Format**
- Your code should be clean and production-ready
- DO NOT include any comments in the generated code (no # comments, no docstrings)
- Only return the necessary import statements and the solve() function
- The code should be executable without any explanatory text or comments

**IMPORTANT: Output as JSON ONLY**
Return exactly one JSON object and nothing else (no markdown, no extra text).
Use this format:
{{"think":"<optional reasoning>","solve_lines":["import ...","def solve():","    ..."]}}
You may include reasoning in "think", but all reasoning MUST be inside that field.
The "solve_lines" array must be non-empty and contain ONLY valid Python code lines
that form a complete solution (imports + def solve).
If you do not want to include reasoning, set "think" to an empty string.
Do not wrap the JSON in markdown.

{question}"""
    records: list[dict[str, Any]] = []
    sample_id = 0
    for problem in data:
        statements = problem.get("problem_statements", [])
        constraints = problem.get("constraints_list", [])
        for dp_idx, (statement, constraint) in enumerate(zip(statements, constraints)):
            records.append(
                {
                    "_sample_id": sample_id,
                    "query": template.format(question=statement),
                    "problem_id": problem.get("problem_id"),
                    "dp_idx": dp_idx,
                    "constraints": constraint,
                    "problem_statement": statement,
                }
            )
            sample_id += 1
    return _limit_records(records, limit)


def _load_transformation_records() -> list[dict[str, Any]]:
    records = json.loads(DEFAULT_DATASETS["transformation"].read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Unsupported transformation dataset format: {DEFAULT_DATASETS['transformation']}")
    return [{"_sample_id": idx, **record, "query": _transformation_prompt(record)} for idx, record in enumerate(records)]


def _transformation_prompt(record: dict[str, Any]) -> str:
    return f"""Solve the following benchmark item.

Benchmark item:
{json.dumps(record, ensure_ascii=False, indent=2)}

Answer requirements:
- Explain the rebuilt core mechanism.
- Explain how operations, interfaces, records, institutions, and terminology should change.
- Explain how the system reaches the performance goals under the new rule world.
- Include migration, validation, audit, training, and fallback logic when relevant.
- Include concrete thresholds, monitoring rules, or decision triggers whenever the item naturally supports them.
- Keep the answer practical rather than literary."""


def _cache_is_complete(cache_path: Path, expected_ids: list[int]) -> bool:
    rows = _dedupe_rows(_read_jsonl(cache_path))
    if len(rows) < len(expected_ids):
        return False
    row_ids = {_row_id(row) for row in rows}
    return all(idx in row_ids for idx in expected_ids)


def _expected_row_ids(records: list[dict[str, Any]]) -> list[int]:
    return [int(record.get("_sample_id", idx)) for idx, record in enumerate(records)]


def _write_task_cache(
    cache_path: Path,
    task: str,
    subset: str,
    records: list[dict[str, Any]],
    model_entry: dict[str, Any],
    model_id: str,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    max_parallel: int,
    method: str,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = cache_path.with_suffix(".json")
    existing_rows = _dedupe_rows(_read_jsonl(cache_path))
    existing_by_id = {_row_id(row): row for row in existing_rows}

    output_rows: list[dict[str, Any]] = list(existing_rows)

    def build_row(idx: int, record: dict[str, Any]) -> dict[str, Any]:
        sample_id = int(record.get("_sample_id", idx))
        if method == "triskill_full":
            content, artifact = _run_full_workflow(
                task=task,
                record=record,
                model_entry=model_entry,
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=request_timeout,
            )
            metadata = {**_metadata(task, record, method=method), "artifact": artifact}
        else:
            base_prompt = _adapter_prompt(task, record)
            prompt = _final_prompt(task, enhance_prompt(base_prompt, task))
            content = _chat_completion(
                model_entry=model_entry,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=request_timeout,
            )
            metadata = _metadata(task, record, method=method)
        return _cache_row(
            idx=sample_id,
            model_name=str(model_entry.get("model") or model_id),
            model_id=model_id,
            content=content,
            metadata=metadata,
            method=method,
        )

    expected_ids = _expected_row_ids(records)
    if len(existing_by_id) >= len(expected_ids) and all(row_id in existing_by_id for row_id in expected_ids):
        print(f"[SKIP] Existing TriSkill cache: {cache_path}")
        ordered_rows = [existing_by_id[row_id] for row_id in expected_ids]
        _rewrite_jsonl(cache_path, ordered_rows)
        if not json_path.exists():
            json_path.write_text(json.dumps(ordered_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    pending = [
        (idx, record)
        for idx, record in enumerate(records)
        if int(record.get("_sample_id", idx)) not in existing_by_id
    ]
    workers = max(1, min(max_parallel, len(pending)))
    with cache_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(build_row, idx, record): idx for idx, record in pending}
        for future in as_completed(futures):
            idx = futures[future]
            row = future.result()
            sample_id = _row_id(row)
            existing_by_id[sample_id] = row
            output_rows = list(existing_by_id.values())
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[DONE] {model_id} {task} {idx + 1}/{len(records)}")
    output_rows = [existing_by_id[row_id] for row_id in expected_ids if row_id in existing_by_id]
    _rewrite_jsonl(cache_path, output_rows)
    json_path.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _adapter_prompt(task: str, record: dict[str, Any]) -> str:
    query = str(record.get("query", ""))
    if task == "dat":
        return query + """

Please provide your answer in the following JSON format inside <answer> tags:

<answer>
{
  "words": [
    "word1",
    "word2",
    "word3",
    "word4",
    "word5",
    "word6",
    "word7",
    "word8",
    "word9",
    "word10"
  ]
}
</answer>

Remember to put your JSON response inside <answer></answer> tags."""
    if task == "rat":
        return query + """

Please provide your answer in the following JSON format inside <answer> tags:

<answer>
{
  "word": "connecting_word"
}
</answer>

Remember to put your JSON response inside <answer></answer> tags."""
    if task == "metaphor":
        return query + """

You must return exactly one single-word replacement that preserves the meaning of the highlighted word in context.
Do not include any analysis, discussion, or extra text.
Your entire response must contain exactly one <answer>...</answer> block, using this format:

<answer>
{
  "word": "replacement_word"
}
</answer>"""
    if task == "aut":
        return query + """

Please provide your answer in the following JSON format inside <answer> tags:

<answer>
{
  "uses": [
    "use 1",
    "use 2",
    "use 3"
  ]
}
</answer>

Remember to put your JSON response inside <answer></answer> tags."""
    return query


def _final_prompt(task: str, prompt: str) -> str:
    if task in {"creative_math", "cs4"}:
        return f"""{prompt}

Final response contract:
- Output only the final answer text required by the benchmark.
- Do not include meta-commentary about the workflow.
""".strip()
    if task == "neocoder":
        return f"""{prompt}

Final response contract:
- Return exactly one JSON object with fields "think" and "solve_lines".
- The solve_lines array must contain only valid Python code lines for imports plus def solve().
- Do not use markdown or explanatory prose outside JSON.
""".strip()
    if task == "transformation":
        return f"""{prompt}

Final response contract:
- Start your visible response immediately with <answer>.
- Output exactly one <answer>...</answer> block.
- Inside the block, provide the final reconstructed system only.
- Do not include meta-commentary about the workflow.
""".strip()
    schema = {
        "dat": '<answer>{"words":["word1","word2","word3","word4","word5","word6","word7","word8","word9","word10"]}</answer>',
        "bats": '<answer>{"target":"answer_word"}</answer>',
        "rat": '<answer>{"word":"connecting_word"}</answer>',
        "metaphor": '<answer>{"word":"replacement_word"}</answer>',
        "aut": '<answer>{"uses":["use 1","use 2","use 3"]}</answer>',
    }[task]
    return f"""{prompt}

Final response contract:
- Start your visible response immediately with <answer>.
- Output exactly one <answer>...</answer> block.
- Do not write analysis, markdown, commentary, or any text before or after the block.
- Use this schema exactly: {schema}
""".strip()


def _run_full_workflow(
    task: str,
    record: dict[str, Any],
    model_entry: dict[str, Any],
    temperature: float,
    max_tokens: int,
    request_timeout: float,
) -> tuple[str, dict[str, Any]]:
    item = dict(record)
    item["query"] = _final_prompt(task, _adapter_prompt(task, record))
    api_url = str(_resolve_env(model_entry.get("api_url"))).rstrip("/")
    host = urlparse(api_url).hostname
    if host:
        _ensure_no_proxy(["localhost", "127.0.0.1", host])
    llm = OpenAICompatibleLLM(
        api_url=api_url,
        model=str(_resolve_env(model_entry.get("model"))),
        api_key=_resolve_api_key(model_entry),
        timeout=int(request_timeout),
        disable_thinking=True,
    )
    start = time.time()
    artifact = run_triskill(task, item, llm=llm, method="triskill_full")
    elapsed = time.time() - start
    artifact["runtime_config"] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "request_timeout": request_timeout,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    artifact["safe_item"] = dict(artifact.get("safe_item") or {})
    artifact["safe_item"]["query"] = _adapter_prompt(task, record)
    print(f"[WORKFLOW] {model_entry.get('model')} {task} calls={artifact.get('num_llm_calls')} {elapsed:.1f}s")
    return str(artifact.get("final_answer") or ""), artifact


def _metadata(task: str, record: dict[str, Any], method: str) -> dict[str, Any]:
    if task == "dat":
        return {"category": record.get("category", "DAT"), "triskill_method": method}
    if task == "bats":
        return {
            "word_a": record.get("word_a"),
            "word_b": record.get("word_b"),
            "word_c": record.get("word_c"),
            "direction": record.get("direction"),
            "category": record.get("category"),
            "category_name": record.get("category_name"),
            "relation_type": record.get("relation_type"),
            "triskill_method": method,
        }
    if task == "rat":
        return {"question": record.get("question"), "category": record.get("category", "RAT"), "triskill_method": method}
    if task == "metaphor":
        return {
        "metaphor_word": record.get("metaphor_word"),
        "category": record.get("category", "METAPHOR"),
        "novelty": record.get("novelty"),
        "triskill_method": method,
        }
    metadata = {key: value for key, value in record.items() if not str(key).startswith("_") and key != "query"}
    metadata["triskill_method"] = method
    if task == "transformation":
        metadata["item"] = {key: value for key, value in record.items() if not str(key).startswith("_") and key != "query"}
    return metadata


def _cache_row(idx: int, model_name: str, model_id: str, content: str, metadata: dict[str, Any], method: str) -> dict[str, Any]:
    message = {
        "content": content,
        "source": "generate",
        "metadata": {"method": method},
        "internal": None,
        "role": "assistant",
        "tool_calls": None,
        "model": model_name,
    }
    return {
        "index": idx,
        "sample_id": idx,
        "sample_key": None,
        "model": model_name,
        "model_output": {
            "model": model_name,
            "choices": [{"message": message, "stop_reason": "stop", "logprobs": None}],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "time": None,
            "metadata": {"model_id": model_id, "method": method},
            "error": None,
        },
        "messages": [message],
        "metadata": metadata,
    }


def _chat_completion(
    model_entry: dict[str, Any],
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> str:
    api_url = str(_resolve_env(model_entry.get("api_url"))).rstrip("/")
    api_key = _resolve_api_key(model_entry)
    model = str(_resolve_env(model_entry.get("model")))
    host = urlparse(api_url).hostname
    if host:
        _ensure_no_proxy(["localhost", "127.0.0.1", host])
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_url + "/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    for attempt in range(3):
        try:
            start = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            elapsed = time.time() - start
            print(f"[CALL] {model} {elapsed:.1f}s")
            return str(body["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if attempt == 2:
                raise RuntimeError(f"HTTP {exc.code}: {detail[:500]}") from exc
            time.sleep(5 * (attempt + 1))
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def _resolve_env(value: Any) -> Any:
    return os.path.expandvars(value) if isinstance(value, str) else value


def _resolve_api_key(entry: dict[str, Any]) -> str:
    api_key = str(_resolve_env(entry.get("api_key", "EMPTY")))
    api_key_env = entry.get("api_key_env")
    if api_key_env:
        api_key = os.getenv(str(api_key_env), api_key)
    if api_key in {"", "YOUR_API_KEY"}:
        api_key = os.getenv("EVALSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or api_key
    return api_key or "EMPTY"


def _ensure_no_proxy(hosts: list[str]) -> None:
    for key in ("NO_PROXY", "no_proxy"):
        current = [part.strip() for part in os.environ.get(key, "").split(",") if part.strip()]
        for host in hosts:
            if host and host not in current:
                current.append(host)
        os.environ[key] = ",".join(current)


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"[WARN] Ignoring malformed JSONL row while resuming: {path}")
    return rows


def _row_id(row: dict[str, Any]) -> int:
    return int(row.get("sample_id", row.get("index", 0)))


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        by_id[_row_id(row)] = row
    return [by_id[idx] for idx in sorted(by_id)]


def _rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
