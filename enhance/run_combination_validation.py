#!/usr/bin/env python3
"""Run paired combinational-task validation for direct vs TriSkill prompts.

The script keeps evalscope untouched.  It writes TriSkill model outputs into
evalscope's prediction-cache format, then lets evalscope reuse those cached
predictions to run the original benchmark reviews and reports.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
ENHANCE_DIR = ROOT / "enhance"
EVALSCOPE_DIR = ROOT / "evalscope"
TASKS = ("dat", "bats", "rat", "metaphor")
DEFAULT_DATASETS = {
    "dat": EVALSCOPE_DIR / "evalscope/benchmarks/dat/data/dat.json",
    "bats": EVALSCOPE_DIR / "evalscope/benchmarks/bats/data/bats_sampled.json",
    "rat": EVALSCOPE_DIR / "evalscope/benchmarks/rat/data/rat.json",
    "metaphor": EVALSCOPE_DIR / "evalscope/benchmarks/metaphor/data/metaphor.json",
}

sys.path.insert(0, str(ENHANCE_DIR))
from triskill.core import enhance_prompt  # noqa: E402
from triskill.executor import run_triskill  # noqa: E402
from triskill.llm import OpenAICompatibleLLM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TriSkill on four combinational creativity tasks.")
    parser.add_argument("--models-json", default=str(EVALSCOPE_DIR / "run/models2.json"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--work-dir", default=str(ROOT / "outputs/combination_validation"))
    parser.add_argument("--run-name", default="models2_combination_limit50")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--judge-worker-num", type=int, default=1)
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
    work_root = Path(args.work_dir).expanduser().resolve()
    run_root = work_root / args.run_name
    direct_name = "direct"
    triskill_name = args.triskill_method
    run_root.mkdir(parents=True, exist_ok=True)

    metadata = {
        "tasks": list(TASKS),
        "limit": args.limit,
        "models_json": str(Path(args.models_json).expanduser().resolve()),
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
            datasets=TASKS,
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
        )

    if not args.skip_triskill_review:
        _run_evalscope(
            args=args,
            run_name=f"{args.run_name}_{triskill_name}",
            datasets=TASKS,
            limit=args.limit,
            work_root=work_root,
        )

    print(f"[OK] Combination validation artifacts under: {work_root}")
    return 0


def _run_evalscope(args: argparse.Namespace, run_name: str, datasets: tuple[str, ...], limit: int, work_root: Path) -> None:
    cmd = [
        sys.executable,
        "run/run_parallel_eval.py",
        "--models-json",
        str(Path(args.models_json).expanduser().resolve()),
        "--datasets",
        ",".join(datasets),
        "--limit",
        str(limit),
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
    limit: int,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    max_parallel: int,
    method: str,
) -> None:
    jobs: list[tuple[dict[str, Any], str, str, Path, list[dict[str, Any]]]] = []
    for model in models:
        model_id = str(model.get("name") or model.get("model"))
        pred_dir = run_dir / model_id / "predictions" / model_id
        pred_dir.mkdir(parents=True, exist_ok=True)
        for task in TASKS:
            rows = _load_task_records(task, limit=limit)
            subset = "sampled" if task == "bats" else "default"
            jobs.append((model, model_id, task, pred_dir / f"{task}_{subset}.jsonl", rows))

    workers = max(1, min(max_parallel, len(jobs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _write_task_cache,
                cache_path=cache_path,
                task=task,
                records=rows,
                model_entry=model,
                model_id=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=request_timeout,
                max_parallel=max(1, max_parallel // workers),
                method=method,
            )
            for model, model_id, task, cache_path, rows in jobs
        ]
        for future in as_completed(futures):
            future.result()


def _load_task_records(task: str, limit: int) -> list[dict[str, Any]]:
    records = json.loads(DEFAULT_DATASETS[task].read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Unsupported dataset format for {task}: {DEFAULT_DATASETS[task]}")
    return records[:limit]


def _write_task_cache(
    cache_path: Path,
    task: str,
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
    existing_rows = _read_jsonl(cache_path)
    if len(existing_rows) >= len(records):
        print(f"[SKIP] Existing TriSkill cache: {cache_path}")
        if not json_path.exists():
            json_path.write_text(json.dumps(existing_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    start = len(existing_rows)
    output_rows: list[dict[str, Any]] = list(existing_rows)

    def build_row(idx: int, record: dict[str, Any]) -> dict[str, Any]:
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
            idx=idx,
            model_name=str(model_entry.get("model") or model_id),
            model_id=model_id,
            content=content,
            metadata=metadata,
            method=method,
        )

    pending = list(enumerate(records[start:], start=start))
    workers = max(1, min(max_parallel, len(pending)))
    with cache_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(build_row, idx, record): idx for idx, record in pending}
        for future in as_completed(futures):
            idx = futures[future]
            row = future.result()
            output_rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[DONE] {model_id} {task} {idx + 1}/{len(records)}")
    output_rows = sorted(output_rows, key=lambda row: int(row.get("sample_id", row.get("index", 0))))
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
    return query


def _final_prompt(task: str, prompt: str) -> str:
    schema = {
        "dat": '<answer>{"words":["word1","word2","word3","word4","word5","word6","word7","word8","word9","word10"]}</answer>',
        "bats": '<answer>{"target":"answer_word"}</answer>',
        "rat": '<answer>{"word":"connecting_word"}</answer>',
        "metaphor": '<answer>{"word":"replacement_word"}</answer>',
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
    )
    start = time.time()
    artifact = run_triskill(task, item, llm=llm, method="triskill_full")
    elapsed = time.time() - start
    artifact["runtime_config"] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "request_timeout": request_timeout,
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
    return {
        "metaphor_word": record.get("metaphor_word"),
        "category": record.get("category", "METAPHOR"),
        "novelty": record.get("novelty"),
        "triskill_method": method,
    }


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
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
