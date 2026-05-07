#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
import atexit
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalscope.config import TaskConfig
from evalscope.run import run_task


def _parse_csv(value: str) -> List[str]:
    items = [v.strip() for v in value.split(",")]
    return [v for v in items if v]


def _load_models(models_json_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(models_json_path.read_text(encoding="utf-8"))
    models = data.get("models", [])
    if not isinstance(models, list):
        raise ValueError("`models` must be a list in models.json")
    return models


def _resolve_env(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return os.path.expandvars(value)


def _resolve_api_key(entry: Dict[str, Any]) -> str:
    api_key = str(_resolve_env(entry.get("api_key", "EMPTY")))
    api_key_env = entry.get("api_key_env")
    if api_key_env:
        api_key = os.getenv(str(api_key_env), api_key)
    if api_key in {"", "YOUR_API_KEY"}:
        api_key = os.getenv("EVALSCOPE_API_KEY", api_key)
    if api_key in {"", "YOUR_API_KEY"}:
        api_key = os.getenv("OPENAI_API_KEY", api_key)
    return api_key


def _judge_model_args(entry: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
    return {
        "api_url": str(_resolve_env(entry.get("api_url"))),
        "api_key": _resolve_api_key(entry),
        "model_id": str(_resolve_env(entry.get("model"))),
        "generation_config": {"temperature": 0.0, "max_tokens": max_tokens},
    }


def _load_default_judge_model_args_list(max_tokens: int) -> List[Dict[str, Any]]:
    judge_cfg_path = Path(__file__).with_name('llm_judge.json')
    data = json.loads(judge_cfg_path.read_text(encoding='utf-8'))
    models = data.get('models', [])
    if not models:
        raise ValueError(f'No judge models found in {judge_cfg_path}')
    return [_judge_model_args(model, max_tokens=max_tokens) for model in models]


def _load_default_judge_model_args(max_tokens: int) -> Dict[str, Any]:
    return _load_default_judge_model_args_list(max_tokens=max_tokens)[0]


def _generation_config(entry: Dict[str, Any], max_tokens: int, temperature: float, timeout: float) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout": timeout,
        "retries": 1,
        "retry_interval": 10,
    }

    model_name = str(_resolve_env(entry.get("model") or ""))
    if model_name.startswith("qwen3"):
        config["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": False,
            }
        }

    return config


def _parse_batch_mode(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"on", "true", "1", "yes"}:
        return True
    if text in {"off", "false", "0", "no"}:
        return False
    raise ValueError(f"Invalid --batch-mode value: {value}. Expected one of: on, off")


def _ensure_no_proxy_for_hosts(hosts: List[str]) -> None:
    clean_hosts = [h.strip() for h in hosts if h and h.strip()]
    if not clean_hosts:
        return
    for env_key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(env_key, "")
        parts = [p.strip() for p in current.split(",") if p.strip()]
        merged = parts + [h for h in clean_hosts if h not in parts]
        os.environ[env_key] = ",".join(merged)


def _is_embedding_entry(entry: Dict[str, Any]) -> bool:
    name = str(entry.get("name", "")).lower()
    model = str(entry.get("model", "")).lower()
    kind = str(entry.get("kind", "")).lower()
    model_task = str(entry.get("model_task", "")).lower()
    return (
        "embed" in name
        or "embedding" in name
        or "embed" in model
        or "embedding" in model
        or kind == "embedding"
        or model_task == "embedding"
    )


def _select_models(models: List[Dict[str, Any]], include_names: List[str], include_embedding: bool) -> List[Dict[str, Any]]:
    if include_names:
        wanted = {n.strip() for n in include_names if n.strip()}
        selected = [m for m in models if str(m.get("name", "")).strip() in wanted]
        missing = sorted(wanted - {str(m.get("name", "")).strip() for m in selected})
        if missing:
            raise ValueError(f"Models not found in models.json: {missing}")
        return selected
    if include_embedding:
        return list(models)
    return [m for m in models if not _is_embedding_entry(m)]


def _parse_limit(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    if "." in text:
        return float(text)
    return float(int(text))


def _slugify(text: str) -> str:
    cleaned = []
    for ch in text:
        if ch.isalnum():
            cleaned.append(ch.lower())
        elif ch in {"-", "_"}:
            cleaned.append("_")
        else:
            cleaned.append("_")
    slug = "".join(cleaned)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


@dataclass
class _TeeStream:
    streams: List[Any]

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def write(self, data: str) -> None:
        for stream in self.streams:
            try:
                stream.write(data)
            except Exception:
                continue

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                continue

    def isatty(self) -> bool:
        for stream in self.streams:
            if hasattr(stream, "isatty") and stream.isatty():
                return True
        return False

    @property
    def encoding(self) -> str:
        for stream in self.streams:
            enc = getattr(stream, "encoding", None)
            if enc:
                return enc
        return "utf-8"


def _limit_slug(limit: Optional[float]) -> str:
    if limit is None:
        return "full"
    text = str(limit).replace(".", "p")
    return f"limit{text}"


def _default_outputs_root(datasets: List[str]) -> Path:
    benchmark_root = Path(__file__).resolve().parents[2] / "outputs"
    if len(datasets) > 1:
        return benchmark_root / "combination"
    return benchmark_root


def _has_complete_reports(base_dir: Path, model_id: str, datasets: List[str]) -> bool:
    reports_dir = base_dir / model_id / "reports" / model_id
    if not reports_dir.exists():
        return False
    for dataset in datasets:
        report_path = reports_dir / f"{dataset}.json"
        if not report_path.exists():
            return False
    return True


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return value.to_dict()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _worker(
    entry: Dict[str, Any],
    datasets: List[str],
    limit: Optional[float],
    work_dir: str,
    dataset_args: Dict[str, Any],
    seed: int,
    max_tokens: int,
    temperature: float,
    request_timeout: float,
    eval_batch_size: int,
    judge_worker_num: int,
    use_batch_processing: bool,
    queue: Queue,
) -> None:
    try:
        model_name = _resolve_env(entry.get("model"))
        api_url = _resolve_env(entry.get("api_url"))
        api_key = _resolve_api_key(entry)
        model_id = str(entry.get("name") or model_name)

        host = urlparse(str(api_url)).hostname
        if host:
            _ensure_no_proxy_for_hosts(["localhost", "127.0.0.1", host])

        cfg = TaskConfig(
            model=str(model_name),
            model_id=model_id,
            api_url=str(api_url),
            api_key=str(api_key),
            datasets=list(datasets),
            limit=limit,
            dataset_args=dataset_args,
            eval_batch_size=eval_batch_size,
            use_cache=work_dir,
            work_dir=work_dir,
            no_timestamp=True,
            seed=seed,
            judge_worker_num=judge_worker_num,
            ignore_errors=True,
            use_batch_processing=use_batch_processing,
            generation_config=_generation_config(
                entry,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=request_timeout,
            ),
        )
        if any(dataset in datasets for dataset in {"creative_math", "cs4", "drivel_writing", "transformation"}):
            cfg.judge_model_args = _load_default_judge_model_args(max_tokens=max_tokens)
            cfg.judge_model_args_list = _load_default_judge_model_args_list(max_tokens=max_tokens)
        result = run_task(task_cfg=cfg)
        queue.put(("ok", model_id, _to_jsonable(result)))
    except Exception as exc:
        queue.put(("error", str(entry.get("name", "")), repr(exc)))


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Parallel eval across models in models.json.")
    parser.add_argument(
        "--models-json",
        default=str(Path(__file__).with_name("models.json")),
        help="Path to models.json (default: evalscope/run/models.json).",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model `name` values to run. Default: run all non-embedding models.",
    )
    parser.add_argument(
        "--datasets",
        default="aut,dat,bats,rat,metaphor",
        help="Comma-separated datasets to evaluate.",
    )
    parser.add_argument(
        "--limit",
        default="none",
        help="Max samples per dataset. Use 'none' for full dataset.",
    )
    parser.add_argument(
        "--work-dir",
        default="",
        help="Base output directory. Defaults to benchmark/outputs (or benchmark/outputs/combination for multiple datasets).",
    )
    parser.add_argument(
        "--dataset-args",
        default="{}",
        help="JSON string for dataset args (e.g. '{\"problem_method\": {\"extra_params\": {\"evaluation_mode\": \"full\"}}}').",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Run folder name under work-dir. Defaults to <datasets>_<limit> (e.g. aut_dat_full).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=30000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=180,
        help="Per-request timeout in seconds for model inference.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=1,
        help="Per-model inference concurrency (number of samples evaluated in parallel).",
    )
    parser.add_argument(
        "--judge-worker-num",
        type=int,
        default=1,
        help="Per-model review/judge concurrency. Set this explicitly if judge stages should also run in parallel.",
    )
    parser.add_argument(
        "--batch-mode",
        default="on",
        help="Batch processing switch for supported inference/review paths: on or off.",
    )
    parser.add_argument("--max-parallel", type=int, default=0, help="Max concurrent models (0 = all).")
    parser.add_argument("--include-embedding", action="store_true", help="Include embedding-like entries.")
    parser.add_argument(
        "--no-skip-done",
        action="store_true",
        help="Do not skip models that already have complete reports under the run folder.",
    )
    parser.add_argument(
        "--skip-done-from",
        default="",
        help="Path to a summary.json to skip models with status=ok.",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="Path to run log file. Defaults to <run-dir>/run.log. Use '-' or 'none' to disable.",
    )

    args = parser.parse_args(argv)

    models_json_path = Path(args.models_json).expanduser().resolve()
    models = _load_models(models_json_path)
    selected = _select_models(models, _parse_csv(args.models), args.include_embedding)
    if not selected:
        raise ValueError("No models selected to run.")

    datasets = _parse_csv(args.datasets)
    limit = _parse_limit(args.limit)
    use_batch_processing = _parse_batch_mode(args.batch_mode)
    max_parallel = args.max_parallel if args.max_parallel > 0 else len(selected)
    try:
        dataset_args = json.loads(args.dataset_args) if args.dataset_args else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --dataset-args: {exc}") from exc

    skip_done = set()
    if args.skip_done_from:
        summary_path = Path(args.skip_done_from).expanduser().resolve()
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for model_id, info in summary.items():
                if isinstance(info, dict) and info.get("status") == "ok":
                    skip_done.add(model_id)

    work_root = Path(args.work_dir).expanduser().resolve() if args.work_dir else _default_outputs_root(datasets)
    if args.run_name:
        run_name = args.run_name
    else:
        datasets_slug = "_".join(_slugify(ds) for ds in datasets)
        run_name = f"{datasets_slug}_{_limit_slug(limit)}"

    base_dir = work_root / run_name
    base_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(args.log_file).strip()
    if not log_path:
        log_path = str(base_dir / "run.log")
    if log_path.lower() in {"-", "none", "null", "false"}:
        log_path = ""
    if log_path:
        log_file_path = Path(log_path).expanduser().resolve()
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            log_handle = log_file_path.open("w", encoding="utf-8")
            atexit.register(log_handle.close)
            sys.stdout = _TeeStream([sys.stdout, log_handle])
            sys.stderr = _TeeStream([sys.stderr, log_handle])
            print(f"[INFO] Logging to: {log_file_path}")
        except Exception as exc:
            print(f"[WARN] Failed to open log file {log_file_path}: {exc}")

    if skip_done:
        selected = [m for m in selected if str(m.get("name") or m.get("model")) not in skip_done]

    if not args.no_skip_done:
        remaining = []
        for entry in selected:
            model_id = str(entry.get("name") or entry.get("model"))
            if _has_complete_reports(base_dir, model_id, datasets):
                continue
            remaining.append(entry)
        selected = remaining
    if not selected:
        print("[OK] All selected models already completed; nothing to run.")
        return 0

    queue: Queue = Queue()
    active: List[Process] = []
    pending = list(selected)

    def _start_next():
        entry = pending.pop(0)
        model_id = str(entry.get("name") or entry.get("model"))
        work_dir = str(base_dir / model_id)
        proc = Process(
            target=_worker,
            args=(
                entry,
                datasets,
                limit,
                work_dir,
                dataset_args,
                args.seed,
                args.max_tokens,
                args.temperature,
                args.request_timeout,
                args.eval_batch_size,
                args.judge_worker_num,
                use_batch_processing,
                queue,
            ),
        )
        proc.start()
        active.append(proc)

    while pending or active:
        while pending and len(active) < max_parallel:
            _start_next()
        # Reap finished processes
        alive = []
        for proc in active:
            if proc.is_alive():
                alive.append(proc)
            else:
                proc.join()
        active = alive
        time.sleep(0.5)

    results: Dict[str, Any] = {}
    while not queue.empty():
        status, model_id, payload = queue.get()
        results[model_id] = {"status": status, "result": payload}

    summary_path = base_dir / "summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
