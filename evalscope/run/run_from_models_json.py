#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalscope.config import TaskConfig
from evalscope.run import run_task


def _parse_csv(value: str) -> List[str]:
    items = [v.strip() for v in value.split(",")]
    return [v for v in items if v]

def _load_dotenv_files() -> None:
    try:
        from dotenv import load_dotenv  # python-dotenv
    except Exception:
        return

    candidates = [
        Path.cwd() / ".env",
        Path(__file__).with_name(".env"),
        Path(__file__).with_name(".env.local"),
    ]
    for path in candidates:
        if path.exists():
            load_dotenv(dotenv_path=path, override=False)


def _load_models(models_json_path: Path) -> List[Dict[str, Any]]:
    with models_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    models = data.get("models", [])
    if not isinstance(models, list):
        raise ValueError('`models` must be a list in models.json')
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


def _select_models(
    models: List[Dict[str, Any]],
    include_names: List[str],
    include_embedding: bool,
) -> List[Dict[str, Any]]:
    if include_names:
        name_set = {n.strip() for n in include_names if n.strip()}
        selected = [m for m in models if str(m.get("name", "")).strip() in name_set]
        missing = sorted(name_set - {str(m.get("name", "")).strip() for m in selected})
        if missing:
            raise ValueError(f"Models not found in models.json: {missing}")
    else:
        selected = list(models)

    if not include_embedding:
        selected = [m for m in selected if not _is_embedding_entry(m)]
    return selected


def _make_task_configs(
    model_entries: List[Dict[str, Any]],
    datasets: List[str],
    limit: float,
    work_dir: str,
    seed: int,
    max_tokens: int,
    temperature: float,
    eval_batch_size: int,
    judge_worker_num: int,
) -> Tuple[str, List[TaskConfig]]:
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    configs: List[TaskConfig] = []
    for entry in model_entries:
        model_name = _resolve_env(entry.get("model"))
        api_url = _resolve_env(entry.get("api_url"))
        api_key = _resolve_api_key(entry)
        if not model_name or not api_url:
            raise ValueError(f"Each model entry must include `model` and `api_url`: {entry}")

        configs.append(
            TaskConfig(
                model=str(model_name),
                model_id=str(entry.get("name") or model_name),
                api_url=str(api_url),
                api_key=str(api_key),
                datasets=list(datasets),
                limit=limit,
                work_dir=os.path.join(work_dir, str(entry.get("name") or model_name)),
                seed=seed,
                eval_batch_size=eval_batch_size,
                judge_worker_num=judge_worker_num,
                generation_config={
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
        )
    return run_tag, configs


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run evalscope tasks for multiple models defined in a JSON file."
    )
    parser.add_argument(
        "--models-json",
        default=str(Path(__file__).with_name("models.json")),
        help="Path to models.json (default: evalscope/run/models.json).",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model `name` values to run. Default: run all non-embedding models in models.json.",
    )
    parser.add_argument(
        "--include-embedding",
        action="store_true",
        help="Include embedding-like entries (name/model contains embedding). Default: skip them.",
    )
    parser.add_argument(
        "--datasets",
        default="aut,dat",
        help="Comma-separated datasets to evaluate (e.g. aut,dat,cs4,creative_math,bats,metaphor).",
    )
    parser.add_argument("--limit", type=float, default=3, help="Max samples per dataset (int) or fraction (float).")
    parser.add_argument("--work-dir", default="./outputs/batch_eval", help="Base output directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=30000)
    parser.add_argument("--temperature", type=float, default=0.0)
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
        help="Per-model review/judge concurrency.",
    )

    args = parser.parse_args()

    _load_dotenv_files()

    models_json_path = Path(args.models_json).expanduser().resolve()
    models = _load_models(models_json_path)
    selected_models = _select_models(models, _parse_csv(args.models), args.include_embedding)
    if not selected_models:
        raise ValueError("No models selected to run.")

    # Bypass broken HTTP(S)_PROXY for model endpoints while keeping it for other traffic (e.g. dataset downloads).
    api_hosts = []
    for entry in selected_models:
        api_url = _resolve_env(entry.get("api_url"))
        host = urlparse(str(api_url)).hostname
        if host:
            api_hosts.append(host)
    _ensure_no_proxy_for_hosts(["localhost", "127.0.0.1", *api_hosts])

    datasets = _parse_csv(args.datasets)
    if not datasets:
        raise ValueError("No datasets specified.")

    run_tag, task_cfgs = _make_task_configs(
        model_entries=selected_models,
        datasets=datasets,
        limit=args.limit,
        work_dir=args.work_dir,
        seed=args.seed,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        eval_batch_size=args.eval_batch_size,
        judge_worker_num=args.judge_worker_num,
    )

    results = {}
    for cfg in task_cfgs:
        results[cfg.model_id] = _to_jsonable(run_task(task_cfg=cfg))

    out_dir = Path(args.work_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"summary_{run_tag}.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
