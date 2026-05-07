#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from datetime import datetime
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


def _select_models(models: List[Dict[str, Any]], include_names: List[str]) -> List[Dict[str, Any]]:
    if include_names:
        wanted = {n.strip() for n in include_names if n.strip()}
        selected = [m for m in models if str(m.get("name", "")).strip() in wanted]
        missing = sorted(wanted - {str(m.get("name", "")).strip() for m in selected})
        if missing:
            raise ValueError(f"Models not found in models.json: {missing}")
        return selected
    return [m for m in models if not _is_embedding_entry(m)]


def _judge_model_args(entry: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
    return {
        "api_url": str(_resolve_env(entry.get("api_url"))),
        "api_key": _resolve_api_key(entry),
        "model_id": str(_resolve_env(entry.get("model"))),
        "generation_config": {"temperature": 0.0, "max_tokens": max_tokens},
    }


def _load_default_judge_model_args(max_tokens: int) -> Dict[str, Any]:
    judge_cfg_path = Path(__file__).with_name('llm_judge.json')
    data = json.loads(judge_cfg_path.read_text(encoding='utf-8'))
    models = data.get('models', [])
    if not models:
        raise ValueError(f'No judge models found in {judge_cfg_path}')
    return _judge_model_args(models[0], max_tokens=max_tokens)


def _run_one(
    entry: Dict[str, Any],
    dataset: str,
    work_dir: str,
    limit: Optional[int],
    seed: int,
    max_tokens: int,
    temperature: float,
    eval_batch_size: int,
    judge_worker_num: int,
) -> Dict[str, Any]:
    model_name = _resolve_env(entry.get("model"))
    api_url = _resolve_env(entry.get("api_url"))
    api_key = _resolve_api_key(entry)
    model_id = str(entry.get("name") or model_name)

    cfg = TaskConfig(
        model=str(model_name),
        model_id=model_id,
        api_url=str(api_url),
        api_key=str(api_key),
        datasets=[dataset],
        limit=limit,
        work_dir=work_dir,
        seed=seed,
        eval_batch_size=eval_batch_size,
        judge_worker_num=judge_worker_num,
        generation_config={
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )

    if dataset == "drivel_writing":
        cfg.judge_model_args = _load_default_judge_model_args(max_tokens=max_tokens)

    return run_task(task_cfg=cfg)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Run NeoCoder, CreativeMath, and drivel_writing for multiple models.")
    parser.add_argument(
        "--models-json",
        default=str(Path(__file__).with_name("models2.json")),
        help="Path to models.json (default: evalscope/run/models2.json).",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model `name` values to run. Default: run all non-embedding models in models.json.",
    )
    parser.add_argument("--work-dir", default="./outputs/remaining_three", help="Base output directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=30000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Per-model inference concurrency.")
    parser.add_argument("--judge-worker-num", type=int, default=1, help="Per-model review/judge concurrency.")
    parser.add_argument("--neocoder-limit", type=int, default=200, help="Sample limit for neocoder (avoid huge runs).")
    parser.add_argument("--full", action="store_true", help="Run full neocoder as well (ignores --neocoder-limit).")

    args = parser.parse_args(argv)

    models_json_path = Path(args.models_json).expanduser().resolve()
    models = _load_models(models_json_path)
    selected = _select_models(models, _parse_csv(args.models))
    if not selected:
        raise ValueError("No models selected to run.")

    api_hosts = []
    for entry in selected:
        api_url = _resolve_env(entry.get("api_url"))
        host = urlparse(str(api_url)).hostname
        if host:
            api_hosts.append(host)
    _ensure_no_proxy_for_hosts(["localhost", "127.0.0.1", *api_hosts])

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.work_dir).expanduser().resolve() / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}
    for entry in selected:
        model_name = str(entry.get("name") or entry.get("model"))
        results[model_name] = {}

        # creative_math full (605 expanded samples)
        results[model_name]["creative_math"] = _run_one(
            entry=entry,
            dataset="creative_math",
            work_dir=str(out_dir / model_name / "creative_math"),
            limit=None,
            seed=args.seed,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            eval_batch_size=args.eval_batch_size,
            judge_worker_num=args.judge_worker_num,
        )

        # drivel_writing full (requires judge_model_args)
        results[model_name]["drivel_writing"] = _run_one(
            entry=entry,
            dataset="drivel_writing",
            work_dir=str(out_dir / model_name / "drivel_writing"),
            limit=None,
            seed=args.seed,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            eval_batch_size=args.eval_batch_size,
            judge_worker_num=args.judge_worker_num,
        )

        # neocoder is very large; default to a capped run unless --full.
        neo_limit = None if args.full else args.neocoder_limit
        results[model_name]["neocoder"] = _run_one(
            entry=entry,
            dataset="neocoder",
            work_dir=str(out_dir / model_name / "neocoder"),
            limit=neo_limit,
            seed=args.seed,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            eval_batch_size=args.eval_batch_size,
            judge_worker_num=args.judge_worker_num,
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
