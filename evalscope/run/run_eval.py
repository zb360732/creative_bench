#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _load_models(models_json_path: Path) -> list[dict]:
    try:
        data = json.loads(models_json_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"models.json not found: {models_json_path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON in {models_json_path}: {e}")

    models = data.get("models")
    if not isinstance(models, list) or not models:
        raise SystemExit(f"`models` must be a non-empty list in {models_json_path}")
    return models


def _no_proxy_env() -> dict:
    env = dict(os.environ)
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(key, None)
    env.setdefault("NO_PROXY", "*")
    env.setdefault("no_proxy", "*")
    return env


def _list_models(models: list[dict]) -> int:
    for model in models:
        name = model.get("name", "")
        model_id = model.get("model", "")
        api_url = model.get("api_url", "")
        is_embedding = bool(model.get("is_embedding"))
        suffix = " (embedding)" if is_embedding else ""
        print(f"- {name}{suffix}: model={model_id} api_url={api_url}")
    return 0


def _find_model(models: list[dict], name: str) -> dict:
    for model in models:
        if model.get("name") == name:
            return model
    available = ", ".join([m.get("name", "<missing-name>") for m in models])
    raise SystemExit(f"Model not found: {name}. Available: {available}")


def _run_eval(model: dict, datasets: list[str], limit: int | None, work_dir: str | None, extra: list[str]) -> int:
    cmd = [
        "evalscope",
        "eval",
        "--model",
        str(model.get("model", "")),
        "--model-id",
        str(model.get("name", "")),
        "--datasets",
        *datasets,
    ]

    api_url = model.get("api_url")
    api_key = model.get("api_key")
    if api_url:
        cmd += ["--api-url", str(api_url)]
    if api_key:
        cmd += ["--api-key", str(api_key)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if work_dir:
        cmd += ["--work-dir", str(work_dir)]

    cmd += extra

    print("Running:", " ".join([subprocess.list2cmdline([c]) if " " in c else c for c in cmd]))
    return subprocess.call(cmd, env=_no_proxy_env())


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run evalscope eval using evalscope/run/models.json")
    parser.add_argument(
        "--models-json",
        default=str(Path(__file__).with_name("models.json")),
        help="Path to models.json (default: evalscope/run/models.json)",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List models from models.json")

    run = sub.add_parser("eval", help="Run evaluation for one model")
    run.add_argument("--model-name", required=True, help="Model name (models.json: models[].name)")
    run.add_argument("--datasets", nargs="+", required=True, help="Dataset ids (e.g. aut dat cs4)")
    run.add_argument("--limit", type=int, default=5, help="Max samples per subset")
    run.add_argument("--work-dir", default=None, help="Work dir for outputs (optional)")
    run.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed to `evalscope eval`")

    args = parser.parse_args(argv)
    models_json_path = Path(args.models_json).resolve()
    models = _load_models(models_json_path)

    if args.cmd == "list":
        return _list_models(models)

    if args.cmd == "eval":
        model = _find_model(models, args.model_name)
        if model.get("is_embedding"):
            raise SystemExit(f"Refusing to run eval for embedding model: {args.model_name}")
        extra = list(args.extra)
        if extra and extra[0] == "--":
            extra = extra[1:]
        return _run_eval(model, args.datasets, args.limit, args.work_dir, extra)

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
