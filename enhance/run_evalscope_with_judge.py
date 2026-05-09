#!/usr/bin/env python3
"""Run evalscope's parallel evaluator with an explicit judge config path.

This wrapper keeps evalscope source untouched.  It loads the original
``evalscope/run/run_parallel_eval.py`` module, then redirects the judge-config
lookups used by the runner and benchmark adapters to a user-provided JSON file.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVALSCOPE_DIR = ROOT / "evalscope"
RUN_PARALLEL_PATH = EVALSCOPE_DIR / "run/run_parallel_eval.py"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--judge-config", required=True)
    known, remaining = parser.parse_known_args(argv)
    judge_config = Path(known.judge_config).expanduser().resolve()
    if not judge_config.exists():
        raise FileNotFoundError(f"Judge config not found: {judge_config}")

    sys.path.insert(0, str(EVALSCOPE_DIR))
    module = _load_run_parallel_eval()
    _patch_runner_judge_loader(module, judge_config)
    _patch_adapter_judge_paths(judge_config)
    return int(module.main(remaining))


def _load_run_parallel_eval() -> Any:
    spec = importlib.util.spec_from_file_location("enhance_evalscope_run_parallel_eval", RUN_PARALLEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load evalscope runner: {RUN_PARALLEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_runner_judge_loader(module: Any, judge_config: Path) -> None:
    def _load_default_judge_model_args_list(max_tokens: int) -> list[dict[str, Any]]:
        data = json.loads(judge_config.read_text(encoding="utf-8"))
        models = data.get("models", [])
        if not models:
            raise ValueError(f"No judge models found in {judge_config}")
        return [_judge_model_args(entry, max_tokens=max_tokens) for entry in models]

    def _load_default_judge_model_args(max_tokens: int) -> dict[str, Any]:
        return _load_default_judge_model_args_list(max_tokens=max_tokens)[0]

    module._load_default_judge_model_args_list = _load_default_judge_model_args_list
    module._load_default_judge_model_args = _load_default_judge_model_args


def _judge_model_args(entry: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    api_key = str(_resolve_env(entry.get("api_key", "EMPTY")))
    api_key_env = entry.get("api_key_env")
    if api_key_env:
        api_key = os.getenv(str(api_key_env), api_key)
    if api_key in {"", "YOUR_API_KEY"}:
        api_key = os.getenv("EVALSCOPE_API_KEY", api_key)
    if api_key in {"", "YOUR_API_KEY"}:
        api_key = os.getenv("OPENAI_API_KEY", api_key)
    return {
        "api_url": str(_resolve_env(entry.get("api_url"))),
        "api_key": api_key,
        "model_id": str(_resolve_env(entry.get("model"))),
        "generation_config": {"temperature": 0.0, "max_tokens": max_tokens},
    }


def _resolve_env(value: Any) -> Any:
    return os.path.expandvars(value) if isinstance(value, str) else value


def _patch_adapter_judge_paths(judge_config: Path) -> None:
    module_names = [
        "evalscope.benchmarks.aut.aut_adapter",
        "evalscope.benchmarks.creative_math.creative_math_adapter",
        "evalscope.benchmarks.cs4.cs4_adapter",
        "evalscope.benchmarks.transformation.transformation_adapter",
    ]
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(module, "_DEFAULT_JUDGE_CONFIG_PATH"):
            setattr(module, "_DEFAULT_JUDGE_CONFIG_PATH", judge_config)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
