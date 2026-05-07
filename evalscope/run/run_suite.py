#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def load_suites(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    suites = data.get("suites", {})
    if not isinstance(suites, dict):
        raise ValueError("`suites` must be a dict in task_suites.json")
    return suites


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a named task suite from task_suites.json")
    parser.add_argument("--suite", required=True, help="Suite name, e.g. combination/exploration/transformation/fulltask")
    parser.add_argument("--models-json", default="run/models.json")
    parser.add_argument("--models", default="")
    parser.add_argument("--limit", default="none")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--log-file", default="")
    parser.add_argument("--suite-config", default="run/task_suites.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    suites = load_suites((repo_root / args.suite_config).resolve())
    if args.suite not in suites:
        raise ValueError(f"Unknown suite: {args.suite}. Available: {', '.join(sorted(suites))}")

    cfg = suites[args.suite]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{cfg['run_name_prefix']}_{timestamp}"
    work_dir = str((repo_root / cfg["work_dir"]).resolve())
    run_dir = Path(work_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.log_file or str(run_dir / "run.log")

    cmd = [
        "python",
        str((repo_root / "run" / "run_parallel_eval.py").resolve()),
        "--models-json", args.models_json,
        "--datasets", ",".join(cfg["datasets"]),
        "--limit", args.limit,
        "--max-tokens", str(cfg.get("max_tokens", 4096)),
        "--temperature", str(cfg.get("temperature", 0.0)),
        "--eval-batch-size", str(cfg.get("eval_batch_size", 1)),
        "--judge-worker-num", str(cfg.get("judge_worker_num", 1)),
        "--max-parallel", str(cfg.get("max_parallel", 4)),
        "--work-dir", work_dir,
        "--run-name", run_name,
        "--log-file", log_file,
    ]
    if args.models:
        cmd.extend(["--models", args.models])
    dataset_args = cfg.get("dataset_args") or {}
    if dataset_args:
        cmd.extend(["--dataset-args", json.dumps(dataset_args, ensure_ascii=False)])

    print("[suite]", args.suite)
    print("[run_name]", run_name)
    print("[work_dir]", work_dir)
    print("[cmd]", " ".join(cmd))
    return subprocess.call(cmd, cwd=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
