#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


ROOT = Path("/inspire/hdd/project/ai4education/qianhong-p-qianhong")
EXPLORATION = ROOT / "benchmark/outputs/exploration"

RUNS = {
    "neocoder": {
        "long": [EXPLORATION / "neocoder_newdeploy_long_mt30000_v1"],
        "olmo": [
            EXPLORATION / "neocoder_newdeploy_olmo_mt2048_v2",
            EXPLORATION / "neocoder_newdeploy_olmo_mt4096_v1",
        ],
    },
    "creative_math": {
        "long": [EXPLORATION / "creative_math_newdeploy_long_mt30000_v1"],
        "olmo": [
            EXPLORATION / "creative_math_newdeploy_olmo_mt1024_v3",
            EXPLORATION / "creative_math_newdeploy_olmo_mt2048_v2",
            EXPLORATION / "creative_math_newdeploy_olmo_mt4096_v1",
        ],
    },
    "cs4": {
        "long": [EXPLORATION / "cs4_newdeploy_long_mt30000_v1"],
        "olmo": [
            EXPLORATION / "cs4_newdeploy_olmo_mt2048_v2",
            EXPLORATION / "cs4_newdeploy_olmo_mt4096_v1",
        ],
    },
}

GROUP_NOTE = {
    "long": {"max_tokens": 30000, "judge_max_tokens": 8192},
    "olmo": {"max_tokens": 2048, "judge_max_tokens": 2048},
}

RUN_NOTE = {
    "creative_math_newdeploy_olmo_mt1024_v3": {"max_tokens": 1024, "judge_max_tokens": 2048},
}


def safe_unlink(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def model_has_outputs(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    for sub in ("predictions", "reviews", "reports"):
        if (model_dir / sub).exists():
            return True
    return False


def link_model_dir(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        safe_unlink(dst)
    os.symlink(src, dst, target_is_directory=True)


def summarize_model_dir(model_dir: Path, dataset: str, run_info: dict) -> tuple[str, object]:
    model_name = model_dir.name
    report_path = model_dir / "reports" / model_name / f"{dataset}.json"
    pred_path = model_dir / "predictions" / model_name / f"{dataset}_default.jsonl"
    review_path = model_dir / "reviews" / model_name / f"{dataset}_default.jsonl"

    if report_path.exists():
        try:
            return "ok", json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            return "ok", str(report_path)

    if run_info.get("status") == "error":
        return "error", run_info.get("result")

    detail = {}
    if pred_path.exists():
        try:
            with pred_path.open("r", encoding="utf-8") as f:
                detail["prediction_lines"] = sum(1 for _ in f)
        except Exception:
            pass
    if review_path.exists():
        try:
            with review_path.open("r", encoding="utf-8") as f:
                detail["review_lines"] = sum(1 for _ in f)
        except Exception:
            pass
    if detail:
        return "partial", detail

    return "running", run_info.get("result")


def gather_dataset(dataset: str, run_map: dict[str, list[Path]]) -> dict[str, dict]:
    dataset_dir = EXPLORATION / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    runs_summary: dict[str, dict] = {}

    for group_name, run_dirs in run_map.items():
        for run_dir in run_dirs:
            if not run_dir.exists():
                continue

            run_summary_path = run_dir / "summary.json"
            if run_summary_path.exists():
                try:
                    runs_summary[run_dir.name] = json.loads(run_summary_path.read_text(encoding="utf-8"))
                except Exception:
                    runs_summary[run_dir.name] = {}
            else:
                runs_summary[run_dir.name] = {}

            for child in sorted(run_dir.iterdir()):
                if not child.is_dir():
                    continue
                if child.name in {"reports"}:
                    continue
                if not model_has_outputs(child):
                    continue
                if child.name in summary:
                    continue

                dst = dataset_dir / child.name
                link_model_dir(child, dst)

                run_info = runs_summary[run_dir.name].get(child.name, {})
                status, result = summarize_model_dir(child, dataset, run_info)
                summary[child.name] = {
                    "group": group_name,
                    "source_run": run_dir.name,
                    "source_dir": str(child),
                    "linked_dir": str(dst),
                    "status": status,
                    "result": result,
                    **RUN_NOTE.get(run_dir.name, GROUP_NOTE[group_name]),
                }

    summary_path = dataset_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    all_summary = {}
    for dataset, run_map in RUNS.items():
        all_summary[dataset] = gather_dataset(dataset, run_map)

    overall_path = EXPLORATION / "newdeploy_summary.json"
    overall_path.write_text(json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {overall_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
