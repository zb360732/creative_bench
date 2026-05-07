#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_csv(value: str) -> List[str]:
    items = [v.strip() for v in value.split(",")]
    return [v for v in items if v]


def _relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _infer_run_dir(report_path: Path) -> Path:
    parts = report_path.parts
    indices = [i for i, part in enumerate(parts) if part == "reports"]
    if not indices:
        return report_path.parent
    idx = indices[-1]
    if idx >= 2:
        return Path(*parts[: idx - 1])
    return report_path.parent


def _collect_reports(search_dirs: List[Path]) -> List[Path]:
    report_files: List[Path] = []
    for base in search_dirs:
        if not base.exists():
            continue
        for reports_dir in base.rglob("reports"):
            if not reports_dir.is_dir():
                continue
            report_files.extend(
                [
                    p for p in reports_dir.rglob("*.json")
                    if p.is_file() and ".ipynb_checkpoints" not in p.parts
                ]
            )
    return report_files


def _load_report(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if "dataset_name" not in data or "model_name" not in data:
        return None
    return data


def _metrics_to_map(metrics: Any) -> Dict[str, Any]:
    if not isinstance(metrics, list):
        return {}
    result: Dict[str, Any] = {}
    for entry in metrics:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name:
            continue
        result[str(name)] = entry.get("score")
    return result


def _aut_round_columns(metrics_map: Dict[str, Any]) -> List[str]:
    rounds: List[int] = []
    for name in metrics_map:
        if not isinstance(name, str):
            continue
        if name.startswith("mean_aut_fluency_r"):
            suffix = name.split("mean_aut_fluency_r", 1)[-1]
            if suffix.isdigit():
                rounds.append(int(suffix))
    return [f"aut_r{idx}" for idx in sorted(set(rounds))]


def _aut_metric_columns(metrics_map: Dict[str, Any]) -> Dict[str, Any]:
    """Map AUT metric names to matrix column names."""
    result: Dict[str, Any] = {}
    for name, val in metrics_map.items():
        if not isinstance(name, str):
            continue
        if name == "mean_aut_fluency":
            result["aut_fluency"] = val
        elif name == "mean_aut_elaboration":
            result["aut_elaboration"] = val
        elif name == "mean_aut_flexibility":
            result["aut_flexibility"] = val
        elif name == "mean_aut_originality":
            result["aut_originality"] = val
        elif name == "mean_aut_applicability":
            result["aut_applicability"] = val
        elif name.startswith("mean_aut_fluency_r"):
            suffix = name.split("mean_aut_fluency_r", 1)[-1]
            if suffix.isdigit():
                result[f"aut_fluency_r{suffix}"] = val
        elif name.startswith("mean_aut_elaboration_r"):
            suffix = name.split("mean_aut_elaboration_r", 1)[-1]
            if suffix.isdigit():
                result[f"aut_elaboration_r{suffix}"] = val
        elif name.startswith("mean_aut_flexibility_r"):
            suffix = name.split("mean_aut_flexibility_r", 1)[-1]
            if suffix.isdigit():
                result[f"aut_flexibility_r{suffix}"] = val
        elif name.startswith("mean_aut_originality_r"):
            suffix = name.split("mean_aut_originality_r", 1)[-1]
            if suffix.isdigit():
                result[f"aut_originality_r{suffix}"] = val
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize completed report scores into JSON and CSV.")
    parser.add_argument(
        "--run-dir",
        default="",
        help="Comma-separated run directories to scan. If empty, scan --scan-root.",
    )
    parser.add_argument(
        "--scan-root",
        default="benchmark/outputs,benchmark/outputs/combination",
        help="Comma-separated roots to scan when --run-dir is empty.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory for summary files. Defaults to run dir when one is provided.",
    )
    parser.add_argument(
        "--out-name",
        default="scores_summary",
        help="Base name for output files (default: scores_summary).",
    )
    parser.add_argument(
        "--matrix-name",
        default="score_matrix",
        help="Base name for matrix files (default: score_matrix).",
    )
    parser.add_argument(
        "--no-matrix",
        action="store_true",
        help="Skip generating the model x dataset matrix outputs.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    run_dirs = _parse_csv(args.run_dir) if args.run_dir else []
    if run_dirs:
        search_dirs = [Path(p).expanduser().resolve() for p in run_dirs]
    else:
        scan_roots = _parse_csv(args.scan_root)
        search_dirs = [Path(p).expanduser().resolve() for p in scan_roots]

    report_files = _collect_reports(search_dirs)
    records: List[Dict[str, Any]] = []

    for report_path in report_files:
        data = _load_report(report_path)
        if not data:
            continue
        run_dir = _infer_run_dir(report_path)
        record = {
            "run_dir": _relpath(run_dir, repo_root),
            "model": data.get("model_name"),
            "dataset": data.get("dataset_name"),
            "dataset_pretty_name": data.get("dataset_pretty_name"),
            "score": data.get("score"),
            "metrics": data.get("metrics", []),
            "report_path": _relpath(report_path, repo_root),
        }
        records.append(record)

    records.sort(
        key=lambda r: (
            str(r.get("run_dir", "")),
            str(r.get("model", "")),
            str(r.get("dataset", "")),
        )
    )

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    elif run_dirs and len(search_dirs) == 1:
        out_dir = search_dirs[0]
    else:
        out_dir = repo_root / "benchmark" / "outputs" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{args.out_name}.json"
    json_path.write_text(
        json.dumps({"records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = out_dir / f"{args.out_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_dir",
                "model",
                "dataset",
                "dataset_pretty_name",
                "score",
                "metrics",
                "report_path",
            ],
        )
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["metrics"] = json.dumps(record.get("metrics", []), ensure_ascii=True)
            writer.writerow(row)

    if not args.no_matrix:
        dataset_order: List[str] = []
        dataset_seen = set()
        model_order: List[str] = []
        model_seen = set()
        matrix: Dict[str, Dict[str, Any]] = {}

        for record in records:
            dataset = str(record.get("dataset", "")).strip()
            model = str(record.get("model", "")).strip()
            if dataset and dataset not in dataset_seen:
                dataset_seen.add(dataset)
                dataset_order.append(dataset)
            metrics_map = _metrics_to_map(record.get("metrics", []))
            if dataset == "aut" and metrics_map:
                for col in _aut_metric_columns(metrics_map):
                    if col not in dataset_seen:
                        dataset_seen.add(col)
                        dataset_order.append(col)
                for col in _aut_round_columns(metrics_map):
                    if col not in dataset_seen:
                        dataset_seen.add(col)
                        dataset_order.append(col)
            if model and model not in model_seen:
                model_seen.add(model)
                model_order.append(model)
            if not model or not dataset:
                continue
            matrix.setdefault(model, {})[dataset] = record.get("score")
            if dataset == "aut" and metrics_map:
                for col, val in _aut_metric_columns(metrics_map).items():
                    matrix.setdefault(model, {})[col] = val
                for col in _aut_round_columns(metrics_map):
                    round_idx = col.split("aut_r", 1)[-1]
                    metric_key = f"mean_aut_fluency_r{round_idx}"
                    matrix.setdefault(model, {})[col] = metrics_map.get(metric_key, "")

        matrix_json_path = out_dir / f"{args.matrix_name}.json"
        matrix_json_path.write_text(
            json.dumps(
                {
                    "run_dirs": sorted({r.get("run_dir") for r in records}),
                    "models": model_order,
                    "datasets": dataset_order,
                    "matrix": matrix,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        matrix_csv_path = out_dir / f"{args.matrix_name}.csv"
        with matrix_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["model"] + dataset_order)
            for model in model_order:
                row = [model]
                for dataset in dataset_order:
                    row.append(matrix.get(model, {}).get(dataset, ""))
                writer.writerow(row)

    print(f"[OK] JSON: {json_path}")
    print(f"[OK] CSV: {csv_path}")
    if not args.no_matrix:
        print(f"[OK] JSON: {out_dir / f'{args.matrix_name}.json'}")
        print(f"[OK] CSV: {out_dir / f'{args.matrix_name}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
