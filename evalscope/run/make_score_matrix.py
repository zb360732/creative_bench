#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _parse_csv(value: str) -> List[str]:
    items = [v.strip() for v in value.split(",")]
    return [v for v in items if v]


def _load_records(summary_path: Path) -> List[Dict[str, Any]]:
    if summary_path.suffix.lower() == ".json":
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        if isinstance(records, list):
            return [r for r in records if isinstance(r, dict)]
        return []
    records: List[Dict[str, Any]] = []
    with summary_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records


def _normalize_run_dir(run_dir: str, repo_root: Path) -> List[str]:
    if not run_dir:
        return []
    path = Path(run_dir).expanduser().resolve()
    candidates = {str(path)}
    try:
        candidates.add(str(path.relative_to(repo_root)))
    except ValueError:
        pass
    candidates.add(run_dir)
    return list(candidates)


def _filter_records(records: List[Dict[str, Any]], run_dir: str, repo_root: Path) -> List[Dict[str, Any]]:
    if not run_dir:
        return records
    candidates = _normalize_run_dir(run_dir, repo_root)
    filtered = []
    for record in records:
        record_run = str(record.get("run_dir", ""))
        if record_run in candidates:
            filtered.append(record)
            continue
        for cand in candidates:
            if cand and record_run.endswith(cand):
                filtered.append(record)
                break
    return filtered


def _dataset_order(records: List[Dict[str, Any]]) -> List[str]:
    order: List[str] = []
    seen = set()
    for record in records:
        dataset = str(record.get("dataset", "")).strip()
        if dataset and dataset not in seen:
            seen.add(dataset)
            order.append(dataset)
    return order


def _model_order(records: List[Dict[str, Any]]) -> List[str]:
    order: List[str] = []
    seen = set()
    for record in records:
        model = str(record.get("model", "")).strip()
        if model and model not in seen:
            seen.add(model)
            order.append(model)
    return order


def _score_value(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    if value is None:
        return None
    text = str(value)
    try:
        return float(text)
    except ValueError:
        return text


def _is_transformation_dataset(name: str) -> bool:
    return str(name).strip() in {"transformation", "problem_method"}


def _metrics_to_map(metrics: Any) -> Dict[str, Any]:
    if isinstance(metrics, str):
        try:
            metrics = json.loads(metrics)
        except Exception:
            metrics = []
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


def _collect_aut_metric_ranges(records: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Tuple[float, float]]]:
    """Collect AUT metric values by model and compute per-metric min/max."""
    aut_metrics_by_model: Dict[str, Dict[str, Any]] = {}
    metric_ranges: Dict[str, Tuple[float, float]] = {}

    for record in records:
        if str(record.get("dataset", "")).strip() != "aut":
            continue
        model = str(record.get("model", "")).strip()
        if not model:
            continue
        metric_map = _metrics_to_map(record.get("metrics", []))
        aut_metrics_by_model[model] = metric_map

    # gather ranges
    all_metric_names = set()
    for metric_map in aut_metrics_by_model.values():
        all_metric_names.update(metric_map.keys())

    for name in all_metric_names:
        values: List[float] = []
        for metric_map in aut_metrics_by_model.values():
            val = _score_value(metric_map.get(name))
            if isinstance(val, (int, float)):
                values.append(float(val))
        if not values:
            continue
        metric_ranges[name] = (min(values), max(values))

    return aut_metrics_by_model, metric_ranges


def _collect_transformation_metric_ranges(
    records: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Tuple[float, float]]]:
    """Collect transformation/problem_method metric values by model and compute per-metric min/max."""
    metrics_by_model: Dict[str, Dict[str, Any]] = {}
    metric_ranges: Dict[str, Tuple[float, float]] = {}

    for record in records:
        if not _is_transformation_dataset(str(record.get("dataset", "")).strip()):
            continue
        model = str(record.get("model", "")).strip()
        if not model:
            continue
        metric_map = _metrics_to_map(record.get("metrics", []))
        metrics_by_model[model] = metric_map

    target_metrics = ["fluency", "novelty", "appropriateness", "flexibility", "feasibility"]
    for name in target_metrics:
        values: List[float] = []
        for metric_map in metrics_by_model.values():
            val = _score_value(metric_map.get(name))
            if isinstance(val, (int, float)):
                values.append(float(val))
        if not values:
            continue
        metric_ranges[name] = (min(values), max(values))

    return metrics_by_model, metric_ranges


def _aut_normalized_total(
    aut_metrics_by_model: Dict[str, Dict[str, Any]],
    metric_ranges: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    """Compute normalized AUT total score per model."""
    # Increase contribution of originality-related metrics.
    weight_map: Dict[str, float] = {}
    for name in metric_ranges:
        if "originality" in name:
            weight_map[name] = 2.0  # amplify originality impact
        else:
            weight_map[name] = 1.0

    total: Dict[str, float] = {}
    for model, metric_map in aut_metrics_by_model.items():
        norm_values: List[float] = []
        weights: List[float] = []
        for name, (vmin, vmax) in metric_ranges.items():
            raw = _score_value(metric_map.get(name))
            if not isinstance(raw, (int, float)):
                continue
            if "originality" in name:
                # originality: keep raw score
                norm = float(raw)
            elif vmax == vmin:
                norm = 1.0  # identical scores; treat as fully normalized
            else:
                norm = (float(raw) - vmin) / (vmax - vmin)
            norm_values.append(norm)
            weights.append(weight_map.get(name, 1.0))
        if norm_values and weights:
            weighted = sum(v * w for v, w in zip(norm_values, weights))
            total_weight = sum(weights)
            total[model] = weighted / total_weight if total_weight else 0.0
    return total


def _transformation_normalized_total(
    metrics_by_model: Dict[str, Dict[str, Any]],
    metric_ranges: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    """Compute normalized transformation total score per model."""
    total: Dict[str, float] = {}
    for model, metric_map in metrics_by_model.items():
        norm_values: List[float] = []
        for name, (vmin, vmax) in metric_ranges.items():
            raw = _score_value(metric_map.get(name))
            if not isinstance(raw, (int, float)):
                continue
            if vmax == vmin:
                norm = 1.0
            else:
                norm = (float(raw) - vmin) / (vmax - vmin)
            norm_values.append(norm)
        if norm_values:
            total[model] = sum(norm_values) / len(norm_values)
    return total


def _collect_cs4_total(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute CS4 total score as the mean of four selected metrics."""
    cs4_total: Dict[str, float] = {}
    target_metrics = ["fluency", "novelty", "flexibility", "appropriateness"]

    for record in records:
        if str(record.get("dataset", "")).strip() != "cs4":
            continue
        model = str(record.get("model", "")).strip()
        if not model:
            continue
        metrics_map = _metrics_to_map(record.get("metrics", []))
        values: List[float] = []
        for name in target_metrics:
            val = _score_value(metrics_map.get(name))
            if isinstance(val, (int, float)):
                values.append(float(val))
        if values:
            cs4_total[model] = round(sum(values) / len(values), 4)

    return cs4_total


def _collect_neocoder_total(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute NeoCoder total score as the mean of three selected metrics."""
    neocoder_total: Dict[str, float] = {}
    target_metrics = [
        "mean_correctness",
        "mean_follow_constraints",
        "mean_new_techniques_ratio",
    ]

    for record in records:
        if str(record.get("dataset", "")).strip() != "neocoder":
            continue
        model = str(record.get("model", "")).strip()
        if not model:
            continue
        metrics_map = _metrics_to_map(record.get("metrics", []))
        values: List[float] = []
        for name in target_metrics:
            val = _score_value(metrics_map.get(name))
            if isinstance(val, (int, float)):
                values.append(float(val))
        if values:
            neocoder_total[model] = round(sum(values) / len(values), 4)

    return neocoder_total


def _collect_creative_math_total(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute CreativeMath total score as the mean of three selected ratios."""
    creative_math_total: Dict[str, float] = {}
    target_metrics = [
        "correctness_ratio",
        "novelty_ratio",
        "novel_unknown_ratio",
    ]

    for record in records:
        if str(record.get("dataset", "")).strip() != "creative_math":
            continue
        model = str(record.get("model", "")).strip()
        if not model:
            continue
        metrics_map = _metrics_to_map(record.get("metrics", []))
        values: List[float] = []
        for name in target_metrics:
            val = _score_value(metrics_map.get(name))
            if isinstance(val, (int, float)):
                values.append(float(val))
        if values:
            creative_math_total[model] = round(sum(values) / len(values), 4)

    return creative_math_total


def _aut_round_columns(metrics_map: Dict[str, Any]) -> List[str]:
    rounds = []
    for name in metrics_map:
        if not isinstance(name, str):
            continue
        if name.startswith("mean_aut_fluency_r"):
            suffix = name.split("mean_aut_fluency_r", 1)[-1]
            if suffix.isdigit():
                rounds.append(int(suffix))
    return [f"aut_r{idx}" for idx in sorted(set(rounds))]


def _transformation_metric_columns(metrics_map: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Map transformation metric names to column names."""
    result: Dict[str, Any] = {}
    for name, val in metrics_map.items():
        if not isinstance(name, str):
            continue
        if name == "feasibility":
            result[f"{prefix}_feasibility"] = val
        elif name == "fluency":
            result[f"{prefix}_fluency"] = val
        elif name == "novelty":
            result[f"{prefix}_novelty"] = val
        elif name == "appropriateness":
            result[f"{prefix}_appropriateness"] = val
        elif name == "flexibility":
            result[f"{prefix}_flexibility"] = val
    return result


def _aut_metric_columns(metrics_map: Dict[str, Any]) -> Dict[str, Any]:
    """Map AUT metric names to column names."""
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


def _write_matrix_csv(
    out_path: Path, model_order: List[str], dataset_order: List[str], matrix: Dict[str, Dict[str, Any]]
) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + dataset_order)
        for model in model_order:
            row = [model]
            dataset_scores = matrix.get(model, {})
            for dataset in dataset_order:
                row.append(dataset_scores.get(dataset, ""))
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate model x dataset score matrix.")
    parser.add_argument(
        "--summary",
        default="",
        help="Path to scores_summary.json or scores_summary.csv.",
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help="Run directory to filter records and to locate summary if --summary is empty.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory for matrix files. Defaults to run dir when provided.",
    )
    parser.add_argument(
        "--out-name",
        default="score_matrix",
        help="Base name for output files (default: score_matrix).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    summary_path = Path(args.summary).expanduser().resolve() if args.summary else None
    if summary_path and not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")

    if not summary_path:
        if not args.run_dir:
            raise ValueError("Provide --summary or --run-dir to locate scores_summary.")
        run_dir = Path(args.run_dir).expanduser().resolve()
        candidate_json = run_dir / "scores_summary.json"
        candidate_csv = run_dir / "scores_summary.csv"
        if candidate_json.exists():
            summary_path = candidate_json
        elif candidate_csv.exists():
            summary_path = candidate_csv
        else:
            raise FileNotFoundError("scores_summary.json or scores_summary.csv not found in run dir.")

    records = _load_records(summary_path)
    records = _filter_records(records, args.run_dir, repo_root)

    aut_metrics_by_model, aut_metric_ranges = _collect_aut_metric_ranges(records)
    aut_total_map = _aut_normalized_total(aut_metrics_by_model, aut_metric_ranges)
    transformation_metrics_by_model, transformation_metric_ranges = _collect_transformation_metric_ranges(records)
    transformation_total_map = _transformation_normalized_total(
        transformation_metrics_by_model, transformation_metric_ranges
    )
    cs4_total_map = _collect_cs4_total(records)
    neocoder_total_map = _collect_neocoder_total(records)
    creative_math_total_map = _collect_creative_math_total(records)

    dataset_order: List[str] = []
    dataset_seen = set()
    model_order = _model_order(records)

    for record in records:
        dataset = str(record.get("dataset", "")).strip()
        if dataset and dataset not in dataset_seen:
            dataset_seen.add(dataset)
            dataset_order.append(dataset)
            if dataset == "creative_math" and creative_math_total_map and "creative_math_total" not in dataset_seen:
                dataset_seen.add("creative_math_total")
                dataset_order.append("creative_math_total")
            if dataset == "aut" and aut_total_map and "aut_total" not in dataset_seen:
                dataset_seen.add("aut_total")
                dataset_order.append("aut_total")
            if dataset == "cs4" and cs4_total_map and "cs4_total" not in dataset_seen:
                dataset_seen.add("cs4_total")
                dataset_order.append("cs4_total")
            if dataset == "neocoder" and neocoder_total_map and "neocoder_total" not in dataset_seen:
                dataset_seen.add("neocoder_total")
                dataset_order.append("neocoder_total")
            if _is_transformation_dataset(dataset) and transformation_total_map:
                total_col = f"{dataset}_total"
                if total_col not in dataset_seen:
                    dataset_seen.add(total_col)
                    dataset_order.append(total_col)
        if dataset == "aut":
            metrics_map = _metrics_to_map(record.get("metrics", []))
            # add fine-grained AUT columns
            for col in _aut_metric_columns(metrics_map).keys():
                if col not in dataset_seen:
                    dataset_seen.add(col)
                    dataset_order.append(col)
            for col in _aut_round_columns(metrics_map):
                if col not in dataset_seen:
                    dataset_seen.add(col)
                    dataset_order.append(col)
        if _is_transformation_dataset(dataset):
            metrics_map = _metrics_to_map(record.get("metrics", []))
            for col in _transformation_metric_columns(metrics_map, dataset).keys():
                if col not in dataset_seen:
                    dataset_seen.add(col)
                    dataset_order.append(col)

    matrix: Dict[str, Dict[str, Any]] = {}
    for record in records:
        model = str(record.get("model", "")).strip()
        dataset = str(record.get("dataset", "")).strip()
        if not model or not dataset:
            continue
        # For transformation/problem_method we want the dataset column to reflect the blended total.
        if _is_transformation_dataset(dataset) and transformation_total_map:
            matrix.setdefault(model, {})[dataset] = _score_value(
                transformation_total_map.get(model, record.get("score"))
            )
        elif dataset == "creative_math" and creative_math_total_map:
            matrix.setdefault(model, {})[dataset] = _score_value(creative_math_total_map.get(model, record.get("score")))
        elif dataset == "cs4" and cs4_total_map:
            matrix.setdefault(model, {})[dataset] = _score_value(cs4_total_map.get(model, record.get("score")))
        elif dataset == "neocoder" and neocoder_total_map:
            matrix.setdefault(model, {})[dataset] = _score_value(neocoder_total_map.get(model, record.get("score")))
        else:
            matrix.setdefault(model, {})[dataset] = _score_value(record.get("score"))
        if dataset == "creative_math" and creative_math_total_map:
            matrix.setdefault(model, {})["creative_math_total"] = _score_value(creative_math_total_map.get(model, ""))
        if dataset == "cs4" and cs4_total_map:
            matrix.setdefault(model, {})["cs4_total"] = _score_value(cs4_total_map.get(model, ""))
        if dataset == "neocoder" and neocoder_total_map:
            matrix.setdefault(model, {})["neocoder_total"] = _score_value(neocoder_total_map.get(model, ""))
        if dataset == "aut":
            metrics_map = _metrics_to_map(record.get("metrics", []))
            metric_cols = _aut_metric_columns(metrics_map)
            for col, val in metric_cols.items():
                matrix.setdefault(model, {})[col] = _score_value(val)
            for col in _aut_round_columns(metrics_map):
                round_idx = col.split("aut_r", 1)[-1]
                metric_key = f"mean_aut_fluency_r{round_idx}"
                matrix.setdefault(model, {})[col] = _score_value(metrics_map.get(metric_key, ""))
            if aut_total_map:
                matrix.setdefault(model, {})["aut_total"] = _score_value(aut_total_map.get(model, ""))
        if _is_transformation_dataset(dataset):
            metrics_map = _metrics_to_map(record.get("metrics", []))
            metric_cols = _transformation_metric_columns(metrics_map, dataset)
            for col, val in metric_cols.items():
                matrix.setdefault(model, {})[col] = _score_value(val)
            if transformation_total_map:
                matrix.setdefault(model, {})[f"{dataset}_total"] = _score_value(
                    transformation_total_map.get(model, "")
                )

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    elif args.run_dir:
        out_dir = Path(args.run_dir).expanduser().resolve()
    else:
        out_dir = repo_root / "benchmark" / "outputs" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{args.out_name}.json"
    json_payload = {
        "run_dir": args.run_dir,
        "models": model_order,
        "datasets": dataset_order,
        "matrix": matrix,
    }
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / f"{args.out_name}.csv"
    _write_matrix_csv(csv_path, model_order, dataset_order, matrix)

    print(f"[OK] JSON: {json_path}")
    print(f"[OK] CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
