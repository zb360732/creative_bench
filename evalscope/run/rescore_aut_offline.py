#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass(frozen=True)
class SweepConfig:
    flex_threshold_scale: float
    flex_novel_bonus: float
    orig_agg: str
    orig_topk_ratio: float
    orig_outlier_scale: float
    orig_outlier_bonus: float
    orig_model_agg: str
    orig_model_topk_ratio: float

    @property
    def config_id(self) -> str:
        agg_suffix = self.orig_agg
        if self.orig_agg in TOPK_ORIG_AGGS:
            agg_suffix = f"{agg_suffix}_topk{str(self.orig_topk_ratio).replace('.', 'p')}"
        if self.orig_agg in OUTLIER_THRESHOLD_ORIG_AGGS:
            outlier_scale = str(self.orig_outlier_scale).replace(".", "p")
            agg_suffix = f"{agg_suffix}_out{outlier_scale}"
        if self.orig_agg in OUTLIER_BONUS_ORIG_AGGS:
            outlier_bonus = str(self.orig_outlier_bonus).replace(".", "p")
            agg_suffix = f"{agg_suffix}_w{outlier_bonus}"
        model_agg_suffix = self.orig_model_agg
        if self.orig_model_agg == "topk":
            model_agg_suffix = f"topk{str(self.orig_model_topk_ratio).replace('.', 'p')}"
        scale = str(self.flex_threshold_scale).replace(".", "p")
        bonus = str(self.flex_novel_bonus).replace(".", "p")
        return f"thr{scale}_bonus{bonus}_{agg_suffix}_model{model_agg_suffix}"


TOPK_ORIG_AGGS = {
    "topk",
    "topk_norm",
    "tail_gap",
    "tail_gap_norm",
    "hybrid_topk_norm",
    "hybrid_tail_norm",
    "exceed_sum_norm_topk",
}

OUTLIER_THRESHOLD_ORIG_AGGS = {
    "outlier_ratio",
    "hybrid_topk_norm",
    "hybrid_tail_norm",
    "exceed_sum_norm",
    "exceed_sum_norm_topk",
}

OUTLIER_BONUS_ORIG_AGGS = {
    "hybrid_topk_norm",
    "hybrid_tail_norm",
}


def _parse_csv_str(value: str) -> List[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_csv_float(value: str) -> List[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _normalize_use_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _fast_text_deduplicate(uses: Sequence[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for use in uses:
        norm = _normalize_use_text(use)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(str(use).strip())
    return deduped


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.pstdev(values))


def _extract_nested(record: Dict[str, Any], path: Sequence[str], default: Any = None) -> Any:
    current: Any = record
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def _extract_sample_metrics(record: Dict[str, Any]) -> Dict[str, float]:
    values = _extract_nested(record, ["sample_score", "score", "value"], {}) or {}
    result: Dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, (int, float)):
            result[str(key)] = float(value)
    return result


def _extract_prediction_json(record: Dict[str, Any]) -> str:
    extracted = _extract_nested(record, ["sample_score", "score", "extracted_prediction"], "")
    if extracted:
        return str(extracted)
    prediction = _extract_nested(record, ["sample_score", "score", "prediction"], "")
    return str(prediction or "")


def _safe_topk_mean(values: np.ndarray, topk_ratio: float) -> float:
    if values.size == 0:
        return 0.0
    k = max(1, int(math.ceil(values.size * topk_ratio)))
    topk = np.partition(values, values.size - k)[-k:]
    return float(np.mean(topk))


def _aggregate_sample_scores(values: Sequence[float], agg: str, topk_ratio: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float32)
    if agg == "mean":
        return float(np.mean(arr))
    if agg == "p75":
        return float(np.percentile(arr, 75))
    if agg == "p90":
        return float(np.percentile(arr, 90))
    if agg == "topk":
        return _safe_topk_mean(arr, topk_ratio)
    raise ValueError(f"Unsupported sample aggregation: {agg}")


def _compute_flexibility_v2(
    min_distances: np.ndarray,
    nearest_cluster_indices: np.ndarray,
    avg_intra_distance: float,
    config: SweepConfig,
) -> float:
    if min_distances.size == 0:
        return 0.0
    threshold = avg_intra_distance * config.flex_threshold_scale
    in_cluster_mask = min_distances <= threshold
    covered_clusters = len(set(nearest_cluster_indices[in_cluster_mask].tolist()))
    outlier_count = int(np.sum(~in_cluster_mask))
    denom = max(1.0, math.sqrt(float(min_distances.size)))
    return (covered_clusters + config.flex_novel_bonus * outlier_count) / denom


def _compute_originality_v2(
    min_distances: np.ndarray,
    avg_intra_distance: float,
    config: SweepConfig,
) -> float:
    if min_distances.size == 0:
        return 0.0
    intra = max(float(avg_intra_distance), 1e-6)
    normalized_distances = min_distances / intra
    outlier_ratio = float(np.mean(min_distances > intra * config.orig_outlier_scale))

    if config.orig_agg == "mean":
        return float(np.mean(min_distances))
    if config.orig_agg == "mean_norm":
        return float(np.mean(normalized_distances))
    if config.orig_agg == "p90":
        return float(np.percentile(min_distances, 90))
    if config.orig_agg == "p95":
        return float(np.percentile(min_distances, 95))
    if config.orig_agg == "max":
        return float(np.max(min_distances))
    if config.orig_agg == "topk":
        return _safe_topk_mean(min_distances, config.orig_topk_ratio)
    if config.orig_agg == "topk_norm":
        return _safe_topk_mean(normalized_distances, config.orig_topk_ratio)
    if config.orig_agg == "tail_gap":
        return _safe_topk_mean(min_distances, config.orig_topk_ratio) - float(np.mean(min_distances))
    if config.orig_agg == "tail_gap_norm":
        return _safe_topk_mean(normalized_distances, config.orig_topk_ratio) - float(np.mean(normalized_distances))
    if config.orig_agg == "outlier_ratio":
        return outlier_ratio
    if config.orig_agg == "exceed_sum_norm":
        exceedances = normalized_distances[normalized_distances > config.orig_outlier_scale]
        if exceedances.size == 0:
            return 0.0
        return float(np.sum(exceedances - config.orig_outlier_scale))
    if config.orig_agg == "exceed_sum_norm_topk":
        exceedances = normalized_distances[normalized_distances > config.orig_outlier_scale]
        if exceedances.size == 0:
            return 0.0
        return _safe_topk_mean(exceedances - config.orig_outlier_scale, config.orig_topk_ratio)
    if config.orig_agg == "hybrid_topk_norm":
        return _safe_topk_mean(normalized_distances, config.orig_topk_ratio) + (
            config.orig_outlier_bonus * outlier_ratio
        )
    if config.orig_agg == "hybrid_tail_norm":
        return (
            _safe_topk_mean(normalized_distances, config.orig_topk_ratio)
            - float(np.mean(normalized_distances))
            + config.orig_outlier_bonus * outlier_ratio
        )
    raise ValueError(f"Unsupported originality aggregation: {config.orig_agg}")


def _score_gap(scores: Dict[str, float]) -> float:
    if len(scores) < 2:
        return 0.0
    values = sorted(scores.values())
    return float(values[-1] - values[0])


def _zscore_map(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    mean_val = _mean(values)
    std_val = _stdev(values)
    if std_val == 0.0:
        return {model: 0.0 for model in scores}
    return {model: (value - mean_val) / std_val for model, value in scores.items()}


def _compute_aut_total_v2(
    model_rows: Dict[str, Dict[str, float]],
    metric_names: Sequence[str],
) -> Dict[str, float]:
    normalized_maps = {
        metric: _zscore_map({model: row[metric] for model, row in model_rows.items()})
        for metric in metric_names
    }
    totals: Dict[str, float] = {}
    for model in model_rows:
        values = [normalized_maps[metric][model] for metric in metric_names if model in normalized_maps[metric]]
        totals[model] = _mean(values)
    return totals


def _config_grid(args: argparse.Namespace) -> List[SweepConfig]:
    configs: List[SweepConfig] = []
    flex_threshold_scales = _parse_csv_float(args.flex_threshold_scales)
    flex_novel_bonuses = _parse_csv_float(args.flex_novel_bonuses)
    orig_aggs = _parse_csv_str(args.orig_aggs)
    orig_topk_ratios = _parse_csv_float(args.orig_topk_ratios)
    orig_outlier_scales = _parse_csv_float(args.orig_outlier_scales)
    orig_outlier_bonuses = _parse_csv_float(args.orig_outlier_bonuses)
    orig_model_aggs = _parse_csv_str(args.orig_model_aggs)
    orig_model_topk_ratios = _parse_csv_float(args.orig_model_topk_ratios)

    for scale, bonus, agg in product(
        flex_threshold_scales,
        flex_novel_bonuses,
        orig_aggs,
    ):
        ratio_values = orig_topk_ratios if agg in TOPK_ORIG_AGGS else [orig_topk_ratios[0]]
        outlier_scale_values = (
            orig_outlier_scales if agg in OUTLIER_THRESHOLD_ORIG_AGGS else [orig_outlier_scales[0]]
        )
        outlier_bonus_values = (
            orig_outlier_bonuses if agg in OUTLIER_BONUS_ORIG_AGGS else [orig_outlier_bonuses[0]]
        )

        model_agg_values = orig_model_aggs
        for ratio, outlier_scale, outlier_bonus, model_agg in product(
            ratio_values,
            outlier_scale_values,
            outlier_bonus_values,
            model_agg_values,
        ):
            model_topk_values = orig_model_topk_ratios if model_agg == "topk" else [orig_model_topk_ratios[0]]
            for model_topk_ratio in model_topk_values:
                configs.append(
                    SweepConfig(
                        flex_threshold_scale=scale,
                        flex_novel_bonus=bonus,
                        orig_agg=agg,
                        orig_topk_ratio=ratio,
                        orig_outlier_scale=outlier_scale,
                        orig_outlier_bonus=outlier_bonus,
                        orig_model_agg=model_agg,
                        orig_model_topk_ratio=model_topk_ratio,
                    )
                )
    return configs


def _collect_model_review_files(run_dir: Path, models: Iterable[str]) -> Dict[str, Path]:
    wanted = {model.strip() for model in models if model.strip()}
    result: Dict[str, Path] = {}
    for path in sorted(run_dir.glob("*/reviews/*/aut_default.jsonl")):
        model = path.parent.name
        if wanted and model not in wanted:
            continue
        result[model] = path
    if not result:
        raise FileNotFoundError(f"No AUT review files found under {run_dir}")
    return result


def _build_sample_cache(metric: Any, review_path: Path, use_semantic_dedup: bool) -> List[Dict[str, Any]]:
    sample_cache: List[Dict[str, Any]] = []
    review_records = _load_jsonl(review_path)
    for record in review_records:
        item = str(record.get("target") or _extract_nested(record, ["sample_metadata", "item"], "")).strip().lower()
        if not item:
            continue

        extracted_prediction = _extract_prediction_json(record)
        uses = metric._parse_json_response(extracted_prediction)
        uses = metric._semantic_deduplicate(uses) if use_semantic_dedup else _fast_text_deduplicate(uses)
        if not uses:
            sample_cache.append(
                {
                    "item": item,
                    "baseline": _extract_sample_metrics(record),
                    "min_distances": np.array([], dtype=np.float32),
                    "nearest_cluster_indices": np.array([], dtype=np.int64),
                    "avg_intra_distance": 1.0,
                    "use_count": 0,
                }
            )
            continue

        pool = metric.clustering_pools.get(item)
        if not pool:
            continue

        encode_device = metric._ensure_bert_ready()
        use_embeddings = metric._safe_encode(uses, device=encode_device)
        cluster_centers = pool["cluster_centers"]
        distances = np.linalg.norm(
            use_embeddings[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :],
            axis=2,
        )
        min_distances = np.min(distances, axis=1)
        nearest_cluster_indices = np.argmin(distances, axis=1)

        sample_cache.append(
            {
                "item": item,
                "baseline": _extract_sample_metrics(record),
                "min_distances": min_distances,
                "nearest_cluster_indices": nearest_cluster_indices,
                "avg_intra_distance": float(pool["avg_intra_distance"]),
                "use_count": len(uses),
            }
        )
    return sample_cache


def _summarize_model(
    sample_cache: List[Dict[str, Any]],
    config: SweepConfig,
) -> Dict[str, float]:
    baseline_names = [
        "aut_fluency",
        "aut_elaboration",
        "aut_flexibility",
        "aut_originality",
        "aut_applicability",
    ]
    round_names = [
        "aut_fluency_r1",
        "aut_fluency_r2",
        "aut_fluency_r3",
        "aut_fluency_r4",
        "aut_fluency_r5",
    ]

    row: Dict[str, float] = {}
    for name in baseline_names + round_names:
        row[name] = _mean(
            [
                sample.get("baseline", {}).get(name, 0.0)
                for sample in sample_cache
                if name in sample.get("baseline", {})
            ]
        )

    flex_v2_values: List[float] = []
    orig_v2_values: List[float] = []
    for sample in sample_cache:
        min_distances = sample["min_distances"]
        nearest_cluster_indices = sample["nearest_cluster_indices"]
        avg_intra_distance = sample["avg_intra_distance"]
        flex_v2_values.append(
            _compute_flexibility_v2(min_distances, nearest_cluster_indices, avg_intra_distance, config)
        )
        orig_v2_values.append(_compute_originality_v2(min_distances, avg_intra_distance, config))

    row["aut_flexibility_v2"] = _mean(flex_v2_values)
    row["aut_originality_v2"] = _aggregate_sample_scores(
        orig_v2_values,
        config.orig_model_agg,
        config.orig_model_topk_ratio,
    )
    row["aut"] = row.get("aut_fluency", 0.0)
    row["aut_r1"] = row.get("aut_fluency_r1", 0.0)
    row["aut_r2"] = row.get("aut_fluency_r2", 0.0)
    row["aut_r3"] = row.get("aut_fluency_r3", 0.0)
    row["aut_r4"] = row.get("aut_fluency_r4", 0.0)
    row["aut_r5"] = row.get("aut_fluency_r5", 0.0)
    return row


def _load_existing_task_scores(run_dir: Path, models: Iterable[str]) -> Dict[str, Dict[str, float]]:
    summary_path = run_dir / "scores_summary.json"
    existing: Dict[str, Dict[str, float]] = {}
    if not summary_path.exists():
        return existing
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    for record in payload.get("records", []):
        model = str(record.get("model", "")).strip()
        dataset = str(record.get("dataset", "")).strip()
        score = record.get("score")
        if model and dataset and isinstance(score, (int, float)):
            existing.setdefault(model, {})[dataset] = float(score)
    if models:
        return {model: existing.get(model, {}) for model in models}
    return existing


def _select_best_config(
    model_rows_by_config: Dict[str, Dict[str, Dict[str, float]]],
    configs: Sequence[SweepConfig],
) -> SweepConfig:
    best_config = configs[0]
    best_score = float("-inf")
    for config in configs:
        rows = model_rows_by_config[config.config_id]
        flex_scores = {model: row["aut_flexibility_v2"] for model, row in rows.items()}
        orig_scores = {model: row["aut_originality_v2"] for model, row in rows.items()}
        flex_gap = _score_gap(flex_scores)
        orig_gap = _score_gap(orig_scores)
        flex_std = _stdev(list(flex_scores.values()))
        orig_std = _stdev(list(orig_scores.values()))
        score = flex_gap + orig_gap + flex_std + orig_std
        if score > best_score:
            best_score = score
            best_config = config
    return best_config


def _combined_sweep_score(
    flex_gap: float,
    orig_gap: float,
    flex_std: float,
    orig_std: float,
) -> float:
    return flex_gap + orig_gap + flex_std + orig_std


def _write_matrix_csv(path: Path, rows: Dict[str, Dict[str, float]]) -> None:
    columns = [
        "model",
        "aut",
        "aut_total_v2",
        "aut_fluency",
        "aut_elaboration",
        "aut_flexibility",
        "aut_flexibility_v2",
        "aut_originality",
        "aut_originality_v2",
        "aut_applicability",
        "aut_r1",
        "aut_r2",
        "aut_r3",
        "aut_r4",
        "aut_r5",
        "bats",
        "dat",
        "metaphor",
        "rat",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for model in sorted(rows):
            row = rows[model]
            writer.writerow([model] + [row.get(col, "") for col in columns[1:]])


def _write_sweep_csv(path: Path, configs: Sequence[SweepConfig], model_rows_by_config: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "config_id",
                "flex_threshold_scale",
                "flex_novel_bonus",
                "orig_agg",
                "orig_topk_ratio",
                "orig_outlier_scale",
                "orig_outlier_bonus",
                "orig_model_agg",
                "orig_model_topk_ratio",
                "flex_gap",
                "orig_gap",
                "flex_std",
                "orig_std",
                "combined_score",
            ]
        )
        for config in configs:
            rows = model_rows_by_config[config.config_id]
            flex_scores = {model: row["aut_flexibility_v2"] for model, row in rows.items()}
            orig_scores = {model: row["aut_originality_v2"] for model, row in rows.items()}
            flex_gap = _score_gap(flex_scores)
            orig_gap = _score_gap(orig_scores)
            flex_std = _stdev(list(flex_scores.values()))
            orig_std = _stdev(list(orig_scores.values()))
            writer.writerow(
                [
                    config.config_id,
                    config.flex_threshold_scale,
                    config.flex_novel_bonus,
                    config.orig_agg,
                    config.orig_topk_ratio,
                    config.orig_outlier_scale,
                    config.orig_outlier_bonus,
                    config.orig_model_agg,
                    config.orig_model_topk_ratio,
                    flex_gap,
                    orig_gap,
                    flex_std,
                    orig_std,
                    _combined_sweep_score(flex_gap, orig_gap, flex_std, orig_std),
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline AUT rescoring and parameter sweep using existing review outputs."
    )
    parser.add_argument(
        "--run-dir",
        default="benchmark/outputs/exploration/aut_creative_math_drivel_neocoder_full",
        help="Existing run directory containing predictions/reviews/reports.",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model names to include. Default: all models found under run-dir.",
    )
    parser.add_argument(
        "--fluency-threshold",
        type=float,
        default=0.6,
        help="Semantic dedup threshold reused from AUT fluency.",
    )
    parser.add_argument(
        "--flex-threshold-scales",
        default="1.0,1.2,1.5,2.0",
        help="Comma-separated threshold multipliers over avg_intra_distance for flexibility_v2 sweep.",
    )
    parser.add_argument(
        "--flex-novel-bonuses",
        default="1.0,1.5,2.0,3.0",
        help="Comma-separated outlier bonuses for flexibility_v2 sweep.",
    )
    parser.add_argument(
        "--orig-aggs",
        default="mean,mean_norm,topk,topk_norm,p90,p95,max,tail_gap_norm,hybrid_topk_norm,exceed_sum_norm",
        help=(
            "Comma-separated originality_v2 aggregations to sweep: "
            "mean, mean_norm, topk, topk_norm, p90, p95, max, tail_gap, tail_gap_norm, "
            "outlier_ratio, hybrid_topk_norm, hybrid_tail_norm, exceed_sum_norm, exceed_sum_norm_topk."
        ),
    )
    parser.add_argument(
        "--orig-topk-ratios",
        default="0.05,0.1,0.2,0.3,0.4",
        help="Comma-separated top-k ratios used when originality_v2 aggregation is topk.",
    )
    parser.add_argument(
        "--orig-outlier-scales",
        default="1.0,1.5,2.0",
        help="Comma-separated distance thresholds over avg_intra_distance used for outlier-based originality_v2 aggregations.",
    )
    parser.add_argument(
        "--orig-outlier-bonuses",
        default="0.5,1.0,2.0,3.0",
        help="Comma-separated weights applied to outlier ratio in hybrid originality_v2 aggregations.",
    )
    parser.add_argument(
        "--orig-model-aggs",
        default="mean",
        help="Comma-separated model-level aggregations over sample originality_v2 values: mean, p75, p90, topk.",
    )
    parser.add_argument(
        "--orig-model-topk-ratios",
        default="0.2",
        help="Comma-separated top-k ratios used when model-level originality_v2 aggregation is topk.",
    )
    parser.add_argument(
        "--out-prefix",
        default="aut_metric_retune",
        help="Prefix for generated output files under run-dir.",
    )
    parser.add_argument(
        "--semantic-dedup",
        action="store_true",
        help="Use embedding-based semantic dedup before rescoring. Slower but closer to the original fluency pipeline.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    from evalscope.metrics.aut_metrics import AUTFluency

    models = _parse_csv_str(args.models)
    review_files = _collect_model_review_files(run_dir, models)
    models = list(review_files.keys())

    AUTFluency._instance = None
    metric = AUTFluency(fluency_similarity_threshold=args.fluency_threshold)

    print(f"[INFO] Loading AUT review data for {len(models)} models from {run_dir}", flush=True)
    sample_cache_by_model = {
        model: _build_sample_cache(metric, review_path, args.semantic_dedup)
        for model, review_path in review_files.items()
    }

    configs = _config_grid(args)
    if not configs:
        raise ValueError("No sweep configs generated.")

    print(f"[INFO] Evaluating {len(configs)} AUT rescoring configs", flush=True)
    existing_task_scores = _load_existing_task_scores(run_dir, models)

    model_rows_by_config: Dict[str, Dict[str, Dict[str, float]]] = {}
    for config in configs:
        model_rows: Dict[str, Dict[str, float]] = {}
        for model, sample_cache in sample_cache_by_model.items():
            row = _summarize_model(sample_cache, config)
            row.update(existing_task_scores.get(model, {}))
            model_rows[model] = row
        aut_total_v2_map = _compute_aut_total_v2(
            model_rows,
            ["aut_fluency", "aut_elaboration", "aut_flexibility_v2", "aut_originality_v2", "aut_applicability"],
        )
        for model, total in aut_total_v2_map.items():
            model_rows[model]["aut_total_v2"] = total
        model_rows_by_config[config.config_id] = model_rows

    best_config = _select_best_config(model_rows_by_config, configs)
    best_rows = model_rows_by_config[best_config.config_id]

    sweep_csv = run_dir / f"{args.out_prefix}_sweep.csv"
    sweep_json = run_dir / f"{args.out_prefix}_sweep.json"
    matrix_csv = run_dir / f"{args.out_prefix}_matrix.csv"
    matrix_json = run_dir / f"{args.out_prefix}_matrix.json"

    _write_sweep_csv(sweep_csv, configs, model_rows_by_config)
    sweep_json.write_text(
        json.dumps(
            {
                "best_config": best_config.__dict__,
                "configs": [config.__dict__ for config in configs],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_matrix_csv(matrix_csv, best_rows)
    matrix_json.write_text(
        json.dumps(
            {
                "best_config": best_config.__dict__,
                "models": sorted(best_rows.keys()),
                "rows": best_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[OK] Sweep CSV: {sweep_csv}", flush=True)
    print(f"[OK] Sweep JSON: {sweep_json}", flush=True)
    print(f"[OK] Matrix CSV: {matrix_csv}", flush=True)
    print(f"[OK] Matrix JSON: {matrix_json}", flush=True)
    print(f"[OK] Best config: {best_config.config_id} -> {best_config}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
