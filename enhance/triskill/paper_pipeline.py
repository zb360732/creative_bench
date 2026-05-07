"""Paper-grade experiment utilities for TriSkill.

This module encodes the reproducibility and reporting layer needed for a
benchmark + elicitation-framework paper: experiment manifests, artifact audits,
score joins, method deltas, level/profile summaries, and table-ready JSON.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .analysis import load_jsonl
from .diagnostics import diagnose_artifact
from .state import GOLD_FIELD_BLACKLIST


DEFAULT_TASKS: dict[str, str] = {
    "dat": "evalscope/evalscope/benchmarks/dat/data/dat.json",
    "bats": "evalscope/evalscope/benchmarks/bats/data/bats_sampled.json",
    "rat": "evalscope/evalscope/benchmarks/rat/data/rat.json",
    "metaphor": "evalscope/evalscope/benchmarks/metaphor/data/metaphor.json",
    "aut": "evalscope/custom_eval/text/qa/exploration/aut.json",
    "transformation": "evalscope/dataprocess/transformation/generated/final_runs/transformation_eval_1235_all.json",
}

DEFAULT_METHODS = (
    "direct",
    "generic_creativity_prompt",
    "cot_structured",
    "triskill_full",
    "triskill_without_verifier",
    "triskill_wrong_skill_assignment",
)

REQUIRED_ARTIFACT_FIELDS = {
    "task_name",
    "method",
    "safe_item",
    "original_prompt",
    "enhanced_prompt",
    "parse_success",
    "output_length",
    "num_llm_calls",
    "warnings",
}


def create_experiment_manifest(
    output_path: str | Path,
    tasks: dict[str, str] | None = None,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    output_dir: str = "enhance/runs",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    task_map = tasks or DEFAULT_TASKS
    rows: list[dict[str, Any]] = []
    for task_name, input_path in task_map.items():
        for method in methods:
            stem = f"{task_name}_{method}"
            rows.append({
                "task_name": task_name,
                "method": method,
                "input_path": input_path,
                "artifact_path": f"{output_dir}/{stem}.jsonl",
                "prediction_path": f"{output_dir}/{stem}.predictions.json",
                "requires_llm": method != "triskill_prompt_only",
                "limit": limit,
            })
    _write_jsonl(output_path, rows)
    return rows


def audit_artifacts(path: str | Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    failures: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(row))
        if missing:
            failures.append({"index": idx, "type": "missing_fields", "fields": missing})
        safe_item = row.get("safe_item") or {}
        leaked_safe_fields = sorted(set(str(key).lower() for key in safe_item) & GOLD_FIELD_BLACKLIST)
        if leaked_safe_fields:
            failures.append({"index": idx, "type": "safe_item_gold_leak", "fields": leaked_safe_fields})
        final = str(row.get("final_answer") or "")
        if row.get("method") not in {"triskill_full", "direct", "generic_creativity_prompt", "cot_structured", "triskill_without_verifier", "triskill_wrong_skill_assignment"}:
            continue
        if row.get("method") != "triskill_prompt_only" and not final and row.get("num_llm_calls", 0):
            failures.append({"index": idx, "type": "missing_final_answer"})
    return {
        "path": str(path),
        "num_rows": len(rows),
        "num_failures": len(failures),
        "pass": not failures,
        "failures": failures[:200],
    }


def join_scores(artifact_path: str | Path, score_path: str | Path, output_path: str | Path) -> list[dict[str, Any]]:
    artifacts = load_jsonl(artifact_path)
    scores = _load_scores(score_path)
    joined: list[dict[str, Any]] = []
    for idx, row in enumerate(artifacts):
        key = str(row.get("item_id") if row.get("item_id") is not None else idx)
        score = scores.get(key) or scores.get(str(idx)) or {}
        merged = dict(row)
        merged["scores"] = score
        merged["diagnostics"] = row.get("diagnostics") or diagnose_artifact(row)
        joined.append(merged)
    _write_jsonl(output_path, joined)
    return joined


def summarize_scored(paths: list[str | Path], primary_score: str = "score", baseline_method: str = "direct") -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(load_jsonl(path))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("task_name", "")), str(row.get("method", "")))].append(row)
    task_method = []
    score_means: dict[tuple[str, str], float] = {}
    for (task, method), items in sorted(grouped.items()):
        values = [_extract_score(item, primary_score) for item in items]
        values = [value for value in values if value is not None and math.isfinite(value)]
        lengths = [float(item.get("output_length") or 0) for item in items]
        calls = [float(item.get("num_llm_calls") or 0) for item in items]
        avg = _safe_mean(values)
        score_means[(task, method)] = avg
        task_method.append({
            "task": task,
            "method": method,
            "num_items": len(items),
            "mean_score": avg,
            "std_score": pstdev(values) if len(values) > 1 else 0.0,
            "parse_success_rate": _safe_mean([1.0 if item.get("parse_success") else 0.0 for item in items]),
            "mean_output_length": _safe_mean(lengths),
            "mean_num_llm_calls": _safe_mean(calls),
        })
    deltas = []
    for entry in task_method:
        task = entry["task"]
        base = score_means.get((task, baseline_method))
        if base is None:
            continue
        deltas.append({
            "task": task,
            "method": entry["method"],
            "baseline_method": baseline_method,
            "mean_score": entry["mean_score"],
            "baseline_mean_score": base,
            "delta": entry["mean_score"] - base,
        })
    return {
        "primary_score": primary_score,
        "baseline_method": baseline_method,
        "task_method": task_method,
        "deltas": deltas,
        "level_profile": _level_profile(rows, primary_score),
        "length_budget_controls": _length_budget_controls(task_method),
    }


def write_scored_summary(paths: list[str | Path], output_path: str | Path, primary_score: str = "score", baseline_method: str = "direct") -> dict[str, Any]:
    summary = summarize_scored(paths, primary_score=primary_score, baseline_method=baseline_method)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _load_scores(path: str | Path) -> dict[str, dict[str, Any]]:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        rows = load_jsonl(p)
    else:
        payload = json.loads(p.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("scores", [])
    out = {}
    for idx, row in enumerate(rows):
        if isinstance(row, dict):
            key = str(row.get("id") or row.get("item_id") or idx)
            out[key] = row
    return out


def _extract_score(row: dict[str, Any], primary_score: str) -> float | None:
    scores = row.get("scores") or {}
    if isinstance(scores, dict):
        value = scores.get(primary_score)
        if value is None and isinstance(scores.get("score"), dict):
            value = scores["score"].get(primary_score)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _level_profile(rows: list[dict[str, Any]], primary_score: str) -> dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        value = _extract_score(row, primary_score)
        if value is not None:
            grouped[(str(row.get("method", "")), str(row.get("level", "")))].append(value)
    return {f"{method}:{level}": _safe_mean(values) for (method, level), values in sorted(grouped.items())}


def _length_budget_controls(task_method: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "task": row["task"],
            "method": row["method"],
            "mean_output_length": row["mean_output_length"],
            "mean_num_llm_calls": row["mean_num_llm_calls"],
            "mean_score": row["mean_score"],
        }
        for row in task_method
    ]


def _safe_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(mean(finite)) if finite else 0.0


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
