"""Analysis utilities for TriSkill experiment JSONL files."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .diagnostics import diagnose_artifact


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_runs(paths: list[str | Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(load_jsonl(path))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("task_name", "")), str(row.get("method", "")))].append(row)
    summaries = []
    for (task, method), items in sorted(grouped.items()):
        lengths = [float(item.get("output_length") or 0) for item in items]
        calls = [float(item.get("num_llm_calls") or 0) for item in items]
        parse = [1.0 if item.get("parse_success") else 0.0 for item in items]
        summaries.append({
            "task": task,
            "method": method,
            "num_items": len(items),
            "parse_success_rate": _safe_mean(parse),
            "mean_output_length": _safe_mean(lengths),
            "std_output_length": pstdev(lengths) if len(lengths) > 1 else 0.0,
            "mean_num_llm_calls": _safe_mean(calls),
        })
    return {
        "num_rows": len(rows),
        "by_task_method": summaries,
        "profile_shift_template": profile_shift_template(rows),
        "transformation_diagnostics": transformation_diagnostic_summary(rows),
    }


def profile_shift_template(rows: list[dict[str, Any]]) -> dict[str, Any]:
    levels = defaultdict(lambda: defaultdict(int))
    for row in rows:
        levels[str(row.get("method", ""))][str(row.get("level", ""))] += 1
    return {method: dict(counts) for method, counts in levels.items()}


def transformation_diagnostic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    total = 0
    for row in rows:
        if str(row.get("task_name", "")).lower() != "transformation":
            continue
        total += 1
        diag = row.get("diagnostics") or diagnose_artifact(row)
        for name in diag.get("active_failure_modes", []):
            counts[str(name)] += 1
    return {"num_transformation_rows": total, "active_failure_counts": dict(sorted(counts.items()))}


def write_summary(input_paths: list[str | Path], output_path: str | Path) -> dict[str, Any]:
    summary = summarize_runs(input_paths)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _safe_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(mean(finite)) if finite else 0.0
