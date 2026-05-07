"""Dataset-level prompt enhancement utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .executor import build_artifact


def load_records(path: str | Path) -> list[dict[str, Any]]:
    data_path = Path(path)
    if data_path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with data_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    records.append(payload)
        return records
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [item for item in payload["data"] if isinstance(item, dict)]
    raise ValueError(f"Unsupported dataset format: {data_path}")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def enhance_records(task_name: str, records: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    selected = records[:limit] if limit is not None else records
    return [build_artifact(task_name, item) for item in selected]


def enhance_dataset_file(task_name: str, input_path: str | Path, output_path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = enhance_records(task_name, load_records(input_path), limit=limit)
    write_jsonl(output_path, rows)
    return rows
