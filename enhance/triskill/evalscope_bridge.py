"""External bridge files for using TriSkill outputs with evalscope workflows.

This module does not import or modify evalscope.  It converts TriSkill JSONL
artifacts into simple prediction JSON/JSONL files that downstream scripts or
adapters can consume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .analysis import load_jsonl


def artifacts_to_predictions(input_path: str | Path, output_path: str | Path, include_prompts: bool = False) -> list[dict[str, Any]]:
    rows = load_jsonl(input_path)
    predictions: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        item = {
            "id": row.get("item_id") if row.get("item_id") is not None else idx,
            "task_name": row.get("task_name"),
            "method": row.get("method"),
            "prediction": row.get("final_answer") or row.get("enhanced_prompt") or "",
        }
        if include_prompts:
            item["original_prompt"] = row.get("original_prompt", "")
            item["enhanced_prompt"] = row.get("enhanced_prompt", "")
        predictions.append(item)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".jsonl":
        with out.open("w", encoding="utf-8") as handle:
            for item in predictions:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        out.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    return predictions
