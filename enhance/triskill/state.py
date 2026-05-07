"""Runtime state and audit helpers for TriSkill."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


GOLD_FIELD_BLACKLIST = {
    "answer",
    "answers",
    "reference",
    "references",
    "gold",
    "gold_answer",
    "target",
    "targets",
    "label",
    "labels",
    "candidate_answers",
    "correct_answer",
    "target_words",
    "solution",
    "solutions",
}

VISIBLE_FIELD_ALLOWLIST = {
    "id",
    "item_id",
    "query",
    "question",
    "prompt",
    "input",
    "category",
    "category_name",
    "task_name",
    "word_a",
    "word_b",
    "word_c",
    "direction",
    "relation_type",
    "metaphor_word",
    "novelty",
    "item",
    "constraints",
    "rules",
    "goals",
    "metadata",
}


@dataclass
class ElicitationState:
    task_name: str
    item_id: str | None
    raw_item: Mapping[str, Any]
    original_prompt: str
    level: str
    workflow: str
    raw_metrics: list[str] = field(default_factory=list)
    canonical_metrics: dict[str, str] = field(default_factory=dict)
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    visible_choices: list[str] | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    selected_candidate: Any = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    skill_trace: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str | None = None
    parse_success: bool = False
    warnings: list[str] = field(default_factory=list)


def safe_item_view(item: Mapping[str, Any]) -> dict[str, Any]:
    """Return visible item fields while excluding known scoring-only fields."""

    visible: dict[str, Any] = {}
    for key, value in item.items():
        key_text = str(key)
        normalized = key_text.strip().lower()
        if normalized in GOLD_FIELD_BLACKLIST:
            continue
        if normalized in VISIBLE_FIELD_ALLOWLIST or normalized.startswith(("visible_", "public_")):
            visible[key_text] = value
    return visible


def detect_leakage_fields(item: Mapping[str, Any]) -> list[str]:
    return [str(key) for key in item.keys() if str(key).strip().lower() in GOLD_FIELD_BLACKLIST]


def item_id_from(item: Mapping[str, Any], default: str | None = None) -> str | None:
    for key in ("id", "item_id", "qid", "uid"):
        if key in item:
            return str(item[key])
    return default


def prompt_from_item(item: Mapping[str, Any]) -> str:
    for key in ("query", "prompt", "input", "question"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""
