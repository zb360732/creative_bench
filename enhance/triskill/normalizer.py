"""Final answer normalization helpers for benchmark adapter compatibility."""

from __future__ import annotations

import json
import re
from typing import Any


PLACEHOLDER_WORDS = {
    "alright",
    "answer_word",
    "are",
    "asking",
    "connecting_word",
    "for",
    "i",
    "need",
    "okay",
    "replacement_word",
    "semantically",
    "so",
    "word",
    "words",
    "that",
    "the",
    "to",
    "target",
    "candidate",
    "solution",
    "answer",
    "up",
    "user",
    "with",
    "semantic_domain_expansion",
    "lexical_validity_check",
    "diversity_filtering",
    "output_normalization",
}

DEFAULT_DIVERSE_WORDS = [
    "nebula",
    "cucumber",
    "justice",
    "violin",
    "thunder",
    "invoice",
    "penguin",
    "surgery",
    "castle",
    "laughter",
]


def answer_block(payload: str) -> str:
    return f"<answer>\n{payload.strip()}\n</answer>"


def normalize_word_field(word: str, field: str = "word") -> str:
    tokens = re.sub(r"[^A-Za-z0-9_ -]", "", str(word)).strip().split() if str(word).strip() else []
    cleaned = tokens[0] if tokens else ""
    if cleaned.lower() in PLACEHOLDER_WORDS:
        cleaned = ""
    return answer_block(json.dumps({field: cleaned}, ensure_ascii=False))


def normalize_words(words: list[Any], final_count: int = 10) -> str:
    cleaned: list[str] = []
    seen: set[str] = set()
    for word in words:
        value = re.sub(r"[^A-Za-z0-9_ -]", "", str(word)).strip()
        if not value or " " in value or "_" in value:
            continue
        if value.lower() in PLACEHOLDER_WORDS:
            continue
        key = value.lower()
        if key in seen:
            continue
        cleaned.append(value)
        seen.add(key)
        if len(cleaned) >= final_count:
            break
    for word in DEFAULT_DIVERSE_WORDS:
        if len(cleaned) >= final_count:
            break
        key = word.lower()
        if key not in seen:
            cleaned.append(word)
            seen.add(key)
    return answer_block(json.dumps({"words": cleaned}, ensure_ascii=False, indent=2))


def normalize_uses(uses: list[Any]) -> str:
    cleaned = [str(item).strip() for item in uses if str(item).strip()]
    return answer_block(json.dumps({"uses": cleaned}, ensure_ascii=False, indent=2))


def normalize_text(text: str, wrap_answer: bool = True) -> str:
    cleaned = str(text or "").strip()
    return answer_block(cleaned) if wrap_answer else cleaned


def normalize_code(code: str) -> str:
    return str(code or "").strip()


def normalize_selected(task_name: str, selected: Any, output_schema: dict[str, Any] | None = None) -> str:
    """Normalize a selected candidate when an executor has one.

    Prompt-only enhancement usually lets the model produce the final answer.
    This helper is used by tests, future executors, and batch artifacts.
    """

    task = task_name.lower()
    schema = output_schema or {}
    if isinstance(selected, dict):
        if task == "dat" and "words" in selected:
            return normalize_words(selected["words"], int(schema.get("final_count") or 10))
        if task in {"rat", "metaphor", "bats"}:
            field = "target" if task == "bats" else "word"
            return normalize_word_field(selected.get(field) or selected.get("word") or selected.get("answer", ""), field=field)
        if task == "aut" and "uses" in selected:
            return normalize_uses(selected["uses"])
        if "final_answer" in selected:
            return normalize_text(str(selected["final_answer"]))
    if task in {"rat", "metaphor", "bats"}:
        return normalize_word_field(str(selected), field="target" if task == "bats" else "word")
    if task == "neocoder":
        return normalize_code(str(selected))
    return normalize_text(str(selected))
