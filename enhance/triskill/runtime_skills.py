"""Runtime skill implementations for TriSkill.

The skills are intentionally lightweight and generic: each skill asks the LLM
for structured JSON, stores artifacts, and lets the final normalization step
produce adapter-compatible output.  This implements the execution contract in
solution.md while keeping benchmark scoring logic outside this package.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .llm import LLM, call_llm_json, parse_json_lenient
from .normalizer import normalize_selected
from .state import ElicitationState
from .task_prompts import guidance_for


@dataclass
class RuntimeSkill:
    name: str
    instruction: str

    def prompt(self, state: ElicitationState) -> str:
        artifact_summary = {key: value for key, value in state.artifacts.items() if key != "enhanced_prompt"}
        task_guidance = guidance_for(state.task_name, self.name)
        return f"""You are executing one TriSkill skill.

Task: {state.task_name}
Workflow: {state.workflow}
Skill: {self.name}
Instruction: {self.instruction}
Task-specific guidance: {task_guidance or 'Use the generic skill instruction.'}

Original prompt:
{state.original_prompt}

Visible item fields, excluding scoring-only fields:
{state.raw_item}

Prior artifacts:
{artifact_summary}

Return compact JSON. Include useful fields for this skill. If you propose candidates, put them under a candidates-like field. Do not include hidden references or gold answers.
""".strip()

    def fallback(self, state: ElicitationState, raw: str | None = None) -> dict[str, Any]:
        return {"warning": f"{self.name} returned non-JSON", "raw_response": (raw or "")[:1000]}

    def run(self, state: ElicitationState, llm: LLM, config: dict[str, Any]) -> ElicitationState:
        temp = float(config.get("verification_temperature", 0.0) if _is_verifier(self.name) else config.get("generation_temperature", 0.7))
        max_tokens = int(config.get("default_max_json_tokens", 1024))
        parsed, raw = call_llm_json(llm, self.prompt(state), temperature=temp, max_tokens=max_tokens)
        if parsed is None:
            state.warnings.append(f"{self.name} failed JSON parse")
            parsed = self.fallback(state, raw)
        state.artifacts[self.name] = parsed
        _update_candidates_and_selection(state, parsed)
        return state


class OutputNormalizationRuntimeSkill(RuntimeSkill):
    def run(self, state: ElicitationState, llm: LLM, config: dict[str, Any]) -> ElicitationState:
        selected = state.selected_candidate
        if selected is None:
            selected = _best_available_artifact(state)
        if selected is None:
            prompt = f"""You are the final TriSkill output normalizer.

Task: {state.task_name}
Output schema: {state.output_schema}

Convert the workflow artifacts into the final benchmark answer only.
Do not copy prior reasoning. Do not explain. Do not include markdown.

For DAT return JSON with "words".
For BATS return JSON with "target".
For RAT and metaphor return JSON with "word".

Original prompt:
{state.original_prompt}

Artifacts:
{_compact_artifacts_for_normalization(state.artifacts)}

Return compact JSON only.
"""
            raw = llm.generate(prompt=prompt, temperature=0.0, max_tokens=int(config.get("max_final_tokens", 160)))
            selected = parse_json_lenient(raw)
            if selected is None:
                selected = _heuristic_final_selection(state.task_name, raw, state.artifacts)
            state.final_answer = normalize_selected(state.task_name, selected or raw, dict(state.output_schema))
        else:
            state.final_answer = normalize_selected(state.task_name, selected, dict(state.output_schema))
        state.parse_success = bool(state.final_answer)
        state.artifacts[self.name] = {"final_answer": state.final_answer, "parse_success": state.parse_success}
        return state


def _is_verifier(name: str) -> bool:
    return any(marker in name for marker in ("verification", "check", "audit", "normalization"))


def _candidate_values(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in (
        "candidates",
        "candidate_concepts",
        "bridge_candidates",
        "scored_candidates",
        "unique_candidates",
        "uses",
        "plans",
        "new_primitives",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _selected_value(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return None
    for key in ("best_candidate", "selected", "selected_solution", "selected_plan", "final_answer", "word", "target"):
        if key in payload and payload[key]:
            return payload[key]
    scored = payload.get("scored_candidates")
    if isinstance(scored, list) and scored:
        return max(scored, key=lambda row: row.get("score", 0) if isinstance(row, dict) else 0)
    return None


def _update_candidates_and_selection(state: ElicitationState, parsed: Any) -> None:
    for candidate in _candidate_values(parsed):
        if isinstance(candidate, dict):
            state.candidates.append(candidate)
        else:
            state.candidates.append({"candidate": candidate})
    selected = _selected_value(parsed)
    if selected is not None:
        state.selected_candidate = selected


def _best_available_artifact(state: ElicitationState) -> Any:
    if state.candidates:
        return state.candidates[-1]
    for payload in reversed(list(state.artifacts.values())):
        selected = _selected_value(payload)
        if selected is not None:
            return selected
    return None


def _compact_artifacts_for_normalization(artifacts: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in artifacts.items():
        if key in {"enhanced_prompt", "plan"}:
            continue
        compact[key] = _truncate_artifact(value)
    return compact


def _truncate_artifact(value: Any, max_text: int = 700) -> Any:
    if isinstance(value, dict):
        return {key: _truncate_artifact(child, max_text=max_text) for key, child in value.items()}
    if isinstance(value, list):
        return [_truncate_artifact(child, max_text=max_text) for child in value[:12]]
    if isinstance(value, str):
        text = value.strip()
        if len(text) > max_text:
            return text[:max_text] + "..."
        return text
    return value


def _heuristic_final_selection(task_name: str, raw: str, artifacts: dict[str, Any]) -> Any:
    task = task_name.lower()
    if task == "dat":
        words = _extract_word_list(raw)
        if not words:
            words = _extract_word_list(str(artifacts))
        return {"words": words[:10]} if words else ""
    if task in {"rat", "metaphor", "bats"}:
        field = "target" if task == "bats" else "word"
        word = _extract_single_word(raw) or _extract_single_word(str(artifacts))
        return {field: word} if word else ""
    return raw


def _extract_single_word(text: str) -> str:
    source = str(text or "")
    parsed = parse_json_lenient(source)
    if isinstance(parsed, dict):
        for key in ("target", "word", "answer", "best_candidate", "selected", "final_answer"):
            value = parsed.get(key)
            if isinstance(value, str):
                candidate = _clean_word(value)
                if candidate:
                    return candidate
            if isinstance(value, dict):
                nested = _extract_single_word(str(value))
                if nested:
                    return nested
    patterns = (
        r"(?:final answer|answer|connecting word|replacement word|target)\s*(?:is|:)\s*['\"]?([A-Za-z][A-Za-z0-9_-]*)",
        r"['\"](?:word|target|answer)['\"]\s*:\s*['\"]([^'\"]+)['\"]",
    )
    for pattern in patterns:
        matches = re.findall(pattern, source, flags=re.IGNORECASE)
        for match in reversed(matches):
            candidate = _clean_word(match)
            if candidate:
                return candidate
    return ""


def _extract_word_list(text: str) -> list[str]:
    source = str(text or "")
    parsed = parse_json_lenient(source)
    if isinstance(parsed, dict) and isinstance(parsed.get("words"), list):
        return [_clean_word(item) for item in parsed["words"] if _clean_word(item)]
    quoted = re.findall(r"['\"]([A-Za-z][A-Za-z0-9_-]{1,30})['\"]", source)
    seen: set[str] = set()
    words: list[str] = []
    for item in quoted:
        candidate = _clean_word(item)
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        words.append(candidate)
        if len(words) >= 10:
            break
    return words


def _clean_word(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"[A-Za-z][A-Za-z0-9_-]*", text)
    return match.group(0) if match else ""


def make_runtime_skill(name: str, instruction: str) -> RuntimeSkill:
    if name == "output_normalization":
        return OutputNormalizationRuntimeSkill(name=name, instruction=instruction)
    return RuntimeSkill(name=name, instruction=instruction)
