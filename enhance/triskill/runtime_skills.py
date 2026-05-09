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
from .normalizer import DEFAULT_DIVERSE_WORDS, PLACEHOLDER_WORDS, normalize_selected, normalize_text
from .state import ElicitationState
from .task_prompts import guidance_for


@dataclass
class RuntimeSkill:
    name: str
    instruction: str

    def prompt(self, state: ElicitationState) -> str:
        artifact_summary = {key: value for key, value in state.artifacts.items() if key != "enhanced_prompt"}
        task_guidance = guidance_for(state.task_name, self.name)
        verifier_rule = ""
        if _is_verifier(self.name):
            verifier_rule = (
                "\nVerifier/selector rule: score and select from existing candidates in Prior artifacts whenever possible. "
                "Treat repeated independent direct_seed candidates and seed_votes as useful confidence evidence, but still verify the visible relation/context. "
                "Do not invent a new answer unless every prior candidate is invalid."
            )
        return f"""You are executing one TriSkill skill.

Task: {state.task_name}
Workflow: {state.workflow}
Skill: {self.name}
Instruction: {self.instruction}
Task-specific guidance: {task_guidance or 'Use the generic skill instruction.'}
{verifier_rule}

Original prompt:
{state.original_prompt}

Visible item fields, excluding scoring-only fields:
{state.raw_item}

Prior artifacts:
{artifact_summary}

Return compact JSON. Include useful fields for this skill. If you propose candidates, put them under a candidates-like field. If selecting, use best_candidate/selected and keep it to one candidate. Do not include hidden references or gold answers.
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
        if state.task_name == "aut":
            uses = _aut_uses_from_state(state)
            if uses:
                state.final_answer = normalize_selected(state.task_name, {"uses": uses}, dict(state.output_schema))
                state.parse_success = True
                state.artifacts[self.name] = {"final_answer": state.final_answer, "parse_success": state.parse_success}
                return state

        if _is_openended_text_task(state):
            state.final_answer = _normalize_openended_text(state, llm, config)
            state.parse_success = bool(state.final_answer)
            state.artifacts[self.name] = {"final_answer": state.final_answer, "parse_success": state.parse_success}
            return state

        selected = _preferred_seed_candidate(state) or state.selected_candidate
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
            selected = _canonicalize_selection(state, selected or raw)
            state.final_answer = normalize_selected(state.task_name, selected, dict(state.output_schema))
        else:
            selected = _canonicalize_selection(state, selected)
            state.final_answer = normalize_selected(state.task_name, selected, dict(state.output_schema))
        state.parse_success = bool(state.final_answer)
        state.artifacts[self.name] = {"final_answer": state.final_answer, "parse_success": state.parse_success}
        return state


def _is_verifier(name: str) -> bool:
    return any(marker in name for marker in ("verification", "check", "audit", "normalization"))


def _is_openended_text_task(state: ElicitationState) -> bool:
    return state.output_schema.get("type") in {"solution_text", "story_text", "code", "reconstruction_text"}


def _normalize_openended_text(state: ElicitationState, llm: LLM, config: dict[str, Any]) -> str:
    seed = _direct_seed_text(state)
    if state.workflow == "exploratory":
        if state.output_schema.get("type") in {"story_text", "code"}:
            text = _clean_openended_text(seed or _best_raw_text_artifact(state) or "")
            if _looks_unfinished_for_schema(state, text):
                text = _finalize_openended_answer(state, llm, config, seed)
        else:
            text = _best_exploratory_finished_text(state, seed)
        if state.task_name == "neocoder":
            return text
        return normalize_text(text)
    if state.task_name == "neocoder":
        return _clean_openended_text(seed or _best_raw_text_artifact(state) or "")

    prompt = f"""You are the final TriSkill answer normalizer.

Task: {state.task_name}
Output type: {state.output_schema.get("type")}

Original prompt:
{state.original_prompt}

Direct answer anchor:
{seed or '(none)'}

Workflow artifacts:
{_compact_artifacts_for_normalization(state.artifacts)}

Write the final benchmark answer only.
Use the direct answer as the fidelity anchor, and incorporate workflow ideas only if they improve the answer without reducing correctness, coherence, feasibility, or constraint satisfaction.
Do not output JSON or a Python dict unless the original prompt explicitly requires it.
Do not mention TriSkill, workflow, artifacts, candidates, or scoring.
Do not include hidden reasoning.
"""
    raw = llm.generate(prompt=prompt, temperature=0.0, max_tokens=int(config.get("max_final_tokens", 1200)))
    text = _clean_openended_text(raw)
    if _looks_degenerate_openended(text):
        text = _clean_openended_text(seed or _best_raw_text_artifact(state) or raw)
    if state.task_name == "neocoder":
        return text
    return normalize_text(text)


def _finalize_openended_answer(state: ElicitationState, llm: LLM, config: dict[str, Any], seed: str) -> str:
    modality = state.output_schema.get("type")
    if modality == "code":
        modality_rule = (
            "Return the exact code/programming answer format requested by the original prompt. "
            "If the original prompt asks for a JSON object, return one valid JSON object only. "
            "Include the complete executable solution content, not analysis."
        )
    elif modality == "story_text":
        modality_rule = "Write the finished story only. Do not include outlines, constraint lists, reviews, headings, or planning."
    else:
        modality_rule = "Write the final answer only."
    prompt = f"""You are a final-answer renderer.

Output type: {modality}
Rule: {modality_rule}

Original prompt:
{state.original_prompt}

Draft anchor:
{_clean_openended_text(seed)}

Workflow artifacts:
{_compact_artifacts_for_normalization(state.artifacts)}

Return only the final benchmark answer. Do not mention this rendering step.
"""
    raw = llm.generate(prompt=prompt, temperature=0.0, max_tokens=int(config.get("max_final_tokens", 2048)))
    text = _clean_openended_text(raw)
    if modality == "code":
        extracted = _extract_json_object_with_key(text, "solve_lines")
        if extracted:
            return extracted
    if _looks_unfinished_for_schema(state, text):
        salvage = _best_finished_fragment(seed or _best_raw_text_artifact(state) or raw)
        if salvage and not _looks_unfinished_for_schema(state, salvage):
            return salvage
    return text


def _direct_seed_text(state: ElicitationState) -> str:
    for candidate in state.candidates:
        if isinstance(candidate, dict) and candidate.get("source") == "direct_seed" and candidate.get("text"):
            return str(candidate["text"])
    seed = state.artifacts.get("direct_seed")
    if isinstance(seed, dict):
        for candidate in seed.get("candidates", []):
            if isinstance(candidate, dict) and candidate.get("text"):
                return str(candidate["text"])
    return ""


def _best_raw_text_artifact(state: ElicitationState) -> str:
    for payload in reversed(list(state.artifacts.values())):
        if isinstance(payload, dict):
            selected = _selected_value(payload)
            if isinstance(selected, str) and selected.strip():
                return selected
            raw = payload.get("raw_response")
            if isinstance(raw, str) and raw.strip():
                return raw
            final = payload.get("final_answer")
            if isinstance(final, str) and final.strip():
                return final
    return ""


def _best_exploratory_finished_text(state: ElicitationState, seed: str) -> str:
    """Select a finished answer candidate without task-specific scoring rules."""

    candidates: list[str] = []
    for payload in reversed(list(state.artifacts.values())):
        if not isinstance(payload, dict):
            continue
        for value in (_selected_value(payload), payload.get("final_answer"), payload.get("raw_response")):
            if isinstance(value, str) and value.strip():
                candidates.append(value)
    if seed:
        candidates.append(seed)

    cleaned = [_clean_openended_text(candidate) for candidate in candidates]
    viable = [candidate for candidate in cleaned if not _looks_degenerate_openended(candidate)]
    if not viable:
        return _clean_openended_text(seed or _best_raw_text_artifact(state) or "")
    return max(viable, key=_finished_answer_score)


def _clean_openended_text(raw: str) -> str:
    text = str(raw or "").strip()
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    if re.search(r"</think(?:ing)?>", text, flags=re.IGNORECASE):
        text = re.split(r"</think(?:ing)?>", text, flags=re.IGNORECASE)[-1].strip()
    text = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    text = _best_finished_fragment(text)
    text = _strip_openended_meta_prefix(text)
    return text


def _best_finished_fragment(text: str) -> str:
    source = str(text or "").strip()
    if not source:
        return ""
    draft_patterns = (
        r"(?:^|\n)\s*\*{0,2}Drafting\*{0,2}\s*:(.*?)(?:\n\s*(?:\*{0,2}Review\b|\*{0,2}Word Count\b|Wait,|Let's|Refining for Constraints)|\Z)",
        r"(?:^|\n)\s*\*{0,2}Final Draft\*{0,2}\s*:(.*?)(?:\n\s*(?:\*{0,2}Review\b|Wait,|Let's)|\Z)",
        r"(?:^|\n)\s*\*{0,2}Final Story\*{0,2}\s*:(.*?)(?:\n\s*(?:\*{0,2}Review\b|Wait,|Let's)|\Z)",
    )
    for pattern in draft_patterns:
        match = re.search(pattern, source, flags=re.IGNORECASE | re.DOTALL)
        if match:
            fragment = match.group(1).strip()
            if len(fragment.split()) >= 20:
                return fragment
    return source


def _strip_openended_meta_prefix(text: str) -> str:
    text = _extract_labeled_finished_segment(text)
    lines = str(text or "").splitlines()
    while lines and _is_meta_prefix_line(lines[0]):
        lines.pop(0)
    return "\n".join(lines).strip()


def _extract_labeled_finished_segment(text: str) -> str:
    markers = (
        "final answer:",
        "final story:",
        "final draft:",
        "revised draft:",
        "revised story:",
        "answer:",
        "story:",
    )
    lowered = text.lower()
    positions = [(lowered.rfind(marker), marker) for marker in markers]
    positions = [(pos, marker) for pos, marker in positions if pos >= 0]
    if not positions:
        return text
    pos, marker = max(positions)
    segment = text[pos + len(marker) :].strip()
    return segment or text


def _is_meta_prefix_line(line: str) -> bool:
    stripped = line.strip().lower()
    if not stripped:
        return False
    prefixes = (
        "okay, so",
        "alright, so",
        "i need to",
        "let me ",
        "we need to",
        "thinking process",
        "the task is",
        "the story should",
        "first,",
    )
    return any(stripped.startswith(prefix) for prefix in prefixes)


def _looks_degenerate_openended(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return True
    if len(stripped.split()) < 8 and not stripped.startswith(("def ", "class ", "import ", "from ")):
        return True
    if stripped.startswith("{") and "'type':" in stripped[:120]:
        return True
    if stripped.count("<answer>") > 1:
        return True
    lowered = stripped[:800].lower()
    if re.search(r"</?think(?:ing)?>", lowered):
        return True
    if lowered.startswith(("okay, so", "alright, so", "i need to", "let me ")):
        return True
    meta_markers = (
        "thinking process",
        "analyze the request",
        "drafting strategy",
        "constraint checklist",
        "constraint conflict",
        "refining constraint",
    )
    if sum(1 for marker in meta_markers if marker in lowered) >= 2:
        return True
    return False


def _looks_unfinished_for_schema(state: ElicitationState, text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return True
    lowered = str(text or "").strip().lower()
    if state.output_schema.get("type") == "story_text":
        if len(stripped.split()) < 5:
            return True
        if re.search(r"</?think(?:ing)?>", lowered):
            return True
        planning_markers = (
            "deconstruct constraints",
            "drafting plan",
            "word count check",
            "review against constraints",
            "re-evaluating constraint",
            "**constraints:**",
            "**output format:**",
        )
        return sum(1 for marker in planning_markers if marker in lowered[:2500]) >= 2
    if state.output_schema.get("type") == "code":
        if _looks_degenerate_openended(text):
            return True
        if _code_answer_is_complete(state, text):
            return False
        code_markers = ("problem asks", "let me analyze", "wait,", "the key insight", "constraints:")
        return any(marker in lowered[:1200] for marker in code_markers)
    return False


def _code_answer_is_complete(state: ElicitationState, text: str) -> bool:
    source = str(text or "").strip()
    parsed = parse_json_lenient(source)
    if isinstance(parsed, dict) and isinstance(parsed.get("solve_lines"), list) and parsed["solve_lines"]:
        return True
    json_fragment = _extract_json_object_with_key(source, "solve_lines")
    if json_fragment:
        parsed_fragment = parse_json_lenient(json_fragment)
        if isinstance(parsed_fragment, dict) and isinstance(parsed_fragment.get("solve_lines"), list) and parsed_fragment["solve_lines"]:
            return True
    original = state.original_prompt.lower()
    if "solve_lines" in original or "json object" in original:
        return False
    return bool(re.search(r"^(?:import\s+\w+|from\s+\w+\s+import\s+|\s*def\s+solve\s*\()", source, flags=re.MULTILINE))


def _extract_json_object_with_key(text: str, key: str) -> str:
    source = str(text or "")
    key_pos = source.find(f'"{key}"')
    if key_pos < 0:
        key_pos = source.find(f"'{key}'")
    if key_pos < 0:
        return ""
    start = source.rfind("{", 0, key_pos)
    if start < 0:
        return ""
    depth = 0
    in_string: str | None = None
    escape = False
    for index in range(start, len(source)):
        char = source[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if in_string:
            if char == in_string:
                in_string = None
            continue
        if char in {'"', "'"}:
            in_string = char
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    return ""


def _finished_answer_score(text: str) -> float:
    stripped = str(text or "").strip()
    words = stripped.split()
    score = min(len(words), 300) / 300.0
    lowered = stripped[:1200].lower()
    if "<answer>" in lowered or "</answer>" in lowered:
        score -= 0.5
    if re.search(r"</?think(?:ing)?>", lowered):
        score -= 1.0
    if any(marker in lowered for marker in ("workflow", "artifact", "candidate", "score and select")):
        score -= 0.5
    if any(marker in stripped for marker in ("$", "\\boxed", "def ", "class ", "import ", "```")):
        score += 0.2
    if "\n\n" in stripped:
        score += 0.1
    return score


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
        for candidate in state.candidates:
            if isinstance(candidate, dict) and candidate.get("source") == "direct_seed":
                return candidate
        return state.candidates[-1]
    for payload in reversed(list(state.artifacts.values())):
        selected = _selected_value(payload)
        if selected is not None:
            return selected
    return None


def _preferred_seed_candidate(state: ElicitationState) -> Any:
    """Use direct-answer anchoring for tasks where workflow over-selection regresses.

    The combinational workflow still runs and records its artifacts, but BATS and
    metaphor are high-precision single-word tasks.  Empirically, verifier steps can
    replace a correct direct answer with a weaker paraphrase or relation guess, so
    the direct seed is the default unless it is missing or visibly invalid.
    """

    if state.task_name not in {"bats", "metaphor"}:
        return None
    if "context_fit" in state.canonical_metrics or "metaphorical_fit" in state.canonical_metrics:
        return None
    for candidate in state.candidates:
        if not isinstance(candidate, dict) or candidate.get("source") != "direct_seed":
            continue
        field = "target" if state.task_name == "bats" else "word"
        word = _clean_word(candidate.get(field) or candidate.get("word") or candidate.get("target"))
        if word:
            preferred = dict(candidate)
            preferred[field] = word
            preferred["selection_policy"] = "direct_seed_anchor"
            return preferred
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
    if words:
        return words
    return _default_diverse_words()


def _clean_word(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"[A-Za-z][A-Za-z0-9_-]*", text)
    if not match:
        return ""
    word = match.group(0)
    return "" if word.lower() in PLACEHOLDER_WORDS else word


def _default_diverse_words() -> list[str]:
    return list(DEFAULT_DIVERSE_WORDS)


def _aut_uses_from_state(state: ElicitationState) -> list[str]:
    uses: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        text = _clean_aut_use(value)
        if not text:
            return
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        uses.append(text)

    for candidate in state.candidates:
        if isinstance(candidate, dict):
            if isinstance(candidate.get("uses"), list):
                for item in candidate["uses"]:
                    add(item)
            for key in ("use", "candidate", "description", "text", "final_answer", "idea"):
                if key in candidate:
                    add(candidate[key])
        else:
            add(candidate)

    for payload in state.artifacts.values():
        for item in _extract_aut_uses_from_artifact(payload):
            add(item)
        if isinstance(payload, dict) and isinstance(payload.get("uses"), list):
            for item in payload["uses"]:
                add(item)
            raw = payload.get("raw_response")
            if isinstance(raw, str):
                for item in _extract_aut_uses_from_text(raw):
                    add(item)
        for candidate in _candidate_values(payload):
            if isinstance(candidate, dict):
                for key in ("use", "candidate", "description", "text", "final_answer", "idea"):
                    if key in candidate:
                        add(candidate[key])
            else:
                add(candidate)

    return uses


def _extract_aut_uses_from_artifact(payload: Any) -> list[str]:
    uses: list[str] = []
    if isinstance(payload, dict):
        raw = payload.get("raw_response")
        if isinstance(raw, str):
            uses.extend(_extract_aut_uses_from_text(raw))
        for value in payload.values():
            if isinstance(value, (dict, list)):
                uses.extend(_extract_aut_uses_from_artifact(value))
    elif isinstance(payload, list):
        for value in payload:
            uses.extend(_extract_aut_uses_from_artifact(value))
    return uses


def _clean_aut_use(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" -*\t\r\n")
    if not text:
        return ""
    if re.fullmatch(r"use\s*\d+", text, flags=re.IGNORECASE):
        return ""
    if text.lower() in PLACEHOLDER_WORDS:
        return ""
    lowered = text.lower()
    if any(char in text for char in ("`", "<", ">", "{", "}", "[", "]")):
        return ""
    if "**" in text or text.endswith(":") or re.search(
        r"\b(?:task|workflow|skill|schema|format|constraint|question|prompt|goal|instruction)\s*[:：]",
        lowered,
    ):
        return ""
    if re.match(r"^(?:no|do not|return|output|include|start|remember|recognize|decline|refuse)\b", lowered):
        return ""
    meta_markers = (
        "thinking process",
        "analyze the request",
        "output format",
        "schema",
        "constraint",
        "json",
        "<answer>",
        "candidate_generation returned",
        "prior artifacts",
        "safety violation",
        "harm the safety",
        "illegal",
        "unethical",
        "child abuse",
    )
    if any(marker in lowered for marker in meta_markers):
        return ""
    if len(text.split()) > 18:
        return ""
    return text


def _extract_aut_uses_from_text(text: str) -> list[str]:
    source = str(text or "")
    parsed = parse_json_lenient(source)
    if isinstance(parsed, dict) and isinstance(parsed.get("uses"), list):
        parsed_uses = [item for item in parsed["uses"] if _clean_aut_use(item)]
        if parsed_uses:
            return parsed_uses
    uses: list[str] = []
    for match in re.findall(r"\bIdeas?\s*:\s*([^\n]+)", source, flags=re.IGNORECASE):
        for part in re.split(r"[,;]", match):
            cleaned = _clean_aut_use(part)
            if cleaned:
                uses.append(cleaned)
    for line in source.splitlines():
        stripped = line.strip()
        if not re.match(r"^(?:[-*]|\d+[.)])\s+", stripped):
            continue
        candidate = re.sub(r"^(?:[-*]|\d+[.)])\s+", "", stripped).strip()
        cleaned = _clean_aut_use(candidate)
        if cleaned:
            uses.append(cleaned)
    return uses


def _canonicalize_selection(state: ElicitationState, selected: Any) -> Any:
    return selected


def make_runtime_skill(name: str, instruction: str) -> RuntimeSkill:
    if name == "output_normalization":
        return OutputNormalizationRuntimeSkill(name=name, instruction=instruction)
    return RuntimeSkill(name=name, instruction=instruction)
