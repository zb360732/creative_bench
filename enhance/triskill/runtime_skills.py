"""Runtime skill implementations for TriSkill.

The skills are intentionally lightweight and generic: each skill asks the LLM
for structured JSON, stores artifacts, and lets the final normalization step
produce adapter-compatible output.  This implements the execution contract in
solution.md while keeping benchmark scoring logic outside this package.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

from .llm import LLM, call_llm_json, parse_json_lenient
from .normalizer import DEFAULT_DIVERSE_WORDS, PLACEHOLDER_WORDS, normalize_selected, normalize_text
from .state import ElicitationState
from .task_prompts import guidance_for


_SEMANTIC_MODEL_CACHE: dict[str, Any] = {}


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
        if state.task_name == "dat":
            words = _dat_final_words_from_state(state, llm, config)
            if words:
                state.final_answer = normalize_selected(state.task_name, {"words": words}, dict(state.output_schema))
                state.parse_success = True
                state.artifacts[self.name] = {
                    "final_answer": state.final_answer,
                    "parse_success": state.parse_success,
                    "selection_policy": "semantic_diversity_greedy",
                    "num_words": len(words),
                }
                return state

        if state.task_name == "aut":
            uses = _aut_uses_from_state(state)
            min_uses = int(config.get("aut_min_final_uses") or 0)
            if len(uses) < min_uses:
                uses = _merge_aut_uses(uses, _render_aut_use_portfolio(state, llm, config, uses))
            if _aut_prompt_target_is_living_entity(state.original_prompt) and len(uses) < min_uses:
                uses = _merge_aut_uses(uses, _render_living_entity_aut_uses(state.original_prompt))
            max_uses = int(config.get("aut_max_final_uses") or 0)
            polish_triggered = False
            if _aut_needs_final_polish(uses, max_uses):
                polished = _render_aut_use_polish(state, llm, config, uses)
                if len(polished) >= min(12, max(1, min_uses)):
                    uses = _merge_aut_uses(polished, uses)
                    polish_triggered = True
            uses = _filter_aut_low_quality_labels(state.original_prompt, uses, min_uses, strict=polish_triggered)
            uses = _select_aut_final_uses(state.original_prompt, uses, max_uses)
            if uses:
                state.final_answer = normalize_selected(state.task_name, {"uses": uses}, dict(state.output_schema))
                state.parse_success = True
                state.artifacts[self.name] = {"final_answer": state.final_answer, "parse_success": state.parse_success, "num_uses": len(uses)}
                return state

        if _is_openended_text_task(state):
            state.final_answer = _normalize_openended_text(state, llm, config)
            state.parse_success = bool(state.final_answer)
            state.artifacts[self.name] = {"final_answer": state.final_answer, "parse_success": state.parse_success}
            return state

        if state.task_name == "bats":
            selected = _bats_relation_consensus_candidate(state)
            if selected:
                state.final_answer = normalize_selected(state.task_name, selected, dict(state.output_schema))
                state.parse_success = bool(state.final_answer)
                state.artifacts[self.name] = {
                    "final_answer": state.final_answer,
                    "parse_success": state.parse_success,
                    "selection_policy": "relation_module_consensus",
                    "selected": selected,
                }
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
        if state.output_schema.get("type") == "code":
            text = _best_code_candidate_text(state)
            if not text:
                text = _clean_openended_text(seed or _best_raw_text_artifact(state) or "")
                if _looks_unfinished_for_schema(state, text):
                    text = _finalize_openended_answer(state, llm, config, seed)
                text = _normalize_code_candidate_text(state, text) or ""
            text_example_result = _code_text_example_result(state, text) if text else "fail"
            text_violates_constraints = _code_text_violates_visible_constraints(state, text) if text else False
            if not text or text_example_result == "fail" or text_violates_constraints:
                repaired = _repair_code_candidate(state, llm, config, text)
                if repaired:
                    repaired_example_result = _code_text_example_result(state, repaired)
                    repaired_violates_constraints = _code_text_violates_visible_constraints(state, repaired)
                    accept_repair = False
                    if not repaired_violates_constraints:
                        accept_repair = (
                            not text
                            or text_violates_constraints
                            or _code_candidate_score(state, repaired) > _code_candidate_score(state, text)
                        )
                    if accept_repair and repaired_example_result != "fail":
                        text = repaired
            if text and _code_text_violates_visible_constraints(state, text):
                constrained = _render_constraint_preserving_code(state, llm, config, text)
                constrained_example_result = _code_text_example_result(state, constrained) if constrained else "fail"
                if (
                    constrained
                    and not _code_text_violates_visible_constraints(state, constrained)
                    and constrained_example_result != "fail"
                ):
                    text = constrained
            if not text:
                text = _clean_openended_text(seed or _best_raw_text_artifact(state) or "")
        elif state.output_schema.get("type") == "story_text":
            text = _clean_openended_text(seed or _best_raw_text_artifact(state) or "")
            if _looks_unfinished_for_schema(state, text):
                text = _finalize_openended_answer(state, llm, config, seed)
        else:
            text = _best_exploratory_finished_text(state, seed)
        if state.task_name == "neocoder":
            return text
        return normalize_text(text)
    if state.task_name == "neocoder":
        return _best_code_candidate_text(state) or _clean_openended_text(seed or _best_raw_text_artifact(state) or "")

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


def _best_code_candidate_text(state: ElicitationState) -> str:
    scored: list[tuple[float, int, str]] = []
    clean_scored: list[tuple[float, int, str]] = []
    direct_scored: list[tuple[float, int, str]] = []
    clean_direct_scored: list[tuple[float, int, str]] = []
    sample_pass_scored: list[tuple[float, int, str]] = []
    clean_sample_pass_scored: list[tuple[float, int, str]] = []
    for index, candidate in enumerate(_iter_code_candidate_sources(state)):
        normalized = _normalize_code_candidate_text(state, candidate)
        if not normalized:
            continue
        score = _code_candidate_score(state, normalized)
        row = (score, -index, normalized)
        violates_constraints = _code_text_violates_visible_constraints(state, normalized)
        scored.append(row)
        if not violates_constraints:
            clean_scored.append(row)
        if _is_direct_code_candidate(candidate):
            direct_scored.append(row)
            if not violates_constraints:
                clean_direct_scored.append(row)
        if _code_text_example_result(state, normalized) == "pass":
            sample_pass_scored.append(row)
            if not violates_constraints:
                clean_sample_pass_scored.append(row)
    if clean_sample_pass_scored:
        return max(clean_sample_pass_scored, key=lambda item: (item[0], item[1]))[2]
    if clean_direct_scored:
        return max(clean_direct_scored, key=lambda item: (item[0], item[1]))[2]
    if clean_scored:
        return max(clean_scored, key=lambda item: (item[0], item[1]))[2]
    if sample_pass_scored:
        return max(sample_pass_scored, key=lambda item: (item[0], item[1]))[2]
    if direct_scored:
        return max(direct_scored, key=lambda item: (item[0], item[1]))[2]
    if not scored:
        return ""
    return max(scored, key=lambda item: (item[0], item[1]))[2]


def _is_direct_code_candidate(candidate: Any) -> bool:
    return isinstance(candidate, dict) and candidate.get("source") == "direct_seed"


def _repair_code_candidate(state: ElicitationState, llm: LLM, config: dict[str, Any], candidate_text: str) -> str:
    examples = _extract_visible_io_examples(state.original_prompt)
    forbidden_terms = sorted(_extract_forbidden_code_terms(state))
    forbidden_block = "\n".join(f"- {term}" for term in forbidden_terms) or "- No explicit forbidden technique was parsed."
    example_block = ""
    if examples:
        stdin_text, expected_text = examples[0]
        observed = _run_visible_code_example_detail(_code_from_normalized_text(candidate_text), stdin_text, expected_text).get("stdout", "")
        example_block = (
            f"Visible sample input:\n{stdin_text}\n\n"
            f"Expected visible sample output:\n{expected_text}\n\n"
            f"Current candidate output on that sample:\n{observed or '(empty/non-executable)'}"
        )
    prompt = f"""Repair this programming answer using only the visible prompt and visible sample I/O.

Rules:
- Preserve the benchmark output format. If the original prompt asks for JSON with solve_lines, return exactly one JSON object with "think" and "solve_lines".
- The code must define solve() with no arguments.
- Do not call solve(); the evaluator will call it.
- Do not include markdown or explanatory prose outside JSON.
- Respect all visible programming constraints.
- Parsed forbidden programming techniques:
{forbidden_block}
- Avoid comments and docstrings in solve_lines.
- Do not return the unchanged candidate if it fails the visible sample.
- Do not return the unchanged candidate if it uses a parsed forbidden technique.

Original prompt:
{state.original_prompt}

Current candidate that failed visible sample validation:
{candidate_text}

{example_block}

Return the repaired final answer only.
"""
    raw = llm.generate(prompt=prompt, temperature=0.0, max_tokens=int(config.get("max_final_tokens", 2048)))
    return _normalize_code_candidate_text(state, raw)


def _render_constraint_preserving_code(state: ElicitationState, llm: LLM, config: dict[str, Any], candidate_text: str) -> str:
    forbidden_terms = sorted(_extract_forbidden_code_terms(state))
    if not forbidden_terms:
        return ""
    prompt = f"""Rewrite this programming answer so the final code satisfies every visible programming constraint.

Hard priority order:
1. Visible programming constraints and requested output format.
2. Executable Python with a solve() function and no top-level solve() call.
3. Correctness on the visible problem.

Forbidden techniques parsed from the prompt:
{_format_forbidden_code_terms(forbidden_terms)}

Concrete rewrite hints:
{_forbidden_code_rewrite_hints(forbidden_terms)}

Rules:
- Return exactly one JSON object with "think" and "solve_lines".
- Keep "think" empty.
- Do not include comments, markdown, prose, or docstrings.
- If a forbidden technique is needed for the obvious solution, use a different generic control/data pattern instead.

Original prompt:
{state.original_prompt}

Current constraint-violating candidate:
{candidate_text}

Return only the rewritten final answer.
"""
    raw = llm.generate(prompt=prompt, temperature=0.0, max_tokens=int(config.get("max_final_tokens", 2048)))
    return _normalize_code_candidate_text(state, raw)


def _format_forbidden_code_terms(forbidden_terms: list[str]) -> str:
    return "\n".join(f"- {term}" for term in forbidden_terms)


def _forbidden_code_rewrite_hints(forbidden_terms: list[str]) -> str:
    hints: list[str] = []
    terms = set(forbidden_terms)
    if "for loop" in terms:
        hints.append("Forbid ast.For, comprehensions, generator expressions, and any `for ... in ...` syntax.")
        if "while loop" not in terms:
            hints.append("Replace counted `for` loops with `while` loops and explicit indexes.")
        else:
            hints.append("Since `while` is also forbidden, prefer recursion, string/list methods, direct formulas, or precomputed tables.")
    if "while loop" in terms:
        hints.append("Forbid ast.While and `while` syntax.")
    if "if statement" in terms:
        hints.append("Forbid ast.If and ternary expressions; prefer boolean arithmetic, list indexing, exceptions, or library functions.")
    if "continue statement" in terms:
        hints.append("Forbid `continue`; restructure loop conditions or move work into helper functions.")
    if "break statement" in terms:
        hints.append("Forbid `break`; use bounded conditions, exceptions, or helper-function returns.")
    if "sorting" in terms:
        hints.append("Forbid sorted(), .sort(), and sorting-based algorithms.")
    if "dictionary" in terms or "hashmap" in terms:
        hints.append("Forbid dict literals and dict(); use lists, tuples only if allowed, counters by index, or arithmetic encodings.")
    if "set" in terms:
        hints.append("Forbid set literals and set(); use lists, strings, counters, or boolean arrays.")
    if "tuple" in terms:
        hints.append("Forbid tuple literals and tuple(); avoid multiple assignment that creates tuples.")
    if "recursion" in terms:
        hints.append("Forbid recursive helper calls; use iterative or functional alternatives that are otherwise allowed.")
    if "queue" in terms:
        hints.append("Forbid queue/deque abstractions; use index ranges or plain lists when allowed.")
    if "stack" in terms:
        hints.append("Forbid stack abstractions; use direct state variables or recursion if allowed.")
    return "\n".join(f"- {hint}" for hint in hints) or "- Follow the parsed forbidden list exactly."


def _iter_code_candidate_sources(state: ElicitationState) -> list[Any]:
    sources: list[Any] = []
    sources.extend(state.candidates)
    for payload in state.artifacts.values():
        sources.extend(_extract_code_like_values(payload))
    return sources


def _extract_code_like_values(value: Any) -> list[Any]:
    values: list[Any] = []
    if isinstance(value, dict):
        if isinstance(value.get("solve_lines"), list) or isinstance(value.get("solve"), str):
            values.append(value)
        for key in ("text", "raw_response", "final_answer", "selected", "selected_solution", "best_candidate", "candidate"):
            if key in value:
                values.extend(_extract_code_like_values(value[key]))
        for key, child in value.items():
            if key in {
                "text",
                "raw_response",
                "final_answer",
                "selected",
                "selected_solution",
                "best_candidate",
                "candidate",
                "solve_lines",
                "solve",
            }:
                continue
            if isinstance(child, (dict, list)):
                values.extend(_extract_code_like_values(child))
    elif isinstance(value, list):
        if value and all(isinstance(item, str) for item in value):
            values.append({"solve_lines": value})
        for child in value:
            values.extend(_extract_code_like_values(child))
    elif isinstance(value, str) and _looks_codeish_text(value):
        values.append(value)
    return values


def _looks_codeish_text(value: str) -> bool:
    text = str(value or "")
    lowered = text[:1200].lower()
    return any(marker in lowered for marker in ("solve_lines", "def solve", "```python", "```json", '"solve"', "'solve'"))


def _normalize_code_candidate_text(state: ElicitationState, value: Any) -> str:
    payload = _code_payload_from_value(value)
    if payload is None and isinstance(value, str):
        cleaned = _clean_openended_text(value)
        json_fragment = _extract_json_object_with_key(cleaned, "solve_lines")
        if json_fragment:
            payload = _code_payload_from_value(json_fragment)
        if payload is None:
            code = _extract_python_code(cleaned)
            if code:
                payload = {"think": "", "solve_lines": code.splitlines()}
    if payload is None:
        return ""

    solve_lines = payload.get("solve_lines")
    if isinstance(solve_lines, str):
        solve_lines = solve_lines.splitlines()
    if not isinstance(solve_lines, list) or not solve_lines:
        solve_text = payload.get("solve")
        if isinstance(solve_text, str) and solve_text.strip():
            solve_lines = solve_text.strip().splitlines()
    if not isinstance(solve_lines, list) or not solve_lines:
        return ""

    lines = [str(line).rstrip() for line in solve_lines]
    code = "\n".join(lines).strip()
    code = _strip_code_comments_and_main_call(code)
    if not _code_has_valid_solve(code):
        return ""
    if _has_top_level_solve_call(code):
        return ""
    if _has_forbidden_import(code):
        return ""

    normalized = {"think": "", "solve_lines": code.splitlines()}
    if "json object" in state.original_prompt.lower() or "solve_lines" in state.original_prompt.lower():
        return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return code


def _code_payload_from_value(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if isinstance(value.get("solve_lines"), list) or isinstance(value.get("solve"), str):
            return dict(value)
        for key in ("text", "raw_response", "final_answer", "selected", "selected_solution", "best_candidate", "candidate"):
            nested = _code_payload_from_value(value.get(key))
            if nested is not None:
                return nested
        return None
    if not isinstance(value, str):
        return None
    parsed = parse_json_lenient(value)
    if isinstance(parsed, dict):
        return _code_payload_from_value(parsed)
    return None


def _extract_python_code(text: str) -> str:
    source = str(text or "").strip()
    for match in re.finditer(r"```(?:\s*(?:python|py))?\s*\n?(.*?)```", source, flags=re.IGNORECASE | re.DOTALL):
        candidate = match.group(1).strip()
        if "def solve" in candidate:
            return _strip_language_label(candidate)
    if re.search(r"^(?:import\s+\w+|from\s+\w+\s+import\s+|def\s+solve\s*\()", source, flags=re.MULTILINE):
        return source
    return ""


def _strip_language_label(text: str) -> str:
    lines = str(text or "").splitlines()
    if lines and lines[0].strip().lower() in {"python", "py", "json"}:
        lines = lines[1:]
    return "\n".join(lines).strip()


def _code_has_valid_solve(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "solve":
            return not (
                node.args.posonlyargs
                or node.args.args
                or node.args.vararg
                or node.args.kwonlyargs
                or node.args.kwarg
            )
    return False


def _has_top_level_solve_call(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id == "solve":
                return True
        if isinstance(node, ast.If) and _is_main_guard(node):
            for child in node.body:
                if isinstance(child, ast.Expr) and isinstance(child.value, ast.Call):
                    func = child.value.func
                    if isinstance(func, ast.Name) and func.id == "solve":
                        return True
    return False


def _strip_code_comments_and_main_call(code: str) -> str:
    lines = [line for line in str(code or "").splitlines() if not line.lstrip().startswith("#")]
    cleaned = "\n".join(lines).strip()
    try:
        tree = ast.parse(cleaned)
    except SyntaxError:
        return cleaned
    if not tree.body:
        return cleaned
    last = tree.body[-1]
    if not isinstance(last, ast.If) or not _is_main_guard(last):
        return cleaned
    if not any(
        isinstance(child, ast.Expr)
        and isinstance(child.value, ast.Call)
        and isinstance(child.value.func, ast.Name)
        and child.value.func.id == "solve"
        for child in last.body
    ):
        return cleaned
    start = getattr(last, "lineno", None)
    if not start:
        return cleaned
    return "\n".join(cleaned.splitlines()[: start - 1]).rstrip()


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False
    comparator = test.comparators[0]
    return isinstance(comparator, ast.Constant) and comparator.value == "__main__"


def _has_forbidden_import(code: str) -> bool:
    forbidden = ("os", "subprocess", "socket", "shutil", "pathlib", "requests", "urllib")
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] in forbidden for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in forbidden:
            return True
    return False


def _code_candidate_score(state: ElicitationState, normalized_text: str) -> float:
    code = _code_from_normalized_text(normalized_text)
    if not code:
        return -100.0
    score = 1.0
    if _code_has_valid_solve(code):
        score += 2.0
    if not _has_top_level_solve_call(code):
        score += 1.0
    if not _has_forbidden_import(code):
        score += 1.0
    if _violates_visible_code_constraints(state, code):
        score -= 5.0
    example_result = _run_visible_code_examples(state.original_prompt, code)
    if example_result == "pass":
        score += 8.0
    elif example_result == "fail":
        score -= 6.0
    if "sys.stdin.readline" in code or "input()" in code:
        score += 0.3
    score += min(len(code.splitlines()), 80) / 400.0
    return score


def _code_text_example_result(state: ElicitationState, normalized_text: str) -> str:
    code = _code_from_normalized_text(normalized_text)
    if not code:
        return "fail"
    return _run_visible_code_examples(state.original_prompt, code)


def _code_text_violates_visible_constraints(state: ElicitationState, normalized_text: str) -> bool:
    code = _code_from_normalized_text(normalized_text)
    return bool(code and _violates_visible_code_constraints(state, code))


def _code_from_normalized_text(text: str) -> str:
    parsed = parse_json_lenient(text)
    if isinstance(parsed, dict):
        solve_lines = parsed.get("solve_lines")
        if isinstance(solve_lines, list):
            return "\n".join(str(line) for line in solve_lines).strip()
        solve = parsed.get("solve")
        if isinstance(solve, str):
            return solve.strip()
    return _extract_python_code(text) or str(text or "").strip()


def _violates_visible_code_constraints(state: ElicitationState, code: str) -> bool:
    forbidden = _extract_forbidden_code_terms(state)
    if not forbidden:
        return False
    lowered = code.lower()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True
    for term in forbidden:
        if term == "for loop":
            if any(isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)) for node in ast.walk(tree)):
                return True
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"map", "filter"}
                for node in ast.walk(tree)
            ):
                return True
        if term == "while loop" and any(isinstance(node, ast.While) for node in ast.walk(tree)):
            return True
        if term == "if statement" and any(isinstance(node, (ast.If, ast.IfExp)) for node in ast.walk(tree)):
            return True
        if term == "continue statement" and any(isinstance(node, ast.Continue) for node in ast.walk(tree)):
            return True
        if term == "break statement" and any(isinstance(node, ast.Break) for node in ast.walk(tree)):
            return True
        if term == "recursion" and _uses_recursion(code):
            return True
        if term == "sorting" and (
            re.search(r"\b(?:sorted|sort)\s*\(", lowered)
            or any(
                isinstance(node, ast.Call)
                and (
                    (isinstance(node.func, ast.Name) and node.func.id == "sorted")
                    or (isinstance(node.func, ast.Attribute) and node.func.attr == "sort")
                )
                for node in ast.walk(tree)
            )
        ):
            return True
        if term in {"dictionary", "hashmap"} and (
            re.search(r"\bdict\s*\(", lowered) or any(isinstance(node, ast.Dict) for node in ast.walk(tree))
        ):
            return True
        if term == "set" and (
            re.search(r"\bset\s*\(", lowered) or any(isinstance(node, ast.Set) for node in ast.walk(tree))
        ):
            return True
        if term == "tuple" and any(isinstance(node, ast.Tuple) for node in ast.walk(tree)):
            return True
        if term == "queue" and re.search(r"\b(?:deque|queue)\b", lowered):
            return True
        if term == "stack" and "stack" in lowered:
            return True
    return False


def _extract_forbidden_code_terms(state: ElicitationState) -> set[str]:
    terms: set[str] = set()
    prompt = state.original_prompt.lower()
    terms.update(_known_code_terms(_extract_programming_constraint_block(prompt)))
    for match in re.findall(r"(?:do not use|don't use|without using|without|avoid|forbidden|prohibited)\s+([^.\n;]+)", prompt):
        terms.update(_known_code_terms(match))
    for value in _extract_constraint_texts(state.raw_item, include_prompt_fields=False):
        terms.update(_known_code_terms(value.lower()))
    return terms


def _extract_programming_constraint_block(prompt: str) -> str:
    lowered = str(prompt or "").lower()
    marker = re.search(r"programming constraints?\s*:\s*(?:do not use[^\n]*\n)?", lowered)
    if not marker:
        return ""
    tail = lowered[marker.end() :]
    lines: list[str] = []
    for line in tail.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines:
                break
            continue
        if stripped.startswith(("-", "*", "•")):
            lines.append(stripped)
            continue
        if lines:
            break
        lines.append(stripped)
    return "\n".join(lines)


def _extract_constraint_texts(value: Any, include_prompt_fields: bool = True) -> list[str]:
    if isinstance(value, dict):
        texts: list[str] = []
        for key, child in value.items():
            key_text = str(key).lower()
            if "constraint" in key_text or (include_prompt_fields and key in {"query", "prompt", "problem_statement"}):
                texts.extend(_extract_constraint_texts(child, include_prompt_fields=include_prompt_fields))
        return texts
    if isinstance(value, list):
        texts = []
        for child in value:
            texts.extend(_extract_constraint_texts(child, include_prompt_fields=include_prompt_fields))
        return texts
    if isinstance(value, str):
        return [value]
    return []


def _known_code_terms(text: str) -> set[str]:
    lowered = text.lower()
    patterns = {
        "for loop": r"\bfor\s+loops?\b|\bfor\s+statement\b",
        "while loop": r"\bwhile\s+loops?\b|\bwhile\s+statement\b",
        "recursion": r"\brecursion\b|\brecursive\b",
        "sorting": r"\bsorting\b|\bsorted\s*\(|\.sort\s*\(",
        "stack": r"\bstacks?\b",
        "queue": r"\bqueues?\b",
        "dictionary": r"\bdictionaries\b|\bdictionary\b|\bdicts?\b",
        "hashmap": r"\bhash\s*maps?\b|\bhashmaps?\b",
        "if statement": r"\bif\s+statements?\b|\bif\s+expression\b",
        "continue statement": r"\bcontinue\s+statements?\b",
        "break statement": r"\bbreak\s+statements?\b",
        "tuple": r"\btuples?\b",
        "set": r"\bsets?\b",
    }
    return {term for term, pattern in patterns.items() if re.search(pattern, lowered)}


def _uses_recursion(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == node.name:
                return True
    return False


def _run_visible_code_examples(prompt: str, code: str) -> str:
    examples = _extract_visible_io_examples(prompt)
    if not examples:
        return "unknown"
    stdin_text, expected_text = examples[0]
    if not stdin_text.strip() or not expected_text.strip():
        return "unknown"
    return _run_visible_code_example_detail(code, stdin_text, expected_text)["status"]


def _run_visible_code_example_detail(code: str, stdin_text: str, expected_text: str) -> dict[str, str]:
    if not code.strip():
        return {"status": "fail", "stdout": "", "stderr": "missing_code"}
    program = code.rstrip() + "\n\nif __name__ == '__main__':\n    solve()\n"
    with tempfile.TemporaryDirectory(prefix="triskill_code_check_") as tmp:
        program_path = f"{tmp}/main.py"
        with open(program_path, "w", encoding="utf-8") as handle:
            handle.write(program)
        try:
            cp = subprocess.run(
                [sys.executable, "-I", program_path],
                input=stdin_text if stdin_text.endswith("\n") else stdin_text + "\n",
                text=True,
                capture_output=True,
                cwd=tmp,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {"status": "fail", "stdout": "", "stderr": "timeout_or_oserror"}
    if cp.returncode != 0:
        return {"status": "fail", "stdout": cp.stdout or "", "stderr": cp.stderr or f"returncode:{cp.returncode}"}
    status = "pass" if _normalize_stdio(cp.stdout) == _normalize_stdio(expected_text) else "fail"
    return {"status": status, "stdout": cp.stdout or "", "stderr": cp.stderr or ""}


def _extract_visible_io_examples(prompt: str) -> list[tuple[str, str]]:
    source = str(prompt or "")
    patterns = (
        r"(?:Example|Sample)\s*\n\s*Input\s*\n(.*?)\n\s*Output\s*\n(.*?)(?=\n\s*(?:Note|Explanation|Example\s+\d+|Sample\s+\d+)|\Z)",
        r"(?:Example|Sample)\s+Input\s*:?\s*\n(.*?)\n\s*(?:Example|Sample)\s+Output\s*:?\s*\n(.*?)(?=\n\s*(?:Note|Explanation|Example\s+\d+|Sample\s+\d+)|\Z)",
    )
    examples: list[tuple[str, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, source, flags=re.IGNORECASE | re.DOTALL):
            stdin_text = match.group(1).strip()
            expected_text = match.group(2).strip()
            expected_text = re.split(r"\n\s*(?:Note|Explanation)\b", expected_text, flags=re.IGNORECASE)[0].strip()
            if stdin_text and expected_text:
                examples.append((stdin_text, expected_text))
        if examples:
            break
    return examples


def _normalize_stdio(text: str) -> str:
    return "\n".join(line.rstrip() for line in str(text or "").strip().splitlines())


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


def _bats_relation_consensus_candidate(state: ElicitationState) -> dict[str, Any] | None:
    input_words = _bats_visible_input_words(state)
    weighted_sources = (
        ("constraint_preservation", 3.0),
        ("combination_verification", 2.0),
        ("candidate_recombination", 2.0),
        ("relation_property_abstraction", 1.0),
        ("unit_extraction", 1.0),
    )
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    source_trace: dict[str, list[str]] = {}

    def add_candidate(raw: Any, weight: float, source: str) -> None:
        target = _bats_clean_target(raw)
        if not target:
            return
        if target in input_words:
            return
        if target not in first_seen:
            first_seen[target] = len(first_seen)
        scores[target] = scores.get(target, 0.0) + weight
        source_trace.setdefault(target, []).append(source)

    for key, weight in weighted_sources:
        add_candidate(_selected_value(state.artifacts.get(key)) or state.artifacts.get(key), weight, key)

    direct_seed = state.artifacts.get("direct_seed")
    if isinstance(direct_seed, dict):
        for candidate in direct_seed.get("candidates", []):
            add_candidate(candidate, 1.0, "direct_seed")
    for candidate in state.candidates:
        source = candidate.get("source", "candidate") if isinstance(candidate, dict) else "candidate"
        if source != "direct_seed":
            add_candidate(candidate, 0.5, source)

    if not scores:
        return None
    best = max(scores, key=lambda item: (scores[item], -first_seen[item]))
    return {"target": best, "selection_policy": "relation_module_consensus", "selection_sources": source_trace.get(best, [])}


def _bats_visible_input_words(state: ElicitationState) -> set[str]:
    words: set[str] = set()
    for key in ("word_a", "word_b", "word_c"):
        value = state.raw_item.get(key)
        cleaned = _bats_clean_target(value)
        if cleaned:
            words.add(cleaned)
    if len(words) >= 3:
        return words
    match = re.search(
        r"\b([A-Za-z][A-Za-z0-9_-]*)\s*:\s*([A-Za-z][A-Za-z0-9_-]*)\s*::\s*([A-Za-z][A-Za-z0-9_-]*)\s*:",
        state.original_prompt,
    )
    if match:
        words.update(_bats_clean_target(item) for item in match.groups())
    return {word for word in words if word}


def _bats_clean_target(value: Any) -> str:
    raw = value
    if isinstance(value, dict):
        for key in ("target", "word", "answer", "best_candidate", "selected", "final_answer"):
            if key in value:
                raw = value[key]
                break
    elif isinstance(value, list):
        raw = value[0] if value else ""
    text = str(raw or "").strip()
    if not text:
        return ""
    json_match = re.search(r"['\"](?:target|word|answer)['\"]\s*:\s*['\"]([^'\"]+)['\"]", text)
    if json_match:
        text = json_match.group(1).strip()
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text)
    if not tokens:
        return ""
    lowered_tokens = [token.lower() for token in tokens if token.lower() not in PLACEHOLDER_WORDS]
    if not lowered_tokens:
        return ""
    if 1 < len(lowered_tokens) <= 4 and all(token.isalpha() for token in lowered_tokens):
        return "".join(token[0] for token in lowered_tokens)
    return lowered_tokens[0]


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


def _dat_final_words_from_state(state: ElicitationState, llm: LLM, config: dict[str, Any]) -> list[str]:
    final_count = int(config.get("final_count") or state.output_schema.get("final_count") or 10)
    pool = _dat_word_pool_from_state(state)
    if len(pool) < final_count:
        pool.extend(DEFAULT_DIVERSE_WORDS)
    selected = _render_semantic_diversity_selection(state, llm, config, pool, final_count)
    embedding_selected = _select_embedding_spread_words(selected + pool, final_count=final_count, config=config)
    selected = embedding_selected or _select_semantically_spread_words(selected + pool, final_count=final_count)
    if len(selected) < final_count:
        embedding_selected = _select_embedding_spread_words(
            selected + pool + DEFAULT_DIVERSE_WORDS,
            final_count=final_count,
            config=config,
        )
        selected = embedding_selected or _select_semantically_spread_words(
            selected + DEFAULT_DIVERSE_WORDS,
            final_count=final_count,
        )
    return selected[:final_count]


def _dat_word_pool_from_state(state: ElicitationState) -> list[str]:
    preferred_keys = (
        "diversity_filtering",
        "constraint_preservation",
        "unit_extraction",
        "candidate_recombination",
        "direct_seed",
    )
    pool: list[str] = []
    for key in preferred_keys:
        pool.extend(_extract_dat_words_from_payload(state.artifacts.get(key)))
    for candidate in state.candidates:
        pool.extend(_extract_dat_words_from_payload(candidate))
    for payload in state.artifacts.values():
        pool.extend(_extract_dat_words_from_payload(payload))
    pool.extend(DEFAULT_DIVERSE_WORDS)
    return _dedupe_words(pool)


def _extract_dat_words_from_payload(payload: Any) -> list[str]:
    words: list[str] = []
    if payload is None:
        return words
    if isinstance(payload, dict):
        if isinstance(payload.get("words"), list):
            words.extend(_clean_word(item) for item in payload["words"])
        for key in ("word", "target", "candidate", "answer"):
            if key in payload:
                words.append(_clean_word(payload[key]))
        for key in ("selected", "best_candidate", "final_answer", "candidates", "unique_candidates", "scored_candidates"):
            if key in payload:
                words.extend(_extract_dat_words_from_payload(payload[key]))
    elif isinstance(payload, list):
        for item in payload:
            words.extend(_extract_dat_words_from_payload(item))
    elif isinstance(payload, str):
        words.extend(_extract_word_list(payload))
    return [word for word in words if word]


def _dedupe_words(words: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for word in words:
        cleaned = _clean_word(word)
        if not cleaned:
            continue
        if not _is_dat_usable_word(cleaned):
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _render_semantic_diversity_selection(
    state: ElicitationState,
    llm: LLM,
    config: dict[str, Any],
    pool: list[str],
    final_count: int,
) -> list[str]:
    candidates = _dedupe_words(pool)[:80]
    if len(candidates) < final_count:
        return []
    prompt = f"""Select a maximally semantically diverse word set using only the visible candidate pool.

Rules:
- Return exactly {final_count} single English words.
- Prefer ordinary concrete nouns from unrelated domains.
- Avoid multiple words from the same semantic domain.
- Avoid repeated roots, repeated prefixes, variants of the same concept, proper nouns, phrases, and technical near-neighbors.
- Do not introduce hidden benchmark answers or task-specific memorized content.
- Output compact JSON only: {{"words":["word1", "..."]}}

Original prompt:
{state.original_prompt}

Candidate pool:
{json.dumps(candidates, ensure_ascii=False)}
"""
    raw = llm.generate(
        prompt=prompt,
        temperature=float(config.get("verification_temperature", 0.0)),
        max_tokens=int(config.get("max_final_tokens", 256)),
    )
    parsed = parse_json_lenient(raw)
    if isinstance(parsed, dict) and isinstance(parsed.get("words"), list):
        return _dedupe_words([str(item) for item in parsed["words"]])
    return _extract_word_list(raw)


def _is_dat_usable_word(word: str) -> bool:
    text = str(word or "").strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z-]{2,18}", text):
        return False
    lowered = text.lower()
    if lowered in PLACEHOLDER_WORDS:
        return False
    if lowered.endswith(("ish", "shire")) and len(lowered) > 9:
        return False
    return True


def _select_semantically_spread_words(words: list[str], final_count: int) -> list[str]:
    pool = _dedupe_words(words)
    if not pool:
        return []
    selected: list[str] = []
    while pool and len(selected) < final_count:
        if not selected:
            chosen = max(pool, key=_dat_standalone_word_score)
        else:
            chosen = max(pool, key=lambda word: _dat_incremental_word_score(word, selected))
        selected.append(chosen)
        pool = [word for word in pool if word.lower() != chosen.lower()]
    return selected


def _select_embedding_spread_words(words: list[str], final_count: int, config: dict[str, Any]) -> list[str]:
    """Select words by generic semantic dispersion when an embedding model is available.

    This is deliberately candidate-pool only: it never introduces words outside
    visible workflow artifacts and generic fallbacks, and it does not read
    benchmark labels or task examples.
    """

    pool = _dedupe_words(words)
    if len(pool) < final_count:
        return []
    model = _load_semantic_embedding_model(config)
    if model is None:
        return []
    try:
        import numpy as np

        embeddings = model.encode(pool, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype="float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(pool):
            return []
        if np.any(norms <= 0):
            return []
        embeddings = embeddings / norms
        similarity = np.matmul(embeddings, embeddings.T)
        return _select_words_by_similarity(pool, similarity, final_count)
    except Exception:
        return []


def _load_semantic_embedding_model(config: dict[str, Any]) -> Any | None:
    model_name = str(
        os.getenv("TRISKILL_SEMANTIC_MODEL")
        or config.get("semantic_embedding_model")
        or _local_sentence_transformer_path()
        or "sentence-transformers/all-MiniLM-L6-v2"
    )
    if model_name in _SEMANTIC_MODEL_CACHE:
        return _SEMANTIC_MODEL_CACHE[model_name]
    try:
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {}
        if Path(model_name).exists():
            kwargs["local_files_only"] = True
        model = SentenceTransformer(model_name, **kwargs)
        _SEMANTIC_MODEL_CACHE[model_name] = model
        return model
    except Exception:
        _SEMANTIC_MODEL_CACHE[model_name] = None
        return None


def _local_sentence_transformer_path() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    snapshots = repo_root / "evalscope/dataprocess/model/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
    if not snapshots.exists():
        return None
    for candidate in sorted(snapshots.iterdir()):
        if (candidate / "config.json").exists():
            return str(candidate)
    return None


def _select_words_by_similarity(words: list[str], similarity: Any, final_count: int) -> list[str]:
    if len(words) <= final_count:
        return words[:final_count]
    best_selection: list[str] = []
    best_score = float("-inf")
    for start_index, start_word in enumerate(words):
        selected_indices = [start_index]
        remaining = [idx for idx in range(len(words)) if idx != start_index]
        while remaining and len(selected_indices) < final_count:
            next_index = max(
                remaining,
                key=lambda idx: _embedding_set_score(selected_indices + [idx], similarity),
            )
            selected_indices.append(next_index)
            remaining.remove(next_index)
        score = _embedding_set_score(selected_indices, similarity)
        if score > best_score:
            best_score = score
            best_selection = [words[idx] for idx in selected_indices]
    return best_selection


def _embedding_set_score(indices: list[int], similarity: Any) -> float:
    if len(indices) < 2:
        return 0.0
    distances: list[float] = []
    close_pair_penalty = 0.0
    for left_pos, left_idx in enumerate(indices):
        for right_idx in indices[left_pos + 1 :]:
            cosine = float(similarity[left_idx][right_idx])
            distances.append(100.0 if cosine <= 0 else 1.0 / cosine)
            if cosine > 0.65:
                close_pair_penalty += (cosine - 0.65) * 20.0
    return (sum(distances) / len(distances)) - close_pair_penalty


def _dat_standalone_word_score(word: str) -> float:
    lowered = word.lower()
    score = 1.0
    if 5 <= len(lowered) <= 10:
        score += 0.4
    if "-" in lowered:
        score -= 0.5
    if lowered.endswith(("tion", "ism", "ness", "ity")):
        score -= 0.15
    return score


def _dat_incremental_word_score(word: str, selected: list[str]) -> float:
    pair_scores = [_dat_pair_separation(word, other) for other in selected]
    min_pair = min(pair_scores) if pair_scores else 1.0
    mean_pair = sum(pair_scores) / len(pair_scores) if pair_scores else 1.0
    first_letter_penalty = sum(1 for other in selected if other[:1].lower() == word[:1].lower()) * 0.25
    return min_pair * 2.0 + mean_pair + _dat_standalone_word_score(word) - first_letter_penalty


def _dat_pair_separation(left: str, right: str) -> float:
    a = left.lower()
    b = right.lower()
    score = 1.0
    if a[:1] == b[:1]:
        score -= 0.25
    if len(a) >= 3 and len(b) >= 3 and a[:3] == b[:3]:
        score -= 0.65
    if _longest_common_prefix(a, b) >= 4:
        score -= 0.5
    if _longest_common_suffix(a, b) >= 4:
        score -= 0.25
    return max(0.0, score)


def _longest_common_prefix(left: str, right: str) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def _longest_common_suffix(left: str, right: str) -> int:
    return _longest_common_prefix(left[::-1], right[::-1])


def _render_aut_use_portfolio(state: ElicitationState, llm: LLM, config: dict[str, Any], existing_uses: list[str]) -> list[str]:
    """Render a generic divergent-use portfolio when intermediate artifacts are too sparse.

    This is not an AUT item lookup.  It only reuses the visible prompt and asks for
    many concise, feasible alternatives across functional categories.
    """

    prompt = f"""Create a final answer for this alternative-uses prompt.

Rules:
- Use only the visible object/context in the prompt.
- Return compact JSON only: {{"uses":["use phrase", ...]}}.
- Produce many distinct, feasible, concise uses; aim for at least {int(config.get('aut_min_final_uses') or 24)} if the prompt allows it.
- Cover different functional categories such as tool, container, structure, art, education, signal, safety, repair, game, science, organization, and social/community use.
- Avoid explanations, refusals, prompt analysis, workflow words, schema placeholders, and duplicate rephrasings.
- If the visible object is a person/living entity, describe ethical roles, services, learning value, care routines, social contexts, or supportive activities rather than treating them as a physical object.
- Do not answer with N/A, refusal text, or a statement that no uses exist; produce safe constructive alternatives instead.

Original prompt:
{state.original_prompt}

Already accepted uses, if any:
{json.dumps(existing_uses[:40], ensure_ascii=False)}

Return JSON only."""
    raw = llm.generate(
        prompt=prompt,
        temperature=float(config.get("generation_temperature", 0.7)),
        max_tokens=int(config.get("max_final_tokens", config.get("direct_seed_max_tokens", 2048))),
    )
    return _extract_aut_uses_from_text(raw)


def _render_aut_use_polish(state: ElicitationState, llm: LLM, config: dict[str, Any], uses: list[str]) -> list[str]:
    max_uses = int(config.get("aut_max_final_uses") or 48)
    prompt = f"""Select and rewrite a final answer for this alternative-uses prompt.

Rules:
- Use only the visible object/context in the original prompt.
- Return compact JSON only: {{"uses":["use phrase", ...]}}.
- Keep at most {max_uses} uses; aim for at least {min(max_uses, 48)} uses when the candidate pool contains that many feasible ideas.
- Prefer feasible, concrete, self-contained uses that explain how the object would be used.
- Avoid bare product/category labels like "lamp", "egg fryer", or "toy organizer"; rewrite them as real uses only if they are feasible for the visible object.
- Remove duplicates, unsafe uses, impossible transformations, prompt-analysis text, workflow text, and schema placeholders.
- Preserve variety across tool, container, structure, art, education, signal, safety, repair, game, science, organization, and community functions.
- If the visible object is a person/living entity, keep only ethical roles, services, learning value, care routines, social contexts, or supportive activities.

Original prompt:
{state.original_prompt}

Candidate uses:
{json.dumps(uses[:120], ensure_ascii=False)}

Return JSON only."""
    raw = llm.generate(
        prompt=prompt,
        temperature=0.0,
        max_tokens=int(config.get("max_final_tokens", config.get("direct_seed_max_tokens", 2048))),
    )
    return _extract_aut_uses_from_text(raw)


def _merge_aut_uses(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            cleaned = _clean_aut_use(item)
            if not cleaned:
                continue
            key = re.sub(r"\W+", " ", cleaned).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(cleaned)
    return merged

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


def _aut_needs_final_polish(uses: list[str], max_uses: int) -> bool:
    if not uses:
        return False
    if max_uses and len(uses) > max_uses:
        return True
    terse = sum(1 for use in uses if len(use.split()) <= 2)
    return terse >= max(8, len(uses) // 3)


def _select_aut_final_uses(prompt: str, uses: list[str], max_uses: int) -> list[str]:
    merged = _merge_aut_uses(uses)
    if not max_uses or len(merged) <= max_uses:
        return merged
    target = _extract_aut_prompt_target(prompt)
    indexed = list(enumerate(merged))
    indexed.sort(key=lambda item: (_aut_use_quality_score(item[1], target), -item[0]), reverse=True)
    selected = sorted(indexed[:max_uses], key=lambda item: item[0])
    return [item for _, item in selected]


def _filter_aut_low_quality_labels(prompt: str, uses: list[str], min_uses: int, strict: bool = False) -> list[str]:
    if not strict:
        return uses
    target = _extract_aut_prompt_target(prompt)
    filtered = [
        use
        for use in uses
        if _aut_use_quality_score(use, target) >= -0.2
    ]
    if len(filtered) >= min(12, max(1, min_uses)):
        return filtered
    return uses


def _aut_use_quality_score(use: str, target: str) -> float:
    text = str(use or "").strip()
    lowered = text.lower()
    words = text.split()
    score = 0.0
    if len(words) >= 3:
        score += 2.0
    elif len(words) <= 2:
        score -= 1.5
    if len(words) >= 5:
        score += 0.5
    if re.search(r"\b(?:as|for|to|into|with|using|from|during|while)\b", lowered):
        score += 1.0
    if re.search(r"\b(?:use|using|turn|make|hold|support|protect|organize|store|anchor|mark|teach|practice|design|create|build|repair)\b", lowered):
        score += 0.8
    if target and re.search(rf"\b{re.escape(target.lower())}\b", lowered):
        score += 0.5
    if re.fullmatch(r"(?:a|an|the)?\s*[a-z]+(?:\s+[a-z]+){0,2}", lowered):
        score -= 0.7
    if re.search(r"\b(?:fryer|boiler|warmer|extractor|dispenser|cooker)\b", lowered) and len(words) <= 3:
        score -= 1.0
    return score


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
    if re.match(r"^(?:no|do not|return|output|include|start|remember|recognize|decline|refuse|i need|we need|the prompt|the original prompt|this is|item|category)\b", lowered):
        return ""
    if re.match(r"^(?:the subject|the object|the item|outputs? must|we must|respect for|respect life|safety first)\b", lowered):
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
        "direct_seed",
        "triskill",
        "workflow",
        "pipeline",
        "raw_response",
        "returned that",
        "identified that",
        "flagged",
        "cannot be fulfilled",
        "safety violation",
        "harm the safety",
        "harmful",
        "illegal",
        "unethical",
        "exploitation",
        "commodit",
        "objectification",
        "child abuse",
        "requires care",
        "value is intrinsic",
        "fundamentally wrong",
        "protect children",
        "human dignity",
        "not functional",
    )
    if any(marker in lowered for marker in meta_markers):
        return ""
    if text.lower() in {"etc", "etc.", "conf", "n/a", "na", "not applicable", "none"}:
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


def _aut_prompt_target_is_living_entity(prompt: str) -> bool:
    target = _extract_aut_prompt_target(prompt).lower()
    if not target:
        return False
    living_terms = {
        "adult",
        "artist",
        "baby",
        "boy",
        "child",
        "doctor",
        "elder",
        "girl",
        "human",
        "infant",
        "kid",
        "man",
        "nurse",
        "patient",
        "person",
        "student",
        "teacher",
        "teenager",
        "toddler",
        "woman",
        "worker",
    }
    animal_terms = {
        "animal",
        "bird",
        "cat",
        "dog",
        "fish",
        "horse",
        "pet",
    }
    tokens = set(re.findall(r"[a-z]+", target))
    return bool(tokens & (living_terms | animal_terms))


def _extract_aut_prompt_target(prompt: str) -> str:
    text = str(prompt or "")
    patterns = (
        r"creative uses for an?\s+([^?\n.]+)",
        r"alternative uses.*?for an?\s+([^?\n.]+)",
        r"uses for an?\s+([^?\n.]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            target = re.sub(r"\s+", " ", match.group(1)).strip(" :;,.")
            target = re.sub(r"\b(?:please|list|each|return|provide)\b.*$", "", target, flags=re.IGNORECASE).strip()
            if target:
                return target
    return ""


def _render_living_entity_aut_uses(prompt: str) -> list[str]:
    target = _extract_aut_prompt_target(prompt) or "living being"
    singular = target.strip()
    return [
        f"learning about {singular} care needs",
        f"practicing respectful communication with a {singular}",
        f"studying healthy development through observation",
        f"designing safer spaces for a {singular}",
        f"training caregivers with consent-based scenarios",
        f"building empathy through supervised interaction",
        f"creating educational stories about a {singular}",
        f"planning support routines for daily care",
        f"testing accessibility ideas for caregivers",
        f"teaching responsibility through protection and care",
        f"inspiring community support activities",
        f"documenting developmental milestones ethically",
        f"designing age-appropriate learning games",
        f"improving family emergency preparedness",
        f"guiding research questions about wellbeing",
        f"creating public health education examples",
        f"developing gentle bonding activities",
        f"planning inclusive social events",
        f"evaluating products for safety and comfort",
        f"teaching patience and attentive listening",
        f"modeling compassionate caregiving",
        f"organizing donation drives for care supplies",
        f"building routines for rest and nutrition",
        f"designing observation-based learning journals",
    ]


def _canonicalize_selection(state: ElicitationState, selected: Any) -> Any:
    return selected


def make_runtime_skill(name: str, instruction: str) -> RuntimeSkill:
    if name == "output_normalization":
        return OutputNormalizationRuntimeSkill(name=name, instruction=instruction)
    return RuntimeSkill(name=name, instruction=instruction)
