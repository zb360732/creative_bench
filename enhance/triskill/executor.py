"""Workflow executors and artifacts for TriSkill."""

from __future__ import annotations

from dataclasses import asdict
import re
from typing import Any, Mapping

from .core import TriSkillEnhancer
from .llm import LLM, parse_json_lenient
from .normalizer import PLACEHOLDER_WORDS
from .profiles import TaskProfile, profile_task
from .runtime_skills import make_runtime_skill
from .state import ElicitationState, detect_leakage_fields, item_id_from, prompt_from_item, safe_item_view


def build_state(task_name: str, item: Mapping[str, Any], profile: TaskProfile | None = None) -> ElicitationState:
    profile = profile or profile_task(task_name)
    safe_item = safe_item_view(item)
    warnings: list[str] = []
    leakage_fields = detect_leakage_fields(item)
    if leakage_fields:
        warnings.append(f"excluded scoring-only fields: {', '.join(sorted(leakage_fields))}")
    return ElicitationState(
        task_name=profile.task_name,
        item_id=item_id_from(item),
        raw_item=safe_item,
        original_prompt=prompt_from_item(item),
        level=profile.creativity_level,
        workflow=profile.workflow,
        raw_metrics=list(profile.raw_metrics),
        canonical_metrics=dict(profile.canonical_metric_weights),
        output_schema=dict(profile.output_schema_spec),
        visible_choices=None,
        warnings=warnings,
    )


class WorkflowExecutor:
    """Execute TriSkill workflows without mutating benchmark data."""

    def __init__(self, profile: TaskProfile):
        self.profile = profile
        self.enhancer = TriSkillEnhancer(profile)

    def execute_prompt_only(self, item: Mapping[str, Any]) -> dict[str, Any]:
        state = build_state(self.profile.task_name, item, self.profile)
        enhanced_prompt = self.enhancer.enhance(state.original_prompt)
        state.artifacts["enhanced_prompt"] = enhanced_prompt
        state.artifacts["plan"] = self.enhancer.plan()
        for idx, skill_name in enumerate(self.enhancer.plan()["skills"], start=1):
            state.skill_trace.append(
                {
                    "step": idx,
                    "skill": skill_name,
                    "mode": "prompt_instruction",
                    "artifacts_keys": list(state.artifacts.keys()),
                    "warnings": state.warnings[-3:],
                }
            )
        return _artifact_from_state(
            state=state,
            profile=self.profile,
            enhancer=self.enhancer,
            method="triskill_full_prompt_only",
            enhanced_prompt=enhanced_prompt,
            num_llm_calls=0,
        )

    def execute(self, item: Mapping[str, Any], llm: LLM, method: str = "triskill_full") -> dict[str, Any]:
        state = build_state(self.profile.task_name, item, self.profile)
        enhanced_prompt = self.enhancer.enhance(state.original_prompt)
        state.artifacts["enhanced_prompt"] = enhanced_prompt
        state.artifacts["plan"] = self.enhancer.plan()
        num_llm_calls = 0
        if self.profile.task_name in {"dat", "bats", "rat", "metaphor", "aut"}:
            seed_artifact = _direct_seed(self.profile.task_name, state.original_prompt, llm, dict(self.profile.budgets))
            state.artifacts["direct_seed"] = seed_artifact
            for candidate in seed_artifact.get("candidates", []):
                state.candidates.append(candidate)
            num_llm_calls += int(seed_artifact.get("num_calls") or 1)
        elif self.profile.output_schema_spec.get("type") in {"solution_text", "story_text", "code", "reconstruction_text"}:
            seed_artifact = _openended_direct_seed(self.profile.task_name, state.original_prompt, llm, dict(self.profile.budgets))
            state.artifacts["direct_seed"] = seed_artifact
            for candidate in seed_artifact.get("candidates", []):
                state.candidates.append(candidate)
            num_llm_calls += int(seed_artifact.get("num_calls") or 1)
        for idx, skill in enumerate(self.enhancer.skills, start=1):
            runtime = make_runtime_skill(skill.name, skill.instruction)
            before_artifact_keys = set(state.artifacts.keys())
            state = runtime.run(state, llm, dict(self.profile.budgets))
            num_llm_calls += 0 if skill.name == "output_normalization" and state.selected_candidate is not None else 1
            state.skill_trace.append(
                {
                    "step": idx,
                    "skill": skill.name,
                    "mode": "llm_json" if skill.name != "output_normalization" else "output_normalization",
                    "num_candidates": len(state.candidates),
                    "new_artifacts": sorted(set(state.artifacts.keys()) - before_artifact_keys),
                    "artifacts_keys": list(state.artifacts.keys()),
                    "warnings": state.warnings[-3:],
                }
            )
        return _artifact_from_state(
            state=state,
            profile=self.profile,
            enhancer=self.enhancer,
            method=method,
            enhanced_prompt=enhanced_prompt,
            num_llm_calls=num_llm_calls,
        )


def _artifact_from_state(
    state: ElicitationState,
    profile: TaskProfile,
    enhancer: TriSkillEnhancer,
    method: str,
    enhanced_prompt: str,
    num_llm_calls: int,
) -> dict[str, Any]:
    final_answer = state.final_answer or ""
    return {
        "task_name": state.task_name,
        "item_id": state.item_id,
        "method": method,
        "level": state.level,
        "workflow": state.workflow,
        "raw_metrics": state.raw_metrics,
        "canonical_metrics": state.canonical_metrics,
        "skills": enhancer.plan()["skills"],
        "budgets": dict(profile.budgets),
        "safe_item": dict(state.raw_item),
        "original_prompt": state.original_prompt,
        "enhanced_prompt": enhanced_prompt,
        "artifacts": state.artifacts,
        "skill_trace": state.skill_trace,
        "final_answer": final_answer,
        "parse_success": state.parse_success or bool(enhanced_prompt),
        "output_length": len(final_answer.split()) if final_answer else 0,
        "num_llm_calls": num_llm_calls,
        "warnings": state.warnings,
    }


def build_artifact(task_name: str, item: Mapping[str, Any]) -> dict[str, Any]:
    profile = profile_task(task_name)
    return WorkflowExecutor(profile).execute_prompt_only(item)


def run_triskill(task_name: str, item: Mapping[str, Any], llm: LLM, method: str = "triskill_full") -> dict[str, Any]:
    profile = profile_task(task_name)
    return WorkflowExecutor(profile).execute(item, llm=llm, method=method)


def profile_asdict(profile: TaskProfile) -> dict[str, Any]:
    return asdict(profile)


def _direct_seed(task_name: str, prompt: str, llm: LLM, config: dict[str, Any]) -> dict[str, Any]:
    task = task_name.lower()
    schemas = {
        "dat": '{"words":["word1","word2","word3","word4","word5","word6","word7","word8","word9","word10"]}',
        "aut": '{"uses":["use 1","use 2","use 3"]}',
        "bats": '{"target":"answer_word"}',
        "rat": '{"word":"connecting_word"}',
        "metaphor": '{"word":"replacement_word"}',
    }
    schema = schemas.get(task)
    if not schema:
        return {"candidates": [], "warning": "direct_seed unsupported task"}
    sample_count = max(1, int(config.get("direct_seed_samples", 1)))
    sample_temperatures = _seed_temperatures(config, sample_count)
    samples: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for sample_index, temperature in enumerate(sample_temperatures, start=1):
        seed_prompt = _direct_seed_prompt(prompt, schema, sample_index, sample_count)
        raw = llm.generate(
            prompt=seed_prompt,
            temperature=temperature,
            max_tokens=int(config.get("direct_seed_max_tokens", 160)),
        )
        parsed = parse_json_lenient(raw)
        sample_candidates = _seed_candidates_from_value(task, parsed if parsed is not None else raw)
        for candidate in sample_candidates:
            candidate["source"] = "direct_seed"
            candidate["seed_sample"] = sample_index
            candidates.append(candidate)
        samples.append(
            {
                "sample": sample_index,
                "temperature": temperature,
                "raw_response": raw[:1000],
                "parsed": parsed,
                "candidates": sample_candidates,
            }
        )
    return {"num_calls": len(samples), "samples": samples, "candidates": _rank_seed_candidates(task, candidates)}


def _openended_direct_seed(task_name: str, prompt: str, llm: LLM, config: dict[str, Any]) -> dict[str, Any]:
    """Capture a high-fidelity baseline answer for open-ended tasks.

    The workflow may explore variants, but open-ended benchmark quality depends
    heavily on a complete, coherent final response.  Keeping a direct seed gives
    the final normalizer a reliable anchor without using hidden scoring fields.
    """

    seed_prompt = _openended_seed_prompt(task_name, prompt)
    raw = llm.generate(
        prompt=seed_prompt,
        temperature=float(config.get("direct_seed_temperature", 0.0)),
        max_tokens=int(config.get("direct_seed_max_tokens", config.get("max_final_tokens", 1200))),
    )
    text = _clean_openended_seed(raw)
    return {
        "num_calls": 1,
        "samples": [{"sample": 1, "temperature": float(config.get("direct_seed_temperature", 0.0)), "raw_response": raw[:2000]}],
        "candidates": [{"source": "direct_seed", "text": text}],
    }


def _openended_seed_prompt(task_name: str, prompt: str) -> str:
    task = task_name.lower()
    if task == "creative_math":
        guidance = (
            "Give one complete, correct, self-contained mathematical solution. "
            "It should be meaningfully distinct from visible reference solutions while preserving correctness."
        )
    elif task == "cs4":
        guidance = (
            "Write one complete, coherent story that satisfies every visible constraint. "
            "Preserve grammar, causal flow, and ending coherence."
        )
    elif task == "neocoder":
        guidance = "Write the final code or exact benchmark-required programming answer. Do not add explanatory prose unless the prompt requires it."
    elif task == "transformation":
        guidance = (
            "Write one complete reconstruction plan that rebuilds the system under the changed rules, "
            "restores key performance, defines new norms/interfaces, and removes old invalid assumptions."
        )
    else:
        guidance = "Write the final answer required by the benchmark."
    return f"""{prompt}

Final answer anchor:
{guidance}
Return only the final answer. Do not output JSON unless the original prompt explicitly requires JSON. Do not describe this workflow.
""".strip()


def _clean_openended_seed(raw: str) -> str:
    text = str(raw or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    return text


def _direct_seed_prompt(prompt: str, schema: str, sample_index: int, sample_count: int) -> str:
    diversity_note = ""
    if sample_count > 1:
        diversity_note = (
            f"\nThis is independent attempt {sample_index} of {sample_count}. "
            "Use the same visible prompt, but check a different plausible relation path before committing."
        )
    portfolio_note = ""
    if '"uses"' in schema:
        portfolio_note = (
            "\nFor use-list prompts, produce a large portfolio of concise, feasible, diverse uses. "
            "Do not stop at the three schema examples; include as many distinct useful ideas as you can fit."
        )
    return f"""{prompt}

Give your best direct answer first. Return compact JSON only, using this schema:
{schema}
{diversity_note}
{portfolio_note}

For analogy-style prompts, preserve the relation direction, entity type, and abstraction level from the first pair when a word is ambiguous.
Do not explain. Do not use placeholder values from the schema. Put the final compact JSON on the last line if you need to think internally.
""".strip()


def _seed_temperatures(config: dict[str, Any], sample_count: int) -> list[float]:
    base = float(config.get("direct_seed_temperature", 0.0))
    spread = config.get("direct_seed_temperature_spread")
    if spread is None:
        spread = [base, 0.2, 0.4, 0.6]
    values = [float(item) for item in spread]
    if not values:
        values = [base]
    while len(values) < sample_count:
        values.append(values[-1])
    return values[:sample_count]


def _rank_seed_candidates(task: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    field = "target" if task == "bats" else "word"
    if task in {"dat", "aut"}:
        return candidates
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for index, candidate in enumerate(candidates):
        word = _clean_seed_word(candidate.get(field) or candidate.get("word") or candidate.get("target"))
        if not word:
            continue
        key = word.lower()
        counts[key] = counts.get(key, 0) + 1
        first_seen.setdefault(key, index)
    ranked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in sorted(
        candidates,
        key=lambda item: (
            -counts.get(_clean_seed_word(item.get(field) or item.get("word") or item.get("target")).lower(), 0),
            first_seen.get(_clean_seed_word(item.get(field) or item.get("word") or item.get("target")).lower(), 10**9),
        ),
    ):
        word = _clean_seed_word(candidate.get(field) or candidate.get("word") or candidate.get("target"))
        if not word:
            continue
        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        ranked_candidate = dict(candidate)
        ranked_candidate[field] = word
        ranked_candidate["seed_votes"] = counts[key]
        ranked.append(ranked_candidate)
    return ranked


def _seed_candidates_from_value(task: str, value: Any) -> list[dict[str, Any]]:
    if task == "dat":
        words: list[str] = []
        if isinstance(value, dict) and isinstance(value.get("words"), list):
            words = [_clean_seed_word(item) for item in value["words"]]
        words = [word for word in words if word]
        return [{"words": words, "source": "direct_seed"}] if words else []
    if task == "aut":
        uses: list[str] = []
        if isinstance(value, dict) and isinstance(value.get("uses"), list):
            uses = [str(item).strip() for item in value["uses"] if str(item).strip()]
        return [{"uses": uses, "source": "direct_seed"}] if uses else []
    field = "target" if task == "bats" else "word"
    word = ""
    if isinstance(value, dict):
        for key in (field, "word", "target", "answer", "best_candidate", "selected"):
            if key in value:
                word = _clean_seed_word(value[key])
                if word:
                    break
    if not word:
        word = _extract_seed_single_word(str(value))
    return [{field: word, "source": "direct_seed"}] if word else []


def _extract_seed_single_word(text: str) -> str:
    patterns = (
        r"['\"](?:word|target|answer)['\"]\s*:\s*['\"]([^'\"]+)['\"]",
        r"(?:final answer|answer|target|word)\s*(?:is|:)\s*['\"]?([A-Za-z][A-Za-z0-9_-]*)",
        r"(?:choose|select|use|go with)\s+['\"]?([A-Za-z][A-Za-z0-9_-]*)",
    )
    for pattern in patterns:
        for match in reversed(re.findall(pattern, text, flags=re.IGNORECASE)):
            word = _clean_seed_word(match)
            if word:
                return word
    return ""


def _extract_seed_words(text: str) -> list[str]:
    seen: set[str] = set()
    words: list[str] = []
    for match in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text):
        word = _clean_seed_word(match)
        if not word:
            continue
        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        words.append(word)
        if len(words) >= 10:
            break
    return words


def _clean_seed_word(value: Any) -> str:
    match = re.search(r"[A-Za-z][A-Za-z0-9_-]*", str(value or ""))
    if not match:
        return ""
    word = match.group(0)
    return "" if word.lower() in PLACEHOLDER_WORDS else word
