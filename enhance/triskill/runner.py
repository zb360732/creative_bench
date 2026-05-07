"""Dataset runner for TriSkill methods and ablations."""

from __future__ import annotations

from typing import Any

from .ablations import apply_ablation_to_skill_names
from .core import TriSkillEnhancer
from .dataset import load_records, write_jsonl
from .diagnostics import diagnose_artifact
from .executor import WorkflowExecutor
from .llm import LLM
from .profiles import TaskProfile, profile_task
from .state import detect_leakage_fields, prompt_from_item, safe_item_view


def run_direct(task_name: str, item: dict[str, Any], llm: LLM, method: str = "direct") -> dict[str, Any]:
    prompt = prompt_from_item(item)
    final = llm.generate(prompt=prompt, temperature=0.7, max_tokens=1200).strip()
    leakage = detect_leakage_fields(item)
    return {
        "task_name": task_name,
        "item_id": str(item.get("id")) if item.get("id") is not None else None,
        "method": method,
        "safe_item": safe_item_view(item),
        "original_prompt": prompt,
        "enhanced_prompt": prompt,
        "final_answer": final,
        "parse_success": bool(final),
        "output_length": len(final.split()),
        "num_llm_calls": 1,
        "warnings": [f"excluded scoring-only fields: {', '.join(sorted(leakage))}"] if leakage else [],
    }


def run_generic_creativity(task_name: str, item: dict[str, Any], llm: LLM) -> dict[str, Any]:
    prompt = prompt_from_item(item) + "\n\nBe creative and think outside the box, while still following the required answer format."
    final = llm.generate(prompt=prompt, temperature=0.9, max_tokens=1200).strip()
    artifact = run_direct(task_name, item, llm=_StaticLLM(final), method="generic_creativity_prompt")
    artifact["enhanced_prompt"] = prompt
    return artifact


def run_cot_structured(task_name: str, item: dict[str, Any], llm: LLM) -> dict[str, Any]:
    prompt = prompt_from_item(item) + "\n\nThink through the constraints privately, then output only the final answer in the required format."
    final = llm.generate(prompt=prompt, temperature=0.7, max_tokens=1200).strip()
    artifact = run_direct(task_name, item, llm=_StaticLLM(final), method="cot_structured")
    artifact["enhanced_prompt"] = prompt
    return artifact


def run_triskill_method(task_name: str, item: dict[str, Any], llm: LLM | None, method: str) -> dict[str, Any]:
    profile = profile_task(task_name)
    if method == "triskill_without_verifier":
        profile = _profile_with_skills(profile, apply_ablation_to_skill_names(list(profile.skills), method))
    elif method == "triskill_wrong_skill_assignment":
        profile = _profile_with_skills(profile, apply_ablation_to_skill_names(list(profile.skills), method))
    executor = WorkflowExecutor(profile)
    if llm is None or method == "triskill_prompt_only":
        return executor.execute_prompt_only(item)
    return executor.execute(item, llm=llm, method=method)


def run_dataset(
    task_name: str,
    input_path: str,
    output_path: str,
    method: str = "triskill_prompt_only",
    llm: LLM | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    records = load_records(input_path)
    if limit is not None:
        records = records[:limit]
    rows: list[dict[str, Any]] = []
    for item in records:
        if method == "direct":
            if llm is None:
                raise ValueError("direct requires llm")
            rows.append(run_direct(task_name, item, llm=llm))
        elif method == "generic_creativity_prompt":
            if llm is None:
                raise ValueError("generic_creativity_prompt requires llm")
            rows.append(run_generic_creativity(task_name, item, llm=llm))
        elif method == "cot_structured":
            if llm is None:
                raise ValueError("cot_structured requires llm")
            rows.append(run_cot_structured(task_name, item, llm=llm))
        else:
            rows.append(run_triskill_method(task_name, item, llm=llm, method=method))
        if str(rows[-1].get("task_name", "")).lower() == "transformation":
            rows[-1]["diagnostics"] = diagnose_artifact(rows[-1])
    write_jsonl(output_path, rows)
    return rows


def _profile_with_skills(profile: TaskProfile, skill_names: list[str]) -> TaskProfile:
    return TaskProfile(
        task_name=profile.task_name,
        creativity_level=profile.creativity_level,
        workflow=profile.workflow,
        raw_metrics=profile.raw_metrics,
        canonical_metrics=profile.canonical_metrics,
        canonical_metric_weights=profile.canonical_metric_weights,
        output_schema=profile.output_schema,
        output_schema_spec=profile.output_schema_spec,
        skills=tuple(skill_names),
        budgets=profile.budgets,
        visible_constraints=profile.visible_constraints,
        metadata=profile.metadata,
    )


class _StaticLLM:
    def __init__(self, text: str):
        self.text = text

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        return self.text
