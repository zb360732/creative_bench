"""Workflow executors and artifacts for TriSkill."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from .core import TriSkillEnhancer
from .llm import LLM
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
