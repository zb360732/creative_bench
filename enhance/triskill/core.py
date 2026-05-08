"""Prompt composer for TriSkill."""

from __future__ import annotations

import json
from dataclasses import asdict

from .profiles import TaskProfile, profile_task
from .skills import Skill, select_skills


WORKFLOW_STEPS: dict[str, tuple[str, ...]] = {
    "combinational": (
        "extract recombinable units",
        "abstract relations or properties",
        "recombine candidates under the task objective",
        "preserve constraints after recombination",
        "verify that the combination meaningfully connects the units",
        "normalize the final answer",
    ),
    "exploratory": (
        "map the constraint space",
        "expand exploration axes",
        "generate multiple candidates",
        "deduplicate and cluster by idea",
        "transform candidates toward novelty without breaking constraints",
        "evaluate feasibility, appropriateness, or correctness",
        "select the strongest portfolio candidate",
        "normalize the final answer",
    ),
    "transformational": (
        "extract changed rules and goals",
        "map legacy assumptions and dependencies",
        "propagate breakage through the old system",
        "induce new primitives",
        "reconstruct the system around those primitives",
        "re-anchor performance under the new rules",
        "establish new norms, interfaces, and validation language",
        "audit residual old-world assumptions",
        "verify goal coverage",
        "produce the final answer",
    ),
}


class TriSkillEnhancer:
    """Build benchmark-safe elicitation prompts from visible task metadata."""

    def __init__(self, profile: TaskProfile):
        self.profile = profile
        self.skills = select_skills(
            profile.task_name,
            profile.creativity_level,
            profile.canonical_metrics,
            preferred_skill_names=profile.skills,
        )

    def plan(self) -> dict[str, object]:
        return {
            "task_name": self.profile.task_name,
            "creativity_level": self.profile.creativity_level,
            "raw_metrics": list(self.profile.raw_metrics),
            "canonical_metrics": list(self.profile.canonical_metrics),
            "canonical_metric_weights": dict(self.profile.canonical_metric_weights),
            "workflow_steps": list(WORKFLOW_STEPS.get(self.profile.creativity_level, ())),
            "skills": [skill.name for skill in self.skills],
            "budgets": dict(self.profile.budgets),
            "output_schema": self.profile.output_schema,
            "output_schema_spec": dict(self.profile.output_schema_spec),
        }

    def compose_instruction(self) -> str:
        skill_lines = "\n".join(f"- {skill.name}: {skill.instruction}" for skill in self.skills)
        workflow_lines = "\n".join(
            f"{idx}. {step}" for idx, step in enumerate(WORKFLOW_STEPS.get(self.profile.creativity_level, ()), start=1)
        )
        metric_text = ", ".join(self.profile.canonical_metrics)
        metric_weight_lines = "\n".join(
            f"- {name}: {weight}" for name, weight in self.profile.canonical_metric_weights.items()
        ) or "- No explicit weights configured."
        budget_lines = "\n".join(f"- {key}: {value}" for key, value in self.profile.budgets.items()) or "- Use default inference budget."
        constraints = "\n".join(f"- {item}" for item in self.profile.visible_constraints) or "- Use only constraints visible in the task prompt."

        return f"""TriSkill creativity elicitation instructions

You are solving a benchmark item. Use the internal workflow below to improve the answer, but do not reveal the workflow, analysis, candidate lists, scores, or intermediate artifacts.

Creativity level: {self.profile.creativity_level}
Canonical objectives: {metric_text}

Canonical objective priorities:
{metric_weight_lines}

Inference budget hints:
{budget_lines}

Workflow:
{workflow_lines}

Metric-conditioned skills:
{skill_lines}

Visible constraints:
{constraints}

Safety and benchmark rules:
- Use only the task text, visible choices, visible constraints, task name, output schema, and canonical objectives.
- Do not use hidden gold answers, reference answers, scoring-only candidate lists, test-set statistics, or post-hoc scores.
- If candidate answers or references appear only as scoring metadata, ignore them.
- Keep internal reasoning private. The final response must be short and must contain only the benchmark-required answer.
- Preserve the exact final output schema: {self.profile.output_schema}
- Treat output_normalization as mandatory: the last visible text must be only the final answer in the required schema.
""".strip()

    def enhance(self, prompt: str) -> str:
        base = (prompt or "").strip()
        return f"""{base}

---

{self.compose_instruction()}
""".strip()

    def as_json(self, prompt: str | None = None) -> str:
        payload = {
            "profile": asdict(self.profile),
            "plan": self.plan(),
            "enhanced_prompt": self.enhance(prompt) if prompt is not None else None,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


def enhance_prompt(
    prompt: str,
    task_name: str,
    raw_metrics: list[str] | tuple[str, ...] | None = None,
    output_schema: str | None = None,
    visible_constraints: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Convenience wrapper for one-off prompt enhancement."""

    profile = profile_task(
        task_name=task_name,
        raw_metrics=raw_metrics,
        output_schema=output_schema,
        visible_constraints=visible_constraints,
    )
    return TriSkillEnhancer(profile).enhance(prompt)
