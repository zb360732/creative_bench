"""Skill registry for TriSkill prompt composition."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Skill:
    name: str
    metrics: tuple[str, ...]
    instruction: str


OUTPUT_NORMALIZATION_INSTRUCTION = (
    "Normalize the selected answer to the benchmark adapter's exact final format. "
    "Do not include explanations, hidden artifacts, candidate pools, or validation notes in the final answer."
)


SKILLS: dict[str, Skill] = {
    "semantic_domain_expansion": Skill(
        name="semantic_domain_expansion",
        metrics=("semantic_diversity", "fluency"),
        instruction=(
            "Generate candidates from distant common semantic domains; avoid proper nouns, obscure terms, "
            "near-synonym clusters, and multi-word phrases unless the task explicitly allows them."
        ),
    ),
    "relation_abstraction": Skill(
        name="relation_abstraction",
        metrics=("relation_validity",),
        instruction=(
            "Infer the relation behind the task before answering. Consider part-whole, function, agent-tool, "
            "habitat, category, synonymy, antonymy, object-location, and morphology."
        ),
    ),
    "bridge_search": Skill(
        name="bridge_search",
        metrics=("associative_bridge", "relation_validity"),
        instruction=(
            "Search for a bridge candidate that connects every clue through common compounds, phrases, idioms, "
            "shared properties, prefixes, or suffixes; reject partial fits."
        ),
    ),
    "metaphorical_property_mapping": Skill(
        name="metaphorical_property_mapping",
        metrics=("metaphorical_fit", "relation_validity"),
        instruction=(
            "Identify the target meaning and implied properties in context, then choose a source-domain word "
            "that preserves meaning rather than merely sounding unusual."
        ),
    ),
    "relation_verification": Skill(
        name="relation_verification",
        metrics=("relation_validity", "appropriateness"),
        instruction=(
            "Verify candidates against the task relation. Do not reward novelty alone; prefer direct, common, "
            "strong relations over clever weak associations."
        ),
    ),
    "lexical_validity_check": Skill(
        name="lexical_validity_check",
        metrics=("lexical_validity", "format_validity"),
        instruction=(
            "Check that final word answers are common, valid, non-proper words with the requested part of speech "
            "and exact output field."
        ),
    ),
    "diversity_filtering": Skill(
        name="diversity_filtering",
        metrics=("semantic_diversity", "flexibility"),
        instruction=(
            "Remove duplicates, near-synonyms, and same-category clusters; preserve candidates that cover distinct domains."
        ),
    ),
    "constraint_parser": Skill(
        name="constraint_parser",
        metrics=("appropriateness", "constraint_satisfaction"),
        instruction=(
            "Separate hard constraints, soft preferences, forbidden outputs, output format, and degrees of creative freedom."
        ),
    ),
    "strategy_axis_expansion": Skill(
        name="strategy_axis_expansion",
        metrics=("flexibility", "novelty"),
        instruction=(
            "Explore different strategy axes such as conventional, minimal, cross-domain, reverse-assumption, "
            "mechanism-based, edge-case, aesthetic, algorithmic, social, physical, and symbolic."
        ),
    ),
    "candidate_multiplication": Skill(
        name="candidate_multiplication",
        metrics=("fluency", "flexibility"),
        instruction=(
            "Generate multiple distinct candidates from different strategies before selecting a final answer."
        ),
    ),
    "semantic_deduplication": Skill(
        name="semantic_deduplication",
        metrics=("fluency", "flexibility"),
        instruction=(
            "Drop candidates that share the same mechanism or differ only in wording; keep structurally distinct ideas."
        ),
    ),
    "category_coverage": Skill(
        name="category_coverage",
        metrics=("flexibility",),
        instruction="Ensure final candidates cover different categories, mechanisms, or solution structures.",
    ),
    "novelty_shift": Skill(
        name="novelty_shift",
        metrics=("novelty",),
        instruction=(
            "Increase novelty by using uncommon but valid mechanisms, distant-domain combinations, or non-obvious perspectives; "
            "do not make the answer infeasible."
        ),
    ),
    "appropriateness_check": Skill(
        name="appropriateness_check",
        metrics=("appropriateness", "constraint_satisfaction"),
        instruction=(
            "Reject candidates that violate hard constraints, are irrelevant, infeasible, incoherent, or sacrifice correctness for novelty."
        ),
    ),
    "coherence_check": Skill(
        name="coherence_check",
        metrics=("coherence",),
        instruction="Check causal flow, consistency, completeness, and absence of contradictions before finalizing.",
    ),
    "execution_verification": Skill(
        name="execution_verification",
        metrics=("execution_validity",),
        instruction=(
            "For code or math, reason through executable or symbolic checks. Do not claim tests passed unless actual execution is available."
        ),
    ),
    "pareto_selection": Skill(
        name="pareto_selection",
        metrics=("novelty", "flexibility", "appropriateness", "correctness"),
        instruction=(
            "Select the final answer by prioritizing hard constraints and correctness, then novelty, flexibility, clarity, and format compliance."
        ),
    ),
    "rule_parser": Skill(
        name="rule_parser",
        metrics=("rule_utilization", "goal_coverage"),
        instruction="Parse changed rules and goals, and explicitly use them to guide reconstruction.",
    ),
    "old_dependency_mapping": Skill(
        name="old_dependency_mapping",
        metrics=("rule_utilization", "old_assumption_removal"),
        instruction="Identify old assumptions, mechanisms, interfaces, measurements, and language that the legacy system relied on.",
    ),
    "breakage_propagation": Skill(
        name="breakage_propagation",
        metrics=("system_reconstruction", "old_assumption_removal"),
        instruction="Trace how changed rules break legacy modules, interfaces, incentives, measurements, and coordination routines.",
    ),
    "new_primitive_induction": Skill(
        name="new_primitive_induction",
        metrics=("system_reconstruction", "novelty"),
        instruction="Introduce new primitives, variables, roles, interfaces, or validity conditions required by the new rule world.",
    ),
    "architecture_reconstruction": Skill(
        name="architecture_reconstruction",
        metrics=("system_reconstruction", "interface_coordination"),
        instruction="Rebuild the core architecture around the new primitives instead of patching the old system superficially.",
    ),
    "performance_restoration": Skill(
        name="performance_restoration",
        metrics=("performance_restoration",),
        instruction="Explain how the rebuilt system restores key performance, reliability, throughput, safety, or interpretability goals.",
    ),
    "norm_establishment": Skill(
        name="norm_establishment",
        metrics=("norm_establishment", "cognitive_execution"),
        instruction="Define new standards, records, training language, validation procedures, and escalation norms.",
    ),
    "old_world_residue_audit": Skill(
        name="old_world_residue_audit",
        metrics=("old_assumption_removal",),
        instruction="Audit the answer for hidden reliance on old-world assumptions and revise those parts.",
    ),
    "goal_coverage_check": Skill(
        name="goal_coverage_check",
        metrics=("goal_coverage",),
        instruction="Verify that every stated goal is addressed by concrete mechanisms, not just mentioned.",
    ),
    "output_normalization": Skill(
        name="output_normalization",
        metrics=("format_validity",),
        instruction=OUTPUT_NORMALIZATION_INSTRUCTION,
    ),
}


WORKFLOW_SKILLS: dict[str, tuple[str, ...]] = {
    "combinational": (
        "semantic_domain_expansion",
        "relation_abstraction",
        "bridge_search",
        "metaphorical_property_mapping",
        "relation_verification",
        "lexical_validity_check",
        "diversity_filtering",
        "output_normalization",
        "pareto_selection",
    ),
    "exploratory": (
        "constraint_parser",
        "strategy_axis_expansion",
        "candidate_multiplication",
        "semantic_deduplication",
        "category_coverage",
        "novelty_shift",
        "appropriateness_check",
        "coherence_check",
        "execution_verification",
        "pareto_selection",
        "output_normalization",
    ),
    "transformational": (
        "rule_parser",
        "old_dependency_mapping",
        "breakage_propagation",
        "new_primitive_induction",
        "architecture_reconstruction",
        "performance_restoration",
        "norm_establishment",
        "old_world_residue_audit",
        "goal_coverage_check",
        "pareto_selection",
        "output_normalization",
    ),
}


TASK_SKILL_HINTS: dict[str, tuple[str, ...]] = {
    "dat": ("semantic_domain_expansion", "diversity_filtering", "lexical_validity_check"),
    "bats": ("relation_abstraction", "relation_verification", "lexical_validity_check"),
    "rat": ("bridge_search", "relation_verification", "lexical_validity_check"),
    "metaphor": ("metaphorical_property_mapping", "relation_verification", "lexical_validity_check"),
    "aut": (
        "constraint_parser",
        "strategy_axis_expansion",
        "candidate_multiplication",
        "semantic_deduplication",
        "category_coverage",
        "novelty_shift",
        "appropriateness_check",
    ),
    "creative_math": (
        "constraint_parser",
        "strategy_axis_expansion",
        "novelty_shift",
        "execution_verification",
        "appropriateness_check",
        "pareto_selection",
    ),
    "cs4": (
        "constraint_parser",
        "strategy_axis_expansion",
        "category_coverage",
        "novelty_shift",
        "coherence_check",
        "appropriateness_check",
    ),
    "neocoder": (
        "constraint_parser",
        "strategy_axis_expansion",
        "execution_verification",
        "appropriateness_check",
        "pareto_selection",
    ),
    "transformation": (
        "rule_parser",
        "old_dependency_mapping",
        "breakage_propagation",
        "new_primitive_induction",
        "architecture_reconstruction",
        "performance_restoration",
        "norm_establishment",
        "old_world_residue_audit",
        "goal_coverage_check",
    ),
}


def select_skills(
    task_name: str,
    level: str,
    canonical_metrics: tuple[str, ...],
    preferred_skill_names: tuple[str, ...] | list[str] = (),
) -> list[Skill]:
    """Select ordered skills from the task hints and canonical metrics."""

    metric_set = set(canonical_metrics)
    task_hints = tuple(preferred_skill_names) or TASK_SKILL_HINTS.get(task_name, ())
    candidate_names = list(task_hints)

    # For known benchmarks, keep the task-specific skill set tight.  Generic
    # metric overlap like relation_validity should not pull metaphor skills into
    # RAT or analogy skills into Metaphor.
    if task_hints:
        if "pareto_selection" not in candidate_names and metric_set.intersection({"novelty", "appropriateness"}):
            candidate_names.append("pareto_selection")
    else:
        for name in WORKFLOW_SKILLS.get(level, ()):
            skill = SKILLS[name]
            if metric_set.intersection(skill.metrics):
                candidate_names.append(name)

    candidate_names = [name for name in candidate_names if name != "output_normalization"]
    selected: list[Skill] = []
    seen: set[str] = set()
    for name in candidate_names:
        if name in SKILLS and name not in seen:
            selected.append(SKILLS[name])
            seen.add(name)
    selected.append(SKILLS["output_normalization"])
    return selected
