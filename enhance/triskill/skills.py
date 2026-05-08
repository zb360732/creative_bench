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
    "unit_extraction": Skill(
        name="unit_extraction",
        metrics=("semantic_diversity", "relation_validity", "associative_bridge", "metaphorical_fit", "constraint_satisfaction"),
        instruction=(
            "Extract the visible units that can be recombined: words, concepts, entities, properties, relations, "
            "contexts, constraints, and required output fields."
        ),
    ),
    "relation_property_abstraction": Skill(
        name="relation_property_abstraction",
        metrics=("relation_validity", "associative_bridge", "metaphorical_fit", "semantic_diversity"),
        instruction=(
            "Abstract the relation, property, role, or semantic axis connecting the extracted units before generating "
            "new combinations."
        ),
    ),
    "candidate_recombination": Skill(
        name="candidate_recombination",
        metrics=("semantic_diversity", "relation_validity", "associative_bridge", "metaphorical_fit", "fluency"),
        instruction=(
            "Generate candidate recombinations by transferring relations, bridging distant units, mapping properties, "
            "or assembling semantically diverse units under the visible constraints."
        ),
    ),
    "constraint_preservation": Skill(
        name="constraint_preservation",
        metrics=("lexical_validity", "format_validity", "appropriateness", "constraint_satisfaction", "execution_validity"),
        instruction=(
            "Preserve hard constraints after recombination: relation direction, entity type, part of speech, context, "
            "grammar, feasibility, execution requirements, and output schema."
        ),
    ),
    "combination_verification": Skill(
        name="combination_verification",
        metrics=("relation_validity", "appropriateness"),
        instruction=(
            "Verify whether the recombined candidate meaningfully connects all required units. Do not reward novelty "
            "alone; prefer strong, direct, constraint-preserving combinations over clever weak associations."
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
    "constraint_space_mapping": Skill(
        name="constraint_space_mapping",
        metrics=("appropriateness", "constraint_satisfaction"),
        instruction=(
            "Map the exploration space by separating hard constraints, soft preferences, forbidden outputs, output format, "
            "and degrees of creative freedom."
        ),
    ),
    "exploration_axis_expansion": Skill(
        name="exploration_axis_expansion",
        metrics=("flexibility", "novelty"),
        instruction=(
            "Explore different strategy axes such as conventional, minimal, cross-domain, reverse-assumption, "
            "mechanism-based, edge-case, aesthetic, algorithmic, social, physical, and symbolic."
        ),
    ),
    "candidate_generation": Skill(
        name="candidate_generation",
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
    "coverage_balancing": Skill(
        name="coverage_balancing",
        metrics=("flexibility",),
        instruction="Ensure final candidates cover different categories, mechanisms, or solution structures.",
    ),
    "novelty_transformation": Skill(
        name="novelty_transformation",
        metrics=("novelty",),
        instruction=(
            "Increase novelty by using uncommon but valid mechanisms, distant-domain combinations, or non-obvious perspectives; "
            "do not make the answer infeasible."
        ),
    ),
    "feasibility_evaluation": Skill(
        name="feasibility_evaluation",
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
    "portfolio_selection": Skill(
        name="portfolio_selection",
        metrics=("novelty", "flexibility", "appropriateness", "correctness"),
        instruction=(
            "Select the final answer by prioritizing hard constraints and correctness, then novelty, flexibility, clarity, and format compliance."
        ),
    ),
    "rule_change_extraction": Skill(
        name="rule_change_extraction",
        metrics=("rule_utilization", "goal_coverage"),
        instruction="Parse changed rules and goals, and explicitly use them to guide reconstruction.",
    ),
    "legacy_assumption_mapping": Skill(
        name="legacy_assumption_mapping",
        metrics=("rule_utilization", "old_assumption_removal"),
        instruction="Identify old assumptions, mechanisms, interfaces, measurements, and language that the legacy system relied on.",
    ),
    "breakage_propagation": Skill(
        name="breakage_propagation",
        metrics=("system_reconstruction", "old_assumption_removal"),
        instruction="Trace how changed rules break legacy modules, interfaces, incentives, measurements, and coordination routines.",
    ),
    "primitive_induction": Skill(
        name="primitive_induction",
        metrics=("system_reconstruction", "novelty"),
        instruction="Introduce new primitives, variables, roles, interfaces, or validity conditions required by the new rule world.",
    ),
    "system_reconstruction": Skill(
        name="system_reconstruction",
        metrics=("system_reconstruction", "interface_coordination"),
        instruction="Rebuild the core architecture around the new primitives instead of patching the old system superficially.",
    ),
    "performance_reanchoring": Skill(
        name="performance_reanchoring",
        metrics=("performance_restoration",),
        instruction="Explain how the rebuilt system restores key performance, reliability, throughput, safety, or interpretability goals.",
    ),
    "norm_interface_establishment": Skill(
        name="norm_interface_establishment",
        metrics=("norm_establishment", "cognitive_execution"),
        instruction="Define new standards, records, training language, validation procedures, and escalation norms.",
    ),
    "residue_audit": Skill(
        name="residue_audit",
        metrics=("old_assumption_removal",),
        instruction="Audit the answer for hidden reliance on old-world assumptions and revise those parts.",
    ),
    "goal_coverage_verification": Skill(
        name="goal_coverage_verification",
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
        "unit_extraction",
        "relation_property_abstraction",
        "candidate_recombination",
        "constraint_preservation",
        "combination_verification",
        "lexical_validity_check",
        "diversity_filtering",
        "output_normalization",
        "portfolio_selection",
    ),
    "exploratory": (
        "constraint_space_mapping",
        "exploration_axis_expansion",
        "candidate_generation",
        "semantic_deduplication",
        "coverage_balancing",
        "novelty_transformation",
        "feasibility_evaluation",
        "coherence_check",
        "execution_verification",
        "portfolio_selection",
        "output_normalization",
    ),
    "transformational": (
        "rule_change_extraction",
        "legacy_assumption_mapping",
        "breakage_propagation",
        "primitive_induction",
        "system_reconstruction",
        "performance_reanchoring",
        "norm_interface_establishment",
        "residue_audit",
        "goal_coverage_verification",
        "portfolio_selection",
        "output_normalization",
    ),
}


TASK_SKILL_HINTS: dict[str, tuple[str, ...]] = {
    "dat": ("unit_extraction", "candidate_recombination", "diversity_filtering", "constraint_preservation"),
    "bats": ("unit_extraction", "relation_property_abstraction", "candidate_recombination", "combination_verification", "constraint_preservation"),
    "rat": ("unit_extraction", "relation_property_abstraction", "candidate_recombination", "combination_verification", "constraint_preservation"),
    "metaphor": ("unit_extraction", "relation_property_abstraction", "candidate_recombination", "combination_verification", "constraint_preservation"),
    "aut": (
        "constraint_space_mapping",
        "exploration_axis_expansion",
        "candidate_generation",
        "semantic_deduplication",
        "coverage_balancing",
        "novelty_transformation",
        "feasibility_evaluation",
    ),
    "creative_math": (
        "constraint_space_mapping",
        "exploration_axis_expansion",
        "novelty_transformation",
        "execution_verification",
        "feasibility_evaluation",
        "portfolio_selection",
    ),
    "cs4": (
        "constraint_space_mapping",
        "exploration_axis_expansion",
        "coverage_balancing",
        "novelty_transformation",
        "coherence_check",
        "feasibility_evaluation",
    ),
    "neocoder": (
        "constraint_space_mapping",
        "exploration_axis_expansion",
        "execution_verification",
        "feasibility_evaluation",
        "portfolio_selection",
    ),
    "transformation": (
        "rule_change_extraction",
        "legacy_assumption_mapping",
        "breakage_propagation",
        "primitive_induction",
        "system_reconstruction",
        "performance_reanchoring",
        "norm_interface_establishment",
        "residue_audit",
        "goal_coverage_verification",
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
        if "portfolio_selection" not in candidate_names and metric_set.intersection({"novelty", "appropriateness"}):
            candidate_names.append("portfolio_selection")
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
