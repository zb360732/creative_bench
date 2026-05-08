"""Task profiling and metric abstraction for TriSkill."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


COMBINATIONAL_TASKS = {"dat", "bats", "rat", "metaphor"}
EXPLORATORY_TASKS = {"aut", "creative_math", "cs4", "neocoder"}
TRANSFORMATIONAL_TASKS = {"transformation", "transformational_creativity"}


RAW_METRIC_ALIASES: dict[str, list[str]] = {
    "dat": ["dat_semantic_distance"],
    "bats": ["bats_accuracy"],
    "rat": ["rat_accuracy"],
    "metaphor": ["metaphor_accuracy"],
    "aut": ["aut_fluency", "aut_elaboration", "aut_flexibility", "aut_originality"],
    "creative_math": ["correctness", "novelty", "creativity"],
    "cs4": ["constraint_satisfaction", "quality", "diversity", "appropriateness", "novelty"],
    "neocoder": ["correctness", "technique_diversity", "creativity"],
    "transformation": ["novelty", "appropriateness", "rule_coverage", "goal_coverage"],
}


RAW_TO_CANONICAL: dict[str, list[str]] = {
    "dat_semantic_distance": ["semantic_diversity", "lexical_validity", "format_validity"],
    "bats_accuracy": ["relation_validity", "lexical_validity", "format_validity"],
    "rat_accuracy": ["associative_bridge", "relation_validity", "lexical_validity", "format_validity"],
    "metaphor_accuracy": ["metaphorical_fit", "relation_validity", "lexical_validity", "format_validity"],
    "aut_fluency": ["fluency"],
    "aut_elaboration": ["elaboration"],
    "aut_flexibility": ["flexibility"],
    "aut_originality": ["novelty"],
    "originality": ["novelty"],
    "novelty": ["novelty"],
    "creativity": ["novelty", "flexibility", "appropriateness"],
    "correctness": ["appropriateness", "execution_validity"],
    "quality": ["coherence", "appropriateness"],
    "diversity": ["flexibility"],
    "appropriateness": ["appropriateness"],
    "constraint_satisfaction": ["constraint_satisfaction"],
    "technique_diversity": ["flexibility", "execution_validity"],
    "rule_coverage": ["rule_utilization"],
    "goal_coverage": ["goal_coverage"],
    "judge_score": ["rule_utilization", "system_reconstruction", "performance_restoration", "norm_establishment", "old_assumption_removal", "goal_coverage"],
    "rebuild_core_mechanism": ["system_reconstruction"],
    "restore_key_performance": ["performance_restoration"],
    "establish_new_norm": ["norm_establishment"],
    "execution": ["execution_validity"],
    "test_pass": ["execution_validity", "correctness"],
    "fluency": ["fluency"],
    "flexibility": ["flexibility"],
    "grammar": ["coherence"],
    "coherence": ["coherence"],
    "likability": ["story_quality", "appropriateness"],
    "QUC": ["constraint_satisfaction"],
    "RCS": ["constraint_satisfaction", "story_quality"],
}


TASK_CANONICAL_DEFAULTS: dict[str, list[str]] = {
    "dat": ["semantic_diversity", "lexical_validity", "format_validity"],
    "bats": ["relation_validity", "lexical_validity", "format_validity"],
    "rat": ["associative_bridge", "relation_validity", "lexical_validity", "format_validity"],
    "metaphor": ["metaphorical_fit", "relation_validity", "lexical_validity", "format_validity"],
    "aut": ["fluency", "flexibility", "novelty", "appropriateness", "elaboration", "format_validity"],
    "creative_math": ["novelty", "appropriateness", "constraint_satisfaction", "execution_validity"],
    "cs4": ["constraint_satisfaction", "coherence", "flexibility", "novelty", "appropriateness"],
    "neocoder": ["execution_validity", "constraint_satisfaction", "flexibility", "novelty"],
    "transformation": [
        "rule_utilization",
        "system_reconstruction",
        "performance_restoration",
        "norm_establishment",
        "old_assumption_removal",
        "interface_coordination",
        "cognitive_execution",
        "goal_coverage",
    ],
}


@dataclass(frozen=True)
class TaskProfile:
    """Visible task metadata used to select a workflow and skills."""

    task_name: str
    creativity_level: str
    workflow: str
    raw_metrics: tuple[str, ...]
    canonical_metrics: tuple[str, ...]
    canonical_metric_weights: Mapping[str, str] = field(default_factory=dict)
    output_schema: str = "preserve_original"
    output_schema_spec: Mapping[str, Any] = field(default_factory=dict)
    skills: tuple[str, ...] = field(default_factory=tuple)
    budgets: Mapping[str, Any] = field(default_factory=dict)
    visible_constraints: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


TASK_PROFILE_CONFIGS: dict[str, dict[str, Any]] = {
    "dat": {
        "raw_metrics": ["dat_semantic_distance"],
        "canonical_metric_weights": {"semantic_diversity": "high", "lexical_validity": "medium", "format_validity": "high"},
        "output_schema_spec": {"type": "json", "required_fields": ["words"], "final_count": 10},
        "skills": ["unit_extraction", "candidate_recombination", "diversity_filtering", "constraint_preservation", "output_normalization"],
        "budgets": {"candidate_count": 50, "final_count": 10, "generation_temperature": 0.9, "verification_temperature": 0.0},
    },
    "bats": {
        "raw_metrics": ["bats_accuracy"],
        "canonical_metric_weights": {"relation_validity": "high", "lexical_validity": "high", "format_validity": "high"},
        "output_schema_spec": {"type": "json", "required_fields": ["target"]},
        "skills": ["unit_extraction", "relation_property_abstraction", "candidate_recombination", "combination_verification", "constraint_preservation", "output_normalization"],
        "budgets": {"relation_hypotheses": 3, "candidates_per_relation": 3, "generation_temperature": 0.6, "verification_temperature": 0.0, "direct_seed_max_tokens": 1024, "direct_seed_samples": 1},
    },
    "rat": {
        "raw_metrics": ["rat_accuracy"],
        "canonical_metric_weights": {"associative_bridge": "high", "relation_validity": "high", "lexical_validity": "high", "format_validity": "high"},
        "output_schema_spec": {"type": "json", "required_fields": ["word"]},
        "skills": ["unit_extraction", "relation_property_abstraction", "candidate_recombination", "combination_verification", "constraint_preservation", "output_normalization"],
        "budgets": {"candidate_count": 30, "generation_temperature": 0.8, "verification_temperature": 0.0, "direct_seed_max_tokens": 1024, "direct_seed_samples": 1},
    },
    "metaphor": {
        "raw_metrics": ["metaphor_accuracy"],
        "canonical_metric_weights": {"metaphorical_fit": "high", "context_fit": "high", "relation_validity": "high", "lexical_validity": "high", "format_validity": "high"},
        "output_schema_spec": {"type": "json", "required_fields": ["word"]},
        "skills": ["unit_extraction", "relation_property_abstraction", "candidate_recombination", "combination_verification", "constraint_preservation", "output_normalization"],
        "budgets": {"candidate_count": 12, "generation_temperature": 0.7, "verification_temperature": 0.0, "direct_seed_max_tokens": 1024, "direct_seed_samples": 1},
    },
    "aut": {
        "raw_metrics": ["aut_fluency", "aut_flexibility", "aut_originality", "aut_elaboration"],
        "canonical_metric_weights": {"fluency": "high", "flexibility": "high", "novelty": "high", "elaboration": "medium", "appropriateness": "high", "format_validity": "high"},
        "output_schema_spec": {"type": "json", "required_fields": ["uses"]},
        "skills": ["constraint_space_mapping", "candidate_generation", "coverage_balancing", "novelty_transformation", "semantic_deduplication", "feasibility_evaluation", "output_normalization"],
        "budgets": {"candidate_count": 30, "final_count": None, "generation_temperature": 0.9, "verification_temperature": 0.0},
    },
    "creative_math": {
        "raw_metrics": ["fluency", "novelty", "flexibility", "appropriateness", "correctness"],
        "canonical_metric_weights": {"fluency": "medium", "novelty": "high", "flexibility": "high", "appropriateness": "high", "correctness": "high", "format_validity": "high"},
        "output_schema_spec": {"type": "solution_text"},
        "skills": ["constraint_space_mapping", "exploration_axis_expansion", "candidate_generation", "execution_verification", "novelty_transformation", "portfolio_selection", "output_normalization"],
        "budgets": {"strategy_count": 5, "candidates_per_strategy": 1, "generation_temperature": 0.8, "verification_temperature": 0.0},
    },
    "cs4": {
        "raw_metrics": ["fluency", "grammar", "coherence", "likability", "flexibility", "appropriateness", "novelty", "QUC", "RCS"],
        "canonical_metric_weights": {"novelty": "high", "appropriateness": "high", "coherence": "high", "constraint_satisfaction": "high", "story_quality": "high", "format_validity": "high"},
        "output_schema_spec": {"type": "story_text"},
        "skills": ["constraint_space_mapping", "exploration_axis_expansion", "candidate_generation", "novelty_transformation", "coherence_check", "feasibility_evaluation", "portfolio_selection", "output_normalization"],
        "budgets": {"plot_variants": 4, "generation_temperature": 0.9, "verification_temperature": 0.0},
    },
    "neocoder": {
        "raw_metrics": ["execution", "test_pass", "novelty", "flexibility", "appropriateness"],
        "canonical_metric_weights": {"execution_validity": "high", "correctness": "high", "appropriateness": "high", "novelty": "medium", "flexibility": "medium", "format_validity": "high"},
        "output_schema_spec": {"type": "code"},
        "skills": ["constraint_space_mapping", "exploration_axis_expansion", "candidate_generation", "execution_verification", "portfolio_selection", "output_normalization"],
        "budgets": {"strategy_count": 4, "candidates_per_strategy": 1, "generation_temperature": 0.7, "verification_temperature": 0.0},
    },
    "transformation": {
        "raw_metrics": ["judge_score", "rebuild_core_mechanism", "restore_key_performance", "establish_new_norm"],
        "canonical_metric_weights": {"rule_utilization": "high", "system_reconstruction": "high", "performance_restoration": "high", "norm_establishment": "high", "old_assumption_removal": "high", "interface_coordination": "high", "cognitive_execution": "high", "goal_coverage": "high", "format_validity": "high"},
        "output_schema_spec": {"type": "reconstruction_text", "sections": ["Rebuilt Core Mechanism", "Restoring Key Performance", "New Norms, Interfaces, and Verification", "Old-Assumption Safeguards"]},
        "skills": ["rule_change_extraction", "legacy_assumption_mapping", "breakage_propagation", "primitive_induction", "system_reconstruction", "performance_reanchoring", "norm_interface_establishment", "residue_audit", "goal_coverage_verification", "output_normalization"],
        "budgets": {"generation_temperature": 0.7, "verification_temperature": 0.0, "max_final_tokens": 1200},
    },
}


def normalize_task_name(task_name: str) -> str:
    normalized = (task_name or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "transformational_creativity": "transformation",
        "creative_math": "creative_math",
        "creativemath": "creative_math",
        "neo_coder": "neocoder",
    }
    return aliases.get(normalized, normalized)


def infer_level(task_name: str) -> str:
    name = normalize_task_name(task_name)
    if name in COMBINATIONAL_TASKS:
        return "combinational"
    if name in EXPLORATORY_TASKS:
        return "exploratory"
    if name in TRANSFORMATIONAL_TASKS:
        return "transformational"
    return "exploratory"


def canonicalize_metrics(task_name: str, raw_metrics: list[str] | tuple[str, ...] | None = None) -> tuple[str, ...]:
    name = normalize_task_name(task_name)
    raw = list(raw_metrics or RAW_METRIC_ALIASES.get(name, []))
    canonical: list[str] = []
    for metric in raw:
        for item in RAW_TO_CANONICAL.get(metric, [metric]):
            if item not in canonical:
                canonical.append(item)
    for item in TASK_CANONICAL_DEFAULTS.get(name, []):
        if item not in canonical:
            canonical.append(item)
    return tuple(canonical)


def infer_output_schema(task_name: str) -> str:
    name = normalize_task_name(task_name)
    schemas = {
        "dat": '<answer>{"words": ["word1", ..., "word10"]}</answer>',
        "bats": '<answer>{"target": "answer_word"}</answer>',
        "rat": '<answer>{"word": "connecting_word"}</answer>',
        "metaphor": '<answer>{"word": "replacement_word"}</answer>',
        "aut": '<answer>{"uses": ["use 1", "use 2", ...]}</answer>',
        "transformation": "preserve the benchmark prompt answer format, usually an <answer> block with prose or JSON",
    }
    return schemas.get(name, "preserve the original benchmark-required answer format")


def profile_task(
    task_name: str,
    raw_metrics: list[str] | tuple[str, ...] | None = None,
    output_schema: str | None = None,
    visible_constraints: list[str] | tuple[str, ...] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TaskProfile:
    """Create a task profile from visible benchmark metadata only."""

    name = normalize_task_name(task_name)
    config = TASK_PROFILE_CONFIGS.get(name, {})
    metrics = tuple(raw_metrics or config.get("raw_metrics") or RAW_METRIC_ALIASES.get(name, ()))
    canonical = canonicalize_metrics(name, metrics)
    weights = dict(config.get("canonical_metric_weights", {}))
    for item in canonical:
        weights.setdefault(item, "medium")
    return TaskProfile(
        task_name=name,
        creativity_level=infer_level(name),
        workflow=infer_level(name),
        raw_metrics=metrics,
        canonical_metrics=canonical,
        canonical_metric_weights=weights,
        output_schema=output_schema or infer_output_schema(name),
        output_schema_spec=dict(config.get("output_schema_spec", {})),
        skills=tuple(config.get("skills", ())),
        budgets=dict(config.get("budgets", {})),
        visible_constraints=tuple(visible_constraints or ()),
        metadata=metadata or {},
    )
