"""Task-specific runtime prompt guidance for TriSkill skills."""

from __future__ import annotations


TASK_SKILL_GUIDANCE: dict[str, dict[str, str]] = {
    "dat": {
        "unit_extraction": "Identify the output units as common single English words and identify broad semantic domains that can be combined into a diverse set.",
        "candidate_recombination": "Generate 30-50 common single English nouns from maximally different broad domains. Return compact JSON only with candidates containing word and domain. Do not reason in prose.",
        "constraint_preservation": "Reject discourse words, proper nouns, rare terms, phrases, synonyms, close associates, and words that violate the required single-word format.",
        "diversity_filtering": "Select exactly 10 common single words, at most one from each broad domain. Prefer concrete nouns from unrelated domains such as astronomy, food, emotion, tool, animal, law, music, weather, medicine, finance. Reject discourse words, proper nouns, rare terms, phrases, synonyms, and close associates.",
        "output_normalization": "Return only JSON with field words inside <answer> tags.",
    },
    "bats": {
        "unit_extraction": "Extract A, B, and C from the analogy and note visible morphology or entity-type hints.",
        "relation_property_abstraction": "Infer A:B relation and direction. Preserve entity type and abstraction level from the first pair when a term is ambiguous.",
        "candidate_recombination": "Apply the abstracted relation to C to produce candidate targets.",
        "combination_verification": "Score whether A:B and C:y instantiate the same relation, direction, entity type, and abstraction level. Prefer common direct answers.",
        "constraint_preservation": "Keep the answer as one common target word in the requested target field.",
        "output_normalization": "Return only JSON with field word inside <answer> tags.",
    },
    "rat": {
        "unit_extraction": "Extract the three visible clues and their likely phrase, compound, idiom, property, prefix, or suffix affordances.",
        "relation_property_abstraction": "Abstract what kind of connection could bind all three clues through one bridge word.",
        "candidate_recombination": "Find common English bridge candidates connecting all three clues via compounds, phrases, idioms, shared properties, or affixes.",
        "combination_verification": "Reject any candidate that fits fewer than all three clues. Prefer widely known expressions.",
        "constraint_preservation": "Keep the answer as one common connecting word in the requested word field.",
        "output_normalization": "Return only JSON with field word inside <answer> tags.",
    },
    "metaphor": {
        "unit_extraction": "Extract the target word, local context, grammatical role, and implied meaning to preserve.",
        "relation_property_abstraction": "Identify the target meaning, implied property, and source-domain property that can be mapped into context.",
        "candidate_recombination": "Generate replacement candidates by mapping source-domain properties into the target context. Do not choose novelty over fit.",
        "combination_verification": "Verify the replacement preserves meaning and grammar in context.",
        "constraint_preservation": "Keep the answer as one replacement word in the requested word field.",
        "output_normalization": "Return only JSON with field word inside <answer> tags.",
    },
    "aut": {
        "constraint_space_mapping": "Identify the object and any constraints on uses.",
        "candidate_generation": "Generate many feasible uses across physical tool, container, weight, decoration, safety, signal, educational, artistic, social, and scientific categories.",
        "coverage_balancing": "Cluster uses and preserve category diversity.",
        "novelty_transformation": "Make some uses unusual but still feasible.",
        "semantic_deduplication": "Remove duplicate uses and trivial rewordings.",
        "feasibility_evaluation": "Reject impossible, unsafe, irrelevant, or incoherent uses.",
        "output_normalization": "Return only JSON with field uses inside <answer> tags.",
    },
    "creative_math": {
        "exploration_axis_expansion": "Use distinct mathematical strategies: constructive, algebraic, geometric, invariant, extremal, counterexample, probabilistic, or algorithmic.",
        "execution_verification": "Check correctness. Do not claim external tests passed unless actual execution occurred.",
        "output_normalization": "Return only the selected final solution in the benchmark-required format.",
    },
    "cs4": {
        "exploration_axis_expansion": "Create 3-4 plot variants with premise, conflict, key events, resolution, constraint coverage, and novelty note.",
        "coherence_check": "Check causal flow, character consistency, constraint satisfaction, and ending coherence.",
        "output_normalization": "Return only the final story in the benchmark-required format.",
    },
    "neocoder": {
        "exploration_axis_expansion": "Generate algorithmic strategies: brute force, optimized, DP, graph, greedy, data structure, or mathematical simplification.",
        "execution_verification": "Prefer actual runnable Python validation when test cases are available; otherwise mark the result as llm_check.",
        "output_normalization": "Return only code or the exact field required by the benchmark; no extra prose.",
    },
    "transformation": {
        "rule_change_extraction": "Extract changed rules and goals. Do not merely paraphrase them.",
        "legacy_assumption_mapping": "Identify old assumptions, mechanisms, interfaces, institutions, measures, and language conventions.",
        "breakage_propagation": "Classify failures as core mechanism, secondary structure, interface/infrastructure, institutional coordination, or cognitive/language execution.",
        "primitive_induction": "Invent minimum necessary new measurements, state variables, causal primitives, protocols, coordination rules, terminology, or validation methods.",
        "system_reconstruction": "Rebuild a coherent system around the new primitives, not a local patch to the old system.",
        "performance_reanchoring": "Explain how performance is restored under the new rule world, with trade-offs and verification.",
        "norm_interface_establishment": "Define standards, terminology, training, verification, audit, and institutional coordination norms.",
        "residue_audit": "Find and revise hidden dependence on invalid old assumptions.",
        "goal_coverage_verification": "Verify rebuild_core_mechanism, restore_key_performance, and establish_new_norm are concretely covered.",
        "output_normalization": "Return only the final reconstruction answer, preferably with the four required sections if allowed by the original prompt.",
    },
}


def guidance_for(task_name: str, skill_name: str) -> str:
    return TASK_SKILL_GUIDANCE.get(task_name, {}).get(skill_name, "")
