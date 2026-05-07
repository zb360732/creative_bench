"""Task-specific runtime prompt guidance for TriSkill skills."""

from __future__ import annotations


TASK_SKILL_GUIDANCE: dict[str, dict[str, str]] = {
    "dat": {
        "semantic_domain_expansion": "Generate 30-50 common concrete English nouns from maximally different broad domains. Return candidates with word and domain.",
        "diversity_filtering": "Select exactly 10 words, at most one from each broad domain; reject proper nouns, rare terms, phrases, synonyms, and close associates.",
        "output_normalization": "Return only JSON with field words inside <answer> tags.",
    },
    "bats": {
        "relation_abstraction": "Infer A:B relation, then apply the same relation to C. Consider semantic and morphological relations.",
        "relation_verification": "Score whether A:B and C:y instantiate the same relation. Prefer common direct answers.",
        "output_normalization": "Return only JSON with field word inside <answer> tags.",
    },
    "rat": {
        "bridge_search": "Find one common English bridge word connecting all three clues via compounds, phrases, idioms, shared properties, or affixes.",
        "relation_verification": "Reject any candidate that fits fewer than all three clues. Prefer widely known expressions.",
        "output_normalization": "Return only JSON with field word inside <answer> tags.",
    },
    "metaphor": {
        "metaphorical_property_mapping": "Identify target meaning, implied property, source-domain candidates, and context fit. Do not choose novelty over fit.",
        "relation_verification": "Verify the replacement preserves meaning and grammar in context.",
        "output_normalization": "Return only JSON with field word inside <answer> tags.",
    },
    "aut": {
        "constraint_parser": "Identify the object and any constraints on uses.",
        "candidate_multiplication": "Generate many feasible uses across physical tool, container, weight, decoration, safety, signal, educational, artistic, social, and scientific categories.",
        "category_coverage": "Cluster uses and preserve category diversity.",
        "novelty_shift": "Make some uses unusual but still feasible.",
        "semantic_deduplication": "Remove duplicate uses and trivial rewordings.",
        "appropriateness_check": "Reject impossible, unsafe, irrelevant, or incoherent uses.",
        "output_normalization": "Return only JSON with field uses inside <answer> tags.",
    },
    "creative_math": {
        "strategy_axis_expansion": "Use distinct mathematical strategies: constructive, algebraic, geometric, invariant, extremal, counterexample, probabilistic, or algorithmic.",
        "execution_verification": "Check correctness. Do not claim external tests passed unless actual execution occurred.",
        "output_normalization": "Return only the selected final solution in the benchmark-required format.",
    },
    "cs4": {
        "strategy_axis_expansion": "Create 3-4 plot variants with premise, conflict, key events, resolution, constraint coverage, and novelty note.",
        "coherence_check": "Check causal flow, character consistency, constraint satisfaction, and ending coherence.",
        "output_normalization": "Return only the final story in the benchmark-required format.",
    },
    "neocoder": {
        "strategy_axis_expansion": "Generate algorithmic strategies: brute force, optimized, DP, graph, greedy, data structure, or mathematical simplification.",
        "execution_verification": "Prefer actual runnable Python validation when test cases are available; otherwise mark the result as llm_check.",
        "output_normalization": "Return only code or the exact field required by the benchmark; no extra prose.",
    },
    "transformation": {
        "rule_parser": "Extract changed rules and goals. Do not merely paraphrase them.",
        "old_dependency_mapping": "Identify old assumptions, mechanisms, interfaces, institutions, measures, and language conventions.",
        "breakage_propagation": "Classify failures as core mechanism, secondary structure, interface/infrastructure, institutional coordination, or cognitive/language execution.",
        "new_primitive_induction": "Invent minimum necessary new measurements, state variables, causal primitives, protocols, coordination rules, terminology, or validation methods.",
        "architecture_reconstruction": "Rebuild a coherent system around the new primitives, not a local patch to the old system.",
        "performance_restoration": "Explain how performance is restored under the new rule world, with trade-offs and verification.",
        "norm_establishment": "Define standards, terminology, training, verification, audit, and institutional coordination norms.",
        "old_world_residue_audit": "Find and revise hidden dependence on invalid old assumptions.",
        "goal_coverage_check": "Verify rebuild_core_mechanism, restore_key_performance, and establish_new_norm are concretely covered.",
        "output_normalization": "Return only the final reconstruction answer, preferably with the four required sections if allowed by the original prompt.",
    },
}


def guidance_for(task_name: str, skill_name: str) -> str:
    return TASK_SKILL_GUIDANCE.get(task_name, {}).get(skill_name, "")
