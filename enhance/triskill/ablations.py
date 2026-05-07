"""Ablation prompt variants for TriSkill experiments."""

from __future__ import annotations


ABLATIONS: dict[str, str] = {
    "direct": "Use the original benchmark prompt without TriSkill instructions.",
    "generic_creativity_prompt": "Add a generic instruction to be creative and think outside the box.",
    "cot_structured": "Use generic structured reasoning without level-specific creativity workflows.",
    "high_temperature": "Use the original prompt with a higher generation temperature.",
    "multi_sample": "Sample multiple direct answers and select one without TriSkill skills.",
    "self_refine": "Draft, critique, and revise without definition-guided skill routing.",
    "triskill_full": "Use the full TriSkill workflow and metric-conditioned skills.",
    "triskill_level_only": "Use only the creativity-level workflow, without metric-conditioned skills.",
    "triskill_metric_only": "Use metric-conditioned skills without enforcing workflow order.",
    "triskill_without_verifier": "Remove verification, appropriateness, and residue-audit skills.",
    "triskill_wrong_skill_assignment": "Use deliberately mismatched skills for diagnostic experiments.",
    "triskill_length_matched": "Control final answer length to match TriSkill output.",
    "direct_long": "Use direct prompting but request a length comparable to TriSkill.",
    "budget_matched": "Match call or token budget without using TriSkill workflows.",
}


VERIFIER_SKILLS = {
    "relation_verification",
    "lexical_validity_check",
    "appropriateness_check",
    "coherence_check",
    "execution_verification",
    "old_world_residue_audit",
    "goal_coverage_check",
}


def apply_ablation_to_skill_names(skill_names: list[str], method: str) -> list[str]:
    if method == "triskill_without_verifier":
        return [name for name in skill_names if name not in VERIFIER_SKILLS]
    if method == "triskill_wrong_skill_assignment":
        return ["semantic_domain_expansion", "novelty_shift", "output_normalization"]
    return skill_names
