"""Failure-mode diagnostics for TriSkill artifacts."""

from __future__ import annotations

import re
from typing import Any


TRANSFORMATION_FAILURE_MODES = {
    "rule_paraphrase": "Mostly repeats changed rules without reconstructing mechanisms.",
    "local_patching": "Only proposes local fixes without system reconstruction.",
    "old_world_residue": "Still relies on invalid old-world assumptions.",
    "no_new_primitives": "Does not introduce new abstractions, variables, interfaces, or mechanisms.",
    "missing_performance_restoration": "Does not explain how key performance is restored.",
    "missing_norm_establishment": "Does not establish standards, training, explanation, or verification norms.",
    "interface_neglect": "Ignores infrastructure, interfaces, or institutional coordination.",
    "cognitive_execution_neglect": "Ignores terminology, public understanding, training, or execution language.",
}


def diagnose_transformation(text: str, artifacts: dict[str, Any] | None = None) -> dict[str, Any]:
    lowered = (text or "").lower()
    artifacts = artifacts or {}
    findings: dict[str, bool] = {}
    findings["rule_paraphrase"] = _low_signal(lowered, ["mechanism", "architecture", "workflow", "module", "protocol"])
    findings["local_patching"] = _has_any(lowered, ["patch", "fix", "adjust", "add a rule"]) and not _has_any(lowered, ["rebuild", "reconstruct", "new primitive", "architecture"])
    findings["old_world_residue"] = _has_any(lowered, ["as before", "existing system", "keep the old", "unchanged assumption"])
    findings["no_new_primitives"] = not _has_any(lowered, ["new primitive", "new measurement", "new variable", "new interface", "new protocol", "new terminology"])
    findings["missing_performance_restoration"] = not _has_any(lowered, ["performance", "restore", "throughput", "reliability", "verification", "validate"])
    findings["missing_norm_establishment"] = not _has_any(lowered, ["standard", "norm", "training", "audit", "verification", "terminology"])
    findings["interface_neglect"] = not _has_any(lowered, ["interface", "infrastructure", "handoff", "coordination", "protocol"])
    findings["cognitive_execution_neglect"] = not _has_any(lowered, ["language", "terminology", "training", "public", "operator", "understanding"])
    return {
        "failure_modes": findings,
        "active_failure_modes": [name for name, active in findings.items() if active],
        "num_active_failure_modes": sum(1 for active in findings.values() if active),
        "artifact_keys": sorted(artifacts.keys()),
    }


def diagnose_artifact(row: dict[str, Any]) -> dict[str, Any]:
    if str(row.get("task_name", "")).lower() != "transformation":
        return {"failure_modes": {}, "active_failure_modes": [], "num_active_failure_modes": 0}
    return diagnose_transformation(str(row.get("final_answer") or ""), row.get("artifacts") or {})


def _has_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _low_signal(text: str, required_terms: list[str]) -> bool:
    words = re.findall(r"[a-zA-Z_]+", text)
    return len(words) < 80 or not _has_any(text, required_terms)
