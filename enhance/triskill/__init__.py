"""TriSkill creativity elicitation framework.

This package is intentionally independent from evalscope.  It can wrap a
benchmark prompt with definition-guided, metric-conditioned elicitation
instructions while preserving the benchmark adapter's final answer format.
"""

from .core import TriSkillEnhancer, enhance_prompt
from .executor import WorkflowExecutor, build_artifact, build_state
from .evalscope_bridge import artifacts_to_predictions
from .execution_hooks import verify_math_solution, verify_python_code
from .llm import OpenAICompatibleLLM, parse_json_lenient
from .normalizer import normalize_selected
from .profiles import TaskProfile, profile_task
from .runner import run_dataset

__all__ = [
    "TaskProfile",
    "TriSkillEnhancer",
    "WorkflowExecutor",
    "OpenAICompatibleLLM",
    "artifacts_to_predictions",
    "build_artifact",
    "build_state",
    "enhance_prompt",
    "normalize_selected",
    "parse_json_lenient",
    "profile_task",
    "run_dataset",
    "verify_math_solution",
    "verify_python_code",
]
