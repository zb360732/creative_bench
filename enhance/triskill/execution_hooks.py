"""Optional execution verification hooks.

These hooks are conservative: if real execution is unavailable, they return an
unchecked result instead of pretending that a candidate passed.
"""

from __future__ import annotations

import ast
import subprocess
import tempfile
from pathlib import Path
from typing import Any


FORBIDDEN_CODE_MARKERS = ("import os", "from os", "import subprocess", "from subprocess", "import socket", "open(", "eval(", "exec(")


def verify_python_code(code: str, stdin: str | None = None, expected_stdout: str | None = None, timeout_s: int = 5) -> dict[str, Any]:
    if not code or not code.strip():
        return {"status": "invalid", "execution_pass": False, "reason": "empty code"}
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return {"status": "syntax_error", "execution_pass": False, "reason": str(exc)}
    lowered = code.lower()
    if any(marker in lowered for marker in FORBIDDEN_CODE_MARKERS):
        return {"status": "unsafe", "execution_pass": False, "reason": "forbidden operation marker"}
    if stdin is None and expected_stdout is None:
        return {"status": "syntax_only", "execution_pass": None, "reason": "no executable test case provided"}
    with tempfile.TemporaryDirectory(prefix="triskill_exec_") as tmp:
        program = Path(tmp) / "main.py"
        program.write_text(code, encoding="utf-8")
        try:
            result = subprocess.run(
                ["python", str(program)],
                input=stdin or "",
                text=True,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "execution_pass": False, "reason": "timeout"}
    actual = result.stdout.strip()
    if expected_stdout is None:
        return {"status": "executed", "execution_pass": result.returncode == 0, "stdout": actual, "stderr": result.stderr.strip()}
    return {
        "status": "executed",
        "execution_pass": result.returncode == 0 and actual == expected_stdout.strip(),
        "stdout": actual,
        "expected_stdout": expected_stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def verify_math_solution(solution: str) -> dict[str, Any]:
    # Placeholder for symbolic checks.  It is explicit about being unchecked.
    return {"status": "llm_self_check_required", "execution_pass": None, "reason": "no symbolic checker configured", "solution_length": len((solution or "").split())}
