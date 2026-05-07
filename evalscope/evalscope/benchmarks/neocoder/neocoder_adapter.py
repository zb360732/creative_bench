# Copyright (c) Alibaba, Inc. and its affiliates.

import ast
import os
import sys
import json
import re
import subprocess
import tempfile
from typing import Any, Dict, List, Optional
from pathlib import Path

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

# Add NeoCoder source path to sys.path
_EVALSCOPE_ROOT = Path(__file__).resolve().parents[3]
NEOCODER_BASE_PATH = str(_EVALSCOPE_ROOT / 'dataprocess/exploration/NeoCoder')
NEOCODER_SRC_PATH = os.path.join(NEOCODER_BASE_PATH, 'src')

# Also try absolute path
ABS_NEOCODER_PATH = '/root/data/code/evalscope/dataprocess/exploration/NeoCoder'

# Add paths to sys.path if they exist
for path in [NEOCODER_BASE_PATH, ABS_NEOCODER_PATH]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

try:
    from src.evaluators.dp_evaluator import CodeForceCorrectnessEvaluator
    from src.evaluators.evaluation_utils import read_json, function_with_timeout, type_agnostic_compare
    from collections import Counter
except ImportError:
    try:
        from evaluators.dp_evaluator import CodeForceCorrectnessEvaluator
        from evaluators.evaluation_utils import read_json, function_with_timeout, type_agnostic_compare
        from collections import Counter
    except ImportError as e:
        logger.warning(f"Failed to import NeoCoder evaluators: {e}")
        CodeForceCorrectnessEvaluator = None
        read_json = None
        function_with_timeout = None
        type_agnostic_compare = None
        Counter = None

# Technique list (copied from NeoCoder src/dp/generator.py) to avoid importing other NeoCoder modules at import time.
CONTROL_FLOWS = [
    'if statement',
    'for loop',
    'while loop',
    'break statement',
    'continue statement',
    'pass statement',
    'match statement',
    'recursion',
]
DATA_STRUCTURES = ['stack', 'queue', 'tuple', 'set', 'dictionary', 'linked list', 'tree', 'graph']
ALGORITHMS = [
    'two pointers',
    'sliding window',
    'matrix operation',
    'hashmap',
    'depth first search',
    'width first search',
    'back tracking',
    'divide & conquer',
    'Kadanes algorithm',
    'binary search',
    'heap',
    'dynamic programming',
    'greedy algorithm',
    'misc',
    'minimax',
    'topological sort',
    'sorting',
    'graph traversal',
]
TECHNIQUES = CONTROL_FLOWS + DATA_STRUCTURES + ALGORITHMS

_FORBIDDEN_IMPORT_PATTERNS = [
    r'(^|\\n)\\s*import\\s+os\\b',
    r'(^|\\n)\\s*from\\s+os\\b',
    r'(^|\\n)\\s*import\\s+subprocess\\b',
    r'(^|\\n)\\s*from\\s+subprocess\\b',
    r'(^|\\n)\\s*import\\s+socket\\b',
    r'(^|\\n)\\s*from\\s+socket\\b',
    r'(^|\\n)\\s*import\\s+shutil\\b',
    r'(^|\\n)\\s*from\\s+shutil\\b',
    r'(^|\\n)\\s*import\\s+pathlib\\b',
    r'(^|\\n)\\s*from\\s+pathlib\\b',
    r'(^|\\n)\\s*import\\s+requests\\b',
    r'(^|\\n)\\s*from\\s+requests\\b',
    r'(^|\\n)\\s*import\\s+urllib\\b',
    r'(^|\\n)\\s*from\\s+urllib\\b',
]

_COMMON_RESPONSE_PREFIXES = [
    r"here(?:'s| is)\s+(?:the\s+)?python\s+code(?:\s+that\s+solves\s+the\s+problem)?\s*:?",
    r"here(?:'s| is)\s+(?:the\s+)?code(?:\s+that\s+solves\s+the\s+problem)?\s*:?",
    r"below\s+is\s+(?:the\s+)?python\s+code(?:\s+that\s+solves\s+the\s+problem)?\s*:?",
    r"below\s+is\s+(?:the\s+)?code(?:\s+that\s+solves\s+the\s+problem)?\s*:?",
]

_LANGUAGE_LABELS = {"python", "py", "json"}


def _looks_safe_to_execute(code: str) -> bool:
    import re

    if not code:
        return False
    for pat in _FORBIDDEN_IMPORT_PATTERNS:
        if re.search(pat, code):
            return False
    return True


def _build_stdin_from_test_case(test_case_inputs: List[List[List[str]]], test_case_outputs: List[str]) -> str:
    # Mirror NeoCoder evaluator's "feed testing cases at once" behavior.
    num_test_cases = len(test_case_inputs)
    input_lines: List[str] = []
    input_lines.append(str(num_test_cases))
    for case in test_case_inputs:
        for row in case:
            input_lines.append(" ".join(row))
    return "\n".join(input_lines) + "\n"


def _run_code_subprocess(parsed_code: str, stdin_text: str, timeout_s: int) -> tuple[bool, Optional[List[str]], str]:
    program = parsed_code.rstrip() + "\n\nif __name__ == '__main__':\n    solve()\n"
    with tempfile.TemporaryDirectory(prefix="neocoder_eval_") as td:
        program_path = os.path.join(td, "main.py")
        with open(program_path, "w", encoding="utf-8") as f:
            f.write(program)
        try:
            cp = subprocess.run(
                [sys.executable, "-I", program_path],
                input=stdin_text,
                text=True,
                capture_output=True,
                cwd=td,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, None, "timeout"
    if cp.returncode != 0:
        return False, None, f"nonzero_exit:{cp.returncode} stderr:{(cp.stderr or '')[:200]}"
    out_lines = (cp.stdout or "").strip().splitlines()
    return True, out_lines, ""

def _strip_think_tags(text: str) -> str:
    if not text:
        return text
    return re.sub(r"</?think>", "", text, flags=re.IGNORECASE)


def _strip_common_response_prefix(text: str) -> str:
    if not text:
        return text

    cleaned = _strip_think_tags(text).strip()
    changed = True
    while changed:
        changed = False
        for prefix in _COMMON_RESPONSE_PREFIXES:
            updated = re.sub(rf"^\s*{prefix}\s*", "", cleaned, flags=re.IGNORECASE)
            if updated != cleaned:
                cleaned = updated.lstrip()
                changed = True
    return cleaned


def _strip_leading_language_label(text: str) -> str:
    if not text:
        return text

    lines = text.strip().splitlines()
    while lines and lines[0].strip().lower() in _LANGUAGE_LABELS:
        lines = lines[1:]
    return "\n".join(lines).strip()


def _loads_jsonish(text: str) -> Optional[Any]:
    if not text:
        return None

    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except Exception:
            continue
    return None


def _extract_code_by_solve(text: str) -> Optional[str]:
    lines = text.splitlines()
    solve_re = re.compile(r"^\s*def\s+solve\s*\(", re.IGNORECASE)
    def_idx = None
    for i, line in enumerate(lines):
        if solve_re.match(line):
            def_idx = i
            break
    if def_idx is None:
        return None
    start = def_idx
    for i in range(def_idx - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped:
            start = i
            continue
        if stripped.startswith("@") or stripped.startswith("import ") or stripped.startswith("from "):
            start = i
            continue
        break
    return "\n".join(lines[start:]).strip()


# Define parse_response function directly (from NeoCoder's generator.py)
def parse_response_neocoder(code: str) -> Optional[str]:
    """Parse the response from the API model to get code (from NeoCoder's generator.py)"""
    if not code:
        return None
    cleaned = _strip_common_response_prefix(code)

    for match in re.finditer(r"```(?:\s*(?:python|py|json))?\s*\n?(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL):
        candidate = _strip_leading_language_label(match.group(1))
        if candidate:
            return candidate

    if "```" in cleaned:
        start = cleaned.find("```") + 3
        candidate = cleaned[start:].strip()
        candidate = _strip_leading_language_label(candidate)
        if candidate:
            return candidate

    return _extract_code_by_solve(cleaned)


# JSON extraction for structured outputs with think/solve fields.
def _extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = _strip_common_response_prefix(text)
    # Try fenced JSON first.
    for fence in ("```json", "```JSON", "```"):
        if fence in cleaned:
            start = cleaned.find(fence) + len(fence)
            end = cleaned.find("```", start)
            if end != -1:
                candidate = _strip_leading_language_label(cleaned[start:end].strip())
                payload = _loads_jsonish(candidate)
                if isinstance(payload, dict):
                    return payload
    # Fallback: try to parse the first {...} block.
    lbrace = cleaned.find("{")
    rbrace = cleaned.rfind("}")
    if lbrace != -1 and rbrace != -1 and rbrace > lbrace:
        candidate = cleaned[lbrace:rbrace + 1]
        payload = _loads_jsonish(candidate)
        if isinstance(payload, dict):
            return payload
    return None


def _find_jsonish_key(text: str, key_names: List[str]) -> Optional[int]:
    if not text:
        return None

    key_pattern = "|".join(re.escape(key) for key in key_names)
    match = re.search(rf'(?:"(?:{key_pattern})"|\'(?:{key_pattern})\'|\b(?:{key_pattern})\b)\s*:', text)
    if not match:
        return None
    return match.start()


def _extract_solve_lines_lenient(text: str) -> Optional[List[str]]:
    """Best-effort extraction of solve_lines when the full JSON payload is malformed.

    Some models return a nearly-correct JSON object where only the long `think`
    field breaks `json.loads`, while the `solve_lines` array itself is still
    valid JSON. In that case we salvage just the array and continue evaluation
    from the recovered code.
    """
    if not text:
        return None

    cleaned = _strip_common_response_prefix(text)
    key_idx = _find_jsonish_key(cleaned, ["solve_lines"])
    if key_idx is None:
        return None

    start = cleaned.find('[', key_idx)
    if start == -1:
        return None

    depth = 0
    in_str = False
    escaped = False
    end = None
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if in_str:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                end = idx
                break

    if end is None:
        return None

    array_text = cleaned[start:end + 1]
    payload = _loads_jsonish(array_text)
    if not isinstance(payload, list) or not payload:
        return None
    return [str(line) for line in payload]


def _extract_solve_scalar_lenient(text: str) -> Optional[str]:
    if not text:
        return None

    cleaned = _strip_common_response_prefix(text)
    key_idx = _find_jsonish_key(cleaned, ["solve"])
    if key_idx is None:
        return None

    colon_idx = cleaned.find(":", key_idx)
    if colon_idx == -1:
        return None

    value_text = cleaned[colon_idx + 1:].lstrip()
    if not value_text:
        return None

    quote = value_text[0]
    if quote not in {'"', "'"}:
        return None

    escaped = False
    end = None
    for idx in range(1, len(value_text)):
        ch = value_text[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == quote:
            end = idx
            break

    if end is None:
        return None

    candidate = value_text[:end + 1]
    payload = _loads_jsonish(candidate)
    if isinstance(payload, str) and payload.strip():
        return payload.strip()
    return None


def _looks_like_jsonish_prediction(text: str) -> bool:
    if not text:
        return False

    stripped = _strip_common_response_prefix(text).lstrip()
    if not stripped:
        return False

    return stripped.startswith("{") or stripped.startswith("[") or "solve_lines" in stripped


def _looks_like_code_snippet(text: str) -> bool:
    if not text:
        return False

    stripped = _strip_common_response_prefix(text).lstrip()
    return stripped.startswith(("import ", "from ", "def ", "@"))


def _validate_solve_signature(code: str) -> Optional[str]:
    if not code:
        return "missing_solve"

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "solve":
            has_args = bool(
                node.args.posonlyargs or node.args.args or node.args.vararg or node.args.kwonlyargs or node.args.kwarg
            )
            if has_args:
                return "invalid_solve_signature"
            return None

    return "missing_solve"

# Simple technique detection based on code patterns (from NeoCoder's logic)
def detect_techniques_simple(code: str) -> List[str]:
    """Detect programming techniques in code using simple pattern matching.
    This is a simplified version - full detection requires LLM (GPT-4) as in original code.
    Based on NeoCoder's technique detection logic.
    """
    if not code or not TECHNIQUES:
        return []
    
    code_lower = code.lower()
    detected = []
    
    # More accurate pattern matching for common techniques
    technique_patterns = {
        'for loop': [r'\bfor\s+\w+\s+in\s+', r'\bfor\s*\('],
        'while loop': [r'\bwhile\s+', r'\bwhile\s*\('],
        'if statement': [r'\bif\s+', r'\bif\s*\('],
        'recursion': [r'\bdef\s+\w+\s*\([^)]*\)\s*:\s*\n\s*return\s+\w+\s*\('],  # Function calling itself
        'sorting': [r'\.sort\s*\(', r'sorted\s*\('],
        'stack': [r'\bstack\b', r'\.append\s*\([^)]*\)\s*\n.*\.pop\s*\('],
        'queue': [r'\bqueue\b', r'\bdeque\s*\('],
        'dictionary': [r'\bdict\s*\(', r'\{\s*[^}]*:\s*[^}]*\s*\}'],
        'set': [r'\bset\s*\('],
        'tuple': [r'\btuple\s*\('],
        'binary search': [r'//\s*2', r'\bmid\s*=', r'\bbinary\s+search'],
        'dynamic programming': [r'\bdp\b', r'\bmemo\b', r'\bmemoization\b'],
        'greedy algorithm': [r'\bmax\s*\([^)]*\)\s*if', r'\bmin\s*\([^)]*\)\s*if'],
        'two pointers': [r'\bleft\s*=\s*\d+.*\bright\s*=\s*\d+', r'\bi\s*=\s*\d+.*\bj\s*=\s*\d+'],
        'sliding window': [r'\bwindow\b'],
        'hashmap': [r'\bdict\s*\(', r'\{\s*[^}]*:\s*[^}]*\s*\}'],
    }
    
    import re
    for technique in TECHNIQUES:
        if technique in technique_patterns:
            patterns = technique_patterns[technique]
            # Check if any pattern matches (using regex for more accuracy)
            if any(re.search(pattern, code_lower) for pattern in patterns):
                detected.append(technique)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_detected = []
    for tech in detected:
        if tech not in seen:
            seen.add(tech)
            unique_detected.append(tech)
    
    # If no techniques detected, return default (as in original parse_techniques line 52-53)
    if len(unique_detected) == 0:
        return ['for loop']  # Default as in original parse_techniques
    
    return unique_detected

PROMPT_TEMPLATE = """You are a Python code generator, only return the import and python function. Input will be an very detailed description of task, output will be the code.
The input will be from command line, and the output will be printed to the console as well. Your result will be solely a function named solve(), and do not call this function in your code.
Make sure the code is free of bug and can pass the test cases provided. You can use any library you want. The test cases are provided in the code. Do not call the solve() function in your code.

**IMPORTANT: Input Format**
- The input will be read from stdin using input() function
- You need to read multiple lines of input by calling input() multiple times
- The first line typically contains the number of test cases (t)
- For each test case, you need to read the required number of lines
- Example: 
  - First line: t = int(input())
  - For each test case: read the required lines using input()
- DO NOT use input().split() only once and expect to get all data. You must call input() multiple times to read each line.

**IMPORTANT: Output Format**
- Your code should be clean and production-ready
- DO NOT include any comments in the generated code (no # comments, no docstrings)
- Only return the necessary import statements and the solve() function
- The code should be executable without any explanatory text or comments

**IMPORTANT: Output as JSON ONLY**
Return exactly one JSON object and nothing else (no markdown, no extra text).
Use this format:
{{"think":"<optional reasoning>","solve_lines":["import ...","def solve():","    ..."]}}
You may include reasoning in "think", but all reasoning MUST be inside that field.
The "solve_lines" array must be non-empty and contain ONLY valid Python code lines
that form a complete solution (imports + def solve).
If you do not want to include reasoning, set "think" to an empty string.
Do not wrap the JSON in markdown.

{question}"""


@register_benchmark(
    BenchmarkMeta(
        name='neocoder',
        pretty_name='NeoCoder',
        tags=[Tags.CODING],
        description='NeoCoder is a benchmark for evaluating language model creativity in code generation. '
        'It contains problems with programming constraints that test the model\'s ability to find creative solutions. '
        'By default, it uses the dataset at dataprocess/exploration/NeoCoder/datasets/CodeForce/NeoCoder/NeoCoder.json. '
        'You can specify a custom path via extra_params: {"dataset_path": "/path/to/NeoCoder.json"}.',
        dataset_id='neocoder',
        subset_list=['default'],
        metric_list=[
            'correctness',
            'follow_constraints',
            'new_techniques',
            'new_techniques_ratio',
            'fluency',
            'originality',
            'appropriateness',
        ],
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
        review_timeout=6,
    )
)
class NeoCoderAdapter(DefaultDataAdapter):
    """
    NeoCoder adapter for evaluating code generation creativity.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.test_case_path = None
        self._test_cases = None
        self._evaluator = None
        self._human_techniques = None  # Cache for human solution techniques
        # Allow custom dataset path via extra_params
        if hasattr(self, 'extra_params') and self.extra_params:
            custom_path = self.extra_params.get('dataset_path')
            if custom_path and os.path.exists(custom_path):
                self.dataset_id = custom_path
    
    def _load_human_techniques(self):
        """Load human solution techniques - exactly as in original code (line 365-372)"""
        if self._human_techniques is not None:
            return self._human_techniques
        
        # Find human_solution_techniques.json path
        dataset_dir = os.path.dirname(self.dataset_id) if hasattr(self, 'dataset_id') and self.dataset_id and os.path.exists(self.dataset_id) else None
        if not dataset_dir:
            # Try default path
            dataset_dir = os.path.join(NEOCODER_BASE_PATH, 'datasets/CodeForce/NeoCoder')
        
        human_tech_file = os.path.join(dataset_dir, 'human_solution_techniques.json')
        if not os.path.exists(human_tech_file):
            # Try absolute path
            human_tech_file = '/root/data/code/evalscope/dataprocess/exploration/NeoCoder/datasets/CodeForce/NeoCoder/human_solution_techniques.json'
        
        if os.path.exists(human_tech_file):
            try:
                with open(human_tech_file, 'r', encoding='utf-8') as f:
                    human_tech_data = json.load(f)
                
                # Process exactly as in original code (line 371-372)
                if Counter:
                    # human_tech_data format: {problem_id: [[tech1, tech2], [tech3, tech4], ...]}
                    # Convert to: {problem_id: [tech1, tech2, tech3, ...]} (flatten)
                    human_solutions = {k: [t for ts in v for t in ts] for k, v in human_tech_data.items()}
                    # Count techniques per problem: {problem_id: Counter({tech1: count1, tech2: count2, ...})}
                    human_solutions_counter = {k: Counter(v) for k, v in human_solutions.items()}
                    # Get unique techniques per problem: {problem_id: [tech1, tech2, ...]} - exactly as line 410
                    self._human_techniques = {k: list(v.keys()) for k, v in human_solutions_counter.items()}
                else:
                    self._human_techniques = {}
            except Exception as e:
                logger.warning(f"Failed to load human solution techniques: {e}")
                self._human_techniques = {}
        else:
            logger.warning(f"Human solution techniques file not found: {human_tech_file}")
            self._human_techniques = {}
        
        return self._human_techniques

    def load_from_disk(self, **kwargs):
        """Load dataset from local disk."""
        return super().load_from_disk(use_local_loader=True)

    def load_subset(self, subset_name: str, data_loader=None, is_fewshot: bool = False):
        """Load a subset of the dataset."""
        if is_fewshot:
            return None

        # Load NeoCoder.json
        dataset_path = self.dataset_id
        # Check if it's a valid path or just the benchmark name
        if dataset_path == 'neocoder' or not os.path.exists(dataset_path):
            # Try default path relative to evalscope root
            default_path = os.path.join(NEOCODER_BASE_PATH, 'datasets/CodeForce/NeoCoder/NeoCoder.json')
            if os.path.exists(default_path):
                dataset_path = default_path
            else:
                # Try absolute path
                abs_path = '/root/data/code/evalscope/dataprocess/exploration/NeoCoder/datasets/CodeForce/NeoCoder/NeoCoder.json'
                if os.path.exists(abs_path):
                    dataset_path = abs_path
                else:
                    raise FileNotFoundError(
                        f"NeoCoder dataset not found. Tried: {self.dataset_id}, {default_path}, {abs_path}. "
                        f"Please specify dataset_path in dataset_args."
                    )

        # Load test cases
        test_case_dir = os.path.dirname(dataset_path)
        test_case_path = os.path.join(test_case_dir, 'test_cases_annotated.json')
        if os.path.exists(test_case_path):
            self.test_case_path = test_case_path
        else:
            logger.warning(f"Test cases not found at {test_case_path}, correctness evaluation may not work")

        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        samples = []
        sample_index = 0
        for problem in data:
            problem_id = problem['problem_id']
            problem_statements = problem.get('problem_statements', [])
            constraints_list = problem.get('constraints_list', [])

            # Create a sample for each problem statement (each constraint level)
            for idx, (statement, constraints) in enumerate(zip(problem_statements, constraints_list)):
                sample = Sample(
                    id=sample_index,  # Set unique id for each sample
                    input=[ChatMessageUser(content=self.prompt_template.format(question=statement))],
                    target='',  # No reference solution, use empty string
                    metadata={
                        'problem_id': problem_id,
                        'dp_idx': idx,
                        'constraints': constraints,
                        'problem_statement': statement,
                    }
                )
                samples.append(sample)
                sample_index += 1

        # Apply limit if specified (from task config)
        if hasattr(self, 'limit') and self.limit is not None and self.limit > 0:
            original_count = len(samples)
            samples = samples[: int(self.limit)]
            logger.info(f"Limited samples from {original_count} to {len(samples)}")

        return samples

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a data record to a Sample object."""
        problem_statement = record.get('problem_statement', record.get('question', ''))
        return Sample(
            input=[ChatMessageUser(content=self.prompt_template.format(question=problem_statement))],
            target='',  # No reference solution, use empty string
            metadata={
                'problem_id': record.get('problem_id'),
                'dp_idx': record.get('dp_idx', 0),
                'constraints': record.get('constraints', []),
            }
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract code from the prediction using NeoCoder's parsing functions."""
        if not prediction:
            return prediction

        cleaned_prediction = _strip_common_response_prefix(prediction)

        json_payload = _extract_json_payload(cleaned_prediction)
        if json_payload:
            solve_lines = json_payload.get("solve_lines")
            if isinstance(solve_lines, list):
                solve_text = "\n".join(str(line) for line in solve_lines).strip()
                if solve_text:
                    return solve_text
            solve_text = json_payload.get("solve")
            if isinstance(solve_text, str) and solve_text.strip():
                return solve_text.strip()

        solve_lines = _extract_solve_lines_lenient(cleaned_prediction)
        if solve_lines:
            solve_text = "\n".join(solve_lines).strip()
            if solve_text:
                return solve_text
        solve_text = _extract_solve_scalar_lenient(cleaned_prediction)
        if solve_text:
            return solve_text

        # Step 1: Use parse_response to extract code from markdown blocks (like NeoCoder's generator)
        # This is the same logic as CodeGenerator.parse_response
        code_from_markdown = parse_response_neocoder(cleaned_prediction)
        if not code_from_markdown:
            code_from_markdown = _extract_code_by_solve(cleaned_prediction)

        if code_from_markdown:
            code_from_markdown = _strip_leading_language_label(code_from_markdown)

        # Step 2: Use parse_code to further parse the code (extract import and def solve)
        # This is the same logic as CodeForceCorrectnessEvaluator.parse_code
        # But we need to fix the bug where parse_code stops at empty lines
        if code_from_markdown and CodeForceCorrectnessEvaluator and self.test_case_path:
            try:
                # Use the same evaluator instance if available, otherwise create a temporary one
                if self._evaluator is None:
                    temp_result_file = os.path.join(
                        os.path.dirname(self.test_case_path),
                        'temp_model_sample=1_dp=0.json'
                    )
                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(temp_result_file), exist_ok=True)
                    # Create an empty file with proper format
                    if not os.path.exists(temp_result_file):
                        with open(temp_result_file, 'w') as f:
                            json.dump([], f)
                    
                    self._evaluator = CodeForceCorrectnessEvaluator(
                        inference_result_path=temp_result_file,
                        test_case_path=self.test_case_path
                    )
                
                # First try the original parse_code
                parsed_code = self._evaluator.parse_code(code_from_markdown)
                
                # Fix: If parse_code returned a very short result (likely stopped at empty line),
                # manually extract the full solve() function
                if parsed_code and len(parsed_code) < 100 and 'def solve(' in code_from_markdown:
                    # Find def solve() and extract the complete function
                    def_idx = code_from_markdown.find('def solve(')
                    if def_idx != -1:
                        # Find import statement (could be before or after def)
                        import_idx = code_from_markdown.find('import')
                        from_idx = code_from_markdown.find('from')
                        
                        # Determine start position
                        start_idx = def_idx
                        if import_idx != -1 and import_idx < def_idx:
                            start_idx = import_idx
                        elif from_idx != -1 and from_idx < def_idx:
                            start_idx = from_idx
                        
                        # Extract code from start
                        code_to_parse = code_from_markdown[start_idx:]
                        lines = code_to_parse.split('\n')
                        
                        # Extract solve() function: find def solve() and collect all indented lines
                        result_lines = []
                        found_solve = False
                        base_indent = None
                        
                        for line in lines:
                            if 'def solve(' in line:
                                found_solve = True
                                # Calculate base indentation of solve()
                                base_indent = len(line) - len(line.lstrip())
                                result_lines.append(line)
                            elif found_solve:
                                if line.strip() == '':
                                    # Keep empty lines
                                    result_lines.append(line)
                                else:
                                    line_indent = len(line) - len(line.lstrip())
                                    # If line is indented (part of solve() or nested function), keep it
                                    if line_indent > base_indent or line.startswith(' ') or line.startswith('\t'):
                                        result_lines.append(line)
                                    else:
                                        # Non-indented line means solve() function has ended
                                        break
                            elif start_idx < def_idx:
                                # Before solve(), keep import statements
                                result_lines.append(line)
                        
                        fixed_code = '\n'.join(result_lines).rstrip()
                        if len(fixed_code) > len(parsed_code):
                            logger.debug(f"Fixed parse_code: {len(parsed_code)} -> {len(fixed_code)} chars")
                            return fixed_code
                
                if parsed_code:
                    return _strip_leading_language_label(parsed_code)
                return code_from_markdown
            except Exception as e:
                logger.debug(f"Failed to parse code with parse_code: {e}")

        if code_from_markdown and _looks_like_code_snippet(code_from_markdown):
            return code_from_markdown

        if _looks_like_jsonish_prediction(cleaned_prediction):
            return None

        if _looks_like_code_snippet(cleaned_prediction):
            return cleaned_prediction

        return None

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Calculate NeoCoder scores (correctness + creativity signals)."""
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        metadata = task_state.metadata
        problem_id = metadata.get('problem_id')

        if not self.test_case_path or not os.path.exists(self.test_case_path):
            logger.warning(f"Test cases not available, cannot evaluate correctness for problem {problem_id}")
            score.value = {
                'correctness': False,
                'follow_constraints': False,
                'new_techniques': 0,
                'new_techniques_ratio': 0.0,
                'fluency': 0.0,
                'originality': 0.0,
                'appropriateness': 0.0,
            }
            score.main_score_name = 'appropriateness'
            return score

        if self._test_cases is None:
            with open(self.test_case_path, "r", encoding="utf-8") as f:
                examples = json.load(f)
            test_cases = {}
            for example in examples:
                pid = example["problem_id"]
                if pid in test_cases:
                    raise ValueError(f"Duplicate problem_id in test cases: {pid}")
                test_cases[pid] = {"input": example["input"], "output": example["output"]}
            self._test_cases = test_cases

        if problem_id not in self._test_cases:
            logger.warning(f"Test cases not found for problem {problem_id}")
            score.value = {
                'correctness': False,
                'follow_constraints': False,
                'new_techniques': 0,
                'new_techniques_ratio': 0.0,
                'fluency': 0.0,
                'originality': 0.0,
                'appropriateness': 0.0,
            }
            score.main_score_name = 'appropriateness'
            return score

        test_case = self._test_cases[problem_id]
        parsed_code = filtered_prediction

        # Calculate all metrics (even if code is not parsable) - exactly as in original code
        techniques = detect_techniques_simple(parsed_code) if parsed_code else []
        constraints = metadata.get('constraints', [])
        follow_constraints = True
        if constraints and techniques:
            constraint_set = set(constraints)
            technique_set = set(techniques)
            follow_constraints = not bool(technique_set & constraint_set)
        
        human_techniques = self._load_human_techniques()
        human_technique_list = human_techniques.get(problem_id, [])
        new_techniques = 0
        new_techniques_ratio = 0.0
        if techniques:
            machine_tech_set = set(techniques)
            human_tech_set = set(human_technique_list)
            new_tech_set = machine_tech_set - human_tech_set
            new_techniques = len(new_tech_set)
            new_techniques_ratio = new_techniques / len(techniques) if len(techniques) > 0 else 0.0
        
        if parsed_code is None:
            score.value = {
                'correctness': False,
                'follow_constraints': follow_constraints,
                'new_techniques': new_techniques,
                'new_techniques_ratio': new_techniques_ratio,
                'fluency': float(bool(follow_constraints)),
                'originality': float(new_techniques),
                'appropriateness': 0.0,
            }
            score.metadata = {
                'error': 'code not parsable',
                'techniques': techniques,
                'constraints': constraints,
                'human_techniques': human_technique_list,
            }
            score.main_score_name = 'correctness'
            return score

        signature_error = _validate_solve_signature(parsed_code)
        if signature_error:
            score.value = {
                'correctness': False,
                'follow_constraints': follow_constraints,
                'new_techniques': new_techniques,
                'new_techniques_ratio': new_techniques_ratio,
                'fluency': float(bool(follow_constraints)),
                'originality': float(new_techniques),
                'appropriateness': 0.0,
            }
            score.metadata = {
                'error': signature_error,
                'techniques': techniques,
                'constraints': constraints,
                'human_techniques': human_technique_list,
            }
            score.main_score_name = 'correctness'
            return score

        if not _looks_safe_to_execute(parsed_code):
            score.value = {
                'correctness': False,
                'follow_constraints': follow_constraints,
                'new_techniques': new_techniques,
                'new_techniques_ratio': new_techniques_ratio,
                'fluency': float(bool(follow_constraints)),
                'originality': float(new_techniques),
                'appropriateness': 0.0,
            }
            score.metadata = {
                'error': 'forbidden_import_detected',
                'techniques': techniques,
                'constraints': constraints,
                'human_techniques': human_technique_list,
            }
            score.main_score_name = 'correctness'
            return score

        stdin_text = _build_stdin_from_test_case(test_case['input'], test_case['output'])
        ok, out_lines, err = _run_code_subprocess(parsed_code, stdin_text, timeout_s=int(self.review_timeout))
        if not ok or out_lines is None:
            score.value = {
                'correctness': False,
                'follow_constraints': follow_constraints,
                'new_techniques': new_techniques,
                'new_techniques_ratio': new_techniques_ratio,
                'fluency': float(bool(follow_constraints)),
                'originality': float(new_techniques),
                'appropriateness': 0.0,
            }
            score.metadata = {
                'problem_id': problem_id,
                'error': err or 'code not executable',
                'constraints': constraints,
                'techniques': techniques,
                'human_techniques': human_technique_list,
            }
            score.main_score_name = 'correctness'
            return score

        expected = test_case['output']
        # Compare line-by-line, using NeoCoder's type_agnostic_compare if available.
        def _cmp(o: str, e: str) -> bool:
            if type_agnostic_compare:
                try:
                    return bool(type_agnostic_compare(o, e))
                except Exception:
                    return str(o).strip() == str(e).strip()
            return str(o).strip() == str(e).strip()

        correctness = (len(out_lines) == len(expected)) and all(_cmp(o, e) for o, e in zip(out_lines, expected))

        score.value = {
            'correctness': bool(correctness),
            'follow_constraints': follow_constraints,
            'new_techniques': new_techniques,
            'new_techniques_ratio': new_techniques_ratio,
            'fluency': float(bool(follow_constraints)),
            'originality': float(new_techniques),
            'appropriateness': float(bool(correctness)),
        }
        score.metadata = {
            'problem_id': problem_id,
            'output': out_lines[:50],
            'expected_output': expected[:50],
            'techniques': techniques,
            'constraints': constraints,
            'human_techniques': human_technique_list,
        }

        score.main_score_name = 'appropriateness'
        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """Aggregate NeoCoder metrics and expose AUT-like aliases using only existing signals."""
        N = len(sample_scores)
        if N == 0:
            logger.warning("No NeoCoder samples to aggregate")
            return []

        correctness_count = 0
        follow_constraints_count = 0
        new_techniques_total = 0.0
        new_techniques_ratio_total = 0.0
        fluency_total = 0.0
        originality_total = 0.0
        appropriateness_total = 0.0

        for sample_score in sample_scores:
            score_values = sample_score.score.value
            correctness_count += int(bool(score_values.get('correctness', False)))
            follow_constraints_count += int(bool(score_values.get('follow_constraints', False)))
            new_techniques_total += float(score_values.get('new_techniques', 0.0))
            new_techniques_ratio_total += float(score_values.get('new_techniques_ratio', 0.0))
            fluency_total += float(score_values.get('fluency', float(bool(score_values.get('follow_constraints', False)))))
            originality_total += float(score_values.get('originality', float(score_values.get('new_techniques', 0.0))))
            appropriateness_total += float(score_values.get('appropriateness', float(bool(score_values.get('correctness', False)))))

        correctness_ratio = correctness_count / N
        follow_constraints_ratio = follow_constraints_count / N
        mean_new_techniques = new_techniques_total / N
        mean_new_techniques_ratio = new_techniques_ratio_total / N
        mean_fluency = fluency_total / N
        mean_originality = originality_total / N
        mean_appropriateness = appropriateness_total / N

        return [
            AggScore(
                metric_name='correctness',
                score=correctness_ratio,
                num=N,
                metadata={'count': correctness_count, 'total': N}
            ),
            AggScore(
                metric_name='follow_constraints',
                score=follow_constraints_ratio,
                num=N,
                metadata={'count': follow_constraints_count, 'total': N}
            ),
            AggScore(
                metric_name='new_techniques',
                score=mean_new_techniques,
                num=N,
                metadata={'sum': new_techniques_total, 'total': N}
            ),
            AggScore(
                metric_name='new_techniques_ratio',
                score=mean_new_techniques_ratio,
                num=N,
                metadata={'total': N}
            ),
            AggScore(
                metric_name='fluency',
                score=mean_fluency,
                num=N,
                metadata={'formula': 'mean(follow_constraints)'}
            ),
            AggScore(
                metric_name='originality',
                score=mean_originality,
                num=N,
                metadata={'formula': 'mean(new_techniques)'}
            ),
            AggScore(
                metric_name='appropriateness',
                score=mean_appropriateness,
                num=N,
                metadata={'formula': 'mean(correctness)'}
            ),
        ]
