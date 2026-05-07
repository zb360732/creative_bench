# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.metrics.llm_judge import LLMJudge
from evalscope.benchmarks.judge_cache import BenchmarkJudgeCache
from evalscope.utils.logger import get_logger

logger = get_logger()

_DEFAULT_JUDGE_CONFIG_PATH = Path(__file__).resolve().parents[3] / 'run' / 'llm_judge.json'


def _resolve_judge_config(entry: Dict[str, Any]) -> Optional[Dict[str, str]]:
    if not isinstance(entry, dict):
        logger.warning(f"Invalid judge config entry: {entry}")
        return None
    api_url = str(entry.get('api_url') or '').strip()
    model_id = str(entry.get('model_id') or entry.get('model') or entry.get('name') or '').strip()
    api_key = str(entry.get('api_key', 'EMPTY'))
    api_key_env = entry.get('api_key_env')
    if api_key_env:
        api_key = os.getenv(str(api_key_env), api_key)
    if api_key in {'', 'YOUR_API_KEY'}:
        api_key = os.getenv('EVALSCOPE_API_KEY', api_key)
    if api_key in {'', 'YOUR_API_KEY'}:
        api_key = os.getenv('OPENAI_API_KEY', api_key)
    if not api_url or not model_id:
        logger.warning(f"Invalid judge config entry: {entry}")
        return None
    return {
        'api_url': api_url,
        'api_key': api_key,
        'model_id': model_id,
    }


def _load_default_judge_configs() -> List[Dict[str, str]]:
    if not _DEFAULT_JUDGE_CONFIG_PATH.exists():
        return []
    try:
        data = json.loads(_DEFAULT_JUDGE_CONFIG_PATH.read_text(encoding='utf-8'))
    except Exception as exc:
        logger.warning(f"Failed to read judge config: {exc}")
        return []
    models = data.get('models')
    if not isinstance(models, list) or not models:
        logger.warning(f"Invalid judge config, missing models list: {_DEFAULT_JUDGE_CONFIG_PATH}")
        return []
    configs = [_resolve_judge_config(entry) for entry in models]
    return [config for config in configs if config]


def _load_task_judge_configs(task_cfg: Any) -> List[Dict[str, str]]:
    if task_cfg is None:
        return []
    configs = getattr(task_cfg, 'judge_model_args_list', None)
    if isinstance(configs, list) and configs:
        resolved = [_resolve_judge_config(entry) for entry in configs]
        return [config for config in resolved if config]
    config = getattr(task_cfg, 'judge_model_args', None)
    if isinstance(config, dict) and config:
        resolved = _resolve_judge_config(config)
        return [resolved] if resolved else []
    return []


def _majority_threshold(count: int) -> int:
    return count // 2 + 1

PROMPTS_IMPORTED = False


# Fallback: Define prompt functions directly (from CreativeMath/src/prompts/prompts.py)
def load_novel_solution_generation_prompt(problem, solutions, k):
    """Generate prompt for novel solution generation"""
    # Provide the first k reference solutions
    first_k_solutions = solutions[:k]
    reference_solutions = "\n\n".join(
        [
            f"Solution {i + 1}:\n{solution}"
            for i, solution in enumerate(first_k_solutions)
        ]
    )

    prompt = f"""Criteria for evaluating the difference between two mathematical solutions include:
i). If the methods used to arrive at the solutions are fundamentally different, such as algebraic manipulation versus geometric reasoning, they can be considered distinct;
ii). Even if the final results are the same, if the intermediate steps or processes involved in reaching those solutions vary significantly, the solutions can be considered different;
iii). If two solutions rely on different assumptions or conditions, they are likely to be distinct;
iv). A solution might generalize to a broader class of problems, while another solution might be specific to certain conditions. In such cases, they are considered distinct;
v). If one solution is significantly simpler or more complex than the other, they can be regarded as essentially different, even if they lead to the same result.

Given the following mathematical problem:
{problem}

And some typical solutions:
{reference_solutions}

Please output a novel solution distinct from the given ones for this math problem."""

    return prompt


def load_correctness_evaluation_prompt(problem, solutions, new_solution):
    """Generate prompt for correctness evaluation"""
    # Provide two reference solutions if number of solutions more than one.
    if len(solutions) == 1:
        reference_solutions = f"Solution 1:\n{solutions[0]}"
    else:
        reference_solutions = "\n\n".join(
            [
                f"Solution {i + 1}:\n{solution}"
                for i, solution in enumerate(solutions[:2])
            ]
        )

    prompt = f"""Given the following mathematical problem:
{problem}

Reference solutions:
{reference_solutions}

New solution:
{new_solution}

Please respond ONLY with a JSON object in this exact format:
{{"verdict":"YES","error":""}}
or
{{"verdict":"NO","error":"<brief reason for the error>"}}

Output YES only if the new solution is mathematically correct and leads to the same result as the reference solutions.
Do not include any extra text."""

    return prompt


def load_coarse_grained_novelty_evaluation_prompt(problem, solutions, k, new_solution):
    """Generate prompt for coarse-grained novelty evaluation"""
    # Provide the first k reference solutions
    first_k_solutions = solutions[:k]
    reference_solutions = "\n\n".join(
        [
            f"Solution {i + 1}:\n{solution}"
            for i, solution in enumerate(first_k_solutions)
        ]
    )

    prompt = f"""Criteria for evaluating the novelty of a new mathematical solution include:
1. If the new solution used to arrive at the solutions is fundamentally different from reference solutions, such as algebraic manipulation versus geometric reasoning, it can be considered novel;
2. Even if the final results are the same, if the intermediate steps or processes involved in reaching those solutions vary significantly, the new solution can be considered novel;
3. If the new solution relies on different assumptions or conditions, it should be considered novel;
4. A solution might generalize to a broader class of problems, while another solution might be specific to certain conditions. In such cases, they are considered distinct;
5. If the new solution is significantly simpler or more complex than the others, it can be regarded as essentially novel, even if they lead to the same result.

Given the following mathematical problem:
{problem}

Reference solutions:
{reference_solutions}

New solution:
{new_solution}

Please respond ONLY with a JSON object in this exact format:
{{"verdict":"YES","error":""}}
or
{{"verdict":"NO","error":"<brief reason for the lack of novelty>"}}

Output YES only if the new solution is novel compared to the reference solutions.
Do not include any extra text."""

    return prompt


def load_fine_grained_novelty_evaluation_prompt(problem, solutions, k, new_solution):
    """Generate prompt for fine-grained novelty evaluation"""
    # Provide the (k+1)-th to n-th reference solutions
    remaining_solutions = solutions[k:]
    reference_solutions = "\n\n".join(
        [
            f"Solution {i + 1}:\n{solution}"
            for i, solution in enumerate(remaining_solutions)
        ]
    )

    prompt = f"""Criteria for evaluating the novelty of a new mathematical solution include:
1. If the new solution used to arrive at the solutions is fundamentally different from reference solutions, such as algebraic manipulation versus geometric reasoning, it can be considered novel;
2. Even if the final results are the same, if the intermediate steps or processes involved in reaching those solutions vary significantly, the new solution can be considered novel;
3. If the new solution relies on different assumptions or conditions, it should be considered novel;
4. A solution might generalize to a broader class of problems, while another solution might be specific to certain conditions. In such cases, they are considered distinct;
5. If the new solution is significantly simpler or more complex than the others, it can be regarded as essentially novel, even if they lead to the same result.

Given the following mathematical problem:
{problem}

Reference solutions:
{reference_solutions}

New solution:
{new_solution}

Please respond ONLY with a JSON object in this exact format:
{{"verdict":"YES","error":""}}
or
{{"verdict":"NO","error":"<brief reason for the lack of novelty>"}}

Output YES only if the new solution is novel compared to the reference solutions.
Do not include any extra text."""

    return prompt


def extract_yes_no(response):
    """Extract YES/NO from a structured JSON response."""
    if not response:
        return "NO"
    text = response.strip()
    # Try to extract JSON object if wrapped in code fences.
    if "```" in text:
        parts = re.findall(r"```(?:json)?\\s*([\\s\\S]*?)```", text, flags=re.IGNORECASE)
        if parts:
            text = parts[0].strip()
    try:
        data = json.loads(text)
    except Exception:
        # Fallback: try to locate a JSON object substring
        match = re.search(r"\{[\\s\\S]*\}", text)
        if not match:
            return "NO"
        try:
            data = json.loads(match.group(0))
        except Exception:
            return "NO"

    verdict = str(data.get("verdict", "")).strip().upper()
    return "YES" if verdict == "YES" else "NO"


def _has_answer_marker(text: str) -> bool:
    """Heuristic: detect if a truncated response still includes a final answer marker."""
    if not text:
        return False
    patterns = [
        r'\\boxed\{',
        r'\bfinal answer\b',
        r'\banswer\s*:',
        r'\bthe answer is\b',
        r'\bans\s*:',
        r'答案是',
        r'最终答案',
        r'因此答案',
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


@register_benchmark(
    BenchmarkMeta(
        name='creative_math',
        pretty_name='CreativeMath',
        tags=[Tags.MATH],
        description='CreativeMath benchmark for evaluating mathematical creativity through novel solution generation. '
        'Models are given k reference solutions and asked to generate a novel (k+1)th solution. '
        'Evaluation includes correctness, coarse-grained novelty, and fine-grained novelty assessments. '
        'By default, uses simplified evaluation mode. Set evaluation_mode="full" for multi-LLM evaluation.',
        dataset_id='creative_math',
        subset_list=['default'],
        metric_list=[
            'correctness_ratio',
            'novelty_ratio',
            'novel_unknown_ratio',
            'novelty_to_correctness_ratio',
            'novel_unknown_to_novelty_ratio',
            'originality',
            'appropriateness',
        ],
        eval_split='test',
        prompt_template='{question}',  # Dynamically constructed in _build_generation_prompt
        review_timeout=30,
        extra_params={
            'evaluation_mode': {
                'type': 'str',
                'description': 'Evaluation mode: "simplified" (fast, assumes correctness) or "full" (3 LLM evaluators)',
                'value': 'simplified'
            },
            'evaluator_models': {
                'type': 'list',
                'description': 'Models to use as evaluators in full mode',
                'value': ['claude-3-opus', 'gemini-1.5-pro', 'gpt-4']
            },
            'dataset_path': {
                'type': 'str',
                'description': 'Custom path to subset.json (optional)',
                'value': None
            },
            'judge_api_url': {
                'type': 'str',
                'description': 'API URL for LLM judges in full mode',
                'value': None
            },
            'judge_api_key': {
                'type': 'str',
                'description': 'API key for LLM judges in full mode',
                'value': 'EMPTY'
            },
            'judge_model_id': {
                'type': 'str',
                'description': 'Model ID for LLM judges in full mode',
                'value': None
            },
            'judge_temperature': {
                'type': 'float',
                'description': 'Sampling temperature for LLM judges in full mode',
                'value': 0.0
            }
        }
    )
)
class CreativeMathAdapter(DefaultDataAdapter):
    """
    CreativeMath adapter for evaluating mathematical creativity.

    This benchmark tests models' ability to generate novel mathematical solutions.
    For each problem with n reference solutions, the model is tested on k=1,2,...,n:
    - Given k reference solutions, generate a novel (k+1)th solution
    - Evaluate: correctness, coarse-grained novelty (vs k solutions),
      fine-grained novelty (vs remaining solutions)

    Supports two evaluation modes:
    - simplified (default): Fast, assumes correctness and novelty for testing
    - full: Uses 3 LLM evaluators (Claude-3-Opus, Gemini-1.5-Pro, GPT-4) for accurate evaluation
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Setup CreativeMath paths
        self.creative_math_base_path = os.path.join(
            os.path.dirname(__file__),
            '../../../dataprocess/exploration/CreativeMath'
        )
        self.creative_math_base_path = os.path.abspath(self.creative_math_base_path)

        # Get evaluation mode from extra_params
        self.evaluation_mode = 'simplified'  # default
        if hasattr(self, 'extra_params') and self.extra_params:
            self.evaluation_mode = self.extra_params.get('evaluation_mode', 'simplified')
            # Allow custom dataset path
            custom_path = self.extra_params.get('dataset_path')
            if custom_path and os.path.exists(custom_path):
                self.dataset_id = custom_path

        logger.info(f"CreativeMath evaluation mode: {self.evaluation_mode}")

        # Initialize LLM judges for full mode
        task_cfg = getattr(self, '_task_config', None)
        self._judge_cache = BenchmarkJudgeCache(
            benchmark_name='creative_math',
            work_dir=getattr(task_cfg, 'work_dir', None),
            model_name=getattr(task_cfg, 'model_id', None),
        )
        self._llm_judges = None
        self.evaluator_model_names = ['evaluator1', 'evaluator2', 'evaluator3']
        self.evaluator_display_names = ['evaluator1', 'evaluator2', 'evaluator3']
        if self.evaluation_mode == 'full':
            if hasattr(self, 'extra_params') and self.extra_params:
                custom_evaluators = self.extra_params.get('evaluator_models')
                if custom_evaluators:
                    self.evaluator_model_names = custom_evaluators
            self._init_llm_judges()
            task_cfg = getattr(self, '_task_config', None)
            self.use_batch_scoring = False

    def _init_llm_judges(self):
        """Initialize LLM judges for full evaluation mode."""
        if self.evaluation_mode != 'full':
            return

        self._llm_judges = {}
        judge_configs: List[Dict[str, str]] = []
        judge_temperature = 0.0

        if hasattr(self, 'extra_params') and self.extra_params:
            judge_temperature = float(self.extra_params.get('judge_temperature', judge_temperature))
            candidate = _resolve_judge_config({
                'api_url': self.extra_params.get('judge_api_url', ''),
                'api_key': self.extra_params.get('judge_api_key', 'EMPTY'),
                'model_id': self.extra_params.get('judge_model_id', ''),
            })
            if candidate:
                judge_configs = [candidate]

        if not judge_configs:
            judge_configs = _load_task_judge_configs(getattr(self, '_task_config', None))
        if not judge_configs:
            judge_configs = _load_default_judge_configs()
        if not judge_configs:
            judge_configs = [{
                'api_url': 'http://localhost:8007/v1/chat/completions',
                'api_key': 'EMPTY',
                'model_id': 'Qwen2.5-7B-Instruct',
            }]

        if len(self.evaluator_model_names) != len(judge_configs):
            self.evaluator_model_names = [f'evaluator{i + 1}' for i in range(len(judge_configs))]

        self.evaluator_display_names = [config['model_id'] for config in judge_configs]

        logger.info(
            'Initializing CreativeMath judges: %s',
            [config['model_id'] for config in judge_configs],
        )

        for model_name, judge_config in zip(self.evaluator_model_names, judge_configs):
            try:
                judge = LLMJudge(
                    api_key=judge_config['api_key'],
                    api_url=judge_config['api_url'],
                    model_id=judge_config['model_id'],
                    generation_config={'temperature': judge_temperature, 'max_tokens': 512}
                )
                self._llm_judges[model_name] = judge
                logger.info(f"Initialized LLM judge: {model_name} -> {judge_config['model_id']}")
            except Exception as e:
                logger.error(f"Failed to initialize LLM judge {model_name}: {e}")
                raise

    def _get_llm_judge(self, model_name: str):
        """Get LLM judge instance."""
        if self._llm_judges is None or model_name not in self._llm_judges:
            raise ValueError(f"LLM judge {model_name} not initialized")
        return self._llm_judges[model_name]

    def _display_name(self, evaluator_name: str) -> str:
        try:
            idx = self.evaluator_model_names.index(evaluator_name)
        except ValueError:
            return evaluator_name
        if 0 <= idx < len(self.evaluator_display_names):
            return self.evaluator_display_names[idx]
        return evaluator_name

    def defer_score_calculation_to_batch(self) -> bool:
        return False

    def _sample_key_from_metadata(self, metadata: Dict[str, Any]) -> str:
        return f"{metadata.get('problem_id', 'unknown')}::k={metadata.get('k', 'unknown')}"

    def _judge_stage(self, sample_key: str, stage_name: str, prompt: str) -> Dict[str, Any]:
        results: Dict[str, str] = {}
        judge_notes: List[Dict[str, Any]] = []
        raw_responses: Dict[str, str] = {}
        failures: List[str] = []

        for evaluator_name in self.evaluator_model_names:
            display_name = self._display_name(evaluator_name)
            cached = self._judge_cache.get(sample_key, stage_name, display_name)
            if cached and cached.get('status') == 'success':
                verdict = str(cached.get('verdict') or 'NO').upper()
                results[display_name] = 'YES' if verdict == 'YES' else 'NO'
                raw_responses[display_name] = str(cached.get('raw_response') or '')
                judge_notes.append({
                    'model_id': display_name,
                    'status': 'success',
                    'attempts': cached.get('attempts'),
                    'cached': True,
                    'verdict': results[display_name],
                    'raw_response_excerpt': str(cached.get('raw_response') or '')[:500],
                })
                continue

            try:
                judge = self._get_llm_judge(evaluator_name)
                response = judge.judge(prompt)
                verdict = extract_yes_no(response)
                if response.strip().startswith('[ERROR]'):
                    raise RuntimeError(response)
                self._judge_cache.put(
                    sample_key,
                    stage_name,
                    display_name,
                    {
                        'benchmark': 'creative_math',
                        'status': 'success',
                        'model_id': display_name,
                        'attempts': 1,
                        'verdict': verdict,
                        'raw_response': response,
                    },
                )
                results[display_name] = verdict
                raw_responses[display_name] = response
                judge_notes.append({
                    'model_id': display_name,
                    'status': 'success',
                    'attempts': 1,
                    'cached': False,
                    'verdict': verdict,
                    'raw_response_excerpt': response[:500],
                })
            except Exception as exc:
                detail = str(exc)
                self._judge_cache.put(
                    sample_key,
                    stage_name,
                    display_name,
                    {
                        'benchmark': 'creative_math',
                        'status': 'failed',
                        'model_id': display_name,
                        'attempts': 1,
                        'error': detail,
                    },
                )
                failures.append(f'{display_name}: {detail}')

        if failures:
            raise RuntimeError(f'{stage_name} judge failed for {len(failures)} evaluators. First failure: {failures[0]}')

        return {
            'votes': results,
            'judge_notes': judge_notes,
            'raw_responses': raw_responses,
        }

    def _get_dataset_path(self) -> str:
        """Get dataset path with fallbacks to default locations."""
        dataset_path = self.dataset_id

        # Check if it's a valid path
        if dataset_path == 'creative_math' or not os.path.exists(dataset_path):
            # Try default relative path
            default_path = os.path.join(
                self.creative_math_base_path,
                'data/subset.json'
            )
            if os.path.exists(default_path):
                return default_path

            # Try absolute path
            abs_path = '/root/data/code/evalscope/dataprocess/exploration/CreativeMath/data/subset.json'
            if os.path.exists(abs_path):
                return abs_path

            raise FileNotFoundError(
                f"CreativeMath dataset not found. Tried: {dataset_path}, {default_path}, {abs_path}. "
                f"Please specify dataset_path in extra_params."
            )

        return dataset_path

    def load_from_disk(self, **kwargs):
        """Load dataset from local disk."""
        return super().load_from_disk(use_local_loader=True)

    def load_subset(self, subset_name: str, data_loader=None, is_fewshot: bool = False):
        """
        Load CreativeMath dataset and expand into multiple samples per problem.

        Each problem with n reference solutions is expanded into n samples (k=1 to n),
        where each sample asks the model to generate a novel solution given k reference solutions.
        """
        if is_fewshot:
            return None

        # Load dataset
        dataset_path = self._get_dataset_path()
        logger.info(f"Loading CreativeMath dataset from {dataset_path}")

        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} problems from CreativeMath dataset")

        samples = []
        sample_id = 0

        # Each problem becomes n samples (one for each k from 1 to n)
        for problem_record in data:
            problem = problem_record['problem']
            solutions = problem_record['solutions']  # Dict with keys "1", "2", etc.
            n = len(solutions)

            # Create samples for k = 1, 2, ..., n
            for k in range(1, n + 1):
                # Build prompt: problem + first k solutions + request for novel solution
                try:
                    prompt = self._build_generation_prompt(problem, solutions, k)
                except Exception as e:
                    logger.warning(f"Failed to build prompt for problem {problem_record.get('problem_id', 'unknown')}, k={k}: {e}")
                    continue

                sample = Sample(
                    id=sample_id,
                    input=[ChatMessageUser(content=prompt)],
                    target='',  # No single target answer - evaluated via multi-stage process
                    metadata={
                        'problem_id': problem_record.get('problem_id', sample_id),
                        'k': k,
                        'n': n,
                        'problem': problem,
                        'solutions': solutions,  # Store all reference solutions
                        'competition': problem_record.get('competition', ''),
                        'difficulty': problem_record.get('difficulty', 0.0),
                        'competition_id': problem_record.get('competition_id', '')
                    }
                )
                samples.append(sample)
                sample_id += 1

        logger.info(f"Expanded {len(data)} problems into {len(samples)} samples")

        # Apply limit if specified (from task config)
        if hasattr(self, 'limit') and self.limit is not None and self.limit > 0:
            original_count = len(samples)
            samples = samples[:self.limit]
            logger.info(f"Limited samples from {original_count} to {len(samples)}")

        return samples

    def _build_generation_prompt(self, problem: str, solutions: dict, k: int) -> str:
        """Build prompt for novel solution generation (reusing original prompt logic)."""
        # Convert solutions dict to list
        solutions_list = [solutions[str(i)] for i in range(1, len(solutions) + 1)]

        # Use prompt generation function (either imported or fallback)
        return load_novel_solution_generation_prompt(problem, solutions_list, k)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a data record to a Sample object."""
        # This method is not used since we override load_subset
        problem = record.get('problem', '')
        return Sample(
            input=[ChatMessageUser(content=problem)],
            target='',
            metadata=record
        )

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """
        Calculate score using three-stage evaluation:
        1. Correctness evaluation (all 3 evaluators must agree YES)
        2. Coarse-grained novelty (majority voting, only if correct)
        3. Fine-grained novelty (majority voting, only if novel and k < n)
        """
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        metadata = task_state.metadata
        problem_id = metadata.get('problem_id', 'unknown')
        k = metadata.get('k', 1)
        n = metadata.get('n', 1)
        problem = metadata.get('problem', '')
        solutions = metadata.get('solutions', {})
        new_solution = filtered_prediction

        # Convert solutions dict to list
        solutions_list = [solutions[str(i)] for i in range(1, n + 1)]

        stop_reason = getattr(task_state.output, 'stop_reason', None)
        if stop_reason in {'max_tokens', 'model_length'}:
            has_answer_marker = _has_answer_marker(new_solution)
            score.value = {
                'correctness': 0,
                'coarse_grained_novelty': 0,
                'fine_grained_novelty': 0,
            }
            score.metadata = {
                'problem_id': problem_id,
                'k': k,
                'n': n,
                'evaluation_mode': self.evaluation_mode,
                'truncated': True,
                'truncation_reason': stop_reason,
                'answer_marker_present': has_answer_marker,
                'reason': 'truncated_output',
            }
            score.main_score_name = 'correctness'
            return score

        try:
            sample_key = self._sample_key_from_metadata(metadata)

            # Stage 1: Correctness Evaluation
            correctness_result = self._evaluate_correctness(problem, solutions_list, new_solution, sample_key)
            is_correct = (correctness_result['final_decision'] == 'YES')

            # Stage 2: Coarse-Grained Novelty (only if correct)
            if is_correct:
                coarse_novelty_result = self._evaluate_coarse_novelty(
                    problem, solutions_list, k, new_solution, sample_key
                )
                is_novel = (coarse_novelty_result['final_decision'] == 'YES')
            else:
                coarse_novelty_result = {'final_decision': 'NO', 'reason': 'not correct'}
                is_novel = False

            # Stage 3: Fine-Grained Novelty (only if novel and k < n)
            if is_novel and k < n:
                fine_novelty_result = self._evaluate_fine_novelty(
                    problem, solutions_list, k, new_solution, sample_key
                )
                is_novel_unknown = (fine_novelty_result['final_decision'] == 'YES')
            else:
                fine_novelty_result = {
                    'final_decision': 'NO',
                    'reason': 'not novel' if not is_novel else 'k >= n'
                }
                is_novel_unknown = False

            # Set score values (binary for each stage)
            score.value = {
                'correctness': 1 if is_correct else 0,
                'coarse_grained_novelty': 1 if is_novel else 0,
                'fine_grained_novelty': 1 if is_novel_unknown else 0,
            }

            score.metadata = {
                'problem_id': problem_id,
                'k': k,
                'n': n,
                'evaluation_mode': self.evaluation_mode,
                'correctness_evaluations': correctness_result,
                'coarse_novelty_evaluations': coarse_novelty_result,
                'fine_novelty_evaluations': fine_novelty_result,
            }

        except Exception as e:
            raise RuntimeError(f"CreativeMath judge failed for problem_id={problem_id}, k={k}: {e}") from e

        score.main_score_name = 'correctness'
        return score

    def _build_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        metadata: Dict[str, Any],
        correctness_result: Dict[str, Any],
        coarse_novelty_result: Dict[str, Any],
        fine_novelty_result: Dict[str, Any],
    ) -> Score:
        is_correct = correctness_result.get('final_decision') == 'YES'
        is_novel = coarse_novelty_result.get('final_decision') == 'YES'
        is_novel_unknown = fine_novelty_result.get('final_decision') == 'YES'

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        score.value = {
            'correctness': 1 if is_correct else 0,
            'coarse_grained_novelty': 1 if is_novel else 0,
            'fine_grained_novelty': 1 if is_novel_unknown else 0,
        }
        score.metadata = {
            'problem_id': metadata.get('problem_id', 'unknown'),
            'k': metadata.get('k', 1),
            'n': metadata.get('n', 1),
            'evaluation_mode': self.evaluation_mode,
            'correctness_evaluations': correctness_result,
            'coarse_novelty_evaluations': coarse_novelty_result,
            'fine_novelty_evaluations': fine_novelty_result,
        }
        score.main_score_name = 'correctness'
        return score

    def _batch_collect_stage_votes(
        self,
        prompts: List[str],
        active_indices: List[int],
        stage_name: str,
    ) -> Dict[int, Dict[str, str]]:
        if not active_indices:
            return {}

        results: Dict[int, Dict[str, str]] = {idx: {} for idx in active_indices}
        for evaluator_name in self.evaluator_model_names:
            judge = self._get_llm_judge(evaluator_name)
            try:
                responses = judge.batch_judge(prompts=prompts)
                if len(responses) != len(active_indices):
                    raise ValueError(
                        f'{stage_name} batch judge response count mismatch: {len(responses)} vs {len(active_indices)}'
                    )
                for idx, response in zip(active_indices, responses):
                    results[idx][self._display_name(evaluator_name)] = extract_yes_no(response)
            except Exception as exc:
                logger.error(f'Error querying batch judge {evaluator_name} for {stage_name}: {exc}')
                for idx in active_indices:
                    results[idx][self._display_name(evaluator_name)] = 'NO'
        return results

    def batch_match_score(
        self,
        original_predictions: List[str],
        filtered_predictions: List[str],
        references: List[str],
        task_states: List[TaskState],
    ) -> Optional[List[Score]]:
        return None

    def _evaluate_correctness(self, problem: str, solutions: list, new_solution: str, sample_key: str) -> dict:
        """
        Stage 1: Correctness evaluation
        - Simplified mode: Assume YES
        - Full mode: Use 3 LLM judges, all must agree YES
        """
        if self.evaluation_mode == 'simplified':
            return {'final_decision': 'YES', 'mode': 'simplified'}

        prompt = load_correctness_evaluation_prompt(problem, solutions, new_solution)
        stage_result = self._judge_stage(sample_key, 'correctness', prompt)
        results = dict(stage_result['votes'])
        all_yes = all(v == 'YES' for v in results.values())
        results['final_decision'] = 'YES' if all_yes else 'NO'
        results['mode'] = 'full'
        results['judge_notes'] = stage_result['judge_notes']
        results['raw_responses'] = stage_result['raw_responses']
        return results

    def _evaluate_coarse_novelty(self, problem: str, solutions: list, k: int, new_solution: str, sample_key: str) -> dict:
        """
        Stage 2: Coarse-grained novelty evaluation
        - Simplified mode: Assume YES
        - Full mode: Use 3 LLM judges, majority voting
        """
        if self.evaluation_mode == 'simplified':
            return {'final_decision': 'YES', 'mode': 'simplified'}

        prompt = load_coarse_grained_novelty_evaluation_prompt(problem, solutions, k, new_solution)
        stage_result = self._judge_stage(sample_key, 'coarse_novelty', prompt)
        results = dict(stage_result['votes'])
        yes_count = sum(1 for v in results.values() if v == 'YES')
        results['final_decision'] = 'YES' if yes_count >= _majority_threshold(len(self.evaluator_model_names)) else 'NO'
        results['mode'] = 'full'
        results['judge_notes'] = stage_result['judge_notes']
        results['raw_responses'] = stage_result['raw_responses']
        return results

    def _evaluate_fine_novelty(self, problem: str, solutions: list, k: int, new_solution: str, sample_key: str) -> dict:
        """
        Stage 3: Fine-grained novelty evaluation
        - Simplified mode: Assume NO
        - Full mode: Use 3 LLM judges, majority voting
        """
        if self.evaluation_mode == 'simplified':
            return {'final_decision': 'NO', 'mode': 'simplified'}

        prompt = load_fine_grained_novelty_evaluation_prompt(problem, solutions, k, new_solution)
        stage_result = self._judge_stage(sample_key, 'fine_novelty', prompt)
        results = dict(stage_result['votes'])
        yes_count = sum(1 for v in results.values() if v == 'YES')
        results['final_decision'] = 'YES' if yes_count >= _majority_threshold(len(self.evaluator_model_names)) else 'NO'
        results['mode'] = 'full'
        results['judge_notes'] = stage_result['judge_notes']
        results['raw_responses'] = stage_result['raw_responses']
        return results

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """
        Calculate CreativeMath-specific metrics:
        1. Correctness Ratio: % of correct solutions
        2. Novelty Ratio: % of novel solutions (coarse-grained)
        3. Novel-Unknown Ratio: % of novel-unknown solutions (fine-grained)
        4. Novelty-to-Correctness Ratio: novel / correct
        5. Novel-Unknown-to-Novelty Ratio: novel-unknown / novel
        """
        N = len(sample_scores)
        if N == 0:
            logger.warning("No samples to aggregate")
            return []

        # Count samples for each metric
        correctness_count = 0
        novelty_count = 0
        novel_unknown_count = 0

        for sample_score in sample_scores:
            score_values = sample_score.score.value

            if score_values.get('correctness', 0) == 1:
                correctness_count += 1

            if score_values.get('coarse_grained_novelty', 0) == 1:
                novelty_count += 1

            if score_values.get('fine_grained_novelty', 0) == 1:
                novel_unknown_count += 1

        # Calculate ratios
        correctness_ratio = correctness_count / N
        novelty_ratio = novelty_count / N
        novel_unknown_ratio = novel_unknown_count / N

        if correctness_count > 0:
            novelty_to_correctness_ratio = novelty_count / correctness_count
        else:
            novelty_to_correctness_ratio = 0.0

        if novelty_count > 0:
            novel_unknown_to_novelty_ratio = novel_unknown_count / novelty_count
        else:
            novel_unknown_to_novelty_ratio = 0.0

        # AUT-like remapping based only on existing metrics.
        # originality: weighted combination of coarse-grained novelty and novel-unknown novelty.
        originality = 0.7 * novelty_ratio + 0.3 * novel_unknown_ratio
        appropriateness = correctness_ratio

        logger.info(f"CreativeMath metrics: correctness={correctness_ratio:.3f}, novelty={novelty_ratio:.3f}, "
                   f"novel_unknown={novel_unknown_ratio:.3f}, novelty/correctness={novelty_to_correctness_ratio:.3f}, "
                   f"novel_unknown/novelty={novel_unknown_to_novelty_ratio:.3f}, originality={originality:.3f}, "
                   f"appropriateness={appropriateness:.3f}")

        # Create AggScore objects
        agg_scores = [
            AggScore(
                metric_name='correctness_ratio',
                score=correctness_ratio,
                num=N,
                metadata={'count': correctness_count, 'total': N}
            ),
            AggScore(
                metric_name='novelty_ratio',
                score=novelty_ratio,
                num=N,
                metadata={'count': novelty_count, 'total': N}
            ),
            AggScore(
                metric_name='novel_unknown_ratio',
                score=novel_unknown_ratio,
                num=N,
                metadata={'count': novel_unknown_count, 'total': N}
            ),
            AggScore(
                metric_name='novelty_to_correctness_ratio',
                score=novelty_to_correctness_ratio,
                num=N,
                metadata={
                    'novelty_count': novelty_count,
                    'correctness_count': correctness_count
                }
            ),
            AggScore(
                metric_name='novel_unknown_to_novelty_ratio',
                score=novel_unknown_to_novelty_ratio,
                num=N,
                metadata={
                    'novel_unknown_count': novel_unknown_count,
                    'novelty_count': novelty_count
                }
            ),
            AggScore(
                metric_name='originality',
                score=originality,
                num=N,
                metadata={
                    'formula': '0.7 * novelty_ratio + 0.3 * novel_unknown_ratio',
                    'novelty_ratio': novelty_ratio,
                    'novel_unknown_ratio': novel_unknown_ratio,
                }
            ),
            AggScore(
                metric_name='appropriateness',
                score=appropriateness,
                num=N,
                metadata={
                    'formula': 'correctness_ratio',
                    'correctness_ratio': correctness_ratio,
                }
            ),
        ]

        return agg_scores
