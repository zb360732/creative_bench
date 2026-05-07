#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS4 Benchmark Adapter for Evalscope

CS4 (Comparing the Skill of Creating Stories by Controlling the Synthesized Constraint Specificity)
evaluates models' ability to create stories under varying constraint levels.

Features:
- 250 samples with 5 constraint levels (7, 15, 23, 31, 39 constraints)
- Two evaluation modes: simplified (fast) and full (LLM judges)
- 9 metrics: constraint satisfaction, quality (grammar/coherence/likability), diversity, appropriateness, novelty, QUC, RCS
- Multi-dimensional independent evaluation (unlike sequential evaluation)
"""

import json
import os
import re
import string
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

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

    # Ignore fully empty placeholders silently; these are common default extra_params values.
    if not api_url and not model_id and api_key in {'', 'EMPTY', 'YOUR_API_KEY'} and not api_key_env:
        return None

    if not api_url or not model_id:
        logger.warning(f"Invalid judge config entry: {entry}")
        return None
    return {
        'api_url': api_url,
        'api_key': api_key,
        'model_id': model_id,
    }


def _is_judge_error_response(text: str) -> bool:
    normalized = (text or '').strip()
    lowered = normalized.lower()
    if not normalized:
        return True
    if normalized.startswith('[ERROR]'):
        return True
    if normalized.startswith('<html'):
        return True
    return any(
        marker in lowered
        for marker in (
            'content management policy',
            'resource exhausted',
            'resource has been exhausted',
            'rate limit',
            'quota',
            'upstream_error',
        )
    )


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


def _extract_answer_content(text: str) -> str:
    text = (text or '').strip()
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    open_match = re.search(r'<answer>\s*', text, flags=re.IGNORECASE)
    if open_match:
        return text[open_match.end():].replace('</answer>', '').strip()
    return text


@register_benchmark(
    BenchmarkMeta(
        name='cs4',
        pretty_name='CS4 Benchmark',
        tags=[Tags.REASONING, Tags.CUSTOM],
        description='评估模型在约束条件下的故事创作能力和创造力。CS4 (Comparing the Skill of Creating Stories by Controlling the Synthesized Constraint Specificity) benchmark evaluates models\' ability to create creative stories under varying constraint levels.',
        dataset_id='cs4',
        subset_list=[
            'constraints_7',    # 7 constraints, 50 samples
            'constraints_15',   # 15 constraints, 50 samples
            'constraints_23',   # 23 constraints, 50 samples
            'constraints_31',   # 31 constraints, 50 samples
            'constraints_39'    # 39 constraints, 50 samples
        ],
        default_subset='constraints_7',
        metric_list=[
            'fluency',                        # Fluency (constraint satisfaction ratio)
            'grammar_score',                  # Grammar rating (1-5)
            'coherence_score',                # Coherence rating (1-5)
            'likability_score',               # Likability rating (1-5)
            'flexibility',                    # Flexibility (vocabulary diversity)
            'appropriateness',                # Appropriateness (relevance x coherence)
            'novelty',                        # Novelty (1-5)
            'quc_score',                      # Quality Under Constraints
            'rcs_score'                       # Relative Creativity Score (optional)
        ],
        eval_split='test',
        prompt_template='{instruction}',
        review_timeout=60,
        extra_params={
            'evaluation_mode': {
                'type': 'str',
                'description': 'Evaluation mode: "simplified" (fast, assumes perfect scores) or "full" (LLM judges)',
                'value': 'simplified'
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
            'judge_max_tokens': {
                'type': 'int',
                'description': 'Max tokens for LLM judges in full mode',
                'value': 4096
            },
            'judge_temperature': {
                'type': 'float',
                'description': 'Sampling temperature for LLM judges in full mode',
                'value': 0.0
            },
            'constraint_satisfaction_judge': {
                'type': 'str',
                'description': 'Judge model for constraint satisfaction evaluation',
                'value': 'judge1'
            },
            'story_quality_judge': {
                'type': 'str',
                'description': 'Judge model for story quality evaluation',
                'value': 'judge2'
            },
            'dataset_path': {
                'type': 'str',
                'description': 'Custom path to Story-based Base Stories.csv',
                'value': None
            }
        }
    )
)
class CS4Adapter(DefaultDataAdapter):
    """Adapter for CS4 story creativity benchmark"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set CS4 source code path
        self.cs4_base_path = os.path.join(
            os.path.dirname(__file__),
            '../../../dataprocess/exploration/cs4_benchmark'
        )
        self.cs4_base_path = os.path.abspath(self.cs4_base_path)

        # Get evaluation mode
        self.evaluation_mode = 'simplified'
        if hasattr(self, 'extra_params') and self.extra_params:
            self.evaluation_mode = self.extra_params.get('evaluation_mode', 'simplified')
            custom_path = self.extra_params.get('dataset_path')
            if custom_path and os.path.exists(custom_path):
                self.dataset_id = custom_path

        logger.info(f"CS4 evaluation mode: {self.evaluation_mode}")

        # Initialize LLM judges for full mode
        task_cfg = getattr(self, '_task_config', None)
        self._judge_cache = BenchmarkJudgeCache(
            benchmark_name='cs4',
            work_dir=getattr(task_cfg, 'work_dir', None),
            model_name=getattr(task_cfg, 'model_id', None),
        )
        self._llm_judges = None
        if self.evaluation_mode == 'full':
            self._init_llm_judges()

        # Ensure NLTK data is available
        self._ensure_nltk_data()

    def _ensure_nltk_data(self):
        """Ensure NLTK punkt tokenizer is available"""
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.info("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)
        except Exception as e:
            logger.warning(f"Failed to ensure NLTK data: {e}")

    def _init_llm_judges(self):
        """Initialize LLM judges for full evaluation mode"""
        if self.evaluation_mode != 'full':
            return

        self._llm_judges = {}
        judge_max_tokens = 4096
        judge_temperature = 0.0
        judge_configs: List[Dict[str, str]] = []
        if hasattr(self, 'extra_params') and self.extra_params:
            judge_max_tokens = int(self.extra_params.get('judge_max_tokens', judge_max_tokens))
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
            logger.warning('No valid CS4 judge configs found; review requests will fail fast and be skipped.')

        logger.info(
            'Initializing CS4 judges: %s, judge_max_tokens: %s',
            [config['model_id'] for config in judge_configs],
            judge_max_tokens,
        )

        # Create judges: constraint satisfaction + appropriateness bundle + novelty
        judge_types = ['constraint_satisfaction', 'story_quality', 'novelty']

        for judge_type in judge_types:
            self._llm_judges[judge_type] = []
            for judge_config in judge_configs:
                try:
                    judge = LLMJudge(
                        api_key=judge_config['api_key'],
                        api_url=judge_config['api_url'],
                        model_id=judge_config['model_id'],
                        generation_config={
                            'temperature': judge_temperature,
                            'max_tokens': judge_max_tokens
                        }
                    )
                    self._llm_judges[judge_type].append(judge)
                    logger.info(f"Initialized LLM judge: {judge_type} -> {judge_config['model_id']}")
                except Exception as e:
                    logger.warning(f"Failed to initialize LLM judge {judge_type}: {e}")
                    continue

    def _get_llm_judge(self, judge_type: str):
        """Get LLM judge by type"""
        if self._llm_judges is None or judge_type not in self._llm_judges:
            raise ValueError(f"LLM judge {judge_type} not initialized. Mode: {self.evaluation_mode}")
        judges = self._llm_judges[judge_type]
        if not judges:
            raise ValueError(f"LLM judge {judge_type} has no valid backends configured.")
        return judges

    def _sample_key_from_metadata(self, metadata: Dict[str, Any]) -> str:
        return ':'.join([
            str(metadata.get('story_id', 'unknown')),
            str(metadata.get('number_of_constraints', 'unknown')),
            str(metadata.get('instruction', ''))[:120],
        ])

    def _judge_key(self, judge: LLMJudge) -> str:
        return str(getattr(judge, 'model_id', '') or 'unknown_judge')

    def _validate_cached_judge_response(self, judge_type: str, response: str, prompt: str) -> None:
        if _is_judge_error_response(response):
            raise ValueError(f'{judge_type} cached response is an unusable error response.')

        if judge_type == 'constraint_satisfaction':
            constraints_text = prompt.split('Constraints:\n', 1)[-1].split('\n\nOutput format:', 1)[0]
            total = len([line for line in constraints_text.splitlines() if line.strip()])
            self._parse_constraint_satisfaction_response(response, total)
            return

        if judge_type == 'story_quality':
            self._parse_quality_scores(response)
            return

        if judge_type == 'novelty':
            self._parse_novelty_score(response)
            return

    def _judge_all_cached(self, sample_key: str, judge_type: str, prompt: str) -> Dict[str, Any]:
        judges = self._get_llm_judge(judge_type)
        responses: List[str] = []
        judge_notes: List[Dict[str, Any]] = []
        failures: List[str] = []

        for judge in judges:
            judge_key = self._judge_key(judge)
            cached = self._judge_cache.get(sample_key, judge_type, judge_key)
            if cached and cached.get('status') == 'success':
                response = str(cached.get('raw_response') or '')
                try:
                    self._validate_cached_judge_response(judge_type, response, prompt)
                except Exception as exc:
                    logger.warning(
                        'Discarding invalid cached %s judge response for sample_key=%s judge=%s: %s',
                        judge_type,
                        sample_key,
                        judge_key,
                        exc,
                    )
                    self._judge_cache.put(
                        sample_key,
                        judge_type,
                        judge_key,
                        {
                            'benchmark': 'cs4',
                            'status': 'failed',
                            'model_id': judge_key,
                            'attempts': cached.get('attempts'),
                            'raw_response': response,
                            'error': f'invalid cached response: {exc}',
                        },
                    )
                else:
                    responses.append(response)
                    judge_notes.append({
                        'model_id': judge_key,
                        'status': 'success',
                        'attempts': cached.get('attempts'),
                        'cached': True,
                        'raw_response_excerpt': response[:500],
                    })
                    continue

            response = judge.judge(prompt)
            if _is_judge_error_response(response):
                self._judge_cache.put(
                    sample_key,
                    judge_type,
                    judge_key,
                    {
                        'benchmark': 'cs4',
                        'status': 'failed',
                        'model_id': judge_key,
                        'attempts': 1,
                        'raw_response': response,
                        'error': response[:1000],
                    },
                )
                failures.append(f'{judge_key}: {(response or "")[:500]}')
                continue

            try:
                self._validate_cached_judge_response(judge_type, response, prompt)
            except Exception as exc:
                self._judge_cache.put(
                    sample_key,
                    judge_type,
                    judge_key,
                    {
                        'benchmark': 'cs4',
                        'status': 'failed',
                        'model_id': judge_key,
                        'attempts': 1,
                        'raw_response': response,
                        'error': str(exc),
                    },
                )
                failures.append(f'{judge_key}: {str(exc)[:500]}')
                continue

            self._judge_cache.put(
                sample_key,
                judge_type,
                judge_key,
                {
                    'benchmark': 'cs4',
                    'status': 'success',
                    'model_id': judge_key,
                    'attempts': 1,
                    'raw_response': response,
                },
            )
            responses.append(response)
            judge_notes.append({
                'model_id': judge_key,
                'status': 'success',
                'attempts': 1,
                'cached': False,
                'raw_response_excerpt': response[:500],
            })

        if failures:
            raise RuntimeError(
                f"{judge_type} judge returned {len(failures)} failed responses. First failure: {failures[0]}"
            )
        if not responses:
            raise RuntimeError(f"{judge_type} judge returned no usable responses.")
        return {'responses': responses, 'judge_notes': judge_notes}

    def _judge_all(self, judge_type: str, prompt: str) -> List[str]:
        judges = self._get_llm_judge(judge_type)
        responses: List[str] = []
        failures: List[str] = []
        for judge in judges:
            response = judge.judge(prompt)
            if _is_judge_error_response(response):
                failures.append((response or '')[:500])
                continue
            responses.append(response)

        if failures:
            raise RuntimeError(
                f"{judge_type} judge returned {len(failures)} failed responses. First failure: {failures[0]}"
            )
        if not responses:
            raise RuntimeError(f"{judge_type} judge returned no usable responses.")
        return responses

    def _get_dataset_path(self) -> str:
        """Get dataset path with fallback logic"""
        dataset_path = self.dataset_id

        if dataset_path == 'cs4' or not os.path.exists(dataset_path):
            # Try default path
            default_path = os.path.join(
                self.cs4_base_path,
                'CS4_dataset/Story-based Base Stories.csv'
            )
            if os.path.exists(default_path):
                return default_path

            # Try absolute path
            abs_path = '/root/data/code/evalscope/dataprocess/exploration/cs4_benchmark/CS4_dataset/Story-based Base Stories.csv'
            if os.path.exists(abs_path):
                return abs_path

            raise FileNotFoundError(
                f"CS4 dataset not found. Tried: {dataset_path}, {default_path}, {abs_path}"
            )

        return dataset_path

    def load_subset(self, subset_name, data_loader=None, is_fewshot=False):
        """
        Load CS4 dataset subset

        Args:
            subset_name: one of 'constraints_7/15/23/31/39'

        Returns:
            List of Sample objects
        """
        # Load CSV
        dataset_path = self._get_dataset_path()
        logger.info(f"Loading CS4 dataset from: {dataset_path}")

        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded {len(df)} samples from CSV")

        # Add story_id based on Instruction (group same story across constraint levels)
        df['story_id'] = df.groupby('Instruction').ngroup()
        logger.info(f"Identified {df['story_id'].nunique()} unique stories")

        # Filter by constraint level
        valid_subsets = {'constraints_7', 'constraints_15', 'constraints_23', 'constraints_31', 'constraints_39'}
        if subset_name not in valid_subsets:
            raise ValueError(f"Unsupported CS4 subset: {subset_name}. Expected one of {sorted(valid_subsets)}")

        constraint_num = int(subset_name.split('_')[1])
        df = df[df['Number_of_Constraints'] == constraint_num]
        logger.info(f"Filtered to {len(df)} samples with {constraint_num} constraints")

        # Build samples
        samples = []
        for idx, row in df.iterrows():
            # Build story revision prompt
            prompt = self._build_story_revision_prompt(row)

            # Use original index if provided (for targeted reruns/merges)
            sample_id = idx
            if 'orig_index' in row and not pd.isna(row['orig_index']):
                try:
                    sample_id = int(row['orig_index'])
                except Exception:
                    sample_id = idx

            sample = Sample(
                id=sample_id,
                input=[ChatMessageUser(content=prompt)],
                target='',  # No ground truth, evaluated through scoring
                metadata={
                    'story_id': int(row['story_id']),  # Group samples by story
                    'instruction': row['Instruction'],
                    'base_story': row['BaseStory'],
                    'selected_constraints': row['SelectedConstraints'],
                    'number_of_constraints': row['Number_of_Constraints'],
                    'direction': row.get('Direction', ''),
                    'constraint_level': row['Number_of_Constraints']  # For RCS calculation
                }
            )
            samples.append(sample)

        # Apply limit
        if hasattr(self, 'limit') and self.limit and self.limit > 0:
            original_len = len(samples)
            samples = samples[:self.limit]
            logger.info(f"Limited samples from {original_len} to {len(samples)}")

        logger.info(f"CS4 {subset_name}: {len(samples)} samples ready")
        return samples

    def _build_story_revision_prompt(self, row) -> str:
        """Build CS4 story revision prompt"""
        instruction = row['Instruction']
        base_story = row['BaseStory']
        constraints = row['SelectedConstraints']

        # Follow original storygen.py format
        revision_prompt = f"""Now revise the given BaseStory to satisfy the following constraints within 500 words:
{constraints}"""

        full_prompt = f"""Story Instruction: {instruction}

BaseStory:
{base_story}

Task: {revision_prompt}"""

        return full_prompt

    def match_score(self, original_prediction, filtered_prediction, reference, task_state):
        """
        CS4 multi-dimensional evaluation:
        1. Constraint satisfaction
        2. Story quality (grammar, coherence, likability)
        3. Diversity (N-gram based)
        4. QUC (Quality Under Constraints)
        """
        metadata = task_state.metadata
        generated_story = filtered_prediction
        constraints = metadata['selected_constraints']
        num_constraints = metadata['number_of_constraints']
        base_story = metadata['base_story']
        sample_key = self._sample_key_from_metadata(metadata)

        # Initialize results
        results = {
            'fluency': 0.0,
            'grammar': 0.0,
            'coherence': 0.0,
            'likability': 0.0,
            'flexibility': 0.0,
            'appropriateness': 0.0,
            'novelty': 0.0,
            'quc': 0.0
        }

        # 1. Constraint satisfaction evaluation (fluency)
        constraint_result = self._evaluate_constraint_satisfaction(
            generated_story, constraints, num_constraints, sample_key
        )
        results['fluency'] = constraint_result['fluency_binary']

        # 2. Story quality + relevance evaluation
        quality_result = self._evaluate_story_quality(
            generated_story, base_story, metadata.get('instruction', ''), constraints, sample_key
        )
        results['grammar'] = quality_result['grammar_score']
        results['coherence'] = quality_result['coherence_score']
        results['likability'] = quality_result['likability_score']
        relevance_score = quality_result['relevance_score']

        # 2.6. Novelty evaluation
        novelty_result = self._evaluate_novelty(
            generated_story, metadata.get('instruction', ''), constraints, sample_key
        )
        results['novelty'] = novelty_result['novelty_score']

        # 3. Flexibility calculation (vocabulary diversity, no LLM needed)
        flexibility_result = self._calculate_diversity(generated_story)
        results['flexibility'] = flexibility_result['product_diversity']

        # 4. Calculate QUC (Quality Under Constraints)
        # QUC = normalized_coherence × fluency
        normalized_coherence = results['coherence'] / 5.0  # Normalize to 0-1
        results['quc'] = normalized_coherence * results['fluency']

        # 5. Calculate Appropriateness
        # appropriateness = (relevance/5) × (coherence/5)
        normalized_relevance = relevance_score / 5.0
        results['appropriateness'] = normalized_relevance * normalized_coherence

        # Return Score object
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value=results,
            metadata={
                'constraint_details': constraint_result.get('details'),
                'constraint_judge_notes': constraint_result.get('judge_notes'),
                'quality_details': quality_result.get('details'),
                'quality_judge_notes': quality_result.get('judge_notes'),
                'flexibility_details': flexibility_result.get('details'),
                'novelty_details': novelty_result.get('details'),
                'novelty_judge_notes': novelty_result.get('judge_notes')
            },
            main_score_name='quc'
        )

        return score

    def _evaluate_constraint_satisfaction(self, story: str, constraints: str,
                                         num_constraints: int, sample_key: str) -> dict:
        """
        Evaluate story constraint satisfaction

        Simplified mode: Assume 100% satisfaction
        Full mode: Use LLMJudge to evaluate each constraint
        """
        if self.evaluation_mode == 'simplified':
            return {
                'satisfaction_ratio': 1.0,
                'fluency_binary': 1.0,
                'satisfied_count': num_constraints,
                'total_count': num_constraints,
                'mode': 'simplified'
            }

        # Full mode: Use LLMJudge
        prompt = self._build_constraint_satisfaction_prompt(story, constraints)

        cached = self._judge_all_cached(sample_key, 'constraint_satisfaction', prompt)
        responses = cached['responses']
        counts = [self._parse_constraint_satisfaction_response(response, num_constraints) for response in responses]
        mean_count = sum(counts) / len(counts) if counts else 0.0

        return {
            'satisfaction_ratio': mean_count / num_constraints if num_constraints > 0 else 0.0,
            'fluency_binary': 1.0 if num_constraints > 0 and mean_count >= num_constraints else 0.0,
            'satisfied_count': mean_count,
            'total_count': num_constraints,
            'mode': 'full',
            'details': responses,
            'judge_notes': cached['judge_notes']
        }

    def _build_constraint_satisfaction_prompt(self, story: str, constraints: str) -> str:
        """Build constraint satisfaction evaluation prompt"""
        system_prompt = """You are an expert reader. I will give you a story followed by a set of constraints.
Your task is to carefully read both of them and tell how many constraints are being satisfied in the story.

As the output, I want you to print yes/no for each constraint based on whether it is being satisfied or not, followed by a 1 line explanation.
- If a constraint is fully satisfied, mark it as 'yes' and provide the sentence(s) from the story as evidence.
- If a constraint is not satisfied or only partially satisfied, mark it as 'no' and explain why.

At the end, output: "Number of constraints satisfied: [number]"

Story:
{story}

Constraints:
{constraints}

Output format:
First output one <think>...</think> block for hidden reasoning.
Then output exactly one <answer>...</answer> block using this format inside <answer>:
1. yes/no - [explanation with evidence from story]
2. yes/no - [explanation]
...
Number of constraints satisfied: X"""

        return system_prompt.format(story=story, constraints=constraints)

    def _parse_constraint_satisfaction_response(self, response: str, total: int) -> int:
        """Parse LLM response to extract satisfied constraint count"""
        response = _extract_answer_content(response)
        if _is_judge_error_response(response):
            raise ValueError(f"Constraint judge returned an unusable response: {response[:1000]}")
        # Find "Number of constraints satisfied: X"
        match = re.search(r'Number of constraints satisfied:\s*(\d+)', response, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Fallback: Count 'yes' occurrences at the beginning of constraint evaluations
        # Match patterns like "1. yes -", "2. no -", etc.
        yes_matches = re.findall(r'^\s*\d+\.\s*(yes)\s*[-–:]', response, re.IGNORECASE | re.MULTILINE)
        no_matches = re.findall(r'^\s*\d+\.\s*(no)\s*[-–:]', response, re.IGNORECASE | re.MULTILINE)
        yes_count = len(yes_matches)
        no_count = len(no_matches)

        # Require a complete per-constraint judgment to avoid silently biasing scores.
        if yes_count + no_count == total:
            return yes_count

        if yes_count + no_count > 0:
            raise ValueError(
                f"Constraint count mismatch: found {yes_count} yes + {no_count} no = {yes_count + no_count}, "
                f"expected {total}. Raw response: {response[:1000]}"
            )

        raise ValueError(f"Failed to parse constraint satisfaction count. Raw response: {response[:1000]}")

    def _evaluate_story_quality(self, story: str, base_story: str = None, instruction: str = '', constraints: str = '', sample_key: str = '') -> dict:
        """
        Evaluate story quality (grammar, coherence, likability)

        Simplified mode: Return perfect scores
        Full mode: Use LLMJudge
        """
        if self.evaluation_mode == 'simplified':
            return {
                'grammar_score': 5.0,
                'coherence_score': 5.0,
                'likability_score': 5.0,
                'relevance_score': 5.0,
                'mode': 'simplified'
            }

        prompt = self._build_quality_evaluation_prompt(story, instruction, constraints)

        cached = self._judge_all_cached(sample_key, 'story_quality', prompt)
        responses = cached['responses']
        parsed_scores = [self._parse_quality_scores(response) for response in responses]
        scores = {
            'grammar': sum(score['grammar'] for score in parsed_scores) / len(parsed_scores),
            'coherence': sum(score['coherence'] for score in parsed_scores) / len(parsed_scores),
            'likability': sum(score['likability'] for score in parsed_scores) / len(parsed_scores),
            'relevance': sum(score['relevance'] for score in parsed_scores) / len(parsed_scores),
        }

        return {
            'grammar_score': scores['grammar'],
            'coherence_score': scores['coherence'],
            'likability_score': scores['likability'],
            'relevance_score': scores['relevance'],
            'mode': 'full',
            'details': responses,
            'judge_notes': cached['judge_notes']
        }

    def _build_quality_evaluation_prompt(self, story: str, instruction: str, constraints: str) -> str:
        """Build bundled quality/relevance evaluation prompt with explicit anchors."""
        prompt = f"""You are an expert story evaluator. Evaluate the following story on four dimensions using the anchor rules below.

Scoring anchors:
- 1/5 = severely deficient; major failures that make the dimension largely unmet.
- 2/5 = weak; some partial evidence but clear and important problems remain.
- 3/5 = adequate; the story is workable but has noticeable weaknesses.
- 4/5 = strong; the story meets the dimension well with only minor weaknesses.
- 5/5 = excellent; the dimension is fully and clearly satisfied.

Dimension-specific guidance:
1. Grammar: judge sentence-level fluency, clarity, and absence of obvious writing errors.
2. Coherence: judge whether the plot, causal flow, and scene logic stay internally consistent from beginning to end.
3. Likability: judge reader-level engagement and readability, but do not reward irrelevant ornament over clear story execution.
4. Relevance: judge how well the story actually uses the instruction and constraints in its concrete content, not just surface keyword overlap.

Instruction:
{instruction}

Constraints:
{constraints}

Story:
{story}

Output format:
First output one <think>...</think> block for hidden reasoning.
Then output exactly one <answer>...</answer> block using this format inside <answer>:
Grammar: X/5 - [brief explanation]
Coherence: X/5 - [brief explanation]
Likability: X/5 - [brief explanation]
Relevance: X/5 - [brief explanation]"""

        return prompt

    def _parse_quality_scores(self, response: str) -> dict:
        """Parse quality scores from LLM response"""
        response = _extract_answer_content(response)
        if _is_judge_error_response(response):
            raise ValueError(f"Story quality judge returned an unusable response: {response[:1000]}")

        grammar_match = re.search(r'Grammar:\s*(\d+(?:\.\d+)?)/5', response, re.IGNORECASE)
        coherence_match = re.search(r'Coherence:\s*(\d+(?:\.\d+)?)/5', response, re.IGNORECASE)
        likability_match = re.search(r'Likability:\s*(\d+(?:\.\d+)?)/5', response, re.IGNORECASE)
        relevance_match = re.search(r'Relevance:\s*(\d+(?:\.\d+)?)/5', response, re.IGNORECASE)

        if not all((grammar_match, coherence_match, likability_match, relevance_match)):
            raise ValueError(f"Failed to parse story quality scores. Raw response: {response[:1000]}")

        return {
            'grammar': float(grammar_match.group(1)),
            'coherence': float(coherence_match.group(1)),
            'likability': float(likability_match.group(1)),
            'relevance': float(relevance_match.group(1)),
        }

    def _evaluate_novelty(self, story: str, instruction: str, constraints: str, sample_key: str) -> dict:
        """
        Evaluate novelty (1-5 scale)

        Simplified mode: Return perfect score
        Full mode: Use LLMJudge
        """
        if self.evaluation_mode == 'simplified':
            return {
                'novelty_score': 5.0,
                'mode': 'simplified'
            }

        prompt = self._build_novelty_evaluation_prompt(story, instruction, constraints)

        cached = self._judge_all_cached(sample_key, 'novelty', prompt)
        responses = cached['responses']
        scores = [self._parse_novelty_score(response) for response in responses]
        return {
            'novelty_score': sum(scores) / len(scores),
            'mode': 'full',
            'details': responses,
            'judge_notes': cached['judge_notes']
        }

    def _build_novelty_evaluation_prompt(self, story: str, instruction: str, constraints: str) -> str:
        """Build novelty evaluation prompt with explicit anchors."""
        prompt = f"""You are an expert evaluator. Rate how novel the story is given the instruction and constraints.

Novelty anchors:
- 1/5 = highly conventional; mostly predictable, template-like, or minimally changed from an obvious baseline.
- 2/5 = slightly novel; a few fresh details but the main story logic remains generic or familiar.
- 3/5 = moderately novel; some meaningful fresh ideas, scenes, or constraint integration, but still partly conventional.
- 4/5 = clearly novel; the story contains distinctive developments or combinations that go beyond standard template responses.
- 5/5 = highly novel; the story is distinctly original in its core developments while still respecting the task and constraints.

Judging guidance:
- Judge novelty by story content, event design, and constraint integration.
- Do not reward incoherence, randomness, or gratuitous weirdness.
- Do not reward mere surface wording changes if the underlying story is conventional.

Instruction:
{instruction}

Constraints:
{constraints}

Story:
{story}

Output format:
First output one <think>...</think> block for hidden reasoning.
Then output exactly one <answer>...</answer> block using this format inside <answer>:
Novelty: X/5 - [brief explanation]"""
        return prompt

    def _parse_novelty_score(self, response: str) -> float:
        """Parse novelty score from LLM response"""
        response = _extract_answer_content(response)
        if _is_judge_error_response(response):
            raise ValueError(f"Novelty judge returned an unusable response: {response[:1000]}")
        match = re.search(r'Novelty:\s*(\d+(?:\.\d+)?)/5', response, re.IGNORECASE)
        if match:
            return float(match.group(1))
        raise ValueError(f"Failed to parse novelty score. Raw response: {response[:1000]}")

    def _calculate_diversity(self, story: str) -> dict:
        """
        Calculate vocabulary diversity using N-grams

        Reference: original diversity_calculation.py
        """
        try:
            import nltk
            from nltk import word_tokenize
            from nltk.util import ngrams
        except ImportError:
            logger.warning("NLTK not available, returning default diversity score")
            return {
                'product_diversity': 0.5,
                'diversity_2g': 0.5,
                'diversity_3g': 0.5,
                'diversity_4g': 0.5,
                'details': 'NLTK not available'
            }

        # Text preprocessing
        text = story.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))

        try:
            tokens = word_tokenize(text)
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}")
            return {
                'product_diversity': 0.5,
                'diversity_2g': 0.5,
                'diversity_3g': 0.5,
                'diversity_4g': 0.5,
                'details': f'Tokenization error: {e}'
            }

        # Calculate 2-gram, 3-gram, 4-gram diversity
        diversity_scores = {}
        for n in [2, 3, 4]:
            ngram_list = list(ngrams(tokens, n))
            unique_ngrams = len(set(ngram_list))
            total_ngrams = len(ngram_list)

            if total_ngrams > 0:
                diversity_scores[f'diversity_{n}g'] = unique_ngrams / total_ngrams
            else:
                diversity_scores[f'diversity_{n}g'] = 0.0

        # Calculate composite diversity score
        product_diversity = (
            diversity_scores['diversity_2g'] *
            diversity_scores['diversity_3g'] *
            diversity_scores['diversity_4g']
        )

        return {
            'product_diversity': product_diversity,
            'diversity_2g': diversity_scores['diversity_2g'],
            'diversity_3g': diversity_scores['diversity_3g'],
            'diversity_4g': diversity_scores['diversity_4g'],
            'details': diversity_scores
        }

    def aggregate_scores(self, sample_scores: List) -> List[AggScore]:
        """
        Calculate CS4 aggregate metrics

        Strategy:
        1. Group by story_id and constraint_level
        2. Calculate per-story averages across constraint levels (for overall scores)
        3. Calculate per-constraint-level averages across stories (for constraint-specific scores)
        4. Return both overall and per-constraint-level scores

        Returns: Overall metrics + per-constraint-level metrics + RCS
        """
        N = len(sample_scores)
        if N == 0:
            return []

        # Group samples by story_id and constraint_level
        story_groups = {}  # story_id -> {metric -> [values]}
        constraint_level_groups = {}  # constraint_level -> {story_id -> {metric -> values}}

        for sample_score in sample_scores:
            values = sample_score.score.value
            story_id = sample_score.sample_metadata.get('story_id', -1)
            constraint_level = sample_score.sample_metadata.get('constraint_level', 0)

            # Group by story_id (for overall scores)
            if story_id not in story_groups:
                story_groups[story_id] = {
                    'fluency': [],
                    'grammar': [],
                    'coherence': [],
                    'likability': [],
                    'flexibility': [],
                    'appropriateness': [],
                    'novelty': [],
                    'quc': []
                }

            story_groups[story_id]['fluency'].append(values.get('fluency', 0.0))
            story_groups[story_id]['grammar'].append(values.get('grammar', 0.0))
            story_groups[story_id]['coherence'].append(values.get('coherence', 0.0))
            story_groups[story_id]['likability'].append(values.get('likability', 0.0))
            story_groups[story_id]['flexibility'].append(values.get('flexibility', 0.0))
            story_groups[story_id]['appropriateness'].append(values.get('appropriateness', 0.0))
            story_groups[story_id]['novelty'].append(values.get('novelty', 0.0))
            story_groups[story_id]['quc'].append(values.get('quc', 0.0))

            # Group by constraint level (for per-level scores)
            if constraint_level not in constraint_level_groups:
                constraint_level_groups[constraint_level] = {}
            if story_id not in constraint_level_groups[constraint_level]:
                constraint_level_groups[constraint_level][story_id] = {
                    'fluency': 0.0,
                    'grammar': 0.0,
                    'coherence': 0.0,
                    'likability': 0.0,
                    'flexibility': 0.0,
                    'appropriateness': 0.0,
                    'novelty': 0.0,
                    'quc': 0.0
                }

            # Store the score for this story at this constraint level
            constraint_level_groups[constraint_level][story_id]['fluency'] = values.get('fluency', 0.0)
            constraint_level_groups[constraint_level][story_id]['grammar'] = values.get('grammar', 0.0)
            constraint_level_groups[constraint_level][story_id]['coherence'] = values.get('coherence', 0.0)
            constraint_level_groups[constraint_level][story_id]['likability'] = values.get('likability', 0.0)
            constraint_level_groups[constraint_level][story_id]['flexibility'] = values.get('flexibility', 0.0)
            constraint_level_groups[constraint_level][story_id]['appropriateness'] = values.get('appropriateness', 0.0)
            constraint_level_groups[constraint_level][story_id]['novelty'] = values.get('novelty', 0.0)
            constraint_level_groups[constraint_level][story_id]['quc'] = values.get('quc', 0.0)

        num_stories = len(story_groups)
        logger.info(f"Aggregating scores across {num_stories} stories with {N} total samples")
        logger.info(f"Constraint levels present: {sorted(constraint_level_groups.keys())}")

        # === 1. Calculate overall scores (average across all stories) ===
        story_averages = {
            'fluency': [],
            'grammar': [],
            'coherence': [],
            'likability': [],
            'flexibility': [],
            'appropriateness': [],
            'novelty': [],
            'quc': []
        }

        for story_id, metrics in story_groups.items():
            for metric_name, values in metrics.items():
                if values:
                    story_avg = sum(values) / len(values)
                    story_averages[metric_name].append(story_avg)

        agg_scores = [
            AggScore(metric_name='fluency',
                    score=sum(story_averages['fluency']) / num_stories if num_stories > 0 else 0.0,
                    num=num_stories),
            AggScore(metric_name='grammar_score',
                    score=sum(story_averages['grammar']) / num_stories if num_stories > 0 else 0.0,
                    num=num_stories),
            AggScore(metric_name='coherence_score',
                    score=sum(story_averages['coherence']) / num_stories if num_stories > 0 else 0.0,
                    num=num_stories),
            AggScore(metric_name='likability_score',
                    score=sum(story_averages['likability']) / num_stories if num_stories > 0 else 0.0,
                    num=num_stories),
            AggScore(metric_name='flexibility',
                    score=sum(story_averages['flexibility']) / num_stories if num_stories > 0 else 0.0,
                    num=num_stories),
            AggScore(metric_name='appropriateness',
                    score=sum(story_averages['appropriateness']) / num_stories if num_stories > 0 else 0.0,
                    num=num_stories),
            AggScore(metric_name='novelty',
                    score=sum(story_averages['novelty']) / num_stories if num_stories > 0 else 0.0,
                    num=num_stories),
            AggScore(metric_name='quc_score',
                    score=sum(story_averages['quc']) / num_stories if num_stories > 0 else 0.0,
                    num=num_stories),
        ]

        # === 2. Calculate per-constraint-level scores ===
        for level in sorted(constraint_level_groups.keys()):
            if level == 0:  # Skip invalid constraint level
                continue

            level_stories = constraint_level_groups[level]
            num_level_stories = len(level_stories)

            # Calculate average across stories for this constraint level
            level_metrics = {
                'fluency': [],
                'grammar': [],
                'coherence': [],
                'likability': [],
                'flexibility': [],
                'appropriateness': [],
                'novelty': [],
                'quc': []
            }

            for story_id, metrics in level_stories.items():
                for metric_name, value in metrics.items():
                    level_metrics[metric_name].append(value)

            # Add scores for this constraint level
            agg_scores.extend([
                AggScore(metric_name=f'fluency_c{level}',
                        score=sum(level_metrics['fluency']) / num_level_stories if num_level_stories > 0 else 0.0,
                        aggregation_name=f'constraints_{level}',
                        num=num_level_stories),
                AggScore(metric_name=f'grammar_score_c{level}',
                        score=sum(level_metrics['grammar']) / num_level_stories if num_level_stories > 0 else 0.0,
                        aggregation_name=f'constraints_{level}',
                        num=num_level_stories),
                AggScore(metric_name=f'coherence_score_c{level}',
                        score=sum(level_metrics['coherence']) / num_level_stories if num_level_stories > 0 else 0.0,
                        aggregation_name=f'constraints_{level}',
                        num=num_level_stories),
                AggScore(metric_name=f'likability_score_c{level}',
                        score=sum(level_metrics['likability']) / num_level_stories if num_level_stories > 0 else 0.0,
                        aggregation_name=f'constraints_{level}',
                        num=num_level_stories),
                AggScore(metric_name=f'flexibility_c{level}',
                        score=sum(level_metrics['flexibility']) / num_level_stories if num_level_stories > 0 else 0.0,
                        aggregation_name=f'constraints_{level}',
                        num=num_level_stories),
                AggScore(metric_name=f'appropriateness_c{level}',
                        score=sum(level_metrics['appropriateness']) / num_level_stories if num_level_stories > 0 else 0.0,
                        aggregation_name=f'constraints_{level}',
                        num=num_level_stories),
                AggScore(metric_name=f'novelty_c{level}',
                        score=sum(level_metrics['novelty']) / num_level_stories if num_level_stories > 0 else 0.0,
                        aggregation_name=f'constraints_{level}',
                        num=num_level_stories),
                AggScore(metric_name=f'quc_score_c{level}',
                        score=sum(level_metrics['quc']) / num_level_stories if num_level_stories > 0 else 0.0,
                        aggregation_name=f'constraints_{level}',
                        num=num_level_stories),
            ])

            logger.info(f"Constraint level {level}: QUC={sum(level_metrics['quc']) / num_level_stories if num_level_stories > 0 else 0.0:.4f} (n={num_level_stories} stories)")

        # === 3. Calculate RCS (Relative Creativity Score) if multiple constraint levels ===
        levels = sorted([k for k in constraint_level_groups.keys() if k > 0])
        if len(levels) >= 2:
            # Calculate QUC for highest and lowest constraint levels
            quc_low_stories = [constraint_level_groups[levels[0]][sid]['quc']
                             for sid in constraint_level_groups[levels[0]]]
            quc_high_stories = [constraint_level_groups[levels[-1]][sid]['quc']
                              for sid in constraint_level_groups[levels[-1]]]

            quc_low = sum(quc_low_stories) / len(quc_low_stories) if quc_low_stories else 0.0
            quc_high = sum(quc_high_stories) / len(quc_high_stories) if quc_high_stories else 0.0
            rcs = quc_high - quc_low

            agg_scores.append(AggScore(metric_name='rcs_score',
                                      score=rcs,
                                      num=num_stories))
            logger.info(f"RCS calculated: {rcs:.4f} (QUC_high@{levels[-1]}={quc_high:.4f}, QUC_low@{levels[0]}={quc_low:.4f})")
        else:
            agg_scores.append(AggScore(metric_name='rcs_score',
                                      score=0.0,
                                      num=num_stories))

        logger.info(f"Total aggregated metrics: {len(agg_scores)}")
        return agg_scores
