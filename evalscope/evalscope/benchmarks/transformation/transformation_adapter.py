# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import os
import re
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.metric import AggScore, Score, SampleScore
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.metrics.llm_judge import LLMJudge
from evalscope.benchmarks.judge_cache import BenchmarkJudgeCache
from evalscope.utils.logger import get_logger

logger = get_logger()

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATASET_PATH = (
    _REPO_ROOT
    / 'dataprocess'
    / 'transformation'
    / 'generated'
    / 'final_runs'
    / 'transformation_eval_1235_all.json'
)
_DEFAULT_PROMPT_CATALOG_PATH = _REPO_ROOT / 'dataprocess' / 'transformation' / 'prompt_catalog.json'
_DEFAULT_JUDGE_CONFIG_PATH = _REPO_ROOT / 'run' / 'llm_judge.json'

STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'in', 'into', 'is', 'it', 'its', 'of', 'on',
    'or', 'that', 'the', 'their', 'then', 'there', 'these', 'this', 'to', 'under', 'use', 'using', 'with', 'within',
    'must', 'should', 'can', 'cannot', 'will', 'while', 'through', 'across', 'more', 'than', 'each', 'same', 'new',
    'old', 'system', 'systems', 'goal', 'goals', 'rule', 'rules',
}

ACTION_MARKERS = {
    'adopt', 'allocate', 'audit', 'bridge', 'calibrate', 'classify', 'convert', 'coordinate', 'deploy', 'design',
    'dispatch', 'enforce', 'escalate', 'fallback', 'govern', 'handoff', 'inspect', 'label', 'log', 'migrate',
    'monitor', 'phase', 'protocol', 'rebuild', 'reconcile', 'record', 'reroute', 'restore', 'rollback', 'route',
    'schedule', 'segment', 'sequence', 'stabilize', 'standard', 'train', 'triage', 'validate', 'verify', 'workflow',
}

NOVELTY_MARKERS = {
    'bridge', 'compatibility', 'fallback', 'gate', 'layer', 'lineage', 'mechanism', 'path-dependent', 'phase',
    'protocol', 'reclassify', 'redefine', 'relabel', 'replace', 'routing', 'segmentation', 'sequence', 'tier',
    'translation', 'two-track',
}

NOVELTY_SCORE_RUBRIC = [
    "0 = no meaningful transformational invention; the answer is empty, mostly restates the item, violates the rule world, or only names the new rule.",
    "1 = generic repair or surface variation: standard rollout, training, audit, monitoring, governance, automation, personalization, dashboarding, or coordination advice without changing the problem's underlying frame.",
    "2 = competent expected reconstruction: the answer correctly implements the rewritten rule world with concrete, item-specific mechanisms, but mainly operationalizes the premise already provided by the item or reproduces a known/historically expected solution path.",
    "3 = strong non-obvious reconstruction, not yet a paradigm shift: the answer identifies an unstated but important intermediate abstraction, failure model, coordination principle, measurement frame, or validity condition, and uses it to organize several modules beyond direct rule implementation.",
    "4 = paradigm-shift candidate: the answer replaces a hidden false premise of the old system with a new generative frame that changes what counts as evidence, success, responsibility, valid state, or legitimate action across the system.",
    "5 = axiom-level transformation: the answer creates a new reasoning space and a family of practices that could not be obtained by improving, completing, or engineering the expected solution.",
]

APPROPRIATENESS_SCORE_RUBRIC = [
    "0 = unworkable, mostly a summary, contradicts the rewritten rule world, or fails task completion/required goal/rule coverage.",
    "1 = superficially relevant but generic, underspecified, or dependent on vague authority/infrastructure; it may sound plausible but lacks enforceable rule-goal traceability.",
    "2 = partially workable reconstruction with concrete mechanisms, but substantial gaps remain in rule integration, validation, migration, failure handling, or fit to the legacy environment.",
    "3 = strong expected reconstruction: compatible with the active rules and goals, operationally detailed, and robust as an engineering plan, but still mainly runs the rule world that the item already specifies.",
    "4 = appropriate paradigm reconstruction: the answer's non-obvious new frame is not only creative but runnable, with workflows/interfaces/records/institutions/terminology/validation/fallback coherently derived from that frame.",
    "5 = axiom-level operational reconstruction: the proposed new axioms are operationally complete, self-consistent, failure-aware, and capable of replacing the old system's evaluation and coordination logic without major hidden dependencies.",
]

def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _resolve_judge_config(entry: Dict[str, Any]) -> Optional[Dict[str, str]]:
    if not isinstance(entry, dict):
        logger.warning('Invalid transformation judge config entry: %s', entry)
        return None
    api_url = os.path.expandvars(str(entry.get('api_url') or '').strip())
    model_id = os.path.expandvars(str(entry.get('model_id') or entry.get('model') or entry.get('name') or '').strip())
    api_key = os.path.expandvars(str(entry.get('api_key', 'EMPTY')))
    api_key_env = entry.get('api_key_env')
    if api_key_env:
        api_key = os.getenv(str(api_key_env), api_key)
    if api_key in {'', 'YOUR_API_KEY'}:
        api_key = os.getenv('EVALSCOPE_API_KEY', api_key)
    if api_key in {'', 'YOUR_API_KEY'}:
        api_key = os.getenv('OPENAI_API_KEY', api_key)
    if not api_url or not model_id:
        logger.warning('Invalid transformation judge config entry: %s', entry)
        return None
    return {'api_url': api_url, 'api_key': api_key, 'model_id': model_id}


def _load_default_judge_configs() -> List[Dict[str, str]]:
    if not _DEFAULT_JUDGE_CONFIG_PATH.exists():
        return []
    payload = _load_json(_DEFAULT_JUDGE_CONFIG_PATH)
    models = payload.get('models', [])
    if not isinstance(models, list) or not models:
        raise ValueError(f'Invalid judge config, missing models list: {_DEFAULT_JUDGE_CONFIG_PATH}')
    resolved = [_resolve_judge_config(entry) for entry in models]
    configs = [config for config in resolved if config]
    if not configs:
        raise ValueError(f'Invalid judge config entries: {_DEFAULT_JUDGE_CONFIG_PATH}')
    return configs


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


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == '.jsonl':
        records: List[Dict[str, Any]] = []
        with path.open('r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    records.append(payload)
        return records

    payload = _load_json(path)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError(f'Unsupported transformation dataset format: {path}')


def _load_prompt_catalog(path: Path) -> Dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f'Prompt catalog must be a JSON object: {path}')
    return payload


def _get_prompt_text(catalog: Dict[str, Any], key: str) -> str:
    for entry in catalog.get('prompts', []):
        if isinstance(entry, dict) and entry.get('key') == key:
            return str(entry.get('content', ''))
    raise KeyError(f'Prompt key not found in catalog: {key}')


def _strip_think_blocks(text: str) -> str:
    return re.sub(r'<think>\s*.*?\s*</think>\s*', '', text, flags=re.DOTALL | re.IGNORECASE).strip()


def _extract_answer_content(text: str) -> str:
    text = (text or '').strip()

    full_match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, flags=re.DOTALL | re.IGNORECASE)
    if full_match:
        return _strip_think_blocks(full_match.group(1))

    open_match = re.search(r'<answer>\s*', text, flags=re.IGNORECASE)
    if open_match:
        tail = text[open_match.end():].replace('</answer>', '')
        return _strip_think_blocks(tail).strip()

    cleaned = _strip_think_blocks(text)
    return re.sub(r'</?answer>', '', cleaned, flags=re.IGNORECASE).strip()


def _truncate_for_log(text: str, limit: int = 1200) -> str:
    normalized = (text or '').strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + '...<truncated>'


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = _extract_answer_content(text)

    fenced = re.search(r'```json\s*(\{.*\})\s*```', cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
        try:
            return json.loads(candidate)
        except Exception as exc:
            raise ValueError(
                f'Failed to parse fenced JSON object: {exc}. Raw response: {_truncate_for_log(cleaned)}'
            ) from exc

    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f'Judge response does not contain a JSON object. Raw response: {_truncate_for_log(cleaned)}')
    candidate = cleaned[start:end + 1]
    try:
        return json.loads(candidate)
    except Exception as exc:
        raise ValueError(
            f'Failed to parse JSON object: {exc}. Raw response: {_truncate_for_log(cleaned)}'
        ) from exc


def _parse_score_mapping(payload: Any, expected_keys: List[str]) -> Dict[str, bool]:
    source = payload if isinstance(payload, dict) else {}
    return {key: (source.get(key) is True) for key in expected_keys}


def _is_half_step(value: float) -> bool:
    return 0.0 <= value <= 5.0 and abs(value * 2 - round(value * 2)) < 1e-9


def _parse_required_score(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{field_name} must be a 0-5 score in 0.5 increments, got boolean: {value}')
    if isinstance(value, int):
        if _is_half_step(float(value)):
            return float(value)
        raise ValueError(f'{field_name} must be in [0, 5] with 0.5 increments, got: {value}')
    if isinstance(value, float):
        if _is_half_step(value):
            return float(value)
        raise ValueError(f'{field_name} must be a 0-5 score in 0.5 increments, got float: {value}')
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r'(?:[0-4](?:\.0|\.5)?|5(?:\.0)?)', stripped):
            parsed = float(stripped)
            if _is_half_step(parsed):
                return parsed
        raise ValueError(f'{field_name} must be a 0-5 score in 0.5 increments, got string: {value!r}')
    raise ValueError(f'{field_name} must be a 0-5 score in 0.5 increments, got {type(value).__name__}')


def _count_true(mapping: Dict[str, bool]) -> int:
    return sum(1 for value in mapping.values() if value)


def _all_true(mapping: Dict[str, bool]) -> bool:
    return all(mapping.values()) if mapping else False


def _tokenize(text: str) -> List[str]:
    return re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)?', (text or '').lower())


def _keyword_set(text: str, limit: int = 16) -> set[str]:
    words: List[str] = []
    seen = set()
    for token in _tokenize(text):
        if len(token) < 4 or token in STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        words.append(token)
        if len(words) >= limit:
            break
    return set(words)


def _overlap_ratio(reference_text: str, answer_tokens: set[str], limit: int = 16) -> float:
    reference_keywords = _keyword_set(reference_text, limit=limit)
    if not reference_keywords:
        return 0.0
    return len(reference_keywords & answer_tokens) / len(reference_keywords)


def _clipped_score(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _heuristic_score_response(item: Dict[str, Any], answer_text: str) -> Dict[str, Any]:
    answer_tokens_list = _tokenize(answer_text)
    answer_tokens = set(answer_tokens_list)
    answer_word_count = len(answer_tokens_list)

    goals = item.get('goals', [])
    rules = item.get('changeable_rules', [])

    goal_scores: Dict[str, float] = {}
    goal_hits = 0
    for goal in goals:
        goal_id = goal.get('goal_id', 'goal')
        score = _overlap_ratio(goal.get('text', ''), answer_tokens)
        goal_scores[str(goal_id)] = round(score, 3)
        if score >= 0.2:
            goal_hits += 1

    rule_scores: Dict[str, float] = {}
    rule_hits = 0
    for rule in rules:
        rule_id = rule.get('rule_id', 'rule')
        score = _overlap_ratio(rule.get('text', ''), answer_tokens)
        rule_scores[str(rule_id)] = round(score, 3)
        if score >= 0.14:
            rule_hits += 1

    action_marker_hits = sum(1 for marker in ACTION_MARKERS if marker in answer_tokens)
    novelty_marker_hits = sum(1 for marker in NOVELTY_MARKERS if marker in answer_text.lower())
    paragraph_count = len([part for part in re.split(r'\n\s*\n', answer_text) if part.strip()])
    numbered_step_count = len(re.findall(r'(?m)^\s*(?:\d+\.)\s+', answer_text))
    bullet_count = len(re.findall(r'(?m)^\s*[-*]\s+', answer_text))
    quantitative_mentions = len(re.findall(r'\b\d+(?:\.\d+)?%?\b', answer_text))
    scenario_overlap = _overlap_ratio(item.get('scenario_text', ''), answer_tokens, limit=24)

    goal_coverage = goal_hits / len(goals) if goals else 0.0
    rule_coverage = rule_hits / len(rules) if rules else 0.0
    actionability = (
        0.35 * _clipped_score(action_marker_hits / 8.0, 0.0, 1.0)
        + 0.25 * _clipped_score((paragraph_count + numbered_step_count + bullet_count) / 6.0, 0.0, 1.0)
        + 0.25 * _clipped_score(quantitative_mentions / 6.0, 0.0, 1.0)
        + 0.15 * _clipped_score(answer_word_count / 300.0, 0.0, 1.0)
    )

    task_completion = (
        answer_word_count >= 120
        and goal_hits == len(goals)
        and rule_scores.get('R1', 0.0) >= 0.12
        and rule_hits >= min(3, len(rules))
        and actionability >= 0.45
    )
    fluency = 1 if task_completion else 0

    novelty = None
    appropriateness = None
    if fluency == 1:
        novelty_raw = (
            1.5
            + 1.4 * _clipped_score(novelty_marker_hits / 5.0, 0.0, 1.0)
            + 1.2 * _clipped_score(action_marker_hits / 8.0, 0.0, 1.0)
            + 1.0 * _clipped_score((1.0 - scenario_overlap) / 0.8, 0.0, 1.0)
            + 0.9 * _clipped_score(quantitative_mentions / 6.0, 0.0, 1.0)
        )
        novelty = int(round(_clipped_score(novelty_raw, 0.0, 5.0)))

        appropriateness_raw = (
            1.3
            + 1.4 * goal_coverage
            + 1.2 * rule_coverage
            + 0.9 * actionability
            + 0.2 * _clipped_score(quantitative_mentions / 6.0, 0.0, 1.0)
        )
        appropriateness = int(round(_clipped_score(appropriateness_raw, 0.0, 5.0)))

    return {
        'mode': 'heuristic',
        'fluency': fluency,
        'novelty': novelty,
        'appropriateness': appropriateness,
        'task_completion': task_completion,
        'goal_scores': goal_scores,
        'rule_scores': rule_scores,
        'signals': {
            'answer_word_count': answer_word_count,
            'goal_coverage': round(goal_coverage, 3),
            'rule_coverage': round(rule_coverage, 3),
            'actionability': round(actionability, 3),
            'action_marker_hits': action_marker_hits,
            'novelty_marker_hits': novelty_marker_hits,
            'paragraph_count': paragraph_count,
            'numbered_step_count': numbered_step_count,
            'bullet_count': bullet_count,
            'quantitative_mentions': quantitative_mentions,
            'scenario_overlap': round(scenario_overlap, 3),
        },
        'judge_notes': (
            'Heuristic proxy only. Fluency is based on goal/rule keyword coverage, '
            'minimum response length, and actionability signals.'
        ),
    }


@register_benchmark(
    BenchmarkMeta(
        name='transformation',
        pretty_name='Transformational Creativity',
        tags=[Tags.REASONING, Tags.INSTRUCTION_FOLLOWING, Tags.CUSTOM],
        description=(
            'Transformational Creativity benchmark. Models must rebuild a system under a rewritten rule world and '
            'are scored on fluency, novelty, appropriateness, and cross-constraint flexibility.'
        ),
        dataset_id=str(_DEFAULT_DATASET_PATH),
        subset_list=['default'],
        default_subset='default',
        metric_list=['fluency', 'novelty', 'appropriateness', 'flexibility'],
        eval_split='test',
        prompt_template='{query}',
        review_timeout=180,
        extra_params={
            'evaluation_mode': {
                'type': 'str',
                'description': 'Scoring mode: "heuristic" or "llm_judge".',
                'value': 'llm_judge',
            },
            'dataset_path': {
                'type': 'str',
                'description': 'Optional path to the benchmark item JSON/JSONL file.',
                'value': None,
            },
            'prompt_catalog_path': {
                'type': 'str',
                'description': 'Optional path to the shared prompt catalog JSON.',
                'value': str(_DEFAULT_PROMPT_CATALOG_PATH),
            },
            'answer_system_prompt_key': {
                'type': 'str',
                'description': 'Prompt-catalog key for the generation system prompt.',
                'value': 'answer_sampling_system_v1',
            },
            'answer_user_prompt_key': {
                'type': 'str',
                'description': 'Prompt-catalog key for the generation user template.',
                'value': 'answer_sampling_user_template_v1',
            },
            'judge_system_prompt_key': {
                'type': 'str',
                'description': 'Prompt-catalog key for the LLM judge system prompt.',
                'value': 'evaluation_llm_judge_system_v1',
            },
            'judge_user_prompt_key': {
                'type': 'str',
                'description': 'Prompt-catalog key for the LLM judge user prompt template.',
                'value': 'evaluation_llm_judge_user_template_v1',
            },
            'judge_api_url': {
                'type': 'str',
                'description': 'Optional OpenAI-compatible judge API URL override.',
                'value': None,
            },
            'judge_api_key': {
                'type': 'str',
                'description': 'Optional OpenAI-compatible judge API key override.',
                'value': None,
            },
            'judge_model_id': {
                'type': 'str',
                'description': 'Optional OpenAI-compatible judge model override.',
                'value': None,
            },
            'judge_temperature': {
                'type': 'float',
                'description': 'Judge decoding temperature.',
                'value': 0.0,
            },
            'judge_max_tokens': {
                'type': 'int',
                'description': 'Judge max_tokens.',
                'value': 4096,
            },
            'judge_timeout': {
                'type': 'int',
                'description': 'Judge request timeout in seconds.',
                'value': 180,
            },
            'judge_max_retries': {
                'type': 'int',
                'description': 'Judge retry count after the first attempt fails.',
                'value': 2,
            },
            'judge_sleep_seconds': {
                'type': 'float',
                'description': 'Sleep between judge retries.',
                'value': 1.0,
            },
        },
    )
)
class TransformationAdapter(DefaultDataAdapter):
    """Native evalscope adapter for the Transformational Creativity benchmark."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        params = self.extra_params or {}
        dataset_path = params.get('dataset_path')
        prompt_catalog_path = params.get('prompt_catalog_path') or _DEFAULT_PROMPT_CATALOG_PATH

        self.dataset_path = Path(dataset_path).expanduser().resolve() if dataset_path else _DEFAULT_DATASET_PATH
        self.prompt_catalog_path = Path(prompt_catalog_path).expanduser().resolve()
        self.evaluation_mode = str(params.get('evaluation_mode', 'llm_judge')).strip().lower()
        self.answer_system_prompt_key = str(params.get('answer_system_prompt_key', 'answer_sampling_system_v1'))
        self.answer_user_prompt_key = str(params.get('answer_user_prompt_key', 'answer_sampling_user_template_v1'))
        self.judge_system_prompt_key = str(params.get('judge_system_prompt_key', 'evaluation_llm_judge_system_v1'))
        self.judge_user_prompt_key = str(params.get('judge_user_prompt_key', 'evaluation_llm_judge_user_template_v1'))
        self.judge_temperature = float(params.get('judge_temperature', 0.0))
        self.judge_max_tokens = int(params.get('judge_max_tokens', 4096))
        self.judge_timeout = int(params.get('judge_timeout', self.review_timeout))
        self.judge_max_retries = int(params.get('judge_max_retries', 2))
        self.judge_sleep_seconds = float(params.get('judge_sleep_seconds', 1.0))
        self._judges: Optional[List[LLMJudge]] = None
        self._judges_lock = threading.Lock()
        task_cfg = getattr(self, '_task_config', None)
        self._judge_cache = BenchmarkJudgeCache(
            benchmark_name='transformation',
            work_dir=getattr(task_cfg, 'work_dir', None),
            model_name=getattr(task_cfg, 'model_id', None),
        )
        self._prompt_catalog = _load_prompt_catalog(self.prompt_catalog_path)
        self.system_prompt = _get_prompt_text(self._prompt_catalog, self.answer_system_prompt_key)
        self._answer_user_template = _get_prompt_text(self._prompt_catalog, self.answer_user_prompt_key)
        self._judge_system_prompt = _get_prompt_text(self._prompt_catalog, self.judge_system_prompt_key)
        self._judge_user_template = _get_prompt_text(self._prompt_catalog, self.judge_user_prompt_key)

        if self.evaluation_mode not in {'heuristic', 'llm_judge'}:
            raise ValueError(f'Unsupported transformation evaluation_mode: {self.evaluation_mode}')
        task_cfg = getattr(self, '_task_config', None)
        self.use_batch_scoring = False

        logger.info(
            'TransformationAdapter init: mode=%s dataset=%s prompt_catalog=%s',
            self.evaluation_mode,
            self.dataset_path,
            self.prompt_catalog_path,
        )

    def _load_judge_configs(self) -> List[Dict[str, str]]:
        params = self.extra_params or {}
        api_url = params.get('judge_api_url')
        api_key = params.get('judge_api_key')
        model_id = params.get('judge_model_id')
        if api_url and model_id:
            return [{
                'api_url': str(api_url),
                'api_key': str(api_key or 'EMPTY'),
                'model_id': str(model_id),
            }]

        task_cfg_configs = _load_task_judge_configs(getattr(self, '_task_config', None))
        if task_cfg_configs:
            return task_cfg_configs

        return _load_default_judge_configs()

    def _get_judges(self) -> List[LLMJudge]:
        if self._judges is not None:
            return self._judges

        with self._judges_lock:
            if self._judges is not None:
                return self._judges

            judges: List[LLMJudge] = []
            for config in self._load_judge_configs():
                judge = LLMJudge(
                    api_url=config['api_url'],
                    api_key=config['api_key'],
                    model_id=config['model_id'],
                    generation_config={
                        'temperature': self.judge_temperature,
                        'max_tokens': self.judge_max_tokens,
                        'timeout': self.judge_timeout,
                    },
                )
                judges.append(judge)
                logger.info('Initialized transformation judge using %s', config['model_id'])
            self._judges = judges
            return self._judges

    def _sample_key(self, item: Dict[str, Any]) -> str:
        candidate_id = item.get('candidate_id')
        if candidate_id not in (None, ''):
            return str(candidate_id)
        return str(item.get('source_axiom_id') or 'unknown')

    def _judge_key(self, judge: LLMJudge) -> str:
        return str(getattr(judge, 'model_id', '') or 'unknown_judge')

    def _judge_cache_note(self, record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not record:
            return {}
        note = {
            'model_id': record.get('model_id'),
            'status': record.get('status'),
            'attempts': record.get('attempts'),
            'cached': True,
        }
        parsed = record.get('parsed_result') if isinstance(record.get('parsed_result'), dict) else {}
        if parsed:
            note.update({
                'fluency': parsed.get('fluency'),
                'task_completion': parsed.get('task_completion'),
                'goal_coverage': parsed.get('goal_coverage'),
                'rule_compatibility': parsed.get('rule_compatibility'),
                'novelty': parsed.get('novelty'),
                'appropriateness': parsed.get('appropriateness'),
                'judge_notes': parsed.get('judge_notes'),
            })
        if record.get('error'):
            note['error'] = record.get('error')
        if record.get('raw_response'):
            note['raw_response_excerpt'] = _truncate_for_log(str(record.get('raw_response')))
        return note

    def _judge_failure_result(self, error: Exception, response_text: Optional[str] = None) -> Dict[str, Any]:
        return {
            'mode': 'llm_judge_fallback',
            'fluency': 0,
            'novelty': None,
            'appropriateness': None,
            'task_completion': False,
            'goal_scores': {},
            'rule_scores': {},
            'goal_coverage': {},
            'rule_compatibility': {},
            'signals': {},
            'judge_notes': {'error': str(error)},
            'raw_judge_response': response_text or '',
        }

    def _aggregate_judge_results(
        self,
        results: List[Dict[str, Any]],
        raw_responses: List[str],
        judge_notes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not results:
            raise ValueError('No transformation judge results to aggregate.')

        task_completion_votes = [bool(result.get('task_completion')) for result in results]

        def _majority_bool(values: List[bool]) -> bool:
            return sum(1 for value in values if value) >= (len(values) // 2 + 1)

        def _all_judges_map(key: str) -> Dict[str, bool]:
            merged_keys = sorted({
                str(item_key)
                for result in results
                for item_key in (result.get(key, {}) or {}).keys()
            })
            aggregated: Dict[str, bool] = {}
            for item_key in merged_keys:
                votes = [bool((result.get(key, {}) or {}).get(item_key)) for result in results]
                aggregated[item_key] = all(votes) and len(votes) == len(results)
            return aggregated

        def _score_mean(key: str) -> Optional[float]:
            values = [
                float(result.get(key))
                for result in results
                if isinstance(result.get(key), (int, float)) and not isinstance(result.get(key), bool)
            ]
            return round(statistics.mean(values), 4) if values else None

        goal_coverage = _all_judges_map('goal_coverage')
        rule_compatibility = _all_judges_map('rule_compatibility')
        task_completion = all(task_completion_votes) and len(task_completion_votes) == len(results)
        fluency = 1 if task_completion and _all_true(goal_coverage) and _all_true(rule_compatibility) else 0

        novelty_score = _score_mean('novelty') if fluency == 1 else None
        appropriateness_score = _score_mean('appropriateness') if fluency == 1 else None

        aggregate_notes: Dict[str, Any] = {
            'per_judge': judge_notes or [result.get('judge_notes', {}) for result in results]
        }
        if fluency == 1:
            aggregate_notes.update({
                'novelty_mean_raw': novelty_score,
                'appropriateness_mean_raw': appropriateness_score,
            })

        return {
            'mode': 'llm_judge',
            'fluency': fluency,
            'novelty': novelty_score,
            'appropriateness': appropriateness_score,
            'task_completion': task_completion,
            'goal_coverage': goal_coverage,
            'rule_compatibility': rule_compatibility,
            'judge_notes': aggregate_notes,
            'raw_judge_response': raw_responses,
        }

    def load_from_disk(self, dataset_name_or_path=None, subset_list=None, **kwargs):
        data_path = Path(dataset_name_or_path).expanduser().resolve() if dataset_name_or_path else self.dataset_path
        if not data_path.exists():
            raise FileNotFoundError(f'Transformation dataset not found: {data_path}')

        records = _load_records(data_path)
        samples: List[Sample] = []
        for idx, item in enumerate(records):
            prompt = self._answer_user_template.format(
                item_json=json.dumps(item, ensure_ascii=False, indent=2),
            )
            samples.append(
                Sample(
                    input=[ChatMessageUser(content=prompt)],
                    target='',
                    id=idx,
                    metadata={
                        'candidate_id': item.get('candidate_id'),
                        'source_axiom_id': item.get('source_axiom_id'),
                        'constraint_count': item.get('constraint_count'),
                        'item': item,
                    },
                )
            )

        if self.limit is not None:
            if isinstance(self.limit, float) and 0 < self.limit < 1:
                limit_num = int(len(samples) * self.limit)
            else:
                limit_num = int(self.limit)
            samples = samples[:limit_num]

        logger.info('Loaded %d transformation samples from %s', len(samples), data_path)
        return {'default': samples}, None

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        return _extract_answer_content(prediction)

    def _build_judge_messages(self, item: Dict[str, Any], answer_text: str) -> List[Any]:
        goals = item.get('goals', []) if isinstance(item.get('goals'), list) else []
        goal_ids = [str(goal.get('goal_id')) for goal in goals if isinstance(goal, dict) and goal.get('goal_id')]
        active_rules = item.get('changeable_rules', []) if isinstance(item.get('changeable_rules'), list) else []
        rule_ids = [str(rule.get('rule_id')) for rule in active_rules if isinstance(rule, dict) and rule.get('rule_id')]

        goal_schema = ', '.join([f'"{goal_id}": false' for goal_id in goal_ids]) or '"G1": false'
        rule_schema = ', '.join([f'"{rule_id}": false' for rule_id in rule_ids]) or '"R1": false'
        score_values = [value / 2 for value in range(0, 11)]
        output_schema = {
            'type': 'object',
            'additionalProperties': False,
            'required': [
                'task_completion',
                'goal_coverage',
                'rule_compatibility',
                'novelty_score',
                'appropriateness_score',
                'novelty_band_reason',
                'appropriateness_band_reason',
                'novelty_evidence',
                'appropriateness_evidence',
            ],
            'properties': {
                'task_completion': {'type': 'boolean'},
                'goal_coverage': {
                    'type': 'object',
                    'additionalProperties': False,
                    'required': goal_ids or ['G1'],
                    'properties': {
                        goal_id: {'type': 'boolean'}
                        for goal_id in (goal_ids or ['G1'])
                    },
                },
                'rule_compatibility': {
                    'type': 'object',
                    'additionalProperties': False,
                    'required': rule_ids or ['R1'],
                    'properties': {
                        rule_id: {'type': 'boolean'}
                        for rule_id in (rule_ids or ['R1'])
                    },
                },
                'novelty_score': {'type': 'number', 'enum': score_values},
                'appropriateness_score': {'type': 'number', 'enum': score_values},
                'novelty_band_reason': {'type': 'string'},
                'appropriateness_band_reason': {'type': 'string'},
                'novelty_evidence': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'minItems': 1,
                },
                'appropriateness_evidence': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'minItems': 1,
                },
            },
        }
        novelty_rubric = '\n'.join([f'- {line}' for line in NOVELTY_SCORE_RUBRIC])
        appropriateness_rubric = '\n'.join([f'- {line}' for line in APPROPRIATENESS_SCORE_RUBRIC])
        json_shape = (
            '{\n'
            '  "task_completion": false,\n'
            f'  "goal_coverage": {{{goal_schema}}},\n'
            f'  "rule_compatibility": {{{rule_schema}}},\n'
            '  "novelty_score": 0,\n'
            '  "appropriateness_score": 0,\n'
            '  "novelty_band_reason": "",\n'
            '  "appropriateness_band_reason": "",\n'
            '  "novelty_evidence": ["", ""],\n'
            '  "appropriateness_evidence": ["", ""]\n'
            '}'
        )

        user_prompt = self._judge_user_template.format(
            item_json=json.dumps(item, ensure_ascii=False, indent=2),
            answer_text=answer_text,
            goal_ids=json.dumps(goal_ids, ensure_ascii=False),
            rule_ids=json.dumps(rule_ids, ensure_ascii=False),
            novelty_rubric=novelty_rubric,
            appropriateness_rubric=appropriateness_rubric,
            json_shape=json_shape,
            json_schema=json.dumps(output_schema, ensure_ascii=False, indent=2),
        )
        return [
            ChatMessageSystem(content=self._judge_system_prompt),
            ChatMessageUser(content=user_prompt),
        ]

    def _llm_judge_score_response(self, item: Dict[str, Any], answer_text: str) -> Dict[str, Any]:
        judges = self._get_judges()
        sample_key = self._sample_key(item)
        results: List[Dict[str, Any]] = []
        raw_responses: List[str] = []
        judge_notes: List[Dict[str, Any]] = []

        for judge in judges:
            judge_key = self._judge_key(judge)
            cached = self._judge_cache.get(sample_key, 'default', judge_key)
            if cached and cached.get('status') == 'success':
                parsed_payload = cached.get('parsed_result')
                if isinstance(parsed_payload, dict):
                    results.append(parsed_payload)
                    raw_responses.append(str(cached.get('raw_response') or ''))
                    judge_notes.append(self._judge_cache_note(cached))
                    continue

            last_error: Optional[str] = None
            last_response_text = ''
            for attempt in range(1, self.judge_max_retries + 2):
                try:
                    response_text = judge.judge(messages=self._build_judge_messages(item, answer_text))
                    last_response_text = response_text
                    parsed_result = self._parse_llm_judge_response(response_text)
                    self._judge_cache.put(
                        sample_key,
                        'default',
                        judge_key,
                        {
                            'benchmark': 'transformation',
                            'status': 'success',
                            'model_id': judge_key,
                            'attempts': attempt,
                            'raw_response': response_text,
                            'parsed_result': parsed_result,
                        },
                    )
                    results.append(parsed_result)
                    raw_responses.append(response_text)
                    judge_notes.append({
                        'model_id': judge_key,
                        'status': 'success',
                        'attempts': attempt,
                        'cached': False,
                        'fluency': parsed_result.get('fluency'),
                        'task_completion': parsed_result.get('task_completion'),
                        'goal_coverage': parsed_result.get('goal_coverage'),
                        'rule_compatibility': parsed_result.get('rule_compatibility'),
                        'novelty': parsed_result.get('novelty'),
                        'appropriateness': parsed_result.get('appropriateness'),
                        'judge_notes': parsed_result.get('judge_notes'),
                        'raw_response_excerpt': _truncate_for_log(response_text),
                    })
                    break
                except Exception as exc:
                    last_error = str(exc)
                    if attempt <= self.judge_max_retries:
                        time.sleep(self.judge_sleep_seconds)
                        continue
                    raw_excerpt = _truncate_for_log(last_response_text) if last_response_text else ''
                    detail = last_error
                    if raw_excerpt and raw_excerpt not in detail:
                        detail = f'{detail}. Last raw response: {raw_excerpt}'
                    self._judge_cache.put(
                        sample_key,
                        'default',
                        judge_key,
                        {
                            'benchmark': 'transformation',
                            'status': 'failed',
                            'model_id': judge_key,
                            'attempts': attempt,
                            'raw_response': last_response_text,
                            'error': detail,
                        },
                    )
                    raise RuntimeError(
                        f'Transformation judge failed for judge={judge_key} after {self.judge_max_retries + 1} attempts: {detail}'
                    )

        return self._aggregate_judge_results(results, raw_responses, judge_notes=judge_notes)

    def _parse_llm_judge_response(self, response_text: str) -> Dict[str, Any]:
        parsed = _extract_json_object(response_text)

        goal_coverage = {
            str(key): bool(value)
            for key, value in (parsed.get('goal_coverage', {}) or {}).items()
            if isinstance(key, str)
        }
        rule_compatibility = {
            str(key): bool(value)
            for key, value in (parsed.get('rule_compatibility', {}) or {}).items()
            if isinstance(key, str)
        }
        task_completion = bool(parsed.get('task_completion'))
        fluency = 1 if task_completion and _all_true(goal_coverage) and _all_true(rule_compatibility) else 0

        novelty = _parse_required_score(parsed.get('novelty_score'), 'novelty_score') if fluency == 1 else None
        appropriateness = _parse_required_score(parsed.get('appropriateness_score'), 'appropriateness_score') if fluency == 1 else None

        novelty_band_reason = parsed.get('novelty_band_reason', '')
        appropriateness_band_reason = parsed.get('appropriateness_band_reason', '')
        novelty_evidence = parsed.get('novelty_evidence', [])
        appropriateness_evidence = parsed.get('appropriateness_evidence', [])

        if fluency == 1:
            if not isinstance(novelty_band_reason, str):
                raise ValueError(f'novelty_band_reason must be a string, got {type(novelty_band_reason).__name__}')
            if not isinstance(appropriateness_band_reason, str):
                raise ValueError(f'appropriateness_band_reason must be a string, got {type(appropriateness_band_reason).__name__}')
            if not isinstance(novelty_evidence, list) or not all(isinstance(x, str) for x in novelty_evidence):
                raise ValueError('novelty_evidence must be a list of strings')
            if not isinstance(appropriateness_evidence, list) or not all(isinstance(x, str) for x in appropriateness_evidence):
                raise ValueError('appropriateness_evidence must be a list of strings')

        return {
            'mode': 'llm_judge',
            'fluency': fluency,
            'novelty': novelty,
            'appropriateness': appropriateness,
            'task_completion': task_completion,
            'goal_coverage': goal_coverage,
            'rule_compatibility': rule_compatibility,
            'judge_notes': {
                'status': 'success',
                'novelty_band_reason': novelty_band_reason,
                'appropriateness_band_reason': appropriateness_band_reason,
                'novelty_evidence': novelty_evidence,
                'appropriateness_evidence': appropriateness_evidence,
            },
            'raw_judge_response': response_text,
        }

    def _build_score(self, metadata: Dict[str, Any], original_prediction: str, filtered_prediction: str,
                     result: Dict[str, Any]) -> Score:
        required_keys = ['fluency', 'novelty', 'appropriateness']
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            raise ValueError(
                f"Transformation score result missing keys {missing_keys} for candidate_id={metadata.get('candidate_id')}"
            )

        novelty_value = result['novelty']
        if novelty_value is None:
            novelty_value = 0.0

        appropriateness_value = result['appropriateness']
        if appropriateness_value is None:
            appropriateness_value = 0.0

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        score.value = {
            'fluency': float(result['fluency']),
            'novelty': float(novelty_value),
            'appropriateness': float(appropriateness_value),
            'flexibility': float(result['fluency']),
        }
        score.metadata = {
            'candidate_id': metadata.get('candidate_id'),
            'source_axiom_id': metadata.get('source_axiom_id'),
            'constraint_count': metadata.get('constraint_count'),
            'evaluation_mode': result.get('mode'),
            'task_completion': result.get('task_completion'),
            'goal_scores': result.get('goal_scores'),
            'rule_scores': result.get('rule_scores'),
            'goal_coverage': result.get('goal_coverage'),
            'rule_compatibility': result.get('rule_compatibility'),
            'signals': result.get('signals'),
            'judge_notes': result.get('judge_notes'),
            'novelty_raw': result.get('novelty'),
            'appropriateness_raw': result.get('appropriateness'),
            'raw_judge_response': result.get('raw_judge_response'),
        }
        score.main_score_name = 'fluency'
        return score

    def defer_score_calculation_to_batch(self) -> bool:
        return False

    def batch_match_score(
        self,
        original_predictions: List[str],
        filtered_predictions: List[str],
        references: List[str],
        task_states: List[TaskState],
    ) -> Optional[List[Score]]:
        return None

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        metadata = task_state.metadata or {}
        item = metadata.get('item') or {}
        answer_text = filtered_prediction.strip()

        if self.evaluation_mode == 'heuristic':
            result = _heuristic_score_response(item, answer_text)
        else:
            try:
                result = self._llm_judge_score_response(item, answer_text)
            except Exception as exc:
                raise RuntimeError(
                    f"Transformation judge failed for candidate_id={metadata.get('candidate_id')}: {exc}"
                ) from exc
        return self._build_score(metadata, original_prediction, filtered_prediction, result)

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        total = len(sample_scores)
        if total == 0:
            return []

        fluency_values: List[float] = []
        novelty_values: List[float] = []
        appropriateness_values: List[float] = []
        per_source: Dict[str, List[Dict[str, Any]]] = {}

        for sample_score in sample_scores:
            score_value = sample_score.score.value or {}
            metadata = sample_score.score.metadata or {}
            fluency = float(score_value.get('fluency', 0.0))
            fluency_values.append(fluency)

            novelty = score_value.get('novelty')
            if isinstance(novelty, (int, float)):
                novelty_values.append(float(novelty))

            appropriateness = score_value.get('appropriateness')
            if isinstance(appropriateness, (int, float)):
                appropriateness_values.append(float(appropriateness))

            source_axiom_id = metadata.get('source_axiom_id')
            if source_axiom_id:
                per_source.setdefault(str(source_axiom_id), []).append(
                    {
                        'constraint_count': metadata.get('constraint_count'),
                        'fluency': fluency,
                    }
                )

        flexibility_by_source: Dict[str, Dict[str, Any]] = {}
        for source_axiom_id, rows in per_source.items():
            level_map: Dict[int, List[float]] = {}
            for row in rows:
                level = row.get('constraint_count')
                if not isinstance(level, int):
                    continue
                level_map.setdefault(level, []).append(float(row['fluency']))

            if len(level_map) < 2:
                continue

            weighted_numerator = 0.0
            weighted_denominator = 0.0
            level_details: Dict[str, float] = {}
            for level in sorted(level_map):
                mean_fluency = statistics.mean(level_map[level])
                weighted_numerator += level * mean_fluency
                weighted_denominator += level
                level_details[str(level)] = round(mean_fluency, 4)

            flexibility_by_source[source_axiom_id] = {
                'levels': level_details,
                'flexibility': round(weighted_numerator / weighted_denominator, 4),
            }

        flexibility_values = [row['flexibility'] for row in flexibility_by_source.values()]
        success_count = int(round(sum(fluency_values)))
        fluency_rate = round(statistics.mean(fluency_values), 4) if fluency_values else 0.0
        novelty_mean = round(statistics.mean(novelty_values), 4) if novelty_values else 0.0
        appropriateness_mean = round(statistics.mean(appropriateness_values), 4) if appropriateness_values else 0.0
        flexibility_mean = round(statistics.mean(flexibility_values), 4) if flexibility_values else 0.0

        common_metadata = {
            'item_count': total,
            'success_count': success_count,
            'flexibility_available_groups': len(flexibility_values),
            'flexibility_by_source_axiom_id': flexibility_by_source,
        }

        return [
            AggScore(
                metric_name='fluency',
                score=fluency_rate,
                num=total,
                metadata={**common_metadata, 'score_scale': '0/1'},
            ),
            AggScore(
                metric_name='novelty',
                score=novelty_mean,
                num=len(novelty_values),
                metadata={**common_metadata, 'score_scale': '0-5 on successful items only'},
            ),
            AggScore(
                metric_name='appropriateness',
                score=appropriateness_mean,
                num=len(appropriateness_values),
                metadata={**common_metadata, 'score_scale': '0-5 on successful items only'},
            ),
            AggScore(
                metric_name='flexibility',
                score=flexibility_mean,
                num=len(flexibility_values),
                metadata={
                    **common_metadata,
                    'score_scale': (
                        'weighted mean fluency across available constraint levels within the same source_axiom_id'
                    ),
                },
            ),
        ]
