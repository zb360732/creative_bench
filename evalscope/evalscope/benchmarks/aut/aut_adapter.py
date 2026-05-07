# Copyright (c) Alibaba, Inc. and its affiliates.

import difflib
import hashlib
import json
import math
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import AggScore, SampleScore
from evalscope.api.messages import ChatMessageUser, dict_to_chat_message
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_metric
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.metrics.llm_judge import LLMJudge
from evalscope.utils.logger import get_logger

logger = get_logger()

# 获取数据文件的绝对路径
_DATA_DIR = Path(__file__).parent / 'data'
_DEFAULT_DATASET_PATH = str(Path(__file__).resolve().parents[3] / 'custom_eval' / 'text' / 'qa' / 'exploration' / 'aut.json')
_DEFAULT_JUDGE_CONFIG_PATH = Path(__file__).resolve().parents[3] / 'run' / 'llm_judge.json'


@register_benchmark(
    BenchmarkMeta(
        name='aut',
        pretty_name='AUT (Alternative Uses Test)',
        tags=[Tags.REASONING, Tags.INSTRUCTION_FOLLOWING],
        description='Alternative Uses Test (AUT) benchmark for evaluating creative thinking and divergent reasoning. '
                    'The task requires generating multiple creative and alternative uses for common objects.',
        dataset_id=_DEFAULT_DATASET_PATH,  # 使用数据文件的绝对路径
        metric_list=['aut_fluency', 'aut_elaboration', 'aut_flexibility', 'aut_originality'],
        aggregation='mean',
        subset_list=['default'],
        default_subset='default',
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template='{query}',  # 直接使用数据中的query字段（已包含完整prompt）
        extra_params={
            'multi_round': {
                'type': 'bool',
                'description': 'Enable multi-round AUT generation.',
                'value': True,
                'choices': [True, False],
            },
            'max_rounds': {
                'type': 'int',
                'description': 'Maximum rounds for multi-round AUT (capped at 5).',
                'value': 5,
            },
            'stop_on_no_new': {
                'type': 'bool',
                'description': 'Stop when a round adds no new uses.',
                'value': True,
                'choices': [True, False],
            },
            'sample_semantic_dedup': {
                'type': 'bool',
                'description': 'Enable semantic deduplication during multi-round sampling.',
                'value': True,
                'choices': [True, False],
            },
            'sample_llm_dedup': {
                'type': 'bool',
                'description': 'Enable LLM-judge deduplication during multi-round sampling.',
                'value': True,
                'choices': [True, False],
            },
        },
    )
)
class AUTAdapter(DefaultDataAdapter):
    """
    Adapter for Alternative Uses Test (AUT) benchmark.

    AUT evaluates creative thinking by asking participants to generate
    alternative uses for common objects. The evaluation includes:
    - Fluency: Number of unique uses (after semantic deduplication)
    - Elaboration: Level of detail in descriptions
    - Flexibility: Diversity of use categories
    - Originality: Novelty of uses compared to reference data
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        params = self.extra_params or {}
        self.multi_round = bool(params.get('multi_round', True))
        self.max_rounds = int(params.get('max_rounds', 5))
        self.stop_on_no_new = bool(params.get('stop_on_no_new', True))
        self.sample_semantic_dedup = bool(params.get('sample_semantic_dedup', True))
        self.sample_llm_dedup = bool(params.get('sample_llm_dedup', True))
        self._applicability_judge = None
        self._applicability_cache: Dict[str, Any] = {}
        self._extract_judge = None
        self._extract_cache: Dict[str, Any] = {}
        self._fluency_metric = None
        if self.max_rounds < 1:
            self.max_rounds = 1
        if self.max_rounds > 5:
            self.max_rounds = 5

    @staticmethod
    def _mean(values: List[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    @staticmethod
    def _topk_mean(values: List[float], ratio: float) -> float:
        if not values:
            return 0.0
        k = max(1, int(math.ceil(len(values) * ratio)))
        topk = sorted(values)[-k:]
        return float(sum(topk) / len(topk))

    def load_from_disk(self, dataset_name_or_path=None, subset_list=None, **kwargs):
        """
        Load AUT dataset from local JSON file.

        Args:
            dataset_name_or_path: Path to the dataset (defaults to self.dataset_id)
            subset_list: List of subsets to load (not used for AUT)
            **kwargs: Additional arguments

        Returns:
            Tuple[Dict, None]: test_dataset dict and fewshot_dataset (None for AUT)
        """
        import json
        from pathlib import Path

        # 确定数据文件路径
        if dataset_name_or_path is None:
            dataset_name_or_path = self.dataset_id

        data_path = Path(dataset_name_or_path)

        if not data_path.exists():
            raise FileNotFoundError(f'AUT dataset not found at: {data_path}')

        logger.info(f'Loading AUT dataset from: {data_path}')

        # 加载JSON数据
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f'Loaded {len(data)} AUT test items')

        # 应用limit限制（如果设置了）
        if self.limit is not None:
            if isinstance(self.limit, float) and 0 < self.limit < 1:
                # limit是百分比
                limit_num = int(len(data) * self.limit)
            else:
                # limit是具体数量
                limit_num = int(self.limit)
            data = data[:limit_num]
            logger.info(f'Limited to {len(data)} samples (limit={self.limit})')

        # 转换为 Sample 对象列表
        samples = []
        for idx, item in enumerate(data):
            sample = self.record_to_sample(item)
            # 设置样本的index
            sample.id = idx
            samples.append(sample)

        # 包装为Dataset对象
        from datasets import Dataset
        # 创建空的 Dataset，然后直接赋值 samples
        dataset = samples  # 直接使用 list 作为 dataset

        # 返回格式：(test_dataset, fewshot_dataset)
        # AUT 不使用 few-shot，所以返回 None
        return {'default': dataset}, None

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        if not sample_scores:
            return []

        metric_values = defaultdict(list)
        metric_sample_ids = defaultdict(list)

        for sample_score in sample_scores:
            for metric_name, value in sample_score.score.value.items():
                metric_values[metric_name].append(float(value))
                metric_sample_ids[metric_name].append(sample_score.sample_id)

        aggregated_scores: List[AggScore] = []
        for metric_name, values in metric_values.items():
            if metric_name.startswith('aut_originality'):
                aggregated_value = self._topk_mean(values, ratio=0.1)
                metadata = {'actual_aggregation': 'top10_mean'}
            else:
                aggregated_value = self._mean(values)
                metadata = {'actual_aggregation': 'mean'}

            aggregated_scores.append(
                AggScore(
                    score=aggregated_value,
                    metric_name=metric_name,
                    aggregation_name='mean',
                    num=len(values),
                    ids=metric_sample_ids[metric_name],
                    metadata=metadata,
                ))

        return aggregated_scores

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a dataset record to a Sample object.

        Args:
            record: Dict containing:
                - query: Full prompt text (already formatted)
                - item: Object name (e.g., 'box', 'rope')
                - category: Category name (e.g., 'AUT')

        Returns:
            Sample object with:
                - input: The query/prompt text
                - target: The item name (used as reference by metrics)
                - metadata: Additional information for metrics
        """
        return Sample(
            input=record['query'],
            target=record['item'],  # 物品名称传递给metrics作为reference
            metadata={
                'item': record['item'],
                'category': record.get('category', 'AUT'),
            }
        )

    def format_prompt_template(self, sample: Sample) -> str:
        """
        Format the prompt for AUT task with <answer> tags requirement.

        Args:
            sample: Sample object with input containing the full prompt

        Returns:
            str: The complete prompt text with answer tag instructions
        """
        # 在原有prompt基础上添加 <answer> 标签要求
        base_prompt = sample.input

        answer_format_instruction = """

Please provide your answer in the following JSON format inside <answer> tags:

<answer>
{
  "uses": [
    "use 1",
    "use 2",
    "use 3"
  ]
}
</answer>

Remember to put your JSON response inside <answer></answer> tags."""

        return self._append_instruction(base_prompt, answer_format_instruction)

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """
        Extract JSON answer from <answer> tags.

        Args:
            prediction: Model's raw prediction string
            task_state: Current task state (not used)

        Returns:
            Extracted JSON string or original prediction if extraction fails
        """
        # 尝试从 <answer></answer> 标签中提取
        answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(answer_pattern, prediction, re.DOTALL | re.IGNORECASE)

        if match:
            extracted = match.group(1).strip()
            # 验证是否是有效的JSON
            try:
                json.loads(extracted)
                return extracted
            except json.JSONDecodeError:
                logger.warning(f'Extracted content from <answer> tags is not valid JSON: {extracted[:100]}')

        # 如果没有找到标签，尝试直接从文本中提取JSON
        # 查找 {...} 结构
        json_pattern = r'\{[^{}]*"uses"[^{}]*\[[^\]]*\][^{}]*\}'
        match = re.search(json_pattern, prediction, re.DOTALL)
        if match:
            try:
                extracted = match.group(0)
                json.loads(extracted)
                return extracted
            except json.JSONDecodeError:
                pass

        # 如果都失败，返回原始预测
        return prediction.strip()

    def _clone_messages(self, messages: List[Any]) -> List[Any]:
        cloned = []
        for msg in messages:
            if isinstance(msg, dict):
                msg = dict_to_chat_message(msg)
            if hasattr(msg, 'model_copy'):
                cloned.append(msg.model_copy(deep=True))
            else:
                cloned.append(msg)
        return cloned

    def _append_instruction(self, prompt: Any, instruction: str) -> Any:
        if isinstance(prompt, list):
            messages = self._clone_messages(prompt)
            for msg in reversed(messages):
                if getattr(msg, 'role', None) == 'user':
                    msg.text = f'{msg.text}{instruction}'
                    return messages
            messages.append(ChatMessageUser(content=instruction))
            return messages
        return f'{prompt}{instruction}'

    def _normalize_use(self, text: str) -> str:
        raw = unicodedata.normalize('NFKC', str(text))
        cleaned = ''.join(ch for ch in raw if not unicodedata.combining(ch))
        cleaned = cleaned.lower()
        cleaned = re.sub(r'[^\w\s-]', ' ', cleaned, flags=re.UNICODE)
        cleaned = cleaned.replace('_', ' ')
        cleaned = re.sub(r'[-/]+', ' ', cleaned)
        return re.sub(r'\s+', ' ', cleaned).strip()

    def _singularize_token(self, token: str) -> str:
        if token.endswith('ies') and len(token) > 4:
            return f'{token[:-3]}y'
        if token.endswith('s') and len(token) > 3 and not token.endswith('ss'):
            return token[:-1]
        return token

    def _canonicalize_use(self, text: str) -> str:
        stopwords = {
            'a', 'an', 'the', 'of', 'for', 'to', 'and', 'with', 'in', 'on', 'at', 'from', 'by',
            'as', 'into', 'via', 'using', 'use', 'used', 'be', 'is', 'are', 'was', 'were',
            'this', 'that', 'these', 'those', 'your', 'their', 'our', 'its',
        }
        normalized = self._normalize_use(text)
        tokens = [
            self._singularize_token(token)
            for token in normalized.split()
            if token and token not in stopwords
        ]
        return ' '.join(tokens)

    def _is_near_duplicate(self, left: str, right: str) -> bool:
        left_key = self._canonicalize_use(left)
        right_key = self._canonicalize_use(right)
        if not left_key or not right_key:
            return False
        if left_key == right_key:
            return True
        ratio = difflib.SequenceMatcher(None, left_key, right_key).ratio()
        if ratio >= 0.84:
            return True
        left_tokens = set(left_key.split())
        right_tokens = set(right_key.split())
        if not left_tokens or not right_tokens:
            return False
        jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        if jaccard >= 0.75:
            return True
        # Treat a more specific variant as a duplicate of a general one (one extra qualifier).
        if left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens):
            smaller = min(len(left_tokens), len(right_tokens))
            larger = max(len(left_tokens), len(right_tokens))
            if smaller >= 2 and (larger - smaller) <= 1:
                return True
        if left_key.startswith(right_key) or right_key.startswith(left_key):
            shorter = min(len(left_key), len(right_key))
            longer = max(len(left_key), len(right_key))
            if longer and shorter / longer >= 0.8:
                return True
        return False

    def _dedupe_with_history(
        self, uses: List[str], history_uses: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str]]:
        history_uses = history_uses or []
        kept: List[str] = []
        dropped: List[str] = []
        for use in uses:
            if not use or not str(use).strip():
                continue
            if any(self._is_near_duplicate(use, item) for item in history_uses):
                dropped.append(use)
                continue
            if any(self._is_near_duplicate(use, item) for item in kept):
                dropped.append(use)
                continue
            kept.append(use)
        return kept, dropped

    def _align_to_kept(self, uses: List[str], kept: List[str]) -> List[str]:
        if not kept:
            return []
        aligned: List[str] = []
        seen: set = set()
        for use in uses:
            match = None
            for kept_use in kept:
                if self._is_near_duplicate(use, kept_use):
                    match = kept_use
                    break
            if match is None:
                continue
            key = self._normalize_use(match)
            if key in seen:
                continue
            seen.add(key)
            aligned.append(match)
        return aligned

    def _parse_uses_from_prediction(self, prediction: str, item: str = '') -> List[str]:
        uses = self._parse_uses_without_llm(prediction)
        if uses:
            return uses

        if item:
            try:
                recovered = self._extract_uses_with_deepseek(item, prediction)
            except Exception as exc:
                recovered = []
                logger.warning(f'Deepseek extraction failed for {item}: {exc}')
            if recovered:
                logger.info(f'Recovered {len(recovered)} uses via deepseek for {item}')
                return recovered

        logger.warning(f'Failed to parse uses from prediction: {prediction[:100]}')
        return []

    def _parse_uses_without_llm(self, prediction: str) -> List[str]:
        # Try direct JSON
        try:
            data = json.loads(prediction)
            if isinstance(data, dict) and 'uses' in data:
                uses = data['uses']
                if isinstance(uses, list):
                    return [str(u).strip() for u in uses if u and str(u).strip()]
        except json.JSONDecodeError:
            pass

        # Try <answer>...</answer>
        answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(answer_pattern, prediction, re.DOTALL | re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            try:
                data = json.loads(extracted)
                if isinstance(data, dict) and 'uses' in data:
                    uses = data['uses']
                    if isinstance(uses, list):
                        return [str(u).strip() for u in uses if u and str(u).strip()]
            except json.JSONDecodeError:
                pass

        # Try JSON in code block
        code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(code_block_pattern, prediction, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and 'uses' in data:
                    uses = data['uses']
                    if isinstance(uses, list):
                        return [str(u).strip() for u in uses if u and str(u).strip()]
            except json.JSONDecodeError:
                pass
        return []

    def _parse_uses_from_predictions_batch(self, predictions: List[str], items: List[str]) -> List[List[str]]:
        parsed_batches: List[List[str]] = []
        missing_indices: List[int] = []
        missing_items: List[str] = []
        missing_outputs: List[str] = []

        for idx, (prediction, item) in enumerate(zip(predictions, items)):
            uses = self._parse_uses_without_llm(prediction)
            if uses:
                parsed_batches.append(uses)
                continue
            parsed_batches.append([])
            if item:
                missing_indices.append(idx)
                missing_items.append(item)
                missing_outputs.append(prediction)
            else:
                logger.warning(f'Failed to parse uses from prediction: {prediction[:100]}')

        if missing_indices:
            recovered_batches = self._extract_uses_with_deepseek_batch(missing_items, missing_outputs)
            for idx, item, recovered in zip(missing_indices, missing_items, recovered_batches):
                if recovered:
                    logger.info(f'Recovered {len(recovered)} uses via deepseek batch for {item}')
                    parsed_batches[idx] = recovered
                else:
                    logger.warning(f'Failed to parse uses from prediction: {predictions[idx][:100]}')

        return parsed_batches

    def _build_round_prompt(self, base_prompt: Any, previous_uses: List[str], round_index: int) -> Any:
        if not previous_uses:
            return self._append_instruction(base_prompt, '')
        prev_json = json.dumps({'uses': previous_uses}, ensure_ascii=False, indent=2)
        return self._append_instruction(
            base_prompt,
            (
                f'\n\nRound {round_index}:\n'
                f'Here are your previous answers. Do NOT repeat any of them.\n'
                f'<previous_uses>\n{prev_json}\n</previous_uses>\n'
                f'Provide ONLY NEW uses not listed above. Output JSON inside <answer> tags.'
            ),
        )

    def _build_uses_json(self, uses: List[str]) -> str:
        return json.dumps({'uses': uses}, ensure_ascii=False, indent=2)

    def _build_combined_answer(self, uses: List[str]) -> str:
        return f"<answer>\n{self._build_uses_json(uses)}\n</answer>"

    def _load_applicability_judge(self) -> LLMJudge:
        if self._applicability_judge is not None:
            return self._applicability_judge

        if not _DEFAULT_JUDGE_CONFIG_PATH.exists():
            raise FileNotFoundError(f'LLM judge config not found: {_DEFAULT_JUDGE_CONFIG_PATH}')

        data = json.loads(_DEFAULT_JUDGE_CONFIG_PATH.read_text(encoding='utf-8'))
        models = data.get('models', [])
        if not isinstance(models, list) or not models:
            raise ValueError(f'Invalid judge config, missing models list: {_DEFAULT_JUDGE_CONFIG_PATH}')

        entry = models[0]
        model_id = os.path.expandvars(str(entry.get('model') or entry.get('name') or '').strip())
        api_url = os.path.expandvars(str(entry.get('api_url') or '').strip())
        api_key = os.path.expandvars(str(entry.get('api_key', 'EMPTY')))
        api_key_env = entry.get('api_key_env')
        if api_key_env:
            api_key = os.getenv(str(api_key_env), api_key)
        if api_key in {'', 'YOUR_API_KEY'}:
            api_key = os.getenv('EVALSCOPE_API_KEY', api_key)
        if api_key in {'', 'YOUR_API_KEY'}:
            api_key = os.getenv('OPENAI_API_KEY', api_key)

        if not model_id or not api_url:
            raise ValueError(f'Invalid judge config entry: {entry}')

        host = urlparse(api_url).hostname
        if host:
            for env_key in ('NO_PROXY', 'no_proxy'):
                current = os.environ.get(env_key, '')
                parts = [p.strip() for p in current.split(',') if p.strip()]
                for item in (host, 'localhost', '127.0.0.1'):
                    if item not in parts:
                        parts.append(item)
                os.environ[env_key] = ','.join(parts)

        generation_config = entry.get('generation_config') or {'temperature': 0.0, 'max_tokens': 2048}
        self._applicability_judge = LLMJudge(
            model_id=model_id,
            api_url=api_url,
            api_key=api_key,
            generation_config=generation_config,
        )
        return self._applicability_judge

    def _build_applicability_prompt(self, item: str, use: str) -> str:
        return (
            'You are judging whether a proposed use is appropriate for the given item in the AUT task.\n'
            f'Item: {item}\n'
            f'Use: {use}\n\n'
            'Decide if the use is reasonable and logically consistent with the item\'s nature.\n'
            'If the item is abstract, accept only conceptually coherent uses and reject physical-object uses.\n'
            'If the use is nonsensical, physically impossible, or unrelated, mark B.\n\n'
            'Answer with only one letter: A (reasonable) or B (unreasonable).'
        )

    def _build_applicability_batch_prompt(self, item: str, uses: List[str]) -> str:
        numbered = '\n'.join(f'{idx + 1}. {use}' for idx, use in enumerate(uses))
        return (
            'You are judging a list of proposed uses for an AUT item.\n'
            f'Item: {item}\n\n'
            'The list may contain duplicates or near-duplicates and may include unreasonable uses.\n'
            'Your tasks:\n'
            '1) Deduplicate by meaning and keep only the best (most general) variant for each idea.\n'
            '   Treat as duplicates if they only differ by spelling, singular/plural, or minor modifiers\n'
            '   (audience, size, location, time, material, medium, delivery channel).\n'
            '   If one use is just a more specific version of another (adds one extra qualifier),\n'
            '   keep the more general one.\n'
            '   Treat wording changes like "with/for/in/using/including/featuring" clauses as modifiers,\n'
            '   not new ideas.\n'
            '   Collapse "center/experience center/facility/hub/lab" variants into one.\n'
            '   Treat "X-shaped Y" and "Y shaped like X" as duplicates.\n'
            '   If only the target object changes within the same role/category (pet toys/beds/collars),\n'
            '   keep one generalized version.\n'
            '   Examples to merge: "VR center" vs "VR experience center";\n'
            '   "indoor virtual reef" vs "indoor virtual coral reef";\n'
            '   "outdoor oven" vs "outdoor pizza oven";\n'
            '   "birdhouse camera" vs "birdhouse with a camera feed".\n'
            '2) Remove uses that are unreasonable, physically impossible, or unrelated.\n'
            "If the item is abstract, reject physical-object uses and keep only conceptually coherent ones.\n\n"
            'Return JSON ONLY in this format:\n'
            '{\n'
            '  "deduped_uses": ["..."],\n'
            '  "valid_uses": ["..."],\n'
            '  "invalid_uses": ["..."]\n'
            '}\n\n'
            'Rules:\n'
            '- Every entry in valid_uses or invalid_uses must appear in deduped_uses.\n'
            '- Do not invent new uses.\n'
            '- valid_uses and invalid_uses must not overlap.\n\n'
            'Uses:\n'
            f'{numbered}'
        )

    def _build_applicability_incremental_prompt(
        self, item: str, history_uses: List[str], new_uses: List[str]
    ) -> str:
        history_block = '\n'.join(f'{idx + 1}. {use}' for idx, use in enumerate(history_uses)) or 'None'
        new_block = '\n'.join(f'{idx + 1}. {use}' for idx, use in enumerate(new_uses)) or 'None'
        return (
            'You are judging new proposed uses for an AUT item.\n'
            f'Item: {item}\n\n'
            'History uses (already deduplicated and valid). Keep them as-is:\n'
            f'{history_block}\n\n'
            'New uses to evaluate:\n'
            f'{new_block}\n\n'
            'Your tasks:\n'
            '1) Deduplicate new uses by meaning and drop any that overlap with history uses.\n'
            '   Treat as duplicates if they only differ by spelling, singular/plural, or minor modifiers\n'
            '   (audience, size, location, time, material, medium, delivery channel).\n'
            '   If one use is just a more specific version of another (adds one extra qualifier),\n'
            '   keep the more general one.\n'
            '   Treat wording changes like "with/for/in/using/including/featuring" clauses as modifiers,\n'
            '   not new ideas.\n'
            '   Collapse "center/experience center/facility/hub/lab" variants into one.\n'
            '   Treat "X-shaped Y" and "Y shaped like X" as duplicates.\n'
            '   If only the target object changes within the same role/category (pet toys/beds/collars),\n'
            '   keep one generalized version.\n'
            '   Examples to merge: "VR center" vs "VR experience center";\n'
            '   "indoor virtual reef" vs "indoor virtual coral reef";\n'
            '   "outdoor oven" vs "outdoor pizza oven";\n'
            '   "birdhouse camera" vs "birdhouse with a camera feed".\n'
            '2) Remove new uses that are unreasonable, physically impossible, or unrelated.\n'
            'If the item is abstract, reject physical-object uses and keep only conceptually coherent ones.\n\n'
            'Return JSON ONLY in this format:\n'
            '{\n'
            '  "deduped_new_uses": ["..."],\n'
            '  "valid_new_uses": ["..."],\n'
            '  "invalid_new_uses": ["..."]\n'
            '}\n\n'
            'Rules:\n'
            '- Every entry in valid_new_uses or invalid_new_uses must appear in deduped_new_uses.\n'
            '- Do not include any history uses in the output.\n'
            '- Do not invent new uses.\n'
            '- valid_new_uses and invalid_new_uses must not overlap.'
        )

    def _build_deepseek_extract_prompt(self, item: str, raw_output: str) -> str:
        return (
            'You are extracting alternative uses for an AUT item from a raw model output.\n'
            f'Item: {item}\n\n'
            'Raw output:\n'
            f'<raw>\n{raw_output}\n</raw>\n\n'
            'Tasks:\n'
            '1) Extract all candidate uses (ignore refusals, meta commentary, or formatting).\n'
            '2) Normalize each use to a short phrase or sentence.\n'
            '3) Deduplicate by meaning and keep only the best variant for each idea.\n\n'
            'Dedup rules (treat as duplicates):\n'
            '- Same core action + same goal, even if wording or grammar differs.\n'
            '- Synonyms or paraphrases (e.g., \"door wedge\" vs \"doorstop\").\n'
            '- Same use with minor modifiers (audience, location, time, size, material, adjectives).\n'
            '- One use is just a more specific version of another (adds one extra qualifier).\n'
            '- Wording changes using "with/for/in/using/including/featuring" clauses (treat as modifiers).\n'
            '- "center/experience center/facility/hub/lab" variants.\n'
            '- "X-shaped Y" vs "Y shaped like X".\n'
            '- Same idea with spelling/diacritics/singular-plural variants.\n'
            '- Same functional role with only the target object swapped within the same role/category\n'
            '  (e.g., \"decorative element in a wall-mounted mirror\" vs \"decorative element in a wall-mounted mailbox\").\n'
            '- Same base phrase with extra qualifiers (\"online art challenges\" vs \"online art challenges for kids\").\n'
            '- Same product line variants (\"pet products\" vs \"pet toys\" vs \"pet beds\").\n'
            '- Specific subtype vs general type (\"pizza oven\" vs \"oven\").\n\n'
            'Examples to merge:\n'
            '- \"virtual reality center\" vs \"virtual reality experience center\"\n'
            '- \"indoor virtual reef\" vs \"indoor virtual coral reef\"\n'
            '- \"birdhouse camera\" vs \"birdhouse with a camera feed\"\n\n'
            'Keep as distinct when the primary function or mechanism differs\n'
            '(e.g., \"paperweight\" vs \"doorstop\").\n\n'
            'Return JSON ONLY in this format:\n'
            '{\n'
            '  \"uses\": [\"...\", \"...\"]\n'
            '}\n'
            'No extra text.'
        )

    def _semantic_deduplicate(self, uses: List[str]) -> List[str]:
        try:
            from evalscope.metrics.aut_metrics import AUTFluency

            metric = AUTFluency()
            return metric._semantic_deduplicate(uses)
        except Exception as exc:
            logger.warning(f'Applicability dedup fallback: {exc}')
            seen = set()
            unique = []
            for use in uses:
                norm = self._normalize_use(use)
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                unique.append(use)
            return unique

    def _get_fluency_metric(self):
        if self._fluency_metric is None:
            from evalscope.metrics.aut_metrics import AUTFluency

            self._fluency_metric = AUTFluency()
        return self._fluency_metric

    def _rebuild_fluency_metric_cpu(self):
        """Force rebuild AUTFluency singleton on CPU to clear meta tensors."""
        from evalscope.metrics.aut_metrics import AUTFluency

        try:
            AUTFluency._instance = None
        except Exception:
            pass
        metric = AUTFluency()
        metric._ensure_bert_ready(device='cpu')
        self._fluency_metric = metric
        return metric

    def _semantic_filter_new_uses(self, history_uses: List[str], new_uses: List[str]) -> List[str]:
        if not new_uses:
            return []
        try:
            import numpy as np

            metric = self._get_fluency_metric()
            threshold = metric.fluency_similarity_threshold
            combined = list(history_uses) + list(new_uses)
            embeddings = metric._safe_encode(combined, device='cpu')
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
            history_len = len(history_uses)
            kept_embeddings = list(embeddings[:history_len])

            kept: List[str] = []
            for use, emb in zip(new_uses, embeddings[history_len:]):
                if kept_embeddings:
                    sims = np.dot(kept_embeddings, emb)
                    if float(np.max(sims)) >= threshold:
                        continue
                kept.append(use)
                kept_embeddings.append(emb)
            return kept
        except Exception as exc:
            logger.warning(f'Semantic sampling dedup failed: {exc}')
            return self._dedupe_with_history(new_uses, history_uses)[0]

    def _llm_filter_new_uses(
        self, item: str, history_uses: List[str], new_uses: List[str]
    ) -> Optional[List[str]]:
        if not new_uses:
            return []
        try:
            judged = self._judge_applicability_incremental(item, history_uses, new_uses)
            deduped_new = judged.get('deduped_new_uses', []) or []
            valid_new = judged.get('valid_new_uses', []) or []
            picked = valid_new if valid_new else deduped_new
            if not picked:
                return []
            # Align to original candidates to avoid LLM-invented strings.
            return self._align_to_kept(picked, new_uses)
        except Exception as exc:
            logger.warning(f'LLM sampling dedup failed: {exc}')
            return None

    def _extract_json_obj(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}
        answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(answer_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(code_block_pattern, text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        if not text.startswith('{'):
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                text = text[start:end + 1]
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _load_deepseek_extractor(self) -> LLMJudge:
        if self._extract_judge is not None:
            return self._extract_judge

        if not _DEFAULT_JUDGE_CONFIG_PATH.exists():
            raise FileNotFoundError(f'Deepseek judge config not found: {_DEFAULT_JUDGE_CONFIG_PATH}')

        data = json.loads(_DEFAULT_JUDGE_CONFIG_PATH.read_text(encoding='utf-8'))
        models = data.get('models', [])
        if not isinstance(models, list):
            raise ValueError(f'Invalid models list in {_DEFAULT_JUDGE_CONFIG_PATH}')

        entry = None
        for candidate in models:
            name = str(candidate.get('name', '')).lower()
            model = str(candidate.get('model', '')).lower()
            if 'deepseek' in name or 'deepseek' in model:
                entry = candidate
                break
        if not entry:
            raise ValueError(f'Deepseek model not found in {_DEFAULT_JUDGE_CONFIG_PATH}')

        model_id = str(entry.get('model') or entry.get('name'))
        api_url = str(entry.get('api_url', ''))
        api_key = str(entry.get('api_key', 'EMPTY'))
        api_key_env = entry.get('api_key_env')
        if api_key_env:
            api_key = os.getenv(str(api_key_env), api_key)
        if not api_key:
            api_key = 'EMPTY'

        host = urlparse(api_url).hostname
        if host:
            for env_key in ('NO_PROXY', 'no_proxy'):
                current = os.environ.get(env_key, '')
                parts = [p.strip() for p in current.split(',') if p.strip()]
                for item in (host, 'localhost', '127.0.0.1'):
                    if item not in parts:
                        parts.append(item)
                os.environ[env_key] = ','.join(parts)

        generation_config = entry.get('generation_config') or {'temperature': 0.0, 'max_tokens': 2048}
        self._extract_judge = LLMJudge(
            model_id=model_id,
            api_url=api_url,
            api_key=api_key,
            generation_config=generation_config,
        )
        return self._extract_judge

    def _extract_uses_with_deepseek(self, item: str, raw_output: str) -> List[str]:
        cache_key = f'extract:{item}:{hashlib.md5(raw_output.encode("utf-8")).hexdigest()}'
        if cache_key in self._extract_cache:
            return list(self._extract_cache[cache_key])

        judge = self._load_deepseek_extractor()
        prompt = self._build_deepseek_extract_prompt(item, raw_output)
        response = judge.judge(prompt=prompt)
        data = self._extract_json_obj(response)
        uses = data.get('uses')
        if not isinstance(uses, list):
            uses = []
        cleaned = [str(u).strip() for u in uses if u and str(u).strip()]
        deduped, _ = self._dedupe_with_history(cleaned)
        self._extract_cache[cache_key] = list(deduped)
        return deduped

    def _extract_uses_with_deepseek_batch(self, items: List[str], raw_outputs: List[str]) -> List[List[str]]:
        if not items:
            return []

        results: List[Optional[List[str]]] = [None] * len(items)
        pending_indices: List[int] = []
        pending_prompts: List[str] = []
        judge = None

        for idx, (item, raw_output) in enumerate(zip(items, raw_outputs)):
            cache_key = f'extract:{item}:{hashlib.md5(raw_output.encode("utf-8")).hexdigest()}'
            cached = self._extract_cache.get(cache_key)
            if cached is not None:
                results[idx] = list(cached)
                continue
            if judge is None:
                judge = self._load_deepseek_extractor()
            pending_indices.append(idx)
            pending_prompts.append(self._build_deepseek_extract_prompt(item, raw_output))

        if pending_indices:
            assert judge is not None
            responses = judge.batch_judge(prompts=pending_prompts)
            if len(responses) != len(pending_indices):
                raise RuntimeError(
                    f'Deepseek batch extraction response count mismatch: {len(responses)} vs {len(pending_indices)}'
                )
            for result_idx, response in zip(pending_indices, responses):
                item = items[result_idx]
                raw_output = raw_outputs[result_idx]
                cache_key = f'extract:{item}:{hashlib.md5(raw_output.encode("utf-8")).hexdigest()}'
                data = self._extract_json_obj(response)
                uses = data.get('uses')
                if not isinstance(uses, list):
                    uses = []
                cleaned = [str(u).strip() for u in uses if u and str(u).strip()]
                deduped, _ = self._dedupe_with_history(cleaned)
                self._extract_cache[cache_key] = list(deduped)
                results[result_idx] = deduped

        return [result or [] for result in results]

    def _judge_applicability_batch(self, item: str, uses: List[str]) -> Dict[str, List[str]]:
        cache_key = f'{item}||{json.dumps(uses, ensure_ascii=False)}'
        if cache_key in self._applicability_cache:
            return self._applicability_cache[cache_key]

        judge = self._load_applicability_judge()
        prompt = self._build_applicability_batch_prompt(item, uses)
        response = judge.judge(prompt=prompt)
        parsed = self._extract_json_obj(response)
        deduped = parsed.get('deduped_uses')
        valid = parsed.get('valid_uses')
        invalid = parsed.get('invalid_uses')
        if not isinstance(deduped, list):
            deduped = []
        if not isinstance(valid, list):
            valid = []
        if not isinstance(invalid, list):
            invalid = []

        # Normalize and enforce subset relationships
        deduped_norm = {self._normalize_use(u): u for u in deduped if u and str(u).strip()}
        valid_norm = {self._normalize_use(u): u for u in valid if u and str(u).strip()}
        invalid_norm = {self._normalize_use(u): u for u in invalid if u and str(u).strip()}

        final_deduped = list(deduped_norm.values())
        final_valid = [u for k, u in valid_norm.items() if k in deduped_norm]
        final_invalid = [u for k, u in invalid_norm.items() if k in deduped_norm and k not in valid_norm]

        final_deduped, dropped = self._dedupe_with_history(final_deduped)
        final_valid = self._align_to_kept(final_valid, final_deduped)
        final_invalid = self._align_to_kept(final_invalid, final_deduped)
        valid_norm = {self._normalize_use(u) for u in final_valid}
        final_invalid = [u for u in final_invalid if self._normalize_use(u) not in valid_norm]

        result = {
            'deduped_uses': final_deduped,
            'valid_uses': final_valid,
            'invalid_uses': final_invalid,
            'postprocess_dropped': dropped,
            'judge_response': response,
        }
        self._applicability_cache[cache_key] = result
        return result

    def _judge_applicability_incremental(
        self, item: str, history_uses: List[str], new_uses: List[str]
    ) -> Dict[str, List[str]]:
        cache_key = (
            f'inc:{item}:{json.dumps(history_uses, ensure_ascii=True)}:'
            f'{json.dumps(new_uses, ensure_ascii=True)}'
        )
        if cache_key in self._applicability_cache:
            return self._applicability_cache[cache_key]

        judge = self._load_applicability_judge()
        prompt = self._build_applicability_incremental_prompt(item, history_uses, new_uses)
        response = judge.judge(prompt=prompt)
        result = self._finalize_incremental_judge_result(history_uses, response)
        self._applicability_cache[cache_key] = result
        return result

    def _finalize_incremental_judge_result(self, history_uses: List[str], response: str) -> Dict[str, List[str]]:
        parsed = self._extract_json_obj(response)
        deduped = parsed.get('deduped_new_uses')
        valid = parsed.get('valid_new_uses')
        invalid = parsed.get('invalid_new_uses')
        if not isinstance(deduped, list):
            deduped = []
        if not isinstance(valid, list):
            valid = []
        if not isinstance(invalid, list):
            invalid = []

        deduped_norm = {self._normalize_use(u): u for u in deduped if u and str(u).strip()}
        valid_norm = {self._normalize_use(u): u for u in valid if u and str(u).strip()}
        invalid_norm = {self._normalize_use(u): u for u in invalid if u and str(u).strip()}

        final_deduped = list(deduped_norm.values())
        final_valid = [u for k, u in valid_norm.items() if k in deduped_norm]
        final_invalid = [u for k, u in invalid_norm.items() if k in deduped_norm and k not in valid_norm]

        final_deduped, dropped = self._dedupe_with_history(final_deduped, history_uses)
        final_valid = self._align_to_kept(final_valid, final_deduped)
        final_invalid = self._align_to_kept(final_invalid, final_deduped)
        final_valid_norms = {self._normalize_use(u) for u in final_valid}
        final_invalid = [u for u in final_invalid if self._normalize_use(u) not in final_valid_norms]

        return {
            'deduped_new_uses': final_deduped,
            'valid_new_uses': final_valid,
            'invalid_new_uses': final_invalid,
            'postprocess_dropped': dropped,
            'judge_response': response,
        }

    def _judge_applicability_incremental_batch(
        self, items: List[str], histories: List[List[str]], new_uses_batches: List[List[str]]
    ) -> List[Dict[str, List[str]]]:
        if not items:
            return []

        results: List[Optional[Dict[str, List[str]]]] = [None] * len(items)
        pending_indices: List[int] = []
        pending_prompts: List[str] = []
        judge = None

        for idx, (item, history_uses, new_uses) in enumerate(zip(items, histories, new_uses_batches)):
            cache_key = (
                f'inc:{item}:{json.dumps(history_uses, ensure_ascii=True)}:'
                f'{json.dumps(new_uses, ensure_ascii=True)}'
            )
            cached = self._applicability_cache.get(cache_key)
            if cached is not None:
                results[idx] = cached
                continue
            if judge is None:
                judge = self._load_applicability_judge()
            pending_indices.append(idx)
            pending_prompts.append(self._build_applicability_incremental_prompt(item, history_uses, new_uses))

        if pending_indices:
            assert judge is not None
            responses = judge.batch_judge(prompts=pending_prompts)
            if len(responses) != len(pending_indices):
                raise RuntimeError(
                    'Applicability incremental batch response count mismatch: '
                    f'{len(responses)} vs {len(pending_indices)}'
                )
            for result_idx, response in zip(pending_indices, responses):
                item = items[result_idx]
                history_uses = histories[result_idx]
                new_uses = new_uses_batches[result_idx]
                cache_key = (
                    f'inc:{item}:{json.dumps(history_uses, ensure_ascii=True)}:'
                    f'{json.dumps(new_uses, ensure_ascii=True)}'
                )
                result = self._finalize_incremental_judge_result(history_uses, response)
                self._applicability_cache[cache_key] = result
                results[result_idx] = result

        return [result or {
            'deduped_new_uses': [],
            'valid_new_uses': [],
            'invalid_new_uses': [],
            'postprocess_dropped': [],
            'judge_response': None,
        } for result in results]

    def _llm_filter_new_uses_batch(
        self, items: List[str], history_batches: List[List[str]], candidate_batches: List[List[str]]
    ) -> List[Optional[List[str]]]:
        if not candidate_batches:
            return []

        outputs: List[Optional[List[str]]] = [None] * len(candidate_batches)
        active_indices = [idx for idx, candidates in enumerate(candidate_batches) if candidates]
        if not active_indices:
            return [[] for _ in candidate_batches]

        active_items = [items[idx] for idx in active_indices]
        active_histories = [history_batches[idx] for idx in active_indices]
        active_candidates = [candidate_batches[idx] for idx in active_indices]

        try:
            judged_batches = self._judge_applicability_incremental_batch(
                active_items, active_histories, active_candidates
            )
            for idx, candidates, judged in zip(active_indices, active_candidates, judged_batches):
                deduped_new = judged.get('deduped_new_uses', []) or []
                valid_new = judged.get('valid_new_uses', []) or []
                picked = valid_new if valid_new else deduped_new
                outputs[idx] = self._align_to_kept(picked, candidates) if picked else []
        except Exception as exc:
            logger.warning(f'LLM batch sampling dedup failed: {exc}')
            for idx in active_indices:
                outputs[idx] = None

        for idx, candidates in enumerate(candidate_batches):
            if not candidates and outputs[idx] is None:
                outputs[idx] = []
        return outputs

    def _compute_applicability(self, filtered_prediction: str, reference: str) -> Dict[str, Any]:
        raw_uses = self._parse_uses_from_prediction(filtered_prediction, item=reference)
        raw_count = len(raw_uses)
        if not raw_uses:
            return {
                'ratio': 0.0,
                'raw_uses': [],
                'deduped_uses': [],
                'valid_uses': [],
                'invalid_uses': [],
                'raw_count': 0,
                'deduped_count': 0,
                'valid_count': 0,
                'invalid_count': 0,
                'judge_errors': 0,
            }

        judge_errors = 0
        try:
            judged = self._judge_applicability_batch(reference, raw_uses)
            deduped_uses = judged.get('deduped_uses', [])
            valid_uses = judged.get('valid_uses', [])
            invalid_uses = judged.get('invalid_uses', [])
            judge_response = judged.get('judge_response')
        except Exception as exc:
            judge_errors = 1
            logger.warning(f'Applicability batch judge error for {reference}: {exc}')
            deduped_uses = self._semantic_deduplicate(raw_uses)
            valid_uses = list(deduped_uses)
            invalid_uses = []
            judge_response = None

        ratio = float(len(valid_uses)) / float(raw_count) if raw_count else 0.0
        return {
            'ratio': ratio,
            'raw_uses': raw_uses,
            'deduped_uses': deduped_uses,
            'valid_uses': valid_uses,
            'invalid_uses': invalid_uses,
            'raw_count': raw_count,
            'deduped_count': len(deduped_uses),
            'valid_count': len(valid_uses),
            'invalid_count': len(invalid_uses),
            'judge_errors': judge_errors,
            'judge_response': judge_response,
        }

    def _compute_applicability_from_rounds(
        self, rounds: List[Dict[str, Any]], reference: str
    ) -> Dict[str, Any]:
        raw_uses: List[str] = []
        deduped_uses: List[str] = []
        valid_uses: List[str] = []
        invalid_uses: List[str] = []
        round_details: List[Dict[str, Any]] = []
        judge_errors = 0
        judge_responses: List[Any] = []

        history_uses: List[str] = []

        for round_info in rounds:
            uses = round_info.get('uses', []) or []
            history_before = list(history_uses)
            raw_uses.extend(uses)
            if not uses:
                round_details.append({
                    'round': round_info.get('round'),
                    'raw_uses': [],
                    'deduped_new_uses': [],
                    'valid_new_uses': [],
                    'invalid_new_uses': [],
                    'history_uses_before': history_before,
                    'history_uses_after': list(history_uses),
                    'history_size': len(history_uses),
                })
                continue
            try:
                judged = self._judge_applicability_incremental(reference, history_uses, uses)
                deduped_new = judged.get('deduped_new_uses', [])
                valid_new = judged.get('valid_new_uses', [])
                invalid_new = judged.get('invalid_new_uses', [])
                postprocess_dropped = judged.get('postprocess_dropped', [])
                judge_responses.append(judged.get('judge_response'))
            except Exception as exc:
                judge_errors += 1
                logger.warning(f'Applicability incremental judge error for {reference}: {exc}')
                combined = list(history_uses) + list(uses)
                deduped_combined = self._semantic_deduplicate(combined)
                deduped_new = deduped_combined[len(history_uses):]
                valid_new = list(deduped_new)
                invalid_new = []
                postprocess_dropped = []
                judge_responses.append(None)

            deduped_uses.extend(deduped_new)
            valid_uses.extend(valid_new)
            invalid_uses.extend(invalid_new)
            history_uses.extend(valid_new)

            round_details.append({
                'round': round_info.get('round'),
                'raw_uses': uses,
                'deduped_new_uses': deduped_new,
                'valid_new_uses': valid_new,
                'invalid_new_uses': invalid_new,
                'postprocess_dropped': postprocess_dropped,
                'history_uses_before': history_before,
                'history_uses_after': list(history_uses),
                'history_size': len(history_uses),
            })

        raw_count = len(raw_uses)
        ratio = float(len(valid_uses)) / float(raw_count) if raw_count else 0.0
        return {
            'ratio': ratio,
            'raw_uses': raw_uses,
            'deduped_uses': deduped_uses,
            'valid_uses': valid_uses,
            'invalid_uses': invalid_uses,
            'raw_count': raw_count,
            'deduped_count': len(deduped_uses),
            'valid_count': len(valid_uses),
            'invalid_count': len(invalid_uses),
            'judge_errors': judge_errors,
            'judge_response': judge_responses,
            'rounds': round_details,
        }

    def _metric_name_and_func(self, metric_entry):
        if isinstance(metric_entry, str):
            metric_name = metric_entry
            metric_func = get_metric(metric_name)()
            return metric_name, metric_func
        metric_name = list(metric_entry.keys())[0]
        metric_cls = get_metric(metric_name)
        metric_func = metric_cls(**metric_entry[metric_name])
        return metric_name, metric_func

    def run_inference(self, model, sample: Sample, output_dir: str, **kwargs) -> TaskState:
        if not self.multi_round:
            return super().run_inference(model, sample, output_dir, **kwargs)

        self._on_inference_start(model, sample)
        base_prompt = self.format_prompt_template(sample)

        combined_uses: List[str] = []
        seen = set()
        rounds: List[Dict[str, Any]] = []
        total_usage = None
        stop_reason = 'max_rounds'

        for round_index in range(1, self.max_rounds + 1):
            prompt = base_prompt if round_index == 1 else self._build_round_prompt(
                base_prompt, combined_uses, round_index
            )
            model_output = model.generate(input=prompt, tools=sample.tools)
            if model_output.usage:
                total_usage = model_output.usage if total_usage is None else total_usage + model_output.usage

            raw_output = model_output.completion or ''
            item = (sample.metadata or {}).get('item', '')
            uses = self._parse_uses_from_prediction(raw_output, item=item)

            candidates: List[str] = []
            for use in uses:
                norm = self._normalize_use(use)
                if not norm or norm in seen:
                    continue
                if any(self._is_near_duplicate(use, prev) for prev in combined_uses):
                    continue
                candidates.append(use)

            if self.sample_llm_dedup:
                llm_result = self._llm_filter_new_uses(item, combined_uses, candidates)
                if llm_result is None:
                    # Fallback only when LLM judge fails.
                    new_uses = candidates
                else:
                    new_uses = llm_result
            else:
                new_uses = candidates

            added_via_semantic_metric = False
            if self.sample_semantic_dedup:
                new_uses = self._semantic_filter_new_uses(combined_uses, new_uses)

                # 追加一次与最终metric一致的语义去重，保证采样阶段与评测口径一致
                try:
                    metric = self._get_fluency_metric()
                    prior_combined = list(combined_uses)
                    combined_candidate = prior_combined + list(new_uses)
                    deduped_combined = metric._semantic_deduplicate(list(combined_candidate))
                    if deduped_combined is None:
                        deduped_combined = combined_candidate

                    existing_norms = set()
                    for prev_use in prior_combined:
                        norm_prev = self._normalize_use(prev_use)
                        if norm_prev:
                            existing_norms.add(norm_prev)

                    deduped_new: List[str] = []
                    for use in deduped_combined:
                        norm = self._normalize_use(use)
                        if not norm:
                            continue
                        if norm in existing_norms:
                            continue
                        deduped_new.append(use)
                        existing_norms.add(norm)

                    # 用语义去重后的结果重建历史，避免重复计数
                    seen = set()
                    combined_uses = []
                    for use in deduped_combined:
                        norm = self._normalize_use(use)
                        if not norm or norm in seen:
                            continue
                        seen.add(norm)
                        combined_uses.append(use)

                    new_uses = deduped_new
                    added_via_semantic_metric = True
                except Exception as exc:
                    logger.warning(f'Semantic sampling dedup (metric) failed: {exc}')

            if not added_via_semantic_metric:
                for use in new_uses:
                    norm = self._normalize_use(use)
                    if not norm or norm in seen:
                        continue
                    seen.add(norm)
                    combined_uses.append(use)

            rounds.append({
                'round': round_index,
                'uses': uses,
                'new_uses': new_uses,
                'cumulative_uses': list(combined_uses),
            })

            if self.stop_on_no_new and not new_uses:
                stop_reason = 'no_new_uses'
                break

        combined_output = ModelOutput.from_content(
            model=model.name,
            content=self._build_combined_answer(combined_uses),
            stop_reason='stop',
        )
        combined_output.usage = total_usage
        combined_output.metadata = {
            'aut_rounds': rounds,
            'aut_round_count': len(rounds),
            'aut_stop_reason': stop_reason,
        }

        task_state = self._on_inference_end(model, sample, combined_output, output_dir, **kwargs)
        if task_state.metadata is None:
            task_state.metadata = {}
        task_state.metadata['aut_rounds'] = rounds
        task_state.metadata['aut_round_count'] = len(rounds)
        task_state.metadata['aut_stop_reason'] = stop_reason
        return task_state

    def supports_batch_inference(self) -> bool:
        return True

    def run_batch_inference(self, model, samples: List[Sample], output_dir: str, **kwargs) -> List[TaskState]:
        if not samples:
            return []

        if not self.multi_round:
            return super().run_batch_inference(model=model, samples=samples, output_dir=output_dir, **kwargs)

        states: List[Dict[str, Any]] = []
        for sample in samples:
            self._on_inference_start(model, sample)
            states.append({
                'sample': sample,
                'base_prompt': self.format_prompt_template(sample),
                'item': (sample.metadata or {}).get('item', ''),
                'combined_uses': [],
                'seen': set(),
                'rounds': [],
                'total_usage': None,
                'stop_reason': 'max_rounds',
                'finished': False,
            })

        for round_index in range(1, self.max_rounds + 1):
            active_indices = [idx for idx, state in enumerate(states) if not state['finished']]
            if not active_indices:
                break

            batch_inputs = []
            batch_tools = []
            batch_items = []
            for idx in active_indices:
                state = states[idx]
                prompt = state['base_prompt'] if round_index == 1 else self._build_round_prompt(
                    state['base_prompt'], state['combined_uses'], round_index
                )
                batch_inputs.append(prompt)
                batch_tools.append(state['sample'].tools)
                batch_items.append(state['item'])

            outputs = list(
                model.batch_generate(
                    inputs=batch_inputs,
                    tools=batch_tools,
                    tool_choices=[None for _ in batch_inputs],
                    configs=[None for _ in batch_inputs],
                )
            )
            if len(outputs) != len(batch_inputs):
                raise RuntimeError(
                    f'AUT batch generation response count mismatch: {len(outputs)} vs {len(batch_inputs)}'
                )

            raw_outputs = [output.completion or '' for output in outputs]
            parsed_uses_batches = self._parse_uses_from_predictions_batch(raw_outputs, batch_items)
            candidate_batches: List[List[str]] = []

            for state_idx, uses in zip(active_indices, parsed_uses_batches):
                state = states[state_idx]
                candidates: List[str] = []
                for use in uses:
                    norm = self._normalize_use(use)
                    if not norm or norm in state['seen']:
                        continue
                    if any(self._is_near_duplicate(use, prev) for prev in state['combined_uses']):
                        continue
                    candidates.append(use)
                candidate_batches.append(candidates)

            if self.sample_llm_dedup:
                llm_filtered_batches = self._llm_filter_new_uses_batch(
                    items=batch_items,
                    history_batches=[states[idx]['combined_uses'] for idx in active_indices],
                    candidate_batches=candidate_batches,
                )
            else:
                llm_filtered_batches = [list(candidates) for candidates in candidate_batches]

            for state_idx, model_output, uses, candidates, llm_result in zip(
                active_indices, outputs, parsed_uses_batches, candidate_batches, llm_filtered_batches
            ):
                state = states[state_idx]
                if model_output.usage:
                    current_usage = state['total_usage']
                    state['total_usage'] = model_output.usage if current_usage is None else current_usage + model_output.usage

                new_uses = llm_result if llm_result is not None else candidates
                added_via_semantic_metric = False

                if self.sample_semantic_dedup:
                    new_uses = self._semantic_filter_new_uses(state['combined_uses'], new_uses)
                    try:
                        metric = self._get_fluency_metric()
                        prior_combined = list(state['combined_uses'])
                        combined_candidate = prior_combined + list(new_uses)
                        deduped_combined = metric._semantic_deduplicate(list(combined_candidate))
                        if deduped_combined is None:
                            deduped_combined = combined_candidate

                        existing_norms = set()
                        for prev_use in prior_combined:
                            norm_prev = self._normalize_use(prev_use)
                            if norm_prev:
                                existing_norms.add(norm_prev)

                        deduped_new: List[str] = []
                        for use in deduped_combined:
                            norm = self._normalize_use(use)
                            if not norm or norm in existing_norms:
                                continue
                            deduped_new.append(use)
                            existing_norms.add(norm)

                        state['seen'] = set()
                        state['combined_uses'] = []
                        for use in deduped_combined:
                            norm = self._normalize_use(use)
                            if not norm or norm in state['seen']:
                                continue
                            state['seen'].add(norm)
                            state['combined_uses'].append(use)

                        new_uses = deduped_new
                        added_via_semantic_metric = True
                    except Exception as exc:
                        logger.warning(f'Semantic sampling dedup (metric) failed: {exc}')

                if not added_via_semantic_metric:
                    for use in new_uses:
                        norm = self._normalize_use(use)
                        if not norm or norm in state['seen']:
                            continue
                        state['seen'].add(norm)
                        state['combined_uses'].append(use)

                state['rounds'].append({
                    'round': round_index,
                    'uses': uses,
                    'new_uses': new_uses,
                    'cumulative_uses': list(state['combined_uses']),
                })

                if self.stop_on_no_new and not new_uses:
                    state['stop_reason'] = 'no_new_uses'
                    state['finished'] = True

        task_states: List[TaskState] = []
        for state in states:
            combined_output = ModelOutput.from_content(
                model=model.name,
                content=self._build_combined_answer(state['combined_uses']),
                stop_reason='stop',
            )
            combined_output.usage = state['total_usage']
            combined_output.metadata = {
                'aut_rounds': state['rounds'],
                'aut_round_count': len(state['rounds']),
                'aut_stop_reason': state['stop_reason'],
            }

            task_state = self._on_inference_end(model, state['sample'], combined_output, output_dir, **kwargs)
            if task_state.metadata is None:
                task_state.metadata = {}
            task_state.metadata['aut_rounds'] = state['rounds']
            task_state.metadata['aut_round_count'] = len(state['rounds'])
            task_state.metadata['aut_stop_reason'] = state['stop_reason']
            task_states.append(task_state)

        return task_states

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ):
        score = super().match_score(original_prediction, filtered_prediction, reference, task_state)
        rounds = (task_state.metadata or {}).get('aut_rounds', [])
        if rounds:
            round_scores = []
            for round_info in rounds:
                round_index = round_info.get('round')
                uses = round_info.get('cumulative_uses', [])
                # 附加语义去重版（不替换原始采样结果），用于对齐最终 metric 的去重标准
                if self.sample_semantic_dedup:
                    metric = self._get_fluency_metric()
                    try:
                        uses = metric._semantic_deduplicate(list(uses))
                    except Exception as exc:
                        logger.warning(f'Round {round_index} semantic dedup for scoring failed: {exc}')
                round_prediction = self._build_uses_json(uses)
                per_metric = {}
                for metric_entry in self.metric_list:
                    metric_name, metric_func = self._metric_name_and_func(metric_entry)
                    try:
                        if metric_name == 'aut_fluency' and self.sample_llm_dedup:
                            metric_score = float(len(uses))
                        else:
                            metric_score = metric_func(prediction=round_prediction, reference=reference)
                    except Exception as exc:
                        logger.error(f'Error calculating {metric_name} for round {round_index}: {exc}')
                        metric_score = 0.0
                    per_metric[metric_name] = metric_score
                    score.value[f'{metric_name}_r{round_index}'] = metric_score
                round_scores.append({
                    'round': round_index,
                    'uses_count': len(uses),
                    'scores': per_metric,
                })

            score.metadata['round_scores'] = round_scores
            if self.sample_llm_dedup:
                score.value['aut_fluency'] = float(len(rounds[-1].get('cumulative_uses', [])))

        applicability = (
            self._compute_applicability_from_rounds(rounds, reference)
            if rounds else self._compute_applicability(filtered_prediction, reference)
        )
        score.value['aut_applicability'] = applicability['ratio']
        score.metadata['aut_applicability'] = applicability
        return score
