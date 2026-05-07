# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import re
from pathlib import Path
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

# 获取数据文件的绝对路径
_DATA_DIR = Path(__file__).parent / 'data'
_DEFAULT_DATASET_PATH = str(_DATA_DIR / 'bats_sampled.json')


@register_benchmark(
    BenchmarkMeta(
        name='bats',
        pretty_name='BATS (Word Analogy Test)',
        dataset_id=_DEFAULT_DATASET_PATH,  # 默认使用sampled数据集
        metric_list=['bats_accuracy'],
        aggregation='mean',
        subset_list=['sampled', 'full'],  # 两个子集，sampled为默认
        default_subset='sampled',
        tags=[Tags.REASONING, Tags.INSTRUCTION_FOLLOWING],
        description='BATS (Bigger Analogy Test Set) benchmark for word analogy reasoning. '
                    'Tests convergent thinking through semantic relations. '
                    'Supports two subsets: "sampled" (4,000 samples, default) and "full" (98,000 samples).',
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template='{query}',
    )
)
class BATSAdapter(DefaultDataAdapter):
    """
    Adapter for BATS (Bigger Analogy Test Set) benchmark.

    BATS evaluates word analogy reasoning through semantic relations.
    The task format is: A : B :: C : ?

    Two subsets available:
    - sampled: 4,000 analogies (200 per category, mixed forward/backward)
    - full: 98,000 analogies (all possible combinations)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_from_disk(self, dataset_name_or_path=None, subset_list=None, **kwargs):
        """
        Load BATS dataset from local JSON files.

        Args:
            dataset_name_or_path: Path to the dataset (not used, determined by subset)
            subset_list: List of subsets to load ('sampled' and/or 'full')
            **kwargs: Additional arguments

        Returns:
            Tuple[Dict, None]: datasets dict and fewshot_dataset (None for BATS)
        """
        # 确定要加载的subset
        if subset_list is None:
            subset_list = [self._benchmark_meta.default_subset]

        logger.info(f'Loading BATS subsets: {subset_list}')

        # 根据subset选择对应的JSON文件
        datasets = {}
        for subset in subset_list:
            if subset == 'full':
                data_path = _DATA_DIR / 'bats_full.json'
            elif subset == 'sampled':
                data_path = _DATA_DIR / 'bats_sampled.json'
            else:
                raise ValueError(f"Unknown BATS subset: {subset}. Available: ['sampled', 'full']")

            if not data_path.exists():
                raise FileNotFoundError(f'BATS dataset not found at: {data_path}')

            logger.info(f'Loading BATS {subset} subset from: {data_path}')

            # 加载JSON数据
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.info(f'Loaded {len(data)} BATS {subset} items')

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
            datasets[subset] = samples

        # 返回格式：(test_dataset, fewshot_dataset)
        # BATS 不使用 few-shot，所以返回 None
        return datasets, None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a dataset record to a Sample object.

        Args:
            record: Dict containing:
                - query: Full prompt text (already formatted)
                - word_a, word_b, word_c: The three words in analogy
                - target_words: List of correct answer words
                - direction: 'forward' or 'backward'
                - category: Category code (E01-E10, L01-L10)
                - category_name: Category description
                - relation_type: 'encyclopedic_semantics' or 'lexicographic_semantics'

        Returns:
            Sample object with:
                - input: The query/prompt text
                - target: JSON string of target_words list (for metric)
                - metadata: Additional information
        """
        return Sample(
            input=record['query'],
            target=json.dumps(record['target_words']),  # 序列化为JSON字符串供metric使用
            metadata={
                'word_a': record['word_a'],
                'word_b': record['word_b'],
                'word_c': record['word_c'],
                'direction': record['direction'],
                'category': record['category'],
                'category_name': record['category_name'],
                'relation_type': record['relation_type'],
            }
        )

    def format_prompt_template(self, sample: Sample) -> str:
        """
        Format the prompt for BATS task with <answer> tags requirement.

        Args:
            sample: Sample object with input containing the full prompt

        Returns:
            str: The complete prompt text (already includes <answer> tag instructions)
        """
        # query字段已经包含完整的prompt和<answer>标签说明
        return sample.input

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
        json_pattern = r'\{[^{}]*"target"[^{}]*:[^{}]*\}'
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
