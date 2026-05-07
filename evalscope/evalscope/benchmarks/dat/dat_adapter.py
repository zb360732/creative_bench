# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import os
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
_DEFAULT_DATASET_PATH = str(_DATA_DIR / 'dat.json')


@register_benchmark(
    BenchmarkMeta(
        name='dat',
        pretty_name='DAT (Divergent Association Task)',
        tags=[Tags.REASONING, Tags.INSTRUCTION_FOLLOWING],
        description='Divergent Association Task (DAT) for evaluating creative thinking by generating '
                    'semantically distant words. Participants provide 10 words that are as unrelated as possible.',
        dataset_id=_DEFAULT_DATASET_PATH,
        metric_list=['dat_semantic_distance'],
        aggregation='mean',
        subset_list=['default'],
        default_subset='default',
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template='{query}',
    )
)
class DATAdapter(DefaultDataAdapter):
    """
    Adapter for Divergent Association Task (DAT) benchmark.

    DAT evaluates creative thinking by asking participants to generate
    10 semantically distant words. The evaluation includes:
    - Semantic Distance: Average pairwise semantic distance between words
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_from_disk(self, dataset_name_or_path=None, subset_list=None, **kwargs):
        """
        Load DAT dataset from local JSON file.

        Args:
            dataset_name_or_path: Path to the dataset (defaults to self.dataset_id)
            subset_list: List of subsets to load (not used for DAT)
            **kwargs: Additional arguments

        Returns:
            Tuple[Dict, None]: test_dataset dict and fewshot_dataset (None for DAT)
        """
        import json
        from pathlib import Path

        # 确定数据文件路径
        if dataset_name_or_path is None:
            dataset_name_or_path = self.dataset_id

        data_path = Path(dataset_name_or_path)

        if not data_path.exists():
            raise FileNotFoundError(f'DAT dataset not found at: {data_path}')

        logger.info(f'Loading DAT dataset from: {data_path}')

        # 加载JSON数据
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f'Loaded {len(data)} DAT test items')

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
        # DAT 不使用 few-shot，所以返回 None
        return {'default': dataset}, None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a dataset record to a Sample object.

        Args:
            record: Dict containing:
                - query: Full prompt text (already formatted)
                - category: Category name (e.g., 'DAT')

        Returns:
            Sample object with:
                - input: The query/prompt text
                - target: Empty string (DAT doesn't use references)
                - metadata: Additional information
        """
        return Sample(
            input=record['query'],
            target='',  # DAT不需要参考答案，使用空字符串占位
            metadata={
                'category': record.get('category', 'DAT'),
            }
        )

    def format_prompt_template(self, sample: Sample) -> str:
        """
        Format the prompt for DAT task with <answer> tags requirement.

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
  "words": [
    "word1",
    "word2",
    "word3",
    "word4",
    "word5",
    "word6",
    "word7",
    "word8",
    "word9",
    "word10"
  ]
}
</answer>

Remember to put your JSON response inside <answer></answer> tags."""

        return base_prompt + answer_format_instruction

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
        json_pattern = r'\{[^{}]*"words"[^{}]*\[[^\]]*\][^{}]*\}'
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
