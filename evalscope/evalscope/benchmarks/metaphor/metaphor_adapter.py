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
_DEFAULT_DATASET_PATH = str(_DATA_DIR / 'metaphor.json')


@register_benchmark(
    BenchmarkMeta(
        name='metaphor',
        pretty_name='Metaphor Paraphrase Task',
        tags=[Tags.REASONING, Tags.INSTRUCTION_FOLLOWING],
        description='Metaphor paraphrase task for evaluating the ability to substitute words '
                    'while maintaining sentence meaning. The task requires replacing a highlighted '
                    'word with a single appropriate alternative word.',
        dataset_id=_DEFAULT_DATASET_PATH,
        metric_list=['metaphor_accuracy'],
        aggregation='mean',
        subset_list=['default'],
        default_subset='default',
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template='{query}',
    )
)
class MetaphorAdapter(DefaultDataAdapter):
    """
    Adapter for Metaphor paraphrase task.

    Metaphor evaluates the ability to substitute words while maintaining
    sentence meaning. The evaluation includes:
    - Accuracy: Whether the predicted word is in the list of acceptable candidates
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_from_disk(self, dataset_name_or_path=None, subset_list=None, **kwargs):
        """
        Load Metaphor dataset from local JSON file.

        Args:
            dataset_name_or_path: Path to the dataset (defaults to self.dataset_id)
            subset_list: List of subsets to load (not used for Metaphor)
            **kwargs: Additional arguments

        Returns:
            Tuple[Dict, None]: test_dataset dict and fewshot_dataset (None for Metaphor)
        """
        import json
        from pathlib import Path

        # 确定数据文件路径
        if dataset_name_or_path is None:
            dataset_name_or_path = self.dataset_id

        data_path = Path(dataset_name_or_path)

        if not data_path.exists():
            raise FileNotFoundError(f'Metaphor dataset not found at: {data_path}')

        logger.info(f'Loading Metaphor dataset from: {data_path}')

        # 加载JSON数据
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f'Loaded {len(data)} Metaphor test items')

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
        # Metaphor 不使用 few-shot，所以返回 None
        return {'default': dataset}, None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a dataset record to a Sample object.

        Args:
            record: Dict containing:
                - query: Full prompt text (already formatted)
                - metaphor_word: The word to be replaced
                - candidate_answers: List of acceptable replacement words

        Returns:
            Sample object with:
                - input: The query/prompt text
                - target: JSON string of candidate answers list
                - metadata: Additional information for metrics
        """
        import json
        # 将候选答案列表序列化为JSON字符串，以便框架正确传递
        return Sample(
            input=record['query'],
            target=json.dumps(record['candidate_answers']),  # 序列化为JSON字符串
            metadata={
                'metaphor_word': record['metaphor_word'],
                'category': record.get('category', 'METAPHOR'),
                'novelty': record.get('novelty'),
            }
        )

    def format_prompt_template(self, sample: Sample) -> str:
        """
        Format the prompt for Metaphor task with <answer> tags requirement.

        Args:
            sample: Sample object with input containing the full prompt

        Returns:
            str: The complete prompt text with answer tag instructions
        """
        # 在原有prompt基础上添加 <answer> 标签要求
        base_prompt = sample.input

        answer_format_instruction = """

You must return exactly one single-word replacement that preserves the meaning of the highlighted word in context.
Do not include any analysis, discussion, or extra text.
Your entire response must contain exactly one <answer>...</answer> block, using this format:

<answer>
{
  "word": "replacement_word"
}
</answer>"""

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
        json_pattern = r'\{[^{}]*"word"[^{}]*:[^{}]*\}'
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
