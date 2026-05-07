# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Metaphor (隐喻词替换) 评估指标实现
评分逻辑：模型输出的词只要在候选答案列表中就算正确
"""

import json
import re
from typing import List

from evalscope.api.metric import Metric
from evalscope.api.registry import register_metric
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_metric(name='metaphor_accuracy')
class MetaphorAccuracy(Metric):
    """Metaphor词替换准确率指标：判断输出词是否在候选答案列表中"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _parse_json_response(self, prediction: str) -> str:
        """解析模型输出的JSON格式，提取word字段"""
        # 尝试直接解析JSON
        try:
            data = json.loads(prediction)
            if isinstance(data, dict) and 'word' in data:
                word = data['word']
                if word:
                    return str(word).strip().lower()
        except json.JSONDecodeError:
            pass

        # 尝试从markdown代码块中提取
        code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(code_block_pattern, prediction, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and 'word' in data:
                    word = data['word']
                    if word:
                        return str(word).strip().lower()
            except json.JSONDecodeError:
                pass

        # 如果都失败，尝试直接从文本中提取单词
        # 查找引号中的单词
        word_pattern = r'"word"\s*:\s*"([^"]+)"'
        match = re.search(word_pattern, prediction)
        if match:
            return match.group(1).strip().lower()

        logger.warning(f'Failed to parse word from prediction: {prediction[:100]}')
        return ''

    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        """计算准确率分数

        Args:
            predictions: 模型预测的JSON字符串列表，格式为 {"word": "replacement_word"}
            references: JSON序列化的候选答案列表的字符串列表

        Returns:
            准确率列表（1.0表示正确，0.0表示错误）
        """
        results = []

        for prediction, reference_json in zip(predictions, references):
            # 解析预测的词
            predicted_word = self._parse_json_response(prediction)

            if not predicted_word:
                # 无法解析预测结果
                results.append(0.0)
                continue

            # 从JSON字符串反序列化候选答案列表
            try:
                candidate_answers = json.loads(reference_json)
                if not isinstance(candidate_answers, list):
                    logger.warning(f'Expected list in reference, got {type(candidate_answers)}')
                    results.append(0.0)
                    continue
            except json.JSONDecodeError:
                logger.warning(f'Failed to parse reference JSON: {reference_json[:100]}')
                results.append(0.0)
                continue

            # 将候选答案也转为小写进行比较
            candidate_answers_lower = [ans.strip().lower() for ans in candidate_answers]

            # 检查预测词是否在候选答案中
            if predicted_word in candidate_answers_lower:
                results.append(1.0)
            else:
                results.append(0.0)

        return results
