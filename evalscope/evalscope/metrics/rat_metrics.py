# Copyright (c) Alibaba, Inc. and its affiliates.
"""
RAT (Remote Associates Test) 评估指标实现
评分逻辑：模型输出的词必须与参考答案完全匹配（不区分大小写）
"""

import json
import re
from typing import List

from evalscope.api.metric import Metric
from evalscope.api.registry import register_metric
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_metric(name='rat_accuracy')
class RATAccuracy(Metric):
    """RAT连接词准确率指标：判断输出词是否与参考答案完全匹配"""

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
            predictions: 模型预测的JSON字符串列表，格式为 {"word": "connecting_word"}
            references: 参考答案（正确的连接词）列表

        Returns:
            准确率列表（1.0表示正确，0.0表示错误）
        """
        results = []

        for prediction, reference in zip(predictions, references):
            # 解析预测的词
            predicted_word = self._parse_json_response(prediction)

            if not predicted_word:
                # 无法解析预测结果
                results.append(0.0)
                continue

            # 标准化参考答案（小写，去除空格）
            reference_word = reference.strip().lower()

            # 检查预测词是否与参考答案完全匹配（不区分大小写）
            if predicted_word == reference_word:
                results.append(1.0)
            else:
                results.append(0.0)

        return results
