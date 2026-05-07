#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临时脚本：更新Metaphor数据集中的prompt，添加<answer>标签要求
"""

import json
from pathlib import Path

# 数据文件路径
data_file = Path('/root/data/code/evalscope/evalscope/benchmarks/metaphor/data/metaphor.json')

# 读取数据
print(f"Reading {data_file}...")
with open(data_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples")

# 更新每个样本的query，添加<answer>标签要求
updated_count = 0
for item in data:
    original_query = item['query']

    # 检查是否已经包含<answer>标签说明
    if '<answer>' in original_query:
        continue

    # 原始格式中有这样的部分:
    # "Please provide your answer in the following JSON format:\n{\n  \"word\": \"replacement_word\"\n}\n\n"
    # 需要替换为:
    # "Please provide your answer in the following JSON format inside <answer> tags:\n\n<answer>\n{\n  \"word\": \"replacement_word\"\n}\n</answer>\n\n"

    # 查找并替换
    old_pattern = 'Please provide your answer in the following JSON format:\n{\n  "word": "replacement_word"\n}\n\nReplace'
    new_pattern = 'Please provide your answer in the following JSON format inside <answer> tags:\n\n<answer>\n{\n  "word": "replacement_word"\n}\n</answer>\n\nReplace'

    if old_pattern in original_query:
        item['query'] = original_query.replace(old_pattern, new_pattern)

        # 在末尾添加提醒
        if not item['query'].endswith('Remember to put your JSON response inside <answer></answer> tags.'):
            # 找到最后的句子末尾
            if item['query'].endswith('.'):
                item['query'] = item['query'] + ' Remember to put your JSON response inside <answer></answer> tags.'

        updated_count += 1

print(f"Updated {updated_count} samples")

# 写回文件
print(f"Writing back to {data_file}...")
with open(data_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Done!")
print(f"\nSample after update:")
print(data[0]['query'][:500])
