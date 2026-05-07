#!/root/data/conda/envs/evalscope/bin/python
# -*- coding: utf-8 -*-
"""
Test BATS benchmark integration with evalscope framework

This script tests the BATS (Bigger Analogy Test Set) benchmark using the evalscope
framework with a local API endpoint.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from evalscope.run import run_task
from evalscope.config import TaskConfig

print("=" * 80)
print("开始运行 BATS Benchmark 评估")
print("=" * 80)

# 生成带任务名的输出目录
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
work_dir = f'./outputs/bats_{timestamp}'

task_cfg = TaskConfig(
    model='Qwen2.5-7B-Instruct',
    api_url='http://localhost:8007/v1',
    api_key='EMPTY',
    datasets=['bats'],
    limit=1000,  # 测试前1000个样本
    work_dir=work_dir,  # 指定输出目录包含任务名
    generation_config={
        'temperature': 0.0,  # 确定性输出，适合准确率任务
        'max_tokens': 128,
    }
)

print(f"模型: {task_cfg.model}")
print(f"API地址: {task_cfg.api_url}")
print(f"数据集: {task_cfg.datasets}")
print(f"测试样本数: {task_cfg.limit}")
print("=" * 80)

results = run_task(task_cfg=task_cfg)

print("\n" + "=" * 80)
print("评估完成！结果：")
print("=" * 80)
print(results)
