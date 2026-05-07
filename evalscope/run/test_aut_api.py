#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用本地API测试 AUT Benchmark
使用 run_task 接口进行端到端测试
"""

import os
import sys
from pathlib import Path

# 添加项目路径，使用本地开发版本
sys.path.insert(0, str(Path(__file__).parent.parent))

from evalscope.run import run_task
from evalscope.config import TaskConfig
from datetime import datetime

# 设置CUDA设备（如果需要）
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# 生成带任务名的输出目录
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
work_dir = f'./outputs/aut_{timestamp}'

# 配置任务
task_cfg = TaskConfig(
    model='Qwen2.5-7B-Instruct',  # 模型名称
    api_url='http://localhost:8007/v1',  # 本地API地址
    api_key='EMPTY',  # API key (本地API通常不需要)
    datasets=['aut'],  # AUT benchmark
    limit=5,  # 只测试5个样本
    work_dir=work_dir,  # 指定输出目录包含任务名
    generation_config={
        'temperature': 0.7,
        'max_tokens': 2048,
    },
)

# 运行评估任务
print("=" * 80)
print("开始运行 AUT Benchmark 评估")
print("=" * 80)
print(f"模型: Qwen2.5-7B-Instruct")
print(f"API地址: http://localhost:8007/v1")
print(f"测试样本数: {task_cfg.limit}")
print("=" * 80)

results = run_task(task_cfg=task_cfg)

print("\n" + "=" * 80)
print("评估完成！")
print("=" * 80)
print("\n结果:")
print(results)
