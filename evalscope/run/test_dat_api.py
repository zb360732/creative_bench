#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from evalscope.run import run_task
from evalscope.config import TaskConfig

print("=" * 80)
print("开始运行 DAT Benchmark 评估")
print("=" * 80)

from datetime import datetime

# 生成带任务名的输出目录
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
work_dir = f'./outputs/dat_{timestamp}'

task_cfg = TaskConfig(
    model='Qwen2.5-7B-Instruct',
    api_url='http://localhost:8007/v1',
    api_key='EMPTY',
    datasets=['dat'],
    limit=1,  # DAT只有1个样本
    work_dir=work_dir,  # 指定输出目录包含任务名
    generation_config={
        'temperature': 0.7,
        'max_tokens': 2048,
    }
)

print(f"模型: {task_cfg.model}")
print(f"API地址: {task_cfg.api_url}")
print(f"测试样本数: {task_cfg.limit}")
print("=" * 80)

results = run_task(task_cfg=task_cfg)

print("\n" + "=" * 80)
print("评估完成！结果：")
print("=" * 80)
print(results)
