#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试使用本地 API 的 evalscope 评估
使用本地 8007 端口的 API，模型名是 Qwen2.5-7B-Instruct
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from evalscope.config import TaskConfig
from evalscope.constants import EvalType, JudgeStrategy
from evalscope.run import run_task
from evalscope.utils.logger import get_logger

logger = get_logger()


def test_local_api(limit=10):
    """测试使用本地 API 进行评估"""
    
    print("=" * 80)
    print("测试本地 API 评估")
    print("=" * 80)
    print(f"API URL: http://localhost:8007/v1")
    print(f"Model: Qwen2.5-7B-Instruct")
    print(f"Limit: {limit}")
    print("=" * 80)
    
    # 配置任务
    config = {
        # 模型配置
        'model': 'Qwen2.5-7B-Instruct',
        'api_url': 'http://localhost:8007/v1',  # 本地 API 地址
        'api_key': 'EMPTY',  # 本地 API 通常不需要 key，但需要提供
        
        # 评估类型
        'eval_type': EvalType.SERVICE,
        
        # 数据集配置 - 使用刚创建的隐喻数据集
        'datasets': ['general_qa'],
        'dataset_args': {
            'general_qa': {
                'dataset_id': str(Path(__file__).parent.parent / 'custom_eval/text/qa/combination'),  # 使用绝对路径
                'subset_list': ['metaphor'],
            }
        },
        
        # 评估配置
        'eval_batch_size': 5,  # 批次大小
        'limit': limit,  # 限制测试样本数量（可以调整）
        
        # 生成配置
        'generation_config': {
            'max_tokens': 512,
            'temperature': 0.0,
            'seed': 42,
        },
        
        # 判断策略
        'judge_strategy': JudgeStrategy.AUTO,
        
        # 调试模式
        'debug': True,
        
        # 忽略错误（可选）
        'ignore_errors': False,
    }
    
    # 创建任务配置
    task_cfg = TaskConfig(**config)
    
    # 运行评估
    print("\n开始运行评估...")
    try:
        run_task(task_cfg=task_cfg)
        print("\n评估完成！")
    except Exception as e:
        print(f"\n评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_local_api_simple(limit=3):
    """简单的测试 - 只测试几个样本"""
    
    print("=" * 80)
    print("简单测试 - 本地 API")
    print("=" * 80)
    print(f"API URL: http://localhost:8007/v1")
    print(f"Model: Qwen2.5-7B-Instruct")
    print(f"Limit: {limit}")
    print("=" * 80)
    
    config = {
        'model': 'Qwen2.5-7B-Instruct',
        'api_url': 'http://localhost:8007/v1',
        'api_key': 'EMPTY',
        'eval_type': EvalType.SERVICE,
        'datasets': ['general_qa'],
        'dataset_args': {
            'general_qa': {
                'dataset_id': str(Path(__file__).parent.parent / 'custom_eval/text/qa/combination'),  # 使用绝对路径
                'subset_list': ['metaphor'],
            }
        },
        'eval_batch_size': 2,
        'limit': limit,  # 只测试几个样本
        'generation_config': {
            'max_tokens': 256,
            'temperature': 0.0,
        },
        'judge_strategy': JudgeStrategy.AUTO,
        'debug': True,
    }
    
    task_cfg = TaskConfig(**config)
    
    print("\n开始运行简单测试...")
    try:
        run_task(task_cfg=task_cfg)
        print("\n测试完成！")
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试本地 API 评估")
    parser.add_argument(
        "--mode",
        choices=["full", "simple"],
        default="simple",
        help="测试模式：full（完整测试）或 simple（简单测试）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制测试样本数量"
    )
    
    args = parser.parse_args()
    
    # 确定 limit 值
    limit = args.limit if args.limit else (3 if args.mode == "simple" else 10)
    
    if args.mode == "simple":
        test_local_api_simple(limit=limit)
    else:
        test_local_api(limit=limit)

