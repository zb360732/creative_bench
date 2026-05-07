#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 NeoCoder benchmark
使用本地部署在 8007 端口的 Qwen2.5-7B-Instruct 模型
"""

import json
import os
import sys
from pathlib import Path

# 添加 evalscope 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evalscope import TaskConfig, run_task
from evalscope.utils.logger import get_logger

logger = get_logger()


def _load_models(models_json_path: Path) -> list[dict]:
    data = json.loads(models_json_path.read_text(encoding='utf-8'))
    models = data.get('models', [])
    if not isinstance(models, list) or not models:
        raise ValueError(f'`models` must be a non-empty list in {models_json_path}')
    return models


def _resolve_env(value):
    if not isinstance(value, str):
        return value
    return os.path.expandvars(value)


def _resolve_api_key(entry: dict) -> str:
    api_key = str(_resolve_env(entry.get('api_key', 'EMPTY')))
    api_key_env = entry.get('api_key_env')
    if api_key_env:
        api_key = os.getenv(str(api_key_env), api_key)
    if api_key in {'', 'YOUR_API_KEY'}:
        api_key = os.getenv('EVALSCOPE_API_KEY', api_key)
    if api_key in {'', 'YOUR_API_KEY'}:
        api_key = os.getenv('OPENAI_API_KEY', api_key)
    return api_key


def _select_model(models: list[dict], name: str) -> dict:
    if name:
        for model in models:
            if model.get('name') == name:
                return model
        available = ', '.join([m.get('name', '<missing-name>') for m in models])
        raise ValueError(f'Model not found: {name}. Available: {available}')
    return models[0]


def _load_model_entry(model_name: str, models_json_path: Path) -> dict:
    models = _load_models(models_json_path)
    entry = _select_model(models, model_name)
    return {
        'name': str(entry.get('name') or entry.get('model') or ''),
        'model': str(_resolve_env(entry.get('model', ''))),
        'api_url': str(_resolve_env(entry.get('api_url', ''))),
        'api_key': _resolve_api_key(entry),
    }


def _outputs_root() -> Path:
    return Path(__file__).resolve().parents[2] / 'outputs' / 'exploration'


def test_neocoder(model_cfg: dict, work_dir: str):
    """测试 NeoCoder benchmark"""

    # 配置任务
    task_cfg = TaskConfig(
        model=model_cfg['model'],
        model_id=model_cfg['name'],
        api_url=model_cfg['api_url'],
        api_key=model_cfg['api_key'],
        eval_type='openai_api',
        datasets=['neocoder'],
        dataset_args={
            'neocoder': {
                # 可以指定自定义数据集路径
                # 'extra_params': {
                #     'dataset_path': '/path/to/NeoCoder.json'
                # }
            }
        },
        generation_config={
            'max_tokens': 2048,  # 代码生成可能需要较多 tokens
            'temperature': 0.7,
            'top_p': 0.95,
        },
        timeout=120,  # 超时时间（秒）
        # limit 参数控制每个问题的样本数，我们已经在 adapter 中限制为前5个问题
        # 每个问题有多个约束级别，所以总样本数会更多
        work_dir=work_dir,  # 输出目录
        no_timestamp=True,
    )
    
    logger.info("开始测试 NeoCoder benchmark...")
    logger.info(f"模型: {model_cfg['name']}")
    logger.info(f"API URL: {model_cfg['api_url']}")
    logger.info(f"测试问题数: 前5个问题（每个问题有多个约束级别）")
    logger.info(f"输出目录: {task_cfg.work_dir}")
    
    try:
        # 运行任务
        run_task(task_cfg=task_cfg)
        logger.info("测试完成！")
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test NeoCoder benchmark integration')
    parser.add_argument(
        '--model-name',
        default='',
        help='Model name from models.json (default: first entry)'
    )
    parser.add_argument(
        '--models-json',
        default=str(Path(__file__).with_name('models.json')),
        help='Path to models.json'
    )
    parser.add_argument(
        '--run-name',
        default='neocoder',
        help='Run folder name under benchmark/outputs/exploration'
    )
    args = parser.parse_args()

    model_cfg = _load_model_entry(args.model_name, Path(args.models_json))
    base_dir = _outputs_root() / args.run_name / model_cfg['name']
    test_neocoder(model_cfg, str(base_dir))
