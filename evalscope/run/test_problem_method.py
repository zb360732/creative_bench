#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Problem-Method Transfer benchmark integration.

This script tests the problem_method benchmark in evalscope.
It supports two evaluation modes:
- simplified: Fast mode (no LLM judges)
- full: Uses 3 LLM judges for feasibility and novelty
"""

import json
import os
import sys
from pathlib import Path

# Add evalscope to path
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


def _limit_slug(limit: int | None) -> str:
    if limit is None:
        return 'full'
    return f'limit{limit}'


def _default_run_name(mode: str, limit: int | None) -> str:
    return f'problem_method_{mode}_{_limit_slug(limit)}'


def _build_task_cfg(model_cfg: dict, mode: str, work_dir: str, limit: int | None):
    extra_params = {
        'evaluation_mode': mode,
        'cluster_count': 50,
        'max_problems_per_target': 2,
        'max_refs_per_target': 3,
        'max_methods_per_ref': 1,
        'reference_selection': 'farthest',
        'random_seed': 42,
    }

    if mode == 'full':
        extra_params.update({
            'judge_api_url': model_cfg['api_url'],
            'judge_api_key': model_cfg['api_key'],
            'judge_model_id': model_cfg['model'],
        })

    return TaskConfig(
        model=model_cfg['model'],
        model_id=model_cfg['name'],
        api_url=model_cfg['api_url'],
        api_key=model_cfg['api_key'],
        eval_type='openai_api',
        datasets=['problem_method'],
        limit=limit,
        dataset_args={
            'problem_method': {
                'extra_params': extra_params,
            }
        },
        generation_config={
            'max_tokens': 1024,
            'temperature': 0.7,
        },
        timeout=180,
        work_dir=work_dir,
        no_timestamp=True,
    )


def test_mode(model_cfg: dict, mode: str, limit: int | None):
    run_name = _default_run_name(mode, limit)
    work_dir = str(_outputs_root() / run_name)

    logger.info('=' * 80)
    logger.info('Testing problem_method in %s mode', mode.upper())
    logger.info('=' * 80)
    logger.info('Model: %s', model_cfg['name'])
    logger.info('API: %s', model_cfg['api_url'])
    logger.info('Limit: %s', limit)

    task_cfg = _build_task_cfg(model_cfg, mode, work_dir, limit)
    run_task(task_cfg=task_cfg)

    logger.info('Results saved to: %s', task_cfg.work_dir)


def main():
    models_json = Path(__file__).resolve().parent / 'models.json'
    model_name = os.getenv('EVAL_MODEL_NAME', '')
    model_cfg = _load_model_entry(model_name, models_json)

    mode = os.getenv('PROBLEM_METHOD_MODE', 'simplified').strip().lower()
    limit_env = os.getenv('PROBLEM_METHOD_LIMIT', '5').strip()
    limit = int(limit_env) if limit_env else None

    if mode not in {'simplified', 'full'}:
        raise ValueError(f'Unsupported mode: {mode}')

    test_mode(model_cfg, mode, limit)


if __name__ == '__main__':
    main()
