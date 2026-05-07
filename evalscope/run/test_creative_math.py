#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test CreativeMath benchmark integration

This script tests the CreativeMath benchmark integration in evalscope.
It supports two evaluation modes:
- simplified: Fast testing mode (assumes correctness and novelty)
- full: Complete evaluation with 3 LLM evaluators (requires API keys)
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
    return f'creative_math_{mode}_{_limit_slug(limit)}'


def test_simplified_mode_local(model_cfg: dict, work_dir: str):
    """Test CreativeMath in simplified mode with configured API."""

    logger.info("=" * 80)
    logger.info("Testing CreativeMath in SIMPLIFIED mode with configured API")
    logger.info("=" * 80)
    logger.info(f"Using API at {model_cfg['api_url']}")
    logger.info(f"Model: {model_cfg['name']}")
    logger.info("This mode assumes correctness and novelty for speed")
    logger.info("")

    task_cfg = TaskConfig(
        model=model_cfg['model'],
        model_id=model_cfg['name'],
        api_url=model_cfg['api_url'],
        api_key=model_cfg['api_key'],
        eval_type='openai_api',
        datasets=['creative_math'],
        limit=5,  # Limit to first 5 samples (move to top level)
        dataset_args={
            'creative_math': {
                'extra_params': {
                    'evaluation_mode': 'simplified',  # Fast mode
                }
            }
        },
        generation_config={
            'max_tokens': 2048,
            'temperature': 0.7,
        },
        timeout=120,
        work_dir=work_dir,
        no_timestamp=True,
    )

    logger.info(f"Configuration: limit=5 samples, evaluation_mode=simplified")
    logger.info(f"Output directory: {task_cfg.work_dir}")
    logger.info("")

    run_task(task_cfg=task_cfg)

    logger.info("")
    logger.info("=" * 80)
    logger.info("Simplified test complete!")
    logger.info(f"Results saved to: {task_cfg.work_dir}")
    logger.info("=" * 80)


def test_full_mode_local(model_cfg: dict, work_dir: str):
    """Test CreativeMath in full mode with configured API."""

    logger.info("=" * 80)
    logger.info("Testing CreativeMath in FULL mode with configured API")
    logger.info("=" * 80)
    logger.info(f"Using API at {model_cfg['api_url']} for all judges")
    logger.info(f"Model: {model_cfg['name']}")
    logger.info("This mode uses 3 LLM judges for complete evaluation")
    logger.info("")

    task_cfg = TaskConfig(
        model=model_cfg['model'],
        model_id=model_cfg['name'],
        api_url=model_cfg['api_url'],
        api_key=model_cfg['api_key'],
        eval_type='openai_api',
        datasets=['creative_math'],
        limit=5,  # Test with first 5 samples
        dataset_args={
            'creative_math': {
                'extra_params': {
                    'evaluation_mode': 'full',  # Full evaluation
                    'evaluator_models': [
                        'evaluator1',
                        'evaluator2',
                        'evaluator3'
                    ],
                    # Configure LLM judges to use local API
                    'judge_api_url': model_cfg['api_url'],
                    'judge_api_key': model_cfg['api_key'],
                    'judge_model_id': model_cfg['model']
                }
            }
        },
        generation_config={
            'max_tokens': 2048,
            'temperature': 0.7,
        },
        timeout=180,
        work_dir=work_dir,
        no_timestamp=True,
    )

    logger.info(f"Configuration: limit=5 samples, evaluation_mode=full")
    logger.info(f"Evaluators: All using local Qwen2.5-7B-Instruct")
    logger.info(f"Output directory: {task_cfg.work_dir}")
    logger.info("")

    run_task(task_cfg=task_cfg)

    logger.info("")
    logger.info("=" * 80)
    logger.info("Full test complete!")
    logger.info(f"Results saved to: {task_cfg.work_dir}")
    logger.info("=" * 80)


def test_dataset_loading(model_cfg: dict):
    """Test dataset loading without running full evaluation"""

    logger.info("=" * 80)
    logger.info("Testing CreativeMath dataset loading")
    logger.info("=" * 80)

    # Import the adapter directly
    from evalscope.api.registry import get_benchmark
    from evalscope.config import TaskConfig as TC

    # Create a minimal config
    config = TC(
        model=model_cfg['model'],
        model_id=model_cfg['name'],
        datasets=['creative_math'],
        dataset_args={
            'creative_math': {
                'limit': 10,
                'extra_params': {
                    'evaluation_mode': 'simplified',
                }
            }
        }
    )

    # Get the adapter
    adapter = get_benchmark('creative_math', config)

    # Load dataset
    dataset_dict = adapter.load_dataset()

    logger.info(f"Dataset loaded successfully!")
    logger.info(f"Subsets: {list(dataset_dict.keys())}")

    for subset_name, samples in dataset_dict.items():
        logger.info(f"Subset '{subset_name}': {len(samples)} samples")

        if samples:
            # Show first sample
            first_sample = samples[0]
            logger.info(f"\nFirst sample metadata:")
            logger.info(f"  - problem_id: {first_sample.metadata.get('problem_id')}")
            logger.info(f"  - k: {first_sample.metadata.get('k')}")
            logger.info(f"  - n: {first_sample.metadata.get('n')}")
            logger.info(f"  - competition: {first_sample.metadata.get('competition')}")
            logger.info(f"  - difficulty: {first_sample.metadata.get('difficulty')}")
            logger.info(f"  - input length: {len(first_sample.input[0].content)} chars")

    logger.info("=" * 80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test CreativeMath benchmark integration')
    parser.add_argument(
        '--mode',
        choices=['simplified', 'full', 'load'],
        default='simplified',
        help='Test mode: simplified (fast), full (accurate with 3 judges using local API), or load (just test loading)'
    )
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
        default='',
        help='Run folder name under benchmark/outputs/exploration'
    )
    args = parser.parse_args()

    model_cfg = _load_model_entry(args.model_name, Path(args.models_json))
    limit_for_run_name = 5 if args.mode in {'simplified', 'full'} else None
    run_name = args.run_name or _default_run_name(args.mode, limit_for_run_name)
    base_dir = _outputs_root() / run_name / model_cfg['name']

    try:
        if args.mode == 'simplified':
            test_simplified_mode_local(model_cfg, str(base_dir))
        elif args.mode == 'full':
            test_full_mode_local(model_cfg, str(base_dir))
        elif args.mode == 'load':
            test_dataset_loading(model_cfg)
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
