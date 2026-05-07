#!/usr/bin/env python3
"""Test CS4 story grouping - test a few complete stories with all 5 constraint levels"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evalscope import TaskConfig, run_task
from evalscope.utils.logger import get_logger

logger = get_logger()

# Test with limit=10, which should give us 2 complete stories (2 stories × 5 constraints = 10 samples)
task_cfg = TaskConfig(
    model='Qwen2.5-7B-Instruct',
    api_url='http://localhost:8007/v1/chat/completions',
    api_key='EMPTY',
    eval_type='openai_api',
    datasets=['cs4'],
    limit=10,  # Should give us first 2 complete stories
    dataset_args={
        'cs4': {
            'subset_list': ['default'],  # Use default to get all constraint levels
            'extra_params': {
                'evaluation_mode': 'simplified',
            }
        }
    },
    generation_config={
        'max_tokens': 512,
        'temperature': 0.8,
    },
    work_dir='outputs/cs4_test_story_grouping',
)

logger.info("="*80)
logger.info("Testing CS4 story grouping with limit=10")
logger.info("Expected: 2 stories × 5 constraint levels = 10 samples")
logger.info("="*80)

run_task(task_cfg=task_cfg)

logger.info("="*80)
logger.info("Check the aggregation results to see story-level averaging")
logger.info("="*80)
