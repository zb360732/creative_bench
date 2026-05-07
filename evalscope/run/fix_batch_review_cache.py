#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from evalscope.config import TaskConfig
from evalscope.run import run_task


TARGET_DATASETS = {
    'transformation',
    'creative_math',
    'drivel_writing',
}

DEFAULT_JUDGE_CONFIG = Path(__file__).with_name('llm_judge.json')


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding='utf-8'))


def load_default_judge_model_args(max_tokens: int) -> Dict:
    payload = load_json(DEFAULT_JUDGE_CONFIG)
    models = payload.get('models', [])
    if not models:
        raise ValueError(f'No judge models found in {DEFAULT_JUDGE_CONFIG}')
    entry = models[0]
    return {
        'api_url': entry['api_url'],
        'api_key': entry.get('api_key', 'EMPTY'),
        'model_id': entry['model'],
        'generation_config': {'temperature': 0.0, 'max_tokens': max_tokens},
    }


def iter_review_files(run_dir: Path) -> Iterable[Tuple[str, str, Path]]:
    for model_dir in run_dir.iterdir():
        reviews_root = model_dir / 'reviews' / model_dir.name
        if not reviews_root.exists():
            continue
        for review_file in reviews_root.glob('*.jsonl'):
            stem = review_file.stem
            if stem.startswith('drivel_writing_'):
                dataset = 'drivel_writing'
            elif stem.startswith('creative_math_'):
                dataset = 'creative_math'
            elif stem.startswith('transformation_'):
                dataset = 'transformation'
            else:
                continue
            yield model_dir.name, dataset, review_file


def has_empty_scores(review_file: Path) -> bool:
    with review_file.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            score = ((row.get('sample_score') or {}).get('score') or {})
            if score.get('value') == {}:
                return True
    return False


def dataset_args_for(name: str) -> Dict:
    if name == 'creative_math':
        return {'extra_params': {'evaluation_mode': 'full'}}
    if name == 'transformation':
        return {'extra_params': {'evaluation_mode': 'llm_judge', 'judge_max_tokens': 4096}}
    return {}


def build_task_config(run_dir: Path, model_name: str, dataset: str, model_entry: Dict) -> TaskConfig:
    model_dir = run_dir / model_name
    judge_model_args = load_default_judge_model_args(max_tokens=30000)
    return TaskConfig(
        model=model_entry['model'],
        model_id=model_entry['name'],
        api_url=model_entry['api_url'],
        api_key=model_entry.get('api_key', 'EMPTY'),
        eval_type='openai_api',
        eval_backend='Native',
        datasets=[dataset],
        dataset_args={dataset: dataset_args_for(dataset)},
        generation_config={
            'max_tokens': 30000,
            'temperature': 0.0,
            'timeout': 180,
            'retries': 1,
            'retry_interval': 10,
            'batch_size': 64,
        },
        eval_batch_size=64,
        judge_worker_num=64,
        judge_model_args=judge_model_args,
        use_batch_processing=True,
        use_cache=str(model_dir),
        work_dir=str(model_dir),
        rerun_review=True,
        limit=None,
        debug=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--models-json', required=True)
    parser.add_argument('--models', default='')
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    models_cfg = load_json(Path(args.models_json))
    model_map = {m['name']: m for m in models_cfg['models']}
    allowed_models = {m.strip() for m in args.models.split(',') if m.strip()}

    repair_jobs: List[Tuple[str, str, Path]] = []
    for model_name, dataset, review_file in iter_review_files(run_dir):
        if dataset not in TARGET_DATASETS:
            continue
        if allowed_models and model_name not in allowed_models:
            continue
        if has_empty_scores(review_file):
            repair_jobs.append((model_name, dataset, review_file))

    print(f'Found {len(repair_jobs)} repair jobs', flush=True)
    for model_name, dataset, review_file in repair_jobs:
        print(f'  - model={model_name} dataset={dataset} review={review_file.name}', flush=True)

    for model_name, dataset, review_file in repair_jobs:
        model_dir = run_dir / model_name
        report_file = model_dir / 'reports' / model_name / f'{dataset}.json'
        print(f'Repairing model={model_name} dataset={dataset}', flush=True)
        if review_file.exists():
            review_file.unlink()
        if report_file.exists():
            report_file.unlink()

        task_cfg = build_task_config(run_dir, model_name, dataset, model_map[model_name])
        run_task(task_cfg=task_cfg)


if __name__ == '__main__':
    main()
