#!/usr/bin/env python3

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalscope.config import TaskConfig
from evalscope.run import run_task


DEFAULT_SOURCE_RUN = Path(
    '/inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/outputs/transformation/'
    'transformation_closed_full_20260409_1149'
)
DEFAULT_OUT_ROOT = Path(
    '/inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/outputs/transformation'
)
DEFAULT_JUDGE_CONFIG = Path(__file__).with_name('llm_judge.json')


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, 'to_dict') and callable(getattr(value, 'to_dict')):
        try:
            return _to_jsonable(value.to_dict())
        except Exception:
            pass
    if hasattr(value, 'model_dump') and callable(getattr(value, 'model_dump')):
        try:
            return _to_jsonable(value.model_dump())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _resolve_api_key(entry: Dict[str, Any]) -> str:
    api_key = str(entry.get('api_key', 'EMPTY'))
    api_key_env = entry.get('api_key_env')
    if api_key_env:
        import os
        api_key = os.getenv(str(api_key_env), api_key)
    return api_key


def _judge_model_args(entry: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
    return {
        'api_url': str(entry.get('api_url')),
        'api_key': _resolve_api_key(entry),
        'model_id': str(entry.get('model_id') or entry.get('model') or entry.get('name')),
        'generation_config': {'temperature': 0.0, 'max_tokens': max_tokens},
    }


def _load_judge_model_args_list(max_tokens: int, include_names: List[str]) -> List[Dict[str, Any]]:
    payload = _load_json(DEFAULT_JUDGE_CONFIG)
    models = payload.get('models', [])
    if not isinstance(models, list) or not models:
        raise ValueError(f'No judge models found in {DEFAULT_JUDGE_CONFIG}')
    if include_names:
        wanted = {name.strip() for name in include_names if name.strip()}
        models = [entry for entry in models if str(entry.get('name') or entry.get('model')).strip() in wanted]
        found = {str(entry.get('name') or entry.get('model')).strip() for entry in models}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(f'Judge models not found in {DEFAULT_JUDGE_CONFIG}: {missing}')
    return [_judge_model_args(entry, max_tokens=max_tokens) for entry in models]


def _parse_source_runs(values: List[str]) -> List[Path]:
    raw_values = values or [str(DEFAULT_SOURCE_RUN)]
    paths: List[Path] = []
    for value in raw_values:
        for part in str(value).split(','):
            text = part.strip()
            if text:
                paths.append(Path(text).expanduser().resolve())
    return paths


def _iter_source_models(source_runs: List[Path]) -> Dict[str, Path]:
    models: Dict[str, Path] = {}
    for source_run in source_runs:
        if not source_run.exists():
            raise FileNotFoundError(f'Source run not found: {source_run}')
        for model_dir in sorted(source_run.iterdir()):
            if not model_dir.is_dir():
                continue
            pred = model_dir / 'predictions' / model_dir.name / 'transformation_default.jsonl'
            if pred.exists() and model_dir.name not in models:
                models[model_dir.name] = source_run
    return models


def _find_model_source(source_runs: List[Path], model_name: str) -> Path:
    for source_run in source_runs:
        pred = source_run / model_name / 'predictions' / model_name / 'transformation_default.jsonl'
        if pred.exists():
            return source_run
    raise FileNotFoundError(f'No source prediction found for model={model_name} in {source_runs}')


def _old_iter_source_models(source_run: Path) -> List[str]:
    models = []
    for model_dir in sorted(source_run.iterdir()):
        pred = model_dir / 'predictions' / model_dir.name / 'transformation_default.jsonl'
        if pred.exists():
            models.append(model_dir.name)
    return models


def _copy_prediction_prefix(source_run: Path, out_run: Path, model_name: str, limit: int) -> None:
    src = source_run / model_name / 'predictions' / model_name / 'transformation_default.jsonl'
    if not src.exists():
        raise FileNotFoundError(f'Missing source predictions: {src}')

    dst = out_run / model_name / 'predictions' / model_name / 'transformation_default.jsonl'
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows_by_id: Dict[int, str] = {}
    with src.open('r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get('sample_id', row.get('index'))
            if isinstance(sample_id, str) and sample_id.isdigit():
                sample_id = int(sample_id)
            if isinstance(sample_id, int) and 0 <= sample_id < limit:
                rows_by_id[sample_id] = line
    missing = [idx for idx in range(limit) if idx not in rows_by_id]
    if missing:
        raise ValueError(f'Missing prediction sample_ids {missing} in {src}')
    dst.write_text(''.join(rows_by_id[idx] for idx in range(limit)), encoding='utf-8')

    # Keep a compact source marker for reproducibility without copying old review/cache files.
    marker = out_run / model_name / 'source_predictions.txt'
    marker.write_text(f'{src}\nlimit={limit}\n', encoding='utf-8')


def _copy_model_config_if_present(source_run: Path, out_run: Path, model_name: str) -> None:
    src_configs = source_run / model_name / 'configs'
    if not src_configs.exists():
        return
    dst_configs = out_run / model_name / 'source_configs'
    dst_configs.mkdir(parents=True, exist_ok=True)
    for path in sorted(src_configs.glob('task_config_*.yaml'))[-1:]:
        shutil.copy2(path, dst_configs / path.name)


def _build_task_config(
    out_run: Path,
    model_name: str,
    limit: int,
    judge_worker_num: int,
    judge_model_names: List[str],
) -> TaskConfig:
    judges = _load_judge_model_args_list(max_tokens=4096, include_names=judge_model_names)
    return TaskConfig(
        model=model_name,
        model_id=model_name,
        eval_backend='Native',
        datasets=['transformation'],
        dataset_args={
            'transformation': {
                'extra_params': {
                    'evaluation_mode': 'llm_judge',
                    'judge_max_tokens': 4096,
                    'judge_temperature': 0.0,
                    'judge_max_retries': 4,
                    'judge_sleep_seconds': 1.0,
                }
            }
        },
        generation_config={
            'max_tokens': 1,
            'temperature': 0.0,
            'timeout': 30,
            'retries': 0,
        },
        limit=limit,
        eval_batch_size=1,
        judge_worker_num=judge_worker_num,
        judge_model_args=judges[0],
        judge_model_args_list=judges,
        use_batch_processing=False,
        use_cache=str(out_run / model_name),
        work_dir=str(out_run / model_name),
        no_timestamp=True,
        rerun_review=True,
        ignore_errors=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description='Rerun transformation reviews from existing prediction caches.')
    parser.add_argument(
        '--source-run',
        action='append',
        default=[],
        help='Existing run directory containing predictions. Can be repeated or comma-separated.',
    )
    parser.add_argument('--out-root', default=str(DEFAULT_OUT_ROOT))
    parser.add_argument('--run-name', default='transformation_review_only_limit10_newjudge')
    parser.add_argument('--models', default='', help='Comma-separated model names. Default: all models with predictions.')
    parser.add_argument('--judge-models', default='', help='Comma-separated judge names from run/llm_judge.json.')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--judge-worker-num', type=int, default=2)
    args = parser.parse_args()

    source_runs = _parse_source_runs(args.source_run)
    out_run = Path(args.out_root).expanduser().resolve() / args.run_name
    out_run.mkdir(parents=True, exist_ok=True)

    if args.models.strip():
        models = [item.strip() for item in args.models.split(',') if item.strip()]
        model_sources = {model_name: _find_model_source(source_runs, model_name) for model_name in models}
    else:
        model_sources = _iter_source_models(source_runs)
        models = sorted(model_sources)
    if not models:
        raise ValueError(f'No source models found in {source_runs}')
    judge_model_names = [item.strip() for item in args.judge_models.split(',') if item.strip()]

    summary: Dict[str, Any] = {}
    for model_name in models:
        source_run = model_sources[model_name]
        print(f'Preparing review-only run for {model_name} from {source_run}', flush=True)
        _copy_prediction_prefix(source_run, out_run, model_name, args.limit)
        _copy_model_config_if_present(source_run, out_run, model_name)
        cfg = _build_task_config(out_run, model_name, args.limit, args.judge_worker_num, judge_model_names)
        try:
            result = run_task(task_cfg=cfg)
            summary[model_name] = {'status': 'ok', 'result': _to_jsonable(result)}
        except Exception as exc:
            summary[model_name] = {'status': 'error', 'error': repr(exc)}
            print(f'ERROR model={model_name}: {exc!r}', flush=True)

    summary_path = out_run / 'summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {summary_path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
