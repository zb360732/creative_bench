#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _score_value(value: Any) -> Any:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return value
    if value is None:
        return None
    text = str(value)
    try:
        return float(text)
    except ValueError:
        return text


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _iter_model_review_dirs(run_dir: Path) -> Iterable[Tuple[str, Path]]:
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir() or child.name.startswith('.'):
            continue
        review_dir = child / 'reviews' / child.name
        if review_dir.is_dir():
            yield child.name, review_dir


def _dataset_from_review_file(path: Path) -> Optional[str]:
    name = path.name
    if name.startswith('aut_') and name.endswith('.jsonl'):
        return 'aut'
    if name.startswith('bats_') and name.endswith('.jsonl'):
        return 'bats'
    if name.startswith('dat_') and name.endswith('.jsonl'):
        return 'dat'
    if name.startswith('rat_') and name.endswith('.jsonl'):
        return 'rat'
    if name.startswith('metaphor_') and name.endswith('.jsonl'):
        return 'metaphor'
    if name.startswith('creative_math_') and name.endswith('.jsonl'):
        return 'creative_math'
    if name.startswith('neocoder_') and name.endswith('.jsonl'):
        return 'neocoder'
    if name.startswith('transformation_') and name.endswith('.jsonl'):
        return 'transformation'
    if name.startswith('cs4_constraints_') and name.endswith('.jsonl'):
        return 'cs4'
    return None


def _iter_jsonl_records(path: Path) -> Iterable[Dict[str, Any]]:
    latest_by_index: Dict[Any, Dict[str, Any]] = {}
    with path.open('r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = obj.get('index')
            if key is None:
                key = len(latest_by_index)
                while key in latest_by_index:
                    key += 1
            latest_by_index[key] = obj
    for _, obj in sorted(latest_by_index.items(), key=lambda item: str(item[0])):
        yield obj


def _extract_value(record: Dict[str, Any]) -> Dict[str, Any]:
    return (
        record.get('sample_score', {})
        .get('score', {})
        .get('value', {})
        or {}
    )


def _extract_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    return record.get('sample_score', {}).get('sample_metadata', {}) or {}


def _extract_score_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    return record.get('sample_score', {}).get('score', {}).get('metadata', {}) or {}


def _aggregate_model_reviews(model_name: str, review_dir: Path) -> Dict[str, Any]:
    simple_accumulators: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    creative_math_counts = {'total': 0, 'correctness': 0.0, 'coarse': 0.0, 'fine': 0.0}
    cs4_story_groups: Dict[Any, Dict[str, List[float]]] = {}
    transformation_rows: List[Dict[str, Any]] = []

    for path in sorted(review_dir.glob('*.jsonl')):
        if '.ipynb_checkpoints' in path.parts:
            continue
        dataset = _dataset_from_review_file(path)
        if not dataset:
            continue

        for record in _iter_jsonl_records(path):
            value = _extract_value(record)
            if not isinstance(value, dict) or not value:
                continue

            if dataset == 'aut':
                for src, dst in {
                    'aut_fluency': 'aut_fluency',
                    'aut_elaboration': 'aut_elaboration',
                    'aut_flexibility': 'aut_flexibility',
                    'aut_originality': 'aut_originality',
                    'aut_applicability': 'aut_applicability',
                }.items():
                    if src in value:
                        simple_accumulators[dataset][dst].append(float(_score_value(value[src])))
                continue

            if dataset == 'bats':
                if 'bats_accuracy' in value:
                    simple_accumulators[dataset]['bats_accuracy'].append(float(_score_value(value['bats_accuracy'])))
                continue

            if dataset == 'dat':
                if 'dat_semantic_distance' in value:
                    simple_accumulators[dataset]['dat_semantic_distance'].append(float(_score_value(value['dat_semantic_distance'])))
                continue

            if dataset == 'rat':
                if 'rat_accuracy' in value:
                    simple_accumulators[dataset]['rat_accuracy'].append(float(_score_value(value['rat_accuracy'])))
                continue

            if dataset == 'metaphor':
                if 'metaphor_accuracy' in value:
                    simple_accumulators[dataset]['metaphor_accuracy'].append(float(_score_value(value['metaphor_accuracy'])))
                continue

            if dataset == 'transformation':
                for key in ('appropriateness', 'flexibility', 'fluency', 'novelty'):
                    if key in value:
                        simple_accumulators[dataset][f'transformation_{key}'].append(float(_score_value(value[key])))
                score_metadata = _extract_score_metadata(record)
                transformation_rows.append(
                    {
                        'fluency': float(_score_value(value.get('fluency', 0.0))),
                        'source_axiom_id': score_metadata.get('source_axiom_id'),
                        'constraint_count': score_metadata.get('constraint_count'),
                    }
                )
                continue

            if dataset == 'neocoder':
                for key in ('fluency', 'originality', 'appropriateness'):
                    if key in value:
                        simple_accumulators[dataset][f'neocoder_{key}'].append(float(_score_value(value[key])))
                continue

            if dataset == 'creative_math':
                creative_math_counts['total'] += 1
                creative_math_counts['correctness'] += float(_score_value(value.get('correctness', 0.0)))
                creative_math_counts['coarse'] += float(_score_value(value.get('coarse_grained_novelty', 0.0)))
                creative_math_counts['fine'] += float(_score_value(value.get('fine_grained_novelty', 0.0)))
                continue

            if dataset == 'cs4':
                metadata = _extract_metadata(record)
                story_id = metadata.get('story_id')
                if story_id is None:
                    story_id = f'unknown_{path.stem}_{record.get("index", "")}'
                if story_id not in cs4_story_groups:
                    cs4_story_groups[story_id] = {
                        'fluency': [],
                        'flexibility': [],
                        'appropriateness': [],
                        'novelty': [],
                    }
                for key in ('fluency', 'flexibility', 'appropriateness', 'novelty'):
                    if key in value:
                        cs4_story_groups[story_id][key].append(float(_score_value(value[key])))

    result: Dict[str, Any] = {'model': model_name}

    for dataset_metrics in simple_accumulators.values():
        for name, values in dataset_metrics.items():
            mean_value = _mean(values)
            if mean_value is not None:
                result[name] = round(mean_value, 4)

    if creative_math_counts['total'] > 0:
        total = creative_math_counts['total']
        correctness_ratio = creative_math_counts['correctness'] / total
        novelty_ratio = creative_math_counts['coarse'] / total
        novel_unknown_ratio = creative_math_counts['fine'] / total
        originality = 0.7 * novelty_ratio + 0.3 * novel_unknown_ratio
        result['creative_math_appropriateness'] = round(correctness_ratio, 4)
        result['creative_math_originality'] = round(originality, 4)

    if transformation_rows:
        per_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in transformation_rows:
            source_axiom_id = row.get('source_axiom_id')
            if source_axiom_id:
                per_source[str(source_axiom_id)].append(row)

        flexibility_values: List[float] = []
        for rows in per_source.values():
            level_map: Dict[int, List[float]] = defaultdict(list)
            for row in rows:
                level = row.get('constraint_count')
                if isinstance(level, int):
                    level_map[level].append(float(row['fluency']))
            if len(level_map) < 2:
                continue

            weighted_numerator = 0.0
            weighted_denominator = 0.0
            for level in sorted(level_map):
                mean_fluency = statistics.mean(level_map[level])
                weighted_numerator += level * mean_fluency
                weighted_denominator += level
            if weighted_denominator > 0:
                flexibility_values.append(round(weighted_numerator / weighted_denominator, 4))

        flexibility_mean = _mean(flexibility_values)
        if flexibility_mean is not None:
            result['transformation_flexibility'] = round(flexibility_mean, 4)

    if cs4_story_groups:
        story_means: Dict[str, List[float]] = defaultdict(list)
        for metrics in cs4_story_groups.values():
            for key in ('fluency', 'flexibility', 'appropriateness', 'novelty'):
                if metrics[key]:
                    story_means[key].append(sum(metrics[key]) / len(metrics[key]))
        for key in ('appropriateness', 'flexibility', 'fluency', 'novelty'):
            mean_value = _mean(story_means.get(key, []))
            if mean_value is not None:
                result[f'cs4_{key}'] = round(mean_value, 4)

    return result


def _column_order(rows: List[Dict[str, Any]]) -> List[str]:
    preferred = [
        'aut_fluency',
        'aut_elaboration',
        'aut_flexibility',
        'aut_originality',
        'aut_applicability',
        'bats_accuracy',
        'creative_math_originality',
        'creative_math_appropriateness',
        'cs4_appropriateness',
        'cs4_flexibility',
        'cs4_fluency',
        'cs4_novelty',
        'dat_semantic_distance',
        'metaphor_accuracy',
        'neocoder_fluency',
        'neocoder_originality',
        'neocoder_appropriateness',
        'rat_accuracy',
        'transformation_appropriateness',
        'transformation_flexibility',
        'transformation_fluency',
        'transformation_novelty',
    ]
    seen = {k for row in rows for k in row.keys() if k != 'model'}
    return [col for col in preferred if col in seen]


def _write_matrix_csv(
    out_path: Path, model_order: List[str], column_order: List[str], matrix: Dict[str, Dict[str, Any]]
) -> None:
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model'] + column_order)
        for model in model_order:
            row = [model]
            for col in column_order:
                row.append(matrix.get(model, {}).get(col, ''))
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate a subdimension matrix from review jsonl files.')
    parser.add_argument('--run-dir', required=True, help='Run directory containing model review jsonl files.')
    parser.add_argument('--out-dir', default='', help='Output directory. Defaults to run dir.')
    parser.add_argument('--out-name', default='subdimension_matrix', help='Base name for output files.')
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f'Run dir not found: {run_dir}')

    rows: List[Dict[str, Any]] = []
    for model_name, review_dir in _iter_model_review_dirs(run_dir):
        row = _aggregate_model_reviews(model_name, review_dir)
        if len(row) > 1:
            rows.append(row)

    model_order = [row['model'] for row in rows]
    column_order = _column_order(rows)
    matrix = {row['model']: {k: v for k, v in row.items() if k != 'model'} for row in rows}

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f'{args.out_name}.json'
    json_payload = {
        'run_dir': str(run_dir),
        'source': 'reviews_jsonl',
        'models': model_order,
        'columns': column_order,
        'matrix': matrix,
    }
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding='utf-8')

    csv_path = out_dir / f'{args.out_name}.csv'
    _write_matrix_csv(csv_path, model_order, column_order, matrix)

    print(f'[OK] JSON: {json_path}')
    print(f'[OK] CSV: {csv_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
