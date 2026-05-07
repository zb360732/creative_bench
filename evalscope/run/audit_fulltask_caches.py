#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


TASK_SUBSETS = {
    'dat': ['default'],
    'bats': ['sampled'],
    'rat': ['default'],
    'metaphor': ['default'],
    'aut': ['default'],
    'creative_math': ['default'],
    'cs4': ['constraints_7', 'constraints_15', 'constraints_23', 'constraints_31', 'constraints_39'],
    'neocoder': ['default'],
    'transformation': ['default'],
}

TASK_VALUE_KEYS = {
    'dat': ['dat_semantic_distance'],
    'bats': ['bats_accuracy'],
    'rat': ['rat_accuracy'],
    'metaphor': ['metaphor_accuracy'],
    'aut': ['aut_fluency', 'aut_elaboration', 'aut_flexibility', 'aut_originality', 'aut_applicability'],
    'creative_math': ['correctness', 'coarse_grained_novelty', 'fine_grained_novelty'],
    'cs4': ['fluency', 'grammar', 'coherence', 'likability', 'flexibility', 'appropriateness', 'novelty', 'quc'],
    'neocoder': ['correctness', 'follow_constraints', 'new_techniques', 'new_techniques_ratio', 'fluency', 'originality', 'appropriateness'],
    'transformation': ['fluency', 'novelty', 'appropriateness', 'flexibility'],
}

LLM_JUDGE_TASKS = {'creative_math', 'cs4', 'transformation'}

CREATIVE_MATH_STAGES = ['correctness', 'coarse_novelty', 'fine_novelty']
CS4_STAGES = ['constraint_satisfaction', 'story_quality', 'novelty']
TRANSFORMATION_STAGES = ['default']

JUDGE_ALIASES = {
    'deepseek-chat': 'deepseek',
    'deepseek-v3-2-251201': 'deepseek',
    'deepseek-v3': 'deepseek',
    'gpt-4': 'gpt',
    'gpt-4o': 'gpt',
    'gpt-5': 'gpt',
    'gpt-5.2': 'gpt',
    'claude-3-opus': 'claude',
    'gemini-1.5-pro': 'gemini',
    'gemini-3.1-pro-preview': 'gemini',
}


def load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def canonicalize_judge_name(name: Any) -> str:
    raw = str(name or '').strip()
    if not raw:
        return raw
    return JUDGE_ALIASES.get(raw, raw)


def load_expected_judge_names() -> Optional[List[str]]:
    cfg_path = Path('/root/benchmark/evalscope/run/llm_judge.json')
    if not cfg_path.exists():
        return None
    try:
        data = load_json(cfg_path)
    except Exception:
        return None
    models = data.get('models')
    if not isinstance(models, list):
        return None
    names: List[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get('model') or item.get('model_id') or item.get('name') or '').strip()
        if name:
            names.append(canonicalize_judge_name(name))
    return sorted(set(names)) or None


def find_model_dirs(run_dir: Path) -> List[Path]:
    return sorted(
        p for p in run_dir.iterdir()
        if p.is_dir() and (p / 'predictions').exists() and (p / 'reviews').exists()
    )


def read_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == '.jsonl':
        records = read_jsonl_records(path)
        bad_rows = [row for row in records if '__parse_error__' in row]
        if bad_rows:
            raise ValueError(f'Invalid JSONL rows in {path}: {bad_rows[0]["__parse_error__"]}')
        return records

    data = load_json(path)
    if isinstance(data, list):
        return data
    raise ValueError(f'Expected list in {path}')


def read_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                records.append({'__parse_error__': f'line {line_no}: {exc}'})
                continue
            if isinstance(row, dict):
                records.append(row)
    return records


def normalize_instruction_prefix(text: Any, limit: int = 120) -> str:
    return str(text or '')[:limit]


def sample_key_for_task(task: str, row: Dict[str, Any]) -> Optional[str]:
    metadata = row.get('metadata') or {}
    if task == 'creative_math':
        return f"{metadata.get('problem_id', 'unknown')}::k={metadata.get('k', 'unknown')}"
    if task == 'cs4':
        return ':'.join([
            str(metadata.get('story_id', 'unknown')),
            str(metadata.get('number_of_constraints', 'unknown')),
            normalize_instruction_prefix(metadata.get('instruction', '')),
        ])
    if task == 'transformation':
        candidate_id = metadata.get('candidate_id')
        if candidate_id not in (None, ''):
            return str(candidate_id)
    return None


def judge_cache_file(model_dir: Path, task: str) -> Path:
    return model_dir / 'judge_cache' / model_dir.name / f'{task}.jsonl'


def expected_stages_for_task(task: str) -> List[str]:
    if task == 'creative_math':
        return CREATIVE_MATH_STAGES
    if task == 'cs4':
        return CS4_STAGES
    if task == 'transformation':
        return TRANSFORMATION_STAGES
    return []


def load_judge_cache_index(model_dir: Path, task: str) -> Tuple[Dict[Tuple[str, str], Dict[str, Dict[str, Any]]], List[str]]:
    cache_path = judge_cache_file(model_dir, task)
    indexed: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    issues: List[str] = []
    if not cache_path.exists():
        return indexed, issues

    for row in read_jsonl_records(cache_path):
        if '__parse_error__' in row:
            issues.append(f'judge cache parse error in {cache_path}: {row["__parse_error__"]}')
            continue
        sample_key = str(row.get('sample_key', '') or '')
        stage = str(row.get('stage', '') or '')
        judge_key = canonicalize_judge_name(row.get('judge_key', '') or '')
        if not sample_key or not stage or not judge_key:
            issues.append(f'judge cache record missing sample_key/stage/judge_key in {cache_path}')
            continue
        indexed.setdefault((sample_key, stage), {})[judge_key] = row
    return indexed, issues


def judge_progress_for_sample(
    task: str,
    sample_key: Optional[str],
    judge_index: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]],
    expected_judges: Optional[List[str]],
) -> Optional[Dict[str, Any]]:
    if task not in LLM_JUDGE_TASKS or not sample_key:
        return None

    expected = list(expected_judges or [])
    expected_set = set(expected)
    stages = expected_stages_for_task(task)
    by_stage: Dict[str, Any] = {}
    total_success = 0
    total_failed = 0
    total_observed = 0

    for stage in stages:
        records = judge_index.get((sample_key, stage), {})
        success = sorted([judge for judge, rec in records.items() if rec.get('status') == 'success'])
        failed = sorted([judge for judge, rec in records.items() if rec.get('status') != 'success'])
        observed = sorted(records.keys())
        if expected_set:
            missing = sorted(expected_set - set(success))
        else:
            missing = []
        by_stage[stage] = {
            'success_judges': success,
            'failed_judges': failed,
            'observed_judges': observed,
            'success_count': len(success),
            'failed_count': len(failed),
            'observed_count': len(observed),
            'missing_judges': missing,
            'missing_count': len(missing) if expected_set else None,
        }
        total_success += len(success)
        total_failed += len(failed)
        total_observed += len(observed)

    return {
        'sample_key': sample_key,
        'expected_judges': expected,
        'expected_judge_count': len(expected) if expected else None,
        'stages': by_stage,
        'total_success_judges': total_success,
        'total_failed_judges': total_failed,
        'total_observed_judges': total_observed,
    }


def check_required_value_keys(task: str, sample_score: Dict[str, Any]) -> List[str]:
    score = sample_score.get('score') or {}
    value = score.get('value') or {}
    missing = [k for k in TASK_VALUE_KEYS[task] if k not in value]
    issues = []
    if missing:
        issues.append(f'missing value keys: {missing}')
    if not value:
        issues.append('empty score.value')
    return issues


def check_judge_completeness(task: str, sample_score: Dict[str, Any], expected_judges: Optional[List[str]]) -> List[str]:
    score = sample_score.get('score') or {}
    metadata = score.get('metadata') or {}
    issues: List[str] = []
    expected_set: Set[str] = set(expected_judges or [])

    if task == 'creative_math':
        for field in ['correctness_evaluations', 'coarse_novelty_evaluations']:
            block = metadata.get(field)
            if not isinstance(block, dict):
                issues.append(f'{field} missing or not dict')
                continue
            model_votes = [
                canonicalize_judge_name(k)
                for k in block.keys()
                if k not in {'final_decision', 'mode', 'reason', 'error'}
            ]
            if len(model_votes) < 3:
                issues.append(f'{field} has only {len(model_votes)} judge votes: {model_votes}')
            if expected_set and set(model_votes) != expected_set:
                issues.append(f'{field} judge names mismatch: expected {sorted(expected_set)}, got {sorted(model_votes)}')
            if 'final_decision' not in block:
                issues.append(f'{field} missing final_decision')

        fine_block = metadata.get('fine_novelty_evaluations')
        if not isinstance(fine_block, dict):
            issues.append('fine_novelty_evaluations missing or not dict')
        elif 'final_decision' not in fine_block:
            issues.append('fine_novelty_evaluations missing final_decision')
        else:
            fine_votes = [
                canonicalize_judge_name(k)
                for k in fine_block.keys()
                if k not in {'final_decision', 'mode', 'reason', 'error'}
            ]
            if fine_votes and expected_set and set(fine_votes) != expected_set:
                issues.append(f'fine_novelty_evaluations judge names mismatch: expected {sorted(expected_set)}, got {sorted(fine_votes)}')

    elif task == 'cs4':
        for field in ['constraint_details', 'quality_details', 'novelty_details']:
            block = metadata.get(field)
            if not isinstance(block, list):
                issues.append(f'{field} missing or not list')
            elif len(block) < 3:
                issues.append(f'{field} has only {len(block)} judge responses')

    elif task == 'transformation':
        raw = metadata.get('raw_judge_response')
        if not isinstance(raw, list):
            issues.append('raw_judge_response missing or not list')
        elif len(raw) < 3:
            issues.append(f'raw_judge_response has only {len(raw)} judge responses')

    return issues


def audit_subset(
    task: str,
    subset: str,
    pred_path: Path,
    review_path: Path,
    expected_judges: Optional[List[str]],
    model_dir: Path,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        'task': task,
        'subset': subset,
        'prediction_file': str(pred_path),
        'review_file': str(review_path),
        'judge_cache_file': str(judge_cache_file(model_dir, task)),
        'status': 'ok',
        'expected_total': 0,
        'prediction_count': 0,
        'review_count': 0,
        'successful_review_count': 0,
        'issues': [],
        'sample_issues': [],
        'missing_review_samples': [],
    }

    judge_index: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    judge_cache_issues: List[str] = []
    if task in LLM_JUDGE_TASKS:
        judge_index, judge_cache_issues = load_judge_cache_index(model_dir, task)
        result['issues'].extend(judge_cache_issues)
        if judge_cache_issues:
            result['status'] = 'error'

    if not pred_path.exists():
        result['status'] = 'error'
        result['issues'].append('missing prediction file')
        return result

    predictions = read_records(pred_path)
    reviews = read_records(review_path) if review_path.exists() else []
    if not review_path.exists():
        result['status'] = 'error'
        result['issues'].append('missing review file')

    result['expected_total'] = len(predictions)
    result['prediction_count'] = len(predictions)
    result['review_count'] = len(reviews)

    if len(predictions) != len(reviews):
        result['status'] = 'error'
        result['issues'].append(f'prediction/review count mismatch: {len(predictions)} vs {len(reviews)}')

    review_by_index = {row.get('index'): row for row in reviews}
    pred_indexes = [row.get('index') for row in predictions]
    missing_review_indexes = [idx for idx in pred_indexes if idx not in review_by_index]
    if missing_review_indexes:
        result['status'] = 'error'
        result['issues'].append(f'missing reviews for indexes: {missing_review_indexes[:20]}')

    prediction_by_index = {row.get('index'): row for row in predictions}
    for idx in missing_review_indexes:
        pred_row = prediction_by_index.get(idx) or {}
        sample_key = sample_key_for_task(task, pred_row)
        missing_entry: Dict[str, Any] = {'index': idx}
        if sample_key:
            missing_entry['sample_key'] = sample_key
        judge_progress = judge_progress_for_sample(task, sample_key, judge_index, expected_judges)
        if judge_progress:
            missing_entry['judge_progress'] = judge_progress
        result['missing_review_samples'].append(missing_entry)

    for row in reviews:
        idx = row.get('index')
        sample_score = row.get('sample_score') or {}
        sample_issues = []
        sample_issues.extend(check_required_value_keys(task, sample_score))
        sample_issues.extend(check_judge_completeness(task, sample_score, expected_judges))
        score = sample_score.get('score') or {}
        if score.get('prediction') in (None, ''):
            sample_issues.append('empty original prediction')
        if score.get('extracted_prediction') in (None, ''):
            sample_issues.append('empty extracted prediction')

        if sample_issues:
            result['status'] = 'error'
            result['sample_issues'].append({'index': idx, 'issues': sample_issues})
        else:
            result['successful_review_count'] += 1

    return result


def audit_model(run_dir: Path, model_dir: Path, verbose: bool = False) -> Dict[str, Any]:
    model_name = model_dir.name
    pred_root = model_dir / 'predictions' / model_name
    review_root = model_dir / 'reviews' / model_name
    model_result = {'model': model_name, 'status': 'ok', 'subsets': []}
    expected_judges = load_expected_judge_names()

    for task, subsets in TASK_SUBSETS.items():
        for subset in subsets:
            pred_path = pred_root / f'{task}_{subset}.jsonl'
            review_path = review_root / f'{task}_{subset}.jsonl'
            subset_result = audit_subset(task, subset, pred_path, review_path, expected_judges, model_dir)
            model_result['subsets'].append(subset_result)
            if subset_result['status'] != 'ok':
                model_result['status'] = 'error'

    if verbose:
        print(json.dumps(model_result, ensure_ascii=False, indent=2))
    return model_result


def build_resume_candidates(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for model_result in results:
        model_name = model_result.get('model')
        for subset_result in model_result.get('subsets', []):
            missing_samples = subset_result.get('missing_review_samples') or []
            sample_issues = subset_result.get('sample_issues') or []
            if not missing_samples and not sample_issues:
                continue

            candidate_entry: Dict[str, Any] = {
                'model': model_name,
                'task': subset_result.get('task'),
                'subset': subset_result.get('subset'),
                'expected_total': subset_result.get('expected_total'),
                'review_count': subset_result.get('review_count'),
                'successful_review_count': subset_result.get('successful_review_count'),
                'missing_review_count': len(missing_samples),
                'bad_review_count': len(sample_issues),
                'missing_review_samples': missing_samples,
                'bad_review_samples': sample_issues,
            }
            candidates.append(candidate_entry)
    return candidates


def print_summary(results: List[Dict[str, Any]]) -> None:
    total_subsets = 0
    bad_subsets = 0
    bad_samples = 0
    for model_result in results:
        print(f'Model: {model_result["model"]} status={model_result["status"]}')
        for subset_result in model_result['subsets']:
            total_subsets += 1
            sample_issue_count = len(subset_result['sample_issues'])
            if subset_result['status'] != 'ok':
                bad_subsets += 1
                bad_samples += sample_issue_count
                print(
                    f'  [BAD] {subset_result["task"]}@{subset_result["subset"]} '
                    f'expected={subset_result["expected_total"]} '
                    f'review={subset_result["review_count"]} '
                    f'successful_review={subset_result["successful_review_count"]}'
                )
                for issue in subset_result['issues'][:10]:
                    print(f'    - {issue}')
                for missing in subset_result['missing_review_samples'][:10]:
                    parts = [f'index {missing.get("index")}']
                    if missing.get('sample_key'):
                        parts.append(f'sample_key={missing["sample_key"]}')
                    judge_progress = missing.get('judge_progress')
                    if judge_progress:
                        parts.append(f'success_judges={judge_progress.get("total_success_judges")}')
                        parts.append(f'failed_judges={judge_progress.get("total_failed_judges")}')
                        parts.append(f'observed_judges={judge_progress.get("total_observed_judges")}')
                        stage_parts = []
                        for stage_name, stage_info in (judge_progress.get('stages') or {}).items():
                            stage_parts.append(
                                f'{stage_name}: success={stage_info.get("success_count")}, '
                                f'failed={stage_info.get("failed_count")}, '
                                f'missing={stage_info.get("missing_count")}, '
                                f'missing_judges={stage_info.get("missing_judges")}'
                            )
                        if stage_parts:
                            parts.append(' | '.join(stage_parts))
                    print(f'    - missing review sample: {"; ".join(parts)}')
                for sample_issue in subset_result['sample_issues'][:10]:
                    print(f'    - sample {sample_issue["index"]}: {"; ".join(sample_issue["issues"])}')
            else:
                print(
                    f'  [OK]  {subset_result["task"]}@{subset_result["subset"]} '
                    f'expected={subset_result["expected_total"]} '
                    f'review={subset_result["review_count"]} '
                    f'successful_review={subset_result["successful_review_count"]}'
                )
    print(f'\nSummary: bad_subsets={bad_subsets}/{total_subsets}, bad_samples={bad_samples}')


def main() -> int:
    parser = argparse.ArgumentParser(description='Audit fulltask prediction/review caches for 9 tasks.')
    parser.add_argument('run_dir', help='Fulltask run directory')
    parser.add_argument('--model', help='Only audit one model directory name')
    parser.add_argument('--json-out', help='Write full audit result to JSON file')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f'Run dir not found: {run_dir}')

    model_dirs = find_model_dirs(run_dir)
    if args.model:
        model_dirs = [p for p in model_dirs if p.name == args.model]
    if not model_dirs:
        raise SystemExit(f'No model dirs with predictions/reviews found in {run_dir}')

    results = [audit_model(run_dir, model_dir, verbose=args.verbose) for model_dir in model_dirs]
    print_summary(results)

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        payload = {
            'results': results,
            'resume_candidates': build_resume_candidates(results),
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Wrote JSON audit to {out_path}')

    return 0 if all(r['status'] == 'ok' for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
