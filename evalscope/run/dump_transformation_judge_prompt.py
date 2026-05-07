#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.benchmarks.transformation.transformation_adapter import TransformationAdapter


DEFAULT_RUN = Path(
    '/inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/outputs/transformation/'
    'transformation_review_only_limit5_novelty3_recalib_smoke_20260425'
)


def _load_json_records(path: Path):
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, list):
        raise ValueError(f'Expected list dataset: {path}')
    return payload


def _load_prediction(path: Path, sample_id: int) -> str:
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row.get('sample_id', row.get('index'))
            if isinstance(row_id, str) and row_id.isdigit():
                row_id = int(row_id)
            if row_id == sample_id:
                return str(row.get('prediction') or row.get('response') or row.get('output') or '')
    raise ValueError(f'No prediction sample_id={sample_id} in {path}')


def main() -> int:
    parser = argparse.ArgumentParser(description='Dump the rendered transformation judge prompt for one sample.')
    parser.add_argument('--run-dir', default=str(DEFAULT_RUN), help='Review/output run directory containing predictions.')
    parser.add_argument('--model', default='gpt-5.2', help='Model directory name under --run-dir.')
    parser.add_argument('--sample-id', type=int, default=0, help='Dataset sample id to render.')
    parser.add_argument('--out', default='', help='Optional output markdown path.')
    args = parser.parse_args()

    adapter = TransformationAdapter(_task_config=SimpleNamespace(work_dir=None, model_id=args.model))
    records = _load_json_records(adapter.dataset_path)
    item = records[args.sample_id]

    pred_path = (
        Path(args.run_dir).expanduser().resolve()
        / args.model
        / 'predictions'
        / args.model
        / 'transformation_default.jsonl'
    )
    prediction = _load_prediction(pred_path, args.sample_id)
    answer_text = adapter.extract_answer(prediction, SimpleNamespace(metadata={'item': item}))
    messages = adapter._build_judge_messages(item, answer_text)

    system = next((m.content for m in messages if isinstance(m, ChatMessageSystem)), '')
    user = next((m.content for m in messages if isinstance(m, ChatMessageUser)), '')

    rendered = (
        f'# Transformation Judge Prompt\n\n'
        f'- run_dir: `{Path(args.run_dir).expanduser().resolve()}`\n'
        f'- model: `{args.model}`\n'
        f'- sample_id: `{args.sample_id}`\n\n'
        f'## System Message\n\n'
        f'```text\n{system}\n```\n\n'
        f'## User Message\n\n'
        f'```text\n{user}\n```\n'
    )

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding='utf-8')
        print(out_path)
    else:
        print(rendered)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
