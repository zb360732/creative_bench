# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from evalscope.utils.logger import get_logger

logger = get_logger()


class BenchmarkJudgeCache:
    """Append-only per-judge cache with latest-write-wins semantics."""

    def __init__(self, benchmark_name: str, work_dir: Optional[str], model_name: Optional[str] = None):
        self.benchmark_name = benchmark_name
        self.work_dir = Path(work_dir).expanduser().resolve() if work_dir else None
        self.model_name = str(model_name or 'default')
        self._lock = threading.RLock()
        self._loaded = False
        self._records: Dict[str, Dict[str, Any]] = {}

    @property
    def enabled(self) -> bool:
        return self.work_dir is not None

    @property
    def cache_file(self) -> Optional[Path]:
        if not self.work_dir:
            return None
        return self.work_dir / 'judge_cache' / self.model_name / f'{self.benchmark_name}.jsonl'

    def _record_key(self, sample_key: str, stage: str, judge_key: str) -> str:
        return f'{sample_key}\t{stage}\t{judge_key}'

    def _iter_records(self, text: str):
        decoder = json.JSONDecoder()
        idx = 0
        length = len(text)
        while idx < length:
            while idx < length and text[idx].isspace():
                idx += 1
            if idx >= length:
                break
            record, next_idx = decoder.raw_decode(text, idx)
            yield record
            idx = next_idx
            while idx < length and text[idx].isspace():
                idx += 1
            if text.startswith('\\n', idx):
                idx += 2

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        cache_file = self.cache_file
        if not cache_file or not cache_file.exists():
            return
        try:
            text = cache_file.read_text(encoding='utf-8')
            for record in self._iter_records(text):
                if not isinstance(record, dict):
                    continue
                sample_key = str(record.get('sample_key', ''))
                stage = str(record.get('stage', ''))
                judge_key = str(record.get('judge_key', ''))
                if not sample_key or not stage or not judge_key:
                    continue
                self._records[self._record_key(sample_key, stage, judge_key)] = record
        except Exception as exc:
            logger.warning('Failed to load judge cache from %s: %s', cache_file, exc)

    def get(self, sample_key: str, stage: str, judge_key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        with self._lock:
            self._load()
            record = self._records.get(self._record_key(sample_key, stage, judge_key))
            if record is None:
                return None
            return dict(record)

    def put(self, sample_key: str, stage: str, judge_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return dict(payload)
        with self._lock:
            self._load()
            cache_file = self.cache_file
            assert cache_file is not None
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            record = dict(payload)
            record.update({
                'sample_key': str(sample_key),
                'stage': str(stage),
                'judge_key': str(judge_key),
                'updated_at': time.time(),
            })
            with cache_file.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            self._records[self._record_key(str(sample_key), str(stage), str(judge_key))] = record
            return dict(record)
