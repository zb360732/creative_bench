#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

repo_root = Path('/inspire/hdd/project/ai4education/qianhong-p-qianhong')
sys.path.insert(0, str(repo_root / 'benchmark' / 'evalscope'))

from evalscope.config import TaskConfig
from evalscope.run import run_task

models_path = repo_root / 'benchmark' / 'evalscope' / 'run' / 'models.json'
models = json.loads(models_path.read_text(encoding='utf-8')).get('models', [])
entry = None
for item in models:
    if str(item.get('name', '')).strip() == 'qwen2.5-7b':
        entry = item
        break
if entry is None:
    raise SystemExit('qwen2.5-7b not found in models.json')

model = os.path.expandvars(str(entry.get('model')))
api_url = os.path.expandvars(str(entry.get('api_url')))
api_key = str(entry.get('api_key', 'EMPTY'))
api_key_env = entry.get('api_key_env')
if api_key_env:
    api_key = os.getenv(str(api_key_env), api_key)
if api_key in {'', 'YOUR_API_KEY'}:
    api_key = os.getenv('EVALSCOPE_API_KEY', api_key)
if api_key in {'', 'YOUR_API_KEY'}:
    api_key = os.getenv('OPENAI_API_KEY', api_key)

host = urlparse(api_url).hostname
if host:
    for env_key in ('NO_PROXY', 'no_proxy'):
        current = os.environ.get(env_key, '')
        parts = [p.strip() for p in current.split(',') if p.strip()]
        if host not in parts:
            parts.append(host)
        for local in ('localhost', '127.0.0.1'):
            if local not in parts:
                parts.append(local)
        os.environ[env_key] = ','.join(parts)

work_dir = repo_root / 'benchmark' / 'outputs' / 'test' / 'aut_multiround_qwen2.5-7b_full'
work_dir.mkdir(parents=True, exist_ok=True)

cfg = TaskConfig(
    model=model,
    model_id='qwen2.5-7b',
    api_url=api_url,
    api_key=api_key,
    datasets=['aut'],
    limit=None,
    work_dir=str(work_dir),
    no_timestamp=True,
    seed=42,
    generation_config={
        'max_tokens': 2048,
        'temperature': 0.0,
    },
)

run_task(cfg)
print(f'[OK] AUT multi-round full run done in {work_dir}')
PY
