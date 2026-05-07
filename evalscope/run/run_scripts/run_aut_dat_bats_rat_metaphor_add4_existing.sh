#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"

run_dir="${RUN_DIR:-${repo_root}/benchmark/outputs/exploration/aut_creative_math_drivel_neocoder_full}"
models="${MODELS:-llama3.1-8b,mistral-7b,mistral-24b,qwen2.5-32b}"
run_log="${RUN_LOG:-${run_dir}/run_add4_detailed.log}"
wrap_log="${WRAP_LOG:-${run_dir}/run_exploration_add4_$(date -u +%Y%m%d_%H%M%S).wrapper.log}"
backup_summary="${BACKUP_SUMMARY:-}"

mkdir -p "${run_dir}"

if [[ -z "${backup_summary}" ]]; then
  latest_backup="$(find "${run_dir}" -maxdepth 1 -type f -name 'summary.before_add4_*.json' | sort | tail -n 1 || true)"
  if [[ -n "${latest_backup}" ]]; then
    backup_summary="${latest_backup}"
  else
    backup_summary="${run_dir}/summary.before_add4_$(date -u +%Y%m%d_%H%M%S).json"
    cp "${run_dir}/summary.json" "${backup_summary}"
  fi
fi

nohup setsid bash -lc "
  set -euo pipefail
  cd '${repo_root}'
  unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
  export NO_PROXY='localhost,127.0.0.1,notebook-inspire.sii.edu.cn,ai-notebook-inspire.sii.edu.cn'
  export no_proxy=\"\${NO_PROXY}\"
  python benchmark/evalscope/run/run_parallel_eval.py \
    --models-json benchmark/evalscope/run/models.json \
    --models '${models}' \
    --datasets aut,creative_math,drivel_writing,neocoder \
    --limit none \
    --max-tokens 30000 \
    --temperature 0.0 \
    --eval-batch-size 4 \
    --max-parallel 4 \
    --work-dir benchmark/outputs/exploration \
    --run-name aut_creative_math_drivel_neocoder_full \
    --log-file '${run_log}'
  python - <<'PY'
import json
from pathlib import Path

run_dir = Path(r'''${run_dir}''')
backup = Path(r'''${backup_summary}''')
summary = run_dir / 'summary.json'

old = json.loads(backup.read_text(encoding='utf-8')) if backup.exists() else {}
new = json.loads(summary.read_text(encoding='utf-8')) if summary.exists() else {}
old.update(new)
summary.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'[OK] merged summary -> {summary}')
PY
  python benchmark/evalscope/run/summarize_reports.py \
    --run-dir benchmark/outputs/exploration/aut_creative_math_drivel_neocoder_full \
    --out-name scores_summary
  python benchmark/evalscope/run/make_score_matrix.py \
    --run-dir benchmark/outputs/exploration/aut_creative_math_drivel_neocoder_full \
    --out-name score_matrix
" </dev/null >"${wrap_log}" 2>&1 &

echo "$! ${wrap_log}"
