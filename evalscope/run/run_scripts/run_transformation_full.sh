#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

python run/run_suite.py \
  --suite transformation \
  --models-json run/models.json \
  --limit none \
  "$@"
