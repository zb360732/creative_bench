#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/inspire/hdd/project/ai4education/qianhong-p-qianhong}"
MODEL_ROOT="${MODEL_ROOT:-/inspire/hdd/project/ai4education/public/Models/Qwen}"
SEARCH_ROOTS=(
  "$REPO_ROOT"
  "$MODEL_ROOT"
  "$HOME"
  /root
  /tmp
)

if command -v rg >/dev/null 2>&1; then
  SEARCH_BIN=(rg -n --no-messages)
else
  SEARCH_BIN=(grep -RIn)
fi

section() {
  printf '\n===== %s =====\n' "$1"
}

show_file_head() {
  local file="$1"
  if [[ -f "$file" ]]; then
    echo "--- $file ---"
    sed -n '1,120p' "$file"
  fi
}

section "Process"
ps -efww | grep -E 'qwen2\.5-32b|Qwen2\.5-32B|vllm\.entrypoints\.openai\.api_server|--port 8022' | grep -v grep || true

section "Repo References"
"${SEARCH_BIN[@]}" 'qwen2\.5-32b|Qwen2\.5-32B|Qwen2\.5-32B-Instruct' \
  "$REPO_ROOT/benchmark/evalscope/run/model_employ_script" \
  "$REPO_ROOT/benchmark/evalscope/run/models.json" \
  "$REPO_ROOT/CoELA/cwah2/testing_agents/models.json" || true

section "Shell History"
for hist in "$HOME/.bash_history" /root/.bash_history; do
  if [[ -f "$hist" ]]; then
    echo "--- $hist ---"
    grep -En 'Qwen2\.5-32B|Qwen2\.5-32B-Instruct|qwen2\.5-32b|huggingface-cli download|hf download|modelscope download|vllm' "$hist" | tail -n 80 || true
  fi
done

section "Recent Logs"
find /tmp "$REPO_ROOT" -type f \
  \( -name '*.log' -o -name '*.out' -o -name '*.txt' \) \
  2>/dev/null | grep -E 'qwen|8022|vllm|serve' | sort | tail -n 80 || true

section "Model Dirs"
find "$MODEL_ROOT" -maxdepth 2 -type d | grep -E 'Qwen2\.5-32B|Qwen2\.5-32B-Instruct' | sort || true

for model_dir in \
  "$MODEL_ROOT/Qwen2.5-32B" \
  "$MODEL_ROOT/Qwen2.5-32B-Instruct"; do
  if [[ -d "$model_dir" ]]; then
    section "Model Metadata: $model_dir"
    du -sh "$model_dir" || true
    ls -lah "$model_dir" | sed -n '1,40p'
    show_file_head "$model_dir/config.json"
    show_file_head "$model_dir/generation_config.json"
    show_file_head "$model_dir/tokenizer_config.json"
    if [[ -d "$model_dir/.cache" ]]; then
      find "$model_dir/.cache" -maxdepth 3 -type f | sed -n '1,40p'
    fi
  fi
done

section "Download Traces"
"${SEARCH_BIN[@]}" 'Qwen2\.5-32B-Instruct|Qwen2\.5-32B|huggingface-cli download|snapshot_download' \
  /root "$HOME" "$REPO_ROOT" /tmp 2>/dev/null | tail -n 200 || true

section "Serve Script"
show_file_head "$REPO_ROOT/benchmark/evalscope/run/model_employ_script/serve_vllm_qwen2_5_32b.sh"

section "Models JSON"
show_file_head "$REPO_ROOT/benchmark/evalscope/run/models.json"

section "Done"
echo "If you see only Qwen2.5-32B and not Qwen2.5-32B-Instruct, then the deployed 32B is the base model."
