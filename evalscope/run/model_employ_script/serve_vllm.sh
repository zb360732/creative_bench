CUDA_VISIBLE_DEVICES=3  \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m vllm.entrypoints.openai.api_server \
  --model /inspire/hdd/project/ai4education/public/Models/Qwen/Qwen2.5-3B-Instruct   \
  --served-model-name  qwen3-4b-Instruct \
  --host 0.0.0.0 \
  --port 8009 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 32768 \
  --tensor-parallel-size 1\
  --max-num-seqs 32 \
  --max-num-batched-tokens 4096\
  --disable-custom-all-reduce
