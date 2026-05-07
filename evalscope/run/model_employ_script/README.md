Model deployment scripts (vLLM)

This folder contains vLLM OpenAI-compatible server launchers for common models.
Each script is self-contained and follows the same `serve_vllm.sh` template.

Quick start
- Run one script directly: `bash serve_vllm_qwen2.5_7b.sh`
- Override settings: `PORT=9001 TENSOR_PARALLEL_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash serve_vllm_qwen3_32b.sh`

Common environment variables
- MODEL_PATH: local model directory (required by each script)
- SERVED_MODEL_NAME: model id exposed by the server (must match request "model")
- HOST, PORT
- GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN, TENSOR_PARALLEL_SIZE
- MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS, DTYPE
- VLLM_ARGS: extra vLLM args (space-separated)

Scripts and defaults
- serve_vllm_deepseek_r1.sh: /inspire/hdd/project/ai4education/public/Models/deepseek-ai/DeepSeek-R1-0528 (port 8011)
- serve_vllm_deepseek_v3_2.sh: /inspire/hdd/project/ai4education/public/Models/deepseek-ai/DeepSeek-V3.2 (port 8012)
- serve_vllm_qwen3_235b_a22b.sh: /inspire/hdd/project/ai4education/public/Models/Qwen/Qwen3-235B-A22B (port 8013)
- serve_vllm_qwen3_32b.sh: /inspire/hdd/project/ai4education/public/Models/Qwen/Qwen3-32B (port 8014)
- serve_vllm_qwen3_5_122b_a10b.sh: /inspire/hdd/project/ai4education/public/Models/Qwen/Qwen3.5-122B-A10B (port 8038)
- serve_vllm_qwen3_5_35b_a3b.sh: /inspire/hdd/project/ai4education/public/Models/Qwen/Qwen3.5-35B-A3B (port 8037)
- serve_vllm_qwen3_5_27b.sh: /inspire/hdd/project/ai4education/public/Models/Qwen/Qwen3.5-27B (port 8036)
- serve_vllm_qwen3_5_9b.sh: /inspire/hdd/project/ai4education/public/Models/Qwen/Qwen3.5-9B (port 8035)
- serve_vllm_qwen3_5_4b.sh: /inspire/hdd/project/ai4education/public/Models/Qwen/Qwen3.5-4B (port 8039)
- serve_vllm_llama3_3_70b.sh: /inspire/hdd/project/ai4education/public/Models/Llama/Llama-3.3-70B-Instruct (port 8015)
- serve_vllm_qwen2_5_72b.sh: /inspire/hdd/project/ai4education/public/Models/Qwen/Qwen2.5-72B-Instruct (port 8016)
- serve_vllm_mixtral_8x7b.sh: /inspire/hdd/project/ai4education/public/Models/Mistral/Mixtral-8x7B-Instruct-v0.1 (port 8017)
- serve_vllm_qwen3_14b.sh: /inspire/hdd/project/ai4education/public/Models/Qwen/Qwen3-14B (port 8021)
- serve_vllm_qwen2_5_32b.sh: /inspire/hdd/project/ai4education/public/Models/Qwen/Qwen2.5-32B (port 8022)
- serve_vllm_llama3_2_3b.sh: /inspire/hdd/project/ai4education/public/Models/Llama/Llama3.2-3B-Instruct (port 8025)
- serve_vllm_qwen2_5_14b.sh: /inspire/hdd/project/ai4education/public/Models/Qwen/Qwen2.5-14B (port 8026)
- serve_vllm_mistral_24b.sh: /inspire/hdd/project/ai4education/public/Models/Mistral/Mistral-24B-Instruct (port 8023)
- serve_vllm_olmo2_13b.sh: /inspire/hdd/project/ai4education/public/Models/OLMo/OLMo-2-13B-Instruct (port 8024)
- serve_vllm_qwen3_8b.sh: /inspire/hdd/project/ai4education/public/Models/Qwen/Qwen3-8B (port 8031)
- serve_vllm_llama3_1_8b.sh: /inspire/hdd/project/ai4education/public/Models/Llama/Llama-3.1-8B-Instruct (port 8032)
- serve_vllm_qwen2.5_7b.sh: /inspire/hdd/project/ai4education/public/Models/Qwen/Qwen2.5-7B-Instruct (port 8007)
- serve_vllm_qwen2.5_7b_lora.sh: /inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/diversitytuning_creative/DiversityTuning/checkpoints_merged/Qwen2.5-7B-Instruct_SFT_AUT_full20_allcombo_gap030_deepseekdedup_epoch-final_merged (port 8008)
- serve_vllm_qwen2.5_7b_online_dpo_selected.sh: /inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/diversitytuning_creative/DiversityTuning/checkpoints_merged/Qwen2.5-7B-Instruct_OnlineDPO_AUT_full20_allcombo_gap030_deepseekdedup_checkpoint-284_selected_merged (port 8009)
- serve_vllm_qwen2.5_7b_online_ddpo_selected.sh: /inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/diversitytuning_creative/DiversityTuning/checkpoints_merged/Qwen2.5-7B-Instruct_OnlineDDPO_AUT_full20_allcombo_gap030_deepseekdedup_checkpoint-568_selected_merged (port 8010)
- serve_vllm_qwen2.5_7b_offline_dpo.sh: /inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/diversitytuning_creative/DiversityTuning/checkpoints_merged/Qwen2.5-7B-Instruct_DPO_AUT_full20_allcombo_gap030_deepseekdedup_20260322_160529_checkpoint-final_merged (port 8018)
- serve_vllm_qwen2.5_7b_offline_ddpo.sh: /inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/diversitytuning_creative/DiversityTuning/checkpoints_merged/Qwen2.5-7B-Instruct_DDPO_AUT_full20_allcombo_gap030_deepseekdedup_20260322_160507_checkpoint-final_merged (port 8019)
- serve_vllm_mistral_7b.sh: /inspire/hdd/project/ai4education/public/Models/Mistral/Mistral-7B-Instruct (port 8033)
- serve_vllm_olmo2_7b.sh: /inspire/hdd/project/ai4education/public/Models/OLMo/OLMo-2-7B-Instruct (port 8034)

Notes
- Some defaults (Mixtral/Mistral/OLMo) are placeholders. If the directory does not exist,
  set MODEL_PATH to the correct location before launching.
- For large models, set CUDA_VISIBLE_DEVICES and TENSOR_PARALLEL_SIZE to match your GPU setup.
- `serve_vllm.sh` is a generic template; you can copy it to create new model scripts.
- Meta's official Llama 3.2 text models are 1B and 3B; there is no official 7B text release in this series.
