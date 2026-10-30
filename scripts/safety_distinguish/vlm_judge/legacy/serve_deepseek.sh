#!/usr/bin/env bash
# Serve DeepSeek-VL2-small on the container's stable vLLM 0.11.2, GPUs 6-7.
#
# Why -small (not the base): deepseek-ai/deepseek-vl2 (base) ships a broken
# config.json on HF Hub - its `language_config` omits `num_hidden_layers`, so
# vLLM falls back to a default (32) that does not match the checkpoint's 30
# layers and aborts during weight load.  -small declares num_hidden_layers=27
# correctly, so it loads cleanly.
#
# Why --hf-overrides: VL2 nests architectures inside language_config; the
# top-level config has none, so vLLM still needs the explicit hint or it
# aborts with "No model architectures are specified".
set -u
LOG=/workspace/outputs/deepseek_vl2_small_server.log
mkdir -p /workspace/outputs

echo "launching DeepSeek-VL2-small server (TP=2, GPUs 6-7, port 8007)"
echo "log -> $LOG"
CUDA_VISIBLE_DEVICES=6,7 nohup python3 -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/deepseek-vl2-small \
  --served-model-name DeepSeek-VL2-small \
  --port 8007 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --trust-remote-code \
  --hf-overrides '{"architectures": ["DeepseekVLV2ForCausalLM"]}' \
  > "$LOG" 2>&1 &
echo "DeepSeek server PID: $!"
disown
echo "poll readiness with: curl -s localhost:8007/health"
