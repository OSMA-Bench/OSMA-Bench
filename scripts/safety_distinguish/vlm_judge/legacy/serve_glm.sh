#!/usr/bin/env bash
# Serve GLM-4.6V-FP8 from the dedicated conda env (vllm 0.21.0 / transformers
# 5.9.0) on GPUs 4-7.  Detached: returns immediately, server logs to a file.
#
# VLLM_USE_FLASHINFER_SAMPLER=0 : the cu129 wheels ship no nvcc, so FlashInfer's
# sampler cannot JIT-compile its CUDA kernel ("Could not find nvcc").  Force the
# PyTorch-native sampler instead; the judge runs greedy (temperature=0) so the
# sampler choice does not affect results.
set -u
LOG=/workspace/outputs/glm_server.log
mkdir -p /workspace/outputs

echo "launching GLM-4.6V-FP8 server (TP=4, GPUs 4-7, port 8006)"
echo "log -> $LOG"
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_USE_FLASHINFER_SAMPLER=0 nohup /opt/conda/envs/glm/bin/vllm serve \
  zai-org/GLM-4.6V-FP8 \
  --served-model-name GLM-4.6V \
  --port 8006 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 8192 \
  --trust-remote-code \
  > "$LOG" 2>&1 &
echo "GLM server PID: $!"
disown
echo "poll readiness with: curl -s localhost:8006/health"
