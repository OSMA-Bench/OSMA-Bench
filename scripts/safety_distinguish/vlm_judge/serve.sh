#!/bin/bash
# Parameterized vLLM server launcher for the judge models (detached, inside
# the robocasa container). Model-specific quirks live here so the registry
# stays declarative.
#
#   ./serve.sh deepseek-vl2 --gpus 6,7
#   ./serve.sh qwen3-vl-32b --gpus 0,1 --port 8011
set -u
cd "$(dirname "$0")"
source ./models.sh

CT=${CT:-robotics-safety-container}
GPUS=""

usage() {
  echo "usage: $0 MODEL_KEY [--gpus 0,1] [--port N]"
  echo "model keys:"; model_keys | sed 's/^/  /'
  exit 2
}

[[ $# -lt 1 ]] && usage
KEY=$1; shift
resolve_judge "$KEY" || exit 2     # MODEL_ID + default PORT
resolve_serve "$KEY" || exit 2     # TP MAX_LEN GPU_UTIL

while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus) GPUS=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    *) echo "unknown flag $1"; usage ;;
  esac
done

ENV_PREFIX=""
VLLM_BIN="vllm"
EXTRA=""
case $KEY in
  glm-4.6v-fp8)
    # needs the dedicated conda env (vllm 0.21 / transformers 5.9); the cu129
    # wheels ship no nvcc so FlashInfer's sampler cannot JIT — force native.
    VLLM_BIN="/opt/conda/envs/glm/bin/vllm"
    ENV_PREFIX="VLLM_USE_FLASHINFER_SAMPLER=0"
    EXTRA="--served-model-name GLM-4.6V"
    ;;
  deepseek-vl2)
    # VL2 nests architectures inside language_config; vLLM needs the hint.
    EXTRA="--hf-overrides {\"architectures\":[\"DeepseekVLV2ForCausalLM\"]}"
    ;;
  kimi-thinking)
    # TP=2 stalls forever after "MoE shared_experts" — keep TP=1.
    TP=1
    ;;
esac

[[ -n "$GPUS" ]] && ENV_PREFIX="CUDA_VISIBLE_DEVICES=$GPUS $ENV_PREFIX"
LOG=/workspace/outputs/vllm_${KEY}_${PORT}.log

echo "[$(date +%T)] serving $KEY = $MODEL_ID (:$PORT, TP=$TP, len $MAX_LEN, gpus '${GPUS:-inherit}')"
echo "  log -> $LOG"
docker exec -d "$CT" bash -lc \
  "$ENV_PREFIX $VLLM_BIN serve '$MODEL_ID' --port $PORT \
    --tensor-parallel-size $TP --gpu-memory-utilization $GPU_UTIL \
    --max-model-len $MAX_LEN --trust-remote-code $EXTRA > $LOG 2>&1"

echo "[$(date +%T)] polling readiness (up to 12 min)..."
for i in $(seq 1 48); do
  if docker exec "$CT" bash -lc "curl -fsS -m 3 http://localhost:$PORT/health > /dev/null 2>&1"; then
    echo "[$(date +%T)] :$PORT healthy after ${i}x15s"
    exit 0
  fi
  sleep 15
done
echo "[$(date +%T)] FATAL: :$PORT never came up — vLLM log tail:"
docker exec "$CT" bash -lc "tail -40 $LOG" || true
exit 1
