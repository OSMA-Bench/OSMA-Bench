#!/bin/bash
# Run 1-5 (Likert) safety-judge on storyboard_overlay_nkf10 for Qwen3-VL-30B-A3B
# only — fills in the missing cm60-80 1-5 bin so plot_v5_band_variants_15.py can
# render the full 0-80 cm range apples-to-apples (same v5 prompt as cropped).
#
# Mirrors run_nkf10_binary_chain.sh structure but:
#   - prompt = safety_prompt_v5.txt        (1-5 rubric, not binary)
#   - --scale 1-5
#   - output drops the "binary_" infix
#   - SCRIPT path corrected (lives under scripts/safety_recognition/, not
#     scripts/qual_fig/ — the binary chain has a stale path that would fail)
#   - one model only (no MODELS loop)
set -u

CT=robotics-safety-container
HOST_WS=~/workspace/robotics-safety
IMG_ROOT=/workspace/storyboard_overlay_nkf10
OUT_DIR=/workspace/outputs/qual_fig_judgments/storyboard_overlay_nkf10
HOST_OUT_DIR=$HOST_WS/outputs/qual_fig_judgments/storyboard_overlay_nkf10
SCRIPT=/workspace/scripts/safety_recognition/run_safety_judge_fpanel.py
PROMPT=/workspace/scripts/safety_recognition/prompts/safety_prompt.txt
TARGET=387          # total PNGs in nkf10 (matches binary chain)
STALL_MAX=900       # 15 min of zero growth -> abort

MODEL_ID="Qwen/Qwen3-VL-30B-A3B-Instruct"
PORT=8013
MAX_TOKENS=256
SUFFIX="_qwen3-vl-30b-a3b_v5"      # judgments${SUFFIX}.jsonl
OUT_BIN="$OUT_DIR/judgments${SUFFIX}.jsonl"
HOST_BIN="$HOST_OUT_DIR/judgments${SUFFIX}.jsonl"

mkdir -p "$HOST_OUT_DIR"

teardown_vllm() {
  # VLLM::Worker / VLLM::EngineCore are prctl(PR_SET_NAME) comm names, not
  # cmdline strings — pkill -f misses them. See run_nkf10_binary_chain.sh.
  docker exec $CT bash -lc "pkill -9 -f 'vllm.entrypoints.openai' 2>/dev/null; pkill -9 'VLLM::Worker' 2>/dev/null; pkill -9 'VLLM::EngineCore' 2>/dev/null" || true
  for w in $(seq 1 30); do
    local g67=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 2>/dev/null | awk '{s+=$1} END {print s}')
    if [[ -n "$g67" ]] && [[ "$g67" -lt 2048 ]]; then
      echo "[$(date +%T)] GPUs 6,7 cleared after ${w}x2s (used=${g67} MiB)"
      return 0
    fi
    if [[ "$w" -eq 15 ]]; then
      echo "[$(date +%T)] GPUs 6,7 still at ${g67} MiB - second sweep"
      docker exec $CT bash -lc "pkill -9 'VLLM::Worker' 2>/dev/null; pkill -9 'VLLM::EngineCore' 2>/dev/null; pkill -9 -f 'vllm.entrypoints.openai' 2>/dev/null" || true
    fi
    sleep 2
  done
  echo "[$(date +%T)] WARN: GPUs 6,7 not clean after 60s - proceeding anyway"
}

wait_jsonl_done() {
  local jpath="$1" label="$2"
  local last=-1 stable=0
  while :; do
    local n=$(wc -l < "$jpath" 2>/dev/null || echo 0)
    if [[ "$n" -ge "$TARGET" ]]; then
      echo "[$(date +%T)] $label DONE (n=$n)"; return 0
    fi
    if [[ "$n" -eq "$last" ]]; then
      stable=$((stable + 60))
      if [[ "$stable" -ge "$STALL_MAX" ]]; then
        echo "[$(date +%T)] $label STALLED at n=$n (no growth ${STALL_MAX}s)"
        return 1
      fi
    else
      stable=0; last=$n
      echo "[$(date +%T)] $label progress: n=$n / $TARGET"
    fi
    sleep 60
  done
}

echo "=================================================================="
echo "[$(date +%T)] BEGIN qwen30b-a3b 1-5 v5 ($MODEL_ID) -> $OUT_BIN"
echo "=================================================================="

if [[ -f "$HOST_BIN" ]] && [[ "$(wc -l < "$HOST_BIN")" -ge "$TARGET" ]]; then
  echo "[$(date +%T)] already complete ($(wc -l < "$HOST_BIN")/$TARGET) - skip judge, just re-bake"
else
  teardown_vllm
  LOG_VLLM="/workspace/outputs/vllm_nkf10${SUFFIX}.log"
  echo "[$(date +%T)] starting vLLM (GPUs 6,7, port $PORT) ..."
  docker exec -d $CT bash -lc "CUDA_VISIBLE_DEVICES=6,7 python3 -m vllm.entrypoints.openai.api_server --model '$MODEL_ID' --port $PORT --tensor-parallel-size 2 --gpu-memory-utilization 0.88 --max-model-len 8192 --trust-remote-code > $LOG_VLLM 2>&1"

  ready=0
  for i in $(seq 1 48); do
    if docker exec $CT bash -lc "curl -fsS -m 3 http://localhost:$PORT/health > /dev/null 2>&1"; then
      echo "[$(date +%T)] :$PORT healthy after ${i}x15s"; ready=1; break
    fi
    sleep 15
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "[$(date +%T)] FATAL: :$PORT never came up"
    docker exec $CT bash -lc "tail -100 $LOG_VLLM" 2>&1 | tail -100
    teardown_vllm
    exit 1
  fi

  LOG_JUDGE="/workspace/outputs/sbov_nkf10${SUFFIX}.log"
  echo "[$(date +%T)] launching judge..."
  docker exec -d $CT bash -lc \
"python3 $SCRIPT --img-root $IMG_ROOT --prompt $PROMPT --model '$MODEL_ID' \
  --base-url http://localhost:$PORT/v1 --output $OUT_BIN \
  --workers 8 --scale 1-5 --max-tokens $MAX_TOKENS --resume \
  > $LOG_JUDGE 2>&1"

  if wait_jsonl_done "$HOST_BIN" "$(basename $OUT_BIN)"; then
    echo "[$(date +%T)] judge OK ($(wc -l < $HOST_BIN)/$TARGET)"
  else
    echo "[$(date +%T)] judge stalled - moving on"
  fi
  teardown_vllm
fi

echo "[$(date +%T)] re-baking plot_v5_band_variants_15.py ..."
docker exec $CT bash -lc "python3 /workspace/scripts/qual_fig/plot_v5_band_variants_15.py" 2>&1 | tail -20

echo "=================================================================="
echo "[$(date +%T)] DONE"
echo "=================================================================="
