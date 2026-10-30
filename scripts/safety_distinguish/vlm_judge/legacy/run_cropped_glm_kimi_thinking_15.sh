#!/bin/bash
# Two-pass chain to fill the missing 1-5 cells in the 2x3 grid:
#   pass 1: GLM-4.6V-Flash      (cropped 2133, was missing entirely)
#   pass 2: Kimi-VL-A3B-Thinking (cropped 2133, currently 1061 only cm 0-15)
# Sequential because both want GPUs 6,7. Single vLLM cold-start per pass.
#
# Output goes alongside the other cropped 1-5 jsonls so plot_v5_band_variants_15
# picks it up unchanged.
set -u

CT=robotics-safety-container
HOST_OUT=~/workspace/robotics-safety/outputs/qual_fig_judgments/storyboard_overlay_cropped
mkdir -p "$HOST_OUT"

IMG_ROOT=/workspace/storyboard_overlay_cropped_final
SCRIPT=/workspace/scripts/safety_recognition/run_safety_judge_fpanel.py
PROMPT_15=/workspace/scripts/safety_recognition/prompts/safety_prompt.txt
TARGET=2133
STALL_MAX=1200  # 20 min — Kimi-Thinking is slow due to reasoning traces

teardown_vllm() {
  # comm-name pkill alone misses VLLM::EngineCore (PR_SET_NAME); also kill any
  # vllm.entrypoints process tree and (as fallback) any python on GPUs 6,7.
  docker exec $CT bash -lc "
    pkill -9 -f 'vllm.entrypoints.openai' 2>/dev/null
    pkill -9 'VLLM::Worker' 2>/dev/null
    pkill -9 'VLLM::EngineCo' 2>/dev/null
    pkill -9 'VLLM::EngineC' 2>/dev/null
    for pid in \$(ps -ef | awk '/VLLM::|vllm.entrypoints/ && !/awk/ {print \$2}'); do
      kill -9 \$pid 2>/dev/null
    done
  " || true
  for w in $(seq 1 30); do
    local g67=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 2>/dev/null | awk '{s+=$1} END {print s}')
    if [[ -n "$g67" ]] && [[ "$g67" -lt 2048 ]]; then
      echo "[$(date +%T)] GPUs 6,7 cleared (used=${g67} MiB after ${w}x2s)"
      return 0
    fi
    sleep 2
  done
  echo "[$(date +%T)] WARN GPUs 6,7 not clean - continuing"
}

wait_jsonl_done() {
  local jpath="$1" target="$2" label="$3" last=-1 stable=0
  while :; do
    local n=$(wc -l < "$jpath" 2>/dev/null || echo 0)
    if [[ "$n" -ge "$target" ]]; then echo "[$(date +%T)] $label DONE (n=$n)"; return 0; fi
    if [[ "$n" -eq "$last" ]]; then
      stable=$((stable + 60))
      if [[ "$stable" -ge "$STALL_MAX" ]]; then
        echo "[$(date +%T)] $label STALLED at n=$n"
        return 1
      fi
    else
      stable=0; last=$n
      echo "[$(date +%T)] $label progress: n=$n / $target"
    fi
    sleep 60
  done
}

run_pass() {
  local label="$1" model="$2" port="$3" jsonl_name="$4" max_tok="$5"
  local out_path="/workspace/outputs/qual_fig_judgments/storyboard_overlay_cropped/$jsonl_name"
  local host_out_path="$HOST_OUT/$jsonl_name"

  echo "=================================================="
  echo "[$(date +%T)] BEGIN pass $label ($model)"
  echo "=================================================="

  if [[ -f "$host_out_path" ]] && [[ "$(wc -l < $host_out_path)" -ge "$TARGET" ]]; then
    echo "[$(date +%T)] $label already complete ($(wc -l < $host_out_path)/$TARGET) - skip"
    return 0
  fi

  teardown_vllm

  local log_vllm="/workspace/outputs/vllm_cropped_${label}.log"
  echo "[$(date +%T)] starting vLLM ($model on :$port GPUs 6,7) ..."
  docker exec -d $CT bash -lc \
    "CUDA_VISIBLE_DEVICES=6,7 python3 -m vllm.entrypoints.openai.api_server \
      --model '$model' --port $port --tensor-parallel-size 2 \
      --gpu-memory-utilization 0.88 --max-model-len 8192 --trust-remote-code \
      > $log_vllm 2>&1"

  local ready=0
  for i in $(seq 1 48); do
    if docker exec $CT bash -lc "curl -fsS -m 3 http://localhost:$port/health > /dev/null 2>&1"; then
      echo "[$(date +%T)] :$port healthy after ${i}x15s"; ready=1; break
    fi
    sleep 15
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "[$(date +%T)] FATAL: :$port never came up - dumping vLLM tail"
    docker exec $CT bash -lc "tail -80 $log_vllm" 2>&1 | tail -80
    teardown_vllm
    return 1
  fi

  local log_judge="/workspace/outputs/sbov_cropped_${label}_v5.log"
  echo "[$(date +%T)] launching judge ..."
  docker exec -d $CT bash -lc \
    "python3 $SCRIPT --img-root $IMG_ROOT --prompt $PROMPT_15 --model '$model' \
      --base-url http://localhost:$port/v1 --output $out_path \
      --workers 4 --scale 1-5 --max-tokens $max_tok --resume \
      > $log_judge 2>&1"

  if wait_jsonl_done "$host_out_path" "$TARGET" "$label"; then
    echo "[$(date +%T)] $label OK ($(wc -l < $host_out_path)/$TARGET)"
  else
    echo "[$(date +%T)] $label STALLED — moving on"
  fi
  teardown_vllm
}

# Pass 1 — GLM-4.6V-Flash (smaller model, ~5-7 min judge)
run_pass "glm-4.6v-flash" \
         "zai-org/GLM-4.6V-Flash" \
         8016 \
         "judgments_glm-4.6v-flash_v5.jsonl" \
         512

# Pass 2 — Kimi-VL-A3B-Thinking-2506 (reasoning trace, max_tokens=1536)
run_pass "kimi-vl-thinking" \
         "moonshotai/Kimi-VL-A3B-Thinking-2506" \
         8012 \
         "judgments_kimi-vl-thinking_v5.jsonl" \
         1536

echo "=================================================="
echo "[$(date +%T)] all passes complete - re-baking plots"
echo "=================================================="
docker exec $CT bash -lc "python3 /workspace/scripts/qual_fig/plot_v5_band_variants_15.py" 2>&1 | tail -20
echo "[$(date +%T)] DONE"
