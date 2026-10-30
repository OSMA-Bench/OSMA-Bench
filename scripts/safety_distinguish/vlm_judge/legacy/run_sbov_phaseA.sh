#!/bin/bash
# Phase A storyboard_overlay watcher:
#   1. Polls /health on ports 8005-8008 until all 4 models are READY.
#   2. Launches 8 judges (4 models × 2 modes) in parallel inside the container.
#   3. Waits for all judges to finish and prints line counts.
#
# Logs go to /workspace/outputs/sbov_*.log.

set -u

OUT=~/workspace/robotics-safety/outputs/qual_fig_judgments/storyboard_overlay
PROMPT=/workspace/scripts/qual_fig/wp_follow_f/safety_prompt.txt
PROMPT_BIN=/workspace/scripts/qual_fig/wp_follow_f/safety_prompt_binary.txt
IMG_ROOT=/workspace/storyboard_overlay
SCRIPT=/workspace/scripts/safety_recognition/run_safety_judge_fpanel.py
CT=robotics-safety-container

declare -A MODELS=(
  [qwen3vl_32b]="Qwen/Qwen3-VL-32B-Instruct:8005"
  [kimi]="moonshotai/Kimi-VL-A3B-Instruct:8006"
  [molmo]="allenai/Molmo-7B-D-0924:8007"
  [deepseek_vl2]="deepseek-ai/deepseek-vl2-small:8008"
)

# Map model_key -> output filename suffix matching what plot_model_compare.py expects.
declare -A SUFFIX=(
  [qwen3vl_32b]=""                        # legacy: blank suffix = Qwen3-VL-32B
  [kimi]="_kimi-vl-instruct"
  [molmo]="_molmo-7b-d"
  [deepseek_vl2]="_deepseek-vl2-small"
)

# ---- Phase 1: wait for all 4 servers to be READY ----
START=$(date +%s)
TIMEOUT=2400  # 40 min, VL-32B can take a while

echo "[$(date +%T)] waiting for 4 vLLM servers to become healthy (timeout ${TIMEOUT}s)..."

ready_ports=""
while true; do
  ready_ports=""
  for port in 8005 8006 8007 8008; do
    if docker exec $CT bash -lc "curl -fsS -m 2 http://localhost:$port/health > /dev/null 2>&1"; then
      ready_ports="$ready_ports $port"
    fi
  done
  count=$(echo "$ready_ports" | wc -w)
  elapsed=$(($(date +%s) - START))
  echo "[$(date +%T)] elapsed=${elapsed}s ready=${count}/4 :${ready_ports}"
  if [ "$count" -eq 4 ]; then
    echo "[$(date +%T)] ALL 4 SERVERS READY after ${elapsed}s"
    break
  fi
  if [ "$elapsed" -gt "$TIMEOUT" ]; then
    echo "[$(date +%T)] TIMEOUT waiting for servers"
    docker exec $CT bash -lc "for p in 8005 8006 8007 8008; do echo === :\$p ===; curl -fsS -m 2 http://localhost:\$p/health 2>&1 || echo NOT READY; done"
    exit 1
  fi
  sleep 30
done

# ---- Phase 2: launch 8 judges in parallel via docker exec -d ----
echo ""
echo "[$(date +%T)] launching 8 judges (4 models x 2 modes) in parallel..."
for key in "${!MODELS[@]}"; do
  spec=${MODELS[$key]}
  model=${spec%:*}
  port=${spec##*:}
  suffix=${SUFFIX[$key]}

  # 1-5 mode
  out_15=/workspace/outputs/qual_fig_judgments/storyboard_overlay/judgments${suffix}.jsonl
  log_15=/workspace/outputs/sbov_${key}_judge_1-5.log
  docker exec -d $CT bash -lc \
"python3 $SCRIPT \
  --img-root $IMG_ROOT \
  --prompt $PROMPT \
  --model $model \
  --base-url http://localhost:$port/v1 \
  --output $out_15 \
  --workers 8 \
  --scale 1-5 \
  --resume \
  > $log_15 2>&1"
  echo "  [launched] $key 1-5 -> $out_15  log=$log_15"

  # binary mode
  out_bin=/workspace/outputs/qual_fig_judgments/storyboard_overlay/judgments${suffix}_binary.jsonl
  log_bin=/workspace/outputs/sbov_${key}_judge_binary.log
  docker exec -d $CT bash -lc \
"python3 $SCRIPT \
  --img-root $IMG_ROOT \
  --prompt $PROMPT_BIN \
  --model $model \
  --base-url http://localhost:$port/v1 \
  --output $out_bin \
  --workers 8 \
  --scale binary \
  --resume \
  > $log_bin 2>&1"
  echo "  [launched] $key binary -> $out_bin  log=$log_bin"
done

# ---- Phase 3: wait until all 8 judge processes exit ----
echo ""
echo "[$(date +%T)] waiting for all 8 judges to finish..."
TARGET=2133
J_TIMEOUT=14400  # 4h
J_START=$(date +%s)
while true; do
  running=$(docker exec $CT bash -lc "ps -ef | grep run_safety_judge_fpanel | grep -v grep | wc -l")
  elapsed=$(($(date +%s) - J_START))
  echo "[$(date +%T)] elapsed=${elapsed}s judges_running=$running"
  if [ "$running" -eq 0 ]; then
    echo "[$(date +%T)] ALL_JUDGES_DONE after ${elapsed}s"
    break
  fi
  if [ "$elapsed" -gt "$J_TIMEOUT" ]; then
    echo "[$(date +%T)] JUDGE_TIMEOUT after ${elapsed}s; still $running running"
    break
  fi
  sleep 60
done

# ---- Phase 4: report line counts ----
echo ""
echo "=== final line counts (target $TARGET) ==="
docker exec $CT bash -lc "cd /workspace/outputs/qual_fig_judgments/storyboard_overlay && for f in judgments*.jsonl; do printf '%-55s %d\n' \"\$f\" \$(wc -l < \"\$f\"); done"
