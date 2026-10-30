#!/bin/bash
# Run GLM-4.6V-Flash on the cropped (cm0-60, 2133 images) set, but only AFTER
# the running nkf10 chain finishes — so we don't fight for GPUs 6,7. Triggered
# manually or by run_nkf10_binary_chain.sh on its way out.
#
# Pre-flight: only proceeds if the nkf10 GLM-Flash run succeeded (387/387).
# That way if vLLM 0.11.2 can't load zai-org/GLM-4.6V-Flash, we don't burn
# another 20+ minutes attempting it on the cropped set.
set -u

CT=robotics-safety-container
HOST_WS=~/workspace/robotics-safety
NKF10_BIN=$HOST_WS/outputs/qual_fig_judgments/storyboard_overlay_nkf10/judgments_glm-4.6v-flash_binary_v5.jsonl
CROPPED_BIN=$HOST_WS/outputs/qual_fig_judgments/storyboard_overlay_cropped/judgments_glm-4.6v-flash_binary_v5.jsonl
IMG_ROOT=/workspace/storyboard_overlay_cropped
OUT_DIR=/workspace/outputs/qual_fig_judgments/storyboard_overlay_cropped
SCRIPT=/workspace/scripts/safety_recognition/run_safety_judge_fpanel.py
PROMPT_BIN=/workspace/scripts/safety_recognition/prompts/safety_prompt_binary.txt
MODEL="zai-org/GLM-4.6V-Flash"
PORT=8016
TARGET=2133
STALL_MAX=900

# 1) Wait for nkf10 chain to either finish GLM-Flash (387/387) OR stop running.
echo "[$(date +%T)] waiting for nkf10 chain to finish GLM-Flash ..."
while pgrep -f run_nkf10_binary_chain.sh > /dev/null; do
  sleep 60
done

if [[ ! -f "$NKF10_BIN" ]] || [[ "$(wc -l < $NKF10_BIN)" -lt "$TARGET" ]]; then
  n=$(wc -l < $NKF10_BIN 2>/dev/null || echo 0)
  echo "[$(date +%T)] nkf10 GLM-Flash incomplete ($n/387) — aborting cropped run"
  exit 1
fi

# 2) Cropped already done? Skip.
if [[ -f "$CROPPED_BIN" ]] && [[ "$(wc -l < $CROPPED_BIN)" -ge "$TARGET" ]]; then
  echo "[$(date +%T)] cropped GLM-Flash already complete — skip"
  exit 0
fi

# 3) Spin up vLLM on GPUs 6,7 (same as nkf10 chain — chain has torn down).
echo "[$(date +%T)] starting vLLM for GLM-Flash on :$PORT (GPUs 6,7) ..."
docker exec $CT bash -lc \
"pkill -9 'VLLM::Worker' 2>/dev/null; pkill -9 'VLLM::EngineCore' 2>/dev/null; pkill -9 -f vllm.entrypoints.openai 2>/dev/null; sleep 10"

docker exec -d $CT bash -lc \
"CUDA_VISIBLE_DEVICES=6,7 python3 -m vllm.entrypoints.openai.api_server \
  --model '$MODEL' --port $PORT --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.88 --max-model-len 8192 --trust-remote-code \
  > /workspace/outputs/vllm_cropped_glm-flash.log 2>&1"

# 4) Wait health.
ready=0
for i in $(seq 1 48); do
  if docker exec $CT bash -lc "curl -fsS -m 3 http://localhost:$PORT/health > /dev/null 2>&1"; then
    echo "[$(date +%T)] :$PORT healthy after ${i}x15s"; ready=1; break
  fi
  sleep 15
done
[[ "$ready" -ne 1 ]] && { echo "[$(date +%T)] :$PORT never came up"; exit 1; }

# 5) Launch judge.
OUT_BIN="$OUT_DIR/judgments_glm-4.6v-flash_binary_v5.jsonl"
echo "[$(date +%T)] launching judge (cropped, target=$TARGET) ..."
docker exec -d $CT bash -lc \
"python3 $SCRIPT --img-root $IMG_ROOT --prompt $PROMPT_BIN --model '$MODEL' \
  --base-url http://localhost:$PORT/v1 --output $OUT_BIN \
  --workers 8 --scale binary --max-tokens 512 --resume \
  > /workspace/outputs/sbov_cropped_glm-flash.log 2>&1"

# 6) Wait for completion.
last=-1; stable=0
while :; do
  n=$(wc -l < "$CROPPED_BIN" 2>/dev/null || echo 0)
  if [[ "$n" -ge "$TARGET" ]]; then
    echo "[$(date +%T)] DONE (n=$n)"
    break
  fi
  if [[ "$n" -eq "$last" ]]; then
    stable=$((stable + 60))
    if [[ "$stable" -ge "$STALL_MAX" ]]; then
      echo "[$(date +%T)] STALLED at n=$n"
      break
    fi
  else
    stable=0; last=$n
    echo "[$(date +%T)] progress: n=$n / $TARGET"
  fi
  sleep 60
done

# 7) Teardown + rebake plots.
docker exec $CT bash -lc "pkill -9 'VLLM::Worker' 2>/dev/null; pkill -9 'VLLM::EngineCore' 2>/dev/null; pkill -9 -f vllm.entrypoints.openai 2>/dev/null"
sleep 20
docker exec $CT bash -lc "python3 /workspace/scripts/qual_fig/plot_v5_2x3_binary.py" 2>&1 | tail -10
echo "[$(date +%T)] all done"
