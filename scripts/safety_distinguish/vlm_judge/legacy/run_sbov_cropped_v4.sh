#!/bin/bash
# Launch 8 v4 judges (4 models x 2 modes) on the CROPPED storyboard_overlay images.
#   IMG: /workspace/storyboard_overlay_cropped_final
#   PROMPT: safety_prompt_binary_v4.txt + safety_prompt_v4.txt
#   OUT:   /workspace/outputs/qual_fig_judgments/storyboard_overlay_cropped/
#
# Logs to /workspace/outputs/sbov_cropped_v4_*.log
set -u

IMG_ROOT=/workspace/storyboard_overlay_cropped_final
OUT_DIR=/workspace/outputs/qual_fig_judgments/storyboard_overlay_cropped
PROMPT_BIN=/workspace/scripts/qual_fig/wp_follow_f/safety_prompt_binary_v4.txt
PROMPT_15=/workspace/scripts/qual_fig/wp_follow_f/safety_prompt_v4.txt
SCRIPT=/workspace/scripts/safety_recognition/run_safety_judge_fpanel.py
CT=robotics-safety-container

# model_key -> "model_id:port"
declare -A MODELS=(
  [qwen3vl_32b]="Qwen/Qwen3-VL-32B-Instruct:8011"
  [kimi]="moonshotai/Kimi-VL-A3B-Instruct:8006"
  [molmo]="allenai/Molmo-7B-D-0924:8007"
  [deepseek_vl2]="deepseek-ai/deepseek-vl2-small:8008"
)

# Output filename suffix (matches existing convention)
declare -A SUFFIX=(
  [qwen3vl_32b]=""                  # blank = Qwen3-VL-32B
  [kimi]="_kimi-vl-instruct"
  [molmo]="_molmo-7b-d"
  [deepseek_vl2]="_deepseek-vl2-small"
)

# token caps (kimi needs more headroom)
declare -A MAXTOK=(
  [qwen3vl_32b]=256
  [kimi]=768
  [molmo]=256
  [deepseek_vl2]=256
)

# ---- Phase 1: confirm all 5 servers healthy (Qwen on 8011, others on 8006-8008) ----
echo "[$(date +%T)] confirming server health..."
for spec in "${MODELS[@]}"; do
  port=${spec##*:}
  if docker exec $CT bash -lc "curl -fsS -m 3 http://localhost:$port/health > /dev/null 2>&1"; then
    echo "  :$port healthy"
  else
    echo "  :$port NOT HEALTHY — aborting"
    exit 1
  fi
done

mkdir -p ~/workspace/robotics-safety/outputs/qual_fig_judgments/storyboard_overlay_cropped

# ---- Phase 2: launch 8 judges (binary then 1-5) ----
echo ""
echo "[$(date +%T)] launching 8 v4 judges..."
for key in "${!MODELS[@]}"; do
  spec=${MODELS[$key]}
  model=${spec%:*}
  port=${spec##*:}
  suffix=${SUFFIX[$key]}
  tok=${MAXTOK[$key]}

  # binary
  out_bin=$OUT_DIR/judgments${suffix}_binary_v4.jsonl
  log_bin=/workspace/outputs/sbov_cropped_v4_${key}_binary.log
  docker exec -d $CT bash -lc \
"python3 $SCRIPT --img-root $IMG_ROOT --prompt $PROMPT_BIN --model $model --base-url http://localhost:$port/v1 \
 --output $out_bin --workers 8 --scale binary --max-tokens $tok --resume > $log_bin 2>&1"
  echo "  [launched] $key binary -> $out_bin"

  # 1-5
  out_15=$OUT_DIR/judgments${suffix}_v4.jsonl
  log_15=/workspace/outputs/sbov_cropped_v4_${key}_1-5.log
  docker exec -d $CT bash -lc \
"python3 $SCRIPT --img-root $IMG_ROOT --prompt $PROMPT_15 --model $model --base-url http://localhost:$port/v1 \
 --output $out_15 --workers 8 --scale 1-5 --max-tokens $tok --resume > $log_15 2>&1"
  echo "  [launched] $key 1-5    -> $out_15"
done

# ---- Phase 3: brief status check (don't block waiting) ----
echo ""
echo "[$(date +%T)] launched. Initial process snapshot:"
sleep 5
docker exec $CT bash -lc "ps -ef | grep safety_prompt_.*v4 | grep -v grep | wc -l" | xargs echo "  v4 judges currently running:"
