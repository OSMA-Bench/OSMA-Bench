#!/bin/bash
# Launch v5 judges for the 3 non-Qwen models on the cropped storyboard_overlay set.
# Qwen v5 was done separately by run_sbov_cropped_v5_qwen.sh.
# Output goes to /workspace/outputs/qual_fig_judgments/storyboard_overlay_cropped/
set -u

IMG_ROOT=/workspace/storyboard_overlay_cropped_final
OUT_DIR=/workspace/outputs/qual_fig_judgments/storyboard_overlay_cropped
PROMPT_BIN=/workspace/scripts/safety_recognition/prompts/safety_prompt_binary.txt
PROMPT_15=/workspace/scripts/safety_recognition/prompts/safety_prompt.txt
SCRIPT=/workspace/scripts/safety_recognition/run_safety_judge_fpanel.py
CT=robotics-safety-container

# model_key -> "model_id:port:suffix:max_tokens"  (kimi needs more headroom)
declare -A MODELS=(
  [kimi]="moonshotai/Kimi-VL-A3B-Instruct:8006:_kimi-vl-instruct:768"
  [molmo]="allenai/Molmo-7B-D-0924:8007:_molmo-7b-d:256"
  [deepseek_vl2]="deepseek-ai/deepseek-vl2-small:8008:_deepseek-vl2-small:256"
)

echo "[$(date +%T)] confirming server health..."
for spec in "${MODELS[@]}"; do
  port=$(echo "$spec" | cut -d: -f2)
  if docker exec $CT bash -lc "curl -fsS -m 3 http://localhost:$port/health > /dev/null 2>&1"; then
    echo "  :$port healthy"
  else
    echo "  :$port NOT HEALTHY — aborting"
    exit 1
  fi
done

mkdir -p ~/workspace/robotics-safety/outputs/qual_fig_judgments/storyboard_overlay_cropped

echo ""
echo "[$(date +%T)] launching 6 v5 judges (3 models x 2 scales)..."
for key in "${!MODELS[@]}"; do
  spec=${MODELS[$key]}
  model=$(echo "$spec" | cut -d: -f1)
  port=$(echo "$spec" | cut -d: -f2)
  suffix=$(echo "$spec" | cut -d: -f3)
  tok=$(echo "$spec" | cut -d: -f4)

  # binary
  out_bin=$OUT_DIR/judgments${suffix}_binary_v5.jsonl
  log_bin=/workspace/outputs/sbov_cropped_v5_${key}_binary.log
  docker exec -d $CT bash -lc \
"python3 $SCRIPT --img-root $IMG_ROOT --prompt $PROMPT_BIN --model $model --base-url http://localhost:$port/v1 \
 --output $out_bin --workers 8 --scale binary --max-tokens $tok --resume > $log_bin 2>&1"
  echo "  [launched] $key binary -> $out_bin"

  # 1-5
  out_15=$OUT_DIR/judgments${suffix}_v5.jsonl
  log_15=/workspace/outputs/sbov_cropped_v5_${key}_1-5.log
  docker exec -d $CT bash -lc \
"python3 $SCRIPT --img-root $IMG_ROOT --prompt $PROMPT_15 --model $model --base-url http://localhost:$port/v1 \
 --output $out_15 --workers 8 --scale 1-5 --max-tokens $tok --resume > $log_15 2>&1"
  echo "  [launched] $key 1-5    -> $out_15"
done

sleep 5
echo ""
echo "[$(date +%T)] process snapshot:"
docker exec $CT bash -lc "ps -ef | grep safety_prompt_.*_v5 | grep -v grep | awk '{print \$NF\":\"\$2}'" | head -20
