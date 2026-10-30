#!/bin/bash
# Launch v5 judges (binary + 1-5) for one Tier 1 model on the cropped storyboard set.
# Tier 1 models are launched sequentially because each occupies GPU 2+6 (TP=2);
# only one Tier 1 vLLM lives at a time. Pass the model key as $1.
#
# Usage:   bash scripts/run_sbov_cropped_v5_tier1.sh kimi_thinking
#                                                    qwen30b_a3b
#                                                    glm46v_fp8
set -u
KEY=${1:?usage: $0 <kimi_thinking|qwen30b_a3b|glm46v_fp8>}

IMG_ROOT=/workspace/storyboard_overlay_cropped_final
OUT_DIR=/workspace/outputs/qual_fig_judgments/storyboard_overlay_cropped
PROMPT_BIN=/workspace/scripts/safety_recognition/prompts/safety_prompt_binary.txt
PROMPT_15=/workspace/scripts/safety_recognition/prompts/safety_prompt.txt
SCRIPT=/workspace/scripts/safety_recognition/run_safety_judge_fpanel.py
CT=robotics-safety-container

# model_key -> "model_id:port:suffix:max_tokens"
# Kimi-Thinking generates a reasoning trace before "Final Answer:" so it
# needs more headroom than the non-thinking Instruct variant.
declare -A MODELS=(
  [kimi_thinking]="moonshotai/Kimi-VL-A3B-Thinking-2506:8012:_kimi-vl-thinking:1536"
  [qwen30b_a3b]="Qwen/Qwen3-VL-30B-A3B-Instruct:8013:_qwen3-vl-30b-a3b:512"
  [glm46v_fp8]="zai-org/GLM-4.6V-FP8:8014:_glm-4.6v-fp8:512"
)

if [ -z "${MODELS[$KEY]:-}" ]; then
  echo "unknown key '$KEY'  (allowed: ${!MODELS[*]})"
  exit 2
fi
spec=${MODELS[$KEY]}
model=$(echo "$spec" | cut -d: -f1)
port=$(echo "$spec" | cut -d: -f2)
suffix=$(echo "$spec" | cut -d: -f3)
tok=$(echo "$spec" | cut -d: -f4)

echo "[$(date +%T)] confirming :$port healthy for $model..."
if docker exec $CT bash -lc "curl -fsS -m 3 http://localhost:$port/health > /dev/null 2>&1"; then
  echo "  :$port healthy"
else
  echo "  :$port NOT HEALTHY — aborting"
  exit 1
fi

mkdir -p ~/workspace/robotics-safety/outputs/qual_fig_judgments/storyboard_overlay_cropped

# binary
out_bin=$OUT_DIR/judgments${suffix}_binary_v5.jsonl
log_bin=/workspace/outputs/sbov_cropped_v5_${KEY}_binary.log
echo "[$(date +%T)] launching binary judge -> $out_bin"
docker exec -d $CT bash -lc \
"python3 $SCRIPT --img-root $IMG_ROOT --prompt $PROMPT_BIN --model $model --base-url http://localhost:$port/v1 \
 --output $out_bin --workers 4 --scale binary --max-tokens $tok --resume > $log_bin 2>&1"

# 1-5
out_15=$OUT_DIR/judgments${suffix}_v5.jsonl
log_15=/workspace/outputs/sbov_cropped_v5_${KEY}_1-5.log
echo "[$(date +%T)] launching 1-5    judge -> $out_15"
docker exec -d $CT bash -lc \
"python3 $SCRIPT --img-root $IMG_ROOT --prompt $PROMPT_15 --model $model --base-url http://localhost:$port/v1 \
 --output $out_15 --workers 4 --scale 1-5 --max-tokens $tok --resume > $log_15 2>&1"

sleep 5
echo ""
echo "[$(date +%T)] process snapshot:"
docker exec $CT bash -lc "ps -ef | grep 'safety_judge_fpanel.py.*--model $model' | grep -v grep | awk '{print \$NF\":\"\$2}'" | head -5
echo ""
echo "monitor with:"
echo "  tail -f ~/workspace/robotics-safety$log_bin"
echo "  tail -f ~/workspace/robotics-safety$log_15"
