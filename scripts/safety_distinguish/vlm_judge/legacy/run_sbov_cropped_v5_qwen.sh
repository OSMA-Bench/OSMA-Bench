#!/bin/bash
# Launch Qwen3-VL-32B v5 judges (binary + 1-5) on the CROPPED storyboard_overlay images.
# v5 prompt change: induce monotonic cm-vs-safety via PROPORTIONAL gap-to-obstacle framing,
# drop the "any-timestep-fails => unsafe" collapse, anchor 1..5 to visible gap / obstacle width.
#
# IMG: /workspace/storyboard_overlay_cropped_final
# PROMPT: safety_prompt_binary_v5.txt + safety_prompt_v5.txt
# OUT:   /workspace/outputs/qual_fig_judgments/storyboard_overlay_cropped/
set -u

IMG_ROOT=/workspace/storyboard_overlay_cropped_final
OUT_DIR=/workspace/outputs/qual_fig_judgments/storyboard_overlay_cropped
PROMPT_BIN=/workspace/scripts/safety_recognition/prompts/safety_prompt_binary.txt
PROMPT_15=/workspace/scripts/safety_recognition/prompts/safety_prompt.txt
SCRIPT=/workspace/scripts/safety_recognition/run_safety_judge_fpanel.py
CT=robotics-safety-container

MODEL=Qwen/Qwen3-VL-32B-Instruct
PORT=8011
SUFFIX=""             # blank suffix = Qwen3-VL-32B
TOK=256

echo "[$(date +%T)] confirming :$PORT health..."
docker exec $CT bash -lc "curl -fsS -m 3 http://localhost:$PORT/health > /dev/null" || { echo "  :$PORT NOT HEALTHY"; exit 1; }

mkdir -p ~/workspace/robotics-safety/outputs/qual_fig_judgments/storyboard_overlay_cropped

out_bin=$OUT_DIR/judgments${SUFFIX}_binary_v5.jsonl
log_bin=/workspace/outputs/sbov_cropped_v5_qwen_binary.log
docker exec -d $CT bash -lc \
"python3 $SCRIPT --img-root $IMG_ROOT --prompt $PROMPT_BIN --model $MODEL --base-url http://localhost:$PORT/v1 \
 --output $out_bin --workers 8 --scale binary --max-tokens $TOK --resume > $log_bin 2>&1"
echo "  [launched] qwen v5 binary -> $out_bin"

out_15=$OUT_DIR/judgments${SUFFIX}_v5.jsonl
log_15=/workspace/outputs/sbov_cropped_v5_qwen_1-5.log
docker exec -d $CT bash -lc \
"python3 $SCRIPT --img-root $IMG_ROOT --prompt $PROMPT_15 --model $MODEL --base-url http://localhost:$PORT/v1 \
 --output $out_15 --workers 8 --scale 1-5 --max-tokens $TOK --resume > $log_15 2>&1"
echo "  [launched] qwen v5 1-5    -> $out_15"

sleep 5
echo ""
echo "[$(date +%T)] process snapshot:"
docker exec $CT bash -lc "ps -ef | grep safety_prompt_.*_v5 | grep -v grep"
