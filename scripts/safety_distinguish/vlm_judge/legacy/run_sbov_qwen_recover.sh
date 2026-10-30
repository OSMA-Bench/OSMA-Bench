#!/bin/bash
# Recover Qwen3-VL-32B judges on :8011.
#   1. Wait for :8011 to be FULLY ready (real inference, not just /health).
#   2. Delete stale judgments.jsonl / judgments_binary.jsonl (Qwen outputs).
#   3. Launch 2 Qwen judges (1-5 + binary).
#   4. Wait for them to finish (or all 8 judges to finish).

set -u

CT=robotics-safety-container
SCRIPT=/workspace/scripts/safety_recognition/run_safety_judge_fpanel.py
IMG_ROOT=/workspace/storyboard_overlay
PROMPT=/workspace/scripts/qual_fig/wp_follow_f/safety_prompt.txt
PROMPT_BIN=/workspace/scripts/qual_fig/wp_follow_f/safety_prompt_binary.txt
MODEL="Qwen/Qwen3-VL-32B-Instruct"
PORT=8011
OUT_DIR=/workspace/outputs/qual_fig_judgments/storyboard_overlay
TIMEOUT=2400

START=$(date +%s)
echo "[$(date +%T)] waiting for :$PORT to serve $MODEL via REAL inference..."

while true; do
  elapsed=$(($(date +%s) - START))
  # Step 1: check /v1/models lists Qwen3-VL-32B (not some ghost)
  models=$(docker exec $CT bash -lc "curl -fsS -m 3 http://localhost:$PORT/v1/models 2>/dev/null" || echo "")
  if echo "$models" | grep -q "Qwen3-VL-32B-Instruct"; then
    # Step 2: send a tiny inference probe; only declare READY if 200 OK + non-empty content
    probe=$(docker exec $CT bash -lc "curl -fsS -m 30 -X POST http://localhost:$PORT/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say PONG\"}],\"max_tokens\":8}' 2>/dev/null" || echo "")
    if echo "$probe" | grep -q '"content"'; then
      echo "[$(date +%T)] READY after ${elapsed}s; probe: $(echo $probe | head -c 200)"
      break
    fi
  fi
  echo "[$(date +%T)] elapsed=${elapsed}s :$PORT not ready yet"
  if [ "$elapsed" -gt "$TIMEOUT" ]; then
    echo "[$(date +%T)] TIMEOUT after ${elapsed}s"
    docker exec $CT bash -lc "tail -15 /workspace/outputs/sbov_qwen3vl32b_v2_server.log"
    exit 1
  fi
  sleep 30
done

# Step 3: clean any previously written Qwen outputs
docker exec $CT bash -lc "rm -f $OUT_DIR/judgments.jsonl $OUT_DIR/judgments_binary.jsonl"
echo "[$(date +%T)] cleaned previous Qwen outputs"

# Step 4: launch judges pointing to :8011
docker exec -d $CT bash -lc \
"python3 $SCRIPT \
  --img-root $IMG_ROOT \
  --prompt $PROMPT \
  --model $MODEL \
  --base-url http://localhost:$PORT/v1 \
  --output $OUT_DIR/judgments.jsonl \
  --workers 8 --scale 1-5 --max-tokens 256 --resume \
  > /workspace/outputs/sbov_qwen3vl_32b_judge_1-5.log 2>&1"
echo "[$(date +%T)] [launched] Qwen3-VL-32B 1-5  (:$PORT)"

docker exec -d $CT bash -lc \
"python3 $SCRIPT \
  --img-root $IMG_ROOT \
  --prompt $PROMPT_BIN \
  --model $MODEL \
  --base-url http://localhost:$PORT/v1 \
  --output $OUT_DIR/judgments_binary.jsonl \
  --workers 8 --scale binary --max-tokens 256 --resume \
  > /workspace/outputs/sbov_qwen3vl_32b_judge_binary.log 2>&1"
echo "[$(date +%T)] [launched] Qwen3-VL-32B binary (:$PORT)"

# Step 5: wait for all 8 judges to finish
echo ""
echo "[$(date +%T)] waiting for all 8 judges to finish..."
J_TIMEOUT=14400
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

echo ""
echo "=== final line counts (target 2133) ==="
docker exec $CT bash -lc "cd $OUT_DIR && for f in judgments*.jsonl; do printf '%-55s %d\n' \"\$f\" \$(wc -l < \"\$f\"); done"
