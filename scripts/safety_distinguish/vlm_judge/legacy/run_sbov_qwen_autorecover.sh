#!/bin/bash
# Auto-recover Qwen3-VL-32B:
#   1. Wait until GPUs 3 and 4 are free (orphan PIDs killed externally).
#   2. Launch Qwen3-VL-32B vLLM server on :8011 (TP=2 GPUs 3,4).
#   3. Wait for FULL inference readiness (probe /v1/chat/completions).
#   4. Delete stale Qwen output files.
#   5. Launch Qwen3-VL-32B judges (1-5 + binary).
#   6. Wait until ALL judges across all models finish; print final counts.

set -u

CT=robotics-safety-container
SCRIPT=/workspace/scripts/safety_recognition/run_safety_judge_fpanel.py
IMG_ROOT=/workspace/storyboard_overlay
PROMPT=/workspace/scripts/qual_fig/wp_follow_f/safety_prompt.txt
PROMPT_BIN=/workspace/scripts/qual_fig/wp_follow_f/safety_prompt_binary.txt
MODEL="Qwen/Qwen3-VL-32B-Instruct"
PORT=8011
OUT_DIR=/workspace/outputs/qual_fig_judgments/storyboard_overlay
GPU_TIMEOUT=3600        # wait up to 1h for orphans to be killed
READY_TIMEOUT=2400      # wait up to 40min for engine to come up
JUDGE_TIMEOUT=14400     # wait up to 4h for judges to finish

echo "[$(date +%T)] watching GPUs 3 & 4 for orphan PID kill..."
G_START=$(date +%s)

while true; do
  free3=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 3 2>/dev/null | tr -d ' ')
  free4=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 4 2>/dev/null | tr -d ' ')
  elapsed=$(($(date +%s) - G_START))
  echo "[$(date +%T)] elapsed=${elapsed}s GPU3_free=${free3}MiB GPU4_free=${free4}MiB"
  # Need ~42 GiB free per GPU (vllm --gpu-memory-utilization 0.9 of 48GB).
  if [ "${free3:-0}" -ge 42000 ] && [ "${free4:-0}" -ge 42000 ]; then
    echo "[$(date +%T)] GPUs 3 & 4 are FREE (>42GiB each)"
    break
  fi
  if [ "$elapsed" -gt "$GPU_TIMEOUT" ]; then
    echo "[$(date +%T)] TIMEOUT waiting for GPU memory"
    exit 1
  fi
  sleep 30
done

# ---- Launch Qwen3-VL-32B :8011 ----
echo ""
echo "[$(date +%T)] launching Qwen3-VL-32B on :$PORT (GPUs 3,4 TP=2)..."
docker exec -d -e CUDA_VISIBLE_DEVICES=3,4 $CT bash -lc \
"/usr/bin/python -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --port $PORT \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --trust-remote-code \
  > /workspace/outputs/sbov_qwen3vl32b_v2_server.log 2>&1"

# ---- Wait for full inference readiness ----
R_START=$(date +%s)
echo "[$(date +%T)] waiting for :$PORT inference readiness..."
while true; do
  elapsed=$(($(date +%s) - R_START))
  models=$(docker exec $CT bash -lc "curl -fsS -m 3 http://localhost:$PORT/v1/models 2>/dev/null" || echo "")
  if echo "$models" | grep -q "Qwen3-VL-32B-Instruct"; then
    probe=$(docker exec $CT bash -lc "curl -fsS -m 30 -X POST http://localhost:$PORT/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"PING\"}],\"max_tokens\":4}' 2>/dev/null" || echo "")
    if echo "$probe" | grep -q '"content"'; then
      echo "[$(date +%T)] READY after ${elapsed}s; probe: $(echo $probe | head -c 200)"
      break
    fi
  fi
  echo "[$(date +%T)] elapsed=${elapsed}s :$PORT not ready yet"
  if [ "$elapsed" -gt "$READY_TIMEOUT" ]; then
    echo "[$(date +%T)] TIMEOUT waiting for inference readiness"
    docker exec $CT bash -lc "tail -20 /workspace/outputs/sbov_qwen3vl32b_v2_server.log"
    exit 1
  fi
  sleep 30
done

# ---- Reset Qwen outputs ----
docker exec $CT bash -lc "rm -f $OUT_DIR/judgments.jsonl $OUT_DIR/judgments_binary.jsonl"
echo "[$(date +%T)] cleaned previous Qwen outputs"

# ---- Launch Qwen judges ----
docker exec -d $CT bash -lc \
"python3 $SCRIPT \
  --img-root $IMG_ROOT \
  --prompt $PROMPT \
  --model $MODEL \
  --base-url http://localhost:$PORT/v1 \
  --output $OUT_DIR/judgments.jsonl \
  --workers 8 --scale 1-5 --max-tokens 256 --resume \
  > /workspace/outputs/sbov_qwen3vl_32b_judge_1-5.log 2>&1"
echo "[$(date +%T)] [launched] Qwen3-VL-32B 1-5 (:$PORT)"

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

# ---- Wait until ALL judges finish ----
echo ""
echo "[$(date +%T)] waiting for all judges to finish..."
J_START=$(date +%s)
while true; do
  running=$(docker exec $CT bash -lc "ps -ef | grep run_safety_judge_fpanel | grep -v grep | wc -l")
  elapsed=$(($(date +%s) - J_START))
  echo "[$(date +%T)] elapsed=${elapsed}s judges_running=$running"
  if [ "$running" -eq 0 ]; then
    echo "[$(date +%T)] ALL_JUDGES_DONE after ${elapsed}s"
    break
  fi
  if [ "$elapsed" -gt "$JUDGE_TIMEOUT" ]; then
    echo "[$(date +%T)] JUDGE_TIMEOUT after ${elapsed}s; still $running running"
    break
  fi
  sleep 60
done

echo ""
echo "=== final line counts (target 2133) ==="
docker exec $CT bash -lc "cd $OUT_DIR && for f in judgments*.jsonl; do printf '%-55s %d\n' \"\$f\" \$(wc -l < \"\$f\"); done"

echo ""
echo "=== ok/err breakdown ==="
docker exec $CT python3 -c "
import json, os
d='$OUT_DIR'
for f in sorted(os.listdir(d)):
  if not f.endswith('.jsonl'): continue
  recs = [json.loads(l) for l in open(f'{d}/{f}')]
  n_ok = sum(1 for r in recs if r.get('error') is None and r.get('score') is not None)
  n_err = sum(1 for r in recs if r.get('error'))
  n_np = sum(1 for r in recs if r.get('error') is None and r.get('score') is None)
  print(f'{f:55s} total={len(recs)} ok={n_ok} err={n_err} noparse={n_np}')
"
