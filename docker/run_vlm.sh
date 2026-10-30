#!/bin/bash
# vLLM server + LMP tmux session — both run inside the same docker container.
#
# Window 0: vLLM server (`docker exec` runs python inside the container)
# Window 1: waits for vLLM /health, then runs LMP (same container, localhost:{PORT})
#
# vLLM version is pinned to 0.11.2 in the Dockerfile. If you need a different
# model family that requires a newer vLLM, bump the pin in docker/Dockerfile
# and rebuild the image.
#
# Usage: bash docker/run_vlm.sh [-m MODEL] [-p PORT] [-g GPU_ID]
#
# Examples:
#   bash docker/run_vlm.sh                                          # defaults (Qwen3-VL-8B)
#   bash docker/run_vlm.sh -m Qwen/Qwen2.5-VL-7B-Instruct -g 0,1   # 7B model, 2 GPUs

set -e

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
PORT="${PORT:-8001}"
GPU_ID="${GPU_ID:-0}"
SESSION="vlm-lmp"
CONTAINER="robotics-safety-container"
HEALTH_TIMEOUT=300

while getopts "m:p:g:" opt; do
  case $opt in
    m) MODEL="$OPTARG" ;;
    p) PORT="$OPTARG" ;;
    g) GPU_ID="$OPTARG" ;;
    *) echo "Usage: $0 [-m model] [-p port] [-g gpu_id]"; exit 1 ;;
  esac
done

PYTHON="docker exec -it -e CUDA_VISIBLE_DEVICES=$GPU_ID $CONTAINER /usr/bin/python"

# Make sure the Docker container is running
if ! docker ps -q -f name="$CONTAINER" | grep -q .; then
  echo "ERROR: Docker container '$CONTAINER' is not running."
  echo "Start it first: bash docker/run_docker.sh"
  exit 1
fi

# Kill any existing tmux session with the same name
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Kill any existing vLLM server bound to the same port
pids=$(ps aux | grep "api_server.*--port ${PORT}" | grep -v grep | awk '{print $2}')
if [ -n "$pids" ]; then
  echo "Killing existing vLLM server on port $PORT (PIDs: $pids)"
  kill $pids 2>/dev/null || true
  sleep 3
fi

# Create tmux session — window 0: vLLM server (in container, via docker exec)
tmux new-session -d -s "$SESSION" -n "vllm"
tmux send-keys -t "$SESSION:vllm" \
  "$PYTHON -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --port $PORT \
  --gpu-memory-utilization 0.8 \
  --max-model-len 16384 \
  --trust-remote-code \
  --limit-mm-per-prompt '{\"image\": 5}'" C-m

# Window 1: wait for the vLLM server, then run LMP (also in the container)
tmux new-window -t "$SESSION" -n "lmp"
tmux send-keys -t "$SESSION:lmp" \
  "echo 'Waiting for vLLM server on port $PORT...' && \
  elapsed=0 && \
  while ! curl -sf http://localhost:$PORT/health > /dev/null 2>&1; do \
    sleep 10; elapsed=\$((elapsed + 10)); \
    if [ \$elapsed -ge $HEALTH_TIMEOUT ]; then echo 'TIMEOUT'; exit 1; fi; \
    [ \$((elapsed % 30)) -eq 0 ] && echo \"  still loading... (\${elapsed}s)\"; \
  done && \
  echo 'Server ready!' && \
  docker exec -it $CONTAINER bash -c 'cd /workspace/policy/Voxposer && MUJOCO_GL=egl python3 src/run_LMP.py -m \"$MODEL\" -p $PORT'" C-m

echo "tmux session '$SESSION' started"
echo "  window 0 (vllm): vLLM server on HOST ($MODEL, GPU $GPU_ID)"
echo "  window 1 (lmp):  LMP in Docker ($CONTAINER), connects to localhost:$PORT"
echo ""
echo "Attach: tmux attach -t $SESSION"
echo "Switch: Ctrl+b 0 (vllm) / Ctrl+b 1 (lmp)"
echo "Kill:   tmux kill-session -t $SESSION"
