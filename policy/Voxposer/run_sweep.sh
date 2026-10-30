#!/usr/bin/env bash
# run_sweep.sh — Run VoxPoser navigation evaluation across multiple models.
#
# Usage:
#   bash run_sweep.sh                   # all tasks, all MODELS below
#   bash run_sweep.sh task1 task2 ...   # specific task specs
#
# Each model entry: "MODEL_ID|GPU_IDS|PORT|TP_SIZE|GPU_UTIL|TMUX_SESSION"
# Servers are expected to already be running (or started here if SESSION=new).
# Models run sequentially within the eval loop.

set -euo pipefail

PYTHON=~/miniconda3/envs/alfworld/bin/python3.9
VOXPOSER_DIR="$(cd "$(dirname "$0")" && pwd)"
HF_CACHE=~/.cache/huggingface/hub
HEALTH_TIMEOUT=300

# ── Model registry ──────────────────────────────────────────────────────────
# Format: "MODEL_ID | local_path_or_hf_id | GPU_IDS | PORT | TP | GPU_UTIL | SERVER_SESSION"
declare -A MODEL_PATH=(
    ["meta-llama/Llama-3.1-8B-Instruct"]="${HF_CACHE}/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    ["openai/gpt-oss-20b"]="${HF_CACHE}/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
    ["Qwen/Qwen3-4B-Instruct-2507"]="Qwen/Qwen3-4B-Instruct-2507"
)
declare -A MODEL_GPUS=(
    ["meta-llama/Llama-3.1-8B-Instruct"]="1"
    ["openai/gpt-oss-20b"]="2,3"
    ["Qwen/Qwen3-4B-Instruct-2507"]="0"
)
declare -A MODEL_PORT=(
    ["meta-llama/Llama-3.1-8B-Instruct"]="8001"
    ["openai/gpt-oss-20b"]="8002"
    ["Qwen/Qwen3-4B-Instruct-2507"]="8000"
)
declare -A MODEL_TP=(
    ["meta-llama/Llama-3.1-8B-Instruct"]="1"
    ["openai/gpt-oss-20b"]="2"
    ["Qwen/Qwen3-4B-Instruct-2507"]="1"
)
declare -A MODEL_GPU_UTIL=(
    ["meta-llama/Llama-3.1-8B-Instruct"]="0.85"
    ["openai/gpt-oss-20b"]="0.80"
    ["Qwen/Qwen3-4B-Instruct-2507"]="0.70"
)
declare -A MODEL_MAX_LEN=(
    ["meta-llama/Llama-3.1-8B-Instruct"]="4096"
    ["openai/gpt-oss-20b"]="8192"
    ["Qwen/Qwen3-4B-Instruct-2507"]="4096"
)
declare -A MODEL_SESSION=(
    ["meta-llama/Llama-3.1-8B-Instruct"]="llama-server"
    ["openai/gpt-oss-20b"]="gpt-server"
    ["Qwen/Qwen3-4B-Instruct-2507"]="qwen-server"
)

# ── Eval target models (order matters) ──────────────────────────────────────
EVAL_MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "openai/gpt-oss-20b"
)

# task specs passed as args, or empty (run_LMP.py picks all tasks)
TASK_ARGS=("$@")

# ── Helpers ──────────────────────────────────────────────────────────────────
log() { echo "[sweep $(date +%H:%M:%S)] $*"; }

wait_for_health() {
    local port=$1
    local elapsed=0
    local url="http://localhost:${port}/health"
    log "Waiting for vLLM at ${url} ..."
    while ! curl -sf "${url}" > /dev/null 2>&1; do
        sleep 10
        elapsed=$((elapsed + 10))
        if (( elapsed >= HEALTH_TIMEOUT )); then
            log "ERROR: did not become healthy within ${HEALTH_TIMEOUT}s"
            return 1
        fi
        (( elapsed % 30 == 0 )) && log "  still loading... (${elapsed}s)"
    done
    log "Server on port ${port} ready after ${elapsed}s"
}

start_server() {
    local model=$1
    local session="${MODEL_SESSION[$model]}"
    local path="${MODEL_PATH[$model]}"
    local gpus="${MODEL_GPUS[$model]}"
    local port="${MODEL_PORT[$model]}"
    local tp="${MODEL_TP[$model]}"
    local util="${MODEL_GPU_UTIL[$model]}"
    local maxlen="${MODEL_MAX_LEN[$model]}"

    tmux new-session -d -s "${session}" 2>/dev/null || true
    tmux send-keys -t "${session}" \
        "CUDA_VISIBLE_DEVICES=${gpus} ${PYTHON} -m vllm.entrypoints.openai.api_server \
  --model ${path} \
  --served-model-name ${model} \
  --port ${port} \
  --tensor-parallel-size ${tp} \
  --gpu-memory-utilization ${util} \
  --max-model-len ${maxlen} \
  --trust-remote-code" Enter
    log "Started ${model} in tmux:${session} (GPU=${gpus}, port=${port}, tp=${tp}, max_len=${maxlen})"
}

# ── Main ─────────────────────────────────────────────────────────────────────
SESSION="voxposer-eval"
if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux new-session -d -s "${SESSION}" -n "init"
fi

for MODEL in "${EVAL_MODELS[@]}"; do
    PORT="${MODEL_PORT[$MODEL]}"
    log "====== ${MODEL} (port ${PORT}) ======"

    # Start server if not already up
    if ! curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        start_server "${MODEL}"
    else
        log "Server already running on port ${PORT}"
    fi

    # Wait for healthy
    if ! wait_for_health "${PORT}"; then
        log "Skipping ${MODEL} — server failed to start"
        continue
    fi

    # Create eval window: left=server logs, right=run_LMP.py
    MODEL_SHORT=$(echo "${MODEL}" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr '-' '_')
    WINDOW="${MODEL_SHORT}"
    tmux new-window -t "${SESSION}" -n "${WINDOW}" 2>/dev/null || true
    tmux split-window -h -t "${SESSION}:${WINDOW}" 2>/dev/null || true
    # pipe server logs to left pane
    tmux send-keys -t "${SESSION}:${WINDOW}.0" \
        "tmux pipe-pane -t ${MODEL_SESSION[$MODEL]} 'cat >> /tmp/${MODEL_SHORT}_server.log'; tail -f /tmp/${MODEL_SHORT}_server.log" Enter

    # run_LMP.py in right pane (inside Docker container)
    TASK_STR="${TASK_ARGS[*]+"${TASK_ARGS[*]}"}"
    tmux send-keys -t "${SESSION}:${WINDOW}.1" \
        "docker exec -it robocasa-container bash -c 'cd /workspace/policy/Voxposer && MUJOCO_GL=egl python3 src/run_LMP.py ${TASK_STR} -m ${MODEL} -p ${PORT}'" Enter

    # Wait for eval to finish
    log "Evaluation running in ${SESSION}:${WINDOW}.1 ..."
    while tmux list-panes -t "${SESSION}:${WINDOW}.1" -F "#{pane_current_command}" 2>/dev/null \
          | grep -qE "docker|python"; do
        sleep 15
    done
    log "Evaluation done for ${MODEL}"
done

log "All done. Results in ${VOXPOSER_DIR}/outputs/"
