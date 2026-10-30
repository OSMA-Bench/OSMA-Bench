#!/usr/bin/env bash
# Run parallel eval for multiple models sequentially.
# Each model: start vLLM → run_parallel_eval.sh → merge+viz → kill vLLM → next model
#
# Usage:
#   bash scripts/run_multi_model_eval.sh
#   bash scripts/run_multi_model_eval.sh "Qwen/Qwen3-14B" "microsoft/phi-4"

set -uo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

CONTAINER=robotics-safety-container
PORT=8002
HEALTH_TIMEOUT=900

if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("Qwen/Qwen3-4B-Instruct-2507" "Qwen/Qwen3-14B" "microsoft/phi-4")
fi

# Per-model GPU/TP settings — vLLM runs inside ${CONTAINER}, TP=2 baseline.
# phi-4 has 10 KV heads (not divisible by 4), must use TP=2 anyway
declare -A MODEL_GPUS=(
    ["Qwen/Qwen3-4B-Instruct-2507"]="0,1"
    ["Qwen/Qwen3-14B"]="0,1"
    ["microsoft/phi-4"]="0,1"
)
declare -A MODEL_TP=(
    ["Qwen/Qwen3-4B-Instruct-2507"]="2"
    ["Qwen/Qwen3-14B"]="2"
    ["microsoft/phi-4"]="2"
)
declare -A MODEL_GPU_UTIL=(
    ["Qwen/Qwen3-4B-Instruct-2507"]="0.45"
    ["Qwen/Qwen3-14B"]="0.90"
    ["microsoft/phi-4"]="0.90"
)
declare -A MODEL_MAX_LEN=(
    ["Qwen/Qwen3-4B-Instruct-2507"]="8192"
    ["Qwen/Qwen3-14B"]="8192"
    ["microsoft/phi-4"]="8192"
)
declare -A MODEL_MAX_SEQS=(
    ["Qwen/Qwen3-4B-Instruct-2507"]="256"
    ["Qwen/Qwen3-14B"]="256"
    ["microsoft/phi-4"]="256"
)

log() { echo "[multi-eval $(date '+%H:%M:%S')] $*"; }

wait_for_health() {
    local elapsed=0
    log "Waiting for vLLM at port ${PORT}..."
    while ! curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        sleep 10
        elapsed=$((elapsed + 10))
        if (( elapsed >= HEALTH_TIMEOUT )); then
            log "ERROR: server did not become healthy within ${HEALTH_TIMEOUT}s"
            return 1
        fi
        (( elapsed % 60 == 0 )) && log "  still loading... (${elapsed}s)"
    done
    log "Server ready after ${elapsed}s"
}

kill_server() {
    if docker exec "${CONTAINER}" pgrep -f "api_server.*--port ${PORT}" >/dev/null 2>&1; then
        log "Killing vLLM server inside ${CONTAINER} (port ${PORT})"
        docker exec "${CONTAINER}" pkill -f "api_server.*--port ${PORT}" 2>/dev/null || true
        sleep 5
        if docker exec "${CONTAINER}" pgrep -f "api_server.*--port ${PORT}" >/dev/null 2>&1; then
            docker exec "${CONTAINER}" pkill -9 -f "api_server.*--port ${PORT}" 2>/dev/null || true
        fi
        sleep 2
    else
        log "No vLLM server on port ${PORT} to kill"
    fi
    return 0
}

for MODEL in "${MODELS[@]}"; do
    log "========================================"
    log "MODEL: ${MODEL}"
    log "========================================"

    kill_server

    GPUS="${MODEL_GPUS[$MODEL]:-0,2}"
    TP="${MODEL_TP[$MODEL]:-2}"
    UTIL="${MODEL_GPU_UTIL[$MODEL]:-0.90}"
    MAXLEN="${MODEL_MAX_LEN[$MODEL]:-8192}"
    MODEL_SAFE="${MODEL//\//_}"

    MAXSEQS="${MODEL_MAX_SEQS[$MODEL]:-256}"
    log "Starting vLLM in ${CONTAINER}: GPU=${GPUS}, TP=${TP}, util=${UTIL}, max_len=${MAXLEN}, max_seqs=${MAXSEQS}"
    docker exec -d -e CUDA_VISIBLE_DEVICES=${GPUS} "${CONTAINER}" \
        bash -c "/usr/bin/python -m vllm.entrypoints.openai.api_server \
            --model '${MODEL}' \
            --port ${PORT} \
            --tensor-parallel-size ${TP} \
            --gpu-memory-utilization ${UTIL} \
            --max-model-len ${MAXLEN} \
            --max-num-seqs ${MAXSEQS} \
            --trust-remote-code \
            > /tmp/vllm_${MODEL_SAFE}_port${PORT}.log 2>&1"
    log "Server launched (log: /tmp/vllm_${MODEL_SAFE}_port${PORT}.log inside ${CONTAINER})"

    if ! wait_for_health; then
        log "SKIP ${MODEL} — server failed to start"
        kill_server
        continue
    fi

    log "Starting parallel evaluation..."
    set +e
    MODEL="${MODEL}" PORT=${PORT} bash scripts/run_parallel_eval.sh 2>&1 | tee "/tmp/eval_${MODEL_SAFE}.log"
    EVAL_EXIT=${PIPESTATUS[0]}
    set -e

    if [ ${EVAL_EXIT} -ne 0 ]; then
        log "WARNING: Evaluation exited with code ${EVAL_EXIT} for ${MODEL}"
    fi

    log "Evaluation done for ${MODEL}"
    kill_server
    sleep 5
done

log "All models complete. Results in policy/Voxposer/outputs/"
