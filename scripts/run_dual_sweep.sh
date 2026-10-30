#!/bin/bash
# Run Qwen3-14B + Qwen3-4B sweeps SIMULTANEOUSLY, then auto-retry until each
# model produced ≥ MIN_COMPLETE entries (default 1000) in results.json, up
# to MAX_RETRY rounds (default 3).
#
# Layout: docker workers are split across the two models (default 3 + 3).
#   robocasa-eval-{0,1,2}  → Qwen3-14B  via vLLM @ port 8002
#   robocasa-eval-{3,4,5}  → Qwen3-4B   via vLLM @ port 8003
#
# Both sweeps enumerate `LAYOUT_IDS=all × STYLE_IDS=3` (1240 tasks each)
# unless overridden.
#
# Usage:
#   bash scripts/run_dual_sweep.sh
#
# Override knobs:
#   PORT_14B=8002      PORT_4B=8003
#   MODEL_14B='Qwen/Qwen3-14B'
#   MODEL_4B='Qwen/Qwen3-4B-Instruct-2507'
#   N_WORKERS_14B=3    N_WORKERS_4B=3
#   LAYOUT_IDS=all     STYLE_IDS=3
#   MIN_COMPLETE=1000  MAX_RETRY=3
#   IMAGE='robotics-safety:committed'

set -e

MODEL_14B="${MODEL_14B:-Qwen/Qwen3-14B}"
MODEL_4B="${MODEL_4B:-Qwen/Qwen3-4B-Instruct-2507}"
PORT_14B="${PORT_14B:-8002}"
PORT_4B="${PORT_4B:-8003}"
N_WORKERS_14B="${N_WORKERS_14B:-3}"
N_WORKERS_4B="${N_WORKERS_4B:-3}"
LAYOUT_IDS="${LAYOUT_IDS:-all}"
STYLE_IDS="${STYLE_IDS:-3}"
MIN_COMPLETE="${MIN_COMPLETE:-1000}"
MAX_RETRY="${MAX_RETRY:-3}"
IMAGE="${IMAGE:-robotics-safety:committed}"

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
TS=$(date +%Y%m%d_%H%M%S)

mk_run_dir() {
    local model="$1"
    local short=$(echo "$model" | sed 's|.*/||; s/-/_/g; s/\./_/g')
    echo "outputs/navigation_${short}_dual_${TS}"
}
RUN_DIR_14B=$(mk_run_dir "$MODEL_14B")
RUN_DIR_4B=$(mk_run_dir "$MODEL_4B")
echo "== 14B run dir: $RUN_DIR_14B =="
echo "== 4B  run dir: $RUN_DIR_4B  =="

# Container assignment
CONTAINERS_14B=()
for i in $(seq 0 $((N_WORKERS_14B-1))); do CONTAINERS_14B+=("robocasa-eval-$i"); done
CONTAINERS_4B=()
for i in $(seq $N_WORKERS_14B $((N_WORKERS_14B + N_WORKERS_4B - 1))); do
    CONTAINERS_4B+=("robocasa-eval-$i")
done

# ---- helpers --------------------------------------------------------------

get_all_tasks() {
    docker exec "${CONTAINERS_14B[0]}" python3 -c "
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.StringIO()
sys.path.insert(0, '/workspace/benchmark/robocasa/robocasa')
from robocasa.utils.result_utils import get_navigate_tasks
tasks = get_navigate_tasks()
sys.stdout = sys.__stdout__
print('\n'.join(tasks))
" 2>/dev/null
}

count_completed() {
    # Number of distinct task_dirs in the merged results.json (success +
    # permanent fail). Returns 0 if file missing.
    local rd="$1"
    local p="${WORKSPACE}/policy/Voxposer/${rd}/results.json"
    if [ -f "$p" ]; then
        python3 -c "
import json, sys
try:
    d = json.load(open('$p'))
    seen = set((r.get('task_info') or {}).get('task_dir') for r in d.get('results', []))
    seen.discard(None)
    print(len(seen))
except Exception:
    print(0)
"
    else
        echo 0
    fi
}

merge_run() {
    local rd="$1"
    local container="$2"
    if docker ps --format '{{.Names}}' | grep -qx "$container"; then
        docker exec "$container" python3 /workspace/scripts/merge_workers.py \
            "/workspace/policy/Voxposer/${rd}" || true
    fi
}

launch_workers_for() {
    local model="$1"; shift
    local port="$1";  shift
    local rd="$1";    shift
    local containers=("$@")
    local n_workers=${#containers[@]}

    local TASKS
    TASKS=$(get_all_tasks)
    local TOTAL=$(echo "$TASKS" | wc -l)
    echo "  [$model] dispatching $TOTAL task names across ${n_workers} workers"

    for i in $(seq 0 $((n_workers-1))); do
        local c="${containers[$i]}"
        local chunk
        chunk=$(echo "$TASKS" | awk -v i="$i" -v W="$n_workers" '(NR-1)%W==i' | tr '\n' ' ')
        local short=$(echo "$model" | sed 's|.*/||; s/-/_/g; s/\./_/g')
        echo "    starting $c (worker_id=$i)"
        docker exec -d "$c" bash -c \
            "cd /workspace/policy/Voxposer && \
             OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 MUJOCO_GL=egl \
             python3 src/run_LMP.py -m $model -p $port -w $i -o $rd \
                 --layout-ids $LAYOUT_IDS --style-ids $STYLE_IDS \
                 $chunk > /tmp/eval_${short}_w${i}.log 2>&1"
    done
}

wait_for_workers() {
    local containers=("$@")
    while true; do
        local n_busy=0
        for c in "${containers[@]}"; do
            local procs=$(docker exec "$c" pgrep -f "run_LMP" 2>/dev/null | wc -l)
            [ "$procs" -gt 0 ] && n_busy=$((n_busy + 1))
        done
        [ "$n_busy" -eq 0 ] && break
        sleep 30
    done
}

# Auto-merge on any exit
_cleanup_on_exit() {
    local rc=$?
    echo ""
    echo "== auto-merge on exit (rc=$rc) =="
    merge_run "$RUN_DIR_14B" "${CONTAINERS_14B[0]}"
    merge_run "$RUN_DIR_4B"  "${CONTAINERS_4B[0]}"
    exit $rc
}
trap _cleanup_on_exit EXIT INT TERM

# ---- main loop ------------------------------------------------------------

for round in $(seq 1 $MAX_RETRY); do
    echo ""
    echo "############### ROUND $round / $MAX_RETRY ###############"

    # Launch both models in parallel
    launch_workers_for "$MODEL_14B" "$PORT_14B" "$RUN_DIR_14B" "${CONTAINERS_14B[@]}"
    launch_workers_for "$MODEL_4B"  "$PORT_4B"  "$RUN_DIR_4B"  "${CONTAINERS_4B[@]}"

    echo "  waiting for both sweeps to drain..."
    wait_for_workers "${CONTAINERS_14B[@]}" "${CONTAINERS_4B[@]}"
    echo "  round $round finished at $(date +%H:%M:%S)"

    # Merge each run dir so missing_tasks.txt + results.json get the latest
    merge_run "$RUN_DIR_14B" "${CONTAINERS_14B[0]}"
    merge_run "$RUN_DIR_4B"  "${CONTAINERS_4B[0]}"

    n14=$(count_completed "$RUN_DIR_14B")
    n4=$(count_completed "$RUN_DIR_4B")
    echo "  14B: $n14 / 1240 complete   4B: $n4 / 1240 complete   (threshold $MIN_COMPLETE)"

    if [ "$n14" -ge "$MIN_COMPLETE" ] && [ "$n4" -ge "$MIN_COMPLETE" ]; then
        echo "  both models passed the threshold — stopping retry loop."
        break
    fi
    if [ "$round" -lt "$MAX_RETRY" ]; then
        echo "  below threshold — retrying round $((round + 1))."
    fi
done

echo ""
echo "Final: 14B=$n14 / 4B=$n4   (target ≥ $MIN_COMPLETE)"
echo "  14B → policy/Voxposer/${RUN_DIR_14B}/results.json"
echo "  4B  → policy/Voxposer/${RUN_DIR_4B}/results.json"
