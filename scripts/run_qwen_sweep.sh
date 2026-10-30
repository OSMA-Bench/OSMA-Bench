#!/bin/bash
# Qwen3 sweep: B-heavy + NB-light split with NB:B ≈ 3:7.
#
#   B-heavy   = all B (Blocking) task names  × all 10 layouts × style 3   = 620 tasks
#   NB-light  = all NB (NonBlocking) names   × 3 layouts (0,3,6) × style 3 = 186 tasks
#
# Both sub-sweeps go into the SAME RUN_DIR so the merged results.json /
# summary.ssi_srl / summary.ssi_oct are computed on the union.
#
# Usage:
#   MODEL=Qwen/Qwen3-14B PORT=8002 bash scripts/run_qwen_sweep.sh
#   MODEL=Qwen/Qwen3-4B-Instruct-2507 PORT=8002 bash scripts/run_qwen_sweep.sh
#
# Override knobs:
#   WORKERS              (default 6)
#   B_LAYOUTS            (default 'all')
#   NB_LAYOUTS           (default '0,3,6')
#   STYLE_IDS            (default 3)

set -e
MODEL="${MODEL:?MODEL is required (e.g. Qwen/Qwen3-14B)}"
PORT="${PORT:-8002}"
WORKERS="${WORKERS:-6}"
B_LAYOUTS="${B_LAYOUTS:-all}"
NB_LAYOUTS="${NB_LAYOUTS:-0,3,6}"
STYLE_IDS="${STYLE_IDS:-3}"

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||; s/-/_/g')
RUN_DIR="outputs/navigation_${MODEL_SHORT}_sweep_${TIMESTAMP}"
echo "== Sweep RUN_DIR: $RUN_DIR =="

# Pull task lists from inside one of the eval containers.
get_task_subset() {
    local container="$1"
    local subset="$2"   # "B" (Blocking) | "NB" (NonBlocking)
    docker exec "$container" python3 -c "
import sys, io, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.StringIO()
sys.path.insert(0, '/workspace/benchmark/robocasa/robocasa')
from robocasa.utils.result_utils import get_navigate_tasks
all_t = get_navigate_tasks()
if '$subset' == 'B':
    out = [t for t in all_t if 'NonBlocking' not in t]
else:
    out = [t for t in all_t if 'NonBlocking' in t]
sys.stdout = sys.__stdout__
print('\n'.join(out))
" 2>/dev/null
}

run_subset() {
    local subset="$1"     # "B" or "NB"
    local layout_ids="$2"
    local label="$3"
    echo ""
    echo "==== $label ($subset task names × layouts=$layout_ids × style=$STYLE_IDS) ===="

    # Build container list
    local CONTAINERS=()
    for i in $(seq 0 $((WORKERS-1))); do CONTAINERS+=("robocasa-eval-$i"); done
    local primary="${CONTAINERS[0]}"

    # Fetch task list
    local TASKS
    TASKS=$(get_task_subset "$primary" "$subset")
    local TOTAL=$(echo "$TASKS" | wc -l)
    echo "  $TOTAL task names in $subset subset"

    # Round-robin split across workers
    local TASK_LISTS=()
    for i in $(seq 0 $((WORKERS-1))); do
        chunk=$(echo "$TASKS" | awk -v i="$i" -v W="$WORKERS" '(NR-1)%W==i')
        cnt=$(echo "$chunk" | wc -l)
        TASK_LISTS+=("$chunk")
        echo "    Worker $i: $cnt task names"
    done

    # Launch
    for i in $(seq 0 $((WORKERS-1))); do
        local name="${CONTAINERS[$i]}"
        local args
        args=$(echo "${TASK_LISTS[$i]}" | tr '\n' ' ')
        echo "  starting $name ($subset)"
        docker exec -d "$name" bash -c \
            "cd /workspace/policy/Voxposer && \
             OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 MUJOCO_GL=egl \
             python3 src/run_LMP.py -m $MODEL -p $PORT -w $i -o $RUN_DIR \
                 --layout-ids $layout_ids --style-ids $STYLE_IDS \
                 $args > /tmp/eval_${MODEL_SHORT}_${subset}_w${i}.log 2>&1"
    done

    # Wait for all workers in this subset to drain
    while true; do
        done_count=0
        for c in "${CONTAINERS[@]}"; do
            procs=$(docker exec "$c" pgrep -f "run_LMP" 2>/dev/null | wc -l)
            [ "$procs" -eq 0 ] && done_count=$((done_count + 1))
        done
        [ "$done_count" -eq "$WORKERS" ] && break
        sleep 30
    done
    echo "  $label finished at $(date +%H:%M:%S)"
}

# Auto-merge on any exit path
_merge_on_exit() {
    local rc=$?
    if [ -d "${WORKSPACE}/policy/Voxposer/${RUN_DIR}" ]; then
        echo ""
        echo "== auto-merge (exit_code=$rc) =="
        for c in robocasa-eval-0 robocasa-eval-1 robocasa-eval-2; do
            if docker ps --format '{{.Names}}' | grep -qx "$c"; then
                docker exec "$c" python3 /workspace/scripts/merge_workers.py \
                    "/workspace/policy/Voxposer/${RUN_DIR}" || true
                break
            fi
        done
    fi
    exit $rc
}
trap _merge_on_exit EXIT INT TERM

# 1) B-heavy (10 layouts)
run_subset "B" "$B_LAYOUTS" "B-heavy"

# 2) NB-light (3 layouts) — same RUN_DIR; resume logic skips already-done slots
run_subset "NB" "$NB_LAYOUTS" "NB-light"

echo ""
echo "Done. Final results: policy/Voxposer/${RUN_DIR}/results.json"
