#!/bin/bash
# Run navigation eval in parallel across multiple Docker containers, split by route.
#
# Default split (when WORKERS=3) is:
#   Worker 0: Route A, B, none (33 tasks)
#   Worker 1: Route C, D, E    (48 tasks)
#   Worker 2: Route F, G       (30 tasks)
# With other WORKERS values, tasks are round-robin split across the workers.
#
# All workers share the same vLLM server (localhost:${PORT}).
# All workers mount the same workspace; outputs are timestamped so they don't collide.

set -e

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
PORT="${PORT:-8002}"
IMAGE="robotics-safety:committed"
WORKERS="${WORKERS:-6}"
WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
LAYOUT_IDS="${LAYOUT_IDS:-all}"   # 'all' = 0..9, or a comma-separated list e.g. '0,3'
STYLE_IDS="${STYLE_IDS:-3}"        # default style 3
MAX_TASKS="${MAX_TASKS:-}"         # optional cap for smoke runs

# Auto-merge worker outputs on ANY exit path (success / error / Ctrl-C).
# RUN_DIR is set later in step 3, so we guard against the unset case.
_merge_on_exit() {
    local rc=$?
    if [ -n "${RUN_DIR:-}" ] && [ -d "${WORKSPACE}/policy/Voxposer/${RUN_DIR}" ]; then
        echo ""
        echo "=== auto-merge worker artefacts (exit_code=$rc) ==="
        # Run inside one of the eval containers (which already has access to
        # utils.ssi via PYTHONPATH); pick the first that's still up.
        for c in "${CONTAINERS[@]}"; do
            if docker ps --format '{{.Names}}' | grep -qx "$c"; then
                docker exec "$c" python3 /workspace/scripts/merge_workers.py \
                    "/workspace/policy/Voxposer/${RUN_DIR}" || \
                    echo "  merge_workers.py failed (exit code = $?)"
                break
            fi
        done
    fi
    exit $rc
}
trap _merge_on_exit EXIT INT TERM

# Worker container names (dynamic)
CONTAINERS=()
for i in $(seq 0 $((WORKERS-1))); do CONTAINERS+=("robocasa-eval-$i"); done

get_all_tasks() {
    local container="${CONTAINERS[0]}"
    docker exec -e STYLES="${STYLES:-}" "$container" python3 -c "
import sys, os, io, warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
_real_stdout = sys.stdout
sys.stdout = io.StringIO()
sys.path.insert(0, '/workspace/benchmark/robocasa/robocasa')
from robocasa.utils.result_utils import get_navigate_tasks
sys.stdout = _real_stdout
tasks = get_navigate_tasks()
styles_env = os.environ.get('STYLES', '').strip()
style_ids = [int(s) for s in styles_env.split(',')] if styles_env else [None]
for t in tasks:
    for sid in style_ids:
        print(t if sid is None else f'{t}#style{sid}')
" 2>/dev/null
}

# Step 1: Create worker containers (if not already running)
echo "=== Creating worker containers ==="
for i in $(seq 0 $((WORKERS-1))); do
    name="${CONTAINERS[$i]}"

    # Remove existing container
    if docker ps -aq -f name="^${name}$" | grep -q .; then
        echo "Removing existing container: $name"
        docker rm -f "$name" > /dev/null 2>&1
    fi

    echo "Creating container: $name"
    docker run -d \
        --name "$name" \
        --gpus all \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e MUJOCO_GL=egl \
        -e PYOPENGL_PLATFORM=egl \
        -e __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
        -v /usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d:ro \
        -v "${HOME}/.cache/huggingface:/workspace/.cache/huggingface" \
        -v "$WORKSPACE":/workspace \
        --workdir /workspace \
        --net=host \
        --privileged \
        "$IMAGE" \
        sleep infinity
    echo "  $name started"
done

# Wait for containers to fully initialize before running docker exec
sleep 10

echo ""

# Step 2: Get task lists (round-robin split across workers)
echo "=== Task assignments ==="
ALL_TASKS=$(get_all_tasks)
TOTAL=$(echo "$ALL_TASKS" | wc -l)
TASK_LISTS=()
for i in $(seq 0 $((WORKERS-1))); do
    tasks=$(echo "$ALL_TASKS" | awk -v i="$i" -v W="$WORKERS" '(NR-1)%W==i')
    count=$(echo "$tasks" | wc -l)
    TASK_LISTS+=("$tasks")
    echo "Worker $i (${CONTAINERS[$i]}): $count tasks"
done
echo "Total: $TOTAL tasks"

echo ""

# Step 3: Create shared output directory and launch eval
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||; s/-/_/g')
RUN_DIR="outputs/navigation_${MODEL_SHORT}_${TIMESTAMP}"

echo "=== Shared output directory: $RUN_DIR ==="
echo ""

echo "=== Launching evaluations ==="
for i in $(seq 0 $((WORKERS-1))); do
    name="${CONTAINERS[$i]}"
    tasks="${TASK_LISTS[$i]}"
    # Convert newlines to spaces for CLI args
    task_args=$(echo "$tasks" | tr '\n' ' ')

    extra_args=""
    [ -n "$MAX_TASKS" ] && extra_args="--max-tasks $MAX_TASKS"
    echo "Starting worker $i ($name)..."
    docker exec -d "$name" bash -c \
        "cd /workspace/policy/Voxposer && OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 MUJOCO_GL=egl python3 src/run_LMP.py -m $MODEL -p $PORT -w $i -o $RUN_DIR --layout-ids $LAYOUT_IDS --style-ids $STYLE_IDS $extra_args $task_args > /tmp/eval_worker_${i}.log 2>&1"
done

echo ""
echo "=== All workers launched ==="
echo ""

# Step 4: Wait for all workers to finish
echo "Waiting for all workers to complete..."
echo "  (check progress: bash scripts/check_parallel_status.sh)"
echo ""

while true; do
    done_count=0
    for i in $(seq 0 $((WORKERS-1))); do
        name="${CONTAINERS[$i]}"
        procs=$(docker exec "$name" pgrep -f "run_LMP" 2>/dev/null | wc -l)
        if [ "$procs" -eq 0 ]; then
            done_count=$((done_count + 1))
        fi
    done
    if [ "$done_count" -eq "$WORKERS" ]; then
        break
    fi
    sleep 30
done

echo "=== All workers finished at $(date) ==="
echo ""

# Step 5: explicit success-path merge happens via the EXIT trap above too,
# so nothing extra is needed here. The trap fires whether we exit normally,
# fail mid-run, or get Ctrl-C'd.
echo ""
echo "Done! Results: policy/Voxposer/$RUN_DIR/results.json"
