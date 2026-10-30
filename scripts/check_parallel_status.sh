#!/bin/bash
# Check progress of parallel eval workers

WORKSPACE="${ROBOTICS_SAFETY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
CONTAINERS=(robocasa-eval-0 robocasa-eval-1 robocasa-eval-2)
ROUTES=("A,B,none" "C,D,E" "F,G")

echo "=== Parallel Eval Status ($(date)) ==="
echo ""

# Find latest shared output directory
RUN_DIR=$(ls -dt "${WORKSPACE}/policy/Voxposer/outputs/navigation_"* 2>/dev/null | head -1)
if [ -n "$RUN_DIR" ]; then
    echo "Output: $(basename "$RUN_DIR")"
    echo ""
fi

total_done=0
total_tasks=111

for i in 0 1 2; do
    name="${CONTAINERS[$i]}"

    # Check if container is running
    if ! docker ps -q -f name="^${name}$" | grep -q .; then
        echo "Worker $i ($name): NOT RUNNING"
        continue
    fi

    # Check if process is alive
    procs=$(docker exec "$name" pgrep -f "run_LMP" 2>/dev/null | wc -l)

    # Get accumulated count from worker-specific log
    count=0
    if [ -n "$RUN_DIR" ]; then
        count=$(grep -c "Accumulated" "$RUN_DIR/run_output_w${i}.txt" 2>/dev/null || echo 0)
        latest=$(grep "Accumulated" "$RUN_DIR/run_output_w${i}.txt" 2>/dev/null | tail -1 | sed 's/\x1b\[[0-9;]*m//g')
    fi

    status="RUNNING"
    if [ "$procs" -eq 0 ]; then
        status="FINISHED"
    fi

    echo "Worker $i ($name) [Routes: ${ROUTES[$i]}] — $status"
    echo "  Progress: $count tasks done"
    if [ -n "$latest" ]; then
        echo "  Latest: $latest"
    fi
    echo ""

    total_done=$((total_done + count))
done

echo "Total: $total_done / $total_tasks"
