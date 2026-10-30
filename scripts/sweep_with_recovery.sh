#!/usr/bin/env bash
# Resilient sweep launcher: detects EGL framebuffer corruption (#31) and
# silent OOM kills (#32), recovers automatically, resumes from where the
# .done markers say a task was already completed.
#
# Usage:
#   sweep_with_recovery.sh <container> <output_dir> <sweep_args...>
#
# Example:
#   sweep_with_recovery.sh robotics-safety-container /workspace/scripts/__sweep_v5__ \
#       NavigateKitchenCatBlockingRouteA NavigateKitchenCatBlockingRouteD \
#       -m Qwen/Qwen3-14B -p 8002 --layout-ids 0,2,7 --style-ids 0 --max-retries 1

set -u
container="${1:?container name required}"; shift
out_dir="${1:?output dir required}"; shift
sweep_args=("$@")
log="$out_dir/sweep_recovery.log"
# Host path that the container's /workspace is bind-mounted from. Defaults to
# this repo's root (this script lives in scripts/); override if your mount differs.
HOST_REPO_ROOT="${HOST_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
host_log="${out_dir/\/workspace/$HOST_REPO_ROOT}/sweep_recovery.log"
mkdir -p "${host_log%/*}"

# Threshold: minimum host free memory (GB) before launching a new sweep
MIN_FREE_GB=20

wait_for_memory() {
    local needed=$1
    while true; do
        local free_gb=$(free -g | awk '/^Mem:/ {print $7}')
        if [ "$free_gb" -ge "$needed" ]; then
            echo "[$(date +%H:%M:%S)] memory OK: ${free_gb}GB free (need ${needed}GB)" | tee -a "$host_log"
            return
        fi
        echo "[$(date +%H:%M:%S)] memory low: ${free_gb}GB free, waiting 60s..." | tee -a "$host_log"
        sleep 60
    done
}

attempt=0
max_attempts=5
while [ "$attempt" -lt "$max_attempts" ]; do
    attempt=$((attempt + 1))
    echo "[$(date +%H:%M:%S)] === attempt $attempt/$max_attempts ===" | tee -a "$host_log"

    # OOM guard: don't launch if host memory critical
    wait_for_memory $MIN_FREE_GB

    # Launch sweep in background
    sweep_log="$out_dir/sweep_attempt${attempt}.log"
    docker exec -d "$container" bash -c "
        cd /workspace/policy/Voxposer && \
        OMP_NUM_THREADS=4 MUJOCO_GL=egl \
        nohup python3 src/run_LMP.py ${sweep_args[*]} -o $out_dir > $sweep_log 2>&1
    "
    sleep 4
    # Get inner PID
    proc=$(docker exec "$container" bash -c "ps -ef | grep run_LMP | grep -v grep | head -1 | awk '{print \$2}'" 2>/dev/null)
    echo "[$(date +%H:%M:%S)] launched PID=$proc, watching $sweep_log" | tee -a "$host_log"

    # Watch loop
    while true; do
        sleep 30
        # Check process alive
        alive=$(docker exec "$container" bash -c "kill -0 $proc 2>/dev/null && echo yes || echo no")
        if [ "$alive" = "no" ]; then
            echo "[$(date +%H:%M:%S)] process $proc died" | tee -a "$host_log"
            # Diagnose
            host_sweep_log="${sweep_log/\/workspace/$HOST_REPO_ROOT}"
            if grep -q "framebuffer.*not complete" "$host_sweep_log" 2>/dev/null; then
                echo "[$(date +%H:%M:%S)] EGL framebuffer error detected → restarting $container" | tee -a "$host_log"
                docker restart "$container" && sleep 10
                break  # to outer loop, retry attempt
            fi
            # Check OOM via container state
            oom=$(docker inspect "$container" --format '{{.State.OOMKilled}}' 2>/dev/null)
            if [ "$oom" = "true" ]; then
                echo "[$(date +%H:%M:%S)] OOM kill detected → waiting for memory + retry" | tee -a "$host_log"
                sleep 60
                break
            fi
            # Check normal completion
            if grep -q "Results saved to" "$host_sweep_log" 2>/dev/null; then
                echo "[$(date +%H:%M:%S)] sweep completed normally" | tee -a "$host_log"
                exit 0
            fi
            # Unknown death — retry once
            echo "[$(date +%H:%M:%S)] process died with unknown cause, retrying" | tee -a "$host_log"
            break
        fi
    done
done

echo "[$(date +%H:%M:%S)] max attempts reached, giving up" | tee -a "$host_log"
exit 1
