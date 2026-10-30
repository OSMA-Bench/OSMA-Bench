#!/bin/bash
# Prompt ablation chain: Qwen Wave 2 (dd) done → migrate ppdebug→Voxposer
#   → d-sa 690-task → sa-d 690-task
# Uses existing Qwen3-14B vLLM (no swap). Replaces _phase4_chain.sh.
# Usage: nohup bash _ablation_chain.sh > /tmp/ablation_chain.log 2>&1 &
set -u
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
LOG=/tmp/ablation_chain.log
exec >> "$LOG" 2>&1
echo "[$(date '+%H:%M:%S')] === Ablation chain start ==="

# Step 1: Wait for current Qwen Wave 2 (dd) to finish (using ppdebug)
echo "[$(date '+%H:%M:%S')] Step 1: waiting for Qwen Wave 2 (dd) report end..."
while ! grep -q "report end" /tmp/pp_bench_qwen3_14b.log 2>/dev/null; do
  sleep 60
done
echo "[$(date '+%H:%M:%S')] Qwen Wave 2 (dd) done."

# Step 2: Verify all workers exited (ppdebug no longer in use)
sleep 10
for d in eval-0 eval-1 eval-2 eval-3 eval-4 eval-5; do
  if docker exec robocasa-$d pgrep -f run_LMP >/dev/null 2>&1; then
    echo "[$(date '+%H:%M:%S')] ⚠️ robocasa-$d still has run_LMP — kill remnants"
    docker exec robocasa-$d pkill -f run_LMP 2>&1
  fi
done
sleep 5

# Step 3: Migrate ppdebug outputs to Voxposer, then remove ppdebug
echo "[$(date '+%H:%M:%S')] Step 3: migrating Voxposer_ppdebug → Voxposer..."
PPDEBUG="$REPO/policy/Voxposer_ppdebug"
VOXPOSER="$REPO/policy/Voxposer"
if [ -d "$PPDEBUG" ]; then
  # Preserve outputs (Phase 4 dd Wave 1+2 results) — merge into Voxposer/outputs/
  if [ -d "$PPDEBUG/outputs" ]; then
    mkdir -p "$VOXPOSER/outputs"
    # Use Docker container with root permissions (host may not own these dirs)
    docker exec robocasa-eval-0 bash -c "
      cp -rn /workspace/policy/Voxposer_ppdebug/outputs/* /workspace/policy/Voxposer/outputs/ 2>&1 || true
    "
    echo "  → outputs/ merged"
  fi
  # Preserve LMP cache
  if [ -d "$PPDEBUG/cache" ]; then
    docker exec robocasa-eval-0 bash -c "
      cp -rn /workspace/policy/Voxposer_ppdebug/cache/* /workspace/policy/Voxposer/cache/ 2>&1 || true
    " || true
    echo "  → cache/ merged"
  fi
  # Move ppdebug to backup (don't delete — keep for safety)
  BACKUP="${PPDEBUG}_backup_$(date +%Y%m%d_%H%M)"
  mv "$PPDEBUG" "$BACKUP" 2>&1 || {
    echo "  ⚠️ mv ppdebug failed (likely permissions); leaving as-is"
  }
  echo "  → ppdebug backed up to $(basename $BACKUP)"
else
  echo "  → ppdebug already gone, skipping"
fi
echo "[$(date '+%H:%M:%S')] Migration done."

# Step 4: Verify vLLM still alive
if ! curl -s -m 5 http://localhost:8000/v1/models 2>/dev/null | grep -q '"id"'; then
  echo "[$(date '+%H:%M:%S')] ⚠️ vLLM port 8000 unreachable — abort"
  exit 1
fi
echo "[$(date '+%H:%M:%S')] vLLM OK, ready for ablation"

# Step 5: Run d-sa (system=default + few-shot=safety_aware)
echo "[$(date '+%H:%M:%S')] Step 5: launching d-sa bench (Voxposer path)..."
bash "$REPO/scripts/_pp_bench_690.sh" dsa default safety_aware
echo "[$(date '+%H:%M:%S')] d-sa bench script returned"

# Step 6: Verify d-sa wrote completion marker
sleep 5
if ! grep -q "report end" /tmp/pp_bench_qwen3_14b_dsa.log 2>/dev/null; then
  echo "[$(date '+%H:%M:%S')] ⚠️ d-sa did not finish cleanly — checking anyway"
fi

# Step 7: Verify vLLM still alive
if ! curl -s -m 5 http://localhost:8000/v1/models 2>/dev/null | grep -q '"id"'; then
  echo "[$(date '+%H:%M:%S')] ⚠️ vLLM died during d-sa — abort"
  exit 1
fi

# Step 8: Run sa-d (system=safety_aware + few-shot=default)
echo "[$(date '+%H:%M:%S')] Step 8: launching sa-d bench (Voxposer path)..."
bash "$REPO/scripts/_pp_bench_690.sh" sad safety_aware default
echo "[$(date '+%H:%M:%S')] sa-d bench script returned"

echo "[$(date '+%H:%M:%S')] === Ablation chain ALL DONE ==="
