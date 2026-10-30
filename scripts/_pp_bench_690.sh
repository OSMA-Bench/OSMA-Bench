#!/bin/bash
# Generic 5-layout 6-shard bench for prompt ablation (L0/L2/L5/L7/L8 only)
# Usage: bash _pp_bench_690.sh <cfg_name> <system_prompt> <few_shot>
# Example: bash _pp_bench_690.sh dsa default safety_aware
#          bash _pp_bench_690.sh sad safety_aware default
#
# Structure: 5 sequential waves (one per layout), each wave = 6-shard parallel
#            shard = 25 tasks (out of 150 task list, ~138 valid after Person skip)
# Total per config: 5 layouts × 138 valid ≈ 690 task
#
# NOTE: Uses Voxposer (NOT Voxposer_ppdebug) — official path after 2026-05-24 reorg.
# NOTE: PPENV omitted because defaults in interfaces.py now match best config
#       (LH=0.5, replan_max=4, goal_tol=0.25, decel_r=0.4, turn_slow=100,
#        stuck_goal_r=0.6, escape_enabled=1, NAV_MODE=pure_pursuit,
#        START_CLEAR_CELLS=2, ROBOT_FOOTPRINT_Z_MAX=0.6).
set -u
CFG_NAME=$1
SYS_PROMPT=$2
FEW_SHOT=$3

LOG=/tmp/pp_bench_qwen3_14b_${CFG_NAME}.log
exec >> "$LOG" 2>&1
echo ""
echo "[$(date '+%H:%M:%S')] === pp_bench_qwen3_14b_${CFG_NAME} START ==="
echo "    config: system_prompt=$SYS_PROMPT few_shot=$FEW_SHOT"
echo "    layouts: 0,2,5,7,8 (5 layouts × 6 shards × 25 task = 690 valid est)"

REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
OUT=outputs/pp_bench_qwen3_14b_${CFG_NAME}
MODEL="Qwen/Qwen3-14B"
PORT=8000

# Load 150-task list
mapfile -t TASK_ARR < /tmp/task_list.txt
N_TASKS=${#TASK_ARR[@]}
if [ "$N_TASKS" -ne 150 ]; then
  echo "  ⚠️ task_list.txt has $N_TASKS tasks (expected 150)"
fi
declare -a SHARD_A SHARD_B SHARD_C SHARD_D SHARD_E SHARD_F
SHARD_A=("${TASK_ARR[@]:0:25}")
SHARD_B=("${TASK_ARR[@]:25:25}")
SHARD_C=("${TASK_ARR[@]:50:25}")
SHARD_D=("${TASK_ARR[@]:75:25}")
SHARD_E=("${TASK_ARR[@]:100:25}")
SHARD_F=("${TASK_ARR[@]:125:25}")

wait_for_vllm() {
  for i in $(seq 1 90); do
    curl -s -m 5 http://localhost:${PORT}/v1/models 2>/dev/null | grep -q '"id"' && return
    sleep 10
  done
  echo "  ⚠️ vLLM not ready after 15min"
  return 1
}

wait_done() {
  sleep 30
  while :; do
    n=0
    for d in eval-0 eval-1 eval-2 eval-3 eval-4 eval-5; do
      docker exec robocasa-$d pgrep -f run_LMP >/dev/null 2>&1 && n=$((n+1))
    done
    [ "$n" -eq 0 ] && return
    sleep 60
  done
}

# Voxposer (no PPENV — defaults match best config)
launch_shard() {
  local idx=$1 L=$2; shift 2
  echo "  [$(date '+%H:%M:%S')] eval-$idx ← L$L ($# tasks)"
  docker exec -d robocasa-eval-$idx bash -c "
    cd /workspace/policy/Voxposer && OMP_NUM_THREADS=8 MUJOCO_GL=egl \
      python3 src/run_LMP.py -m '$MODEL' -p $PORT -w $idx \
      -o $OUT --system-prompt $SYS_PROMPT --few-shot $FEW_SHOT \
      --layout-ids $L --style-ids 3 $* \
      > /tmp/pp_bench_${CFG_NAME}_d${idx}_L${L}.log 2>&1
  "
}

wait_for_vllm
wait_done   # ensure no leftover procs from previous run

# 5 sequential waves: L0, L2, L5, L7, L8
for L in 0 2 5 7 8; do
  echo "[$(date '+%H:%M:%S')] --- Wave L$L start ---"
  launch_shard 0 $L "${SHARD_A[@]}"
  launch_shard 1 $L "${SHARD_B[@]}"
  launch_shard 2 $L "${SHARD_C[@]}"
  launch_shard 3 $L "${SHARD_D[@]}"
  launch_shard 4 $L "${SHARD_E[@]}"
  launch_shard 5 $L "${SHARD_F[@]}"
  wait_done
  echo "[$(date '+%H:%M:%S')] --- Wave L$L done ---"
done

# Final count
ABS="$REPO/policy/Voxposer/$OUT"
total=$(find "$ABS" -name 'run.log' 2>/dev/null | wc -l)
succ=$(find "$ABS" -name 'run.log' 2>/dev/null | xargs grep -l "SUCCESS " 2>/dev/null | wc -l)
echo "[$(date '+%H:%M:%S')] === pp_bench_qwen3_14b_${CFG_NAME} DONE ==="
echo "    total $total tasks, SUCCESS $succ"
echo "[$(date '+%H:%M:%S')] === pp_bench_qwen3_14b_${CFG_NAME} report end ==="
