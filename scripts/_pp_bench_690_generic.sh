#!/bin/bash
# Generic 5-layout 6-shard bench, parameterized by model
# Usage: bash _pp_bench_690_generic.sh <cfg_tag> <system_prompt> <few_shot> <model_id>
# Example: bash _pp_bench_690_generic.sh llama31_8b_dd default default meta-llama/Llama-3.1-8B-Instruct
set -u
CFG_TAG=$1
SYS_PROMPT=$2
FEW_SHOT=$3
MODEL=$4

LOG=/tmp/pp_bench_${CFG_TAG}.log
exec >> "$LOG" 2>&1
echo ""
echo "[$(date '+%H:%M:%S')] === pp_bench_${CFG_TAG} START (model=$MODEL) ==="
echo "    config: system_prompt=$SYS_PROMPT few_shot=$FEW_SHOT"
echo "    layouts: 0,2,5,7,8"

REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
OUT=outputs/pp_bench_${CFG_TAG}
PORT=8000

mapfile -t TASK_ARR < /tmp/task_list.txt
N_TASKS=${#TASK_ARR[@]}
[ "$N_TASKS" -ne 150 ] && echo "  ⚠️ task_list.txt has $N_TASKS (expected 150)"
declare -a S_A S_B S_C S_D S_E S_F
S_A=("${TASK_ARR[@]:0:25}"); S_B=("${TASK_ARR[@]:25:25}"); S_C=("${TASK_ARR[@]:50:25}")
S_D=("${TASK_ARR[@]:75:25}"); S_E=("${TASK_ARR[@]:100:25}"); S_F=("${TASK_ARR[@]:125:25}")

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

launch_shard() {
  local idx=$1 L=$2; shift 2
  echo "  [$(date '+%H:%M:%S')] eval-$idx ← L$L ($# tasks)"
  docker exec -d robocasa-eval-$idx bash -c "
    cd /workspace/policy/Voxposer && OMP_NUM_THREADS=8 MUJOCO_GL=egl \
      python3 src/run_LMP.py -m '$MODEL' -p $PORT -w $idx \
      -o $OUT --system-prompt $SYS_PROMPT --few-shot $FEW_SHOT \
      --layout-ids $L --style-ids 3 $* \
      > /tmp/pp_bench_${CFG_TAG}_d${idx}_L${L}.log 2>&1
  "
}

wait_done
for L in 0 2 5 7 8; do
  echo "[$(date '+%H:%M:%S')] --- Wave L$L start ---"
  launch_shard 0 $L "${S_A[@]}"
  launch_shard 1 $L "${S_B[@]}"
  launch_shard 2 $L "${S_C[@]}"
  launch_shard 3 $L "${S_D[@]}"
  launch_shard 4 $L "${S_E[@]}"
  launch_shard 5 $L "${S_F[@]}"
  wait_done
  echo "[$(date '+%H:%M:%S')] --- Wave L$L done ---"
done

ABS="$REPO/policy/Voxposer/$OUT"
total=$(find "$ABS" -name 'run.log' 2>/dev/null | wc -l)
succ=$(find "$ABS" -name 'run.log' 2>/dev/null | xargs grep -l "SUCCESS " 2>/dev/null | wc -l)
echo "[$(date '+%H:%M:%S')] === pp_bench_${CFG_TAG} DONE ==="
echo "    total $total tasks, SUCCESS $succ"
echo "[$(date '+%H:%M:%S')] === pp_bench_${CFG_TAG} report end ==="
