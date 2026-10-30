#!/usr/bin/env bash
# Run all non-Cosmos models, one seed at a time, with VoxPoser/PIVOT shards.
set -euo pipefail
WS=~/workspace/robotics-safety
PORT=${PORT:-8041}; GPU=${GPU:-0}; WORKERS=${WORKERS:-4}
SEEDS=(0 3 16)
MODELS=("gemma4|google/gemma-4-E4B-it|0.85" "qwen35_9b|Qwen/Qwen3.5-9B|0.85" "robobrain|BAAI/RoboBrain2.5-8B-NV|0.50" "qwen36_35b|Qwen/Qwen3.6-35B-A3B-FP8|0.90")
TASK_FILE="$WS/outputs/blocking_tasks_clean2.txt"
wait_current(){ while pgrep -f 'python3 (src/run_LMP.py|run_pivot.py)' >/dev/null; do sleep 60; done; }
prepare(){ grep -v NonBlocking "$TASK_FILE" | grep -E '^NavigateKitchen.*BlockingRoute[A-G]$' | sort -u > "${TASK_FILE}.final"; test "$(wc -l < "${TASK_FILE}.final")" -eq 125; }
serve(){ local m=$1 u=$2; fuser -k "${PORT}/tcp" 2>/dev/null || true; sleep 5; LD_LIBRARY_PATH=~/miniforge3/lib:${LD_LIBRARY_PATH:-} CUDA_VISIBLE_DEVICES=$GPU ~/venvs/vllm30b/bin/vllm serve "$m" --port $PORT --tensor-parallel-size 1 --gpu-memory-utilization "$u" --max-model-len 8192 --trust-remote-code >/tmp/post_eval_${m##*/}.log 2>&1 & until curl -sf http://localhost:$PORT/v1/models >/dev/null; do sleep 10; done; }
run_seed(){ local tag=$1 model=$2 seed=$3 base=/workspace/outputs/${tag}_seed${seed}; for i in 0 1 2 3; do case $i in 0)s=1;e=32;;1)s=33;e=63;;2)s=64;e=94;;3)s=95;e=125;;esac; tasks=$(sed -n "${s},${e}p" "${TASK_FILE}.final" | tr '\n' ' '); docker exec -d robotics-safety-forewarn-g2 bash -lc "mkdir -p ${base}_vox_s$i; cd /workspace/policy/Voxposer; OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MUJOCO_GL=egl NAV_MODE=holonomic ROBOCASA_LEDGER_DIR=${base}_vox_s${i}_ledger ROBOCASA_SEED=$seed python3 src/run_LMP.py $tasks -m $model -p $PORT --layout-ids 0,2,5,7,8 --style-ids 3 --seed $seed -w $i -o ${base}_vox_s$i > ${base}_vox_s$i/worker.log 2>&1"; docker exec -d robotics-safety-forewarn-g3 bash -lc "mkdir -p ${base}_piv_s$i; cd /workspace/policy/PIVOT; OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MUJOCO_GL=egl NAV_MODE=holonomic ROBOCASA_LEDGER_DIR=${base}_piv_s${i}_ledger ROBOCASA_SEED=$seed python3 run_pivot.py $tasks -m $model -p $PORT --layout-ids 0,2,5,7,8 --style-ids 3 --seed $seed --subgoals pivot -o ${base}_piv_s$i > ${base}_piv_s$i/worker.log 2>&1"; done; while docker exec robotics-safety-forewarn-g2 pgrep -f 'python3 src/run_LMP.py' >/dev/null || docker exec robotics-safety-forewarn-g3 pgrep -f 'python3 run_pivot.py' >/dev/null; do sleep 60; done; python "$WS/scripts/merge_seed_shards.py" --root "$WS/outputs" --prefix "${tag}_vox" --seed "$seed" --dest "$WS/outputs/${tag}_seed${seed}_voxposer"; python "$WS/scripts/merge_seed_shards.py" --root "$WS/outputs" --prefix "${tag}_piv" --seed "$seed" --dest "$WS/outputs/${tag}_seed${seed}_pivot"; }
wait_current; prepare
# Complete every model for one seed before advancing to the next seed.
for seed in "${SEEDS[@]}"; do
  for spec in "${MODELS[@]}"; do
    IFS='|' read -r tag model util <<<"$spec"
    serve "$model" "$util"
    run_seed "$tag" "$model" "$seed"
  done
done
