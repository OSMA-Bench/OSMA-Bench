#!/usr/bin/env bash
# Full OSMA-Bench sweep template — one model, paper layouts 0,2,5,7,8 × every Navigate* task,
# two workers. Shown for π0.5-Human300 (chunk 50 / replan 5); change PROFILE/REPLAN/PORT0
# for another panel model (native replan: xr1 16, pi05_opentau 20, groot_n1_5 16).
#
# The model servers are NOT started here — start one per worker first (see README.md
# § "Model servers"), on ports $PORT0 and $PORT0+1, one GPU each.
#
# Sweep size: 138 tasks × 5 layouts = 690 instances on the inline (pre-v2) tree,
#             250 tasks × 5 layouts = 1,250 on the upstream v2 tree — name the sweep
#             directory so the two can never be confused.
#
# Pitfalls this script already accounts for — keep them when you derive from it:
#   * `--render-gpu-device-id` is an EGL ordinal; always pass `--render-gpu-physical-index`.
#   * Do not create ANY file under policy/VLA while this runs (reproducibility gate).
#   * Run `--finalize` inside the container afterwards (the sweep dir is container-owned):
#       docker exec $CONTAINER bash -lc "cd /workspace/policy/VLA && python3 src/run_vla_eval.py --out-dir $OUT --finalize"
set -u

# ---- Site variables (export before running; see README.md § "Environment variables")
: "${OSMA_BENCH_REPO_ROOT:?export OSMA_BENCH_REPO_ROOT = parent repository root (mounted as /workspace inside the simulator container)}"
: "${OSMA_BENCH_VLA_ROOT:?export OSMA_BENCH_VLA_ROOT = site root that holds venv/ logs/ runs/ sweeps/ scratch/ (outside the repository)}"
: "${HF_HOME:?export HF_HOME = Hugging Face cache holding the pinned checkpoint snapshots}"
CONTAINER=${OSMA_BENCH_SIM_CONTAINER:-robotics-safety-x11}   # simulator container name

PROFILE=${PROFILE:-pi05_human300}
REPLAN=${REPLAN:-5}
PORT0=${PORT0:-8104}
V2=${OSMA_BENCH_ROBOCASA_ROOT:-$OSMA_BENCH_REPO_ROOT/benchmark/robocasa}   # tree the rollouts import
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$OSMA_BENCH_VLA_ROOT/sweeps/${TS}__${PROFILE}__osmabench__seed0__h500__r${REPLAN}
mkdir -p "$OSMA_BENCH_VLA_ROOT/scratch" && echo "$OUT" > "$OSMA_BENCH_VLA_ROOT/scratch/.last_full_sweep"
echo "sweep: $OUT"

for W in 0 1; do
  docker exec -i -e CUDA_VISIBLE_DEVICES=$W -e MUJOCO_GL=egl -e PYTHONUNBUFFERED=1 \
    -e OSMA_BENCH_ROBOCASA_ROOT=$V2 "$CONTAINER" bash -lc "
      cd /workspace/policy/VLA && python3 -u src/run_vla_eval.py \
        --out-dir $OUT \
        --layout-ids 0,2,5,7,8 --style MODERN_1 \
        --seed 0 --policy-seed 0 --horizon 500 \
        --goal-distance-threshold-m 0.60 \
        --instruction-mode environment \
        --base-controller legacy --action-bound-mode official_passthrough \
        --stop-on-env-success \
        --replan-steps $REPLAN \
        --model-profile $PROFILE \
        --policy-url http://127.0.0.1:$((PORT0 + W)) \
        --render-gpu-physical-index $W \
        --worker-id $W --num-workers 2 \
        --require-reproducible-source" &
done
wait
echo "SWEEP WORKERS DONE — now run --finalize inside the container (see header)"
