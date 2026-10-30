#!/usr/bin/env bash
# One FOREWARN-steered rollout (src/configs/steering/forewarn.yaml) for any model profile.
#   CONTAINER=robotics-safety-forewarn-g3 GPU=0 PORT=8105 PROFILE=pi05_opentau TASK=NavigateKitchenDogBlockingRouteA \
#     examples/run_forewarn.sh
# The policy server (src/vla_server.py) on PORT and the verifier in the yaml must already be up.
# Models without "noise_scale" in /meta act_sampling_keys need sampling.shared_noise.noise_scale: 1.0 (STEERING=<copy>).
set -eu
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TREE=/workspace/${HERE#"$(cd "$HERE/../.." && pwd)"/}   # the repo root is mounted as /workspace
: "${CONTAINER:?}" "${GPU:?}" "${PORT:?}" "${PROFILE:?}" "${TASK:?}"
LAYOUT=${LAYOUT:-ONE_WALL_SMALL}
STEERING=${STEERING:-$TREE/src/configs/steering/forewarn.yaml}
OUT=${OUT:-/workspace/outputs/vla_forewarn_demo}
docker exec -e CUDA_VISIBLE_DEVICES="$GPU" -e MUJOCO_GL=egl -e PYTHONUNBUFFERED=1 -e OSMA_BENCH_VLA_RUN_ROOT="$OUT" \
  "$CONTAINER" bash -c "cd $TREE && python3 -u src/run_vla_rollout.py \
    --specific-env $TASK --layout $LAYOUT --style MODERN_1 --seed 0 --policy-seed 0 --horizon 500 \
    --goal-distance-threshold-m 0.60 --instruction-mode environment --base-controller legacy \
    --action-bound-mode official_passthrough --stop-on-env-success \
    --model-profile $PROFILE --policy-url http://127.0.0.1:$PORT --render-gpu-physical-index $GPU \
    --steering-config $STEERING --run-tag forewarn --out-root $OUT/$PROFILE/$TASK"
