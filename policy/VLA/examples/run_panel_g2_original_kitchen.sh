#!/usr/bin/env bash
# G2 positive control — the original `NavigateKitchen` task, all four panel models,
# on any robocasa tree (set OSMA_BENCH_ROBOCASA_ROOT to point at the upstream v2 checkout).
#
# Conditions: 5 layouts × env seed 0..9 = 50 scenes per model, horizon 500,
#   goal radius 0.20 m, `environment` instruction, legacy base controller, stop on env success.
#
# Scheduling: two lanes in parallel (one model = one GPU); sequential inside a lane.
#   lane A (GPU 0, port 8104): pi05_human300 -> xr1
#   lane B (GPU 1, port 8105): groot_n1_5    -> pi05_opentau
#   The slow models go first so both lanes finish at about the same time.
#
# Servers are started by this script on the host (one venv per model) and killed
# afterwards. Rollouts run inside the simulator container.
#
# Pitfalls this script already accounts for — keep them when you derive from it:
#   * `--render-gpu-device-id` is an EGL ordinal, not a physical GPU. Always pass
#     `--render-gpu-physical-index`.
#   * Do not create ANY file under policy/VLA while this runs: the reproducibility
#     gate would reject every remaining rollout in zero seconds.
#   * Direct rollouts need OSMA_BENCH_VLA_RUN_ROOT; it is passed to the container below.
#   * G2 success is `termination.reason == "env_success"` in meta.json, not
#     `evaluation.success` (always False for the original task).
set -u

# ---- Site variables (export before running; see README.md § "Environment variables")
: "${OSMA_BENCH_REPO_ROOT:?export OSMA_BENCH_REPO_ROOT = parent repository root (mounted as /workspace inside the simulator container)}"
: "${OSMA_BENCH_VLA_ROOT:?export OSMA_BENCH_VLA_ROOT = site root that holds venv/ logs/ runs/ sweeps/ scratch/ (outside the repository)}"
: "${HF_HOME:?export HF_HOME = Hugging Face cache holding the pinned checkpoint snapshots}"
CONTAINER=${OSMA_BENCH_SIM_CONTAINER:-robotics-safety-x11}   # simulator container name

V2=${OSMA_BENCH_ROBOCASA_ROOT:-$OSMA_BENCH_REPO_ROOT/benchmark/robocasa}   # tree the rollouts import
OUT=$OSMA_BENCH_VLA_ROOT/runs
LOGD=$OSMA_BENCH_VLA_ROOT/logs
VENV=$OSMA_BENCH_VLA_ROOT/venv
SNAP=$HF_HOME/hub

xr1_path="$SNAP/models--XiaomiRobotics--Xiaomi-Robotics-1-RoboCasa365/snapshots/0d1aa76d0d82debc9b611e4d1e231096434d5be4"
opentau_path="$SNAP/models--TensorAuto--Robocasa_navigatekitchen/snapshots/5deec1b54819ace7c07acadf540460bdb6648715"
rc365="$SNAP/models--robocasa--robocasa365_checkpoints/snapshots/c484448aba1a9b60a04c9b0ca117241518ea69f3"
groot_path="$rc365/gr00t_n1-5/multitask_learning/checkpoint-120000"
h300_path="$rc365/pi05_pretrain_human300/multitask_learning/75000"

server_cmd() { # profile gpu port  -> stdout: the server command line
  local p=$1 g=$2 port=$3
  case $p in
    xr1)           echo "$VENV/xr1/bin/python -u src/vla_server.py --model-profile xr1 --model-path $xr1_path --model-revision 0d1aa76d0d82debc9b611e4d1e231096434d5be4 --device cuda:0 --port $port --require-reproducible-source";;
    pi05_opentau)  echo "$VENV/opentau/bin/python -u src/vla_server.py --model-profile pi05_opentau --model-path $opentau_path --model-revision 5deec1b54819ace7c07acadf540460bdb6648715 --device cuda:0 --port $port --require-reproducible-source";;
    groot_n1_5)    echo "$VENV/groot_n1_5/bin/python -u src/vla_server.py --model-profile groot_n1_5 --model-path $groot_path --model-revision c484448aba1a9b60a04c9b0ca117241518ea69f3 --host 127.0.0.1 --port $port --require-reproducible-source";;
    pi05_human300) echo "$VENV/pi05_human300/bin/python -u src/vla_server.py --model-profile pi05_human300 --model-path $h300_path --model-revision c484448aba1a9b60a04c9b0ca117241518ea69f3 --device cuda:0 --port $port --require-reproducible-source";;
  esac
}

lane() { # lane_id gpu port  profile1 profile2
  local lane=$1 gpu=$2 port=$3; shift 3
  for prof in "$@"; do
    local log="$LOGD/server_${prof}_g2v2_$(date -u +%Y%m%d_%H%M)_gpu${gpu}.log"
    echo "[lane$lane] === $prof start (GPU $gpu, port $port) ==="
    CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false HF_HOME=$HF_HOME \
      bash -c "cd $OSMA_BENCH_REPO_ROOT/policy/VLA && $(server_cmd $prof $gpu $port)" \
      > "$log" 2>&1 &
    local spid=$!

    # wait for /health (up to 180 s)
    local ok=0
    for _ in $(seq 1 90); do
      sleep 2
      curl -s -m 2 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"ok":true' && { ok=1; break; }
      kill -0 $spid 2>/dev/null || break
    done
    if [ $ok -ne 1 ]; then
      echo "[lane$lane] !! $prof server failed to start — log: $log"; kill $spid 2>/dev/null; continue
    fi
    echo "[lane$lane] $prof 서버 ready"

    local i=0 okc=0 failc=0
    for L in ONE_WALL_SMALL L_SHAPED_SMALL U_SHAPED_SMALL G_SHAPED_SMALL G_SHAPED_LARGE; do
      for S in 0 1 2 3 4 5 6 7 8 9; do
        i=$((i+1))
        if docker exec -e CUDA_VISIBLE_DEVICES=$gpu -e MUJOCO_GL=egl -e PYTHONUNBUFFERED=1 \
             -e OSMA_BENCH_ROBOCASA_ROOT=$V2 -e OSMA_BENCH_VLA_RUN_ROOT=$OUT "$CONTAINER" bash -c "
               cd /workspace/policy/VLA && python3 -u src/run_vla_rollout.py \
                 --specific-env NavigateKitchen --layout $L --style MODERN_1 \
                 --seed $S --policy-seed 0 --horizon 500 --stop-on-env-success \
                 --goal-distance-threshold-m 0.20 \
                 --instruction-mode environment \
                 --base-controller legacy --model-profile $prof \
                 --policy-url http://127.0.0.1:$port --render-gpu-physical-index $gpu \
                 --require-reproducible-source --run-tag ${prof}_g2v2_L${L}_s${S} \
                 --out-root $OUT" >/dev/null 2>&1; then
          okc=$((okc+1))
        else
          failc=$((failc+1)); echo "[lane$lane]   FAIL $prof L$L s$S"
        fi
      done
    done
    echo "[lane$lane] === $prof done: ok=$okc fail=$failc / $i ==="
    kill $spid 2>/dev/null; sleep 8
  done
  echo "[lane$lane] LANE DONE"
}

echo "start $(date -u '+%F %T UTC')  ·  robocasa=$V2"
lane A 0 8104 pi05_human300 xr1 &
lane B 1 8105 groot_n1_5 pi05_opentau &
wait
echo "ALL LANES DONE $(date -u '+%F %T UTC')"
