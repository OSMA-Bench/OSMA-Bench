# Getting started

Set up OSMA-Bench and run one task end to end. Clone the repository first
([README](../README.md#clone)).

## Build and start the container

From the repository root:

```bash
docker build -f docker/Dockerfile -t robotics-safety:latest .
bash docker/run_docker.sh
```

`run_docker.sh` opens a shell in `robotics-safety-container` (all GPUs, host network, repository at
`/workspace`, host `~/.cache/huggingface` at `/root/.cache/huggingface`), replacing a running
container of that name.
More shells: `docker exec -it robotics-safety-container bash`.

The image renders headless (`MUJOCO_GL=egl`) and pins `vllm==0.11.2` and `numpy==1.26.4` so that
RoboCasa and vLLM share one environment; newer models may need a newer vLLM.

## Download assets and check the simulator

Once, inside the container:

```bash
python3 benchmark/robocasa/robocasa/scripts/download_kitchen_assets.py
python3 benchmark/robocasa/robocasa/scripts/setup_macros.py
python3 benchmark/robocasa/check_running.py   # random rollout, prints "It works!"
```

## Serve a VLM

For VoxPoser and PIVOT, in a second container shell:

```bash
python3 -m vllm.entrypoints.openai.api_server --model <hf model id> --port 8003 \
  --max-model-len 16384 --gpu-memory-utilization 0.85 \
  --trust-remote-code --limit-mm-per-prompt '{"image": 5}'
```

Ready when `curl -sf http://localhost:8003/health` succeeds. A runner's `-m` is the served
name (`--model` unless `--served-model-name` is set). VLAs use their own servers
([Policies](policies.md#vla-policies)).

## Smoke run

The dog on Route C, layout 0, with VoxPoser:

```bash
cd /workspace/policy/Voxposer
MUJOCO_GL=egl NAV_MODE=holonomic ROBOCASA_SEED=0 \
ROBOCASA_LEDGER_DIR=/workspace/outputs/smoke/ledger \
  python3 src/run_LMP.py NavigateKitchenDogBlockingRouteC \
    -m <hf model id> -p 8003 --layout-ids 0 --style-ids 3 --seed 0 \
    -o /workspace/outputs/smoke
```

`/workspace/outputs/smoke/ledger/episodes.jsonl` then holds one row with `task_success` and
`collision_free_success`, and `ledger/run.json` shows `"nav_mode": "holonomic"`.

## Full evaluation

1. Write the 125 benchmark classes to `blocking_tasks.txt` ([Benchmark](benchmark.md#tasks-in-code)).
2. Run every model for seeds {0, 3, 16} ([Policies](policies.md)).
3. Score each seed; every model should have 625 episodes
   ([Metrics](metrics.md#computing-the-metrics)).
4. Run the RQ2 and RQ3 conditions ([Experiments](experiments.md)).

## Where results go

A merged VoxPoser or PIVOT folder:

```
outputs/<model>_seed<S>_voxposer/
├── layout{0,2,5,7,8}/<task class>/    # trajectory_log.json, run.log, videos
├── results.json, results_progress.jsonl
├── merge_manifest.json                # which worker folders were merged
└── ledger/
    ├── episodes.jsonl                 # one row per task
    ├── traj/l<layout>_s<style>_<task class>_seed<S>.npz   # per-step trajectory
    └── run.json                       # policy, model, nav_mode, seed, code commits
```

VLA runs write the same ledger to `--ledger-dir`. The metrics read only the ledger.
