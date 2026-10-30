# Policies

Three **policy groups** run in closed loop (paper §4.1, App. D): VLAs, and VLMs driving VoxPoser
[[1]](#references) or PIVOT [[2]](#references). Commands run in the container
([Getting started](getting_started.md)) with the repository at `/workspace`; `blocking_tasks.txt`
holds the 125 benchmark classes ([Benchmark](benchmark.md#tasks-in-code)).

## VLA policies

Released RoboCasa checkpoints of four VLAs, weights fixed. Each is a profile in
`policy/VLA/src/configs/models.yaml`:

| Policy | Profile | Actions predicted / executed per query | Server port |
|---|---|---|---:|
| XR-1 (Xiaomi-Robotics-1) [[3]](#references) | `xr1` | 16 / 16 | 8100 |
| π0.5-OpenTau [[4]](#references) | `pi05_opentau` | 20 / 20 | 8101 |
| GR00T N1.5 [[5, 6]](#references) | `groot_n1_5_atomic_seen` | 16 / 16 | 8102 |
| π0.5-Human300 [[7]](#references) | `pi05_human300` | 50 / 50 with `--replan-steps 50` (profile default 5; paper App. D.1) | 8103 |

- **Input:** instruction c1 ([Experiments](experiments.md)), left, right and wrist RGB at 256 × 256,
  and proprioception; no top-down image, obstacle position or tier label.
- **Output:** 12-D actions through the `HYBRID_MOBILE_BASE` composite controller with
  `--base-controller legacy`, at 20 Hz.

Serve each model from its own environment (`policy/VLA/README.md` §2.2; GR00T N1.5 with the
`groot_n1_5_atomic_seen` profile). One seed of one model:

```bash
python3 policy/VLA/src/run_vla_eval.py \
  --model-profile xr1 --policy-url http://127.0.0.1:8100 \
  --tasks $(cat blocking_tasks.txt) --layout-ids 0,2,5,7,8 --style MODERN_1 \
  --seed 0 --policy-seed 0 --horizon 500 \
  --out-dir /workspace/outputs/xr1_seed0 --ledger-dir /workspace/outputs/xr1_seed0/ledger
```

`--num-workers N --worker-id i` splits a seed over workers. Without `--tasks` the runner also takes
navigation classes outside OSMA-Bench.

## VLM-based policies

| VLM | Served with |
|---|---|
| Qwen3.6-35B-A3B, Qwen3.5-4B, Qwen3.5-9B [[8]](#references); Gemma-4-E4B-it [[9]](#references); Cosmos-Reason2-8B, Cosmos-Reason2-32B [[10]](#references); RoboBrain2.5-8B-NV [[11]](#references) | vLLM, bf16, one GPU per replica ([Serve a VLM](getting_started.md#serve-a-vlm)) |
| Gemma-4-12B-it [[9]](#references) | Hugging Face Transformers; the server is *not yet in the repository* |

A runner reaches the model's OpenAI-compatible endpoint through `-m <served model name> -p <port>`.

- **VoxPoser** [[1]](#references) (`policy/Voxposer`, code generation): the VLM writes code that builds value maps and
  sets obstacle-specific clearance and velocity; an A* planner turns the maps into a path.
- **PIVOT** [[2]](#references) (`policy/PIVOT`, waypoint selection): the VLM picks the next waypoint among candidate
  points marked on the scene image, over several rounds.

Both drive the base through a **shared holonomic waypoint follower** (robosuite's 11-D `BASIC`
action). `NAV_MODE=holonomic` selects it over `navigation_controller.mode` in
`policy/Voxposer/src/configs/robocasa_config.yaml`, and `run.json` records it as `nav_mode`.

One seed of one model. Worker `i` runs a slice of `blocking_tasks.txt`; the folder names are what
`scripts/merge_seed_shards.py` looks for:

```bash
OUT=/workspace/outputs; M=<model tag>; S=0; i=0; TASKS="<slice i of blocking_tasks.txt>"

# VoxPoser
cd /workspace/policy/Voxposer && MUJOCO_GL=egl NAV_MODE=holonomic \
  ROBOCASA_SEED=$S ROBOCASA_LEDGER_DIR=$OUT/${M}_seed${S}_vox_s${i}_ledger \
  python3 src/run_LMP.py $TASKS -m <served model name> -p <port> \
    --layout-ids 0,2,5,7,8 --style-ids 3 --seed $S -w $i -o $OUT/${M}_seed${S}_vox_s${i}

# PIVOT
cd /workspace/policy/PIVOT && MUJOCO_GL=egl NAV_MODE=holonomic \
  ROBOCASA_SEED=$S ROBOCASA_LEDGER_DIR=$OUT/${M}_seed${S}_piv_s${i}_ledger \
  python3 run_pivot.py $TASKS -m <served model name> -p <port> \
    --layout-ids 0,2,5,7,8 --style-ids 3 --seed $S --subgoals pivot -o $OUT/${M}_seed${S}_piv_s${i}

# after all workers finish: one folder per model, seed and policy
python3 /workspace/scripts/merge_seed_shards.py --root $OUT --prefix $M --seed $S \
  --suffix vox --dest $OUT/${M}_seed${S}_voxposer
python3 /workspace/scripts/merge_seed_shards.py --root $OUT --prefix $M --seed $S \
  --suffix piv --dest $OUT/${M}_seed${S}_pivot
```

- `ROBOCASA_SEED` sets the environment seed (42 when unset) and `--seed` the policy's sampling seed;
  the paper sets both to the same value.
- PIVOT needs `--layout-ids` and `--subgoals pivot`; its defaults are layout 0 and `solver` subgoals.

## All policies

- A task ends at task success or when the step budget runs out: `--horizon` control steps for VLAs,
  `max_steps_per_waypoint` (`robocasa_config.yaml`) per waypoint for the planners.
- Re-running a command with the same output folder (and worker id, where there is one) skips the
  tasks it has already finished.
- Score the ledgers with [Metrics](metrics.md#computing-the-metrics).

## References

1. W. Huang et al. VoxPoser: Composable 3D value maps for robotic manipulation with language models.
   *CoRL*, 2023.
2. S. Nasiriany et al. PIVOT: Iterative visual prompting elicits actionable knowledge for VLMs.
   *ICML*, 2024.
3. Xiaomi Robotics Team et al. Xiaomi-Robotics-1: Scaling vision-language-action models with over
   100k hours of real-world trajectories. arXiv:2607.15330, 2026.
4. Tensor. OpenTau: Tensor's VLA training infrastructure for real-world robotics in PyTorch.
   <https://github.com/TensorAuto/OpenTau>, 2025.
5. NVIDIA et al. GR00T N1: An open foundation model for generalist humanoid robots.
   arXiv:2503.14734, 2025.
6. NVIDIA. GR00T N1.5: An improved open foundation model for generalist humanoid robots.
   <https://research.nvidia.com/labs/gear/gr00t-n1_5/>, 2025.
7. Physical Intelligence et al. π0.5: A vision-language-action model with open-world
   generalization. *CoRL*, 2025.
8. S. Bai et al. Qwen3-VL technical report. arXiv:2511.21631, 2025.
9. Gemma Team. Gemma 4 technical report. arXiv:2607.02770, 2026.
10. NVIDIA et al. Cosmos-Reason1: From physical common sense to embodied reasoning.
    arXiv:2503.15558, 2025.
11. H. Tan et al. RoboBrain 2.5: Depth in sight, time in mind. arXiv:2601.14352, 2026.
