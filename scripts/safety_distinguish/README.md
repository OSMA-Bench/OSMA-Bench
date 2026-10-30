# safety_distinguish

Image-generation side of the RQ3 (Perception of Safety) experiment: renders
the trajectory images that the VLM safety judge scores, plus the runner
scripts that orchestrated the judge sweeps over those images. The judge
implementation itself lives in [`../safety_recognition/`](../safety_recognition/).

```
safety_distinguish/
├── traj_storyboard/        # 2-panel trajectory storyboards (judge input)
│   ├── render_storyboards.py       # batch: every (task, cm) pair; --crop option
│   ├── render_one_storyboard.py    # single-task prototype (per-KF raws + 1-panel)
│   ├── examples/                   # committed sample outputs
│   └── final_images/               # full copy of the frozen judged set (2,133 PNG)
├── keyframe_storyboard/    # one raw PNG per keyframe pose (no compositing)
│   ├── render_at_keypoints.py
│   └── examples/                   # committed sample outputs
└── vlm_judge/              # parameterized judge orchestration
    ├── models.sh                   # model registry (id/port/suffix/tokens)
    ├── run_judge.sh                # run_judge.sh MODEL_KEY [--scale ...]
    ├── serve.sh                    # serve.sh MODEL_KEY [--gpus ...]
    └── legacy/                     # per-model scripts from the recorded runs
```

## Environment

Everything renders inside the robocasa Docker container (repo mounted at
`/workspace`, `MUJOCO_GL=egl` set by the launcher):

```bash
./docker/run_docker.sh          # builds/starts robotics-safety-container
docker exec robotics-safety-container python3 scripts/safety_distinguish/...
```

Rendering needs a GPU for EGL. The judge runners additionally need a
vLLM-served VLM (`vlm_judge/serve.sh`).

Path knobs (env vars, all with working defaults):

| Var | Default | Meaning |
|---|---|---|
| `DATA_ROOT` | `/workspace/fpanel_cm_sweep/_data` | waypoint cache (`{task}.npz`) |
| `OUT_ROOT` | script-specific under `/workspace/outputs/` | output tree |
| `N_KF` | `5` | equal-distance keyframes per trajectory |
| `CFG_PATH` | robocasa_config.yaml | VoxPoser env config |

## Naming conventions

- **Task tag**: `L{layout}_{Obstacle}_Route{A-G}`, e.g. `L5_Wine_RouteA`.
- **cm**: clearance margin in mm/10 (`cm00` … `cm80` = 0–800 mm stand-off);
  17 steps of 5.
- **Tier folders**: `1_High` (Human, CrawlingBaby, Cat, Dog), `2_Medium`
  (Wine, GlassOfWater, HotChocolate, Vase), `3_Low` (Kettlebell, Trashbin).
- The waypoint cache covers the corrected task pool: layouts {0, 2, 5} ×
  69 tasks × 17 cm.

## traj_storyboard/

Renders the 2-panel storyboard per (task, cm): left = planner waypoints,
right = ghost composite of the robot teleported to each keyframe (titled
"Executed trajectory", but it is a re-enactment of the *plan* — physics is
never stepped).

```bash
# smoke: 1 task, all 17 cm
docker exec robotics-safety-container python3 \
  scripts/safety_distinguish/traj_storyboard/render_storyboards.py \
  --layouts 5 --obstacles Wine --routes A

# full sweep: 207 tasks x 17 cm (3-4 h, resumable; --overwrite to regen)
docker exec robotics-safety-container python3 \
  scripts/safety_distinguish/traj_storyboard/render_storyboards.py
```

`--crop` windows both panels tightly on the trajectory corridor + obstacle
and drops the panel titles/legend — the style of the frozen judged set,
where the robot-obstacle gap dominates the frame. Cropped output defaults
to `{OUT_ROOT}_cropped`; the margin is `CROP_MARGIN` (fraction, 0.15).

`render_one_storyboard.py` is the hard-coded single-task prototype
(`L5_Wine_RouteA`, `CM_TARGET` env); it also saves each keyframe's raw
render, which is useful for debugging mask/pose issues.

`examples/` holds three finished storyboards from this renderer:
`fullview_L7_Dog_RouteB_cm00` vs `_cm80` (the cm effect — cm00 hugs the dog,
cm80 detours) and `cropview_L5_Wine_RouteA_cm20` (the `--crop` output).

`final_images/` is a full copy of the frozen judged set (2,133 PNG) — the
*actual* cropped images the paper's judge scored, e.g.
`final_images/cm00/1_High/L7_Dog_RouteB.png`. Pair any of these with the
`fullview_*` examples to see the crop the original judged pipeline applied;
`--crop` reproduces that style for new (corrected-pool) renders. Like every
image set in this repo it is gitignored (data, not source).

## keyframe_storyboard/

One raw PNG per keyframe pose — no matplotlib, no panel. Poses come from
the waypoint cache (`--cm N` or `--cm all`) or an explicit list.

```bash
# every keyframe of every cached cm for one task (schematic topview)
docker exec robotics-safety-container python3 \
  scripts/safety_distinguish/keyframe_storyboard/render_at_keypoints.py \
  --task L5_Wine_RouteA --cm all

# first-person view from the posed human, gaze following the robot
... render_at_keypoints.py --task L5_Wine_RouteA --cm all --view human

# arbitrary poses
... render_at_keypoints.py --task L5_Wine_RouteA --poses "1.0,-0.5,0.0;1.5,-0.5,1.57"
```

`--view human` renders the scene's `posed_human_main_group_1stview` camera:
the eye stays on the posed human and is re-aimed at the robot per keyframe
(its stock orientation faces a wall). Frames are saved separately only — no
compositing in this view. `HUMAN_CAM` / `AIM_HEIGHT` envs override the
camera name and aim height.

Output per cm: the individual `kf{i}_{role}.png` frames, a `montage.png`
tiling all keyframes of that cm into one labeled contact sheet (frames stay
separate — this is not the ghost overlay), `poses.json`, and (topview
`--composite` only) a ghost-overlay `composite.png`.

`examples/` holds the three finished deliverables for `L5_Wine_RouteA` cm20:
`topview_montage`, `topview_composite`, and `humanview_montage`.

## vlm_judge/

Parameterized orchestration: the model is an argument, not a script.

```bash
# 1. serve the model (detached vLLM inside the container, health-polled)
./vlm_judge/serve.sh qwen3-vl-32b --gpus 0,1

# 2. run the judge (binary + 1-5 by default; --resume built in)
./vlm_judge/run_judge.sh qwen3-vl-32b
./vlm_judge/run_judge.sh kimi-thinking --scale 1-5 --wait
./vlm_judge/run_judge.sh glm-flash --img-root /workspace/storyboard_overlay_nkf10
```

`models.sh` is the registry — key → model id, default port, output-suffix,
max-tokens, workers. Output naming matches the recorded runs exactly
(`judgments{suffix}[_binary]_v5.jsonl` in
`outputs/qual_fig_judgments/storyboard_overlay_cropped/`), so `--resume`
continues the existing files. Serving quirks are encoded in `serve.sh`:
GLM-FP8 (dedicated conda env, FlashInfer sampler off), DeepSeek-VL2
(`--hf-overrides` architectures hint), Kimi-Thinking (TP=1 only — TP=2
stalls after "MoE shared_experts").

| Model key | Suffix (JSONL) | Note |
|---|---|---|
| `qwen3-vl-32b` | *(blank)* | **final paper-figure model** |
| `qwen3-vl-30b-a3b` | `_qwen3-vl-30b-a3b` | panel |
| `glm-4.6v-fp8` / `glm-flash` | `_glm-4.6v-fp8` / `_glm-4.6v-flash` | panel (plots use flash) |
| `kimi-instruct` / `kimi-thinking` | `_kimi-vl-instruct` / `_kimi-vl-thinking` | panel |
| `deepseek-vl2` | `_deepseek-vl2-small` | panel |
| `molmo` | `_molmo-7b-d` | judged, excluded from plots |
| `cosmos-8b` / `qwen3-vl-4b` | `_cosmos-reason2-8b` / `_qwen3-vl-4b` | small-VLM comparison (degenerate) |

`legacy/` preserves the per-model scripts from the recorded v4/v5/nkf10
runs (paths repointed to the current tree where they had gone stale; the
exact prompt used by every sweep is frozen next to its JSONL as
`*.prompt.txt`). The v5 prompts are committed as
`safety_recognition/prompts/safety_prompt.txt` (1-5) and
`safety_prompt_binary.txt`.

## Image sets (repo root, gitignored)

| Set | Count | Task pool | Status |
|---|---|---|---|
| `storyboard_overlay/` | 2,133 | legacy {0,3,5,7,8} | frozen full-view source |
| `storyboard_overlay_cropped_final/` | 2,133 | legacy (1:1 with above) | **frozen canonical judged set** (the images the paper figure was judged on; copied into `traj_storyboard/final_images/`) |
| `storyboard_overlay_nkf10/` | 387 | corrected, L2-heavy | cm60–80 N_KF=10 extension |

The canonical judged set is frozen: it was rendered from the legacy task
pool whose L3/L7/L8 waypoint caches no longer exist, and the original
crop/recompose step was never committed. New renders (corrected pool
{0,2,5}) come from `traj_storyboard/` — full-view by default, or `--crop`
for a cropped_final-style tight window.
