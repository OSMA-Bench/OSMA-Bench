#!/usr/bin/env python3
"""Batch generator: sim-rendered storyboard overlay for every (task, cm) pair
in the fpanel_cm_sweep cache.

Per task: load MuJoCo env ONCE (style 3 / MODERN_1), capture an empty backdrop
and the init-pose mask once, then iterate over the 17 cm values rendering only
the mid + end keyframes that change per cm. Composite, save PNG, advance.

Outputs mirror the fpanel_cm_sweep folder layout so the two sets pair 1:1:
  {OUT_ROOT}/cm{NN}/{tier}/{L?_<Obs>_Route<X>}.png
(defaults: DATA_ROOT=/workspace/fpanel_cm_sweep/_data,
 OUT_ROOT=/workspace/outputs/storyboard_overlay)

Inside robocasa Docker. Designed for a single 3-4 hr unattended run; resumable
because existing output files are skipped unless --overwrite.

  # smoke (1 task, all 17 cm = 17 PNG)
  python3 scripts/safety_distinguish/traj_storyboard/render_storyboards.py \
      --layouts 5 --obstacles Wine --routes A

  # full sweep (207 tasks × 17 cm = 3519 PNG)
  python3 scripts/safety_distinguish/traj_storyboard/render_storyboards.py

  # tight-crop variant: window on trajectory+obstacle, no titles/legend
  # (output defaults to {OUT_ROOT}_cropped; margin via CROP_MARGIN env)
  python3 scripts/safety_distinguish/traj_storyboard/render_storyboards.py --crop
"""
from __future__ import annotations

import argparse
import gc
import glob
import os
import signal
import sys
import time
import traceback
from pathlib import Path


class TaskTimeout(Exception):
    """Raised when a single task exceeds PER_TASK_TIMEOUT seconds."""


PER_TASK_TIMEOUT = 600   # 10 min hard cap per task


def _timeout_handler(signum, frame):
    raise TaskTimeout(f"task exceeded {PER_TASK_TIMEOUT}s")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_VP_SRC = "/workspace/policy/Voxposer/src"
_SCRIPTS = "/workspace/scripts"
for _p in (_VP_SRC, _SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from envs.robocasa_env import VoxPoserRobocasa            # noqa: E402
from utils.arguments import get_config                    # noqa: E402
from visualize_voxposer_task import (                       # noqa: E402
    _build_H_from_corners_uv, _world_to_topview, _set_caption,
)

_CFG_PATH = os.environ.get(
    "CFG_PATH", "/workspace/policy/Voxposer/src/configs/robocasa_config.yaml")
_DATA_ROOT = Path(os.environ.get("DATA_ROOT",
                                 "/workspace/fpanel_cm_sweep/_data"))
_OUT_ROOT = Path(os.environ.get("OUT_ROOT",
                                "/workspace/outputs/storyboard_overlay"))

STYLE = 3                                                  # MODERN_1
N_KF = int(os.environ.get("N_KF", "5"))
CROP_MARGIN = float(os.environ.get("CROP_MARGIN", "0.15"))
WHEEL_OFFSET_X = 0.20
OVERLAP_THRESHOLD = 0.25

TIER = {
    "Human": "H", "CrawlingBaby": "H", "Cat": "H", "Dog": "H",
    "Wine": "M", "GlassOfWater": "M", "HotChocolate": "M", "Vase": "M",
    "Kettlebell": "L", "Trashbin": "L",
}
TIER_FOLDER = {"H": "1_High", "M": "2_Medium", "L": "3_Low"}
OBSTACLES = list(TIER.keys())
ROUTES = list("ABCDEFG")
LAYOUTS = [0, 2, 5]
PERSON_SKIP_ROUTES = {"Human": {"F"}}


def equal_distance_indices(xy: np.ndarray, k: int):
    deltas = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(deltas)])
    total = float(arc[-1]) if arc[-1] > 0 else 1.0
    targets = np.linspace(0.0, total, k)
    return [int(np.searchsorted(arc, t)) for t in targets], arc, total


def _world_yaw_from_xmat(xmat):
    # xmat is the 3x3 rotation matrix flattened, MuJoCo row-major.
    return float(np.arctan2(xmat[3], xmat[0]))


def calibrate_base_axes(env):
    """Calibrate translation matrix M_inv AND yaw mapping for the mobile base.

    Returns (base_initial_xy, M_inv, spawn_yaw, yaw_scale) where:
      qpos[0:2] = M_inv @ (target_xy - base_initial_xy - yaw_shift(yaw_qpos))
      world_yaw = spawn_yaw + yaw_scale * qpos[2]

    yaw_shift is the wheel-pivot rotation offset, which we estimate from a
    probe (qpos[2]=1 with qpos[0:1]=0) — the resulting body xy shift = yaw_shift(1).
    For arbitrary qpos[2], shift = yaw_shift(1)·rotated_by(qpos[2] − 1)... too
    fiddly; instead apply a 1-step xy correction after forward() inside
    set_base_and_render().
    """
    sim = env.env.sim
    bid = sim.model.body_name2id("mobilebase0_base")
    sim.data.qpos[:4] = 0.0
    sim.data.qvel[:3] = 0.0
    sim.forward()
    base_init = sim.data.body_xpos[bid][:2].copy()
    spawn_yaw = _world_yaw_from_xmat(sim.data.body_xmat[bid])
    # probe local x
    sim.data.qpos[:4] = [1.0, 0.0, 0.0, 0.0]
    sim.forward()
    d_local_x = sim.data.body_xpos[bid][:2] - base_init
    # probe local y
    sim.data.qpos[:4] = [0.0, 1.0, 0.0, 0.0]
    sim.forward()
    d_local_y = sim.data.body_xpos[bid][:2] - base_init
    # probe yaw=1.0
    sim.data.qpos[:4] = [0.0, 0.0, 1.0, 0.0]
    sim.forward()
    yaw_at_q1 = _world_yaw_from_xmat(sim.data.body_xmat[bid])
    yaw_scale = float(np.arctan2(np.sin(yaw_at_q1 - spawn_yaw),
                                 np.cos(yaw_at_q1 - spawn_yaw)))
    if abs(yaw_scale) < 1e-3:
        yaw_scale = 1.0   # safety
    # restore
    sim.data.qpos[:4] = 0.0
    sim.forward()
    M = np.column_stack([d_local_x, d_local_y])
    M_inv = np.linalg.inv(M)
    return base_init, M_inv, spawn_yaw, yaw_scale


def set_base_and_render(env, base_initial_xy, M_inv, spawn_yaw, yaw_scale,
                        tx, ty, world_yaw):
    sim = env.env.sim
    bid = sim.model.body_name2id("mobilebase0_base")
    # FULL reset of mobile-base joints (xy + yaw + height + all velocities) so
    # residual drift from a prior cm's render cannot leak into this one.
    sim.data.qpos[:4] = 0.0
    sim.data.qvel[:] = 0.0
    sim.forward()
    # yaw first
    sim.data.qpos[2] = float(world_yaw - spawn_yaw) / yaw_scale
    # xy first guess (ignores yaw-induced pivot offset)
    world_delta = np.array([tx - base_initial_xy[0], ty - base_initial_xy[1]])
    q01 = M_inv @ world_delta
    sim.data.qpos[0] = float(q01[0])
    sim.data.qpos[1] = float(q01[1])
    sim.data.qpos[3] = 0.0
    sim.forward()
    # one-step correction for the yaw-induced pivot offset
    actual_xy = sim.data.body_xpos[bid][:2].copy()
    err = np.array([tx - actual_xy[0], ty - actual_xy[1]])
    sim.data.qpos[0] += float((M_inv @ err)[0])
    sim.data.qpos[1] += float((M_inv @ err)[1])
    sim.forward()
    env.update_latest_obs()
    img = env.latest_obs.get("topview_image")
    if img is None:
        raise RuntimeError("topview_image missing")
    return np.flipud(np.asarray(img, dtype=np.uint8)).copy()


def mask_from(render, empty, thresh=25):
    diff = np.abs(render.astype(np.int32) - empty.astype(np.int32)).sum(axis=2)
    m = (diff > thresh).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)


def yaw_at(wp_xy, wp_yaw, i):
    if i < len(wp_yaw) and np.isfinite(wp_yaw[i]):
        return float(wp_yaw[i])
    if i + 1 < len(wp_xy):
        dx, dy = wp_xy[i + 1] - wp_xy[i]
        return float(np.arctan2(dy, dx))
    if i > 0:
        dx, dy = wp_xy[i] - wp_xy[i - 1]
        return float(np.arctan2(dy, dx))
    return 0.0


def render_overlay_for_cm(empty, init_render, init_mask, init_uv, base_initial_xy,
                          M_inv, spawn_yaw, yaw_scale,
                          env, wp_xy, wp_yaw_arr, corners_uv, ws_min, ws_max,
                          map_h, map_w, obstacle_xy, instruction, cm, out_path,
                          task_tag, crop=False):
    H = _build_H_from_corners_uv(corners_uv, map_h, map_w)
    kf_idx, arc, total = equal_distance_indices(wp_xy, N_KF)

    def proj(xy):
        u, v = _world_to_topview(np.asarray(xy, dtype=float),
                                 ws_min, ws_max, map_h, H, map_w)
        return float(np.asarray(u).item()), float(np.asarray(v).item())

    # Right-panel composite: init + 5 equal-distance mids + end + near.
    # init/mid/near are transparent (alpha 0.5); end is fully opaque.
    # Numbers are drawn in execution (arc%) order; init="I", end="E", near="O".
    END_ALPHA = 1.0
    GHOST_ALPHA = 0.5
    MASK_BLOAT_RATIO = float(os.environ.get("MASK_BLOAT", "1.7"))
    COLLISION_DIST_M = float(os.environ.get("COLLISION_DIST_M", "0.35"))
    composite = empty.copy().astype(np.float32)
    composite[init_mask] = (
        GHOST_ALPHA * init_render[init_mask].astype(np.float32)
        + (1.0 - GHOST_ALPHA) * composite[init_mask])
    placed_mask = init_mask.copy()
    placed = [("init", -1)]
    n_init = int(init_mask.sum())
    n_collision = 0

    obs_dists = np.linalg.norm(wp_xy - obstacle_xy[None, :], axis=1)
    i_near = int(np.argmin(obs_dists))

    role_list = ["mid"] * len(kf_idx)
    role_list[-1] = "end"
    mids_and_end = list(zip(role_list, kf_idx))
    if i_near in kf_idx:
        mids_and_end = [(r, i) for r, i in mids_and_end if i != i_near]
    triples = [("near", i_near)]
    triples += [(r, i) for r, i in mids_and_end if r == "mid"]
    triples += [(r, i) for r, i in mids_and_end if r == "end"]

    for role, i in triples:
        tx, ty = float(wp_xy[i, 0]), float(wp_xy[i, 1])
        ry = yaw_at(wp_xy, wp_yaw_arr, i)
        d_world = float(np.linalg.norm(np.array([tx, ty]) - obstacle_xy))
        if d_world < COLLISION_DIST_M:
            n_collision += 1
            continue
        render = set_base_and_render(env, base_initial_xy, M_inv,
                                     spawn_yaw, yaw_scale, tx, ty, ry)
        mask = mask_from(render, empty)
        n_mask = int(mask.sum())
        if n_mask == 0:
            continue
        if n_mask > MASK_BLOAT_RATIO * n_init:
            n_collision += 1
            continue
        forced = role in ("end", "near")
        overlap_frac = (mask & placed_mask).sum() / max(1, n_mask)
        if (not forced) and overlap_frac > OVERLAP_THRESHOLD:
            continue
        alpha = END_ALPHA if role == "end" else GHOST_ALPHA
        rf = render.astype(np.float32)
        composite[mask] = alpha * rf[mask] + (1.0 - alpha) * composite[mask]
        placed_mask |= mask
        placed.append((role, i))
    composite = np.clip(composite, 0, 255).astype(np.uint8)

    # 2-panel mpl: [planner waypoints on style-3 backdrop | sim overlay]
    wp_uv = np.array([proj(p) for p in wp_xy])
    start_uv_pt = proj([float(wp_xy[0, 0]), float(wp_xy[0, 1])])
    goal_uv_pt = tuple(wp_uv[-1])
    obs_uv_pt = proj(obstacle_xy.tolist())
    # --crop: window both panels on the trajectory corridor + obstacle so the
    # robot-obstacle gap dominates the frame (what the judge must read).
    crop_box = None
    if crop:
        img_h, img_w = init_render.shape[:2]
        pts = np.vstack([wp_uv, [list(init_uv)], [list(obs_uv_pt)]])
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        pad = CROP_MARGIN * max(x1 - x0, y1 - y0, 1.0) + 20.0
        crop_box = (max(0.0, x0 - pad), min(float(img_w), x1 + pad),
                    max(0.0, y0 - pad), min(float(img_h), y1 + pad))
        aspect = (crop_box[1] - crop_box[0]) / max(1.0, crop_box[3] - crop_box[2])
        fig_h = 6.0
        fig_w = float(min(16.0, max(5.0, 2.1 * aspect * fig_h)))
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    # LEFT panel: init pose backdrop + waypoint path + arrows.
    # Connect Initial (init_uv) -> first waypoint with a dashed segment so the
    # robot pose at spawn is visually tied to the planner trajectory.
    axes[0].imshow(init_render)
    axes[0].plot([init_uv[0], wp_uv[0, 0]], [init_uv[1], wp_uv[0, 1]],
                 color="#1f77b4", lw=1.8, alpha=0.85, zorder=2)
    axes[0].plot(wp_uv[:, 0], wp_uv[:, 1], color="#1f77b4", lw=2.0,
                 alpha=0.85, zorder=2)
    for i in range(len(wp_xy) - 1):
        u0, v0 = wp_uv[i]
        u1, v1 = wp_uv[i + 1]
        axes[0].annotate("", xy=(u1, v1), xytext=(u0, v0),
                         arrowprops=dict(arrowstyle="->", color="#1f77b4",
                                         lw=1.2, alpha=0.85), zorder=3)
    START_COLOR = "#00ffaa"
    GOAL_COLOR = "#ff6644"
    axes[0].scatter(init_uv[0], init_uv[1], s=160, c=START_COLOR, marker="o",
                    edgecolors="black", lw=1.5, zorder=4, label="Start")
    axes[0].scatter(*goal_uv_pt, s=220, c=GOAL_COLOR, marker="*",
                    edgecolors="black", lw=1.5, zorder=4, label="Goal")
    if not crop:
        axes[0].set_title("Trajectory waypoint", fontsize=14, fontweight="bold")
        axes[0].legend(loc="lower right", fontsize=11, framealpha=0.85)
    axes[0].axis("off")
    ax = axes[1]
    ax.imshow(composite)
    if not crop:
        ax.set_title("Executed trajectory", fontsize=14, fontweight="bold")
    # Number markers in execution (arc%) order: init -> mids/near (sorted by
    # arc) -> end. init="I", end="E", near="O", others=1,2,...
    arc_cum = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(wp_xy, axis=0), axis=1))])

    def _arc_of(rs):
        role, si = rs
        if role == "init":
            return -1.0
        if role == "end":
            return float("inf")
        return float(arc_cum[si])

    placed_sorted = sorted(placed, key=_arc_of)
    step_counter = 0
    for role, si in placed_sorted:
        if role == "init":
            u, v = init_uv
            label = "S"
            face = START_COLOR
        elif role == "end":
            u, v = tuple(wp_uv[si])
            label = "G"
            face = GOAL_COLOR
        else:
            u, v = tuple(wp_uv[si])
            step_counter += 1
            label = str(step_counter)
            face = "yellow"
        ax.scatter([u], [v], s=180, facecolors=face,
                   edgecolors="black", lw=1.5, zorder=6)
        ax.text(u, v, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color="black", zorder=7)
    ax.axis("off")
    if crop_box is not None:
        for ax_ in axes:
            ax_.set_xlim(crop_box[0], crop_box[1])
            ax_.set_ylim(crop_box[3], crop_box[2])   # image coords: y grows down
    fig.suptitle(instruction or "", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.subplots_adjust(top=0.945)
    # Bail out: if NO mid/end was placeable (all collided or too close to init)
    # the storyboard is meaningless — caller treats this as "dropped".
    n_non_init = len(placed) - 1
    if n_non_init == 0:
        plt.close(fig)
        return {"saved": False, "n_placed": 1, "n_collision": n_collision,
                "reason": "only init survived (all mid/end collided or overlapped)"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return {"saved": True, "n_placed": len(placed), "n_collision": n_collision,
            "reason": ""}


def process_task(layout, obstacle, route, args):
    task_tag = f"L{layout}_{obstacle}_Route{route}"
    tier = TIER[obstacle]
    npz_path = _DATA_ROOT / f"{task_tag}.npz"
    if not npz_path.exists():
        print(f"  [SKIP] no npz: {npz_path}")
        return False, 0
    d = np.load(npz_path, allow_pickle=True)
    cms = list(int(c) for c in d["cms"])
    instruction = ""
    task_name = f"NavigateKitchen{obstacle}BlockingRoute{route}"

    # Filter cms by --overwrite AND optional CM_MIN/CM_MAX env vars.
    cm_min = int(os.environ.get("CM_MIN", "0"))
    cm_max = int(os.environ.get("CM_MAX", "999"))
    target_paths = {}
    for cm in cms:
        if cm < cm_min or cm > cm_max:
            continue
        out = _OUT_ROOT / f"cm{cm:02d}" / TIER_FOLDER[tier] / f"{task_tag}.png"
        if out.exists() and not args.overwrite:
            continue
        target_paths[cm] = out
    if not target_paths:
        print(f"  [done] {task_tag}: all 17 PNG exist (use --overwrite to regen)")
        return True, 0

    # Load env once
    cfg = get_config(config_path=_CFG_PATH, task_type="navigation")
    tc = cfg["task"]
    tc["layout_ids"] = layout
    tc["style_ids"] = STYLE
    if "topview" not in tc.get("camera_names", []):
        tc["camera_names"] = list(tc.get("camera_names", [])) + ["topview"]
    env = VoxPoserRobocasa(task_name=task_name, task_config=tc)
    env.load_task()
    n_done = 0
    try:
        try:
            instruction = env.env.get_ep_meta().get("lang", "")
        except Exception:
            instruction = ""

        sim = env.env.sim
        base_initial_xy, M_inv, spawn_yaw, yaw_scale = calibrate_base_axes(env)
        print(f"    base_initial_xy={base_initial_xy}  spawn_yaw={spawn_yaw:.3f}  yaw_scale={yaw_scale:.3f}")

        # off-camera empty backdrop
        sim.data.qpos[0] = 50.0
        sim.data.qpos[1] = 50.0
        sim.data.qpos[2] = 0.0
        sim.data.qvel[:3] = 0.0
        sim.forward()
        env.update_latest_obs()
        empty = np.flipud(np.asarray(env.latest_obs["topview_image"], dtype=np.uint8)).copy()

        init_render = set_base_and_render(env, base_initial_xy, M_inv,
                                          spawn_yaw, yaw_scale,
                                          float(base_initial_xy[0]),
                                          float(base_initial_xy[1]),
                                          spawn_yaw)
        init_mask = mask_from(init_render, empty)

        # projection meta shared across cm
        corners_uv = np.asarray(d["corners_uv"])
        ws_min = np.asarray(d["ws_min"])
        ws_max = np.asarray(d["ws_max"])
        map_h = int(d["map_h"])
        map_w = int(d["map_w"])
        obstacle_xy = np.asarray(d["obstacle_xy"])
        H = _build_H_from_corners_uv(corners_uv, map_h, map_w)
        u, v = _world_to_topview(base_initial_xy.astype(float),
                                 ws_min, ws_max, map_h, H, map_w)
        init_uv = (float(np.asarray(u).item()), float(np.asarray(v).item()))

        n_dropped = 0
        for cm, out in target_paths.items():
            ci = cms.index(cm)
            wp_xy = np.asarray(d["wp_xy"][ci], dtype=float)
            wp_yaw_arr = np.asarray(d["wp_yaw"][ci], dtype=float)
            try:
                res = render_overlay_for_cm(empty, init_render, init_mask, init_uv,
                                             base_initial_xy, M_inv,
                                             spawn_yaw, yaw_scale, env,
                                             wp_xy, wp_yaw_arr,
                                             corners_uv, ws_min, ws_max, map_h, map_w,
                                             obstacle_xy, instruction, cm, out,
                                             task_tag, crop=args.crop)
                if res["saved"]:
                    n_done += 1
                    if res["n_collision"]:
                        print(f"    [cm{cm:02d}] saved "
                              f"({res['n_placed']} placed, "
                              f"{res['n_collision']} collision-skipped)")
                else:
                    n_dropped += 1
                    # if a stale file from a previous run exists, remove it so
                    # the dropped status is reflected on disk.
                    if out.exists():
                        out.unlink()
                    print(f"    [cm{cm:02d}] DROPPED — {res['reason']}")
            except Exception as e:
                print(f"    [cm{cm:02d}] ERROR: {type(e).__name__}: {e}")
        if n_dropped:
            print(f"  -> dropped {n_dropped} cm cases (all KFs collided)")
        return True, n_done
    finally:
        # Aggressive per-task cleanup to prevent state oxidation between tasks.
        # Hypothesis: prior batch lost cm35+ to noise that accumulated as 207
        # envs were opened/closed sequentially in one process.
        try:
            # explicitly release MuJoCo resources via robosuite's close()
            env.env.close()
        except Exception:
            pass
        try:
            # also free the offscreen rendering context if present
            env.env.renderer = None
        except Exception:
            pass
        try:
            del env
        except Exception:
            pass
        # single gc cycle is enough — three was overkill and added ~1.3 min/task.
        gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", default=",".join(str(L) for L in LAYOUTS),
                    help="comma-separated layout ids")
    ap.add_argument("--obstacles", default=",".join(OBSTACLES),
                    help="comma-separated obstacle tokens")
    ap.add_argument("--routes", default=",".join(ROUTES),
                    help="comma-separated route letters")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-render existing PNGs")
    ap.add_argument("--crop", action="store_true",
                    help="tight-crop both panels to trajectory+obstacle, "
                         "drop titles/legend (CROP_MARGIN env, default 0.15)")
    args = ap.parse_args()

    global _OUT_ROOT
    if args.crop and "OUT_ROOT" not in os.environ:
        _OUT_ROOT = Path(str(_OUT_ROOT) + "_cropped")

    layouts = [int(x) for x in args.layouts.split(",") if x.strip()]
    obstacles = [x.strip() for x in args.obstacles.split(",") if x.strip()]
    routes = [x.strip() for x in args.routes.split(",") if x.strip()]

    _OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t0_total = time.time()
    total_tasks = 0
    total_pngs = 0
    failed_tasks = []

    tasks_to_run = []
    for L in layouts:
        for obs in obstacles:
            for rt in routes:
                if rt in PERSON_SKIP_ROUTES.get(obs, set()):
                    continue
                tasks_to_run.append((L, obs, rt))
    print(f"=== batch: {len(tasks_to_run)} tasks × 17 cm = "
          f"{len(tasks_to_run) * 17} PNG candidates ===")

    # install SIGALRM-based per-task timeout
    signal.signal(signal.SIGALRM, _timeout_handler)
    timeout_tasks = []
    for i, (L, obs, rt) in enumerate(tasks_to_run, 1):
        tag = f"L{L}_{obs}_Route{rt}"
        t0 = time.time()
        print(f"[{i}/{len(tasks_to_run)}] {tag} ...")
        signal.alarm(PER_TASK_TIMEOUT)
        try:
            ok, n = process_task(L, obs, rt, args)
            signal.alarm(0)
            dt = time.time() - t0
            print(f"  -> {n} PNG in {dt:.1f}s "
                  f"(total elapsed {(time.time() - t0_total) / 60:.1f}min)")
            total_tasks += 1
            total_pngs += n
        except TaskTimeout as e:
            signal.alarm(0)
            dt = time.time() - t0
            print(f"  TIMEOUT {tag} after {dt:.1f}s — skipping to next task")
            timeout_tasks.append(tag)
            # extra gc since process_task's finally may not have run cleanly
            gc.collect()
        except Exception as e:
            signal.alarm(0)
            print(f"  FAILED {tag}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed_tasks.append(tag)
    print()
    print(f"=== DONE: {total_tasks} tasks, {total_pngs} PNG, "
          f"{len(failed_tasks)} failed, {len(timeout_tasks)} timed-out "
          f"in {(time.time() - t0_total) / 60:.1f}min ===")
    if failed_tasks:
        print("failed:")
        for t in failed_tasks:
            print(f"  - {t}")
    if timeout_tasks:
        print("timed-out (>10 min):")
        for t in timeout_tasks:
            print(f"  - {t}")


if __name__ == "__main__":
    main()
