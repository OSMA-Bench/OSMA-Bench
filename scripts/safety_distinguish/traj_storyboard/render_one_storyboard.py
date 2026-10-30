#!/usr/bin/env python3
"""Real-sim overlay: render robot in MuJoCo at each of 5 equal-distance keyframe
waypoint poses, extract robot pixels via diff masking, composite all onto a
single style-3 topview. Numbered labels (1..5) drawn at each waypoint.

Inside robocasa Docker. Single-task prototype of render_storyboards.py, kept
for debugging the per-keyframe raw renders (DATA_ROOT / OUT_ROOT envs;
default output OUT_ROOT=/workspace/outputs/storyboard_samples).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

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
from visualize_voxposer_task import _build_H_from_corners_uv, _world_to_topview  # noqa: E402

_CFG_PATH = "/workspace/policy/Voxposer/src/configs/robocasa_config.yaml"

TASK_NAME = "NavigateKitchenWineBlockingRouteA"
TASK_TAG = "L5_Wine_RouteA"
TIER = "2_Medium"
LAYOUT = 5
STYLE = 3
CM_TARGET = int(os.environ.get("CM_TARGET", "20"))
N_KF = 5

DATA_ROOT = Path(os.environ.get("DATA_ROOT",
                                "/workspace/fpanel_cm_sweep/_data"))
NPZ = DATA_ROOT / f"{TASK_TAG}.npz"
WAYPOINT_PNG = DATA_ROOT.parent / f"cm{CM_TARGET:02d}" / TIER / f"{TASK_TAG}.png"
OUT_DIR = Path(os.environ.get("OUT_ROOT", "/workspace/outputs/storyboard_samples"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / f"sim_overlay_{TASK_TAG}_style{STYLE}_cm{CM_TARGET}.png"

# Mobile base wheel pivot is ~20cm behind body origin on local +x axis.
# yaw rotation shifts body_xpos by R(θ)*o - o where o=(WHEEL_OFFSET_X, 0).
WHEEL_OFFSET_X = 0.20


def equal_distance_indices(xy, k):
    deltas = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(deltas)])
    total = float(arc[-1]) if arc[-1] > 0 else 1.0
    return [int(np.searchsorted(arc, t)) for t in np.linspace(0.0, total, k)], arc


def set_base_and_render(env, base_initial_xy: np.ndarray,
                        tx: float, ty: float, yaw: float) -> np.ndarray:
    """Set mobile-base joints so body_xpos lands on (tx,ty) with the given yaw,
    correcting for the wheel-pivot rotation offset; then re-render topview.

    `base_initial_xy` MUST be the body_xpos when qpos[:4]==0 (i.e. the model's
    base spawn point). Computing the offset from a CURRENT body_xpos that
    already reflects nonzero qpos produces incorrect targets.
    """
    sim = env.env.sim
    yaw_shift_x = WHEEL_OFFSET_X * (np.cos(yaw) - 1.0)
    yaw_shift_y = WHEEL_OFFSET_X * np.sin(yaw)
    sim.data.qpos[0] = (tx - base_initial_xy[0]) - yaw_shift_x
    sim.data.qpos[1] = (ty - base_initial_xy[1]) - yaw_shift_y
    sim.data.qpos[2] = float(yaw)
    sim.data.qvel[:3] = 0.0
    sim.forward()
    env.update_latest_obs()
    img = env.latest_obs.get("topview_image")
    if img is None:
        raise RuntimeError("topview_image missing")
    arr = np.flipud(np.asarray(img, dtype=np.uint8)).copy()
    return arr


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


def main():
    d = np.load(NPZ, allow_pickle=True)
    cms = list(d["cms"])
    ci = cms.index(CM_TARGET)
    wp_xy = np.asarray(d["wp_xy"][ci])
    wp_yaw_arr = np.asarray(d["wp_yaw"][ci])
    corners_uv = np.asarray(d["corners_uv"])
    ws_min = np.asarray(d["ws_min"])
    ws_max = np.asarray(d["ws_max"])
    map_h = int(d["map_h"])
    map_w = int(d["map_w"])

    kf_idx, arc = equal_distance_indices(wp_xy, N_KF)
    total = float(arc[-1])
    H = _build_H_from_corners_uv(corners_uv, map_h, map_w)

    def proj(xy):
        u, v = _world_to_topview(np.asarray(xy, dtype=float),
                                 ws_min, ws_max, map_h, H, map_w)
        return float(np.asarray(u).item()), float(np.asarray(v).item())

    # Load env once
    cfg = get_config(config_path=_CFG_PATH, task_type="navigation")
    tc = cfg["task"]
    tc["layout_ids"] = LAYOUT
    tc["style_ids"] = STYLE
    if "topview" not in tc.get("camera_names", []):
        tc["camera_names"] = list(tc.get("camera_names", [])) + ["topview"]
    env = VoxPoserRobocasa(task_name=TASK_NAME, task_config=tc)
    env.load_task()
    try:
        instruction = env.env.get_ep_meta().get("lang", "")
    except Exception:
        instruction = ""
    print(f"instruction: {instruction!r}")

    try:
        sim = env.env.sim
        # capture base_initial_xy with qpos=0 BEFORE any pose modification
        sim.data.qpos[:4] = 0.0
        sim.data.qvel[:3] = 0.0
        sim.forward()
        base_initial_xy = sim.data.body_xpos[
            sim.model.body_name2id("mobilebase0_base")][:2].copy()
        print(f"base_initial_xy = {base_initial_xy}")

        # 1) empty backdrop: send robot far off-camera (huge translation)
        sim.data.qpos[0] = 50.0
        sim.data.qpos[1] = 50.0
        sim.data.qpos[2] = 0.0
        sim.forward()
        env.update_latest_obs()
        empty = np.flipud(np.asarray(env.latest_obs["topview_image"], dtype=np.uint8)).copy()
        Image.fromarray(empty).save(OUT_DIR / f"{TASK_TAG}_style{STYLE}_empty.png")
        print(f"saved empty backdrop")

        # 2) render robot at each KF pose, diff-mask, composite onto empty.
        # Sequence: init pose (role="init") + 5 equal-distance planner KFs.
        # The last planner KF is tagged role="end". Intermediate get role="mid"
        # and are renumbered 1,2,... after overlap-skip. init/end always forced.
        composite = empty.copy()
        kf_targets = []   # list of (role, world_x, world_y, yaw, src_idx)
        kf_targets.append(("init", float(base_initial_xy[0]),
                           float(base_initial_xy[1]), 0.0, -1))
        for j, i in enumerate(kf_idx):
            tx, ty = float(wp_xy[i, 0]), float(wp_xy[i, 1])
            ry = yaw_at(wp_xy, wp_yaw_arr, i)
            role = "end" if j == len(kf_idx) - 1 else "mid"
            kf_targets.append((role, tx, ty, ry, i))

        OVERLAP_THRESHOLD = 0.25
        placed_mask = np.zeros(empty.shape[:2], dtype=bool)
        placed = []        # list of (role, src_idx, target_xy)
        skipped = []
        for role, tx, ty, ry, si in kf_targets:
            render = set_base_and_render(env, base_initial_xy, tx, ty, ry)
            Image.fromarray(render).save(
                OUT_DIR / f"{TASK_TAG}_style{STYLE}_{role}{si if role=='mid' else ''}_raw.png")
            actual_xy = sim.data.body_xpos[
                sim.model.body_name2id("mobilebase0_base")][:2]
            diff = np.abs(render.astype(np.int32) - empty.astype(np.int32)).sum(axis=2)
            mask = diff > 25
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                                     np.ones((3, 3), np.uint8)).astype(bool)
            n_mask = int(mask.sum())
            n_overlap = int((mask & placed_mask).sum())
            overlap_frac = n_overlap / max(1, n_mask)
            forced = role in ("init", "end")
            if (not forced) and overlap_frac > OVERLAP_THRESHOLD:
                skipped.append((role, si))
                print(f"  [{role}{si if role=='mid' else ''}] target=({tx:.3f},{ty:.3f},yaw={ry:.2f}) "
                      f"actual=({actual_xy[0]:.3f},{actual_xy[1]:.3f}) "
                      f"mask_px={n_mask}  OVERLAP {overlap_frac:.0%} -> SKIPPED")
                continue
            composite[mask] = render[mask]
            placed_mask |= mask
            placed.append((role, si, (tx, ty)))
            print(f"  [{role}{si if role=='mid' else ''}] target=({tx:.3f},{ty:.3f},yaw={ry:.2f}) "
                  f"actual=({actual_xy[0]:.3f},{actual_xy[1]:.3f}) "
                  f"mask_px={n_mask}  overlap={overlap_frac:.0%} -> placed"
                  f"{' (forced)' if forced else ''}")
        n_mid = sum(1 for r, _, _ in placed if r == "mid")
        print(f"placed: init=1, mid={n_mid}, end=1   skipped(mid)={len(skipped)}")

        # 3) overlay numbered labels NEXT TO each robot position (offset),
        # so the rendered robot stays visible.
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.imshow(composite)
        wp_uv = np.array([proj(p) for p in wp_xy])
        ax.plot(wp_uv[:, 0], wp_uv[:, 1], color="#1f77b4", lw=1.5,
                alpha=0.4, zorder=2)
        # connect init -> first planner KF with dashed line
        init_uv = proj(base_initial_xy)
        ax.plot([init_uv[0], wp_uv[kf_idx[0], 0]], [init_uv[1], wp_uv[kf_idx[0], 1]],
                color="#1f77b4", lw=1.5, alpha=0.4, ls="--", zorder=2)
        LABEL_OFFSET_PX = 28
        # Small numbered marker AT the robot base position (no offset) so the
        # robot pose is still visible and the label is unambiguously located.
        mid_counter = 0
        for role, si, (tx, ty) in placed:
            if role == "init":
                u, v = init_uv
                label = "I"
                face = "#00ffaa"
            elif role == "end":
                u, v = tuple(wp_uv[si])
                label = "E"
                face = "#ff6644"
            else:
                u, v = tuple(wp_uv[si])
                mid_counter += 1
                label = str(mid_counter)
                face = "yellow"
            ax.scatter([u], [v], s=110, facecolors=face,
                       edgecolors="black", lw=1.2, zorder=6)
            ax.text(u, v, label, ha="center", va="center",
                    fontsize=7, fontweight="bold", color="black", zorder=7)
        ax.set_title(instruction, fontsize=10, wrap=True)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUT, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {OUT}")
        # also save just composite (no mpl decoration)
        Image.fromarray(composite).save(
            OUT_DIR / f"sim_overlay_{TASK_TAG}_style{STYLE}_composite_raw.png")
    finally:
        try:
            env.env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
