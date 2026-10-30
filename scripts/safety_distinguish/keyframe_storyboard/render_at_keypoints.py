#!/usr/bin/env python3
"""Render the robot at individual keypoint poses; save one raw PNG per pose.

Minimal extraction of the storyboard pipeline's core mechanic — teleport the
mobile base and capture the topview camera — without the 2-panel figure or
batch machinery. The base is positioned by writing qpos directly and calling
sim.forward(); physics is never stepped, so each image is a static re-pose of
the planned trajectory, not an execution.

Keypoint poses come from either:
  * the fpanel waypoint cache (--cm N or --cm all): equal-arc-length
    keyframes plus the closest-to-obstacle pose, same selection as
    render_storyboards.py, or
  * an explicit list: --poses "x,y,yaw;x,y,yaw;..." (world meters/radians).

Output tree: {OUT_ROOT}/{task}/cm{NN}/kf{i}_{role}.png  (+ poses.json and a
montage.png tiling all keyframes of that cm into one file; + empty.png /
composite.png ghost overlay when --composite).

Run inside the robocasa Docker (repo mounted at /workspace):

  # every keyframe of every cached cm for one task
  python3 scripts/safety_distinguish/keyframe_storyboard/render_at_keypoints.py \
      --task L5_Wine_RouteA --cm all

  # single cm, plus a ghost composite of its keyframes
  python3 scripts/safety_distinguish/keyframe_storyboard/render_at_keypoints.py \
      --task L5_Wine_RouteA --cm 20 --composite

  # explicit poses
  python3 scripts/safety_distinguish/keyframe_storyboard/render_at_keypoints.py \
      --task L5_Wine_RouteA --poses "1.0,-0.5,0.0;1.5,-0.5,1.57"

  # first-person view of the posed human fixture (separate PNGs, no composite)
  python3 scripts/safety_distinguish/keyframe_storyboard/render_at_keypoints.py \
      --task L5_Wine_RouteA --cm all --view human
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_VP_SRC = "/workspace/policy/Voxposer/src"
_SCRIPTS = "/workspace/scripts"
for _p in (_VP_SRC, _SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from envs.robocasa_env import VoxPoserRobocasa            # noqa: E402
from utils.arguments import get_config                    # noqa: E402

CFG_PATH = os.environ.get(
    "CFG_PATH", "/workspace/policy/Voxposer/src/configs/robocasa_config.yaml")
DATA_ROOT = Path(os.environ.get(
    "DATA_ROOT", "/workspace/fpanel_cm_sweep/_data"))
OUT_ROOT = Path(os.environ.get(
    "OUT_ROOT", "/workspace/outputs/keypoint_renders"))

STYLE = int(os.environ.get("STYLE", "3"))                  # MODERN_1
TASK_RE = re.compile(r"^L(\d+)_([A-Za-z]+)_Route([A-G])$")
# the kitchen scenes ship a first-person camera on the posed-human fixture;
# its stock orientation faces away from the route, so we re-aim it per
# keyframe at the robot (eye position stays on the human).
HUMAN_CAM = os.environ.get("HUMAN_CAM", "posed_human_main_group_1stview")
AIM_HEIGHT = float(os.environ.get("AIM_HEIGHT", "0.5"))   # aim at robot torso


def _world_yaw_from_xmat(xmat):
    # xmat is the 3x3 rotation matrix flattened, MuJoCo row-major.
    return float(np.arctan2(xmat[3], xmat[0]))


def calibrate_base_axes(env):
    """Probe qpos->world mapping of the mobile base (same as batch renderer).

    Returns (base_initial_xy, M_inv, spawn_yaw, yaw_scale) so that
      qpos[0:2] = M_inv @ (target_xy - base_initial_xy)   (+1-step correction)
      world_yaw = spawn_yaw + yaw_scale * qpos[2]
    """
    sim = env.env.sim
    bid = sim.model.body_name2id("mobilebase0_base")
    sim.data.qpos[:4] = 0.0
    sim.data.qvel[:3] = 0.0
    sim.forward()
    base_init = sim.data.body_xpos[bid][:2].copy()
    spawn_yaw = _world_yaw_from_xmat(sim.data.body_xmat[bid])
    sim.data.qpos[:4] = [1.0, 0.0, 0.0, 0.0]
    sim.forward()
    d_local_x = sim.data.body_xpos[bid][:2] - base_init
    sim.data.qpos[:4] = [0.0, 1.0, 0.0, 0.0]
    sim.forward()
    d_local_y = sim.data.body_xpos[bid][:2] - base_init
    sim.data.qpos[:4] = [0.0, 0.0, 1.0, 0.0]
    sim.forward()
    yaw_at_q1 = _world_yaw_from_xmat(sim.data.body_xmat[bid])
    yaw_scale = float(np.arctan2(np.sin(yaw_at_q1 - spawn_yaw),
                                 np.cos(yaw_at_q1 - spawn_yaw)))
    if abs(yaw_scale) < 1e-3:
        yaw_scale = 1.0   # safety
    sim.data.qpos[:4] = 0.0
    sim.forward()
    M = np.column_stack([d_local_x, d_local_y])
    return base_init, np.linalg.inv(M), spawn_yaw, yaw_scale


def set_base_and_render(env, base_initial_xy, M_inv, spawn_yaw, yaw_scale,
                        tx, ty, world_yaw, obs_key="topview_image"):
    """Teleport the base to (tx, ty, world_yaw) and return one camera image."""
    sim = env.env.sim
    bid = sim.model.body_name2id("mobilebase0_base")
    sim.data.qpos[:4] = 0.0
    sim.data.qvel[:] = 0.0
    sim.forward()
    sim.data.qpos[2] = float(world_yaw - spawn_yaw) / yaw_scale
    world_delta = np.array([tx - base_initial_xy[0], ty - base_initial_xy[1]])
    q01 = M_inv @ world_delta
    sim.data.qpos[0] = float(q01[0])
    sim.data.qpos[1] = float(q01[1])
    sim.data.qpos[3] = 0.0
    sim.forward()
    # one-step correction for the yaw-induced wheel-pivot offset
    actual_xy = sim.data.body_xpos[bid][:2].copy()
    err = np.array([tx - actual_xy[0], ty - actual_xy[1]])
    sim.data.qpos[0] += float((M_inv @ err)[0])
    sim.data.qpos[1] += float((M_inv @ err)[1])
    sim.forward()
    env.update_latest_obs()
    img = env.latest_obs.get(obs_key)
    if img is None:
        raise RuntimeError(f"{obs_key} missing (available: "
                           f"{sorted(k for k in env.latest_obs if k.endswith('_image'))})")
    return np.flipud(np.asarray(img, dtype=np.uint8)).copy()


def _mat2quat_wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    if R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if R[1, 1] >= R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def aim_cam_at(env, cam_name, target_xyz):
    """Point a body-mounted camera at a world target, keeping its position.

    MuJoCo cameras look along their -Z axis (+X right, +Y up); cam_quat is
    expressed in the parent-body frame, hence the R_parent^T factor.
    """
    sim = env.env.sim
    m, d = sim.model, sim.data
    cid = m.camera_name2id(cam_name)
    eye = d.cam_xpos[cid].copy()
    fwd = np.asarray(target_xyz, dtype=float) - eye
    n = np.linalg.norm(fwd)
    if n < 1e-6:
        return
    fwd /= n
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    rn = np.linalg.norm(right)
    right = np.array([1.0, 0.0, 0.0]) if rn < 1e-6 else right / rn
    cam_up = np.cross(right, fwd)
    R_world = np.column_stack([right, cam_up, -fwd])
    R_parent = d.body_xmat[m.cam_bodyid[cid]].reshape(3, 3)
    m.cam_quat[cid] = _mat2quat_wxyz(R_parent.T @ R_world)
    sim.forward()


def mask_from(render, empty, thresh=25):
    diff = np.abs(render.astype(np.int32) - empty.astype(np.int32)).sum(axis=2)
    m = (diff > thresh).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8)).astype(bool)


def equal_distance_indices(xy, k):
    deltas = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(deltas)])
    total = float(arc[-1]) if arc[-1] > 0 else 1.0
    return [int(np.searchsorted(arc, t)) for t in np.linspace(0.0, total, k)]


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


def load_cache(task_tag):
    npz_path = DATA_ROOT / f"{task_tag}.npz"
    if not npz_path.exists():
        sys.exit(f"no waypoint cache: {npz_path} (set DATA_ROOT?)")
    d = np.load(npz_path, allow_pickle=True)
    return d, [int(c) for c in d["cms"]]


def poses_at(d, ci, n_kf, include_near):
    """(role, x, y, yaw) keypoints for cache entry ci."""
    wp_xy = np.asarray(d["wp_xy"][ci], dtype=float)
    wp_yaw = np.asarray(d["wp_yaw"][ci], dtype=float)
    kf_idx = equal_distance_indices(wp_xy, n_kf)
    roles = {i: "mid" for i in kf_idx}
    roles[kf_idx[-1]] = "end"
    roles[kf_idx[0]] = "start"
    if include_near and "obstacle_xy" in d:
        obstacle_xy = np.asarray(d["obstacle_xy"], dtype=float)
        i_near = int(np.argmin(
            np.linalg.norm(wp_xy - obstacle_xy[None, :], axis=1)))
        roles.setdefault(i_near, "near")
    return [(roles[i], float(wp_xy[i, 0]), float(wp_xy[i, 1]),
             yaw_at(wp_xy, wp_yaw, i))
            for i in sorted(roles)]


def parse_explicit_poses(spec):
    poses = []
    for part in spec.split(";"):
        vals = [float(v) for v in part.split(",")]
        if len(vals) == 2:
            vals.append(0.0)
        if len(vals) != 3:
            sys.exit(f"bad pose {part!r} (want x,y[,yaw])")
        poses.append(("kp", vals[0], vals[1], vals[2]))
    return poses


def make_montage(frames, out_path, thumb_w=360, pad=10, cols=None):
    """Tile the per-keyframe renders into one labeled contact sheet.

    frames: list of (image_path, caption). Frames stay separate (each in its
    own cell) — this is NOT the ghost composite, just a single-file overview.
    """
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    cells = []
    for path, cap in frames:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        th = int(round(h * thumb_w / w))
        im = im.resize((thumb_w, th))
        cap_h = 24
        cell = Image.new("RGB", (thumb_w, th + cap_h), (255, 255, 255))
        cell.paste(im, (0, cap_h))
        ImageDraw.Draw(cell).text((6, 5), cap, fill=(0, 0, 0), font=font)
        cells.append(cell)
    if not cells:
        return
    n = len(cells)
    if cols is None:
        cols = n if n <= 6 else (n + 1) // 2
    rows = (n + cols - 1) // cols
    cw, ch = cells[0].size
    grid = Image.new("RGB",
                     (cols * cw + pad * (cols + 1), rows * ch + pad * (rows + 1)),
                     (235, 235, 235))
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        grid.paste(cell, (pad + c * (cw + pad), pad + r * (ch + pad)))
    grid.save(out_path)


def render_pose_set(env, calib, empty, poses, out_dir, want_composite,
                    obs_key="topview_image", aim_cam=None):
    """Render every pose into out_dir; return the number of PNGs written."""
    base_xy, M_inv, spawn_yaw, yaw_scale = calib
    out_dir.mkdir(parents=True, exist_ok=True)
    composite = None
    if want_composite and empty is not None:
        composite = empty.astype(np.float32).copy()
    meta = []
    montage_frames = []
    for i, (role, tx, ty, yaw) in enumerate(poses):
        if aim_cam:
            aim_cam_at(env, aim_cam, (tx, ty, AIM_HEIGHT))
        render = set_base_and_render(env, base_xy, M_inv, spawn_yaw,
                                     yaw_scale, tx, ty, yaw, obs_key=obs_key)
        name = f"kf{i:02d}_{role}.png"
        Image.fromarray(render).save(out_dir / name)
        meta.append({"file": name, "role": role, "x": tx, "y": ty, "yaw": yaw})
        montage_frames.append((out_dir / name, f"{i}: {role}"))
        print(f"    [{i}] {role:<5} ({tx:+.3f}, {ty:+.3f}, yaw {yaw:+.2f}) -> {name}")
        if composite is not None:
            mask = mask_from(render, empty)
            alpha = 1.0 if role == "end" else 0.5
            rf = render.astype(np.float32)
            composite[mask] = alpha * rf[mask] + (1 - alpha) * composite[mask]
    if composite is not None:
        Image.fromarray(np.clip(composite, 0, 255).astype(np.uint8)).save(
            out_dir / "composite.png")
    # montage: all keyframes tiled into one file (frames stay separate)
    make_montage(montage_frames, out_dir / "montage.png")
    (out_dir / "poses.json").write_text(json.dumps(meta, indent=2))
    return len(meta)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True,
                    help="task tag, e.g. L5_Wine_RouteA")
    ap.add_argument("--cm", default=None,
                    help='clearance value from the waypoint cache, or "all"')
    ap.add_argument("--poses",
                    help='explicit "x,y,yaw;x,y,yaw" (overrides --cm)')
    ap.add_argument("--n-kf", type=int, default=5,
                    help="equal-distance keyframes from the cache (default 5)")
    ap.add_argument("--no-near", action="store_true",
                    help="skip the closest-to-obstacle keypoint")
    ap.add_argument("--composite", action="store_true",
                    help="also save a ghost composite per pose set")
    ap.add_argument("--view", choices=("topview", "human"), default="topview",
                    help="camera: schematic topview, or the posed human's "
                         "first-person camera (HUMAN_CAM env to override name)")
    ap.add_argument("--out", help=f"output root (default {OUT_ROOT}/<task>)")
    args = ap.parse_args()

    m = TASK_RE.match(args.task)
    if not m:
        sys.exit(f"bad task tag {args.task!r} (want L<layout>_<Obstacle>_Route<A-G>)")
    layout, obstacle, route = int(m.group(1)), m.group(2), m.group(3)

    # jobs: list of (subdir_name, poses)
    if args.poses:
        jobs = [("custom", parse_explicit_poses(args.poses))]
    elif args.cm is not None:
        d, cms = load_cache(args.task)
        targets = cms if args.cm == "all" else [int(args.cm)]
        bad = [c for c in targets if c not in cms]
        if bad:
            sys.exit(f"cm {bad} not in cache (available: {cms})")
        jobs = [(f"cm{c:02d}",
                 poses_at(d, cms.index(c), args.n_kf,
                          include_near=not args.no_near))
                for c in targets]
    else:
        sys.exit("need --cm or --poses")

    if args.view == "human" and args.composite:
        print("note: --composite ignored with --view human (separate frames only)")
        args.composite = False
    obs_key = f"{HUMAN_CAM}_image" if args.view == "human" else "topview_image"
    default_dir = args.task if args.view == "topview" else f"{args.task}_{args.view}"
    out_base = Path(args.out) if args.out else OUT_ROOT / default_dir

    cfg = get_config(config_path=CFG_PATH, task_type="navigation")
    tc = cfg["task"]
    tc["layout_ids"] = layout
    tc["style_ids"] = STYLE
    want_cams = ["topview"] + ([HUMAN_CAM] if args.view == "human" else [])
    for cam in want_cams:
        if cam not in tc.get("camera_names", []):
            tc["camera_names"] = list(tc.get("camera_names", [])) + [cam]
    env = VoxPoserRobocasa(
        task_name=f"NavigateKitchen{obstacle}BlockingRoute{route}",
        task_config=tc)
    env.load_task()
    if args.view == "human":
        try:
            env.env.sim.model.camera_name2id(HUMAN_CAM)
        except Exception:
            all_cams = [env.env.sim.model.camera_id2name(i)
                        for i in range(env.env.sim.model.ncam)]
            sys.exit(f"camera {HUMAN_CAM!r} not in this scene — posed-human "
                     f"cams: {[c for c in all_cams if c and 'human' in c.lower()]} "
                     f"(override with HUMAN_CAM env)")
    try:
        calib = calibrate_base_axes(env)
        print(f"base_initial_xy={calib[0]}  spawn_yaw={calib[2]:.3f}  "
              f"yaw_scale={calib[3]:.3f}")

        empty = None
        if args.composite:
            sim = env.env.sim
            sim.data.qpos[0:2] = 50.0   # off-camera for an empty backdrop
            sim.forward()
            env.update_latest_obs()
            empty = np.flipud(np.asarray(
                env.latest_obs["topview_image"], dtype=np.uint8)).copy()
            out_base.mkdir(parents=True, exist_ok=True)
            Image.fromarray(empty).save(out_base / "empty.png")

        total = 0
        for sub, poses in jobs:
            print(f"  [{sub}] {len(poses)} keypoints")
            total += render_pose_set(env, calib, empty, poses,
                                     out_base / sub, args.composite,
                                     obs_key=obs_key,
                                     aim_cam=HUMAN_CAM if args.view == "human" else None)
        print(f"saved {total} keypoint renders across {len(jobs)} pose sets "
              f"-> {out_base}")
    finally:
        try:
            env.env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
