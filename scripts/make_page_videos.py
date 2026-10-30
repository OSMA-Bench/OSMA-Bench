#!/usr/bin/env python3
"""Build the qualitative videos on the project page from recorded episodes.

  same_obstacle.mp4   one boy (High tier), three policies: the setup of Fig. 3b
  vla_tiers.mp4       pi0.5-OpenTau on one route: a cardboard box (Low), a cat and a boy (High)
  pivot_tiers.mp4     PIVOT with Gemma-4-12B-it on one route, one obstacle per caution tier
  voxposer_tiers.mp4  VoxPoser with Qwen3.6-35B-A3B on one route, one obstacle per caution tier

The frames are each episode's own recording, so nothing is re-simulated. The path
is shown with the robot's own pixels, stamped where it passed key positions: the
first recorded frame, every GHOST_SPACING of path, and the closest approach (the
first contact when there is one). The stamps are blended at GHOST_ALPHA so they
read as earlier positions; the live robot is drawn opaque. The clearance readout comes from the same
episode's trajectory log. These are example episodes from the runs in outputs/,
not the paper's figure episodes.

Frame-to-log alignment (checked by drawing the logged position on the frames):
  topview_image.mp4 (VoxPoser, PIVOT)  frame i = log step i + (L - N); the
      recording starts L - N steps into the log
  rollout.mp4 (VLA)                    frame k = step k (N = L + 1, last frame
      repeats the last step); rotated 90 degrees counter-clockwise to match
      the topview camera

The results library (--library) puts the models of LIBRARY_MODELS on the same matched
cells: for each policy, model and cell one clip with one obstacle per tier (the VLA
clips: LIBRARY_VLA_OBSTACLES), played at LIBRARY_STRIDE x speed, plus library.json,
which lists the clips for the page.

Usage: python3 scripts/make_page_videos.py [--out page/static/videos] [--only NAME ...]
       python3 scripts/make_page_videos.py --library [--out page/static/videos/library] [--jobs N]
"""
import argparse
import json
import multiprocessing
import re
import subprocess
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs"
FPS = 20            # control rate of every policy (dt = 0.05 s)
HOLD = 2 * FPS      # frames the final state stays on screen
PANEL_W = 400       # width of one panel in the output video (px)
GAP = 12            # white gap between panels (px)
HEAD, FOOT, BAR = 40, 40, 6
SS = 2              # supersampling factor for drawing
GHOST_ALPHA = 0.6   # opacity of the stamped earlier robots; the live robot stays opaque
GHOST_SPACING = 0.7   # m of path between two stamps
GHOST_MIN_GAP = 0.45  # m; a stamp this close to an earlier one is skipped
ROBOT_RADIUS = 0.5  # m; disk around the robot's visual center that holds the robot
DIFF = 45           # summed RGB difference from the background that counts as robot

INK, MUTED, CONTACT = "#1f2328", "#57606a", "#c72f2e"
COLOR = {"vla": "#1baf7a", "pivot": "#eb6834", "voxposer": "#2a78d6",
         "high": "#c72f2e", "moderate": "#c78b11", "low": "#309341"}


def mix(a, b, t):
    """Blend two hex colors: t of a, 1 - t of b."""
    a, b = (np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)]) for c in (a, b))
    return tuple(int(round(v)) for v in t * a + (1 - t) * b)

TASK = "NavigateKitchen{}BlockingRoute{}"
VOX0 = "qwen36_35b_topview_seed0_voxposer"
PIV0 = "gemma4_12b_seed0_pivot"
PIV16 = "gemma4_12b_tf_seed16_pivot"
VOX3 = "qwen36_35b_topview_seed3_voxposer"
# `layout` is the RoboCasa layout id, as in the run folders. The paper numbers the
# five benchmark layouts 0-4, which are RoboCasa ids 0, 2, 5, 7 and 8.
VIDEOS = {
    # paper Layout 0 (RoboCasa 0), route D, seed 0
    "same_obstacle": [
        dict(kind="vla", task=TASK.format("ChildBoy", "D"), layout=0,
             name="π0.5-OpenTau", tag="VLA", color="vla"),
        dict(kind="topview", run=PIV0, task=TASK.format("ChildBoy", "D"), layout=0,
             name="Gemma-4-12B-it", tag="PIVOT", color="pivot"),
        dict(kind="topview", run=VOX0, task=TASK.format("ChildBoy", "D"), layout=0,
             name="Qwen3.6-35B-A3B", tag="VoxPoser", color="voxposer"),
    ],
    # paper Layout 1 (RoboCasa 2), route B, seed 0
    "vla_tiers": [
        dict(kind="vla", task=TASK.format("CardboardBox", "B"), layout=2,
             name="Cardboard box", tag="Low", color="low"),
        dict(kind="vla", task=TASK.format("Cat", "B"), layout=2,
             name="Cat", tag="High", color="high"),
        dict(kind="vla", task=TASK.format("ChildBoy", "B"), layout=2,
             name="Boy", tag="High", color="high"),
    ],
    # paper Layout 2 (RoboCasa 5), route E, seed 16
    "pivot_tiers": [
        dict(kind="topview", run=PIV16, task=TASK.format("DuffelBag", "E"), layout=5,
             name="Bag", tag="Low", color="low"),
        dict(kind="topview", run=PIV16, task=TASK.format("FlowerPot", "E"), layout=5,
             name="Floor pot", tag="Moderate", color="moderate"),
        dict(kind="topview", run=PIV16, task=TASK.format("Human", "E"), layout=5,
             name="Human", tag="High", color="high"),
    ],
    # paper Layout 0 (RoboCasa 0), route A, seed 3
    "voxposer_tiers": [
        dict(kind="topview", run=VOX3, task=TASK.format("FloorCushion", "A"), layout=0,
             name="Cushion", tag="Low", color="low"),
        dict(kind="topview", run=VOX3, task=TASK.format("FlowerPot", "A"), layout=0,
             name="Floor pot", tag="Moderate", color="moderate"),
        dict(kind="topview", run=VOX3, task=TASK.format("ChildBoy", "A"), layout=0,
             name="Boy", tag="High", color="high"),
    ],
}

# Results library (--library): the models below on the same matched cells, one obstacle
# per tier, from the seed 0 runs of the paper's tables. The cells are those of the
# first three videos, one per layout. The VLA run (vla_625_original, a separate run with
# a 0.6 m goal radius) has recordings only for the cardboard box, the cat and the boy.
LIBRARY_SEED = 0
LIBRARY_CELLS = [(0, "D"), (2, "B"), (5, "E")]  # (RoboCasa layout id, route)
LIBRARY_OBSTACLES = [("CardboardBox", "Cardboard box", "Low", "low"),
                     ("FlowerPot", "Floor pot", "Moderate", "moderate"),
                     ("ChildBoy", "Boy", "High", "high")]
LIBRARY_VLA_OBSTACLES = [("CardboardBox", "Cardboard box", "Low", "low"),
                         ("Cat", "Cat", "High", "high"),
                         ("ChildBoy", "Boy", "High", "high")]
LIBRARY_MODELS = {  # policy -> [(model, seed 0 run folder under outputs/)]
    "voxposer": [("Gemma-4-E4B-it", "gemma4_e4b_topview_seed0_voxposer"),
                 ("Gemma-4-12B-it", "from_remote/gemma-12B/gemma4_12b_tf_topview_seed0_voxposer"),
                 ("Qwen3.5-4B", "qwen35_4b_topview_seed0_voxposer"),
                 ("Qwen3.5-9B", "qwen35_9b_topview_seed0_voxposer"),
                 ("Qwen3.6-35B-A3B", "qwen36_35b_topview_seed0_voxposer"),
                 ("RoboBrain2.5-8B", "robobrain_topview_seed0_voxposer")],
    "pivot": [("Gemma-4-E4B-it", "gemma4_seed0_pivot"),
              ("Gemma-4-12B-it", "gemma4_12b_seed0_pivot"),
              ("Qwen3.5-4B", "qwen35_4b_pivot_seed0"),
              ("Qwen3.5-9B", "qwen35_9b_seed0_pivot"),
              ("Qwen3.6-35B-A3B", "qwen36_35b_seed0_pivot"),
              ("RoboBrain2.5-8B", "robobrain_seed0_pivot")],
    "vla": [("π0.5-OpenTau", None)],
}
PAPER_LAYOUT = {0: 0, 2: 1, 5: 2, 7: 3, 8: 4}  # RoboCasa layout id -> paper layout number
POLICY_NAME = {"voxposer": "VoxPoser", "pivot": "PIVOT", "vla": "VLA"}
LIBRARY_STRIDE = 2         # every 2nd frame: the clips play at 2x speed
LIBRARY_PANEL_W = 300
LIBRARY_CRF = 26
LIBRARY_POSTER_QUALITY = 65


def font(size, bold=False):
    candidates = [
        Path("/usr/share/fonts/opentype/urw-base35") / ("NimbusSans-Bold.otf" if bold else "NimbusSans-Regular.otf"),
        Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def read_video(path):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True)
    w, h = map(int, probe.stdout.strip().split(","))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def affine_of(fn):
    """2x3 matrix of an affine map R^2 -> R^2 given as a function."""
    o, ex, ey = fn(np.array([0.0, 0.0])), fn(np.array([1.0, 0.0])), fn(np.array([0.0, 1.0]))
    return np.array([[ex[0] - o[0], ey[0] - o[0], o[0]], [ex[1] - o[1], ey[1] - o[1], o[1]]])


def topview_evaluation(spec):
    return next(e["evaluation"] for e in json.load(open(RUNS / spec["run"] / "results.json"))["results"]
                if e["task_info"]["task_dir"] == f"layout{spec['layout']}/{spec['task']}")


def vla_run(spec):
    runs = sorted((RUNS / "vla_625_original" / "_runs").glob(f"*__{spec['task']}__L*__sweep_L{spec['layout']}"))
    assert len(runs) == 1, runs
    return runs[0]


def logged_path(spec):
    """Base positions and goal of an episode, without reading its frames."""
    if spec["kind"] == "vla":
        run = vla_run(spec)
        pos = np.load(run / "vla_dump.npz", allow_pickle=True)["robot_base_pos"][:, :2]
        goal = json.load(open(run / "evaluation.json"))["goal_pos_xy"]
    else:
        d = RUNS / spec["run"] / f"layout{spec['layout']}" / spec["task"]
        pos = np.array(json.load(open(d / "trajectory_log.json"))["robot_pos"], float)[:, :2]
        goal = topview_evaluation(spec)["goal_pos_xy"]
    return dict(pos=pos.astype(float), goal=np.array(goal, float))


def topview_episode(spec):
    d = RUNS / spec["run"] / f"layout{spec['layout']}" / spec["task"]
    frames = read_video(d / "topview_image.mp4")
    z = np.load(d / "voxposer_dump.npz", allow_pickle=True)
    bmin, bmax, uv = z["workspace_bounds_min"], z["workspace_bounds_max"], z["topview_corners_uv"]
    s = (uv[:, 0].max() - uv[:, 0].min()) / (bmax[0] - bmin[0])
    umax, vmax = uv[:, 0].max(), uv[:, 1].max()
    world_to_px = affine_of(lambda p: np.array([umax - (p[0] - bmin[0]) * s, vmax - (bmax[1] - p[1]) * s]))
    log = json.load(open(d / "trajectory_log.json"))
    pos = np.array(log["robot_pos"], float)[:, :2]
    n, steps = len(frames), len(pos)
    ev = topview_evaluation(spec)
    return dict(frames=frames, world_to_px=world_to_px, pos=pos, yaw=np.array(log["robot_yaw"], float),
                clearance=np.array(log["min_obstacle_distance"], float),
                contact=np.array([any(h.values()) for h in log["obstacle_contact_history"]]),
                frame_to_step=np.arange(n) + (steps - n), goal=np.array(ev["goal_pos_xy"], float),
                min_clearance=float(ev["min_clearance_m"]), collided=bool(ev["obstacle_contact_ever"]))


def vla_episode(spec):
    run = vla_run(spec)
    frames = np.ascontiguousarray(np.rot90(read_video(run / "rollout.mp4"), k=1, axes=(1, 2)))
    z = np.load(run / "vla_dump.npz", allow_pickle=True)
    pos = z["robot_base_pos"][:, :2].astype(float)
    n, steps = len(frames), len(pos)
    ev = json.load(open(run / "evaluation.json"))
    return dict(frames=frames, pos=pos, yaw=z["robot_base_yaw"].astype(float),
                clearance=z["min_obstacle_distance_m"].astype(float),
                contact=z["obstacle_contact"].any(axis=1), frame_to_step=np.minimum(np.arange(n), steps - 1),
                goal=np.array(ev["goal_pos_xy"], float),
                min_clearance=float(ev["min_clearance_m"]), collided=bool(ev["obstacle_contact_ever"]))


def track_robot(frames, background=None):
    """Centroid of the largest blob that differs from the median frame (or `background`),
    per frame (NaN if none)."""
    if background is None:
        background = np.median(frames[::3].astype(np.int16), axis=0)
    centers = []
    for f in frames:
        moving = ndimage.binary_opening(np.abs(f.astype(np.int16) - background).sum(-1) > 60, iterations=2)
        labels, count = ndimage.label(moving)
        if count == 0:
            centers.append((np.nan, np.nan))
            continue
        biggest = np.argmax(ndimage.sum(moving, labels, range(1, count + 1))) + 1
        ys, xs = np.nonzero(labels == biggest)
        centers.append((xs.mean(), ys.mean()))
    return np.array(centers)


def fit_camera(episodes, skip_contact=False):
    """Top-down camera and the robot's visual center, fitted to the tracked robot.

    The tracked pixel is world_to_px(base + R(yaw) @ offset) with world_to_px =
    [[-s, 0, tx], [0, s, ty]] and offset the visual center in the robot's frame
    (forward, left). The topview camera is known from the dump, so only the
    offset is fitted. The VLA rollout camera has no stored calibration, but it
    is the same for every episode of a layout (their median frames differ only
    at the obstacle), so s, tx and ty are fitted over all of them. Frames more
    than three times the median residual off are dropped and the fit repeated.
    With skip_contact, frames with obstacle contact are left out: an obstacle the
    robot pushes joins the robot's blob.
    """
    known = "world_to_px" in episodes[0]
    rows, rhs = [], []
    for ep in episodes:
        f2s = ep["frame_to_step"]
        x, y = ep["pos"][f2s].T
        c, s = np.cos(ep["yaw"][f2s]), np.sin(ep["yaw"][f2s])
        one, zero = np.ones_like(x), np.zeros_like(x)
        a = np.empty((2 * len(x), 5))
        a[0::2] = np.stack([-x, -c, s, one, zero], 1)
        a[1::2] = np.stack([y, s, c, zero, one], 1)
        b = ep["tracked"].reshape(-1).copy()
        if skip_contact:
            b[np.repeat(ep["contact"][f2s], 2)] = np.nan
        if known:
            m = ep["world_to_px"]
            b -= (ep["pos"][f2s] @ m[:, :2].T + m[:, 2]).reshape(-1)
        rows.append(a)
        rhs.append(b)
    cols = [1, 2] if known else [0, 1, 2, 3, 4]
    a_all, b_all = np.vstack(rows)[:, cols], np.concatenate(rhs)
    keep = ~np.isnan(b_all)
    keep &= np.repeat(keep.reshape(-1, 2).all(1), 2)
    for _ in range(2):
        x, *_ = np.linalg.lstsq(a_all[keep], b_all[keep], rcond=None)
        err = np.repeat(np.linalg.norm((a_all @ x - b_all).reshape(-1, 2), axis=1), 2)
        keep = ~np.isnan(err) & (err <= 3 * np.nanmedian(err))
    bad = []
    for ep, a, b in zip(episodes, rows, rhs):
        if known:
            d = x
            m = ep["world_to_px"]
        else:
            s, d, (tx, ty) = x[0], x[1:3], x[3:5]
            m = ep["world_to_px"] = np.array([[-s, 0, tx], [0, s, ty]])
        scale = abs(m[0, 0])
        ep["offset"] = d / scale
        resid = np.nanmedian(np.linalg.norm((a[:, cols] @ x - b).reshape(-1, 2), axis=1))
        print(f"  camera ({ep['name']}): {scale:.2f} px/m, visual center {ep['offset'].round(3)} m "
              f"from the base, median residual {resid:.1f} px")
        if not resid < 0.1 * scale:
            bad.append(ep["name"])
    assert not bad, f"robot tracking does not fit a top-down camera ({', '.join(bad)})"


def refit_camera(episodes):
    """Second try at fit_camera, for a robot that stands still so long that the plain
    median frame holds it, or that pushes the obstacle: track against a background
    without the logged robot, placed with the camera of the first try, and leave out
    the frames with contact."""
    for ep in episodes:
        print(f"  camera ({ep['name']}): refit, tracking against a background without the logged robot")
        ep["tracked"] = track_robot_away_from_log(ep)
    fit_camera(episodes, skip_contact=True)


def track_robot_away_from_log(ep):
    """Track the robot against a median frame that leaves out the logged base position in
    each frame. For a robot that stands still so long that the plain median frame holds
    it; needs the camera. The disk also covers the visual center's offset from the base
    (under 0.25 m)."""
    m = ep["world_to_px"]
    base = ep["pos"][ep["frame_to_step"]] @ m[:, :2].T + m[:, 2]
    background = robot_free_background(dict(frames=ep["frames"], centers=base,
                                            r_px=(ROBOT_RADIUS + 0.25) * abs(m[0, 0])))
    return track_robot(ep["frames"], background)


def fill_one_step_gaps(contact):
    """Contact flags with single-step gaps closed, so the red border does not flash for one frame."""
    out = contact.copy()
    out[1:-1] |= contact[:-2] & contact[2:]
    return out


def robot_free_background(ep, samples=60):
    """Per-pixel median over frames, leaving out a disk around the robot in each frame."""
    frames, centers, r = ep["frames"], ep["centers"], 1.3 * ep["r_px"]
    idx = np.unique(np.linspace(0, len(frames) - 1, min(len(frames), samples)).astype(int))
    h, w = frames.shape[1:3]
    yy, xx = np.mgrid[:h, :w]
    stack = np.full((len(idx), h, w, 3), np.nan, np.float32)
    for k, i in enumerate(idx):
        far = (xx - centers[i, 0]) ** 2 + (yy - centers[i, 1]) ** 2 > r * r
        stack[k][far] = frames[i][far]
    bg = np.nanmedian(stack, axis=0)
    hole = np.isnan(bg[..., 0])
    bg[hole] = np.median(frames[idx], axis=0)[hole]
    return bg


def robot_frame(ep, i, xs, ys):
    """Coordinates (forward, left) in m of pixels (xs, ys) of frame i in the robot's base frame."""
    step = ep["frame_to_step"][i]
    lin, t = ep["world_to_px"][:, :2], ep["world_to_px"][:, 2]
    inv = np.linalg.inv(lin)
    dx = inv[0, 0] * (xs - t[0]) + inv[0, 1] * (ys - t[1]) - ep["pos"][step, 0]
    dy = inv[1, 0] * (xs - t[0]) + inv[1, 1] * (ys - t[1]) - ep["pos"][step, 1]
    c, s = np.cos(ep["yaw"][step]), np.sin(ep["yaw"][step])
    return c * dx + s * dy, -s * dx + c * dy


def robot_mask(ep, i):
    """Box and mask of the robot's pixels in frame i: those that differ from the
    robot-free background inside a disk around the robot's visual center and,
    once it is known, inside the robot's footprint."""
    h, w = ep["frames"].shape[1:3]
    (cx, cy), r = ep["centers"][i], ep["r_px"]
    x0, x1 = max(int(cx - r), 0), min(int(cx + r) + 2, w)
    y0, y1 = max(int(cy - r), 0), min(int(cy + r) + 2, h)
    if x0 >= x1 or y0 >= y1:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    if "footprint" in ep:
        (f0, l0), (f1, l1) = ep["footprint"]
        f, l = robot_frame(ep, i, xx, yy)
        disk &= (f >= f0) & (f <= f1) & (l >= l0) & (l <= l1)
    diff = np.abs(ep["frames"][i, y0:y1, x0:x1].astype(np.float32) - ep["bg"][y0:y1, x0:x1]).sum(-1) > DIFF
    m = ndimage.binary_opening(diff & disk)
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=2) & disk)
    labels, count = ndimage.label(m)
    if count > 1:
        sizes = ndimage.sum(m, labels, range(1, count + 1))
        m = np.isin(labels, 1 + np.nonzero(sizes >= 0.15 * sizes.max())[0])
    return (y0, y1, x0, x1, m) if m.any() else None


def key_frames(ep):
    """Frames whose robot is stamped: the first recorded frame, the closest approach
    (the first contact when there is one), and one per GHOST_SPACING of path unless
    it lies within GHOST_MIN_GAP of a stamp already chosen."""
    f2s = ep["frame_to_step"]
    p = ep["pos"][f2s]
    contact = ep["contact"][f2s]
    closest = int(np.argmax(contact)) if contact.any() else int(np.argmin(ep["clearance"][f2s]))
    keys = [0, closest]
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    mark = 0.0
    for i in range(1, len(p)):
        if arc[i] - mark < GHOST_SPACING:
            continue
        mark = arc[i]
        if np.linalg.norm(p[keys] - p[i], axis=1).min() >= GHOST_MIN_GAP:
            keys.append(i)
    return sorted(set(keys))


def fit_footprint(ep, far=1.0, margin=0.03):
    """Box around the robot in its base frame, (forward, left) min and max in m, from
    its pixels in frames where it is at least `far` m from the obstacle."""
    clear = ep["clearance"][ep["frame_to_step"]]
    idx = np.nonzero(clear >= far)[0]
    if len(idx) < 10:
        idx = np.argsort(clear)[-max(10, len(clear) // 3):]
    fs, ls = [], []
    for i in idx[np.linspace(0, len(idx) - 1, min(len(idx), 30)).astype(int)]:
        box = robot_mask(ep, i)
        if box is not None:
            y0, y1, x0, x1, m = box
            ys, xs = np.nonzero(m)
            f, l = robot_frame(ep, i, xs + x0, ys + y0)
            fs.append(f)
            ls.append(l)
    fs, ls = np.concatenate(fs), np.concatenate(ls)
    return ((np.percentile(fs, 0.5) - margin, np.percentile(ls, 0.5) - margin),
            (np.percentile(fs, 99.5) + margin, np.percentile(ls, 99.5) + margin))


def prepare_ghosts(ep):
    """Robot-free background, the robot's footprint, and the robot pixels of every key frame."""
    f2s = ep["frame_to_step"]
    yaw, (of, ol) = ep["yaw"][f2s], ep["offset"]
    center = ep["pos"][f2s] + np.stack([np.cos(yaw) * of - np.sin(yaw) * ol,
                                        np.sin(yaw) * of + np.cos(yaw) * ol], 1)
    ep["centers"] = center @ ep["world_to_px"][:, :2].T + ep["world_to_px"][:, 2]
    ep["r_px"] = ROBOT_RADIUS * abs(ep["world_to_px"][0, 0])
    ep["bg"] = robot_free_background(ep)
    ep["footprint"] = fit_footprint(ep)
    ep["stamps"] = []
    for k in key_frames(ep):
        box = robot_mask(ep, k)
        if box is not None:
            y0, y1, x0, x1, m = box
            ep["stamps"].append((k, box, ep["frames"][k, y0:y1, x0:x1]))


def frame_with_ghosts(ep, i):
    """Frame i with the robot of every earlier key frame stamped in, and the live robot on top."""
    img = ep["frames"][i].copy()
    for k, (y0, y1, x0, x1, m), rgb in ep["stamps"]:
        if k >= i:
            break
        region = img[y0:y1, x0:x1]
        region[m] = (GHOST_ALPHA * rgb[m] + (1 - GHOST_ALPHA) * region[m]).astype(np.uint8)
    live = robot_mask(ep, i)
    if live is not None:
        y0, y1, x0, x1, m = live
        img[y0:y1, x0:x1][m] = ep["frames"][i, y0:y1, x0:x1][m]
    return img


def panel_geometry(episodes, pad=0.35, min_ratio=0.5, panel_w=PANEL_W):
    """World box around every path, start and goal, and the world-to-panel map.

    The panel keeps the topview orientation (x to the left, y down) unless the
    box is taller than wide; then it is turned 90 degrees so the route runs
    left to right.
    """
    pts = np.concatenate([e["pos"] for e in episodes] + [e["goal"][None] for e in episodes])
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    rotate = (hi[1] - lo[1]) > (hi[0] - lo[0])
    long_axis, short_axis = (1, 0) if rotate else (0, 1)
    need = (hi[long_axis] - lo[long_axis]) * min_ratio - (hi[short_axis] - lo[short_axis])
    if need > 0:
        lo[short_axis] -= need / 2
        hi[short_axis] += need / 2
    k = panel_w / (hi[long_axis] - lo[long_axis])
    height = int(round((hi[short_axis] - lo[short_axis]) * k / 2) * 2)
    if rotate:   # horizontal = y, vertical = x
        m = np.array([[0, k, -lo[1] * k], [k, 0, -lo[0] * k]])
    else:        # horizontal = -x, vertical = y
        m = np.array([[-k, 0, hi[0] * k], [0, k, -lo[1] * k]])
    return m, (panel_w, height)


def to3(m):
    return np.vstack([m, [0, 0, 1]])


def draw_panel(ep, t, m, size):
    ended = t >= len(ep["frames"]) - 1
    if ended and "final_panel" in ep:
        return ep["final_panel"]
    w, h = size
    i = min(t, len(ep["frames"]) - 1)
    step = ep["frame_to_step"][i]
    mss = to3(m * SS)
    src = (to3(ep["world_to_px"]) @ np.linalg.inv(mss))[:2]
    frame = Image.fromarray(frame_with_ghosts(ep, i)).transform(
        (w * SS, h * SS), Image.AFFINE, tuple(src.reshape(-1)), resample=Image.BICUBIC)
    canvas = Image.new("RGB", (w * SS, (BAR + HEAD + h + FOOT) * SS), "white")
    top = (BAR + HEAD) * SS
    canvas.paste(frame, (0, top))
    dr = ImageDraw.Draw(canvas)
    in_contact = bool(ep["contact"][step]) and not ended

    def px(p):
        q = mss[:2] @ np.array([p[0], p[1], 1.0])
        return float(q[0]), float(q[1]) + top

    halo = dict(stroke_width=3 * SS, stroke_fill="white")
    for letter, p in (("S", ep["pos"][0]), ("G", ep["goal"])):
        x, y = px(p)
        dr.text((x, y), letter, font=font(15 * SS, True), fill=INK, anchor="mm", **halo)
    if in_contact:
        dr.rectangle([0, top, w * SS - 1, top + h * SS - 1], outline=CONTACT, width=4 * SS)

    # Header: the name, then the tag as a pill (tinted fill, colored dot), as on the page.
    color = COLOR[ep["color"]]
    dr.rectangle([0, 0, w * SS, BAR * SS], fill=color)
    name, tag = font(17 * SS, True), font(14 * SS, True)
    pad, dot, gap, space, ph = 8 * SS, 8 * SS, 5 * SS, 10 * SS, 22 * SS
    nw = dr.textlength(ep["name"], font=name)
    pw = pad + dot + gap + dr.textlength(ep["tag"], font=tag) + pad
    x0 = (w * SS - nw - space - pw) / 2
    ymid = (BAR + HEAD / 2) * SS
    dr.text((x0, ymid), ep["name"], font=name, fill=INK, anchor="lm")
    bx = x0 + nw + space
    dr.rounded_rectangle([bx, ymid - ph / 2, bx + pw, ymid + ph / 2], radius=ph / 2,
                         fill=mix(color, "#ffffff", 0.14))
    dr.ellipse([bx + pad, ymid - dot / 2, bx + pad + dot, ymid + dot / 2], fill=color)
    dr.text((bx + pad + dot + gap, ymid), ep["tag"], font=tag, fill=mix(color, INK, 0.65),
            anchor="lm")

    yf = top + h * SS + FOOT / 2 * SS
    big, small = font(17 * SS, True), font(15 * SS)
    if ended:
        if ep["collided"]:
            dr.text((w * SS / 2, yf), "collision", font=big, fill=CONTACT, anchor="mm")
        else:
            label, value = "min clearance  ", f"{max(ep['min_clearance'], 0):.2f} m"
            lw, vw = dr.textlength(label, font=small), dr.textlength(value, font=big)
            x0 = (w * SS - lw - vw) / 2
            dr.text((x0, yf), label, font=small, fill=MUTED, anchor="lm")
            dr.text((x0 + lw, yf), value, font=big, fill=INK, anchor="lm")
    elif in_contact:
        dr.text((w * SS / 2, yf), "collision", font=big, fill=CONTACT, anchor="mm")
    else:
        label, value = "clearance  ", f"{max(ep['clearance'][step], 0):.2f} m"
        lw, vw = dr.textlength(label, font=small), dr.textlength(value, font=big)
        x0 = (w * SS - lw - vw) / 2
        dr.text((x0, yf), label, font=small, fill=MUTED, anchor="lm")
        dr.text((x0 + lw, yf), value, font=big, fill=INK, anchor="lm")
    out = canvas.resize((w, BAR + HEAD + h + FOOT), Image.LANCZOS)
    if ended:
        ep["final_panel"] = out
    return out


def load(specs):
    episodes = []
    for spec in specs:
        ep = vla_episode(spec) if spec["kind"] == "vla" else topview_episode(spec)
        ep.update(name=spec["name"], tag=spec["tag"], color=spec["color"], layout=spec["layout"],
                  contact=fill_one_step_gaps(ep["contact"]), tracked=track_robot(ep["frames"]))
        episodes.append(ep)
    vla = defaultdict(list)
    for ep in episodes:
        if "world_to_px" in ep:
            try:
                fit_camera([ep])
            except AssertionError:
                refit_camera([ep])
        else:
            vla[ep["layout"]].append(ep)
    for group in vla.values():
        try:
            fit_camera(group)
        except AssertionError:
            refit_camera(group)
    for ep in episodes:
        prepare_ghosts(ep)
        (f0, l0), (f1, l1) = ep["footprint"]
        print(f"  {ep['name']:>16}: {len(ep['frames'])} frames from log step {ep['frame_to_step'][0]}, "
              f"{len(ep['pos'])} steps, min clearance {ep['min_clearance']:.3f} m, contact {ep['collided']}, "
              f"footprint {f1 - f0:.2f} x {l1 - l0:.2f} m, stamps at frames {[k for k, _, _ in ep['stamps']]}")
    return episodes


def build(name, specs, out_dir, crf=22, stride=1, panel_w=PANEL_W, poster_quality=90, geometry=None):
    print(name)
    episodes = load(specs)
    goals = np.array([e["goal"] for e in episodes])
    assert np.allclose(goals, goals[0], atol=1e-3), f"episodes disagree on the goal: {goals}"
    m, size = panel_geometry(episodes, panel_w=panel_w) if geometry is None else geometry
    longest = max(len(e["frames"]) for e in episodes)
    # every stride-th frame, then the final state for HOLD frames as at stride 1
    times = range(longest + HOLD) if stride == 1 else [*range(0, longest, stride)] + [longest - 1] * HOLD
    width = len(episodes) * size[0] + (len(episodes) - 1) * GAP
    height = BAR + HEAD + size[1] + FOOT
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / f"{name}.mp4"
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], stdin=subprocess.PIPE)
    for t in times:
        sheet = Image.new("RGB", (width, height), "white")
        for j, ep in enumerate(episodes):
            sheet.paste(draw_panel(ep, t, m, size), (j * (size[0] + GAP), 0))
        enc.stdin.write(sheet.tobytes())
    enc.stdin.close()
    assert enc.wait() == 0
    sheet.save(out_dir / f"{name}_poster.jpg", quality=poster_quality)
    print(f"  -> {mp4} ({width}x{height}, {len(times)} frames at {FPS} fps)")
    return episodes


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower().replace("π", "pi")).strip("-")


def cell_id(layout, route):
    """Paper layout number and route, e.g. "0d"."""
    return f"{PAPER_LAYOUT[layout]}{route.lower()}"


def library_clips():
    """Every policy, model and cell of the library, with the specs of the clip's panels."""
    for policy, models in LIBRARY_MODELS.items():
        obstacles = LIBRARY_VLA_OBSTACLES if policy == "vla" else LIBRARY_OBSTACLES
        for model, run in models:
            for layout, route in LIBRARY_CELLS:
                cell = cell_id(layout, route)
                specs = [dict(kind="topview" if run else "vla", run=run, task=TASK.format(cls, route),
                              layout=layout, name=name, tag=tag, color=color)
                         for cls, name, tag, color in obstacles]
                yield dict(name=f"{policy}_{slug(model)}_{cell}", policy=policy, model=model, cell=cell,
                           specs=specs)


def cell_geometry():
    """One world-to-panel map per cell, around the paths of every clip of the cell, so
    all models are seen through the same frame."""
    paths = defaultdict(list)
    for clip in library_clips():
        paths[clip["cell"]] += [logged_path(spec) for spec in clip["specs"]]
    return {cell: panel_geometry(p, panel_w=LIBRARY_PANEL_W) for cell, p in paths.items()}


def build_library_clip(clip, out_dir, geometry):
    try:
        episodes = build(clip["name"], clip["specs"], out_dir, crf=LIBRARY_CRF, stride=LIBRARY_STRIDE,
                         panel_w=LIBRARY_PANEL_W, poster_quality=LIBRARY_POSTER_QUALITY,
                         geometry=geometry[clip["cell"]])
    except Exception as err:  # reported after the other clips are built
        return dict(clip=clip["name"], error=f"{type(err).__name__}: {err}")
    return dict(policy=clip["policy"], model=clip["model"], cell=clip["cell"],
                video=f"{clip['name']}.mp4", poster=f"{clip['name']}_poster.jpg",
                panels=[dict(obstacle=ep["name"], tier=ep["tag"], min_clearance=round(ep["min_clearance"], 3),
                             collision=ep["collided"]) for ep in episodes])


def build_library(out_dir, jobs, names=None):
    clips = [c for c in library_clips() if not names or c["name"] in names]
    assert clips, "no library clip matches"
    geometry = cell_geometry()
    for cell, (_, (w, h)) in geometry.items():
        print(f"cell {cell}: panels {w}x{h}")
    with multiprocessing.Pool(jobs) as pool:
        entries = pool.map(partial(build_library_clip, out_dir=out_dir, geometry=geometry), clips, chunksize=1)
    failed = [e for e in entries if "error" in e]
    for e in failed:
        print(f"FAILED {e['clip']}: {e['error']}")
    if names:
        return
    cells = []
    for layout, route in LIBRARY_CELLS:
        _, (w, h) = geometry[cell_id(layout, route)]
        cells.append(dict(id=cell_id(layout, route), layout=PAPER_LAYOUT[layout], route=route,
                          width=3 * w + 2 * GAP, height=BAR + HEAD + h + FOOT))
    manifest = dict(
        seed=LIBRARY_SEED, speed=LIBRARY_STRIDE,
        policies=[dict(id=p, name=POLICY_NAME[p], models=[m for m, _ in models]) for p, models in LIBRARY_MODELS.items()],
        cells=cells, clips=[e for e in entries if "error" not in e])
    with open(out_dir / "library.json", "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print(f"{len(manifest['clips'])} clips, {len(failed)} failed -> {out_dir / 'library.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="default: page/static/videos (with --library: its library/)")
    ap.add_argument("--only", nargs="+", help="build only these videos (with --library: clips, no library.json)")
    ap.add_argument("--library", action="store_true", help="build the results library")
    ap.add_argument("--jobs", type=int, default=8, help="clips built in parallel (--library)")
    args = ap.parse_args()
    videos = ROOT / "page" / "static" / "videos"
    if args.library:
        build_library(args.out or videos / "library", args.jobs, args.only)
        return
    unknown = set(args.only or []) - set(VIDEOS)
    assert not unknown, f"unknown videos {sorted(unknown)}; choose from {sorted(VIDEOS)}"
    args.out = args.out or videos
    for name in args.only or VIDEOS:
        build(name, VIDEOS[name], args.out)


if __name__ == "__main__":
    main()
