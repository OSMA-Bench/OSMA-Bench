#!/usr/bin/env python3
"""Per-task voxposer visualisation — 6-panel layout matching the paper figure.

    (a) Scene             — cropped topview backdrop
    (b) Affordance T      — affordance map overlay (Blues)
    (c) Avoidance O       — avoidance map overlay (Reds)
    (d) Velocity V        — slow-down zones overlay (YlOrRd)
    (e) Cost C            — combined planner cost + planner path
    (f) Trajectory        — executed robot trajectory (magenta ghosts + yaw arrows)

Alignment:
    Uses `topview_corners_uv` saved in voxposer_dump.npz — these are the
    workspace_bounds 4-corners projected through the topview pinhole camera,
    so planner-map pixels land *exactly* on the kitchen floor (no slack
    overflow into walls / outside region).

    If `topview_corners_uv` is missing (legacy dump), falls back to detecting
    the kitchen floor in the image with `kitchen_floor_mask`.

Inputs (under one `task_dir`):
    voxposer_dump.npz       — per-plan-iter map snapshots + corners_uv
    trajectory_log.json     — per-step robot_pos / robot_yaw / velocity / jerk / clearance
    initial_topview.png     — backdrop image (saved flipped; we unflip)

Usage:
    python3 scripts/visualize_voxposer_task.py \\
        outputs/navigation_<model>_<ts>/layout3/NavigateKitchenCatBlockingRouteA
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from visualize_navigation_costmap import (  # noqa: E402
    crop_topview_content,
    kitchen_floor_mask,
    recolor_brown_walls_to_gray,
)

# Display size every workspace crop is resized to (off with --no-resize).
TARGET_W, TARGET_H = 640, 480
# A* never enters a cell whose stored avoidance is above this: hard_avoidance
# in policy/Voxposer/src/modules/planners.py, on in robocasa_config.yaml.
HARD_AVOID_THR = 0.05


# ---- Coordinate transforms ------------------------------------------------

def _planner_corners(map_h, map_w=None):
    """Source corners (cv2 (x,y)=(col,row)) for cv2.getPerspectiveTransform.
    Supports rectangular grid: shape (map_h, map_w) where rows=y_cells,
    cols=x_cells. corners_uv ordering (per interfaces.py):
        [0] = world (ws_min_x, ws_min_y) = planner (row=0, col=0)
        [1] = world (ws_min_x, ws_max_y) = planner (row=map_h-1, col=0)
        [2] = world (ws_max_x, ws_max_y) = planner (row=map_h-1, col=map_w-1)
        [3] = world (ws_max_x, ws_min_y) = planner (row=0, col=map_w-1)

    For rectangular workspaces (map_h ≠ map_w), the legacy scalar map_size
    produced an aspect-ratio mismatch: the planner grid was assumed square
    so the avoidance map got stretched against its true cell aspect.
    """
    if map_w is None:
        map_w = map_h  # legacy square fallback
    Mh = map_h - 1
    Mw = map_w - 1
    return np.array(
        [[0,  0],    # corner[0]: row=0,    col=0      → ws_min_x, ws_min_y
         [0,  Mh],   # corner[1]: row=Mh,   col=0      → ws_min_x, ws_max_y
         [Mw, Mh],   # corner[2]: row=Mh,   col=Mw     → ws_max_x, ws_max_y
         [Mw, 0]],   # corner[3]: row=0,    col=Mw     → ws_max_x, ws_min_y
        dtype=np.float32,
    )


def _build_H_from_corners_uv(corners_uv, map_h, map_w=None):
    """planner-grid → topview-pixel homography given the 4 projected corners."""
    return cv2.getPerspectiveTransform(_planner_corners(map_h, map_w),
                                       corners_uv.astype(np.float32))


def _warp_map(planner_map, H, out_w, out_h, interp=cv2.INTER_LINEAR):
    if planner_map is None:
        return None
    m = np.asarray(planner_map, dtype=np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    return cv2.warpPerspective(
        m, H.astype(np.float32), (out_w, out_h),
        flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan,
    )


def _planner_to_topview(planner_rc, H):
    """planner (row, col) array, shape (N, 2) → topview pixel (x, y)."""
    if planner_rc is None or len(planner_rc) == 0:
        return np.array([]), np.array([])
    arr = np.asarray(planner_rc).reshape(-1, 2)
    rows, cols = arr[:, 0], arr[:, 1]
    pts = np.stack([cols, rows, np.ones_like(rows)], axis=0).astype(np.float64)
    q = H.astype(np.float64) @ pts
    return q[0] / q[2], q[1] / q[2]


def _world_to_planner(xy_world, ws_min, ws_max, map_h, map_w=None):
    """World (x, y) → planner (row, col). row counts y_cells (map_h),
    col counts x_cells (map_w).

    Uses (map_w - 1) / (map_h - 1) for the denominator — consistent with the
    inverse used everywhere else (`_path2traj_navigation`, `pc2pixel_map`,
    `_world_to_pos_coords`). The previous `* map_w` form was off-by-one and
    shifted markers ~0.5-1 cell off the actual grid cell.
    """
    if map_w is None:
        map_w = map_h
    xy = np.asarray(xy_world, dtype=np.float64).reshape(-1, 2)
    rng = np.asarray(ws_max, dtype=np.float64) - np.asarray(ws_min, dtype=np.float64)
    cols = (xy[:, 0] - ws_min[0]) / rng[0] * (map_w - 1)
    rows = (xy[:, 1] - ws_min[1]) / rng[1] * (map_h - 1)
    return np.stack([rows, cols], axis=-1)


def _world_to_topview(xy_world, ws_min, ws_max, map_h, H, map_w=None):
    rc = _world_to_planner(xy_world, ws_min, ws_max, map_h, map_w)
    return _planner_to_topview(rc, H)


# ---- Result lookup --------------------------------------------------------

def _find_task_entry(run_dir, task_rel_dir):
    target = str(task_rel_dir).replace(os.sep, "/")
    for rp in sorted(run_dir.glob("results*.json")):
        try:
            with open(rp) as f:
                data = json.load(f)
        except Exception:
            continue
        for r in data.get("results", []):
            td = (r.get("task_info") or {}).get("task_dir")
            if td and td.replace(os.sep, "/") == target:
                return r
    return None


# ---- Panel decorators -----------------------------------------------------

def _draw_markers(ax, start, goal, final, obstacle=None, obstacle_name=None,
                  keep_label_inside=False):
    def _ok(arr):
        """True if has at least one finite point."""
        return arr[0].size > 0 and np.isfinite(arr[0]).all() and np.isfinite(arr[1]).all()
    if _ok(start):
        ax.scatter(start[0], start[1], marker="*", s=320, c="gold",
                   edgecolors="black", linewidths=1.0, zorder=10)
    if _ok(goal):
        ax.scatter(goal[0], goal[1], marker="o", s=120, c="royalblue",
                   edgecolors="white", linewidths=1.5, zorder=10)
    if _ok(final):
        ax.scatter(final[0], final[1], marker="X", s=140, c="red",
                   edgecolors="white", linewidths=1.0, zorder=10)
        ax.add_patch(plt.Circle(
            (float(final[0][0]), float(final[1][0])), radius=22,
            fill=False, edgecolor="orange", linewidth=2.5, zorder=9))
    if obstacle is not None and _ok(obstacle):
        ax.scatter(obstacle[0], obstacle[1], marker="D", s=180, c="lime",
                   edgecolors="black", linewidths=1.5, zorder=11)
        if obstacle_name:
            ann = ax.annotate(str(obstacle_name),
                              (float(obstacle[0][0]), float(obstacle[1][0])),
                              xytext=(8, 8), textcoords="offset points",
                              fontsize=10, color="black", fontweight="bold",
                              bbox=dict(facecolor="lime", alpha=0.85, edgecolor="black", pad=2),
                              zorder=12)
            if keep_label_inside:
                # Split panels: a label past the right edge would leave the
                # panel or sit under the colorbar, so shift it left just
                # enough to fit (4 pt = bbox pad 2 pt + edge + margin).
                r = ax.figure.canvas.get_renderer()
                over_px = (ann.get_window_extent(r).x1 + 4 * ax.figure.dpi / 72
                           - ax.get_window_extent(r).x1)
                if over_px > 0:
                    ann.xyann = (8 - over_px * 72 / ax.figure.dpi, 8)


def _set_caption(ax, text):
    ax.set_title(text, fontsize=18, fontweight="bold", pad=12,
                 bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=4))


# ---- Main render ----------------------------------------------------------

def _tick_labels(vals):
    """The values with the fewest shared decimals (at most 3) that show each one."""
    vals = [round(float(v), 3) for v in vals]
    d = max((len(f"{v:g}".partition(".")[2]) for v in vals), default=0)
    return [f"{v:.{d}f}" for v in vals]


def render(task_dir, no_crop=False, out_suffix="", split_panels=False, no_smooth=False,
           no_resize=False, colorbar=False, binary_avoidance=False):
    task_dir = Path(task_dir).resolve()
    run_dir = task_dir.parent.parent
    rel = task_dir.relative_to(run_dir)

    # ---- Load voxposer dump
    dump_path = task_dir / "voxposer_dump.npz"
    if not dump_path.exists():
        print(f"[skip] voxposer_dump.npz missing at {task_dir}")
        return
    dump = np.load(dump_path, allow_pickle=True)
    iters = dump["iters"]
    iter0 = iters[0] if len(iters) > 0 else {}
    ws_min = np.asarray(dump["workspace_bounds_min"], dtype=np.float64)
    ws_max = np.asarray(dump["workspace_bounds_max"], dtype=np.float64)
    map_size = int(dump["map_size"])
    # Rectangular grid: prefer map_h/map_w from dump (added 2026-05-09).
    # Fall back to scalar map_size for legacy dumps.
    map_h = int(dump["map_h"]) if "map_h" in dump.files else map_size
    map_w = int(dump["map_w"]) if "map_w" in dump.files else map_size
    corners_uv = dump["topview_corners_uv"] if "topview_corners_uv" in dump.files else None
    # Prefer obstacle_xy_init (captured at load_task = same instant as
    # initial_topview.png) so the marker aligns with the cat as visible in
    # the topview backdrop. Fall back to obstacle_xy (LMP-planning-time) for
    # legacy dumps that didn't save the init field.
    if "obstacle_xy_init" in dump.files:
        obstacle_xy_world = np.asarray(dump["obstacle_xy_init"], dtype=np.float64)
    elif "obstacle_xy" in dump.files:
        obstacle_xy_world = np.asarray(dump["obstacle_xy"], dtype=np.float64)
    else:
        obstacle_xy_world = None
    obstacle_name = (str(dump["obstacle_name"]) if "obstacle_name" in dump.files else None)
    # Prefer fixture.pos for the goal marker if the dumper saved it (newer
    # runs); falls back to evaluation.goal_pos_xy otherwise.
    goal_xy_fixture = (np.asarray(dump["goal_xy_fixture"], dtype=np.float64)
                       if "goal_xy_fixture" in dump.files else None)

    # ---- Load topview backdrop.
    # initial_topview.png is saved with `[::-1]` applied to MuJoCo's raw
    # frame, which converts MuJoCo's bottom-left origin to standard
    # top-left origin. The corners_uv stored in the dump were computed
    # via pinhole projection (top-left origin too), so the loaded PNG
    # already aligns with corners_uv — DO NOT apply another flip here.
    bg_path = task_dir / "initial_topview.png"
    if not bg_path.exists():
        print("[skip] initial_topview.png missing"); return
    bg_full = np.asarray(Image.open(bg_path).convert("RGB")).copy()

    if corners_uv is not None:
        corners_uv = np.asarray(corners_uv, dtype=np.float64)
        if no_crop:
            # No-crop: keep full topview backdrop. Polygon = corners_uv,
            # i.e. the workspace_bounds AABB projected via the actual
            # camera intrinsics (fovy) + extrinsics (cam_pos, cam_xmat) +
            # floor z. This is the geometrically correct image position
            # for each map cell — same as projecting (world_x, world_y,
            # floor_z) for each map corner.
            cropped = bg_full.copy()
            corners_in_cropped = corners_uv.copy()
        else:
            # Topview camera was fit to workspace + 5% per layout. Renderer
            # crops to workspace corners (no extra margin) so panels show the
            # workspace exactly. Then resize to a fixed display size for
            # uniform appearance across all layouts.
            x0_ws = max(0, int(corners_uv[:, 0].min()))
            y0_ws = max(0, int(corners_uv[:, 1].min()))
            x1_ws = min(bg_full.shape[1], int(corners_uv[:, 0].max()))
            y1_ws = min(bg_full.shape[0], int(corners_uv[:, 1].max()))
            cropped = bg_full[y0_ws:y1_ws, x0_ws:x1_ws].copy()
            corners_in_cropped = corners_uv - [x0_ws, y0_ws]
            # Uniform resize so every layout produces panels of equal pixel size.
            # --no-resize keeps the crop's own pixels, so x and y share one scale.
            if not no_resize:
                _ch, _cw = cropped.shape[:2]
                cropped = cv2.resize(cropped, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
                corners_in_cropped = corners_in_cropped * np.array(
                    [TARGET_W / _cw, TARGET_H / _ch])
        H = _build_H_from_corners_uv(corners_in_cropped, map_h, map_w)
        ws_polygon = np.zeros(cropped.shape[:2], dtype=np.uint8)
        cv2.fillPoly(ws_polygon, [corners_in_cropped.astype(np.int32)], 1)
        ws_polygon = ws_polygon.astype(bool)
    else:
        # Fallback for legacy dumps without projected corners
        cropped, _, _ = crop_topview_content(bg_full)
        floor = kitchen_floor_mask(cropped)
        if floor.any():
            ys, xs = np.where(floor)
            f_x0, f_y0 = int(xs.min()), int(ys.min())
            f_x1, f_y1 = int(xs.max()), int(ys.max())
        else:
            f_x0, f_y0 = 0, 0
            f_x1, f_y1 = cropped.shape[1] - 1, cropped.shape[0] - 1
        floor_corners = np.array(
            [[f_x1, f_y1], [f_x1, f_y0], [f_x0, f_y0], [f_x0, f_y1]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(_planner_corners(map_h, map_w), floor_corners)
        ws_polygon = floor

    cropped = recolor_brown_walls_to_gray(cropped)
    H_full, W_full = cropped.shape[:2]

    # Show the map on the entire polygon (= floor_mask AABB). The polygon
    # itself is already aligned to visible floor extents (set above when
    # no_crop), so the map cells map to the correct image-xy positions. We
    # do NOT intersect with the irregular floor_mask anymore — that hid
    # obstacle cells that fall on counter pixels and made the user say
    # "invisible" (per user feedback).
    map_mask = ws_polygon

    # ---- Maps in cropped pixel coords (no extra image warp; backdrop keeps
    # the full topview so walls/cabinets are visible like the paper figure).
    affordance = iter0.get("affordance_map")
    avoidance  = iter0.get("avoidance_map")
    velocity   = iter0.get("velocity_map")
    path_pixel = iter0.get("path_pixel")

    # Affordance smoothed for visibility (off with --no-smooth) — set_pixel_by_radius disks
    # cover only ~1% of grid (e.g. 130 cells in 162×162) and read as a few
    # pixels post-warp. Stronger Gaussian (sigma=12) creates a soft halo
    # that's recognisable at panel scale without changing planner data.
    from scipy.ndimage import gaussian_filter
    aff_for_display = None
    if affordance is not None:
        af_arr = np.asarray(affordance, dtype=np.float32)
        if (af_arr > 0).any() and not no_smooth:
            smoothed = gaussian_filter(af_arr, sigma=12.0)
            peak = float(smoothed.max())
            aff_for_display = smoothed / peak if peak > 0 else smoothed
        else:
            aff_for_display = af_arr

    # Velocity slowdown zones are also sparse (a few set_pixel_by_radius
    # disks). Smooth `slow = 1 - v` so the panel reads as dense as
    # avoidance. We feed the SMOOTHED slow back into v_for_display
    # = 1 - slow_smoothed so the imshow path stays unchanged.
    v_for_display = velocity
    if velocity is not None and not no_smooth:
        v_arr = np.asarray(velocity, dtype=np.float32)
        slow = 1.0 - v_arr
        slow_smoothed = gaussian_filter(slow, sigma=4.0)
        peak = float(slow_smoothed.max())
        if peak > 0:
            slow_smoothed = slow_smoothed / peak * float(slow.max())
        v_for_display = 1.0 - slow_smoothed

    # The stored avoidance halo is graded: set_pixel_by_radius decays it
    # linearly from 1 at the mesh to 0 at the commanded radius. With
    # --binary-avoidance, draw instead the cells above the hard_avoidance
    # threshold (A* dilates this set by the robot radius, not drawn); (e)
    # then uses the same map.
    av_for_display = avoidance
    if avoidance is not None and binary_avoidance:
        av_for_display = (np.asarray(avoidance, dtype=np.float32)
                          > HARD_AVOID_THR).astype(np.float32)

    # --no-smooth: stored values, nearest cell (no blur, no interpolated edges).
    _step_interp = cv2.INTER_NEAREST if no_smooth else cv2.INTER_LINEAR
    # A binary avoidance map stays two-valued: nearest cell even without --no-smooth.
    _av_interp = cv2.INTER_NEAREST if binary_avoidance else _step_interp
    af_warp = _warp_map(aff_for_display, H, W_full, H_full, _step_interp)
    av_warp = _warp_map(av_for_display,   H, W_full, H_full, _av_interp)
    v_warp  = _warp_map(v_for_display,    H, W_full, H_full, _step_interp)
    cost = None
    if affordance is not None and avoidance is not None:
        af = np.asarray(affordance, dtype=np.float32)
        av = np.asarray(av_for_display, dtype=np.float32)
        cost = av - af
        rng = np.nanmax(cost) - np.nanmin(cost)
        if rng > 1e-9:
            cost = (cost - np.nanmin(cost)) / rng
    cost_warp = _warp_map(cost, H, W_full, H_full, _av_interp)
    for m in (af_warp, av_warp, v_warp, cost_warp):
        if m is not None:
            m[~map_mask] = np.nan

    # Aliases so the plotting block below stays the same regardless of frame.
    bg_sq, af_sq, av_sq, v_sq, cost_sq = cropped, af_warp, av_warp, v_warp, cost_warp

    px, py = _planner_to_topview(path_pixel, H)

    # ---- Trajectory + markers
    entry = _find_task_entry(run_dir, rel) or {}
    ti = entry.get("task_info") or {}
    ev = entry.get("evaluation") or {}
    start = _world_to_topview(ev.get("start_pos_xy", [np.nan, np.nan]),
                              ws_min, ws_max, map_h, H, map_w)
    # If results.json hasn't been written yet, fall back to iter0.start_pos
    # (planner cell coords) so the ROBOT marker still appears.
    if isinstance(iter0, dict):
        _sp_iter = iter0.get('start_pos')
        if (_sp_iter is not None
                and (start[0].size == 0 or not np.isfinite(start[0]).any())):
            _sp_arr = np.asarray(_sp_iter).reshape(-1)
            if len(_sp_arr) >= 2:
                # planner cell (row, col) → image px via H
                _sx, _sy = _planner_to_topview(
                    np.asarray([[_sp_arr[0], _sp_arr[1]]]).astype(float), H)
                start = (_sx, _sy)
    goal  = _world_to_topview(ev.get("goal_pos_xy",  [np.nan, np.nan]),
                              ws_min, ws_max, map_h, H, map_w)
    if goal_xy_fixture is not None and (goal[0].size == 0 or not np.isfinite(goal[0]).any()):
        goal = _world_to_topview(np.asarray(goal_xy_fixture, dtype=float),
                                 ws_min, ws_max, map_h, H, map_w)
    final = _world_to_topview(ev.get("final_pos_xy", [np.nan, np.nan]),
                              ws_min, ws_max, map_h, H, map_w)
    obstacle_uv = (_world_to_topview(obstacle_xy_world, ws_min, ws_max, map_h, H, map_w)
                   if obstacle_xy_world is not None else (np.array([]), np.array([])))

    ts = {}
    ts_path = task_dir / "trajectory_log.json"
    if ts_path.exists():
        with open(ts_path) as f:
            ts = json.load(f)
    robot_pos_world = np.asarray(ts.get("robot_pos") or []).reshape(-1, 2)
    robot_yaw       = np.asarray(ts.get("robot_yaw") or [])
    tx, ty = (np.array([]), np.array([]))
    if len(robot_pos_world):
        tx, ty = _world_to_topview(robot_pos_world, ws_min, ws_max, map_h, H, map_w)

    # Helper alias so the same plotting code works
    def _world_to_sq(xy_world):
        return _world_to_topview(xy_world, ws_min, ws_max, map_h, H, map_w)

    # ---- Plot 6 panels ----------------------------------------------------
    # Figsize chosen so each panel ends up at the SAME pixel scale —
    # 3 cols × 2 rows of (W, H) panels + title band + LLM text band at
    # bottom. Use the actual backdrop aspect to set figure ratio so
    # ax.set_aspect('equal') doesn't shrink top vs bottom row asymmetrically.
    _bg_h, _bg_w = bg_sq.shape[:2]
    _fig_w_in = 15.0
    _panel_h_in = (_fig_w_in / 3.0) * (_bg_h / _bg_w)  # height of 1 panel row
    _fig_h_in = _panel_h_in * 2.0 + 1.4 + 4.0  # 2 rows + title + LLM text band
    if split_panels:
        # One figure per panel, the axes about the size of a panel in the
        # combined figure (see subplots_adjust below) so markers and lines keep
        # their relative size. No captions, colorbar or text bands. The 3 pt
        # margin keeps the cyan workspace outline: its right/bottom side can
        # sit just past the axes (crop truncated to whole pixels), and Agg
        # drops path segments that lie past the canvas edge.
        _split_w_in = _fig_w_in * (0.98 - 0.02) / (3 + 2 * 0.04)
        _split_h_in = _split_w_in * _bg_h / _bg_w
        _quiver_w = 0.006
        if no_resize:
            # Same height as a resized panel; the width follows the crop.
            # Arrow width is a fraction of the axes width, so scale it back.
            _split_h_in = _split_w_in * TARGET_H / TARGET_W
            _quiver_w *= _split_w_in / (_split_h_in * _bg_w / _bg_h)
            _split_w_in = _split_h_in * _bg_w / _bg_h
        _pad_in = 3 / 72
        # --colorbar: a bar beside (b)-(e). All six canvases get the same room
        # on the right (blank for (a) and (f)), so the panels keep one scale
        # when placed at equal widths. The room is trimmed to the widest bar
        # and its labels at save time.
        _cb_gap_in, _cb_w_in = 0.08, 0.12
        _right_in = 1.5 if colorbar else 0.0
        _fw = _split_w_in + 2 * _pad_in + _right_in
        _fh = _split_h_in + 2 * _pad_in
        _split_figs = [plt.figure(figsize=(_fw, _fh)) for _ in range(6)]
        axes = np.array([f.add_axes([_pad_in / _fw, _pad_in / _fh,
                                     _split_w_in / _fw, _split_h_in / _fh])
                         for f in _split_figs], dtype=object).reshape(2, 3)
    else:
        _quiver_w = 0.006
        fig, axes = plt.subplots(2, 3, figsize=(_fig_w_in, _fig_h_in),
                                 gridspec_kw={'height_ratios': [1, 1]},
                                 constrained_layout=False)
    captions = [
        r"(a) Scene",
        r"(b) Affordance $\mathcal{T}$",
        r"(c) Avoidance $\mathcal{O}$",
        r"(d) Velocity $\mathcal{V}$",
        r"(e) Cost $\mathcal{C}$",
        r"(f) Trajectory",
    ]
    for ax, cap in zip(axes.flat, captions):
        ax.imshow(bg_sq)
        if not split_panels:
            _set_caption(ax, cap)
        ax.set_xticks([]); ax.set_yticks([])
        # Force axis limits to backdrop image — prevents NaN markers /
        # off-image patches from expanding limits and squeezing the
        # entire image into a corner.
        ax.set_xlim(0, _bg_w)
        ax.set_ylim(_bg_h, 0)
        # Force equal aspect so subplot resizing doesn't stretch X/Y
        # differently — otherwise scatter markers (start, goal, cat,
        # etc.) and overlays land at wrong panel pixel positions.
        ax.set_aspect('equal', adjustable='box')
        if corners_in_cropped is not None:
            poly = corners_in_cropped.astype(np.int32)
            ax.plot(np.append(poly[:, 0], poly[0, 0]),
                    np.append(poly[:, 1], poly[0, 1]),
                    color="cyan", linewidth=2.0, zorder=5)

    # Unify b/c/d to the same visual density as Avoidance: alpha=0.65 +
    # per-panel normalisation to its own [vmin, vmax] so a panel with a
    # narrow value range still reads as bold as one with values up to 1.
    OVERLAY_ALPHA = 0.65
    def _vminmax(m):
        if m is None: return 0.0, 1.0
        finite = m[~np.isnan(m)]
        if finite.size == 0: return 0.0, 1.0
        lo, hi = float(finite.min()), float(finite.max())
        return lo, max(hi, lo + 1e-6)
    # --no-smooth: no display filter either, so a cell edge stays a step.
    _ov = dict(interpolation="nearest") if no_smooth else {}
    _ov_av = dict(interpolation="nearest") if (no_smooth or binary_avoidance) else {}

    _caxes = []
    def _colorbar(ax, im, label, flip=False, tick_labels=None):
        """--colorbar: a labelled bar beside a split panel, in the image's own
        colours. A map with a few values gets one equal block per value. flip:
        the image shows 1 - value (velocity), so ticks read the value and it
        grows upward."""
        if not (split_panels and colorbar):
            return
        fig = ax.figure
        cax = fig.add_axes([(_pad_in + _split_w_in + _cb_gap_in) / _fw, _pad_in / _fh,
                            _cb_w_in / _fw, _split_h_in / _fh])
        _caxes.append(cax)
        data = np.asarray(im.get_array(), dtype=float)
        vals = np.unique(data[np.isfinite(data)])
        if 0 < len(vals) <= 6:
            order = vals[::-1] if flip else vals          # bottom -> top
            colors = [(*im.cmap(im.norm(v))[:3], im.get_alpha()) for v in order]
            sm = plt.cm.ScalarMappable(
                cmap=ListedColormap(colors),
                norm=BoundaryNorm(np.arange(len(order) + 1) - 0.5, len(order)))
            cb = fig.colorbar(sm, cax=cax, ticks=np.arange(len(order)))
            if tick_labels is None or len(tick_labels) != len(order):
                tick_labels = _tick_labels(1.0 - order if flip else order)
            cb.set_ticklabels(tick_labels)
        else:
            cb = fig.colorbar(im, cax=cax)
            if flip:
                cb.ax.invert_yaxis()
                cb.ax.yaxis.set_major_formatter(
                    FuncFormatter(lambda s, _: f"{round(1.0 - s, 3):g}"))
        cb.set_label(label, fontsize=10)
        cb.ax.tick_params(labelsize=9)

    if af_sq is not None:
        lo, hi = _vminmax(af_sq)
        im = axes[0, 1].imshow(af_sq, cmap="Blues", alpha=OVERLAY_ALPHA, vmin=lo, vmax=hi, **_ov)
        _colorbar(axes[0, 1], im, r"Affordance $\mathcal{T}$")
    if av_sq is not None:
        lo, hi = _vminmax(av_sq)
        im = axes[0, 2].imshow(av_sq, cmap="Reds", alpha=OVERLAY_ALPHA, vmin=lo, vmax=hi, **_ov_av)
        _colorbar(axes[0, 2], im, r"Avoidance $\mathcal{O}$",
                  tick_labels=([f"$\\leq${HARD_AVOID_THR:g}", f"$>${HARD_AVOID_THR:g}"]
                               if binary_avoidance else None))
    if v_sq is not None:
        slow = 1.0 - v_sq
        lo, hi = _vminmax(slow)
        im = axes[1, 0].imshow(slow, cmap="Oranges", alpha=OVERLAY_ALPHA, vmin=lo, vmax=hi, **_ov)
        _colorbar(axes[1, 0], im, r"Velocity $\mathcal{V}$", flip=True)
    if cost_sq is not None:
        cost_im = axes[1, 1].imshow(cost_sq, cmap="viridis", alpha=0.55, **_ov_av)
        _colorbar(axes[1, 1], cost_im, r"Cost $\mathcal{C}$")
        # Colorbar attached DIRECTLY below the cost axes (no figure-coordinate
        # math). make_axes_locatable adjusts the cost axes itself so the
        # colorbar is rendered immediately under it, regardless of the LLM
        # text band reserved at the figure bottom.
        if not split_panels:
            divider = make_axes_locatable(axes[1, 1])
            cax = divider.append_axes("bottom", size="3%", pad=0.05)
            cb = fig.colorbar(cost_im, cax=cax, orientation='horizontal')
            cb.set_label("cost (LOW=goal zone, HIGH=avoid)", fontsize=8)
            cb.ax.tick_params(labelsize=7)
    if px.size:
        axes[1, 1].plot(px, py, "-", color="red", linewidth=1.6, zorder=4)
        axes[1, 1].scatter(px, py, s=18, c="red", edgecolors="white",
                           linewidths=0.6, zorder=5)
    if tx.size:
        axes[1, 2].plot(tx, ty, "-", color="magenta", linewidth=1.0,
                        alpha=0.4, zorder=3)

    # Key waypoints (planner-output traj_world) — sparse, with yaw arrows.
    wp_world = iter0.get("traj_world")
    wp_yaw   = iter0.get("traj_world_yaw")
    if wp_world is not None and len(wp_world):
        wp_world = np.asarray(wp_world)[:, :2]
        wpx, wpy = _world_to_sq(wp_world)
        n_wp = len(wpx)
        for i in range(n_wp):
            a = 0.25 + 0.6 * (i / max(1, n_wp - 1))
            axes[1, 2].scatter(wpx[i], wpy[i], s=320, c="magenta",
                               alpha=a, edgecolors="none", zorder=4)
        if wp_yaw is not None and len(wp_yaw) == n_wp:
            yaws = np.asarray(wp_yaw)
            eps = 0.22
            fwd = wp_world.copy()
            fwd[:, 0] += eps * np.cos(yaws)
            fwd[:, 1] += eps * np.sin(yaws)
            fx, fy = _world_to_sq(fwd)
            axes[1, 2].quiver(
                wpx, wpy, fx - wpx, fy - wpy,
                angles="xy", scale_units="xy", scale=1.0,
                color="cyan", width=_quiver_w, headwidth=5.0, headlength=6.0,
                alpha=0.95, zorder=6,
            )

    # Show start / goal / final markers ONLY on the (f) Trajectory panel
    # — they describe robot motion, not the per-panel cost field. Keeping
    # them off (b)–(e) declutters the maps so the value field is what the
    # eye reads.
    _draw_markers(axes[1, 2], start, goal, final, obstacle_uv, obstacle_name,
                  keep_label_inside=split_panels)
    # Also overlay start (robot) + goal + obstacle on (c) Avoidance so the
    # avoidance map context (where robot is, where target is, where the
    # obstacle is) is clear without flipping back to (f).
    _draw_markers(axes[0, 2], start, goal,
                  (np.array([]), np.array([])),
                  obstacle_uv, obstacle_name, keep_label_inside=split_panels)

    if split_panels:
        _names = ["a_scene", "b_affordance", "c_avoidance",
                  "d_velocity", "e_cost", "f_trajectory"]
        _bbox = None
        if _caxes:
            # One canvas for all six: cut right of the widest bar's labels, and
            # taller only if an end tick label runs past the top or bottom.
            _tb = [c.get_tightbbox(c.figure.canvas.get_renderer())
                    .transformed(c.figure.dpi_scale_trans.inverted()) for c in _caxes]
            _bbox = Bbox.from_extents(
                0, min(0, min(b.y0 for b in _tb) - 1 / 72),
                max(b.x1 for b in _tb) + _pad_in, max(_fh, max(b.y1 for b in _tb) + 1 / 72))
        for f, name in zip(_split_figs, _names):
            out = task_dir / f"voxposer_panel_{name}{out_suffix}.png"
            f.savefig(out, dpi=300, bbox_inches=_bbox)
            plt.close(f)
            print(f"saved {out}")
        return
    # Obstacle disk overlay was removed — obstacle is already visible in
    # panel (a) Scene and its avoidance halo shows up in panel (c).

    def _xy(s):
        try:
            v = list(s) if s is not None else []
            return f"({v[0]:.2f}, {v[1]:.2f})" if len(v) >= 2 else "(?, ?)"
        except Exception:
            return "(?, ?)"
    # Fallbacks when results.json isn't ready yet (smoke still running):
    # parse from task_dir name and pull from dump.
    import re as _re
    _m = _re.match(r"layout(\d+)/(.+)", str(rel))
    _layout_fb = int(_m.group(1)) if _m else None
    _task_name_fb = _m.group(2) if _m else task_dir.name
    _route_fb = None
    _rm = _re.search(r"Route([A-Z])", _task_name_fb)
    if _rm: _route_fb = "Route" + _rm.group(1)
    _obstacle_fb = obstacle_name or "?"
    # start_pos: dump iter0 has planner-cell start_pos, convert to world.
    _start_xy_fb = ev.get('start_pos_xy')
    if _start_xy_fb is None and isinstance(iter0, dict):
        _sp = iter0.get('start_pos')
        if _sp is not None:
            _sp = np.asarray(_sp).reshape(-1)
            if len(_sp) >= 2:
                # planner cell (row, col) → world (x, y)
                _start_xy_fb = [
                    float(ws_min[0] + (_sp[1] / max(map_w, 1)) * (ws_max[0] - ws_min[0])),
                    float(ws_min[1] + (_sp[0] / max(map_h, 1)) * (ws_max[1] - ws_min[1])),
                ]
    _goal_xy_fb = ev.get('goal_pos_xy')
    if _goal_xy_fb is None and goal_xy_fixture is not None:
        _goal_xy_fb = list(np.asarray(goal_xy_fixture, dtype=float))

    _task_name = ti.get('task_name', _task_name_fb)
    title_l1 = (f"obstacle={ti.get('obstacle', _obstacle_fb)}  |  "
                f"{ti.get('safety_mode','safety_demanding')}")
    instr = ti.get('instruction') or ''
    title_l2 = f"task:  {instr}" if instr else ""
    title_l3 = (f"start={_xy(_start_xy_fb)}  →  "
                f"goal={_xy(_goal_xy_fb)}  "
                f"(layout={ti.get('layout_id', _layout_fb)}, "
                f"style={ti.get('style_id', 3)}, "
                f"route={ti.get('route', _route_fb)})")
    title_l4 = ("markers:    yellow ★ = initial (start_pos)    "
                "blue ● = target (goal_pos)    "
                "red ✕ = final pose (where the robot stopped)")
    title_lines = [title_l1]
    if title_l2: title_lines.append(title_l2)
    title_lines += [title_l3, title_l4]
    # Big task name banner on top, then smaller meta lines.
    fig.text(0.5, 0.997, _task_name, ha='center', va='top',
             fontsize=18, fontweight='bold')
    fig.suptitle("\n".join(title_lines), fontsize=11, y=0.99)
    # Use constrained_layout-like manual placement to ensure ALL panels
    # have the same axes-box size. Otherwise tight_layout + ax.set_aspect
    # produced asymmetric top vs bottom row heights, which made the same
    # obstacle_uv land at different panel-pixel positions in (c) and (f).
    # Manual placement: title at top (0.97-1.0), panels at top portion,
    # text band at bottom 25% of figure. With aspect='equal' the panels
    # will shrink uniformly inside the allotted region.
    _title_band = 1.4 / _fig_h_in
    _text_band = 4.0 / _fig_h_in
    _panel_top = 1.0 - _title_band
    _panel_bot = _text_band
    fig.subplots_adjust(left=0.02, right=0.98,
                        top=_panel_top, bottom=_panel_bot,
                        hspace=0.05, wspace=0.04)
    # Append LLM inference text below the panel grid in TWO columns:
    # LEFT  = composer + get_affordance_map
    # RIGHT = get_avoidance_map + get_velocity_map + get_rotation_map
    _left_sections, _right_sections = [], []
    try:
        _log = task_dir / "run.log"
        if _log.exists():
            _content = _log.read_text(errors="ignore")
            import re as _re_log
            # composer-only output, left-aligned. No truncation.
            for _which in ("composer",):
                _matches = list(_re_log.finditer(
                    rf'## "{_which}" generated code.*?\n#+\n(.*?)\n#+', _content, _re_log.DOTALL))
                if _matches:
                    _code = _matches[-1].group(1).strip()
                    _block = f"[{_which}]\n{_code}"
                    _left_sections.append(_block)
    except Exception:
        pass
    # Wrap long lines in LLM code so they don't bleed into the right column.
    def _wrap(_blocks, _width):
        import textwrap
        out = []
        for blk in _blocks:
            wrapped = []
            for ln in blk.split("\n"):
                if len(ln) <= _width:
                    wrapped.append(ln)
                else:
                    wrapped.extend(textwrap.wrap(ln, width=_width,
                                                 break_long_words=False,
                                                 subsequent_indent="    "))
            out.append("\n".join(wrapped))
        return "\n\n".join(out)

    if _left_sections:
        # composer-only, left-aligned at the leftmost panel's left edge
        # (= (a) Scene panel x0). No right margin — text wraps wider so it
        # extends across the figure width.
        TEXT_TOP_Y = 0.25
        _left_x = axes[0, 0].get_position().x0
        fig.text(_left_x, TEXT_TOP_Y, _wrap(_left_sections, 170), fontsize=10,
                 family='monospace', va='top', ha='left')
    out = task_dir / f"voxposer_overview{out_suffix}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir", help="path to outputs/.../layout{L}/{TaskName}")
    ap.add_argument("--no-crop", action="store_true",
                    help="do not crop topview to workspace AABB; show full frame")
    ap.add_argument("--out-suffix", default="",
                    help="append suffix to output filename (e.g. _nocrop)")
    ap.add_argument("--split-panels", action="store_true",
                    help="save each panel (a)-(f) as its own PNG, without "
                         "captions or colorbar, instead of the overview")
    ap.add_argument("--no-smooth", action="store_true",
                    help="draw the maps as stored: no display blur of affordance "
                         "(b) and velocity (d), nearest-cell warp and display")
    ap.add_argument("--no-resize", action="store_true",
                    help="with --split-panels: keep the workspace crop at its own "
                         "pixels instead of resizing it to 640x480, so x and y "
                         "share one scale")
    ap.add_argument("--colorbar", action="store_true",
                    help="with --split-panels: a labelled colorbar beside (b)-(e); "
                         "all six PNGs get the same canvas")
    ap.add_argument("--binary-avoidance", action="store_true",
                    help=f"draw avoidance (c) as two values, stored value > "
                         f"{HARD_AVOID_THR:g} (the hard_avoidance threshold A* "
                         f"blocks; its robot-radius dilation is not drawn) or "
                         f"not, instead of the graded halo; (e) uses the same map")
    args = ap.parse_args()
    if (args.no_resize or args.colorbar) and not args.split_panels:
        ap.error("--no-resize and --colorbar need --split-panels")
    render(args.task_dir, no_crop=args.no_crop, out_suffix=args.out_suffix,
           split_panels=args.split_panels, no_smooth=args.no_smooth,
           no_resize=args.no_resize, colorbar=args.colorbar,
           binary_avoidance=args.binary_avoidance)


if __name__ == "__main__":
    main()
