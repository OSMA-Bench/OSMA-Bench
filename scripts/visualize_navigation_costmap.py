"""
Navigation-Voxposer pipeline figure: REAL maps from a live RoboCasa run.

  Row 1:  (a) Scene (topview)   (b) Affordance T   (c) Avoidance O
  Row 2:  (d) Velocity V        (e) Cost C + path  (f) Trajectory + keyframes

Maps are extracted from a fresh `NavigateKitchenCatBlockingRouteA` env reset
(see scripts/extract_real_maps.py) so target_v / obstacle_v / costmap /
smoothed_path / velocity_map are exactly what the planner produces in a real
run, not synthesized. Only the closed-loop trajectory and 6 keyframes are
re-simulated for panel (f).

Inputs (from /tmp/real_maps): meta.json, target_v.npy, obstacle_v.npy,
costmap.npy, smoothed_path.npy, velocity_map.npy, scene_collision_pixel.npy,
avoidance_map_preprocessed.npy, affordance_map.npy, topview.png.
"""
import os
import json
import numpy as np
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Circle, FancyArrowPatch
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import repo_root  # noqa: E402

REAL = os.environ.get('REAL_MAPS_DIR',
                       str(repo_root() / "tmp/real_maps"))
MAP_SIZE = 100
SAVGOL_WIN = 20
SAVGOL_ORDER = 3


def normalize_map(m):
    d = m.max() - m.min()
    return m if d == 0 else (m - m.min()) / d


def detect_kitchen_room_bbox(topview_rgb):
    """Tight bbox of the kitchen room (brown walls + gray floor blob)."""
    r = topview_rgb[..., 0].astype(int)
    g = topview_rgb[..., 1].astype(int)
    b = topview_rgb[..., 2].astype(int)
    brown = (r > 60) & (r < 200) & (g > 30) & (g < 160) & (b < 130) \
            & (r > g + 8) & (g > b + 4)
    luma = (r + g + b) / 3.0
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    floor = (luma > 60) & (luma < 170) & (sat < 35)
    interior = brown | floor
    from scipy.ndimage import label, binary_closing
    interior = binary_closing(interior, iterations=3)
    lbl, n = label(interior)
    if n == 0:
        ys, xs = np.where(np.any(topview_rgb < 240, axis=2))
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    main = int(sizes.argmax())
    ys, xs = np.where(lbl == main)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_topview_content(topview_rgb, bounds_corners_uv=None, pad=0):
    """Tight crop to the kitchen room (brown walls + floor connected blob)."""
    x0, y0, x1, y1 = detect_kitchen_room_bbox(topview_rgb)
    return topview_rgb[y0:y1, x0:x1], x0, y0


def recolor_brown_walls_to_gray(rgb, gray=(180, 180, 180)):
    """Recolour brownish wall pixels (R > G > B with red dominant) to a
    uniform light gray for a cleaner figure border."""
    out = rgb.copy()
    r = out[..., 0].astype(int)
    g = out[..., 1].astype(int)
    b = out[..., 2].astype(int)
    brown = (r > 60) & (r < 200) & (g > 30) & (g < 160) & (b > 10) & (b < 130) \
            & (r > g + 8) & (g > b + 4) & (r - b > 20)
    out[brown] = np.array(gray, dtype=out.dtype)
    return out


def planner_to_topview_xy(row, col, H_planner_to_uv, dx=0, dy=0):
    """Transform planner pixel (row, col) → cropped-topview pixel (x, y)."""
    p = np.array([col, row, 1.0])
    q = H_planner_to_uv @ p
    u, v = q[0] / q[2], q[1] / q[2]
    return float(u - dx), float(v - dy)


def warp_planner_map_to_topview(heat, H_planner_to_uv, out_w, out_h, dx=0, dy=0):
    """Warp a planner-grid heatmap into cropped-topview pixel space."""
    T = np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1]], dtype=np.float64)
    H_full = T @ H_planner_to_uv
    return cv2.warpPerspective(
        heat.astype(np.float32), H_full.astype(np.float32), (out_w, out_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan,
    )


def kitchen_floor_mask(topview_rgb):
    """Detect the gray floor pixels of the kitchen room. Returns a 2D bool
    mask the same size as topview_rgb. Used to clip heatmap overlays so they
    don't extend past the visible floor (workspace_bounds is wider than the
    actual room)."""
    r = topview_rgb[..., 0].astype(int)
    g = topview_rgb[..., 1].astype(int)
    b = topview_rgb[..., 2].astype(int)
    luma = (r + g + b) / 3.0
    # gray-ish: low saturation, mid luma
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    floor = (luma > 60) & (luma < 170) & (sat < 35)
    # morphology to fill small holes
    from scipy.ndimage import binary_closing, binary_fill_holes
    floor = binary_closing(floor, iterations=4)
    floor = binary_fill_holes(floor)
    return floor


def detect_cat_visual_centroid_cropped(cropped_rgb):
    """Orange cat centroid directly in cropped-topview pixel coords (x, y)."""
    r = cropped_rgb[..., 0].astype(int)
    g = cropped_rgb[..., 1].astype(int)
    b = cropped_rgb[..., 2].astype(int)
    orange = (r > 180) & (g > 100) & (g < 180) & (b < 130) & (r > b + 60)
    if orange.sum() == 0:
        return None
    ys, xs = np.where(orange)
    return float(xs.mean()), float(ys.mean())


def load_real_maps():
    with open(os.path.join(REAL, 'meta.json')) as f:
        meta = json.load(f)
    M = {
        'target_v': np.load(os.path.join(REAL, 'target_v.npy')),
        'obstacle_v': np.load(os.path.join(REAL, 'obstacle_v.npy')),
        'costmap': np.load(os.path.join(REAL, 'costmap.npy')),
        'velocity': np.load(os.path.join(REAL, 'velocity_map.npy')),
        'avoidance_pre': np.load(os.path.join(REAL, 'avoidance_map_preprocessed.npy')),
        'affordance': np.load(os.path.join(REAL, 'affordance_map.npy')),
        'scene_collision': np.load(os.path.join(REAL, 'scene_collision_pixel.npy')),
        'smoothed_path': np.load(os.path.join(REAL, 'smoothed_path.npy')),
    }
    raw = np.array(Image.open(os.path.join(REAL, 'topview.png')))
    M['topview_raw'] = raw
    cropped, dx, dy = crop_topview_content(raw, pad=0)
    M['topview_cropped'] = cropped
    M['crop_offset'] = (dx, dy)
    floor = kitchen_floor_mask(cropped)
    M['floor_mask'] = floor

    # ALIGNMENT CORRECTION: build H from the actual visible floor extent
    # (not workspace_bounds, which projects past the kitchen).  Maps planner
    # grid → floor bbox in cropped coords so heatmap fills the visible floor
    # exactly and markers land on rendered features.
    fy_idx, fx_idx = np.where(floor)
    if len(fy_idx) > 0:
        f_x0 = int(fx_idx.min()); f_x1 = int(fx_idx.max())
        f_y0 = int(fy_idx.min()); f_y1 = int(fy_idx.max())
        M['floor_bbox'] = (f_x0, f_y0, f_x1, f_y1)
    else:
        M['floor_bbox'] = (0, 0, cropped.shape[1] - 1, cropped.shape[0] - 1)

    # H_planner_to_uv (raw topview) = camera projection — used for visual
    # cat detection only.
    src = np.array(meta['topview_bounds_corners_uv'], dtype=np.float32)
    planner_corners = np.array([[0, 0], [0, MAP_SIZE - 1],
                                 [MAP_SIZE - 1, MAP_SIZE - 1],
                                 [MAP_SIZE - 1, 0]], dtype=np.float32)
    H_uv_to_planner = cv2.getPerspectiveTransform(src, planner_corners)
    H_planner_to_uv = cv2.getPerspectiveTransform(planner_corners, src)

    # H_planner_to_cropped maps planner grid → visible kitchen FLOOR bbox in
    # cropped coords.  This anchors heatmap + markers to what's actually
    # rendered, instead of to workspace_bounds (which extends past the room).
    # Orientation: planner row=0 is xmin (low world x = bottom of topview);
    # planner col=0 is ymin (low world y = right of topview).
    f_x0, f_y0, f_x1, f_y1 = M['floor_bbox']
    floor_bbox_corners = np.array([
        [f_x1, f_y1],  # planner (0, 0) = (xmin, ymin) → bottom-right
        [f_x1, f_y0],  # planner (0, 99) = (xmax, ymin) → top-right
        [f_x0, f_y0],  # planner (99, 99) = (xmax, ymax) → top-left
        [f_x0, f_y1],  # planner (99, 0) = (xmin, ymax) → bottom-left
    ], dtype=np.float32)
    H_planner_to_cropped = cv2.getPerspectiveTransform(
        planner_corners, floor_bbox_corners
    ).astype(np.float64)
    M['H_uv_to_planner'] = H_uv_to_planner
    M['H_planner_to_uv'] = H_planner_to_uv
    M['H_planner_to_cropped'] = H_planner_to_cropped
    # Visual orange detection only for Cat / Kettlebell tasks (other tasks
    # have non-orange obstacles where this would catch decorations).
    env_name = meta.get('env_name', '')
    if any(k in env_name for k in ('Cat', 'Kettlebell')):
        M['cat_visual_cropped_xy'] = detect_cat_visual_centroid_cropped(cropped)
    else:
        M['cat_visual_cropped_xy'] = None
    return M, meta


def closed_loop_execution(start, target_raw, obstacle_v_raw, max_iters=4):
    """Closed-loop replanning that reuses the real obstacle_v / target gradient.

    Replans by re-running greedy descent on the existing costmap from the
    current robot position, advancing to the waypoint that best reduces the
    (real) target EDT.
    """
    edt_target = distance_transform_edt(1 - target_raw)
    cur = start.copy()
    full_traj = [cur.copy()]
    iter_segments = []
    half = max(1, int(2 * MAP_SIZE / 100))
    offsets = np.arange(-half, half + 1)
    grid = np.array(np.meshgrid(offsets, offsets)).T.reshape(-1, 2)
    grid = grid[np.any(grid != [0, 0], axis=1)]

    for _ in range(max_iters):
        # rebuild cost from real value maps (T from EDT, O from sim) — fast
        T = normalize_map(edt_target)
        cost = normalize_map(2.0 * T + 20.0 * obstacle_v_raw)
        c = cost.copy()
        path = [cur.copy()]
        cp = cur.copy()
        for _ in range(300):
            nb = np.unique(np.clip(cp + grid, 0, MAP_SIZE - 1).astype(int), axis=0)
            sc = c[nb[:, 0], nb[:, 1]]
            nx = nb[np.argmin(sc)].astype(float)
            ci = cp.round().clip(0, MAP_SIZE - 1).astype(int)
            c[ci[0], ci[1]] += 1.0
            path.append(nx)
            cp = nx
            ci2 = cp.round().clip(0, MAP_SIZE - 1).astype(int)
            nb2 = np.unique(np.clip(cp + grid, 0, MAP_SIZE - 1).astype(int), axis=0)
            if not np.any(c[nb2[:, 0], nb2[:, 1]] < c[ci2[0], ci2[1]] + 0.01):
                break
        raw = np.array(path)
        win = min(len(raw), SAVGOL_WIN)
        order = min(SAVGOL_ORDER, win - 1)
        smooth = savgol_filter(raw, win, order, axis=0)

        idxs = smooth.round().clip(0, MAP_SIZE - 1).astype(int)
        prog = edt_target[idxs[:, 0], idxs[:, 1]]
        best = int(np.argmin(prog))
        seg = smooth[: max(2, best + 1)]
        iter_segments.append(seg)
        full_traj.extend(seg.tolist())
        cur = seg[-1].copy()
        ci = cur.round().clip(0, MAP_SIZE - 1).astype(int)
        if edt_target[ci[0], ci[1]] <= 2:
            break
    return np.array(full_traj), iter_segments


def overlay_heat(ax, heat, cmap, scene_bg=None, alpha=0.65, vmin=None, vmax=None):
    """Heatmap drawn on top of the warped topview (or a plain background
    when scene_bg is None)."""
    if scene_bg is not None:
        ax.imshow(scene_bg, origin='lower')
    else:
        ax.imshow(np.full((MAP_SIZE, MAP_SIZE, 3), 0.96, dtype=np.float32),
                  origin='lower')
    ax.imshow(heat, origin='lower', cmap=cmap, alpha=alpha,
              vmin=vmin, vmax=vmax, interpolation='bilinear')


def draw_robot_keyframe(ax, pos, heading, color, alpha, radius=2.4):
    ax.add_patch(Circle((pos[1], pos[0]), radius=radius, facecolor=color,
                         edgecolor='black', linewidth=0.7, alpha=alpha, zorder=8))
    if heading is not None:
        dx = np.cos(heading) * radius * 1.6
        dy = np.sin(heading) * radius * 1.6
        ax.add_patch(FancyArrowPatch(
            (pos[1], pos[0]), (pos[1] + dx, pos[0] + dy),
            arrowstyle='-|>', mutation_scale=10, color='black',
            alpha=alpha, lw=1.2, zorder=9,
        ))


def main(save_path):
    M, meta = load_real_maps()
    start_p = np.array(meta['start_pixel'], dtype=np.float32)
    goal_p = np.array(meta['goal_pixel'], dtype=np.float32)
    if meta.get('cat_pixel') is not None:
        cat_p = np.array(meta['cat_pixel'], dtype=np.float32)
    else:
        cat_p = None
    cat_visual_xy = M.get('cat_visual_cropped_xy')

    smooth_path = M['smoothed_path']
    obstacle_raw = (M['avoidance_pre'] > 0.5).astype(np.float32)
    target_raw = M['affordance']
    exec_traj, iter_segs = closed_loop_execution(
        start_p, target_raw, M['obstacle_v']
    )

    cropped = M['topview_cropped']
    cH, cW = cropped.shape[:2]
    dx, dy = M['crop_offset']
    H_p2c = M['H_planner_to_cropped']

    def planner_xy_in_cropped(row, col):
        return planner_to_topview_xy(row, col, H_p2c, 0, 0)

    # markers in cropped-topview pixel coords
    sx, sy = planner_xy_in_cropped(start_p[0], start_p[1])
    gx, gy = planner_xy_in_cropped(goal_p[0], goal_p[1])
    # cat: prefer visual orange (always lands on rendered mesh)
    if cat_visual_xy is not None:
        cx, cy = cat_visual_xy
    elif cat_p is not None:
        cx, cy = planner_xy_in_cropped(cat_p[0], cat_p[1])
    else:
        cx = cy = None

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.6),
                              gridspec_kw={'wspace': 0.06, 'hspace': 0.18})
    TITLES = ['(a) Scene', '(b) Affordance  $T$', '(c) Avoidance  $O$',
              '(d) Velocity  $V$', '(e) Cost  $C$', '(f) Trajectory']
    TFS = 17

    def style(ax, title):
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#888888')   # gray axes border
            spine.set_linewidth(1.0)
        ax.set_xlabel(title, fontsize=TFS, fontweight='bold', labelpad=8)

    def mark(ax):
        ax.scatter(sx, sy, s=110, c='royalblue', edgecolors='white',
                   linewidths=1.4, zorder=10)
        ax.scatter(gx, gy, s=140, marker='*', c='gold', edgecolors='black',
                   linewidths=0.8, zorder=10)
        if cx is not None:
            ax.scatter(cx, cy, s=80, c='#e85a4f', edgecolors='white',
                       linewidths=1.0, zorder=10, marker='X')

    def show_scene(ax):
        ax.imshow(cropped)
        ax.set_xlim(0, cW); ax.set_ylim(cH, 0)
        # 'equal' so every subfigure has identical pixel aspect / size;
        # the cropped image is the same shape across all panels.
        ax.set_aspect('equal')

    floor_mask = M['floor_mask']
    def overlay_planner_heat(ax, heat, cmap, alpha=0.7, vmin=None, vmax=None):
        warped = cv2.warpPerspective(
            heat.astype(np.float32), H_p2c.astype(np.float32), (cW, cH),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan,
        )
        warped = np.where(floor_mask, warped, np.nan)
        ax.imshow(warped, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax,
                   interpolation='bilinear')

    # (a) Scene — pure topview, no markers / overlays
    ax = axes[0, 0]; show_scene(ax); style(ax, TITLES[0])

    # (b) Affordance T
    ax = axes[0, 1]; show_scene(ax)
    overlay_planner_heat(ax, M['target_v'], cmap='Blues_r')
    mark(ax); style(ax, TITLES[1])

    # (c) Avoidance O
    ax = axes[0, 2]; show_scene(ax)
    overlay_planner_heat(ax, M['obstacle_v'], cmap='Reds')
    mark(ax); style(ax, TITLES[2])

    # (d) Velocity V
    ax = axes[1, 0]; show_scene(ax)
    overlay_planner_heat(ax, M['velocity'], cmap='YlOrRd_r',
                          vmin=0.0, vmax=1.2, alpha=0.55)
    mark(ax); style(ax, TITLES[3])

    # (e) Cost C + planner output
    ax = axes[1, 1]; show_scene(ax)
    overlay_planner_heat(ax, M['costmap'], cmap='viridis', alpha=0.55)
    if len(smooth_path) > 1:
        xs_e, ys_e = [], []
        for r, c in smooth_path:
            x, y = planner_xy_in_cropped(r, c)
            xs_e.append(x); ys_e.append(y)
        ax.plot(xs_e, ys_e, '-', color='red', lw=2.2, alpha=0.95, zorder=7)
        step = max(1, len(xs_e) // 12)
        ax.scatter(xs_e[::step], ys_e[::step], s=22, c='red',
                   edgecolors='white', linewidths=0.6, zorder=8)
    mark(ax); style(ax, TITLES[4])

    # (f) Trajectory: 5 real planner outputs from different seeds, on full scene
    ax = axes[1, 2]; show_scene(ax)
    import glob as _glob
    seed_glob = REAL.rstrip('/') + '_seed*'
    seed_dirs = sorted([p for p in _glob.glob(seed_glob)
                        if os.path.isfile(os.path.join(p, 'smoothed_path.npy'))])
    seed_paths = []
    for sd in seed_dirs:
        try:
            seed_paths.append(np.load(os.path.join(sd, 'smoothed_path.npy')))
        except Exception:
            pass
    # also include the primary REAL directory as a path
    seed_paths.append(smooth_path)
    # de-dup by length only (cheap heuristic)
    if len(seed_paths) > 1:
        unique = []
        for p in seed_paths:
            if not any(np.array_equal(p, u) for u in unique):
                unique.append(p)
        seed_paths = unique
    cmap_seeds = plt.cm.cool(np.linspace(0.05, 0.9, len(seed_paths)))
    # plot trajectory lines (faint) for context
    for i, p in enumerate(seed_paths):
        xs_t, ys_t = [], []
        for r, c in p:
            x, y = planner_xy_in_cropped(r, c)
            xs_t.append(x); ys_t.append(y)
        ax.plot(xs_t, ys_t, '-', color=cmap_seeds[i], lw=1.6, alpha=0.55, zorder=5)
    # 6 keyframes per trajectory, all overlaid with transparency so every
    # trajectory's robot ghosts coexist (multi-exposure view).
    K = 6
    kf_alpha = np.linspace(0.30, 0.85, K)        # time-progression alpha
    kf_color = plt.cm.plasma(np.linspace(0.05, 0.85, K))
    radius = 7
    for i, p in enumerate(seed_paths):
        n = len(p)
        if n < 2:
            continue
        kf_idx = list(np.linspace(0, n - 1, K).astype(int))
        for kf_i, ix in enumerate(kf_idx):
            r0, c0 = p[ix]
            x0_, y0_ = planner_xy_in_cropped(r0, c0)
            if ix < n - 1:
                r1, c1 = p[min(ix + 3, n - 1)]
                x1_, y1_ = planner_xy_in_cropped(r1, c1)
                d = np.array([x1_ - x0_, y1_ - y0_])
                heading = (np.arctan2(d[1], d[0])
                           if np.linalg.norm(d) > 1e-3 else 0.0)
            else:
                heading = None
            a = kf_alpha[kf_i] * (0.6 + 0.4 / max(1, len(seed_paths)))
            ax.add_patch(Circle((x0_, y0_), radius=radius,
                                 facecolor=kf_color[kf_i],
                                 edgecolor='black', linewidth=0.5,
                                 alpha=a, zorder=8))
            if heading is not None:
                dxa = np.cos(heading) * radius * 1.8
                dya = np.sin(heading) * radius * 1.8
                ax.add_patch(FancyArrowPatch(
                    (x0_, y0_), (x0_ + dxa, y0_ + dya),
                    arrowstyle='-|>', mutation_scale=8, color='black',
                    alpha=a * 0.9, lw=1.0, zorder=9,
                ))
    print(f'panel (f): {len(seed_paths)} trajectories × {K} keyframes overlaid')
    mark(ax)
    style(ax, TITLES[5])

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    pdf = os.path.splitext(save_path)[0] + '.pdf'
    plt.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {save_path}')
    print(f'saved: {pdf}')
    print(f'env={meta["env_name"]}  '
          f'real planner waypoints={len(smooth_path)} '
          f'(raw={meta["n_raw_waypoints"]})  '
          f'closed-loop sim: {len(exec_traj)} pts, {len(iter_segs)} plan iters')

    # Console legend: explain the markers shown on every panel
    print()
    print('=' * 72)
    print('Markers overlaid on every panel (a)-(f):')
    print('=' * 72)
    sw = meta.get('robot_pos_world', [None, None])
    gw = meta.get('target_pos_world', [None, None])
    cw = meta.get('cat_world', [None, None])
    print(f'  BLUE CIRCLE  (start) = robot initial base pose')
    print(f'                 planner pixel (row, col) = ({start_p[0]:.0f}, {start_p[1]:.0f})')
    print(f'                 world (x, y) = ({sw[0]:.3f}, {sw[1]:.3f}) m')
    print(f'  GOLD STAR    (goal)  = target fixture for RouteA: CoffeeMachine')
    print(f'                 planner pixel (row, col) = ({goal_p[0]:.0f}, {goal_p[1]:.0f})')
    print(f'                 world (x, y) = ({gw[0]:.3f}, {gw[1]:.3f}) m')
    print(f'  RED  X       (cat)   = animate obstacle (mujoco body "obstacle_1_main")')
    if cat_p is not None:
        print(f'                 planner pixel (row, col) = ({cat_p[0]:.0f}, {cat_p[1]:.0f})')
    if cw[0] is not None:
        print(f'                 world (x, y) = ({cw[0]:.3f}, {cw[1]:.3f}) m')
    print(f'  RED  LINE    (panel e) = real planner output, {len(smooth_path)} waypoints')
    print(f'  CYAN/PLASMA  (panel f) = closed-loop replan segments + robot ghosts (t=0..5)')
    print('=' * 72)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else \
        str(repo_root() / "tmp/navigation_costmap_pipeline.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    main(out)
