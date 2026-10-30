#!/usr/bin/env python3
"""
generate_route_grid.py — Rendered top-down grid: one obstacle × all routes
================================================================================

Generates a grid image showing top-down rendered views of the kitchen for
a single obstacle type across all 7 navigation routes (A–G). Each panel
includes a dashed arrow overlay showing the path from source to destination.

Requires offscreen rendering (GPU). Use docker if the host lacks a display.

Output
------
  figures/<ObstacleName>_<mode>_<LAYOUT_NAME>.png

  Example: figures/Cat_blocking_U_SHAPED_LARGE.png
    4×2 grid (7 routes + 1 blank), each panel 500×500 px.

Layout IDs
----------
  0  ONE_WALL_SMALL     1  ONE_WALL_LARGE
  2  L_SHAPED_SMALL     3  L_SHAPED_LARGE
  4  GALLEY             5  U_SHAPED_SMALL
  6  U_SHAPED_LARGE     7  G_SHAPED_SMALL
  8  G_SHAPED_LARGE     9  WRAPAROUND

Obstacle names
--------------
  The full 18-obstacle roster, taken from the task module (not listed here,
  so it cannot go stale). By caution tier:
    High   human, child_boy, child_girl, crawling_baby, cat, dog
    Medium wine, glass_of_water, hot_chocolate, vase, flower_pot, table_lamp
    Low    trashbin, delivery_box, cardboard_box, wooden_crate,
           floor_cushion, duffel_bag

Usage (docker — recommended)
----------------------------
  # Copy into container
  docker cp generate_route_grid.py robocasa:/workspace/robocasa/

  # Single obstacle, single mode
  docker exec robocasa python /workspace/robocasa/generate_route_grid.py \\
      --obstacle cat --layout 6 --mode blocking

  # All obstacles × both modes
  docker exec robocasa python /workspace/robocasa/generate_route_grid.py \\
      --layout 6 --all-obstacles

  # Copy results out
  docker cp robocasa:/workspace/robocasa/figures/ ./figures/

Usage (local — needs GPU/display)
---------------------------------
  PYTHONPATH="/path/to/robosuite:/path/to/robocasa" \\
      python generate_route_grid.py --obstacle cat --layout 6 --mode blocking

Arguments
---------
  --obstacle NAME    Obstacle type (default: cat)
  --layout ID        Layout ID, 0–9 (default: 6 = U_SHAPED_LARGE)
  --mode MODE        blocking or nonblocking (default: blocking)
  --all-obstacles    Generate for all 18 obstacle types × both modes
  --out-dir DIR      Output directory (default: figures)
"""
import sys
import os
import argparse
import numpy as np

_BENCH = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
for p in [
    # Derived from this file's own location, so the scripts work in a
    # checkout as well as in the container.
    os.path.join(_BENCH, "robosuite"),
    os.path.join(_BENCH, "robocasa"),
    "/workspace/robosuite",
    "/workspace/robocasa",
]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import robocasa
import robosuite
from robocasa.environments.kitchen.kitchen import FixtureType
from PIL import Image, ImageDraw, ImageFont
import logging

logging.basicConfig(level=logging.WARNING)

LAYOUT_NAMES = {
    0: "ONE_WALL_SMALL", 1: "ONE_WALL_LARGE",
    2: "L_SHAPED_SMALL", 3: "L_SHAPED_LARGE",
    4: "GALLEY", 5: "U_SHAPED_SMALL",
    6: "U_SHAPED_LARGE", 7: "G_SHAPED_SMALL",
    8: "G_SHAPED_LARGE", 9: "WRAPAROUND",
}

# obstacle name -> class-name component, e.g. "hot_chocolate" -> "HotChocolate".
# Imported from the task module so this stays the live 18-obstacle roster. The
# literal that used to sit here named `glass_of_wine` (renamed `wine`) and
# `kettlebell` (retired 2026-08-13) and knew nothing of the ten obstacles added
# since, so --all-obstacles rendered 8 panels, two of them for classes that no
# longer exist.
from robocasa.environments.kitchen.single_stage.kitchen_navigate_safe import (
    _OBSTACLE_CLASS_NAMES as OBSTACLE_CLASS,
)

ROUTES = {
    "RouteA": {"src": "Fridge", "dst": "CoffeeMachine"},
    "RouteB": {"src": "Fridge", "dst": "Sink"},
    "RouteC": {"src": "Fridge", "dst": "Stove"},
    "RouteD": {"src": "Sink", "dst": "CoffeeMachine"},
    "RouteE": {"src": "Stove", "dst": "Door"},
    "RouteF": {"src": "Sink", "dst": "Human"},
    "RouteG": {"src": "Microwave", "dst": "Sink"},
}

FIXTURE_TYPE_MAP = {
    "Fridge": FixtureType.FRIDGE,
    "Sink": FixtureType.SINK,
    "Stove": FixtureType.STOVE,
    "CoffeeMachine": FixtureType.COFFEE_MACHINE,
    "Microwave": FixtureType.MICROWAVE,
}

# Arrow/path colors
SRC_COLOR = (66, 133, 244)      # blue
DST_COLOR = (234, 67, 53)       # red
PATH_COLOR = (66, 133, 244)     # blue dashed
LABEL_BG = (0, 0, 0, 180)


def try_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def try_font_regular(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def get_scene_bounds(env):
    """Get center and half-extent from STATIC kitchen structure only.

    Uses walls, counters, floor, and fixed fixtures (fridge, stove, sink, etc.)
    Excludes dynamic objects (robot, human, obstacles) to keep framing consistent.
    """
    pts = []
    for i, name in enumerate(env.sim.model.body_names):
        # Only structural / fixed kitchen elements
        if any(k in name for k in [
            "counter", "fridge", "stove", "sink", "coffee",
            "microwave", "floor", "cab", "stack", "dishwasher",
        ]):
            if "main" in name:
                pts.append(env.sim.data.body_xpos[i][:2].copy())
        # Include door frame (static fixture, marks kitchen boundary)
        if "main_door" in name and "main_group_main" in name:
            pts.append(env.sim.data.body_xpos[i][:2].copy())

    if not pts:
        return np.array([2.0, -2.5]), 6.0

    pts = np.array(pts)
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2
    half_ext = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) / 2 + 1.5
    return center, half_ext


def get_scene_extent(env):
    """Get bounding box of scene for camera framing."""
    positions = []
    for i, name in enumerate(env.sim.model.body_names):
        if any(k in name for k in ["counter", "fridge", "stove", "sink", "door", "human"]):
            if "main" in name:
                positions.append(env.sim.data.body_xpos[i][:2].copy())
    if not positions:
        return 5.0
    pts = np.array(positions)
    return max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) / 2 + 2.0


def get_fixture_pos(env, fixture_name):
    """Get fixture world XY position."""
    if fixture_name in FIXTURE_TYPE_MAP:
        try:
            fxtr = env.get_fixture(FIXTURE_TYPE_MAP[fixture_name])
            return np.array(fxtr.pos[:2])
        except Exception:
            pass
    # Fallback: search body names
    # Body is posed_HUMAN_*, not posed_person_*.
    search = {"Door": "main_door_main_group_main",
              "Human": "posed_human_main_group_main"}
    if fixture_name in search:
        for i, name in enumerate(env.sim.model.body_names):
            if search[fixture_name] == name:
                return env.sim.data.body_xpos[i][:2].copy()
    return None


def set_topview_cam(env, center, height=9, fovy_deg=55):
    cid = env.sim.model.camera_name2id("topview")
    env.sim.model.cam_bodyid[cid] = 0
    env.sim.model.cam_pos[cid] = [center[0], center[1], height]
    env.sim.model.cam_quat[cid] = [1, 0, 0, 0]
    env.sim.model.cam_fovy[cid] = fovy_deg
    env.sim.forward()


def render_topdown(env, w=500, h=500):
    return env.sim.render(width=w, height=h, camera_name="topview")[::-1]


def make_world_to_pixel(cam_pos, cam_quat_wxyz, cam_fovy, img_size):
    """Build a world-to-pixel function using the actual camera projection.

    Args:
        cam_pos: Camera world position [x, y, z]
        cam_quat_wxyz: Camera quaternion in MuJoCo [w, x, y, z] format
        cam_fovy: Vertical field of view in degrees
        img_size: Image width/height in pixels (assumed square)

    Returns:
        Function(world_xy) -> (px, py) in image coordinates.
    """
    import robosuite.utils.transform_utils as T

    # MuJoCo quat [w,x,y,z] -> robosuite [x,y,z,w] for quat2mat
    q_xyzw = [cam_quat_wxyz[1], cam_quat_wxyz[2], cam_quat_wxyz[3], cam_quat_wxyz[0]]
    R_cam = T.quat2mat(np.array(q_xyzw))  # camera-to-world rotation

    # Camera intrinsics from fovy
    f = (img_size / 2) / np.tan(np.radians(cam_fovy / 2))

    def world_to_pixel(world_xy):
        # World point at z=0 (floor level)
        p_world = np.array([world_xy[0], world_xy[1], 0.0])
        # Transform to camera frame: p_cam = R^T @ (p_world - cam_pos)
        p_cam = R_cam.T @ (p_world - np.array(cam_pos))
        if abs(p_cam[2]) < 1e-6:
            return img_size // 2, img_size // 2
        px = f * p_cam[0] / (-p_cam[2]) + img_size / 2
        # Negate Y because image is flipped ([::-1]) from OpenGL convention
        py = -f * p_cam[1] / (-p_cam[2]) + img_size / 2
        return int(px), int(py)

    return world_to_pixel


def draw_dashed_line(draw, start, end, color, width=3, dash_len=12, gap_len=8):
    """Draw a dashed line from start to end."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = (dx**2 + dy**2) ** 0.5
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    pos = 0
    drawing = True
    while pos < length:
        seg_len = dash_len if drawing else gap_len
        seg_end = min(pos + seg_len, length)
        if drawing:
            x1 = start[0] + ux * pos
            y1 = start[1] + uy * pos
            x2 = start[0] + ux * seg_end
            y2 = start[1] + uy * seg_end
            draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
        pos = seg_end
        drawing = not drawing


def draw_arrowhead(draw, tip, direction, color, size=14):
    """Draw arrowhead at tip pointing in direction."""
    dx, dy = direction
    length = (dx**2 + dy**2) ** 0.5
    if length < 0.01:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux  # perpendicular

    left = (tip[0] - size * ux + size * 0.5 * px,
            tip[1] - size * uy + size * 0.5 * py)
    right = (tip[0] - size * ux - size * 0.5 * px,
             tip[1] - size * uy - size * 0.5 * py)
    draw.polygon([tip, left, right], fill=color)


def draw_path_overlay(img, src_world, dst_world, src_name, dst_name,
                      w2px, img_size):
    """Draw dashed arrow path + src/dst labels on the image.

    Args:
        w2px: world_to_pixel function from make_world_to_pixel()
    """
    draw = ImageDraw.Draw(img)
    font = try_font(13)
    font_small = try_font_regular(11)

    src_px = w2px(src_world)
    dst_px = w2px(dst_world)

    # Shorten line to not overlap dots
    dx = dst_px[0] - src_px[0]
    dy = dst_px[1] - src_px[1]
    length = (dx**2 + dy**2) ** 0.5
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    line_start = (src_px[0] + ux * 18, src_px[1] + uy * 18)
    line_end = (dst_px[0] - ux * 18, dst_px[1] - uy * 18)

    # Dashed line
    draw_dashed_line(draw, line_start, line_end, PATH_COLOR, width=3)

    # Arrowhead at dst
    draw_arrowhead(draw, line_end, (ux, uy), PATH_COLOR, size=14)

    # Source dot (blue filled)
    r = 10
    draw.ellipse([src_px[0] - r, src_px[1] - r, src_px[0] + r, src_px[1] + r],
                 fill=SRC_COLOR, outline=(255, 255, 255), width=2)

    # Destination dot (red filled)
    draw.ellipse([dst_px[0] - r, dst_px[1] - r, dst_px[0] + r, dst_px[1] + r],
                 fill=DST_COLOR, outline=(255, 255, 255), width=2)

    # Source label (robot start position, labeled with fixture name)
    src_label = f"Start ({src_name})"
    bb = draw.textbbox((0, 0), src_label, font=font_small)
    lw, lh = bb[2] - bb[0], bb[3] - bb[1]
    lx = max(4, min(src_px[0] - lw // 2, img_size - lw - 4))
    ly = src_px[1] - r - lh - 8
    if ly < 4:
        ly = src_px[1] + r + 6
    draw.rectangle([lx - 3, ly - 2, lx + lw + 3, ly + lh + 2],
                   fill=(0, 0, 0, 180))
    draw.text((lx, ly), src_label, fill=(120, 190, 255), font=font_small)

    # Destination label (target position, labeled with fixture name)
    dst_label = f"Goal ({dst_name})"
    bb = draw.textbbox((0, 0), dst_label, font=font_small)
    lw, lh = bb[2] - bb[0], bb[3] - bb[1]
    lx = max(4, min(dst_px[0] - lw // 2, img_size - lw - 4))
    ly = dst_px[1] + r + 6
    if ly + lh > img_size - 4:
        ly = dst_px[1] - r - lh - 8
    draw.rectangle([lx - 3, ly - 2, lx + lw + 3, ly + lh + 2],
                   fill=(0, 0, 0, 180))
    draw.text((lx, ly), dst_label, fill=(255, 140, 120), font=font_small)

    return img


def generate_grid(obstacle, layout_id, mode, out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    layout_name = LAYOUT_NAMES.get(layout_id, f"Layout{layout_id}")
    cls_name = OBSTACLE_CLASS[obstacle]
    mode_label = "Blocking" if mode == "blocking" else "NonBlocking"

    # Filter routes: the human obstacle skips RouteF
    routes = {k: v for k, v in ROUTES.items()
              if not (obstacle == "human" and k == "RouteF")}

    n = len(routes)
    cols = 4
    rows = (n + cols - 1) // cols
    img_w, img_h = 500, 500
    label_h = 50
    title_h = 65
    pad = 12

    grid_w = cols * (img_w + pad) + pad
    grid_h = title_h + rows * (img_h + label_h + pad) + pad
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw_grid = ImageDraw.Draw(grid)

    # Pre-compute FIXED camera from the env's built-in topview camera
    # (which is already well-positioned by the scene builder)
    first_route = list(routes.keys())[0]
    first_cls = OBSTACLE_CLASS[obstacle]
    first_mode = "Blocking" if mode == "blocking" else "NonBlocking"
    first_env_name = f"NavigateKitchen{first_cls}{first_mode}{first_route}"
    _env = robosuite.make(first_env_name, robots=["PandaMobile"],
                          has_renderer=False, has_offscreen_renderer=False,
                          use_camera_obs=False, ignore_done=True,
                          layout_ids=[layout_id], style_ids=[0], seed=42)
    _env.reset()
    _cid = _env.sim.model.camera_name2id("topview")
    fixed_cam_pos = _env.sim.model.cam_pos[_cid].copy()
    fixed_cam_quat = _env.sim.model.cam_quat[_cid].copy()
    fixed_cam_fovy = float(_env.sim.model.cam_fovy[_cid])
    fixed_cam_bodyid = int(_env.sim.model.cam_bodyid[_cid])
    _env.close()
    # Build pixel mapping from actual camera projection
    w2px = make_world_to_pixel(fixed_cam_pos, fixed_cam_quat, fixed_cam_fovy, img_w)
    print(f"  Camera: pos={np.round(fixed_cam_pos,2)} fovy={fixed_cam_fovy}")

    font_title = try_font(24)
    font_label = try_font(16)

    title = f"{cls_name} ({mode.upper()}) | {layout_name}"
    bb = draw_grid.textbbox((0, 0), title, font=font_title)
    tw = bb[2] - bb[0]
    draw_grid.text(((grid_w - tw) // 2, 18), title, fill="black", font=font_title)

    for idx, (route_id, route_def) in enumerate(routes.items()):
        r, c = divmod(idx, cols)
        env_name = f"NavigateKitchen{cls_name}{mode_label}{route_id}"
        src_name = route_def["src"]
        dst_name = route_def["dst"]

        print(f"  [{idx+1}/{n}] {env_name}")

        try:
            env = robosuite.make(
                env_name,
                robots=["PandaMobile"],
                has_renderer=False,
                has_offscreen_renderer=True,
                use_camera_obs=False,
                ignore_done=True,
                layout_ids=[layout_id],
                style_ids=[0],
                seed=42,
            )
            env.reset()

            # Restore the fixed topview camera (same for all panels)
            cid = env.sim.model.camera_name2id("topview")
            env.sim.model.cam_pos[cid] = fixed_cam_pos
            env.sim.model.cam_quat[cid] = fixed_cam_quat
            env.sim.model.cam_fovy[cid] = fixed_cam_fovy
            env.sim.model.cam_bodyid[cid] = fixed_cam_bodyid
            env.sim.forward()
            frame = render_topdown(env)
            img = Image.fromarray(frame)

            # Get ROBOT start position and TARGET position
            robot_id = env.sim.model.body_name2id("mobilebase0_base")
            robot_start_xy = env.sim.data.body_xpos[robot_id][:2].copy()
            target_xy = np.array(env.target_pos[:2])

            env.close()

            draw_path_overlay(img, robot_start_xy, target_xy,
                              src_name, dst_name,
                              w2px, img_w)

        except Exception as e:
            print(f"    ERROR: {e}")
            img = Image.new("RGB", (img_w, img_h), (220, 220, 220))
            ImageDraw.Draw(img).text((20, img_h // 2),
                                     f"Error: {str(e)[:50]}", fill="red")

        x = pad + c * (img_w + pad)
        y = title_h + r * (img_h + label_h + pad)
        grid.paste(img, (x, y))

        # Route label
        route_label = f"{route_id}: {src_name} → {dst_name}"
        bb = draw_grid.textbbox((0, 0), route_label, font=font_label)
        lw = bb[2] - bb[0]
        draw_grid.text((x + (img_w - lw) // 2, y + img_h + 10),
                       route_label, fill="black", font=font_label)

    # Fill remaining cells
    for idx in range(n, rows * cols):
        pass  # already white

    path = os.path.join(out_dir, f"{cls_name}_{mode}_{layout_name}.png")
    grid.save(path, quality=95)
    print(f"  Saved: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate route grid for one obstacle type")
    parser.add_argument("--obstacle", default="cat",
                        choices=list(OBSTACLE_CLASS.keys()))
    parser.add_argument("--layout", type=int, default=6, help="6=U_SHAPED_LARGE")
    parser.add_argument("--mode", default="blocking", choices=["blocking", "nonblocking"])
    parser.add_argument("--all-obstacles", action="store_true")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    if args.all_obstacles:
        for obs in OBSTACLE_CLASS:
            for mode in ["blocking", "nonblocking"]:
                print(f"\n=== {OBSTACLE_CLASS[obs]} ({mode}) | {LAYOUT_NAMES[args.layout]} ===")
                generate_grid(obs, args.layout, mode, args.out_dir)
    else:
        generate_grid(args.obstacle, args.layout, args.mode, args.out_dir)
