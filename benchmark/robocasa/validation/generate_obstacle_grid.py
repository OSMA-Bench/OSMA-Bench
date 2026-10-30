#!/usr/bin/env python3
"""
Generate grid image showing all obstacle types for a given route/layout/mode.
Renders top-down view for each obstacle variant.

Can run inside docker or locally.

Usage:
    # Inside docker
    docker exec robocasa python /workspace/robocasa/generate_obstacle_grid.py \
        --route RouteB --layout 6 --mode blocking

    # Locally
    PYTHONPATH=".../robosuite:.../robocasa" python generate_obstacle_grid.py \
        --route RouteB --layout 6 --mode blocking

    # All routes for a layout
    python generate_obstacle_grid.py --layout 6 --all-routes

One panel per obstacle: 18 for most routes, 17 for RouteF, where the
posed_human is the destination and so cannot also be the obstacle.
"""
import sys
import os
import argparse
import numpy as np

# Try to add paths if not in docker
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

# (obstacle name, class-name component) for the full 18-obstacle roster, in
# caution-tier order (High, then Medium, then Low) so the grid reads top-left
# to bottom-right as decreasing caution. Names come from the task module: the
# literal that used to sit here still listed `glass_of_wine` (renamed `wine`)
# and `kettlebell` (retired 2026-08-13), and was missing the ten obstacles
# added since, so the grid silently showed 8 of 18.
from robocasa.environments.kitchen.single_stage.kitchen_navigate_safe import (
    _OBSTACLE_CLASS_NAMES,
    HIGH_TIER_OBSTACLES,
    MODERATE_TIER_OBSTACLES,
    LOW_TIER_OBSTACLES,
)

OBSTACLES = [
    (obs, _OBSTACLE_CLASS_NAMES[obs])
    for tier in (HIGH_TIER_OBSTACLES, MODERATE_TIER_OBSTACLES, LOW_TIER_OBSTACLES)
    for obs in tier
]

ROUTES = ["RouteA", "RouteB", "RouteC", "RouteD", "RouteE", "RouteF", "RouteG"]


def make_world_to_pixel(cam_pos, cam_quat_wxyz, cam_fovy, img_size):
    """Build world-to-pixel function using actual camera projection."""
    import robosuite.utils.transform_utils as T
    q_xyzw = [cam_quat_wxyz[1], cam_quat_wxyz[2], cam_quat_wxyz[3], cam_quat_wxyz[0]]
    R_cam = T.quat2mat(np.array(q_xyzw))
    f = (img_size / 2) / np.tan(np.radians(cam_fovy / 2))

    def w2px(world_xy):
        p_world = np.array([world_xy[0], world_xy[1], 0.0])
        p_cam = R_cam.T @ (p_world - np.array(cam_pos))
        if abs(p_cam[2]) < 1e-6:
            return img_size // 2, img_size // 2
        px = f * p_cam[0] / (-p_cam[2]) + img_size / 2
        py = -f * p_cam[1] / (-p_cam[2]) + img_size / 2
        return int(px), int(py)

    return w2px


def draw_path(img, src_xy, dst_xy, w2px):
    """Draw dashed arrow from robot start to target on the image."""
    draw = ImageDraw.Draw(img)
    src_px = w2px(src_xy)
    dst_px = w2px(dst_xy)

    dx = dst_px[0] - src_px[0]
    dy = dst_px[1] - src_px[1]
    length = (dx**2 + dy**2) ** 0.5
    if length < 1:
        return
    ux, uy = dx / length, dy / length

    # Shorten to not overlap dots
    s = (src_px[0] + ux * 14, src_px[1] + uy * 14)
    e = (dst_px[0] - ux * 14, dst_px[1] - uy * 14)

    # Dashed line
    pos = 0
    drawing = True
    while pos < length - 28:
        seg = min(10, length - 28 - pos)
        if drawing:
            x1 = s[0] + ux * pos
            y1 = s[1] + uy * pos
            x2 = s[0] + ux * (pos + seg)
            y2 = s[1] + uy * (pos + seg)
            draw.line([(x1, y1), (x2, y2)], fill=(66, 133, 244), width=3)
        pos += seg if drawing else 7
        drawing = not drawing

    # Arrowhead
    tip = e
    px, py = -uy, ux
    left = (tip[0] - 12*ux + 6*px, tip[1] - 12*uy + 6*py)
    right = (tip[0] - 12*ux - 6*px, tip[1] - 12*uy - 6*py)
    draw.polygon([tip, left, right], fill=(66, 133, 244))

    # Source dot (blue)
    r = 8
    draw.ellipse([src_px[0]-r, src_px[1]-r, src_px[0]+r, src_px[1]+r],
                 fill=(66, 133, 244), outline=(255, 255, 255), width=2)
    # Destination dot (red)
    draw.ellipse([dst_px[0]-r, dst_px[1]-r, dst_px[0]+r, dst_px[1]+r],
                 fill=(234, 67, 53), outline=(255, 255, 255), width=2)


def get_scene_bounds(env):
    """Get center and half-extent that covers the entire kitchen scene."""
    pts = []
    for i, name in enumerate(env.sim.model.body_names):
        if any(k in name for k in [
            "counter", "fridge", "stove", "sink", "coffee",
            "microwave", "door", "human", "floor", "wall",
            "mobilebase", "standing_table",
        ]):
            if "main" in name or "base" in name:
                pts.append(env.sim.data.body_xpos[i][:2].copy())
    try:
        rid = env.sim.model.body_name2id("mobilebase0_base")
        pts.append(env.sim.data.body_xpos[rid][:2].copy())
    except Exception:
        pass
    if hasattr(env, 'target_pos'):
        pts.append(np.array(env.target_pos[:2]))
    if not pts:
        return np.array([2.0, -2.5]), 6.0
    pts = np.array(pts)
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2
    half_ext = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) / 2 + 1.0
    return center, half_ext


def set_topview_cam(env, center, height=9, fovy_deg=55):
    cid = env.sim.model.camera_name2id("topview")
    env.sim.model.cam_bodyid[cid] = 0
    env.sim.model.cam_pos[cid] = [center[0], center[1], height]
    env.sim.model.cam_quat[cid] = [1, 0, 0, 0]
    env.sim.model.cam_fovy[cid] = fovy_deg
    env.sim.forward()


def render_topdown(env, w=500, h=500):
    return env.sim.render(width=w, height=h, camera_name="topview")[::-1]


def try_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def generate_grid(route, layout_id, mode, out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    layout_name = LAYOUT_NAMES.get(layout_id, f"Layout{layout_id}")
    mode_label = "Blocking" if mode == "blocking" else "NonBlocking"

    obstacles = [(obs, cls) for obs, cls in OBSTACLES
                 if not (obs == "human" and route == "RouteF")]

    cols = 4
    rows = (len(obstacles) + cols - 1) // cols
    img_w, img_h = 500, 500
    label_h = 45
    title_h = 60
    pad = 10

    grid_w = cols * (img_w + pad) + pad
    grid_h = title_h + rows * (img_h + label_h + pad) + pad
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    font_title = try_font(26)
    font_label = try_font(18)

    title = f"{mode.upper()} | {route} | {layout_name}"
    bb = draw.textbbox((0, 0), title, font=font_title)
    tw = bb[2] - bb[0]
    draw.text(((grid_w - tw) // 2, 16), title, fill="black", font=font_title)

    for idx, (obs_name, cls_name) in enumerate(obstacles):
        r, c = divmod(idx, cols)
        env_name = f"NavigateKitchen{cls_name}{mode_label}{route}"

        print(f"  [{idx+1}/{len(obstacles)}] {env_name}")

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

            # Use the env's built-in topview camera (well-positioned by scene builder)
            cid = env.sim.model.camera_name2id("topview")
            cam_pos = env.sim.model.cam_pos[cid].copy()
            cam_quat = env.sim.model.cam_quat[cid].copy()
            cam_fovy = float(env.sim.model.cam_fovy[cid])
            frame = render_topdown(env)

            # Get robot start and target positions for path overlay
            robot_id = env.sim.model.body_name2id("mobilebase0_base")
            robot_xy = env.sim.data.body_xpos[robot_id][:2].copy()
            target_xy = np.array(env.target_pos[:2])
            env.close()

            img = Image.fromarray(frame)

            # Draw path with proper camera projection
            w2px = make_world_to_pixel(cam_pos, cam_quat, cam_fovy, img_w)
            draw_path(img, robot_xy, target_xy, w2px)
        except Exception as e:
            print(f"    ERROR: {e}")
            img = Image.new("RGB", (img_w, img_h), (220, 220, 220))
            ImageDraw.Draw(img).text((20, img_h // 2), f"Error: {str(e)[:50]}", fill="red")

        x = pad + c * (img_w + pad)
        y = title_h + r * (img_h + label_h + pad)
        grid.paste(img, (x, y))

        bb = draw.textbbox((0, 0), cls_name, font=font_label)
        lw = bb[2] - bb[0]
        draw.text((x + (img_w - lw) // 2, y + img_h + 8), cls_name,
                  fill="black", font=font_label)

    path = os.path.join(out_dir, f"{mode}_{route}_{layout_name}.png")
    grid.save(path, quality=95)
    print(f"  Saved: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate obstacle grid images")
    parser.add_argument("--route", default="RouteB")
    parser.add_argument("--layout", type=int, default=6, help="6=U_SHAPED_LARGE")
    parser.add_argument("--mode", default="blocking", choices=["blocking", "nonblocking"])
    parser.add_argument("--all-routes", action="store_true")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    if args.all_routes:
        for route in ROUTES:
            for mode in ["blocking", "nonblocking"]:
                print(f"\n=== {mode.upper()} | {route} | {LAYOUT_NAMES[args.layout]} ===")
                generate_grid(route, args.layout, mode, args.out_dir)
    else:
        generate_grid(args.route, args.layout, args.mode, args.out_dir)
