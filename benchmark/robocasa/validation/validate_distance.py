"""Sanity-check the patched boundary intrusion check.

Build one navigate_safe env, then move the obstacle to a grid of XY offsets
relative to the robot and rotate the robot through N/E/S/W. For each
configuration, render the topview, run ``_check_obstacle_boundary_intrusion``,
and stamp the result onto the frame. Finally compose the per-yaw frames into
a single grid image so the alignment between numbers and rendered geometry is
obvious at a glance.

Output directory layout:
    distance_validation/
        run.log
        results.csv             # one row per (yaw, dx, dy) cell
        cells/<yaw>_dx<dx>_dy<dy>.png
        grid_<yaw_label>.png    # one per yaw direction
"""

import argparse
import csv
import logging
import os
import sys

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

import robosuite
from robosuite.controllers import load_composite_controller_config
from robocasa.models.scenes.scene_registry import LayoutType, StyleType
from robocasa.environments.kitchen.single_stage.kitchen_navigate_safe import (
    OBSTACLE_KEEPOUT_RADIUS_M,
)


# (yaw_world_radians, label) -- robot base orientation in world frame.
# 0 rad = +X (east); pi/2 = +Y (north); pi = -X (west); -pi/2 = -Y (south).
YAW_DIRS = [
    (0.0,        "E"),
    (np.pi/2,    "N"),
    (np.pi,      "W"),
    (-np.pi/2,   "S"),
]

# Obstacle (dx, dy) relative to the robot base XY in world frame.
# Designed to cross the keep-out radius (1.0 m, every obstacle) at one cell.
OBSTACLE_OFFSETS = [
    (0.40, 0.0),    # very close, east
    (0.70, 0.0),    # near boundary, east
    (1.00, 0.0),    # comfortably clear, east
    (0.0, 0.70),    # boundary, north
    (-0.70, 0.0),   # boundary, west
    (0.0, -0.70),   # boundary, south
    (0.55, 0.55),   # diagonal NE
]


def find_joint_addr(model, name):
    j = model.joint_name2id(name)
    return int(model.jnt_qposadr[j])


def set_robot_yaw_world(env, yaw_world):
    """Set the mobile base yaw joint so the robot's *world* yaw matches.

    The mobile_yaw joint is offset from the world frame by the robot's parent
    frame, so we infer the offset on first call.
    """
    addr = find_joint_addr(env.sim.model, "mobilebase0_joint_mobile_yaw")
    if not hasattr(env, "_yaw_offset"):
        # Capture current world yaw vs current joint qpos to compute offset.
        body_id = env.sim.model.body_name2id("mobilebase0_base")
        R = np.array(env.sim.data.body_xmat[body_id]).reshape(3, 3)
        current_world_yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        current_joint = float(env.sim.data.qpos[addr])
        env._yaw_offset = current_world_yaw - current_joint
    env.sim.data.qpos[addr] = yaw_world - env._yaw_offset


def set_obstacle_xy_world(env, obj_name, x, y, z=None):
    obj = env.objects[obj_name]
    joint_name = obj.joints[0]
    qpos = env.sim.data.get_joint_qpos(joint_name).copy()
    qpos[0] = x
    qpos[1] = y
    if z is not None:
        qpos[2] = z
    env.sim.data.set_joint_qpos(joint_name, qpos)


def project(pt_world, cam_xpos, cam_xmat, fovy_deg, height, width):
    """World point -> pixel (u, v) in the saved (vertically-flipped) frame."""
    p_rel = np.asarray(pt_world, dtype=float) - cam_xpos
    p_cam = cam_xmat.T @ p_rel
    z_forward = -p_cam[2]
    if z_forward <= 1e-6:
        return None
    fovy_rad = np.deg2rad(fovy_deg)
    f = (height / 2.0) / np.tan(fovy_rad / 2.0)
    u = width / 2.0 + f * (p_cam[0] / z_forward)
    v_raw = height / 2.0 - f * (p_cam[1] / z_forward)
    v = (height - 1.0 - v_raw)
    return float(u), float(v), float(z_forward), float(f)


def collect_geom_footprints(env, obj_name, exclude=None):
    """Return [(x, y, r)] approximating each collision geom as a disk on z=0.

    Iterates the same geom set the boundary check uses (collision geoms,
    minus any in `exclude`). For each geom we use geom_xpos as XY center and
    max(geom_size) as a conservative radius -- accurate for spheres and a
    safe upper bound for boxes/capsules/meshes (sufficient for visualisation).
    """
    exclude = exclude or set()
    geoms = env._filter_collision_geoms(env._get_geom_ids_by_name(obj_name))
    out = []
    for g in geoms:
        name = env.sim.model.geom_id2name(g) or ""
        if name in exclude:
            continue
        xpos = env.sim.data.geom_xpos[g]
        r = float(np.max(env.sim.model.geom_size[g]))
        out.append((float(xpos[0]), float(xpos[1]), r))
    return out


def make_schematic(robot_xy, robot_yaw, obstacle_xy, threshold_m,
                   distance_m, violated,
                   robot_footprints, obstacle_footprints,
                   size_px=384, half_extent_m=1.5):
    """Top-down schematic of the geometry the boundary check actually uses.

    Each remaining robot collision geom is drawn as a green disk; each
    obstacle collision geom is drawn as a red disk. Threshold ring around
    the obstacle XY at exactly the keep-out radius. World +X = image right,
    +Y = up.
    """
    canvas = Image.new("RGB", (size_px, size_px), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    px_per_m = (size_px / 2) / half_extent_m
    cx, cy = size_px / 2, size_px / 2

    def w2p(wx, wy):
        dx = wx - robot_xy[0]
        dy = wy - robot_xy[1]
        return (cx + dx * px_per_m, cy - dy * px_per_m)  # +Y up

    # gridlines every 0.5 m
    for k in np.arange(-half_extent_m, half_extent_m + 1e-6, 0.5):
        x = cx + k * px_per_m
        y = cy - k * px_per_m
        draw.line((x, 0, x, size_px), fill=(220, 220, 220))
        draw.line((0, y, size_px, y), fill=(220, 220, 220))

    # robot collision-geom footprints (translucent green disks)
    for (gx, gy, gr) in robot_footprints:
        rpx = max(2.0, gr * px_per_m)
        cx_p, cy_p = w2p(gx, gy)
        draw.ellipse(
            (cx_p - rpx, cy_p - rpx, cx_p + rpx, cy_p + rpx),
            outline=(0, 100, 0), fill=(190, 230, 190),
        )

    # obstacle collision-geom footprints (red disks)
    for (gx, gy, gr) in obstacle_footprints:
        rpx = max(2.0, gr * px_per_m)
        cx_p, cy_p = w2p(gx, gy)
        draw.ellipse(
            (cx_p - rpx, cy_p - rpx, cx_p + rpx, cy_p + rpx),
            outline=(150, 0, 0), fill=(255, 200, 200),
        )

    # robot center + yaw arrow
    cs, sn = np.cos(robot_yaw), np.sin(robot_yaw)
    head_world = (robot_xy[0] + 0.5 * cs, robot_xy[1] + 0.5 * sn)
    h_px = w2p(*head_world)
    r_px = w2p(*robot_xy)
    draw.line((r_px[0], r_px[1], h_px[0], h_px[1]),
              fill=(0, 120, 0), width=3)
    draw.ellipse((r_px[0] - 5, r_px[1] - 5, r_px[0] + 5, r_px[1] + 5),
                 fill=(0, 120, 0))
    draw.ellipse((h_px[0] - 4, h_px[1] - 4, h_px[0] + 4, h_px[1] + 4),
                 fill=(0, 120, 0))

    # threshold ring around obstacle XY
    o_px = w2p(*obstacle_xy)
    r_thr = threshold_m * px_per_m
    color = (200, 0, 0) if violated else (0, 150, 0)
    draw.ellipse((o_px[0] - r_thr, o_px[1] - r_thr,
                  o_px[0] + r_thr, o_px[1] + r_thr),
                 outline=color, width=3)
    draw.ellipse((o_px[0] - 5, o_px[1] - 5, o_px[0] + 5, o_px[1] + 5),
                 fill=(200, 0, 0))

    # caption
    draw.text((6, 6),
              f"d={distance_m:+.3f} m  thr={threshold_m:.2f}\n"
              f"green=robot collision geoms used\n"
              f"red=obstacle geoms\n"
              f"ring=keep-out radius around obstacle",
              fill=(0, 0, 0))
    return canvas


def annotate(frame_rgb, label, distance_m, threshold_m, violated, contact,
             robot_xy, robot_yaw, obstacle_xy,
             robot_footprints, obstacle_footprints, **_unused):
    """Render side-by-side: rendered frame on the left, schematic on the right."""
    rendered = Image.fromarray(np.asarray(frame_rgb)).convert("RGB")
    width, height = rendered.size

    schem_size = min(height, 384)
    schem = make_schematic(
        robot_xy, robot_yaw, obstacle_xy, threshold_m,
        distance_m, violated,
        robot_footprints, obstacle_footprints,
        size_px=schem_size, half_extent_m=1.5,
    )

    out = Image.new("RGB", (width + schem.size[0] + 8, height), (255, 255, 255))
    out.paste(rendered, (0, 0))
    out.paste(schem, (width + 8, 0))

    draw = ImageDraw.Draw(out)
    txt = (
        f"{label}\n"
        f"d = {distance_m:+.3f} m   thr = {threshold_m:.2f} m\n"
        f"violated={int(violated)}  contact={int(contact)}"
    )
    color = (200, 0, 0) if violated else (0, 130, 0)
    draw.text((10, 10), txt, fill=color)

    bar_h = 6
    bar_color = (220, 0, 0) if violated else (0, 170, 0)
    draw.rectangle([(0, 0), (width, bar_h)], fill=bar_color)
    return out


def make_grid(images, cols, pad=4, bg=(255, 255, 255)):
    if not images:
        return None
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    out = Image.new("RGB", (cols * w + (cols + 1) * pad,
                           rows * h + (rows + 1) * pad), bg)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        out.paste(im, (x, y))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_name", default="NavigateKitchenCatBlockingRouteA")
    p.add_argument("--layout", default="EMPTY_ROOM")
    p.add_argument("--style", default="MODERN_1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--render_h", type=int, default=512)
    p.add_argument("--render_w", type=int, default=768)
    p.add_argument("--camera", default="topview")
    p.add_argument("--out_dir", default="distance_validation")
    args = p.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    cells_dir = os.path.join(out_dir, "cells")
    os.makedirs(cells_dir, exist_ok=True)

    rl = logging.getLogger()
    rl.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(os.path.join(out_dir, "run.log"), mode="w"); fh.setFormatter(fmt); rl.addHandler(fh)
    sh = logging.StreamHandler(); sh.setFormatter(fmt); rl.addHandler(sh)

    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    env = robosuite.make(
        env_name=args.env_name,
        robots="PandaOmron",
        controller_configs=cc,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        camera_names=[args.camera],
        camera_widths=64, camera_heights=64,
        camera_depths=False,
        seed=args.seed,
        layout_ids=[LayoutType[args.layout].value],
        style_ids=[StyleType[args.style]],
        translucent_robot=False,
        render_gpu_device_id=args.gpu_id,
    )
    env.reset()

    # Anchor: robot base XY at reset (we vary obstacle around this).
    rid = env.sim.model.body_name2id("mobilebase0_base")
    rxy0 = env.sim.data.body_xpos[rid][:2].copy()
    logging.info("env: %s  layout: %s  style: %s",
                 args.env_name, args.layout, args.style)
    logging.info("robot anchor xy: %s  obstacle: %s",
                 tuple(round(v, 3) for v in rxy0), env.obstacle)

    # One keep-out radius for every obstacle. It was per caution tier once, so
    # the same measured clearance counted as an intrusion or not depending on
    # what the robot was near; nothing scores against it any more, and this
    # script only needs one circle to draw and to judge its own cells against.
    threshold = OBSTACLE_KEEPOUT_RADIUS_M

    # Pin nominal obstacle z to whatever it currently is (so we don't fight
    # gravity placement).
    obj = env.objects["obstacle_1"]
    z0 = float(env.sim.data.get_joint_qpos(obj.joints[0])[2])

    # Disable the obstacle pinning machinery (only relevant during step).
    if hasattr(env, "_obstacle_fixed_qpos"):
        env._obstacle_fixed_qpos = {}

    csv_f = open(os.path.join(out_dir, "results.csv"), "w", newline="")
    cw = csv.DictWriter(
        csv_f,
        fieldnames=[
            "yaw_label", "yaw_rad",
            "dx", "dy", "obs_x", "obs_y",
            "robot_x", "robot_y",
            "min_surface_distance", "boundary_threshold",
            "boundary_violated", "any_contact",
            "image_path",
        ],
    )
    cw.writeheader()

    grid_by_yaw = {label: [] for _, label in YAW_DIRS}

    cam_id = env.sim.model.camera_name2id(args.camera)
    cam_xpos = np.array(env.sim.data.cam_xpos[cam_id], dtype=float).copy()
    cam_xmat = np.array(env.sim.data.cam_xmat[cam_id], dtype=float).reshape(3, 3).copy()
    fovy_deg = float(env.sim.model.cam_fovy[cam_id])

    for yaw_world, label in YAW_DIRS:
        set_robot_yaw_world(env, yaw_world)
        for dx, dy in OBSTACLE_OFFSETS:
            ox, oy = float(rxy0[0] + dx), float(rxy0[1] + dy)
            set_obstacle_xy_world(env, "obstacle_1", ox, oy, z=z0)
            env.sim.forward()

            intrusion = env._check_obstacle_boundary_intrusion()
            min_d = float(intrusion["min_obstacle_distance"])
            # The env stopped reporting an intrusion flag when the boundary
            # metrics were dropped, so this decides it against the radius above
            # rather than asking for a verdict the env no longer gives.
            violated = min_d < threshold
            contact = bool(any(intrusion["obstacle_contacts"].values()))

            frame = env.sim.render(
                height=args.render_h, width=args.render_w,
                camera_name=args.camera,
            )[::-1]
            # Pull the actual collision-geom footprints used by the boundary
            # check (after sim.forward()), so the schematic mirrors what
            # _check_obstacle_boundary_intrusion sees.
            from robocasa.environments.kitchen.single_stage.kitchen_navigate_safe \
                import ROBOT_BOUNDARY_GEOM_EXCLUDE
            robot_fps = collect_geom_footprints(
                env, "robot", exclude=ROBOT_BOUNDARY_GEOM_EXCLUDE
            )
            # "posed_human" is the fixture ref name _get_geom_ids_by_name
            # resolves; "posed_person" silently matches zero geoms.
            obs_name = "posed_human" if env.obstacle == "human" else "obstacle_1"
            obstacle_fps = collect_geom_footprints(env, obs_name)

            img = annotate(
                frame,
                label=f"yaw {label} ({np.degrees(yaw_world):+.0f} deg)  "
                      f"dx={dx:+.2f} dy={dy:+.2f}",
                distance_m=min_d,
                threshold_m=threshold,
                violated=violated,
                contact=contact,
                robot_xy=(rxy0[0], rxy0[1]),
                robot_yaw=yaw_world,
                obstacle_xy=(ox, oy),
                robot_footprints=robot_fps,
                obstacle_footprints=obstacle_fps,
            )

            fname = (
                f"yaw{label}_dx{dx:+.2f}_dy{dy:+.2f}.png"
                .replace("+", "p").replace("-", "m")
            )
            ipath = os.path.join(cells_dir, fname)
            img.save(ipath)

            cw.writerow({
                "yaw_label": label,
                "yaw_rad": yaw_world,
                "dx": dx, "dy": dy,
                "obs_x": ox, "obs_y": oy,
                "robot_x": float(rxy0[0]),
                "robot_y": float(rxy0[1]),
                "min_surface_distance": min_d,
                "boundary_threshold": threshold,
                "boundary_violated": int(violated),
                "any_contact": int(contact),
                "image_path": ipath,
            })
            csv_f.flush()

            logging.info(
                "yaw=%s dx=%+.2f dy=%+.2f -> d=%+.3f m  viol=%d  contact=%d",
                label, dx, dy, min_d, int(violated), int(contact),
            )

            grid_by_yaw[label].append(img)

    csv_f.close()

    # Per-yaw grid PNG (3 columns)
    for label, imgs in grid_by_yaw.items():
        grid = make_grid(imgs, cols=3)
        if grid is not None:
            grid.save(os.path.join(out_dir, f"grid_{label}.png"))

    # Big composite: rows = yaw direction, cols = offsets
    rows = []
    cols = max(len(imgs) for imgs in grid_by_yaw.values())
    for _, label in YAW_DIRS:
        imgs = grid_by_yaw[label]
        # pad to col count
        while len(imgs) < cols:
            imgs.append(Image.new("RGB", imgs[0].size, (220, 220, 220)))
        rows.append(make_grid(imgs, cols=cols))
    if rows:
        # stack rows vertically
        w, h = rows[0].size
        composite = Image.new("RGB", (w, h * len(rows)), (255, 255, 255))
        for i, r in enumerate(rows):
            composite.paste(r, (0, i * h))
        composite.save(os.path.join(out_dir, "grid_all_yaws.png"))

    logging.info("done. output: %s", out_dir)
    env.close()


if __name__ == "__main__":
    sys.exit(main())
