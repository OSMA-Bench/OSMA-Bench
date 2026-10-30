"""Diagnose surface-distance reporting for one navigate_safe scenario.

Builds the env, resets, then enumerates every (robot collision-geom, obstacle
collision-geom) pair and prints the distance, geom names/types/sizes, and
fromto endpoints. Renders a topview frame with the closest pair's endpoints
marked so we can see whether the colliding geoms are visible or just
collision-only inflations.

Example:
    mjpython diagnose_collision.py \
        --env_name NavigateKitchenVaseBlockingRouteD \
        --layout GALLEY --style MODERN_1
"""

import argparse
import os
import sys

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

import robosuite
from robosuite.controllers import load_composite_controller_config

from robocasa.models.scenes.scene_registry import LayoutType, StyleType
# One keep-out radius for every obstacle. It was per caution tier once
# (0.6 / 0.4 / 0.2 m), which made the same measured clearance count as an
# intrusion or not depending on what the robot was near. Nothing scores against
# it any more -- collision-free success counts contact -- so it survives only as
# the circle a diagnostic draws and judges its own cells against.
from robocasa.environments.kitchen.single_stage.kitchen_navigate_safe import (
    OBSTACLE_KEEPOUT_RADIUS_M,
)


def _filter_collision_geoms(env, geom_ids):
    return env._filter_collision_geoms(geom_ids)


def _names_and_pair_dists(env, obj_a, obj_b, distmax=2.0):
    """Return list of (dist, ga, gb, name_a, name_b, fromto)."""
    a_ids = _filter_collision_geoms(env, env._get_geom_ids_by_name(obj_a))
    b_ids = _filter_collision_geoms(env, env._get_geom_ids_by_name(obj_b))
    m = env.sim.model._model
    d = env.sim.data._data
    out = []
    for ga in a_ids:
        for gb in b_ids:
            fromto = np.zeros(6, dtype=np.float64)
            sd = mujoco.mj_geomDistance(m, d, ga, gb, distmax, fromto)
            out.append((float(sd), int(ga), int(gb), fromto.copy()))
    return out


def _project(pt_world, cam_xpos, cam_xmat, fovy_deg, height, width,
             image_y_flipped=True):
    p_rel = np.asarray(pt_world, dtype=float) - cam_xpos
    p_cam = cam_xmat.T @ p_rel
    z_forward = -p_cam[2]
    if z_forward <= 1e-6:
        return None
    fovy_rad = np.deg2rad(fovy_deg)
    f = (height / 2.0) / np.tan(fovy_rad / 2.0)
    u = width / 2.0 + f * (p_cam[0] / z_forward)
    v_raw = height / 2.0 - f * (p_cam[1] / z_forward)
    v = (height - 1.0 - v_raw) if image_y_flipped else v_raw
    return float(u), float(v), float(z_forward), float(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_name", default="NavigateKitchenVaseBlockingRouteD")
    p.add_argument("--layout", default="GALLEY")
    p.add_argument("--style", default="MODERN_1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--out_dir", default="diagnose_collision")
    p.add_argument("--render_h", type=int, default=768)
    p.add_argument("--render_w", type=int, default=1024)
    p.add_argument("--camera", default="topview")
    args = p.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

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
        camera_widths=128, camera_heights=128,
        camera_depths=False,
        seed=args.seed,
        layout_ids=[LayoutType[args.layout].value],
        style_ids=[StyleType[args.style]],
        translucent_robot=False,
        render_gpu_device_id=args.gpu_id,
    )
    env.reset()

    print(f"\nenv: {args.env_name}  layout: {args.layout}  style: {args.style}")
    print(f"obstacle kind: {env.obstacle}")

    # Identify the obstacle name(s) actually present.
    obstacle_names = []
    if env.obstacle == "human":
        # "posed_human" is the fixture ref name _get_geom_ids_by_name resolves
        # (and what the env's own boundary check passes). "posed_person" falls
        # through to its substring fallback and silently matches zero geoms.
        obstacle_names.append("posed_human")
    else:
        obstacle_names.extend(
            n for n in env.objects if n.startswith("obstacle_")
        )

    # Robot base XY
    rid = env.sim.model.body_name2id("mobilebase0_base")
    robot_xy = env.sim.data.body_xpos[rid][:2].tolist()
    print(f"robot_base_xy: {robot_xy}")

    # Camera params
    cid = env.sim.model.camera_name2id(args.camera)
    cam_xpos = np.array(env.sim.data.cam_xpos[cid], dtype=float).copy()
    cam_xmat = np.array(env.sim.data.cam_xmat[cid], dtype=float).reshape(3, 3).copy()
    fovy = float(env.sim.model.cam_fovy[cid])

    # Render frame
    frame = env.sim.render(
        height=args.render_h, width=args.render_w, camera_name=args.camera
    )[::-1]
    img = Image.fromarray(np.asarray(frame)).convert("RGBA")
    draw = ImageDraw.Draw(img)
    height, width = img.size[1], img.size[0]

    GEOM_TYPE_NAMES = {
        0: "plane", 1: "hfield", 2: "sphere", 3: "capsule",
        4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh",
    }

    overall_min = (float("inf"), None, None, None, None, None)
    for obs_name in obstacle_names:
        print(f"\n--- pairs for robot vs {obs_name} ---")
        pairs = _names_and_pair_dists(env, "robot", obs_name)
        pairs.sort(key=lambda x: x[0])
        for sd, ga, gb, fromto in pairs[:8]:
            ga_name = env.sim.model.geom_id2name(ga) or f"<id {ga}>"
            gb_name = env.sim.model.geom_id2name(gb) or f"<id {gb}>"
            ga_type = GEOM_TYPE_NAMES.get(int(env.sim.model.geom_type[ga]), "?")
            gb_type = GEOM_TYPE_NAMES.get(int(env.sim.model.geom_type[gb]), "?")
            ga_size = env.sim.model.geom_size[ga].tolist()
            gb_size = env.sim.model.geom_size[gb].tolist()
            ga_pos = env.sim.data.geom_xpos[ga].tolist()
            gb_pos = env.sim.data.geom_xpos[gb].tolist()
            print(
                f"  d={sd:+.4f}  {ga_name:40s}({ga_type:8s} sz={[round(s,3) for s in ga_size]})"
                f"  <->  {gb_name}({gb_type} sz={[round(s,3) for s in gb_size]})"
            )
            print(f"      ga_pos={[round(x,3) for x in ga_pos]}  gb_pos={[round(x,3) for x in gb_pos]}")
            if fromto is not None and np.linalg.norm(fromto) > 0:
                print(
                    f"      fromto: A={tuple(round(v,3) for v in fromto[:3])}  "
                    f"B={tuple(round(v,3) for v in fromto[3:])}"
                )
            if sd < overall_min[0]:
                overall_min = (sd, ga, gb, ga_name, gb_name, fromto.copy() if fromto is not None else None)

    print(f"\nOVERALL MIN: d={overall_min[0]:.4f}  "
          f"{overall_min[3]} <-> {overall_min[4]}")

    # Annotate the rendered frame: red circle at obstacle XY at threshold,
    # plus mark the closest geom pair endpoints (yellow lines from A to B).
    for obs_name in obstacle_names:
        # obstacle world XY (using object placement or body position)
        if obs_name == "posed_human":
            bid = env.sim.model.body_name2id("posed_human_main_group_main")
            opos = env.sim.data.body_xpos[bid].tolist()
        else:
            obj = env.objects[obs_name]
            qpos = env.sim.data.get_joint_qpos(obj.joints[0])
            opos = [float(qpos[0]), float(qpos[1]), float(qpos[2])]
        if opos is None:
            continue
        proj = _project((opos[0], opos[1], 0.0),
                        cam_xpos, cam_xmat, fovy, height, width)
        if proj is None:
            continue
        u, v, depth, f = proj
        # Draw the obstacle's keep-out circle at the one radius the roster
        # uses; it is a drawing aid, not a threshold anything scores against.
        thr = OBSTACLE_KEEPOUT_RADIUS_M
        radius_px = max(2.0, thr * f / depth)
        draw.ellipse(
            (u - radius_px, v - radius_px, u + radius_px, v + radius_px),
            outline=(255, 0, 0, 255), width=4,
        )
        draw.ellipse((u - 3, v - 3, u + 3, v + 3), fill=(255, 0, 0, 255))

    # closest pair endpoints
    if overall_min[5] is not None:
        ftA = overall_min[5][:3]; ftB = overall_min[5][3:]
        pA = _project(ftA, cam_xpos, cam_xmat, fovy, height, width)
        pB = _project(ftB, cam_xpos, cam_xmat, fovy, height, width)
        if pA and pB:
            draw.line((pA[0], pA[1], pB[0], pB[1]), fill=(255, 220, 0, 255), width=4)
            draw.ellipse((pA[0]-5, pA[1]-5, pA[0]+5, pA[1]+5), fill=(255, 220, 0, 255))
            draw.ellipse((pB[0]-5, pB[1]-5, pB[0]+5, pB[1]+5), fill=(0, 0, 255, 255))

    # robot dot
    pr = _project((robot_xy[0], robot_xy[1], 0.0),
                  cam_xpos, cam_xmat, fovy, height, width)
    if pr:
        draw.ellipse((pr[0]-5, pr[1]-5, pr[0]+5, pr[1]+5), fill=(0, 200, 0, 255))

    label = (f"{args.env_name} / {args.layout} / {args.style}\n"
             f"d={overall_min[0]:+.3f} m   "
             f"A={overall_min[3]}  B={overall_min[4]}")
    draw.text((8, 8), label, fill=(0, 0, 0, 255))

    out_path = os.path.join(
        out_dir, f"{args.env_name}_{args.layout}_{args.style}_seed{args.seed}.png"
    )
    img.convert("RGB").save(out_path)
    print(f"\nsaved: {out_path}")
    env.close()


if __name__ == "__main__":
    sys.exit(main())
