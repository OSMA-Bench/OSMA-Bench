"""Check whether the route's destination pose is itself inside an obstacle's
safety boundary.

For every row in violations_only.csv (or any CSV produced by
log_initial_violations.py with the same schema), build the env, reset, then
teleport the mobile base to (target_x, target_y) the env recorded as the
destination. Re-run the boundary intrusion check at that pose and log:
    - destination_min_surface_distance
    - destination_boundary_violated
    - destination_any_contact

A violation here means the destination cannot be reached without entering
the obstacle's keep-out radius -- the route itself is mis-specified.

Example:
    python check_destination_reachable.py \
        --in initial_violations/violations_only.csv \
        --out initial_violations/destination_reachability.csv
"""
import argparse
import csv
import logging
import os
import sys
import traceback

import numpy as np

import robosuite
from robosuite.controllers import load_composite_controller_config

from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS  # noqa: F401  (registers envs)
from robocasa.models.scenes.scene_registry import LayoutType, StyleType


# Same source as log_initial_violations.py: the env's own per-obstacle radii,
# One keep-out radius for every obstacle. It was per caution tier once
# (0.6 / 0.4 / 0.2 m), which made the same measured clearance count as an
# intrusion or not depending on what the robot was near. Nothing scores against
# it any more -- collision-free success counts contact -- so it survives only as
# the circle a diagnostic draws and judges its own cells against.
from robocasa.environments.kitchen.single_stage.kitchen_navigate_safe import (
    OBSTACLE_KEEPOUT_RADIUS_M,
)


def _joint_qposadr(model, name):
    return int(model.jnt_qposadr[model.joint_name2id(name)])


def teleport_base_to_world_xy(env, target_x, target_y):
    """Translate the mobile base so its world XY equals (target_x, target_y).

    Mobile-base XY is driven by two slide joints whose axes are local. We solve
    for the joint deltas given the body's current world rotation matrix.
    """
    fwd = _joint_qposadr(env.sim.model, "mobilebase0_joint_mobile_forward")
    side = _joint_qposadr(env.sim.model, "mobilebase0_joint_mobile_side")
    body_id = env.sim.model.body_name2id("mobilebase0_base")

    cur_xy = env.sim.data.body_xpos[body_id][:2].copy()
    R = np.array(env.sim.data.body_xmat[body_id]).reshape(3, 3)

    delta_world = np.array([target_x - cur_xy[0], target_y - cur_xy[1]])
    delta_q = np.linalg.solve(R[:2, :2], delta_world)
    env.sim.data.qpos[fwd] += delta_q[0]
    env.sim.data.qpos[side] += delta_q[1]
    env.sim.forward()


def build_env(env_name, layout_id, style_id, seed, gpu_id):
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    return robosuite.make(
        env_name=env_name,
        robots="PandaOmron",
        controller_configs=cc,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        seed=seed,
        layout_ids=[layout_id],
        style_ids=[style_id],
        translucent_robot=False,
        render_gpu_device_id=gpu_id,
    )


def threshold_for(env):
    """The keep-out radius reachability is judged against -- one for every
    obstacle; see OBSTACLE_KEEPOUT_RADIUS_M."""
    return OBSTACLE_KEEPOUT_RADIUS_M


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_csv",
                   default="initial_violations/violations_only.csv",
                   help="Input CSV (rows with env_name/layout/style/seed/target_x/target_y)")
    p.add_argument("--out", dest="out_csv",
                   default="initial_violations/destination_reachability.csv",
                   help="Output CSV path")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--limit", type=int, default=None,
                   help="Only process first N rows (debug)")
    return p.parse_args()


OUT_FIELDS = [
    "env_name", "layout", "style", "seed",
    "obstacle_kind", "route", "blocking_mode",
    "target_x", "target_y",
    "boundary_threshold",
    "init_boundary_violated", "init_min_surface_distance",
    "dest_robot_x", "dest_robot_y",
    "dest_boundary_violated", "dest_any_contact",
    "dest_min_surface_distance",
    "reachable",  # 1 if dest_boundary_violated == 0
    "status", "error",
]


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    rows_in = []
    with open(args.in_csv) as f:
        for row in csv.DictReader(f):
            rows_in.append(row)
    if args.limit:
        rows_in = rows_in[:args.limit]
    logging.info("Loaded %d rows from %s", len(rows_in), args.in_csv)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out_f = open(args.out_csv, "w", newline="")
    cw = csv.DictWriter(out_f, fieldnames=OUT_FIELDS)
    cw.writeheader()

    n_reach = 0
    n_unreach = 0
    n_err = 0
    try:
        for i, row in enumerate(rows_in):
            env_name = row["env_name"]
            layout = row["layout"]
            style = row["style"]
            seed = int(row["seed"])
            target_x = row.get("target_x", "")
            target_y = row.get("target_y", "")
            try:
                tx = float(target_x)
                ty = float(target_y)
            except (TypeError, ValueError):
                logging.warning("[skip] %s/%s seed=%d: missing target xy",
                                env_name, layout, seed)
                continue

            out = {k: "" for k in OUT_FIELDS}
            out.update({
                "env_name": env_name, "layout": layout, "style": style,
                "seed": seed, "target_x": tx, "target_y": ty,
                "obstacle_kind": row.get("obstacle_kind", ""),
                "route": row.get("route", ""),
                "blocking_mode": row.get("blocking_mode", ""),
                "init_boundary_violated": row.get("boundary_violated", ""),
                "init_min_surface_distance": row.get("min_surface_distance", ""),
                "status": "ok",
            })

            env = None
            try:
                layout_id = LayoutType[layout].value
                style_id = StyleType[style]
                env = build_env(env_name, layout_id, style_id, seed, args.gpu_id)
                env.reset()
                thr = threshold_for(env)
                out["boundary_threshold"] = thr

                teleport_base_to_world_xy(env, tx, ty)
                bid = env.sim.model.body_name2id("mobilebase0_base")
                rb = env.sim.data.body_xpos[bid][:2].tolist()
                out["dest_robot_x"] = float(rb[0])
                out["dest_robot_y"] = float(rb[1])

                intrusion = env._check_obstacle_boundary_intrusion()
                violated = float(intrusion["min_obstacle_distance"]) < thr
                contact = bool(any(intrusion["obstacle_contacts"].values()))
                out["dest_boundary_violated"] = int(violated)
                out["dest_any_contact"] = int(contact)
                out["dest_min_surface_distance"] = float(intrusion["min_obstacle_distance"])
                out["reachable"] = int(not violated)
                if violated:
                    n_unreach += 1
                else:
                    n_reach += 1
                logging.info(
                    "[%3d/%3d] %s/%s seed=%d  thr=%.2f  dest_dist=%+.3f  reachable=%d",
                    i + 1, len(rows_in), env_name, layout, seed, thr,
                    out["dest_min_surface_distance"], out["reachable"],
                )
            except Exception as e:
                out["status"] = "error"
                out["error"] = repr(e)
                n_err += 1
                logging.error("[err] %s/%s seed=%d: %s", env_name, layout, seed, e)
                traceback.print_exc()
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass

            cw.writerow(out)
            out_f.flush()
    finally:
        out_f.close()

    logging.info("Done. reachable=%d  unreachable=%d  errors=%d  -> %s",
                 n_reach, n_unreach, n_err, args.out_csv)


if __name__ == "__main__":
    sys.exit(main())
