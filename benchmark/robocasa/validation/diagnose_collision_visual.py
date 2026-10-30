"""Compare collision-geom vs. visual-mesh surface distance.

For one (env, layout, style), build the env, reset, and compute the minimum
signed surface-to-surface distance between robot and obstacle separately for:

    - collision-only geoms  (what _check_obstacle_boundary_intrusion uses)
    - visual-mesh geoms     (what the user actually sees)

Robot visual geoms = group >= 1, vis-mesh; collision geoms = group 0 with
contype/conaffinity != 0. The split mirrors ``_filter_collision_geoms``.

Example:
    python diagnose_collision_visual.py \\
        --env_name NavigateKitchenVaseBlockingRouteD --layout GALLEY \\
        --style MODERN_1
"""

import argparse
import os
import sys

import mujoco
import numpy as np

import robosuite
from robosuite.controllers import load_composite_controller_config
from robocasa.models.scenes.scene_registry import LayoutType, StyleType


GEOM_TYPE_NAMES = {
    0: "plane", 1: "hfield", 2: "sphere", 3: "capsule",
    4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh",
}


def split_geoms(env, obj_name):
    """Return (collision_geoms, visual_geoms) for `obj_name`.

    collision: contype!=0 or conaffinity!=0, AND group < 1 (matches the
        existing _filter_collision_geoms logic).
    visual: group >= 1 OR (contype==0 and conaffinity==0).
    """
    raw = env._get_geom_ids_by_name(obj_name)
    coll, vis = set(), set()
    m = env.sim.model
    for g in raw:
        ct = int(m.geom_contype[g])
        ca = int(m.geom_conaffinity[g])
        gp = int(m.geom_group[g])
        if (ct != 0 or ca != 0) and gp < 1:
            coll.add(g)
        if gp >= 1 or (ct == 0 and ca == 0):
            vis.add(g)
    return coll, vis


def min_dist(env, a_ids, b_ids, distmax=2.0):
    """Min signed distance + closest pair (id_a, id_b, fromto)."""
    m = env.sim.model._model
    d = env.sim.data._data
    best = (float("inf"), None, None, None)
    for ga in a_ids:
        for gb in b_ids:
            ft = np.zeros(6, dtype=np.float64)
            sd = mujoco.mj_geomDistance(m, d, ga, gb, distmax, ft)
            if sd < best[0]:
                best = (float(sd), int(ga), int(gb), ft.copy())
    return best


def fmt_geom(env, g):
    name = env.sim.model.geom_id2name(g) or f"<{g}>"
    t = GEOM_TYPE_NAMES.get(int(env.sim.model.geom_type[g]), "?")
    sz = [round(float(s), 3) for s in env.sim.model.geom_size[g]]
    grp = int(env.sim.model.geom_group[g])
    ct = int(env.sim.model.geom_contype[g])
    ca = int(env.sim.model.geom_conaffinity[g])
    return f"{name} ({t} sz={sz} group={grp} contype={ct} conaffinity={ca})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_name", default="NavigateKitchenVaseBlockingRouteD")
    p.add_argument("--layout", default="GALLEY")
    p.add_argument("--style", default="MODERN_1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--distmax", type=float, default=2.0)
    args = p.parse_args()

    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    env = robosuite.make(
        env_name=args.env_name,
        robots="PandaOmron",
        controller_configs=cc,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        camera_names=["topview"],
        camera_widths=64, camera_heights=64,
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

    # "posed_human" is the fixture ref name _get_geom_ids_by_name resolves;
    # "posed_person" silently matches zero geoms.
    obstacle_names = (
        ["posed_human"] if env.obstacle == "human"
        else [n for n in env.objects if n.startswith("obstacle_")]
    )

    rob_coll, rob_vis = split_geoms(env, "robot")
    print(f"\nrobot   geom counts -- collision: {len(rob_coll)}   visual: {len(rob_vis)}")

    for obs in obstacle_names:
        obs_coll, obs_vis = split_geoms(env, obs)
        print(f"\n=== obstacle: {obs} -- collision: {len(obs_coll)}  visual: {len(obs_vis)} ===")

        # 1) Collision-only (current behaviour)
        if rob_coll and obs_coll:
            sd, ga, gb, ft = min_dist(env, rob_coll, obs_coll, args.distmax)
            print(f"\n[COLLISION-only]  min surface distance = {sd:+.4f} m")
            if ga is not None:
                print(f"  A: {fmt_geom(env, ga)}")
                print(f"  B: {fmt_geom(env, gb)}")
                print(f"  fromto: A={tuple(round(v,3) for v in ft[:3])}  "
                      f"B={tuple(round(v,3) for v in ft[3:])}")
        else:
            print("\n[COLLISION-only]  empty geom set on one side")

        # 2) Visual meshes
        if rob_vis and obs_vis:
            sd, ga, gb, ft = min_dist(env, rob_vis, obs_vis, args.distmax)
            print(f"\n[VISUAL meshes]   min surface distance = {sd:+.4f} m")
            if ga is not None:
                print(f"  A: {fmt_geom(env, ga)}")
                print(f"  B: {fmt_geom(env, gb)}")
                print(f"  fromto: A={tuple(round(v,3) for v in ft[:3])}  "
                      f"B={tuple(round(v,3) for v in ft[3:])}")
        else:
            print("\n[VISUAL meshes]   empty visual geom set on one side")

    env.close()


if __name__ == "__main__":
    sys.exit(main())
