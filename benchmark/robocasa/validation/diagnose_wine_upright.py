"""Programmatic upright-pose check for every wine variant on every layout.

For each (env_name, layout) we build the env, reset, step `horizon` zero
actions, and record the wine body's local-z axis in world frame at t=0 and
t=horizon. A wine bottle that is upright has world-z component ~1; a fallen
bottle drifts toward 0.

Output:
    wine_upright_check/upright_status.csv
        env_name, layout, init_z_align, final_z_align,
        init_upright (>=0.95), final_upright,
        status   (ok / falls / starts_fallen / error), error
"""
import argparse
import csv
import os
import sys
import traceback

import numpy as np

import robosuite
from robosuite.controllers import load_composite_controller_config

from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS  # noqa
from robocasa.models.scenes.scene_registry import LayoutType, StyleType


WINE_VARIANTS = [
    f"NavigateKitchenWine{mode}Route{r}"
    for mode in ("Blocking", "NonBlocking")
    for r in "ABCDEFG"
]

ALL_LAYOUTS = [
    "ONE_WALL_SMALL", "ONE_WALL_LARGE",
    "L_SHAPED_SMALL", "L_SHAPED_LARGE",
    "U_SHAPED_SMALL", "U_SHAPED_LARGE",
    "G_SHAPED_SMALL", "G_SHAPED_LARGE",
    "GALLEY", "WRAPAROUND",
]

UPRIGHT_THRESHOLD = 0.95  # |body_z . world_z| above this = upright


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="wine_upright_check/upright_status.csv")
    p.add_argument("--style", default="MODERN_1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--layouts", nargs="+", default=None)
    p.add_argument("--variants", nargs="+", default=None)
    return p.parse_args()


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


def wine_z_alignment(env):
    """Return |body_z . world_z| for the obstacle wine bottle."""
    # The wine obstacle is registered as obstacle_1.
    name = "obstacle_1"
    if name not in env.objects:
        return None
    body_id = None
    for k in range(env.sim.model.nbody):
        bn = env.sim.model.body_id2name(k) or ""
        if bn.startswith(name + "_") or bn == name + "_main":
            body_id = k
            break
    if body_id is None:
        # try via geom -> body
        for g in env._get_geom_ids_by_name(name):
            body_id = int(env.sim.model.geom_bodyid[g])
            break
    if body_id is None:
        return None
    R = np.array(env.sim.data.body_xmat[body_id]).reshape(3, 3)
    body_z_world = R[:, 2]  # third column = body z-axis in world frame
    return float(abs(body_z_world[2]))


def run_one(env_name, layout_name, args):
    layout_id = LayoutType[layout_name].value
    style_id = StyleType[args.style]
    env = None
    try:
        env = build_env(env_name, layout_id, style_id, args.seed, args.gpu_id)
        env.reset()
        init_z = wine_z_alignment(env)
        low, high = env.action_spec
        zero = np.zeros_like(high)
        for _ in range(args.horizon):
            env.step(zero)
        final_z = wine_z_alignment(env)
        return {
            "env_name": env_name,
            "layout": layout_name,
            "init_z_align": init_z,
            "final_z_align": final_z,
            "status": "ok",
            "error": "",
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "env_name": env_name,
            "layout": layout_name,
            "init_z_align": None,
            "final_z_align": None,
            "status": "error",
            "error": repr(e),
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def classify(row):
    init_ok = row["init_z_align"] is not None and row["init_z_align"] >= UPRIGHT_THRESHOLD
    final_ok = row["final_z_align"] is not None and row["final_z_align"] >= UPRIGHT_THRESHOLD
    if row["status"] == "error":
        return "error"
    if init_ok and final_ok:
        return "ok"
    if not init_ok:
        return "starts_fallen"
    if init_ok and not final_ok:
        return "falls"
    return "unknown"


def main():
    args = parse_args()
    layouts = args.layouts or ALL_LAYOUTS
    variants = args.variants or WINE_VARIANTS
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = []
    n_total = len(layouts) * len(variants)
    i = 0
    for L in layouts:
        for V in variants:
            i += 1
            row = run_one(V, L, args)
            row["init_upright"] = int(row["init_z_align"] is not None
                                      and row["init_z_align"] >= UPRIGHT_THRESHOLD)
            row["final_upright"] = int(row["final_z_align"] is not None
                                       and row["final_z_align"] >= UPRIGHT_THRESHOLD)
            row["status_class"] = classify(row)
            rows.append(row)
            print(f"[{i:3d}/{n_total}] {V}/{L}: init={row['init_z_align']}  "
                  f"final={row['final_z_align']}  -> {row['status_class']}",
                  flush=True)

    fields = ["env_name", "layout",
              "init_z_align", "final_z_align",
              "init_upright", "final_upright",
              "status", "status_class", "error"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    bad = [r for r in rows if r["status_class"] not in ("ok",)]
    print(f"\nTotal={len(rows)}  ok={len(rows)-len(bad)}  bad={len(bad)}")
    by_class = {}
    for r in bad:
        by_class.setdefault(r["status_class"], []).append(r)
    for cls, items in sorted(by_class.items()):
        print(f"\n=== {cls} ({len(items)}) ===")
        for r in items[:30]:
            print(f"  {r['env_name']:50s} {r['layout']:18s}  "
                  f"init={r['init_z_align']}  final={r['final_z_align']}")
        if len(items) > 30:
            print(f"  ... and {len(items) - 30} more")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
