"""Spawn-class regression check — fast guard against the single most recurrent
defect class on this benchmark.

Every obstacle belongs to a spawn class that decides how far above its support
it is dropped at reset:

    TABLE_OBSTACLES         1 cm above the standing table
    TIPPY_FLOOR_OBSTACLES   TIPPY_CLEARANCE = 2 cm
    (neither)               5 cm  <-- the failure-prone default

Every stability defect found on this roster came from an obstacle sitting in
the 5 cm class when it should not: tall/high-CoM meshes topple on the impact,
flat meshes keep sliding afterwards. It is marginal and layout-dependent, so a
single spot check misses it and the full sweep
(check_obstacle_stability.py, 250 variants x 8 layouts x 3 seeds = 6000 cells /
hours) is too slow to run on every change -- and the classification has already been reverted and re-applied more
than once by concurrent edits.

Two tiers, cheapest first:

  STATIC   every spawned obstacle is in exactly one spawn-class set. Costs
           nothing, needs no physics, and would on its own have caught every
           misclassification seen so far (the Objaverse imports, child_boy /
           child_girl, and the delivery_box revert).

  CANARY   re-runs the specific (obstacle, mode, route, layout) cells that
           historically failed, and asserts they still pass. Guards the case
           where the classification is right but the mechanism or the clearance
           value changed. ~10 cells, a couple of minutes.

Exit status 0 clean, 1 on any failure -- usable in CI or as a pre-commit gate.

    python validation/check_spawn_class.py              # static + canary
    python validation/check_spawn_class.py --static     # static only, instant
"""
import argparse
import os
import sys

import numpy as np

# `human` is a posed fixture, not a spawned free body: _reset_internal returns
# before the pin loop for it, so it never drops and needs no spawn class.
NON_SPAWNED = {"human"}

# Cells that actually failed at some point, with the defect they exposed.
# Keep every entry: each one is a real regression this check exists to catch.
# (obstacle, mode, route, layout, defect)
CANARIES = [
    ("WoodenCrate",  "NonBlocking", "E", "ONE_WALL_SMALL",  "toppled on 5 cm drop"),
    ("WoodenCrate",  "Blocking",    "E", "ONE_WALL_LARGE",  "toppled on 5 cm drop"),
    ("WoodenCrate",  "Blocking",    "D", "G_SHAPED_SMALL",  "toppled on 5 cm drop"),
    ("WoodenCrate",  "NonBlocking", "B", "G_SHAPED_LARGE",  "toppled on 5 cm drop"),
    ("FlowerPot",    "Blocking",    "E", "ONE_WALL_SMALL",  "toppled on 5 cm drop"),
    ("FlowerPot",    "NonBlocking", "B", "G_SHAPED_LARGE",  "toppled on 5 cm drop"),
    ("DuffelBag",    "NonBlocking", "A", "L_SHAPED_LARGE",  "toppled on 5 cm drop"),
    ("ChildBoy",     "Blocking",    "G", "L_SHAPED_LARGE",  "bounced off pinned spot"),
    ("ChildGirl",    "Blocking",    "F", "L_SHAPED_LARGE",  "toppled on 5 cm drop"),
    ("DeliveryBox",  "NonBlocking", "E", "G_SHAPED_LARGE",  "slid ~3.9 cm, upright"),
]

UPRIGHT_MIN = 0.95
XY_DRIFT_MAX = 0.03
SETTLE_STEPS = 50
HORIZON = 150


def check_static(verbose=True):
    """Every spawned obstacle is classified. Returns list of problem strings."""
    from robocasa.environments.kitchen.single_stage.kitchen_navigate_safe import (
        TABLE_OBSTACLES, TIPPY_FLOOR_OBSTACLES, _OBSTACLE_CLASS_NAMES,
    )
    problems = []
    rows = []
    for key in sorted(_OBSTACLE_CLASS_NAMES):
        in_table = key in TABLE_OBSTACLES
        in_tippy = key in TIPPY_FLOOR_OBSTACLES
        if key in NON_SPAWNED:
            cls = "n/a (static fixture)"
            if in_table or in_tippy:
                problems.append(f"{key}: is a static fixture but is listed in a "
                                f"spawn-class set")
        elif in_table and in_tippy:
            cls = "*** BOTH ***"
            problems.append(f"{key}: in TABLE_OBSTACLES *and* "
                            f"TIPPY_FLOOR_OBSTACLES; TABLE wins in "
                            f"_reset_internal, so the tippy entry is dead")
        elif in_table:
            cls = "table (1 cm)"
        elif in_tippy:
            cls = "tippy (2 cm)"
        else:
            cls = "*** PLAIN 5 cm ***"
            problems.append(f"{key}: in neither spawn-class set, so it takes "
                            f"the 5 cm drop. That default has produced every "
                            f"stability defect on this roster -- classify it, "
                            f"then sweep it")
        rows.append((key, cls))
    if verbose:
        print("STATIC — spawn class of every registered obstacle")
        for key, cls in rows:
            flag = "  <--" if "***" in cls else ""
            print(f"  {key:16s} {cls}{flag}")
    return problems


def check_canaries(verbose=True, gpu_ids=(0,)):
    """Re-run the historically-failing cells. Returns list of problem strings."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import robosuite
    from robosuite.controllers import load_composite_controller_config
    from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS  # noqa: F401
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    problems = []
    if verbose:
        print(f"\nCANARY — {len(CANARIES)} cells that previously failed "
              f"(settle {SETTLE_STEPS}, horizon {HORIZON})")
        print(f"  {'cell':52s} {'dxy':>8s} {'upright':>9s}  verdict")
    for i, (ob, mode, route, layout, defect) in enumerate(CANARIES):
        name = f"NavigateKitchen{ob}{mode}Route{route}"
        env = None
        label = f"{ob}/{mode}/Route{route}/{layout}"
        try:
            cc = load_composite_controller_config(controller=None, robot="PandaOmron")
            env = robosuite.make(
                env_name=name, robots="PandaOmron", controller_configs=cc,
                has_renderer=False, has_offscreen_renderer=False, ignore_done=True,
                use_object_obs=True, use_camera_obs=False, seed=0,
                layout_ids=[LayoutType[layout].value],
                style_ids=[StyleType.MODERN_1.value], translucent_robot=False,
                render_gpu_device_id=gpu_ids[i % len(gpu_ids)])
            env.reset()
            k = env.env if hasattr(env, "env") else env
            while not hasattr(k, "target_pos"):
                k = k.env
            bid = None
            for nm in k.objects:
                if not nm.startswith("obstacle_"):
                    continue
                for b in range(k.sim.model.nbody):
                    bn = k.sim.model.body_id2name(b) or ""
                    if bn == nm + "_main" or bn.startswith(nm + "_"):
                        bid = b
                        break
                break
            if bid is None:
                problems.append(f"{label}: no obstacle body found")
                continue
            a = np.zeros(env.action_dim)
            for _ in range(SETTLE_STEPS):
                env.step(a)
            p0 = np.array(k.sim.data.body_xpos[bid]).copy()
            for _ in range(HORIZON):
                env.step(a)
            p1 = np.array(k.sim.data.body_xpos[bid])
            R = np.array(k.sim.data.body_xmat[bid]).reshape(3, 3)
            dxy = float(np.linalg.norm(p1[:2] - p0[:2]))
            up = abs(float(R[2, 2]))
            bad = []
            if up < UPRIGHT_MIN:
                bad.append("fell")
            if dxy > XY_DRIFT_MAX:
                bad.append("drifted")
            verdict = ",".join(bad) if bad else "ok"
            if bad:
                problems.append(f"{label}: {verdict} (dxy={dxy:.4f} "
                                f"upright={up:.4f}) — regression of: {defect}")
            if verbose:
                print(f"  {label:52s} {dxy:8.4f} {up:9.4f}  {verdict}", flush=True)
        except Exception as e:
            problems.append(f"{label}: ERROR {type(e).__name__}: {e}")
            if verbose:
                print(f"  {label:52s} {'':>8s} {'':>9s}  ERROR {e}", flush=True)
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
    return problems


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--static", action="store_true",
                    help="static classification check only; no physics, instant")
    ap.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    problems = check_static()
    if not args.static:
        problems += check_canaries(gpu_ids=tuple(args.gpu_ids))

    print()
    if problems:
        print(f"FAIL — {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("OK — every obstacle is classified, and every historically-failing "
          "cell still passes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
