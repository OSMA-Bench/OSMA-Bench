"""Verify every navigate_safe obstacle weighs what OBSTACLE_MASS_KG says it does.

MuJoCo never reads a mass for these objects. It derives one:

    mass = density * V_proxy * scale^3

where `density` is stamped onto every geom by MJCFObject and V_proxy is the volume of
the COLLISION PROXY -- a convex decomposition whose hulls overlap, so it overshoots the
real object's volume by roughly 2x. The densities in OBJ_CATEGORIES are therefore not
material densities; each is an effective figure picked as `target_mass / V_proxy` to land
the object on the mass OBSTACLE_MASS_KG specifies.

That makes the realized mass silently dependent on the collision proxy. Retightening a
proxy -- which this repo does (see validation/rebuild_obstacle_collision.py, and the dog
proxy rebuilt on 2026-08-25) -- shrinks V_proxy and lightens the object by the same ratio,
with nothing raising. This script closes that loop: it builds each obstacle exactly as the
env does, weighs it, and fails if any obstacle has drifted off its target. On failure it
prints the corrected density to paste back into OBJ_CATEGORIES.

Multi-instance categories (`wine` has 12 bottles) are checked per model, since each may
carry its own density.

`human` is checked differently, and must be. It is not an OBJ_CATEGORIES object at all but
the `posed_human` fixture, which reads the asset directly and rescales it per layout, so
there is no scale to compile it at outside a built scene. Weighing the (unused)
OBJ_CATEGORIES["posed_human"] entry instead would report a number the simulation never
uses -- the trap this script exists to catch. So the human is weighed inside a real env.
That costs ~40 s and a render device; pass --skip-human to drop it, at the price of leaving
the roster's heaviest obstacle unchecked.

Exit status: 0 clean, 1 if any obstacle is off target -- usable in CI.

    python validation/check_obstacle_mass.py
    python validation/check_obstacle_mass.py --tol 0.05    # 5% instead of 2%
    python validation/check_obstacle_mass.py --skip-human  # no env build, no GPU needed
    python validation/check_obstacle_mass.py --quiet       # only problems
"""
import argparse
import os
import sys

import mujoco
import numpy as np
from robosuite.models.arenas import EmptyArena
from robosuite.models.world import MujocoWorldBase

from robocasa.environments.kitchen.single_stage.kitchen_navigate_safe import (
    OBSTACLE_MASS_KG,
    TIER_TO_OBSTACLES,
)
from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES, OBJ_GROUPS
from robocasa.models.objects.objects import MJCFObject

TIER_ORDER = ("High", "Medium", "Low")
_TIER_OF = {o: t for t, objs in TIER_TO_OBSTACLES.items() for o in objs}


def _categories_for(obstacle):
    """Obstacle key -> the OBJ_CATEGORIES entries it spawns from.

    Obstacle keys are the env's spelling and mostly ARE category names, but some are
    group names that redirect ('human' -> 'posed_human'), which is why this goes through
    OBJ_GROUPS rather than indexing OBJ_CATEGORIES directly.
    """
    grp = OBJ_GROUPS.get(obstacle, obstacle)
    cats = grp if isinstance(grp, (list, tuple)) else [grp]
    missing = [c for c in cats if c not in OBJ_CATEGORIES]
    if missing:
        raise KeyError(f"{obstacle}: no OBJ_CATEGORIES entry for {missing}")
    return cats


def weigh(obstacle):
    """Build every model of `obstacle` the way the env does; return per-model results.

    Yields (model_name, density, mass_kg, v_proxy_m3). The env applies no density or
    scale override for obstacles, so `get_mjcf_kwargs` reproduces the spawned object
    exactly -- including a per-model density when the category declares one.
    """
    out = []
    for cat in _categories_for(obstacle):
        for reg, objcat in OBJ_CATEGORIES[cat].items():
            for path in objcat.mjcf_paths:
                kwargs = objcat.get_mjcf_kwargs(mjcf_path=path)
                kwargs.pop("priority", None)
                density = kwargs["density"]
                obj = MJCFObject(name="obs", mjcf_path=path, **kwargs)
                world = MujocoWorldBase()
                world.merge(EmptyArena())
                world.merge_assets(obj)
                world.worldbody.append(obj.get_obj())
                model = mujoco.MjModel.from_xml_string(world.get_xml())
                # The object may compile to several bodies; sum the ones under its root.
                mass = sum(
                    model.body_mass[b]
                    for b in range(model.nbody)
                    if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")
                    .startswith("obs")
                )
                name = os.path.basename(os.path.dirname(path))
                if density is None:
                    # No density declared: the object inherits whatever the asset was
                    # exported with, which is the pre-2026-08 state every obstacle was in
                    # and the state a newly added one starts in. Recompile at a known
                    # density to recover V_proxy, so the report can still suggest the
                    # density this obstacle needs instead of crashing on a None.
                    probe = MJCFObject(name="obs", mjcf_path=path,
                                       **{**kwargs, "density": 100.0})
                    pw = MujocoWorldBase()
                    pw.merge(EmptyArena())
                    pw.merge_assets(probe)
                    pw.worldbody.append(probe.get_obj())
                    pm = mujoco.MjModel.from_xml_string(pw.get_xml())
                    ref = sum(
                        pm.body_mass[b]
                        for b in range(pm.nbody)
                        if (mujoco.mj_id2name(pm, mujoco.mjtObj.mjOBJ_BODY, b) or "")
                        .startswith("obs")
                    )
                    out.append((name, None, float(mass), float(ref) / 100.0))
                else:
                    out.append((name, density, float(mass), float(mass) / density))
    return out


# The human fixture is welded to the world, so this mass never enters the dynamics -- it is
# checked for reporting consistency, not because it changes a rollout. See OBSTACLE_MASS_KG.
HUMAN_LAYOUT = "ONE_WALL_SMALL"
HUMAN_VARIANT_PREFIX = "NavigateKitchenHuman"


def weigh_human_in_env():
    """Build one navigate_safe env and weigh the posed_human fixture in it.

    Returns (mass_kg, height_m, dofnum). Raises on any failure -- the caller reports it as
    a problem rather than silently passing, since a human that cannot be weighed is exactly
    the case this check is here to notice.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from check_obstacle_stability import all_variants, build_env

    variant = next(v for v in all_variants() if v.startswith(HUMAN_VARIANT_PREFIX))
    env = build_env(variant, HUMAN_LAYOUT, "MODERN_1", 0, 0)
    try:
        env.reset()
        model, data = env.sim.model._model, env.sim.data._data
        bid = model.body("posed_human_main_group_main").id
        mass = float(model.body_mass[bid])
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for g in range(model.ngeom):
            bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                   model.geom_bodyid[g]) or ""
            if not bn.startswith("posed_human"):
                continue
            if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
                continue
            rot, ctr = data.geom_xmat[g].reshape(3, 3), data.geom_xpos[g]
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[g]
                a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                pts = model.mesh_vert[a:a + n] @ rot.T + ctr
            else:
                sz = model.geom_size[g]
                pts = np.array([[x, y, z] for x in (-sz[0], sz[0])
                                for y in (-sz[1], sz[1])
                                for z in (-sz[2], sz[2])]) @ rot.T + ctr
            lo, hi = np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))
        return mass, float(hi[2] - lo[2]), int(model.body_dofnum[bid])
    finally:
        try:
            env.close()
        except Exception:
            pass


def check(tol=0.02, quiet=False, skip_human=False):
    problems = []
    tier_means = {t: [] for t in TIER_ORDER}

    for tier in TIER_ORDER:
        rows = []
        for obstacle in TIER_TO_OBSTACLES[tier]:
            if obstacle == "human":
                continue  # fixture, not an object -- weighed in a built env below
            target = OBSTACLE_MASS_KG[obstacle]
            results = weigh(obstacle)
            masses = np.array([r[2] for r in results])
            tier_means[tier].append(masses.mean())
            for name, density, mass, vol in results:
                err = mass / target - 1.0
                if abs(err) > tol:
                    have = ("no density declared — inherits the asset's exported value"
                            if density is None else f"currently {density:.1f}")
                    problems.append(
                        (obstacle, name,
                         f"{mass:.3f} kg vs target {target:.2f} kg "
                         f"({err:+.1%}); set density={target / vol:.1f} ({have})")
                    )
            rows.append((obstacle, target, masses, len(results)))

        if not quiet:
            print(f"\n{tier} tier")
            for obstacle, target, masses, n in rows:
                spread = (f"  [{masses.min():.3f}, {masses.max():.3f}] over {n} models"
                          if n > 1 else "")
                print(f"  {obstacle:15s} target {target:6.2f} kg   "
                      f"realized {masses.mean():7.3f} kg{spread}")

    if skip_human:
        print("\nSKIPPED human — the posed_human fixture was not weighed "
              "(--skip-human). The roster's heaviest obstacle is unchecked.")
    else:
        target = OBSTACLE_MASS_KG["human"]
        try:
            mass, height, dofnum = weigh_human_in_env()
        except Exception as exc:
            problems.append(("human", "posed_human fixture",
                             f"could not be weighed in {HUMAN_LAYOUT}: "
                             f"{type(exc).__name__}: {exc}"))
        else:
            tier_means["High"].append(mass)
            err = mass / target - 1.0
            if abs(err) > tol:
                problems.append(
                    ("human", "posed_human fixture",
                     f"{mass:.3f} kg vs target {target:.2f} kg ({err:+.1%}); scale the "
                     f"density in objects/lrs_objs/human/rp_posedplus/model.xml by "
                     f"{target / mass:.4f}")
                )
            if not quiet:
                note = "welded, mass inert" if dofnum == 0 else f"dofnum={dofnum}"
                print(f"\nHigh tier (fixture, weighed in {HUMAN_LAYOUT})")
                print(f"  {'human':15s} target {target:6.2f} kg   "
                      f"realized {mass:7.3f} kg   height {height:.2f} m   [{note}]")

    if not quiet:
        print("\nTier mean of per-obstacle masses:")
        for tier in TIER_ORDER:
            # Flagged rather than silently averaged over 5: dropping the 70 kg human pulls
            # the High mean from 22.5 to 13.0, which would read as a real tier figure.
            note = ("   (over 5 of 6 — human skipped)"
                    if tier == "High" and skip_human else "")
            print(f"  {tier:7s} {np.mean(tier_means[tier]):7.3f} kg{note}")

    if problems:
        print(f"\nFAIL — {len(problems)} obstacle model(s) off target "
              f"(tolerance {tol:.0%}):")
        for obstacle, name, msg in problems:
            print(f"  {obstacle} [{name}]: {msg}")
        print("\nIf a collision proxy was rebuilt, paste the suggested densities into "
              "OBJ_CATEGORIES in robocasa/models/objects/kitchen_objects.py.")
        return 1

    n_checked = len(OBSTACLE_MASS_KG) - (1 if skip_human else 0)
    print(f"\nOK — {n_checked}/{len(OBSTACLE_MASS_KG)} obstacles within {tol:.0%} of "
          f"their OBSTACLE_MASS_KG target.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tol", type=float, default=0.02,
                    help="relative mass tolerance (default 0.02)")
    ap.add_argument("--quiet", action="store_true",
                    help="print only problems (for CI)")
    ap.add_argument("--skip-human", action="store_true",
                    help="skip the posed_human fixture check, which builds one env "
                         "(~40 s) and needs a render device")
    args = ap.parse_args()
    sys.exit(check(tol=args.tol, quiet=args.quiet, skip_human=args.skip_human))


if __name__ == "__main__":
    main()
