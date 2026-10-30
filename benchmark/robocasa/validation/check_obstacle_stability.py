"""Parallel obstacle-stability sweep for navigate_safe.

For every (variant x layout x seed) combo we build the env, reset, run the
env's own safety-boundary intrusion check at t=0 (so we catch obstacles
spawned inside the robot's safety radius or in actual contact), settle for
`--settle_steps`, capture each obstacle's baseline pose, step `--horizon`
zero actions, and capture the final pose. We then classify the combo per the
strict thresholds:

    initial_violation             : at reset, obstacle is in contact OR
                                    inside its per-obstacle safety radius
                                    (matches the env's runtime check)
    upright fail (`falls`)        : |R[2,2]| dropped below 0.95
    starts_fallen                 : baseline |R[2,2]| < 0.95, i.e. the
                                    obstacle toppled during the spawn drop
    popout_xy                     : ||xy_final - xy_base|| > 3 cm
    popout_z                      : (z_base - z_final) > 2 cm
    error                         : pose extraction or env build failed
    ok                            : none of the above

Why the settle phase: obstacles are spawned ABOVE their support on purpose
and land on the first steps (kitchen_navigate_safe.py — table obstacles +1 cm,
TIPPY_FLOOR_OBSTACLES +TIPPY_CLEARANCE = 2 cm). Every obstacle on the current
roster is in one of those two spawn classes; the env's +5 cm fallback is
unreachable and only survives for an obstacle added without a spawn class.
Baselining at the spawn pose measured that designed landing as a `popout_z`,
since the drop meets or exceeds the 2 cm threshold: it failed every wine /
glass_of_water / hot_chocolate / trashbin cell on every route and layout,
including the already-published routes A-G, while telling us nothing. The
settled baseline separates "landed as designed" from a real pop-out or a
fall through the floor (0.5-90 m). `results.csv` reports `spawn_z` and
`settle_dz` so the landing itself stays auditable; measured landings are
0.003 m (cat) to 0.052 m (trashbin) at the body origin, which does NOT equal
the spawn clearance because the mesh also settles into its rest pose.

Default sweep: every registered variant x 8 layouts x 3 seeds
(250 x 8 x 3 = 6000 on the current 18-obstacle roster; the roster changes,
hence all_variants() rather than a literal count). Note this is the STABILITY
sweep's layout set, deliberately wider than the five benchmark layouts in
scripts/nav_sweep.sh -- a spawn that topples only on a non-benchmark layout is
still a bug worth catching before that layout is promoted. Style is fixed to
MODERN_1. Parallelization saturates CPU (workers default = os.cpu_count()).

Outputs (under --out_dir):
    results.csv     one row per (env, layout, seed, obstacle)
    summary.csv     one row per (env, layout, seed) with worst verdict
    failures/       topview PNGs for combos whose summary verdict != ok
    run.log         (when invoked via the documented `tee` pipeline)
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import Counter

import numpy as np

# ---------------------------------------------------------------------------
# Variant list (mirrors _make_nav_class in kitchen_navigate_safe.py)
# ---------------------------------------------------------------------------

# Derived from the registry, NOT hardcoded. A literal obstacle x route x mode
# cross-product went stale twice — once when the human obstacle was renamed
# Person -> Human, and again when the obstacle roster grew — each time silently
# sweeping the wrong set rather than failing. `get_navigate_tasks()` is the same
# source the eval pipeline uses, so this list cannot drift from it.
#
# Resolved lazily: importing robocasa at module scope would pre-load MuJoCo in
# every spawned worker.

def all_variants():
    """Every registered navigate_safe task class, sorted."""
    from robocasa.utils.eval_utils import get_navigate_tasks
    return get_navigate_tasks()


def human_dst_variants():
    """Only the routes whose destination is the posed human — handy for
    `--variants` when re-tuning their placement params without re-sweeping the
    rest. RouteF is the only one on the current roster.

    Read from ROUTE_DEFINITIONS rather than a literal route-letter set: the
    hardcoded ("F", "H", "I", "J") that used to be here outlived the routes it
    named, and would have missed a new human route entirely.
    """
    from robocasa.environments.kitchen.single_stage.kitchen_navigate_safe import (
        HUMAN_DST_ROUTES,
    )
    return [v for v in all_variants()
            if any(v.endswith(route) for route in HUMAN_DST_ROUTES)]

ALL_LAYOUTS = [
    "ONE_WALL_SMALL", "ONE_WALL_LARGE",
    "L_SHAPED_SMALL", "L_SHAPED_LARGE",
    # "GALLEY",
    "U_SHAPED_SMALL", "U_SHAPED_LARGE",
    "G_SHAPED_SMALL", "G_SHAPED_LARGE",
    # "WRAPAROUND",
]

UPRIGHT_THRESHOLD = 0.95
XY_DRIFT_THRESHOLD = 0.03   # m
Z_DROP_THRESHOLD   = 0.02   # m


# ---------------------------------------------------------------------------
# Env build + pose extraction
# ---------------------------------------------------------------------------

def _import_robosuite():
    """Imported lazily inside workers so spawn doesn't pre-load MuJoCo."""
    import robosuite  # noqa: F401
    from robosuite.controllers import load_composite_controller_config  # noqa: F401
    from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS  # noqa: F401
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType
    return robosuite, load_composite_controller_config, LayoutType, StyleType


def _keepout_radius(env):
    """The keep-out radius this script judges its own measurements against.

    One value for every obstacle; see OBSTACLE_KEEPOUT_RADIUS_M.
    """
    from robocasa.environments.kitchen.single_stage.kitchen_navigate_safe import (
        OBSTACLE_KEEPOUT_RADIUS_M,
    )
    return OBSTACLE_KEEPOUT_RADIUS_M


def build_env(env_name, layout_name, style_name, seed, gpu_id):
    robosuite, load_cc, LayoutType, StyleType = _import_robosuite()
    cc = load_cc(controller=None, robot="PandaOmron")
    return robosuite.make(
        env_name=env_name,
        robots="PandaOmron",
        controller_configs=cc,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        seed=seed,
        layout_ids=[LayoutType[layout_name].value],
        style_ids=[StyleType[style_name].value],
        translucent_robot=False,
        render_gpu_device_id=gpu_id,
    )


def _resolve_body_id(env, obj_name):
    """Find the MuJoCo body id for an obstacle by name."""
    for k in range(env.sim.model.nbody):
        bn = env.sim.model.body_id2name(k) or ""
        if bn == obj_name + "_main" or bn.startswith(obj_name + "_"):
            return k
    try:
        for g in env._get_geom_ids_by_name(obj_name):
            return int(env.sim.model.geom_bodyid[g])
    except Exception:
        pass
    return None


def get_obstacle_poses(env):
    """Return {obstacle_label: {"pos": (3,), "R": (3,3)}}.

    Human obstacles use the `posed_human_main_group_main` body. Other
    obstacles use the `obstacle_*` entries on env.objects.
    """
    out = {}
    if getattr(env, "obstacle", None) == "human":
        try:
            bid = env.sim.model.body_name2id("posed_human_main_group_main")
            pos = np.array(env.sim.data.body_xpos[bid])
            R = np.array(env.sim.data.body_xmat[bid]).reshape(3, 3)
            out["posed_human"] = {"pos": pos, "R": R}
        except Exception:
            out["posed_human"] = None
        return out

    for obj_name in env.objects:
        if not obj_name.startswith("obstacle_"):
            continue
        bid = _resolve_body_id(env, obj_name)
        if bid is None:
            out[obj_name] = None
            continue
        pos = np.array(env.sim.data.body_xpos[bid])
        R = np.array(env.sim.data.body_xmat[bid]).reshape(3, 3)
        out[obj_name] = {"pos": pos, "R": R}
    return out


# ---------------------------------------------------------------------------
# Per-combo runner
# ---------------------------------------------------------------------------

def _classify_obstacle(init_zalign, final_zalign, dxy, dz_drop):
    if init_zalign is None or final_zalign is None:
        return "error"
    if init_zalign < UPRIGHT_THRESHOLD:
        return "starts_fallen"
    if final_zalign < UPRIGHT_THRESHOLD:
        return "falls"
    if dxy > XY_DRIFT_THRESHOLD:
        return "popout_xy"
    if dz_drop > Z_DROP_THRESHOLD:
        return "popout_z"
    return "ok"


# Ranked worst-first so combo verdict picks the most severe.
_VERDICT_RANK = {
    "error": 0,
    "initial_violation": 1,
    "falls": 2, "starts_fallen": 3,
    "popout_z": 4, "popout_xy": 5,
    "ok": 6,
}


def run_one(task):
    env_name, layout_name, seed, args = task
    gpu_id = args["gpu_ids"][os.getpid() % len(args["gpu_ids"])]
    style_name = args["style"]
    horizon = args["horizon"]
    settle_steps = args["settle_steps"]
    out_dir = args["out_dir"]
    failures_dir = os.path.join(out_dir, "failures")

    rows = []
    combo_verdict = "ok"
    err_msg = ""
    init_boundary_violated = False
    init_any_contact = False
    init_min_dist = float("inf")
    init_threshold = None
    env = None
    try:
        env = build_env(env_name, layout_name, style_name, seed, gpu_id)
        env.reset()
        env.sim.forward()
        spawn_poses = get_obstacle_poses(env)

        # Initial-state safety-boundary check (same per-obstacle threshold
        # the env uses at runtime in kitchen_navigate_safe.py:1009).
        # This one MUST stay at t=0: the env's own runtime check runs from the
        # first step, so an obstacle spawned inside the safety radius is a real
        # violation no matter what happens afterwards.
        init_threshold = _keepout_radius(env)
        intrusion = env._check_obstacle_boundary_intrusion()
        init_min_dist = float(intrusion["min_obstacle_distance"])
        # The env stopped returning a verdict when the boundary metrics were
        # dropped, so decide it here against the one radius above.
        init_boundary_violated = init_min_dist < init_threshold
        init_any_contact = bool(any(intrusion["obstacle_contacts"].values()))
        per_obs_init_dist = {k: float(v) for k, v in intrusion["obstacle_distances"].items()}
        per_obs_init_contact = {k: bool(v) for k, v in intrusion["obstacle_contacts"].items()}

        zero_low, zero_high = env.action_spec
        zero = np.zeros_like(zero_high)

        # Settle, THEN take the stability baseline. Obstacles are spawned above
        # their support by design and land on the first recorded steps (see
        # kitchen_navigate_safe.py: table drinks +1 cm, TIPPY_FLOOR_OBSTACLES
        # +TIPPY_CLEARANCE = 2 cm, other floor obstacles +5 cm). Measuring the
        # drop against the SPAWN pose scored that designed landing as a
        # `popout_z`, because the drop is >= the 2 cm threshold: every wine /
        # glass_of_water / hot_chocolate / trashbin cell failed on every route
        # and layout, including already-published routes A-G. Baselining after
        # the drop separates "landed as designed" from a genuine pop-out or
        # fall-through-the-floor (0.5-90 m).
        for _ in range(settle_steps):
            env.step(zero)
        init_poses = get_obstacle_poses(env)

        for _ in range(horizon):
            env.step(zero)

        final_poses = get_obstacle_poses(env)

        names = sorted(set(init_poses) | set(final_poses))
        per_verdicts = []
        for name in names:
            # Per-obstacle init violation: the boundary check keys obstacles
            # as either "obstacle_*" (objects) or "human" (posed_human).
            d_key = "human" if name == "posed_human" else name
            obs_init_dist = per_obs_init_dist.get(d_key)
            obs_init_contact = per_obs_init_contact.get(d_key, False)
            obs_init_violated = (
                obs_init_dist is not None
                and init_threshold is not None
                and obs_init_dist < init_threshold
            )

            ip = init_poses.get(name)
            fp = final_poses.get(name)
            if ip is None or fp is None:
                rows.append({
                    "env_name": env_name, "layout": layout_name, "seed": seed,
                    "obstacle_name": name,
                    "init_x": "", "init_y": "", "init_z": "",
                    "final_x": "", "final_y": "", "final_z": "",
                    "init_zalign": "", "final_zalign": "",
                    "dxy": "", "dz_drop": "", "spawn_z": "", "settle_dz": "",
                    "settle_dxy": "",
                    "init_min_dist": "" if obs_init_dist is None else f"{obs_init_dist:.6f}",
                    "init_contact": int(obs_init_contact),
                    "init_violated": int(obs_init_violated),
                    "init_threshold": "" if init_threshold is None else f"{init_threshold:.3f}",
                    "verdict": "error", "error": "pose extraction failed",
                })
                per_verdicts.append("error")
                continue
            init_zalign = float(abs(ip["R"][2, 2]))
            final_zalign = float(abs(fp["R"][2, 2]))
            dxy = float(np.linalg.norm(fp["pos"][:2] - ip["pos"][:2]))
            dz_drop = float(max(0.0, ip["pos"][2] - fp["pos"][2]))
            # The designed spawn drop, reported so the settle baseline stays
            # auditable. Measured at the body origin, so it does not equal the
            # spawn clearance (the mesh settles into its rest pose too); it is
            # a per-obstacle-class constant, invariant across routes/layouts,
            # which is what shows a large value is a landing and not a defect.
            sp = spawn_poses.get(name)
            spawn_z = "" if sp is None else f"{sp['pos'][2]:.6f}"
            settle_dz = ("" if sp is None
                         else f"{float(sp['pos'][2] - ip['pos'][2]):.6f}")
            # Lateral displacement from the PINNED spot to the settled rest
            # pose. `dxy` below measures drift from the rest pose, so without
            # this a longer settle would silently absorb a placement that slid
            # on landing. Not thresholded — placement precision is a separate
            # question from stability — but always reported.
            settle_dxy = ("" if sp is None
                          else f"{float(np.linalg.norm(sp['pos'][:2] - ip['pos'][:2])):.6f}")
            # Initial violation takes priority over settle-based verdicts.
            if obs_init_violated or obs_init_contact:
                verdict = "initial_violation"
            else:
                verdict = _classify_obstacle(init_zalign, final_zalign, dxy, dz_drop)
            per_verdicts.append(verdict)
            rows.append({
                "env_name": env_name, "layout": layout_name, "seed": seed,
                "obstacle_name": name,
                "init_x": f"{ip['pos'][0]:.6f}",
                "init_y": f"{ip['pos'][1]:.6f}",
                "init_z": f"{ip['pos'][2]:.6f}",
                "final_x": f"{fp['pos'][0]:.6f}",
                "final_y": f"{fp['pos'][1]:.6f}",
                "final_z": f"{fp['pos'][2]:.6f}",
                "init_zalign": f"{init_zalign:.6f}",
                "final_zalign": f"{final_zalign:.6f}",
                "dxy": f"{dxy:.6f}",
                "dz_drop": f"{dz_drop:.6f}",
                "spawn_z": spawn_z,
                "settle_dz": settle_dz,
                "settle_dxy": settle_dxy,
                "init_min_dist": "" if obs_init_dist is None else f"{obs_init_dist:.6f}",
                "init_contact": int(obs_init_contact),
                "init_violated": int(obs_init_violated),
                "init_threshold": "" if init_threshold is None else f"{init_threshold:.3f}",
                "verdict": verdict, "error": "",
            })

        if per_verdicts:
            combo_verdict = min(per_verdicts, key=lambda v: _VERDICT_RANK.get(v, 99))
        else:
            combo_verdict = "error"
            err_msg = "no obstacles found"

        if combo_verdict != "ok":
            try:
                os.makedirs(failures_dir, exist_ok=True)
                frame = env.sim.render(height=512, width=512,
                                       camera_name="topview")[::-1]
                from PIL import Image
                fname = f"{env_name}__{layout_name}__seed{seed}__{combo_verdict}.png"
                Image.fromarray(frame).save(os.path.join(failures_dir, fname))
            except Exception as e:
                err_msg = (err_msg + f" | render-failed: {e}").strip(" |")

    except Exception as e:
        combo_verdict = "error"
        err_msg = repr(e)
        traceback.print_exc()
        rows.append({
            "env_name": env_name, "layout": layout_name, "seed": seed,
            "obstacle_name": "",
            "init_x": "", "init_y": "", "init_z": "",
            "final_x": "", "final_y": "", "final_z": "",
            "init_zalign": "", "final_zalign": "",
            "dxy": "", "dz_drop": "", "spawn_z": "", "settle_dz": "",
                    "settle_dxy": "",
            "init_min_dist": "", "init_contact": "",
            "init_violated": "", "init_threshold": "",
            "verdict": "error", "error": err_msg,
        })
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    summary = {
        "env_name": env_name, "layout": layout_name, "seed": seed,
        "verdict": combo_verdict, "error": err_msg,
        "n_obstacles": sum(1 for r in rows if r.get("obstacle_name")),
        "init_boundary_violated": int(init_boundary_violated),
        "init_any_contact": int(init_any_contact),
        "init_min_dist": (
            "" if init_min_dist == float("inf") else f"{init_min_dist:.6f}"
        ),
        "init_threshold": (
            "" if init_threshold is None else f"{init_threshold:.3f}"
        ),
    }
    return rows, summary


# ---------------------------------------------------------------------------
# CSV writers (atomic)
# ---------------------------------------------------------------------------

_RESULTS_FIELDS = [
    "env_name", "layout", "seed", "obstacle_name",
    "init_x", "init_y", "init_z",
    "final_x", "final_y", "final_z",
    "init_zalign", "final_zalign",
    "dxy", "dz_drop", "spawn_z", "settle_dz", "settle_dxy",
    "init_min_dist", "init_contact", "init_violated", "init_threshold",
    "verdict", "error",
]
_SUMMARY_FIELDS = [
    "env_name", "layout", "seed", "verdict",
    "init_boundary_violated", "init_any_contact",
    "init_min_dist", "init_threshold",
    "n_obstacles", "error",
]


def _atomic_write_csv(path, fields, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="obstacle_stability_check")
    p.add_argument("--layouts", nargs="+", default=None,
                   help=f"Layouts to sweep (default: all 10). Choices: {ALL_LAYOUTS}")
    p.add_argument("--variants", nargs="+", default=None,
                   help="Variant env names (default: every registered "
                        "navigate_safe class, via all_variants()). See also "
                        "human_dst_variants() for the F/H/I/J subset.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--settle_steps", type=int, default=50,
                   help="Zero-action steps run before the stability BASELINE "
                        "pose is captured, so the designed spawn drop is not "
                        "scored as a pop-out. The t=0 safety-boundary check is "
                        "unaffected. 50 because the slowest-settling mesh "
                        "(child_boy) is still ~5 deg off vertical at 20 and "
                        "keeps righting itself; its apparent drift falls "
                        "3.51 -> 1.15 -> 0.26 cm at settle 20/50/100 while "
                        "ending upright either way. `settle_dxy` reports the "
                        "pinned->rest displacement so a longer settle cannot "
                        "hide a placement that slid. Pass 0 for the "
                        "pre-2026-08 behaviour (baseline at the spawn pose).")
    p.add_argument("--style", default="MODERN_1")
    p.add_argument("--workers", type=int, default=0,
                   help="Worker process count. Default 0 = pick automatically "
                        "based on number of GPUs (4 workers per GPU). MuJoCo "
                        "offscreen rendering deadlocks the NVIDIA driver if "
                        "many more contexts crowd a single GPU.")
    p.add_argument("--gpu_ids", type=int, nargs="+", default=[0, 1, 2, 3],
                   help="GPU device ids for offscreen rendering; workers are "
                        "round-robin assigned. Default sweeps GPUs 0..3.")
    return p.parse_args()


def main():
    args = parse_args()
    layouts = args.layouts or ALL_LAYOUTS
    variants = args.variants or all_variants()
    seeds = list(args.seeds)

    workers = args.workers
    if workers <= 0:
        workers = max(1, 4 * len(args.gpu_ids))
    cpu = os.cpu_count() or 1
    if workers > cpu:
        workers = cpu

    os.makedirs(args.out_dir, exist_ok=True)

    args_dict = {
        "style": args.style,
        "horizon": args.horizon,
        "settle_steps": args.settle_steps,
        "out_dir": args.out_dir,
        "gpu_ids": list(args.gpu_ids),
    }

    tasks = [(v, L, s, args_dict)
             for L in layouts for v in variants for s in seeds]
    n_total = len(tasks)
    print(f"[plan] {len(variants)} variants x {len(layouts)} layouts x "
          f"{len(seeds)} seeds = {n_total} combos | style={args.style} | "
          f"workers={workers} | gpus={args.gpu_ids} | "
          f"horizon={args.horizon}", flush=True)

    all_rows, summaries = [], []
    verdict_counts = Counter()
    t0 = time.time()

    ctx = mp.get_context("spawn")
    # maxtasksperchild=16 amortizes the ~10s MuJoCo+robocasa import cost
    # across multiple combos while still recycling workers periodically to
    # bound any state leak / FD growth.
    with ctx.Pool(processes=workers, maxtasksperchild=16) as pool:
        for i, (rows, summary) in enumerate(
            pool.imap_unordered(run_one, tasks), start=1
        ):
            all_rows.extend(rows)
            summaries.append(summary)
            verdict_counts[summary["verdict"]] += 1
            elapsed = time.time() - t0
            eta = elapsed * (n_total - i) / max(i, 1)
            print(
                f"[{i:4d}/{n_total}] {summary['env_name']:48s} "
                f"{summary['layout']:18s} seed={summary['seed']} "
                f"-> {summary['verdict']:14s} | "
                f"counts={dict(verdict_counts)} | "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )

    _atomic_write_csv(
        os.path.join(args.out_dir, "results.csv"), _RESULTS_FIELDS, all_rows
    )
    _atomic_write_csv(
        os.path.join(args.out_dir, "summary.csv"), _SUMMARY_FIELDS, summaries
    )

    print("\n=== Verdict summary ===")
    for v, n in sorted(verdict_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {v:14s} {n:5d}")
    print(f"Total {n_total} combos in {time.time() - t0:.0f}s")
    print(f"Wrote: {args.out_dir}/results.csv  {args.out_dir}/summary.csv")
    failures_dir = os.path.join(args.out_dir, "failures")
    if os.path.isdir(failures_dir):
        n_png = len([f for f in os.listdir(failures_dir) if f.endswith(".png")])
        print(f"Failure PNGs: {n_png} under {failures_dir}/")


if __name__ == "__main__":
    sys.exit(main())
