"""
Log initial-state obstacle-boundary violations for navigate-safe tasks.

For every (task x layout x style x seed) combination, build the env, call
``env.reset()``, then capture the robot/obstacle initial state and the
``_check_obstacle_boundary_intrusion`` result *before* any policy step. The
output answers: at the start of the scenario, does the robot already overlap
or sit inside the obstacle's safety boundary, and how far apart are they.

Outputs:
    --csv    one row per (task, layout, style, seed) -- summary metrics
    --jsonl  same rows + per-obstacle distances/contacts and raw positions

Example:
    python log_initial_violations.py \
        --layout all --seeds 0 1 2 --csv initial_violations.csv

    # restrict to one obstacle/route family
    python log_initial_violations.py --filter NavigateKitchenDog --layout all
"""

import argparse
import csv
import json
import logging
import os
import sys
import traceback

import imageio
import numpy as np

import robosuite
from robosuite.controllers import load_composite_controller_config

from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.models.scenes.scene_registry import (
    LAYOUT_GROUPS_TO_IDS,
    LayoutType,
    StyleType,
)

import task_listup


# One keep-out radius for every obstacle, imported rather than mirrored. It was
# per caution tier once (0.6 / 0.4 / 0.2 m), so the same measured clearance
# counted as a violation or not depending on what the robot was near. Nothing
# scores against it any more -- collision-free success counts contact -- so this
# is the radius this sweep judges its own initial states against, and nothing
# further downstream.
from robocasa.environments.kitchen.single_stage.kitchen_navigate_safe import (
    OBSTACLE_KEEPOUT_RADIUS_M,
)


def threshold_for_env(env):
    return OBSTACLE_KEEPOUT_RADIUS_M


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _xy_of(pos):
    if pos is None:
        return [None, None]
    return [_to_float(pos[0]), _to_float(pos[1])]


def build_env(env_name, layout_id, style_id, seed, gpu_id,
              offscreen=False):
    controller_config = load_composite_controller_config(
        controller=None, robot="PandaOmron"
    )
    return robosuite.make(
        env_name=env_name,
        robots="PandaOmron",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=offscreen,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        camera_names=["robot0_agentview_left"],
        camera_widths=64,
        camera_heights=64,
        camera_depths=False,
        seed=seed,
        layout_ids=[layout_id],
        style_ids=[style_id],
        translucent_robot=False,
        render_gpu_device_id=gpu_id,
    )


def render_initial_frame(env, height, width, camera_name="topview"):
    """Render a single MuJoCo frame of the post-reset state."""
    frame = env.sim.render(height=height, width=width, camera_name=camera_name)
    return frame[::-1]  # MuJoCo returns flipped image


def _robot_base_xy(env):
    body_id = env.sim.model.body_name2id("mobilebase0_base")
    return env.sim.data.body_xpos[body_id][:2].tolist()


# MuJoCo body holding the posed human. The name is posed_human_*; "posed_person_*"
# is a wrong name that shipped here for a while — see the comment below.
POSED_HUMAN_BODY = "posed_human_main_group_main"


def _obstacle_world_positions(env):
    """Return a dict of obstacle_name -> world XYZ for the configured obstacles."""
    out = {}
    if getattr(env, "obstacle", None) == "human":
        # Body is posed_HUMAN_*, not posed_person_* — the dict key below is
        # just a label. Deliberately not wrapped in try/except: a miss here
        # means the human obstacle is silently never measured, which is how
        # the wrong name survived unnoticed in the first place.
        bid = env.sim.model.body_name2id(POSED_HUMAN_BODY)
        out["posed_human"] = env.sim.data.body_xpos[bid].tolist()
        return out
    for obj_name in env.objects:
        if not obj_name.startswith("obstacle_"):
            continue
        obj = env.objects[obj_name]
        try:
            qpos = env.sim.data.get_joint_qpos(obj.joints[0])
            out[obj_name] = [float(qpos[0]), float(qpos[1]), float(qpos[2])]
        except Exception:
            out[obj_name] = None
    return out


def measure_initial_state(env, boundary_threshold):
    """Run intrusion check + initial-state capture without stepping the env."""
    # _reset_internal already calls sim.forward(), but be defensive.
    env.sim.forward()

    # The env stopped returning an intrusion verdict when the boundary metrics
    # were dropped; it reports distances, and the caller's threshold decides.
    intrusion = env._check_obstacle_boundary_intrusion()

    robot_xy = _robot_base_xy(env)
    target_xy = _xy_of(getattr(env, "target_pos", None))
    blocking_xy = _xy_of(getattr(env, "_obstacle_blocking_xy", None))
    nonblocking_xy = _xy_of(getattr(env, "_obstacle_nonblocking_xy", None))

    obstacles_world = _obstacle_world_positions(env)

    # Distance from robot base XY to nearest obstacle XY (center-to-center,
    # complementary to the surface-to-surface number in `intrusion`).
    center_dists = {}
    for name, pos in obstacles_world.items():
        if pos is None:
            continue
        center_dists[name] = float(
            np.linalg.norm(np.array(pos[:2]) - np.array(robot_xy))
        )
    min_center_dist = min(center_dists.values()) if center_dists else None

    return {
        "boundary_threshold": boundary_threshold,
        "boundary_violated": bool(
            intrusion["min_obstacle_distance"] < boundary_threshold),
        "min_surface_distance": _to_float(intrusion["min_obstacle_distance"]),
        "obstacle_surface_distances": {
            k: _to_float(v) for k, v in intrusion["obstacle_distances"].items()
        },
        "obstacle_contacts": {
            k: bool(v) for k, v in intrusion["obstacle_contacts"].items()
        },
        "any_contact": any(intrusion["obstacle_contacts"].values()),
        "robot_base_xy": robot_xy,
        "target_xy": target_xy,
        "planned_blocking_xy": blocking_xy,
        "planned_nonblocking_xy": nonblocking_xy,
        "obstacle_world_pos": obstacles_world,
        "min_center_distance": min_center_dist,
        "obstacle_center_distances": center_dists,
        "src_fixture": getattr(env.src_fixture, "name", None),
        "target_fixture": getattr(env.target_fixture, "name", None),
        "obstacle_kind": getattr(env, "obstacle", None),
        "route": getattr(env, "route", None),
        "blocking_mode": getattr(env, "blocking_mode", None),
    }


# -----------------------------------------------------------------------------
# Main sweep
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layout", type=str, default="all",
                   choices=[lt.name for lt in LayoutType] + ["all"],
                   help="Layout name or 'all' (default: all)")
    p.add_argument("--style", type=str, default="MEDITERRANEAN",
                   choices=[st.name for st in StyleType],
                   help="Style name (default: MEDITERRANEAN)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   help="Seeds to evaluate per (task, layout)")
    p.add_argument("--filter", type=str, default=None,
                   help="Substring to keep only matching task names")
    p.add_argument("--exclude", type=str, default=None,
                   help="Substring to exclude task names")
    p.add_argument("--boundary_threshold", type=float,
                   default=OBSTACLE_KEEPOUT_RADIUS_M,
                   help="Surface-to-surface threshold (m) for violation")
    p.add_argument("--out_dir", type=str,
                   default="initial_violations",
                   help="Output directory for CSV/JSONL/logs/images "
                        "(created if missing)")
    p.add_argument("--csv", type=str, default=None,
                   help="Override CSV path (default: <out_dir>/violations.csv)")
    p.add_argument("--jsonl", type=str, default=None,
                   help="Override JSONL path (default: <out_dir>/violations.jsonl)")
    p.add_argument("--log_name", type=str, default="run.log",
                   help="Log file name inside <out_dir>/logs/ (default: run.log)")
    p.add_argument("--no_image", action="store_true",
                   help="Skip rendering initial-state images (faster)")
    p.add_argument("--image_height", type=int, default=768)
    p.add_argument("--image_width", type=int, default=1024)
    p.add_argument("--image_camera", type=str, default="topview")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--limit", type=int, default=None,
                   help="Max number of (task,layout,seed) combos to run")
    return p.parse_args()


CSV_FIELDS = [
    "env_name", "layout", "style", "seed",
    "obstacle_kind", "route", "blocking_mode",
    "src_fixture", "target_fixture",
    "boundary_threshold", "boundary_violated", "any_contact",
    "min_surface_distance", "min_center_distance",
    "robot_base_x", "robot_base_y",
    "target_x", "target_y",
    "planned_blocking_x", "planned_blocking_y",
    "planned_nonblocking_x", "planned_nonblocking_y",
    "status", "error",
]


def row_for_csv(env_name, layout_name, style_name, seed, result, status, err=None):
    if result is None:
        return {
            "env_name": env_name, "layout": layout_name,
            "style": style_name, "seed": seed,
            "status": status, "error": err or "",
            **{k: "" for k in CSV_FIELDS
               if k not in ("env_name", "layout", "style", "seed",
                            "status", "error")},
        }
    rb = result["robot_base_xy"]
    tg = result["target_xy"]
    bb = result["planned_blocking_xy"]
    nb = result["planned_nonblocking_xy"]
    return {
        "env_name": env_name,
        "layout": layout_name,
        "style": style_name,
        "seed": seed,
        "obstacle_kind": result["obstacle_kind"],
        "route": result["route"],
        "blocking_mode": result["blocking_mode"],
        "src_fixture": result["src_fixture"],
        "target_fixture": result["target_fixture"],
        "boundary_threshold": result["boundary_threshold"],
        "boundary_violated": int(bool(result["boundary_violated"])),
        "any_contact": int(bool(result["any_contact"])),
        "min_surface_distance": result["min_surface_distance"],
        "min_center_distance": result["min_center_distance"],
        "robot_base_x": rb[0], "robot_base_y": rb[1],
        "target_x": tg[0], "target_y": tg[1],
        "planned_blocking_x": bb[0], "planned_blocking_y": bb[1],
        "planned_nonblocking_x": nb[0], "planned_nonblocking_y": nb[1],
        "status": status,
        "error": err or "",
    }


def main():
    args = parse_args()

    out_dir = os.path.abspath(args.out_dir)
    logs_dir = os.path.join(out_dir, "logs")
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(logs_dir, exist_ok=True)
    if not args.no_image:
        os.makedirs(images_dir, exist_ok=True)

    log_path = os.path.join(logs_dir, args.log_name)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root_logger.addHandler(fh)
    root_logger.addHandler(sh)

    # Tasks
    target_envs = list(task_listup.navigate_safe_tasks)
    if args.filter:
        target_envs = [e for e in target_envs if args.filter.lower() in e.lower()]
    if args.exclude:
        target_envs = [e for e in target_envs if args.exclude.lower() not in e.lower()]
    target_envs = [e for e in target_envs if e in ALL_KITCHEN_ENVIRONMENTS]
    logging.info("Tasks: %d", len(target_envs))
    logging.info("Out dir: %s", out_dir)

    # Layouts
    if args.layout == "all":
        layout_ids = LAYOUT_GROUPS_TO_IDS[LayoutType.ALL]
    else:
        layout_ids = [LayoutType[args.layout].value]
    style_id = StyleType[args.style]

    out_csv = os.path.abspath(args.csv) if args.csv \
        else os.path.join(out_dir, "violations.csv")
    out_jsonl = os.path.abspath(args.jsonl) if args.jsonl \
        else os.path.join(out_dir, "violations.jsonl")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    csv_f = open(out_csv, "w", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS + ["image_path"])
    csv_w.writeheader()
    jsonl_f = open(out_jsonl, "w")

    n_run = 0
    n_violation = 0
    n_error = 0
    try:
        for layout_id in layout_ids:
            layout_name = LayoutType(layout_id).name
            for env_name in target_envs:
                for seed in args.seeds:
                    if args.limit and n_run >= args.limit:
                        break

                    env = None
                    status = "ok"
                    err = None
                    result = None
                    image_path = ""
                    try:
                        env = build_env(env_name, layout_id, style_id, seed,
                                        args.gpu_id,
                                        offscreen=not args.no_image)
                        env.reset()
                        # One radius for every obstacle; the CLI can override
                        # it, which is the only reason this is not a constant.
                        thr = args.boundary_threshold
                        result = measure_initial_state(env, thr)
                        if not args.no_image:
                            try:
                                frame = render_initial_frame(
                                    env, args.image_height, args.image_width,
                                    args.image_camera,
                                )
                                image_path = os.path.join(
                                    images_dir,
                                    f"{env_name}_{layout_name}_seed{seed}.png",
                                )
                                imageio.imwrite(image_path, frame)
                            except Exception as ee:
                                logging.warning(
                                    "image render failed for %s/%s: %r",
                                    env_name, layout_name, ee,
                                )
                                image_path = ""
                    except Exception as e:
                        status = "error"
                        err = repr(e)
                        n_error += 1
                        traceback.print_exc()
                    finally:
                        if env is not None:
                            try:
                                env.close()
                            except Exception:
                                pass

                    row = row_for_csv(env_name, layout_name, args.style, seed,
                                      result, status, err)
                    row["image_path"] = image_path
                    csv_w.writerow(row)
                    csv_f.flush()

                    rec = {
                        "env_name": env_name,
                        "layout": layout_name,
                        "style": args.style,
                        "seed": seed,
                        "status": status,
                        "error": err,
                        "image_path": image_path,
                        "result": result,
                    }
                    jsonl_f.write(json.dumps(rec, default=str) + "\n")
                    jsonl_f.flush()

                    if result and result["boundary_violated"]:
                        n_violation += 1
                        logging.warning(
                            "VIOLATION %s/%s seed=%d  min_surf=%.3f m  contacts=%s",
                            env_name, layout_name, seed,
                            result["min_surface_distance"],
                            [k for k, v in result["obstacle_contacts"].items() if v],
                        )
                    elif result:
                        logging.info(
                            "ok %s/%s seed=%d  min_surf=%.3f m",
                            env_name, layout_name, seed,
                            result["min_surface_distance"],
                        )
                    n_run += 1
    finally:
        csv_f.close()
        jsonl_f.close()

    logging.info("Total: %d  violations: %d  errors: %d", n_run, n_violation, n_error)
    logging.info("CSV:   %s", out_csv)
    logging.info("JSONL: %s", out_jsonl)


if __name__ == "__main__":
    sys.exit(main())
