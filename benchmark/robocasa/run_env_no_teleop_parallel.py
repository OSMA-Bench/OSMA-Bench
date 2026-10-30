"""
Run environments without teleoperation in parallel (offscreen recording).
Refined to follow run_env_no_teleop.py behavior while keeping multiprocessing.
"""

import os
import csv
import json
import numpy as np
import imageio
import argparse
from multiprocessing import Pool, cpu_count
from termcolor import colored
from tqdm import tqdm
import logging

import robosuite
from robosuite.controllers import load_composite_controller_config

from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.utils.env_utils import hide_gripper_marker_sites
from robocasa.models.scenes.scene_registry import LayoutType, StyleType, LAYOUT_GROUPS_TO_IDS
import task_listup
from robocasa.utils.egl_device import resolve_egl_devices


ENV_CATEGORIES = {
    "handover": task_listup.handover_tasks,
    "navigate_safe": task_listup.navigate_safe_tasks,
    "move_hot_object": task_listup.move_hot_object_to_table_tasks,
    "open_door_safe": ["OpenDoorSafe"],
    "close_door_safe": ["CloseDoorSafeCenter", "CloseDoorSafeThreshold", "CloseDoorSafeEdge"],
}

def create_env_offscreen(
    env_name,
    robots="PandaOmron",
    camera_names=[
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
        "robot0_frontview",
    ],
    camera_widths=128,
    camera_heights=128,
    seed=None,
    render_camera="topview",
    layout_ids=None,
    style_ids=None,
    has_human=True,
    gpu_id=0,
):
    """Create environment in offscreen mode only."""
    controller_config = load_composite_controller_config(
        controller=None,
        robot=robots if isinstance(robots, str) else robots[0],
    )

    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera=render_camera,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        camera_depths=False,
        seed=seed,
        layout_ids=layout_ids,
        style_ids=style_ids,
        translucent_robot=False,
        render_gpu_device_id=gpu_id,
    )
    if "Door" in env_name and "NoHuman" not in env_name:
        env_kwargs['has_human'] = has_human
    env = robosuite.make(**env_kwargs)
    hide_gripper_marker_sites(env)
    return env


def _unwrap_env(env):
    return env.env if hasattr(env, "env") else env


def _snapshot_obstacles(env):
    """Snapshot every obstacle's free-joint pose+velocity (post-reset anchored pose)."""
    base = _unwrap_env(env)
    if getattr(base, "obstacle", None) == "human":
        return {}
    snap = {}
    objects = getattr(base, "objects", None) or {}
    for name, obj in objects.items():
        if not name.startswith("obstacle_"):
            continue
        try:
            jname = obj.joints[0]
            qpos = base.sim.data.get_joint_qpos(jname).copy()
            s, e = base.sim.model.get_joint_qvel_addr(jname)
            qvel = base.sim.data.qvel[s:e].copy()
        except Exception:
            continue
        snap[name] = {
            "xy": np.asarray(qpos[:2], dtype=float),
            "z": float(qpos[2]),
            "quat": np.asarray(qpos[3:7], dtype=float),
            "vlin": np.asarray(qvel[:3], dtype=float),
            "vang": np.asarray(qvel[3:6], dtype=float),
        }
    return snap


def _diff_pop_out(ref_snap, end_snap, xy_tol=0.05, z_up_tol=0.05, z_down_tol=0.10):
    """Compare ref vs final obstacle snapshots; flag any non-trivial drift."""
    per = {}
    xy_max = 0.0
    up_max = 0.0
    dn_max = 0.0
    pop = False
    for name, ref in ref_snap.items():
        end = end_snap.get(name)
        if end is None:
            continue
        xy_drift = float(np.linalg.norm(end["xy"] - ref["xy"]))
        dz = end["z"] - ref["z"]
        z_up = float(max(0.0, dz))
        z_dn = float(max(0.0, -dz))
        flag = (xy_drift > xy_tol) or (z_up > z_up_tol) or (z_dn > z_down_tol)
        per[name] = {
            "xy_drift": xy_drift,
            "z_drift_up": z_up,
            "z_drift_down": z_dn,
            "pop_out": bool(flag),
        }
        xy_max = max(xy_max, xy_drift)
        up_max = max(up_max, z_up)
        dn_max = max(dn_max, z_dn)
        pop = pop or flag
    return {
        "pop_out": bool(pop),
        "xy_drift_max": xy_max,
        "z_drift_up_max": up_max,
        "z_drift_down_max": dn_max,
        "per_obstacle": per,
    }


def run_simulation(
    env,
    horizon=100,
    record_path=None,
    camera_name="topview",
    render_height=1024,
    render_width=1536,
    action_mode="zero",
    check_success_interval=10,
    settle_steps=60,
    capture_initial_stage=False,
):
    """
    Run simulation with random or zero actions.

    Args:
        env: robosuite environment
        horizon: number of steps to run
        record_path: path to save video (if None, no recording)
        camera_name: camera to render from
        render_height: height of rendered frames
        render_width: width of rendered frames
        action_mode: "random" for random actions, "zero" for no movement

    Returns:
        dict with simulation results
    """
    writer = imageio.get_writer(record_path, fps=20) if record_path else None

    low, high = env.action_spec
    # NOTE: do not call env.reset() here — robosuite.make() already runs
    # _load_model() + _reset_internal() once, so an extra reset would
    # re-sample obstacles/objects at different positions (RNG advances,
    # it does not rewind on reset). Matches run_env.py behavior.

    # Settle physics with zero actions before snapshotting / recording so
    # any tiny fixture-contact penetration from the initial snap-to-floor
    # gets resolved here instead of being amplified across the recorded
    # horizon (the 10m+ xy drifts in the validation report were caused by
    # an unsettled initial impulse compounding over the horizon).
    zero_action = np.zeros_like(high)
    for _ in range(settle_steps):
        env.step(zero_action)

    # Snapshot obstacle anchored pose after settling, before recording.
    ref_snap = _snapshot_obstacles(env)
    obstacle_name = getattr(_unwrap_env(env), "obstacle", None)

    if writer is not None:
        if capture_initial_stage:
            env.reset()
            hide_gripper_marker_sites(env)
        frame = env.sim.render(height=render_height, width=render_width, camera_name=camera_name)[::-1]
        writer.append_data(frame)

    success = False
    success_step = None

    for t in range(horizon):
        if action_mode == "random":
            action = np.random.uniform(low, high)
        else:
            action = np.zeros_like(high)

        obs, reward, done, info = env.step(action)

        if writer is not None:
            frame = env.sim.render(height=render_height, width=render_width, camera_name=camera_name)[::-1]

            if t == 10 and capture_initial_stage:
                print(colored(f"Captured initial stage frame for {obstacle_name} at step {t}", "green"))
                imageio.imwrite(os.path.join(os.path.dirname(record_path), f"initial_{os.path.basename(record_path).replace('.mp4', '.png')}"), frame)
            writer.append_data(frame)

        if (t % check_success_interval) == 0:
            try:
                if env._check_success() and not success:
                    success = True
                    success_step = t
            except Exception:
                pass

    # Post-rollout snapshot for pop-out diff
    end_snap = _snapshot_obstacles(env)
    pop_report = _diff_pop_out(ref_snap, end_snap)

    if writer is not None:
        writer.close()
        print(colored(f"Video saved: {record_path}", "green"))

    return {
        "success": success,
        "success_step": success_step,
        "total_steps": horizon,
        "obstacle": obstacle_name,
        "pop_report": pop_report,
    }


def run_single_task(task_config):
    """
    Worker function to run a single simulation task.

    Args:
        task_config: dict containing all parameters for the task

    Returns:
        dict with task results
    """
    env_name = task_config["env_name"]
    layout_id = task_config["layout_id"]
    style_id = task_config["style_id"]
    record_path = task_config["record_path"]
    horizon = task_config["horizon"]
    seed = task_config["seed"]
    has_human = task_config["has_human"]
    gpu_id = task_config.get("gpu_id", 0)
    action_mode = task_config.get("action_mode", "zero")
    settle_steps = task_config.get("settle_steps", 60)
    camera_view = task_config.get("camera_view", "topview")
    capture_initial_stage = task_config.get("capture_initial_stage", False)

    try:
        env = create_env_offscreen(
            env_name=env_name,
            seed=seed,
            layout_ids=[layout_id],
            style_ids=[style_id],
            camera_names=[camera_view],
            has_human=has_human,
            gpu_id=gpu_id,
        )

        print(f"Starting simulation for {env_name} on layout {LayoutType(layout_id).name} with camera view {camera_view} capture initial stage {capture_initial_stage} (GPU {gpu_id})")

        sim_result = run_simulation(
            env,
            horizon=horizon,
            record_path=record_path,
            camera_name=camera_view,
            action_mode=action_mode,
            settle_steps=settle_steps,
            capture_initial_stage=capture_initial_stage,
        )

        env.close()

        return {
            "env_name": env_name,
            "layout_id": layout_id,
            "style_id": style_id,
            "record_path": record_path,
            "status": "success",
            **sim_result,
        }
    except Exception as e:
        return {
            "env_name": env_name,
            "layout_id": layout_id,
            "style_id": style_id,
            "record_path": record_path,
            "status": "error",
            "error": str(e),
        }


def run_parallel_simulations(tasks, num_workers=None):
    """Run pre-built task configs in parallel."""
    if num_workers is None:
        num_workers = min(cpu_count(), 4)

    print(f"[info] Running {len(tasks)} simulations with {num_workers} workers")

    results = []
    with Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(run_single_task, tasks),
            total=len(tasks),
            desc="Running simulations"
        ):
            results.append(result)
            if result["status"] == "success":
                print(colored(f"  [done] {result['env_name']} - {LayoutType(result['layout_id']).name}", "black"))
            else:
                print(colored(f"  [error] {result['env_name']} - {result.get('error', 'unknown')}", "red"))

    return results


def determine_target_envs(args):
    if args.specific_env is not None:
        target_envs = [args.specific_env]
    elif args.env == "custom":
        if args.env_names is None:
            raise ValueError("--env_names required when --env custom")
        target_envs = args.env_names
    else:
        target_envs = ENV_CATEGORIES.get(args.env, [])
    print(f"Before filtering, {len(target_envs)} environments")
    if args.filter_env_keyword:
        print(f"[info] Filtering environments with keyword: {args.filter_env_keyword}")
        target_envs = [env for env in target_envs if args.filter_env_keyword.lower() in env.lower()]

    # Keep parity with run_env_no_teleop.py.
    target_envs = [env for env in target_envs if "coffee" not in env.lower()]

    if args.filter_out_keyword:
        print(f"[info] Filtering out environments with keyword: {args.filter_out_keyword}")
        target_envs = [env for env in target_envs if args.filter_out_keyword.lower() not in env.lower()]

    valid_envs = []
    for env_name in target_envs:
        if env_name in ALL_KITCHEN_ENVIRONMENTS:
            valid_envs.append(env_name)
        else:
            print(colored(f"[warn] Environment '{env_name}' not found in registry, skipping", "yellow"))
    return valid_envs


def determine_layout_ids(layout_arg):
    if layout_arg == "all":
        excluded = {LayoutType.WRAPAROUND.value, LayoutType.GALLEY.value}
        return sorted(set(LAYOUT_GROUPS_TO_IDS[LayoutType.ALL]) - excluded)
    if layout_arg is not None:
        return [LayoutType[layout_arg].value]
    return [LayoutType.ONE_WALL_SMALL.value, LayoutType.L_SHAPED_SMALL.value]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run environments in parallel without teleoperation")
    parser.add_argument("--env", type=str, default="move_hot_object",
                        choices=list(ENV_CATEGORIES.keys()) + ["custom"],
                        help="Environment category to run")
    parser.add_argument("--env_names", type=str, nargs="+", default=None,
                        help="Specific environment names (for custom mode)")
    parser.add_argument("--specific_env", type=str, default=None,
                        help="Specify a single specific environment name")
    parser.add_argument("--filter_env_keyword", type=str, default=None,
                        help="Keyword to filter environment names")
    parser.add_argument("--filter_out_keyword", type=str, default=None,
                        help="Keyword to filter out environment names")
    parser.add_argument("--record_path", type=str, required=True,
                        help="Directory to save recorded videos")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Number of simulation steps")
    parser.add_argument("--settle_steps", type=int, default=60,
                        help="Zero-action steps to run after make() before snapshotting/recording (0 disables settling)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of parallel workers (default: min(cpu_count, 4))")
    parser.add_argument("--layout", type=str, default=None,
                        choices=[lt.name for lt in LayoutType] + ["all"],
                        help="Specific layout to test")
    parser.add_argument("--no-human", action="store_true",
                        help="Disable human in the environment")
    parser.add_argument("--action_mode", type=str, default="zero",
                        choices=["zero", "random"],
                        help="Action mode: zero (stationary) or random")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device ID for rendering (single-GPU; use --gpu_ids for multi-GPU)")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=None,
                        help="List of GPU device IDs. Workers are round-robined across them. Overrides --gpu_id.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip recording if file already exists")
    parser.add_argument("--validation_csv", type=str, default=None,
                        help="Optional path to write per-task pop-out validation report (CSV).")
    parser.add_argument("--capture_initial_stage", action="store_true",
                        help="capture initial step")
    parser.add_argument("--camera_view", type=str, default="topview",
                        help="camera view for recording and rendering")

    args = parser.parse_args()

    env_names = determine_target_envs(args)
    if not env_names:
        print(colored("[error] No valid environments to run", "red"))
        exit(1)

    layout_ids = determine_layout_ids(args.layout)
    style_ids = [StyleType.MODERN_1.value]
    os.makedirs(args.record_path, exist_ok=True)

    print(f"[info] Environments: {env_names}")
    print(f"[info] Layouts: {[LayoutType(lid).name for lid in layout_ids]}")
    print(f"[info] Has human: {not args.no_human}")
    print(f"[info] Action mode: {args.action_mode}")

    gpu_ids = args.gpu_ids if args.gpu_ids else [args.gpu_id]
    # MuJoCo addresses GPUs by EGL enumeration order, not CUDA ordinal, so the
    # requested id can be a device that cannot create a headless context.
    gpu_ids = resolve_egl_devices(gpu_ids) or gpu_ids
    print(f"[info] GPU pool (round-robin): {gpu_ids}")

    tasks = []
    for env_name in env_names:
        for layout_id in layout_ids:
            for style_id in style_ids:
                layout_name = LayoutType(layout_id).name
                style_name = StyleType(style_id).name
                record_path = os.path.join(args.record_path, f"{env_name}_{layout_name}_{style_name}.mp4")
                if args.skip_existing and os.path.exists(record_path):
                    print(f"[info] Skipping (already exists): {record_path}")
                    continue

                log_dir = os.path.join(args.record_path, "log")
                os.makedirs(log_dir, exist_ok=True)
                logging_file = os.path.join(log_dir, f"{env_name}_{layout_name}.log")
                logging.basicConfig(level=logging.INFO, filename=logging_file, filemode="a", format="%(asctime)s - %(levelname)s - %(message)s")
                tasks.append({
                    "env_name": env_name,
                    "layout_id": layout_id,
                    "style_id": style_id,
                    "record_path": record_path,
                    "horizon": args.horizon,
                    "seed": args.seed,
                    "has_human": not args.no_human,
                    "gpu_id": gpu_ids[len(tasks) % len(gpu_ids)],
                    "action_mode": args.action_mode,
                    "settle_steps": args.settle_steps,
                    "camera_view": args.camera_view,
                    "capture_initial_stage": args.capture_initial_stage,
                })

    if not tasks:
        print("[info] No tasks to run (all skipped or filtered).")
        exit(0)

    print(f"[info] Output directory: {args.record_path}")
    results = run_parallel_simulations(tasks=tasks, num_workers=args.num_workers)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")
    task_success_count = sum(1 for r in results if r.get("success", False))
    pop_out_count = sum(
        1 for r in results
        if r.get("status") == "success" and (r.get("pop_report") or {}).get("pop_out", False)
    )
    print(f"Total tasks: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Task successes: {task_success_count}")
    print(f"Pop-out: {pop_out_count}")

    if error_count > 0:
        print("\nErrors:")
        for r in results:
            if r["status"] == "error":
                print(f"  - {r['env_name']} ({LayoutType(r['layout_id']).name}): {r.get('error', 'unknown')}")

    if pop_out_count > 0:
        print("\nPop-out tasks:")
        for r in results:
            pr = r.get("pop_report") or {}
            if not pr.get("pop_out"):
                continue
            print(
                f"  - {r['env_name']} ({LayoutType(r['layout_id']).name}) "
                f"xy={pr['xy_drift_max']:.3f} up={pr['z_drift_up_max']:.3f} "
                f"down={pr['z_drift_down_max']:.3f}"
            )
    if args.record_path is not None:
        print(f"\n[info] Videos saved to: {args.record_path}")
        args.validation_csv = os.path.join(args.record_path, "validation_report.csv")
    if args.validation_csv:
        print("")
        with open(args.validation_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "env_name", "layout", "style", "obstacle", "status",
                "pop_out", "xy_drift_max", "z_drift_up_max", "z_drift_down_max",
                "per_obstacle_json", "error",
            ])
            for r in results:
                pr = r.get("pop_report") or {}
                w.writerow([
                    r.get("env_name", ""),
                    LayoutType(r["layout_id"]).name if r.get("layout_id") is not None else "",
                    StyleType(r["style_id"]).name if r.get("style_id") is not None else "",
                    r.get("obstacle", ""),
                    r.get("status", ""),
                    int(bool(pr.get("pop_out", False))),
                    pr.get("xy_drift_max", ""),
                    pr.get("z_drift_up_max", ""),
                    pr.get("z_drift_down_max", ""),
                    json.dumps(pr.get("per_obstacle", {})),
                    r.get("error", ""),
                ])
        print(f"[info] Validation CSV: {args.validation_csv}")

    print("\nDone.")
