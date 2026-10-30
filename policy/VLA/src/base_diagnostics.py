"""Model-free PandaOmron base diagnostics for the XR-1 action seam.

This program deliberately avoids camera rendering and model inference. It tests
the part of the stack that must work even when a VLA is completely wrong:

* action[7:10] pulse sign and cross-axis response before and after base rotation
* the local v0.2 and upstream-compatible legacy controller as paired conditions
* closed-loop convergence to NavigateKitchen's exact pose with a geometric oracle
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np


from repo_paths import resolve_paths  # noqa: E402

PATHS = resolve_paths()
REPO_ROOT = PATHS.superproject_root
# Honours OSMA_BENCH_ROBOCASA_ROOT exactly like the rollout runner; activated in main().
ROBOCASA_RUN_ROOT = PATHS.robocasa_root

from run_vla_rollout import (  # noqa: E402
    BASE_CONTROLLER_TYPES,
    configure_base_controller,
    inspect_base_controller,
    write_json,
)
from reproducibility import (  # noqa: E402
    build_reproducibility_manifest,
    require_output_root,
    require_reproducible_source,
    utc_now_iso,
    utc_run_stamp,
    write_artifact_manifest,
    write_json_file,
)


def wrap_angle(value: float) -> float:
    return float(np.arctan2(np.sin(value), np.cos(value)))


def base_pose(env: Any) -> tuple[np.ndarray, float]:
    import robosuite.utils.transform_utils as transform_utils

    robot_id = env.sim.model.body_name2id("mobilebase0_base")
    position = np.asarray(env.sim.data.body_xpos[robot_id], dtype=np.float64).copy()
    rotation = np.asarray(env.sim.data.body_xmat[robot_id], dtype=np.float64).reshape(3, 3)
    yaw = float(transform_utils.mat2euler(rotation)[2])
    return position, yaw


def world_delta_to_body_action(world_delta: Any, yaw: float) -> np.ndarray:
    """Map a desired world XY displacement into the current base frame."""
    dx, dy = np.asarray(world_delta, dtype=np.float64).reshape(2)
    return np.array(
        [
            np.cos(yaw) * dx + np.sin(yaw) * dy,
            -np.sin(yaw) * dx + np.cos(yaw) * dy,
        ],
        dtype=np.float64,
    )


def make_hybrid_action(env: Any, base_command: Any) -> np.ndarray:
    low, high = (np.asarray(value, dtype=np.float32) for value in env.action_spec)
    if low.shape != (12,) or high.shape != (12,):
        raise ValueError(f"Expected a 12D hybrid action space, got {low.shape} / {high.shape}")
    command = np.asarray(base_command, dtype=np.float32).reshape(3)
    action = np.zeros(12, dtype=np.float32)
    action[6] = -1.0
    action[7:10] = np.clip(command, low[7:10], high[7:10])
    action[11] = 1.0
    return action


def create_environment(args: argparse.Namespace, controller_mode: str) -> Any:
    import robosuite
    from robosuite.controllers import load_composite_controller_config
    from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    if args.specific_env not in ALL_KITCHEN_ENVIRONMENTS:
        raise KeyError(f"Environment {args.specific_env!r} is not registered")
    controller_config = configure_base_controller(
        load_composite_controller_config(controller=None, robot="PandaOmron"),
        controller_mode,
    )
    return robosuite.make(
        env_name=args.specific_env,
        robots="PandaOmron",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        seed=args.seed,
        layout_ids=[LayoutType[args.layout].value],
        style_ids=[StyleType[args.style].value],
        translucent_robot=False,
    )


def controller_details(env: Any) -> dict[str, Any]:
    details: dict[str, Any] = inspect_base_controller(env)
    controller = env.robots[0].composite_controller.part_controllers["base"]
    axes = {}
    for joint_name in controller.joint_names:
        joint_id = env.sim.model.joint_name2id(joint_name)
        axes[str(joint_name)] = np.asarray(env.sim.model.jnt_axis[joint_id]).tolist()
    details["joint_axes"] = axes
    return details


def run_pulse(
    args: argparse.Namespace,
    controller_mode: str,
    label: str,
    command: tuple[float, float, float],
    pre_yaw_angle: float = 0.0,
) -> dict[str, Any]:
    env = create_environment(args, controller_mode)
    try:
        env.reset()
        for _ in range(args.settle_steps):
            env.step(make_hybrid_action(env, (0.0, 0.0, 0.0)))
        initial_pos, initial_yaw = base_pose(env)
        if pre_yaw_angle:
            # A commanded in-place rotation drifts the base into its nearby
            # spawn fixture, which makes the following translation pulse a
            # collision test instead of a frame-transform test. Set only the
            # yaw joint for this controlled condition and leave XY unchanged.
            controller = env.robots[0].composite_controller.part_controllers["base"]
            qpos_indexes = np.asarray(controller.qpos_index).reshape(-1)
            # Put the base in open floor space before changing yaw so its
            # rotated footprint does not overlap the spawn-side cabinet.
            forward_qpos_index = int(qpos_indexes[0])
            env.sim.data.qpos[forward_qpos_index] -= args.pre_yaw_clearance_m
            yaw_qpos_index = int(qpos_indexes[2])
            env.sim.data.qpos[yaw_qpos_index] += pre_yaw_angle
            env.sim.forward()
        start_pos, start_yaw = base_pose(env)
        for _ in range(args.pulse_steps):
            env.step(make_hybrid_action(env, command))
        end_pos, end_yaw = base_pose(env)

        world_delta = end_pos[:2] - start_pos[:2]
        body_delta = world_delta_to_body_action(world_delta, start_yaw)
        yaw_delta = wrap_angle(end_yaw - start_yaw)
        command_array = np.asarray(command, dtype=np.float64)
        active_axis = int(np.argmax(np.abs(command_array)))
        if active_axis < 2:
            primary = float(body_delta[active_axis] * np.sign(command_array[active_axis]))
            cross = float(body_delta[1 - active_axis])
            purity = abs(primary) / max(abs(primary) + abs(cross), 1e-12)
            passed = bool(primary >= args.minimum_translation_m and purity >= args.minimum_axis_purity)
        else:
            primary = float(yaw_delta * np.sign(command_array[2]))
            cross = float(np.linalg.norm(world_delta))
            purity = abs(primary) / max(abs(primary) + cross, 1e-12)
            passed = bool(primary >= args.minimum_yaw_rad and cross <= args.maximum_yaw_translation_m)

        result = {
            "label": label,
            "controller_mode": controller_mode,
            "controller": controller_details(env),
            "command": command_array.tolist(),
            "pre_yaw_angle_rad": pre_yaw_angle,
            "initial_pose": {"position": initial_pos.tolist(), "yaw": initial_yaw},
            "pulse_start_pose": {"position": start_pos.tolist(), "yaw": start_yaw},
            "end_pose": {"position": end_pos.tolist(), "yaw": end_yaw},
            "pre_yaw_delta_rad": wrap_angle(start_yaw - initial_yaw),
            "world_delta_xy_m": world_delta.tolist(),
            "body_delta_xy_m": body_delta.tolist(),
            "yaw_delta_rad": yaw_delta,
            "primary_response": primary,
            "cross_response": cross,
            "axis_purity": purity,
            "passed": passed,
        }
        logging.info(
            "pulse %-16s controller=%-6s body=(%+.4f,%+.4f) yaw=%+.4f purity=%.3f pass=%s",
            label,
            controller_mode,
            body_delta[0],
            body_delta[1],
            yaw_delta,
            purity,
            passed,
        )
        return result
    finally:
        env.close()


def oracle_command(
    env: Any,
    waypoint_xy: Any,
    speed: float,
    yaw_gain: float,
) -> tuple[np.ndarray, float, float]:
    position, yaw = base_pose(env)
    target_yaw = float(np.asarray(env.target_ori, dtype=np.float64)[2])
    world_error = np.asarray(waypoint_xy, dtype=np.float64).reshape(2) - position[:2]
    distance = float(np.linalg.norm(world_error))
    body_error = world_delta_to_body_action(world_error, yaw)
    if distance > 1e-9:
        # The Omron forward axis has enough static friction that small
        # proportional commands can asymptotically stall. Keep a fixed action
        # magnitude and let the waypoint/goal thresholds stop the segment.
        translation = body_error / distance * speed
    else:
        translation = np.zeros(2, dtype=np.float64)
    yaw_error = wrap_angle(target_yaw - yaw)
    yaw_action = float(np.clip(yaw_gain * yaw_error, -speed, speed))
    return np.array([translation[0], translation[1], yaw_action]), distance, yaw_error


def run_oracle(
    args: argparse.Namespace,
    controller_mode: str,
    output_path: Path,
) -> dict[str, Any]:
    env = create_environment(args, controller_mode)
    records = []
    try:
        env.reset()
        for _ in range(args.settle_steps):
            env.step(make_hybrid_action(env, (0.0, 0.0, 0.0)))
        initial_pos, initial_yaw = base_pose(env)
        target_xy = np.asarray(env.target_pos, dtype=np.float64)[:2]
        target_yaw = float(np.asarray(env.target_ori, dtype=np.float64)[2])
        # The direct source-to-target line in ONE_WALL_SMALL intersects the
        # long counter. Move away from the counter, traverse parallel to it,
        # then approach the target pose. This is a deterministic seam oracle,
        # not a general planner.
        detour_y = min(float(initial_pos[1]), float(target_xy[1])) - args.oracle_clearance_m
        waypoints = [
            np.array([initial_pos[0], detour_y], dtype=np.float64),
            np.array([target_xy[0], detour_y], dtype=np.float64),
            target_xy.copy(),
        ]
        waypoint_index = 0
        success = False
        closest_distance = float("inf")
        closest_step = None
        with output_path.open("w", encoding="utf-8") as stream:
            for step in range(args.oracle_horizon):
                position_before, _ = base_pose(env)
                while waypoint_index < len(waypoints) - 1:
                    waypoint_distance = float(
                        np.linalg.norm(waypoints[waypoint_index] - position_before[:2])
                    )
                    if waypoint_distance > args.oracle_waypoint_threshold_m:
                        break
                    waypoint_index += 1
                command, distance_before, yaw_error_before = oracle_command(
                    env,
                    waypoints[waypoint_index],
                    args.oracle_speed,
                    args.oracle_yaw_gain,
                )
                observation, reward, done, info = env.step(make_hybrid_action(env, command))
                del observation
                position, yaw = base_pose(env)
                distance = float(np.linalg.norm(target_xy - position[:2]))
                yaw_error = wrap_angle(target_yaw - yaw)
                success = bool(env._check_success())
                record = {
                    "schema_version": 1,
                    "step": step,
                    "command": command.tolist(),
                    "waypoint_index": waypoint_index,
                    "waypoint_xy": waypoints[waypoint_index].tolist(),
                    "waypoint_distance_before_m": distance_before,
                    "distance_m": distance,
                    "yaw_error_before_rad": yaw_error_before,
                    "yaw_error_rad": yaw_error,
                    "base_position": position.tolist(),
                    "base_yaw": yaw,
                    "reward": float(reward),
                    "done": bool(done),
                    "info_success": bool(info.get("success", False)),
                    "env_success": success,
                }
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                records.append(record)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_step = step
                if step == 0 or (step + 1) % args.log_interval == 0 or success:
                    logging.info(
                        "oracle controller=%-6s step=%3d dist=%.3f yaw_err=%+.3f cmd=(%+.2f,%+.2f,%+.2f) success=%s",
                        controller_mode,
                        step,
                        distance,
                        yaw_error,
                        command[0],
                        command[1],
                        command[2],
                        success,
                    )
                if success:
                    break

        final = records[-1]
        result = {
            "controller_mode": controller_mode,
            "controller": controller_details(env),
            "initial_position": initial_pos.tolist(),
            "initial_yaw": initial_yaw,
            "target_xy": target_xy.tolist(),
            "target_yaw": target_yaw,
            "waypoints": [waypoint.tolist() for waypoint in waypoints],
            "initial_distance_m": float(np.linalg.norm(target_xy - initial_pos[:2])),
            "closest_distance_m": closest_distance,
            "closest_step": closest_step,
            "final_distance_m": final["distance_m"],
            "final_yaw_error_rad": final["yaw_error_rad"],
            "executed_steps": len(records),
            "env_success": success,
            "passed": bool(success and closest_distance <= 0.20),
            "actions_path": str(output_path),
        }
        logging.info(
            "oracle result controller=%s closest=%.3f final=%.3f steps=%d success=%s",
            controller_mode,
            closest_distance,
            final["distance_m"],
            len(records),
            success,
        )
        return result
    finally:
        env.close()


def allocate_run_dir(output_root: Path, name: str) -> Path:
    candidate = output_root / name
    retry = 1
    while candidate.exists():
        retry += 1
        candidate = output_root / f"{name}__r{retry}"
    candidate.mkdir(parents=True)
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model-free PandaOmron base and oracle diagnostics")
    parser.add_argument("--specific-env", default="NavigateKitchen")
    parser.add_argument("--layout", default="ONE_WALL_SMALL")
    parser.add_argument("--style", default="MODERN_1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--controllers", nargs="+", choices=tuple(BASE_CONTROLLER_TYPES), default=["local", "legacy"]
    )
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument("--pulse-steps", type=int, default=16)
    parser.add_argument("--pre-yaw-angle-rad", type=float, default=0.60)
    parser.add_argument("--pre-yaw-clearance-m", type=float, default=0.25)
    parser.add_argument("--pulse-magnitude", type=float, default=0.5)
    parser.add_argument("--minimum-translation-m", type=float, default=0.03)
    parser.add_argument("--minimum-yaw-rad", type=float, default=0.10)
    parser.add_argument("--minimum-axis-purity", type=float, default=0.65)
    parser.add_argument("--maximum-yaw-translation-m", type=float, default=0.06)
    parser.add_argument("--oracle-horizon", type=int, default=300)
    parser.add_argument("--oracle-speed", type=float, default=0.5)
    parser.add_argument("--oracle-yaw-gain", type=float, default=0.5)
    parser.add_argument("--oracle-clearance-m", type=float, default=0.20)
    parser.add_argument("--oracle-waypoint-threshold-m", type=float, default=0.08)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--require-reproducible-source",
        action="store_true",
        help="Refuse the run unless relevant code is exactly recoverable from the recorded Git commit",
    )
    parser.add_argument("--out-root", required=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = (
        "pulse_steps",
        "oracle_horizon",
        "log_interval",
    )
    for name in positive_ints:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.settle_steps < 0:
        raise ValueError("settle steps cannot be negative")
    if not 0 < args.pulse_magnitude <= 1:
        raise ValueError("--pulse-magnitude must be in (0, 1]")
    if not 0 < args.oracle_speed <= 1:
        raise ValueError("--oracle-speed must be in (0, 1]")
    if not 0 <= args.minimum_axis_purity <= 1:
        raise ValueError("--minimum-axis-purity must be in [0, 1]")
    if args.oracle_clearance_m <= 0 or args.oracle_waypoint_threshold_m <= 0:
        raise ValueError("oracle clearance and waypoint threshold must be positive")
    if args.pre_yaw_clearance_m <= 0:
        raise ValueError("--pre-yaw-clearance-m must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    PATHS.activate_robocasa("base_diagnostics")
    output_root = require_output_root(args.out_root)
    reproducibility = build_reproducibility_manifest(
        REPO_ROOT,
        resolved_args=vars(args),
        package_names=("mujoco", "numpy", "robocasa", "robosuite"),
    )
    if args.require_reproducible_source:
        require_reproducible_source(reproducibility)
    timestamp = utc_run_stamp()
    run_dir = allocate_run_dir(
        output_root,
        f"{timestamp}__base_diagnostics__{args.specific_env}__L{args.layout}",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(run_dir / "run.log"), logging.StreamHandler()],
        force=True,
    )
    logging.info("Run directory: %s", run_dir)
    reproducibility_path = run_dir / "reproducibility.json"
    write_json_file(reproducibility_path, reproducibility)

    magnitude = args.pulse_magnitude
    pulse_conditions = (
        ("initial_forward", (magnitude, 0.0, 0.0), 0.0),
        ("initial_lateral", (0.0, magnitude, 0.0), 0.0),
        ("initial_yaw", (0.0, 0.0, magnitude), 0.0),
        ("rotated_forward", (magnitude, 0.0, 0.0), args.pre_yaw_angle_rad),
        ("rotated_lateral", (0.0, magnitude, 0.0), args.pre_yaw_angle_rad),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "scope": "model-free XR-1 base action/controller implementation diagnostic",
        "created_at": utc_now_iso(),
        "git_commit": reproducibility["repository"]["commit"],
        "reproducibility": {
            "manifest": str(reproducibility_path),
            "source_reproducible_from_commit": reproducibility["repository"][
                "source_reproducible_from_commit"
            ],
            "source_tree_sha256": reproducibility["sources"]["tree_sha256"],
        },
        "environment": {
            "name": args.specific_env,
            "layout": args.layout,
            "style": args.style,
            "robocasa_root": str(ROBOCASA_RUN_ROOT),
            "robocasa_root_source": PATHS.robocasa_root_source,
            "seed": args.seed,
        },
        "parameters": vars(args),
        "controllers": {},
    }
    write_json(run_dir / "meta.json", report)

    for controller_mode in args.controllers:
        logging.info("Starting controller condition: %s", controller_mode)
        pulses = [
            run_pulse(args, controller_mode, label, command, pre_yaw)
            for label, command, pre_yaw in pulse_conditions
        ]
        oracle = run_oracle(
            args,
            controller_mode,
            run_dir / f"oracle_{controller_mode}_actions.jsonl",
        )
        report["controllers"][controller_mode] = {
            "pulses": pulses,
            "pulse_passed": all(item["passed"] for item in pulses),
            "oracle": oracle,
        }

    rotated_purity = {
        mode: float(
            np.mean(
                [
                    item["axis_purity"]
                    for item in condition["pulses"]
                    if item["label"].startswith("rotated_")
                ]
            )
        )
        for mode, condition in report["controllers"].items()
    }
    legacy_purity_advantage = (
        rotated_purity.get("legacy", float("nan"))
        - rotated_purity.get("local", float("nan"))
    )
    report["paired_conclusion"] = {
        "all_initial_axes_passed": all(
            all(item["passed"] for item in condition["pulses"] if item["label"].startswith("initial_"))
            for condition in report["controllers"].values()
        ),
        "local_rotated_axes_passed": all(
            item["passed"]
            for item in report["controllers"].get("local", {}).get("pulses", [])
            if item["label"].startswith("rotated_")
        ),
        "legacy_rotated_axes_passed": all(
            item["passed"]
            for item in report["controllers"].get("legacy", {}).get("pulses", [])
            if item["label"].startswith("rotated_")
        ),
        "local_oracle_passed": report["controllers"].get("local", {}).get("oracle", {}).get("passed"),
        "legacy_oracle_passed": report["controllers"].get("legacy", {}).get("oracle", {}).get("passed"),
        "mean_rotated_axis_purity": rotated_purity,
        "legacy_axis_purity_advantage": legacy_purity_advantage,
        "legacy_frame_improvement_passed": bool(legacy_purity_advantage >= 0.15),
    }
    write_json(run_dir / "diagnostic_report.json", report)
    write_artifact_manifest(
        run_dir,
        (
            "diagnostic_report.json",
            "meta.json",
            "oracle_legacy_actions.jsonl",
            "oracle_local_actions.jsonl",
            "reproducibility.json",
        ),
    )
    logging.info("Diagnostic report: %s", run_dir / "diagnostic_report.json")


if __name__ == "__main__":
    main()
