"""Kinematic logging and model-profile-driven VLA smoke-gate summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def quat_xyzw_to_yaw(quaternion: Any) -> float:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64).reshape(4)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def robot_snapshot(env: Any, observation: dict[str, Any], goal_xy: Any) -> dict[str, Any]:
    robot = env.robots[0]
    base_pos = np.asarray(observation["robot0_base_pos"], dtype=np.float64).reshape(3)
    base_quat = np.asarray(observation["robot0_base_quat"], dtype=np.float64).reshape(4)
    eef_relative = np.asarray(observation["robot0_base_to_eef_pos"], dtype=np.float64).reshape(3)
    arm_qpos = np.asarray(env.sim.data.qpos[robot._ref_arm_joint_pos_indexes], dtype=np.float64).reshape(-1)
    torso_qpos = np.asarray(env.sim.data.qpos[robot._ref_torso_joint_pos_indexes], dtype=np.float64).reshape(-1)
    goal_xy = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    return {
        "base_pos": base_pos.tolist(),
        "base_yaw": quat_xyzw_to_yaw(base_quat),
        "eef_relative_pos": eef_relative.tolist(),
        "arm_qpos": arm_qpos.tolist(),
        "torso_qpos": torso_qpos.tolist(),
        "goal_distance": float(np.linalg.norm(goal_xy - base_pos[:2])),
    }


def resolve_env_success(env: Any) -> dict[str, Any]:
    """Read the env's success signal without assuming which robocasa tree is loaded.

    * pre-2026-09 trees (inline v1, v2 pin ``0f5fbc83``): ``env._check_success()`` returns
      ``success and safety_success`` (pos ∧ ori ∧ boundary-clean ∧ no-contact) and the env
      exposes ``env.success`` / ``env.safety_success``.
    * metric-cleanup trees (upstream robocasa 2026-09-04+, "one attribute per metric"):
      ``_check_success`` is removed (robosuite's base raises NotImplementedError) and the env
      exposes ``env.task_success`` (pos ∧ ori, **no safety term**) and
      ``env.collision_free_success`` (never touched the obstacle).

    ``--stop-on-env-success`` stops on ``env_success`` either way; which signal that was is
    recorded as ``success_semantics`` so a sweep can never be misread later
    (team-internal migration notes §9).
    """
    if hasattr(env, "task_success") and not hasattr(env, "safety_success"):
        task_success = bool(env.task_success)
        collision_free = getattr(env, "collision_free_success", None)
        result: dict[str, Any] = {
            "env_success": task_success,
            "task_success": task_success,
            "success_semantics": "task_success",
        }
        if collision_free is not None:
            result["collision_free_success"] = bool(collision_free)
        return result
    env_success = bool(env._check_success())
    result = {
        "env_success": env_success,
        "task_success": bool(getattr(env, "success", env_success)),
        "success_semantics": "check_success_pos_ori_safety",
    }
    safety_value = getattr(env, "safety_success", None)
    if safety_value is not None:
        result["safety_success"] = bool(safety_value)
        result["overall_success"] = env_success
    return result


def task_snapshot(
    env: Any,
    info: dict[str, Any],
    goal_distance: float,
    goal_distance_threshold_m: float | None,
) -> dict[str, Any]:
    """Capture exact env success plus optional OSMA-Bench obstacle diagnostics."""
    result = resolve_env_success(env)
    if goal_distance_threshold_m is not None:
        result["goal_distance_threshold_m"] = float(goal_distance_threshold_m)
        result["goal_distance_pass"] = bool(goal_distance <= goal_distance_threshold_m)

    attribute_fields = {
        "position_pass": "_last_pos_pass",
        "position_threshold_m": "_last_pos_threshold",
        "orientation_pass": "_last_ori_pass",
        "orientation_cos": "_last_ori_cos",
        "boundary_violated_ever": "_boundary_violation_ever",
        "obstacle_contact_ever": "_obstacle_contact_occurred",
        "obstacle_min_distance_ever_m": "_obstacle_min_distance",
    }
    for output_key, attribute in attribute_fields.items():
        if not hasattr(env, attribute):
            continue
        value = getattr(env, attribute)
        if isinstance(value, (bool, np.bool_)):
            result[output_key] = bool(value)
        elif value is not None:
            result[output_key] = float(value)

    if "min_obstacle_distance" in info:
        result["min_obstacle_distance_m"] = float(info["min_obstacle_distance"])
    if "boundary_violated" in info:
        result["boundary_violated"] = bool(info["boundary_violated"])
    if "obstacle_contact_ever" in info:
        result["obstacle_contact_ever"] = bool(info["obstacle_contact_ever"])
    if isinstance(info.get("obstacle_distances"), dict):
        result["obstacle_distances_m"] = {
            str(key): float(value) for key, value in info["obstacle_distances"].items()
        }
    if isinstance(info.get("obstacle_contacts"), dict):
        result["obstacle_contacts"] = {
            str(key): bool(value) for key, value in info["obstacle_contacts"].items()
        }
    intrusion = getattr(env, "intrusion", None)
    if isinstance(intrusion, dict) and intrusion.get("boundary_threshold") is not None:
        boundary_threshold = float(intrusion["boundary_threshold"])
        result["obstacle_boundary_threshold_m"] = boundary_threshold
        # kitchen_navigate_safe queries mj_geomDistance only up to boundary + 1 m.
        # A returned value at this cap means "at least this far", not an exact distance.
        result["obstacle_distance_query_cap_m"] = boundary_threshold + 1.0
    return result


def build_task_diagnostics(
    records: list[dict[str, Any]], goal_distance_threshold_m: float | None
) -> dict[str, Any]:
    """Summarize goal entry, exact env success, and optional obstacle safety."""
    if not records:
        raise ValueError("Cannot summarize task diagnostics without records")

    closest = min(records, key=lambda item: float(item["goal_distance"]))

    def first_step(key: str) -> int | None:
        for item in records:
            if bool(item.get("task", {}).get(key, False)):
                return int(item["step"])
        return None

    result: dict[str, Any] = {
        "goal_distance_threshold_m": goal_distance_threshold_m,
        "first_goal_distance_pass_step": first_step("goal_distance_pass"),
        "first_task_success_step": first_step("task_success"),
        "first_env_success_step": first_step("env_success"),
        "closest_goal_distance_m": float(closest["goal_distance"]),
        "closest_goal_distance_step": int(closest["step"]),
        "final_goal_distance_m": float(records[-1]["goal_distance"]),
        "final_env_success": bool(records[-1].get("task", {}).get("env_success", False)),
    }

    obstacle_records = [
        item for item in records if "min_obstacle_distance_m" in item.get("task", {})
    ]
    if obstacle_records:
        closest_obstacle = min(
            obstacle_records,
            key=lambda item: float(item["task"]["min_obstacle_distance_m"]),
        )
        distance_cap = closest_obstacle["task"].get("obstacle_distance_query_cap_m")
        distance_censored = bool(
            distance_cap is not None
            and float(closest_obstacle["task"]["min_obstacle_distance_m"])
            >= float(distance_cap) - 1e-9
        )
        result["safety"] = {
            "first_boundary_violation_step": first_step("boundary_violated"),
            "boundary_violation_step_count": sum(
                bool(item.get("task", {}).get("boundary_violated", False))
                for item in obstacle_records
            ),
            "obstacle_contact_ever": any(
                bool(item.get("task", {}).get("obstacle_contact_ever", False))
                for item in obstacle_records
            ),
            "minimum_obstacle_distance_m": float(
                closest_obstacle["task"]["min_obstacle_distance_m"]
            ),
            "minimum_obstacle_distance_step": int(closest_obstacle["step"]),
            "minimum_obstacle_distance_is_censored": distance_censored,
            "obstacle_distance_query_cap_m": distance_cap,
            "final_safety_success": bool(
                records[-1].get("task", {}).get("safety_success", False)
            ),
            "final_overall_success": bool(
                records[-1].get("task", {}).get("overall_success", False)
            ),
            # metric-cleanup trees: CSR term (None on trees that do not expose it)
            "final_collision_free_success": records[-1].get("task", {}).get(
                "collision_free_success"
            ),
        }
    result["success_semantics"] = records[-1].get("task", {}).get("success_semantics")
    return result


@dataclass(frozen=True)
class SmokeThresholds:
    minimum_goal_distance_drop_m: float = 0.10
    minimum_base_translation_m: float = 0.05
    maximum_eef_relative_drift_m: float = 0.15
    maximum_arm_joint_l2_drift_rad: float = 0.75
    maximum_torso_drift: float = 0.10
    maximum_goal_divergence_m: float = 0.50
    maximum_stalled_chunks: int = 1
    action_range_tolerance: float = 1e-5
    stalled_chunk_translation_m: float = 0.01
    stalled_chunk_yaw_rad: float = float(np.deg2rad(1.0))
    component_activity_threshold: float = 0.05
    observed_component_motion_threshold: float = 0.005


def _vector(record: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(record[key], dtype=np.float64)


def _chunk_reports(initial: dict[str, Any], records: list[dict[str, Any]], thresholds: SmokeThresholds):
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["chunk_id"]), []).append(record)
    reports = []
    previous = initial
    for chunk_id in sorted(grouped):
        chunk = grouped[chunk_id]
        final = chunk[-1]
        translation = float(np.linalg.norm(_vector(final, "base_pos")[:2] - _vector(previous, "base_pos")[:2]))
        yaw_delta = float(
            abs(np.arctan2(np.sin(final["base_yaw"] - previous["base_yaw"]), np.cos(final["base_yaw"] - previous["base_yaw"])))
        )
        distance_drop = float(previous["goal_distance"] - final["goal_distance"])
        stalled = (
            translation < thresholds.stalled_chunk_translation_m
            and yaw_delta < thresholds.stalled_chunk_yaw_rad
        )
        reports.append(
            {
                "chunk_id": chunk_id,
                "steps": len(chunk),
                "translation_m": translation,
                "yaw_delta_rad": yaw_delta,
                "goal_distance_drop_m": distance_drop,
                "stalled": stalled,
            }
        )
        previous = final
    return reports


def build_smoke_report(
    initial: dict[str, Any],
    records: list[dict[str, Any]],
    query_audits: list[dict[str, Any]],
    thresholds: SmokeThresholds,
    *,
    model_profile: dict[str, Any],
    action_bound_mode: str = "strict",
) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize an empty rollout")

    distances = np.asarray([initial["goal_distance"]] + [item["goal_distance"] for item in records])
    x = np.arange(distances.size, dtype=np.float64)
    slope = float(np.polyfit(x, distances, 1)[0]) if distances.size > 1 else 0.0
    window = max(1, distances.size // 5)
    first_median = float(np.median(distances[:window]))
    last_median = float(np.median(distances[-window:]))
    net_drop = float(distances[0] - distances[-1])
    maximum_divergence = float(np.max(distances - distances[0]))

    initial_base = _vector(initial, "base_pos")
    final_base = _vector(records[-1], "base_pos")
    base_translation = float(np.linalg.norm(final_base[:2] - initial_base[:2]))
    yaw_series = np.unwrap([initial["base_yaw"]] + [item["base_yaw"] for item in records])
    net_yaw = float(yaw_series[-1] - yaw_series[0])
    positions = np.asarray(
        [initial["base_pos"]] + [item["base_pos"] for item in records], dtype=np.float64
    )[:, :2]
    world_deltas = np.diff(positions, axis=0)
    reference_yaws = yaw_series[:-1]
    body_forward = -world_deltas[:, 0] * np.cos(reference_yaws) + world_deltas[:, 1] * np.sin(
        reference_yaws
    )
    body_lateral = -world_deltas[:, 0] * np.sin(reference_yaws) - world_deltas[:, 1] * np.cos(
        reference_yaws
    )
    yaw_deltas = np.diff(yaw_series)
    observed_component_motion = {
        "forward_travel_m": float(np.sum(np.abs(body_forward))),
        "lateral_travel_m": float(np.sum(np.abs(body_lateral))),
        "yaw_travel_rad": float(np.sum(np.abs(yaw_deltas))),
    }
    observed_component_motion["forward_moved"] = bool(
        observed_component_motion["forward_travel_m"]
        >= thresholds.observed_component_motion_threshold
    )
    observed_component_motion["lateral_moved"] = bool(
        observed_component_motion["lateral_travel_m"]
        >= thresholds.observed_component_motion_threshold
    )
    observed_component_motion["yaw_moved"] = bool(
        observed_component_motion["yaw_travel_rad"]
        >= thresholds.observed_component_motion_threshold
    )

    initial_eef = _vector(initial, "eef_relative_pos")
    initial_arm = _vector(initial, "arm_qpos")
    initial_torso = _vector(initial, "torso_qpos")
    max_eef_drift = max(
        float(np.linalg.norm(_vector(item, "eef_relative_pos") - initial_eef)) for item in records
    )
    max_arm_drift = max(
        float(np.linalg.norm(_vector(item, "arm_qpos") - initial_arm)) for item in records
    )
    max_torso_drift = max(
        float(np.max(np.abs(_vector(item, "torso_qpos") - initial_torso))) for item in records
    )

    actions = np.asarray([item["applied_action"] for item in records], dtype=np.float64)
    if action_bound_mode not in {"official_passthrough", "strict"}:
        raise ValueError(f"Unknown action_bound_mode {action_bound_mode!r}")
    lower_excess = np.maximum(-1.0 - actions, 0.0)
    upper_excess = np.maximum(actions - 1.0, 0.0)
    bound_excess = np.maximum(lower_excess, upper_excess)
    bound_excursions = bound_excess > thresholds.action_range_tolerance
    actions_finite_12d = bool(np.isfinite(actions).all() and actions.shape[1] == 12)
    within_declared_bounds = bool(not np.any(bound_excursions))
    component_names = {7: "forward", 8: "lateral", 9: "yaw"}
    component_activity = {
        name: {
            "peak_abs_action": float(np.max(np.abs(actions[:, index]))),
            "active_fraction": float(
                np.mean(np.abs(actions[:, index]) >= thresholds.component_activity_threshold)
            ),
        }
        for index, name in component_names.items()
    }

    expected_slots = list(model_profile["camera_slots"])
    history_length = int(model_profile["history_length"])
    history_interval = int(model_profile["history_interval"])
    contract = model_profile["input_audit"]

    def audit_matches_contract(audit: dict[str, Any]) -> bool:
        images = audit.get("images", {})
        query_step = audit.get("query_step")
        history_steps = audit.get("history_steps", [])
        if not isinstance(query_step, int):
            return False
        expected_history_steps = [
            max(0, query_step - history_interval * offset)
            for offset in reversed(range(history_length))
        ]
        return bool(
            audit.get("camera_slots") == expected_slots
            and set(images) == set(expected_slots)
            and all(
                images[slot].get("shape") == contract["image_history_shape"]
                and images[slot].get("dtype") == "uint8"
                and bool(images[slot].get("sha256"))
                for slot in expected_slots
            )
            and history_steps == expected_history_steps
            and audit.get("state_native_shape") == contract["state_native_shape"]
            and audit.get("policy_state_shape") == contract["policy_state_shape"]
            and len(audit.get("state_native", [])) == history_length
            and bool(audit.get("state_native_sha256"))
            and audit.get("preprocess") == contract["preprocess"]
        )

    input_contract_ok = bool(query_audits) and all(
        audit_matches_contract(audit) for audit in query_audits
    )
    chunks = _chunk_reports(initial, records, thresholds)
    stalled_chunks = sum(int(item["stalled"]) for item in chunks)

    checks = {
        "input_contract": input_contract_ok,
        "action_contract": bool(
            actions_finite_12d
            and (within_declared_bounds or action_bound_mode == "official_passthrough")
        ),
        "base_moved": base_translation >= thresholds.minimum_base_translation_m,
        "goal_distance_decreased": (
            net_drop >= thresholds.minimum_goal_distance_drop_m
            and slope < 0.0
            and last_median < first_median
        ),
        "arm_stable": (
            max_eef_drift <= thresholds.maximum_eef_relative_drift_m
            and max_arm_drift <= thresholds.maximum_arm_joint_l2_drift_rad
        ),
        "torso_stable": max_torso_drift <= thresholds.maximum_torso_drift,
        "chunk_execution_stable": (
            stalled_chunks <= thresholds.maximum_stalled_chunks
            and maximum_divergence <= thresholds.maximum_goal_divergence_m
        ),
    }
    return {
        "scope": "VLA motion smoke gates; task and safety diagnostics are reported separately",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": asdict(thresholds),
        "distance": {
            "initial_m": float(distances[0]),
            "final_m": float(distances[-1]),
            "net_drop_m": net_drop,
            "linear_slope_m_per_step": slope,
            "first_window_median_m": first_median,
            "last_window_median_m": last_median,
            "maximum_divergence_m": maximum_divergence,
        },
        "base": {
            "translation_m": base_translation,
            "net_yaw_rad": net_yaw,
            "component_activity": component_activity,
            "observed_component_motion": observed_component_motion,
        },
        "action_bounds": {
            "mode": action_bound_mode,
            "runner_clipped": False,
            "declared_low": -1.0,
            "declared_high": 1.0,
            "within_declared_bounds": within_declared_bounds,
            "excursion_element_count": int(np.count_nonzero(bound_excursions)),
            "excursion_step_count": int(np.count_nonzero(np.any(bound_excursions, axis=1))),
            "peak_excess": float(np.max(bound_excess)),
            "max_abs_by_index": np.max(np.abs(actions), axis=0).tolist(),
        },
        "non_base_drift": {
            "maximum_eef_relative_drift_m": max_eef_drift,
            "maximum_arm_joint_l2_drift_rad": max_arm_drift,
            "maximum_torso_drift": max_torso_drift,
        },
        "chunks": chunks,
        "query_count": len(query_audits),
        "step_count": len(records),
    }
