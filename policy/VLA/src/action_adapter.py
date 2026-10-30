"""Validate and map model-native actions into local HYBRID_MOBILE_BASE actions."""

from __future__ import annotations

from typing import Any

import numpy as np


EXPECTED_SPLITS = {
    "right": (0, 6),
    "right_gripper": (6, 7),
    "base": (7, 10),
    "torso": (10, 11),
}


def inspect_hybrid_action_space(env: Any) -> dict[str, Any]:
    low, high = (np.asarray(part, dtype=np.float32) for part in env.action_spec)
    controller = env.robots[0].composite_controller
    split = getattr(controller, "_action_split_indexes", None)
    if split is None:
        raise RuntimeError("Composite controller has no _action_split_indexes")
    observed = {key: tuple(value) for key, value in split.items()}
    info = {
        "action_dim": int(low.size),
        "controller": getattr(controller, "name", type(controller).__name__),
        "splits": observed,
        "low": low.tolist(),
        "high": high.tolist(),
    }
    if low.shape != (12,) or high.shape != (12,):
        raise ValueError(f"XR-1 requires local action_dim=12; got low={low.shape}, high={high.shape}")
    if info["controller"] != "HYBRID_MOBILE_BASE":
        raise ValueError(f"Expected HYBRID_MOBILE_BASE; got {info['controller']}")
    for part, expected in EXPECTED_SPLITS.items():
        if observed.get(part) != expected:
            raise ValueError(f"Action split {part!r} is {observed.get(part)}; expected {expected}")
    return info


def validate_action_chunk(
    actions: Any,
    low: np.ndarray,
    high: np.ndarray,
    tolerance: float = 1e-5,
    enforce_bounds: bool = True,
    action_convention: str = "robocasa_ee_first",
) -> np.ndarray:
    chunk = np.asarray(actions, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != 12:
        raise ValueError(f"Expected actions[N,12], got {chunk.shape}")
    if chunk.shape[0] < 1:
        raise ValueError("Action chunk is empty")
    if not np.isfinite(chunk).all():
        raise ValueError("Action chunk contains NaN or Inf")
    low = np.asarray(low, dtype=np.float32).reshape(1, 12)
    high = np.asarray(high, dtype=np.float32).reshape(1, 12)
    # Range checks must happen in the local controller layout: binary logits
    # and base-first permutations are part of the upstream action convention.
    # Continuous channels remain unmodified and un-clipped.
    local_for_range_check = np.stack(
        [to_local_hybrid(row, action_convention) for row in chunk], axis=0
    )
    bad = np.argwhere(
        (local_for_range_check < low - tolerance)
        | (local_for_range_check > high + tolerance)
    )
    if bad.size and enforce_bounds:
        row, column = bad[0]
        raise ValueError(
            f"Action[{row},{column}]={local_for_range_check[row, column]:.6f} is outside "
            f"[{low[0, column]:.6f}, {high[0, column]:.6f}]"
        )
    return np.ascontiguousarray(chunk)


def robocasa_ee_first_to_local_hybrid(action: Any) -> np.ndarray:
    """Map upstream RoboCasa365's flat action to local v0.2.

    EE pose [0:6], base xyz/yaw velocity [7:10], and torso [10] preserve
    their values. Matching RoboCasa's official ``PandaOmronKeyConverter``,
    gripper-close [6] and control-mode [11] are thresholded at 0.5 into
    the local controller's -1/+1 representation. Upstream
    ``base_motion[7:11]`` combines local base [7:10] and torso [10].
    """
    result = np.asarray(action, dtype=np.float32).reshape(-1)
    if result.shape != (12,):
        raise ValueError(f"Expected a 12D action, got {result.shape}")
    result = result.copy()
    result[6] = -1.0 if result[6] < 0.5 else 1.0
    result[11] = -1.0 if result[11] < 0.5 else 1.0
    return result


def lerobot_base_first_to_local_hybrid(action: Any) -> np.ndarray:
    """Map OpenTau/LeRobot RoboCasa base-first actions to local v0.2.

    Source layout is ``[base_motion4, control_mode1, ee_pos3, ee_rot3, gripper1]``.
    Local layout is ``[ee_pos3, ee_rot3, gripper1, base3, torso1, control_mode1]``.
    """
    source = np.asarray(action, dtype=np.float32).reshape(-1)
    if source.shape != (12,):
        raise ValueError(f"Expected a 12D action, got {source.shape}")
    result = source[[5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]].copy()
    result[6] = -1.0 if result[6] < 0.5 else 1.0
    result[11] = -1.0 if result[11] < 0.5 else 1.0
    return result


CONVENTION_TO_LOCAL = {
    "robocasa_ee_first": robocasa_ee_first_to_local_hybrid,
    "lerobot_base_first": lerobot_base_first_to_local_hybrid,
}


def to_local_hybrid(action: Any, action_convention: str) -> np.ndarray:
    try:
        converter = CONVENTION_TO_LOCAL[action_convention]
    except KeyError as error:
        choices = ", ".join(sorted(CONVENTION_TO_LOCAL))
        raise ValueError(
            f"Unknown action convention {action_convention!r}; available: {choices}"
        ) from error
    return converter(action)
