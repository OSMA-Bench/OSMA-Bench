"""논문 metric toolchain 입력 포맷 생성 (2026-08-18).

`scripts/compute_ssi.py` 와 `benchmark/robocasa/robocasa/utils/ssi.py` 는
`results.json` 의 `evaluation` dict 와 `{task_dir}/trajectory_log.json` 을 먹는다.
그 두 파일을 VLA rollout 산출물로부터 만든다.

키 구성은 `policy/Voxposer/src/run_LMP.py` 의 정책 무관 평가 블록(:547-583)과 동일하게 맞췄다.
그래야 `compute_ssi.py` 를 수정 없이 쓸 수 있다. VoxPoser wrapper 자체는 재사용할 수 없는데
`VoxPoserRobocasa.__init__` 이 `controller="BASIC"`(11D)을 하드코딩하기 때문이다.

`evaluation.success` 는 task 성공만 나타내고, 접촉 없는 성공은
`collision_free_success` 로 분리한다.
"""
from __future__ import annotations

from typing import Any

import numpy as np

CONTROL_FREQ = 20


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def build_paper_artifacts(
    *,
    records: list[dict[str, Any]],
    trajectory_info: dict[str, Any] | None,
    goal_xy: Any,
    instruction: str,
) -> dict[str, Any]:
    """Return {'evaluation': ..., 'trajectory_log': ...} in run_LMP's schema."""
    ti = trajectory_info or {}

    positions = [list(map(float, r["base_pos"][:2])) for r in records if "base_pos" in r]
    yaws = [float(r["base_yaw"]) for r in records if "base_yaw" in r]

    num_steps = len(records)
    duration_s = num_steps / CONTROL_FREQ if num_steps else 0.0
    path_length = _f(ti.get("path_length"))
    if path_length is None and len(positions) > 1:
        arr = np.asarray(positions)
        path_length = float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())
    path_length = path_length or 0.0

    last = records[-1] if records else {}
    task = last.get("task", {}) or {}

    # 논문 정의: S_n = 1[pos_dist ≤ δpos ∧ ori_cos ≥ δori]. safety 는 제외한다.
    dist_to_goal = _f(last.get("goal_distance"))
    ori_cos = _f(task.get("orientation_cos"))
    pos_pass = task.get("position_pass")
    ori_pass = task.get("orientation_pass")
    if pos_pass is not None and ori_pass is not None:
        task_success = bool(pos_pass) and bool(ori_pass)
    else:                                   # env 가 값을 안 준 경우의 보수적 재유도
        thr = _f(task.get("position_threshold_m")) or 0.6
        task_success = bool(
            dist_to_goal is not None and ori_cos is not None
            and dist_to_goal <= thr and ori_cos >= 0.8
        )

    obs_min = _f(task.get("obstacle_min_distance_ever_m"))
    if obs_min is None:
        obs_min = _f(ti.get("obstacle_min_distance"))
    contact_steps = ti.get("collision_steps")
    if contact_steps is None:
        contact_steps = ti.get("obstacle_contact_steps")
    if contact_steps is None:
        contact_steps = ti.get("obstacle_contact_count")
    contact_steps = int(contact_steps or 0)

    evaluation = {
        "success": task_success,
        "collision_free_success": int(task_success and contact_steps == 0),
        "contact_steps": contact_steps,
        # env-reported terms (metric-cleanup trees; None on older trees)
        "task_success_env": task.get("task_success"),
        "collision_free_success_env": task.get("collision_free_success"),
        "obstacle_contact_ever": task.get("obstacle_contact_ever"),
        "success_semantics": task.get("success_semantics"),
        "dist_to_goal_m": dist_to_goal,
        "ori_cos": ori_cos,
        "avg_velocity_m_s": (path_length / duration_s) if duration_s else 0.0,
        "duration_s": duration_s,
        "jerk_max": _f(ti.get("jerk_max")),
        "min_clearance_m": obs_min,
        "num_steps": num_steps,
        "path_length_m": path_length,
        "start_pos_xy": positions[0] if positions else None,
        "final_pos_xy": positions[-1] if positions else None,
        "final_yaw_rad": yaws[-1] if yaws else None,
        "goal_pos_xy": [float(x) for x in np.asarray(goal_xy).ravel()[:2]],
        "goal_yaw_rad": _f(task.get("goal_yaw_rad")),
        "instruction": instruction,
    }

    trajectory_log = {
        "velocity": list(ti.get("timeseries_velocity") or []),
        "jerk": list(ti.get("timeseries_jerk") or []),
        "min_obstacle_distance": list(ti.get("timeseries_min_obstacle_distance") or []),
        "obstacle_distances": dict(ti.get("timeseries_obstacle_distances") or {}),
        "robot_pos": positions,
        "robot_yaw": yaws,
    }
    return {"evaluation": evaluation, "trajectory_log": trajectory_log}
