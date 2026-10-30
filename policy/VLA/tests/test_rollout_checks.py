from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from rollout_checks import SmokeThresholds, build_smoke_report, build_task_diagnostics  # noqa: E402


XR1_PROFILE = {
    "camera_slots": ["left", "right", "wrist"],
    "history_length": 4,
    "history_interval": 2,
    "input_audit": {
        "image_history_shape": [4, 256, 256, 3],
        "state_native_shape": [4, 14],
        "policy_state_shape": [1, 4, 60],
        "preprocess": {"crop_ratio": 0.95},
    },
}


def input_audit(query_step=0):
    return {
        "camera_slots": ["left", "right", "wrist"],
        "images": {
            slot: {"shape": [4, 256, 256, 3], "dtype": "uint8", "sha256": "digest"}
            for slot in ("left", "right", "wrist")
        },
        "history_steps": [max(0, query_step - offset) for offset in (6, 4, 2, 0)],
        "query_step": query_step,
        "state_native_shape": [4, 14],
        "state_native": [[0.0] * 14 for _ in range(4)],
        "policy_state_shape": [1, 4, 60],
        "state_native_sha256": "digest",
        "preprocess": {"crop_ratio": 0.95},
        "state14_shape": [4, 14],
        "state14": [[0.0] * 14 for _ in range(4)],
        "state60_shape": [1, 4, 60],
        "state14_sha256": "digest",
        "crop_ratio": 0.95,
    }


class RolloutChecksTest(unittest.TestCase):
    def make_rollout(self):
        initial = {
            "base_pos": [0.0, 0.0, 0.0],
            "base_yaw": 0.0,
            "eef_relative_pos": [0.3, 0.0, 0.5],
            "arm_qpos": [0.0] * 7,
            "torso_qpos": [0.0],
            "goal_distance": 2.0,
        }
        records = []
        for step in range(32):
            action = [0.0] * 12
            action[7] = 0.2
            records.append(
                {
                    "chunk_id": step // 16,
                    "applied_action": action,
                    "base_pos": [0.02 * (step + 1), 0.0, 0.0],
                    "base_yaw": 0.0,
                    "eef_relative_pos": [0.3, 0.0, 0.5],
                    "arm_qpos": [0.0] * 7,
                    "torso_qpos": [0.0],
                    "goal_distance": 2.0 - 0.02 * (step + 1),
                }
            )
        return initial, records

    def test_passing_smoke_report(self):
        initial, records = self.make_rollout()
        report = build_smoke_report(
            initial, records, [input_audit(0), input_audit(16)], SmokeThresholds(),
            model_profile=XR1_PROFILE,
        )
        self.assertTrue(report["passed"])
        self.assertTrue(report["checks"]["goal_distance_decreased"])

    def test_torso_drift_fails_only_torso_gate(self):
        initial, records = self.make_rollout()
        drifted = copy.deepcopy(records)
        drifted[-1]["torso_qpos"] = [0.2]
        report = build_smoke_report(
            initial, drifted, [input_audit(0), input_audit(16)], SmokeThresholds(),
            model_profile=XR1_PROFILE,
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["torso_stable"])

    def test_incomplete_camera_or_wrong_history_fails_input_gate(self):
        initial, records = self.make_rollout()
        bad = input_audit(0)
        del bad["images"]["wrist"]
        bad["history_steps"] = [0, 0, 0, 1]
        report = build_smoke_report(
            initial, records, [bad], SmokeThresholds(), model_profile=XR1_PROFILE
        )
        self.assertFalse(report["checks"]["input_contract"])

    def test_out_of_range_applied_action_fails_action_gate(self):
        initial, records = self.make_rollout()
        records[0]["applied_action"][7] = 1.1
        report = build_smoke_report(
            initial, records, [input_audit(0), input_audit(16)], SmokeThresholds(),
            model_profile=XR1_PROFILE,
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["action_contract"])

    def test_official_passthrough_keeps_bound_excursion_as_diagnostic(self):
        initial, records = self.make_rollout()
        records[0]["applied_action"][7] = 1.1
        report = build_smoke_report(
            initial,
            records,
            [input_audit(0), input_audit(16)],
            SmokeThresholds(),
            model_profile=XR1_PROFILE,
            action_bound_mode="official_passthrough",
        )
        self.assertTrue(report["passed"])
        self.assertTrue(report["checks"]["action_contract"])
        self.assertFalse(report["action_bounds"]["within_declared_bounds"])
        self.assertEqual(report["action_bounds"]["excursion_element_count"], 1)

    def test_task_diagnostics_records_first_success_and_safety_minimum(self):
        records = [
            {
                "step": 0,
                "goal_distance": 1.0,
                "task": {
                    "env_success": False,
                    "task_success": False,
                    "goal_distance_pass": False,
                    "min_obstacle_distance_m": 0.9,
                    "obstacle_distance_query_cap_m": 1.6,
                    "boundary_violated": False,
                    "obstacle_contact_ever": False,
                    "safety_success": True,
                    "overall_success": False,
                },
            },
            {
                "step": 1,
                "goal_distance": 0.5,
                "task": {
                    "env_success": True,
                    "task_success": True,
                    "goal_distance_pass": True,
                    "min_obstacle_distance_m": 0.7,
                    "obstacle_distance_query_cap_m": 1.6,
                    "boundary_violated": False,
                    "obstacle_contact_ever": False,
                    "safety_success": True,
                    "overall_success": True,
                },
            },
        ]
        report = build_task_diagnostics(records, 0.6)
        self.assertEqual(report["first_goal_distance_pass_step"], 1)
        self.assertEqual(report["first_env_success_step"], 1)
        self.assertEqual(report["safety"]["minimum_obstacle_distance_m"], 0.7)
        self.assertFalse(report["safety"]["minimum_obstacle_distance_is_censored"])
        self.assertTrue(report["safety"]["final_overall_success"])



from rollout_checks import resolve_env_success, task_snapshot  # noqa: E402


class _LegacyEnv:
    """v1 인라인 / v2 0f5fbc83: _check_success = success and safety_success."""

    def __init__(self, success, safety):
        self.success = success
        self.safety_success = safety
        self._last_pos_pass = success
        self._last_pos_threshold = 0.9          # RouteF 완화(0.6+0.3)된 값을 흉내
        self._last_ori_pass = True
        self._last_ori_cos = 0.95
        self._boundary_violation_ever = not safety
        self._obstacle_contact_occurred = False
        self.intrusion = {"boundary_threshold": 0.6}

    def _check_success(self):
        return self.success and self.safety_success


class _MetricCleanupEnv:
    """robocasa metric-cleanup(2026-09-04+): task_success / collision_free_success, _check_success 없음."""

    def __init__(self, task, collision_free):
        self.task_success = task
        self.collision_free_success = collision_free
        self._last_pos_pass = task
        self._last_pos_threshold = 0.5
        self._last_ori_pass = True
        self._last_ori_cos = 0.9
        self._obstacle_contact_occurred = not collision_free
        self.intrusion = {}

    def _check_success(self):               # robosuite base 와 동일
        raise NotImplementedError


class SuccessSemanticsTest(unittest.TestCase):
    def test_legacy_tree_stops_on_pos_ori_safety(self):
        t = task_snapshot(_LegacyEnv(success=True, safety=False), {}, goal_distance=0.5,
                          goal_distance_threshold_m=0.6)
        self.assertFalse(t["env_success"])           # 안전 항 때문에 정지 신호는 False
        self.assertTrue(t["task_success"])
        self.assertFalse(t["safety_success"])
        self.assertEqual(t["success_semantics"], "check_success_pos_ori_safety")
        self.assertEqual(t["position_threshold_m"], 0.9)
        self.assertTrue(t["boundary_violated_ever"])
        self.assertNotIn("collision_free_success", t)

    def test_metric_cleanup_tree_stops_on_task_success_only(self):
        t = task_snapshot(_MetricCleanupEnv(task=True, collision_free=False), {}, goal_distance=0.4,
                          goal_distance_threshold_m=0.5)
        self.assertTrue(t["env_success"])            # 접촉이 있어도 도착이면 정지 신호 True
        self.assertTrue(t["task_success"])
        self.assertFalse(t["collision_free_success"])
        self.assertTrue(t["obstacle_contact_ever"])
        self.assertEqual(t["success_semantics"], "task_success")
        self.assertEqual(t["position_threshold_m"], 0.5)
        self.assertNotIn("safety_success", t)
        self.assertNotIn("boundary_violated_ever", t)

    def test_metric_cleanup_never_calls_check_success(self):
        r = resolve_env_success(_MetricCleanupEnv(task=False, collision_free=True))
        self.assertFalse(r["env_success"])
        self.assertTrue(r["collision_free_success"])

    def test_diagnostics_carry_semantics_and_collision_free(self):
        records = []
        for step in range(3):
            records.append({
                "step": step, "goal_distance": 1.0 - 0.3 * step,
                "task": {"env_success": step == 2, "task_success": step == 2,
                         "collision_free_success": False, "success_semantics": "task_success",
                         "min_obstacle_distance_m": 0.4, "obstacle_contact_ever": True},
            })
        diag = build_task_diagnostics(records, 0.5)
        self.assertEqual(diag["success_semantics"], "task_success")
        self.assertEqual(diag["first_env_success_step"], 2)
        self.assertIs(diag["safety"]["final_collision_free_success"], False)
        self.assertTrue(diag["safety"]["obstacle_contact_ever"])


if __name__ == "__main__":
    unittest.main()
