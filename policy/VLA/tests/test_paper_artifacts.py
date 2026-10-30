"""build_paper_artifacts 회귀 테스트 — 모델·시뮬 없이 돈다."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paper_artifacts import build_paper_artifacts  # noqa: E402


def _record(step, pos, yaw, goal_distance, task):
    return {"step": step, "base_pos": [*pos, 0.0], "base_yaw": yaw,
            "goal_distance": goal_distance, "task": task}


class TestPaperArtifacts(unittest.TestCase):
    def setUp(self):
        self.task_ok = {"position_pass": True, "orientation_pass": True,
                        "orientation_cos": 0.99, "position_threshold_m": 0.6,
                        "obstacle_min_distance_ever_m": 0.35}
        self.records = [
            _record(0, (0.0, 0.0), 0.0, 2.0, self.task_ok),
            _record(1, (1.0, 0.0), 0.1, 1.0, self.task_ok),
            _record(2, (1.5, 0.0), 0.2, 0.5, self.task_ok),
        ]
        self.ti = {"path_length": 1.5, "jerk_max": 12.0,
                   "collision_steps": 0,
                   "timeseries_velocity": [0.1, 0.2],
                   "timeseries_jerk": [1.0, 2.0],
                   "timeseries_min_obstacle_distance": [0.9, 0.35]}

    def _build(self, **kw):
        args = dict(records=self.records, trajectory_info=self.ti,
                    goal_xy=[1.6, 0.0], instruction="navigate safely")
        args.update(kw)
        return build_paper_artifacts(**args)

    def test_collision_free_success_uses_contact_steps(self):
        out = self._build()
        self.assertTrue(out["evaluation"]["success"])
        self.assertEqual(out["evaluation"]["collision_free_success"], 1)
        collided = self._build(trajectory_info={**self.ti, "collision_steps": 2})
        self.assertEqual(collided["evaluation"]["collision_free_success"], 0)

    def test_position_failure(self):
        bad = {**self.task_ok, "position_pass": False}
        recs = [_record(i, (0.0, 0.0), 0.0, 3.0, bad) for i in range(3)]
        out = self._build(records=recs)
        self.assertFalse(out["evaluation"]["success"])
        self.assertEqual(out["evaluation"]["collision_free_success"], 0)

    def test_required_keys_for_compute_ssi(self):
        ev = self._build()["evaluation"]
        for key in ("success", "collision_free_success", "contact_steps", "dist_to_goal_m", "ori_cos",
                    "jerk_max", "min_clearance_m",
                    "num_steps", "path_length_m"):
            self.assertIn(key, ev, key)

    def test_trajectory_log_shape(self):
        log = self._build()["trajectory_log"]
        self.assertEqual(log["velocity"], [0.1, 0.2])
        self.assertEqual(log["jerk"], [1.0, 2.0])
        self.assertEqual(len(log["robot_pos"]), 3)
        self.assertEqual(len(log["robot_yaw"]), 3)
        self.assertEqual(log["robot_pos"][0], [0.0, 0.0])

    def test_missing_trajectory_info_is_tolerated(self):
        out = self._build(trajectory_info=None)
        self.assertIsNotNone(out["evaluation"]["path_length_m"])   # 위치에서 재계산
        self.assertEqual(out["trajectory_log"]["velocity"], [])

    def test_non_finite_values_become_none(self):
        out = self._build(trajectory_info={**self.ti, "jerk_max": float("inf")})
        self.assertIsNone(out["evaluation"]["jerk_max"])



class TestMetricCleanupFields(unittest.TestCase):
    """metric-cleanup 트리(2026-09-04+): boundary ratio 없음, env 가 task/collision-free 항을 따로 준다."""

    def setUp(self):
        self.task = {"position_pass": True, "orientation_pass": True, "orientation_cos": 0.9,
                     "position_threshold_m": 0.5, "task_success": True,
                     "collision_free_success": False, "obstacle_contact_ever": True,
                     "success_semantics": "task_success", "obstacle_min_distance_ever_m": 0.2}
        self.records = [_record(i, (0.5 * i, 0.0), 0.0, 1.0 - 0.4 * i, self.task) for i in range(3)]

    def test_missing_contact_count_defaults_to_clean(self):
        ti = {"path_length": 1.0, "timeseries_velocity": [0.1]}          # boundary ratio 없음
        ev = build_paper_artifacts(records=self.records, trajectory_info=ti, goal_xy=[1.0, 0.0],
                                   instruction="x")["evaluation"]
        self.assertTrue(ev["success"])
        self.assertEqual(ev["collision_free_success"], 1)
        self.assertEqual(ev["contact_steps"], 0)

    def test_env_terms_are_carried(self):
        ev = build_paper_artifacts(records=self.records, trajectory_info={}, goal_xy=[1.0, 0.0],
                                   instruction="x")["evaluation"]
        self.assertTrue(ev["task_success_env"])
        self.assertFalse(ev["collision_free_success_env"])
        self.assertTrue(ev["obstacle_contact_ever"])
        self.assertEqual(ev["success_semantics"], "task_success")

    def test_legacy_task_dict_gives_none_env_terms(self):
        legacy = {"position_pass": True, "orientation_pass": True, "orientation_cos": 0.9,
                  "position_threshold_m": 0.6}
        recs = [_record(0, (0.0, 0.0), 0.0, 0.3, legacy)]
        ev = build_paper_artifacts(records=recs, trajectory_info={"obstacle_contact_steps": 1},
                                   goal_xy=[0.0, 0.0], instruction="x")["evaluation"]
        self.assertEqual(ev["collision_free_success"], 0)
        self.assertIsNone(ev["collision_free_success_env"])


if __name__ == "__main__":
    unittest.main()
