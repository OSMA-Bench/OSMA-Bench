"""Model-free tests for episode_dump.EpisodeDumpRecorder (vla_dump.npz)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import episode_dump as ed  # noqa: E402


def _fake_sim(nbody=4, ngeom=6):
    # bodies: 0 world, 1 robot, 2 obstacle_dog_main, 3 fixture
    geom_bodyid = np.array([1, 1, 2, 2, 3, 3])[:ngeom]
    names = {0: "world", 1: "robot0_base", 2: "obstacle_dog_main", 3: "counter_main"}
    model = SimpleNamespace(
        geom_bodyid=geom_bodyid,
        nbody=nbody,
        body_id2name=lambda i: names.get(i),
        geom_id2name=lambda i: f"g{i}",
    )
    data = SimpleNamespace(
        geom_xpos=np.arange(ngeom * 3, dtype=float).reshape(ngeom, 3),
        body_xpos=np.arange(nbody * 3, dtype=float).reshape(nbody, 3) * 10,
        body_xquat=np.tile(np.array([1.0, 0, 0, 0]), (nbody, 1)),
    )
    return SimpleNamespace(model=model, data=data)


def _fake_env(v2=True):
    env = SimpleNamespace(sim=_fake_sim(), obstacle="dog", control_freq=20)
    if v2:
        env._obstacle_geom_groups = lambda: ({"obstacle_dog": [2, 3]}, {0, 1})
        env.intrusion = {
            "obstacle_contact_forces": {"obstacle_dog": 0.7},
            "boundary_threshold": 0.5,
            "boundary_violated_ever": True,
        }
    return env


def _snapshot(x):
    return {"base_pos": [x, 0.0, 0.7], "base_yaw": 1.5, "eef_relative_pos": [0.2, 0, 0.6],
            "arm_qpos": [0.0] * 7, "torso_qpos": [0.0], "goal_distance": 2.0 - x}


class EpisodeDumpTests(unittest.TestCase):
    def test_v2_groups_and_contact_roundtrip(self):
        env = _fake_env(v2=True)
        rec = ed.EpisodeDumpRecorder(env, goal_xy=[2.0, -0.8])
        self.assertEqual(rec.names, ["obstacle_dog"])
        self.assertEqual(rec.obstacle_pos_source, "env._obstacle_geom_groups")
        self.assertEqual(rec.groups["obstacle_dog"]["body_ids"], [2])
        for step in range(3):
            info = {"obstacle_distances": {"obstacle_dog": 0.3 - 0.1 * step},
                    "obstacle_contacts": {"obstacle_dog": step == 2},
                    "min_obstacle_distance": 0.3 - 0.1 * step,
                    "boundary_violated": step >= 1,
                    "obstacle_contact_ever": step == 2, "obstacle_contact_count": int(step == 2)}
            env.sim.data.body_xpos[2] += 0.05  # obstacle gets pushed
            rec.record(step, info, _snapshot(0.1 * step), np.ones(12) * step, np.zeros(12), chunk_id=step // 2,
                       task={"env_success": step == 2})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ed.DUMP_FILENAME
            shapes = rec.save(path, meta={"environment": {"name": "NavigateKitchenDogBlockingRouteB"}})
            d = np.load(path, allow_pickle=False)
            self.assertEqual(d["schema_version"], ed.SCHEMA_VERSION)
            self.assertEqual(d["robot_base_pos"].shape, (3, 3))
            self.assertEqual(d["obstacle_pos"].shape, (3, 1, 3))
            self.assertEqual(d["obstacle_body_quat_wxyz"].shape, (3, 1, 4))
            self.assertEqual(d["obstacle_contact"].tolist(), [[False], [False], [True]])
            self.assertTrue(np.allclose(d["obstacle_contact_force_peak_n"][:, 0], 0.7))
            self.assertEqual(d["boundary_violated"].tolist(), [False, True, True])
            self.assertEqual(d["env_success"].tolist(), [False, False, True])
            self.assertEqual(d["applied_action"].shape, (3, 12))
            self.assertEqual(str(d["obstacle_name"]), "dog")
            self.assertEqual(d["obstacle_names"].tolist(), ["obstacle_dog"])
            # obstacle moved: body pos increases 0.05 per step
            self.assertTrue(np.all(np.diff(d["obstacle_body_pos"][:, 0, 0]) > 0.04))
            # init position = collision-geom centroid before the first push (geoms 2,3 → [7.5, 8.5, 9.5])
            self.assertTrue(np.allclose(d["obstacle_xy_init"], [7.5, 8.5]))
            self.assertTrue(np.allclose(d["obstacle_pos_init"][0], [7.5, 8.5, 9.5]))
            self.assertTrue(np.allclose(d["obstacle_pos"][0, 0], [7.5, 8.5, 9.5]))
            self.assertTrue(np.allclose(d["obstacle_body_pos"][0, 0], [60.05, 70.05, 80.05]))
            self.assertTrue(np.allclose(d["goal_xy_fixture"], [2.0, -0.8]))
            self.assertAlmostEqual(float(d["distance_cap_m"]), 1.5)
            self.assertIn("robocasa_v2", str(d["contact_definition"]))
            meta = json.loads(str(d["meta_json"]))
            self.assertEqual(meta["environment"]["name"], "NavigateKitchenDogBlockingRouteB")
            self.assertEqual(shapes["time_s"], (3,))
            self.assertTrue(np.allclose(d["time_s"], [0.0, 0.05, 0.1]))

    def test_fallback_body_scan_without_v2_api(self):
        env = _fake_env(v2=False)
        rec = ed.EpisodeDumpRecorder(env, goal_xy=[0, 0])
        self.assertEqual(rec.obstacle_pos_source, "body_name_scan")
        self.assertEqual(rec.names, ["obstacle_dog"])
        self.assertEqual(rec.groups["obstacle_dog"]["geom_ids"], [2, 3])
        rec.record(0, {"obstacle_contacts": {"obstacle_dog": True}}, _snapshot(0.0), np.zeros(12), np.zeros(12), 0)
        arrays = rec.arrays()
        self.assertEqual(arrays["obstacle_contact"].tolist(), [[True]])
        self.assertTrue(np.isnan(arrays["obstacle_contact_force_peak_n"][0, 0]))
        self.assertIn("pre-v2", str(arrays["contact_definition"]))

    def test_no_obstacle_env_still_writes(self):
        env = SimpleNamespace(sim=_fake_sim(), control_freq=20)
        env.sim.model.body_id2name = lambda i: {0: "world", 1: "robot0_base"}.get(i)
        rec = ed.EpisodeDumpRecorder(env, goal_xy=[1, 1])
        self.assertEqual(rec.names, [])
        rec.record(0, {}, _snapshot(0.0), np.zeros(12), np.zeros(12), 0)
        with tempfile.TemporaryDirectory() as tmp:
            rec.save(Path(tmp) / "x.npz")
            d = np.load(Path(tmp) / "x.npz")
            self.assertEqual(d["obstacle_pos"].shape, (1, 0, 3))
            self.assertTrue(np.isnan(d["obstacle_xy_init"]).all())
            self.assertEqual(d["robot_base_pos"].shape, (1, 3))

    def test_empty_episode(self):
        rec = ed.EpisodeDumpRecorder(_fake_env(), goal_xy=[0, 0])
        arrays = rec.arrays()
        self.assertEqual(arrays["step"].shape, (0,))
        self.assertEqual(arrays["obstacle_pos"].shape, (0, 1, 3))



class EpisodeDumpSchema2Tests(unittest.TestCase):
    """schema 2 (2026-09-06): per-step task terms for offline arrival-rule re-scoring."""

    def test_task_terms_and_semantics_roundtrip(self):
        env = _fake_env(v2=True)
        rec = ed.EpisodeDumpRecorder(env, goal_xy=[1.0, 0.0])
        for step in range(3):
            task = {"env_success": step == 2, "task_success": step == 2,
                    "collision_free_success": step < 1, "orientation_cos": 0.5 + 0.2 * step,
                    "success_semantics": "task_success"}
            rec.record(step, {}, _snapshot(0.1 * step), np.zeros(12), np.zeros(12), 0, task=task)
        a = rec.arrays()
        self.assertEqual(int(a["schema_version"]), 2)
        self.assertEqual(a["task_success"].tolist(), [False, False, True])
        self.assertEqual(a["collision_free_success"].tolist(), [1, 0, 0])
        self.assertEqual(a["collision_free_success"].dtype, np.int8)
        self.assertTrue(np.allclose(a["ori_cos"], [0.5, 0.7, 0.9]))
        self.assertEqual(str(a["success_semantics"]), "task_success")

    def test_legacy_task_dict_yields_unknown_markers(self):
        env = _fake_env(v2=True)
        rec = ed.EpisodeDumpRecorder(env, goal_xy=[1.0, 0.0])
        rec.record(0, {}, _snapshot(0.0), np.zeros(12), np.zeros(12), 0, task={"env_success": False})
        a = rec.arrays()
        self.assertEqual(a["collision_free_success"].tolist(), [-1])
        self.assertTrue(np.isnan(a["ori_cos"][0]))
        self.assertEqual(str(a["success_semantics"]), "")

    def test_distance_cap_falls_back_when_tree_has_no_boundary(self):
        env = _fake_env(v2=True)
        env.intrusion = {"obstacle_contact_forces": {"obstacle_dog": 0.0}}   # metric-cleanup: boundary 키 없음
        rec = ed.EpisodeDumpRecorder(env, goal_xy=[1.0, 0.0])
        rec.record(0, {}, _snapshot(0.0), np.zeros(12), np.zeros(12), 0, task={})
        a = rec.arrays()
        self.assertTrue(np.isnan(float(a["boundary_threshold_m"])))
        self.assertFalse(a["boundary_violated"][0])
        cap = float(a["distance_cap_m"])   # 호스트(robocasa 없음) → NaN, 컨테이너 → DISTANCE_MEASURE_MAX_M
        self.assertTrue(np.isnan(cap) or cap > 0)


if __name__ == "__main__":
    unittest.main()
