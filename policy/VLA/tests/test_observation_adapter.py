from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from observation_adapter import ObservationHistory, extract_raw_state  # noqa: E402


def raw_state(value: float = 0.0):
    return {
        "end_effector_position_relative": np.full(3, value, dtype=np.float32),
        "end_effector_rotation_relative": np.array([0, 0, 0, 1], dtype=np.float32),
        "gripper_qpos": np.full(2, value, dtype=np.float32),
        "base_position": np.full(3, value, dtype=np.float32),
        "base_rotation": np.array([0, 0, 0, 1], dtype=np.float32),
    }


class ObservationAdapterTest(unittest.TestCase):
    def test_official_history_sampling_and_initial_padding(self):
        history = ObservationHistory(length=4, interval=2, camera_slots=["left", "right", "wrist"])
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        history.append(0, {slot: image for slot in ("left", "right", "wrist")}, raw_state())
        self.assertEqual(history.sample().steps, [0, 0, 0, 0])

        for step in range(1, 8):
            value = np.full_like(image, step)
            history.append(
                step,
                {slot: value for slot in ("left", "right", "wrist")},
                raw_state(float(step)),
            )
        sample = history.sample()
        self.assertEqual(sample.steps, [1, 3, 5, 7])
        self.assertEqual(sample.images["left"].shape, (4, 2, 2, 3))

    def test_strict_local_key_mapping(self):
        observation = {
            "robot0_base_to_eef_pos": np.arange(3),
            "robot0_base_to_eef_quat": np.array([0, 0, 0, 1]),
            "robot0_gripper_qpos": np.arange(2),
            "robot0_base_pos": np.arange(3),
            "robot0_base_quat": np.array([0, 0, 0, 1]),
        }
        result = extract_raw_state(observation)
        self.assertEqual(result["end_effector_position_relative"].shape, (3,))
        self.assertEqual(result["end_effector_rotation_relative"].shape, (4,))
        with self.assertRaises(KeyError):
            extract_raw_state({"robot0_eef_pos": np.arange(3)})
        site_only = dict(observation)
        site_only["robot0_base_to_eef_quat_site"] = site_only.pop("robot0_base_to_eef_quat")
        with self.assertRaises(KeyError):
            extract_raw_state(site_only)


if __name__ == "__main__":
    unittest.main()
