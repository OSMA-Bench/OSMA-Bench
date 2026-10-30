from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from adapters.xr1 import build_state14, pad_state_history, quat_xyzw_to_axis_angle  # noqa: E402


class XR1AdapterTest(unittest.TestCase):
    def test_axis_angle_and_official_state_order(self):
        half = np.pi / 4
        quaternion = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
        np.testing.assert_allclose(
            quat_xyzw_to_axis_angle(quaternion),
            [0.0, 0.0, np.pi / 2],
            rtol=1e-6,
            atol=1e-6,
        )
        raw = {
            "end_effector_position_relative": [1, 2, 3],
            "end_effector_rotation_relative": [0, 0, 0, 1],
            "gripper_qpos": [4, 5],
            "base_position": [6, 7, 8],
            "base_rotation": quaternion,
        }
        expected = np.array([1, 2, 3, 0, 0, 0, 4, 5, 6, 7, 8, 0, 0, np.pi / 2])
        np.testing.assert_allclose(build_state14(raw), expected, rtol=1e-6, atol=1e-6)

    def test_state_padding(self):
        raw = {
            "end_effector_position_relative": [0, 0, 0],
            "end_effector_rotation_relative": [0, 0, 0, 1],
            "gripper_qpos": [0, 0],
            "base_position": [0, 0, 0],
            "base_rotation": [0, 0, 0, 1],
        }
        native, padded = pad_state_history([raw] * 4)
        self.assertEqual(native.shape, (4, 14))
        self.assertEqual(padded.shape, (1, 4, 60))
        np.testing.assert_array_equal(padded[0, :, :14], native)
        self.assertFalse(np.any(padded[0, :, 14:]))


if __name__ == "__main__":
    unittest.main()
