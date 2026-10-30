from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from adapters.pi05_opentau import build_state16, pad_state32, resolve_hf_snapshot  # noqa: E402


class Pi05OpenTauAdapterTest(unittest.TestCase):
    def test_official_base_first_state_and_padding(self):
        raw = {
            "base_position": [1, 2, 3],
            "base_rotation": [4, 5, 6, 7],
            "end_effector_position_relative": [8, 9, 10],
            "end_effector_rotation_relative": [11, 12, 13, 14],
            "gripper_qpos": [15, 16],
        }
        native = build_state16(raw)
        np.testing.assert_array_equal(native, np.arange(1, 17, dtype=np.float32))
        padded = pad_state32(native)
        self.assertEqual(padded.shape, (32,))
        np.testing.assert_array_equal(padded[:16], native)
        self.assertFalse(np.any(padded[16:]))

    def test_state_rejects_nonfinite_values(self):
        raw = {
            "base_position": [np.nan, 0, 0],
            "base_rotation": [0, 0, 0, 1],
            "end_effector_position_relative": [0, 0, 0],
            "end_effector_rotation_relative": [0, 0, 0, 1],
            "gripper_qpos": [0, 0],
        }
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            build_state16(raw)

    def test_offline_snapshot_resolution_is_revision_exact(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "models--org--model" / "snapshots" / ("a" * 40)
            expected.mkdir(parents=True)
            self.assertEqual(
                resolve_hf_snapshot(root, "org/model", "a" * 40), expected
            )
            with self.assertRaises(FileNotFoundError):
                resolve_hf_snapshot(root, "org/model", "b" * 40)


if __name__ == "__main__":
    unittest.main()
