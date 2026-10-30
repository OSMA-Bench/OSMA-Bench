from __future__ import annotations

import sys
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from action_adapter import (  # noqa: E402
    inspect_hybrid_action_space,
    lerobot_base_first_to_local_hybrid,
    robocasa_ee_first_to_local_hybrid,
    to_local_hybrid,
    validate_action_chunk,
)


class ActionAdapterTest(unittest.TestCase):
    def make_env(self):
        controller = SimpleNamespace(
            name="HYBRID_MOBILE_BASE",
            _action_split_indexes=OrderedDict(
                right=(0, 6), right_gripper=(6, 7), base=(7, 10), torso=(10, 11)
            ),
        )
        return SimpleNamespace(
            action_spec=(np.full(12, -1.0), np.full(12, 1.0)),
            robots=[SimpleNamespace(composite_controller=controller)],
        )

    def test_local_contract_and_official_gym_unmapping(self):
        info = inspect_hybrid_action_space(self.make_env())
        self.assertEqual(info["action_dim"], 12)
        action = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
        action[6] = 0.49
        action[11] = 0.5
        expected = action.copy()
        expected[6] = -1.0
        expected[11] = 1.0
        np.testing.assert_array_equal(robocasa_ee_first_to_local_hybrid(action), expected)

    def test_chunk_rejects_bad_shape_nan_and_range(self):
        low = np.full(12, -1.0)
        high = np.full(12, 1.0)
        with self.assertRaises(ValueError):
            validate_action_chunk(np.zeros((2, 11)), low, high)
        chunk = np.zeros((2, 12))
        chunk[0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_action_chunk(chunk, low, high)

    def test_lerobot_base_first_permutation_and_binary_unmapping(self):
        source = np.arange(12, dtype=np.float32) / 10.0
        source[4] = 0.49
        source[11] = 0.5
        expected = source[[5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]].copy()
        expected[6] = 1.0
        expected[11] = -1.0
        np.testing.assert_array_equal(
            lerobot_base_first_to_local_hybrid(source), expected
        )
        np.testing.assert_array_equal(
            to_local_hybrid(source, "lerobot_base_first"), expected
        )

    def test_base_first_chunk_is_range_checked_after_permutation(self):
        low = np.full(12, -1.0)
        high = np.full(12, 1.0)
        chunk = np.zeros((1, 12), dtype=np.float32)
        chunk[0, 0] = 1.1
        with self.assertRaises(ValueError):
            validate_action_chunk(
                chunk, low, high, action_convention="lerobot_base_first"
            )
        chunk[0, 0] = 1.1
        with self.assertRaises(ValueError):
            validate_action_chunk(chunk, low, high)

    def test_chunk_range_check_applies_official_binary_unmapping_first(self):
        low = np.full(12, -1.0)
        high = np.full(12, 1.0)
        chunk = np.zeros((2, 12), dtype=np.float32)
        chunk[0, 6] = -0.0078125
        chunk[0, 11] = 1.0078125
        chunk[1, 6] = 1.1
        chunk[1, 11] = -0.1

        validated = validate_action_chunk(chunk, low, high)

        np.testing.assert_array_equal(validated, chunk)
        np.testing.assert_array_equal(
            robocasa_ee_first_to_local_hybrid(validated[0])[[6, 11]],
            np.array([-1.0, 1.0], dtype=np.float32),
        )

    def test_chunk_accepts_only_one_bfloat16_ulp_at_continuous_boundary(self):
        low = np.full(12, -1.0)
        high = np.full(12, 1.0)
        chunk = np.zeros((1, 12), dtype=np.float32)
        chunk[0, 8] = 1.0078125

        validated = validate_action_chunk(chunk, low, high, tolerance=0.008)

        self.assertEqual(float(validated[0, 8]), 1.0078125)
        chunk[0, 8] = 1.01
        with self.assertRaises(ValueError):
            validate_action_chunk(chunk, low, high, tolerance=0.008)

    def test_official_passthrough_audits_without_clipping_or_rejecting(self):
        low = np.full(12, -1.0)
        high = np.full(12, 1.0)
        chunk = np.zeros((1, 12), dtype=np.float32)
        chunk[0, 8] = 1.1

        validated = validate_action_chunk(chunk, low, high, enforce_bounds=False)

        self.assertEqual(float(validated[0, 8]), float(chunk[0, 8]))


if __name__ == "__main__":
    unittest.main()
