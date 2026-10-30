from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from base_diagnostics import make_hybrid_action, world_delta_to_body_action, wrap_angle  # noqa: E402


class BaseDiagnosticsTest(unittest.TestCase):
    def test_world_delta_to_body_uses_current_yaw(self):
        np.testing.assert_allclose(world_delta_to_body_action([1.0, 0.0], 0.0), [1.0, 0.0])
        np.testing.assert_allclose(
            world_delta_to_body_action([0.0, 1.0], np.pi / 2), [1.0, 0.0], atol=1e-7
        )

    def test_hybrid_action_only_activates_base_and_binary_channels(self):
        env = SimpleNamespace(action_spec=(np.full(12, -1.0), np.full(12, 1.0)))
        action = make_hybrid_action(env, [0.2, -0.3, 0.4])

        np.testing.assert_allclose(action[7:10], [0.2, -0.3, 0.4])
        self.assertEqual(action[6], -1.0)
        self.assertEqual(action[11], 1.0)
        np.testing.assert_array_equal(action[:6], np.zeros(6))
        self.assertEqual(action[10], 0.0)

    def test_wrap_angle_uses_shortest_difference(self):
        self.assertAlmostEqual(wrap_angle(3 * np.pi), np.pi)


if __name__ == "__main__":
    unittest.main()
