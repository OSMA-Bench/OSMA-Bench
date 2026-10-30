from __future__ import annotations

import sys
import base64
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from policy_client import PolicyClient  # noqa: E402


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class PolicyHTTPTest(unittest.TestCase):
    def test_reset_sends_policy_seed(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse({"ok": True, "policy_seed": 7})

        with patch("policy_client.urllib.request.urlopen", side_effect=fake_urlopen):
            client = PolicyClient("http://127.0.0.1:8100", timeout_seconds=2)
            response = client.reset("navigate", "xr1", policy_seed=7)

        self.assertEqual(captured["payload"]["policy_seed"], 7)
        self.assertEqual(response["policy_seed"], 7)

    def test_act_serializes_lossless_arrays_and_decodes_response(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(
                {
                    "actions": [[0.0] * 12, [0.1] * 12],
                    "metadata": {"chunk_length": 2},
                    "input_audit": {"ok": True},
                }
            )

        with patch("policy_client.urllib.request.urlopen", side_effect=fake_urlopen):
            client = PolicyClient("http://127.0.0.1:8100", timeout_seconds=2)
            image = np.zeros((4, 2, 2, 3), dtype=np.uint8)
            image[3, 1, 1, 2] = 255
            state = {
                "end_effector_position_relative": np.zeros(3),
                "end_effector_rotation_relative": np.array([0, 0, 0, 1]),
                "gripper_qpos": np.zeros(2),
                "base_position": np.zeros(3),
                "base_rotation": np.array([0, 0, 0, 1]),
            }
            response = client.act(
                images={slot: image for slot in ("left", "right", "wrist")},
                state_raw=[state] * 4,
                instruction="navigate",
                step_id=0,
                history_steps=[0, 0, 0, 0],
            )
            self.assertEqual(response.actions.shape, (2, 12))
            self.assertEqual(captured["url"], "http://127.0.0.1:8100/act")
            encoded = captured["payload"]["images"]["left"]
            self.assertEqual(encoded["shape"], [4, 2, 2, 3])
            decoded = np.frombuffer(base64.b64decode(encoded["data_b64"]), dtype=np.uint8).reshape(
                encoded["shape"]
            )
            np.testing.assert_array_equal(decoded, image)


if __name__ == "__main__":
    unittest.main()
