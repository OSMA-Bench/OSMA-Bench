"""π0.5-Human300 adapter 회귀 테스트 (model-free).

openpi 가 설치돼 있지 않아도 도는 부분만 여기서 본다. 공식 원문과의 대조는
`test_pi05_human300_golden_payload.py` 가 담당한다.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from adapters.pi05_human300 import (  # noqa: E402
    ACTION_DIM,
    CAMERA_SLOT_ORDER,
    CAMERA_SLOT_TO_ELEMENT_KEY,
    CAMERA_SLOT_TO_VIDEO_KEY,
    MODEL_ACTION_DIM,
    STATE_KEY_ORDER,
    STATE_NATIVE_DIM,
    STATE_PAD_DIM,
    build_state16,
    resolve_checkpoint_snapshot,
)
from model_config import load_model_profile  # noqa: E402


def _observation(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "end_effector_position_relative": rng.normal(size=3),
        "end_effector_rotation_relative": rng.normal(size=4),
        "base_position": rng.normal(size=3),
        "base_rotation": rng.normal(size=4),
        "gripper_qpos": rng.normal(size=2),
    }


class BuildState16Test(unittest.TestCase):
    def test_layout_is_ee_first(self):
        """⛔ EE-first. OpenTau(base-first)와 뒤바뀌면 차원이 같아 조용히 통과한다."""
        rng = np.random.default_rng(20260824)
        observation = _observation(rng)
        state = build_state16(observation)

        self.assertEqual(state.shape, (STATE_NATIVE_DIM,))
        self.assertEqual(state.dtype, np.float32)
        np.testing.assert_allclose(state[0:3], observation["end_effector_position_relative"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(state[3:7], observation["end_effector_rotation_relative"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(state[7:10], observation["base_position"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(state[10:14], observation["base_rotation"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(state[14:16], observation["gripper_qpos"], rtol=0, atol=1e-6)

    def test_does_not_pad(self):
        """openpi 의 `RobocasaInputs` 가 32D 로 pad 한다. adapter 가 또 하면 안 된다."""
        state = build_state16(_observation(np.random.default_rng(1)))
        self.assertEqual(state.shape[0], STATE_NATIVE_DIM)
        self.assertNotEqual(state.shape[0], STATE_PAD_DIM)

    def test_key_order_dims_sum_to_16(self):
        self.assertEqual(sum(dim for _, _, dim in STATE_KEY_ORDER), STATE_NATIVE_DIM)

    def test_rejects_wrong_shape(self):
        observation = _observation(np.random.default_rng(2))
        observation["base_rotation"] = np.zeros(3)
        with self.assertRaises(ValueError):
            build_state16(observation)

    def test_rejects_non_finite(self):
        observation = _observation(np.random.default_rng(3))
        observation["gripper_qpos"] = np.array([np.nan, 0.0])
        with self.assertRaises(ValueError):
            build_state16(observation)

    def test_rejects_missing_key(self):
        observation = _observation(np.random.default_rng(4))
        del observation["base_position"]
        with self.assertRaises(KeyError):
            build_state16(observation)


class CameraSlotTest(unittest.TestCase):
    def test_slot_order_is_left_wrist_right(self):
        """XR-1 · GR00T 의 left/right/wrist 와 다르다."""
        self.assertEqual(CAMERA_SLOT_ORDER, ("left", "wrist", "right"))

    def test_element_and_video_keys(self):
        self.assertEqual(
            [CAMERA_SLOT_TO_ELEMENT_KEY[s] for s in CAMERA_SLOT_ORDER],
            ["observation/image", "observation/wrist_image", "observation/right_image"],
        )
        self.assertEqual(
            [CAMERA_SLOT_TO_VIDEO_KEY[s] for s in CAMERA_SLOT_ORDER],
            [
                "video.robot0_agentview_left",
                "video.robot0_eye_in_hand",
                "video.robot0_agentview_right",
            ],
        )


class SnapshotResolutionTest(unittest.TestCase):
    REV = "c484448aba1a9b60a04c9b0ca117241518ea69f3"
    SUB = "pi05_pretrain_human300/multitask_learning/75000"

    def _make(self, root: Path, *, params=True, assets=True, norm=True) -> Path:
        snapshot = root / "snapshots" / self.REV / self.SUB
        snapshot.mkdir(parents=True)
        if params:
            (snapshot / "params").mkdir()
        if assets:
            (snapshot / "assets").mkdir()
            if norm:
                (snapshot / "assets" / "norm_stats.json").write_text("{}", encoding="utf-8")
        return snapshot

    def test_accepts_valid_snapshot(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            snapshot = self._make(Path(tmp))
            self.assertEqual(
                resolve_checkpoint_snapshot(str(snapshot), self.REV, self.SUB),
                snapshot.resolve(),
            )

    def test_rejects_missing_norm_stats(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            snapshot = self._make(Path(tmp), norm=False)
            with self.assertRaises(ValueError):
                resolve_checkpoint_snapshot(str(snapshot), self.REV, self.SUB)

    def test_rejects_missing_params(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            snapshot = self._make(Path(tmp), params=False)
            with self.assertRaises(ValueError):
                resolve_checkpoint_snapshot(str(snapshot), self.REV, self.SUB)

    def test_rejects_wrong_subfolder(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            snapshot = self._make(Path(tmp))
            with self.assertRaises(ValueError):
                resolve_checkpoint_snapshot(
                    str(snapshot), self.REV, "gr00t_n1-5/multitask_learning/checkpoint-120000"
                )


class ProfileTest(unittest.TestCase):
    def setUp(self):
        self.profile = load_model_profile("pi05_human300")

    def test_chunk_and_replan_are_separate(self):
        """이 패널에서 유일하게 chunk != replan 이다 (50:5, open-loop 0.25s)."""
        self.assertEqual(self.profile["action_chunk_size"], 50)
        self.assertEqual(self.profile["replan_steps"], 5)
        self.assertLess(self.profile["replan_steps"], self.profile["action_chunk_size"])

    def test_contract_fields(self):
        self.assertEqual(self.profile["action_dim"], ACTION_DIM)
        self.assertEqual(self.profile["action_convention"], "robocasa_ee_first")
        self.assertEqual(list(self.profile["camera_slots"]), list(CAMERA_SLOT_ORDER))
        self.assertEqual(self.profile["image_size"], 256)
        self.assertEqual(list(self.profile["resize_target"]), [224, 224])
        self.assertEqual(self.profile["history_length"], 1)
        self.assertEqual(self.profile["state_dim_native"], STATE_NATIVE_DIM)
        self.assertEqual(self.profile["state_dim_padded"], STATE_PAD_DIM)
        self.assertEqual(
            self.profile["checkpoint_subfolder"],
            "pi05_pretrain_human300/multitask_learning/75000",
        )

    def test_input_audit_matches_contract(self):
        audit = self.profile["input_audit"]
        self.assertEqual(audit["image_history_shape"], [1, 256, 256, 3])
        self.assertEqual(audit["state_native_shape"], [1, STATE_NATIVE_DIM])
        self.assertEqual(audit["policy_state_shape"], [1, STATE_PAD_DIM])

    def test_model_action_dim_is_32(self):
        self.assertEqual(MODEL_ACTION_DIM, 32)


if __name__ == "__main__":
    unittest.main()
