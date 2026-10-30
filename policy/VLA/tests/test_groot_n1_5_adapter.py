"""GR00T N1.5 adapter 의 model-free 계약 회귀 테스트."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from adapters.groot_n1_5 import (  # noqa: E402
    ACTION_KEY_ORDER,
    CAMERA_SLOT_ORDER,
    CAMERA_SLOT_TO_VIDEO_KEY,
    STATE_KEY_ORDER,
    STATE_NATIVE_DIM,
    build_state_dict,
    concat_action_chunk,
    resolve_checkpoint_snapshot,
)

SUBFOLDER = "gr00t_n1-5/multitask_learning/checkpoint-120000"
REVISION = "c484448aba1a9b60a04c9b0ca117241518ea69f3"


def canonical_observation(**overrides):
    observation = {
        "end_effector_position_relative": np.array([0.1, 0.2, 0.3]),
        "end_effector_rotation_relative": np.array([0.0, 0.0, 0.0, 1.0]),
        "gripper_qpos": np.array([0.01, -0.01]),
        "base_position": np.array([1.0, 2.0, 3.0]),
        "base_rotation": np.array([0.0, 0.0, 0.7071, 0.7071]),
    }
    observation.update(overrides)
    return observation


class BuildStateDictTest(unittest.TestCase):
    def test_keys_and_dims(self):
        state = build_state_dict(canonical_observation())
        self.assertEqual(list(state), [key for key, _, _ in STATE_KEY_ORDER])
        self.assertEqual(sum(v.size for v in state.values()), STATE_NATIVE_DIM)
        for key, _, dim in STATE_KEY_ORDER:
            self.assertEqual(state[key].shape, (dim,))
            self.assertEqual(state[key].dtype, np.float64)

    def test_rejects_wrong_dimension(self):
        with self.assertRaises(ValueError):
            build_state_dict(canonical_observation(gripper_qpos=np.zeros(3)))

    def test_rejects_non_finite(self):
        with self.assertRaises(ValueError):
            build_state_dict(canonical_observation(base_position=np.array([0.0, np.nan, 0.0])))

    def test_missing_key_raises(self):
        observation = canonical_observation()
        del observation["base_rotation"]
        with self.assertRaises(KeyError):
            build_state_dict(observation)


class ConcatActionChunkTest(unittest.TestCase):
    @staticmethod
    def _chunk(n=16):
        return {
            key: np.arange(n * dim, dtype=np.float32).reshape(n, dim)
            for key, dim in ACTION_KEY_ORDER
        }

    def test_layout_and_dtype(self):
        flat = concat_action_chunk(self._chunk(), 16)
        self.assertEqual(flat.shape, (16, 12))
        self.assertEqual(flat.dtype, np.float32)
        self.assertTrue(flat.flags["C_CONTIGUOUS"])

    def test_column_offsets_follow_official_action_keys(self):
        offset, expected = 0, {}
        for key, dim in ACTION_KEY_ORDER:
            expected[key] = (offset, offset + dim)
            offset += dim
        self.assertEqual(offset, 12)
        self.assertEqual(expected["action.end_effector_position"], (0, 3))
        self.assertEqual(expected["action.end_effector_rotation"], (3, 6))
        self.assertEqual(expected["action.gripper_close"], (6, 7))
        self.assertEqual(expected["action.base_motion"], (7, 11))
        self.assertEqual(expected["action.control_mode"], (11, 12))

    def test_accepts_one_dimensional_scalar_streams(self):
        chunk = self._chunk()
        chunk["action.gripper_close"] = np.zeros(16, dtype=np.float32)
        self.assertEqual(concat_action_chunk(chunk, 16).shape, (16, 12))

    def test_rejects_wrong_chunk_length(self):
        with self.assertRaises(ValueError):
            concat_action_chunk(self._chunk(8), 16)

    def test_rejects_missing_key(self):
        chunk = self._chunk()
        del chunk["action.base_motion"]
        with self.assertRaises(ValueError):
            concat_action_chunk(chunk, 16)


class CameraSlotTest(unittest.TestCase):
    def test_slots_map_to_official_video_keys(self):
        self.assertEqual(CAMERA_SLOT_ORDER, ("left", "right", "wrist"))
        self.assertEqual(
            [CAMERA_SLOT_TO_VIDEO_KEY[s] for s in CAMERA_SLOT_ORDER],
            [
                "video.robot0_agentview_left",
                "video.robot0_agentview_right",
                "video.robot0_eye_in_hand",
            ],
        )


class ResolveCheckpointSnapshotTest(unittest.TestCase):
    """π0.5 의 `snapshot.name == revision` 검사는 GR00T 에 쓸 수 없다 — subfolder 를 가리킨다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.snapshot = self.root / "snapshots" / REVISION / SUBFOLDER
        self.snapshot.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_accepts_valid_layout(self):
        resolved = resolve_checkpoint_snapshot(str(self.snapshot), REVISION, SUBFOLDER)
        self.assertEqual(resolved, self.snapshot.resolve())

    def test_rejects_wrong_revision(self):
        with self.assertRaises(ValueError):
            resolve_checkpoint_snapshot(str(self.snapshot), "0" * 40, SUBFOLDER)

    def test_rejects_missing_directory(self):
        with self.assertRaises(ValueError):
            resolve_checkpoint_snapshot(str(self.root / "nope"), REVISION, SUBFOLDER)

    def test_rejects_path_outside_snapshots_cache(self):
        stray = self.root / "elsewhere" / REVISION / SUBFOLDER
        stray.mkdir(parents=True)
        with self.assertRaises(ValueError):
            resolve_checkpoint_snapshot(str(stray), REVISION, SUBFOLDER)

    def test_rejects_wrong_subfolder(self):
        other = self.root / "snapshots" / REVISION / "pi05_pretrain_human300/multitask_learning/75000"
        other.mkdir(parents=True)
        with self.assertRaises(ValueError):
            resolve_checkpoint_snapshot(str(other), REVISION, SUBFOLDER)


if __name__ == "__main__":
    unittest.main()
