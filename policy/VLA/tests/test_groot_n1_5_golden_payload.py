"""G0 golden payload for GR00T N1.5 against frozen official sources.

GR00T adapter 는 공식 전처리를 전사하지 않고 `Gr00tPolicy` 를 그대로 호출한다. 따라서 G0 의
역할이 XR-1/π0.5 와 다르다 — "전사가 맞나" 가 아니라 다음 둘을 본다.

  1. 우리가 만든 obs dict 가 공식 `PandaOmronKeyConverter.map_obs_in_eval()` 과 같은가
  2. **설치된/고정된** `PandaOmronDataConfig` 의 key 순서가 adapter 선언과 같은가 (upstream drift)

계약 단일 출처: README.md § "Model contracts" (GR00T N1.5)
"""

from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from repo_paths import resolve_paths  # noqa: E402

REF_ROOT = resolve_paths().official_ref_root  # official reference copies (OSMA_BENCH_OFFICIAL_REF_ROOT)

GROOT_REF = REF_ROOT / "groot_official_0821"
DATA_CONFIG = GROOT_REF / "data_config.py"
EMBODIMENT = GROOT_REF / "embodiment_tags.py"
CKPT_METADATA = GROOT_REF / "checkpoint_metadata.json"
WRAPPER = REF_ROOT / "robocasa_official_0819/gym_wrapper.py"


def _literal(node: ast.AST) -> Any:
    """리터럴, 그리고 공식 소스가 실제로 쓰는 `list(range(N))` 만 평가한다."""
    try:
        return ast.literal_eval(node)
    except ValueError:
        pass
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "list"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Name)
        and node.args[0].func.id == "range"
    ):
        return list(range(*[ast.literal_eval(a) for a in node.args[0].args]))
    raise ValueError(f"Unsupported attribute expression at line {getattr(node, 'lineno', '?')}")


def _class_attributes(path: Path, class_name: str) -> dict[str, Any]:
    """공식 소스를 실행하지 않고 class 본문의 attribute 만 읽는다."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            attrs: dict[str, Any] = {}
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            attrs[target.id] = _literal(item.value)
            return attrs
    raise AssertionError(f"{class_name} not found in {path}")


def _extract(path: Path, names: set[str], namespace: dict[str, Any]) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [
        node
        for node in tree.body
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        )
    ]
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - frozen source audit
    return namespace


@unittest.skipUnless(
    DATA_CONFIG.exists() and EMBODIMENT.exists() and CKPT_METADATA.exists() and WRAPPER.exists(),
    "공식 참조본이 없다",
)
class GrootOfficialGoldenPayloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _class_attributes(DATA_CONFIG, "PandaOmronDataConfig")
        cls.converter = _extract(WRAPPER, {"PandaOmronKeyConverter"}, {"np": np})[
            "PandaOmronKeyConverter"
        ]
        cls.metadata = json.loads(CKPT_METADATA.read_text(encoding="utf-8"))

    # ---- 1. obs dict 가 공식 map_obs_in_eval 과 같은가 -------------------------------

    def test_state_dict_matches_official_map_obs_in_eval(self):
        from adapters.groot_n1_5 import build_state_dict

        rng = np.random.default_rng(20260821)
        for _ in range(200):
            observation = {
                "robot0_base_pos": rng.normal(size=3).astype(np.float32),
                "robot0_base_quat": rng.normal(size=4).astype(np.float32),
                "robot0_base_to_eef_pos": rng.normal(size=3).astype(np.float32),
                "robot0_base_to_eef_quat": rng.normal(size=4).astype(np.float32),
                "robot0_gripper_qpos": rng.normal(size=2).astype(np.float32),
            }
            expected = self.converter.map_obs_in_eval(observation)
            canonical = {
                "base_position": observation["robot0_base_pos"],
                "base_rotation": observation["robot0_base_quat"],
                "end_effector_position_relative": observation["robot0_base_to_eef_pos"],
                "end_effector_rotation_relative": observation["robot0_base_to_eef_quat"],
                "gripper_qpos": observation["robot0_gripper_qpos"],
            }
            produced = build_state_dict(canonical)
            self.assertEqual(set(produced), set(expected))
            for key in expected:
                np.testing.assert_array_equal(produced[key], np.asarray(expected[key], np.float64))

    def test_quaternion_is_passed_through_unreordered(self):
        """⛔ xyzw -> wxyz 로 "고치면" 안 된다. 공식 파이프라인이 그대로 넣어 학습했다."""
        from adapters.groot_n1_5 import build_state_dict

        quat = np.array([0.1, 0.2, 0.3, 0.9])
        produced = build_state_dict(
            {
                "base_position": np.zeros(3),
                "base_rotation": quat,
                "end_effector_position_relative": np.zeros(3),
                "end_effector_rotation_relative": quat,
                "gripper_qpos": np.zeros(2),
            }
        )
        np.testing.assert_array_equal(produced["state.base_rotation"], quat)
        np.testing.assert_array_equal(produced["state.end_effector_rotation_relative"], quat)

    # ---- 2. adapter 선언 == 고정된 공식 DataConfig ----------------------------------

    def test_video_key_order_matches_official(self):
        from adapters.groot_n1_5 import CAMERA_SLOT_ORDER, CAMERA_SLOT_TO_VIDEO_KEY

        self.assertEqual(
            [CAMERA_SLOT_TO_VIDEO_KEY[slot] for slot in CAMERA_SLOT_ORDER],
            self.cfg["video_keys"],
        )

    def test_state_key_order_matches_official(self):
        from adapters.groot_n1_5 import STATE_KEY_ORDER

        self.assertEqual([key for key, _, _ in STATE_KEY_ORDER], self.cfg["state_keys"])

    def test_action_key_order_matches_official(self):
        from adapters.groot_n1_5 import ACTION_KEY_ORDER

        self.assertEqual([key for key, _ in ACTION_KEY_ORDER], self.cfg["action_keys"])

    def test_language_key_and_horizons_match_official(self):
        from adapters.groot_n1_5 import LANGUAGE_KEY

        self.assertEqual(self.cfg["language_keys"], [LANGUAGE_KEY])
        self.assertEqual(self.cfg["observation_indices"], [0])
        self.assertEqual(len(self.cfg["action_indices"]), 16)

    def test_rotation_targets_and_binary_actions_match_official(self):
        self.assertEqual(
            self.cfg["state_target_rotations"],
            {
                "state.end_effector_rotation_relative": "rotation_6d",
                "state.base_rotation": "rotation_6d",
            },
        )
        binary = {k for k, v in self.cfg["action_normalization_modes"].items() if v == "binary"}
        self.assertEqual(binary, {"action.gripper_close", "action.control_mode"})

    def test_transformed_state_dim_is_twenty(self):
        """quaternion 2개가 rotation_6d 로 바뀌어 16D 가 아니라 20D 가 된다."""
        from adapters.groot_n1_5 import (
            STATE_KEY_ORDER,
            STATE_NATIVE_DIM,
            STATE_TRANSFORMED_DIM,
        )

        raw = sum(dim for _, _, dim in STATE_KEY_ORDER)
        rotations = self.cfg["state_target_rotations"]
        transformed = sum(
            6 if key in rotations else dim for key, _, dim in STATE_KEY_ORDER
        )
        self.assertEqual(raw, STATE_NATIVE_DIM)
        self.assertEqual(transformed, STATE_TRANSFORMED_DIM)
        self.assertEqual(transformed, 20)

    # ---- 3. embodiment tag ---------------------------------------------------------

    def test_embodiment_tag_is_new_embodiment(self):
        from adapters.groot_n1_5 import EMBODIMENT_TAG

        ns = _extract(EMBODIMENT, {"EmbodimentTag", "EMBODIMENT_TAG_MAPPING"}, {"Enum": __import__("enum").Enum})
        values = {member.value for member in ns["EmbodimentTag"]}
        self.assertIn(EMBODIMENT_TAG, values)
        # 문서 정정: 공식 enum 에 ROBOCASA_PANDA_OMRON 은 존재하지 않는다.
        self.assertNotIn("robocasa_panda_omron", values)
        self.assertEqual(ns["EMBODIMENT_TAG_MAPPING"][EMBODIMENT_TAG], 31)

    def test_checkpoint_metadata_declares_exactly_that_embodiment(self):
        from adapters.groot_n1_5 import EMBODIMENT_TAG

        self.assertEqual(list(self.metadata), [EMBODIMENT_TAG])
        self.assertEqual(self.metadata[EMBODIMENT_TAG]["embodiment_tag"], EMBODIMENT_TAG)

    def test_metadata_key_order_is_not_the_concat_order(self):
        """metadata.json 은 알파벳순이라 base-first 로 보인다. 순서 근거로 쓰면 안 된다."""
        meta_state = list(self.metadata["new_embodiment"]["modalities"]["state"])
        config_state = [key.split(".", 1)[1] for key in self.cfg["state_keys"]]
        self.assertNotEqual(meta_state, config_state)
        self.assertEqual(sorted(meta_state), sorted(config_state))

    # ---- 4. action concat == local flat 12D, binary unmap 동치 ----------------------

    def test_concat_action_chunk_builds_local_flat_layout(self):
        from adapters.groot_n1_5 import concat_action_chunk

        chunk = 16
        action_dict = {
            "action.end_effector_position": np.tile([1.0, 2.0, 3.0], (chunk, 1)),
            "action.end_effector_rotation": np.tile([4.0, 5.0, 6.0], (chunk, 1)),
            "action.gripper_close": np.zeros((chunk, 1)),
            "action.base_motion": np.tile([7.0, 8.0, 9.0, 10.0], (chunk, 1)),
            "action.control_mode": np.ones((chunk, 1)),
        }
        flat = concat_action_chunk(action_dict, chunk)
        self.assertEqual(flat.shape, (chunk, 12))
        np.testing.assert_array_equal(flat[0, 0:3], [1.0, 2.0, 3.0])   # EE pos
        np.testing.assert_array_equal(flat[0, 3:6], [4.0, 5.0, 6.0])   # EE rot
        self.assertEqual(flat[0, 6], 0.0)                              # gripper
        np.testing.assert_array_equal(flat[0, 7:10], [7.0, 8.0, 9.0])  # base
        self.assertEqual(flat[0, 10], 10.0)                            # torso
        self.assertEqual(flat[0, 11], 1.0)                             # control mode

    def test_binary_zero_one_unmaps_the_same_as_official_gym_wrapper(self):
        """GR00T binary inverse 는 정확히 0/1 을 낸다. 공식 0.5 임계 unmap 과 동치여야 한다."""
        from action_adapter import to_local_hybrid

        for gripper in (0.0, 1.0):
            for mode in (0.0, 1.0):
                flat = np.zeros(12, dtype=np.float32)
                flat[6] = gripper
                flat[11] = mode
                ours = to_local_hybrid(flat, "robocasa_ee_first")
                official = self.converter.unmap_action(
                    {
                        "action.end_effector_position": flat[0:3],
                        "action.end_effector_rotation": flat[3:6],
                        "action.gripper_close": flat[6],
                        "action.base_motion": flat[7:11],
                        "action.control_mode": flat[11],
                    }
                )
                self.assertEqual(float(ours[6]), float(official["robot0_right_gripper"]))
                self.assertEqual(float(ours[11]), float(official["robot0_base_mode"]))


def _installed_data_config():
    """gr00t 가 설치돼 있으면 공식 DataConfig 객체를, 없으면 None."""
    try:
        from gr00t.experiment.data_config import DATA_CONFIG_MAP
    except Exception:  # noqa: BLE001 - 서버 venv 밖에서는 없는 것이 정상
        return None
    return DATA_CONFIG_MAP.get("panda_omron")


@unittest.skipUnless(
    _installed_data_config() is not None and CKPT_METADATA.exists(),
    "gr00t 가 설치된 서버 venv 에서만 실행한다",
)
class GrootInstalledRuntimeTest(unittest.TestCase):
    """freeze 사본이 아니라 **실제로 설치된** gr00t 로 계약을 확인한다 (upstream drift gate)."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = _installed_data_config()

    def test_installed_key_orders_match_adapter(self):
        from adapters.groot_n1_5 import (
            ACTION_KEY_ORDER,
            CAMERA_SLOT_ORDER,
            CAMERA_SLOT_TO_VIDEO_KEY,
            LANGUAGE_KEY,
            STATE_KEY_ORDER,
        )

        self.assertEqual(
            list(self.cfg.video_keys),
            [CAMERA_SLOT_TO_VIDEO_KEY[slot] for slot in CAMERA_SLOT_ORDER],
        )
        self.assertEqual(list(self.cfg.state_keys), [k for k, _, _ in STATE_KEY_ORDER])
        self.assertEqual(list(self.cfg.action_keys), [k for k, _ in ACTION_KEY_ORDER])
        self.assertEqual(list(self.cfg.language_keys), [LANGUAGE_KEY])
        self.assertEqual(list(self.cfg.observation_indices), [0])

    def test_installed_embodiment_mapping(self):
        from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag

        from adapters.groot_n1_5 import EMBODIMENT_TAG

        self.assertEqual(EMBODIMENT_TAG_MAPPING[EMBODIMENT_TAG], 31)
        self.assertNotIn("robocasa_panda_omron", {m.value for m in EmbodimentTag})

    def test_official_transform_produces_20d_state_in_64d_slot(self):
        """raw 16D -> rotation_6d 변환 후 20D -> max_state_dim 64 padding."""
        from gr00t.data.schema import DatasetMetadata

        from adapters.groot_n1_5 import (
            CAMERA_SLOT_ORDER,
            CAMERA_SLOT_TO_VIDEO_KEY,
            LANGUAGE_KEY,
            STATE_PAD_DIM,
            STATE_TRANSFORMED_DIM,
        )

        transform = self.cfg.transform()
        transform.eval()  # Gr00tPolicy.__init__ 과 동일. CenterCrop, ColorJitter 없음
        # ⚠ composed 객체의 training 은 True 로 남는다 — 자식 transform 을 봐야 한다.
        self.assertTrue(transform.transforms)
        self.assertEqual(
            [type(c).__name__ for c in transform.transforms if getattr(c, "training", False)], []
        )
        metadata = json.loads(CKPT_METADATA.read_text(encoding="utf-8"))["new_embodiment"]
        transform.set_metadata(DatasetMetadata.model_validate(metadata))

        rng = np.random.default_rng(20260821)

        def run(ee_quat, base_quat):
            observation = {
                "state.end_effector_position_relative": rng.normal(size=(1, 1, 3)),
                "state.end_effector_rotation_relative": np.asarray(ee_quat).reshape(1, 1, 4),
                "state.gripper_qpos": rng.normal(size=(1, 1, 2)),
                "state.base_position": rng.normal(size=(1, 1, 3)),
                "state.base_rotation": np.asarray(base_quat).reshape(1, 1, 4),
                LANGUAGE_KEY: np.array([["navigate to the sink"]]),
            }
            for slot in CAMERA_SLOT_ORDER:
                observation[CAMERA_SLOT_TO_VIDEO_KEY[slot]] = rng.integers(
                    0, 256, size=(1, 1, 256, 256, 3), dtype=np.uint8
                )
            out = transform(observation)
            state = np.asarray(out["state"])
            self.assertEqual(state.shape, (1, 1, STATE_PAD_DIM))
            return state.reshape(-1), out

        # (a) 일반 회전: 마지막 유효 slot(index 19)까지 실제로 값이 실린다
        generic, out = run(rng.normal(size=4), rng.normal(size=4))
        nonzero = np.flatnonzero(generic != 0)
        self.assertEqual(int(nonzero.max()) + 1, STATE_TRANSFORMED_DIM)
        np.testing.assert_allclose(generic[STATE_TRANSFORMED_DIM:], 0.0)
        # 이미지는 카메라 3개 x 224x224
        self.assertEqual(tuple(np.asarray(out["eagle_pixel_values"]).shape), (3, 3, 224, 224))

        # (b) 항등 회전: rotation_6d 정규화가 identity 임을 보인다
        identity = [1.0, 0.0, 0.0, 0.0]
        flat, _ = run(identity, identity)
        np.testing.assert_allclose(flat[3:9], [1, 0, 0, 0, 1, 0], atol=1e-6)
        np.testing.assert_allclose(flat[14:20], [1, 0, 0, 0, 1, 0], atol=1e-6)
        np.testing.assert_allclose(flat[STATE_TRANSFORMED_DIM:], 0.0)

    def test_official_unapply_returns_local_flat_twelve(self):
        import torch

        from gr00t.data.schema import DatasetMetadata

        from adapters.groot_n1_5 import concat_action_chunk

        transform = self.cfg.transform()
        transform.eval()
        metadata = json.loads(CKPT_METADATA.read_text(encoding="utf-8"))["new_embodiment"]
        transform.set_metadata(DatasetMetadata.model_validate(metadata))

        back = transform.unapply({"action": torch.zeros(1, 16, 32)})
        chunk = concat_action_chunk({k: np.asarray(v)[0] for k, v in back.items()}, 16)
        self.assertEqual(chunk.shape, (16, 12))
        # binary 역변환은 정확히 0/1 만 낸다
        self.assertEqual(set(np.unique(chunk[:, 6]).tolist()), {0.0})
        self.assertEqual(set(np.unique(chunk[:, 11]).tolist()), {0.0})


if __name__ == "__main__":
    unittest.main()
