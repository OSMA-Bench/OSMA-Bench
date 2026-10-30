"""G0 golden payload against frozen OpenTau v0.13.0 and RoboCasa sources."""

from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from repo_paths import resolve_paths  # noqa: E402

REF_ROOT = resolve_paths().official_ref_root  # official reference copies (OSMA_BENCH_OFFICIAL_REF_ROOT)

SERVER = REF_ROOT / "pi05_opentau_official_0819/opentau_server.py"
ENV = REF_ROOT / "pi05_opentau_official_0819/opentau_env_robocasa.py"
WRAPPER = REF_ROOT / "robocasa_official_0819/gym_wrapper.py"


def _extract(path: Path, names: set[str], namespace: dict[str, Any]) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names for target in node.targets
        ):
            selected.append(node)
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - frozen source audit
    return namespace


@unittest.skipUnless(
    SERVER.exists() and ENV.exists() and WRAPPER.exists(), "공식 참조본이 없다"
)
class Pi05OfficialGoldenPayloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wrapper = _extract(WRAPPER, {"PandaOmronKeyConverter"}, {"np": np})
        env = _extract(ENV, {"convert_action"}, {"np": np, "Any": Any})
        cls.converter = wrapper["PandaOmronKeyConverter"]
        cls.convert_action = staticmethod(env["convert_action"])

    def test_state16_matches_official_map_obs(self):
        from adapters.pi05_opentau import build_state16

        rng = np.random.default_rng(20260819)
        for _ in range(200):
            observation = {
                "robot0_base_pos": rng.normal(size=3).astype(np.float32),
                "robot0_base_quat": rng.normal(size=4).astype(np.float32),
                "robot0_base_to_eef_pos": rng.normal(size=3).astype(np.float32),
                "robot0_base_to_eef_quat": rng.normal(size=4).astype(np.float32),
                "robot0_gripper_qpos": rng.normal(size=2).astype(np.float32),
            }
            mapped = self.converter.map_obs(observation)
            expected = np.concatenate(
                [
                    mapped["body.base_position"],
                    mapped["body.base_rotation"],
                    mapped["body.end_effector_position_relative"],
                    mapped["body.end_effector_rotation_relative"],
                    mapped["hand.gripper_qpos"],
                ]
            )
            canonical = {
                "base_position": observation["robot0_base_pos"],
                "base_rotation": observation["robot0_base_quat"],
                "end_effector_position_relative": observation["robot0_base_to_eef_pos"],
                "end_effector_rotation_relative": observation["robot0_base_to_eef_quat"],
                "gripper_qpos": observation["robot0_gripper_qpos"],
            }
            np.testing.assert_array_equal(build_state16(canonical), expected)

    def test_action_mapping_matches_convert_then_unmap(self):
        from action_adapter import lerobot_base_first_to_local_hybrid

        rng = np.random.default_rng(7)
        for _ in range(200):
            source = rng.uniform(-1.1, 1.1, size=12).astype(np.float32)
            mapped = self.converter.unmap_action(self.convert_action(source))
            expected = np.concatenate(
                [
                    mapped["robot0_right"],
                    np.atleast_1d(mapped["robot0_right_gripper"]),
                    mapped["robot0_base"],
                    mapped["robot0_torso"],
                    np.atleast_1d(mapped["robot0_base_mode"]),
                ]
            ).astype(np.float32)
            np.testing.assert_array_equal(
                lerobot_base_first_to_local_hybrid(source), expected
            )


_TORCH = importlib.util.find_spec("torch") is not None
_PIL = importlib.util.find_spec("PIL") is not None


@unittest.skipUnless(
    SERVER.exists() and _TORCH and _PIL, "공식 server 사본·torch·PIL 중 없는 것이 있다"
)
class Pi05OfficialBatchTest(unittest.TestCase):
    def test_batch_tensors_match_official_builder(self):
        import torch
        from PIL import Image

        from adapters.pi05_opentau import build_opentau_batch, pad_state32

        official = _extract(
            SERVER,
            {
                "ROBOCASA_CAMERA_ORDER",
                "_numpy_rgb_to_camera_tensor",
                "build_opentau_batch",
            },
            {
                "np": np,
                "torch": torch,
                "Image": Image,
                "Any": Any,
                "Dict": Dict,
                "TrainPipelineConfig": Any,
            },
        )
        rng = np.random.default_rng(42)
        slots = ("wrist", "left", "right")
        images = {
            slot: rng.integers(0, 256, (1, 256, 256, 3), dtype=np.uint8)
            for slot in slots
        }
        state32 = pad_state32(np.arange(16, dtype=np.float32))
        raw_images = {
            raw_name: images[slot][0]
            for raw_name, slot in zip(official["ROBOCASA_CAMERA_ORDER"], slots)
        }
        cfg = SimpleNamespace(num_cams=3, resolution=(256, 256), max_state_dim=32)
        expected = official["build_opentau_batch"](
            cfg, raw_images, state32, " navigate to sink ", torch.device("cpu"), torch.float32
        )
        actual = build_opentau_batch(
            images,
            state32,
            " navigate to sink ",
            (256, 256),
            torch.device("cpu"),
            torch.float32,
            torch,
        )
        self.assertEqual(set(actual), set(expected))
        for key in sorted(actual):
            if key == "prompt":
                self.assertEqual(actual[key], expected[key])
            else:
                self.assertEqual(actual[key].dtype, expected[key].dtype)
                self.assertEqual(tuple(actual[key].shape), tuple(expected[key].shape))
                self.assertTrue(torch.equal(actual[key], expected[key]), key)


if __name__ == "__main__":
    unittest.main()
