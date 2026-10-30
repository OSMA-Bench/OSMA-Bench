"""G0 golden payload — 동결한 공식 openpi/robocasa 원문과 adapter 계약을 대조한다.

원문은 `$OSMA_BENCH_OFFICIAL_REF_ROOT/openpi_official_0824/` 에 revision·SHA-256 으로 고정돼 있다.
openpi 를 설치하지 않아도 돌도록 **AST 로 원문을 읽어** 순서·키·기본값을 비교한다
(π0.5-OpenTau·GR00T G0 와 같은 방식).

여기서 잡아야 하는 것은 "차원은 같은데 순서가 다른" 조용한 실패다.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from repo_paths import resolve_paths  # noqa: E402

REF_ROOT = resolve_paths().official_ref_root  # official reference copies (OSMA_BENCH_OFFICIAL_REF_ROOT)

REF = REF_ROOT / "openpi_official_0824"
MAIN = REF / "openpi_robocasa_main.py"
POLICY = REF / "openpi_robocasa_policy.py"
PI0_CONFIG = REF / "openpi_pi0_config.py"
TRAIN_CONFIG = REF / "openpi_training_config.py"
ENV_UTILS = REF / "robocasa_env_utils.py"
WRAPPER = REF_ROOT / "robocasa_official_0819/gym_wrapper.py"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _exec_selected(path: Path, names: set[str], namespace: dict[str, Any]) -> dict[str, Any]:
    """frozen source 에서 지정한 top-level 정의만 뽑아 실행한다."""
    tree = _parse(path)
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


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in frozen source")


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found in frozen source")


def _class_defaults(cls: ast.ClassDef) -> dict[str, Any]:
    """dataclass 필드의 리터럴 기본값."""
    out: dict[str, Any] = {}
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            try:
                out[node.target.id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def _obs_subscript_key(node: ast.AST) -> str | None:
    """`obs["<key>"]` 이면 key 를, 아니면 None."""
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "obs"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return node.slice.value
    return None


@unittest.skipUnless(MAIN.exists(), "공식 openpi 참조본이 없다")
class OfficialEvaluatorContractTest(unittest.TestCase):
    """`examples/robocasa/main.py` 원문 ↔ adapter 상수."""

    @classmethod
    def setUpClass(cls):
        cls.tree = _parse(MAIN)
        cls.eval_env = _find_function(cls.tree, "eval_env")

    def test_state_concat_order_matches_adapter(self):
        """⛔ 이 프로젝트 최대 위험 지점. 순서가 틀려도 16D 라 에러가 안 난다."""
        from adapters.pi05_human300 import STATE_KEY_ORDER

        concat_keys: list[str] | None = None
        for node in ast.walk(self.eval_env):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            target = node.targets[0]
            if not (isinstance(target, ast.Name) and target.id == "state"):
                continue
            call = node.value
            self.assertIsInstance(call, ast.Call)
            self.assertEqual(call.func.attr, "concatenate")
            tuple_arg = call.args[0]
            self.assertIsInstance(tuple_arg, ast.Tuple)
            concat_keys = [_obs_subscript_key(e) for e in tuple_arg.elts]
            break

        self.assertIsNotNone(concat_keys, "official state concat not found")
        self.assertNotIn(None, concat_keys, "official concat used a non-obs element")
        self.assertEqual(concat_keys, [key for key, _, _ in STATE_KEY_ORDER])

    def test_state_concat_is_ee_first_not_base_first(self):
        """OpenTau(base-first)와 뒤바뀐 경우를 명시적으로 잡는다."""
        from adapters.pi05_human300 import STATE_KEY_ORDER

        order = [key for key, _, _ in STATE_KEY_ORDER]
        self.assertLess(
            order.index("state.end_effector_position_relative"),
            order.index("state.base_position"),
            "π0.5-Human300(openpi) 은 EE-first 다",
        )
        self.assertEqual(order[-1], "state.gripper_qpos")

    def test_camera_element_keys_match_adapter(self):
        from adapters.pi05_human300 import (
            CAMERA_SLOT_ORDER,
            CAMERA_SLOT_TO_ELEMENT_KEY,
            CAMERA_SLOT_TO_VIDEO_KEY,
        )

        # 변수 -> obs video key  (`img = np.ascontiguousarray(obs["video...."])`)
        var_to_video: dict[str, str] = {}
        for node in ast.walk(self.eval_env):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            call = node.value
            if not (isinstance(call, ast.Call) and getattr(call.func, "attr", None) == "ascontiguousarray"):
                continue
            key = _obs_subscript_key(call.args[0])
            if key is not None and target.id not in var_to_video:
                var_to_video[target.id] = key

        # element dict key -> 변수
        element_to_var: dict[str, str] = {}
        for node in ast.walk(self.eval_env):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            target = node.targets[0]
            if not (isinstance(target, ast.Name) and target.id == "element"):
                continue
            self.assertIsInstance(node.value, ast.Dict)
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and isinstance(v, ast.Name):
                    element_to_var[k.value] = v.id
            break

        self.assertTrue(element_to_var, "official element dict not found")
        official = {
            element_key: var_to_video[var]
            for element_key, var in element_to_var.items()
            if var in var_to_video
        }
        expected = {
            CAMERA_SLOT_TO_ELEMENT_KEY[slot]: CAMERA_SLOT_TO_VIDEO_KEY[slot]
            for slot in CAMERA_SLOT_ORDER
        }
        self.assertEqual(official, expected)

    def test_element_carries_state_and_prompt(self):
        for node in ast.walk(self.eval_env):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "element"
            ):
                keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
                self.assertIn("observation/state", keys)
                self.assertIn("prompt", keys)
                return
        self.fail("official element dict not found")

    def test_resize_uses_official_helpers(self):
        """`resize_with_pad` / `convert_to_uint8` 을 재구현하지 않고 부르는지."""
        calls = {
            node.func.attr
            for node in ast.walk(self.eval_env)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("resize_with_pad", calls)
        self.assertIn("convert_to_uint8", calls)

    def test_official_defaults_match_profile(self):
        from model_config import load_model_profile

        defaults = _class_defaults(_find_class(self.tree, "Args"))
        profile = load_model_profile("pi05_human300")
        self.assertEqual(defaults["resize_size"], 224)
        self.assertEqual(list(profile["resize_target"]), [224, 224])
        self.assertEqual(defaults["replan_steps"], 5)
        self.assertEqual(profile["replan_steps"], defaults["replan_steps"])

    def test_no_rotation_despite_stale_comment(self):
        """`main.py` 의 'rotate 180 degrees' 주석은 stale — 코드는 회전하지 않는다."""
        source = MAIN.read_text(encoding="utf-8")
        self.assertIn("rotate 180 degrees", source, "주석이 사라졌다면 이 테스트를 갱신할 것")
        calls = {
            node.func.attr
            for node in ast.walk(self.eval_env)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        for rotation_fn in ("rot90", "flipud", "fliplr", "rotate"):
            self.assertNotIn(rotation_fn, calls)


@unittest.skipUnless(POLICY.exists() and PI0_CONFIG.exists(), "공식 openpi 참조본이 없다")
class OfficialModelContractTest(unittest.TestCase):
    """`robocasa_policy.py` / `pi0_config.py` 원문 ↔ profile."""

    def test_pi0_config_defaults(self):
        from adapters.pi05_human300 import MODEL_ACTION_DIM
        from model_config import load_model_profile

        defaults = _class_defaults(_find_class(_parse(PI0_CONFIG), "Pi0Config"))
        profile = load_model_profile("pi05_human300")
        self.assertEqual(defaults["action_dim"], MODEL_ACTION_DIM)
        self.assertEqual(defaults["action_horizon"], profile["action_chunk_size"])
        self.assertEqual(defaults["dtype"], "bfloat16")

    def test_train_config_does_not_override_action_horizon(self):
        """`pi05_pretrain_human300` 이 chunk 를 덮어쓰면 profile 50 이 틀려진다."""
        tree = _parse(TRAIN_CONFIG)
        found = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            names = {
                kw.arg: kw.value
                for kw in node.keywords
                if kw.arg is not None
            }
            name_node = names.get("name")
            if (
                isinstance(name_node, ast.Constant)
                and name_node.value == "pi05_pretrain_human300"
            ):
                found = names
                break
        self.assertIsNotNone(found, "pi05_pretrain_human300 config not found")
        model_call = found["model"]
        self.assertIsInstance(model_call, ast.Call)
        model_kwargs = {kw.arg: kw.value for kw in model_call.keywords}
        self.assertIn("pi05", model_kwargs)
        self.assertTrue(ast.literal_eval(model_kwargs["pi05"]))
        self.assertNotIn(
            "action_horizon", model_kwargs, "공식 config 가 chunk 를 덮어쓰면 profile 을 갱신할 것"
        )
        self.assertNotIn("action_dim", model_kwargs)

    def test_outputs_trim_to_12(self):
        source = POLICY.read_text(encoding="utf-8")
        cls = _find_class(_parse(POLICY), "RobocasaOutputs")
        upper = None
        for node in ast.walk(cls):
            if isinstance(node, ast.Slice) and isinstance(node.upper, ast.Constant):
                upper = node.upper.value
        self.assertEqual(upper, 12, f"RobocasaOutputs trim changed:\n{source[-300:]}")

    def test_inputs_pad_state_to_model_action_dim(self):
        cls = _find_class(_parse(POLICY), "RobocasaInputs")
        pads = [
            node
            for node in ast.walk(cls)
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "pad_to_dim"
        ]
        self.assertTrue(pads, "official pad_to_dim call not found")
        state_pad = pads[0]
        self.assertEqual(_subscript_const(state_pad.args[0]), "observation/state")
        self.assertEqual(state_pad.args[1].attr, "action_dim")

    def test_pi05_requires_right_image(self):
        """pi05 는 right camera 가 없으면 명시적으로 죽는다 — 우리는 항상 세 장을 보낸다."""
        source = POLICY.read_text(encoding="utf-8")
        self.assertIn("observation/right_image is required for pi05", source)


def _subscript_const(node: ast.AST) -> Any:
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        return node.slice.value
    return None


@unittest.skipUnless(ENV_UTILS.exists() and WRAPPER.exists(), "공식 robocasa 참조본이 없다")
class OfficialActionPathTest(unittest.TestCase):
    """`convert_action` → `unmap_action` 이 우리 `robocasa_ee_first` 경로와 같은지."""

    @classmethod
    def setUpClass(cls):
        cls.convert_action = staticmethod(
            _exec_selected(ENV_UTILS, {"convert_action"}, {"np": np})["convert_action"]
        )
        cls.converter = _exec_selected(WRAPPER, {"PandaOmronKeyConverter"}, {"np": np})[
            "PandaOmronKeyConverter"
        ]

    def test_action_mapping_matches_official_convert_then_unmap(self):
        from action_adapter import robocasa_ee_first_to_local_hybrid

        rng = np.random.default_rng(20260824)
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
                robocasa_ee_first_to_local_hybrid(source), expected
            )

    def test_convert_action_has_no_permutation(self):
        """OpenTau 의 base-first permutation 을 여기에 적용하면 안 된다."""
        source = np.arange(12, dtype=np.float32)
        converted = self.convert_action(source)
        np.testing.assert_array_equal(converted["action.end_effector_position"], source[0:3])
        np.testing.assert_array_equal(converted["action.end_effector_rotation"], source[3:6])
        np.testing.assert_array_equal(converted["action.gripper_close"], source[6:7])
        np.testing.assert_array_equal(converted["action.base_motion"], source[7:11])
        np.testing.assert_array_equal(converted["action.control_mode"], source[11:12])


if __name__ == "__main__":
    unittest.main()
