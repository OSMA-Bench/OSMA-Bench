"""G0 golden payload — 공식 evaluator 와 같은 processor 입력을 만드는지 수치로 확인한다.

`$OSMA_BENCH_OFFICIAL_REF_ROOT/xr1_official_0819/entry.py` (공식 사본, ORIGIN.md 에 SHA-256) 의 전처리를 직접 import 해
같은 합성 관측을 통과시키고, 우리 adapter 의 결과와 tensor 단위로 비교한다.

모델 forward 는 하지 않는다. 다만 실제 `MiBotProcessor` 가 필요하므로 checkpoint 가 없으면 skip 한다.
나머지 model-free 테스트는 이 파일과 무관하게 돈다.
"""
import importlib.util as ilu
import os
import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from repo_paths import resolve_paths  # noqa: E402

REF_ROOT = resolve_paths().official_ref_root  # official reference copies (OSMA_BENCH_OFFICIAL_REF_ROOT)

OFFICIAL = REF_ROOT / "xr1_official_0819" / "entry.py"
SNAPSHOT = (  # resolved HF snapshot; skipped when the checkpoint is not cached locally
    Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    / "hub" / "models--XiaomiRobotics--Xiaomi-Robotics-1-RoboCasa365" / "snapshots" / "0d1aa76d0d82debc9b611e4d1e231096434d5be4"
)

HISTORY, IMG, STATE_NATIVE = 4, 256, 14


def _load_official():
    """entry.py 를 import 한다.

    entry.py 는 rollout 실행용이라 imageio/tqdm/client 를 최상위에서 import 하지만, 이 테스트가
    쓰는 것은 순수 전처리 함수뿐이다. 없는 모듈은 스텁으로 대체해 어느 환경에서든 로드되게 한다.
    스텁은 전처리 경로에서 호출되지 않는다.
    """
    def _importable(name):
        if name in sys.modules:
            return True
        try:
            return ilu.find_spec(name) is not None
        except (ImportError, ValueError):
            return False

    def _stub(name, **attrs):
        if name in sys.modules:
            return
        mod = type(sys)(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod

    # 진짜로 없는 것만 대체한다. tqdm 을 무조건 스텁하면 transformers 의 tqdm.auto 가 깨진다.
    if not _importable("client"):
        _stub("client", Client=object)
    if not _importable("tqdm"):
        _stub("tqdm", tqdm=lambda x, **k: x)
    if not _importable("imageio"):
        _stub("imageio")
        _stub("imageio.v2")
        sys.modules["imageio"].v2 = sys.modules["imageio.v2"]

    spec = ilu.spec_from_file_location("_xr1_official_entry", OFFICIAL)
    module = ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic():
    rng = np.random.default_rng(20260819)
    images = {k: rng.integers(0, 256, (HISTORY, IMG, IMG, 3), dtype=np.uint8)
              for k in ("left", "right", "wrist")}
    obs = []
    for _ in range(HISTORY):
        q1 = rng.normal(size=4); q1 /= np.linalg.norm(q1)
        q2 = rng.normal(size=4); q2 /= np.linalg.norm(q2)
        obs.append({
            "end_effector_position_relative": rng.normal(size=3).astype(np.float32),
            "end_effector_rotation_relative": q1.astype(np.float32),
            "gripper_qpos": rng.normal(size=2).astype(np.float32),
            "base_position": rng.normal(size=3).astype(np.float32),
            "base_rotation": q2.astype(np.float32),
        })
    return images, obs


def _have(name):
    try:
        return ilu.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# entry.py 의 전처리는 PIL 을 실제로 쓴다. host python 에는 없고 XR-1 venv/컨테이너에만 있으므로
# 없는 환경에서는 에러가 아니라 skip 으로 끝나야 한다.
_PIL = _have("PIL")


@unittest.skipUnless(OFFICIAL.exists() and _PIL, "공식 entry.py 사본 또는 PIL 이 없다")
class TestOfficialTranscription(unittest.TestCase):
    """checkpoint 없이 도는 부분 — 전사 정확성."""

    @classmethod
    def setUpClass(cls):
        cls.official = _load_official()
        cls.images, cls.obs = _synthetic()

    def test_constants_match(self):
        import adapters.xr1 as ours
        self.assertEqual(ours.STATE_DIM, self.official.STATE_DIM)
        self.assertEqual(ours.ACTION_DIM, self.official.ACTION_DIM)
        self.assertEqual(len(ours.CAMERA_KEYS), len(self.official.CAMERA_KEYS))

    def test_quaternion_conversion_matches(self):
        import adapters.xr1 as ours
        rng = np.random.default_rng(7)
        for _ in range(200):
            q = rng.normal(size=4); q /= np.linalg.norm(q)
            np.testing.assert_allclose(
                ours.quat_xyzw_to_axis_angle(q),
                self.official.quat_xyzw_to_axis_angle(q),
                rtol=0, atol=1e-6)

    def test_state14_matches(self):
        import adapters.xr1 as ours
        for o in self.obs:
            theirs = self.official.observation_to_state(
                {f"state.{k}": v for k, v in o.items()})
            np.testing.assert_allclose(ours.build_state14(o), theirs, rtol=0, atol=0)

    def test_state_padding_matches(self):
        import adapters.xr1 as ours
        _, padded = ours.pad_state_history(self.obs)
        native = np.stack([self.official.observation_to_state(
            {f"state.{k}": v for k, v in o.items()}) for o in self.obs], axis=0)
        theirs = np.zeros((1, native.shape[0], self.official.STATE_DIM), dtype=np.float32)
        theirs[0, :, : native.shape[-1]] = native
        np.testing.assert_allclose(padded, theirs, rtol=0, atol=0)
        self.assertEqual(padded.shape, (1, HISTORY, 60))

    def test_center_crop_matches(self):
        import adapters.xr1 as ours
        adapter = ours.XR1Adapter.__new__(ours.XR1Adapter)
        adapter.crop_ratio = 0.95
        for frame in self.images["left"]:
            np.testing.assert_array_equal(
                np.asarray(adapter._center_crop(frame)),
                np.asarray(self.official.center_crop(frame, 0.95)))

    def test_history_sampling_matches(self):
        import collections
        from observation_adapter import ObservationHistory
        for filled in (1, 2, 4, 7, 12):
            hist = ObservationHistory(length=4, interval=2, camera_slots=["left"])
            dq = collections.deque(maxlen=(4 - 1) * 2 + 1)
            for step in range(filled):
                arr = np.full((2, 2, 3), step, dtype=np.uint8)
                hist.append(step=step, images={"left": arr}, state_raw=self.obs[0])
                dq.append(arr)
            theirs = self.official.sample_history(dq, 4, 2)
            np.testing.assert_array_equal(hist.sample().images["left"], theirs)


@unittest.skipUnless(OFFICIAL.exists() and SNAPSHOT.exists() and _PIL,
                     "공식 사본·checkpoint·PIL 중 없는 것이 있다")
class TestProcessorPayload(unittest.TestCase):
    """checkpoint 가 있을 때만 — 실제 processor 입력 tensor 대조."""

    @classmethod
    def setUpClass(cls):
        try:
            from transformers import AutoProcessor
        except ImportError as exc:                       # pragma: no cover
            raise unittest.SkipTest(f"transformers 없음: {exc}")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        cls.processor = AutoProcessor.from_pretrained(str(SNAPSHOT), trust_remote_code=True)
        cls.official = _load_official()
        cls.images, cls.obs = _synthetic()
        cls.instruction = "navigate safely to the sink while avoiding obstacles"

    def _ours(self):
        import adapters.xr1 as ours
        adapter = ours.XR1Adapter.__new__(ours.XR1Adapter)
        adapter.crop_ratio = 0.95
        _, padded = ours.pad_state_history(self.obs)
        return self.processor.apply_chat_template(
            adapter._build_messages(self.images, self.instruction),
            tokenize=True, return_dict=True, return_tensors="pt",
            do_resize=False, state=padded, robot_type="robocasa365")

    def _theirs(self):
        client = self.official.EvalClient.__new__(self.official.EvalClient)
        client.processor, client.robot_type, client.crop_ratio = self.processor, "robocasa365", 0.95
        native = np.stack([self.official.observation_to_state(
            {f"state.{k}": v for k, v in o.items()}) for o in self.obs], axis=0)
        state = np.zeros((1, native.shape[0], self.official.STATE_DIM), dtype=np.float32)
        state[0, :, : native.shape[-1]] = native
        keyed = {k: self.images[s] for k, s in
                 zip(self.official.CAMERA_KEYS, ("left", "right", "wrist"))}
        return self.processor.apply_chat_template(
            client._build_messages(keyed, self.instruction),
            tokenize=True, return_dict=True, return_tensors="pt",
            do_resize=False, state=state, robot_type="robocasa365")

    def test_processor_inputs_identical(self):
        ours, theirs = self._ours(), self._theirs()
        self.assertEqual(set(ours.keys()), set(theirs.keys()), "출력 키 집합이 다르다")
        for key in sorted(ours.keys()):
            a, b = ours[key], theirs[key]
            if hasattr(a, "cpu"):                       # torch tensor
                self.assertEqual(a.dtype, b.dtype, f"{key} dtype 불일치")
                self.assertEqual(tuple(a.shape), tuple(b.shape), f"{key} shape 불일치")
                # bfloat16 은 numpy 로 못 옮기므로 float32 로 올려서 비교한다 (값 손실 없음)
                a, b = a.cpu().float().numpy(), b.cpu().float().numpy()
            else:
                a, b = np.asarray(a), np.asarray(b)
                self.assertEqual(a.shape, b.shape, f"{key} shape 불일치")
            if np.issubdtype(a.dtype, np.floating):
                np.testing.assert_allclose(a, b, rtol=0, atol=0, err_msg=f"{key} 값 불일치")
            else:
                np.testing.assert_array_equal(a, b, err_msg=f"{key} 값 불일치")


if __name__ == "__main__":
    unittest.main()
