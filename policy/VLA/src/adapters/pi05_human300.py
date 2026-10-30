"""π0.5-Human300 adapter for the RoboCasa team's official openpi checkpoint.

GR00T 와 같은 원칙이다 — **공식 전처리를 전사하지 않고 공식 inference 진입점을 그대로 부른다.**
`scripts/serve_policy.py` 가 하는 일이 정확히 `create_trained_policy(...)` → `policy.infer(element)`
이므로 in-process 호출은 공식 websocket 서버와 **같은 경로**다. adapter 가 직접 하는 일은
`element` dict 구성뿐이고, 정규화 · 토크나이즈 · 32D padding · flow denoising · 역정규화는
전부 openpi 안에 있다.

계약의 단일 출처는 README.md § "Model contracts" (π0.5-Human300) 이고 공식 사본은
`$OSMA_BENCH_OFFICIAL_REF_ROOT/openpi_official_0824/` 에 revision 과 SHA-256 으로 고정돼 있다.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import time
from pathlib import Path
from typing import Any

import numpy as np

from adapters.xr1 import decode_array


TRAIN_CONFIG_NAME = "pi05_pretrain_human300"

# 공식 `examples/robocasa/main.py` 의 canonical slot 순서. XR-1 · GR00T 의 left/right/wrist 와 다르다.
CAMERA_SLOT_ORDER = ("left", "wrist", "right")

# slot -> 공식 evaluator 가 `element` 에 쓰는 key.
CAMERA_SLOT_TO_ELEMENT_KEY = {
    "left": "observation/image",
    "wrist": "observation/wrist_image",
    "right": "observation/right_image",
}

# slot -> robocasa gym wrapper 의 video key (감사 기록용).
CAMERA_SLOT_TO_VIDEO_KEY = {
    "left": "video.robot0_agentview_left",
    "wrist": "video.robot0_eye_in_hand",
    "right": "video.robot0_agentview_right",
}

# 공식 `main.py` 의 `np.concatenate` 순서. ⛔ EE-first 다 — OpenTau(base-first)와 뒤바꾸지 말 것.
STATE_KEY_ORDER = (
    ("state.end_effector_position_relative", "end_effector_position_relative", 3),
    ("state.end_effector_rotation_relative", "end_effector_rotation_relative", 4),
    ("state.base_position", "base_position", 3),
    ("state.base_rotation", "base_rotation", 4),
    ("state.gripper_qpos", "gripper_qpos", 2),
)

STATE_NATIVE_DIM = 16
STATE_PAD_DIM = 32  # openpi 의 `RobocasaInputs` 가 `pad_to_dim(action_dim=32)` 로 채운다
MODEL_ACTION_DIM = 32
ACTION_DIM = 12
RESIZE_SIZE = 224


def build_state16(observation: dict[str, Any]) -> np.ndarray:
    """공식 evaluator 의 16D concat 을 그대로 만든다.

    ⛔ quaternion 을 재배열하지 말 것. robosuite 는 xyzw 를 주고 공식 파이프라인이 그 값을
    **그대로 넣어 학습했다.** "고치면" 훈련 분포와 어긋나 조용히 망가진다.
    상세: README.md § "Model contracts" (π0.5-Human300 — state layout).

    ⛔ 여기서 32D padding 을 하지 말 것 — openpi 의 `RobocasaInputs` 가 한다. 두 번 하면 안 된다.
    """
    columns = []
    for _, raw_key, dim in STATE_KEY_ORDER:
        value = np.asarray(observation[raw_key], dtype=np.float64).reshape(-1)
        if value.shape != (dim,):
            raise ValueError(f"{raw_key} is {value.shape}; expected ({dim},)")
        if not np.isfinite(value).all():
            raise ValueError(f"π0.5-Human300 state {raw_key} contains NaN or Inf")
        columns.append(value)
    state = np.concatenate(columns).astype(np.float32)
    if state.shape != (STATE_NATIVE_DIM,):
        raise ValueError(f"Expected a {STATE_NATIVE_DIM}D RoboCasa state, got {state.shape}")
    return state


def resolve_checkpoint_snapshot(model_path: str, revision: str, subfolder: str) -> Path:
    """`snapshots/<revision>/<subfolder>` 인지 확인한다.

    GR00T 와 같은 형태다 — 경로 끝이 revision 이 아니라 subfolder 이므로 π0.5-OpenTau 의
    `snapshot.name == revision` 검사는 재사용할 수 없다.
    """
    snapshot = Path(model_path).resolve()
    if not snapshot.is_dir():
        raise ValueError(
            "π0.5-Human300 requires a local immutable checkpoint snapshot; pass --model-path "
            f"pointing at snapshots/<revision>/{subfolder}"
        )
    parts = Path(subfolder).parts
    if snapshot.parts[-len(parts):] != parts:
        raise ValueError(f"Snapshot {snapshot} does not end with subfolder {subfolder!r}")
    if revision not in snapshot.parts:
        raise ValueError(f"Snapshot {snapshot} does not carry revision {revision}")
    for required in ("params", "assets"):
        if not (snapshot / required).is_dir():
            raise ValueError(f"Checkpoint is missing {snapshot / required}")
    if not (snapshot / "assets" / "norm_stats.json").is_file():
        raise ValueError(
            f"Checkpoint is missing {snapshot / 'assets' / 'norm_stats.json'}; "
            "openpi loads normalization statistics from this file"
        )
    return snapshot


class Pi05Human300Adapter:
    def __init__(
        self,
        profile: dict[str, Any],
        model_path: str,
        model_revision: str,
        device: str = "cuda:0",
        attention_implementation: str = "eager",
        reproducibility: dict[str, Any] | None = None,
    ) -> None:
        import dataclasses

        import jax
        from openpi_client import image_tools
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        self.profile = profile
        self.model_revision = model_revision
        self.device = device
        self.reproducibility = reproducibility or {}
        self.episode_seed: int | None = None
        self._jax = jax
        self._image_tools = image_tools

        subfolder = str(profile["checkpoint_subfolder"])
        self.snapshot = resolve_checkpoint_snapshot(model_path, model_revision, subfolder)
        self.model_path = str(self.snapshot)
        self.resolved_model_revision = self.snapshot.parts[
            self.snapshot.parts.index(model_revision)
        ]

        train_config = _config.get_config(TRAIN_CONFIG_NAME)

        # ⚠ 공식 `LeRobotRobocasaDataConfig.create()` 는 `norm_stats` 가 없을 때
        # `data_dirs` 의 각 dataset 에서 `meta/stats.json` 을 **읽는다**
        # (`groot_openpi_dataset._load_norm_stats_from_groot_dataset`). 우리는 추론만 하므로
        # Human300 482h 원본 dataset 이 디스크에 없고, 그대로 두면 FileNotFoundError 로 죽는다.
        # `data_dirs=None` 으로만 바꾸면 그 fallback 분기가 통째로 건너뛰어지고
        # (`if base.norm_stats is None and self.data_dirs and len(...) > 0`), 나머지 transform 은
        # 공식 그대로다. 그러면 `asset_id is None` 경로가 살아서
        # `create_trained_policy` 가 checkpoint 의 `assets/norm_stats.json` 을 읽는다 —
        # 즉 **학습 때 쓰던 통계를 그대로** 쓴다. 상세: 계약 문서 §6.
        train_config = dataclasses.replace(
            train_config,
            data=dataclasses.replace(train_config.data, data_dirs=None),
        )
        self._assert_official_contract(train_config)

        self.policy = _policy_config.create_trained_policy(train_config, self.snapshot)

        # 로드된 뒤 실제 data config 도 확인한다 (upstream drift gate).
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        if data_config.asset_id is not None:
            raise ValueError(
                f"Official data config asset_id is {data_config.asset_id!r}; expected None so that "
                "norm stats come from the checkpoint assets directory"
            )
        self.norm_stats_source = str(self.snapshot / "assets")

        self.runtime_version = importlib.metadata.version("openpi")
        self.jax_version = importlib.metadata.version("jax")
        self.jax_devices = [str(d) for d in jax.devices()]
        self.jax_default_backend = jax.default_backend()

    def _assert_official_contract(self, train_config: Any) -> None:
        """profile 선언 == 설치된 공식 config. 다르면 rollout 하지 않는다."""
        model = train_config.model
        chunk = int(self.profile["action_chunk_size"])
        if int(model.action_horizon) != chunk:
            raise ValueError(
                f"Official action_horizon {model.action_horizon} != profile action_chunk_size {chunk}"
            )
        if int(model.action_dim) != MODEL_ACTION_DIM:
            raise ValueError(f"Official action_dim {model.action_dim} != {MODEL_ACTION_DIM}")
        if model.model_type.name != "PI05":
            raise ValueError(f"Official model_type {model.model_type.name} != PI05")
        if not getattr(model, "pi05", False):
            raise ValueError("Official model config is not pi05")
        if int(self.profile["history_length"]) != 1:
            raise ValueError("π0.5-Human300 observes a single frame; history_length must be 1")
        slots = list(self.profile["camera_slots"])
        if slots != list(CAMERA_SLOT_ORDER):
            raise ValueError(f"Profile camera_slots {slots} != {list(CAMERA_SLOT_ORDER)}")
        if list(self.profile["resize_target"]) != [RESIZE_SIZE, RESIZE_SIZE]:
            raise ValueError(f"Profile resize_target must be [{RESIZE_SIZE}, {RESIZE_SIZE}]")
        if self.profile["action_convention"] != "robocasa_ee_first":
            raise ValueError(
                "π0.5-Human300 emits local flat 12D directly; action_convention must be "
                "robocasa_ee_first (the base-first permutation belongs to OpenTau)"
            )

    def _preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """공식 `main.py` 와 같은 `resize_with_pad` → `convert_to_uint8`.

        ⛔ 재구현하지 말 것 — `openpi_client.image_tools` 를 그대로 부른다.
        ⚠ 공식 `main.py` 의 "rotate 180 degrees" 주석은 stale 이다. 코드는 회전하지 않는다.
        """
        resized = self._image_tools.resize_with_pad(frame, RESIZE_SIZE, RESIZE_SIZE)
        return self._image_tools.convert_to_uint8(resized)

    def metadata(self) -> dict[str, Any]:
        return {
            "model_profile": self.profile["name"],
            "checkpoint": self.model_path,
            "checkpoint_revision": self.model_revision,
            "resolved_checkpoint_revision": self.resolved_model_revision,
            "checkpoint_subfolder": self.profile["checkpoint_subfolder"],
            "train_config_name": TRAIN_CONFIG_NAME,
            "robot_type": self.profile["robot_type"],
            "action_dim": ACTION_DIM,
            "model_action_dim": MODEL_ACTION_DIM,
            "action_convention": self.profile["action_convention"],
            "camera_slots": list(CAMERA_SLOT_ORDER),
            "element_keys": [CAMERA_SLOT_TO_ELEMENT_KEY[s] for s in CAMERA_SLOT_ORDER],
            "video_keys": [CAMERA_SLOT_TO_VIDEO_KEY[s] for s in CAMERA_SLOT_ORDER],
            "history_length": int(self.profile["history_length"]),
            "history_interval": int(self.profile["history_interval"]),
            "action_chunk_size": int(self.profile["action_chunk_size"]),
            "replan_steps": int(self.profile["replan_steps"]),
            "state_dim_native": STATE_NATIVE_DIM,
            "state_dim_padded": STATE_PAD_DIM,
            "resize_target": list(self.profile["resize_target"]),
            "norm_stats_source": self.norm_stats_source,
            "runtime_package": "openpi",
            "runtime_version": self.runtime_version,
            "runtime_repo": self.profile.get("runtime_repo"),
            "runtime_revision": self.profile.get("runtime_revision"),
            "device": self.device,
            "episode_seed": self.episode_seed,
            "runtime": {
                "jax": self.jax_version,
                "jax_backend": self.jax_default_backend,
                "jax_devices": self.jax_devices,
            },
            "reproducibility": self.reproducibility,
        }

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        """policy seed 를 JAX rng key 로 심는다.

        openpi `Policy.__init__` 은 `self._rng = rng or jax.random.key(0)` 로 두고 `infer()` 마다
        `jax.random.split` 한다. 생성자에 rng 를 넣는 경로가 `create_trained_policy` 에는 없으므로
        같은 결과가 되도록 여기서 key 를 다시 심는다 — `Policy(rng=jax.random.key(seed))` 와 동일하다.
        """
        requested_seed = payload.get("policy_seed")
        if requested_seed is not None:
            self.episode_seed = int(requested_seed)
            if self.episode_seed < 0:
                raise ValueError("policy_seed cannot be negative")
            np.random.seed(self.episode_seed)
            self.policy._rng = self._jax.random.key(self.episode_seed)  # noqa: SLF001
        return {
            "ok": True,
            "instruction": payload.get("instruction", ""),
            "policy_seed": self.episode_seed,
        }

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported request schema")
        expected_history = int(self.profile["history_length"])
        image_size = int(self.profile["image_size"])
        expected_image_shape = (expected_history, image_size, image_size, 3)

        images = {slot: decode_array(payload["images"][slot]) for slot in CAMERA_SLOT_ORDER}
        for slot, image in images.items():
            if image.dtype != np.uint8 or image.shape != expected_image_shape:
                raise ValueError(
                    f"Camera {slot} is {image.dtype} {image.shape}; "
                    f"expected uint8 {expected_image_shape}"
                )

        state_raw = payload["state_raw"]
        if len(state_raw) != expected_history:
            raise ValueError(f"Received {len(state_raw)} states; expected {expected_history}")
        state16 = build_state16(state_raw[-1])

        element: dict[str, Any] = {
            CAMERA_SLOT_TO_ELEMENT_KEY[slot]: self._preprocess_image(
                np.ascontiguousarray(images[slot][0])
            )
            for slot in CAMERA_SLOT_ORDER
        }
        element["observation/state"] = state16
        element["prompt"] = str(payload["instruction"])

        started = time.perf_counter()
        result = self.policy.infer(element)
        inference_ms = (time.perf_counter() - started) * 1000.0

        if "actions" not in result:
            raise ValueError(f"openpi returned no actions; keys={sorted(result)}")
        actions = np.asarray(result["actions"], dtype=np.float32)
        expected_chunk = int(self.profile["action_chunk_size"])
        if actions.shape != (expected_chunk, ACTION_DIM):
            raise ValueError(
                f"openpi returned {actions.shape}; expected ({expected_chunk}, {ACTION_DIM})"
            )
        if not np.isfinite(actions).all():
            raise ValueError("π0.5-Human300 returned a non-finite action chunk")
        actions = np.ascontiguousarray(actions, dtype=np.float32)

        image_audit = {
            slot: {
                "element_key": CAMERA_SLOT_TO_ELEMENT_KEY[slot],
                "video_key": CAMERA_SLOT_TO_VIDEO_KEY[slot],
                "shape": list(images[slot].shape),
                "dtype": str(images[slot].dtype),
                "sha256": hashlib.sha256(images[slot].tobytes()).hexdigest(),
                "preprocessed_shape": list(element[CAMERA_SLOT_TO_ELEMENT_KEY[slot]].shape),
            }
            for slot in CAMERA_SLOT_ORDER
        }
        return {
            "actions": actions.tolist(),
            "metadata": {
                "chunk_length": int(actions.shape[0]),
                "inference_ms": inference_ms,
                # ⚠ torch 기준 peak VRAM 은 JAX 모델에서 의미가 없다. VRAM 은
                # `nvidia-smi --query-compute-apps` 로 프로세스 단위로 따로 잰다 (계약 문서 §7).
                "peak_vram_mb": None,
                "policy_timing": result.get("policy_timing"),
            },
            "input_audit": {
                "camera_slots": list(CAMERA_SLOT_ORDER),
                "element_keys": [CAMERA_SLOT_TO_ELEMENT_KEY[s] for s in CAMERA_SLOT_ORDER],
                "video_keys": [CAMERA_SLOT_TO_VIDEO_KEY[s] for s in CAMERA_SLOT_ORDER],
                "images": image_audit,
                "instruction_sha256": hashlib.sha256(str(payload["instruction"]).encode("utf-8")).hexdigest(),
                "history_steps": list(payload.get("history_steps", [])),
                "state_native_shape": [expected_history, STATE_NATIVE_DIM],
                "policy_state_shape": [1, STATE_PAD_DIM],
                "state_native": [state16.tolist()],
                "state_native_sha256": hashlib.sha256(state16.tobytes()).hexdigest(),
                "state_keys": [key for key, _, _ in STATE_KEY_ORDER],
                "preprocess": {
                    "resize_target": [RESIZE_SIZE, RESIZE_SIZE],
                    "resize_fn": "openpi_client.image_tools.resize_with_pad",
                },
            },
        }
