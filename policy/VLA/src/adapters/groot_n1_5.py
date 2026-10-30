"""GR00T N1.5 adapter for the RoboCasa Human300 checkpoint.

XR-1 과 π0.5 adapter 는 공식 전처리를 우리 코드로 전사했지만, GR00T 는 그렇게 하지 않는다.
`Gr00tPolicy.get_action()` 이 공식 inference 진입점이고 transform · normalization · concat ·
unapply 가 전부 그 안에 있다. 이 adapter 는 **raw observation dict 만 만들고 공식 policy 를
그대로 호출**한다. dict modality + rotation_6d + 3중 정규화라 전사 위험이 이 패널에서 가장 크고,
전사하지 않으면 그 위험이 원천적으로 사라진다.

계약의 단일 출처는 README.md § "Model contracts" (GR00T N1.5) 이고 공식 사본은
`$OSMA_BENCH_OFFICIAL_REF_ROOT/groot_official_0821/` 에 revision 과 SHA-256 으로 고정돼 있다.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import time
from pathlib import Path
from typing import Any

import numpy as np

from adapters.xr1 import decode_array


# profile 의 canonical slot 이름 -> 공식 PandaOmronDataConfig 의 video key.
# 순서는 `PandaOmronDataConfig.video_keys` 와 같아야 한다 (left -> right -> wrist).
CAMERA_SLOT_ORDER = ("left", "right", "wrist")
CAMERA_SLOT_TO_VIDEO_KEY = {
    "left": "video.robot0_agentview_left",
    "right": "video.robot0_agentview_right",
    "wrist": "video.robot0_eye_in_hand",
}

# `PandaOmronDataConfig.state_keys` 순서. observation_adapter 의 canonical raw key 와 1:1 이고
# 공식 `PandaOmronKeyConverter.map_obs_in_eval()` 이 내는 키와 같다.
STATE_KEY_ORDER = (
    ("state.end_effector_position_relative", "end_effector_position_relative", 3),
    ("state.end_effector_rotation_relative", "end_effector_rotation_relative", 4),
    ("state.gripper_qpos", "gripper_qpos", 2),
    ("state.base_position", "base_position", 3),
    ("state.base_rotation", "base_rotation", 4),
)

# `PandaOmronDataConfig.action_keys` 순서. 이 순서로 concat 하면 local flat 12D 와 같다:
# [0:3] EE pos · [3:6] EE rot · [6] gripper · [7:10] base · [10] torso · [11] mode
ACTION_KEY_ORDER = (
    ("action.end_effector_position", 3),
    ("action.end_effector_rotation", 3),
    ("action.gripper_close", 1),
    ("action.base_motion", 4),
    ("action.control_mode", 1),
)

LANGUAGE_KEY = "annotation.human.task_description"
STATE_NATIVE_DIM = 16
STATE_TRANSFORMED_DIM = 20
STATE_PAD_DIM = 64
ACTION_DIM = 12
DATA_CONFIG_NAME = "panda_omron"
EMBODIMENT_TAG = "new_embodiment"


def build_state_dict(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    """공식 `map_obs_in_eval()` 과 같은 `state.*` dict 를 만든다.

    ⛔ quaternion 을 재배열하지 말 것. robosuite 는 xyzw 를 주고 GR00T 의
    `quaternion_to_matrix` 는 docstring 상 wxyz 를 기대하지만, 공식 파이프라인이 이 값을
    **그대로 넣어 학습했다.** "고치면" 훈련 분포와 어긋나 조용히 망가진다.
    상세: README.md § "Model contracts" (GR00T N1.5 — quaternion order).
    """
    state: dict[str, np.ndarray] = {}
    total = 0
    for state_key, raw_key, dim in STATE_KEY_ORDER:
        value = np.asarray(observation[raw_key], dtype=np.float64).reshape(-1)
        if value.shape != (dim,):
            raise ValueError(f"{raw_key} is {value.shape}; expected ({dim},)")
        if not np.isfinite(value).all():
            raise ValueError(f"GR00T state {raw_key} contains NaN or Inf")
        state[state_key] = value
        total += dim
    if total != STATE_NATIVE_DIM:
        raise ValueError(f"Expected a {STATE_NATIVE_DIM}D raw state, got {total}")
    return state


def concat_action_chunk(action_dict: dict[str, Any], expected_chunk: int) -> np.ndarray:
    """unapply 결과 dict 를 `action_keys` 순서로 이어붙여 local flat 12D chunk 를 만든다."""
    columns = []
    for action_key, dim in ACTION_KEY_ORDER:
        if action_key not in action_dict:
            raise ValueError(f"GR00T returned no {action_key}")
        value = np.asarray(action_dict[action_key], dtype=np.float32)
        if value.ndim == 1:
            value = value.reshape(-1, 1)
        if value.shape != (expected_chunk, dim):
            raise ValueError(f"{action_key} is {value.shape}; expected ({expected_chunk}, {dim})")
        columns.append(value)
    chunk = np.concatenate(columns, axis=-1)
    if chunk.shape != (expected_chunk, ACTION_DIM):
        raise ValueError(f"Concatenated action chunk is {chunk.shape}")
    return np.ascontiguousarray(chunk, dtype=np.float32)


def resolve_checkpoint_snapshot(model_path: str, revision: str, subfolder: str) -> Path:
    """`snapshots/<revision>/<subfolder>` 인지 확인한다.

    π0.5 의 `snapshot.name == revision` 검사는 재사용할 수 없다 — GR00T 는 repo 안의
    subfolder 를 가리키므로 경로 끝이 revision 이 아니라 `checkpoint-120000` 이다.
    """
    snapshot = Path(model_path).resolve()
    if not snapshot.is_dir():
        raise ValueError(
            "GR00T requires a local immutable checkpoint snapshot; pass --model-path pointing at "
            f"snapshots/<revision>/{subfolder}"
        )
    expected_suffix = Path(subfolder)
    parts = expected_suffix.parts
    if snapshot.parts[-len(parts):] != parts:
        raise ValueError(f"Snapshot {snapshot} does not end with subfolder {subfolder!r}")
    revision_dir = snapshot.parents[len(parts) - 1]
    if revision_dir.name != revision:
        raise ValueError(
            f"Snapshot revision directory {revision_dir.name!r} does not match {revision!r}"
        )
    if revision_dir.parent.name != "snapshots":
        raise ValueError(f"Snapshot {snapshot} is not inside an HF snapshots/ cache layout")
    return snapshot


class GrootN15Adapter:
    def __init__(
        self,
        profile: dict[str, Any],
        model_path: str,
        model_revision: str,
        device: str = "cuda:0",
        attention_implementation: str = "eager",
        reproducibility: dict[str, Any] | None = None,
    ) -> None:
        import torch
        from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
        from gr00t.experiment.data_config import DATA_CONFIG_MAP
        from gr00t.model.policy import Gr00tPolicy

        del attention_implementation  # checkpoint config is the source of truth
        self.torch = torch
        self.profile = dict(profile)
        self.model_revision = model_revision
        self.device = device
        self.reproducibility = reproducibility or {}
        self.episode_seed: int | None = None

        subfolder = str(profile["checkpoint_subfolder"])
        self.snapshot = resolve_checkpoint_snapshot(model_path, model_revision, subfolder)
        self.model_path = str(self.snapshot)
        self.resolved_model_revision = self.snapshot.parents[len(Path(subfolder).parts) - 1].name

        # profile 이 선언한 계약과 **설치된** 공식 DataConfig 가 어긋나면 즉시 멈춘다.
        # upstream drift 를 rollout 전에 잡는 gate다.
        data_config_name = str(profile.get("data_config", DATA_CONFIG_NAME))
        if data_config_name not in DATA_CONFIG_MAP:
            raise ValueError(f"Unknown GR00T data config {data_config_name!r}")
        self.data_config = DATA_CONFIG_MAP[data_config_name]
        self._assert_official_contract()

        embodiment_tag = str(profile.get("embodiment_tag", EMBODIMENT_TAG))
        if embodiment_tag not in EMBODIMENT_TAG_MAPPING:
            raise ValueError(f"Unknown embodiment tag {embodiment_tag!r}")
        self.embodiment_tag = embodiment_tag
        self.embodiment_projector_index = int(EMBODIMENT_TAG_MAPPING[embodiment_tag])

        metadata_path = self.snapshot / "experiment_cfg" / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError(f"Checkpoint is missing {metadata_path}")

        self.denoising_steps = int(profile["denoising_steps"])
        self.runtime_version = importlib.metadata.version("gr00t")

        self.policy = Gr00tPolicy(
            model_path=self.model_path,
            modality_config=self.data_config.modality_config(),
            modality_transform=self.data_config.transform(),
            embodiment_tag=self.embodiment_tag,
            denoising_steps=self.denoising_steps,
            device=device,
        )
        # `Gr00tPolicy.__init__` 은 transform 을 eval() 로 바꾼다 (CenterCrop, ColorJitter 없음).
        # 그 상태가 실제로 유지됐는지 확인한다 — train 모드면 RandomCrop/ColorJitter 가 섞인다.
        # ⚠ `ComposedModalityTransform.eval()` 은 **자식 transform 에만 전파**하고 자기 자신의
        # `training` 은 기본값 True 로 남는다 (`gr00t/data/transform/base.py`). composed 객체의
        # 플래그를 보면 정상 eval 인데도 거짓 실패한다 — 반드시 자식을 본다.
        children = list(getattr(self.policy.modality_transform, "transforms", []))
        if not children:
            raise ValueError("GR00T modality transform exposes no child transforms")
        still_training = sorted(
            {type(child).__name__ for child in children if getattr(child, "training", False)}
        )
        if still_training:
            raise ValueError(
                f"GR00T modality transforms still in train mode: {still_training}; "
                "eval mode is required (CenterCrop, no ColorJitter)"
            )

        # `DatasetStatistics.state/.action` 은 `dict[str, DatasetStatisticalValues]` 다
        # (`gr00t/data/schema.py:186-188`). 객체가 아니라 dict 이므로 그대로 키를 읽는다.
        stats = self.policy.metadata.statistics
        self.normalization_state_keys = sorted(stats.state)
        self.normalization_action_keys = sorted(stats.action)
        if not self.normalization_state_keys or not self.normalization_action_keys:
            raise ValueError("Checkpoint metadata carries no normalization statistics")
        expected_state = {key.split(".", 1)[1] for key, _, _ in STATE_KEY_ORDER}
        expected_action = {key.split(".", 1)[1] for key, _ in ACTION_KEY_ORDER}
        if set(self.normalization_state_keys) != expected_state:
            raise ValueError(
                f"Checkpoint state statistics {self.normalization_state_keys} "
                f"do not cover {sorted(expected_state)}"
            )
        if set(self.normalization_action_keys) != expected_action:
            raise ValueError(
                f"Checkpoint action statistics {self.normalization_action_keys} "
                f"do not cover {sorted(expected_action)}"
            )

    def _assert_official_contract(self) -> None:
        """profile 선언 == 설치된 `PandaOmronDataConfig`. 다르면 rollout 하지 않는다."""
        cfg = self.data_config
        expected_video = [CAMERA_SLOT_TO_VIDEO_KEY[slot] for slot in CAMERA_SLOT_ORDER]
        if list(cfg.video_keys) != expected_video:
            raise ValueError(f"Official video_keys {cfg.video_keys} != adapter order {expected_video}")
        if list(cfg.state_keys) != [key for key, _, _ in STATE_KEY_ORDER]:
            raise ValueError(f"Official state_keys {cfg.state_keys} != adapter order")
        if list(cfg.action_keys) != [key for key, _ in ACTION_KEY_ORDER]:
            raise ValueError(f"Official action_keys {cfg.action_keys} != adapter order")
        if list(cfg.language_keys) != [LANGUAGE_KEY]:
            raise ValueError(f"Official language_keys {cfg.language_keys} != [{LANGUAGE_KEY!r}]")
        if list(cfg.observation_indices) != [0]:
            raise ValueError(f"Official observation_indices {cfg.observation_indices} != [0]")
        if len(cfg.action_indices) != int(self.profile["action_chunk_size"]):
            raise ValueError(
                f"Official action horizon {len(cfg.action_indices)} != profile "
                f"{self.profile['action_chunk_size']}"
            )
        if int(self.profile["history_length"]) != 1:
            raise ValueError("GR00T observes a single frame; history_length must be 1")
        slots = list(self.profile["camera_slots"])
        if slots != list(CAMERA_SLOT_ORDER):
            raise ValueError(f"Profile camera_slots {slots} != {list(CAMERA_SLOT_ORDER)}")

    def metadata(self) -> dict[str, Any]:
        return {
            "model_profile": self.profile["name"],
            "checkpoint": self.model_path,
            "checkpoint_revision": self.model_revision,
            "resolved_checkpoint_revision": self.resolved_model_revision,
            "checkpoint_subfolder": self.profile["checkpoint_subfolder"],
            "robot_type": self.profile["robot_type"],
            "action_dim": ACTION_DIM,
            "action_convention": self.profile["action_convention"],
            "camera_slots": list(CAMERA_SLOT_ORDER),
            "video_keys": [CAMERA_SLOT_TO_VIDEO_KEY[slot] for slot in CAMERA_SLOT_ORDER],
            "history_length": int(self.profile["history_length"]),
            "history_interval": int(self.profile["history_interval"]),
            "action_chunk_size": int(self.profile["action_chunk_size"]),
            "state_dim_native": STATE_NATIVE_DIM,
            "state_dim_transformed": STATE_TRANSFORMED_DIM,
            "state_dim_padded": STATE_PAD_DIM,
            "resize_target": list(self.profile["resize_target"]),
            "crop_ratio": float(self.profile["crop_ratio"]),
            "data_config": self.profile.get("data_config", DATA_CONFIG_NAME),
            "embodiment_tag": self.embodiment_tag,
            "embodiment_projector_index": self.embodiment_projector_index,
            "denoising_steps": self.denoising_steps,
            "runtime_package": "gr00t",
            "runtime_version": self.runtime_version,
            "runtime_repo": self.profile.get("runtime_repo"),
            "runtime_revision": self.profile.get("runtime_revision"),
            "normalization_state_keys": self.normalization_state_keys,
            "normalization_action_keys": self.normalization_action_keys,
            "device": self.device,
            "episode_seed": self.episode_seed,
            "runtime": {
                "torch": self.torch.__version__,
                "torch_cuda": self.torch.version.cuda,
                "gpu_name": (
                    self.torch.cuda.get_device_name(self.device)
                    if self.device.startswith("cuda")
                    else None
                ),
            },
            "reproducibility": self.reproducibility,
        }

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        requested_seed = payload.get("policy_seed")
        if requested_seed is not None:
            self.episode_seed = int(requested_seed)
            if self.episode_seed < 0:
                raise ValueError("policy_seed cannot be negative")
            np.random.seed(self.episode_seed)
            self.torch.manual_seed(self.episode_seed)
            if self.device.startswith("cuda"):
                self.torch.cuda.manual_seed_all(self.episode_seed)
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
        state_dict = build_state_dict(state_raw[-1])
        native_state = np.concatenate(
            [state_dict[key] for key, _, _ in STATE_KEY_ORDER]
        ).astype(np.float32)

        observation: dict[str, Any] = {
            CAMERA_SLOT_TO_VIDEO_KEY[slot]: images[slot] for slot in CAMERA_SLOT_ORDER
        }
        for key, value in state_dict.items():
            observation[key] = value.reshape(1, -1)
        observation[LANGUAGE_KEY] = np.array([str(payload["instruction"])])

        if self.device.startswith("cuda"):
            self.torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        action_dict = self.policy.get_action(observation)
        inference_ms = (time.perf_counter() - started) * 1000.0

        expected_chunk = int(self.profile["action_chunk_size"])
        actions = concat_action_chunk(action_dict, expected_chunk)
        if not np.isfinite(actions).all():
            raise ValueError("GR00T returned a non-finite action chunk")

        peak_vram_mb = None
        if self.device.startswith("cuda"):
            peak_vram_mb = self.torch.cuda.max_memory_allocated(self.device) / (1024**2)
        image_audit = {
            slot: {
                "video_key": CAMERA_SLOT_TO_VIDEO_KEY[slot],
                "shape": list(image.shape),
                "dtype": str(image.dtype),
                "sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            }
            for slot, image in images.items()
        }
        return {
            "actions": actions.tolist(),
            "metadata": {
                "chunk_length": int(actions.shape[0]),
                "inference_ms": inference_ms,
                "peak_vram_mb": peak_vram_mb,
                "denoising_steps": self.denoising_steps,
            },
            "input_audit": {
                "camera_slots": list(CAMERA_SLOT_ORDER),
                "video_keys": [CAMERA_SLOT_TO_VIDEO_KEY[s] for s in CAMERA_SLOT_ORDER],
                "images": image_audit,
                "instruction_sha256": hashlib.sha256(str(payload["instruction"]).encode("utf-8")).hexdigest(),
                "history_steps": list(payload.get("history_steps", [])),
                "state_native_shape": [expected_history, STATE_NATIVE_DIM],
                "policy_state_shape": [1, 1, STATE_PAD_DIM],
                "state_dim_transformed": STATE_TRANSFORMED_DIM,
                "state_native": [native_state.tolist()],
                "state_native_sha256": hashlib.sha256(native_state.tobytes()).hexdigest(),
                "state_keys": [key for key, _, _ in STATE_KEY_ORDER],
                "action_keys": [key for key, _ in ACTION_KEY_ORDER],
                "embodiment_tag": self.embodiment_tag,
                "preprocess": {
                    "crop_ratio": float(self.profile["crop_ratio"]),
                    "resize_target": list(self.profile["resize_target"]),
                },
            },
        }
