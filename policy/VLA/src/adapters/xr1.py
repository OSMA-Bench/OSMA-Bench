"""XR-1 RoboCasa365 adapter matching Xiaomi's official evaluator contract."""

from __future__ import annotations

import base64
import hashlib
import time
from typing import Any

import numpy as np


CAMERA_KEYS = ("left", "right", "wrist")
STATE_DIM = 60
ACTION_DIM = 12


def decode_array(encoded: dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(encoded["dtype"])
    shape = tuple(int(value) for value in encoded["shape"])
    raw = base64.b64decode(encoded["data_b64"], validate=True)
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(raw) != expected_bytes:
        raise ValueError(f"Encoded array has {len(raw)} bytes; expected {expected_bytes}")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def quat_xyzw_to_axis_angle(quaternion: Any) -> np.ndarray:
    """Exact conversion used by Xiaomi's eval_robocasa365/entry.py."""
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if quaternion.shape != (4,):
        raise ValueError(f"Expected an xyzw quaternion, got {quaternion.shape}")
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    quaternion = quaternion / norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    xyz = quaternion[:3]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, np.clip(quaternion[3], -1.0, 1.0))
    return (xyz / sin_half * angle).astype(np.float32)


def build_state14(observation: dict[str, Any]) -> np.ndarray:
    """Build XR-1's EE-first 14D state from canonical raw observations."""
    state = np.concatenate(
        [
            np.asarray(observation["end_effector_position_relative"], dtype=np.float32).reshape(-1),
            quat_xyzw_to_axis_angle(observation["end_effector_rotation_relative"]),
            np.asarray(observation["gripper_qpos"], dtype=np.float32).reshape(-1),
            np.asarray(observation["base_position"], dtype=np.float32).reshape(-1),
            quat_xyzw_to_axis_angle(observation["base_rotation"]),
        ]
    ).astype(np.float32)
    if state.shape != (14,):
        raise ValueError(f"Expected a 14D RoboCasa365 state, got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("XR-1 state contains NaN or Inf")
    return state


def pad_state_history(state_history: Any) -> tuple[np.ndarray, np.ndarray]:
    native = np.stack([build_state14(item) for item in state_history], axis=0)
    padded = np.zeros((1, native.shape[0], STATE_DIM), dtype=np.float32)
    padded[0, :, : native.shape[-1]] = native
    return native, padded


class XR1Adapter:
    def __init__(
        self,
        profile: dict[str, Any],
        model_path: str,
        model_revision: str,
        device: str = "cuda:0",
        attention_implementation: str = "flash_attention_2",
        reproducibility: dict[str, Any] | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.profile = dict(profile)
        self.model_path = model_path
        self.model_revision = model_revision
        self.device = device
        self.reproducibility = reproducibility or {}
        self.crop_ratio = float(profile.get("crop_ratio", 0.95))
        self.robot_type = str(profile["robot_type"])

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            revision=model_revision,
        )
        robot_types = list(self.processor.list_robot_types())
        if self.robot_type not in robot_types:
            raise ValueError(
                f"Robot type {self.robot_type!r} is absent from checkpoint; available: {robot_types}"
            )
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation=attention_implementation,
            dtype=torch.bfloat16,
            revision=model_revision,
        ).to(device=device, dtype=torch.bfloat16)
        self.model.eval()
        self.resolved_model_revision = getattr(self.model.config, "_commit_hash", None)
        if self.resolved_model_revision is not None and self.resolved_model_revision != model_revision:
            raise ValueError(
                f"Loaded checkpoint revision {self.resolved_model_revision} does not match {model_revision}"
            )
        self.robot_types = robot_types
        self.episode_seed: int | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "model_profile": self.profile["name"],
            "checkpoint": self.model_path,
            "checkpoint_revision": self.model_revision,
            "resolved_checkpoint_revision": self.resolved_model_revision,
            "robot_type": self.robot_type,
            "available_robot_types": self.robot_types,
            "action_dim": ACTION_DIM,
            "action_convention": self.profile["action_convention"],
            "camera_slots": list(CAMERA_KEYS),
            "history_length": int(self.profile["history_length"]),
            "history_interval": int(self.profile["history_interval"]),
            "crop_ratio": self.crop_ratio,
            "state_order": [
                "end_effector_position_relative[3]",
                "end_effector_rotation_relative_axis_angle[3]",
                "gripper_qpos[2]",
                "base_position[3]",
                "base_rotation_axis_angle[3]",
            ],
            "state_dim_native": 14,
            "state_dim_padded": STATE_DIM,
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

    def _center_crop(self, image: np.ndarray) -> Any:
        from PIL import Image

        pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
        if self.crop_ratio >= 1.0:
            return pil_image
        width, height = pil_image.size
        crop_width = max(1, int(width * self.crop_ratio))
        crop_height = max(1, int(height * self.crop_ratio))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        cropped = pil_image.crop((left, top, left + crop_width, top + crop_height))
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        return cropped.resize((width, height), resampling)

    def _build_messages(self, images: dict[str, np.ndarray], instruction: str) -> list[dict[str, Any]]:
        videos = {
            key: [self._center_crop(frame) for frame in images[key]]
            for key in CAMERA_KEYS
        }
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Left camera: "},
                    {"type": "video", "video": videos["left"]},
                    {"type": "text", "text": "\nRight camera: "},
                    {"type": "video", "video": videos["right"]},
                    {"type": "text", "text": "\nWrist camera: "},
                    {"type": "video", "video": videos["wrist"]},
                    {
                        "type": "text",
                        "text": f"\n\nGenerate robot actions for the task:\n{instruction} /no_cot",
                    },
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported request schema")
        expected_history = int(self.profile["history_length"])
        images = {key: decode_array(payload["images"][key]) for key in CAMERA_KEYS}
        expected_image_shape = (
            expected_history,
            int(self.profile["image_size"]),
            int(self.profile["image_size"]),
            3,
        )
        for key, value in images.items():
            if value.dtype != np.uint8 or value.shape != expected_image_shape:
                raise ValueError(
                    f"Camera {key} is {value.dtype} {value.shape}; expected uint8 {expected_image_shape}"
                )
        if len(payload["state_raw"]) != expected_history:
            raise ValueError(
                f"Received {len(payload['state_raw'])} states; expected {expected_history}"
            )
        native_state, padded_state = pad_state_history(payload["state_raw"])
        instruction = str(payload["instruction"])

        inputs = self.processor.apply_chat_template(
            self._build_messages(images, instruction),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            do_resize=False,
            state=padded_state,
            robot_type=self.robot_type,
        )
        request = dict(inputs)
        request["task_id"] = self.robot_type
        for key, value in list(request.items()):
            if isinstance(value, self.torch.Tensor):
                if value.is_floating_point():
                    request[key] = value.to(device=self.device, dtype=self.model.dtype)
                else:
                    request[key] = value.to(device=self.device)

        if self.device.startswith("cuda"):
            self.torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model(**request)
        decoded = self.processor.decode_action(outputs.actions.cpu(), robot_type=self.robot_type)
        actions = decoded[0, :, :ACTION_DIM].float().cpu().numpy().astype(np.float32, copy=False)
        inference_ms = (time.perf_counter() - started) * 1000.0
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not np.isfinite(actions).all():
            raise ValueError(f"XR-1 returned an invalid action chunk: {actions.shape}")

        peak_vram_mb = None
        if self.device.startswith("cuda"):
            peak_vram_mb = self.torch.cuda.max_memory_allocated(self.device) / (1024**2)
        image_audit = {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
            for key, value in images.items()
        }
        return {
            "actions": actions.tolist(),
            "metadata": {
                "chunk_length": int(actions.shape[0]),
                "inference_ms": inference_ms,
                "peak_vram_mb": peak_vram_mb,
            },
            "input_audit": {
                "camera_slots": list(CAMERA_KEYS),
                "images": image_audit,
                "instruction_sha256": hashlib.sha256(str(payload["instruction"]).encode("utf-8")).hexdigest(),
                "history_steps": list(payload.get("history_steps", [])),
                "state_native_shape": list(native_state.shape),
                "policy_state_shape": list(padded_state.shape),
                "state_native": native_state.tolist(),
                "state_native_sha256": hashlib.sha256(native_state.tobytes()).hexdigest(),
                "preprocess": {"crop_ratio": self.crop_ratio},
                "state14_shape": list(native_state.shape),
                "state14": native_state.tolist(),
                "state_raw": payload["state_raw"],
                "state60_shape": list(padded_state.shape),
                "state14_sha256": hashlib.sha256(native_state.tobytes()).hexdigest(),
                "crop_ratio": self.crop_ratio,
            },
        }
