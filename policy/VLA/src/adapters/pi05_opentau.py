"""π0.5 OpenTau adapter for TensorAuto's RoboCasa NavigateKitchen checkpoint."""

from __future__ import annotations

import hashlib
import importlib.metadata
import time
from pathlib import Path
from typing import Any

import numpy as np

from adapters.xr1 import decode_array


CAMERA_SLOT_ORDER = ("wrist", "left", "right")
STATE_NATIVE_DIM = 16
STATE_PAD_DIM = 32
ACTION_DIM = 12


def resolve_hf_snapshot(hub_root: Path, repo_id: str, revision: str) -> Path:
    repo_dir = "models--" + repo_id.replace("/", "--")
    snapshot = hub_root / repo_dir / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(
            f"Required offline asset {repo_id}@{revision} is absent at {snapshot}"
        )
    return snapshot


def build_state16(observation: dict[str, Any]) -> np.ndarray:
    """Build the official base-first PandaOmron state vector."""
    state = np.concatenate(
        [
            np.asarray(observation["base_position"], dtype=np.float32).reshape(-1),
            np.asarray(observation["base_rotation"], dtype=np.float32).reshape(-1),
            np.asarray(
                observation["end_effector_position_relative"], dtype=np.float32
            ).reshape(-1),
            np.asarray(
                observation["end_effector_rotation_relative"], dtype=np.float32
            ).reshape(-1),
            np.asarray(observation["gripper_qpos"], dtype=np.float32).reshape(-1),
        ]
    ).astype(np.float32)
    if state.shape != (STATE_NATIVE_DIM,):
        raise ValueError(f"Expected a {STATE_NATIVE_DIM}D RoboCasa state, got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("π0.5 state contains NaN or Inf")
    return state


def pad_state32(state: Any) -> np.ndarray:
    native = np.asarray(state, dtype=np.float32).reshape(-1)
    if native.shape != (STATE_NATIVE_DIM,):
        raise ValueError(f"Expected state16, got {native.shape}")
    padded = np.zeros(STATE_PAD_DIM, dtype=np.float32)
    padded[:STATE_NATIVE_DIM] = native
    return padded


def _numpy_rgb_to_camera_tensor(
    rgb_uint8: np.ndarray,
    resolution: tuple[int, int],
    device: Any,
    dtype: Any,
    torch_module: Any,
) -> Any:
    from PIL import Image

    pil = Image.fromarray(np.asarray(rgb_uint8, dtype=np.uint8))
    pil = pil.resize((resolution[1], resolution[0]), Image.Resampling.BILINEAR)
    array = np.asarray(pil, dtype=np.float32) / 255.0
    tensor = torch_module.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)


def build_opentau_batch(
    images: dict[str, np.ndarray],
    state32: np.ndarray,
    prompt: str,
    resolution: tuple[int, int],
    device: Any,
    dtype: Any,
    torch_module: Any,
) -> dict[str, Any]:
    """Transcribe OpenTau v0.13.0's official batch builder for camera slots B."""
    batch: dict[str, Any] = {}
    for camera_index, slot in enumerate(CAMERA_SLOT_ORDER):
        history = np.asarray(images[slot])
        if history.ndim != 4 or history.shape[0] != 1:
            raise ValueError(f"Camera {slot} must contain exactly one HWC frame, got {history.shape}")
        batch[f"camera{camera_index}"] = _numpy_rgb_to_camera_tensor(
            history[0], resolution, device, dtype, torch_module
        )
    state_list = np.asarray(state32, dtype=np.float64).ravel().tolist()
    if len(state_list) != STATE_PAD_DIM:
        raise ValueError(f"Expected state32, got {len(state_list)} values")
    batch["state"] = torch_module.tensor([state_list], dtype=dtype, device=device)
    batch["prompt"] = [str(prompt).strip() if prompt else ""]
    batch["img_is_pad"] = torch_module.tensor(
        [[False] * len(CAMERA_SLOT_ORDER)], dtype=torch_module.bool, device=device
    )
    return batch


class Pi05OpenTauAdapter:
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
        import opentau.policies.pi05.modeling_pi05 as pi05_modeling
        from opentau.configs.train import TrainPipelineConfig
        from opentau.policies.factory import get_policy_class
        from opentau.policies.utils import to_dtype_preserving_siglip_float32

        del attention_implementation  # checkpoint config is the source of truth
        self.torch = torch
        self.profile = dict(profile)
        self.model_path = str(Path(model_path).resolve())
        self.model_revision = model_revision
        self.device = device
        self.dtype = torch.bfloat16
        self.reproducibility = reproducibility or {}
        self.episode_seed: int | None = None

        snapshot = Path(self.model_path)
        if not snapshot.is_dir():
            raise ValueError(
                "π0.5 requires a local immutable checkpoint snapshot; pass --model-path "
                "ending in the pinned revision"
            )
        if snapshot.name != model_revision:
            raise ValueError(
                f"Snapshot directory {snapshot.name!r} does not match revision {model_revision!r}"
            )
        self.resolved_model_revision = snapshot.name
        hub_root = snapshot.parents[2]
        self.language_tokenizer_path = resolve_hf_snapshot(
            hub_root,
            str(profile["language_tokenizer"]),
            str(profile["language_tokenizer_revision"]),
        )
        self.action_tokenizer_path = resolve_hf_snapshot(
            hub_root,
            str(profile["action_tokenizer"]),
            str(profile["action_tokenizer_revision"]),
        )
        self.runtime_version = importlib.metadata.version("opentau")
        expected_runtime = str(profile["runtime_version"])
        if self.runtime_version != expected_runtime:
            raise ValueError(
                f"OpenTau runtime {self.runtime_version} does not match {expected_runtime}"
            )

        self.cfg = TrainPipelineConfig.from_pretrained(
            snapshot, local_files_only=True
        )
        self.cfg.policy.pretrained_path = self.model_path
        self.cfg.policy.discrete_action_tokenizer_path = str(self.action_tokenizer_path)
        if tuple(self.cfg.resolution) != (profile["image_size"], profile["image_size"]):
            raise ValueError(f"Checkpoint resolution {self.cfg.resolution} disagrees with profile")
        if self.cfg.num_cams != len(CAMERA_SLOT_ORDER):
            raise ValueError(f"Checkpoint num_cams={self.cfg.num_cams}; expected 3")
        if self.cfg.max_state_dim != STATE_PAD_DIM or self.cfg.max_action_dim < ACTION_DIM:
            raise ValueError("Checkpoint state/action dimensions disagree with adapter contract")
        if self.cfg.action_chunk != profile["action_chunk_size"]:
            raise ValueError("Checkpoint action chunk disagrees with profile")

        policy_class = get_policy_class(self.cfg.policy.type)
        official_auto_tokenizer = pi05_modeling.AutoTokenizer

        class PinnedAutoTokenizer:
            @classmethod
            def from_pretrained(cls, pretrained_name_or_path: Any, *args: Any, **kwargs: Any) -> Any:
                if str(pretrained_name_or_path) != str(profile["language_tokenizer"]):
                    raise ValueError(
                        f"Unexpected π0.5 language tokenizer source {pretrained_name_or_path!r}"
                    )
                kwargs["local_files_only"] = True
                return official_auto_tokenizer.from_pretrained(
                    self.language_tokenizer_path, *args, **kwargs
                )

        pi05_modeling.AutoTokenizer = PinnedAutoTokenizer
        try:
            self.policy = policy_class.from_pretrained(
                snapshot, config=self.cfg.policy, local_files_only=True
            )
        finally:
            pi05_modeling.AutoTokenizer = official_auto_tokenizer
        to_dtype_preserving_siglip_float32(
            self.policy, device=torch.device(device), dtype=self.dtype
        )
        self.policy.eval()
        self.policy.reset()
        # OpenTau 0.13.0 은 정규화 통계를 buffer 가 아니라 requires_grad=False 인
        # nn.Parameter 로 등록하고, checkpoint 의 (32,) 를 (1,32) 로 promote 한다.
        # buffer 만 조회하면 통계가 정상 로드돼도 0개로 보여 거짓 실패한다.
        normalization_stats = {
            name: tensor
            for name, tensor in (
                list(self.policy.named_buffers()) + list(self.policy.named_parameters())
            )
            if name.startswith(("normalize_", "unnormalize_"))
        }
        self.normalization_buffer_keys = sorted(normalization_stats)
        if not self.normalization_buffer_keys:
            raise ValueError("Checkpoint loaded without normalization-stat buffers")
        # 키만 있고 값이 안 실린 checkpoint 를 거른다. std==0 은 검사하지 않는다 —
        # base-first action 의 index 3(torso)·11(gripper)과 12 이후 padding 은
        # navigation 데모에서 상수라 std 가 0 인 것이 정상이다.
        for name, tensor in normalization_stats.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Normalization stat {name} has non-finite values")
            if not bool((tensor != 0).any()):
                raise ValueError(f"Normalization stat {name} is all zeros")

    def metadata(self) -> dict[str, Any]:
        return {
            "model_profile": self.profile["name"],
            "checkpoint": self.model_path,
            "checkpoint_revision": self.model_revision,
            "resolved_checkpoint_revision": self.resolved_model_revision,
            "robot_type": self.profile["robot_type"],
            "action_dim": ACTION_DIM,
            "action_convention": self.profile["action_convention"],
            "camera_slots": list(CAMERA_SLOT_ORDER),
            "history_length": int(self.profile["history_length"]),
            "history_interval": int(self.profile["history_interval"]),
            "action_chunk_size": int(self.profile["action_chunk_size"]),
            "state_dim_native": STATE_NATIVE_DIM,
            "state_dim_padded": STATE_PAD_DIM,
            "act_sampling_keys": ["noise_scale"],
            "resize_target": list(self.profile["resize_target"]),
            "runtime_version": self.runtime_version,
            "runtime_revision": self.profile["runtime_revision"],
            "language_tokenizer": {
                "repo": self.profile["language_tokenizer"],
                "revision": self.profile["language_tokenizer_revision"],
                "path": str(self.language_tokenizer_path),
            },
            "action_tokenizer": {
                "repo": self.profile["action_tokenizer"],
                "revision": self.profile["action_tokenizer_revision"],
                "path": str(self.action_tokenizer_path),
            },
            "normalization_buffer_keys": self.normalization_buffer_keys,
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
        self.policy.reset()
        return {
            "ok": True,
            "instruction": payload.get("instruction", ""),
            "policy_seed": self.episode_seed,
        }

    def _requested_noise(self, sampling: Any) -> Any:
        """sampling={"noise_scale": s}: OpenTau's own initial-noise draw (sample_noise: a standard normal
        from the global torch RNG, which /reset seeds from policy_seed), times s."""
        if not isinstance(sampling, dict) or set(sampling) != {"noise_scale"}:
            raise ValueError(f"sampling must be {{'noise_scale': positive number}}, got {sampling!r}")
        scale = sampling["noise_scale"]
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not scale > 0:
            raise ValueError(f"sampling.noise_scale must be a positive number, got {scale!r}")
        shape = (1, int(self.policy.config.chunk_size), int(self.policy.config.max_action_dim))
        noise = self.torch.normal(mean=0.0, std=1.0, size=shape, dtype=self.torch.float32, device=self.device)
        return noise * float(scale)

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported request schema")
        expected_history = int(self.profile["history_length"])
        images = {slot: decode_array(payload["images"][slot]) for slot in CAMERA_SLOT_ORDER}
        expected_image_shape = (
            expected_history,
            int(self.profile["image_size"]),
            int(self.profile["image_size"]),
            3,
        )
        for slot, image in images.items():
            if image.dtype != np.uint8 or image.shape != expected_image_shape:
                raise ValueError(
                    f"Camera {slot} is {image.dtype} {image.shape}; "
                    f"expected uint8 {expected_image_shape}"
                )
        state_raw = payload["state_raw"]
        if len(state_raw) != expected_history:
            raise ValueError(f"Received {len(state_raw)} states; expected {expected_history}")
        native_state = np.stack([build_state16(item) for item in state_raw], axis=0)
        state32 = pad_state32(native_state[-1])
        batch = build_opentau_batch(
            images=images,
            state32=state32,
            prompt=str(payload["instruction"]),
            resolution=tuple(int(value) for value in self.cfg.resolution),
            device=self.torch.device(self.device),
            dtype=self.dtype,
            torch_module=self.torch,
        )

        if self.device.startswith("cuda"):
            self.torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            noise = None if payload.get("sampling") is None else self._requested_noise(payload["sampling"])
            output = self.policy.sample_actions(batch, noise=noise)
        inference_ms = (time.perf_counter() - started) * 1000.0
        actions = output.squeeze(0).to("cpu", self.torch.float32).numpy()
        actions = np.ascontiguousarray(actions[:, :ACTION_DIM], dtype=np.float32)
        expected_chunk = int(self.profile["action_chunk_size"])
        if actions.shape != (expected_chunk, ACTION_DIM) or not np.isfinite(actions).all():
            raise ValueError(f"π0.5 returned an invalid action chunk: {actions.shape}")

        peak_vram_mb = None
        if self.device.startswith("cuda"):
            peak_vram_mb = self.torch.cuda.max_memory_allocated(self.device) / (1024**2)
        image_audit = {
            slot: {
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
                **({"sampling": payload["sampling"]} if payload.get("sampling") is not None else {}),
            },
            "input_audit": {
                "camera_slots": list(CAMERA_SLOT_ORDER),
                "images": image_audit,
                "instruction_sha256": hashlib.sha256(str(payload["instruction"]).encode("utf-8")).hexdigest(),
                "history_steps": list(payload.get("history_steps", [])),
                "state_native_shape": list(native_state.shape),
                "policy_state_shape": [1, STATE_PAD_DIM],
                "state_native": native_state.tolist(),
                "state_native_sha256": hashlib.sha256(native_state.tobytes()).hexdigest(),
                "preprocess": {"resize_target": list(self.profile["resize_target"])},
                "state16": native_state.tolist(),
                "state32_shape": [1, STATE_PAD_DIM],
                "state32_sha256": hashlib.sha256(state32.tobytes()).hexdigest(),
            },
        }
