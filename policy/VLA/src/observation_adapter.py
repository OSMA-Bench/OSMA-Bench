"""Model-agnostic extraction of images and raw PandaOmron proprioception."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any

import numpy as np


CAMERA_NAME_BY_SLOT = {
    "left": "robot0_agentview_left",
    "right": "robot0_agentview_right",
    "wrist": "robot0_eye_in_hand",
}

RAW_STATE_KEY_MAP = {
    "end_effector_position_relative": "robot0_base_to_eef_pos",
    "end_effector_rotation_relative": "robot0_base_to_eef_quat",
    "gripper_qpos": "robot0_gripper_qpos",
    "base_position": "robot0_base_pos",
    "base_rotation": "robot0_base_quat",
}

RAW_STATE_SHAPES = {
    "end_effector_position_relative": (3,),
    "end_effector_rotation_relative": (4,),
    "gripper_qpos": (2,),
    "base_position": (3,),
    "base_rotation": (4,),
}


def capture_camera_images(env: Any, image_size: int, camera_slots: list[str]) -> dict[str, np.ndarray]:
    """Render upright uint8 HWC images from the three robot cameras."""
    images: dict[str, np.ndarray] = {}
    for slot in camera_slots:
        if slot not in CAMERA_NAME_BY_SLOT:
            raise KeyError(f"Unknown camera slot {slot!r}")
        camera_name = CAMERA_NAME_BY_SLOT[slot]
        image = env.sim.render(height=image_size, width=image_size, camera_name=camera_name)[::-1]
        image = np.ascontiguousarray(image, dtype=np.uint8)
        expected = (image_size, image_size, 3)
        if image.shape != expected:
            raise ValueError(f"{camera_name} returned {image.shape}; expected {expected}")
        images[slot] = image
    return images


def extract_raw_state(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    """Extract the canonical raw fields used by RoboCasa365 model adapters.

    The local v0.2 keys are deliberately strict. Falling back from
    ``base_to_eef`` to world-frame ``eef`` would keep the shape valid while
    silently feeding XR-1 the wrong values.
    """
    missing = [source for source in RAW_STATE_KEY_MAP.values() if source not in observation]
    if missing:
        available = ", ".join(sorted(observation))
        raise KeyError(f"Missing required local observations {missing}; available keys: {available}")

    result: dict[str, np.ndarray] = {}
    for canonical, source in RAW_STATE_KEY_MAP.items():
        value = np.asarray(observation[source], dtype=np.float32).reshape(-1)
        if value.shape != RAW_STATE_SHAPES[canonical]:
            raise ValueError(
                f"{source} has shape {value.shape}; expected {RAW_STATE_SHAPES[canonical]}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"{source} contains NaN or Inf")
        result[canonical] = value.copy()
    return result


@dataclass(frozen=True)
class HistorySample:
    images: dict[str, np.ndarray]
    state_raw: list[dict[str, np.ndarray]]
    steps: list[int]


class ObservationHistory:
    """Keep and sample observations exactly like XR-1's official evaluator."""

    def __init__(self, length: int, interval: int, camera_slots: list[str]):
        if length < 1 or interval < 1:
            raise ValueError("length and interval must be positive")
        self.length = length
        self.interval = interval
        self.camera_slots = list(camera_slots)
        self.queue_length = (length - 1) * interval + 1
        self._items: collections.deque[
            tuple[int, dict[str, np.ndarray], dict[str, np.ndarray]]
        ] = collections.deque(maxlen=self.queue_length)

    def append(
        self,
        step: int,
        images: dict[str, np.ndarray],
        state_raw: dict[str, np.ndarray],
    ) -> None:
        if set(images) != set(self.camera_slots):
            raise ValueError(f"Camera slots are {sorted(images)}; expected {sorted(self.camera_slots)}")
        copied_images = {key: np.ascontiguousarray(value).copy() for key, value in images.items()}
        copied_state = {key: np.asarray(value, dtype=np.float32).copy() for key, value in state_raw.items()}
        self._items.append((int(step), copied_images, copied_state))

    def sample(self) -> HistorySample:
        items = list(self._items)
        if not items:
            raise ValueError("Observation history cannot be empty")
        indices = [
            max(0, len(items) - 1 - (self.length - 1 - index) * self.interval)
            for index in range(self.length)
        ]
        selected = [items[index] for index in indices]
        images = {
            slot: np.ascontiguousarray(np.stack([item[1][slot] for item in selected], axis=0))
            for slot in self.camera_slots
        }
        states = [
            {key: value.copy() for key, value in item[2].items()}
            for item in selected
        ]
        return HistorySample(images=images, state_raw=states, steps=[item[0] for item in selected])
