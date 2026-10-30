"""Load one VLA adapter and run one deterministic synthetic G1 inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from model_config import DEFAULT_CONFIG_PATH, load_model_profile
from policy_client import encode_array
from reproducibility import write_json_file
from vla_server import import_symbol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-profile", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def synthetic_state() -> dict[str, np.ndarray]:
    return {
        "end_effector_position_relative": np.array([0.45, 0.0, 0.55], dtype=np.float32),
        "end_effector_rotation_relative": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "gripper_qpos": np.zeros(2, dtype=np.float32),
        "base_position": np.zeros(3, dtype=np.float32),
        "base_rotation": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }


def main() -> None:
    args = parse_args()
    profile = load_model_profile(args.model_profile, args.config)
    adapter_class = import_symbol(profile["adapter"])
    adapter = adapter_class(
        profile=profile,
        model_path=args.model_path,
        model_revision=profile["checkpoint_revision"],
        device=args.device,
        attention_implementation="eager",
        reproducibility={"scope": "G1 synthetic single inference"},
    )
    adapter.reset({"instruction": "navigate to the sink", "policy_seed": 0})
    history_length = int(profile["history_length"])
    image_shape = (
        history_length,
        int(profile["image_size"]),
        int(profile["image_size"]),
        3,
    )
    images = {
        slot: encode_array(np.zeros(image_shape, dtype=np.uint8))
        for slot in profile["camera_slots"]
    }
    payload = {
        "schema_version": 1,
        "instruction": "navigate to the sink",
        "step_id": 0,
        "history_steps": [0] * history_length,
        "images": images,
        "state_raw": [synthetic_state() for _ in range(history_length)],
    }
    response = adapter.act(payload)
    actions = np.asarray(response["actions"], dtype=np.float32)
    result = {
        "schema_version": 1,
        "profile": args.model_profile,
        "model_path": str(Path(args.model_path).resolve()),
        "model_revision": profile["checkpoint_revision"],
        "device": args.device,
        "action_shape": list(actions.shape),
        "action_finite": bool(np.isfinite(actions).all()),
        "action_min": float(np.min(actions)),
        "action_max": float(np.max(actions)),
        "inference": response["metadata"],
        "adapter_metadata": adapter.metadata(),
        "input_audit": response["input_audit"],
    }
    if args.output:
        write_json_file(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
