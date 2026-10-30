"""Load and validate model profiles shared by the simulator and model server."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "models.yaml"


def validate_checkpoint_revision(value: Any) -> str:
    revision = str(value)
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("checkpoint revision must be an immutable 40-character Git commit SHA")
    return revision


def load_model_profile(name: str, config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as stream:
        # JSON is a strict YAML subset. Keeping this file JSON-compatible lets
        # the simulator/client use it without installing PyYAML.
        document = json.load(stream)

    if not isinstance(document, dict) or document.get("schema_version") != 3:
        raise ValueError(f"Unsupported model config schema in {path}")
    profiles = document.get("profiles")
    if not isinstance(profiles, dict) or name not in profiles:
        choices = ", ".join(sorted(profiles or {}))
        raise KeyError(f"Unknown model profile {name!r}; available: {choices}")

    profile = dict(profiles[name])
    required = {
        "adapter",
        "checkpoint",
        "checkpoint_revision",
        "robot_type",
        "action_dim",
        "action_convention",
        "camera_slots",
        "image_size",
        "history_length",
        "history_interval",
        "action_chunk_size",
        "replan_steps",
        "input_audit",
    }
    missing = sorted(required - set(profile))
    if missing:
        raise ValueError(f"Profile {name!r} is missing keys: {missing}")
    if profile["action_dim"] != 12:
        raise ValueError(f"This milestone requires a 12D action profile, got {profile['action_dim']}")
    if len(profile["camera_slots"]) != 3:
        raise ValueError("This milestone requires exactly three camera slots")
    if profile["history_length"] < 1 or profile["history_interval"] < 1:
        raise ValueError("History length and interval must be positive")
    if profile["action_chunk_size"] < 1 or profile["replan_steps"] < 1:
        raise ValueError("Action chunk size and replan steps must be positive")
    if profile["replan_steps"] > profile["action_chunk_size"]:
        raise ValueError("replan_steps cannot exceed action_chunk_size")
    if not isinstance(profile["input_audit"], dict):
        raise ValueError("input_audit must be a mapping")
    profile["checkpoint_revision"] = validate_checkpoint_revision(profile["checkpoint_revision"])
    for revision_key in (
        "runtime_revision",
        "language_tokenizer_revision",
        "action_tokenizer_revision",
    ):
        if revision_key in profile:
            profile[revision_key] = validate_checkpoint_revision(profile[revision_key])
    profile["name"] = name
    profile["profile_schema_version"] = document["schema_version"]
    profile["config_path"] = str(path.resolve())
    return profile
