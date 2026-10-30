"""Model-agnostic HTTP client for a VLA policy server."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np


def encode_array(array: np.ndarray) -> dict[str, Any]:
    value = np.ascontiguousarray(array)
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "data_b64": base64.b64encode(value.tobytes()).decode("ascii"),
    }


@dataclass(frozen=True)
class PolicyResponse:
    actions: np.ndarray
    metadata: dict[str, Any]
    input_audit: dict[str, Any]


class PolicyClient:
    def __init__(self, base_url: str, timeout_seconds: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Policy server {path} returned HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise ConnectionError(f"Cannot reach policy server at {self.base_url}: {error}") from error

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def meta(self) -> dict[str, Any]:
        return self._request("GET", "/meta")

    def reset(
        self, instruction: str, model_profile: str, policy_seed: int | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "instruction": instruction,
            "model_profile": model_profile,
        }
        if policy_seed is not None:
            payload["policy_seed"] = int(policy_seed)
        return self._request(
            "POST",
            "/reset",
            payload,
        )

    def act(
        self,
        images: dict[str, np.ndarray],
        state_raw: list[dict[str, np.ndarray]],
        instruction: str,
        step_id: int,
        history_steps: list[int],
        sampling: dict[str, Any] | None = None,
    ) -> PolicyResponse:
        payload = {
            "schema_version": 1,
            "instruction": instruction,
            "step_id": int(step_id),
            "history_steps": [int(step) for step in history_steps],
            "images": {key: encode_array(value) for key, value in images.items()},
            "state_raw": [
                {key: np.asarray(value, dtype=np.float32).tolist() for key, value in state.items()}
                for state in state_raw
            ],
        }
        # Adapters list the keys they read in /meta act_sampling_keys; without it the server samples as before.
        if sampling is not None:
            payload["sampling"] = dict(sampling)
        response = self._request("POST", "/act", payload)
        return PolicyResponse(
            actions=np.asarray(response["actions"], dtype=np.float32),
            metadata=dict(response.get("metadata", {})),
            input_audit=dict(response.get("input_audit", {})),
        )
