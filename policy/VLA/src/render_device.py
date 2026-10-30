"""Resolve MuJoCo EGL ordinals to physical NVIDIA GPU indices.

EGL enumerates DRM devices, which need not match CUDA / ``nvidia-smi`` order.
The resolved mapping is recorded so a rollout cannot silently render on another
user's GPU while its CLI says ``--render-gpu-device-id 0``.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any


_PCI_SUFFIX = re.compile(r"([0-9a-f]{2}:[0-9a-f]{2}\.[0-7])$", re.IGNORECASE)


def normalize_pci_bus_id(value: str) -> str:
    match = _PCI_SUFFIX.search(str(value).strip())
    if match is None:
        raise ValueError(f"Cannot parse PCI bus id from {value!r}")
    return match.group(1).lower()


def choose_render_device(
    mappings: list[dict[str, Any]],
    *,
    egl_device_id: int | None,
    physical_gpu_index: int | None,
) -> dict[str, Any]:
    """Select one mapping and cross-check EGL and physical indices when both are given."""
    if egl_device_id is None and physical_gpu_index is None:
        raise ValueError(
            "Specify --render-gpu-physical-index after gpu-check (or an audited EGL device id)"
        )
    candidates = mappings
    if egl_device_id is not None:
        candidates = [m for m in candidates if m["egl_device_id"] == egl_device_id]
    if physical_gpu_index is not None:
        candidates = [m for m in candidates if m["physical_gpu_index"] == physical_gpu_index]
    if len(candidates) != 1:
        raise ValueError(
            "EGL/physical GPU selection does not resolve uniquely: "
            f"egl={egl_device_id}, physical={physical_gpu_index}, mappings={mappings}"
        )
    return dict(candidates[0])


def query_render_device_mappings() -> list[dict[str, Any]]:
    """Query EGL→DRM→PCI→nvidia-smi mapping in the simulator runtime."""
    from mujoco.egl import egl_ext as egl
    from OpenGL.EGL.EXT import device_drm, device_query

    gpu_by_pci: dict[str, int] = {}
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for line in output.splitlines():
        index, pci_bus_id = (part.strip() for part in line.split(",", 1))
        gpu_by_pci[normalize_pci_bus_id(pci_bus_id)] = int(index)

    mappings = []
    for egl_device_id, device in enumerate(egl.eglQueryDevicesEXT()):
        raw_drm = device_query.eglQueryDeviceStringEXT(
            device, device_drm.EGL_DRM_DEVICE_FILE_EXT
        )
        if not raw_drm:
            continue
        drm_device = raw_drm.decode() if isinstance(raw_drm, bytes) else str(raw_drm)
        sysfs_device = (Path("/sys/class/drm") / Path(drm_device).name / "device").resolve()
        pci_bus_id = normalize_pci_bus_id(sysfs_device.name)
        if pci_bus_id not in gpu_by_pci:
            continue
        mappings.append(
            {
                "egl_device_id": egl_device_id,
                "drm_device": drm_device,
                "pci_bus_id": pci_bus_id,
                "physical_gpu_index": gpu_by_pci[pci_bus_id],
            }
        )
    if not mappings:
        raise RuntimeError("No EGL device could be mapped to an nvidia-smi GPU")
    return mappings


def resolve_render_device(
    *,
    egl_device_id: int | None,
    physical_gpu_index: int | None,
) -> dict[str, Any]:
    return choose_render_device(
        query_render_device_mappings(),
        egl_device_id=egl_device_id,
        physical_gpu_index=physical_gpu_index,
    )
