"""EGL ordinal ↔ physical GPU selection regression tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from render_device import choose_render_device, normalize_pci_bus_id  # noqa: E402


MAPPINGS = [
    {"egl_device_id": 0, "physical_gpu_index": 2, "drm_device": "/dev/dri/card1"},
    {"egl_device_id": 1, "physical_gpu_index": 3, "drm_device": "/dev/dri/card2"},
    {"egl_device_id": 2, "physical_gpu_index": 1, "drm_device": "/dev/dri/card3"},
    {"egl_device_id": 3, "physical_gpu_index": 0, "drm_device": "/dev/dri/card4"},
]


class TestRenderDevice(unittest.TestCase):
    def test_normalizes_nvidia_and_sysfs_bus_ids(self):
        self.assertEqual(normalize_pci_bus_id("00000000:21:00.0"), "21:00.0")
        self.assertEqual(normalize_pci_bus_id("0000:21:00.0"), "21:00.0")

    def test_resolves_physical_index_to_egl_ordinal(self):
        selected = choose_render_device(
            MAPPINGS, egl_device_id=None, physical_gpu_index=1
        )
        self.assertEqual(selected["egl_device_id"], 2)

    def test_cross_checks_both_indices(self):
        with self.assertRaisesRegex(ValueError, "does not resolve uniquely"):
            choose_render_device(MAPPINGS, egl_device_id=0, physical_gpu_index=1)

    def test_requires_explicit_selection(self):
        with self.assertRaisesRegex(ValueError, "Specify --render-gpu"):
            choose_render_device(MAPPINGS, egl_device_id=None, physical_gpu_index=None)


if __name__ == "__main__":
    unittest.main()
