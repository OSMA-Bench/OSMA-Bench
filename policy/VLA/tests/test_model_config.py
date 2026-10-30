from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from model_config import DEFAULT_CONFIG_PATH, load_model_profile  # noqa: E402


class ModelConfigTest(unittest.TestCase):
    def test_schema3_profiles_keep_xr1_and_add_pi05(self):
        xr1 = load_model_profile("xr1")
        pi05 = load_model_profile("pi05_opentau")
        self.assertEqual(xr1["profile_schema_version"], 3)
        self.assertEqual(xr1["action_chunk_size"], 16)
        self.assertEqual(xr1["camera_slots"], ["left", "right", "wrist"])
        self.assertEqual(pi05["camera_slots"], ["wrist", "left", "right"])
        self.assertEqual(pi05["history_length"], 1)
        self.assertEqual(pi05["action_chunk_size"], 20)
        self.assertEqual(pi05["replan_steps"], 20)

    def test_replan_cannot_exceed_native_chunk(self):
        document = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        document["profiles"]["xr1"]["replan_steps"] = 17
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.yaml"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot exceed"):
                load_model_profile("xr1", path)


if __name__ == "__main__":
    unittest.main()
