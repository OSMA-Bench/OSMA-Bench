from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from reproducibility import (  # noqa: E402
    require_output_root,
    require_reproducible_source,
    runtime_snapshot,
    sha256_file,
    utc_run_stamp,
    write_artifact_manifest,
)
from model_config import load_model_profile, validate_checkpoint_revision  # noqa: E402


class ReproducibilityTest(unittest.TestCase):
    def test_utc_stamp_is_explicit_and_collision_resistant(self):
        instant = datetime(2026, 8, 14, 3, 4, 5, 6789, tzinfo=timezone.utc)
        self.assertEqual(utc_run_stamp(instant), "20260814T030405.006789Z")

    def test_artifact_manifest_records_hash_size_and_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "meta.json").write_text("{}\n", encoding="utf-8")
            output = write_artifact_manifest(root, ("meta.json", "rollout.mp4"))
            manifest = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(manifest["artifacts"]["meta.json"]["sha256"], sha256_file(root / "meta.json"))
            self.assertEqual(manifest["artifacts"]["meta.json"]["size_bytes"], 3)
            self.assertEqual(manifest["missing"], ["rollout.mp4"])

    def test_reproducible_source_gate_is_explicit(self):
        require_reproducible_source(
            {"repository": {"source_reproducible_from_commit": True}}
        )
        with self.assertRaises(RuntimeError):
            require_reproducible_source(
                {
                    "repository": {
                        "source_reproducible_from_commit": False,
                        "relevant_tracked_dirty": True,
                        "relevant_untracked": [],
                    }
                }
            )

    def test_output_root_can_be_relocated_by_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = os.environ.get("OSMA_BENCH_VLA_RUN_ROOT")
            os.environ["OSMA_BENCH_VLA_RUN_ROOT"] = temporary
            try:
                self.assertEqual(require_output_root(temporary), Path(temporary).resolve())
                with self.assertRaises(ValueError):
                    require_output_root(Path(temporary).parent)
            finally:
                if previous is None:
                    os.environ.pop("OSMA_BENCH_VLA_RUN_ROOT", None)
                else:
                    os.environ["OSMA_BENCH_VLA_RUN_ROOT"] = previous

    def test_run_root_env_is_required(self):
        previous = os.environ.pop("OSMA_BENCH_VLA_RUN_ROOT", None)
        try:
            with self.assertRaises(RuntimeError) as raised:
                require_output_root("/anywhere")
            self.assertIn("OSMA_BENCH_VLA_RUN_ROOT is not set", str(raised.exception))
        finally:
            if previous is not None:
                os.environ["OSMA_BENCH_VLA_RUN_ROOT"] = previous

    def test_public_checkpoint_uses_immutable_revision(self):
        profile = load_model_profile("xr1")
        self.assertRegex(profile["checkpoint_revision"], r"^[0-9a-f]{40}$")
        with self.assertRaises(ValueError):
            validate_checkpoint_revision("main")

    def test_offline_checkpoint_mode_is_recorded(self):
        previous_hf = os.environ.get("HF_HUB_OFFLINE")
        previous_transformers = os.environ.get("TRANSFORMERS_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            environment = runtime_snapshot(())["environment"]
            self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
            self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
        finally:
            for key, previous in (
                ("HF_HUB_OFFLINE", previous_hf),
                ("TRANSFORMERS_OFFLINE", previous_transformers),
            ):
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous


if __name__ == "__main__":
    unittest.main()
