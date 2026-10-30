"""repo_paths: layout-independent path resolution and import safety of the runners."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from repo_paths import (  # noqa: E402
    COMPUTE_SSI_ENV,
    OFFICIAL_REF_ROOT_ENV,
    REPO_ROOT_ENV,
    ROBOCASA_ROOT_ENV,
    resolve_paths,
)


class RepoPathsTest(unittest.TestCase):
    def test_derived_superproject_is_two_levels_above_vla_root(self):
        paths = resolve_paths(vla_root="/x/policy/VLA", environ={})
        self.assertEqual(paths.superproject_root, Path("/x"))
        self.assertEqual(paths.superproject_root_source, "derived")
        # Same value the runners used to compute as Path(<vla>/src/x.py).parents[3].
        self.assertEqual(paths.superproject_root, Path("/x/policy/VLA/src/x.py").parents[3])

    def test_defaults_hang_off_the_superproject(self):
        paths = resolve_paths(vla_root="/x/policy/VLA", environ={})
        self.assertEqual(paths.robocasa_root, Path("/x/benchmark/robocasa"))
        self.assertEqual(paths.robocasa_root_source, "repo_inline")
        self.assertEqual(paths.canonical_compute_ssi, Path("/x/scripts/compute_ssi.py"))
        self.assertEqual(paths.official_ref_root, Path("/x/anon/ref"))
        self.assertEqual(paths.inline_robocasa_root, Path("/x/benchmark/robocasa"))

    def test_environment_overrides_every_location(self):
        environ = {
            REPO_ROOT_ENV: "/team",
            ROBOCASA_ROOT_ENV: "/ext/robocasa_v2",
            COMPUTE_SSI_ENV: "/team/scripts/compute_ssi.py",
            OFFICIAL_REF_ROOT_ENV: "/refs",
        }
        paths = resolve_paths(vla_root="/x/policy/VLA", environ=environ)
        self.assertEqual(paths.superproject_root, Path("/team"))
        self.assertEqual(paths.superproject_root_source, REPO_ROOT_ENV)
        self.assertEqual(paths.robocasa_root, Path("/ext/robocasa_v2"))
        self.assertEqual(paths.robocasa_root_source, ROBOCASA_ROOT_ENV)
        self.assertEqual(paths.canonical_compute_ssi_source, COMPUTE_SSI_ENV)
        self.assertEqual(paths.official_ref_root, Path("/refs"))
        self.assertEqual(set(paths.as_manifest()), {
            "vla_root", "superproject_root", "superproject_root_source", "robocasa_root",
            "robocasa_root_source", "robosuite_root", "robosuite_root_source",
            "canonical_compute_ssi", "canonical_compute_ssi_source",
            "official_ref_root", "official_ref_root_source",
        })

    def test_explicit_superproject_argument_wins_over_environment(self):
        paths = resolve_paths(vla_root="/x/policy/VLA", superproject_root="/given", environ={REPO_ROOT_ENV: "/env"})
        self.assertEqual(paths.superproject_root, Path("/given"))
        self.assertEqual(paths.superproject_root_source, "argument")

    def test_resolution_is_safe_when_nothing_exists(self):
        paths = resolve_paths(vla_root="/does/not/exist/policy/VLA", environ={})
        self.assertFalse(paths.robocasa_is_present())
        with self.assertRaises(SystemExit) as raised:
            paths.require_robocasa("unit_test")
        self.assertIn("[unit_test] robocasa tree not found: /does/not/exist/benchmark/robocasa", str(raised.exception))
        with self.assertRaises(SystemExit) as raised:
            paths.require_compute_ssi("unit_test")
        self.assertIn("canonical script missing", str(raised.exception))

    def test_activate_robocasa_puts_the_tree_first_on_sys_path_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            tree = Path(temporary) / "robocasa_tree"
            (tree / "robocasa").mkdir(parents=True)
            (tree / "robocasa" / "__init__.py").write_text("", encoding="utf-8")
            paths = resolve_paths(vla_root="/x/policy/VLA", environ={ROBOCASA_ROOT_ENV: str(tree)})
            before = list(sys.path)
            try:
                self.assertEqual(paths.activate_robocasa("unit_test"), tree.resolve())
                paths.activate_robocasa("unit_test")
                self.assertEqual(sys.path[0], str(tree.resolve()))
                self.assertEqual(sys.path.count(str(tree.resolve())), 1)
            finally:
                sys.path[:] = before


class RunnerImportTest(unittest.TestCase):
    """The runners must import and answer --help without a simulator tree."""

    def _env(self, robocasa_root: Path, run_root: Path) -> dict[str, str]:
        return {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            ROBOCASA_ROOT_ENV: str(robocasa_root),
            "OSMA_BENCH_VLA_RUN_ROOT": str(run_root),
        }

    def test_modules_import_and_help_without_a_robocasa_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            empty = Path(temporary) / "empty"
            empty.mkdir()
            env = self._env(empty, Path(temporary))
            imports = subprocess.run(
                [sys.executable, "-c", "import run_vla_rollout, base_diagnostics, run_vla_eval, compute_ssi_v2, vla_server"],
                cwd=SRC, env=env, capture_output=True, text=True,
            )
            self.assertEqual(imports.returncode, 0, imports.stderr)
            for argv in (
                [sys.executable, str(SRC / "run_vla_rollout.py"), "--help"],
                [sys.executable, str(SRC / "run_vla_eval.py"), "--help"],
                [sys.executable, str(SRC / "base_diagnostics.py"), "--help"],
                [sys.executable, str(SRC / "vla_server.py"), "--model-profile", "xr1", "--print-profile"],
            ):
                result = subprocess.run(argv, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, f"{argv[1:]}: {result.stderr}")

    def test_running_without_a_robocasa_tree_fails_with_the_historical_message(self):
        with tempfile.TemporaryDirectory() as temporary:
            empty = Path(temporary) / "empty"
            empty.mkdir()
            result = subprocess.run(
                [sys.executable, str(SRC / "run_vla_rollout.py"), "--out-root", temporary],
                env=self._env(empty, Path(temporary)), capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("[run_vla_rollout] robocasa tree not found:", result.stderr)


if __name__ == "__main__":
    unittest.main()
