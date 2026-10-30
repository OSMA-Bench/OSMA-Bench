"""The --require-reproducible-source gate across the three checkout layouts.

Fixtures are real git repositories built in a temporary directory:

* single_repo — policy/VLA and benchmark/robocasa are plain directories of one repo
* submodule   — both are submodules of a superproject
* standalone  — the VLA repo alone, OSMA_BENCH_ROBOCASA_ROOT naming an external repo

Every layout must give the same verdict for the same working-tree state.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from repo_paths import ROBOCASA_ROOT_ENV, RepoPaths, resolve_paths  # noqa: E402
from reproducibility import (  # noqa: E402
    RUNTIME_UNTRACKED_SUFFIXES,
    build_reproducibility_manifest,
    git_provenance,
    git_toplevel,
    require_reproducible_source,
    source_tree_manifest,
)


GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "fixture",
    "GIT_AUTHOR_EMAIL": "fixture@example.com",
    "GIT_COMMITTER_NAME": "fixture",
    "GIT_COMMITTER_EMAIL": "fixture@example.com",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}
GIT_CONFIG = ["-c", "commit.gpgsign=false", "-c", "protocol.file.allow=always", "-c", "init.defaultBranch=main"]

VLA_FILES = ("src/a.py", "requirements/x.txt")
ROBOCASA_FILES = ("robocasa/__init__.py", "robocasa/utils/ssi.py")
IGNORE = "__pycache__/\n*.pyc\n"


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *GIT_CONFIG, *args], cwd=cwd, env=GIT_ENV, check=True, capture_output=True, text=True
    ).stdout


def write(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def init_repo(root: Path, files: tuple[str, ...] | list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q")
    for relative in files:
        write(root / relative)
    write(root / ".gitignore", IGNORE)
    git(root, "add", "-A")
    git(root, "commit", "-qm", "init")


Fixture = tuple[RepoPaths, Path, Path]  # (paths, vla_dir, robocasa_dir)


def build_single_repo(tmp: Path) -> Fixture:
    root = tmp / "repo"
    init_repo(root, [f"policy/VLA/{f}" for f in VLA_FILES] + [f"benchmark/robocasa/{f}" for f in ROBOCASA_FILES] + ["scripts/compute_ssi.py"])
    return resolve_paths(vla_root=root / "policy/VLA", superproject_root=root, environ={}), root / "policy/VLA", root / "benchmark/robocasa"


def build_superproject_with_submodules(tmp: Path) -> Fixture:
    vla_origin = tmp / "vla_origin"
    init_repo(vla_origin, VLA_FILES)
    robocasa_origin = tmp / "robocasa_origin"
    init_repo(robocasa_origin, ROBOCASA_FILES)
    root = tmp / "team"
    init_repo(root, ["scripts/compute_ssi.py"])
    git(root, "submodule", "add", "-q", str(vla_origin), "policy/VLA")
    git(root, "submodule", "add", "-q", str(robocasa_origin), "benchmark/robocasa")
    git(root, "commit", "-qm", "attach submodules")
    return resolve_paths(vla_root=root / "policy/VLA", superproject_root=root, environ={}), root / "policy/VLA", root / "benchmark/robocasa"


def build_standalone(tmp: Path) -> Fixture:
    vla = tmp / "VLA-nav"
    init_repo(vla, VLA_FILES)
    robocasa = tmp / "robocasa_external"
    init_repo(robocasa, ROBOCASA_FILES)
    return resolve_paths(vla_root=vla, environ={ROBOCASA_ROOT_ENV: str(robocasa)}), vla, robocasa


LAYOUTS: dict[str, Callable[[Path], Fixture]] = {
    "single_repo": build_single_repo,
    "submodule": build_superproject_with_submodules,
    "standalone": build_standalone,
}

# (name, mutate(vla_dir, robocasa_dir), expected source_reproducible_from_commit)
SCENARIOS: list[tuple[str, Callable[[Path, Path], None], bool]] = [
    ("clean", lambda vla, rc: None, True),
    ("vla_untracked_py", lambda vla, rc: write(vla / "src/new.py"), False),
    ("vla_untracked_md", lambda vla, rc: write(vla / "notes.md"), False),
    ("vla_tracked_edit", lambda vla, rc: write(vla / "src/a.py", "changed\n"), False),
    ("vla_ignored_pycache", lambda vla, rc: write(vla / "src/__pycache__/a.cpython-310.pyc"), True),
    ("robocasa_untracked_txt", lambda vla, rc: write(rc / "x.txt"), True),
    ("robocasa_untracked_py", lambda vla, rc: write(rc / "x.py"), False),
    ("robocasa_tracked_edit", lambda vla, rc: write(rc / "robocasa/__init__.py", "changed\n"), False),
]


class ProvenanceLayoutsTest(unittest.TestCase):
    def test_every_layout_gives_the_same_verdict_per_scenario(self):
        for layout, build in LAYOUTS.items():
            for name, mutate, expected in SCENARIOS:
                with self.subTest(layout=layout, scenario=name), tempfile.TemporaryDirectory() as temporary:
                    paths, vla, robocasa = build(Path(temporary))
                    mutate(vla, robocasa)
                    result = git_provenance(paths)
                    self.assertEqual(result["layout"], layout)
                    self.assertEqual(result["source_reproducible_from_commit"], expected, result)
                    self.assertEqual(result["relevant_roots"], ["policy/VLA", "benchmark/robocasa"])
                    manifest = {"repository": result}
                    if expected:
                        require_reproducible_source(manifest)
                    else:
                        with self.assertRaises(RuntimeError):
                            require_reproducible_source(manifest)

    def test_single_repo_untracked_list_matches_the_historical_rule(self):
        """Pre-layout rule: one ls-files over both roots, prefix policy/VLA/ or a runtime suffix."""
        with tempfile.TemporaryDirectory() as temporary:
            paths, vla, robocasa = build_single_repo(Path(temporary))
            write(vla / "notes.md")
            write(vla / "src/new.py")
            write(robocasa / "x.txt")
            write(robocasa / "layout.yaml")
            root = paths.superproject_root
            historical = [
                path
                for path in git(root, "ls-files", "--others", "--exclude-standard", "--", "policy/VLA", "benchmark/robocasa").splitlines()
                if path.startswith("policy/VLA/") or Path(path).suffix in RUNTIME_UNTRACKED_SUFFIXES
            ]
            result = git_provenance(paths)
            self.assertEqual(result["relevant_untracked"], historical)
            self.assertEqual(historical, ["benchmark/robocasa/layout.yaml", "policy/VLA/notes.md", "policy/VLA/src/new.py"])
            self.assertEqual(result["commit"], git(root, "rev-parse", "HEAD").strip())
            self.assertEqual(result["trees"]["vla"]["toplevel"], str(root.resolve()))
            self.assertEqual(result["trees"]["vla"]["relpath"], "policy/VLA")
            self.assertEqual(result["trees"]["robocasa"]["untracked"], ["benchmark/robocasa/layout.yaml"])

    def test_submodule_layout_scans_inside_each_submodule(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, vla, robocasa = build_superproject_with_submodules(Path(temporary))
            root = paths.superproject_root
            clean = git_provenance(paths)
            self.assertEqual(clean["layout"], "submodule")
            self.assertEqual(clean["trees"]["vla"]["toplevel"], str(vla.resolve()))
            self.assertEqual(clean["trees"]["vla"]["relpath"], ".")
            self.assertEqual(clean["trees"]["robocasa"]["toplevel"], str(robocasa.resolve()))
            self.assertEqual(clean["commit"], git(vla, "rev-parse", "HEAD").strip())
            self.assertEqual(clean["superproject"]["toplevel"], str(root.resolve()))
            self.assertEqual(clean["superproject"]["commit"], git(root, "rev-parse", "HEAD").strip())
            self.assertEqual(len(clean["superproject"]["submodule_status"]), 2)
            self.assertNotIn("robocasa_inline", clean["trees"])

            # An untracked file inside the submodule is invisible to the superproject's
            # ls-files — the historical scan would have passed here. It must fail.
            write(vla / "src/zz.py")
            self.assertEqual(git(root, "ls-files", "--others", "--exclude-standard", "--", "policy/VLA"), "")
            self.assertFalse(git_provenance(paths)["source_reproducible_from_commit"])
            (vla / "src/zz.py").unlink()

            # Committing inside the submodule keeps it reproducible and is visible as a
            # moved pointer in the superproject's submodule status.
            write(vla / "src/a.py", "new\n")
            git(vla, "commit", "-qam", "move")
            moved = git_provenance(paths)
            self.assertTrue(moved["source_reproducible_from_commit"])
            self.assertEqual(moved["commit"], git(vla, "rev-parse", "HEAD").strip())
            # `git submodule status` prints "+<sha> policy/VLA (<describe>)" once the pointer moved.
            self.assertTrue(any(line.startswith("+") and " policy/VLA" in line for line in moved["superproject"]["submodule_status"]),
                            moved["superproject"]["submodule_status"])

    def test_standalone_layout_has_no_superproject(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, vla, robocasa = build_standalone(Path(temporary))
            result = git_provenance(paths)
            self.assertEqual(result["layout"], "standalone")
            self.assertIsNone(result["superproject"]["toplevel"])
            self.assertEqual(result["trees"]["robocasa"]["source"], ROBOCASA_ROOT_ENV)
            self.assertNotIn("robocasa_inline", result["trees"])
            self.assertTrue(result["source_reproducible_from_commit"])

    def test_external_robocasa_keeps_the_inline_tree_watched(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            paths, vla, inline = build_single_repo(tmp)
            external = tmp / "robocasa_v2"
            init_repo(external, ROBOCASA_FILES)
            paths = resolve_paths(vla_root=vla, superproject_root=paths.superproject_root, environ={ROBOCASA_ROOT_ENV: str(external)})
            result = git_provenance(paths)
            self.assertEqual(set(result["trees"]), {"vla", "robocasa", "robocasa_inline"})
            self.assertTrue(result["source_reproducible_from_commit"])
            write(external / "layout.yaml")
            self.assertFalse(git_provenance(paths)["source_reproducible_from_commit"])
            (external / "layout.yaml").unlink()
            write(inline / "stray.py")
            self.assertFalse(git_provenance(paths)["source_reproducible_from_commit"])

    def test_a_watched_tree_outside_git_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            paths, vla, _ = build_standalone(tmp)
            plain = tmp / "robocasa_plain"
            for relative in ROBOCASA_FILES:
                write(plain / relative)
            paths = resolve_paths(vla_root=vla, environ={ROBOCASA_ROOT_ENV: str(plain)})
            result = git_provenance(paths)
            self.assertFalse(result["source_reproducible_from_commit"])
            self.assertIsNone(result["trees"]["robocasa"]["commit"])
            with self.assertRaises(RuntimeError) as raised:
                require_reproducible_source({"repository": result})
            self.assertIn("not inside a git repository", str(raised.exception))

    def test_source_tree_manifest_is_layout_independent(self):
        digests = set()
        for layout, build in LAYOUTS.items():
            with tempfile.TemporaryDirectory() as temporary:
                paths, vla, _ = build(Path(temporary))
                sources = source_tree_manifest(paths.vla_root)
                self.assertEqual(sorted(sources["files"]), ["policy/VLA/requirements/x.txt", "policy/VLA/src/a.py"])
                digests.add(sources["tree_sha256"])
        self.assertEqual(len(digests), 1)

    def test_manifest_builder_accepts_explicit_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, vla, _ = build_superproject_with_submodules(Path(temporary))
            manifest = build_reproducibility_manifest(paths.superproject_root, {}, (), paths=paths)
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["repository"]["layout"], "submodule")
            self.assertEqual(manifest["repository"]["provenance_schema_version"], 2)
            self.assertIn("policy/VLA/src/a.py", manifest["sources"]["files"])

    def test_git_toplevel(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            self.assertIsNone(git_toplevel(tmp / "missing"))
            plain = tmp / "plain"
            plain.mkdir()
            paths, vla, robocasa = build_superproject_with_submodules(tmp)
            root = paths.superproject_root
            self.assertEqual(git_toplevel(root / "scripts"), root.resolve())
            self.assertEqual(git_toplevel(vla / "src"), vla.resolve())
            self.assertEqual(git_toplevel(robocasa), robocasa.resolve())
            # A plain directory under the temp dir may or may not be inside some outer
            # repository; the only guarantee is that it is not one of the fixtures.
            self.assertNotIn(git_toplevel(plain), {root.resolve(), vla.resolve(), robocasa.resolve()})


if __name__ == "__main__":
    unittest.main()
