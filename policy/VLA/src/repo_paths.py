"""Path resolution shared by every entry point in this repository.

Three checkout layouts are supported and must behave identically where they
overlap:

* ``single_repo`` — ``<repo>/policy/VLA/src/*.py`` inside one git repository in
  which ``benchmark/robocasa`` is a plain directory (the anonymised inline
  snapshot the VLA panel was developed on).
* ``submodule`` — ``<team-repo>/policy/VLA`` is a git submodule of the team
  repository and ``benchmark/robocasa`` is another submodule.
* ``standalone`` — this repository cloned on its own. There is no parent
  repository, so the robocasa tree must be named with ``OSMA_BENCH_ROBOCASA_ROOT``.

Nothing here touches the filesystem at import time, so importing a runner
module never fails just because a parent directory does not exist.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

REPO_ROOT_ENV = "OSMA_BENCH_REPO_ROOT"
ROBOCASA_ROOT_ENV = "OSMA_BENCH_ROBOCASA_ROOT"
ROBOSUITE_ROOT_ENV = "OSMA_BENCH_ROBOSUITE_ROOT"
COMPUTE_SSI_ENV = "OSMA_BENCH_COMPUTE_SSI"
OFFICIAL_REF_ROOT_ENV = "OSMA_BENCH_OFFICIAL_REF_ROOT"

# Where this policy lives inside the parent repository and what it needs from it.
VLA_MOUNT_SUBPATH = "policy/VLA"
ROBOCASA_SUBPATH = "benchmark/robocasa"
ROBOSUITE_SUBPATH = "benchmark/robosuite"
COMPUTE_SSI_SUBPATH = "scripts/compute_ssi.py"
OFFICIAL_REF_SUBPATH = "anon/ref"

SOURCE_DERIVED = "derived"
SOURCE_REPO_INLINE = "repo_inline"

# ``src/`` is this file's directory; the repository root is one level up.
DEFAULT_VLA_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RepoPaths:
    """Resolved locations. ``*_source`` records which rule produced each path."""

    vla_root: Path
    superproject_root: Path
    superproject_root_source: str
    robocasa_root: Path
    robocasa_root_source: str
    robosuite_root: Path
    robosuite_root_source: str
    canonical_compute_ssi: Path
    canonical_compute_ssi_source: str
    official_ref_root: Path
    official_ref_root_source: str

    @property
    def inline_robocasa_root(self) -> Path:
        return self.superproject_root / ROBOCASA_SUBPATH

    @property
    def inline_robosuite_root(self) -> Path:
        return self.superproject_root / ROBOSUITE_SUBPATH

    def robosuite_is_present(self) -> bool:
        return (self.robosuite_root / "robosuite" / "__init__.py").is_file()

    def activate_robosuite(self, caller: str) -> Path | None:
        """Put the selected robosuite tree ahead of the installed package. Idempotent.

        robocasa imports robosuite, so this must run before ``activate_robocasa``. The
        upstream bump that introduced robocasa v2 also moved robosuite (a2d0d1ec: the
        mobile base collides by the convex hull of its visual shell); with robocasa on
        v2 and robosuite on the old tree the env has no base collision geom at all and a
        pushed obstacle logs "no contact" (measured 2026-08-31). Unlike robocasa, robosuite
        may legitimately come from site-packages (standalone layout), so a missing inline
        tree is only an error when the tree was named explicitly.
        """
        if self.robosuite_is_present():
            if str(self.robosuite_root) not in sys.path:
                sys.path.insert(0, str(self.robosuite_root))
            return self.robosuite_root
        if self.robosuite_root_source == ROBOSUITE_ROOT_ENV:
            raise SystemExit(
                f"[{caller}] robosuite tree not found: {self.robosuite_root}\n"
                f"  set {ROBOSUITE_ROOT_ENV} to a robosuite checkout (pair it with the robocasa tree)"
            )
        return None

    def robocasa_is_present(self) -> bool:
        return (self.robocasa_root / "robocasa" / "__init__.py").is_file()

    def require_robocasa(self, caller: str) -> Path:
        if not self.robocasa_is_present():
            raise SystemExit(
                f"[{caller}] robocasa tree not found: {self.robocasa_root}\n"
                f"  set {ROBOCASA_ROOT_ENV} to a robocasa checkout, or run from a parent "
                f"repository that provides {ROBOCASA_SUBPATH}"
            )
        return self.robocasa_root

    def activate_robocasa(self, caller: str) -> Path:
        """Make the selected robocasa tree importable. Idempotent."""
        root = self.require_robocasa(caller)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        return root

    def require_compute_ssi(self, caller: str) -> Path:
        if not self.canonical_compute_ssi.is_file():
            raise SystemExit(
                f"[{caller}] canonical script missing: {self.canonical_compute_ssi}\n"
                f"  set {COMPUTE_SSI_ENV} to the parent repository's {COMPUTE_SSI_SUBPATH}"
            )
        return self.canonical_compute_ssi

    def as_manifest(self) -> dict[str, str]:
        return {
            "vla_root": str(self.vla_root),
            "superproject_root": str(self.superproject_root),
            "superproject_root_source": self.superproject_root_source,
            "robocasa_root": str(self.robocasa_root),
            "robocasa_root_source": self.robocasa_root_source,
            "robosuite_root": str(self.robosuite_root),
            "robosuite_root_source": self.robosuite_root_source,
            "canonical_compute_ssi": str(self.canonical_compute_ssi),
            "canonical_compute_ssi_source": self.canonical_compute_ssi_source,
            "official_ref_root": str(self.official_ref_root),
            "official_ref_root_source": self.official_ref_root_source,
        }


def _pick(environ: Mapping[str, str], key: str, fallback: Path, fallback_source: str) -> tuple[Path, str]:
    value = environ.get(key)
    if value:
        return Path(value).expanduser().resolve(), key
    return fallback, fallback_source


def resolve_paths(
    *,
    vla_root: str | Path | None = None,
    superproject_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> RepoPaths:
    """Resolve every path the runners need.

    Precedence for ``superproject_root``: explicit argument → ``OSMA_BENCH_REPO_ROOT``
    → two levels above ``vla_root`` (``<parent>/policy/VLA`` → ``<parent>``), which
    is exactly what the runners assumed before this helper existed.
    """
    env = os.environ if environ is None else environ
    vla = Path(vla_root).expanduser().resolve() if vla_root is not None else DEFAULT_VLA_ROOT

    if superproject_root is not None:
        superproject, superproject_source = Path(superproject_root).expanduser().resolve(), "argument"
    else:
        superproject, superproject_source = _pick(env, REPO_ROOT_ENV, vla.parents[1], SOURCE_DERIVED)

    robocasa, robocasa_source = _pick(env, ROBOCASA_ROOT_ENV, superproject / ROBOCASA_SUBPATH, SOURCE_REPO_INLINE)
    robosuite, robosuite_source = _pick(env, ROBOSUITE_ROOT_ENV, superproject / ROBOSUITE_SUBPATH, SOURCE_REPO_INLINE)
    compute_ssi, compute_ssi_source = _pick(
        env, COMPUTE_SSI_ENV, superproject / COMPUTE_SSI_SUBPATH, SOURCE_REPO_INLINE
    )
    official_ref, official_ref_source = _pick(
        env, OFFICIAL_REF_ROOT_ENV, superproject / OFFICIAL_REF_SUBPATH, SOURCE_REPO_INLINE
    )
    return RepoPaths(
        vla_root=vla,
        superproject_root=superproject,
        superproject_root_source=superproject_source,
        robocasa_root=robocasa,
        robocasa_root_source=robocasa_source,
        robosuite_root=robosuite,
        robosuite_root_source=robosuite_source,
        canonical_compute_ssi=compute_ssi,
        canonical_compute_ssi_source=compute_ssi_source,
        official_ref_root=official_ref,
        official_ref_root_source=official_ref_source,
    )
