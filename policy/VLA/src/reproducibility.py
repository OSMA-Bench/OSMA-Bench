"""Reproducibility manifests shared by VLA servers and simulator runs.

The provenance gate (``--require-reproducible-source``) refuses to run when the
source that will execute is not reproducible from a recorded commit. It watches
two trees — this repository and the robocasa tree the run will import — and
scans each one **inside its own git toplevel**, so the verdict is the same
whether this code lives inline in the parent repository, as a git submodule of
it, or as a standalone clone (see ``repo_paths``).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from repo_paths import (
    ROBOCASA_ROOT_ENV,
    SOURCE_REPO_INLINE,
    VLA_MOUNT_SUBPATH,
    RepoPaths,
    resolve_paths,
)


MANIFEST_SCHEMA_VERSION = 1
PROVENANCE_SCHEMA_VERSION = 2
RUN_ROOT_ENV = "OSMA_BENCH_VLA_RUN_ROOT"
# Kept verbatim in every manifest (``repository.relevant_roots``) so downstream
# readers of pre-2026-09 manifests keep working. The scan itself is driven by
# ``watched_trees``.
RELEVANT_GIT_ROOTS = ("policy/VLA", "benchmark/robocasa")
RUNTIME_UNTRACKED_SUFFIXES = (".py", ".json", ".xml", ".yaml", ".yml")
SAFE_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "EGL_DEVICE_ID",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "MUJOCO_GL",
    "PYOPENGL_PLATFORM",
    "PYTHONHASHSEED",
    "TRANSFORMERS_OFFLINE",
    RUN_ROOT_ENV,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def utc_run_stamp(now: datetime | None = None) -> str:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def configured_run_root() -> Path:
    value = os.environ.get(RUN_ROOT_ENV)
    if not value:
        raise RuntimeError(
            f"{RUN_ROOT_ENV} is not set; export the site run root (for example "
            f"/data/<user>/vla/runs) and pass --out-root as that directory or a descendant"
        )
    return Path(value).expanduser().resolve()


def require_output_root(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    allowed = configured_run_root()
    if output != allowed and allowed not in output.parents:
        raise ValueError(
            f"--out-root must be {allowed} or a descendant; set {RUN_ROOT_ENV} to change the site root"
        )
    return output


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_file(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


# --------------------------------------------------------------------------- git


def _git(repo_root: Path, *args: str, safe: Iterable[Path] = ()) -> subprocess.CompletedProcess[str]:
    """Run git in ``repo_root``.

    ``safe.directory`` is passed for the root and for every directory in ``safe``
    (a submodule's superproject, for instance) because the simulator container
    runs as root over a tree owned by the host user.
    """
    config: list[str] = []
    for directory in (repo_root, *safe):
        config += ["-c", f"safe.directory={directory}"]
    return subprocess.run(
        ["git", *config, *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )


def _dot_git_chain(path: Path) -> list[Path]:
    """Directories from ``path`` upwards that carry a ``.git`` (dir or gitlink file)."""
    return [candidate for candidate in (path, *path.parents) if (candidate / ".git").exists()]


def git_toplevel(path: str | Path) -> Path | None:
    """Toplevel of the git work tree containing ``path``; ``None`` outside any repo.

    For a submodule this is the submodule's own toplevel, not the superproject's.
    """
    target = Path(path).resolve()
    if not target.exists():
        return None
    chain = _dot_git_chain(target)
    if not chain:
        return None
    result = _git(chain[0], "rev-parse", "--show-toplevel", safe=chain[1:])
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return Path(result.stdout.strip()).resolve()


@dataclass(frozen=True)
class WatchedTree:
    name: str
    path: Path
    source: str
    # ``None`` means every untracked file counts, whatever its suffix.
    untracked_suffixes: tuple[str, ...] | None


def watched_trees(paths: RepoPaths) -> list[WatchedTree]:
    trees = [
        WatchedTree("vla", paths.vla_root, "vla_root", None),
        WatchedTree("robocasa", paths.robocasa_root, paths.robocasa_root_source, RUNTIME_UNTRACKED_SUFFIXES),
    ]
    inline = paths.inline_robocasa_root
    if (
        paths.robocasa_root_source != SOURCE_REPO_INLINE
        and inline.exists()
        and inline.resolve() != paths.robocasa_root.resolve()
    ):
        # The inline tree is still watched so an external override never narrows
        # the scan below what the single-repo layout has always covered.
        trees.append(WatchedTree("robocasa_inline", inline, SOURCE_REPO_INLINE, RUNTIME_UNTRACKED_SUFFIXES))
    # robosuite is watched the same way when a tree is present (it may also come from
    # site-packages in a standalone clone, in which case there is nothing to scan).
    if paths.robosuite_is_present():
        trees.append(WatchedTree("robosuite", paths.robosuite_root, paths.robosuite_root_source, RUNTIME_UNTRACKED_SUFFIXES))
        inline_rs = paths.inline_robosuite_root
        if (
            paths.robosuite_root_source != SOURCE_REPO_INLINE
            and inline_rs.exists()
            and inline_rs.resolve() != paths.robosuite_root.resolve()
        ):
            trees.append(WatchedTree("robosuite_inline", inline_rs, SOURCE_REPO_INLINE, RUNTIME_UNTRACKED_SUFFIXES))
    return trees


def _belongs(entry: str, relpath: str) -> bool:
    if relpath == ".":
        return True
    return entry == relpath or entry.startswith(relpath + "/")


def _accepted(entry: str, suffixes: tuple[str, ...] | None) -> bool:
    return suffixes is None or Path(entry).suffix in suffixes


def _relpath(path: Path, toplevel: Path) -> str:
    resolved = path.resolve()
    if resolved == toplevel:
        return "."
    return resolved.relative_to(toplevel).as_posix()


def _detect_layout(paths: RepoPaths, vla_top: Path | None) -> tuple[str, Path | None]:
    super_top = git_toplevel(paths.superproject_root) if paths.superproject_root.exists() else None
    mounted = (paths.superproject_root / VLA_MOUNT_SUBPATH).resolve() == paths.vla_root.resolve()
    if vla_top is not None and super_top is not None and mounted:
        if vla_top == super_top:
            return "single_repo", super_top
        if vla_top == paths.vla_root.resolve():
            return "submodule", super_top
    return "standalone", None


def git_provenance(paths: RepoPaths | str | Path) -> dict[str, Any]:
    """Scan every watched tree inside its own git toplevel.

    The eight historical keys (``commit`` … ``source_reproducible_from_commit``)
    keep their exact definition; ``layout``, ``superproject`` and ``trees`` are
    additive.
    """
    if not isinstance(paths, RepoPaths):
        root = Path(paths).resolve()
        mounted = root / VLA_MOUNT_SUBPATH
        paths = resolve_paths(
            vla_root=mounted if (mounted / "src").is_dir() else None,
            superproject_root=root,
        )

    trees = watched_trees(paths)
    info: dict[str, dict[str, Any]] = {}
    groups: dict[Path, list[WatchedTree]] = {}
    for tree in trees:
        present = tree.path.exists()
        toplevel = git_toplevel(tree.path) if present else None
        info[tree.name] = {
            "path": str(tree.path),
            "source": tree.source,
            "present": present,
            "toplevel": str(toplevel) if toplevel else None,
            "relpath": _relpath(tree.path, toplevel) if toplevel else None,
            "commit": None,
            "branch": None,
            "dirty": False,
            "dirty_files": [],
            "untracked": [],
            "untracked_suffixes": list(tree.untracked_suffixes) if tree.untracked_suffixes else None,
        }
        if toplevel is not None:
            groups.setdefault(toplevel, []).append(tree)

    diff_blob = b""
    relevant_untracked: list[str] = []
    for toplevel, members in groups.items():
        safe = _dot_git_chain(toplevel)[1:]
        relpaths = [info[tree.name]["relpath"] for tree in members]
        commit_result = _git(toplevel, "rev-parse", "HEAD", safe=safe)
        commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
        branch_result = _git(toplevel, "rev-parse", "--abbrev-ref", "HEAD", safe=safe)
        branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None

        diff_result = _git(toplevel, "diff", "--binary", "HEAD", "--", *relpaths, safe=safe)
        diff_blob += diff_result.stdout.encode("utf-8") if diff_result.returncode == 0 else b""
        others_result = _git(toplevel, "ls-files", "--others", "--exclude-standard", "--", *relpaths, safe=safe)
        others = others_result.stdout.splitlines() if others_result.returncode == 0 else []

        for tree in members:
            entry = info[tree.name]
            entry["commit"] = commit
            entry["branch"] = branch
            names_result = _git(toplevel, "diff", "--name-only", "HEAD", "--", entry["relpath"], safe=safe)
            entry["dirty_files"] = names_result.stdout.splitlines() if names_result.returncode == 0 else []
            entry["dirty"] = bool(entry["dirty_files"])
            entry["untracked"] = [
                path
                for path in others
                if _belongs(path, entry["relpath"]) and _accepted(path, tree.untracked_suffixes)
            ]
        # Keep git's own ordering across the trees that share this toplevel.
        for path in others:
            for tree in members:
                if _belongs(path, info[tree.name]["relpath"]) and _accepted(path, tree.untracked_suffixes):
                    relevant_untracked.append(path)
                    break

    vla = info["vla"]
    vla_top = Path(vla["toplevel"]) if vla["toplevel"] else None
    status: list[str] = []
    if vla_top is not None:
        status_result = _git(vla_top, "status", "--short", "--untracked-files=all", safe=_dot_git_chain(vla_top)[1:])
        status = status_result.stdout.splitlines() if status_result.returncode == 0 else []

    layout, super_top = _detect_layout(paths, vla_top)
    superproject: dict[str, Any] = {
        "root": str(paths.superproject_root),
        "root_source": paths.superproject_root_source,
        "toplevel": str(super_top) if super_top else None,
        "commit": None,
        "branch": None,
        "submodule_status": [],
    }
    if super_top is not None:
        safe = _dot_git_chain(super_top)[1:]
        commit_result = _git(super_top, "rev-parse", "HEAD", safe=safe)
        superproject["commit"] = commit_result.stdout.strip() if commit_result.returncode == 0 else None
        branch_result = _git(super_top, "rev-parse", "--abbrev-ref", "HEAD", safe=safe)
        superproject["branch"] = branch_result.stdout.strip() if branch_result.returncode == 0 else None
        if layout == "submodule":
            status_result = _git(super_top, "submodule", "status", safe=safe)
            superproject["submodule_status"] = (
                status_result.stdout.splitlines() if status_result.returncode == 0 else []
            )

    present_trees = [entry for entry in info.values() if entry["present"]]
    all_committed = bool(present_trees) and all(entry["commit"] for entry in present_trees)
    relevant_tracked_dirty = bool(diff_blob)
    source_reproducible = all_committed and not relevant_tracked_dirty and not relevant_untracked
    return {
        "commit": vla["commit"],
        "branch": vla["branch"],
        "status_short": status,
        "relevant_roots": list(RELEVANT_GIT_ROOTS),
        "relevant_tracked_dirty": relevant_tracked_dirty,
        "relevant_diff_sha256": hashlib.sha256(diff_blob).hexdigest(),
        "relevant_untracked": relevant_untracked,
        "source_reproducible_from_commit": source_reproducible,
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "layout": layout,
        "superproject": superproject,
        "trees": info,
    }


def source_tree_manifest(vla_root: str | Path) -> dict[str, Any]:
    """Hash ``src/`` and ``requirements/`` of this repository.

    Keys are prefixed with ``policy/VLA/`` regardless of where the checkout
    lives, so the same sources hash to the same ``tree_sha256`` in every layout.
    """
    root = Path(vla_root).resolve()
    source_roots = (root / "src", root / "requirements")
    entries: dict[str, dict[str, Any]] = {}
    tree_digest = hashlib.sha256()
    for source_root in source_roots:
        if not source_root.exists():
            continue
        for path in sorted(item for item in source_root.rglob("*") if item.is_file()):
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            relative = f"{VLA_MOUNT_SUBPATH}/{path.relative_to(root).as_posix()}"
            digest = sha256_file(path)
            size = path.stat().st_size
            entries[relative] = {"sha256": digest, "size_bytes": size}
            tree_digest.update(relative.encode("utf-8") + b"\0")
            tree_digest.update(digest.encode("ascii") + b"\0")
            tree_digest.update(str(size).encode("ascii") + b"\n")
    return {"tree_sha256": tree_digest.hexdigest(), "files": entries}


def runtime_snapshot(package_names: Iterable[str]) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(package_names),
        "environment": {key: os.environ[key] for key in SAFE_ENVIRONMENT_KEYS if key in os.environ},
    }


def build_reproducibility_manifest(
    repo_root: str | Path,
    resolved_args: dict[str, Any],
    package_names: Iterable[str],
    *,
    paths: RepoPaths | None = None,
) -> dict[str, Any]:
    """``repo_root`` is the parent-repository root; ``paths`` overrides every location."""
    if paths is None:
        paths = resolve_paths(superproject_root=Path(repo_root))
    repository = git_provenance(paths)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": utc_now_iso(),
        "command": {
            "argv": [sys.executable, *sys.argv],
            "shell_escaped": shlex.join([sys.executable, *sys.argv]),
            "cwd": str(Path.cwd()),
        },
        "resolved_args": resolved_args,
        "repository": repository,
        "sources": source_tree_manifest(paths.vla_root),
        "runtime": runtime_snapshot(package_names),
    }


def require_reproducible_source(manifest: dict[str, Any]) -> None:
    repository = manifest["repository"]
    if repository["source_reproducible_from_commit"]:
        return
    message = (
        "Relevant benchmark source is not reproducible from the recorded commit: "
        f"tracked_dirty={repository['relevant_tracked_dirty']}, "
        f"untracked={repository['relevant_untracked']}"
    )
    trees = repository.get("trees")
    if trees:
        per_tree = {
            name: {"dirty": entry.get("dirty"), "untracked": entry.get("untracked"), "commit": entry.get("commit")}
            for name, entry in trees.items()
            if entry.get("present")
        }
        message += f", per_tree={per_tree}"
        if any(entry.get("present") and not entry.get("commit") for entry in trees.values()):
            message += f" (a watched tree is not inside a git repository; check {ROBOCASA_ROOT_ENV})"
    raise RuntimeError(message)


def write_artifact_manifest(run_dir: str | Path, filenames: Iterable[str]) -> Path:
    directory = Path(run_dir)
    artifacts: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for filename in filenames:
        path = directory / filename
        if path.is_file():
            artifacts[filename] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        else:
            missing.append(filename)
    output = directory / "artifacts.json"
    write_json_file(
        output,
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "generated_at_utc": utc_now_iso(),
            "artifacts": artifacts,
            "missing": missing,
        },
    )
    return output
