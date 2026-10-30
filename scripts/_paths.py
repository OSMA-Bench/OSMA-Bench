"""Portable workspace-root resolution for scripts/.

Resolution order:
  1. Explicit override (e.g. an argparse --workspace-dir value) passed to repo_root().
  2. $ROBOTICS_SAFETY_ROOT environment variable.
  3. __file__ ancestry — assumes this file lives at <repo>/scripts/_paths.py.
"""
import os
from pathlib import Path


def repo_root(override=None) -> Path:
    if override:
        return Path(override).resolve()
    env = os.environ.get("ROBOTICS_SAFETY_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent
