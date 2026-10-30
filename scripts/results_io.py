"""I/O helpers for results.json + per-task trajectory log.

`load_run(path)` is a drop-in replacement for `json.load(open(...))`. It
inflates each task's trajectory log (velocity / jerk / clearance / per-step
robot_pos / robot_yaw) inline so analysis scripts keep working with one
flat dict per evaluation.

`load_trajectory(result_entry, run_dir)` reads the trajectory log for a
single task — use this when you only need a few episodes from a large run.

The trajectory log lives at `{run_dir}/{task_dir}/trajectory_log.json`.
"""
from __future__ import annotations

import json
import os
from typing import Any


_LEGACY_KEYS = (
    "timeseries_velocity",
    "timeseries_jerk",
    "timeseries_min_obstacle_distance",
    "timeseries_obstacle_distances",
    "robot_pos",
    "robot_yaw",
)
_KEY_MAP = {
    "velocity":              "timeseries_velocity",
    "jerk":                  "timeseries_jerk",
    "min_obstacle_distance": "timeseries_min_obstacle_distance",
    "obstacle_distances":    "timeseries_obstacle_distances",
    "robot_pos":             "robot_pos",
    "robot_yaw":             "robot_yaw",
}


def load_trajectory(result_entry: dict, run_dir: str) -> dict:
    """Return per-task trajectory log keyed by canonical names.

    Resolves `trajectory_log.json` from `task_info["task_dir"]`. Returns
    empty values when the file is missing.
    """
    out = {k: None for k in _LEGACY_KEYS}
    task_dir = (result_entry.get("task_info") or {}).get("task_dir")
    if not task_dir:
        return out
    path = os.path.join(run_dir, task_dir, "trajectory_log.json")
    try:
        with open(path) as f:
            ts = json.load(f)
    except FileNotFoundError:
        return out
    for short, target in _KEY_MAP.items():
        if short in ts:
            out[target] = ts[short]
    return out


def load_run(path: str) -> dict:
    """Load results.json and inflate trajectory logs inline."""
    with open(path) as f:
        data = json.load(f)
    run_dir = os.path.dirname(os.path.abspath(path))
    for r in data.get("results", []):
        ev = r.get("evaluation") or {}
        if "failure_message" in ev or "error" in ev:
            continue
        ts = load_trajectory(r, run_dir)
        for k, v in ts.items():
            if v is not None and ev.get(k) is None:
                ev[k] = v
    return data


__all__ = ["load_run", "load_trajectory"]
