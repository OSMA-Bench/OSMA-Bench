"""Per-episode raw dump (`vla_dump.npz`) so a rollout can be re-scored later
without re-running the simulator.

Why this exists (2026-08-31)
----------------------------
The benchmark's scoring definitions (distance thresholds, contact rule, path
metrics) can change after a sweep has burned days of GPU time. What we keep
today is not enough to re-score:

* `trajectory_log.json` holds the metric *inputs* sampled every
  ``TRAJECTORY_LOG_INTERVAL`` control steps — no obstacle positions, no
  per-step contact flags.
* VoxPoser's `voxposer_dump.npz` (`policy/Voxposer/src/modules/interfaces.py`)
  keeps planning-time obstacle xy, the planned path and the planner maps.
  The VLA runner had no npz at all.

This module records, at **every control step**:

* robot base pose, eef offset, arm/torso joints, applied and raw action
* every obstacle's world pose (collision-geom centroid + root body pos/quat)
* surface distance, contact flag as the env defines it, peak contact force
  (robocasa v2 only), boundary flags, env success

Team-parity keys (`obstacle_name`, `obstacle_xy_init`, `goal_xy_fixture`) keep
the same names as `voxposer_dump.npz`. The file is deliberately **not** called
`voxposer_dump.npz`: `scripts/visualize_voxposer_task.py` opens that name and
requires the planner `iters` record, which a VLA rollout does not have.

Contact semantics follow the tree that ran the episode and are recorded in
``contact_definition``: robocasa v2 = solver contact force OR-ed over physics
substeps (``CONTACT_FORCE_THRESHOLD_N``); the inlined pre-v2 tree = signed
geom distance <= 0 at the control step.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

DUMP_FILENAME = "vla_dump.npz"
# 2 (2026-09-06): + per-step ``ori_cos`` / ``task_success`` / ``collision_free_success`` and
# the scalar ``success_semantics``; ``distance_cap_m`` falls back to the metric-cleanup
# constant DISTANCE_MEASURE_MAX_M when the tree no longer exposes a boundary threshold.
SCHEMA_VERSION = 2


def _distance_measure_max() -> float:
    """Tier-independent mj_geomDistance cap of the metric-cleanup trees (NaN elsewhere)."""
    try:
        from robocasa.metrics._config import DISTANCE_MEASURE_MAX_M  # type: ignore
        return float(DISTANCE_MEASURE_MAX_M)
    except Exception:
        return float("nan")


def _model_array(model: Any, name: str):
    """robosuite's MjModel wraps mujoco.MjModel as ``_model``; try both."""
    for owner in (model, getattr(model, "_model", None)):
        if owner is not None and hasattr(owner, name):
            return np.asarray(getattr(owner, name))
    raise AttributeError(name)


def _data_array(data: Any, name: str):
    for owner in (data, getattr(data, "_data", None)):
        if owner is not None and hasattr(owner, name):
            return np.asarray(getattr(owner, name))
    raise AttributeError(name)


def resolve_obstacle_groups(env: Any) -> tuple[dict[str, dict[str, list[int]]], str]:
    """Return ({obstacle name: {"geom_ids": [...], "body_ids": [...]}}, source).

    Primary source is the env's own grouping (`_obstacle_geom_groups`, robocasa
    v2) so the dump names match `info["obstacle_distances"]` /
    `info["obstacle_contacts"]` exactly. Fallback scans body names that start
    with ``obstacle`` or ``posed_human`` (pre-v2 tree, or tests).
    """
    model = env.sim.model
    geom_bodyid = _model_array(model, "geom_bodyid")
    groups: dict[str, dict[str, list[int]]] = {}

    fn = getattr(env, "_obstacle_geom_groups", None)
    if callable(fn):
        try:
            raw, _robot = fn()
        except Exception:  # pragma: no cover - defensive; fall through to scan
            raw = None
        if raw:
            for name, geom_ids in raw.items():
                gids = sorted(int(g) for g in geom_ids)
                bids = sorted({int(geom_bodyid[g]) for g in gids})
                groups[str(name)] = {"geom_ids": gids, "body_ids": bids}
            return groups, "env._obstacle_geom_groups"

    nbody = int(_model_array(model, "nbody")) if hasattr(model, "nbody") or hasattr(
        getattr(model, "_model", None), "nbody") else len(_data_array(env.sim.data, "body_xpos"))
    for bid in range(nbody):
        bname = model.body_id2name(bid) or ""
        if bname.startswith("obstacle") or "posed_human" in bname:
            key = "human" if "posed_human" in bname else bname.split("_main")[0]
            entry = groups.setdefault(key, {"geom_ids": [], "body_ids": []})
            entry["body_ids"].append(bid)
            entry["geom_ids"].extend(int(g) for g in np.flatnonzero(geom_bodyid == bid))
    for entry in groups.values():
        entry["geom_ids"] = sorted(set(entry["geom_ids"]))
        entry["body_ids"] = sorted(set(entry["body_ids"]))
    return groups, "body_name_scan"


class EpisodeDumpRecorder:
    """Accumulate per-step raw state and write `vla_dump.npz`."""

    def __init__(self, env: Any, goal_xy: Any, control_freq: float | None = None) -> None:
        self.env = env
        self.goal_xy = np.asarray(goal_xy, dtype=np.float64).reshape(2)
        self.control_freq = float(control_freq or getattr(env, "control_freq", 20.0))
        self.groups, self.obstacle_pos_source = resolve_obstacle_groups(env)
        self.names = list(self.groups)
        self.obstacle_name = str(getattr(env, "obstacle", "") or "")
        self.obstacle_pos_init = self._obstacle_positions()[0]
        self.rows: list[dict[str, Any]] = []
        self.success_semantics: str | None = None

    # -- sampling ---------------------------------------------------------
    def _obstacle_positions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = self.env.sim.data
        k = len(self.names)
        centroid = np.full((k, 3), np.nan, dtype=np.float64)
        body_pos = np.full((k, 3), np.nan, dtype=np.float64)
        body_quat = np.full((k, 4), np.nan, dtype=np.float64)
        if k == 0:
            return centroid, body_pos, body_quat
        geom_xpos = _data_array(data, "geom_xpos")
        body_xpos = _data_array(data, "body_xpos")
        body_xquat = _data_array(data, "body_xquat")
        for i, name in enumerate(self.names):
            g = self.groups[name]["geom_ids"]
            b = self.groups[name]["body_ids"]
            if g:
                centroid[i] = geom_xpos[g].mean(axis=0)
            if b:
                body_pos[i] = body_xpos[b[0]]
                body_quat[i] = body_xquat[b[0]]
        return centroid, body_pos, body_quat

    def record(
        self,
        step: int,
        info: dict[str, Any],
        snapshot: dict[str, Any],
        applied_action: Any,
        raw_action: Any,
        chunk_id: int,
        task: dict[str, Any] | None = None,
    ) -> None:
        intrusion = getattr(self.env, "intrusion", None) or {}
        dists = info.get("obstacle_distances") or intrusion.get("obstacle_distances") or {}
        contacts = info.get("obstacle_contacts") or intrusion.get("obstacle_contacts") or {}
        forces = intrusion.get("obstacle_contact_forces") or {}
        centroid, body_pos, body_quat = self._obstacle_positions()
        task = task or {}
        self.rows.append({
            "step": int(step),
            "base_pos": np.asarray(snapshot["base_pos"], dtype=np.float64).reshape(3),
            "base_yaw": float(snapshot["base_yaw"]),
            "eef_relative_pos": np.asarray(snapshot.get("eef_relative_pos", [np.nan] * 3), dtype=np.float64).reshape(3),
            "arm_qpos": np.asarray(snapshot.get("arm_qpos", []), dtype=np.float64).reshape(-1),
            "torso_qpos": np.asarray(snapshot.get("torso_qpos", []), dtype=np.float64).reshape(-1),
            "goal_distance": float(snapshot.get("goal_distance", np.nan)),
            "applied_action": np.asarray(applied_action, dtype=np.float64).reshape(-1),
            "raw_action": np.asarray(raw_action, dtype=np.float64).reshape(-1),
            "chunk_id": int(chunk_id),
            "centroid": centroid,
            "body_pos": body_pos,
            "body_quat": body_quat,
            "distance": np.asarray([float(dists.get(n, np.nan)) for n in self.names], dtype=np.float64),
            "contact": np.asarray([bool(contacts.get(n, False)) for n in self.names], dtype=bool),
            "force": np.asarray([float(forces.get(n, np.nan)) for n in self.names], dtype=np.float64),
            "min_distance": float(info.get("min_obstacle_distance", intrusion.get("min_obstacle_distance", np.nan))),
            "boundary_violated": bool(info.get("boundary_violated", intrusion.get("boundary_violated", False))),
            "boundary_violated_ever": bool(intrusion.get("boundary_violated_ever", info.get("boundary_violated_ever", False))),
            "contact_ever": bool(info.get("obstacle_contact_ever", False)),
            "contact_count": int(info.get("obstacle_contact_count", 0) or 0),
            "env_success": bool(task.get("env_success", False)),
            # schema 2 (2026-09-06): per-step task terms so an arrival rule (radius / stop
            # point) can be re-scored offline for any radius ≥ the one that stopped the run.
            "ori_cos": float(task["orientation_cos"]) if task.get("orientation_cos") is not None else np.nan,
            "task_success": bool(task.get("task_success", False)),
            "collision_free_success": (-1 if task.get("collision_free_success") is None
                                       else int(bool(task["collision_free_success"]))),
        })
        if task.get("success_semantics") and not self.success_semantics:
            self.success_semantics = str(task["success_semantics"])

    # -- output -----------------------------------------------------------
    def arrays(self, meta: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
        rows = self.rows
        t = len(rows)
        k = len(self.names)
        intrusion = getattr(self.env, "intrusion", None) or {}
        boundary = intrusion.get("boundary_threshold", getattr(self.env, "SAFETY_BOUNDARY_DEFAULT_M", np.nan))
        boundary_f = float(boundary) if boundary is not None else np.nan
        if np.isfinite(boundary_f):
            distance_cap = boundary_f + 1.0          # pre-2026-09 trees: mj_geomDistance distmax = r_b + 1.0
        else:
            distance_cap = _distance_measure_max()   # metric-cleanup trees: one tier-independent constant
        if callable(getattr(self.env, "_obstacle_geom_groups", None)):
            contact_definition = "robocasa_v2: solver contact force OR-ed over physics substeps (CONTACT_FORCE_THRESHOLD_N)"
        else:
            contact_definition = "pre-v2: signed geom distance <= 0 at the control step"

        def stack(key, shape_tail=()):
            if t == 0:
                return np.zeros((0, *shape_tail), dtype=np.float32)
            return np.asarray([r[key] for r in rows])

        out: dict[str, np.ndarray] = {
            "schema_version": np.asarray(SCHEMA_VERSION),
            "meta_json": np.asarray(json.dumps(meta or {}, ensure_ascii=False, default=str)),
            "control_freq_hz": np.asarray(self.control_freq),
            "contact_definition": np.asarray(contact_definition),
            "obstacle_pos_source": np.asarray(self.obstacle_pos_source),
            "boundary_threshold_m": np.asarray(boundary_f),
            "distance_cap_m": np.asarray(distance_cap),
            "success_semantics": np.asarray(str(self.success_semantics or "")),
            "step": stack("step").astype(np.int32),
            "time_s": (stack("step").astype(np.float64) / self.control_freq).astype(np.float32),
            "robot_base_pos": stack("base_pos", (3,)).astype(np.float32),
            "robot_base_yaw": stack("base_yaw").astype(np.float32),
            "robot_eef_relative_pos": stack("eef_relative_pos", (3,)).astype(np.float32),
            "robot_arm_qpos": stack("arm_qpos", (0,)).astype(np.float32),
            "robot_torso_qpos": stack("torso_qpos", (0,)).astype(np.float32),
            "goal_distance_m": stack("goal_distance").astype(np.float32),
            "applied_action": stack("applied_action", (0,)).astype(np.float32),
            "raw_action": stack("raw_action", (0,)).astype(np.float32),
            "chunk_id": stack("chunk_id").astype(np.int32),
            "env_success": stack("env_success").astype(bool),
            "ori_cos": stack("ori_cos").astype(np.float32),
            "task_success": stack("task_success").astype(bool),
            "collision_free_success": stack("collision_free_success").astype(np.int8),  # -1 = tree exposes none
            "obstacle_names": np.asarray(self.names, dtype=str),
            "obstacle_pos": stack("centroid", (k, 3)).astype(np.float32),
            "obstacle_body_pos": stack("body_pos", (k, 3)).astype(np.float32),
            "obstacle_body_quat_wxyz": stack("body_quat", (k, 4)).astype(np.float32),
            "obstacle_distance_m": stack("distance", (k,)).astype(np.float32),
            "obstacle_contact": stack("contact", (k,)).astype(bool),
            "obstacle_contact_force_peak_n": stack("force", (k,)).astype(np.float32),
            "min_obstacle_distance_m": stack("min_distance").astype(np.float32),
            "boundary_violated": stack("boundary_violated").astype(bool),
            "boundary_violated_ever": stack("boundary_violated_ever").astype(bool),
            "obstacle_contact_ever": stack("contact_ever").astype(bool),
            "obstacle_contact_count": stack("contact_count").astype(np.int32),
            # team-parity keys (same names as voxposer_dump.npz)
            "obstacle_name": np.asarray(self.obstacle_name),
            "obstacle_pos_init": np.asarray(self.obstacle_pos_init, dtype=np.float32).reshape(k, 3),
            "obstacle_xy_init": (np.asarray(self.obstacle_pos_init[0][:2], dtype=np.float32)
                                 if k else np.full(2, np.nan, dtype=np.float32)),
            "goal_xy_fixture": self.goal_xy.astype(np.float32),
        }
        return out

    def save(self, path: Any, meta: dict[str, Any] | None = None) -> dict[str, tuple[int, ...]]:
        arrays = self.arrays(meta)
        np.savez_compressed(str(path), **arrays)
        return {k: tuple(v.shape) for k, v in arrays.items()}
