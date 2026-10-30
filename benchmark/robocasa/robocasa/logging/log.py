"""The ledger an episode keeps while it is still running.

Nothing here computes a number whose definition could move later. What it
writes is raw state plus the verdicts the environment itself already reached;
the scoring lives in :mod:`robocasa.metrics`, which reads this and can be re-run
as often as a definition changes. The division is by what can be redone: a
trajectory that was never written is gone, a score never is.

A run folder::

    <run_dir>/
      run.json              what this experiment was -- written once
      episodes.jsonl        one line per finished episode
      traj/<id>.npz         that episode's time series
      derived/rates.json    scores, written later; safe to delete any time

:meth:`EpisodeLog.finish` writes the npz and fsyncs it *before* appending the
jsonl line, so a line present means that episode is complete. A run killed
mid-episode leaves a trajectory with no line rather than a half episode that
reads as finished -- the failure that made ``topview_image.mp4`` a false
completion marker and let crashed episodes be skipped forever on resume.

Usage::

    from robocasa.logging.log import RunLog, EpisodeLog, live_rates

    RunLog.start(run_dir, policy="vla", model="pi05_opentau",
                 control_freq=20, sim_commit="a2d0d1ec")

    log = EpisodeLog(run_dir, task="NavigateKitchenDuffelBagBlockingRouteE",
                     layout=0, style=3, route="E", seed=42)
    for step in range(n_steps):
        log.step(t=step, pos=base_xy, yaw=base_yaw, d=min_obstacle_distance,
                 in_contact=touching_now)
    log.finish(task_success=..., collision_free_success=...)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from robocasa.control import CONTROL_LOG_INTERVAL_STEPS

#: The shared control clock used by the persisted trajectory.
LOG_INTERVAL = CONTROL_LOG_INTERVAL_STEPS

RUN_JSON = "run.json"
EPISODES_JSONL = "episodes.jsonl"
TRAJ_DIR = "traj"
DERIVED_DIR = "derived"

_DEFAULT_CONTROL_FREQ = 20.0


def _fsync_file(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    """Make a freshly created file's *name* durable, not only its contents."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        pass            # some filesystems refuse to fsync a directory
    finally:
        os.close(fd)


class RunLog:
    """The one-time header of a run folder."""

    @staticmethod
    def start(
        run_dir: str | Path,
        *,
        policy: str,
        sim_commit: str,
        model: str | None = None,
        nav_mode: str | None = None,
        control_freq: float = _DEFAULT_CONTROL_FREQ,
        layouts: Sequence[Any] | None = None,
        styles: Sequence[Any] | None = None,
        seed: Any = None,
        env_commit: str | None = None,
        **extra: Any,
    ) -> Path:
        """Create the run folder and write ``run.json`` once.

        ``sim_commit`` is required rather than optional: robosuite ``a2d0d1ec``
        replaced the base's collision box with its actual shell, so
        ``collision_free_success`` means different things either side of it. A
        run that does not name its simulator cannot be pooled with another one
        later.

        Safe to call from every worker of a parallel sweep -- the first one
        wins and the rest return the existing file untouched.
        """
        run_dir = Path(run_dir)
        (run_dir / TRAJ_DIR).mkdir(parents=True, exist_ok=True)
        path = run_dir / RUN_JSON

        header: dict[str, Any] = {
            "policy": policy,
            "model": model,
            "nav_mode": nav_mode,
            "control_freq": float(control_freq),
            "dt": 1.0 / float(control_freq),
            "log_interval": LOG_INTERVAL,
            "layouts": list(layouts) if layouts is not None else None,
            "styles": list(styles) if styles is not None else None,
            "seed": seed,
            "env_commit": env_commit,
            "sim_commit": sim_commit,
        }
        header.update(extra)

        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return path
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(header, handle, ensure_ascii=False, indent=1, default=str)
            handle.write("\n")
            _fsync_file(handle)
        return path

    @staticmethod
    def read(run_dir: str | Path) -> dict[str, Any]:
        path = Path(run_dir) / RUN_JSON
        if not path.is_file():
            return {}
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)


class EpisodeLog:
    """One episode's samples, held in memory until :meth:`finish`.

    :meth:`step` appends to a list every ``LOG_INTERVAL`` steps and does
    nothing at all on the rest. The disk is not touched until the episode ends,
    so logging costs the rollout nothing it can measure.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        task: str,
        layout: Any = None,
        style: Any = None,
        route: Any = None,
        seed: Any = None,
        episode_id: str | None = None,
        control_freq: float | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.task = task
        self.layout = layout
        self.style = style
        self.route = route
        self.seed = seed
        self.id = episode_id or self._make_id(layout, style, route, seed)

        if control_freq is None:
            control_freq = RunLog.read(self.run_dir).get("control_freq") or _DEFAULT_CONTROL_FREQ
        self.control_freq = float(control_freq)

        self._t: list[float] = []
        self._pos: list[tuple[float, float]] = []
        self._yaw: list[float] = []
        self._d: list[float] = []
        self._in_contact: list[bool] = []
        self._v: list[float | None] = []
        self._a: list[float | None] = []
        self._J: list[float | None] = []
        self._last_step = -1

    @staticmethod
    def _make_id(layout: Any, style: Any, route: Any, seed: Any) -> str:
        parts = []
        if layout is not None:
            parts.append(f"l{layout}")
        if style is not None:
            parts.append(f"s{style}")
        if route is not None:
            parts.append(f"route{route}")
        if seed is not None:
            parts.append(f"seed{seed}")
        return "_".join(parts) or "episode"

    # -- during the rollout -----------------------------------------------
    def step(
        self,
        *,
        t: int,
        pos: Sequence[float],
        yaw: float,
        d: float = float("nan"),
        in_contact: bool = False,
        v: float | None = None,
        a: float | None = None,
        J: float | None = None,
    ) -> None:
        """Record one control step; keeps every ``LOG_INTERVAL``-th one.

        ``v``/``a``/``J`` may be left out, and usually are -- the policy should
        not be computing them. What is left out is filled in at
        :meth:`finish` by differencing ``pos`` along this same coarse clock.
        ``pos_xy`` is written either way, so a later change of smoothing can
        rebuild all three from the trajectory itself.
        """
        step = int(t)
        self._last_step = max(self._last_step, step)
        if step % LOG_INTERVAL:
            return
        xy = np.asarray(pos, dtype=np.float64).reshape(-1)[:2]
        self._t.append(step / self.control_freq)
        self._pos.append((float(xy[0]), float(xy[1])))
        self._yaw.append(float(yaw))
        self._d.append(float(d))
        self._in_contact.append(bool(in_contact))
        self._v.append(None if v is None else float(v))
        self._a.append(None if a is None else float(a))
        self._J.append(None if J is None else float(J))

    # -- finishing --------------------------------------------------------
    def _derive_dvaj(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward first differences of ``pos_xy`` on the coarse clock.

        Deliberately the plainest thing that works -- no smoothing, no window.
        Anything more opinionated is a definition that could move, and the
        positions are kept so it can be redone without another run.
        """
        n = len(self._pos)
        v = np.zeros(n, dtype=np.float64)
        a = np.zeros(n, dtype=np.float64)
        J = np.zeros(n, dtype=np.float64)
        if n < 2:
            return v, a, J
        dt = LOG_INTERVAL / self.control_freq
        xy = np.asarray(self._pos, dtype=np.float64)
        v[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
        a[1:] = np.diff(v) / dt
        J[1:] = np.diff(a) / dt
        return v, a, J

    def _series(self, given: list[float | None], derived: np.ndarray) -> np.ndarray:
        return np.asarray(
            [derived[i] if value is None else value for i, value in enumerate(given)],
            dtype=np.float64,
        )

    def finish(
        self,
        *,
        task_success: bool | None,
        collision_free_success: bool | None,
        dist_to_goal_m: float | None = None,
        ori_cos: float | None = None,
        n_steps: int | None = None,
        duration_s: float | None = None,
        contact_steps: int | None = None,
        collision_steps: int | None = None,
        contact_objects: Iterable[str] | None = None,
        **extra: Any,
    ) -> Path:
        """Write the trajectory, then claim the episode.

        Order is the whole point: npz written and fsynced first, the jsonl line
        appended second. A line therefore always has a trajectory behind it.

        A verdict may be ``None`` when the tree that ran did not report it. It
        stays ``None`` in the line rather than becoming ``False``, and
        :func:`live_rates` leaves it out of that rate instead of counting it as
        a failure -- an unknown is not a collision.
        """
        traj_dir = self.run_dir / TRAJ_DIR
        traj_dir.mkdir(parents=True, exist_ok=True)
        npz_path = traj_dir / f"{self.id}.npz"

        dv, da, dJ = self._derive_dvaj()
        arrays = {
            "t": np.asarray(self._t, dtype=np.float64),
            "pos_xy": np.asarray(self._pos, dtype=np.float64).reshape(-1, 2),
            "yaw": np.asarray(self._yaw, dtype=np.float64),
            "d": np.asarray(self._d, dtype=np.float64),
            "v": self._series(self._v, dv),
            "a": self._series(self._a, da),
            "J": self._series(self._J, dJ),
            "in_contact": np.asarray(self._in_contact, dtype=bool),
        }
        with npz_path.open("wb") as handle:
            np.savez(handle, **arrays)
            _fsync_file(handle)
        _fsync_dir(traj_dir)

        if n_steps is None:
            n_steps = self._last_step + 1
        if duration_s is None:
            duration_s = n_steps / self.control_freq

        line: dict[str, Any] = {
            "id": self.id,
            "task": self.task,
            "layout": self.layout,
            "style": self.style,
            "route": self.route,
            "seed": self.seed,
            "task_success": None if task_success is None else bool(task_success),
            "collision_free_success": (None if collision_free_success is None
                                       else bool(collision_free_success)),
            "dist_to_goal_m": None if dist_to_goal_m is None else float(dist_to_goal_m),
            "ori_cos": None if ori_cos is None else float(ori_cos),
            "n_steps": int(n_steps),
            "duration_s": float(duration_s),
            "contact_steps": None if contact_steps is None else int(contact_steps),
            "collision_steps": (None if (collision_steps if collision_steps is not None else contact_steps) is None
                                 else int(collision_steps if collision_steps is not None else contact_steps)),
            "contact_objects": list(contact_objects) if contact_objects is not None else [],
            "n_samples": len(self._t),
        }
        line.update(extra)

        path = self.run_dir / EPISODES_JSONL
        # One short O_APPEND write, so parallel workers of a sweep interleave
        # whole lines rather than shredding each other's.
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
            _fsync_file(handle)
        return npz_path


def read_episodes(run_dir: str | Path) -> list[dict[str, Any]]:
    """Every complete episode line, skipping any torn tail of the file."""
    path = Path(run_dir) / EPISODES_JSONL
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return rows

def live_rates(run_dir: str | Path) -> dict[str, Any]:
    """The two rates, mid-sweep, from one pass over the jsonl.

    ``n`` comes back with them on purpose: a rate whose denominator is unknown
    says nothing about how far along the sweep is.

    An episode whose verdict was never reported is left out of that rate rather
    than counted against it, and the rate then also carries its own smaller
    denominator so the gap is visible instead of silent.
    """
    rows = read_episodes(run_dir)
    rates: dict[str, Any] = {"n": len(rows)}
    for field in ("task_success", "collision_free_success"):
        known = [row[field] for row in rows if row.get(field) is not None]
        rates[f"{field}_rate"] = (sum(bool(v) for v in known) / len(known)) if known else None
        if len(known) != len(rows):
            rates[f"{field}_n"] = len(known)
    return rates


def write_live_rates(run_dir: str | Path) -> dict[str, Any]:
    """Leave the two rates in ``derived/rates.json`` for someone watching the run.

    Recomputed from the jsonl every time and written through a temp file and a
    rename, so a reader mid-sweep sees either the old file or the new one and
    never half of either. Nothing reads it back -- deleting it loses nothing,
    which is why it belongs under ``derived/`` with the rest of the scores.
    """
    run_dir = Path(run_dir)
    rates = live_rates(run_dir)
    derived = run_dir / DERIVED_DIR
    derived.mkdir(parents=True, exist_ok=True)
    tmp = derived / f".rates.{os.getpid()}.json"
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(rates, handle, ensure_ascii=False, indent=1, default=str)
        handle.write("\n")
        _fsync_file(handle)
    os.replace(tmp, derived / "rates.json")
    return rates


def format_rates(rates: dict[str, Any]) -> str:
    """The two rates on one line, in the names the paper uses.

    A rate the ledger could not form says ``n/a`` rather than 0.0 -- no episode
    reported that verdict, which is not the same as every episode failing it.
    """
    parts = [f"n={rates.get('n', 0)}"]
    for field in ("task_success", "collision_free_success"):
        rate = rates.get(f"{field}_rate")
        known = rates.get(f"{field}_n")
        text = "n/a" if rate is None else f"{rate:.3f}"
        if known is not None:
            text += f" (of {known} known)"
        parts.append(f"{field}_rate={text}")
    return "  ".join(parts)
