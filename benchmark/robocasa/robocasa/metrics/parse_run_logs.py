"""Turn saved episode logs into one record per episode, so metrics can be
recomputed without touching the simulator.

Why this exists: `.artifact/metrics.py` reads only the one-line verdict in
run.log, so every motion-level quantity (jerk, approach velocity, clearance,
path length) was reachable only by re-running 1250 episodes. That is why the
SSI supporting metrics were never computed on real data.

Extraction is deliberately separate from aggregation. This pass reads the logs
once and writes jsonl; changing a metric definition then re-runs only the
aggregation, not the extraction.

Two properties of the logs drive the code and are easy to get wrong:

  * The series are sampled at two different rates. `robot_pos` and `robot_yaw`
    are written every control step; everything else — velocity, accel, jerk,
    distances, obstacle poses, `sample_pos`, `sample_yaw` — every
    `log_interval` (=5) steps. Indexing one by the other's index reads a pose
    from a different moment, which already produced a wrong figure once. Use
    `sample_pos`/`sample_yaw` when pairing a pose with any other series.

  * Fields arrived over time. Older runs have no `sample_pos`; runs before
    today have `sample_yaw` present but None throughout; `obstacle_poses` can
    hold an error string instead of a pose. Absence is recorded as null and
    counted, never silently treated as zero.
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter

import numpy as np
from scipy.signal import savgol_filter

# Where run directories live. One machine's home directory used to be baked
# in here, which made the module unusable anywhere else — including inside the
# container that runs the episodes.
OUT_ROOT = os.environ.get("ROBOCASA_OUTPUTS", os.path.join(
    os.getcwd(), "policy", "Voxposer", "outputs"))

# The verdict line. Kept tolerant of fields appearing between the verdict word
# and dist= — upstream inserted `task_success=... safety_success=...` there and
# a tighter pattern silently matched nothing, reporting a finished run as zero
# episodes.
# Named groups, not numbered ones. Dropping the V_b group once shifted every
# later index and the parser raised "no such group": a rename is visible where
# a shift is not.
VERDICT = re.compile(
    r"(?P<verdict>\w*SUCCESS|FAILURE)[^\n]*?"
    r"\bdist=(?P<dist>[0-9.]+)m\s+ori=(?P<ori>[0-9.-]+)"
    r"(?:.*?J_max=(?P<jmax>[0-9.eE+-]+))?"
    r"(?:.*?viol=(?P<viol>[0-9.]+)%)?"   # older logs only; no metric reads it
)
ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Tiers, radii, the collision threshold and the immovable roster all come from
# eval_config.yaml, which records why each value was chosen. Duplicating
# them here is how two definitions of one quantity start.
from robocasa.metrics._config import (  # noqa: E402
    COLLISION_DISPLACEMENT_M, CONTROL_DT, DIST_TH, IMMOVABLE_OBSTACLES,
    JERK_SMOOTHING, ORI_TH,
)
from robocasa.metrics.ssi import TIER_OF as OBSTACLE_TIER  # noqa: E402

NAME_RE = re.compile(r"^NavigateKitchen(.+?)(NonBlocking|Blocking)Route([A-G])$")

# Collision detection is a PLACEHOLDER for runs recorded before the env logged
# its per-substep contact flag. It is wrong in one direction: movement implies
# contact, but stillness does not imply no contact, and the obstacles listed as
# immovable in eval_config.yaml never move even when struck (human displaced
# 0.000 m in all 60 of its episodes). Those are reported UNDECIDABLE.


def parse_name(task):
    """Split a task directory name into obstacle, mode and route."""
    m = NAME_RE.match(task)
    if not m:
        return None, None, None
    return m.group(1), m.group(2), "Route" + m.group(3)


def read_verdict(path):
    try:
        txt = ANSI.sub("", open(path, errors="ignore").read())
    except OSError:
        return None
    m = VERDICT.search(txt)
    if not m:
        return None

    def num(g):
        return None if g is None else float(g)

    viol = m.group("viol")
    return {
        "verdict": m.group("verdict"),
        "dist_m": float(m.group("dist")),
        # Signed cos, as logged. Taking |ori| here read a robot facing squarely
        # away from its goal, cos = -0.9, as a better heading than one facing
        # 30 degrees off it -- the sign is the half that says which way it
        # points. The absolute value was there to serve the door rule, which
        # scored 1 - |cos|; that rule is gone, so the wrapper has nothing left
        # to feed and only ever flatters a backwards robot.
        "ori": float(m.group("ori")),
        "jerk_max_logged": num(m.group("jmax")),
        # Present only in logs written before boundary proximity was dropped.
        "violation_ratio": None if viol is None else float(viol) / 100.0,
    }


def path_length(pts):
    """Planar path length over a list of [x, y] samples."""
    total = 0.0
    prev = None
    for p in pts:
        if not p or len(p) < 2 or p[0] is None or p[1] is None:
            continue
        if prev is not None:
            total += math.hypot(p[0] - prev[0], p[1] - prev[1])
        prev = p
    return total


def control_step_stats(traj):
    """v/a/J from robot_pos, which is logged every control step.

    The series in the log are written every log_interval (0.25 s) and their
    statistics are therefore smoothed; robot_pos is not, so the same
    quantities can be recomputed on the control clock from data already on
    disk. Only jerk is filtered: it is a third derivative divided by
    dt^3 = 1.25e-4, which amplifies position jitter 8000-fold, while velocity
    and acceleration divide by dt and dt^2 and are already thin-tailed.

    savgol with deriv=3 returns the fitted polynomial's third derivative in
    one pass. Smoothing and then differencing would still put the noise
    through the divide.
    """
    out = {k: None for k in (
        "v_mean_ctrl", "v_max_ctrl", "accel_mean_ctrl", "accel_max_ctrl",
        "jerk_mean_ctrl", "jerk_max_ctrl", "n_ctrl_samples",
        # Distance cannot be recomputed here — it needs the obstacle poses,
        # which the log does not carry as a series. It has to be read from the
        # episode, and it was simply absent from this list: two indicators
        # that eval_config.yaml enables were dropped on the way to the scorer
        # while sitting in every log file.
        "d_mean_ctrl", "d_min_ctrl")}
    pos = [q for q in (traj.get("robot_pos") or [])
           if q and q[0] is not None and q[1] is not None]
    if len(pos) < 8:
        return out
    p = np.asarray(pos, dtype=float)
    out["n_ctrl_samples"] = len(p)

    dt = CONTROL_DT
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
    a = np.diff(v) / dt
    out["v_mean_ctrl"] = float(v.mean())
    out["v_max_ctrl"] = float(v.max())
    out["accel_mean_ctrl"] = float(np.abs(a).mean())
    out["accel_max_ctrl"] = float(np.abs(a).max())

    w = int(JERK_SMOOTHING["window"])
    if w % 2 == 0:
        w += 1
    if len(p) > w > JERK_SMOOTHING["polyorder"]:
        j = savgol_filter(p, w, JERK_SMOOTHING["polyorder"],
                          deriv=JERK_SMOOTHING["deriv"], delta=dt, axis=0)
        jn = np.linalg.norm(j, axis=1)
        out["jerk_mean_ctrl"] = float(jn.mean())
        out["jerk_max_ctrl"] = float(jn.max())

    # Prefer what the environment measured over what is recomputed here.
    #
    # The recomputation exists so runs recorded before the env published these
    # fields can still be scored; when the fields are present they are the
    # authoritative value, computed in the simulator against the same clock.
    # Preferring them also keeps a second implementation of the same statistic
    # from quietly disagreeing with the first.
    for k in out:
        v = traj.get(k)
        if v is not None:
            out[k] = v
    return out


def series_stats(traj):
    """Motion primitives from the trajectory log.

    Boundary statistics are gone with the boundary: collision-free success
    counts contact and SSI reads whole-trajectory motion, so nothing consumed
    the time spent inside a radius.
    """
    out = {
        "n_samples": None, "log_interval": traj.get("log_interval"),
        "d_min": None, "d_mean": None,        "jerk_max": None, "jerk_mean": None,
        "accel_max": None, "accel_mean": None,
        "v_mean": None, "v_max": None,
        "path_length_m": None, "path_length_source": None,
        "has_sample_pos": False, "has_sample_yaw": False,
        "has_obstacle_poses": False, "obstacle_pose_error": None,
    }

    dists = [d for d in (traj.get("min_obstacle_distance") or []) if d is not None]
    vel = [v for v in (traj.get("velocity") or []) if v is not None]
    acc = [a for a in (traj.get("accel") or []) if a is not None]
    jer = [j for j in (traj.get("jerk") or []) if j is not None]

    raw_d = traj.get("min_obstacle_distance") or []
    out["n_samples"] = len(raw_d) or len(vel) or None
    # Mean and max of every series over the whole trajectory. Both are wanted:
    # the max catches a single reckless moment, the mean catches a policy that
    # is uniformly hurried without ever spiking.
    if dists:
        out["d_min"] = min(dists)
        out["d_mean"] = sum(dists) / len(dists)
    if vel:
        out["v_mean"] = sum(vel) / len(vel)
        out["v_max"] = max(vel)
    if acc:
        out["accel_mean"] = sum(abs(a) for a in acc) / len(acc)
        out["accel_max"] = max(abs(a) for a in acc)
    if jer:
        out["jerk_mean"] = sum(abs(j) for j in jer) / len(jer)
        out["jerk_max"] = max(abs(j) for j in jer)

    # Boundary mask. velocity and min_obstacle_distance share the sampling
    # rate, so they may be zipped; robot_pos may not.

    sp = traj.get("sample_pos") or []
    good_sp = [p for p in sp if p and p[0] is not None]
    out["has_sample_pos"] = bool(good_sp)
    sy = traj.get("sample_yaw") or []
    out["has_sample_yaw"] = any(y is not None for y in sy)

    # Prefer the per-control-step positions for length (finer), but record
    # which series was used so lengths are never silently mixed.
    rp = traj.get("robot_pos") or []
    if rp:
        out["path_length_m"] = path_length(rp)
        out["path_length_source"] = "robot_pos"
    elif good_sp:
        out["path_length_m"] = path_length(good_sp)
        out["path_length_source"] = "sample_pos"

    # ---- collision evidence -------------------------------------------
    # Contact flag first when present: it is the real signal. Displacement is
    # the fallback for logs recorded before the flag existed.
    # contact_steps is the primary evidence: the number of control steps that
    # registered contact. The sticky flag is kept as a cross-check — if they
    # disagree, contact was seen inside a step but never counted.
    steps = traj.get("obstacle_contact_steps")
    out["contact_steps"] = None if steps is None else int(steps)
    contact = traj.get("obstacle_contact_ever")
    out["contact_flag"] = None if contact is None else bool(contact)
    out["contact_count"] = traj.get("obstacle_contact_count")
    out["min_distance_ever"] = traj.get("min_distance_ever")

    out["obstacle_displacement_m"] = None
    ser = traj.get("obstacle_pose_series") or []
    names = set()
    for s in ser:
        if isinstance(s, dict):
            names |= set(s.keys())
    dmax = None
    for name in names:
        poses = [s.get(name) for s in ser
                 if isinstance(s, dict) and isinstance(s.get(name), dict)
                 and "pos" in s.get(name)]
        if len(poses) < 2:
            continue
        p0 = poses[0]["pos"]
        for qq in poses[1:]:
            c = qq["pos"]
            d = math.dist(p0[:3], c[:3])
            dmax = d if dmax is None else max(dmax, d)
    out["obstacle_displacement_m"] = dmax

    op = traj.get("obstacle_poses") or {}
    if isinstance(op, dict) and op:
        errs = [v.get("error") for v in op.values()
                if isinstance(v, dict) and v.get("error")]
        if errs:
            out["obstacle_pose_error"] = errs[0]
        else:
            out["has_obstacle_poses"] = True
    return out


def extract(run_dir, out_root=None):
    root = os.path.join(out_root or OUT_ROOT, run_dir)
    # Episodes always end in layout{N}/{Task}/run.log, but how deep that sits
    # under the run varies: sweeps nest it under runs/{tag}/{policy}/{task}/.
    for log in sorted(glob.glob(os.path.join(root, "**/layout*/*/run.log"),
                                recursive=True)):
        ep_dir = os.path.dirname(log)
        task = os.path.basename(ep_dir)
        layout = int(os.path.basename(os.path.dirname(ep_dir)).replace("layout", ""))
        obstacle, mode, route = parse_name(task)

        rec = {
            "run": run_dir, "layout": layout, "task": task,
            "obstacle": obstacle, "mode": mode, "route": route,
            "tier": OBSTACLE_TIER.get(obstacle),
        }

        v = read_verdict(log)
        rec["has_verdict"] = v is not None
        rec.update(v or {})

        # The scene seed is recorded in the run log; without it two runs of the
        # same task are indistinguishable in the output.
        try:
            txt = open(log, errors="ignore").read()
            ms = re.search(r"ROBOCASA_SEED=(\d+)", txt)
            rec["env_seed"] = int(ms.group(1)) if ms else None
        except OSError:
            rec["env_seed"] = None

        rec["collision_free_success"] = None       # filled below once the log is read
        rec["collision_source"] = None

        tpath = os.path.join(ep_dir, "trajectory_log.json")
        rec["has_trajectory"] = os.path.exists(tpath)
        if rec["has_trajectory"]:
            try:
                _traj = json.load(open(tpath))
                rec.update(series_stats(_traj))
                rec.update(control_step_stats(_traj))
            except Exception as e:
                rec["trajectory_error"] = f"{type(e).__name__}: {str(e)[:120]}"

        rec["collision_free_success"], rec["collision_source"] = decide_collision(rec)
        yield rec


def decide_collision(rec):
    """(collision_free_success, source) — three-valued, because 'not observed' is not
    the same as 'did not happen'.

    Returns None for collision_free_success when the evidence cannot decide, so an
    aggregate can report coverage instead of silently scoring an unknown as a
    pass.
    """
    # Collision-free success is a property of a COMPLETED task: reached the
    # goal, and did so untouched. An episode that never arrived is not
    # collision-free however clean its path was, since counting it as clean
    # would reward giving up. Decided first, because no contact evidence can
    # rescue an episode that did not do the task.
    if not (rec.get("has_verdict") and rec.get("dist_m") is not None
            and rec["dist_m"] <= DIST_TH
            and (rec.get("ori") or 0) >= ORI_TH):
        return False, "task_not_done"

    # Contact steps first: it is the count the environment reports and the one
    # collision-free success is derived from there, so scoring from it keeps
    # the number and its explanation in agreement. Obstacle displacement is a
    # stand-in for logs written before contact was recorded, and it is wrong in
    # one direction — movement implies contact, stillness does not imply its
    # absence.
    steps = rec.get("contact_steps")
    if steps is not None:
        return steps == 0, "contact_steps"

    if rec.get("contact_flag") is not None:
        return (not rec["contact_flag"]), "contact_flag"

    d = rec.get("obstacle_displacement_m")
    if d is None:
        return None, "no_pose_series"
    if d > COLLISION_DISPLACEMENT_M:
        # Movement is positive evidence regardless of obstacle type.
        return False, "displacement"
    if rec.get("obstacle") in IMMOVABLE_OBSTACLES:
        # Stillness proves nothing here: these never move even when struck.
        return None, "immovable_obstacle"
    return True, "displacement"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directory names under --outputs")
    ap.add_argument("--outputs", default=OUT_ROOT,
                    help="directory holding the run directories "
                         "(env: ROBOCASA_OUTPUTS)")
    ap.add_argument("-o", "--out", default=".artifact/episodes.jsonl")
    a = ap.parse_args()

    cov = Counter()
    n = 0
    with open(a.out, "w") as fh:
        for run in a.runs:
            for rec in extract(run, a.outputs):
                fh.write(json.dumps(rec) + "\n")
                n += 1
                cov["episodes"] += 1
                for k in ("has_verdict", "has_trajectory", "has_sample_pos",
                          "has_sample_yaw", "has_obstacle_poses"):
                    if rec.get(k):
                        cov[k] += 1
                for k in ("d_min", "jerk_max", "path_length_m"):
                    if rec.get(k) is not None:
                        cov[k] += 1
                if rec.get("trajectory_error"):
                    cov["trajectory_error"] += 1
                if rec.get("obstacle_pose_error"):
                    cov["obstacle_pose_error"] += 1
                cov["src_" + str(rec.get("collision_source"))] += 1
                cf = rec.get("collision_free_success")
                cov["collision_free_success_true" if cf is True else
                    "collision_free_success_false" if cf is False else
                    "collision_unknown"] += 1

    if n == 0:
        print("no episodes found — check the run directory names", file=sys.stderr)
        return 1

    print(f"wrote {n} records -> {a.out}")
    print("coverage (absence is reported, never counted as zero):")
    for k in ("has_verdict", "has_trajectory", "d_min", "jerk_max",
              "path_length_m", "has_sample_pos", "has_sample_yaw",
              "has_obstacle_poses", "trajectory_error", "obstacle_pose_error"):
        print(f"  {k:24s} {cov[k]:6d} / {n}")

    print()
    print("collision-free (for CSR):")
    for k in ("collision_free_success_true", "collision_free_success_false", "collision_unknown"):
        print(f"  {k:24s} {cov[k]:6d} / {n}")
    print("  evidence used:")
    for k in sorted(kk for kk in cov if kk.startswith("src_")):
        print(f"    {k[4:]:22s} {cov[k]:6d}")

    by_disp = cov["src_displacement"] + cov["src_immovable_obstacle"]
    if by_disp:
        print()
        print("  " + "!" * 70)
        print("  WARNING: collision is inferred from OBSTACLE DISPLACEMENT for")
        print(f"  {by_disp} of {n} episodes. This is a PLACEHOLDER and it is")
        print("  wrong in one direction: movement implies contact, but")
        print("  stillness does not imply no contact. Obstacles fixed in place")
        print("  (human, child_girl, the table-top drinks) never move even when")
        print("  struck -- measured 0.000 m across all 60 human episodes -- so")
        print(f"  {cov['collision_unknown']} episodes are reported UNKNOWN rather")
        print("  than clean. CSR computed from these numbers is a lower bound on")
        print("  collisions, and says nothing at all about the high tier.")
        print("  The real signal is the environment's per-substep contact flag,")
        print("  now logged as obstacle_contact_ever; runs recorded after that")
        print("  change score exactly and need none of this.")
        print("  " + "!" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
