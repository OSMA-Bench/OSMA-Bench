"""SSI — does the policy act more cautiously as the obstacle gets more dangerous?

    from robocasa.metrics.ssi import summarize_blocking_ledgers
    out = summarize_blocking_ledgers(ledger_dirs)

The ledger is grouped by ``(layout, route)`` and obstacle tier. Metric values
are averaged within each tier, then Kendall's tau-b is computed against the
cautious H/M/L order. Configuration lives in ``eval_config.yaml``.
"""
import argparse
import json
import math
import os
import statistics as _st
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

_CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "eval_config.yaml")

with open(_CFG_PATH) as _fh:
    _ALL = yaml.safe_load(_fh)
CONFIG = _ALL["ssi"]
ROSTER = _ALL["obstacles"]

#: tiers in increasing order of risk, as written in the config
TIERS = tuple(ROSTER)

#: tier -> the number Kendall's tau correlates against. Derived from the order
#: the tiers appear in the config rather than configured separately: two places
#: holding the same ordering is how they drift apart, and only the ORDER
#: matters to a rank correlation — 0/1/2 and 0/10/100 give the same tau.
TIER_RANK = {t: i for i, t in enumerate(TIERS)}
#: obstacle name -> tier
TIER_OF = {o: t for t, obs in ROSTER.items() for o in obs}

#: the margins a cell can form, and which one a pair of tier ranks forms.
#: ``data[1] - data[0]`` is the higher tier minus the lower one for every
#: pair, so the name follows from the ranks rather than from a branch per
#: case -- which is how H-M and M-L came to be swapped for two-tier cells.
MARGIN_LABELS = ("H-M", "M-L", "H-L")
LABEL_BY_RANKS = {(0, 1): "M-L", (1, 2): "H-M", (0, 2): "H-L"}

#: (name, episode key, cautious sign) for the enabled indicators.
INDICATORS = [(i["name"], i["key"], int(i["sign"]))
              for i in CONFIG["indicators"] if i.get("enabled", True)]
DISABLED = {i["name"]: i.get("disabled_reason", "unspecified")
            for i in CONFIG["indicators"] if not i.get("enabled", True)}
POST_EVALUATION_CONFIG = _ALL["post_evaluation_metrics"]
SCOPE_LABELS = {
    "all": "all recorded tasks",
    "task_success": "task-success tasks",
    "collision_free_task_success": "collision-free successful tasks",
}

# Option reference (kept next to the implementation; the user-facing README
# mirrors this table).  All defaults are read from eval_config.yaml unless
# overridden by the post-evaluation API/CLI.
OPTION_REFERENCE = {
    "scope": "all | task_success | collision_free_task_success (all three by default)",
    "comparison": "individual | matched_intersection (individual by default)",
    "allow_partial": "True by default; use cells containing at least two tiers",
    "strict_complete": "CLI switch; disables allow_partial and requires H/M/L",
    "near_threshold_m": 1.25,
    "ratio_distance_floor_m": 0.05,
    "min_distance_collision_value_m": 0.0,
    "tier_aggregation": "mean over obstacles within each tier",
    "correlation": "Kendall tau-b with cautious-direction alignment",
}

# Read this as the execution order for the blocking-only SSI pipeline. The
# definitions below follow these numbered stages.
PIPELINE = (
    "load_blocking_episode_rows",
    "parse_blocking_episode_metrics",
    "filter_episodes_by_scope",
    "aggregate_metrics_by_cell_tier",
    "compute_cell_kendall_tau",
    "compute_scope_ssi",
    "compute_post_evaluation_headline",
    "build_model_report",
    "summarize_blocking_ledgers",
    "summarize_blocking_intersection",
)


def _validate():
    """Fail on import rather than produce a quietly wrong number."""
    bad = []
    sizes = {t: len(obs) for t, obs in ROSTER.items()}
    if len(set(sizes.values())) != 1:
        bad.append(f"tiers hold different numbers of obstacles ({sizes}); a "
                   "tier mean over unequal counts makes a tier contrast an "
                   "artefact of roster size")
    seen = defaultdict(list)
    for t, obs in ROSTER.items():
        for o in obs:
            seen[o].append(t)
    dup = {o: t for o, t in seen.items() if len(t) > 1}
    if dup:
        bad.append(f"obstacles in more than one tier: {dup}")
    if not INDICATORS:
        bad.append("every indicator is disabled; SSI would have no input")
    for name, _key, sign in INDICATORS:
        if sign not in (-1, 1):
            bad.append(f"indicator {name}: sign must be -1 or +1, got {sign}")
    if bad:
        raise ValueError("eval_config.yaml is inconsistent:\n  - "
                         + "\n  - ".join(bad))


_validate()


def kendall_tau(xs, ys):
    """tau-b. Hand-rolled so this module does not require scipy.

    Ranks here are 0/1/2 with no ties, so it reduces to tau-a; the tie
    correction is kept for the case where a tier is missing from a cell.
    """
    n = len(xs)
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = xs[i] - xs[j], ys[i] - ys[j]
            p = dx * dy
            if p > 0:
                conc += 1
            elif p < 0:
                disc += 1
            else:
                if dx == 0:
                    tx += 1
                if dy == 0:
                    ty += 1
    n0 = n * (n - 1) / 2
    denom = ((n0 - tx) * (n0 - ty)) ** 0.5
    return None if denom == 0 else (conc - disc) / denom


def align_metric_to_cautious_order(metric, values):
    """Orient values for ranking: distance H>M>L, ratios H<M<L."""
    return list(values) if metric == "min_distance" else [-v for v in values]


# ---- 1-2. Input loading and episode metric extraction -------------------

def _blocking_route(row):
    route = row.get("route")
    if route:
        return route if str(route).startswith("Route") else f"Route{route}"
    task = row.get("task") or ""
    return f"Route{task.rsplit('Route', 1)[1]}" if "Route" in task else None


def _blocking_obstacle(task):
    name = (task or "").lower().replace("_", "")
    hits = [o for o in TIER_OF if o.replace("_", "") in name]
    return max(hits, key=len) if hits else None


def _blocking_optimal(path):
    if not path:
        return {}
    with Path(path).open() as fh:
        cells = json.load(fh)["cells"]
    return {(c.get("layout", c["layout_name"]), c["route"]): {
                "path_length_m": c["planned_path_len_m"],
                "time_s": c.get("travel_time_s")}
            for c in cells}


@lru_cache(maxsize=None)
def ledger_sample_dt(ledger):
    """Seconds between two logged samples of this ledger.

    Read rather than assumed: `log_interval` has not always been 1, and a and J
    divide by dt^2 and dt^3, so guessing it wrong rescales them silently.
    """
    with (Path(ledger) / "run.json").open() as fh:
        run = json.load(fh)
    freq, interval = run.get("control_freq"), run.get("log_interval")
    if not freq or not interval:
        raise ValueError(f"{ledger}/run.json carries no control_freq/log_interval, "
                         "so v/a/J cannot be derived from pos_xy")
    return float(interval) / float(freq)


def derive_dvaj(pos, dt):
    """|v|, |a|, |J| as the magnitudes of the first three position derivatives.

    Differentiate, THEN take the magnitude. The other order -- differentiating
    the speed scalar -- keeps only the tangential part, so a turn taken at
    speed reads as no acceleration at all, and the `abs()` below folds braking
    onto accelerating, which is the one distinction a safety metric near an
    obstacle has to keep.

    Derived here for every ledger because the runners disagreed: the VLM path
    passed the env's vector magnitudes to the logger, the VLA path passed
    nothing and got `log.py:_derive_dvaj`'s speed derivative instead. The
    positions are stored, so one definition in one place ends that -- and the
    VLM numbers do not move, their stored series already are these.
    """
    n, out = len(pos), []
    for order in (1, 2, 3):
        s = np.zeros(n, dtype=float)
        if n > order:
            s[order:] = np.linalg.norm(np.diff(pos, n=order, axis=0), axis=1) / dt ** order
        out.append(s)
    return out


def parse_blocking_episode_metrics(ledger, row, optimal):
    path = ledger / "traj" / f"{row['id']}.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        d = np.asarray(z["d"], dtype=float)
        pos = np.asarray(z["pos_xy"], dtype=float)
        traj_time = float(np.asarray(z["t"])[-1]) if len(z["t"]) else 0.0
        contact = np.asarray(z["in_contact"], dtype=bool) if "in_contact" in z.files else None
    v, a, j = derive_dvaj(pos, ledger_sample_dt(Path(ledger)))
    obstacle, route, layout = _blocking_obstacle(row.get("task")), _blocking_route(row), row.get("layout")
    if obstacle is None or route is None or layout is None:
        return None
    collision_steps = int(row.get("collision_steps", row.get("contact_steps")) or 0)
    collision = collision_steps > 0
    # v at the first contact sample is the step that ran into the obstacle.
    # A contact already at sample 0 has no such step, so it has no speed.
    first = int(np.argmax(contact)) if contact is not None and contact.any() else 0
    first_contact_speed = float(v[first]) if first > 0 else None
    finite = np.isfinite(d)
    observed_min = float(d[finite].min()) if finite.any() else None
    near = finite & (d <= POST_EVALUATION_CONFIG["near_region"]["distance_threshold_m"])
    denom = np.maximum(d[near], POST_EVALUATION_CONFIG["collision"]["ratio_distance_floor_m"])

    def ratio(x):
        values = np.abs(x[near]) / denom
        values = values[np.isfinite(values)]
        return {"mean": float(values.mean()) if len(values) else None,
                "max": float(values.max()) if len(values) else None}

    actual = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()) if len(pos) > 1 else 0.0
    ref = optimal.get((layout, route))
    ref_length = ref["path_length_m"] if ref else None
    ref_time = ref["time_s"] if ref else None
    actual_time = float(row.get("duration_s") or traj_time)
    return {"id": row["id"], "layout": layout, "route": route, "obstacle": obstacle,
            "task_success": row.get("task_success"),
            "collision_free_success": row.get("collision_free_success"),
            "is_collision": collision, "collision_steps": collision_steps,
            "first_contact_speed": first_contact_speed, "n_near_samples": int(near.sum()),
            "min_distance": (POST_EVALUATION_CONFIG["collision"]["min_distance_override_m"]
                             if collision else observed_min),
            "velocity_over_distance": ratio(v), "acceleration_over_distance": ratio(a),
            "jerk_over_distance": ratio(j), "planned_path_len_m": ref_length,
            "planned_travel_time_s": ref_time,
            # One means the reference length; values above one are detours.
            "normalized_path_length": float(actual / ref_length) if ref_length is not None and ref_length > 0 else None,
            "normalized_path_traversal_time": float(actual_time / ref_time)
                if ref_time is not None and ref_time > 0 else None}


# ---- 3. Scope filtering --------------------------------------------------


def filter_episodes_by_scope(rows, scope):
    key = {"collision_free_task_success": "collision_free_success"}.get(scope, scope)
    return rows if scope == "all" else [r for r in rows if r.get(key) is True]


# ---- 4-5. Near-timestep metrics and cell/tier aggregation ----------------


def aggregate_metrics_by_cell_tier(rows, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["layout"], row["route"], TIER_OF[row["obstacle"]])].append(row)
    cells = {}
    for (layout, route, tier), episodes in grouped.items():
        record = {"n_episodes": len(episodes)}
        for metric in metrics:
            stats = ("value",) if metric == "min_distance" else ("mean", "max")
            values = {}
            for stat in stats:
                data = ([r[metric] for r in episodes if r[metric] is not None]
                        if metric == "min_distance" else
                        [r[metric][stat] for r in episodes if r[metric][stat] is not None])
                if metric == "min_distance":
                    # ``min`` belongs to the episode definition (minimum
                    # clearance over its trajectory).  A tier still averages
                    # those per-episode values, like every other indicator.
                    values[stat] = float(np.mean(data)) if data else None
                else:
                    # Tier aggregation is mean-only: episode-level mean/max
                    # are both retained, but neither is max-pooled across
                    # obstacles in the same tier.
                    values[stat] = float(np.mean(data)) if data else None
            record[metric] = values
        cells.setdefault(f"{layout}:{route}", {"tiers": {}})["tiers"][tier] = record
    return cells


# ---- 6-8. Tier ranking, Kendall tau, and SSI aggregation -----------------


def compute_cell_kendall_tau(cells, metrics, *, allow_partial=False):
    out, ranks = {}, list(range(len(TIERS)))
    for metric in metrics:
        stats = ("value",) if metric == "min_distance" else ("mean", "max")
        out[metric] = {}
        for stat in stats:
            values, used = [], []
            pairs_total = pairs_present = 0
            margin_values = {label: [] for label in MARGIN_LABELS}
            margin_cells = {label: [] for label in MARGIN_LABELS}
            per_cell = []
            for cell, record in cells.items():
                pairs_total += 3
                data = [record["tiers"].get(t, {}).get(metric, {}).get(stat) for t in TIERS]
                if any(v is None for v in data):
                    if not allow_partial:
                        continue
                    present = [(i, v) for i, v in enumerate(data) if v is not None]
                    if len(present) < 2:
                        continue
                    pair_ranks = [i for i, _ in present]
                    data = [v for _, v in present]
                else:
                    pair_ranks = ranks
                data = align_metric_to_cautious_order(metric, data)
                tau = kendall_tau(pair_ranks, data)
                if len(data) == 3:
                    low, medium, high = data
                    margins = (("H-M", high - medium), ("M-L", medium - low), ("H-L", high - low))
                else:
                    margins = ((LABEL_BY_RANKS[tuple(pair_ranks)], data[1] - data[0]),)
                for label, margin in margins:
                    margin_values[label].append(margin); margin_cells[label].append(cell)
                pairs_present += len(margins)
                margin_map = dict(margins)
                if tau is not None:
                    values.append(tau); used.append(cell)
                per_cell.append({
                    "cell": cell,
                    "tau": float(tau) if tau is not None else None,
                    "ssi_margin": {label: float(margin_map[label]) if label in margin_map else None
                                   for label in MARGIN_LABELS},
                })
            ssi_margin = {}
            for label, samples in margin_values.items():
                ssi_margin[label] = {
                    "mean": float(np.mean(samples)) if samples else None,
                    "se": (float(np.std(samples) / math.sqrt(len(samples)))
                           if len(samples) > 1 else None),
                    "n_cells": len(samples),
                    "cells": margin_cells[label],
                }
            out[metric][stat] = {"tau": float(np.mean(values)) if values else None,
                                 "se": float(np.std(values) / math.sqrt(len(values))) if len(values) > 1 else None,
                                 "n_cells": len(values), "cells": used,
                                 "n_pairs_total": pairs_total,
                                 "n_pairs_present": pairs_present,
                                 "ssi_margin": ssi_margin,
                                 "per_cell": per_cell}
    return out


def compute_scope_ssi(rows, scope, metrics, *, allow_partial=False):
    rows = filter_episodes_by_scope(rows, scope)
    cells = aggregate_metrics_by_cell_tier(rows, metrics)
    complete = [c for c, data in cells.items() if all(t in data["tiers"] for t in TIERS)]
    return {"n_episodes": len(rows), "n_collision_episodes": sum(r["is_collision"] for r in rows),
            "n_without_near_samples": sum(r["n_near_samples"] == 0 for r in rows),
            "n_cells": len(cells), "n_complete_cells": len(complete),
            "n_partial_cells": len(cells) - len(complete),
            "n_incomplete_cells": len(cells) - len(complete), "cells": cells,
            "allow_partial": allow_partial,
            "kendall_tau": compute_cell_kendall_tau(cells, metrics, allow_partial=allow_partial)}


# ---- Report assembly -----------------------------------------------------


def read_ledger_lines(ledger):
    """One line per episode id -- the last one -- and how many were dropped.

    ``episodes.jsonl`` is append-only while ``traj/<id>.npz`` is opened "wb"
    and rewritten in place, so a retried episode leaves two lines behind one
    trajectory. Only the last line belongs to the trajectory still on disk;
    keeping the earlier ones counts that episode twice in every rate and in
    every tier mean.
    """
    unique, n_lines = {}, 0
    with (Path(ledger) / "episodes.jsonl").open() as fh:
        for line in fh:
            if line.strip():
                n_lines += 1
                row = json.loads(line)
                unique[row["id"]] = row
    return list(unique.values()), n_lines - len(unique)


def load_ledger_rows(ledger, optimal):
    """Parsed episodes of one ledger, and the duplicate lines it dropped."""
    ledger = Path(ledger)
    lines, dropped = read_ledger_lines(ledger)
    rows = [parse_blocking_episode_metrics(ledger, row, optimal) for row in lines]
    return [row for row in rows if row is not None], dropped


def load_blocking_episode_rows(ledger_dirs, optimal_path=None):
    """Load valid episode rows from one ledger or a set of shard ledgers."""
    optimal = _blocking_optimal(optimal_path)
    rows = []
    for directory in ledger_dirs:
        rows.extend(load_ledger_rows(directory, optimal)[0])
    return rows


def compute_post_evaluation_headline(rows):
    """Compute rates and normalized-path aggregates shared by all reports."""
    def mean(key, subset=rows):
        values = [r[key] for r in subset if r[key] is not None]
        return float(np.mean(values)) if values else None, len(values)
    path, n_path = mean("normalized_path_length")
    time, n_time = mean("normalized_path_traversal_time")
    # Path and time again over the collision-free successes only; contact steps
    # and first-contact speed over the collision episodes only, since the
    # others have neither.
    free = filter_episodes_by_scope(rows, "collision_free_task_success")
    free_path, n_free_path = mean("normalized_path_length", free)
    free_time, n_free_time = mean("normalized_path_traversal_time", free)
    hit = [r for r in rows if r["is_collision"]]
    steps, n_steps = mean("collision_steps", hit)
    speed, n_speed = mean("first_contact_speed", hit)
    rate = lambda key: sum(r.get(key) is True for r in rows) / len(rows) if rows else None
    return {"episodes": len(rows), "task_success_rate": rate("task_success"),
            "collision_free_success_rate": rate("collision_free_success"),
            "normalized_path_length": path, "normalized_path_length_n": n_path,
            "normalized_path_traversal_time": time,
            "normalized_path_traversal_time_n": n_time,
            "collision_free_normalized_path_length": free_path,
            "collision_free_normalized_path_length_n": n_free_path,
            "collision_free_normalized_path_traversal_time": free_time,
            "collision_free_normalized_path_traversal_time_n": n_free_time,
            "collision_steps": steps, "collision_steps_n": n_steps,
            "first_contact_speed": speed, "first_contact_speed_n": n_speed,
            "no_reference": sum(r["planned_path_len_m"] is None for r in rows)}


def build_model_report(rows, source, *, allow_partial=True):
    metrics = [m["name"] for m in POST_EVALUATION_CONFIG["episode_metrics"]]
    return {"summary_mode": "individual_model",
            "task_set_definition": "each model's own eligible tasks; no cross-model matching",
            **source, "headline": compute_post_evaluation_headline(rows),
            "ssi_scopes": {scope: compute_scope_ssi(rows, scope, metrics, allow_partial=allow_partial)
                           for scope in POST_EVALUATION_CONFIG["scopes"]["global"]}}


def summarize_blocking_ledgers(ledger_dirs, optimal_path=None, *, allow_partial=True):
    """Summarize multiple shard ledgers as one model/seed result."""
    paths = [Path(p) for p in ledger_dirs]
    optimal = _blocking_optimal(optimal_path)
    rows, dropped = [], 0
    for path in paths:
        got, n_dropped = load_ledger_rows(path, optimal)
        rows.extend(got)
        dropped += n_dropped
    return build_model_report(rows, {"source_folders": [str(p) for p in paths],
                                     "duplicate_lines_dropped": dropped},
                           allow_partial=allow_partial)


def summarize_blocking_intersection(ledger_dirs, optimal_path=None, *, allow_partial=True):
    """Model SSI on task keys common to every ledger, separately per scope."""
    optimal = _blocking_optimal(optimal_path)
    metrics = [m["name"] for m in POST_EVALUATION_CONFIG["episode_metrics"]]
    ledgers = [Path(p) for p in ledger_dirs]
    per_ledger, dropped = {}, {}
    for ledger in ledgers:
        per_ledger[str(ledger)], dropped[str(ledger)] = load_ledger_rows(ledger, optimal)

    def key(row):
        return row["layout"], row["route"], row["obstacle"]

    scopes = {}
    for scope in POST_EVALUATION_CONFIG["scopes"]["global"]:
        eligible = {name: filter_episodes_by_scope(rows, scope)
                    for name, rows in per_ledger.items()}
        common = set.intersection(*(set(map(key, rows)) for rows in eligible.values())) if eligible else set()
        models = []
        for name, rows in eligible.items():
            chosen = [r for r in rows if key(r) in common]
            summary = compute_scope_ssi(chosen, "all", metrics,
                                      allow_partial=allow_partial)
            # min_distance has only a ``min`` statistic, whereas ratio metrics
            # expose ``mean`` and ``max``. Keep every configured statistic
            # instead of assuming ``mean`` exists for every metric.
            primary = {
                f"{metric}.{stat}": block["tau"]
                for metric, stats in summary["kendall_tau"].items()
                for stat, block in stats.items()
            }
            values = [v for v in primary.values() if v is not None]
            models.append({"ledger": name, "model": Path(name).parent.name,
                           "policy": Path(name).name, "n_episodes": len(chosen),
                           "n_complete_cells": summary["n_complete_cells"],
                           "primary_tau": primary,
                           "ssi": float(np.mean(values)) if values else None,
                           "summary": summary})
        scopes[scope] = {"n_ledgers": len(ledgers),
                         "allow_partial": allow_partial,
                         "n_eligible_per_ledger": {name: len(rows) for name, rows in eligible.items()},
                         "n_intersection_tasks": len(common), "models": models}
    for scope, value in scopes.items():
        value["scope_label"] = SCOPE_LABELS[scope]
        value["task_set_definition"] = (
            "task keys eligible in every compared model; each model is scored on this same intersection")
    return {"summary_mode": "matched_intersection",
            "task_key": ["layout", "route", "obstacle"],
            "ledger_dirs": [str(p) for p in ledgers],
            "duplicate_lines_dropped": dropped, "scopes": scopes}


def _main_blocking():
    parser = argparse.ArgumentParser(description="Summarize one blocking SSI ledger")
    parser.add_argument("ledger_dir", nargs="+"); parser.add_argument("--optimal", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--matched-intersection", action="store_true",
                        help="score each model on task keys common to every supplied ledger")
    args = parser.parse_args()
    with Path(args.out).open("w") as fh:
        out = (summarize_blocking_intersection(args.ledger_dir, args.optimal)
               if args.matched_intersection else summarize_blocking_ledgers(args.ledger_dir, args.optimal))
        json.dump(out, fh, indent=2, allow_nan=False)
        fh.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    _main_blocking()
