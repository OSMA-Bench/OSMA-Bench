"""Roll the ledgers under one outputs folder up into the headline numbers.

    python3 -m robocasa.metrics.summarize <folder>

`<folder>` is a run directory — an absolute path, or a name under
`ROBOCASA_OUTPUTS`. Every ledger beneath it is found and read; a sweep that
wrote one ledger per task and a single-episode check both work, and nothing
has to be named in advance.

This reads the ledgers that `robocasa.logging.log` writes DURING the run,
not the saved logs. `parse_run_logs.py` covers the other direction — parsing run.log
after the fact — and the two are kept separate because they fail differently:
extraction is a regex against text that changed shape over time, this is a
jsonl the runner wrote itself.

Three numbers come out:

    task_success_rate            reached the goal pose
    collision_free_success_rate  reached it without touching the obstacle
    normalized_path_length       driven path length / optimal path length
    normalized_path_traversal_time  driven traversal time / optimal time

The optimal lengths come from the non-blocking sweep, which planned one path
per (layout, route) — the obstacle does not enter the reference, so a Blocking
episode is measured against the unobstructed plan and pays for every metre it
spends going around.

    Both references are selected by the exact (layout, route) cell. Values above
    1 indicate a longer/slower rollout than the unobstructed reference.
"""
import argparse
import json
import os
import sys
import re
import importlib.util
from pathlib import Path

import numpy as np


def _ssi_module():
    """Load the canonical SSI implementation without importing robocasa."""
    path = Path(__file__).with_name("ssi.py")
    spec = importlib.util.spec_from_file_location("robocasa_post_evaluation_ssi", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Same default as parse_run_logs.py, and the same env var. Two modules in one package
# disagreeing about where runs live is how a report ends up empty.
OUT_ROOT = os.environ.get("ROBOCASA_OUTPUTS", os.path.join(
    os.getcwd(), "policy", "Voxposer", "outputs"))

# The A* reference, written by the non-blocking optimal-path sweep.
_DEFAULT_OPTIMAL = Path(__file__).with_name("nonblocking_optimal_paths") / "path_length_time.json"
OPTIMAL = os.environ.get("ROBOCASA_OPTIMAL_PATHS", str(_DEFAULT_OPTIMAL))


def find_ledgers(root):
    """Every directory under `root` that holds an episodes.jsonl, `root` included."""
    root = Path(root)
    hits = sorted({p.parent for p in root.rglob("episodes.jsonl")})
    return hits


def load_planned(path):
    """(layout_name, route) -> optimal path length and traversal time."""
    cells = json.load(open(path))["cells"]
    return {(c.get("layout", c["layout_name"]), c["route"]): {
        "path_length_m": c["planned_path_len_m"],
        "time_s": c.get("travel_time_s"),
    } for c in cells}


def route_of(task):
    """`NavigateKitchenDogBlockingRouteA` -> `RouteA`, matching the optimal json."""
    return "Route" + task.rsplit("Route", 1)[1] if "Route" in (task or "") else None


def executed_metrics(ledger, ep):
    """(metres, source) driven by this episode.

    The episode's own evaluation.json is preferred: it sums every control step.
    The ledger's trajectory holds one sample in `log_interval` steps, so the
    length recomputed from it runs a few percent short — usable, but the source
    is reported so two runs are never compared across the two.
    """
    run = ledger / (ep.get("run_dir") or "") / "evaluation.json"
    if ep.get("run_dir") and run.exists():
        value = json.load(open(run)).get("path_length_m")
        if value is not None:
            return float(value), float(json.load(open(run)).get("duration_s") or 0.0), "control-step"
    traj = ledger / "traj" / f"{ep['id']}.npz"
    if traj.exists():
        pos = np.load(traj)["pos_xy"]
        return (float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))),
                float(np.asarray(np.load(traj)["t"])[-1]), "sampled")
    return None, None, None


def read_ledger(ledger, planned):
    """One record per episode, with the normalized path attached where it exists.

    `episodes.jsonl` is append-only while `traj/<id>.npz` is opened "wb" and
    rewritten in place, so a retried episode leaves two lines behind one
    trajectory. The last line is the one that matches the stored trajectory and
    the surviving `run_dir`, so it wins — the same rule `ssi.read_ledger_lines`
    applies, which is what keeps this table and the SSI report on one `n`.
    """
    unique = {}
    for line in open(ledger / "episodes.jsonl"):
        if not line.strip():
            continue
        ep = json.loads(line)
        unique[ep["id"]] = ep
    rows = []
    for ep in unique.values():
        ref = planned.get((ep.get("layout"), route_of(ep.get("task"))))
        actual, actual_time, source = executed_metrics(ledger, ep)
        ref_length = ref["path_length_m"] if ref else None
        ref_time = ref["time_s"] if ref else None
        rows.append({
            **ep,
            "ledger": str(ledger),
            "planned_path_len_m": ref_length,
            "planned_travel_time_s": ref_time,
            "executed_path_len_m": actual,
            "executed_travel_time_s": actual_time,
            "path_length_source": source,
            "normalized_path_length": (actual / ref_length) if ref_length and actual else None,
            "normalized_path_traversal_time": (actual_time / ref_time)
                if ref_time and actual_time else None,
        })
    return rows


def aggregate(rows):
    """The three rates over a list of episode records."""
    def rate(key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return (sum(bool(x) for x in v) / len(v), len(v)) if v else (None, 0)

    npath = [r["normalized_path_length"] for r in rows if r["normalized_path_length"] is not None]
    ntime = [r["normalized_path_traversal_time"] for r in rows
             if r["normalized_path_traversal_time"] is not None]
    tsr, n_tsr = rate("task_success")
    csr, n_csr = rate("collision_free_success")
    return {
        "episodes": len(rows),
        "task_success_rate": tsr, "task_success_decided": n_tsr,
        "collision_free_success_rate": csr, "collision_free_success_decided": n_csr,
        "normalized_path_length": float(np.mean(npath)) if npath else None,
        "normalized_path_length_n": len(npath),
        "normalized_path_traversal_time": float(np.mean(ntime)) if ntime else None,
        "normalized_path_traversal_time_n": len(ntime),
        # A missing reference is not a zero. It means the optimal sweep never
        # planned this (layout, route), and saying so beats a silent gap.
        "no_reference": sum(1 for r in rows if r["planned_path_len_m"] is None),
    }


def summarize_post_evaluation(ledger_dirs, optimal_path=None, *, matched_intersection=False,
                              comparison=None, scopes=None, allow_partial=True,
                              aggregate_seeds=False):
    """Return the canonical post-evaluation report.

    SSI and matched-intersection logic remain in :mod:`metrics.ssi`; this
    function is the single public post-evaluation entry point used by both
    the CLI and validation helpers.
    """
    ssi = _ssi_module()
    paths = []
    for value in ledger_dirs:
        path = Path(value)
        if (path / "episodes.jsonl").exists():
            paths.append(str(path))
        else:
            paths.extend(str(found) for found in find_ledgers(path))
    if not paths:
        raise ValueError("no ledger found under the supplied output folders")
    optimal_path = optimal_path or OPTIMAL
    if not Path(optimal_path).exists():
        print(f"warning: optimal-path json not found: {optimal_path}; "
              "normalized path metrics will be n/a", file=sys.stderr)
        optimal_path = None
    comparison = comparison or ("matched_intersection" if matched_intersection else "individual")
    if comparison not in {"individual", "matched_intersection"}:
        raise ValueError("comparison must be individual or matched_intersection")
    if aggregate_seeds:
        groups = {}
        for p in paths:
            m = re.match(r"(.+)_seed(\d+)_s\d+_ledger$", Path(p).name)
            if m:
                groups.setdefault((m.group(1),), {}).setdefault(m.group(2), []).append(p)
        models = []
        for (model,), seeds in groups.items():
            per_seed = [ssi.summarize_blocking_ledgers(v, optimal_path, allow_partial=allow_partial)
                        for _, v in sorted(seeds.items(), key=lambda kv: int(kv[0]))]
            def stats(values):
                values = [float(v) for v in values if v is not None]
                return {"mean": float(np.mean(values)) if values else None,
                        "std": float(np.std(values)) if len(values) > 1 else 0.0,
                        "n_seeds": len(values)}
            headline = {k: stats([r["headline"].get(k) for r in per_seed])
                        for k in ("task_success_rate", "collision_free_success_rate",
                                  "normalized_path_length", "normalized_path_traversal_time")}
            ssi_stats = {}
            for scope in per_seed[0]["ssi_scopes"] if per_seed else []:
                ssi_stats[scope] = {}
                for metric, block in per_seed[0]["ssi_scopes"][scope]["kendall_tau"].items():
                    ssi_stats[scope][metric] = {stat: stats([r["ssi_scopes"][scope]["kendall_tau"][metric][stat]["tau"] for r in per_seed])
                                                for stat in block}
            models.append({"model": model, "policy": ("pivot" if "_piv" in model else "voxposer"), "seeds": sorted(seeds),
                           "per_seed": per_seed, "headline": headline, "ssi": ssi_stats})
        report = {"summary_mode": "seed_aggregate", "comparison": comparison,
                  "models": models}
    elif comparison == "matched_intersection":
        report = ssi.summarize_blocking_intersection(
            paths, optimal_path, allow_partial=allow_partial)
    elif len(paths) == 1:
        report = ssi.summarize_blocking_ledgers(paths, optimal_path, allow_partial=allow_partial)
    else:
        report = {"summary_mode": "individual_models", "comparison": comparison,
                  "models": [ssi.summarize_blocking_ledgers([p], optimal_path, allow_partial=allow_partial) for p in paths]}
    if scopes:
        allowed = set(scopes)
        key = "scopes" if report.get("summary_mode") == "matched_intersection" else "ssi_scopes"
        if key in report:
            report[key] = {k: v for k, v in report[key].items() if k in allowed}
        elif report.get("summary_mode") == "individual_models":
            for model in report["models"]:
                model["ssi_scopes"] = {k: v for k, v in model["ssi_scopes"].items() if k in allowed}
    return report


def check_post_evaluation(report):
    """Validate the stable post-evaluation report contract."""
    headline = report.get("headline", {})
    scopes = report.get("ssi_scopes", report.get("scopes", {}))
    if report.get("summary_mode") == "matched_intersection":
        if not scopes:
            raise ValueError("matched post-evaluation report has no scopes")
        return True
    if report.get("summary_mode") == "individual_models":
        return all(report_ok.get("ssi_scopes") for report_ok in report.get("models", []))
    if report.get("summary_mode") == "seed_aggregate":
        return all(model.get("headline") and model.get("ssi")
                   for model in report.get("models", []))
    required = {"task_success_rate", "collision_free_success_rate",
                "normalized_path_length", "normalized_path_traversal_time"}
    missing = sorted(required - headline.keys())
    if missing:
        raise ValueError(f"post-evaluation headline missing: {', '.join(missing)}")
    if not scopes:
        raise ValueError("post-evaluation report has no SSI scopes")
    return True


def fmt(x, width=7, pct=False):
    if x is None:
        return f"{'n/a':>{width}}"
    return f"{x:{width}.1%}" if pct else f"{x:{width}.4f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder",
                    help="run directory: an absolute path, or a name under "
                         "ROBOCASA_OUTPUTS")
    ap.add_argument("--optimal", default=OPTIMAL,
                    help="optimal-path json (env: ROBOCASA_OPTIMAL_PATHS)")
    ap.add_argument("--episodes", action="store_true",
                    help="also print one line per episode")
    ap.add_argument("--json", default="",
                    help="also write the full result to this path")
    a = ap.parse_args()

    root = Path(a.folder)
    if not root.is_absolute() and not root.exists():
        root = Path(OUT_ROOT) / a.folder
    if not root.exists():
        sys.exit(f"no such folder: {root}")
    if os.path.exists(a.optimal):
        planned = load_planned(a.optimal)
    else:
        planned = {}
        print(f"warning: optimal-path json not found: {a.optimal}; "
              "normalized path metrics will be n/a", file=sys.stderr)
    ledgers = find_ledgers(root)
    if not ledgers:
        sys.exit(f"no episodes.jsonl under {root}")

    per_ledger, rows = {}, []
    for ledger in ledgers:
        got = read_ledger(ledger, planned)
        if not got:
            continue
        per_ledger[str(ledger)] = aggregate(got)
        rows += got

    print(f"folder : {root}")
    print(f"optimal: {a.optimal}  ({len(planned)} cells)")
    print(f"ledgers: {len(per_ledger)}  episodes: {len(rows)}")
    print()
    hdr = (f"{'ledger':<58}{'n':>5}{'task_succ':>11}{'coll_free':>11}"
           f"{'norm_path':>11}")
    print(hdr)
    print("-" * len(hdr))
    for name, s in per_ledger.items():
        short = name[-57:] if len(name) > 57 else name
        print(f"{short:<58}{s['episodes']:>5}"
              f"{fmt(s['task_success_rate'], 11, pct=True)}"
              f"{fmt(s['collision_free_success_rate'], 11, pct=True)}"
              f"{fmt(s['normalized_path_length'], 11)}"
              f"{fmt(s['normalized_path_traversal_time'], 11)}")
    total = aggregate(rows)
    print("-" * len(hdr))
    print(f"{'TOTAL':<58}{total['episodes']:>5}"
          f"{fmt(total['task_success_rate'], 11, pct=True)}"
          f"{fmt(total['collision_free_success_rate'], 11, pct=True)}"
          f"{fmt(total['normalized_path_length'], 11)}"
          f"{fmt(total['normalized_path_traversal_time'], 11)}")

    print()
    print(f"task_success_rate           {fmt(total['task_success_rate'], 8, pct=True)}"
          f"   (decided {total['task_success_decided']}/{total['episodes']})")
    print(f"collision_free_success_rate {fmt(total['collision_free_success_rate'], 8, pct=True)}"
          f"   (decided {total['collision_free_success_decided']}/{total['episodes']})")
    print(f"normalized_path_length      {fmt(total['normalized_path_length'], 8)}"
          f"   (actual/planned over {total['normalized_path_length_n']}/{total['episodes']})")
    print(f"normalized_path_traversal_time {fmt(total['normalized_path_traversal_time'], 8)}"
          f"   (actual/optimal over {total['normalized_path_traversal_time_n']}/{total['episodes']})")
    if total["no_reference"]:
        print(f"  {total['no_reference']} episode(s) have no (layout, route) in the "
              f"optimal json and are left out of normalized path metrics")

    if a.episodes:
        print()
        for r in rows:
            npath = fmt(r["normalized_path_length"], 7)
            print(f"  {r['task']:<46} {r['layout']:<18} "
                  f"task={str(r['task_success']):<5} "
                  f"coll_free={str(r['collision_free_success']):<5} "
                  f"planned={r['planned_path_len_m']} actual={r['executed_path_len_m']} "
                  f"[{r['path_length_source']}] norm={npath}")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump({"folder": str(root), "optimal": a.optimal,
                       "total": total, "per_ledger": per_ledger,
                       "episodes": rows}, fh, indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
