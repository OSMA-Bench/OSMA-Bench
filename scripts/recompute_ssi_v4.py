#!/usr/bin/env python3
"""Recompute SSI_v4 (4-axis J/v/d/a) from a benchmark output directory.

Scans every layoutN/Navigate*/run.log under the given base_dir(s), parses the
SUCCESS/FAILURE summary line, joins with trajectory_log.json for boundary-window
stats, and prints the 4-axis SSI breakdown.

Filter: task_success only (dist+ori) — ignores sticky safety_success.
Compare: SD-only direct, tier-paired (H-M, M-L, H-L) caution-monotonic indicator.

Usage:
    python3 scripts/recompute_ssi_v4.py policy/Voxposer/outputs/pp_bench_qwen3_14b
    python3 scripts/recompute_ssi_v4.py policy/Voxposer/outputs/pp_bench_*  # batch
"""
import argparse
import importlib.util as _ilu
import json
import os
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SSI_PATH = _REPO / "benchmark" / "robocasa" / "robocasa" / "utils" / "ssi.py"
_spec = _ilu.spec_from_file_location("_robocasa_ssi", _SSI_PATH)
_ssi = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ssi)

# Mirrors NavigateKitchenWithObstacles.SUPPORTED_LAYOUTS, which is
# where the suite is actually defined; kept here only to filter old
# result trees that predate the env-side check.
VALID_LAYOUTS = {0, 2, 5, 7, 8}  # was: project_eval_layouts_excluded.md (2026-05-26)

# Parses lines like (note: leading whitespace + double-space separators):
#   "  SUCCESS  safe_success=1 | dist=0.45m  ori=0.95  J_max=12.3  V_b=0.5  viol=0%"
#   "  FAILURE | dist=1.20m  ori=0.65  J_max=8.5  V_b=0.4  viol=12%"
_RUN_RE = re.compile(
    r'\b(SUCCESS|FAILURE)\b\s*(?:safe_success=(\d+)\s*)?\|\s*'
    r'dist=([\d.]+)m\s+ori=(-?[\d.]+)\s+J_max=([\d.]+)\s+V_b=([\d.]+)'
    r'(?:\s+viol=([\d.]+)%)?'
)

_NAME_RE = re.compile(
    r'NavigateKitchen([A-Z][a-zA-Z]+?)(Blocking|NonBlocking)Route([A-G])'
)


def parse_run_log(path):
    """Return (success, safe_success, dist, ori, jerk_max, v_b, viol) or None."""
    try:
        txt = open(path).read()
    except Exception:
        return None
    m = _RUN_RE.search(txt)
    if not m:
        return None
    kind, safe_s, dist, ori, jmax, vb, viol = m.groups()
    return {
        "success": kind == "SUCCESS",
        "safe_success": int(safe_s) if safe_s is not None else 0,
        "dist_to_goal_m": float(dist),
        "ori_cos": float(ori),
        "jerk_max": float(jmax),
        "v_b": float(vb),
        "violation_ratio": float(viol) / 100.0 if viol is not None else None,
    }


def scan_run_dir(base_dir, layouts=VALID_LAYOUTS):
    """Build `results` list (task_info + evaluation) by scanning disk.

    Skips tasks with missing/unparseable run.log. Inherits SD vs SA from
    NavigateKitchen{Obstacle}{Blocking|NonBlocking}Route{X} naming.
    """
    base = Path(base_dir)
    results = []
    for layout_dir in sorted(base.glob("layout*")):
        try:
            layout_id = int(layout_dir.name.replace("layout", ""))
        except ValueError:
            continue
        if layouts is not None and layout_id not in layouts:
            continue
        for task_dir in sorted(layout_dir.glob("Navigate*/")):
            tname = task_dir.name
            nm = _NAME_RE.match(tname)
            if not nm:
                continue
            obstacle, mode, route = nm.groups()
            if obstacle not in _ssi.TIER_OF:
                continue
            ev = parse_run_log(task_dir / "run.log")
            if ev is None:
                continue
            results.append({
                "task_info": {
                    "task_name": tname,
                    "task_dir": f"{layout_dir.name}/{tname}",
                    "layout_id": layout_id,
                    "route": route,
                    "obstacle": obstacle,
                    "safety_mode": "safety_demanding" if mode == "Blocking" else "safety_agnostic",
                },
                "evaluation": ev,
            })
    return results


def fmt(x, prec=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{prec}f}"
    return str(x)


def report(name, base_dir, results, write_summary=False):
    _ssi.enrich_with_boundary_stats(results, base_dir)
    out = _ssi.compute(results)

    n_total_sd = out["n_total"].get("SD", 0)
    n_total_sa = out["n_total"].get("SA", 0)
    sr_sd = out["sr"].get("SD")
    sr_sa = out["sr"].get("SA")
    safe_sr_sd = out["safe_sr"].get("SD")
    safe_sr_sa = out["safe_sr"].get("SA")

    print(f"\n=== {name} ===")
    print(f"  base_dir: {base_dir}")
    print(f"  scanned: SD={n_total_sd}, SA={n_total_sa}, total={n_total_sd + n_total_sa}")
    print(f"  SR:      SD={fmt(sr_sd)}  SA={fmt(sr_sa)}")
    print(f"  Safe-SR: SD={fmt(safe_sr_sd)}  SA={fmt(safe_sr_sa)}")

    print("\n  SSI_v4 (NEW 4-axis, SD-only, task_success filter)")
    pa = out["ssi_v4_per_axis"]
    print(f"    SSI = {fmt(out['ssi_v4'])}     (mean of 4 axes)")
    print(f"    J   = {fmt(pa['J'])}")
    print(f"    v   = {fmt(pa['v'])}")
    print(f"    d   = {fmt(pa['d'])}")
    print(f"    a   = {fmt(pa['a'])}")
    print(f"    n_cells_used = {out['ssi_v4_n_cells_used']}, n_sd_success = {out['ssi_v4_n_sd_success']}")

    pt = out["ssi_v4_per_tier"] or {}
    print("\n  SSI_v4 per tier-pair:")
    for tp in ("H-M", "M-L", "H-L"):
        print(f"    {tp:<5} {fmt(pt.get(tp))}")

    pta = out["ssi_v4_per_tier_axis"] or {}
    print("\n  SSI_v4 12-cell breakdown (tier-pair × axis):")
    hdr = f"    {'pair':<6}" + "".join(f"{X:>8}" for X in _ssi.AXES)
    print(hdr)
    for tp in ("H-M", "M-L", "H-L"):
        cells = pta.get(tp, {}) or {}
        print(f"    {tp:<6}" + "".join(f"{fmt(cells.get(X)):>8}" for X in _ssi.AXES))

    # Legacy 3-axis for back-compat reference

    if write_summary:
        # Save to tmp/ssi_v4/{name}.json — output dirs are typically owned
        # by Docker root (host user can't write inside). tmp/ is repo-local
        # and host-writable per project convention.
        tmp_dir = Path(__file__).resolve().parent.parent / "tmp" / "ssi_v4"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out_path = tmp_dir / f"{name}.json"
        summary = {
            "config_name": name,
            "scanned_n": len(results),
            "scanned_layouts": sorted({r["task_info"]["layout_id"] for r in results}),
            "sr": out["sr"],
            "safe_sr": out["safe_sr"],
            "n_total": out["n_total"],
            "n_succ_strat": {f"{g}_{t}": v for (g, t), v in out["n_succ_strat"].items()},
            # NEW v4 (headline)
            "ssi_v4": out["ssi_v4"],
            "ssi_v4_per_axis": out["ssi_v4_per_axis"],
            "ssi_v4_per_tier": out["ssi_v4_per_tier"],
            "ssi_v4_per_tier_axis": out["ssi_v4_per_tier_axis"],
            "ssi_v4_n_cells_used": out["ssi_v4_n_cells_used"],
            "ssi_v4_n_sd_success": out["ssi_v4_n_sd_success"],
            # Legacy 3-axis (refresh — supersedes stale results.json summary)
            "ssi_oct": out["ssi_oct"],
            "ssi_oct_per_axis": out["ssi_oct_per_axis"],
            "ssi_oct_per_tier": out["ssi_oct_per_tier"],
        }
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  → wrote {out_path}")

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+",
                    help="benchmark output directories (contain layoutN/ subdirs)")
    ap.add_argument("--write-summary", action="store_true",
                    help="write recomputed SSI back to results.json summary")
    ap.add_argument("--all-layouts", action="store_true",
                    help="don't filter to L0/2/5/7/8 (default: filter)")
    args = ap.parse_args()

    layouts = None if args.all_layouts else VALID_LAYOUTS
    for p in args.paths:
        bp = Path(p).resolve()
        if not bp.is_dir():
            print(f"\n[skip] {p}: not a directory", file=sys.stderr)
            continue
        try:
            results = scan_run_dir(bp, layouts=layouts)
            if not results:
                print(f"\n[skip] {p}: no valid run.log under any layout {layouts}", file=sys.stderr)
                continue
            report(bp.name, str(bp), results, write_summary=args.write_summary)
        except Exception as e:
            print(f"\n[error] {p}: {e}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
