"""Aggregate the per-layout validation CSVs into one stability verdict.

The runner records, per episode, how far the obstacle moved from its anchored
pose: xy_drift (slid), z_drift_up (climbed/bounced), z_drift_down (sank or fell
from a spawn above its rest height), plus its own pop_out flag.

    python scripts/nav_sweep_stability.py [figures/nav_sweep]
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

XY_TOL, Z_UP_TOL, Z_DOWN_TOL = 0.05, 0.05, 0.10


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "figures/nav_sweep"
    paths = sorted(glob.glob(os.path.join(root, "stability", "validation_*.csv")))
    if not paths:
        print("no validation CSVs under", root)
        return 1

    rows = []
    for p in paths:
        layout = os.path.basename(p)[len("validation_"):-len(".csv")]
        for r in csv.DictReader(open(p)):
            r["_layout"] = layout
            rows.append(r)

    bad_status = [r for r in rows if r["status"] != "success"]
    popped = [r for r in rows if r["pop_out"] not in ("0", "", "False")]
    over = [r for r in rows
            if num(r["xy_drift_max"]) > XY_TOL
            or num(r["z_drift_up_max"]) > Z_UP_TOL
            or num(r["z_drift_down_max"]) > Z_DOWN_TOL]

    print(f"episodes            {len(rows)}   ({len(paths)} layouts)")
    print(f"status != success   {len(bad_status)}")
    print(f"pop_out flagged     {len(popped)}")
    print(f"over tolerance      {len(over)}   "
          f"(xy>{XY_TOL} or z_up>{Z_UP_TOL} or z_down>{Z_DOWN_TOL} m)")

    # worst episode per obstacle, so a single bad cell cannot hide in the mean
    worst = defaultdict(lambda: (0.0, 0.0, 0.0, ""))
    for r in rows:
        obs = r["obstacle"] or "?"
        cur = (num(r["xy_drift_max"]), num(r["z_drift_up_max"]),
               num(r["z_drift_down_max"]))
        if max(cur) > max(worst[obs][:3]):
            worst[obs] = cur + (f"{r['env_name']} / {r['_layout']}",)

    print(f"\nworst episode per obstacle (cm)")
    print(f"{'obstacle':16s} {'xy':>7} {'z_up':>7} {'z_down':>7}  cell")
    for obs in sorted(worst, key=lambda o: -max(worst[o][:3])):
        xy, up, dn, cell = worst[obs]
        mark = "  <-- over tolerance" if (xy > XY_TOL or up > Z_UP_TOL
                                          or dn > Z_DOWN_TOL) else ""
        print(f"{obs:16s} {xy*100:7.2f} {up*100:7.2f} {dn*100:7.2f}  {cell}{mark}")

    for label, group in (("status", bad_status), ("pop_out", popped),
                         ("tolerance", over)):
        if group:
            print(f"\n{label} failures:")
            for r in group[:20]:
                print(f"  {r['env_name']:52s} {r['_layout']:16s} "
                      f"xy={num(r['xy_drift_max'])*100:.2f} "
                      f"up={num(r['z_drift_up_max'])*100:.2f} "
                      f"down={num(r['z_drift_down_max'])*100:.2f} cm "
                      f"{r['error'][:60]}")

    verdict = not (bad_status or popped or over)
    print(f"\nVERDICT: {'all episodes stable' if verdict else 'unstable cells present'}")
    return 0 if verdict else 2


if __name__ == "__main__":
    sys.exit(main())
