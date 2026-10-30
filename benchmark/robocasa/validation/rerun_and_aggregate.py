"""Rerun log_initial_violations.py per layout, aggregate, and run the
destination-reachability check on the result.

Pipeline (each stage independently skippable):
    1. Per-layout rerun  (--skip-rerun / --aggregate-only)
    2. Aggregate per-layout CSVs into <root>/violations_all_layouts.csv
       and split into violations_only.csv / non_violations.csv
    3. Copy initial-state images of violating rows into
       <root>/violation_images/      (--skip-images)
    4. Destination-reachability check on violations_only.csv and
       non_violations.csv             (--skip-destination)

Layout subdir convention matches initial_violations_1st/aggregate.py:
    <root>/<LAYOUT_NAME>/violations.csv

Examples:
    # Full pipeline: rerun all layouts, aggregate, images, destination check
    python rerun_and_aggregate.py --root initial_violations

    # Multi-GPU rerun + parallel destination check (workers default to len(--gpus))
    python rerun_and_aggregate.py --gpus 0 1 2 3

    # Skip rerun; only aggregate + filter + destination check
    python rerun_and_aggregate.py --root initial_violations --aggregate-only

    # Just aggregate / filter, skip the destination check
    python rerun_and_aggregate.py --aggregate-only --skip-destination
"""
import argparse
import csv
import glob
import os
import queue
import shutil
import subprocess
import sys
import threading
from collections import defaultdict

from robocasa.models.scenes.scene_registry import LAYOUT_GROUPS_TO_IDS, LayoutType

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_SCRIPT = os.path.join(HERE, "log_initial_violations.py")
DEST_SCRIPT = os.path.join(HERE, "check_destination_reachable.py")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="initial_violations",
                   help="Root output dir; per-layout CSVs land in <root>/<LAYOUT>/")
    p.add_argument("--layouts", nargs="+", default=None,
                   help="Layout names to rerun (default: all)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--style", default="MODERN_1")
    p.add_argument("--filter", default=None,
                   help="Substring filter passed through to log_initial_violations.py")
    p.add_argument("--exclude", default=None,
                   help="Substring exclude passed through to log_initial_violations.py")
    p.add_argument("--no_image", action="store_true")
    p.add_argument("--gpu_id", type=int, default=0,
                   help="Single-GPU id (used when --gpus is not set)")
    p.add_argument("--gpus", type=int, nargs="+", default=None,
                   help="GPU ids for parallel execution; one layout per GPU "
                        "concurrently (e.g. --gpus 0 1 2 3). Also sets the "
                        "default destination-check worker count.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip layouts whose violations.csv already exists")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Skip rerun; only aggregate existing per-layout CSVs")
    p.add_argument("--skip-images", action="store_true",
                   help="Don't copy initial-state images of violating rows")
    p.add_argument("--skip-destination", action="store_true",
                   help="Don't run the destination-reachability check")
    p.add_argument("--dest-workers", type=int, default=None,
                   help="Number of parallel workers for the destination check "
                        "(default: len(--gpus) or 1)")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Stage 1 — per-layout rerun
# -----------------------------------------------------------------------------

def _build_cmd(layout_name, out_dir, args, gpu_id):
    cmd = [
        sys.executable, LOG_SCRIPT,
        "--layout", layout_name,
        "--out_dir", out_dir,
        "--style", args.style,
        "--seeds", *map(str, args.seeds),
        "--gpu_id", str(gpu_id),
    ]
    if args.filter:
        cmd += ["--filter", args.filter]
    if args.exclude:
        cmd += ["--exclude", args.exclude]
    if args.no_image:
        cmd += ["--no_image"]
    return cmd


def rerun_one(layout_name, args, gpu_id=None, prefix=""):
    if gpu_id is None:
        gpu_id = args.gpu_id
    out_dir = os.path.join(args.root, layout_name)
    csv_path = os.path.join(out_dir, "violations.csv")
    if args.skip_existing and os.path.exists(csv_path):
        print(f"{prefix}[skip] {layout_name}: {csv_path} exists")
        return
    cmd = _build_cmd(layout_name, out_dir, args, gpu_id)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"{prefix}[run ] gpu={gpu_id} {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def rerun_parallel(layout_names, args):
    q: "queue.Queue[str]" = queue.Queue()
    for L in layout_names:
        q.put(L)
    errors: list[tuple[str, BaseException]] = []
    err_lock = threading.Lock()

    def worker(gpu_id):
        prefix = f"[gpu{gpu_id}] "
        while True:
            try:
                L = q.get_nowait()
            except queue.Empty:
                return
            try:
                rerun_one(L, args, gpu_id=gpu_id, prefix=prefix)
            except BaseException as e:
                with err_lock:
                    errors.append((L, e))
                print(f"{prefix}[err ] {L}: {e!r}")
            finally:
                q.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True)
               for g in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError(
            f"{len(errors)} layout(s) failed: "
            + ", ".join(L for L, _ in errors)
        )


# -----------------------------------------------------------------------------
# Stage 2 — aggregate + filter
# -----------------------------------------------------------------------------

def aggregate(root):
    """Merge per-layout CSVs into violations_all_layouts.csv and split it
    into violations_only.csv / non_violations.csv. Returns (violations, non,
    header) for downstream stages.
    """
    merged = []
    header = None
    for csv_path in sorted(glob.glob(os.path.join(root, "*", "violations.csv"))):
        with open(csv_path) as f:
            r = csv.DictReader(f)
            if header is None:
                header = r.fieldnames
            for row in r:
                merged.append(row)
    if not merged:
        print(f"[warn] no per-layout CSVs found under {root}")
        return [], [], None

    all_csv = os.path.join(root, "violations_all_layouts.csv")
    only_csv = os.path.join(root, "violations_only.csv")
    nv_csv = os.path.join(root, "non_violations.csv")
    viol = [r for r in merged if r.get("boundary_violated") == "1"]
    nv = [r for r in merged if r.get("boundary_violated") == "0"]
    for path, rows in [(all_csv, merged), (only_csv, viol), (nv_csv, nv)]:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)

    by_layout = defaultdict(lambda: [0, 0, 0])
    for r in merged:
        by_layout[r["layout"]][0] += 1
        by_layout[r["layout"]][1] += int(r["boundary_violated"] or 0)
        by_layout[r["layout"]][2] += int(r["any_contact"] or 0)

    by_env = defaultdict(lambda: [0, 0])
    for r in merged:
        by_env[r["env_name"]][0] += 1
        by_env[r["env_name"]][1] += int(r["boundary_violated"] or 0)

    by_combo = defaultdict(lambda: [0, 0])
    for r in merged:
        key = (r["obstacle_kind"], r["route"], r["blocking_mode"])
        by_combo[key][0] += 1
        by_combo[key][1] += int(r["boundary_violated"] or 0)

    print("=== per layout (total, violations, contacts) ===")
    for L, v in sorted(by_layout.items()):
        print(f"  {L:18s}  total={v[0]:3d}  viol={v[1]:3d}  contact={v[2]:3d}")

    print("\n=== tasks that violate in >= 5 layouts ===")
    for env, v in sorted(by_env.items(), key=lambda kv: (-kv[1][1], kv[0])):
        if v[1] >= 5:
            print(f"  {env:55s}  layouts={v[0]:2d}  violations={v[1]:2d}")

    print("\n=== (obstacle, route, mode) violation rate (top 25) ===")
    ranked = sorted(by_combo.items(),
                    key=lambda kv: (-kv[1][1] / max(1, kv[1][0]), -kv[1][1]))
    for (obs, rt, bm), v in ranked[:25]:
        rate = v[1] / max(1, v[0])
        print(f"  {obs:14s} {rt:7s} {bm:11s}  viol={v[1]:2d}/{v[0]:2d}  rate={rate:.2f}")

    print(f"\nMerged: {all_csv} ({len(merged)})  "
          f"violations: {only_csv} ({len(viol)})  "
          f"non: {nv_csv} ({len(nv)})")
    return viol, nv, header


# -----------------------------------------------------------------------------
# Stage 3 — copy violation images
# -----------------------------------------------------------------------------

def copy_violation_images(root, viol_rows):
    img_dir = os.path.join(root, "violation_images")
    shutil.rmtree(img_dir, ignore_errors=True)
    os.makedirs(img_dir, exist_ok=True)
    copied = missed = 0
    for r in viol_rows:
        base = f"{r['env_name']}_{r['layout']}_seed{r['seed']}.png"
        # Local layout matches log_initial_violations.py: <root>/<LAYOUT>/images/
        src = os.path.join(root, r["layout"], "images", base)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(img_dir, base))
            copied += 1
        else:
            missed += 1
    print(f"[images] {img_dir}: copied={copied}  missing={missed}")


# -----------------------------------------------------------------------------
# Stage 4 — destination-reachability check (parallel by row chunking)
# -----------------------------------------------------------------------------

def _split_csv_into_chunks(in_csv, n_chunks, chunk_dir):
    rows = list(csv.DictReader(open(in_csv)))
    if not rows:
        return [], []
    fields = list(rows[0].keys())
    os.makedirs(chunk_dir, exist_ok=True)
    chunks = [[] for _ in range(n_chunks)]
    for i, r in enumerate(rows):
        chunks[i % n_chunks].append(r)
    in_paths, out_paths = [], []
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        ic = os.path.join(chunk_dir, f"in_{i:02d}.csv")
        oc = os.path.join(chunk_dir, f"out_{i:02d}.csv")
        with open(ic, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(chunk)
        in_paths.append(ic)
        out_paths.append(oc)
    return in_paths, out_paths


def destination_check(in_csv, out_csv, args, label):
    """Run check_destination_reachable.py on `in_csv`, writing to `out_csv`.
    Parallelized by sharding rows across `args.dest_workers` subprocesses.
    """
    if not os.path.exists(in_csv):
        print(f"[dest] skip {label}: {in_csv} missing")
        return
    n_rows = sum(1 for _ in csv.DictReader(open(in_csv)))
    if n_rows == 0:
        print(f"[dest] skip {label}: {in_csv} empty")
        return

    n_workers = args.dest_workers or (len(args.gpus) if args.gpus else 1)
    n_workers = max(1, min(n_workers, n_rows))
    gpus = args.gpus or [args.gpu_id]
    print(f"[dest] {label}: {n_rows} rows, {n_workers} workers, gpus={gpus}")

    if n_workers == 1:
        cmd = [sys.executable, DEST_SCRIPT,
               "--in", in_csv, "--out", out_csv,
               "--gpu_id", str(gpus[0])]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
        subprocess.run(cmd, check=True, env=env)
        return

    chunk_dir = os.path.join(args.root, f"_dest_chunks_{label}")
    shutil.rmtree(chunk_dir, ignore_errors=True)
    in_paths, out_paths = _split_csv_into_chunks(in_csv, n_workers, chunk_dir)

    procs = []
    for i, (ic, oc) in enumerate(zip(in_paths, out_paths)):
        gpu = gpus[i % len(gpus)]
        cmd = [sys.executable, DEST_SCRIPT,
               "--in", ic, "--out", oc, "--gpu_id", str(gpu)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_path = os.path.join(chunk_dir, f"worker_{i:02d}.log")
        log_f = open(log_path, "w")
        print(f"[dest][{label}] worker {i} gpu={gpu} -> {oc}  (log: {log_path})")
        procs.append((subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT),
                      log_f, i))

    rcs = []
    for p, log_f, i in procs:
        rc = p.wait()
        log_f.close()
        rcs.append(rc)
    bad = [i for i, rc in zip(range(len(rcs)), rcs) if rc != 0]
    if bad:
        raise RuntimeError(
            f"[dest] {label}: workers failed: "
            + ", ".join(f"#{i}(rc={rcs[i]})" for i in bad)
            + f"; chunk_dir={chunk_dir}"
        )

    out_fields = None
    out_rows = []
    for oc in out_paths:
        with open(oc) as f:
            r = csv.DictReader(f)
            if out_fields is None:
                out_fields = r.fieldnames
            for row in r:
                out_rows.append(row)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(out_rows)
    shutil.rmtree(chunk_dir, ignore_errors=True)

    n_reach = sum(1 for r in out_rows if r.get("reachable") == "1")
    n_unreach = sum(1 for r in out_rows if r.get("reachable") == "0")
    n_err = sum(1 for r in out_rows if r.get("status") == "error")
    print(f"[dest] {label}: wrote {out_csv}  "
          f"rows={len(out_rows)}  reachable={n_reach}  "
          f"unreachable={n_unreach}  errors={n_err}")


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    if not args.aggregate_only:
        if args.layouts:
            layout_names = args.layouts
        else:
            layout_names = [LayoutType(lid).name
                            for lid in LAYOUT_GROUPS_TO_IDS[LayoutType.ALL]]
        os.makedirs(args.root, exist_ok=True)
        if args.gpus:
            rerun_parallel(layout_names, args)
        else:
            for L in layout_names:
                rerun_one(L, args)

    viol, _nv, _hdr = aggregate(args.root)
    if viol is None:
        return

    if not args.skip_images:
        copy_violation_images(args.root, viol)

    if not args.skip_destination:
        destination_check(
            os.path.join(args.root, "violations_only.csv"),
            os.path.join(args.root, "destination_reachability.csv"),
            args, label="violations",
        )
        destination_check(
            os.path.join(args.root, "non_violations.csv"),
            os.path.join(args.root, "destination_reachability_nonviolated.csv"),
            args, label="nonviolations",
        )


if __name__ == "__main__":
    main()
