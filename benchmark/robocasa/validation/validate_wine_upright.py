"""Verify the wine-bottle-standing-upright fix across every layout and every
wine variant (Blocking/NonBlocking x Route A-G).

Originally reproduced with:
    mjpython run_env.py --env navigate_safe --layout ONE_WALL_SMALL \
        --filter_env_keyword=NavigateKitchenWineBlockingRouteA

The fix only had to be sanity-checked on a single (variant, layout). This
driver renders all 14 wine variants on all 10 layouts (= 140 videos) into a
flat output folder so the upright pose can be eyeballed quickly. A copy of
this script and the exact command used are stored alongside the videos for
reproducibility.

Examples:
    python verify_wine_upright.py
    python verify_wine_upright.py --gpus 0 1 2 3        # one layout per GPU
    python verify_wine_upright.py --layouts ONE_WALL_SMALL L_SHAPED_LARGE
"""
import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "run_env_no_teleop.py")

ALL_LAYOUTS = [
    "ONE_WALL_SMALL", "ONE_WALL_LARGE",
    "L_SHAPED_SMALL", "L_SHAPED_LARGE",
    "U_SHAPED_SMALL", "U_SHAPED_LARGE",
    "G_SHAPED_SMALL", "G_SHAPED_LARGE",
    "GALLEY", "WRAPAROUND",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="wine_upright_check",
                   help="Flat output folder for all videos")
    p.add_argument("--layouts", nargs="+", default=None,
                   help=f"Layouts to render (default: all 10, {ALL_LAYOUTS})")
    p.add_argument("--horizon", type=int, default=20,
                   help="Sim steps per variant (long enough to see falling)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--gpus", type=int, nargs="+", default=None,
                   help="GPU ids; one layout per GPU concurrently")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing",
                   action="store_false")
    return p.parse_args()


def render_layout(layout_name, args, gpu_id=None, prefix=""):
    if gpu_id is None:
        gpu_id = args.gpu_id
    cmd = [
        sys.executable, RUNNER,
        "--env", "navigate_safe",
        "--layout", layout_name,
        "--filter_env_keyword", "Wine",
        "--record_path", args.out,
        "--horizon", str(args.horizon),
        "--seed", str(args.seed),
        "--gpu_id", str(gpu_id),
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"{prefix}[run ] gpu={gpu_id} layout={layout_name}")
    subprocess.run(cmd, check=True, env=env)


def render_parallel(layouts, args):
    q: "queue.Queue[str]" = queue.Queue()
    for L in layouts:
        q.put(L)
    errors = []
    err_lock = threading.Lock()

    def worker(gpu_id):
        prefix = f"[gpu{gpu_id}] "
        while True:
            try:
                L = q.get_nowait()
            except queue.Empty:
                return
            try:
                render_layout(L, args, gpu_id=gpu_id, prefix=prefix)
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


def save_provenance(args, layouts):
    """Drop a copy of this script and the exact command into the output folder
    so the run is reproducible without re-deriving it from shell history.
    """
    os.makedirs(args.out, exist_ok=True)
    shutil.copy2(__file__, os.path.join(args.out, os.path.basename(__file__)))
    cmd = ["python", os.path.basename(__file__)]
    if args.layouts:
        cmd += ["--layouts", *layouts]
    if args.gpus:
        cmd += ["--gpus", *map(str, args.gpus)]
    if args.horizon != 20:
        cmd += ["--horizon", str(args.horizon)]
    if args.seed != 0:
        cmd += ["--seed", str(args.seed)]
    if not args.skip_existing:
        cmd += ["--no-skip-existing"]
    with open(os.path.join(args.out, "run_command.sh"), "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("# Command used to render this folder.\n")
        f.write("# PYTHONPATH must include both robosuite and robocasa.\n")
        f.write(" ".join(cmd) + "\n")
    os.chmod(os.path.join(args.out, "run_command.sh"), 0o755)


def main():
    args = parse_args()
    layouts = args.layouts or ALL_LAYOUTS
    os.makedirs(args.out, exist_ok=True)

    print(f"[info] Layouts ({len(layouts)}): {layouts}")
    print(f"[info] 14 wine variants per layout = {14 * len(layouts)} videos")
    print(f"[info] Output: {args.out}")

    save_provenance(args, layouts)

    if args.gpus:
        print(f"[info] Parallel across GPUs: {args.gpus}")
        render_parallel(layouts, args)
    else:
        for L in layouts:
            render_layout(L, args)
    print("[info] Done.")


if __name__ == "__main__":
    main()
