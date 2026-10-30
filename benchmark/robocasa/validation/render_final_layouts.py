"""Render every navigate_safe variant on the 8 enclosed/standard layouts.

Variants: 18 obstacles x 2 blocking modes x 7 routes - 2 (the human obstacle skips RouteF
in both blocking and non-blocking) = 250 task classes.
Layouts (8): ONE_WALL_SMALL/LARGE, L_SHAPED_SMALL/LARGE, U_SHAPED_SMALL/LARGE,
G_SHAPED_SMALL/LARGE.

Total: 250 x 8 = 2000 videos saved to <out>/<layout>/<env>_<layout>.mp4
(one subfolder per layout).

This is a WIDER sweep than the benchmark itself. The released benchmark is the
five layouts in scripts/nav_sweep.sh (ONE_WALL_SMALL, L_SHAPED_SMALL,
L_SHAPED_LARGE, G_SHAPED_SMALL, G_SHAPED_LARGE) = 250 x 5 = 1250 tasks; the
extra three here are rendered to eyeball placement on layouts that are tuned in
nav_placement_params.py but not part of the released set. For the benchmark
figures use scripts/nav_sweep.sh, which also builds the per-tier overlays.

Examples:
    python render_final_layouts.py
    python render_final_layouts.py --gpus 0 1 2 3        # one layout per GPU
    python render_final_layouts.py --skip-existing        # resume
    python render_final_layouts.py --layouts U_SHAPED_LARGE WRAPAROUND
"""
import argparse
import os
import queue
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "run_env_no_teleop.py")

DEFAULT_LAYOUTS = [
    "ONE_WALL_SMALL",
    "ONE_WALL_LARGE",
    "L_SHAPED_SMALL",
    "L_SHAPED_LARGE",
    "U_SHAPED_SMALL",
    "U_SHAPED_LARGE",
    "G_SHAPED_SMALL",
    "G_SHAPED_LARGE",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="final_layout_check",
                   help="Output directory for videos")
    p.add_argument("--layouts", nargs="+", default=None,
                   help=f"Layout names to render (default: {DEFAULT_LAYOUTS})")
    p.add_argument("--horizon", type=int, default=11)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu_id", type=int, default=0,
                   help="Single-GPU id (used when --gpus not set)")
    p.add_argument("--gpus", type=int, nargs="+", default=None,
                   help="GPU ids for parallel rendering; one layout per GPU")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip videos that already exist (default: on)")
    p.add_argument("--no-skip-existing", dest="skip_existing",
                   action="store_false")
    return p.parse_args()


def render_layout(layout_name, args, gpu_id=None, prefix=""):
    if gpu_id is None:
        gpu_id = args.gpu_id
    layout_out = os.path.join(args.out, layout_name)
    os.makedirs(layout_out, exist_ok=True)
    cmd = [
        sys.executable, RUNNER,
        "--env", "navigate_safe",
        "--layout", layout_name,
        "--record_path", layout_out,
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


def main():
    args = parse_args()
    layouts = args.layouts or DEFAULT_LAYOUTS
    os.makedirs(args.out, exist_ok=True)

    print(f"[info] Layouts ({len(layouts)}): {layouts}")
    print(f"[info] Output: {args.out}")
    if args.gpus:
        print(f"[info] Parallel across GPUs: {args.gpus}")
        render_parallel(layouts, args)
    else:
        for L in layouts:
            render_layout(L, args)
    print("[info] Done.")


if __name__ == "__main__":
    main()
