"""Stress-test offscreen rendering stability for navigate_safe tasks.

For each ``(env_name, layout)`` under a fixed style (MODERN_1 by default), build
the env, call ``env.reset()``, then render N frames in a row. Record per-render
timing and flag any exception or per-render timeout (treated as a deadlock).

Output: CSV row per task with success/failure counts, max render time, total
time, and the error string if anything went wrong. The first and last frames
are saved when --save_frames is set, so visually-broken scenes are inspectable.
"""

import argparse
import csv
import logging
import os
import signal
import sys
import time
import traceback

import imageio
import numpy as np

import robosuite
from robosuite.controllers import load_composite_controller_config

from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.models.scenes.scene_registry import (
    LAYOUT_GROUPS_TO_IDS, LayoutType, StyleType,
)

import task_listup


class RenderTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise RenderTimeout("render timed out")


def build_env(env_name, layout_id, style_id, seed, gpu_id):
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    return robosuite.make(
        env_name=env_name,
        robots="PandaOmron",
        controller_configs=cc,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        camera_names=["topview"],
        camera_widths=64,
        camera_heights=64,
        camera_depths=False,
        seed=seed,
        layout_ids=[layout_id],
        style_ids=[style_id],
        translucent_robot=False,
        render_gpu_device_id=gpu_id,
    )


def stress_one(env_name, layout_id, style_id, seed, gpu_id,
               n_renders, render_h, render_w, camera, render_timeout_s,
               save_frames_dir=None):
    """Build env, reset, render n times. Returns dict with metrics."""
    t0 = time.monotonic()
    info = {
        "env_name": env_name,
        "layout": LayoutType(layout_id).name,
        "style": StyleType(style_id).name,
        "seed": seed,
        "n_renders_requested": n_renders,
        "n_success": 0,
        "n_failure": 0,
        "n_timeout": 0,
        "max_render_ms": 0.0,
        "min_render_ms": float("inf"),
        "sum_render_ms": 0.0,
        "build_ms": 0.0,
        "reset_ms": 0.0,
        "total_ms": 0.0,
        "deadlock": 0,
        "error": "",
        "first_frame_path": "",
        "last_frame_path": "",
    }
    env = None
    last_frame = None
    first_frame = None
    try:
        ts = time.monotonic()
        env = build_env(env_name, layout_id, style_id, seed, gpu_id)
        info["build_ms"] = (time.monotonic() - ts) * 1000

        ts = time.monotonic()
        env.reset()
        info["reset_ms"] = (time.monotonic() - ts) * 1000

        for i in range(n_renders):
            signal.alarm(int(render_timeout_s))
            try:
                tr = time.monotonic()
                frame = env.sim.render(
                    height=render_h, width=render_w, camera_name=camera
                )[::-1]
                dt_ms = (time.monotonic() - tr) * 1000
                signal.alarm(0)
                info["n_success"] += 1
                info["max_render_ms"] = max(info["max_render_ms"], dt_ms)
                info["min_render_ms"] = min(info["min_render_ms"], dt_ms)
                info["sum_render_ms"] += dt_ms
                if i == 0:
                    first_frame = np.asarray(frame)
                last_frame = np.asarray(frame)
            except RenderTimeout:
                signal.alarm(0)
                info["n_timeout"] += 1
                info["n_failure"] += 1
                info["deadlock"] = 1
                info["error"] = (
                    info["error"] + f"; render#{i}: TIMEOUT>{render_timeout_s}s"
                ).lstrip("; ")
                # If a render deadlocked, the env is likely in a bad state.
                break
            except Exception as e:
                signal.alarm(0)
                info["n_failure"] += 1
                info["error"] = (
                    info["error"] + f"; render#{i}: {e!r}"
                ).lstrip("; ")
                break
    except RenderTimeout:
        info["n_timeout"] += 1
        info["n_failure"] += 1
        info["deadlock"] = 1
        info["error"] = (info["error"] + "; build/reset: TIMEOUT").lstrip("; ")
    except Exception as e:
        info["n_failure"] += 1
        info["error"] = (info["error"] + f"; setup: {e!r}").lstrip("; ")
    finally:
        signal.alarm(0)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    if not np.isfinite(info["min_render_ms"]):
        info["min_render_ms"] = 0.0
    info["total_ms"] = (time.monotonic() - t0) * 1000

    if save_frames_dir is not None:
        os.makedirs(save_frames_dir, exist_ok=True)
        if first_frame is not None:
            p = os.path.join(
                save_frames_dir, f"{env_name}_{info['layout']}_seed{seed}_first.png"
            )
            try:
                imageio.imwrite(p, first_frame)
                info["first_frame_path"] = p
            except Exception:
                pass
        if last_frame is not None and info["n_success"] > 1:
            p = os.path.join(
                save_frames_dir, f"{env_name}_{info['layout']}_seed{seed}_last.png"
            )
            try:
                imageio.imwrite(p, last_frame)
                info["last_frame_path"] = p
            except Exception:
                pass

    return info


CSV_FIELDS = [
    "env_name", "layout", "style", "seed",
    "n_renders_requested", "n_success", "n_failure", "n_timeout",
    "deadlock",
    "build_ms", "reset_ms",
    "min_render_ms", "max_render_ms", "sum_render_ms",
    "total_ms",
    "error",
    "first_frame_path", "last_frame_path",
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layout", type=str, default="all",
                   choices=[lt.name for lt in LayoutType] + ["all"])
    p.add_argument("--style", type=str, default="MODERN_1",
                   choices=[st.name for st in StyleType])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--filter", type=str, default=None)
    p.add_argument("--exclude", type=str, default=None)
    p.add_argument("--n_renders", type=int, default=10)
    p.add_argument("--render_h", type=int, default=512)
    p.add_argument("--render_w", type=int, default=768)
    p.add_argument("--camera", type=str, default="topview")
    p.add_argument("--render_timeout_s", type=float, default=30.0,
                   help="Per-render deadlock timeout (seconds).")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--out_dir", type=str,
                   default="render_stability",
                   help="Output directory")
    p.add_argument("--save_frames", action="store_true",
                   help="Save first/last frame per task (debug).")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    suffix = args.layout if args.layout != "all" else "all"
    log_path = os.path.join(logs_dir, f"run_{suffix}.log")
    rl = logging.getLogger()
    rl.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, mode="w"); fh.setFormatter(fmt); rl.addHandler(fh)
    sh = logging.StreamHandler(); sh.setFormatter(fmt); rl.addHandler(sh)

    signal.signal(signal.SIGALRM, _alarm_handler)

    target_envs = list(task_listup.navigate_safe_tasks)
    if args.filter:
        target_envs = [e for e in target_envs if args.filter.lower() in e.lower()]
    if args.exclude:
        target_envs = [e for e in target_envs if args.exclude.lower() not in e.lower()]
    target_envs = [e for e in target_envs if e in ALL_KITCHEN_ENVIRONMENTS]
    logging.info("Tasks: %d", len(target_envs))
    logging.info("Style: %s (fixed)", args.style)
    logging.info("Out dir: %s", out_dir)

    if args.layout == "all":
        layout_ids = LAYOUT_GROUPS_TO_IDS[LayoutType.ALL]
    else:
        layout_ids = [LayoutType[args.layout].value]
    style_id = StyleType[args.style]

    out_csv = os.path.join(out_dir, f"render_stability_{suffix}.csv")
    fdir = os.path.join(out_dir, f"frames_{suffix}") if args.save_frames else None
    csv_f = open(out_csv, "w", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
    csv_w.writeheader()

    n_run = 0
    n_deadlock = 0
    n_failure = 0
    try:
        for layout_id in layout_ids:
            layout_name = LayoutType(layout_id).name
            for env_name in target_envs:
                if args.limit and n_run >= args.limit:
                    break
                info = stress_one(
                    env_name, layout_id, style_id, args.seed,
                    args.gpu_id, args.n_renders,
                    args.render_h, args.render_w, args.camera,
                    args.render_timeout_s,
                    save_frames_dir=fdir,
                )
                csv_w.writerow(info)
                csv_f.flush()
                n_run += 1
                if info["deadlock"]:
                    n_deadlock += 1
                if info["n_failure"] > 0:
                    n_failure += 1
                level = (
                    logging.ERROR if info["n_failure"] else logging.INFO
                )
                logging.log(
                    level,
                    "%s/%s  ok=%d/%d  fail=%d  timeout=%d  "
                    "max_render=%.1fms  total=%.1fms%s",
                    env_name, layout_name,
                    info["n_success"], info["n_renders_requested"],
                    info["n_failure"], info["n_timeout"],
                    info["max_render_ms"], info["total_ms"],
                    f"  ERR={info['error']}" if info["error"] else "",
                )
    finally:
        csv_f.close()

    logging.info(
        "Total runs: %d  deadlocks: %d  failures: %d  CSV: %s",
        n_run, n_deadlock, n_failure, out_csv,
    )


if __name__ == "__main__":
    sys.exit(main())
