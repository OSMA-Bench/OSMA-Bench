"""
Visualize a kitchen layout with the robot and posed_person hidden.

Default layout: U_SHAPED_LARGE.

Usage:
    python view_layout.py                              # interactive viewer
    python view_layout.py --layout galley
    python view_layout.py --save out.png               # offscreen, save PNG
    python view_layout.py --save out.png --camera robot0_frontview
    python view_layout.py --no-render                  # quick smoke test
"""
import argparse
import os
import numpy as np
import imageio

import robosuite
from robosuite.controllers import load_composite_controller_config

from robocasa.models.scenes.scene_registry import LayoutType, StyleType


LAYOUT_MAP = {
    "one_wall_small":  LayoutType.ONE_WALL_SMALL,
    "one_wall_large":  LayoutType.ONE_WALL_LARGE,
    "l_shaped_small":  LayoutType.L_SHAPED_SMALL,
    "l_shaped_large":  LayoutType.L_SHAPED_LARGE,
    "u_shaped_small":  LayoutType.U_SHAPED_SMALL,
    "u_shaped_large":  LayoutType.U_SHAPED_LARGE,
    "g_shaped_small":  LayoutType.G_SHAPED_SMALL,
    "g_shaped_large":  LayoutType.G_SHAPED_LARGE,
    "galley":          LayoutType.GALLEY,
    "wraparound":      LayoutType.WRAPAROUND,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="u_shaped_large", choices=list(LAYOUT_MAP.keys()))
    parser.add_argument("--style", default="modern_1")
    parser.add_argument("--camera", default="topview")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-render", action="store_true",
                        help="Smoke-test only (no window, no save)")
    parser.add_argument("--save", default=None,
                        help="Save an offscreen render to this path (PNG). Disables interactive window.")
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--settle-steps", type=int, default=5,
                        help="Zero-action steps before capture (lets visualize() apply)")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Steps for interactive mode")
    args = parser.parse_args()

    layout = LAYOUT_MAP[args.layout]
    style = StyleType[args.style.upper()]
    offscreen = args.no_render or (args.save is not None)
    print(f"Layout: {args.layout}  Style: {args.style}  Camera: {args.camera}  "
          f"Mode: {'offscreen' if offscreen else 'interactive'}")

    cfg = load_composite_controller_config(controller=None, robot="PandaOmron")

    env = robosuite.make(
        env_name="KitchenLayoutView",
        robots="PandaOmron",
        controller_configs=cfg,
        has_renderer=not offscreen,
        has_offscreen_renderer=offscreen,
        render_camera=args.camera,
        ignore_done=True,
        use_object_obs=False,
        use_camera_obs=False,
        seed=args.seed,
        layout_ids=[layout],
        style_ids=[style],
        translucent_robot=False,
    )

    env.reset()
    print("Scene loaded. Robot and posed_person are hidden.")

    low, high = env.action_spec
    zero = np.zeros_like(high)

    if args.save is not None:
        # Step a few times so visualize() runs and alpha overrides take effect
        for _ in range(args.settle_steps):
            env.step(zero)
        frame = env.sim.render(
            height=args.height, width=args.width, camera_name=args.camera
        )[::-1]
        out_path = os.path.abspath(args.save)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        imageio.imwrite(out_path, frame)
        print(f"Saved {frame.shape[1]}x{frame.shape[0]} image -> {out_path}")
    elif args.no_render:
        for _ in range(args.settle_steps):
            env.step(zero)
        print("Smoke test OK.")
    else:
        for _ in range(args.steps):
            env.step(zero)
            env.render()

    env.close()


if __name__ == "__main__":
    main()
