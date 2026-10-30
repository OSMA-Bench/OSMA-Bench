"""
Run environments without teleoperation.
Executes simulations with random or zero actions and optionally records videos.

Usage examples:
    # Run a specific environment with zero actions
    python run_own_asset_no_teleop.py --env move_hot_object --filter_env_keyword MoveCoffeeToTableNoHuman

    # Run all layouts for an environment category
    python run_own_asset_no_teleop.py --layout all --env move_hot_object --record_path=./videos

    # Run with random actions
    python run_own_asset_no_teleop.py --env handover --action_mode random --record_path=./videos

    # Skip existing recordings
    python run_own_asset_no_teleop.py --layout all --env navigate_safe --record_path=./videos --skip-existing
"""

import os
import numpy as np
import imageio
import argparse
import random
from termcolor import colored
from tqdm import tqdm

import robosuite
from robosuite.controllers import load_composite_controller_config
import logging

from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.utils.env_utils import hide_gripper_marker_sites
from robocasa.models.scenes.scene_registry import LayoutType, StyleType, LAYOUT_GROUPS_TO_IDS
import task_listup
from robocasa.utils.egl_device import resolve_egl_device


# Environment category definitions
ENV_CATEGORIES = {
    'handover': task_listup.handover_tasks,
    'navigate_safe': task_listup.navigate_safe_tasks,
    'move_hot_object': task_listup.move_hot_object_to_table_tasks,
    'open_door_safe': ["OpenDoorSafe"],
    'close_door_safe': ["CloseDoorSafeCenter", "CloseDoorSafeThreshold", "CloseDoorSafeEdge"],
}


def create_env(
    env_name,
    robots="PandaOmron",
    camera_names=None,
    camera_widths=128,
    camera_heights=128,
    seed=None,
    render_onscreen=False,
    render_camera="topview",
    layout_ids=None,
    style_ids=None,
    has_human=True,
    gpu_id=0,
):
    """Create environment for simulation."""
    if camera_names is None:
        camera_names = [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
            "robot0_frontview",
        ]

    controller_config = load_composite_controller_config(
        controller=None,
        robot=robots if isinstance(robots, str) else robots[0],
    )

    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_config,
        has_renderer=render_onscreen,
        has_offscreen_renderer=not render_onscreen,
        render_camera=render_camera,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        camera_depths=False,
        seed=seed,
        layout_ids=layout_ids,
        style_ids=style_ids,
        translucent_robot=False,
        render_gpu_device_id=gpu_id,
    )

    # Pass has_human for environments that support it (but not for NoHuman variants which set it internally)
    if 'Door' in env_name and 'NoHuman' not in env_name:
        env_kwargs['has_human'] = has_human

    env = robosuite.make(**env_kwargs)
    hide_gripper_marker_sites(env)
    return env

# ===== 액션 인덱스 (테이블 기준 고정) =====
IDX_FWD_BACK = 7     # 앞으로/뒤로
IDX_LEFT_RIGHT = 8   # 왼/오
IDX_LEFT_TURN=9      # 왼쪽으로 돌기t
IDX_UP = 10          # 위로
IDX_DOWN = 11        # 아래로 (양수 가속)
IDX_YAW = 3          # 시계/반시계 (부호로 양방향)
IDX_PITCH = 4        # 앞/뒤로 돌기
IDX_ROLL = 5         # 왼/오로 돌기
IDX_GRIP = 6         # 집기(그리퍼)
IDX_GRIP_FRONT = 0   # 그리퍼 앞으로
IDX_GRIP_LEFT = 1    # 그리퍼 왼쪽으로
IDX_GRIP_UP = 2      # 그리퍼 위로
lin = 1.0
rot = 0.5
grip = 1.0

def run_simulation(
    env,
    horizon=100,
    record_path=None,
    camera_name="topview",
    render_height=1024,
    render_width=1536,
    action_mode="zero",
    check_success_interval=10,
    render_onscreen=False,
    settle_steps=30,
    args=None,
):
    """
    Run simulation with specified action mode.

    Args:
        env: robosuite environment
        horizon: number of steps to run
        record_path: path to save video (if None, no recording)
        camera_name: camera to render from
        render_height: height of rendered frames
        render_width: width of rendered frames
        action_mode: "random" for random actions, "zero" for no movement
        check_success_interval: how often to check for success
        render_onscreen: whether to render onscreen

    Returns:
        dict with simulation results
    """
    writer = imageio.get_writer(record_path, fps=20) if record_path else None

    low, high = env.action_spec
    if action_mode != "zero":
        obs = env.reset()
        hide_gripper_marker_sites(env)

    # Settle physics with zero actions before recording so the first frame
    # shows objects resting on surfaces instead of mid-clip / mid-fall.
    zero_action = np.zeros_like(high)
    for _ in range(settle_steps):
        env.step(zero_action)

    # Initial frame
    if writer is not None:
        if args.capture_initial_stage:
            obs = env.reset()
            hide_gripper_marker_sites(env)
        frame = env.sim.render(height=render_height, width=render_width, camera_name=camera_name)[::-1]
        writer.append_data(frame)
    elif render_onscreen:
        env.render()

    success = False
    success_step = None

    pbar = tqdm(range(horizon), desc="Running simulation")
    for t in pbar:
        action = np.zeros_like(high)
        # Generate action
        if action_mode == "random":
            action = np.random.uniform(low, high)
        elif action_mode == "turn_left":
            if t < 25:
                action[IDX_LEFT_TURN] += lin
            elif 25 <= t < 50:
                action[IDX_GRIP_FRONT] += lin
        else:  # zero
            action = np.zeros_like(high)

        # Step environment
        obs, reward, done, info = env.step(action)

        # Record frame
        if writer is not None:
            frame = env.sim.render(height=render_height, width=render_width, camera_name=camera_name)[::-1]
            if action_mode == "zero" and t == 10 and args.capture_initial_stage:
                imageio.imwrite(os.path.join(os.path.dirname(record_path), f"initial_{os.path.basename(record_path).replace('.mp4', '.png')}"), frame)
            writer.append_data(frame)
        elif render_onscreen:
            env.render()

        # Check success periodically
        if (t % check_success_interval) == 0:
            try:
                if env._check_success() and not success:
                    success = True
                    success_step = t
                    pbar.set_description(f"Success at step {t}!")
            except Exception as e:
                pass  # Some environments may not have _check_success

    if action_mode != 'zero' and args.capture_initial_stage:
        imageio.imwrite(os.path.join(os.path.dirname(record_path), f"initial_{os.path.basename(record_path).replace('.mp4', '.png')}"), frame)
    if writer is not None:
        writer.close()
        logging.info(colored(f"Video saved: {record_path}", "green"))
        

    return {
        "success": success,
        "success_step": success_step,
        "total_steps": horizon,
    }


def main():
    parser = argparse.ArgumentParser(description="Run environments without teleoperation")
    parser.add_argument('--env', type=str, default='move_hot_object',
                        choices=list(ENV_CATEGORIES.keys()) + ['custom'],
                        help='Environment category to run')
    parser.add_argument('--env_names', type=str, nargs='+', default=None,
                        help='Specific environment names (for --env custom)')
    parser.add_argument('--specific_env', type=str, default=None,
                        help='Specify a single specific environment name')
    parser.add_argument('--filter_env_keyword', type=str, default=None,
                        help='Keyword to filter environment names')
    parser.add_argument('--record_path', type=str, default=None,
                        help='Directory to save recorded videos')
    parser.add_argument('--horizon', type=int, default=15,
                        help='Number of simulation steps')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--layout', type=str, default=None,
                        choices=[lt.name for lt in LayoutType] + ['all'],
                        help='Specific layout to test')
    parser.add_argument('--no-human', action='store_true',
                        help='Disable human in the environment')
    parser.add_argument('--action_mode', type=str, default='zero',
                        choices=['zero', 'random', 'turn_left'],
                        help='Action mode: zero (stationary) or random')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip recording if file already exists')
    parser.add_argument('--render_onscreen', action='store_true',
                        help='Render onscreen instead of offscreen')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU device ID for rendering')
    parser.add_argument('--filter_out_keyword', type=str, default=None,
                        help='Keyword to filter out environment names')
    parser.add_argument("--capture_initial_stage", action="store_true", help="capture initial step")
    parser.add_argument("--camera_view", type=str, default="topview", help="camera view for recording and rendering")
    parser.add_argument("--camera_shape", type=int, nargs=2, default=[1024, 1536], help="camera resolution for recording and rendering (height width)")
    parser.add_argument("--settle_steps", type=int, default=30,
                        help="Zero-action steps run BEFORE recording starts, so "
                             "the first frame shows objects at rest. Pass 0 to "
                             "record the spawn drop itself (obstacles spawn 1-2 "
                             "cm above their support by design: 1 cm for the "
                             "standing-table obstacles, 2 cm for the floor "
                             "ones) — needed when "
                             "inspecting popout_z / stability verdicts.")
    args = parser.parse_args()

    # Determine environments to run
    if args.specific_env is not None:
        target_envs = [args.specific_env]
    elif args.env == 'custom':
        if args.env_names is None:
            raise ValueError("--env_names required when --env custom")
        target_envs = args.env_names
    else:
        target_envs = ENV_CATEGORIES.get(args.env, [])

    # Filter by keyword if provided
    if args.filter_env_keyword:
        logging.info(f"Filtering environments with keyword: {args.filter_env_keyword}")
        target_envs = [env for env in target_envs if args.filter_env_keyword.lower() in env.lower()]
    # # Filter out coffee-related environment classes.
    target_envs = [env for env in target_envs if "coffee" not in env.lower()]
    # Filter out environments with the specified keyword
    if args.filter_out_keyword:
        logging.info(f"[info] Filtering out environments with keyword: {args.filter_out_keyword}")
        target_envs = [env for env in target_envs if args.filter_out_keyword.lower() not in env.lower()]
    logging.info(f"[info] Number of target environments: {len(target_envs)}")

    # Validate environments
    valid_envs = []
    for env_name in target_envs:
        if env_name in ALL_KITCHEN_ENVIRONMENTS:
            valid_envs.append(env_name)
        else:
            logging.warning(f"Environment '{env_name}' not found in registry, skipping")

    if not valid_envs:
        logging.error("[error] No valid environments to run")
        logging.info(f"[info] Available environments in category '{args.env}':")
        for env in ENV_CATEGORIES.get(args.env, []):
            status = "OK" if env in ALL_KITCHEN_ENVIRONMENTS else "NOT REGISTERED"
            logging.info(f"  - {env}: {status}")
        return

    target_envs = valid_envs
    logging.info(f"[info] Target environments: {len(target_envs)} - {target_envs}")
    logging.info(f"[info] The number of tasks : {len(target_envs)}")
    # import sys
    # sys.exit()
    # Determine layouts
    if args.layout == 'all':
        layout_ids = LAYOUT_GROUPS_TO_IDS[LayoutType.ALL]
    elif args.layout is not None:
        layout_ids = [LayoutType[args.layout].value]
    else:
        # Default layouts
        layout_ids = [
            LayoutType.ONE_WALL_SMALL.value,
            LayoutType.L_SHAPED_SMALL.value,
        ]

    logging.info(f"[info] Layouts: {[LayoutType(lid).name for lid in layout_ids]}")
    logging.info(f"[info] Action mode: {args.action_mode}")
    logging.info(f"[info] Has human: {not args.no_human}")

    # Setup output directory
    if args.record_path:
        os.makedirs(args.record_path, exist_ok=True)

    # MuJoCo addresses GPUs by EGL enumeration order, not CUDA ordinal, so the
    # requested id can be a device that cannot create a headless context.
    # Resolve once, before the run, rather than failing on every episode.
    gpu_id = resolve_egl_device(args.gpu_id)
    if gpu_id is None:
        gpu_id = args.gpu_id

    # Run simulations
    results = []
    for layout_id in layout_ids:
        layout_name = LayoutType(layout_id).name
        logging.info(f"\n[info] Testing layout: {layout_name}")

        for env_name in target_envs:
            logging.info(f"[info] Running: {env_name}")

            # Determine record path
            if args.record_path:
                record_file = os.path.join(args.record_path, f"{env_name}_{layout_name}.mp4")
                if args.skip_existing and os.path.exists(record_file):
                    logging.info(f"[info] Skipping (already exists): {record_file}")
                    continue
            else:
                record_file = None
            if record_file is not None:
                log_dir = os.path.join(args.record_path, "log")
                os.makedirs(log_dir, exist_ok=True)
                logging_file = os.path.join(log_dir, f"{env_name}_{layout_name}.log")
                logging.basicConfig(level=logging.DEBUG, filename=logging_file, filemode="a", format="%(asctime)s - %(levelname)s - %(message)s")
            else:
                logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
            try:
                print(f"Starting simulation for {env_name} on layout {layout_name} with camera view {args.camera_view} ")
                # Create environment
                env = create_env(
                    env_name=env_name,
                    seed=args.seed,
                    layout_ids=[layout_id],
                    style_ids=[StyleType.MODERN_1],
                    render_onscreen=args.render_onscreen,
                    camera_names=[args.camera_view],
                    has_human=not args.no_human,
                    gpu_id=gpu_id,
                )

                # Run simulation
                result = run_simulation(
                    env,
                    horizon=args.horizon,
                    record_path=record_file,
                    action_mode=args.action_mode,
                    render_onscreen=args.render_onscreen,
                    render_width=args.camera_shape[1],
                    render_height=args.camera_shape[0],
                    settle_steps=args.settle_steps,
                    camera_name=args.camera_view,
                    args=args,
                )

                result['env_name'] = env_name
                result['layout'] = layout_name
                result['status'] = 'success'
                results.append(result)

                env.close()

                if result['success']:
                    logging.info(f"  [done] Task success at step {result['success_step']}")
                else:
                    logging.info(f"  [done] Completed {result['total_steps']} steps")

            except Exception as e:
                logging.error(f"  [error] {env_name}: {str(e)}")
                results.append({
                    'env_name': env_name,
                    'layout': layout_name,
                    'status': 'error',
                    'error': str(e),
                })

    # summary
    logging.info("\n" + "=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    success_count = sum(1 for r in results if r.get('status') == 'success')
    error_count = sum(1 for r in results if r.get('status') == 'error')
    task_success_count = sum(1 for r in results if r.get('success', False))

    logging.info(f"Total runs: {len(results)}")
    logging.info(f"Successful runs: {success_count}")
    logging.info(f"Errors: {error_count}")
    logging.info(f"Task successes: {task_success_count}")

    if error_count > 0:
        logging.info("\nErrors:")
        for r in results:
            if r.get('status') == 'error':
                logging.info(f"  - {r['env_name']} ({r['layout']}): {r.get('error', 'unknown')}")

    logging.info("\nDone.")


if __name__ == "__main__":
    main()
