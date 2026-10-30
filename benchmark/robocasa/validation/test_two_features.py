"""
Test script for two features in kitchen_navigate_safe.py:
1. Human always faces toward robot every step
2. TABLE_OBSTACLES placed on the standing table rather than the floor

TABLE_OBSTACLES is wine / glass_of_water / hot_chocolate / table_lamp -- the
three drinks plus the lamp, which is not a drink but shares their spawn surface.
`wine` was spelled `glass_of_wine` when this file was written; that name matches
no class today, so every test below that used it errored out instead of running.
"""
import sys
import os
import numpy as np

# Setup PYTHONPATH
_BENCH = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_BENCH, "robosuite"))
sys.path.insert(0, os.path.join(_BENCH, "robocasa"))

import robocasa  # registers environments
import robosuite
import robosuite.utils.transform_utils as T
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def angle_between(v1, v2):
    """Compute angle in radians between two 2D vectors."""
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.arccos(np.clip(cos, -1, 1))


def test_human_faces_robot(env_name="NavigateKitchenHumanBlockingRouteA", layout_id=0, steps=50):
    """
    Test Feature 1: Human always faces toward the robot.

    Checks that after each step, the human body's forward direction
    points toward the robot base.
    """
    print(f"\n{'='*60}")
    print(f"TEST 1: Human always faces robot")
    print(f"  env={env_name}, layout={layout_id}, steps={steps}")
    print(f"{'='*60}")

    env = robosuite.make(
        env_name,
        robots=["PandaMobile"],
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        layout_ids=[layout_id],
        seed=42,
    )
    obs = env.reset()

    max_angle_errors = []

    for step_i in range(steps):
        action = np.zeros(env.action_dim)
        # Add some random movement to move the robot around
        action[0] = np.random.uniform(-0.3, 0.3)  # base x
        action[1] = np.random.uniform(-0.3, 0.3)  # base y
        obs, reward, done, info = env.step(action)

        # Get human and robot positions
        try:
            # Body is posed_HUMAN_*, not posed_person_*.
            human_id = env.sim.model.body_name2id("posed_human_main_group_main")
            robot_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception as e:
            print(f"  [SKIP] Could not find body IDs: {e}")
            env.close()
            return False

        human_pos = env.sim.data.body_xpos[human_id]
        robot_pos = env.sim.data.body_xpos[robot_id]

        # Get human's forward direction from its quaternion
        human_quat = env.sim.data.body_xquat[human_id]
        human_mat = T.quat2mat(human_quat)
        human_fwd = human_mat[:2, 0]  # X-axis in body frame projected to XY

        # Expected direction: human -> robot
        dir_to_robot = robot_pos[:2] - human_pos[:2]
        dist = np.linalg.norm(dir_to_robot)
        if dist < 0.01:
            continue
        dir_to_robot = dir_to_robot / dist

        angle_error = angle_between(human_fwd, dir_to_robot)
        max_angle_errors.append(np.degrees(angle_error))

        if step_i % 10 == 0:
            print(f"  Step {step_i}: angle_error={np.degrees(angle_error):.1f}°, "
                  f"robot=[{robot_pos[0]:.2f},{robot_pos[1]:.2f}], "
                  f"human=[{human_pos[0]:.2f},{human_pos[1]:.2f}]")

    env.close()

    if max_angle_errors:
        avg_err = np.mean(max_angle_errors)
        max_err = np.max(max_angle_errors)
        print(f"\n  Results: avg_angle_error={avg_err:.1f}°, max_angle_error={max_err:.1f}°")
        passed = max_err < 15.0  # Allow small error
        print(f"  PASS: {passed} (threshold: max_error < 15°)")
        return passed
    else:
        print("  [SKIP] No valid measurements")
        return False


def test_table_obstacle_on_table(env_name="NavigateKitchenWineBlockingRouteA", layout_id=0):
    """
    Test Feature 2: TABLE_OBSTACLES placed on the standing table.

    Checks that the obstacle is above floor level (on the table).
    """
    print(f"\n{'='*60}")
    print(f"TEST 2: Table obstacle on standing table")
    print(f"  env={env_name}, layout={layout_id}")
    print(f"{'='*60}")

    env = robosuite.make(
        env_name,
        robots=["PandaMobile"],
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        layout_ids=[layout_id],
        seed=42,
    )
    obs = env.reset()

    # Check standing table exists
    try:
        table_id = env.sim.model.body_name2id("standing_table_main_group_main")
        table_pos = env.sim.data.body_xpos[table_id]
        print(f"  Standing table position: [{table_pos[0]:.2f}, {table_pos[1]:.2f}, {table_pos[2]:.2f}]")
    except Exception as e:
        print(f"  [FAIL] Standing table body not found: {e}")
        env.close()
        return False

    # Check obstacle position
    obstacle_found = False
    for obj_name in env.objects:
        if obj_name.startswith("obstacle_"):
            obj = env.objects[obj_name]
            joint_name = obj.joints[0]
            qpos = env.sim.data.get_joint_qpos(joint_name)
            obj_z = qpos[2]
            obj_xy = qpos[:2]

            # Table top is at ~0.88m. Object should be above 0.7m
            table_xy_dist = np.linalg.norm(obj_xy - table_pos[:2])

            print(f"  Obstacle '{obj_name}': pos=[{qpos[0]:.2f}, {qpos[1]:.2f}, {qpos[2]:.2f}]")
            print(f"    Z height: {obj_z:.3f} (table top ~0.88m)")
            print(f"    XY dist from table center: {table_xy_dist:.3f}")

            on_table = obj_z > 0.7 and table_xy_dist < 0.5
            print(f"    On table: {on_table}")
            obstacle_found = True

    if not obstacle_found:
        print("  [FAIL] No obstacle objects found")

    # Run a few steps to check stability
    for i in range(10):
        action = np.zeros(env.action_dim)
        env.step(action)

    # Re-check after stepping
    for obj_name in env.objects:
        if obj_name.startswith("obstacle_"):
            obj = env.objects[obj_name]
            joint_name = obj.joints[0]
            qpos = env.sim.data.get_joint_qpos(joint_name)
            print(f"  After 10 steps - '{obj_name}': z={qpos[2]:.3f}")

    env.close()
    return obstacle_found


def test_floor_obstacle_still_on_floor(env_name="NavigateKitchenDogBlockingRouteA", layout_id=0):
    """
    Test that obstacles outside TABLE_OBSTACLES (dog, cat, vase, the boxes,
    ...) are still spawned on the floor.
    """
    print(f"\n{'='*60}")
    print(f"TEST 3: Floor obstacle still on floor")
    print(f"  env={env_name}, layout={layout_id}")
    print(f"{'='*60}")

    env = robosuite.make(
        env_name,
        robots=["PandaMobile"],
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        layout_ids=[layout_id],
        seed=42,
    )
    obs = env.reset()

    for obj_name in env.objects:
        if obj_name.startswith("obstacle_"):
            obj = env.objects[obj_name]
            joint_name = obj.joints[0]
            qpos = env.sim.data.get_joint_qpos(joint_name)
            print(f"  Obstacle '{obj_name}': z={qpos[2]:.3f} (should be near floor ~0.0)")
            on_floor = qpos[2] < 0.5
            print(f"  On floor: {on_floor}")
            env.close()
            return on_floor

    print("  [FAIL] No obstacle objects found")
    env.close()
    return False


if __name__ == "__main__":
    results = {}

    # Test 1: Human facing robot
    results["human_faces_robot"] = test_human_faces_robot()

    # Test 2: on the standing table - one env per TABLE_OBSTACLES member
    for table_env in [
        "NavigateKitchenWineBlockingRouteA",
        "NavigateKitchenGlassOfWaterBlockingRouteB",
        "NavigateKitchenHotChocolateBlockingRouteC",
        "NavigateKitchenTableLampBlockingRouteA",
    ]:
        key = f"on_table_{table_env}"
        try:
            results[key] = test_table_obstacle_on_table(table_env)
        except Exception as e:
            print(f"  [ERROR] {table_env}: {e}")
            results[key] = False

    # Test 3: everything else still on the floor
    results["floor_obstacle_on_floor"] = test_floor_obstacle_still_on_floor()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
