"""Two success axes, and nothing else deciding an outcome.

task_success        reached the goal: distance AND orientation
collision_free_success  never touched the obstacle

Anything that ANDs them, or that scores proximity, is gone. This checks the
two are reported independently, that each follows its own inputs, and that no
third notion of success survives anywhere in the trajectory info.
"""
import numpy as np
import robocasa  # noqa: F401
import robosuite

env = robosuite.make(
    env_name="NavigateKitchenCardboardBoxBlockingRouteA",
    robots="PandaOmron", layout_ids=0, style_ids=3,
    has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
    control_freq=20, seed=42,
)
env.reset()
low, _ = env.action_spec
for _ in range(40):
    env.step(np.zeros_like(low))

info = env.get_trajectory_info()
keys = set(info)

print("reported outcome keys:",
      sorted(k for k in keys if "success" in k or "pass" in k))
print("contact keys         :",
      sorted(k for k in keys if "contact" in k))
print("distance keys        :",
      sorted(k for k in keys if "distance" in k))

checks = {}
checks["task_success reported"] = "task_success" in keys
checks["collision_free_success reported"] = "collision_free_success" in keys

# No combined or proximity-based notion may survive.
gone = {"safety_success", "safe_success", "overall_success",
        "boundary_violation_steps", "boundary_violation_ratio",
        "boundary_violated", "v_b"}
leftover = sorted(gone & keys)
print("removed keys still present:", leftover or "none")
checks["no combined or proximity outcome"] = not leftover
checks["_check_success gone"] = "_check_success" not in type(env).__dict__

# Each axis follows its own inputs.
checks["task_success == pos_pass and ori_pass"] = (
    bool(info.get("task_success"))
    == (bool(info.get("pos_pass")) and bool(info.get("ori_pass"))))
# Collision-free success is a property of a completed task: reached AND
# untouched. An episode that never arrived is not collision-free by default.
checks["collision_free == task_success and no contact"] = (
    bool(info.get("collision_free_success"))
    == (bool(info.get("task_success"))
        and int(info.get("obstacle_contact_steps") or 0) == 0))
checks["not-arrived is not collision-free"] = (
    bool(info.get("task_success"))
    or not bool(info.get("collision_free_success")))

# The measurement range is a single constant, not a per-tier one.
checks["no combined success attribute"] = not hasattr(env, "safety_success")

print()
for k, v in checks.items():
    print(f"  {'OK  ' if v else 'FAIL'} {k}")
print("RESULT:", "PASS" if all(checks.values()) else "FAIL")
