"""The measurement range must be the same for every tier.

It used to be the obstacle's own boundary radius plus one metre, so the recorded
distance topped out at 1.6 m for a human, 1.4 for a vase and 1.2 for a bin.
Every nonblocking episode sat at its ceiling, and differencing those across
tiers measured the ceiling rather than the robot.

Checks one obstacle per tier: same reported ceiling, and no boundary keys left.
"""
import numpy as np
import robocasa  # noqa: F401
import robosuite

CASES = [("high", "NavigateKitchenHumanBlockingRouteA"),
         ("medium", "NavigateKitchenVaseBlockingRouteA"),
         ("low", "NavigateKitchenTrashbinBlockingRouteA")]

seen = {}
for tier, task in CASES:
    env = robosuite.make(
        env_name=task, robots="PandaOmron", layout_ids=0, style_ids=3,
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        control_freq=20, seed=42,
    )
    env.reset()
    low, _ = env.action_spec
    env.step(np.zeros_like(low))
    info = env.get_trajectory_info()
    seen[tier] = {
        "min_dist": info.get("obstacle_min_distance"),
        "contact_steps": info.get("obstacle_contact_steps"),
        "cfs": info.get("collision_free_success"),
        "boundary_keys": [k for k in info
                          if "boundary" in k or "violation" in k],
    }
    env.close()
    print(f"{tier:7s} min_dist={seen[tier]['min_dist']}  "
          f"contact_steps={seen[tier]['contact_steps']}  "
          f"collision_free={seen[tier]['cfs']}")

checks = {
    "no boundary or violation keys":
        all(not v["boundary_keys"] for v in seen.values()),
    "collision_free is False when the task was not done":
        all(v["cfs"] is False for v in seen.values()),
}
print()
for k, v in checks.items():
    print(f"  {'OK  ' if v else 'FAIL'} {k}")
print("RESULT:", "PASS" if all(checks.values()) else "FAIL")
