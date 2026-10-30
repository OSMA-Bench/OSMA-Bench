"""For each (env, layout) where wine topples in zero-action sim, build the env,
let it settle, and dump the wine bottle's contact partners + nearby fixture
positions. The expectation: a non-table fixture (counter, island) intersects
the standing_table top and pushes the bottle over.
"""
import sys
import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config
from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS  # noqa
from robocasa.models.scenes.scene_registry import LayoutType, StyleType


FAIL_CASES = [
    ("NavigateKitchenWineNonBlockingRouteC", "L_SHAPED_LARGE"),
    ("NavigateKitchenWineNonBlockingRouteD", "L_SHAPED_LARGE"),
    ("NavigateKitchenWineNonBlockingRouteC", "U_SHAPED_SMALL"),
    ("NavigateKitchenWineNonBlockingRouteC", "G_SHAPED_SMALL"),
    ("NavigateKitchenWineNonBlockingRouteB", "G_SHAPED_LARGE"),
    ("NavigateKitchenWineNonBlockingRouteC", "G_SHAPED_LARGE"),
    ("NavigateKitchenWineNonBlockingRouteD", "GALLEY"),
    ("NavigateKitchenWineNonBlockingRouteG", "GALLEY"),
]


def inspect(env_name, layout_name):
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    env = robosuite.make(
        env_name=env_name, robots="PandaOmron", controller_configs=cc,
        has_renderer=False, has_offscreen_renderer=False,
        ignore_done=True, use_object_obs=True, use_camera_obs=False,
        seed=0,
        layout_ids=[LayoutType[layout_name].value],
        style_ids=[StyleType.MODERN_1],
        translucent_robot=False, render_gpu_device_id=0,
    )
    env.reset()
    sim = env.sim
    m = sim.model
    d = sim.data

    # Standing table position
    table_body_id = None
    wine_body_id = None
    for k in range(m.nbody):
        bn = m.body_id2name(k) or ""
        if bn.startswith("standing_table") and "main" in bn:
            table_body_id = k
        if bn.startswith("obstacle_1"):
            wine_body_id = k
    table_pos = d.body_xpos[table_body_id][:2] if table_body_id is not None else None
    wine_pos = d.body_xpos[wine_body_id]
    R = np.array(d.body_xmat[wine_body_id]).reshape(3, 3)
    z_align = abs(R[2, 2])

    # Step 5 times to let any micro-collision express itself
    contacts_seen = []
    for _ in range(5):
        env.step(np.zeros_like(env.action_spec[1]))
        for ic in range(d.ncon):
            c = d.contact[ic]
            n1 = m.geom_id2name(c.geom1) or f"g{c.geom1}"
            n2 = m.geom_id2name(c.geom2) or f"g{c.geom2}"
            wine_geoms = env._get_geom_ids_by_name("obstacle_1")
            if c.geom1 in wine_geoms or c.geom2 in wine_geoms:
                other = n2 if c.geom1 in wine_geoms else n1
                contacts_seen.append(other)

    R2 = np.array(d.body_xmat[wine_body_id]).reshape(3, 3)
    z_align_after = abs(R2[2, 2])

    print(f"\n=== {env_name} / {layout_name} ===")
    if table_pos is not None:
        print(f"  standing_table xy: ({table_pos[0]:+.3f}, {table_pos[1]:+.3f})")
    print(f"  wine pos: ({wine_pos[0]:+.3f}, {wine_pos[1]:+.3f}, {wine_pos[2]:+.3f})")
    print(f"  z_align: {z_align:.3f} -> {z_align_after:.3f}")
    if contacts_seen:
        # most common partner
        from collections import Counter
        for partner, cnt in Counter(contacts_seen).most_common(8):
            print(f"  contact: {partner}  (x{cnt})")
    else:
        print("  no contacts detected during 5 steps")

    # Find nearby fixture bodies
    if table_pos is not None:
        cands = []
        for k in range(m.nbody):
            bn = m.body_id2name(k) or ""
            if bn.startswith(("standing_table", "obstacle_", "robot", "mobilebase",
                              "panda", "gripper", "pedestal", "world", "floor",
                              "wall", "ceiling")):
                continue
            xy = d.body_xpos[k][:2]
            dist = float(np.linalg.norm(xy - np.asarray(table_pos)))
            if dist < 0.7:
                cands.append((dist, bn, xy))
        cands.sort()
        print(f"  bodies within 0.7m of standing_table:")
        for dist, bn, xy in cands[:8]:
            print(f"    {dist:.3f}m  {bn}  ({xy[0]:+.3f}, {xy[1]:+.3f})")

    env.close()


def main():
    for env_name, layout in FAIL_CASES:
        try:
            inspect(env_name, layout)
        except Exception as e:
            print(f"\n=== {env_name} / {layout} ===  ERROR: {e!r}")


if __name__ == "__main__":
    sys.exit(main())
