"""
Dummy task: render a kitchen layout with the robot and posed_human hidden.

Used purely for visualizing layouts (no objects, no goal, no contact).
"""
from robocasa.environments.kitchen.kitchen import *


class KitchenLayoutView(Kitchen):
    """
    Loads the scene and hides robot + posed_human so only the layout is visible.
    """

    # Where the hidden robot / human are parked. Far enough below the floor to
    # be outside any topview frustum, but not so far that MuJoCo warns about
    # huge coordinates.
    HIDDEN_Z = -100.0

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        try:
            self.human = self.register_fixture_ref(
                "posed_human", dict(id="posed_human")
            )
            # Move human far below the floor as a backup to the visual hide
            self.human.set_pos([0.0, 0.0, self.HIDDEN_Z])
        except Exception:
            self.human = None

    def _load_model(self, _retry_count=0):
        super()._load_model(_retry_count=_retry_count)
        # Park the robot below the floor as a backup to the visual hide.
        #
        # This has to happen here, not in _setup_kitchen_references: Kitchen
        # calls _setup_kitchen_references() and *then* positions the robot via
        # robot_model.set_base_xpos(), so anything set earlier is overwritten.
        # (The previous attempt did it there AND called a `set_base_pos` that
        # does not exist on WheeledRobot, so it never ran at all.)
        #
        # set_base_xpos edits the MJCF root_body element, which is why this
        # must land after super()._load_model() built the tree but before
        # _initialize_sim() compiles it.
        self.robots[0].robot_model.set_base_xpos([0.0, 0.0, self.HIDDEN_Z])

    def _get_obj_cfgs(self):
        return []

    def _check_success(self):
        return False

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "kitchen layout visualization"
        return ep_meta

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)

        # Cover arm (robot0_), mobile base (mobilebase0_), and gripper (gripper0_*).
        # Each has its own naming prefix; robot_model.visual_geoms only lists
        # arm geoms, so the parent visualize() leaves base + gripper visible.
        robot_prefixes = []
        for idx, robot in enumerate(self.robots):
            pfx = robot.robot_model.naming_prefix
            robot_prefixes.append(pfx)
            i = pfx.replace("robot", "").rstrip("_")
            robot_prefixes.append(f"mobilebase{i}_")
            robot_prefixes.append(f"gripper{idx}_")

        for gid in range(self.sim.model.ngeom):
            name = self.sim.model.geom_id2name(gid) or ""
            if any(name.startswith(p) for p in robot_prefixes):
                self.sim.model.geom_rgba[gid][-1] = 0.0
            elif "posed_human" in name:
                self.sim.model.geom_rgba[gid][-1] = 0.0
