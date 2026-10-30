"""
MainDoor fixture class for room entrance doors.
"""

import os
import numpy as np
from robocasa.models.fixtures.fixture import Fixture
import robocasa


class MainDoor(Fixture):
    """
    Main entrance door fixture for kitchen/room environments.
    This door has a hinge joint that allows it to be opened/closed.

    The door model has:
    - A door frame (fixed)
    - A door panel with a hinge joint (can rotate -90 to 90 degrees)
    """

    def __init__(
        self,
        xml=None,
        name="main_door",
        pos=None,
        size=None,
        rot=None,
        *args,
        **kwargs
    ):
        # Default to the main_door model
        if xml is None:
            xml = os.path.join(
                robocasa.models.assets_root,
                "objects/lrs_objs/main_door/model.xml"
            )

        super().__init__(
            xml=xml,
            name=name,
            pos=pos,
            *args,
            **kwargs
        )

        self._door_joint_name = None
        self._door_open_threshold = 0.85  # Fraction of max range to consider "open"
        # self.set_scale_from_size(size)
        if rot is not None:
            # adjust rotation if provided
            self.set_rotation(rot) 
    def _postprocess_model(self):
        """
        Post-process the model after loading.
        Updates joint and body names with fixture prefix.
        """
        super()._postprocess_model()

        # Find and store the door joint name
        joints = self.worldbody.findall(".//joint")
        for joint in joints:
            joint_name = joint.get("name")
            if joint_name and "doorhinge" in joint_name.lower():
                self._door_joint_name = f"{self.naming_prefix}{joint_name}"
                break

    def get_door_joint_name(self):
        """Get the name of the door hinge joint."""
        return self._door_joint_name

    def get_door_state(self, env):
        """
        Get the current state of the door (how open it is).

        Args:
            env: The environment containing the simulation

        Returns:
            dict: Dictionary with door joint names as keys and normalized positions (0-1) as values.
                  0 = closed, 1 = fully open
        """
        door_state = {}

        # Find the door joint in the simulation
        try:
            # Try to find joint by constructed name
            joint_name = f"{self.naming_prefix}doorhinge"
            if joint_name in env.sim.model.joint_names:
                joint_id = env.sim.model.joint_name2id(joint_name)
                qpos_addr = env.sim.model.jnt_qposadr[joint_id]
                joint_pos = env.sim.data.qpos[qpos_addr]

                # Get joint limits
                joint_range = env.sim.model.jnt_range[joint_id]
                range_min, range_max = joint_range[0], joint_range[1]

                # Normalize to 0-1 (0 = closed at 0 degrees, 1 = fully open)
                # Door is open when rotated away from 0
                normalized_pos = abs(joint_pos) / max(abs(range_min), abs(range_max))
                door_state[joint_name] = normalized_pos
        except Exception as e:
            print(f"[MainDoor] Warning: Could not get door state: {e}")
            # Try alternative naming
            for jnt_name in env.sim.model.joint_names:
                if self.naming_prefix in jnt_name and "door" in jnt_name.lower():
                    try:
                        joint_id = env.sim.model.joint_name2id(jnt_name)
                        qpos_addr = env.sim.model.jnt_qposadr[joint_id]
                        joint_pos = env.sim.data.qpos[qpos_addr]
                        joint_range = env.sim.model.jnt_range[joint_id]
                        range_min, range_max = joint_range[0], joint_range[1]
                        normalized_pos = abs(joint_pos) / max(abs(range_min), abs(range_max))
                        door_state[jnt_name] = normalized_pos
                    except:
                        pass

        return door_state

    def set_door_state(self, min=0.0, max=0.0, env=None, rng=None):
        """
        Set the door state (how open it is).

        Args:
            min: Minimum normalized position (0-1)
            max: Maximum normalized position (0-1)
            env: The environment containing the simulation
            rng: Random number generator
        """
        if env is None:
            return

        try:
            joint_name = f"{self.naming_prefix}doorhinge"
            if joint_name in env.sim.model.joint_names:
                joint_id = env.sim.model.joint_name2id(joint_name)
                qpos_addr = env.sim.model.jnt_qposadr[joint_id]

                # Get joint limits
                joint_range = env.sim.model.jnt_range[joint_id]
                range_min, range_max = joint_range[0], joint_range[1]

                # Convert normalized position to actual angle
                if rng is not None:
                    target_norm = rng.uniform(min, max)
                else:
                    target_norm = (min + max) / 2

                # Set to positive angle (open direction)
                target_angle = target_norm * range_max
                env.sim.data.qpos[qpos_addr] = target_angle
                env.sim.forward()
        except Exception as e:
            print(f"[MainDoor] Warning: Could not set door state: {e}")

    def get_reset_regions(self, env=None):
        """
        Get reset regions for object placement.
        Main door doesn't typically have objects placed on it.
        """
        return {}

    @property
    def nat_lang(self):
        """Natural language description of the fixture."""
        return "door"
   