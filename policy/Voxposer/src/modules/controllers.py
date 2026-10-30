import numpy as np
from utils.utils import get_logger

logger = get_logger(__name__)


class NavigationController:
    def __init__(self, env, config):
        self.config = config
        self.env = env

    def execute(self, waypoint):
        """
        Execute a navigation waypoint.
        :param waypoint: [target_xy, target_rotation, target_velocity]
        :return: info dict with mp_info
        """
        info = dict()
        result = self.env.apply_navigation_action(waypoint)
        info['mp_info'] = result
        return info


class ManipulationController:
    """Pick-and-place controller. Moves the end-effector directly to each
    planned waypoint and sets the gripper open/closed. No MPC / pushing."""

    def __init__(self, env, config):
        self.config = config
        self.env = env

    def execute(self, movable_obs, waypoint):
        """
        :param movable_obs: observation dict of the entity being moved (the EE)
        :param waypoint: [target_xyz, target_rotation, target_velocity, target_gripper]
        :return: info dict with mp_info
        """
        info = dict()
        target_xyz, target_rotation, target_velocity, target_gripper = waypoint
        target_pose = np.concatenate([target_xyz, target_rotation])
        result = self.env.apply_action(
            np.concatenate([target_pose, [target_gripper]])
        )
        info['mp_info'] = result
        return info
