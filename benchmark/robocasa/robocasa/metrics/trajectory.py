import numpy as np


def compute_jerk(positions, dt):
    """Compute smoothness metrics from a position trajectory.

    Jerk = 3rd derivative of position w.r.t. time (finite differences).

    Args:
        positions: (T, D) array of positions.
        dt: time step in seconds (1/control_freq).

    Returns:
        dict with jerk_mean, jerk_max, jerk_rms, num_steps.
    """
    if len(positions) < 4:
        return {'jerk_mean': 0.0, 'jerk_max': 0.0, 'jerk_rms': 0.0, 'num_steps': len(positions)}

    vel = np.diff(positions, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    jerk_norms = np.linalg.norm(jerk, axis=1)

    return {
        'jerk_mean': float(jerk_norms.mean()),
        'jerk_max': float(jerk_norms.max()),
        'jerk_rms': float(np.sqrt((jerk_norms ** 2).mean())),
        'num_steps': len(positions),
    }


def compute_path_length(positions):
    """Total Euclidean path length (sum of step-by-step distances)."""
    if len(positions) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))


def compute_obstacle_contact_metrics(obstacle_distance_history,
                                     obstacle_contact_history):
    """Contact and clearance, from per-step tracking data.

    Args:
        obstacle_distance_history: list of dicts, each {obstacle_name: distance}.
        obstacle_contact_history: list of dicts, each {obstacle_name: bool}.

    Returns:
        dict with contact counts and the distance extremes.

    Boundary violation used to be counted here against a per-obstacle radius.
    It is gone: collision-free success is contact only, and SSI reads motion
    quantities, so nothing consumed the violation counts. Proximity short of
    contact is no longer measured — see ssi_config.yaml, where the distance
    indicators are disabled and say why.
    """
    if not obstacle_distance_history:
        return {
            'obstacle_min_distance': float('inf'),
            'obstacle_contact_steps': 0,
            'collision_steps': 0,
            'obstacle_contact_ratio': 0.0,
        }

    T = len(obstacle_distance_history)

    # Min distance across all steps and obstacles
    min_dists_per_step = []
    for dist_dict in obstacle_distance_history:
        if dist_dict:
            min_dists_per_step.append(min(dist_dict.values()))
        else:
            min_dists_per_step.append(float('inf'))
    min_dists_per_step = np.array(min_dists_per_step)

    # Contact steps
    contact_steps = 0
    for contact_dict in obstacle_contact_history:
        if any(contact_dict.values()):
            contact_steps += 1

    return {
        'obstacle_min_distance': float(min_dists_per_step.min()),
        'obstacle_mean_distance': float(min_dists_per_step.mean()),
        'obstacle_contact_steps': contact_steps,
        'collision_steps': contact_steps,
        'obstacle_contact_ratio': float(contact_steps / T),
    }


def compute_task_success(pos_dist, pos_threshold, ori_cos, ori_threshold):
    """Did the robot reach the goal pose? Position AND orientation.

    This is task_success and nothing else: contact is not consulted here, and
    collision-free success is built on top of this result rather than mixed
    into it.

    One scale for every target, doors included: the caller aims a door at the
    panel normal, so cos already means "facing the way out" there as it does
    everywhere else, and nothing has to pass below its threshold.

    Args:
        pos_dist: distance from robot to target position.
        pos_threshold: threshold for position success.
        ori_cos: cosine similarity for orientation check.
        ori_threshold: threshold for orientation success.

    Returns:
        dict with the two components, their thresholds, and task_success.
    """
    pos_pass = pos_dist <= pos_threshold
    ori_pass = ori_cos >= ori_threshold

    return {
        'pos_dist': float(pos_dist),
        'pos_threshold': float(pos_threshold),
        'pos_pass': bool(pos_pass),
        'ori_cos': float(ori_cos),
        'ori_threshold': float(ori_threshold),
        'ori_pass': bool(ori_pass),
        'task_success': bool(pos_pass and ori_pass),
    }
