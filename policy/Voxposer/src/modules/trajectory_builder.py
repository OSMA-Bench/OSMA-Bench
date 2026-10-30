"""maps -> trajectory 공통 빌더 (navigation/manipulation).

env rollout / LLM 없이 value map들로 trajectory를 만든다 — 테스트/viz/디버깅용.
planner와 path2traj 변환 함수를 주입받아 호출만 한다 (재구현하지 않음).
"""


def build_trajectory(planner, path2traj_fn, start_pos,
                     affordance_map, avoidance_map,
                     rotation_map=None, velocity_map=None, gripper_map=None,
                     task_type='navigation', object_centric=False,
                     robot_radius_cells=0):
    """Returns {'path', 'traj_world', 'planner_info'}.

    - navigation: planner.navigation_optimize(...) → path2traj_fn(=_path2traj_navigation)
        path2traj_fn(path, avoidance_map, rotation_map, velocity_map)
    - manipulation: planner.optimize(...) → path2traj_fn(=_path2traj)
        path2traj_fn(path, rotation_map, velocity_map, gripper_map)
    """
    if task_type == 'navigation':
        path, info = planner.navigation_optimize(
            start_pos, affordance_map, avoidance_map,
            object_centric=object_centric, robot_radius_cells=robot_radius_cells)
        traj_world = path2traj_fn(path, avoidance_map, rotation_map, velocity_map)
    elif task_type == 'manipulation':
        path, info = planner.optimize(
            start_pos, affordance_map, avoidance_map, object_centric=object_centric)
        traj_world = path2traj_fn(path, rotation_map, velocity_map, gripper_map)
    else:
        raise ValueError(f'unknown task_type: {task_type}')
    return {'path': path, 'traj_world': traj_world, 'planner_info': info}
