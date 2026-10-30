"""Unit smoke for the restored manipulation (3D) path + maps->trajectory.

Run inside the robocasa Docker container (needs plotly/scipy):
    docker exec robocasa-container python3 /workspace/policy/Voxposer/src/smoke_manip.py

Covers (no live env needed):
  S2  PathPlanner.optimize() 3D voxel planner
  S3  ValueMapVisualizer render_mode='scatter'
  S4  ValueMapVisualizer render_mode='volume' (+ 'auto')
  S5  run_LMP.classify_task_type navigation/manipulation
  T2/T3  build_trajectory() navigation + manipulation
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np


class Cfg(dict):
    """dict that also supports attribute access (PathPlanner reads both
    self.config.x and self.config['x'])."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def s2_planner():
    from modules.planners import PathPlanner
    cfg = Cfg(obstacle_map_gaussian_sigma=10, target_map_weight=2, obstacle_map_weight=1,
              max_steps=50, stop_threshold=0.001, max_curvature=3,
              savgol_polyorder=3, savgol_window_size=20, target_spacing=5,
              stop_criteria='no_nearby_equal', pushing_skip_per_k=5)
    # map_size must be >=50: planner's _calculate_nearby_voxel uses
    # half_size=int(2*map_size/100), which is 0 (empty neighborhood) below 50.
    S = 50
    p = PathPlanner(cfg, map_size=S)
    tgt = np.zeros((S, S, S)); tgt[S - 5, S // 2, S // 2] = 1
    obs = np.zeros((S, S, S))
    path, info = p.optimize(np.array([5, S // 2, S // 2]), tgt, obs)
    path = np.asarray(path)
    assert path.shape[0] > 0 and path.shape[1] == 3, f'bad path shape {path.shape}'
    assert not np.isnan(path).any(), 'NaN in path'
    assert 'costmap' in info, 'missing costmap in info'
    print(f'S2 planner ok: {path.shape[0]} pts, 3D')


def s3_s4_viz():
    from utils.visualizers import ValueMapVisualizer
    os.makedirs('/tmp/manip_viz', exist_ok=True)
    cm = np.zeros((20, 20, 20)); cm[5:8, 5:8, 5:8] = 1
    info = dict(planner_info=dict(costmap=cm, start_pos=np.array([2, 10, 10]),
                                  raw_target_map=cm),
                traj_world=[(np.array([0.0, 0.0, 0.9]), 0, 1)],
                start_pos_world=np.array([0.0, 0.0, 0.8]),
                targets_world=np.array([[0.1, 0.1, 0.95]]))
    for mode, label in (('scatter', 'S3'), ('volume', 'S4'), ('auto', 'S4-auto')):
        cfg = dict(save_dir='/tmp/manip_viz', quality='low', map_size=20, render_mode=mode)
        v = ValueMapVisualizer(cfg)
        assert v.render_mode == mode
        v.update_bounds(np.array([-0.5, -0.5, 0.7]), np.array([0.5, 0.5, 1.2]))
        v.visualize(info, show=False, save=True)
        print(f'{label} viz {mode} ok')
    # default (no render_mode) must be scatter
    assert ValueMapVisualizer(dict(save_dir='/tmp/manip_viz', quality='low',
                                   map_size=20)).render_mode == 'scatter'
    print('S3/S4 default render_mode=scatter ok')


def s5_classify():
    from run_LMP import classify_task_type
    nav = classify_task_type('NavigateKitchenCatBlockingRouteA')
    man = classify_task_type('PnPCounterToCab')
    assert nav == 'navigation', f'expected navigation, got {nav}'
    assert man == 'manipulation', f'expected manipulation, got {man}'
    print(f'S5 classify ok: NavigateKitchen*->{nav}, PnP*->{man}')


def t2_t3_build():
    from modules.trajectory_builder import build_trajectory

    class P:
        def navigation_optimize(self, s, a, o, object_centric=False, robot_radius_cells=0):
            return np.array([[0, 0], [1, 1]]), {'costmap': a}

        def optimize(self, s, a, o, object_centric=False):
            return np.array([[0, 0, 0], [1, 1, 1]]), {'costmap': a}

    nav = build_trajectory(P(), lambda path, *a: [(path[-1], 0, 1)],
                           np.array([0, 0]), np.zeros((2, 2)), np.zeros((2, 2)),
                           task_type='navigation')
    man = build_trajectory(P(), lambda path, *a: [(path[-1], 0, 1, 0)],
                           np.array([0, 0, 0]), np.zeros((2, 2, 2)), np.zeros((2, 2, 2)),
                           task_type='manipulation')
    assert nav['traj_world'] and 'planner_info' in nav
    assert man['traj_world'] and 'planner_info' in man
    print('T2/T3 build_trajectory ok (nav + manip)')


if __name__ == '__main__':
    s2_planner()
    s3_s4_viz()
    s5_classify()
    t2_t3_build()
    print('ALL UNIT SMOKE PASS')
