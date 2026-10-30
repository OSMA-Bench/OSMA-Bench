"""R2 검증: solver 4~6.

4. 제약이 없으면 subgoal 이 현재 자세 근처여야 한다 (consistency 지배).
   멀리 가면 bounds 나 정규화가 틀린 것이다.
5. progress_cost 를 주면 목표 근처 자유공간에 착지해야 한다.
6. path solver 가 제약에 따라 우회해야 한다.
"""
import os
import sys

import numpy as np

_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _H)
sys.path.insert(0, os.path.join(_H, "src"))
# keypoint_nav 는 같은 트리 안의 것을 쓴다 - 원본 checkout 을 가리키면
# 이 트리에서 고친 것이 반영되지 않는다.
sys.path.insert(0, os.path.join(_H, "lib"))

from src import sdf as S                                          # noqa: E402
from src.geometry import TopviewFrame                             # noqa: E402
from src.path_solver import PathSolver                            # noqa: E402
from src.subgoal_solver import SubgoalSolver                      # noqa: E402
import constraints as K                                           # noqa: E402

DUMP = (os.environ.get("PIVOT_FIXTURES",
        "./policy/keypoint_nav/outputs") + "/"
        "e6_20/layout0/NavigateKitchenCatBlockingRouteA/voxposer_dump.npz")
npz = np.load(DUMP, allow_pickle=True)
it = npz["iters"][0]
avoid = np.asarray(it["avoidance_map"], float)
fr = TopviewFrame.from_dump(npz)
si, _ = S.build(avoid)
start_xy = np.asarray(it["traj_world"], float)[0]
goal = np.asarray(npz["goal_xy_fixture"], float)
obst = np.asarray(npz["obstacle_xy"], float)
ws_min = np.asarray(npz["workspace_bounds_min"], float)[:2]
ws_max = np.asarray(npz["workspace_bounds_max"], float)[:2]
bxy = ((ws_min[0], ws_max[0]), (ws_min[1], ws_max[1]))
K.set_keypoints({"A": obst})
K.set_keypoint_clouds(None)
K.set_keypoint_names({})

cur = np.array([start_xy[0], start_xy[1], 0.0])
ctx = dict(sdf_interp=si, frame=fr, robot_r_m=0.15, keypoints=K.get_keypoints(),
           subgoal_fns=None, path_fns=None,
           make_traj=lambda xy: K.Traj(np.asarray(xy, float), dt=0.2))
ok = True

SG_CFG = {"sampling_maxfun": 1500, "minimizer_options": {"maxiter": 100}}
sg = SubgoalSolver(SG_CFG, bxy)
p0, d0 = sg.solve(cur, ctx, from_scratch=True)
drift = float(np.linalg.norm(p0[:2] - cur[:2]))
print(f"4. 제약 없음: 해 {np.round(p0, 2)} | 현재에서 {drift:.3f} m | {d0['solve_time']:.2f}s")
print("   비용: " + " ".join(f"{k}={v:.2f}" for k, v in d0.items()
                             if isinstance(v, float) and k != "solve_time"))
ok &= drift < 0.5

sg.reset()
ctx5 = dict(ctx, subgoal_fns=[lambda st, kps: K.progress_cost(st, goal, 0.3)])
p1, d1 = sg.solve(cur, ctx5, from_scratch=True)
dg = float(np.linalg.norm(p1[:2] - goal))
free = float(si([fr.world_to_cell([p1[:2]])[0]])[0]) > 0
print(f"5. progress:  해 {np.round(p1, 2)} | 목표까지 {dg:.3f} m | 자유공간 {free} "
      f"| {d1['solve_time']:.2f}s")
ok &= dg < 1.0 and free

PS_CFG = {"sampling_maxfun": 1500, "opt_pos_step_size": 0.60,
          "opt_interpolate_pos_step_size": 0.10, "minimizer_options": {"maxiter": 100}}
ps = PathSolver(PS_CFG, bxy)
end = np.array([goal[0], goal[1], 0.0])
res = {}
for tag, fns in (("없음", None),
                 ("clearance 1.2", [lambda tr, kps: K.clearance_cost(tr, "A", 1.2)])):
    ps.reset()
    poses, dd = ps.solve(cur, end, dict(ctx, path_fns=fns), from_scratch=True)
    near = float(np.linalg.norm(poses[:, :2] - obst, axis=1).min())
    length = float(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1).sum())
    res[tag] = (near, length)
    print(f"6. path {tag:14s}: 최근접 {near:.2f} m | 길이 {length:.2f} m "
          f"| 제어점 {dd['n_control']} | {dd['solve_time']:.2f}s")
ok &= res["clearance 1.2"][0] > res["없음"][0] + 0.3

print("\nR2", "통과" if ok else "실패")
sys.exit(0 if ok else 1)
