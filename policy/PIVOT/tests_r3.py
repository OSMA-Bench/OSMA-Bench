"""R3 검증 7: 가상 상태 루프와 backtracking.

세 가지를 본다.
  (a) 정상 2 단계는 되돌리기 없이 끝나고 waypoint 가 이어진다
  (b) stage 2 진입 자세가 제약을 어기면 stage 1 로 되돌아가고 emitted 가 잘린다
  (c) 풀 수 없는 제약이면 무한히 돌지 않고 상한 안에서 멈춘다
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
from src.loop import PivotLoop                                    # noqa: E402
import constraints as K                                           # noqa: E402

DUMP = (os.environ.get("PIVOT_FIXTURES",
        "./policy/keypoint_nav/outputs") + "/"
        "e6_20/layout0/NavigateKitchenCatBlockingRouteA/voxposer_dump.npz")
npz = np.load(DUMP, allow_pickle=True)
it = npz["iters"][0]
fr = TopviewFrame.from_dump(npz)
si, _ = S.build(np.asarray(it["avoidance_map"], float))
start_xy = np.asarray(it["traj_world"], float)[0]
goal = np.asarray(npz["goal_xy_fixture"], float)
obst = np.asarray(npz["obstacle_xy"], float)
ws_min = np.asarray(npz["workspace_bounds_min"], float)[:2]
ws_max = np.asarray(npz["workspace_bounds_max"], float)[:2]
bxy = ((ws_min[0], ws_max[0]), (ws_min[1], ws_max[1]))
K.set_keypoints({"A": obst})
K.set_keypoint_clouds(None)
K.set_keypoint_names({})

CFG = {"main": {"constraint_tolerance": 0.10, "max_iterations": 40, "max_backtracks": 8},
       "subgoal_solver": {"sampling_maxfun": 800, "minimizer_options": {"maxiter": 60}},
       "path_solver": {"sampling_maxfun": 800, "opt_pos_step_size": 0.60,
                       "opt_interpolate_pos_step_size": 0.15,
                       "minimizer_options": {"maxiter": 60}}}
ctx = dict(sdf_interp=si, frame=fr, robot_r_m=0.15, keypoints=K.get_keypoints(),
           make_traj=lambda xy: K.Traj(np.asarray(xy, float), dt=0.2))
start = np.array([start_xy[0], start_xy[1], 0.0])
mid = (start_xy + goal) / 2.0
ok = True

# (a) 정상 2 단계
prog_ok = {
    1: {"name": "중간까지", "subgoal": [lambda st, kps: K.progress_cost(st, mid, 0.4)], "path": []},
    2: {"name": "목표까지", "subgoal": [lambda st, kps: K.progress_cost(st, goal, 0.3)], "path": []},
}
loop = PivotLoop(CFG, bxy)
poses, info = loop.run(start, prog_ok, ctx)
gap = float(np.abs(np.diff(poses[:, :2], axis=0)).max())
print(f"(a) 정상 2 단계: waypoint {len(poses)} | 되돌리기 {len(info['backtracks'])} "
      f"| 끝 {np.round(poses[-1][:2], 2)} | 목표까지 {np.linalg.norm(poses[-1][:2] - goal):.2f} m "
      f"| 최대 간격 {gap:.2f} m | stop={info['stop']}")
ok &= info["stop"] == "done" and len(info["backtracks"]) == 0 and len(poses) > 5

# (b) stage 2 의 path 제약을 stage 2 진입점에서 어기게 만든다.
#     "시작점에서 3.5m 이상 떨어져 있어라" 를 stage 2 에 건다 - stage 1 이 중간
#     (시작에서 약 2.4m)까지만 가므로 진입 시점에 이 제약이 깨진다.
#     헬퍼를 합성하지 않고 직접 쓴다 - progress_cost 를 뒤집어 만들려다
#     부호가 반대인 값을 만들어 위반이 아예 안 났다.
# path 제약은 **Traj 를 받는다** - 비용 함수(costs.py)도 make_traj 로 감싸 넘긴다.
# 처음에는 pose 배열을 받게 써 놓았는데, 그것이 실제 계약과 달라 실행 중에는 매번
# 예외가 났고 loop 의 except 가 그것을 삼켜 위반이 -inf 로 남았다. 즉 이 시험이
# 버그를 인코딩하고 있었다 - 되돌리기는 실제 15 건에서 한 번도 발동하지 못했다.
far = [lambda traj, kps: 3.5 - float(np.linalg.norm(
    np.asarray(traj.p, float).reshape(-1, 2)[-1] - start_xy))]
prog_bt = {
    1: {"name": "중간까지", "subgoal": [lambda st, kps: K.progress_cost(st, mid, 0.4)], "path": []},
    2: {"name": "위반 단계", "subgoal": [lambda st, kps: K.progress_cost(st, goal, 0.3)],
        "path": far},
}
loop_b = PivotLoop(CFG, bxy)
poses_b, info_b = loop_b.run(start, prog_bt, ctx)
print(f"(b) 되돌리기: {len(info_b['backtracks'])} 회 | stop={info_b['stop']} "
      f"| waypoint {len(poses_b)} | iters {info_b['iters']}")
for b in info_b["backtracks"][:3]:
    print(f"    stage {b['from']} -> {b['to']}, 위반 {b['violation']}, {b['cut_to']} 까지 자름")
ok &= len(info_b["backtracks"]) > 0
ok &= info_b["stop"] in ("backtrack_stuck", "backtrack_limit", "done")
ok &= info_b["iters"] <= CFG["main"]["max_iterations"]

# (c) 어떤 단계에서도 만족 못 하는 제약 - 상한 안에 멈추는지
never = [lambda st, kps: 999.0]
prog_bad = {
    1: {"name": "s1", "subgoal": [lambda st, kps: K.progress_cost(st, mid, 0.4)], "path": []},
    2: {"name": "s2", "subgoal": [lambda st, kps: K.progress_cost(st, goal, 0.3)], "path": never},
}
loop_c = PivotLoop(CFG, bxy)
poses_c, info_c = loop_c.run(start, prog_bad, ctx)
print(f"(c) 불가능 제약: stop={info_c['stop']} | 되돌리기 {len(info_c['backtracks'])} "
      f"| iters {info_c['iters']} (상한 {CFG['main']['max_iterations']})")
ok &= info_c["stop"] in ("backtrack_stuck", "backtrack_limit")
ok &= info_c["iters"] < CFG["main"]["max_iterations"]

print("\nR3", "통과" if ok else "실패")
sys.exit(0 if ok else 1)
