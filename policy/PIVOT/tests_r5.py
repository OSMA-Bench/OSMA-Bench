"""R5 검증 9(사전): 훅을 가짜 lmp_env 로 돌려 반환 계약을 확인한다.

시뮬레이터를 띄우기 전에 형태를 잡는다. 여기서 잡히는 종류의 버그(반환 키 이름,
좌표 공간, 예외 전파)는 컨테이너 안에서는 로그 한 줄로만 보이고 원인을 찾기 어렵다.
"""
import os
import sys

import numpy as np

_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _H)
sys.path.insert(0, os.path.join(_H, "src"))
sys.path.insert(0, os.path.join(_H, "lib"))

from adapters.planner_hook import PivotPlannerHook           # noqa: E402
from src.geometry import TopviewFrame                        # noqa: E402

DUMP = (os.environ.get("PIVOT_FIXTURES",
        "./policy/keypoint_nav/outputs") + "/"
        "e6_20/layout0/NavigateKitchenCatBlockingRouteA/voxposer_dump.npz")
npz = np.load(DUMP, allow_pickle=True)
it = npz["iters"][0]
avoid = np.asarray(it["avoidance_map"], float)
goal_xy = np.asarray(npz["goal_xy_fixture"], float)
obst = np.asarray(npz["obstacle_xy"], float)
ws_min = np.asarray(npz["workspace_bounds_min"], float)
ws_max = np.asarray(npz["workspace_bounds_max"], float)
fr = TopviewFrame.from_dump(npz)
start_cell = np.round(fr.world_to_cell([np.asarray(it["traj_world"], float)[0]])[0])

# affordance_map: 목표 칸 주변을 1 로
aff = np.zeros_like(avoid)
gc = np.round(fr.world_to_cell([goal_xy])[0]).astype(int)
aff[max(gc[0] - 2, 0):gc[0] + 3, max(gc[1] - 2, 0):gc[1] + 3] = 1.0


class FakeInner:
    workspace_bounds_min = ws_min
    workspace_bounds_max = ws_max
    robot_yaw = 0.0
    env = None


class FakeEnv:
    """detect() 만 흉내 낸다. keypoints.collect 가 쓰는 것이 그것뿐이다."""

    _env = FakeInner()
    _task_dir = os.path.join(_H, "_r5_out")

    # 그림에 보이는 것과 목록이 맞아야 한다. 사람이 서 있는 장면인데 목록에서
    # 빼 두었더니 VLM 이 쓴 제약이 전부 "없는 이름" 으로 거절됐다 - 실제 훅에서는
    # KPMOD.collect 가 환경 목록을 그대로 받으므로 생기지 않는 상황이다.
    OBJS = {"cat": obst, "sink": np.array([2.68, -0.30]),
            "stove": np.array([4.30, -0.35]), "fridge": np.array([0.60, -0.30]),
            "microwave": np.array([3.0, 0.2]), "human": np.array([2.1, 0.6])}

    def get_visible_object_names(self):
        return list(self.OBJS) + ["kitchen", "robot_mobile_base"]

    def detect(self, name):
        p = self.OBJS.get(str(name).split("#")[0])
        if p is None:
            return {}
        return {"_position_world": np.array([p[0], p[1], 0.0])}


CFG = {"main": {"constraint_tolerance": 0.10, "max_iterations": 40, "max_backtracks": 8},
       "subgoal_solver": {"sampling_maxfun": 800, "minimizer_options": {"maxiter": 60}},
       "path_solver": {"sampling_maxfun": 800, "opt_pos_step_size": 0.60,
                       "opt_interpolate_pos_step_size": 0.15,
                       "minimizer_options": {"maxiter": 60}},
       "model": {"name": "Qwen/Qwen3-VL-8B-Instruct",
                 "base_url": "http://localhost:8003/v1",
                 "temperature": 0.0, "max_tokens": 900}}

hook = PivotPlannerHook(CFG, prompts_dir=os.path.join(_H, "prompts"))
env = FakeEnv()
res = hook(env, start_pos=start_cell, affordance_map=aff, avoidance_map=avoid,
           robot_radius_cells=3)

ok = True
print(f"반환 키: {sorted(res)}")
ok &= set(res) == {"path", "planner_info", "traj_world"}

path = np.asarray(res["path"])
print(f"path: {path.shape} | 비어있지 않음 {len(path) > 0}")
ok &= len(path) > 0

tw = res["traj_world"]
xy = np.asarray([t[0] for t in tw], dtype=float)
inside = bool((xy[:, 0] >= ws_min[0] - 0.2).all() and (xy[:, 0] <= ws_max[0] + 0.2).all()
              and (xy[:, 1] >= ws_min[1] - 0.2).all() and (xy[:, 1] <= ws_max[1] + 0.2).all())
d_goal = float(np.linalg.norm(xy[-1] - goal_xy))
near = float(np.linalg.norm(xy - obst, axis=1).min())
print(f"traj_world: {len(tw)} 점 | 작업공간 안 {inside} | 끝→목표 {d_goal:.3f} m "
      f"| 장애물 최근접 {near:.2f} m")
print(f"planner_info: {res['planner_info']}")
ok &= inside and d_goal < 0.05 and len(tw) > 3

# 삼중항 모양
t0 = tw[0]
shape_ok = (len(t0) == 3 and np.asarray(t0[0]).shape == (2,)
            and isinstance(t0[1], float) and isinstance(t0[2], float))
print(f"삼중항 (xy, yaw, speed): {shape_ok}")
ok &= shape_ok

log = os.path.join(FakeEnv._task_dir, "pivot_log.json")
print(f"pivot_log.json: {os.path.exists(log)}")
ok &= os.path.exists(log)
if os.path.exists(log):
    import json
    doc = json.load(open(log))
    kinds = [e.get("stage") for e in doc["log"]]
    print(f"  로그 항목: {kinds}")
    solves = [e for e in doc["log"] if e.get("stage") == "solve"]
    if solves:
        s0 = solves[0]
        print(f"  첫 solve 비용: subgoal={ {k: v for k, v in s0['subgoal'].items() if isinstance(v, float)} }")

print("\nR5(사전)", "통과" if ok else "실패")
sys.exit(0 if ok else 1)
