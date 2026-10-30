"""다음 지점을 푼다. PIVOT 의 subgoal_solver.py 를 2D 로 옮긴 것.

결정 변수는 (x, y, yaw) 3D. PIVOT 의 6D 에서 z 와 roll/pitch 를 뺐다 - 바닥을 굴러가는
로봇에게는 자유도가 없다.

yaw 를 뺄까 고민했지만 넣었다: 컨트롤러가 (xy, yaw, speed) 를 소비하고 yaw 가 SR 판정의
절반(ori >= 0.8)을 정한다. solver 가 yaw 를 안 갖고 나중에 붙이면 "최적화 기반" 이라는
말이 그 축에서 공허하다.
"""
import time

import numpy as np
from scipy.optimize import dual_annealing, minimize

from . import costs
from . import interp as I


class SubgoalSolver:
    def __init__(self, cfg, bounds_xy):
        """bounds_xy: ((x_min, x_max), (y_min, y_max)) 작업공간."""
        self.cfg = dict(cfg or {})
        self.bounds = [tuple(bounds_xy[0]), tuple(bounds_xy[1]), (-np.pi, np.pi)]
        self.last_result = None

    def reset(self):
        self.last_result = None

    def solve(self, cur_pose, ctx, from_scratch=False):
        """Returns: (pose (3,), debug dict)"""
        t0 = time.time()
        ctx = dict(ctx, cur_pose=np.asarray(cur_pose, dtype=float))

        def objective(z):
            return costs.subgoal_cost(I.unnormalize_vars(z, self.bounds), ctx)

        # 초기값: 첫 solve 는 현재 자세, 이후는 직전 해. PIVOT 과 같다.
        if from_scratch or self.last_result is None:
            x0 = np.asarray(cur_pose, dtype=float)
        else:
            x0 = np.asarray(self.last_result, dtype=float)
        z0 = np.clip(I.normalize_vars(x0, self.bounds), -1.0, 1.0)
        nb = [(-1.0, 1.0)] * 3

        if from_scratch or self.last_result is None:
            res = dual_annealing(
                objective, bounds=nb, x0=z0,
                maxfun=int(self.cfg.get("sampling_maxfun", 1500)),
                no_local_search=False,
                minimizer_kwargs={"method": "SLSQP",
                                  "options": dict(self.cfg.get("minimizer_options",
                                                               {"maxiter": 100}))},
                seed=int(self.cfg.get("seed", 0)),
            )
        else:
            res = minimize(objective, x0=z0, bounds=nb, method="SLSQP",
                           options=dict(self.cfg.get("minimizer_options", {"maxiter": 100})))

        pose = I.unnormalize_vars(res.x, self.bounds)
        pose[2] = I.wrap_angle(pose[2])
        self.last_result = pose.copy()
        _, detail = costs.subgoal_cost(pose, ctx, return_detail=True)
        # 해 자체를 남긴다. 비용 항만 있으면 "어디로 가기로 했는가" 를 사후에 볼 수 없다 -
        # 단계별 subgoal 위치는 이 방식을 설명하는 그림의 재료이기도 하다.
        detail["pose"] = [float(v) for v in pose]
        detail["solve_time"] = time.time() - t0
        detail["from_scratch"] = bool(from_scratch or self.last_result is None)
        return pose, detail
