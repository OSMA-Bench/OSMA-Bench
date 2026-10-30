"""시작에서 subgoal 까지의 경로를 푼다. PIVOT 의 path_solver.py 를 2D 로.

결정 변수는 중간 제어점들 (x, y, yaw). 시작과 끝은 고정이다.
제어점 개수는 거리에 따라 3~6 개(시작·끝 포함), 즉 중간 1~4 개.
"""
import time

import numpy as np
from scipy.optimize import dual_annealing, minimize

from . import costs
from . import interp as I


class PathSolver:
    def __init__(self, cfg, bounds_xy):
        self.cfg = dict(cfg or {})
        self.bxy = [tuple(bounds_xy[0]), tuple(bounds_xy[1])]
        self.last_result = None

    def reset(self):
        self.last_result = None

    def _dense(self, start, end, mid_flat, n_dense):
        mid = np.asarray(mid_flat, dtype=float).reshape(-1, 3)
        ctrl = np.concatenate([np.asarray(start)[None, :], mid, np.asarray(end)[None, :]])
        return I.interpolate(ctrl, n_dense)

    def solve(self, start_pose, end_pose, ctx, from_scratch=False):
        """Returns: (poses (N,3) 조밀 열, debug dict)"""
        t0 = time.time()
        start = np.asarray(start_pose, dtype=float)
        end = np.asarray(end_pose, dtype=float)

        n_ctrl = I.num_control_points(
            start, end,
            float(self.cfg.get("opt_pos_step_size", 0.60)),
            float(self.cfg.get("opt_rot_step_size", 0.78)))
        n_mid = max(n_ctrl - 2, 0)
        # 충돌·제약을 검사할 조밀도. PIVOT 의 opt_interpolate_pos_step_size 와 같은 뜻.
        span = float(np.linalg.norm(end[:2] - start[:2]))
        n_dense = int(np.clip(
            round(span / max(float(self.cfg.get("opt_interpolate_pos_step_size", 0.10)), 1e-6)),
            8, 60))

        if n_mid == 0:
            poses = I.interpolate(np.stack([start, end]), n_dense)
            _, detail = costs.path_cost(poses, ctx, return_detail=True)
            detail.update(solve_time=time.time() - t0, n_control=n_ctrl, n_dense=n_dense,
                          from_scratch=bool(from_scratch))
            return poses, detail

        bounds = [self.bxy[0], self.bxy[1], (-np.pi, np.pi)] * n_mid

        def objective(z):
            return costs.path_cost(self._dense(start, end, I.unnormalize_vars(z, bounds),
                                               n_dense), ctx)

        # 초기값: 시작-끝 직선 위의 균등점. 직전 해가 있고 크기가 같으면 그것을 쓴다.
        lin = I.interpolate(np.stack([start, end]), n_mid + 2)[1:-1]
        x0 = lin.reshape(-1)
        if (not from_scratch) and self.last_result is not None \
                and len(self.last_result) == len(x0):
            x0 = np.asarray(self.last_result, dtype=float)
        z0 = np.clip(I.normalize_vars(x0, bounds), -1.0, 1.0)
        nb = [(-1.0, 1.0)] * len(bounds)

        if from_scratch or self.last_result is None or len(self.last_result) != len(x0):
            res = dual_annealing(
                objective, bounds=nb, x0=z0,
                maxfun=int(self.cfg.get("sampling_maxfun", 1500)),
                # PIVOT 은 path solver 만 no_local_search=True 를 쓴다. 제어점이 많아
                # 국소 탐색이 비싸고, 전역 표본이 이미 충분하기 때문이다.
                no_local_search=True,
                minimizer_kwargs={"method": "SLSQP",
                                  "options": dict(self.cfg.get("minimizer_options",
                                                               {"maxiter": 100}))},
                seed=int(self.cfg.get("seed", 0)),
            )
        else:
            res = minimize(objective, x0=z0, bounds=nb, method="SLSQP",
                           options=dict(self.cfg.get("minimizer_options", {"maxiter": 100})))

        mid = I.unnormalize_vars(res.x, bounds)
        self.last_result = mid.copy()
        poses = self._dense(start, end, mid, n_dense)
        _, detail = costs.path_cost(poses, ctx, return_detail=True)
        detail.update(solve_time=time.time() - t0, n_control=n_ctrl, n_dense=n_dense,
                      from_scratch=bool(from_scratch))
        return poses, detail
