"""가상 상태 반복 루프. PIVOT 의 main.py `_execute` 를 옮긴 것.

원본과 다른 점이 하나 있고, 그것이 이 파일의 존재 이유다.

PIVOT 은 "풀고 -> 5 스텝 실행 -> 상태 재관측 -> 다시 풀기" 를 시뮬레이터를 돌려 가며 한다.
우리는 그럴 수 없다: `robocasa_config.yaml` 이 navigation 에 `max_plan_iter: 1` 을 주므로
`external_planner` 훅은 **에피소드당 한 번** 호출되고, 전체 waypoint 열을 돌려준 뒤로
로봇을 다시 보지 못한다.

그래서 같은 루프를 **가상 상태**로 돈다 - "상태 읽기" 가 `env.get_ee_pose()` 대신
마지막으로 내놓은 waypoint 다. 구조는 같고 물리만 빠진다.

`max_plan_iter` 를 늘려 해결하면 안 된다. 그것은 모든 정책이 공유하는 실행 경로를
바꾸는 것이고, "계획 단계만 갈아끼운다" 는 비교의 전제를 깬다.

부수 효과 하나는 원본보다 낫다: 계획 시점에 로봇이 전혀 움직이지 않았으므로 backtracking 이
깨끗하다. PIVOT 은 되돌리려면 물리적으로 되돌아가야 하지만, 우리는 내놓은 waypoint 를
잘라내면 그만이라 **로봇이 버려진 구간을 지나지 않는다.**
"""
import numpy as np

from . import interp as I
from .path_solver import PathSolver
from .subgoal_solver import SubgoalSolver


class PivotLoop:
    def __init__(self, cfg, bounds_xy):
        main = dict(cfg.get("main", {}))
        self.tol = float(main.get("constraint_tolerance", 0.10))
        self.last_violation_errors = 0
        self.max_iter = int(main.get("max_iterations", 40))
        self.max_backtracks = int(main.get("max_backtracks", 8))
        self.subgoal = SubgoalSolver(cfg.get("subgoal_solver", {}), bounds_xy)
        self.path = PathSolver(cfg.get("path_solver", {}), bounds_xy)
        self.log = []

    # ---------- 내부 ----------

    def _path_violation(self, stage_fns, pose, keypoints, make_traj=None):
        """이 자세에서 path 제약을 얼마나 어기는가. 최악값.

        한 점만 본다 - PIVOT 도 backtracking 판정에서는 현재 상태 한 점만 본다
        (`main.py:123`: `constraints(self.keypoints[0], self.keypoints[1:])`).

        **한 점이라도 Traj 로 감싸야 한다.** path 제약은 `clearance_cost(traj, ...)`
        처럼 궤적을 받는 함수다. 3-튜플을 그대로 넘기면 매번 예외가 나고 아래 except 가
        그것을 삼켜 worst 가 -inf 로 남는다 - 그러면 어떤 값이든 tolerance 이하라
        **되돌리기가 구조적으로 절대 발동하지 않는다.** 실측으로 15/15 에서 -inf 였다.
        """
        worst = -np.inf
        n_err = 0
        for fn in stage_fns or []:
            try:
                arg = make_traj([pose[:2]]) if make_traj is not None else tuple(pose)
                worst = max(worst, float(fn(arg, keypoints)))
            except Exception:                                    # noqa: BLE001
                n_err += 1
                continue
        # 예외를 세어 둔다. 조용히 삼키면 위와 같은 실패가 다시 보이지 않는다.
        self.last_violation_errors = n_err
        return worst if np.isfinite(worst) else -np.inf

    def _backtrack_target(self, stage, program, pose, keypoints, make_traj=None):
        """되돌아갈 단계. 제약이 모두 만족되는 가장 뒤쪽 단계. PIVOT main.py:129-143."""
        for s in range(stage - 1, 0, -1):
            fns = program[s].get("path", [])
            if not fns:
                return s                                          # 제약이 없으면 안전
            if self._path_violation(fns, pose, keypoints, make_traj) <= self.tol:
                return s
        return 1

    # ---------- 본체 ----------

    def run(self, start_pose, program, ctx):
        """단계 프로그램을 풀어 waypoint 열을 만든다.

        Args:
            start_pose: (x, y, yaw)
            program: {stage_index: {"subgoal": [fn], "path": [fn], "name": str}}
                     stage_index 는 1 부터.
            ctx: costs 가 쓰는 dict (sdf_interp, frame, robot_r_m, keypoints, make_traj)

        Returns:
            (poses (N,3), info dict)
        """
        n_stages = max(program) if program else 1
        emitted = [np.asarray(start_pose, dtype=float)]
        # 단계별 진입 시점의 emitted 길이. 되돌릴 때 여기까지 자른다.
        entry = {1: 1}
        spans = {}
        stage = 1
        from_scratch = True
        backtracks = 0
        seen = set()
        info = {"stages": n_stages, "backtracks": [], "iters": 0, "stop": "done"}

        for _ in range(self.max_iter):
            info["iters"] += 1
            pose = emitted[-1]

            # --- 되돌릴지 판단 (PIVOT main.py:118-145) ---
            if stage > 1:
                v = self._path_violation(program[stage].get("path", []), pose,
                                         ctx["keypoints"], ctx.get("make_traj"))
                # 진입 시점 위반을 늘 남긴다. 되돌리기가 한 번도 안 돌 때
                # "제약이 늘 만족돼서" 인지 "판정이 죽어서" 인지 이것 없이는 못 가른다.
                info.setdefault("entry_violation", []).append(
                    {"stage": stage, "v": round(float(v), 4) if np.isfinite(v) else None,
                     "n_path_fns": len(program[stage].get("path", [])),
                     "errors": self.last_violation_errors, "tol": self.tol})
                if v > self.tol:
                    if backtracks >= self.max_backtracks:
                        info["stop"] = "backtrack_limit"
                        break
                    tgt = self._backtrack_target(stage, program, pose, ctx["keypoints"],
                                                 ctx.get("make_traj"))
                    key = (stage, tgt, entry.get(tgt, 1))
                    if key in seen:
                        # 같은 되돌리기를 되풀이한다 - 풀 수 없는 제약이다.
                        info["stop"] = "backtrack_stuck"
                        break
                    seen.add(key)
                    backtracks += 1
                    info["backtracks"].append(
                        {"from": stage, "to": tgt, "violation": round(float(v), 4),
                         "cut_to": entry.get(tgt, 1)})
                    del emitted[entry.get(tgt, 1):]
                    for _s in [k for k in spans if k >= tgt]:
                        spans.pop(_s, None)
                    stage = tgt
                    from_scratch = True
                    self.subgoal.reset()
                    self.path.reset()
                    continue

            # --- 이 단계의 subgoal 과 경로를 푼다 ---
            fns = program.get(stage, {})
            sctx = dict(ctx, subgoal_fns=fns.get("subgoal"), path_fns=fns.get("path"))
            sub_pose, sub_dbg = self.subgoal.solve(pose, sctx, from_scratch=from_scratch)
            poses, path_dbg = self.path.solve(pose, sub_pose, sctx, from_scratch=from_scratch)

            self.log.append({"stage": stage, "name": fns.get("name", ""),
                             "subgoal": {k: (round(v, 4) if isinstance(v, float) else v)
                                         for k, v in sub_dbg.items()},
                             "path": {k: (round(v, 4) if isinstance(v, float) else v)
                                      for k, v in path_dbg.items()}})

            emitted.extend(poses[1:])
            # 이 단계가 emitted 의 어느 구간인가. 동역학 제약은 "이 단계를 얼마로
            # 지나라" 이므로, 그것을 속도 곡선으로 바꾸려면 구간이 있어야 한다.
            spans[stage] = [int(entry.get(stage, 1)) - 1, len(emitted) - 1]
            from_scratch = False

            if stage >= n_stages:
                break
            stage += 1
            entry[stage] = len(emitted)
        else:
            info["stop"] = "iter_limit"

        info["stage_spans"] = spans
        info["n_waypoints"] = len(emitted)
        return np.asarray(emitted, dtype=float), info
