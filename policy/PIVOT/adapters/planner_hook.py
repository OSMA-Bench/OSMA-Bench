"""`external_planner` 훅. PIVOT 방식 계획을 VoxPoser 실행 경로에 끼운다.

`keypoint_nav` 훅과 같은 자리에 같은 계약으로 붙는다 - 환경 생성·에피소드 루프·컨트롤러·
평가·판정문은 VoxPoser 의 `run_tasks` 를 그대로 공유한다. 세 정책이 같은 잣대로 채점되어야
비교가 성립하고, 평가를 복제하면 임계가 조용히 어긋난다.

이 훅이 keypoint_nav 훅과 다른 점은 하나다: **VLM 에게 경로를 묻지 않는다.**
단계 분해와 제약만 받고, 경로는 solver 가 푼다.
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_KN = _ROOT
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "lib"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import constraint_code as CC                             # noqa: E402
import constraints as K                                  # noqa: E402
import keypoints as KPMOD                                # noqa: E402
from annotate import annotate                            # noqa: E402
from geometry import TopviewFrame                        # noqa: E402
from vlm import VLMClient                                # noqa: E402

from src import sdf as SDF                               # noqa: E402
from src import pivot as PIVOT                            # noqa: E402
from src import stops as STOPS                            # noqa: E402
from src import stages as STG                            # noqa: E402
from src.loop import PivotLoop                           # noqa: E402
from src.path_solver import PathSolver                   # noqa: E402


class PivotPlannerHook:
    """VoxPoser 의 `run_tasks(external_planner=...)` 로 넘길 호출 가능 객체."""

    def __init__(self, cfg, prompts_dir=None, out_root=None):
        self.cfg = dict(cfg or {})
        self.prompts_dir = prompts_dir or os.path.join(_ROOT, "prompts")
        self.out_root = out_root
        # v/a/J 를 켤지. 끄면 프롬프트에서 문단이 빠지고 어휘도 막히므로,
        # 두 설정의 차이가 동역학 하나로만 남는다 (ablation).
        self.dynamics = bool(self.cfg.get("main", {}).get("dynamics", False))
        # 단계 끝점을 누가 정하는가: "solver"(기본) 또는 "pivot".
        # pivot 이면 반복 시각 질의로 점을 찍고, finalize 가 주입하는 progress_cost 의
        # **대상만** 목표에서 그 점으로 바뀐다 - solver 구조는 그대로다.
        self.subgoals = str(self.cfg.get("main", {}).get("subgoals", "solver")).lower()
        # 한 곳에서 나온 시드를 VLM · pivot · 두 solver 가 나눠 쓴다. 갈래마다
        # 따로 두면 "무엇을 바꿔서 결과가 달라졌는가" 를 다시 물을 수 없다.
        self.seed = self.cfg.get("main", {}).get("seed", 0)
        m = dict(self.cfg.get("model", {}))
        self.client = VLMClient(
            m.get("name", "Qwen/Qwen3-VL-8B-Instruct"),
            base_url=m.get("base_url", "http://localhost:8000/v1"),
            temperature=float(m.get("temperature", 0.0)),
            max_tokens=int(m.get("max_tokens", 900)),
            seed=self.seed)
        self.log = []

    # ---------- 내부 ----------

    def _corners_uv(self, lmp_env, ws_min, ws_max):
        """topview 카메라로 워크스페이스 네 모서리를 바닥 평면에 투영한다.

        keypoint_nav 훅과 **같은 계산**이다. 처음에는 `world_to_topview_uv` 라는
        메서드를 찾고 없으면 이미지 네 귀퉁이로 대체했는데, 그런 메서드는 없으므로
        늘 대체값이 쓰였다. 실제 투영은 u∈[147,493] 인데 [0,640] 으로 잡으니
        **그림 위의 표시가 전부 어긋난 자리에 찍혔다** - 로봇 고리가 작업공간 밖
        (578, 407) 에 그려지고, 고양이 마름모가 바닥의 고양이가 아니라 조리대 위에
        놓였다. VLM 은 그 그림을 보고 답해 왔다.

        계획에는 영향이 없다: `cell_to_world`/`world_to_cell` 은 작업공간 경계와
        격자 크기만 쓰고 이 투영을 쓰지 않는다. 어긋난 것은 그림뿐이다 -
        그래서 아무 오류 없이 지나갔다.
        """
        try:
            sim = lmp_env._env.env.sim
            cid = sim.model.camera_name2id("topview")
            cam_pos = sim.data.cam_xpos[cid].copy()
            cam_mat = sim.data.cam_xmat[cid].reshape(3, 3).copy()
            fovy = float(sim.model.cam_fovy[cid])
            W, H = int(lmp_env._env.cam_width), int(lmp_env._env.cam_height)
            fy = (H / 2.0) / np.tan(np.radians(fovy / 2.0))
            fx = fy
            # 바닥 z 는 바닥 geom 에서 읽는다. ws_min[2] 는 바디 bbox 하단이라
            # 지면보다 한참 아래일 수 있다.
            floor_z = 0.0
            fids = getattr(lmp_env._env, "floor_mask_ids", [])
            if len(fids):
                floor_z = float(sim.data.geom_xpos[fids[0], 2])
            corners_world = np.array([
                [ws_min[0], ws_min[1], floor_z], [ws_min[0], ws_max[1], floor_z],
                [ws_max[0], ws_max[1], floor_z], [ws_max[0], ws_min[1], floor_z]])
            out = []
            for w in corners_world:
                cam_frame = cam_mat.T @ (w - cam_pos)
                depth = -cam_frame[2]
                out.append([(W / 2.0) + fx * (cam_frame[0] / depth),
                            (H / 2.0) - fy * (cam_frame[1] / depth)])
            return np.asarray(out)
        except Exception as e:                           # noqa: BLE001
            # 대체값을 쓰면 그림이 조용히 거짓말을 한다. 반드시 로그에 남긴다.
            self.log.append({"stage": "corners_fallback", "msg": str(e)[:140]})
            return np.array([[0.0, 0.0], [0.0, 480.0], [640.0, 480.0], [640.0, 0.0]])

    @staticmethod
    def _topview(lmp_env):
        """현재 topview RGB. keypoint_nav 훅과 **같은 방법**을 쓴다.

        처음에는 `get_topview_image` / `topview_image` 라는 이름의 메서드를 찾았는데
        환경에 그런 것이 없다. 둘 다 없으면 조용히 None 이 되고, 훅은 이미지 없이
        질의를 보낸다 - 예외도 로그도 남지 않는다. 그렇게 15 건 전부 **VLM 이 장면을
        보지 못한 채** 물체 목록만으로 단계를 나눴다. 단계 수가 늘 3 이던 것도
        이것으로 설명된다: 목록은 에피소드마다 거의 같다.

        직접 렌더를 요청하면 두 번째 오프스크린 컨텍스트를 만들려다 EGL 이 깨지므로
        환경이 이미 렌더해 둔 `latest_obs` 에서 꺼낸다. MuJoCo 관측 이미지는 상하가
        뒤집혀 있어 뒤집어야 모서리 투영과 방향이 맞는다.
        """
        env = getattr(lmp_env, "_env", None)
        if env is None:
            return None
        try:
            obs = getattr(env, "latest_obs", None)
            if not isinstance(obs, dict) or "topview_image" not in obs:
                env.update_latest_obs()
                obs = env.latest_obs
            img = np.asarray(obs["topview_image"])
            return img[::-1] if img.ndim == 3 else img
        except Exception:                                # noqa: BLE001
            return None

    def _ask(self, image, kps, robot_xy, n_stops=0):
        """단계 분해와 제약을 한 번 묻는다. 실패해도 예외를 내지 않는다."""
        # dynamics 를 함께 남긴다. 이것이 없으면 "모델이 v/a/J 를 안 썼다" 와
        # "블록이 애초에 프롬프트에 없었다" 가 로그에서 구별되지 않는다 - 그 둘은
        # 정반대의 결론이고, 실제로 한 번 잘못 보고했다.
        info = {"asked": True, "dynamics": bool(self.dynamics)}
        try:
            prompt = STG.load_prompt(self.prompts_dir, dynamics=self.dynamics).format(
                objects=KPMOD.as_table(kps, robot_xy))
            info["prompt_chars"] = len(prompt)
            info["pace_offered"] = "speed_cost" in prompt
            text = self.client.ask(prompt, image)
        except Exception as e:                           # noqa: BLE001
            return {}, {"asked": False, "error": str(e)[:160]}
        code = CC.extract(text)
        if not code:
            info.update(rejected="코드 없음", reply=text[-240:])
            return {}, info
        allowed = ({k["label"] for k in kps} | {k["name"] for k in kps})
        prog, pinfo = STG.parse(code, allowed, dynamics=self.dynamics)
        info.update(pinfo)
        info["src"] = code[:900]
        # VLM 이 고른 값. 어느 대상에 몇 미터를 요구했는지가 이 방식의 측정 대상이다.
        try:
            info["margins"] = CC.margins_used(code)
        except Exception:                                # noqa: BLE001
            pass
        if pinfo.get("rejected"):
            info["reply"] = text[-240:]
        return prog, info

    # ---------- 본체 ----------

    def __call__(self, lmp_env, start_pos, affordance_map, avoidance_map,
                 rotation_map=None, velocity_map=None, robot_radius_cells=0):
        self.log = []
        ws_min = np.asarray(lmp_env._env.workspace_bounds_min, dtype=float)
        ws_max = np.asarray(lmp_env._env.workspace_bounds_max, dtype=float)
        avoid = np.asarray(getattr(avoidance_map, "array", avoidance_map), dtype=float)
        frame = TopviewFrame(self._corners_uv(lmp_env, ws_min, ws_max),
                             ws_min, ws_max, avoid.shape)

        yx = np.argwhere(np.asarray(affordance_map) > 0.5)
        goal_cell = yx.mean(axis=0) if len(yx) else np.asarray(start_pos, float)
        goal_xy = frame.cell_to_world([goal_cell])[0]
        robot_xy = frame.cell_to_world([np.asarray(start_pos, float)])[0]
        robot_yaw = float(getattr(lmp_env._env, "robot_yaw", 0.0) or 0.0)

        # 에피소드 디렉터리. `_task_dir` 은 실제 환경에 없는 이름이라 늘 None 이었고,
        # 로그가 실행 루트 한 곳에 덮어써져 15 건을 돌려도 마지막 한 건만 남았다.
        # keypoint_nav 훅과 같은 속성을 쓴다 - voxposer_dump.npz 가 놓이는 자리다.
        _od = getattr(lmp_env, "_output_dir", None)
        out_dir = _od if _od and os.path.isdir(_od) else (
            getattr(lmp_env, "_task_dir", None) or self.out_root)

        # --- 키포인트: keypoint_nav 와 같은 목록을 쓴다 ---
        # 두 정책이 같은 정보를 받아야 "그것을 어떻게 쓰는가" 만 남는다.
        kps, skipped = KPMOD.collect(lmp_env, goal_xy=goal_xy)
        K.set_keypoints(KPMOD.as_mapping(kps))
        K.set_keypoint_names({k["label"]: k["name"] for k in kps})
        K.set_keypoint_aliases({k["label"]: k["name"] for k in kps})
        K.set_keypoint_clouds(None)
        # 좌표까지 남긴다. 표만 있으면 "무엇을 지목했나" 는 알아도 "그것이 어디였나" 를
        # 그림으로 확인할 수 없다 - 지목한 대상과 실제로 지나간 자리를 겹쳐 보는 것이
        # 이 방식을 검증하는 가장 직접적인 수단이다.
        self.log.append({"stage": "keypoints",
                         "table": {k["label"]: k["name"] for k in kps},
                         "xy": {k["name"]: [float(k["xy"][0]), float(k["xy"][1])]
                                for k in kps},
                         "skipped": skipped})

        # --- 질의: 단계 분해 + 제약 ---
        # SDF 는 질의보다 **먼저** 만든다. PIVOT 이 후보를 뽑을 때 자유공간
        # 판정이 필요하고, 그 판정은 solver 가 쓰는 것과 같아야 한다 -
        # 두 곳이 다른 기준을 쓰면 '고를 수는 있는데 풀 수 없는' 점이 나온다.
        si, _ = SDF.build(avoid)
        robot_r_m = max(float(robot_radius_cells) * SDF.CELL_M, 0.15)

        # "지날 수 없다" 를 선언할 반경. 출발 위치의 실제 여유에서 정한다.
        # 로봇 반경을 그대로 쓰면 출발점부터 불가능해지고(이 맵은 보수적이라 출발
        # 여유가 0.06~0.22 m 인 곳이 있다), 0 으로 두면 최소 여유 0.06 m 짜리
        # 경로가 나와 몸통이 벽에 걸린다. 출발점이 증명한 만큼만 요구한다.
        start_clear = float(np.asarray(si(np.atleast_2d(
            frame.world_to_cell([robot_xy])))).ravel()[0])
        infeasible_r = float(np.clip(min(robot_r_m, start_clear - 0.02), 0.0, robot_r_m))
        self.log.append({"stage": "infeasible_radius",
                         "start_clearance": round(start_clear, 3),
                         "robot_r_m": round(robot_r_m, 3),
                         "used": round(infeasible_r, 3)})

        topview = self._topview(lmp_env)
        # 정차 지점 후보. 화면 전체에서 한 번에 뽑아 번호를 붙인다 - VLM 이 한 응답
        # 안에서 단계 수·정차 지점·제약을 모두 말하게 하려면 후보가 그림에 있어야 한다.
        stop_cells, stop_xy = STOPS.propose(
            avoid, frame, robot_xy, goal_xy,
            max_n=int(self.cfg.get("main", {}).get("stop_candidates", 14)),
            inflate_cells=robot_radius_cells or None)
        # 좌표까지 남긴다. 개수만 남기면 "VLM 이 고른 번호 -> 어느 좌표" 를 사후에
        # 검증할 수 없다 - 실제로 재현 계산과 로그가 어긋났는데 어느 쪽이 맞는지
        # 가리지 못했다. 번호는 1 부터이고 이 목록의 순서와 같다.
        self.log.append({"stage": "stops", "n": int(len(stop_cells)),
                         "xy": [[round(float(a), 3), round(float(b), 3)]
                                for a, b in np.asarray(stop_xy, float)]})

        img = None
        if topview is not None:
            try:
                img, _ = annotate(topview, frame, stop_cells,
                                  goal_xy=goal_xy, robot_xy=robot_xy, robot_yaw=robot_yaw,
                                  radius_px=7, label_pt=12, keypoints=kps)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                    img.save(os.path.join(out_dir, "pivot_query.png"))
            except Exception as e:                       # noqa: BLE001
                self.log.append({"stage": "annotate_error", "msg": str(e)[:120]})
        # 이미지 없이 물었는지를 반드시 남긴다. 조용히 None 이 되면 "VLM 이 장면을 보고
        # 나눴다" 가 사실이 아닌 채로 결과만 그럴듯하게 나온다 - 실제로 그렇게 15 건을
        # 돌렸다.
        self.log.append({"stage": "query_image", "has_image": img is not None})
        program, qinfo = self._ask(img, kps, robot_xy)

        # --- 단계 끝점: VLM 이 고른 번호를 좌표로 되돌린다 ---
        # 구조는 그대로다. finalize 가 주입하던 progress_cost 의 **대상만** 목표에서
        # 이 점으로 바뀐다 - 끝점이 "목표 주위 고리" 에서 "찍은 한 점" 이 된다.
        targets = None
        # 단계 수는 **파싱된 program** 에서 센다. qinfo["stages"] 는 finalize 가
        # 나중에 채우므로 여기서는 늘 None 이고, 그러면 n=1 로 읽혀 이 블록이
        # 통째로 건너뛰어진다 - 실측으로 pivot 이 한 번도 돌지 않았다.
        n_stages = max(program) if program else 1
        if self.subgoals == "stops":
            targets = self._stop_targets(qinfo, stop_xy, n_stages)
        elif self.subgoals == "pivot":
            targets = self._pivot_targets(topview, frame, si, robot_xy, goal_xy,
                                          robot_yaw, n_stages, infeasible_r,
                                          program)

        program, qinfo = STG.finalize(program, goal_xy, robot_xy, qinfo,
                                      targets=targets)
        self.log.append({"stage": "query", **{k: v for k, v in qinfo.items()
                                              if k != "src"}})

        # --- 풀기 ---

        ctx = dict(sdf_interp=si, frame=frame,
                   robot_r_m=robot_r_m, infeasible_r_m=infeasible_r,
                   keypoints=K.get_keypoints(),
                   make_traj=lambda xy: K.Traj(np.asarray(xy, dtype=float), dt=0.2))
        bxy = ((float(ws_min[0]), float(ws_max[0])), (float(ws_min[1]), float(ws_max[1])))
        loop = PivotLoop(self.cfg, bxy)
        try:
            poses, info = loop.run(np.array([robot_xy[0], robot_xy[1], robot_yaw]),
                                   program, ctx)
        except Exception as e:                           # noqa: BLE001
            self.log.append({"stage": "solve_error", "msg": str(e)[:200]})
            self._dump(out_dir, goal_xy, robot_xy)
            return {"path": np.empty((0, 2), dtype=int),
                    "planner_info": {"source": "pivot", "failure": "solve_error"},
                    "traj_world": []}

        self.log.append({"stage": "loop", **info})
        # e 에 "stage"(단계 번호)가 들어 있어 {"stage": "solve", **e} 로 쓰면
        # 번호가 이름을 덮어 로그가 읽히지 않는다. 키를 나눈다.
        self.log.extend({"stage": "solve", "idx": e.pop("stage", None), **e}
                        for e in loop.log)

        # 제자리 waypoint 를 걷어낸다. 같은 점이 연속으로 들어가면 실행 루프가
        # 그것을 "이미 도달" 로 읽어 전진 없이 소진한다 - 계획은 멀쩡한데 로봇이
        # 출발점 근처에서 멈추는 실패로 나타난다.
        keep = [0]
        for i in range(1, len(poses)):
            if float(np.linalg.norm(poses[i, :2] - poses[keep[-1], :2])) > 0.03:
                keep.append(i)
        if len(keep) < len(poses):
            self.log.append({"stage": "dedupe", "before": int(len(poses)),
                             "after": len(keep)})
        poses = poses[keep]

        # 마지막 점을 정확한 목표로.
        #
        # 목표점을 그냥 이어 붙이면 안 된다. 루프가 일찍 끊기면(backtrack_stuck,
        # backtrack_limit) 마지막 자세가 목표에서 멀고, 거기에 목표를 붙이면
        # **컨트롤러에게 순간이동인 한 구간**이 생긴다. 실측: 정상 에피소드의 최대
        # waypoint 간격은 0.11~0.30 m 인데 끊긴 에피소드만 1.7~3.2 m 짜리 구간이
        # 하나씩 있었고, 실패한 에피소드가 정확히 그것들이었다.
        # 남은 거리가 한 걸음보다 크면 path solver 로 한 구간 더 푼다.
        xy = poses[:, :2].copy()
        yaws = poses[:, 2].copy()
        gap = float(np.linalg.norm(xy[-1] - goal_xy))
        STEP = 0.35                                  # 이보다 크면 이어 붙이지 않고 푼다
        if gap > STEP:
            try:
                tail_ctx = dict(ctx, cur_pose=poses[-1],
                                subgoal_fns=[], path_fns=program[max(program)]["path"])
                tail, dbg = PathSolver(self.cfg.get("path_solver", {}), bxy).solve(
                    poses[-1], np.array([goal_xy[0], goal_xy[1], poses[-1][2]]),
                    tail_ctx, from_scratch=True)
                self.log.append({"stage": "tail_solve", "gap": round(gap, 3),
                                 "n": int(len(tail)),
                                 "time": round(float(dbg.get("solve_time", 0.0)), 2)})
                xy = np.vstack([xy, np.asarray(tail)[1:, :2]])
                yaws = np.concatenate([yaws, np.asarray(tail)[1:, 2]])
            except Exception as e:                   # noqa: BLE001
                self.log.append({"stage": "tail_solve_error", "gap": round(gap, 3),
                                 "msg": str(e)[:160]})
        if float(np.linalg.norm(xy[-1] - goal_xy)) > 1e-3:
            xy = np.vstack([xy, np.asarray(goal_xy, dtype=float)])
            yaws = np.concatenate([yaws, yaws[-1:]])
        goal_yaw = self._goal_yaw(lmp_env, goal_xy)
        if goal_yaw is not None:
            # 도착 자세를 마지막 점에만 꽂으면 접근 방향과 크게 다를 때 로봇이 그
            # 자리에서 통째로 돌아야 한다. 실측: CatBlockingRouteB 에서 마지막 한 점에
            # yaw 가 166 도 튀었고, 로봇이 도는 사이 목표를 0.72 m 지나쳐 실패했다.
            # 마지막 1 m 구간에 나눠 실어 접근하면서 돌게 한다.
            gy = float(goal_yaw)
            step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            run = np.concatenate([[0.0], np.cumsum(step[::-1])])[::-1]   # 끝에서의 거리
            w = np.clip(1.0 - run / 1.0, 0.0, 1.0)                       # 끝 1 m 안에서 1 로
            delta = np.arctan2(np.sin(gy - yaws), np.cos(gy - yaws))
            yaws = np.arctan2(np.sin(yaws + w * delta), np.cos(yaws + w * delta))
            yaws[-1] = gy
            self.log.append({"stage": "goal_yaw_blend", "goal_yaw": round(gy, 3),
                             "blended_pts": int((w > 0).sum())})

        K.set_path(xy)
        lmp_env._kp_constraints = None
        lmp_env._pace_profile = self._pace_profile(program, info, poses, xy)
        self._dump(out_dir, goal_xy, robot_xy, xy, yaws)

        path_cells = np.round(frame.world_to_cell(xy)).astype(int)
        return {"path": path_cells,
                "planner_info": {"source": "pivot", "stages": info.get("stages"),
                                 "backtracks": len(info.get("backtracks", [])),
                                 "stop": info.get("stop")},
                "traj_world": [(np.asarray(p, dtype=float), float(y), 1.0)
                               for p, y in zip(xy, yaws)]}

    def _pivot_targets(self, topview, frame, sdf_interp, robot_xy, goal_xy,
                       robot_yaw, n, min_clear, program=None):
        """중간 단계마다 반복 질의로 끝점을 하나씩 고른다.

        마지막 단계는 목표가 곧 끝점이라 묻지 않는다. 단계가 하나뿐이면 개입할 자리가 없다.
        한 점을 고르면 다음 질의는 그 점에서 다시 시작한다 - 걸음마다 남은 거리가 줄어야
        하므로 반경도 함께 줄인다.

        실패하면 그 단계를 건너뛴다. 여기서 예외를 내면 에피소드가 통째로 죽는다.
        """
        n = int(n or 1)
        if topview is None or n < 2:
            self.log.append({"stage": "pivot", "skipped": "단계 1개 또는 이미지 없음",
                             "stages": n})
            return None
        main = self.cfg.get("main", {})
        ch = PIVOT.PivotChooser(self.client, self.prompts_dir,
                                rounds=int(main.get("pivot_rounds", 3)),
                                n_samples=int(main.get("pivot_samples", 9)),
                                seed=int(self.seed or 0))
        cur = np.asarray(robot_xy, float)[:2]
        goal = np.asarray(goal_xy, float)[:2]
        kps = K.get_keypoints()

        def reject_for(stage):
            """이 단계의 path 제약을 어기는 후보를 뺀다.

            없으면 같은 응답 안에서 두 답이 어긋난다 - 모델이 "사람에게서 0.7 m" 라고
            써 놓고 그보다 가까운 점을 골라도 막을 것이 없다. 그러면 두 제약이
            가중치 200 으로 맞붙어 경로가 그 사이를 지난다.
            """
            fns = ((program or {}).get(stage, {}) or {}).get("path") or []
            if not fns:
                return None

            def _r(pt):
                tr = K.Traj(np.asarray([pt], float), dt=0.2)
                for fn in fns:
                    try:
                        if float(fn(tr, kps)) > 0.0:
                            return True
                    except Exception:            # noqa: BLE001
                        continue
                return False
            return _r

        targets = {}
        for s in range(1, n):
            remain = float(np.linalg.norm(goal - cur))
            step = max(0.6, remain / max(n - s + 1, 1))
            pt = ch.choose(annotate, topview, frame, sdf_interp, cur, goal,
                           step_m=step, min_clear=min_clear,
                           robot_yaw=robot_yaw if s == 1 else None,
                           tag=f"stage{s}", reject=reject_for(s))
            if pt is None:
                continue
            targets[s] = [float(pt[0]), float(pt[1])]
            cur = np.asarray(pt, float)
        self.log.append({"stage": "pivot", "stages": n, "picked": targets,
                         "queries": sum(len(r["rounds"]) for r in ch.log),
                         "rounds": ch.log})
        return targets or None

    def _stop_targets(self, qinfo, stop_xy, n):
        """`stop_points = [...]` 번호를 좌표로 바꾼다.

        -1 은 "목표에서 끝난다" 는 뜻이라 건너뛴다 - 마지막 단계에는 이미
        progress_cost(GOAL) 이 주입되므로 덮어쓸 이유가 없다. 범위를 벗어난 번호도
        건너뛰고 로그에 남긴다: 조용히 1 번으로 떨어뜨리면 "모델이 1 번을 골랐다" 와
        "번호가 틀렸다" 가 구별되지 않는다.
        """
        pts = qinfo.get("stop_points")
        n = int(n or 0)
        if not pts or not len(stop_xy):
            self.log.append({"stage": "stop_targets", "why": "번호 없음",
                             "raw": pts, "n_cand": int(len(stop_xy))})
            return None
        out, bad = {}, []
        for s_idx, num in enumerate(pts[:max(n, len(pts))], start=1):
            if num == -1:
                continue
            if not (1 <= int(num) <= len(stop_xy)):
                bad.append(num)
                continue
            if n and s_idx >= n:            # 마지막 단계는 목표가 끝점이다
                continue
            xy = stop_xy[int(num) - 1]
            out[s_idx] = [float(xy[0]), float(xy[1])]
        self.log.append({"stage": "stop_targets", "raw": pts, "used": out,
                         "out_of_range": bad, "n_cand": int(len(stop_xy))})
        return out or None

    def _pace_profile(self, program, info, poses, xy):
        """단계별 동역학 선언을 실행부가 쓰는 속도 곡선으로 바꾼다.

        **이것이 v/a/J 가 실제로 작동하는 유일한 경로다.** solver 비용에 넣을 수 없는
        이유는 path solver 의 제어점이 시간이 아니라 거리 간격이라, 그 위의 속도·가속도는
        실행과 아무 관계가 없는 숫자이기 때문이다 (비용은 내려가는데 아무것도 재지 않는다).

        대신 기하와 시간을 나눈다 - 경로는 solver 가 풀고, 그 경로 위의 속도만 여기서
        정한다. 단계가 구간을 주므로 `feasible_profile` 이 요구하는 (from%, to%) 를
        VLM 에게 물을 필요가 없다.

        만든 곡선 위에서 선언을 **다시 평가해 로그에 남긴다**. 그러지 않으면 이 함수들이
        "호출되지 않는 선언" 이 되어, 값이 반영됐는지 아닌지 사후에 알 수 없다.
        """
        if not self.dynamics:
            return None
        spans = (info or {}).get("stage_spans") or {}
        n = len(poses)
        if n < 2 or not spans:
            return None
        # emitted 인덱스 -> 전체 경로의 호 백분율
        seg = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        if total <= 1e-6:
            return None

        specs, decl = [], []
        for st, sp in sorted(spans.items()):
            vals = {}
            for d in (program.get(st, {}) or {}).get("pace", []) or []:
                for a in d.get("args", []):
                    if a.get("fn") == "speed_cost" and a.get("v_max") is not None:
                        vals["speed"] = float(a["v_max"])
                    elif a.get("fn") == "accel_cost" and a.get("a_max") is not None:
                        vals["accel"] = float(a["a_max"])
                    elif a.get("fn") == "jerk_cost" and a.get("j_max") is not None:
                        vals["jerk"] = float(a["j_max"])
            if not vals:
                continue
            i0, i1 = int(np.clip(sp[0], 0, n - 1)), int(np.clip(sp[1], 0, n - 1))
            f0, f1 = 100.0 * cum[i0] / total, 100.0 * cum[i1] / total
            # feasible_profile 은 "평상시 대비 배수" 를 받는다. VLM 은 절대값(m/s)을
            # 주므로 여기서 나눈다 - 단위를 VLM 에게 맡기면 답마다 기준이 달라진다.
            sc = vals.get("speed", K.NOMINAL_V) / max(K.NOMINAL_V, 1e-6)
            ac = vals.get("accel", K.A_MAX) / max(K.A_MAX, 1e-6)
            jk = vals.get("jerk", K.J_MAX) / max(K.J_MAX, 1e-6)
            specs.append((f0, f1, float(sc), float(ac), float(jk)))
            decl.append({"stage": st, "from_pct": round(f0, 1), "to_pct": round(f1, 1),
                         **{k: round(v, 3) for k, v in vals.items()}})
        if not specs:
            return None
        # 백분율은 poses(정리 전) 기준으로 냈지만 곡선은 최종 경로 위에 놓인다.
        # dedupe·꼬리 구간이 길이를 조금 바꾸므로 길이는 최종 경로에서 다시 잰다.
        xy = np.asarray(xy, dtype=float)
        total_xy = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        try:
            s_arr, v_arr = K.feasible_profile(specs, max(total_xy, 1e-6))
        except Exception as e:                           # noqa: BLE001
            self.log.append({"stage": "pace_error", "msg": str(e)[:160]})
            return None

        # 검산: 만들어진 곡선이 실제로 그 값을 지키는가.
        check = []
        for d in decl:
            m = (s_arr >= total_xy * d["from_pct"] / 100.0 - 1e-9) & \
                (s_arr <= total_xy * d["to_pct"] / 100.0 + 1e-9)
            if m.any() and "speed" in d:
                check.append({"stage": d["stage"], "asked": d["speed"],
                              "profile_max": round(float(v_arr[m].max()), 3)})
        self.log.append({"stage": "pace_profile", "declared": decl,
                         "len_m": round(total_xy, 2), "check": check,
                         "v_min": round(float(v_arr.min()), 3),
                         "v_max": round(float(v_arr.max()), 3)})
        return (s_arr, v_arr, xy)

    @staticmethod
    def _goal_yaw(lmp_env, goal_xy):
        """도착 자세. keypoint_nav 훅과 같은 규칙 - 다르면 판정이 갈린다."""
        env = getattr(getattr(lmp_env, "_env", None), "env", None)
        if env is None:
            return None
        if getattr(env, "dst_is_human", False):
            hp = getattr(env, "target_pos", None)
            if hp is not None:
                d = np.asarray(hp, float)[:2] - np.asarray(goal_xy, float)[:2]
                if np.linalg.norm(d) > 0.1:
                    return float(np.arctan2(d[1], d[0]))
        ori = getattr(env, "target_ori", None)
        return float(ori[2]) if ori is not None else None

    def _dump(self, out_dir, goal_xy, robot_xy, xy=None, yaws=None):
        """단계별 비용 항까지 남긴다.

        스칼라 하나로 줄이면 "어느 항이 지배했는가" 를 사후에 물을 수 없다 -
        나쁜 경로를 받았을 때 진단할 유일한 수단이다.
        """
        if not out_dir:
            return
        try:
            os.makedirs(out_dir, exist_ok=True)
            doc = {"log": self.log,
                   "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
                   "robot_xy": [float(robot_xy[0]), float(robot_xy[1])]}
            if xy is not None:
                doc["waypoints"] = [[float(a), float(b), float(c)]
                                    for (a, b), c in zip(xy, yaws)]
            with open(os.path.join(out_dir, "pivot_log.json"), "w") as f:
                json.dump(doc, f, ensure_ascii=False, default=str)
        except OSError:
            pass
