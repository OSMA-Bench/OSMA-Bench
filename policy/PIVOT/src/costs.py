"""비용 항. PIVOT 의 objective 를 2D 네비게이션으로 옮긴 것.

가중치는 PIVOT 값을 그대로 쓴다 - 그 값이 옳아서가 아니라, 바꾸면 "구조를 옮겼는데
결과가 다르다" 와 "가중치를 손봤더니 결과가 다르다" 를 구분할 수 없기 때문이다.

옮기지 않은 것:
    IK 비용(20.0)   -> turn_cost. 홀로노믹 베이스는 자유공간이면 어디든 도달하므로
                       "도달 가능한가" 가 아니라 "싸게 도달하는가" 가 대응한다.
    관절 정규화(0.2) -> 드롭. 기준 관절 자세에 해당하는 것이 없다.
    잡기 지표(10.0)  -> 드롭. is_grasp_stage 플래그도 이식하지 않는다 -
                       죽은 인자를 다섯 군데로 끌고 다니게 된다.
"""
import numpy as np

from . import interp as I
from . import sdf as S

# PIVOT 원본 가중치
W_COLLISION_SUBGOAL = 0.8
W_COLLISION_PATH = 0.5
W_CONSISTENCY = 1.0
W_TURN = 20.0
W_PATH_LENGTH = 4.0
W_CONSTRAINT = 200.0

# 충돌 여유. PIVOT 이 subgoal 0.10 / path 0.20 을 쓴다.
MARGIN_SUBGOAL = 0.10
MARGIN_PATH = 0.20

# 베이스가 점유 셀 안에 있는 것은 비싼 것이 아니라 **불가능**하다.
# PIVOT 에서 IK 비용(20.0)이 하던 역할 - "물리적으로 취할 수 없는 자세를 배제한다" -
# 의 네비게이션 대응이다. 조작에서는 팔이 물체 중심 **위로** 뻗을 수 있어 중심을
# 목표로 삼는 것이 말이 되지만, 바퀴 베이스는 그 자리에 있을 수 없다.
#
# 이 항이 없으면 이렇게 깨진다: 키포인트는 물체 중심이라 sink·stove 는 조리대
# **안쪽**이다. VLM 이 progress_cost(state, keypoints['sink'], 0.4) 를 쓰면 가중치
# 200 이 subgoal 을 조리대 안으로 끌어당기고 충돌 항 0.8 은 상대가 되지 않는다.
# 실측: 15 건 중 8 건에서 계획 경로가 장애물을 관통했고 최대 0.90 m 깊이였다.
W_INFEASIBLE = 5000.0

# 진행 방향이 얼마나 자주 꺾이는가. PIVOT 에는 대응이 없다 - 팔 경로는 짧아
# 경로길이 항(4.0)만으로 충분히 곧게 펴진다.
#
# 우리는 두 가지가 겹쳐 그것이 성립하지 않는다. (1) 경로길이를 직선거리로 나눠
# 무차원화하면서(스케일 보정) 지그재그를 억제하던 힘이 함께 약해졌고, (2) turn_cost 는
# 홀로노믹 베이스의 **yaw** 를 보므로 공간적 지그재그를 전혀 벌하지 않는다.
# 그 결과 진행 방향을 벌하는 항이 하나도 없었다.
#
# 실측: 실패한 에피소드의 계획 경로는 53 점 중 10 곳이 20 도 이상 꺾이고 최대 158 도로
# 되돌아왔다. 컨트롤러 추종 편차가 평균 0.215 m(정상 건은 0.03~0.07 m)로 벌어졌다.
W_SMOOTH = 4.0


def infeasibility(interp, cells, radius_m=0.0):
    """베이스가 지날 수 없는 깊이의 합.

    `radius_m` 은 로봇 반경을 그대로 쓰지 않는다. 이 맵은 보수적이라 로봇의 출발
    위치조차 여유가 0.06~0.22 m 밖에 안 되는 곳이 있고, 반경(0.43 m)으로 불가능을
    선언하면 **출발점부터 불가능해져 solver 가 아무 데도 못 간다.**

    그렇다고 0 으로 두면(중심만 검사) 반대 실패가 난다: 관통은 없는데 최소 여유가
    0.06 m 인 경로가 나오고, 몸통이 벽에 걸려 로봇이 물리적으로 지나가지 못한다
    (실측: HumanBlockingRouteB, 계획은 done 인데 목표 1.43 m 앞에서 정지).

    그래서 훅이 **출발 위치의 실제 여유**에서 반경을 정해 넘긴다 - 출발점이 증명한
    만큼은 요구하되 그 이상은 요구하지 않는다.
    """
    return S.collision_cost(interp, cells, float(radius_m), 0.0)


def constraint_violation(fns, arg, keypoints):
    """제약 위반의 합. PIVOT 과 같이 clip(v, 0, inf) 후 더한다.

    음수(여유 있음)를 그대로 더하면 한 제약의 여유가 다른 제약의 위반을 상쇄한다.
    예외를 내는 제약은 건너뛴다 - 하나 때문에 최적화 전체가 죽으면 안 된다.
    """
    total = 0.0
    detail = []
    for fn in fns or []:
        try:
            v = float(fn(arg, keypoints))
        except Exception:                                    # noqa: BLE001
            detail.append(None)
            continue
        detail.append(v)
        total += float(np.clip(v, 0.0, None))
    return total, detail


def subgoal_cost(pose, ctx, return_detail=False):
    """subgoal 자세 하나의 비용.

    ctx 는 dict: sdf_interp, frame, robot_r_m, cur_pose, subgoal_fns, path_fns, keypoints
    """
    d = {}
    cell = ctx["frame"].world_to_cell([pose[:2]])
    d["collision"] = W_COLLISION_SUBGOAL * S.collision_cost(
        ctx["sdf_interp"], cell, ctx["robot_r_m"], MARGIN_SUBGOAL)
    d["infeasible"] = W_INFEASIBLE * infeasibility(
        ctx["sdf_interp"], cell, ctx.get("infeasible_r_m", 0.0))
    d["consistency"] = W_CONSISTENCY * I.consistency(pose, ctx["cur_pose"])
    d["turn"] = W_TURN * I.turn_cost(ctx["cur_pose"], pose)

    v_sub, det_sub = constraint_violation(ctx.get("subgoal_fns"), tuple(pose), ctx["keypoints"])
    d["subgoal_constraint"] = W_CONSTRAINT * v_sub
    # PIVOT 은 subgoal 을 풀 때도 path 제약을 함께 본다 - 그 지점이 path 제약을
    # 어기는 자리라면 애초에 목표로 삼을 이유가 없다.
    # path 제약은 Traj 를 받는다. 한 점이라도 감싸서 넘겨야 한다 - 튜플을 그대로
    # 넘기면 매번 예외가 나고 constraint_violation 이 그것을 삼켜 기여가 0 이 된다.
    # 그러면 "subgoal 을 풀 때도 path 제약을 함께 본다" 는 성질이 이름만 남는다.
    _mk = ctx.get("make_traj")
    _arg = _mk([pose[:2]]) if _mk is not None else tuple(pose)
    v_path, det_path = constraint_violation(ctx.get("path_fns"), _arg, ctx["keypoints"])
    d["path_constraint"] = W_CONSTRAINT * v_path

    total = float(sum(d.values()))
    if return_detail:
        d["total"] = total
        d["subgoal_violation"] = det_sub
        d["path_violation"] = det_path
        return total, d
    return total


def path_cost(poses, ctx, return_detail=False):
    """조밀한 자세 열의 비용.

    poses: (N, 3) 시작·끝 포함. 제약은 시작·끝을 뺀 안쪽만 본다 - PIVOT 과 같다.
    시작은 이미 지나온 곳이고 끝은 subgoal solver 가 정한 곳이라, 여기서 벌해 봐야
    제어점이 바꿀 수 없는 값이다.
    """
    d = {}
    inner = poses[1:-1] if len(poses) > 2 else poses
    cells = ctx["frame"].world_to_cell(inner[:, :2])
    d["collision"] = W_COLLISION_PATH * S.collision_cost(
        ctx["sdf_interp"], cells, ctx["robot_r_m"], MARGIN_PATH)
    d["infeasible"] = W_INFEASIBLE * infeasibility(
        ctx["sdf_interp"], cells, ctx.get("infeasible_r_m", 0.0))

    # 직선 거리로 나눠 무차원으로 만든다. **작업공간 크기 보정이지 가중치 변경이 아니다.**
    # PIVOT 은 0.55 m 작업공간이라 이 항이 0.3 수준이지만 6 m 주방에서는 그대로 두면
    # 5~6 이 되어 충돌 항(0.5 × 침범 합)을 압도한다 - 실측: HumanBlockingRouteB 에서
    # 41 점 중 30 점이 장애물 **안쪽**(최대 0.56 m 깊이)인 경로가 최적해로 나왔다.
    # 벽을 뚫고 직선으로 가는 쪽이 비용상 이득이었기 때문이다.
    # opt_pos_step_size 를 다시 잡은 것과 같은 종류의 스케일 보정이다.
    pos_len, rot_len = I.path_length(poses, rot_weight=1.0)
    direct = max(float(np.linalg.norm(poses[-1][:2] - poses[0][:2])), 0.5)
    d["path_length"] = W_PATH_LENGTH * (pos_len + rot_len) / direct

    # 진행 방향 변화. [0,1] 로 정규화한다 - turn_cost 와 같은 꼴.
    d["smooth"] = W_SMOOTH * I.heading_change(poses)

    # 회전 비용은 제어점마다. PIVOT 도 IK 비용을 제어점마다 더한다.
    d["turn"] = W_TURN * float(np.mean([
        I.turn_cost(poses[i], poses[i + 1]) for i in range(len(poses) - 1)]))

    # path 제약은 궤적 전체를 본다 - Traj 를 받는 함수라 위치만 넘긴다.
    v, det = constraint_violation(ctx.get("path_fns"), ctx["make_traj"](inner[:, :2]),
                                  ctx["keypoints"])
    d["path_constraint"] = W_CONSTRAINT * v

    total = float(sum(d.values()))
    if return_detail:
        d["total"] = total
        d["path_violation"] = det
        return total, d
    return total
