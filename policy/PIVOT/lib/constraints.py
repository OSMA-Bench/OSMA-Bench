"""VLM 이 호출하는 제약 원시 함수.

VLM 은 **무엇을, 얼마나** 만 정한다. 단위 변환·정규화·미분은 전부 여기가
소유한다. VLM 이 np.linalg.norm 을 직접 쓰기 시작하면 항마다 단위가 달라져
서로 더할 수 없게 되고, 무엇이 위반인지도 코드마다 달라진다.

부호 규약은 PIVOT 과 같다: **반환 <= 0 이면 만족, > 0 이면 위반량**.
그래서 제약을 그냥 더할 수 있다.

정규화: 모든 반환값은 기준값 대비 무차원이다. clearance 는 margin 대비,
속도는 v_max 대비. 그래야 거리(m)와 저크(m/s^3)를 같은 저울에 올릴 수 있다.

티어 하한을 걸지 않는다. VLM 이 쓴 값을 그대로 쓴다. 이 실험이 재려는 것이
"장면을 보고 적절한 안전거리·속도를 정할 수 있는가" 이므로, max(vlm, tier) 로
덮어쓰면 재려던 능력이 측정에서 사라진다. 프롬프트에도 티어 값을 노출하지
않는다. 대신 VLM 이 정한 값을 전부 기록해 티어와 사후 비교한다.

배치 지원: traj 는 단일 궤적 (N, 2) 도, MPPI 롤아웃 (K, N, 2) 도 받는다.
전자는 스칼라를, 후자는 (K,) 를 돌려준다. MPPI 는 K 개 궤적을 한 번에
평가해야 하는데 파이썬 루프로는 감당이 안 된다.
"""
import os

import numpy as np

# 제어 주기 (s). RoboCasa 는 10 Hz.
DEFAULT_DT = 0.1


class Traj:
    """제약 평가 대상 궤적. 위치와 dt 만 있으면 나머지는 파생된다.

    Args:
        p: (N, 2) 또는 (K, N, 2) 월드 좌표.
        dt: 제어 주기. 속도·가속도·저크는 이것 없이 정의되지 않는다.
    """

    def __init__(self, p, dt=DEFAULT_DT):
        p = np.asarray(p, dtype=float)
        if p.ndim == 2:
            p = p[None, :, :]          # (1, N, 2) 로 통일해 내부는 항상 배치
            self._single = True
        elif p.ndim == 3:
            self._single = False
        else:
            raise ValueError(f"traj 는 (N,2) 또는 (K,N,2) 여야 한다: {p.shape}")
        self.p = p
        self.dt = float(dt)

    def _out(self, vals):
        """내부 (K,) 결과를 입력 모양에 맞춰 돌려준다."""
        return float(vals[0]) if self._single else vals

    @property
    def v(self):
        """속도 크기 (K, N-1)."""
        return np.linalg.norm(np.diff(self.p, axis=1), axis=-1) / self.dt

    @property
    def a(self):
        """가속도 크기 (K, N-2)."""
        return np.abs(np.diff(self.v, axis=1)) / self.dt

    @property
    def j(self):
        """저크 크기 (K, N-3)."""
        return np.abs(np.diff(self.a, axis=1)) / self.dt


# ---------- 키포인트 해석 ----------
#
# VLM 은 좌표를 모르고 문자 라벨만 안다 ("A 에서 0.8m"). 실행 시점에 라벨을
# 월드 좌표로 바꾸는 표를 여기에 둔다. 매 에피소드 시작 시 러너가 채운다.

_KEYPOINTS = {}

# 대상별 반경 보정 (m). 벤치마크는 로봇/장애물 지오메트리의 표면-대-표면
# 거리로 위반을 판정하는데(kitchen_navigate_safe.py:812), clearance_cost 는
# 중심-대-중심 거리를 잰다. 부피가 큰 대상일수록 격차가 커진다 - 실측:
# 사람 과제에서 중심 기준 위반 1.6% 인데 벤치마크 판정은 35~41% 였고,
# MPPI 가중치를 10배 바꿔도 이 값이 거의 변하지 않았다.
# 로봇 반경 약 0.30m 를 공통으로 더하고, 대상 크기를 이름으로 보정한다.
# 실측값이다. 로봇/장애물 지오메트리 표면거리와 중심거리의 차이를 실행 로그
# 6과제에서 재 중앙값을 썼다 (robot_pos 는 min_obstacle_distance 보다 5배 조밀해
# 5:1 로 정렬해야 한다 - 처음에 앞에서부터 잘라 맞췄다가 값이 2~4배 부풀었다).
_ROBOT_R = 0.0
_OBJ_R = {
    "human": 0.52, "posed": 0.52,
    "crawling_baby": 1.38,      # 바닥에 엎드려 지오메트리가 넓게 퍼진다
    "dog": 0.35, "cat": 0.29,
    "vase": 0.30, "wine": 0.30, "glass_of_water": 0.30, "hot_chocolate": 0.30,
    "kettlebell": 0.33, "trashbin": 0.32,
}
_KP_NAMES = {}

# 라벨별 점군 (M, 2). 있으면 거리는 "중심까지" 가 아니라 "가장 가까운 점까지" 로
# 잰다. 벤치마크가 지오메트리 표면거리로 판정하는데 우리는 중심 좌표만 갖고 있어
# 등방 반경으로 메우고 있었고, 그래서 팔다리가 뻗은 대상에서만 크게 틀렸다
# (덩어리 0~5% 대 사람·아기 18~37%).
#
# 앞서 실패한 "spread" 와 다른 점: 그때는 물체를 A/B/C 로 쪼개 VLM 에게 보여
# 라벨 예산을 먹고 반경 보정도 나눠 오히려 나빠졌다 (사람 37.2 -> 53.0%).
# 여기서는 VLM 이 보는 것은 그대로 문자 하나이고, 헬퍼 안에서만 점군을 쓴다.
_KP_CLOUDS = {}


def set_keypoint_clouds(mapping):
    """{라벨: (M,2)} 를 등록한다. 비우려면 None."""
    global _KP_CLOUDS
    _KP_CLOUDS = {str(k): np.asarray(v, dtype=float).reshape(-1, 2)
                  for k, v in (mapping or {}).items()
                  if v is not None and len(np.asarray(v).reshape(-1, 2))}


def set_keypoint_names(mapping):
    """{'A': 'crawling_baby', ...} - 반경 보정에 쓴다."""
    global _KP_NAMES
    _KP_NAMES = {str(k): str(v) for k, v in (mapping or {}).items()}


def _radius_pad(kp):
    """중심 거리를 표면 거리로 바꾸기 위해 더할 값.

    'human#2' 처럼 분할된 점은 전체 형상이 아니라 그 조각만 담당한다. 조각
    개수로 나누지 않으면 같은 대상을 여러 번 세게 되어 실효 margin 이 배로
    커진다 - 실측: 사람을 3점으로 나누고 각 점에 전체 보정(0.52m)을 그대로
    적용했더니 통로가 막혀 도달이 0.169m 에서 2.105m 로 무너졌다.
    """
    name = _KP_NAMES.get(str(kp), "")
    base, _, _sfx = name.partition("#")
    r = 0.0
    for key, val in _OBJ_R.items():
        if key in base.lower():
            r = val
            break
    if _sfx:
        # 몇 조각으로 나뉘었는지 세어 그만큼 줄인다.
        k = sum(1 for v in _KP_NAMES.values() if v.startswith(base + "#"))
        if k > 1:
            r /= float(k)
    return _ROBOT_R + r


def set_keypoint_aliases(mapping):
    """이름으로도 키포인트를 찾을 수 있게 한다.

    VLM 에게 문자('F')로 가리키게 하면 코드만 봐서는 무엇을 피하는지 알 수 없다.
    docstring 이 유일한 설명이었는데 그마저 없으면 로그가 읽히지 않는다.
    이름('human')을 쓰면 코드 자체가 설명이 된다.
    """
    for lab, name in (mapping or {}).items():
        if lab in _KEYPOINTS:
            _KEYPOINTS[str(name)] = _KEYPOINTS[lab]
            if lab in _KP_CLOUDS:
                _KP_CLOUDS[str(name)] = _KP_CLOUDS[lab]
            _KP_NAMES[str(name)] = str(name)


def set_keypoints(mapping):
    """{'A': (x, y), 'B': (x, y), ...} 로 교체한다."""
    global _KEYPOINTS
    _KEYPOINTS = {str(k): np.asarray(v, dtype=float)[:2] for k, v in mapping.items()}


def get_keypoints():
    return dict(_KEYPOINTS)


class UnknownKeypoint(KeyError):
    """VLM 이 없는 라벨을 참조했다. 재질의 대상."""


def _kp_xy(kp):
    """라벨('A') 도 좌표((x, y)) 도 받는다.

    PIVOT 의 관용구가 keypoints[i] 라서 VLM 이 좌표를 직접 넘기는 쪽을 더
    자연스럽게 쓴다 (실제 응답에서 확인). 둘 다 받아야 멀쩡한 제약이 형식
    때문에 버려지지 않는다.
    """
    if isinstance(kp, (str, bytes)):
        key = kp.decode() if isinstance(kp, bytes) else kp
        if key not in _KEYPOINTS:
            raise UnknownKeypoint(
                f"'{key}' 는 없는 키포인트다. 사용 가능: {sorted(_KEYPOINTS)}")
        return _KEYPOINTS[key]
    xy = np.asarray(kp, dtype=float).ravel()
    if xy.shape[0] < 2 or not np.all(np.isfinite(xy[:2])):
        raise UnknownKeypoint(f"키포인트로 쓸 수 없는 값: {kp!r}")
    return xy[:2]


def _label_of(kp):
    """좌표로 넘어온 키포인트를 등록된 라벨로 되돌린다.

    PIVOT 관용구가 keypoints['G'] 라서 VLM 은 좌표를 직접 넘기는 쪽을 자주 쓴다.
    그런데 점군과 반경 보정은 라벨로 조회하므로, 좌표로 들어오면 둘 다 조용히
    빠진 채 중심거리만 남는다 - 실측: 같은 제약이 'A' 로는 +0.057(위반),
    keypoints['A'] 로는 -1.427(만족) 이 나왔다. 표기 때문에 판정이 뒤집히면 안 된다.
    """
    if isinstance(kp, (str, bytes)):
        return kp.decode() if isinstance(kp, bytes) else kp
    try:
        xy = np.asarray(kp, dtype=float).ravel()[:2]
    except (TypeError, ValueError):
        return None
    for lab, val in _KEYPOINTS.items():
        if np.allclose(np.asarray(val, dtype=float).ravel()[:2], xy, atol=1e-6):
            return lab
    return None


def _dist_to(traj, kp):
    """궤적 각 점에서 키포인트까지 거리 (K, N).

    점군이 등록돼 있으면 그중 가장 가까운 점까지의 거리다 - 표면거리에 해당한다.
    없으면 중심까지의 거리이고, 그 경우에만 반경 보정이 의미를 갖는다.
    """
    _lab = _label_of(kp)
    cloud = _KP_CLOUDS.get(_lab) if _lab is not None else None
    if cloud is not None and len(cloud):
        # (K, N, 1, 2) - (1, 1, M, 2) -> (K, N, M) -> min
        d = np.linalg.norm(traj.p[:, :, None, :] - cloud[None, None, :, :], axis=-1)
        return d.min(axis=2)
    return np.linalg.norm(traj.p - _kp_xy(kp)[None, None, :], axis=-1)


# ---------- 제약 (반환 <= 0 이면 만족) ----------

# 제약이 꺼진 경우의 반환값. 0 이 아니라 음수여야 "만족" 으로 읽힌다.
_OFF = -1.0

# 평상시 속도 (m/s). 주변에 아무것도 없을 때의 순항 속도이고, 배수 1 이 이 값이다.
# 기준선(VoxPoser 14B)의 평균 속도가 0.51 m/s 였다.
NOMINAL_V = float(os.environ.get("NOMINAL_V", "0.5"))


def clearance_cost(traj, kp, margin):
    """kp 에서 margin(m) 이상 떨어져 있는가. SSI d 축.

    반환은 margin 대비 부족분이다. margin=0.6 인데 0.42m 로 지나면 0.3.
    이진 필터와 다른 점이 여기다 - 0.42m 와 0.30m 가 둘 다 위반이지만 값이
    다르므로, margin 을 못 지키는 좁은 통로에서도 "가능한 만큼 최대로 벌린"
    쪽이 낮은 cost 를 받는다.

    궤적 전체에서 최악의 점을 본다. 한 점이라도 침범하면 위반이기 때문이다.
    """
    margin = float(margin)
    if margin <= 0:
        return _OFF                      # 제약 없음 (0 나누기도 막는다)
    # 표면 기준으로 환산한다. VLM 이 정한 margin 은 "표면에서 이만큼" 이라는
    # 뜻이고, 우리가 가진 거리는 중심-대-중심이므로 반경을 더해 비교한다.
    # 점군으로 표면거리를 재는 경우 반경 보정은 이미 반영된 셈이라 더하지 않는다.
    _lab = _label_of(kp)
    _has_cloud = _lab is not None and _lab in _KP_CLOUDS
    eff = margin if _has_cloud else margin + _radius_pad(_lab if _lab else kp)
    d = _dist_to(traj, kp)
    return traj._out(np.max((eff - d) / eff, axis=1))


def speed_limit_cost(traj, kp, v_max, radius):
    """kp 반경 radius(m) 안에서 v_max(m/s) 이하로 가는가. SSI v 축.

    radius 가 있어 근처에서만 발동한다. 경로 전체에 속도 상한을 걸면 무관한
    구간까지 느려져 도달이 늦어진다.

    반경 안에 든 점이 없으면 만족으로 본다 - 그 대상 근처를 지나지 않았다는
    뜻이므로 벌할 이유가 없다.
    """
    v_max = float(v_max)
    if v_max <= 0:
        return _OFF
    d = _dist_to(traj, kp)[:, 1:]        # v 와 길이를 맞춘다 (차분으로 1 짧다)
    inside = d < float(radius)
    over = (traj.v - v_max) / v_max
    over = np.where(inside, over, -np.inf)
    worst = np.max(over, axis=1)
    return traj._out(np.where(np.isfinite(worst), worst, _OFF))


def accel_limit_cost(traj, a_max):
    """가속도 크기가 a_max(m/s^2) 이하인가. SSI a 축."""
    a_max = float(a_max)
    if a_max <= 0:
        return _OFF
    a = traj.a
    if a.shape[1] == 0:
        return traj._out(np.full(traj.p.shape[0], _OFF))
    return traj._out(np.max((a - a_max) / a_max, axis=1))


def jerk_limit_cost(traj, j_max):
    """저크 크기가 j_max(m/s^3) 이하인가. SSI J 축.

    저크는 3차 차분이라 최소 4점이 필요하다. 계획 단계의 waypoint 열에서는
    사실상 의미가 없고, MPPI 롤아웃처럼 제어 주기 위에 놓인 궤적에서만
    제대로 정의된다.
    """
    j_max = float(j_max)
    if j_max <= 0:
        return _OFF
    j = traj.j
    if j.shape[1] == 0:
        return traj._out(np.full(traj.p.shape[0], _OFF))
    return traj._out(np.max((j - j_max) / j_max, axis=1))



def standoff_cost(state, kp, min_d):
    """이 자세 하나가 kp 에서 min_d(m) 이상 떨어져 있는가.

    clearance_cost 를 한 점으로 줄인 것이다. PIVOT 방식 solver 의 subgoal 제약은
    궤적이 아니라 자세 하나를 받으므로 traj 판이 쓰이지 않는다.
    """
    min_d = float(min_d)
    if min_d <= 0:
        return _OFF
    xy = np.asarray(state, dtype=float)[:2]
    _lab = _label_of(kp)
    cloud = _KP_CLOUDS.get(_lab) if _lab is not None else None
    if cloud is not None and len(cloud):
        d = float(np.linalg.norm(cloud - xy[None, :], axis=1).min())
        eff = min_d
    else:
        d = float(np.linalg.norm(xy - _kp_xy(kp)))
        eff = min_d + _radius_pad(_lab if _lab else kp)
    return (eff - d) / eff

def progress_cost(state, kp, tol):
    """kp 에서 tol(m) 안으로 들어왔는가. 목표 도달 판정.

    state 는 (x, y) 또는 (x, y, yaw).
    """
    tol = float(tol)
    if tol <= 0:
        return _OFF
    xy = np.asarray(state, dtype=float)[:2]
    d = float(np.linalg.norm(xy - _kp_xy(kp)))
    return (d - tol) / tol


def heading_cost(state, kp, tol_rad):
    """kp 를 향해 서 있는가. 도착 자세 판정.

    state 는 (x, y, yaw) 여야 한다. yaw 가 없으면 판정할 수 없다.
    """
    tol_rad = float(tol_rad)
    if tol_rad <= 0:
        return _OFF
    s = np.asarray(state, dtype=float)
    if s.shape[0] < 3 or not np.isfinite(s[2]):
        return _OFF
    d = _kp_xy(kp) - s[:2]
    want = float(np.arctan2(d[1], d[0]))
    err = abs(float(np.arctan2(np.sin(want - s[2]), np.cos(want - s[2]))))
    return (err - tol_rad) / tol_rad


# VLM 에게 노출하는 이름. 샌드박스가 이 목록만 허용한다 (#49).
# 프롬프트에 적어 준 것과 같아야 한다. 알려주지 않은 함수를 허용하면
# VLM 이 쓸 수는 있는데 아무도 그 동작을 보증하지 않는다 - progress_cost 와
# heading_cost 는 traj 가 아니라 state 를 받으므로 MPPI 에서 예외가 나고
# try/except 에 걸려 조용히 건너뛰어진다.
DOCUMENTED = {
    "clearance_cost": clearance_cost,
    "speed_limit_cost": speed_limit_cost,
    "accel_limit_cost": accel_limit_cost,
    "jerk_limit_cost": jerk_limit_cost,
}

# 계획 단계(경로 검증)에서만 쓰는 것들. VLM 에게는 노출하지 않는다.
PLAN_ONLY = {
    "progress_cost": progress_cost,
    "heading_cost": heading_cost,
    "standoff_cost": standoff_cost,
}

# PIVOT 방식 solver 가 VLM 에게 노출하는 것. subgoal 은 자세 하나, path 는 궤적을 받는다.
# pace_cost 는 넣지 않는다 - solver 의 제어점은 시간이 아니라 거리 간격이라
# Traj.v/a/j 가 무의미해진다. 속도는 _pace_profile 로 따로 강제한다.
PIVOT_SUBGOAL = {
    "progress_cost": progress_cost,
    "heading_cost": heading_cost,
    "standoff_cost": standoff_cost,
}
PIVOT_PATH = {
    "clearance_cost": clearance_cost,
}

# ---------- PIVOT 단계용 동역학 제약 ----------
#
# 기존 함수를 PIVOT 의 단계 구조에 맞게 변형한 것이다. 달라진 것은 **speed 하나**다:
#
#   keypoint_nav:  speed_limit_cost(traj, kp, v_max, radius)
#   여기:          speed_cost(traj, v_max)
#
# keypoint_nav 에는 구간 개념이 없어 "어디서부터 어디까지 느리게" 를 표현하려면
# 대상과 반경을 함께 줘야 했다. PIVOT 은 주행을 이미 단계로 나누므로 **단계가 곧
# 구간**이고, 그 두 인자가 사라진다. accel·jerk 는 원래도 구간 전체에 걸리는 값이라
# 이름만 맞춘다.
#
# 이 셋은 solver 비용에 들어가지 않는다. path solver 의 제어점은 시간이 아니라 거리
# 간격이라 그 위의 v/a/J 는 아무것도 재지 않는다 - 대신 단계별 호 구간을 속도 곡선으로
# 바꿔 실행부에 넘기고(_pace_profile), 만들어진 곡선 위에서 이 함수들로 다시 검산한다.


def speed_cost(traj, v_max):
    """이 단계를 v_max(m/s) 이하로 지나는가. SSI v 축.

    speed_limit_cost 에서 (kp, radius) 를 뺀 것. 단계가 적용 구간을 대신한다.
    """
    v_max = float(v_max)
    if v_max <= 0:
        return _OFF
    v = traj.v
    if v.shape[1] == 0:
        return traj._out(np.full(traj.p.shape[0], _OFF))
    return traj._out(np.max((v - v_max) / v_max, axis=1))


PIVOT_PACE = {
    "speed_cost": speed_cost,
    "accel_cost": accel_limit_cost,
    "jerk_cost": jerk_limit_cost,
}

ALLOWED = dict(DOCUMENTED)

# 동역학 전용 질의에서 알려주는 것들. 공간 회피는 후보 선택이 이미 담당한다.
DYNAMICS_ONLY = {
    "pace_cost": None,          # 아래에서 채운다 (정의 순서 때문)
    "arrive_cost": None,
    "speed_limit_cost": speed_limit_cost,
    "accel_limit_cost": accel_limit_cost,
    "jerk_limit_cost": jerk_limit_cost,
}

# ---------- 경로 위 구간별 속도 ----------

# 계획된 경로. pace_cost 는 궤적점을 이 경로에 투영해 "몇 % 지점" 을 구한다.
# 롤아웃 안에서의 비율이 아니라 경로 전체에서의 비율이어야, VLM 이 그림을 보고
# 말한 "중간쯤" 과 같은 것을 가리킨다.
_PATH = None
_PATH_S = None


def set_path(path_xy):
    """계획 경로를 등록한다. (N,2) 월드 좌표."""
    global _PATH, _PATH_S
    if path_xy is None or len(np.asarray(path_xy)) < 2:
        _PATH, _PATH_S = None, None
        return
    p = np.asarray(path_xy, dtype=float)[:, :2]
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    total = float(seg.sum())
    _PATH = p
    _PATH_S = np.concatenate([[0.0], np.cumsum(seg)]) / (total if total > 1e-9 else 1.0)


def _percent_of(traj):
    """궤적 각 점이 경로의 몇 % 지점인가 (K, N), 0~100."""
    if _PATH is None:
        # 경로가 없으면 궤적 자체의 진행 비율로 대신한다. 그래야 단위 테스트나
        # 계획 없이 쓰는 경우에도 값이 정의된다.
        n = traj.p.shape[1]
        return np.tile(np.linspace(0.0, 100.0, n)[None, :], (traj.p.shape[0], 1))
    d = np.linalg.norm(traj.p[:, :, None, :] - _PATH[None, None, :, :], axis=-1)
    return _PATH_S[np.argmin(d, axis=2)] * 100.0


def pace_cost(traj, from_pct, to_pct, speed=None, accel=None, jerk=None, scale=None):
    """경로의 from_pct~to_pct 구간을 이 배수들로 지난다.

    세 배수 모두 "평상시" 를 1 로 둔 상대값이다. 평상시란 주변에 아무것도 없을
    때의 속도(NOMINAL_V) 와 그때 쓰는 가감속(A_MAX)·저크(J_MAX) 다.

        speed=0.25   평상시의 1/4 속도로
        accel=0.5    평상시의 절반 가감속으로 (더 일찍, 더 부드럽게 줄인다)
        jerk=0.5     가감속이 바뀌는 속도도 절반으로

    속도만 낮추고 가감속을 그대로 두면 "급제동으로 0.25 배에 도달" 이 되는데,
    그건 조심스럽게 지나가라는 뜻이 아니다. 그래서 셋을 함께 받는다.

    반환은 세 축의 상대 초과분 중 최악이다: <=0 이면 만족, >0 이면 위반량.
    ``scale`` 은 옛 이름으로 speed 와 같다.
    """
    sp = float(speed if speed is not None else (scale if scale is not None else 1.0))
    ac = float(accel if accel is not None else sp)
    jk = float(jerk if jerk is not None else sp)
    if sp <= 0:
        return _OFF
    a, b = float(from_pct), float(to_pct)
    # 0~1 로 오면 비율로 읽는다. 프롬프트가 "퍼센트" 라고 해도 0.25 를 쓰는
    # 응답이 나온다 (실측). 25% 를 0.25 로 읽으면 감속 구간이 사실상 사라진다.
    if 0.0 < b <= 1.0 and 0.0 <= a <= 1.0:
        a, b = a * 100.0, b * 100.0
    if b <= a:
        # 빈 구간은 "전 구간" 으로 읽는다 (실측: from=0, to=0 으로 전체를 뜻했다).
        a, b = 0.0, 100.0

    pct = _percent_of(traj)
    inside_v = (pct[:, 1:] >= a) & (pct[:, 1:] <= b)
    worst = np.full(traj.p.shape[0], -np.inf)

    over_v = np.where(inside_v, (traj.v - sp * NOMINAL_V) / (sp * NOMINAL_V), -np.inf)
    worst = np.maximum(worst, np.max(over_v, axis=1))
    if ac > 0 and traj.a.shape[1]:
        m = inside_v[:, 1:]
        over_a = np.where(m, (traj.a - ac * A_MAX) / (ac * A_MAX), -np.inf)
        worst = np.maximum(worst, np.max(over_a, axis=1))
    if jk > 0 and traj.j.shape[1]:
        m = inside_v[:, 2:]
        over_j = np.where(m, (traj.j - jk * J_MAX) / (jk * J_MAX), -np.inf)
        worst = np.maximum(worst, np.max(over_j, axis=1))
    # 정확히 상한인 경우는 만족으로 본다. 부동소수 때문에 +1e-17 같은 값이
    # 나오는데, 그것이 "위반" 으로 찍히면 지킨 궤적과 넘긴 궤적이 구분되지 않는다.
    worst = np.where(np.abs(worst) < 1e-9, -1e-9, worst)
    return traj._out(np.where(np.isfinite(worst), worst, _OFF))


def pace_profile(specs, n=101):
    """[(from_pct, to_pct, scale), ...] -> 경로 0~100% 의 배수 곡선 (n,).

    겹치는 구간은 평균이다. 아무 제약도 없는 지점은 1.0 (평상시).
    """
    grid = np.linspace(0.0, 100.0, n)
    acc = np.zeros(n)
    cnt = np.zeros(n)
    for spec in specs or []:
        a, b, sc = spec[0], spec[1], spec[2]
        m = (grid >= float(a)) & (grid <= float(b))
        acc[m] += float(sc)
        cnt[m] += 1
    out = np.ones(n)
    hit = cnt > 0
    out[hit] = acc[hit] / cnt[hit]
    return grid, out

def arrive_cost(traj, scale):
    """구간 끝에 도착할 때의 속도가 평상시의 scale 배 이하인가.

    PIVOT 의 sub-goal 제약에 해당한다 - 구간 내내가 아니라 끝에서만 만족하면
    된다. 이것이 다음 구간의 시작 조건이 되므로, 구간별 배수가 하나의 연속된
    속도 스케줄로 이어진다. "아기 옆에서는 멈춰 있어라" 는 scale=0 에 가깝다.
    """
    scale = float(scale)
    lim = max(scale, 1e-3) * NOMINAL_V
    v_end = traj.v[:, -1] if traj.v.shape[1] else np.zeros(traj.p.shape[0])
    return traj._out((v_end - lim) / lim)

DYNAMICS_ONLY["pace_cost"] = pace_cost
DYNAMICS_ONLY["arrive_cost"] = arrive_cost
ALLOWED["pace_cost"] = pace_cost
# margins_used 가 인자를 뽑으려면 ALLOWED 에 있어야 한다.
ALLOWED.update(PIVOT_PACE)
ALLOWED["arrive_cost"] = arrive_cost

# 기계의 한계. 장면과 무관한 로봇의 성질이라 VLM 에게 묻지 않는다.
A_MAX = float(os.environ.get("PACE_A_MAX", "0.4"))     # m/s^2
J_MAX = float(os.environ.get("PACE_J_MAX", "1.0"))     # m/s^3


def feasible_profile(specs, path_len_m, n=201, a_max=None):
    """배수 곡선을 실제로 낼 수 있는 속도 곡선으로 바꾼다.

    구간 경계에서 배수가 1.0 -> 0.25 로 뚝 떨어지면 그 자리에서 갑자기 느려질 수
    없다. 경계 이전부터 줄여야 도달할 수 있고, 얼마나 이전인지는 판단이 아니라
    물리다: v1^2 = v0^2 - 2*a*d.

    앞뒤로 한 번씩 훑는다 - 뒤에서 앞으로는 감속 가능성을, 앞에서 뒤로는 가속
    가능성을 강제한다. 궤적 생성에서 쓰는 속도 프로파일링과 같은 방식이다.

    Returns: (거리 격자 (n,), 속도 m/s (n,))
    """
    a0 = float(A_MAX if a_max is None else a_max)
    grid_pct, scale = pace_profile(specs, n=n)
    scale = np.asarray(scale, dtype=float)
    v = scale * NOMINAL_V
    ds = float(path_len_m) / max(n - 1, 1)

    # 가속도 상한도 배수를 따라간다. VLM 이 "여기는 0.25 배" 라고 했을 때 그 뜻은
    # 느리게만이 아니라 조심스럽게 지나가라는 것이다 - 급제동으로 0.25 배에
    # 도달하는 것은 그 뜻이 아니다. 배수 하나로 속도와 부드러움이 함께 정해지므로
    # VLM 에게 a_max 를 따로 묻지 않아도 된다 (물어도 24 회 중 1 회만 답했다).
    # 바닥을 두는 이유: 배수가 작아질수록 감속이 한없이 부드러워지면 제동 거리가
    # 경로보다 길어진다.
    # 가감속·저크 배수는 VLM 이 직접 준다. 안 주면 속도 배수를 따라간다.
    a_mul = np.ones(n); j_mul = np.ones(n)
    for spec in specs or []:
        m = (grid_pct >= float(spec[0])) & (grid_pct <= float(spec[1]))
        a_mul[m] = float(spec[3]) if len(spec) > 3 else float(spec[2])
        j_mul[m] = float(spec[4]) if len(spec) > 4 else float(spec[2])
    a_arr = a0 * np.clip(a_mul, 0.35, 1.25)

    # 감속: 뒤에서 앞으로. 앞 지점이 너무 빠르면 뒤 지점에 맞춰 낮춘다.
    # 감속이 이어지는 동안에는 목표 구간의 상한을 계속 쓴다. 한 칸만 전파하면
    # 그 앞은 다시 평상시 가감속이 되어, 결국 구간 직전에 급제동하게 된다.
    a_run = a_arr[-1]
    for i in range(n - 2, -1, -1):
        a_run = min(a_run, a_arr[i + 1]) if v[i] > v[i + 1] + 1e-9 else a_arr[i + 1]
        cand = float(np.sqrt(v[i + 1] ** 2 + 2 * a_run * ds))
        if cand < v[i]:
            v[i] = cand
        else:
            a_run = a_arr[i]
    # 가속: 앞에서 뒤로. 뒤 지점이 갑자기 빨라질 수 없다.
    for i in range(1, n):
        v[i] = min(v[i], float(np.sqrt(v[i - 1] ** 2 + 2 * a_arr[i - 1] * ds)))

    # 저크 제한. 위의 두 번 훑기는 가속도의 크기만 묶는다 - 가속도가 0 에서
    # a_max 로 한 칸에 튀는 것은 그대로 통과하고, 그게 실측에서 J_max 190~250 으로
    # 나타났다. 거리축 곡선을 시간축으로 옮겨 가속도 변화율을 제한한 뒤 되돌린다.
    j0 = float(os.environ.get("PACE_J_MAX", str(J_MAX)))
    j_arr = j0 * np.clip(j_mul, 0.35, 1.25)
    j = j0
    if j > 0 and n > 4:
        v = np.maximum(v, 1e-3)
        dt = ds / v                       # 각 칸을 지나는 데 걸리는 시간
        acc = np.gradient(v, np.cumsum(dt))
        for _ in range(3):                # 몇 번 훑으면 충분히 매끄러워진다
            for i in range(1, n):
                lim = acc[i - 1] + j_arr[i] * dt[i]
                acc[i] = min(acc[i], lim)
            for i in range(n - 2, -1, -1):
                acc[i] = min(acc[i], acc[i + 1] + j_arr[i] * dt[i])
        # 제한된 가속도로 속도를 다시 쌓되, 원래 상한을 넘지 않게 한다.
        # 아래로도 바닥을 둔다 - 저크를 묶다 보면 음의 가속이 계속 쌓여 속도가
        # 0 으로 무너진다 (실측: x0.25 구간에서 0.001 m/s 까지 내려갔다).
        # 요청된 배수의 70% 아래로는 내려가지 않게 한다.
        floor = 0.7 * np.maximum(scale, 0.05) * NOMINAL_V
        v2 = np.empty_like(v)
        v2[0] = v[0]
        for i in range(1, n):
            v2[i] = min(v[i], max(floor[i], v2[i - 1] + acc[i - 1] * dt[i]))
        v = np.minimum(v2, v)
    return grid_pct * float(path_len_m) / 100.0, v
