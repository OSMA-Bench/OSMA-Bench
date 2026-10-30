"""장면의 물체를 문자 키포인트로 만든다.

VoxPoser 가 LMP 에게 주는 것과 **같은 목록**을 쓴다
(env.get_visible_object_names(), robocasa_env.py:196). 그래야 두 방식의 차이가
"정보 차이" 가 아니라 "그 정보를 어떻게 쓰는가" 로 좁혀진다. 후보 격자를 만들 때
costmap 과 같은 avoidance_map 을 쓰기로 한 것과 같은 원칙이다.

목록에는 위험도 표시가 없다. crawling_baby 와 microwave 가 나란히 있을 뿐이고,
어느 쪽을 피해야 하는지는 표시돼 있지 않다. 그 판단이 곧 측정 대상이다.

번호 공간을 분리한다: 후보 위치는 숫자 1..10, 물체는 문자 A/B/C, 목표는 별표.
같은 공간을 쓰면 "가라" 와 "피하라" 가 섞인다 - 장애물을 그리기만 했더니 모델이
그것을 목표로 읽어 위반이 0% 에서 72% 가 된 전례가 있다.
"""
import string

import numpy as np

# 점으로 표현되지 않는 것들. 방 전체(kitchen), 통로(door), 로봇 자신은
# "여기서 얼마 떨어져라" 가 의미를 갖지 않는다.
NOT_A_POINT = {
    "kitchen", "door", "floor", "wall", "robot_mobile_base", "mobilebase0",
}

LABELS = string.ascii_uppercase   # A, B, C, ...


def collect(lmp_env, goal_xy=None, max_n=len(LABELS), spread=()):
    """VoxPoser 의 objects 목록을 문자 키포인트로 바꾼다.

    Args:
        lmp_env: NavigationLMPInterface. detect(name) 로 위치를 조회한다.
        goal_xy: 목표 좌표. 있으면 GOAL 키포인트로 함께 넣는다.
        max_n: 라벨 개수 상한.

    Returns:
        (keypoints, skipped)
        keypoints: [{"label": "A", "name": "crawling_baby", "xy": array([x, y])}, ...]
        skipped:   [(이름, 사유)] - 위치 조회 실패 등. 로그에 남겨야 한다.
                   목록에 이름은 뜨는데 조회가 안 되면 VLM 이 이미지에서 보고도
                   지목할 수 없으므로, 조용히 넘어가면 안 된다.
    """
    names = _object_names(lmp_env)
    keypoints, skipped = [], []

    for name in names:
        if name in NOT_A_POINT:
            continue
        if len(keypoints) >= max_n - (1 if goal_xy is not None else 0):
            skipped.append((name, "라벨 상한 초과"))
            continue
        # 뻗은 형상은 여러 점으로 나눈다. 나머지는 중심 하나로 충분하다
        # (실측: 덩어리진 대상은 viol 0~5%, 사람·아기만 18~37%).
        if any(k in name.lower() for k in spread):
            pts = _spread_of(lmp_env, name)
            if not pts:
                skipped.append((name, "점군 분할 실패 - 단일 키포인트로 대체"))
            if pts:
                for j, xy in enumerate(pts):
                    if len(keypoints) >= max_n - (1 if goal_xy is not None else 0):
                        break
                    keypoints.append({"label": LABELS[len(keypoints)],
                                      "name": f"{name}#{j + 1}", "xy": xy})
                continue
        xy = _position_of(lmp_env, name)
        if xy is None:
            skipped.append((name, "위치 조회 실패"))
            continue
        keypoints.append({"label": LABELS[len(keypoints)], "name": name, "xy": xy})

    if goal_xy is not None:
        keypoints.append({"label": LABELS[len(keypoints)], "name": "GOAL",
                          "xy": np.asarray(goal_xy, dtype=float)[:2]})
    return keypoints, skipped


def as_mapping(keypoints):
    """constraints.set_keypoints() 에 넣을 {라벨: xy} 로 바꾼다."""
    return {k["label"]: k["xy"] for k in keypoints}


def avoidables(keypoints):
    """회피 대상이 될 수 있는 것만. 목표는 뺀다.

    목표는 별표로 이미 구분되는데 문자 목록에도 넣으면 VLM 이
    "GOAL 에서 0.5m 떨어져라" 를 쓴다 (실측). 가야 할 곳을 피하라는 제약이다.
    """
    return [k for k in keypoints if k.get("name") != "GOAL"]


def as_table(keypoints, robot_xy=None):
    """프롬프트에 넣을 문자↔이름 대응표.

    좌표도 거리도 주지 않는다. 숫자를 주면 그림을 보지 않고 계산으로 답할 수
    있게 되어, 장면 이해 능력을 측정할 수 없다. 무엇이 어디 있는지는 그림에
    마름모로 표시돼 있고, 이 표는 그 문자가 무슨 물건인지만 알려 준다.
    """
    return "\n".join(f"{k['label']} = {k['name']}" for k in avoidables(keypoints))


# ---------- 내부 ----------

def _object_names(lmp_env):
    """VoxPoser 가 LMP 에게 주는 목록. 실측 예:
    ['coffee_machine','kitchen','crawling_baby','dishwasher','door','fridge',
     'knife','microwave','robot_mobile_base','human','sink','stove','stove_main']
    """
    env = getattr(lmp_env, "_env", None)
    for src in (env, lmp_env):
        fn = getattr(src, "get_visible_object_names", None)
        if callable(fn):
            try:
                return list(fn())
            except Exception:
                pass
    # 훅이 값을 못 가져와도 파이썬은 조용히 넘어간다. 이 방식으로 버그 세 개를
    # 연달아 냈으므로 빈 목록을 돌려주되 호출자가 알아채게 한다.
    return []


def _spread_of(lmp_env, name, k=3):
    """대상을 여러 점으로 표현한다. 점군을 바닥에 투영해 k개로 나눈다.

    중심점 하나로는 팔다리가 뻗은 대상을 표현할 수 없다 - 실측: 사람 과제에서
    위반 294스텝의 중심거리가 0.56~1.39m 로 퍼져 있고 방향도 130도 범위에
    몰렸다. 등방 반경으로는 이 비대칭이 표현되지 않아 viol 37.2% 였다.
    반면 덩어리진 대상(kettlebell 0.0%, cat 0.4%)은 한 점으로 충분했다.

    점군을 못 얻으면 None - 호출자가 중심점 하나로 되돌아간다.
    """
    fn = getattr(lmp_env, "detect", None)
    if not callable(fn):
        return None
    try:
        obs = fn(name)
    except Exception:
        return None
    pts = None
    # 실제 키는 _point_cloud_world 다 (interfaces.py detect 반환).
    for key in ("_point_cloud_world", "_point_cloud", "point_cloud", "points"):
        cand = obs.get(key) if isinstance(obs, dict) else getattr(obs, key, None)
        if cand is not None:
            pts = np.asarray(cand, dtype=float)
            break
    if pts is None or pts.ndim != 2 or pts.shape[0] < k:
        return None
    xy = pts[:, :2]
    # 가장 긴 축으로 정렬해 k등분한다. 클러스터링보다 단순하고, 뻗은 방향을
    # 그대로 따라간다.
    c = xy.mean(axis=0)
    d = xy - c
    try:
        axis = np.linalg.svd(d, full_matrices=False)[2][0]
    except np.linalg.LinAlgError:
        return None
    t = d @ axis
    order = np.argsort(t)
    chunks = np.array_split(order, k)
    return [xy[ch].mean(axis=0) for ch in chunks if len(ch)]


def _position_of(lmp_env, name):
    """detect(name) -> 월드 xy. 실패하면 None.

    detect 는 fixture 면 fixture.pos 를, 움직이는 물체면 점군 중심을 준다
    (interfaces.py:357). 실패 방식이 조용하다 - 이름이 name2ids 에 없으면
    [0,0,0] 을 돌려준 전례가 있어, 원점 근처 값은 조회 실패로 본다.
    """
    fn = getattr(lmp_env, "detect", None)
    if not callable(fn):
        return None
    try:
        obs = fn(name)
    except Exception:
        return None
    pos = None
    for key in ("_position_world", "position"):
        if isinstance(obs, dict) and key in obs:
            pos = obs[key]
            break
        if hasattr(obs, key):
            pos = getattr(obs, key)
            break
    if pos is None:
        return None
    xy = np.asarray(pos, dtype=float).ravel()[:2]
    if xy.shape[0] < 2 or not np.all(np.isfinite(xy)):
        return None
    if float(np.linalg.norm(xy)) < 1e-6:
        return None          # [0,0,0] 폴백 = 조회 실패
    return xy


def clouds_of(lmp_env, keypoints, max_pts=64):
    """{라벨: (M,2)} 점군. 제약 헬퍼가 표면거리를 재는 데 쓴다.

    VLM 에게는 여전히 문자 하나만 보여 준다 - 점군은 헬퍼 안에서만 쓰인다.
    점군을 못 얻는 대상은 넣지 않는다. 그러면 그 대상만 중심+반경 보정으로
    되돌아간다 (조용히 0 이 되는 것보다 낫다).
    """
    out = {}
    for k in keypoints:
        if k.get("name") == "GOAL":
            continue
        fn = getattr(lmp_env, "detect", None)
        if not callable(fn):
            break
        try:
            obs = fn(k["name"].split("#")[0])
        except Exception:
            continue
        pts = None
        for key in ("_point_cloud_world", "_point_cloud", "point_cloud", "points"):
            cand = obs.get(key) if isinstance(obs, dict) else getattr(obs, key, None)
            if cand is not None:
                pts = np.asarray(cand, dtype=float)
                break
        if pts is None or pts.ndim != 2 or pts.shape[0] < 3:
            continue
        xy = pts[:, :2]
        if len(xy) > max_pts:                    # 균등 다운샘플 - 거리 계산은
            idx = np.linspace(0, len(xy) - 1, max_pts).astype(int)   # (K,N,M) 이라
            xy = xy[idx]                          # M 이 크면 그대로 비용이 된다
        out[k["label"]] = xy
    return out
