"""정규화·보간·경로길이. PIVOT 의 utils.py 에서 2D 로 옮긴 것들.

PIVOT 은 6D 자세(위치 3 + 오일러 3)를 쓰지만 우리는 (x, y, yaw) 3D 다. 회전이 한 축뿐이라
쿼터니언 구면보간이 필요 없고 각도 하나를 감아 주면 된다 - 그 "감아 준다" 를 한 군데로
모으는 것이 이 파일의 목적이다. 각도 wrap 은 여러 곳에서 조용히 틀리는 종류의 계산이다.
"""
import numpy as np


def wrap_angle(a):
    """각도를 (-pi, pi] 로. 스칼라도 배열도 받는다."""
    return np.arctan2(np.sin(a), np.cos(a))


def angle_diff(a, b):
    """a - b 를 감아서. |결과| <= pi."""
    return wrap_angle(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def normalize_vars(x, bounds):
    """실제 값 -> [-1, 1]. PIVOT 의 normalize_vars 와 같다.

    최적화기에 원래 단위(미터와 라디안)를 그대로 주면 축마다 눈금이 달라 한 축만
    움직이는 해가 나온다.
    """
    lo = np.asarray([b[0] for b in bounds], dtype=float)
    hi = np.asarray([b[1] for b in bounds], dtype=float)
    return 2.0 * (np.asarray(x, dtype=float) - lo) / np.maximum(hi - lo, 1e-9) - 1.0


def unnormalize_vars(z, bounds):
    """[-1, 1] -> 실제 값."""
    lo = np.asarray([b[0] for b in bounds], dtype=float)
    hi = np.asarray([b[1] for b in bounds], dtype=float)
    return lo + (np.asarray(z, dtype=float) + 1.0) * 0.5 * (hi - lo)


def num_control_points(start_pose, end_pose, pos_step_m, rot_step_rad, lo=3, hi=6):
    """제어점 개수. PIVOT 의 get_linear_interpolation_steps 를 2D 로.

    PIVOT 의 pos_step 0.20 m 는 0.55 m 작업공간 기준이다. 6 m 주방에 그대로 쓰면 늘
    상한 6 개로 포화해 가장 비싼 분기만 탄다 - config 에서 0.60 m 로 올려 잡는다.
    """
    d = float(np.linalg.norm(np.asarray(end_pose)[:2] - np.asarray(start_pose)[:2]))
    dth = float(abs(angle_diff(end_pose[2], start_pose[2])))
    n = max(int(np.ceil(d / max(pos_step_m, 1e-6))),
            int(np.ceil(dth / max(rot_step_rad, 1e-6))), 1) + 1
    return int(np.clip(n, lo, hi))


def interpolate(poses, n):
    """제어점 -> 조밀한 자세 열 (n, 3). 위치는 선형, 각도는 감아서 선형.

    각도를 그냥 선형보간하면 179 도에서 -179 도로 갈 때 358 도를 도는 경로가 나온다.
    """
    p = np.asarray(poses, dtype=float).reshape(-1, 3)
    if len(p) == 1:
        return np.repeat(p, n, axis=0)
    seg = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    s = cum / total if total > 1e-9 else np.linspace(0.0, 1.0, len(p))
    t = np.linspace(0.0, 1.0, int(n))
    xy = np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], axis=1)

    # 각도는 누적 차분으로 편다 - 편 값에서 보간한 뒤 다시 감는다.
    unwrapped = np.concatenate([[p[0, 2]], p[0, 2] + np.cumsum(angle_diff(p[1:, 2], p[:-1, 2]))])
    th = wrap_angle(np.interp(t, s, unwrapped))
    return np.concatenate([xy, th[:, None]], axis=1)


def path_length(poses, rot_weight=1.0):
    """(위치 길이, 회전 길이). PIVOT 의 path_length 와 같은 분해."""
    p = np.asarray(poses, dtype=float).reshape(-1, 3)
    if len(p) < 2:
        return 0.0, 0.0
    pos = float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum())
    rot = float(np.abs(angle_diff(p[1:, 2], p[:-1, 2])).sum()) * float(rot_weight)
    return pos, rot


def consistency(pose, ref, rot_weight=1.5):
    """기준 자세에서 얼마나 벗어났나. PIVOT 의 consistency 와 같은 뜻.

    subgoal 이 주방 저편으로 순간이동하는 것을 막는다. 조작에서보다 네비게이션에서
    더 중요하다 - 이것이 "단계" 를 한 걸음으로 만드는 유일한 항이다.
    """
    p = np.asarray(pose, dtype=float)
    r = np.asarray(ref, dtype=float)
    return float(np.linalg.norm(p[:2] - r[:2])
                 + float(rot_weight) * abs(float(angle_diff(p[2], r[2]))))


def turn_cost(cur_pose, next_pose):
    """제자리 회전이 얼마나 필요한가. [0, 1] 로 정규화.

    PIVOT 의 IK 비용(가중치 20)이 하던 일은 "물리적으로 못 가는 자세를 내지 마라" 다.
    홀로노믹 베이스에서는 자유공간이면 어디든 갈 수 있으므로 도달성이 아니라
    *싸게 도달 가능한가* 가 대응한다 - 돌아야 하는 각도의 합이다.

    PIVOT 의 num_descents/max_iterations 처럼 [0,1] 로 맞춰야 가중치 20 이 그대로 옮겨진다.
    """
    c = np.asarray(cur_pose, dtype=float)
    n = np.asarray(next_pose, dtype=float)
    d = n[:2] - c[:2]
    if float(np.linalg.norm(d)) < 1e-6:
        # 제자리면 자세 차이만 본다.
        return float(abs(float(angle_diff(n[2], c[2])))) / (2.0 * np.pi)
    heading = float(np.arctan2(d[1], d[0]))
    turn = abs(float(angle_diff(heading, c[2]))) + abs(float(angle_diff(n[2], heading)))
    return float(turn / (2.0 * np.pi))


def heading_change(poses, min_seg_m=0.02):
    """진행 방향이 구간마다 얼마나 꺾이는가. 평균 |Δheading| / π 로 [0,1].

    아주 짧은 구간은 뺀다 - 길이가 0 에 가까우면 arctan2 가 잡음이 되어, 실제로는
    곧은 경로에도 큰 값이 붙는다.

    yaw 가 아니라 **이동 방향**을 본다. 홀로노믹 베이스는 둘이 분리되어 있어
    turn_cost(yaw)로는 공간적 지그재그가 전혀 잡히지 않는다.
    """
    p = np.asarray(poses, dtype=float).reshape(-1, 3)[:, :2]
    seg = np.diff(p, axis=0)
    keep = np.linalg.norm(seg, axis=1) > float(min_seg_m)
    seg = seg[keep]
    if len(seg) < 2:
        return 0.0
    h = np.arctan2(seg[:, 1], seg[:, 0])
    return float(np.mean(np.abs(angle_diff(h[1:], h[:-1]))) / np.pi)
