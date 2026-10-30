"""통행 가능한 격자 후보 생성.

VLM 에게 좌표를 만들게 하지 않고 **고르게** 하는 것이 이 방식의 핵심이다. 그러려면
후보가 애초에 갈 수 있는 칸이어야 한다. 장애물 지도를 로봇 반경만큼 팽창시킨 뒤
그 바깥만 남기므로, 벽 안이나 장애물 위를 고르는 출력이 구조적으로 불가능하다.

costmap 방식과 **같은 장애물 정보**(avoidance_map)를 쓰되 사용 방식만 다르다는
점이 중요하다. 그래야 두 방식의 비교가 "정보 차이"가 아니라 "사용 방식 차이"가
된다.
"""
import numpy as np
from scipy.ndimage import binary_dilation

# 플래너 격자 한 칸의 크기 (m). robocasa_config 의 5 cm 격자와 맞춘다.
CELL_M = 0.05
# 로봇 반경 팽창 (칸). 3칸 = 15 cm.
DEFAULT_INFLATE_CELLS = 3


def free_mask(avoidance_map, inflate_cells=DEFAULT_INFLATE_CELLS, threshold=0.5):
    """통행 가능 칸의 불리언 마스크.

    avoidance_map 은 장애물 근처가 1 에 가까운 맵이다. threshold 로 이진화한 뒤
    팽창시켜 로봇 반경을 반영한다.
    """
    obs = np.asarray(avoidance_map) > threshold
    if inflate_cells > 0:
        obs = binary_dilation(obs, iterations=int(inflate_cells))
    return ~obs


def grid_candidates(avoidance_map, spacing_m=0.5, inflate_cells=DEFAULT_INFLATE_CELLS,
                    bounds=None):
    """일정 간격 격자에서 통행 가능한 칸만 뽑는다.

    Args:
        spacing_m: 후보 간격. 0.5 m 면 실측상 지도당 75~88개가 나온다.
        bounds: (r0, r1, c0, c1) 로 범위를 제한. 계층 질의의 촘촘한 단계에서
            직전에 고른 점 주변만 보여줄 때 쓴다. None 이면 지도 전체.

    Returns:
        (n, 2) int 배열 — 격자 (row, col).
    """
    free = free_mask(avoidance_map, inflate_cells)
    step = max(1, int(round(spacing_m / CELL_M)))
    H, W = free.shape
    r0, r1, c0, c1 = bounds if bounds else (0, H, 0, W)
    r0, c0 = max(0, int(r0)), max(0, int(c0))
    r1, c1 = min(H, int(r1)), min(W, int(c1))

    rows = np.arange(r0, r1, step)
    cols = np.arange(c0, c1, step)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    pts = np.stack([rr.ravel(), cc.ravel()], axis=1)
    keep = free[pts[:, 0], pts[:, 1]]
    return pts[keep]


def stratified(cands, max_n, center):
    """열린 방향을 고르게 덮도록 뽑는다.

    ``subsample`` 은 인덱스를 균등 추출한다. 후보 배열이 행->열 순서라
    그 균등은 공간적으로 균등하지 않다 - 실측: 25개에서 10개를 뽑았더니
    방향 180도가 3개, 거리 0.75m 가 3개로 한쪽에 몰렸다.

    여기서는 중심에서 본 각도를 max_n 개 부채꼴로 나누고, 부채꼴마다 하나씩
    돌아가며 가져온다. 부채꼴 안에서는 먼 것을 먼저 준다 - 한 걸음으로 더
    나아가는 쪽이 낭비가 적다. 빈 부채꼴의 몫은 남은 부채꼴이 나눠 갖는다.
    """
    cands = np.asarray(cands)
    if len(cands) <= max_n:
        return cands
    d = cands.astype(float) - np.asarray(center, dtype=float)[None, :]
    r = np.hypot(d[:, 0], d[:, 1])
    th = np.arctan2(d[:, 0], d[:, 1])                 # (row, col) 기준 각도
    sec = np.floor((th + np.pi) / (2 * np.pi) * max_n).astype(int)
    sec = np.clip(sec, 0, max_n - 1)

    buckets = {}
    for i, sctr in enumerate(sec):
        buckets.setdefault(int(sctr), []).append(i)
    for key in buckets:                                # 먼 것부터
        buckets[key].sort(key=lambda i: -r[i])

    out, keys = [], sorted(buckets)
    while len(out) < max_n and keys:
        for key in list(keys):
            if not buckets[key]:
                keys.remove(key)
                continue
            out.append(buckets[key].pop(0))
            if len(out) >= max_n:
                break
    return cands[np.asarray(out, dtype=int)]


def subsample(cands, max_n, keep=()):
    """후보가 너무 많을 때 균등하게 솎아 낸다.

    VLM 이 선택에 실패했을 때의 재시도 수단이다. 번호가 적은 이미지는 읽기 쉽고
    형식 오류가 줄어든다. ``keep`` 에 준 점(현재 위치·목표 근처 등)은 항상 남긴다.

    공간적으로 고르게 남기려고 격자 순서가 아니라 좌표 기준으로 나눈다 — 앞에서
    자르면 지도 한쪽만 남는다.
    """
    cands = np.asarray(cands)
    if len(cands) <= max_n:
        return cands
    keep = np.asarray(keep).reshape(-1, 2) if len(keep) else np.empty((0, 2), int)
    n_slots = max(1, max_n - len(keep))
    idx = np.linspace(0, len(cands) - 1, n_slots).round().astype(int)
    picked = cands[np.unique(idx)]
    if len(keep):
        picked = np.vstack([keep, picked])
        _, first = np.unique(picked, axis=0, return_index=True)
        picked = picked[np.sort(first)]
    return picked[:max_n]


def nearest_candidate(cands, rc):
    """주어진 격자 좌표에 가장 가까운 후보. 시작·목표를 후보에 붙일 때 쓴다."""
    cands = np.asarray(cands)
    if len(cands) == 0:
        return None
    d = np.hypot(cands[:, 0] - rc[0], cands[:, 1] - rc[1])
    return cands[int(np.argmin(d))]


def local_bounds(rc, radius_m, map_shape):
    """한 점 주변 정사각 범위를 격자 인덱스로. 계층 질의의 촘촘한 단계용."""
    r = int(round(radius_m / CELL_M))
    H, W = map_shape
    return (max(0, rc[0] - r), min(H, rc[0] + r + 1),
            max(0, rc[1] - r), min(W, rc[1] + r + 1))


def segment_clear(free, a, b, samples=None):
    """두 격자점을 잇는 직선이 통행 가능 영역 안에 있는가.

    점 사이를 촘촘히 샘플링해 모두 free 인지 본다. 대각선 이동이 장애물 모서리를
    스치는 경우를 잡기 위해 샘플 간격을 한 칸보다 작게 둔다.
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n = samples or max(2, int(np.hypot(*(b - a)) * 2) + 1)
    ts = np.linspace(0.0, 1.0, n)
    pts = np.round(a[None, :] + ts[:, None] * (b - a)[None, :]).astype(int)
    H, W = free.shape
    pts[:, 0] = np.clip(pts[:, 0], 0, H - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, W - 1)
    return bool(free[pts[:, 0], pts[:, 1]].all())

def reachable(free, cur, cands, radius_m, spacing_m=None, min_step_m=0.5):
    """직선으로 갈 수 있고 반경 안에 있는 후보만 남긴다.

    두 가지를 거른다:
      - 정사각 bounds 의 모서리. bounds 는 사각형이라 반경 0.75 m 로 잘라도
        대각선 방향은 1.06 m 까지 들어온다. "한 걸음" 의 길이가 방향마다
        달라지면 VLM 이 고른 번호의 의미도 달라진다.
      - 사이가 막힌 점. 조리대 건너편 칸은 지도상 통행 가능이지만 이번 걸음에
        갈 수 있는 곳이 아니다. 갈 수 없는 곳에 번호를 붙이면 그 번호를 고른
        만큼이 그대로 낭비된다.
    """
    cands = np.asarray(cands)
    if len(cands) == 0:
        return cands
    cur = np.asarray(cur, dtype=float)
    d = np.linalg.norm(cands.astype(float) - cur[None, :], axis=1) * CELL_M
    # 바로 옆 칸은 뺀다. 한 걸음이 0.25 m 면 그림에서 로봇 표식과 겹쳐 번호를
    # 읽을 수 없고, 골라 봐야 그 구간이 거의 낭비된다. 남는 것이 없으면 되돌린다.
    keep = (d <= radius_m + 1e-6) & (d >= min_step_m - 1e-6)
    if not keep.any():
        keep = d <= radius_m + 1e-6
    out = [rc for rc, k in zip(cands, keep) if k and segment_clear(free, cur, rc)]
    return np.asarray(out) if out else np.empty((0, 2), dtype=int)

def drop_recent(cands, recent, radius_cells=2):
    """최근에 지나온 자리 근처를 후보에서 뺀다.

    되돌아갈 수 있게 두면 국소 최소에서 진동한다 - 실측: 직진이 조리대에 막히자
    -90도/+90도를 번갈아 0.25 m 씩 오가며 남은 구간을 전부 태웠다. 목표까지의
    거리가 같은 두 후보가 서로를 부르는 구조라 프롬프트로는 끊기지 않는다.

    반경을 두는 이유: 정확히 같은 칸만 빼면 옆 칸으로 한 칸 비껴 같은 진동을
    한다. 2 칸(=10 cm)이면 같은 자리로 치기에 충분하다.
    """
    cands = np.asarray(cands)
    if len(cands) == 0 or not len(recent):
        return cands
    rec = np.asarray(recent, dtype=float).reshape(-1, 2)
    d = np.linalg.norm(cands.astype(float)[:, None, :] - rec[None, :, :], axis=2)
    keep = d.min(axis=1) > radius_cells
    out = cands[keep]
    # 전부 걸러지면 원본을 돌려준다 - 갈 곳이 없다고 실패시키는 것보다
    # 되돌아가더라도 움직이는 편이 낫다 (막다른 골목에서 빠져나오려면 필요하다).
    return out if len(out) else cands

def require_progress(cands, goal_cell, cur, best_d=None, slack_m=0.30, budget_m=0.60):
    """목표에서 크게 멀어지는 후보를 뺀다.

    VLM 은 그림만으로 "어느 원이 목표에 더 가까운가" 를 잘 판단하지 못한다 -
    실측: 아기 과제에서 16 구간 동안 목표까지 38 칸까지 멀어졌다가 16 칸으로
    돌아오기를 반복해 직선거리 5 m 를 9.33 m 로 걸었다. 되돌아간 자리가 정확히
    같은 칸은 아니라 drop_recent 로는 걸리지 않는다.

    엄격한 단조 전진은 우회를 막는다 - 장애물을 돌려면 일시적으로 멀어져야 한다.
    그래서 두 가지 여유를 둔다:
        slack_m:  현재 거리보다 이만큼까지는 멀어져도 좋다 (옆으로 비키기)
        budget_m: 지금까지의 최선 거리보다 이만큼 넘게 멀어지지는 않는다

    전부 걸러지면 원본을 돌려준다. 갈 곳이 없다고 실패시키는 것보다, 멀어지더라도
    움직이는 편이 낫다.
    """
    cands = np.asarray(cands)
    if len(cands) == 0:
        return cands
    g = np.asarray(goal_cell, dtype=float)
    d = np.linalg.norm(cands.astype(float) - g[None, :], axis=1) * CELL_M
    cur_d = float(np.linalg.norm(np.asarray(cur, dtype=float) - g)) * CELL_M
    lim = cur_d + slack_m
    if best_d is not None:
        lim = min(lim, float(best_d) + budget_m)
    out = cands[d <= lim]
    return out if len(out) else cands

def inflate_per_object(avoidance_map, obstacles, base_cells, cell_m=CELL_M):
    """대상마다 다른 여유로 팽창한 통행 가능 마스크.

    균일 팽창(로봇 반경 15cm)은 큰 대상에서 부족하다 - 실측: 목표에는 다 닿는데
    위반이 0.6m 티어에만 몰렸다(cat 21.8 / baby 26.1 / human 27.0%, 무생물 0.0%).
    속도를 줄여도 애초에 가까운 자리를 후보로 내주면 위반은 남는다.

    Args:
        obstacles: [(xy_cell, margin_m)] - VLM 이 대상별로 부른 여유.
        base_cells: 그 밖의 모든 것에 쓰는 기본 팽창(로봇 반경).
    """
    free = free_mask(avoidance_map, base_cells)
    if not obstacles:
        return free
    H, W = free.shape
    rr, cc = np.mgrid[0:H, 0:W]
    for xy, margin in obstacles:
        r = float(margin) / cell_m
        if r <= 0:
            continue
        d = np.hypot(rr - float(xy[0]), cc - float(xy[1]))
        free &= d > r
    return free
