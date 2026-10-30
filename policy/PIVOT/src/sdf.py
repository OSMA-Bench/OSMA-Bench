"""점유 격자 -> 부호거리장.

PIVOT 은 시뮬레이터가 주는 SDF 복셀을 그대로 받지만(`env.get_sdf_voxels`), 우리에게는
2D 점유 격자(`avoidance_map`)뿐이다. 거리 변환으로 만든다.

최적화기가 이 값을 미분해 쓰므로 두 가지가 중요하다: 부호가 맞을 것, 그리고 매끄러울 것.
실측(122x112 실제 맵): 범위 -1.09 ~ 1.80 m, 기울기 최대 0.085/셀.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

CELL_M = 0.05          # candidates.CELL_M 과 같아야 한다


def build(avoidance_map, cell_m=CELL_M, threshold=0.5):
    """부호거리장 보간기를 만든다.

    Args:
        avoidance_map: (H, W) 점유도 0~1. VoxPoser 가 가우시안으로 흐려 둔 맵이다.

    Returns:
        (interp, sdf_m) — interp((row, col)) -> 미터. 자유공간 양수, 장애물 안 음수.

    부호를 양쪽에서 만드는 이유: 한쪽만 하면 장애물 안이 전부 0 이 되어, 최적화기가
    "벽 속" 과 "벽에 딱 붙음" 을 구분하지 못한다. 벽으로 파고드는 해가 나온다.

    흐린 원본값을 섞는 안을 시험했다가 뺐다. "문턱을 씌우면 먼 곳의 기울기가 사라진다" 는
    걱정이었는데, 거리 변환이 이미 자유공간 전체에 기울기를 준다. 섞으면 경계 근처
    자유칸(거리 0.05 m)에서 0.2 x 0.4 를 빼 부호가 뒤집혔다 - 최적화기에게는
    "자유공간인데 장애물 안" 이 된다.
    """
    occ = np.asarray(avoidance_map, dtype=float) > threshold
    sdf_m = (distance_transform_edt(~occ) - distance_transform_edt(occ)) * float(cell_m)
    h, w = sdf_m.shape
    interp = RegularGridInterpolator(
        (np.arange(h, dtype=float), np.arange(w, dtype=float)), sdf_m,
        bounds_error=False,
        # 맵 밖은 "장애물 안" 으로 친다. PIVOT 은 0 을 쓰지만 그러면 최적화기가
        # 맵 밖이 공짜임을 발견해 subgoal 을 주방 밖에 세운다.
        fill_value=-1.0,
    )
    return interp, sdf_m


def collision_cost(interp, cells, radius_m, margin_m):
    """이 점들이 얼마나 침범했는가. PIVOT 의 calculate_collision_cost 와 같은 꼴.

    PIVOT 은 잡은 물체의 점군을 자세마다 변환해 검사하지만, 우리 로봇은 등방 원판이라
    중심 한 점이면 충분하다 - 테두리 8 점을 재도 값이 같고 보간 호출만 8 배가 된다.
    """
    d = np.asarray(interp(np.atleast_2d(cells)), dtype=float)
    return float(np.clip(float(radius_m) + float(margin_m) - d, 0.0, None).sum())
