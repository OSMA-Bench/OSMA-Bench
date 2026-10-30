"""topview 픽셀 <-> 월드 좌표 변환.

VoxPoser 는 에피소드마다 플래너 격자의 네 모서리를 바닥 평면에 투영해
``topview_corners_uv`` 로 저장한다(interfaces.py). 그 네 점이 곧
평면-대-평면 대응이므로, 호모그래피 하나면 topview 픽셀과 월드 xy 를 양방향으로
오갈 수 있다. 새로 캘리브레이션할 것이 없다는 뜻이다.

대응 순서는 저장 시점과 같아야 한다:

    corners_uv[0] <- (ws_min_x, ws_min_y)   격자 (0, 0)
    corners_uv[1] <- (ws_min_x, ws_max_y)   격자 (0, W-1)
    corners_uv[2] <- (ws_max_x, ws_max_y)   격자 (H-1, W-1)
    corners_uv[3] <- (ws_max_x, ws_min_y)   격자 (H-1, 0)
"""
import numpy as np


def _solve_homography(src, dst):
    """네 점 대응에서 3x3 호모그래피를 푼다 (DLT).

    src, dst: (4, 2). 반환 H 는 ``dst ~ H @ [x, y, 1]`` 를 만족한다.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError(f"need 4x2 point sets, got {src.shape} and {dst.shape}")
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.asarray(A, dtype=float))
    H = Vt[-1].reshape(3, 3)
    if abs(H[2, 2]) < 1e-12:
        raise ValueError("degenerate homography (H[2,2] ~ 0)")
    return H / H[2, 2]


class TopviewFrame:
    """한 에피소드의 topview 이미지와 월드 평면 사이 변환.

    Args:
        corners_uv: (4, 2) 픽셀 좌표. voxposer_dump.npz 의 topview_corners_uv.
        ws_min, ws_max: 워크스페이스 경계 (x, y). dump 의 workspace_bounds_*[:2].
        map_shape: (H, W) 플래너 격자 크기. 격자 인덱스 변환에 쓴다.
    """

    def __init__(self, corners_uv, ws_min, ws_max, map_shape):
        self.corners_uv = np.asarray(corners_uv, dtype=float)
        self.ws_min = np.asarray(ws_min, dtype=float)[:2]
        self.ws_max = np.asarray(ws_max, dtype=float)[:2]
        self.map_h, self.map_w = int(map_shape[0]), int(map_shape[1])

        world_corners = np.array([
            [self.ws_min[0], self.ws_min[1]],
            [self.ws_min[0], self.ws_max[1]],
            [self.ws_max[0], self.ws_max[1]],
            [self.ws_max[0], self.ws_min[1]],
        ])
        self.H_world_to_uv = _solve_homography(world_corners, self.corners_uv)
        self.H_uv_to_world = np.linalg.inv(self.H_world_to_uv)

    @classmethod
    def from_dump(cls, npz):
        """voxposer_dump.npz (np.load 결과) 에서 만든다."""
        return cls(npz["topview_corners_uv"],
                   npz["workspace_bounds_min"], npz["workspace_bounds_max"],
                   (int(npz["map_h"]), int(npz["map_w"])))

    @staticmethod
    def _apply(H, pts):
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        homo = np.hstack([pts, np.ones((len(pts), 1))])
        out = homo @ H.T
        w = out[:, 2:3]
        if np.any(np.abs(w) < 1e-12):
            raise ValueError("point projects to infinity")
        return out[:, :2] / w

    def world_to_uv(self, xy):
        """월드 (x, y) -> topview 픽셀 (u, v)."""
        return self._apply(self.H_world_to_uv, xy)

    def uv_to_world(self, uv):
        """topview 픽셀 (u, v) -> 월드 (x, y)."""
        return self._apply(self.H_uv_to_world, uv)

    def cell_to_world(self, rc):
        """플래너 격자 (row, col) -> 월드 (x, y).

        행은 y, 열은 x 에 대응한다. 덤프의 affordance_map 무게중심을 실제
        goal_xy_fixture 와 맞춰 확인한 규약이다 (이 방향 평균 오차 0.5 m,
        행/열을 바꾸면 4.0 m). 왕복 변환만 시험하면 두 규약 모두 통과하므로
        반드시 외부 기준점과 대조할 것.
        """
        rc = np.atleast_2d(np.asarray(rc, dtype=float))
        fy = rc[:, 0] / max(self.map_h - 1, 1)
        fx = rc[:, 1] / max(self.map_w - 1, 1)
        x = self.ws_min[0] + fx * (self.ws_max[0] - self.ws_min[0])
        y = self.ws_min[1] + fy * (self.ws_max[1] - self.ws_min[1])
        return np.stack([x, y], axis=1)

    def world_to_cell(self, xy):
        """월드 (x, y) -> 플래너 격자 (row, col), 실수값."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        fx = (xy[:, 0] - self.ws_min[0]) / max(self.ws_max[0] - self.ws_min[0], 1e-9)
        fy = (xy[:, 1] - self.ws_min[1]) / max(self.ws_max[1] - self.ws_min[1], 1e-9)
        return np.stack([fy * (self.map_h - 1), fx * (self.map_w - 1)], axis=1)

    def cell_to_uv(self, rc):
        """격자 (row, col) -> 픽셀 (u, v). 후보 번호를 이미지에 찍을 때 쓴다."""
        return self.world_to_uv(self.cell_to_world(rc))
