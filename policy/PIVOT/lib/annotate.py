"""topview 위에 후보 번호·목표·경로를 그린다.

PIVOT 의 시각적 프롬프팅과 같은 발상이다 — 좌표를 상상하게 하는 대신, 선택
가능한 것을 이미지에 직접 그려 넣고 번호로 고르게 한다. 읽히는 이미지를 만드는
것이 이 파일의 전부이고, 그게 성능을 크게 좌우한다.

렌더링 규칙 세 가지:
  - 번호는 점 위에 흰 글자 + 어두운 테두리. 주방 바닥이 밝고 어두운 곳이 섞여
    있어 한 가지 색으로는 어디선가 반드시 묻힌다.
  - 목표는 다른 모양(별)과 색으로. 번호와 같은 원으로 그리면 후보로 오인한다.
  - 현재 위치는 화살표로 방향까지. VLM 이 "앞/뒤"를 판단할 근거가 된다.
"""
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _font(size):
    for p in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_with_outline(draw, xy, text, font, fill=(255, 255, 255), outline=(0, 0, 0), w=2):
    x, y = xy
    for dx in range(-w, w + 1):
        for dy in range(-w, w + 1):
            if dx or dy:
                draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text((x, y), text, font=font, fill=fill)


def annotate(image, frame, cand_cells, goal_xy=None, robot_xy=None, robot_yaw=None,
             path_xy=None, obstacle_xy=None, obstacle_r_m=None, radius_px=9,
             label_pt=15, keypoints=None, leg_xy=None, route_xy=None):
    """후보가 번호와 함께 표시된 이미지를 만든다.

    Args:
        image: topview RGB (H, W, 3) ndarray 또는 PIL.Image
        frame: geometry.TopviewFrame
        cand_cells: (n, 2) 격자 후보. 1번부터 순서대로 번호가 붙는다.
        goal_xy: 목표 월드 좌표 — 별표로 표시
        robot_xy, robot_yaw: 현재 위치와 방향 — 화살표
        path_xy: 이미 정해진 경로 (제약 질의용) — 선으로
        obstacle_xy, obstacle_r_m: 장애물과 경계 반경 — annotate_obstacles 옵션

    Returns:
        (PIL.Image, labels) — labels[i] 는 i+1 번 라벨의 격자 좌표.
    """
    img = Image.fromarray(np.asarray(image).astype(np.uint8)) if not isinstance(image, Image.Image) else image.copy()
    img = img.convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    font = _font(label_pt)

    # 장애물 경계 (옵션) — 후보보다 먼저 그려 뒤에 깔리게 한다.
    if obstacle_xy is not None and obstacle_r_m:
        c = frame.world_to_uv(np.asarray(obstacle_xy).reshape(1, 2))[0]
        edge = frame.world_to_uv(np.asarray([obstacle_xy[0] + obstacle_r_m,
                                             obstacle_xy[1]]).reshape(1, 2))[0]
        r = float(np.linalg.norm(edge - c))
        d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r],
                  fill=(220, 60, 40, 55), outline=(220, 60, 40, 200), width=2)

    # 이미 정해진 경로 (제약 질의용)
    if path_xy is not None and len(path_xy) >= 2:
        uv = frame.world_to_uv(np.asarray(path_xy, dtype=float))
        # 흰 테두리를 깔고 그 위에 빨강. 나무 바닥 위에서 얇은 빨간 선은
        # 사라진다 - 대비를 주지 않으면 "지나온 길" 이 그림에 없는 것과 같다.
        # 지나온 길과 이번 구간은 색이 달라야 한다. 둘 다 빨강이면 어느 쪽이
        # "앞으로 갈 곳" 인지 그림만 보고는 알 수 없다.
        d.line([tuple(p) for p in uv], fill=(255, 255, 255, 190), width=7)
        d.line([tuple(p) for p in uv], fill=(150, 25, 25, 255), width=4)
        for q in uv[:-1]:
            d.ellipse([q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4],
                      fill=(150, 25, 25, 255), outline=(255, 255, 255, 220), width=1)
        # 진행 방향 화살촉을 중간중간 찍는다.
        for a0, b0 in zip(uv[:-1:2], uv[1::2]):
            v = np.asarray(b0, float) - np.asarray(a0, float)
            n = np.hypot(*v)
            if n < 12:
                continue
            v = v / n
            perp = np.array([-v[1], v[0]])
            m = (np.asarray(a0, float) + np.asarray(b0, float)) / 2
            d.polygon([tuple(m + v * 6), tuple(m - v * 4 + perp * 4),
                       tuple(m - v * 4 - perp * 4)], fill=(150, 25, 25, 255))

    # 앞으로 갈 경로 전체와 % 눈금. 구간을 "몇 % 에서 몇 % 까지" 로 말하려면
    # 그 눈금이 그림에 있어야 한다 - 번호만 있으면 "1 번 근처" 까지밖에 못 쓴다.
    if route_xy is not None and len(route_xy) >= 2:
        ruv = frame.world_to_uv(np.asarray(route_xy, dtype=float))
        # 지나온 길의 끝과 이번 구간의 시작을 이어 준다. 둘을 따로 그리면
        # 로봇 자리에서 선이 끊겨 "여기까지 와서 여기로 간다" 가 안 보인다.
        if path_xy is not None and len(path_xy) >= 1:
            _puv = frame.world_to_uv(np.asarray(path_xy, dtype=float))
            d.line([tuple(_puv[-1]), tuple(ruv[0])], fill=(255, 255, 255, 190), width=7)
            d.line([tuple(_puv[-1]), tuple(ruv[0])], fill=(150, 25, 25, 255), width=4)
        d.line([tuple(p) for p in ruv], fill=(255, 255, 255, 200), width=11)
        d.line([tuple(p) for p in ruv], fill=(255, 150, 0, 235), width=7)
        seg = np.linalg.norm(np.diff(np.asarray(route_xy, dtype=float), axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1]) or 1.0
        f_tick = _font(12)
        # 구간 끝을 주황 고리로. 프롬프트가 "orange ring" 이라 지목하므로
        # 그림에 그 표식이 있어야 한다.
        d.ellipse([ruv[-1][0] - 13, ruv[-1][1] - 13, ruv[-1][0] + 13, ruv[-1][1] + 13],
                  outline=(255, 255, 255, 235), width=5)
        d.ellipse([ruv[-1][0] - 13, ruv[-1][1] - 13, ruv[-1][0] + 13, ruv[-1][1] + 13],
                  outline=(255, 140, 0, 255), width=3)
        # 눈금 개수는 화면상 길이에 맞춘다. 한 걸음이 0.25 m 면 픽셀로 짧아
        # 다섯 개를 다 찍으면 글자가 서로 겹쳐 읽히지 않는다.
        _px = float(np.linalg.norm(ruv[-1] - ruv[0]))
        # 0% 눈금은 그리지 않는다 - 그 자리에 초록 고리(로봇)가 있고, 흰 점과
        # 글자가 지나온 길과 이번 구간의 이음매를 덮어 선이 끊겨 보인다.
        _ticks = (25, 50, 75, 100) if _px > 190 else (
            (50, 100) if _px > 90 else (100,))
        _text_with_outline(d, (ruv[0][0] - 34, ruv[0][1] - 26), "0%", _font(12),
                           fill=(255, 150, 0))
        for pct in _ticks:
            q = np.interp(pct / 100.0 * total, cum, np.arange(len(cum)))
            uvq = ruv[int(round(q))]
            d.ellipse([uvq[0] - 5, uvq[1] - 5, uvq[0] + 5, uvq[1] + 5],
                      fill=(255, 255, 255, 245), outline=(190, 90, 0, 255), width=2)
            _text_with_outline(d, (uvq[0] - 12, uvq[1] - 24), f"{pct}%", f_tick,
                               fill=(255, 150, 0))

    # 이번 구간 — 어디에서 어디로 가는지. 제약 질의는 이 구간에 걸 규칙을 묻는
    # 것이므로, 그 구간이 그림에 없으면 무엇에 대한 질문인지 알 수 없다.
    if leg_xy is not None and len(leg_xy) >= 2:
        luv = frame.world_to_uv(np.asarray(leg_xy, dtype=float))
        a, b = luv[0], luv[-1]
        d.line([tuple(a), tuple(b)], fill=(255, 255, 255, 230), width=12)
        d.line([tuple(a), tuple(b)], fill=(255, 145, 0, 255), width=8)
        # 화살촉
        v = b - a
        n = np.hypot(*v)
        if n > 1e-6:
            v = v / n
            perp = np.array([-v[1], v[0]])
            tip = b + v * 3
            d.polygon([tuple(tip), tuple(b - v * 13 + perp * 9),
                       tuple(b - v * 13 - perp * 9)], fill=(255, 145, 0, 255),
                      outline=(255, 255, 255, 235))
        # 도착점에 고리. 한 걸음이 0.25 m 라 화살표만으로는 눈에 띄지 않는다.
        d.ellipse([b[0] - 13, b[1] - 13, b[0] + 13, b[1] + 13],
                  outline=(255, 255, 255, 220), width=5)
        d.ellipse([b[0] - 13, b[1] - 13, b[0] + 13, b[1] + 13],
                  outline=(255, 145, 0, 255), width=4)

    # 후보 — 번호는 1부터. 이미지에서 보이는 순서(위→아래, 왼→오른쪽)로 정렬해
    # 붙인다. 격자 인덱스 순서로 붙이면 화면상 번호가 뒤죽박죽이 되어 VLM 이
    # "위쪽 점"과 "3번" 을 연결하지 못한다.
    cand_cells = np.asarray(cand_cells)
    if len(cand_cells):
        uv_all = frame.cell_to_uv(cand_cells)
        order = np.lexsort((uv_all[:, 0], np.round(uv_all[:, 1] / 24)))
        cand_cells = cand_cells[order]

    # 물체 키포인트 — 후보(숫자 원)와 번호 공간도 모양도 달라야 한다.
    # 같은 공간을 쓰면 "가라"(숫자)와 "피하라"(문자)가 섞인다.
    if keypoints:
        kf = _font(max(11, label_pt))
        for k in keypoints:
            if k.get("name") == "GOAL":
                continue                      # 목표는 아래에서 별표로 그린다
            u, v = frame.world_to_uv(np.asarray(k["xy"], dtype=float).reshape(1, 2))[0]
            # 마름모가 작으면 바닥 무늬에 묻힌다. 흰 테두리를 두껍게 깔고
            # 그 위에 진한 파랑을 얹는다.
            R = radius_px + 7
            d.polygon([(u, v - R - 2), (u + R + 2, v), (u, v + R + 2), (u - R - 2, v)],
                      fill=(255, 255, 255, 235))
            d.polygon([(u, v - R), (u + R, v), (u, v + R), (u - R, v)],
                      fill=(21, 62, 130, 255))
            t = str(k["label"])
            tb = d.textbbox((0, 0), t, font=kf)
            _text_with_outline(d, (u - (tb[2] - tb[0]) / 2, v - (tb[3] - tb[1]) / 2 - 1), t, kf)

    # 목표 — 후보와 다른 모양이어야 오인하지 않는다
    if goal_xy is not None:
        u, v = frame.world_to_uv(np.asarray(goal_xy).reshape(1, 2))[0]
        R = 15
        pts = []
        for k in range(10):
            ang = -np.pi / 2 + k * np.pi / 5
            rad = R if k % 2 == 0 else R * 0.42
            pts.append((u + rad * np.cos(ang), v + rad * np.sin(ang)))
        d.polygon(pts, fill=(255, 209, 102, 235), outline=(90, 60, 0))
        _text_with_outline(d, (u + R + 3, v - 9), "GOAL", _font(13),
                           fill=(255, 209, 102))

    # 현재 위치와 방향
    if robot_xy is not None:
        u, v = frame.world_to_uv(np.asarray(robot_xy).reshape(1, 2))[0]
        # 속 빈 고리로 그린다. 채운 원은 바로 옆 후보 번호를 삼켜 읽을 수 없게
        # 만든다 - 후보는 로봇 주변에 모이므로 겹침이 상시로 생긴다.
        d.ellipse([u - 10, v - 10, u + 10, v + 10],
                  outline=(255, 255, 255, 235), width=5)
        d.ellipse([u - 10, v - 10, u + 10, v + 10],
                  outline=(30, 190, 110, 255), width=3)
        if robot_yaw is not None and np.isfinite(robot_yaw):
            tip_w = np.asarray(robot_xy, dtype=float) + 0.45 * np.array(
                [np.cos(robot_yaw), np.sin(robot_yaw)])
            tu, tv = frame.world_to_uv(tip_w.reshape(1, 2))[0]
            _dv = np.array([tu - u, tv - v], dtype=float)
            _n = np.hypot(*_dv)
            if _n > 1e-6:
                # 고리 위에 붙은 작은 쐐기로 그린다. 밖으로 뻗는 화살표는 마치
                # 어느 번호를 가리키는 것처럼 읽힌다 (실측 피드백).
                _dv /= _n
                _pp = np.array([-_dv[1], _dv[0]])
                _c = np.array([u, v])
                _tip = _c + _dv * 16
                d.polygon([tuple(_tip), tuple(_c + _dv * 7 + _pp * 7),
                           tuple(_c + _dv * 7 - _pp * 7)],
                          fill=(20, 170, 95, 255), outline=(255, 255, 255, 240))

        # 라벨은 항상 그린다 - 프롬프트가 "ROBOT 이라 표시된 초록 고리" 라고
        # 지목하는데 글자가 없으면 무엇을 가리키는지 알 수 없다. 대신 후보가
        # 가장 적은 쪽에 놓는다.
        # constraint 질의(후보 없음)에는 글자를 넣지 않는다. 초록 고리가 곧
        # 로봇이고 프롬프트가 "green ring" 으로 지목한다 - 글자는 구간 선과
        # 이음매를 덮기만 한다.
        if route_xy is not None and len(route_xy) >= 2:
            return img, []
        _cu = frame.cell_to_uv(np.asarray(cand_cells)) if len(np.asarray(cand_cells)) else np.empty((0, 2))
        _best, _bestn = (-16, 13), None
        for _off in ((-16, 13), (-16, -26), (-52, -6), (14, -6)):
            _p = np.array([u + _off[0] + 18, v + _off[1] + 6])
            _n = int(np.sum(np.hypot(_cu[:, 0] - _p[0], _cu[:, 1] - _p[1]) < 26)) if len(_cu) else 0
            if _bestn is None or _n < _bestn:
                _best, _bestn = _off, _n
        _text_with_outline(d, (u + _best[0], v + _best[1]), "ROBOT", _font(12),
                           fill=(30, 200, 120))

    # 후보 번호는 마지막에 그린다. ROBOT 배지와 별표가 번호를 덮으면
    # 그 번호는 고를 수 없는 것과 같다 (실측: 6번이 배지에 가렸다).
    # 픽셀 상 너무 가까운 후보는 버린다. 남은 것에만 번호를 붙이므로
    # 번호와 위치는 항상 1:1 이다.
    if len(cand_cells):
        min_gap = 2.8 * radius_px   # 2.2 로는 0.25m 간격에서 붙어 보인다
        kept, kept_uv = [], []
        for rc in cand_cells:
            q = frame.cell_to_uv(np.asarray(rc).reshape(1, 2))[0]
            if any((q[0] - o[0]) ** 2 + (q[1] - o[1]) ** 2 < min_gap ** 2
                   for o in kept_uv):
                continue
            kept.append(rc); kept_uv.append(q)
        cand_cells = np.asarray(kept) if kept else np.empty((0, 2), dtype=int)

    labels = []
    for i, rc in enumerate(cand_cells, start=1):
        u2, v2 = frame.cell_to_uv(np.asarray(rc).reshape(1, 2))[0]
        d.ellipse([u2 - radius_px, v2 - radius_px, u2 + radius_px, v2 + radius_px],
                  fill=(20, 30, 40, 190), outline=(255, 255, 255, 230), width=2)
        t = str(i)
        tb = d.textbbox((0, 0), t, font=font)
        _text_with_outline(d, (u2 - (tb[2] - tb[0]) / 2, v2 - (tb[3] - tb[1]) / 2 - 1), t, font)
        labels.append(tuple(int(x) for x in rc))

    return img, labels


def to_png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
