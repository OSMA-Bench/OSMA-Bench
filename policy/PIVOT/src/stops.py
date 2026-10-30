"""정차 지점 후보를 **화면 전체에서 한 번에** 뽑아 번호를 붙인다.

앞서 쓰던 PIVOT 은 후보를 로봇 주위 고리에서 뽑아 라운드마다 반경을 좁혔다. 그것은
"다음 한 걸음" 을 정하는 데는 맞지만 PIVOT 의 출력 형식과 어긋난다 - 원본은 한 번의
응답으로 **단계 전체**를 돌려준다(`num_stages`, 단계별 제약, `grasp_keypoints` 배열).

그래서 후보를 화면 전체에 고르게 깔고, VLM 이 한 응답 안에서 단계 수 · 단계별 정차
지점 · 단계별 제약을 모두 말하게 한다. 질의가 에피소드당 한 번으로 줄고, 원본의
"프로그램을 통째로 받는다" 는 성질이 유지된다.

번호는 그림의 원 번호와 같다. 좌표를 말하게 하지 않는 이유는 그대로다 - 숫자를 주면
그림을 보지 않고 계산으로 답할 수 있게 되어 장면을 읽는지 알 수 없다.
"""
import numpy as np

import candidates as CAND


def propose(avoid, frame, robot_xy, goal_xy, max_n=14, spacing_m=0.6,
            inflate_cells=None, min_sep_m=0.55):
    """통행 가능한 칸에서 후보를 뽑아 (cells, world_xy) 로 돌려준다.

    Args:
        max_n: 번호 상한. 너무 많으면 그림에서 번호가 겹쳐 읽히지 않는다.
        spacing_m: 격자 간격.
        min_sep_m: 뽑은 뒤 서로 이보다 가까운 것은 솎아 낸다.

    목표 근처 한 점을 반드시 남긴다 - 마지막 단계가 목표에서 끝나는 것이 정상인데
    그 자리에 후보가 없으면 모델이 엉뚱한 번호를 고르거나 -1 만 쓰게 된다.
    """
    kw = {} if inflate_cells is None else {"inflate_cells": int(inflate_cells)}
    cells = CAND.grid_candidates(avoid, spacing_m=spacing_m, **kw)
    if not len(cells):
        return np.empty((0, 2), dtype=int), np.empty((0, 2))

    robot_cell = np.round(frame.world_to_cell([robot_xy])[0]).astype(int)
    goal_cell = np.round(frame.world_to_cell([goal_xy])[0]).astype(int)

    # 목표에 가장 가까운 후보를 따로 빼 두었다가 마지막에 되돌려 넣는다.
    d_goal = np.linalg.norm(cells.astype(float) - goal_cell[None, :], axis=1)
    near_goal = cells[int(d_goal.argmin())]

    picked = CAND.stratified(cells, max_n=max(max_n * 2, 8), center=robot_cell)

    # 서로 너무 가까운 것을 솎는다. 번호가 겹치면 그림이 읽히지 않는다.
    sep = float(min_sep_m) / CAND.CELL_M
    keep = []
    for i in range(len(picked)):
        if all(np.linalg.norm(picked[i] - picked[j]) > sep for j in keep):
            keep.append(i)
        if len(keep) >= max_n:
            break
    out = picked[np.asarray(keep, dtype=int)] if keep else picked[:max_n]

    if not any(np.array_equal(c, near_goal) for c in out):
        out = np.vstack([out[:max_n - 1], near_goal[None, :]])

    # 로봇에서 가까운 순으로 번호를 매긴다. 번호가 공간적으로 뒤죽박죽이면
    # "1 번 다음 2 번" 같은 잘못된 순서 감각을 준다.
    order = np.argsort(np.linalg.norm(out.astype(float) - robot_cell[None, :], axis=1))
    out = out[order]
    return out.astype(int), frame.cell_to_world(out)


def as_table(world_xy):
    """프롬프트에 넣을 번호 목록. 좌표는 주지 않는다 - 그림에서 읽게 한다."""
    return "  ".join(str(i + 1) for i in range(len(world_xy)))


def parse_stops(text):
    """`stop_points = [3, 7]` 을 뽑는다. 없으면 None.

    PIVOT 원본이 `grasp_keypoints = [...]` 로 단계별 요약을 돌려주는 것과 같은 꼴이다.
    다만 우리 AST 검증기는 함수 정의 외의 문장을 거절하므로, 값을 뽑은 뒤 그 줄을
    **코드에서 지우고** 검증에 넘긴다 (`strip` 참조).
    """
    import re
    m = re.search(r"stop_points\s*=\s*\[([^\]]*)\]", text or "")
    if not m:
        return None
    out = []
    for tok in m.group(1).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            return None
    return out or None


def strip(text):
    """`stop_points = [...]` 줄을 지운 코드. 검증기에 넘길 것."""
    import re
    return re.sub(r"^\s*stop_points\s*=\s*\[[^\]]*\]\s*$", "", text or "",
                  flags=re.M)
