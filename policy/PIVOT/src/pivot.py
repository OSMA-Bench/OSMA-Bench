"""PIVOT 식 반복 시각 질의로 다음 정차 지점을 고른다.

화면 전체에서 한 번에 뽑는 방식(`stops.propose`)과 다른 점은 **후보를 어디서 뽑는가**다.
전체 뷰에서 고르게 뽑으면 목표에서 멀어지는 점이 후보의 절반 가까이 된다 - 실측으로
14 개 중 5 개가 그랬고, 모델이 그중 하나를 골라 경로가 22.6 m(직선 4.78 m 의 4.7 배)가 됐다.
그림에 목표가 별로 표시돼 있는데도 그랬으니 "목표를 모른다" 로는 설명되지 않는다.

여기서는 로봇 주위 고리에서, **목표 쪽으로 치우치게** 뽑는다. 나쁜 선택지가 애초에
목록에 없다. 그리고 이긴 점 주위로 반경을 좁혀 다시 물어, 한 번에 정확할 필요를 없앤다.

대신 질의가 늘어난다 - 단계마다 라운드 수만큼. PIVOT 의 "한 응답에 프로그램 전체" 형태를
포기하는 대가이고, 그래서 스위치로 남긴다.
"""
import os
import re

import numpy as np

_PICK_RE = re.compile(r"\b(?:PICK|CHOICE|ANSWER)\s*[:=]?\s*(\d+)\b", re.I)
_NUM_RE = re.compile(r"\b(\d{1,2})\b")


def load_prompt(prompts_dir, name="pivot.txt"):
    with open(os.path.join(prompts_dir, name), encoding="utf-8") as f:
        return f.read()


def parse_pick(text, n):
    """응답에서 번호를 뽑는다. 범위를 벗어나면 None.

    조용히 1 번으로 떨어뜨리지 않는다 - "모델이 1 번을 골랐다" 와 "파싱 실패" 가
    로그에서 구별되지 않으면, 나중에 결과를 보고 무엇을 고쳐야 할지 알 수 없다.
    """
    m = _PICK_RE.search(text or "")
    if m:
        v = int(m.group(1))
    else:
        nums = _NUM_RE.findall(text or "")
        if not nums:
            return None
        v = int(nums[-1])
    return v if 1 <= v <= n else None


def ring_candidates(sdf_interp, frame, center, goal, radius, n_want,
                    min_clear, rng, spread_rad=1.1, reject=None):
    """center 주위 고리에서 목표 쪽으로 치우치게 뽑는다.

    고르게 뿌리면 절반이 뒤쪽에 떨어져 번호를 낭비한다. 목표 반대편도 조금 남긴다 -
    우회가 필요한 장면에서는 그쪽이 정답이다.

    자유공간 판정은 solver 가 쓰는 SDF 그대로다. 두 곳이 다른 기준을 쓰면
    "고를 수는 있는데 풀 수 없는" 점이 나온다.
    """
    center = np.asarray(center, float)[:2]
    to_goal = np.asarray(goal, float)[:2] - center
    base = float(np.arctan2(to_goal[1], to_goal[0]))
    for _ in range(6):                       # 여유가 없으면 반경을 넓혀 다시
        th = base + rng.normal(0.0, spread_rad, size=n_want * 5)
        rr = radius * rng.uniform(0.45, 1.0, size=n_want * 5)
        cand = center + np.stack([rr * np.cos(th), rr * np.sin(th)], axis=1)
        cells = frame.world_to_cell(cand)
        d = np.asarray(sdf_interp(np.atleast_2d(cells)), float).ravel()
        cand = cand[d > float(min_clear)]
        # 모델이 방금 쓴 제약을 후보에도 적용한다. 없으면 같은 응답 안에서 두 답이
        # 서로 어긋난다 - 실측: "사람에게서 0.7 m" 라고 써 놓고 사람에게서 0.98 m 인
        # 점을 골랐고(요구 실효거리 1.22 m), 두 제약이 가중치 200 으로 맞붙어
        # 경로가 0.72 m 로 지나가 채점에서 98.2% 위반이 됐다.
        if reject is not None and len(cand):
            cand = cand[~np.asarray([bool(reject(c)) for c in cand], bool)]
        if len(cand) >= 3:
            break
        radius *= 1.35
    else:
        return np.empty((0, 2))
    # 서로 너무 가까운 것은 솎는다. 번호가 겹치면 그림이 읽히지 않는다.
    keep = [0]
    for i in range(1, len(cand)):
        if min(np.linalg.norm(cand[keep] - cand[i], axis=1)) > radius * 0.30:
            keep.append(i)
        if len(keep) >= n_want:
            break
    return cand[keep]


class PivotChooser:
    """라운드마다 반경을 좁혀 가며 다음 정차 지점을 고른다."""

    def __init__(self, client, prompts_dir, rounds=3, n_samples=9, seed=0):
        self.client = client
        self.prompt = load_prompt(prompts_dir)
        self.rounds = int(rounds)
        self.n = int(n_samples)
        self.rng = np.random.default_rng(seed)
        self.log = []

    def choose(self, annotate_fn, image, frame, sdf_interp, cur_xy, goal_xy,
               step_m, min_clear=0.20, robot_yaw=None, tag="", reject=None):
        """다음 정차 지점 하나. 실패하면 None - 호출부가 목표 쪽 기본값으로 떨어진다."""
        center = np.asarray(cur_xy, float)[:2]
        radius = float(step_m)
        picked, rounds_log = None, []

        for r in range(self.rounds):
            cand = ring_candidates(sdf_interp, frame, center, goal_xy, radius,
                                   self.n, min_clear, self.rng, reject=reject)
            if len(cand) < 2:
                rounds_log.append({"round": r + 1, "why": "후보 없음"})
                break
            cells = np.round(frame.world_to_cell(cand)).astype(int)
            try:
                img, _ = annotate_fn(image, frame, cells, goal_xy=goal_xy,
                                     robot_xy=center, robot_yaw=robot_yaw,
                                     radius_px=9, label_pt=14)
                text = self.client.ask(
                    self.prompt.format(n=len(cand), round=r + 1,
                                       total=self.rounds), img)
            except Exception as e:           # noqa: BLE001
                rounds_log.append({"round": r + 1, "why": str(e)[:90]})
                break
            k = parse_pick(text, len(cand))
            rounds_log.append({
                "round": r + 1, "n": int(len(cand)), "radius": round(radius, 2),
                "pick": k,
                "cand": [[round(float(a), 2), round(float(b), 2)] for a, b in cand],
                "reply": (text or "").strip()[-60:]})
            if k is None:
                break
            picked = cand[k - 1]
            center = picked
            radius *= 0.5                    # 이긴 점 주위로 좁혀 다시 묻는다

        self.log.append({"tag": tag, "rounds": rounds_log,
                         "picked": None if picked is None
                         else [float(picked[0]), float(picked[1])]})
        return picked
