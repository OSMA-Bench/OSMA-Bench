"""VLM 에게 단계 분해와 제약을 받아 프로그램으로 만든다.

PIVOT 은 과제 문장에서 단계를 뽑는다 ("잡는다" -> 1, "넣는다" -> 2). 네비게이션에는 그런
자연스러운 분해가 없지만 **"무엇을 지나가는가" 가 곧 단계**다. 그래서 VLM 이 정하게 하되
1~4 로 제한한다 - 더 잘게 나누면 단계당 이동이 짧아 최적화 여지가 없고, solver 호출만 는다.

자동 주입이 두 곳 있다. 편법이 아니라 이 구조에서 필요한 최소한이다:
    - 마지막 단계에 progress_cost(GOAL). 없으면 아무것도 로봇을 목표로 당기지 않는다 -
      일관성 1.0 과 경로길이 4.0 만 남아 subgoal 이 현재 자리에 주저앉는다.
      PIVOT 에서는 과제 문장("컵을 홀더에 넣어라")이 그 역할을 한다.
    - 중간 단계에 subgoal 제약이 하나도 없으면 약한 progress_cost.

**주입은 전부 로그에 남긴다.** 사후에 "VLM 이 쓴 것" 과 "우리가 채운 것" 을 가르지 못하면,
결과가 모델의 판단인지 우리 발판인지 알 수 없다.
"""
import os
import re

import numpy as np

import constraint_code as CC
import constraints as K

from . import stops as STOPS

_STAGES_RE = re.compile(r"#\s*STAGES:\s*(\d+)", re.I)
# 모델은 "# Phase 1: ..." / "# Stage 1: ..." 로도 쓴다. 접두어를 받지 않으면 단계
# 이름이 통째로 비고, 로그만 보고는 VLM 이 무엇을 하려 했는지 알 수 없다.
_NAME_RE = re.compile(r"#\s*(?:phase|stage)?\s*(\d+)\s*:\s*(.+)", re.I)
_FN_RE = re.compile(r"stage(\d+)_(subgoal|path|pace)_constraint(\d+)")

MAX_STAGES = 4


def _split_defs(code):
    """코드를 def 단위로 자른다. 하나가 거절돼도 나머지를 살리기 위해서다."""
    out, cur = [], []
    for line in (code or "").splitlines():
        if line.startswith("def ") and cur:
            out.append("\n".join(cur))
            cur = []
        if line.startswith("def ") or cur:
            cur.append(line)
    if cur:
        out.append("\n".join(cur))
    return out


def parse(code, allowed_kps, dynamics=False):
    """코드 -> {stage: {"subgoal": [...], "path": [...], "name": str}}.

    검증은 keypoint_nav 의 AST 검사기를 그대로 쓴다 - 두 번째 검증기를 만들면 두 정책이
    서로 다른 기준으로 걸러지고, 그러면 "무엇이 거절됐는가" 를 비교할 수 없다.
    """
    info = {"declared": None, "names": {}, "rejected": None, "compiled": []}
    # PIVOT 원본이 `grasp_keypoints = [...]` 로 단계별 요약을 돌려주는 것과 같은 꼴이다.
    # 우리 AST 검증기는 함수 정의 외의 문장을 거절하므로, 값을 먼저 뽑고 그 줄을
    # 코드에서 지운 뒤 검증에 넘긴다.
    info["stop_points"] = STOPS.parse_stops(code)
    code = STOPS.strip(code)
    m = _STAGES_RE.search(code or "")
    if m:
        info["declared"] = int(m.group(1))
    for mm in _NAME_RE.finditer(code or ""):
        info["names"][int(mm.group(1))] = mm.group(2).strip()

    allowed_fns = dict(K.PIVOT_SUBGOAL)
    allowed_fns.update(K.PIVOT_PATH)
    # 동역학 어휘는 켰을 때만 알려준다. 끄면 프롬프트에도 안 나가고 여기서도
    # 거절되므로, 두 설정의 차이가 v/a/J 하나로 깨끗하게 남는다.
    if dynamics:
        allowed_fns.update(K.PIVOT_PACE)
    try:
        fns = CC.compile_fns(code, allowed_kps, allowed_fns=allowed_fns)
    except CC.CodeRejected as e:
        # 전부 버리기 전에 함수 하나씩 다시 시도한다. 다섯 개 중 하나가 없는
        # 이름을 가리켰다고 나머지 넷까지 잃으면, VLM 의 단계 분해가 통째로
        # 사라지고 우리 발판만 남는다 - 그러면 무엇을 측정했는지 알 수 없다.
        info["rejected"] = str(e)[:200]
        fns, dropped = [], []
        for block in _split_defs(code):
            try:
                fns.extend(CC.compile_fns(block, allowed_kps, allowed_fns=allowed_fns))
            except CC.CodeRejected as e2:
                dropped.append({"src": block.splitlines()[0][:80], "why": str(e2)[:120]})
        info["dropped"] = dropped
        if not fns:
            return {}, info

    # VLM 이 **무엇에 주목했는가**. 원본 PIVOT 은 DINOv2 로 후보 키포인트를 제안하고
    # VLM 은 그중에서 고르지만, 우리는 환경의 물체 목록을 그대로 준다. 그러면 "주목
    # 대상" 은 목록이 정하고 VLM 은 그중 일부만 실제로 제약에 쓴다 - 무엇을 쓰고
    # 무엇을 지나쳤는지는 이 기록이 없으면 알 수 없다.
    used = sorted({m for _n, _f, src in fns
                   for m in re.findall(r"keypoints\[['\"]([^'\"]+)['\"]\]", src or "")})
    info["keypoints_used"] = used

    program = {}
    for name, fn, _src in fns:
        g = _FN_RE.match(name)
        if not g:
            continue
        stage, kind = int(g.group(1)), g.group(2)
        if stage < 1 or stage > MAX_STAGES:
            continue
        program.setdefault(stage, {"subgoal": [], "path": [], "pace": [], "name": ""})
        if kind == "pace":
            # 동역학 제약은 solver 비용에 넣지 않는다 - path solver 의 제어점은 시간이
            # 아니라 거리 간격이라 그 위의 v/a/J 는 아무것도 재지 않는다. 대신 **선언**
            # 으로 읽어 두었다가 단계 호 구간을 속도 곡선으로 바꿀 때 쓴다.
            program[stage]["pace"].append(
                {"fn_name": name, "args": CC.margins_used(_src or "")})
        else:
            program[stage][kind].append(fn)
        info["compiled"].append(name)
    for s in program:
        program[s]["name"] = info["names"].get(s, "")
    return program, info


def finalize(program, goal_xy, start_xy, info=None, targets=None):
    """빈 곳을 채우고 단계 번호를 1..n 으로 다시 매긴다.

    VLM 이 stage 1 과 3 만 쓰는 경우가 있어(2 를 건너뜀) 연속 번호로 눌러 준다 -
    루프가 stage+1 로 나아가므로 구멍이 있으면 거기서 멈춘다.
    """
    info = info if info is not None else {}
    injected = []

    if not program:
        program = {1: {"subgoal": [], "path": [], "pace": [],
                       "name": "(파싱 실패 - 단일 단계)"}}

    ordered = sorted(program)[:MAX_STAGES]
    out = {}
    for i, old in enumerate(ordered, start=1):
        out[i] = program[old]
    n = len(out)

    goal = np.asarray(goal_xy, dtype=float)[:2]
    total = float(np.linalg.norm(goal - np.asarray(start_xy, dtype=float)[:2]))

    for s in range(1, n + 1):
        is_last = (s == n)
        if is_last:
            # 마지막은 반드시 목표에 닿아야 한다. VLM 이 뭘 썼든 추가한다.
            out[s]["subgoal"].append(
                lambda st, kps, _g=goal: K.progress_cost(st, _g, 0.30))
            injected.append({"stage": s, "why": "final_goal", "tol": 0.30})
        else:
            # 중간 단계에는 **항상** 약한 progress 를 넣는다. "비어 있을 때만" 으로는
            # 부족했다 - VLM 이 쓴 standoff_cost 가 출발 시점에 이미 만족되면 비용이
            # 0 이라 전진을 만드는 힘이 없고, subgoal 이 현재 자리에 주저앉는다.
            # (실측: HumanBlockingRouteB 에서 앞 5개 waypoint 가 로봇 위치와 동일,
            #  로봇이 0.9 m 만 가고 멈춰 1.44 m 미달로 실패.)
            # 만족된 제약과 "비어 있음" 은 solver 입장에서 구별되지 않는다.
            # PIVOT 이 이 단계의 끝점을 골랐으면 목표 대신 그 점으로 당긴다.
            # 구조는 그대로다 - 끝점이 "목표 주위 반지름 tol 고리" 에서
            # "VLM 이 찍은 한 점" 으로 바뀔 뿐이다.
            tgt = (targets or {}).get(s)
            if tgt is not None:
                pt = np.asarray(tgt, dtype=float)[:2]
                out[s]["subgoal"].append(
                    lambda st, kps, _g=pt: K.progress_cost(st, _g, 0.25))
                injected.append({"stage": s, "why": "pivot_target", "tol": 0.25,
                                 "xy": [round(float(pt[0]), 3),
                                        round(float(pt[1]), 3)]})
                continue
            tol = max(0.5, total * (1.0 - s / (n + 1.0)))
            out[s]["subgoal"].append(
                lambda st, kps, _g=goal, _t=tol: K.progress_cost(st, _g, _t))
            injected.append({"stage": s, "why": "weak_progress", "tol": round(tol, 2)})

    info["injected"] = injected
    info["stages"] = n
    return out, info


def load_prompt(prompts_dir, name="stage_decomposition.txt", dynamics=False):
    """프롬프트를 읽는다. dynamics 를 끄면 동역학 문단이 아예 들어가지 않는다.

    두 설정의 차이가 v/a/J 어휘 하나로만 남아야 ablation 이 성립한다 - 문단을 남긴 채
    함수만 막으면 모델이 쓸 수 없는 것을 계속 시도하고, 그 거절이 결과에 섞인다.
    """
    with open(os.path.join(prompts_dir, name), encoding="utf-8") as f:
        text = f.read()
    block, pace_slot = "", ""
    if dynamics:
        with open(os.path.join(prompts_dir, "dynamics_block.txt"), encoding="utf-8") as f:
            block = f.read().rstrip() + "\n\n"
        # 답안 틀에도 자리를 낸다. 설명 문단만 위에 붙이면 소용이 없다 - 모델이
        # 마지막으로 읽는 것은 "Use this exact shape" 아래의 틀이고, 그 틀에 pace 줄이
        # 없으면 없는 것이 정답이 된다. 실측: v/a/J 를 켠 36 개 에피소드에서 사용 0 건,
        # 거부도 0 건이었다. 검증기가 막은 것이 아니라 아무도 시도하지 않았다.
        pace_slot = ("""
def stage1_pace_constraint1(traj, keypoints):
    \"\"\"<why this phase is slow; omit this function when it is not>\"\"\"
    return <one call from the pace list>
""")
    return text.replace("{dynamics}", block).replace("{pace_slot}", pace_slot)
