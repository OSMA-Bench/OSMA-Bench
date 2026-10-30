"""VLM 이 낸 제약 코드를 파싱·검증·실행 가능한 형태로 바꾼다.

원칙: 안전은 코드가 보장하고 판단만 모델에게 남긴다. VLM 이 np.linalg.norm 을
직접 쓰기 시작하면 단위가 뒤섞이고 무엇이 위반인지도 코드마다 달라지므로,
헬퍼(constraints.ALLOWED) 외의 호출을 전부 막는다.

실패는 조용히 넘어가지 않는다. 훅이 값을 못 가져와도 파이썬은 None 을 돌려주고
실행은 계속되는데, 이 방식으로 격자 규약 반전·목표 yaw 미전달·장애물 위치
미전달 버그 세 개를 연달아 냈다. 셋 다 오류 없이 몇 시간짜리 실험을 무의미하게
만들었다. 그래서 여기서는 실패를 전부 사유와 함께 기록한다.
"""
import ast
import re

import constraints as K

# ```python ... ``` 블록. 코드 펜스 없이 바로 def 로 시작하는 답도 받는다.
_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)
_DEF = re.compile(r"^\s*def\s+\w+\s*\(", re.M)

# 함수 이름 규약. leg{N}_constraint{M} 또는 PIVOT 식 stage{N}_*_constraint{M}.
_NAME = re.compile(r"^(?:leg|stage)\d+_(?:subgoal_|path_|pace_)?constraint\d+$")


class CodeRejected(ValueError):
    """검증 실패. 사유를 붙여 재질의한다."""


def extract(text):
    """응답에서 코드 부분만 꺼낸다. 없으면 None."""
    m = _FENCE.search(text or "")
    if m:
        return m.group(1).strip() or None
    m = _DEF.search(text or "")
    return (text[m.start():].strip() or None) if m else None


def validate(code, allowed_kps, allowed_fns=None):
    """AST 로 검사한다. 통과하면 [(이름, 노드)] 를 돌려준다.

    실행 전에 막아야 하는 것들:
      - 헬퍼 외 호출 (VLM 이 직접 계산하면 단위·정규화가 무너진다)
      - import, 대입 외의 문장 (샌드박스를 빠져나갈 수 있다)
      - 없는 키포인트 참조 (실행 시점에 터지면 원인을 찾기 어렵다)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeRejected(f"문법 오류: {e.msg} (line {e.lineno})") from e

    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not fns:
        raise CodeRejected("함수 정의가 없다")
    for n in tree.body:
        if not isinstance(n, ast.FunctionDef):
            raise CodeRejected(f"함수 정의 외의 문장이 있다: {type(n).__name__}")

    # 같은 이름이 두 번 나오면 파이썬은 뒤엣것으로 덮어쓴다 - 앞의 제약이
    # 조용히 사라지고 로그에도 남지 않는다. 이름을 하나만 쓰는 응답이 실제로
    # 나오므로 여기서 잡는다.
    seen = set()
    for fn in fns:
        if fn.name in seen:
            raise CodeRejected(
                f"'{fn.name}' 이 두 번 정의됐다. 함수마다 번호를 다르게 붙일 것 "
                f"(stage1_path_constraint1, stage1_path_constraint2, ...)")
        seen.add(fn.name)

    out = []
    for fn in fns:
        if not _NAME.match(fn.name):
            raise CodeRejected(
                f"함수 이름 '{fn.name}' 이 규약에 맞지 않는다 "
                f"(legN_constraintM 또는 stageN_path_constraintM)")
        if not (ast.get_docstring(fn) or "").strip():
            raise CodeRejected(f"'{fn.name}' 에 docstring 이 없다 - "
                               "왜 그 제약을 뒀는지가 유일한 설명 수단이다")
        for node in ast.walk(fn):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise CodeRejected("import 는 쓸 수 없다")
            if isinstance(node, ast.Attribute):
                raise CodeRejected("속성 접근(예: np.linalg)은 쓸 수 없다 - "
                                   "주어진 함수만 쓸 것")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise CodeRejected("호출 형태가 허용되지 않는다")
                # 프롬프트에 적어 준 것만 받는다. 질의마다 알려주는 함수가
                # 다르므로(동역학 전용 질의는 clearance 를 알려주지 않는다)
                # 호출자가 목록을 좁힐 수 있어야 한다.
                _ok = allowed_fns if allowed_fns else K.ALLOWED
                if node.func.id not in _ok:
                    raise CodeRejected(
                        f"'{node.func.id}' 는 이 질의에서 쓸 수 없다. "
                        f"사용 가능: {sorted(_ok)}")
                _check_kp(node, allowed_kps)
        out.append((fn.name, fn))
    return out


_KP_FNS = ("clearance_cost", "speed_limit_cost", "progress_cost",
           "heading_cost", "standoff_cost")


def _check_kp(call, allowed_kps):
    """kp 인자가 존재하는 라벨인지 본다.

    세 가지 형태를 다 본다:
        kp="A"                 키워드 + 상수
        f(traj, "A", ...)      위치 + 상수
        keypoints['human']     PIVOT 관용구 - VLM 이 실제로 가장 많이 쓴다

    마지막을 빼먹었더니 없는 이름(`keypoints['star']`)이 검증을 통과해 최적화기
    안에서 KeyError 로 죽었다. 그 자리는 try/except 로 감싸여 있어 제약이 조용히
    사라진다 - 거절해서 로그에 남기는 편이 낫다.
    """
    vals = []
    for kw in call.keywords:
        if kw.arg == "kp" and isinstance(kw.value, ast.Constant):
            vals.append(kw.value.value)
    if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant) \
            and call.func.id in _KP_FNS:
        vals.append(call.args[1].value)
    for a in list(call.args) + [kw.value for kw in call.keywords]:
        if isinstance(a, ast.Subscript) and isinstance(a.slice, ast.Constant):
            vals.append(a.slice.value)
    for val in vals:
        if val is not None and str(val) not in allowed_kps:
            raise CodeRejected(
                f"'{val}' 는 없는 키포인트다. 사용 가능: {sorted(allowed_kps)}")


def compile_fns(code, allowed_kps, allowed_fns=None):
    """검증 후 실행 가능한 함수 목록으로 만든다.

    Returns: [(이름, 호출가능, 소스)]
    """
    validate(code, allowed_kps, allowed_fns)
    # 실행 이름공간은 검증에 쓴 목록과 같아야 한다. 검증만 allowed_fns 로 하고
    # 실행은 K.ALLOWED 로 두면, 통과한 코드가 실행에서 NameError 로 죽는다 -
    # 그것도 컴파일이 아니라 첫 호출 때라 최적화기 안에서 조용히 삼켜진다.
    env = dict(allowed_fns if allowed_fns else K.ALLOWED)
    exec(compile(code, "<vlm>", "exec"), env)   # noqa: S102 - AST 로 이미 검증했다
    out = []
    for name, obj in env.items():
        if callable(obj) and _NAME.match(name):
            out.append((name, obj, code))
    return out


def evaluate(fns, traj, keypoints, log=None):
    """제약들을 평가해 위반량 합과 항목별 값을 돌려준다.

    실행 중 예외는 그 제약만 무시하고 기록한다 - 에피소드를 죽이지 않는다.
    """
    total, detail = 0.0, {}
    for name, fn, _src in fns:
        try:
            v = float(fn(traj, keypoints))
        except Exception as e:                      # noqa: BLE001
            detail[name] = None
            if log is not None:
                log.append({"constraint": name, "error": f"{type(e).__name__}: {e}"})
            continue
        detail[name] = v
        total += max(0.0, v) ** 2
    return total, detail


def covered_keys(fns_records):
    """이미 제약이 걸린 (함수, 대상) 조합. 중복 생성을 막는 데 쓴다.

    "이미 유효한 규칙은 다시 쓰지 말라" 고 프롬프트에 적어도 8B 는 같은 제약을
    반복해서 낸다 (실측: leg1 과 leg3 이 동일). 프롬프트에 기대지 말고 코드가
    거른다.
    """
    keys = set()
    for rec in fns_records:
        for m in rec.get("vals", []):
            keys.add((m.get("fn"), str(m.get("kp"))))
    return keys


def is_duplicate(code, covered):
    """이 코드가 이미 걸린 제약만 담고 있으면 True."""
    vals = margins_used(code)
    if not vals:
        return False
    return all((m.get("fn"), str(m.get("kp"))) in covered for m in vals)


def margins_used(code):
    """VLM 이 정한 값을 뽑아낸다. 측정의 핵심이다.

    Returns: [{"fn":..., "kp":..., "margin":...}] 형태의 목록.
    """
    out = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id not in K.ALLOWED:
            continue
        rec = {"fn": node.func.id}
        # 표에 없는 함수를 만나면 KeyError 로 에피소드가 통째로 죽는다 -
        # 실측: pace_cost 를 추가하고 여기를 빼먹어 3 회 재시도가 모두 실패했다.
        # 위치 인자 이름을 모르면 키워드만 읽어도 기록으로는 충분하다.
        names = {"clearance_cost": ["traj", "kp", "margin"],
                 "speed_limit_cost": ["traj", "kp", "v_max", "radius"],
                 "accel_limit_cost": ["traj", "a_max"],
                 "jerk_limit_cost": ["traj", "j_max"],
                 "pace_cost": ["traj", "from_pct", "to_pct", "scale"],
                 "arrive_cost": ["traj", "scale"],
                 "progress_cost": ["state", "kp", "tol"],
                 "heading_cost": ["state", "kp", "tol_rad"],
                 "standoff_cost": ["state", "kp", "min_d"],
                 # PIVOT 단계용. 구간이 단계이므로 speed 에 kp·radius 가 없다.
                 "speed_cost": ["traj", "v_max"],
                 "accel_cost": ["traj", "a_max"],
                 "jerk_cost": ["traj", "j_max"]}.get(node.func.id, ["traj"])
        for i, a in enumerate(node.args):
            if isinstance(a, ast.Constant) and i < len(names):
                rec[names[i]] = a.value
        for kw in node.keywords:
            if isinstance(kw.value, ast.Constant):
                rec[kw.arg] = kw.value.value
        if "kp" not in rec:
            # VLM 은 keypoints['A'] 형태를 더 자연스럽게 쓴다 (PIVOT 관용구).
            for a in node.args:
                if (isinstance(a, ast.Subscript) and isinstance(a.slice, ast.Constant)):
                    rec["kp"] = a.slice.value
                    break
        out.append(rec)
    return out
