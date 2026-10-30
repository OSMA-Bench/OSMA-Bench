"""R4 검증 8: 단계 분해 파싱과 주입, 그리고 VLM 왕복.

시뮬레이터 없이 돈다. VLM 서버가 없으면 파싱만 확인하고 넘어간다.
"""
import base64
import io
import json
import os
import sys
import urllib.request

import numpy as np

_H = os.path.dirname(os.path.abspath(__file__))
_KN = os.path.join(_H, "lib")
sys.path.insert(0, _H)
sys.path.insert(0, os.path.join(_H, "src"))
sys.path.insert(0, os.path.join(_ROOT, "lib"))

from src import stages                                            # noqa: E402
import constraints as K                                           # noqa: E402

Q = chr(34) * 3
ok = True

# --- 1. 손으로 쓴 프로그램이 파싱되는가 ---
GOOD = f"""```python
# STAGES: 2
# 1: get clear of the counter
# 2: come up to the sink

def stage1_subgoal_constraint1(state, keypoints):
    {Q}move away from the cat before turning{Q}
    return standoff_cost(state, keypoints['cat'], 1.2)

def stage1_path_constraint1(traj, keypoints):
    {Q}keep room from the cat the whole way{Q}
    return clearance_cost(traj, keypoints['cat'], 1.0)

def stage2_subgoal_constraint1(state, keypoints):
    {Q}finish at the sink{Q}
    return progress_cost(state, keypoints['sink'], 0.4)
```"""

K.set_keypoints({"cat": np.array([2.13, -1.35]), "sink": np.array([2.68, -0.30])})
K.set_keypoint_clouds(None)
K.set_keypoint_names({})
allowed = {"cat", "sink"}

code = __import__("constraint_code").extract(GOOD)
prog, info = stages.parse(code, allowed)
print(f"1. 파싱: 단계 {sorted(prog)} | 선언 {info['declared']} | 이름 {info['names']}")
print(f"   컴파일된 함수 {info['compiled']} | 거절 {info['rejected']}")
ok &= info["rejected"] is None and set(prog) == {1, 2}
ok &= len(prog[1]["subgoal"]) == 1 and len(prog[1]["path"]) == 1

# --- 2. 주입 ---
start = np.array([4.99, -1.02])
goal = np.array([0.20, -0.30])
prog2, info2 = stages.finalize(prog, goal, start, info)
print(f"2. 주입: {info2['injected']}")
ok &= any(x["why"] == "final_goal" for x in info2["injected"])
ok &= len(prog2[2]["subgoal"]) == 2          # VLM 것 + 주입된 목표

# --- 3. 실제로 평가되는가 ---
st = (1.0, -1.0, 0.0)
vals = [float(f(st, K.get_keypoints())) for f in prog2[1]["subgoal"]]
tr = K.Traj(np.stack([start, goal]), dt=0.2)
pvals = [float(f(tr, K.get_keypoints())) for f in prog2[1]["path"]]
print(f"3. 평가: subgoal {np.round(vals, 3)} | path {np.round(pvals, 3)}")
ok &= all(np.isfinite(vals)) and all(np.isfinite(pvals))

# --- 4. 허용되지 않은 함수는 거절되는가 ---
BAD = (f"def stage1_subgoal_constraint1(state, keypoints):\n"
       f"    {Q}pace 는 이 정책에서 노출하지 않는다{Q}\n"
       f"    return pace_cost(traj, from_pct=0, to_pct=100, speed=0.5)")
_, info_bad = stages.parse(BAD, allowed)
print(f"4. 거절: {str(info_bad['rejected'])[:70]}")
ok &= info_bad["rejected"] is not None

# --- 5. 빈 프로그램도 단계 하나로 살아나는가 ---
prog3, info3 = stages.finalize({}, goal, start, {})
print(f"5. 파싱 실패 시: 단계 {sorted(prog3)} | 주입 {info3['injected']}")
ok &= set(prog3) == {1} and len(prog3[1]["subgoal"]) == 1

# --- 6. VLM 왕복 (서버 있으면) ---
try:
    urllib.request.urlopen("http://localhost:8003/v1/models", timeout=4).read()
except Exception:                                                 # noqa: BLE001
    print("6. VLM 서버 없음 - 왕복 건너뜀")
else:
    from PIL import Image
    # outputs 는 worktree 에 없다 (원본 checkout 의 미커밋 산출물).
    img_path = (os.environ.get("PIVOT_FIXTURES",
        "./policy/keypoint_nav/outputs") + "/"
                "outputs/e6_20/layout0/NavigateKitchenCatBlockingRouteA/hook_topview.png")
    im = Image.open(img_path).convert("RGB")
    b = io.BytesIO()
    im.save(b, "PNG")
    tbl = "\n".join(["A = cat", "B = sink", "C = stove", "D = fridge"])
    prompt = stages.load_prompt(os.path.join(_H, "prompts")).format(objects=tbl)
    body = json.dumps({
        "model": "Qwen/Qwen3-VL-8B-Instruct", "temperature": 0, "max_tokens": 900,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()}}]}],
    }).encode()
    r = urllib.request.urlopen(urllib.request.Request(
        "http://localhost:8003/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}), timeout=180)
    out = json.loads(r.read())["choices"][0]["message"]["content"]
    vcode = __import__("constraint_code").extract(out)
    K.set_keypoints({"cat": np.array([2.13, -1.35]), "sink": np.array([2.68, -0.30]),
                     "stove": np.array([4.3, -0.35]), "fridge": np.array([0.6, -0.3])})
    vprog, vinfo = stages.parse(vcode, {"cat", "sink", "stove", "fridge"})
    vprog, vinfo = stages.finalize(vprog, goal, start, vinfo)
    print(f"6. VLM: 선언 {vinfo['declared']} | 실제 단계 {vinfo['stages']} "
          f"| 함수 {vinfo['compiled']} | 거절 {vinfo['rejected']}")
    print(f"   주입 {vinfo['injected']}")
    if vinfo["rejected"]:
        print("   원문:", (out or "")[:220].replace("\n", " | "))
    else:
        for s in sorted(vprog):
            vv = [round(float(f(st, K.get_keypoints())), 3) for f in vprog[s]["subgoal"]]
            print(f"   stage {s} '{vprog[s]['name']}' subgoal 평가 {vv}")

print("\nR4", "통과" if ok else "실패")
sys.exit(0 if ok else 1)
