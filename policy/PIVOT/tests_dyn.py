"""동역학 on/off 가 실제로 갈리는지. 스위치는 켜지지 않으면 없는 것과 같다."""
import os
import sys

import numpy as np

_H = os.path.dirname(os.path.abspath(__file__))
for _p in (_H, os.path.join(_H, "src"), os.path.join(_H, "lib")):
    sys.path.insert(0, _p)

import constraints as K                                           # noqa: E402
from src import stages as STG                                     # noqa: E402

Q = chr(34) * 3
K.set_keypoints({"cat": np.array([2.0, 1.0]), "GOAL": np.array([4.0, 0.0])})
K.set_keypoint_clouds(None)
K.set_keypoint_names({})

CODE = (f"def stage1_subgoal_constraint1(state, keypoints):\n"
        f"    {Q}전진{Q}\n"
        f"    return progress_cost(state, keypoints['GOAL'], 0.3)\n"
        f"\n"
        f"def stage1_pace_constraint1(traj, keypoints):\n"
        f"    {Q}느리게{Q}\n"
        f"    return speed_cost(traj, 0.2)\n")

ok = True

# --- 꺼진 상태: 동역학 어휘가 막혀야 한다 ---
prog_off, info_off = STG.parse(CODE, {"cat", "GOAL"}, dynamics=False)
pace_off = (prog_off.get(1) or {}).get("pace", [])
print(f"off | 컴파일 {info_off['compiled']} | pace {len(pace_off)}")
ok &= "stage1_pace_constraint1" not in info_off["compiled"] and not pace_off

# --- 켠 상태: 선언으로 잡히고 인자가 읽혀야 한다 ---
prog_on, info_on = STG.parse(CODE, {"cat", "GOAL"}, dynamics=True)
pace_on = (prog_on.get(1) or {}).get("pace", [])
args = [a for d in pace_on for a in d["args"] if a.get("fn") == "speed_cost"]
print(f"on  | 컴파일 {info_on['compiled']} | pace {len(pace_on)} | 인자 {args}")
ok &= bool(pace_on) and bool(args) and args[0].get("v_max") == 0.2

# --- 선언이 solver 비용에 들어가면 안 된다 ---
print(f"path 목록에 섞이지 않음: {len((prog_on.get(1) or {}).get('path', [])) == 0}")
ok &= len((prog_on.get(1) or {}).get("path", [])) == 0

# --- 프롬프트도 갈려야 한다 ---
p_off = STG.load_prompt(os.path.join(_H, "prompts"), dynamics=False)
p_on = STG.load_prompt(os.path.join(_H, "prompts"), dynamics=True)
print(f"프롬프트 off {len(p_off)}자 / on {len(p_on)}자 | speed_cost 언급 "
      f"off={'speed_cost' in p_off} on={'speed_cost' in p_on}")
ok &= ("speed_cost" not in p_off) and ("speed_cost" in p_on) and len(p_on) > len(p_off)

# --- speed_cost 자체가 부호 규약을 지키는가 ---
slow = K.Traj(np.stack([np.linspace(0, 0.5, 20), np.zeros(20)], axis=1), dt=0.1)
fast = K.Traj(np.stack([np.linspace(0, 5.0, 20), np.zeros(20)], axis=1), dt=0.1)
c_slow, c_fast = K.speed_cost(slow, 0.5), K.speed_cost(fast, 0.5)
print(f"speed_cost: 느린 궤적 {c_slow:.3f} (<=0), 빠른 궤적 {c_fast:.3f} (>0)")
ok &= c_slow <= 0 < c_fast

print("\n동역학 스위치", "통과" if ok else "실패")
sys.exit(0 if ok else 1)
