"""부분 수용 시험: 함수 하나가 틀려도 나머지가 살아남는가.

VLM 이 다섯 함수 중 하나에서 없는 이름을 가리키면, 전부 버릴 때는 단계 분해가 통째로
사라지고 우리 발판(주입된 progress_cost)만 남는다. 그러면 무엇을 측정했는지 알 수 없다.
"""
import os
import sys

import numpy as np

_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _H)
sys.path.insert(0, os.path.join(_H, "src"))
sys.path.insert(0, os.path.join(_H, "lib"))

from src import stages                                       # noqa: E402
import constraints as K                                      # noqa: E402

Q = chr(34) * 3
K.set_keypoints({"cat": np.array([2.0, 1.0]), "sink": np.array([3.0, 0.0])})
K.set_keypoint_clouds(None)
K.set_keypoint_names({})

MIX = (f"def stage1_subgoal_constraint1(state, keypoints):\n"
       f"    {Q}좋은 것{Q}\n"
       f"    return standoff_cost(state, keypoints['cat'], 1.2)\n"
       f"\n"
       f"def stage1_path_constraint1(traj, keypoints):\n"
       f"    {Q}없는 이름{Q}\n"
       f"    return clearance_cost(traj, keypoints['unicorn'], 1.0)\n"
       f"\n"
       f"def stage2_subgoal_constraint1(state, keypoints):\n"
       f"    {Q}좋은 것{Q}\n"
       f"    return progress_cost(state, keypoints['sink'], 0.4)\n")

prog, info = stages.parse(MIX, {"cat", "sink"})
print("컴파일:", info["compiled"])
print("떨어뜨림:", info.get("dropped"))
print("단계:", sorted(prog))

ok = (set(info["compiled"]) == {"stage1_subgoal_constraint1", "stage2_subgoal_constraint1"}
      and len(info.get("dropped", [])) == 1
      and sorted(prog) == [1, 2])
print("\n부분 수용", "통과 - 3개 중 1개만 떨어지고 2개 살아남음" if ok else "실패")
sys.exit(0 if ok else 1)
