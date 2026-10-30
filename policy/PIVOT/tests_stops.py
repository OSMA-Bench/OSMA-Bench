"""전체 뷰 한 번 질의로 단계 · 정차 지점 · 제약을 함께 받는지.

시뮬레이터 없이 저장된 덤프로 후보를 뽑고, 실제 모델에 한 번 물어 응답을 검사한다.
확인할 것: 후보가 자유공간인가 · stop_points 가 파싱되는가 · 그 줄이 검증기를 통과하게
지워지는가 · 번호가 좌표로 되돌려지는가.
"""
import json
import os
import sys

import numpy as np

_H = os.path.dirname(os.path.abspath(__file__))
for _p in (_H, os.path.join(_H, "src"), os.path.join(_H, "lib")):
    sys.path.insert(0, _p)

from PIL import Image                                             # noqa: E402

import constraint_code as CC                                      # noqa: E402
import keypoints as KPMOD                                         # noqa: E402
from annotate import annotate                                     # noqa: E402
from geometry import TopviewFrame                                 # noqa: E402
from vlm import VLMClient                                         # noqa: E402

from src import sdf as SDF                                        # noqa: E402
from src import stages as STG                                     # noqa: E402
from src import stops as STOPS                                    # noqa: E402

EP = sys.argv[1] if len(sys.argv) > 1 else (
    "/out2/layout0/NavigateKitchenCatBlockingRouteA")
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/stops_test"
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8003
os.makedirs(OUT, exist_ok=True)

npz = np.load(os.path.join(EP, "voxposer_dump.npz"), allow_pickle=True)
it = npz["iters"][0]
avoid = np.asarray(it["avoidance_map"], float)
fr = TopviewFrame.from_dump(npz)
log = json.load(open(os.path.join(EP, "pivot_log.json")))
start = np.asarray(log["robot_xy"], float)
goal = np.asarray(log["goal_xy"], float)
kp_entry = next(e for e in log["log"] if e.get("stage") == "keypoints")
table = kp_entry["table"]
kxy = kp_entry.get("xy") or {}

cells, xy = STOPS.propose(avoid, fr, start, goal, max_n=14, inflate_cells=8)
si, _ = SDF.build(avoid)
clear = np.asarray(si(np.atleast_2d(fr.world_to_cell(xy)))).ravel()
print(f"후보 {len(cells)}개 | 여유 최소 {clear.min():.2f} m 최대 {clear.max():.2f} m")
print(f"목표에 가장 가까운 후보까지 "
      f"{float(np.linalg.norm(xy - goal, axis=1).min()):.2f} m")

topview = np.asarray(Image.open(os.path.join(EP, "initial_topview.png")).convert("RGB"))
kps = [{"label": lab, "name": nm, "xy": np.asarray(kxy.get(nm, [0, 0]), float)}
       for lab, nm in table.items() if nm in kxy]
img, _ = annotate(topview, fr, cells, goal_xy=goal, robot_xy=start, robot_yaw=0.0,
                  radius_px=8, label_pt=13, keypoints=kps)
img.save(os.path.join(OUT, "stops_query.png"))

prompt = STG.load_prompt(os.path.join(_H, "prompts")).format(
    objects=KPMOD.as_table(kps, start))
client = VLMClient("Qwen/Qwen3-VL-8B-Instruct",
                   base_url=f"http://localhost:{PORT}/v1",
                   temperature=0.0, max_tokens=900)
text = client.ask(prompt, img)
code = CC.extract(text)
print("\n--- 응답 ---")
print(code)

prog, info = STG.parse(code, {k["label"] for k in kps} | {k["name"] for k in kps})
prog, info = STG.finalize(prog, goal, start, info)
print("\n단계", info.get("stages"), "| 컴파일", info.get("compiled"))
print("stop_points", info.get("stop_points"))
print("주입", info.get("injected"))

ok = bool(info.get("compiled")) and clear.min() > 0
sp = info.get("stop_points")
print(f"\n후보 전부 자유공간: {clear.min() > 0}")
print(f"stop_points 파싱: {sp}")
if sp:
    valid = [n for n in sp if n == -1 or 1 <= n <= len(xy)]
    print(f"  유효 번호 {len(valid)}/{len(sp)}")
    for i, n in enumerate(sp, 1):
        if n != -1 and 1 <= n <= len(xy):
            print(f"  단계 {i} -> 후보 {n} = {np.round(xy[n - 1], 2)}")
    ok &= len(valid) == len(sp)
else:
    print("  (없음 - 모델이 안 썼거나 형식이 다르다)")
print("\nSTOPS", "통과" if ok else "실패")
sys.exit(0 if ok else 1)
