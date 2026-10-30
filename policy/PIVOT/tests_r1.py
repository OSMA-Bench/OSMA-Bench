"""R1 검증: SDF 부호 / 좌표 왕복 / 비용 항 단위 시험."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import sdf, interp
from geometry import TopviewFrame

DUMP = (os.environ.get("PIVOT_FIXTURES",
        "./policy/keypoint_nav/outputs") + "/"
        "e6_20/layout0/NavigateKitchenCatBlockingRouteA/voxposer_dump.npz")
npz = np.load(DUMP, allow_pickle=True)
it = npz["iters"][0]
avoid = np.asarray(it["avoidance_map"], float)
fr = TopviewFrame.from_dump(npz)
ok = True

# --- 1. SDF 부호 ---
si, sm = sdf.build(avoid)
occ = avoid > 0.5
free_ok = (sm[~occ] > 0).all()
occ_ok = (sm[occ] <= 0).all()
off = float(si([[-5.0, -5.0]])[0])
print(f"1. SDF  자유공간 양수 {free_ok} | 장애물 안 음수 {occ_ok} | 맵 밖 {off:+.2f} (음수여야)")
print(f"        범위 {sm.min():+.2f} ~ {sm.max():+.2f} m")
ok &= bool(free_ok and occ_ok and off < 0)

# --- 2. 좌표 왕복을 외부 기준과 대조 ---
# affordance_map 이 dump 에 없으므로 goal_xy_fixture 와 셀 왕복으로 대조한다.
goal = np.asarray(npz["goal_xy_fixture"], float)
cell = fr.world_to_cell([goal])[0]
back = fr.cell_to_world([cell])[0]
err = float(np.linalg.norm(back - goal))
print(f"2. 좌표  goal {np.round(goal,3)} -> cell {np.round(cell,1)} -> {np.round(back,3)} | 오차 {err:.4f} m")
ok &= err < 0.05

# --- 3. 비용 항 ---
pose = np.array([1.0, 2.0, 0.5])
c0 = interp.consistency(pose, pose)
straight = np.stack([np.linspace(0, 3, 7), np.zeros(7), np.zeros(7)], axis=1)
pl, rl = interp.path_length(straight)
t_zero = interp.turn_cost(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
t_half = interp.turn_cost(np.array([0.0, 0.0, 0.0]), np.array([-1.0, 0.0, np.pi]))
w = interp.wrap_angle(3 * np.pi)
print(f"3. 비용  consistency(자기자신) {c0:.6f} (0 이어야)")
print(f"        직선 길이 {pl:.3f} m (3.0 이어야) | 회전 {rl:.3f} (0 이어야)")
print(f"        turn_cost 정면 {t_zero:.4f} (0) | 뒤로 {t_half:.4f} (0.5)")
print(f"        wrap(3pi) {w:+.4f} (±pi)")
ok &= abs(c0) < 1e-9 and abs(pl - 3.0) < 1e-6 and abs(rl) < 1e-9
ok &= abs(t_zero) < 1e-9 and abs(t_half - 0.5) < 1e-6

# 각도 보간이 짧은 쪽으로 도는지
p = np.array([[0.0, 0.0, np.deg2rad(179)], [1.0, 0.0, np.deg2rad(-179)]])
d = interp.interpolate(p, 5)
span = float(np.abs(interp.angle_diff(d[1:, 2], d[:-1, 2])).sum())
print(f"        179deg -> -179deg 보간 총 회전 {np.rad2deg(span):.1f}deg (2 여야, 358 이면 버그)")
ok &= span < np.deg2rad(10)

print("\nR1", "통과" if ok else "실패")
sys.exit(0 if ok else 1)
