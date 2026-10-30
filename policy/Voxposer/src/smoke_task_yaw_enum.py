"""모든 (task, layout) 조합의 시작 yaw enumeration.

너무 많아서 (138 × 8 = 1104) 두 strategy:
  - 모드 A: 모든 138 task × L0 (138 instances, ~12분)
  - 모드 B: 샘플 task × 모든 8 layouts

환경 변수 MODE=A or B 로 선택.
출력: CSV-like 표 (task, layout, yaw_deg, start_pos)
"""
import os
import sys
import numpy as np

os.environ.setdefault('MUJOCO_GL', 'egl')

from envs.robocasa_env import VoxPoserRobocasa
from utils.arguments import get_config
from transforms3d.euler import quat2euler


def get_tasks():
    import robocasa
    from robocasa.environments.kitchen.kitchen import REGISTERED_KITCHEN_ENVS
    return sorted(
        name for name in REGISTERED_KITCHEN_ENVS
        if name.startswith("NavigateKitchen")
        and name not in ("NavigateKitchenWithObstacles", "NavigateKitchen")
    )


def measure(task, layout, style=3):
    config = get_config(config_path='configs/robocasa_config.yaml', task_type='navigation')
    task_cfg = config['task']
    task_cfg['layout_ids'] = layout
    task_cfg['style_ids'] = style
    env = VoxPoserRobocasa(task_name=task, task_config=task_cfg)
    env.env.reset()
    obs = env.env._get_observations()
    q = obs['robot0_base_quat']
    yaw = float(quat2euler([q[3], q[0], q[1], q[2]])[2])
    pos = obs['robot0_base_pos'][:2]
    return float(np.degrees(yaw)), float(pos[0]), float(pos[1])


def main():
    mode = os.environ.get('MODE', 'A')
    tasks = get_tasks()
    print(f"# {len(tasks)} tasks, mode={mode}", file=sys.stderr)
    print("task,layout,yaw_deg,start_x,start_y")
    if mode == 'A':
        # 138 tasks × L0
        for i, t in enumerate(tasks):
            try:
                yaw, x, y = measure(t, layout=0)
                print(f"{t},0,{yaw:+.1f},{x:+.3f},{y:+.3f}", flush=True)
            except Exception as e:
                print(f"{t},0,ERR,{type(e).__name__},{str(e)[:30]}", flush=True)
    else:
        # 샘플 task × all layouts
        SAMPLE_TASKS = [t for t in tasks if 'BlockingRoute' in t][:6]
        for t in SAMPLE_TASKS:
            for L in [0, 1, 2, 3, 5, 6, 7, 8]:
                try:
                    yaw, x, y = measure(t, layout=L)
                    print(f"{t},{L},{yaw:+.1f},{x:+.3f},{y:+.3f}", flush=True)
                except Exception as e:
                    print(f"{t},{L},ERR,{type(e).__name__},{str(e)[:30]}", flush=True)


if __name__ == "__main__":
    main()
