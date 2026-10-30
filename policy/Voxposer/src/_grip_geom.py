import os, sys
os.environ['MUJOCO_GL']='egl'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from utils.arguments import get_config
from envs.robocasa_env import VoxPoserRobocasa
cfg = get_config('src/configs/robocasa_config.yaml', task_type='manipulation')
tc = dict(cfg['task']); tc['layout_ids']=2; tc['style_ids']=3
env = VoxPoserRobocasa(visualizer=None, task_name='PnPCounterToCab', task_config=tc)
env.load_task(); obs = env.reset()
m = env.env.sim.model; d = env.env.sim.data
ee = np.asarray(obs['robot0_eef_pos']); print('EE site z:', round(float(ee[2]),4))
# gripper finger geoms 최저점
tips=[]
for i in range(m.ngeom):
    n = m.geom_id2name(i) or ''
    if 'gripper0' in n and ('finger' in n or 'tip' in n or 'pad' in n):
        tips.append((n, float(d.geom_xpos[i][2])))
tips.sort(key=lambda t:t[1])
print('finger geoms lowest:', tips[:4])
if tips:
    print('EE-to-lowest-fingertip offset:', round(float(ee[2]-tips[0][1]),4))
# kiwi 기하
try:
    b = m.body_name2id('obj_main'); op = d.body_xpos[b]
    gids=[i for i in range(m.ngeom) if m.geom_bodyid[i]==b]
    zs=[float(d.geom_xpos[i][2]) for i in gids]
    sz=[float(m.geom_size[i][:2].max()) for i in gids]
    print('kiwi center z:', round(float(op[2]),4), 'geom half-size max:', round(max(sz),4) if sz else '-')
except Exception as e: print('kiwi geom err', e)
