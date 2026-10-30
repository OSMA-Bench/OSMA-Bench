import os, sys
os.environ['MUJOCO_GL']='egl'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from utils.arguments import get_config
from envs.robocasa_env import VoxPoserRobocasa
from modules.interfaces import setup_LMP

cfg = get_config('src/configs/robocasa_config.yaml', task_type='manipulation')
tc = dict(cfg['task']); tc['layout_ids']=2; tc['style_ids']=3
env = VoxPoserRobocasa(visualizer=None, task_name='PnPCounterToCab', task_config=tc)
env.load_task()
obs = env.reset()
ee = np.array(obs['robot0_eef_pos'])
print('EE world:', ee.round(3))
print('bounds:', env.workspace_bounds_min.round(2), env.workspace_bounds_max.round(2))
lmps, lmp_env = setup_LMP(env, cfg, debug=False, output_dir='/tmp/xform_dbg')
# object obs via the same path the LMP uses
names = list(getattr(env,'name2ids',{}).keys())
print('names:', names)
qn='kiwi'  # alias 경유 강제
print('query ->', qn)
# mask-diag: obj_ids vs masks 교집합 직접 관찰
oid = env.name2ids.get('obj')
print('obj_ids:', oid)
env.update_latest_obs()
pts_dbg, cols_dbg, msks = [], [], []
for cam in env.camera_names:
    if f"{cam}_mask" not in env.latest_obs: continue
    _mk=env.latest_obs[f"{cam}_mask"][:,:,1][::-1].reshape(-1)
    msks.append(_mk)
    print(cam, 'mask dtype', _mk.dtype, 'max', int(_mk.max()))
import numpy as np
msks=np.concatenate(msks); print('mask-diag: total', len(msks), 'match obj_ids:', int(np.isin(msks, oid).sum()), 'unique sample:', np.unique(msks)[:15])
od = lmp_env.detect(qn)
o = od() if callable(od) else od
pos_voxel = np.array(o['position'], dtype=float)
print('obj position (voxel/pos coords):', pos_voxel.round(2))
w = lmp_env._voxel_to_world(pos_voxel)
print('voxel->world of position:', np.array(w).round(3))
if '_position_world' in o:
    print('obs _position_world:', np.array(o['_position_world']).round(3))
print('dist EE<->voxel2world:', float(np.linalg.norm(np.array(w)-ee)).__round__(3))
# EE voxel per interface
print('EE voxel via get_ee_pos:', np.array(lmp_env.get_ee_pos()).round(2))
