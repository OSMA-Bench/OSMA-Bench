import os, sys
os.environ['MUJOCO_GL']='egl'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from utils.arguments import get_config
from envs.robocasa_env import VoxPoserRobocasa

cfg = get_config('src/configs/robocasa_config.yaml', task_type='manipulation')
tc = dict(cfg['task']); tc['layout_ids']=2; tc['style_ids']=3
env = VoxPoserRobocasa(visualizer=None, task_name='PnPCounterToCab', task_config=tc)
env.load_task() if hasattr(env,'load_task') else None
obs = env.reset()
print('bounds_min', env.workspace_bounds_min.round(2), 'bounds_max', env.workspace_bounds_max.round(2))
ee = np.array(obs['robot0_eef_pos']); print('EE', ee.round(3))
# true kiwi pos from sim
kid=[i for i in range(env.env.sim.model.nbody) if 'kiwi' in (env.env.sim.model.body_id2name(i) or '').lower()]
true_kiwi = np.array(env.env.sim.data.body_xpos[kid[0]]) if kid else None
print('true kiwi body:', env.env.sim.model.body_id2name(kid[0]) if kid else None, true_kiwi.round(3) if kid else '-')
# detected centroid
names = list(getattr(env,'name2ids',{}).keys()) or env.get_object_names() if hasattr(env,'get_object_names') else []
print('registered names:', names[:20])
qn = next((n for n in names if 'kiwi' in n.lower()), None) or next((n for n in names if n.lower() in ('obj','object')), None) or (names[0] if names else 'obj')
print('query name ->', qn)
(_, _), (kpc, _) = env.get_3d_obs_by_name(qn)
det = kpc.mean(axis=0) if kpc is not None and len(kpc)>0 else None
print('detected kiwi centroid:', det.round(3) if det is not None else 'EMPTY', '| n_pts:', 0 if kpc is None else len(kpc))
pts, cols = env.get_scene_3d_obs(ignore_robot=False, ignore_grasped_obj=False)
print('scene pts:', len(pts))
# render
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
idx = np.random.choice(len(pts), min(20000,len(pts)), replace=False)
fig = plt.figure(figsize=(14,6))
for k,(el,az) in enumerate([(35,-60),(88,-90)]):
    ax = fig.add_subplot(1,2,k+1, projection='3d')
    ax.scatter(pts[idx,0],pts[idx,1],pts[idx,2], c=cols[idx]/255.0, s=1, alpha=0.5)
    if kpc is not None and len(kpc)>0:
        ax.scatter(kpc[:,0],kpc[:,1],kpc[:,2], c='lime', s=8, label='detected kiwi pts')
    if det is not None: ax.scatter(*det, c='green', s=220, marker='X', label='detected centroid')
    if true_kiwi is not None: ax.scatter(*true_kiwi, c='red', s=220, marker='*', label='TRUE kiwi')
    ax.scatter(*ee, c='blue', s=200, marker='o', label='EE')
    ax.set_title(f'view el={el}'); ax.legend(loc='upper left', fontsize=8)
    ax.view_init(elev=el, azim=az)
plt.tight_layout(); plt.savefig('/workspace/tmp/pointcloud_debug.png', dpi=110)
print('saved /workspace/tmp/pointcloud_debug.png')
