from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.utils.env_utils import create_env, run_random_rollouts
import numpy as np

# choose random task
env_name = np.random.choice(list(ALL_KITCHEN_ENVIRONMENTS))

env = create_env(
    env_name=env_name,
    render_onscreen=False,
    seed=0, # set seed=None to run unseeded
)

# run rollouts with random actions and save video
info = run_random_rollouts(
    env, num_rollouts=1, num_steps=100, video_path="test.mp4"
)
print(info)
print("It works!")