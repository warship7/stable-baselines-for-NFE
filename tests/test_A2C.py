import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
from stable_baselines.common.callbacks import TensorboardCallback

import os
import sys
# get current working directory -- Better to set the PYTHONPATH env variable
#current_working_directory = "/home/warship/projects/remote/stable_baselines"
#sys.path.append(current_working_directory)

env = AI2ThorEnv()
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

#log_dir = "/home/warship/experiments/20200720-A2C-MTPT-F119"
log_dir = "/home/xw/JJ-Experiments/event/20200807-A2C-MTPT-F121"
model = A2C(MlpPolicy, env, verbose=1, tensorboard_log=log_dir)
#model = ACKTR(MlpPolicy, env, verbose=1, tensorboard_log="/tmp/ddpg_log/")
#model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=int(1e6), callback=TensorboardCallback())
model.save(log_dir + "/A2C_halfcheetah")
