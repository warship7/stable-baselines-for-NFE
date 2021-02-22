Stable Baselines for NFE is a set of implementations in AI2THOR environment of reinforcement learning algorithms based on [Stable Baselines](https://github.com/hill-a/stable-baselines)
=======

Example
-------
```python
import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
from stable_baselines.common.callbacks import TensorboardCallback

import os
import sys

env = AI2ThorEnv()
log_dir = "/home/warship/experiments/20210120-A2C-MTPT-F119"
model = A2C(MlpPolicy, env, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=int(1e6), callback=TensorboardCallback())
model.save(log_dir + "/A2C_halfcheetah")
```
