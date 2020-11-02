import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import DQNPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import ACER
from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
from stable_baselines.common.callbacks import TensorboardCallback

env = AI2ThorEnv()
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

log_dir = "/home/xw/JJ-Experiments/event/20200807-ACER-MTPT-F121"
model = ACER(MlpPolicy, env, verbose=1, tensorboard_log=log_dir)
#model = ACKTR(MlpPolicy, env, verbose=1, tensorboard_log="/tmp/ddpg_log/")
#model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=int(1e6), callback=TensorboardCallback())
model.save(log_dir + "/ACER_halfcheetah")

