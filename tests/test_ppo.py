import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from stable_baselines import DQN
from stable_baselines import ACKTR
from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
from stable_baselines.common import make_vec_env
from stable_baselines.common.callbacks import TensorboardCallback

def make_env():
    def _init():
        env = AI2ThorEnv()
        return env
    return _init

if __name__ == "__main__":
    #env = SubprocVecEnv([lambda: make_env() for i in range(2)])
    env = AI2ThorEnv()
    #env = make_vec_env(AI2ThorEnv, n_envs=1, vec_env_cls=SubprocVecEnv)
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    # env = DummyVecEnv([lambda: env])
    log_dir = "/home/xw/JJ-Experiments/event/20200807-PPO-MTPT-F121"
    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=log_dir)
    #model = ACKTR(MlpPolicy, env, verbose=1, tensorboard_log="/tmp/ddpg_log/")
    #model = A2C(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=int(1e6))
    model.save(log_dir + "/PPO_halfcheetah")
