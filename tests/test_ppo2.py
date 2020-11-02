import os

import pytest

from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv


@pytest.mark.parametrize("cliprange", [0.2, lambda x: 0.1 * x])
@pytest.mark.parametrize("cliprange_vf", [None, 0.2, lambda x: 0.3 * x, -1.0])
def test_clipping(cliprange, cliprange_vf):
    """Test the different clipping (policy and vf)"""
    model = PPO2('MlpPolicy', 'CartPole-v1',
                 cliprange=cliprange, cliprange_vf=cliprange_vf).learn(1000)
    model.save('./ppo2_clip.zip')
    env = model.get_env()
    model = PPO2.load('./ppo2_clip.zip', env=env)
    model.learn(1000)

    if os.path.exists('./ppo2_clip.zip'):
        os.remove('./ppo2_clip.zip')

if __name__ == "__main__":
    env = make_vec_env(AI2ThorEnv, n_envs=1)
    model = PPO2('MlpPolicy', env, verbose=1, tensorboard_log="/media/warship/文件/JJ/AI/05.Reinforcement Learning/Experiment/20200630-PPO-CartPloe")
    model.learn(total_timesteps=2500000)
    model.save("ppo2_cartpole")

    del model # remove to demonstrate saving and loading

    model = PPO2.load("ppo2_cartpole")

    # Enjoy trained agent
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()