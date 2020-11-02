from stable_baselines import A2C
from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.callbacks import TensorboardCallback

# Load the trained agent
log_dir = "/media/warship/文件/JJ/AI/05.Reinforcement Learning/Experiment/20200728-A2C-MTPT-F119"

env = AI2ThorEnv()
env = env = DummyVecEnv([lambda: env])
#model = A2C(MlpPolicy, env, verbose=1)
model = A2C.load(log_dir + "/20200728-A2C-MTPT-F119A2C_halfcheetah", env = env)
#model.set_env(env)

# Evaluate the agent
episode_rewards, episode_lengths = evaluate_policy(model, model.get_env(), n_eval_episodes=600, return_episode_rewards=True)

# Evaluate the agent by transfer learning
#model.learn(total_timesteps=int(6e5), callback=TensorboardCallback())
#model.save(log_dir + "/20200728-A2C-MTPT-F119A2C_halfcheetah_test")
