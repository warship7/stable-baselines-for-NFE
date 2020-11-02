from stable_baselines import ACER
from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.callbacks import TensorboardCallback

# Load the trained agent
log_dir = "/home/xw/JJ-Experiments/event/20200806-ACER-MTPT-F121/ACER_halfcheetah"

env = AI2ThorEnv()
env = env = DummyVecEnv([lambda: env])
#model = A2C(MlpPolicy, env, verbose=1)
model = ACER.load(log_dir, env = env)
#model.set_env(env)

# Evaluate the agent
episode_rewards, episode_lengths = evaluate_policy(model, model.get_env(), return_episode_rewards=True)

# Evaluate the agent by transfer learning
#model.learn(total_timesteps=int(6e5), callback=TensorboardCallback())
#model.save(log_dir + "/20200728-A2C-MTPT-F119A2C_halfcheetah_test")
