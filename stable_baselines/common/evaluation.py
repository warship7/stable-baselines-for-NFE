import numpy as np
import tensorflow as tf

from stable_baselines.common.vec_env import VecEnv


def evaluate_policy(model, env, n_eval_steps=1e6, deterministic=False,
                    render=False, callback=None, reward_threshold=None,
                    return_episode_rewards=False):
    """
    Runs policy for `n_eval_episodes` episodes and returns average reward.
    This is made to work only with one env.

    :param model: (BaseRLModel) The RL agent you want to evaluate.
    :param env: (gym.Env or VecEnv) The gym environment. In the case of a `VecEnv`
        this must contain only one environment.
    :param n_eval_episodes: (int) Number of episode to evaluate the agent
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :param render: (bool) Whether to render the environment or not
    :param callback: (callable) callback function to do additional checks,
        called after each step.
    :param reward_threshold: (float) Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: (bool) If True, a list of reward per episode
        will be returned instead of the mean.
    :return: (float, float) Mean reward per episode, std of reward per episode
        returns ([float], [int]) when `return_episode_rewards` is True
    """
    global_t = 0
    log_dir = "/home/xw/JJ-Experiments/event/20200806-A2C-MTPT-F121/FineTune"

    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    ep_length = tf.placeholder(tf.int32)
    tf.summary.scalar("episode_length", ep_length)

    ep_reward = tf.placeholder(tf.float32)
    tf.summary.scalar("episode_reward", ep_reward)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir,
                                                sess.graph)


    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, episode_lengths = [], []
    #for _ in range(n_eval_episodes):
    while True:
        if global_t >= n_eval_steps:
            break
        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action, state = model.predict(obs)
            obs, reward, done, _info = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            global_t += 1
            if render:
                env.render()

        summary_str = sess.run(summary_op, feed_dict={
            ep_length: episode_length, ep_reward: episode_reward[0]
        })
        summary_writer.add_summary(summary_str, global_t)

        summary_writer.flush()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, 'Mean reward below threshold: '\
                                         '{:.2f} < {:.2f}'.format(mean_reward, reward_threshold)
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
