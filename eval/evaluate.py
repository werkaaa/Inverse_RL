import json
import gym

from attrdict import AttrDict
import stable_baselines3 as sb

import iq_learn

SEED = 0


def load_expert(env_name, model_path):
    env = gym.make(env_name)
    env.reset(seed=SEED)
    return sb.SAC.load(model_path, env)


def load_iq_learn_model(env_name, config_file, model_path):
    with open(config_file) as f:
        args = AttrDict(json.load(f))
    env = gym.make(env_name)
    env.reset(seed=SEED)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = iq_learn.SAC(obs_dim, action_dim, args, env.action_space.low, env.action_space.high)
    model.load(model_path)
    return model


def evaluate_models(model_groups, env_names):
    num_episodes = 10

    for env_idx, env_name in enumerate(env_names):
        for group_idx, models in model_groups.items():
            model = models[env_idx]
            env = gym.make(env_name)
            env.reset(seed=SEED)
            total_steps = 0
            total_reward = 0

            for _ in range(num_episodes):
                obs, _ = env.reset()
                done = False
                episode_end = False
                steps = 0
                episode_reward = 0

                while not (done or episode_end):
                    action = model.predict(obs, deterministic=True)
                    action = action[0] if type(action) == tuple else action
                    obs, reward, done, episode_end, _ = env.step(action)
                    steps += 1
                    episode_reward += reward

                total_steps += steps
                total_reward += episode_reward

            mean_steps = total_steps / num_episodes
            mean_reward = total_reward / num_episodes

            print(f"{group_idx}, {env_name}: Mean Steps: {mean_steps:.2f}, Mean Reward: {mean_reward:.2f}")
            print()

        print()


if __name__ == "__main__":
    env_names = ['HalfCheetah-v4', 'Walker2d-v2', 'Hopper-v2']

    expert_paths = ["experts/cheetah", "experts/walker", "experts/hopper"]
    experts = [load_expert(env, ep) for env, ep in zip(env_names, expert_paths)]

    iq_learn_configs = ["configs/half_cheetah.json", "configs/walker.json", "configs/hopper.json"]
    iq_learn_paths = ["results/iq_HalfCheetah-v4_20230514232943",
                      "results/iq_Walker2d-v2_20230519003426",
                      "results/iq_Hopper-v2_20230519130351"]
    iq_learn = [load_iq_learn_model(env, iqc, iqp) for env, iqc, iqp in
                zip(env_names, iq_learn_configs, iq_learn_paths)]

    # gail = []
    # TODO: Add GAIL evaluation

    model_groups = {"Expert": experts,
                    "IQ_learn": iq_learn}

    evaluate_models(model_groups, env_names)
