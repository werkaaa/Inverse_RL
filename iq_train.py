from attrdict import AttrDict
import json
import pathlib

import gym

from iq_learn.SAC import SAC
from utils.memory import MemoryBuffer


def make_environment(args):
    # For now we run the simplest environment with continuous
    # action space. It should be extended later.
    return gym.make(args.env.name)


def make_agent(env, args):
    # When we have an agent for discrete action spaces,
    # it can also be created here.
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    return SAC(obs_dim, action_dim, args)


def main():
    with open('configs/sac.json') as f:
        args = AttrDict(json.load(f))

    # Make environments and set the seed
    env = make_environment(args)
    eval_env = make_environment(args)
    env.reset(seed=args.seed)
    eval_env.reset(seed=args.seed + 1)

    # make agent
    agent = make_agent(env, args)

    # Create online memory buffer
    online_memory_replay = MemoryBuffer(args.seed + 2)

    # Create expert memory buffer
    expert_memory_replay = MemoryBuffer(args.seed + 3)
    expert_memory_replay.generate_expert_data(
        env, args.expert_agent_dir, args.num_trajectories, args.seed + 4
    )

    # Train
    total_steps = 0
    learn_steps = 0
    
    for epoch in args.epochs:
        state, _ = env.reset(args.seed + 1e6)
        episode_reward = 0

        for episode_step in range(args.train.episode_steps + args.train.warmup_steps):

            if total_steps < args.train.warmup_steps:
                # At the beginning the agent takes random actions.
                action = env.action_space.sample()
            else:
                # We need to exit the train mode for the actor to play an action.
                agent.train(False)
                action = agent.get_action(state, sample=True)
                agent.train(True)
            # Values returned by env.step:
            # (2)=True if environment terminates (eg. due to task completion, failure etc.)
            # (3)=True if episode truncates due to a time limit or a reason that is not defined as part of the task MDP.
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            total_steps += 1

            online_memory_replay.add((state, next_state, action, reward, done))

            if online_memory_replay.length > args.initial_mem:
                learn_steps += 1
                # IQ-Learn step.
                losses = agent.iq_update(online_memory_replay, expert_memory_replay, learn_steps)

            if done:
                break
            state = next_state


if __name__ == '__main__':
    main()

# TODO: Integrate with our own expert data generation.
# TODO: Add evaluation during training.
# TODO: Add loging.
# TODO: Add saving the model.
# TODO: Add hp tuning.
