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
    action_space_dim = env.action_space.shape[0]
    return SAC(action_space_dim, args)


def main():
    with open('config.json') as f:
        args = json.load(f)

    # Make environments and set the seed
    env = make_environment()
    eval_env = make_environment()
    env.seed(args.seed)
    eval_env.seed(args.seed + 1)

    # make agent
    agent = make_agent(env, args)

    # Create online memory buffer
    online_memory_replay = MemoryBuffer(args.seed + 2)

    # Create expert memory buffer
    expert_memory_replay = MemoryBuffer(args.seed + 3)
    expert_memory_replay.load(pathlib.Path(f'./experts/{args.env.demo}'),
                              num_trajs=args.expert.num_trajs,
                              seed=args.seed + 4)

    # Train
    total_steps = 0
    learn_steps = 0

    for epoch in args.epochs:
        state = env.reset()
        episode_reward = 0

        for episode_step in range(args.episode_steps + args.warmup_steps):

            if total_steps < args.warmup_steps:
                # At the beginning the agent takes random actions.
                action = env.action_space.sample()
            else:
                # We need to exit the train mode for the actor to play an action.
                agent.train(False)
                action = agent.get_action(state, sample=True)
                agent.train(True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            total_steps += 1

            # Only store done true when episode finishes without hitting timelimit.
            done_no_lim = done
            if str(env.__class__.__name__).find('TimeLimit') >= 0:
                done_no_lim = 0

            online_memory_replay.add((state, next_state, action, reward, done_no_lim))

            if online_memory_replay.length > args.initial_mem:
                learn_steps += 1
                # IQ-Learn step.
                losses = agent.iq_update(online_memory_replay, expert_memory_replay)

            if done:
                break
            state = next_state


if __name__ == '__main__':
    main()

# TODO: Write a config file to run the model.
# TODO: Add evaluation during training.
# TODO: Add loging.
# TODO: Add saving the model.
# TODO: Add hp tunning.
