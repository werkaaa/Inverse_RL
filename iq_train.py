import json
import pathlib
from typing import Literal

import gym
from attrdict import AttrDict
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from iq_learn.SAC import SAC
from utils.memory import MemoryBuffer


def make_environment(args, render_mode=None):
    # For now we run the simplest environment with continuous
    # action space. It should be extended later.
    return gym.make(args.env.name, render_mode=render_mode)


def make_agent(env, args):
    # When we have an agent for discrete action spaces,
    # it can also be created here.
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    return SAC(obs_dim, action_dim, args)


def evaluate(agent, args, epoch, learn_steps, writer):
    render_mode = "human" if args.eval.show_vis else None
    eval_env = make_environment(args, render_mode=render_mode)
    eval_env.reset(seed=args.seed + 1)

    eval_reward = 0
    eval_steps = 0
    for _ in range(args.eval.num_trajs):
        state, _ = eval_env.reset(seed=args.seed + 2000)
        for _ in range(args.eval.max_traj_steps):
            eval_steps += 1

            agent.train(False)
            action = agent.get_action(state, sample=False)
            agent.train(True)
            next_state, reward, done, _, _ = eval_env.step(action)
            eval_reward += reward
            state = next_state

            if done:
                break

    avg_eval_reward = eval_reward / eval_steps
    writer.add_scalar("eval/epoch_mean_reward", avg_eval_reward, global_step=epoch)
    print(f"Epoch {epoch + 1} (learn step {learn_steps + 1}) average evaluation reward: {avg_eval_reward:.2f}")


def main():
    with open('configs/sac.json') as f:
        args = AttrDict(json.load(f))

    # Save logs
    writer = SummaryWriter(log_dir='logs')

    # Make environments and set the seed
    env = make_environment(args)
    env.reset(seed=args.seed)

    # make agent
    agent = make_agent(env, args)

    # Create online memory buffer
    online_memory_replay = MemoryBuffer(args.seed + 2)

    # Create expert memory buffer
    expert_memory_replay = MemoryBuffer(args.seed + 3)
    expert_memory_replay.generate_expert_data(
        env, args.expert, args.seed + 4
    )

    # Train
    total_steps = 0
    learn_steps = 0

    for epoch in tqdm(range(args.train.epochs)):
        state, _ = env.reset(seed=args.seed + 1000)
        episode_reward = 0

        for episode_step in range(args.train.episode_steps):

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

            if learn_steps % args.train.log_interval == 0 and online_memory_replay.length > args.initial_mem:
                for key, loss in losses.items():
                    writer.add_scalar(key, loss, global_step=learn_steps)

            if done:
                break
            state = next_state

        avg_train_reward = episode_reward / args.train.episode_steps
        writer.add_scalar("train/epoch_mean_reward", avg_train_reward, global_step=epoch)
        print(f"Epoch {epoch + 1} average train reward: {avg_train_reward:.2f}")

        evaluate(agent, args, epoch, learn_steps, writer)


if __name__ == '__main__':
    main()

# TODO: Add saving the model.
