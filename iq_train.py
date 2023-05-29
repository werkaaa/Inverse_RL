import json
import os
import random
import time

import gym
from attrdict import AttrDict
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from iq_learn.SAC import SAC
from utils.memory import MemoryBuffer


def make_environment(args, render_mode=None):
    # For now we run the simplest environment with continuous
    # action space. It should be extended later.
    return gym.make(args.env.name, render_mode=render_mode)#, healthy_z_range=(0.2, 1.5))


def make_agent(env, args):
    # When we have an agent for discrete action spaces,
    # it can also be created here.
    action_dim = env.action_space.shape[0]
    return SAC(obs_dim, action_dim, args, env.action_space.low, env.action_space.high)

def evaluate(agent, args, epoch, learn_steps, writer):
    render_mode = "human" if args.eval.show_vis else None
    eval_env = make_environment(args, render_mode=render_mode)
    eval_env.reset(seed=args.seed + 1)
    rewards = []

    for _ in range(args.eval.num_trajs):
        episode_reward = 0
        state, _ = eval_env.reset()
        episode_end = False
        done = False
        steps = 0
        while not (done or episode_end):
            agent.train(False)
            action = agent.get_action(state, sample=False)
            agent.train(True)
            next_state, reward, done, episode_end, _ = eval_env.step(action)
            episode_reward += reward
            state = next_state
            steps += 1
        rewards.append(episode_reward)
        print(steps, episode_reward)

    avg_eval_reward = np.mean(rewards)
    writer.add_scalar("eval/mean_episode_reward", avg_eval_reward, global_step=learn_steps)
    print(f"Episode {epoch + 1} (learn step {learn_steps}) evaluation reward: {avg_eval_reward:.2f}")
    return avg_eval_reward


def save(agent, args, timestamp, output_dir='./results'):
    name = f'iq_{args.env.name}_{timestamp}'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    agent.save(f'{output_dir}/{name}')


def main():
    with open('configs/humanoid.json') as f:
        args = AttrDict(json.load(f))

    # Set the seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Save logs
    timestamp = time.strftime("%Y%m%d%H%M%S")
    writer = SummaryWriter(log_dir=f'logs/{timestamp}')

    # Make environments and set the seed
    env = make_environment(args)
    env.reset(seed=args.seed)
    env.action_space.seed(seed=args.seed + 100)

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

    # Prepare for saving the model
    best_eval_episode_reward = -np.inf
    epoch = 0
    while epoch < args.train.epochs:
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        episode_end = False

        while not (done or episode_end):
            episode_steps += 1

            if total_steps < args.initial_mem:
                # At the beginning the agent takes random actions.
                action = env.action_space.sample()
            else:
                # We need to exit the train mode for the actor to play an action.
                agent.train(False)
                action = agent.get_action(state, sample=True)
                agent.train(True)
            # Values returned by env.step:
            # done=True if environment terminates (eg. due to task completion, failure etc.)
            # episode_end=True if episode truncates due to a time limit or a reason that is not defined as part of the task MDP.
            next_state, reward, done, episode_end, _ = env.step(action)
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

            # We compare to 1 to avoid multiple empty evaluations during warmup
            if learn_steps % args.eval.eval_interval == 1:
                eval_episode_reward = evaluate(agent, args, epoch, learn_steps, writer)
                if eval_episode_reward > best_eval_episode_reward:
                    # Store best eval returns
                    best_eval_episode_reward = eval_episode_reward
                    # wandb.run.summary["best_returns"] = best_eval_returns
                    save(agent, args, timestamp, output_dir='./results')

            state = next_state

        if episode_steps == args.standard_episode_length:
            epoch += 1

        writer.add_scalar("train/episode_reward", episode_reward, global_step=learn_steps)
        print(f"Episode {epoch + 1} (learn step {learn_steps + 1}) episode reward: {episode_reward:.2f}")


if __name__ == '__main__':
    main()
