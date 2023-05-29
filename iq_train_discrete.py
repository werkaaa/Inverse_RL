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

from utils.memory import MemoryBuffer


def make_environment(args, render_mode=None):
    # For now we run the simplest environment with continuous
    # action space. It should be extended later.
    return gym.make(args.env.name, render_mode=render_mode)#, healthy_z_range=(0.2, 1.5))


def make_agent(env, args):
    # When we have an agent for discrete action spaces,
    # it can also be created here.
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.discrete.Discrete):
        print('--> Using Soft-Q agent')
        action_dim = env.action_space.n
        agent = SoftQ(obs_dim, action_dim, args)

    return agent




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
    with open('configs/cartpole.json') as f:
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
        env, args.expert, args.seed + 4,"DQN"
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


import numpy as np
import torch
from torch.optim import Adam



def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def iq_loss(agent, current_Q, current_v, next_v, obs, done, is_expert):
    y = (1 - done) * agent.gamma * next_v
    reward = current_Q - y
    reward_expert = (current_Q - y)[is_expert]

    #  calculate 1st term for IQ loss
    #  -E_(ρ_expert)[Q(s, a) - γV(s')]
    loss = -reward_expert.mean()

    # Calculate 2nd term of the loss (use expert and policy states)
    if agent.loss == 'v0':
        # (1-γ)E_(ρ0)[V(s0)]
        v0 = agent.getV(obs[is_expert.squeeze(1), ...]).mean()
        v0_loss = (1 - agent.gamma) * v0
        loss += v0_loss
    elif agent.loss == 'value':
        # E_(ρ)[Q(s,a) - γV(s')]
        value_loss = (current_v - y).mean()
        loss += value_loss
    else:
        raise NotImplementedError(f'Loss {agent.loss} not implemented!')

    # Use χ2 divergence (adds an extra term to the loss)
    chi2_loss = 1 / (4 * agent.alpha_ksi) * (reward ** 2).mean()
    loss += chi2_loss

    return loss


def get_concat_samples(policy_batch, expert_batch):
    online_batch_state, online_batch_next_state, online_batch_action, online_batch_reward, online_batch_done = policy_batch

    expert_batch_state, expert_batch_next_state, expert_batch_action, expert_batch_reward, expert_batch_done = expert_batch

    batch_state = torch.cat([online_batch_state, expert_batch_state], dim=0)
    batch_next_state = torch.cat(
        [online_batch_next_state, expert_batch_next_state], dim=0)
    batch_action = torch.cat([online_batch_action, expert_batch_action], dim=0)
    batch_reward = torch.cat([online_batch_reward, expert_batch_reward], dim=0)
    batch_done = torch.cat([online_batch_done, expert_batch_done], dim=0)
    is_expert = torch.cat([torch.zeros_like(online_batch_reward, dtype=torch.bool),
                           torch.ones_like(expert_batch_reward, dtype=torch.bool)], dim=0)

    return batch_state, batch_next_state, batch_action, batch_reward, batch_done, is_expert


import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class SoftQ(object):
    def __init__(self, obs_dim, action_dim, args):
        self.gamma = args.gamma
        if args.train.cuda:
            self.device = torch.device("cuda")
        elif args.train.mps:
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.args = args
        self.batch_size = args.train.batch_size
        self.actor = None
        self.soft_update = args.soft_update
        if self.soft_update:
            self.critic_tau = args.critic_tau
        self.target_update_frequency = args.target_update_frequency
        if args.alpha is None:
            self.log_alpha = torch.tensor(np.log(1e-2)).to(device=self.device)
            self.log_alpha.requires_grad = True
        else:
            self.log_alpha = np.log(args.alpha)
        
        
        self.critic= SimpleQNetwork(
            obs_dim,
            action_dim,
            args.critic).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.critic.critic_lr)
        self.target_net = SimpleQNetwork(
            obs_dim,
            action_dim,
            args.critic).to(device=self.device)
        for target_param, param in zip(self.target_net.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        self.train()
        self.target_net.train()

    

    def iq_update(self, policy_buffer, expert_buffer, step):
        policy_batch = policy_buffer.get_batch(self.batch_size)
        expert_batch = expert_buffer.get_batch(self.batch_size)
        #nostra iq_update_critic
        batch = (
            torch.cat(
                [policy_data, expert_data], dim=0) for policy_data, expert_data in zip(policy_batch, expert_batch)
        )
        is_expert = torch.cat([torch.zeros_like(policy_batch[3], dtype=torch.bool),
                               torch.ones_like(expert_batch[3], dtype=torch.bool)], dim=0)
        obs, next_obs, action, _, done = batch
        #implement iq_loss qui
        with torch.no_grad():
            q = self.target_net(next_obs)
            #[V ∗(s0)], 
            next_v = self.alpha * \
                torch.logsumexp(q/self.alpha, dim=1, keepdim=True)
        #phi=1, Es primo=s primo. s' e' un'indice di somma (somma su tuti i possiibli stati da cui puo essere vneuto un s)
        # y e' la soluzione dell'equazione di bellman che e' lo stato fiale e dev'essere uguale a q asterisco che risolve l'equazione
        #cerca iterativamente qualcosa che miimizzi l'errore
            y = (1 - done) * self.gamma * next_v
        #Q(s, a)
        current_Q = self.critic(obs, action)
        #γV(s')
        reward = (current_Q - y)
        reward_expert = (current_Q - y)            
            #  calculate 1st term for IQ loss
            #  E_(ρ_expert)[Q(s, a) - γV(s')]
        loss = -reward_expert.mean()
            # Calculate 2nd term of the loss (use expert and policy states)
            # (1-γ)E_(ρ0)[V(s0)]
        value_loss = F.mse_loss(self.critic(obs, action), y)
        #value_loss=
        
        loss += value_loss
        # Use χ2 e' psi divergence (adds an extra term to the loss)
        chi2_loss = 1 / (4 * self.alpha) * (reward ** 2).mean()
        loss += chi2_loss

        # PRECEDENTE
        # critic_loss = F.mse_loss(self.critic(obs, action), y)
        # logger.log('train_critic/loss', critic_loss, step)

        #iq_update_critic
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        losses = {
            'loss/iq_critic_loss': loss.item()}

        #nostra iq_update
        if step % self.target_update_frequency == 0:
            if self.soft_update:
                soft_update(self.target_net, self.critic, self.critic_tau)
            else:
                hard_update(self.target_net, self.critic)
        return losses





    def train(self, training=True):
        self.training = training
        self.critic.train(training)

    @property
    def alpha(self):
        if not torch.is_tensor(self.log_alpha):
            return np.exp(self.log_alpha)
        return self.log_alpha.exp().detach()

    @property
    def critic_net(self):
        return self.critic

    @property
    def critic_target_net(self):
        return self.target_net

    # Save model parameters
    def save(self, path, suffix=""):
        critic_path = f"{path}{suffix}"
        # print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load(self, path, suffix=""):
        critic_path = f'{path}_critic'
        print('Loading models from {}'.format(critic_path))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))


    



















    def predict(self, state, deterministic=True):
        """Makes the API compatible with stable baselines for evaluation."""
        return self.get_action(state, sample=deterministic)

    def get_action(self, state, sample=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.critic(state)
            dist = F.softmax(q/self.alpha, dim=1)
            dist = Categorical(dist)
            action = dist.sample() 
        return action.detach().cpu().numpy()[0]





import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.autograd import Variable, grad


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, args, device='cpu'):
        super(SoftQNetwork, self).__init__()
        self.args = args
        self.device = device
        self.tanh = nn.Tanh()

    def _forward(self, x, *args):
        return NotImplementedError

    def forward(self, x, both=False):
        out = self._forward(x)

        return out

    def jacobian(self, outputs, inputs):
        """Computes the jacobian of outputs with respect to inputs

        :param outputs: tensor for the output of some function
        :param inputs: tensor for the input of some function (probably a vector)
        :returns: a tensor containing the jacobian of outputs with respect to inputs
        """
        batch_size, output_dim = outputs.shape
        jacobian = []
        for i in range(output_dim):
            v = torch.zeros_like(outputs)
            v[:, i] = 1.
            dy_i_dx = grad(outputs,
                           inputs,
                           grad_outputs=v,
                           retain_graph=True,
                           create_graph=True)[0]  # shape [B, N]
            jacobian.append(dy_i_dx)

        jacobian = torch.stack(jacobian, dim=-1).requires_grad_()
        return jacobian


class SimpleQNetwork(SoftQNetwork):
    def __init__(self, obs_dim, action_dim, args, device='cpu'):
        super(SimpleQNetwork, self).__init__(obs_dim, action_dim, args, device)
        self.args = args
        self.fc1 = nn.Linear(obs_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def _forward(self, x, *args):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x





if __name__ == '__main__':
    main()
