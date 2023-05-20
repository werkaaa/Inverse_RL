import numpy as np
import torch
from torch.optim import Adam

from iq_learn.SAC_models import SingleQCritic, DiagGaussianActor


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


class SAC(object):
    def __init__(self, obs_dim, action_dim, args, action_low, action_high):

        self.gamma = args.gamma
        self.alpha_ksi = args.alpha_ksi
        self.loss = args.loss
        self.batch_size = args.train.batch_size
        self.action_low = action_low
        self.action_high = action_high

        if args.train.cuda:
            self.device = torch.device("cuda")
        elif args.train.mps:
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # This can be made a learnable parameter (automatic entropy tuning)
        if args.alpha is None:
            self.log_alpha = torch.tensor(np.log(1e-2)).to(device=self.device)
            self.log_alpha.requires_grad = True
            self.log_alpha_optimizer = Adam([self.alpha],
                                            lr=args.actor.alpha_lr)
        else:
            self.log_alpha = np.log(args.alpha)
        self.target_entropy = -action_dim
        self.offline = args.offline
        self.soft_update = args.soft_update
        if self.soft_update:
            self.critic_tau = args.critic_tau

        self.target_update_frequency = args.target_update_frequency
        self.actor_update_frequency = args.actor_update_frequency

        # TODO: Models are fixed now, but for the future experiments they should be passed in the config.
        self.critic = SingleQCritic(
            obs_dim,
            action_dim,
            args.critic).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.critic.critic_lr)

        self.critic_target = SingleQCritic(
            obs_dim,
            action_dim,
            args.critic).to(device=self.device)
        hard_update(self.critic_target, self.critic)

        self.actor = DiagGaussianActor(
            obs_dim,
            action_dim,
            args.actor).to(device=self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.actor.actor_lr)

        self.train()
        self.critic_target.train()

    @property
    def alpha(self):
        if not torch.is_tensor(self.log_alpha):
            return np.exp(self.log_alpha)
        return self.log_alpha.exp().detach()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def get_action(self, state, sample=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        dist = self.actor(state)
        action = dist.sample() if sample else dist.mean
        action = action.detach().cpu().numpy()[0]
        if (np.abs(self.action_low + np.ones_like(self.action_low)).max() > 1e-6 or
                np.abs(self.action_high - np.ones_like(self.action_high)).max() > 1e-6):
            action = self.action_low + (action + 1.0) * (self.action_high - self.action_low) / 2.0
            action = np.clip(action, self.action_low, self.action_high)

        return action

    def predict(self, state, deterministic=True):
        """Makes the API compatible with stable baselines for evaluation."""
        return self.get_action(state, sample=not deterministic)

    def getV(self, obs):
        action, log_prob, _ = self.actor.sample(obs)
        current_Q = self.critic(obs, action)
        current_V = current_Q - self.alpha * log_prob
        return current_V

    def get_targetV(self, obs):
        action, log_prob, _ = self.actor.sample(obs)
        target_Q = self.critic_target(obs, action)
        target_V = target_Q - self.alpha * log_prob
        return target_V

    def iq_update_critic(self, policy_batch, expert_batch):

        batch = (
            torch.cat(
                [policy_data, expert_data], dim=0) for policy_data, expert_data in zip(policy_batch, expert_batch)
        )
        # Follow the size of the reward vector.
        is_expert = torch.cat([torch.zeros_like(policy_batch[3], dtype=torch.bool),
                               torch.ones_like(expert_batch[3], dtype=torch.bool)], dim=0)
        obs, next_obs, action, enw_reward, done = batch

        current_V = self.getV(obs)
        # We use target critic for stability.
        # Original paper has a flag for that, deciding if we use target critic or the current critic.
        with torch.no_grad():
            next_V = self.get_targetV(next_obs)

        current_Q = self.critic(obs, action)

        loss = iq_loss(self, current_Q, current_V, next_V, obs, done, is_expert)

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        losses = {
            'loss/iq_critic_loss': loss.item()
        }

        return losses

    def update_actor(self, obs):
        action, log_prob, _ = self.actor.sample(obs)
        actor_Q = self.critic(obs, action)

        actor_loss = (self.alpha * log_prob - actor_Q).mean()

        # optimize the actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        losses = {
            'loss/iq_actor_loss': actor_loss.item(),
            'loss/iq_actor_entropy': -log_prob.mean().item(),
            'loss/target_entropy:': self.target_entropy
        }
        if torch.is_tensor(self.log_alpha):
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.log_alpha.exp() *
                          (-log_prob - self.target_entropy).detach()).mean()

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            losses.update({
                'loss/alpha_loss': alpha_loss.item(),
                'loss/alpha_value': self.log_alpha.exp().item()
            })

        return losses

    def iq_update(self, policy_buffer, expert_buffer, step):
        policy_batch = policy_buffer.get_batch(self.batch_size)
        expert_batch = expert_buffer.get_batch(self.batch_size)

        losses = self.iq_update_critic(policy_batch, expert_batch)

        if step % self.actor_update_frequency == 0:
            if self.offline:
                obs = expert_batch[0]
            else:
                # Use both policy and expert observations
                obs = torch.cat([policy_batch[0], expert_batch[0]], dim=0)

            # Alternatively, we could do multiple updates of the actor here
            actor_alpha_losses = self.update_actor(obs)
            losses.update(actor_alpha_losses)

        if step % self.target_update_frequency == 0:
            if self.soft_update:
                soft_update(self.critic_target, self.critic, self.critic_tau)
            else:
                hard_update(self.critic_target, self.critic)

        return losses

    def save(self, path):
        critic_path = f"{path}_critic"
        torch.save(self.critic.state_dict(), critic_path)
        actor_path = f"{path}_actor"
        torch.save(self.actor.state_dict(), actor_path)
        print(f'Saved models to {actor_path} and {critic_path}')

    def load(self, path):
        critic_path = f'{path}_critic'
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        actor_path = f'{path}_actor'
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        print(f'Loaded models from {actor_path} and {critic_path}')
