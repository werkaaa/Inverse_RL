import torch
from torch.optim import Adam


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class SAC(object):
    def __init__(self, action_space, batch_size, args):

        self.gamma = args.gamma
        self.batch_size = batch_size
        self.action_space = action_space

        # This can be made a learnable parameter (automatic entropy tuning)
        self.alpha = args.alpha
        self.offline = args.offline
        self.soft_update = args.soft_update
        if self.soft_update:
            self.critic_tau = args.critic_tau

        self.target_update_frequency = args.target_update_interval
        self.actor_update_frequency = args.actor_update_frequency

        self.device = torch.device("cuda" if args.cuda else "cpu")

        # TODO: Fill in model parameter according to model definition
        self.critic = args.critic(args.critic_config).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = args.critic(args.critic_config).to(device=self.device)
        hard_update(self.critic_target, self.critic)

        self.actor = args.actor(args.actor_config).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr)

    def get_action(self, state, sample=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        dist = self.actor(state)
        action = dist.sample() if sample else dist.mean
        return action.detach().cpu().numpy()[0]

    def getV(self, obs):
        action, log_prob, _ = self.actor.sample(obs)
        current_Q = self.critic(obs, action)
        current_V = current_Q - self.alpha.detach() * log_prob
        return current_V

    def get_targetV(self, obs):
        action, log_prob, _ = self.actor.sample(obs)
        target_Q = self.critic_target(obs, action)
        target_V = target_Q - self.alpha.detach() * log_prob
        return target_V

    def iq_update_critic(self, policy_batch, expert_batch):

        batch = (
            torch.cat(
                [policy_data, expert_data], dim=0) for policy_data, expert_data in zip(policy_batch, expert_batch)
        )
        is_expert = torch.cat([torch.zeros_like(policy_batch[0], dtype=torch.bool),
                               torch.ones_like(expert_batch[0], dtype=torch.bool)], dim=0)
        obs, next_obs, action, reward, done = batch

        # IQ-Learn with X^2 divergence
        # Calculate 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
        current_Q = self.critic(obs, action)
        y = (1 - done) * self.gamma * self.getV(next_obs)

        # We use target critic for stability
        with torch.no_grad():
            y = (1 - done) * self.gamma * self.get_targetV(next_obs)

        reward = (current_Q - y)[is_expert]
        loss = - reward.mean()

        # Calculate 2nd term of the loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
        value_loss = (self.getV(obs) - y).mean()
        loss += value_loss

        # Use χ2 divergence (adds an extra term to the loss)
        chi2_loss = 1 / (4 * self.alpha) * (reward ** 2).mean()
        loss += chi2_loss

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        losses = {
            'loss/iq_critic': loss.item()
        }

        return losses

    def update_actor(self, obs):
        action, log_prob, _ = self.actor.sample(obs)
        actor_Q = self.critic(obs, action)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        losses = {
            'loss/actor': actor_loss.item(),
            'actor_loss/entropy': -log_prob.mean().item()}

        return losses

    def iq_update(self, policy_buffer, expert_buffer, step):
        # TODO: Implement replay buffers
        policy_batch = policy_buffer.get_samples(self.batch_size)
        expert_batch = expert_buffer.get_samples(self.batch_size)

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
