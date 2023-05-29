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
        loss += value_loss
        chi2_loss = 1 / (4 * self.alpha) * (reward ** 2).mean()
        loss += chi2_loss
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        losses = {
            'loss/critic': loss.item()}
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





