import math

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd


# This initialisation makes the weights neither disappear not vanish
def orthogonal_init_(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [torch.nn.Linear(input_dim, output_dim)]
    else:
        mods = [torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(inplace=True)]
        mods.append(torch.nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    return torch.nn.Sequential(*mods)


class SingleQCritic(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, args):
        super(SingleQCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.args = args

        # Q architecture
        self.Q = mlp(obs_dim + action_dim, args.hidden_dim, 1, args.hidden_depth)

        # Apply custom weight initialisation
        self.apply(orthogonal_init_)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q = self.Q(obs_action)

        # I'm not sure where this comes from.
        # if self.args.method.tanh:
        #     q = torch.tanh(q) * 1/(1-self.args.gamma)

        return q


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(torch.nn.Module):
    """torch.distributions implementation of a diagonal Gaussian policy."""

    def __init__(self, obs_dim, action_dim, args):
        super().__init__()

        self.log_std_bounds = args.log_std_bounds
        self.trunk = mlp(obs_dim, args.hidden_dim, 2 * action_dim,
                         args.hidden_depth)

        self.outputs = dict()
        self.apply(orthogonal_init_)

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        dist = SquashedNormal(mu, std)
        return dist

    def sample(self, obs):
        dist = self.forward(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        return action, log_prob, dist.mean
