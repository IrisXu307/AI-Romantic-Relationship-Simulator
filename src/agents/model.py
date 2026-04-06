import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    """
    MLP policy: obs → action logits over N_ACTIONS.
    Uses Tanh activations (better than ReLU for normalized [0,1] inputs).
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def act(self, obs: torch.Tensor) -> tuple[int, torch.Tensor]:
        """Sample one action and return (action_int, log_prob)."""
        dist = Categorical(logits=self.forward(obs))
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return Categorical(logits=self.forward(obs)).log_prob(actions)

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        return Categorical(logits=self.forward(obs)).entropy()


class ValueNet(nn.Module):
    """
    MLP baseline: obs → scalar state value.
    Used to compute advantages (G_t - V(s_t)) for variance reduction.
    """

    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
