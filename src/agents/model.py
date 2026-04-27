import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    """
    Split-head MLP policy: obs → action logits over N_ACTIONS.

    Architecture:
      - personality_head: processes X_self (first x_dim dims) — who the agent IS
      - situation_head:   processes X_partner + Y + event — what is happening
      - action_head:      merges both embeddings → action logits

    This forces the network to explicitly condition action choices on personality,
    so an emotional agent and a rational agent learn different responses to the
    same situation.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int, x_dim: int = 10):
        super().__init__()
        self.x_dim = x_dim
        sit_dim = obs_dim - x_dim
        half = hidden_dim // 2

        self.personality_head = nn.Sequential(
            nn.Linear(x_dim, half),
            nn.Tanh(),
            nn.Linear(half, half),
            nn.Tanh(),
            nn.Linear(half, half),
            nn.Tanh(),
        )
        self.situation_head = nn.Sequential(
            nn.Linear(sit_dim, half),
            nn.Tanh(),
            nn.Linear(half, half),
            nn.Tanh(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        personality = self.personality_head(x[:, :self.x_dim])
        situation   = self.situation_head(x[:, self.x_dim:])
        merged = torch.cat([personality, situation], dim=-1)
        return self.action_head(merged)

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
    Split-head MLP baseline: obs → scalar state value.
    Same personality/situation split as PolicyNet for consistent representations.
    """

    def __init__(self, obs_dim: int, hidden_dim: int, x_dim: int = 10):
        super().__init__()
        self.x_dim = x_dim
        sit_dim = obs_dim - x_dim
        half = hidden_dim // 2

        self.personality_head = nn.Sequential(
            nn.Linear(x_dim, half),
            nn.Tanh(),
            nn.Linear(half, half),
            nn.Tanh(),
            nn.Linear(half, half),
            nn.Tanh(),
        )
        self.situation_head = nn.Sequential(
            nn.Linear(sit_dim, half),
            nn.Tanh(),
            nn.Linear(half, half),
            nn.Tanh(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        personality = self.personality_head(x[:, :self.x_dim])
        situation   = self.situation_head(x[:, self.x_dim:])
        merged = torch.cat([personality, situation], dim=-1)
        return self.value_head(merged).squeeze(-1)
