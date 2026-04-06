import numpy as np
import torch
import torch.nn as nn

from src.agents.model import PolicyNet, ValueNet


class Agent:
    """
    Wraps a PolicyNet + ValueNet for one partner (husband or wife).

    Stores the episode trajectory internally, then computes a REINFORCE
    update with a learned value baseline (advantage = G_t - V(s_t)).
    Gradient clipping is applied to keep training stable.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int,
        lr: float,
        device: torch.device,
    ):
        self.device = device
        self.policy = PolicyNet(obs_dim, n_actions, hidden_dim).to(device)
        self.value  = ValueNet(obs_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr,
        )
        self._trajectory: list[tuple] = []  # (obs, action, log_prob, reward)

    # ── Episode collection ─────────────────────────────────────────────────────

    def act(self, obs: np.ndarray) -> tuple[int, torch.Tensor]:
        """Select an action from the current policy (no grad needed here)."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.policy.act(obs_t)
        return action, log_prob

    def store(self, obs: np.ndarray, action: int, log_prob: torch.Tensor, reward: float):
        self._trajectory.append((obs, action, log_prob, reward))

    def clear(self):
        self._trajectory.clear()

    # ── Policy update ──────────────────────────────────────────────────────────

    def update(self, gamma: float, entropy_coef: float = 0.01) -> tuple[float, float]:
        """
        Run one REINFORCE update over the stored episode trajectory.
        Returns (policy_loss, value_loss) as Python floats.
        """
        if not self._trajectory:
            return 0.0, 0.0

        obs_list, actions, _log_probs, rewards = zip(*self._trajectory)

        # Discounted returns G_t = r_t + γ·r_{t+1} + ...
        returns: list[float] = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        obs_t  = torch.tensor(np.array(obs_list), dtype=torch.float32, device=self.device)
        acts_t = torch.tensor(actions, dtype=torch.long,  device=self.device)
        ret_t  = torch.tensor(returns,  dtype=torch.float32, device=self.device)

        # Normalize returns for training stability
        if ret_t.numel() > 1:
            ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)

        # Advantage = G_t - V(s_t)  (detach value so it doesn't affect policy grad)
        values     = self.value(obs_t)
        advantages = ret_t - values.detach()

        # Policy loss: -E[log π(a|s) · A]  minus entropy bonus
        log_probs = self.policy.log_prob(obs_t, acts_t)
        entropy   = self.policy.entropy(obs_t).mean()
        policy_loss = -(log_probs * advantages).mean() - entropy_coef * entropy

        # Value loss: MSE(V(s), G_t)
        value_loss = nn.functional.mse_loss(values, ret_t)

        loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.value.parameters()),
            max_norm=0.5,
        )
        self.optimizer.step()
        self.clear()

        return policy_loss.item(), value_loss.item()

