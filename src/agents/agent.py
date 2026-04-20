import numpy as np
import torch
import torch.nn as nn

from src.agents.model import PolicyNet, ValueNet


class Agent:
    """
    Wraps a PolicyNet + ValueNet for one partner (husband or wife).

    Uses PPO (Proximal Policy Optimization) with:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Multiple update epochs per episode
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int,
        lr: float,
        device: torch.device,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        gae_lambda: float = 0.95,
        x_dim: int = 10,
    ):
        self.device = device
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.gae_lambda = gae_lambda

        self.policy = PolicyNet(obs_dim, n_actions, hidden_dim, x_dim=x_dim).to(device)
        self.value  = ValueNet(obs_dim, hidden_dim, x_dim=x_dim).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr,
        )
        self._trajectory: list[tuple] = []  # (obs, action, log_prob, reward, value)
        # Running return statistics — updated across episodes so the value network
        # can distinguish a great marriage from a mediocre one (per-episode norm kills this signal).
        self._ret_mean: float = 0.0
        self._ret_std:  float = 1.0
        self._ret_ema_decay: float = 0.99

    # ── Episode collection ─────────────────────────────────────────────────────

    def act(self, obs: np.ndarray) -> tuple[int, torch.Tensor, float]:
        """Sample action; return (action, log_prob, value_estimate)."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.policy.act(obs_t)
            value = self.value(obs_t).item()
        return action, log_prob, value

    def store(self, obs: np.ndarray, action: int, log_prob: torch.Tensor,
              reward: float, value: float):
        self._trajectory.append((obs, action, log_prob, reward, value))

    def clear(self):
        self._trajectory.clear()

    # ── PPO update ─────────────────────────────────────────────────────────────

    def update(self, gamma: float, entropy_coef: float = 0.01) -> tuple[float, float]:
        """
        Run PPO update over the stored episode trajectory.
        Returns (mean_policy_loss, mean_value_loss) across epochs.
        """
        if not self._trajectory:
            return 0.0, 0.0

        obs_list, actions, old_log_probs, rewards, values = zip(*self._trajectory)

        obs_t    = torch.tensor(np.array(obs_list), dtype=torch.float32, device=self.device)
        acts_t   = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_lp_t = torch.stack(list(old_log_probs)).detach().to(self.device)

        vals_np    = np.array(values,  dtype=np.float32)
        rewards_np = np.array(rewards, dtype=np.float32)

        # Generalized Advantage Estimation (GAE)
        advantages = np.zeros_like(rewards_np)
        gae = 0.0
        for t in reversed(range(len(rewards_np))):
            next_val = vals_np[t + 1] if t + 1 < len(vals_np) else 0.0
            delta = rewards_np[t] + gamma * next_val - vals_np[t]
            gae = delta + gamma * self.gae_lambda * gae
            advantages[t] = gae

        returns = advantages + vals_np  # GAE returns as value targets

        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns,    dtype=torch.float32, device=self.device)

        # Normalize advantages
        if adv_t.numel() > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # Normalize returns using running cross-episode mean/std.
        # Per-episode normalization collapses the signal that separates good marriages from bad ones.
        ep_mean = float(ret_t.mean())
        ep_std  = max(float(ret_t.std()), 1e-8)
        d = self._ret_ema_decay
        self._ret_mean = d * self._ret_mean + (1 - d) * ep_mean
        self._ret_std  = d * self._ret_std  + (1 - d) * ep_std
        ret_norm = (ret_t - self._ret_mean) / (self._ret_std + 1e-8)

        total_pol_loss = 0.0
        total_val_loss = 0.0

        for _ in range(self.ppo_epochs):
            new_log_probs = self.policy.log_prob(obs_t, acts_t)
            entropy       = self.policy.entropy(obs_t).mean()

            # Clipped surrogate objective
            ratio  = torch.exp(new_log_probs - old_lp_t)
            surr1  = ratio * adv_t
            surr2  = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy

            # Value loss
            value_loss = nn.functional.mse_loss(self.value(obs_t), ret_norm)

            loss = policy_loss + 0.5 * value_loss
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value.parameters()),
                max_norm=0.5,
            )
            self.optimizer.step()

            total_pol_loss += policy_loss.item()
            total_val_loss += value_loss.item()

        self.clear()
        return total_pol_loss / self.ppo_epochs, total_val_loss / self.ppo_epochs
