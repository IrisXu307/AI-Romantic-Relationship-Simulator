"""
train.py — Main entry point for training the marriage RL agents.

Usage:
    python train.py                          # full run with defaults
    python train.py --episodes 2000          # quick test
    python train.py --resume                 # continue from checkpoint
    python train.py --plot                   # show training curves after run

Both the husband and wife have separate PolicyNet + ValueNet pairs trained
with REINFORCE + value baseline (advantage = G_t - V(s_t)).

Reflection: after any step where |ΔY| > threshold, the agents' learned traits
are nudged based on the outcome — positive outcomes grow emotional/relational
traits; negative outcomes erode mental stability. This is bounded by
x_change_magnitude (0.05) so traits can never fully converge.
"""

import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml

from src.agents.agent import Agent
from src.env.marriage_env import MarriageEnv
from src.env.state import _TRAIT_NAMES


# ── Reflection ────────────────────────────────────────────────────────────────

def reflect(x_traits, delta_y: dict, magnitude: float):
    """
    Nudge an agent's *learned* traits after a significant life event.

    Mapping:
      happiness ↑ → eq, ability_to_love ↑
      happiness ↓ → mental_stability ↓
      stability ↑ → mental_stability, responsibility ↑
      stability ↓ → mental_stability ↓
      pressure  ↑ → mental_stability ↓
      love_support ↑ → kindness, ability_to_love ↑
    """
    dh = delta_y.get("happiness", 0.0)
    ds = delta_y.get("stability", 0.0)
    dp = delta_y.get("pressure",  0.0)
    dl = delta_y.get("love_support", 0.0)

    adj = {name: 0.0 for name in _TRAIT_NAMES}

    adj["eq"]              += magnitude * max(dh, 0)
    adj["ability_to_love"] += magnitude * max(dh, 0) * 0.5
    adj["mental_stability"] += magnitude * dh  # negative dh → negative nudge

    adj["mental_stability"] += magnitude * ds * 0.5
    adj["responsibility"]   += magnitude * max(ds, 0) * 0.3

    adj["mental_stability"] -= magnitude * dp * 0.5  # pressure hurts stability

    adj["kindness"]        += magnitude * max(dl, 0) * 0.3
    adj["ability_to_love"] += magnitude * max(dl, 0) * 0.3

    for name, delta in adj.items():
        if delta != 0.0:
            current = getattr(x_traits, name)
            setattr(x_traits, name, float(np.clip(current + delta, 0.0, 1.0)))

    x_traits.clip()


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    env: MarriageEnv,
    agent_h: Agent,
    agent_w: Agent,
    x_change_magnitude: float,
    train: bool = True,
) -> dict:
    obs_h, info = env.reset()
    obs_w = info["obs_w"]

    ep_reward_h = ep_reward_w = 0.0
    ep_happiness: list[float] = []
    ep_stability: list[float] = []
    reflections = 0

    done = False
    while not done:
        action_h, log_prob_h = agent_h.act(obs_h)
        action_w, log_prob_w = agent_w.act(obs_w)

        obs_h_next, _scalar_reward, done, _, info = env.step([action_h, action_w])
        obs_w_next = info["obs_w"]
        reward_h   = info["reward_h"]
        reward_w   = info["reward_w"]

        if train:
            agent_h.store(obs_h, action_h, log_prob_h, reward_h)
            agent_w.store(obs_w, action_w, log_prob_w, reward_w)

        if info["reflection_triggered"]:
            reflect(env.x_h, info["delta_y"], x_change_magnitude)
            reflect(env.x_w, info["delta_y"], x_change_magnitude)
            reflections += 1

        ep_reward_h += reward_h
        ep_reward_w += reward_w
        ep_happiness.append(info["happiness"])
        ep_stability.append(info["stability"])

        obs_h, obs_w = obs_h_next, obs_w_next

    return {
        "reward_h":       ep_reward_h,
        "reward_w":       ep_reward_w,
        "mean_happiness": float(np.mean(ep_happiness)),
        "mean_stability": float(np.mean(ep_stability)),
        "final_happiness": ep_happiness[-1],
        "final_stability": ep_stability[-1],
        "reflections":    reflections,
    }


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(path: str, episode: int, agent_h: Agent, agent_w: Agent, history: list):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "episode":          episode,
        "agent_h_policy":   agent_h.policy.state_dict(),
        "agent_h_value":    agent_h.value.state_dict(),
        "agent_h_optim":    agent_h.optimizer.state_dict(),
        "agent_w_policy":   agent_w.policy.state_dict(),
        "agent_w_value":    agent_w.value.state_dict(),
        "agent_w_optim":    agent_w.optimizer.state_dict(),
        "history":          history[-1000:],  # cap for file size
    }, path)


def load_checkpoint(path: str, agent_h: Agent, agent_w: Agent) -> tuple[int, list]:
    ckpt = torch.load(path, weights_only=False)
    agent_h.policy.load_state_dict(ckpt["agent_h_policy"])
    agent_h.value.load_state_dict(ckpt["agent_h_value"])
    agent_h.optimizer.load_state_dict(ckpt["agent_h_optim"])
    agent_w.policy.load_state_dict(ckpt["agent_w_policy"])
    agent_w.value.load_state_dict(ckpt["agent_w_value"])
    agent_w.optimizer.load_state_dict(ckpt["agent_w_optim"])
    return ckpt["episode"], ckpt.get("history", [])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train marriage RL agents (REINFORCE + value baseline)")
    parser.add_argument("--config",     default="config/default.yaml")
    parser.add_argument("--events",     default="config/events.yaml")
    parser.add_argument("--checkpoint", default="data/checkpoints/latest.pt")
    parser.add_argument("--resume",     action="store_true", help="Continue from checkpoint")
    parser.add_argument("--episodes",   type=int,   default=None,  help="Override episode count from config")
    parser.add_argument("--lr",         type=float, default=None,  help="Override learning rate")
    parser.add_argument("--seed",       type=int,   default=None)
    parser.add_argument("--log",        default="data/metrics.json")
    parser.add_argument("--save-every", type=int,   default=500)
    parser.add_argument("--plot",       action="store_true", help="Plot training curves after run")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Seed everything
    seed = args.seed if args.seed is not None else cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seed: {seed}")

    # Environment
    env = MarriageEnv(args.config, args.events)
    obs_dim   = env.observation_space.shape[0]
    n_actions = 5  # N_ACTIONS per agent

    # Hyperparameters (config, overridable via CLI)
    hidden_dim        = cfg["training"]["hidden_dim"]
    lr                = args.lr if args.lr is not None else cfg["training"]["learning_rate"]
    gamma             = cfg["training"]["gamma"]
    n_episodes        = args.episodes if args.episodes is not None else cfg["training"]["episodes"]
    x_change_mag      = cfg["reflection"]["x_change_magnitude"]

    print(f"obs_dim={obs_dim}  n_actions={n_actions}  hidden={hidden_dim}  "
          f"lr={lr}  gamma={gamma}  episodes={n_episodes}")

    # Agents
    agent_h = Agent(obs_dim, n_actions, hidden_dim, lr, device)
    agent_w = Agent(obs_dim, n_actions, hidden_dim, lr, device)

    start_ep   = 0
    all_metrics: list[dict] = []

    if args.resume and Path(args.checkpoint).exists():
        start_ep, all_metrics = load_checkpoint(args.checkpoint, agent_h, agent_w)
        print(f"Resumed from episode {start_ep}")

    # Rolling window for console display
    window: deque[dict] = deque(maxlen=100)

    print(f"\n{'Ep':>7}  {'Rew-H':>7}  {'Rew-W':>7}  {'Happiness':>9}  "
          f"{'Stability':>9}  {'Reflect':>7}  {'Loss-H':>8}")
    print("─" * 72)

    t0 = time.time()

    for ep in range(start_ep, n_episodes):
        ep_info = run_episode(env, agent_h, agent_w, x_change_mag, train=True)

        loss_h_pol, loss_h_val = agent_h.update(gamma)
        loss_w_pol, loss_w_val = agent_w.update(gamma)

        ep_info.update({
            "episode":       ep,
            "loss_h_policy": loss_h_pol,
            "loss_h_value":  loss_h_val,
            "loss_w_policy": loss_w_pol,
            "loss_w_value":  loss_w_val,
        })
        all_metrics.append(ep_info)
        window.append(ep_info)

        # Console log every 50 episodes
        if (ep + 1) % 50 == 0 or ep == start_ep:
            keys = ["reward_h", "reward_w", "mean_happiness",
                    "mean_stability", "reflections", "loss_h_policy"]
            avg = {k: float(np.mean([m[k] for m in window])) for k in keys}
            print(
                f"{ep+1:>7}  {avg['reward_h']:>7.3f}  {avg['reward_w']:>7.3f}  "
                f"{avg['mean_happiness']:>9.3f}  {avg['mean_stability']:>9.3f}  "
                f"{avg['reflections']:>7.1f}  {avg['loss_h_policy']:>8.4f}"
            )

        # Periodic checkpoint
        if (ep + 1) % args.save_every == 0:
            save_checkpoint(args.checkpoint, ep + 1, agent_h, agent_w, all_metrics)

    elapsed = time.time() - t0
    print(f"\nDone — {elapsed:.1f}s total  ({elapsed / n_episodes * 1000:.1f} ms/ep)")

    # Final checkpoint + full metrics log
    save_checkpoint(args.checkpoint, n_episodes, agent_h, agent_w, all_metrics)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log, "w") as f:
        json.dump(all_metrics, f)
    print(f"Checkpoint → {args.checkpoint}")
    print(f"Metrics    → {args.log}")

    # Optional training curves
    if args.plot:
        _plot(all_metrics)


def _plot(metrics: list[dict]):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    def smooth(vals, w=100):
        if len(vals) < w:
            return vals
        kernel = np.ones(w) / w
        return np.convolve(vals, kernel, mode="valid").tolist()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Marriage RL — Training Curves", fontsize=14)

    axes[0, 0].plot(smooth([m["reward_h"] for m in metrics]), label="Husband")
    axes[0, 0].plot(smooth([m["reward_w"] for m in metrics]), label="Wife")
    axes[0, 0].set_title("Episode Reward (100-ep avg)")
    axes[0, 0].set_xlabel("Episode"); axes[0, 0].legend()

    axes[0, 1].plot(smooth([m["mean_happiness"] for m in metrics]))
    axes[0, 1].set_title("Mean Happiness (100-ep avg)")
    axes[0, 1].set_xlabel("Episode")

    axes[1, 0].plot(smooth([m["mean_stability"] for m in metrics]))
    axes[1, 0].set_title("Mean Stability (100-ep avg)")
    axes[1, 0].set_xlabel("Episode")

    axes[1, 1].plot(smooth([m["loss_h_policy"] for m in metrics]), label="Husband", alpha=0.8)
    axes[1, 1].plot(smooth([m["loss_w_policy"] for m in metrics]), label="Wife",    alpha=0.8)
    axes[1, 1].set_title("Policy Loss (100-ep avg)")
    axes[1, 1].set_xlabel("Episode"); axes[1, 1].legend()

    plt.tight_layout()
    out = "data/training_curves.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"Plot saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
