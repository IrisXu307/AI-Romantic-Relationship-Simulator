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
import multiprocessing as mp
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml

from src.agents.agent import Agent
from src.env.marriage_env import MarriageEnv
from src.env.state import _TRAIT_NAMES, X_DIM

# ── Personality archetypes for interpretability eval ─────────────────────────
#
# Fixed trait profiles used during eval to show that different personalities
# lead to different action distributions — even with the same trained policy.
# Each run injects these traits into both agents and records what they choose.

ARCHETYPES: dict[str, dict[str, float]] = {
    # Secure: emotionally intelligent, stable, loving, reliable — the gold standard.
    # Non-defining traits (iq, rational_thinking) = 0.5.
    "secure":     dict(mental_stability=0.9, faithfulness=0.9, kindness=0.85,
                       eq=0.85, ability_to_love=0.85, emotional_reasoning=0.75,
                       responsibility=0.8,
                       iq=0.5, rational_thinking=0.5),
    # Anxious attachment: emotionally intense, craves connection, mentally volatile.
    # Non-defining traits (iq, rational_thinking, kindness, faithfulness, responsibility) = 0.5.
    "emotional":  dict(eq=0.85, emotional_reasoning=0.85, ability_to_love=0.9,
                       mental_stability=0.3,
                       iq=0.5, rational_thinking=0.5, kindness=0.5,
                       faithfulness=0.5, responsibility=0.5),
    # Dismissive attachment: logic-driven, emotionally reserved, suppresses feelings.
    # Non-defining traits (kindness, faithfulness, responsibility, mental_stability) = 0.5.
    "rational":   dict(iq=0.85, rational_thinking=0.85, eq=0.3,
                       emotional_reasoning=0.25,
                       kindness=0.5, faithfulness=0.5, responsibility=0.5,
                       mental_stability=0.5, ability_to_love=0.5),
    # Fearful-avoidant: fears intimacy, distrusts love, low emotional literacy, unreliable.
    # Non-defining traits (iq, rational_thinking, kindness, responsibility) = 0.5.
    "avoidant":   dict(eq=0.2, ability_to_love=0.15, emotional_reasoning=0.2,
                       mental_stability=0.25, faithfulness=0.35,
                       iq=0.5, rational_thinking=0.5, kindness=0.5, responsibility=0.5),
}


EVAL_CATEGORIES = {
    "Financial":    ["job_loss", "promotion", "financial_crisis", "financial_windfall", "financial_disagreement"],
    "Family":       ["new_child", "parenting_conflict", "family_death"],
    "Health":       ["health_crisis", "mental_health_episode"],
    "Relationship": ["infidelity", "emotional_conflict", "romantic_gesture"],
    "Life Change":  ["relocation", "shared_achievement"],
}


# ── Archetype evaluation ─────────────────────────────────────────────────────

def log_archetypes(
    env: MarriageEnv, agent_h: Agent, agent_w: Agent, episode: int, n_runs: int = 30
) -> dict[str, list[float]]:
    """
    For each personality archetype, run n_runs full episodes with that archetype's
    traits injected into both agents, print the action distribution, save a bar
    chart snapshot, and return the distributions for long-run plotting.

    Returns: {arch_name: [frac_support, frac_argue, frac_ignore, frac_compromise, frac_withdraw]}
    """
    from collections import Counter
    from src.env.events import ACTION_NAMES as ANAMES

    header = f"    {'archetype':<12}" + "".join(f"  {a:<11}" for a in ANAMES)
    print(f"\n  [personality archetypes ep={episode}]")
    print(header)
    print("    " + "─" * (12 + 13 * len(ANAMES)))

    distributions: dict[str, list[float]] = {}

    for arch_name, traits in ARCHETYPES.items():
        counts: Counter = Counter()
        innate = np.array([traits.get(t, 0.5) for t in _TRAIT_NAMES], dtype=np.float32)

        for _ in range(n_runs):
            env.reset()
            for attr, val in traits.items():
                setattr(env.x_h, attr, float(val))
                setattr(env.x_w, attr, float(val))
            env.x_h.innate = innate.copy()
            env.x_w.innate = innate.copy()
            obs_h = env._get_obs("h")
            obs_w = env._get_obs("w")

            done = False
            while not done:
                action_h, _, _ = agent_h.act(obs_h)
                action_w, _, _ = agent_w.act(obs_w)
                obs_h, _, done, _, info = env.step([action_h, action_w])
                obs_w = info["obs_w"]
                counts[action_h] += 1

        total = sum(counts.values())
        fracs = [counts[i] / total for i in range(len(ANAMES))]
        distributions[arch_name] = fracs
        dist_str = "".join(f"  {f:>10.1%}" for f in fracs)
        print(f"    {arch_name:<12}{dist_str}")

    print()
    _plot_archetype_snapshot(distributions, episode)
    return distributions


def _plot_archetype_snapshot(distributions: dict[str, list[float]], episode: int):
    """Save a grouped bar chart of action distributions per archetype."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    from src.env.events import ACTION_NAMES as ANAMES

    archs  = list(distributions.keys())
    n_arch = len(archs)
    n_act  = len(ANAMES)
    x      = np.arange(n_arch)
    width  = 0.15
    colors = ["#4C9BE8", "#E8694C", "#8DC87A", "#F5C842", "#B07FD4"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (action, color) in enumerate(zip(ANAMES, colors)):
        vals = [distributions[a][i] for a in archs]
        ax.bar(x + i * width, vals, width, label=action, color=color)

    ax.set_xticks(x + width * (n_act - 1) / 2)
    ax.set_xticklabels(archs, fontsize=11)
    ax.set_ylabel("Action frequency")
    ax.set_ylim(0, 1)
    ax.set_title(f"Action distribution by personality archetype  (ep {episode})")
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    plt.tight_layout()
    out = f"data/archetypes_ep{episode:07d}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Archetype chart → {out}")


# ── Category evaluation ───────────────────────────────────────────────────────

def evaluate_by_category(env: MarriageEnv, agent_h: Agent, agent_w: Agent, n_eval: int = 50) -> dict:
    """
    For each event category, run n_eval episodes where only events from that
    category can fire (50% chance each step, else no event).
    Returns mean final happiness per category.
    """
    results = {}
    for cat, event_names in EVAL_CATEGORIES.items():
        cat_events = [e for e in env.events.events if e["name"] in event_names]
        happiness_list = []
        for _ in range(n_eval):
            obs, _ = env.reset()
            done = False
            while not done:
                env.current_event = random.choice(cat_events) if (cat_events and random.random() < 0.5) else None
                action_h, _, _ = agent_h.act(obs)
                action_w, _, _ = agent_w.act(obs)
                obs, _, done, _, info = env.step([action_h, action_w])
            happiness_list.append(info["happiness"])
        results[cat] = float(np.mean(happiness_list))
    return results


# ── Background eval worker ───────────────────────────────────────────────────

def _eval_worker(
    state_dicts: dict,
    config_path: str,
    events_path: str,
    obs_dim: int,
    hidden_dim: int,
    n_eval: int,
    episode: int,
    result_queue: mp.Queue,
) -> None:
    """
    Runs in a separate process so the main training loop is never blocked.
    Receives a snapshot of model weights, builds its own env + agents,
    runs evaluate_by_category + log_archetypes, then puts results on the queue.
    """
    device = torch.device("cpu")
    env = MarriageEnv(config_path, events_path)
    agent_h = Agent(obs_dim, 5, hidden_dim, lr=1e-3, device=device, x_dim=X_DIM)
    agent_w = Agent(obs_dim, 5, hidden_dim, lr=1e-3, device=device, x_dim=X_DIM)
    agent_h.policy.load_state_dict(state_dicts["h_policy"])
    agent_h.value.load_state_dict(state_dicts["h_value"])
    agent_w.policy.load_state_dict(state_dicts["w_policy"])
    agent_w.value.load_state_dict(state_dicts["w_value"])
    agent_h.policy.eval()
    agent_w.policy.eval()

    cat_scores = evaluate_by_category(env, agent_h, agent_w, n_eval=n_eval)
    cat_scores["episode"] = episode
    arch_dists = log_archetypes(env, agent_h, agent_w, episode)
    cat_scores["archetypes"] = arch_dists
    result_queue.put((episode, cat_scores))


def _snapshot_weights(agent_h: "Agent", agent_w: "Agent") -> dict:
    """Deep-copy model weights to plain CPU dicts (safe to pickle across processes)."""
    return {
        "h_policy": {k: v.cpu().clone() for k, v in agent_h.policy.state_dict().items()},
        "h_value":  {k: v.cpu().clone() for k, v in agent_h.value.state_dict().items()},
        "w_policy": {k: v.cpu().clone() for k, v in agent_w.policy.state_dict().items()},
        "w_value":  {k: v.cpu().clone() for k, v in agent_w.value.state_dict().items()},
    }


# ── Reflection ────────────────────────────────────────────────────────────────

# Major life events warrant larger trait shifts than everyday friction.
# Values are caps on any single trait change; config x_change_magnitude is the default.
REFLECTION_MAGNITUDES: dict[str, float] = {
    "infidelity":            0.20,   # trust shattered — identity-level disruption
    "family_death":          0.15,   # grief reshapes emotional capacity and stability
    "health_crisis":         0.12,   # confronting mortality changes priorities
    "mental_health_episode": 0.12,   # psychological rupture forces self-examination
    "new_child":             0.10,   # parenthood is a documented personality shift
    "relocation":            0.08,   # uprooting disrupts identity and social self
    "financial_crisis":      0.07,   # sustained stress erodes responsibility and stability
}


def reflect(x_traits, delta_y: dict, magnitude: float):
    """
    Nudge an agent's *learned* traits after a significant life event.

    Personality-typed: the same event grows different traits depending on who you are.
      - happiness ↑: analytical agents (high IQ/RT) grow rational_thinking; emotional grow eq
      - happiness ↑: rational agents also grow eq slowly — they can learn emotional skills analytically
      - wealth ↑: high rational_thinking agents grow iq and rational_thinking
      - stability ↑: already-responsible agents reinforce responsibility further
      - pressure ↑: unstable agents take a larger mental_stability hit
      - love_support ↑: high-EQ agents grow eq/ability_to_love most
    """
    dh = delta_y.get("happiness",    0.0)
    ds = delta_y.get("stability",    0.0)
    dp = delta_y.get("pressure",     0.0)
    dl = delta_y.get("love_support", 0.0)
    dw = delta_y.get("wealth",       0.0)

    iq_eff   = x_traits.effective("iq")
    eq_eff   = x_traits.effective("eq")
    rt_eff   = x_traits.effective("rational_thinking")
    ms_eff   = x_traits.effective("mental_stability")
    resp_eff = x_traits.effective("responsibility")

    adj = {name: 0.0 for name in _TRAIT_NAMES}

    # Happiness: analytical types learn from what worked; emotional types deepen feeling.
    # Rational types also gain eq slowly — they can learn emotional skills through analysis.
    if dh > 0:
        adj["rational_thinking"] += magnitude * dh * iq_eff
        adj["eq"]                += magnitude * dh * (1.0 - iq_eff)
        adj["eq"]                += magnitude * dh * rt_eff * 0.25   # rational self-improvement path
        adj["ability_to_love"]   += magnitude * dh * eq_eff * 0.5
    else:
        adj["mental_stability"]  += magnitude * dh  # universal hit for unhappiness

    # Stability: reinforces existing strengths
    adj["mental_stability"]  += magnitude * ds * 0.5
    adj["responsibility"]    += magnitude * max(ds, 0.0) * resp_eff * 0.4

    # Pressure: unstable agents are hit harder
    instability = max(0.0, 0.5 - ms_eff) * 2.0
    adj["mental_stability"]  -= magnitude * max(dp, 0.0) * (0.3 + 0.4 * instability)

    # Love/connection: high-EQ agents absorb the growth most
    if dl > 0:
        adj["kindness"]         += magnitude * dl * 0.3
        adj["eq"]               += magnitude * dl * eq_eff * 0.3
        adj["ability_to_love"]  += magnitude * dl * eq_eff * 0.4

    # Financial success: rational types extract strategic insight
    if dw > 0:
        adj["iq"]                += magnitude * dw * rt_eff * 0.3
        adj["rational_thinking"] += magnitude * dw * rt_eff * 0.4
        adj["responsibility"]    += magnitude * dw * resp_eff * 0.2

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
    steps = 0
    while not done:
        action_h, log_prob_h, value_h = agent_h.act(obs_h)
        action_w, log_prob_w, value_w = agent_w.act(obs_w)

        obs_h_next, _scalar_reward, done, _, info = env.step([action_h, action_w])
        obs_w_next = info["obs_w"]
        reward_h   = info["reward_h"]
        reward_w   = info["reward_w"]

        if train:
            agent_h.store(obs_h, action_h, log_prob_h, reward_h, value_h)
            agent_w.store(obs_w, action_w, log_prob_w, reward_w, value_w)

        if info["reflection_triggered"]:
            event_name = info.get("event", "none")
            magnitude = REFLECTION_MAGNITUDES.get(event_name, x_change_magnitude)
            reflect(env.x_h, info["delta_h"], magnitude)
            reflect(env.x_w, info["delta_w"], magnitude)
            reflections += 1

        ep_reward_h += reward_h
        ep_reward_w += reward_w
        ep_happiness.append(info["happiness"])
        ep_stability.append(info["stability"])
        steps += 1

        obs_h, obs_w = obs_h_next, obs_w_next

    return {
        "reward_h":        ep_reward_h,
        "reward_w":        ep_reward_w,
        "reward_h_mean":   ep_reward_h / steps,   # per-step, comparable to happiness
        "reward_w_mean":   ep_reward_w / steps,
        "steps":           steps,
        "mean_happiness":  float(np.mean(ep_happiness)),
        "mean_stability":  float(np.mean(ep_stability)),
        "final_happiness": ep_happiness[-1],
        "final_stability": ep_stability[-1],
        "reflections":     reflections,
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
    parser.add_argument("--eval-every", type=int,   default=1000, help="Evaluate per-category every N episodes")
    parser.add_argument("--eval-log",   default="data/eval_history.json")
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

    # PPO hyperparameters
    clip_eps    = cfg["ppo"]["clip_eps"]
    ppo_epochs  = cfg["ppo"]["epochs"]
    gae_lambda  = cfg["ppo"]["gae_lambda"]
    entropy_start = cfg["ppo"]["entropy_start"]
    entropy_end   = cfg["ppo"]["entropy_end"]
    update_every  = cfg["ppo"].get("update_every", 1)

    # Agents
    agent_h = Agent(obs_dim, n_actions, hidden_dim, lr, device, clip_eps, ppo_epochs, gae_lambda, x_dim=X_DIM)
    agent_w = Agent(obs_dim, n_actions, hidden_dim, lr, device, clip_eps, ppo_epochs, gae_lambda, x_dim=X_DIM)

    start_ep   = 0
    all_metrics: list[dict] = []
    eval_history: list[dict] = []

    if args.resume and Path(args.checkpoint).exists():
        start_ep, all_metrics = load_checkpoint(args.checkpoint, agent_h, agent_w)
        print(f"Resumed from episode {start_ep}")

    # Rolling window for console display
    window: deque[dict] = deque(maxlen=100)

    print(f"\n{'Ep':>7}  {'Rew-H':>7}  {'Rew-W':>7}  {'Happiness':>9}  "
          f"{'Stability':>9}  {'Reflect':>7}  {'Loss-H':>8}")
    print("─" * 72)

    # Background eval state
    _eval_proc:  mp.Process | None = None
    _eval_queue: mp.Queue = mp.Queue()

    def _drain_eval_queue():
        """Collect any finished background eval results into eval_history."""
        while not _eval_queue.empty():
            ep_done, cat_scores = _eval_queue.get_nowait()
            eval_history.append(cat_scores)
            scores_str = "  ".join(
                f"{k}={v:.3f}" for k, v in cat_scores.items()
                if k not in ("episode", "archetypes")
            )
            print(f"  [eval ep={ep_done}] {scores_str}")

    t0 = time.time()

    for ep in range(start_ep, n_episodes):
        progress     = (ep - start_ep) / max(1, n_episodes - 1 - start_ep)
        entropy_coef = entropy_start + (entropy_end - entropy_start) * progress

        # Collect update_every episodes before each PPO update.
        # Trajectories are accumulated inside each agent's buffer across calls.
        ep_info = run_episode(env, agent_h, agent_w, x_change_mag, train=True)
        for _ in range(update_every - 1):
            extra = run_episode(env, agent_h, agent_w, x_change_mag, train=True)
            # Merge extra episode stats into ep_info (running sum; averaged below)
            for k in ("reward_h", "reward_w", "reward_h_mean", "reward_w_mean",
                      "mean_happiness", "mean_stability", "final_happiness",
                      "final_stability", "reflections", "steps"):
                ep_info[k] = ep_info.get(k, 0) + extra.get(k, 0)
        if update_every > 1:
            for k in ("reward_h_mean", "reward_w_mean", "mean_happiness",
                      "mean_stability", "final_happiness", "final_stability",
                      "reflections"):
                ep_info[k] /= update_every

        loss_h_pol, loss_h_val = agent_h.update(gamma, entropy_coef)
        loss_w_pol, loss_w_val = agent_w.update(gamma, entropy_coef)

        ep_info.update({
            "episode":       ep,
            "loss_h_policy": loss_h_pol,
            "loss_h_value":  loss_h_val,
            "loss_w_policy": loss_w_pol,
            "loss_w_value":  loss_w_val,
        })
        all_metrics.append(ep_info)
        window.append(ep_info)

        # Console log every 1000 episodes
        if (ep + 1) % 1000 == 0 or ep == start_ep:
            keys = ["reward_h_mean", "reward_w_mean", "mean_happiness",
                    "mean_stability", "reflections", "loss_h_policy"]
            avg = {k: float(np.mean([m[k] for m in window])) for k in keys}
            print(
                f"{ep+1:>7}  {avg['reward_h_mean']:>7.3f}  {avg['reward_w_mean']:>7.3f}  "
                f"{avg['mean_happiness']:>9.3f}  {avg['mean_stability']:>9.3f}  "
                f"{avg['reflections']:>7.1f}  {avg['loss_h_policy']:>8.4f}"
            )

        # Periodic checkpoint
        if (ep + 1) % args.save_every == 0:
            save_checkpoint(args.checkpoint, ep + 1, agent_h, agent_w, all_metrics)

        # Per-category evaluation — runs in a background process, never blocks training.
        # Drain previous results first, then launch a new worker if the last one finished.
        if (ep + 1) % args.eval_every == 0:
            _drain_eval_queue()
            ckpt_path = Path(args.checkpoint).parent / f"ep_{ep+1}.pt"
            save_checkpoint(str(ckpt_path), ep + 1, agent_h, agent_w, all_metrics)

            # Skip if previous eval is still running (training is faster than eval)
            if _eval_proc is None or not _eval_proc.is_alive():
                weights = _snapshot_weights(agent_h, agent_w)
                _eval_proc = mp.Process(
                    target=_eval_worker,
                    args=(weights, args.config, args.events,
                          obs_dim, hidden_dim, 200, ep + 1, _eval_queue),
                    daemon=True,
                )
                _eval_proc.start()
                print(f"  [eval ep={ep+1}] launched in background (pid={_eval_proc.pid})")

    # Wait for the last background eval to finish and collect its results
    if _eval_proc is not None and _eval_proc.is_alive():
        print("Waiting for background eval to finish...")
        _eval_proc.join()
    _drain_eval_queue()

    elapsed = time.time() - t0
    print(f"\nDone — {elapsed:.1f}s total  ({elapsed / n_episodes * 1000:.1f} ms/ep)")

    # Final checkpoint + full metrics log
    save_checkpoint(args.checkpoint, n_episodes, agent_h, agent_w, all_metrics)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Checkpoint → {args.checkpoint}")
    print(f"Metrics    → {args.log}")
    if eval_history:
        Path(args.eval_log).parent.mkdir(parents=True, exist_ok=True)
        with open(args.eval_log, "w") as f:
            json.dump(eval_history, f, indent=2)
        print(f"Eval log   → {args.eval_log}")

    # Optional training curves
    if args.plot:
        _plot(all_metrics, eval_history)


def _plot(metrics: list[dict], eval_history: list[dict] | None = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    from src.env.events import ACTION_NAMES as ANAMES

    def smooth(vals, w=100):
        if len(vals) < w:
            return vals
        kernel = np.ones(w) / w
        return np.convolve(vals, kernel, mode="valid").tolist()

    # ── Training curves (2×2) ─────────────────────────────────────────────────
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

    # ── Archetype evolution over training ─────────────────────────────────────
    # One subplot per archetype: lines show how each action's frequency evolves.
    if not eval_history:
        return
    arch_records = [e for e in eval_history if "archetypes" in e]
    if not arch_records:
        return

    episodes   = [e["episode"] for e in arch_records]
    arch_names = list(ARCHETYPES.keys())
    colors     = ["#4C9BE8", "#E8694C", "#8DC87A", "#F5C842", "#B07FD4"]

    fig2, axes2 = plt.subplots(1, len(arch_names), figsize=(5 * len(arch_names), 4), sharey=True)
    fig2.suptitle("How each personality's action preferences evolve over training", fontsize=13)

    for ax, arch in zip(axes2, arch_names):
        for i, (action, color) in enumerate(zip(ANAMES, colors)):
            vals = [e["archetypes"][arch][i] for e in arch_records]
            ax.plot(episodes, vals, label=action, color=color, linewidth=1.8)
        ax.set_title(arch, fontsize=11)
        ax.set_xlabel("Episode")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        if ax is axes2[0]:
            ax.set_ylabel("Action frequency")
            ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    out2 = "data/archetype_evolution.png"
    plt.savefig(out2, dpi=120)
    print(f"Plot saved → {out2}")
    plt.show()


if __name__ == "__main__":
    main()
