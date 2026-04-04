# AI Romantic Relationship Simulator

A reinforcement learning research project that simulates a marriage between two AI agents across a 55-year lifespan (ages 25–80). Agents learn optimal interpersonal behaviors — not through hard-coded rules, but through a neural network policy trained on experience and reward.

> **Motivation:** Can an AI learn what makes a marriage last? This project treats partnership as a sequential decision-making problem and applies policy gradient methods to find out.

---

## Table of Contents

- [Research Goal](#research-goal)
- [System Design](#system-design)
  - [Variable Set X — Internal Traits](#variable-set-x--internal-traits)
  - [Variable Set Y — Environmental State](#variable-set-y--environmental-state)
  - [Events](#events)
  - [Reflection Mechanism](#reflection-mechanism)
  - [Learning: Policy Gradient](#learning-policy-gradient)
- [Why This Is Real ML Training](#why-this-is-real-ml-training)
- [Repo Structure](#repo-structure)
- [Getting Started](#getting-started)
- [Training Loop](#training-loop)
- [Results & Metrics](#results--metrics)
- [Roadmap](#roadmap)

---

## Research Goal

To determine whether a neural network policy can learn optimal responses to life events in a long-term relationship — maximizing **happiness** and **marriage stability** across a simulated lifespan — given two agents with distinct internal trait profiles.

---

## System Design

### Variable Set X — Internal Traits

Each agent is initialized with their own independent X vector. These evolve only during **Reflection**.

| Variable | Description |
|---|---|
| `iq` | Cognitive intelligence |
| `eq` | Emotional intelligence |
| `rational_thinking` | Tendency toward logic-based decisions |
| `emotional_reasoning` | Tendency toward feeling-based decisions |
| `kindness` | Baseline prosocial disposition |
| `ability_to_love` | Capacity for deep attachment |
| `faithfulness` | Commitment to the relationship |
| `responsibility` | Follow-through on obligations |
| `mental_stability` | Resilience to stress and volatility |
| `kids` | Presence/number of children (shared, affects both agents) |

### Variable Set Y — Environmental State

Y variables are shared between both agents and change in response to events and actions.

| Variable | Description |
|---|---|
| `wealth` | Financial status of the couple |
| `love_support` | Perceived mutual affection and support |
| `pressure` | External stress load (work, family, health, etc.) |
| `happiness` | Primary reward signal |
| `stability` | Secondary reward signal; measures relationship continuity |

### Events

Random life events are sampled each simulation step (each "year" of the marriage). Each event targets one or more Y variables with a baseline delta. Examples:

| Event | Affected Y | Default ΔY |
|---|---|---|
| Job loss | wealth, pressure | −0.3, +0.4 |
| New child | kids, pressure, love_support | +1, +0.3, ±0.2 |
| Infidelity | faithfulness, stability, love_support | −0.8, −0.5, −0.6 |
| Health crisis | wealth, pressure, love_support | −0.2, +0.5, ±0.3 |
| Financial windfall | wealth, pressure | +0.4, −0.2 |
| Emotional conflict | love_support, stability | −0.3, −0.2 |
| Shared achievement | happiness, love_support | +0.3, +0.2 |

Each agent independently selects an **action** (response) from a discrete action set. The combined response of both agents determines the actual ΔY applied.

### Reflection Mechanism

Reflection is triggered when `|ΔY| > threshold` on any Y variable. It governs whether an agent's X traits change as a result of accumulated experience.

```
if |delta_Y| > reflection_threshold:
    # Early training: perturb X randomly, observe resulting Y
    # Later training: update X in direction of gradient(reward)
    adjust_X(agent, delta_X)
```

This is the mechanism by which agents are allowed to "grow" — slowly shifting their internal traits in response to major life events, mimicking real psychological change over time.

### Learning: Policy Gradient

The core of the system is a learned policy:

```
action = π_θ(state)
```

Where:
- `π` is a neural network (MLP or LSTM for temporal memory)
- `θ` are the learned parameters
- `state = [X_self, X_partner, Y_shared, event_encoding]`
- `action` is sampled from a probability distribution over the discrete response set

Update rule (REINFORCE):

```
θ ← θ + α · ∇_θ log π_θ(a|s) · G_t
```

Where `G_t` is the discounted return (future happiness + stability) from timestep `t`.

---

## Why This Is Real ML Training

| Property | Rule-based System | This Project |
|---|---|---|
| Behavior source | Hand-coded `if/else` | Learned from experience |
| Parameters | Manual tuning | Gradient updates from reward |
| Generalization | Fixed rules | Adapts to new event sequences |
| Improvement | Only when you rewrite it | Improves each training episode |

The key distinction: **you do not specify what the correct response is**. The agent discovers it by trying responses, observing outcomes, and reinforcing what worked.

---

## Repo Structure

```
ai-romantic-relationship-simulator/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/
│   ├── default.yaml          # hyperparameters, thresholds, sim settings
│   └── events.yaml           # event catalog with base deltas and probabilities
│
├── src/
│   ├── env/
│   │   ├── __init__.py
│   │   ├── marriage_env.py   # Gymnasium-compatible environment
│   │   ├── events.py         # event sampling and delta application
│   │   └── state.py          # state representation and normalization
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── agent.py          # Agent class (holds X, calls policy, handles reflection)
│   │   ├── model.py          # Neural network policy (MLP / LSTM)
│   │   └── reflection.py     # Reflection trigger logic and X update rules
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py        # Training loop, episode rollout, reward computation
│   │   ├── reward.py         # Reward shaping: happiness + stability weighting
│   │   └── buffer.py         # Trajectory buffer for policy gradient updates
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py        # Metrics logging per episode
│       └── visualization.py  # Plot Y trajectories, reward curves, X drift
│
├── scripts/
│   ├── train.py              # Entry point: run training
│   ├── evaluate.py           # Run a single marriage simulation (no gradient update)
│   └── plot_results.py       # Generate charts from saved run data
│
├── data/
│   ├── runs/                 # Saved episode logs (JSON/CSV)
│   └── checkpoints/          # Model weight snapshots
│
└── tests/
    ├── test_env.py
    ├── test_agent.py
    └── test_reflection.py
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- PyTorch
- Gymnasium
- NumPy, PyYAML, Matplotlib

### Install

```bash
git clone https://github.com/your-username/ai-romantic-relationship-simulator.git
cd ai-romantic-relationship-simulator
pip install -r requirements.txt
```

### Train

```bash
python scripts/train.py --config config/default.yaml --episodes 10000
```

### Evaluate a trained policy

```bash
python scripts/evaluate.py --checkpoint data/checkpoints/best_model.pt --render
```

### Plot results

```bash
python scripts/plot_results.py --run data/runs/run_001/
```

---

## Training Loop

```
for episode in range(num_episodes):
    reset marriage (age=25, sample X for each agent, reset Y)

    for age in range(25, 80):
        sample event from event catalog
        state = [X_husband, X_wife, Y_shared, event]

        action_h = π_θ_h(state)    # husband's policy
        action_w = π_θ_w(state)    # wife's policy

        ΔY = compute_delta_Y(event, action_h, action_w)
        Y  = update_Y(Y, ΔY)

        if |ΔY| > reflection_threshold:
            trigger_reflection(agent, ΔY, reward_signal)

        reward = compute_reward(Y.happiness, Y.stability)
        store (state, action, reward) in trajectory buffer

    update θ_h, θ_w via policy gradient on full episode trajectory
```

---

## Results & Metrics

Each episode logs:
- Happiness trajectory (age 25–80)
- Stability trajectory
- X drift per agent (how much traits shifted via reflection)
- Actions taken per event type (behavioral fingerprint)
- Final reward

After training, the goal is to surface:
1. Which X trait profiles correlate with high long-term happiness
2. Which action strategies are robust across different event sequences
3. Whether reflection-driven X changes converge toward a consistent "good partner" profile

---

## Roadmap

- [ ] Implement `marriage_env.py` (Gymnasium environment)
- [ ] Implement event catalog and sampling (`events.py`)
- [ ] Build neural network policy (`model.py`)
- [ ] Implement REINFORCE training loop (`trainer.py`)
- [ ] Add reflection mechanism (`reflection.py`)
- [ ] Add visualization (`visualization.py`)
- [ ] Run baseline experiments (random policy vs. trained)
- [ ] Experiment: vary initial X distributions, compare outcomes
- [ ] Experiment: asymmetric traits (mismatched partners)
- [ ] Write up findings

---

## License

MIT
