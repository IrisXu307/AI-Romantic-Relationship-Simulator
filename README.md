# AI Romantic Relationship Simulator

A reinforcement learning research project that simulates a marriage between two AI agents across a 55-year lifespan (ages 25–80). Each agent learns optimal interpersonal behaviors through a neural network policy trained with Proximal Policy Optimization (PPO) — not hard-coded rules.

> **Motivation:** Can an AI learn what makes a marriage last? This project treats long-term partnership as a sequential multi-agent decision problem and applies deep RL to find out.

---

## Table of Contents

- [Research Goal](#research-goal)
- [System Design](#system-design)
  - [Internal Traits (X)](#internal-traits-x)
  - [Environmental State (Y)](#environmental-state-y)
  - [Events](#events)
  - [Reflection Mechanism](#reflection-mechanism)
  - [Learning: PPO with Value Baseline](#learning-ppo-with-value-baseline)
- [Personality Archetypes](#personality-archetypes)
- [Repo Structure](#repo-structure)
- [Getting Started](#getting-started)
- [Training Loop](#training-loop)
- [Evaluation](#evaluation)
- [Results & Metrics](#results--metrics)

---

## Research Goal

To determine whether a neural network policy can learn optimal responses to life events in a long-term relationship — maximizing **happiness** and **relationship stability** across a simulated lifespan — given two agents with distinct internal trait profiles and asymmetric reward priorities.

---

## System Design

### Internal Traits (X)

Each agent is initialized with their own independent X vector, sampled uniformly from `[0.05, 0.95]`. These evolve only during **Reflection**.

| Variable | Description |
|---|---|
| `iq` | Cognitive intelligence |
| `eq` | Emotional intelligence — reduces self-observation noise |
| `rational_thinking` | Tendency toward logic-based decisions |
| `emotional_reasoning` | Tendency toward feeling-based decisions |
| `kindness` | Baseline prosocial disposition |
| `ability_to_love` | Capacity for deep attachment |
| `faithfulness` | Commitment to the relationship |
| `responsibility` | Follow-through on obligations |
| `mental_stability` | Resilience to stress and volatility |
| `kids` | Presence/number of children (shared, affects both agents) |

**Observation noise** is trait-dependent: agents with higher EQ perceive their own traits more accurately (`noise_std = obs_noise_scale × (1 − eq)`). Partner traits are estimated with additional noise scaled by relationship trust (`partner_noise_std = partner_obs_noise_scale × (1 − trust)`).

### Environmental State (Y)

Y variables are shared between both agents and evolve in response to events and actions.

| Variable | Description |
|---|---|
| `wealth` | Financial status of the couple |
| `love_support` | Perceived mutual affection and support |
| `pressure` | External stress load (work, family, health, etc.) |
| `happiness` | Primary reward signal |
| `stability` | Secondary reward signal; measures relationship continuity |

**Reward weights are asymmetric by design** — husband and wife have different life priorities, creating genuine multi-agent tension with no single dominant strategy:

| Agent | Happiness | Stability | Wealth |
|---|---|---|---|
| Husband | 0.5 | 0.3 | 0.2 |
| Wife | 0.6 | 0.4 | 0.0 |

### Events

Random life events are sampled each simulation step (one per "year"). Each event targets one or more Y variables with a baseline delta. Both agents independently select an **action** (support, argue, ignore, compromise, withdraw); the combined response determines the actual ΔY applied.

| Event | Affected Y | Notes |
|---|---|---|
| Job loss | wealth, pressure | Financial shock + stress |
| New child | kids, pressure, love_support | Documented personality shift |
| Infidelity | faithfulness, stability, love_support | Identity-level disruption |
| Health crisis | wealth, pressure, love_support | Confronts mortality |
| Financial windfall | wealth, pressure | Stress relief |
| Emotional conflict | love_support, stability | Relationship friction |
| Shared achievement | happiness, love_support | Bonding event |
| Mental health episode | pressure, stability | Psychological rupture |
| Relocation | multiple | Identity and social disruption |

**Social support context:** Each episode samples a couple's external support network strength. Isolated couples (low `social_support`) take amplified hits from negative events: `effective_impact = 1 + 0.30 × (1 − social_support)`.

### Reflection Mechanism

Reflection triggers when `|ΔY| > 0.15` on any single Y variable. It nudges an agent's learned traits based on the outcome, mimicking real psychological growth — with personality-typed effects:

- **Happiness ↑**: analytical agents grow `rational_thinking`; emotional agents grow `eq`; rational agents gain `eq` slowly (they can learn emotional skills through analysis)
- **Stability ↑**: reinforces `mental_stability` and `responsibility` for already-responsible agents
- **Pressure ↑**: unstable agents take a larger `mental_stability` hit
- **Love/support ↑**: high-EQ agents absorb more growth in `eq` and `ability_to_love`
- **Wealth ↑**: rational agents extract strategic insight into `iq` and `rational_thinking`

Major life events trigger larger reflection magnitudes:

| Event | Max trait Δ |
|---|---|
| Infidelity | 0.20 |
| Family death | 0.15 |
| Health/mental health crisis | 0.12 |
| New child | 0.10 |
| Relocation | 0.08 |
| Financial crisis | 0.07 |
| Minor events (default) | 0.05 |

### Learning: PPO with Value Baseline

Each agent runs an independent policy:

```
action = π_θ(state)
state  = [X_self, X_partner, Y_shared, event_encoding]
```

Update rule uses **PPO-clip** with **GAE** (Generalized Advantage Estimation):

```
L_CLIP = E[ min(r_t · A_t,  clip(r_t, 1−ε, 1+ε) · A_t) ]
A_t    = GAE(rewards, values, γ=0.99, λ=0.95)
```

Entropy is annealed from `0.05 → 0.005` over training to encourage early exploration and late convergence. Trajectories from `update_every=4` episodes are accumulated before each PPO update.

---

## Personality Archetypes

Four fixed trait profiles used during evaluation to verify that different personalities produce different action distributions under the same trained policy:

| Archetype | Description | Key traits |
|---|---|---|
| **Secure** | Emotionally intelligent, stable, loving | High EQ, stability, faithfulness, kindness |
| **Emotional** | Anxious attachment, emotionally intense | High EQ + love, low mental stability |
| **Rational** | Dismissive attachment, logic-driven | High IQ + rational_thinking, low EQ |
| **Avoidant** | Fearful attachment, distrusts intimacy | Low EQ, love, stability, faithfulness |

---

## Repo Structure

```
ai-romantic-relationship-simulator/
│
├── README.md
├── requirements.txt
├── train.py                  # Entry point: PPO training loop
│
├── config/
│   ├── default.yaml          # Hyperparameters, thresholds, reward weights
│   └── events.yaml           # Event catalog with base deltas and probabilities
│
├── src/
│   ├── env/
│   │   ├── marriage_env.py   # Gymnasium-compatible environment
│   │   ├── events.py         # Event sampling and delta application
│   │   └── state.py          # State representation, normalization, trait names
│   │
│   └── agents/
│       └── agent.py          # Agent class (PolicyNet + ValueNet, PPO update)
│
├── data/
│   ├── checkpoints/          # Model weight snapshots (.pt)
│   ├── metrics.json          # Per-episode training metrics
│   ├── eval_history.json     # Per-category evaluation scores over training
│   └── archetypes_ep*.png    # Archetype action distribution snapshots
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
- PyTorch 2.0+
- Gymnasium 0.29+
- NumPy, PyYAML, Matplotlib

### Install

```bash
git clone https://github.com/your-username/ai-romantic-relationship-simulator.git
cd ai-romantic-relationship-simulator
pip install -r requirements.txt
```

### Train

```bash
python train.py                          # full run (1M episodes, defaults)
python train.py --episodes 5000          # quick smoke test
python train.py --resume                 # continue from latest checkpoint
python train.py --plot                   # show training curves after run
```

### Key CLI flags

| Flag | Default | Description |
|---|---|---|
| `--config` | `config/default.yaml` | Hyperparameter config |
| `--events` | `config/events.yaml` | Event catalog |
| `--checkpoint` | `data/checkpoints/latest.pt` | Checkpoint path |
| `--resume` | false | Resume from checkpoint |
| `--episodes` | (from config) | Override episode count |
| `--lr` | (from config) | Override learning rate |
| `--save-every` | 500 | Checkpoint interval |
| `--eval-every` | 1000 | Category evaluation interval |
| `--plot` | false | Plot curves after training |

---

## Training Loop

```
for episode in range(num_episodes):
    reset marriage (age=25, sample X for each agent, reset Y, sample social_support)

    for age in range(25, 80):
        sample event from catalog
        state = [X_self + noise(eq), X_partner + noise(trust), Y_shared, event]

        action_h, log_prob_h, value_h = π_θ_h(state)
        action_w, log_prob_w, value_w = π_θ_w(state)

        ΔY = compute_delta_Y(event, action_h, action_w, social_support)
        Y  = update_Y(Y, ΔY)

        if |ΔY| > reflection_threshold:
            reflect(x_h, ΔY, magnitude)   # nudge traits based on personality type
            reflect(x_w, ΔY, magnitude)

        reward_h = happiness_weight·Y.happiness + stability_weight·Y.stability + wealth_weight·Y.wealth
        store (state, action, log_prob, reward, value) in trajectory buffer

    # Every update_every episodes:
    update θ_h, θ_w via PPO-clip on accumulated trajectories
    anneal entropy_coef toward entropy_end
```

---

## Evaluation

Category evaluation runs in a **background process** every `eval-every` episodes, so it never blocks training. For each event category (Financial, Family, Health, Relationship, Life Change), 200 episodes are run with only that category's events active.

Archetype evaluation injects each of the four fixed trait profiles into both agents across 30 runs and records the resulting action distributions — producing a behavioral fingerprint of the learned policy.

Snapshots are saved to `data/archetypes_ep{N}.png` and logged as `data/eval_history.json`.

---

## Results & Metrics

Each episode logs:

- `reward_h_mean`, `reward_w_mean` — per-step reward for each agent
- `mean_happiness`, `mean_stability` — average Y state over the episode
- `final_happiness`, `final_stability` — Y state at age 80
- `reflections` — number of reflection events triggered
- `loss_h_policy`, `loss_w_policy` — PPO policy loss

### Trained vs. Random Policy (Ablation)

The trained policy consistently outperforms random action selection across all archetypes. The largest gains appear in stability (secure +14%, rational +10%) and divorce rate reduction (emotional −38%, rational −26%, avoidant −26%).

![Ablation: trained vs random policy](data/0_ablation_trained_vs_random.png)

### Outcomes by Personality Archetype

Secure couples achieve the best outcomes across all metrics (89% happiness, 65% stability, 4% divorce rate). Avoidant couples have the worst (50% happiness, 2% stability, 60% divorce rate) — even with a trained policy, attachment style dominates long-run outcomes.

![Outcomes by archetype](data/2_outcomes_by_archetype.png)

### Action Distributions by Personality Type

All four archetypes show similar action frequencies, clustered near the random baseline (20% dashed line). The trained policy learns a broadly similar strategy — the key differentiation comes from *which events trigger which actions*, not the marginal frequency.

![Action distributions by personality type](data/1_action_distributions.png)

### Partner Mismatch — All 16 Archetype Combinations

Heatmaps across all 16 husband × wife archetype pairings. Secure partners consistently lift outcomes regardless of who they're paired with. Avoidant × avoidant is the worst combination at every metric. Notably, a secure husband can partially compensate for an avoidant wife but not vice versa.

![Partner mismatch heatmaps](data/3_partner_mismatch_heatmaps.png)

### Warmth vs. Stability Trade-off

Scatter of final love & support vs. final stability for all 16 pairings. Secure-anchored pairs (top-right) achieve both; emotional and avoidant pairs cluster in the low-stability region regardless of warmth.

![Warmth vs stability trade-off](data/4_warmth_vs_stability.png)

### Y-State Trajectories Over the Lifespan

Mean trajectories (ages 25–80) for surviving couples across archetype pairs. Happiness and love & support decline gradually for most pairs; pressure rises mid-life. Trust grows slowly while resentment accumulates steadily — even in successful marriages.

![Y-state trajectories](data/5_trajectories.png)

### Event Mix Across the Lifespan

Stacked area of event category frequency over age for each archetype. All archetypes face a similar event distribution — the divergence in outcomes is driven by policy response, not event exposure.

![Life stage event profile](data/6_life_stage_event_profile.png)

---

## License

MIT
