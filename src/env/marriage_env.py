import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from typing import Optional

from src.env.state import XTraits, YState, X_DIM, Y_DIM
from src.env.events import EventCatalog, N_ACTIONS


class MarriageEnv(gym.Env):
    """
    Gymnasium environment: a marriage simulated from age 25 to 80.

    Observation vector (flat, all values in [0, 1]):
        [ X_self (10) | X_partner (10) | Y_shared (5) | event_one_hot (n_events+1) ]

    Each agent gets their own observation: X_self comes first so the same
    policy architecture works for both husband and wife. Observations include
    perceptual noise scaled by (1 - eq): lower EQ → noisier read of the world.

    Action:
        MultiDiscrete([N_ACTIONS, N_ACTIONS])
        actions[0] = husband's response, actions[1] = wife's response

    Reward:
        Each agent has their own weighted reward over Y variables (see config).
        step() returns the average as the Gymnasium scalar reward; per-agent
        rewards are available in info["reward_h"] and info["reward_w"] for
        the trainer to use separately.

    Episode:
        One episode = one full marriage (age 25 → 80, i.e. 55 steps).
        reset() samples fresh X traits for both agents and resets Y to defaults.
    """

    metadata = {"render_modes": []}

    def __init__(self, config_path: str, events_path: str):
        super().__init__()

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.age_start: int = cfg["simulation"]["age_start"]
        self.age_end: int   = cfg["simulation"]["age_end"]
        self._x_init_low:  float = cfg["agents"]["x_init_low"]
        self._x_init_high: float = cfg["agents"]["x_init_high"]
        self._obs_noise_scale: float = cfg["agents"]["obs_noise_scale"]
        self._reflection_threshold: float = cfg["reflection"]["threshold"]

        # Per-agent reward weights — different objectives create natural tension
        rwd = cfg["reward"]
        self._happiness_w_h: float = rwd["agent_h"]["happiness_weight"]
        self._stability_w_h: float = rwd["agent_h"]["stability_weight"]
        self._wealth_w_h:    float = rwd["agent_h"]["wealth_weight"]
        self._happiness_w_w: float = rwd["agent_w"]["happiness_weight"]
        self._stability_w_w: float = rwd["agent_w"]["stability_weight"]
        self._wealth_w_w:    float = rwd["agent_w"]["wealth_weight"]

        self.events = EventCatalog(events_path)

        obs_dim = X_DIM + X_DIM + Y_DIM + (self.events.n_events + 1)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([N_ACTIONS, N_ACTIONS])

        # Episode state — populated by reset()
        self.x_h: Optional[XTraits] = None
        self.x_w: Optional[XTraits] = None
        self.y:   Optional[YState]  = None
        self.age: int = self.age_start
        self.current_event: Optional[dict] = None

    # ── Core Gymnasium interface ───────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.x_h = self._sample_x()
        self.x_w = self._sample_x()
        self.y   = YState()
        self.age = self.age_start
        self.current_event = self.events.sample(self.x_h, self.x_w)

        obs_h = self._get_obs("h")
        obs_w = self._get_obs("w")
        return obs_h, {"obs_w": obs_w}

    def step(self, actions):
        action_h = int(actions[0])
        action_w = int(actions[1])

        # Apply event + agent responses → ΔY
        delta = self.events.compute_delta_y(
            self.current_event, action_h, action_w, self.x_h, self.x_w
        )
        self.y.apply_delta(delta)

        # Check if reflection should be triggered for either agent
        max_abs_delta = max(abs(v) for v in delta.values()) if delta else 0.0
        reflection_triggered = bool(max_abs_delta > self._reflection_threshold)

        # Handle events with direct X side-effects (e.g. new_child increments kids)
        if self.current_event and self.current_event.get("special") == "increment_kids":
            self.x_h.increment_kids()
            self.x_w.increment_kids()

        reward_h, reward_w = self._compute_rewards()
        reward = (reward_h + reward_w) / 2.0  # Gymnasium scalar; trainer uses per-agent

        self.age += 1
        done = self.age >= self.age_end

        self.current_event = self.events.sample(self.x_h, self.x_w)

        obs_h = self._get_obs("h")
        obs_w = self._get_obs("w")

        info = {
            "age": self.age,
            "event": self.current_event["name"] if self.current_event else "none",
            "delta_y": delta,
            "reflection_triggered": reflection_triggered,
            "happiness": self.y.happiness,
            "stability": self.y.stability,
            "y_state": self.y.to_array().tolist(),
            "reward_h": reward_h,
            "reward_w": reward_w,
            "obs_h": obs_h,
            "obs_w": obs_w,
        }

        return obs_h, reward, done, False, info

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_obs(self, agent: str = "h") -> np.ndarray:
        """
        Build observation vector from the given agent's perspective.

        X_self comes first so both agents can share the same policy architecture.
        Gaussian noise is added, scaled by (1 - eq): an agent with low EQ gets
        a noisier, less accurate read of themselves, their partner, and the world.
        This models bounded rationality without corrupting the action signal.
        """
        x_self  = self.x_h if agent == "h" else self.x_w
        x_other = self.x_w if agent == "h" else self.x_h

        raw = np.concatenate([
            x_self.to_array(),
            x_other.to_array(),
            self.y.to_array(),
            self.events.one_hot(self.current_event),
        ]).astype(np.float32)

        # Scale noise by (1 - effective_eq): lower EQ → more perceptual noise
        noise_std = self._obs_noise_scale * (1.0 - x_self.effective("eq"))
        if noise_std > 0.0:
            noise = self.np_random.normal(0.0, noise_std, size=raw.shape).astype(np.float32)
            raw = np.clip(raw + noise, 0.0, 1.0)

        return raw

    def _compute_rewards(self) -> tuple[float, float]:
        """
        Per-agent rewards over the shared Y state.

        Different weights encode different priorities (e.g. one partner values
        financial security more; the other values emotional closeness more).
        This creates genuine multi-agent tension with no single global optimum.
        """
        r_h = float(
            self._happiness_w_h * self.y.happiness
            + self._stability_w_h * self.y.stability
            + self._wealth_w_h    * self.y.wealth
        )
        r_w = float(
            self._happiness_w_w * self.y.happiness
            + self._stability_w_w * self.y.stability
            + self._wealth_w_w    * self.y.wealth
        )
        return r_h, r_w

    def _sample_x(self) -> XTraits:
        """Sample one agent's X traits uniformly from [x_init_low, x_init_high]."""
        lo, hi = self._x_init_low, self._x_init_high
        v = np.random.uniform(lo, hi, size=X_DIM)
        return XTraits(
            iq=v[0],
            eq=v[1],
            rational_thinking=v[2],
            emotional_reasoning=v[3],
            kindness=v[4],
            ability_to_love=v[5],
            faithfulness=v[6],
            responsibility=v[7],
            mental_stability=v[8],
            kids=0.0,
        )
