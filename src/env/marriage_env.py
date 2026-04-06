import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from dataclasses import fields
from typing import Optional

from src.env.state import XTraits, YState, X_DIM, Y_DIM
from src.env.events import EventCatalog, N_ACTIONS


class MarriageEnv(gym.Env):
    """
    Gymnasium environment: a marriage simulated from age 25 to 80.

    Observation vector (flat, all values in [0, 1]):
        [ X_husband (10) | X_wife (10) | Y_shared (5) | event_one_hot (n_events+1) ]

    Action:
        MultiDiscrete([N_ACTIONS, N_ACTIONS])
        actions[0] = husband's response, actions[1] = wife's response

    Reward:
        happiness_weight * Y.happiness + stability_weight * Y.stability
        Computed on the Y state *after* applying the timestep's delta.

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
        self.age_end: int = cfg["simulation"]["age_end"]
        self._x_init_low: float = cfg["agents"]["x_init_low"]
        self._x_init_high: float = cfg["agents"]["x_init_high"]
        self._reflection_threshold: float = cfg["reflection"]["threshold"]
        self._happiness_w: float = cfg["reward"]["happiness_weight"]
        self._stability_w: float = cfg["reward"]["stability_weight"]

        self.events = EventCatalog(events_path)

        obs_dim = X_DIM + X_DIM + Y_DIM + (self.events.n_events + 1)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([N_ACTIONS, N_ACTIONS])

        # Episode state — populated by reset()
        self.x_h: Optional[XTraits] = None
        self.x_w: Optional[XTraits] = None
        self.y: Optional[YState] = None
        self.age: int = self.age_start
        self.current_event: Optional[dict] = None

    # ── Core Gymnasium interface ───────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.x_h = self._sample_x()
        self.x_w = self._sample_x()
        self.y = YState()
        self.age = self.age_start
        self.current_event = self.events.sample()

        return self._get_obs(), {}

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

        reward = self._compute_reward()

        self.age += 1
        done = self.age >= self.age_end

        self.current_event = self.events.sample()

        info = {
            "age": self.age,
            "event": self.current_event["name"] if self.current_event else "none",
            "delta_y": delta,
            "reflection_triggered": reflection_triggered,
            "happiness": self.y.happiness,
            "stability": self.y.stability,
            "y_state": self.y.to_array().tolist(),
        }

        return self._get_obs(), reward, done, False, info

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([
            self.x_h.to_array(),
            self.x_w.to_array(),
            self.y.to_array(),
            self.events.one_hot(self.current_event),
        ]).astype(np.float32)

    def _compute_reward(self) -> float:
        return float(
            self._happiness_w * self.y.happiness
            + self._stability_w * self.y.stability
        )

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
