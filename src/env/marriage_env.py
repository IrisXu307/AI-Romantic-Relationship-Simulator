import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from typing import Optional

from src.env.state import XTraits, YState, X_DIM, Y_DIM
from src.env.events import EventCatalog, N_ACTIONS

# Consecutive steps both partners must be in relationship distress before
# the episode terminates as a divorce.
_DIVORCE_STREAK = 3
_DIVORCE_LOVE_THRESHOLD = 0.15
_DIVORCE_RESENTMENT_THRESHOLD = 0.75


class MarriageEnv(gym.Env):
    """
    Gymnasium environment: a marriage simulated from age 25 to 80.

    Phase-1 changes vs original design:
      - Each partner has their own subjective YState (y_h, y_w) instead of
        a single shared state. Actions affect the *receiving* partner's Y.
      - YState now includes trust and resentment as slow-accumulating history
        signals, giving the policy memory of past behaviour.
      - Divorce terminates the episode early if either partner sustains very
        low love_support AND high resentment for _DIVORCE_STREAK consecutive steps.
      - Reward weights shift with age: wealth matters more in early marriage,
        happiness and connection matter more in later life.

    Observation vector (flat, all values in [0, 1]):
        [ X_self (10) | X_partner (10) | Y_own (7) | event_one_hot (n_events+1) ]

    Action:
        MultiDiscrete([N_ACTIONS, N_ACTIONS])
        actions[0] = husband's response, actions[1] = wife's response

    Reward:
        Per-agent reward over own YState with age-dependent weights.
        step() returns the average as the Gymnasium scalar; per-agent rewards
        are in info["reward_h"] and info["reward_w"].
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

        # Base reward weights — scaled by age inside _compute_rewards()
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
        self.y_h: Optional[YState]  = None   # husband's subjective relationship state
        self.y_w: Optional[YState]  = None   # wife's subjective relationship state
        self.age: int = self.age_start
        self.current_event: Optional[dict] = None
        self._distress_streak: int = 0       # consecutive steps in divorce-risk territory

    # ── Core Gymnasium interface ───────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):  # noqa: ARG002
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.x_h = self._sample_x()
        self.x_w = self._sample_x()
        self.y_h = YState()
        self.y_w = YState()
        self.age = self.age_start
        self._distress_streak = 0
        self.current_event = self.events.sample(self.x_h, self.x_w)

        obs_h = self._get_obs("h")
        obs_w = self._get_obs("w")
        return obs_h, {"obs_w": obs_w}

    def step(self, actions):
        action_h = int(actions[0])
        action_w = int(actions[1])

        # Compute per-partner ΔY (event + actions + trust/resentment)
        delta_h, delta_w = self.events.compute_delta_y(
            self.current_event, action_h, action_w, self.x_h, self.x_w
        )
        self.y_h.apply_delta(delta_h)
        self.y_w.apply_delta(delta_w)

        # Reflection: triggered when either partner experiences a large Y shift
        all_deltas = list(delta_h.values()) + list(delta_w.values())
        max_abs_delta = max(abs(v) for v in all_deltas) if all_deltas else 0.0
        reflection_triggered = bool(max_abs_delta > self._reflection_threshold)

        # Handle events with direct X side-effects
        if self.current_event and self.current_event.get("special") == "increment_kids":
            self.x_h.increment_kids()
            self.x_w.increment_kids()

        reward_h, reward_w = self._compute_rewards()
        reward = (reward_h + reward_w) / 2.0

        self.age += 1
        done = self.age >= self.age_end

        # Divorce check: sustained distress in either partner ends the marriage
        h_in_distress = (
            self.y_h.love_support < _DIVORCE_LOVE_THRESHOLD
            and self.y_h.resentment > _DIVORCE_RESENTMENT_THRESHOLD
        )
        w_in_distress = (
            self.y_w.love_support < _DIVORCE_LOVE_THRESHOLD
            and self.y_w.resentment > _DIVORCE_RESENTMENT_THRESHOLD
        )
        if h_in_distress or w_in_distress:
            self._distress_streak += 1
        else:
            self._distress_streak = 0

        if self._distress_streak >= _DIVORCE_STREAK:
            done = True
            reward_h = reward_w = 0.0   # terminal penalty via missed future rewards
            reward = 0.0

        self.current_event = self.events.sample(self.x_h, self.x_w)

        obs_h = self._get_obs("h")
        obs_w = self._get_obs("w")

        info = {
            "age":                  self.age,
            "event":                self.current_event["name"] if self.current_event else "none",
            "delta_h":              delta_h,
            "delta_w":              delta_w,
            "reflection_triggered": reflection_triggered,
            "happiness":            (self.y_h.happiness + self.y_w.happiness) / 2.0,
            "stability":            (self.y_h.stability + self.y_w.stability) / 2.0,
            "y_state_h":            self.y_h.to_array().tolist(),
            "y_state_w":            self.y_w.to_array().tolist(),
            "divorced":             self._distress_streak >= _DIVORCE_STREAK,
            "reward_h":             reward_h,
            "reward_w":             reward_w,
            "obs_h":                obs_h,
            "obs_w":                obs_w,
        }

        return obs_h, reward, done, False, info

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_obs(self, agent: str = "h") -> np.ndarray:
        """
        Build observation vector from the given agent's perspective.

        X_self comes first so both agents can share the same policy architecture.
        Y_own is each partner's *subjective* state — they observe their own
        experience of the relationship, not a shared average.
        Gaussian noise scaled by (1 - eq) models bounded rationality.
        """
        x_self  = self.x_h if agent == "h" else self.x_w
        x_other = self.x_w if agent == "h" else self.x_h
        y_own   = self.y_h if agent == "h" else self.y_w

        raw = np.concatenate([
            x_self.to_array(),
            x_other.to_array(),
            y_own.to_array(),
            self.events.one_hot(self.current_event),
        ]).astype(np.float32)

        noise_std = self._obs_noise_scale * (1.0 - x_self.effective("eq"))
        if noise_std > 0.0:
            noise = self.np_random.normal(0.0, noise_std, size=raw.shape).astype(np.float32)
            raw = np.clip(raw + noise, 0.0, 1.0)

        return raw

    def _compute_rewards(self) -> tuple[float, float]:
        """
        Per-agent rewards from each partner's own subjective YState.

        Reward weights shift with life stage:
          - Wealth matters more in early marriage (financial pressure, career building)
          - Happiness and connection matter more in later life
        Trust contributes positively, resentment negatively — encoding the
        cumulative quality of the relationship into the reward signal.
        """
        age_frac = (self.age - self.age_start) / (self.age_end - self.age_start)
        happiness_scale = 0.7 + 0.6 * age_frac    # grows 0.7 → 1.3 over the marriage
        wealth_scale    = 1.3 - 0.6 * age_frac    # shrinks 1.3 → 0.7

        def _reward(y: YState, hw: float, sw: float, ww: float) -> float:
            hw_age = hw * happiness_scale
            ww_age = ww * wealth_scale
            raw = (
                hw_age * y.happiness
                + sw    * y.stability
                + ww_age * y.wealth
                + 0.15  * y.love_support
                + 0.10  * y.trust
                - 0.10  * y.resentment
            )
            # Normalise by sum of positive weights so reward stays in [0, 1]
            max_possible = hw_age + sw + ww_age + 0.15 + 0.10
            return float(np.clip(raw / max_possible, 0.0, 1.0))

        r_h = _reward(self.y_h, self._happiness_w_h, self._stability_w_h, self._wealth_w_h)
        r_w = _reward(self.y_w, self._happiness_w_w, self._stability_w_w, self._wealth_w_w)
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
