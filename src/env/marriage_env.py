import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from typing import Optional

from src.env.state import XTraits, YState, X_DIM, Y_DIM
from src.env.events import (
    EventCatalog, N_ACTIONS,
    ACTION_SUPPORT, ACTION_ARGUE, ACTION_IGNORE, ACTION_COMPROMISE, ACTION_WITHDRAW,
)

# How much social_support shifts after specific life events.
# Negative = isolation (relocation breaks community ties, grief shrinks social world).
# Positive = expansion (shared success and intimacy tend to open social circles).
_SOCIAL_SUPPORT_SHIFTS: dict[str, float] = {
    "relocation":         -0.10,
    "family_death":       -0.08,
    "new_child":          -0.05,
    "shared_achievement": +0.05,
    "romantic_gesture":   +0.02,
}

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
        self._partner_obs_noise_scale: float = cfg["agents"]["partner_obs_noise_scale"]
        self._reflection_threshold: float = cfg["reflection"]["threshold"]

        ss = cfg["social_support"]
        self._social_init_low:       float = ss["init_low"]
        self._social_init_high:      float = ss["init_high"]
        self._isolation_amplifier:   float = ss["isolation_amplifier"]

        # Base reward weights — scaled by age inside _compute_rewards()
        rwd = cfg["reward"]
        self._happiness_w_h: float = rwd["agent_h"]["happiness_weight"]
        self._stability_w_h: float = rwd["agent_h"]["stability_weight"]
        self._wealth_w_h:    float = rwd["agent_h"]["wealth_weight"]
        self._happiness_w_w: float = rwd["agent_w"]["happiness_weight"]
        self._stability_w_w: float = rwd["agent_w"]["stability_weight"]
        self._wealth_w_w:    float = rwd["agent_w"]["wealth_weight"]

        self.events = EventCatalog(events_path)

        # +1 for social_support, +1 for life_stage_frac appended to every observation
        obs_dim = X_DIM + X_DIM + Y_DIM + (self.events.n_events + 1) + 1 + 1
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
        self._event_counts: dict[str, int] = {}  # how many times each event has fired this episode
        self.social_support: float = 0.6    # couple's external support network strength
        self._last_action_h: int = ACTION_SUPPORT
        self._last_action_w: int = ACTION_SUPPORT

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
        self._event_counts = {}
        self.social_support = float(np.random.uniform(self._social_init_low, self._social_init_high))
        self._last_action_h = ACTION_SUPPORT
        self._last_action_w = ACTION_SUPPORT
        self.current_event = self.events.sample(self.x_h, self.x_w, age=self.age)

        obs_h = self._get_obs("h")
        obs_w = self._get_obs("w")
        return obs_h, {"obs_w": obs_w}

    def step(self, actions):
        action_h = int(actions[0])
        action_w = int(actions[1])
        self._last_action_h = action_h
        self._last_action_w = action_w

        # Event habituation: repeated events have diminishing base impact.
        # Negative events: 0.85^count floor 0.35 (people numb to chronic stress).
        # Positive events: 0.90^count floor 0.40 (novelty fades over time).
        habituation = 1.0
        if self.current_event:
            name = self.current_event["name"]
            count = self._event_counts.get(name, 0)
            base_sum = sum(self.current_event["base_delta_y"].values())
            rate = 0.85 if base_sum < 0 else 0.90
            floor = 0.35 if base_sum < 0 else 0.40
            habituation = max(rate ** count, floor)
            self._event_counts[name] = count + 1

        # Social isolation amplifier: couples with weak external support networks
        # experience negative events more acutely — no one to lean on outside the marriage.
        # At social_support=1.0: no amplification. At 0.0: base impact × (1 + isolation_amplifier).
        social_scale = 1.0
        if self.current_event:
            base_sum = sum(self.current_event["base_delta_y"].values())
            if base_sum < 0:
                social_scale = 1.0 + self._isolation_amplifier * (1.0 - self.social_support)

        # Update social_support: slow mean-reversion toward 0.6, plus event-specific shifts.
        # Applied before delta computation so this step's social_support state is observed.
        event_name_now = self.current_event["name"] if self.current_event else "none"
        social_shift = _SOCIAL_SUPPORT_SHIFTS.get(event_name_now, 0.0)
        self.social_support = float(np.clip(
            self.social_support + 0.005 * (0.6 - self.social_support) + social_shift,
            0.0, 1.0,
        ))

        # Compute per-partner ΔY (event + actions + trust/resentment).
        # base_scale = habituation × social_scale: both operate on the base event component.
        delta_h, delta_w = self.events.compute_delta_y(
            self.current_event, action_h, action_w, self.x_h, self.x_w,
            habituation=habituation * social_scale,
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

        self.current_event = self.events.sample(self.x_h, self.x_w, age=self.age)

        obs_h = self._get_obs("h")
        obs_w = self._get_obs("w")

        life_stage_frac = (self.age - self.age_start) / (self.age_end - self.age_start)

        info = {
            "age":                  self.age,
            "life_stage":           float(life_stage_frac),
            "social_support":       self.social_support,
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

        # Self-perception noise: higher EQ → more accurate self-read.
        # Applied only to x_self — the agent's introspection of their own traits.
        x_self_arr = x_self.to_array().copy()
        self_noise_std = self._obs_noise_scale * (1.0 - x_self.effective("eq"))
        if self_noise_std > 0.0:
            x_self_arr = np.clip(
                x_self_arr + self.np_random.normal(
                    0.0, self_noise_std, size=x_self_arr.shape
                ).astype(np.float32),
                0.0, 1.0,
            )

        # Partner model noise: trust determines how accurately the agent reads their
        # partner. noise_std = partner_obs_noise_scale × (1 − trust).
        # At trust=1.0 the partner model is exact; at trust=0.0 it is maximally noisy.
        x_other_arr = x_other.to_array().copy()
        partner_noise_std = self._partner_obs_noise_scale * (1.0 - y_own.trust)
        if partner_noise_std > 0.0:
            x_other_arr = np.clip(
                x_other_arr + self.np_random.normal(
                    0.0, partner_noise_std, size=x_other_arr.shape
                ).astype(np.float32),
                0.0, 1.0,
            )

        # Y state, event, and life_stage are objective observations — no noise.
        life_stage_frac = np.float32(
            (self.age - self.age_start) / (self.age_end - self.age_start)
        )

        return np.concatenate([
            x_self_arr,
            x_other_arr,
            y_own.to_array(),
            self.events.one_hot(self.current_event),
            [np.float32(self.social_support)],
            [life_stage_frac],
        ]).astype(np.float32)

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

        def _consistency_bonus(x: XTraits, action: int) -> float:
            """Small reward for acting in character — reinforces personality differentiation.
            Caring/empathetic agents flourish by supporting; rational agents by compromising;
            volatile agents get slight relief from arguing. Max bonus is 0.05."""
            kindness    = x.effective("kindness")
            eq          = x.effective("eq")
            rt          = x.effective("rational_thinking")
            ms          = x.effective("mental_stability")
            if action == ACTION_SUPPORT:
                return 0.05 * kindness * eq
            if action == ACTION_COMPROMISE:
                return 0.05 * rt * x.effective("responsibility")
            if action == ACTION_ARGUE:
                # hot-headed low-stability agents get tiny cathartic relief
                volatility = max(0.0, 0.5 - ms) * 2.0
                return 0.02 * volatility
            return 0.0

        bonus_h = _consistency_bonus(self.x_h, self._last_action_h)
        bonus_w = _consistency_bonus(self.x_w, self._last_action_w)

        r_h = min(1.0, _reward(self.y_h, self._happiness_w_h, self._stability_w_h, self._wealth_w_h) + bonus_h)
        r_w = min(1.0, _reward(self.y_w, self._happiness_w_w, self._stability_w_w, self._wealth_w_w) + bonus_w)
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
