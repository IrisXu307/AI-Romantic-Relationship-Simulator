import yaml
import numpy as np
from typing import Optional

from src.env.state import XTraits

# ── Action space ──────────────────────────────────────────────────────────────

ACTION_SUPPORT    = 0  # try to help, comfort, or encourage
ACTION_ARGUE      = 1  # escalate conflict, blame, push back
ACTION_IGNORE     = 2  # disengage, avoid dealing with the situation
ACTION_COMPROMISE = 3  # negotiate, meet in the middle
ACTION_WITHDRAW   = 4  # pull back emotionally without fighting

N_ACTIONS = 5
ACTION_NAMES = ["support", "argue", "ignore", "compromise", "withdraw"]

# ── How each action shifts Y (base modifier before X scaling) ─────────────────
# Each action is applied per-agent and averaged, so the total effect per agent
# is halved to avoid double-counting.

_ACTION_BASE: dict[int, dict[str, float]] = {
    ACTION_SUPPORT:    {"love_support":  0.15, "pressure": -0.10, "stability":  0.08, "happiness":  0.05},
    ACTION_ARGUE:      {"love_support": -0.25, "pressure":  0.20, "stability": -0.15, "happiness": -0.10},
    ACTION_IGNORE:     {"love_support": -0.10, "pressure":  0.05, "stability": -0.08, "happiness": -0.05},
    ACTION_COMPROMISE: {"love_support":  0.10, "pressure": -0.12, "stability":  0.10, "happiness":  0.08},
    ACTION_WITHDRAW:   {"love_support": -0.12, "pressure": -0.05, "stability": -0.10, "happiness": -0.03},
}


def x_scale_factor(action: int, x: XTraits) -> float:
    """
    Return a multiplier in [0.5, 1.5] that scales an action's effect
    based on the agent's relevant X traits.

    Higher relevant traits → more effective positive actions,
    more damaging negative ones (e.g. a low-stability agent's arguing hurts more).
    """
    if action == ACTION_SUPPORT:
        # Effectiveness driven by emotional capacity
        return 0.5 + (x.eq + x.kindness + x.ability_to_love) / 3.0
    elif action == ACTION_ARGUE:
        # Damage driven by poor emotional regulation
        return 0.5 + (1.0 - x.mental_stability)
    elif action == ACTION_IGNORE:
        return 1.0
    elif action == ACTION_COMPROMISE:
        # Effectiveness driven by rationality and empathy
        return 0.5 + (x.rational_thinking + x.eq) / 2.0
    elif action == ACTION_WITHDRAW:
        return 1.0
    return 1.0


# ── Event catalog ─────────────────────────────────────────────────────────────

class EventCatalog:
    """Loads events from YAML and provides sampling + ΔY computation."""

    def __init__(self, events_path: str):
        with open(events_path, "r") as f:
            data = yaml.safe_load(f)
        self.events: list[dict] = data["events"]
        self.names: list[str] = [e["name"] for e in self.events]
        self.probs: np.ndarray = np.array([e["probability"] for e in self.events], dtype=np.float64)
        self.n_events: int = len(self.events)

    def sample(self) -> Optional[dict]:
        """
        Sample one event or None (no major event this year).
        Remaining probability mass after all events = chance of a quiet year.
        """
        roll = np.random.random()
        cumulative = 0.0
        for event, p in zip(self.events, self.probs):
            cumulative += p
            if roll < cumulative:
                return event
        return None

    def event_index(self, event: Optional[dict]) -> int:
        """Return index of event in catalog, or n_events for 'no event'."""
        if event is None:
            return self.n_events
        return self.names.index(event["name"])

    def one_hot(self, event: Optional[dict]) -> np.ndarray:
        """One-hot vector of length n_events+1 (last slot = no event)."""
        vec = np.zeros(self.n_events + 1, dtype=np.float32)
        vec[self.event_index(event)] = 1.0
        return vec

    def compute_delta_y(
        self,
        event: Optional[dict],
        action_h: int,
        action_w: int,
        x_h: XTraits,
        x_w: XTraits,
    ) -> dict[str, float]:
        """
        Compute total ΔY for one timestep.

        Order of contributions:
          1. Base event delta (from YAML)
          2. Husband's action modifier (scaled by his X traits)
          3. Wife's action modifier (scaled by her X traits)

        Each agent's action contribution is halved so two agents together
        don't double the impact of their responses.
        """
        delta: dict[str, float] = {}

        # 1. Base event effect
        if event is not None:
            for key, val in event["base_delta_y"].items():
                delta[key] = delta.get(key, 0.0) + val

        # 2 & 3. Action effects (one per agent, each at half weight)
        for action, x in [(action_h, x_h), (action_w, x_w)]:
            scale = x_scale_factor(action, x)
            for key, val in _ACTION_BASE[action].items():
                delta[key] = delta.get(key, 0.0) + val * scale * 0.5

        return delta
