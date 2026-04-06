import yaml
import numpy as np
from typing import Callable, Optional

from src.env.state import XTraits

# Type alias for an interaction entry
_Interaction = tuple[float, float, dict[str, float]]

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
    Uses effective() so the innate baseline is respected.
    """
    if action == ACTION_SUPPORT:
        return 0.5 + (x.effective("eq") + x.effective("kindness") + x.effective("ability_to_love")) / 3.0
    elif action == ACTION_ARGUE:
        return 0.5 + (1.0 - x.effective("mental_stability"))
    elif action == ACTION_IGNORE:
        return 1.0
    elif action == ACTION_COMPROMISE:
        return 0.5 + (x.effective("rational_thinking") + x.effective("eq")) / 2.0
    elif action == ACTION_WITHDRAW:
        return 1.0
    return 1.0


# ── Trait-dependent event probability modifiers ───────────────────────────────
#
# Each entry maps an event name to a function (x_h, x_w) → float multiplier.
# Multiplier is applied to the event's base probability before sampling.
# Values are clamped to [0.2, 3.0] to prevent any event from disappearing
# entirely or dominating.
#
# Motivation: faithfulness=0.9 and faithfulness=0.1 should NOT face the same
# infidelity risk. Traits must causally influence what happens, not just
# how agents respond.

_TRAIT_PROB_MODIFIERS: dict[str, Callable[[XTraits, XTraits], float]] = {
    # Low faithfulness → higher infidelity risk
    "infidelity": lambda h, w: (
        1.0 - 0.6 * (h.effective("faithfulness") + w.effective("faithfulness")) / 2.0
    ),
    # Low mental stability → more emotional conflict
    "emotional_conflict": lambda h, w: (
        1.0 + 0.5 * (1.0 - (h.effective("mental_stability") + w.effective("mental_stability")) / 2.0)
    ),
    # Low responsibility → more financial friction
    "financial_disagreement": lambda h, w: (
        1.0 + 0.4 * (1.0 - (h.effective("responsibility") + w.effective("responsibility")) / 2.0)
    ),
    # Having kids amplifies parenting conflict chance
    "parenting_conflict": lambda h, w: (
        1.0 + 0.5 * h.kids  # kids is shared, so x_h.kids == x_w.kids
    ),
    # Low responsibility → more job instability
    "job_loss": lambda h, w: (
        1.0 - 0.3 * (h.effective("responsibility") + w.effective("responsibility")) / 2.0
    ),
    # Low mental stability → more mental health episodes
    "mental_health_episode": lambda h, w: (
        1.0 + 0.4 * (1.0 - (h.effective("mental_stability") + w.effective("mental_stability")) / 2.0)
    ),
}


# ── Action interaction table ──────────────────────────────────────────────────
#
# Defines how the *combination* of both agents' actions plays out.
# Each entry: (action_h, action_w) → (scale_h, scale_w, extra_delta)
#
#   scale_h  — multiplier on H's individual action effect given W's response.
#              < 1.0 means W's response undermines H's action.
#              > 1.0 means W's response amplifies H's action.
#   scale_w  — same from W's perspective.
#   extra_delta — Y changes that emerge from the dynamic itself, beyond what
#                 either agent's individual action produces (e.g. stonewall
#                 penalty, mutual-support synergy).
#
# Pairs not listed default to (1.0, 1.0, {}) — no interaction effect.

_ACTION_INTERACTIONS: dict[tuple[int, int], _Interaction] = {
    # ── Both constructive ─────────────────────────────────────────────────────
    # Mutual support: warmth amplifies warmth
    (ACTION_SUPPORT, ACTION_SUPPORT):       (1.3, 1.3, {"love_support":  0.06, "happiness":  0.04}),
    # Support meets compromise: constructive alignment
    (ACTION_SUPPORT, ACTION_COMPROMISE):    (1.1, 1.1, {}),
    (ACTION_COMPROMISE, ACTION_SUPPORT):    (1.1, 1.1, {}),
    # Both willing to negotiate: solutions actually stick
    (ACTION_COMPROMISE, ACTION_COMPROMISE): (1.2, 1.2, {"stability":  0.05}),

    # ── Both disengaging ─────────────────────────────────────────────────────
    # Mutual silence: relationship fades without conflict
    (ACTION_IGNORE, ACTION_IGNORE):       (1.0, 1.0, {"love_support": -0.08, "stability": -0.06}),
    # Both pulling back emotionally: slow disconnection
    (ACTION_WITHDRAW, ACTION_WITHDRAW):   (1.0, 1.0, {"love_support": -0.06, "stability": -0.10}),
    # One stonewalls, other retreats: parallel disengagement
    (ACTION_WITHDRAW, ACTION_IGNORE):     (1.0, 1.0, {"love_support": -0.05, "stability": -0.05}),
    (ACTION_IGNORE, ACTION_WITHDRAW):     (1.0, 1.0, {"love_support": -0.05, "stability": -0.05}),

    # ── Both destructive ─────────────────────────────────────────────────────
    # Full mutual conflict: escalation compounds damage
    (ACTION_ARGUE, ACTION_ARGUE): (1.3, 1.3, {"love_support": -0.08, "stability": -0.08, "pressure": 0.08}),

    # ── Constructive vs destructive ───────────────────────────────────────────
    # Support vs argue: argue wins; supporter's effort is undermined
    (ACTION_SUPPORT, ACTION_ARGUE): (0.5, 1.2, {"pressure":  0.06}),
    (ACTION_ARGUE, ACTION_SUPPORT): (1.2, 0.5, {"pressure":  0.06}),
    # Compromise vs argue: offer to negotiate partially absorbs the attack
    (ACTION_COMPROMISE, ACTION_ARGUE): (0.7, 0.9, {"pressure":  0.04}),
    (ACTION_ARGUE, ACTION_COMPROMISE): (0.9, 0.7, {"pressure":  0.04}),

    # ── Constructive vs disengaged ────────────────────────────────────────────
    # Support vs ignore: care goes unreciprocated — one-sided and isolating
    (ACTION_SUPPORT, ACTION_IGNORE): (0.6, 1.0, {"love_support": -0.04}),
    (ACTION_IGNORE, ACTION_SUPPORT): (1.0, 0.6, {"love_support": -0.04}),
    # Support vs withdraw: classic pursuer-distancer strain
    (ACTION_SUPPORT, ACTION_WITHDRAW): (0.7, 1.0, {"love_support": -0.04, "stability": -0.04}),
    (ACTION_WITHDRAW, ACTION_SUPPORT): (1.0, 0.7, {"love_support": -0.04, "stability": -0.04}),
    # Compromise vs ignore: good-faith effort met with indifference
    (ACTION_COMPROMISE, ACTION_IGNORE):   (0.6, 1.0, {"love_support": -0.03}),
    (ACTION_IGNORE, ACTION_COMPROMISE):   (1.0, 0.6, {"love_support": -0.03}),
    # Compromise vs withdraw: offer made, partner already gone emotionally
    (ACTION_COMPROMISE, ACTION_WITHDRAW): (0.7, 1.0, {"stability": -0.03}),
    (ACTION_WITHDRAW, ACTION_COMPROMISE): (1.0, 0.7, {"stability": -0.03}),

    # ── Destructive vs disengaged ─────────────────────────────────────────────
    # Argue vs ignore: stonewalling — one of the most damaging dynamics
    (ACTION_ARGUE, ACTION_IGNORE): (1.2, 1.0, {"love_support": -0.08, "stability": -0.06, "pressure": 0.08}),
    (ACTION_IGNORE, ACTION_ARGUE): (1.0, 1.2, {"love_support": -0.08, "stability": -0.06, "pressure": 0.08}),
    # Argue vs withdraw: pursuer-distancer under active conflict
    (ACTION_ARGUE, ACTION_WITHDRAW): (1.1, 1.0, {"love_support": -0.06, "pressure":  0.05}),
    (ACTION_WITHDRAW, ACTION_ARGUE): (1.0, 1.1, {"love_support": -0.06, "pressure":  0.05}),
}


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

    def _adjusted_probs(self, x_h: XTraits, x_w: XTraits) -> np.ndarray:
        """
        Compute per-event probabilities adjusted for the couple's current traits.

        The total probability mass is preserved (same overall event frequency),
        but redistributed: e.g. faithful couples face less infidelity risk and
        more of that mass shifts to other events.
        """
        probs = self.probs.copy()
        for i, event in enumerate(self.events):
            name = event["name"]
            if name in _TRAIT_PROB_MODIFIERS:
                modifier = float(np.clip(_TRAIT_PROB_MODIFIERS[name](x_h, x_w), 0.2, 3.0))
                probs[i] *= modifier
        # Rescale to preserve the original total event probability
        original_total = self.probs.sum()
        adjusted_total = probs.sum()
        if adjusted_total > 0:
            probs *= original_total / adjusted_total
        return probs

    def sample(
        self,
        x_h: Optional[XTraits] = None,
        x_w: Optional[XTraits] = None,
    ) -> Optional[dict]:
        """
        Sample one event or None (no major event this year).

        If x_h and x_w are provided, event probabilities are adjusted by
        the couple's traits before sampling. Remaining probability mass
        after all events = chance of a quiet year.
        """
        if x_h is not None and x_w is not None:
            probs = self._adjusted_probs(x_h, x_w)
        else:
            probs = self.probs

        roll = np.random.random()
        cumulative = 0.0
        for event, p in zip(self.events, probs):
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

        # 2. Per-agent action effects, modulated by interaction context.
        #    scale_h / scale_w encode how the *other* agent's response affects
        #    the effectiveness of each action — so (support, argue) is not just
        #    the average of support and argue; the argue actively undermines the
        #    support (scale_h < 1) and the support doesn't stop the argue (scale_w > 1).
        scale_h, scale_w, interaction_delta = _ACTION_INTERACTIONS.get(
            (action_h, action_w), (1.0, 1.0, {})
        )
        for action, x, interaction_scale in [
            (action_h, x_h, scale_h),
            (action_w, x_w, scale_w),
        ]:
            trait_scale = x_scale_factor(action, x)
            for key, val in _ACTION_BASE[action].items():
                delta[key] = delta.get(key, 0.0) + val * trait_scale * interaction_scale * 0.5

        # 3. Interaction delta: effects that emerge from the dynamic itself
        #    (e.g. stonewall penalty, mutual-support warmth) beyond either
        #    agent's individual contribution.
        for key, val in interaction_delta.items():
            delta[key] = delta.get(key, 0.0) + val

        return delta
