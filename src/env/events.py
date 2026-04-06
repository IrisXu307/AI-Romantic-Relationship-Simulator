import yaml
import numpy as np
from typing import Callable, Optional

from src.env.state import XTraits

# ── Action space ──────────────────────────────────────────────────────────────

ACTION_SUPPORT    = 0  # try to help, comfort, or encourage
ACTION_ARGUE      = 1  # escalate conflict, blame, push back
ACTION_IGNORE     = 2  # disengage, avoid dealing with the situation
ACTION_COMPROMISE = 3  # negotiate, meet in the middle
ACTION_WITHDRAW   = 4  # pull back emotionally without fighting

N_ACTIONS = 5
ACTION_NAMES = ["support", "argue", "ignore", "compromise", "withdraw"]


# ── Action formula functions ───────────────────────────────────────────────────
#
# Each function computes one agent's ΔY contribution for a single action.
# Signature: f(x_self, x_other, partner_action) → dict[str, float]
#
# Design rules:
#   - x_self  drives the *quality* of the action (how well they execute it)
#   - x_other drives *receptiveness* (how the partner receives or reacts)
#   - partner_action provides context (hard to support someone who is attacking)
#
# No action is universally good or bad:
#   - SUPPORT backfires with a highly rational, non-emotional partner
#   - COMPROMISE feels cold to a highly emotional partner
#   - ARGUE is less damaging when done by a high-IQ rational agent (productive friction)
#   - IGNORE is strategic de-escalation when self has high stability + responsibility
#   - WITHDRAW always lowers pressure (why people do it), but damages love_support
#     proportional to emotional shutdown
#
# The RL agent must learn which action fits its partner's personality and the
# current context — not just memorize "support = always good."

# Context modifiers: how partner's action changes MY action's effectiveness.
# Positive = partner context helps my action; Negative = undermines it.
# Applied multiplicatively to quality/destructiveness inside each formula.
_CONTEXT: dict[tuple[int, int], float] = {
    # My action, partner's action → modifier on MY effectiveness
    (ACTION_SUPPORT,    ACTION_SUPPORT):    0.20,   # warmth is mutual, quality amplified
    (ACTION_SUPPORT,    ACTION_ARGUE):     -0.35,   # hard to support someone attacking you
    (ACTION_SUPPORT,    ACTION_IGNORE):    -0.20,   # support doesn't land with absent partner
    (ACTION_SUPPORT,    ACTION_COMPROMISE): 0.10,   # compatible context
    (ACTION_SUPPORT,    ACTION_WITHDRAW):  -0.15,   # partner isn't fully present

    (ACTION_ARGUE,      ACTION_SUPPORT):   -0.15,   # being supported slightly de-escalates
    (ACTION_ARGUE,      ACTION_ARGUE):      0.30,   # mutual escalation amplifies damage
    (ACTION_ARGUE,      ACTION_IGNORE):     0.20,   # stonewalling enrages, argue worsens
    (ACTION_ARGUE,      ACTION_COMPROMISE):-0.20,   # offer to negotiate absorbs some anger
    (ACTION_ARGUE,      ACTION_WITHDRAW):   0.15,   # pursuer-distancer: argue intensifies

    (ACTION_IGNORE,     ACTION_SUPPORT):    0.10,   # ignoring a supporter feels worse (guilt/damage)
    (ACTION_IGNORE,     ACTION_ARGUE):     -0.20,   # strategic ignore when attacked is most valid
    (ACTION_IGNORE,     ACTION_IGNORE):     0.25,   # mutual neglect compounds
    (ACTION_IGNORE,     ACTION_COMPROMISE): 0.20,   # ignoring someone trying is especially bad
    (ACTION_IGNORE,     ACTION_WITHDRAW):   0.10,   # parallel disengagement compounds

    (ACTION_COMPROMISE, ACTION_SUPPORT):    0.15,   # warm environment helps compromise land
    (ACTION_COMPROMISE, ACTION_ARGUE):     -0.15,   # hard to negotiate when under attack
    (ACTION_COMPROMISE, ACTION_IGNORE):    -0.30,   # effort wasted when ignored
    (ACTION_COMPROMISE, ACTION_COMPROMISE): 0.25,   # mutual problem-solving most effective
    (ACTION_COMPROMISE, ACTION_WITHDRAW):  -0.15,   # compromising with someone checked out

    (ACTION_WITHDRAW,   ACTION_SUPPORT):    0.20,   # withdrawing from a supporter = rejection; more damage
    (ACTION_WITHDRAW,   ACTION_ARGUE):     -0.25,   # withdrawal as de-escalation is most justified
    (ACTION_WITHDRAW,   ACTION_IGNORE):     0.10,   # mutual disconnection compounds
    (ACTION_WITHDRAW,   ACTION_COMPROMISE): 0.15,   # withdrawing when partner is trying = bad
    (ACTION_WITHDRAW,   ACTION_WITHDRAW):   0.10,   # mutual retreat compounds
}


def _support_delta(x_self: XTraits, x_other: XTraits, partner_action: int) -> dict[str, float]:
    """
    Support quality = avg(eq, ability_to_love).
    Receptiveness = avg(partner eq, partner emotional_reasoning).

    Backfire condition: if partner's rational_thinking >> emotional_reasoning,
    they interpret unsolicited support as patronizing or controlling.
    The love_support delta can go negative for highly rational, non-emotional partners.
    """
    quality    = (x_self.effective("eq") + x_self.effective("ability_to_love")) / 2.0
    receptive  = (x_other.effective("eq") + x_other.effective("emotional_reasoning")) / 2.0
    # How much more rational than emotional is the partner?
    # Positive gap → support feels like interference, not comfort.
    rational_gap = max(0.0, x_other.effective("rational_thinking") - x_other.effective("emotional_reasoning"))

    context = _CONTEXT.get((ACTION_SUPPORT, partner_action), 0.0)
    effective_quality = quality * (1.0 + context)

    return {
        "love_support": 0.20 * effective_quality * receptive - rational_gap * 0.10,
        "pressure":    -0.12 * quality,                        # always reduces pressure regardless
        "stability":    0.10 * effective_quality * receptive,
        "happiness":    0.08 * effective_quality * receptive - rational_gap * 0.05,
    }


def _argue_delta(x_self: XTraits, x_other: XTraits, partner_action: int) -> dict[str, float]:
    """
    Destructiveness = 0.5 + 0.5*(1 - mental_stability): unstable agents do more damage.
    Productive friction: high iq * rational_thinking → argument surfaces real issues,
    slightly reducing the damage to stability and love_support (but never making argue good).
    """
    destructiveness = 0.5 + 0.5 * (1.0 - x_self.effective("mental_stability"))
    productive      = x_self.effective("iq") * x_self.effective("rational_thinking") * 0.3

    context = _CONTEXT.get((ACTION_ARGUE, partner_action), 0.0)
    net_destructiveness = destructiveness * (1.0 + context)

    return {
        "love_support": -0.28 * net_destructiveness + productive * 0.06,
        "pressure":      0.22 * net_destructiveness,
        "stability":    -0.18 * net_destructiveness + productive * 0.05,
        "happiness":    -0.10 * net_destructiveness + productive * 0.03,
    }


def _ignore_delta(x_self: XTraits, x_other: XTraits, partner_action: int) -> dict[str, float]:
    """
    Strategic ignore (high mental_stability + responsibility) = de-escalation.
    Chronic ignore (low both) = neglect.
    Partner's eq + ability_to_love determines how acutely they feel the absence.
    """
    strategic = (x_self.effective("mental_stability") + x_self.effective("responsibility")) / 2.0
    neglect   = 1.0 - strategic
    partner_sensitivity = (x_other.effective("eq") + x_other.effective("ability_to_love")) / 2.0

    context = _CONTEXT.get((ACTION_IGNORE, partner_action), 0.0)
    # Context amplifies neglect, not strategic value
    net_neglect = neglect * (1.0 + context)

    return {
        "love_support": -0.15 * net_neglect - partner_sensitivity * 0.06,
        "pressure":     -0.06 * strategic + 0.04 * net_neglect,  # strategic lowers pressure
        "stability":    -0.10 * net_neglect + 0.02 * strategic,
        "happiness":    -0.05 * net_neglect,
    }


def _compromise_delta(x_self: XTraits, x_other: XTraits, partner_action: int) -> dict[str, float]:
    """
    Effectiveness requires rational_thinking + iq to execute well.
    Backfire condition: if partner's emotional_reasoning >> rational_thinking,
    compromise feels cold and transactional — love_support can go slightly negative.
    """
    rationality = (x_self.effective("rational_thinking") + x_self.effective("iq")) / 2.0
    # How much more emotional than rational is the partner?
    # Positive gap → compromise feels dismissive of their feelings.
    emotional_gap = max(0.0, x_other.effective("emotional_reasoning") - x_other.effective("rational_thinking"))

    context = _CONTEXT.get((ACTION_COMPROMISE, partner_action), 0.0)
    effective_rationality = rationality * (1.0 + context)

    return {
        "love_support":  0.12 * effective_rationality - emotional_gap * 0.10,
        "pressure":     -0.15 * rationality,               # always reduces pressure
        "stability":     0.15 * effective_rationality,
        "happiness":     0.08 * effective_rationality - emotional_gap * 0.05,
    }


def _withdraw_delta(x_self: XTraits, x_other: XTraits, partner_action: int) -> dict[str, float]:
    """
    Withdraw always reduces pressure — that's why people do it.
    Emotional shutdown (low ability_to_love + eq) = damage to love_support.
    Healthy space-taking (high mental_stability + eq) = minimal damage.
    Partner with high attachment feels the abandonment more acutely.
    """
    emotional_presence = (x_self.effective("ability_to_love") + x_self.effective("eq")) / 2.0
    shutdown = 1.0 - emotional_presence
    partner_attachment = (x_other.effective("ability_to_love") + x_other.effective("eq")) / 2.0

    context = _CONTEXT.get((ACTION_WITHDRAW, partner_action), 0.0)
    net_shutdown = shutdown * (1.0 + context)

    return {
        "love_support": -0.15 * net_shutdown - partner_attachment * 0.05,
        "pressure":     -0.10,                  # always lowers pressure (intentional)
        "stability":    -0.12 * net_shutdown,
        "happiness":    -0.05 * net_shutdown,
    }


_ACTION_FORMULAS: dict[int, Callable[[XTraits, XTraits, int], dict[str, float]]] = {
    ACTION_SUPPORT:    _support_delta,
    ACTION_ARGUE:      _argue_delta,
    ACTION_IGNORE:     _ignore_delta,
    ACTION_COMPROMISE: _compromise_delta,
    ACTION_WITHDRAW:   _withdraw_delta,
}


def _action_delta(action: int, x_self: XTraits, x_other: XTraits, partner_action: int) -> dict[str, float]:
    """Dispatch to the correct action formula."""
    return _ACTION_FORMULAS[action](x_self, x_other, partner_action)


# ── Pair-level emergent effects ───────────────────────────────────────────────
#
# Effects that emerge from the *combination* of both actions — beyond what
# either agent's individual formula already captures.
# Only non-trivial pairs are listed; all others have no extra effect.

_PAIR_EXTRAS: dict[tuple[int, int], dict[str, float]] = {
    # Mutual warmth: emotional climate synergy
    (ACTION_SUPPORT, ACTION_SUPPORT):       {"love_support":  0.05, "happiness":  0.04},
    # Joint problem-solving: solutions that stick
    (ACTION_COMPROMISE, ACTION_COMPROMISE): {"stability":  0.04},
    # Mutual escalation: spiral beyond individual damage
    (ACTION_ARGUE, ACTION_ARGUE):           {"love_support": -0.06, "stability": -0.06, "pressure": 0.06},
    # Stonewall: one attacks, one shuts down — relationship damage beyond both actions
    (ACTION_ARGUE, ACTION_IGNORE):          {"love_support": -0.06, "stability": -0.05, "pressure": 0.06},
    (ACTION_IGNORE, ACTION_ARGUE):          {"love_support": -0.06, "stability": -0.05, "pressure": 0.06},
    # Mutual disengagement
    (ACTION_IGNORE, ACTION_IGNORE):         {"love_support": -0.04, "stability": -0.04},
    (ACTION_WITHDRAW, ACTION_WITHDRAW):     {"love_support": -0.04, "stability": -0.06},
    # Pursuer-distancer under conflict
    (ACTION_ARGUE, ACTION_WITHDRAW):        {"love_support": -0.04, "pressure":  0.04},
    (ACTION_WITHDRAW, ACTION_ARGUE):        {"love_support": -0.04, "pressure":  0.04},
}


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

        1. Base event delta (from YAML)
        2. Each agent's action formula, at half weight so both together don't double-count.
           Formula takes (x_self, x_other, partner_action) — same action produces
           different deltas depending on who is doing it and who is receiving it.
        3. Pair-level emergent effects (_PAIR_EXTRAS): dynamics that arise from
           the combination and aren't captured by individual formulas.
        """
        delta: dict[str, float] = {}

        # 1. Base event effect
        if event is not None:
            for key, val in event["base_delta_y"].items():
                delta[key] = delta.get(key, 0.0) + val

        # 2. Per-agent action deltas, each at half weight
        for d in [
            _action_delta(action_h, x_h, x_w, action_w),
            _action_delta(action_w, x_w, x_h, action_h),
        ]:
            for key, val in d.items():
                delta[key] = delta.get(key, 0.0) + val * 0.5

        # 3. Emergent pair dynamics
        for key, val in _PAIR_EXTRAS.get((action_h, action_w), {}).items():
            delta[key] = delta.get(key, 0.0) + val

        return delta
