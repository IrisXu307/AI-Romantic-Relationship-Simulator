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
    Quality = avg(eq, ability_to_love). Receptiveness = avg(partner eq, emotional_reasoning).
    Backfire: rational_thinking >> emotional_reasoning → support feels patronizing.

    Sensitivity amplifier: receiver's low rational_thinking (< 0.5) means they feel support
    more deeply — bigger love gain, bigger pressure relief, bigger happiness boost.
    Mirror of compromise: emotional types thrive on support; rational types on compromise.
    """
    quality      = (x_self.effective("eq") + x_self.effective("ability_to_love")) / 2.0
    receptive    = (x_other.effective("eq") + x_other.effective("emotional_reasoning")) / 2.0
    rational_gap = max(0.0, x_other.effective("rational_thinking") - x_other.effective("emotional_reasoning"))
    sensitivity  = max(0.0, 0.5 - x_other.effective("rational_thinking")) * 2.0

    context = _CONTEXT.get((ACTION_SUPPORT, partner_action), 0.0)
    effective_quality = quality * (1.0 + context)

    return {
        "love_support": 0.20 * effective_quality * receptive + sensitivity * 0.12 - rational_gap * 0.12,
        "pressure":    -0.12 * quality - sensitivity * 0.08,
        "stability":    0.10 * effective_quality * receptive + sensitivity * 0.05,
        "happiness":    0.08 * effective_quality * receptive + sensitivity * 0.10 - rational_gap * 0.06,
    }


def _argue_delta(x_self: XTraits, x_other: XTraits, partner_action: int) -> dict[str, float]:
    """
    Destructiveness = 0.5 + 0.5*(1 - mental_stability): unstable actors do more damage.
    Productive friction: high iq * rational_thinking slightly reduces damage.

    Sensitivity amplifier: receiver's low rational_thinking (< 0.5) means they feel argued-at
    more intensely — bigger love loss and happiness damage.
    Instability amplifier: receiver's low mental_stability (< 0.5) means their stability
    crumbles faster under conflict.
    """
    destructiveness = 0.5 + 0.5 * (1.0 - x_self.effective("mental_stability"))
    productive      = x_self.effective("iq") * x_self.effective("rational_thinking") * 0.3
    sensitivity     = max(0.0, 0.5 - x_other.effective("rational_thinking")) * 2.0
    instability     = max(0.0, 0.5 - x_other.effective("mental_stability")) * 2.0

    context = _CONTEXT.get((ACTION_ARGUE, partner_action), 0.0)
    net_destructiveness = destructiveness * (1.0 + context)

    return {
        "love_support": -0.28 * net_destructiveness - sensitivity * 0.10 + productive * 0.06,
        "pressure":      0.22 * net_destructiveness,
        "stability":    -0.18 * net_destructiveness - instability * 0.06 + productive * 0.05,
        "happiness":    -0.10 * net_destructiveness - sensitivity * 0.06 + productive * 0.03,
    }


def _ignore_delta(x_self: XTraits, x_other: XTraits, partner_action: int) -> dict[str, float]:
    """
    Strategic ignore (high mental_stability + responsibility) = de-escalation.
    Chronic ignore (low both) = neglect.

    Sensitivity amplifier: receiver's low rational_thinking (< 0.5) means they feel
    abandoned more acutely — bigger love loss and happiness damage from being ignored.
    """
    strategic   = (x_self.effective("mental_stability") + x_self.effective("responsibility")) / 2.0
    neglect     = 1.0 - strategic
    sensitivity = max(0.0, 0.5 - x_other.effective("rational_thinking")) * 2.0

    context = _CONTEXT.get((ACTION_IGNORE, partner_action), 0.0)
    net_neglect = neglect * (1.0 + context)

    return {
        "love_support": -0.15 * net_neglect - sensitivity * 0.08,
        "pressure":     -0.06 * strategic + 0.04 * net_neglect,
        "stability":    -0.10 * net_neglect + 0.02 * strategic,
        "happiness":    -0.05 * net_neglect - sensitivity * 0.04,
    }


def _compromise_delta(x_self: XTraits, x_other: XTraits, partner_action: int) -> dict[str, float]:
    """
    Effectiveness requires rational_thinking + iq to execute well.
    Backfire: emotional_reasoning >> rational_thinking → compromise feels cold.

    Rational amplifier: receiver's high rational_thinking (> 0.5) means they appreciate
    structured problem-solving — bigger stability gain, bigger pressure relief, more love.
    Mirror of support: rational types thrive on compromise; emotional types on support.
    """
    rationality   = (x_self.effective("rational_thinking") + x_self.effective("iq")) / 2.0
    emotional_gap = max(0.0, x_other.effective("emotional_reasoning") - x_other.effective("rational_thinking"))
    rational_amp  = max(0.0, x_other.effective("rational_thinking") - 0.5) * 2.0

    context = _CONTEXT.get((ACTION_COMPROMISE, partner_action), 0.0)
    effective_rationality = rationality * (1.0 + context)

    return {
        "love_support":  0.12 * effective_rationality + rational_amp * 0.08 - emotional_gap * 0.12,
        "pressure":     -0.15 * rationality - rational_amp * 0.08,
        "stability":     0.15 * effective_rationality + rational_amp * 0.12,
        "happiness":     0.08 * effective_rationality + rational_amp * 0.06 - emotional_gap * 0.06,
    }


def _withdraw_delta(x_self: XTraits, x_other: XTraits, partner_action: int) -> dict[str, float]:
    """
    Emotional shutdown (low ability_to_love + eq) = damage to love_support.
    Healthy space-taking = minimal damage.

    Sensitivity amplifier: receiver's low rational_thinking (< 0.5) means they feel
    abandoned more intensely — bigger love, stability, and happiness loss.
    """
    emotional_presence = (x_self.effective("ability_to_love") + x_self.effective("eq")) / 2.0
    shutdown    = 1.0 - emotional_presence
    sensitivity = max(0.0, 0.5 - x_other.effective("rational_thinking")) * 2.0

    context = _CONTEXT.get((ACTION_WITHDRAW, partner_action), 0.0)
    net_shutdown = shutdown * (1.0 + context)

    return {
        "love_support": -0.15 * net_shutdown - sensitivity * 0.08,
        "pressure":     -0.10,
        "stability":    -0.12 * net_shutdown - sensitivity * 0.04,
        "happiness":    -0.05 * net_shutdown - sensitivity * 0.05,
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


# ── Actor self-effects ────────────────────────────────────────────────────────
#
# What the ACTOR gains or loses from their own action, applied to their own YState.
# Separate from the receiver effects computed by the action formulas above.
#
# Design: each uses the same max(0, 0.5 - trait) * 2 or (trait - 0.5) * 2 pattern
# so the personality crossover is always at 0.5 and the range is [0, 1].

_SELF_EFFECTS: dict[int, Callable[[XTraits], dict[str, float]]] = {
    # Hot-headed actors (mental_stability < 0.5) vent frustration through arguing —
    # pressure release valve. Stable actors get no self-benefit from arguing.
    ACTION_ARGUE: lambda x: {
        "pressure": -(max(0.0, 0.5 - x.effective("mental_stability")) * 2.0) * 0.30,
        "happiness": (max(0.0, 0.5 - x.effective("mental_stability")) * 2.0) * 0.03,
    },
    # Avoidant actors (low eq + ATL, < 0.5 average) feel genuine relief from distance.
    # Warm actors feel no self-benefit — withdrawal is uncomfortable for them.
    ACTION_WITHDRAW: lambda x: {
        "pressure": -(max(0.0, 0.5 - (x.effective("ability_to_love") + x.effective("eq")) / 2.0) * 2.0) * 0.15,
        "happiness": (max(0.0, 0.5 - (x.effective("ability_to_love") + x.effective("eq")) / 2.0) * 2.0) * 0.04,
    },
    # Caring actors (high ATL + kindness) experience giving-joy from nurturing others.
    ACTION_SUPPORT: lambda x: {
        "happiness":    x.effective("ability_to_love") * x.effective("kindness") * 0.05,
        "love_support": x.effective("eq") * x.effective("ability_to_love") * 0.03,
    },
    # Rational actors (rational_thinking > 0.5) feel intellectual satisfaction from solving problems.
    # Also gain love_support: resolving conflict together is a form of intimacy for analytical people.
    ACTION_COMPROMISE: lambda x: {
        "pressure":     -(max(0.0, x.effective("rational_thinking") - 0.5) * 2.0) * 0.10,
        "stability":     (max(0.0, x.effective("rational_thinking") - 0.5) * 2.0) * x.effective("responsibility") * 0.05,
        "love_support":  (max(0.0, x.effective("rational_thinking") - 0.5) * 2.0) * 0.06,
    },
    # Strategic ignorers (high mental_stability) feel calm; chronic neglectors feel guilt.
    # Linear around 0.5: positive self-effect for stable, negative for unstable.
    ACTION_IGNORE: lambda x: {
        "pressure": -(x.effective("mental_stability") - 0.5) * 0.08,
        "happiness": (x.effective("mental_stability") - 0.5) * 0.04,
    },
}


# ── Personality-modulated event base effects ──────────────────────────────────
#
# Additive deltas applied to BOTH partners' Y states when a specific event fires,
# on top of the event's fixed base_delta_y from events.yaml.
# These encode how the same event hits differently depending on personality:
# an infidelity devastates an emotionally open couple but hits a rational couple
# mainly as a stability shock; a promotion excites ambitious couples more.
#
# Magnitudes are intentionally small (≤ 0.10) — amplifiers, not replacements.
# Applied after action effects so the event's base shock is already in delta.

_EVENT_BASE_MODIFIERS: dict[str, Callable[[XTraits, XTraits], dict[str, float]]] = {
    # Volatile couples feel emotional conflict more intensely
    "emotional_conflict": lambda h, w: {
        "pressure":     0.06 * max(0.0, 1.0 - (h.effective("mental_stability") + w.effective("mental_stability")) / 2.0),
        "love_support": -0.04 * max(0.0, 1.0 - (h.effective("eq") + w.effective("eq")) / 2.0),
    },
    # Romantic gestures land much harder for emotionally open couples
    "romantic_gesture": lambda h, w: {
        "happiness":    0.10 * (h.effective("ability_to_love") + w.effective("ability_to_love")) / 2.0,
        "love_support": 0.08 * (h.effective("eq") + w.effective("eq")) / 2.0,
    },
    # Responsible + rational couples navigate financial friction better
    "financial_disagreement": lambda h, w: {
        "stability": 0.06 * (h.effective("responsibility") + w.effective("responsibility")) / 2.0,
        "pressure":  -0.05 * (h.effective("rational_thinking") + w.effective("rational_thinking")) / 2.0,
    },
    # Resilient couples absorb health crises with less panic
    "health_crisis": lambda h, w: {
        "pressure":  -0.07 * (h.effective("mental_stability") + w.effective("mental_stability")) / 2.0,
        "stability":  0.05 * (h.effective("responsibility") + w.effective("responsibility")) / 2.0 - 0.025,
    },
    # Emotional couples feel betrayal as love devastation; rational couples as stability shock
    "infidelity": lambda h, w: {
        "love_support": -0.10 * (h.effective("ability_to_love") + w.effective("ability_to_love")) / 2.0,
        "stability":    -0.08 * (h.effective("rational_thinking") + w.effective("rational_thinking")) / 2.0,
    },
    # High-achieving couples get a bigger lift from shared wins
    "shared_achievement": lambda h, w: {
        "happiness": 0.08 * (h.effective("iq") + w.effective("iq")) / 2.0,
        "stability": 0.06 * (h.effective("responsibility") + w.effective("responsibility")) / 2.0,
    },
    # Unstable couples spiral harder during mental health episodes
    "mental_health_episode": lambda h, w: {
        "happiness": -0.07 * max(0.0, 1.0 - (h.effective("mental_stability") + w.effective("mental_stability")) / 2.0),
        "pressure":   0.09 * max(0.0, 1.0 - (h.effective("mental_stability") + w.effective("mental_stability")) / 2.0),
    },
    # Grief hits emotional couples harder but can pull them closer
    "family_death": lambda h, w: {
        "happiness":     -0.06 * (h.effective("eq") + w.effective("eq")) / 2.0,
        "love_support":   0.05 * (h.effective("ability_to_love") + w.effective("ability_to_love")) / 2.0,
    },
    # Irresponsible couples are hit harder by job loss; stable ones recover faster
    "job_loss": lambda h, w: {
        "pressure":  0.07 * max(0.0, 1.0 - (h.effective("responsibility") + w.effective("responsibility")) / 2.0),
        "stability": -0.05 * max(0.0, 1.0 - (h.effective("mental_stability") + w.effective("mental_stability")) / 2.0),
    },
    # Ambitious couples get a bigger happiness lift from career advancement
    "promotion": lambda h, w: {
        "happiness": 0.07 * (h.effective("iq") + w.effective("iq")) / 2.0,
        "stability": 0.04 * (h.effective("responsibility") + w.effective("responsibility")) / 2.0,
    },
    # Responsible couples capitalise on windfalls; irresponsible ones gain little lasting benefit
    "financial_windfall": lambda h, w: {
        "happiness": 0.07 * (h.effective("responsibility") + w.effective("responsibility")) / 2.0,
        "stability": 0.05 * (h.effective("rational_thinking") + w.effective("rational_thinking")) / 2.0,
    },
    # Low-patience, low-stability couples spiral in parenting conflicts
    "parenting_conflict": lambda h, w: {
        "pressure":  0.06 * max(0.0, 1.0 - (h.effective("mental_stability") + w.effective("mental_stability")) / 2.0),
        "stability": -0.05 * max(0.0, 1.0 - (h.effective("responsibility") + w.effective("responsibility")) / 2.0),
    },
    # Resilient, adaptable couples find relocation less disruptive
    "relocation": lambda h, w: {
        "pressure":  -0.05 * (h.effective("mental_stability") + w.effective("mental_stability")) / 2.0,
        "happiness":  0.04 * (h.effective("iq") + w.effective("iq")) / 2.0 - 0.02,
    },
    # Financial crises devastate irresponsible, irrational couples most
    "financial_crisis": lambda h, w: {
        "pressure":  0.09 * max(0.0, 1.0 - (h.effective("responsibility") + w.effective("responsibility")) / 2.0),
        "stability": -0.07 * max(0.0, 1.0 - (h.effective("rational_thinking") + w.effective("rational_thinking")) / 2.0),
    },
    # Caring couples bond more deeply over a new child
    "new_child": lambda h, w: {
        "love_support": 0.07 * (h.effective("kindness") + w.effective("kindness")) / 2.0,
        "happiness":    0.05 * (h.effective("ability_to_love") + w.effective("ability_to_love")) / 2.0,
    },
    # Quality time replenishes emotionally connected couples most
    "quality_time": lambda h, w: {
        "love_support": 0.07 * (h.effective("ability_to_love") + w.effective("ability_to_love")) / 2.0,
        "happiness":    0.05 * (h.effective("eq") + w.effective("eq")) / 2.0,
    },
    # Everyday kindness resonates most with kind, empathetic couples
    "everyday_kindness": lambda h, w: {
        "happiness":    0.05 * (h.effective("kindness") + w.effective("kindness")) / 2.0,
        "love_support": 0.04 * (h.effective("eq") + w.effective("eq")) / 2.0,
    },
}


# ── Pair-level emergent effects ───────────────────────────────────────────────
#
# Effects that emerge from the *combination* of both actions — beyond what
# either agent's individual formula already captures.
# Only non-trivial pairs are listed; all others have no extra effect.

# ── Trust & resentment accumulation rates ────────────────────────────────────
#
# These small per-step deltas are applied to the *receiving* partner's YState
# based on what their partner just did.  They accumulate slowly into a history
# signal without breaking the Markov property of the observation.
#
# Design: magnitudes are intentionally small (≤ 0.04) so they take many
# consistent steps to saturate — a single bad action doesn't destroy trust,
# but a chronic pattern does.

_TRUST_DELTA: dict[int, float] = {
    ACTION_SUPPORT:    +0.020,
    ACTION_ARGUE:      -0.040,
    ACTION_IGNORE:     -0.030,
    ACTION_COMPROMISE: +0.015,
    ACTION_WITHDRAW:   -0.015,
}

_RESENTMENT_DELTA: dict[int, float] = {
    ACTION_SUPPORT:    -0.015,
    ACTION_ARGUE:      +0.040,
    ACTION_IGNORE:     +0.030,
    ACTION_COMPROMISE: -0.010,
    ACTION_WITHDRAW:   +0.020,
}


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


# ── Life-stage event probability modifiers ────────────────────────────────────
#
# Each entry maps an event name to a function age → float multiplier.
# Applied on top of trait modifiers; both are renormalized together so total
# event frequency is preserved — only the composition shifts across life stages.
#
# Design reflects empirically observed lifetime distributions:
#   - New children: peak 25-35, near-zero after 45
#   - Health crises: rise sharply after 50
#   - Job loss / financial friction: higher in early career
#   - Family death: accelerates as the couple's parents age (couple's 50s+)
#   - Quality time: rises with empty-nest and retirement (60+)

_STAGE_PROB_MODIFIERS: dict[str, Callable[[int], float]] = {
    # New child: elevated in 20s, drops off sharply after 35, near-zero after 45
    "new_child": lambda age: max(
        0.0, 1.5 - max(0.0, (age - 28) / 8.0)
    ),
    # Health crisis: low when young, rises steeply after 50
    "health_crisis": lambda age: (
        0.3 + 1.7 * min(1.0, max(0.0, (age - 45) / 30.0))
    ),
    # Mental health episodes: modest rise with age and accumulated stress
    "mental_health_episode": lambda age: (
        0.7 + 0.6 * min(1.0, max(0.0, (age - 35) / 40.0))
    ),
    # Job loss: higher early in career, stabilises after 55
    "job_loss": lambda age: (
        1.4 - 0.7 * min(1.0, max(0.0, (age - 25) / 35.0))
    ),
    # Promotion: career peak 30-50, falls toward retirement
    "promotion": lambda age: (
        0.4 + 1.2 * min(1.0, max(0.0, (age - 28) / 15.0))
        - 1.0 * min(1.0, max(0.0, (age - 52) / 20.0))
    ),
    # Financial disagreement: more acute in budget-tight early years
    "financial_disagreement": lambda age: (
        1.5 - 1.0 * min(1.0, max(0.0, (age - 25) / 30.0))
    ),
    # Financial crisis: elevated in middle years (mortgage, children's education)
    "financial_crisis": lambda age: (
        0.7 + 0.7 * min(1.0, max(0.0, (age - 30) / 20.0))
        - 0.6 * min(1.0, max(0.0, (age - 55) / 20.0))
    ),
    # Family death: accelerates as the couple's parents age (couple in their 50s+)
    "family_death": lambda age: (
        0.2 + 1.8 * min(1.0, max(0.0, (age - 45) / 30.0))
    ),
    # Parenting conflict: most acute during active parenting (30-50)
    "parenting_conflict": lambda age: (
        0.2 + 1.4 * min(1.0, max(0.0, (age - 28) / 12.0))
        - 1.2 * min(1.0, max(0.0, (age - 50) / 15.0))
    ),
    # Relocation: most common in early-career mobility, rare after 60
    "relocation": lambda age: max(
        0.1, 1.4 - 1.0 * min(1.0, max(0.0, (age - 25) / 35.0))
    ),
    # Shared achievement: peaks in mid-career, falls at retirement
    "shared_achievement": lambda age: (
        0.5 + 1.0 * min(1.0, max(0.0, (age - 30) / 15.0))
        - 0.6 * min(1.0, max(0.0, (age - 55) / 20.0))
    ),
    # Quality time: rises with empty-nest / retirement freedom
    "quality_time": lambda age: (
        0.6 + 0.8 * min(1.0, max(0.0, (age - 50) / 25.0))
    ),
    # Romantic gesture: slightly higher when relationship is young
    "romantic_gesture": lambda age: (
        1.3 - 0.5 * min(1.0, max(0.0, (age - 25) / 55.0))
    ),
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

# B1: coefficients halved from original — traits nudge the distribution,
# they don't decide it. Policy choices carry more relative weight.
_TRAIT_PROB_MODIFIERS: dict[str, Callable[[XTraits, XTraits], float]] = {
    "infidelity": lambda h, w: (
        1.0 - 0.3 * (h.effective("faithfulness") + w.effective("faithfulness")) / 2.0
    ),
    "emotional_conflict": lambda h, w: (
        1.0
        + 0.2 * (1.0 - (h.effective("mental_stability") + w.effective("mental_stability")) / 2.0)
        + 0.1 * (h.effective("emotional_reasoning") + w.effective("emotional_reasoning")) / 2.0
        - 0.1 * (h.effective("eq") + w.effective("eq")) / 2.0
    ),
    "financial_disagreement": lambda h, w: (
        1.0 + 0.2 * (1.0 - (h.effective("responsibility") + w.effective("responsibility")) / 2.0)
    ),
    "parenting_conflict": lambda h, _: (
        1.0 + 0.25 * h.kids
    ),
    "job_loss": lambda h, w: (
        1.0 - 0.15 * (h.effective("responsibility") + w.effective("responsibility")) / 2.0
    ),
    "mental_health_episode": lambda h, w: (
        1.0 + 0.2 * (1.0 - (h.effective("mental_stability") + w.effective("mental_stability")) / 2.0)
    ),
    "romantic_gesture": lambda h, w: (
        1.0
        + 0.15 * (h.effective("ability_to_love") + w.effective("ability_to_love")) / 2.0
        + 0.10 * (h.effective("eq") + w.effective("eq")) / 2.0
    ),
    "quality_time": lambda h, w: (
        1.0
        + 0.15 * (h.effective("eq") + w.effective("eq")) / 2.0
        + 0.10 * (h.effective("ability_to_love") + w.effective("ability_to_love")) / 2.0
    ),
    "everyday_kindness": lambda h, w: (
        1.0 + 0.2 * (h.effective("kindness") + w.effective("kindness")) / 2.0
    ),
    "shared_achievement": lambda h, w: (
        1.0
        + 0.10 * (h.effective("responsibility") + w.effective("responsibility")) / 2.0
        + 0.075 * (h.effective("iq") + w.effective("iq")) / 2.0
    ),
}


# ── C1: Behavior-pattern event probability modifiers ─────────────────────────
#
# A consistent behavioral pattern over the last N steps shifts the event mix.
# Cooperative couples (support/compromise dominant) experience less conflict
# and more positive events — their behavior shapes their circumstances.
# Destructive couples (argue/ignore dominant) face escalating conflict.
#
# This creates genuine long-term strategy: a patient cooperator gets a
# different future than a chronic avoider, regardless of their traits.

_BEHAVIOR_PROB_MODIFIERS: dict[str, dict[str, float]] = {
    "cooperative": {   # support/compromise > 60% of recent actions
        "emotional_conflict":    0.55,
        "infidelity":            0.65,
        "romantic_gesture":      1.40,
        "quality_time":          1.30,
        "everyday_kindness":     1.20,
    },
    "destructive": {   # argue/ignore > 60% of recent actions
        "emotional_conflict":    1.55,
        "infidelity":            1.35,
        "mental_health_episode": 1.25,
        "romantic_gesture":      0.65,
        "quality_time":          0.70,
    },
}

# C2: During high-stakes events, action choice matters much more.
# Same event + different action → very different trajectory.
_HIGH_STAKES_EVENTS: frozenset = frozenset({
    "infidelity", "family_death", "health_crisis", "mental_health_episode",
})
_HIGH_STAKES_ACTION_SCALE: float = 2.2


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

    def _adjusted_probs(
        self,
        x_h: XTraits,
        x_w: XTraits,
        age: int = 25,
        action_history_h: Optional[list] = None,
        action_history_w: Optional[list] = None,
    ) -> np.ndarray:
        """
        Compute per-event probabilities adjusted for traits, life stage, and behavior history.

        C1: if the combined recent action history is >60% cooperative (support/compromise)
        or >60% destructive (argue/ignore), apply _BEHAVIOR_PROB_MODIFIERS to shift
        the event mix toward their behavioral pattern's natural consequences.
        """
        probs = self.probs.copy()
        for i, event in enumerate(self.events):
            name = event["name"]
            if name in _TRAIT_PROB_MODIFIERS:
                modifier = float(np.clip(_TRAIT_PROB_MODIFIERS[name](x_h, x_w), 0.2, 3.0))
                probs[i] *= modifier
            if name in _STAGE_PROB_MODIFIERS:
                modifier = float(np.clip(_STAGE_PROB_MODIFIERS[name](age), 0.0, 3.0))
                probs[i] *= modifier

        # C1: behavior history shifts event probabilities
        all_actions: list[int] = []
        if action_history_h:
            all_actions.extend(action_history_h)
        if action_history_w:
            all_actions.extend(action_history_w)
        if all_actions:
            cooperative_actions = {ACTION_SUPPORT, ACTION_COMPROMISE}
            destructive_actions = {ACTION_ARGUE, ACTION_IGNORE}
            n = len(all_actions)
            coop_frac = sum(1 for a in all_actions if a in cooperative_actions) / n
            dest_frac = sum(1 for a in all_actions if a in destructive_actions) / n
            behavior_key: Optional[str] = None
            if coop_frac > 0.60:
                behavior_key = "cooperative"
            elif dest_frac > 0.60:
                behavior_key = "destructive"
            if behavior_key is not None:
                mods = _BEHAVIOR_PROB_MODIFIERS[behavior_key]
                for i, event in enumerate(self.events):
                    if event["name"] in mods:
                        probs[i] *= float(np.clip(mods[event["name"]], 0.2, 3.0))

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
        age: int = 25,
        action_history_h: Optional[list] = None,
        action_history_w: Optional[list] = None,
    ) -> Optional[dict]:
        """
        Sample one event or None (no major event this year).

        If x_h and x_w are provided, event probabilities are adjusted by
        the couple's traits and life stage before sampling.
        Remaining probability mass after all events = chance of a quiet year.
        """
        if x_h is not None and x_w is not None:
            probs = self._adjusted_probs(
                x_h, x_w, age=age,
                action_history_h=action_history_h,
                action_history_w=action_history_w,
            )
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
        habituation: float = 1.0,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Compute per-partner ΔY for one timestep.

        Returns (delta_h, delta_w) — each partner's subjective experience of the step.

        Design:
          - Base event delta applies equally to both (the event happens to the couple),
            scaled by habituation [0, 1]. Repeated events have diminishing base impact.
          - H's subjective state is shaped by W's action toward H: W acts, H receives.
          - W's subjective state is shaped by H's action toward W: H acts, W receives.
          - Pair-level emergent effects are felt equally by both.
          - Trust/resentment accumulate slowly based on the partner's action,
            encoding relationship history into the observation without breaking Markov.
          - Action formula deltas are NOT habituated — behavioural choices retain
            full impact regardless of how often an event type recurs.
        """
        # 1. Base event effect — same for both partners, scaled by habituation
        base: dict[str, float] = {}
        if event is not None:
            for key, val in event["base_delta_y"].items():
                base[key] = val * habituation

        delta_h: dict[str, float] = dict(base)
        delta_w: dict[str, float] = dict(base)

        # C2: high-stakes events amplify action impact — the same response to a
        # crisis vs a mild event has very different consequences.
        event_name = event["name"] if event is not None else ""
        action_scale = _HIGH_STAKES_ACTION_SCALE if event_name in _HIGH_STAKES_EVENTS else 1.0

        # 2. Action effects: each partner experiences what their partner does to them
        #    W's action → how H's subjective state changes (H is the receiver)
        for key, val in _action_delta(action_w, x_w, x_h, action_h).items():
            delta_h[key] = delta_h.get(key, 0.0) + val * action_scale

        #    H's action → how W's subjective state changes (W is the receiver)
        for key, val in _action_delta(action_h, x_h, x_w, action_w).items():
            delta_w[key] = delta_w.get(key, 0.0) + val * action_scale

        # 2.5. Actor self-effects: personality-driven gains/costs of your own action
        for key, val in _SELF_EFFECTS[action_h](x_h).items():
            delta_h[key] = delta_h.get(key, 0.0) + val
        for key, val in _SELF_EFFECTS[action_w](x_w).items():
            delta_w[key] = delta_w.get(key, 0.0) + val

        # 3. Emergent pair dynamics — felt by both
        for key, val in _PAIR_EXTRAS.get((action_h, action_w), {}).items():
            delta_h[key] = delta_h.get(key, 0.0) + val
            delta_w[key] = delta_w.get(key, 0.0) + val

        # 4. Trust & resentment: H's accumulate based on W's action, and vice versa
        delta_h["trust"]      = _TRUST_DELTA[action_w]
        delta_h["resentment"] = _RESENTMENT_DELTA[action_w]
        delta_w["trust"]      = _TRUST_DELTA[action_h]
        delta_w["resentment"] = _RESENTMENT_DELTA[action_h]

        # 5. Personality-modulated event base effects — same event hits differently per couple
        if event is not None:
            name = event["name"]
            if name in _EVENT_BASE_MODIFIERS:
                for key, val in _EVENT_BASE_MODIFIERS[name](x_h, x_w).items():
                    delta_h[key] = delta_h.get(key, 0.0) + val
                    delta_w[key] = delta_w.get(key, 0.0) + val

        return delta_h, delta_w
