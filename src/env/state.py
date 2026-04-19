from dataclasses import dataclass, field
from typing import ClassVar, Optional
import numpy as np

# Dimensionality constants used throughout the codebase
X_DIM = 10
Y_DIM = 7  # wealth, love_support, pressure, happiness, stability, trust, resentment

# Weighting between fixed innate baseline and mutable learned component.
# Effective trait = INNATE_W * innate + LEARNED_W * learned
# The innate floor anchors each agent's identity so traits can never fully
# converge to perfect — growth is real but bounded by who you started as.
INNATE_W: float = 0.5
LEARNED_W: float = 0.5

# Ordered names of the 9 mutable traits (excludes kids).
# Used by to_array(), clip(), and effective() to stay in sync.
_TRAIT_NAMES: tuple[str, ...] = (
    "iq", "eq", "rational_thinking", "emotional_reasoning",
    "kindness", "ability_to_love", "faithfulness",
    "responsibility", "mental_stability",
)


@dataclass
class XTraits:
    """
    Internal trait vector for one agent.

    Each trait has two components:
      - innate:  fixed baseline sampled at init, never changes.
      - learned: mutable component updated by reflection (the named fields below).

    Effective value = INNATE_W * innate + LEARNED_W * learned

    This prevents convergence to a "perfect" personality: the innate floor
    anchors each agent's identity regardless of how much they grow.
    kids is handled separately — it's a shared life-event counter, not a
    psychological trait.
    """

    # Learned (mutable) trait components — reflection modifies these
    iq: float = 0.5
    eq: float = 0.5
    rational_thinking: float = 0.5
    emotional_reasoning: float = 0.5
    kindness: float = 0.5
    ability_to_love: float = 0.5
    faithfulness: float = 0.5
    responsibility: float = 0.5
    mental_stability: float = 0.5
    kids: float = 0.0  # normalized: actual_kids / MAX_KIDS

    # Fixed innate baseline — written once in __post_init__, never updated.
    # Excluded from repr and equality checks; not a psychological trait.
    innate: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    MAX_KIDS: ClassVar[float] = 5.0

    def __post_init__(self):
        if self.innate is None:
            # Freeze innate to match the initial learned values at construction time
            self.innate = np.array(
                [getattr(self, t) for t in _TRAIT_NAMES], dtype=np.float32
            )

    # ── Trait access ──────────────────────────────────────────────────────────

    def _learned_array(self) -> np.ndarray:
        """Current learned component as an array (9 traits, excludes kids)."""
        return np.array([getattr(self, t) for t in _TRAIT_NAMES], dtype=np.float32)

    def effective(self, trait: str) -> float:
        """
        Effective value of a named trait after innate weighting.
        Use this wherever trait values influence game logic (e.g. event
        probabilities, action scaling) so the innate anchor is respected.
        """
        idx = _TRAIT_NAMES.index(trait)
        learned_val = getattr(self, trait)
        return float(INNATE_W * self.innate[idx] + LEARNED_W * learned_val)

    def to_array(self) -> np.ndarray:
        """
        Effective trait vector seen by the policy (length X_DIM).
        = INNATE_W * innate + LEARNED_W * learned, with kids appended.
        """
        effective = INNATE_W * self.innate + LEARNED_W * self._learned_array()
        return np.append(effective, self.kids).astype(np.float32)

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def clip(self):
        """Clip all mutable trait values to [0, 1]."""
        for name in _TRAIT_NAMES:
            setattr(self, name, float(np.clip(getattr(self, name), 0.0, 1.0)))
        self.kids = float(np.clip(self.kids, 0.0, 1.0))

    def increment_kids(self):
        """Increase the normalized kids count by one child, capped at MAX_KIDS."""
        raw = round(self.kids * self.MAX_KIDS) + 1
        self.kids = float(np.clip(raw / self.MAX_KIDS, 0.0, 1.0))


@dataclass
class YState:
    """
    One partner's subjective relationship state. All values in [0, 1].

    Each partner holds their own YState — two people in the same marriage
    can experience it differently. trust and resentment are accumulated
    history signals: they change slowly each step based on the *partner's*
    action, giving the policy a memory of past behaviour without violating
    the Markov property of the observation.

    Field order (matches to_array):
        wealth, love_support, pressure, happiness, stability, trust, resentment
    """
    wealth:       float = 0.5
    love_support: float = 0.7
    pressure:     float = 0.3
    happiness:    float = 0.7
    stability:    float = 0.8
    trust:        float = 0.5   # this partner's trust IN their partner
    resentment:   float = 0.1   # this partner's resentment TOWARD their partner

    def to_array(self) -> np.ndarray:
        return np.array([
            self.wealth, self.love_support, self.pressure,
            self.happiness, self.stability, self.trust, self.resentment,
        ], dtype=np.float32)

    def apply_delta(self, delta: dict):
        """Add delta values and clip to [0, 1]."""
        for key, val in delta.items():
            if hasattr(self, key):
                current = getattr(self, key)
                setattr(self, key, float(np.clip(current + val, 0.0, 1.0)))

    def copy(self) -> "YState":
        return YState(
            wealth=self.wealth,
            love_support=self.love_support,
            pressure=self.pressure,
            happiness=self.happiness,
            stability=self.stability,
            trust=self.trust,
            resentment=self.resentment,
        )
