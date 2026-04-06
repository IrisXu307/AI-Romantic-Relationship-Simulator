from dataclasses import dataclass, fields
import numpy as np

# Dimensionality constants used throughout the codebase
X_DIM = 10
Y_DIM = 5


@dataclass
class XTraits:
    """
    Internal trait vector for one agent.
    All values are in [0, 1].
    These change only during Reflection.
    """
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

    MAX_KIDS: float = 5.0  # class-level constant, not a field

    def to_array(self) -> np.ndarray:
        return np.array([
            self.iq, self.eq, self.rational_thinking, self.emotional_reasoning,
            self.kindness, self.ability_to_love, self.faithfulness,
            self.responsibility, self.mental_stability, self.kids,
        ], dtype=np.float32)

    def clip(self):
        """Clip all trait values to [0, 1]."""
        for f in fields(self):
            setattr(self, f.name, float(np.clip(getattr(self, f.name), 0.0, 1.0)))

    def increment_kids(self):
        """Increase the normalized kids count by one child, capped at MAX_KIDS."""
        raw = round(self.kids * self.MAX_KIDS) + 1
        self.kids = float(np.clip(raw / self.MAX_KIDS, 0.0, 1.0))


@dataclass
class YState:
    """
    Shared environmental state for the couple.
    All values are in [0, 1].
    happiness and stability are the primary reward signals.
    """
    wealth: float = 0.5
    love_support: float = 0.7
    pressure: float = 0.3
    happiness: float = 0.7
    stability: float = 0.8

    def to_array(self) -> np.ndarray:
        return np.array([
            self.wealth, self.love_support, self.pressure,
            self.happiness, self.stability,
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
        )
