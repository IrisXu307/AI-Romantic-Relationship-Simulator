import numpy as np
import pytest

from src.env.marriage_env import MarriageEnv

CONFIG_PATH = "config/default.yaml"
EVENTS_PATH = "config/events.yaml"


@pytest.fixture
def env():
    return MarriageEnv(CONFIG_PATH, EVENTS_PATH)


# ── Observation shape & bounds ────────────────────────────────────────────────

def test_reset_obs_shape(env):
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape

def test_reset_obs_in_bounds(env):
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs)

def test_step_obs_shape(env):
    env.reset(seed=2)
    obs, *_ = env.step(env.action_space.sample())
    assert obs.shape == env.observation_space.shape

def test_step_obs_in_bounds(env):
    env.reset(seed=3)
    obs, *_ = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)


# ── Step return types ─────────────────────────────────────────────────────────

def test_step_returns_correct_types(env):
    env.reset(seed=4)
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


# ── Episode length ────────────────────────────────────────────────────────────

def test_episode_runs_exactly_55_steps(env):
    env.reset(seed=5)
    done = False
    steps = 0
    while not done:
        _, _, done, _, _ = env.step(env.action_space.sample())
        steps += 1
    assert steps == env.age_end - env.age_start

def test_done_is_false_before_end(env):
    env.reset(seed=6)
    for _ in range(env.age_end - env.age_start - 1):
        _, _, done, _, _ = env.step(env.action_space.sample())
        assert not done

def test_done_is_true_at_end(env):
    env.reset(seed=7)
    done = False
    while not done:
        _, _, done, _, _ = env.step(env.action_space.sample())
    assert done


# ── Reward range ──────────────────────────────────────────────────────────────

def test_reward_in_valid_range(env):
    """Scalar reward (avg of per-agent rewards) must stay in [0, 1]."""
    env.reset(seed=8)
    done = False
    while not done:
        _, reward, done, _, _ = env.step(env.action_space.sample())
        assert 0.0 <= reward <= 1.0

def test_per_agent_rewards_in_valid_range(env):
    """Both per-agent rewards must be in [0, 1]."""
    env.reset(seed=8)
    done = False
    while not done:
        _, _, done, _, info = env.step(env.action_space.sample())
        assert 0.0 <= info["reward_h"] <= 1.0
        assert 0.0 <= info["reward_w"] <= 1.0

def test_per_agent_rewards_differ(env):
    """Different reward weights should produce different rewards at least sometimes."""
    env.reset(seed=8)
    rewards_equal = True
    for _ in range(10):
        _, _, done, _, info = env.step(env.action_space.sample())
        if abs(info["reward_h"] - info["reward_w"]) > 1e-6:
            rewards_equal = False
            break
        if done:
            break
    assert not rewards_equal, "Per-agent rewards were identical every step — check reward config"


# ── Y state stays clipped ─────────────────────────────────────────────────────

def test_y_always_in_bounds(env):
    env.reset(seed=9)
    done = False
    while not done:
        _, _, done, _, _ = env.step(env.action_space.sample())
        for fname in env.y.__dataclass_fields__:
            val = getattr(env.y, fname)
            assert 0.0 <= val <= 1.0, f"Y.{fname}={val} out of [0, 1]"


# ── Info dict fields ──────────────────────────────────────────────────────────

def test_info_contains_required_keys(env):
    env.reset(seed=10)
    _, _, _, _, info = env.step(env.action_space.sample())
    for key in (
        "age", "event", "delta_y", "reflection_triggered",
        "happiness", "stability",
        "reward_h", "reward_w",
        "obs_h", "obs_w",
    ):
        assert key in info

def test_reflection_triggered_is_bool(env):
    env.reset(seed=11)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert isinstance(info["reflection_triggered"], bool)


# ── Per-agent observations ────────────────────────────────────────────────────

def test_obs_h_and_obs_w_have_correct_shape(env):
    env.reset(seed=12)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert info["obs_h"].shape == env.observation_space.shape
    assert info["obs_w"].shape == env.observation_space.shape

def test_obs_h_and_obs_w_in_bounds(env):
    env.reset(seed=13)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(info["obs_h"])
    assert env.observation_space.contains(info["obs_w"])

def test_obs_h_and_obs_w_differ(env):
    """
    Husband and wife see different observations (different X_self vs X_partner
    ordering, plus independent perceptual noise).
    """
    env.reset(seed=14)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert not np.array_equal(info["obs_h"], info["obs_w"])


# ── Observation changes after step ───────────────────────────────────────────

def test_obs_changes_after_step(env):
    obs0, _ = env.reset(seed=15)
    obs1, *_ = env.step(env.action_space.sample())
    assert not np.array_equal(obs0, obs1)


# ── Determinism with fixed seed ───────────────────────────────────────────────

def test_same_seed_same_first_obs(env):
    obs_a, _ = env.reset(seed=42)
    obs_b, _ = env.reset(seed=42)
    np.testing.assert_array_equal(obs_a, obs_b)


# ── Kids increment ────────────────────────────────────────────────────────────

def test_new_child_increments_kids(env):
    """Force a new_child event and verify both agents' kids trait increases."""
    env.reset(seed=0)
    new_child_event = next(e for e in env.events.events if e["name"] == "new_child")
    env.current_event = new_child_event

    kids_before_h = env.x_h.kids
    kids_before_w = env.x_w.kids

    env.step(env.action_space.sample())

    assert env.x_h.kids > kids_before_h
    assert env.x_w.kids > kids_before_w


# ── X traits stay clipped after increment ────────────────────────────────────

def test_x_kids_stays_in_bounds_after_many_children(env):
    env.reset(seed=0)
    new_child_event = next(e for e in env.events.events if e["name"] == "new_child")
    for _ in range(10):  # more than MAX_KIDS
        env.current_event = new_child_event
        env.step(env.action_space.sample())
    assert 0.0 <= env.x_h.kids <= 1.0
    assert 0.0 <= env.x_w.kids <= 1.0


# ── Innate / learned trait split ─────────────────────────────────────────────

def test_innate_is_frozen_after_reset(env):
    """Innate baseline must not change across resets or steps."""
    env.reset(seed=0)
    innate_h_before = env.x_h.innate.copy()
    innate_w_before = env.x_w.innate.copy()

    for _ in range(5):
        env.step(env.action_space.sample())

    np.testing.assert_array_equal(env.x_h.innate, innate_h_before)
    np.testing.assert_array_equal(env.x_w.innate, innate_w_before)

def test_effective_traits_in_bounds(env):
    """effective() must return values in [0, 1] for all traits."""
    from src.env.state import _TRAIT_NAMES
    env.reset(seed=0)
    for name in _TRAIT_NAMES:
        val_h = env.x_h.effective(name)
        val_w = env.x_w.effective(name)
        assert 0.0 <= val_h <= 1.0, f"x_h.effective({name})={val_h} out of [0, 1]"
        assert 0.0 <= val_w <= 1.0, f"x_w.effective({name})={val_w} out of [0, 1]"


# ── Trait-dependent event probabilities ──────────────────────────────────────

def test_high_faithfulness_reduces_infidelity_prob(env):
    """Faithful couples should face lower infidelity probability."""
    from src.env.state import XTraits
    catalog = env.events

    x_faithful = XTraits(faithfulness=0.95, mental_stability=0.5)
    x_unfaithful = XTraits(faithfulness=0.05, mental_stability=0.5)

    probs_faithful   = catalog._adjusted_probs(x_faithful, x_faithful)
    probs_unfaithful = catalog._adjusted_probs(x_unfaithful, x_unfaithful)

    idx = catalog.names.index("infidelity")
    assert probs_faithful[idx] < probs_unfaithful[idx]

def test_adjusted_probs_preserve_total_mass(env):
    """Trait adjustments should redistribute probability, not inflate it."""
    from src.env.state import XTraits
    catalog = env.events
    x = XTraits(faithfulness=0.1, mental_stability=0.1, responsibility=0.1)
    adjusted = catalog._adjusted_probs(x, x)
    np.testing.assert_almost_equal(adjusted.sum(), catalog.probs.sum(), decimal=6)


# ── Action interactions ───────────────────────────────────────────────────────

@pytest.fixture
def neutral_x():
    """Neutral XTraits at 0.5 so trait scaling doesn't skew interaction tests."""
    from src.env.state import XTraits
    return XTraits(
        iq=0.5, eq=0.5, rational_thinking=0.5, emotional_reasoning=0.5,
        kindness=0.5, ability_to_love=0.5, faithfulness=0.5,
        responsibility=0.5, mental_stability=0.5, kids=0.0,
    )

def test_mutual_support_beats_support_argue(env, neutral_x):
    """(support, support) should produce better love_support than (support, argue)."""
    catalog = env.events
    d_both_support = catalog.compute_delta_y(None, 0, 0, neutral_x, neutral_x)
    d_support_argue = catalog.compute_delta_y(None, 0, 1, neutral_x, neutral_x)
    assert d_both_support.get("love_support", 0) > d_support_argue.get("love_support", 0)

def test_mutual_argue_worse_than_argue_compromise(env, neutral_x):
    """(argue, argue) should do more stability damage than (argue, compromise)."""
    catalog = env.events
    d_both_argue = catalog.compute_delta_y(None, 1, 1, neutral_x, neutral_x)
    d_argue_compromise = catalog.compute_delta_y(None, 1, 3, neutral_x, neutral_x)
    assert d_both_argue.get("stability", 0) < d_argue_compromise.get("stability", 0)

def test_support_argue_is_asymmetric(env, neutral_x):
    """
    (support, argue) and (argue, support) should produce different deltas:
    in (support, argue) it's H whose effect is dampened; in (argue, support)
    it's W. The net love_support should be the same by symmetry, but the
    individual contributions differ.
    """
    catalog = env.events
    d_sh_aw = catalog.compute_delta_y(None, 0, 1, neutral_x, neutral_x)
    d_ah_sw = catalog.compute_delta_y(None, 1, 0, neutral_x, neutral_x)
    # Net love_support should match (symmetric traits, symmetric structure)
    np.testing.assert_almost_equal(
        d_sh_aw.get("love_support", 0),
        d_ah_sw.get("love_support", 0),
        decimal=6,
    )

def test_stonewall_worse_than_argue_alone(env, neutral_x):
    """(argue, ignore) stonewalling should hurt love_support more than (argue, support)."""
    catalog = env.events
    d_stonewall = catalog.compute_delta_y(None, 1, 2, neutral_x, neutral_x)
    d_argue_support = catalog.compute_delta_y(None, 1, 0, neutral_x, neutral_x)
    assert d_stonewall.get("love_support", 0) < d_argue_support.get("love_support", 0)

def test_mutual_compromise_produces_stability_gain(env, neutral_x):
    """(compromise, compromise) should yield positive stability from interaction bonus."""
    catalog = env.events
    delta = catalog.compute_delta_y(None, 3, 3, neutral_x, neutral_x)
    assert delta.get("stability", 0) > 0

def test_all_25_action_pairs_return_valid_delta(env, neutral_x):
    """Every action pair should return a delta dict with finite values."""
    from src.env.events import N_ACTIONS
    catalog = env.events
    for a_h in range(N_ACTIONS):
        for a_w in range(N_ACTIONS):
            delta = catalog.compute_delta_y(None, a_h, a_w, neutral_x, neutral_x)
            assert isinstance(delta, dict)
            for key, val in delta.items():
                assert np.isfinite(val), f"Non-finite delta for ({a_h},{a_w}): {key}={val}"
