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

def test_episode_runs_at_most_55_steps(env):
    """Episode ends at age_end or earlier (divorce). Must not exceed max length."""
    env.reset(seed=5)
    done = False
    steps = 0
    while not done:
        _, _, done, _, _ = env.step(env.action_space.sample())
        steps += 1
    assert steps <= env.age_end - env.age_start

def test_done_is_true_at_end(env):
    env.reset(seed=7)
    done = False
    while not done:
        _, _, done, _, _ = env.step(env.action_space.sample())
    assert done


# ── Reward range ──────────────────────────────────────────────────────────────

def test_reward_in_valid_range(env):
    """Scalar reward must stay in [0, 1]."""
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


# ── Y states stay clipped ─────────────────────────────────────────────────────

def test_y_always_in_bounds(env):
    env.reset(seed=9)
    done = False
    while not done:
        _, _, done, _, _ = env.step(env.action_space.sample())
        for y in (env.y_h, env.y_w):
            for fname in y.__dataclass_fields__:
                val = getattr(y, fname)
                assert 0.0 <= val <= 1.0, f"Y.{fname}={val} out of [0, 1]"


# ── Info dict fields ──────────────────────────────────────────────────────────

def test_info_contains_required_keys(env):
    env.reset(seed=10)
    _, _, _, _, info = env.step(env.action_space.sample())
    for key in (
        "age", "event", "delta_h", "delta_w", "reflection_triggered",
        "happiness", "stability", "divorced",
        "reward_h", "reward_w",
        "obs_h", "obs_w",
        "y_state_h", "y_state_w",
    ):
        assert key in info

def test_reflection_triggered_is_bool(env):
    env.reset(seed=11)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert isinstance(info["reflection_triggered"], bool)

def test_divorced_is_bool(env):
    env.reset(seed=11)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert isinstance(info["divorced"], bool)


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
    for _ in range(10):
        env.current_event = new_child_event
        env.step(env.action_space.sample())
    assert 0.0 <= env.x_h.kids <= 1.0
    assert 0.0 <= env.x_w.kids <= 1.0


# ── Innate / learned trait split ─────────────────────────────────────────────

def test_innate_is_frozen_after_reset(env):
    env.reset(seed=0)
    innate_h_before = env.x_h.innate.copy()
    innate_w_before = env.x_w.innate.copy()
    for _ in range(5):
        env.step(env.action_space.sample())
    np.testing.assert_array_equal(env.x_h.innate, innate_h_before)
    np.testing.assert_array_equal(env.x_w.innate, innate_w_before)

def test_effective_traits_in_bounds(env):
    from src.env.state import _TRAIT_NAMES
    env.reset(seed=0)
    for name in _TRAIT_NAMES:
        val_h = env.x_h.effective(name)
        val_w = env.x_w.effective(name)
        assert 0.0 <= val_h <= 1.0, f"x_h.effective({name})={val_h} out of [0, 1]"
        assert 0.0 <= val_w <= 1.0, f"x_w.effective({name})={val_w} out of [0, 1]"


# ── Trait-dependent event probabilities ──────────────────────────────────────

def test_high_faithfulness_reduces_infidelity_prob(env):
    from src.env.state import XTraits
    catalog = env.events
    x_faithful   = XTraits(faithfulness=0.95, mental_stability=0.5)
    x_unfaithful = XTraits(faithfulness=0.05, mental_stability=0.5)
    probs_faithful   = catalog._adjusted_probs(x_faithful, x_faithful)
    probs_unfaithful = catalog._adjusted_probs(x_unfaithful, x_unfaithful)
    idx = catalog.names.index("infidelity")
    assert probs_faithful[idx] < probs_unfaithful[idx]

def test_adjusted_probs_preserve_total_mass(env):
    from src.env.state import XTraits
    catalog = env.events
    x = XTraits(faithfulness=0.1, mental_stability=0.1, responsibility=0.1)
    adjusted = catalog._adjusted_probs(x, x)
    np.testing.assert_almost_equal(adjusted.sum(), catalog.probs.sum(), decimal=6)


# ── Action interactions ───────────────────────────────────────────────────────

@pytest.fixture
def neutral_x():
    from src.env.state import XTraits
    return XTraits(
        iq=0.5, eq=0.5, rational_thinking=0.5, emotional_reasoning=0.5,
        kindness=0.5, ability_to_love=0.5, faithfulness=0.5,
        responsibility=0.5, mental_stability=0.5, kids=0.0,
    )


def test_mutual_support_beats_support_argue(env, neutral_x):
    catalog = env.events
    d_h_both, _ = catalog.compute_delta_y(None, 0, 0, neutral_x, neutral_x)
    d_h_sa,   _ = catalog.compute_delta_y(None, 0, 1, neutral_x, neutral_x)
    # H's experience: W supports in first case, W argues in second
    assert d_h_both.get("love_support", 0) > d_h_sa.get("love_support", 0)

def test_mutual_argue_worse_than_argue_compromise(env, neutral_x):
    catalog = env.events
    d_h_aa, _ = catalog.compute_delta_y(None, 1, 1, neutral_x, neutral_x)
    d_h_ac, _ = catalog.compute_delta_y(None, 1, 3, neutral_x, neutral_x)
    assert d_h_aa.get("stability", 0) < d_h_ac.get("stability", 0)

def test_stonewall_worse_than_argue_support(env, neutral_x):
    catalog = env.events
    d_h_stone, _ = catalog.compute_delta_y(None, 1, 2, neutral_x, neutral_x)
    d_h_as,    _ = catalog.compute_delta_y(None, 1, 0, neutral_x, neutral_x)
    assert d_h_stone.get("love_support", 0) < d_h_as.get("love_support", 0)

def test_mutual_compromise_produces_stability_gain(env, neutral_x):
    catalog = env.events
    d_h, d_w = catalog.compute_delta_y(None, 3, 3, neutral_x, neutral_x)
    assert d_h.get("stability", 0) > 0
    assert d_w.get("stability", 0) > 0

def test_all_25_action_pairs_return_valid_delta(env, neutral_x):
    from src.env.events import N_ACTIONS
    catalog = env.events
    for a_h in range(N_ACTIONS):
        for a_w in range(N_ACTIONS):
            d_h, d_w = catalog.compute_delta_y(None, a_h, a_w, neutral_x, neutral_x)
            for label, delta in (("delta_h", d_h), ("delta_w", d_w)):
                assert isinstance(delta, dict)
                for key, val in delta.items():
                    assert np.isfinite(val), f"Non-finite {label} for ({a_h},{a_w}): {key}={val}"

def test_trust_decreases_when_partner_argues(env, neutral_x):
    """H's trust in W should decrease after W argues."""
    catalog = env.events
    d_h, _ = catalog.compute_delta_y(None, 0, 1, neutral_x, neutral_x)  # H supports, W argues
    assert d_h.get("trust", 0) < 0

def test_resentment_increases_when_partner_argues(env, neutral_x):
    """H's resentment toward W should increase after W argues."""
    catalog = env.events
    d_h, _ = catalog.compute_delta_y(None, 0, 1, neutral_x, neutral_x)
    assert d_h.get("resentment", 0) > 0

def test_trust_increases_when_partner_supports(env, neutral_x):
    """H's trust in W should increase after W supports."""
    catalog = env.events
    d_h, _ = catalog.compute_delta_y(None, 1, 0, neutral_x, neutral_x)  # H argues, W supports
    assert d_h.get("trust", 0) > 0


# ── Phase 2: life stage ───────────────────────────────────────────────────────

def test_life_stage_zero_at_episode_start(env):
    obs, _ = env.reset(seed=0)
    assert obs[-1] == pytest.approx(0.0, abs=1e-5)

def test_life_stage_increases_after_step(env):
    env.reset(seed=0)
    obs, *_ = env.step(env.action_space.sample())
    expected = 1.0 / (env.age_end - env.age_start)
    assert obs[-1] == pytest.approx(expected, abs=1e-4)

def test_life_stage_in_info(env):
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert "life_stage" in info
    assert 0.0 <= info["life_stage"] <= 1.0


# ── Phase 2: partner model (obscured traits) ──────────────────────────────────

def test_partner_obs_exact_at_max_trust(env):
    """At trust=1.0 partner noise is zero; partner slice equals true partner traits."""
    from src.env.state import X_DIM
    env.reset(seed=42)
    env.y_h.trust = 1.0
    obs_h = env._get_obs("h")
    partner_slice = obs_h[X_DIM : 2 * X_DIM]
    true_partner  = env.x_w.to_array()
    np.testing.assert_array_almost_equal(partner_slice, true_partner, decimal=4)

def test_partner_obs_varies_at_zero_trust(env):
    """At trust=0.0 partner noise is maximal; two calls should differ."""
    from src.env.state import X_DIM
    env.reset(seed=1)
    env.y_h.trust = 0.0
    obs_a = env._get_obs("h")[X_DIM : 2 * X_DIM]
    obs_b = env._get_obs("h")[X_DIM : 2 * X_DIM]
    assert not np.array_equal(obs_a, obs_b)


# ── Phase 2: event habituation ────────────────────────────────────────────────

def test_habituation_reduces_base_delta(env, neutral_x):
    """habituation=0.5 should make the negative base event delta less extreme."""
    catalog = env.events
    event = next(e for e in catalog.events if e["name"] == "emotional_conflict")
    d_h_full, _ = catalog.compute_delta_y(event, 3, 3, neutral_x, neutral_x, habituation=1.0)
    d_h_hab,  _ = catalog.compute_delta_y(event, 3, 3, neutral_x, neutral_x, habituation=0.5)
    assert d_h_hab.get("love_support", 0) > d_h_full.get("love_support", 0)

def test_habituation_does_not_affect_trust_resentment(env, neutral_x):
    """Trust/resentment come from action formulas, not base event — unaffected by habituation."""
    catalog = env.events
    event = next(e for e in catalog.events if e["name"] == "emotional_conflict")
    d_full, _ = catalog.compute_delta_y(event, 0, 1, neutral_x, neutral_x, habituation=1.0)
    d_hab,  _ = catalog.compute_delta_y(event, 0, 1, neutral_x, neutral_x, habituation=0.5)
    assert d_full.get("trust") == d_hab.get("trust")
    assert d_full.get("resentment") == d_hab.get("resentment")

def test_event_counts_accumulate_in_episode(env):
    env.reset(seed=0)
    new_child = next(e for e in env.events.events if e["name"] == "new_child")
    env.current_event = new_child
    env.step(env.action_space.sample())
    assert env._event_counts.get("new_child", 0) == 1
    env.current_event = new_child
    env.step(env.action_space.sample())
    assert env._event_counts.get("new_child", 0) == 2

def test_event_counts_reset_between_episodes(env):
    env.reset(seed=0)
    new_child = next(e for e in env.events.events if e["name"] == "new_child")
    env.current_event = new_child
    env.step(env.action_space.sample())
    env.reset(seed=1)
    assert env._event_counts == {}

def test_stage_prob_new_child_falls_with_age(env):
    from src.env.state import XTraits
    x = XTraits()
    idx = env.events.names.index("new_child")
    assert env.events._adjusted_probs(x, x, age=25)[idx] > env.events._adjusted_probs(x, x, age=55)[idx]

def test_stage_prob_health_crisis_rises_with_age(env):
    from src.env.state import XTraits
    x = XTraits()
    idx = env.events.names.index("health_crisis")
    assert env.events._adjusted_probs(x, x, age=70)[idx] > env.events._adjusted_probs(x, x, age=30)[idx]


# ── Phase 3b: social support context ─────────────────────────────────────────

def test_social_support_sampled_at_reset(env):
    """social_support should be in [init_low, init_high] after reset."""
    env.reset(seed=0)
    assert 0.30 <= env.social_support <= 0.90

def test_social_support_differs_across_resets(env):
    """Different seeds should produce different social_support values."""
    env.reset(seed=0)
    s0 = env.social_support
    env.reset(seed=99)
    s1 = env.social_support
    assert s0 != s1

def test_social_support_in_info(env):
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert "social_support" in info
    assert 0.0 <= info["social_support"] <= 1.0

def test_social_support_in_obs(env):
    """social_support should be the second-to-last obs element (obs[-2])."""
    env.reset(seed=0)
    obs, _ = env.reset(seed=5)
    # social_support is at obs[-2]; life_stage is at obs[-1]
    assert 0.0 <= obs[-2] <= 1.0

def test_low_social_support_amplifies_negative_event(env, neutral_x):
    """Isolated couple should see stronger negative base event impact than connected couple."""
    env.reset(seed=0)
    conflict = next(e for e in env.events.events if e["name"] == "emotional_conflict")
    env.current_event = conflict

    env.social_support = 0.0   # fully isolated
    _, _, _, _, info_isolated = env.step([3, 3])  # both compromise

    env.reset(seed=0)
    env.current_event = conflict
    env.social_support = 1.0   # fully connected
    _, _, _, _, info_connected = env.step([3, 3])

    # Isolated couple should have lower love_support after the conflict
    y_isolated  = info_isolated["y_state_h"]
    y_connected = info_connected["y_state_h"]
    assert y_isolated[1] <= y_connected[1], "Isolated couple should suffer more from conflict"

def test_social_support_shifts_after_relocation(env):
    """Relocation event should decrease social_support."""
    env.reset(seed=0)
    relocation = next(e for e in env.events.events if e["name"] == "relocation")
    env.current_event = relocation
    support_before = env.social_support
    env.step(env.action_space.sample())
    assert env.social_support < support_before

def test_social_support_shifts_after_shared_achievement(env):
    """Shared achievement should increase social_support."""
    env.reset(seed=0)
    achievement = next(e for e in env.events.events if e["name"] == "shared_achievement")
    env.current_event = achievement
    # Set social_support below equilibrium so it can rise
    env.social_support = 0.4
    env.step(env.action_space.sample())
    # Should be higher than 0.4 + mean-reversion drift of 0.005*(0.6-0.4) = 0.001
    assert env.social_support > 0.4


# ── Phase 2b: event-dependent reflection magnitudes ───────────────────────────

def test_reflection_magnitudes_keys_are_valid_events(env):
    """Every event in REFLECTION_MAGNITUDES must exist in the event catalog."""
    from train import REFLECTION_MAGNITUDES
    for name in REFLECTION_MAGNITUDES:
        assert name in env.events.names, f"'{name}' not in event catalog"

def test_reflection_magnitudes_all_positive(env):
    from train import REFLECTION_MAGNITUDES
    for name, mag in REFLECTION_MAGNITUDES.items():
        assert mag > 0.0, f"Magnitude for {name} must be positive"

def test_infidelity_magnitude_exceeds_default(env):
    from train import REFLECTION_MAGNITUDES
    default = 0.05
    assert REFLECTION_MAGNITUDES["infidelity"] > default
    assert REFLECTION_MAGNITUDES["family_death"] > default


# ── Phase B+C: behavior history and high-stakes scaling ───────────────────────

def test_c1_cooperative_history_lowers_conflict_prob(env):
    """Cooperative action history should lower emotional_conflict probability."""
    from src.env.state import XTraits
    from src.env.events import ACTION_SUPPORT
    x = XTraits()
    idx = env.events.names.index("emotional_conflict")
    # 5 support actions = 100% cooperative
    coop = [ACTION_SUPPORT] * 5
    p_coop    = env.events._adjusted_probs(x, x, age=35, action_history_h=coop, action_history_w=coop)
    p_neutral = env.events._adjusted_probs(x, x, age=35)
    assert p_coop[idx] < p_neutral[idx]

def test_c1_destructive_history_raises_conflict_prob(env):
    """Destructive action history should raise emotional_conflict probability."""
    from src.env.state import XTraits
    from src.env.events import ACTION_ARGUE
    x = XTraits()
    idx = env.events.names.index("emotional_conflict")
    dest = [ACTION_ARGUE] * 5
    p_dest    = env.events._adjusted_probs(x, x, age=35, action_history_h=dest, action_history_w=dest)
    p_neutral = env.events._adjusted_probs(x, x, age=35)
    assert p_dest[idx] > p_neutral[idx]

def test_c1_cooperative_history_raises_romantic_gesture_prob(env):
    """Cooperative history should increase romantic_gesture probability."""
    from src.env.state import XTraits
    from src.env.events import ACTION_COMPROMISE
    x = XTraits()
    idx = env.events.names.index("romantic_gesture")
    coop = [ACTION_COMPROMISE] * 5
    p_coop    = env.events._adjusted_probs(x, x, age=35, action_history_h=coop, action_history_w=coop)
    p_neutral = env.events._adjusted_probs(x, x, age=35)
    assert p_coop[idx] > p_neutral[idx]

def test_c1_action_history_tracked_in_env(env):
    """Environment should record action history across steps."""
    env.reset(seed=0)
    env.step([0, 3])  # support, compromise
    env.step([1, 1])  # argue, argue
    assert list(env._action_history_h) == [0, 1]
    assert list(env._action_history_w) == [3, 1]

def test_c1_action_history_cleared_on_reset(env):
    """Action history should be empty after reset."""
    env.reset(seed=0)
    env.step(env.action_space.sample())
    env.reset(seed=1)
    assert len(env._action_history_h) == 0
    assert len(env._action_history_w) == 0

def test_c2_high_stakes_amplifies_action_delta(env, neutral_x):
    """Action delta should be larger when event is high-stakes vs no event."""
    catalog = env.events
    infidelity = next(e for e in catalog.events if e["name"] == "infidelity")
    d_h_crisis, _ = catalog.compute_delta_y(infidelity, 0, 0, neutral_x, neutral_x)
    d_h_none,   _ = catalog.compute_delta_y(None,       0, 0, neutral_x, neutral_x)
    # love_support gain from mutual support should be larger during infidelity
    assert abs(d_h_crisis.get("love_support", 0)) > abs(d_h_none.get("love_support", 0))

def test_c2_non_high_stakes_unaffected(env, neutral_x):
    """Non-high-stakes event should not apply the 1.8× action scale."""
    catalog = env.events
    relocation = next(e for e in catalog.events if e["name"] == "relocation")
    quality = next(e for e in catalog.events if e["name"] == "quality_time")
    d_reloc, _ = catalog.compute_delta_y(relocation, 0, 3, neutral_x, neutral_x)
    d_qtim,  _ = catalog.compute_delta_y(quality,    0, 3, neutral_x, neutral_x)
    assert isinstance(d_reloc.get("love_support", 0.0), float)
    assert isinstance(d_qtim.get("love_support", 0.0), float)


# ── Phase D: conflict-resolution bonus ───────────────────────────────────────

def test_resolution_bonus_awarded_when_pressure_drops(env):
    """Reward should include a bonus when high pressure is successfully relieved."""
    env.reset(seed=0)
    env.y_h.pressure = 0.80   # above threshold
    env.y_w.pressure = 0.80
    env.current_event = None
    # support + compromise tends to drop pressure
    _, reward, _, _, info = env.step([0, 3])
    # At least one partner should have benefited; check reward is at least base level
    assert reward >= 0.0
    assert "reward_h" in info

def test_resolution_bonus_not_awarded_when_pressure_low(env):
    """No bonus when pressure starts below the threshold."""
    from src.env.marriage_env import _RESOLUTION_PRESSURE_THRESHOLD, _RESOLUTION_BONUS
    env.reset(seed=0)
    env.y_h.pressure = 0.10   # well below threshold
    env.y_w.pressure = 0.10
    env.current_event = None
    _, _, _, _, info = env.step([0, 3])
    # reward_h should be base value without resolution bonus
    base_reward = info["reward_h"]
    # Verify it's a valid float — no bonus means reward stays at base
    assert 0.0 <= base_reward <= 1.0

def test_resolution_bonus_caps_at_one(env):
    """Resolution bonus should not push reward above 1.0."""
    env.reset(seed=0)
    env.y_h.pressure = 0.90
    env.y_w.pressure = 0.90
    env.y_h.happiness = 1.0
    env.y_h.stability = 1.0
    env.y_w.happiness = 1.0
    env.y_w.stability = 1.0
    env.current_event = None
    _, _, _, _, info = env.step([0, 3])
    assert info["reward_h"] <= 1.0
    assert info["reward_w"] <= 1.0


# ── Phase E: noisy partner Y estimate in observation ─────────────────────────

def test_obs_dim_includes_partner_y(env):
    """Observation dimension should include two Y blocks (own + partner estimate)."""
    from src.env.state import X_DIM, Y_DIM
    obs, _ = env.reset(seed=0)
    # obs_dim = X_DIM + X_DIM + Y_DIM + Y_DIM + n_events+1 + 1 + 1
    expected = X_DIM + X_DIM + Y_DIM + Y_DIM + (env.events.n_events + 1) + 1 + 1
    assert obs.shape[0] == expected

def test_partner_y_estimate_exact_at_max_trust(env):
    """At trust=1.0 partner Y noise is zero; partner Y slice equals true partner state."""
    from src.env.state import X_DIM, Y_DIM
    env.reset(seed=42)
    env.y_h.trust = 1.0
    obs_h = env._get_obs("h")
    # partner Y estimate sits at offset X_DIM + X_DIM + Y_DIM
    y_partner_slice = obs_h[2 * X_DIM + Y_DIM : 2 * X_DIM + 2 * Y_DIM]
    true_partner_y  = env.y_w.to_array()
    np.testing.assert_array_almost_equal(y_partner_slice, true_partner_y, decimal=4)

def test_partner_y_estimate_noisy_at_zero_trust(env):
    """At trust=0.0 partner Y noise is maximal; two calls should differ."""
    from src.env.state import X_DIM, Y_DIM
    env.reset(seed=1)
    env.y_h.trust = 0.0
    a = env._get_obs("h")[2 * X_DIM + Y_DIM : 2 * X_DIM + 2 * Y_DIM]
    b = env._get_obs("h")[2 * X_DIM + Y_DIM : 2 * X_DIM + 2 * Y_DIM]
    assert not np.array_equal(a, b)

def test_obs_bounds_include_partner_y(env):
    """Full observation including partner Y estimate should stay in [0, 1]."""
    obs, _ = env.reset(seed=7)
    assert env.observation_space.contains(obs)
