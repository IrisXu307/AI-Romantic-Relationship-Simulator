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
    env.reset(seed=8)
    done = False
    while not done:
        _, reward, done, _, _ = env.step(env.action_space.sample())
        assert 0.0 <= reward <= 1.0


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
    for key in ("age", "event", "delta_y", "reflection_triggered", "happiness", "stability"):
        assert key in info

def test_reflection_triggered_is_bool(env):
    env.reset(seed=11)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert isinstance(info["reflection_triggered"], bool)


# ── Observation changes after step ───────────────────────────────────────────

def test_obs_changes_after_step(env):
    obs0, _ = env.reset(seed=12)
    obs1, *_ = env.step(env.action_space.sample())
    # At minimum the age-dependent portion (event one-hot) will differ
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
    # Manually inject the new_child event
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
