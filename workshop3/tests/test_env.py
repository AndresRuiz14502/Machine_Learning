"""
Unit tests for Workshop 3 MovieLensRecEnv (Gymnasium API & MDP invariants).
"""

from __future__ import annotations

import numpy as np
import pytest

from workshop3.src.env import MovieLensRecEnv


def _first_user(env: MovieLensRecEnv, label: int, count: int) -> int:
    """Return the first split user whose slate holds at least `count` slots with `label`."""
    for u in env.split_user_indices:
        if int(np.sum(env.cand_is_liked[int(u)] == label)) >= count:
            return int(u)
    raise AssertionError(f"No user with {count} slots labelled {label}")


def test_env_spaces_and_reset() -> None:
    env = MovieLensRecEnv(split="train", horizon=10, seed=42)
    assert env.action_space.n == 50
    assert env.observation_space.shape == (250,)
    assert env.observation_space.dtype == np.float32

    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (250,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert "valid_mask" in info
    assert info["valid_mask"].shape == (50,)
    # Only the session's real ratings are available; padding slots start masked
    np.testing.assert_array_equal(info["valid_mask"], env.cand_valid[info["user_idx"]])
    assert info["num_valid_in_pool"] >= env.horizon
    assert isinstance(info["user_idx"], int)


def test_env_step_and_visited_mask() -> None:
    env = MovieLensRecEnv(split="val", horizon=10, seed=123)
    obs, info = env.reset(seed=123)
    n_valid = info["num_valid_in_pool"]

    obs1, reward1, terminated1, truncated1, info1 = env.step(0)
    assert obs1.shape == (250,)
    assert env.observation_space.contains(obs1)
    assert isinstance(reward1, float)
    assert np.isfinite(reward1)
    assert isinstance(terminated1, bool)
    assert isinstance(truncated1, bool)

    # Action 0 should now be marked visited in state [150:200] and invalid in valid_mask
    assert obs1[150 + 0] == pytest.approx(1.0)
    assert info1["valid_mask"][0] == pytest.approx(0.0)
    assert np.sum(info1["valid_mask"]) == pytest.approx(n_valid - 1)


def test_repeated_action_penalty() -> None:
    env = MovieLensRecEnv(split="test", horizon=10, seed=99)
    env.reset(seed=99)

    env.step(5)
    # Repeating action 5 must yield -1.5 penalty and is_hit = 0
    obs2, reward2, terminated2, truncated2, info2 = env.step(5)
    assert reward2 == pytest.approx(-1.5)
    assert info2["is_hit"] == 0
    assert env.observation_space.contains(obs2)


def test_padding_slot_is_penalized_like_a_repeat() -> None:
    env = MovieLensRecEnv(split="test", horizon=10, seed=3)
    padded = [int(u) for u in env.split_user_indices if env.cand_valid[int(u)].sum() < env.num_candidates]
    assert padded, "expected at least one test session shorter than 50 ratings"
    env.reset(seed=3, options={"user_idx": padded[0]})
    pad_slot = int(np.flatnonzero(env.cand_valid[padded[0]] < 0.5)[0])

    _, reward, _, _, info = env.step(pad_slot)
    assert reward == pytest.approx(-1.5)
    assert info["is_hit"] == 0
    assert info["is_valid_slot"] == 0


def test_early_termination_on_five_consecutive_misses() -> None:
    env = MovieLensRecEnv(split="train", horizon=10, seed=7)
    u = _first_user(env, label=-1, count=5)
    env.reset(seed=7, options={"user_idx": u})

    # Five real dislikes (rating < 4.0) in a row trigger user abandonment
    disliked_actions = np.where(env.cand_is_liked[u] == -1)[0][:5]
    for idx, act in enumerate(disliked_actions):
        obs, reward, terminated, truncated, info = env.step(int(act))
        assert reward == pytest.approx(-0.4)
        if idx < 4:
            assert not terminated
            assert not truncated
        else:
            assert terminated is True
            assert env.t == 5
            assert info["consecutive_misses"] == 5


def test_full_horizon_truncation_and_render() -> None:
    env = MovieLensRecEnv(split="test", horizon=10, seed=42)
    u = _first_user(env, label=1, count=10)
    env.reset(seed=42, options={"user_idx": u})

    # Pick 10 liked actions so episode never terminates early
    liked_actions = np.where(env.cand_is_liked[u] == 1)[0][:10]
    for idx, act in enumerate(liked_actions):
        obs, reward, terminated, truncated, info = env.step(int(act))
        assert info["is_hit"] == 1
        assert reward >= 1.0
        if idx < 9:
            assert not terminated
            assert not truncated
        else:
            assert not terminated
            assert truncated is True

    rendered = env.render()
    assert "MovieLensRecEnv Trajectory" in rendered
    assert "HIT" in rendered
