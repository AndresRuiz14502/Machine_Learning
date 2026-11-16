"""
Leakage guards for the Workshop 3 simulator.

The observation must be built only from training-split statistics and observed feedback.
These tests recompute every candidate feature from the stored train-split FunkSVD parameters,
check the temporal separation of RL sessions from the training data, and bound how well the
initial observation separates liked from disliked slots (an earlier version of the simulator
mixed the hidden label into the candidate vectors, reaching AUC = 1.0).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from workshop3.src.env import MovieLensRecEnv, resolve_repo_path

CACHE = resolve_repo_path("workshop3/data/processed_cache.npz")


@pytest.fixture(scope="module")
def cache() -> dict:
    with np.load(CACHE) as data:
        return {key: data[key] for key in data.files}


def test_candidate_factors_are_train_split_funksvd_factors(cache: dict) -> None:
    valid = cache["cand_valid"].astype(bool)
    q = cache["item_factors"][cache["cand_items"][valid]]
    q_unit = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    np.testing.assert_allclose(cache["cand_item_factors"][valid], q_unit, atol=1e-6)
    assert np.all(cache["cand_item_factors"][~valid] == 0.0)


def test_static_scores_are_funksvd_predictions(cache: dict) -> None:
    rows, slots = np.nonzero(cache["cand_valid"])
    items = cache["cand_items"][rows, slots]
    pred = (
        float(cache["global_mean"])
        + cache["user_biases"][rows]
        + cache["item_biases"][items]
        + np.sum(cache["user_factors"][rows] * cache["item_factors"][items], axis=1)
    )
    np.testing.assert_allclose(cache["cand_svd_pred"][rows, slots], pred, rtol=1e-5, atol=1e-4)

    # Slots are ordered by the static score only (never by the hidden rating)
    for r in range(len(cache["cand_valid"])):
        n = int(cache["cand_valid"][r].sum())
        assert np.all(np.diff(cache["cand_svd_pred"][r, :n]) <= 1e-6)


def test_labels_are_real_ratings(cache: dict) -> None:
    valid = cache["cand_valid"].astype(bool)
    ratings = cache["cand_true_ratings"][valid]
    assert np.all(np.isin(ratings, np.arange(0.5, 5.01, 0.5)))
    liked = cache["cand_is_liked"][valid]
    np.testing.assert_array_equal(liked == 1, ratings >= 4.0)
    np.testing.assert_array_equal(liked == -1, ratings < 4.0)
    assert np.all(cache["cand_items"][~valid] == -1)
    assert np.all(cache["cand_is_liked"][~valid] == 0)


def test_sessions_come_after_the_training_cut(cache: dict) -> None:
    train_end = int(cache["train_end_timestamp"])
    val_end = int(cache["val_end_timestamp"])
    valid = cache["cand_valid"].astype(bool)
    ts = cache["cand_timestamps"]
    for key, lo, hi in (
        ("train_user_indices", train_end, val_end),
        ("val_user_indices", train_end, val_end),
        ("test_user_indices", val_end, np.iinfo(np.int64).max),
    ):
        rows = cache[key]
        stamps = ts[rows][valid[rows]]
        assert stamps.min() >= lo and stamps.max() <= hi, key
    assert np.all(cache["pool_period"][cache["test_user_indices"]] == 1)
    assert set(cache["train_user_indices"]).isdisjoint(cache["val_user_indices"])


def test_reset_starts_from_the_user_factor_only() -> None:
    env = MovieLensRecEnv(split="test", horizon=10, seed=0)
    assert not hasattr(env, "u_target_vecs")
    for seed in range(5):
        env.reset(seed=seed)
        np.testing.assert_allclose(env.session_vec, env.user_factors_norm[env.u], atol=1e-6)


def test_initial_observation_does_not_reveal_the_label() -> None:
    env = MovieLensRecEnv(split="test", horizon=10, seed=0)
    users = [int(u) for u in env.split_user_indices]
    valid = env.cand_valid[users] > 0.5
    liked = (env.cand_is_liked[users] == 1)[valid]
    sims, steps, hits = [], 0, 0
    for u in users:
        obs, info = env.reset(seed=0, options={"user_idx": u})
        sims.append(obs[0:50][env.cand_valid[u] > 0.5])
        done = False
        while not done:  # greedy on the session cosine, the feature that leaked before
            scores = np.where(info["valid_mask"] > 0.5, obs[0:50], -np.inf)
            obs, _, term, trunc, info = env.step(int(np.argmax(scores)))
            steps += 1
            hits += info["is_hit"]
            done = term or trunc
    auc = roc_auc_score(liked, np.concatenate(sims))
    assert auc < 0.75, f"session cosine separates liked slots too well (AUC={auc:.3f})"
    assert hits / steps < 0.8, f"cosine-greedy hit rate {hits / steps:.3f} suggests label leakage"
