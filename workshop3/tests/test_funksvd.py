"""
Unit tests for the opt-in Workshop 1 FunkSVD baseline (`src/baselines.py`).

FunkSVD backs the Workshop 1 matrix factorization figures cited by the Workshop 3
report (e.g., Test RMSE 0.8041 for k=50) and supplies the Greedy FunkSVD policy
comparison. These tests use a small synthetic rating matrix so they run in seconds
without the raw MovieLens 32M CSVs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.baselines import FunkSVD, rmse


def _synthetic_ratings(n_users: int = 40, n_items: int = 30, rank: int = 2, seed: int = 0):
    """Generate a low-rank rating matrix with biases and observation noise."""
    rng = np.random.default_rng(seed)
    p = rng.normal(0.0, 0.6, size=(n_users, rank))
    q = rng.normal(0.0, 0.6, size=(n_items, rank))
    bu = rng.normal(0.0, 0.3, size=n_users)
    bi = rng.normal(0.0, 0.3, size=n_items)
    rows = []
    for u in range(n_users):
        for i in range(n_items):
            if rng.random() < 0.6:
                r = 3.5 + bu[u] + bi[i] + p[u] @ q[i] + rng.normal(0.0, 0.1)
                rows.append((u, i, float(np.clip(r, 0.5, 5.0))))
    df = pd.DataFrame(rows, columns=["u", "i", "rating"])
    perm = rng.permutation(len(df))
    n_test = len(df) // 5
    test_df = df.iloc[perm[:n_test]].reset_index(drop=True)
    train_df = df.iloc[perm[n_test:]].reset_index(drop=True)
    return train_df, test_df, n_users, n_items


def test_funksvd_beats_global_mean_on_held_out_ratings() -> None:
    train_df, test_df, n_users, n_items = _synthetic_ratings()
    model = FunkSVD(n_users, n_items, n_factors=4, lr=0.02, reg=0.02, n_epochs=40, seed=42)
    model.fit(train_df, verbose=False)

    y_test = test_df["rating"].to_numpy()
    rmse_mean = rmse(y_test, np.full_like(y_test, train_df["rating"].mean()))
    rmse_mf = rmse(y_test, np.clip(model.predict_df(test_df), 0.5, 5.0))
    assert rmse_mf < 0.8 * rmse_mean, (rmse_mf, rmse_mean)


def test_funksvd_score_candidates_matches_predict_df() -> None:
    train_df, _, n_users, n_items = _synthetic_ratings(seed=1)
    model = FunkSVD(n_users, n_items, n_factors=3, n_epochs=5, seed=7).fit(train_df, verbose=False)

    items = np.arange(n_items)
    scores = model.score_candidates(3, items)
    pairs = pd.DataFrame({"u": np.full(n_items, 3), "i": items})
    assert scores.shape == (n_items,)
    np.testing.assert_allclose(scores, model.predict_df(pairs), rtol=1e-5, atol=1e-5)


def test_funksvd_is_deterministic_for_fixed_seed() -> None:
    train_df, test_df, n_users, n_items = _synthetic_ratings(seed=2)
    preds = [
        FunkSVD(n_users, n_items, n_factors=3, n_epochs=5, seed=11)
        .fit(train_df, verbose=False)
        .predict_df(test_df)
        for _ in range(2)
    ]
    np.testing.assert_array_equal(preds[0], preds[1])
