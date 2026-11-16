"""
Data preparation pipeline for Workshop 3:
Reinforcement Learning for Sequential Movie Recommendation on MovieLens 32M.

Replicates the exact Workshop 1 sampling and global chronological split (seed=42,
80/10/10 by timestamp), trains a vectorized FunkSVD model (k=20, lr=0.01, reg=0.02,
15 epochs) on the training split only, and builds leak-free RL sessions from real
feedback recorded after the training cut:

- RL train/val sessions: clean ratings of warm users inside the validation period
  (users split 80/20 at random between RL train and RL val).
- RL test sessions: clean ratings of warm users inside the test period, i.e. the same
  held-out future that Workshops 1 and 2 report on.

A session's candidate slate holds its first 50 real ratings in chronological order
(selection never looks at the ratings). Every observation feature (latent factors,
biases, FunkSVD scores, popularity, novelty) is computed from the training split
only; ratings and like labels are used exclusively by the environment to emit rewards
and by the evaluation metrics.

Raw CSVs (ratings.csv, movies.csv) are read from <repo>/dataSet/ when present,
with workshop3/data/ml-32m/ml-32m/ as fallback; outputs go to workshop3/data/.
Default paths are resolved against the repository root, so the script can be
launched with any working directory.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# <repo>/workshop3/src/data_prep.py -> parents[2] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
W3_DATA_DIR = REPO_ROOT / "workshop3" / "data"

# Folders searched for the raw MovieLens 32M CSVs, in priority order: the
# repository-level dataSet/ folder shared with Workshops 1-2, then the W3 fallback.
RAW_DATA_DIRS = (
    REPO_ROOT / "dataSet",
    W3_DATA_DIR / "ml-32m" / "ml-32m",
)

LIKE_THRESHOLD = 4.0
NUM_CANDIDATE_SLOTS = 50
MIN_SESSION_RATINGS = 20
RL_VAL_USER_FRACTION = 0.2

ALL_GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def resolve_raw_csv(filename: str) -> Path:
    """Return the first existing raw MovieLens 32M ``filename`` among ``RAW_DATA_DIRS``."""
    candidates = [directory / filename for directory in RAW_DATA_DIRS]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"[data_prep] {filename} not found. Searched: {searched}")


def load_and_filter_movielens(
    ratings_path: str | Path | None = None,
    movies_path: str | Path | None = None,
    n_sample_users: int = 15000,
    min_interactions: int = 10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Load MovieLens 32M, sample 15,000 complete users (seed=42), and iteratively filter <10 ratings.

    ``ratings_path``/``movies_path`` default to ``resolve_raw_csv("ratings.csv")`` and
    ``resolve_raw_csv("movies.csv")``.
    """
    ratings_path = resolve_raw_csv("ratings.csv") if ratings_path is None else ratings_path
    movies_path = resolve_raw_csv("movies.csv") if movies_path is None else movies_path
    t0 = time.time()
    print(f"[data_prep] Loading ratings from {ratings_path} ...")
    df = pd.read_csv(
        ratings_path,
        dtype={
            "userId": np.int32,
            "movieId": np.int32,
            "rating": np.float32,
            "timestamp": np.int64,
        },
    )
    total_raw_ratings = int(len(df))

    rng = np.random.default_rng(seed)
    unique_users = df["userId"].unique()
    total_raw_users = int(len(unique_users))
    sampled_users = rng.choice(unique_users, size=n_sample_users, replace=False)

    sub = df[df["userId"].isin(sampled_users)].copy()
    sampled_ratings_count = int(len(sub))
    print(
        f"[data_prep] Sampled {n_sample_users:,} users -> {sampled_ratings_count:,} ratings "
        f"({time.time() - t0:.2f}s)"
    )

    # Iterative k-core filtering (< min_interactions)
    while True:
        n_before = len(sub)
        u_counts = sub["userId"].value_counts()
        valid_u = u_counts[u_counts >= min_interactions].index
        sub = sub[sub["userId"].isin(valid_u)]

        m_counts = sub["movieId"].value_counts()
        valid_m = m_counts[m_counts >= min_interactions].index
        sub = sub[sub["movieId"].isin(valid_m)]

        if len(sub) == n_before:
            break

    sub = sub.copy()
    # Map userId and movieId to contiguous 0..n_u-1 and 0..n_i-1
    sorted_users = np.sort(sub["userId"].unique())
    sorted_movies = np.sort(sub["movieId"].unique())

    user_to_idx = pd.Series(
        np.arange(len(sorted_users), dtype=np.int32), index=sorted_users
    )
    movie_to_idx = pd.Series(
        np.arange(len(sorted_movies), dtype=np.int32), index=sorted_movies
    )

    sub["u_idx"] = sub["userId"].map(user_to_idx).astype(np.int32)
    sub["i_idx"] = sub["movieId"].map(movie_to_idx).astype(np.int32)

    # Sort temporally (mergesort preserves deterministic ordering on ties)
    sub = sub.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    movies_df = pd.read_csv(movies_path)
    movies_sub = (
        movies_df[movies_df["movieId"].isin(sorted_movies)]
        .copy()
        .assign(i_idx=lambda x: x["movieId"].map(movie_to_idx).astype(np.int32))
        .sort_values("i_idx")
        .reset_index(drop=True)
    )

    stats = {
        "total_raw_ratings": total_raw_ratings,
        "total_raw_users": total_raw_users,
        "sampled_users_target": n_sample_users,
        "sampled_ratings_raw": sampled_ratings_count,
        "filtered_ratings": int(len(sub)),
        "filtered_users": int(len(sorted_users)),
        "filtered_movies": int(len(sorted_movies)),
        "like_ratio": float((sub["rating"] >= 4.0).mean()),
    }
    print(
        f"[data_prep] After iterative >=10 filter: {stats['filtered_ratings']:,} ratings, "
        f"{stats['filtered_users']:,} users, {stats['filtered_movies']:,} movies."
    )
    return sub, movies_sub, stats


def parse_genres_and_lookup(
    movies_sub: pd.DataFrame, n_i: int
) -> Tuple[np.ndarray, Dict[str, Dict[str, object]]]:
    """Build multi-hot genre matrix (n_i, 19) and movie lookup dictionary."""
    genre_to_col = {g: idx for idx, g in enumerate(ALL_GENRES)}
    item_genres = np.zeros((n_i, len(ALL_GENRES)), dtype=np.float32)
    movie_lookup: Dict[str, Dict[str, object]] = {}

    for row in movies_sub.itertuples(index=False):
        i_idx = int(row.i_idx)
        g_str = str(row.genres) if pd.notna(row.genres) else ""
        for g in g_str.split("|"):
            if g in genre_to_col:
                item_genres[i_idx, genre_to_col[g]] = 1.0
        if item_genres[i_idx].sum() == 0:
            # Default fallback if "(no genres listed)"
            item_genres[i_idx, genre_to_col["Drama"]] = 1.0
        movie_lookup[str(i_idx)] = {
            "movieId": int(row.movieId),
            "title": str(row.title),
            "genres": g_str,
        }

    return item_genres, movie_lookup


def train_funk_svd(
    u_train: np.ndarray,
    i_train: np.ndarray,
    r_train: np.ndarray,
    n_u: int,
    n_i: int,
    k: int = 20,
    lr: float = 0.01,
    reg: float = 0.02,
    epochs: int = 15,
    batch_size: int = 1024,
    seed: int = 42,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train FunkSVD with mini-batch vectorized NumPy SGD."""
    rng = np.random.default_rng(seed)
    global_mean = float(np.mean(r_train))
    user_biases = np.zeros(n_u, dtype=np.float32)
    item_biases = np.zeros(n_i, dtype=np.float32)
    user_factors = rng.normal(0.0, 0.08, size=(n_u, k)).astype(np.float32)
    item_factors = rng.normal(0.0, 0.08, size=(n_i, k)).astype(np.float32)

    n_obs = len(r_train)
    indices = np.arange(n_obs, dtype=np.int32)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        rng.shuffle(indices)
        for start in range(0, n_obs, batch_size):
            batch_idx = indices[start : start + batch_size]
            bu = u_train[batch_idx]
            bi = i_train[batch_idx]
            br = r_train[batch_idx]

            pu = user_factors[bu]
            qi = item_factors[bi]
            pred = global_mean + user_biases[bu] + item_biases[bi] + np.sum(pu * qi, axis=1)
            err = br - pred

            # Parameter decay + unbuffered scatter gradient updates
            user_biases[bu] *= (1.0 - lr * reg)
            item_biases[bi] *= (1.0 - lr * reg)
            np.add.at(user_biases, bu, lr * err)
            np.add.at(item_biases, bi, lr * err)

            grad_u = (lr * err[:, None]) * qi
            grad_i = (lr * err[:, None]) * pu
            user_factors[bu] *= (1.0 - lr * reg)
            item_factors[bi] *= (1.0 - lr * reg)
            np.add.at(user_factors, bu, grad_u)
            np.add.at(item_factors, bi, grad_i)

        if epoch in (1, 5, 10, epochs):
            preds_all = (
                global_mean
                + user_biases[u_train]
                + item_biases[i_train]
                + np.sum(user_factors[u_train] * item_factors[i_train], axis=1)
            )
            rmse = float(np.sqrt(np.mean((r_train - preds_all) ** 2)))
            print(f"[FunkSVD] Epoch {epoch:02d}/{epochs:02d} - Train RMSE: {rmse:.4f} ({time.time() - t0:.2f}s)")

    return global_mean, user_biases, item_biases, user_factors, item_factors


def evaluate_rmse(
    u_idx: np.ndarray,
    i_idx: np.ndarray,
    r_true: np.ndarray,
    global_mean: float,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
) -> float:
    """Compute clipped RMSE on [0.5, 5.0] as in Workshop 1."""
    pred = (
        global_mean
        + user_biases[u_idx]
        + item_biases[i_idx]
        + np.sum(user_factors[u_idx] * item_factors[i_idx], axis=1)
    )
    pred = np.clip(pred, 0.5, 5.0)
    return float(np.sqrt(np.mean((r_true - pred) ** 2)))


def _minmax(values: np.ndarray) -> np.ndarray:
    """Min-max normalize ``values`` into [0, 1] (constant arrays map to 0)."""
    span = float(values.max() - values.min())
    return ((values - values.min()) / (span + 1e-8)).astype(np.float32)


def _iso(ts: int) -> str:
    return pd.to_datetime(int(ts), unit="s").strftime("%Y-%m-%d %H:%M:%S")


def _period_sessions(
    clean_df: pd.DataFrame,
    max_candidates: int,
    min_session_ratings: int,
    like_threshold: float,
) -> Dict[int, pd.DataFrame]:
    """Chronological sessions of eligible warm users inside one period.

    A user is eligible when the period holds at least ``min_session_ratings`` clean ratings and
    the first ``max_candidates`` of them (the candidate slate) contain at least one like. Slate
    membership depends only on timestamps, never on the ratings.
    """
    ordered = clean_df.sort_values(["u_idx", "timestamp"], kind="mergesort")
    sessions: Dict[int, pd.DataFrame] = {}
    for u, grp in ordered.groupby("u_idx", sort=True):
        if len(grp) < min_session_ratings:
            continue
        slate = grp.iloc[:max_candidates]
        if bool((slate["rating"].to_numpy() >= like_threshold).any()):
            sessions[int(u)] = slate
    return sessions


def build_rl_candidate_pools(
    sub: pd.DataFrame,
    n_u: int,
    n_i: int,
    item_genres: np.ndarray,
    max_candidates: int = NUM_CANDIDATE_SLOTS,
    min_session_ratings: int = MIN_SESSION_RATINGS,
    val_user_fraction: float = RL_VAL_USER_FRACTION,
    like_threshold: float = LIKE_THRESHOLD,
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Build leak-free RL sessions and candidate slates from real post-cut feedback.

    1. Global chronological split shared with Workshops 1-2 (80% train / 10% val / 10% test by
       timestamp). FunkSVD (k=20) is trained on the train split only, so every latent factor,
       bias, popularity count, and novelty value predates every RL session.
    2. A session is the chronologically ordered list of a warm user's clean ratings (user and
       movie both seen in train) inside one period; see ``_period_sessions``.
    3. RL train/val sessions come from the validation period (eligible users split at random),
       RL test sessions from the test period. A user may contribute an earlier session to RL
       training and a later one to testing, mirroring the temporal protocol of Workshops 1-2.
    4. The candidate slate holds the session's first ``max_candidates`` real ratings; shorter
       sessions are padded with invalid slots (``cand_valid == 0``) that the environment masks.
       Valid slots are ordered by the static FunkSVD score, which never looks at the ratings.
    """
    rng = np.random.default_rng(seed)

    # 1. Exact Workshop 1 global temporal split (80% train, 10% val, 10% test)
    n_total = len(sub)
    n_train_cut = int(0.8 * n_total)
    n_val_cut = int(0.9 * n_total)

    global_train_df = sub.iloc[:n_train_cut]
    global_val_df = sub.iloc[n_train_cut:n_val_cut]
    global_test_df = sub.iloc[n_val_cut:]

    train_u_set = set(global_train_df["u_idx"].unique())
    train_i_set = set(global_train_df["i_idx"].unique())

    clean_val_df = global_val_df[
        global_val_df["u_idx"].isin(train_u_set) & global_val_df["i_idx"].isin(train_i_set)
    ]
    clean_test_df = global_test_df[
        global_test_df["u_idx"].isin(train_u_set) & global_test_df["i_idx"].isin(train_i_set)
    ]

    # 2. FunkSVD on exactly the 80% global train split (the Workshop 1/2 training data)
    print(f"[data_prep] Training FunkSVD on {len(global_train_df):,} train interactions ...")
    global_mean, user_biases_all, item_biases_all, user_factors_all, item_factors_all = train_funk_svd(
        u_train=global_train_df["u_idx"].to_numpy(),
        i_train=global_train_df["i_idx"].to_numpy(),
        r_train=global_train_df["rating"].to_numpy(),
        n_u=n_u,
        n_i=n_i,
        k=20,
        lr=0.01,
        reg=0.02,
        epochs=15,
        batch_size=1024,
        seed=seed,
    )
    factor_args = (global_mean, user_biases_all, item_biases_all, user_factors_all, item_factors_all)
    train_rmse = evaluate_rmse(
        global_train_df["u_idx"].to_numpy(),
        global_train_df["i_idx"].to_numpy(),
        global_train_df["rating"].to_numpy(),
        *factor_args,
    )
    val_rmse = evaluate_rmse(
        clean_val_df["u_idx"].to_numpy(),
        clean_val_df["i_idx"].to_numpy(),
        clean_val_df["rating"].to_numpy(),
        *factor_args,
    )
    test_rmse = evaluate_rmse(
        clean_test_df["u_idx"].to_numpy(),
        clean_test_df["i_idx"].to_numpy(),
        clean_test_df["rating"].to_numpy(),
        *factor_args,
    )
    print(
        f"[data_prep] FunkSVD RMSE -> Train: {train_rmse:.4f}, "
        f"Clean Val: {val_rmse:.4f}, Clean Test: {test_rmse:.4f}"
    )

    # 3. Item popularity and self-information novelty (bits), from the train split only
    item_pop_counts = np.bincount(global_train_df["i_idx"].to_numpy(), minlength=n_i).astype(np.float32)
    item_pop_probs = np.clip(item_pop_counts / float(len(global_train_df)), 1e-7, 1.0)
    item_novelty_bits = (-np.log2(item_pop_probs)).astype(np.float32)
    item_novelty_norm = _minmax(item_novelty_bits)
    item_pop_norm = _minmax(np.log1p(item_pop_counts))

    item_norms = np.linalg.norm(item_factors_all, axis=1, keepdims=True) + 1e-8
    item_factors_unit = (item_factors_all / item_norms).astype(np.float32)

    # 4. Real sessions: RL train/val from the validation period, RL test from the test period
    val_sessions = _period_sessions(clean_val_df, max_candidates, min_session_ratings, like_threshold)
    test_sessions = _period_sessions(clean_test_df, max_candidates, min_session_ratings, like_threshold)

    val_period_users = np.array(sorted(val_sessions), dtype=np.int32)
    shuffled = val_period_users[rng.permutation(len(val_period_users))]
    n_rl_val = int(round(val_user_fraction * len(shuffled)))
    rl_val_users = np.sort(shuffled[:n_rl_val])
    rl_train_users = np.sort(shuffled[n_rl_val:])
    rl_test_users = np.array(sorted(test_sessions), dtype=np.int32)

    pool_rows = (
        [(int(u), val_sessions[int(u)], 0) for u in rl_train_users]
        + [(int(u), val_sessions[int(u)], 0) for u in rl_val_users]
        + [(int(u), test_sessions[int(u)], 1) for u in rl_test_users]
    )
    n_pool = len(pool_rows)
    n_train_users, n_val_users, n_test_users = len(rl_train_users), len(rl_val_users), len(rl_test_users)
    print(
        f"[data_prep] Eligible sessions: {len(val_period_users)} validation-period users "
        f"({n_train_users} RL train / {n_val_users} RL val), {n_test_users} test-period users"
    )

    user_ids_of = sub.groupby("u_idx")["userId"].first().reindex(range(n_u)).to_numpy(np.int64)
    movie_ids_of = sub.groupby("i_idx")["movieId"].first().reindex(range(n_i)).to_numpy(np.int64)

    k = int(user_factors_all.shape[1])
    m = int(max_candidates)
    n_genres = int(item_genres.shape[1])
    pool_users = np.zeros(n_pool, dtype=np.int32)
    pool_period = np.zeros(n_pool, dtype=np.int8)
    user_factors_pool = np.zeros((n_pool, k), dtype=np.float32)
    user_biases_pool = np.zeros(n_pool, dtype=np.float32)
    cand_items = np.full((n_pool, m), -1, dtype=np.int32)
    cand_movie_ids = np.full((n_pool, m), -1, dtype=np.int64)
    cand_valid = np.zeros((n_pool, m), dtype=np.int8)
    cand_timestamps = np.zeros((n_pool, m), dtype=np.int64)
    cand_true_ratings = np.zeros((n_pool, m), dtype=np.float32)
    cand_is_liked = np.zeros((n_pool, m), dtype=np.int8)
    cand_item_factors = np.zeros((n_pool, m, k), dtype=np.float32)
    cand_svd_pred = np.zeros((n_pool, m), dtype=np.float32)
    cand_svd_scores = np.zeros((n_pool, m), dtype=np.float32)
    cand_bias_scores = np.zeros((n_pool, m), dtype=np.float32)
    cand_pop_scores = np.zeros((n_pool, m), dtype=np.float32)
    cand_novelty = np.zeros((n_pool, m), dtype=np.float32)
    cand_novelty_bits = np.zeros((n_pool, m), dtype=np.float32)
    cand_genres = np.zeros((n_pool, m, n_genres), dtype=np.float32)

    for row, (u, session, period) in enumerate(pool_rows):
        items = session["i_idx"].to_numpy(np.int32)
        ratings = session["rating"].to_numpy(np.float32)
        stamps = session["timestamp"].to_numpy(np.int64)
        pu, bu = user_factors_all[u], float(user_biases_all[u])
        svd_raw = (global_mean + bu + item_biases_all[items] + item_factors_all[items] @ pu).astype(np.float32)

        # Slot order: static FunkSVD score (descending); ties keep chronological order
        order = np.argsort(-svd_raw, kind="mergesort")
        items, ratings, stamps, svd_raw = items[order], ratings[order], stamps[order], svd_raw[order]
        n = len(items)

        pool_users[row] = u
        pool_period[row] = period
        user_factors_pool[row] = pu
        user_biases_pool[row] = bu
        cand_items[row, :n] = items
        cand_movie_ids[row, :n] = movie_ids_of[items]
        cand_valid[row, :n] = 1
        cand_timestamps[row, :n] = stamps
        cand_true_ratings[row, :n] = ratings
        cand_is_liked[row, :n] = np.where(ratings >= like_threshold, 1, -1).astype(np.int8)
        cand_item_factors[row, :n] = item_factors_unit[items]
        cand_svd_pred[row, :n] = svd_raw
        cand_svd_scores[row, :n] = _minmax(svd_raw)
        cand_bias_scores[row, :n] = _minmax((global_mean + bu + item_biases_all[items]).astype(np.float32))
        cand_pop_scores[row, :n] = _minmax(item_pop_norm[items])
        cand_novelty[row, :n] = item_novelty_norm[items]
        cand_novelty_bits[row, :n] = item_novelty_bits[items]
        cand_genres[row, :n] = item_genres[items]

    train_user_indices = np.arange(0, n_train_users, dtype=np.int32)
    val_user_indices = np.arange(n_train_users, n_train_users + n_val_users, dtype=np.int32)
    test_user_indices = np.arange(n_train_users + n_val_users, n_pool, dtype=np.int32)

    arrays = {
        "pool_users": pool_users,
        "pool_user_ids": user_ids_of[pool_users],
        "pool_period": pool_period,
        "user_factors": user_factors_pool,
        "user_biases": user_biases_pool,
        "cand_items": cand_items,
        "cand_movie_ids": cand_movie_ids,
        "cand_valid": cand_valid,
        "cand_timestamps": cand_timestamps,
        "cand_item_factors": cand_item_factors,
        "cand_true_ratings": cand_true_ratings,
        "cand_is_liked": cand_is_liked,
        "cand_svd_pred": cand_svd_pred,
        "cand_svd_scores": cand_svd_scores,
        "cand_bias_scores": cand_bias_scores,
        "cand_pop_scores": cand_pop_scores,
        "cand_novelty": cand_novelty,
        "cand_novelty_bits": cand_novelty_bits,
        "cand_genres": cand_genres,
        "train_user_indices": train_user_indices,
        "val_user_indices": val_user_indices,
        "test_user_indices": test_user_indices,
        # Train-split FunkSVD parameters, stored so tests can recompute every observation feature
        "item_factors": item_factors_all.astype(np.float32),
        "item_biases": item_biases_all.astype(np.float32),
        "global_mean": np.array(global_mean, dtype=np.float64),
        "train_end_timestamp": np.array(int(global_train_df["timestamp"].iloc[-1]), dtype=np.int64),
        "val_end_timestamp": np.array(int(global_val_df["timestamp"].iloc[-1]), dtype=np.int64),
    }

    valid = cand_valid.astype(bool)
    test_rows = test_user_indices
    valid_counts = valid.sum(axis=1)
    liked_counts = (cand_is_liked == 1).sum(axis=1)
    top10_hits = [
        float(np.mean(cand_is_liked[r, : min(10, int(valid_counts[r]))] == 1)) for r in test_rows
    ]

    def _like_ratio(rows: np.ndarray) -> float:
        mask = valid[rows]
        return round(float(np.mean(cand_true_ratings[rows][mask] >= like_threshold)), 4)

    extra_stats = {
        "train_ratings_global": int(len(global_train_df)),
        "val_ratings_global": int(len(global_val_df)),
        "test_ratings_global": int(len(global_test_df)),
        "clean_val_ratings": int(len(clean_val_df)),
        "clean_test_ratings": int(len(clean_test_df)),
        "train_end_utc": _iso(int(global_train_df["timestamp"].iloc[-1])),
        "val_end_utc": _iso(int(global_val_df["timestamp"].iloc[-1])),
        "test_end_utc": _iso(int(global_test_df["timestamp"].iloc[-1])),
        "funksvd_k": k,
        "funksvd_lr": 0.01,
        "funksvd_reg": 0.02,
        "funksvd_epochs": 15,
        "global_mean_rating": round(global_mean, 4),
        "funksvd_train_rmse": round(train_rmse, 4),
        "funksvd_val_rmse": round(val_rmse, 4),
        "funksvd_test_rmse": round(test_rmse, 4),
        "session_rule": (
            "first 50 clean ratings (warm user and movie) of a user inside one period, in "
            f"chronological order; eligible if the period holds >= {min_session_ratings} clean "
            "ratings and the slate contains >= 1 like (rating >= 4.0)"
        ),
        "min_session_ratings": int(min_session_ratings),
        "num_candidate_slots": int(m),
        "eligible_validation_period_users": int(len(val_period_users)),
        "eligible_test_period_users": int(n_test_users),
        "rl_pool_users": int(n_pool),
        "rl_train_users": int(n_train_users),
        "rl_val_users": int(n_val_users),
        "rl_test_users": int(n_test_users),
        "rl_train_test_user_overlap": int(len(set(rl_train_users.tolist()) & set(rl_test_users.tolist()))),
        "mean_valid_slots_train": round(float(valid_counts[train_user_indices].mean()), 2),
        "mean_valid_slots_test": round(float(valid_counts[test_rows].mean()), 2),
        "mean_liked_slots_train": round(float(liked_counts[train_user_indices].mean()), 2),
        "mean_liked_slots_test": round(float(liked_counts[test_rows].mean()), 2),
        "like_ratio_train_sessions": _like_ratio(train_user_indices),
        "like_ratio_val_sessions": _like_ratio(val_user_indices),
        "like_ratio_test_sessions": _like_ratio(test_rows),
        "static_top10_svd_precision_test": round(float(np.mean(top10_hits)), 4),
        "mean_candidate_novelty_bits": round(float(np.mean(cand_novelty_bits[valid])), 3),
        "unique_movies_in_candidate_pools": int(len(np.unique(cand_items[valid]))),
    }
    return arrays, extra_stats


def main() -> None:
    t_start = time.time()
    sub, movies_sub, base_stats = load_and_filter_movielens(
        ratings_path=resolve_raw_csv("ratings.csv"),
        movies_path=resolve_raw_csv("movies.csv"),
        n_sample_users=15000,
        min_interactions=10,
        seed=42,
    )

    n_u = base_stats["filtered_users"]
    n_i = base_stats["filtered_movies"]

    item_genres, movie_lookup = parse_genres_and_lookup(movies_sub, n_i=n_i)
    arrays, extra_stats = build_rl_candidate_pools(
        sub=sub,
        n_u=n_u,
        n_i=n_i,
        item_genres=item_genres,
        seed=42,
    )

    out_dir = W3_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = out_dir / "processed_cache.npz"
    np.savez_compressed(cache_path, **arrays)
    print(f"[data_prep] Saved compressed cache to {cache_path}")

    # Movie metadata for every candidate movie (titles and genres for trajectory rendering)
    used_movies = np.unique(arrays["cand_items"][arrays["cand_valid"].astype(bool)])
    compact_lookup = {str(int(i)): movie_lookup[str(int(i))] for i in used_movies if str(int(i)) in movie_lookup}
    lookup_path = out_dir / "movie_lookup.json"
    with open(lookup_path, "w", encoding="utf-8") as f:
        json.dump(compact_lookup, f, indent=2, ensure_ascii=False)
    print(f"[data_prep] Saved {len(compact_lookup):,} movie entries to {lookup_path}")

    full_stats = {
        **base_stats,
        **extra_stats,
        "prep_wall_time_sec": round(time.time() - t_start, 2),
    }
    stats_path = out_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(full_stats, f, indent=2)
    print(f"[data_prep] Saved dataset statistics to {stats_path}")
    print(json.dumps(full_stats, indent=2))


if __name__ == "__main__":
    main()
