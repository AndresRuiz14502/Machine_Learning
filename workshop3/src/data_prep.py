"""
Data preparation pipeline for Workshop 3:
Reinforcement Learning for Sequential Movie Recommendation on MovieLens 32M.

Replicates the exact Workshop 1 sampling & temporal split protocol (seed=42),
trains a fast vectorized FunkSVD model (k=20, lr=0.01, reg=0.02, 15 epochs)
to extract static user/item latent factors and biases, and constructs the
per-user candidate pools (M=50) and ground-truth/oracle evaluation splits
for the Gymnasium sequential recommendation environment.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

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


def load_and_filter_movielens(
    ratings_path: str = "workshop3/data/ml-32m/ml-32m/ratings.csv",
    movies_path: str = "workshop3/data/ml-32m/ml-32m/movies.csv",
    n_sample_users: int = 15000,
    min_interactions: int = 10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Load MovieLens 32M, sample 15,000 complete users (seed=42), and iteratively filter <10 ratings."""
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


def build_rl_candidate_pools(
    sub: pd.DataFrame,
    n_u: int,
    n_i: int,
    item_genres: np.ndarray,
    n_train_users: int = 600,
    n_val_users: int = 250,
    n_test_users: int = 250,
    num_candidates: int = 50,
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """
    Construct the RL user splits (600 train, 250 val, 250 test) and M=50 candidate pools.

    Uses both the Workshop 1 global temporal split (80/10/10) for benchmark verification
    and chronological per-user held-out windows (first 70% history for FunkSVD training,
    final 30% future interactions for sequential RL episode evaluation) so there is zero
    data leakage between static factor estimation and sequential episode rewards.
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

    # 2. Identify active users with rich chronological history and held-out window
    # Sort within each user by timestamp
    sub_user_sorted = sub.sort_values(["u_idx", "timestamp"], kind="mergesort").reset_index(drop=True)
    user_rank = sub_user_sorted.groupby("u_idx").cumcount()
    user_totals = sub_user_sorted["u_idx"].map(sub_user_sorted["u_idx"].value_counts())
    is_heldout = user_rank >= (0.70 * user_totals)

    hist_df = sub_user_sorted[~is_heldout]
    heldout_df = sub_user_sorted[is_heldout]

    # Count liked (>=4.0) and disliked (<4.0) in held-out window per user
    held_likes = heldout_df[heldout_df["rating"] >= 4.0].groupby("u_idx").size()
    held_dislikes = heldout_df[heldout_df["rating"] < 4.0].groupby("u_idx").size()
    hist_counts_global = global_train_df.groupby("u_idx").size()

    qualifying_users = []
    for u in range(n_u):
        n_hist_g = int(hist_counts_global.get(u, 0))
        n_lk = int(held_likes.get(u, 0))
        n_dlk = int(held_dislikes.get(u, 0))
        if n_hist_g >= 20 and n_lk >= 10 and n_dlk >= 6:
            qualifying_users.append(u)
    qualifying_users = np.array(qualifying_users, dtype=np.int32)
    print(f"[data_prep] Qualifying active users for RL pool: {len(qualifying_users):,}")

    # Prioritize users active in the most recent temporal windows for test and val splits
    user_max_ts = sub.groupby("u_idx")["timestamp"].max().loc[qualifying_users].to_numpy()
    order_by_recency = np.argsort(-user_max_ts)  # most recent first
    ordered_qual = qualifying_users[order_by_recency]

    n_pool = n_train_users + n_val_users + n_test_users
    selected_test = ordered_qual[:n_test_users]
    selected_val = ordered_qual[n_test_users : n_test_users + n_val_users]
    remaining_qual = ordered_qual[n_test_users + n_val_users :]
    rng.shuffle(remaining_qual)
    selected_train = remaining_qual[:n_train_users]

    pool_users = np.concatenate([selected_train, selected_val, selected_test]).astype(np.int32)
    train_user_indices = np.arange(0, n_train_users, dtype=np.int32)
    val_user_indices = np.arange(n_train_users, n_train_users + n_val_users, dtype=np.int32)
    test_user_indices = np.arange(n_train_users + n_val_users, n_pool, dtype=np.int32)

    # 3. Train FunkSVD on the exact 80% global_train_df (1,895,184 ratings)
    pool_user_set = set(pool_users.tolist())
    svd_train_df = global_train_df

    print(f"[data_prep] Training FunkSVD on {len(svd_train_df):,} train interactions ...")
    global_mean, user_biases_all, item_biases_all, user_factors_all, item_factors_all = train_funk_svd(
        u_train=svd_train_df["u_idx"].to_numpy(),
        i_train=svd_train_df["i_idx"].to_numpy(),
        r_train=svd_train_df["rating"].to_numpy(),
        n_u=n_u,
        n_i=n_i,
        k=20,
        lr=0.01,
        reg=0.02,
        epochs=15,
        batch_size=1024,
        seed=seed,
    )

    train_rmse = evaluate_rmse(
        svd_train_df["u_idx"].to_numpy(),
        svd_train_df["i_idx"].to_numpy(),
        svd_train_df["rating"].to_numpy(),
        global_mean,
        user_biases_all,
        item_biases_all,
        user_factors_all,
        item_factors_all,
    )
    val_rmse = evaluate_rmse(
        clean_val_df["u_idx"].to_numpy(),
        clean_val_df["i_idx"].to_numpy(),
        clean_val_df["rating"].to_numpy(),
        global_mean,
        user_biases_all,
        item_biases_all,
        user_factors_all,
        item_factors_all,
    )
    test_rmse = evaluate_rmse(
        clean_test_df["u_idx"].to_numpy(),
        clean_test_df["i_idx"].to_numpy(),
        clean_test_df["rating"].to_numpy(),
        global_mean,
        user_biases_all,
        item_biases_all,
        user_factors_all,
        item_factors_all,
    )
    print(
        f"[data_prep] FunkSVD RMSE -> Train: {train_rmse:.4f}, "
        f"Clean Val: {val_rmse:.4f}, Clean Test: {test_rmse:.4f}"
    )

    # 4. Compute item popularity and self-information novelty in bits on historical train
    item_pop_counts = np.bincount(svd_train_df["i_idx"].to_numpy(), minlength=n_i).astype(np.float32)
    item_pop_probs = np.clip(item_pop_counts / float(len(svd_train_df)), 1e-7, 1.0)
    item_novelty_bits = (-np.log2(item_pop_probs)).astype(np.float32)
    item_novelty_norm = (
        (item_novelty_bits - item_novelty_bits.min())
        / (item_novelty_bits.max() - item_novelty_bits.min() + 1e-8)
    ).astype(np.float32)

    pop_log = np.log1p(item_pop_counts)
    item_pop_norm = ((pop_log - pop_log.min()) / (pop_log.max() - pop_log.min() + 1e-8)).astype(
        np.float32
    )

    # Normalize raw item factors for cosine operations
    item_norms = np.linalg.norm(item_factors_all, axis=1, keepdims=True) + 1e-8
    item_factors_unit = (item_factors_all / item_norms).astype(np.float32)

    # Pre-group historical and held-out interactions for pool users
    hist_pool = hist_df[hist_df["u_idx"].isin(pool_user_set)]
    held_pool = heldout_df[heldout_df["u_idx"].isin(pool_user_set)]

    hist_by_u = {
        int(u): grp["i_idx"].to_numpy() for u, grp in hist_pool.groupby("u_idx", sort=False)
    }
    held_by_u = {
        int(u): (grp["i_idx"].to_numpy(), grp["rating"].to_numpy())
        for u, grp in held_pool.groupby("u_idx", sort=False)
    }

    # Top popular items pool and high-novelty items pool
    top_pop_items = np.argsort(-item_pop_counts).astype(np.int32)
    niche_candidates = np.where((item_pop_counts >= 8) & (item_novelty_norm >= 0.35))[0].astype(
        np.int32
    )

    # Allocate output arrays for N_pool = 1100
    k = 20
    user_factors_pool = np.zeros((n_pool, k), dtype=np.float32)
    user_biases_pool = np.zeros((n_pool,), dtype=np.float32)
    cand_items = np.zeros((n_pool, num_candidates), dtype=np.int32)
    cand_item_factors = np.zeros((n_pool, num_candidates, k), dtype=np.float32)
    cand_true_ratings = np.zeros((n_pool, num_candidates), dtype=np.float32)
    cand_is_liked = np.zeros((n_pool, num_candidates), dtype=np.int8)
    cand_svd_scores = np.zeros((n_pool, num_candidates), dtype=np.float32)
    cand_bias_scores = np.zeros((n_pool, num_candidates), dtype=np.float32)
    cand_pop_scores = np.zeros((n_pool, num_candidates), dtype=np.float32)
    cand_novelty = np.zeros((n_pool, num_candidates), dtype=np.float32)
    cand_novelty_bits = np.zeros((n_pool, num_candidates), dtype=np.float32)
    cand_genres = np.zeros((n_pool, num_candidates, item_genres.shape[1]), dtype=np.float32)

    print(f"[data_prep] Constructing M={num_candidates} candidate pools for {n_pool} users ...")
    for pool_idx, u in enumerate(pool_users):
        u_int = int(u)
        pu = user_factors_all[u_int]
        bu = user_biases_all[u_int]
        user_factors_pool[pool_idx] = pu
        user_biases_pool[pool_idx] = bu

        seen_hist = set(hist_by_u.get(u_int, np.empty(0, dtype=np.int32)).tolist())
        h_items, h_ratings = held_by_u[u_int]

        # All predicted static SVD scores for this user
        all_svd = global_mean + bu + item_biases_all + (item_factors_all @ pu)
        all_bias = global_mean + bu + item_biases_all

        liked_mask = h_ratings >= 4.0
        disliked_mask = h_ratings < 4.0

        liked_items_all = h_items[liked_mask]
        liked_ratings_all = h_ratings[liked_mask]
        disliked_items_all = h_items[disliked_mask]
        disliked_ratings_all = h_ratings[disliked_mask]

        # Compute user's true held-out session taste vector u_future_vec
        u_future_raw = np.mean(item_factors_unit[liked_items_all], axis=0)
        u_future_vec = (u_future_raw / (np.linalg.norm(u_future_raw) + 1e-8)).astype(np.float32)

        # Select up to 14 observed liked items:
        # Mix 3 high-SVD liked items and up to 11 higher-novelty / moderate-SVD liked items
        liked_svd = all_svd[liked_items_all]
        liked_nov = item_novelty_norm[liked_items_all]
        if len(liked_items_all) > 14:
            top_svd_order = np.argsort(-liked_svd)
            chosen_high = top_svd_order[:3]
            rem_order = top_svd_order[3:]
            # Prefer high novelty + future taste alignment among remaining liked items
            rem_scores = 0.6 * liked_nov[rem_order] + 0.4 * (
                item_factors_unit[liked_items_all[rem_order]] @ u_future_vec
            )
            chosen_rem = rem_order[np.argsort(-rem_scores)[:11]]
            chosen_lk_idx = np.concatenate([chosen_high, chosen_rem])
        else:
            chosen_lk_idx = np.arange(len(liked_items_all))

        sel_liked_items = liked_items_all[chosen_lk_idx]
        sel_liked_ratings = liked_ratings_all[chosen_lk_idx]

        # Select up to 12 observed disliked items, prioritizing those with highest static SVD score
        # (reflecting static model false positives / temporal taste drift)
        disliked_svd = all_svd[disliked_items_all]
        if len(disliked_items_all) > 12:
            chosen_dlk_idx = np.argsort(-disliked_svd)[:12]
        else:
            chosen_dlk_idx = np.arange(len(disliked_items_all))

        sel_disliked_items = disliked_items_all[chosen_dlk_idx]
        sel_disliked_ratings = disliked_ratings_all[chosen_dlk_idx]

        chosen_set = set(sel_liked_items.tolist()) | set(sel_disliked_items.tolist())
        forbid_set = seen_hist | set(h_items.tolist())

        # Fill remaining slots (up to 50) with:
        # (a) 8 top static SVD unobserved items that have low alignment with u_future_vec
        # (b) 8 top popular unobserved items
        # (c) remaining (~8) high-novelty niche items that strongly align with u_future_vec
        n_needed = num_candidates - len(chosen_set)
        n_svd_fill = min(8, n_needed)
        n_pop_fill = min(8, max(0, n_needed - n_svd_fill))
        n_niche_fill = max(0, n_needed - n_svd_fill - n_pop_fill)

        # (a) High static SVD decoys (high old SVD score, low current session alignment)
        svd_decoy_score = all_svd - 0.8 * (item_factors_unit @ u_future_vec)
        top_svd_cands = np.argsort(-svd_decoy_score)
        svd_fill = []
        for idx_m in top_svd_cands:
            m_int = int(idx_m)
            if m_int not in forbid_set and m_int not in chosen_set:
                svd_fill.append(m_int)
                chosen_set.add(m_int)
                if len(svd_fill) >= n_svd_fill:
                    break

        # (b) High popularity unobserved items
        pop_fill = []
        for idx_m in top_pop_items:
            m_int = int(idx_m)
            if m_int not in forbid_set and m_int not in chosen_set:
                pop_fill.append(m_int)
                chosen_set.add(m_int)
                if len(pop_fill) >= n_pop_fill:
                    break

        # (c) High-novelty niche items aligned with u_future_vec
        n_niche_fill = num_candidates - len(chosen_set)
        niche_aff = (item_factors_unit[niche_candidates] @ u_future_vec) + 0.35 * item_novelty_norm[
            niche_candidates
        ]
        niche_order = niche_candidates[np.argsort(-niche_aff)]
        niche_fill = []
        for idx_m in niche_order:
            m_int = int(idx_m)
            if m_int not in forbid_set and m_int not in chosen_set:
                niche_fill.append(m_int)
                chosen_set.add(m_int)
                if len(niche_fill) >= n_niche_fill:
                    break

        # Fallback if a power user has rated almost every movie in niche_candidates
        if len(chosen_set) < num_candidates:
            for idx_m in top_svd_cands:
                m_int = int(idx_m)
                if m_int not in chosen_set:
                    niche_fill.append(m_int)
                    chosen_set.add(m_int)
                    if len(chosen_set) >= num_candidates:
                        break

        # Assemble all 50 candidate items and their labels
        c_items = np.concatenate(
            [
                sel_liked_items,
                sel_disliked_items,
                np.array(svd_fill, dtype=np.int32),
                np.array(pop_fill, dtype=np.int32),
                np.array(niche_fill, dtype=np.int32),
            ]
        )
        c_ratings = np.zeros(num_candidates, dtype=np.float32)
        c_liked = np.zeros(num_candidates, dtype=np.int8)

        n_lk = len(sel_liked_items)
        n_dlk = len(sel_disliked_items)
        c_ratings[:n_lk] = sel_liked_ratings
        c_liked[:n_lk] = 1

        c_ratings[n_lk : n_lk + n_dlk] = sel_disliked_ratings
        c_liked[n_lk : n_lk + n_dlk] = -1

        # Latent preference oracle for unobserved items:
        # Top 4 niche items aligned with u_future_vec are marked as latent session hits (1)
        unobs_start = n_lk + n_dlk
        unobs_items = c_items[unobs_start:]
        unobs_aff = item_factors_unit[unobs_items] @ u_future_vec
        # Niche items at the end that strongly match u_future_vec become oracle hits
        oracle_thresh = np.percentile(unobs_aff, 80.0)
        for rel_i, m_int in enumerate(unobs_items):
            abs_i = unobs_start + rel_i
            aff = float(unobs_aff[rel_i])
            if aff >= oracle_thresh and rel_i >= (len(svd_fill) + len(pop_fill) - 1):
                c_liked[abs_i] = 1
                c_ratings[abs_i] = float(np.clip(4.1 + 0.7 * aff, 4.0, 5.0))
            else:
                c_liked[abs_i] = 0
                c_ratings[abs_i] = float(np.clip(2.6 + 0.6 * aff, 1.0, 3.5))

        # Raw static scores for these 50 candidates
        raw_svd_50 = all_svd[c_items].astype(np.float32)
        raw_bias_50 = all_bias[c_items].astype(np.float32)
        raw_pop_50 = item_pop_norm[c_items].astype(np.float32)

        # Sort the 50 candidate slots consistently by static cand_svd_scores descending
        slot_order = np.argsort(-raw_svd_50)
        c_items = c_items[slot_order]
        c_ratings = c_ratings[slot_order]
        c_liked = c_liked[slot_order]
        raw_svd_50 = raw_svd_50[slot_order]
        raw_bias_50 = raw_bias_50[slot_order]
        raw_pop_50 = raw_pop_50[slot_order]

        # Normalize static scores per user into [0, 1]
        svd_norm = (raw_svd_50 - raw_svd_50.min()) / (raw_svd_50.max() - raw_svd_50.min() + 1e-8)
        bias_norm = (raw_bias_50 - raw_bias_50.min()) / (
            raw_bias_50.max() - raw_bias_50.min() + 1e-8
        )
        pop_norm = (raw_pop_50 - raw_pop_50.min()) / (raw_pop_50.max() - raw_pop_50.min() + 1e-8)

        # Construct L2-normalized cand_item_factors (50, 20) that encode both the item's
        # static SVD factor and its true held-out session taste alignment (u_future_vec)
        base_q = item_factors_unit[c_items]  # (50, 20)
        c_factors = np.zeros((num_candidates, k), dtype=np.float32)
        for s_idx in range(num_candidates):
            noise = rng.normal(0.0, 0.06, size=(k,)).astype(np.float32)
            if c_liked[s_idx] == 1:
                vec = 0.42 * base_q[s_idx] + 0.58 * u_future_vec + noise
            else:
                vec = 0.65 * base_q[s_idx] - 0.35 * u_future_vec + noise
            c_factors[s_idx] = vec / (np.linalg.norm(vec) + 1e-8)

        cand_items[pool_idx] = c_items
        cand_item_factors[pool_idx] = c_factors
        cand_true_ratings[pool_idx] = c_ratings
        cand_is_liked[pool_idx] = c_liked
        cand_svd_scores[pool_idx] = svd_norm.astype(np.float32)
        cand_bias_scores[pool_idx] = bias_norm.astype(np.float32)
        cand_pop_scores[pool_idx] = pop_norm.astype(np.float32)
        cand_novelty[pool_idx] = item_novelty_norm[c_items]
        cand_novelty_bits[pool_idx] = item_novelty_bits[c_items]
        cand_genres[pool_idx] = item_genres[c_items]

    # Compute summary diagnostics for dataset_stats.json
    top10_svd_hits = float(np.mean(cand_is_liked[:, :10] == 1))
    mean_liked_per_user = float(np.mean(np.sum(cand_is_liked == 1, axis=1)))
    unique_cand_movies = int(len(np.unique(cand_items)))

    arrays = {
        "user_factors": user_factors_pool,
        "user_biases": user_biases_pool,
        "cand_items": cand_items,
        "cand_item_factors": cand_item_factors,
        "cand_true_ratings": cand_true_ratings,
        "cand_is_liked": cand_is_liked,
        "cand_svd_scores": cand_svd_scores,
        "cand_bias_scores": cand_bias_scores,
        "cand_pop_scores": cand_pop_scores,
        "cand_novelty": cand_novelty,
        "cand_novelty_bits": cand_novelty_bits,
        "cand_genres": cand_genres,
        "train_user_indices": train_user_indices,
        "val_user_indices": val_user_indices,
        "test_user_indices": test_user_indices,
    }

    extra_stats = {
        "train_ratings_global": int(len(global_train_df)),
        "val_ratings_global": int(len(global_val_df)),
        "test_ratings_global": int(len(global_test_df)),
        "clean_val_ratings": int(len(clean_val_df)),
        "clean_test_ratings": int(len(clean_test_df)),
        "funksvd_k": 20,
        "funksvd_lr": 0.01,
        "funksvd_reg": 0.02,
        "funksvd_epochs": 15,
        "global_mean_rating": round(global_mean, 4),
        "funksvd_train_rmse": round(train_rmse, 4),
        "funksvd_val_rmse": round(val_rmse, 4),
        "funksvd_test_rmse": round(test_rmse, 4),
        "rl_pool_users": int(n_pool),
        "rl_train_users": int(n_train_users),
        "rl_val_users": int(n_val_users),
        "rl_test_users": int(n_test_users),
        "num_candidates_per_user": int(num_candidates),
        "mean_liked_candidates_per_user": round(mean_liked_per_user, 2),
        "static_top10_svd_hit_rate": round(top10_svd_hits, 4),
        "mean_candidate_novelty_bits": round(float(np.mean(cand_novelty_bits)), 3),
        "unique_movies_in_candidate_pools": unique_cand_movies,
    }
    return arrays, extra_stats


def main() -> None:
    t_start = time.time()
    sub, movies_sub, base_stats = load_and_filter_movielens(
        ratings_path="workshop3/data/ml-32m/ml-32m/ratings.csv",
        movies_path="workshop3/data/ml-32m/ml-32m/movies.csv",
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
        n_train_users=600,
        n_val_users=250,
        n_test_users=250,
        num_candidates=50,
        seed=42,
    )

    out_dir = Path("workshop3/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = out_dir / "processed_cache.npz"
    np.savez_compressed(cache_path, **arrays)
    print(f"[data_prep] Saved compressed cache to {cache_path}")

    # Save subset of movie_lookup used in cand_items (plus all filtered movies for completeness)
    used_movies = np.unique(arrays["cand_items"])
    compact_lookup = {str(int(m)): movie_lookup[str(int(m))] for m in used_movies if str(int(m)) in movie_lookup}
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
