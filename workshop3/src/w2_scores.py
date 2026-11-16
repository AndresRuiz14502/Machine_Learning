"""
Workshop 2 HybridNeuMF scores for the Workshop 3 candidate slots.

Precomputes the predicted rating of the official Workshop 2 model (``checkpoints/model_best.pt``)
for every (RL pool user, candidate slot) pair of ``workshop3/data/processed_cache.npz`` and saves
them in the small versioned artifact ``workshop3/data/w2_candidate_scores.npz``. The Greedy
HybridNeuMF policy and ``workshop3/src/evaluate.py`` read that artifact, so they need neither the
raw MovieLens CSVs nor the Workshop 2 checkpoint.

The artifact is written only if every check passes:

1. The Workshop 2 pipeline is rebuilt as in ``src/evaluate.py`` (``prepare_datasets`` with
   ``save_splits_file=False``, ``prepare_features``, ``load_best_checkpoint``). The rebuilt split
   must reproduce the Workshop 3 split statistics in ``workshop3/data/dataset_stats.json``, the
   checkpoint side-feature buffers must equal the ones rebuilt from the training split, and the
   test RMSE, MAE, ROC-AUC and top-10 ranking metrics must reproduce ``runs/evaluation_summary.json``.
2. Pool users are mapped to Workshop 2 user indices through their MovieLens userId
   (``pool_user_ids`` in the cache) and candidate slots to Workshop 2 item indices through their
   MovieLens movieId (``cand_movie_ids``). Every id must exist in the Workshop 2 mappings.
3. Every real slot is a held-out rating for Workshop 2: no slot is cold (the user and the movie
   both have Workshop 2 training ratings), no slot's (user, movie) rating is in the Workshop 2
   training split, and every slot matches the rating and timestamp of a Workshop 2 clean
   validation rating (RL train/val sessions) or clean test rating (RL test sessions).

Scores are ``HybridNeuMF.forward`` outputs (``model.eval()`` under ``torch.no_grad()``) without the
[0.5, 5.0] clamp that Workshop 2 applies before computing RMSE. The clamp is monotone, so dropping
it keeps the ranking and avoids ties at 5.0 that a greedy argmax would break by slot index, which
follows the FunkSVD order. Padding slots (``cand_valid == 0``) hold the finite sentinel
``PADDING_SCORE``; the artifact stores ``cand_valid`` so the policy can exclude them explicitly.

Artifact contents (rows follow the pool-user order of ``processed_cache.npz``):

- ``scores`` float32 [n_pool, M]: predicted rating of each real slot, ``PADDING_SCORE`` on padding.
- ``cand_valid`` int8 [n_pool, M], ``cand_items`` int32 [n_pool, M], ``cand_movie_ids`` int64
  [n_pool, M] and ``pool_user_ids`` int64 [n_pool]: copies of the cache arrays (alignment checks).
- ``w2_user_idx`` int32 [n_pool] and ``w2_item_idx`` int32 [n_pool, M] (-1 on padding): Workshop 2
  indices used for inference.
- ``w2_cold_item`` bool [n_pool, M]: the slot's movie has no rating in the Workshop 2 training split.
- ``rating_in_w2_train`` bool [n_pool, M]: the user's rating of that movie is in the Workshop 2
  training split (in-sample for the supervised model).
- ``checkpoint_sha256`` and ``meta_json``: checkpoint hash plus provenance and check results.

Usage, from the repository root (needs ``dataSet/ratings.csv`` and ``checkpoints/model_best.pt``):

    python workshop3/src/w2_scores.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CACHE_PATH = PROJECT_ROOT / "workshop3" / "data" / "processed_cache.npz"
DEFAULT_STATS_PATH = PROJECT_ROOT / "workshop3" / "data" / "dataset_stats.json"
DEFAULT_SCORES_PATH = PROJECT_ROOT / "workshop3" / "data" / "w2_candidate_scores.npz"
W2_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "model_best.pt"
W2_SUMMARY_PATH = PROJECT_ROOT / "runs" / "evaluation_summary.json"

FORMAT_VERSION = 2
PADDING_SCORE = np.float32(-1.0e9)
W2_RANKING_KEY = "HybridNeuMF (W2 Deep CF)"
SCORE_DEFINITION = (
    "HybridNeuMF.forward(u, i) of checkpoints/model_best.pt in eval mode under torch.no_grad(), "
    "without the [0.5, 5.0] clamp used for RMSE (monotone, so the ranking is unchanged); "
    f"padding slots (cand_valid == 0) hold {float(PADDING_SCORE):.0e}"
)
# Cache arrays the artifact must reproduce exactly, slot by slot or user by user.
ALIGNED_KEYS = ("cand_valid", "cand_items", "cand_movie_ids", "pool_user_ids")
REQUIRED_CACHE_KEYS = ALIGNED_KEYS + (
    "pool_users",
    "pool_period",
    "cand_true_ratings",
    "cand_timestamps",
    "test_user_indices",
    "train_end_timestamp",
)


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    """Return `path` relative to the repository root as a POSIX string."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name


def require_cache_keys(cache: Dict[str, Any], keys: Iterable[str], source: str = "processed_cache.npz") -> None:
    """Raise a clear error when the Workshop 3 cache predates the id columns this module needs."""
    missing = [key for key in keys if key not in cache]
    if missing:
        raise KeyError(
            f"{source} lacks {missing}. The Workshop 2 scores map users through `pool_user_ids` "
            "(MovieLens userId) and slots through `cand_movie_ids` (MovieLens movieId); rebuild the "
            "cache with `python workshop3/src/data_prep.py`."
        )


def load_w2_candidate_scores(
    path: Path | str = DEFAULT_SCORES_PATH,
    cache_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load `w2_candidate_scores.npz` and check that it is aligned with the Workshop 3 cache.

    Returns a dict with the stored arrays plus `meta` (parsed metadata). When `cache_data`
    (the arrays of `processed_cache.npz`) is given, `cand_valid`, `cand_items`, `cand_movie_ids`
    and `pool_user_ids` must be identical to the cache's, so that `scores[u, a]` refers to the
    same user and movie as the environment's candidate slot `a` of pool user `u`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Workshop 2 candidate scores not found at {path}. "
            "Run `python workshop3/src/w2_scores.py` from the repository root."
        )
    with np.load(path, allow_pickle=False) as npz:
        out: Dict[str, Any] = {key: npz[key] for key in npz.files}

    meta = json.loads(str(out.pop("meta_json")))
    if int(meta.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(
            f"{path.name}: format_version {meta.get('format_version')} != {FORMAT_VERSION}. "
            "Rebuild it with `python workshop3/src/w2_scores.py`."
        )
    out["meta"] = meta
    out["checkpoint_sha256"] = str(out["checkpoint_sha256"])

    scores = out["scores"]
    if scores.ndim != 2 or scores.dtype != np.float32 or not np.all(np.isfinite(scores)):
        raise ValueError(f"{path.name}: `scores` must be a finite float32 [n_pool_users, M] array")
    for key in ("cand_valid", "cand_items", "cand_movie_ids", "w2_item_idx", "w2_cold_item", "rating_in_w2_train"):
        if out[key].shape != scores.shape:
            raise ValueError(f"{path.name}: `{key}` shape {out[key].shape} != scores {scores.shape}")
    for key in ("pool_user_ids", "w2_user_idx"):
        if out[key].shape != (scores.shape[0],):
            raise ValueError(f"{path.name}: `{key}` must have shape ({scores.shape[0]},)")
    valid = out["cand_valid"] > 0
    if not np.all(scores[~valid] == PADDING_SCORE) or not np.all(scores[valid] > PADDING_SCORE):
        raise ValueError(f"{path.name}: padding slots must hold PADDING_SCORE and real slots a real score")

    if cache_data is not None:
        require_cache_keys(cache_data, ALIGNED_KEYS)
        for key in ALIGNED_KEYS:
            cached = np.asarray(cache_data[key])
            if cached.shape != out[key].shape or not np.array_equal(cached, out[key]):
                raise ValueError(
                    f"{path.name} is not aligned with processed_cache.npz (`{key}` differs). "
                    "Rebuild it with `python workshop3/src/w2_scores.py`."
                )
    return out


def build_w2_candidate_scores(
    cache_path: Path = DEFAULT_CACHE_PATH,
    stats_path: Path = DEFAULT_STATS_PATH,
    checkpoint_path: Path = W2_CHECKPOINT_PATH,
    summary_path: Path = W2_SUMMARY_PATH,
    device_name: str = "auto",
) -> tuple:
    """Rebuild the Workshop 2 pipeline, verify alignment, and score every real candidate slot.

    Returns `(arrays, meta)` ready for `np.savez_compressed`. Raises `RuntimeError` listing
    every failed check instead of returning partially aligned scores.
    """
    import pandas as pd
    import torch

    from src.baselines import evaluate_ranking
    from src.data_loader import (
        LIKE_THRESHOLD,
        MIN_RATINGS_PER_ITEM,
        MIN_RATINGS_PER_USER,
        N_USERS_SAMPLE,
        SEED,
        TOP_K,
        filter_min_support,
        load_raw_data,
        prepare_datasets,
        sample_users,
    )
    from src.features import prepare_features
    from src.test_model import load_best_checkpoint
    from src.train import build_training_side_features, compute_metrics, predict_batches

    t0 = time.time()
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    def fail_now() -> None:
        raise RuntimeError("Alignment checks failed:\n- " + "\n- ".join(failures))

    with np.load(cache_path) as npz:
        cache = {key: npz[key] for key in npz.files}
    require_cache_keys(cache, REQUIRED_CACHE_KEYS)
    w3_stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    official = json.loads(Path(summary_path).read_text(encoding="utf-8"))

    # 1. Workshop 2 pipeline, exactly as in src/evaluate.py (splits.json is not rewritten)
    data = prepare_datasets(save_splits_file=False, verbose=False)
    train_df, val_df, test_df = data["train_df"], data["val_df"], data["test_df"]
    user2idx, item2idx = data["user2idx"], data["item2idx"]
    n_users, n_items = int(data["n_users"]), int(data["n_items"])
    feats = prepare_features(train_df, val_df, test_df, data["movies"], item2idx)
    model, payload = load_best_checkpoint(checkpoint_path=Path(checkpoint_path), device=device)
    model.eval()
    print(f"[w2_scores] Workshop 2 pipeline rebuilt ({time.time() - t0:.1f}s, device={device})")

    # Full filtered sample (before the temporal split) with the same Workshop 2 primitives
    ratings_raw, _ = load_raw_data(load_links_tags=False)
    full_df, _ = sample_users(ratings_raw, N_USERS_SAMPLE, SEED)
    del ratings_raw
    full_df = filter_min_support(full_df, MIN_RATINGS_PER_USER, MIN_RATINGS_PER_ITEM)
    n_filtered = int(len(full_df))
    del full_df

    split_pairs = {
        "filtered_users": n_users,
        "filtered_movies": n_items,
        "filtered_ratings": n_filtered,
        "train_ratings_global": int(len(train_df)),
        "clean_val_ratings": int(len(val_df)),
        "clean_test_ratings": int(len(test_df)),
    }
    for key, w2_value in split_pairs.items():
        check(int(w3_stats[key]) == w2_value, f"split: W3 {key}={w3_stats[key]} != W2 {w2_value}")
    check(
        round(float(feats["mu"]), 4) == float(w3_stats["global_mean_rating"]),
        f"split: W2 train mean {feats['mu']:.4f} != W3 global_mean_rating {w3_stats['global_mean_rating']}",
    )
    w2_train_end = int(train_df["timestamp"].max())
    check(
        int(cache["train_end_timestamp"]) == w2_train_end,
        f"split: W3 train_end_timestamp {int(cache['train_end_timestamp'])} != W2 train max timestamp {w2_train_end}",
    )

    # 2. Checkpoint faithfulness: config, side-feature buffers, global mean, test metrics, ranking
    cfg = payload["config"]
    check(cfg["n_users"] == n_users and cfg["n_items"] == n_items, "checkpoint: n_users/n_items mismatch")
    side = build_training_side_features(train_df, data["movies"], item2idx, n_users, n_items)
    buffer_diff = max(
        float((getattr(model, name).detach().cpu() - side[name]).abs().max())
        for name in ("item_genres", "user_priors", "item_priors")
    )
    check(buffer_diff <= 1e-6, f"checkpoint: side-feature buffers differ from rebuilt ones ({buffer_diff:.2e})")
    global_mean = float(model.global_mean.detach().cpu())
    check(abs(global_mean - float(feats["mu"])) <= 1e-6, "checkpoint: global_mean != training mean")

    u_test = torch.tensor(test_df["u"].to_numpy(), dtype=torch.long)
    i_test = torch.tensor(test_df["i"].to_numpy(), dtype=torch.long)
    test_preds = predict_batches(model, u_test, i_test, device=device)
    test_metrics = compute_metrics(
        test_df["rating"].to_numpy(dtype=np.float32),
        test_preds,
        like_threshold=LIKE_THRESHOLD,
        decision_threshold=float(official["decision_threshold"]),
    )
    metric_check = {}
    for key in ("RMSE", "MAE", "roc_auc"):
        reproduced, reference = float(test_metrics[key]), float(official["hybrid_test_metrics"][key])
        metric_check[key] = {"reproduced": reproduced, "official": reference}
        check(abs(reproduced - reference) <= 1e-6, f"inference: test {key} {reproduced:.6f} != official {reference:.6f}")

    item_pop = train_df.groupby("i").size()
    pop_vec = np.zeros(n_items, dtype=np.float32)
    pop_vec[item_pop.index.to_numpy()] = item_pop.to_numpy(dtype=np.float32)
    pop_log_norm = np.log1p(pop_vec) / (np.log1p(pop_vec).max() + 1e-6)

    @torch.no_grad()
    def hybrid_rank_score(u: int, items: np.ndarray) -> np.ndarray:
        # Same scoring rule as `hybrid_rank_score` inside src/evaluate.py
        u_t = torch.full((len(items),), int(u), dtype=torch.long, device=device)
        i_t = torch.tensor(items, dtype=torch.long, device=device)
        return model(u_t, i_t).clamp(0.5, 5.0).cpu().numpy() + 1.35 * pop_log_norm[items]

    ranking = evaluate_ranking(hybrid_rank_score, train_df, test_df, n_items, k=TOP_K)
    official_ranking = official["ranking_comparison"][W2_RANKING_KEY]
    for key, reference in official_ranking.items():
        check(abs(float(ranking[key]) - float(reference)) <= 5.1e-5, f"inference: ranking {key} {ranking[key]:.4f} != {reference}")
    ranking_users = int(test_df.loc[test_df["like"] == 1, "u"].nunique())
    print(f"[w2_scores] Faithfulness checks done ({time.time() - t0:.1f}s)")

    # 3. Map pool users (userId) and real candidate slots (movieId) to Workshop 2 indices
    pool_user_ids = cache["pool_user_ids"].astype(np.int64)
    cand_movie_ids = cache["cand_movie_ids"].astype(np.int64)
    cand_items = cache["cand_items"].astype(np.int32)
    valid = cache["cand_valid"].astype(bool)
    check(
        bool(np.all(cand_movie_ids[~valid] == -1) and np.all(cand_items[~valid] == -1)),
        "slots: padding slots must hold cand_items == cand_movie_ids == -1",
    )
    check(bool(np.all(cand_movie_ids[valid] >= 0)), "slots: a real slot has no MovieLens movieId")

    w2_user_idx = np.array([user2idx.get(int(u), -1) for u in pool_user_ids], dtype=np.int64)
    missing_users = sorted({int(u) for u in pool_user_ids[w2_user_idx < 0]})
    check(not missing_users, f"users: {len(missing_users)} pool userIds missing from W2 user2idx: {missing_users[:10]}")

    w2_item_idx = np.full(cand_movie_ids.shape, -1, dtype=np.int64)
    w2_item_idx[valid] = [item2idx.get(int(m), -1) for m in cand_movie_ids[valid]]
    unmapped = valid & (w2_item_idx < 0)
    missing_movies = sorted({int(m) for m in cand_movie_ids[unmapped]})
    check(not missing_movies, f"items: {len(missing_movies)} slot movieIds missing from W2 item2idx: {missing_movies[:10]}")
    if failures:
        fail_now()

    # W3 item index = rank of the movieId in the sorted filtered catalog (same catalog as W2)
    catalog_sorted = np.sort(np.fromiter(item2idx.keys(), dtype=np.int64))
    check(
        bool(np.array_equal(catalog_sorted[cand_items[valid]], cand_movie_ids[valid])),
        "items: cand_items is not the rank of cand_movie_ids in the sorted W2 catalog",
    )
    users_sorted = np.sort(np.fromiter(user2idx.keys(), dtype=np.int64))
    check(
        bool(np.array_equal(users_sorted[cache["pool_users"]], pool_user_ids)),
        "users: pool_users is not the rank of pool_user_ids in the sorted W2 user list",
    )

    # 4. Every real slot must be a held-out rating for Workshop 2
    train_counts = train_df.groupby("u").size().reindex(w2_user_idx, fill_value=0).to_numpy()
    cold_users = int((train_counts == 0).sum())
    train_items = np.unique(train_df["i"].to_numpy())
    w2_cold_item = valid & ~np.isin(w2_item_idx, train_items)
    train_codes = train_df["u"].to_numpy(np.int64) * n_items + train_df["i"].to_numpy(np.int64)
    pair_codes = np.where(valid, w2_user_idx[:, None] * n_items + w2_item_idx, -1)
    rating_in_w2_train = valid & np.isin(pair_codes, train_codes)

    rows, slot_idx = np.nonzero(valid)
    slot_df = pd.DataFrame(
        {
            "code": pair_codes[rows, slot_idx],
            "rating": cache["cand_true_ratings"][rows, slot_idx].astype(np.float32),
            "timestamp": cache["cand_timestamps"][rows, slot_idx].astype(np.int64),
            "period": cache["pool_period"][rows].astype(np.int8),
        }
    )

    def heldout_match(split_df: Any, period: int) -> Dict[str, int]:
        ref = pd.DataFrame(
            {
                "code": split_df["u"].to_numpy(np.int64) * n_items + split_df["i"].to_numpy(np.int64),
                "ref_rating": split_df["rating"].to_numpy(np.float32),
                "ref_timestamp": split_df["timestamp"].to_numpy(np.int64),
            }
        )
        part = slot_df[slot_df["period"] == period].merge(ref, on="code", how="left", validate="many_to_one")
        found = part["ref_rating"].notna()
        exact = found & (part["ref_rating"] == part["rating"]) & (part["ref_timestamp"] == part["timestamp"])
        return {"slots": int(len(part)), "found": int(found.sum()), "rating_and_timestamp_match": int(exact.sum())}

    val_match = heldout_match(val_df, 0)
    test_match = heldout_match(test_df, 1)
    test_rows = cache["test_user_indices"].astype(np.int64)

    check(cold_users == 0, f"held-out: {cold_users} pool users have no W2 training ratings")
    check(int(w2_cold_item.sum()) == 0, f"held-out: {int(w2_cold_item.sum())} slots hold a movie unseen in W2 train")
    check(int(rating_in_w2_train.sum()) == 0, f"held-out: {int(rating_in_w2_train.sum())} slot ratings are in W2 train")
    for name, match in (("clean val (RL train/val sessions)", val_match), ("clean test (RL test sessions)", test_match)):
        check(
            match["rating_and_timestamp_match"] == match["slots"],
            f"held-out: only {match['rating_and_timestamp_match']}/{match['slots']} slots match a W2 {name} rating",
        )
    if failures:
        fail_now()

    # 5. Score every real (user, slot) pair with the Workshop 2 model
    n_pool, n_slots = cand_items.shape
    u_flat = torch.tensor(w2_user_idx[rows], dtype=torch.long)
    i_flat = torch.tensor(w2_item_idx[rows, slot_idx], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        valid_scores = predict_batches(model, u_flat, i_flat, device=device, clip=False)
    scores = np.full((n_pool, n_slots), PADDING_SCORE, dtype=np.float32)
    scores[rows, slot_idx] = valid_scores.astype(np.float32)
    real = scores[valid]
    gaps = [np.diff(np.sort(scores[r][valid[r]])) for r in range(n_pool) if valid[r].sum() > 1]
    min_gap = float(min(g.min() for g in gaps)) if gaps else float("nan")

    meta = {
        "format_version": FORMAT_VERSION,
        "created_by": "workshop3/src/w2_scores.py",
        "score_definition": SCORE_DEFINITION,
        "padding_score": float(PADDING_SCORE),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "checkpoint": {
            "path": repo_relative(Path(checkpoint_path)),
            "sha256": sha256_file(Path(checkpoint_path)),
            "epoch": int(payload["epoch"]),
            "phase": str(payload["phase"]),
        },
        "cache": {"path": repo_relative(Path(cache_path)), "sha256": sha256_file(Path(cache_path))},
        "alignment": {
            "user_mapping": "processed_cache.npz pool_user_ids (MovieLens userId) -> W2 user2idx",
            "item_mapping": "processed_cache.npz cand_movie_ids (MovieLens movieId) -> W2 item2idx",
            "w2_n_users": n_users,
            "w2_n_items": n_items,
            "w2_train_ratings": int(len(train_df)),
            "w2_clean_val_ratings": int(len(val_df)),
            "w2_clean_test_ratings": int(len(test_df)),
            "filtered_ratings": n_filtered,
            "split_matches_w3_dataset_stats": True,
            "w2_train_end_timestamp": w2_train_end,
            "pool_users": int(n_pool),
            "pool_users_mapped": int((w2_user_idx >= 0).sum()),
            "candidate_slots": int(cand_items.size),
            "valid_slots": int(valid.sum()),
            "padding_slots": int((~valid).sum()),
            "valid_slots_mapped": int((valid & (w2_item_idx >= 0)).sum()),
            "valid_slots_test_users": int(valid[test_rows].sum()),
            "w3_user_index_equals_w2_user_index": bool(np.array_equal(cache["pool_users"], w2_user_idx)),
            "w3_item_index_equals_w2_item_index": bool(np.array_equal(cand_items[valid], w2_item_idx[valid])),
            "min_w2_train_ratings_per_rl_user": int(train_counts.min()),
            "cold_users": cold_users,
            "cold_item_slots": int(w2_cold_item.sum()),
            "cold_item_slots_test_users": int(w2_cold_item[test_rows].sum()),
            "slots_with_rating_in_w2_train": int(rating_in_w2_train.sum()),
            "slots_with_rating_in_w2_train_test_users": int(rating_in_w2_train[test_rows].sum()),
            "no_session_rating_in_w2_train": bool(int(rating_in_w2_train.sum()) == 0),
            "slots_in_w2_clean_val": val_match,
            "slots_in_w2_clean_test": test_match,
        },
        "inference_check": {
            "test_metrics": metric_check,
            "ranking_at_10": {
                "reproduced": {k: round(float(v), 4) for k, v in ranking.items()},
                "official": official_ranking,
                "ranking_test_users": ranking_users,
            },
            "side_feature_buffers_max_abs_diff": buffer_diff,
            "global_mean": global_mean,
        },
        "score_stats": {
            "scope": "real slots only (cand_valid == 1)",
            "min": float(real.min()),
            "max": float(real.max()),
            "slots_above_5": int((real > 5.0).sum()),
            "slots_below_0_5": int((real < 0.5).sum()),
            "min_gap_between_slot_scores": min_gap,
        },
    }
    arrays = {
        "scores": scores,
        "cand_valid": cache["cand_valid"].astype(np.int8),
        "cand_items": cand_items,
        "cand_movie_ids": cand_movie_ids,
        "pool_user_ids": pool_user_ids,
        "w2_user_idx": w2_user_idx.astype(np.int32),
        "w2_item_idx": w2_item_idx.astype(np.int32),
        "w2_cold_item": w2_cold_item,
        "rating_in_w2_train": rating_in_w2_train,
        "checkpoint_sha256": np.array(meta["checkpoint"]["sha256"]),
        "meta_json": np.array(json.dumps(meta, sort_keys=True)),
    }
    print(f"[w2_scores] Scored {int(valid.sum()):,} real candidate slots ({time.time() - t0:.1f}s)")
    return arrays, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--device", default="auto", help="auto (default), cpu, or cuda")
    parser.add_argument("--out", default=str(DEFAULT_SCORES_PATH), help="output .npz path")
    args = parser.parse_args()

    arrays, meta = build_w2_candidate_scores(device_name=args.device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    with np.load(DEFAULT_CACHE_PATH) as npz:
        load_w2_candidate_scores(out_path, cache_data={key: npz[key] for key in npz.files})
    print(f"[w2_scores] Saved {repo_relative(out_path)} ({out_path.stat().st_size / 1024:.1f} KiB)")
    print(json.dumps(meta["alignment"], indent=2))
    print(json.dumps(meta["inference_check"], indent=2))
    print(json.dumps(meta["score_stats"], indent=2))


if __name__ == "__main__":
    main()
