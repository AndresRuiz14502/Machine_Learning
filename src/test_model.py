"""Load saved checkpoint (checkpoints/model_best.pt) and run inference on new data."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.data_loader import TOP_K, prepare_datasets
from src.model import HybridNeuMF
from src.train import CHECKPOINT_PATH


def load_best_checkpoint(
    checkpoint_path: Path = CHECKPOINT_PATH,
    device: torch.device | None = None,
) -> tuple[HybridNeuMF, dict]:
    """Reconstruct HybridNeuMF and load trained weights from checkpoints/model_best.pt."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Run '.venv/bin/python -m src.train' first."
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]

    model = HybridNeuMF(
        n_users=cfg["n_users"],
        n_items=cfg["n_items"],
        n_genres=cfg["n_genres"],
        n_user_Prior_feats=cfg.get("n_user_Prior_feats", 2),
        n_item_prior_feats=cfg.get("n_item_prior_feats", 3),
        gmf_dim=cfg["gmf_dim"],
        mlp_emb_dim=cfg["mlp_emb_dim"],
        genre_proj_dim=cfg["genre_proj_dim"],
        mlp_hidden_dims=tuple(cfg["mlp_hidden_dims"]),
        dropout_rate=cfg["dropout_rate"],
        global_mean=cfg["global_mean"],
    ).to(device)

    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


@torch.no_grad()
def recommend_top_k(
    model: HybridNeuMF,
    user_idx: int,
    seen_items: set[int],
    idx2item: dict[int, int],
    title_by_movie_id: dict[int, str],
    k: int = TOP_K,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Generate Top-K unseen movie recommendations for a given user."""
    if device is None:
        device = next(model.parameters()).device

    candidate_items = np.array(
        [i for i in range(model.n_items) if i not in seen_items], dtype=np.int64
    )
    u_tensor = torch.full((len(candidate_items),), int(user_idx), dtype=torch.long, device=device)
    i_tensor = torch.tensor(candidate_items, dtype=torch.long, device=device)

    scores = model(u_tensor, i_tensor).clamp(0.5, 5.0).cpu().numpy()
    probs = model.predict_proba(u_tensor, i_tensor).cpu().numpy()

    top_order = np.argsort(-scores)[:k]
    rows = []
    for rank, pos in enumerate(top_order, start=1):
        i_idx = int(candidate_items[pos])
        movie_id = idx2item.get(i_idx, i_idx)
        rows.append(
            {
                "rank": rank,
                "movieId": movie_id,
                "title": title_by_movie_id.get(movie_id, f"Movie #{movie_id}"),
                "pred_rating": round(float(scores[pos]), 4),
                "like_prob": round(float(probs[pos]), 4),
            }
        )
    return pd.DataFrame(rows)


def run_inference_verification(checkpoint_path: Path = CHECKPOINT_PATH) -> dict:
    """Verify checkpoint loading, batch inference on unseen test pairs, and Top-K ranking."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_best_checkpoint(checkpoint_path=checkpoint_path, device=device)

    print("=== Checkpoint Verification ===")
    print(f"Loaded from     : {checkpoint_path}")
    print(f"Best Epoch      : {payload['epoch']} ({payload['phase']})")
    print(f"Best Val RMSE   : {payload['best_val_rmse']:.4f}")
    print(f"Model Config    : {payload['config']}")

    # 1. Synthetic / new user-item batch check (including boundary and out-of-vocabulary indices)
    synth_u = torch.tensor([0, 10, 250, 1024, model.n_users + 50], dtype=torch.long, device=device)
    synth_i = torch.tensor([0, 25, 500, 2048, model.n_items + 50], dtype=torch.long, device=device)
    with torch.no_grad():
        synth_preds = model(synth_u, synth_i).clamp(0.5, 5.0).cpu().numpy()
        synth_probs = model.predict_proba(synth_u, synth_i).cpu().numpy()

    assert synth_preds.shape == (5,), "Unexpected synthetic output shape"
    assert np.all(np.isfinite(synth_preds)), "Non-finite predictions detected"
    print("\n1. Synthetic / Cold-Start Safeguard Inference:")
    for u_val, i_val, r_hat, p_hat in zip(
        synth_u.cpu().tolist(), synth_i.cpu().tolist(), synth_preds, synth_probs
    ):
        print(
            f"   (u={u_val:6d}, i={i_val:6d}) -> pred_rating = {r_hat:.4f} stars | P(like) = {p_hat:.4f}"
        )

    # 2. Held-out test dataset sample predictions
    data = prepare_datasets(save_splits_file=False, verbose=False)
    test_df = data["test_df"]
    train_df = data["train_df"]
    movies = data["movies"]
    title_map = movies.set_index("movieId")["title"].to_dict()

    sample_test = test_df.head(10).copy()
    u_t = torch.tensor(sample_test["u"].to_numpy(), dtype=torch.long, device=device)
    i_t = torch.tensor(sample_test["i"].to_numpy(), dtype=torch.long, device=device)

    with torch.no_grad():
        sample_preds = model(u_t, i_t).clamp(0.5, 5.0).cpu().numpy()
        sample_probs = model.predict_proba(u_t, i_t).cpu().numpy()

    sample_table = pd.DataFrame(
        {
            "userId": sample_test["userId"].to_numpy(),
            "movieId": sample_test["movieId"].to_numpy(),
            "title": sample_test["movieId"].map(title_map).to_numpy(),
            "true_rating": sample_test["rating"].to_numpy(),
            "pred_rating": np.round(sample_preds, 4),
            "abs_error": np.round(np.abs(sample_test["rating"].to_numpy() - sample_preds), 4),
            "like_prob": np.round(sample_probs, 4),
        }
    )
    print("\n2. Inference on 10 Held-Out Test Interactions:")
    print(sample_table.to_string(index=False))

    # 3. Personalized Top-10 Recommendations for a test user
    sample_u = int(sample_test["u"].iloc[0])
    sample_uid = int(sample_test["userId"].iloc[0])
    seen_u = set(train_df[train_df["u"] == sample_u]["i"].unique())
    recs_df = recommend_top_k(
        model=model,
        user_idx=sample_u,
        seen_items=seen_u,
        idx2item=data["idx2item"],
        title_by_movie_id=title_map,
        k=TOP_K,
        device=device,
    )
    print(f"\n3. Top-{TOP_K} Personalized Recommendations for UserId={sample_uid} (u={sample_u}):")
    print(recs_df.to_string(index=False))

    return {
        "sample_predictions": sample_table,
        "top_k_recommendations": recs_df,
    }


if __name__ == "__main__":
    run_inference_verification()
