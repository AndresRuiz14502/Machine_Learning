"""Latent space visualization (t-SNE & UMAP) and Silhouette score comparison (AE vs VAE vs W2)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.autoencoder import build_workshop4_datasets
from src.data_loader import SEED
from src.evaluate_ae import load_w4_models
from src.train_ae import W4_RUNS_DIR


def analyze_latent_spaces(n_vis_samples: int = 3_500) -> dict:
    """Encode test set with AE, beta-VAEs (mu), and W2 Supervised model; compute t-SNE, UMAP, and Silhouette."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_w4_models(device=device)
    ae_model = loaded["ae_model"]
    dae_model = loaded["dae_model"]
    best_vae = loaded["best_vae"]
    best_beta = loaded["best_beta"]
    all_beta_vaes = loaded["all_beta_vaes"]

    w4_data = build_workshop4_datasets(max_train_normal=200_000, seed=SEED, device=device)
    x_te_all = torch.tensor(w4_data["X_test_all"], dtype=torch.float32, device=device)
    y_te_anom = w4_data["y_test_anomaly"]
    te_ratings = w4_data["test_ratings"]
    w2_lat_test = w4_data["w2_latent_test"]

    # 1. Encode all 23,551 test samples with AE, Denoising AE, and each beta-VAE (using mu)
    with torch.no_grad():
        z_ae_all = ae_model.encode(x_te_all).cpu().numpy()
        z_dae_all = dae_model.encode(x_te_all).cpu().numpy()
        mu_vae_best_all, _ = best_vae.encode(x_te_all)
        mu_vae_best_all = mu_vae_best_all.cpu().numpy()

        mu_by_beta = {}
        for b_val, m_b in sorted(all_beta_vaes.items()):
            mu_b, _ = m_b.encode(x_te_all)
            mu_by_beta[b_val] = mu_b.cpu().numpy()

    # Subsample deterministically for 2D t-SNE / UMAP projection and Silhouette evaluation
    rng = np.random.default_rng(SEED)
    vis_idx = np.sort(rng.choice(len(x_te_all), size=min(n_vis_samples, len(x_te_all)), replace=False))
    y_vis_anom = y_te_anom[vis_idx]
    r_vis = te_ratings[vis_idx]

    # Extreme rating mask (ratings <= 2.5 vs ratings >= 4.5) for high-contrast separability check
    ext_mask = (r_vis <= 2.5) | (r_vis >= 4.5)
    y_ext = (r_vis[ext_mask] < 4.0).astype(np.int32)

    representations = {
        "Standard AE (z_dim=16)": z_ae_all[vis_idx],
        "Denoising AE (z_dim=16)": z_dae_all[vis_idx],
        "VAE (beta=0.5, mu)": mu_by_beta[0.5][vis_idx],
        "VAE (beta=1.0, mu)": mu_by_beta[1.0][vis_idx],
        "VAE (beta=4.0, mu)": mu_by_beta[4.0][vis_idx],
        "W2 Supervised HybridNeuMF": w2_lat_test[vis_idx],
    }

    # 2. Compute Silhouette Score and Davies-Bouldin Index on Full Latent Vectors
    sep_rows = []
    for name, Z_mat in representations.items():
        Z_norm = StandardScaler().fit_transform(Z_mat)
        sil_all = float(silhouette_score(Z_norm, y_vis_anom))
        sil_ext = float(silhouette_score(Z_norm[ext_mask], y_ext))
        db_idx = float(davies_bouldin_score(Z_norm, y_vis_anom))
        sep_rows.append(
            {
                "Representation": name,
                "Latent_Dim": int(Z_mat.shape[1]),
                "Silhouette_All_Test": round(sil_all, 4),
                "Silhouette_Extreme_Ratings": round(sil_ext, 4),
                "Davies_Bouldin_Index": round(db_idx, 4),
            }
        )

    # 3. Compute 2D t-SNE and 2D UMAP projections for AE, Best VAE (mu), and W2 Supervised
    vis_models = [
        ("Standard AE (Unsupervised z)", z_ae_all[vis_idx]),
        (f"Variational AE (beta={best_beta}, mu)", mu_vae_best_all[vis_idx]),
        ("W2 HybridNeuMF (Supervised Frozen)", w2_lat_test[vis_idx]),
    ]

    tsne_coords = {}
    umap_coords = {}
    sil_2d_rows = []

    for title, Z_mat in vis_models:
        Z_scaled = StandardScaler().fit_transform(Z_mat)
        tsne = TSNE(n_components=2, perplexity=35, max_iter=800, random_state=SEED, init="pca")
        z_tsne = tsne.fit_transform(Z_scaled)
        tsne_coords[title] = z_tsne

        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.15, random_state=SEED)
        z_umap = reducer.fit_transform(Z_scaled)
        umap_coords[title] = z_umap

        sil_2d_rows.append(
            {
                "Model": title,
                "Silhouette_2D_tSNE": round(float(silhouette_score(z_tsne, y_vis_anom)), 4),
                "Silhouette_2D_UMAP": round(float(silhouette_score(z_umap, y_vis_anom)), 4),
                "Silhouette_2D_tSNE_Extreme": round(
                    float(silhouette_score(z_tsne[ext_mask], y_ext)), 4
                ),
                "Silhouette_2D_UMAP_Extreme": round(
                    float(silhouette_score(z_umap[ext_mask], y_ext)), 4
                ),
            }
        )

    sep_df = pd.DataFrame(sep_rows)
    sil_2d_df = pd.DataFrame(sil_2d_rows)

    print("\n=== 1. Latent Space Separability Metrics (Full Latent Vectors) ===")
    print(sep_df.to_string(index=False))

    print("\n=== 2. 2D Projection Silhouette Scores (t-SNE & UMAP) ===")
    print(sil_2d_df.to_string(index=False))

    # 4. Plot 2x3 Grid: Top Row = t-SNE (AE, VAE, W2), Bottom Row = UMAP (AE, VAE, W2)
    W4_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.8))

    for col_idx, (title, _) in enumerate(vis_models):
        # Top row: t-SNE colored by Normal (like>=4.0) vs Anomaly (like<4.0)
        ax_t = axes[0, col_idx]
        zt = tsne_coords[title]
        ax_t.scatter(
            zt[y_vis_anom == 0, 0],
            zt[y_vis_anom == 0, 1],
            c="#4C72B0",
            s=8,
            alpha=0.50,
            label="Normal (like >= 4.0)",
        )
        ax_t.scatter(
            zt[y_vis_anom == 1, 0],
            zt[y_vis_anom == 1, 1],
            c="#C44E52",
            s=8,
            alpha=0.50,
            label="Anomaly (like < 4.0)",
        )
        sil_t = sil_2d_rows[col_idx]["Silhouette_2D_tSNE"]
        ax_t.set_title(f"t-SNE: {title}\n(2D Silhouette = {sil_t:.4f})", fontsize=10.5)
        ax_t.set_xlabel("t-SNE Dim 1")
        ax_t.set_ylabel("t-SNE Dim 2")
        ax_t.legend(loc="upper right", fontsize=8, markerscale=1.8)
        ax_t.grid(alpha=0.25)

        # Bottom row: UMAP colored by continuous true rating (0.5 - 5.0 stars)
        ax_u = axes[1, col_idx]
        zu = umap_coords[title]
        sc = ax_u.scatter(zu[:, 0], zu[:, 1], c=r_vis, cmap="viridis", s=8, alpha=0.60)
        sil_u = sil_2d_rows[col_idx]["Silhouette_2D_UMAP"]
        ax_u.set_title(f"UMAP: {title}\n(2D Silhouette = {sil_u:.4f})", fontsize=10.5)
        ax_u.set_xlabel("UMAP Dim 1")
        ax_u.set_ylabel("UMAP Dim 2")
        plt.colorbar(sc, ax=ax_u, fraction=0.046, pad=0.04, label="True Rating (Stars)")
        ax_u.grid(alpha=0.25)

    plt.suptitle(
        "Latent Space Comparison: Unsupervised AE & VAE vs. Workshop 2 Supervised HybridNeuMF",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    latent_fig_path = W4_RUNS_DIR / "latent_space_tsne_umap.png"
    fig.savefig(latent_fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved t-SNE and UMAP latent space visualization to: {latent_fig_path}")

    sep_df.to_csv(W4_RUNS_DIR / "latent_separability_metrics.csv", index=False)
    sil_2d_df.to_csv(W4_RUNS_DIR / "latent_2d_silhouette.csv", index=False)

    summary = {
        "latent_separability": sep_rows,
        "silhouette_2d": sil_2d_rows,
    }
    with open(W4_RUNS_DIR / "latent_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    analyze_latent_spaces()
