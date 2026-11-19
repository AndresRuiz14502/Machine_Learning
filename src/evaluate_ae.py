"""Evaluation of AE/VAE reconstruction quality and anomaly detection on the held-out test set."""

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
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.autoencoder import (
    StandardAutoencoder,
    VariationalAutoencoder,
    build_workshop4_datasets,
)
from src.data_loader import SEED
from src.train_ae import AE_CHECKPOINT_PATH, VAE_CHECKPOINT_PATH, W4_RUNS_DIR


def load_w4_models(
    ae_path: Path = AE_CHECKPOINT_PATH,
    vae_path: Path = VAE_CHECKPOINT_PATH,
    device: torch.device | None = None,
) -> dict:
    """Load trained Standard AE, Denoising AE, and beta-VAE checkpoints."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae_payload = torch.load(ae_path, map_location=device, weights_only=False)
    ae_cfg = ae_payload["config"]
    ae_model = StandardAutoencoder(
        input_dim=ae_cfg["input_dim"],
        hidden_dims=tuple(ae_cfg["hidden_dims"]),
        z_dim=ae_cfg["z_dim"],
        dropout_rate=ae_cfg["dropout_rate"],
    ).to(device)
    ae_model.load_state_dict(ae_payload["model_state_dict"])
    ae_model.eval()

    dae_model = StandardAutoencoder(
        input_dim=ae_cfg["input_dim"],
        hidden_dims=tuple(ae_cfg["hidden_dims"]),
        z_dim=ae_cfg["z_dim"],
        dropout_rate=ae_cfg["dropout_rate"],
    ).to(device)
    dae_model.load_state_dict(ae_payload["dae_state_dict"])
    dae_model.eval()

    vae_payload = torch.load(vae_path, map_location=device, weights_only=False)
    vae_cfg = vae_payload["config"]
    best_vae = VariationalAutoencoder(
        input_dim=vae_cfg["input_dim"],
        hidden_dims=tuple(vae_cfg["hidden_dims"]),
        z_dim=vae_cfg["z_dim"],
        beta=vae_cfg["beta"],
        dropout_rate=vae_cfg["dropout_rate"],
    ).to(device)
    best_vae.load_state_dict(vae_payload["model_state_dict"])
    best_vae.eval()

    all_beta_vaes = {}
    for beta_str, st_dict in vae_payload["all_beta_state_dicts"].items():
        b_val = float(beta_str)
        m_b = VariationalAutoencoder(
            input_dim=vae_cfg["input_dim"],
            hidden_dims=tuple(vae_cfg["hidden_dims"]),
            z_dim=vae_cfg["z_dim"],
            beta=b_val,
            dropout_rate=vae_cfg["dropout_rate"],
        ).to(device)
        m_b.load_state_dict(st_dict)
        m_b.eval()
        all_beta_vaes[b_val] = m_b

    return {
        "ae_model": ae_model,
        "dae_model": dae_model,
        "best_vae": best_vae,
        "best_beta": float(vae_payload["best_beta"]),
        "all_beta_vaes": all_beta_vaes,
        "ae_payload": ae_payload,
        "vae_payload": vae_payload,
    }


def compute_reconstruction_metrics(x_true: np.ndarray, x_recon: np.ndarray) -> dict[str, float]:
    """Compute MSE, RMSE, MAE, R2, sample-wise Pearson r, and Cosine Similarity."""
    mse_val = float(mean_squared_error(x_true, x_recon))
    rmse_val = float(np.sqrt(mse_val))
    mae_val = float(mean_absolute_error(x_true, x_recon))
    r2_val = float(r2_score(x_true.ravel(), x_recon.ravel()))

    # Sample-wise Pearson correlation across the 64 features
    x_c = x_true - x_true.mean(axis=1, keepdims=True)
    xr_c = x_recon - x_recon.mean(axis=1, keepdims=True)
    num = np.sum(x_c * xr_c, axis=1)
    den = np.sqrt(np.sum(x_c**2, axis=1) * np.sum(xr_c**2, axis=1)) + 1e-8
    sample_corr = float(np.mean(num / den))

    # Cosine similarity
    cos_num = np.sum(x_true * x_recon, axis=1)
    cos_den = np.linalg.norm(x_true, axis=1) * np.linalg.norm(x_recon, axis=1) + 1e-8
    cos_sim = float(np.mean(cos_num / cos_den))

    return {
        "MSE": round(mse_val, 5),
        "RMSE": round(rmse_val, 5),
        "MAE": round(mae_val, 5),
        "R2": round(r2_val, 4),
        "Pearson_r": round(sample_corr, 4),
        "Cosine_Sim": round(cos_sim, 4),
    }


def calibrate_threshold_on_val(val_scores: np.ndarray, y_val_anom: np.ndarray) -> float:
    """Tune anomaly score decision threshold strictly on the validation split to maximize F1."""
    quantiles = np.percentile(val_scores, np.linspace(15, 85, 71))
    best_thr = float(np.median(val_scores))
    best_f1 = -1.0
    for thr in quantiles:
        pred = (val_scores >= thr).astype(np.int32)
        f1_c = float(f1_score(y_val_anom, pred, zero_division=0))
        if f1_c > best_f1:
            best_f1 = f1_c
            best_thr = float(thr)
    return best_thr


def plot_sample_reconstructions(
    x_test: np.ndarray,
    ae_recon: np.ndarray,
    vae_recon: np.ndarray,
    y_test_anom: np.ndarray,
    save_path: Path,
    n_samples: int = 8,
) -> None:
    """Plot 8 side-by-side test sample profiles (Original, AE, VAE) and element-wise difference maps."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    norm_idx = np.where(y_test_anom == 0)[0][: n_samples // 2]
    anom_idx = np.where(y_test_anom == 1)[0][: n_samples // 2]
    sel_idx = np.concatenate([norm_idx, anom_idx])

    orig_panel = x_test[sel_idx]
    ae_panel = ae_recon[sel_idx]
    vae_panel = vae_recon[sel_idx]
    diff_ae = np.abs(orig_panel - ae_panel)
    diff_vae = np.abs(orig_panel - vae_panel)

    row_labels = [f"Normal #{k+1}" for k in range(len(norm_idx))] + [
        f"Anomaly #{k+1}" for k in range(len(anom_idx))
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.6), sharey=True)
    panels = [
        (orig_panel, "Original Input (x)", "coolwarm", -2.5, 2.5),
        (ae_panel, "AE Reconstruction", "coolwarm", -2.5, 2.5),
        (vae_panel, "VAE Reconstruction", "coolwarm", -2.5, 2.5),
        (diff_ae, "|Original - AE| Error", "magma", 0.0, 2.0),
        (diff_vae, "|Original - VAE| Error", "magma", 0.0, 2.0),
    ]

    for ax, (mat, title, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("Feature Index (0..63)")
        ax.axhline(len(norm_idx) - 0.5, color="white", linestyle="--", linewidth=1.5)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[0].set_yticks(range(len(sel_idx)))
    axes[0].set_yticklabels(row_labels)
    plt.suptitle(
        "Side-by-Side Test Sample Reconstructions (Top 4: Normal like>=4.0 | Bottom 4: Anomaly like<4.0)",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Sample reconstruction comparison panel saved to: {save_path}")


def plot_anomaly_detection_results(
    y_test_anom: np.ndarray,
    ae_scores: np.ndarray,
    vae_recon_scores: np.ndarray,
    vae_elbo_scores: np.ndarray,
    dae_scores: np.ndarray,
    save_path: Path,
) -> None:
    """Plot ROC curves and reconstruction error distributions for Normal vs Anomaly test samples."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # Panel 1: ROC Curves
    curves = [
        ("Standard AE (Recon MSE)", ae_scores, "#4C72B0", "-"),
        ("Denoising AE (Recon MSE)", dae_scores, "#55A868", "--"),
        ("VAE (Recon MSE Only)", vae_recon_scores, "#8172B3", ":"),
        ("VAE (ELBO = Recon + beta*KL)", vae_elbo_scores, "#C44E52", "-."),
    ]
    for label, sc, col, ls in curves:
        fpr, tpr, _ = roc_curve(y_test_anom, sc)
        auc_val = roc_auc_score(y_test_anom, sc)
        axes[0].plot(fpr, tpr, color=col, linestyle=ls, linewidth=2.0, label=f"{label} (AUC={auc_val:.4f})")

    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.6, label="Random Baseline (AUC=0.5000)")
    axes[0].set_title("Unsupervised Anomaly Detection ROC Curves (Test Set)")
    axes[0].set_xlabel("False Positive Rate (Normal Flagged as Anomaly)")
    axes[0].set_ylabel("True Positive Rate (Anomaly Recall)")
    axes[0].legend(loc="lower right", fontsize=8.8)
    axes[0].grid(alpha=0.3)

    # Panel 2: Distribution of AE Reconstruction Error on Normal vs Anomaly
    norm_err = np.clip(ae_scores[y_test_anom == 0], 0, np.percentile(ae_scores, 98))
    anom_err = np.clip(ae_scores[y_test_anom == 1], 0, np.percentile(ae_scores, 98))
    axes[1].hist(norm_err, bins=45, density=True, alpha=0.60, color="#4C72B0", label=f"Normal (like>=4.0, mean={ae_scores[y_test_anom==0].mean():.3f})")
    axes[1].hist(anom_err, bins=45, density=True, alpha=0.60, color="#C44E52", label=f"Anomaly (like<4.0, mean={ae_scores[y_test_anom==1].mean():.3f})")
    axes[1].set_title("Test Reconstruction Error Distribution: Normal vs. Anomaly")
    axes[1].set_xlabel("Per-Sample Reconstruction MSE")
    axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Anomaly detection ROC and distribution plot saved to: {save_path}")


def evaluate_workshop4() -> dict:
    """Run full reconstruction quality, denoising, and anomaly detection evaluation on the test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_w4_models(device=device)
    ae_model = loaded["ae_model"]
    dae_model = loaded["dae_model"]
    best_vae = loaded["best_vae"]
    best_beta = loaded["best_beta"]
    all_beta_vaes = loaded["all_beta_vaes"]

    w4_data = build_workshop4_datasets(max_train_normal=200_000, seed=SEED, device=device)
    x_va_all = torch.tensor(w4_data["X_val_all"], dtype=torch.float32, device=device)
    y_va_anom = w4_data["y_val_anomaly"]

    x_te_norm = torch.tensor(w4_data["X_test_norm"], dtype=torch.float32, device=device)
    x_te_all = torch.tensor(w4_data["X_test_all"], dtype=torch.float32, device=device)
    y_te_anom = w4_data["y_test_anomaly"]

    # 1. Reconstruction Quality on Normal Test Data and Mixed Test Data
    with torch.no_grad():
        ae_recon_norm, _ = ae_model(x_te_norm)
        dae_recon_norm, _ = dae_model(x_te_norm)
        vae_recon_norm, _, _, _ = best_vae(x_te_norm)

        # Denoising test: corrupt x_te_norm with Gaussian noise (std=0.25) and measure clean recovery
        torch.manual_seed(SEED)
        x_te_noisy = x_te_norm + 0.25 * torch.randn_like(x_te_norm)
        ae_denoise_out, _ = ae_model(x_te_noisy)
        dae_denoise_out, _ = dae_model(x_te_noisy)

        ae_recon_all, _ = ae_model(x_te_all)
        vae_recon_all, _, _, _ = best_vae(x_te_all)

    x_te_norm_np = x_te_norm.cpu().numpy()
    recon_table = {
        "Standard AE (z=16) [Normal Test]": compute_reconstruction_metrics(
            x_te_norm_np, ae_recon_norm.cpu().numpy()
        ),
        "Denoising AE (z=16) [Normal Test]": compute_reconstruction_metrics(
            x_te_norm_np, dae_recon_norm.cpu().numpy()
        ),
        f"Best VAE (beta={best_beta}, z=16) [Normal Test]": compute_reconstruction_metrics(
            x_te_norm_np, vae_recon_norm.cpu().numpy()
        ),
        "Standard AE on Noisy Input (std=0.25)": compute_reconstruction_metrics(
            x_te_norm_np, ae_denoise_out.cpu().numpy()
        ),
        "Denoising AE on Noisy Input (std=0.25)": compute_reconstruction_metrics(
            x_te_norm_np, dae_denoise_out.cpu().numpy()
        ),
    }
    for b_val, m_b in sorted(all_beta_vaes.items()):
        with torch.no_grad():
            xr_b, _, _, _ = m_b(x_te_norm)
        recon_table[f"VAE (beta={b_val}) [Normal Test]"] = compute_reconstruction_metrics(
            x_te_norm_np, xr_b.cpu().numpy()
        )

    recon_df = pd.DataFrame(recon_table).T
    print("\n=== 1. Test Set Reconstruction Quality Evaluation ===")
    print(recon_df.to_string())

    # 2. Sample Reconstructions Visualization (8 test pairs + difference maps)
    plot_sample_reconstructions(
        x_test=w4_data["X_test_all"],
        ae_recon=ae_recon_all.cpu().numpy(),
        vae_recon=vae_recon_all.cpu().numpy(),
        y_test_anom=y_te_anom,
        save_path=W4_RUNS_DIR / "reconstruction_samples.png",
        n_samples=8,
    )

    # 3. Feature-Group Reconstruction Error Analysis (Normal vs. Anomaly)
    err_norm_by_feat = np.mean((w4_data["X_test_all"][y_te_anom == 0] - ae_recon_all.cpu().numpy()[y_te_anom == 0]) ** 2, axis=0)
    err_anom_by_feat = np.mean((w4_data["X_test_all"][y_te_anom == 1] - ae_recon_all.cpu().numpy()[y_te_anom == 1]) ** 2, axis=0)
    feat_group_summary = pd.DataFrame(
        [
            {
                "Feature Group": "User & Item Historical Stats (cols 0..7)",
                "Normal_MSE": round(float(err_norm_by_feat[:8].mean()), 4),
                "Anomaly_MSE": round(float(err_anom_by_feat[:8].mean()), 4),
                "Anomaly_to_Normal_Ratio": round(float(err_anom_by_feat[:8].mean() / max(err_norm_by_feat[:8].mean(), 1e-6)), 2),
            },
            {
                "Feature Group": "Multi-Hot Genres & Movie Age (cols 8..28)",
                "Normal_MSE": round(float(err_norm_by_feat[8:29].mean()), 4),
                "Anomaly_MSE": round(float(err_anom_by_feat[8:29].mean()), 4),
                "Anomaly_to_Normal_Ratio": round(float(err_anom_by_feat[8:29].mean() / max(err_norm_by_feat[8:29].mean(), 1e-6)), 2),
            },
            {
                "Feature Group": "Collaborative GMF Interactions (cols 29..47)",
                "Normal_MSE": round(float(err_norm_by_feat[29:48].mean()), 4),
                "Anomaly_MSE": round(float(err_anom_by_feat[29:48].mean()), 4),
                "Anomaly_to_Normal_Ratio": round(float(err_anom_by_feat[29:48].mean() / max(err_norm_by_feat[29:48].mean(), 1e-6)), 2),
            },
            {
                "Feature Group": "Normal-Manifold Compatibility Deviations (cols 48..63)",
                "Normal_MSE": round(float(err_norm_by_feat[48:].mean()), 4),
                "Anomaly_MSE": round(float(err_anom_by_feat[48:].mean()), 4),
                "Anomaly_to_Normal_Ratio": round(float(err_anom_by_feat[48:].mean() / max(err_norm_by_feat[48:].mean(), 1e-6)), 2),
            },
        ]
    )
    print("\n=== 2. Reconstruction Error Breakdown by Feature Group ===")
    print(feat_group_summary.to_string(index=False))

    # 4. Anomaly Detection Evaluation & Cumulative Comparison across Workshops 1, 2, and 4
    with torch.no_grad():
        val_ae_sc = ae_model.reconstruction_mse(x_va_all).cpu().numpy()
        val_dae_sc = dae_model.reconstruction_mse(x_va_all).cpu().numpy()
        val_vae_out = best_vae.anomaly_scores(x_va_all, beta_weight=best_beta)
        val_vae_recon_sc = val_vae_out["recon_mse"].cpu().numpy()
        val_vae_elbo_sc = val_vae_out["elbo_score"].cpu().numpy()

        te_ae_sc = ae_model.reconstruction_mse(x_te_all).cpu().numpy()
        te_dae_sc = dae_model.reconstruction_mse(x_te_all).cpu().numpy()
        te_vae_out = best_vae.anomaly_scores(x_te_all, beta_weight=best_beta)
        te_vae_recon_sc = te_vae_out["recon_mse"].cpu().numpy()
        te_vae_elbo_sc = te_vae_out["elbo_score"].cpu().numpy()

    def eval_anomaly_detector(val_sc: np.ndarray, te_sc: np.ndarray) -> dict[str, float]:
        thr = calibrate_threshold_on_val(val_sc, y_va_anom)
        pred_anom = (te_sc >= thr).astype(np.int32)
        return {
            "Threshold_Val": round(thr, 4),
            "Accuracy": round(float(accuracy_score(y_te_anom, pred_anom)), 4),
            "Precision": round(float(precision_score(y_te_anom, pred_anom, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_te_anom, pred_anom, zero_division=0)), 4),
            "F1": round(float(f1_score(y_te_anom, pred_anom, zero_division=0)), 4),
            "ROC_AUC": round(float(roc_auc_score(y_te_anom, te_sc)), 4),
        }

    anom_results = {
        "W4 Standard AE (Recon MSE)": eval_anomaly_detector(val_ae_sc, te_ae_sc),
        "W4 Denoising AE (Recon MSE)": eval_anomaly_detector(val_dae_sc, te_dae_sc),
        f"W4 VAE (beta={best_beta}, Recon MSE)": eval_anomaly_detector(
            val_vae_recon_sc, te_vae_recon_sc
        ),
        f"W4 VAE (beta={best_beta}, ELBO = Recon + beta*KL)": eval_anomaly_detector(
            val_vae_elbo_sc, te_vae_elbo_sc
        ),
    }

    for b_val, m_b in sorted(all_beta_vaes.items()):
        with torch.no_grad():
            v_sc = m_b.anomaly_scores(x_va_all, beta_weight=b_val)["elbo_score"].cpu().numpy()
            t_sc = m_b.anomaly_scores(x_te_all, beta_weight=b_val)["elbo_score"].cpu().numpy()
        anom_results[f"W4 VAE (beta={b_val}, ELBO)"] = eval_anomaly_detector(v_sc, t_sc)

    # Cumulative Comparison Table (Workshops 1, 2, and 4 on the identical 23,551 test split)
    cumulative_rows = {
        "W1 GlobalMean Baseline": {
            "Paradigm": "W1 Supervised Baseline",
            "Accuracy": 0.6356,
            "Precision": 0.6356,
            "Recall": 1.0000,
            "F1": 0.7772,
            "ROC_AUC": 0.5000,
        },
        "W1 GradientBoostingClf": {
            "Paradigm": "W1 Supervised Tree",
            "Accuracy": 0.6979,
            "Precision": 0.5697,
            "Recall": 0.6986,
            "F1": 0.6276,
            "ROC_AUC": 0.7518,
        },
        "W1 Ridge Threshold": {
            "Paradigm": "W1 Supervised Linear",
            "Accuracy": 0.6987,
            "Precision": 0.7089,
            "Recall": 0.2938,
            "F1": 0.4154,
            "ROC_AUC": 0.7590,
        },
        "W1 LogisticRegression": {
            "Paradigm": "W1 Supervised Linear",
            "Accuracy": 0.7134,
            "Precision": 0.6026,
            "Recall": 0.6270,
            "F1": 0.6146,
            "ROC_AUC": 0.7604,
        },
        "W1 Bias (mu+bu+bi)": {
            "Paradigm": "W1 Collaborative Bias",
            "Accuracy": 0.7128,
            "Precision": 0.6721,
            "Recall": 0.4139,
            "F1": 0.5123,
            "ROC_AUC": 0.7607,
        },
        "W1 RandomForestClf": {
            "Paradigm": "W1 Supervised Tree",
            "Accuracy": 0.7031,
            "Precision": 0.5773,
            "Recall": 0.6914,
            "F1": 0.6293,
            "ROC_AUC": 0.7645,
        },
        "W2 HybridNeuMF (Deep CF)": {
            "Paradigm": "W2 Supervised Deep Learning",
            "Accuracy": 0.7156,
            "Precision": 0.5838,
            "Recall": 0.7651,
            "F1": 0.6623,
            "ROC_AUC": 0.7942,
        },
    }
    for k_name, m_dict in anom_results.items():
        cumulative_rows[k_name] = {
            "Paradigm": "W4 Unsupervised Autoencoder",
            "Accuracy": m_dict["Accuracy"],
            "Precision": m_dict["Precision"],
            "Recall": m_dict["Recall"],
            "F1": m_dict["F1"],
            "ROC_AUC": m_dict["ROC_AUC"],
        }

    cum_df = pd.DataFrame(cumulative_rows).T.sort_values("ROC_AUC", ascending=False)
    print("\n=== 3. Cumulative Anomaly / Preference Detection Comparison (Workshops 1, 2 & 4) ===")
    print(cum_df.to_string())

    plot_anomaly_detection_results(
        y_test_anom=y_te_anom,
        ae_scores=te_ae_sc,
        vae_recon_scores=te_vae_recon_sc,
        vae_elbo_scores=te_vae_elbo_sc,
        dae_scores=te_dae_sc,
        save_path=W4_RUNS_DIR / "anomaly_detection_roc.png",
    )

    W4_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    recon_df.to_csv(W4_RUNS_DIR / "reconstruction_metrics.csv")
    feat_group_summary.to_csv(W4_RUNS_DIR / "feature_group_errors.csv", index=False)
    cum_df.to_csv(W4_RUNS_DIR / "cumulative_comparison.csv")

    eval_payload = {
        "reconstruction_metrics": recon_table,
        "feature_group_errors": feat_group_summary.to_dict(orient="records"),
        "anomaly_detection_results": anom_results,
        "cumulative_comparison": cum_df.to_dict(orient="index"),
    }
    with open(W4_RUNS_DIR / "w4_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)

    return eval_payload


if __name__ == "__main__":
    evaluate_workshop4()
