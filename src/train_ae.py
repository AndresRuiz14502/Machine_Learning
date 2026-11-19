"""Training script for Standard Autoencoder (AE), beta-VAEs, and Denoising AE."""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, silhouette_score
from torch.utils.tensorboard import SummaryWriter

from src.autoencoder import (
    DEFAULT_Z_DIM,
    FEATURE_DIM,
    StandardAutoencoder,
    VariationalAutoencoder,
    build_workshop4_datasets,
)
from src.data_loader import SEED
from src.train import set_all_seeds

AE_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "ae_best.pt"
VAE_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "vae_best.pt"
W4_RUNS_DIR = PROJECT_ROOT / "runs" / "workshop4"

AE_EPOCHS = 35
VAE_EPOCHS = 35
KL_ANNEAL_EPOCHS = 10
BATCH_SIZE = 4_096
LR_AE = 1.5e-3
LR_VAE = 1.5e-3

Z_DIM_CANDIDATES = (8, 16, 32)
BETA_CANDIDATES = (0.5, 1.0, 4.0)


@torch.no_grad()
def eval_ae_mse(model: StandardAutoencoder, x_t: torch.Tensor, device: torch.device) -> float:
    """Compute average reconstruction MSE of StandardAutoencoder on x_t."""
    model.eval()
    errs = []
    for s in range(0, len(x_t), 16_384):
        bx = x_t[s : s + 16_384].to(device)
        errs.append(model.reconstruction_mse(bx).cpu().numpy())
    return float(np.mean(np.concatenate(errs)))


@torch.no_grad()
def eval_vae_losses(
    model: VariationalAutoencoder, x_t: torch.Tensor, beta_eff: float, device: torch.device
) -> dict[str, float]:
    """Compute average validation reconstruction MSE, KL divergence, and ELBO for VAE."""
    model.eval()
    recons, kls = [], []
    for s in range(0, len(x_t), 16_384):
        bx = x_t[s : s + 16_384].to(device)
        mu, logvar = model.encode(bx)
        x_hat = model.decode(mu)
        recons.append((bx - x_hat).pow(2).mean(dim=-1).cpu().numpy())
        kls.append(model.kl_divergence_per_sample(mu, logvar, normalize_by_dim=True).cpu().numpy())
    recon_val = float(np.mean(np.concatenate(recons)))
    kl_val = float(np.mean(np.concatenate(kls)))
    return {
        "recon_mse": recon_val,
        "kl_div": kl_val,
        "elbo": recon_val + float(beta_eff) * kl_val,
    }


def train_standard_ae(
    x_tr: torch.Tensor,
    x_va_norm: torch.Tensor,
    input_dim: int,
    z_dim: int,
    n_epochs: int,
    lr: float,
    device: torch.device,
    noise_std: float = 0.0,
    tb_writer: SummaryWriter | None = None,
    tag_prefix: str = "AE",
    verbose: bool = True,
) -> tuple[StandardAutoencoder, list[dict], float]:
    """Train a Standard or Denoising Autoencoder with Adam and return best checkpoint."""
    set_all_seeds(SEED)
    model = StandardAutoencoder(
        input_dim=input_dim,
        hidden_dims=(128, 64),
        z_dim=z_dim,
        dropout_rate=0.10,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.1)

    best_val_mse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict] = []
    n_tr = len(x_tr)

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(n_tr)
        tr_loss_sum = 0.0
        n_batches = 0

        for s in range(0, n_tr, BATCH_SIZE):
            bx = x_tr[perm[s : s + BATCH_SIZE]].to(device)
            optimizer.zero_grad()
            x_hat, _ = model(bx, noise_std=noise_std)
            loss = (bx - x_hat).pow(2).mean()
            loss.backward()
            optimizer.step()
            tr_loss_sum += float(loss.item())
            n_batches += 1

        scheduler.step()
        tr_mse = tr_loss_sum / max(n_batches, 1)
        va_mse = eval_ae_mse(model, x_va_norm, device=device)

        if va_mse < best_val_mse:
            best_val_mse = va_mse
            best_state = copy.deepcopy(model.state_dict())

        history.append(
            {
                "epoch": epoch,
                "train_recon_mse": tr_mse,
                "val_recon_mse": va_mse,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if tb_writer is not None:
            tb_writer.add_scalar(f"{tag_prefix}/train_recon_mse", tr_mse, epoch)
            tb_writer.add_scalar(f"{tag_prefix}/val_recon_mse", va_mse, epoch)

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == n_epochs):
            print(
                f"[{tag_prefix} z={z_dim}] Epoch {epoch:02d}/{n_epochs:02d} | "
                f"Train MSE: {tr_mse:.4f} | Val MSE: {va_mse:.4f}"
            )

    model.load_state_dict(best_state)
    return model, history, best_val_mse


def train_beta_vae(
    x_tr: torch.Tensor,
    x_va_norm: torch.Tensor,
    input_dim: int,
    z_dim: int,
    beta: float,
    n_epochs: int,
    lr: float,
    device: torch.device,
    tb_writer: SummaryWriter | None = None,
    verbose: bool = True,
) -> tuple[VariationalAutoencoder, list[dict], float]:
    """Train a beta-VAE with linear KL annealing over the first 10 epochs."""
    set_all_seeds(SEED)
    model = VariationalAutoencoder(
        input_dim=input_dim,
        hidden_dims=(128, 64),
        z_dim=z_dim,
        beta=beta,
        dropout_rate=0.10,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.1)

    best_val_elbo = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict] = []
    n_tr = len(x_tr)

    for epoch in range(1, n_epochs + 1):
        model.train()
        # KL Annealing: linearly ramp beta_eff from 0.0 to target beta over KL_ANNEAL_EPOCHS
        anneal_factor = min(1.0, float(epoch - 1) / float(max(KL_ANNEAL_EPOCHS, 1)))
        beta_eff = float(beta) * anneal_factor

        perm = torch.randperm(n_tr)
        tr_recon_sum, tr_kl_sum, tr_elbo_sum = 0.0, 0.0, 0.0
        n_batches = 0

        for s in range(0, n_tr, BATCH_SIZE):
            bx = x_tr[perm[s : s + BATCH_SIZE]].to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar, _ = model(bx)
            losses = model.compute_elbo_loss(bx, x_hat, mu, logvar, beta_eff=beta_eff)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            tr_recon_sum += float(losses["recon_loss"].item())
            tr_kl_sum += float(losses["kl_loss"].item())
            tr_elbo_sum += float(losses["loss"].item())
            n_batches += 1

        scheduler.step()
        tr_recon = tr_recon_sum / max(n_batches, 1)
        tr_kl = tr_kl_sum / max(n_batches, 1)
        tr_elbo = tr_elbo_sum / max(n_batches, 1)

        val_m = eval_vae_losses(model, x_va_norm, beta_eff=beta, device=device)

        if epoch >= KL_ANNEAL_EPOCHS and val_m["elbo"] < best_val_elbo:
            best_val_elbo = val_m["elbo"]
            best_state = copy.deepcopy(model.state_dict())

        history.append(
            {
                "epoch": epoch,
                "beta_target": beta,
                "beta_eff": beta_eff,
                "train_recon_mse": tr_recon,
                "train_kl": tr_kl,
                "train_elbo": tr_elbo,
                "val_recon_mse": val_m["recon_mse"],
                "val_kl": val_m["kl_div"],
                "val_elbo": val_m["elbo"],
            }
        )
        if tb_writer is not None:
            tb_writer.add_scalar(f"VAE_b{beta}/train_recon_mse", tr_recon, epoch)
            tb_writer.add_scalar(f"VAE_b{beta}/val_recon_mse", val_m["recon_mse"], epoch)
            tb_writer.add_scalar(f"VAE_b{beta}/val_kl", val_m["kl_div"], epoch)
            tb_writer.add_scalar(f"VAE_b{beta}/val_elbo", val_m["elbo"], epoch)

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == n_epochs):
            print(
                f"[VAE beta={beta:.1f}] Epoch {epoch:02d}/{n_epochs:02d} (beta_eff={beta_eff:.2f}) | "
                f"Train Recon: {tr_recon:.4f} | Val Recon: {val_m['recon_mse']:.4f} | "
                f"Val KL: {val_m['kl_div']:.4f} | Val ELBO: {val_m['elbo']:.4f}"
            )

    model.load_state_dict(best_state)
    return model, history, best_val_elbo


def plot_w4_learning_curves(
    ae_history: list[dict],
    vae_histories: dict[float, list[dict]],
    z_dim_results: list[dict],
    save_path: Path,
) -> None:
    """Plot and save AE and beta-VAE learning curves and z_dim bottleneck ablation."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ae_df = pd.DataFrame(ae_history)
    axes[0].plot(ae_df["epoch"], ae_df["train_recon_mse"], label="AE Train Recon MSE", color="#4C72B0", linewidth=2)
    axes[0].plot(ae_df["epoch"], ae_df["val_recon_mse"], label="AE Val Recon MSE", color="#C44E52", linestyle="--", linewidth=2)
    axes[0].set_title(f"Standard Autoencoder (z_dim={DEFAULT_Z_DIM}) Learning Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Reconstruction MSE")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    colors_b = {0.5: "#55A868", 1.0: "#8172B3", 4.0: "#DD8452"}
    for beta, hist in vae_histories.items():
        vdf = pd.DataFrame(hist)
        axes[1].plot(
            vdf["epoch"],
            vdf["val_recon_mse"],
            label=f"VAE (beta={beta}) Val Recon",
            color=colors_b[beta],
            linewidth=1.8,
        )
        axes[1].plot(
            vdf["epoch"],
            vdf["val_kl"] * 0.1,
            label=f"VAE (beta={beta}) 0.1x KL",
            color=colors_b[beta],
            linestyle=":",
            linewidth=1.4,
        )
    axes[1].axvline(KL_ANNEAL_EPOCHS, color="gray", linestyle="--", alpha=0.6, label="KL Anneal End")
    axes[1].set_title("Variational Autoencoder (beta-VAE) Dynamics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Loss Component")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    zdf = pd.DataFrame(z_dim_results)
    ax2 = axes[2]
    ax2.plot(zdf["z_dim"], zdf["val_recon_mse"], marker="o", color="#4C72B0", linewidth=2, label="Val Recon MSE")
    ax2.set_title("Bottleneck Dimension (z_dim) Trade-Off")
    ax2.set_xlabel("Latent Bottleneck Dimension (z_dim)")
    ax2.set_ylabel("Validation Reconstruction MSE", color="#4C72B0")
    ax2.set_xticks(zdf["z_dim"].tolist())
    ax2.grid(alpha=0.3)

    ax2_r = ax2.twinx()
    ax2_r.plot(zdf["z_dim"], zdf["val_anomaly_auc"], marker="s", color="#C44E52", linestyle="--", linewidth=2, label="Val Anomaly AUC")
    ax2_r.set_ylabel("Validation Anomaly ROC-AUC", color="#C44E52")

    plt.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Workshop 4 learning curves saved to: {save_path}")


def run_all_workshop4_training() -> dict:
    """Execute AE z_dim ablation, Standard AE training, beta-VAE sweep, and Denoising AE training."""
    t_start = time.time()
    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Workshop 4 Training Pipeline (Device: {device}) ===")

    w4_data = build_workshop4_datasets(max_train_normal=200_000, seed=SEED, device=device)
    input_dim = w4_data["input_dim"]

    x_tr_norm = torch.tensor(w4_data["X_train_norm"], dtype=torch.float32)
    x_va_norm = torch.tensor(w4_data["X_val_norm"], dtype=torch.float32)
    x_va_all = torch.tensor(w4_data["X_val_all"], dtype=torch.float32)
    y_va_anom = w4_data["y_val_anomaly"]

    print(
        f"Normal Training Samples (like==1): {len(x_tr_norm):,} x {input_dim}D | "
        f"Val Normal: {len(x_va_norm):,} | Val Mixed: {len(x_va_all):,}"
    )

    W4_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(W4_RUNS_DIR / "tensorboard"))

    # 1. Bottleneck Dimension Experiment (z_dim in {8, 16, 32})
    print("\n--- 1. Bottleneck Dimension Experiment (z_dim in {8, 16, 32}) ---")
    z_dim_results = []
    ae_models_by_z = {}
    ae_histories_by_z = {}

    for z_dim in Z_DIM_CANDIDATES:
        ae_m, ae_hist, best_va_mse = train_standard_ae(
            x_tr=x_tr_norm,
            x_va_norm=x_va_norm,
            input_dim=input_dim,
            z_dim=z_dim,
            n_epochs=AE_EPOCHS,
            lr=LR_AE,
            device=device,
            noise_std=0.0,
            tb_writer=tb_writer,
            tag_prefix=f"AE_z{z_dim}",
            verbose=(z_dim == DEFAULT_Z_DIM),
        )
        with torch.no_grad():
            va_scores = ae_m.reconstruction_mse(x_va_all.to(device)).cpu().numpy()
            va_z = ae_m.encode(x_va_all[:4000].to(device)).cpu().numpy()
        va_auc = float(roc_auc_score(y_va_anom, va_scores))
        va_sil = float(silhouette_score(va_z, y_va_anom[:4000]))

        z_dim_results.append(
            {
                "z_dim": z_dim,
                "compression_ratio": round(input_dim / z_dim, 2),
                "val_recon_mse": round(best_va_mse, 5),
                "val_anomaly_auc": round(va_auc, 4),
                "val_silhouette": round(va_sil, 4),
            }
        )
        ae_models_by_z[z_dim] = ae_m
        ae_histories_by_z[z_dim] = ae_hist
        print(
            f"  z_dim={z_dim:2d} ({input_dim/z_dim:.1f}x compression) -> "
            f"Val Recon MSE={best_va_mse:.4f} | Val Anomaly AUC={va_auc:.4f} | Silhouette={va_sil:.4f}"
        )

    best_ae = ae_models_by_z[DEFAULT_Z_DIM]
    best_ae_hist = ae_histories_by_z[DEFAULT_Z_DIM]

    # 2. Optional Extension: Denoising Autoencoder (z_dim = 16, Gaussian noise_std = 0.25)
    print("\n--- 2. Denoising Autoencoder Training (Gaussian noise_std = 0.25) ---")
    dae_model, dae_hist, dae_val_mse = train_standard_ae(
        x_tr=x_tr_norm,
        x_va_norm=x_va_norm,
        input_dim=input_dim,
        z_dim=DEFAULT_Z_DIM,
        n_epochs=AE_EPOCHS,
        lr=LR_AE,
        device=device,
        noise_std=0.25,
        tb_writer=tb_writer,
        tag_prefix="DenoisingAE",
        verbose=True,
    )

    # Save best AE + Denoising AE + z_dim ablation in checkpoints/ae_best.pt
    AE_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": best_ae.get_config(),
            "model_state_dict": best_ae.state_dict(),
            "dae_state_dict": dae_model.state_dict(),
            "z_dim_ablation": z_dim_results,
            "history": best_ae_hist,
            "dae_history": dae_hist,
            "seed": SEED,
        },
        AE_CHECKPOINT_PATH,
    )
    print(f"Saved AE checkpoint to: {AE_CHECKPOINT_PATH}")

    # 3. Variational Autoencoder (VAE) Runs for beta in {0.5, 1.0, 4.0}
    print("\n--- 3. Variational Autoencoder (beta-VAE) Sweep (beta in {0.5, 1.0, 4.0}) ---")
    vae_models_by_beta = {}
    vae_histories_by_beta = {}
    beta_summary_rows = []

    for beta in BETA_CANDIDATES:
        vae_m, vae_hist, best_elbo = train_beta_vae(
            x_tr=x_tr_norm,
            x_va_norm=x_va_norm,
            input_dim=input_dim,
            z_dim=DEFAULT_Z_DIM,
            beta=beta,
            n_epochs=VAE_EPOCHS,
            lr=LR_VAE,
            device=device,
            tb_writer=tb_writer,
            verbose=True,
        )
        with torch.no_grad():
            va_out = vae_m.anomaly_scores(x_va_all.to(device), beta_weight=beta)
            mu_va, _ = vae_m.encode(x_va_all[:4000].to(device))
            mu_va_np = mu_va.cpu().numpy()

        va_norm_m = eval_vae_losses(vae_m, x_va_norm, beta_eff=beta, device=device)
        auc_recon = float(roc_auc_score(y_va_anom, va_out["recon_mse"].cpu().numpy()))
        auc_elbo = float(roc_auc_score(y_va_anom, va_out["elbo_score"].cpu().numpy()))
        sil_mu = float(silhouette_score(mu_va_np, y_va_anom[:4000]))

        beta_summary_rows.append(
            {
                "beta": beta,
                "val_recon_mse": round(va_norm_m["recon_mse"], 5),
                "val_kl_div": round(va_norm_m["kl_div"], 5),
                "val_elbo": round(va_norm_m["elbo"], 5),
                "val_auc_recon": round(auc_recon, 4),
                "val_auc_elbo": round(auc_elbo, 4),
                "val_silhouette_mu": round(sil_mu, 4),
            }
        )
        vae_models_by_beta[beta] = vae_m
        vae_histories_by_beta[beta] = vae_hist

    beta_df = pd.DataFrame(beta_summary_rows)
    print("\n=== beta-VAE Validation Summary ===")
    print(beta_df.to_string(index=False))

    # Select best VAE by validation anomaly ELBO AUC (defaulting to beta = 1.0 if tied)
    best_beta_row = max(beta_summary_rows, key=lambda r: r["val_auc_elbo"])
    best_beta = float(best_beta_row["beta"])
    best_vae = vae_models_by_beta[best_beta]

    torch.save(
        {
            "best_beta": best_beta,
            "config": best_vae.get_config(),
            "model_state_dict": best_vae.state_dict(),
            "all_beta_state_dicts": {str(b): m.state_dict() for b, m in vae_models_by_beta.items()},
            "beta_summary": beta_summary_rows,
            "histories": {str(b): h for b, h in vae_histories_by_beta.items()},
            "seed": SEED,
        },
        VAE_CHECKPOINT_PATH,
    )
    print(f"Saved best VAE (beta={best_beta}) checkpoint to: {VAE_CHECKPOINT_PATH}")

    tb_writer.flush()
    tb_writer.close()

    # Save CSV/JSON logs and plot learning curves
    pd.DataFrame(best_ae_hist).to_csv(W4_RUNS_DIR / "ae_training_history.csv", index=False)
    pd.DataFrame(z_dim_results).to_csv(W4_RUNS_DIR / "z_dim_ablation.csv", index=False)
    beta_df.to_csv(W4_RUNS_DIR / "vae_beta_ablation.csv", index=False)

    plot_w4_learning_curves(
        ae_history=best_ae_hist,
        vae_histories=vae_histories_by_beta,
        z_dim_results=z_dim_results,
        save_path=W4_RUNS_DIR / "w4_learning_curves.png",
    )

    print(f"\nWorkshop 4 training finished in {time.time() - t_start:.1f}s.")
    return {
        "z_dim_results": z_dim_results,
        "beta_summary": beta_summary_rows,
        "best_beta": best_beta,
    }


if __name__ == "__main__":
    run_all_workshop4_training()
