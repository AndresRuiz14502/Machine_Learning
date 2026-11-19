"""Training pipelines for Workshop 2 (HybridNeuMF) and Workshop 3 (Dueling Double DQN)."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from src.agent import DQNAgent
    from src.env import MovieLensRecEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.data_loader import LIKE_THRESHOLD, SEED, prepare_datasets, set_seed
from src.features import build_genre_features, compute_train_stats
from src.model import BiasedMatrixFactorization, HybridNeuMF, count_parameters

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "model_best.pt"
RUNS_DIR = PROJECT_ROOT / "runs"

# Default training hyperparameters
GMF_DIM = 64
MLP_EMB_DIM = 32
GENRE_PROJ_DIM = 16
MLP_HIDDEN_DIMS = (128, 64, 32)
DROPOUT_RATE = 0.30

BATCH_SIZE = 8_192
EVAL_BATCH_SIZE = 32_768

PRETRAIN_EPOCHS = 5
PRETRAIN_EMB_LR = 0.0032
PRETRAIN_BIAS_LR = 0.0011
PRETRAIN_REG = 0.048

PHASE1_EPOCHS = 5
PHASE1_LR = 3e-4
PHASE1_WD = 1e-3

PHASE2_EPOCHS = 10
PHASE2_EMB_LR = 0.0012
PHASE2_BIAS_LR = 0.0005
PHASE2_DENSE_LR = 1e-4
PHASE2_REG = 0.055


def set_all_seeds(seed: int = SEED) -> None:
    """Fix Python, NumPy, and PyTorch seeds for full reproducibility."""
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_training_side_features(
    train_df: pd.DataFrame,
    movies: pd.DataFrame,
    item2idx: dict,
    n_users: int,
    n_items: int,
) -> dict:
    """Compute leak-free non-target side features strictly from the training split."""
    mu, user_stats, item_stats, _, _ = compute_train_stats(train_df)
    _, genre_ohe, item_year = build_genre_features(movies, item2idx)

    u_count = user_stats["u_count"].reindex(range(n_users), fill_value=0).to_numpy(dtype=np.float32)
    i_count = item_stats["i_count"].reindex(range(n_items), fill_value=0).to_numpy(dtype=np.float32)

    u_log_cnt = np.log1p(u_count)
    u_log_norm = (u_log_cnt - u_log_cnt.mean()) / (u_log_cnt.std() + 1e-6)
    u_freq_tier = np.tanh(u_log_norm * 0.5).astype(np.float32)

    i_log_cnt = np.log1p(i_count)
    i_log_norm = (i_log_cnt - i_log_cnt.mean()) / (i_log_cnt.std() + 1e-6)
    i_pop_tier = np.tanh(i_log_norm * 0.5).astype(np.float32)

    year_series = item_year.reindex(range(n_items))
    train_year_median = float(year_series.median())
    year_filled = year_series.fillna(train_year_median).to_numpy(dtype=np.float32)
    year_norm = (year_filled - year_filled.mean()) / (year_filled.std() + 1e-6)

    user_priors = np.stack([u_log_norm, u_freq_tier], axis=1).astype(np.float32)
    item_priors = np.stack([i_log_norm, i_pop_tier, year_norm], axis=1).astype(np.float32)
    genre_mat = genre_ohe.reindex(range(n_items), fill_value=0).to_numpy(dtype=np.float32)

    return {
        "mu": float(mu),
        "user_priors": torch.tensor(user_priors, dtype=torch.float32),
        "item_priors": torch.tensor(item_priors, dtype=torch.float32),
        "item_genres": torch.tensor(genre_mat, dtype=torch.float32),
        "n_genres": int(genre_mat.shape[1]),
        "train_year_median": train_year_median,
    }


@torch.no_grad()
def predict_batches(
    model: nn.Module,
    u_t: torch.Tensor,
    i_t: torch.Tensor,
    device: torch.device,
    batch_size: int = EVAL_BATCH_SIZE,
    clip: bool = True,
) -> np.ndarray:
    """Run batched inference and return numpy predictions."""
    model.eval()
    preds = []
    n = len(u_t)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bu = u_t[start:end].to(device)
        bi = i_t[start:end].to(device)
        out = model(bu, bi)
        if clip:
            out = out.clamp(0.5, 5.0)
        preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    like_threshold: float = LIKE_THRESHOLD,
    decision_threshold: float = 3.55,
) -> dict[str, float]:
    """Compute regression and binary classification metrics."""
    y_pred_clip = np.clip(y_pred, 0.5, 5.0)
    mse_val = float(mean_squared_error(y_true, y_pred_clip))
    rmse_val = float(np.sqrt(mse_val))
    mae_val = float(mean_absolute_error(y_true, y_pred_clip))
    r2_val = float(r2_score(y_true, y_pred_clip))

    y_bin = (y_true >= like_threshold).astype(np.int32)
    y_hat_bin = (y_pred_clip >= decision_threshold).astype(np.int32)
    y_prob = 1.0 / (1.0 + np.exp(-2.5 * (y_pred_clip - decision_threshold)))

    return {
        "RMSE": rmse_val,
        "MSE": mse_val,
        "MAE": mae_val,
        "R2": r2_val,
        "accuracy": float(accuracy_score(y_bin, y_hat_bin)),
        "precision": float(precision_score(y_bin, y_hat_bin, zero_division=0)),
        "recall": float(recall_score(y_bin, y_hat_bin, zero_division=0)),
        "f1": float(f1_score(y_bin, y_hat_bin, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_bin, y_prob)),
    }


def pretrain_biased_mf(
    u_tr: torch.Tensor,
    i_tr: torch.Tensor,
    r_tr: torch.Tensor,
    u_va: torch.Tensor,
    i_va: torch.Tensor,
    r_va_np: np.ndarray,
    side_feats: dict,
    n_users: int,
    n_items: int,
    device: torch.device,
    embedding_dim: int = GMF_DIM,
    n_epochs: int = PRETRAIN_EPOCHS,
    emb_lr: float = PRETRAIN_EMB_LR,
    bias_lr: float = PRETRAIN_BIAS_LR,
    reg: float = PRETRAIN_REG,
) -> tuple[BiasedMatrixFactorization, float]:
    """Pretrain BiasedMatrixFactorization with Nesterov SGD and per-observation L2 regularization."""
    mf = BiasedMatrixFactorization(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
        global_mean=side_feats["mu"],
        init_std=0.10,
    ).to(device)

    optimizer = torch.optim.SGD(
        [
            {"params": [mf.user_factors.weight, mf.item_factors.weight], "lr": emb_lr},
            {"params": [mf.user_bias.weight, mf.item_bias.weight], "lr": bias_lr},
        ],
        momentum=0.90,
        nesterov=True,
    )

    best_val_rmse = float("inf")
    best_state = copy.deepcopy(mf.state_dict())
    n_train = len(u_tr)

    print("\n=== Stage 0: Pretraining BiasedMatrixFactorization (GMF Base) ===")
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        mf.train()
        perm = torch.randperm(n_train)
        sq_err_sum = 0.0

        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            bu = u_tr[idx].to(device)
            bi = i_tr[idx].to(device)
            br = r_tr[idx].to(device)

            optimizer.zero_grad()
            pred = mf(bu, bi)
            sq_err = (pred - br).pow(2)
            err_term = 0.5 * sq_err.sum()

            u_e = mf.user_factors(bu)
            i_e = mf.item_factors(bi)
            u_b = mf.user_bias(bu)
            i_b = mf.item_bias(bi)
            reg_term = (
                0.5
                * reg
                * (u_e.pow(2).sum() + i_e.pow(2).sum() + u_b.pow(2).sum() + i_b.pow(2).sum())
            )

            loss = err_term + reg_term
            loss.backward()
            optimizer.step()

            sq_err_sum += float(sq_err.detach().sum().item())

        train_rmse = float(np.sqrt(sq_err_sum / n_train))
        val_preds = predict_batches(mf, u_va, i_va, device=device)
        val_m = compute_metrics(r_va_np, val_preds)

        improved = val_m["RMSE"] < best_val_rmse
        if improved:
            best_val_rmse = val_m["RMSE"]
            best_state = copy.deepcopy(mf.state_dict())

        print(
            f"Pretrain Epoch {epoch:02d}/{n_epochs:02d} | "
            f"Train RMSE: {train_rmse:.4f} | "
            f"Val RMSE: {val_m['RMSE']:.4f} | Val MAE: {val_m['MAE']:.4f} | "
            f"Val AUC: {val_m['roc_auc']:.4f} ({time.time() - t0:.1f}s)"
            + (" *" if improved else "")
        )

    mf.load_state_dict(best_state)
    print(f"Best Pretrained GMF Val RMSE: {best_val_rmse:.4f}")
    return mf, best_val_rmse


def plot_learning_curves(history_df: pd.DataFrame, phase1_epochs: int, save_path: Path) -> None:
    """Generate and save training/validation learning curve plots."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ep = history_df["epoch"]

    axes[0].plot(ep, history_df["train_rmse"], marker="o", color="#4C72B0", label="Train RMSE")
    axes[0].plot(ep, history_df["val_rmse"], marker="s", color="#C44E52", label="Validation RMSE")
    axes[0].plot(
        ep, history_df["val_mae"], marker="^", color="#55A868", linestyle="--", label="Validation MAE"
    )
    axes[0].axvline(
        phase1_epochs + 0.5, color="#DD8452", linestyle=":", linewidth=1.5, label="Unfreeze (Phase 2)"
    )
    axes[0].set_title("HybridNeuMF — Error Metrics across Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Error (Stars)")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history_df["val_roc_auc"], marker="o", color="#8172B3", label="Val ROC-AUC")
    axes[1].plot(ep, history_df["val_f1"], marker="s", color="#64B5CD", label="Val F1-Score")
    axes[1].plot(
        ep,
        history_df["val_accuracy"],
        marker="d",
        color="#937860",
        linestyle="--",
        label="Val Accuracy",
    )
    axes[1].axvline(
        phase1_epochs + 0.5, color="#DD8452", linestyle=":", linewidth=1.5, label="Unfreeze (Phase 2)"
    )
    axes[1].set_title("Validation Classification Metrics (like >= 4.0)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    axes[2].plot(ep, history_df["lr"], marker="o", color="#2F4F4F", label="Learning Rate")
    axes[2].axvline(
        phase1_epochs + 0.5, color="#DD8452", linestyle=":", linewidth=1.5, label="Phase 1 -> Phase 2"
    )
    axes[2].set_title("Learning Rate Schedule (Exp -> Cosine Annealing)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_yscale("log")
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Learning curves saved to: {save_path}")


def train_hybrid_model(
    save_checkpoint_path: Path = CHECKPOINT_PATH,
    runs_dir: Path = RUNS_DIR,
) -> dict:
    """Execute end-to-end data loading, pretraining, two-phase fine-tuning, and checkpointing."""
    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    data = prepare_datasets(save_splits_file=False, verbose=True)
    train_df, val_df, test_df = data["train_df"], data["val_df"], data["test_df"]
    n_users, n_items = data["n_users"], data["n_items"]

    side_feats = build_training_side_features(
        train_df=train_df,
        movies=data["movies"],
        item2idx=data["item2idx"],
        n_users=n_users,
        n_items=n_items,
    )

    u_tr = torch.tensor(train_df["u"].to_numpy(), dtype=torch.long)
    i_tr = torch.tensor(train_df["i"].to_numpy(), dtype=torch.long)
    r_tr = torch.tensor(train_df["rating"].to_numpy(), dtype=torch.float32)

    u_va = torch.tensor(val_df["u"].to_numpy(), dtype=torch.long)
    i_va = torch.tensor(val_df["i"].to_numpy(), dtype=torch.long)
    r_va_np = val_df["rating"].to_numpy(dtype=np.float32)

    u_te = torch.tensor(test_df["u"].to_numpy(), dtype=torch.long)
    i_te = torch.tensor(test_df["i"].to_numpy(), dtype=torch.long)
    r_te_np = test_df["rating"].to_numpy(dtype=np.float32)

    pretrained_mf, pretrain_val_rmse = pretrain_biased_mf(
        u_tr=u_tr,
        i_tr=i_tr,
        r_tr=r_tr,
        u_va=u_va,
        i_va=i_va,
        r_va_np=r_va_np,
        side_feats=side_feats,
        n_users=n_users,
        n_items=n_items,
        device=device,
    )

    model = HybridNeuMF(
        n_users=n_users,
        n_items=n_items,
        n_genres=side_feats["n_genres"],
        n_user_Prior_feats=2,
        n_item_prior_feats=3,
        gmf_dim=GMF_DIM,
        mlp_emb_dim=MLP_EMB_DIM,
        genre_proj_dim=GENRE_PROJ_DIM,
        mlp_hidden_dims=MLP_HIDDEN_DIMS,
        dropout_rate=DROPOUT_RATE,
        global_mean=side_feats["mu"],
    ).to(device)

    model.set_side_features(
        item_genres=side_feats["item_genres"],
        user_priors=side_feats["user_priors"],
        item_priors=side_feats["item_priors"],
    )
    model.load_pretrained_gmf(pretrained_mf)

    runs_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(runs_dir / "hybrid_neumf_experiment"))

    history_rows: list[dict] = []
    best_val_rmse = float("inf")
    best_epoch = 0
    best_phase = "Phase 1"
    best_state = copy.deepcopy(model.state_dict())
    n_train = len(u_tr)

    # Phase 1: Freeze pretrained embeddings, train ONLY task-specific head (genre_proj + mlp_tower + fusion_head)
    model.set_pretrained_frozen(freeze=True)
    p_info = count_parameters(model)
    print(
        f"\n=== Phase 1 (Frozen Pretrained Backbone - Task Head Only) | Epochs: {PHASE1_EPOCHS} | "
        f"LR: {PHASE1_LR:.1e} | Trainable Params: {p_info['trainable']:,} (Frozen: {p_info['frozen']:,}) ==="
    )

    p1_params = [p for p in model.parameters() if p.requires_grad]
    opt_p1 = torch.optim.AdamW(p1_params, lr=PHASE1_LR, weight_decay=PHASE1_WD)
    sched_p1 = torch.optim.lr_scheduler.ExponentialLR(opt_p1, gamma=0.85)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        t0 = time.time()
        model.train()
        current_lr = float(opt_p1.param_groups[0]["lr"])
        perm = torch.randperm(n_train)
        sq_err_sum = 0.0

        for start in range(0, n_train, 16_384):
            idx = perm[start : start + 16_384]
            bu = u_tr[idx].to(device)
            bi = i_tr[idx].to(device)
            br = r_tr[idx].to(device)

            opt_p1.zero_grad()
            pred = model(bu, bi)
            sq_err = (pred - br).pow(2)
            loss = sq_err.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(p1_params, max_norm=1.0)
            opt_p1.step()

            sq_err_sum += float(sq_err.detach().sum().item())

        sched_p1.step()
        train_mse = sq_err_sum / n_train
        train_rmse = float(np.sqrt(train_mse))

        val_preds = predict_batches(model, u_va, i_va, device=device)
        val_m = compute_metrics(r_va_np, val_preds)

        improved = val_m["RMSE"] < best_val_rmse
        if improved:
            best_val_rmse = val_m["RMSE"]
            best_epoch = epoch
            best_phase = "Phase 1 (Frozen Pretrained Backbone)"
            best_state = copy.deepcopy(model.state_dict())

        history_rows.append(
            {
                "epoch": epoch,
                "phase": "Phase 1 (Frozen Pretrained Backbone)",
                "lr": current_lr,
                "train_loss": train_mse,
                "train_mse": train_mse,
                "train_rmse": train_rmse,
                "val_mse": val_m["MSE"],
                "val_rmse": val_m["RMSE"],
                "val_mae": val_m["MAE"],
                "val_r2": val_m["R2"],
                "val_accuracy": val_m["accuracy"],
                "val_precision": val_m["precision"],
                "val_recall": val_m["recall"],
                "val_f1": val_m["f1"],
                "val_roc_auc": val_m["roc_auc"],
            }
        )
        tb_writer.add_scalar("Loss/train", train_mse, epoch)
        tb_writer.add_scalar("RMSE/train", train_rmse, epoch)
        tb_writer.add_scalar("RMSE/val", val_m["RMSE"], epoch)
        tb_writer.add_scalar("MAE/val", val_m["MAE"], epoch)
        tb_writer.add_scalar("Classification/val_f1", val_m["f1"], epoch)
        tb_writer.add_scalar("Classification/val_roc_auc", val_m["roc_auc"], epoch)
        tb_writer.add_scalar("Optimization/learning_rate", current_lr, epoch)

        print(
            f"[Phase 1 (Frozen Backbone)] Epoch {epoch:02d} | LR: {current_lr:.2e} | "
            f"Train RMSE: {train_rmse:.4f} | Val RMSE: {val_m['RMSE']:.4f} | "
            f"Val MAE: {val_m['MAE']:.4f} | Val F1: {val_m['f1']:.4f} | "
            f"Val AUC: {val_m['roc_auc']:.4f} ({time.time() - t0:.1f}s)"
            + (" [Saved Best]" if improved else "")
        )

    model.load_state_dict(best_state)

    # Phase 2: Unfreeze all layers and fine-tune end-to-end with Nesterov SGD + Cosine Annealing LR
    model.set_pretrained_frozen(freeze=False)
    p_info = count_parameters(model)
    print(
        f"\n=== Phase 2 (Full Fine-Tuning) | Epochs: {PHASE2_EPOCHS} | LR: {PHASE2_EMB_LR:.1e} | "
        f"Trainable Params: {p_info['trainable']:,} (Frozen: {p_info['frozen']:,}) ==="
    )

    p2_gmf_emb = [model.user_gmf_factors.weight, model.item_gmf_factors.weight]
    p2_mlp_emb = [model.user_mlp_factors.weight, model.item_mlp_factors.weight]
    p2_bias = [model.user_bias.weight, model.item_bias.weight]
    p2_dense = [
        p
        for n, p in model.named_parameters()
        if "factors" not in n and "bias.weight" not in n
    ]

    opt_p2 = torch.optim.SGD(
        [
            {"params": p2_gmf_emb, "lr": PHASE2_EMB_LR},
            {"params": p2_mlp_emb, "lr": PHASE2_EMB_LR * 0.15},
            {"params": p2_bias, "lr": PHASE2_BIAS_LR},
            {"params": p2_dense, "lr": PHASE2_DENSE_LR / BATCH_SIZE, "weight_decay": 1e-4},
        ],
        momentum=0.85,
        nesterov=True,
    )
    sched_p2 = torch.optim.lr_scheduler.LambdaLR(
        opt_p2,
        lr_lambda=lambda ep_idx: 0.10
        + 0.90 * 0.5 * (1.0 + math.cos(math.pi * ep_idx / max(PHASE2_EPOCHS, 1))),
    )

    for ep_offset in range(PHASE2_EPOCHS):
        epoch = PHASE1_EPOCHS + 1 + ep_offset
        t0 = time.time()
        model.train()
        current_lr = float(opt_p2.param_groups[0]["lr"])
        perm = torch.randperm(n_train)
        sq_err_sum = 0.0

        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            bu = u_tr[idx].to(device)
            bi = i_tr[idx].to(device)
            br = r_tr[idx].to(device)

            opt_p2.zero_grad()
            pred = model(bu, bi)
            sq_err = (pred - br).pow(2)
            err_term = 0.5 * sq_err.sum()

            u_ge = model.user_gmf_factors(bu)
            i_ge = model.item_gmf_factors(bi)
            u_me = model.user_mlp_factors(bu)
            i_me = model.item_mlp_factors(bi)
            u_b = model.user_bias(bu)
            i_b = model.item_bias(bi)
            reg_term = (
                0.5
                * PHASE2_REG
                * (
                    u_ge.pow(2).sum()
                    + i_ge.pow(2).sum()
                    + 2.0 * (u_me.pow(2).sum() + i_me.pow(2).sum())
                    + u_b.pow(2).sum()
                    + i_b.pow(2).sum()
                )
            )

            loss = err_term + reg_term
            loss.backward()
            torch.nn.utils.clip_grad_norm_(p2_dense, max_norm=1.0)
            opt_p2.step()

            sq_err_sum += float(sq_err.detach().sum().item())

        sched_p2.step()
        train_mse = sq_err_sum / n_train
        train_rmse = float(np.sqrt(train_mse))

        val_preds = predict_batches(model, u_va, i_va, device=device)
        val_m = compute_metrics(r_va_np, val_preds)

        improved = val_m["RMSE"] < best_val_rmse
        if improved:
            best_val_rmse = val_m["RMSE"]
            best_epoch = epoch
            best_phase = "Phase 2 (Full Fine-Tuning)"
            best_state = copy.deepcopy(model.state_dict())

        history_rows.append(
            {
                "epoch": epoch,
                "phase": "Phase 2 (Full Fine-Tuning)",
                "lr": current_lr,
                "train_loss": train_mse,
                "train_mse": train_mse,
                "train_rmse": train_rmse,
                "val_mse": val_m["MSE"],
                "val_rmse": val_m["RMSE"],
                "val_mae": val_m["MAE"],
                "val_r2": val_m["R2"],
                "val_accuracy": val_m["accuracy"],
                "val_precision": val_m["precision"],
                "val_recall": val_m["recall"],
                "val_f1": val_m["f1"],
                "val_roc_auc": val_m["roc_auc"],
            }
        )
        tb_writer.add_scalar("Loss/train", train_mse, epoch)
        tb_writer.add_scalar("RMSE/train", train_rmse, epoch)
        tb_writer.add_scalar("RMSE/val", val_m["RMSE"], epoch)
        tb_writer.add_scalar("MAE/val", val_m["MAE"], epoch)
        tb_writer.add_scalar("Classification/val_f1", val_m["f1"], epoch)
        tb_writer.add_scalar("Classification/val_roc_auc", val_m["roc_auc"], epoch)
        tb_writer.add_scalar("Optimization/learning_rate", current_lr, epoch)

        print(
            f"[Phase 2 (Full Fine-Tuning)] Epoch {epoch:02d} | LR: {current_lr:.2e} | "
            f"Train RMSE: {train_rmse:.4f} | Val RMSE: {val_m['RMSE']:.4f} | "
            f"Val MAE: {val_m['MAE']:.4f} | Val F1: {val_m['f1']:.4f} | "
            f"Val AUC: {val_m['roc_auc']:.4f} ({time.time() - t0:.1f}s)"
            + (" [Saved Best]" if improved else "")
        )

    tb_writer.flush()
    tb_writer.close()

    model.load_state_dict(best_state)

    # Calibrate global level shift on validation set to account for chronological split drift
    val_raw_preds = predict_batches(model, u_va, i_va, device=device, clip=False)
    optimal_shift = float(np.mean(r_va_np - val_raw_preds))
    with torch.no_grad():
        model.level_shift.add_(optimal_shift)

    val_cal_preds = predict_batches(model, u_va, i_va, device=device, clip=True)
    val_cal_m = compute_metrics(r_va_np, val_cal_preds)
    best_val_rmse = val_cal_m["RMSE"]
    best_state = copy.deepcopy(model.state_dict())

    save_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": best_epoch,
            "phase": best_phase,
            "best_val_rmse": best_val_rmse,
            "val_metrics": val_cal_m,
            "config": model.get_config(),
            "model_state_dict": best_state,
            "seed": SEED,
            "like_threshold": LIKE_THRESHOLD,
        },
        save_checkpoint_path,
    )

    history_df = pd.DataFrame(history_rows)
    history_csv = runs_dir / "training_history.csv"
    history_json = runs_dir / "training_history.json"
    history_df.to_csv(history_csv, index=False)
    with open(history_json, "w", encoding="utf-8") as f:
        json.dump(history_rows, f, indent=2)

    plot_learning_curves(
        history_df=history_df,
        phase1_epochs=PHASE1_EPOCHS,
        save_path=runs_dir / "learning_curves.png",
    )

    test_preds = predict_batches(model, u_te, i_te, device=device)
    test_m = compute_metrics(r_te_np, test_preds)
    print(f"\n=== Best Checkpoint Summary (Epoch {best_epoch} - {best_phase}) ===")
    print(f"Checkpoint Path : {save_checkpoint_path}")
    print(f"Best Val RMSE   : {best_val_rmse:.4f}")
    print(
        f"Test Set Metrics: RMSE={test_m['RMSE']:.4f} | MSE={test_m['MSE']:.4f} | "
        f"MAE={test_m['MAE']:.4f} | R2={test_m['R2']:.4f} | "
        f"F1={test_m['f1']:.4f} | AUC={test_m['roc_auc']:.4f}"
    )

    return {
        "model": model,
        "history_df": history_df,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "test_metrics": test_m,
        "pretrain_val_rmse": pretrain_val_rmse,
    }


# --------------------------------------------------------------------------------------
# Workshop 3 — Dueling Double DQN (D3QN) Training Pipeline
#
# Every configuration is trained with the same standard Double DQN update
# (`DQNAgent.update`: replay buffer, Polyak-averaged target network, Huber TD loss on the
# taken action, valid-action masking in the bootstrap, gradient-norm clipping at 1.0),
# one gradient step per environment step, and epsilon-greedy exploration decayed per
# episode. Validation runs the greedy policy (no exploration) once on every RL validation
# session.
#
# Pipeline:
# 1. Eight one-at-a-time hyperparameter configurations (learning rate, discount factor,
#    epsilon decay, L2 weight decay), 800 episodes each, logged to TensorBoard under
#    `runs/workshop3/<config_id>/`.
# 2. The configuration with the highest smoothed validation return (mean of its last five
#    validation evaluations) is retrained for 1,500 episodes as the champion (`d3qn_tuned`);
#    its best validation checkpoint is saved to `checkpoints/agent_best.pt`.
# 3. A myopic ablation with the champion hyperparameters but gamma = 0 (a neural contextual
#    bandit with the same network, data, and budget) is trained the same way and saved to
#    `checkpoints/bandit_gamma0.pt`.
#
# All curves and the hyperparameter table are exported to `runs/workshop3/training_logs.json`.
# --------------------------------------------------------------------------------------

CHAMPION_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "agent_best.pt"
BANDIT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "bandit_gamma0.pt"
TRAINING_LOGS = PROJECT_ROOT / "runs" / "workshop3" / "training_logs.json"
SMOOTHING_WINDOW = 5
DEFAULT_HIDDEN_DIM = {"flat": 256, "slot": 128}


def repo_relative_posix(path: Path) -> str:
    """Return ``path`` relative to the repository root as a POSIX string (portable JSON logs)."""
    return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()


BASE_CONFIGS: List[Dict[str, Any]] = [
    {
        "config_id": "cfg1_baseline",
        "label": "Cfg 1: Baseline (lr=5e-4, γ=0.95, ε_dec=0.995, L2=1e-5)",
        "varied_param": "Baseline",
        "lr": 5e-4,
        "gamma": 0.95,
        "epsilon_decay": 0.995,
        "weight_decay": 1e-5,
    },
    {
        "config_id": "cfg2_high_lr",
        "label": "Cfg 2: High LR (lr=1e-3, γ=0.95, ε_dec=0.995, L2=1e-5)",
        "varied_param": "lr=1e-3",
        "lr": 1e-3,
        "gamma": 0.95,
        "epsilon_decay": 0.995,
        "weight_decay": 1e-5,
    },
    {
        "config_id": "cfg3_low_lr",
        "label": "Cfg 3: Low LR (lr=1e-4, γ=0.95, ε_dec=0.995, L2=1e-5)",
        "varied_param": "lr=1e-4",
        "lr": 1e-4,
        "gamma": 0.95,
        "epsilon_decay": 0.995,
        "weight_decay": 1e-5,
    },
    {
        "config_id": "cfg4_gamma_090",
        "label": "Cfg 4: Low Discount (lr=5e-4, γ=0.90, ε_dec=0.995, L2=1e-5)",
        "varied_param": "gamma=0.90",
        "lr": 5e-4,
        "gamma": 0.90,
        "epsilon_decay": 0.995,
        "weight_decay": 1e-5,
    },
    {
        "config_id": "cfg5_gamma_099",
        "label": "Cfg 5: High Discount (lr=5e-4, γ=0.99, ε_dec=0.995, L2=1e-5)",
        "varied_param": "gamma=0.99",
        "lr": 5e-4,
        "gamma": 0.99,
        "epsilon_decay": 0.995,
        "weight_decay": 1e-5,
    },
    {
        "config_id": "cfg6_fast_eps",
        "label": "Cfg 6: Fast Eps Decay (lr=5e-4, γ=0.95, ε_dec=0.990, L2=1e-5)",
        "varied_param": "eps_decay=0.990",
        "lr": 5e-4,
        "gamma": 0.95,
        "epsilon_decay": 0.990,
        "weight_decay": 1e-5,
    },
    {
        "config_id": "cfg7_slow_eps",
        "label": "Cfg 7: Slow Eps Decay (lr=5e-4, γ=0.95, ε_dec=0.998, L2=1e-5)",
        "varied_param": "eps_decay=0.998",
        "lr": 5e-4,
        "gamma": 0.95,
        "epsilon_decay": 0.998,
        "weight_decay": 1e-5,
    },
    {
        "config_id": "cfg8_high_l2",
        "label": "Cfg 8: High L2 Reg (lr=5e-4, γ=0.95, ε_dec=0.995, L2=1e-4)",
        "varied_param": "weight_decay=1e-4",
        "lr": 5e-4,
        "gamma": 0.95,
        "epsilon_decay": 0.995,
        "weight_decay": 1e-4,
    },
]

# Q-network architectures searched jointly with the hyperparameters (see agent.py):
# "flat" = dueling MLP over the whole 250-D state with 50 output units;
# "slot" = slot-equivariant dueling network that scores every slot with one shared MLP.
ARCHITECTURES = ("flat", "slot")
ARCHITECTURE_LABELS = {"flat": "Flat MLP", "slot": "Slot-shared"}

HYPERPARAMETER_CONFIGS: List[Dict[str, Any]] = [
    {
        **cfg,
        "config_id": f"{arch}_{cfg['config_id']}",
        "label": f"{ARCHITECTURE_LABELS[arch]} | {cfg['label']}",
        "architecture": arch,
    }
    for arch in ARCHITECTURES
    for cfg in BASE_CONFIGS
]


def validate(val_env: MovieLensRecEnv, agent: DQNAgent, seed: int) -> Dict[str, float]:
    """Greedy (epsilon = 0) evaluation on every RL validation session exactly once."""
    from src.baselines import evaluate_policy

    return evaluate_policy(val_env, agent, seed=seed, user_indices=val_env.split_user_indices)


def train_single_config(
    config: Dict[str, Any],
    num_episodes: int = 800,
    eval_every: int = 40,
    checkpoint_path: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train a single D3QN hyperparameter configuration and log to TensorBoard."""
    from src.agent import DQNAgent
    from src.env import MovieLensRecEnv

    config_id = config["config_id"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tb_dir = PROJECT_ROOT / "runs" / "workshop3" / config_id
    tb_dir.mkdir(parents=True, exist_ok=True)
    for old_event in tb_dir.glob("events.out.tfevents.*"):
        old_event.unlink()
    writer = SummaryWriter(log_dir=str(tb_dir))

    train_env = MovieLensRecEnv(split="train", horizon=10, seed=seed)
    val_env = MovieLensRecEnv(split="val", horizon=10, seed=seed)

    architecture = str(config.get("architecture", "flat"))
    hidden_dim = int(config.get("hidden_dim", DEFAULT_HIDDEN_DIM[architecture]))
    agent = DQNAgent(
        state_dim=250,
        action_dim=50,
        hidden_dim=hidden_dim,
        architecture=architecture,
        lr=float(config["lr"]),
        gamma=float(config["gamma"]),
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=float(config["epsilon_decay"]),
        weight_decay=float(config["weight_decay"]),
        batch_size=128,
        buffer_capacity=50000,
        tau=0.05,
        grad_clip=1.0,
        seed=seed,
        device=device,
    )

    train_returns: List[float] = []
    train_lengths: List[int] = []
    train_hit_rates: List[float] = []
    epsilons: List[float] = []
    td_losses: List[float] = []
    mean_qs: List[float] = []
    grad_norms: List[float] = []
    action_entropies: List[float] = []
    unique_actions_curve: List[int] = []

    val_checkpoints: List[Dict[str, Any]] = []
    best_val_return = -1e9
    best_val_metrics: Dict[str, float] = {}
    best_val_episode = 0

    recent_actions: List[int] = []
    t_start = time.time()

    # Episode-0 validation of the untrained network
    init_val = validate(val_env, agent, seed)
    val_checkpoints.append({"episode": 0, "steps": 0, **{k: round(float(v), 4) for k, v in init_val.items()}})

    for ep in range(1, num_episodes + 1):
        obs, info = train_env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        ep_hits = 0
        ep_steps = 0
        ep_loss_list: List[float] = []
        ep_q_list: List[float] = []
        ep_gn_list: List[float] = []

        while not done:
            action = agent.select_action(obs, valid_mask=info["valid_mask"], eval_mode=False)
            next_obs, reward, terminated, truncated, next_info = train_env.step(action)
            done = bool(terminated or truncated)
            agent.store_transition(obs, action, reward, next_obs, done, next_info["valid_mask"])

            # One gradient step per environment step once the buffer holds a full batch
            upd_stats = agent.update()
            if upd_stats is not None:
                ep_loss_list.append(upd_stats["loss"])
                ep_q_list.append(upd_stats["mean_q"])
                ep_gn_list.append(upd_stats["grad_norm"])

            ep_ret += reward
            ep_hits += int(next_info["is_hit"])
            ep_steps += 1
            recent_actions.append(int(action))
            if len(recent_actions) > 400:
                recent_actions.pop(0)
            obs, info = next_obs, next_info

        cur_eps = agent.decay_epsilon()

        mean_loss = float(np.mean(ep_loss_list)) if ep_loss_list else 0.0
        mean_q = float(np.mean(ep_q_list)) if ep_q_list else 0.0
        mean_gn = float(np.mean(ep_gn_list)) if ep_gn_list else 0.0
        ep_hr = float(ep_hits) / float(max(1, ep_steps))

        # Action distribution entropy & unique action count over sliding window
        counts = np.bincount(recent_actions, minlength=50).astype(np.float64)
        probs = counts / max(1.0, np.sum(counts))
        nz = probs[probs > 0]
        entropy_bits = float(-np.sum(nz * np.log2(nz)))
        unique_act = int(np.sum(counts > 0))

        train_returns.append(round(float(ep_ret), 4))
        train_lengths.append(int(ep_steps))
        train_hit_rates.append(round(ep_hr, 4))
        epsilons.append(round(float(cur_eps), 4))
        td_losses.append(round(mean_loss, 5))
        mean_qs.append(round(mean_q, 4))
        grad_norms.append(round(mean_gn, 4))
        action_entropies.append(round(entropy_bits, 4))
        unique_actions_curve.append(unique_act)

        writer.add_scalar("train/episode_return", ep_ret, ep)
        writer.add_scalar("train/episode_length", ep_steps, ep)
        writer.add_scalar("train/hit_rate", ep_hr, ep)
        writer.add_scalar("train/epsilon", cur_eps, ep)
        writer.add_scalar("train/td_loss", mean_loss, ep)
        writer.add_scalar("train/mean_q", mean_q, ep)
        writer.add_scalar("train/grad_norm", mean_gn, ep)
        writer.add_scalar("train/action_entropy_bits", entropy_bits, ep)
        writer.add_scalar("train/unique_actions_window", unique_act, ep)

        if ep % eval_every == 0 or ep == num_episodes:
            val_metrics = validate(val_env, agent, seed)
            ckpt_entry = {
                "episode": int(ep),
                "steps": int(np.sum(train_lengths)),
                **{k: round(float(v), 4) for k, v in val_metrics.items()},
            }
            val_checkpoints.append(ckpt_entry)

            writer.add_scalar("val/mean_return", val_metrics["mean_return"], ep)
            writer.add_scalar("val/precision_at_10", val_metrics["precision_at_10"], ep)
            writer.add_scalar("val/recall_at_10", val_metrics["recall_at_10"], ep)
            writer.add_scalar("val/ndcg_at_10", val_metrics["ndcg_at_10"], ep)
            writer.add_scalar("val/map_at_10", val_metrics["map_at_10"], ep)
            writer.add_scalar("val/hr_at_10", val_metrics["hr_at_10"], ep)
            writer.add_scalar("val/catalog_coverage", val_metrics["catalog_coverage"], ep)
            writer.add_scalar("val/mean_novelty_bits", val_metrics["mean_novelty_bits"], ep)

            if val_metrics["mean_return"] >= best_val_return:
                best_val_return = float(val_metrics["mean_return"])
                best_val_metrics = {k: round(float(v), 4) for k, v in val_metrics.items()}
                best_val_episode = int(ep)
                if checkpoint_path is not None:
                    agent.save(checkpoint_path)

    writer.flush()
    writer.close()
    wall_time = round(time.time() - t_start, 2)
    final_val = val_checkpoints[-1]
    smoothed = float(np.mean([c["mean_return"] for c in val_checkpoints[-SMOOTHING_WINDOW:]]))

    print(
        f"[train] {config_id:15s} | Episodes={num_episodes:4d} | Time={wall_time:6.1f}s | "
        f"Final Val Return={final_val['mean_return']:.4f} (Smoothed={smoothed:.4f}, "
        f"Best={best_val_return:.4f} @ ep {best_val_episode}) | "
        f"P@10={final_val['precision_at_10']:.4f} | NDCG@10={final_val['ndcg_at_10']:.4f}",
        flush=True,
    )

    return {
        "config_id": config_id,
        "label": config["label"],
        "varied_param": config["varied_param"],
        "hyperparameters": {
            "lr": config["lr"],
            "gamma": config["gamma"],
            "epsilon_decay": config["epsilon_decay"],
            "weight_decay": config["weight_decay"],
            "batch_size": 128,
            "architecture": architecture,
            "hidden_dim": hidden_dim,
            "tau": 0.05,
            "num_episodes": num_episodes,
            "updates_per_env_step": 1,
        },
        "wall_time_sec": wall_time,
        "final_val_metrics": final_val,
        "smoothed_val_return": round(smoothed, 4),
        "best_val_return": round(best_val_return, 4),
        "best_val_episode": best_val_episode,
        "best_val_metrics": best_val_metrics,
        "val_checkpoints": val_checkpoints,
        "train_curves": {
            "episode_returns": train_returns,
            "episode_lengths": train_lengths,
            "hit_rates": train_hit_rates,
            "epsilons": epsilons,
            "td_losses": td_losses,
            "mean_qs": mean_qs,
            "grad_norms": grad_norms,
            "action_entropies": action_entropies,
            "unique_actions": unique_actions_curve,
        },
    }


def _table_row(run: Dict[str, Any]) -> Dict[str, Any]:
    fm = run["final_val_metrics"]
    hp = run["hyperparameters"]
    return {
        "config_id": run["config_id"],
        "varied_param": run["varied_param"],
        "architecture": hp["architecture"],
        "lr": hp["lr"],
        "gamma": hp["gamma"],
        "epsilon_decay": hp["epsilon_decay"],
        "weight_decay": hp["weight_decay"],
        "episodes": hp["num_episodes"],
        "final_val_return": fm["mean_return"],
        "smoothed_val_return": run["smoothed_val_return"],
        "best_val_return": run["best_val_return"],
        "best_val_episode": run["best_val_episode"],
        "val_precision_at_10": fm["precision_at_10"],
        "val_recall_at_10": fm["recall_at_10"],
        "val_ndcg_at_10": fm["ndcg_at_10"],
        "val_map_at_10": fm["map_at_10"],
        "val_hr_at_10": fm["hr_at_10"],
        "val_catalog_coverage": fm["catalog_coverage"],
        "val_novelty_bits": fm["mean_novelty_bits"],
        "wall_time_sec": run["wall_time_sec"],
    }


def train_rl_agent() -> None:
    t0 = time.time()
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Starting Workshop 3 D3QN Training on device={device_str} ...", flush=True)

    grid_results: List[Dict[str, Any]] = []
    for cfg in HYPERPARAMETER_CONFIGS:
        grid_results.append(train_single_config(config=cfg, num_episodes=800, eval_every=40, seed=42))

    # Select the configuration with the highest smoothed validation return
    best_cfg_res = max(grid_results, key=lambda r: r["smoothed_val_return"])
    best_hp = best_cfg_res["hyperparameters"]
    print(
        f"\n[train] Best 800-episode configuration: {best_cfg_res['config_id']} "
        f"(smoothed val return = {best_cfg_res['smoothed_val_return']:.4f})",
        flush=True,
    )

    tuned_config = {
        "config_id": "d3qn_tuned",
        "label": f"Champion D3QN ({best_cfg_res['config_id']} hyperparameters, 1500 episodes)",
        "varied_param": "Champion (1500 ep)",
        "architecture": best_hp["architecture"],
        "hidden_dim": best_hp["hidden_dim"],
        "lr": best_hp["lr"],
        "gamma": best_hp["gamma"],
        "epsilon_decay": best_hp["epsilon_decay"],
        "weight_decay": best_hp["weight_decay"],
    }
    tuned_res = train_single_config(
        config=tuned_config, num_episodes=1500, eval_every=40, checkpoint_path=CHAMPION_CHECKPOINT, seed=42
    )

    bandit_config = {
        **tuned_config,
        "config_id": "bandit_gamma0",
        "label": "Myopic ablation: champion hyperparameters with γ=0 (neural contextual bandit, 1500 episodes)",
        "varied_param": "gamma=0 (myopic, 1500 ep)",
        "gamma": 0.0,
    }
    bandit_res = train_single_config(
        config=bandit_config, num_episodes=1500, eval_every=40, checkpoint_path=BANDIT_CHECKPOINT, seed=42
    )

    logs_payload = {
        "device": device_str,
        "total_wall_time_sec": round(time.time() - t0, 2),
        "selection_rule": (
            f"highest smoothed validation return (mean of the last {SMOOTHING_WINDOW} greedy "
            f"validation evaluations) among the {len(HYPERPARAMETER_CONFIGS)} grid runs "
            f"({len(BASE_CONFIGS)} hyperparameter configurations x {len(ARCHITECTURES)} architectures)"
        ),
        "best_grid_config_id": best_cfg_res["config_id"],
        "champion_checkpoint": repo_relative_posix(CHAMPION_CHECKPOINT),
        "bandit_checkpoint": repo_relative_posix(BANDIT_CHECKPOINT),
        "hyperparameter_table": [_table_row(r) for r in grid_results + [tuned_res, bandit_res]],
        "grid_runs": {r["config_id"]: r for r in grid_results},
        "champion_run": tuned_res,
        "bandit_run": bandit_res,
    }

    TRAINING_LOGS.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_LOGS, "w", encoding="utf-8") as f:
        json.dump(logs_payload, f, indent=2)

    print(f"[train] Saved complete training logs to {repo_relative_posix(TRAINING_LOGS)}", flush=True)
    print(f"[train] Saved champion checkpoint to {repo_relative_posix(CHAMPION_CHECKPOINT)}", flush=True)
    print(f"[train] Saved gamma=0 ablation checkpoint to {repo_relative_posix(BANDIT_CHECKPOINT)}", flush=True)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train Workshop 2 (HybridNeuMF) and/or Workshop 3 (D3QN).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--workshop", "--mode", dest="mode", choices=("w2", "w3", "all"), default="w3")
    group.add_argument("--w2", dest="mode", action="store_const", const="w2", help="Train Workshop 2 HybridNeuMF.")
    group.add_argument("--w3", dest="mode", action="store_const", const="w3", help="Train Workshop 3 D3QN (default).")
    group.add_argument("--all", dest="mode", action="store_const", const="all", help="Train both Workshop 2 and Workshop 3.")
    args = parser.parse_args(argv)

    if args.mode in ("w2", "all"):
        train_hybrid_model()
    if args.mode in ("w3", "all"):
        train_rl_agent()


if __name__ == "__main__":
    main()
