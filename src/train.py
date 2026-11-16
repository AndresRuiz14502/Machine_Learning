"""Training pipeline for HybridNeuMF with Transfer Learning, Fine-Tuning, and TensorBoard logging."""

from __future__ import annotations

import copy
import json
import math
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


if __name__ == "__main__":
    train_hybrid_model()
