"""Held-out test set evaluation and comprehensive comparison against Workshop 1 baselines."""

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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.baselines import clip_rating, evaluate_ranking
from src.data_loader import LIKE_THRESHOLD, SEED, TOP_K, prepare_datasets
from src.features import prepare_features
from src.test_model import load_best_checkpoint
from src.train import CHECKPOINT_PATH, RUNS_DIR, compute_metrics, predict_batches

# Exact Workshop 1 benchmark results on the identical temporal split (from src/baselines.py)
WORKSHOP1_REGRESSION_BENCHMARKS = {
    "GlobalMean (W1)": {"RMSE": 1.0212, "MSE": 1.0428, "MAE": 0.7686, "R2": -0.0492},
    "DecisionTree (W1)": {"RMSE": 0.9061, "MSE": 0.8210, "MAE": 0.6741, "R2": 0.1740},
    "Bias (mu+bu+bi) (W1)": {"RMSE": 0.8966, "MSE": 0.8039, "MAE": 0.6680, "R2": 0.1912},
    "GradientBoosting (W1)": {"RMSE": 0.8868, "MSE": 0.7864, "MAE": 0.6614, "R2": 0.2088},
    "RandomForest (W1)": {"RMSE": 0.8778, "MSE": 0.7705, "MAE": 0.6529, "R2": 0.2248},
    "Ridge (W1)": {"RMSE": 0.8730, "MSE": 0.7621, "MAE": 0.6552, "R2": 0.2332},
    "Lasso (W1)": {"RMSE": 0.8727, "MSE": 0.7616, "MAE": 0.6550, "R2": 0.2337},
    "kNN item-item (W1)": {"RMSE": 0.8287, "MSE": 0.6867, "MAE": 0.6144, "R2": 0.3091},
}


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 500,
    seed: int = SEED,
) -> dict[str, tuple[float, float]]:
    """Compute 95% bootstrap confidence intervals for Test RMSE, MAE, F1, and ROC-AUC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_pred_c = clip_rating(y_pred)
    y_bin = (y_true >= LIKE_THRESHOLD).astype(np.int32)
    y_hat_bin = (y_pred_c >= 3.75).astype(np.int32)
    y_prob = 1.0 / (1.0 + np.exp(-2.5 * (y_pred_c - 3.75)))

    rmses, maes, f1s, aucs = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rmses.append(float(np.sqrt(np.mean((y_true[idx] - y_pred_c[idx]) ** 2))))
        maes.append(float(np.mean(np.abs(y_true[idx] - y_pred_c[idx]))))
        f1s.append(float(f1_score(y_bin[idx], y_hat_bin[idx], zero_division=0)))
        aucs.append(float(roc_auc_score(y_bin[idx], y_prob[idx])))

    return {
        "RMSE_95CI": (float(np.percentile(rmses, 2.5)), float(np.percentile(rmses, 97.5))),
        "MAE_95CI": (float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))),
        "F1_95CI": (float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))),
        "AUC_95CI": (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))),
    }


def slice_error_analysis(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
) -> dict[str, pd.DataFrame]:
    """Evaluate HybridNeuMF performance across user activity and item popularity slices."""
    u_counts = train_df.groupby("u").size()
    i_counts = train_df.groupby("i").size()

    eval_df = test_df[["u", "i", "rating", "like"]].copy()
    eval_df["pred"] = clip_rating(y_pred)
    eval_df["u_train_cnt"] = eval_df["u"].map(u_counts).fillna(0)
    eval_df["i_train_cnt"] = eval_df["i"].map(i_counts).fillna(0)

    eval_df["user_activity_tier"] = pd.qcut(
        eval_df["u_train_cnt"],
        q=3,
        labels=["Low Activity Users", "Medium Activity Users", "Power Users"],
        duplicates="drop",
    )
    eval_df["item_popularity_tier"] = pd.qcut(
        eval_df["i_train_cnt"],
        q=3,
        labels=["Long-Tail Movies", "Mid-Popularity Movies", "Blockbuster Movies"],
        duplicates="drop",
    )

    def summarize_slice(group_col: str) -> pd.DataFrame:
        rows = []
        for tier, sub in eval_df.groupby(group_col, observed=False):
            m = compute_metrics(sub["rating"].to_numpy(), sub["pred"].to_numpy())
            rows.append(
                {
                    "Slice": str(tier),
                    "Count": int(len(sub)),
                    "RMSE": round(m["RMSE"], 4),
                    "MAE": round(m["MAE"], 4),
                    "F1": round(m["f1"], 4),
                    "ROC_AUC": round(m["roc_auc"], 4),
                }
            )
        return pd.DataFrame(rows)

    return {
        "user_slices": summarize_slice("user_activity_tier"),
        "item_slices": summarize_slice("item_popularity_tier"),
    }


def plot_evaluation_figures(
    reg_df: pd.DataFrame,
    clf_df: pd.DataFrame,
    y_true: np.ndarray,
    bias_preds: np.ndarray,
    hybrid_preds: np.ndarray,
    model,
    train_df: pd.DataFrame,
    runs_dir: Path,
) -> None:
    """Save comparison bar charts, residual analysis, confusion matrices, and embedding PCA."""
    runs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Regression & Classification Comparison Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    reg_sorted = reg_df.sort_values("RMSE", ascending=False)
    colors_reg = [
        "#C44E52" if "HybridNeuMF" in idx else "#4C72B0" for idx in reg_sorted.index
    ]
    axes[0].barh(reg_sorted.index, reg_sorted["RMSE"], color=colors_reg)
    axes[0].set_xlim(0.70, 1.06)
    axes[0].set_title("Test RMSE Comparison (Lower is Better)")
    axes[0].set_xlabel("Test RMSE (Stars)")
    for i, v in enumerate(reg_sorted["RMSE"]):
        axes[0].text(v + 0.004, i, f"{v:.4f}", va="center", fontsize=8.5)
    axes[0].grid(axis="x", alpha=0.3)

    clf_sorted = clf_df.sort_values("ROC_AUC", ascending=True)
    x_pos = np.arange(len(clf_sorted))
    width = 0.36
    axes[1].barh(x_pos - width / 2, clf_sorted["F1"], height=width, color="#55A868", label="F1-Score")
    axes[1].barh(x_pos + width / 2, clf_sorted["ROC_AUC"], height=width, color="#8172B3", label="ROC-AUC")
    axes[1].set_yticks(x_pos)
    axes[1].set_yticklabels(clf_sorted.index)
    axes[1].set_xlim(0.50, 0.90)
    axes[1].set_title("Binary Preference Classification on Test (Higher is Better)")
    axes[1].set_xlabel("Score")
    axes[1].legend(loc="lower right")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(runs_dir / "workshop2_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 2. Residuals and Confusion Matrix Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    res_bias = y_true - clip_rating(bias_preds)
    res_hybrid = y_true - clip_rating(hybrid_preds)

    axes[0].hist(res_bias, bins=40, alpha=0.55, color="#4C72B0", label=f"W1 Bias (std={res_bias.std():.3f})")
    axes[0].hist(res_hybrid, bins=40, alpha=0.65, color="#C44E52", label=f"W2 HybridNeuMF (std={res_hybrid.std():.3f})")
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1.1)
    axes[0].set_title("Test Residual Distribution (y - y_hat)")
    axes[0].set_xlabel("Residual (Stars)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend(fontsize=8.5)

    y_bin = (y_true >= LIKE_THRESHOLD).astype(np.int32)
    cm_bias = confusion_matrix(y_bin, (clip_rating(bias_preds) >= LIKE_THRESHOLD).astype(np.int32))
    cm_hyb = confusion_matrix(y_bin, (clip_rating(hybrid_preds) >= 3.75).astype(np.int32))

    ConfusionMatrixDisplay(cm_bias, display_labels=["Dislike (0)", "Like (1)"]).plot(
        ax=axes[1], cmap="Blues", colorbar=False, values_format="d"
    )
    axes[1].set_title("W1 Bias Baseline — Test Confusion Matrix")

    ConfusionMatrixDisplay(cm_hyb, display_labels=["Dislike (0)", "Like (1)"]).plot(
        ax=axes[2], cmap="Greens", colorbar=False, values_format="d"
    )
    axes[2].set_title("W2 HybridNeuMF — Test Confusion Matrix")

    plt.tight_layout()
    fig.savefig(runs_dir / "workshop2_diagnostics.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 3. Learned Item Embeddings PCA Colored by Average Rating
    with torch.no_grad():
        item_emb = (
            torch.cat([model.item_gmf_factors.weight, model.item_mlp_factors.weight], dim=-1)
            .cpu()
            .numpy()
        )
    item_stats = train_df.groupby("i")["rating"].agg(i_mean="mean", i_count="size")
    popular_items = item_stats[item_stats["i_count"] >= 30].index.to_numpy()
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(item_emb[popular_items])
    means = item_stats.loc[popular_items, "i_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=means, cmap="viridis", alpha=0.65, s=14)
    plt.colorbar(sc, ax=ax, label="Mean Training Rating (Stars)")
    ax.set_title("PCA of Learned Movie Embeddings Colored by Average Rating")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(runs_dir / "embedding_pca.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def evaluate_on_test_set(
    checkpoint_path: Path = CHECKPOINT_PATH,
    runs_dir: Path = RUNS_DIR,
) -> dict:
    """Evaluate HybridNeuMF on the held-out test set and compare against all Workshop 1 baselines."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_best_checkpoint(checkpoint_path=checkpoint_path, device=device)

    data = prepare_datasets(save_splits_file=False, verbose=True)
    train_df, val_df, test_df = data["train_df"], data["val_df"], data["test_df"]
    feats = prepare_features(train_df, val_df, test_df, data["movies"], data["item2idx"])

    u_te = torch.tensor(test_df["u"].to_numpy(), dtype=torch.long)
    i_te = torch.tensor(test_df["i"].to_numpy(), dtype=torch.long)
    y_te = test_df["rating"].to_numpy(dtype=np.float32)
    y_te_like = test_df["like"].to_numpy(dtype=np.int32)

    # 1. Compute HybridNeuMF predictions and calibrate optimal decision threshold on validation set
    u_va = torch.tensor(val_df["u"].to_numpy(), dtype=torch.long)
    i_va = torch.tensor(val_df["i"].to_numpy(), dtype=torch.long)
    val_preds = predict_batches(model, u_va, i_va, device=device)
    val_like = val_df["like"].to_numpy(dtype=np.int32)

    best_thr, best_val_f1 = 3.75, 0.0
    for thr in np.linspace(3.40, 4.05, 27):
        f1_cand = float(f1_score(val_like, (val_preds >= thr).astype(np.int32), zero_division=0))
        if f1_cand > best_val_f1:
            best_val_f1 = f1_cand
            best_thr = float(thr)

    hybrid_test_preds = predict_batches(model, u_te, i_te, device=device)
    hybrid_metrics = compute_metrics(
        y_te, hybrid_test_preds, like_threshold=LIKE_THRESHOLD, decision_threshold=best_thr
    )
    ci_95 = bootstrap_ci(y_te, hybrid_test_preds)

    # 2. Compute live Bias, Ridge, and LogisticRegression baselines on the exact test split
    mu = feats["mu"]
    bu_map = feats["user_bias"].to_dict()
    bi_map = feats["item_bias"].to_dict()
    bias_test_preds = clip_rating(
        mu
        + test_df["u"].map(bu_map).fillna(0.0).to_numpy()
        + test_df["i"].map(bi_map).fillna(0.0).to_numpy()
    )
    bias_live_m = compute_metrics(
        y_te, bias_test_preds, like_threshold=LIKE_THRESHOLD, decision_threshold=LIKE_THRESHOLD
    )

    Xtr = feats["X_train"][feats["FEATURE_COLS"]].to_numpy(dtype=np.float32)
    ytr_reg = feats["X_train"]["rating"].to_numpy(dtype=np.float32)
    ytr_clf = feats["X_train"]["like"].to_numpy(dtype=np.int32)
    Xte = feats["X_test"][feats["FEATURE_COLS"]].to_numpy(dtype=np.float32)

    ridge = Ridge(alpha=1.0, random_state=SEED)
    ridge.fit(Xtr, ytr_reg)
    ridge_preds = clip_rating(ridge.predict(Xte))
    ridge_m = compute_metrics(
        y_te, ridge_preds, like_threshold=LIKE_THRESHOLD, decision_threshold=LIKE_THRESHOLD
    )

    log_reg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=SEED)),
        ]
    )
    log_reg.fit(Xtr, ytr_clf)
    lr_pred = log_reg.predict(Xte)
    lr_prob = log_reg.predict_proba(Xte)[:, 1]

    # 3. Assemble Regression Comparison Table (Workshop 1 Baselines vs Workshop 2 HybridNeuMF)
    reg_rows = {}
    for name, bm in WORKSHOP1_REGRESSION_BENCHMARKS.items():
        reg_rows[name] = bm.copy()
    reg_rows["HybridNeuMF (W2 Deep CF)"] = {
        "RMSE": hybrid_metrics["RMSE"],
        "MSE": hybrid_metrics["MSE"],
        "MAE": hybrid_metrics["MAE"],
        "R2": hybrid_metrics["R2"],
    }
    reg_df = pd.DataFrame(reg_rows).T.sort_values("RMSE")

    # 4. Assemble Binary Classification Comparison Table (like = rating >= 4.0)
    clf_rows = {
        "GlobalMean (W1)": {
            "Accuracy": float(accuracy_score(y_te_like, np.zeros_like(y_te_like))),
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0,
            "ROC_AUC": 0.5000,
        },
        "Bias Threshold (W1)": {
            "Accuracy": bias_live_m["accuracy"],
            "Precision": bias_live_m["precision"],
            "Recall": bias_live_m["recall"],
            "F1": bias_live_m["f1"],
            "ROC_AUC": bias_live_m["roc_auc"],
        },
        "Ridge Threshold (W1)": {
            "Accuracy": ridge_m["accuracy"],
            "Precision": ridge_m["precision"],
            "Recall": ridge_m["recall"],
            "F1": ridge_m["f1"],
            "ROC_AUC": ridge_m["roc_auc"],
        },
        "LogisticRegression (W1)": {
            "Accuracy": float(accuracy_score(y_te_like, lr_pred)),
            "Precision": float(precision_score(y_te_like, lr_pred, zero_division=0)),
            "Recall": float(recall_score(y_te_like, lr_pred, zero_division=0)),
            "F1": float(f1_score(y_te_like, lr_pred, zero_division=0)),
            "ROC_AUC": float(roc_auc_score(y_te_like, lr_prob)),
        },
        "RandomForestClf (W1)": {
            "Accuracy": 0.7031,
            "Precision": 0.5773,
            "Recall": 0.6914,
            "F1": 0.6293,
            "ROC_AUC": 0.7645,
        },
        "GradientBoostingClf (W1)": {
            "Accuracy": 0.6979,
            "Precision": 0.5697,
            "Recall": 0.6986,
            "F1": 0.6276,
            "ROC_AUC": 0.7518,
        },
        "HybridNeuMF (W2 Deep CF)": {
            "Accuracy": hybrid_metrics["accuracy"],
            "Precision": hybrid_metrics["precision"],
            "Recall": hybrid_metrics["recall"],
            "F1": hybrid_metrics["f1"],
            "ROC_AUC": hybrid_metrics["roc_auc"],
        },
    }
    clf_df = pd.DataFrame(clf_rows).T.sort_values("ROC_AUC", ascending=False)

    # 5. Top-10 Ranking Evaluation (Candidate Ranking + Popularity-Calibrated Propensity Score)
    item_pop = train_df.groupby("i").size()
    pop_vec = np.zeros(data["n_items"], dtype=np.float32)
    pop_vec[item_pop.index.to_numpy()] = item_pop.to_numpy(dtype=np.float32)
    pop_log_norm = np.log1p(pop_vec) / (np.log1p(pop_vec).max() + 1e-6)

    bu_arr = np.zeros(data["n_users"], dtype=np.float32)
    for u_k, v_k in bu_map.items():
        bu_arr[u_k] = v_k
    bi_vec = np.zeros(data["n_items"], dtype=np.float32)
    for i_k, v_k in bi_map.items():
        bi_vec[i_k] = v_k

    @torch.no_grad()
    def hybrid_rank_score(u: int, items: np.ndarray) -> np.ndarray:
        u_t = torch.full((len(items),), int(u), dtype=torch.long, device=device)
        i_t = torch.tensor(items, dtype=torch.long, device=device)
        pred_r = model(u_t, i_t).clamp(0.5, 5.0).cpu().numpy()
        return pred_r + 1.35 * pop_log_norm[items]

    ranking_rows = {
        "Bias (W1)": evaluate_ranking(
            lambda u, items: mu + bu_arr[u] + bi_vec[items],
            train_df,
            test_df,
            data["n_items"],
            k=TOP_K,
        ),
        "Popularity (W1)": evaluate_ranking(
            lambda u, items: pop_vec[items],
            train_df,
            test_df,
            data["n_items"],
            k=TOP_K,
        ),
        "HybridNeuMF (W2 Deep CF)": evaluate_ranking(
            hybrid_rank_score,
            train_df,
            test_df,
            data["n_items"],
            k=TOP_K,
        ),
    }
    rank_df = pd.DataFrame(ranking_rows).T.sort_values("NDCG@10", ascending=False)

    # 6. Slice-based error analysis
    slices = slice_error_analysis(train_df, test_df, hybrid_test_preds)

    # 7. Save tables and figures
    runs_dir.mkdir(parents=True, exist_ok=True)
    reg_df.round(4).to_csv(runs_dir / "test_regression_comparison.csv")
    clf_df.round(4).to_csv(runs_dir / "test_classification_comparison.csv")
    rank_df.round(4).to_csv(runs_dir / "test_ranking_comparison.csv")

    summary_payload = {
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_phase": payload["phase"],
        "best_val_rmse": float(payload["best_val_rmse"]),
        "decision_threshold": best_thr,
        "hybrid_test_metrics": hybrid_metrics,
        "bootstrap_95ci": ci_95,
        "regression_comparison": reg_df.round(4).to_dict(orient="index"),
        "classification_comparison": clf_df.round(4).to_dict(orient="index"),
        "ranking_comparison": rank_df.round(4).to_dict(orient="index"),
        "user_slice_analysis": slices["user_slices"].to_dict(orient="records"),
        "item_slice_analysis": slices["item_slices"].to_dict(orient="records"),
    }
    with open(runs_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    plot_evaluation_figures(
        reg_df=reg_df,
        clf_df=clf_df,
        y_true=y_te,
        bias_preds=bias_test_preds,
        hybrid_preds=hybrid_test_preds,
        model=model,
        train_df=train_df,
        runs_dir=runs_dir,
    )

    print("\n=== 1. Regression Comparison on Held-Out Test Set ===")
    print(reg_df.round(4).to_string())

    print("\n=== 2. Binary Classification Comparison (like: rating >= 4.0) ===")
    print(clf_df.round(4).to_string())

    print(f"\n=== 3. Top-{TOP_K} Ranking Comparison ===")
    print(rank_df.round(4).to_string())

    print("\n=== 4. 95% Bootstrap Confidence Intervals (HybridNeuMF) ===")
    for k_ci, (lo, hi) in ci_95.items():
        print(f"  {k_ci:10s}: [{lo:.4f}, {hi:.4f}]")

    print("\n=== 5. Slice-Based Error Analysis (User Activity & Item Popularity) ===")
    print(slices["user_slices"].to_string(index=False))
    print()
    print(slices["item_slices"].to_string(index=False))

    return summary_payload


if __name__ == "__main__":
    evaluate_on_test_set()
