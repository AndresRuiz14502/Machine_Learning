"""Métricas y modelos base (Popularity, Bias, kNN item-item y modelos tabulares)."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.data_loader import (
    N_USERS_SAMPLE_TREES,
    PROJECT_ROOT,
    SEED,
    TOP_K,
    prepare_datasets,
)
from src.features import prepare_features


def mse(y, yhat) -> float:
    return float(mean_squared_error(y, yhat))


def rmse(y, yhat) -> float:
    return float(np.sqrt(mean_squared_error(y, yhat)))


def mae(y, yhat) -> float:
    return float(mean_absolute_error(y, yhat))


def r2(y, yhat) -> float:
    return float(r2_score(y, yhat))


def clip_rating(x):
    """Clipping a la escala válida [0.5, 5.0]."""
    return np.clip(x, 0.5, 5.0)


def log_result(results: dict, name: str, y, yhat):
    yhat_c = clip_rating(yhat)
    r = rmse(y, yhat_c)
    mse_val = mse(y, yhat_c)
    m = mae(y, yhat_c)
    r2_val = r2(y, yhat_c)
    results[name] = {"RMSE": r, "MSE": mse_val, "MAE": m, "R2": r2_val}
    print(f"{name:22s}  RMSE={r:.4f}  MSE={mse_val:.4f}  MAE={m:.4f}  R2={r2_val:.4f}")
    return r, m


def dcg_at_k(rels) -> float:
    rels = np.asarray(rels, dtype=float)
    return float(np.sum(rels / np.log2(np.arange(2, rels.size + 2))))


def ranking_metrics(recommended, relevant, k: int):
    rec_k = recommended[:k]
    hits = [1 if it in relevant else 0 for it in rec_k]
    n_rel = len(relevant)
    if n_rel == 0:
        return None
    precision = sum(hits) / k
    recall = sum(hits) / n_rel
    hit_rate = 1.0 if sum(hits) > 0 else 0.0
    idcg = dcg_at_k(sorted(hits, reverse=True)) or 1.0
    ndcg = dcg_at_k(hits) / idcg
    ap, acc = 0.0, 0
    for rank, h in enumerate(hits, start=1):
        if h:
            acc += 1
            ap += acc / rank
    ap = ap / min(n_rel, k)
    return dict(precision=precision, recall=recall, ndcg=ndcg, map=ap, hit=hit_rate)


def evaluate_ranking(score_fn, train_df: pd.DataFrame, test_df: pd.DataFrame, n_items: int, k: int = TOP_K, max_users: int = 2000, n_neg: int = 200, seed: int = SEED):
    """score_fn(u, items)->scores. Relevantes = items con like==1 en test."""
    rng_local = np.random.default_rng(seed)
    test_pos = test_df[test_df["like"] == 1].groupby("u")["i"].apply(set)
    users = test_pos.index.to_numpy()
    if len(users) > max_users:
        users = rng_local.choice(users, max_users, replace=False)
    seen = train_df.groupby("u")["i"].apply(set)
    all_items = np.arange(n_items)
    agg = {kk: [] for kk in ["precision", "recall", "ndcg", "map", "hit"]}
    for u in users:
        relevant = test_pos[u]
        seen_u = seen.get(u, set())
        negs = rng_local.choice(all_items, size=min(n_neg, n_items), replace=False)
        cand = np.array(list(relevant | (set(negs) - seen_u)))
        scores = score_fn(u, cand)
        order = cand[np.argsort(-scores)]
        m = ranking_metrics(list(order), relevant, k)
        if m:
            for kk in agg:
                agg[kk].append(m[kk])
    return {
        f"P@{k}": float(np.mean(agg["precision"])),
        f"R@{k}": float(np.mean(agg["recall"])),
        f"NDCG@{k}": float(np.mean(agg["ndcg"])),
        f"MAP@{k}": float(np.mean(agg["map"])),
        f"HitRate@{k}": float(np.mean(agg["hit"])),
    }


def run_popularity_and_bias(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mu: float,
    user_bias: pd.Series,
    item_bias: pd.Series,
    n_users: int,
    n_items: int,
    results: dict,
    ranking_results: dict,
):
    """Evalúa GlobalMean, Bias (mu+bu+bi) y Popularity."""
    bu = user_bias.to_dict()
    bi = item_bias.to_dict()

    def bias_predict(df_part):
        return (
            mu
            + df_part["u"].map(bu).fillna(0).to_numpy()
            + df_part["i"].map(bi).fillna(0).to_numpy()
        )

    log_result(results, "GlobalMean", test_df["rating"], np.full(len(test_df), mu))
    log_result(results, "Bias(mu+bu+bi)", test_df["rating"], bias_predict(test_df))

    item_pop = train_df.groupby("i").size()
    pop_vec = np.zeros(n_items)
    pop_vec[item_pop.index.to_numpy()] = item_pop.to_numpy()
    bi_vec = np.zeros(n_items)
    for i_, v in bi.items():
        bi_vec[i_] = v
    bu_arr = np.zeros(n_users)
    for u_, v in bu.items():
        bu_arr[u_] = v

    ranking_results["Popularity"] = evaluate_ranking(
        lambda u, items: pop_vec[items], train_df, test_df, n_items
    )
    ranking_results["Bias"] = evaluate_ranking(
        lambda u, items: mu + bu_arr[u] + bi_vec[items], train_df, test_df, n_items
    )
    for name in ["Popularity", "Bias"]:
        print(name, {k: round(v, 4) for k, v in ranking_results[name].items()})

    return pop_vec, bu_arr, bi_vec


def run_knn_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    item_stats: pd.DataFrame,
    mu: float,
    n_users: int,
    n_items: int,
    results: dict,
    knn_k: int = 40,
    seed: int = SEED,
):
    """kNN item-item basado en similitud coseno sobre la matriz centrada."""
    r_centered = train_df["rating"].to_numpy() - train_df["i"].map(item_stats["i_mean"]).to_numpy()
    R = csr_matrix((r_centered, (train_df["u"], train_df["i"])), shape=(n_users, n_items))

    item_sim = cosine_similarity(R.T, dense_output=False)

    def knn_predict_rating(df_part):
        Rt = R.tocsr()
        preds = np.empty(len(df_part))
        i_mean = item_stats["i_mean"].to_dict()
        for row, (u_, i_) in enumerate(zip(df_part["u"].to_numpy(), df_part["i"].to_numpy())):
            sims = item_sim.getrow(i_).toarray().ravel()
            user_items = Rt.getrow(u_).indices
            if user_items.size == 0:
                preds[row] = i_mean.get(i_, mu)
                continue
            s = sims[user_items]
            top = np.argsort(-s)[:knn_k]
            idx = user_items[top]
            w = sims[idx]
            denom = np.abs(w).sum()
            if denom == 0:
                preds[row] = i_mean.get(i_, mu)
                continue
            preds[row] = i_mean.get(i_, mu) + (w * Rt[u_, idx].toarray().ravel()).sum() / denom
        return preds

    knn_eval = test_df.sample(n=min(20_000, len(test_df)), random_state=seed)
    log_result(results, "kNN item-item", knn_eval["rating"], knn_predict_rating(knn_eval))


def analyze_residuals(
    y,
    predictions: dict,
    save_path: Path | None = PROJECT_ROOT / "runs" / "residuals_regression.png",
    show: bool = False,
) -> pd.DataFrame:
    """Calcula estadísticos y grafica el análisis de residuales (y - yhat)."""
    y_arr = np.asarray(y, dtype=float)
    stats_rows = []

    n_models = len(predictions)
    fig, axes = plt.subplots(n_models, 2, figsize=(12, 3.6 * n_models))
    if n_models == 1:
        axes = np.array([axes])

    for idx, (name, yhat) in enumerate(predictions.items()):
        yhat_c = clip_rating(np.asarray(yhat, dtype=float))
        res = y_arr - yhat_c
        stats_rows.append(
            {
                "modelo": name,
                "mean": float(np.mean(res)),
                "std": float(np.std(res)),
                "median": float(np.median(res)),
                "q25": float(np.percentile(res, 25)),
                "q75": float(np.percentile(res, 75)),
                "min": float(np.min(res)),
                "max": float(np.max(res)),
            }
        )

        ax_scatter, ax_hist = axes[idx]
        ax_scatter.scatter(yhat_c, res, alpha=0.15, s=8, color="#4C72B0")
        ax_scatter.axhline(0.0, color="#C44E52", linestyle="--", linewidth=1.2)
        ax_scatter.set_title(f"{name} - Residuales vs. Prediccion")
        ax_scatter.set_xlabel("Prediccion (yhat)")
        ax_scatter.set_ylabel("Residual (y - yhat)")

        ax_hist.hist(res, bins=40, color="#55A868", edgecolor="white", alpha=0.85)
        ax_hist.axvline(0.0, color="#C44E52", linestyle="--", linewidth=1.2)
        ax_hist.set_title(f"{name} - Distribucion de residuales")
        ax_hist.set_xlabel("Residual (y - yhat)")
        ax_hist.set_ylabel("Frecuencia")

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Grafico de residuales guardado en: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    summary_df = pd.DataFrame(stats_rows).set_index("modelo")
    print("\n=== Analisis de residuales (y - yhat) ===")
    print(summary_df.round(4).to_string())
    return summary_df


def run_tabular_baselines(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_cols: list,
    results: dict,
    n_users_sample_trees: int = N_USERS_SAMPLE_TREES,
    seed: int = SEED,
    plot_residuals_path: Path | None = PROJECT_ROOT / "runs" / "residuals_regression.png",
    show_plots: bool = False,
):
    """Entrena y evalúa Ridge, Lasso, DecisionTree, RandomForest y GradientBoosting."""
    Xtr, ytr = X_train[feature_cols].to_numpy(), X_train["rating"].to_numpy()
    Xte, yte = X_test[feature_cols].to_numpy(), X_test["rating"].to_numpy()

    n_tree_rows = min(len(Xtr), n_users_sample_trees * 150)
    ridx = np.random.default_rng(seed).choice(len(Xtr), n_tree_rows, replace=False)
    Xtr_s, ytr_s = Xtr[ridx], ytr[ridx]

    models = {
        "Ridge": Ridge(alpha=1.0, random_state=seed),
        "Lasso": Lasso(alpha=0.001, random_state=seed, max_iter=5000),
        "DecisionTree": DecisionTreeRegressor(max_depth=12, random_state=seed),
        "RandomForest": RandomForestRegressor(
            n_estimators=80, max_depth=16, n_jobs=-1, random_state=seed
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=3, random_state=seed
        ),
    }

    preds = {}
    for name, mdl in models.items():
        t = time.time()
        use_sub = name in ("DecisionTree", "RandomForest", "GradientBoosting")
        mdl.fit(Xtr_s if use_sub else Xtr, ytr_s if use_sub else ytr)
        yhat = mdl.predict(Xte)
        preds[name] = yhat
        log_result(results, name, yte, yhat)
        print(f"   ({time.time() - t:.1f}s)")

    analyze_residuals(yte, preds, save_path=plot_residuals_path, show=show_plots)
    return models


def log_classification_result(clf_results: dict, name: str, y_true, y_pred, y_prob):
    """Calcula accuracy, precision, recall, F1, AUC-ROC y matriz de confusión."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }
    cm = confusion_matrix(y_true, y_pred)
    clf_results[name] = {**metrics, "confusion_matrix": cm.tolist()}
    print(
        f"{name:22s}  Acc={metrics['accuracy']:.4f}  Prec={metrics['precision']:.4f}  "
        f"Rec={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  AUC={metrics['roc_auc']:.4f}"
    )
    print(f"  Matriz de confusion:\n{cm}")
    return metrics, cm


def run_classification_baselines(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_cols: list,
    clf_results: dict,
    n_users_sample_trees: int = N_USERS_SAMPLE_TREES,
    seed: int = SEED,
    plot_cm_path: Path | None = PROJECT_ROOT / "runs" / "confusion_matrices.png",
    show_plots: bool = False,
):
    """Entrena y evalúa 3 clasificadores base sobre la etiqueta binaria 'like'."""
    Xtr, ytr = X_train[feature_cols].to_numpy(), X_train["like"].to_numpy()
    Xte, yte = X_test[feature_cols].to_numpy(), X_test["like"].to_numpy()

    n_tree_rows = min(len(Xtr), n_users_sample_trees * 150)
    ridx = np.random.default_rng(seed).choice(len(Xtr), n_tree_rows, replace=False)
    Xtr_s, ytr_s = Xtr[ridx], ytr[ridx]

    classifiers = {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        ),
        "RandomForestClf": RandomForestClassifier(
            n_estimators=80, max_depth=16, n_jobs=-1, random_state=seed
        ),
        "GradientBoostingClf": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, random_state=seed
        ),
    }

    print("\n=== Clasificacion binaria (like: rating >= 4.0) ===")
    cms = {}
    for name, clf in classifiers.items():
        t = time.time()
        use_sub = name in ("RandomForestClf", "GradientBoostingClf")
        clf.fit(Xtr_s if use_sub else Xtr, ytr_s if use_sub else ytr)
        y_pred = clf.predict(Xte)
        y_prob = clf.predict_proba(Xte)[:, 1]
        _, cm = log_classification_result(clf_results, name, yte, y_pred, y_prob)
        cms[name] = cm
        print(f"   ({time.time() - t:.1f}s)")

    fig, axes = plt.subplots(1, len(cms), figsize=(5 * len(cms), 4))
    if len(cms) == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, cms.items()):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["no-like (0)", "like (1)"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(name)
    plt.tight_layout()
    if plot_cm_path is not None:
        plot_cm_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_cm_path, dpi=120, bbox_inches="tight")
        print(f"Matrices de confusion guardadas en: {plot_cm_path}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    return classifiers


def run_all_baselines():
    """Ejecuta de punta a punta carga, features y todos los baselines."""
    data = prepare_datasets(save_splits_file=False, verbose=True)
    feats = prepare_features(
        data["train_df"],
        data["val_df"],
        data["test_df"],
        data["movies"],
        data["item2idx"],
    )

    results = {}
    ranking_results = {}
    classification_results = {}

    pop_vec, bu_arr, bi_vec = run_popularity_and_bias(
        data["train_df"],
        data["test_df"],
        feats["mu"],
        feats["user_bias"],
        feats["item_bias"],
        data["n_users"],
        data["n_items"],
        results,
        ranking_results,
    )
    run_knn_baseline(
        data["train_df"],
        data["test_df"],
        feats["item_stats"],
        feats["mu"],
        data["n_users"],
        data["n_items"],
        results,
    )
    run_tabular_baselines(
        feats["X_train"],
        feats["X_test"],
        feats["FEATURE_COLS"],
        results,
    )
    run_classification_baselines(
        feats["X_train"],
        feats["X_test"],
        feats["FEATURE_COLS"],
        classification_results,
    )
    return {
        "data": data,
        "features": feats,
        "results": results,
        "ranking_results": ranking_results,
        "classification_results": classification_results,
        "pop_vec": pop_vec,
        "bu_arr": bu_arr,
        "bi_vec": bi_vec,
    }


if __name__ == "__main__":
    run_all_baselines()
