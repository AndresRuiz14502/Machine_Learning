"""
Evaluation and figure generation for Workshop 3:
Reinforcement Learning for Sequential Movie Recommendation on MovieLens 32M.

Every policy is evaluated once on every RL test session (`test_user_indices` of
`workshop3/data/processed_cache.npz`) at horizons K=10 and K=20:

- `Random`, `BiasOnly`, `Popularity`: non-personalized or bias-only baselines;
- `GreedyFunkSVD`: greedy on the static Workshop 1 FunkSVD scores of the state;
- `GreedyHybridNeuMF`: greedy on the Workshop 2 HybridNeuMF scores precomputed in
  `workshop3/data/w2_candidate_scores.npz` (see `workshop3/src/w2_scores.py`);
- `MyopicBandit`: the gamma = 0 DQN (`workshop3/checkpoints/bandit_gamma0.pt`), greedy;
- `D3QN`: the champion Dueling Double DQN (`workshop3/checkpoints/agent_best.pt`), greedy.

The script also computes paired per-user comparisons (bootstrap CI and Wilcoxon test),
evaluates the baselines on the RL validation sessions for the sample-efficiency analysis
(the split of the training-time validation curves), adds the Workshop 1 and Workshop 2
reference blocks, records trajectories of fixed test users, writes
`workshop3/report/metrics_summary.json`, and draws Figures 3-7 (300 DPI) in
`workshop3/report/figures/`.

Usage, from the repository root (after `python workshop3/src/train.py`):

    python workshop3/src/evaluate.py
"""

from __future__ import annotations

import inspect
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workshop3.src.agent import DQNAgent
from workshop3.src.baselines import (
    BiasPolicy,
    GreedyFunkSVDPolicy,
    GreedyHybridNeuMFPolicy,
    PopularityPolicy,
    RandomPolicy,
    evaluate_policy,
)
from workshop3.src.env import MovieLensRecEnv
from workshop3.src.train import SMOOTHING_WINDOW
from workshop3.src.w2_scores import DEFAULT_SCORES_PATH, load_w2_candidate_scores, repo_relative, sha256_file

W3_DIR = PROJECT_ROOT / "workshop3"
CACHE_PATH = W3_DIR / "data" / "processed_cache.npz"
TRAINING_LOGS_PATH = W3_DIR / "report" / "training_logs.json"
CHAMPION_CHECKPOINT = W3_DIR / "checkpoints" / "agent_best.pt"
BANDIT_CHECKPOINT = W3_DIR / "checkpoints" / "bandit_gamma0.pt"
SUMMARY_PATH = W3_DIR / "report" / "metrics_summary.json"
FIG_DIR = W3_DIR / "report" / "figures"
W2_SUMMARY_PATH = PROJECT_ROOT / "runs" / "evaluation_summary.json"
W2_HISTORY_PATH = PROJECT_ROOT / "runs" / "training_history.json"
W2_SPLITS_PATH = PROJECT_ROOT / "data" / "splits.json"
W2_RANKING_KEY = "HybridNeuMF (W2 Deep CF)"

SEED = 42
N_BOOTSTRAP = 10_000
NUM_TRAJECTORIES = 4
TEST_HORIZONS = (10, 20)

STATIC_BASELINES = ("Random", "BiasOnly", "Popularity", "GreedyFunkSVD", "GreedyHybridNeuMF")
POLICY_ORDER = STATIC_BASELINES + ("MyopicBandit", "D3QN")
VALIDATION_BASELINES = STATIC_BASELINES + ("MyopicBandit",)
PAIRED_COMPARISONS = (
    ("D3QN", "MyopicBandit"),
    ("D3QN", "GreedyFunkSVD"),
    ("D3QN", "GreedyHybridNeuMF"),
    ("GreedyHybridNeuMF", "GreedyFunkSVD"),
)
SNAKE_NAMES = {
    "Random": "random",
    "BiasOnly": "bias_only",
    "Popularity": "popularity",
    "GreedyFunkSVD": "greedy_funksvd",
    "GreedyHybridNeuMF": "greedy_hybrid_neumf",
    "MyopicBandit": "myopic_bandit",
}
POLICY_COLORS = {
    "Random": "#94a3b8",
    "BiasOnly": "#f59e0b",
    "Popularity": "#ef4444",
    "GreedyFunkSVD": "#3b82f6",
    "GreedyHybridNeuMF": "#0e7490",
    "MyopicBandit": "#8b5cf6",
    "D3QN": "#10b981",
}
BASELINE_LINESTYLES = {
    "Random": ":",
    "BiasOnly": "--",
    "Popularity": (0, (1, 1)),
    "GreedyFunkSVD": "-.",
    "GreedyHybridNeuMF": (0, (6, 2, 1, 2, 1, 2)),
    "MyopicBandit": (0, (4, 1.5)),
}
EVALUATION_PROTOCOL = (
    "Every policy is evaluated once on every RL test session (test_user_indices of "
    "workshop3/data/processed_cache.npz): one episode per test user at horizon 10 (test_k10) and "
    "horizon 20 (test_k20). DQN policies act greedily (epsilon = 0) over the valid-action mask; "
    "RandomPolicy uses a fresh generator seeded with 42 in every evaluation. Ranking metrics treat "
    "a step as relevant when the recommended slot holds a real rating >= 4.0; recall, NDCG and MAP "
    "use the number of liked slots of the session as the relevant set. catalog_coverage is the "
    "number of distinct movies recommended divided by the number of distinct movies in the "
    "evaluated sessions' candidate slates (padding slots excluded)."
)

# Workshop 1 report: "Supervised Matrix Factorization for Movie Recommendation on the MovieLens
# 32M Dataset" (group7_workshop1.pdf, not versioned here). Only values checked against that
# report are kept; each entry records where it appears.
W1_REPORT = (
    "Workshop 1 report 'Supervised Matrix Factorization for Movie Recommendation on the "
    "MovieLens 32M Dataset' (group7_workshop1.pdf, sha256 "
    "f6bdb664ffaef335b144eabe529bc09179407512979ab391fbc910be0b6fef43)"
)
W1_RANKING_SOURCE = (
    "W1 report Table III (ranking metrics on the test set), row FunkSVD, K = 10. Sec. III-D "
    "describes this row as the tuned FunkSVD (k = 20, lambda = 0.02, eta = 0.01), and Sec. II-F "
    "the protocol (each test user's liked items ranked with sampled negatives not seen in train), "
    "the same src/baselines.py evaluate_ranking protocol as Workshop 2"
)
WORKSHOP1_REFERENCE: Dict[str, Any] = {
    "source_document": W1_REPORT,
    "funksvd_k50_test_rmse": {
        "value": 0.8041,
        "source": "W1 report Table I (rating metrics on the test set), row FunkSVD (k = 50); also Sec. III-B and Sec. V",
    },
    "funksvd_k50_test_mae": {
        "value": 0.5998,
        "source": "W1 report Table I, row FunkSVD (k = 50); also Sec. III-B and Sec. V",
    },
    "funksvd_precision_at_10": {"value": 0.3260, "source": W1_RANKING_SOURCE},
    "funksvd_recall_at_10": {"value": 0.2127, "source": W1_RANKING_SOURCE},
    "funksvd_ndcg_at_10": {"value": 0.5881, "source": W1_RANKING_SOURCE},
    "funksvd_map_at_10": {"value": 0.2589, "source": W1_RANKING_SOURCE},
    "funksvd_hr_at_10": {"value": 0.8043, "source": W1_RANKING_SOURCE},
    "funksvd_top10_catalog_coverage_pct": {
        "value": 8.00,
        "source": "W1 report Sec. III-E (beyond-accuracy metrics of the FunkSVD top-10 lists)",
    },
    "funksvd_top10_novelty_bits": {
        "value": 14.06,
        "source": "W1 report Sec. III-E (mean novelty of the FunkSVD top-10 lists)",
    },
}
W1_PROTOCOL = (
    "Static one-shot evaluation reported in the Workshop 1 report: RMSE and MAE of FunkSVD with "
    "k = 50 (Table I); top-10 ranking metrics of the tuned FunkSVD with k = 20 (Table III) under a "
    "sampled-negative protocol, where each test user's liked test items are ranked together with "
    "sampled movies not seen in train; coverage and novelty describe the top-10 lists over the "
    "Workshop 1 catalog. Not comparable 1:1 with the Workshop 3 sessions, where a policy picks 10 "
    "of at most 50 real ratings of one session and coverage is measured over the movies of the "
    "evaluated sessions."
)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _r4(value: float) -> float:
    return round(float(value), 4)


def format_gamma(gamma: float) -> str:
    """Format a discount factor without trailing zeros (0.95 -> '0.95', 0.0 -> '0')."""
    return f"{float(gamma):g}"


def policy_labels(champion_gamma: float, bandit_gamma: float) -> Dict[str, str]:
    """Display names of the evaluated policies (gammas read from the checkpoint configs)."""
    return {
        "Random": "Random",
        "BiasOnly": "Bias Only (W1)",
        "Popularity": "Popularity (W1)",
        "GreedyFunkSVD": "Greedy FunkSVD (W1)",
        "GreedyHybridNeuMF": "Greedy HybridNeuMF (W2)",
        "MyopicBandit": f"Myopic DQN (γ={format_gamma(bandit_gamma)})",
        "D3QN": f"D3QN (γ={format_gamma(champion_gamma)})",
    }


# --------------------------------------------------------------------------------------
# Reference blocks (Workshops 1 and 2)
# --------------------------------------------------------------------------------------
def build_w2_static_reference() -> Dict[str, Any]:
    """Official Workshop 2 HybridNeuMF results, read from the files the Workshop 2 code wrote."""
    from src.baselines import evaluate_ranking

    n_sampled = int(inspect.signature(evaluate_ranking).parameters["n_neg"].default)
    summary = _read_json(W2_SUMMARY_PATH)
    history = _read_json(W2_HISTORY_PATH)
    split_counts = _read_json(W2_SPLITS_PATH)["counts"]
    hybrid = summary["hybrid_test_metrics"]
    ranking = summary["ranking_comparison"][W2_RANKING_KEY]
    train_ratings = int(split_counts["train"])
    training_epochs = len(history)
    return {
        "source": (
            f"{repo_relative(W2_SUMMARY_PATH)} (hybrid_test_metrics; ranking_comparison['{W2_RANKING_KEY}']), "
            f"{repo_relative(W2_HISTORY_PATH)} (one row per epoch), {repo_relative(W2_SPLITS_PATH)} (counts)"
        ),
        "protocol": (
            "Static one-shot Workshop 2 evaluation on the global chronological 80/10/10 split that "
            "Workshop 3 also uses. RMSE, MAE and ROC-AUC cover all clean test ratings. The top-10 "
            "metrics follow a sampled-candidate protocol, not a full-catalog ranking "
            "(src/baselines.py evaluate_ranking): each test user with liked (rating >= 4.0) test "
            f"items ranks them together with {n_sampled} sampled unrated movies ({n_sampled} movies "
            "drawn at random from the catalog, minus those the user rated in train), scored by "
            "clamp(prediction, 0.5, 5) + 1.35 x normalized log train popularity."
        ),
        "ranking_protocol": "sampled_candidate",
        "ranking_sampled_movies_per_user": n_sampled,
        "test_rmse": _r4(hybrid["RMSE"]),
        "test_mae": _r4(hybrid["MAE"]),
        "roc_auc": _r4(hybrid["roc_auc"]),
        "precision_at_10": _r4(ranking["P@10"]),
        "recall_at_10": _r4(ranking["R@10"]),
        "ndcg_at_10": _r4(ranking["NDCG@10"]),
        "map_at_10": _r4(ranking["MAP@10"]),
        "hr_at_10": _r4(ranking["HitRate@10"]),
        "train_ratings": train_ratings,
        "test_ratings": int(split_counts["test"]),
        "training_epochs": training_epochs,
        "epochs_by_phase": dict(Counter(str(row["phase"]) for row in history)),
        "ratings_processed": train_ratings * training_epochs,
        "ratings_processed_note": (
            "train_ratings x training_epochs: one pass over the training split per logged HybridNeuMF "
            "epoch. The BiasedMF pretraining epochs of src/train.py (PRETRAIN_EPOCHS) are not logged "
            "in runs/training_history.json and are not counted."
        ),
    }


def _sequential_block(metrics: Dict[str, float], source: str) -> Dict[str, Any]:
    return {
        "mean_return_k10": metrics["mean_return"],
        "precision_at_10": metrics["precision_at_10"],
        "recall_at_10": metrics["recall_at_10"],
        "ndcg_at_10": metrics["ndcg_at_10"],
        "map_at_10": metrics["map_at_10"],
        "hr_at_10": metrics["hr_at_10"],
        "catalog_coverage_pct": round(metrics["catalog_coverage"] * 100.0, 2),
        "novelty_bits": metrics["mean_novelty_bits"],
        "source": source,
    }


def build_workshop1_comparison(test_k10: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    ref = WORKSHOP1_REFERENCE
    return {
        "w1_report_funksvd": {
            "test_rmse_k50": ref["funksvd_k50_test_rmse"]["value"],
            "test_mae_k50": ref["funksvd_k50_test_mae"]["value"],
            "precision_at_10": ref["funksvd_precision_at_10"]["value"],
            "recall_at_10": ref["funksvd_recall_at_10"]["value"],
            "ndcg_at_10": ref["funksvd_ndcg_at_10"]["value"],
            "map_at_10": ref["funksvd_map_at_10"]["value"],
            "hr_at_10": ref["funksvd_hr_at_10"]["value"],
            "catalog_coverage_pct": ref["funksvd_top10_catalog_coverage_pct"]["value"],
            "novelty_bits": ref["funksvd_top10_novelty_bits"]["value"],
            "protocol": W1_PROTOCOL,
            "source": "workshop1_reference (values transcribed from the Workshop 1 report, with locations)",
        },
        "w3_sequential_greedy_funksvd": _sequential_block(
            test_k10["GreedyFunkSVD"],
            "test_k10.GreedyFunkSVD: greedy on the Workshop 3 FunkSVD scores (k = 20, trained on the "
            "same 80% training split) in the Workshop 3 environment",
        ),
    }


def build_workshop2_comparison(test_k10: Dict[str, Dict[str, float]], w2: Dict[str, Any]) -> Dict[str, Any]:
    block = _sequential_block(
        test_k10["GreedyHybridNeuMF"],
        "test_k10.GreedyHybridNeuMF: GreedyHybridNeuMFPolicy acting on "
        f"{repo_relative(DEFAULT_SCORES_PATH)} in the Workshop 3 environment",
    )
    block.update(
        {
            "scores_artifact": repo_relative(DEFAULT_SCORES_PATH),
            "score_definition": w2["meta"]["score_definition"],
            "checkpoint": w2["meta"]["checkpoint"],
            "alignment": w2["meta"]["alignment"],
            "inference_check": w2["meta"]["inference_check"],
        }
    )
    return {"w2_hybrid_neumf_static": build_w2_static_reference(), "w3_sequential_greedy_hybrid_neumf": block}


# --------------------------------------------------------------------------------------
# Agents and policies
# --------------------------------------------------------------------------------------
AGENT_INIT_KEYS = (
    "state_dim",
    "action_dim",
    "hidden_dim",
    "lr",
    "gamma",
    "epsilon_start",
    "epsilon_end",
    "epsilon_decay",
    "weight_decay",
    "batch_size",
    "target_update_freq",
    "tau",
    "grad_clip",
    "seed",
)


def load_greedy_agent(path: Path, device: torch.device) -> Tuple[DQNAgent, Dict[str, Any]]:
    """Load a `DQNAgent` checkpoint as a greedy policy; return it with its provenance."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {repo_relative(path)}. Run `python workshop3/src/train.py` first.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(payload["config"])
    kwargs = {key: config[key] for key in AGENT_INIT_KEYS if key in config}
    # Evaluation only: the replay buffer is never filled
    agent = DQNAgent(**kwargs, buffer_capacity=1, device=device)
    agent.load(path)
    agent.epsilon = 0.0
    agent.policy_net.eval()
    info = {
        "checkpoint": repo_relative(path),
        "sha256": sha256_file(path),
        "gamma": float(config["gamma"]),
        "saved_after_episodes": int(payload.get("episodes_done", -1)),
        "saved_after_updates": int(payload.get("update_steps", -1)),
        "config": config,
    }
    return agent, info


class PolicyFactory:
    """Build a fresh policy instance per evaluation, so seeded generators restart every time."""

    def __init__(self, cache_data: Dict[str, Any], w2: Dict[str, Any], champion: DQNAgent, bandit: DQNAgent) -> None:
        self.cache_data = cache_data
        self.w2 = w2
        self.agents = {"D3QN": champion, "MyopicBandit": bandit}

    def __call__(self, name: str) -> Any:
        if name in self.agents:
            return self.agents[name]  # greedy and stateless at evaluation time
        if name == "Random":
            return RandomPolicy(seed=SEED, cache_data=self.cache_data)
        if name == "BiasOnly":
            return BiasPolicy(cache_data=self.cache_data)
        if name == "Popularity":
            return PopularityPolicy(cache_data=self.cache_data)
        if name == "GreedyFunkSVD":
            return GreedyFunkSVDPolicy(cache_data=self.cache_data)
        if name == "GreedyHybridNeuMF":
            return GreedyHybridNeuMFPolicy(self.w2["scores"], slot_valid=self.w2["cand_valid"])
        raise KeyError(f"Unknown policy '{name}'")


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def ranking_metrics_at_k(per_episode: Sequence[Dict[str, Any]], k: int) -> Dict[str, float]:
    """Precision, recall, NDCG, MAP and HR at `k`, with the formulas of `evaluate_policy` (@10)."""
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float32))
    ranks = np.arange(1, k + 1, dtype=np.float32)
    values: Dict[str, List[float]] = {"precision": [], "recall": [], "ndcg": [], "map": [], "hr": []}
    for episode in per_episode:
        hits = np.zeros(k, dtype=np.float32)
        observed = np.asarray(episode["step_hits"][:k], dtype=np.float32)
        hits[: len(observed)] = observed
        n_hits = float(np.sum(hits))
        total_rel = max(1, int(episode["num_relevant"]))
        ideal_k = min(k, total_rel)
        values["precision"].append(n_hits / float(k))
        values["recall"].append(float(min(1.0, n_hits / float(total_rel))))
        values["hr"].append(1.0 if n_hits >= 1.0 else 0.0)
        values["ndcg"].append(float(np.sum(hits * discounts)) / float(np.sum(discounts[:ideal_k])))
        values["map"].append(float(np.sum((np.cumsum(hits) / ranks) * hits)) / float(ideal_k))
    return {name: float(np.mean(v)) if v else 0.0 for name, v in values.items()}


def evaluate_policy_at_k(
    env: MovieLensRecEnv,
    policy: Any,
    k_eval: int,
    user_indices: Sequence[int],
    seed: int = SEED,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Evaluate `policy` once on every user of `user_indices`; return metrics and per-episode records.

    The @10 metrics come from `evaluate_policy`; the @K metrics (`*_at_k`, plus `*_at_<K>` when
    K != 10) are computed from the per-episode hits of the same rollout, so every metric of a
    policy refers to the same users and the same recommendations.
    """
    if env.horizon < k_eval:
        raise ValueError(f"horizon {env.horizon} < K={k_eval}")
    raw = evaluate_policy(env, policy, seed=seed, user_indices=user_indices, return_per_episode=True)
    per_episode = raw.pop("per_episode")
    if len(per_episode) != len(user_indices) or [e["user_idx"] for e in per_episode] != [int(u) for u in user_indices]:
        raise RuntimeError("evaluate_policy did not evaluate every listed user exactly once, in order")
    at_k = ranking_metrics_at_k(per_episode, k_eval)
    if k_eval == 10:
        for name in at_k:
            if abs(at_k[name] - raw[f"{name}_at_10"]) > 1e-6:
                raise RuntimeError(f"{name}@10 recomputed from per-episode hits differs from evaluate_policy")

    metrics = {key: _r4(value) for key, value in raw.items()}
    if k_eval != 10:
        metrics.update({f"{name}_at_{k_eval}": _r4(value) for name, value in at_k.items()})
    metrics.update({f"{name}_at_k": _r4(value) for name, value in at_k.items()})
    return metrics, per_episode


def paired_comparison(
    per_a: Sequence[Dict[str, Any]],
    per_b: Sequence[Dict[str, Any]],
    metric: str,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Per-user paired comparison of `metric` (A - B): mean, bootstrap 95% CI, Wilcoxon p-value."""
    a_by_user = {int(e["user_idx"]): float(e[metric]) for e in per_a}
    b_by_user = {int(e["user_idx"]): float(e[metric]) for e in per_b}
    if set(a_by_user) != set(b_by_user):
        raise ValueError("Paired comparison needs both policies evaluated on the same users")
    users = sorted(a_by_user)
    a = np.array([a_by_user[u] for u in users], dtype=np.float64)
    b = np.array([b_by_user[u] for u in users], dtype=np.float64)
    diff = a - b

    rng = np.random.default_rng(seed)
    boot_means = diff[rng.integers(0, len(diff), size=(n_resamples, len(diff)))].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    n_nonzero = int(np.count_nonzero(diff))
    if n_nonzero == 0:
        statistic, p_value = 0.0, 1.0  # identical outcomes for every user: no evidence of a difference
    else:
        result = wilcoxon(diff)  # two-sided, zero differences dropped (zero_method="wilcox")
        statistic, p_value = float(result.statistic), float(result.pvalue)
    return {
        "mean_a": _r4(a.mean()),
        "mean_b": _r4(b.mean()),
        "mean_diff": _r4(diff.mean()),
        "ci95_low": _r4(ci_low),
        "ci95_high": _r4(ci_high),
        "wilcoxon_statistic": _r4(statistic),
        "wilcoxon_p_value": _r4(p_value),
        "n_users": int(len(users)),
        "n_nonzero_differences": n_nonzero,
        "a_better_users": int(np.sum(diff > 0)),
        "b_better_users": int(np.sum(diff < 0)),
        "tied_users": int(np.sum(diff == 0)),
    }


def build_paired_comparisons(per_user: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "protocol": (
            "K = 10 test sessions, one episode per test user and policy. For each pair (A, B) and "
            "metric, the per-user difference is A - B. mean_diff is its mean, [ci95_low, ci95_high] "
            f"the percentile 95% CI of a paired bootstrap over users ({N_BOOTSTRAP:,} resamples, "
            f"seed {SEED}), and wilcoxon_p_value the two-sided Wilcoxon signed-rank p-value "
            "(scipy.stats.wilcoxon, zero differences dropped; p = 1 when every difference is zero). "
            "All values are rounded to 4 decimals, so p-values below 0.00005 appear as 0.0."
        ),
        "metrics": ["return", "ndcg_at_10"],
    }
    for a_name, b_name in PAIRED_COMPARISONS:
        out[f"{a_name}_vs_{b_name}"] = {
            "policy_a": a_name,
            "policy_b": b_name,
            "return": paired_comparison(per_user[a_name], per_user[b_name], "return"),
            "ndcg_at_10": paired_comparison(per_user[a_name], per_user[b_name], "ndcg_at_10"),
        }
    return out


def first_surpassing_checkpoint(
    checkpoints: Sequence[Dict[str, Any]], metric: str, threshold: float
) -> Optional[Dict[str, Any]]:
    """First validation checkpoint whose `metric` strictly exceeds `threshold` (None if none does)."""
    for ckpt in checkpoints:
        if float(ckpt[metric]) > float(threshold):
            return {"steps": int(ckpt["steps"]), "episode": int(ckpt["episode"]), "value": float(ckpt[metric])}
    return None


def build_sample_efficiency(
    training_logs: Dict[str, Any],
    validation_baselines: Dict[str, Dict[str, float]],
    checkpoint_check: Dict[str, Any],
    num_val_users: int,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Optional[Dict[str, Any]]]]]:
    champ_ckpts = training_logs["champion_run"]["val_checkpoints"]
    bandit_ckpts = training_logs["bandit_run"]["val_checkpoints"]
    crossings: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = {}
    block: Dict[str, Any] = {
        "split": "validation",
        "num_validation_users": int(num_val_users),
        "criterion": (
            "steps_to_surpass_<baseline> is the cumulative number of training environment steps at "
            "the first champion validation checkpoint (training_logs.json champion_run.val_checkpoints, "
            "greedy policy, every RL validation session once, episode 0 included) whose mean_return "
            "strictly exceeds the baseline's mean_return on the same validation sessions; the "
            "_ndcg_at_10 variant uses NDCG@10. null means no checkpoint surpasses the baseline. The "
            "MyopicBandit reference is its saved (best validation return) checkpoint."
        ),
        "validation_reference": {
            name: {"mean_return": m["mean_return"], "ndcg_at_10": m["ndcg_at_10"]}
            for name, m in validation_baselines.items()
        },
    }
    for name in VALIDATION_BASELINES:
        snake = SNAKE_NAMES[name]
        ret_cross = first_surpassing_checkpoint(champ_ckpts, "mean_return", validation_baselines[name]["mean_return"])
        ndcg_cross = first_surpassing_checkpoint(champ_ckpts, "ndcg_at_10", validation_baselines[name]["ndcg_at_10"])
        crossings[name] = {"mean_return": ret_cross, "ndcg_at_10": ndcg_cross}
        block[f"steps_to_surpass_{snake}"] = None if ret_cross is None else ret_cross["steps"]
        block[f"steps_to_surpass_{snake}_ndcg_at_10"] = None if ndcg_cross is None else ndcg_cross["steps"]
    block["episodes_to_surpass"] = {
        SNAKE_NAMES[name]: {metric: None if c is None else c["episode"] for metric, c in crossings[name].items()}
        for name in VALIDATION_BASELINES
    }
    block.update(
        {
            "total_champion_training_steps": int(champ_ckpts[-1]["steps"]),
            "total_bandit_training_steps": int(bandit_ckpts[-1]["steps"]),
            "checkpoint_check": checkpoint_check,
            "champion_val_checkpoints": champ_ckpts,
            "bandit_val_checkpoints": bandit_ckpts,
        }
    )
    return block, crossings


def rollout_trajectory(env: MovieLensRecEnv, policy: Any, user_idx: int, cache_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run one greedy episode for `user_idx` and record every recommendation."""
    obs, info = env.reset(seed=SEED, options={"user_idx": int(user_idx)})
    done = False
    cum_return = 0.0
    steps: List[Dict[str, Any]] = []
    while not done:
        action = int(policy.select_action(obs, info))
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        cum_return += float(reward)
        h_last = env.history[-1]
        raw_title = str(h_last["title"])
        steps.append(
            {
                "t": int(h_last["step"]),
                "action_slot": int(h_last["action"]),
                "item_idx": int(h_last["movie_id"]),
                "movielens_movie_id": int(cache_data["cand_movie_ids"][int(user_idx), action]),
                "title": raw_title if len(raw_title) <= 28 else raw_title[:25] + "...",
                "full_title": raw_title,
                "genres": "|".join(str(h_last["genres"]).split("|")[:2]),
                "is_hit": int(h_last["is_hit"]),
                "true_rating": float(h_last["true_rating"]),
                "step_reward": _r4(reward),
                "novelty_bits": round(float(h_last["novelty_bits"]), 2),
                "cumulative_return": _r4(cum_return),
            }
        )
    return {
        "total_return": _r4(cum_return),
        "total_hits": int(sum(s["is_hit"] for s in steps)),
        "episode_length": len(steps),
        "mean_novelty_bits": round(float(np.mean([s["novelty_bits"] for s in steps])), 2),
        "steps": steps,
    }


def trajectory_positions(n_users: int, count: int = NUM_TRAJECTORIES) -> List[int]:
    """Evenly spaced positions in the test split, fixed before evaluation (never chosen by outcome)."""
    return sorted({int(p) for p in np.linspace(0, n_users - 1, num=min(count, n_users)).round()})


def extract_sample_trajectories(
    env: MovieLensRecEnv,
    champion: DQNAgent,
    bandit: DQNAgent,
    test_users: Sequence[int],
    positions: Sequence[int],
    cache_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Champion and gamma = 0 trajectories for the test users at the fixed `positions`."""
    trajectories: List[Dict[str, Any]] = []
    for idx, pos in enumerate(positions):
        u = int(test_users[pos])
        trajectories.append(
            {
                "trajectory_id": idx + 1,
                "test_position": int(pos),
                "user_pool_idx": u,
                "movielens_user_id": int(cache_data["pool_user_ids"][u]),
                "num_valid_slots": int(np.sum(cache_data["cand_valid"][u] > 0)),
                "num_liked_slots": int(np.sum(cache_data["cand_is_liked"][u] == 1)),
                **rollout_trajectory(env, champion, u, cache_data),
                "myopic_bandit": rollout_trajectory(env, bandit, u, cache_data),
            }
        )
    return trajectories


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def apply_academic_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.titlesize": 12.5,
        }
    )


def _save(fig: plt.Figure, save_path: Path) -> None:
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _padded_limits(values: Sequence[float], low_pad: float = 0.08, high_pad: float = 0.08) -> Tuple[float, float]:
    lo, hi = float(np.min(values)), float(np.max(values))
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    return lo - low_pad * span, hi + high_pad * span


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(int(window), len(arr)))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(np.asarray(arr, dtype=np.float64), (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _spread(values: Sequence[float], min_gap: float, lo: float = 0.04, hi: float = 0.96) -> np.ndarray:
    """Label positions (same order as `values`) inside [lo, hi], `min_gap` apart and close to `values`.

    An upward pass separates crowded labels, a downward pass pulls the top ones back under
    `hi`; if the labels cannot fit at all, they are spread evenly over [lo, hi].
    """
    vals = np.asarray(values, dtype=np.float64)
    order = np.argsort(vals, kind="mergesort")
    pos = np.clip(vals[order], lo, hi)
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    if len(pos):
        pos[-1] = min(pos[-1], hi)
    for i in range(len(pos) - 2, -1, -1):
        pos[i] = min(pos[i], pos[i + 1] - min_gap)
    if len(pos) and pos[0] < lo:
        pos = np.linspace(lo, hi, len(pos))
    out = np.empty_like(pos)
    out[order] = pos
    return out


def _config_label(row: Dict[str, Any]) -> str:
    match = re.match(r"cfg(\d+)", str(row["config_id"]))
    prefix = f"Cfg {match.group(1)}" if match else str(row["config_id"])
    varied = str(row["varied_param"])
    for old, new in (("weight_decay=", "L2="), ("eps_decay=", "ε-decay="), ("gamma=", "γ=")):
        varied = varied.replace(old, new)
    return f"{prefix}: {varied}"


def plot_fig3_hyperparameter_curves(
    training_logs: Dict[str, Any],
    labels: Dict[str, str],
    num_val_users: int,
    horizon: int,
    save_path: Path,
) -> None:
    """Figure 3: validation curves of the grid and champion runs, and smoothed validation returns."""
    apply_academic_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0), dpi=300)
    grid_runs = training_logs["grid_runs"]
    champion = training_logs["champion_run"]
    bandit = training_logs["bandit_run"]
    selected = training_logs.get("best_grid_config_id")
    cmap = matplotlib.colormaps["tab10"]
    linestyles = ["-", "--", "-.", ":"]
    grid_colors = {cid: cmap(i % 10) for i, cid in enumerate(grid_runs)}

    ax0 = axes[0]
    rows_by_id = {row["config_id"]: row for row in training_logs["hyperparameter_table"]}
    for i, (cid, run) in enumerate(grid_runs.items()):
        ckpts = run["val_checkpoints"]
        label = _config_label(rows_by_id.get(cid, {"config_id": cid, "varied_param": run.get("varied_param", "")}))
        if cid == selected:
            label += " (selected)"
        ax0.plot(
            [c["episode"] for c in ckpts],
            [c["mean_return"] for c in ckpts],
            color=grid_colors[cid],
            linestyle=linestyles[i % len(linestyles)],
            linewidth=1.4,
            alpha=0.85,
            label=label,
        )
    c_ckpts = champion["val_checkpoints"]
    ax0.plot(
        [c["episode"] for c in c_ckpts],
        [c["mean_return"] for c in c_ckpts],
        color="#111827",
        linewidth=2.4,
        label=f"Champion {labels['D3QN']} ({champion['hyperparameters']['num_episodes']} ep)",
    )
    grid_budget = max(run["val_checkpoints"][-1]["episode"] for run in grid_runs.values())
    ax0.axvline(grid_budget, color="#6b7280", linestyle=":", linewidth=1.2)
    ax0.annotate(
        f"grid budget: {grid_budget} episodes",
        xy=(grid_budget, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(4, -12),
        textcoords="offset points",
        fontsize=8,
        color="#4b5563",
    )
    ax0.set_title(f"(a) Greedy Validation Return during Training ({num_val_users} users)")
    ax0.set_xlabel("Training Episode")
    ax0.set_ylabel(f"Mean Validation Episode Return (H={horizon})")
    ax0.grid(True, linestyle=":", alpha=0.6)
    ax0.legend(loc="lower right", frameon=True, fontsize=7.6)

    # Panel (b): smoothed validation return (mean of the last SMOOTHING_WINDOW evaluations)
    ax1 = axes[1]
    runs = {**grid_runs, champion["config_id"]: champion, bandit["config_id"]: bandit}
    table = training_logs["hyperparameter_table"]
    names, vals, errs, colors = [], [], [], []
    for row in table:
        cid = row["config_id"]
        run = runs.get(cid)
        tail = [c["mean_return"] for c in run["val_checkpoints"][-SMOOTHING_WINDOW:]] if run else []
        if cid == champion["config_id"]:
            names.append(f"Champion {labels['D3QN']}\n({row['episodes']} ep)")
            colors.append("#111827")
        elif cid == bandit["config_id"]:
            names.append(f"{labels['MyopicBandit']}\n({row['episodes']} ep)")
            colors.append(POLICY_COLORS["MyopicBandit"])
        else:
            names.append(_config_label(row) + (" (selected)" if cid == selected else ""))
            colors.append(grid_colors.get(cid, "#64748b"))
        vals.append(float(row["smoothed_val_return"]))
        errs.append(float(np.std(tail)) if tail else 0.0)

    vals_arr, errs_arr = np.asarray(vals), np.asarray(errs)
    y_pos = np.arange(len(vals))
    bars = ax1.barh(y_pos, vals_arr, xerr=errs_arr, color=colors, edgecolor="#1f2937", height=0.68, alpha=0.88,
                    error_kw={"elinewidth": 1.0, "capsize": 3, "ecolor": "#374151"})
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8.2)
    ax1.invert_yaxis()
    lo, hi = float(np.min(vals_arr - errs_arr)), float(np.max(vals_arr + errs_arr))
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    ax1.set_xlim(lo - 0.45 * span, hi + 0.30 * span)
    for bar, v, e in zip(bars, vals_arr, errs_arr):
        ax1.text(v + e + 0.03 * span, bar.get_y() + bar.get_height() / 2.0, f"{v:.2f}",
                 va="center", ha="left", fontsize=8.3, fontweight="bold")
    ax1.set_title(f"(b) Smoothed Validation Return (mean ± std of last {SMOOTHING_WINDOW} evaluations)")
    ax1.set_xlabel("Smoothed Mean Validation Return")
    ax1.grid(axis="x", linestyle=":", alpha=0.6)
    _save(fig, save_path)


def plot_fig4_exploration_dynamics(training_logs: Dict[str, Any], labels: Dict[str, str], num_slots: int, save_path: Path) -> None:
    """Figure 4: exploration and training dynamics of the champion run."""
    apply_academic_style()
    champ = training_logs["champion_run"]
    curves = champ["train_curves"]
    ckpts = champ["val_checkpoints"]
    n_ep = len(curves["episode_returns"])
    ep_arr = np.arange(1, n_ep + 1)
    window = max(5, n_ep // 40)
    legend_below = {"loc": "upper center", "bbox_to_anchor": (0.5, -0.16), "ncol": 1, "frameon": True, "fontsize": 7.8}

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=300)
    fig.suptitle(f"Champion {labels['D3QN']} training run", fontweight="bold")

    ax0 = axes[0]
    ax0_twin = ax0.twinx()
    l1 = ax0.plot(ep_arr, _rolling_mean(np.asarray(curves["episode_returns"]), window), color="#3b82f6",
                  linewidth=1.8, label=f"Train return ({window}-episode moving average)")
    l2 = ax0.plot([c["episode"] for c in ckpts], [c["mean_return"] for c in ckpts], color="#1d4ed8",
                  linestyle="--", marker="o", markersize=3.0, linewidth=1.8, label="Validation return (greedy)")
    l3 = ax0_twin.plot(ep_arr, curves["epsilons"], color="#dc2626", linestyle=":", linewidth=2.0, label="Exploration rate ε")
    ax0.set_title("(a) Exploration Decay & Episode Return")
    ax0.set_xlabel("Training Episode")
    ax0.set_ylabel("Episode Return", color="#1d4ed8")
    ax0_twin.set_ylabel("Exploration Rate (ε)", color="#dc2626")
    ax0_twin.set_ylim(0.0, 1.05)
    ax0.grid(True, linestyle=":", alpha=0.6)
    lines = l1 + l2 + l3
    ax0.legend(lines, [ln.get_label() for ln in lines], **legend_below)

    ax1 = axes[1]
    ax1_twin = ax1.twinx()
    q_line = ax1.plot(ep_arr, _rolling_mean(np.asarray(curves["mean_qs"]), window), color="#059669",
                      linewidth=2.0, label="Mean predicted Q(s, a) of sampled batches")
    loss_line = ax1_twin.plot(ep_arr, _rolling_mean(np.asarray(curves["td_losses"]), window), color="#d97706",
                              linestyle="--", linewidth=1.8, label="TD smooth-L1 (Huber) loss")
    ax1.set_title("(b) Value Estimates & TD Loss")
    ax1.set_xlabel("Training Episode")
    ax1.set_ylabel("Mean Predicted Q(s, a)", color="#059669")
    ax1_twin.set_ylabel("TD Huber Loss", color="#d97706")
    ax1.grid(True, linestyle=":", alpha=0.6)
    lines_b = q_line + loss_line
    ax1.legend(lines_b, [ln.get_label() for ln in lines_b], **legend_below)

    ax2 = axes[2]
    ax2_twin = ax2.twinx()
    max_entropy = float(np.log2(num_slots))
    e_line = ax2.plot(ep_arr, _rolling_mean(np.asarray(curves["action_entropies"]), window), color="#7c3aed",
                      linewidth=2.0, label="Action entropy (bits)")
    ref_line = [ax2.axhline(max_entropy, color="#6b7280", linestyle=":", linewidth=1.2,
                            label=f"Uniform over {num_slots} slots ({max_entropy:.2f} bits)")]
    u_line = ax2_twin.plot(ep_arr, _rolling_mean(np.asarray(curves["unique_actions"], dtype=np.float64), window),
                           color="#0284c7", linestyle="-.", linewidth=1.8, label=f"Distinct slots selected (of {num_slots})")
    ax2.set_title("(c) Policy Entropy & Slot Coverage")
    ax2.set_xlabel("Training Episode")
    ax2.set_ylabel("Action Entropy (bits, sliding window)", color="#7c3aed")
    ax2_twin.set_ylabel("Distinct Slots (sliding window)", color="#0284c7")
    ax2_twin.set_ylim(0, num_slots + 2)
    ax2.set_ylim(0, max_entropy * 1.08)
    ax2.grid(True, linestyle=":", alpha=0.6)
    lines_c = e_line + ref_line + u_line
    ax2.legend(lines_c, [ln.get_label() for ln in lines_c], **legend_below)
    _save(fig, save_path)


def plot_fig5_test_metrics_comparison(
    test_k10: Dict[str, Dict[str, float]],
    labels: Dict[str, str],
    num_test_users: int,
    horizon: int,
    save_path: Path,
) -> None:
    """Figure 5: test ranking metrics and the return-coverage-novelty trade-off (K=10)."""
    apply_academic_style()
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.4), dpi=300)
    metric_keys = [
        ("precision_at_10", "P@10"),
        ("recall_at_10", "R@10"),
        ("ndcg_at_10", "NDCG@10"),
        ("map_at_10", "MAP@10"),
        ("hr_at_10", "HR@10"),
    ]

    ax0 = axes[0]
    x = np.arange(len(metric_keys))
    width = 0.8 / len(POLICY_ORDER)
    offsets = (np.arange(len(POLICY_ORDER)) - (len(POLICY_ORDER) - 1) / 2.0) * width
    for name, off in zip(POLICY_ORDER, offsets):
        ax0.bar(x + off, [test_k10[name][m] for m, _ in metric_keys], width=width, color=POLICY_COLORS[name],
                edgecolor="#1e293b", linewidth=0.6, label=labels[name], alpha=0.9)
    ax0.set_xticks(x)
    ax0.set_xticklabels([lbl for _, lbl in metric_keys], fontweight="bold")
    top = max(test_k10[name][m] for name in POLICY_ORDER for m, _ in metric_keys)
    ax0.set_ylim(0.0, min(1.0, top) * 1.08 if top > 0 else 1.0)
    ax0.set_ylabel(f"Score on Test Split ({num_test_users} users, K=10)")
    ax0.set_title("(a) Top-10 Ranking Metrics on the Test Sessions")
    ax0.grid(axis="y", linestyle=":", alpha=0.6)
    ax0.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4, frameon=True, fontsize=8.0)

    # Panel (b): coverage vs. return, bubble area scaled by novelty; labels placed automatically
    ax1 = axes[1]
    cov = np.array([test_k10[n]["catalog_coverage"] * 100.0 for n in POLICY_ORDER])
    ret = np.array([test_k10[n]["mean_return"] for n in POLICY_ORDER])
    nov = np.array([test_k10[n]["mean_novelty_bits"] for n in POLICY_ORDER])
    nov_span = float(nov.max() - nov.min())
    sizes = 70.0 + 380.0 * ((nov - nov.min()) / nov_span if nov_span > 0 else np.full_like(nov, 0.5))
    for i, name in enumerate(POLICY_ORDER):
        ax1.scatter(cov[i], ret[i], s=sizes[i], color=POLICY_COLORS[name], edgecolors="#0f172a", linewidth=1.2,
                    alpha=0.9, zorder=4)

    x_lo, x_hi = _padded_limits(cov, 0.15, 0.15)
    x_lo = max(0.0, x_lo)
    data_fraction = 0.52  # the scatter uses the left half of the axes, the labels the right half
    ax1.set_xlim(x_lo, x_lo + (x_hi - x_lo) / data_fraction)
    ax1.set_xticks([t for t in ax1.get_xticks() if x_lo <= t <= x_hi])
    y_lo, y_hi = _padded_limits(ret, 0.15, 0.15)
    ax1.set_ylim(y_lo, y_hi)
    y_frac = (ret - y_lo) / (y_hi - y_lo)
    label_y = _spread(y_frac, min_gap=0.125, lo=0.08, hi=0.92)
    for i, name in enumerate(POLICY_ORDER):
        ax1.annotate(
            f"{labels[name]}\nRet {ret[i]:.2f} | Cov {cov[i]:.1f}% | Nov {nov[i]:.1f} b",
            xy=(cov[i], ret[i]),
            xycoords="data",
            xytext=(data_fraction + 0.04, label_y[i]),
            textcoords="axes fraction",
            va="center",
            ha="left",
            fontsize=7.6,
            fontweight="bold" if name == "D3QN" else "normal",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=POLICY_COLORS[name], alpha=0.9),
            arrowprops=dict(arrowstyle="-", color=POLICY_COLORS[name], lw=0.9, shrinkA=0, shrinkB=5),
            zorder=5,
        )
    ax1.set_xlabel("Catalog Coverage on Test Sessions (%)")
    ax1.set_ylabel(f"Mean Episode Return (H={horizon})")
    ax1.set_title("(b) Return–Coverage–Novelty Trade-off (bubble area ∝ novelty)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    _save(fig, save_path)


def plot_fig6_sample_trajectories(trajectories: List[Dict[str, Any]], labels: Dict[str, str], save_path: Path) -> None:
    """Figure 6: step-by-step sessions of the champion for fixed test users, with the gamma = 0 policy."""
    apply_academic_style()
    n_panels = len(trajectories)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.5, 4.6 * n_rows), dpi=300, squeeze=False)
    axes_flat = axes.flatten()
    max_len = max(max(t["episode_length"], t["myopic_bandit"]["episode_length"]) for t in trajectories)

    for idx, (ax, traj) in enumerate(zip(axes_flat, trajectories)):
        steps = traj["steps"]
        bandit = traj["myopic_bandit"]
        t_vals = [s["t"] for s in steps]
        cum = [s["cumulative_return"] for s in steps]
        rewards = [s["step_reward"] for s in steps]
        ax.bar(t_vals, rewards, width=0.55, color=["#10b981" if s["is_hit"] else "#ef4444" for s in steps],
               edgecolor="#064e3b", alpha=0.45)
        ax.plot(t_vals, cum, color="#1d4ed8", marker="o", markersize=5.0, linewidth=2.1)
        ax.plot([s["t"] for s in bandit["steps"]], [s["cumulative_return"] for s in bandit["steps"]],
                color=POLICY_COLORS["MyopicBandit"], marker="s", markersize=3.5, linewidth=1.6, linestyle="--")

        # Titles of the odd steps sit in two alternating rows of a band above the curves
        odd_steps = [s for s in steps if s["t"] % 2 == 1]
        for j, s in enumerate(odd_steps):
            ha = "left" if s["t"] == 1 else ("right" if s["t"] >= max_len - 1 else "center")
            rel_x = {"left": 0.0, "center": 0.5, "right": 1.0}[ha]  # vertical connector from x = t
            ax.annotate(
                f"t={s['t']} [a={s['action_slot']}] {s['title'][:20]}\n({s['true_rating']:.1f}★, {s['novelty_bits']:.1f} b)",
                xy=(s["t"], s["cumulative_return"]),
                xycoords="data",
                xytext=(s["t"], 0.94 if j % 2 == 0 else 0.81),
                textcoords=("data", "axes fraction"),
                ha=ha,
                va="center",
                fontsize=6.4,
                arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=0.6, shrinkA=0, shrinkB=3, relpos=(rel_x, 0.0)),
                bbox=dict(boxstyle="round,pad=0.16", fc="#f8fafc", ec="#94a3b8", alpha=0.9),
            )

        all_vals = rewards + cum + [s["cumulative_return"] for s in bandit["steps"]] + [0.0]
        lo, hi = float(min(all_vals)), float(max(all_vals))
        span = hi - lo if hi > lo else 1.0
        y_lo = lo - 0.08 * span
        ax.set_ylim(y_lo, y_lo + (hi + 0.04 * span - y_lo) / 0.70)  # keep the top 30% for the labels
        ax.set_xlim(0.4, max_len + 0.6)
        ax.set_xticks(range(1, max_len + 1))
        ax.set_title(
            f"({chr(ord('a') + idx)}) Test user #{traj['user_pool_idx']} (test position {traj['test_position']})\n"
            f"D3QN: {traj['total_hits']}/{traj['episode_length']} hits, return {traj['total_return']:+.2f} | "
            f"Myopic DQN: {bandit['total_hits']}/{bandit['episode_length']} hits, return {bandit['total_return']:+.2f}",
            fontsize=10,
        )
        ax.set_xlabel(f"Recommendation Step t (1..{max_len})")
        ax.set_ylabel("Reward / Cumulative Return")
        ax.grid(True, linestyle=":", alpha=0.6)

    for ax in axes_flat[n_panels:]:
        ax.axis("off")
    handles = [
        Patch(facecolor="#10b981", alpha=0.45, label="Step reward r_t, hit"),
        Patch(facecolor="#ef4444", alpha=0.45, label="Step reward r_t, miss"),
        Line2D([], [], color="#1d4ed8", marker="o", linewidth=2.1, label=f"{labels['D3QN']} cumulative return"),
        Line2D([], [], color=POLICY_COLORS["MyopicBandit"], marker="s", linestyle="--", linewidth=1.6,
               label=f"{labels['MyopicBandit']} cumulative return"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig7_sample_efficiency(
    training_logs: Dict[str, Any],
    validation_baselines: Dict[str, Dict[str, float]],
    crossings: Dict[str, Dict[str, Optional[Dict[str, Any]]]],
    labels: Dict[str, str],
    num_val_users: int,
    horizon: int,
    save_path: Path,
) -> None:
    """Figure 7: validation NDCG@10 and return vs. environment steps, champion and gamma = 0 runs."""
    apply_academic_style()
    champ = training_logs["champion_run"]["val_checkpoints"]
    bandit = training_logs["bandit_run"]["val_checkpoints"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4), dpi=300)
    panels = (
        (axes[0], "ndcg_at_10", "Validation NDCG@10", "(a) Ranking Sample Efficiency", "{:.3f}"),
        (axes[1], "mean_return", f"Mean Validation Return (H={horizon})", "(b) Return Sample Efficiency", "{:+.2f}"),
    )
    for ax, metric, ylabel, title, fmt in panels:
        ax.plot([c["steps"] for c in champ], [c[metric] for c in champ], color=POLICY_COLORS["D3QN"], marker="o",
                markersize=3.2, linewidth=2.2, label=f"{labels['D3QN']} validation curve", zorder=4)
        ax.plot([c["steps"] for c in bandit], [c[metric] for c in bandit], color=POLICY_COLORS["MyopicBandit"],
                marker="s", markersize=2.8, linewidth=1.6, linestyle="--", label=f"{labels['MyopicBandit']} validation curve",
                zorder=3)
        plotted = [c[metric] for c in champ] + [c[metric] for c in bandit]
        for name in VALIDATION_BASELINES:
            value = float(validation_baselines[name][metric])
            plotted.append(value)
            cross = crossings[name][metric]
            note = f"D3QN passes at {cross['steps']:,} steps" if cross else "never surpassed by D3QN"
            ref_name = f"{labels[name]} saved checkpoint" if name == "MyopicBandit" else labels[name]
            ax.axhline(value, color=POLICY_COLORS[name], linestyle=BASELINE_LINESTYLES[name], linewidth=1.3,
                       label=f"{ref_name}: {fmt.format(value)} ({note})")
            if cross is not None:
                ax.scatter([cross["steps"]], [cross["value"]], color=POLICY_COLORS[name], edgecolors="#111827",
                           s=48, zorder=5)
        ax.set_ylim(*_padded_limits(plotted, 0.08, 0.08))
        ax.set_xlim(left=0)
        ax.set_title(f"{title} on Validation ({num_val_users} users)")
        ax.set_xlabel("Cumulative Training Environment Steps")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, frameon=True, fontsize=7.6)
    _save(fig, save_path)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def _print_table(title: str, table: Dict[str, Dict[str, float]]) -> None:
    print(f"[evaluate] {title}")
    print(f"  {'policy':18s} {'return':>16s} {'P@K':>7s} {'NDCG@K':>7s} {'MAP@K':>7s} {'HR@K':>7s} {'cov%':>7s} {'nov':>7s}")
    for name, m in table.items():
        print(
            f"  {name:18s} {m['mean_return']:+8.4f} +/- {m['std_return']:6.4f} {m['precision_at_k']:7.4f} "
            f"{m['ndcg_at_k']:7.4f} {m['map_at_k']:7.4f} {m['hr_at_k']:7.4f} "
            f"{100.0 * m['catalog_coverage']:7.2f} {m['mean_novelty_bits']:7.3f}"
        )


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] Workshop 3 evaluation on device={device}")

    training_logs = _read_json(TRAINING_LOGS_PATH)
    with np.load(CACHE_PATH) as npz:
        cache_data = {key: npz[key] for key in npz.files}
    # Workshop 2 HybridNeuMF scores per candidate slot (checked against the cache's slots and users)
    w2 = load_w2_candidate_scores(DEFAULT_SCORES_PATH, cache_data=cache_data)

    champion, champion_info = load_greedy_agent(CHAMPION_CHECKPOINT, device)
    bandit, bandit_info = load_greedy_agent(BANDIT_CHECKPOINT, device)
    if bandit_info["gamma"] != 0.0:
        raise ValueError(f"{bandit_info['checkpoint']} has gamma={bandit_info['gamma']}, expected the gamma = 0 ablation")
    labels = policy_labels(champion_info["gamma"], bandit_info["gamma"])
    make_policy = PolicyFactory(cache_data, w2, champion, bandit)

    test_envs = {k: MovieLensRecEnv(cache_path=str(CACHE_PATH), split="test", horizon=k, seed=SEED) for k in TEST_HORIZONS}
    val_env = MovieLensRecEnv(cache_path=str(CACHE_PATH), split="val", horizon=10, seed=SEED)
    test_users = [int(u) for u in test_envs[10].split_user_indices]
    val_users = [int(u) for u in val_env.split_user_indices]
    print(f"[evaluate] {len(test_users)} test users, {len(val_users)} validation users")

    # Test split: every policy on every test user exactly once, at K=10 and K=20
    results: Dict[int, Dict[str, Dict[str, float]]] = {}
    per_user_k10: Dict[str, List[Dict[str, Any]]] = {}
    for k in TEST_HORIZONS:
        results[k] = {}
        for name in POLICY_ORDER:
            metrics, per_episode = evaluate_policy_at_k(test_envs[k], make_policy(name), k, test_users)
            results[k][name] = metrics
            if k == 10:
                per_user_k10[name] = per_episode
        _print_table(f"Test K={k} ({len(test_users)} users)", results[k])
    test_k10, test_k20 = results[10], results[20]
    paired = build_paired_comparisons(per_user_k10)

    # Validation split: baselines on the users of the training-time validation curves
    validation_baselines: Dict[str, Dict[str, float]] = {}
    for name in VALIDATION_BASELINES:
        validation_baselines[name], _ = evaluate_policy_at_k(val_env, make_policy(name), 10, val_users)
    _print_table(f"Validation K=10 ({len(val_users)} users)", validation_baselines)

    # The saved checkpoints must reproduce the validation return logged when they were saved
    checkpoint_check: Dict[str, Any] = {}
    for name, run_key in (("D3QN", "champion_run"), ("MyopicBandit", "bandit_run")):
        evaluated = validation_baselines[name] if name in validation_baselines else evaluate_policy_at_k(
            val_env, make_policy(name), 10, val_users
        )[0]
        run = training_logs[run_key]
        checkpoint_check[name] = {
            "evaluated_val_mean_return": evaluated["mean_return"],
            "logged_best_val_return": run["best_val_return"],
            "logged_best_val_episode": run["best_val_episode"],
            "matches_log": bool(abs(float(evaluated["mean_return"]) - float(run["best_val_return"])) <= 1e-3),
        }
    print(f"[evaluate] Checkpoint check: {checkpoint_check}")
    sample_efficiency, crossings = build_sample_efficiency(training_logs, validation_baselines, checkpoint_check, len(val_users))

    positions = trajectory_positions(len(test_users))
    trajectories = extract_sample_trajectories(test_envs[10], champion, bandit, test_users, positions, cache_data)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_paths = {
        "fig3": FIG_DIR / "fig3_hyperparameter_learning_curves.png",
        "fig4": FIG_DIR / "fig4_exploration_dynamics.png",
        "fig5": FIG_DIR / "fig5_test_metrics_comparison.png",
        "fig6": FIG_DIR / "fig6_sample_trajectories.png",
        "fig7": FIG_DIR / "fig7_sample_efficiency.png",
    }
    plot_fig3_hyperparameter_curves(training_logs, labels, len(val_users), val_env.horizon, fig_paths["fig3"])
    plot_fig4_exploration_dynamics(training_logs, labels, int(test_envs[10].num_candidates), fig_paths["fig4"])
    plot_fig5_test_metrics_comparison(test_k10, labels, len(test_users), test_envs[10].horizon, fig_paths["fig5"])
    plot_fig6_sample_trajectories(trajectories, labels, fig_paths["fig6"])
    plot_fig7_sample_efficiency(training_logs, validation_baselines, crossings, labels, len(val_users),
                                val_env.horizon, fig_paths["fig7"])

    summary_payload = {
        "evaluation_split": "test",
        "num_test_users": len(test_users),
        "num_validation_users": len(val_users),
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "device": str(device),
        "eval_wall_time_sec": None,
        "agents": {"D3QN": champion_info, "MyopicBandit": bandit_info},
        "test_k10": test_k10,
        "test_k20": test_k20,
        "paired_comparisons": paired,
        "validation_baselines": validation_baselines,
        "workshop1_comparison": build_workshop1_comparison(test_k10),
        "workshop1_reference": WORKSHOP1_REFERENCE,
        "workshop2_comparison": build_workshop2_comparison(test_k10, w2),
        "hyperparameter_table": training_logs["hyperparameter_table"],
        "sample_efficiency": sample_efficiency,
        "trajectory_selection": (
            f"Test users at the evenly spaced positions {positions} of the {len(test_users)} RL test users "
            "(test_user_indices order), fixed before evaluation and independent of any outcome; the "
            "champion (top level) and the gamma = 0 policy (myopic_bandit) run the same sessions."
        ),
        "sample_trajectories": trajectories,
        "figures_generated": [repo_relative(p) for p in fig_paths.values()],
    }
    summary_payload["eval_wall_time_sec"] = round(time.time() - t0, 2)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    print(f"[evaluate] Paired comparisons (K=10): {json.dumps({k: v for k, v in paired.items() if '_vs_' in k})}")
    print(f"[evaluate] Sample efficiency: { {k: v for k, v in sample_efficiency.items() if k.startswith('steps_to')} }")
    print(f"[evaluate] Saved {repo_relative(SUMMARY_PATH)} and Figures 3-7 in {repo_relative(FIG_DIR)}/")


if __name__ == "__main__":
    main()
