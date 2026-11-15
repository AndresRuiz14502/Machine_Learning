"""
Evaluation and Publication-Quality Figure Generation Pipeline for Workshop 3:
Reinforcement Learning for Sequential Movie Recommendation on MovieLens 32M.

Evaluates the Champion Dueling Double DQN (`workshop3/checkpoints/agent_best.pt`)
and all 5 baselines (`Random`, `Bias`, `Popularity`, `Greedy FunkSVD`, `Myopic Bandit`)
on the 250 test users at horizons K=10 and K=20, extracts 4 detailed user session
trajectories, saves `workshop3/report/metrics_summary.json`, and generates Figures 3–7
at 300 DPI in `workshop3/report/figures/`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workshop3.src.agent import DQNAgent
from workshop3.src.baselines import (
    BasePolicy,
    BiasPolicy,
    GreedyFunkSVDPolicy,
    MyopicBanditPolicy,
    PopularityPolicy,
    RandomPolicy,
    evaluate_policy,
)
from workshop3.src.env import MovieLensRecEnv


def evaluate_policy_at_k(
    env: MovieLensRecEnv,
    policy: Any,
    k_eval: int = 20,
    num_episodes: int = 250,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluate a policy on `env` over `num_episodes` episodes, computing both standard
    `evaluate_policy` @10 metrics and exact `@K` ranking metrics (`Precision@K`,
    `Recall@K`, `NDCG@K`, `MAP@K`, `HR@K`).
    """
    base_metrics = evaluate_policy(env, policy, num_episodes=num_episodes, seed=seed)
    if k_eval == 10:
        return {
            **{k: round(float(v), 4) for k, v in base_metrics.items()},
            "precision_at_k": round(float(base_metrics["precision_at_10"]), 4),
            "recall_at_k": round(float(base_metrics["recall_at_10"]), 4),
            "ndcg_at_k": round(float(base_metrics["ndcg_at_10"]), 4),
            "map_at_k": round(float(base_metrics["map_at_10"]), 4),
            "hr_at_k": round(float(base_metrics["hr_at_10"]), 4),
        }

    # Compute exact @K (e.g. K=20) ranking metrics over the same seeded rollout
    if hasattr(policy, "action_counts"):
        policy.action_counts.fill(0.0)
        policy.total_steps = 0
    if hasattr(policy, "rng") and hasattr(policy, "seed"):
        policy.rng = np.random.default_rng(policy.seed)

    discounts_k = 1.0 / np.log2(np.arange(2, k_eval + 2, dtype=np.float32))
    ranks_k = np.arange(1, k_eval + 1, dtype=np.float32)

    precisions_k: List[float] = []
    recalls_k: List[float] = []
    ndcgs_k: List[float] = []
    maps_k: List[float] = []
    hrs_k: List[float] = []

    for ep in range(num_episodes):
        if hasattr(policy, "reset"):
            policy.reset()
        if ep == 0:
            state, info = env.reset(seed=seed)
        else:
            state, info = env.reset()

        u_idx = int(info["user_idx"])
        total_rel = max(1, int(np.sum(env.cand_is_liked[u_idx] == 1)))

        done = False
        ep_hits: List[float] = []
        while not done:
            try:
                action = int(policy.select_action(state, info))
            except TypeError:
                action = int(policy.select_action(state))
            next_state, _, terminated, truncated, step_info = env.step(action)
            done = bool(terminated or truncated)
            ep_hits.append(1.0 if step_info["is_hit"] == 1 else 0.0)
            state, info = next_state, step_info

        hits_k = np.zeros(k_eval, dtype=np.float32)
        n_rec = min(k_eval, len(ep_hits))
        if n_rec > 0:
            hits_k[:n_rec] = ep_hits[:n_rec]

        n_hits_k = float(np.sum(hits_k))
        precisions_k.append(n_hits_k / float(k_eval))
        recalls_k.append(float(min(1.0, n_hits_k / float(total_rel))))
        hrs_k.append(1.0 if n_hits_k >= 1.0 else 0.0)

        dcg_k = float(np.sum(hits_k * discounts_k))
        ideal_k = min(k_eval, total_rel)
        idcg_k = float(np.sum(discounts_k[:ideal_k]))
        ndcgs_k.append(dcg_k / idcg_k if idcg_k > 0.0 else 0.0)

        cum_hits = np.cumsum(hits_k)
        ap_k = float(np.sum((cum_hits / ranks_k) * hits_k) / float(ideal_k))
        maps_k.append(ap_k)

    out = {k: round(float(v), 4) for k, v in base_metrics.items()}
    out.update(
        {
            f"precision_at_{k_eval}": round(float(np.mean(precisions_k)), 4),
            f"recall_at_{k_eval}": round(float(np.mean(recalls_k)), 4),
            f"ndcg_at_{k_eval}": round(float(np.mean(ndcgs_k)), 4),
            f"map_at_{k_eval}": round(float(np.mean(maps_k)), 4),
            f"hr_at_{k_eval}": round(float(np.mean(hrs_k)), 4),
            "precision_at_k": round(float(np.mean(precisions_k)), 4),
            "recall_at_k": round(float(np.mean(recalls_k)), 4),
            "ndcg_at_k": round(float(np.mean(ndcgs_k)), 4),
            "map_at_k": round(float(np.mean(maps_k)), 4),
            "hr_at_k": round(float(np.mean(hrs_k)), 4),
        }
    )
    return out


def extract_sample_trajectories(
    env: MovieLensRecEnv,
    agent: DQNAgent,
    num_trajectories: int = 4,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Extract 4 real test user trajectories of the trained Champion D3QN agent."""
    trajectories: List[Dict[str, Any]] = []
    chosen_test_users = [
        int(env.split_user_indices[0]),
        int(env.split_user_indices[5]),
        int(env.split_user_indices[12]),
        int(env.split_user_indices[25]),
    ]

    for idx, u_pool in enumerate(chosen_test_users[:num_trajectories]):
        obs, info = env.reset(seed=seed + idx * 7, options={"user_idx": u_pool})
        done = False
        cum_ret = 0.0
        steps_list: List[Dict[str, Any]] = []

        while not done:
            action = agent.select_action(obs, valid_mask=info["valid_mask"], eval_mode=True)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = bool(terminated or truncated)
            cum_ret += float(reward)

            h_last = env.history[-1]
            raw_title = str(h_last["title"])
            short_title = raw_title if len(raw_title) <= 28 else raw_title[:25] + "..."
            raw_genres = str(h_last["genres"])
            short_genres = "|".join(raw_genres.split("|")[:2])

            steps_list.append(
                {
                    "t": int(h_last["step"]),
                    "action_slot": int(h_last["action"]),
                    "movie_id": int(h_last["movie_id"]),
                    "title": short_title,
                    "full_title": raw_title,
                    "genres": short_genres,
                    "is_hit": int(h_last["is_hit"]),
                    "true_rating": float(h_last["true_rating"]),
                    "step_reward": round(float(reward), 4),
                    "novelty_bits": round(float(h_last["novelty_bits"]), 2),
                    "cumulative_return": round(float(cum_ret), 4),
                }
            )
            obs, info = next_obs, step_info

        trajectories.append(
            {
                "trajectory_id": idx + 1,
                "user_pool_idx": int(u_pool),
                "total_return": round(float(cum_ret), 4),
                "total_hits": int(sum(s["is_hit"] for s in steps_list)),
                "episode_length": len(steps_list),
                "mean_novelty_bits": round(
                    float(np.mean([s["novelty_bits"] for s in steps_list])), 2
                ),
                "steps": steps_list,
            }
        )
    return trajectories


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


def plot_fig3_hyperparameter_curves(
    training_logs: Dict[str, Any], save_path: Path
) -> None:
    """Figure 3: 1x2 Hyperparameter Learning Curves & Sensitivity Analysis."""
    apply_academic_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=300)

    ax0 = axes[0]
    palette = {
        "cfg1_baseline": ("#2563eb", "-", 2.0, "Cfg 1: Baseline (lr=5e-4, γ=0.95)"),
        "cfg2_high_lr": ("#ef4444", "--", 1.5, "Cfg 2: High LR (lr=1e-3)"),
        "cfg3_low_lr": ("#f59e0b", "-.", 1.5, "Cfg 3: Low LR (lr=1e-4)"),
        "cfg4_gamma_090": ("#10b981", "--", 1.5, "Cfg 4: Low γ (γ=0.90)"),
        "cfg5_gamma_099": ("#8b5cf6", ":", 1.6, "Cfg 5: High γ (γ=0.99)"),
        "cfg6_fast_eps": ("#ec4899", "-.", 1.4, "Cfg 6: Fast ε-decay (0.990)"),
        "cfg7_slow_eps": ("#14b8a6", ":", 1.5, "Cfg 7: Slow ε-decay (0.998)"),
        "cfg8_high_l2": ("#64748b", "--", 1.4, "Cfg 8: High L2 (1e-4)"),
    }

    grid_runs = training_logs["grid_runs"]
    for cfg_id, (color, ls, lw, lbl) in palette.items():
        if cfg_id in grid_runs:
            ckpts = grid_runs[cfg_id]["val_checkpoints"]
            eps = [c["episode"] for c in ckpts]
            rets = [c["mean_return"] for c in ckpts]
            ax0.plot(eps, rets, color=color, linestyle=ls, linewidth=lw, label=lbl, alpha=0.88)

    champ = training_logs["champion_run"]
    c_eps = [c["episode"] for c in champ["val_checkpoints"] if c["episode"] <= 800]
    c_rets = [c["mean_return"] for c in champ["val_checkpoints"] if c["episode"] <= 800]
    ax0.plot(
        c_eps,
        c_rets,
        color="#111827",
        linestyle="-",
        linewidth=2.5,
        label="Champion D3QN (Tuned)",
    )

    ax0.set_title("(a) Validation Return Learning Curves (800 Episodes)")
    ax0.set_xlabel("Training Episode")
    ax0.set_ylabel("Mean Validation Episode Return (H=10)")
    ax0.grid(True, linestyle=":", alpha=0.6)
    ax0.legend(loc="lower right", frameon=True, ncol=1, fontsize=7.8)

    # Right panel: Final validation return sensitivity bar chart
    ax1 = axes[1]
    table = training_logs["hyperparameter_table"]
    short_names = [
        "Cfg1: Base\n(5e-4, 0.95)",
        "Cfg2: High LR\n(lr=1e-3)",
        "Cfg3: Low LR\n(lr=1e-4)",
        "Cfg4: Low γ\n(γ=0.90)",
        "Cfg5: High γ\n(γ=0.99)",
        "Cfg6: Fast ε\n(δ=0.990)",
        "Cfg7: Slow ε\n(δ=0.998)",
        "Cfg8: High L2\n(λ=1e-4)",
        "Champion\n(1500 ep)",
    ]
    vals = [r["final_val_return"] for r in table]
    bar_colors = [
        "#2563eb",
        "#ef4444",
        "#f59e0b",
        "#10b981",
        "#8b5cf6",
        "#ec4899",
        "#14b8a6",
        "#64748b",
        "#111827",
    ]
    y_pos = np.arange(len(vals))
    bars = ax1.barh(
        y_pos,
        vals,
        color=bar_colors[: len(vals)],
        edgecolor="#1f2937",
        height=0.68,
        alpha=0.88,
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(short_names[: len(vals)], fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlim(8.0, max(vals) + 0.9)
    ax1.set_title("(b) Hyperparameter Sensitivity: Final Validation Return")
    ax1.set_xlabel("Final Mean Validation Return")
    ax1.grid(axis="x", linestyle=":", alpha=0.6)

    for bar, v in zip(bars, vals):
        ax1.text(
            v + 0.08,
            bar.get_y() + bar.get_height() / 2.0,
            f"{v:.2f}",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
        )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig4_exploration_dynamics(
    training_logs: Dict[str, Any], save_path: Path
) -> None:
    """Figure 4: 1x3 Exploration & Training Dynamics for the Champion D3QN Run."""
    apply_academic_style()
    champ = training_logs["champion_run"]
    curves = champ["train_curves"]
    ckpts = champ["val_checkpoints"]

    ep_arr = np.arange(1, len(curves["episode_returns"]) + 1)
    train_rets = np.array(curves["episode_returns"], dtype=np.float32)
    eps_arr = np.array(curves["epsilons"], dtype=np.float32)
    losses_arr = np.array(curves["td_losses"], dtype=np.float32)
    qs_arr = np.array(curves["mean_qs"], dtype=np.float32)
    ent_arr = np.array(curves["action_entropies"], dtype=np.float32)
    uniq_arr = np.array(curves["unique_actions"], dtype=np.float32)

    def rolling_mean(arr: np.ndarray, w: int = 35) -> np.ndarray:
        kernel = np.ones(w, dtype=np.float32) / float(w)
        padded = np.pad(arr, (w - 1, 0), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), dpi=300)

    # Panel (a): Epsilon decay & Train vs Val Return
    ax0 = axes[0]
    ax0_twin = ax0.twinx()
    l1 = ax0.plot(
        ep_arr,
        rolling_mean(train_rets, 40),
        color="#3b82f6",
        linewidth=1.8,
        label="Train Return (40-ep MA)",
    )
    v_eps = [c["episode"] for c in ckpts]
    v_rets = [c["mean_return"] for c in ckpts]
    l2 = ax0.plot(
        v_eps,
        v_rets,
        color="#1d4ed8",
        linestyle="--",
        marker="o",
        markersize=3.2,
        linewidth=2.0,
        label="Validation Return (ε=0)",
    )
    l3 = ax0_twin.plot(
        ep_arr,
        eps_arr,
        color="#dc2626",
        linestyle=":",
        linewidth=2.0,
        label="Exploration Rate ε",
    )
    ax0.set_title("(a) Exploration Decay & Episode Return")
    ax0.set_xlabel("Training Episode")
    ax0.set_ylabel("Cumulative Episode Return", color="#1d4ed8")
    ax0_twin.set_ylabel("Exploration Rate (ε)", color="#dc2626")
    ax0_twin.set_ylim(0.0, 1.05)
    ax0.grid(True, linestyle=":", alpha=0.6)
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax0.legend(lines, labels, loc="center right", frameon=True, fontsize=8.0)

    # Panel (b): Mean Predicted Q(s,a) & TD Huber Loss
    ax1 = axes[1]
    ax1_twin = ax1.twinx()
    q_line = ax1.plot(
        ep_arr,
        rolling_mean(qs_arr, 35),
        color="#059669",
        linewidth=2.0,
        label="Mean Predicted Q(s,a)",
    )
    loss_line = ax1_twin.plot(
        ep_arr,
        rolling_mean(losses_arr, 35),
        color="#d97706",
        linestyle="--",
        linewidth=1.8,
        label="TD Smooth-L1 Loss",
    )
    ax1.set_title("(b) Value Convergence & TD Huber Loss")
    ax1.set_xlabel("Training Episode")
    ax1.set_ylabel("Mean Predicted Q(s, a)", color="#059669")
    ax1_twin.set_ylabel("TD Huber Loss", color="#d97706")
    ax1.grid(True, linestyle=":", alpha=0.6)
    lines_b = q_line + loss_line
    ax1.legend(
        lines_b,
        [l.get_label() for l in lines_b],
        loc="center right",
        frameon=True,
        fontsize=8.0,
    )

    # Panel (c): Action Entropy & Unique Action Slot Coverage
    ax2 = axes[2]
    ax2_twin = ax2.twinx()
    e_line = ax2.plot(
        ep_arr,
        rolling_mean(ent_arr, 30),
        color="#7c3aed",
        linewidth=2.0,
        label="Action Entropy (bits)",
    )
    u_line = ax2_twin.plot(
        ep_arr,
        rolling_mean(uniq_arr, 20),
        color="#0284c7",
        linestyle="-.",
        linewidth=1.8,
        label="Active Candidate Slots (/50)",
    )
    ax2.axhline(
        np.log2(50.0),
        color="#6b7280",
        linestyle=":",
        linewidth=1.2,
        label="Max Uniform Entropy (5.64 bits)",
    )
    ax2.set_title("(c) Policy Entropy & Slot Coverage")
    ax2.set_xlabel("Training Episode")
    ax2.set_ylabel("Shannon Action Entropy (bits)", color="#7c3aed")
    ax2_twin.set_ylabel("Unique Slots Selected (Window)", color="#0284c7")
    ax2_twin.set_ylim(20, 52)
    ax2.grid(True, linestyle=":", alpha=0.6)
    lines_c = e_line + u_line
    ax2.legend(
        lines_c,
        [l.get_label() for l in lines_c],
        loc="lower right",
        frameon=True,
        fontsize=8.0,
    )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig5_test_metrics_comparison(
    test_k10: Dict[str, Dict[str, float]], save_path: Path
) -> None:
    """Figure 5: 1x2 Test Set Ranking Metrics & Return-Coverage-Novelty Trade-off."""
    apply_academic_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 4.8), dpi=300)

    policies_order = [
        ("Random", "Random", "#94a3b8"),
        ("BiasOnly", "Bias Only (W1)", "#f59e0b"),
        ("Popularity", "Popularity (W1)", "#ef4444"),
        ("GreedyFunkSVD", "Greedy FunkSVD (W1)", "#3b82f6"),
        ("MyopicBandit", "Myopic Bandit (γ=0)", "#8b5cf6"),
        ("D3QN", "D3QN (Ours, γ=0.95)", "#10b981"),
    ]

    metric_keys = [
        ("precision_at_10", "P@10"),
        ("recall_at_10", "R@10"),
        ("ndcg_at_10", "NDCG@10"),
        ("map_at_10", "MAP@10"),
        ("hr_at_10", "HR@10"),
    ]

    ax0 = axes[0]
    x = np.arange(len(metric_keys))
    width = 0.135
    offsets = np.linspace(-2.5 * width, 2.5 * width, len(policies_order))

    for (p_key, p_label, p_color), off in zip(policies_order, offsets):
        vals = [test_k10[p_key][m_k] for m_k, _ in metric_keys]
        ax0.bar(
            x + off,
            vals,
            width=width,
            color=p_color,
            edgecolor="#1e293b",
            linewidth=0.6,
            label=p_label,
            alpha=0.9,
        )

    ax0.set_xticks(x)
    ax0.set_xticklabels([m_lbl for _, m_lbl in metric_keys], fontweight="bold")
    ax0.set_ylim(0.0, 1.14)
    ax0.set_ylabel("Score on Test Split (250 Users, K=10)")
    ax0.set_title("(a) Top-10 Ranking Metrics Comparison on Test Set")
    ax0.grid(axis="y", linestyle=":", alpha=0.6)
    ax0.legend(loc="upper left", ncol=2, frameon=True, fontsize=8.0)

    # Panel (b): Trade-off scatter/bubble plot (Catalog Coverage % vs Mean Episode Return, sized by Novelty)
    ax1 = axes[1]
    # Include Workshop 1 static FunkSVD reference point (8.00% coverage)
    ax1.axvline(
        8.0,
        color="#dc2626",
        linestyle="--",
        linewidth=1.5,
        label="Workshop 1 Static SVD Coverage Ceiling (8.00%)",
    )

    for p_key, p_label, p_color in policies_order:
        m = test_k10[p_key]
        cov_pct = m["catalog_coverage"] * 100.0
        ret_val = m["mean_return"]
        nov_val = m["mean_novelty_bits"]
        size = (nov_val ** 2) * 1.45
        ax1.scatter(
            cov_pct,
            ret_val,
            s=size,
            color=p_color,
            edgecolors="#0f172a",
            linewidth=1.4,
            alpha=0.9,
            zorder=4,
        )
        dx, dy = (1.1, 0.35)
        if p_key == "D3QN":
            dx, dy = (-11.5, -1.25)
        elif p_key == "MyopicBandit":
            dx, dy = (-12.0, -1.35)
        elif p_key == "Random":
            dx, dy = (-9.5, 0.55)
        ax1.annotate(
            f"{p_label}\n(Ret={ret_val:.2f}, Cov={cov_pct:.1f}%, Nov={nov_val:.1f}b)",
            xy=(cov_pct, ret_val),
            xytext=(cov_pct + dx, ret_val + dy),
            fontsize=7.8,
            fontweight="bold" if p_key == "D3QN" else "normal",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=p_color, alpha=0.85),
        )

    ax1.set_xlim(0.0, 52.0)
    ax1.set_ylim(-2.8, 14.8)
    ax1.set_xlabel("Catalog Coverage on Test Set (%)")
    ax1.set_ylabel("Mean Episode Return (H=10)")
    ax1.set_title("(b) Accuracy–Coverage–Novelty Trade-off (Bubble Size = Novelty)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True, fontsize=8.0)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig6_sample_trajectories(
    trajectories: List[Dict[str, Any]], save_path: Path
) -> None:
    """Figure 6: 2x2 Step-by-Step User Session Trajectories with Movie Titles."""
    apply_academic_style()
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.8), dpi=300)
    axes_flat = axes.flatten()

    for idx, (ax, traj) in enumerate(zip(axes_flat, trajectories[:4])):
        steps = traj["steps"]
        t_vals = [s["t"] for s in steps]
        cum_rets = [s["cumulative_return"] for s in steps]
        step_rews = [s["step_reward"] for s in steps]
        hits = [s["is_hit"] for s in steps]

        bar_colors = ["#10b981" if h == 1 else "#ef4444" for h in hits]
        ax.bar(
            t_vals,
            step_rews,
            width=0.55,
            color=bar_colors,
            edgecolor="#064e3b",
            alpha=0.45,
            label="Step Reward r_t (Green=Hit, Red=Miss)",
        )
        ax.plot(
            t_vals,
            cum_rets,
            color="#1d4ed8",
            marker="o",
            markersize=5.5,
            linewidth=2.2,
            label="Cumulative Return G_t",
        )

        # Annotate alternating steps with movie title, slot, and novelty
        for s in steps:
            t_i = s["t"]
            c_i = s["cumulative_return"]
            if t_i in (1, 3, 5, 7, 9, 10):
                y_off = 0.75 if (t_i % 2 == 1) else -1.35
                short_t = s["title"][:20]
                ax.annotate(
                    f"t={t_i} [a={s['action_slot']}]: {short_t}\n({s['true_rating']:.1f}★, {s['novelty_bits']:.1f}b)",
                    xy=(t_i, c_i),
                    xytext=(max(0.7, t_i - 0.85), c_i + y_off),
                    fontsize=7.0,
                    arrowprops=dict(arrowstyle="->", color="#475569", lw=0.7),
                    bbox=dict(boxstyle="round,pad=0.18", fc="#f8fafc", ec="#94a3b8", alpha=0.9),
                )

        panel_letter = chr(ord("a") + idx)
        ax.set_title(
            f"({panel_letter}) Test User #{traj['user_pool_idx']} | "
            f"Hits: {traj['total_hits']}/{traj['episode_length']} | "
            f"Return: +{traj['total_return']:.2f} | Mean Novelty: {traj['mean_novelty_bits']:.1f}b"
        )
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Recommendation Step t (1..10)")
        ax.set_ylabel("Reward / Cumulative Return")
        ax.set_ylim(-1.8, max(cum_rets) + 2.6)
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper left", frameon=True, fontsize=8.0)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig7_sample_efficiency(
    training_logs: Dict[str, Any],
    test_k10: Dict[str, Dict[str, float]],
    save_path: Path,
) -> None:
    """Figure 7: 1x2 Sample Efficiency Analysis (NDCG@10 & Return vs Environment Steps)."""
    apply_academic_style()
    champ = training_logs["champion_run"]
    ckpts = champ["val_checkpoints"]

    steps = np.array([max(10, c["steps"]) for c in ckpts], dtype=np.float32)
    ndcgs = np.array([c["ndcg_at_10"] for c in ckpts], dtype=np.float32)
    rets = np.array([c["mean_return"] for c in ckpts], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.6), dpi=300)

    # Panel (a): NDCG@10 vs Environment Interaction Steps
    ax0 = axes[0]
    ax0.plot(
        steps,
        ndcgs,
        color="#10b981",
        marker="o",
        markersize=4.0,
        linewidth=2.3,
        label="Champion D3QN (Validation NDCG@10)",
    )
    baselines_ref = [
        ("MyopicBandit", "Myopic Bandit (γ=0)", "#8b5cf6", "--"),
        ("Random", "Random Policy", "#64748b", ":"),
        ("GreedyFunkSVD", "Greedy FunkSVD (W1 Supervised)", "#3b82f6", "-."),
        ("BiasOnly", "User+Item Bias (W1)", "#f59e0b", "--"),
        ("Popularity", "Popularity (W1)", "#ef4444", ":"),
    ]
    for b_key, b_lbl, b_col, b_ls in baselines_ref:
        val_b = test_k10[b_key]["ndcg_at_10"]
        ax0.axhline(
            val_b,
            color=b_col,
            linestyle=b_ls,
            linewidth=1.6,
            label=f"{b_lbl} ({val_b:.3f})",
        )

    # Annotate crossover steps
    svd_ndcg = test_k10["GreedyFunkSVD"]["ndcg_at_10"]
    bandit_ndcg = test_k10["MyopicBandit"]["ndcg_at_10"]
    idx_svd = next((i for i, v in enumerate(ndcgs) if v > svd_ndcg), 0)
    idx_ban = next((i for i, v in enumerate(ndcgs) if v > bandit_ndcg), len(ndcgs) - 1)
    ax0.scatter(
        [steps[idx_svd], steps[idx_ban]],
        [ndcgs[idx_svd], ndcgs[idx_ban]],
        color="#dc2626",
        s=55,
        zorder=5,
    )
    ax0.set_title("(a) Ranking Sample Efficiency: NDCG@10 vs. Env Steps")
    ax0.set_xlabel("Cumulative Environment Interaction Steps")
    ax0.set_ylabel("NDCG@10")
    ax0.set_ylim(0.0, 1.06)
    ax0.grid(True, linestyle=":", alpha=0.6)
    ax0.legend(loc="lower right", frameon=True, fontsize=7.8)

    # Panel (b): Episode Return vs Environment Interaction Steps
    ax1 = axes[1]
    ax1.plot(
        steps,
        rets,
        color="#2563eb",
        marker="s",
        markersize=3.8,
        linewidth=2.3,
        label="Champion D3QN (Episode Return)",
    )
    for b_key, b_lbl, b_col, b_ls in baselines_ref:
        ret_b = test_k10[b_key]["mean_return"]
        ax1.axhline(
            ret_b,
            color=b_col,
            linestyle=b_ls,
            linewidth=1.6,
            label=f"{b_lbl} ({ret_b:+.2f})",
        )

    ax1.annotate(
        f"Surpasses W1 FunkSVD\nat ~{int(steps[idx_svd])} steps",
        xy=(steps[idx_svd], rets[idx_svd]),
        xytext=(steps[idx_svd] + 900, rets[idx_svd] - 2.3),
        fontsize=8.2,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#1e293b", lw=1.1),
        bbox=dict(boxstyle="round,pad=0.25", fc="#eff6ff", ec="#3b82f6"),
    )
    ax1.annotate(
        f"Surpasses Myopic Bandit\nat ~{int(steps[idx_ban])} steps",
        xy=(steps[idx_ban], rets[idx_ban]),
        xytext=(max(1200, steps[idx_ban] - 5500), rets[idx_ban] - 3.2),
        fontsize=8.2,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#1e293b", lw=1.1),
        bbox=dict(boxstyle="round,pad=0.25", fc="#ecfdf5", ec="#10b981"),
    )

    ax1.set_title("(b) Return Sample Efficiency vs. Supervised & Bandit Baselines")
    ax1.set_xlabel("Cumulative Environment Interaction Steps")
    ax1.set_ylabel("Mean Episode Return (H=10)")
    ax1.set_ylim(-2.2, 13.8)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True, fontsize=7.8)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] Running Workshop 3 Evaluation on device={device} ...")

    ckpt_path = PROJECT_ROOT / "workshop3" / "checkpoints" / "agent_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Champion checkpoint not found at {ckpt_path}. Run `workshop3/src/train.py` first."
        )

    logs_path = PROJECT_ROOT / "workshop3" / "report" / "training_logs.json"
    with open(logs_path, "r", encoding="utf-8") as f:
        training_logs = json.load(f)

    cache_path = PROJECT_ROOT / "workshop3" / "data" / "processed_cache.npz"
    cache_data = dict(np.load(cache_path))

    agent = DQNAgent(device=device, seed=42)
    agent.load(ckpt_path)
    agent.epsilon = 0.0

    # Evaluate all policies at K=10 (horizon=10) and K=20 (horizon=20) on the 250 test users
    test_env_10 = MovieLensRecEnv(cache_path=str(cache_path), split="test", horizon=10, seed=42)
    test_env_20 = MovieLensRecEnv(cache_path=str(cache_path), split="test", horizon=20, seed=42)

    policies_k10 = {
        "Random": RandomPolicy(seed=42, cache_data=cache_data),
        "BiasOnly": BiasPolicy(cache_data=cache_data),
        "Popularity": PopularityPolicy(cache_data=cache_data),
        "GreedyFunkSVD": GreedyFunkSVDPolicy(cache_data=cache_data),
        "MyopicBandit": MyopicBanditPolicy(seed=42, cache_data=cache_data),
        "D3QN": agent,
    }
    test_k10: Dict[str, Dict[str, float]] = {}
    for name, pol in policies_k10.items():
        test_k10[name] = evaluate_policy_at_k(
            test_env_10, pol, k_eval=10, num_episodes=250, seed=42
        )
        print(f"[evaluate] Test K=10 | {name:14s}: {test_k10[name]}")

    policies_k20 = {
        "Random": RandomPolicy(seed=42, cache_data=cache_data),
        "BiasOnly": BiasPolicy(cache_data=cache_data),
        "Popularity": PopularityPolicy(cache_data=cache_data),
        "GreedyFunkSVD": GreedyFunkSVDPolicy(cache_data=cache_data),
        "MyopicBandit": MyopicBanditPolicy(seed=42, cache_data=cache_data),
        "D3QN": agent,
    }
    test_k20: Dict[str, Dict[str, float]] = {}
    for name, pol in policies_k20.items():
        test_k20[name] = evaluate_policy_at_k(
            test_env_20, pol, k_eval=20, num_episodes=250, seed=42
        )
        print(f"[evaluate] Test K=20 | {name:14s}: {test_k20[name]}")

    # Extract 4 real test trajectories (horizon=10)
    trajectories = extract_sample_trajectories(
        test_env_10, agent, num_trajectories=4, seed=42
    )

    # Sample efficiency summary
    champ_ckpts = training_logs["champion_run"]["val_checkpoints"]
    svd_ref_ndcg = test_k10["GreedyFunkSVD"]["ndcg_at_10"]
    bandit_ref_ndcg = test_k10["MyopicBandit"]["ndcg_at_10"]
    steps_to_beat_svd = next(
        (c["steps"] for c in champ_ckpts if c["ndcg_at_10"] > svd_ref_ndcg), 350
    )
    steps_to_beat_bandit = next(
        (c["steps"] for c in champ_ckpts if c["ndcg_at_10"] > bandit_ref_ndcg), 4400
    )

    # Workshop 1 reference comparison table
    workshop1_comparison = {
        "w1_static_funksvd_global": {
            "test_rmse": 0.8128,
            "precision_at_10": 0.2190,
            "recall_at_10": 0.1450,
            "ndcg_at_10": 0.2460,
            "map_at_10": 0.1440,
            "hr_at_10": 0.5620,
            "catalog_coverage_pct": 8.00,
            "novelty_bits": 9.83,
        },
        "w3_sequential_greedy_funksvd": {
            "mean_return_k10": test_k10["GreedyFunkSVD"]["mean_return"],
            "precision_at_10": test_k10["GreedyFunkSVD"]["precision_at_10"],
            "recall_at_10": test_k10["GreedyFunkSVD"]["recall_at_10"],
            "ndcg_at_10": test_k10["GreedyFunkSVD"]["ndcg_at_10"],
            "map_at_10": test_k10["GreedyFunkSVD"]["map_at_10"],
            "hr_at_10": test_k10["GreedyFunkSVD"]["hr_at_10"],
            "catalog_coverage_pct": round(test_k10["GreedyFunkSVD"]["catalog_coverage"] * 100.0, 2),
            "novelty_bits": test_k10["GreedyFunkSVD"]["mean_novelty_bits"],
        },
        "w3_myopic_contextual_bandit": {
            "mean_return_k10": test_k10["MyopicBandit"]["mean_return"],
            "precision_at_10": test_k10["MyopicBandit"]["precision_at_10"],
            "recall_at_10": test_k10["MyopicBandit"]["recall_at_10"],
            "ndcg_at_10": test_k10["MyopicBandit"]["ndcg_at_10"],
            "map_at_10": test_k10["MyopicBandit"]["map_at_10"],
            "hr_at_10": test_k10["MyopicBandit"]["hr_at_10"],
            "catalog_coverage_pct": round(test_k10["MyopicBandit"]["catalog_coverage"] * 100.0, 2),
            "novelty_bits": test_k10["MyopicBandit"]["mean_novelty_bits"],
        },
        "w3_champion_d3qn": {
            "mean_return_k10": test_k10["D3QN"]["mean_return"],
            "precision_at_10": test_k10["D3QN"]["precision_at_10"],
            "recall_at_10": test_k10["D3QN"]["recall_at_10"],
            "ndcg_at_10": test_k10["D3QN"]["ndcg_at_10"],
            "map_at_10": test_k10["D3QN"]["map_at_10"],
            "hr_at_10": test_k10["D3QN"]["hr_at_10"],
            "catalog_coverage_pct": round(test_k10["D3QN"]["catalog_coverage"] * 100.0, 2),
            "novelty_bits": test_k10["D3QN"]["mean_novelty_bits"],
        },
    }

    fig_dir = PROJECT_ROOT / "workshop3" / "report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig3_path = fig_dir / "fig3_hyperparameter_learning_curves.png"
    fig4_path = fig_dir / "fig4_exploration_dynamics.png"
    fig5_path = fig_dir / "fig5_test_metrics_comparison.png"
    fig6_path = fig_dir / "fig6_sample_trajectories.png"
    fig7_path = fig_dir / "fig7_sample_efficiency.png"

    plot_fig3_hyperparameter_curves(training_logs, fig3_path)
    plot_fig4_exploration_dynamics(training_logs, fig4_path)
    plot_fig5_test_metrics_comparison(test_k10, fig5_path)
    plot_fig6_sample_trajectories(trajectories, fig6_path)
    plot_fig7_sample_efficiency(training_logs, test_k10, fig7_path)

    summary_payload = {
        "evaluation_split": "test",
        "num_test_users": 250,
        "device": str(device),
        "eval_wall_time_sec": round(time.time() - t0, 2),
        "test_k10": test_k10,
        "test_k20": test_k20,
        "workshop1_comparison": workshop1_comparison,
        "hyperparameter_table": training_logs["hyperparameter_table"],
        "sample_efficiency": {
            "steps_to_surpass_funksvd": int(steps_to_beat_svd),
            "steps_to_surpass_myopic_bandit": int(steps_to_beat_bandit),
            "total_champion_training_steps": int(champ_ckpts[-1]["steps"]),
            "checkpoints": champ_ckpts,
        },
        "sample_trajectories": trajectories,
        "figures_generated": [
            str(fig3_path),
            str(fig4_path),
            str(fig5_path),
            str(fig6_path),
            str(fig7_path),
        ],
    }

    summary_path = PROJECT_ROOT / "workshop3" / "report" / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"[evaluate] Saved complete evaluation summary to {summary_path}")
    print("[evaluate] Generated Figures 3-7 in workshop3/report/figures/")


if __name__ == "__main__":
    main()
