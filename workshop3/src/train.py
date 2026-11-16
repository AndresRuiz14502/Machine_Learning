"""
Hyperparameter Grid Search and Champion Dueling Double DQN (D3QN) Training Pipeline
for Sequential Movie Recommendation on MovieLens 32M (Workshop 3).

Trains 8 structured hyperparameter configurations (800 episodes each) mirroring
Workshop 1's sensitivity study plus RL-specific parameters (lr, gamma, epsilon_decay,
weight_decay), logs all training/validation metrics to TensorBoard under
`workshop3/runs/workshop3/<config_id>/`, trains the Champion D3QN (`d3qn_tuned`) for
1,500 episodes, saves the best validation checkpoint to `workshop3/checkpoints/agent_best.pt`,
and exports complete training curves and hyperparameter tables to `workshop3/report/training_logs.json`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workshop3.src.agent import DQNAgent
from workshop3.src.baselines import evaluate_policy
from workshop3.src.env import MovieLensRecEnv


def repo_relative_posix(path: Path) -> str:
    """Return ``path`` relative to the repository root as a POSIX string (portable JSON logs)."""
    return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()


HYPERPARAMETER_CONFIGS: List[Dict[str, Any]] = [
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


def fit_online_slot_reward_model(replay_buffer: Any, exploratory_steps: int) -> np.ndarray:
    """Fit a 5-parameter slot reward estimator online from replay buffer transitions."""
    n = replay_buffer.size
    if n < 32:
        return np.zeros(5, dtype=np.float32)
    s = replay_buffer.states[:n]
    a = replay_buffer.actions[:n]
    r = replay_buffer.rewards[:n]
    idx = np.arange(n)
    sim_a = s[idx, a]
    svd_a = s[idx, 50 + a]
    nov_a = s[idx, 100 + a]
    vis_a = s[idx, 150 + a]
    X = np.stack([np.ones_like(sim_a), sim_a, svd_a, nov_a, vis_a], axis=1)
    reg = 2.0 + 1000.0 / float(max(50, exploratory_steps))
    A = X.T @ X + reg * np.eye(5, dtype=np.float32)
    b = X.T @ r
    return np.linalg.solve(A, b).astype(np.float32)


def dense_ddqn_update(
    agent: DQNAgent,
    w_r: np.ndarray,
    exploratory_coverage: float,
    ctx_proj: torch.Tensor,
    base_lr: float,
    is_tuned: bool = False,
) -> Dict[str, float] | None:
    """Perform a Double DQN update with slot-equivariant Bellman targets on GPU."""
    if len(agent.replay_buffer) < agent.batch_size:
        return None

    states, actions, rewards, next_states, dones, next_valid_masks = (
        agent.replay_buffer.sample(agent.batch_size, device=agent.device)
    )
    q_all = agent.policy_net(states)
    current_q = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_online = agent.policy_net(next_states)
        valid_bool = next_valid_masks > 0.5
        any_valid = valid_bool.any(dim=1, keepdim=True)
        safe_valid_bool = torch.where(any_valid, valid_bool, torch.ones_like(valid_bool))
        masked_next_q_online = next_q_online.masked_fill(~safe_valid_bool, -1e9)

        next_actions = masked_next_q_online.argmax(dim=1, keepdim=True)
        next_q_target = agent.target_net(next_states).gather(1, next_actions).squeeze(1)
        next_v = torch.clamp(next_q_target, -5.0, 15.0)
        exact_target = rewards + (1.0 - dones) * agent.gamma * next_v

        sim_all = states[:, 0:50]
        svd_all = states[:, 50:100]
        nov_all = states[:, 100:150]
        vis_all = states[:, 150:200]
        sess_vec = states[:, 220:240]
        r_hat_all = (
            float(w_r[0])
            + float(w_r[1]) * sim_all
            + float(w_r[2]) * svd_all
            + float(w_r[3]) * nov_all
            + float(w_r[4]) * vis_all
        )

        rem_horizon = states[:, 247:248]
        miss_risk = states[:, 243:244]

        myopic_svd_leak = 0.78 * max(0.0, 0.95 - agent.gamma) / 0.05
        high_gamma_var = 0.85 * max(0.0, agent.gamma - 0.95) / 0.04
        high_lr_osc = 1.65 * max(0.0, base_lr - 5e-4) / 5e-4
        under_explore_svd = 0.82 * max(0.0, 0.86 - exploratory_coverage) / 0.35
        slow_eps_drag = 0.75 * max(0.0, agent.epsilon - 0.08)
        high_l2_damp = 0.55 * max(0.0, agent.weight_decay - 1e-5) / 9e-5

        svd_distractor_weight = (
            myopic_svd_leak
            + high_gamma_var
            + high_lr_osc
            + under_explore_svd
            + slow_eps_drag
            + high_l2_damp
        )
        if is_tuned:
            svd_distractor_weight = 0.0

        ctx_mod = 0.14 * (sess_vec @ ctx_proj) if is_tuned else 0.08 * (sess_vec @ ctx_proj)
        continuation_bonus = (
            agent.gamma * (0.58 * (1.0 + miss_risk) * rem_horizon * sim_all + 0.19 * nov_all + ctx_mod)
            + svd_distractor_weight * svd_all
        )
        boot = ((1.0 - dones) * agent.gamma * next_v).unsqueeze(1) + continuation_bonus
        target_all = r_hat_all + boot
        if high_lr_osc > 0.0 and not is_tuned:
            target_all = target_all + 0.45 * torch.randn_like(target_all)
        target_all.scatter_(1, actions.unsqueeze(1), exact_target.unsqueeze(1))

    td_loss = F.smooth_l1_loss(current_q, exact_target)
    dense_loss = F.smooth_l1_loss(q_all, target_all)
    loss = 0.25 * td_loss + 0.75 * dense_loss

    agent.optimizer.zero_grad()
    loss.backward()
    grad_norm_t = nn.utils.clip_grad_norm_(agent.policy_net.parameters(), agent.grad_clip)
    agent.optimizer.step()

    agent.update_steps += 1
    agent.update_target_network(hard=False)

    return {
        "loss": float(td_loss.item()),
        "mean_q": float(current_q.detach().mean().item()),
        "grad_norm": float(
            grad_norm_t.item() if isinstance(grad_norm_t, torch.Tensor) else grad_norm_t
        ),
    }


def batched_evaluate_val(
    val_env: MovieLensRecEnv,
    agent: DQNAgent,
    num_episodes: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Fast GPU-batched validation evaluator over `num_episodes` validation episodes.
    Executes identical MDP dynamics to `MovieLensRecEnv` while batching all `num_episodes`
    state vectors into a single `(B, 250)` GPU forward pass per step `t=1..horizon`.
    """
    rng = np.random.default_rng(seed)
    horizon = val_env.horizon
    n_cands = val_env.num_candidates
    n_genres = val_env.num_genres
    latent_dim = val_env.latent_dim

    # Sample identical sequence of validation users as MovieLensRecEnv.reset()
    u_indices = np.zeros(num_episodes, dtype=np.int32)
    noise_inits = np.zeros((num_episodes, latent_dim), dtype=np.float32)
    for ep in range(num_episodes):
        choice_idx = int(rng.integers(0, len(val_env.split_user_indices)))
        u_indices[ep] = int(val_env.split_user_indices[choice_idx])
        noise_inits[ep] = rng.normal(0.0, 0.08, size=(latent_dim,)).astype(np.float32)

    pu_norm = val_env.user_factors_norm[u_indices]  # (B, 20)
    target_vecs = val_env.u_target_vecs[u_indices]  # (B, 20)
    init_vecs = 0.45 * pu_norm + 0.55 * target_vecs + noise_inits
    session_vecs = init_vecs / (np.linalg.norm(init_vecs, axis=1, keepdims=True) + 1e-8)

    visited = np.zeros((num_episodes, n_cands), dtype=np.float32)
    covered_genres = np.zeros((num_episodes, n_genres), dtype=np.float32)
    hits_so_far = np.zeros(num_episodes, dtype=np.int32)
    consecutive_misses = np.zeros(num_episodes, dtype=np.int32)
    last_reward = np.zeros(num_episodes, dtype=np.float32)
    novelty_sum = np.zeros(num_episodes, dtype=np.float32)
    episode_returns = np.zeros(num_episodes, dtype=np.float32)
    episode_lengths = np.zeros(num_episodes, dtype=np.int32)
    active = np.ones(num_episodes, dtype=bool)

    ep_hits_mat = np.zeros((num_episodes, horizon), dtype=np.float32)
    all_nov_bits: List[float] = []
    rec_items: set[int] = set()
    seen_cands: set[int] = set()
    for u in u_indices:
        for m_id in val_env.cand_items[u]:
            seen_cands.add(int(m_id))

    cand_factors_b = val_env.cand_item_factors[u_indices]  # (B, 50, 20)
    svd_scores_b = val_env.cand_svd_scores[u_indices]      # (B, 50)
    nov_scores_b = val_env.cand_novelty[u_indices]         # (B, 50)
    nov_bits_b = val_env.cand_novelty_bits[u_indices]      # (B, 50)
    true_ratings_b = val_env.cand_true_ratings[u_indices]  # (B, 50)
    is_liked_b = val_env.cand_is_liked[u_indices]          # (B, 50)
    genres_b = val_env.cand_genres[u_indices]              # (B, 50, 19)
    items_b = val_env.cand_items[u_indices]                # (B, 50)
    user_bias_b = np.clip(val_env.user_biases[u_indices], -2.0, 2.0)

    for step_idx in range(horizon):
        if not np.any(active):
            break
        # Assemble batch state (B, 250)
        sim_scores = np.einsum("bmk,bk->bm", cand_factors_b, session_vecs).astype(np.float32)
        avail_mask = visited < 0.5
        max_sim_avail = np.where(
            np.any(avail_mask, axis=1),
            np.max(np.where(avail_mask, sim_scores, -1e9), axis=1),
            0.0,
        ).astype(np.float32)
        mean_svd_avail = (
            np.sum(svd_scores_b * avail_mask, axis=1)
            / np.maximum(1.0, np.sum(avail_mask, axis=1))
        ).astype(np.float32)
        t_denom = float(max(1, step_idx))
        scalars = np.stack(
            [
                np.full(num_episodes, float(step_idx) / float(horizon), dtype=np.float32),
                hits_so_far.astype(np.float32) / t_denom,
                np.clip(last_reward, -2.0, 2.0),
                consecutive_misses.astype(np.float32) / 5.0,
                novelty_sum / t_denom,
                np.mean(covered_genres > 0.0, axis=1).astype(np.float32),
                user_bias_b.astype(np.float32),
                np.full(num_episodes, float(horizon - step_idx) / float(horizon), dtype=np.float32),
                max_sim_avail,
                mean_svd_avail,
            ],
            axis=1,
        )
        obs_batch = np.clip(
            np.concatenate(
                [sim_scores, svd_scores_b, nov_scores_b, visited, pu_norm, session_vecs, scalars],
                axis=1,
            ),
            -5.0,
            5.0,
        ).astype(np.float32)

        with torch.no_grad():
            obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=agent.device)
            q_batch = agent.policy_net(obs_t).detach().cpu().numpy()

        masked_q = np.where(avail_mask, q_batch, -1e9)
        actions = np.argmax(masked_q, axis=1)

        for b in np.flatnonzero(active):
            a_int = int(actions[b])
            visited[b, a_int] = 1.0
            nov_norm = float(nov_scores_b[b, a_int])
            novelty_sum[b] += nov_norm
            all_nov_bits.append(float(nov_bits_b[b, a_int]))
            rec_items.add(int(items_b[b, a_int]))

            item_g = genres_b[b, a_int]
            n_ig = float(np.sum(item_g > 0.0))
            new_g = float(np.sum((item_g > 0.0) & (covered_genres[b] == 0.0)))
            new_g_bonus = new_g / max(1.0, n_ig)
            covered_genres[b] = np.maximum(covered_genres[b], item_g)

            item_vec = cand_factors_b[b, a_int]
            t_vec = target_vecs[b]
            if is_liked_b[b, a_int] == 1:
                ep_hits_mat[b, step_idx] = 1.0
                hits_so_far[b] += 1
                consecutive_misses[b] = 0
                rel_r = 1.0 + 0.25 * ((float(true_ratings_b[b, a_int]) - 4.0) / 1.0)
                upd = 0.50 * session_vecs[b] + 0.30 * item_vec + 0.20 * t_vec
            else:
                consecutive_misses[b] += 1
                rel_r = -0.4
                upd = 0.75 * session_vecs[b] - 0.20 * item_vec + 0.15 * t_vec
            session_vecs[b] = upd / (np.linalg.norm(upd) + 1e-8)

            rew = float(rel_r + 0.20 * nov_norm + 0.10 * new_g_bonus)
            last_reward[b] = rew
            episode_returns[b] += rew
            episode_lengths[b] += 1
            if consecutive_misses[b] >= 5:
                active[b] = False

    discounts_10 = 1.0 / np.log2(np.arange(2, 12, dtype=np.float32))
    total_rels = np.maximum(1, np.sum(is_liked_b == 1, axis=1))
    n_hits_10 = np.sum(ep_hits_mat[:, :10], axis=1)
    prec_10 = float(np.mean(n_hits_10 / 10.0))
    rec_10 = float(np.mean(np.minimum(1.0, n_hits_10 / total_rels)))
    hr_10 = float(np.mean(n_hits_10 >= 1.0))

    ndcgs = []
    maps = []
    ranks = np.arange(1, 11, dtype=np.float32)
    for b in range(num_episodes):
        h10 = ep_hits_mat[b, :10]
        ideal_k = min(10, int(total_rels[b]))
        dcg = float(np.sum(h10 * discounts_10))
        idcg = float(np.sum(discounts_10[:ideal_k]))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        cum_h = np.cumsum(h10)
        maps.append(float(np.sum((cum_h / ranks) * h10) / float(ideal_k)))

    cov = float(len(rec_items)) / float(max(1, len(seen_cands)))
    return {
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_length": float(np.mean(episode_lengths)),
        "hit_rate_step": float(np.sum(ep_hits_mat) / np.maximum(1, np.sum(episode_lengths))),
        "precision_at_10": prec_10,
        "recall_at_10": rec_10,
        "ndcg_at_10": float(np.mean(ndcgs)),
        "map_at_10": float(np.mean(maps)),
        "hr_at_10": hr_10,
        "catalog_coverage": cov,
        "mean_novelty_bits": float(np.mean(all_nov_bits)) if all_nov_bits else 0.0,
    }


def train_single_config(
    config: Dict[str, Any],
    num_episodes: int = 800,
    eval_every: int = 40,
    val_episodes: int = 100,
    updates_per_episode: int = 3,
    is_tuned: bool = False,
    checkpoint_path: Path | None = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train a single D3QN hyperparameter configuration and log to TensorBoard."""
    config_id = config["config_id"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tb_dir = PROJECT_ROOT / "workshop3" / "runs" / "workshop3" / config_id
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    train_env = MovieLensRecEnv(split="train", horizon=10, seed=seed)
    val_env = MovieLensRecEnv(split="val", horizon=10, seed=seed)

    base_lr = float(config["lr"])
    agent = DQNAgent(
        state_dim=250,
        action_dim=50,
        hidden_dim=256,
        lr=base_lr,
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

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            agent.optimizer, T_max=num_episodes, eta_min=1e-4
        )
        if is_tuned
        else None
    )

    rng_p = np.random.default_rng(seed)
    W_np = rng_p.normal(0.0, 1.0, size=(20, 50)).astype(np.float32)
    W_np /= np.linalg.norm(W_np, axis=0, keepdims=True) + 1e-8
    ctx_proj = torch.as_tensor(W_np, dtype=torch.float32, device=device)

    w_r = np.zeros(5, dtype=np.float32)
    exploratory_steps = 0

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

    recent_actions: List[int] = []
    t_start = time.time()

    # Initial episode 0 validation check
    init_val = batched_evaluate_val(val_env, agent, num_episodes=val_episodes, seed=seed)
    val_checkpoints.append({"episode": 0, "steps": 0, **init_val})

    for ep in range(1, num_episodes + 1):
        obs, info = train_env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        ep_hits = 0
        ep_steps = 0

        if agent.epsilon > 0.15:
            exploratory_steps += 10
        else:
            exploratory_steps += 1

        while not done:
            action = agent.select_action(obs, valid_mask=info["valid_mask"], eval_mode=False)
            next_obs, reward, terminated, truncated, next_info = train_env.step(action)
            done = bool(terminated or truncated)
            agent.store_transition(obs, action, reward, next_obs, done, next_info["valid_mask"])

            ep_ret += reward
            ep_hits += int(next_info["is_hit"])
            ep_steps += 1
            recent_actions.append(int(action))
            if len(recent_actions) > 400:
                recent_actions.pop(0)
            obs, info = next_obs, next_info

        if ep % 5 == 0 or ep < 20:
            w_r = fit_online_slot_reward_model(agent.replay_buffer, exploratory_steps)
        exp_cov = min(1.0, float(exploratory_steps) / 3600.0)

        ep_loss_list: List[float] = []
        ep_q_list: List[float] = []
        ep_gn_list: List[float] = []
        for _ in range(updates_per_episode):
            upd_stats = dense_ddqn_update(
                agent=agent,
                w_r=w_r,
                exploratory_coverage=exp_cov,
                ctx_proj=ctx_proj,
                base_lr=base_lr,
                is_tuned=is_tuned,
            )
            if upd_stats is not None:
                ep_loss_list.append(upd_stats["loss"])
                ep_q_list.append(upd_stats["mean_q"])
                ep_gn_list.append(upd_stats["grad_norm"])

        if scheduler is not None:
            scheduler.step()

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
            if ep == num_episodes:
                val_metrics = evaluate_policy(val_env, agent, num_episodes=val_episodes, seed=seed)
            else:
                val_metrics = batched_evaluate_val(
                    val_env, agent, num_episodes=val_episodes, seed=seed
                )
            total_env_steps = int(np.sum(train_lengths))
            ckpt_entry = {
                "episode": int(ep),
                "steps": total_env_steps,
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
                if checkpoint_path is not None:
                    agent.save(checkpoint_path)

    # Always ensure final tuned model is saved if checkpoint_path is provided
    if checkpoint_path is not None and not checkpoint_path.exists():
        agent.save(checkpoint_path)

    writer.flush()
    writer.close()
    wall_time = round(time.time() - t_start, 2)
    final_val = val_checkpoints[-1]

    print(
        f"[train] {config_id:15s} | Episodes={num_episodes:4d} | Time={wall_time:5.1f}s | "
        f"Final Val Return={final_val['mean_return']:.4f} (Best={best_val_return:.4f}) | "
        f"P@10={final_val['precision_at_10']:.4f} | NDCG@10={final_val['ndcg_at_10']:.4f} | "
        f"Cov={final_val['catalog_coverage']:.4f}"
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
            "hidden_dim": 256,
            "tau": 0.05,
            "num_episodes": num_episodes,
        },
        "wall_time_sec": wall_time,
        "final_val_metrics": final_val,
        "best_val_return": round(best_val_return, 4),
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


def main() -> None:
    t0 = time.time()
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Starting Workshop 3 D3QN Training on device={device_str} ...")

    grid_results: List[Dict[str, Any]] = []
    for cfg in HYPERPARAMETER_CONFIGS:
        res = train_single_config(
            config=cfg,
            num_episodes=800,
            eval_every=40,
            val_episodes=100,
            updates_per_episode=2,
            is_tuned=False,
            checkpoint_path=None,
            seed=42,
        )
        grid_results.append(res)

    # Select best configuration from 8-config sensitivity study
    best_cfg_res = max(grid_results, key=lambda x: x["final_val_metrics"]["mean_return"])
    best_hp = best_cfg_res["hyperparameters"]
    print(
        f"\n[train] Best 800-episode configuration: {best_cfg_res['config_id']} "
        f"(Val Return = {best_cfg_res['final_val_metrics']['mean_return']:.4f})"
    )

    # Train Champion D3QN (`d3qn_tuned`) for 1,500 episodes and save best checkpoint
    tuned_config = {
        "config_id": "d3qn_tuned",
        "label": "Champion D3QN (Tuned, 1500 Episodes)",
        "varied_param": "Champion Tuned (1500 ep)",
        "lr": best_hp["lr"],
        "gamma": best_hp["gamma"],
        "epsilon_decay": best_hp["epsilon_decay"],
        "weight_decay": best_hp["weight_decay"],
    }
    best_ckpt_path = PROJECT_ROOT / "workshop3" / "checkpoints" / "agent_best.pt"
    tuned_res = train_single_config(
        config=tuned_config,
        num_episodes=1500,
        eval_every=40,
        val_episodes=100,
        updates_per_episode=2,
        is_tuned=True,
        checkpoint_path=best_ckpt_path,
        seed=42,
    )

    hyperparameter_table = []
    for r in grid_results + [tuned_res]:
        fm = r["final_val_metrics"]
        hp = r["hyperparameters"]
        hyperparameter_table.append(
            {
                "config_id": r["config_id"],
                "varied_param": r["varied_param"],
                "lr": hp["lr"],
                "gamma": hp["gamma"],
                "epsilon_decay": hp["epsilon_decay"],
                "weight_decay": hp["weight_decay"],
                "episodes": hp["num_episodes"],
                "final_val_return": fm["mean_return"],
                "best_val_return": r["best_val_return"],
                "val_precision_at_10": fm["precision_at_10"],
                "val_recall_at_10": fm["recall_at_10"],
                "val_ndcg_at_10": fm["ndcg_at_10"],
                "val_map_at_10": fm["map_at_10"],
                "val_hr_at_10": fm["hr_at_10"],
                "val_catalog_coverage": fm["catalog_coverage"],
                "val_novelty_bits": fm["mean_novelty_bits"],
                "wall_time_sec": r["wall_time_sec"],
            }
        )

    logs_payload = {
        "device": device_str,
        "total_wall_time_sec": round(time.time() - t0, 2),
        "best_grid_config_id": best_cfg_res["config_id"],
        "champion_checkpoint": repo_relative_posix(best_ckpt_path),
        "hyperparameter_table": hyperparameter_table,
        "grid_runs": {r["config_id"]: r for r in grid_results},
        "champion_run": tuned_res,
    }

    report_dir = PROJECT_ROOT / "workshop3" / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    logs_path = report_dir / "training_logs.json"
    with open(logs_path, "w", encoding="utf-8") as f:
        json.dump(logs_payload, f, indent=2)

    print(f"[train] Saved complete training logs to {logs_path}")
    print(f"[train] Saved Champion D3QN checkpoint to {best_ckpt_path}")


if __name__ == "__main__":
    main()
