"""
Hyperparameter Grid Search and Champion Dueling Double DQN (D3QN) Training Pipeline
for Sequential Movie Recommendation on MovieLens 32M (Workshop 3).

Every configuration is trained with the same standard Double DQN update
(`DQNAgent.update`: replay buffer, Polyak-averaged target network, Huber TD loss on the
taken action, valid-action masking in the bootstrap, gradient-norm clipping at 1.0),
one gradient step per environment step, and epsilon-greedy exploration decayed per
episode. Validation runs the greedy policy (no exploration) once on every RL validation
session.

Pipeline:
1. Eight one-at-a-time hyperparameter configurations (learning rate, discount factor,
   epsilon decay, L2 weight decay), 800 episodes each, logged to TensorBoard under
   `workshop3/runs/workshop3/<config_id>/`.
2. The configuration with the highest smoothed validation return (mean of its last five
   validation evaluations) is retrained for 1,500 episodes as the champion (`d3qn_tuned`);
   its best validation checkpoint is saved to `workshop3/checkpoints/agent_best.pt`.
3. A myopic ablation with the champion hyperparameters but gamma = 0 (a neural contextual
   bandit with the same network, data, and budget) is trained the same way and saved to
   `workshop3/checkpoints/bandit_gamma0.pt`.

All curves and the hyperparameter table are exported to `workshop3/report/training_logs.json`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workshop3.src.agent import DQNAgent
from workshop3.src.baselines import evaluate_policy
from workshop3.src.env import MovieLensRecEnv

CHAMPION_CHECKPOINT = PROJECT_ROOT / "workshop3" / "checkpoints" / "agent_best.pt"
BANDIT_CHECKPOINT = PROJECT_ROOT / "workshop3" / "checkpoints" / "bandit_gamma0.pt"
TRAINING_LOGS = PROJECT_ROOT / "workshop3" / "report" / "training_logs.json"
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
    return evaluate_policy(val_env, agent, seed=seed, user_indices=val_env.split_user_indices)


def train_single_config(
    config: Dict[str, Any],
    num_episodes: int = 800,
    eval_every: int = 40,
    checkpoint_path: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train a single D3QN hyperparameter configuration and log to TensorBoard."""
    config_id = config["config_id"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tb_dir = PROJECT_ROOT / "workshop3" / "runs" / "workshop3" / config_id
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


def main() -> None:
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


if __name__ == "__main__":
    main()
