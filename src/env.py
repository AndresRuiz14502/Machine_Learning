"""
Gymnasium environment for Sequential Movie Recommendation on MovieLens 32M (Workshop 3).

Implements `MovieLensRecEnv(gymnasium.Env)` with:
- Discrete(50) candidate slot action space. A slot holds one real rating of the session
  (see `data_loader.py`); sessions shorter than 50 ratings are padded with invalid slots that
  start masked (`valid_mask == 0`).
- Box(250,) state space built only from training-split statistics and observed feedback:
  cosine similarity between each candidate's FunkSVD item factor and the session vector,
  static FunkSVD scores, self-information novelty, the unavailable-slot mask, the user's
  FunkSVD factor, the session vector, and 10 session progress scalars. The session vector
  starts at the user's FunkSVD factor and moves toward liked items and away from missed
  ones (Rocchio-style feedback); no hidden label or future information enters the state.
- Reward: relevance from the real rating (hit iff rating >= 4.0), plus novelty and genre
  diversity bonuses paid only on hits; repeated or padding slots cost -1.5; the user
  abandons the session after 5 consecutive misses.
- Random policy verification and publication-quality diagnostic plotting (Figure 2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_PATH = "data/processed_cache.npz"

LIKE_THRESHOLD = 4.0
REPEAT_PENALTY = -1.5
MISS_REWARD = -0.4
NOVELTY_WEIGHT = 0.20
DIVERSITY_WEIGHT = 0.10
ABANDON_AFTER_MISSES = 5


def resolve_repo_path(path: str | Path) -> Path:
    """Return ``path`` unchanged if it exists (or is absolute), else resolve it against the repo root."""
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return REPO_ROOT / candidate


class MovieLensRecEnv(gym.Env):
    """Sequential Movie Recommendation Gymnasium Environment on MovieLens 32M."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        cache_path: str | Path = DEFAULT_CACHE_PATH,
        split: str = "train",
        horizon: int = 10,
        seed: int = 42,
        render_mode: Optional[str] = "ansi",
    ) -> None:
        super().__init__()
        resolved_cache = resolve_repo_path(cache_path)
        self.cache_path = str(resolved_cache)
        self.split = split.lower()
        self.horizon = int(horizon)
        self.default_seed = int(seed)
        self.render_mode = render_mode

        if not resolved_cache.exists():
            raise FileNotFoundError(
                f"Processed cache not found at '{self.cache_path}'. "
                f"Run `python -m src.data_loader --w3` first."
            )

        data = np.load(self.cache_path)
        self.user_factors: np.ndarray = data["user_factors"].astype(np.float32)
        self.user_biases: np.ndarray = data["user_biases"].astype(np.float32)
        self.cand_items: np.ndarray = data["cand_items"].astype(np.int32)
        self.cand_valid: np.ndarray = data["cand_valid"].astype(np.float32)
        self.cand_item_factors: np.ndarray = data["cand_item_factors"].astype(np.float32)
        self.cand_true_ratings: np.ndarray = data["cand_true_ratings"].astype(np.float32)
        self.cand_is_liked: np.ndarray = data["cand_is_liked"].astype(np.int8)
        self.cand_svd_scores: np.ndarray = data["cand_svd_scores"].astype(np.float32)
        self.cand_bias_scores: np.ndarray = data["cand_bias_scores"].astype(np.float32)
        self.cand_pop_scores: np.ndarray = data["cand_pop_scores"].astype(np.float32)
        self.cand_novelty: np.ndarray = data["cand_novelty"].astype(np.float32)
        self.cand_novelty_bits: np.ndarray = data["cand_novelty_bits"].astype(np.float32)
        self.cand_genres: np.ndarray = data["cand_genres"].astype(np.float32)

        if self.split == "train":
            self.split_user_indices: np.ndarray = data["train_user_indices"].astype(np.int32)
        elif self.split in ("val", "validation"):
            self.split_user_indices = data["val_user_indices"].astype(np.int32)
        elif self.split == "test":
            self.split_user_indices = data["test_user_indices"].astype(np.int32)
        else:
            raise ValueError(f"Unknown split '{split}'. Expected 'train', 'val', or 'test'.")

        self.num_candidates = int(self.cand_items.shape[1])
        self.latent_dim = int(self.user_factors.shape[1])
        self.num_genres = int(self.cand_genres.shape[2])

        min_valid = int(self.cand_valid[self.split_user_indices].sum(axis=1).min())
        if min_valid < self.horizon:
            raise ValueError(
                f"Split '{self.split}' has a session with {min_valid} valid slots (< horizon {self.horizon})."
            )

        # Load optional movie title lookup for human-readable trajectory rendering
        lookup_path = Path(self.cache_path).parent / "movie_lookup.json"
        if lookup_path.exists():
            with open(lookup_path, "r", encoding="utf-8") as f:
                self.movie_lookup: Dict[str, Dict[str, Any]] = json.load(f)
        else:
            self.movie_lookup = {}

        u_norms = np.linalg.norm(self.user_factors, axis=1, keepdims=True) + 1e-8
        self.user_factors_norm = (self.user_factors / u_norms).astype(np.float32)

        self.action_space = spaces.Discrete(self.num_candidates)
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(250,), dtype=np.float32
        )

        self._rng = np.random.default_rng(self.default_seed)
        self.u: int = int(self.split_user_indices[0])
        self.t: int = 0
        self.visited: np.ndarray = np.zeros(self.num_candidates, dtype=np.float32)
        self.session_vec: np.ndarray = np.zeros(self.latent_dim, dtype=np.float32)
        self.covered_genres: np.ndarray = np.zeros(self.num_genres, dtype=np.float32)
        self.hits_so_far: int = 0
        self.consecutive_misses: int = 0
        self.last_reward: float = 0.0
        self.novelty_sum: float = 0.0
        self.episode_return: float = 0.0
        self.history: List[Dict[str, Any]] = []

    def _get_obs(self) -> np.ndarray:
        """Assemble the 250-dimensional state vector s_t."""
        u = self.u
        # [0:50] cosine similarity between the session vector and each candidate's FunkSVD factor
        sim_scores = (self.cand_item_factors[u] @ self.session_vec).astype(np.float32)
        # [50:100] static normalized FunkSVD scores
        svd_scores = self.cand_svd_scores[u]
        # [100:150] candidate novelty in [0, 1]
        nov_scores = self.cand_novelty[u]
        # [150:200] unavailable mask (1.0 if already recommended or padding, 0.0 if available)
        visited_mask = self.visited
        # [200:220] normalized static user factor p_u
        pu_norm = self.user_factors_norm[u]
        # [220:240] current session vector \bar{q}_t
        sess_vec = self.session_vec

        # [240:250] 10 session summary scalars
        avail_mask = self.visited < 0.5
        if np.any(avail_mask):
            max_sim_avail = float(np.max(sim_scores[avail_mask]))
            mean_svd_avail = float(np.mean(svd_scores[avail_mask]))
        else:
            max_sim_avail = 0.0
            mean_svd_avail = 0.0

        mean_nov_so_far = float(self.novelty_sum / max(1, self.t))
        genre_cov_so_far = float(np.mean(self.covered_genres > 0.0))
        user_bias_norm = float(np.clip(self.user_biases[u], -2.0, 2.0))

        scalars = np.array(
            [
                float(self.t) / float(self.horizon),
                float(self.hits_so_far) / float(max(1, self.t)),
                float(np.clip(self.last_reward, -2.0, 2.0)),
                float(self.consecutive_misses) / float(ABANDON_AFTER_MISSES),
                mean_nov_so_far,
                genre_cov_so_far,
                user_bias_norm,
                float(self.horizon - self.t) / float(self.horizon),
                max_sim_avail,
                mean_svd_avail,
            ],
            dtype=np.float32,
        )

        obs = np.concatenate(
            [
                sim_scores,
                svd_scores,
                nov_scores,
                visited_mask,
                pu_norm,
                sess_vec,
                scalars,
            ]
        ).astype(np.float32)
        return np.clip(obs, -5.0, 5.0)

    def _info(self) -> Dict[str, Any]:
        return {
            "valid_mask": (1.0 - self.visited).astype(np.float32),
            "user_idx": int(self.u),
            "num_liked_in_pool": int(np.sum(self.cand_is_liked[self.u] == 1)),
            "num_valid_in_pool": int(np.sum(self.cand_valid[self.u] > 0.5)),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if options is not None and "user_idx" in options:
            self.u = int(options["user_idx"])
        else:
            choice_idx = int(self._rng.integers(0, len(self.split_user_indices)))
            self.u = int(self.split_user_indices[choice_idx])

        self.t = 0
        # Padding slots of shorter sessions start unavailable
        self.visited = (1.0 - self.cand_valid[self.u]).astype(np.float32)
        self.covered_genres = np.zeros(self.num_genres, dtype=np.float32)
        self.hits_so_far = 0
        self.consecutive_misses = 0
        self.last_reward = 0.0
        self.novelty_sum = 0.0
        self.episode_return = 0.0
        self.history = []

        # The session starts at the user's static FunkSVD taste vector
        self.session_vec = self.user_factors_norm[self.u].copy()

        return self._get_obs(), self._info()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_int = int(action)
        if action_int < 0 or action_int >= self.num_candidates:
            raise ValueError(
                f"Invalid action {action_int} outside [0, {self.num_candidates - 1}]"
            )

        u = self.u
        self.t += 1

        is_valid_slot = bool(self.cand_valid[u, action_int] > 0.5)
        movie_id = int(self.cand_items[u, action_int])
        true_rating = float(self.cand_true_ratings[u, action_int])
        novelty_norm = float(self.cand_novelty[u, action_int])
        novelty_bits = float(self.cand_novelty_bits[u, action_int])

        if self.visited[action_int] >= 0.5:
            # Repeated recommendation or padding slot
            reward = REPEAT_PENALTY
            is_hit = 0
            self.consecutive_misses += 1
        else:
            self.visited[action_int] = 1.0
            self.novelty_sum += novelty_norm
            item_vec = self.cand_item_factors[u, action_int]

            # Genre coverage of the recommended list (the bonus is paid on hits only)
            item_g = self.cand_genres[u, action_int]
            n_item_genres = float(np.sum(item_g > 0.0))
            new_g_count = float(np.sum((item_g > 0.0) & (self.covered_genres == 0.0)))
            new_genre_bonus = new_g_count / max(1.0, n_item_genres)
            self.covered_genres = np.maximum(self.covered_genres, item_g)

            if self.cand_is_liked[u, action_int] == 1:
                is_hit = 1
                self.hits_so_far += 1
                self.consecutive_misses = 0
                reward = float(
                    1.0
                    + 0.25 * (true_rating - LIKE_THRESHOLD)
                    + NOVELTY_WEIGHT * novelty_norm
                    + DIVERSITY_WEIGHT * new_genre_bonus
                )
                updated = 0.50 * self.session_vec + 0.30 * item_vec
            else:
                is_hit = 0
                self.consecutive_misses += 1
                reward = MISS_REWARD
                updated = 0.75 * self.session_vec - 0.20 * item_vec
            self.session_vec = (updated / (np.linalg.norm(updated) + 1e-8)).astype(np.float32)

        self.last_reward = float(reward)
        self.episode_return += float(reward)

        terminated = bool(self.consecutive_misses >= ABANDON_AFTER_MISSES)
        truncated = bool(self.t >= self.horizon)

        if is_valid_slot:
            m_info = self.movie_lookup.get(
                str(movie_id), {"title": f"Movie #{movie_id}", "genres": "Unknown"}
            )
        else:
            m_info = {"title": "(padding slot)", "genres": "-"}
        self.history.append(
            {
                "step": self.t,
                "action": action_int,
                "movie_id": movie_id,
                "title": m_info["title"],
                "genres": m_info["genres"],
                "is_hit": is_hit,
                "true_rating": round(true_rating, 2),
                "novelty_bits": round(novelty_bits, 2),
                "reward": round(float(reward), 4),
            }
        )

        obs = self._get_obs()
        info = {
            **self._info(),
            "is_hit": int(is_hit),
            "is_valid_slot": int(is_valid_slot),
            "movie_id": int(movie_id),
            "novelty_bits": float(novelty_bits),
            "true_rating": float(true_rating),
            "consecutive_misses": int(self.consecutive_misses),
            "hits_so_far": int(self.hits_so_far),
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> str:
        """Return a formatted string summary of the current episode trajectory."""
        lines = [
            f"=== MovieLensRecEnv Trajectory (User pool_idx={self.u}, Split={self.split}, "
            f"Step={self.t}/{self.horizon}, Return={self.episode_return:+.3f}, Hits={self.hits_so_far}) ==="
        ]
        for h in self.history:
            status = "HIT " if h["is_hit"] == 1 else "MISS"
            lines.append(
                f"  t={h['step']:02d} | a={h['action']:02d} | [{status}] r={h['reward']:+.3f} "
                f"(rating={h['true_rating']:.1f}, nov={h['novelty_bits']:.1f}b) | "
                f"{h['title']} [{h['genres']}]"
            )
        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
        return output


def run_random_policy_diagnostics(
    num_episodes: int = 100,
    save_fig_path: str = "runs/workshop3/fig2_random_policy_diagnostics.png",
    save_json_path: str = "runs/workshop3/random_diagnostics.json",
    cache_path: str = DEFAULT_CACHE_PATH,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Execute a uniform random policy over all 50 slots for `num_episodes` episodes to verify
    MDP dynamics, state/reward bounds, termination behavior, and action coverage.
    Saves Figure 2 (300 DPI) and summary metrics JSON.
    """
    env = MovieLensRecEnv(cache_path=cache_path, split="test", horizon=10, seed=seed)
    rng = np.random.default_rng(seed)

    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    episode_hit_rates: List[float] = []
    episode_novelties: List[float] = []
    early_terminations = 0
    action_counts = np.zeros(env.num_candidates, dtype=np.int32)
    repeat_actions = 0
    invalid_actions = 0
    total_steps = 0

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        assert env.observation_space.contains(obs), "Initial observation out of bounds!"
        ep_ret = 0.0
        ep_hits = 0
        ep_nov: List[float] = []
        visited_set = set()

        for _ in range(env.horizon):
            action = int(rng.integers(0, env.num_candidates))
            if env.cand_valid[env.u, action] < 0.5:
                invalid_actions += 1
            elif action in visited_set:
                repeat_actions += 1
            visited_set.add(action)
            action_counts[action] += 1
            total_steps += 1

            obs, reward, terminated, truncated, step_info = env.step(action)
            assert env.observation_space.contains(obs), "Step observation out of bounds!"
            assert np.isfinite(reward), "Non-finite reward encountered!"

            ep_ret += reward
            ep_hits += step_info["is_hit"]
            ep_nov.append(step_info["novelty_bits"])

            if terminated or truncated:
                if terminated and env.t < env.horizon:
                    early_terminations += 1
                break

        episode_returns.append(float(ep_ret))
        episode_lengths.append(int(env.t))
        episode_hit_rates.append(float(ep_hits) / float(env.t))
        episode_novelties.append(float(np.mean(ep_nov)))

    returns_arr = np.array(episode_returns, dtype=np.float32)
    lengths_arr = np.array(episode_lengths, dtype=np.int32)

    diagnostics = {
        "num_episodes": int(num_episodes),
        "mean_return": round(float(np.mean(returns_arr)), 4),
        "std_return": round(float(np.std(returns_arr)), 4),
        "min_return": round(float(np.min(returns_arr)), 4),
        "max_return": round(float(np.max(returns_arr)), 4),
        "mean_episode_length": round(float(np.mean(lengths_arr)), 2),
        "std_episode_length": round(float(np.std(lengths_arr)), 2),
        "early_termination_rate": round(float(early_terminations) / float(num_episodes), 4),
        "full_horizon_rate": round(float(np.mean(lengths_arr == env.horizon)), 4),
        "mean_hit_rate": round(float(np.mean(episode_hit_rates)), 4),
        "mean_novelty_bits": round(float(np.mean(episode_novelties)), 3),
        "repeat_action_rate": round(float(repeat_actions) / float(max(1, total_steps)), 4),
        "invalid_slot_action_rate": round(float(invalid_actions) / float(max(1, total_steps)), 4),
        "total_steps_executed": int(total_steps),
        "unique_actions_sampled": int(np.sum(action_counts > 0)),
    }

    # Save JSON diagnostics
    json_path = resolve_repo_path(save_json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    # Generate publication-quality 1x3 Figure 2
    fig_path = resolve_repo_path(save_fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=300)

    # Panel (a): Distribution of Episode Returns
    ax0 = axes[0]
    ax0.hist(
        returns_arr,
        bins=16,
        color="#3b82f6",
        edgecolor="#1e3a8a",
        alpha=0.82,
    )
    mean_ret = float(np.mean(returns_arr))
    std_ret = float(np.std(returns_arr))
    ax0.axvline(
        mean_ret,
        color="#dc2626",
        linestyle="--",
        linewidth=1.8,
        label=f"Mean = {mean_ret:.2f} ± {std_ret:.2f}",
    )
    ax0.set_title("(a) Episode Return Distribution")
    ax0.set_xlabel("Cumulative Episode Return")
    ax0.set_ylabel("Number of Episodes")
    ax0.grid(axis="y", linestyle=":", alpha=0.6)
    ax0.legend(loc="upper right", frameon=True)

    # Panel (b): Episode Length Distribution
    ax1 = axes[1]
    unique_lens, len_counts = np.unique(lengths_arr, return_counts=True)
    bar_colors = ["#f59e0b" if l < env.horizon else "#10b981" for l in unique_lens]
    ax1.bar(
        unique_lens,
        len_counts,
        width=0.65,
        color=bar_colors,
        edgecolor="#064e3b",
        alpha=0.88,
    )
    ax1.axvline(
        float(np.mean(lengths_arr)),
        color="#dc2626",
        linestyle="--",
        linewidth=1.6,
        label=f"Mean Length = {np.mean(lengths_arr):.2f} steps",
    )
    ax1.set_xticks(range(ABANDON_AFTER_MISSES, env.horizon + 1))
    ax1.set_title("(b) Episode Length & Early Termination")
    ax1.set_xlabel("Episode Length (Steps)")
    ax1.set_ylabel("Number of Episodes")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True)

    # Panel (c): Action Frequency across 50 Discrete Actions
    ax2 = axes[2]
    slots = np.arange(env.num_candidates)
    ax2.bar(
        slots,
        action_counts,
        width=0.78,
        color="#6366f1",
        edgecolor="#312e81",
        alpha=0.82,
    )
    expected_count = float(total_steps) / float(env.num_candidates)
    ax2.axhline(
        expected_count,
        color="#dc2626",
        linestyle="--",
        linewidth=1.6,
        label=f"Uniform Expected = {expected_count:.1f}",
    )
    ax2.set_title("(c) Action Coverage (50 Candidate Slots)")
    ax2.set_xlabel("Discrete Action Index (0..49)")
    ax2.set_ylabel("Selection Count")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[env] Saved random policy figure to {fig_path}")
    print(f"[env] Saved random policy diagnostics to {json_path}")
    print(json.dumps(diagnostics, indent=2))
    return diagnostics


if __name__ == "__main__":
    run_random_policy_diagnostics()
