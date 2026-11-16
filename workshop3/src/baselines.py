"""Non-RL and supervised baselines for Workshop 3, plus the shared policy evaluation loop.

The myopic baseline is not defined here: it is the gamma = 0 DQN trained by `train.py`
(`workshop3/checkpoints/bandit_gamma0.pt`), evaluated as a greedy `DQNAgent`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

import numpy as np


class BasePolicy(ABC):
    """Common interface for recommendation policies over the 50-item candidate slate.

    State vector layout (`state_dim = 250`):
        - `state[0:50]`   : dynamic session cosine similarity per candidate
        - `state[50:100]` : static Workshop 1 Funk-SVD normalized predicted scores
        - `state[100:150]`: candidate self-information / novelty scores
        - `state[150:200]`: visited action mask (`1.0` = already recommended, `0.0` = available)
        - `state[200:250]`: user embedding, session context embedding, and session scalars
    """

    name: str = "BasePolicy"
    action_dim: int = 50

    def reset(self) -> None:
        """Reset any per-episode internal state (optional hook)."""
        pass

    @staticmethod
    def extract_valid_mask(
        state: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        info: Optional[dict[str, Any]] = None,
        action_dim: int = 50,
    ) -> np.ndarray:
        """Return valid action mask (`1.0` = valid/unvisited, `0.0` = visited/invalid)."""
        if valid_mask is None and isinstance(info, dict) and "valid_mask" in info:
            valid_mask = info["valid_mask"]

        if valid_mask is not None:
            mask = np.asarray(valid_mask, dtype=np.float32).reshape(-1)
        else:
            state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
            if state_arr.size >= 150 + action_dim:
                visited = state_arr[150 : 150 + action_dim]
                mask = 1.0 - visited
            else:
                mask = np.ones(action_dim, dtype=np.float32)

        if not np.any(mask > 0.5):
            mask = np.ones(action_dim, dtype=np.float32)
        return np.clip(mask, 0.0, 1.0).astype(np.float32)

    @abstractmethod
    def score_actions(
        self,
        state: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        info: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        """Return preference scores `(50,)` with invalid actions masked to `-1e9`."""
        raise NotImplementedError

    def select_action(
        self,
        state: np.ndarray,
        info: Optional[dict[str, Any]] = None,
        valid_mask: Optional[np.ndarray] = None,
        eval_mode: bool = True,
    ) -> int:
        """Select the highest-scoring valid candidate action."""
        scores = self.score_actions(state, valid_mask=valid_mask, info=info)
        return int(np.argmax(scores))

    def rank_actions(
        self,
        state: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        info: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        """Return candidate indices `(50,)` sorted from highest to lowest preference score."""
        scores = self.score_actions(state, valid_mask=valid_mask, info=info)
        return np.argsort(-scores).astype(np.int64)


class RandomPolicy(BasePolicy):
    """Uniform random recommendation policy over unvisited candidate actions."""

    name: str = "Random"

    def __init__(self, seed: int = 42, cache_data: Optional[dict[str, Any]] = None) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.cache_data = cache_data

    def score_actions(
        self,
        state: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        info: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        mask = self.extract_valid_mask(state, valid_mask=valid_mask, info=info)
        random_scores = self.rng.uniform(0.0, 1.0, size=self.action_dim).astype(np.float32)
        return np.where(mask > 0.5, random_scores, -1e9).astype(np.float32)

    def select_action(
        self,
        state: np.ndarray,
        info: Optional[dict[str, Any]] = None,
        valid_mask: Optional[np.ndarray] = None,
        eval_mode: bool = True,
    ) -> int:
        mask = self.extract_valid_mask(state, valid_mask=valid_mask, info=info)
        valid_indices = np.flatnonzero(mask > 0.5)
        return int(self.rng.choice(valid_indices))


class PopularityPolicy(BasePolicy):
    """Global popularity baseline selecting the most popular unvisited candidate.

    Uses `cache_data['cand_pop_scores'][user_idx]` when available via `info['user_idx']`,
    and otherwise uses `-state[100:150]` (since novelty is monotonically `-log2(pop)`).
    """

    name: str = "Popularity"

    def __init__(self, cache_data: Optional[dict[str, Any]] = None) -> None:
        self.cache_data = cache_data

    def score_actions(
        self,
        state: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        info: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        mask = self.extract_valid_mask(state, valid_mask=valid_mask, info=info)
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)

        scores: Optional[np.ndarray] = None
        if (
            self.cache_data is not None
            and isinstance(info, dict)
            and "user_idx" in info
            and "cand_pop_scores" in self.cache_data
        ):
            u_idx = int(info["user_idx"])
            pop_arr = self.cache_data["cand_pop_scores"]
            if 0 <= u_idx < len(pop_arr):
                scores = np.asarray(pop_arr[u_idx], dtype=np.float32)

        if scores is None:
            # Novelty state[100:150] is monotonically -log2(popularity), so -novelty ranks by popularity
            novelty = state_arr[100:150]
            scores = -novelty

        return np.where(mask > 0.5, scores, -1e9).astype(np.float32)


class BiasPolicy(BasePolicy):
    """User + Item Bias baseline (`mu + b_u + b_i`) from Workshop 1.

    Uses `cache_data['cand_bias_scores'][user_idx]` when available via `info['user_idx']`,
    and falls back to `0.5 * svd_scores - 0.5 * novelty`.
    """

    name: str = "BiasOnly"

    def __init__(self, cache_data: Optional[dict[str, Any]] = None) -> None:
        self.cache_data = cache_data

    def score_actions(
        self,
        state: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        info: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        mask = self.extract_valid_mask(state, valid_mask=valid_mask, info=info)
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)

        scores: Optional[np.ndarray] = None
        if (
            self.cache_data is not None
            and isinstance(info, dict)
            and "user_idx" in info
            and "cand_bias_scores" in self.cache_data
        ):
            u_idx = int(info["user_idx"])
            bias_arr = self.cache_data["cand_bias_scores"]
            if 0 <= u_idx < len(bias_arr):
                scores = np.asarray(bias_arr[u_idx], dtype=np.float32)

        if scores is None:
            svd_scores = state_arr[50:100]
            novelty = state_arr[100:150]
            scores = 0.5 * svd_scores - 0.5 * novelty

        return np.where(mask > 0.5, scores, -1e9).astype(np.float32)


class GreedyFunkSVDPolicy(BasePolicy):
    """Workshop 1 Supervised Matrix Factorization baseline acting greedily on static SVD scores.

    Selects the valid action with highest `state[50:100]` (`svd_scores`) without sequential
    session adaptation or long-horizon RL value propagation.
    """

    name: str = "GreedyFunkSVD"

    def __init__(self, cache_data: Optional[dict[str, Any]] = None) -> None:
        self.cache_data = cache_data

    def score_actions(
        self,
        state: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        info: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        mask = self.extract_valid_mask(state, valid_mask=valid_mask, info=info)
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        svd_scores = state_arr[50:100]
        return np.where(mask > 0.5, svd_scores, -1e9).astype(np.float32)


class GreedyHybridNeuMFPolicy(BasePolicy):
    """Workshop 2 HybridNeuMF baseline acting greedily on static predicted ratings.

    `candidate_scores[u, a]` is the Workshop 2 HybridNeuMF predicted rating of candidate slot `a`
    for pool user `u` (`workshop3/data/w2_candidate_scores.npz`, built by `w2_scores.py`). Like
    `GreedyFunkSVDPolicy`, it selects the valid action with the highest static score, without
    sequential session adaptation or long-horizon RL value propagation. The pool user is read
    from `info['user_idx']`, which the environment returns on every `reset` and `step`.

    `slot_valid[u, a]` (the artifact's `cand_valid`) marks real slots. Padding slots are never
    selected, even when every real slot of the session has already been recommended: the
    policy then repeats a real slot instead.
    """

    name: str = "GreedyHybridNeuMF"

    def __init__(self, candidate_scores: np.ndarray, slot_valid: Optional[np.ndarray] = None) -> None:
        scores = np.asarray(candidate_scores, dtype=np.float32)
        if scores.ndim != 2 or scores.shape[1] != self.action_dim:
            raise ValueError(
                f"candidate_scores must have shape (n_pool_users, {self.action_dim}), got {scores.shape}"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError("candidate_scores contains non-finite values")
        if slot_valid is None:
            valid = np.ones(scores.shape, dtype=bool)
        else:
            valid = np.asarray(slot_valid) > 0.5
            if valid.shape != scores.shape:
                raise ValueError(f"slot_valid shape {valid.shape} != candidate_scores shape {scores.shape}")
        self.candidate_scores = scores
        self.slot_valid = valid

    def score_actions(
        self,
        state: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        info: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        mask = self.extract_valid_mask(state, valid_mask=valid_mask, info=info)
        if not isinstance(info, dict) or info.get("user_idx") is None:
            raise ValueError("GreedyHybridNeuMFPolicy needs info['user_idx'] to look up W2 scores")
        u_idx = int(info["user_idx"])
        if not 0 <= u_idx < len(self.candidate_scores):
            raise IndexError(f"user_idx {u_idx} outside [0, {len(self.candidate_scores) - 1}]")
        real_slots = self.slot_valid[u_idx]
        allowed = (mask > 0.5) & real_slots
        if not np.any(allowed):
            # Every real slot was already recommended: repeat one rather than pick a padding slot
            allowed = real_slots
        return np.where(allowed, self.candidate_scores[u_idx], -1e9).astype(np.float32)


def _extract_env_cache(env: Any) -> dict[str, Any]:
    """Inspect `env` for cached candidate arrays if available."""
    if hasattr(env, "cache_data") and isinstance(env.cache_data, dict):
        return env.cache_data
    if hasattr(env, "data") and isinstance(env.data, dict):
        return env.data
    return {}


def _infer_total_relevant(
    env: Any,
    user_idx: Optional[int],
    initial_state: np.ndarray,
    info: dict[str, Any],
    observed_hits: int,
) -> int:
    """Infer total number of relevant items in the user's 50-item candidate slate."""
    obs = max(1, int(observed_hits))
    for key in ("num_relevant", "total_relevant", "n_relevant", "n_pos"):
        if key in info and info[key] is not None:
            return max(obs, int(info[key]))

    cache = _extract_env_cache(env)
    if user_idx is not None and 0 <= int(user_idx):
        u = int(user_idx)
        for attr in (
            "cand_is_liked",
            "cand_Relevance",
            "cand_relevance",
            "cand_rel",
            "cand_hits",
            "cand_labels",
            "cand_is_pos",
        ):
            arr = getattr(env, attr, cache.get(attr, None))
            if arr is not None and u < len(arr):
                return max(obs, int(np.sum(np.asarray(arr[u]) > 0.5)))
        for attr in ("cand_true_ratings", "cand_ratings", "cand_svd_raw"):
            arr = getattr(env, attr, cache.get(attr, None))
            if arr is not None and u < len(arr):
                return max(obs, int(np.sum(np.asarray(arr[u]) >= 4.0)))

    # Fallback: estimate from static SVD scores > 0.55 in state[50:100]
    svd_slice = np.asarray(initial_state, dtype=np.float32).reshape(-1)[50:100]
    est_rel = int(np.sum(svd_slice >= 0.55))
    return max(obs, min(10, est_rel if est_rel > 0 else 5))


def evaluate_policy(
    env: Any,
    policy: Any,
    num_episodes: int = 250,
    seed: Optional[int] = 42,
    user_indices: Optional[Sequence[int]] = None,
    return_per_episode: bool = False,
) -> dict[str, Any]:
    """Evaluate a recommendation policy or DQNAgent on `env` over `num_episodes` episodes.

    When `user_indices` is given, every listed pool user is evaluated exactly once (one episode
    each, `num_episodes` is ignored), which removes user-sampling noise from the comparison.
    Padding slots (`cand_items == -1`) never count toward catalog coverage.

    When `return_per_episode` is true, the result also holds `per_episode`: one record per
    episode, in evaluation order, with `user_idx`, `return`, `ndcg_at_10`, `precision_at_10`,
    `hits` (number of hits in the episode), `length` (steps), `step_hits` (0/1 per step), and
    `num_relevant` (liked slots in the session, the recall/NDCG denominator). Paired
    statistical tests and metrics at other cut-offs are computed from these records.

    Returns a dictionary containing:
        - `mean_return`: Mean cumulative episode reward
        - `std_return`: Standard deviation of cumulative episode reward
        - `mean_length`: Mean episode length (steps)
        - `hit_rate_step`: Fraction of recommendation steps resulting in a relevant hit
        - `precision_at_10`: Mean Precision@10 across evaluation episodes
        - `recall_at_10`: Mean Recall@10 across evaluation episodes
        - `ndcg_at_10`: Mean NDCG@10 across evaluation episodes
        - `map_at_10`: Mean AP@10 (MAP@10) across evaluation episodes
        - `hr_at_10`: Mean HitRate@10 (at least 1 hit in top-10 session)
        - `catalog_coverage`: Fraction of unique catalog/candidate items recommended
        - `mean_novelty_bits`: Mean self-information novelty (in bits) of recommended items
        - `intra_list_diversity`: Mean intra-list cosine diversity among recommended items
        - `per_episode`: only when `return_per_episode` is true (see above)
    """
    per_episode: list[dict[str, Any]] = []
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    all_step_hits: list[float] = []
    all_step_novelties: list[float] = []
    episode_ilds: list[float] = []

    precisions_10: list[float] = []
    recalls_10: list[float] = []
    ndcgs_10: list[float] = []
    maps_10: list[float] = []
    hrs_10: list[float] = []

    recommended_items: set[int] = set()
    all_seen_candidate_items: set[int] = set()
    recommended_action_slots: set[int] = set()

    cache = _extract_env_cache(env)
    cand_items_arr = getattr(
        env,
        "cand_items",
        getattr(
            env,
            "cand_movie_idx",
            cache.get("cand_items", cache.get("cand_movie_idx", cache.get("cand_item_ids", None))),
        ),
    )
    cand_novelty_bits_arr = getattr(
        env,
        "cand_novelty_bits",
        cache.get("cand_novelty_bits", None),
    )
    cand_item_factors_arr = getattr(
        env,
        "cand_item_factors",
        cache.get("cand_item_factors", None),
    )
    item_embeddings_arr = getattr(
        env,
        "item_embeddings",
        getattr(
            env,
            "item_factors",
            cache.get("item_embeddings", cache.get("item_factors", cache.get("V", None))),
        ),
    )

    discounts_10 = 1.0 / np.log2(np.arange(2, 12, dtype=np.float32))

    eval_users = None if user_indices is None else [int(u) for u in user_indices]
    n_episodes = len(eval_users) if eval_users is not None else int(num_episodes)

    for ep in range(n_episodes):
        if hasattr(policy, "reset"):
            policy.reset()

        # Support both env.reset(seed=...) and env.reset()
        try:
            seed_kw = {"seed": int(seed)} if (seed is not None and ep == 0) else {}
            if eval_users is not None:
                reset_out = env.reset(options={"user_idx": eval_users[ep]}, **seed_kw)
            else:
                reset_out = env.reset(**seed_kw)
        except TypeError:
            reset_out = env.reset()

        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            state, info = reset_out
        else:
            state, info = reset_out, {}
        if not isinstance(info, dict):
            info = {}

        state = np.asarray(state, dtype=np.float32)
        initial_state = state.copy()

        user_idx = info.get("user_idx", None)
        if user_idx is None:
            for attr in ("current_user_idx", "user_idx", "u_idx", "active_user"):
                if hasattr(env, attr):
                    user_idx = getattr(env, attr)
                    if user_idx is not None:
                        info["user_idx"] = int(user_idx)
                        break

        if user_idx is not None and cand_items_arr is not None and 0 <= int(user_idx) < len(cand_items_arr):
            for c_item in np.asarray(cand_items_arr[int(user_idx)]).reshape(-1):
                if int(c_item) >= 0:  # padding slots hold -1
                    all_seen_candidate_items.add(int(c_item))

        done = False
        ep_return = 0.0
        ep_steps = 0
        ep_hits: list[float] = []
        ep_chosen_actions: list[int] = []
        ep_chosen_items: list[int] = []
        ep_step_diversities: list[float] = []
        last_step_info: dict[str, Any] = dict(info)

        while not done:
            valid_mask = BasePolicy.extract_valid_mask(state, info=info)

            # Call policy.select_action cleanly for both BasePolicy and DQNAgent
            try:
                action = int(policy.select_action(state, info))
            except TypeError:
                action = int(policy.select_action(state))

            state_flat = state.reshape(-1)
            step_sim = float(state_flat[action]) if state_flat.size >= 50 else 0.0
            step_nov = float(state_flat[100 + action]) if state_flat.size >= 150 else 0.0

            step_out = env.step(action)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                next_state, reward, terminated, truncated, step_info = step_out
                done = bool(terminated or truncated)
            elif isinstance(step_out, tuple) and len(step_out) == 4:
                next_state, reward, done, step_info = step_out
                done = bool(done)
            else:
                raise ValueError(f"Unexpected env.step() output format: {type(step_out)}")

            if not isinstance(step_info, dict):
                step_info = {}
            last_step_info = step_info

            reward_val = float(reward)
            ep_return += reward_val
            ep_steps += 1
            ep_chosen_actions.append(action)
            recommended_action_slots.add(action)

            # Track catalog item ID if available
            item_id: Optional[int] = None
            for k in ("movie_id", "item_id", "item_idx", "cand_item"):
                if k in step_info and step_info[k] is not None:
                    item_id = int(step_info[k])
                    break
            if (
                item_id is None
                and user_idx is not None
                and cand_items_arr is not None
                and 0 <= int(user_idx) < len(cand_items_arr)
            ):
                item_id = int(cand_items_arr[int(user_idx)][action])
            if item_id is not None:
                if item_id >= 0:  # padding slots (-1) are not catalog items
                    recommended_items.add(item_id)
                    ep_chosen_items.append(item_id)
            elif user_idx is not None:
                recommended_items.add(int(user_idx) * 50 + action)

            # Determine step hit (binary relevance)
            is_hit: Optional[bool] = None
            for k in ("is_hit", "hit", "relevant", "is_positive", "pos"):
                if k in step_info and step_info[k] is not None:
                    is_hit = bool(step_info[k])
                    break
            if is_hit is None:
                if "rating" in step_info or "true_rating" in step_info:
                    r_true = float(step_info.get("rating", step_info.get("true_rating", 0.0)))
                    is_hit = r_true >= 4.0
                elif "relevance" in step_info:
                    is_hit = float(step_info["relevance"]) > 0.0
                else:
                    is_hit = reward_val > 0.0

            hit_float = 1.0 if is_hit else 0.0
            ep_hits.append(hit_float)
            all_step_hits.append(hit_float)

            # Determine novelty in bits
            nov_bits: Optional[float] = None
            for k in ("novelty_bits", "item_novelty_bits", "raw_novelty"):
                if k in step_info and step_info[k] is not None:
                    nov_bits = float(step_info[k])
                    break
            if (
                nov_bits is None
                and user_idx is not None
                and cand_novelty_bits_arr is not None
                and 0 <= int(user_idx) < len(cand_novelty_bits_arr)
            ):
                nov_bits = float(cand_novelty_bits_arr[int(user_idx)][action])
            if nov_bits is None:
                if "novelty" in step_info and step_info["novelty"] is not None:
                    val = float(step_info["novelty"])
                    nov_bits = val if val > 1.2 else val * 12.0
                else:
                    # If state[100:150] is normalized in [0, 1], scale to bits (~12 bits max)
                    nov_bits = step_nov if step_nov > 1.2 else step_nov * 12.0
            all_step_novelties.append(nov_bits)

            # Step-level diversity fallback (1 - dynamic session similarity after step 0)
            if "diversity" in step_info or "ild" in step_info:
                ep_step_diversities.append(
                    float(step_info.get("diversity", step_info.get("ild", 0.0)))
                )
            elif ep_steps > 1:
                ep_step_diversities.append(float(np.clip(1.0 - step_sim, 0.0, 2.0)))

            state = np.asarray(next_state, dtype=np.float32)
            if "user_idx" not in step_info and user_idx is not None:
                step_info["user_idx"] = user_idx
            info = step_info

        episode_returns.append(ep_return)
        episode_lengths.append(ep_steps)

        # Intra-list diversity for the episode
        if (
            cand_item_factors_arr is not None
            and user_idx is not None
            and 0 <= int(user_idx) < len(cand_item_factors_arr)
            and len(ep_chosen_actions) >= 2
        ):
            emb = np.asarray(
                cand_item_factors_arr[int(user_idx)][ep_chosen_actions], dtype=np.float32
            )
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms = np.where(norms > 1e-8, norms, 1.0)
            normed = emb / norms
            sim_mat = normed @ normed.T
            n_items = len(ep_chosen_actions)
            triu_idx = np.triu_indices(n_items, k=1)
            ild_val = float(np.mean(1.0 - np.clip(sim_mat[triu_idx], -1.0, 1.0)))
            episode_ilds.append(ild_val)
        elif (
            item_embeddings_arr is not None
            and len(ep_chosen_items) >= 2
            and max(ep_chosen_items) < len(item_embeddings_arr)
        ):
            emb = np.asarray(item_embeddings_arr[ep_chosen_items], dtype=np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms = np.where(norms > 1e-8, norms, 1.0)
            normed = emb / norms
            sim_mat = normed @ normed.T
            n_items = len(ep_chosen_items)
            triu_idx = np.triu_indices(n_items, k=1)
            ild_val = float(np.mean(1.0 - np.clip(sim_mat[triu_idx], -1.0, 1.0)))
            episode_ilds.append(ild_val)
        elif ep_step_diversities:
            episode_ilds.append(float(np.mean(ep_step_diversities)))
        else:
            episode_ilds.append(0.0)

        # Top-10 ranking metrics for the episode
        hits_10 = np.zeros(10, dtype=np.float32)
        n_recorded = min(10, len(ep_hits))
        if n_recorded > 0:
            hits_10[:n_recorded] = ep_hits[:n_recorded]

        n_hits_10 = float(np.sum(hits_10))
        total_rel = _infer_total_relevant(
            env=env,
            user_idx=user_idx,
            initial_state=initial_state,
            info=last_step_info,
            observed_hits=int(n_hits_10),
        )

        precisions_10.append(n_hits_10 / 10.0)
        recalls_10.append(float(min(1.0, n_hits_10 / max(1.0, float(total_rel)))))
        hrs_10.append(1.0 if n_hits_10 >= 1.0 else 0.0)

        dcg_10 = float(np.sum(hits_10 * discounts_10))
        ideal_k = min(10, max(1, int(total_rel)))
        idcg_10 = float(np.sum(discounts_10[:ideal_k]))
        ndcgs_10.append(dcg_10 / idcg_10 if idcg_10 > 0.0 else 0.0)

        cum_hits = np.cumsum(hits_10)
        ranks = np.arange(1, 11, dtype=np.float32)
        ap_10 = float(np.sum((cum_hits / ranks) * hits_10) / float(ideal_k))
        maps_10.append(ap_10)

        if return_per_episode:
            per_episode.append(
                {
                    "user_idx": None if user_idx is None else int(user_idx),
                    "return": float(ep_return),
                    "ndcg_at_10": ndcgs_10[-1],
                    "precision_at_10": precisions_10[-1],
                    "hits": int(sum(ep_hits)),
                    "length": int(ep_steps),
                    "step_hits": [int(h) for h in ep_hits],
                    "num_relevant": int(
                        _infer_total_relevant(
                            env=env,
                            user_idx=user_idx,
                            initial_state=initial_state,
                            info=last_step_info,
                            observed_hits=int(sum(ep_hits)),
                        )
                    ),
                }
            )

    # Compute catalog coverage
    total_catalog_size: Optional[int] = None
    for attr in ("num_items", "n_items", "catalog_size", "total_items"):
        val = getattr(env, attr, cache.get(attr, None))
        if val is not None and int(val) > 0:
            total_catalog_size = int(val)
            break
    if total_catalog_size is None and len(all_seen_candidate_items) > 0:
        total_catalog_size = len(all_seen_candidate_items)

    if total_catalog_size is not None and total_catalog_size > 0 and len(recommended_items) > 0:
        coverage = float(min(1.0, len(recommended_items) / float(total_catalog_size)))
    elif len(recommended_items) > 50:
        coverage = float(min(1.0, len(recommended_items) / float(max(len(recommended_items), num_episodes * 10))))
    else:
        coverage = float(len(recommended_action_slots) / 50.0)

    results: dict[str, Any] = {
        "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "std_return": float(np.std(episode_returns)) if episode_returns else 0.0,
        "mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "hit_rate_step": float(np.mean(all_step_hits)) if all_step_hits else 0.0,
        "precision_at_10": float(np.mean(precisions_10)) if precisions_10 else 0.0,
        "recall_at_10": float(np.mean(recalls_10)) if recalls_10 else 0.0,
        "ndcg_at_10": float(np.mean(ndcgs_10)) if ndcgs_10 else 0.0,
        "map_at_10": float(np.mean(maps_10)) if maps_10 else 0.0,
        "hr_at_10": float(np.mean(hrs_10)) if hrs_10 else 0.0,
        "catalog_coverage": coverage,
        "mean_novelty_bits": float(np.mean(all_step_novelties)) if all_step_novelties else 0.0,
        "intra_list_diversity": float(np.mean(episode_ilds)) if episode_ilds else 0.0,
    }
    if return_per_episode:
        results["per_episode"] = per_episode
    return results
