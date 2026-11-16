"""
Tests for the Workshop 2 HybridNeuMF baseline inside the Workshop 3 simulator.

They use only versioned artifacts (``workshop3/data/w2_candidate_scores.npz``,
``workshop3/data/processed_cache.npz``, ``workshop3/report/metrics_summary.json`` and
``runs/evaluation_summary.json``), so they need neither the raw MovieLens CSVs nor the
Workshop 2 checkpoint and run in a few seconds.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from workshop3.src.baselines import GreedyHybridNeuMFPolicy, evaluate_policy
from workshop3.src.env import MovieLensRecEnv, resolve_repo_path
from workshop3.src.w2_scores import DEFAULT_SCORES_PATH, PADDING_SCORE, load_w2_candidate_scores

CACHE = resolve_repo_path("workshop3/data/processed_cache.npz")
SUMMARY = resolve_repo_path("workshop3/report/metrics_summary.json")
W2_SUMMARY = resolve_repo_path("runs/evaluation_summary.json")
PAIRS = (
    "D3QN_vs_MyopicBandit",
    "D3QN_vs_GreedyFunkSVD",
    "D3QN_vs_GreedyHybridNeuMF",
    "GreedyHybridNeuMF_vs_GreedyFunkSVD",
)


@pytest.fixture(scope="module")
def cache() -> dict:
    with np.load(CACHE) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture(scope="module")
def w2(cache: dict) -> dict:
    # The loader itself raises if the artifact is not aligned with the cache
    return load_w2_candidate_scores(DEFAULT_SCORES_PATH, cache_data=cache)


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def _policy(w2: dict) -> GreedyHybridNeuMFPolicy:
    return GreedyHybridNeuMFPolicy(w2["scores"], slot_valid=w2["cand_valid"])


def _rollout_actions(env: MovieLensRecEnv, policy: GreedyHybridNeuMFPolicy) -> list:
    actions = []
    for u in env.split_user_indices:
        obs, info = env.reset(seed=0, options={"user_idx": int(u)})
        done, episode = False, []
        while not done:
            action = policy.select_action(obs, info)
            assert info["valid_mask"][action] == 1.0, f"user {u}: slot {action} is visited or padding"
            obs, _, terminated, truncated, info = env.step(action)
            assert info["is_valid_slot"] == 1, f"user {u}: picked padding slot {action}"
            episode.append(int(action))
            done = terminated or truncated
        assert len(set(episode)) == len(episode), f"user {u}: repeated a slot {episode}"
        actions.append(episode)
    return actions


def test_artifact_is_aligned_with_the_cache(cache: dict, w2: dict) -> None:
    for key in ("cand_items", "cand_movie_ids", "cand_valid", "pool_user_ids"):
        np.testing.assert_array_equal(w2[key], cache[key], err_msg=key)
    valid = w2["cand_valid"] > 0
    assert np.all(w2["scores"][~valid] == PADDING_SCORE)
    real = w2["scores"][valid]
    assert np.all(np.isfinite(real)) and np.all(real > PADDING_SCORE) and real.std() > 0.0
    assert np.all(w2["w2_item_idx"][~valid] == -1) and np.all(w2["w2_item_idx"][valid] >= 0)
    assert np.all(w2["w2_user_idx"] >= 0)


def test_artifact_slots_are_held_out_for_workshop2(cache: dict, w2: dict) -> None:
    align = w2["meta"]["alignment"]
    n_valid = int(np.sum(cache["cand_valid"] > 0))
    assert align["valid_slots"] == align["valid_slots_mapped"] == n_valid
    assert align["pool_users"] == align["pool_users_mapped"] == len(cache["pool_user_ids"])
    assert align["cold_users"] == 0 and align["cold_item_slots"] == 0
    assert align["slots_with_rating_in_w2_train"] == 0 and align["no_session_rating_in_w2_train"] is True
    assert not w2["w2_cold_item"].any() and not w2["rating_in_w2_train"].any()
    for key in ("slots_in_w2_clean_val", "slots_in_w2_clean_test"):
        assert align[key]["rating_and_timestamp_match"] == align[key]["slots"], key
    assert align["slots_in_w2_clean_val"]["slots"] + align["slots_in_w2_clean_test"]["slots"] == n_valid

    # The artifact was built from a model that reproduced the official Workshop 2 test metrics
    official = json.loads(W2_SUMMARY.read_text(encoding="utf-8"))["hybrid_test_metrics"]
    for key, check in w2["meta"]["inference_check"]["test_metrics"].items():
        assert check["reproduced"] == pytest.approx(official[key], abs=1e-6), key


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_policy_never_picks_a_visited_or_padding_slot(w2: dict, split: str) -> None:
    env = MovieLensRecEnv(split=split, horizon=20, seed=0)
    _rollout_actions(env, _policy(w2))


def test_padding_is_excluded_even_when_the_mask_allows_it(w2: dict) -> None:
    policy = _policy(w2)
    valid = w2["cand_valid"] > 0
    padded_users = np.flatnonzero(~valid.all(axis=1))
    assert len(padded_users) > 0, "expected sessions shorter than 50 ratings"
    for u in padded_users[:20]:
        info = {"user_idx": int(u)}
        state = np.zeros(250, dtype=np.float32)
        # A permissive mask must not let a padding slot win
        assert valid[u, policy.select_action(state, info=info, valid_mask=np.ones(50, dtype=np.float32))]
        # With every real slot already recommended, the policy repeats a real slot
        only_padding_open = (~valid[u]).astype(np.float32)
        assert valid[u, policy.select_action(state, info=info, valid_mask=only_padding_open)]


def test_policy_is_deterministic(w2: dict) -> None:
    env = MovieLensRecEnv(split="test", horizon=10, seed=0)
    first = _rollout_actions(env, _policy(w2))
    second = _rollout_actions(env, _policy(w2))
    assert first == second
    # Greedy order equals the descending W2 score order over the real slots
    u = int(env.split_user_indices[0])
    real = np.flatnonzero(w2["cand_valid"][u] > 0)
    expected = real[np.argsort(-w2["scores"][u, real], kind="stable")][: len(first[0])]
    assert first[0] == [int(a) for a in expected]


def test_evaluate_policy_per_episode_records(w2: dict) -> None:
    env = MovieLensRecEnv(split="test", horizon=10, seed=0)
    users = [int(u) for u in env.split_user_indices]
    default = evaluate_policy(env, _policy(w2), seed=42, user_indices=users)
    detailed = evaluate_policy(env, _policy(w2), seed=42, user_indices=users, return_per_episode=True)
    assert "per_episode" not in default
    records = detailed.pop("per_episode")
    assert detailed == default
    assert [r["user_idx"] for r in records] == users
    assert set(records[0]) >= {"user_idx", "return", "ndcg_at_10", "precision_at_10", "hits"}
    assert np.mean([r["return"] for r in records]) == pytest.approx(default["mean_return"])
    assert np.mean([r["ndcg_at_10"] for r in records]) == pytest.approx(default["ndcg_at_10"])
    assert all(r["hits"] == sum(r["step_hits"]) for r in records)


def test_metrics_summary_reports_w2_static_block(summary: dict, cache: dict) -> None:
    w2_block = summary["workshop2_comparison"]
    assert "w2_hybrid_neumf_full_catalog" not in w2_block
    static = w2_block["w2_hybrid_neumf_static"]
    expected_keys = {
        "source",
        "protocol",
        "test_rmse",
        "test_mae",
        "roc_auc",
        "precision_at_10",
        "recall_at_10",
        "ndcg_at_10",
        "map_at_10",
        "hr_at_10",
    }
    assert expected_keys <= set(static)
    official = json.loads(W2_SUMMARY.read_text(encoding="utf-8"))
    ranking = official["ranking_comparison"]["HybridNeuMF (W2 Deep CF)"]
    for key, value in (
        ("test_rmse", official["hybrid_test_metrics"]["RMSE"]),
        ("test_mae", official["hybrid_test_metrics"]["MAE"]),
        ("roc_auc", official["hybrid_test_metrics"]["roc_auc"]),
        ("precision_at_10", ranking["P@10"]),
        ("ndcg_at_10", ranking["NDCG@10"]),
        ("hr_at_10", ranking["HitRate@10"]),
    ):
        assert static[key] == pytest.approx(round(value, 4), abs=1e-9), key
    protocol = static["protocol"].lower()
    assert "sampled-candidate" in protocol and "200 sampled unrated movies" in protocol
    assert "not a full-catalog" in protocol
    assert "w3_sequential_greedy_hybrid_neumf" in w2_block
    assert summary["num_test_users"] == len(cache["test_user_indices"])


def test_metrics_summary_has_the_paired_comparisons(summary: dict) -> None:
    paired = summary["paired_comparisons"]
    for pair in PAIRS:
        assert pair in paired, pair
        for metric in ("return", "ndcg_at_10"):
            stats = paired[pair][metric]
            assert stats["n_users"] == summary["num_test_users"]
            assert stats["ci95_low"] <= stats["mean_diff"] <= stats["ci95_high"], (pair, metric)
            assert 0.0 <= stats["wilcoxon_p_value"] <= 1.0
            # The per-user means must reproduce the K=10 table of the same policies
            a, b = pair.split("_vs_")
            table_key = "mean_return" if metric == "return" else "ndcg_at_10"
            assert stats["mean_a"] == pytest.approx(summary["test_k10"][a][table_key], abs=1e-4)
            assert stats["mean_b"] == pytest.approx(summary["test_k10"][b][table_key], abs=1e-4)
    assert summary["agents"]["MyopicBandit"]["gamma"] == 0.0
