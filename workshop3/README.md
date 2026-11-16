# Workshop 3 — Sequential Movie Recommendation with Deep Q-Learning on MovieLens 32M

**Course**: Machine Learning (2026-III) — Universidad Distrital Francisco José de Caldas  
**Professor**: Eng. Carlos Andrés Sierra, M.Sc. (`cavirguezs@udistrital.edu.co`)  
**Team 7**:
- **Andrés Felipe Ruiz Vasallo** (`afruizv@udistrital.edu.co`)
- **Josep Emanuel León Joya** (`jeleonj@udistrital.edu.co`)
- **Matthew Stev Toro Mondragón** (`mstorom@udistrital.edu.co`)

📄 **IEEE report**: [`group7_workshop3.pdf`](./group7_workshop3.pdf) · **Repository**: https://github.com/AndresRuiz14502/Machine_Learning

---

## 1. Overview

Workshop 1 (FunkSVD) and Workshop 2 (HybridNeuMF, the official version on `main`) rank movies once per user and never react to feedback. Workshop 3 asks whether a sequential decision-maker that adapts to in-session feedback recommends better. We cast top-K recommendation as a finite-horizon Markov Decision Process, implement it as a `Gymnasium` environment (`MovieLensRecEnv`), and train a Dueling Double Deep Q-Network (D3QN) against the Workshop 1 and Workshop 2 models acting inside the same environment.

- **State** (`Box(250)`): for each of the 50 candidate slots, the cosine similarity between the candidate's FunkSVD item factor and the session vector, the static FunkSVD score, self-information novelty, and an unavailable-slot flag; plus the user's FunkSVD factor, the session vector, and ten progress scalars. The session vector starts at the user's factor and moves toward liked items and away from missed ones.
- **Actions** (`Discrete(50)`): one candidate slot per step; already recommended and padding slots are masked.
- **Reward**: a hit (real rating ≥ 4.0) earns `1 + 0.25 (r − 4)` plus novelty and genre-diversity bonuses; a miss earns `−0.4`; a repeated or padding slot costs `−1.5`; the user abandons after five consecutive misses.
- **Agent**: Double DQN with a dueling head, replay buffer, Polyak target network, Huber loss, gradient clipping, and ε-greedy decay. Two Q-network architectures are searched: a flat MLP over the whole state and a slot-shared network that scores every slot with one MLP. A copy of the champion with γ = 0 (a neural contextual bandit) isolates the value of multi-step planning.

## 2. Leak-free protocol

Workshop 3 reuses the Workshop 1 sample of MovieLens 32M (32,000,204 ratings, 200,948 users, 87,585 movies → 15,000 sampled users → 2,368,980 ratings, 14,998 users, 11,589 movies) and the global chronological 80/10/10 split shared by Workshops 1 and 2. FunkSVD (k = 20) is trained on the 1,895,184 training ratings only, so every observation feature predates every RL session.

- **Sessions**: the chronological real ratings of a warm user inside one later period. RL train and validation sessions come from the validation period (227 RL train users, 57 RL validation users); RL test sessions come from the test period (177 users), the same held-out future that Workshops 1 and 2 report on.
- **Candidate slate**: the first 50 ratings of the session, chosen by timestamp only and ordered by the static FunkSVD score; shorter sessions are padded with masked slots.
- **Labels**: a like is a real rating ≥ 4.0. No label, future rating, or synthetic preference enters the observation.

> **Integrity note.** An earlier version of this workshop built the candidate vectors from the user's future likes and used synthetic labels. The observation then revealed the answer (AUC = 1.0 for liked vs. not liked), an untrained cosine rule reached a 100% hit rate, and the training script added hand-made penalties that decided the hyperparameter ranking. All of it was removed; the simulator was rebuilt from real post-cut feedback, the agent is trained with the standard Double DQN update, and `tests/test_leakage.py` guards against the regression.

## 3. Results

Every policy is evaluated once on every RL test session. Greedy FunkSVD (W1) and Greedy HybridNeuMF (W2) are the supervised models of the previous workshops acting greedily inside the same environment.

<!-- results:start -->

**K = 10** (177 held-out test sessions, one episode each)

| Policy | Return (mean ± SD) | P@10 | R@10 | NDCG@10 | MAP@10 | HR@10 | Coverage | Novelty (bits) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Random | 3.4839 ± 4.12 | 0.4497 | 0.2249 | 0.4706 | 0.3330 | 0.9040 | 34.10% | 13.16 |
| Popularity | 4.1500 ± 4.12 | 0.5079 | 0.2667 | 0.5335 | 0.3993 | 0.8983 | 21.01% | 10.50 |
| Bias (μ + b_u + b_i) | 6.2347 ± 4.15 | 0.6350 | 0.3454 | 0.6622 | 0.5425 | 0.9379 | 22.72% | 12.67 |
| Greedy FunkSVD (W1) | 6.5536 ± 4.12 | 0.6565 | 0.3613 | 0.6972 | 0.5826 | 0.9435 | 27.04% | 12.50 |
| Greedy HybridNeuMF (W2) | 6.5495 ± 4.07 | 0.6576 | 0.3756 | 0.7088 | 0.5906 | 0.9661 | 25.35% | 12.32 |
| Myopic DQN (γ = 0) | 6.3310 ± 4.22 | 0.6379 | 0.3424 | 0.6672 | 0.5522 | 0.9379 | 27.43% | 12.58 |
| D3QN (γ = 0.95, slot-shared) | 6.4798 ± 4.38 | 0.6480 | 0.3448 | 0.6767 | 0.5644 | 0.9322 | 27.79% | 12.77 |

**K = 20** (177 held-out test sessions, one episode each)

| Policy | Return (mean ± SD) | P@20 | R@20 | NDCG@20 | MAP@20 | HR@20 | Coverage | Novelty (bits) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Random | 6.6112 ± 7.49 | 0.4119 | 0.4028 | 0.4723 | 0.3254 | 0.8814 | 50.51% | 13.07 |
| Popularity | 7.2497 ± 7.44 | 0.4486 | 0.4410 | 0.5259 | 0.3765 | 0.8983 | 37.59% | 11.24 |
| Bias (μ + b_u + b_i) | 10.4907 ± 7.44 | 0.5613 | 0.5721 | 0.6621 | 0.5216 | 0.9379 | 42.04% | 12.85 |
| Greedy FunkSVD (W1) | 10.9680 ± 7.67 | 0.5749 | 0.5903 | 0.6926 | 0.5591 | 0.9435 | 45.67% | 12.80 |
| Greedy HybridNeuMF (W2) | 10.8601 ± 7.63 | 0.5701 | 0.5942 | 0.6985 | 0.5626 | 0.9661 | 43.23% | 12.61 |
| Myopic DQN (γ = 0) | 10.8314 ± 7.73 | 0.5692 | 0.5788 | 0.6720 | 0.5368 | 0.9435 | 47.16% | 12.92 |
| D3QN (γ = 0.95, slot-shared) | 10.8422 ± 7.76 | 0.5695 | 0.5709 | 0.6719 | 0.5389 | 0.9379 | 46.86% | 12.96 |

**Static references (different protocol).** Workshop 1 and Workshop 2 report one-shot rankings in which each test user's liked items compete with sampled unrated movies, so these numbers give context and are not comparable one-to-one with the sessions above:

| Model (own report) | Test RMSE | P@10 | NDCG@10 | HR@10 |
| :--- | :---: | :---: | :---: | :---: |
| FunkSVD (Workshop 1; RMSE for k = 50) | 0.8041 | 0.3260 | 0.5881 | 0.8043 |
| HybridNeuMF (Workshop 2) | 0.7994 | 0.4618 | 0.7663 | 0.9266 |

**Paired per-user differences at K = 10** (A − B; paired bootstrap 95% CI, Wilcoxon signed-rank p-value)

| A − B | Return diff [95% CI] | p | NDCG@10 diff [95% CI] | p |
| :--- | :---: | :---: | :---: | :---: |
| D3QN (γ = 0.95, slot-shared) − Myopic DQN (γ = 0) | 0.1488 [-0.0468, 0.3394] | 0.0553 | 0.0095 [-0.0070, 0.0251] | 0.1340 |
| D3QN (γ = 0.95, slot-shared) − Greedy FunkSVD (W1) | -0.0737 [-0.2765, 0.1315] | 0.6750 | -0.0205 [-0.0393, -0.0044] | 0.0404 |
| D3QN (γ = 0.95, slot-shared) − Greedy HybridNeuMF (W2) | -0.0697 [-0.3711, 0.2226] | 0.4321 | -0.0321 [-0.0595, -0.0072] | 0.1970 |
| Greedy HybridNeuMF (W2) − Greedy FunkSVD (W1) | -0.0040 [-0.2479, 0.2382] | 0.6023 | 0.0116 [-0.0076, 0.0332] | 0.8319 |

<!-- results:end -->

## 4. Repository structure (`workshop3/`)

```text
workshop3/
├── README.md                          # This guide
├── requirements.txt                   # Pinned direct dependencies (UTF-8)
├── group7_workshop3.pdf               # Compiled IEEE report (single copy)
├── checkpoints/
│   ├── agent_best.pt                  # Champion D3QN (best validation checkpoint)
│   └── bandit_gamma0.pt               # Same agent trained with gamma = 0 (myopic ablation)
├── data/
│   ├── dataset_stats.json             # Cohort, split, and session statistics
│   ├── movie_lookup.json              # Titles and genres of every candidate movie
│   ├── processed_cache.npz            # RL sessions, slates, and train-split FunkSVD parameters
│   ├── w2_candidate_scores.npz        # Workshop 2 HybridNeuMF scores of every candidate slot
│   └── ml-32m/                        # Optional fallback for the raw CSVs (git-ignored)
├── notebooks/
│   └── w3_reinforcement_learning.ipynb  # Read-only walkthrough that loads the JSON results
├── report/
│   ├── main.tex                       # IEEEtran master file
│   ├── bibliography.bib               # References
│   ├── sections/                      # LaTeX sections
│   ├── tables/                        # Tables rendered from metrics_summary.json
│   ├── figures/                       # 300-DPI figures (fig2 to fig7)
│   ├── metrics_summary.json           # Test, validation, paired-statistics, W1/W2 references
│   ├── random_diagnostics.json        # 100-episode random-policy validation
│   └── training_logs.json             # Curves of every training run
├── runs/
│   └── workshop3/                     # TensorBoard logs of every training run
├── src/
│   ├── __init__.py
│   ├── data_prep.py                   # Split, FunkSVD, and RL session construction
│   ├── env.py                         # MovieLensRecEnv (Gymnasium) and random-policy diagnostics
│   ├── agent.py                       # Dueling Double DQN (flat and slot-shared networks), replay buffer
│   ├── baselines.py                   # Random, Popularity, Bias, Greedy FunkSVD, Greedy HybridNeuMF, evaluation
│   ├── w2_scores.py                   # Scores every slot with the official Workshop 2 checkpoint
│   ├── train.py                       # Architecture x hyperparameter search, champion, gamma = 0 ablation
│   ├── evaluate.py                    # Test evaluation, paired statistics, Figures 3 to 7
│   └── render_tables.py               # Writes the README and LaTeX result tables from the JSON files
├── tests/                             # Environment, leakage, W2 baseline, FunkSVD, notebook, docs tests
└── tools/                             # Portable Tectonic for the report (local only, git-ignored)
```

## 5. Reproducibility (Windows PowerShell, from the repository root)

```powershell
# Environment (Python 3.12). For the CUDA build of torch, install it first:
#   .\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r workshop3\requirements.txt

# Tests (no raw CSVs needed)
.\.venv\Scripts\python.exe -m pytest workshop3\tests -v

# Random-policy validation (Figure 2 and report\random_diagnostics.json)
.\.venv\Scripts\python.exe workshop3\src\env.py

# Evaluate the versioned checkpoints, regenerate Figures 3 to 7 and the tables
.\.venv\Scripts\python.exe workshop3\src\evaluate.py
.\.venv\Scripts\python.exe workshop3\src\render_tables.py

# Full training: 8 configurations x 2 architectures, champion, gamma = 0 ablation (about 30 min on a GPU)
.\.venv\Scripts\python.exe workshop3\src\train.py
.\.venv\Scripts\tensorboard.exe --logdir workshop3\runs\workshop3
```

Rebuilding the data needs the raw MovieLens 32M CSVs in `dataSet\` (see the root `README.md`), and rebuilding the Workshop 2 scores also needs `checkpoints\model_best.pt` from `main`:

```powershell
.\.venv\Scripts\python.exe workshop3\src\data_prep.py
.\.venv\Scripts\python.exe workshop3\src\w2_scores.py
```

To compile the report, place a portable `tectonic.exe` in `workshop3\tools\` and run:

```powershell
workshop3\tools\tectonic.exe workshop3\report\main.tex
Copy-Item workshop3\report\main.pdf workshop3\group7_workshop3.pdf
```
