# MovieLens 32M Recommendation System — Workshops 1, 2, 3 & 4

Machine Learning course project (Team 7, Universidad Distrital Francisco José de Caldas):
- **Workshop 1:** Exploratory data analysis (EDA), leak-free chronological splitting, feature engineering, and classical supervised baselines (Popularity, Bias, Item-Item kNN, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, and Logistic Regression).
- **Workshop 2:** Deep Collaborative Filtering (`HybridNeuMF` + `BiasedMatrixFactorization`) in PyTorch with two-phase Transfer Learning & Fine-Tuning, TensorBoard experiment tracking, held-out test set evaluation, and checkpoint verification.
- **Workshop 3:** Sequential recommendation as a Markov Decision Process: a `Gymnasium` environment built from real post-split sessions and a Dueling Double DQN compared with the Workshop 1 and Workshop 2 models acting in the same environment (see [Workshop 3](#workshop-3--sequential-recommendation-with-a-dueling-double-dqn)).
- **Workshop 4:** Unsupervised Representation Learning with Standard Autoencoders (`AE`), Variational Autoencoders (`beta-VAE` with $\beta \in \{0.5, 1.0, 4.0\}$ and KL annealing), and Denoising Autoencoders (`DAE`) for reconstruction analysis, anomaly detection, and 2D latent manifold comparison (`t-SNE`, `UMAP`, and Silhouette scores).

---

## Repository Structure

```text
Machine_Learning/
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # W1/W2/W4 loading & temporal split + W3 RL session cache builder (--w3)
│   ├── features.py        # W1/W2 leak-free feature engineering (user/item stats, biases, genres, movie age)
│   ├── baselines.py       # W1 classical/FunkSVD baselines and W3 sequential RL policies + evaluate_policy
│   ├── model.py           # W2 PyTorch architectures (BiasedMatrixFactorization & HybridNeuMF)
│   ├── train.py           # W2 HybridNeuMF training (--w2) and W3 D3QN training pipeline (default / --w3)
│   ├── evaluate.py        # W2 evaluation (--w2), W2 slot scoring (--build-w2-scores), and W3 RL evaluation (--w3)
│   ├── test_model.py      # W2 checkpoint loader (checkpoints/model_best.pt) and inference script
│   ├── env.py             # W3 MovieLensRecEnv (Gymnasium) and 100-episode random-policy diagnostics
│   ├── agent.py           # W3 Dueling Double DQN (flat and slot-shared networks) and replay buffer
│   ├── autoencoder.py     # W4 architectures (StandardAutoencoder, VariationalAutoencoder, DenoisingAE)
│   ├── train_ae.py        # W4 training pipeline (z_dim ablation, AE, DAE, and beta-VAE sweep)
│   ├── evaluate_ae.py     # W4 reconstruction quality, sample panels, and anomaly detection ROC-AUC
│   └── analyze_latent.py  # W4 latent space visualization (t-SNE, UMAP) and Silhouette analysis
├── notebooks/
│   ├── eda.ipynb          # W1 EDA notebook with visualizations and statistical analysis
│   └── w3_reinforcement_learning.ipynb  # W3 read-only walkthrough loading JSON results and replaying a session
├── data/
│   ├── splits.json        # W1/W2/W4 reproducible train/validation/test split indices and metadata (SEED = 42)
│   ├── dataset_stats.json             # W3 cohort, split, and session statistics
│   ├── movie_lookup.json              # W3 titles and genres of every candidate movie
│   ├── processed_cache.npz            # W3 RL sessions, slates, and train-split FunkSVD parameters
│   └── w2_candidate_scores.npz        # W3 HybridNeuMF scores of every candidate slot
├── dataSet/               # Raw MovieLens 32M CSV files (or mounted in Colab)
├── checkpoints/
│   ├── model_best.pt      # W2 best HybridNeuMF checkpoint
│   ├── agent_best.pt      # W3 champion D3QN (best validation checkpoint)
│   ├── bandit_gamma0.pt   # W3 myopic ablation (gamma = 0)
│   ├── ae_best.pt         # W4 best Standard & Denoising Autoencoder checkpoint
│   └── vae_best.pt        # W4 best Variational Autoencoder (beta-VAE) checkpoint
├── runs/
│   ├── hybrid_neumf_experiment/       # W2 TensorBoard logs
│   ├── workshop3/                     # W3 TensorBoard logs, JSON metrics, and Figures 2-7
│   └── workshop4/                     # W4 TensorBoard logs, learning curves, ROC, and t-SNE/UMAP plots
├── requirements.txt       # Pinned virtual environment dependencies
└── README.md
```

---

## Dataset Setup (Local & Google Colab / Drive)

**MovieLens 32M** (GroupLens Research, University of Minnesota).
- **Local setup:** Extract `ml-32m.zip` into `dataSet/` (`ratings.csv`, `movies.csv`, `tags.csv`, `links.csv`).
- **Google Colab + Google Drive setup:** `src/data_loader.py` automatically resolves `/content/drive/MyDrive/ml-32m` when Google Drive is mounted in Colab:

```python
from google.colab import drive
drive.mount("/content/drive")
```

---

## Workshop 4 — Reproduction Commands (Autoencoders & VAEs)

1. **Verify AE and VAE architectures (`src/autoencoder.py`):**

```bash
.venv/bin/python -m src.autoencoder
```

2. **Train Standard AE (`z_dim` $\in \{8, 16, 32\}$), Denoising AE, and $\beta$-VAE ($\beta \in \{0.5, 1.0, 4.0\}$) (`src/train_ae.py`):**

```bash
.venv/bin/python -m src.train_ae
```

Saves `checkpoints/ae_best.pt`, `checkpoints/vae_best.pt`, and learning curves in `runs/workshop4/w4_learning_curves.png`.

3. **Evaluate Reconstruction Quality and Unsupervised Anomaly Detection (`src/evaluate_ae.py`):**

```bash
.venv/bin/python -m src.evaluate_ae
```

Generates `runs/workshop4/reconstruction_samples.png`, `runs/workshop4/anomaly_detection_roc.png`, and the cumulative comparison table across Workshops 1, 2, and 4 (`runs/workshop4/cumulative_comparison.csv`).

4. **Analyze Latent Space with t-SNE, UMAP, and Silhouette Scores (`src/analyze_latent.py`):**

```bash
.venv/bin/python -m src.analyze_latent
```

Generates `runs/workshop4/latent_space_tsne_umap.png` and `runs/workshop4/latent_separability_metrics.csv`.

---

## Workshop 2 — Reproduction Commands (Deep Collaborative Filtering)

```bash
.venv/bin/python -m src.model
.venv/bin/python -m src.train --w2
.venv/bin/python -m src.evaluate --w2
.venv/bin/python -m src.test_model
```

---

## Workshop 1 — Reproduction Commands (EDA & Baselines)

```bash
.venv/bin/python -m src.data_loader
.venv/bin/python -m src.baselines
```

   Add `--funksvd` to also train the biased matrix factorization model (FunkSVD, k = 50) whose figures Workshop 3 cites (slow: pure-Python SGD over the full training split).

---

## Workshop 3 — Sequential Recommendation with a Dueling Double DQN

**Repository**: https://github.com/AndresRuiz14502/Machine_Learning

Workshop 1 (FunkSVD) and Workshop 2 (HybridNeuMF) rank movies once per user and never react to feedback. Workshop 3 casts top-K recommendation as a finite-horizon Markov Decision Process, implements it as a `Gymnasium` environment (`MovieLensRecEnv` in `src/env.py`), and trains a Dueling Double Deep Q-Network (`src/agent.py`) against the Workshop 1 and Workshop 2 models acting inside the same environment.

- **State** (`Box(250)`): for each of the 50 candidate slots, the cosine similarity between the candidate's FunkSVD item factor and the session vector, the static FunkSVD score, self-information novelty, and an unavailable-slot flag; plus the user's FunkSVD factor, the session vector, and ten progress scalars. The session vector starts at the user's factor and moves toward liked items and away from missed ones.
- **Actions** (`Discrete(50)`): one candidate slot per step; already recommended and padding slots are masked.
- **Reward**: a hit (real rating ≥ 4.0) earns `1 + 0.25 (r − 4)` plus novelty and genre-diversity bonuses; a miss earns `−0.4`; a repeated or padding slot costs `−1.5`; the user abandons after five consecutive misses.
- **Agent & Leak-free protocol**: Reuses the Workshop 1 sample of MovieLens 32M (32,000,204 ratings, 200,948 users, 87,585 movies → 15,000 sampled users → 2,368,980 ratings, 14,998 users, 11,589 movies) and the global chronological 80/10/10 split (1,895,184 training ratings). RL train/validation sessions come from the validation period (227 RL train users, 57 RL validation users); RL test sessions come from the test period (177 users). The champion checkpoint is `checkpoints/agent_best.pt` and the γ = 0 ablation is `checkpoints/bandit_gamma0.pt`.

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

```bash
# Install dependencies (see requirements.txt)
.venv/bin/pip install -r requirements.txt

# Rebuild processed cache and Workshop 2 candidate scores (requires raw CSVs and checkpoints/model_best.pt)
.venv/bin/python -m src.data_loader --w3
.venv/bin/python -m src.evaluate --build-w2-scores

# Random-policy validation (runs/workshop3/fig2_random_policy_diagnostics.png & runs/workshop3/random_diagnostics.json)
.venv/bin/python src/env.py

# Full training: 8 configurations x 2 architectures, champion, gamma = 0 ablation
.venv/bin/python src/train.py
.venv/bin/tensorboard --logdir runs/workshop3

# Evaluate the versioned checkpoints and regenerate Figures 3 to 7 in runs/workshop3/
.venv/bin/python src/evaluate.py
```
