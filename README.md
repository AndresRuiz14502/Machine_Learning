# MovieLens 32M Recommendation System — Workshops 1, 2, 3 & 4

Machine Learning course project (Team 7, Universidad Distrital Francisco José de Caldas):
- **Workshop 1:** Exploratory data analysis (EDA), leak-free chronological splitting, feature engineering, and classical supervised baselines (Popularity, Bias, Item-Item kNN, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, and Logistic Regression).
- **Workshop 2:** Deep Collaborative Filtering (`HybridNeuMF` + `BiasedMatrixFactorization`) in PyTorch with two-phase Transfer Learning & Fine-Tuning, TensorBoard experiment tracking, held-out test set evaluation, and checkpoint verification.
- **Workshop 3:** Sequential Recommendation as a Markov Decision Process (`MovieLensRecEnv` in `Gymnasium`) with a Dueling Double Deep Q-Network (`D3QN`), hyperparameter/architecture search (flat MLP vs. slot-shared), a myopic `gamma = 0` ablation, and held-out test evaluation against Workshop 1 (FunkSVD) and Workshop 2 (HybridNeuMF) policies.
- **Workshop 4:** Unsupervised Representation Learning with Standard Autoencoders (`AE`), Variational Autoencoders (`beta-VAE` with $\beta \in \{0.5, 1.0, 4.0\}$ and KL annealing), and Denoising Autoencoders (`DAE`) for reconstruction analysis, anomaly detection, and 2D latent manifold comparison (`t-SNE`, `UMAP`, and Silhouette scores).

---

## Repository Structure

```text
Machine_Learning/
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Loading, subsampling, temporal split (Local & Colab/Drive) + W3 RL cache (--w3)
│   ├── features.py        # Leak-free feature engineering (user/item stats, biases, genres, movie age)
│   ├── baselines.py       # Workshop 1 baselines (regression, classification, ranking) + W3 RL policies
│   ├── model.py           # Workshop 2 PyTorch architectures (BiasedMatrixFactorization & HybridNeuMF)
│   ├── train.py           # Workshop 2 HybridNeuMF training (--w2) and Workshop 3 D3QN training (--w3)
│   ├── evaluate.py        # Workshop 2 evaluation (--w2), W2 slot scoring (--build-w2-scores), and W3 evaluation (--w3)
│   ├── test_model.py      # Workshop 2 checkpoint loader (checkpoints/model_best.pt) and inference script
│   ├── env.py             # Workshop 3 Gymnasium environment (MovieLensRecEnv) and random-policy diagnostics
│   ├── agent.py           # Workshop 3 Dueling Double DQN (flat and slot-shared) and replay buffer
│   ├── autoencoder.py     # Workshop 4 architectures (StandardAutoencoder, VariationalAutoencoder, DenoisingAE)
│   ├── train_ae.py        # Workshop 4 training pipeline (z_dim ablation, AE, DAE, and beta-VAE sweep)
│   ├── evaluate_ae.py     # Workshop 4 reconstruction quality, sample panels, and anomaly detection ROC-AUC
│   └── analyze_latent.py  # Workshop 4 latent space visualization (t-SNE, UMAP) and Silhouette analysis
├── notebooks/
│   ├── eda.ipynb                          # Workshop 1 EDA notebook with visualizations and statistical analysis
│   └── w3_reinforcement_learning.ipynb    # Workshop 3 read-only results and trajectory walkthrough
├── data/
│   ├── splits.json                # Reproducible train/validation/test split indices and metadata (SEED = 42)
│   ├── dataset_stats.json         # Workshop 3 cohort, split, and RL session statistics
│   ├── movie_lookup.json          # Workshop 3 movie titles and genres lookup
│   ├── processed_cache.npz        # Workshop 3 RL sessions, candidate slates, and FunkSVD features
│   └── w2_candidate_scores.npz    # Workshop 2 HybridNeuMF scores for Workshop 3 candidate slots
├── dataSet/               # Raw MovieLens 32M CSV files (or mounted from Google Drive at /content/drive/MyDrive/ml-32m)
├── checkpoints/
│   ├── model_best.pt      # Workshop 2 best HybridNeuMF checkpoint
│   ├── agent_best.pt      # Workshop 3 best Dueling Double DQN checkpoint
│   ├── bandit_gamma0.pt   # Workshop 3 myopic ablation checkpoint (gamma = 0)
│   ├── ae_best.pt         # Workshop 4 best Standard & Denoising Autoencoder checkpoint
│   └── vae_best.pt        # Workshop 4 best Variational Autoencoder (beta-VAE) checkpoint
├── runs/
│   ├── ...                # Workshop 1 & 2 logs, tables, and diagnostic plots
│   ├── workshop3/         # Workshop 3 TensorBoard logs, metrics JSONs, and diagnostic/evaluation figures
│   └── workshop4/         # Workshop 4 TensorBoard logs, learning curves, reconstructions, ROC, and t-SNE/UMAP plots
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

## Workshop 3 — Reproduction Commands (Reinforcement Learning — D3QN)

1. **Build the Workshop 3 RL session cache and Workshop 2 candidate scores (`src/data_loader.py`, `src/evaluate.py`):**

```bash
.venv/bin/python -m src.data_loader --w3
.venv/bin/python -m src.evaluate --build-w2-scores
```

Saves `data/processed_cache.npz`, `data/movie_lookup.json`, `data/dataset_stats.json`, and `data/w2_candidate_scores.npz`.

2. **Validate `MovieLensRecEnv` with 100 random-policy episodes (`src/env.py`):**

```bash
.venv/bin/python -m src.env
```

Generates `runs/workshop3/random_diagnostics.json` and `runs/workshop3/fig2_random_policy_diagnostics.png`.

3. **Train D3QN grid search, champion, and $\gamma = 0$ ablation (`src/train.py`):**

```bash
.venv/bin/python -m src.train --w3
```

Saves `checkpoints/agent_best.pt`, `checkpoints/bandit_gamma0.pt`, TensorBoard logs in `runs/workshop3/`, and `runs/workshop3/training_logs.json`.

4. **Evaluate on held-out test sessions ($K \in \{10, 20\}$) against Workshop 1/2 baselines and generate Figures 3–7 (`src/evaluate.py`):**

```bash
.venv/bin/python -m src.evaluate --w3
```

Generates `runs/workshop3/metrics_summary.json` and `runs/workshop3/fig3_*.png` through `runs/workshop3/fig7_*.png`.

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

Add `--funksvd` to `src.baselines` to also train the biased matrix factorization model (`FunkSVD`, $k = 50$).
