# MovieLens 32M Recommendation System — Workshops 1, 2, 3 & 4

Machine Learning course project (Team 7, Universidad Distrital Francisco José de Caldas):
- **Workshop 1:** Exploratory data analysis (EDA), leak-free chronological splitting, feature engineering, and classical supervised baselines (Popularity, Bias, Item-Item kNN, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, and Logistic Regression).
- **Workshop 2:** Deep Collaborative Filtering (`HybridNeuMF` + `BiasedMatrixFactorization`) in PyTorch with two-phase Transfer Learning & Fine-Tuning, TensorBoard experiment tracking, held-out test set evaluation, and checkpoint verification.
- **Workshop 3:** Sequential recommendation as a Markov Decision Process: a `Gymnasium` environment built from real post-split sessions and a Dueling Double DQN compared with the Workshop 1 and Workshop 2 models acting in the same environment (see [`workshop3/README.md`](workshop3/README.md)).
- **Workshop 4:** Unsupervised Representation Learning with Standard Autoencoders (`AE`), Variational Autoencoders (`beta-VAE` with $\beta \in \{0.5, 1.0, 4.0\}$ and KL annealing), and Denoising Autoencoders (`DAE`) for reconstruction analysis, anomaly detection, and 2D latent manifold comparison (`t-SNE`, `UMAP`, and Silhouette scores).

---

## Repository Structure

```text
Machine_Learning/
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Loading, subsampling, min-support filtering, contiguous IDs, and temporal split (Local & Colab/Drive)
│   ├── features.py        # Leak-free feature engineering (user/item stats, biases, genres, movie age)
│   ├── baselines.py       # Workshop 1 baseline models (regression, binary classification, and ranking)
│   ├── model.py           # Workshop 2 PyTorch architectures (BiasedMatrixFactorization & HybridNeuMF)
│   ├── train.py           # Workshop 2 training pipeline (pretraining, 2-phase fine-tuning, TensorBoard logs)
│   ├── evaluate.py        # Workshop 2 held-out test set evaluation & comparison vs. Workshop 1 baselines
│   ├── test_model.py      # Workshop 2 checkpoint loader (checkpoints/model_best.pt) and inference script
│   ├── autoencoder.py     # Workshop 4 architectures (StandardAutoencoder, VariationalAutoencoder, DenoisingAE)
│   ├── train_ae.py        # Workshop 4 training pipeline (z_dim ablation, AE, DAE, and beta-VAE sweep)
│   ├── evaluate_ae.py     # Workshop 4 reconstruction quality, sample panels, and anomaly detection ROC-AUC
│   └── analyze_latent.py  # Workshop 4 latent space visualization (t-SNE, UMAP) and Silhouette analysis
├── notebooks/
│   └── eda.ipynb          # Workshop 1 EDA notebook with visualizations and statistical analysis
├── data/
│   └── splits.json        # Reproducible train/validation/test split indices and metadata (SEED = 42)
├── dataSet/               # Raw MovieLens 32M CSV files (or mounted from Google Drive at /content/drive/MyDrive/ml-32m)
├── checkpoints/
│   ├── model_best.pt      # Workshop 2 best HybridNeuMF checkpoint
│   ├── ae_best.pt         # Workshop 4 best Standard & Denoising Autoencoder checkpoint
│   └── vae_best.pt        # Workshop 4 best Variational Autoencoder (beta-VAE) checkpoint
├── runs/
│   ├── ...                # Workshop 1 & 2 logs, tables, and diagnostic plots
│   └── workshop4/         # Workshop 4 TensorBoard logs, learning curves, reconstructions, ROC, and t-SNE/UMAP plots
├── workshop3/             # Workshop 3: Gymnasium MDP, Dueling Double DQN, report (see workshop3/README.md)
├── group7_workshop2.pdf   # Workshop 2 Technical Report (PDF)
├── group7_workshop4.pdf   # Workshop 4 Technical Report (PDF)
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
.venv/bin/python -m src.train
.venv/bin/python -m src.evaluate
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

Workshop 3 lives in [`workshop3/`](workshop3/README.md), which documents the MDP, the leak-free session protocol, and every command. RL sessions are real ratings recorded after the Workshop 1/2 training cut (validation period for RL training, test period for RL testing), so the agent and the Workshop 1/2 models are compared on the same held-out future. The champion checkpoint is `workshop3/checkpoints/agent_best.pt`.

<!-- w3-results:start -->

Held-out test sessions (177 users, one episode each, K = 10; coverage also at K = 20). Full tables and paired statistics: `workshop3/README.md`.

| Policy | Return (mean ± SD) | P@10 | NDCG@10 | MAP@10 | Coverage@10 | Coverage@20 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| D3QN (γ = 0.95, slot-shared) | 6.4798 ± 4.38 | 0.6480 | 0.6767 | 0.5644 | 27.79% | 46.86% |
| Myopic DQN (γ = 0) | 6.3310 ± 4.22 | 0.6379 | 0.6672 | 0.5522 | 27.43% | 47.16% |
| Greedy FunkSVD (W1) | 6.5536 ± 4.12 | 0.6565 | 0.6972 | 0.5826 | 27.04% | 45.67% |
| Greedy HybridNeuMF (W2) | 6.5495 ± 4.07 | 0.6576 | 0.7088 | 0.5906 | 25.35% | 43.23% |

<!-- w3-results:end -->

```powershell
.\.venv\Scripts\python.exe -m pip install -r workshop3\requirements.txt
.\.venv\Scripts\python.exe -m pytest workshop3\tests -v      # environment, leakage, baselines, docs
.\.venv\Scripts\python.exe workshop3\src\evaluate.py         # test evaluation and Figures 3-7
.\.venv\Scripts\python.exe workshop3\src\train.py            # full training (about 30 min on a GPU)
```
