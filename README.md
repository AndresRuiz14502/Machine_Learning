# MovieLens 32M Recommendation System — Workshops 1, 2 & 4

Machine Learning course project (Team 7, Universidad Distrital Francisco José de Caldas):
- **Workshop 1:** Exploratory data analysis (EDA), leak-free chronological splitting, feature engineering, and classical supervised baselines (Popularity, Bias, Item-Item kNN, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, and Logistic Regression).
- **Workshop 2:** Deep Collaborative Filtering (`HybridNeuMF` + `BiasedMatrixFactorization`) in PyTorch with two-phase Transfer Learning & Fine-Tuning, TensorBoard experiment tracking, held-out test set evaluation, and checkpoint verification.
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
