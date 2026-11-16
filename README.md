# MovieLens 32M Recommendation System — Workshops 1, 2 & 3

Machine Learning course project (Team 7, Universidad Distrital Francisco José de Caldas):
- **Workshop 1:** Exploratory data analysis (EDA), leak-free chronological splitting, feature engineering, and classical supervised baselines (Popularity, Bias, Item-Item kNN, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, and Logistic Regression).
- **Workshop 2:** Deep Collaborative Filtering (`HybridNeuMF` + `BiasedMatrixFactorization`) in PyTorch with two-phase Transfer Learning & Fine-Tuning, TensorBoard experiment tracking, held-out test set evaluation, and checkpoint verification.
- **Workshop 3:** Sequential recommendation as a Markov Decision Process: a `Gymnasium` environment built from real post-split sessions and a Dueling Double DQN compared with the Workshop 1 and Workshop 2 models acting in the same environment (see [`workshop3/README.md`](workshop3/README.md)).

---

## Repository Structure

```text
Machine_Learning/
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Loading, subsampling, min-support filtering, contiguous IDs, and temporal split
│   ├── features.py        # Leak-free feature engineering (user/item stats, biases, genres, movie age)
│   ├── baselines.py       # Workshop 1 baseline models (regression, binary classification, and ranking)
│   ├── model.py           # Workshop 2 PyTorch architectures (BiasedMatrixFactorization & HybridNeuMF)
│   ├── train.py           # Workshop 2 training pipeline (pretraining, 2-phase fine-tuning, TensorBoard logs)
│   ├── evaluate.py        # Workshop 2 held-out test set evaluation & comparison vs. Workshop 1 baselines
│   └── test_model.py      # Checkpoint loader (checkpoints/model_best.pt) and inference verification script
├── notebooks/
│   └── eda.ipynb          # Workshop 1 EDA notebook with visualizations and statistical analysis
├── data/
│   └── splits.json        # Reproducible train/validation/test split indices and metadata (SEED = 42)
├── dataSet/               # Raw MovieLens 32M CSV files (ratings.csv, movies.csv, tags.csv, links.csv)
├── checkpoints/
│   └── model_best.pt      # Best validation-RMSE HybridNeuMF model checkpoint and configuration
├── runs/                  # TensorBoard event logs, training history CSV/JSON, learning curves, and plots
├── workshop3/             # Workshop 3: Gymnasium MDP, Dueling Double DQN, report (see workshop3/README.md)
├── group7_workshop2.pdf   # Workshop 2 Technical Report (IEEE format)
├── requirements.txt       # Pinned virtual environment dependencies
└── README.md
```

---

## Dataset Setup

**MovieLens 32M** (GroupLens Research, University of Minnesota).
- **Official source:** https://grouplens.org/datasets/movielens/32m/
- **Contents:** 32,000,204 ratings (`0.5` to `5.0` stars in `0.5` increments), 200,948 users, 87,585 movies (January 1995 – October 2023).
- Extract `ml-32m.zip` into the `dataSet/` directory (`dataSet/ratings.csv`, `dataSet/movies.csv`, `dataSet/tags.csv`, `dataSet/links.csv`).

---

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
.venv/bin/pip install -r requirements.txt
```

---

## Workshop 2 — Reproduction Commands

All Workshop 2 scripts use the fixed global seed (`SEED = 42`) and the identical leak-free chronological split (`train: 1,895,184 | val: 33,092 | test: 23,551` across `14,998` users and `11,589` movies) defined in `src/data_loader.py`.

1. **Verify the model architecture and forward pass (`src/model.py`):**

```bash
.venv/bin/python -m src.model
```

2. **Train `HybridNeuMF` with Transfer Learning & Two-Phase Fine-Tuning (`src/train.py`):**

```bash
.venv/bin/python -m src.train
```

- **Stage 0 (Pretraining):** Trains `BiasedMatrixFactorization` (`GMF_DIM = 64`) on the training split using Nesterov SGD with per-observation $L_2$ regularization (`reg = 0.045`).
- **Phase 1 (Frozen Pretrained Backbone, 5 epochs):** Transfers the pretrained GMF user/item embeddings and biases into `HybridNeuMF`, freezes them (`requires_grad = False`), and trains the content-aware `genre_proj` layer, `mlp_tower` (`128 -> 64 -> 32` with `BatchNorm1d`, `GELU`, `Dropout(0.30)`), and `fusion_head` using `AdamW` with exponential learning rate decay.
- **Phase 2 (Full Fine-Tuning, 10 epochs):** Unfreezes all layers (`requires_grad = True`) and jointly fine-tunes the entire network with a reduced learning rate governed by `CosineAnnealingLR`.
- Saves the best validation checkpoint to `checkpoints/model_best.pt`, logs epoch metrics to TensorBoard (`runs/hybrid_neumf_experiment/`) and `runs/training_history.csv`, and exports learning curves to `runs/learning_curves.png`.

3. **Evaluate on the held-out test set and compare against Workshop 1 baselines (`src/evaluate.py`):**

```bash
.venv/bin/python -m src.evaluate
```

Computes test-set regression metrics (`RMSE`, `MSE`, `MAE`, `$R^2$`), binary preference classification metrics (`like = 1` if `rating >= 4.0`: `Accuracy`, `Precision`, `Recall`, `F1-Score`, `ROC-AUC`), Top-10 ranking metrics (`P@10`, `R@10`, `NDCG@10`, `MAP@10`, `HitRate@10`), 95% bootstrap confidence intervals, and slice-based error analyses, saving all comparison tables and diagnostic figures in `runs/`.

4. **Load `checkpoints/model_best.pt` and verify inference on unseen data (`src/test_model.py`):**

```bash
.venv/bin/python -m src.test_model
```

5. **Launch TensorBoard to inspect training logs:**

```bash
.venv/bin/tensorboard --logdir runs/
```

---

## Workshop 1 — Reproduction Commands

1. **Generate reproducible splits (`data/splits.json`):**

```bash
.venv/bin/python -m src.data_loader
```

2. **Train and evaluate Workshop 1 baseline models (`src/baselines.py`):**

```bash
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
