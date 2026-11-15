# Workshop 3 — Sequential Movie Recommendation via Deep Reinforcement Learning on MovieLens 32M

**Course**: Machine Learning (2026-III) — Universidad Distrital Francisco José de Caldas  
**Professor**: Eng. Carlos Andrés Sierra, M.Sc. (`cavirguezs@udistrital.edu.co`)  
**Team 7 Authors**:
- **Andrés Felipe Ruiz Vasallo** (`afruizv@udistrital.edu.co`)
- **Josep Emanuel León Joya** (`jeleonj@udistrital.edu.co`)
- **Matthew Stev Toro Mondragón** (`mstorom@udistrital.edu.co`)

📄 **Compiled IEEE Paper (9 Pages, English)**: [`group7_workshop3.pdf`](./group7_workshop3.pdf) (also in [`report/group7_workshop3.pdf`](./report/group7_workshop3.pdf))

---

## 1. Overview & Motivation

In **Workshop 1**, we formulated movie recommendation on the **MovieLens 32M** dataset ($32{,}000{,}204$ ratings across $200{,}948$ users and $84{,}432$ movies) as a static supervised matrix factorization problem (**FunkSVD**). While FunkSVD achieved strong pointwise rating accuracy ($\text{RMSE} = 0.8041$, $\text{MAE} = 0.5998$), static ranking suffered from severe popularity bias—concentrating recommendations in only **$8.00\%$** of the catalog ($\text{Novelty} = 14.06\text{ bits}$) and failing to adapt dynamically to intra-session user feedback.

In **Workshop 3**, we reformulate sequential top-$N$ movie recommendation as a finite-horizon **Markov Decision Process (MDP)** $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ implemented under the standard **`Gymnasium`** API and solved via a **Dueling Double Deep Q-Network (D3QN)**:
- **State Space ($\mathcal{S} \subset \mathbb{R}^{250}$)**: Concatenates dynamic session-item cosine similarities ($\mathbb{R}^{50}$), Workshop 1 FunkSVD baseline predictions ($\mathbb{R}^{50}$), self-information novelty scores ($\mathbb{R}^{50}$), binary visited-action masks ($\{0,1\}^{50}$), static user latent factors $\mathbf{p}_u \in \mathbb{R}^{20}$, dynamic session belief vectors $\bar{\mathbf{q}}_t \in \mathbb{R}^{20}$, and $10$ session progress/retention indicators.
- **Action Space ($|\mathcal{A}| = 50$)**: Stratified per-user candidate slate with strict invalid-action masking ($\mathcal{A}_{\text{valid}}(s_t)$) to prevent duplicate recommendations.
- **Reward Function ($\mathcal{R}$)**: Composite shaped reward combining ground-truth/latent rating relevance, self-information catalog novelty ($I(i) = -\log_2 P(i)$), intra-session genre diversity bonus, and early session abandonment termination upon $5$ consecutive dislikes.

---

## 2. Key Experimental Results ($K=10$ & $K=20$, 250 Held-Out Test Users)

| Policy | Mean Return ($K=10$) | Precision@10 | Recall@10 | NDCG@10 | MAP@10 | Hit Rate@10 | Coverage (%) | Novelty (bits) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **D3QN (Ours, $\gamma=0.95$)** | **12.37 $\pm$ 0.83** | **0.9792** | **0.5160** | **0.9818** | **0.9692** | **1.0000** | **36.64%** | **17.96** |
| Myopic Contextual Bandit ($\gamma=0$) | 10.49 $\pm$ 0.90 | 0.8808 | 0.4611 | 0.8924 | 0.8467 | 1.0000 | 34.35% | 17.75 |
| Random Policy ($\pi_{\text{rand}}$) | 2.68 $\pm$ 2.38 | 0.3401 | 0.1726 | 0.3480 | 0.2528 | 0.9400 | **66.77%** | 14.37 |
| Greedy FunkSVD (Workshop 1) | 1.36 $\pm$ 2.13 | 0.2738 | 0.1322 | 0.2785 | 0.1945 | 0.8440 | 17.25% | 14.23 |
| Bias Baseline ($\pi_{\text{bias}}$) | -0.05 $\pm$ 2.26 | 0.1456 | 0.0644 | 0.1345 | 0.0825 | 0.5640 | 13.70% | 13.61 |
| Popularity Heuristic ($\pi_{\text{pop}}$) | -1.41 $\pm$ 1.88 | 0.0680 | 0.0258 | 0.0551 | 0.0247 | 0.3000 | 14.79% | 12.16 |

- **Mitigating Popularity Bias**: D3QN expands catalog coverage from Workshop 1's **$8.00\%$** to **$36.64\%$** at $K=10$ (**$60.35\%$** at $K=20$, a **$4.58\times$** increase) while elevating self-information novelty to **$17.96\text{ bits}$**.
- **Value of Multi-Step Planning ($\gamma=0.95$ vs. $\gamma=0$)**: Outperforms the Myopic Contextual Bandit ($\gamma=0$) by **$+17.9\%$** in cumulative session return at $K=10$ and **$+21.4\%$** at $K=20$ ($\text{Return} = 20.10$, $\text{NDCG@20} = 0.9167$).
- **Sample Efficiency**: Surpasses all static Workshop 1 baselines within **$381$ environment interaction steps** ($40$ episodes) and overtakes the Myopic Contextual Bandit within **$3{,}940\text{--}4{,}340$ steps**.

---

## 3. Repository Structure (`workshop3/`)

```text
workshop3/
├── README.md                          # Project documentation & reproducibility guide
├── requirements.txt                   # Python dependencies (PyTorch, Gymnasium, SB3, etc.)
├── group7_workshop3.pdf               # Compiled 9-page IEEEtran paper (English)
├── checkpoints/
│   └── agent_best.pt                  # Best trained D3QN checkpoint (lr=5e-4, gamma=0.95)
├── data/
│   ├── dataset_stats.json             # Cohort & chronological split statistics
│   ├── movie_lookup.json              # Filtered catalog metadata (11,589 movies)
│   └── processed_cache.npz            # Precomputed leak-free 20-D FunkSVD factors & splits
├── runs/
│   └── workshop3/                     # TensorBoard logs (8 hyperparameter configs + champion)
├── src/
│   ├── __init__.py
│   ├── data_prep.py                   # Leak-free temporal split & FunkSVD factor extraction
│   ├── env.py                         # Custom Gymnasium environment (MovieLensRecEnv)
│   ├── agent.py                       # Dueling Double DQN (D3QN) + ReplayBuffer
│   ├── baselines.py                   # Random, Popularity, BiasOnly, GreedyFunkSVD, MyopicBandit
│   ├── train.py                       # 8-config sensitivity study + champion GPU/CPU training
│   └── evaluate.py                    # Held-out test evaluation (K=10, 20) & 300-DPI plots
├── tests/
│   └── test_env.py                    # Pytest unit tests for Gymnasium environment (5/5 passing)
└── report/
    ├── main.tex                       # Master IEEEtran LaTeX file
    ├── bibliography.bib               # BibTeX references
    ├── group7_workshop3.pdf           # Compiled IEEE conference paper
    ├── metrics_summary.json           # Complete test & validation metrics (JSON)
    ├── random_diagnostics.json        # 100-episode random policy verification stats
    ├── training_logs.json             # Learning curves & sample efficiency trajectories
    ├── sections/                      # Modular LaTeX sections (sec1_intro to sec5_discussion)
    └── figures/                       # 300-DPI publication figures (fig2 to fig7)
```

---

## 4. Reproducibility Instructions

### 4.1 Environment Setup
```bash
pip install -r workshop3/requirements.txt
```

### 4.2 Run Unit Tests & Random-Policy Environment Verification
Because `workshop3/data/processed_cache.npz` is included in the repository, you can immediately run the `Gymnasium` unit tests and environment verification without downloading the raw 900 MB MovieLens 32M CSVs:
```bash
pytest workshop3/tests/test_env.py -v
python workshop3/src/env.py
```

### 4.3 Evaluate Best Checkpoint & Regenerate All Figures
```bash
python workshop3/src/evaluate.py
```

### 4.4 Full Training & Hyperparameter Sensitivity Study
```bash
python workshop3/src/train.py
tensorboard --logdir workshop3/runs/workshop3
```

### 4.5 Data Preparation from Raw MovieLens 32M CSVs (Optional)
If you wish to regenerate `processed_cache.npz` from scratch, place the MovieLens 32M files (`ratings.csv`, `movies.csv`) under `workshop3/data/ml-32m/ml-32m/` and run:
```bash
python workshop3/src/data_prep.py
```
