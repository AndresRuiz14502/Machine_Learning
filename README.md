# Workshop 1 — MovieLens 32M Recommendation System

Machine Learning Workshop: exploratory data analysis, data preparation,
feature engineering, and evaluation of supervised baseline models on
MovieLens 32M.

---

## Repository Structure

```text
Machine_Learning/
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Loading, subsampling, filtering, contiguous IDs, and temporal split
│   ├── features.py        # Feature engineering (user/item stats, biases, genres, movie age)
│   └── baselines.py       # Training and evaluation of baseline models (regression & classification)
├── notebooks/
│   └── eda.ipynb          # Full EDA notebook with visualizations and analysis
├── data/
│   └── splits.json        # Train/validation/test split indices and metadata for reproducibility
├── dataSet/               # Raw MovieLens 32M data (not tracked in Git)
├── checkpoints/           # Saved model weights (future workshops)
├── runs/                  # Evaluation logs and plots (residuals and confusion matrices)
├── requirements.txt       # Pinned virtual environment dependencies
└── README.md
```

---

## Dataset

**MovieLens 32M** (GroupLens Research, University of Minnesota).

- **Official source:** https://grouplens.org/datasets/movielens/32m/
- **Alternative link:** https://drive.google.com/drive/folders/1RB8vg3Sc2HD2XlrOCdaUIwm5cEbPXRQ4?usp=sharing
- **Archive:** `ml-32m.zip` (~250 MB compressed, ~1 GB uncompressed)
- **Contents:** 32,000,204 ratings, 200,948 users, 87,585 movies, January 1995 – October 2023.
- **License:** academic use (see `README.txt` inside the dataset).

### Download and Setup

1. Download `ml-32m.zip` from the official source or the alternative link.
2. Extract it into the `dataSet/` folder with the following structure:

```text
Machine_Learning/
└── dataSet/
    ├── ratings.csv    # userId, movieId, rating, timestamp
    ├── movies.csv     # movieId, title, genres
    ├── tags.csv       # userId, movieId, tag, timestamp
    └── links.csv      # movieId, imdbId, tmdbId
```

---

## Environment Setup

Create and activate the virtual environment and install dependencies from `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Reproduction Commands

1. **Generate reproducible splits (`data/splits.json`):**

```bash
.venv/bin/python -m src.data_loader
```

2. **Train and evaluate baseline models (regression and classification):**

```bash
.venv/bin/python -m src.baselines
```

This prints all metrics to the console and saves the diagnostic plots in `runs/`:
- `runs/residuals_regression.png` — residual analysis ($y - \hat{y}$) for the regression models.
- `runs/confusion_matrices.png` — confusion matrices for the `like` classifiers.

3. **Exploratory Data Analysis (EDA):**
   - Open and run `notebooks/eda.ipynb` using the `.venv` kernel.

---

## Most Important Metrics for the Task

- **Regression (`rating` on the 0.5–5.0 scale):**
  - **RMSE / MSE** and **MAE** are the primary metrics because they directly measure rating prediction error in stars (RMSE penalizes larger errors more heavily, while MAE reflects typical absolute error).
  - **$R^2$** complements these by showing how much of the rating variance each model explains compared to predicting the global mean.
- **Binary Classification (`like = 1` if `rating >= 4.0`):**
  - **AUC-ROC** is the most important metric because it evaluates ranking/discrimination ability between liked and disliked items across all probability thresholds, which is essential for recommendation.
  - **F1-score** (alongside **precision** and **recall**) balances false positives (recommending movies the user dislikes) and false negatives (missing relevant movies), making it more informative than `accuracy` under class imbalance.
