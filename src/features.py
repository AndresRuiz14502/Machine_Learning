"""Feature engineering on the training set (no data leakage)."""

import pandas as pd


def compute_train_stats(train_df: pd.DataFrame):
    """Compute global mean, per-user/item statistics, and biases ONLY on train."""
    mu = float(train_df["rating"].mean())

    user_stats = train_df.groupby("u")["rating"].agg(u_mean="mean", u_count="size")
    item_stats = train_df.groupby("i")["rating"].agg(i_mean="mean", i_count="size")

    user_bias = (user_stats["u_mean"] - mu).rename("b_u")
    item_bias = (item_stats["i_mean"] - mu).rename("b_i")

    return mu, user_stats, item_stats, user_bias, item_bias


def build_genre_features(movies: pd.DataFrame, item2idx: dict):
    """Build one-hot genre features and release year per movie indexed by 'i'."""
    movies_idx = movies[movies["movieId"].isin(item2idx)].copy()
    movies_idx["i"] = movies_idx["movieId"].map(item2idx)
    if "year_movie" not in movies_idx.columns:
        movies_idx["year_movie"] = movies_idx["title"].str.extract(r"\((\d{4})\)").astype("float")

    genre_ohe = (
        movies_idx.assign(g=movies_idx["genres"].str.split("|"))
        .explode("g")
        .pivot_table(index="i", columns="g", aggfunc="size", fill_value=0)
    )
    genre_ohe = (genre_ohe > 0).astype("int8")
    genre_ohe.columns = [f"g_{c}" for c in genre_ohe.columns]

    item_year = movies_idx.set_index("i")["year_movie"].rename("item_year")
    return movies_idx, genre_ohe, item_year


def build_feature_matrix(
    part: pd.DataFrame,
    mu: float,
    user_stats: pd.DataFrame,
    item_stats: pd.DataFrame,
    user_bias: pd.Series,
    item_bias: pd.Series,
    item_year: pd.Series,
    genre_ohe: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the feature table (user/item/biases/genre/year/temporal)."""
    X = part[["u", "i", "year", "rating", "like"]].copy()
    X = X.join(user_stats, on="u").join(item_stats, on="i")
    X = X.join(user_bias, on="u").join(item_bias, on="i")
    X = X.join(item_year, on="i").join(genre_ohe, on="i")
    # Temporal feature: movie age at rating time.
    X["movie_age"] = (X["year"] - X["item_year"]).clip(lower=0)
    X = X.fillna(
        {
            "u_mean": mu,
            "i_mean": mu,
            "b_u": 0.0,
            "b_i": 0.0,
            "u_count": 0,
            "i_count": 0,
            "item_year": X["item_year"].median(),
            "movie_age": 0,
        }
    )
    return X.fillna(0)


def prepare_features(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, movies: pd.DataFrame, item2idx: dict):
    """Build X_train, X_val, X_test and return auxiliary statistics."""
    mu, user_stats, item_stats, user_bias, item_bias = compute_train_stats(train_df)
    movies_idx, genre_ohe, item_year = build_genre_features(movies, item2idx)

    X_train = build_feature_matrix(train_df, mu, user_stats, item_stats, user_bias, item_bias, item_year, genre_ohe)
    X_val = build_feature_matrix(val_df, mu, user_stats, item_stats, user_bias, item_bias, item_year, genre_ohe)
    X_test = build_feature_matrix(test_df, mu, user_stats, item_stats, user_bias, item_bias, item_year, genre_ohe)

    feature_cols = [c for c in X_train.columns if c not in ("rating", "like", "u", "i")]
    return {
        "mu": mu,
        "user_stats": user_stats,
        "item_stats": item_stats,
        "user_bias": user_bias,
        "item_bias": item_bias,
        "movies_idx": movies_idx,
        "genre_ohe": genre_ohe,
        "item_year": item_year,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "FEATURE_COLS": feature_cols,
    }
