"""Carga, preprocesamiento y partición temporal de MovieLens 32M."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

# === Reproducibilidad: una única semilla para todo el proyecto ===
SEED = 42

# === Hiperparámetros de escala y de preparación de datos ===
N_USERS_SAMPLE = 15_000       # factorización (holgado en 16 GB)
N_USERS_SAMPLE_TREES = 6_000  # baselines con árboles/GBM

MIN_RATINGS_PER_USER = 10
MIN_RATINGS_PER_ITEM = 10

LIKE_THRESHOLD = 4.0
TOP_K = 10

TEST_FRACTION = 0.10
VAL_FRACTION = 0.10

RATINGS_DTYPES = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32",
    "timestamp": "int64",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLITS_PATH = PROJECT_ROOT / "data" / "splits.json"


def set_seed(seed: int = SEED) -> None:
    """Fija las semillas globales de Python y NumPy."""
    random.seed(seed)
    np.random.seed(seed)


def resolve_data_dir(candidate_paths=None) -> Path:
    """Resuelve automáticamente la ruta del dataset."""
    if candidate_paths is None:
        candidate_paths = [
            Path(r"C:\Users\andre\OneDrive\Documentos\TranajosML\Workshop1\dataSet"),
            PROJECT_ROOT / "dataSet",
            Path("dataSet"),
        ]
    data_dir = next((p for p in candidate_paths if (p / "ratings.csv").exists()), None)
    if data_dir is None:
        raise FileNotFoundError(
            "No se encontró 'ratings.csv'. Edita CANDIDATE_PATHS con la ruta correcta de tu dataSet."
        )
    return data_dir


def load_raw_data(data_dir: Path | None = None, load_links_tags: bool = True):
    """Carga los CSV crudos de MovieLens 32M y extrae columnas temporales."""
    if data_dir is None:
        data_dir = resolve_data_dir()

    ratings = pd.read_csv(data_dir / "ratings.csv", dtype=RATINGS_DTYPES)
    movies = pd.read_csv(data_dir / "movies.csv", dtype={"movieId": "int32"})

    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings["year"] = ratings["datetime"].dt.year.astype("int16")
    ratings["month"] = ratings["datetime"].dt.month.astype("int8")

    movies["year_movie"] = movies["title"].str.extract(r"\((\d{4})\)").astype("float")
    movies["genres_list"] = movies["genres"].str.split("|")

    if not load_links_tags:
        return ratings, movies

    links = pd.read_csv(data_dir / "links.csv")
    tags = pd.read_csv(
        data_dir / "tags.csv", dtype={"userId": "int32", "movieId": "int32"}
    )
    return ratings, movies, links, tags


def sample_users(ratings: pd.DataFrame, n_users_sample: int = N_USERS_SAMPLE, seed: int = SEED):
    """Muestrea usuarios completos para no partir perfiles."""
    all_users = ratings["userId"].unique()
    rng = np.random.default_rng(seed)
    if n_users_sample < len(all_users):
        sampled = rng.choice(all_users, size=n_users_sample, replace=False)
        df = ratings[ratings["userId"].isin(sampled)].copy()
        return df, sampled
    return ratings.copy(), all_users


def filter_min_support(data: pd.DataFrame, min_u: int, min_i: int, max_iter: int = 10) -> pd.DataFrame:
    """Filtra iterativamente usuarios/ítems con poco soporte hasta estabilizar."""
    prev = None
    for _ in range(max_iter):
        uc = data["userId"].value_counts()
        data = data[data["userId"].isin(uc[uc >= min_u].index)]
        ic = data["movieId"].value_counts()
        data = data[data["movieId"].isin(ic[ic >= min_i].index)]
        if prev == len(data):
            break
        prev = len(data)
    return data.copy()


def build_contiguous_ids(df: pd.DataFrame):
    """Asigna índices contiguos u (0..N_U-1) e i (0..N_I-1)."""
    user_ids = df["userId"].unique()
    item_ids = df["movieId"].unique()

    user2idx = {int(uid): idx for idx, uid in enumerate(user_ids)}
    item2idx = {int(iid): idx for idx, iid in enumerate(item_ids)}
    idx2item = {idx: int(iid) for iid, idx in item2idx.items()}

    df = df.copy()
    df["u"] = df["userId"].map(user2idx).astype("int32")
    df["i"] = df["movieId"].map(item2idx).astype("int32")
    return df, user2idx, item2idx, idx2item


def temporal_split(data: pd.DataFrame, test_frac: float = TEST_FRACTION, val_frac: float = VAL_FRACTION):
    """Corte cronológico global: train (pasado) / val / test (futuro reciente)."""
    data = data.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    n = len(data)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    train = data.iloc[:n_train]
    val = data.iloc[n_train:n_train + n_val]
    test = data.iloc[n_train + n_val:]
    return train.copy(), val.copy(), test.copy()


def report_and_clean(name: str, part: pd.DataFrame, train_users: set, train_items: set, verbose: bool = True) -> pd.DataFrame:
    """Retira usuarios/ítems en val/test no vistos en train para evitar cold-start en evaluación."""
    mask_known = part["u"].isin(train_users) & part["i"].isin(train_items)
    cold = int((~mask_known).sum())
    if verbose:
        print(
            f"{name}: {cold:,} filas con usuario/ítem no visto en train "
            f"({cold/len(part)*100:.2f}%) -> se retiran para evaluación limpia"
        )
    return part[mask_known].copy()


def save_splits(
    sampled_users: np.ndarray,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    splits_path: Path = DEFAULT_SPLITS_PATH,
) -> Path:
    """Guarda los parámetros, usuarios muestreados e índices de partición en JSON."""
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SEED,
        "n_users_sample": N_USERS_SAMPLE,
        "min_ratings_per_user": MIN_RATINGS_PER_USER,
        "min_ratings_per_item": MIN_RATINGS_PER_ITEM,
        "like_threshold": LIKE_THRESHOLD,
        "test_fraction": TEST_FRACTION,
        "val_fraction": VAL_FRACTION,
        "sampled_user_ids": [int(u) for u in sampled_users],
        "counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "timestamp_bounds": {
            "train": [int(train_df["timestamp"].min()), int(train_df["timestamp"].max())],
            "val": [int(val_df["timestamp"].min()), int(val_df["timestamp"].max())],
            "test": [int(test_df["timestamp"].min()), int(test_df["timestamp"].max())],
        },
        "indices": {
            "train": train_df.index.astype(int).tolist(),
            "val": val_df.index.astype(int).tolist(),
            "test": test_df.index.astype(int).tolist(),
        },
    }
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return splits_path


def load_splits(splits_path: Path = DEFAULT_SPLITS_PATH) -> dict:
    """Carga el archivo de particiones data/splits.json."""
    with open(splits_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_datasets(
    data_dir: Path | None = None,
    splits_path: Path | None = DEFAULT_SPLITS_PATH,
    save_splits_file: bool = False,
    verbose: bool = True,
):
    """Ejecuta el flujo completo de carga, submuestreo, filtrado, IDs contiguos y split temporal."""
    set_seed(SEED)
    ratings, movies = load_raw_data(data_dir=data_dir, load_links_tags=False)

    df, sampled_users = sample_users(ratings, N_USERS_SAMPLE, SEED)
    if verbose:
        print(f"Submuestra: {len(sampled_users):,} usuarios -> {len(df):,} ratings.")

    before = len(df)
    df = filter_min_support(df, MIN_RATINGS_PER_USER, MIN_RATINGS_PER_ITEM)
    if verbose:
        print(f"Ratings: {before:,} -> {len(df):,} (eliminados {before - len(df):,})")
        print(f"Usuarios: {df['userId'].nunique():,}  |  Películas: {df['movieId'].nunique():,}")

    df, user2idx, item2idx, idx2item = build_contiguous_ids(df)
    df["like"] = (df["rating"] >= LIKE_THRESHOLD).astype("int8")

    train_df, val_df, test_df = temporal_split(df, TEST_FRACTION, VAL_FRACTION)
    train_users = set(train_df["u"].unique())
    train_items = set(train_df["i"].unique())

    val_df = report_and_clean("val ", val_df, train_users, train_items, verbose=verbose)
    test_df = report_and_clean("test", test_df, train_users, train_items, verbose=verbose)

    if verbose:
        print(f"Tamaños finales -> train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,}")

    if save_splits_file and splits_path is not None:
        out = save_splits(sampled_users, train_df, val_df, test_df, splits_path=splits_path)
        if verbose:
            print(f"Particiones guardadas en: {out}")

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "movies": movies,
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2item": idx2item,
        "n_users": len(user2idx),
        "n_items": len(item2idx),
        "sampled_users": sampled_users,
    }


if __name__ == "__main__":
    prepare_datasets(save_splits_file=True, verbose=True)
