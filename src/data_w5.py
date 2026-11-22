"""Workshop 5 — Carga de los vectores de interaccion 64-D para entrenar la GAN.

Reutilizamos build_workshop4_datasets() de Workshop 4 (sin modificarlo) para obtener los
mismos vectores 64-D leak-free y el MISMO split cronologico (train/val/test) de W1-W4.
Asi la comparacion es justa y no hay fuga de datos: las estadisticas de escalado se
ajustan SOLO con el train.

El generador usa Tanh (salida en [-1, 1]), pero los vectores de W4 estan en ~[-5, 5].
Por eso escalamos los datos a [-1, 1] con un min-max ajustado solo en train, y guardamos
esos limites para poder des-escalar las muestras sinteticas cuando las usemos luego.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.autoencoder import build_workshop4_datasets
from src.data_loader import SEED

# carpeta donde W5 guarda todo (cache, checkpoints, resultados)
W5_DIR = PROJECT_ROOT / "src" / "w5_artifacts"
CACHE_PATH = W5_DIR / "w5_data_cache.npz"


def _fit_minmax(x_train: np.ndarray):
    """Calcula min y max por columna SOLO en train (para escalar a [-1, 1] sin leakage)."""
    x_min = x_train.min(axis=0, keepdims=True)
    x_max = x_train.max(axis=0, keepdims=True)
    span = (x_max - x_min)
    span[span < 1e-6] = 1e-6  # evitar division por cero en columnas constantes
    return x_min, span


def scale_to_tanh(x: np.ndarray, x_min: np.ndarray, span: np.ndarray) -> np.ndarray:
    """Escala a [-1, 1] usando los limites de train."""
    x01 = (x - x_min) / span          # a [0, 1]
    return np.clip(2.0 * x01 - 1.0, -1.0, 1.0)  # a [-1, 1]


def unscale_from_tanh(x_scaled: np.ndarray, x_min: np.ndarray, span: np.ndarray) -> np.ndarray:
    """Operacion inversa: de [-1, 1] de vuelta a la escala original de W4."""
    x01 = (x_scaled + 1.0) / 2.0
    return x01 * span + x_min


def load_w5_data(use_cache: bool = True, max_train_normal: int = 200_000, device: torch.device | None = None) -> dict:
    """Devuelve los vectores 64-D reales listos para la GAN, ya escalados a [-1, 1].

    Claves devueltas:
      - X_train_scaled: perfiles 'normales' (like==1) de train, escalados -> entrenan la GAN vanilla/WGAN
      - X_train_all_scaled / y_train_all: train completo con ambas clases -> entrena la cGAN y el downstream
      - X_val_all_scaled / y_val, X_test_all_scaled / y_test: val y test completos para el downstream
      - x_min, span: limites de escalado (para des-escalar muestras sinteticas)
    """
    W5_DIR.mkdir(parents=True, exist_ok=True)

    if use_cache and CACHE_PATH.exists():
        print(f"[data_w5] Cargando cache: {CACHE_PATH}")
        d = np.load(CACHE_PATH)
        return {k: d[k] for k in d.files}

    print("[data_w5] Construyendo vectores 64-D desde Workshop 4 (puede tardar la primera vez)...")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build_workshop4_datasets ya hace: split cronologico W1, features leak-free, carga del W2, 64-D
    w4 = build_workshop4_datasets(max_train_normal=max_train_normal, seed=SEED, device=device)

    # --- Decision metodologica (importante, va explicada en el informe) ---
    # build_workshop4_datasets() solo expone X_train_norm (clase like==1) para el train, porque
    # W4 entrenaba el autoencoder solo con 'normales'. Val y test SI traen ambas clases (like 0 y 1).
    #
    # Para el experimento de data augmentation (R5) necesitamos un conjunto con ambas clases para
    # entrenar el clasificador downstream y otro held-out para evaluarlo. Usamos:
    #   - la GAN (vanilla/WGAN) aprende la clase mayoritaria like==1 desde X_train_norm (igual que el AE de W4);
    #   - el downstream REAL se entrena sobre val (ambas clases) y se evalua sobre test (ambas clases).
    # Ambos son particiones held-out cronologicas del MISMO split de W1-W4, asi que no hay leakage
    # y la comparacion entre condiciones (real / real+sintetico) es justa.

    x_train_norm = w4["X_train_norm"].astype(np.float32)          # (N, 64) clase 1
    x_val_all = w4["X_val_all"].astype(np.float32)                # (Nval, 64) ambas clases
    x_test_all = w4["X_test_all"].astype(np.float32)              # (Ntest, 64) ambas clases
    y_val_like = w4["y_val_like"].astype(np.int64)                # 1 = like, 0 = dislike
    y_test_like = w4["y_test_like"].astype(np.int64)

    # ajustamos el escalado SOLO con el train normal (sin leakage)
    x_min, span = _fit_minmax(x_train_norm)

    X_train_scaled = scale_to_tanh(x_train_norm, x_min, span)
    X_val_all_scaled = scale_to_tanh(x_val_all, x_min, span)
    X_test_all_scaled = scale_to_tanh(x_test_all, x_min, span)

    # para la cGAN necesitamos datos con ambas clases; usamos el val (tiene like y dislike)
    # como fuente de datos etiquetados reales para condicionar la generacion.
    out = {
        "X_train_scaled": X_train_scaled.astype(np.float32),
        "X_val_all_scaled": X_val_all_scaled.astype(np.float32),
        "y_val": y_val_like,
        "X_test_all_scaled": X_test_all_scaled.astype(np.float32),
        "y_test": y_test_like,
        "x_min": x_min.astype(np.float32),
        "span": span.astype(np.float32),
    }

    np.savez_compressed(CACHE_PATH, **out)
    print(f"[data_w5] Cache guardada en: {CACHE_PATH}")
    print(
        f"[data_w5] train(like=1): {X_train_scaled.shape} | "
        f"val: {X_val_all_scaled.shape} (like%={y_val_like.mean():.3f}) | "
        f"test: {X_test_all_scaled.shape} (like%={y_test_like.mean():.3f})"
    )
    return out


if __name__ == "__main__":
    data = load_w5_data(use_cache=True)
    for k, v in data.items():
        print(f"{k:20s} shape={getattr(v, 'shape', None)}")
