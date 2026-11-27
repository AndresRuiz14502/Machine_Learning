"""Workshop 5 — Generacion de muestras sinteticas y visualizacion (real vs generado).

Como los datos son tabulares (vectores 64-D) y no imagenes, las "rejillas 8x8" del
enunciado las mostramos como:
  (1) heatmaps lado a lado: 8x8 = 64 muestras reales vs 64 generadas (cada fila un vector),
  (2) comparacion de distribuciones de unas pocas features (real vs generado),
  (3) proyeccion 2D (PCA) de reales vs generados para ver si cubren la misma region.

Guarda las figuras en runs/workshop5/.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # backend sin ventana, para que funcione en servidores/Kaggle
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from src.data_w5 import load_w5_data
from src.evaluate_gan import load_generator
from src.train_gan import CKPT_VANILLA, CKPT_WGAN, W5_RUNS_DIR


def grid_heatmaps(real: np.ndarray, fake: np.ndarray, title: str, save_path: Path):
    """Heatmaps 8x8 (64 muestras) de reales vs generados, lado a lado."""
    n = 64
    r = real[:n]
    f = fake[:n]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vmin, vmax = -1.0, 1.0
    im0 = axes[0].imshow(r, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[0].set_title("Reales (64 muestras x 64 features)")
    axes[0].set_xlabel("feature"); axes[0].set_ylabel("muestra")
    im1 = axes[1].imshow(f, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Generadas ({title})")
    axes[1].set_xlabel("feature"); axes[1].set_ylabel("muestra")
    fig.colorbar(im1, ax=axes, shrink=0.8)
    fig.suptitle(f"Real vs Generado — {title}")
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[gen] heatmap guardado: {save_path}")


def feature_distributions(real: np.ndarray, fake: np.ndarray, title: str, save_path: Path, feats=(0, 1, 2, 3, 4, 5)):
    """Histogramas de unas pocas features: real vs generado."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, k in zip(axes.ravel(), feats):
        ax.hist(real[:, k], bins=40, alpha=0.5, label="real", density=True, color="#4C72B0")
        ax.hist(fake[:, k], bins=40, alpha=0.5, label="generado", density=True, color="#C44E52")
        ax.set_title(f"feature {k}")
        ax.legend(fontsize=8)
    fig.suptitle(f"Distribuciones por feature — {title}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[gen] distribuciones guardadas: {save_path}")


def pca_overlay(real: np.ndarray, fakes: dict, save_path: Path):
    """Proyeccion PCA 2D: reales vs generados de cada modelo (ver cobertura del espacio)."""
    pca = PCA(n_components=2, random_state=42)
    pca.fit(real)
    r2 = pca.transform(real)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(r2[:, 0], r2[:, 1], s=6, alpha=0.25, label="real", color="gray")
    colors = {"vanilla": "#C44E52", "wgan_gp": "#55A868"}
    for name, f in fakes.items():
        f2 = pca.transform(f)
        ax.scatter(f2[:, 0], f2[:, 1], s=6, alpha=0.35, label=f"generado ({name})",
                   color=colors.get(name, "#8172B3"))
    ax.set_title("Proyeccion PCA 2D — real vs generado")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[gen] PCA overlay guardado: {save_path}")


def main(n_samples: int = 3000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_w5_data(use_cache=True, device=device)
    real = data["X_train_scaled"]
    rng = np.random.default_rng(42)
    real_sub = real[rng.choice(len(real), size=min(n_samples, len(real)), replace=False)]

    W5_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    fakes = {}
    for name, ckpt in [("vanilla", CKPT_VANILLA), ("wgan_gp", CKPT_WGAN)]:
        if not ckpt.exists():
            print(f"[gen] falta checkpoint {ckpt}, saltando {name}")
            continue
        G = load_generator(ckpt, device)
        fake = G.sample(n_samples, device).cpu().numpy()
        fakes[name] = fake
        grid_heatmaps(real_sub, fake, name, W5_RUNS_DIR / f"grid_real_vs_{name}.png")
        feature_distributions(real_sub, fake, name, W5_RUNS_DIR / f"feature_dist_{name}.png")

    if fakes:
        pca_overlay(real_sub, fakes, W5_RUNS_DIR / "pca_real_vs_generated.png")
    print("[gen] listo.")


if __name__ == "__main__":
    main()
