"""Workshop 5 — Figuras adicionales para el informe (curvas de entrenamiento y muestras por epoca).

Genera dos cosas que el resto del pipeline no dibuja como PNG:
  1. training_curves.png : curvas de perdida de la GAN vanilla (D_loss, G_loss) y de la WGAN-GP
     (W_dist), leidas de los JSON guardados por src/train_gan.py.
  2. samples_grid_vanilla.png / samples_grid_wgan.png : rejillas de las muestras sinteticas guardadas
     cada 10 epocas (archivos .npy), que es la evidencia "save generated samples every 10 epochs".

No reentrena nada: solo lee los artefactos ya producidos en runs/workshop5/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

W5_RUNS_DIR = PROJECT_ROOT / "runs" / "workshop5"


def plot_training_curves(save_path: Path) -> None:
    """Dibuja D_loss/G_loss (vanilla) y W_dist (WGAN-GP) a lo largo de las epocas."""
    van_path = W5_RUNS_DIR / "vanilla_history.json"
    wgan_path = W5_RUNS_DIR / "wgan_history.json"
    if not van_path.exists() or not wgan_path.exists():
        print("[figuras] faltan vanilla_history.json / wgan_history.json; corre src.train_gan primero")
        return

    van = json.load(open(van_path, encoding="utf-8"))
    wgan = json.load(open(wgan_path, encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # panel izquierdo: perdidas de la GAN vanilla
    ep = [r["epoch"] for r in van]
    axes[0].plot(ep, [r["d_loss"] for r in van], label="D loss", color="#4C72B0", linewidth=2)
    axes[0].plot(ep, [r["g_loss"] for r in van], label="G loss", color="#C44E52", linewidth=2)
    axes[0].set_title("Vanilla GAN training losses")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (BCE)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # panel derecho: distancia de Wasserstein de la WGAN-GP
    ep2 = [r["epoch"] for r in wgan]
    axes[1].plot(ep2, [r["w_distance"] for r in wgan], label="Wasserstein distance", color="#55A868", linewidth=2)
    axes[1].set_title("WGAN-GP Wasserstein distance (stability)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("W distance (critic)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle("Workshop 5 — GAN training dynamics")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[figuras] curvas guardadas en {save_path}")


def plot_samples_grid(samples_dir: Path, title: str, save_path: Path) -> None:
    """Rejilla de heatmaps de las muestras .npy guardadas cada 10 epocas (evolucion del generador)."""
    if not samples_dir.exists():
        print(f"[figuras] no existe {samples_dir}; corre src.train_gan primero")
        return
    files = sorted(samples_dir.glob("epoch_*.npy"))
    if not files:
        print(f"[figuras] no hay .npy en {samples_dir}")
        return

    # elegimos hasta 6 epocas repartidas (primera, ultimas y algunas intermedias)
    if len(files) > 6:
        idx = np.linspace(0, len(files) - 1, 6).astype(int)
        files = [files[i] for i in idx]

    n = len(files)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, f in zip(axes, files):
        smp = np.load(f)  # (64, 64) = 64 muestras x 64 features
        ax.imshow(smp, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ep = f.stem.replace("epoch_", "")
        ax.set_title(f"epoch {int(ep)}")
        ax.set_xlabel("feature")
        ax.set_yticks([])
    axes[0].set_ylabel("sample")
    fig.suptitle(f"Generated samples every ~10 epochs — {title}")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[figuras] rejilla de muestras guardada en {save_path}")


def main() -> None:
    plot_training_curves(W5_RUNS_DIR / "training_curves.png")
    plot_samples_grid(W5_RUNS_DIR / "samples_vanilla", "Vanilla GAN", W5_RUNS_DIR / "samples_grid_vanilla.png")
    plot_samples_grid(W5_RUNS_DIR / "samples_wgan", "WGAN-GP", W5_RUNS_DIR / "samples_grid_wgan.png")
    print("[figuras] listo.")


if __name__ == "__main__":
    main()
