"""Workshop 5 — Evaluacion de calidad de la generacion (metricas de dominio) y modos de fallo.

Inception Score / Frechet con una red Inception son para imagenes, no aplican a datos
tabulares. Usamos metricas apropiadas a nuestro dominio (vectores 64-D):

  - Frechet Distance tabular: misma idea que FID pero directamente sobre los 64-D.
    FD = ||mu_r - mu_g||^2 + Tr(C_r + C_g - 2*(C_r*C_g)^{1/2}). Mas bajo = mejor.
  - Preservacion de correlaciones: diferencia entre la matriz de correlacion real y la
    generada (si la GAN captura la estructura entre features).
  - Error en medias/desviaciones por columna (que tan bien casa cada feature).
  - Diversidad de lo generado y chequeo de mode collapse (distancia media entre muestras).

Compara vanilla vs WGAN-GP y guarda metricas en runs/workshop5/gan_quality_metrics.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from scipy import linalg

from src.data_w5 import load_w5_data
from src.gan import DATA_DIM, Generator, NOISE_DIM
from src.train_gan import CKPT_VANILLA, CKPT_WGAN, W5_RUNS_DIR


def load_generator(ckpt_path: Path, device: torch.device) -> Generator:
    """Reconstruye el generador y carga pesos desde un checkpoint."""
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    G = Generator(payload.get("noise_dim", NOISE_DIM), payload.get("data_dim", DATA_DIM)).to(device)
    G.load_state_dict(payload["generator_state"])
    G.eval()
    return G


def frechet_distance(real: np.ndarray, fake: np.ndarray) -> float:
    """Distancia de Frechet entre dos nubes de puntos (versos tabular del FID)."""
    mu_r, mu_g = real.mean(axis=0), fake.mean(axis=0)
    cov_r = np.cov(real, rowvar=False)
    cov_g = np.cov(fake, rowvar=False)
    diff = mu_r - mu_g
    # raiz de la matriz producto; puede devolver parte imaginaria por ruido numerico
    covmean, _ = linalg.sqrtm(cov_r @ cov_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd = float(diff @ diff + np.trace(cov_r + cov_g - 2.0 * covmean))
    return fd


def correlation_preservation(real: np.ndarray, fake: np.ndarray) -> float:
    """Error medio absoluto entre la matriz de correlacion real y la generada."""
    cr = np.corrcoef(real, rowvar=False)
    cg = np.corrcoef(fake, rowvar=False)
    cr = np.nan_to_num(cr); cg = np.nan_to_num(cg)
    return float(np.abs(cr - cg).mean())


def mean_std_error(real: np.ndarray, fake: np.ndarray) -> tuple[float, float]:
    """Error absoluto medio de medias y desviaciones por columna."""
    mean_err = float(np.abs(real.mean(0) - fake.mean(0)).mean())
    std_err = float(np.abs(real.std(0) - fake.std(0)).mean())
    return mean_err, std_err


def sample_diversity(fake: np.ndarray, n_pairs: int = 2000, seed: int = 42) -> float:
    """Distancia euclidea media entre pares de muestras generadas.

    Si es muy baja respecto a los datos reales, hay mode collapse (todo se parece).
    """
    rng = np.random.default_rng(seed)
    n = len(fake)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    d = np.linalg.norm(fake[i] - fake[j], axis=1)
    return float(d.mean())


def evaluate_one(name: str, real: np.ndarray, fake: np.ndarray) -> dict:
    fd = frechet_distance(real, fake)
    corr = correlation_preservation(real, fake)
    mean_err, std_err = mean_std_error(real, fake)
    div_fake = sample_diversity(fake)
    div_real = sample_diversity(real)
    # ratio de diversidad: cerca de 1 = buena cobertura; cerca de 0 = mode collapse
    div_ratio = div_fake / (div_real + 1e-8)
    res = {
        "model": name,
        "frechet_distance": round(fd, 4),
        "correlation_mae": round(corr, 4),
        "feature_mean_mae": round(mean_err, 4),
        "feature_std_mae": round(std_err, 4),
        "diversity_generated": round(div_fake, 4),
        "diversity_real": round(div_real, 4),
        "diversity_ratio": round(div_ratio, 4),
        "mode_collapse_flag": bool(div_ratio < 0.5),
    }
    return res


def main(n_samples: int = 5000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_w5_data(use_cache=True, device=device)
    real = data["X_train_scaled"]
    # submuestra de reales para que las matrices de covarianza no sean enormes
    rng = np.random.default_rng(42)
    real_sub = real[rng.choice(len(real), size=min(n_samples, len(real)), replace=False)]

    results = []
    for name, ckpt in [("vanilla", CKPT_VANILLA), ("wgan_gp", CKPT_WGAN)]:
        if not ckpt.exists():
            print(f"[eval] falta checkpoint {ckpt}, saltando {name}")
            continue
        G = load_generator(ckpt, device)
        fake = G.sample(n_samples, device).cpu().numpy()
        res = evaluate_one(name, real_sub, fake)
        results.append(res)
        print(f"\n--- {name} ---")
        for k, v in res.items():
            print(f"  {k:22s}: {v}")

    W5_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = W5_RUNS_DIR / "gan_quality_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] metricas guardadas en {out_path}")

    # pequeno veredicto comparativo (cual es mejor por Frechet)
    if len(results) == 2:
        best = min(results, key=lambda r: r["frechet_distance"])
        print(f"[eval] Mejor calidad por Frechet Distance: {best['model']}")
    return results


if __name__ == "__main__":
    main()
