"""Workshop 5 — Tabla comparativa acumulativa a traves de todos los workshops (R7).

Junta en una sola tabla el mejor resultado representativo de cada workshop sobre la tarea de
clasificacion binaria (like), que es la que comparten W1, W2, W4 y W5:

  - W1/W2: cifras tomadas de los papers del equipo (resultados ya reportados y fijos).
  - W4   : mejor VAE (beta=4.0, ELBO) reportado en el paper de W4.
  - W5   : se leen en vivo de runs/workshop5/augmentation_results.json (NO se inventan;
           salen de ejecutar augment_and_retrain.py).

W3 es secuencial (RL) y no comparte el protocolo de clasificacion binaria, asi que se cita
aparte en una nota, no en la misma columna de metricas (seria comparar peras con manzanas).

Guarda la tabla en runs/workshop5/cumulative_comparison.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.train_gan import W5_RUNS_DIR

# cifras fijas de los papers (clasificacion binaria like, test set). Fuente: informes W1/W2/W4.
PAPER_ROWS = [
    {"workshop": "W1", "model": "Random Forest Clf",        "paradigm": "Supervisado (arbol)",   "accuracy": 0.7031, "precision": 0.5773, "recall": 0.6914, "f1": 0.6293, "roc_auc": 0.7645},
    {"workshop": "W1", "model": "Logistic Regression",      "paradigm": "Supervisado (lineal)",  "accuracy": 0.7134, "precision": 0.6026, "recall": 0.6270, "f1": 0.6146, "roc_auc": 0.7604},
    {"workshop": "W2", "model": "HybridNeuMF",              "paradigm": "Deep CF supervisado",   "accuracy": 0.7156, "precision": 0.5838, "recall": 0.7651, "f1": 0.6623, "roc_auc": 0.7942},
    {"workshop": "W4", "model": "VAE (beta=4.0, ELBO)",     "paradigm": "No supervisado (VAE)",  "accuracy": 0.6528, "precision": 0.6620, "recall": 0.9270, "f1": 0.7724, "roc_auc": 0.6964},
]

W5_LABELS = {
    "a_real_only":          ("W5", "Clasificador 64-D (solo real)",       "Baseline downstream"),
    "b_real_plus_vanilla":  ("W5", "Clasificador + GAN vanilla",          "Augmentacion (GAN)"),
    "c_real_plus_wgan":     ("W5", "Clasificador + WGAN-GP",              "Augmentacion (WGAN-GP)"),
    "d_real_plus_cgan":     ("W5", "Clasificador + cGAN (balanceado)",    "Augmentacion (cGAN)"),
}


def build_table() -> pd.DataFrame:
    rows = list(PAPER_ROWS)

    aug_path = W5_RUNS_DIR / "augmentation_results.json"
    if aug_path.exists():
        with open(aug_path, "r", encoding="utf-8") as f:
            w5 = json.load(f)
        for key, m in w5.items():
            ws, model, paradigm = W5_LABELS.get(key, ("W5", key, "W5"))
            rows.append({
                "workshop": ws, "model": model, "paradigm": paradigm,
                "accuracy": m["accuracy"], "precision": m["precision"],
                "recall": m["recall"], "f1": m["f1"], "roc_auc": m["roc_auc"],
            })
    else:
        print(f"[tabla] aviso: no existe {aug_path}. Corre 'python -m src.augment_and_retrain' primero.")

    df = pd.DataFrame(rows, columns=["workshop", "model", "paradigm",
                                     "accuracy", "precision", "recall", "f1", "roc_auc"])
    return df


def main():
    df = build_table()
    W5_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = W5_RUNS_DIR / "cumulative_comparison.csv"
    df.to_csv(out, index=False)
    print("\n=== Tabla acumulativa W1-W5 (clasificacion binaria 'like', test set) ===")
    print(df.to_string(index=False))
    print(f"\n[tabla] guardada en {out}")
    print("\nNota: W3 (RL D3QN) usa un protocolo secuencial distinto (return, NDCG@K en sesiones)")
    print("y no entra en esta columna de clasificacion binaria; se reporta aparte en el informe.")
    return df


if __name__ == "__main__":
    main()
