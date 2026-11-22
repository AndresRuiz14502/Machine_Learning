"""Workshop 5 — Experimento de data augmentation (R5).

Modelo downstream: clasificador tabular que predice like in {0,1} a partir del vector 64-D.

Por que un clasificador tabular y no HybridNeuMF de W2: HybridNeuMF consume INDICES de
usuario/pelicula y aprende sus propios embeddings; no recibe vectores de features. Una GAN
que genera vectores 64-D sinteticos no produce IDs validos, asi que no se puede "reentrenar
HybridNeuMF con datos aumentados" sin inventar una asignacion de IDs. El clasificador tabular
sobre los 64-D es el patron exacto de Frid-Adar et al. (2018): entrenar GAN -> generar
sintetico -> aumentar el train del clasificador -> medir si mejora la tarea downstream.
Es la opcion A que acordamos; se menciona explicitamente en el informe.

Comparamos CUATRO condiciones (misma arquitectura, mismo presupuesto, mismo test):
  (a) REAL         : solo datos reales (baseline).
  (b) REAL+VANILLA : real + sinteticos de la GAN vanilla (clase mayoritaria like==1).
  (c) REAL+WGAN    : real + sinteticos de la WGAN-GP (clase mayoritaria like==1).
  (d) REAL+cGAN    : real + sinteticos BALANCEADOS de la cGAN (genera tambien like==0).

Justificacion de incluir las cuatro:
  - (b) y (c) son las condiciones literales del enunciado (GAN no condicional). Como estas GAN
    aprenden la clase mayoritaria (like==1), augmentar refuerza una clase que ya domina; es
    esperable poca mejora o incluso un ligero perjuicio. Mostrarlo es un resultado honesto.
  - (d) es la condicion informativa: la cGAN permite generar la clase MINORITARIA (like==0),
    que es justo la que al clasificador le cuesta. Augmentar de forma balanceada es donde el
    augmentation tiene sentido, y da material para el analisis "por que ayuda o perjudica".

Datos (mismo split de W1-W4, sin leakage):
  - entrenamos el downstream con la particion de VALIDACION (held-out, tiene ambas clases),
  - evaluamos sobre el TEST (held-out, ambas clases). Las estadisticas de escalado vienen de train.
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn

from src.data_loader import SEED
from src.data_w5 import load_w5_data
from src.gan import ConditionalGenerator, DATA_DIM, Generator, NOISE_DIM
from src.train_gan import CKPT_VANILLA, CKPT_WGAN, W5_RUNS_DIR, set_all_seeds
from src.cgan import CKPT_CGAN, N_CLASSES


# ---------------------------------------------------------------------------
# clasificador downstream: una MLP pequena sobre los 64-D
# ---------------------------------------------------------------------------
class LikeClassifier(nn.Module):
    def __init__(self, in_dim: int = DATA_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_classifier(x_tr, y_tr, x_te, y_te, device, n_epochs=40, batch_size=256) -> dict:
    """Entrena el clasificador y devuelve metricas sobre el test. Presupuesto fijo (fair)."""
    set_all_seeds(SEED)
    x_tr_t = torch.tensor(x_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    clf = LikeClassifier().to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()
    n = len(x_tr_t)

    for _ in range(n_epochs):
        clf.train()
        perm = torch.randperm(n)
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            bx = x_tr_t[idx].to(device)
            by = y_tr_t[idx].to(device)
            opt.zero_grad()
            loss = bce(clf(bx), by)
            loss.backward()
            opt.step()

    # evaluacion sobre el test held-out
    clf.eval()
    with torch.no_grad():
        logits = clf(torch.tensor(x_te, dtype=torch.float32).to(device)).cpu().numpy()
    prob = 1.0 / (1.0 + np.exp(-logits))
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "precision": round(float(precision_score(y_te, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_te, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_te, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_te, prob)), 4),
    }


# ---------------------------------------------------------------------------
# generacion de sinteticos para augmentar
# ---------------------------------------------------------------------------
def gen_unconditional(ckpt: Path, n: int, device) -> np.ndarray:
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    G = Generator(payload.get("noise_dim", NOISE_DIM), payload.get("data_dim", DATA_DIM)).to(device)
    G.load_state_dict(payload["generator_state"]); G.eval()
    return G.sample(n, device).cpu().numpy()


def gen_conditional(ckpt: Path, n_per_class: int, device) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    G = ConditionalGenerator(payload.get("noise_dim", NOISE_DIM), payload.get("data_dim", DATA_DIM),
                             payload.get("n_classes", N_CLASSES)).to(device)
    G.load_state_dict(payload["generator_state"]); G.eval()
    x0 = G.sample(n_per_class, label=0, device=device).cpu().numpy()  # like==0 (minoritaria)
    x1 = G.sample(n_per_class, label=1, device=device).cpu().numpy()  # like==1
    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(int)
    return x, y


def main(n_synth: int = 3000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_w5_data(use_cache=True, device=device)

    # datos reales del downstream: val (train del clasificador) y test (evaluacion)
    x_real, y_real = data["X_val_all_scaled"], data["y_val"]
    x_test, y_test = data["X_test_all_scaled"], data["y_test"]

    results = {}

    # (a) baseline: solo real
    print("\n=== (a) REAL (baseline) ===")
    results["a_real_only"] = train_classifier(x_real, y_real, x_test, y_test, device)
    print(results["a_real_only"])

    # (b) real + vanilla (sinteticos de clase mayoritaria like==1)
    if CKPT_VANILLA.exists():
        print("\n=== (b) REAL + VANILLA GAN ===")
        synth = gen_unconditional(CKPT_VANILLA, n_synth, device)
        xb = np.concatenate([x_real, synth], axis=0)
        yb = np.concatenate([y_real, np.ones(n_synth, dtype=int)])  # vanilla aprende like==1
        results["b_real_plus_vanilla"] = train_classifier(xb, yb, x_test, y_test, device)
        print(results["b_real_plus_vanilla"])

    # (c) real + WGAN-GP
    if CKPT_WGAN.exists():
        print("\n=== (c) REAL + WGAN-GP ===")
        synth = gen_unconditional(CKPT_WGAN, n_synth, device)
        xc = np.concatenate([x_real, synth], axis=0)
        yc = np.concatenate([y_real, np.ones(n_synth, dtype=int)])
        results["c_real_plus_wgan"] = train_classifier(xc, yc, x_test, y_test, device)
        print(results["c_real_plus_wgan"])

    # (d) real + cGAN balanceado (genera tambien like==0)
    if CKPT_CGAN.exists():
        print("\n=== (d) REAL + cGAN (balanceado) ===")
        xs, ys = gen_conditional(CKPT_CGAN, n_synth // 2, device)
        xd = np.concatenate([x_real, xs], axis=0)
        yd = np.concatenate([y_real, ys])
        results["d_real_plus_cgan"] = train_classifier(xd, yd, x_test, y_test, device)
        print(results["d_real_plus_cgan"])

    W5_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = W5_RUNS_DIR / "augmentation_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[augment] resultados guardados en {out}")

    # resumen comparativo contra la baseline
    base = results["a_real_only"]
    print("\n=== Resumen (delta de F1 y AUC frente a la baseline real) ===")
    for cond, m in results.items():
        if cond == "a_real_only":
            print(f"  {cond:22s} F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}  (baseline)")
        else:
            print(f"  {cond:22s} F1={m['f1']:.4f} ({m['f1']-base['f1']:+.4f})  "
                  f"AUC={m['roc_auc']:.4f} ({m['roc_auc']-base['roc_auc']:+.4f})")
    return results


if __name__ == "__main__":
    main()
