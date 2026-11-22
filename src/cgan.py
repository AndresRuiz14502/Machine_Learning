"""Workshop 5 — GAN condicional (cGAN) sobre los vectores 64-D, condicionada en like in {0,1}.

Por que una cGAN: la GAN vanilla y la WGAN-GP aprenden la clase mayoritaria (like==1), igual
que el autoencoder de W4. Pero para el experimento de data augmentation lo util es poder
generar muestras de la clase que escasea (like==0, 'no le gusto'). La cGAN concatena la etiqueta
a la entrada de G y de D, asi podemos pedirle muestras de una clase concreta y balancear el
dataset de entrenamiento del clasificador downstream.

Entrenamos con datos reales que SI tienen ambas clases. Usamos la particion de validacion del
split de W1-W4 (es held-out y trae like==0 y like==1), evitando leakage del test.
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
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.data_loader import SEED
from src.data_w5 import load_w5_data
from src.gan import ConditionalDiscriminator, ConditionalGenerator, DATA_DIM, NOISE_DIM
from src.train_gan import BATCH_SIZE, BETAS, LABEL_REAL, LR, N_EPOCHS, set_all_seeds

CKPT_CGAN = PROJECT_ROOT / "checkpoints" / "cgan_best.pt"
W5_RUNS_DIR = PROJECT_ROOT / "runs" / "workshop5"
N_CLASSES = 2


def train_cgan(x_real: np.ndarray, y_real: np.ndarray, n_epochs: int = N_EPOCHS, device: torch.device | None = None) -> dict:
    set_all_seeds(SEED)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== cGAN | device={device} | datos={x_real.shape} | like%={y_real.mean():.3f} ===")

    x_t = torch.tensor(x_real, dtype=torch.float32)
    y_t = torch.tensor(y_real, dtype=torch.long)

    G = ConditionalGenerator(NOISE_DIM, DATA_DIM, N_CLASSES).to(device)
    D = ConditionalDiscriminator(DATA_DIM, N_CLASSES).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    opt_d = torch.optim.Adam(D.parameters(), lr=LR, betas=BETAS)
    bce = nn.BCELoss()

    W5_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(log_dir=str(W5_RUNS_DIR / "tensorboard_cgan"))

    history = []
    best_score = -1.0
    n = len(x_t)

    for epoch in range(1, n_epochs + 1):
        G.train(); D.train()
        perm = torch.randperm(n)
        d_loss_sum = g_loss_sum = real_acc_sum = fake_acc_sum = 0.0
        nb = 0

        for s in range(0, n, BATCH_SIZE):
            idx = perm[s : s + BATCH_SIZE]
            if len(idx) < 2:
                continue
            real = x_t[idx].to(device)
            labels = y_t[idx].to(device)
            bs = real.size(0)

            # --- D ---
            z = torch.randn(bs, NOISE_DIM, device=device)
            # para lo falso usamos etiquetas reales del batch (misma distribucion de clases)
            fake = G(z, labels).detach()
            d_real = D(real, labels)
            d_fake = D(fake, labels)
            loss_d = bce(d_real, torch.full((bs,), LABEL_REAL, device=device)) + \
                     bce(d_fake, torch.zeros(bs, device=device))
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            # --- G ---
            z = torch.randn(bs, NOISE_DIM, device=device)
            fake = G(z, labels)
            loss_g = bce(D(fake, labels), torch.ones(bs, device=device))
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            d_loss_sum += float(loss_d.item()); g_loss_sum += float(loss_g.item())
            real_acc_sum += float((d_real > 0.5).float().mean().item())
            fake_acc_sum += float((d_fake < 0.5).float().mean().item())
            nb += 1

        d_loss = d_loss_sum / nb; g_loss = g_loss_sum / nb
        real_acc = real_acc_sum / nb; fake_acc = fake_acc_sum / nb
        balance = 1.0 - abs(0.5 * (real_acc + fake_acc) - 0.5) * 2.0

        history.append({"epoch": epoch, "d_loss": d_loss, "g_loss": g_loss,
                        "d_real_acc": real_acc, "d_fake_acc": fake_acc})
        tb.add_scalar("cgan/d_loss", d_loss, epoch)
        tb.add_scalar("cgan/g_loss", g_loss, epoch)

        if balance > best_score:
            best_score = balance
            torch.save({"generator_state": G.state_dict(), "epoch": epoch,
                        "noise_dim": NOISE_DIM, "data_dim": DATA_DIM,
                        "n_classes": N_CLASSES, "model": "cgan"}, CKPT_CGAN)

        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            print(f"[cgan] ep {epoch:3d}/{n_epochs} | D_loss={d_loss:.3f} G_loss={g_loss:.3f} "
                  f"| D_real_acc={real_acc:.2f} D_fake_acc={fake_acc:.2f}")

    tb.flush(); tb.close()
    CKPT_CGAN.parent.mkdir(parents=True, exist_ok=True)
    with open(W5_RUNS_DIR / "cgan_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f)
    print(f"[cgan] mejor generador guardado en {CKPT_CGAN}")
    return {"history": history, "checkpoint": str(CKPT_CGAN)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_w5_data(use_cache=True, device=device)
    # datos reales con ambas clases: usamos la validacion (held-out, con like 0 y 1)
    train_cgan(data["X_val_all_scaled"], data["y_val"], device=device)


if __name__ == "__main__":
    main()
