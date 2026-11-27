"""Workshop 5 — Entrenamiento de la GAN vanilla y de la WGAN-GP sobre vectores 64-D.

Dos modelos con el MISMO generador:
  - Vanilla GAN: discriminador con Sigmoid, perdida BCE, label smoothing (reales=0.9),
    optimizadores Adam separados (lr=2e-4) para G y D.
  - WGAN-GP: critico sin Sigmoid, perdida de Wasserstein + gradient penalty (lambda=10),
    5 pasos de critico por cada paso de generador.

Guardamos checkpoints en checkpoints/gan_vanilla_best.pt y checkpoints/gan_wgan_best.pt,
logs de TensorBoard en runs/workshop5/ y muestras generadas cada 10 epocas.
"""

from __future__ import annotations

import json
import sys
import time
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
from src.gan import Critic, DATA_DIM, Discriminator, Generator, NOISE_DIM

# rutas de salida (misma convencion que W2/W3/W4: checkpoints/ y runs/workshopN/)
CKPT_VANILLA = PROJECT_ROOT / "checkpoints" / "gan_vanilla_best.pt"
CKPT_WGAN = PROJECT_ROOT / "checkpoints" / "gan_wgan_best.pt"
W5_RUNS_DIR = PROJECT_ROOT / "runs" / "workshop5"

# hiperparametros pedidos por el enunciado
LR = 2e-4
BETAS = (0.5, 0.9)          # betas tipicas para GANs
BATCH_SIZE = 256
N_EPOCHS = 120              # "100+" epocas
LABEL_REAL = 0.9            # label smoothing de un lado
GP_LAMBDA = 10.0            # peso del gradient penalty
N_CRITIC = 5               # pasos de critico por paso de generador (WGAN-GP)
SAMPLE_EVERY = 10          # guardar muestras cada 10 epocas


def set_all_seeds(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_batches(x: torch.Tensor, batch_size: int, device: torch.device):
    """Generador de mini-batches barajados (una pasada = una epoca)."""
    n = len(x)
    perm = torch.randperm(n)
    for s in range(0, n, batch_size):
        idx = perm[s : s + batch_size]
        if len(idx) < 2:
            continue  # BatchNorm necesita al menos 2 muestras
        yield x[idx].to(device)


# ---------------------------------------------------------------------------
# 1. GAN VANILLA (BCE + label smoothing)
# ---------------------------------------------------------------------------
def train_vanilla_gan(x_train: np.ndarray, n_epochs: int = N_EPOCHS, device: torch.device | None = None) -> dict:
    set_all_seeds(SEED)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== GAN vanilla | device={device} | datos={x_train.shape} ===")

    x_t = torch.tensor(x_train, dtype=torch.float32)
    G = Generator(NOISE_DIM, DATA_DIM).to(device)
    D = Discriminator(DATA_DIM).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    opt_d = torch.optim.Adam(D.parameters(), lr=LR, betas=BETAS)
    bce = nn.BCELoss()

    W5_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(log_dir=str(W5_RUNS_DIR / "tensorboard_vanilla"))
    samples_dir = W5_RUNS_DIR / "samples_vanilla"
    samples_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_score = -1.0  # elegimos el "mejor" por acc de D mas cercana a 0.5 (equilibrio)

    for epoch in range(1, n_epochs + 1):
        G.train(); D.train()
        d_loss_sum = g_loss_sum = 0.0
        real_acc_sum = fake_acc_sum = 0.0
        nb = 0

        for real in get_batches(x_t, BATCH_SIZE, device):
            bs = real.size(0)
            real_label = torch.full((bs,), LABEL_REAL, device=device)   # 0.9 (smoothing)
            fake_label = torch.zeros(bs, device=device)

            # --- entrenar D: real vs falso ---
            z = torch.randn(bs, NOISE_DIM, device=device)
            fake = G(z).detach()
            d_real = D(real)
            d_fake = D(fake)
            loss_d = bce(d_real, real_label) + bce(d_fake, fake_label)
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            # --- entrenar G: que D crea que lo falso es real ---
            z = torch.randn(bs, NOISE_DIM, device=device)
            fake = G(z)
            d_on_fake = D(fake)
            loss_g = bce(d_on_fake, torch.ones(bs, device=device))
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            d_loss_sum += float(loss_d.item()); g_loss_sum += float(loss_g.item())
            real_acc_sum += float((d_real > 0.5).float().mean().item())
            fake_acc_sum += float((d_fake < 0.5).float().mean().item())
            nb += 1

        d_loss = d_loss_sum / nb; g_loss = g_loss_sum / nb
        real_acc = real_acc_sum / nb; fake_acc = fake_acc_sum / nb
        # queremos a D equilibrado (ni gana ni pierde): acc global cercana a 0.5 es sano
        balance = 1.0 - abs(0.5 * (real_acc + fake_acc) - 0.5) * 2.0

        history.append({"epoch": epoch, "d_loss": d_loss, "g_loss": g_loss,
                        "d_real_acc": real_acc, "d_fake_acc": fake_acc})
        tb.add_scalar("vanilla/d_loss", d_loss, epoch)
        tb.add_scalar("vanilla/g_loss", g_loss, epoch)
        tb.add_scalar("vanilla/d_real_acc", real_acc, epoch)
        tb.add_scalar("vanilla/d_fake_acc", fake_acc, epoch)

        if balance > best_score:
            best_score = balance
            torch.save({"generator_state": G.state_dict(), "epoch": epoch,
                        "noise_dim": NOISE_DIM, "data_dim": DATA_DIM,
                        "model": "vanilla"}, CKPT_VANILLA)

        if epoch % SAMPLE_EVERY == 0 or epoch == 1 or epoch == n_epochs:
            with torch.no_grad():
                smp = G.sample(64, device).cpu().numpy()
            np.save(samples_dir / f"epoch_{epoch:03d}.npy", smp)
            print(f"[vanilla] ep {epoch:3d}/{n_epochs} | D_loss={d_loss:.3f} G_loss={g_loss:.3f} "
                  f"| D_real_acc={real_acc:.2f} D_fake_acc={fake_acc:.2f}")

    tb.flush(); tb.close()
    CKPT_VANILLA.parent.mkdir(parents=True, exist_ok=True)
    with open(W5_RUNS_DIR / "vanilla_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f)
    print(f"[vanilla] mejor generador guardado en {CKPT_VANILLA}")
    return {"history": history, "checkpoint": str(CKPT_VANILLA)}


# ---------------------------------------------------------------------------
# 2. WGAN-GP (Wasserstein + gradient penalty)
# ---------------------------------------------------------------------------
def gradient_penalty(critic: Critic, real: torch.Tensor, fake: torch.Tensor, device: torch.device) -> torch.Tensor:
    """GP = lambda * E[(||grad D(x_hat)|| - 1)^2] con x_hat interpolado real/falso."""
    bs = real.size(0)
    eps = torch.rand(bs, 1, device=device)                 # epsilon ~ U[0,1]
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    score = critic(x_hat)
    grads = torch.autograd.grad(
        outputs=score, inputs=x_hat,
        grad_outputs=torch.ones_like(score),
        create_graph=True, retain_graph=True,
    )[0]
    grad_norm = grads.norm(2, dim=1)
    return ((grad_norm - 1.0) ** 2).mean()


def train_wgan_gp(x_train: np.ndarray, n_epochs: int = N_EPOCHS, device: torch.device | None = None) -> dict:
    set_all_seeds(SEED)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== WGAN-GP | device={device} | datos={x_train.shape} ===")

    x_t = torch.tensor(x_train, dtype=torch.float32)
    G = Generator(NOISE_DIM, DATA_DIM).to(device)
    C = Critic(DATA_DIM).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    opt_c = torch.optim.Adam(C.parameters(), lr=LR, betas=BETAS)

    W5_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(log_dir=str(W5_RUNS_DIR / "tensorboard_wgan"))
    samples_dir = W5_RUNS_DIR / "samples_wgan"
    samples_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_w = -1e9  # mejor = menor distancia de Wasserstein (critico mas estable al final)

    n = len(x_t)
    for epoch in range(1, n_epochs + 1):
        G.train(); C.train()
        perm = torch.randperm(n)
        c_loss_sum = g_loss_sum = w_dist_sum = 0.0
        g_steps = 0
        # recorremos en bloques de N_CRITIC batches para D y luego 1 paso de G
        ptr = 0
        batch_starts = list(range(0, n, BATCH_SIZE))
        bi = 0
        while bi < len(batch_starts):
            # --- N_CRITIC pasos de critico ---
            w_dist_local = 0.0
            n_c = 0
            for _ in range(N_CRITIC):
                if bi >= len(batch_starts):
                    break
                idx = perm[batch_starts[bi] : batch_starts[bi] + BATCH_SIZE]
                bi += 1
                if len(idx) < 2:
                    continue
                real = x_t[idx].to(device)
                bs = real.size(0)
                z = torch.randn(bs, NOISE_DIM, device=device)
                fake = G(z).detach()
                c_real = C(real).mean()
                c_fake = C(fake).mean()
                gp = gradient_penalty(C, real, fake, device)
                loss_c = -(c_real - c_fake) + GP_LAMBDA * gp   # minimizar -> maximiza c_real-c_fake
                opt_c.zero_grad(); loss_c.backward(); opt_c.step()
                c_loss_sum += float(loss_c.item())
                w_dist_local += float((c_real - c_fake).item()); n_c += 1

            # --- 1 paso de generador ---
            z = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)
            fake = G(z)
            loss_g = -C(fake).mean()
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()
            g_loss_sum += float(loss_g.item()); g_steps += 1
            if n_c > 0:
                w_dist_sum += w_dist_local / n_c

        c_loss = c_loss_sum / max(g_steps * N_CRITIC, 1)
        g_loss = g_loss_sum / max(g_steps, 1)
        w_dist = w_dist_sum / max(g_steps, 1)

        history.append({"epoch": epoch, "c_loss": c_loss, "g_loss": g_loss, "w_distance": w_dist})
        tb.add_scalar("wgan/critic_loss", c_loss, epoch)
        tb.add_scalar("wgan/generator_loss", g_loss, epoch)
        tb.add_scalar("wgan/wasserstein_distance", w_dist, epoch)

        # guardamos el generador cuando la distancia de Wasserstein es mas pequena/estable
        score = -abs(w_dist)
        if score > best_w:
            best_w = score
            torch.save({"generator_state": G.state_dict(), "epoch": epoch,
                        "noise_dim": NOISE_DIM, "data_dim": DATA_DIM,
                        "model": "wgan_gp"}, CKPT_WGAN)

        if epoch % SAMPLE_EVERY == 0 or epoch == 1 or epoch == n_epochs:
            with torch.no_grad():
                smp = G.sample(64, device).cpu().numpy()
            np.save(samples_dir / f"epoch_{epoch:03d}.npy", smp)
            print(f"[wgan] ep {epoch:3d}/{n_epochs} | C_loss={c_loss:.3f} G_loss={g_loss:.3f} "
                  f"| W_dist={w_dist:.3f}")

    tb.flush(); tb.close()
    CKPT_WGAN.parent.mkdir(parents=True, exist_ok=True)
    with open(W5_RUNS_DIR / "wgan_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f)
    print(f"[wgan] mejor generador guardado en {CKPT_WGAN}")
    return {"history": history, "checkpoint": str(CKPT_WGAN)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_w5_data(use_cache=True, device=device)
    x_train = data["X_train_scaled"]
    train_vanilla_gan(x_train, device=device)
    train_wgan_gp(x_train, device=device)


if __name__ == "__main__":
    main()
