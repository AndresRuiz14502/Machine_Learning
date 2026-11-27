"""Workshop 5 — Generador, Discriminador y Critico para GAN tabular (vectores de interaccion 64-D).

Trabajamos sobre los vectores continuos de 64 dimensiones de Workshop 4 (perfiles de
interaccion usuario-pelicula, leak-free). Por eso usamos capas totalmente conectadas (MLP),
tal como recomienda el enunciado para datos tabulares, y Tanh a la salida del generador
porque los datos de entrada estan estandarizados/acotados.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn

# dimension del vector de datos (64-D de Workshop 4) y del ruido de entrada
DATA_DIM = 64
NOISE_DIM = 64


class Generator(nn.Module):
    """Generador G: mapea un vector de ruido z a un vector sintetico de 64-D."""

    def __init__(self, noise_dim: int = NOISE_DIM, data_dim: int = DATA_DIM, hidden: int = 256):
        super().__init__()
        self.noise_dim = int(noise_dim)
        self.data_dim = int(data_dim)
        # red simple: ruido -> 256 -> 256 -> 64, con BatchNorm y ReLU (guia DCGAN para G)
        self.net = nn.Sequential(
            nn.Linear(self.noise_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.data_dim),
            nn.Tanh(),  # salida acotada en [-1, 1] para datos normalizados
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Genera n muestras sinteticas a partir de ruido gaussiano."""
        self.eval()
        z = torch.randn(n, self.noise_dim, device=device)
        return self.forward(z)


class Discriminator(nn.Module):
    """Discriminador D para la GAN vanilla: clasifica real (1) vs falso (0) con Sigmoid."""

    def __init__(self, data_dim: int = DATA_DIM, hidden: int = 256):
        super().__init__()
        self.data_dim = int(data_dim)
        # LeakyReLU en D (guia DCGAN). Dropout para no ganar demasiado rapido.
        self.net = nn.Sequential(
            nn.Linear(self.data_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),  # probabilidad de ser real
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class Critic(nn.Module):
    """Critico para WGAN-GP: misma idea que D pero SIN Sigmoid (devuelve un score real).

    Ademas no usamos BatchNorm aqui porque rompe la penalizacion de gradiente de WGAN-GP
    (el GP se calcula por muestra y BatchNorm mezcla muestras del batch). Usamos LayerNorm.
    """

    def __init__(self, data_dim: int = DATA_DIM, hidden: int = 256):
        super().__init__()
        self.data_dim = int(data_dim)
        self.net = nn.Sequential(
            nn.Linear(self.data_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),  # score real, sin activacion final
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ConditionalGenerator(nn.Module):
    """Generador condicional (cGAN): concatena la etiqueta de clase al ruido.

    En nuestro dominio la etiqueta discreta es like in {0, 1} (le gusto o no la pelicula).
    """

    def __init__(self, noise_dim: int = NOISE_DIM, data_dim: int = DATA_DIM, n_classes: int = 2, hidden: int = 256):
        super().__init__()
        self.noise_dim = int(noise_dim)
        self.data_dim = int(data_dim)
        self.n_classes = int(n_classes)
        # embedding de la clase para concatenar al ruido
        self.label_emb = nn.Embedding(self.n_classes, self.n_classes)
        self.net = nn.Sequential(
            nn.Linear(self.noise_dim + self.n_classes, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.data_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        c = self.label_emb(labels)
        return self.net(torch.cat([z, c], dim=-1))

    @torch.no_grad()
    def sample(self, n: int, label: int, device: torch.device) -> torch.Tensor:
        """Genera n muestras de una clase concreta (label)."""
        self.eval()
        z = torch.randn(n, self.noise_dim, device=device)
        labels = torch.full((n,), int(label), dtype=torch.long, device=device)
        return self.forward(z, labels)


class ConditionalDiscriminator(nn.Module):
    """Discriminador condicional: concatena la etiqueta a la entrada de datos."""

    def __init__(self, data_dim: int = DATA_DIM, n_classes: int = 2, hidden: int = 256):
        super().__init__()
        self.data_dim = int(data_dim)
        self.n_classes = int(n_classes)
        self.label_emb = nn.Embedding(self.n_classes, self.n_classes)
        self.net = nn.Sequential(
            nn.Linear(self.data_dim + self.n_classes, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        c = self.label_emb(labels)
        return self.net(torch.cat([x, c], dim=-1)).squeeze(-1)


if __name__ == "__main__":
    # chequeo rapido de que las formas cuadran
    torch.manual_seed(42)
    g = Generator()
    d = Discriminator()
    c = Critic()
    z = torch.randn(8, NOISE_DIM)
    fake = g(z)
    print("Generator  ->", tuple(z.shape), "a", tuple(fake.shape))
    print("Discrim.   ->", tuple(d(fake).shape))
    print("Critic     ->", tuple(c(fake).shape))

    cg = ConditionalGenerator()
    cd = ConditionalDiscriminator()
    labels = torch.randint(0, 2, (8,))
    cfake = cg(z, labels)
    print("cGAN gen   ->", tuple(cfake.shape), "| cGAN disc ->", tuple(cd(cfake, labels).shape))
