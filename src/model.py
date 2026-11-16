"""Neural Collaborative Filtering architectures (BiasedMF and HybridNeuMF) in PyTorch."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn


class BiasedMatrixFactorization(nn.Module):
    """Pretrainable Generalized Matrix Factorization (GMF) model with explicit biases."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        global_mean: float = 3.5401,
        init_std: float = 0.10,
    ) -> None:
        super().__init__()
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.embedding_dim = int(embedding_dim)

        self.user_factors = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_factors = nn.Embedding(self.n_items, self.embedding_dim)
        self.user_bias = nn.Embedding(self.n_users, 1)
        self.item_bias = nn.Embedding(self.n_items, 1)
        self.register_buffer(
            "global_mean", torch.tensor(float(global_mean), dtype=torch.float32)
        )

        self._init_weights(init_std)

    def _init_weights(self, std: float) -> None:
        nn.init.normal_(self.user_factors.weight, mean=0.0, std=std)
        nn.init.normal_(self.item_factors.weight, mean=0.0, std=std)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    @torch.no_grad()
    def init_biases_from_stats(
        self, user_bias_vec: torch.Tensor, item_bias_vec: torch.Tensor, shrink: float = 0.85
    ) -> None:
        """Warm-start user and item biases from shrunk training-set deviations."""
        self.user_bias.weight.data.copy_(
            (user_bias_vec.view(-1, 1).to(self.user_bias.weight.device) * shrink)
        )
        self.item_bias.weight.data.copy_(
            (item_bias_vec.view(-1, 1).to(self.item_bias.weight.device) * shrink)
        )

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        u_emb = self.user_factors(user_idx)
        i_emb = self.item_factors(item_idx)
        dot = (u_emb * i_emb).sum(dim=-1)
        bu = self.user_bias(user_idx).squeeze(-1)
        bi = self.item_bias(item_idx).squeeze(-1)
        return self.global_mean + bu + bi + dot


class HybridNeuMF(nn.Module):
    """Hybrid Neural Collaborative Filtering combining pretrained GMF and a content-aware Deep MLP."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_genres: int = 20,
        n_user_Prior_feats: int = 2,
        n_item_prior_feats: int = 3,
        gmf_dim: int = 64,
        mlp_emb_dim: int = 32,
        genre_proj_dim: int = 16,
        mlp_hidden_dims: tuple[int, ...] = (128, 64, 32),
        dropout_rate: float = 0.30,
        global_mean: float = 3.5401,
    ) -> None:
        super().__init__()
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.n_genres = int(n_genres)
        self.n_user_prior_feats = int(n_user_Prior_feats)
        self.n_item_prior_feats = int(n_item_prior_feats)
        self.gmf_dim = int(gmf_dim)
        self.mlp_emb_dim = int(mlp_emb_dim)
        self.genre_proj_dim = int(genre_proj_dim)
        self.mlp_hidden_dims = tuple(int(d) for d in mlp_hidden_dims)
        self.dropout_rate = float(dropout_rate)

        # Pretrainable GMF branch (user/item embeddings and explicit biases)
        self.user_gmf_factors = nn.Embedding(self.n_users, self.gmf_dim)
        self.item_gmf_factors = nn.Embedding(self.n_items, self.gmf_dim)
        self.user_bias = nn.Embedding(self.n_users, 1)
        self.item_bias = nn.Embedding(self.n_items, 1)
        self.register_buffer(
            "global_mean", torch.tensor(float(global_mean), dtype=torch.float32)
        )

        # Lookup buffers for training-only side features (genres and normalized support/year priors)
        self.register_buffer(
            "item_genres", torch.zeros(self.n_items, self.n_genres, dtype=torch.float32)
        )
        self.register_buffer(
            "user_priors",
            torch.zeros(self.n_users, self.n_user_prior_feats, dtype=torch.float32),
        )
        self.register_buffer(
            "item_priors",
            torch.zeros(self.n_items, self.n_item_prior_feats, dtype=torch.float32),
        )

        # Deep MLP collaborative embeddings
        self.user_mlp_factors = nn.Embedding(self.n_users, self.mlp_emb_dim)
        self.item_mlp_factors = nn.Embedding(self.n_items, self.mlp_emb_dim)

        # Content projection for multi-hot movie genres
        self.genre_proj = nn.Sequential(
            nn.Linear(self.n_genres, self.genre_proj_dim),
            nn.LayerNorm(self.genre_proj_dim),
            nn.GELU(),
        )

        mlp_in_dim = (
            self.mlp_emb_dim * 3
            + self.gmf_dim
            + self.genre_proj_dim
            + self.n_user_prior_feats
            + self.n_item_prior_feats
        )

        layers: list[nn.Module] = []
        prev_dim = mlp_in_dim
        for h_dim in self.mlp_hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout_rate),
                ]
            )
            prev_dim = h_dim
        self.mlp_tower = nn.Sequential(*layers)

        # NeuMF fusion layer combining GMF interaction and deep nonlinear representation
        self.fusion_head = nn.Linear(self.gmf_dim + prev_dim, 1)
        self.gmf_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.level_shift = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.user_gmf_factors.weight, mean=0.0, std=0.08)
        nn.init.normal_(self.item_gmf_factors.weight, mean=0.0, std=0.08)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        nn.init.normal_(self.user_mlp_factors.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.item_mlp_factors.weight, mean=0.0, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize fusion weights near zero so Phase 1 starts seamlessly from pretrained GMF
        nn.init.normal_(self.fusion_head.weight, mean=0.0, std=0.002)
        nn.init.zeros_(self.fusion_head.bias)

    @torch.no_grad()
    def set_side_features(
        self,
        item_genres: torch.Tensor,
        user_priors: torch.Tensor,
        item_priors: torch.Tensor,
    ) -> None:
        """Store training-set item genre matrix and user/item prior statistics in model buffers."""
        self.item_genres.copy_(item_genres.to(self.item_genres.device, dtype=torch.float32))
        self.user_priors.copy_(user_priors.to(self.user_priors.device, dtype=torch.float32))
        self.item_priors.copy_(item_priors.to(self.item_priors.device, dtype=torch.float32))

    @torch.no_grad()
    def load_pretrained_gmf(self, pretrained_mf: BiasedMatrixFactorization) -> None:
        """Transfer pretrained embeddings and biases from BiasedMatrixFactorization."""
        self.user_gmf_factors.weight.data.copy_(pretrained_mf.user_factors.weight.data)
        self.item_gmf_factors.weight.data.copy_(pretrained_mf.item_factors.weight.data)
        self.user_bias.weight.data.copy_(pretrained_mf.user_bias.weight.data)
        self.item_bias.weight.data.copy_(pretrained_mf.item_bias.weight.data)
        self.global_mean.copy_(pretrained_mf.global_mean)

        shared_dim = min(self.gmf_dim, self.mlp_emb_dim)
        self.user_mlp_factors.weight.data[:, :shared_dim].copy_(
            pretrained_mf.user_factors.weight.data[:, :shared_dim] * 0.20
        )
        self.item_mlp_factors.weight.data[:, :shared_dim].copy_(
            pretrained_mf.item_factors.weight.data[:, :shared_dim] * 0.20
        )

    def set_pretrained_frozen(self, freeze: bool = True) -> None:
        """Freeze or unfreeze the embedding layers so Phase 1 trains only the task-specific head."""
        requires_grad = not freeze
        for layer in (
            self.user_gmf_factors,
            self.item_gmf_factors,
            self.user_bias,
            self.item_bias,
            self.user_mlp_factors,
            self.item_mlp_factors,
        ):
            for param in layer.parameters():
                param.requires_grad = requires_grad

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        u_safe = user_idx.clamp(0, self.n_users - 1)
        i_safe = item_idx.clamp(0, self.n_items - 1)

        # GMF pathway
        u_gmf = self.user_gmf_factors(u_safe)
        i_gmf = self.item_gmf_factors(i_safe)
        gmf_vec = u_gmf * i_gmf
        gmf_dot = gmf_vec.sum(dim=-1)
        bu = self.user_bias(u_safe).squeeze(-1)
        bi = self.item_bias(i_safe).squeeze(-1)

        # Deep MLP + content pathway
        u_mlp = self.user_mlp_factors(u_safe)
        i_mlp = self.item_mlp_factors(i_safe)
        mlp_inter = u_mlp * i_mlp
        g_emb = self.genre_proj(self.item_genres[i_safe])
        u_pri = self.user_priors[u_safe]
        i_pri = self.item_priors[i_safe]

        mlp_in = torch.cat([u_mlp, i_mlp, mlp_inter, gmf_vec.detach(), g_emb, u_pri, i_pri], dim=-1)
        mlp_hidden = self.mlp_tower(mlp_in)

        fusion_in = torch.cat([gmf_vec, mlp_hidden], dim=-1)
        nonlinear_delta = 0.35 * torch.tanh(self.fusion_head(fusion_in).squeeze(-1))

        return (
            self.global_mean
            + self.level_shift
            + bu
            + bi
            + self.gmf_scale * gmf_dot
            + nonlinear_delta
        )

    def predict_proba(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        like_threshold: float = 3.65,
        temperature: float = 2.5,
    ) -> torch.Tensor:
        """Convert predicted continuous ratings into calibrated probabilities P(like = 1)."""
        pred_rating = self.forward(user_idx, item_idx)
        return torch.sigmoid(temperature * (pred_rating - like_threshold))

    def get_config(self) -> dict:
        """Return serializable architecture hyperparameters for checkpoint reconstruction."""
        return {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "n_genres": self.n_genres,
            "n_user_Prior_feats": self.n_user_prior_feats,
            "n_item_prior_feats": self.n_item_prior_feats,
            "gmf_dim": self.gmf_dim,
            "mlp_emb_dim": self.mlp_emb_dim,
            "genre_proj_dim": self.genre_proj_dim,
            "mlp_hidden_dims": list(self.mlp_hidden_dims),
            "dropout_rate": self.dropout_rate,
            "global_mean": float(self.global_mean.item()),
        }


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total, trainable, and frozen parameters of a PyTorch module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(total - trainable),
    }


def verify_forward_pass(
    n_users: int = 14_998,
    n_items: int = 11_589,
    n_genres: int = 20,
    batch_size: int = 64,
) -> dict:
    """Instantiate HybridNeuMF, run a sample forward pass, and verify output shapes."""
    torch.manual_seed(42)
    model = HybridNeuMF(
        n_users=n_users,
        n_items=n_items,
        n_genres=n_genres,
        gmf_dim=64,
        mlp_emb_dim=32,
        genre_proj_dim=16,
        mlp_hidden_dims=(128, 64, 32),
        dropout_rate=0.30,
    )
    model.eval()

    sample_u = torch.randint(0, n_users, (batch_size,), dtype=torch.long)
    sample_i = torch.randint(0, n_items, (batch_size,), dtype=torch.long)

    with torch.no_grad():
        preds = model(sample_u, sample_i)
        probs = model.predict_proba(sample_u, sample_i)

    param_info = count_parameters(model)
    assert preds.shape == (batch_size,), f"Expected shape ({batch_size},), got {preds.shape}"
    assert probs.shape == (batch_size,), f"Expected shape ({batch_size},), got {probs.shape}"

    print("=== Architecture Verification (HybridNeuMF) ===")
    print(model)
    print(
        f"\nParameter Counts -> Total: {param_info['total']:,} | "
        f"Trainable: {param_info['trainable']:,} | Frozen: {param_info['frozen']:,}"
    )
    print(f"Sample Batch Size : {batch_size}")
    print(f"Output Rating Shape: {tuple(preds.shape)} | Range: [{preds.min():.4f}, {preds.max():.4f}]")
    print(f"Output Prob Shape  : {tuple(probs.shape)} | Range: [{probs.min():.4f}, {probs.max():.4f}]")
    return {
        "model": model,
        "param_info": param_info,
        "output_shape": tuple(preds.shape),
    }


if __name__ == "__main__":
    verify_forward_pass()
