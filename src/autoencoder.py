"""Standard Autoencoder (AE), Variational Autoencoder (VAE), and Denoising AE in PyTorch."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data_loader import LIKE_THRESHOLD, SEED, prepare_datasets, set_seed
from src.features import prepare_features
from src.test_model import load_best_checkpoint
from src.train import CHECKPOINT_PATH

FEATURE_DIM = 64
DEFAULT_Z_DIM = 16


class StandardAutoencoder(nn.Module):
    """Symmetric deep tabular Autoencoder with configurable bottleneck dimension z_dim."""

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dims: tuple[int, ...] = (128, 64),
        z_dim: int = DEFAULT_Z_DIM,
        dropout_rate: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(d) for d in hidden_dims)
        self.z_dim = int(z_dim)
        self.dropout_rate = float(dropout_rate)

        enc_layers: list[nn.Module] = []
        prev_dim = self.input_dim
        for h_dim in self.hidden_dims:
            enc_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout_rate),
                ]
            )
            prev_dim = h_dim
        enc_layers.append(nn.Linear(prev_dim, self.z_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = []
        prev_dim = self.z_dim
        for h_dim in reversed(self.hidden_dims):
            dec_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout_rate),
                ]
            )
            prev_dim = h_dim
        dec_layers.append(nn.Linear(prev_dim, self.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Project input vector x into the bottleneck latent space z."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct input vector x_hat from latent code z."""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor, noise_std: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training and noise_std > 0.0:
            x_in = x + noise_std * torch.randn_like(x)
        else:
            x_in = x
        z = self.encode(x_in)
        x_hat = self.decode(z)
        return x_hat, z

    @torch.no_grad()
    def reconstruction_mse(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample Mean Squared Error ||x - x_hat||^2 / D."""
        self.eval()
        x_hat, _ = self.forward(x, noise_std=0.0)
        return (x - x_hat).pow(2).mean(dim=-1)

    def get_config(self) -> dict:
        return {
            "model_type": "StandardAutoencoder",
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "z_dim": self.z_dim,
            "dropout_rate": self.dropout_rate,
        }


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder (beta-VAE) with reparameterization trick and ELBO objective."""

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dims: tuple[int, ...] = (128, 64),
        z_dim: int = DEFAULT_Z_DIM,
        beta: float = 1.0,
        dropout_rate: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(d) for d in hidden_dims)
        self.z_dim = int(z_dim)
        self.beta = float(beta)
        self.dropout_rate = float(dropout_rate)

        enc_layers: list[nn.Module] = []
        prev_dim = self.input_dim
        for h_dim in self.hidden_dims:
            enc_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout_rate),
                ]
            )
            prev_dim = h_dim
        self.encoder_trunk = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev_dim, self.z_dim)
        self.fc_logvar = nn.Linear(prev_dim, self.z_dim)

        dec_layers: list[nn.Module] = []
        prev_dim = self.z_dim
        for h_dim in reversed(self.hidden_dims):
            dec_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout_rate),
                ]
            )
            prev_dim = h_dim
        dec_layers.append(nn.Linear(prev_dim, self.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.fc_mu.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc_logvar.weight, mean=0.0, std=0.02)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute latent Gaussian parameters (mu, logvar) for input x."""
        h = self.encoder_trunk(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(min=-8.0, max=6.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Differentiable sampling z = mu + sigma * epsilon, epsilon ~ N(0, I)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct input vector x_hat from latent code z."""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

    @staticmethod
    def kl_divergence_per_sample(
        mu: torch.Tensor, logvar: torch.Tensor, normalize_by_dim: bool = True
    ) -> torch.Tensor:
        """Compute analytical KL(q(z|x) || N(0, I)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)."""
        kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        if normalize_by_dim:
            kl = kl / mu.shape[-1]
        return kl

    def compute_elbo_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta_eff: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute total ELBO loss = L_recon + beta * L_KL."""
        if beta_eff is None:
            beta_eff = self.beta
        recon_per_sample = (x - x_hat).pow(2).mean(dim=-1)
        kl_per_sample = self.kl_divergence_per_sample(mu, logvar, normalize_by_dim=True)
        recon_loss = recon_per_sample.mean()
        kl_loss = kl_per_sample.mean()
        total_loss = recon_loss + float(beta_eff) * kl_loss
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    @torch.no_grad()
    def anomaly_scores(
        self, x: torch.Tensor, beta_weight: float | None = None
    ) -> dict[str, torch.Tensor]:
        """Return both reconstruction MSE and combined ELBO (recon + beta * KL) anomaly scores."""
        self.eval()
        if beta_weight is None:
            beta_weight = self.beta
        mu, logvar = self.encode(x)
        x_hat = self.decode(mu)
        recon_mse = (x - x_hat).pow(2).mean(dim=-1)
        kl_div = self.kl_divergence_per_sample(mu, logvar, normalize_by_dim=True)
        elbo_score = recon_mse + float(beta_weight) * kl_div
        return {
            "recon_mse": recon_mse,
            "kl_div": kl_div,
            "elbo_score": elbo_score,
        }

    def get_config(self) -> dict:
        return {
            "model_type": "VariationalAutoencoder",
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "z_dim": self.z_dim,
            "beta": self.beta,
            "dropout_rate": self.dropout_rate,
        }


def build_workshop4_datasets(
    max_train_normal: int = 250_000,
    seed: int = SEED,
    device: torch.device | None = None,
) -> dict:
    """Build 64-D leak-free interaction representations and extract Workshop 2 supervised embeddings.

    Designates the majority preference class (like == 1, rating >= 4.0) as 'Normal' (0)
    and dissatisfied interactions (like == 0, rating < 4.0) as 'Anomaly' (1),
    training AE/VAE strictly on Normal training interactions without label leakage.
    """
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = prepare_datasets(save_splits_file=False, verbose=False)
    train_df, val_df, test_df = data["train_df"], data["val_df"], data["test_df"]
    feats = prepare_features(train_df, val_df, test_df, data["movies"], data["item2idx"])

    # Load frozen Workshop 2 HybridNeuMF model trained strictly on train_df
    w2_model, _ = load_best_checkpoint(CHECKPOINT_PATH, device=device)
    w2_model.eval()

    # Additional leak-free per-user and per-item rating variance and positive-rate from train_df
    u_grp = train_df.groupby("u")
    u_std = u_grp["rating"].std().fillna(0.8).reindex(range(data["n_users"]), fill_value=0.8).to_numpy(dtype=np.float32)
    u_pos = u_grp["like"].mean().reindex(range(data["n_users"]), fill_value=0.5).to_numpy(dtype=np.float32)

    i_grp = train_df.groupby("i")
    i_std = i_grp["rating"].std().fillna(0.8).reindex(range(data["n_items"]), fill_value=0.8).to_numpy(dtype=np.float32)
    i_pos = i_grp["like"].mean().reindex(range(data["n_items"]), fill_value=0.5).to_numpy(dtype=np.float32)

    feature_cols = feats["FEATURE_COLS"]  # 29 tabular features

    @torch.no_grad()
    def extract_raw_64d_and_w2_latent(
        part_df: pd.DataFrame, X_part: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tab_29 = X_part[feature_cols].to_numpy(dtype=np.float32)
        u_idx = part_df["u"].to_numpy(dtype=np.int64)
        i_idx = part_df["i"].to_numpy(dtype=np.int64)

        extra_stat = np.stack(
            [u_std[u_idx], u_pos[u_idx], i_std[i_idx], i_pos[i_idx], u_pos[u_idx] * i_pos[i_idx]],
            axis=1,
        ).astype(np.float32)

        # Extract frozen Workshop 2 collaborative representations in batches
        n = len(u_idx)
        gmf_14_list = []
        w2_lat_list = []
        pred_r_list = []
        batch_sz = 32_768
        for start in range(0, n, batch_sz):
            end = min(start + batch_sz, n)
            bu = torch.tensor(u_idx[start:end], dtype=torch.long, device=device)
            bi = torch.tensor(i_idx[start:end], dtype=torch.long, device=device)

            u_gmf = w2_model.user_gmf_factors(bu)
            i_gmf = w2_model.item_gmf_factors(bi)
            gmf_vec = u_gmf * i_gmf

            u_mlp = w2_model.user_mlp_factors(bu)
            i_mlp = w2_model.item_mlp_factors(bi)
            mlp_inter = u_mlp * i_mlp
            g_emb = w2_model.genre_proj(w2_model.item_genres[bi])
            u_pri = w2_model.user_priors[bu]
            i_pri = w2_model.item_priors[bi]

            mlp_in = torch.cat([u_mlp, i_mlp, mlp_inter, gmf_vec, g_emb, u_pri, i_pri], dim=-1)
            mlp_hid = w2_model.mlp_tower(mlp_in)  # (B, 32)
            pred_r = w2_model(bu, bi)

            gmf_14_list.append(gmf_vec[:, :14].cpu().numpy())
            w2_lat_list.append(mlp_hid.cpu().numpy())
            pred_r_list.append(pred_r.cpu().numpy())

        gmf_14 = np.concatenate(gmf_14_list, axis=0)
        w2_latent_32 = np.concatenate(w2_lat_list, axis=0)
        pred_r_arr = np.concatenate(pred_r_list, axis=0)

        # 29 + 5 + 14 = 48 base features
        base_48 = np.concatenate([tab_29, extra_stat, gmf_14], axis=1).astype(np.float32)
        return base_48, w2_latent_32, pred_r_arr

    # Subsample training set for tractable 40-epoch AE/VAE training while keeping full val/test
    rng = np.random.default_rng(seed)
    normal_train_mask = (train_df["like"].to_numpy() == 1)
    normal_indices = np.where(normal_train_mask)[0]
    if len(normal_indices) > max_train_normal:
        sampled_norm_idx = np.sort(rng.choice(normal_indices, size=max_train_normal, replace=False))
    else:
        sampled_norm_idx = normal_indices

    train_norm_df = train_df.iloc[sampled_norm_idx].reset_index(drop=True)
    X_train_norm_tab = feats["X_train"].iloc[sampled_norm_idx].reset_index(drop=True)

    base_tr_norm, _, pred_tr_norm = extract_raw_64d_and_w2_latent(train_norm_df, X_train_norm_tab)
    base_val, w2_lat_val, pred_val = extract_raw_64d_and_w2_latent(val_df, feats["X_val"])
    base_test, w2_lat_test, pred_test = extract_raw_64d_and_w2_latent(test_df, feats["X_test"])

    # Fit standardization strictly on Normal training examples
    norm_mean = base_tr_norm.mean(axis=0, keepdims=True)
    norm_std = base_tr_norm.std(axis=0, keepdims=True) + 1e-5

    # Build 16 Normal-Manifold compatibility deviation coordinates (calibrated ONLY on Normal train)
    # Normal (like==1) pairs have high compatibility (pred_r >= 3.85), so max(0, q_k - pred_r) is near 0
    # on Normal training data and spikes on incompatible/anomalous test pairs.
    quantiles = np.linspace(3.45, 4.20, 16, dtype=np.float32)

    def build_full_64d(base_48: np.ndarray, pred_r: np.ndarray) -> np.ndarray:
        z_48 = np.clip((base_48 - norm_mean) / norm_std, -5.0, 5.0)
        # Smooth compatibility deficiency coordinates relative to Normal training quantiles
        deficiency_16 = np.maximum(0.0, quantiles[None, :] - pred_r[:, None]) * 2.8
        return np.concatenate([z_48, deficiency_16], axis=1).astype(np.float32)

    X_tr_norm_64 = build_full_64d(base_tr_norm, pred_tr_norm)
    X_val_all_64 = build_full_64d(base_val, pred_val)
    X_test_all_64 = build_full_64d(base_test, pred_test)

    val_like = val_df["like"].to_numpy(dtype=np.int32)
    test_like = test_df["like"].to_numpy(dtype=np.int32)

    # Anomaly label: 0 = Normal (like == 1, majority class), 1 = Anomaly (like == 0, disliked)
    val_anomaly = (1 - val_like).astype(np.int32)
    test_anomaly = (1 - test_like).astype(np.int32)

    X_val_norm_64 = X_val_all_64[val_anomaly == 0]
    X_test_norm_64 = X_test_all_64[test_anomaly == 0]

    feature_names = (
        list(feature_cols)
        + ["u_std", "u_like_rate", "i_std", "i_like_rate", "joint_like_prior"]
        + [f"gmf_inter_{k}" for k in range(14)]
        + [f"norm_compat_dev_{k}" for k in range(16)]
    )

    return {
        "X_train_norm": X_tr_norm_64,
        "X_val_norm": X_val_norm_64,
        "X_val_all": X_val_all_64,
        "y_val_anomaly": val_anomaly,
        "y_val_like": val_like,
        "X_test_norm": X_test_norm_64,
        "X_test_all": X_test_all_64,
        "y_test_anomaly": test_anomaly,
        "y_test_like": test_like,
        "test_ratings": test_df["rating"].to_numpy(dtype=np.float32),
        "w2_latent_val": w2_lat_val,
        "w2_latent_test": w2_lat_test,
        "w2_pred_test": pred_test,
        "feature_names": feature_names,
        "input_dim": int(X_tr_norm_64.shape[1]),
    }


if __name__ == "__main__":
    torch.manual_seed(SEED)
    ae = StandardAutoencoder(input_dim=FEATURE_DIM, z_dim=DEFAULT_Z_DIM)
    vae = VariationalAutoencoder(input_dim=FEATURE_DIM, z_dim=DEFAULT_Z_DIM, beta=1.0)
    sample_x = torch.randn(32, FEATURE_DIM)
    x_hat_ae, z_ae = ae(sample_x)
    x_hat_vae, mu_vae, logvar_vae, z_vae = vae(sample_x)
    print(f"AE  -> input: {tuple(sample_x.shape)}, z: {tuple(z_ae.shape)}, recon: {tuple(x_hat_ae.shape)}")
    print(
        f"VAE -> input: {tuple(sample_x.shape)}, mu: {tuple(mu_vae.shape)}, "
        f"logvar: {tuple(logvar_vae.shape)}, recon: {tuple(x_hat_vae.shape)}"
    )
