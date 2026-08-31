"""
NavIC Motion Prior VAE.

A conditional beta-VAE that encodes a 60-second GPS/NavIC trajectory history
into a 32-dimensional latent prior. During dead reckoning, this prior is
fused with DRIFT-Former's output via KL-weighted posterior update — the core
novelty that constrains drift using learned Indian driving patterns.

Architecture:
  Encoder: GRU + MLP -> mu (32,), log_var (32,)
  Decoder: MLP + GRU -> reconstructed trajectory (60, 4)
  Prior fusion: product of Gaussians between DRIFT-Former posterior and VAE prior
"""

import math
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class TrajectoryEncoder(nn.Module):
    """
    Encodes a GPS trajectory sequence into a latent distribution (mu, log_var).

    Input:  (B, T, 4) — [x_m, y_m, speed_m_s, heading_rad] at 1 Hz
    Output: mu (B, latent_dim), log_var (B, latent_dim)
    """

    def __init__(
        self,
        input_dim:   int = 4,
        hidden_dim:  int = 128,
        n_layers:    int = 2,
        latent_dim:  int = 32,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False,    # causal — no future lookahead
        )
        self.mu_head      = nn.Linear(hidden_dim, latent_dim)
        self.log_var_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, traj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            traj: (B, T, 4)
        Returns:
            mu:      (B, latent_dim)
            log_var: (B, latent_dim)
        """
        _, h_n = self.gru(traj)         # h_n: (n_layers, B, hidden_dim)
        h = h_n[-1]                     # take last layer's hidden state
        return self.mu_head(h), self.log_var_head(h)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class TrajectoryDecoder(nn.Module):
    """
    Decodes a latent vector back to a trajectory sequence.

    Input:  z (B, latent_dim)
    Output: (B, T, 4) reconstructed trajectory
    """

    def __init__(
        self,
        latent_dim:  int = 32,
        hidden_dim:  int = 128,
        n_layers:    int = 2,
        output_dim:  int = 4,
        output_len:  int = 60,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.output_len = output_len
        # Project latent to initial hidden state
        self.init_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * n_layers),
            nn.Tanh(),
        )
        self.gru = nn.GRU(
            input_size=1,           # dummy step input; conditioning via hidden
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self.n_layers  = n_layers
        self.hidden_dim = hidden_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim)
        Returns:
            traj: (B, T, 4)
        """
        B = z.shape[0]
        # Initialize hidden state from latent
        h0 = self.init_proj(z)                          # (B, n_layers * hidden_dim)
        h0 = h0.view(B, self.n_layers, self.hidden_dim)
        h0 = h0.permute(1, 0, 2).contiguous()          # (n_layers, B, hidden_dim)

        # Dummy input tokens
        dummy = torch.zeros(B, self.output_len, 1, device=z.device)
        out, _ = self.gru(dummy, h0)                    # (B, T, hidden_dim)
        return self.out_proj(out)                        # (B, T, 4)


# ---------------------------------------------------------------------------
# NavIC Motion Prior VAE
# ---------------------------------------------------------------------------

class NavICMotionPriorVAE(nn.Module):
    """
    Beta-VAE for learning a motion prior from NavIC/GPS trajectory history.

    Usage during training:
        mu, log_var, z, recon = vae(traj)
        loss = vae_loss(recon, traj, mu, log_var, beta=0.5)

    Usage during dead reckoning (prior fusion):
        vae_mu, vae_log_var = vae.encode(pre_loss_traj)
        fused_mu, fused_log_var = vae.fuse_with_dr_posterior(
            dr_mu, dr_log_var, vae_mu, vae_log_var, kl_weight=0.3
        )
    """

    def __init__(
        self,
        input_dim:   int   = 4,
        hidden_dim:  int   = 128,
        n_layers:    int   = 2,
        latent_dim:  int   = 32,
        output_len:  int   = 60,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = TrajectoryEncoder(input_dim, hidden_dim, n_layers,
                                            latent_dim, dropout)
        self.decoder    = TrajectoryDecoder(latent_dim, hidden_dim, n_layers,
                                            input_dim, output_len, dropout)

    def reparameterize(self, mu: torch.Tensor,
                        log_var: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(mu, exp(log_var)) using the reparameterization trick."""
        if self.training:
            std = (0.5 * log_var).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu    # use mean at inference for deterministic output

    def encode(self, traj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a trajectory into the latent distribution.

        Args:
            traj: (B, T, 4)
        Returns:
            mu:      (B, latent_dim)
            log_var: (B, latent_dim)
        """
        return self.encoder(traj)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent sample to trajectory. Returns (B, T, 4)."""
        return self.decoder(z)

    def forward(
        self, traj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass for training.

        Returns:
            mu:      (B, latent_dim)
            log_var: (B, latent_dim)
            z:       (B, latent_dim)  — sampled latent
            recon:   (B, T, 4)        — reconstructed trajectory
        """
        mu, log_var = self.encode(traj)
        z           = self.reparameterize(mu, log_var)
        recon       = self.decode(z)
        return mu, log_var, z, recon

    @staticmethod
    def fuse_with_dr_posterior(
        dr_mu: torch.Tensor,
        dr_log_var: torch.Tensor,
        prior_mu: torch.Tensor,
        prior_log_var: torch.Tensor,
        kl_weight: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse DRIFT-Former's posterior with the NavIC prior using the
        product-of-Gaussians formula.

        Given two Gaussians N(mu1, sigma1^2) and N(mu2, sigma2^2), their
        product (unnormalized) is N(mu_fused, sigma_fused^2) where:
            sigma_fused^2 = 1 / (1/sigma1^2 + kl_weight/sigma2^2)
            mu_fused = sigma_fused^2 * (mu1/sigma1^2 + kl_weight*mu2/sigma2^2)

        The kl_weight controls how strongly the prior pulls the estimate.

        Args:
            dr_mu, dr_log_var:      DRIFT-Former latent distribution (B, D)
            prior_mu, prior_log_var: VAE prior distribution (B, D)
            kl_weight:              mixing strength of the prior [0, 1]

        Returns:
            fused_mu:      (B, D)
            fused_log_var: (B, D)
        """
        var_dr    = dr_log_var.exp()
        var_prior = prior_log_var.exp()

        precision_dr    = 1.0 / (var_dr    + 1e-8)
        precision_prior = kl_weight / (var_prior + 1e-8)

        fused_var = 1.0 / (precision_dr + precision_prior + 1e-8)
        fused_mu  = fused_var * (precision_dr * dr_mu + precision_prior * prior_mu)
        fused_log_var = fused_var.log()
        return fused_mu, fused_log_var


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Beta-VAE loss = reconstruction MSE + beta * KL divergence.

    Args:
        recon:   (B, T, 4) — reconstructed trajectory
        target:  (B, T, 4) — ground-truth trajectory
        mu:      (B, latent_dim)
        log_var: (B, latent_dim)
        beta:    KL weight (0.5 for moderate disentanglement)

    Returns:
        dict with keys 'total', 'recon', 'kl'
    """
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    total = recon_loss + beta * kl
    return {"total": total, "recon": recon_loss, "kl": kl}
