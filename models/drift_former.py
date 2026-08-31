"""
DRIFT-Former: Causal Transformer for dead-reckoning pose estimation.

Architecture:
- Input projection: 8 features -> d_model=256
- Rotary Position Encoding (RoPE) — respects physical time ordering
- 4 causal Transformer encoder layers, 8 heads
- Heteroscedastic output head: SE(2) pose delta mean + 3x3 covariance

The model outputs a Gaussian distribution over SE(2) pose deltas, enabling
uncertainty-aware trajectory estimation during GNSS outages.
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Rotary Position Encoding (RoPE)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(dim: int, max_seq_len: int,
                           base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cosine and sine frequency tables for RoPE.

    Args:
        dim:         head dimension (must be even)
        max_seq_len: maximum sequence length
        base:        frequency base

    Returns:
        cos_table: (max_seq_len, dim/2)
        sin_table: (max_seq_len, dim/2)
    """
    assert dim % 2 == 0, "RoPE requires even head dimension"
    half = dim // 2
    theta = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, theta)          # (T, half)
    return freqs.cos(), freqs.sin()


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Encoding to query and key tensors.

    Args:
        q, k:    (B, heads, T, head_dim)
        cos, sin: (T, head_dim/2)

    Returns:
        q_rot, k_rot with the same shape
    """
    def rotate(x: torch.Tensor) -> torch.Tensor:
        # Split into two halves along the head_dim
        x1, x2 = x[..., ::2], x[..., 1::2]
        c = cos[:x.shape[-3]].to(x.device)    # (T, D/2)
        s = sin[:x.shape[-3]].to(x.device)
        # Broadcast: (B, H, T, D/2)
        c = c.unsqueeze(0).unsqueeze(0)
        s = s.unsqueeze(0).unsqueeze(0)
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        # Interleave back
        out = torch.zeros_like(x)
        out[..., ::2]  = x1 * c - x2 * s
        out[..., 1::2] = x2 * c + x1 * s
        return out

    return rotate(q), rotate(k)


# ---------------------------------------------------------------------------
# Causal Multi-Head Attention with RoPE
# ---------------------------------------------------------------------------

class CausalRoPEAttention(nn.Module):
    """
    Causal multi-head self-attention with rotary position encoding.
    Uses a causal mask so each position can only attend to earlier positions.
    """

    def __init__(self, d_model: int, n_heads: int,
                 dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)
        self.dropout   = nn.Dropout(dropout)

        cos, sin = precompute_rope_freqs(self.head_dim, max_seq_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        # Causal mask — registered as buffer so it moves to device automatically
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            out: (B, T, d_model)
        """
        B, T, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)             # each: (B, T, H, head_dim)
        q = q.permute(0, 2, 1, 3)               # (B, H, T, head_dim)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Apply RoPE
        q, k = apply_rope(q, k, self.rope_cos, self.rope_sin)

        # Scaled dot-product attention with causal mask
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)              # (B, H, T, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Transformer encoder block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block with causal RoPE attention."""

    def __init__(self, d_model: int, n_heads: int,
                 ffn_mult: int = 4, dropout: float = 0.1,
                 max_seq_len: int = 512):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = CausalRoPEAttention(d_model, n_heads,
                                          dropout=dropout,
                                          max_seq_len=max_seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Heteroscedastic output head
# ---------------------------------------------------------------------------

class HeteroscedasticHead(nn.Module):
    """
    Produces a Gaussian distribution over SE(2) pose deltas.

    Outputs:
        mean:      (B, T, 3)     — [dx, dy, dtheta]
        log_diag:  (B, T, 3)     — log diagonal of lower-triangular L
        L_off:     (B, T, 3)     — off-diagonal entries of L
                                   such that Sigma = L @ L.T

    This parameterization ensures Sigma is always symmetric positive definite.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Mean head
        self.mean_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 3),
        )
        # Covariance: 3 log-diagonal + 3 off-diagonal = 6 params
        self.cov_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 6),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            mean:  (B, T, 3)
            cov:   (B, T, 3, 3)  — full covariance matrix
        """
        mean = self.mean_head(x)                    # (B, T, 3)
        cov_params = self.cov_head(x)               # (B, T, 6)

        log_diag = cov_params[..., :3]              # (B, T, 3)
        off_diag = cov_params[..., 3:]              # (B, T, 3)

        # Build lower-triangular L
        diag = torch.exp(log_diag).clamp(min=1e-4)  # ensure positive diagonal
        B, T, _ = mean.shape
        L = torch.zeros(B, T, 3, 3, device=x.device, dtype=x.dtype)
        L[..., 0, 0] = diag[..., 0]
        L[..., 1, 0] = off_diag[..., 0]
        L[..., 1, 1] = diag[..., 1]
        L[..., 2, 0] = off_diag[..., 1]
        L[..., 2, 1] = off_diag[..., 2]
        L[..., 2, 2] = diag[..., 2]

        # Sigma = L @ L.T
        cov = torch.matmul(L, L.transpose(-2, -1))
        return mean, cov


# ---------------------------------------------------------------------------
# DRIFT-Former: the full model
# ---------------------------------------------------------------------------

class DRIFTFormer(nn.Module):
    """
    Causal Transformer encoder for dead-reckoning.

    Input:  (B, T, 8)   — [accel_xyz(3) | gyro_xyz(3) | speed(1) | steer(1)]
    Output: mean (B, T, 3) and covariance (B, T, 3, 3) for SE(2) pose deltas.

    The latent states (B, T, d_model) are stored during inference and used by
    SNAP-Corrector for differentiable trajectory correction.
    """

    def __init__(
        self,
        input_dim:   int = 8,
        d_model:     int = 256,
        n_layers:    int = 4,
        n_heads:     int = 8,
        ffn_mult:    int = 4,
        dropout:     float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_mult, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)
        self.head     = HeteroscedasticHead(d_model)

    def forward(
        self,
        x: torch.Tensor,
        return_latents: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x:              (B, T, 8) normalized sensor input
            return_latents: if True, also return the (B, T, d_model) latent

        Returns:
            mean:     (B, T, 3)       — pose delta mean
            cov:      (B, T, 3, 3)    — pose delta covariance
            latents:  (B, T, d_model) — only if return_latents=True, else None
        """
        h = self.input_proj(x)                  # (B, T, d_model)
        for block in self.blocks:
            h = block(h)
        h = self.out_norm(h)
        mean, cov = self.head(h)
        latents = h if return_latents else None
        return mean, cov, latents

    def get_uncertainty_ellipse(self, cov: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract 2D position uncertainty ellipse axes from the 3x3 covariance.

        Args:
            cov: (B, T, 3, 3) — full SE(2) covariance

        Returns:
            semi_major: (B, T) — major axis length (metres)
            semi_minor: (B, T) — minor axis length (metres)
        """
        pos_cov = cov[..., :2, :2]               # (B, T, 2, 2)
        # Eigenvalues of the 2x2 position covariance
        eigenvalues = torch.linalg.eigvalsh(pos_cov)   # (B, T, 2)
        semi_major = eigenvalues[..., -1].sqrt()
        semi_minor = eigenvalues[..., 0].sqrt()
        return semi_major, semi_minor


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def nll_se2_loss(
    mean: torch.Tensor,
    cov: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    geodesic_weight: float = 0.5,
) -> torch.Tensor:
    """
    Negative log-likelihood under the predicted Gaussian + SE(2) geodesic error.

    Loss = NLL(targets | mean, Sigma) + geodesic_weight * geodesic_error

    Only computes loss over timesteps where mask=False (GNSS is absent),
    because those are the timesteps the model must predict without GPS help.

    Args:
        mean:    (B, T, 3)
        cov:     (B, T, 3, 3)
        targets: (B, T, 3)   — ground-truth pose deltas
        mask:    (B, T)      — True where GPS is available
        geodesic_weight: weight on the geodesic term

    Returns:
        scalar loss
    """
    B, T, _ = mean.shape

    # Gaussian NLL: -log p(y | mu, Sigma) = 0.5*(log det Sigma + (y-mu)^T Sigma^{-1} (y-mu))
    err = targets - mean                              # (B, T, 3)

    # Add jitter for numerical stability
    eye = torch.eye(3, device=cov.device).unsqueeze(0).unsqueeze(0) * 1e-5
    cov_stable = cov + eye

    # Cholesky solve for efficiency
    L = torch.linalg.cholesky(cov_stable)
    log_det = 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1)  # (B, T)

    # Solve L @ v = err  => v = L^{-1} err
    err_col = err.unsqueeze(-1)                      # (B, T, 3, 1)
    v = torch.linalg.solve_triangular(L, err_col, upper=False)  # (B, T, 3, 1)
    mahal = (v * v).squeeze(-1).sum(-1)              # (B, T)

    nll = 0.5 * (log_det + mahal)                    # (B, T)

    # Geodesic error: translation L2 + heading angular distance
    trans_err  = err[..., :2].norm(dim=-1)            # (B, T)
    angle_err  = (err[..., 2] + math.pi) % (2 * math.pi) - math.pi
    geodesic   = trans_err + angle_err.abs()

    total = nll + geodesic_weight * geodesic           # (B, T)

    # Apply mask: only penalize outage timesteps
    if mask is not None:
        # mask=True means GPS available; we train on masked (GPS-dark) steps
        outage_mask = ~mask[:, 1:]                     # (B, T)  align with deltas
        if outage_mask.any():
            total = total[outage_mask]

    return total.mean()


def count_parameters(model: nn.Module) -> int:
    """Return total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
