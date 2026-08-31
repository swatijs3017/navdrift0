"""
SNAP-Corrector: Differentiable Trajectory Correction on GNSS Reacquisition.

When a GNSS fix returns after an outage, SNAP takes:
  - The stored sequence of DRIFT-Former latent states during the dark period
  - The covariance at each step
  - The new GNSS fix (position + heading)

And solves a constrained least-squares problem via gradient descent to find
the minimal correction to the trajectory such that the endpoint matches the
GNSS fix, respecting covariance constraints at each step.

This produces a smooth corrected path with no discontinuity jump.

Runtime: 10-20 gradient steps, target <50ms on CPU.
"""

import time
from typing import Tuple, Optional, List, Dict

import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Trajectory integrator: convert deltas to absolute poses
# ---------------------------------------------------------------------------

def integrate_se2(
    initial_pose: torch.Tensor,
    deltas: torch.Tensor,
) -> torch.Tensor:
    """
    Integrate SE(2) pose deltas to recover absolute poses.

    Uses the vehicle-frame kinematic model:
        x_{t+1} = x_t + cos(theta_t)*dx - sin(theta_t)*dy
        y_{t+1} = y_t + sin(theta_t)*dx + cos(theta_t)*dy
        theta_{t+1} = wrap(theta_t + dtheta)

    Args:
        initial_pose: (3,) or (B, 3) — [x0, y0, theta0]
        deltas:       (T, 3) or (B, T, 3) — [dx, dy, dtheta]

    Returns:
        poses: (T+1, 3) or (B, T+1, 3)
    """
    batched = deltas.dim() == 3
    if not batched:
        deltas       = deltas.unsqueeze(0)
        initial_pose = initial_pose.unsqueeze(0)

    B, T, _ = deltas.shape
    poses = [initial_pose]                         # (B, 3)

    for t in range(T):
        prev = poses[-1]
        dx, dy, dtheta = deltas[:, t, 0], deltas[:, t, 1], deltas[:, t, 2]
        theta = prev[:, 2]
        c, s = theta.cos(), theta.sin()

        new_x     = prev[:, 0] + c * dx - s * dy
        new_y     = prev[:, 1] + s * dx + c * dy
        new_theta = prev[:, 2] + dtheta
        # Wrap theta to [-pi, pi]
        new_theta = torch.atan2(new_theta.sin(), new_theta.cos())

        poses.append(torch.stack([new_x, new_y, new_theta], dim=-1))

    result = torch.stack(poses, dim=1)             # (B, T+1, 3)
    return result if batched else result.squeeze(0)


# ---------------------------------------------------------------------------
# SNAP-Corrector
# ---------------------------------------------------------------------------

class SNAPCorrector(nn.Module):
    """
    Differentiable trajectory smoother for GNSS reacquisition.

    The corrector learns a set of additive corrections (delta_corrections)
    to the raw DR pose deltas such that the integrated endpoint matches the
    new GNSS fix, while minimizing the weighted correction magnitude
    (weighted by the inverse covariance at each step).

    This is equivalent to solving:
        argmin_eps  sum_t [ eps_t^T * Sigma_t^{-1} * eps_t ]
        subject to: integrate(deltas + eps)[-1] == gnss_fix

    We solve it with gradient descent (no closed-form due to SE(2) nonlinearity).
    """

    def __init__(
        self,
        n_steps:    int   = 15,
        lr:         float = 0.05,
        max_ms:     float = 50.0,    # hard time budget in milliseconds
    ):
        super().__init__()
        self.n_steps = n_steps
        self.lr      = lr
        self.max_ms  = max_ms

    @torch.no_grad()
    def correct(
        self,
        initial_pose: torch.Tensor,      # (3,)  — pose at GNSS loss start
        raw_deltas:   torch.Tensor,      # (T, 3) — raw DR pose deltas
        cov_seq:      torch.Tensor,      # (T, 3, 3) — per-step covariance
        gnss_fix:     torch.Tensor,      # (3,)  — new GNSS fix [x, y, theta]
    ) -> Dict[str, torch.Tensor]:
        """
        Run the SNAP correction optimization.

        This function re-enables gradients internally for the optimizer loop,
        but the outer decorator prevents accidental gradient accumulation.

        Returns:
            {
              'corrected_deltas': (T, 3)   — corrected pose deltas
              'corrected_poses':  (T+1, 3) — integrated corrected trajectory
              'endpoint_error_m': float    — residual position error in metres
              'runtime_ms':       float    — wall-clock time
            }
        """
        start_time = time.perf_counter()

        # Clone and enable gradient tracking for the correction variable
        eps = torch.zeros_like(raw_deltas, requires_grad=True)

        # Compute per-step precision matrices (inverse covariance)
        eye = torch.eye(3, device=cov_seq.device).unsqueeze(0) * 1e-4
        cov_stable   = cov_seq + eye
        precision    = torch.linalg.inv(cov_stable)              # (T, 3, 3)

        optimizer = optim.Adam([eps], lr=self.lr)

        for step in range(self.n_steps):
            # Time budget check
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if elapsed_ms > self.max_ms * 0.9:
                break

            optimizer.zero_grad()

            corrected_deltas = raw_deltas + eps              # (T, 3)

            # Integrate to get the corrected trajectory
            with torch.enable_grad():
                corrected_deltas_grad = raw_deltas.detach() + eps
                poses = integrate_se2(initial_pose, corrected_deltas_grad)

            # Endpoint loss: squared distance to GNSS fix
            endpoint = poses[-1]                             # (3,)
            pos_err  = endpoint[:2] - gnss_fix[:2]
            heading_err = torch.atan2(
                (endpoint[2] - gnss_fix[2]).sin(),
                (endpoint[2] - gnss_fix[2]).cos()
            )
            endpoint_loss = (pos_err * pos_err).sum() + heading_err.pow(2)

            # Regularization: minimize correction weighted by precision
            eps_col   = eps.unsqueeze(-1)                    # (T, 3, 1)
            reg_loss  = (eps_col.transpose(-2, -1) @ precision @ eps_col
                         ).squeeze().mean()

            loss = endpoint_loss + 0.1 * reg_loss
            loss.backward()
            optimizer.step()

        # Final output — no grad
        with torch.no_grad():
            corrected_deltas = (raw_deltas + eps).detach()
            corrected_poses  = integrate_se2(initial_pose, corrected_deltas)
            endpoint         = corrected_poses[-1]
            err_m = (endpoint[:2] - gnss_fix[:2]).norm().item()

        runtime_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "corrected_deltas": corrected_deltas,
            "corrected_poses":  corrected_poses,
            "endpoint_error_m": err_m,
            "runtime_ms":       runtime_ms,
        }

    def smooth_path(
        self,
        raw_poses: torch.Tensor,         # (T, 3) — raw DR path
        gnss_fix:  torch.Tensor,         # (3,)   — new fix
        alpha:     float = 0.1,          # smoothing strength
    ) -> torch.Tensor:
        """
        Simple linear interpolation fallback for fast correction when the
        full optimizer is unavailable (e.g., on extremely low-power devices).

        Distributes the positional error linearly along the trajectory.

        Args:
            raw_poses: (T, 3)
            gnss_fix:  (3,)
            alpha:     lerp fraction

        Returns:
            smoothed_poses: (T, 3)
        """
        T = raw_poses.shape[0]
        endpoint_err = gnss_fix - raw_poses[-1]          # (3,)
        # Linearly ramp the correction from 0 at t=0 to endpoint_err at t=T-1
        ramp = torch.linspace(0.0, 1.0, T, device=raw_poses.device)
        correction = ramp.unsqueeze(-1) * endpoint_err.unsqueeze(0)
        smoothed = raw_poses + alpha * correction
        # Re-wrap heading
        smoothed[:, 2] = torch.atan2(smoothed[:, 2].sin(), smoothed[:, 2].cos())
        return smoothed


# ---------------------------------------------------------------------------
# Trajectory state buffer: stores DR latents and deltas during GNSS outage
# ---------------------------------------------------------------------------

class TrajectoryBuffer:
    """
    Stores the running dead-reckoning state during a GNSS outage window.

    Used by the inference pipeline to accumulate everything SNAP needs.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.raw_deltas:  List[torch.Tensor] = []
        self.cov_seq:     List[torch.Tensor] = []
        self.latents:     List[torch.Tensor] = []
        self.initial_pose: Optional[torch.Tensor] = None
        self.last_known_time: Optional[float] = None

    def push(
        self,
        delta:   torch.Tensor,     # (3,)
        cov:     torch.Tensor,     # (3, 3)
        latent:  Optional[torch.Tensor] = None,  # (d_model,)
        timestamp: Optional[float] = None,
    ) -> None:
        self.raw_deltas.append(delta.detach().cpu())
        self.cov_seq.append(cov.detach().cpu())
        if latent is not None:
            self.latents.append(latent.detach().cpu())
        if timestamp is not None:
            self.last_known_time = timestamp

    def set_initial_pose(self, pose: torch.Tensor) -> None:
        self.initial_pose = pose.detach().cpu()

    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (raw_deltas, cov_seq, initial_pose) as stacked tensors."""
        return (
            torch.stack(self.raw_deltas),       # (T, 3)
            torch.stack(self.cov_seq),           # (T, 3, 3)
            self.initial_pose,                   # (3,)
        )

    def __len__(self) -> int:
        return len(self.raw_deltas)
