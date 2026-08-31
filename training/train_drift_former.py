"""
Training script for DRIFT-Former.

Run:
    python -m navdrift0.training.train_drift_former \
        --data_root /path/to/IO-VNBD \
        --output_dir ./checkpoints/drift_former \
        --epochs 100 \
        --batch_size 64 \
        --wandb_project navdrift0

Trains the causal Transformer for SE(2) dead-reckoning with:
  - AdamW optimizer + cosine LR schedule with warmup
  - Gradient clipping at 1.0
  - Heteroscedastic NLL + geodesic loss on masked (GNSS-dark) steps
  - Best checkpoint saved by validation ATE (Absolute Trajectory Error)
  - Logs to Weights & Biases (free tier)
"""

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from navdrift0.data.loader import (
    IOVNBDParser, synchronize_sequence, compute_norm_stats,
    build_dataloaders,
)
from navdrift0.models.drift_former import DRIFTFormer, nll_se2_loss, count_parameters
from navdrift0.eval.metrics import compute_ate, compute_rte

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Warmup + cosine LR scheduler
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr: float = 1e-6):
        self.optimizer     = optimizer
        self.warmup_steps  = warmup_steps
        self.total_steps   = total_steps
        self.min_lr        = min_lr
        self.base_lrs      = [pg["lr"] for pg in optimizer.param_groups]
        self._step         = 0

    def step(self) -> None:
        self._step += 1
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = self._get_lr(base_lr)

    def _get_lr(self, base_lr: float) -> float:
        s = self._step
        if s < self.warmup_steps:
            return base_lr * s / max(1, self.warmup_steps)
        progress = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (base_lr - self.min_lr) * cosine

    def get_last_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Pose integration for eval (reimplemented without autograd for speed)
# ---------------------------------------------------------------------------

def integrate_poses_np(
    initial_pose: np.ndarray,   # (3,)
    deltas: np.ndarray,         # (T, 3)
) -> np.ndarray:
    """Integrate SE(2) deltas to poses in numpy (fast eval)."""
    T = len(deltas)
    poses = np.zeros((T + 1, 3), dtype=np.float32)
    poses[0] = initial_pose
    for t in range(T):
        dx, dy, dtheta = deltas[t]
        theta = poses[t, 2]
        c, s = np.cos(theta), np.sin(theta)
        poses[t + 1, 0] = poses[t, 0] + c * dx - s * dy
        poses[t + 1, 1] = poses[t, 1] + s * dx + c * dy
        poses[t + 1, 2] = np.arctan2(
            np.sin(poses[t, 2] + dtheta),
            np.cos(poses[t, 2] + dtheta)
        )
    return poses


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     DRIFTFormer,
    loader,
    optimizer,
    scheduler,
    device:    torch.device,
    clip_grad: float = 1.0,
    epoch:     int   = 0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        inputs   = batch["inputs"].to(device)     # (B, W, 8)
        targets  = batch["targets"].to(device)    # (B, W-1, 3)
        mask     = batch["mask"].to(device)       # (B, W)

        # Forward — predict deltas for positions 1..W from context 0..W-1
        mean, cov, _ = model(inputs)              # (B, W, 3), (B, W, 3, 3)
        # Align: predict delta at step t from input at step t
        mean_pred = mean[:, :-1, :]               # (B, W-1, 3)
        cov_pred  = cov[:, :-1, :, :]             # (B, W-1, 3, 3)
        mask_pred = mask[:, :-1]                  # (B, W-1)

        loss = nll_se2_loss(mean_pred, cov_pred, targets, mask_pred)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches  += 1

    return {"train/loss": total_loss / max(n_batches, 1),
            "train/lr":   scheduler.get_last_lr()}


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:  DRIFTFormer,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_ate    = []
    all_rte    = []
    n_batches  = 0

    for batch in loader:
        inputs   = batch["inputs"].to(device)
        targets  = batch["targets"].to(device)
        mask     = batch["mask"].to(device)
        poses_gt = batch["poses_gt"].numpy()        # (B, W, 3)

        mean, cov, _ = model(inputs)
        mean_pred = mean[:, :-1, :]
        cov_pred  = cov[:, :-1, :, :]
        mask_pred = mask[:, :-1]

        loss = nll_se2_loss(mean_pred, cov_pred, targets, mask_pred)
        total_loss += loss.item()
        n_batches  += 1

        # Compute ATE/RTE on the first item in the batch (fast check)
        pred_deltas_np = mean_pred[0].cpu().numpy()   # (W-1, 3)
        gt_poses_np    = poses_gt[0]                  # (W, 3)
        pred_poses_np  = integrate_poses_np(gt_poses_np[0], pred_deltas_np)

        # Only evaluate over outage timesteps
        mask_np = ~mask[0, :-1].cpu().numpy()
        if mask_np.any():
            ate = compute_ate(gt_poses_np[1:][mask_np], pred_poses_np[1:][mask_np])
            rte = compute_rte(gt_poses_np[1:][mask_np], pred_poses_np[1:][mask_np])
            all_ate.append(ate)
            all_rte.append(rte)

    metrics = {
        "val/loss": total_loss / max(n_batches, 1),
        "val/ate_m": float(np.mean(all_ate)) if all_ate else float("nan"),
        "val/rte_pct": float(np.mean(all_rte)) if all_rte else float("nan"),
    }
    return metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # W&B
    if WANDB_AVAILABLE and args.wandb_project:
        wandb.init(project=args.wandb_project,
                   config=vars(args),
                   name=args.run_name or "drift-former")
    else:
        logger.warning("wandb not available or not configured — logging to stdout only")

    # Load data
    parser_obj = IOVNBDParser()
    raw_sequences = parser_obj.parse_dataset(Path(args.data_root))
    synced = [s for seq in raw_sequences
              if (s := synchronize_sequence(seq)) is not None]
    logger.info("Usable sequences after sync: %d", len(synced))

    stats = compute_norm_stats(synced)
    stats.save(output_dir / "norm_stats.npz")
    logger.info("Normalization stats saved")

    train_loader, val_loader, test_loader = build_dataloaders(
        synced, stats,
        window=args.window, stride=args.stride,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    logger.info("Train batches: %d | Val batches: %d | Test batches: %d",
                len(train_loader), len(val_loader), len(test_loader))

    # Model
    model = DRIFTFormer(
        input_dim=8, d_model=256, n_layers=4, n_heads=8,
        ffn_mult=4, dropout=0.1, max_seq_len=args.window + 32,
    ).to(device)
    logger.info("DRIFT-Former parameters: %s", f"{count_parameters(model):,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    total_steps   = args.epochs * len(train_loader)
    warmup_steps  = int(0.05 * total_steps)
    scheduler     = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    best_ate = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer,
                                         scheduler, device,
                                         clip_grad=args.clip_grad, epoch=epoch)
        val_metrics   = evaluate(model, val_loader, device)

        log_line = (f"Epoch {epoch:03d} | "
                    f"train_loss={train_metrics['train/loss']:.4f} | "
                    f"val_loss={val_metrics['val/loss']:.4f} | "
                    f"val_ATE={val_metrics['val/ate_m']:.3f}m | "
                    f"lr={train_metrics['train/lr']:.2e}")
        logger.info(log_line)

        if WANDB_AVAILABLE and args.wandb_project:
            wandb.log({**train_metrics, **val_metrics, "epoch": epoch})

        # Save best checkpoint by ATE
        val_ate = val_metrics["val/ate_m"]
        if not math.isnan(val_ate) and val_ate < best_ate:
            best_ate   = val_ate
            best_epoch = epoch
            ckpt = {
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_ate_m":  best_ate,
                "args":       vars(args),
            }
            torch.save(ckpt, output_dir / "best_drift_former.pt")
            logger.info("New best saved: ATE=%.3fm at epoch %d", best_ate, epoch)

    logger.info("Training complete. Best ATE: %.3fm at epoch %d", best_ate, best_epoch)

    # Final evaluation on test set
    best_ckpt = torch.load(output_dir / "best_drift_former.pt", map_location=device)
    model.load_state_dict(best_ckpt["state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    logger.info("Test metrics: ATE=%.3fm | RTE=%.2f%%",
                test_metrics["val/ate_m"], test_metrics["val/rte_pct"])
    if WANDB_AVAILABLE and args.wandb_project:
        wandb.log({"test/ate_m": test_metrics["val/ate_m"],
                   "test/rte_pct": test_metrics["val/rte_pct"]})
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DRIFT-Former")
    p.add_argument("--data_root",     required=True)
    p.add_argument("--output_dir",    default="./checkpoints/drift_former")
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--clip_grad",     type=float, default=1.0)
    p.add_argument("--window",        type=int,   default=200)
    p.add_argument("--stride",        type=int,   default=50)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--wandb_project", default="navdrift0")
    p.add_argument("--run_name",      default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
