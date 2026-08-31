"""
Training script for the NavIC Motion Prior VAE.

Run:
    python -m navdrift0.training.train_navic_vae \
        --data_root /path/to/IO-VNBD \
        --output_dir ./checkpoints/navic_vae \
        --epochs 50 \
        --beta 0.5

The VAE is trained on the GPS-valid segments only.
It learns a compact latent representation of pre-loss driving patterns
that constrains DRIFT-Former during outages.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from navdrift0.data.loader import IOVNBDParser, synchronize_sequence, NavICVAEDataset
from navdrift0.models.navic_vae import NavICMotionPriorVAE, vae_loss

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     NavICMotionPriorVAE,
    loader:    DataLoader,
    optimizer: optim.Optimizer,
    device:    torch.device,
    beta:      float,
) -> Dict[str, float]:
    model.train()
    totals = {"total": 0.0, "recon": 0.0, "kl": 0.0}
    n = 0

    for traj in loader:
        traj = traj.to(device)                          # (B, T, 4)
        mu, log_var, z, recon = model(traj)
        losses = vae_loss(recon, traj, mu, log_var, beta=beta)

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k in totals:
            totals[k] += losses[k].item()
        n += 1

    return {f"train/{k}": v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate_vae(
    model:  NavICMotionPriorVAE,
    loader: DataLoader,
    device: torch.device,
    beta:   float,
) -> Dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "recon": 0.0, "kl": 0.0}
    n = 0

    for traj in loader:
        traj = traj.to(device)
        mu, log_var, z, recon = model(traj)
        losses = vae_loss(recon, traj, mu, log_var, beta=beta)
        for k in totals:
            totals[k] += losses[k].item()
        n += 1

    return {f"val/{k}": v / max(n, 1) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training NavIC VAE on: %s", device)

    if WANDB_AVAILABLE and args.wandb_project:
        wandb.init(project=args.wandb_project, config=vars(args),
                   name=args.run_name or "navic-vae")

    # Data
    parser_obj    = IOVNBDParser()
    raw_sequences = parser_obj.parse_dataset(Path(args.data_root))
    synced        = [s for seq in raw_sequences
                     if (s := synchronize_sequence(seq)) is not None]

    full_ds = NavICVAEDataset(synced, imu_hz=args.imu_hz,
                               window_s=args.window_s)
    n_val  = max(1, int(0.15 * len(full_ds)))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers,
                               drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers)

    logger.info("Train: %d | Val: %d trajectories", n_train, n_val)

    # Model
    model = NavICMotionPriorVAE(
        input_dim=4, hidden_dim=128, n_layers=2,
        latent_dim=args.latent_dim, output_len=int(args.window_s),
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("NavIC VAE parameters: %s", f"{total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                       eta_min=1e-6)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer,
                                         device, beta=args.beta)
        val_metrics   = evaluate_vae(model, val_loader, device, beta=args.beta)
        scheduler.step()

        log_line = (f"Epoch {epoch:03d} | "
                    f"train_total={train_metrics['train/total']:.4f} "
                    f"recon={train_metrics['train/recon']:.4f} "
                    f"kl={train_metrics['train/kl']:.4f} | "
                    f"val_total={val_metrics['val/total']:.4f}")
        logger.info(log_line)

        if WANDB_AVAILABLE and args.wandb_project:
            wandb.log({**train_metrics, **val_metrics, "epoch": epoch})

        if val_metrics["val/total"] < best_val_loss:
            best_val_loss = val_metrics["val/total"]
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "val_loss":   best_val_loss,
                "args":       vars(args),
            }, output_dir / "best_navic_vae.pt")
            logger.info("New best VAE saved: val_loss=%.4f", best_val_loss)

    logger.info("VAE training complete. Best val_loss: %.4f", best_val_loss)
    if WANDB_AVAILABLE and args.wandb_project:
        wandb.finish()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NavIC Motion Prior VAE")
    p.add_argument("--data_root",     required=True)
    p.add_argument("--output_dir",    default="./checkpoints/navic_vae")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--beta",          type=float, default=0.5)
    p.add_argument("--latent_dim",    type=int,   default=32)
    p.add_argument("--window_s",      type=float, default=60.0)
    p.add_argument("--imu_hz",        type=float, default=100.0)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--wandb_project", default="navdrift0")
    p.add_argument("--run_name",      default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
