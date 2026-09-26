"""
NAVDRIFT-0 | ISRO SIH 2026 PS #26168
File 02 — DRIFTFormer training
4-layer, 8-head causal transformer. 529K parameters.
Input : (batch, 100, 9)  — 1 second of 9-channel IMU features
Output: (batch, 2)       — position correction (delta_x, delta_y) in metres

Estimated A100 time: ~18 hours for 200 epochs on full dataset.
Resumes automatically from Drive checkpoint.
"""

import sys, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
from pathlib import Path

sys.path.insert(0, "/content")
from navdrift_00_setup import (PATHS, mount_drive, create_dirs,
                                get_device, start_keepalive,
                                save_checkpoint, load_checkpoint,
                                standard_train_loop)
from navdrift_01_data_pipeline import (extract_driftformer_windows,
                                        IMU_WINDOW)

# ============================================================
# HYPERPARAMETERS
# ============================================================
CFG = {
    "d_model":      256,
    "nhead":        8,
    "num_layers":   4,
    "ff_dim":       1024,
    "dropout":      0.1,
    "in_channels":  9,
    "seq_len":      IMU_WINDOW,  # 100
    "lr":           3e-4,
    "weight_decay": 1e-5,
    "batch_size":   512,
    "epochs":       200,
    "grad_clip":    1.0,
    "warmup_steps": 500,
    "compliance_target": 0.10,   # 10 % ISRO target
    "compliance_weight": 5.0,    # penalty weight when error > target
    "name":         "driftformer",
}

# ============================================================
# POSITIONAL ENCODING
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

# ============================================================
# DRIFTFORMER MODEL
# ============================================================
class DRIFTFormer(nn.Module):
    """
    Causal transformer for IMU dead-reckoning correction.
    ~529K parameters. INT8-quantisable.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        d   = cfg["d_model"]
        self.input_proj = nn.Sequential(
            nn.Linear(cfg["in_channels"], d),
            nn.LayerNorm(d),
        )
        self.pos_enc = PositionalEncoding(d, cfg["seq_len"] + 10, cfg["dropout"])

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg["nhead"],
            dim_feedforward=cfg["ff_dim"],
            dropout=cfg["dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=True,       # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg["num_layers"],
                                                  enable_nested_tensor=False)

        # causal mask — prevent future tokens
        seq = cfg["seq_len"]
        mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

        self.output_head = nn.Sequential(
            nn.Linear(d, 128),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(128, 2),    # delta_x, delta_y
        )

    def forward(self, x):
        # x: (B, T, 9)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x, mask=self.causal_mask,
                             is_causal=True)
        x = x[:, -1, :]          # last token — causal aggregation
        return self.output_head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================
# COMPLIANCE-AWARE LOSS
# ============================================================
class ComplianceLoss(nn.Module):
    """
    MSE + penalty when predicted correction keeps drift above ISRO target.
    drift_pct = |correction_error| / travel_distance * 100
    """
    def __init__(self, target: float = 0.10, penalty_weight: float = 5.0):
        super().__init__()
        self.target = target
        self.pw     = penalty_weight

    def forward(self, pred, target_delta):
        mse  = F.mse_loss(pred, target_delta)
        # proxy drift: residual error magnitude / unit distance
        resid  = torch.norm(pred - target_delta, dim=-1)
        travel = torch.norm(target_delta, dim=-1).clamp(min=1.0)
        drift  = resid / travel
        penalty = F.relu(drift - self.target).mean() * self.pw
        return mse + penalty

# ============================================================
# DATASET
# ============================================================
class DriftFormerDataset(Dataset):
    def __init__(self, hdf5_path: Path, split: str = "train"):
        self.Xs = []
        self.Ys = []
        print(f"[DATASET] Loading DRIFTFormer windows from {hdf5_path} ({split})...")
        with h5py.File(hdf5_path, "r") as hf:
            grp = hf[split]
            for seq_name in grp:
                sq   = grp[seq_name]
                imu  = sq["imu"][:]
                gtp  = sq["gt_pos"][:]
                dre  = sq["dr_error"][:]
                xs, ys = extract_driftformer_windows(imu, gtp, dre, stride=20)
                if xs is not None:
                    self.Xs.append(xs)
                    self.Ys.append(ys)

        if not self.Xs:
            if split == "train":
                raise RuntimeError(f"No windows extracted from train split. "
                                   f"Run navdrift_01_data_pipeline.py first.")
            # empty val/test split -- create a dummy so DataLoader won't crash
            self.Xs = [np.zeros((1, 100, 9), dtype=np.float32)]
            self.Ys = [np.zeros((1, 2), dtype=np.float32)]
            self._empty = True

        self.Xs = np.concatenate(self.Xs, axis=0)
        self.Ys = np.concatenate(self.Ys, axis=0)

        # normalize inputs
        self.x_mean = self.Xs.mean(axis=(0, 1))
        self.x_std  = self.Xs.std(axis=(0, 1)) + 1e-8
        self.Xs     = (self.Xs - self.x_mean) / self.x_std

        # normalize targets to metres (clip outliers)
        self.Ys = np.clip(self.Ys, -500, 500)
        self.y_scale = np.abs(self.Ys).max() + 1e-8
        self.Ys /= self.y_scale

        print(f"[DATASET]   {split}: {len(self.Xs)} windows")

    def __len__(self):  return len(self.Xs)
    def __getitem__(self, i):
        return (torch.tensor(self.Xs[i], dtype=torch.float32),
                torch.tensor(self.Ys[i], dtype=torch.float32))

    def save_stats(self, path: Path):
        np.savez(path, x_mean=self.x_mean, x_std=self.x_std, y_scale=[self.y_scale])
        print(f"[STATS] Saved normalisation stats -> {path}")

# ============================================================
# LEARNING RATE WARMUP + COSINE DECAY
# ============================================================
def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ============================================================
# TRAINING
# ============================================================
def train_driftformer(hdf5_path: Path, device: torch.device):
    print("\n" + "=" * 60)
    print("  TRAINING DRIFTFormer")
    print("=" * 60)

    train_ds = DriftFormerDataset(hdf5_path, "train")
    val_ds   = DriftFormerDataset(hdf5_path, "val")

    # save normalization stats for inference
    stats_path = PATHS["models"] / "driftformer_stats.npz"
    train_ds.save_stats(stats_path)

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True,  num_workers=4,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"] * 2,
                              shuffle=False, num_workers=4,
                              pin_memory=True)

    model     = DRIFTFormer(CFG).to(device)
    print(f"[MODEL] DRIFTFormer  |  {model.count_params():,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    total_steps = CFG["epochs"] * len(train_loader)
    scheduler   = build_scheduler(optimizer, CFG["warmup_steps"], total_steps)
    loss_fn     = ComplianceLoss(CFG["compliance_target"], CFG["compliance_weight"])

    start_epoch, loss_history = load_checkpoint(
        model, optimizer, scheduler, CFG["name"], device=str(device)
    )

    best_val = float("inf")
    for epoch in range(start_epoch, CFG["epochs"]):
        t0 = time.time()

        # --- train ---
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        # --- val ---
        va_loss = float("nan")
        if not getattr(val_ds, "_empty", False):
            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    va_loss += loss_fn(pred, y).item()
            va_loss /= len(val_loader)

        dt = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        print(f"[DRIFTFormer] Ep {epoch:03d}/{CFG['epochs']}  "
              f"tr={tr_loss:.5f}  val={va_loss:.5f}  "
              f"lr={lr:.2e}  {dt:.1f}s")

        loss_history.append({"epoch": epoch, "train": tr_loss, "val": va_loss})

        if va_loss < best_val:
            best_val = va_loss
            save_checkpoint(model, optimizer, scheduler, epoch,
                            loss_history, CFG["name"],
                            extra={"best_val": best_val})
        elif epoch % 10 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch,
                            loss_history, CFG["name"])

    print(f"\n[DRIFTFormer] Training done. Best val loss: {best_val:.5f}")
    return model

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    mount_drive()
    create_dirs()
    device   = get_device()
    start_keepalive()

    hdf5_path = PATHS["processed"] / "navdrift_dataset.h5"
    if not hdf5_path.exists():
        print("[ERROR] HDF5 dataset not found. Run navdrift_01_data_pipeline.py first.")
        sys.exit(1)

    model = train_driftformer(hdf5_path, device)
    print("\n[DRIFTFormer] Complete.")
