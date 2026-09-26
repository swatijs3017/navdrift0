"""
NAVDRIFT-0 | ISRO SIH 2026 PS #26168
File 05 — Tunnel / GNSS Blackout Detector
Bidirectional LSTM that fires 3-5 seconds BEFORE a GNSS outage starts,
giving the EKF time to pre-arm and warm-start covariances.

Input : (batch, 20, 5)  — 2 seconds of telemetry at 10 Hz
                           [imu_var, imu_mag, gyro_mag, gps_quality, speed_consistency]
Output: (batch, 1)      — blackout probability in next 3-5 seconds (0-1)

Estimated A100 time: ~4 hours for 200 epochs.
"""

import sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve
import h5py
from pathlib import Path

sys.path.insert(0, "/content")
from navdrift_00_setup import (PATHS, mount_drive, create_dirs,
                                get_device, start_keepalive,
                                save_checkpoint, load_checkpoint,
                                save_results)
from navdrift_01_data_pipeline import extract_tunnel_windows, TUNNEL_WIN

# ============================================================
# CFG
# ============================================================
CFG = {
    "input_size":   5,
    "hidden_size":  64,
    "num_layers":   2,
    "dropout":      0.2,
    "lr":           1e-3,
    "batch_size":   512,
    "epochs":       200,
    "pos_weight":   8.0,   # class imbalance — tunnel entries are rare
    "name":         "tunnel_detector",
    "threshold":    0.55,  # fire alarm above this probability
}

# ============================================================
# MODEL
# ============================================================
class TunnelDetector(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        h = cfg["hidden_size"]
        self.norm  = nn.LayerNorm(cfg["input_size"])
        self.lstm  = nn.LSTM(
            input_size=cfg["input_size"],
            hidden_size=h,
            num_layers=cfg["num_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0,
        )
        self.attention = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.Tanh(),
            nn.Linear(h, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(cfg["dropout"]),
            nn.Linear(h * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (B, T, 5)
        x    = self.norm(x)
        out, _ = self.lstm(x)      # (B, T, H*2)
        # attention pooling
        attn   = torch.softmax(self.attention(out), dim=1)  # (B, T, 1)
        ctx    = (out * attn).sum(dim=1)                    # (B, H*2)
        return torch.sigmoid(self.classifier(ctx)).squeeze(-1)  # (B,)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================
# SIMULATE GNSS OUTAGES for training data enrichment
# ============================================================
def simulate_outages(gps_quality: np.ndarray, seed: int = 0):
    """
    For datasets that have continuous GPS coverage (EuRoC),
    synthetically inject GNSS outages of 5-60 seconds to train the detector.
    """
    rng = np.random.default_rng(seed)
    result = gps_quality.copy()
    N = len(result)
    n_outages = max(1, N // 3000)  # ~1 outage per 30 seconds
    for _ in range(n_outages):
        start  = rng.integers(100, N - 600)
        length = rng.integers(50, 600)  # 5 to 60 seconds at 10 Hz
        result[start:start + length] = 0.0
    return result

# ============================================================
# DATASET
# ============================================================
class TunnelDataset(Dataset):
    def __init__(self, hdf5_path: Path, split: str = "train", augment: bool = True):
        Xs, Ys = [], []
        print(f"[TUNNEL DS] Loading from {hdf5_path} ({split})...")

        with h5py.File(hdf5_path, "r") as hf:
            grp = hf[split]
            for seed_i, sname in enumerate(grp):
                sq  = grp[sname]
                imu = sq["imu"][:]
                gpq = sq["gps_quality"][:]

                # downsample to 10 Hz (original is 100 Hz)
                imu_10 = imu[::10]
                gpq_10 = gpq[::10]

                # for EuRoC/TUM-VI, inject synthetic outages
                src = sq.attrs.get("source", "")
                if augment and ("euroc" in src or "tumvi" in src):
                    gpq_10 = simulate_outages(gpq_10, seed=seed_i)

                xs, ys = extract_tunnel_windows(imu_10, gpq_10, stride=5)
                if xs is not None:
                    Xs.append(xs)
                    Ys.append(ys)

        if not Xs:
            if split == "train":
                raise RuntimeError("No windows for TunnelDetector. Check HDF5.")
            # empty val/test -- dummy
            self.X = np.zeros((1, 20, 5), dtype=np.float32)
            self.Y = np.zeros((1,), dtype=np.float32)
            self.x_mean = np.zeros(5, dtype=np.float32)
            self.x_std  = np.ones(5, dtype=np.float32)
            self._empty = True
            print(f"[TUNNEL DS]   {split}: 0 windows (empty, using dummy)")
            return

        self.X = np.concatenate(Xs, axis=0).astype(np.float32)
        self.Y = np.concatenate(Ys, axis=0).astype(np.float32)

        # per-feature standardisation
        self.x_mean = self.X.mean(axis=(0, 1))
        self.x_std  = self.X.std(axis=(0, 1)) + 1e-8
        self.X = (self.X - self.x_mean) / self.x_std

        pos = self.Y.sum()
        neg = len(self.Y) - pos
        print(f"[TUNNEL DS]   {split}: {len(self.X)} windows  "
              f"pos={int(pos)}  neg={int(neg)}  ratio={pos/(neg+1):.3f}")

    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return (torch.tensor(self.X[i]), torch.tensor(self.Y[i]))

# ============================================================
# TRAINING
# ============================================================
def train_tunnel_detector(hdf5_path: Path, device: torch.device):
    print("\n" + "=" * 60)
    print("  TRAINING Tunnel / Blackout Detector")
    print("=" * 60)

    train_ds = TunnelDataset(hdf5_path, "train", augment=True)
    val_ds   = TunnelDataset(hdf5_path, "val",   augment=False)

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"] * 2,
                              shuffle=False, num_workers=4, pin_memory=True)

    model     = TunnelDetector(CFG).to(device)
    print(f"[MODEL] TunnelDetector  |  {model.count_params():,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CFG["lr"] * 3,
        epochs=CFG["epochs"], steps_per_epoch=len(train_loader)
    )
    pos_weight = torch.tensor([CFG["pos_weight"]], device=device)

    start_epoch, loss_history = load_checkpoint(
        model, optimizer, None, CFG["name"], device=str(device)
    )

    best_auc = 0.0
    for epoch in range(start_epoch, CFG["epochs"]):
        t0 = time.time()

        model.train()
        tr = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = F.binary_cross_entropy(pred, y, reduction="none")
            # upweight positive (tunnel entry) samples
            w = torch.where(y > 0.5, pos_weight, torch.ones_like(y))
            loss = (loss * w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tr += loss.item()
        tr /= len(train_loader)

        # AUC on validation
        auc = float("nan")
        if not getattr(val_ds, "_empty", False):
            model.eval()
            all_pred, all_true = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    p = model(x).cpu().numpy()
                    all_pred.extend(p.tolist())
                    all_true.extend(y.numpy().tolist())
            try:
                auc = roc_auc_score(all_true, all_pred)
            except ValueError:
                auc = 0.5

        dt = time.time() - t0
        print(f"[TunnelDet] Ep {epoch:03d}/{CFG['epochs']}  "
              f"tr={tr:.4f}  AUC={auc:.4f}  {dt:.1f}s")

        loss_history.append({"epoch": epoch, "train": tr, "val_auc": auc})

        if (auc == auc) and auc > best_auc:  # skip nan
            best_auc = auc
            save_checkpoint(model, optimizer, None, epoch,
                            loss_history, CFG["name"],
                            extra={"best_auc": best_auc})
            # save threshold analysis
            prec, rec, thresholds = precision_recall_curve(all_true, all_pred)
            save_results({
                "best_auc": best_auc,
                "threshold": CFG["threshold"],
                "precision": prec.tolist(),
                "recall":    rec.tolist(),
            }, "tunnel_detector_pr_curve")
        elif epoch % 20 == 0:
            save_checkpoint(model, optimizer, None, epoch, loss_history, CFG["name"])

    print(f"\n[TunnelDet] Done. Best AUC: {best_auc:.4f}")
    return model

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    mount_drive()
    create_dirs()
    device    = get_device()
    start_keepalive()

    hdf5_path = PATHS["processed"] / "navdrift_dataset.h5"
    if not hdf5_path.exists():
        print("[ERROR] HDF5 not found. Run navdrift_01_data_pipeline.py first.")
        sys.exit(1)

    model = train_tunnel_detector(hdf5_path, device)
    print("\n[Tunnel Detector] Complete.")
