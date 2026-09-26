"""
NAVDRIFT-0 | ISRO SIH 2026 PS #26168
File 04 — Adaptive EKF Noise Predictor
3-layer MLP that predicts optimal Q (process noise) and R (measurement noise)
matrices from observable navigation state features.

Replaces hardcoded EKF noise constants with dynamic, context-aware predictions.
When GNSS is fresh and speed is steady -> tight R.
When turning hard or in post-outage recovery -> wider Q.

Input : (batch, 6)  — [speed, turn_rate, gnss_age_s, sat_count_norm, zupt, imu_temp_norm]
Output: (batch, 5)  — [Q00, Q11, Q22, R00, R11]  (diagonal noise variances)

Estimated A100 time: ~6 hours for 300 epochs.
"""

import sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import h5py
from pathlib import Path
from scipy.signal import savgol_filter

sys.path.insert(0, "/content")
from navdrift_00_setup import (PATHS, mount_drive, create_dirs,
                                get_device, start_keepalive,
                                save_checkpoint, load_checkpoint,
                                save_results)

# ============================================================
# CFG
# ============================================================
CFG = {
    "in_features":  6,
    "hidden":       [128, 128, 64],
    "out_features": 5,     # Q00, Q11, Q22, R00, R11
    "dropout":      0.15,
    "lr":           5e-4,
    "batch_size":   1024,
    "epochs":       300,
    "name":         "adaptive_ekf",
    # physical bounds for noise values
    "q_min": 1e-8,
    "q_max": 1e-2,
    "r_min": 1e-6,
    "r_max": 1e-1,
}

# ============================================================
# MODEL
# ============================================================
class AdaptiveEKFNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        layers = []
        in_ch  = cfg["in_features"]
        for h in cfg["hidden"]:
            layers += [nn.Linear(in_ch, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(cfg["dropout"])]
            in_ch = h
        layers.append(nn.Linear(in_ch, cfg["out_features"]))
        self.net = nn.Sequential(*layers)

        # learnable log-scale outputs (ensure positive values)
        # raw net output -> Softplus -> scale
        self.softplus = nn.Softplus()

    def forward(self, x):
        raw = self.net(x)              # (B, 5)  unbounded
        out = self.softplus(raw)       # (B, 5)  positive
        # scale Q (0:3) and R (3:5) separately to physical ranges
        Q   = out[:, :3] * 1e-4 + CFG["q_min"]
        R   = out[:, 3:] * 1e-3 + CFG["r_min"]
        return torch.cat([Q, R], dim=-1)  # (B, 5)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================
# OPTIMAL Q/R COMPUTATION FROM GROUND TRUTH
# ============================================================
def compute_optimal_qr(imu: np.ndarray, gt_pos: np.ndarray,
                        gps_quality: np.ndarray, dt: float = 0.01):
    """
    Compute 'oracle' Q and R at each timestep by post-hoc smoothing:
    - R[t] = variance of GPS measurement error in a local window
    - Q[t] = variance of IMU integration error growth rate
    Returns arrays of shape (N, 5).
    """
    N  = len(imu)
    QR = np.zeros((N, 5), dtype=np.float32)

    # --- R from GPS noise ---
    # For position channels: R = local variance of GT position fluctuations
    if np.abs(gt_pos[:, 0]).max() < 100:
        pos_m = gt_pos[:, :2]
    else:
        lat0  = gt_pos[0, 0]
        pos_m = np.stack([
            (gt_pos[:, 1] - gt_pos[0, 1]) * 111320 * np.cos(np.radians(lat0)),
            (gt_pos[:, 0] - gt_pos[0, 0]) * 111320,
        ], axis=-1)

    window = 50
    for i in range(N):
        lo = max(0, i - window // 2)
        hi = min(N, i + window // 2)
        local_pos = pos_m[lo:hi]
        r_val = np.var(local_pos, axis=0).mean()
        QR[i, 3] = float(np.clip(r_val, CFG["r_min"], CFG["r_max"]))  # R00
        QR[i, 4] = float(np.clip(r_val * 1.5, CFG["r_min"], CFG["r_max"]))  # R11

    # --- Q from IMU integration noise ---
    # velocity estimate via trapezoidal integration
    acc   = imu[:, :3]          # (N, 3)
    v_int = np.cumsum(acc * dt, axis=0)

    # Q = local variance of acceleration noise (proxy for process noise)
    for i in range(N):
        lo   = max(0, i - window // 2)
        hi   = min(N, i + window // 2)
        q00  = np.var(acc[lo:hi, 0])
        q11  = np.var(acc[lo:hi, 1])
        q22  = np.var(v_int[lo:hi, 2]) * 0.1  # vel-z uncertainty
        QR[i, 0] = float(np.clip(q00, CFG["q_min"], CFG["q_max"]))
        QR[i, 1] = float(np.clip(q11, CFG["q_min"], CFG["q_max"]))
        QR[i, 2] = float(np.clip(q22, CFG["q_min"], CFG["q_max"]))

    return QR

def build_state_features(imu: np.ndarray, gps_quality: np.ndarray,
                          dt: float = 0.01):
    """
    Build 6-feature input vector at each timestep.
    Features: [speed, turn_rate, gnss_age, sat_count_norm, zupt, accel_var]
    """
    N = len(imu)
    F6 = np.zeros((N, 6), dtype=np.float32)

    # speed proxy: magnitude of acceleration (after bias subtraction)
    acc_mag = np.linalg.norm(imu[:, :3], axis=1)
    grav    = 9.81
    speed   = np.clip(acc_mag - grav, 0, 40) / 40.0  # normalize 0-40 m/s

    # turn rate: gyro-z magnitude
    turn_rate = np.abs(imu[:, 5]) / 3.0  # normalize to ~1 at 3 rad/s

    # GNSS age: steps since last good GPS (normalized by 1000 steps)
    gnss_age = np.zeros(N)
    age = 0
    for i in range(N):
        age = 0 if gps_quality[i] > 0.5 else age + 1
        gnss_age[i] = min(age / 1000.0, 1.0)

    # satellite count proxy: from gps_quality smoothed (0-1)
    sat_norm = savgol_filter(gps_quality, 51, 3) if N > 51 else gps_quality

    # ZUPT: standstill detection
    zupt = (acc_mag < 0.2).astype(np.float32)

    # acceleration variance (short window)
    acc_var = np.zeros(N)
    w = 20
    for i in range(N):
        lo = max(0, i - w)
        acc_var[i] = np.var(acc_mag[lo:i+1])
    acc_var = np.clip(acc_var / 5.0, 0, 1)

    F6[:, 0] = speed
    F6[:, 1] = turn_rate
    F6[:, 2] = gnss_age
    F6[:, 3] = sat_norm
    F6[:, 4] = zupt
    F6[:, 5] = acc_var.astype(np.float32)
    return F6

# ============================================================
# DATASET
# ============================================================
class AdaptiveEKFDataset(Dataset):
    def __init__(self, hdf5_path: Path, split: str = "train"):
        Xs, Ys = [], []
        print(f"[AEKF DS] Loading from {hdf5_path} ({split})...")
        with h5py.File(hdf5_path, "r") as hf:
            grp = hf[split]
            for sname in grp:
                sq   = grp[sname]
                imu  = sq["imu"][:]
                gtp  = sq["gt_pos"][:]
                gpq  = sq["gps_quality"][:]

                feats = build_state_features(imu, gpq)
                qr    = compute_optimal_qr(imu, gtp, gpq)

                # subsample every 10 steps (EKF runs at 10 Hz in the UI)
                Xs.append(feats[::10])
                Ys.append(qr[::10])

        if not Xs:
            if split == "train":
                raise RuntimeError("No data for Adaptive EKF. Check HDF5.")
            # empty val/test -- dummy row
            self.X = np.zeros((1, 9), dtype=np.float32)
            self.Y = np.zeros((1, 2), dtype=np.float32)
            self.x_mean = np.zeros(9, dtype=np.float32)
            self.x_std  = np.ones(9, dtype=np.float32)
            self._empty = True
            print(f"[AEKF DS]   {split}: 0 steps (empty, using dummy)")
            return

        self.X = np.concatenate(Xs, axis=0).astype(np.float32)
        self.Y = np.concatenate(Ys, axis=0).astype(np.float32)

        # normalise X (Y stays in physical units, Softplus handles scale)
        self.x_mean = self.X.mean(0)
        self.x_std  = self.X.std(0) + 1e-8
        self.X = (self.X - self.x_mean) / self.x_std

        # log-normalise Y for regression stability
        self.Y = np.log10(self.Y + 1e-10)

        print(f"[AEKF DS]   {split}: {len(self.X)} steps")

    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return (torch.tensor(self.X[i]), torch.tensor(self.Y[i]))

# ============================================================
# TRAINING
# ============================================================
def train_adaptive_ekf(hdf5_path: Path, device: torch.device):
    print("\n" + "=" * 60)
    print("  TRAINING Adaptive EKF Predictor")
    print("=" * 60)

    train_ds = AdaptiveEKFDataset(hdf5_path, "train")
    val_ds   = AdaptiveEKFDataset(hdf5_path, "val")

    # save normalisation stats
    stats = {"x_mean": train_ds.x_mean.tolist(), "x_std": train_ds.x_std.tolist()}
    save_results(stats, "adaptive_ekf_stats")

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"] * 4,
                              shuffle=False, num_workers=4, pin_memory=True)

    model     = AdaptiveEKFNet(CFG).to(device)
    print(f"[MODEL] AdaptiveEKFNet  |  {model.count_params():,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CFG["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"], eta_min=1e-6
    )

    start_epoch, loss_history = load_checkpoint(
        model, optimizer, scheduler, CFG["name"], device=str(device)
    )

    best_val = float("inf")
    for epoch in range(start_epoch, CFG["epochs"]):
        t0 = time.time()

        model.train()
        tr = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            # predict in log space, compare to log-normalised Y
            loss = F.huber_loss(torch.log10(pred + 1e-10), y, delta=0.5)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr += loss.item()
        tr /= len(train_loader)

        va = float("nan")
        if not getattr(val_ds, "_empty", False):
            model.eval()
            va = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    va  += F.huber_loss(torch.log10(pred + 1e-10), y, delta=0.5).item()
            va /= len(val_loader)

        scheduler.step()
        dt = time.time() - t0
        print(f"[AdaptEKF] Ep {epoch:03d}/{CFG['epochs']}  "
              f"tr={tr:.5f}  val={va:.5f}  {dt:.1f}s")

        loss_history.append({"epoch": epoch, "train": tr, "val": va})

        if (va == va) and va < best_val:  # skip nan
            best_val = va
            save_checkpoint(model, optimizer, scheduler, epoch,
                            loss_history, CFG["name"])
        elif epoch % 20 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch,
                            loss_history, CFG["name"])

    print(f"\n[AdaptEKF] Done. Best val: {best_val:.5f}")
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

    model = train_adaptive_ekf(hdf5_path, device)
    print("\n[Adaptive EKF] Complete.")
