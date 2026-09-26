"""
NAVDRIFT-0 | ISRO SIH 2026 PS #26168
File 03 — TCN IMU Denoiser
Temporal Convolutional Network with dilated residual blocks.
Learns to map noisy MEMS IMU -> clean IMU signal.
Trained on EuRoC where FOG ground-truth provides the clean reference.

Input : (batch, 6, 200)  — raw 6-axis IMU at 200 Hz, 1-second window
Output: (batch, 6, 200)  — denoised IMU

Estimated A100 time: ~12 hours for 150 epochs.
"""

import sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
from pathlib import Path
from scipy.signal import butter, filtfilt

sys.path.insert(0, "/content")
from navdrift_00_setup import (PATHS, mount_drive, create_dirs,
                                get_device, start_keepalive,
                                save_checkpoint, load_checkpoint)

# ============================================================
# HYPERPARAMETERS
# ============================================================
CFG = {
    "in_channels":  6,
    "hidden_ch":    64,
    "kernel_size":  3,
    "dilations":    [1, 2, 4, 8, 16],
    "dropout":      0.1,
    "seq_len":      200,           # 200 Hz, 1 second
    "stride":       50,
    "lr":           1e-3,
    "batch_size":   256,
    "epochs":       150,
    "name":         "imu_denoiser",
}

# ============================================================
# TCN BLOCK
# ============================================================
class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self._pad  = pad

    def forward(self, x):
        # causal: remove right-side padding
        h = F.relu(self.bn1(self.conv1(x)[:, :, :x.shape[2]]))
        h = self.drop(h)
        h = F.relu(self.bn2(self.conv2(h)[:, :, :x.shape[2]]))
        return h + self.skip(x)

class IMUDenoiser(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        ch = cfg["hidden_ch"]
        layers = []
        in_ch = cfg["in_channels"]
        for d in cfg["dilations"]:
            layers.append(TCNBlock(in_ch, ch, cfg["kernel_size"], d, cfg["dropout"]))
            in_ch = ch
        self.tcn    = nn.Sequential(*layers)
        self.output = nn.Conv1d(ch, cfg["in_channels"], 1)

    def forward(self, x):
        # x: (B, 6, T)
        return self.output(self.tcn(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================
# NOISE SIMULATION (for datasets without dual IMU)
# ============================================================
def add_mems_noise(imu_clean: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Simulate MEMS-grade noise on top of a clean IMU signal.
    Uses realistic Allan variance parameters for consumer MEMS.
    """
    rng = np.random.default_rng(seed)
    T, C = imu_clean.shape
    noisy = imu_clean.copy()

    # angle random walk (white noise on gyro)
    arw = 0.003 * rng.standard_normal((T, 3))           # rad/sqrt(s)
    # bias instability (random walk on bias)
    bi  = np.cumsum(5e-5 * rng.standard_normal((T, 3)), axis=0)
    # velocity random walk (white noise on accel)
    vrw = 0.1 * rng.standard_normal((T, 3))              # m/s/sqrt(s)
    # accel bias instability
    abi = np.cumsum(2e-4 * rng.standard_normal((T, 3)), axis=0)

    noisy[:, :3] += vrw + abi   # accel channels
    noisy[:, 3:] += arw + bi    # gyro channels
    return noisy.astype(np.float32)

def butterworth_clean(imu: np.ndarray, fs: float = 200.0, cutoff: float = 30.0):
    """Low-pass Butterworth as 'clean' proxy when no FOG reference exists."""
    b, a = butter(4, cutoff / (fs / 2), btype="low")
    return filtfilt(b, a, imu, axis=0).astype(np.float32)

# ============================================================
# DATASET
# ============================================================
class DenoiserDataset(Dataset):
    def __init__(self, hdf5_path: Path, split: str = "train"):
        self.noisy_list = []
        self.clean_list = []
        seq_len  = CFG["seq_len"]
        stride   = CFG["stride"]

        print(f"[DENOISER DS] Loading from {hdf5_path} ({split})...")
        with h5py.File(hdf5_path, "r") as hf:
            grp = hf[split]
            for sname in grp:
                sq  = grp[sname]
                imu = sq["imu"][:]        # (N, 6)
                src = sq.attrs.get("source", "")

                # For EuRoC sequences, the data IS the clean reference
                # (recorded from a tactical-grade IMU alongside MEMS)
                # We add synthetic noise to produce the noisy input
                if "euroc" in src or "tumvi" in src:
                    clean = imu                        # use as-is for clean
                    noisy = add_mems_noise(imu)
                else:
                    # For KAIST/NCLT (already MEMS), use Butterworth as proxy clean
                    clean = butterworth_clean(imu)
                    noisy = imu

                # sliding windows
                for start in range(0, len(imu) - seq_len, stride):
                    end = start + seq_len
                    self.noisy_list.append(noisy[start:end].T)  # (6, T)
                    self.clean_list.append(clean[start:end].T)

        if not self.noisy_list:
            if split == "train":
                raise RuntimeError("No windows for denoiser. Check HDF5 path.")
            # empty val/test -- use a dummy so DataLoader won't crash
            dummy = np.zeros((1, 6, 200), dtype=np.float32)
            self.noisy_list = [dummy[0]]
            self.clean_list = [dummy[0]]
            self._empty = True

        self.noisy = np.array(self.noisy_list, dtype=np.float32)
        self.clean = np.array(self.clean_list, dtype=np.float32)

        # per-channel normalisation
        self.mean = self.noisy.mean(axis=(0, 2), keepdims=True)
        self.std  = self.noisy.std(axis=(0, 2), keepdims=True) + 1e-8
        self.noisy = (self.noisy - self.mean) / self.std
        self.clean = (self.clean - self.mean) / self.std

        print(f"[DENOISER DS]   {split}: {len(self.noisy)} windows")

    def __len__(self):  return len(self.noisy)
    def __getitem__(self, i):
        return (torch.tensor(self.noisy[i], dtype=torch.float32),
                torch.tensor(self.clean[i], dtype=torch.float32))

# ============================================================
# SPECTRAL LOSS — preserve frequency content
# ============================================================
class SpectralMSELoss(nn.Module):
    """MSE in time domain + FFT magnitude MSE in frequency domain."""
    def __init__(self, freq_weight: float = 0.3):
        super().__init__()
        self.fw = freq_weight

    def forward(self, pred, target):
        time_loss = F.mse_loss(pred, target)
        pf = torch.fft.rfft(pred,   dim=-1).abs()
        tf = torch.fft.rfft(target, dim=-1).abs()
        freq_loss = F.mse_loss(pf, tf)
        return time_loss + self.fw * freq_loss

# ============================================================
# TRAINING
# ============================================================
def train_denoiser(hdf5_path: Path, device: torch.device):
    print("\n" + "=" * 60)
    print("  TRAINING IMU Denoiser (TCN)")
    print("=" * 60)

    train_ds = DenoiserDataset(hdf5_path, "train")
    val_ds   = DenoiserDataset(hdf5_path, "val")

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True,  num_workers=4, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"] * 2,
                              shuffle=False, num_workers=4, pin_memory=True)

    model     = IMUDenoiser(CFG).to(device)
    print(f"[MODEL] IMUDenoiser  |  {model.count_params():,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-6
    )
    loss_fn = SpectralMSELoss(freq_weight=0.3)

    start_epoch, loss_history = load_checkpoint(
        model, optimizer, None, CFG["name"], device=str(device)
    )

    best_val = float("inf")
    for epoch in range(start_epoch, CFG["epochs"]):
        t0 = time.time()

        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        va_loss = float("nan")
        if not getattr(val_ds, "_empty", False):
            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    va_loss += loss_fn(model(x), y).item()
            va_loss /= len(val_loader)

        scheduler.step(va_loss if not (va_loss != va_loss) else tr_loss)
        dt = time.time() - t0
        print(f"[Denoiser] Ep {epoch:03d}/{CFG['epochs']}  "
              f"tr={tr_loss:.5f}  val={va_loss:.5f}  {dt:.1f}s")

        loss_history.append({"epoch": epoch, "train": tr_loss, "val": va_loss})

        if (va_loss == va_loss) and va_loss < best_val:  # skip nan
            best_val = va_loss
            save_checkpoint(model, optimizer, None, epoch,
                            loss_history, CFG["name"])
        elif epoch % 15 == 0:
            save_checkpoint(model, optimizer, None, epoch,
                            loss_history, CFG["name"])

    print(f"\n[Denoiser] Done. Best val: {best_val:.5f}")
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

    model = train_denoiser(hdf5_path, device)
    print("\n[IMU Denoiser] Complete.")
