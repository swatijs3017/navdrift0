"""
training/train_iovnbd_speed_lstm.py — trains a real speed-estimation LSTM
on the real IO-VNBD dataset (Synchronised archive), for use on Colab with
a GPU. Designed to run unmodified on CPU too (as a correctness smoke
test, not a real training run — CPU training on this scale of data is
not something this script pretends is fast).

Architecture and I/O contract
------------------------------
Matches navdrift_engine.py's `LSTMSpeedEstimator` inference wrapper
exactly, so the exported ONNX model is a drop-in artifact for that class
(NOT wired into it automatically — that remains a separate, explicit,
later decision):

    input:  (1, seq_len, 6) float32 — normalized
            [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
            per navdrift_engine.py's own `raw = [ax,ay,az,gx,gy,gz]`
            feature order.
    output: (1, 1) float32 — predicted speed in m/s (clamped >= 0 by the
            CALLER, exactly as LSTMSpeedEstimator.push() already does;
            the model itself is not clamped internally).

`seq_len` is a hyperparameter, NOT extracted from any existing checkpoint
or meta.json (none exists in this repository before this training run).
Default is 50 samples = 5.0 s of real IO-VNBD data at its measured 10 Hz
smartphone sampling rate. This is disclosed as a design choice.

Target
------
Ground-truth speed comes from the real V-file's `Velocity (km/hr) / 3.6`
(the vehicle's own reference GPS/INS speed), exposed as
`OdometrySample.speed` by data/iovnbd.py. This is a REAL recorded signal,
not a synthetic or fabricated label.

Data pipeline
-------------
Uses data/iovnbd.load_iovnbd_dataset() to parse the real ZIP directly (no
extraction required) and data/iovnbd_split.make_split()'s manifest
(results/iovnbd/split_manifest.json) for the sequence-level train/val/test
assignment — the same manifest already generated and checked into
results/iovnbd/, so train/val/test here matches what
eval/iovnbd_benchmark.py will later evaluate on.

Usage (Colab, GPU):
    python -m training.train_iovnbd_speed_lstm \
        --zip "raw/Synchronised V abd S datasets.zip" \
        --split_manifest results/iovnbd/split_manifest.json \
        --epochs 40 --seq_len 50 --batch_size 256 \
        --out_dir checkpoints/iovnbd_speed_lstm

Usage (CPU smoke test only, few sequences, few epochs):
    python -m training.train_iovnbd_speed_lstm \
        --zip "raw/Synchronised V abd S datasets.zip" \
        --split_manifest results/iovnbd/split_manifest.json \
        --epochs 1 --max_train_runs 2 --max_val_runs 1 --max_test_runs 1 \
        --out_dir /tmp/iovnbd_smoketest --smoke_test
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.iovnbd import load_iovnbd_dataset
from data.loader import Sequence

logger = logging.getLogger(__name__)

FEATURE_COLS = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]


# ---------------------------------------------------------------------------
# Model — matches the LSTMSpeedEstimator ONNX I/O contract in navdrift_engine.py
# ---------------------------------------------------------------------------

class SpeedLSTM(nn.Module):
    """Predicts speed in NORMALIZED target space (z-scored using the
    train-split target mean/std — see compute_target_norm_stats()). This
    module alone is NOT the final ONNX export target; SpeedLSTMWithDenorm
    below wraps it with a fixed affine layer so the exported ONNX graph's
    output stays in real m/s, matching navdrift_engine.py's
    LSTMSpeedEstimator contract (which does not itself denormalize)."""

    def __init__(self, n_features: int = 6, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, n_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]           # (B, hidden)
        return self.head(last)         # (B, 1), NORMALIZED target space


class SpeedLSTMWithDenorm(nn.Module):
    """Wraps a trained SpeedLSTM with a fixed (non-trainable) affine
    de-normalization so forward() returns real m/s directly. Used ONLY
    for ONNX export — training happens on the bare SpeedLSTM in
    normalized space, since regressing directly onto raw m/s targets is
    what caused the original training run to collapse to predicting a
    constant (see training_report.json / this module's own history for
    that failure mode and why target normalization was added)."""

    def __init__(self, model: SpeedLSTM, target_mean: float, target_std: float):
        super().__init__()
        self.model = model
        self.register_buffer("target_mean", torch.tensor(float(target_mean)))
        self.register_buffer("target_std", torch.tensor(float(target_std)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_pred = self.model(x)
        return norm_pred * self.target_std + self.target_mean


# ---------------------------------------------------------------------------
# Windowed dataset over real sequences
# ---------------------------------------------------------------------------

class WindowedSpeedDataset(Dataset):
    """One item = one seq_len window of real IMU samples, target = real
    reference speed at the window's last timestep."""

    def __init__(self, sequences: List[Sequence], seq_len: int,
                 stride: int, mean: np.ndarray, std: np.ndarray):
        self.seq_len = seq_len
        self.items = []  # (imu_window_normalized (seq_len,6) float32, target float32)
        for seq in sequences:
            n = len(seq.imu)
            if n <= seq_len:
                continue
            imu = np.array([[s.accel_x, s.accel_y, s.accel_z,
                              s.gyro_x, s.gyro_y, s.gyro_z] for s in seq.imu],
                            dtype=np.float32)
            speed = np.array([s.speed for s in seq.odom], dtype=np.float32)
            imu_norm = (imu - mean) / std
            for start in range(0, n - seq_len, stride):
                end = start + seq_len
                target = speed[end - 1]
                if not np.isfinite(target):
                    continue
                self.items.append((imu_norm[start:end].copy(), target))
        logger.info("WindowedSpeedDataset: %d windows from %d sequences",
                    len(self.items), len(sequences))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y = self.items[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def compute_norm_stats(sequences: List[Sequence]) -> tuple:
    all_imu = np.concatenate([
        np.array([[s.accel_x, s.accel_y, s.accel_z, s.gyro_x, s.gyro_y, s.gyro_z]
                   for s in seq.imu], dtype=np.float32)
        for seq in sequences if len(seq.imu) > 0
    ], axis=0)
    mean = all_imu.mean(axis=0)
    std = all_imu.std(axis=0)
    std[std < 1e-6] = 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def compute_target_norm_stats(sequences: List[Sequence]) -> tuple:
    """Mean/std of the real reference speed target, from the TRAIN split
    only (no leakage). Regressing directly onto raw m/s speed (mean ~12.7,
    std ~8.6 in this dataset) with MSE loss caused the LSTM to collapse to
    predicting the constant mean and stop learning entirely — confirmed by
    a real training run where train_mse plateaued at exactly std^2. This
    function exists specifically to fix that, by normalizing the
    regression target the same way the inputs already were."""
    all_speed = np.concatenate([
        np.array([o.speed for o in seq.odom], dtype=np.float32)
        for seq in sequences if len(seq.odom) > 0
    ])
    mean = float(all_speed.mean())
    std = float(all_speed.std())
    if std < 1e-6:
        std = 1e-6
    return mean, std


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, train: bool,
              target_mean: float, target_std: float, grad_clip_norm: float = 5.0) -> float:
    model.train(train)
    total_loss, n = 0.0, 0
    loss_fn = nn.MSELoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device).unsqueeze(-1)
        y_norm = (y - target_mean) / target_std   # train in normalized target space
        if train:
            optimizer.zero_grad()
        pred_norm = model(x)
        loss = loss_fn(pred_norm, y_norm)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, target_mean: float, target_std: float) -> Dict[str, float]:
    """Denormalizes predictions back to real m/s before computing metrics,
    so MAE/RMSE/R^2 are reported in physical units, not normalized units."""
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device)
        pred_norm = model(x).cpu().numpy().reshape(-1)
        pred_mps = pred_norm * target_std + target_mean
        preds.append(pred_mps)
        targets.append(y.numpy().reshape(-1))
    preds = np.concatenate(preds) if preds else np.array([])
    targets = np.concatenate(targets) if targets else np.array([])
    if len(preds) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n_samples": 0}
    err = preds - targets
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-9 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "n_samples": int(len(preds))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", required=True)
    p.add_argument("--split_manifest", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_runs", type=int, default=None)
    p.add_argument("--max_val_runs", type=int, default=None)
    p.add_argument("--max_test_runs", type=int, default=None)
    p.add_argument("--smoke_test", action="store_true",
                    help="Mark this run's report as a code-correctness "
                         "smoke test, not a real training result.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.smoke_test:
        print("=" * 78)
        print("  WARNING: no CUDA device detected. Real IO-VNBD-scale LSTM training")
        print("  on CPU would take an impractical amount of time. This script will")
        print("  NOT silently run a multi-hour CPU training job. Re-run with")
        print("  --smoke_test and small --max_*_runs / --epochs for a code check,")
        print("  or run this on a GPU runtime (e.g. Google Colab with an A100).")
        print("=" * 78)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.split_manifest) as f:
        manifest = json.load(f)

    logger.info("Loading real IO-VNBD sequences from %s ...", args.zip)
    all_sequences, metas = load_iovnbd_dataset(args.zip)
    by_name = {seq.name: seq for seq in all_sequences}

    def pick(names, limit):
        names = names if limit is None else names[:limit]
        return [by_name[n] for n in names if n in by_name]

    train_seqs = pick(manifest["train"], args.max_train_runs)
    val_seqs = pick(manifest["val"], args.max_val_runs)
    test_seqs = pick(manifest["test"], args.max_test_runs)

    logger.info("train=%d val=%d test=%d sequences", len(train_seqs), len(val_seqs), len(test_seqs))
    if not train_seqs or not val_seqs:
        raise RuntimeError("Empty train or val split after run filtering — cannot proceed honestly.")

    mean, std = compute_norm_stats(train_seqs)   # input stats from TRAIN split only — no leakage
    target_mean, target_std = compute_target_norm_stats(train_seqs)  # target stats, TRAIN split only
    logger.info("target speed stats (train split): mean=%.3f std=%.3f m/s", target_mean, target_std)

    train_ds = WindowedSpeedDataset(train_seqs, args.seq_len, args.stride, mean, std)
    val_ds = WindowedSpeedDataset(val_seqs, args.seq_len, args.stride, mean, std)
    test_ds = WindowedSpeedDataset(test_seqs, args.seq_len, args.stride, mean, std) if test_seqs else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False) if test_ds else None

    model = SpeedLSTM(n_features=6, hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_epoch = -1
    epochs_since_improve = 0
    ckpt_path = out_dir / "navdrift_lstm_best.pt"

    t_start = time.time()
    history = []
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True,
                                target_mean=target_mean, target_std=target_std)
        val_metrics = evaluate(model, val_loader, device, target_mean=target_mean, target_std=target_std)
        val_loss = val_metrics["mae"]
        history.append({"epoch": epoch, "train_mse_loss_normalized": train_loss,
                         "val_mae_mps": val_metrics["mae"], "val_rmse_mps": val_metrics["rmse"]})
        logger.info("epoch %d: train_mse(norm)=%.5f val_mae=%.5f m/s  val_rmse=%.5f m/s",
                     epoch, train_loss, val_metrics["mae"], val_metrics["rmse"])

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch, "val_mae_mps": val_loss,
                        "seq_len": args.seq_len, "n_features": 6,
                        "hidden_size": args.hidden_size, "num_layers": args.num_layers,
                        "feature_cols": FEATURE_COLS,
                        "scaler_mean": mean.tolist(), "scaler_scale": std.tolist(),
                        "target_mean": target_mean, "target_std": target_std},
                       ckpt_path)
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                logger.info("Early stopping at epoch %d (best epoch %d, val_mae=%.5f m/s)",
                            epoch, best_epoch, best_val)
                break

    train_time_s = time.time() - t_start

    # Reload best checkpoint for final test evaluation
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, device, target_mean=target_mean, target_std=target_std) \
        if test_loader else {"mae": None, "rmse": None, "r2": None, "n_samples": 0}

    report = {
        "kind": "SMOKE_TEST_CODE_CORRECTNESS_ONLY" if args.smoke_test else "REAL_TRAINING_RUN",
        "device": str(device),
        "seed": args.seed,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "n_params": int(n_params),
        "n_train_sequences": len(train_seqs),
        "n_val_sequences": len(val_seqs),
        "n_test_sequences": len(test_seqs),
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "n_test_windows": len(test_ds) if test_ds else 0,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "training_time_s": train_time_s,
        "target_mean_mps": target_mean,
        "target_std_mps": target_std,
        "val_mae_best_mps": best_val,
        "test_mae_mps": test_metrics["mae"],
        "test_rmse_mps": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_n_samples": test_metrics["n_samples"],
        "history": history,
        "checkpoint_path": str(ckpt_path),
    }
    with open(out_dir / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(out_dir / "navdrift_lstm_meta.json", "w") as f:
        json.dump({"seq_len": args.seq_len, "feature_cols": FEATURE_COLS,
                    "scaler_mean": mean.tolist(), "scaler_scale": std.tolist(),
                    "target_mean_mps": target_mean, "target_std_mps": target_std,
                    "note": "ONNX export applies target denormalization internally "
                            "(see SpeedLSTMWithDenorm) — the exported model's output "
                            "is already real m/s, no denorm needed by the caller."},
                   f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nBest checkpoint: {ckpt_path}")
    print(f"Training report: {out_dir / 'training_report.json'}")
    print("\nNext: python -m training.export_iovnbd_lstm_onnx "
          f"--checkpoint {ckpt_path} --out_dir {out_dir}")


if __name__ == "__main__":
    main()
