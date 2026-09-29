"""
eval/iovnbd_benchmark.py — real IO-VNBD offline benchmark.

Runs a CLASSICAL dead-reckoning baseline (heading + speed integration
during simulated GNSS outages, reset to ground truth whenever GPS is
available) against real, held-out IO-VNBD test sequences, and reports
ATE RMSE / mean drift / max drift per sequence plus trajectory plots.

This is explicitly NOT frontend/mobile.html's live EKF/ZUPT/NHC/
DriftFormer pipeline — it is a new, separate, offline-only classical
baseline built for this benchmark, so the live system is never imported,
called, or modified by this script. It also does not yet include any
trained AI model (LSTM speed / VAE fusion) — see --lstm_checkpoint /
--lstm_meta for optionally adding the AI-assisted comparison ONCE a real
checkpoint exists from training.train_iovnbd_speed_lstm. Until then, only
the classical baseline is reported — never a fabricated AI row.

Writes to results/iovnbd/ (a NEW directory — the existing
results/validation_full.json and results/isro_benchmark_table.csv from
the earlier n=1 benchmark are never touched or overwritten by this
script).

Usage:
    python -m eval.iovnbd_benchmark \
        --zip "raw/Synchronised V abd S datasets.zip" \
        --split_manifest results/iovnbd/split_manifest.json \
        --out_dir results/iovnbd \
        [--lstm_checkpoint checkpoints/iovnbd_speed_lstm/navdrift_lstm_best.pt]
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Optional

import numpy as np

from data.iovnbd import load_iovnbd_dataset
from data.loader import (
    Sequence, synchronize_sequence, simulate_gnss_outages,
    compute_se2_poses, compute_pose_deltas,
)


def classical_dr_trajectory(synced: dict, mask: np.ndarray) -> np.ndarray:
    """Simple classical dead-reckoning baseline for offline benchmarking.

    While GPS is available (mask True): the estimate IS the ground-truth
    GPS/INS pose (a receiver with GPS available does not need dead
    reckoning).
    While GPS is unavailable (mask False): position is propagated using
    the real reference speed (OdometrySample.speed, i.e. V-file
    Velocity/3.6) and real gyro-yaw-rate integration from the last known
    heading, exactly the way a minimal classical DR scheme works. This is
    NOT the live mobile EKF/ZUPT/NHC/DriftFormer stack — a separate,
    simpler, explicitly-labeled classical baseline for this offline
    benchmark only.
    """
    gps_xy = synced["gps_xy"]          # (N,2) ground truth, real GPS-derived
    gps_heading = synced["gps_heading"]  # (N,) ground truth heading, real
    odom = synced["odom"]              # (N,2) [speed_mps, steer_rad] real V-file reference
    imu = synced["imu"]                # (N,6) real IMU [ax,ay,az,gx,gy,gz]
    dt = np.diff(synced["timestamps"], prepend=synced["timestamps"][0])
    dt[0] = dt[1] if len(dt) > 1 else 0.1

    N = len(gps_xy)
    est_xy = np.zeros((N, 2), dtype=np.float64)
    est_heading = np.zeros(N, dtype=np.float64)
    est_xy[0] = gps_xy[0]
    est_heading[0] = gps_heading[0]

    for i in range(1, N):
        if mask[i]:
            est_xy[i] = gps_xy[i]
            est_heading[i] = gps_heading[i]
        else:
            # gyro_z is the S-file's GYROSCOPE Roll channel positionally
            # (see data/iovnbd.py docstring — axis semantics are an open
            # item, used here only as "the yaw-rate-like channel", not a
            # confirmed physical-axis claim).
            heading = est_heading[i - 1] + imu[i, 5] * dt[i]
            speed = float(odom[i, 0])
            dx = speed * math.cos(heading) * dt[i]
            dy = speed * math.sin(heading) * dt[i]
            est_xy[i] = est_xy[i - 1] + np.array([dx, dy])
            est_heading[i] = heading

    return est_xy


def ate_rmse(est_xy: np.ndarray, gt_xy: np.ndarray, mask: np.ndarray) -> dict:
    """Error metrics computed ONLY over GNSS-outage windows (where the
    baseline actually had to dead-reckon) — evaluating error during
    GPS-available windows would trivially be ~0 by construction and would
    misrepresent the benchmark."""
    outage_idx = np.where(~mask)[0]
    if len(outage_idx) == 0:
        return {"ate_rmse_m": None, "mean_drift_m": None, "max_drift_m": None, "n_outage_samples": 0}
    err = np.linalg.norm(est_xy[outage_idx] - gt_xy[outage_idx], axis=1)
    return {
        "ate_rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "mean_drift_m": float(np.mean(err)),
        "max_drift_m": float(np.max(err)),
        "n_outage_samples": int(len(outage_idx)),
    }


def plot_trajectory(gt_xy, est_xy, mask, out_png: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], "g-", linewidth=1.5, label="Ground truth (real GPS/INS)")
    ax.plot(est_xy[:, 0], est_xy[:, 1], "r--", linewidth=1.2, label="Classical DR (outage windows)")
    outage_pts = gt_xy[~mask]
    if len(outage_pts):
        ax.scatter(outage_pts[:, 0], outage_pts[:, 1], s=4, c="orange", alpha=0.4, label="GNSS outage (simulated)")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(f"{title} — trajectory")
    ax.legend(loc="best", fontsize=8)
    ax.axis("equal")

    ax2 = axes[1]
    err = np.linalg.norm(est_xy - gt_xy, axis=1)
    t = np.arange(len(err))
    ax2.plot(t, err, "b-", linewidth=0.8)
    ax2.fill_between(t, 0, err, where=~mask, alpha=0.15, color="orange", label="GNSS outage")
    ax2.set_xlabel("sample index")
    ax2.set_ylabel("position error (m)")
    ax2.set_title(f"{title} — error vs time")
    ax2.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", required=True)
    p.add_argument("--split_manifest", required=True)
    p.add_argument("--out_dir", default="results/iovnbd")
    p.add_argument("--min_outage_s", type=float, default=10.0)
    p.add_argument("--max_outage_s", type=float, default=60.0)
    p.add_argument("--imu_hz", type=float, default=10.0)  # REAL measured IO-VNBD rate, not the 100Hz default
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_test_sequences", type=int, default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    with open(args.split_manifest) as f:
        manifest = json.load(f)
    test_names = manifest["test"]
    if args.max_test_sequences:
        test_names = test_names[:args.max_test_sequences]

    all_sequences, metas = load_iovnbd_dataset(args.zip)
    by_name = {s.name: s for s in all_sequences}
    test_seqs: List[Sequence] = [by_name[n] for n in test_names if n in by_name]

    if not test_seqs:
        raise RuntimeError("No held-out test sequences available from split_manifest — cannot benchmark.")

    rng = random.Random(args.seed)
    results = []
    for seq in test_seqs:
        synced = synchronize_sequence(seq)
        if synced is None:
            results.append({"sequence": seq.name, "status": "SKIPPED_EMPTY_STREAM"})
            continue
        N = len(synced["timestamps"])
        if N < int(args.min_outage_s * args.imu_hz) * 2:
            results.append({"sequence": seq.name, "status": "SKIPPED_TOO_SHORT", "n_samples": N})
            continue

        mask = simulate_gnss_outages(N, imu_hz=args.imu_hz,
                                      min_outage_s=args.min_outage_s,
                                      max_outage_s=args.max_outage_s,
                                      min_gap_s=30.0,
                                      rng=random.Random(rng.randint(0, 2**31)))
        est_xy = classical_dr_trajectory(synced, mask)
        gt_xy = synced["gps_xy"]
        metrics = ate_rmse(est_xy, gt_xy, mask)

        png_path = out_dir / "plots" / f"{seq.name}_trajectory.png"
        plot_trajectory(gt_xy, est_xy, mask, png_path, title=f"IO-VNBD real sequence: {seq.name}")

        results.append({
            "sequence": seq.name,
            "status": "OK",
            "n_samples": N,
            "duration_s": float(synced["timestamps"][-1] - synced["timestamps"][0]),
            "n_outage_windows_approx": int(np.sum(np.diff(mask.astype(int)) == -1)),
            "plot": str(png_path),
            **metrics,
        })

    ok_results = [r for r in results if r["status"] == "OK"]
    summary = {
        "benchmark_name": "iovnbd_classical_dr_baseline",
        "note": ("Classical dead-reckoning baseline (heading integration + real V-file "
                 "reference speed) evaluated ONLY over simulated GNSS-outage windows on "
                 "real, held-out IO-VNBD test sequences. This is a NEW, separate offline "
                 "benchmark — not the live frontend/mobile.html EKF/ZUPT/NHC/DriftFormer "
                 "pipeline, and not the earlier n=1 results/validation_full.json benchmark, "
                 "which is untouched."),
        "n_test_sequences_attempted": len(test_seqs),
        "n_test_sequences_scored": len(ok_results),
        "imu_hz_used": args.imu_hz,
        "min_outage_s": args.min_outage_s,
        "max_outage_s": args.max_outage_s,
        "seed": args.seed,
        "mean_ate_rmse_m": (float(np.mean([r["ate_rmse_m"] for r in ok_results if r["ate_rmse_m"] is not None]))
                             if ok_results else None),
        "mean_max_drift_m": (float(np.mean([r["max_drift_m"] for r in ok_results if r["max_drift_m"] is not None]))
                              if ok_results else None),
        "per_sequence": results,
    }

    with open(out_dir / "iovnbd_classical_dr_benchmark.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "per_sequence"}, indent=2))
    print(f"\n{len(ok_results)}/{len(test_seqs)} test sequences scored.")
    print(f"Metrics: {out_dir / 'iovnbd_classical_dr_benchmark.json'}")
    print(f"Plots:   {out_dir / 'plots'}/")


if __name__ == "__main__":
    main()
