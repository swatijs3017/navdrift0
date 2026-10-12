"""
eval/iovnbd_benchmark_report.py — REPORTING-ONLY addition on top of
eval/iovnbd_benchmark.py.

This script does NOT change the classical dead-reckoning baseline's math.
It imports and calls the exact same functions
(synchronize_sequence, simulate_gnss_outages, classical_dr_trajectory,
ate_rmse) from eval.iovnbd_benchmark with the exact same default
hyperparameters (seed=42, imu_hz=10.0, min_outage_s=10.0, max_outage_s=60.0,
min_gap_s=30.0), so the per-sequence ATE / mean_drift / max_drift numbers
it reproduces are identical to a fresh run of the original script.

What it ADDS (computed, not fabricated, purely as reporting):
  - distance_traveled_m: real ground-truth path length (sum of consecutive
    GPS-derived XY segment lengths) accumulated only over the simulated
    GNSS-outage samples for that sequence (the denominator the SIH
    <10%-of-distance-travelled requirement needs).
  - outage_duration_s: n_outage_samples / imu_hz
  - speed stats (mean/max, m/s) over the outage samples, taken directly
    from the real V-file reference speed channel (odom[:,0]) used by the
    baseline itself.
  - drift_pct = max_drift_m / distance_traveled_m * 100, and the pass/fail
    verdict against the SIH <10% target.

It writes its own output JSON to a disposable directory (passed via
--out_dir) and NEVER touches results/iovnbd/iovnbd_classical_dr_benchmark.json
or results/iovnbd/split_manifest.json.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import numpy as np

from data.iovnbd import load_iovnbd_dataset
from data.loader import Sequence, synchronize_sequence, simulate_gnss_outages
from eval.iovnbd_benchmark import classical_dr_trajectory, ate_rmse


def contiguous_true_runs(mask_false_positions: np.ndarray):
    """Given indices where mask is False (outage), yield contiguous runs."""
    if len(mask_false_positions) == 0:
        return
    start = mask_false_positions[0]
    prev = mask_false_positions[0]
    for idx in mask_false_positions[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        yield (start, prev)
        start = idx
        prev = idx
    yield (start, prev)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", required=True)
    p.add_argument("--split_manifest", required=True)
    p.add_argument("--out_dir", required=True,
                    help="Disposable report output dir — must NOT be results/iovnbd/")
    p.add_argument("--min_outage_s", type=float, default=10.0)
    p.add_argument("--max_outage_s", type=float, default=60.0)
    p.add_argument("--imu_hz", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    assert "results/iovnbd" not in str(out_dir).replace("\\", "/").rstrip("/") or out_dir.name != "iovnbd", \
        "Refusing to write into results/iovnbd/ directly."
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.split_manifest) as f:
        manifest = json.load(f)
    test_names = manifest["test"]

    all_sequences, metas = load_iovnbd_dataset(args.zip)
    by_name = {s.name: s for s in all_sequences}
    test_seqs: List[Sequence] = [by_name[n] for n in test_names if n in by_name]

    rng = random.Random(args.seed)
    rows = []
    for seq in test_seqs:
        synced = synchronize_sequence(seq)
        if synced is None:
            rows.append({"sequence": seq.name, "status": "SKIPPED_EMPTY_STREAM"})
            continue
        N = len(synced["timestamps"])
        if N < int(args.min_outage_s * args.imu_hz) * 2:
            rows.append({"sequence": seq.name, "status": "SKIPPED_TOO_SHORT", "n_samples": N})
            continue

        mask = simulate_gnss_outages(N, imu_hz=args.imu_hz,
                                      min_outage_s=args.min_outage_s,
                                      max_outage_s=args.max_outage_s,
                                      min_gap_s=30.0,
                                      rng=random.Random(rng.randint(0, 2**31)))
        est_xy = classical_dr_trajectory(synced, mask)
        gt_xy = synced["gps_xy"]
        metrics = ate_rmse(est_xy, gt_xy, mask)

        outage_idx = np.where(~mask)[0]
        # distance traveled: sum of ground-truth segment lengths within each
        # contiguous outage run (never bridging the gap between two separate
        # outage windows, which would fabricate distance that wasn't driven
        # during an outage).
        distance_traveled_m = 0.0
        for (s_idx, e_idx) in contiguous_true_runs(outage_idx):
            if e_idx > s_idx:
                seg = gt_xy[s_idx:e_idx + 1]
                distance_traveled_m += float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))

        odom_speed = synced["odom"][:, 0]
        outage_speed = odom_speed[outage_idx] if len(outage_idx) else np.array([])
        full_speed = odom_speed

        drift_pct = (metrics["max_drift_m"] / distance_traveled_m * 100.0
                     if metrics["max_drift_m"] is not None and distance_traveled_m > 0 else None)
        ate_pct = (metrics["ate_rmse_m"] / distance_traveled_m * 100.0
                   if metrics["ate_rmse_m"] is not None and distance_traveled_m > 0 else None)

        rows.append({
            "sequence": seq.name,
            "status": "OK",
            "n_samples": N,
            "duration_s": float(synced["timestamps"][-1] - synced["timestamps"][0]),
            "n_outage_windows_approx": int(np.sum(np.diff(mask.astype(int)) == -1)),
            "outage_duration_s": float(metrics["n_outage_samples"]) / args.imu_hz if metrics["n_outage_samples"] else 0.0,
            "n_outage_samples": metrics["n_outage_samples"],
            "distance_traveled_during_outage_m": distance_traveled_m,
            "mean_speed_outage_mps": float(np.mean(outage_speed)) if len(outage_speed) else None,
            "max_speed_outage_mps": float(np.max(outage_speed)) if len(outage_speed) else None,
            "mean_speed_full_seq_mps": float(np.mean(full_speed)) if len(full_speed) else None,
            "max_speed_full_seq_mps": float(np.max(full_speed)) if len(full_speed) else None,
            "ate_rmse_m": metrics["ate_rmse_m"],
            "mean_drift_m": metrics["mean_drift_m"],
            "max_drift_m": metrics["max_drift_m"],
            "ate_pct_of_distance": ate_pct,
            "max_drift_pct_of_distance": drift_pct,
            "pass_lt_10pct_max_drift": (drift_pct < 10.0) if drift_pct is not None else None,
            "pass_lt_10pct_ate": (ate_pct < 10.0) if ate_pct is not None else None,
        })

    ok_rows = [r for r in rows if r["status"] == "OK"]
    n_pass_max = sum(1 for r in ok_rows if r.get("pass_lt_10pct_max_drift"))
    n_pass_ate = sum(1 for r in ok_rows if r.get("pass_lt_10pct_ate"))
    drift_pcts = [r["max_drift_pct_of_distance"] for r in ok_rows if r["max_drift_pct_of_distance"] is not None]
    ate_pcts = [r["ate_pct_of_distance"] for r in ok_rows if r["ate_pct_of_distance"] is not None]

    summary = {
        "report_name": "iovnbd_classical_dr_per_sequence_report (reporting-only wrapper, same math as eval/iovnbd_benchmark.py)",
        "n_test_sequences_scored": len(ok_rows),
        "n_pass_lt_10pct_max_drift": n_pass_max,
        "n_pass_lt_10pct_ate_rmse": n_pass_ate,
        "mean_max_drift_pct": float(np.mean(drift_pcts)) if drift_pcts else None,
        "median_max_drift_pct": float(np.median(drift_pcts)) if drift_pcts else None,
        "mean_ate_pct": float(np.mean(ate_pcts)) if ate_pcts else None,
        "median_ate_pct": float(np.median(ate_pcts)) if ate_pcts else None,
        "per_sequence": rows,
    }

    out_path = out_dir / "iovnbd_per_sequence_report.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "per_sequence"}, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
