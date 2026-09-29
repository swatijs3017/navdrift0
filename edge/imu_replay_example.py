"""
edge/imu_replay_example.py — simple external-IMU replay example
ISRO SIH 2026 PS #26168

A minimal, reproducible command-line example showing the external-IMU
contract in use end to end: load frames (real CSV or a labeled synthetic
fixture), validate the stream, replay them through the existing, unmodified
NavdriftRTEngine, and print the resulting trajectory summary.

This is a SOFTWARE BENCHMARK / DEMONSTRATION script. It performs no hardware
validation. If --imu is omitted it uses a clearly labeled synthetic fixture
and reports no navigation-accuracy claim.

Usage:
    python -m edge.imu_replay_example --imu path/to/real_imu.csv
    python -m edge.imu_replay_example --hz 100 --n 500     # synthetic fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from navdrift_engine import NavdriftRTEngine  # existing, unmodified class
from edge.external_imu import (
    ExternalIMUFrame, frames_from_csv, replay_frames, engine_step_from_frame,
    validate_stream,
)


def _synthetic_frames(n: int, hz: float) -> list[ExternalIMUFrame]:
    """SYNTHETIC EXAMPLE FIXTURE — a phone sitting still, sampled at `hz`.
    Used only to demonstrate the replay path when no real CSV is given.
    Never used to claim navigation accuracy."""
    rng = np.random.RandomState(7)
    dt = 1.0 / hz
    frames = []
    for i in range(n):
        noise = rng.normal(0, 0.01, size=6)
        frames.append(ExternalIMUFrame(
            timestamp=i * dt,
            ax=float(noise[0]), ay=float(noise[1]), az=float(9.80665 + noise[2]),
            gx=float(noise[3]), gy=float(noise[4]), gz=float(noise[5]),
            sample_rate_hz=hz, source="synthetic_example_fixture",
        ))
    return frames


def run_example(frames: list[ExternalIMUFrame], fs: float) -> dict:
    report = validate_stream(frames, expected_hz=fs)
    print(f"stream validation: {report.n_valid_frames}/{report.n_frames} frames valid, "
          f"{len(report.timestamp_order_violations)} order violations, "
          f"{len(report.dt_outliers)} dt outliers, "
          f"estimated rate {report.sample_rate_hz_estimated} Hz "
          f"(supported: {report.sample_rate_supported})")

    engine = NavdriftRTEngine(lat0=0.0, lon0=0.0, fs=fs)
    trajectory = []
    for frame, dt in replay_frames(frames):
        lat, lon = engine_step_from_frame(engine, frame, dt if dt > 0 else 1.0 / fs)
        trajectory.append((lat, lon))

    lats = np.array([p[0] for p in trajectory])
    lons = np.array([p[1] for p in trajectory])
    return {
        "n_frames": len(frames),
        "n_valid_frames": report.n_valid_frames,
        "sample_rate_hz_estimated": report.sample_rate_hz_estimated,
        "sample_rate_supported": report.sample_rate_supported,
        "final_lat": float(lats[-1]) if len(lats) else None,
        "final_lon": float(lons[-1]) if len(lons) else None,
        "lat_range": float(lats.max() - lats.min()) if len(lats) else None,
        "lon_range": float(lons.max() - lons.min()) if len(lons) else None,
        "validation_type": "SOFTWARE_BENCHMARK",
        "hardware_validation": "NOT_PERFORMED",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAVDRIFT-0 external-IMU replay example")
    parser.add_argument("--imu", default=None, help="Real IMU CSV path. If omitted, uses a labeled synthetic fixture.")
    parser.add_argument("--hz", type=float, default=100.0, help="Nominal sample rate (default 100)")
    parser.add_argument("--n", type=int, default=500, help="Sample count for the synthetic fixture (ignored if --imu given)")
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    if args.imu:
        frames = frames_from_csv(args.imu, sample_rate_hz=args.hz)
        print(f"Loaded {len(frames)} REAL frames from {args.imu}")
    else:
        frames = _synthetic_frames(args.n, args.hz)
        print(f"Using {len(frames)} SYNTHETIC EXAMPLE frames (not real sensor data)")

    result = run_example(frames, args.hz)
    for k, v in result.items():
        print(f"  {k:28s}: {v}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result written to {args.out_json}")
