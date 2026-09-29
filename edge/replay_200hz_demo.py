"""
edge/replay_200hz_demo.py — NAVDRIFT-0 200 Hz External-IMU Replay Demonstration
ISRO SIH 2026 PS #26168

Feeds a real IMU CSV (or, if none is supplied, a clearly-labeled synthetic
timing fixture — see below) through the EXISTING, unmodified
`navdrift_engine.NavdriftRTEngine` at a target processing rate, and measures
real wall-clock per-sample processing latency and throughput.

WHAT THIS DOES NOT CLAIM:
  - This is a SOFTWARE processing-rate demonstration, not a physical hardware
    test. No FOG, no physical 200 Hz IMU board, was used to produce this data
    unless you pass --imu pointing at a real recorded file that actually was
    sampled at 200 Hz.
  - When no --imu is given, this script generates a synthetic, deterministic
    timing fixture PURELY to exercise the streaming pipeline's mechanics
    (rate, dt handling, no dropped/blocked samples). It is printed and
    labeled as synthetic in every output artifact, and must never be quoted
    as a navigation accuracy or FOG-validation result — it has no ground
    truth trajectory and produces no ATE/drift numbers.
  - If you have a real IO-VNBD (or other real) IMU log, pass it via --imu and
    this becomes a real measurement against real data, at whatever true rate
    that file was actually sampled — the script does not alter or resample
    the source timing.

Usage:
    python edge/replay_200hz_demo.py --imu path/to/real_imu.csv --hz 200
    python edge/replay_200hz_demo.py --hz 200          # synthetic timing fixture, clearly labeled
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from navdrift_engine import NavdriftRTEngine  # existing, unmodified class
from edge.external_imu import ExternalIMUFrame, frames_from_csv, replay_frames, engine_step_from_frame


def _synthetic_timing_fixture(n: int, hz: float) -> list[ExternalIMUFrame]:
    """
    SYNTHETIC TEST FIXTURE — timing/mechanics only, not a navigation dataset.
    A phone sitting still (gravity on Z, small deterministic noise) sampled
    at exactly `hz`. Used only to prove the replay/streaming path itself
    runs at the target rate without dropped samples — never used to compute
    or report any position/drift/accuracy metric.
    """
    rng = np.random.RandomState(42)  # fixed seed: deterministic, reproducible
    dt = 1.0 / hz
    frames = []
    for i in range(n):
        noise = rng.normal(0, 0.01, size=6)
        frames.append(ExternalIMUFrame(
            timestamp=i * dt,
            ax=float(noise[0]), ay=float(noise[1]), az=float(9.80665 + noise[2]),
            gx=float(noise[3]), gy=float(noise[4]), gz=float(noise[5]),
        ))
    return frames


def run_replay(frames, target_hz: float, paced: bool = False):
    """
    Streams `frames` through a fresh NavdriftRTEngine at `target_hz`.
    If paced=True, sleeps between samples to approximate real-time delivery
    (useful for an interactive demo); if False (default), runs back-to-back
    as fast as possible, which is what actually measures worst-case
    per-sample processing latency — pacing would hide, not reveal it.

    Returns a dict of REAL measured statistics. Nothing here is estimated or
    assumed — every number is a wall-clock measurement from this run. Every
    stats dict returned by this function carries an explicit
    "validation_type": "SOFTWARE_BENCHMARK" marker and a
    "hardware_validation": "NOT_PERFORMED" marker (see module docstring) —
    never strip these fields when reporting or forwarding these numbers.
    """
    import warnings as _warnings

    engine = NavdriftRTEngine(lat0=0.0, lon0=0.0, fs=target_hz)
    dt_nominal = 1.0 / target_hz

    latencies_ms = []
    non_finite_count = 0
    warning_messages = []
    n = 0
    t_start = time.perf_counter()
    with _warnings.catch_warnings(record=True) as caught:
        # Deliberately does NOT call simplefilter() here: overriding the
        # active filter would defeat a caller's own `simplefilter("error")`
        # (used by tests/test_butterworth_stability.py to assert zero
        # warnings) by turning would-be errors into merely-recorded ones.
        # This only records whatever the CALLER's current filter already
        # lets through.
        for frame, dt_from_data in replay_frames(frames):
            dt = dt_from_data if dt_from_data > 0 else dt_nominal
            t0 = time.perf_counter()
            lat_pred, lon_pred = engine_step_from_frame(engine, frame, dt)
            t1 = time.perf_counter()
            if not (np.isfinite(lat_pred) and np.isfinite(lon_pred)):
                non_finite_count += 1
            latencies_ms.append((t1 - t0) * 1000.0)
            n += 1
            if paced:
                time.sleep(max(0.0, dt_nominal - (t1 - t0)))
        warning_messages = [str(w.message) for w in caught]
    elapsed_s = time.perf_counter() - t_start

    lat = np.array(latencies_ms)
    stats = {
        "validation_type": "SOFTWARE_BENCHMARK",
        "hardware_validation": "NOT_PERFORMED",
        "n_samples": n,
        "target_hz": target_hz,
        "budget_ms_per_sample": round(1000.0 / target_hz, 4),
        "measured_elapsed_s": round(elapsed_s, 4),
        "measured_effective_hz": round(n / elapsed_s, 2) if elapsed_s > 0 else None,
        "latency_ms_mean": round(float(lat.mean()), 4) if n else None,
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 4) if n else None,
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 4) if n else None,
        "latency_ms_max": round(float(lat.max()), 4) if n else None,
        "under_budget_fraction": round(float((lat < (1000.0 / target_hz)).mean()), 4) if n else None,
        "non_finite_output_count": non_finite_count,
        "warnings_raised": warning_messages,
        "errors_raised": [],  # any exception propagates out of this call rather than being swallowed here
    }
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAVDRIFT-0 external-IMU replay / rate demonstration")
    parser.add_argument("--imu", default=None, help="Real IMU CSV path (navdrift_engine.py column format). "
                                                       "If omitted, a labeled SYNTHETIC timing fixture is used instead.")
    parser.add_argument("--hz", type=float, default=200.0, help="Target processing rate (default 200)")
    parser.add_argument("--n", type=int, default=2000, help="Sample count for the synthetic fixture (ignored if --imu given)")
    parser.add_argument("--out_json", default=None, help="Optional path to write the measured stats as JSON")
    args = parser.parse_args()

    if args.imu:
        frames = frames_from_csv(args.imu)
        source_label = f"REAL data from {args.imu} ({len(frames)} samples)"
    else:
        frames = _synthetic_timing_fixture(args.n, args.hz)
        source_label = f"SYNTHETIC TIMING FIXTURE (not real sensor data, not a navigation benchmark) — {len(frames)} samples"

    print(f"\n{'='*70}\n"
          f"  NAVDRIFT-0 External-IMU Replay — {args.hz:.0f} Hz target\n"
          f"  Source: {source_label}\n"
          f"{'='*70}\n"
          f"  SOFTWARE BENCHMARK: this run measures real wall-clock processing\n"
          f"  latency/throughput of the software pipeline only.\n"
          f"  HARDWARE VALIDATION: NOT PERFORMED. No physical IMU/FOG board was\n"
          f"  used by this script. Do not report these numbers as hardware or\n"
          f"  200 Hz real-world validation.\n"
          f"{'='*70}")
    stats = run_replay(frames, target_hz=args.hz, paced=False)
    stats["source"] = source_label
    stats["is_synthetic_fixture"] = args.imu is None
    # run_replay() already sets validation_type/hardware_validation machine-readable
    # markers; this human-readable variant is kept for the printed/exported report.
    stats["hardware_validation_note"] = "NOT PERFORMED — software processing-rate measurement only"

    print("\n  -- SOFTWARE BENCHMARK RESULTS --")
    for k, v in stats.items():
        print(f"  {k:28s}: {v}")
    print(f"\n  -- HARDWARE VALIDATION: {stats['hardware_validation']} --")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats written to {args.out_json}")
