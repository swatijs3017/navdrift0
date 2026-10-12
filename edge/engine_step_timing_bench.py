"""
edge/engine_step_timing_bench.py — tight timing-loop engine.step() benchmark
ISRO SIH 2026 PS #26168

Measures real wall-clock per-call latency and effective throughput of
`engine_step_from_frame()` and `build_nav_state()` (edge/external_imu.py)
directly, feeding a reproducible deterministic synthetic 200 Hz-rate stream
(default 10,000 frames at nominal 5ms spacing, fixed RNG seed 42 — same
fixture shape as `edge/replay_200hz_demo.py`'s synthetic fixture).

This is a SOFTWARE load-test of the Python pipeline on whatever machine runs
it. It is NOT a FOG/physical-IMU hardware validation, and makes no
navigation-accuracy claim — the synthetic fixture has no ground truth
trajectory. Report numbers from this script with that caveat attached.

Usage:
    python edge/engine_step_timing_bench.py
    python edge/engine_step_timing_bench.py --n 10000 --hz 200 --out_json bench.json
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
from edge.external_imu import ExternalIMUFrame, engine_step_from_frame, build_nav_state


def _synthetic_frames(n: int, hz: float) -> list[ExternalIMUFrame]:
    """Deterministic (seed=42), stationary-phone-shaped synthetic fixture —
    timing/throughput mechanics only, no navigation-accuracy claim."""
    rng = np.random.RandomState(42)
    dt = 1.0 / hz
    frames = []
    for i in range(n):
        noise = rng.normal(0, 0.01, size=6)
        frames.append(ExternalIMUFrame(
            timestamp=i * dt, ax=float(noise[0]), ay=float(noise[1]), az=float(9.80665 + noise[2]),
            gx=float(noise[3]), gy=float(noise[4]), gz=float(noise[5]),
        ))
    return frames


def bench_call(call_name: str, frames: list[ExternalIMUFrame], hz: float) -> dict:
    dt_nom = 1.0 / hz
    engine = NavdriftRTEngine(lat0=0.0, lon0=0.0, fs=hz)
    latencies_ms = []
    t_start = time.perf_counter()
    for fr in frames:
        t0 = time.perf_counter()
        if call_name == "engine_step_from_frame":
            engine_step_from_frame(engine, fr, dt_nom)
        else:
            build_nav_state(engine, fr, dt_nom)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
    elapsed_s = time.perf_counter() - t_start

    lat = np.array(latencies_ms)
    return {
        "validation_type": "SOFTWARE_LOAD_TEST",
        "hardware_validation": "NOT_PERFORMED",
        "call": call_name,
        "n_frames": len(frames),
        "target_hz": hz,
        "elapsed_s": round(elapsed_s, 4),
        "effective_hz": round(len(frames) / elapsed_s, 2) if elapsed_s > 0 else None,
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 5),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 5),
        "latency_ms_p99": round(float(np.percentile(lat, 99)), 5),
        "latency_ms_max": round(float(lat.max()), 5),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tight timing-loop benchmark of engine_step_from_frame/build_nav_state")
    parser.add_argument("--n", type=int, default=10000, help="Number of synthetic frames (default 10000)")
    parser.add_argument("--hz", type=float, default=200.0, help="Nominal rate of the synthetic fixture (default 200)")
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    frames = _synthetic_frames(args.n, args.hz)
    results = [
        bench_call("engine_step_from_frame", frames, args.hz),
        bench_call("build_nav_state", frames, args.hz),
    ]
    print(f"\nSOFTWARE LOAD TEST — single-threaded Python, no GPU. "
          f"NOT a FOG-hardware validation. Fixed synthetic seed=42, n={args.n}, nominal hz={args.hz}.\n")
    for r in results:
        print(json.dumps(r, indent=2))

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.out_json}")
