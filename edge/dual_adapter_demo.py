"""
edge/dual_adapter_demo.py — NAVDRIFT-0 "one engine, two input sources" demo
ISRO SIH 2026 PS #26168

Concrete, runnable demonstration that the SAME, UNMODIFIED
`navdrift_engine.NavdriftRTEngine` can be driven by two genuinely different
adapter front-ends built on the same `ExternalIMUFrame` contract
(`edge/external_imu.py`):

  (a) EXTERNAL IMU adapter — `frames_from_csv()` / `frame_from_record()`,
      already implemented and unit-tested (`tests/test_external_imu.py`,
      19 tests). Demonstrated here by generating a small deterministic CSV
      (the one honest way to show "a CSV file, loaded from disk" without
      requiring a real external IMU board on this machine) and loading it
      with the real `frames_from_csv()` loader — no shortcut around the
      loader itself.

  (b) SIMULATED PHONE-SENSOR adapter — a new, clearly-labeled SIMULATED
      record stream shaped like what a phone's DeviceMotion/Geolocation API
      would hand a JS bridge (accelerometer in m/s^2, gyroscope in rad/s,
      optional GNSS fix, phone-style field names before normalization),
      converted record-by-record through `frame_from_record()` — the exact
      same conversion function a real phone-to-Python bridge would call. NO
      REAL PHONE CAPTURE WAS USED OR IS CLAIMED HERE: this is a
      representative, synthetic record shape only, used to prove the adapter
      boundary, not a navigation-accuracy or real-device benchmark.

Both paths end at the identical `engine_step_from_frame()` /
`build_nav_state()` call into ONE shared `NavdriftRTEngine` instance per
run (two separate instances across the two runs, since each adapter gets its
own fresh engine — sharing timelines across unrelated sensors would be
physically meaningless — but it is the same class, same `.step()` signature,
same adapter function, same `NavState` output contract, for both).

This module performs NO navigation-accuracy claim and NO hardware
validation. It is a software wiring/contract demonstration only.

Usage:
    python -m edge.dual_adapter_demo
    python -m edge.dual_adapter_demo --out_json dual_adapter_result.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from navdrift_engine import NavdriftRTEngine  # existing, unmodified class
from edge.external_imu import (
    frames_from_csv,
    frame_from_record,
    replay_frames,
    build_nav_state,
)


# ──────────────────────────────────────────────────────────────────────────
# (a) External-IMU-shaped adapter input: write a small deterministic CSV in
# navdrift_engine.py's existing flexible column format, then load it with
# the REAL frames_from_csv() loader (same code path external_imu.py users
# already rely on, e.g. edge/imu_replay_example.py).
# ──────────────────────────────────────────────────────────────────────────
def _write_external_imu_csv(path: Path, n: int = 50, hz: float = 100.0) -> None:
    rng = np.random.RandomState(11)
    dt_ms = 1000.0 / hz
    lines = ["time_ms,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,lat,lon"]
    lat0, lon0 = 12.9716, 77.5946  # Bengaluru, arbitrary fixed start point
    for i in range(n):
        noise = rng.normal(0, 0.01, size=6)
        lat = lat0 + i * 1e-6
        lon = lon0 + i * 1e-6
        lines.append(
            f"{i * dt_ms:.1f},{noise[0]:.5f},{noise[1]:.5f},{9.80665+noise[2]:.5f},"
            f"{noise[3]:.5f},{noise[4]:.5f},{noise[5]:.5f},{lat:.7f},{lon:.7f}"
        )
    path.write_text("\n".join(lines))


def run_external_imu_adapter(n: int = 50, hz: float = 100.0) -> list[dict]:
    """
    Adapter (a): CSV on disk -> frames_from_csv() -> engine_step_from_frame()
    via build_nav_state() -> list of NavState dicts. Every value in the
    returned list traces back to a row this function itself wrote to the
    CSV; nothing is read from any other source.
    """
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "external_imu_demo.csv"
        _write_external_imu_csv(csv_path, n=n, hz=hz)
        frames = frames_from_csv(str(csv_path), timestamp_is_ms=True, sample_rate_hz=hz)

    engine = NavdriftRTEngine(lat0=frames[0].lat or 12.9716, lon0=frames[0].lon or 77.5946, fs=hz)
    states = []
    for frame, dt in replay_frames(frames):
        ns = build_nav_state(engine, frame, dt if dt > 0 else 1.0 / hz)
        states.append(ns.as_dict())
    return states


# ──────────────────────────────────────────────────────────────────────────
# (b) SIMULATED phone-sensor-shaped adapter. Field names mirror the
# phone-side shapes used by frontend/mobile.html's live DeviceMotion/
# Geolocation handling (acceleration in m/s^2 including gravity, rotation
# rate in rad/s, an occasional GNSS fix) — NOT copied from mobile.html's
# code (this file never reads or imports frontend/mobile.html), just the
# same real-world sensor shape any phone bridge would hand over. Every
# record below is clearly a SIMULATED, deterministic fixture: no real phone
# was used to produce it.
# ──────────────────────────────────────────────────────────────────────────
def _simulated_phone_sensor_stream(n: int = 50, hz: float = 20.0) -> list[dict]:
    """
    SIMULATED, deterministic phone-shaped sensor records (NOT a real phone
    capture). Represents a plausible DeviceMotion+Geolocation record stream:
    accelerometer (m/s^2, gravity-inclusive on Z), gyroscope (rad/s), and a
    GNSS fix on roughly every 1-second boundary (phones typically report
    GPS far slower than IMU). Returned as plain dicts with PHONE-STYLE key
    names, deliberately NOT already in ExternalIMUFrame's own key names, so
    this function demonstrates genuine field-name translation happening in
    `_phone_record_to_engine_record()` below, not a relabeled copy.
    """
    rng = np.random.RandomState(99)
    dt = 1.0 / hz
    lat0, lon0 = 28.6139, 77.2090  # New Delhi, arbitrary fixed start point (different city
                                    # from the external-IMU fixture, underscoring these are
                                    # two independent, unrelated simulated runs)
    records = []
    for i in range(n):
        noise = rng.normal(0, 0.015, size=6)
        rec = {
            "t_phone_ms": i * dt * 1000.0,          # phone-style monotonic ms timestamp
            "accel_incl_gravity": [                  # phone-style 3-vector, m/s^2
                float(noise[0]), float(noise[1]), float(9.80665 + noise[2])
            ],
            "rotation_rate_rad_s": [                 # phone-style 3-vector, rad/s
                float(noise[3]), float(noise[4]), float(noise[5])
            ],
        }
        if i % int(hz) == 0:  # one GNSS fix roughly every second, phone-typical
            rec["gnss_fix"] = {"latitude": lat0 + i * 1e-6, "longitude": lon0 + i * 1e-6}
        records.append(rec)
    return records


def _phone_record_to_engine_record(rec: dict) -> dict:
    """
    Translation layer: phone-shaped record -> the plain dict
    `frame_from_record()` (edge/external_imu.py) already accepts. This is
    the adapter boundary a real phone-to-Python bridge would implement; it
    performs unit/name mapping only, no sensor simulation or fabrication —
    every value comes directly from `rec`.
    """
    ax, ay, az = rec["accel_incl_gravity"]
    gx, gy, gz = rec["rotation_rate_rad_s"]
    out = {
        "timestamp": rec["t_phone_ms"] / 1000.0,
        "ax": ax, "ay": ay, "az": az,
        "gx": gx, "gy": gy, "gz": gz,
    }
    fix = rec.get("gnss_fix")
    if fix:
        out["lat"] = fix["latitude"]
        out["lon"] = fix["longitude"]
    return out


def run_simulated_phone_adapter(n: int = 50, hz: float = 20.0) -> list[dict]:
    """
    Adapter (b): simulated phone-shaped records -> translation ->
    frame_from_record() -> engine_step_from_frame() via build_nav_state() ->
    list of NavState dicts. Uses the SAME build_nav_state()/
    engine_step_from_frame() call as adapter (a) above — the only thing that
    differs between the two adapters is how the record arrived at
    frame_from_record()'s input dict shape.
    """
    phone_records = _simulated_phone_sensor_stream(n=n, hz=hz)
    frames = [
        frame_from_record(_phone_record_to_engine_record(r), source="simulated_phone_sensor",
                           sample_rate_hz=hz)
        for r in phone_records
    ]

    first_fix = next((f for f in frames if f.has_gnss()), None)
    lat0 = first_fix.lat if first_fix else 28.6139
    lon0 = first_fix.lon if first_fix else 77.2090
    engine = NavdriftRTEngine(lat0=lat0, lon0=lon0, fs=hz)
    states = []
    for frame, dt in replay_frames(frames):
        ns = build_nav_state(engine, frame, dt if dt > 0 else 1.0 / hz)
        states.append(ns.as_dict())
    return states


def run_demo(n: int = 50) -> dict:
    ext_states = run_external_imu_adapter(n=n, hz=100.0)
    phone_states = run_simulated_phone_adapter(n=n, hz=20.0)
    return {
        "demo": "one NavdriftRTEngine, two adapters",
        "adapter_a_external_imu_csv": {
            "n_frames": len(ext_states),
            "first_state": ext_states[0] if ext_states else None,
            "last_state": ext_states[-1] if ext_states else None,
            "is_simulated": False,
            "note": "Deterministic fixture CSV written by this script, loaded through the "
                    "real frames_from_csv() loader — not a real external IMU board capture.",
        },
        "adapter_b_simulated_phone_sensor": {
            "n_frames": len(phone_states),
            "first_state": phone_states[0] if phone_states else None,
            "last_state": phone_states[-1] if phone_states else None,
            "is_simulated": True,
            "note": "SIMULATED phone-shaped DeviceMotion/Geolocation record stream — "
                    "NOT a real phone capture. Demonstrates the adapter boundary only.",
        },
        "shared_engine_class": "navdrift_engine.NavdriftRTEngine (unmodified)",
        "shared_output_contract": "edge.external_imu.NavState (via build_nav_state())",
        "validation_type": "SOFTWARE_WIRING_DEMONSTRATION",
        "hardware_validation": "NOT_PERFORMED",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NAVDRIFT-0 dual-adapter ('one engine, two input sources') demo")
    parser.add_argument("--n", type=int, default=50, help="Frames per adapter (default 50)")
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    result = run_demo(n=args.n)
    print(json.dumps(result, indent=2, default=str))

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResult written to {args.out_json}")
