"""
edge/calibration_evidence_demo.py — reproducible calibration evidence generator
ISRO SIH 2026 PS #26168

Runs edge/vehicle_calibration.calibrate() against several deterministic,
labeled synthetic fixtures and prints/exports the results as evidence.
These are SOFTWARE FIXTURES with a known analytical answer (e.g. "a phone
tilted by exactly 12 degrees should recover roll close to 12 degrees"), not
physical vehicle test data. No physical vehicle or road test is implied or
claimed anywhere in this script.

Usage:
    python -m edge.calibration_evidence_demo --out_json calibration_evidence.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from edge.vehicle_calibration import calibrate, GRAVITY_MPS2


def _still_samples(g_vec, n=200, accel_noise=0.01, gyro_noise=0.005, seed=0):
    rng = np.random.RandomState(seed)
    accel = np.tile(g_vec, (n, 1)) + rng.normal(0, accel_noise, size=(n, 3))
    gyro = rng.normal(0, gyro_noise, size=(n, 3))
    return accel, gyro


def run_all_fixtures() -> list:
    cases = []

    # A. Gravity-based roll/pitch: flat phone.
    accel, gyro = _still_samples(np.array([0.0, 0.0, GRAVITY_MPS2]))
    res = calibrate(accel, gyro)
    cases.append({"case": "flat_phone_stationary", "kind": "deterministic_software_fixture",
                  "known_roll_deg": 0.0, "known_pitch_deg": 0.0, "result": res.as_dict()})

    # A. Gravity-based roll/pitch: known tilt.
    known_roll_deg = 15.0
    roll = math.radians(known_roll_deg)
    g_vec = np.array([0.0, GRAVITY_MPS2 * math.sin(roll), GRAVITY_MPS2 * math.cos(roll)])
    accel, gyro = _still_samples(g_vec, accel_noise=0.005)
    res = calibrate(accel, gyro)
    cases.append({"case": "known_roll_15deg", "kind": "deterministic_software_fixture",
                  "known_roll_deg": known_roll_deg, "known_pitch_deg": 0.0, "result": res.as_dict()})

    # B. GNSS-course-based yaw: reliable (moving fast enough).
    accel, gyro = _still_samples(np.array([0.0, 0.0, GRAVITY_MPS2]))
    course = math.radians(75.0)
    res = calibrate(accel, gyro, gnss_course_rad=course, gnss_speed_mps=8.0)
    cases.append({"case": "gnss_yaw_reliable_moving", "kind": "deterministic_software_fixture",
                  "known_course_deg": 75.0, "known_speed_mps": 8.0, "result": res.as_dict()})

    # B. GNSS-course-based yaw: unreliable (too slow), yaw must stay unset.
    accel, gyro = _still_samples(np.array([0.0, 0.0, GRAVITY_MPS2]))
    res = calibrate(accel, gyro, gnss_course_rad=course, gnss_speed_mps=0.2)
    cases.append({"case": "gnss_yaw_unreliable_too_slow", "kind": "deterministic_software_fixture",
                  "known_course_deg": 75.0, "known_speed_mps": 0.2, "result": res.as_dict()})

    # C. Phone-to-vehicle rotation matrix sanity (orthonormal, det=1).
    accel, gyro = _still_samples(np.array([0.0, 1.0, GRAVITY_MPS2]))
    res = calibrate(accel, gyro, gnss_course_rad=math.radians(200.0), gnss_speed_mps=5.0)
    R = res.R_phone_to_vehicle
    ortho_error = float(np.max(np.abs(R @ R.T - np.eye(3)))) if R is not None else None
    cases.append({"case": "phone_to_vehicle_rotation_sanity", "kind": "deterministic_software_fixture",
                  "orthonormality_error": ortho_error, "determinant": float(np.linalg.det(R)) if R is not None else None,
                  "result": res.as_dict()})

    # E. Fallback: insufficient low-motion samples (real vehicle motion the whole window).
    rng = np.random.RandomState(2)
    accel = rng.normal(0, 5.0, size=(50, 3)) + np.array([0, 0, GRAVITY_MPS2])
    gyro = rng.normal(0, 2.0, size=(50, 3))
    res = calibrate(accel, gyro)
    cases.append({"case": "fallback_insufficient_low_motion", "kind": "deterministic_software_fixture",
                  "result": res.as_dict()})

    # E. Fallback: accelerometer magnitude far from 1g (rejected, not miscalibrated).
    rng = np.random.RandomState(3)
    accel = np.tile(np.array([0.0, 0.0, 3 * GRAVITY_MPS2]), (100, 1)) + rng.normal(0, 0.01, size=(100, 3))
    gyro = rng.normal(0, 0.005, size=(100, 3))
    res = calibrate(accel, gyro)
    cases.append({"case": "fallback_accel_magnitude_far_from_g", "kind": "deterministic_software_fixture",
                  "result": res.as_dict()})

    return cases


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NAVDRIFT-0 calibration evidence generator")
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    cases = run_all_fixtures()
    print(f"\n{'='*70}\n  NAVDRIFT-0 Vehicle Calibration Evidence (deterministic software fixtures)\n"
          f"  No physical vehicle testing is implied by any result below.\n{'='*70}")
    for c in cases:
        r = c["result"]
        print(f"\n  case: {c['case']}")
        print(f"    valid={r['valid']}  roll_deg={None if r['roll_rad'] is None else round(math.degrees(r['roll_rad']),3)}"
              f"  pitch_deg={None if r['pitch_rad'] is None else round(math.degrees(r['pitch_rad']),3)}"
              f"  yaw_source={r['yaw_source']}  confidence={r['confidence']}")
        if r["notes"]:
            print(f"    notes: {r['notes']}")

    report = {
        "validation_type": "SOFTWARE_ONLY_VALIDATED",
        "hardware_validation": "NOT_PERFORMED",
        "note": "Every fixture here is a deterministic, seeded synthetic accelerometer/gyroscope "
                "window with a known analytical answer, not physical vehicle test data.",
        "cases": cases,
    }
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWritten to {args.out_json}")
