"""
edge/vehicle_calibration.py — NAVDRIFT-0 In-Vehicle Alignment & Calibration Module
ISRO SIH 2026 PS #26168

Formal phone-frame -> vehicle-frame calibration, extending (not duplicating) the
existing `detect_forward_axis()` heuristic in navdrift_engine.py. That function
answers ONE question ("which discrete axis is forward?"); this module answers the
fuller SIH requirement: a static, low-motion calibration that produces a real
roll/pitch estimate from gravity, an optional yaw/heading initialization from GNSS
course when available, and an explicit continuous phone->vehicle rotation matrix
- not just a best-of-three-axes pick.

Design constraints (per requirements):
  - static, low-motion calibration only (no assumption of a moving vehicle)
  - gravity-based roll/pitch estimation (classical, deterministic, no ML)
  - yaw/heading initialization ONLY when a reliable GNSS course is supplied —
    never fabricated when GNSS heading is unavailable or unreliable
  - explicit phone-frame -> vehicle-frame rotation (3x3 matrix), not just an
    axis index
  - a calibration result object with an explicit confidence/validity flag
  - safe fallback: when heading is unavailable, roll/pitch calibration still
    completes and is marked valid; only the yaw/heading field is left
    unavailable (None) rather than guessed
  - fully deterministic / reproducible: no randomness anywhere in this module

This module is Python-engine-only. It does not import, modify, or depend on
anything in frontend/mobile.html or frontend/desktop.html, and nothing in the
live phone navigation core calls into this module. It is meant to be used by
navdrift_engine.py's offline and real-time engines as an OPTIONAL richer
replacement for the current discrete-axis-only alignment step, once reviewed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────
GRAVITY_MPS2 = 9.80665

# A sample is treated as "low motion" (suitable for gravity-based roll/pitch
# calibration) when its accelerometer magnitude stays within this band around
# 1g AND its gyro magnitude stays below the rate threshold below, for the
# full calibration window. This is intentionally conservative: it is meant to
# reject a calibration window that includes real vehicle motion, not to be a
# generic motion detector.
ACCEL_MAG_TOL_MPS2 = 0.6          # allowed deviation from |g| while "still"
GYRO_RATE_TOL_RAD_S = 0.15        # allowed angular rate while "still"

# GNSS course is only trusted to seed yaw when the vehicle was actually moving
# fast enough for `coords.heading`/course-over-ground to be meaningful (a
# course reading at near-zero speed is essentially noise on real GNSS chips).
GNSS_COURSE_MIN_SPEED_MPS = 1.0


@dataclass
class CalibrationResult:
    """
    Output of `calibrate()`. Every field is either a real computed value or an
    explicit None/False — never a guessed placeholder.
    """
    valid: bool                     # True only if enough low-motion samples were found
    n_samples_used: int              # how many of the input samples were actually low-motion
    n_samples_total: int             # how many samples were offered

    roll_rad: Optional[float] = None
    pitch_rad: Optional[float] = None
    yaw_rad: Optional[float] = None       # None unless a reliable GNSS course seeded it
    yaw_source: Optional[str] = None      # "gnss_course" or None

    # 3x3 rotation matrix mapping a phone-frame vector to vehicle-frame
    # (forward, lateral, vertical). None if `valid` is False.
    R_phone_to_vehicle: Optional[np.ndarray] = None

    # Confidence in [0, 1], not a claim of physical accuracy — a simple,
    # reproducible function of (a) how much of the offered window was
    # actually usable as low-motion and (b) how tightly the accepted samples
    # agreed with each other. 0.0 whenever valid is False.
    confidence: float = 0.0

    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {
            "valid": self.valid,
            "n_samples_used": self.n_samples_used,
            "n_samples_total": self.n_samples_total,
            "roll_rad": self.roll_rad,
            "pitch_rad": self.pitch_rad,
            "yaw_rad": self.yaw_rad,
            "yaw_source": self.yaw_source,
            "confidence": self.confidence,
            "notes": list(self.notes),
        }
        if self.R_phone_to_vehicle is not None:
            d["R_phone_to_vehicle"] = self.R_phone_to_vehicle.tolist()
        else:
            d["R_phone_to_vehicle"] = None
        return d


def _low_motion_mask(accel: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    """
    accel, gyro: (N, 3) arrays in m/s^2 and rad/s respectively.
    Returns a boolean mask of samples that look genuinely stationary.
    """
    accel_mag = np.linalg.norm(accel, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    return (np.abs(accel_mag - GRAVITY_MPS2) < ACCEL_MAG_TOL_MPS2) & (gyro_mag < GYRO_RATE_TOL_RAD_S)


def _rotation_from_gravity(mean_accel: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    Classical gravity-vector tilt calibration. Given the mean stationary
    accelerometer reading (phone frame, m/s^2, gravity included), returns
    (roll_rad, pitch_rad, R_phone_to_vehicle) where R rotates a phone-frame
    vector into a vehicle-frame with:
      vehicle-X = vehicle forward-ish horizontal (see note below)
      vehicle-Y = vehicle lateral
      vehicle-Z = vehicle-up (opposite gravity)

    This determines the full ROLL and PITCH exactly (gravity fixes two of
    three rotational degrees of freedom). YAW is left as identity here — it
    is a separate, optional step in `calibrate()` (from GNSS course), because
    gravity alone cannot observe yaw/heading.
    """
    g = mean_accel / (np.linalg.norm(mean_accel) + 1e-12)
    # "up" in phone frame is the direction gravity is measured from — the
    # accelerometer reads +g when stationary and level (reaction to gravity),
    # so vehicle-up (in phone frame) is the measured accel direction itself.
    up_phone = g

    # Roll/pitch of the phone relative to a level vehicle chassis:
    roll_rad = math.atan2(up_phone[1], up_phone[2])
    pitch_rad = math.atan2(-up_phone[0], math.sqrt(up_phone[1] ** 2 + up_phone[2] ** 2))

    # Build R_phone_to_vehicle as the inverse of the roll/pitch tilt (yaw=0
    # placeholder; a real yaw estimate, if available, is applied on top of
    # this in `calibrate()`).
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    # Vehicle = R_pitch @ R_roll @ Phone (rotate roll first, then pitch),
    # then invert so it maps phone-frame -> vehicle-frame directly.
    R_tilt = R_pitch @ R_roll
    R_phone_to_vehicle = R_tilt.T  # orthonormal, so transpose == inverse

    return roll_rad, pitch_rad, R_phone_to_vehicle


def _apply_yaw(R_tilt_only: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Compose a yaw rotation (about vehicle-Z) on top of the roll/pitch-only matrix."""
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return R_yaw @ R_tilt_only


def calibrate(
    accel_window: np.ndarray,
    gyro_window: np.ndarray,
    gnss_course_rad: Optional[float] = None,
    gnss_speed_mps: Optional[float] = None,
) -> CalibrationResult:
    """
    Run a static, low-motion vehicle-frame calibration against an offered
    window of accelerometer/gyro samples (phone frame, m/s^2 and rad/s).

    accel_window, gyro_window: (N, 3) arrays, same N, same sample times.
    gnss_course_rad: optional course-over-ground in radians (0 = north,
        clockwise positive, matching typical GNSS `coords.heading` convention
        converted to radians by the caller). Only used if gnss_speed_mps
        indicates the vehicle was actually moving meaningfully at the time.
    gnss_speed_mps: optional ground speed at the same instant as
        gnss_course_rad — required to decide whether the course reading is
        trustworthy. If omitted, GNSS-based yaw is never used (safe default).

    Returns a CalibrationResult. Never raises on "not enough good samples" —
    that is represented as valid=False, not an exception, so a caller can
    safely retry with a longer window.
    """
    accel_window = np.asarray(accel_window, dtype=np.float64)
    gyro_window = np.asarray(gyro_window, dtype=np.float64)
    n_total = int(accel_window.shape[0])
    notes: list[str] = []

    if accel_window.ndim != 2 or accel_window.shape[1] != 3 or gyro_window.shape != accel_window.shape:
        return CalibrationResult(valid=False, n_samples_used=0, n_samples_total=n_total,
                                  notes=["malformed input: expected (N,3) accel and gyro arrays of equal shape"])

    mask = _low_motion_mask(accel_window, gyro_window)
    n_used = int(mask.sum())

    # Require a real, non-trivial amount of accepted low-motion evidence —
    # an arbitrary single sample passing the mask by chance is not a
    # calibration. MIN_SAMPLES is deliberately small (works down to a short
    # window) but non-zero.
    MIN_SAMPLES = 10
    if n_used < MIN_SAMPLES:
        notes.append(f"only {n_used}/{n_total} samples passed the low-motion gate "
                      f"(need >= {MIN_SAMPLES}) — calibration not attempted")
        return CalibrationResult(valid=False, n_samples_used=n_used, n_samples_total=n_total, notes=notes)

    accepted_accel = accel_window[mask]
    mean_accel = accepted_accel.mean(axis=0)
    roll_rad, pitch_rad, R_tilt_only = _rotation_from_gravity(mean_accel)

    # Confidence: fraction of the window that was usable, scaled down further
    # by how consistent the accepted samples were with each other (a tight
    # cluster around the mean is more trustworthy than a noisy one that
    # merely happened to pass the gate). Deterministic, bounded to [0, 1].
    coverage = n_used / n_total
    accel_std = float(np.linalg.norm(accepted_accel.std(axis=0)))
    consistency = max(0.0, 1.0 - min(accel_std / 1.0, 1.0))  # 1.0 std m/s^2 -> 0 consistency
    confidence = round(float(coverage * consistency), 4)

    yaw_rad: Optional[float] = None
    yaw_source: Optional[str] = None
    R_final = R_tilt_only

    if gnss_course_rad is not None and gnss_speed_mps is not None and gnss_speed_mps >= GNSS_COURSE_MIN_SPEED_MPS:
        yaw_rad = float(gnss_course_rad)
        yaw_source = "gnss_course"
        R_final = _apply_yaw(R_tilt_only, yaw_rad)
        notes.append(f"yaw initialized from GNSS course ({math.degrees(yaw_rad):.1f} deg) "
                      f"at speed {gnss_speed_mps:.2f} m/s")
    else:
        notes.append("no reliable GNSS course available (speed below "
                      f"{GNSS_COURSE_MIN_SPEED_MPS} m/s or course not supplied) — "
                      "yaw left unset rather than guessed; roll/pitch calibration still valid")

    return CalibrationResult(
        valid=True,
        n_samples_used=n_used,
        n_samples_total=n_total,
        roll_rad=round(float(roll_rad), 6),
        pitch_rad=round(float(pitch_rad), 6),
        yaw_rad=(round(yaw_rad, 6) if yaw_rad is not None else None),
        yaw_source=yaw_source,
        R_phone_to_vehicle=R_final,
        confidence=confidence,
        notes=notes,
    )


def transform_to_vehicle_frame(vec_phone: np.ndarray, result: CalibrationResult) -> np.ndarray:
    """
    Apply a valid CalibrationResult's rotation to a single phone-frame vector
    (e.g. one accelerometer or gyro sample), returning it in vehicle frame
    (forward, lateral, vertical). Raises ValueError if the result isn't valid
    — callers must check `result.valid` (or catch this) rather than silently
    getting an identity/garbage transform.
    """
    if not result.valid or result.R_phone_to_vehicle is None:
        raise ValueError("transform_to_vehicle_frame() called with an invalid CalibrationResult")
    return result.R_phone_to_vehicle @ np.asarray(vec_phone, dtype=np.float64)


if __name__ == "__main__":
    # Minimal, deterministic, documented example run — SYNTHETIC input used
    # only to demonstrate the module's mechanics (a phone lying flat, tilted
    # by a known angle, so the expected roll/pitch is known analytically).
    # This is NOT a navigation benchmark and reports no accuracy claim beyond
    # "the module recovers the angle it was given."
    print("edge/vehicle_calibration.py — example run (synthetic tilt, not real sensor data)")
    N = 200
    known_roll_deg = 12.0
    roll = math.radians(known_roll_deg)
    # Phone tilted by `roll` about its X axis relative to a level chassis:
    g_vec = np.array([0.0, GRAVITY_MPS2 * math.sin(roll), GRAVITY_MPS2 * math.cos(roll)])
    accel = np.tile(g_vec, (N, 1)) + np.random.RandomState(0).normal(0, 0.02, size=(N, 3))
    gyro = np.random.RandomState(1).normal(0, 0.01, size=(N, 3))
    res = calibrate(accel, gyro, gnss_course_rad=None, gnss_speed_mps=None)
    print(f"  known roll injected : {known_roll_deg:.2f} deg")
    print(f"  recovered roll      : {math.degrees(res.roll_rad):.2f} deg")
    print(f"  recovered pitch     : {math.degrees(res.pitch_rad):.2f} deg")
    print(f"  confidence          : {res.confidence}")
    print(f"  valid               : {res.valid}  (used {res.n_samples_used}/{res.n_samples_total})")
    print(f"  notes               : {res.notes}")
