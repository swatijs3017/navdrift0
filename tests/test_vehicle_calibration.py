"""
tests/test_vehicle_calibration.py — unit tests for edge/vehicle_calibration.py

All fixtures here are SYNTHETIC, deterministic test signals constructed to
have a known analytical answer (e.g. "a phone tilted by exactly 12 degrees
should recover a roll of exactly 12 degrees within numerical tolerance").
These are software correctness tests, not navigation accuracy benchmarks —
they assert nothing about real-world sensor behavior.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from edge.vehicle_calibration import calibrate, transform_to_vehicle_frame, GRAVITY_MPS2


def _still_samples(g_vec, n=200, accel_noise=0.01, gyro_noise=0.005, seed=0):
    rng = np.random.RandomState(seed)
    accel = np.tile(g_vec, (n, 1)) + rng.normal(0, accel_noise, size=(n, 3))
    gyro = rng.normal(0, gyro_noise, size=(n, 3))
    return accel, gyro


def test_flat_phone_recovers_zero_roll_pitch():
    g_vec = np.array([0.0, 0.0, GRAVITY_MPS2])
    accel, gyro = _still_samples(g_vec)
    res = calibrate(accel, gyro)
    assert res.valid
    assert abs(res.roll_rad) < math.radians(1.0)
    assert abs(res.pitch_rad) < math.radians(1.0)
    assert res.yaw_rad is None
    assert res.yaw_source is None
    assert res.confidence > 0.5


def test_known_roll_angle_recovered():
    known_roll_deg = 15.0
    roll = math.radians(known_roll_deg)
    g_vec = np.array([0.0, GRAVITY_MPS2 * math.sin(roll), GRAVITY_MPS2 * math.cos(roll)])
    accel, gyro = _still_samples(g_vec, accel_noise=0.005)
    res = calibrate(accel, gyro)
    assert res.valid
    assert abs(math.degrees(res.roll_rad) - known_roll_deg) < 1.0  # within 1 deg


def test_known_pitch_angle_recovered():
    known_pitch_deg = -8.0
    pitch = math.radians(known_pitch_deg)
    g_vec = np.array([-GRAVITY_MPS2 * math.sin(pitch), 0.0, GRAVITY_MPS2 * math.cos(pitch)])
    accel, gyro = _still_samples(g_vec, accel_noise=0.005)
    res = calibrate(accel, gyro)
    assert res.valid
    assert abs(math.degrees(res.pitch_rad) - known_pitch_deg) < 1.0


def test_insufficient_low_motion_samples_marks_invalid():
    # Real vehicle motion the whole window: large, varying accel and gyro —
    # should NOT pass the low-motion gate, and calibration must say so rather
    # than silently returning a bogus result.
    rng = np.random.RandomState(2)
    accel = rng.normal(0, 5.0, size=(50, 3)) + np.array([0, 0, GRAVITY_MPS2])
    gyro = rng.normal(0, 2.0, size=(50, 3))
    res = calibrate(accel, gyro)
    assert res.valid is False
    assert res.confidence == 0.0
    assert res.n_samples_used < 10
    assert len(res.notes) > 0


def test_gnss_course_seeds_yaw_only_when_moving():
    g_vec = np.array([0.0, 0.0, GRAVITY_MPS2])
    accel, gyro = _still_samples(g_vec)

    # Speed below threshold: yaw must remain unset, never guessed.
    res_slow = calibrate(accel, gyro, gnss_course_rad=math.radians(90), gnss_speed_mps=0.2)
    assert res_slow.valid
    assert res_slow.yaw_rad is None
    assert res_slow.yaw_source is None

    # Speed above threshold: yaw is set, and set to exactly the supplied value.
    res_moving = calibrate(accel, gyro, gnss_course_rad=math.radians(90), gnss_speed_mps=5.0)
    assert res_moving.valid
    assert res_moving.yaw_rad is not None
    assert res_moving.yaw_source == "gnss_course"
    # calibrate() intentionally rounds yaw_rad to 6 decimal places for
    # reproducible serialization (see CalibrationResult/as_dict) — tolerance
    # here matches that documented rounding, not floating-point equality.
    assert abs(res_moving.yaw_rad - math.radians(90)) < 1e-5


def test_malformed_input_shape_is_rejected_not_crashed():
    res = calibrate(np.zeros((10, 2)), np.zeros((10, 2)))
    assert res.valid is False
    assert res.n_samples_total == 10


def test_transform_to_vehicle_frame_requires_valid_result():
    g_vec = np.array([0.0, 0.0, GRAVITY_MPS2])
    accel, gyro = _still_samples(g_vec)
    invalid_res = calibrate(np.zeros((5, 3)), np.zeros((5, 3)))  # too few samples
    assert invalid_res.valid is False
    with pytest.raises(ValueError):
        transform_to_vehicle_frame(np.array([1.0, 0.0, 0.0]), invalid_res)

    valid_res = calibrate(accel, gyro)
    out = transform_to_vehicle_frame(np.array([0.0, 0.0, GRAVITY_MPS2]), valid_res)
    # A flat phone's gravity vector, transformed, should land close to [0,0,g].
    assert abs(out[2] - GRAVITY_MPS2) < 0.2


def test_insufficient_gravity_magnitude_rejects_samples():
    """
    Low gyro rate alone isn't enough to count as 'low motion' — the
    accelerometer magnitude must also be close to 1g. A phone in free-fall
    or under sustained non-gravity acceleration (low rotation rate, but
    accel magnitude far from g) must NOT be accepted as a calibration
    window.
    """
    rng = np.random.RandomState(3)
    # Accel magnitude far from GRAVITY_MPS2 (e.g. ~3g, low gyro noise).
    accel = np.tile(np.array([0.0, 0.0, 3 * GRAVITY_MPS2]), (100, 1)) + rng.normal(0, 0.01, size=(100, 3))
    gyro = rng.normal(0, 0.005, size=(100, 3))
    res = calibrate(accel, gyro)
    assert res.valid is False
    assert res.n_samples_used == 0
    assert "low-motion gate" in res.notes[0]


def test_gnss_course_without_speed_leaves_yaw_unset():
    """An unreliable/absent speed reading must never let a course value seed
    yaw — even if a course IS supplied, no speed means the course can't be
    trusted (a stationary GNSS chip's course-over-ground is noise)."""
    g_vec = np.array([0.0, 0.0, GRAVITY_MPS2])
    accel, gyro = _still_samples(g_vec)
    res = calibrate(accel, gyro, gnss_course_rad=math.radians(90), gnss_speed_mps=None)
    assert res.valid
    assert res.yaw_rad is None
    assert res.yaw_source is None


def test_yaw_wraparound_near_2pi_recovered_exactly():
    """A course-over-ground near the 0/2*pi boundary (e.g. 359 degrees) must
    be stored and applied exactly as supplied, not wrapped or corrupted by
    the rotation composition."""
    g_vec = np.array([0.0, 0.0, GRAVITY_MPS2])
    accel, gyro = _still_samples(g_vec)
    course = math.radians(359.0)
    res = calibrate(accel, gyro, gnss_course_rad=course, gnss_speed_mps=10.0)
    assert res.valid
    assert res.yaw_source == "gnss_course"
    assert abs(res.yaw_rad - course) < 1e-5
    # The composed rotation must still be a valid orthonormal matrix (no
    # corruption from the wraparound angle).
    R = res.R_phone_to_vehicle
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-8)
    assert abs(np.linalg.det(R) - 1.0) < 1e-8


def test_confidence_lower_for_noisier_stationary_window():
    """Confidence is a deterministic function of coverage and consistency —
    a noisier (but still low-motion) window must score lower confidence than
    a clean one, not the same value."""
    g_vec = np.array([0.0, 0.0, GRAVITY_MPS2])
    accel_clean, gyro_clean = _still_samples(g_vec, accel_noise=0.005, seed=11)
    accel_noisy, gyro_noisy = _still_samples(g_vec, accel_noise=0.3, seed=11)
    res_clean = calibrate(accel_clean, gyro_clean)
    res_noisy = calibrate(accel_noisy, gyro_noisy)
    assert res_clean.valid and res_noisy.valid
    assert res_clean.confidence > res_noisy.confidence


def test_partial_low_motion_window_uses_only_accepted_samples():
    """A window mixing genuine stationary samples with genuine motion
    samples must calibrate ONLY from the accepted (low-motion) subset, and
    n_samples_used must reflect that honestly rather than claim the full
    window was used."""
    rng = np.random.RandomState(5)
    still_g = np.array([0.0, 0.0, GRAVITY_MPS2])
    still_accel = np.tile(still_g, (150, 1)) + rng.normal(0, 0.01, size=(150, 3))
    still_gyro = rng.normal(0, 0.005, size=(150, 3))
    motion_accel = rng.normal(0, 4.0, size=(50, 3)) + still_g
    motion_gyro = rng.normal(0, 1.0, size=(50, 3))
    accel = np.concatenate([motion_accel, still_accel], axis=0)
    gyro = np.concatenate([motion_gyro, still_gyro], axis=0)
    res = calibrate(accel, gyro)
    assert res.valid
    assert res.n_samples_total == 200
    assert res.n_samples_used <= 150  # cannot exceed the genuinely-still portion
    assert res.n_samples_used >= 100  # most of the still portion should pass


def test_deterministic_reproducibility():
    g_vec = np.array([0.0, 1.0, GRAVITY_MPS2])
    accel, gyro = _still_samples(g_vec, seed=7)
    res1 = calibrate(accel, gyro)
    res2 = calibrate(accel, gyro)
    assert res1.roll_rad == res2.roll_rad
    assert res1.pitch_rad == res2.pitch_rad
    assert res1.confidence == res2.confidence
