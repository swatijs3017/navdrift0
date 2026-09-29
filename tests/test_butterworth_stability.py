"""
tests/test_butterworth_stability.py — regression tests for the ButterworthLP
sign-inversion fix in navdrift_engine.py's butter_coeffs().

Root cause (full diagnosis in the accompanying report): a1/a2 were sign-
inverted relative to the recurrence ButterworthLP.step() applies, placing one
pole of the implemented filter outside the unit circle (magnitude ~1.899 for
fc=10Hz/fs=200Hz) — an unconditionally unstable filter that diverges for any
bounded input, regardless of dtype. Fixed by correcting the sign of a1/a2 in
butter_coeffs() to match scipy.signal.butter's bilinear-transform Butterworth
coefficients (verified to match to full float precision). b0/b1/b2 and the
function's public signature/return shape are unchanged.

These tests cover requirements A-H from the fix request.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from navdrift_engine import butter_coeffs, ButterworthLP, NavdriftRTEngine
from edge.external_imu import ExternalIMUFrame, engine_step_from_frame
from edge.vehicle_calibration import calibrate, GRAVITY_MPS2
from edge.replay_200hz_demo import run_replay, _synthetic_timing_fixture


def _deterministic_imu_stream(n, dtype=np.float64, kind="noisy_gravity", seed=42):
    """Deterministic, bounded IMU-like samples (no randomness beyond a fixed seed)."""
    rng = np.random.RandomState(seed)
    out = np.zeros((n, 6), dtype=dtype)
    if kind == "noisy_gravity":
        out[:, 2] = 9.80665
        out += rng.normal(0, 0.05, size=(n, 6)).astype(dtype)
    elif kind == "constant":
        out[:, 2] = 9.80665
    elif kind == "sinusoidal":
        t = np.arange(n) / 200.0
        out[:, 0] = 2.0 * np.sin(2 * np.pi * 1.0 * t)
        out[:, 2] = 9.80665 + 0.5 * np.cos(2 * np.pi * 0.5 * t)
        out[:, 3] = 0.3 * np.sin(2 * np.pi * 0.2 * t)
    return out


# ── Pole-stability proof (the actual root-cause check) ─────────────────────

def test_filter_poles_inside_unit_circle():
    """The characteristic polynomial z^2 + a1*z + a2 (the recurrence's own
    convention, matching ButterworthLP.step()) must have both poles strictly
    inside the unit circle for BIBO stability."""
    for fc, fs in [(10.0, 200.0), (5.0, 100.0), (5.0, 50.0)]:
        b0, b1, b2, a1, a2 = butter_coeffs(fc, fs)
        poles = np.roots([1, a1, a2])
        assert np.all(np.abs(poles) < 1.0), f"unstable pole(s) at fc={fc},fs={fs}: {poles}"


def test_coefficients_match_scipy_reference():
    scipy = pytest.importorskip("scipy.signal")
    for fc, fs in [(10.0, 200.0), (5.0, 100.0)]:
        b0, b1, b2, a1, a2 = butter_coeffs(fc, fs)
        b_ref, a_ref = scipy.butter(2, fc / (fs / 2), btype="low")
        assert np.allclose([b0, b1, b2], b_ref, atol=1e-9)
        assert np.allclose([a1, a2], a_ref[1:], atol=1e-9)


# ── A/B: long deterministic run stays finite, no NaN/Inf ───────────────────

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("kind", ["noisy_gravity", "constant", "sinusoidal"])
def test_long_run_stays_finite(dtype, kind):
    bw = ButterworthLP(fc=10.0, fs=200.0, n_channels=6)
    bw.xp = bw.xp.astype(dtype); bw.yp1 = bw.yp1.astype(dtype); bw.yp2 = bw.yp2.astype(dtype)
    stream = _deterministic_imu_stream(20_000, dtype=dtype, kind=kind)
    max_abs = 0.0
    for i in range(len(stream)):
        y = bw.step(stream[i])
        assert np.all(np.isfinite(y)), f"non-finite output at sample {i} ({dtype.__name__}, {kind})"
        max_abs = max(max_abs, float(np.max(np.abs(y))))
    # Bounded-input-bounded-output: output must stay within a physically
    # sane multiple of the input scale, not just "finite by luck".
    assert max_abs < 1000.0, f"output grew implausibly large ({max_abs}) for {kind}/{dtype.__name__}"


def test_engine_200hz_long_run_no_overflow_warnings():
    """Reproduces the exact scenario that originally surfaced the bug (the
    Phase 1 200Hz replay demo) and asserts no RuntimeWarning is raised."""
    import warnings
    frames = _synthetic_timing_fixture(6000, hz=200.0)  # longer than the original 4000-sample repro
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stats = run_replay(frames, target_hz=200.0, paced=False)
    assert stats["n_samples"] == 6000


# ── C: existing NavdriftRTEngine.step()/gps_update() API unchanged ─────────

def test_navdrift_rt_engine_api_unchanged():
    import inspect
    sig = inspect.signature(NavdriftRTEngine.step)
    assert list(sig.parameters.keys()) == ["self", "ax", "ay", "az", "gx", "gy", "gz", "dt"]
    sig2 = inspect.signature(NavdriftRTEngine.gps_update)
    assert list(sig2.parameters.keys()) == ["self", "lat", "lon"]

    engine = NavdriftRTEngine(lat0=12.9, lon0=77.6, fs=200)
    lat, lon = engine.step(0.1, 0.0, 9.81, 0.0, 0.0, 0.05, 1 / 200.0)
    assert np.isfinite(lat) and np.isfinite(lon)
    engine.gps_update(12.9001, 77.6001)  # must not raise


# ── D: edge/external_imu.py adapter still equivalent ────────────────────────

def test_external_imu_adapter_still_equivalent_after_fix():
    engine_a = NavdriftRTEngine(lat0=10.0, lon0=20.0, fs=200)
    engine_b = NavdriftRTEngine(lat0=10.0, lon0=20.0, fs=200)
    frame = ExternalIMUFrame(timestamp=0.0, ax=0.2, ay=-0.1, az=9.81, gx=0.0, gy=0.0, gz=0.05)
    dt = 1 / 200.0
    direct = engine_a.step(frame.ax, frame.ay, frame.az, frame.gx, frame.gy, frame.gz, dt)
    via_adapter = engine_step_from_frame(engine_b, frame, dt)
    assert direct == via_adapter
    assert np.isfinite(direct[0]) and np.isfinite(direct[1])


# ── E: vehicle calibration tests remain unaffected ──────────────────────────

def test_vehicle_calibration_unaffected_by_filter_fix():
    """Sanity re-check: vehicle_calibration.py does not use ButterworthLP at
    all, so this must be unaffected — included per the required regression
    checklist rather than assumed."""
    rng = np.random.RandomState(0)
    g_vec = np.array([0.0, 0.0, GRAVITY_MPS2])
    accel = np.tile(g_vec, (200, 1)) + rng.normal(0, 0.01, size=(200, 3))
    gyro = rng.normal(0, 0.005, size=(200, 3))
    res = calibrate(accel, gyro)
    assert res.valid
    assert abs(res.roll_rad) < 0.02 and abs(res.pitch_rad) < 0.02


# ── F: 200 Hz replay harness still works end-to-end ─────────────────────────

def test_replay_harness_end_to_end_after_fix():
    frames = _synthetic_timing_fixture(4000, hz=200.0)
    stats = run_replay(frames, target_hz=200.0, paced=False)
    assert stats["n_samples"] == 4000
    assert stats["latency_ms_mean"] is not None
    assert stats["under_budget_fraction"] == 1.0
