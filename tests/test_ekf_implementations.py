"""
tests/test_ekf_implementations.py — regression tests pinning the CURRENT
numeric behavior of the two Python EKF implementations in this repo:

  1. navdrift_engine.EKF (+ its real-time wrapper NavdriftRTEngine)
  2. inference.runtime._EKF (used internally by NavDriftRuntime)

Context: a repo audit found THREE independently-implemented EKF/position
filters (the third is frontend/mobile.html's inline JS EKF, which is out of
scope here and untouched). Full comparison of all three, including exact
state/covariance/measurement-model differences, is in
docs/EKF_IMPLEMENTATIONS.md.

Neither of these two Python classes had any direct test coverage before this
file (tests/test_butterworth_stability.py imports NavdriftRTEngine but only
exercises the Butterworth filter inside it; inference/runtime.py had no test
coverage at all). These tests exist purely to PIN today's actual output to a
tight numeric tolerance, so that any future refactor/consolidation of these
filters is caught immediately if it changes observable behavior. The
expected values below were captured by running the current, unmodified code
(see docs/EKF_IMPLEMENTATIONS.md section 8 for the capture transcript) — they
are not independently-derived "correct" values, and are NOT a statement that
the current math is bug-free (see docs/EKF_IMPLEMENTATIONS.md section 6 for
known limitations of each filter).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from navdrift_engine import EKF as EngineEKF, NavdriftRTEngine
from inference.runtime import _EKF as RuntimeEKF


# ══════════════════════════════════════════════════════════════════════════
#  navdrift_engine.EKF
# ══════════════════════════════════════════════════════════════════════════
def _build_engine_ekf():
    ekf = EngineEKF(12.9716, 77.5946, 0.0)
    speeds = [5.0, 7.0, 10.0, 8.0, 6.0]
    dheads = [0.01, 0.02, -0.01, 0.0, 0.015]
    dt = 0.1
    for s, dh in zip(speeds, dheads):
        ekf.predict(s, dh, dt)
    return ekf


def test_engine_ekf_predict_state():
    """Pins navdrift_engine.EKF.predict()'s state after 5 deterministic steps."""
    ekf = _build_engine_ekf()
    expected_x = np.array([12.97160057, 77.5946, 0.035])
    np.testing.assert_allclose(ekf.x, expected_x, rtol=0, atol=1e-6)


def test_engine_ekf_predict_covariance_diagonal():
    """Pins the diagonal of P after prediction (position variances grow)."""
    ekf = _build_engine_ekf()
    diag = np.diag(ekf.P)
    expected_diag = np.array([1.05e-08, 1.050000338e-08, 0.1025])
    np.testing.assert_allclose(diag, expected_diag, rtol=1e-5, atol=1e-12)


def test_engine_ekf_predict_covariance_becomes_dense():
    """
    Confirms navdrift_engine.EKF does full (non-diagonal) EKF covariance
    propagation via F @ P @ F.T + Q — off-diagonal heading/position
    cross-terms must be non-zero after prediction. This is a real structural
    difference vs. frontend/mobile.html's decoupled-diagonal-only filter
    (see docs/EKF_IMPLEMENTATIONS.md section 3) and must not regress to a
    diagonal-only implementation silently.
    """
    ekf = _build_engine_ekf()
    assert abs(ekf.P[0, 2]) > 1e-12
    assert abs(ekf.P[1, 2]) > 1e-12


def test_engine_ekf_update_gps():
    """Pins navdrift_engine.EKF.update_gps()'s state after a GPS correction."""
    ekf = _build_engine_ekf()
    ekf.update_gps(12.97165, 77.59465)
    expected_x = np.array([12.97160144, 77.59460076, 0.0350044846])
    np.testing.assert_allclose(ekf.x, expected_x, rtol=0, atol=1e-6)


def test_engine_ekf_update_gps_shrinks_position_variance():
    """A GPS update must reduce (not grow) position covariance."""
    ekf = _build_engine_ekf()
    var_before = np.diag(ekf.P)[:2].copy()
    ekf.update_gps(12.97165, 77.59465)
    var_after = np.diag(ekf.P)[:2]
    assert np.all(var_after < var_before)


# ══════════════════════════════════════════════════════════════════════════
#  navdrift_engine.NavdriftRTEngine (real-time wrapper around EngineEKF)
# ══════════════════════════════════════════════════════════════════════════
def test_rt_engine_step_and_gps_update_pin_state():
    """
    Pins NavdriftRTEngine.step()/.gps_update() end-to-end behavior for a
    short deterministic sample sequence, including the calibration-bias
    subtraction, Butterworth filtering, dead-band speed integration, and the
    underlying EKF.predict()/update_gps() calls. Low-amplitude samples below
    DEAD_BAND intentionally keep int_spd at 0 here -- that is today's real
    behavior for small inputs, not an idealized one.
    """
    eng = NavdriftRTEngine(12.9716, 77.5946, fs=200)
    samples = [
        (0.5, 0.1, 9.8, 0.0, 0.0, 0.02),
        (0.6, 0.05, 9.79, 0.0, 0.0, 0.01),
        (0.4, 0.0, 9.81, 0.0, 0.0, -0.01),
        (0.55, 0.1, 9.8, 0.0, 0.0, 0.0),
    ]
    dt = 0.005
    positions = [eng.step(*s, dt) for s in samples]

    for lat, lon in positions:
        assert lat == pytest.approx(12.9716, abs=1e-4)
        assert lon == pytest.approx(77.5946, abs=1e-4)

    assert eng.heading == pytest.approx(3.480365756082094e-05, abs=1e-9)
    assert eng.int_spd == pytest.approx(0.0, abs=1e-9)

    eng.gps_update(12.97161, 77.59461)
    assert eng.int_spd == 0.0  # gps_update() always resets integration
    np.testing.assert_allclose(
        eng.ekf.x,
        np.array([12.9716002, 77.5946002, 3.48036576e-05]),
        rtol=0, atol=1e-6,
    )


# ══════════════════════════════════════════════════════════════════════════
#  inference.runtime._EKF
# ══════════════════════════════════════════════════════════════════════════
def _build_runtime_ekf():
    ekf = RuntimeEKF(dt=0.1)
    speeds = [5.0, 7.0, 10.0, 8.0, 6.0]
    gzs = [0.01, 0.02, -0.01, 0.0, 0.015]
    for s, g in zip(speeds, gzs):
        ekf.predict(s, g)
    return ekf


def test_runtime_ekf_predict_state():
    """Pins inference.runtime._EKF.predict()'s state after 5 deterministic steps."""
    ekf = _build_runtime_ekf()
    expected_x = np.array([3.59999235, 0.00649999, 0.0035])
    np.testing.assert_allclose(ekf.x, expected_x, rtol=0, atol=1e-6)


def test_runtime_ekf_predict_covariance_becomes_dense():
    """
    Confirms inference.runtime._EKF also does full (non-diagonal) EKF
    covariance propagation, structurally matching navdrift_engine.EKF (both
    use F @ P @ F.T + Q) and differing from frontend/mobile.html's decoupled
    filter. See docs/EKF_IMPLEMENTATIONS.md section 3.
    """
    ekf = _build_runtime_ekf()
    assert abs(ekf.P[0, 1]) > 1e-9
    assert abs(ekf.P[0, 2]) > 1e-9
    assert abs(ekf.P[1, 2]) > 1e-9


def test_runtime_ekf_update_full_state_observation():
    """
    Pins _EKF.update()'s state after a full x/y/heading observation (this
    class's update() takes a full state observation, unlike
    navdrift_engine.EKF.update_gps()'s lat/lon-only observation -- see
    docs/EKF_IMPLEMENTATIONS.md section 4 for why these two update() methods
    are not drop-in compatible).
    """
    ekf = _build_runtime_ekf()
    ekf.update(10.0, 5.0, 0.05)
    expected_x = np.array([7.93503351, 3.63772496, 0.20232292])
    np.testing.assert_allclose(ekf.x, expected_x, rtol=0, atol=1e-6)


def test_runtime_ekf_unc_property():
    """Pins the public .unc (major/minor position uncertainty) property."""
    ekf = _build_runtime_ekf()
    major, minor = ekf.unc
    assert major == pytest.approx(1.5374292528118492, abs=1e-6)
    assert minor == pytest.approx(1.0246950785570053, abs=1e-6)


def test_runtime_ekf_heading_wrap():
    """
    inference.runtime._EKF wraps heading to (-pi, pi] on every predict/update
    (via _wrap()), unlike navdrift_engine.EKF which lets heading grow
    unbounded. This pins that wrapping behavior explicitly so a future
    consolidation can't silently drop it.
    """
    ekf = RuntimeEKF(dt=0.1)
    ekf.predict(0.0, 10.0)  # large gz*dt should wrap
    assert -np.pi < ekf.x[2] <= np.pi
