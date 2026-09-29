"""
tests/test_external_imu.py — unit tests for edge/external_imu.py

Verifies the ExternalIMUFrame contract, CSV loading (against a small,
explicitly-synthetic CSV fixture written to a temp file for this test only —
never presented as navigation data), replay dt derivation, and the adapter
into the EXISTING, unmodified navdrift_engine.NavdriftRTEngine.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from edge.external_imu import (
    ExternalIMUFrame, frames_from_csv, replay_frames, engine_step_from_frame,
    MalformedIMURecordError, validate_stream, frame_from_record,
    SUPPORTED_SAMPLE_RATES_HZ,
)
from navdrift_engine import NavdriftRTEngine


def test_frame_accel_gyro_and_optional_fields():
    fr = ExternalIMUFrame(timestamp=0.0, ax=1, ay=2, az=3, gx=4, gy=5, gz=6)
    assert np.allclose(fr.accel(), [1, 2, 3])
    assert np.allclose(fr.gyro(), [4, 5, 6])
    assert fr.has_gnss() is False
    assert fr.has_magnetometer() is False

    fr2 = ExternalIMUFrame(timestamp=0.0, ax=1, ay=2, az=3, gx=4, gy=5, gz=6,
                            lat=12.9, lon=77.6, mx=1, my=2, mz=3)
    assert fr2.has_gnss() is True
    assert fr2.has_magnetometer() is True


def test_replay_frames_dt_derivation():
    frames = [
        ExternalIMUFrame(timestamp=0.000, ax=0, ay=0, az=9.8, gx=0, gy=0, gz=0),
        ExternalIMUFrame(timestamp=0.005, ax=0, ay=0, az=9.8, gx=0, gy=0, gz=0),
        ExternalIMUFrame(timestamp=0.010, ax=0, ay=0, az=9.8, gx=0, gy=0, gz=0),
    ]
    out = list(replay_frames(frames))
    assert len(out) == 3
    assert out[0][1] == 0.0
    assert abs(out[1][1] - 0.005) < 1e-9
    assert abs(out[2][1] - 0.005) < 1e-9


def test_frames_from_csv_synthetic_fixture(tmp_path):
    # Explicitly-synthetic CSV, written only for this test — 5 rows, known values.
    csv_path = tmp_path / "synthetic_imu_fixture.csv"
    csv_path.write_text(
        "timestamp_ms,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,lat,lon\n"
        "0,0.0,0.0,9.81,0.0,0.0,0.0,12.90,77.58\n"
        "10,0.1,0.0,9.80,0.01,0.0,0.0,,\n"
        "20,0.0,0.1,9.82,0.0,0.01,0.0,,\n"
    )
    frames = frames_from_csv(str(csv_path))
    assert len(frames) == 3
    assert frames[0].timestamp == 0.0
    assert abs(frames[1].timestamp - 0.010) < 1e-9
    assert frames[0].lat == 12.90
    assert frames[1].lat is None  # missing GT for this row must stay None, not 0 or interpolated


def test_frames_from_csv_missing_required_columns_raises(tmp_path):
    csv_path = tmp_path / "bad_fixture.csv"
    csv_path.write_text("timestamp_ms,foo,bar\n0,1,2\n")
    with pytest.raises(ValueError):
        frames_from_csv(str(csv_path))


def test_engine_step_from_frame_matches_direct_engine_call():
    """
    The adapter must be equivalent to calling NavdriftRTEngine.step() directly
    — this proves the contract layer doesn't alter navigation output, only
    wraps the existing, unmodified engine.
    """
    engine_a = NavdriftRTEngine(lat0=10.0, lon0=20.0, fs=100)
    engine_b = NavdriftRTEngine(lat0=10.0, lon0=20.0, fs=100)

    frame = ExternalIMUFrame(timestamp=0.0, ax=0.5, ay=0.0, az=9.81, gx=0.0, gy=0.0, gz=0.1)
    dt = 0.01

    direct = engine_a.step(frame.ax, frame.ay, frame.az, frame.gx, frame.gy, frame.gz, dt)
    via_adapter = engine_step_from_frame(engine_b, frame, dt)

    assert direct == via_adapter


# ── strengthened contract: metadata, validation, streaming records ─────────

def test_frame_carries_source_and_sample_rate_metadata():
    fr = ExternalIMUFrame(timestamp=0.0, ax=0, ay=0, az=9.8, gx=0, gy=0, gz=0,
                           source="external_board:bno085", sample_rate_hz=200.0)
    assert fr.source == "external_board:bno085"
    assert fr.sample_rate_hz == 200.0
    # Defaults stay None — never guessed when the caller doesn't supply them.
    fr2 = ExternalIMUFrame(timestamp=0.0, ax=0, ay=0, az=9.8, gx=0, gy=0, gz=0)
    assert fr2.source is None and fr2.sample_rate_hz is None


def test_validate_rejects_non_finite_required_field():
    fr = ExternalIMUFrame(timestamp=0.0, ax=float("nan"), ay=0, az=9.8, gx=0, gy=0, gz=0)
    problems = fr.validate()
    assert problems
    with pytest.raises(MalformedIMURecordError):
        fr.validate_or_raise()


def test_validate_rejects_non_finite_present_optional_field():
    fr = ExternalIMUFrame(timestamp=0.0, ax=0, ay=0, az=9.8, gx=0, gy=0, gz=0,
                           baro_hpa=float("inf"))
    assert fr.validate()


def test_validate_accepts_genuinely_absent_optional_fields():
    fr = ExternalIMUFrame(timestamp=0.0, ax=0, ay=0, az=9.8, gx=0, gy=0, gz=0)
    assert fr.validate() == []


def test_validate_rejects_out_of_range_latlon():
    fr = ExternalIMUFrame(timestamp=0.0, ax=0, ay=0, az=9.8, gx=0, gy=0, gz=0,
                           lat=200.0, lon=0.0)
    assert fr.validate()


def test_validate_rejects_negative_speed():
    fr = ExternalIMUFrame(timestamp=0.0, ax=0, ay=0, az=9.8, gx=0, gy=0, gz=0, speed_mps=-1.0)
    assert fr.validate()


def test_validate_stream_flags_timestamp_order_violation():
    frames = [
        ExternalIMUFrame(timestamp=0.00, ax=0, ay=0, az=9.8, gx=0, gy=0, gz=0),
        ExternalIMUFrame(timestamp=0.01, ax=0, ay=0, az=9.8, gx=0, gy=0, gz=0),
        ExternalIMUFrame(timestamp=0.005, ax=0, ay=0, az=9.8, gx=0, gy=0, gz=0),  # out of order
    ]
    report = validate_stream(frames)
    assert report.timestamp_order_violations == [2]
    assert report.ok is False


def test_validate_stream_flags_dt_outlier():
    frames = [ExternalIMUFrame(timestamp=i * 0.005, ax=0, ay=0, az=9.8, gx=0, gy=0, gz=0) for i in range(20)]
    # Insert a large gap partway through.
    frames.insert(10, ExternalIMUFrame(timestamp=frames[9].timestamp + 2.0, ax=0, ay=0, az=9.8, gx=0, gy=0, gz=0))
    report = validate_stream(frames)
    assert len(report.dt_outliers) >= 1


def test_validate_stream_estimates_sample_rate_and_flags_support():
    frames = [ExternalIMUFrame(timestamp=i / 200.0, ax=0, ay=0, az=9.8, gx=0, gy=0, gz=0) for i in range(50)]
    report = validate_stream(frames)
    assert report.sample_rate_hz_estimated is not None
    assert abs(report.sample_rate_hz_estimated - 200.0) < 1.0
    assert report.sample_rate_supported is True
    assert 200.0 in SUPPORTED_SAMPLE_RATES_HZ


def test_validate_stream_unsupported_rate_flagged_but_not_rejected():
    frames = [ExternalIMUFrame(timestamp=i / 37.0, ax=0, ay=0, az=9.8, gx=0, gy=0, gz=0) for i in range(20)]
    report = validate_stream(frames)
    assert report.n_frames == 20
    assert report.sample_rate_supported is False  # flagged...
    assert report.n_valid_frames == 20             # ...but frames themselves are still valid, not dropped


def test_validate_stream_reports_per_frame_problems_without_dropping_frames():
    frames = [
        ExternalIMUFrame(timestamp=0.0, ax=0, ay=0, az=9.8, gx=0, gy=0, gz=0),
        ExternalIMUFrame(timestamp=0.01, ax=float("nan"), ay=0, az=9.8, gx=0, gy=0, gz=0),
    ]
    report = validate_stream(frames)
    assert report.n_frames == 2
    assert report.n_valid_frames == 1
    assert report.frame_problems and report.frame_problems[0][0] == 1


def test_frame_from_record_builds_valid_frame():
    record = {"timestamp": 0.0, "ax": 0.1, "ay": 0.0, "az": 9.8, "gx": 0.0, "gy": 0.0, "gz": 0.01}
    fr = frame_from_record(record, source="unit_test", sample_rate_hz=100.0)
    assert fr.ax == 0.1 and fr.source == "unit_test" and fr.sample_rate_hz == 100.0


def test_frame_from_record_missing_required_key_raises_clear_error():
    record = {"timestamp": 0.0, "ax": 0.1, "ay": 0.0, "az": 9.8, "gx": 0.0, "gy": 0.0}  # gz missing
    with pytest.raises(MalformedIMURecordError, match="gz"):
        frame_from_record(record)


def test_frame_from_record_non_numeric_required_field_raises():
    record = {"timestamp": 0.0, "ax": "not_a_number", "ay": 0.0, "az": 9.8, "gx": 0.0, "gy": 0.0, "gz": 0.0}
    with pytest.raises(MalformedIMURecordError):
        frame_from_record(record)


def test_frame_from_record_optional_fields_default_to_none_not_zero():
    record = {"timestamp": 0.0, "ax": 0.1, "ay": 0.0, "az": 9.8, "gx": 0.0, "gy": 0.0, "gz": 0.0}
    fr = frame_from_record(record)
    assert fr.lat is None and fr.speed_mps is None and fr.mx is None


def test_frames_from_csv_stamps_source_and_sample_rate(tmp_path):
    csv_path = tmp_path / "rate_fixture.csv"
    csv_path.write_text(
        "timestamp_ms,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n"
        "0,0.0,0.0,9.81,0.0,0.0,0.0\n"
        "5,0.1,0.0,9.80,0.01,0.0,0.0\n"
    )
    frames = frames_from_csv(str(csv_path), sample_rate_hz=200.0)
    assert all(f.sample_rate_hz == 200.0 for f in frames)
    assert all(f.source == f"csv:{csv_path}" for f in frames)
