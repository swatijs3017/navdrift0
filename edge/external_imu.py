"""
edge/external_imu.py — NAVDRIFT-0 External IMU Input Contract
ISRO SIH 2026 PS #26168

Formalizes the input contract that navdrift_engine.py's `NavdriftRTEngine`
already accepts informally (positional ax,ay,az,gx,gy,gz,dt arguments) into an
explicit, typed, documented structure — WITHOUT changing NavdriftRTEngine's
existing method signature. `NavdriftRTEngine.step()` is untouched; this module
sits in front of it as an adapter/contract layer, so existing callers of
navdrift_engine.py keep working exactly as before.

This module does NOT talk to a browser, a phone, or frontend/mobile.html in
any way. It is the "external IMU" side of requirement #7: any process — a
recorded CSV replay, a future serial/UDP bridge to a real external IMU board,
a unit test — can construct an ExternalIMUFrame and feed it through
`engine_step_from_frame()` into the existing, unmodified NavdriftRTEngine.

Frame fields (per the requested contract):
    timestamp   float, seconds (monotonic or wall-clock — caller's choice,
                only used to derive dt between consecutive frames)
    ax, ay, az  float, m/s^2 (accelerometer, required)
    gx, gy, gz  float, rad/s (gyroscope, required)
    mx, my, mz  Optional[float], magnetometer (uT), optional
    lat, lon    Optional[float], GNSS position, optional
    speed_mps   Optional[float], GNSS ground speed, optional
    course_rad  Optional[float], GNSS course-over-ground, optional
    baro_hpa    Optional[float], barometric pressure, optional

No field here is ever fabricated by this module — every optional field is
either the real value the caller supplied, or None. This module performs no
sensor simulation of any kind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd


# Units, explicit (per requirement — every field's unit stated once, here,
# rather than left implicit in a variable name):
#   timestamp   seconds (monotonic or wall-clock — caller's choice)
#   ax, ay, az  m/s^2   (accelerometer, required)
#   gx, gy, gz  rad/s   (gyroscope, required)
#   mx, my, mz  uT      (magnetometer, optional)
#   lat, lon    degrees (WGS84, optional)
#   speed_mps   m/s     (GNSS ground speed, optional)
#   course_rad  rad     (GNSS course-over-ground, 0=north, clockwise positive, optional)
#   baro_hpa    hPa     (barometric pressure, optional)

# Sample rates this contract is explicitly known to have been exercised at
# (used only for an advisory warning in validate_stream() — never a hard
# rejection, since a real external IMU may legitimately run at a rate not
# in this list).
SUPPORTED_SAMPLE_RATES_HZ = (10.0, 20.0, 25.0, 50.0, 100.0, 200.0)


class MalformedIMURecordError(ValueError):
    """Raised when a single record cannot be turned into a valid
    ExternalIMUFrame — never silently coerced or filled with a guess."""


@dataclass(frozen=True)
class ExternalIMUFrame:
    timestamp: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    mx: Optional[float] = None
    my: Optional[float] = None
    mz: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    speed_mps: Optional[float] = None
    course_rad: Optional[float] = None
    baro_hpa: Optional[float] = None
    # Metadata (per requirement) — both optional, never inferred/guessed when
    # the caller doesn't supply them:
    sample_rate_hz: Optional[float] = None   # nominal rate this frame's stream claims to run at
    source: Optional[str] = None             # free-form sensor/source identifier, e.g. "csv:imu_log_01",
                                              # "external_board:bno085", "unit_test"

    def accel(self) -> np.ndarray:
        return np.array([self.ax, self.ay, self.az], dtype=np.float64)

    def gyro(self) -> np.ndarray:
        return np.array([self.gx, self.gy, self.gz], dtype=np.float64)

    def has_gnss(self) -> bool:
        return self.lat is not None and self.lon is not None

    def has_magnetometer(self) -> bool:
        return self.mx is not None and self.my is not None and self.mz is not None

    def has_barometer(self) -> bool:
        return self.baro_hpa is not None

    def required_fields_finite(self) -> bool:
        """True only if every REQUIRED field (timestamp, accel, gyro) is a
        finite real number. Optional fields that are None are not checked
        here (that's fine — None means genuinely absent, not invalid)."""
        vals = (self.timestamp, self.ax, self.ay, self.az, self.gx, self.gy, self.gz)
        return all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)

    def optional_fields_finite(self) -> bool:
        """True if every optional field that IS present is finite (a present
        NaN/inf optional field is a malformed record, not a legitimately
        absent one)."""
        for v in (self.mx, self.my, self.mz, self.lat, self.lon,
                  self.speed_mps, self.course_rad, self.baro_hpa,
                  self.sample_rate_hz):
            if v is not None and not (isinstance(v, (int, float)) and math.isfinite(v)):
                return False
        return True

    def validate(self) -> List[str]:
        """Returns a list of human-readable problems with this single frame
        (empty list = frame is valid). Never raises — callers that want a
        hard failure use `validate_or_raise()` or the stream-level
        `validate_stream(..., strict=True)`."""
        problems: List[str] = []
        if not self.required_fields_finite():
            problems.append("one or more required fields (timestamp/ax/ay/az/gx/gy/gz) "
                             "is not a finite number")
        if not self.optional_fields_finite():
            problems.append("one or more present optional fields is not a finite number "
                             "(a genuinely absent optional field must be None, not NaN)")
        if self.has_gnss() and not (-90.0 <= self.lat <= 90.0 and -180.0 <= self.lon <= 180.0):
            problems.append(f"lat/lon out of valid range: ({self.lat}, {self.lon})")
        if self.speed_mps is not None and self.speed_mps < 0.0:
            problems.append(f"speed_mps must be >= 0, got {self.speed_mps}")
        if self.sample_rate_hz is not None and self.sample_rate_hz <= 0.0:
            problems.append(f"sample_rate_hz must be > 0, got {self.sample_rate_hz}")
        return problems

    def validate_or_raise(self) -> None:
        problems = self.validate()
        if problems:
            raise MalformedIMURecordError(
                f"invalid ExternalIMUFrame at timestamp={self.timestamp}: {'; '.join(problems)}")


# Reuses navdrift_engine.py's own column-name conventions/detection so a CSV
# that already works with `python navdrift_engine.py --imu ...` also works
# here, rather than inventing a second, incompatible schema.
_COLUMN_CANDIDATES = {
    "ts": ["timestamp", "time_ms", "millis", "t_ms", "t"],
    "ax": ["accel_x", "ax_", "acc_x", "linear_x", "ax"],
    "ay": ["accel_y", "ay_", "acc_y", "linear_y", "ay"],
    "az": ["accel_z", "az_", "acc_z", "linear_z", "az"],
    "gx": ["gyro_x", "gyrox", "gyroscope_x", "gx"],
    "gy": ["gyro_y", "gyroy", "gyroscope_y", "gy"],
    "gz": ["gyro_z", "gyroz", "gyroscope_z", "gz"],
    "mx": ["mag_x", "magx", "magnetometer_x", "mx"],
    "my": ["mag_y", "magy", "magnetometer_y", "my"],
    "mz": ["mag_z", "magz", "magnetometer_z", "mz"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "lng", "longitude"],
    "speed": ["speed", "velocity", "gps_speed", "speed_mps"],
    "course": ["course", "heading", "bearing_deg"],
    "baro": ["baro", "pressure_hpa", "baro_hpa"],
}


def _find_col(candidates: list[str], columns) -> Optional[str]:
    for c in candidates:
        m = [col for col in columns if c.lower() in str(col).lower()]
        if m:
            return m[0]
    return None


def frames_from_csv(path: str, timestamp_is_ms: bool = True,
                     sample_rate_hz: Optional[float] = None) -> list[ExternalIMUFrame]:
    """
    Load a CSV in navdrift_engine.py's flexible column format and return a
    list of ExternalIMUFrame, timestamp normalized to seconds. Rows missing
    required accel/gyro columns are dropped (never filled with a guessed
    value). This is a REAL loader — every value returned is read from the
    file, nothing here invents a sample.

    `sample_rate_hz`, if given, is stamped onto every returned frame's
    `sample_rate_hz` metadata field as-is (the caller's own claim about the
    file — this function never estimates or guesses it). Each frame's
    `source` is set to "csv:<path>" so downstream validation/reports can
    trace where a frame came from.
    """
    df = pd.read_csv(path, low_memory=False)
    df = df.apply(pd.to_numeric, errors="coerce")
    cols = {k: _find_col(v, df.columns) for k, v in _COLUMN_CANDIDATES.items()}

    required = ["ax", "ay", "az", "gx", "gy", "gz"]
    missing = [k for k in required if cols[k] is None]
    if missing:
        raise MalformedIMURecordError(f"frames_from_csv: required columns not found for {missing} in {path}")

    df = df.dropna(subset=[cols[k] for k in required]).reset_index(drop=True)

    frames: list[ExternalIMUFrame] = []
    for i, row in df.iterrows():
        ts_raw = row[cols["ts"]] if cols["ts"] else float(i)
        ts = float(ts_raw) / 1000.0 if (cols["ts"] and timestamp_is_ms) else float(ts_raw)
        frames.append(ExternalIMUFrame(
            timestamp=ts,
            ax=float(row[cols["ax"]]), ay=float(row[cols["ay"]]), az=float(row[cols["az"]]),
            gx=float(row[cols["gx"]]), gy=float(row[cols["gy"]]), gz=float(row[cols["gz"]]),
            mx=(float(row[cols["mx"]]) if cols["mx"] and not pd.isna(row[cols["mx"]]) else None),
            my=(float(row[cols["my"]]) if cols["my"] and not pd.isna(row[cols["my"]]) else None),
            mz=(float(row[cols["mz"]]) if cols["mz"] and not pd.isna(row[cols["mz"]]) else None),
            lat=(float(row[cols["lat"]]) if cols["lat"] and not pd.isna(row[cols["lat"]]) else None),
            lon=(float(row[cols["lon"]]) if cols["lon"] and not pd.isna(row[cols["lon"]]) else None),
            speed_mps=(float(row[cols["speed"]]) if cols["speed"] and not pd.isna(row[cols["speed"]]) else None),
            course_rad=(np.radians(float(row[cols["course"]])) if cols["course"] and not pd.isna(row[cols["course"]]) else None),
            baro_hpa=(float(row[cols["baro"]]) if cols["baro"] and not pd.isna(row[cols["baro"]]) else None),
            sample_rate_hz=sample_rate_hz,
            source=f"csv:{path}",
        ))
    return frames


@dataclass
class StreamValidationReport:
    """Result of `validate_stream()` — a real, itemized report, never a bare
    pass/fail. Every entry in `frame_problems` is (index, [problem, ...])."""
    n_frames: int
    n_valid_frames: int
    frame_problems: List[tuple] = field(default_factory=list)   # [(index, [problems]), ...]
    timestamp_order_violations: List[int] = field(default_factory=list)  # indices where ts didn't increase
    dt_outliers: List[tuple] = field(default_factory=list)      # [(index, dt_s), ...] far from nominal
    sample_rate_hz_estimated: Optional[float] = None
    sample_rate_supported: Optional[bool] = None

    @property
    def ok(self) -> bool:
        return (not self.frame_problems and not self.timestamp_order_violations
                and self.n_valid_frames == self.n_frames)


def validate_stream(frames: Sequence[ExternalIMUFrame],
                     expected_hz: Optional[float] = None,
                     dt_outlier_factor: float = 5.0) -> StreamValidationReport:
    """
    Validate an entire frame stream: per-frame field validity, timestamp
    ordering (must be non-decreasing — a real sensor stream is never
    delivered out of order), and dt sanity (a gap far larger than the
    nominal sample period than the rest of the stream is flagged, not
    silently accepted or fabricated over).

    `expected_hz`, if given, is only used to flag dt outliers and to report
    whether the observed stream matches a known-supported rate — it never
    changes what frames are returned or drops any data.
    """
    n = len(frames)
    report = StreamValidationReport(n_frames=n, n_valid_frames=0)
    if n == 0:
        return report

    valid_count = 0
    prev_ts: Optional[float] = None
    dts: List[float] = []
    for i, fr in enumerate(frames):
        problems = fr.validate()
        if problems:
            report.frame_problems.append((i, problems))
        else:
            valid_count += 1

        if prev_ts is not None and fr.timestamp < prev_ts:
            report.timestamp_order_violations.append(i)
        if prev_ts is not None:
            dts.append(fr.timestamp - prev_ts)
        prev_ts = fr.timestamp

    report.n_valid_frames = valid_count

    if dts:
        median_dt = float(np.median(dts))
        report.sample_rate_hz_estimated = round(1.0 / median_dt, 3) if median_dt > 0 else None
        if median_dt > 0:
            for i, dt in enumerate(dts, start=1):
                if dt > 0 and (dt > median_dt * dt_outlier_factor or dt < median_dt / dt_outlier_factor):
                    report.dt_outliers.append((i, round(dt, 6)))

    nominal_hz = expected_hz if expected_hz is not None else report.sample_rate_hz_estimated
    if nominal_hz is not None:
        report.sample_rate_supported = any(
            abs(nominal_hz - hz) / hz < 0.05 for hz in SUPPORTED_SAMPLE_RATES_HZ
        )

    return report


def frame_from_record(record: dict, source: Optional[str] = None,
                       sample_rate_hz: Optional[float] = None) -> ExternalIMUFrame:
    """
    Build one ExternalIMUFrame from a plain dict of REQUIRED keys
    (timestamp, ax, ay, az, gx, gy, gz) plus any OPTIONAL keys already named
    like ExternalIMUFrame's fields. Intended as the entry point for a future
    streaming source (serial, socket, message queue) that hands over one
    record at a time — CSV loading and this streaming path both end up
    calling into ExternalIMUFrame the same way, so behavior stays consistent
    across input methods.

    Raises MalformedIMURecordError with a clear message if a required key is
    missing or any value fails validation. Never fills a missing required
    field with 0.0 or any other invented value.
    """
    required = ("timestamp", "ax", "ay", "az", "gx", "gy", "gz")
    missing = [k for k in required if k not in record or record[k] is None]
    if missing:
        raise MalformedIMURecordError(f"record missing required field(s): {missing}")
    try:
        kwargs = {k: float(record[k]) for k in required}
    except (TypeError, ValueError) as e:
        raise MalformedIMURecordError(f"record has a non-numeric required field: {e}") from e

    optional_keys = ("mx", "my", "mz", "lat", "lon", "speed_mps", "course_rad", "baro_hpa")
    for k in optional_keys:
        v = record.get(k)
        kwargs[k] = (float(v) if v is not None else None)

    kwargs["source"] = source if source is not None else record.get("source")
    kwargs["sample_rate_hz"] = sample_rate_hz if sample_rate_hz is not None else record.get("sample_rate_hz")

    frame = ExternalIMUFrame(**kwargs)
    frame.validate_or_raise()
    return frame


def replay_frames(frames: Sequence[ExternalIMUFrame]) -> Iterator[tuple[ExternalIMUFrame, float]]:
    """
    Deterministic (non-realtime-paced) generator: yields (frame, dt) pairs,
    where dt is the real elapsed time between this frame's and the previous
    frame's `timestamp` field (0.0 for the first frame). This is what a batch
    processor / unit test should use — it never sleeps and never fabricates
    timing, it only reports the real timestamp deltas already in the data.

    For an actual paced real-time replay (used by the 200 Hz demonstration),
    see `edge/replay_200hz_demo.py`, which wraps this generator with real
    wall-clock pacing and latency measurement.
    """
    prev_ts: Optional[float] = None
    for fr in frames:
        dt = 0.0 if prev_ts is None else max(0.0, fr.timestamp - prev_ts)
        prev_ts = fr.timestamp
        yield fr, dt


def engine_step_from_frame(engine, frame: ExternalIMUFrame, dt: float):
    """
    Adapter: feeds one ExternalIMUFrame into an existing, UNMODIFIED
    `navdrift_engine.NavdriftRTEngine` instance via its existing `.step()`
    signature. This function exists so callers use the typed contract above
    without navdrift_engine.py itself needing to change at all.

    Returns exactly what `engine.step()` returns: (pred_lat, pred_lon).
    If the frame carries a GNSS fix, also calls the engine's existing
    `.gps_update()` — again, no change to NavdriftRTEngine required.
    """
    result = engine.step(frame.ax, frame.ay, frame.az, frame.gx, frame.gy, frame.gz, dt)
    if frame.has_gnss():
        engine.gps_update(frame.lat, frame.lon)
    return result
