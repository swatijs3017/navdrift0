"""
data/iovnbd.py — real IO-VNBD parser/adapter.

Reads the ACTUAL IO-VNBD file layout as shipped in the two official ZIP
archives from https://github.com/onyekpeu/IO-VNBD:

    <root>/Categorised IOVNB Dataset/<driver code>/<run>/S-<run>.csv
    <root>/Categorised IOVNB Dataset/<driver code>/<run>/V-<run>.csv

This is NOT the same layout data/loader.py's IOVNBDParser assumes
(run_NNN/{imu.csv,odom.csv,gps.csv}). That class is left untouched. This
module is a separate, additive parser that produces the same downstream
data/loader.py.Sequence objects (IMUSample / OdometrySample / GPSSample),
so synchronize_sequence(), simulate_gnss_outages(), compute_se2_poses(),
compute_pose_deltas(), NormStats, DRDataset, and build_dataloaders() are
all reused unmodified.

Design decisions made here, and why, documented inline rather than hidden:

- Timestamp: the S-file's "TIME SINCE START (ms)" column resets mid-file
  in real data (confirmed by direct inspection, see
  results/iovnbd_inspection_report.md). The DATE column is used instead
  and converted to elapsed seconds from the first sample. Any reset in
  the raw ms counter is detected and logged, never silently trusted.
- Linear speed: IO-VNBD's V-file gives four wheel speeds in rad/sec with
  no published wheel radius in these CSVs, so a physical radius is never
  assumed. "Velocity (km/hr)" from the V-file's own reference GPS/INS is
  used instead, divided by 3.6 to get m/s. This is explicit, not a wheel
  odometry substitute in disguise: OdometrySample.speed here is really a
  reference-speed channel, and callers should not conflate it with wheel
  pulse-derived odometry.
- Gyroscope axes: S-file gyroscope columns are labelled "Yaw", "Pitch",
  "Roll" (rates), not "X"/"Y"/"Z". They are passed through positionally
  into IMUSample.gyro_x/gyro_y/gyro_z in that column order because that
  is the order the columns appear in the file, but this is NOT a claim
  that gyro_x is physically the phone's X axis. Any code that depends on
  a specific body-frame axis convention must treat this as an open item
  (see results/iovnbd_inspection_report.md section 6) rather than assume
  it silently.
- No file is ever forced into the old run_NNN/{imu,odom,gps}.csv layout.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from data.loader import GPSSample, IMUSample, OdometrySample, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column layouts observed by direct inspection of real IO-VNBD files.
# These are the EXACT header strings found in the shipped CSVs (with the
# mangled degree-sign / superscript-2 bytes normalised away below).
# ---------------------------------------------------------------------------

S_COLS = {
    "gps_lat": "GPS LATITUDE (degrees)",
    "gps_lon": "GPS LONGITUDE (degrees)",
    "gps_alt": "GPS ALTITUDE (m)",
    "gps_speed_kmh": "GPS SPEED (Kmh)",
    "gps_accuracy": "GPS ACCURACY (m)",
    "gps_orientation": "GPS ORIENTATION (deg)",
    "gps_sats": "GPS SATELLITES IN RANGE",
    "time_since_start_ms": "TIME SINCE START (ms)",
    "date": "DATE (YYYY-MO-DD HH-MI-SS_SSS)",
    "accel_x": "ACCELEROMETER X (m/s2)",
    "accel_y": "ACCELEROMETER Y (m/s2)",
    "accel_z": "ACCELEROMETER Z (m/s2)",
    "gravity_x": "GRAVITY X (m/s2)",
    "gravity_y": "GRAVITY Y (m/s2)",
    "gravity_z": "GRAVITY Z (m/s2)",
    "gyro_yaw": "GYROSCOPE Yaw (rad/s)",
    "gyro_pitch": "GYROSCOPE Pitch (rad/s)",
    "gyro_roll": "GYROSCOPE Roll (rad/s)",
    "mag_x": "MAGNETIC FIELD X (uT)",
    "mag_y": "MAGNETIC FIELD Y (uT)",
    "mag_z": "MAGNETIC FIELD Z (uT)",
    "orient_yaw": "ORIENTATION (Yaw) (deg)",
    "orient_pitch": "ORIENTATION (Pitch) (deg)",
    "orient_roll": "ORIENTATION (Roll ) (deg)",
}

V_COLS = {
    "gps_sats": "No of GPS Satellites Available",
    "time_of_day_s": "Time Since Start of Day (seconds)",
    "gps_lat": "Latitude (degrees)",
    "gps_lon": "Longitude (degrees)",
    "velocity_kmh": "Velocity (km/hr)",
    "heading_deg": "Heading (degrees)",
    "height_km": "Height (km)",
    "vvel_kmh": "Vertical velocity (km/hr)",
    "sample_period_s": "Sample period (seconds)",
    "steer_angle_deg": "Steering Angle (degrees)",
    "wheel_fl": "Wheel Speed Front Left (rad/sec)",
    "wheel_fr": "Wheel Speed Front Right (rad/sec)",
    "wheel_rl": "Wheel Speed Rear Left (rad/sec)",
    "wheel_rr": "Wheel Speed Rear Right (rad/sec)",
    "yaw_rate_dps": "Yaw Rate (deg/sec)",
    "indicated_speed_kmh": "Indicated Vehicle Speed (km/hr)",
    "long_accel_g": "Indicated Longitudinal Acceleration (g)",
    "lat_accel_g": "Indicated Lateral Acceleration (g)",
}


def _normalize_header(h: str) -> str:
    """IO-VNBD's shipped CSVs use non-ASCII degree/superscript characters
    that decode inconsistently across encodings. Normalise header text for
    matching only; never touch the numeric row data."""
    h = h.strip()
    h = h.replace("�", "")     # mojibake replacement char
    h = h.replace("m/s²", "m/s2").replace("m/s2", "m/s2")
    h = h.replace("°", "deg").replace("deg)", "deg)")
    h = h.replace("μ", "u")    # micro sign -> 'u' (uT)
    h = " ".join(h.split())         # collapse whitespace
    return h


def _name_only(h: str) -> str:
    """Strip the parenthetical unit suffix and any mangled/non-ASCII bytes,
    keeping only the alnum characters of the field NAME (never the unit
    text, which is where the real files' encoding corruption lives, e.g.
    'ACCELEROMETER X (m/s�)'). Used only to LOCATE a column by name;
    the numeric values themselves are always read from the untouched row
    data, never reconstructed from this normalisation."""
    h = h.split("(")[0]
    return "".join(ch for ch in h.lower() if ch.isalnum())


def _find_col(header: List[str], wanted: str) -> int:
    norm = [_normalize_header(h) for h in header]
    wanted_n = _normalize_header(wanted)
    if wanted_n in norm:
        return norm.index(wanted_n)
    # exact match failed (near-certainly because of the source files' own
    # mangled unit text, e.g. the degree sign or the m/s^2 superscript) —
    # fall back to matching on the field NAME only, ignoring units.
    key = _name_only(wanted)
    for i, h in enumerate(header):
        if _name_only(h) == key:
            return i
    raise KeyError(f"Column not found: {wanted!r} (normalised {wanted_n!r}, "
                    f"name-key {key!r}) in header {header}")


@dataclass
class RunMeta:
    """Metadata about one parsed S/V run pair, kept separate from the
    numeric Sequence so callers can inspect data-quality flags without
    re-parsing."""
    name: str
    driver_code: str
    s_path: str
    v_path: str
    n_rows_s: int
    n_rows_v: int
    n_rows_used: int          # min(n_rows_s, n_rows_v) after trimming
    row_count_mismatch: bool
    duration_s: float
    time_reset_detected: bool
    n_time_resets: int
    n_nonfinite_dropped: int
    sample_rate_hz_estimate: float


def _iter_zip_run_pairs(zip_path: Path,
                         only_categorised: bool = True
                         ) -> List[Tuple[str, str, str, str]]:
    """Return (run_name, driver_code, s_entry, v_entry) tuples for every
    S/V run pair found directly inside the zip, without extracting
    anything. Only the "Categorised" subtree is used by default since the
    "Uncategorised" subtree appears (by filename) to mirror the same
    runs — see results/iovnbd_inspection_report.md section 2."""
    pairs = []
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        s_files = [n for n in names if n.lower().endswith(".csv")
                   and Path(n).name.startswith("S-")]
        v_files = [n for n in names if n.lower().endswith(".csv")
                   and Path(n).name.startswith("V-")]
        if only_categorised:
            s_files = [n for n in s_files if "/Categorised" in n]
            v_files = [n for n in v_files if "/Categorised" in n]
        v_by_dir = {}
        for v in v_files:
            v_by_dir.setdefault(str(Path(v).parent), []).append(v)
        for s in sorted(s_files):
            d = str(Path(s).parent)
            candidates = v_by_dir.get(d, [])
            if not candidates:
                logger.warning("No V-file match for %s, skipping", s)
                continue
            v = candidates[0]
            parts = Path(s).parts
            driver_code = parts[2] if len(parts) > 2 else "unknown"
            run_name = Path(s).stem[2:]  # strip "S-"
            pairs.append((run_name, driver_code, s, v))
    return pairs


def _parse_date(s: str) -> float:
    """Parse IO-VNBD's 'YYYY-MO-DD HH-MI-SS_SSS' style DATE field
    (actual observed separator is ':' before milliseconds, e.g.
    '2019-09-04 20:03:03:493') into a POSIX timestamp in seconds."""
    s = s.strip()
    date_part, time_part = s.split(" ")
    y, mo, d = (int(x) for x in date_part.split("-"))
    hh, mm, ss, ms = (int(x) for x in time_part.split(":"))
    dt = datetime(y, mo, d, hh, mm, ss, ms * 1000)
    return dt.timestamp()


def _read_csv_from_zip(z: zipfile.ZipFile, entry: str) -> Tuple[List[str], List[List[str]]]:
    with z.open(entry) as f:
        text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
        reader = csv.reader(text)
        header = next(reader)
        rows = list(reader)
    return header, rows


def parse_s_file(header: List[str], rows: List[List[str]]
                  ) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Parse a smartphone S-<run>.csv.

    Returns:
        t_s:        (N,) elapsed seconds from first sample, from DATE
        imu:        (N, 6) [accel_x, accel_y, accel_z, gyro_yaw, gyro_pitch, gyro_roll]
        n_resets:   number of TIME SINCE START(ms) resets detected (diagnostic only)
        n_dropped:  number of rows dropped for non-finite values
    """
    idx = {k: _find_col(header, v) for k, v in S_COLS.items()}

    date_col = idx["date"]
    ms_col = idx["time_since_start_ms"]

    # Detect resets in the raw ms counter (diagnostic; DATE is authoritative).
    n_resets = 0
    prev_ms = None
    for r in rows:
        try:
            ms = int(r[ms_col])
        except ValueError:
            continue
        if prev_ms is not None and ms < prev_ms:
            n_resets += 1
        prev_ms = ms

    t_abs = np.array([_parse_date(r[date_col]) for r in rows], dtype=np.float64)
    t_s = t_abs - t_abs[0]

    cols = ["accel_x", "accel_y", "accel_z", "gyro_yaw", "gyro_pitch", "gyro_roll"]
    imu = np.array([[float(r[idx[c]]) for c in cols] for r in rows], dtype=np.float64)

    finite_mask = np.isfinite(imu).all(axis=1) & np.isfinite(t_s)
    n_dropped = int((~finite_mask).sum())
    if n_dropped:
        t_s = t_s[finite_mask]
        imu = imu[finite_mask]

    return t_s.astype(np.float64), imu.astype(np.float32), n_resets, n_dropped


def parse_v_file(header: List[str], rows: List[List[str]]
                  ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Parse a vehicle V-<run>.csv.

    Returns:
        t_s:     (N,) elapsed seconds from first sample, from
                 'Time Since Start of Day (seconds)' (already clean/monotonic
                 per direct inspection; no reconstruction needed)
        vals:    (N, 8) [lat, lon, alt_m, heading_deg, speed_mps,
                          steer_rad, yaw_rate_rad_s, long_accel_mps2]
        n_dropped: rows dropped for non-finite values
    """
    idx = {k: _find_col(header, v) for k, v in V_COLS.items()}
    t_col = idx["time_of_day_s"]

    t_s_raw = np.array([float(r[t_col]) for r in rows], dtype=np.float64)
    t_s = t_s_raw - t_s_raw[0]

    def col(name, r):
        return float(r[idx[name]])

    vals = []
    for r in rows:
        lat = col("gps_lat", r)
        lon = col("gps_lon", r)
        alt_m = col("height_km", r) * 1000.0
        heading_deg = col("heading_deg", r)
        speed_mps = col("velocity_kmh", r) / 3.6     # NEVER wheel radius, see module docstring
        steer_rad = math.radians(col("steer_angle_deg", r))
        yaw_rate_rad_s = math.radians(col("yaw_rate_dps", r))
        long_accel_mps2 = col("long_accel_g", r) * 9.80665
        vals.append([lat, lon, alt_m, heading_deg, speed_mps,
                     steer_rad, yaw_rate_rad_s, long_accel_mps2])
    vals = np.array(vals, dtype=np.float64)

    finite_mask = np.isfinite(vals).all(axis=1) & np.isfinite(t_s)
    n_dropped = int((~finite_mask).sum())
    if n_dropped:
        t_s = t_s[finite_mask]
        vals = vals[finite_mask]

    return t_s.astype(np.float64), vals.astype(np.float32), n_dropped


def load_run_pair(zip_path: Union[str, Path],
                   s_entry: str, v_entry: str,
                   run_name: str, driver_code: str
                   ) -> Tuple[Optional[Sequence], RunMeta]:
    """Parse one S/V run pair directly out of the zip (no extraction) and
    return a data/loader.py-compatible Sequence, plus its RunMeta.

    Returns (None, meta) if the run cannot be honestly parsed (e.g. all
    rows non-finite) rather than fabricating a sequence.
    """
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        s_header, s_rows = _read_csv_from_zip(z, s_entry)
        v_header, v_rows = _read_csv_from_zip(z, v_entry)

    s_t, s_imu, n_resets, n_drop_s = parse_s_file(s_header, s_rows)
    v_t, v_vals, n_drop_v = parse_v_file(v_header, v_rows)

    n_rows_s, n_rows_v = len(s_t), len(v_t)
    n_used = min(n_rows_s, n_rows_v)
    mismatch = n_rows_s != n_rows_v

    if n_used < 2:
        meta = RunMeta(
            name=run_name, driver_code=driver_code,
            s_path=s_entry, v_path=v_entry,
            n_rows_s=n_rows_s, n_rows_v=n_rows_v, n_rows_used=0,
            row_count_mismatch=mismatch, duration_s=0.0,
            time_reset_detected=n_resets > 0, n_time_resets=n_resets,
            n_nonfinite_dropped=n_drop_s + n_drop_v,
            sample_rate_hz_estimate=0.0,
        )
        logger.warning("Run %s has too few usable rows (%d), skipping", run_name, n_used)
        return None, meta

    # Row-index-aligned trim (both files are 10 Hz recordings of the same
    # drive; per results/iovnbd_inspection_report.md most pairs already
    # match exactly, the few that don't differ by a handful of rows at
    # most, so trimming to the shorter length is the documented, disclosed
    # handling rather than an interpolated re-alignment).
    s_t = s_t[:n_used]
    s_imu = s_imu[:n_used]
    v_t = v_t[:n_used]
    v_vals = v_vals[:n_used]

    dt_med = float(np.median(np.diff(s_t))) if n_used > 1 else 0.0
    hz_est = (1.0 / dt_med) if dt_med > 0 else 0.0

    imu_samples = [
        IMUSample(timestamp=float(s_t[i]),
                  accel_x=float(s_imu[i, 0]), accel_y=float(s_imu[i, 1]), accel_z=float(s_imu[i, 2]),
                  gyro_x=float(s_imu[i, 3]), gyro_y=float(s_imu[i, 4]), gyro_z=float(s_imu[i, 5]))
        for i in range(n_used)
    ]
    # OdometrySample.speed here is the V-file reference speed (Velocity/3.6),
    # not wheel-pulse odometry — see module docstring.
    odom_samples = [
        OdometrySample(timestamp=float(v_t[i]), speed=float(v_vals[i, 4]),
                        steer_angle=float(v_vals[i, 5]))
        for i in range(n_used)
    ]
    gps_samples = [
        GPSSample(timestamp=float(v_t[i]), latitude=float(v_vals[i, 0]),
                  longitude=float(v_vals[i, 1]), altitude=float(v_vals[i, 2]),
                  heading=float(v_vals[i, 3]), speed=float(v_vals[i, 4]))
        for i in range(n_used)
    ]

    seq = Sequence(name=run_name, imu=imu_samples, odom=odom_samples, gps=gps_samples)
    meta = RunMeta(
        name=run_name, driver_code=driver_code,
        s_path=s_entry, v_path=v_entry,
        n_rows_s=n_rows_s, n_rows_v=n_rows_v, n_rows_used=n_used,
        row_count_mismatch=mismatch,
        duration_s=float(s_t[-1] - s_t[0]),
        time_reset_detected=n_resets > 0, n_time_resets=n_resets,
        n_nonfinite_dropped=n_drop_s + n_drop_v,
        sample_rate_hz_estimate=hz_est,
    )
    return seq, meta


def load_iovnbd_dataset(sync_zip_path: Union[str, Path],
                         only_categorised: bool = True,
                         max_runs: Optional[int] = None
                         ) -> Tuple[List[Sequence], List[RunMeta]]:
    """Parse every real S/V run pair in the Synchronised IO-VNBD zip.

    Reads directly from the zip archive; the zip itself is opened
    read-only and is never modified or re-extracted onto disk by this
    function.
    """
    sync_zip_path = Path(sync_zip_path)
    pairs = _iter_zip_run_pairs(sync_zip_path, only_categorised=only_categorised)
    if max_runs is not None:
        pairs = pairs[:max_runs]

    sequences: List[Sequence] = []
    metas: List[RunMeta] = []
    for run_name, driver_code, s_entry, v_entry in pairs:
        seq, meta = load_run_pair(sync_zip_path, s_entry, v_entry, run_name, driver_code)
        metas.append(meta)
        if seq is not None:
            sequences.append(seq)

    logger.info("load_iovnbd_dataset: %d/%d runs parsed successfully",
                len(sequences), len(pairs))
    return sequences, metas


def metas_to_json(metas: List[RunMeta]) -> list:
    return [m.__dict__ for m in metas]


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Parse real IO-VNBD Synchronised dataset and report.")
    p.add_argument("--zip", required=True, help="Path to 'Synchronised V abd S datasets.zip'")
    p.add_argument("--max_runs", type=int, default=None)
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    sequences, metas = load_iovnbd_dataset(args.zip, max_runs=args.max_runs)
    print(f"\nParsed {len(sequences)}/{len(metas)} run pairs successfully.\n")
    for m in metas:
        status = "OK"
        if m.n_rows_used == 0:
            status = "FAILED"
        elif m.row_count_mismatch:
            status = "OK (row mismatch trimmed)"
        print(f"  [{status:26s}] {m.name:12s} driver={m.driver_code:20s} "
              f"rows_s={m.n_rows_s:7d} rows_v={m.n_rows_v:7d} used={m.n_rows_used:7d} "
              f"dur={m.duration_s:8.1f}s  hz~={m.sample_rate_hz_estimate:.2f}  "
              f"resets={m.n_time_resets}  dropped_nonfinite={m.n_nonfinite_dropped}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(metas_to_json(metas), f, indent=2)
        print(f"\nWritten to {args.out_json}")
