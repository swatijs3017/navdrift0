"""
IO-VNBD dataset loader and preprocessor.

IO-VNBD: Inertial and Odometry benchmark dataset for ground vehicle positioning
Source: https://github.com/onyekpeu/IO-VNBD

This module handles:
- Downloading and parsing the raw dataset
- Synchronizing IMU, odometry, and GPS streams
- Simulating GNSS outages by masking GPS segments
- Producing training-ready tensors
"""

import os
import csv
import math
import random
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Raw data structures
# ---------------------------------------------------------------------------

@dataclass
class IMUSample:
    """One timestep of raw IMU data."""
    timestamp: float          # seconds
    accel_x: float            # m/s^2
    accel_y: float            # m/s^2
    accel_z: float            # m/s^2
    gyro_x: float             # rad/s
    gyro_y: float             # rad/s
    gyro_z: float             # rad/s


@dataclass
class OdometrySample:
    """One timestep of wheel odometry data."""
    timestamp: float
    speed: float              # m/s (derived from pulse count)
    steer_angle: float        # radians


@dataclass
class GPSSample:
    """One timestep of GPS/NavIC fix."""
    timestamp: float
    latitude: float           # degrees
    longitude: float          # degrees
    altitude: float           # meters
    heading: float            # degrees, 0=North
    speed: float              # m/s


@dataclass
class Sequence:
    """A complete synchronized vehicle sequence."""
    name: str
    imu: List[IMUSample]     = field(default_factory=list)
    odom: List[OdometrySample] = field(default_factory=list)
    gps: List[GPSSample]     = field(default_factory=list)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

EARTH_RADIUS_M = 6_371_000.0


def latlon_to_local_xy(lat: float, lon: float,
                        ref_lat: float, ref_lon: float) -> Tuple[float, float]:
    """
    Convert latitude/longitude to local Cartesian (x=East, y=North) in metres,
    using an equirectangular approximation valid for short distances (<~10 km).
    """
    dlat = math.radians(lat - ref_lat)
    dlon = math.radians(lon - ref_lon)
    x = dlon * EARTH_RADIUS_M * math.cos(math.radians(ref_lat))
    y = dlat * EARTH_RADIUS_M
    return x, y


def heading_to_rad(heading_deg: float) -> float:
    """Convert compass heading (0=North, CW) to math angle (0=East, CCW)."""
    return math.radians(90.0 - heading_deg)


# ---------------------------------------------------------------------------
# Parser: handles the IO-VNBD CSV layout
# IO-VNBD stores each run as a directory with:
#   imu.csv    — timestamp, ax, ay, az, gx, gy, gz
#   odom.csv   — timestamp, speed, steer_angle
#   gps.csv    — timestamp, lat, lon, alt, heading, speed
# If the dataset uses a different layout the user can subclass BaseParser.
# ---------------------------------------------------------------------------

class IOVNBDParser:
    """
    Parses one IO-VNBD run directory into a Sequence object.

    The dataset ships as a zip; after extraction you get:
      <root>/
        run_001/
          imu.csv
          odom.csv
          gps.csv
        run_002/
          ...

    If the actual column names differ, pass a col_map override dict.
    """

    DEFAULT_IMU_COLS  = {"t": "timestamp",
                         "ax": "accel_x", "ay": "accel_y", "az": "accel_z",
                         "gx": "gyro_x",  "gy": "gyro_y",  "gz": "gyro_z"}
    DEFAULT_ODOM_COLS = {"t": "timestamp", "v": "speed", "delta": "steer_angle"}
    DEFAULT_GPS_COLS  = {"t": "timestamp",
                         "lat": "latitude", "lon": "longitude",
                         "alt": "altitude", "hdg": "heading", "v": "speed"}

    def __init__(self,
                 imu_col_map:  Optional[Dict[str, str]] = None,
                 odom_col_map: Optional[Dict[str, str]] = None,
                 gps_col_map:  Optional[Dict[str, str]] = None):
        self.imu_map  = imu_col_map  or self.DEFAULT_IMU_COLS
        self.odom_map = odom_col_map or self.DEFAULT_ODOM_COLS
        self.gps_map  = gps_col_map  or self.DEFAULT_GPS_COLS

    def _read_csv(self, path: Path) -> List[Dict[str, str]]:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _col(self, row: Dict[str, str], col_map: Dict[str, str], key: str) -> float:
        """Resolve a logical field name through the column map and return float."""
        csv_key = col_map.get(key, key)
        return float(row[csv_key])

    def parse_run(self, run_dir: Path) -> Sequence:
        """Parse a single run directory."""
        seq = Sequence(name=run_dir.name)

        # -- IMU --
        imu_path = run_dir / "imu.csv"
        if imu_path.exists():
            for row in self._read_csv(imu_path):
                m = self.imu_map
                seq.imu.append(IMUSample(
                    timestamp = self._col(row, m, "t"),
                    accel_x   = self._col(row, m, "ax"),
                    accel_y   = self._col(row, m, "ay"),
                    accel_z   = self._col(row, m, "az"),
                    gyro_x    = self._col(row, m, "gx"),
                    gyro_y    = self._col(row, m, "gy"),
                    gyro_z    = self._col(row, m, "gz"),
                ))
        else:
            logger.warning("No imu.csv found in %s", run_dir)

        # -- Odometry --
        odom_path = run_dir / "odom.csv"
        if odom_path.exists():
            for row in self._read_csv(odom_path):
                m = self.odom_map
                seq.odom.append(OdometrySample(
                    timestamp   = self._col(row, m, "t"),
                    speed       = self._col(row, m, "v"),
                    steer_angle = self._col(row, m, "delta"),
                ))
        else:
            logger.warning("No odom.csv found in %s", run_dir)

        # -- GPS --
        gps_path = run_dir / "gps.csv"
        if gps_path.exists():
            for row in self._read_csv(gps_path):
                m = self.gps_map
                seq.gps.append(GPSSample(
                    timestamp = self._col(row, m, "t"),
                    latitude  = self._col(row, m, "lat"),
                    longitude = self._col(row, m, "lon"),
                    altitude  = self._col(row, m, "alt"),
                    heading   = self._col(row, m, "hdg"),
                    speed     = self._col(row, m, "v"),
                ))
        else:
            logger.warning("No gps.csv found in %s", run_dir)

        logger.info("Parsed %s: %d IMU, %d odom, %d GPS samples",
                    seq.name, len(seq.imu), len(seq.odom), len(seq.gps))
        return seq

    def parse_dataset(self, root: Path) -> List[Sequence]:
        """Parse all run directories under root."""
        sequences = []
        for run_dir in sorted(root.iterdir()):
            if run_dir.is_dir():
                sequences.append(self.parse_run(run_dir))
        logger.info("Total sequences loaded: %d", len(sequences))
        return sequences


# ---------------------------------------------------------------------------
# Synchronization: interpolate odom and gps onto IMU timestamps
# ---------------------------------------------------------------------------

def interpolate_at(timestamps: np.ndarray,
                   values: np.ndarray,
                   query_times: np.ndarray) -> np.ndarray:
    """
    Linear interpolation of values onto query_times.
    Clamps extrapolation to boundary values.

    Args:
        timestamps: (N,) sorted source timestamps
        values:     (N, D) or (N,) source values
        query_times: (M,) target timestamps

    Returns:
        (M, D) or (M,) interpolated values
    """
    scalar = values.ndim == 1
    if scalar:
        values = values[:, None]
    out = np.empty((len(query_times), values.shape[1]), dtype=np.float32)
    for i, t in enumerate(query_times):
        idx = np.searchsorted(timestamps, t)
        if idx == 0:
            out[i] = values[0]
        elif idx >= len(timestamps):
            out[i] = values[-1]
        else:
            t0, t1 = timestamps[idx - 1], timestamps[idx]
            alpha = (t - t0) / (t1 - t0 + 1e-9)
            out[i] = (1 - alpha) * values[idx - 1] + alpha * values[idx]
    return out[:, 0] if scalar else out


def synchronize_sequence(seq: Sequence) -> Optional[Dict[str, np.ndarray]]:
    """
    Interpolate odom and GPS onto IMU timestamps.

    Returns a dict with keys:
        'timestamps': (N,)
        'imu':        (N, 6)  — [ax, ay, az, gx, gy, gz]
        'odom':       (N, 2)  — [speed, steer_angle]
        'gps':        (N, 5)  — [lat, lon, alt, heading, speed]  (raw degrees)
        'gps_xy':     (N, 2)  — local Cartesian [x, y] in metres
        'gps_heading':(N,)    — heading in radians
    Returns None if any stream is empty.
    """
    if not seq.imu or not seq.gps:
        logger.warning("Skipping %s: empty IMU or GPS stream", seq.name)
        return None

    imu_t = np.array([s.timestamp for s in seq.imu], dtype=np.float64)
    imu_v = np.array([[s.accel_x, s.accel_y, s.accel_z,
                        s.gyro_x, s.gyro_y, s.gyro_z]
                       for s in seq.imu], dtype=np.float32)

    gps_t = np.array([s.timestamp for s in seq.gps], dtype=np.float64)
    gps_v = np.array([[s.latitude, s.longitude, s.altitude,
                        s.heading, s.speed]
                       for s in seq.gps], dtype=np.float32)

    # Interpolate GPS onto IMU timebase
    gps_at_imu = interpolate_at(gps_t, gps_v, imu_t)

    # Convert lat/lon to local XY using first GPS fix as origin
    ref_lat = float(gps_v[0, 0])
    ref_lon = float(gps_v[0, 1])
    gps_xy = np.array([latlon_to_local_xy(row[0], row[1], ref_lat, ref_lon)
                        for row in gps_at_imu], dtype=np.float32)
    gps_heading_rad = np.array([heading_to_rad(row[3]) for row in gps_at_imu],
                                dtype=np.float32)

    result: Dict[str, np.ndarray] = {
        "timestamps":   imu_t.astype(np.float32),
        "imu":          imu_v,
        "gps":          gps_at_imu,
        "gps_xy":       gps_xy,
        "gps_heading":  gps_heading_rad,
    }

    # Odometry (optional — use zeros if absent)
    if seq.odom:
        odom_t = np.array([s.timestamp for s in seq.odom], dtype=np.float64)
        odom_v = np.array([[s.speed, s.steer_angle] for s in seq.odom],
                           dtype=np.float32)
        result["odom"] = interpolate_at(odom_t, odom_v, imu_t)
    else:
        result["odom"] = np.zeros((len(imu_t), 2), dtype=np.float32)

    return result


# ---------------------------------------------------------------------------
# GNSS masking: simulate outage windows
# ---------------------------------------------------------------------------

def simulate_gnss_outages(
    n_timesteps: int,
    imu_hz: float = 100.0,
    min_outage_s: float = 10.0,
    max_outage_s: float = 120.0,
    min_gap_s: float = 30.0,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """
    Generate a boolean mask of shape (n_timesteps,).
    True  = GPS available
    False = GPS masked (simulated outage)

    The mask guarantees:
    - Outages are between min_outage_s and max_outage_s long
    - At least min_gap_s of valid GPS between consecutive outages
    - The sequence starts and ends with valid GPS
    """
    if rng is None:
        rng = random.Random()

    mask = np.ones(n_timesteps, dtype=bool)
    min_outage = int(min_outage_s * imu_hz)
    max_outage = int(max_outage_s * imu_hz)
    min_gap    = int(min_gap_s    * imu_hz)

    cursor = min_gap  # start after a buffer
    while cursor < n_timesteps - min_gap - min_outage:
        length = rng.randint(min_outage, min(max_outage, n_timesteps - cursor - min_gap))
        mask[cursor: cursor + length] = False
        cursor += length + min_gap

    return mask


# ---------------------------------------------------------------------------
# SE(2) pose computation from GPS XY track
# ---------------------------------------------------------------------------

def compute_se2_poses(gps_xy: np.ndarray,
                      gps_heading: np.ndarray) -> np.ndarray:
    """
    Compute SE(2) pose (x, y, theta) array from GPS XY + heading.

    Returns:
        poses: (N, 3) — [x_m, y_m, theta_rad]
    """
    poses = np.stack([gps_xy[:, 0], gps_xy[:, 1], gps_heading], axis=-1)
    return poses.astype(np.float32)


def compute_pose_deltas(poses: np.ndarray) -> np.ndarray:
    """
    Convert absolute SE(2) poses to relative pose deltas in the vehicle frame.

    For each consecutive pair (p_t, p_{t+1}):
        dx, dy  = rotation(-theta_t) @ (p_{t+1}[:2] - p_t[:2])
        dtheta  = wrap(theta_{t+1} - theta_t)

    Returns:
        deltas: (N-1, 3) — [dx, dy, dtheta]
    """
    N = len(poses)
    deltas = np.zeros((N - 1, 3), dtype=np.float32)
    for i in range(N - 1):
        x0, y0, theta0 = poses[i]
        x1, y1, theta1 = poses[i + 1]
        # Global displacement
        gdx = x1 - x0
        gdy = y1 - y0
        # Rotate into vehicle frame
        c, s = math.cos(-theta0), math.sin(-theta0)
        deltas[i, 0] = c * gdx - s * gdy   # dx (forward)
        deltas[i, 1] = s * gdx + c * gdy   # dy (lateral)
        # Heading delta, wrapped to [-pi, pi]
        dtheta = theta1 - theta0
        dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi
        deltas[i, 2] = dtheta
    return deltas


# ---------------------------------------------------------------------------
# Normalization statistics
# ---------------------------------------------------------------------------

@dataclass
class NormStats:
    """Per-channel mean and std for input normalization."""
    imu_mean:  np.ndarray   # (6,)
    imu_std:   np.ndarray   # (6,)
    odom_mean: np.ndarray   # (2,)
    odom_std:  np.ndarray   # (2,)

    def normalize_imu(self, x: np.ndarray) -> np.ndarray:
        return (x - self.imu_mean) / (self.imu_std + 1e-8)

    def normalize_odom(self, x: np.ndarray) -> np.ndarray:
        return (x - self.odom_mean) / (self.odom_std + 1e-8)

    def save(self, path: Path) -> None:
        np.savez(path,
                 imu_mean=self.imu_mean,   imu_std=self.imu_std,
                 odom_mean=self.odom_mean, odom_std=self.odom_std)

    @classmethod
    def load(cls, path: Path) -> "NormStats":
        d = np.load(path)
        return cls(imu_mean  = d["imu_mean"],
                   imu_std   = d["imu_std"],
                   odom_mean = d["odom_mean"],
                   odom_std  = d["odom_std"])


def compute_norm_stats(synced_sequences: List[Dict[str, np.ndarray]]) -> NormStats:
    """Compute per-channel mean/std across all sequences."""
    all_imu  = np.concatenate([s["imu"]  for s in synced_sequences], axis=0)
    all_odom = np.concatenate([s["odom"] for s in synced_sequences], axis=0)
    return NormStats(
        imu_mean  = all_imu.mean(axis=0),
        imu_std   = all_imu.std(axis=0),
        odom_mean = all_odom.mean(axis=0),
        odom_std  = all_odom.std(axis=0),
    )


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class DRDataset(Dataset):
    """
    PyTorch Dataset for dead-reckoning training.

    Each item is a fixed-length sliding window:
      inputs:   (window, 8)     — normalized [imu(6) | odom(2)]
      targets:  (window-1, 3)   — pose deltas [dx, dy, dtheta]
      mask:     (window,)       — bool, True where GPS is available
      poses_gt: (window, 3)     — absolute SE(2) poses (for loss / eval)
    """

    def __init__(
        self,
        synced_sequences: List[Dict[str, np.ndarray]],
        stats: NormStats,
        window: int = 200,
        stride: int = 50,
        min_outage_s: float = 10.0,
        max_outage_s: float = 120.0,
        imu_hz: float = 100.0,
        seed: int = 42,
    ):
        self.window        = window
        self.stride        = stride
        self.stats         = stats
        self.min_outage_s  = min_outage_s
        self.max_outage_s  = max_outage_s
        self.imu_hz        = imu_hz

        self.items: List[Dict[str, np.ndarray]] = []
        rng = random.Random(seed)

        for seq in synced_sequences:
            N = len(seq["imu"])
            poses  = compute_se2_poses(seq["gps_xy"], seq["gps_heading"])
            deltas = compute_pose_deltas(poses)   # (N-1, 3)
            mask   = simulate_gnss_outages(N, imu_hz=imu_hz,
                                           min_outage_s=min_outage_s,
                                           max_outage_s=max_outage_s,
                                           rng=rng)

            imu_norm  = stats.normalize_imu(seq["imu"])
            odom_norm = stats.normalize_odom(seq["odom"])
            inputs    = np.concatenate([imu_norm, odom_norm], axis=-1)  # (N, 8)

            for start in range(0, N - window, stride):
                end = start + window
                self.items.append({
                    "inputs":   inputs[start:end].copy(),           # (W, 8)
                    "targets":  deltas[start:end - 1].copy(),       # (W-1, 3)
                    "mask":     mask[start:end].copy(),             # (W,)
                    "poses_gt": poses[start:end].copy(),            # (W, 3)
                })

        logger.info("DRDataset: %d windows from %d sequences",
                    len(self.items), len(synced_sequences))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        return {
            "inputs":   torch.from_numpy(item["inputs"]).float(),
            "targets":  torch.from_numpy(item["targets"]).float(),
            "mask":     torch.from_numpy(item["mask"]),
            "poses_gt": torch.from_numpy(item["poses_gt"]).float(),
        }


class NavICVAEDataset(Dataset):
    """
    Dataset for training the NavIC Motion Prior VAE.

    Each item is a 60-second pre-loss GPS trajectory:
      traj: (60, 4) — [x_m, y_m, speed_m_s, heading_rad] at 1 Hz
    """

    def __init__(
        self,
        synced_sequences: List[Dict[str, np.ndarray]],
        imu_hz: float = 100.0,
        window_s: float = 60.0,
        stride_s: float = 10.0,
    ):
        downsample = int(imu_hz)           # take every 100th sample → 1 Hz
        window     = int(window_s)         # 60 points
        stride     = int(stride_s * imu_hz)

        self.trajs: List[np.ndarray] = []

        for seq in synced_sequences:
            gps_xy      = seq["gps_xy"]
            gps_heading = seq["gps_heading"]
            gps_speed   = seq["gps"][:, 4]   # interpolated GPS speed

            # Downsample to 1 Hz
            xy_1hz  = gps_xy[::downsample]
            hdg_1hz = gps_heading[::downsample]
            spd_1hz = gps_speed[::downsample]

            for i in range(0, len(xy_1hz) - window, int(stride_s)):
                chunk_xy  = xy_1hz[i:i + window]
                chunk_hdg = hdg_1hz[i:i + window]
                chunk_spd = spd_1hz[i:i + window]
                # Normalize to start at origin
                chunk_xy = chunk_xy - chunk_xy[0]
                traj = np.stack([chunk_xy[:, 0], chunk_xy[:, 1],
                                  chunk_spd, chunk_hdg], axis=-1).astype(np.float32)
                self.trajs.append(traj)

        logger.info("NavICVAEDataset: %d trajectory windows", len(self.trajs))

    def __len__(self) -> int:
        return len(self.trajs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.trajs[idx])   # (60, 4)


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def build_dataloaders(
    sequences: List[Dict[str, np.ndarray]],
    stats: NormStats,
    val_split: float = 0.15,
    test_split: float = 0.10,
    window: int = 200,
    stride: int = 50,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split sequences into train/val/test and return DataLoaders.
    Split is done at the sequence level to avoid leakage.
    """
    rng = random.Random(seed)
    seqs = list(sequences)
    rng.shuffle(seqs)

    n = len(seqs)
    n_test = max(1, int(n * test_split))
    n_val  = max(1, int(n * val_split))
    n_train = n - n_val - n_test

    train_seqs = seqs[:n_train]
    val_seqs   = seqs[n_train:n_train + n_val]
    test_seqs  = seqs[n_train + n_val:]

    def make_loader(seqs_subset, shuffle: bool) -> DataLoader:
        ds = DRDataset(seqs_subset, stats,
                       window=window, stride=stride, seed=seed)
        return DataLoader(ds, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers,
                          pin_memory=torch.cuda.is_available(),
                          drop_last=shuffle)

    return (make_loader(train_seqs, shuffle=True),
            make_loader(val_seqs,   shuffle=False),
            make_loader(test_seqs,  shuffle=False))
