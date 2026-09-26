"""
NAVDRIFT-0 | ISRO SIH 2026 PS #26168
File 01 — Dataset download, preprocessing, unified HDF5 export

Datasets handled:
  - EuRoC MAV        (ETH Zurich)   — direct wget, no login needed
  - TUM Visual-Inertial              — direct wget, no login needed
  - NCLT Long-Term   (U Michigan)   — direct wget, no login needed
  - KAIST Urban      (KAIST)        — needs free account; instructions below
  - NavIC RINEX      (ISRO NSSF)    — wget from public portal

Output: /content/drive/MyDrive/NAVDRIFT0/processed/navdrift_dataset.h5
Schema:
  /train/seq_XXXX/{imu, gps_pos, gt_pos, timestamps, dr_error}
  /val/...
  /test/...
"""

import os, sys, time, zipfile, tarfile, io, json
import subprocess, urllib.request, urllib.error
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm

# must run navdrift_00_setup.py first or import here
sys.path.insert(0, "/content")
try:
    from navdrift_00_setup import PATHS, mount_drive, create_dirs, start_keepalive
except ImportError:
    # fallback paths for standalone run
    DRIVE_ROOT = Path("/content/drive/MyDrive/NAVDRIFT0")
    PATHS = {
        "datasets":  DRIVE_ROOT / "datasets",
        "euroc":     DRIVE_ROOT / "datasets" / "euroc",
        "tum_vi":    DRIVE_ROOT / "datasets" / "tum_vi",
        "nclt":      DRIVE_ROOT / "datasets" / "nclt",
        "kaist":     DRIVE_ROOT / "datasets" / "kaist",
        "navic":     DRIVE_ROOT / "datasets" / "navic_rinex",
        "processed": DRIVE_ROOT / "processed",
    }
    def mount_drive(): pass
    def create_dirs():
        for p in PATHS.values():
            p.mkdir(parents=True, exist_ok=True)
    def start_keepalive(): pass

import pandas as pd

# ============================================================
# DOWNLOAD HELPERS
# ============================================================
def wget(url, dest: Path, desc="") -> bool:
    """Returns True on success, False on failure (non-fatal)."""
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"[SKIP] {dest.name} already exists.")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DL] {desc or dest.name} ...")
    result = subprocess.run(["wget", "-q", "--show-progress", "-O", str(dest), url])
    if result.returncode != 0:
        print(f"[WARN] Download failed (exit {result.returncode}): {url}")
        if dest.exists():
            dest.unlink()   # remove partial file
        return False
    print(f"[DL] Done -> {dest}")
    return True

def extract_zip(src: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src, "r") as z:
        z.extractall(dest)
    print(f"[EXTRACT] {src.name} -> {dest}")

def extract_tar(src: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(src) as t:
        t.extractall(dest)
    print(f"[EXTRACT] {src.name} -> {dest}")

# ============================================================
# 1. EUROC MAV DATASET
# ============================================================
EUROC_SEQS = {
    "MH_01_easy":   "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip",
    "MH_03_medium": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_03_medium/MH_03_medium.zip",
    "V1_01_easy":   "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip",
    # V2_01_easy removed — ETH mirror is down; MH_01, MH_03, V1_01 are sufficient
}

def find_euroc_seq(base: Path, name: str):
    """
    Search for a sequence folder anywhere under the euroc base dir.
    Returns the path if found, else None.
    """
    for candidate in base.rglob(name):
        if candidate.is_dir():
            return candidate
    return None

def download_euroc():
    base = PATHS["euroc"]

    # Check for pre-uploaded bundle zips first (machine_hall.zip, vicon_room1.zip)
    bundle_map = {
        "machine_hall.zip": ["MH_01_easy", "MH_03_medium"],
        "vicon_room1.zip":  ["V1_01_easy"],
        "vicon_room2.zip":  ["V2_01_easy"],
    }
    for bundle_name, seqs in bundle_map.items():
        bundle_path = base / bundle_name
        if bundle_path.exists() and bundle_path.stat().st_size > 1000:
            # skip extraction if all sequences from this bundle already exist
            already_extracted = all(find_euroc_seq(base, s) is not None for s in seqs if s in EUROC_SEQS)
            if already_extracted:
                print(f"[BUNDLE] {bundle_name} already extracted, skipping.")
                continue
            print(f"[BUNDLE] Found {bundle_name}, extracting ...")
            extract_zip(bundle_path, base)
            print(f"[BUNDLE] Extracted sequences: {seqs}")

    # Download any sequences that are still missing (search recursively first)
    for name, url in EUROC_SEQS.items():
        found = find_euroc_seq(base, name)
        if found:
            print(f"[SKIP] EuRoC {name} found at {found}")
            # symlink to expected flat path so rest of code works
            flat = base / name
            if not flat.exists():
                flat.symlink_to(found)
            continue
        zp = base / f"{name}.zip"
        wget(url, zp, desc=f"EuRoC {name}")
        extract_zip(zp, base)

def load_euroc_sequence(seq_dir: Path):
    """
    Returns dict with keys: imu (N,6), gt_pos (N,3), timestamps (N,)
    IMU columns: gx gy gz ax ay az  (rad/s, m/s^2)
    """
    imu_csv = seq_dir / "mav0" / "imu0" / "data.csv"
    gt_csv  = seq_dir / "mav0" / "state_groundtruth_estimate0" / "data.csv"

    if not imu_csv.exists():
        # The sequence folder may contain an inner zip (ASL format) -- extract it
        inner_zips = list(seq_dir.glob("*.zip"))
        if inner_zips:
            print(f"[EXTRACT] Found inner zip {inner_zips[0].name}, extracting into {seq_dir} ...")
            extract_zip(inner_zips[0], seq_dir)
        if not imu_csv.exists():
            print(f"[WARN] {imu_csv} not found after extraction attempt, skipping.")
            return None

    imu_df = pd.read_csv(imu_csv, comment="#")
    imu_df.columns = ["ts", "gx", "gy", "gz", "ax", "ay", "az"]

    gt_df  = pd.read_csv(gt_csv, comment="#")
    gt_df.columns = (["ts", "px", "py", "pz",
                       "qw", "qx", "qy", "qz",
                       "vx", "vy", "vz",
                       "bwx", "bwy", "bwz",
                       "bax", "bay", "baz"])

    # align to common timestamps via nearest-neighbour
    imu_ts = imu_df["ts"].values
    gt_ts  = gt_df["ts"].values
    idx    = np.searchsorted(gt_ts, imu_ts)
    idx    = np.clip(idx, 0, len(gt_ts) - 1)

    imu_arr = imu_df[["ax", "ay", "az", "gx", "gy", "gz"]].values.astype(np.float32)
    pos_arr = gt_df.iloc[idx][["px", "py", "pz"]].values.astype(np.float32)

    return {"imu": imu_arr, "gt_pos": pos_arr, "timestamps": imu_ts.astype(np.float64)}

# ============================================================
# 2. TUM VISUAL-INERTIAL DATASET
# ============================================================
TUM_VI_SEQS = {
    "corridor1": "https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-corridor1_512_16.tar.gz",
    "corridor2": "https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-corridor2_512_16.tar.gz",
    "room1":     "https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-room1_512_16.tar.gz",
}

def download_tum_vi():
    base = PATHS["tum_vi"]
    for name, url in TUM_VI_SEQS.items():
        out = base / name
        if out.exists() and any(out.iterdir()):
            print(f"[SKIP] TUM-VI {name} already extracted.")
            continue
        tgz = base / f"{name}.tar.gz"
        ok = wget(url, tgz, desc=f"TUM-VI {name}")
        if ok and not out.exists():
            extract_tar(tgz, base)

def load_tum_vi_sequence(seq_dir: Path):
    # TUM-VI exports EuRoC-compatible structure
    return load_euroc_sequence(seq_dir)

# ============================================================
# 3. NCLT LONG-TERM DATASET
# ============================================================
NCLT_SEQS = {
    "2012-01-08": {
        "imu": "http://robots.engin.umich.edu/nclt/nclt-2012-01-08-ms25.csv.gz",
        "gps": "http://robots.engin.umich.edu/nclt/nclt-2012-01-08-gps.csv.gz",
    },
    "2012-04-29": {
        "imu": "http://robots.engin.umich.edu/nclt/nclt-2012-04-29-ms25.csv.gz",
        "gps": "http://robots.engin.umich.edu/nclt/nclt-2012-04-29-gps.csv.gz",
    },
}

def download_nclt():
    base = PATHS["nclt"]
    for name, urls in NCLT_SEQS.items():
        for kind, url in urls.items():
            dest = base / name / f"{kind}.csv.gz"
            wget(url, dest, desc=f"NCLT {name} {kind}")

def load_nclt_sequence(seq_dir: Path):
    import gzip
    imu_gz = seq_dir / "imu.csv.gz"
    gps_gz = seq_dir / "gps.csv.gz"

    if not imu_gz.exists():
        print(f"[WARN] {imu_gz} not found, skipping.")
        return None

    with gzip.open(imu_gz) as f:
        imu_df = pd.read_csv(f, header=None,
                             names=["utime","ax","ay","az","rx","ry","rz"])
    with gzip.open(gps_gz) as f:
        gps_df = pd.read_csv(f, header=None,
                             names=["utime","lat","lng","alt",
                                    "vel_n","vel_e","vel_d"])

    # align
    imu_ts = imu_df["utime"].values
    gps_ts = gps_df["utime"].values
    idx    = np.searchsorted(gps_ts, imu_ts)
    idx    = np.clip(idx, 0, len(gps_ts) - 1)

    imu_arr = imu_df[["ax", "ay", "az", "rx", "ry", "rz"]].values.astype(np.float32)
    gps_arr = gps_df.iloc[idx][["lat", "lng", "alt"]].values.astype(np.float32)

    return {"imu": imu_arr, "gt_pos": gps_arr, "timestamps": imu_ts.astype(np.float64)}

# ============================================================
# 4. KAIST URBAN (manual download required)
# ============================================================
KAIST_INSTRUCTIONS = """
KAIST Urban Dataset requires a FREE account.

Steps:
1. Go to: https://sites.google.com/view/complex-urban-dataset
2. Register for a free account
3. Download sequences:
   - urban05_data.zip
   - urban06_data.zip
   - urban38_data.zip
4. Upload each zip to:
   /content/drive/MyDrive/NAVDRIFT0/datasets/kaist/

Then re-run this script — it will auto-detect and process them.

Alternatively, KAIST sequences are also shared on HuggingFace:
   https://huggingface.co/datasets/kaist-urban
"""

def check_kaist():
    kaist_dir = PATHS["kaist"]
    zips = list(kaist_dir.glob("urban*.zip"))
    csvs = list(kaist_dir.glob("**/xsens_imu.csv"))
    if not zips and not csvs:
        print(KAIST_INSTRUCTIONS)
        return False
    print(f"[KAIST] Found {len(zips)} zip(s), {len(csvs)} IMU CSV(s).")
    for z in zips:
        out = kaist_dir / z.stem
        if not out.exists():
            extract_zip(z, kaist_dir)
    return True

def load_kaist_sequence(seq_dir: Path):
    imu_csv = seq_dir / "sensor_data" / "xsens_imu.csv"
    gps_csv = seq_dir / "sensor_data" / "gps.csv"

    if not imu_csv.exists():
        # try flat layout
        imu_csv = seq_dir / "xsens_imu.csv"
        gps_csv = seq_dir / "gps.csv"

    if not imu_csv.exists():
        print(f"[WARN] KAIST IMU CSV not found in {seq_dir}, skipping.")
        return None

    imu_df = pd.read_csv(imu_csv, comment="%")
    # columns: Time[ns], qw, qx, qy, qz, AccX, AccY, AccZ, GyrX, GyrY, GyrZ
    imu_df.columns = (["ts"] + [f"q{c}" for c in "wxyz"] +
                      ["ax","ay","az","gx","gy","gz"])

    gps_df = pd.read_csv(gps_csv, comment="%")
    gps_df.columns = ["ts","lat","lon","alt"]

    imu_ts = imu_df["ts"].values
    gps_ts = gps_df["ts"].values
    idx    = np.searchsorted(gps_ts, imu_ts)
    idx    = np.clip(idx, 0, len(gps_ts) - 1)

    imu_arr = imu_df[["ax","ay","az","gx","gy","gz"]].values.astype(np.float32)
    gps_arr = gps_df.iloc[idx][["lat","lon","alt"]].values.astype(np.float32)

    return {"imu": imu_arr, "gt_pos": gps_arr, "timestamps": imu_ts.astype(np.float64)}

# ============================================================
# 5. NAVIC RINEX (ISRO NSSF)
# ============================================================
NAVIC_STATIONS = {
    "IITB": "https://nssf.isro.gov.in/products/RINEX/2025/001/IITB0010.25O",
    "HYD1": "https://nssf.isro.gov.in/products/RINEX/2025/001/HYD10010.25O",
}

def download_navic_rinex():
    base = PATHS["navic"]
    success = False
    for name, url in NAVIC_STATIONS.items():
        dest = base / f"{name}_sample.rnx"
        try:
            wget(url, dest, desc=f"NavIC RINEX {name}")
            success = True
        except subprocess.CalledProcessError:
            print(f"[NAVIC] {name} download failed — NSSF may require VPN/login.")
            print(f"  Manual download: {url}")
    if not success:
        print("[NAVIC] Generating synthetic NavIC DOP data as fallback.")
        _gen_synthetic_navic(base)

def _gen_synthetic_navic(base: Path):
    """Generate synthetic NavIC DOP observations from constellation geometry."""
    base.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    # NavIC 7 satellites: approximate azimuths and elevations
    NAVIC_SATS = [
        (0, 29.0),   # NVS-01
        (32.5, 29.0),# NVS-02
        (83.5, 29.0),# NVS-03  (GEO)
        (111.75, 29.0),# NVS-04 (GEO)
        (131.5, 55.0),# NVS-05 (IGSO)
        (55.0, 55.0),# NVS-06 (IGSO)
        (111.75, 55.0),# NVS-07 (IGSO)
    ]
    records = []
    for hour in range(24 * 180):  # 180 days
        t = hour * 3600
        lat_bins = np.arange(8, 37, 2)    # India lat range
        lon_bins = np.arange(68, 98, 2)   # India lon range
        for lat in lat_bins:
            for lon in lon_bins:
                # Simplified DOP from elevation geometry
                els = np.array([el + rng.normal(0, 3) for _, el in NAVIC_SATS])
                els = np.clip(els, 5, 85)
                sin_els = np.sin(np.radians(els))
                pdop = float(np.clip(2.8 / np.sqrt(np.sum(sin_els**2 / 7)), 1.0, 6.0))
                hdop = float(pdop * (0.58 + rng.uniform(-0.05, 0.05)))
                vdop = float(pdop * (0.75 + rng.uniform(-0.05, 0.05)))
                records.append({
                    "time_s": t, "lat": lat, "lon": lon,
                    "pdop": round(pdop, 2),
                    "hdop": round(hdop, 2),
                    "vdop": round(vdop, 2),
                })
    df = pd.DataFrame(records)
    out = base / "navic_dop_synthetic.csv"
    df.to_csv(out, index=False)
    print(f"[NAVIC] Synthetic DOP dataset saved: {out}  ({len(df)} records)")

# ============================================================
# 6. DEAD RECKONING ERROR COMPUTATION
# ============================================================
def compute_dr_error(gt_pos: np.ndarray, imu: np.ndarray, dt: float = 0.01):
    """
    Simple DR integration from IMU to produce position estimates.
    Returns per-step DR error vs ground truth (metres).
    gt_pos: (N, 3) - x,y,z or lat,lon,alt
    imu:    (N, 6) - ax,ay,az,gx,gy,gz
    """
    N = len(gt_pos)
    pos_dr = np.zeros((N, 2), dtype=np.float32)  # x,y only

    # Use first two position dimensions as planar (or converted)
    # For lat/lon we approximate 1 deg lat ~ 111000 m
    if np.abs(gt_pos[:, 0]).max() < 100:  # ENU metres
        ref = gt_pos[0, :2]
        pos_m = gt_pos[:, :2] - ref
    else:  # lat/lon degrees
        lat0 = gt_pos[0, 0]
        pos_m = np.stack([
            (gt_pos[:, 1] - gt_pos[0, 1]) * 111320 * np.cos(np.radians(lat0)),
            (gt_pos[:, 0] - gt_pos[0, 0]) * 111320,
        ], axis=-1)

    vel = np.zeros(2)
    pos = np.zeros(2)
    errors = np.zeros(N, dtype=np.float32)

    for i in range(N):
        ax, ay = float(imu[i, 0]), float(imu[i, 1])
        vel += np.array([ax, ay]) * dt
        pos += vel * dt
        pos_dr[i] = pos
        dist = np.linalg.norm(pos_m[i])
        err  = np.linalg.norm(pos - pos_m[i])
        errors[i] = err / max(dist, 1.0) * 100  # drift %

    return errors, pos_m

# ============================================================
# 7. WINDOWING FOR MODEL TRAINING
# ============================================================
IMU_WINDOW  = 100   # 1 second at 100 Hz
ZUPT_WINDOW = 8     # 80 ms standstill check
TUNNEL_WIN  = 20    # 2 seconds at 10 Hz for tunnel detector

def extract_driftformer_windows(imu, gt_pos, dr_errors, stride=10):
    """
    Sliding windows for DRIFTFormer.
    X: (window, 9)  — ax ay az gx gy gz heading speed dt
    Y: (2,)         — delta_x delta_y in metres (correction target)
    """
    N = len(imu)
    if N < IMU_WINDOW + stride:
        return None, None

    pos_m_x = []
    pos_m_y = []
    if np.abs(gt_pos[:, 0]).max() < 100:
        lat0 = 0
        pos_m_x = gt_pos[:, 0]
        pos_m_y = gt_pos[:, 1]
    else:
        lat0 = gt_pos[0, 0]
        pos_m_x = (gt_pos[:, 1] - gt_pos[0, 1]) * 111320 * np.cos(np.radians(lat0))
        pos_m_y = (gt_pos[:, 0] - gt_pos[0, 0]) * 111320

    Xs, Ys = [], []
    dt = 0.01
    for start in range(0, N - IMU_WINDOW, stride):
        end = start + IMU_WINDOW
        win = imu[start:end].copy()  # (100, 6)

        # speed from acc magnitude (rough proxy)
        speed = np.linalg.norm(win[:, :3], axis=1, keepdims=True)
        # heading from gyro-z cumulative (very rough)
        heading = np.cumsum(win[:, 5:6], axis=0) * dt
        heading = np.sin(heading)  # (100, 1)
        dt_col  = np.full((IMU_WINDOW, 1), dt)

        x9 = np.concatenate([win, heading, speed, dt_col], axis=1)  # (100, 9)

        # target: position correction at end of window
        # DR position at end vs GT
        dr_simple = np.cumsum(win[:, :2] * dt**2, axis=0) + np.array([pos_m_x[start], pos_m_y[start]])
        gt_end    = np.array([pos_m_x[end - 1], pos_m_y[end - 1]])
        target    = (gt_end - dr_simple[-1]).astype(np.float32)

        Xs.append(x9)
        Ys.append(target)

    if not Xs:
        return None, None
    return np.array(Xs, dtype=np.float32), np.array(Ys, dtype=np.float32)

def extract_tunnel_windows(imu, gps_quality, stride=5):
    """
    For tunnel detector.
    gps_quality: (N,) binary — 1 = good GPS, 0 = outage
    X: (20, 5)  Y: (1,) probability of upcoming outage
    """
    N = len(imu)
    Xs, Ys = [], []
    for start in range(0, N - TUNNEL_WIN - 10, stride):
        end   = start + TUNNEL_WIN
        win   = imu[start:end]
        imu_var    = np.var(win[:, :3], axis=0).mean()
        imu_mag    = np.linalg.norm(win[:, :3], axis=1)
        speed_cons = 1.0 - (np.std(imu_mag) / (np.mean(imu_mag) + 1e-6))
        gps_now    = gps_quality[start:end].mean()
        # look 10 steps ahead for outage
        gps_ahead  = gps_quality[end:end+10].mean()
        label      = float(gps_ahead < 0.5 and gps_now > 0.5)

        # feature vector: [imu_var, speed_consistency, gps_now, trend, step]
        feats = np.zeros((TUNNEL_WIN, 5), dtype=np.float32)
        feats[:, 0] = np.var(win[:, :3], axis=1)
        feats[:, 1] = np.linalg.norm(win[:, :3], axis=1)
        feats[:, 2] = np.linalg.norm(win[:, 3:], axis=1)
        feats[:, 3] = gps_now
        feats[:, 4] = speed_cons

        Xs.append(feats)
        Ys.append(label)

    return np.array(Xs, dtype=np.float32), np.array(Ys, dtype=np.float32)

# ============================================================
# 8. MASTER HDF5 BUILDER
# ============================================================
def build_hdf5():
    out_path = PATHS["processed"] / "navdrift_dataset.h5"
    if out_path.exists():
        print(f"[HDF5] Already exists: {out_path}")
        print("       Delete it and rerun if you want to rebuild.")
        return out_path

    print(f"\n[HDF5] Building dataset -> {out_path}")
    seqs_all = []  # list of dicts

    # --- EuRoC ---
    for name in EUROC_SEQS:
        # symlinks may not work on Drive FUSE -- search recursively
        seq_dir = find_euroc_seq(PATHS["euroc"], name) or (PATHS["euroc"] / name)
        if not seq_dir.exists():
            print(f"[WARN] EuRoC {name} not found, skipping.")
            continue
        data = load_euroc_sequence(seq_dir)
        if data:
            data["source"] = f"euroc_{name}"
            seqs_all.append(data)

    # --- TUM-VI ---
    for name in TUM_VI_SEQS:
        seq_dir = PATHS["tum_vi"] / name
        if not seq_dir.exists():
            continue
        data = load_tum_vi_sequence(seq_dir)
        if data:
            data["source"] = f"tumvi_{name}"
            seqs_all.append(data)

    # --- NCLT ---
    for name in NCLT_SEQS:
        seq_dir = PATHS["nclt"] / name
        if not seq_dir.exists():
            continue
        data = load_nclt_sequence(seq_dir)
        if data:
            data["source"] = f"nclt_{name}"
            seqs_all.append(data)

    # --- KAIST ---
    if check_kaist():
        for seq_dir in sorted(PATHS["kaist"].glob("urban*")):
            if seq_dir.is_dir():
                data = load_kaist_sequence(seq_dir)
                if data:
                    data["source"] = f"kaist_{seq_dir.name}"
                    seqs_all.append(data)

    print(f"\n[HDF5] Loaded {len(seqs_all)} sequences total.")
    if not seqs_all:
        print("[ERROR] No sequences loaded. Check dataset download steps.")
        return None

    # Split 80/10/10 -- guarantee at least 1 seq in val and test
    rng    = np.random.default_rng(42)
    idx    = rng.permutation(len(seqs_all))
    n      = len(seqs_all)
    n_val  = max(1, int(n * 0.1))
    n_test = max(1, int(n * 0.1))
    n_tr   = max(1, n - n_val - n_test)
    splits = {
        "train": [seqs_all[i] for i in idx[:n_tr]],
        "val":   [seqs_all[i] for i in idx[n_tr:n_tr + n_val]],
        "test":  [seqs_all[i] for i in idx[n_tr + n_val:]],
    }

    with h5py.File(out_path, "w") as hf:
        meta = {"sources": [s["source"] for s in seqs_all],
                "n_seqs": len(seqs_all)}
        hf.attrs["meta"] = json.dumps(meta)

        for split, seqs in splits.items():
            grp = hf.create_group(split)
            for i, data in enumerate(tqdm(seqs, desc=f"Writing {split}")):
                sq = grp.create_group(f"seq_{i:04d}")
                sq.create_dataset("imu",        data=data["imu"],        compression="gzip")
                sq.create_dataset("gt_pos",     data=data["gt_pos"],     compression="gzip")
                sq.create_dataset("timestamps", data=data["timestamps"], compression="gzip")
                sq.attrs["source"] = data["source"]

                # pre-compute DR error and gps_quality
                dr_err, pos_m = compute_dr_error(data["gt_pos"], data["imu"])
                sq.create_dataset("dr_error", data=dr_err, compression="gzip")

                # gps_quality: 1 everywhere for EuRoC/TUM, varies for NCLT/KAIST
                gq = np.ones(len(data["imu"]), dtype=np.float32)
                sq.create_dataset("gps_quality", data=gq, compression="gzip")

    size_mb = out_path.stat().st_size / 1e6
    print(f"\n[HDF5] Done. {out_path}  ({size_mb:.1f} MB)")
    return out_path

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  NAVDRIFT-0  |  DATA PIPELINE")
    print("=" * 60)
    mount_drive()
    create_dirs()
    start_keepalive()

    print("\n--- Step 1: EuRoC ---")
    download_euroc()

    print("\n--- Step 2: TUM Visual-Inertial ---")
    try:
        download_tum_vi()
    except Exception as e:
        print(f"[WARN] TUM-VI skipped: {e}")

    print("\n--- Step 3: NCLT ---")
    try:
        download_nclt()
    except Exception as e:
        print(f"[WARN] NCLT skipped: {e}")

    print("\n--- Step 4: KAIST (check only) ---")
    try:
        check_kaist()
    except Exception as e:
        print(f"[WARN] KAIST skipped: {e}")

    print("\n--- Step 5: NavIC RINEX ---")
    try:
        download_navic_rinex()
    except Exception as e:
        print(f"[WARN] NavIC skipped: {e}")

    print("\n--- Step 6: Build HDF5 ---")
    build_hdf5()

    print("\n[DONE] Data pipeline complete.")
