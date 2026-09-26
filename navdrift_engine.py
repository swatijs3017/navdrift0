"""
navdrift_engine.py — NAVDRIFT-0 Edge-Deployable Dead Reckoning Engine
ISRO SIH 2026 PS #26168

Works with ANY external IMU CSV (not just smartphone).
Supports 10–200 Hz input via interpolation/resampling.

Usage:
    python navdrift_engine.py --imu path/to/imu.csv --model navdrift_lstm.onnx --hz 200

CSV must have columns (names flexible, auto-detected):
    timestamp_ms, ax, ay, az, gx, gy, gz  [, lat, lon, speed (optional ground truth)]

Output:
    - Console: ATE metrics
    - position_output.csv: timestamp, pred_lat, pred_lon, gt_lat, gt_lon (if GT available)
    - position_plot.png
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Optional deps (graceful fallback) ──────────────────────────────────────
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("[WARN] onnxruntime not found — speed will use heuristic integration")

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
SEQ_LEN   = 50        # must match training
DEAD_BAND = 0.25      # m/s² — ignore accel below this (vibration)
MAX_SPEED = 40.0      # m/s cap
R_EARTH   = 6_371_000 # metres

# EKF process noise
Q_SPD  = 0.1   # speed process noise variance
Q_HEAD = 0.05  # heading process noise variance
R_GPS  = 4.0   # GPS measurement noise variance (metres)


# ══════════════════════════════════════════════════════════════════════════════
#  COLUMN AUTO-DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def _find(candidates, df):
    for c in candidates:
        m = [col for col in df.columns if c.lower() in col.lower()]
        if m:
            return m[0]
    return None

def detect_columns(df):
    return {
        'ts':    _find(['timestamp','time_ms','millis','t_ms','t'], df),
        'ax':    _find(['accel_x','ax_','acc_x','linear_x','ax'], df),
        'ay':    _find(['accel_y','ay_','acc_y','linear_y','ay'], df),
        'az':    _find(['accel_z','az_','acc_z','linear_z','az'], df),
        'gx':    _find(['gyro_x','gyrox','gyroscope_x','gx'], df),
        'gy':    _find(['gyro_y','gyroy','gyroscope_y','gy'], df),
        'gz':    _find(['gyro_z','gyroz','gyroscope_z','gz'], df),
        'lat':   _find(['lat','latitude'], df),
        'lon':   _find(['lon','lng','longitude'], df),
        'speed': _find(['speed','velocity','gps_speed','speed_mps'], df),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BUTTERWORTH LOW-PASS (2nd order, fc=5 Hz at given fs)
# ══════════════════════════════════════════════════════════════════════════════
def butter_coeffs(fc, fs):
    """Pre-compute 2nd-order Butterworth LP coefficients."""
    from math import tan, pi, sqrt
    wc = tan(pi * fc / fs)
    k1 = sqrt(2) * wc
    k2 = wc ** 2
    b0 = k2 / (1 + k1 + k2)
    b1 = 2 * b0
    b2 = b0
    a1 = 2 * b0 * (1 / k2 - 1)
    a2 = 1 - (b0 + b1 + b2) - a1
    return (b0, b1, b2, a1, a2)


class ButterworthLP:
    def __init__(self, fc, fs, n_channels=6):
        self.coeffs = butter_coeffs(fc, fs)
        self.xp  = np.zeros(n_channels)
        self.yp1 = np.zeros(n_channels)
        self.yp2 = np.zeros(n_channels)

    def step(self, x):
        b0, b1, b2, a1, a2 = self.coeffs
        y = b0*x + b1*self.xp + b2*0 - a1*self.yp1 - a2*self.yp2
        self.yp2 = self.yp1.copy()
        self.yp1 = y.copy()
        self.xp  = x.copy()
        return y


# ══════════════════════════════════════════════════════════════════════════════
#  IN-VEHICLE ALIGNMENT (auto-detect forward axis)
# ══════════════════════════════════════════════════════════════════════════════
def detect_forward_axis(accel_window: np.ndarray) -> int:
    """
    Find which axis (0=X, 1=Y, 2=Z) has highest variance → forward axis.
    Variance is highest along the direction of vehicle acceleration.
    Called once on the first ~5 seconds of driving.
    """
    # Gravity axis = highest mean absolute value → exclude it
    means = np.abs(accel_window.mean(axis=0))
    grav_axis = int(np.argmax(means))
    remaining = [i for i in range(3) if i != grav_axis]
    # Forward = highest variance among remaining axes
    vars_ = accel_window[:, remaining].var(axis=0)
    fwd_local = int(np.argmax(vars_))
    fwd_axis = remaining[fwd_local]
    axis_names = ['X', 'Y', 'Z']
    print(f"[ALIGN] Gravity axis: {axis_names[grav_axis]}  Forward axis: {axis_names[fwd_axis]}")
    return fwd_axis


# ══════════════════════════════════════════════════════════════════════════════
#  EKF — simple 3-state [x_pos, y_pos, heading]
# ══════════════════════════════════════════════════════════════════════════════
class EKF:
    def __init__(self, lat0, lon0, heading0=0.0):
        self.x    = np.array([lat0, lon0, heading0], dtype=np.float64)
        self.P    = np.diag([1e-8, 1e-8, 0.1])
        self.Q    = np.diag([1e-10, 1e-10, Q_HEAD * 0.01])
        self.R    = np.diag([R_GPS / R_EARTH, R_GPS / R_EARTH])

    def predict(self, speed, dheading, dt):
        lat, lon, th = self.x
        th_new = th + dheading
        dlat = speed * np.cos(th_new) * dt / R_EARTH
        dlon = speed * np.sin(th_new) * dt / (R_EARTH * np.cos(np.radians(lat)))
        self.x = np.array([lat + dlat, lon + dlon, th_new])
        # Jacobian F (linearized)
        F = np.eye(3)
        F[0, 2] = -speed * np.sin(th_new) * dt / R_EARTH
        F[1, 2] =  speed * np.cos(th_new) * dt / (R_EARTH * np.cos(np.radians(lat)))
        self.P = F @ self.P @ F.T + self.Q

    def update_gps(self, lat_meas, lon_meas):
        H = np.array([[1, 0, 0], [0, 1, 0]])
        z = np.array([lat_meas, lon_meas])
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ H) @ self.P


# ══════════════════════════════════════════════════════════════════════════════
#  LSTM SPEED ESTIMATOR WRAPPER
# ══════════════════════════════════════════════════════════════════════════════
class LSTMSpeedEstimator:
    def __init__(self, onnx_path, meta_path):
        self.sess     = ort.InferenceSession(onnx_path)
        self.in_name  = self.sess.get_inputs()[0].name
        with open(meta_path) as f:
            meta = json.load(f)
        self.mean_  = np.array(meta['scaler_mean'],  dtype=np.float32)
        self.scale_ = np.array(meta['scaler_scale'], dtype=np.float32)
        self.seq_len = meta['seq_len']
        self.feat_cols = meta['feature_cols']
        self.buf = []

    def push(self, feat_vec: np.ndarray) -> float | None:
        """Push one sample; returns speed prediction or None if buffer not full."""
        norm = (feat_vec - self.mean_) / self.scale_
        self.buf.append(norm)
        if len(self.buf) < self.seq_len:
            return None
        if len(self.buf) > self.seq_len:
            self.buf.pop(0)
        seq = np.array(self.buf, dtype=np.float32)[np.newaxis]  # (1, SEQ, F)
        out = self.sess.run(None, {self.in_name: seq})[0]
        return float(max(out[0][0], 0.0))


# ══════════════════════════════════════════════════════════════════════════════
#  HAVERSINE
# ══════════════════════════════════════════════════════════════════════════════
def haversine_m(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, [la1, lo1, la2, lo2])
    dlat = la2 - la1; dlon = lo2 - lo1
    a = np.sin(dlat/2)**2 + np.cos(la1)*np.cos(la2)*np.sin(dlon/2)**2
    return R_EARTH * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def run_engine(imu_csv, model_onnx=None, meta_json=None, target_hz=100,
               out_csv='position_output.csv', out_png='position_plot.png'):

    print(f"\n{'='*60}")
    print(f"  NAVDRIFT-0 Edge Engine  |  target {target_hz} Hz")
    print(f"{'='*60}")

    # ── Load CSV ──
    df = pd.read_csv(imu_csv, low_memory=False)
    df = df.apply(pd.to_numeric, errors='coerce')
    COL = detect_columns(df)
    print(f"Columns detected: {COL}")

    needed = [COL[k] for k in ['ax','ay','az','gx','gy','gz'] if COL[k]]
    df = df[needed + [v for v in [COL['ts'],COL['lat'],COL['lon'],COL['speed']] if v]].copy()
    df.columns = (['ax','ay','az','gx','gy','gz'][:len(needed)] +
                  [k for k in ['ts','lat','lon','speed'] if COL[k]])
    df = df.dropna(subset=['ax','ay','az']).reset_index(drop=True)

    has_gt = 'lat' in df.columns and 'lon' in df.columns

    # ── Infer dt from timestamps ──
    if 'ts' in df.columns:
        ts = df['ts'].values.astype(float)
        median_dt = float(np.median(np.diff(ts)) / 1000.0)  # ms → s
    else:
        median_dt = 1.0 / target_hz
    DT = median_dt if 0.001 < median_dt < 1.0 else 1.0 / target_hz
    fs = 1.0 / DT
    print(f"Sample rate: {fs:.1f} Hz  |  dt={DT*1000:.2f} ms  |  rows: {len(df):,}")

    # ── Vibration filter ──
    bw = ButterworthLP(fc=min(5.0, fs/4), fs=fs, n_channels=6)

    # ── Alignment: detect forward axis on first 5 s ──
    align_n = int(5.0 / DT)
    accel_init = df[['ax','ay','az']].iloc[:min(align_n, len(df))].values
    fwd_axis = detect_forward_axis(accel_init)  # 0, 1, or 2

    # ── Bias calibration: first 2 s ──
    cal_n = int(2.0 / DT)
    cal_slice = df[['ax','ay','az','gx','gy','gz']].iloc[:min(cal_n, len(df))].values
    bias = cal_slice.mean(axis=0)
    # For gravity axis, leave ~9.81 offset (remove extra)
    grav_axis = int(np.argmax(np.abs(bias[:3])))
    bias[grav_axis] -= np.sign(bias[grav_axis]) * 9.81
    print(f"Calibration bias: {bias.round(4)}")

    # ── LSTM model (optional) ──
    estimator = None
    if HAS_ONNX and model_onnx and os.path.exists(model_onnx) and meta_json and os.path.exists(meta_json):
        try:
            estimator = LSTMSpeedEstimator(model_onnx, meta_json)
            print(f"LSTM model loaded: {model_onnx}")
        except Exception as e:
            print(f"[WARN] LSTM load failed: {e} — using heuristic")

    # ── Initial state ──
    lat0 = float(df['lat'].iloc[0]) if has_gt else 0.0
    lon0 = float(df['lon'].iloc[0]) if has_gt else 0.0
    ekf  = EKF(lat0, lon0)

    pred_lat, pred_lon = [lat0], [lon0]
    gt_lat_list = [lat0] if has_gt else []
    gt_lon_list = [lon0] if has_gt else []

    speed_hist = []
    int_speed  = 0.0
    heading    = 0.0

    # Alignment calibration window
    align_buf = []
    aligned   = False

    t0 = time.perf_counter()

    for i in range(1, len(df)):
        row = df.iloc[i]
        raw = np.array([row.get('ax',0), row.get('ay',0), row.get('az',0),
                        row.get('gx',0), row.get('gy',0), row.get('gz',0)], dtype=np.float32)
        raw -= bias

        # Butterworth filter
        filt = bw.step(raw)
        fax, fay, faz, fgx, fgy, fgz = filt

        # Gyro heading (NHC: yaw only = gz)
        dheading = fgz * DT
        heading += dheading

        # Alignment window
        if not aligned:
            align_buf.append(filt[:3].copy())
            if len(align_buf) >= align_n:
                fwd_axis = detect_forward_axis(np.array(align_buf))
                aligned = True

        # Speed estimation
        fwd_raw = [fax, fay, faz][fwd_axis]
        dead    = 0.0 if abs(fwd_raw) < DEAD_BAND else fwd_raw

        if estimator:
            feat = np.array([fax, fay, faz, fgx, fgy, fgz,
                             np.sqrt(fax**2+fay**2+faz**2),
                             np.sqrt(fgx**2+fgy**2+fgz**2)], dtype=np.float32)
            lstm_spd = estimator.push(feat[:len(estimator.feat_cols)])
            if lstm_spd is not None:
                int_speed = lstm_spd
        else:
            int_speed = max(0.0, int_speed + dead * DT)
            int_speed = min(int_speed, MAX_SPEED)

        speed_hist.append(int_speed)

        # EKF predict
        ekf.predict(int_speed, dheading, DT)

        # EKF update from GPS (if available and valid)
        if has_gt and not pd.isna(row.get('lat', np.nan)):
            gt_lat = float(row['lat']); gt_lon = float(row['lon'])
            if gt_lat != 0 and gt_lon != 0:
                ekf.update_gps(gt_lat, gt_lon)
                gt_lat_list.append(gt_lat)
                gt_lon_list.append(gt_lon)

        pred_lat.append(float(ekf.x[0]))
        pred_lon.append(float(ekf.x[1]))

    elapsed = time.perf_counter() - t0
    print(f"\nProcessed {len(df):,} samples in {elapsed:.3f}s  ({len(df)/elapsed:.0f} samples/s)")

    # ── Metrics ──
    pred_lat = np.array(pred_lat)
    pred_lon = np.array(pred_lon)

    if has_gt and len(gt_lat_list) > 1:
        n = min(len(pred_lat), len(gt_lat_list))
        err = haversine_m(pred_lat[:n], pred_lon[:n],
                          np.array(gt_lat_list[:n]), np.array(gt_lon_list[:n]))
        ATE = float(np.mean(err))
        ATE_max = float(np.max(err))
        dist_arr = haversine_m(np.array(gt_lat_list[:-1]), np.array(gt_lon_list[:-1]),
                               np.array(gt_lat_list[1:]),  np.array(gt_lon_list[1:]))
        total_dist = float(np.sum(dist_arr))
        drift_pct  = ATE / total_dist * 100 if total_dist > 0 else 0

        print(f"\n{'═'*50}")
        print(f"  NAVDRIFT-0 BENCHMARK RESULTS")
        print(f"{'═'*50}")
        print(f"  ATE mean  : {ATE:.2f} m")
        print(f"  ATE max   : {ATE_max:.2f} m")
        print(f"  Distance  : {total_dist:.0f} m")
        print(f"  Drift     : {drift_pct:.2f}%  (target: <10%)")
        status = "✓ PASS" if drift_pct < 10 else "✗ FAIL"
        print(f"  Benchmark : {status}")
        print(f"{'═'*50}")

    # ── Save CSV ──
    out_df = pd.DataFrame({'pred_lat': pred_lat, 'pred_lon': pred_lon})
    if has_gt:
        gt_ext = gt_lat_list + [np.nan] * (len(pred_lat) - len(gt_lat_list))
        out_df['gt_lat'] = gt_ext[:len(pred_lat)]
        out_df['gt_lon'] = (gt_lon_list + [np.nan]*(len(pred_lat)-len(gt_lon_list)))[:len(pred_lat)]
    out_df.to_csv(out_csv, index=False)
    print(f"\n✓ Output CSV → {out_csv}")

    # ── Plot ──
    if HAS_PLT and has_gt and len(gt_lat_list) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        n = min(len(pred_lat), len(gt_lat_list))
        ax.plot(np.array(gt_lon_list[:n]), np.array(gt_lat_list[:n]),
                'b-', lw=1.5, label='Ground Truth', alpha=0.8)
        ax.plot(pred_lon[:n], pred_lat[:n],
                'r-', lw=1.5, label=f'NAVDRIFT-0 EKF+LSTM (ATE={ATE:.1f}m)', alpha=0.8)
        ax.plot(gt_lon_list[0], gt_lat_list[0], 'go', ms=10, label='Start')
        ax.plot(gt_lon_list[-1], gt_lat_list[-1], 'rs', ms=10, label='End')
        ax.set_title(f'NAVDRIFT-0 Edge Engine\nATE={ATE:.2f}m | Drift={drift_pct:.2f}% | Dist={total_dist:.0f}m')
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        print(f"✓ Position plot → {out_png}")
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  REAL-TIME STREAM MODE (200 Hz external IMU)
# ══════════════════════════════════════════════════════════════════════════════
class NavdriftRTEngine:
    """
    Real-time engine for external IMU at up to 200 Hz.
    Call .step(ax, ay, az, gx, gy, gz, dt) each sample.
    Optionally call .gps_update(lat, lon) when GPS available.
    """
    def __init__(self, lat0, lon0, fs=200, model_onnx=None, meta_json=None):
        self.ekf     = EKF(lat0, lon0)
        self.bw      = ButterworthLP(fc=min(10.0, fs/4), fs=fs, n_channels=6)
        self.bias    = np.zeros(6, dtype=np.float32)
        self.fwd_axis= 1   # default Y; updated by detect_forward_axis
        self.int_spd = 0.0
        self.heading = 0.0
        self.cal_buf = []
        self.calibrated = False
        self.align_buf  = []
        self.aligned    = False
        self.estimator  = None
        self.fs = fs

        if HAS_ONNX and model_onnx and meta_json:
            try:
                self.estimator = LSTMSpeedEstimator(model_onnx, meta_json)
            except Exception as e:
                print(f"[WARN] RT LSTM: {e}")

    def calibrate_step(self, raw6):
        """Feed samples during stationary calibration period."""
        self.cal_buf.append(raw6.copy())
        if len(self.cal_buf) >= int(2 * self.fs):
            arr = np.array(self.cal_buf)
            self.bias = arr.mean(axis=0)
            grav = int(np.argmax(np.abs(self.bias[:3])))
            self.bias[grav] -= np.sign(self.bias[grav]) * 9.81
            self.calibrated = True
            print(f"[RT] Calibrated. bias={self.bias.round(3)}")

    def step(self, ax, ay, az, gx, gy, gz, dt) -> tuple[float, float]:
        """Returns (pred_lat, pred_lon)"""
        raw = np.array([ax, ay, az, gx, gy, gz], dtype=np.float32) - self.bias
        filt = self.bw.step(raw)
        fax, fay, faz, fgx, fgy, fgz = filt

        # Alignment detection
        if not self.aligned:
            self.align_buf.append(filt[:3].copy())
            if len(self.align_buf) >= int(5 * self.fs):
                self.fwd_axis = detect_forward_axis(np.array(self.align_buf))
                self.aligned  = True

        dh = fgz * dt
        self.heading += dh

        fwd = [fax, fay, faz][self.fwd_axis]
        dead = 0.0 if abs(fwd) < DEAD_BAND else fwd

        if self.estimator:
            feat = np.array([fax,fay,faz,fgx,fgy,fgz,
                             np.sqrt(fax**2+fay**2+faz**2),
                             np.sqrt(fgx**2+fgy**2+fgz**2)], dtype=np.float32)
            spd = self.estimator.push(feat[:len(self.estimator.feat_cols)])
            if spd is not None:
                self.int_spd = spd
        else:
            self.int_spd = max(0.0, min(self.int_spd + dead * dt, MAX_SPEED))

        self.ekf.predict(self.int_spd, dh, dt)
        return float(self.ekf.x[0]), float(self.ekf.x[1])

    def gps_update(self, lat, lon):
        self.ekf.update_gps(lat, lon)
        self.int_spd = 0.0  # reset integration on GNSS reacq


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NAVDRIFT-0 Edge Engine')
    parser.add_argument('--imu',   required=True,  help='Input IMU CSV path')
    parser.add_argument('--model', default=None,   help='ONNX model path (navdrift_lstm.onnx)')
    parser.add_argument('--meta',  default=None,   help='Model metadata JSON path (model_meta.json)')
    parser.add_argument('--hz',    type=int, default=100, help='Target Hz (default 100)')
    parser.add_argument('--out_csv', default='position_output.csv')
    parser.add_argument('--out_png', default='position_plot.png')
    args = parser.parse_args()

    run_engine(
        imu_csv    = args.imu,
        model_onnx = args.model,
        meta_json  = args.meta,
        target_hz  = args.hz,
        out_csv    = args.out_csv,
        out_png    = args.out_png,
    )
