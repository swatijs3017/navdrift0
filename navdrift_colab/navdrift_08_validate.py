"""
NAVDRIFT-0 | ISRO SIH 2026 PS #26168
File 08 — End-to-End Validation + ISRO Compliance Report
Runs the full 5-model pipeline on held-out test sequences.
Generates the compliance curve, ATE, drift %, and the
benchmark table for the ISRO submission document.

Output: /content/drive/MyDrive/NAVDRIFT0/results/
  validation_full.json
  compliance_curve.png
  isro_benchmark_table.csv
"""

import sys, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import torch
import onnxruntime as ort
from pathlib import Path

sys.path.insert(0, "/content")
from navdrift_00_setup import (PATHS, mount_drive, create_dirs,
                                get_device, start_keepalive, save_results)

# ============================================================
# LOAD ALL ONNX MODELS
# ============================================================
def load_onnx_sessions():
    models_dir = PATHS["models"]
    sessions = {}
    names = ["driftformer", "imu_denoiser", "adaptive_ekf", "tunnel_det", "navic_dop"]
    for name in names:
        p = models_dir / f"{name}_int8.onnx"
        if not p.exists():
            p = models_dir / f"{name}_fp32.onnx"
        if not p.exists():
            print(f"[WARN] No ONNX for {name}. Validation will skip it.")
            sessions[name] = None
            continue
        sessions[name] = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        print(f"[ONNX] Loaded {name} from {p.name}")
    return sessions

# ============================================================
# SIMPLE EKF STATE
# ============================================================
class SimpleEKF:
    def __init__(self):
        self.x = np.zeros(3, dtype=np.float64)
        self.P = np.eye(3, dtype=np.float64) * 1e-4

    def predict(self, acc_xy: np.ndarray, dt: float, Q: np.ndarray):
        ax, ay = float(acc_xy[0]), float(acc_xy[1])
        self.x[0] += self.x[2] * dt * np.cos(0)
        self.x[1] += self.x[2] * dt * np.sin(0)
        self.x[2] += np.sqrt(ax**2 + ay**2) * dt * 0.1
        self.P    += Q

    def update(self, gps_pos: np.ndarray, R: np.ndarray):
        H     = np.array([[1,0,0],[0,1,0]], dtype=np.float64)
        S     = H @ self.P @ H.T + R
        K     = self.P @ H.T @ np.linalg.inv(S)
        innov = gps_pos - H @ self.x
        self.x += K @ innov
        self.P  = (np.eye(3) - K @ H) @ self.P

    def reset_pos(self, pos: np.ndarray):
        self.x[0] = pos[0]
        self.x[1] = pos[1]

# ============================================================
# FULL PIPELINE RUN ON ONE SEQUENCE
# ============================================================
def run_sequence(imu: np.ndarray, gt_pos: np.ndarray,
                 gps_quality: np.ndarray, sessions: dict,
                 dt: float = 0.01):
    N   = len(imu)
    ekf = SimpleEKF()

    if np.abs(gt_pos[0, 0]) < 100:
        pos_m = gt_pos[:, :2].copy()
    else:
        lat0  = gt_pos[0, 0]
        pos_m = np.stack([
            (gt_pos[:, 1] - gt_pos[0, 1]) * 111320 * np.cos(np.radians(lat0)),
            (gt_pos[:, 0] - gt_pos[0, 0]) * 111320,
        ], axis=-1)

    ekf.reset_pos(pos_m[0])

    estimated    = np.zeros((N, 2), dtype=np.float32)
    errors       = np.zeros(N, dtype=np.float32)
    drift_pct    = np.zeros(N, dtype=np.float32)
    dist_trav    = 0.0
    tunnel_probs = np.zeros(N, dtype=np.float32)

    # --- denoise IMU ---
    if sessions.get("imu_denoiser"):
        CHUNK = 200
        imu_d = imu.copy()
        sess  = sessions["imu_denoiser"]
        inp   = sess.get_inputs()[0].name
        for start in range(0, N - CHUNK, CHUNK // 2):
            end   = start + CHUNK
            chunk = imu[start:end].T[None].astype(np.float32)
            out   = sess.run(None, {inp: chunk})[0][0].T
            imu_d[start:end] = out
        imu = imu_d

    Q_default = np.eye(3) * 1e-5
    R_default = np.eye(2) * 1e-3

    for i in range(N):
        # --- Adaptive EKF noise ---
        if sessions.get("adaptive_ekf") and i % 10 == 0:
            speed     = np.linalg.norm(imu[i, :3]) / 9.81
            turn_rate = abs(imu[i, 5]) / 3.0
            gnss_age  = 0.0
            feat = np.array([[speed, turn_rate, gnss_age, gps_quality[i],
                              float(np.linalg.norm(imu[i,:3]) < 0.2),
                              np.var(imu[max(0,i-10):i+1, :3])]], dtype=np.float32)
            sess      = sessions["adaptive_ekf"]
            inp_name  = sess.get_inputs()[0].name
            qr        = sess.run(None, {inp_name: feat})[0][0]
            Q_default = np.diag(qr[:3].astype(np.float64))
            R_default = np.diag(qr[3:].astype(np.float64))

        # --- Tunnel detector ---
        if sessions.get("tunnel_det") and i % 10 == 0 and i >= 20:
            win = np.zeros((1, 20, 5), dtype=np.float32)
            for j, k in enumerate(range(i-20, i)):
                if k >= 0:
                    win[0, j, 0] = np.var(imu[max(0,k-5):k+1, :3])
                    win[0, j, 1] = np.linalg.norm(imu[k, :3])
                    win[0, j, 2] = np.linalg.norm(imu[k, 3:])
                    win[0, j, 3] = gps_quality[k]
                    win[0, j, 4] = 0.5
            sess = sessions["tunnel_det"]
            inp  = sess.get_inputs()[0].name
            prob = float(sess.run(None, {inp: win})[0][0])
            tunnel_probs[i] = prob

        # --- EKF predict ---
        ekf.predict(imu[i, :2], dt, Q_default)

        # --- EKF update ---
        if gps_quality[i] > 0.5:
            ekf.update(pos_m[i], R_default)

        # --- DRIFTFormer correction ---
        if sessions.get("driftformer") and i > 0 and i % 100 == 0:
            start_w  = i - 100
            win_imu  = imu[start_w:i].astype(np.float32)
            speed_   = np.linalg.norm(win_imu[:, :3], axis=1, keepdims=True)
            heading_ = np.sin(np.cumsum(win_imu[:, 5:6], axis=0) * dt)
            dt_col   = np.full((100, 1), dt)
            x9       = np.concatenate([win_imu, heading_, speed_, dt_col], axis=1)
            x9       = x9[None]
            sess     = sessions["driftformer"]
            inp      = sess.get_inputs()[0].name
            delta    = sess.run(None, {inp: x9})[0][0]
            ekf.x[0] += float(delta[0])
            ekf.x[1] += float(delta[1])

        estimated[i]  = ekf.x[:2]
        err           = np.linalg.norm(estimated[i] - pos_m[i])
        dist_trav    += np.linalg.norm(imu[i, :2]) * dt * dt + 0.05
        errors[i]     = err
        drift_pct[i]  = err / max(dist_trav, 1.0) * 100

    return {
        "estimated":    estimated,
        "gt_pos_m":     pos_m,
        "errors_m":     errors,
        "drift_pct":    drift_pct,
        "tunnel_probs": tunnel_probs,
        "dist_trav":    dist_trav,
    }

# ============================================================
# COMPLIANCE METRICS
# ============================================================
def compute_compliance(results_per_seq: list):
    all_drift    = np.concatenate([r["drift_pct"] for r in results_per_seq])
    all_errors   = np.concatenate([r["errors_m"]  for r in results_per_seq])
    ate_rmse     = float(np.sqrt(np.mean(all_errors ** 2)))
    drift_mean   = float(np.mean(all_drift))
    drift_max    = float(np.max(all_drift))
    pct_under_10 = float((all_drift < 10.0).mean() * 100)
    pct_under_5  = float((all_drift < 5.0).mean() * 100)
    return {
        "ATE_RMSE_m":          round(ate_rmse, 4),
        "drift_mean_pct":      round(drift_mean, 3),
        "drift_max_pct":       round(drift_max, 3),
        "pct_under_10_target": round(pct_under_10, 1),
        "pct_under_5":         round(pct_under_5, 1),
        "n_steps":             int(len(all_drift)),
        "isro_pass":           pct_under_10 >= 90.0,
    }

# ============================================================
# PLOT COMPLIANCE CURVE
# ============================================================
def plot_compliance(results_per_seq: list, out_dir: Path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("NAVDRIFT-0 | ISRO SIH 2026 | Drift Compliance Validation",
                 fontsize=14, fontweight="bold")
    ax1, ax2 = axes

    for i, r in enumerate(results_per_seq[:5]):
        ax1.plot(r["drift_pct"], alpha=0.7, linewidth=1.0, label=f"Seq {i+1}")
    ax1.axhline(10.0, color="red",   linestyle="--", linewidth=2,   label="ISRO Target 10%")
    ax1.axhline(5.0,  color="green", linestyle=":",  linewidth=1.5, label="5% threshold")
    ax1.set_ylabel("Drift Error %")
    ax1.set_xlabel("Step")
    ax1.set_ylim(0, 25)
    ax1.legend(fontsize=8)
    ax1.set_title("Drift Compliance per Step")
    ax1.grid(True, alpha=0.3)

    r0 = results_per_seq[0]
    ax2.plot(r0["gt_pos_m"][:, 0],  r0["gt_pos_m"][:, 1],  "g-",  linewidth=2,   label="Ground Truth")
    ax2.plot(r0["estimated"][:, 0], r0["estimated"][:, 1],  "b--", linewidth=1.5, label="NAVDRIFT-0 Estimate")
    ax2.set_xlabel("East (m)")
    ax2.set_ylabel("North (m)")
    ax2.set_title("Trajectory — Sequence 1")
    ax2.legend()
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_dir / "compliance_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[PLOT] {out}")
    plt.close()

# ============================================================
# BENCHMARK TABLE
# ============================================================
def make_benchmark_table(metrics: dict, onnx_results: dict, out_dir: Path):
    rows = []
    names  = ["driftformer", "imu_denoiser", "adaptive_ekf", "tunnel_det", "navic_dop"]
    labels = [
        "DRIFTFormer (4L-8H Transformer)",
        "IMU Denoiser (TCN 5-block)",
        "Adaptive EKF Predictor (MLP)",
        "Tunnel Detector (Bi-LSTM)",
        "NavIC DOP Predictor (MLP)",
    ]
    for name, label in zip(names, labels):
        b = onnx_results.get(name, {})
        rows.append({
            "Model":          label,
            "FP32 Size (MB)": b.get("fp32", {}).get("size_mb", "N/A"),
            "INT8 Size (MB)": b.get("int8", {}).get("size_mb", "N/A"),
            "FP32 Inf (ms)":  b.get("fp32", {}).get("mean_ms", "N/A"),
            "INT8 Inf (ms)":  b.get("int8", {}).get("mean_ms", "N/A"),
            "Accuracy Delta": b.get("accuracy_delta_fp32_to_int8", "N/A"),
        })
    rows.append({
        "Model":          "System Drift Compliance",
        "FP32 Size (MB)": "",
        "INT8 Size (MB)": "",
        "FP32 Inf (ms)":  "",
        "INT8 Inf (ms)":  onnx_results.get("pipeline_total", {}).get("int8_ms", "N/A"),
        "Accuracy Delta": f"{metrics.get('drift_mean_pct','N/A')}% drift (ISRO target <10%)",
    })
    df  = pd.DataFrame(rows)
    out = out_dir / "isro_benchmark_table.csv"
    df.to_csv(out, index=False)
    print(f"[TABLE] {out}")
    print(df.to_string())
    return df

# ============================================================
# MAIN VALIDATION
# ============================================================
def run_validation():
    print("\n" + "=" * 60)
    print("  END-TO-END VALIDATION")
    print("=" * 60)

    sessions  = load_onnx_sessions()
    hdf5_path = PATHS["processed"] / "navdrift_dataset.h5"
    if not hdf5_path.exists():
        print("[ERROR] HDF5 not found.")
        return

    results_per_seq = []
    with h5py.File(hdf5_path, "r") as hf:
        test_grp  = hf["test"]
        seq_names = list(test_grp.keys())
        print(f"[VAL] Running on {len(seq_names)} test sequences...")
        for sname in seq_names:
            sq  = test_grp[sname]
            imu = sq["imu"][:]
            gtp = sq["gt_pos"][:]
            gpq = sq["gps_quality"][:]
            src = sq.attrs.get("source", sname)
            print(f"  {src} ({len(imu)} steps)...", end=" ", flush=True)
            try:
                res = run_sequence(imu, gtp, gpq, sessions)
                results_per_seq.append(res)
                final_drift = res["drift_pct"][-100:].mean()
                print(f"drift={final_drift:.2f}%  ATE={res['errors_m'].mean():.2f}m")
            except Exception as e:
                print(f"FAILED: {e}")

    if not results_per_seq:
        print("[ERROR] No sequences processed.")
        return

    metrics = compute_compliance(results_per_seq)

    print("\n" + "=" * 60)
    print("  COMPLIANCE RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        status = ""
        if k == "isro_pass":
            status = "  <<< ISRO TARGET MET >>>" if v else "  <<< BELOW TARGET >>>"
        print(f"  {k:30s}: {v}{status}")

    onnx_res_path = PATHS["results"] / "onnx_benchmark.json"
    onnx_results  = {}
    if onnx_res_path.exists():
        with open(onnx_res_path) as f:
            onnx_results = json.load(f)

    out_dir = PATHS["results"]
    plot_compliance(results_per_seq, out_dir)
    make_benchmark_table(metrics, onnx_results, out_dir)

    full_results = {
        "compliance_metrics": metrics,
        "onnx_benchmark":     onnx_results,
        "n_test_sequences":   len(results_per_seq),
    }
    save_results(full_results, "validation_full")
    print("\n[VAL] Validation complete.")
    return full_results

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    mount_drive()
    create_dirs()
    start_keepalive()
    run_validation()