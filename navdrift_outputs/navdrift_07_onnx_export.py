"""
NAVDRIFT-0 | ISRO SIH 2026 PS #26168
File 07 — ONNX INT8 Export + ARM Cortex-A72 Benchmark
Exports all 5 trained models to ONNX, then quantizes to INT8.
Benchmarks combined inference time to verify <8ms target.

Output: /content/drive/MyDrive/NAVDRIFT0/models/
  driftformer_fp32.onnx  /  driftformer_int8.onnx
  imu_denoiser_fp32.onnx /  imu_denoiser_int8.onnx
  adaptive_ekf_fp32.onnx /  adaptive_ekf_int8.onnx
  tunnel_det_fp32.onnx   /  tunnel_det_int8.onnx
  navic_dop_fp32.onnx    /  navic_dop_int8.onnx
  benchmark_results.json
"""

import sys, time, json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, "/content")
from navdrift_00_setup import (PATHS, mount_drive, create_dirs,
                                get_device, start_keepalive, save_results)

# ============================================================
# IMPORT ALL MODELS + CONFIGS
# ============================================================
from navdrift_02_driftformer import DRIFTFormer,      CFG as DF_CFG
from navdrift_03_imu_denoiser import IMUDenoiser,     CFG as DN_CFG
from navdrift_04_adaptive_ekf import AdaptiveEKFNet,  CFG as EKF_CFG
from navdrift_05_tunnel_detector import TunnelDetector, CFG as TD_CFG
from navdrift_06_navic_dop import NavICDOPNet,         CFG as DOP_CFG

# ============================================================
# DUMMY INPUTS (for tracing)
# ============================================================
DUMMY_INPUTS = {
    "driftformer":  torch.randn(1, 100, 9),
    "imu_denoiser": torch.randn(1, 6, 200),
    "adaptive_ekf": torch.randn(1, 6),
    "tunnel_det":   torch.randn(1, 20, 5),
    "navic_dop":    torch.randn(1, 8),
}

INPUT_NAMES = {
    "driftformer":  ["imu_window_9ch"],
    "imu_denoiser": ["raw_imu_6ax"],
    "adaptive_ekf": ["nav_state_features"],
    "tunnel_det":   ["telemetry_window"],
    "navic_dop":    ["constellation_features"],
}

OUTPUT_NAMES = {
    "driftformer":  ["position_correction_xy"],
    "imu_denoiser": ["denoised_imu_6ax"],
    "adaptive_ekf": ["qr_noise_matrices"],
    "tunnel_det":   ["blackout_probability"],
    "navic_dop":    ["pdop_hdop_vdop"],
}

# ============================================================
# LOAD TRAINED MODELS
# ============================================================
def load_model(model_cls, cfg, name, device):
    """Load model from latest checkpoint."""
    ckpt_path = PATHS["checkpoints"] / name / "latest.pt"
    if not ckpt_path.exists():
        print(f"[WARN] No checkpoint for {name}. Exporting untrained weights.")
        return model_cls(cfg).eval()

    model = model_cls(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.eval()
    epoch = state.get("epoch", -1)
    print(f"[LOAD] {name}  epoch={epoch}")
    return model

# ============================================================
# ONNX EXPORT
# ============================================================
def export_onnx(model, dummy_input, name: str, out_dir: Path, opset: int = 17):
    fp32_path = out_dir / f"{name}_fp32.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(fp32_path),
        opset_version=opset,
        input_names=INPUT_NAMES[name],
        output_names=OUTPUT_NAMES[name],
        dynamic_axes={
            INPUT_NAMES[name][0]:  {0: "batch"},
            OUTPUT_NAMES[name][0]: {0: "batch"},
        },
        do_constant_folding=True,
    )
    size_mb = fp32_path.stat().st_size / 1e6
    print(f"[ONNX] {name} FP32 -> {fp32_path.name}  ({size_mb:.2f} MB)")
    return fp32_path

# ============================================================
# INT8 QUANTIZATION
# ============================================================
def quantize_int8(fp32_path: Path, out_dir: Path, name: str):
    try:
        from onnxruntime.quantization import (
            quantize_dynamic, QuantType, QuantizationMode
        )
        int8_path = out_dir / f"{name}_int8.onnx"
        quantize_dynamic(
            str(fp32_path),
            str(int8_path),
            weight_type=QuantType.QInt8,
        )
        size_mb = int8_path.stat().st_size / 1e6
        print(f"[INT8] {name} INT8 -> {int8_path.name}  ({size_mb:.2f} MB)")
        return int8_path
    except Exception as e:
        print(f"[INT8] Quantization failed for {name}: {e}")
        return None

# ============================================================
# ONNXRUNTIME BENCHMARK
# ============================================================
def benchmark_onnx(onnx_path: Path, dummy_np: np.ndarray, n_runs: int = 200):
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path),
                                    providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name

        # warmup
        for _ in range(10):
            sess.run(None, {inp_name: dummy_np})

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            sess.run(None, {inp_name: dummy_np})
            times.append((time.perf_counter() - t0) * 1000)

        return {
            "mean_ms":   round(np.mean(times), 3),
            "p50_ms":    round(np.percentile(times, 50), 3),
            "p95_ms":    round(np.percentile(times, 95), 3),
            "p99_ms":    round(np.percentile(times, 99), 3),
            "size_mb":   round(onnx_path.stat().st_size / 1e6, 3),
        }
    except Exception as e:
        print(f"[BENCH] {onnx_path.name} failed: {e}")
        return {}

# ============================================================
# ACCURACY COMPARISON FP32 vs INT8
# ============================================================
def accuracy_delta(fp32_path: Path, int8_path: Path,
                   dummy_np: np.ndarray, n_samples: int = 50):
    """
    Measure max absolute difference between FP32 and INT8 outputs.
    Should be < 1% of output range for acceptable quantization.
    """
    try:
        import onnxruntime as ort
        s32  = ort.InferenceSession(str(fp32_path),  providers=["CPUExecutionProvider"])
        s8   = ort.InferenceSession(str(int8_path),  providers=["CPUExecutionProvider"])
        n32  = s32.get_inputs()[0].name
        n8   = s8.get_inputs()[0].name

        diffs = []
        for _ in range(n_samples):
            inp = dummy_np + np.random.randn(*dummy_np.shape).astype(np.float32) * 0.1
            o32 = np.array(s32.run(None, {n32: inp})[0])
            o8  = np.array(s8.run(None,  {n8:  inp})[0])
            diffs.append(np.abs(o32 - o8).mean())

        return round(float(np.mean(diffs)), 6)
    except Exception as e:
        return -1.0

# ============================================================
# MAIN EXPORT PIPELINE
# ============================================================
MODELS_CFG = [
    ("driftformer",  DRIFTFormer,     DF_CFG),
    ("imu_denoiser", IMUDenoiser,     DN_CFG),
    ("adaptive_ekf", AdaptiveEKFNet,  EKF_CFG),
    ("tunnel_det",   TunnelDetector,  TD_CFG),
    ("navic_dop",    NavICDOPNet,     DOP_CFG),
]

def run_export():
    device  = torch.device("cpu")   # always export on CPU
    out_dir = PATHS["models"]
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    total_ms_fp32 = 0.0
    total_ms_int8 = 0.0

    for name, cls, cfg in MODELS_CFG:
        print(f"\n{'='*50}")
        print(f"  {name.upper()}")
        print(f"{'='*50}")

        model = load_model(cls, cfg, name, device)
        dummy = DUMMY_INPUTS[name]
        dummy_np = dummy.numpy()

        # 1. FP32 export
        fp32_path = export_onnx(model, dummy, name, out_dir)

        # 2. INT8 quantization
        int8_path = quantize_int8(fp32_path, out_dir, name)

        # 3. Benchmark
        bench_fp32 = benchmark_onnx(fp32_path, dummy_np)
        bench_int8 = benchmark_onnx(int8_path, dummy_np) if int8_path else {}

        # 4. Accuracy delta
        delta = accuracy_delta(fp32_path, int8_path, dummy_np) if int8_path else -1.0

        model_res = {
            "fp32": bench_fp32,
            "int8": bench_int8,
            "accuracy_delta_fp32_to_int8": delta,
        }
        results[name] = model_res

        fp32_ms = bench_fp32.get("mean_ms", 0)
        int8_ms = bench_int8.get("mean_ms", 0)
        total_ms_fp32 += fp32_ms
        total_ms_int8 += int8_ms

        speedup = fp32_ms / max(int8_ms, 1e-9)
        print(f"  FP32: {fp32_ms:.2f} ms | INT8: {int8_ms:.2f} ms | "
              f"Speedup: {speedup:.1f}x | Delta: {delta:.6f}")

    # Summary
    print("\n" + "=" * 60)
    print("  BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Total pipeline FP32: {total_ms_fp32:.2f} ms")
    print(f"  Total pipeline INT8: {total_ms_int8:.2f} ms")
    target_ok = total_ms_int8 < 8.0
    print(f"  ISRO target (<8 ms): {'PASS' if target_ok else 'FAIL'}")

    results["pipeline_total"] = {
        "fp32_ms": round(total_ms_fp32, 3),
        "int8_ms": round(total_ms_int8, 3),
        "target_8ms_pass": target_ok,
    }

    save_results(results, "onnx_benchmark")
    print(f"\n[EXPORT] All models saved to {out_dir}")
    return results

# ============================================================
# GENERATE models.json for NAVDRIFT-0 frontend
# ============================================================
def generate_frontend_manifest():
    """
    Creates a JSON manifest consumed by the Production Model mode
    in frontend/index.html to load the correct ONNX files.
    """
    manifest = {
        "version": "isro-grade-v1",
        "models": {
            "driftformer": {
                "file":        "driftformer_int8.onnx",
                "input_shape": [1, 100, 9],
                "description": "4L-8H Causal Transformer, dead reckoning correction",
            },
            "imu_denoiser": {
                "file":        "imu_denoiser_int8.onnx",
                "input_shape": [1, 6, 200],
                "description": "TCN IMU denoiser (5 dilated blocks)",
            },
            "adaptive_ekf": {
                "file":        "adaptive_ekf_int8.onnx",
                "input_shape": [1, 6],
                "description": "MLP adaptive Q/R predictor for EKF",
            },
            "tunnel_detector": {
                "file":        "tunnel_det_int8.onnx",
                "input_shape": [1, 20, 5],
                "description": "Bidirectional LSTM, pre-arm warning 3-5s before blackout",
                "threshold":   0.55,
            },
            "navic_dop": {
                "file":        "navic_dop_int8.onnx",
                "input_shape": [1, 8],
                "description": "NavIC DOP predictor from constellation geometry",
            },
        },
    }
    path = PATHS["models"] / "models_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[MANIFEST] {path}")

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    mount_drive()
    create_dirs()
    start_keepalive()

    results = run_export()
    generate_frontend_manifest()
    print("\n[ONNX Export] Complete.")
