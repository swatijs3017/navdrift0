"""
edge/model_inference_bench.py — NAVDRIFT-0 offline ONNX inference benchmark
ISRO SIH 2026 PS #26168

A read-only, offline benchmark harness for the repository's existing ONNX
models. It never writes to `frontend/` or `models/`; it only opens the
already-committed `.onnx` files with onnxruntime and runs real forward
inference, measuring real wall-clock latency on this machine's CPU. This is
a SOFTWARE-ONLY benchmark: no claim of mobile-device, embedded, or hardware
latency is made anywhere here, and every result dict says so explicitly.

Why this exists: during a repository audit, two facts were found and are
worth having a reproducible tool to re-check:

  1. The stripped `models/*.onnx` files (driftformer_fp32.onnx,
     adaptive_ekf_fp32.onnx, tunnel_det_fp32.onnx, navic_dop_fp32.onnx) each
     reference an external `<name>.onnx.data` weights file that is NOT
     present in this repository, so onnxruntime cannot load them as-is.
     `models/imu_denoiser_int8.onnx` is the one exception in that directory:
     it is fully self-contained and loads successfully.
  2. The full-weight copies actually served to the live mobile/desktop app,
     `frontend/models/*.onnx` (all five files, including the INT8 denoiser
     and the NavIC DOP predictor), ARE self-contained and load and run
     successfully via onnxruntime's Python CPU execution provider — the
     same real, currently-deployed weights the live app uses, inspected
     read-only.

This module benchmarks whichever of those files are actually loadable at
run time (it does not assume either fact above stays true; it re-checks
every time), and reports load failures honestly rather than skipping them
silently.

Nothing here is wired into or replaces the live mobile pipeline; this is an
offline, read-only, Python-side inspection tool only.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False


# (label, relative_path, input_name, input_shape_with_batch_1, description)
# Shapes and input names come directly from each model's own ONNX graph
# metadata (onnxruntime's get_inputs()), not guessed.
MODEL_SPECS: List[Tuple[str, str, str, Tuple[int, ...], str]] = [
    ("driftformer_frontend", "frontend/models/driftformer_fp32.onnx",
     "imu_window_9ch", (1, 100, 9), "DriftFormer position-correction transformer, live-deployed weights"),
    ("adaptive_ekf_frontend", "frontend/models/adaptive_ekf_fp32.onnx",
     "nav_state_features", (1, 6), "Adaptive EKF Q/R predictor MLP, live-deployed weights"),
    ("tunnel_det_frontend", "frontend/models/tunnel_det_fp32.onnx",
     "telemetry_window", (1, 20, 5), "Tunnel/blackout detector Bi-LSTM, live-deployed weights"),
    ("navic_dop_frontend", "frontend/models/navic_dop_fp32.onnx",
     "constellation_features", (1, 8), "NavIC DOP (PDOP/HDOP/VDOP) predictor MLP, live-deployed weights"),
    ("imu_denoiser_frontend", "frontend/models/imu_denoiser_int8.onnx",
     "raw_imu_6ax", (1, 6, 200), "IMU denoiser INT8 TCN, live-deployed weights (not invoked by "
                                 "frontend/mobile.html's live pipeline; see requirement_evidence.py)"),
    ("imu_denoiser_models_dir", "models/imu_denoiser_int8.onnx",
     "raw_imu_6ax", (1, 6, 200), "Same IMU denoiser INT8 TCN, models/ directory copy"),
    ("driftformer_models_dir", "models/driftformer_fp32.onnx",
     "imu_window_9ch", (1, 100, 9), "models/ directory copy — expected to fail to load (missing "
                                     "external .onnx.data weights file), included so that failure "
                                     "is reported explicitly rather than silently skipped"),
    ("adaptive_ekf_models_dir", "models/adaptive_ekf_fp32.onnx",
     "nav_state_features", (1, 6), "models/ directory copy — expected to fail to load (missing "
                                    "external .onnx.data weights file)"),
    ("tunnel_det_models_dir", "models/tunnel_det_fp32.onnx",
     "telemetry_window", (1, 20, 5), "models/ directory copy — expected to fail to load (missing "
                                      "external .onnx.data weights file)"),
    ("navic_dop_models_dir", "models/navic_dop_fp32.onnx",
     "constellation_features", (1, 8), "models/ directory copy — expected to fail to load (missing "
                                        "external .onnx.data weights file)"),
]


@dataclass
class ModelBenchResult:
    label: str
    path: str
    description: str
    loaded: bool
    load_error: Optional[str] = None
    input_name: Optional[str] = None
    input_shape: Optional[list] = None
    output_names: Optional[list] = None
    output_shapes: Optional[list] = None
    n_runs: int = 0
    latency_ms_mean: Optional[float] = None
    latency_ms_p50: Optional[float] = None
    latency_ms_p95: Optional[float] = None
    latency_ms_max: Optional[float] = None
    validation_type: str = "SOFTWARE_ONLY_VALIDATED"
    hardware_validation: str = "NOT_PERFORMED"

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


def benchmark_one(label: str, rel_path: str, input_name: str, shape: Tuple[int, ...],
                   description: str, n_runs: int = 30, seed: int = 0) -> ModelBenchResult:
    """
    Attempts to load `rel_path` with onnxruntime and, if it loads, runs
    real forward inference `n_runs` times on deterministic seeded synthetic
    input of the model's own documented input shape (this is a latency/
    load-success check, not a training-data or accuracy claim — no ground
    truth is available or asserted here). Returns a full result either way;
    a load failure is captured as data, never raised past this function.
    """
    result = ModelBenchResult(label=label, path=rel_path, description=description, loaded=False)
    full_path = REPO_ROOT / rel_path

    if not _ORT_AVAILABLE:
        result.load_error = "onnxruntime is not installed in this environment"
        return result
    if not full_path.exists():
        result.load_error = f"file does not exist: {rel_path}"
        return result

    try:
        session = ort.InferenceSession(str(full_path), providers=["CPUExecutionProvider"])
    except Exception as e:
        result.load_error = str(e)
        return result

    result.loaded = True
    result.input_name = input_name
    result.input_shape = list(shape)
    result.output_names = [o.name for o in session.get_outputs()]

    rng = np.random.RandomState(seed)
    x = rng.normal(0, 1, size=shape).astype(np.float32)

    latencies_ms = []
    out = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = session.run(None, {input_name: x})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    result.output_shapes = [list(o.shape) for o in out] if out is not None else None
    lat = np.array(latencies_ms)
    result.n_runs = n_runs
    result.latency_ms_mean = round(float(lat.mean()), 4)
    result.latency_ms_p50 = round(float(np.percentile(lat, 50)), 4)
    result.latency_ms_p95 = round(float(np.percentile(lat, 95)), 4)
    result.latency_ms_max = round(float(lat.max()), 4)
    return result


def benchmark_all(n_runs: int = 30) -> Dict:
    results = [benchmark_one(label, path, inname, shape, desc, n_runs=n_runs)
               for label, path, inname, shape, desc in MODEL_SPECS]
    n_loaded = sum(1 for r in results if r.loaded)
    return {
        "validation_type": "SOFTWARE_ONLY_VALIDATED",
        "hardware_validation": "NOT_PERFORMED",
        "note": "Latency measured on this machine's CPU via onnxruntime's Python "
                "CPUExecutionProvider. Not a mobile-device or embedded-hardware "
                "measurement. Input is deterministic seeded synthetic data matching each "
                "model's own documented input shape; this benchmarks load success and "
                "inference latency only, not model accuracy.",
        "n_models_attempted": len(results),
        "n_models_loaded": n_loaded,
        "n_models_failed_to_load": len(results) - n_loaded,
        "results": [r.as_dict() for r in results],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NAVDRIFT-0 offline ONNX inference benchmark (read-only)")
    parser.add_argument("--n_runs", type=int, default=30)
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    report = benchmark_all(n_runs=args.n_runs)
    print(f"\n{'='*70}\n  NAVDRIFT-0 Offline ONNX Inference Benchmark (read-only, software-only)\n{'='*70}")
    print(f"  models attempted: {report['n_models_attempted']}   "
          f"loaded: {report['n_models_loaded']}   failed: {report['n_models_failed_to_load']}\n")
    for r in report["results"]:
        if r["loaded"]:
            print(f"  [LOADED] {r['label']:24s} mean={r['latency_ms_mean']}ms  "
                  f"p95={r['latency_ms_p95']}ms  out={r['output_shapes']}")
        else:
            print(f"  [FAILED] {r['label']:24s} {r['load_error']}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWritten to {args.out_json}")
