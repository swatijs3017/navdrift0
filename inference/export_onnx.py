"""
inference/export_onnx.py — NAVDRIFT-0 ONNX export + quantisation benchmark
===========================================================================

Exports DRIFTFormer to three ONNX variants and benchmarks each:
  1. FP32  — drift_former_fp32.onnx   (baseline export, opset 17)
  2. INT8  — drift_former_int8.onnx   (dynamic quantisation on Linear layers)
  3. INT4  — drift_former_int4.onnx   (weight-only MatMul4Bits quantisation)

Requirements
------------
  pip install torch onnx onnxruntime>=1.16.0

INT4 quantisation (MatMul4BitsQuantizer) was introduced in onnxruntime 1.16.
If you are on an earlier version the INT4 export step will raise ImportError
with a clear message; upgrade with:  pip install --upgrade onnxruntime
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnx  # type: ignore
import onnxruntime as ort  # type: ignore
import torch

from models.drift_former import DRIFTFormer  # type: ignore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)
logger = logging.getLogger("navdrift0.export")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPSET: int = 17
BATCH_SIZE: int = 1
SEQ_LEN: int = 50          # example sequence length for the dummy input
IMU_FEATURES: int = 10     # accel(3) + gyro(3) + mag(3) + baro(1)
BENCHMARK_RUNS: int = 200
WARMUP_RUNS: int = 20

OUTPUT_DIR = Path(os.environ.get("NAVDRIFT_OUTPUT_DIR", "."))
FP32_PATH = OUTPUT_DIR / "drift_former_fp32.onnx"
INT8_PATH = OUTPUT_DIR / "drift_former_int8.onnx"
INT4_PATH = OUTPUT_DIR / "drift_former_int4.onnx"
BENCHMARK_JSON = OUTPUT_DIR / "benchmark_int4_results.json"

# ---------------------------------------------------------------------------
# Helper: dummy IMU tensor
# ---------------------------------------------------------------------------

def _dummy_input() -> torch.Tensor:
    """Returns a (BATCH_SIZE, SEQ_LEN, IMU_FEATURES) float32 tensor."""
    return torch.randn(BATCH_SIZE, SEQ_LEN, IMU_FEATURES, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Step 1 — Export FP32 ONNX (opset 17, dynamic axes)
# ---------------------------------------------------------------------------

def export_fp32(model: torch.nn.Module) -> None:
    logger.info("Exporting FP32 ONNX (opset %d) → %s", OPSET, FP32_PATH)
    model.eval()
    dummy = _dummy_input()

    # Dynamic axes: batch dim=0, sequence dim=1
    dynamic_axes = {
        "imu_seq": {0: "batch_size", 1: "seq_len"},
        "delta_xy": {0: "batch_size"},
        "delta_heading": {0: "batch_size"},
        "uncertainty_major": {0: "batch_size"},
        "uncertainty_minor": {0: "batch_size"},
    }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(FP32_PATH),
            export_params=True,
            opset_version=OPSET,
            do_constant_folding=True,
            input_names=["imu_seq"],
            output_names=[
                "delta_xy",
                "delta_heading",
                "uncertainty_major",
                "uncertainty_minor",
            ],
            dynamic_axes=dynamic_axes,
        )

    # Verify the model is well-formed
    onnx_model = onnx.load(str(FP32_PATH))
    onnx.checker.check_model(onnx_model)
    logger.info("FP32 export verified OK.")


# ---------------------------------------------------------------------------
# Step 2 — INT8 dynamic quantisation on Linear layers
# ---------------------------------------------------------------------------

def quantise_int8() -> None:
    """
    Applies ONNX Runtime dynamic quantisation to all MatMul/Linear nodes.
    The result is saved to INT8_PATH.
    """
    logger.info("Applying INT8 dynamic quantisation → %s", INT8_PATH)
    from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore

    quantize_dynamic(
        model_input=str(FP32_PATH),
        model_output=str(INT8_PATH),
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QInt8,
        # Activations are quantised dynamically at runtime (no calibration set needed).
    )
    logger.info("INT8 quantisation complete.")


# ---------------------------------------------------------------------------
# Step 3 — INT4 weight-only quantisation (requires onnxruntime >= 1.16)
# ---------------------------------------------------------------------------

def quantise_int4() -> None:
    """
    Applies weight-only INT4 (4-bit) quantisation using MatMul4BitsQuantizer.

    Notes
    -----
    • MatMul4BitsQuantizer was introduced in onnxruntime 1.16.  Earlier
      versions will raise ImportError — upgrade with:
          pip install --upgrade onnxruntime
    • Only MatMul (Linear) weight tensors are quantised; activations remain
      in FP32, so no calibration dataset is required.
    • block_size controls the number of weights sharing one scale factor;
      32–128 is typical — smaller → better accuracy, larger → faster.
    • is_symmetric=True uses symmetric (signed) INT4 (range -8 … 7).
    """
    logger.info("Applying INT4 weight-only quantisation → %s", INT4_PATH)

    try:
        # Introduced in onnxruntime 1.16
        from onnxruntime.quantization.matmul_4bits_quantizer import (  # type: ignore
            MatMul4BitsQuantizer,
        )
    except ImportError as exc:
        raise ImportError(
            "INT4 quantisation requires onnxruntime >= 1.16.  "
            "Upgrade with:  pip install --upgrade onnxruntime"
        ) from exc

    onnx_model = onnx.load(str(FP32_PATH))

    quantizer = MatMul4BitsQuantizer(
        model=onnx_model,
        block_size=32,       # weights per quantisation block (power of 2, ≥16)
        is_symmetric=True,   # symmetric INT4 (-8 … 7)
        accuracy_level=0,    # 0 = default; 4 = highest accuracy (slower)
        nodes_to_exclude=[],
    )
    quantizer.process()

    onnx.save(quantizer.model.model, str(INT4_PATH))
    logger.info("INT4 quantisation complete.")


# ---------------------------------------------------------------------------
# Step 4 — Benchmarking
# ---------------------------------------------------------------------------

def _make_session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=opts, providers=["CPUExecutionProvider"]
    )


def _run_benchmark(session: ort.InferenceSession, label: str) -> Dict[str, float]:
    """
    Runs BENCHMARK_RUNS inferences and returns latency statistics (ms).
    """
    dummy_np = np.random.randn(BATCH_SIZE, SEQ_LEN, IMU_FEATURES).astype(np.float32)
    feed = {"imu_seq": dummy_np}

    # Warmup
    for _ in range(WARMUP_RUNS):
        session.run(None, feed)

    latencies: List[float] = []
    for _ in range(BENCHMARK_RUNS):
        t0 = time.perf_counter()
        session.run(None, feed)
        latencies.append((time.perf_counter() - t0) * 1_000)

    arr = np.array(latencies)
    stats = {
        "variant": label,
        "runs": BENCHMARK_RUNS,
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }
    return stats


def benchmark_all() -> List[Dict[str, float]]:
    results = []

    variants = [
        ("FP32", FP32_PATH),
        ("INT8", INT8_PATH),
        ("INT4", INT4_PATH),
    ]

    for label, path in variants:
        if not path.exists():
            logger.warning("Skipping %s benchmark — file not found: %s", label, path)
            continue
        logger.info("Benchmarking %s (%d runs + %d warmup) …", label, BENCHMARK_RUNS, WARMUP_RUNS)
        session = _make_session(path)
        stats = _run_benchmark(session, label)
        results.append(stats)
        logger.info(
            "%s  mean=%.2f ms  std=%.2f ms  p95=%.2f ms",
            label.ljust(5),
            stats["mean_ms"],
            stats["std_ms"],
            stats["p95_ms"],
        )

    return results


def _print_table(results: List[Dict[str, float]]) -> None:
    header = f"{'Variant':<8} {'Mean (ms)':>10} {'Std (ms)':>10} {'P95 (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}"
    sep = "-" * len(header)
    print("\n" + sep)
    print("  NAVDRIFT-0 ONNX Inference Benchmark")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['variant']:<8} "
            f"{r['mean_ms']:>10.3f} "
            f"{r['std_ms']:>10.3f} "
            f"{r['p95_ms']:>10.3f} "
            f"{r['min_ms']:>10.3f} "
            f"{r['max_ms']:>10.3f}"
        )
    print(sep + "\n")

    if len(results) >= 2:
        fp32 = next((r for r in results if r["variant"] == "FP32"), None)
        for r in results:
            if fp32 and r["variant"] != "FP32":
                speedup = fp32["mean_ms"] / r["mean_ms"] if r["mean_ms"] > 0 else float("inf")
                print(f"  {r['variant']} speedup vs FP32: {speedup:.2f}×")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build model — DRIFTFormer uses defaults; adjust kwargs to match your config.
    logger.info("Instantiating DRIFTFormer …")
    model = DRIFTFormer()
    model.eval()

    # -- Export ---------------------------------------------------------------
    export_fp32(model)
    quantise_int8()

    try:
        quantise_int4()
    except ImportError as exc:
        logger.error("%s", exc)
        logger.warning("Continuing benchmark without INT4 variant.")

    # -- Benchmark ------------------------------------------------------------
    results = benchmark_all()
    _print_table(results)

    # -- Save JSON ------------------------------------------------------------
    benchmark_data = {
        "onnxruntime_version": ort.__version__,
        "opset": OPSET,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "imu_features": IMU_FEATURES,
        "benchmark_runs": BENCHMARK_RUNS,
        "warmup_runs": WARMUP_RUNS,
        "results": results,
    }
    BENCHMARK_JSON.write_text(json.dumps(benchmark_data, indent=2))
    logger.info("Benchmark results saved to %s", BENCHMARK_JSON)


if __name__ == "__main__":
    main()
