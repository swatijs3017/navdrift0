"""
Export DRIFT-Former to ONNX and apply INT8 dynamic quantization.

Usage:
    python -m navdrift0.inference.export_onnx \
        --checkpoint ./checkpoints/drift_former/best_drift_former.pt \
        --output_dir ./checkpoints/onnx \
        --window 200

This script:
1. Loads the trained PyTorch checkpoint
2. Exports to ONNX (opset 17)
3. Applies dynamic INT8 quantization
4. Benchmarks CPU inference latency
5. Verifies numerical agreement between PyTorch and ONNX outputs
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from navdrift0.models.drift_former import DRIFTFormer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_to_onnx(
    model:       DRIFTFormer,
    output_path: Path,
    window:      int = 200,
    input_dim:   int = 8,
    opset:       int = 17,
) -> None:
    """
    Export the model to ONNX with dynamic batch and sequence axes.
    """
    model.eval()
    dummy = torch.zeros(1, window, input_dim)

    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        opset_version=opset,
        input_names=["imu_odom"],
        output_names=["pose_mean", "pose_cov"],
        dynamic_axes={
            "imu_odom": {0: "batch", 1: "seq"},
            "pose_mean": {0: "batch", 1: "seq"},
            "pose_cov": {0: "batch", 1: "seq"},
        },
        do_constant_folding=True,
    )
    logger.info("ONNX model saved: %s", output_path)


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_dynamic(model: DRIFTFormer) -> nn.Module:
    """
    Apply dynamic INT8 quantization to Linear layers.
    Works on CPU; gives ~2-4x speedup on ARM/x86 with INT8 BLAS.
    """
    quantized = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8,
    )
    logger.info("Dynamic INT8 quantization applied")
    return quantized


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def benchmark_latency(
    onnx_path: Path,
    window:    int = 200,
    input_dim: int = 8,
    n_runs:    int = 200,
) -> Tuple[float, float, float]:
    """
    Benchmark ONNX Runtime CPU inference latency.

    Returns:
        (mean_ms, std_ms, p95_ms)
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("onnxruntime not installed. Run: pip install onnxruntime")
        return float("nan"), float("nan"), float("nan")

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    dummy = np.zeros((1, window, input_dim), dtype=np.float32)

    # Warmup
    for _ in range(10):
        sess.run(None, {"imu_odom": dummy})

    # Measure
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {"imu_odom": dummy})
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times)
    return float(arr.mean()), float(arr.std()), float(np.percentile(arr, 95))


# ---------------------------------------------------------------------------
# Numerical verification
# ---------------------------------------------------------------------------

def verify_onnx(
    model:     DRIFTFormer,
    onnx_path: Path,
    window:    int = 200,
    input_dim: int = 8,
    atol:      float = 1e-4,
) -> bool:
    """
    Check that ONNX and PyTorch outputs agree within tolerance.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not available — skipping verification")
        return False

    model.eval()
    x = torch.randn(1, window, input_dim)

    with torch.no_grad():
        pt_mean, pt_cov, _ = model(x)

    sess = ort.InferenceSession(str(onnx_path),
                                 providers=["CPUExecutionProvider"])
    ort_outputs = sess.run(None, {"imu_odom": x.numpy()})
    ort_mean, ort_cov = ort_outputs[0], ort_outputs[1]

    mean_ok = np.allclose(pt_mean.numpy(), ort_mean, atol=atol)
    cov_ok  = np.allclose(pt_cov.numpy(), ort_cov,  atol=atol)

    if mean_ok and cov_ok:
        logger.info("ONNX verification PASSED (atol=%.1e)", atol)
    else:
        logger.warning("ONNX verification FAILED — mean_ok=%s, cov_ok=%s",
                       mean_ok, cov_ok)
    return mean_ok and cov_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = DRIFTFormer(
        input_dim=8, d_model=256, n_layers=4, n_heads=8,
        ffn_mult=4, dropout=0.0, max_seq_len=args.window + 32,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    logger.info("Loaded checkpoint from %s (epoch %d, val_ATE=%.3fm)",
                args.checkpoint,
                ckpt.get("epoch", -1),
                ckpt.get("val_ate_m", float("nan")))

    # Optional quantization before export
    if args.quantize:
        model = quantize_dynamic(model)

    # Export to ONNX
    onnx_path = output_dir / "drift_former.onnx"
    export_to_onnx(model, onnx_path, window=args.window)

    # Verify (skip if quantized — quantization changes numerics slightly)
    if not args.quantize:
        # reload the un-quantized model for verification
        model_orig = DRIFTFormer(
            input_dim=8, d_model=256, n_layers=4, n_heads=8,
            ffn_mult=4, dropout=0.0, max_seq_len=args.window + 32,
        )
        model_orig.load_state_dict(
            torch.load(args.checkpoint, map_location="cpu")["state_dict"])
        model_orig.eval()
        verify_onnx(model_orig, onnx_path, window=args.window)

    # Benchmark
    mean_ms, std_ms, p95_ms = benchmark_latency(onnx_path, window=args.window)
    logger.info("ONNX CPU latency — mean=%.2fms | std=%.2fms | p95=%.2fms",
                mean_ms, std_ms, p95_ms)

    if mean_ms != float("nan") and mean_ms < 10.0:
        logger.info("Target met: <10ms per step on CPU")
    elif mean_ms != float("nan"):
        logger.warning("Latency %.2fms exceeds 10ms target — "
                       "consider pruning or a smaller model", mean_ms)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export DRIFT-Former to ONNX")
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--output_dir",  default="./checkpoints/onnx")
    p.add_argument("--window",      type=int,  default=200)
    p.add_argument("--quantize",    action="store_true",
                   help="Apply INT8 dynamic quantization before export")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
