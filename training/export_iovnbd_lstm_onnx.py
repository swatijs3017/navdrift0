"""
training/export_iovnbd_lstm_onnx.py — exports a trained SpeedLSTM
checkpoint to navdrift_lstm.onnx and validates PyTorch vs ONNX outputs
numerically before reporting anything as usable.

Does NOT wire the exported model into frontend/mobile.html or any live
path. Output stays under the given --out_dir (e.g. checkpoints/ or
results/iovnbd/) until an explicit, separate integration decision.

Usage:
    python -m training.export_iovnbd_lstm_onnx \
        --checkpoint checkpoints/iovnbd_speed_lstm/navdrift_lstm_best.pt \
        --out_dir checkpoints/iovnbd_speed_lstm
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from training.train_iovnbd_speed_lstm import SpeedLSTM, SpeedLSTMWithDenorm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--onnx_name", default="navdrift_lstm.onnx")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--n_latency_runs", type=int, default=200)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base_model = SpeedLSTM(n_features=len(ckpt["feature_cols"]),
                            hidden_size=ckpt["hidden_size"], num_layers=ckpt["num_layers"])
    base_model.load_state_dict(ckpt["model_state"])
    base_model.eval()

    if "target_mean" not in ckpt or "target_std" not in ckpt:
        raise RuntimeError(
            "Checkpoint has no target_mean/target_std — it was trained with an older "
            "version of train_iovnbd_speed_lstm.py that regressed directly onto raw "
            "m/s targets (the version that collapsed to predicting the constant mean; "
            "see training_report.json's train_mse_loss history for that failure mode). "
            "Retrain with the current script before exporting.")

    # Wrap with the fixed affine de-normalization so the EXPORTED ONNX graph's
    # output is real m/s, matching navdrift_engine.py's LSTMSpeedEstimator
    # contract exactly — training itself happened in normalized target space.
    model = SpeedLSTMWithDenorm(base_model, ckpt["target_mean"], ckpt["target_std"])
    model.eval()

    seq_len = ckpt["seq_len"]
    n_features = len(ckpt["feature_cols"])
    onnx_path = out_dir / args.onnx_name

    dummy = torch.randn(1, seq_len, n_features, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["imu_window"], output_names=["speed_mps"],
        opset_version=args.opset,
        dynamic_axes=None,  # fixed seq_len, matches LSTMSpeedEstimator's fixed buffer contract
        dynamo=False,       # legacy TorchScript-based exporter; avoids an onnxscript
                             # dependency that recent torch versions otherwise pull in
                             # for the newer dynamo-based exporter path
    )

    # -- Validate PyTorch vs ONNX on real random-but-fixed-seed inputs --
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path))
    in_name = sess.get_inputs()[0].name

    rng = np.random.RandomState(0)
    n_check = 50
    max_abs_diff = 0.0
    for _ in range(n_check):
        x_np = rng.randn(1, seq_len, n_features).astype(np.float32)
        with torch.no_grad():
            torch_out = model(torch.from_numpy(x_np)).numpy()
        onnx_out = sess.run(None, {in_name: x_np})[0]
        diff = float(np.max(np.abs(torch_out - onnx_out)))
        max_abs_diff = max(max_abs_diff, diff)

    # -- Latency on this machine's CPU (NOT a mobile-device or hardware claim) --
    x_np = rng.randn(1, seq_len, n_features).astype(np.float32)
    # warmup
    for _ in range(10):
        sess.run(None, {in_name: x_np})
    t0 = time.perf_counter()
    for _ in range(args.n_latency_runs):
        sess.run(None, {in_name: x_np})
    t1 = time.perf_counter()
    mean_latency_ms = (t1 - t0) / args.n_latency_runs * 1000.0

    onnx_size_bytes = onnx_path.stat().st_size

    report = {
        "checkpoint": str(args.checkpoint),
        "onnx_path": str(onnx_path),
        "onnx_size_bytes": onnx_size_bytes,
        "seq_len": seq_len,
        "n_features": n_features,
        "opset": args.opset,
        "pytorch_vs_onnx_max_abs_diff": max_abs_diff,
        "pytorch_vs_onnx_n_checked": n_check,
        "onnx_cpu_latency_ms_mean": mean_latency_ms,
        "latency_hardware": "THIS MACHINE'S CPU ONLY — NOT a mobile-device or embedded-hardware measurement",
        "validation_type": "SOFTWARE_ONLY_VALIDATED",
        "hardware_validation": "NOT_PERFORMED",
    }
    with open(out_dir / "onnx_export_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    if max_abs_diff > 1e-3:
        print(f"\n[WARNING] PyTorch vs ONNX max abs diff {max_abs_diff:.6f} exceeds 1e-3 — "
              f"investigate before trusting this export.")
    else:
        print(f"\nPyTorch vs ONNX outputs match within {max_abs_diff:.6f} (max abs diff).")


if __name__ == "__main__":
    main()
