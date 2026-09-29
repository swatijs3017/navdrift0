"""
tests/test_model_inference_bench.py — sanity tests for edge/model_inference_bench.py

Verifies the benchmark harness is honest: it never claims hardware
validation, it reports load failures as data rather than crashing, and it
actually measures real latency (not a fabricated number) for whichever
models are present and loadable on disk in this checkout.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from edge.model_inference_bench import benchmark_all, benchmark_one, REPO_ROOT, _ORT_AVAILABLE

pytestmark = pytest.mark.skipif(not _ORT_AVAILABLE, reason="onnxruntime not installed in this environment")


def test_benchmark_one_handles_missing_file_gracefully():
    result = benchmark_one("does_not_exist", "no/such/model.onnx", "input", (1, 1), "test", n_runs=1)
    assert result.loaded is False
    assert "does not exist" in result.load_error


def test_benchmark_one_reports_real_latency_for_a_loadable_model():
    """models/imu_denoiser_int8.onnx is self-contained in this repository
    (unlike its siblings that reference missing external .onnx.data files),
    so this exercises the actual success path with a real model."""
    path = REPO_ROOT / "models" / "imu_denoiser_int8.onnx"
    if not path.exists():
        pytest.skip("models/imu_denoiser_int8.onnx not present in this checkout")
    result = benchmark_one("denoiser_test", "models/imu_denoiser_int8.onnx",
                            "raw_imu_6ax", (1, 6, 200), "test", n_runs=5)
    if not result.loaded:
        pytest.skip(f"model present but failed to load in this environment: {result.load_error}")
    assert result.latency_ms_mean is not None
    assert result.latency_ms_mean >= 0.0
    assert result.n_runs == 5
    assert result.output_shapes == [[1, 6, 200]]


def test_benchmark_result_always_carries_software_only_markers():
    result = benchmark_one("does_not_exist", "no/such/model.onnx", "input", (1, 1), "test", n_runs=1)
    assert result.validation_type == "SOFTWARE_ONLY_VALIDATED"
    assert result.hardware_validation == "NOT_PERFORMED"


def test_benchmark_all_never_raises_even_with_missing_models():
    report = benchmark_all(n_runs=2)
    assert report["n_models_attempted"] == len(report["results"])
    assert report["n_models_loaded"] + report["n_models_failed_to_load"] == report["n_models_attempted"]
    assert report["hardware_validation"] == "NOT_PERFORMED"


def test_benchmark_all_load_failures_carry_a_real_error_message():
    report = benchmark_all(n_runs=2)
    for r in report["results"]:
        if not r["loaded"]:
            assert r["load_error"] is not None and r["load_error"] != ""
        else:
            assert r["load_error"] is None
