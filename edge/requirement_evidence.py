"""
edge/requirement_evidence.py — NAVDRIFT-0 SIH requirement -> evidence mapping
ISRO SIH 2026 PS #26168

A structured, machine-readable record of which SIH requirements this
repository actually implements, with what code, tested how, and with what
honest limitation. This is a DATA + VALIDATION module, not a claims
document: `validate_evidence_records()` checks that every referenced file
and test actually exists in the repository, so this can't silently drift
out of sync with the real codebase.

Status categories (used verbatim, no others):
    IMPLEMENTED              — code exists and runs, not independently validated
    VALIDATED                — validated against real-world/ground-truth data
    SOFTWARE_ONLY_VALIDATED  — validated, but only via software-level tests/benchmarks
                                (no physical hardware involved)
    OFFLINE_ONLY             — implemented and tested, but deliberately not wired
                                into the live mobile pipeline
    DATA_BLOCKED             — cannot be honestly completed without missing data
                                (e.g. no raw IO-VNBD dataset in this repository)
    EXPERIMENTAL             — exploratory/offline, not reviewed for production use
    NOT_IMPLEMENTED           — no working implementation exists yet

No entry here uses subjective language ("excellent", "production-ready",
"complete") — only the categories above and factual limitation text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

VALID_STATUSES = {
    "IMPLEMENTED", "VALIDATED", "SOFTWARE_ONLY_VALIDATED",
    "OFFLINE_ONLY", "DATA_BLOCKED", "EXPERIMENTAL", "NOT_IMPLEMENTED",
}


@dataclass
class RequirementEvidence:
    requirement: str            # short requirement id + one-line description
    implementation: str         # what was built
    files: List[str]            # repo-relative paths this evidence depends on
    tests: List[str]            # repo-relative test file paths
    evidence_artifact: Optional[str]   # a specific artifact (doc, JSON, plot) if one exists
    validation_type: str        # one of VALID_STATUSES
    limitation: str             # honest, specific limitation text

    def as_dict(self) -> dict:
        return asdict(self)


REQUIREMENT_EVIDENCE: List[RequirementEvidence] = [
    RequirementEvidence(
        requirement="Requirement 1: In-vehicle phone alignment / calibration",
        implementation="Static low-motion calibration module: gravity-based roll/pitch, "
                        "GNSS-course-gated yaw initialization, explicit phone->vehicle "
                        "rotation matrix with confidence/validity/fallback.",
        files=["edge/vehicle_calibration.py"],
        tests=["tests/test_vehicle_calibration.py"],
        evidence_artifact=None,
        validation_type="SOFTWARE_ONLY_VALIDATED",
        limitation="Validated only against hand-constructed synthetic tilt/stationary "
                   "fixtures with known analytical answers. No real vehicle mounting "
                   "data was used; this is a software-correctness validation, not a "
                   "real-world calibration accuracy benchmark.",
    ),
    RequirementEvidence(
        requirement="Requirement 2A: Learned (LSTM) speed estimation model",
        implementation="SpeedLSTM (training/train_iovnbd_speed_lstm.py) trained on the real, "
                        "official IO-VNBD dataset on a Google Colab A100 GPU, using a "
                        "deterministic sequence-level train/val/test split (52/10/10 "
                        "sequences, seed 42) produced by data/iovnbd_split.py from "
                        "data/iovnbd.py's real S-<run>/V-<run> parser (72/72 run pairs "
                        "parsed successfully). Exported to "
                        "checkpoints/iovnbd_speed_lstm/navdrift_lstm.onnx via "
                        "training/export_iovnbd_lstm_onnx.py, matching "
                        "navdrift_engine.py's existing LSTMSpeedEstimator ONNX "
                        "input/output contract exactly.",
        files=["navdrift_engine.py", "data/iovnbd.py", "data/iovnbd_split.py",
               "training/train_iovnbd_speed_lstm.py", "training/export_iovnbd_lstm_onnx.py",
               "checkpoints/iovnbd_speed_lstm/navdrift_lstm_best.pt",
               "checkpoints/iovnbd_speed_lstm/navdrift_lstm.onnx",
               "checkpoints/iovnbd_speed_lstm/navdrift_lstm_meta.json",
               "checkpoints/iovnbd_speed_lstm/training_report.json",
               "checkpoints/iovnbd_speed_lstm/onnx_export_report.json",
               "results/iovnbd/browser_demo/index.html",
               "results/iovnbd/browser_demo/browser_demo_samples.json"],
        tests=[],
        evidence_artifact="checkpoints/iovnbd_speed_lstm/training_report.json",
        validation_type="SOFTWARE_ONLY_VALIDATED",
        limitation="Trained and evaluated on a held-out test split of real IO-VNBD "
                   "sequences (10 sequences, 4015 windows): test MAE 3.213 m/s, test RMSE "
                   "4.355 m/s, test R^2 0.697 (best_epoch=5, early stopped at epoch 11, "
                   "training_time_s=21.73, device=cuda, GPU NVIDIA A100-SXM4-40GB; see "
                   "checkpoints/iovnbd_speed_lstm/training_report.json for the full "
                   "record). The first training attempt collapsed to predicting the "
                   "constant mean speed because the regression target was not "
                   "normalized; this was diagnosed and corrected, and the numbers above "
                   "are from the corrected run only, not the failed first attempt. ONNX "
                   "export (navdrift_lstm.onnx) was independently validated against the "
                   "PyTorch checkpoint (max abs diff 3.81e-06, see "
                   "onnx_export_report.json) and against a live onnxruntime sanity "
                   "inference on the actual exported file, confirming the "
                   "de-normalization is baked into the exported graph. ONNX CPU latency "
                   "(0.226 ms mean) was measured on the Colab session's own CPU only and "
                   "is not a mobile-device or browser-runtime measurement. Separately, "
                   "browser execution WAS validated: the exact navdrift_lstm.onnx file "
                   "was run through onnxruntime-web 1.18.0 (execution provider 'wasm', "
                   "the same package version and EP frontend/mobile.html uses), opset 17, "
                   "input imu_window [1,50,6], output speed_mps [1,1] — real session "
                   "creation and real inference both succeeded (~8.9 ms measured in that "
                   "test environment, not a phone performance claim), and again inside a "
                   "real headless-browser run of results/iovnbd/browser_demo/index.html "
                   "against 40 real held-out test windows, matching the Python-side "
                   "cross-check numerically. hardware_validation (a physical mobile "
                   "device) was NOT PERFORMED. This model is offline only: it is not "
                   "wired into frontend/mobile.html and is not part of the live "
                   "navigation path; no claim is made that it improves real-world "
                   "navigation accuracy. Live phone integration is separately blocked, "
                   "not just undone: the model was trained on IO-VNBD's gravity-inclusive "
                   "smartphone accelerometer (trained accel_z mean ~9.85 m/s^2), while "
                   "frontend/mobile.html's live IMU.ax/ay/az are deliberately "
                   "gravity-compensated for its own EKF/dead-reckoning math (a different "
                   "physical quantity, not a relabeling); and IO-VNBD's own phone-mounting "
                   "axis convention is not established in this dataset (see data/iovnbd.py's "
                   "own open-item note) while frontend/mobile.html treats phone orientation "
                   "as arbitrary and auto-detects its own forward axis per session. No "
                   "live-phone mapping is claimed or assumed anywhere in this repository. "
                   "This item covers speed estimation only — the separate learned "
                   "vibration/noise denoiser tracked under Requirement 2B was not "
                   "touched by this work, and the existing classical "
                   "Butterworth/dead-band filtering in the live pipeline is unchanged.",
    ),
    RequirementEvidence(
        requirement="Requirement 2B: Learned vibration/noise denoiser (TCN)",
        implementation="A real, self-contained ONNX inference artifact exists "
                        "(models/imu_denoiser_int8.onnx, same weights also present in "
                        "frontend/models/imu_denoiser_int8.onnx) and was confirmed, this "
                        "audit pass, to load and run correctly via onnxruntime, taking "
                        "input raw_imu_6ax [batch,6,200] and producing denoised_imu_6ax "
                        "[batch,6,200], with real measured inference latency. No training "
                        "script, training data, or accuracy evaluation for this model "
                        "exists in the repository.",
        files=["models/imu_denoiser_int8.onnx", "frontend/models/imu_denoiser_int8.onnx",
               "edge/model_inference_bench.py", "results/isro_benchmark_table.csv"],
        tests=["tests/test_model_inference_bench.py"],
        evidence_artifact="results/isro_benchmark_table.csv",
        validation_type="SOFTWARE_ONLY_VALIDATED",
        limitation="Only the inference path is validated (the model file loads and produces "
                   "correctly shaped output with real measured latency). Its training "
                   "provenance, and whether it actually denoises effectively, cannot be "
                   "assessed: no training script and no ground-truth IO-VNBD data exist in "
                   "this repository to evaluate denoising accuracy against. The pre-existing "
                   "results/isro_benchmark_table.csv row for this model is an artifact from an "
                   "earlier build, not evidence produced in this repository state. Also "
                   "note this model is not invoked by frontend/mobile.html's live "
                   "navigation pipeline (only DriftFormer, Adaptive EKF, and the tunnel "
                   "detector run there); frontend/desktop.html references it but its own "
                   "code comments state it does not actually execute in the browser due to "
                   "an unsupported ONNX op there.",
    ),
    RequirementEvidence(
        requirement="Requirement 3: Map matching against a road network",
        implementation="Two independent implementations: (a) frontend/mobile.html's "
                        "LiveHMM — live, online, greedy per-tick emission+transition "
                        "argmax against real RoadGraph data, unmodified by this build; "
                        "(b) edge/hmm_map_matcher.py — new, offline, bounded-history "
                        "Viterbi decoder with accumulated path score and backpointer "
                        "traceback, demonstrated to correct an early locally-best-looking "
                        "wrong candidate using later global evidence.",
        files=["edge/hmm_map_matcher.py", "frontend/mobile.html"],
        tests=["tests/test_hmm_map_matcher.py"],
        evidence_artifact=None,
        validation_type="OFFLINE_ONLY",
        limitation="edge/hmm_map_matcher.py is validated only against small, "
                   "hand-constructed synthetic road/candidate fixtures built to verify "
                   "algorithmic correctness (straight road, intersection, parallel "
                   "roads, gaps, wrong-early-candidate correction) — not real-world "
                   "map-matching accuracy against real GNSS/road data. It is not wired "
                   "into the live mobile pipeline; frontend/mobile.html's separate, "
                   "already-existing LiveHMM continues to serve live navigation "
                   "unmodified.",
    ),
    RequirementEvidence(
        requirement="Requirement 4: AI-based sensor fusion (NavIC motion-prior)",
        implementation="models/navic_vae.py implements real VAE fusion math "
                        "(NavICMotionPriorVAE, product-of-Gaussians posterior fusion) "
                        "and training/train_navic_vae.py exists, but no trained "
                        "checkpoint, ONNX export, or offline fusion demo artifact exists "
                        "in this repository.",
        files=["models/navic_vae.py", "training/train_navic_vae.py"],
        tests=[],
        evidence_artifact=None,
        validation_type="DATA_BLOCKED",
        limitation="Training requires real GNSS/IMU trajectory data (IO-VNBD or "
                   "equivalent), which is not present in this repository. The live "
                   "Adaptive-EKF fusion in frontend/mobile.html is unrelated and "
                   "unmodified by this item.",
    ),
    RequirementEvidence(
        requirement="Requirement 5: GNSS-denied dead reckoning (blackout handling)",
        implementation="Live ZUPT/NHC/DriftFormer-corrected dead reckoning in "
                        "frontend/mobile.html, including the previously-fixed stationary-"
                        "speed-recovery guard, already in production and unmodified by "
                        "this build.",
        files=["frontend/mobile.html"],
        tests=[],
        evidence_artifact=None,
        validation_type="IMPLEMENTED",
        limitation="Already satisfied before this build phase; documented here for "
                   "completeness only, not re-validated in this pass.",
    ),
    RequirementEvidence(
        requirement="Requirement 6: GNSS reacquisition / trajectory snapping",
        implementation="Live GNSS reacquisition and trajectory-snap logic in "
                        "frontend/mobile.html; also a separate SNAP correction class in "
                        "inference/runtime.py.",
        files=["frontend/mobile.html", "inference/runtime.py"],
        tests=[],
        evidence_artifact=None,
        validation_type="IMPLEMENTED",
        limitation="Already satisfied before this build phase; documented here for "
                   "completeness only, not re-validated in this pass.",
    ),
    RequirementEvidence(
        requirement="Requirement 7: External/plug-in IMU hardware support",
        implementation="Formal ExternalIMUFrame contract (units, optional magnetometer/"
                        "GNSS/barometer, sample-rate and source metadata, strict per-frame "
                        "and per-stream validation), CSV/streaming-record loaders, and an "
                        "adapter into the existing, unmodified NavdriftRTEngine.",
        files=["edge/external_imu.py"],
        tests=["tests/test_external_imu.py"],
        evidence_artifact=None,
        validation_type="SOFTWARE_ONLY_VALIDATED",
        limitation="This is a software input contract and loader, validated with unit "
                   "tests and synthetic/small real-format CSV fixtures. No physical "
                   "external IMU board has been connected or tested against it.",
    ),
    RequirementEvidence(
        requirement="Requirement 8: Benchmark against IO-VNBD dataset",
        implementation="eval/iovnbd_benchmark.py runs a classical dead-reckoning "
                        "baseline (real gyro-yaw integration + real V-file reference "
                        "speed, via data/iovnbd.py's real parser) over the real, "
                        "official IO-VNBD dataset's held-out TEST split (10 sequences, "
                        "results/iovnbd/split_manifest.json), scoring ATE only over "
                        "simulated GNSS-outage windows. This is a separate module from "
                        "the older data/loader.py pipeline and from the pre-existing "
                        "results/validation_full.json / results/isro_benchmark_table.csv "
                        "single-sequence (n=1) benchmark, which this item does not "
                        "modify or re-derive.",
        files=["eval/iovnbd_benchmark.py", "data/iovnbd.py",
               "results/iovnbd/split_manifest.json",
               "results/iovnbd/iovnbd_classical_dr_benchmark.json",
               "results/iovnbd/plots"],
        tests=[],
        evidence_artifact="results/iovnbd/iovnbd_classical_dr_benchmark.json",
        validation_type="SOFTWARE_ONLY_VALIDATED",
        limitation="Real result over 10/10 real held-out IO-VNBD test sequences: mean "
                   "ATE RMSE 71.78 m, mean max drift 240.72 m (see "
                   "results/iovnbd/iovnbd_classical_dr_benchmark.json for the full "
                   "per-sequence record and results/iovnbd/plots/ for trajectory "
                   "plots). This is a CLASSICAL dead-reckoning baseline (gyro-yaw "
                   "integration + real reference speed) — it does not use, and is not "
                   "a benchmark of, the Requirement 2A LSTM speed model; the two are "
                   "separate numbers and must not be conflated. It is an offline, "
                   "Python-side benchmark only: it does not exercise, call, or measure "
                   "frontend/mobile.html's live EKF/ZUPT/NHC/DriftFormer pipeline, and "
                   "makes no claim about live navigation accuracy. "
                   "results/validation_full.json and results/isro_benchmark_table.csv "
                   "(the older, single-sequence n=1 benchmark) were not touched, "
                   "altered, or re-derived by this work.",
    ),
    RequirementEvidence(
        requirement="Requirement 9: Real-time mobile architecture (~10 Hz)",
        implementation="frontend/mobile.html's live navigation tick loop runs the "
                        "EKF/ZUPT/NHC/DriftFormer/road-matching pipeline at its existing "
                        "measured tick rate. edge/model_inference_bench.py additionally "
                        "confirmed, read-only and offline, that the same DriftFormer, "
                        "Adaptive EKF, and tunnel-detector weight files actually deployed "
                        "to frontend/models/ load correctly and produce real measured "
                        "CPU inference latency in Python (DriftFormer ~7ms mean, Adaptive "
                        "EKF and tunnel detector well under 1ms mean, in this environment).",
        files=["frontend/mobile.html", "edge/model_inference_bench.py"],
        tests=["tests/test_model_inference_bench.py"],
        evidence_artifact=None,
        validation_type="IMPLEMENTED",
        limitation="The live in-browser tick rate itself was not re-measured in this pass; "
                   "only the deployed models' own inference latency was benchmarked, "
                   "offline and read-only, via Python onnxruntime on this machine's CPU, "
                   "not the phone's browser/WASM runtime. frontend/mobile.html was not "
                   "modified or instrumented to obtain this number.",
    ),
    RequirementEvidence(
        requirement="Requirement 10: 200 Hz external-IMU-capable edge engine",
        implementation="edge/replay_200hz_demo.py streams frames through the existing "
                        "NavdriftRTEngine and measures real wall-clock per-sample "
                        "processing latency/throughput at a 200 Hz target rate.",
        files=["edge/replay_200hz_demo.py", "edge/external_imu.py", "navdrift_engine.py"],
        tests=["tests/test_butterworth_stability.py"],
        evidence_artifact=None,
        validation_type="SOFTWARE_ONLY_VALIDATED",
        limitation="This measures software processing throughput only (every run's "
                   "stats dict explicitly carries hardware_validation=NOT_PERFORMED). "
                   "No physical 200 Hz IMU or FOG board has been used to produce any "
                   "number in this repository.",
    ),
]


def validate_evidence_records(records: Optional[List[RequirementEvidence]] = None) -> List[str]:
    """
    Checks that every evidence record: (a) uses a valid status category, and
    (b) every referenced file/test path actually exists in the repository.
    Returns a list of problem strings (empty = fully consistent with the
    actual repository state).
    """
    records = records if records is not None else REQUIREMENT_EVIDENCE
    problems: List[str] = []
    for rec in records:
        if rec.validation_type not in VALID_STATUSES:
            problems.append(f"{rec.requirement}: invalid validation_type {rec.validation_type!r}")
        for f in rec.files:
            if not (REPO_ROOT / f).exists():
                problems.append(f"{rec.requirement}: referenced file does not exist: {f}")
        for t in rec.tests:
            if not (REPO_ROOT / t).exists():
                problems.append(f"{rec.requirement}: referenced test does not exist: {t}")
        if rec.evidence_artifact and not (REPO_ROOT / rec.evidence_artifact).exists():
            problems.append(f"{rec.requirement}: referenced evidence artifact does not exist: "
                             f"{rec.evidence_artifact}")
    return problems


def to_json(records: Optional[List[RequirementEvidence]] = None) -> str:
    records = records if records is not None else REQUIREMENT_EVIDENCE
    return json.dumps([r.as_dict() for r in records], indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NAVDRIFT-0 SIH requirement evidence report")
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    problems = validate_evidence_records()
    if problems:
        print("INCONSISTENCIES FOUND (evidence references something that doesn't exist):")
        for p in problems:
            print(f"  - {p}")
    else:
        print("All evidence records reference files/tests that exist in the repository.")

    print(f"\n{len(REQUIREMENT_EVIDENCE)} requirement evidence records:")
    for rec in REQUIREMENT_EVIDENCE:
        print(f"  [{rec.validation_type:24s}] {rec.requirement}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            f.write(to_json())
        print(f"\nWritten to {args.out_json}")
