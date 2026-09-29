# NAVDRIFT-0 Evidence Index

ISRO SIH 2026 PS #26168. This is the answer to "what can I reproduce from
this repository right now." Every row below names the exact command that
regenerates the listed artifact from the current repository state, with
nothing fabricated and nothing assumed. Run `pip install -r requirements.txt`
first if a command reports a missing package.

Categories used, exactly:

- **LIVE / DEPLOYED**: part of the production mobile or desktop app, already
  running before this evidence package existed.
- **OFFLINE**: a new, separate Python module or script, not wired into the
  live app.
- **SOFTWARE-ONLY**: validated by running real code and measuring real
  numbers on this machine's CPU, with no physical or mobile-device hardware
  involved.
- **DATA-BLOCKED**: cannot be honestly completed without a dataset or
  checkpoint that does not exist in this repository or workspace.
- **EXPERIMENTAL**: exploratory code that exists but has not been reviewed
  or validated for any claim.

## 1. Requirement evidence

| Artifact | Category | Command | File |
|---|---|---|---|
| Requirement to implementation mapping, 11 records, self-checked against the real repository | OFFLINE | `python -m edge.requirement_evidence --out_json results/evidence/requirement_evidence.json` | `results/evidence/requirement_evidence.json` |
| Reproducible capability report (status counts, per-requirement detail, live test run) | OFFLINE | `python -m edge.capability_report --out_json results/evidence/capability_report.json` | `results/evidence/capability_report.json` |

Every entry in both files uses only `IMPLEMENTED`, `VALIDATED`,
`SOFTWARE_ONLY_VALIDATED`, `OFFLINE_ONLY`, `DATA_BLOCKED`, `EXPERIMENTAL`, or
`NOT_IMPLEMENTED`. No status was upgraded because code merely exists; see
each record's own `limitation` field.

## 2. Model evidence

| Artifact | Category | Command | File |
|---|---|---|---|
| Real inference against the live-deployed model weights (`frontend/models/*.onnx`) and the `models/` directory copies, load status, input/output shapes, measured latency | SOFTWARE-ONLY | `python -m edge.model_inference_bench --out_json results/evidence/model_inference_bench.json` | `results/evidence/model_inference_bench.json` |

This is **REAL DEPLOYED WEIGHT INFERENCE** (the exact `.onnx` files served to
the live app, loaded read-only and run with onnxruntime on this machine's
CPU). It is explicitly **not MODEL TRAINING VALIDATION** (no training or
accuracy claim is made for any model, including the IMU denoiser) and
explicitly **not HARDWARE VALIDATION** (`hardware_validation: NOT_PERFORMED`
on every result; no mobile device or embedded board was used). Four of the
ten model references fail to load (`models/driftformer_fp32.onnx`,
`models/adaptive_ekf_fp32.onnx`, `models/tunnel_det_fp32.onnx`,
`models/navic_dop_fp32.onnx`), each because its external `.onnx.data`
weights file is missing from the repository. This is reported as data in the
JSON, not hidden or retried.

The NavIC VAE fusion checkpoint still does not exist under any filename in
this repository; `model_inference_bench.py` makes no attempt to load it and
no claim about it. The LSTM speed model now does exist
(`checkpoints/iovnbd_speed_lstm/navdrift_lstm.onnx`, trained on real IO-VNBD
data — see section 6 below) but `model_inference_bench.py` was not modified
to load it in this pass; its own export validation is reported separately
in `checkpoints/iovnbd_speed_lstm/onnx_export_report.json` and in
Requirement 2A of `results/evidence/requirement_evidence.json`.

## 3. Edge engine evidence

| Artifact | Category | Command | File |
|---|---|---|---|
| 200 Hz software processing benchmark: samples, nominal frequency, mean/p95/max latency, effective throughput, non-finite output count, warnings, `hardware_validation: NOT_PERFORMED` | SOFTWARE-ONLY | `python -m edge.replay_200hz_demo --hz 200 --n 6000 --out_json results/evidence/replay_200hz_demo.json` | `results/evidence/replay_200hz_demo.json` |
| External-IMU replay demonstration using the formal contract (`edge/external_imu.py`), stream validation plus a replayed trajectory summary | SOFTWARE-ONLY | `python -m edge.imu_replay_example --hz 100 --n 500 --out_json results/evidence/imu_replay_example.json` | `results/evidence/imu_replay_example.json` |

Both are software processing validation only. Neither used, nor claims to
have used, a physical external IMU or FOG board.

## 4. Calibration evidence

| Artifact | Category | Command | File |
|---|---|---|---|
| Gravity-based roll/pitch, GNSS-course yaw (reliable and unreliable cases), phone-to-vehicle rotation sanity, two fallback/invalid cases | SOFTWARE-ONLY | `python -m edge.calibration_evidence_demo --out_json results/evidence/calibration_evidence.json` | `results/evidence/calibration_evidence.json` |

Every case is an explicitly labeled deterministic, seeded synthetic
accelerometer/gyroscope fixture with a known analytical answer. No physical
vehicle mounting or road test is implied by any result in this file.

## 5. HMM/Viterbi evidence

| Artifact | Category | Command | File |
|---|---|---|---|
| Full offline Viterbi trace: observations, candidates, emission scores, transition scores, accumulated path score, backtracking, final path, with an explicit tick-by-tick comparison against what a purely local/greedy decision would have picked | OFFLINE | `python -m edge.hmm_viterbi_evidence_demo --out_json results/evidence/hmm_viterbi_evidence.json` | `results/evidence/hmm_viterbi_evidence.json` |

The road/candidate data is a hand-constructed synthetic fixture, not real
GNSS or OSM data, and this file makes no real-world map-matching accuracy
claim. This is explicitly a separate module from, and does not call or
modify, `frontend/mobile.html`'s live greedy `LiveHMM` matcher.

## 6. IO-VNBD dataset, LSTM speed model, and classical DR benchmark

**The official IO-VNBD dataset was manually acquired by the user and now
exists in this repository/workspace** (`data/raw/`, not committed as
binary data — see `results/iovnbd_inspection_report.md`). As a direct
consequence, this repository now has, with real numbers:

- a real IO-VNBD parser (`data/iovnbd.py`, 72/72 S/V run pairs parsed
  successfully) and a deterministic sequence-level split
  (`data/iovnbd_split.py`, 52 train / 10 val / 10 test, seed 42;
  `results/iovnbd/split_manifest.json`)
- a trained-and-validated LSTM speed model (`training/train_iovnbd_speed_lstm.py`,
  trained on a Colab A100; `checkpoints/iovnbd_speed_lstm/navdrift_lstm.onnx`
  + `training_report.json` + `onnx_export_report.json`; see Requirement 2A
  in `results/evidence/requirement_evidence.json` for the full real MAE/RMSE/R^2
  numbers and limitations)
- a separate classical dead-reckoning baseline benchmark on 10 real
  held-out IO-VNBD test sequences (`eval/iovnbd_benchmark.py`,
  `results/iovnbd/iovnbd_classical_dr_benchmark.json`: mean ATE RMSE
  71.78 m, mean max drift 240.72 m). **These are classical-DR-baseline
  numbers, not LSTM speed-model numbers — the two are not the same
  benchmark and must not be conflated.**

This still does not and cannot honestly claim:

- NavIC VAE fusion training or ONNX export (Requirement 4 remains
  `DATA_BLOCKED`: no trained checkpoint exists for that model)
- a learned vibration/noise denoiser trained or validated on IO-VNBD
  (Requirement 2B is unrelated to this work and is unchanged)
- any mobile-device, browser, or hardware validation of the LSTM speed
  model (`hardware_validation: NOT_PERFORMED`)
- that the LSTM speed model is wired into `frontend/mobile.html` or the
  live navigation path (it is not) or that it improves real-world
  navigation accuracy (no such claim is made)

`results/validation_full.json` and `results/isro_benchmark_table.csv` are
still pre-existing artifacts from an earlier build (`n_test_sequences: 1`)
and were not touched, altered, or re-derived by this update; the new
IO-VNBD artifacts above live entirely under `results/iovnbd/` and
`checkpoints/iovnbd_speed_lstm/` so as not to overwrite them.

## 7. Model file integrity

- `frontend/models/*.onnx` (all five files): self-contained, load and run
  successfully, the same weights served to the live mobile and desktop app.
- `models/*.onnx` (four of five: `driftformer_fp32.onnx`,
  `adaptive_ekf_fp32.onnx`, `tunnel_det_fp32.onnx`, `navic_dop_fp32.onnx`):
  each references an external `.onnx.data` weights file that is not present
  in this repository, so they fail to load via onnxruntime as shipped there.
  `models/imu_denoiser_int8.onnx` is the one self-contained exception in
  that directory.

This distinction is documented, not repaired. No model file was modified,
regenerated, or replaced by this evidence package.

## Reproducing everything at once

```bash
pip install -r requirements.txt

mkdir -p results/evidence
python -m edge.requirement_evidence --out_json results/evidence/requirement_evidence.json
python -m edge.capability_report --out_json results/evidence/capability_report.json
python -m edge.model_inference_bench --out_json results/evidence/model_inference_bench.json
python -m edge.replay_200hz_demo --hz 200 --n 6000 --out_json results/evidence/replay_200hz_demo.json
python -m edge.imu_replay_example --hz 100 --n 500 --out_json results/evidence/imu_replay_example.json
python -m edge.calibration_evidence_demo --out_json results/evidence/calibration_evidence.json
python -m edge.hmm_viterbi_evidence_demo --out_json results/evidence/hmm_viterbi_evidence.json

python -m pytest tests/ --ignore=tests/test_api.py -v
```

The IO-VNBD/LSTM artifacts in section 6 are not reproduced by the block
above: they require the real IO-VNBD dataset and a GPU, and were produced
via the separate `colab/` package (see `colab/README.md`). Re-running the
classical DR benchmark alone (no GPU needed, once `data/raw/` has the real
ZIPs) is:

```bash
python -m eval.iovnbd_benchmark --out_json results/iovnbd/iovnbd_classical_dr_benchmark.json
```

## What this evidence package deliberately does not touch

`frontend/mobile.html`, `frontend/desktop.html`, `api/app.py`,
`inference/runtime.py`, the live EKF/ZUPT/NHC/DriftFormer/tunnel
detector/GNSS handling/road matcher/RoadGraph/map/UI, native Android/iOS
plugins, and every existing `.onnx` weight file. All of it was inspected
read-only where relevant and none of it was modified to produce any
artifact listed above.
