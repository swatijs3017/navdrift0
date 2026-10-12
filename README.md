# NAVDRIFT-0

Dead reckoning navigation for ground vehicles and smartphones, built to keep a usable position estimate when GNSS (GPS / NavIC) is degraded or unavailable. Built for ISRO Smart India Hackathon 2026, Problem Statement 26168.

**Live production site:** https://navdrift0.pages.dev/ (device-routed: phones and tablets load the navigation app, desktops and laptops load the Mission Control replay dashboard)

`Python 3.11` `JavaScript (vanilla, no framework)` `ONNX Runtime / onnxruntime-web 1.18.0` `PyTorch 2.x` `Cloudflare Pages (frontend)` `Render (backend API, demo mode)` `IO-VNBD dataset (real)` `MIT License`

Status banners (each one is a fact checked against the repository during this audit, not a marketing claim):

| Banner | Status | Basis |
|---|---|---|
| Live smartphone navigation | Working, in production | `frontend/mobile.html`, real GNSS/IMU, unmodified this pass |
| Backend test suite | 96 / 97 passing | see section 12, full run recorded this pass |
| CI | Configured, non-blocking | `.github/workflows/ci.yml` runs on push/PR to `main`; test step uses `|| true`, so a failing test does not fail the workflow |
| Speed LSTM model | Trained and validated offline / in-browser | not wired into live phone navigation, see section 9 |
| IO-VNBD dataset | Real, parsed, split, trained on | 72 paired runs, seed 42, see section 10 |
| Hardware / embedded validation | Not performed | stated explicitly wherever it applies, never implied |

No badge in this file claims continuous integration gating, certification, field validation, or production scale beyond what is verified below.

---

## 1. What NAVDRIFT-0 Does

GNSS works well in open sky and degrades or disappears in tunnels, urban canyons (tall buildings causing multipath and signal blockage), and other temporary outages. When it drops out, a phone still has an accelerometer and gyroscope (and on some devices a barometer and magnetometer), but raw double integration of accelerometer data drifts quickly: small biases and noise accumulate error every tick, so a naive "just integrate the IMU" approach becomes unusable within tens of metres.

NAVDRIFT-0's control flow:

```
GNSS available        -> GNSS + IMU fusion (EKF measurement update on every fresh fix)
GNSS lost              -> dead reckoning: EKF predict-only, ZUPT, non-holonomic constraint,
                           road-graph matching, DriftFormer correction (all IMU-driven)
GNSS remains unavailable -> uncertainty grows with time, inertial estimate continues,
                           heading display switches to the gyro-integrated EKF heading
                           (GPS-derived heading would otherwise freeze)
GNSS returns            -> reacquisition: pre-correction drift is measured and logged,
                           the EKF is corrected via its normal measurement update
                           (no teleport, no discontinuous jump)
```

Commodity smartphone sensors (not a dedicated IMU/GNSS unit) are used deliberately: the target deployment is any Android or iOS phone a driver already carries, with no extra hardware required for the core navigation path. An external, higher-rate IMU is supported as an optional input (section 16) but is not required.

---

## 2. Current System Status

| Capability | Status | Where | Evidence |
|---|---|---|---|
| Live smartphone navigation | Live | `frontend/mobile.html` | production app |
| GNSS (`navigator.geolocation.watchPosition`) | Live | `frontend/mobile.html` | real device API |
| Accelerometer / gyroscope (`DeviceMotionEvent`) | Live | `frontend/mobile.html` | real device API |
| Orientation (`DeviceOrientationEvent`, `deviceorientationabsolute`) | Live | `frontend/mobile.html` | real device API |
| EKF fusion | Live | `frontend/mobile.html` | predict every tick, update on fresh GNSS |
| ZUPT | Live | `frontend/mobile.html` | forward-axis accel threshold + dwell time |
| Non-holonomic constraint (NHC) | Live | `frontend/mobile.html` | applied against matched road bearing when available |
| Road graph (Overpass + IndexedDB cache) | Live | `frontend/mobile.html` | real fetch, real cache fallback |
| Live road matching (`LiveHMM`) | Live, greedy | `frontend/mobile.html` | per-tick local argmax, not full Viterbi (see section 18) |
| DriftFormer (ONNX) | Live | `frontend/models/driftformer_fp32.onnx`, loaded and run in `frontend/mobile.html` | see section 6 |
| Adaptive EKF (ONNX) | Live | `frontend/models/adaptive_ekf_fp32.onnx`, loaded and run in `frontend/mobile.html` | see section 7 |
| Tunnel Detector (ONNX) | Live | `frontend/models/tunnel_det_fp32.onnx`, loaded and run in `frontend/mobile.html` | see section 8 |
| IMU denoiser (ONNX, int8) | Not live | `models/imu_denoiser_int8.onnx` | referenced by `frontend/desktop.html` only, and that page's own comment states it does not execute there (`ConvInteger` op unsupported in ort-web WASM); classical filtering is what actually runs in `mobile.html` |
| NavIC DOP model (ONNX) | Not live | `models/navic_dop_fp32.onnx` | not referenced by `mobile.html`; no live wiring found anywhere |
| Speed LSTM | Offline / browser-demo only | `checkpoints/iovnbd_speed_lstm/navdrift_lstm.onnx` | trained on real IO-VNBD data, not wired into `mobile.html`, see section 9 |
| Vibration filtering (classical) | Live | `frontend/mobile.html` (Butterworth / dead-band) | unmodified this pass |
| Full HMM/Viterbi map matcher | Offline only | `edge/hmm_map_matcher.py` | separate module, not wired into live matching |
| Vehicle calibration (gravity roll/pitch, GNSS-course yaw) | Offline module, software-validated | `edge/vehicle_calibration.py` | not the same code path as `mobile.html`'s own live auto-alignment |
| External IMU input contract | Software-validated, no physical board tested | `edge/external_imu.py` | |
| 200 Hz edge engine (`NavdriftRTEngine`) | Software throughput validated, hardware not validated | `navdrift_engine.py`, `edge/replay_200hz_demo.py` | |
| IO-VNBD training pipeline | Real, executed | `data/iovnbd.py`, `data/iovnbd_split.py`, `training/train_iovnbd_speed_lstm.py` | 72 runs, 52/10/10 split |
| Browser ONNX inference (Speed LSTM demo) | Validated | `results/iovnbd/browser_demo/index.html` | real headless-browser run |
| Desktop replay dashboard | Live (as a replay tool) | `frontend/desktop.html` | recorded/replay data, not live phone telemetry |
| Android | Working, standalone-PWA orientation bug fixed this pass | `frontend/manifest.json` | see section 19 |
| iPhone | Working | `frontend/mobile.html` | barometer unavailable via WebKit, documented |
| Edge/native deployment | Source exists, unbuilt on physical hardware | `native/android-plugin/`, `native/ios-plugin/` | Kotlin/Swift source present, not build-tested this pass |

---

## 3. System Architecture

```
Phone sensors (GNSS, accel, gyro, orientation, optional magnetometer/barometer)
        |
Calibration / auto forward-axis detection (mobile.html, per-session)
        |
Signal processing (classical Butterworth / dead-band filtering, ZUPT trigger)
        |
AI models (DriftFormer position correction, Adaptive EKF Q/R tuning, tunnel-blackout probability)
        |
EKF fusion (predict every tick; measurement update only on a fresh GNSS fix)
        |
Road constraints (NHC against matched road bearing; LiveHMM greedy road matching)
        |
Navigation state (fused position/heading/speed/uncertainty, displayed and logged)
        |
GNSS reacquisition (pre-correction error measured, then EKF measurement update)
        |
Correction + updated state (uncertainty shrinks, blackout counters/telemetry updated)
```

Each stage above is described in detail, with file and line-level pointers, in sections 4 through 8.

---

## 4. Smartphone Runtime (`frontend/mobile.html`)

- **GNSS**: `navigator.geolocation.watchPosition`. A fresh fix drives the EKF measurement update; a stale or missing fix leaves the EKF in predict-only (dead-reckoning) mode.
- **Accelerometer / gyroscope**: `DeviceMotionEvent`. Forward-axis acceleration is derived per-session (see calibration note below), not assumed from a fixed mounting convention.
- **Orientation**: `DeviceOrientationEvent` plus `deviceorientationabsolute` (the latter preferred on Android/Chrome, since plain `deviceorientation` can report a relative/unanchored `alpha` on some devices).
- **Magnetometer**: derived from `deviceorientationabsolute`'s compass heading where the browser provides it; not a separate raw magnetometer read.
- **Barometer**: read via the Generic Sensor API `Barometer` class where the browser supports it; on iOS Safari (WebKit) this class does not exist at all, which is a browser-level restriction the web app reports honestly (unavailable), not something it works around with a fabricated value.
- **Permissions**: standard `navigator.permissions` / iOS motion-permission prompt flow; denied or unavailable sensors report `UNAVAILABLE` / `PERMISSION_DENIED`, never a substituted value.
- **Navigation loop**: a single `step()` tick function drives EKF predict, GNSS-conditional EKF update, ZUPT, NHC, road matching, and rendering.
- **EKF**: `ekfStep(spd, dh, dt, gpsMeasured, gla, glo, gpsAccM)`. Predict always runs (position/heading integrated from IMU); the Kalman-gain measurement update (position only, never heading) runs only when `gpsMeasured` is true.
- **ZUPT**: fires when forward-axis acceleration magnitude stays under 0.35 m/s^2 for more than 800 ms and current integrated speed is under 2.5 m/s.
- **NHC**: applied using the matched road bearing when a road match is available.
- **Road matching**: `LiveHMM`, a per-tick greedy local best-candidate matcher against `RoadGraph` data (Overpass fetch, IndexedDB cache fallback).
- **Uncertainty**: `ekfUncertaintyM()`, derived from the real EKF covariance state, not a fixed or cosmetic number.
- **Blackout handling**: entering blackout seeds inertial speed from the last real GNSS speed (not zero), continues EKF predict-only integration, and freezes GPS-derived heading in favor of the gyro-integrated EKF heading for display.
- **Reacquisition**: `reacq()` measures the distance between the pre-correction EKF position and the new real GNSS fix (a pre-correction drift measurement, logged as `reacquisition_error_m`) before the EKF's own measurement update is applied.

---

## 5. AI Models

| Model | Purpose | Input | Output | Live / Offline | Training | Export |
|---|---|---|---|---|---|---|
| Adaptive EKF | Predict per-tick EKF process/measurement noise (Q/R) | `nav_state_features [1,6]` | `qr_noise_matrices [1,5]` | Live, every tick | Pre-existing (predates this audit pass) | ONNX, `frontend/models/adaptive_ekf_fp32.onnx` |
| Tunnel Detector | Estimate blackout probability from recent telemetry | `telemetry_window [1,20,5]` | `blackout_probability [1]` | Live, every tick | Pre-existing | ONNX, `frontend/models/tunnel_det_fp32.onnx` |
| DriftFormer | Position correction during real GNSS blackout | `imu_window_9ch [1,100,9]` | `position_correction_xy [1,2]` | Live, every 20th tick once the window is filled, gated to blackout only | Pre-existing (`checkpoints/driftformer_best.pt`, `training/train_drift_former.py`) | ONNX, `frontend/models/driftformer_fp32.onnx` |
| Speed LSTM | Speed estimation from a raw IMU window | `imu_window [1,50,6]` (accel_x/y/z, gyro_x/y/z) | `speed_mps [1,1]` (denormalization baked into the graph) | Offline / browser-validated only, not wired into `mobile.html` | Real, trained this project on IO-VNBD (section 9) | ONNX, `checkpoints/iovnbd_speed_lstm/navdrift_lstm.onnx`, 214,277 bytes |
| IMU denoiser | Learned vibration/noise denoising | `raw_imu_6ax [batch,6,200]` | `denoised_imu_6ax [batch,6,200]` | Not live: referenced by `frontend/desktop.html` only, and that page's own code comment states it cannot execute there (`ConvInteger` op unsupported in onnxruntime-web WASM) | No training script or training data exists in this repository | ONNX, int8 only (`models/imu_denoiser_int8.onnx`) |
| NavIC DOP predictor | Dilution-of-precision estimate from NavIC-style features | MLP, shape per `models/navic_dop_fp32.onnx` | n/a | Not live, no reference found in `mobile.html` or `desktop.html`'s live path | Unknown provenance; no training script for this exact checkpoint found in this repository | ONNX, `models/navic_dop_fp32.onnx` |
| NavIC motion-prior VAE | GNSS/IMU fusion prior (product-of-Gaussians posterior) | model code exists (`models/navic_vae.py`) | n/a | Not implemented as a trained artifact | `training/train_navic_vae.py` exists; no trained checkpoint, no ONNX export, no fusion-demo artifact exists anywhere in the repository | none |

A known repository inconsistency, documented and left unrepaired this pass (not a live-navigation risk, since the phone only ever loads from `frontend/models/`): the top-level `models/driftformer_fp32.onnx`, `models/adaptive_ekf_fp32.onnx`, `models/tunnel_det_fp32.onnx`, and `models/navic_dop_fp32.onnx` are much smaller than their `frontend/models/` counterparts (for example `models/driftformer_fp32.onnx` is 35,644 bytes versus `frontend/models/driftformer_fp32.onnx` at 13,085,053 bytes) and fail to load via onnxruntime, each because an external `.onnx.data` weights file it references is missing from the repository. This was confirmed by direct load attempts against both directories this pass (`edge/model_inference_bench.py`, section 6 of `results/EVIDENCE_INDEX.md`). Only `models/imu_denoiser_int8.onnx` is self-contained and loads correctly from that directory.

---

## 6. DriftFormer

Input: a 100-sample, 9-channel IMU window (`imu_window_9ch [1,100,9]`). Runs every 20th navigation tick, and only once the required window is fully populated. The correction it outputs (`position_correction_xy [1,2]`) is applied only during a real GNSS blackout, never while GNSS is locked. The correction is physically bounded: if applying it would imply a displacement greater than 60 m/s over the elapsed tick interval, it is rejected rather than applied, and a fail-open counter (`inferFailCount`) tracks inference failures so a broken model session degrades to unmodified dead reckoning rather than corrupting the position silently. DriftFormer cannot move the displayed position arbitrarily: it can only nudge the existing EKF dead-reckoning estimate within that bound, and only while blackout is genuine.

---

## 7. Adaptive EKF

Input: `nav_state_features [1,6]`, a small vector of current navigation-state features. Output: `qr_noise_matrices [1,5]`, interpreted as process/measurement noise terms for the EKF's own predict/update equations. Outputs are clamped to physically sane ranges and checked for non-finite values before use; a non-finite or out-of-range output is not applied. This model tunes how much the EKF trusts its process model versus incoming measurements at each tick; it does not itself compute position. Diagnostic output channels beyond the ones with a verified physical meaning are not assigned an interpretation in this document, since that was not independently confirmed this pass.

---

## 8. Tunnel Detector

Input: a 20-sample, 5-channel telemetry window (`telemetry_window [1,20,5]`). A small BiLSTM outputs a single blackout probability. This probability is used with hysteresis (separate enter/clear thresholds) rather than a single instantaneous threshold, to avoid rapid flapping between blackout and locked state near the decision boundary. It informs uncertainty growth and blackout-state transition logic; it does not itself perform position correction (that is DriftFormer's role) and does not gate GNSS acquisition, which is driven independently by the real `navigator.geolocation` fix stream.

---

## 9. Speed LSTM

A 2-layer LSTM (hidden size 64, 51,777 parameters) trained on real IO-VNBD data to estimate vehicle speed from a raw 50-sample, 6-channel IMU window (accel_x/y/z, gyro_x/y/z).

**Dataset**: real IO-VNBD paired runs, parsed by `data/iovnbd.py` (72 of 72 run pairs parsed successfully).

**Split**: sequence-level (not window-level, to avoid leakage), seed 42, `data/iovnbd_split.py`: 52 train / 10 validation / 10 test sequences (`results/iovnbd/split_manifest.json`).

**Windows**: 78,473 training windows, 24,258 validation windows, 4,015 test windows (stride 10, sequence length 50).

**Normalization**: input features are standardized using per-channel mean/scale computed from the training split (`checkpoints/iovnbd_speed_lstm/navdrift_lstm_meta.json`); the regression target (speed in m/s) is normalized during training and denormalized back to real m/s inside the exported ONNX graph itself (`SpeedLSTMWithDenorm`), so a caller reads real speed directly with no extra step.

**Training**: `training/train_iovnbd_speed_lstm.py`, device `cuda` on an NVIDIA A100-SXM4-40GB (Colab), seed 42, best epoch 5, early stopped at epoch 11 of 12, training time 21.73 s. The first training attempt collapsed to predicting the constant mean speed because the regression target was not normalized; this was diagnosed and corrected, and every number below is from the corrected run only.

**Test metrics** (4,015 held-out test windows): MAE 3.213 m/s, RMSE 4.355 m/s, R^2 0.6965. Best validation MAE during training: 3.539 m/s.

**ONNX export**: `training/export_iovnbd_lstm_onnx.py`, opset 17, 214,277 bytes. PyTorch-vs-ONNX max absolute output difference: 3.81e-06 over 50 checked samples. CPU latency on the export machine: 0.226 ms mean (that machine's CPU only, not a phone or browser measurement).

**Browser validation**: the exact exported file was run through `onnxruntime-web@1.18.0` (execution provider `wasm`, the same package version and EP `frontend/mobile.html` uses), input `imu_window [1,50,6]`, output `speed_mps [1,1]`. Real session creation and real inference succeeded, approximately 8.9 ms in that test environment (not a phone hardware measurement). This was repeated inside a real headless-browser run of the standalone demo page (section 15) against 40 real held-out test windows: MAE 3.395 m/s, RMSE 4.233 m/s on that subset, numerically consistent with the Python-side cross-check.

**Why it is not part of live phone speed estimation**: two specific, documented mismatches, not a vague "needs more work."

1. Gravity inclusion: the model was trained on IO-VNBD's raw smartphone accelerometer, which includes gravity (trained `accel_z` mean is approximately 9.85 m/s^2, essentially 1 g). `frontend/mobile.html`'s live `IMU.ax/ay/az` are deliberately gravity-*compensated* for its own EKF/dead-reckoning math; this is a different physical quantity, not simply a relabeling exercise.
2. Axis/frame correspondence: IO-VNBD does not establish a fixed phone-mounting axis convention (documented as an open item in `data/iovnbd.py`), while `frontend/mobile.html` treats phone orientation as arbitrary and auto-detects its own forward axis per session. There is no verified correspondence between IO-VNBD's recorded x/y/z axes and a live phone's orientation, so no mapping is assumed.

No live-phone mapping, calibration assumption, or navigation-accuracy improvement is claimed anywhere in this repository as a result of this model's existence.

---

## 10. IO-VNBD Dataset

Official source: the IO-VNBD (smartphone + vehicle) paired-run dataset, manually acquired and present in this workspace (`raw/`, real ZIP archives, excluded from git via `.gitignore` since it is large binary data, not source).

- 72 paired runs, `S-<run>.csv` (smartphone, 24 columns) and `V-<run>.csv` (vehicle, 29 columns), both at 10 Hz.
- Parser: `data/iovnbd.py`. Handles timestamp alignment, row-count mismatches between the S and V files for the same run, and the m/s speed conversion from the vehicle file's own units.
- Axis semantics: smartphone accelerometer channels include gravity (Android `TYPE_ACCELEROMETER` convention); a separate `gravity_x/y/z` channel is also present in the raw data. The dataset does not establish a fixed phone-mounting axis convention across runs, and this repository does not guess one; see section 9 for why that matters for live integration.
- Wheel radius: not guessed. No wheel-radius-dependent speed conversion is applied anywhere in this pipeline; the real vehicle-file reference speed is used directly instead of deriving speed from wheel geometry that was never measured or provided.
- Split: `data/iovnbd_split.py`, sequence-level (not window-level), seed 42: 52 train / 10 validation / 10 test sequences. Three small driver categories (Driver B, Driver E, Driver D) were kept in the training split rather than the validation/test split, since they had too few sequences to split further without leaving a category with zero held-out examples.
- Limitations: 72 runs from a specific data-collection campaign is not a large-scale, multi-city, multi-season dataset; results here should be read as validated on this dataset, not as a general real-world performance claim.

---

## 11. Training

Reproducibility for the Speed LSTM pipeline (the only model trained inside this project during this engineering phase; DriftFormer/Adaptive EKF/Tunnel Detector predate this pass and their own training scripts are `training/train_drift_former.py` and the Colab notebooks under `navdrift_colab/`):

```bash
pip install -r requirements.txt

# 1. Parse and verify the raw IO-VNBD archives
python -m data.iovnbd --verify

# 2. Produce the deterministic sequence-level split (seed 42)
python -m data.iovnbd_split --out_json results/iovnbd/split_manifest.json

# 3. Train the Speed LSTM (GPU strongly recommended; this project trained on an A100 via Colab)
python -m training.train_iovnbd_speed_lstm \
    --out_dir checkpoints/iovnbd_speed_lstm \
    --seed 42

# 4. Export to ONNX and cross-check against the PyTorch checkpoint
python -m training.export_iovnbd_lstm_onnx \
    --checkpoint checkpoints/iovnbd_speed_lstm/navdrift_lstm_best.pt \
    --out checkpoints/iovnbd_speed_lstm/navdrift_lstm.onnx

# 5. Classical dead-reckoning benchmark on the held-out test split
python -m eval.iovnbd_benchmark --out_json results/iovnbd/iovnbd_classical_dr_benchmark.json
```

The full Colab package (notebook build script, requirements, and instructions) is under `colab/`. `colab/README.md` documents the actual Colab workflow used to produce the real training run recorded in `checkpoints/iovnbd_speed_lstm/training_report.json`.

---

## 12. Validation and Benchmarks

### A. Live phone validation

No formal, recorded, analyzed live GNSS-blackout drive-test dataset exists in this repository as of this pass. Live behavior is currently verified by manual phone testing against the checklist in section 32, not by a stored benchmark file.

### B. IO-VNBD offline validation (Speed LSTM)

| Dataset | Split | Test size | Metric | Result | Environment |
|---|---|---|---|---|---|
| IO-VNBD | held-out test sequences (10) | 4,015 windows | MAE / RMSE / R^2 (speed, m/s) | 3.213 / 4.355 / 0.6965 | Colab A100, PyTorch |

This is a speed-prediction error in **metres per second**, not a navigation position error in metres. See section 13.

### C. Browser ONNX validation

| Subset | Test size | Metric | Result | Environment |
|---|---|---|---|---|
| Held-out browser subset | 40 windows | MAE / RMSE (m/s) | 3.395 / 4.233 | Real headless-browser run, onnxruntime-web 1.18.0, wasm EP |

CPU-only session-creation/inference latency observed in that environment: approximately 8.9 ms mean. This is not a phone hardware latency measurement.

### D. Classical dead-reckoning benchmark (IO-VNBD, not the Speed LSTM)

| Dataset | Sequences | Outage windows | Metric | Result |
|---|---|---|---|---|
| IO-VNBD held-out test split | 10 of 10 scored | seeded 10-60 s GNSS outages | mean ATE RMSE / mean max drift | 71.78 m / 240.72 m |

This is a classical gyro-yaw-integration + real reference-speed baseline (`eval/iovnbd_benchmark.py`), not a benchmark of the Speed LSTM, and not a measurement of `frontend/mobile.html`'s own live EKF/ZUPT/NHC/DriftFormer pipeline. Two of the ten sequences (`Vtb7`, `Vtb8`) had no seeded outage window long enough to score and report `null` drift values rather than a fabricated number.

### E. Edge timing benchmark (software only)

| Fixture | Samples | Mean latency | p95 latency | Max latency | Hardware validation |
|---|---|---|---|---|---|
| Synthetic 200 Hz timing fixture (`edge/replay_200hz_demo.py`) | 6,000 | 0.0165 ms | 0.0222 ms | 0.4654 ms | NOT PERFORMED |

Note: this repository's current `results/evidence/replay_200hz_demo.json` (re-checked during this pass) shows the numbers above. If a different set of numbers has been quoted elsewhere for this benchmark, that reflects a different run of the same synthetic fixture; this is a software timing measurement with expected run-to-run variance, not a fixed physical constant, and every run explicitly carries `hardware_validation: NOT_PERFORMED`.

### F. Earlier / legacy benchmark

| Dataset | Sequences | Metric | Result |
|---|---|---|---|
| Single earlier validation sequence (`results/validation_full.json`) | 1 | ATE RMSE / mean drift / max drift, over 36,819 steps | 0.2472 m / 0.023% / 1.742% |

This is a one-sequence result and is not presented as a general performance claim. It predates the IO-VNBD work in this repository and was not touched, altered, or re-derived by this pass. The same file's ONNX pipeline latency figures (FP32 4.853 ms, INT8 4.815 ms) are recorded in `results/isro_benchmark_table.csv` and are also from this single-sequence run, not a multi-sequence benchmark.

---

## 13. IO-VNBD Speed LSTM Results

MAE 3.213 m/s is the Speed LSTM's per-window **speed prediction error**, measured in metres per second. It is not a navigation position error, and it is not measured in metres. Do not compare it directly to the ATE RMSE figures in section 12D or 12F, which are position errors in metres from an entirely different benchmark on an entirely different pipeline (classical dead reckoning, not this LSTM).

---

## 14. Classical Dead-Reckoning Benchmark

10 held-out IO-VNBD test sequences, each with seeded synthetic GNSS-outage windows between 10 and 60 seconds. During each outage window, position is dead-reckoned using gyro-integrated heading and the real vehicle-file reference speed; ATE RMSE and maximum drift are measured against the real ground-truth position for that window. Mean ATE RMSE across the 10 sequences: 71.78 m. Mean maximum drift: 240.72 m. This is a classical baseline, included to give an honest sense of how much drift accumulates with no learned correction at all; it is not a performance claim for the Speed LSTM, and it does not exercise `frontend/mobile.html`'s live pipeline (which additionally has ZUPT, NHC, DriftFormer, and road matching, none of which this baseline uses).

---

## 15. Browser ONNX Validation

- Runtime: `onnxruntime-web@1.18.0`, execution provider `wasm` (the same package version and EP `frontend/mobile.html` uses for its own live models).
- Opset: `ai.onnx 17`.
- Supported operators: the Speed LSTM's operator set (LSTM, standard linear/elementwise ops) all ran without incident; this is distinct from the IMU denoiser (section 5), whose int8 `ConvInteger` op is documented as unsupported in this same runtime by `frontend/desktop.html`'s own code comment.
- Input/output: `imu_window [1,50,6]` float32 in, `speed_mps [1,1]` float32 out (denormalization baked into the graph).
- Numerical agreement: browser output matched the Python-side onnxruntime cross-check.
- 40 held-out windows: MAE 3.395 m/s, RMSE 4.233 m/s (a subset of the full 4,015-sample official test metric, reported separately and not presented as a replacement for it).
- Latency: approximately 8.9 ms mean in the test environment used for this check. This is CPU/WASM latency in that specific environment, explicitly not a phone hardware latency measurement.

---

## 16. Edge Engine

`navdrift_engine.py` implements `NavdriftRTEngine`, a 200 Hz-capable processing pipeline with an external-IMU input contract (`edge/external_imu.py`: validated frame/stream format, optional magnetometer/GNSS/barometer fields, CSV and streaming-record loaders). `edge/replay_200hz_demo.py` streams synthetic frames through the real, unmodified engine and measures real wall-clock per-sample latency and throughput at a 200 Hz target. Current measured result: 6,000 samples, mean 0.0165 ms, p95 0.0222 ms, max 0.4654 ms per sample, all under the 5 ms-per-sample budget for 200 Hz. This is software processing throughput on this machine only. No physical 200 Hz IMU or FOG board, and no embedded target, has been used to produce any number in this repository; `hardware_validation: NOT_PERFORMED` is recorded explicitly in the evidence file for this benchmark.

---

## 17. Vehicle Calibration

`edge/vehicle_calibration.py`: gravity-based roll/pitch from a stationary or low-motion accelerometer window, GNSS-course-gated yaw initialization (only trusted when the GNSS course itself is reliable, with an explicit unreliable-course fallback case), and an explicit phone-to-vehicle rotation matrix with a confidence/validity flag. Validated only against hand-constructed synthetic tilt/stationary fixtures with known analytical answers (`tests/test_vehicle_calibration.py`); no real vehicle mounting data was used. This is a separate module from, and has not been integrated with, `frontend/mobile.html`'s own live auto forward-axis detection, which continues to run unmodified; the two are not the same code path.

---

## 18. Map Matching

**LIVE MOBILE MATCHER**: `frontend/mobile.html`'s `LiveHMM` is a real, online, per-tick greedy matcher: at each tick it scores the current set of road candidates from `RoadGraph` against the live position and picks the locally best-looking one. It is called "HMM" in the code's own naming but does not maintain an accumulated path score or backpointer history across ticks, so it is not a full Hidden Markov Model / Viterbi decoder; it is a greedy local-argmax matcher, and this document does not describe it as more than that. Its output feeds the non-holonomic constraint (road-bearing) when a match is available. `road_candidates = 0` is a legitimate, honestly-reported empty-result state (no successful Overpass fetch and no cache hit, or genuinely no matching road nearby), not an error condition, and is not silently replaced with a fabricated candidate.

**OFFLINE VITERBI MATCHER**: `edge/hmm_map_matcher.py` is a separate, new, bounded-history Viterbi decoder: accumulated path score, explicit backpointers, and full path reconstruction. It is demonstrated (`tests/test_hmm_map_matcher.py`, `results/evidence/hmm_viterbi_evidence.json`) to correct an early, locally-best-looking wrong candidate using later global evidence, something the live greedy matcher structurally cannot do. It is validated only against small, hand-constructed synthetic road/candidate fixtures (straight road, intersection, parallel roads, gaps, wrong-early-candidate correction), not real-world GNSS/road data, and it is not wired into live navigation; `frontend/mobile.html`'s `LiveHMM` continues to serve live navigation unmodified.

---

## 19. Safety and Runtime Hardening

| Mechanism | Where | Behavior |
|---|---|---|
| NaN/Infinity protection | Adaptive EKF output path | non-finite model outputs are detected and not applied |
| GNSS stale protection | `frontend/mobile.html` | a fix must be recent (freshness check against `GPS.lastSpeedMpsTime`) to be trusted; a stale fix does not silently override valid inertial motion |
| IMU watchdog | `frontend/mobile.html` | sensor inactivity is detected and reported, not silently masked |
| DriftFormer correction bound | `frontend/mobile.html` | corrections implying more than 60 m/s of displacement are rejected |
| EKF predict/update structure | `frontend/mobile.html` | predict always runs; measurement update only on a fresh real GNSS fix, never fabricated |
| Road graph failure handling | `frontend/mobile.html` | Overpass failure falls back to the IndexedDB cache; both failing returns an honest empty candidate set, not a fabricated one |
| IndexedDB fallback | `frontend/mobile.html` | cached road data used when network is unavailable |
| Wake lock | `frontend/mobile.html` | requested during active navigation |
| Permission handling | `frontend/mobile.html` | denied/unavailable sensors report an honest unavailable state |
| iOS restrictions | `frontend/mobile.html` | barometer (`Barometer` Generic Sensor API) is not implemented by WebKit at all; reported as unavailable, not approximated |
| Android orientation | `frontend/manifest.json` | fixed this pass, see below |
| Map isolation | `frontend/mobile.html` | map resize/failure handling does not block the navigation tick loop |
| Service worker | `frontend/sw.js` | offline asset caching for the app shell |

**Android orientation fix, this pass**: `frontend/manifest.json` had `"orientation": "landscape"` combined with `"display": "standalone"`. When installed as a PWA (Add to Home Screen) on Android Chrome, this manifest field is enforced by the OS/browser regardless of physical device rotation, forcing landscape display; iOS Safari's PWA implementation does not honor this field the same way, which is why iPhone was unaffected. No CSS, JavaScript rotation hack, or user-agent-based layout branch was found anywhere in `frontend/mobile.html`; the lock was purely the manifest declaration. Fixed by changing the value to `"portrait"`. This is a one-line change confined to `frontend/manifest.json`; it does not touch navigation code, map rendering, or any other UI logic, and iPhone behavior is unaffected because iOS never enforced this field in the first place.

---

## 20. Mobile Platform Support

| Platform | Status | Details | Validation |
|---|---|---|---|
| Android (browser/PWA) | Working | Chrome/Chromium; standalone-PWA orientation bug fixed this pass | manual device testing, not automated |
| Android (native Capacitor build) | Source exists, not build-tested this pass | `native/android-plugin/NavdriftSensorsPlugin.kt`, `android/NavDriftService.kt` | not validated on physical hardware this pass |
| iOS/iPhone (browser/PWA) | Working | Safari; barometer unavailable (WebKit Generic Sensor API limitation, not a bug in this project) | manual device testing, not automated |
| iOS (native Capacitor build) | Source exists, not build-tested this pass | `native/ios-plugin/NavdriftSensorsPlugin.swift` | not validated on physical hardware this pass |
| Desktop/laptop (browser) | Working | Mission Control replay dashboard (`frontend/desktop.html`), device-routed automatically from `index.html` | manual testing |

The web app (browser/PWA path) is the primary, currently-deployed target. The native Capacitor plugin source (Kotlin/Swift) exists and is real, complete source code, but has not been built and run on physical hardware in this audit pass; no native-hardware validation is claimed.

---

## 21. Desktop Mission Control (`frontend/desktop.html`)

Replays recorded IO-VNBD sequences (or an uploaded GPX/CSV ground truth) and renders: Ground Truth (from the loaded replay file), NAVDRIFT fused trajectory (the replay-side EKF/fusion accumulator), Raw IMU trajectory (a separate raw-integration accumulator), and an EKF-baseline trajectory, each from a distinct, separately-tracked accumulator, not shared or aliased variables. GNSS blackout periods are visualized against the replay data. A dashed uncertainty circle is drawn from the real EKF covariance state (eigenvalues of the 2x2 covariance submatrix), scaled and clamped for on-screen legibility, not fabricated. A separate "Live Navigation (this device)" panel reads the browser's own live sensors independently of the replay pipeline (distinct DOM element IDs, explicitly commented as independent in the source); no shared variable was found feeding both, and no replay value can silently overwrite a live value or vice versa. This dashboard is a replay/benchmark tool: anything it displays from a loaded recording is recorded data, not live phone telemetry, and the interface does not claim otherwise.

---

## 22. Data Sources

| Source | Purpose | Used where | Status |
|---|---|---|---|
| IO-VNBD | Real paired smartphone/vehicle sensor dataset | Speed LSTM training, classical DR benchmark | Present in workspace, real, used |
| OpenStreetMap / Overpass API | Road network data for live road matching | `frontend/mobile.html` live `RoadGraph` | Live, used |
| OpenFreeMap Liberty (map tiles) | Base map rendering | `frontend/mobile.html`, `frontend/desktop.html` | Live, used, unmodified this pass |

No other external data source (ERA5, GFS, IMD, IGRA, SRTM, Himawari, IMERG, METAR, or similar) is present or wired into this codebase. Only sources actually found in the repository are listed here.

---

## 23. Repository Structure

```
frontend/
  mobile.html          Live phone navigation app (the production navigation path)
  desktop.html          Mission Control replay/benchmark dashboard
  manifest.json          PWA manifest (orientation fixed to portrait this pass)
  native-bridge.js       Capacitor native sensor bridge glue
  sw.js                   Service worker (offline app-shell caching)
  models/                 Live-deployed ONNX weights actually loaded by mobile.html/desktop.html

api/
  app.py                   FastAPI backend (demo-mode focused; not in the live phone's critical path)

edge/
  requirement_evidence.py  Source-of-truth requirement -> evidence mapping (this README's factual basis)
  capability_report.py     Regenerates a machine-readable status report from requirement_evidence.py
  vehicle_calibration.py   Offline calibration module
  hmm_map_matcher.py       Offline Viterbi map matcher
  external_imu.py           External IMU input contract and loaders
  model_inference_bench.py  Real ONNX load/latency benchmark against deployed weights
  replay_200hz_demo.py       200 Hz software timing benchmark
  README.md                  Detailed narrative walkthrough of this engineering phase's evidence

data/
  iovnbd.py                Real IO-VNBD parser
  iovnbd_split.py            Deterministic train/val/test split generator
  loader.py                  Older, separate dataset loader (pre-existing, not IO-VNBD-specific)

training/
  train_iovnbd_speed_lstm.py   Speed LSTM training script
  export_iovnbd_lstm_onnx.py    ONNX export + cross-check script
  train_drift_former.py          DriftFormer training script (pre-existing)
  train_navic_vae.py              NavIC VAE training script (no trained checkpoint exists)

eval/
  iovnbd_benchmark.py    Classical DR baseline over held-out IO-VNBD test sequences
  metrics.py               ATE/RTE/drift-rate metric implementations

colab/
  build_notebook.py, NAVDRIFT_IOVNBD_SETUP.ipynb, README.md   Real Colab training package

models/
  drift_former.py, navic_vae.py, snap_corrector.py    Python-side model source/architecture
  *.onnx                                                  Root-level model copies; four of five fail to load, see section 5

checkpoints/
  driftformer_best.pt
  iovnbd_speed_lstm/       Trained Speed LSTM checkpoint, ONNX export, and real training/export reports

results/
  EVIDENCE_INDEX.md          Index of every reproducible evidence artifact and the command that regenerates it
  evidence/                   Requirement evidence, capability report, calibration/HMM/model-bench/replay evidence (JSON)
  iovnbd/                       Split manifest, classical DR benchmark, browser demo, trajectory plots
  validation_full.json          Earlier, single-sequence (n=1) legacy benchmark
  isro_benchmark_table.csv       Per-model size/latency table from that legacy benchmark

tests/     9 test modules, pytest
native/      Real Kotlin/Swift native sensor plugin source (not build-tested this pass)
android/     NavDriftService.kt (not a full native Android project)
.github/workflows/ci.yml   GitHub Actions CI, push/PR to main
```

---

## 24. Evidence Package

- `results/EVIDENCE_INDEX.md`: an index of every reproducible artifact in `results/evidence/` and `results/iovnbd/`, with the exact command that regenerates each one from the current repository state. It proves reproducibility of the software-side evidence; it does not and does not claim to prove field/hardware validation, which is separately and explicitly marked `NOT_PERFORMED` throughout.
- `results/evidence/requirement_evidence.json` (generated from `edge/requirement_evidence.py`): the single source of truth for section 29's compliance matrix.
- `results/evidence/capability_report.json` (generated from `edge/capability_report.py`, regenerated this pass after it was found stale against `requirement_evidence.py`): status counts and per-requirement detail, plus a live internal test run at generation time.
- `results/evidence/model_inference_bench.json`: real load/latency results against the actual deployed `.onnx` weights, including the four root-`models/` load failures documented in section 5.
- `results/evidence/calibration_evidence.json`, `results/evidence/hmm_viterbi_evidence.json`: synthetic-fixture software correctness evidence for the two offline modules in sections 17 and 18.
- `results/iovnbd/`: the real IO-VNBD split, classical DR benchmark, and browser demo evidence underlying sections 9 through 15.

None of these prove real-world field performance; each explicitly states what it does and does not validate.

---

## 25. Reproducibility

```bash
# Environment
pip install -r requirements.txt

# Dataset parsing and split
python -m data.iovnbd --verify
python -m data.iovnbd_split --out_json results/iovnbd/split_manifest.json

# Training and export
python -m training.train_iovnbd_speed_lstm --out_dir checkpoints/iovnbd_speed_lstm --seed 42
python -m training.export_iovnbd_lstm_onnx \
    --checkpoint checkpoints/iovnbd_speed_lstm/navdrift_lstm_best.pt \
    --out checkpoints/iovnbd_speed_lstm/navdrift_lstm.onnx

# Benchmarks
python -m eval.iovnbd_benchmark --out_json results/iovnbd/iovnbd_classical_dr_benchmark.json
python -m edge.model_inference_bench --out_json results/evidence/model_inference_bench.json
python -m edge.replay_200hz_demo --hz 200 --n 6000 --out_json results/evidence/replay_200hz_demo.json
python -m edge.imu_replay_example --hz 100 --n 500 --out_json results/evidence/imu_replay_example.json
python -m edge.calibration_evidence_demo --out_json results/evidence/calibration_evidence.json
python -m edge.hmm_viterbi_evidence_demo --out_json results/evidence/hmm_viterbi_evidence.json

# Evidence regeneration
python -m edge.requirement_evidence --out_json results/evidence/requirement_evidence.json
python -m edge.capability_report --out_json results/evidence/capability_report.json

# Tests
pytest tests/ -v
```

The IO-VNBD/Speed-LSTM artifacts require the real IO-VNBD archives (not committed as binary data) and a GPU; they were produced via the `colab/` package documented in `colab/README.md`. Everything else above runs with only `requirements.txt` installed.

---

## 26. Deployment

- Frontend: static files under `frontend/` and the routing `index.html` at the repository root, served by Cloudflare Pages at `https://navdrift0.pages.dev/`.
- Routing: `index.html` is a thin device router only (no navigation logic of its own). It prefers `navigator.userAgentData.mobile` when the browser exposes it (authoritative, unaffected by display scaling or touch hardware), falling back to a touch-plus-viewport heuristic on browsers that do not expose that API (Safari/iPadOS, older Firefox). Phones/tablets are routed to `frontend/mobile.html`; desktops/laptops to `frontend/desktop.html`.
- Backend API: `api/app.py` (FastAPI), deployed separately via `render.yaml` to Render, running in `DEMO_MODE`. It is not in the live phone's navigation critical path (see section 27).
- Static assets: map tiles (OpenFreeMap Liberty) and ONNX model weights under `frontend/models/` are served as static files alongside the app.

---

## 27. Live Data Flow and Network Dependencies

- From sensors, with no network required: GNSS fix (still requires the phone's own GNSS radio, not this app's network), accelerometer, gyroscope, orientation, magnetometer (derived), barometer where available. EKF predict, ZUPT, NHC, and all three live ONNX models (DriftFormer, Adaptive EKF, Tunnel Detector) run entirely on-device once loaded; they consume real-time sensor data, not backend data.
- Requires network: initial page/asset load, map tile fetches, and the Overpass API fetch for new road-graph areas (falls back to the IndexedDB cache when offline or when Overpass fails).
- Does not call the backend: the live navigation pipeline in `frontend/mobile.html` does not send sensor data to, or read navigation state from, `api/app.py`. The phone's navigation core does not depend on the Python backend to function; the backend is a separate, demo-mode service not wired into the production mobile navigation path.

---

## 28. Known Limitations

- No formal, large-scale, multi-city field benchmark exists for the live navigation pipeline; live behavior is verified by manual phone testing, not a stored dataset.
- IO-VNBD's phone-mounting axis convention is not established in the dataset itself; this repository does not guess one (section 10).
- The Speed LSTM is trained and validated (offline and in-browser) but is not wired into live mobile speed estimation, blocked specifically by gravity-inclusion and axis-correspondence mismatches (section 9).
- The live road matcher (`LiveHMM`) is a greedy per-tick matcher, not a full Viterbi/HMM implementation; the full Viterbi decoder exists only offline (`edge/hmm_map_matcher.py`, section 18).
- The barometer is unavailable through iOS WebKit's Generic Sensor API; this is a platform limitation, reported honestly, not worked around with a fabricated value.
- The native Android/iOS Capacitor plugin source exists but has not been build-tested on physical hardware this pass.
- The 200 Hz edge engine's throughput number is a software timing measurement only; no physical 200 Hz IMU or FOG board has produced any number in this repository.
- Four of the five ONNX model files under the top-level `models/` directory fail to load (missing external `.onnx.data` weights) and are stale/undersized relative to their `frontend/models/` counterparts, which are the ones actually served live; this is documented, not repaired, this pass (section 5).
- The IMU denoiser has no training script or training data in this repository, so its denoising accuracy cannot be assessed; it also does not execute in the browser at all (`ConvInteger` op unsupported in onnxruntime-web WASM), per `frontend/desktop.html`'s own code comment.
- The NavIC motion-prior VAE has real fusion-math source code but no trained checkpoint, ONNX export, or fusion-demo artifact anywhere in this repository (`DATA_BLOCKED`).
- `tests/test_api.py::test_init` currently fails: it posts `latitude`/`longitude`/`speed_m_s` fields, while `api/app.py`'s `InitRequest` model requires `lat`/`lon`/`heading_deg`, so the request is rejected with HTTP 422 instead of the expected 200. This is a genuine contract mismatch between the test and the API, confirmed by direct code inspection this pass; it was not fixed, per the explicit instruction not to modify tests just to make them pass, and because the API backend is not in the live phone's navigation path (section 27), it carries no live-navigation risk. The repository's CI workflow (`.github/workflows/ci.yml`) runs `pytest ... || true`, so this failure does not currently fail the CI build.
- The single earlier `results/validation_full.json` benchmark (ATE RMSE 0.2472 m) is from one sequence only and should not be read as a general accuracy figure.

---

## 29. Requirement Compliance Matrix

Generated from `edge/requirement_evidence.py`, the repository's own machine-checked source of truth (`validate_evidence_records()` confirms every referenced file and test path actually exists; this passed with zero inconsistencies during this audit).

| Requirement | Implementation | Status | Evidence | Limitation |
|---|---|---|---|---|
| 1. In-vehicle phone alignment / calibration | `edge/vehicle_calibration.py`: gravity roll/pitch, GNSS-course yaw, rotation matrix, confidence/fallback | SOFTWARE_ONLY_VALIDATED | `tests/test_vehicle_calibration.py` | Synthetic fixtures only, no real vehicle mounting data |
| 2A. Learned (LSTM) speed estimation | `training/train_iovnbd_speed_lstm.py`, real IO-VNBD training | SOFTWARE_ONLY_VALIDATED | `checkpoints/iovnbd_speed_lstm/training_report.json` | Not wired into live phone; gravity/axis mismatch documented |
| 2B. Learned vibration/noise denoiser | `models/imu_denoiser_int8.onnx` inference-only artifact | SOFTWARE_ONLY_VALIDATED | `results/evidence/model_inference_bench.json` | No training script/data; does not execute in the browser at all |
| 3. Map matching against a road network | Live greedy `LiveHMM` + offline `edge/hmm_map_matcher.py` Viterbi | OFFLINE_ONLY (offline Viterbi module) | `tests/test_hmm_map_matcher.py` | Offline module uses synthetic fixtures; live matcher is greedy, not full Viterbi |
| 4. AI-based sensor fusion (NavIC motion-prior) | `models/navic_vae.py` math + `training/train_navic_vae.py` | DATA_BLOCKED | none | No trained checkpoint, no ONNX export, no fusion-demo artifact |
| 5. GNSS-denied dead reckoning (blackout handling) | Live ZUPT/NHC/DriftFormer dead reckoning | IMPLEMENTED | production app | Pre-existing; not re-validated in this pass |
| 6. GNSS reacquisition / trajectory snapping | Live reacquisition + `inference/runtime.py` snap correction | IMPLEMENTED | production app | Pre-existing; not re-validated in this pass |
| 7. External/plug-in IMU hardware support | `edge/external_imu.py`, formal frame/stream contract | SOFTWARE_ONLY_VALIDATED | `tests/test_external_imu.py` | No physical external IMU board tested |
| 8. Benchmark against IO-VNBD dataset | `eval/iovnbd_benchmark.py`, classical DR baseline | SOFTWARE_ONLY_VALIDATED | `results/iovnbd/iovnbd_classical_dr_benchmark.json` | Classical baseline only, not the live pipeline, not the LSTM |
| 9. Real-time mobile architecture (~10 Hz) | Live tick loop + offline model-latency confirmation | IMPLEMENTED | `results/evidence/model_inference_bench.json` | Live in-browser tick rate itself not re-measured this pass |
| 10. 200 Hz external-IMU-capable edge engine | `edge/replay_200hz_demo.py` against real `NavdriftRTEngine` | SOFTWARE_ONLY_VALIDATED | `results/evidence/replay_200hz_demo.json` | Software throughput only; no physical 200 Hz hardware used |

---

## 30. What Is Actually Live Right Now

**LIVE ON PHONE:**
GNSS acquisition and reacquisition, accelerometer/gyroscope/orientation sensing, EKF predict and measurement update, ZUPT, non-holonomic constraint, live greedy road matching (`LiveHMM`) against real Overpass/IndexedDB road data, DriftFormer position correction (blackout-gated), Adaptive EKF noise tuning, Tunnel Detector blackout probability, classical (non-learned) vibration filtering, wake lock, uncertainty display from real EKF covariance, raw-IMU diagnostic trail (segmented at GNSS transitions).

**OFFLINE / BENCHMARK:**
Speed LSTM (trained, browser-validated, not wired into live navigation), classical DR benchmark on IO-VNBD, IMU denoiser (inference-tested only, does not run in-browser), NavIC DOP predictor (no live reference found), NavIC motion-prior VAE (no trained artifact exists), offline Viterbi map matcher, offline vehicle calibration module, external IMU input contract, 200 Hz edge-engine software timing benchmark.

**DESKTOP REPLAY:**
Ground Truth / NAVDRIFT / Raw IMU / EKF-baseline trajectory replay, GNSS-blackout visualization over recorded data, uncertainty-circle rendering from replayed EKF covariance. All of this operates on recorded/uploaded data, never live phone telemetry, except the dashboard's separate, clearly-labeled "Live Navigation (this device)" panel, which reads the browser's own sensors independently.

**EDGE:**
`NavdriftRTEngine` (200 Hz-capable software pipeline), external IMU contract and loaders, offline Viterbi map matcher, offline vehicle calibration module. None of these run inside `frontend/mobile.html`.

---

## 31. Roadmap

Genuine future work only; nothing below is claimed as already existing.

- Live Speed LSTM integration, after the gravity-inclusion and axis-correspondence mismatch documented in section 9 is formally resolved (either by a verified live phone-orientation-to-IO-VNBD-axis mapping, or by retraining on gravity-compensated features matching the live pipeline).
- Learned vibration denoiser integration into the live pipeline, contingent on a real training script and IO-VNBD-based accuracy validation, which do not exist yet.
- Full live Viterbi map matching, replacing or supplementing the current greedy `LiveHMM`, using `edge/hmm_map_matcher.py` as the algorithmic starting point.
- Native sensor bridge build-and-hardware validation for the existing Kotlin/Swift plugin source.
- External IMU hardware validation against a physical board, using the existing `edge/external_imu.py` contract.
- A larger, multi-session, real-world GNSS-denied field dataset collected from the live phone app itself, to replace manual-testing-only validation of the live pipeline.
- Repair or removal of the four broken `models/*.onnx` files (missing external data), and either training or removal of the currently-dataless NavIC VAE and NavIC DOP modules.
- Resolving the `tests/test_api.py::test_init` field-name mismatch against `api/app.py`'s actual `InitRequest` contract.

---

## 32. Demo

- Production URL: https://navdrift0.pages.dev/
- Desktop route: opens automatically on a desktop/laptop browser (Mission Control replay dashboard).
- Mobile route: opens automatically on a phone/tablet browser (live navigation app).
- What judges can observe live: real GNSS lock/loss, real dead reckoning during an actual signal blackout (a tunnel or underground parking works well), real GNSS reacquisition, live DriftFormer/Adaptive EKF/Tunnel-Detector inference (visible in the diagnostics panel), live road matching against real OpenStreetMap data, and the segmented raw-IMU diagnostic trail.
- What is replay, not live: anything shown on the desktop dashboard from a loaded/recorded sequence, including the IO-VNBD trajectory plots and the classical DR benchmark numbers in section 12D. These are offline evaluation results, not a live field test, and are not presented as one.
- Benchmark numbers to cite correctly if asked: the Speed LSTM's 3.213 m/s MAE is a speed error, not a position error; the classical DR benchmark's 71.78 m mean ATE RMSE is a position error from a different, non-learned pipeline. Do not combine the two.

---

## 33. Transparency Note

This project separates live smartphone navigation, offline model validation, recorded-data replay, and edge-runtime experiments. Metrics are labeled by their source environment. Features that are not yet validated in the live mobile path are not presented as live capabilities.

---

## 34. Credits / License / References

License: MIT. ISRO Smart India Hackathon 2026, Problem Statement 26168.

Dataset: IO-VNBD (smartphone + vehicle paired-run navigation dataset).

Map data: OpenStreetMap contributors, via the Overpass API. Map tiles: OpenFreeMap Liberty.

Runtime: ONNX Runtime / onnxruntime-web 1.18.0, PyTorch.
