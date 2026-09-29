# NAVDRIFT-0 Edge Engine (`edge/`)

ISRO SIH 2026 PS #26168. Offline, Python-only engineering additions built
alongside the live phone navigation core (`frontend/mobile.html`). Everything
in this directory is **isolated from the live mobile pipeline**: nothing here
is imported by, called from, or shares runtime state with
`frontend/mobile.html` or `frontend/desktop.html`. It exercises the existing,
unmodified `navdrift_engine.py` classes (`NavdriftRTEngine`, `ButterworthLP`)
through a formal external-IMU contract, and adds new offline capabilities:
vehicle-frame calibration, Viterbi map matching, and reproducible evidence
and capability reporting.

## What's here

| Module | Purpose | Status |
|---|---|---|
| `external_imu.py` | Formal `ExternalIMUFrame` contract with units, sample-rate and source metadata, strict per-frame and per-stream validation, CSV/streaming-record loaders, and an adapter into `NavdriftRTEngine` | Implemented, tested |
| `vehicle_calibration.py` | Static low-motion phone-to-vehicle frame calibration (roll/pitch from gravity, yaw from GNSS course when moving) | Implemented, tested |
| `replay_200hz_demo.py` | Real wall-clock latency/throughput measurement of the external-IMU to `NavdriftRTEngine` path, with an explicit software benchmark versus hardware validation distinction in its output | Implemented, tested |
| `imu_replay_example.py` | Minimal example command showing the external-IMU contract end to end (load, validate, replay, summarize) | Implemented, tested |
| `hmm_map_matcher.py` | Offline bounded-history Viterbi map matcher (candidate to emission to transition to accumulated path score to backpointer traceback) | Implemented, tested |
| `model_inference_bench.py` | Read-only offline ONNX inference benchmark against the repository's existing models, including the live-deployed weight files under `frontend/models/` | Implemented, tested |
| `calibration_evidence_demo.py` | Reproducible calibration evidence generator over deterministic synthetic fixtures | Implemented, tested |
| `hmm_viterbi_evidence_demo.py` | Reproducible offline Viterbi evidence generator with a full observations-to-final-path trace | Implemented, tested |
| `requirement_evidence.py` | Machine-readable mapping from SIH requirements to implementation, tests, and honest limitations, self-checked against the actual repository | Implemented, tested |
| `capability_report.py` | Reproducible script that generates a capability report from the requirement evidence and a live repository/test scan | Implemented, tested |

## Honesty notes (read before quoting any number from this directory)

- **No physical hardware validation has been performed.** `replay_200hz_demo.py`
  measures real wall-clock software processing latency; it has never been run
  against a physical 200Hz IMU/FOG board. Every stats dict it produces carries
  `"validation_type": "SOFTWARE_BENCHMARK"` and `"hardware_validation":
  "NOT_PERFORMED"`. Never strip those fields when reporting results.
- **The `_synthetic_timing_fixture()` benchmarks in this directory stay
  synthetic.** `_synthetic_timing_fixture()` in `replay_200hz_demo.py` and
  every fixture in the `tests/` files under this module are explicitly
  labeled synthetic and are used only to exercise pipeline mechanics or
  algorithmic correctness, never to claim navigation accuracy. If a real
  IMU/GNSS CSV is available (see `external_imu.frames_from_csv`'s column
  conventions, shared with `navdrift_engine.detect_columns()`), pass it via
  `--imu` to get a real measurement; nothing changes about how the script
  computes or reports its numbers. (Note: the real IO-VNBD dataset does now
  exist in this repository/workspace — see below — but it is not in the
  per-sample streaming-CSV format `external_imu.py` expects; `data/iovnbd.py`
  is the module that parses it, separately.)
- **`hmm_map_matcher.py` makes no map-matching accuracy claim.** Its unit
  tests use small, hand-constructed synthetic road/candidate fixtures built
  to have one unambiguous correct answer, verifying the Viterbi algorithm
  itself (does the accumulated-score traceback pick the right path), not
  real-world accuracy against real GNSS/road data. See the module's own
  docstring for its explicit relationship to the live `LiveHMM` matcher in
  `frontend/mobile.html`: they are different algorithms, and neither one is a
  more-accurate stand-in for the other in this repository's current state.
  It remains offline/edge-only and is not wired into `frontend/mobile.html`.
- **Requirement 2A (LSTM speed estimation) is no longer DATA_BLOCKED.** The
  real, official IO-VNBD dataset was acquired and a real speed LSTM was
  trained and validated on it (real held-out test MAE 3.213 m/s, RMSE 4.355
  m/s, R² 0.6965 — see `checkpoints/iovnbd_speed_lstm/training_report.json`).
  Its status is `SOFTWARE_ONLY_VALIDATED`: browser ONNX execution is
  real-validated (`results/iovnbd/browser_demo/`), but it is **not** wired
  into `frontend/mobile.html` — a real, documented sensor-feature-mapping gap
  (gravity-inclusive training data vs. gravity-compensated live IMU values,
  and no established phone-mount axis convention) blocks that, and nothing
  in this repository guesses around it. Requirement 8 (IO-VNBD benchmark) is
  similarly no longer `DATA_BLOCKED`: `eval/iovnbd_benchmark.py` reports a
  real classical dead-reckoning baseline over real held-out sequences (mean
  ATE RMSE 71.78 m — a different, separate number from the LSTM's own MAE,
  see `results/iovnbd/iovnbd_classical_dr_benchmark.json`). Requirements 2B
  (learned vibration denoiser) and 4 (NavIC VAE fusion) remain
  `DATA_BLOCKED`/untrained — no training run exists for either. See
  `requirement_evidence.py` and `results/EVIDENCE_INDEX.md` for the exact
  record per requirement.

## `external_imu.py`, the external IMU contract

Fields, with explicit units: `timestamp` (seconds), `ax/ay/az` (m/s^2, required),
`gx/gy/gz` (rad/s, required), `mx/my/mz` (uT, optional), `lat/lon` (degrees,
optional), `speed_mps` (m/s, optional), `course_rad` (radians, optional),
`baro_hpa` (hPa, optional), `sample_rate_hz` (nominal rate metadata, optional),
`source` (free-form sensor or file identifier, optional). No field is ever
fabricated; every optional field is either the real value the caller supplied
or `None`.

Validation is explicit and layered:
- `ExternalIMUFrame.validate()` checks that required fields are finite, that
  any present optional field is finite (not a stray NaN or inf), and that
  lat/lon and speed are in sane ranges. It never raises; `validate_or_raise()`
  does, with `MalformedIMURecordError`.
- `validate_stream()` checks an entire sequence: per-frame validity,
  timestamp ordering (a real sensor stream is never delivered out of order),
  dt outliers relative to the stream's own median sample period, and whether
  the estimated sample rate matches a rate this contract has been explicitly
  exercised at (`SUPPORTED_SAMPLE_RATES_HZ`). An unsupported rate is flagged,
  never rejected outright, since a real external IMU may legitimately run at
  a rate not in that list.

Three ways to build frames, all going through the same `ExternalIMUFrame`
validation: `frames_from_csv()` for a CSV file, `frame_from_record()` for one
record from a future streaming source (serial, socket, message queue), and
direct construction for a unit test or a custom loader.

`NavdriftRTEngine.step()` is untouched by any of this.

## `vehicle_calibration.py`, static calibration

Cleanly separated into the pieces the requirement specifies:

- A. Gravity-based roll and pitch estimation (`_rotation_from_gravity()`),
  deterministic and classical, no machine learning.
- B. GNSS-course-based yaw estimation, only when a course is supplied
  together with a speed above `GNSS_COURSE_MIN_SPEED_MPS`. Without a
  reliable course, yaw stays `None`, never guessed.
- C. Phone-to-vehicle rotation, an explicit 3x3 orthonormal matrix
  (`R_phone_to_vehicle`), not just a discrete axis index.
- D. Confidence, a deterministic function of how much of the offered window
  was usable and how consistent the accepted samples were with each other.
- E. Fallback and uncertain state: too few low-motion samples, or an
  accelerometer magnitude far from 1g, marks the result `valid=False` with a
  specific note, rather than returning a plausible-looking but wrong
  calibration.

This is static, low-motion calibration only. It does not claim, and does not
implement, dynamic in-motion vehicle-frame calibration.

## `hmm_map_matcher.py`, offline Viterbi map matcher

Genuinely different from the live `LiveHMM` module in `frontend/mobile.html`:

- **Live `LiveHMM`** (frontend/mobile.html, comment header above its
  definition): online, single persisted state, greedy per-tick argmax over a
  fresh emission and transition score, explicitly documented in its own
  source as not a full graph Viterbi (no accumulated path score, no
  backpointers, no offline reconstruction). Runs every navigation tick
  during a real GNSS blackout, on real fused EKF state.
- **This module**: a genuine bounded-history HMM/Viterbi decoder. It
  accumulates per-candidate path log-probabilities across a configurable
  history window (`history_window`, default 50 steps) and reconstructs the
  single maximum-likelihood path over that window via backpointer traceback,
  the textbook Viterbi algorithm. It is offline and batch: you hand it a
  sequence of observations and a candidate-generation function (for example a
  road-graph nearest-K lookup you supply; this module never generates
  candidates itself, so it is graph-agnostic and works with real OSM data, a
  custom road graph, or synthetic test fixtures).

It does not replace, wrap, or get called by the live matcher. Both remain
available; which one is appropriate depends on whether you need an online
per-tick decision (`LiveHMM`, live phone) or an offline best-path
reconstruction over a recorded or replayed sequence (`hmm_map_matcher`, this
module).

Test fixtures cover a straight road, an ambiguous intersection resolved by
accumulated continuity, parallel roads, deterministic noisy GNSS
observations, a temporary candidate gap with recovery, and the critical case
where an early, locally-best-looking wrong candidate is corrected once later
global evidence accumulates, something a purely greedy per-tick matcher
cannot do.

### Minimal usage example

```python
from edge.hmm_map_matcher import Observation, RoadCandidate, run_offline_match

def my_candidate_fn(obs: Observation) -> list[RoadCandidate]:
    # Real usage: query your road graph (OSM, a custom graph, etc.) for the
    # K nearest segments to (obs.lat, obs.lon) and return RoadCandidate(...)
    # for each, with the projected point, its bearing, and the perpendicular
    # distance. This function is entirely your responsibility; the matcher
    # is graph-agnostic.
    ...

observations = [
    Observation(lat=..., lon=..., heading_rad=..., speed_mps=..., dt_s=...),
    # one per recorded or replayed GNSS fix, oldest to newest
]

best_path = run_offline_match(observations, my_candidate_fn, history_window=50)
for state in best_path:
    if state.had_candidates:
        print(state.candidate.seg_id, state.log_prob)
    else:
        print("gap, no usable candidates at this step")
```

## Butterworth filter stability fix (`navdrift_engine.py`)

`navdrift_engine.py`'s `butter_coeffs()` had a sign-inverted feedback
coefficient pair (`a1`/`a2`) that made the implemented `ButterworthLP` filter
unconditionally unstable, with one pole outside the unit circle. This is
unrelated to the edge modules above but underlies the 200Hz software
benchmark and the external-IMU replay path, since both exercise
`NavdriftRTEngine`, which uses this filter. The fix corrects only the two
`a1`/`a2` lines to match `scipy.signal.butter`'s bilinear-transform
coefficients exactly; `ButterworthLP.step()` and the function's signature
and return shape are unchanged. Regression coverage lives in
`tests/test_butterworth_stability.py`, including a pole-stability proof, a
scipy cross-check, long deterministic runs in both float32 and float64, and
a re-run of the 200Hz replay with warnings treated as errors.

## `model_inference_bench.py`, read-only ONNX inference benchmark

A read-only benchmark harness discovered and built during a full repository
audit. It never writes to `frontend/` or `models/`; it only opens the
already-committed `.onnx` files with onnxruntime and runs real forward
inference on deterministic synthetic input matching each model's own
documented input shape.

Two audit findings motivated it. First, four of the five `.onnx` files in
`models/` (`driftformer_fp32.onnx`, `adaptive_ekf_fp32.onnx`,
`tunnel_det_fp32.onnx`, `navic_dop_fp32.onnx`) each reference an external
`<name>.onnx.data` weights file that is not present in this repository, so
onnxruntime cannot load them as shipped there; `models/imu_denoiser_int8.onnx`
is the one exception, fully self-contained. Second, the full-weight copies
actually served to the live app, `frontend/models/*.onnx` (all five files),
are self-contained and load and run successfully, the same weights the live
mobile and desktop pages use, inspected here read-only and never modified.

This is a software-only, CPU-side latency benchmark. It makes no claim about
mobile-device, embedded, or hardware inference latency.

```bash
python -m edge.model_inference_bench
```

## `requirement_evidence.py` and `capability_report.py`

`requirement_evidence.py` holds one record per SIH requirement: what was
implemented, which files and tests back it, an evidence artifact if one
exists, a status from a fixed set of categories, and a specific, honest
limitation. `validate_evidence_records()` checks every referenced file and
test path against the real repository, so this cannot silently drift out of
sync with the code.

`capability_report.py` builds on that data with direct repository checks
(which ONNX models exist, which new offline modules exist) and can actually
run the referenced test suite to report real pass and fail counts, not
assumed ones. Status categories used throughout, exactly:
`IMPLEMENTED`, `VALIDATED`, `SOFTWARE_ONLY_VALIDATED`, `OFFLINE_ONLY`,
`DATA_BLOCKED`, `EXPERIMENTAL`, `NOT_IMPLEMENTED`. No subjective language is
used anywhere in either module's output; a test asserts this directly.

```bash
python -m edge.requirement_evidence --out_json requirement_evidence.json
python -m edge.capability_report --out_json capability_report.json
```

## Running everything in this directory

```bash
pip install -r requirements.txt   # numpy, pandas, pytest, scipy (scipy used only by
                                    # tests/test_butterworth_stability.py's
                                    # cross-validation, auto-skipped if absent)

python -m pytest tests/test_external_imu.py tests/test_vehicle_calibration.py \
                  tests/test_hmm_map_matcher.py tests/test_requirement_evidence.py \
                  tests/test_capability_report.py -v

python -m edge.replay_200hz_demo --hz 200          # synthetic timing fixture demo
python -m edge.replay_200hz_demo --imu path/to/real_imu.csv --hz 200   # real data
python -m edge.imu_replay_example --hz 100         # simple external-IMU replay example
python edge/vehicle_calibration.py                 # worked example, synthetic tilt
python -m edge.requirement_evidence
python -m edge.capability_report
```

## Full evidence package

`results/EVIDENCE_INDEX.md` is the reviewer-facing index: for every
implemented or validated capability it names the exact command that
reproduces the artifact, and separates LIVE/DEPLOYED, OFFLINE,
SOFTWARE-ONLY, DATA-BLOCKED, and EXPERIMENTAL work. Start there for a full
reproduction of everything in this directory in one pass.

## What is explicitly NOT in scope for this directory

- Wiring any of these modules into `frontend/mobile.html` or
  `frontend/desktop.html`. New and experimental capabilities stay
  offline and Python-only until explicitly validated and explicitly approved
  for live integration.
- Any claim of real-world accuracy, latency-under-load, or hardware
  compatibility beyond what a specific, reproducible run in this directory
  actually measured and reported.
- Reconciling `api/app.py`'s and `inference/runtime.py`'s separate
  `NavDriftRuntime` classes. They were inspected and found to be genuinely
  independent implementations with non-overlapping interfaces; reconciling
  them carries real breakage risk to a deployed backend and was explicitly
  parked rather than attempted.
