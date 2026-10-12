# NAVDRIFT-0 Edge Engine: Input Contract, Adapters, Output Contract, Measured Throughput

ISRO SIH 2026 PS #26168. This document describes how `navdrift_engine.py`'s
`NavdriftRTEngine` is fed from two independent input sources through one
shared contract (`edge/external_imu.py`), what it hands back per frame, and
the real, measured processing-rate numbers for this machine — never
estimated. See `edge/README.md` for the full edge-module index; this file is
specifically the engine's input/output contract and throughput record.

## 1. Input schema (`edge/external_imu.py`, `ExternalIMUFrame`)

One dataclass, `ExternalIMUFrame`, with explicit units on every field:

| Field | Type | Unit | Required |
|---|---|---|---|
| `timestamp` | float | seconds (monotonic or wall-clock, caller's choice) | yes |
| `ax, ay, az` | float | m/s^2 (accelerometer) | yes |
| `gx, gy, gz` | float | rad/s (gyroscope) | yes |
| `mx, my, mz` | Optional[float] | uT (magnetometer) | no |
| `lat, lon` | Optional[float] | degrees, WGS84 (GNSS) | no |
| `speed_mps` | Optional[float] | m/s (GNSS ground speed) | no |
| `course_rad` | Optional[float] | rad, 0=north clockwise (GNSS course) | no |
| `baro_hpa` | Optional[float] | hPa (barometer) | no |
| `sample_rate_hz` | Optional[float] | nominal stream rate, metadata only | no |
| `source` | Optional[str] | free-form provenance string | no |

No field is ever fabricated: an absent optional field is `None`, never a
guessed value. Validation is layered — `ExternalIMUFrame.validate()` /
`validate_or_raise()` per frame, `validate_stream()` across a whole
sequence (ordering, dt outliers, rate-support check). See
`edge/external_imu.py` lines 67–147 and 246–297, and `edge/README.md`.

## 2. Two demonstrated adapters into the SAME, unmodified engine

Both adapters end at `NavdriftRTEngine.step()`/`.gps_update()`
(`navdrift_engine.py:453`, `:485`) completely unmodified. Neither adapter
changes the engine's method signature, internals, or return values.

### (a) External IMU adapter — `edge/external_imu.py` (pre-existing, tested)

- `frames_from_csv()` (line 180) — loads a CSV in `navdrift_engine.py`'s own
  flexible column convention.
- `frame_from_record()` (line 300) — builds one frame from a plain dict, the
  entry point a future serial/socket/queue bridge would call one record at a
  time.
- `engine_step_from_frame(engine, frame, dt)` (line 356) — calls
  `engine.step(...)`, and `engine.gps_update(...)` if the frame carries a
  GNSS fix.
- Tested by `tests/test_external_imu.py` (19 tests, all passing — see §6).

### (b) Simulated phone-sensor adapter — `edge/dual_adapter_demo.py` (new)

Phone IMU bridges are JavaScript (`frontend/mobile.html`'s DeviceMotion /
Geolocation handlers) with no existing Python-side equivalent or recorded
capture in this repository. `edge/dual_adapter_demo.py` adds a clearly
labeled **simulated, deterministic** phone-shaped record stream — field
names (`accel_incl_gravity`, `rotation_rate_rad_s`, `gnss_fix.latitude/
longitude`) chosen to look like a real phone bridge's raw payload, not like
`ExternalIMUFrame`'s own field names — and a translation function,
`_phone_record_to_engine_record()`, that maps it into the same
`frame_from_record()` input shape adapter (a) uses. It is explicitly **not**
a real phone capture and makes no navigation-accuracy claim; it demonstrates
the adapter boundary and "one engine, two input sources" wiring only. It
does not read, import, or modify `frontend/mobile.html`.

Run both adapters in one command:

```bash
python -m edge.dual_adapter_demo
```

Each adapter run constructs its own fresh `NavdriftRTEngine` instance (one
engine class, two independent runs — sharing one timeline across two
unrelated synthetic sensor sources would be physically meaningless), and
both runs route every frame through the identical
`engine_step_from_frame()` → `build_nav_state()` call pair.

## 3. Output contract (`edge/external_imu.py`, `NavState`)

`NavdriftRTEngine.step()` itself returns only a bare `(pred_lat, pred_lon)`
tuple (`navdrift_engine.py:453`) — no velocity, heading, uncertainty, mode,
or timestamp, and no documented field names. That return value is left
completely untouched. `NavState` (a frozen dataclass, `edge/external_imu.py`,
near `build_nav_state()`) is a read-only wrapper that reads the engine's own
already-public post-step attributes and formalizes one documented record per
ingested frame:

| Field | Type | Source |
|---|---|---|
| `timestamp` | float, s | copied from the input frame |
| `lat, lon` | float, degrees | `engine.ekf.x[0:2]` (same values `step()`/`gps_update()` already produced) |
| `heading_rad` | float, rad | `engine.heading` (integrated gyro-Z heading, already tracked internally) |
| `speed_mps` | float, m/s | `engine.int_spd` (engine's own forward-speed estimate) |
| `uncertainty_m` | float | derived from `engine.ekf.P[0,0]+P[1,1]` scaled by Earth radius — an **approximate**, relative confidence indicator (see caveat below), not a calibrated metres figure |
| `mode` | str | `"GNSS_FUSED"` if this frame carried a GNSS fix, else `"DEAD_RECKONING"` |
| `calibrated` | bool | `engine.calibrated` |
| `aligned` | bool | `engine.aligned` |
| `source` | Optional[str] | copied from the input frame |

Build one with:

```python
from edge.external_imu import build_nav_state
nav_state = build_nav_state(engine, frame, dt)
nav_state.as_dict()   # plain dict for logging/JSON
```

**Uncertainty caveat (stated once, here, not hidden):** the EKF's own state
vector mixes a latitude expressed in degrees with a covariance matrix
initialized and propagated in the engine's own ad hoc units
(`navdrift_engine.py`, class `EKF`); `uncertainty_m` applies a single Earth
-radius scale factor to approximate metres from that covariance trace. This
is a derived convenience number only — the EKF's actual covariance math is
untouched by this wrapper — and should be read as a relative/qualitative
confidence trend (did it go up or down after a GNSS fix?), not a calibrated
absolute-metres figure, unless that engine-internal unit question is
resolved separately.

Callers that only need the original tuple can still call
`engine_step_from_frame()` directly; both paths call the identical,
unmodified `NavdriftRTEngine.step()`.

## 4. Measured throughput (real numbers, this machine, software only)

**Hardware context:** Intel64 Family 6 Model 154 (GenuineIntel, laptop-class
x86-64), 16 logical CPUs reported by the OS, Windows 11. All benchmarks
below ran **single-threaded CPython, no GPU, no JIT/AOT compilation** — pure
interpreted Python + NumPy. These are **software processing-rate
measurements, not FOG-hardware validation**, and carry no navigation-accuracy
claim (synthetic fixtures have no ground-truth trajectory).

### 4.1 `edge/replay_200hz_demo.py` (pre-existing script, executed for this task)

```
python edge/replay_200hz_demo.py --hz 200 --n 10000
```

| Metric | Value |
|---|---|
| target_hz | 200.0 |
| n_samples | 10,000 |
| measured_elapsed_s | 0.2831 |
| **measured_effective_hz** | **35,324.01** |
| latency_ms mean / p50 / p95 / max | 0.0257 / 0.0235 / 0.0335 / 0.4582 |
| under_budget_fraction | 1.0 (every sample finished inside the 5 ms/200 Hz budget) |
| non_finite_output_count | 0 |
| warnings_raised | [] |

`validation_type: SOFTWARE_BENCHMARK`, `hardware_validation: NOT_PERFORMED`
(both emitted by the script itself).

### 4.2 Tight timing-loop benchmark of the engine call itself (new: `edge/engine_step_timing_bench.py`)

10,000 frames, deterministic synthetic fixture at nominal 200 Hz (5 ms)
spacing, fixed RNG seed 42 — same fixture shape as §4.1's synthetic fixture.
Two calls measured directly, back-to-back (no sleep/pacing, so this is
worst-case processing latency, not an artificially paced demo):

```
python edge/engine_step_timing_bench.py
```

| Call | effective_hz | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---|---|---|---|---|---|
| `engine_step_from_frame()` | 17,954.94 | 0.0440 | 0.1234 | 0.2161 | 0.9667 |
| `build_nav_state()` (adds the output-contract wrapper) | 18,547.76 | 0.0494 | 0.0675 | 0.1451 | 0.6729 |

(Run-to-run variance on this hardware is real — a second run of §4.1's
script measured ~32,000–35,000 effective Hz depending on OS scheduling noise;
every number here is from an actual executed run, not interpolated between
runs.) In all runs the effective rate was at least ~90x the 200 Hz target,
and p99 latency stayed under 0.22 ms, far inside the 5 ms/sample budget —
`build_nav_state()`'s extra formatting work did not measurably change the
achievable rate.

### 4.3 What this does NOT claim

No physical IMU, FOG, or other hardware board was used to produce any number
in §4.1–4.2. These numbers say the **Python software pipeline** can process
far more than 200 synthetic samples/second on this one development machine;
they say nothing about real sensor I/O latency, embedded/mobile-device CPU
performance, or multi-process/real OS scheduling contention in a deployed
setting.

## 5. Cross-check: the 10 Hz mobile live-rate trackers (`frontend/mobile.html`, read-only)

Per a prior audit, `frontend/mobile.html` claims a 10 Hz navigation-tick
target. Read-only verification of the actual tracking code (file untouched):

- `NAV_TICK_MS = 100` (100 ms = 10 Hz nominal tick period), and a real
  interval-based counter:
  ```js
  let lt=0, navTickCount=0, navHz=0, _navHzInterval=null;
  function startNavHzTracker(){
    if(_navHzInterval) return;
    _navHzInterval = setInterval(()=>{ navHz = navTickCount; navTickCount = 0; }, 1000);
  }
  ```
  `navTickCount` is incremented once per actual gated `step()` call inside
  `loop()` (the `requestAnimationFrame` scheduler only advances past the
  100 ms gate on a real elapsed-time check), and `navHz` is reassigned from
  that counter, then reset, once per real wall-clock second via
  `setInterval(...,1000)`. This **is** a real tick counter sampled against
  wall-clock time, not an assumed/hardcoded constant — the debug panel
  prints it as `` `${navHz}Hz (target 10Hz)` ``, i.e. the measured value
  next to the target, not instead of it.
- A second, separate counter, `window.onnxHz`, uses the identical
  measurement pattern (`_inferCount` incremented once per real ONNX
  inference call, reassigned/reset once per second) to report how often the
  ONNX models actually execute — explicitly documented in the file's own
  comment as "can legitimately be lower than navHz if a model call takes
  longer than one 100ms tick," i.e. it is not assumed to equal the tick
  rate.
- `IMU.hz` (referenced near `dbg-imuhz`) is a third, separate counter for
  raw IMU event rate, again distinct from both of the above.

**Conclusion of this read-only cross-check:** the methodology is sound — all
three trackers (`navHz`, `onnxHz`, `IMU.hz`) are real per-second tick/event
counters compared against wall-clock `setInterval` boundaries, not assumed
constants, and the code itself documents that they can legitimately diverge
from each other and from the 10 Hz target under load. This file was not
edited as part of this task.

## 6. Test results

```
python -m pytest tests/
```

**108 passed, 1 warning (pre-existing Pydantic deprecation warning in
`api/app.py`, unrelated to this work), 0 failed.** No regressions introduced
by `NavState`/`build_nav_state()` or `edge/dual_adapter_demo.py` /
`edge/engine_step_timing_bench.py` (both new files are demonstration/
benchmark scripts, not imported by the existing test suite, and the only
change to `edge/external_imu.py` was additive — no existing function's
signature or behavior changed).

## 7. File index (this task's changes)

| File | Status |
|---|---|
| `edge/external_imu.py` | Pre-existing; added `NavState` dataclass + `build_nav_state()` (additive only — no existing signature changed), updated module docstring |
| `edge/dual_adapter_demo.py` | New — "one engine, two input sources" demo (external-IMU CSV adapter + simulated phone-sensor adapter) |
| `edge/engine_step_timing_bench.py` | New — tight timing-loop benchmark of `engine_step_from_frame()`/`build_nav_state()` |
| `edge/replay_200hz_demo.py` | Pre-existing, unmodified; executed for this task (§4.1) |
| `docs/EDGE_ENGINE.md` | This file (new) |
