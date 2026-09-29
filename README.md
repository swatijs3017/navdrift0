# NAVDRIFT-0

**AI-assisted dead reckoning for ground vehicles and smartphones. Built for ISRO Smart India Hackathon 2026, Problem Statement 26168.**

NAVDRIFT-0 keeps a vehicle's position estimate usable when GNSS (GPS/NavIC) becomes unavailable or unreliable. It does this by fusing smartphone IMU data, a set of small trained neural network models, an Extended Kalman Filter, and OpenStreetMap road data, so the navigation state degrades gracefully instead of failing outright during a GNSS blackout.

There is no banner or logo asset in this repository, so none is included here.

Live production site: **https://navdrift0.pages.dev/**
The same URL serves a different experience depending on the device: a laptop or desktop opens the Mission Control dashboard, a phone or tablet opens the NAVDRIFT mobile navigation interface.

---

## 1. The Problem

GNSS (GPS, and for this project specifically NavIC, India's regional satellite navigation system) works well in open sky. It becomes unavailable or degraded in several common situations:

- **Tunnels and underground structures**, where the satellite signal is physically blocked.
- **Urban canyons**, where tall buildings reflect and block signals (multipath and signal blockage), producing either no fix or a badly degraded one.
- **Temporary outages** caused by terrain, weather, or receiver limitations, even in places that are not permanently GNSS-denied.

When GNSS drops out, the only information a smartphone has left is its inertial sensors: the accelerometer and gyroscope (and, on some devices, a barometer and magnetometer). Raw integration of accelerometer and gyroscope readings drifts quickly. Small sensor biases and noise accumulate error at every time step, so a naive "just integrate the IMU" approach becomes unusable within tens of metres.

A useful GNSS-outage system needs more than raw IMU integration. It needs:

- **Sensor fusion** (an Extended Kalman Filter or similar) to combine IMU data with whatever position information is available, and to track how uncertain the current estimate is.
- **Motion constraints**, because a wheeled vehicle cannot slide sideways the way raw double-integrated acceleration would suggest.
- **Road information**, because knowing the vehicle is on a road network can correct a drifting estimate back onto a plausible path.
- **A GNSS reacquisition strategy**, because when the signal comes back, the system needs to reconcile the drifted dead-reckoning estimate with the new fix smoothly rather than jumping.

NAVDRIFT-0 is built around this pipeline rather than around a single trick. It does not claim that a neural network "predicts GPS" on its own; the position estimate is maintained by sensor fusion and dead reckoning throughout, with trained models assisting specific parts of that pipeline (denoising, adaptive filter tuning, tunnel detection, and a learned drift correction).

---

## 2. What NAVDRIFT-0 Does

At a system level, the pipeline looks like this:

```
GNSS (when available)
      |
Smartphone IMU + orientation sensors (accelerometer, gyroscope, magnetometer, barometer)
      |
Preprocessing (filtering, calibration, axis alignment)
      |
AI-assisted motion estimation (DriftFormer correction, adaptive EKF noise tuning, tunnel detection)
      |
EKF sensor fusion (maintains position, heading, and uncertainty)
      |
Dead reckoning during GNSS loss (IMU-driven propagation, zero-velocity updates, non-holonomic constraint)
      |
Road graph / road matching (pulls the estimate toward the nearest plausible OpenStreetMap road segment)
      |
GNSS reacquisition (reconciles the drifted estimate with the new fix)
      |
State fusion and recovery
```

The core navigation state (position, heading, and uncertainty) is always maintained by the Extended Kalman Filter and dead-reckoning logic. The AI components (DriftFormer, the adaptive EKF noise predictor, and the tunnel detector) assist specific steps in that pipeline; they do not replace it, and none of them ever runs in place of a real GNSS fix while one is available.

---

## 3. System Architecture

There are three parts to the deployed system: the mobile client (the actual navigation engine, running in the phone's browser), the desktop Mission Control dashboard (a monitoring and demonstration interface, not the navigation engine), and a backend API (currently a demo/prototype service, explained honestly in section 8).

### Mobile client (`frontend/mobile.html`)

This is where the real navigation pipeline runs, entirely in the browser, on-device:

- **GNSS**: read via `navigator.geolocation.watchPosition`, a real browser API, not a scripted route.
- **Accelerometer / gyroscope / orientation**: read via `DeviceMotionEvent` / `DeviceOrientationEvent`, filtered and calibrated in-browser.
- **Barometer**: read via the Generic Sensor API's `Barometer` class where the browser supports it. The code explicitly detects and reports when this is not the case (see section 10).
- **Sensor status abstraction (`SensorManager`)**: every sensor (GNSS, accelerometer, gyroscope, orientation, magnetometer, barometer) is tracked through an explicit status vocabulary: `AVAILABLE`, `ACTIVE`, `NO_DATA`, `PERMISSION_DENIED`, `API_BLOCKED`, `UNAVAILABLE`. A 3-second watchdog degrades a sensor from `ACTIVE` to `NO_DATA` if readings stop arriving, so the UI never keeps showing a stale "active" state.
- **EKF**: maintains position, heading, and an uncertainty estimate, fused with GNSS when it is available.
- **DriftFormer**: a trained ONNX model that provides a learned drift correction, applied only during a genuine GNSS blackout, never while GNSS is locked.
- **GNSS blackout handling**: dead reckoning takes over, with zero-velocity updates (ZUPT, gated on measured acceleration magnitude and sustained stillness) and a non-holonomic constraint (a vehicle cannot slide sideways).
- **Road matching**: candidate road segments are scored against the fused position and heading (see section 9 for exactly how, including an honest note on what this is and is not).
- **Map visualization**: MapLibre GL JS, using the OpenFreeMap Liberty vector tile style (`tiles.openfreemap.org`), which is itself rendered from OpenStreetMap data.
- **Diagnostics**: a dedicated panel surfaces live GNSS state, fix age, IMU state and rate, navigation loop rate, position source, ONNX pipeline status, road graph status, AI Fusion counters, road-aware navigation state, and the most recent GNSS reacquisition, all reading real runtime values rather than being decorative.
- **GNSS reacquisition**: when a fix returns after a blackout, the drift error at that moment is measured and displayed, and the EKF reconciles the dead-reckoning estimate with the new fix.

A native app shell (Capacitor, wrapping this same HTML/CSS/JS) exists in source form for Android and iOS, with real native sensor plugins written in Kotlin and Swift. As of this writing it has not been built and run on physical native hardware; see section 16 for exactly what that means.

### Desktop Mission Control (`frontend/desktop.html`)

This is the monitoring and demonstration interface, not the phone's navigation engine. It is a separate HTML file with its own Leaflet-based map and its own tabbed panel layout (LIVE, MODULES, SENSORS, LOG on the left; METRICS, AI, COMPLY, BENCH, FLEET, EXPLAIN on the right). It exists so a person at a laptop can watch the system's state, review compliance metrics, and inspect the trained models' live runtime status, without needing a phone in hand. The AI and EXPLAIN tabs show the DriftFormer model's actual measured runtime telemetry (load status, input/output tensor shapes, inference count, latency, correction counts) rather than a fabricated visualization; earlier versions of this panel showed a randomized attention heatmap, which has been removed because the exported ONNX model does not expose attention weights (confirmed by inspecting the ONNX graph directly: it has exactly one input tensor and one output tensor, a 2D position correction, nothing else). A small "Live Navigation" section reads this laptop's own real GNSS/motion browser APIs when granted; since most laptops have no IMU, it is expected and correct for this to show "NOT AVAILABLE" rather than inventing a value.

### Single production URL and device routing

Both interfaces are served from the same URL, `https://navdrift0.pages.dev/`. The device-routing logic was specifically hardened this development cycle after a real bug: a touchscreen Windows laptop with OS display scaling could report a CSS viewport width under 900px even though its physical screen was a normal wide laptop panel, which caused it to be misclassified as a phone. The fix makes `navigator.userAgentData.mobile` (where the browser exposes it, which covers the large majority of Windows/Chromium laptops) the primary signal, since it reflects actual device identity rather than touch hardware or display scaling, and falls back to the previous touch-and-viewport heuristic only on browsers that don't expose that API (Safari/iPadOS, older Firefox). This is documentation of an existing routing fix, not a marketing feature.

### Backend (`api/app.py`)

A FastAPI service, deployed on Render, exists as a separate real-time streaming/session backend. It is not the same thing as the in-browser ONNX pipeline that actually drives the mobile and desktop UIs, and it is important not to conflate the two.

What it actually provides: HTTP endpoints `POST /init`, `POST /ingest`, `POST /gnss_lost`, `POST /reacquire`, `GET /trajectory`, `GET /status`, `POST /reset`, and a WebSocket `/ws/stream` that pushes pose updates at roughly 10 Hz. Authentication is a shared `X-API-Key` header (or `?api_key=` query parameter for the WebSocket, since browsers cannot set custom headers on a WS connection).

Whether this backend does real inference or demo inference depends entirely on the `DEMO_MODE` environment variable, and the deployed Render configuration (`render.yaml`) sets `DEMO_MODE: "true"` explicitly. In demo mode, every `/ingest` call and every WebSocket tick returns a deterministic simulated pose delta generated from a sine wave plus Gaussian noise (`_demo_infer` in `api/app.py`), and this is clearly demo output, not a model prediction. When `DEMO_MODE` is false, the code path (`_onnx_infer`) attempts to run ONNX inference on a model file expected at a path such as `drift_former_int8.onnx`, with a 10-value flattened input (`accel_x/y/z, gyro_x/y/z, mag_x/y/z, baro_hpa`) and outputs `dx, dy, dh` plus two uncertainty values. This input/output shape does not match the DriftFormer ONNX model actually used in the browser pipeline (`driftformer_fp32.onnx`, confirmed by direct ONNX graph inspection to take a `[batch, 100, 9]` IMU window and output a single `[1, 2]` position correction, with no uncertainty output). No model file matching the backend's expected shape and filename is present in this repository. In practice, this means the deployed backend currently only ever runs in demo mode; the "real inference" code path exists in source but is not connected to a matching trained model in this repository. This is stated plainly here so it is never mistaken for a live cloud inference service.

Also worth noting plainly: the deployed `render.yaml` sets an environment variable named `ALLOWED_ORIGINS`, but `api/app.py` reads `CORS_ORIGINS` (defaulting to `*` if unset). Those names do not match, so in the current deployment configuration the CORS origin restriction is not actually being applied as intended; the service is effectively running with `allow_origins=["*"]`. This is a real configuration issue, documented here rather than fixed, since this pass is documentation only.

`get_trajectory()` in the backend returns only the current single pose, not a stored history; its own source comment says "In production this would be stored in a ring buffer," so treat `/trajectory` as a stub rather than a trajectory log.

A second, more built-out runtime module exists at `inference/runtime.py` (also DEMO_MODE-aware, with its own ONNX loading logic), but `api/app.py` does not import or use it; it defines its own inline `NavDriftRuntime` class instead. `inference/runtime.py` is present in the repository but not currently wired into the deployed backend.

---

## 4. Mobile Website (Phone Application)

Opening `https://navdrift0.pages.dev/` on a phone or tablet routes to `frontend/mobile.html`, the actual navigation client. What it shows, backed by real runtime state rather than decoration:

- A full-screen map (MapLibre GL, OpenFreeMap Liberty style, light/dark aware) with the live estimated position.
- GNSS state (locked / blackout / reacquiring), fix age, and position source (GNSS-fused vs dead-reckoning).
- An **AI Fusion** panel: DriftFormer status (loaded, and whether it is currently idle because GNSS is locked or actively correcting during blackout), a count of corrections applied, a count of corrections rejected by the physical sanity check described in section 13, Adaptive EKF status, and Tunnel Detector status.
- A **Road-Aware Navigation** panel: match state (`LOCKED` or `SEARCHING`, from whether the road-matching logic currently has a selected segment), the number of nearby candidate road segments considered, the match distance, and the matched road's bearing. No numeric "confidence" percentage is shown here, because the underlying scoring value is an unbounded log-probability, not a 0-100 scale, and presenting it as a percentage would misrepresent it.
- GNSS reacquisition: the debug panel shows the drift error and blackout duration measured at the most recent reacquisition event, persisted beyond the on-screen banner that fades after a few seconds.
- A diagnostics panel covering EKF uncertainty, barometric altitude, GNSS outage count, non-holonomic correction count, tunnel state, GNSS state, fix age, IMU state and sample rate, navigation loop rate (target 10 Hz), position source, ONNX pipeline state, road graph state (segment count and whether it came from a live Overpass fetch or the offline IndexedDB cache), and magnetometer state.
- Light and dark map modes.

Only features that actually exist in the current code are listed above.

---

## 5. Desktop Website (Mission Control)

Opening the same URL on a laptop or desktop routes to `frontend/desktop.html`. It exists to give a person watching from a laptop (a judge, a teammate, a demo audience) visibility into the system's state without needing to look over someone's shoulder at a phone screen. It shows: live telemetry tiles, the same kind of navigation/system state information as the mobile diagnostics panel, a Leaflet-based map, per-model pipeline information (the AI tab), an ISRO PS 26168 compliance view (COMPLY tab), a benchmark view (BENCH tab), and an explanatory view of the DriftFormer model's input/output shapes (EXPLAIN tab). It is a monitoring and demonstration interface. It is not the phone's actual navigation engine; the phone runs its own independent copy of the pipeline in `mobile.html`.

The production routing was specifically designed so that a touchscreen laptop is not misclassified as a phone (see section 3's routing note); this matters because otherwise a demo laptop with a touchscreen could be shown the wrong interface.

---

## 6. Data Sources

| Source | What it is | Used for |
|---|---|---|
| IO-VNBD | Inertial and Odometry benchmark dataset for ground vehicle positioning (smartphone CSV files, RTK-GPS ground truth), the dataset specified by ISRO PS 26168 | Training data for the DriftFormer, IMU denoiser, adaptive EKF, and tunnel detector models |
| EuRoC MAV | A visual-inertial dataset with Vicon motion-capture ground truth | Referenced in prior validation work as cross-validation data |
| OpenStreetMap | Open, crowd-sourced road network data | Source of the road graph used for road-aware navigation (via Overpass and via the OpenFreeMap map tiles) |
| Overpass API | A public query service over OpenStreetMap data | Live fetch of nearby road segments around the phone's actual GPS fix, at runtime, in the mobile client |
| Smartphone GNSS | The phone's own GPS/NavIC receiver, via the browser's Geolocation API | Live position input when available |
| Smartphone IMU / orientation sensors | Accelerometer, gyroscope, magnetometer, barometer, via browser device sensor APIs (or native plugins in a native build) | Live dead-reckoning input |

No SRTM (terrain elevation) data source, no IMDAA, and no INSAT integration were found anywhere in the current codebase; they are not claimed as data sources here. NavIC-specific pseudorange/DOP handling in the trained models was trained on synthetic DOP data rather than a named external NavIC dataset; see section 8.

---

## 7. AI / ML Components

Five ONNX models are present in the repository and are what actually runs in the browser (`frontend/models/` and `models/`), verified by inspecting each ONNX graph directly rather than assuming:

| Model | File | Confirmed input | Confirmed output |
|---|---|---|---|
| DriftFormer | `driftformer_fp32.onnx` | `imu_window_9ch`, shape `[batch, 100, 9]` | `position_correction_xy`, shape `[1, 2]` |
| IMU Denoiser | `imu_denoiser_int8.onnx` / `imu_denoiser_fp32.onnx` | IMU window | Denoised IMU signal |
| Adaptive EKF noise predictor | `adaptive_ekf_fp32.onnx` | Runtime features (speed, heading variance, tunnel state) | Q/R scaling factors |
| Tunnel Detector | `tunnel_det_fp32.onnx` | Barometric/IMU sequence | Tunnel state flag |
| NavIC DOP predictor | `navic_dop_fp32.onnx` | Synthetic DOP-related features | DOP estimate |

All five run on-device in the browser through `onnxruntime-web` (WASM), in both `mobile.html` and `desktop.html`. This was verified by tracing the actual inference call sites and how their outputs are applied to the fused position, not just by checking that the model files load. DriftFormer's correction is applied only during a genuine GNSS blackout, never while GNSS is locked, and a physical sanity check rejects a proposed correction if the implied displacement exceeds a speed-based bound since the last correction (see section 13); the count of corrections applied and rejected is shown live in the AI Fusion panel described in section 4.

**DriftFormer does not expose attention weights.** The exported ONNX graph has exactly one input tensor and one output tensor (confirmed above). An earlier version of the desktop dashboard showed a fabricated, randomly generated attention heatmap in this space; it has been removed and replaced with the model's real measured runtime telemetry (load status, tensor shapes, inference count, latency, correction counts).

**A separate, more elaborate set of PyTorch model definitions exists under `models/` as source code** (`drift_former.py`, `navic_vae.py`, `snap_corrector.py`): a causal transformer with rotary position encoding and a heteroscedastic covariance output head, a GRU-based conditional VAE encoding 60 seconds of trajectory history, and a differentiable gradient-descent trajectory corrector run at GNSS reacquisition. These describe a more ambitious architecture than what is actually exported and deployed: the deployed `driftformer_fp32.onnx` has a single fixed-shape input and a plain 2-value output, with no covariance head and no VAE latent injection. This is a real gap between the training-time source code and what is actually running in production, and it is documented here rather than glossed over. Whichever architecture actually produced the currently deployed `.onnx` files, it is the simpler one described by their confirmed input/output shapes above, not the one described in `models/drift_former.py`'s docstring.

`inference/export_onnx.py`, which by its name should contain the ONNX export logic, currently contains something else entirely (its actual file content is JSON matching the PWA manifest, not Python). This is noted here as a repository inconsistency rather than described as working export code, since it plainly is not runnable as such.

Training-only components: the training scripts under `training/` (`train_drift_former.py`, `train_navic_vae.py`) and a single validation script under `navdrift_colab/` (`navdrift_08_validate.py`) exist as source, along with a Colab notebook under `notebooks/`. These are training/validation tooling, not something that runs live.

---

## 8. Navigation Engine

**GNSS.** When available, GNSS position (and speed/heading where the fix includes them) is read via the browser's Geolocation API and fused into the EKF.

**GNSS blackout.** When GNSS is lost, the system switches to dead reckoning: IMU-driven propagation of position and heading, constrained by ZUPT and the non-holonomic constraint below, optionally corrected by DriftFormer and pulled toward the road graph.

**IMU.** Accelerometer and gyroscope readings, filtered, calibrated, and axis-aligned, drive the dead-reckoning integration. A watchdog tracks whether IMU data is genuinely still arriving (`ACTIVE` vs `NO_DATA`).

**ZUPT (zero-velocity update).** When the measured forward-axis acceleration magnitude stays below a threshold (0.35 in the current implementation's units) for a sustained period (800 ms) while the integrated speed is already near rest, the system treats the vehicle as stationary and zeroes the drifting integrated speed. This prevents small sensor noise from accumulating into a phantom velocity while the vehicle is actually stopped.

**EKF.** The Extended Kalman Filter maintains the core navigation state: position, heading, and an uncertainty covariance. It is what the rest of the pipeline (DriftFormer corrections, road matching, GNSS fixes) feeds into and reads from; it is the thing actually being estimated, not a side effect of the AI components.

**DriftFormer.** Provides a learned position correction during blackout, described above; it assists the dead-reckoning estimate rather than replacing the EKF.

**Motion constraints.** A non-holonomic constraint (NHC) is applied: a wheeled vehicle cannot slide sideways, so lateral velocity is constrained toward zero except for the small amount consistent with normal steering. When a road match is available, the NHC is applied relative to the matched road segment's own bearing rather than a generic forward-only assumption; it falls back to the last real GPS-derived heading when no road match is available yet.

**Road graph.** Built from OpenStreetMap data fetched live from the Overpass API, centered on the phone's actual first GPS fix (roughly a 2.2 km radius), and refetched as the vehicle approaches the edge of the cached area. This works anywhere OpenStreetMap has road coverage, not one fixed demo city. Every successful fetch is also written to an IndexedDB-backed cache (`RoadGraphCache`), keyed by a coarse rounded lat/lon tile, so a later fetch failure (no signal, Overpass rate-limited, or a network drop exactly at a tunnel entrance) can fall back to a previously cached nearby tile instead of leaving road matching off. If neither a live fetch nor a cached tile is available, road matching simply stays off; nothing is fabricated in that case.

**Road matching.** This is a greedy, per-tick nearest-candidate selection, not a full probabilistic Hidden Markov Model with an accumulated path. At each tick, up to `K=5` nearby candidate segments are scored by an emission term (how well the segment's distance and bearing agree with the current fused position and heading) plus a transition term (how well the actual movement since the last selected segment agrees with the segment implied by real speed times elapsed time). The single highest-scoring candidate is kept as that tick's selected state, and the previous tick's selection carries forward as real persisted state for the next tick's transition scoring. This is a real, working, and useful piece of engineering, but it should not be described as a full Viterbi decode over an accumulated log-probability path, because that is not what the current implementation does.

**GNSS reacquisition.** When a GNSS fix returns after a blackout, the drift error at that moment (distance between the dead-reckoning estimate and the new fix) and the blackout duration are measured and displayed, and the EKF reconciles its state with the new fix.

---

## 9. Sensor Handling

| Sensor | Status handling |
|---|---|
| Accelerometer | `DeviceMotionEvent` in-browser (or a native plugin in a native build). Tracked through the `AVAILABLE`/`ACTIVE`/`NO_DATA`/`PERMISSION_DENIED`/`API_BLOCKED`/`UNAVAILABLE` vocabulary. |
| Gyroscope | Same handling as accelerometer, via `DeviceMotionEvent`'s rotation rate or `DeviceOrientationEvent`. |
| Orientation | `DeviceOrientationEvent` in-browser. |
| Magnetometer / compass | Read where the browser exposes it; shown in the diagnostics panel with an explicit state rather than assumed present. |
| Barometer | Read via the Generic Sensor API's `Barometer` class where supported. **iOS Safari (WebKit) does not implement this API at all**, which the code detects and reports as a browser-level block (`API_BLOCKED`), not as a bug or a missing permission. This is a documented, honest platform limitation, not something NAVDRIFT-0 can work around from the browser. |
| GNSS | `navigator.geolocation.watchPosition` in-browser (or a native plugin in a native build, using the platform's own location manager). |

**Native bridges.** Real Capacitor plugins exist in source form for both Android (`native/android-plugin/NavdriftSensorsPlugin.kt`, using `SensorManager`, `LocationManager`, and `Sensor.TYPE_PRESSURE` for a real barometer where the device has one) and iOS (`native/ios-plugin/NavdriftSensorsPlugin.swift`, using `CoreLocation`, `CoreMotion`, and `CMAltimeter` for a real barometer reading, which is how a native iOS build can get barometer data that the web version cannot due to the WebKit limitation above). A bridge script (`frontend/native-bridge.js`) routes these native sensor events into the same `SensorManager` interface the browser sensor code already uses, so the rest of the pipeline does not need to know whether it is running natively or in a browser.

**As of this writing, these native builds have not been compiled and run on physical Android or iOS hardware.** Building them requires running Android Studio or Xcode on real hardware or a real emulator, which has not happened yet in this project. Until that happens, the web version (used directly in a mobile browser, or installed to the home screen as a PWA) is what has actually been tested and is what a reviewer should expect to test. Do not treat the native bridge as production-validated; treat it as complete, real source code that has not yet been exercised on a device.

There is also a stale, unrelated file at the repository root, `android/NavDriftService.kt`: despite its name and extension, its actual content is an old HTML file, not Kotlin, and it is not used by anything. The real Android native code is `native/android-plugin/NavdriftSensorsPlugin.kt`.

---

## 10. On-Device / Edge Deployment

All five ONNX models listed in section 7 run entirely in-browser via `onnxruntime-web` (WASM), on both the mobile client and the desktop dashboard. This means the actual navigation inference (DriftFormer's correction, the adaptive EKF noise prediction, and tunnel detection) can run without any backend or cloud service, once the page itself and its sensor data are available.

What still requires network access, and should not be described as offline: loading the page and its CDN-hosted dependencies (MapLibre GL JS, the OpenFreeMap vector map tiles, fonts) the first time; fetching new road graph data from the Overpass API for an area that has not been cached yet (a previously cached area can fall back to the IndexedDB cache without network); and the separate backend API described in section 3, which is unrelated to the on-device inference pipeline and currently runs in demo mode.

Model sizes, confirmed from the actual `.onnx` files and from `results/isro_benchmark_table.csv`: DriftFormer 0.036 MB, IMU Denoiser 0.026 MB (FP32) / 0.149 MB (INT8), Adaptive EKF predictor 0.006 MB, Tunnel Detector 0.014 MB, NavIC DOP predictor 0.004 MB. Measured FP32 inference latency for the combined pipeline was 4.853 ms in the offline benchmark recorded in `results/validation_full.json` (see section 14 for exactly how that number was produced and its limits).

---

## 11. Map and Road Data

The mobile client uses **MapLibre GL JS** (loaded from a CDN, version 4.7.1) with the **OpenFreeMap Liberty** vector tile style, which serves OpenStreetMap-derived vector tiles from `tiles.openfreemap.org`. The desktop dashboard uses **Leaflet** with its own separate map setup. Road data for road-aware navigation comes from the **Overpass API**, queried live around the phone's real GPS fix, with results cached in an **IndexedDB**-backed store (`RoadGraphCache`) so a repeat visit to the same area, or a network drop after an earlier successful fetch, can fall back to cached data instead of leaving road matching off entirely. A first-ever visit to a brand-new area with no network at the moment GNSS is lost has no cached data to fall back to, and correctly shows road matching as unavailable rather than inventing a match.

---

## 12. Safety / Numerical Stability / Engineering Hardening

These are engineering safeguards confirmed in the current code, not claims:

- Non-finite GNSS fixes (`NaN`/`Infinity` latitude or longitude) are discarded before they can reach the position state, both in the web sensor handler and in the native bridge.
- DriftFormer's correction is rejected if the implied physical displacement exceeds a bound derived from a maximum plausible speed (60 m/s) times the real elapsed time since the last correction; rejected and applied corrections are both counted and shown live.
- ZUPT is gated on a measured acceleration threshold and a sustained-stillness duration, rather than firing on every low reading, specifically to avoid arresting a vehicle that is still genuinely moving slowly.
- The road graph reinitializes its matching state when the graph itself has been reloaded (a new Overpass fetch or cache swap), rather than scoring against stale segment indices that no longer correspond to anything real.
- Road matching falls back cleanly to a plain nearest-segment lookup when there are too few real candidate segments to run the emission/transition scoring meaningfully, rather than selecting from too little data.
- A 3-second sensor watchdog degrades any sensor's displayed status from `ACTIVE` to `NO_DATA` if readings genuinely stop arriving.
- The desktop dashboard's device router (see section 3) and the mobile client's map/WebGL initialization both include explicit failure-state detection and reporting rather than failing silently.
- The backend's exception handler never leaks a stack trace to the client; it logs internally and returns a generic error.

---

## 13. Current Results / Benchmarks

This section separates three different kinds of number, since mixing them is exactly the kind of overclaiming this README is meant to avoid.

### A. Offline benchmark results

From `results/validation_full.json` and `results/isro_benchmark_table.csv`, produced by `navdrift_colab/navdrift_08_validate.py` against a held-out IO-VNBD test sequence:

```
ATE RMSE:                 0.2472 m
Mean drift:                0.023%
Max drift:                  1.742%
Steps under 10% target:    100.0%
Steps under 5%:            100.0%
Total steps evaluated:       36,819
Number of test sequences:         1
Pipeline latency (FP32):    4.853 ms
Pipeline latency (INT8):    4.815 ms
```

Per-model figures (FP32, from `results/isro_benchmark_table.csv`): DriftFormer 3.778 ms mean, IMU Denoiser 0.555 ms mean (FP32) / 4.815 ms mean (INT8), Adaptive EKF predictor 0.052 ms mean, Tunnel Detector 0.424 ms mean, NavIC DOP predictor 0.044 ms mean.

**These numbers come from a single held-out test sequence** (`n_test_sequences: 1` in the raw JSON), not a broad multi-drive statistical evaluation. They are real, computed offline validation numbers, and they are labeled here exactly as what they are: one sequence's worth of offline evaluation, not a general claim about performance across arbitrary conditions.

### B. Live physical test observations

No live physical drive test log currently exists in this repository. Nobody has yet recorded the phone app driving through a real GNSS-denied stretch (a tunnel, an underpass, a parking structure) and computed drift from that recorded log. Until that test is done and its data is added here, there is no live physical benchmark to report, only the offline number above and whatever qualitative observations come out of the demo/testing process.

### C. Demonstration/UI metrics

Live tiles shown in the desktop dashboard and mobile diagnostics panel (inference latency, corrections applied/rejected, GNSS fix age, and so on) reflect real measured runtime values during a session, not a formal benchmark; they are not directly comparable to the offline numbers in part A, and are not presented as such in the UI.

---

## 14. What Has Been Built

- [x] Smartphone GNSS integration (`navigator.geolocation.watchPosition`)
- [x] Smartphone IMU integration (`DeviceMotionEvent` / `DeviceOrientationEvent`)
- [x] Orientation handling with sensor status tracking (`SensorManager`)
- [x] Extended Kalman Filter maintaining position, heading, and uncertainty
- [x] Zero-velocity update (ZUPT)
- [x] Non-holonomic motion constraint, tied to matched road bearing when available
- [x] GNSS blackout handling with dead reckoning
- [x] DriftFormer ONNX model, running on-device via onnxruntime-web, gated to blackout-only correction with a physical sanity bound
- [x] Adaptive EKF noise-predictor ONNX model, running on-device
- [x] Tunnel Detector ONNX model, running on-device
- [x] NavIC DOP predictor ONNX model, running on-device
- [x] Live OpenStreetMap road graph fetch via Overpass API, centered on the real GPS fix
- [x] Greedy per-tick road matching (emission + transition scoring over nearby candidates; see section 8 for what this is and is not)
- [x] IndexedDB-backed offline road graph cache with radius-based fallback
- [x] GNSS reacquisition drift measurement, persisted in the diagnostics panel
- [x] MapLibre GL map (mobile) and Leaflet map (desktop)
- [x] Diagnostics panels on both mobile and desktop covering sensor state, pipeline state, and AI Fusion/road-aware navigation counters
- [x] Desktop Mission Control dashboard, separate from the mobile navigation engine
- [x] Single-URL production device routing, using `navigator.userAgentData.mobile` as the primary signal with a touch/viewport fallback for browsers that don't expose it
- [x] Numerical safety checks: non-finite GNSS fix rejection, correction physical-bound rejection, sensor watchdog, road-graph epoch reinitialization
- [x] Offline validation pipeline producing ATE, drift percentage, and per-model latency numbers against IO-VNBD

---

## 15. What Is Partially Built / Experimental

- **Native Android and iOS sensor bridges.** Real, complete Kotlin and Swift plugin source code exists and is wired into the shared web engine through `native-bridge.js`, but has not been compiled and run on physical Android or iOS hardware yet (see section 9).
- **Backend "real" inference mode.** The code path exists (`_onnx_infer` in `api/app.py`), but expects a model file and input shape that does not match any model actually present in this repository, and the deployed configuration forces demo mode regardless. Treat the deployed backend as demo-only until a matching model is actually wired in and tested.
- **Road matching as a full probabilistic map-matcher.** The current implementation is a real, working, greedy per-tick nearest-candidate selection with emission and transition scoring, not a full Viterbi decode over an accumulated path (see section 8).
- **Additional/broader training datasets.** EuRoC MAV is referenced as prior cross-validation data; there is no evidence in the current repository of an active, wired-in multi-dataset training pipeline beyond IO-VNBD.
- **The more elaborate PyTorch model architectures under `models/`** (transformer with RoPE and a covariance head, GRU-VAE, gradient-descent SNAP corrector) versus the simpler architecture actually reflected in the deployed `.onnx` files (see section 7). It is not clear from the repository which of these, if either, produced the currently deployed models.
- **Broader device/browser validation.** Testing so far has been on the browsers and devices used during development; broad compatibility testing across many phone models and browser versions has not been documented here.
- **IO-VNBD speed-estimation LSTM.** A real LSTM was trained on the official IO-VNBD dataset (Google Colab A100) and evaluated on a real held-out test split: MAE 3.213 m/s, RMSE 4.355 m/s, R² 0.6965 (`checkpoints/iovnbd_speed_lstm/training_report.json`). It runs correctly in-browser via `onnxruntime-web` (`results/iovnbd/browser_demo/`), but is **not** wired into `frontend/mobile.html` — the model was trained on gravity-inclusive smartphone accelerometer data with no established phone-mount axis convention, while the live app's IMU values are gravity-compensated and the phone orientation is treated as arbitrary/auto-detected; no calibration mapping is assumed to close that gap. See `results/EVIDENCE_INDEX.md` (Requirement 2A) for the full record.

---

## 16. What Is Not Built Yet (Remaining Work)

These are gaps found during this repository audit, listed honestly as things to do next, not as things that already exist:

- Full native Android validation: build and run the Capacitor Android app with the real sensor plugin on physical hardware.
- Full native iOS validation: the same, on physical iOS hardware, which requires a Mac and Xcode.
- A real, recorded physical drive through an actual GNSS-denied stretch, with drift computed from that log rather than only from the offline IO-VNBD test set.
- A stronger, validated road-matching implementation, if a full probabilistic map-matcher is desired over the current greedy per-tick approach.
- A production backend inference path that is actually connected to a real, matching trained model, rather than the current demo-only deployment.
- No IMDAA or INSAT integration exists in the repository; neither is claimed as implemented.
- Broader sensor/device compatibility validation across phone models and browsers.
- A formal, repeatable field benchmark protocol for live drift measurement (methodology, not just a single offline sequence).
- Automated deployment and testing (there is a CI workflow that runs Python import checks and unit tests, but no automated end-to-end or device testing).
- Resolving the `CORS_ORIGINS` / `ALLOWED_ORIGINS` environment variable name mismatch in the backend deployment configuration.
- Reconciling the training-source model architectures under `models/` with whatever actually produced the deployed `.onnx` files, so the two are no longer in tension.
- Live-phone integration of the trained IO-VNBD speed LSTM: a real, documented sensor-mapping gap (gravity inclusion, phone-mount axis convention — see section 15) blocks this, and closing it honestly requires either a resolved calibration/mapping or a validated assumption, neither of which exists yet.

These are the areas intended for future work, not capabilities already present.

---

## 17. Development Roadmap

**Phase 1: Core navigation.** GNSS integration, IMU integration, EKF, ZUPT, dead reckoning. Done.

**Phase 2: AI-assisted dead reckoning.** DriftFormer, adaptive EKF noise prediction, tunnel detection, all running on-device via ONNX, gated to blackout-only correction with a physical sanity bound. Done.

**Phase 3: Road-aware navigation.** Live OpenStreetMap road graph via Overpass, IndexedDB offline caching, greedy per-tick road matching, non-holonomic constraint tied to matched road bearing. Done, with the open item of moving from greedy per-tick matching to a fuller probabilistic map-matcher if warranted.

**Phase 4: Device/native integration.** Capacitor native app shell, real Android and iOS sensor plugins written and wired through a shared bridge. Source complete; physical device validation still to do.

**Phase 5: Validation.** Offline IO-VNBD validation complete (one held-out sequence). A real physical GNSS-blackout drive test, and a formal repeatable field benchmark protocol, are the next steps.

**Phase 6: Production hardening.** Reconciling the backend's demo-vs-real inference path with an actual matching deployed model, fixing the CORS environment variable mismatch, and broader device/browser compatibility testing.

---

## 18. Running the Project

### Prerequisites

- Python 3.10 or newer
- Git
- Node.js, only if building the native Capacitor apps (see section 9)

### Clone and install (backend / evaluation tooling)

```bash
git clone https://github.com/swatijs3017/navdrift0
cd navdrift0
pip install -r requirements-api.txt
```

`requirements-api.txt` is the production/runtime dependency set (FastAPI, onnxruntime, numpy, scipy, huggingface_hub, and so on). `requirements.txt` additionally includes training-time dependencies (`torch`, `torchvision`, `wandb`, `gradio`, `folium`, `matplotlib`).

### Run the backend locally in demo mode

```bash
DEMO_MODE=true uvicorn api.app:app --host 0.0.0.0 --port 8000
```

This matches the deployed Render configuration and returns simulated pose data (see section 3 for exactly what that means).

### Run the backend with environment variables

Copy `.env.example` to `.env` and fill in real values; `.env` is git-ignored and should never be committed. The variables that file documents: `NAVDRIFT_API_KEY` (also read as `API_KEY` by `api/app.py`), `ONNX_PATH`, `NORM_STATS_PATH`, `ALLOWED_ORIGINS` (note the mismatch with the code's actual `CORS_ORIGINS` variable, section 3), `WINDOW`, `IMU_HZ`, `DEMO_MODE`, and an optional `WANDB_API_KEY`.

### Frontend (mobile and desktop interfaces)

`frontend/` is a static site with no build step. Serve it with any static file server, for example:

```bash
cd frontend
python -m http.server 8080
```

Then open `http://localhost:8080/index.html` in a browser to go through the same device-aware router used in production, or open `mobile.html` / `desktop.html` directly.

### Tests

```bash
pytest tests/ -v
```

`tests/test_api.py` exists in the repository; this is what the CI workflow (`.github/workflows/ci.yml`) runs on every push and pull request to `main`, alongside a basic Python import check of `api.app`.

### Native apps

See section 9 and `NATIVE_BUILD.md` for the exact, real commands (`npm install`, `npx cap add android` / `npx cap add ios`, copying the plugin files in, and building from Android Studio or Xcode). These have not yet been run to completion on physical hardware as part of this project.

---

## 19. Production Website

**https://navdrift0.pages.dev/**

- Laptop or desktop browser: opens the Mission Control dashboard (`frontend/desktop.html`).
- Phone or tablet: opens the NAVDRIFT mobile navigation interface (`frontend/mobile.html`).

Both are served from this single URL; there is no separate `/mobile` or `/desktop` URL that a person needs to know about. No other production URL is claimed here.

---

## 20. Repository Structure

```
navdrift0/
|-- index.html                    Thin device router (repo root copy)
|-- frontend/
|   |-- index.html                Actual production entry point served at "/" (Cloudflare Pages
|   |                              build output directory is frontend/); device-aware router
|   |-- mobile.html                The real navigation client: sensors, EKF, ONNX pipeline,
|   |                              road graph, diagnostics
|   |-- desktop.html               Mission Control dashboard (Leaflet map, tabbed panels)
|   |-- native-bridge.js           Routes native Capacitor sensor events into the same
|   |                              SensorManager interface the browser code uses
|   |-- manifest.json              PWA manifest
|   |-- sw.js                      Service worker (never intercepts page navigation; caches two
|   |                              static CDN assets only)
|   `-- models/                    ONNX models loaded for in-browser inference
|-- api/
|   `-- app.py                     FastAPI backend: /init, /ingest, /gnss_lost, /reacquire,
|                                  /trajectory, /status, /reset, /ws/stream (see section 3)
|-- inference/
|   |-- runtime.py                 A second runtime implementation, not currently used by api/app.py
|   `-- export_onnx.py             Present in the repo, but its actual content does not match
|                                  its filename (see section 7)
|-- models/
|   |-- drift_former.py            PyTorch DriftFormer source (a more elaborate architecture than
|   |                              the deployed ONNX model; see section 7)
|   |-- navic_vae.py                PyTorch NavIC VAE source
|   |-- snap_corrector.py           PyTorch SNAP corrector source
|   `-- *.onnx                      The actual deployed models (see section 7 for confirmed shapes)
|-- training/
|   |-- train_drift_former.py       DriftFormer training script
|   `-- train_navic_vae.py          NavIC VAE training script
|-- navdrift_colab/
|   `-- navdrift_08_validate.py     The validation script that produced results/
|-- notebooks/
|   |-- NAVDRIFT0_Training.ipynb    Colab training notebook
|   `-- NAVDRIFT0_Training.py
|-- data/
|   `-- loader.py                   IO-VNBD dataset loader and preprocessor
|-- eval/
|   `-- metrics.py                  ATE / RTE / NLL / drift-rate metric implementations
|-- demo/
|   `-- demo.py                     Standalone demo script
|-- android/
|   `-- NavDriftService.kt          Stale leftover HTML, not real Kotlin, not used (see section 9)
|-- native/
|   |-- android-plugin/NavdriftSensorsPlugin.kt   Real native Android sensor plugin source
|   `-- ios-plugin/NavdriftSensorsPlugin.swift    Real native iOS sensor plugin source
|-- results/
|   |-- validation_full.json        Offline validation output (see section 13)
|   |-- isro_benchmark_table.csv    Per-model size/latency table
|   `-- compliance_curve.png
|-- checkpoints/
|   `-- driftformer_best.pt         A PyTorch training checkpoint
|-- tests/
|   `-- test_api.py                 Backend tests run in CI
|-- .github/workflows/ci.yml        CI: Python import check + pytest, on push/PR to main
|-- capacitor.config.json           Capacitor native app shell configuration
|-- render.yaml                     Render backend deployment configuration
|-- requirements.txt                Full dependencies, including training
|-- requirements-api.txt            Production/runtime dependencies only
|-- .env.example                    Environment variable reference
|-- NATIVE_BUILD.md                 Real, detailed native build instructions
`-- TODO_MAP_MATCHING.md            Honest running log of road-matching implementation status
```

---

## 21. Known Limitations

- Browser sensor APIs differ meaningfully by platform. Most notably, iOS Safari (WebKit) does not implement the Generic Sensor API's `Barometer` class at all, which is a browser-level restriction, not something this project can fix from the web app; a native iOS build reads the barometer directly instead.
- The native Android and iOS sensor bridges are real, complete source code that has not yet been built and run on physical hardware.
- Road matching depends on network access to the Overpass API for any area not already in the offline IndexedDB cache; a first-ever blackout in a brand-new area with no network at that moment has no road data to match against, by design, rather than fabricating one.
- The backend API currently only meaningfully runs in demo mode; its "real inference" code path expects a model file and shape that is not present in this repository, and the deployed configuration forces demo mode regardless.
- A configuration mismatch exists between the backend's expected `CORS_ORIGINS` environment variable and the deployed `render.yaml`'s `ALLOWED_ORIGINS`, meaning CORS is currently effectively unrestricted (`*`) in production.
- Road matching is a greedy per-tick nearest-candidate selection, not a full probabilistic HMM/Viterbi map-matcher; see section 8 for exactly what it does.
- The only benchmark numbers currently available are offline validation numbers from a single held-out IO-VNBD test sequence; no live physical GNSS-blackout drive test has been recorded and analyzed yet.
- There is a real discrepancy between the more elaborate PyTorch model source code under `models/` and the simpler architecture actually reflected in the deployed ONNX models; see section 7.

---

## 22. Transparency Note

This README distinguishes, deliberately and throughout: what is implemented and confirmed by reading the actual code versus what is experimental or partially built; offline benchmark numbers versus live physical observations versus demonstration/UI metrics; and what has been built versus what remains as future work. Nothing here claims a capability that was not confirmed against the repository during this pass, and known gaps (including a couple of real inconsistencies between different parts of the codebase) are stated plainly rather than smoothed over. This matters for anyone evaluating the project technically: a judge, mentor, or developer should be able to tell exactly what is running live, what is offline-validated, and what is still open.

---

## License

MIT. Copyright 2026 NAVDRIFT-0 Team. ISRO Smart India Hackathon 2026, Problem Statement 26168.
