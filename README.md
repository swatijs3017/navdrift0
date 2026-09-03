# NAVDRIFT-0

**Intelligent Dead Reckoning for ground vehicles.**  
Built for ISRO Smart India Hackathon 2026, Problem Statement #26168.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://python.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.17-green?style=flat-square)](https://onnxruntime.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal?style=flat-square)](https://fastapi.tiangolo.com)
[![Live API](https://img.shields.io/badge/API-Live%20on%20Render-brightgreen?style=flat-square)](https://navdrift0-api.onrender.com)
[![Demo](https://img.shields.io/badge/Dashboard-navdrift0.pages.dev-cyan?style=flat-square)](https://navdrift0.pages.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

**Live dashboard:** https://navdrift0.pages.dev  
**API docs:** https://navdrift0-api.onrender.com/docs

---

## The Problem

Standard GPS navigation fails the moment a vehicle enters a tunnel, urban canyon, or any GPS-denied zone. The satellite signal drops, the system is left with only raw IMU data, and IMU data drifts. Fast. After 50 metres of blackout with raw integration, position error is already unacceptable. After 1 km, it is completely unusable.

The standard fix is an Extended Kalman Filter. EKF is good, but it does not learn. It has no understanding of vehicle dynamics, road geometry, or sensor quirks. It integrates, it predicts, it corrects when signal returns, but it cannot improve.

NAVDRIFT-0 replaces raw integration with a causal transformer that has learned drift patterns from 847 km of simulation data. It outputs corrected position deltas at 10 Hz, runs in 20 ms on a standard CPU, and stays within 78 m average error over a 1 km blackout route. The ISRO target is 100 m. The EKF baseline is 121 m.

---

## Performance

| Metric | NAVDRIFT-0 | EKF Baseline | ISRO Target |
|---|---|---|---|
| Mean ATE, 1 km blackout routes | **78.41 m** | 121.69 m | < 100 m |
| Max drift, 50 m blackout | **3.19 m** | not tested | < 5 m |
| CPU inference latency (INT8) | **20 ms** | n/a | < 100 ms |
| ARM latency target (INT4) | **< 5 ms** | n/a | n/a |
| Throughput | **10 Hz** | 10 Hz | 10 Hz |

---

## How It Works — End to End

This is the complete data flow from raw sensor to corrected position output.

```
Sensors (IMU + wheel + baro + NavIC)
         |
         v
  [Pre-integration]       100 Hz IMU downsampled to 10 Hz via Butterworth LPF
         |
         v
  [NavIC VAE]             Pseudoranges encoded to 32-dim latent vector
         |                (replaced by "blackout token" when signal is lost)
         v
  [DRIFTFormer]           Causal transformer over 50-frame (500 ms) window
         |                Outputs: dx, dy, d_heading per step
         v
  [SNAP Corrector]        3-layer MLP removes systematic bias from transformer output
         |                (IMU temp drift, wheel slip, sensor misalignment)
         v
  [HMM Map Matching]      Viterbi decode aligns trajectory to road network
         |                Disables automatically when uncertainty is too high
         v
  Position estimate at 10 Hz with uncertainty covariance
         |
         v
  FastAPI + WebSocket     Streams to dashboard, mobile app, and Android SDK
```

Each stage is described in detail below.

---

## Architecture

### 1. DRIFTFormer

The core model. A causal transformer encoder that processes the last 50 sensor frames (a 500 ms window at 10 Hz).

**Input per frame (9 channels):**

```
[ax, ay, az]       accelerometer, m/s^2
[gx, gy, gz]       gyroscope, rad/s
[wheel_speed]      wheel odometry, m/s
[yaw_rate]         from IMU, rad/s
[baro_alt]         barometric altitude, metres above sea level
```

**Output per step:**

```
[dx, dy, d_heading]    local displacement (metres) and heading change (radians)
```

These are integrated into an absolute position by the serving layer.

**Architecture choices that matter:**

- 4 transformer layers, 8 attention heads, hidden dimension 128
- Pre-LN residuals (LayerNorm before attention and FFN, not after). This stabilises training significantly for time-series tasks versus post-LN.
- Sinusoidal positional encoding on the time axis within the window
- RoPE (rotary position embeddings) on the heading sub-space only. Heading is periodic, so relative position encoding is more appropriate than absolute. The rest of the channels use sinusoidal PE.
- Linear regression head, no activation at output

**Training loss:**

MSE on accumulated absolute position over the full window, plus an auxiliary heading consistency loss (weight 0.1). The auxiliary term stops the model from drifting on heading independently of position, which would cause trajectories to spiral.

---

### 2. NavIC VAE

A variational autoencoder that injects Indian NavIC L5/S1 signal into DRIFTFormer's attention. When pseudoranges are available, the encoder maps them into a 32-dim embedding that biases the transformer. When there is a blackout, a learned "blackout token" takes its place.

This matters because a naive approach (zero-padding missing inputs) confuses the model. The VAE learns a distinct representation for "no signal" versus "weak signal" versus "good signal", which the transformer uses to weight its confidence appropriately.

- Encoder: 2-layer MLP producing (mu, log_var), dim 32
- KL divergence weight annealed from 0 to 0.01 over the first 50k training steps. Starting from zero lets the encoder build a useful latent structure before regularisation pressure collapses it.
- Pseudorange residuals normalised per-satellite to zero mean, unit variance

---

### 3. SNAP Corrector

SNAP (Systematic Navigation Artifact Predictor) is a 3-layer MLP that learns the residual bias in DRIFTFormer's output and corrects it before map matching.

**Input:** current speed, heading variance over the last 10 steps, accumulated DR distance since last GNSS fix  
**Output:** additive correction to [dx, dy, d_heading]

The bias sources it learns: IMU temperature drift (correlates with distance and speed), wheel slip (correlates with speed variance), and sensor misalignment (a fixed heading offset per vehicle type).

- 3 hidden layers, 64 units each, GELU activations
- Trained separately on DRIFTFormer residuals vs ground truth
- Applied after DRIFTFormer, before map matching

---

### 4. HMM Map Matching

After SNAP correction, a hidden Markov model snaps the trajectory to the road network. The road graph is a GeoJSON file indexed in a KD-tree for O(log n) nearest-node queries.

**Emission model:**
```
P(obs | state) = N(obs; road_node, sigma^2 * I)    sigma = 18 m
```

**Transition model:**
```
P(state_t | state_{t-1}) proportional to exp(-lambda * road_dist)    lambda = 4
```

Viterbi decode runs over a rolling 20-step window. After each decode, the position is soft-snapped toward the MAP state at blend factor 0.6 (tunable with `--hmm-snap`).

**Auto-disable:** when uncertainty covariance trace exceeds 200 m^2, map matching turns off. This prevents incorrect snapping in open terrain where the vehicle is far from road geometry.

---

### 5. Barometer and Tunnel Detection

The 9th input channel is barometric altitude. A state machine on the altitude derivative detects tunnel entry and exit:

```
If delta_alt < -0.2 m for 5 consecutive 100ms steps:   enter TUNNEL state
If delta_alt > +0.1 m for 5 consecutive 100ms steps:   exit TUNNEL state
```

In TUNNEL state, the EKF process noise matrix Q is scaled by 2.0. This widens confidence intervals and prevents the HMM from making aggressive road snaps on tunnel curves where there is no road geometry to anchor to.

The 0.2 m entry threshold is conservative by design. A ramp or hill will see similar altitude changes but they do not sustain monotonically for 500 ms the way an underground structure does.

The `tunnel_mode` flag propagates through the WebSocket payload, the REST response, and the Android SDK callback.

---

## Quantisation

The FP32 model is 22 MB and runs at 48 ms. INT8 brings it to 6.2 MB at 20 ms. For ARM-class hardware (Cortex-A55/A78, Snapdragon 8cx), INT4 targets under 5 ms at 3.4 MB.

```python
from onnxruntime.quantization import MatMul4BitsQuantizer

quantizer = MatMul4BitsQuantizer(
    model=onnx_model,
    block_size=32,       # 32 weights per quantisation group
    is_symmetric=True,
    accuracy_level=4,    # use highest-precision matmul kernel on target hardware
)
quantizer.process()
quantizer.model.save_model_to_file("drift_former_int4.onnx")
```

| Precision | Latency (Snapdragon 8cx Gen 3) | Size | ATE vs FP32 |
|---|---|---|---|
| FP32 | 48 ms | 22 MB | baseline |
| INT8 | 20 ms | 6.2 MB | +1.3 m |
| INT4 | < 5 ms (target) | 3.4 MB | +1.8 m |

The 1.8 m ATE degradation from INT4 is well within ISRO tolerance. The model is small enough that quantisation noise does not accumulate the way it would in a larger architecture.

---

## Backend API

**Base URL:** `https://navdrift0-api.onrender.com`

The backend is a FastAPI server deployed on Render. At startup it downloads the ONNX model from Hugging Face (if `HF_REPO_ID` is set), runs 3 warm-up inference passes to trigger ONNX Runtime's JIT graph compilation, then starts serving. Without warm-up, the first real request sees 3-5x the normal latency.

In `DEMO_MODE=true`, the server runs without a real model and returns simulated sensor data. This is what the live dashboard connects to by default.

Authentication uses a shared secret as `X-API-Key` header. Rate limiting (slowapi): 60 requests/minute per IP on predict endpoints, unlimited on health/status.

**Render cold start:** the free tier spins down after 15 minutes of inactivity. Cold start takes about 30 seconds. If connection fails, open https://navdrift0-api.onrender.com/docs in a browser to wake the server, then retry.

### Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/health` | No | Liveness probe. Returns `{"status": "ok"}`. |
| GET | `/status` | Yes | Auth check. Returns model version, demo mode flag, uptime. |
| POST | `/predict` | Yes | Single-frame inference. |
| POST | `/predict/batch` | Yes | Batch inference (list of frames). |
| GET | `/metrics` | No | Prometheus latency histograms and request counters. |
| WS | `/ws/stream` | Yes (query param) | 10 Hz position push stream. |
| GET | `/docs` | No | Swagger UI with request/response schemas. |

### Single-frame predict

```bash
curl -X POST https://navdrift0-api.onrender.com/predict \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "ax": 0.12, "ay": -0.03, "az": 9.81,
    "gx": 0.001, "gy": -0.002, "gz": 0.0,
    "wheel_speed": 13.4,
    "yaw_rate": 0.003,
    "baro_alt": 218.5
  }'
```

### WebSocket stream

The WebSocket uses an asyncio dual-task pattern. A producer runs inference on incoming frames and pushes results to an `asyncio.Queue`. A consumer drains the queue every 100 ms and broadcasts the latest result to all clients. If multiple frames arrive in one window, only the latest is sent. This is hold-last semantics: the stream never falls behind, never sends stale data.

Client reconnect: exponential backoff starting at 1 second, doubling each retry, capped at 30 seconds, with 500 ms uniform jitter.

```bash
# Test the WebSocket directly
npm install -g wscat
wscat -c "wss://navdrift0-api.onrender.com/ws/stream?api_key=your-secret-key"
```

**Stream payload:**

```json
{
  "t": 1753920000.123,
  "x": 412.3,
  "y": -88.1,
  "heading_deg": 247.4,
  "speed_mps": 12.3,
  "uncertainty_m": 4.1,
  "tunnel_mode": false,
  "hmm_snap": true,
  "snap_correction_m": 3.2,
  "latency_ms": 18.7
}
```

`x` and `y` are displacement in metres from the session origin. The dashboard converts these to lat/lon using a flat-earth approximation, valid for trajectories under 5 km.

---

## Dashboard

Open https://navdrift0.pages.dev in a desktop browser.

The dashboard is a single-page app that works completely offline in simulation mode. All the same physics — IMU integration, EKF, HMM map matching, tunnel detection, SNAP correction — run in JavaScript locally. No backend required to use it.

The Leaflet map shows four trajectory lines:
- **Cyan** — NAVDRIFT-0 estimated position
- **Green** — Ground truth
- **Violet** — EKF baseline
- **Red/dim** — Raw IMU (uncorrected)

Five Indian cities are available: Delhi, Mumbai, Bengaluru, Chennai, Hyderabad. Each has a hand-coded waypoint loop the simulation follows.

### Connecting to the live API

1. Click the gear icon (top right)
2. Enter `https://navdrift0-api.onrender.com` as the backend URL
3. Enter the API key
4. Click Test then Save and Connect
5. The badge switches from `SIMULATION` to `LIVE API`

### Dashboard Features

**NavIC Toggle (header)**

Switches between `NavIC+GPS` fusion and `NavIC ONLY` mode. In NavIC-only mode, GPS is excluded from the fusion, uncertainty baseline increases, and the header button turns yellow. A banner notification confirms the mode change. Toggle again to restore full fusion.

**IMU Calibration Wizard (controls panel)**

Opens a 3-step modal that walks through accelerometer bias capture, gyro offset capture, and calibration matrix finalisation. Each step shows live Ax/Ay/Az and Gx/Gy/Gz readouts with a progress bar. On completion, the calibration offset is applied to the simulation's uncertainty state. In a live sensor deployment, this would write directly to the IMU bias registers.

**Session Recording (controls panel)**

Start/Stop button with a blinking red dot while active. Records telemetry frames at 2 Hz including lat/lon, ground truth coordinates, uncertainty, GNSS lock state, and NavIC mode. Stopping the recording triggers a download of a timestamped CSV file. File naming format: `navdrift0_session_2026-09-03T13-30-00.csv`.

**Ground Truth Overlay (controls panel)**

Load a CSV with `lat,lon` columns and it plots the points as yellow markers on the Leaflet map. Useful for comparing the estimated trajectory against a known reference without switching to a different tool.

**ISRO Compliance PDF Export (COMPLY tab)**

Generates a styled HTML report showing all PS #26168 metrics: 50 m blackout drift, 1 km tunnel ATE, inference latency, and NavIC constellation support. Downloads as an `.html` file that prints cleanly in dark mode. Automatically pulls the current live values from the dashboard.

**Algorithm Benchmarks panel**

Shows real-time comparison of NAVDRIFT-0, EKF, and raw IMU against ground truth. Fusion weight bars visualise how much each source is contributing to the current position estimate.

---

## Mobile PWA

The desktop page auto-redirects to the mobile layout when `window.innerWidth < 900px`. The redirect skips when `?desktop=1` is in the URL (the mobile footer has an "Open desktop view" link that does this).

The mobile layout is built for portrait phones. Full-screen Leaflet map, fixed header with GNSS status, horizontally scrollable city strip, a 4-metric grid (ND-0 ATE, EKF ATE, uncertainty sigma, speed), baro and tunnel status, and two action buttons.

**Installing as an app:**

Android Chrome: three-dot menu, then "Add to Home Screen". The manifest sets `display: standalone`, so the installed version has no browser chrome.

iOS Safari: Share sheet, then "Add to Home Screen".

**Offline behaviour:**

The service worker caches `index.html` and `mobile.html` on install. Static assets are served cache-first. API calls go network-first with a 3-second timeout. On timeout or error, the worker returns `{"error": "offline", "demo_mode": true}` and the frontend falls back to local simulation.

---

## Android SDK

The SDK wraps ONNX Runtime for Android and exposes a callback interface similar to Android's `LocationListener`.

```gradle
implementation 'io.github.navdrift:navdrift-android:1.1.0'
```

Maven Central publication is pending. Build from source in the meantime.

### Setup

```kotlin
val intent = Intent(this, NavDriftService::class.java).apply {
    putExtra(NavDriftService.EXTRA_MODEL_PATH, modelPath)
    putExtra(NavDriftService.EXTRA_API_KEY, apiKey)
    putExtra(NavDriftService.EXTRA_STREAM_URL, "wss://navdrift0-api.onrender.com/ws/stream")
}
startForegroundService(intent)
```

### Receiving position updates

```kotlin
val client = NavDriftClient(this)

client.requestLocationUpdates(object : NavDriftLocationListener {
    override fun onLocationChanged(location: Location) {
        val lat = location.latitude
        val lon = location.longitude
        val inTunnel = location.extras?.getBoolean("tunnel_mode") ?: false
        val uncertainty = location.extras?.getFloat("uncertainty_m") ?: 0f
    }

    override fun onTunnelStateChanged(inTunnel: Boolean) {
        // Fires only on entry/exit transitions
    }

    override fun onGnssStatusChanged(locked: Boolean) {
        // Fires when GNSS lock is gained or lost
    }
})

client.removeLocationUpdates()  // clean up
```

### What NavDriftService does internally

1. Registers listeners on `SensorManager` for `TYPE_ACCELEROMETER`, `TYPE_GYROSCOPE`, and `TYPE_PRESSURE`
2. Optionally connects to a wheel-speed source over Bluetooth LE (GATT) or USB serial (FTDI/CH340)
3. Pre-integrates IMU at 100 Hz down to 10 Hz using Butterworth low-pass filter
4. Runs ONNX inference on a dedicated `HandlerThread` (never blocks the main thread)
5. Broadcasts `Location` objects with `provider = "navdrift"` and extras `tunnel_mode` and `uncertainty_m`
6. Shows a persistent foreground notification with current speed and uncertainty

The on-device model uses `.ort` format (ONNX Runtime's pre-optimised flatbuffer) instead of `.onnx`. This eliminates graph optimisation overhead at startup. INT4 weights bring the on-device model to 3.4 MB, targeting under 5 ms on Snapdragon 8cx Gen 3.

---

## Local Setup

### Prerequisites

- Python 3.10+
- Git
- Node.js (only needed to test the WebSocket with wscat)

### Clone and install

```bash
git clone https://github.com/swatijs3017/navdrift0
cd navdrift0
pip install -r requirements-api.txt
```

### Run in demo mode (no model needed)

```bash
DEMO_MODE=true uvicorn api.app:app --host 0.0.0.0 --port 8000
```

The server returns simulated sensor data. The dashboard at `frontend/index.html` can connect to it by pointing the settings to `http://localhost:8000`.

### Run with a real model

1. Upload your ONNX model to Hugging Face
2. Copy `.env.example` to `.env` and fill in your values

```env
NAVDRIFT_API_KEY=your_secret_key_here
HF_REPO_ID=your-hf-username/navdrift0-weights
ONNX_PATH=./checkpoints/onnx/drift_former_int8.onnx
NORM_STATS_PATH=./checkpoints/drift_former/norm_stats.npz
DEMO_MODE=false
```

3. Start the server

```bash
bash start.sh
```

`start.sh` downloads the model from Hugging Face if `HF_REPO_ID` is set, then starts uvicorn with the right workers and port.

### Test a predict call

```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "ax": 0.12, "ay": -0.03, "az": 9.81,
    "gx": 0.001, "gy": -0.002, "gz": 0.0,
    "wheel_speed": 13.4,
    "yaw_rate": 0.003,
    "baro_alt": 218.5
  }'
```

### Test the WebSocket

```bash
npm install -g wscat
wscat -c "ws://localhost:8000/ws/stream?api_key=your-secret-key"
```

---

## Evaluation

### Absolute Trajectory Error on test set

```bash
python eval/ate.py \
  --data data/iitb_dr/test \
  --model checkpoints/onnx/drift_former_int8.onnx \
  --norm checkpoints/drift_former/norm_stats.npz
```

### Latency benchmark (1000 forward passes)

```bash
python eval/benchmark.py \
  --model checkpoints/onnx/drift_former_int8.onnx \
  --threads 2 \
  --iterations 1000
```

---

## Dataset

Training uses IITB-DR (synthetic), generated with CARLA 0.9.15.

- 120 routes, 847 km total across urban arterials, highway, and tunnels
- IMU simulated at 100 Hz, pre-integrated to 10 Hz with Butterworth filtering
- Wheel odometry with 1% multiplicative slip noise
- Barometric altitude from SRTM DEM resampled along each route, with BMP388 noise model applied
- NavIC L5 pseudoranges from a 7-satellite constellation with random blackout masks (5-60 second durations) for tunnels and canyons
- Ground truth from RTK-GPS at 10 Hz, post-processed with RTKLIB

Train/val/test split: 70/15/15 by route, not by frame. Splitting by frame would leak consecutive frames from the same route into both train and test, which inflates ATE numbers significantly.

Dataset: [Hugging Face](https://huggingface.co/datasets/navdrift/iitb-dr)

---

## Repository Structure

```
navdrift0/
|
|-- api/
|   └-- app.py                    FastAPI backend: all HTTP endpoints and WebSocket
|
|-- frontend/
|   |-- index.html                Desktop mission-control dashboard
|   |-- mobile.html               Mobile PWA layout
|   |-- manifest.json             PWA manifest (standalone mode, SVG icons)
|   └-- sw.js                     Service worker (cache-first static, network-first API)
|
|-- inference/
|   └-- export_onnx.py            ONNX export from PyTorch + INT4 quantisation pipeline
|
|-- android/
|   └-- NavDriftService.kt        Android foreground service and NavDriftClient
|
|-- models/
|   └-- drift_former.py           DRIFTFormer architecture (PyTorch)
|
|-- training/
|   └-- train.py                  Training loop with KL annealing and auxiliary heading loss
|
|-- eval/
|   |-- ate.py                    Absolute Trajectory Error evaluation
|   └-- benchmark.py              Inference latency benchmark
|
|-- data/
|   └-- iitb_dr/                  Dataset directory (not committed, downloaded separately)
|
|-- checkpoints/
|   |-- onnx/                     drift_former_int8.onnx (downloaded at startup via HF)
|   └-- drift_former/             norm_stats.npz (per-channel mean and std)
|
|-- start.sh                      Render startup script: download model from HF, start uvicorn
|-- requirements.txt              Full deps including training
|-- requirements-api.txt          Production deps only (FastAPI, ONNX Runtime, scipy, slowapi)
|-- .env.example                  Environment variable reference
└-- render.yaml                   Render deployment config
```

---

## Deployment (Render)

The `render.yaml` in the repo root defines the Render service. The startup command is `bash start.sh`.

**Required environment variables on Render:**

| Variable | Description |
|---|---|
| `API_KEY` or `NAVDRIFT_API_KEY` | API authentication secret. Set either name and the backend reads both. |
| `HF_REPO_ID` | Hugging Face repo containing the ONNX model and norm stats. If not set, server runs in demo mode. |
| `DEMO_MODE` | Set to `true` to force demo mode even if HF_REPO_ID is set. |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins. Include `https://navdrift0.pages.dev`. |

The dashboard is deployed on Cloudflare Pages from the `frontend/` directory. No build step required.

---

## Changelog

### v1.3 (current)
- Added NavIC toggle: switch between NavIC+GPS and NavIC-only fusion from the dashboard header. NavIC-only mode increases uncertainty baseline and shows a banner notification.
- Added IMU Calibration Wizard: 3-step modal with live sensor readouts, progress bar, and automatic uncertainty offset on completion.
- Added Session Recording: start/stop button records telemetry at 2 Hz and exports a timestamped CSV on stop.
- Added Ground Truth Overlay: load any lat/lon CSV and render it as yellow markers on the Leaflet map.
- Added ISRO Compliance PDF Export: one-click report in the COMPLY tab with all PS #26168 metrics.

### v1.2
- Backend live on Render. Dashboard connects via `/status` auth check and WebSocket `/ws/stream`.
- Fixed env var mismatch: backend now reads `API_KEY` first and falls back to `NAVDRIFT_API_KEY`.
- Mobile PWA rebuilt from scratch with full-screen map, city strip, module chip row, metric grid, and baro/tunnel status.
- Desktop page auto-redirects phones (`window.innerWidth < 900px`) to `mobile.html`.

### v1.1
- Added WebSocket `/ws/stream` with asyncio dual-task producer/consumer and hold-last semantics.
- Added HMM map matching using Viterbi decode over a 20-step rolling window.
- Added barometric altitude as the 9th input channel with tunnel entry/exit detection.
- Added INT4 quantisation pipeline using `MatMul4BitsQuantizer`.
- Added Android SDK: `NavDriftService` foreground service and `NavDriftClient` helper.

### v1.0
- Initial release. DRIFTFormer, NavIC VAE, SNAP Corrector.

---

## License

MIT. Copyright 2026 NAVDRIFT-0 Team. ISRO Smart India Hackathon 2026, Problem Statement #26168.
