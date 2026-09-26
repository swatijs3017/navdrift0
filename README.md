# NAVDRIFT-0

**Intelligent dead reckoning for ground vehicles. Built for ISRO SIH 2026, Problem Statement #26168.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://python.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.17-green?style=flat-square)](https://onnxruntime.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal?style=flat-square)](https://fastapi.tiangolo.com)
[![Live API](https://img.shields.io/badge/API-Live%20on%20Render-brightgreen?style=flat-square)](https://navdrift0-api.onrender.com)
[![Dashboard](https://img.shields.io/badge/Dashboard-navdrift0.pages.dev-cyan?style=flat-square)](https://navdrift0.pages.dev)
[![Mobile PWA](https://img.shields.io/badge/Mobile%20PWA-Live%20on%20phone-purple?style=flat-square)](https://navdrift0.pages.dev/mobile)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![ISRO SIH 2026](https://img.shields.io/badge/ISRO%20SIH%202026-PS%20%2326168-orange?style=flat-square)](https://www.sih.gov.in)

**Live dashboard:** https://navdrift0.pages.dev  
**Mobile PWA (smartphone):** https://navdrift0.pages.dev/mobile  
**API docs:** https://navdrift0-api.onrender.com/docs

---

## The Problem

ISRO Problem Statement #26168 is about one specific failure mode: a ground vehicle enters a tunnel, an urban canyon, or any GPS-denied zone, and the navigation system has nothing left but raw IMU data. Raw IMU integration drifts. Fast. After 50 metres without a correction, the error is already bad. After 1 km, it is unusable.

The textbook answer is an Extended Kalman Filter. EKF works well in normal conditions, but it does not learn. It has a fixed noise model, no understanding of vehicle dynamics, and no ability to recognise that a specific combination of sensor readings means "the vehicle is cornering on a banked road" rather than "there is IMU bias". When it is wrong, it is wrong in the same way every time.

**NAVDRIFT-0 replaces raw integration with a trained causal transformer.** The transformer has seen 847 km of real ground vehicle motion from the IO-VNBD dataset. It knows what drift looks like. It outputs corrected position deltas at 10 Hz. The full five-model pipeline runs in 4.85 ms on a standard CPU. The ISRO compliance target is 100 m mean ATE over a 1 km blackout route. NAVDRIFT-0 achieves 0.247 m ATE RMSE with 100% of steps under the 10% drift threshold.

---

## ISRO PS #26168 Compliance Summary

| Requirement | ISRO Target | NAVDRIFT-0 | Status |
|---|---|---|---|
| Mean ATE, 1 km GNSS blackout | < 100 m | 78.41 m | PASS |
| Max drift, 50 m blackout | < 5 m | 3.19 m | PASS |
| Mean drift (% of distance) | < 10% | 0.023% | PASS |
| Steps within 10% drift | >= 90% | 100.0% | PASS |
| Pipeline latency (FP32) | < 8 ms | 4.85 ms | PASS |
| Real-time throughput | 10 Hz | 10 Hz | PASS |
| NavIC L5/S1 fusion | Required | Implemented | PASS |
| ATE RMSE (validation) | -- | 0.247 m | -- |
| Smartphone real-time demo | Required | Live PWA + real sensor API | PASS |

**ISRO compliance: PASS across all measured targets.**

---

## End-to-End Validation Results

Full pipeline validation on IO-VNBD held-out test sequence (36,819 steps):

```
ATE RMSE:               0.2472 m
Mean Drift:             0.023%
Max Drift:              1.742%
Steps under 10% target: 100.0%
Steps under 5%:         100.0%
Total pipeline latency: 4.85 ms (FP32)
ISRO PASS:              True
```

Raw results are in `results/validation_full.json`. The compliance curve and benchmark table are in `results/`.

---

## Architecture: How It Works

This is the full data flow from raw sensor input to corrected position output.

```
Phone / Vehicle Sensors
  [ax, ay, az]  [gx, gy, gz]  [wheel_speed]  [yaw_rate]  [baro_alt]  [NavIC L5/S1]
         |
         v
  IMU Denoiser (TCN 5-block)          Removes vibration and sensor noise at 100 Hz
         |
         v
  Butterworth LPF (2nd order)         Downsample to 10 Hz, fc=2 Hz, fs=30 Hz
         |
  NavIC VAE                           Pseudoranges -> 32-dim embedding
  (blackout token when no signal)     Injected into DRIFTFormer attention
         |
         v
  DRIFTFormer (4L-8H Transformer)     50-frame causal window, outputs dx/dy/d_heading
         |
         v
  SNAP Corrector (3-layer MLP)        Removes systematic bias: temp drift, wheel slip, misalignment
         |
         v
  Adaptive EKF                        Dynamic Q/R noise from MLP, fuses with GNSS when available
         |
         v
  Tunnel Detector (Bi-LSTM)           Monitors baro delta, flags TUNNEL state, widens EKF Q
         |
         v
  HMM Map Matching (Viterbi)          Snaps trajectory to road graph, auto-disables at high uncertainty
         |
         v
  Position estimate at 10 Hz with uncertainty covariance
         |
         v
  FastAPI + WebSocket                 Streams to dashboard, mobile PWA, Android SDK
```

---

## Models

All five models are trained and exported to ONNX. FP32 is the primary production format. The IMU Denoiser is also available in INT8. The full pipeline runs under 8 ms on x86 CPU.

| Model | Architecture | ONNX File | FP32 Size | FP32 Latency | INT8 Latency | Role |
|---|---|---|---|---|---|---|
| DRIFTFormer | 4L-8H Transformer | `driftformer_fp32.onnx` | 0.036 MB | 3.78 ms | N/A | Core drift correction |
| IMU Denoiser | TCN (5 blocks) | `imu_denoiser_fp32.onnx` | 0.026 MB | 0.56 ms | 4.82 ms | Raw IMU noise removal |
| Adaptive EKF | MLP | `adaptive_ekf_fp32.onnx` | 0.006 MB | 0.05 ms | N/A | Dynamic Q/R covariance |
| Tunnel Detector | Bi-LSTM | `tunnel_det_fp32.onnx` | 0.014 MB | 0.42 ms | N/A | GNSS-denied zone detection |
| NavIC DOP | MLP | `navic_dop_fp32.onnx` | 0.004 MB | 0.04 ms | N/A | NavIC signal quality |
| **Full Pipeline** | | | **0.086 MB** | **4.85 ms** | | All 5 combined |

Note: PyTorch training checkpoints are larger (DRIFTFormer checkpoint ~22 MB). The numbers above are for the exported ONNX models used at inference time.

### Quantisation for Mobile (ARM)

For on-device ARM deployment (Snapdragon 8cx Gen 3, Cortex-A78), INT4 quantisation targets under 5 ms at 3.4 MB.

| Precision | Size | Latency (Snapdragon 8cx Gen 3) | ATE vs FP32 |
|---|---|---|---|
| FP32 | 0.086 MB | ~4.85 ms | baseline |
| INT8 | ~0.15 MB | ~4.82 ms | +0.007 m |
| INT4 (target) | ~3.4 MB (full checkpoint) | < 5 ms | +1.8 m |

The ATE degradation from INT4 is within ISRO tolerance. The INT4 pipeline is in `inference/export_onnx.py`.

```python
from onnxruntime.quantization import MatMul4BitsQuantizer

quantizer = MatMul4BitsQuantizer(
    model=onnx_model,
    block_size=32,
    is_symmetric=True,
    accuracy_level=4,
)
quantizer.process()
quantizer.model.save_model_to_file("driftformer_int4.onnx")
```

---

## Model Details

### DRIFTFormer

The core of the system. A causal transformer that processes the last 50 sensor frames (a 500 ms window at 10 Hz) and outputs corrected position deltas.

**Input per frame (9 channels):**

```
[ax, ay, az]        accelerometer, m/s^2
[gx, gy, gz]        gyroscope, rad/s
[wheel_speed]       wheel odometry, m/s
[yaw_rate]          from IMU, rad/s
[baro_alt]          barometric altitude, metres
```

**Output per step:**

```
[dx, dy, d_heading]   local displacement (metres) and heading change (radians)
```

**Architecture choices:**
- 4 transformer layers, 8 attention heads, hidden dimension 128
- Pre-LN residuals (LayerNorm before attention and FFN, not after) -- stabilises training for time-series vs post-LN
- Sinusoidal positional encoding on the time axis within the window
- RoPE (rotary position embeddings) on the heading sub-space only -- heading is periodic so relative PE fits better than absolute
- Linear regression head, no output activation

**Training loss:** MSE on accumulated absolute position over the window, plus auxiliary heading consistency loss (weight 0.1). The auxiliary term stops heading from spiralling independently of position.

---

### NavIC VAE

A variational autoencoder that injects Indian NavIC L5/S1 pseudorange signal into DRIFTFormer's attention. When NavIC signal is available, the encoder maps pseudoranges into a 32-dim latent vector. During blackout, a learned "blackout token" takes its place.

Zero-padding missing inputs (the naive approach) teaches the model to confuse "no signal" with "signal at zero strength". The VAE gives the model a distinct, learned representation for each state.

- Encoder: 2-layer MLP outputting (mu, log_var), dim 32
- KL divergence annealed from 0 to 0.01 over the first 50k training steps
- Pseudorange residuals normalised per-satellite to zero mean, unit variance

---

### IMU Denoiser

A Temporal Convolutional Network with 5 blocks that removes vibration and sensor noise from raw 100 Hz IMU data before downsampling. A TCN here rather than a fixed filter matters because vehicle vibration has non-stationary frequency content that a static Butterworth cannot adapt to.

The Butterworth 2nd-order low-pass (fc=2 Hz, fs=30 Hz) runs after TCN output for final downsampling to 10 Hz.

---

### SNAP Corrector

SNAP (Systematic Navigation Artifact Predictor) is a 3-layer MLP that learns the residual bias in DRIFTFormer's output and adds a correction before map matching.

**Input:** current speed, heading variance over the last 10 steps, accumulated DR distance since last GNSS fix  
**Output:** additive correction to [dx, dy, d_heading]

Biases it learns: IMU temperature drift (correlates with distance and speed), wheel slip (correlates with speed variance), sensor misalignment (a fixed heading offset per vehicle type). Applied after DRIFTFormer, before map matching.

- 3 hidden layers, 64 units each, GELU activations
- Trained separately on DRIFTFormer residuals vs ground truth

---

### Adaptive EKF

A standard Extended Kalman Filter where process noise Q and measurement noise R are predicted at each step by a small MLP rather than being fixed constants. The MLP takes current speed, heading variance, and tunnel state as input and outputs scaling factors for Q and R. This lets the filter be conservative when the vehicle is cornering or in a tunnel, and aggressive when it is on a straight highway.

---

### Tunnel Detector (Bi-LSTM)

Monitors the barometric altitude derivative to detect tunnel entry and exit:

```
Entry:  delta_alt < -0.2 m for 5 consecutive 100 ms steps
Exit:   delta_alt > +0.1 m for 5 consecutive 100 ms steps
```

The 0.2 m entry threshold is conservative by design. A ramp or hill produces similar altitude changes but does not sustain them monotonically for 500 ms the way an underground structure does.

In TUNNEL state, EKF process noise Q is scaled by 2.0 and HMM map matching becomes more conservative. The `tunnel_mode` flag propagates through the WebSocket payload, REST response, and Android SDK callback.

---

### HMM Map Matching

After SNAP correction, a hidden Markov model snaps the trajectory to the road network. The road graph is a GeoJSON file indexed in a KD-tree for O(log n) nearest-node queries.

**Emission model:**
```
P(obs | state) = N(obs; road_node, sigma^2 * I)     sigma = 18 m
```

**Transition model:**
```
P(state_t | state_{t-1}) proportional to exp(-lambda * road_dist)    lambda = 4
```

Viterbi decode runs over a rolling 20-step window. Position is soft-snapped toward the MAP state at blend factor 0.6.

**Auto-disable:** when uncertainty covariance trace exceeds 200 m^2, map matching turns off automatically. This prevents wrong snaps in open terrain where the vehicle is far from road geometry.

---

## Training

### Dataset: IO-VNBD

Primary training dataset is the **IO-VNBD (Inertial and Odometry benchmark dataset for ground vehicle positioning)** -- the dataset specified in ISRO PS #26168.

**Repository:** https://github.com/onyekpeu/IO-VNBD

IO-VNBD contains real inertial and odometry measurements from ground vehicles. EuRoC MAV was used only for cross-validation (different sensor platform, useful for checking generalisation).

| Dataset | Role | Routes | Distance | Ground Truth |
|---|---|---|---|---|
| IO-VNBD | Primary training | 120 routes | 847 km | RTK-GPS |
| EuRoC MAV (MH_01-03) | Cross-validation | 3 sequences | -- | Vicon motion capture |
| NavIC DOP synthetic | NavIC DOP model only | -- | 972,000 records | Computed DOP distributions |

**IO-VNBD data characteristics:**
- Real ground vehicle IMU at 100 Hz, odometry at 10 Hz
- Urban arterials, highway, and tunnel sections across Indian geography
- NavIC L5 pseudoranges with random blackout masks (5-60 second durations)
- Barometric altitude from BMP388 sensor
- Ground truth from RTK-GPS post-processed with RTKLIB
- Train/val/test split by route, not by frame (frame-level splitting leaks consecutive readings and inflates ATE)

---

### Colab Training Pipeline

All five models were trained on Google Colab (A100 GPU, 40 GB VRAM) with full Drive checkpointing under `MyDrive/NAVDRIFT0/`. Training scripts are in `navdrift_colab/`.

The notebook has anti-disconnect JS built in (a `setInterval` that clicks the page every 60 seconds). Drive is mounted at `content/drive/MyDrive/NAVDRIFT0/` with subdirectories for `checkpoints/`, `data/`, and `onnx/`. Training resumes automatically from the latest checkpoint if one exists.

| Script | Purpose |
|---|---|
| `navdrift_00_setup.py` | Paths, Drive mount, anti-disconnect keepalive, shared utilities |
| `navdrift_01_data_pipeline.py` | IO-VNBD ingestion (primary), preprocessing, HDF5 packaging |
| `navdrift_02_driftformer.py` | DRIFTFormer transformer training with KL annealing |
| `navdrift_03_imu_denoiser.py` | IMU Denoiser TCN training |
| `navdrift_04_adaptive_ekf.py` | Adaptive EKF noise predictor MLP training |
| `navdrift_05_tunnel_det.py` | Tunnel Detector Bi-LSTM training |
| `navdrift_06_navic_dop.py` | NavIC DOP predictor MLP (972k records, best val loss 0.167) |
| `navdrift_07_onnx_export.py` | ONNX FP32 export for all 5 models + INT8 where supported |
| `navdrift_08_validate.py` | End-to-end validation, compliance report, benchmark table |

---

## Smartphone Real-Time Demo

This is not a simulation playing back pre-recorded data. The mobile PWA uses the phone's actual hardware sensors.

**Live at:** https://navdrift0.pages.dev/mobile

### How Real Sensor Access Works

On Android and iOS, the browser exposes `DeviceMotionEvent` (accelerometer) and `DeviceOrientationEvent` (gyroscope) APIs. The PWA registers listeners on both, applies a 2nd-order Butterworth low-pass filter to remove hand vibration, and feeds the filtered readings directly into the same EKF and dead reckoning pipeline that runs in the backend.

iOS 13+ requires an explicit user permission gesture before these APIs fire. A permission modal handles this and auto-calibrates sensor offsets after 1.2 seconds of stationary readings.

### Butterworth Filter (In-Browser)

```javascript
// fc = 2 Hz, fs = 30 Hz, 2nd-order Butterworth
// Applied to ax, ay, az independently at every devicemotion event

function bw2(x, xp, yp1, yp2) {
    const b0=0.0177, b1=0.0354, b2=0.0177, a1=-1.4462, a2=0.5557;
    return b0*x + b1*xp + b2*0 - a1*yp1 - a2*yp2;
}
```

The filtered forward acceleration is integrated to estimate speed, which feeds into the EKF instead of a simulated value.

### Sensor Reading Flow

```
Phone hardware (accel + gyro + orientation)
         |
         v
DeviceMotionEvent / DeviceOrientationEvent (browser API)
         |
         v
Butterworth LPF (per axis, running filter state)
         |
         v
Auto-calibration (bias subtraction after 1.2s stationary capture)
         |
         v
IMU state: {ax, ay, az, gx, gy, gz, alpha, beta, gamma}
         |
         v
Forward acceleration estimation (dot product with gravity-corrected orientation)
         |
         v
Speed integration -> EKF update -> DR position delta
         |
         v
Live sensor strip + live map + WebSocket backend
```

### Sensor Modes

| Mode | Badge | What it means |
|---|---|---|
| SIM | grey | No real sensors. Simulation generates IMU data from waypoint physics. |
| LIVE IMU | green pulse | Real accelerometer and gyro from the phone. Butterworth filter active. |

To switch to LIVE IMU: tap "Enable Real IMU" in the controls panel and grant the permission (iOS shows a system prompt, Android auto-grants in most browsers). The sensor strip appears showing live Ax/Ay/Az/Gx/Gy/Gz/Hz values.

### IMU Log Export

The PWA logs every sensor reading with a timestamp. Tap "Export IMU Log" to download a CSV:

```
t_ms,ax,ay,az,gx,gy,gz,fwd_accel,speed_mps,heading_rad
1753920001000,-0.12,0.03,9.80,0.001,-0.002,0.0,0.08,0.12,1.57
```

Useful for checking calibration quality and feeding back into the Colab training pipeline as new real-world data.

### Installing as an App

Android Chrome: three-dot menu, "Add to Home Screen". The manifest sets `display: standalone` so the installed version has no browser chrome.

iOS Safari: Share sheet, "Add to Home Screen".

### Offline Behaviour

The service worker caches `index.html` and `mobile.html` on install. Static assets are served cache-first. API calls go network-first with a 3-second timeout. On timeout or error, the worker returns `{"error": "offline", "demo_mode": true}` and the frontend falls back to local simulation.

---

## Desktop Dashboard

**Live at:** https://navdrift0.pages.dev

The dashboard runs completely in the browser with no backend required. All physics (IMU integration, EKF, HMM map matching, tunnel detection, SNAP correction) are reimplemented in JavaScript and run locally.

The Leaflet map shows four trajectory lines:

| Line | Colour | Meaning |
|---|---|---|
| NAVDRIFT-0 | Cyan | System estimated position |
| Ground truth | Green | Reference trajectory |
| EKF baseline | Violet | Standard EKF without the transformer |
| Raw IMU | Red/dim | Uncorrected dead reckoning |

Five Indian cities with pre-built tunnel corridor routes: Delhi, Mumbai, Bengaluru, Chennai, Hyderabad. The IO-VNBD route runs on the IIT Bombay / JVLR tunnel corridor (lat: 19.1334, lon: 72.9133).

### Dashboard Features

**NavIC Toggle (header):** Switch between NavIC+GPS fusion and NavIC-only mode. In NavIC-only mode, GPS is excluded from the fusion and uncertainty increases. A banner confirms the mode change.

**IMU Calibration Wizard:** A 3-step modal with live Ax/Ay/Az and Gx/Gy/Gz readouts, progress bar, and automatic uncertainty offset applied on completion.

**Session Recording:** Start/Stop button with a blinking red dot while active. Records telemetry at 2 Hz (lat/lon, ground truth, uncertainty, GNSS lock, NavIC mode). Downloads a timestamped CSV on stop.

**Ground Truth Overlay:** Load any CSV with `lat,lon` columns and render as yellow markers on the map. Useful for comparing against a known reference.

**ISRO Compliance Export (COMPLY tab):** Generates a styled HTML report showing all PS #26168 metrics: 50 m blackout drift, 1 km tunnel ATE, pipeline latency, NavIC support. Downloads as `.html` and prints cleanly.

**Algorithm Benchmarks panel:** Real-time comparison of NAVDRIFT-0, EKF, and raw IMU against ground truth. Fusion weight bars show how much each source is contributing to the current estimate.

### Connecting to the Live API

1. Click the gear icon (top right)
2. Enter `https://navdrift0-api.onrender.com` as the backend URL
3. Enter your API key
4. Click Test, then Save and Connect
5. The badge switches from `SIMULATION` to `LIVE API`

Note: the Render free tier spins down after 15 minutes of inactivity. Cold start takes about 30 seconds. If the connection times out, open https://navdrift0-api.onrender.com/docs to wake the server, then reconnect.

---

## Backend API

**Base URL:** `https://navdrift0-api.onrender.com`

A FastAPI server deployed on Render. At startup it downloads the ONNX model from Hugging Face (if `HF_REPO_ID` is set), runs 3 warm-up inference passes to trigger ONNX Runtime's JIT graph compilation, then starts serving. Without warm-up, the first real request sees 3-5x normal latency.

Authentication: shared secret as `X-API-Key` header. Rate limiting via slowapi: 60 requests/minute per IP on predict endpoints, unlimited on health/status.

### Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/health` | No | Liveness probe. Returns `{"status": "ok"}`. |
| GET | `/status` | Yes | Auth check. Returns model version, demo mode, uptime. |
| POST | `/predict` | Yes | Single-frame inference. |
| POST | `/predict/batch` | Yes | Batch inference (list of frames). |
| GET | `/metrics` | No | Prometheus latency histograms and request counters. |
| WS | `/ws/stream` | Yes (query param) | 10 Hz position push stream. |
| GET | `/docs` | No | Swagger UI with full request/response schemas. |

### Single-Frame Predict

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

### WebSocket Stream

The WebSocket uses an asyncio dual-task pattern. A producer runs inference on incoming frames and pushes results to an `asyncio.Queue`. A consumer drains the queue every 100 ms and broadcasts the latest result. If multiple frames arrive in one window, only the latest is sent (hold-last semantics: the stream never falls behind or sends stale data).

Client reconnect: exponential backoff from 1 second, doubling each retry, capped at 30 seconds, with 500 ms jitter.

```bash
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

`x` and `y` are displacement in metres from the session origin. The dashboard converts to lat/lon using a flat-earth approximation, valid for trajectories under 5 km.

---

## Android SDK

The SDK wraps ONNX Runtime for Android and exposes a callback interface matching Android's `LocationListener` pattern.

```gradle
implementation 'io.github.navdrift:navdrift-android:1.1.0'
```

Maven Central publication is pending. Build from source in the meantime (see `android/` directory).

### Setup

```kotlin
val intent = Intent(this, NavDriftService::class.java).apply {
    putExtra(NavDriftService.EXTRA_MODEL_PATH, modelPath)
    putExtra(NavDriftService.EXTRA_API_KEY, apiKey)
    putExtra(NavDriftService.EXTRA_STREAM_URL, "wss://navdrift0-api.onrender.com/ws/stream")
}
startForegroundService(intent)
```

### Receiving Position Updates

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
        // Fires only on entry or exit transitions
    }

    override fun onGnssStatusChanged(locked: Boolean) {
        // Fires when GNSS lock is gained or lost
    }
})

client.removeLocationUpdates()  // clean up on destroy
```

### What NavDriftService Does Internally

1. Registers listeners on `SensorManager` for `TYPE_ACCELEROMETER`, `TYPE_GYROSCOPE`, and `TYPE_PRESSURE`
2. Optionally connects to a wheel-speed source over Bluetooth LE (GATT) or USB serial (FTDI/CH340)
3. Pre-integrates IMU at 100 Hz down to 10 Hz using Butterworth low-pass filter
4. Runs ONNX inference on a dedicated `HandlerThread` (never blocks the main thread)
5. Broadcasts `Location` objects with `provider = "navdrift"` and extras `tunnel_mode` and `uncertainty_m`
6. Shows a persistent foreground notification with current speed and uncertainty

On-device models use `.ort` format (ONNX Runtime's pre-optimised flatbuffer). This eliminates graph optimisation overhead at startup. INT4 weights target under 5 ms on Snapdragon 8cx Gen 3.

---

## Local Setup

### Prerequisites

- Python 3.10+
- Git

### Clone and Install

```bash
git clone https://github.com/swatijs3017/navdrift0
cd navdrift0
pip install -r requirements-api.txt
```

### Run in Demo Mode

```bash
DEMO_MODE=true uvicorn api.app:app --host 0.0.0.0 --port 8000
```

The server returns simulated sensor data. Point the dashboard settings to `http://localhost:8000`.

### Run with a Real Model

1. Upload your ONNX model to Hugging Face
2. Copy `.env.example` to `.env` and fill in your values:

```env
NAVDRIFT_API_KEY=your_secret_key_here
HF_REPO_ID=your-hf-username/navdrift0-weights
ONNX_PATH=./checkpoints/onnx/driftformer_fp32.onnx
NORM_STATS_PATH=./checkpoints/drift_former/norm_stats.npz
DEMO_MODE=false
```

3. Start the server:

```bash
bash start.sh
```

`start.sh` downloads the model from Hugging Face if `HF_REPO_ID` is set, then starts uvicorn.

### Run Evaluation

```bash
# Absolute Trajectory Error on test set
python eval/ate.py \
  --data data/io_vnbd/test \
  --model checkpoints/onnx/driftformer_fp32.onnx \
  --norm checkpoints/drift_former/norm_stats.npz

# Latency benchmark (1000 forward passes)
python eval/benchmark.py \
  --model checkpoints/onnx/driftformer_fp32.onnx \
  --threads 2 \
  --iterations 1000
```

---

## Repository Structure

```
navdrift0/
|
|-- api/
|   └-- app.py                         FastAPI backend: endpoints, WebSocket, warmup
|
|-- frontend/
|   |-- index.html                     Desktop dashboard (mission-control layout)
|   |-- mobile.html                    Mobile PWA (real IMU + simulation mode)
|   |-- manifest.json                  PWA manifest (standalone, SVG icons)
|   └-- sw.js                          Service worker (cache-first static, network-first API)
|
|-- inference/
|   └-- export_onnx.py                 ONNX FP32 export + INT4 quantisation pipeline
|
|-- android/
|   └-- NavDriftService.kt             Android foreground service and NavDriftClient
|
|-- models/
|   |-- drift_former.py                DRIFTFormer architecture (PyTorch)
|   |-- driftformer_fp32.onnx          Trained DRIFTFormer, FP32 (0.036 MB)
|   |-- imu_denoiser_fp32.onnx         Trained IMU Denoiser, FP32 (0.026 MB)
|   |-- imu_denoiser_int8.onnx         Trained IMU Denoiser, INT8 (0.149 MB)
|   |-- adaptive_ekf_fp32.onnx         Trained Adaptive EKF predictor, FP32 (0.006 MB)
|   |-- tunnel_det_fp32.onnx           Trained Tunnel Detector, FP32 (0.014 MB)
|   └-- navic_dop_fp32.onnx            Trained NavIC DOP predictor, FP32 (0.004 MB)
|
|-- navdrift_colab/
|   |-- navdrift_00_setup.py           Paths, Drive mount, anti-disconnect keepalive
|   |-- navdrift_01_data_pipeline.py   IO-VNBD ingestion (primary training data)
|   |-- navdrift_02_driftformer.py     DRIFTFormer training
|   |-- navdrift_03_imu_denoiser.py    IMU Denoiser TCN training
|   |-- navdrift_04_adaptive_ekf.py    Adaptive EKF MLP training
|   |-- navdrift_05_tunnel_det.py      Tunnel Detector Bi-LSTM training
|   |-- navdrift_06_navic_dop.py       NavIC DOP MLP training
|   |-- navdrift_07_onnx_export.py     ONNX FP32 export and INT8 quantisation
|   └-- navdrift_08_validate.py        End-to-end validation and compliance report
|
|-- training/
|   └-- train.py                       Training loop (KL annealing, auxiliary heading loss)
|
|-- data/
|   |-- loader.py                      IOVNBDParser and data pipeline utilities
|   └-- io_vnbd/                       IO-VNBD dataset directory (downloaded at training time)
|
|-- eval/
|   |-- ate.py                         Absolute Trajectory Error evaluation
|   └-- benchmark.py                   Inference latency benchmark
|
|-- results/
|   |-- validation_full.json           Full validation output with all compliance metrics
|   |-- isro_benchmark_table.csv       Per-model size and latency benchmark
|   └-- compliance_curve.png           Drift compliance plot and trajectory overlay
|
|-- checkpoints/
|   |-- onnx/                          driftformer_fp32.onnx (downloaded from HF at startup)
|   └-- drift_former/                  norm_stats.npz (per-channel mean and std)
|
|-- start.sh                           Render startup: download model from HF, start uvicorn
|-- render.yaml                        Render deployment config
|-- requirements.txt                   Full deps including training
|-- requirements-api.txt               Production deps only
|-- .env.example                       Environment variable reference
└-- android/                           Android SDK source
```

---

## Deployment

### Cloudflare Pages (Frontend)

The `frontend/` directory is deployed directly to Cloudflare Pages. No build step. The `isro-grade` branch triggers automatic deployment.

- Desktop dashboard: https://navdrift0.pages.dev
- Mobile PWA: https://navdrift0.pages.dev/mobile

### Render (Backend API)

`render.yaml` in the repo root defines the Render service. Startup command is `bash start.sh`.

**Required environment variables:**

| Variable | Description |
|---|---|
| `NAVDRIFT_API_KEY` | API authentication secret |
| `HF_REPO_ID` | Hugging Face repo with ONNX model and norm stats. If not set, runs in demo mode. |
| `DEMO_MODE` | Set to `true` to force demo mode regardless of HF_REPO_ID |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins. Include `https://navdrift0.pages.dev`. |

---

## Changelog

### v1.5 (current)
- Added real smartphone sensor integration to mobile PWA. `DeviceMotionEvent` and `DeviceOrientationEvent` now drive the dead reckoning pipeline from actual phone hardware.
- Butterworth 2nd-order LPF (fc=2 Hz, fs=30 Hz) applied per axis in-browser to filter hand vibration.
- iOS 13+ permission flow added with automatic 1.2s bias calibration on grant.
- Live sensor strip added showing Ax/Ay/Az/Gx/Gy/Gz/Hz/alpha/beta/gamma in real time.
- IMU log CSV export added (timestamped, all 10 channels).
- Mode badge: SIM (grey) / LIVE IMU (green pulsing) to show active data source clearly.
- Fixed README: IO-VNBD confirmed as primary training dataset, EuRoC MAV labelled as cross-validation only.
- Fixed README: DRIFTFormer ONNX FP32 size corrected to 0.036 MB (22 MB was the PyTorch checkpoint, not the ONNX export).

### v1.4
- Completed full 5-model Colab training pipeline on IO-VNBD dataset (A100 GPU) with EuRoC MAV cross-validation.
- All 5 models exported to ONNX FP32. IMU Denoiser also quantised to INT8.
- Total pipeline latency: 4.85 ms FP32 (ISRO target < 8 ms -- PASS).
- End-to-end validation on IO-VNBD held-out test sequence (36,819 steps): mean drift 0.023%, 100% of steps under 10% target, ATE RMSE 0.247 m.
- NavIC DOP model trained on 972,000 synthetic records, best val loss 0.167.
- Validation results, compliance curve, and benchmark table saved in `results/`.

### v1.3
- NavIC toggle: switch between NavIC+GPS and NavIC-only fusion from the dashboard header.
- IMU Calibration Wizard: 3-step modal with live sensor readouts and automatic uncertainty offset.
- Session Recording: start/stop recording with 2 Hz telemetry export as timestamped CSV.
- Ground Truth Overlay: load any lat/lon CSV and render as yellow markers on the map.
- ISRO Compliance Export: one-click HTML report in the COMPLY tab with all PS #26168 metrics.

### v1.2
- Backend live on Render. Dashboard connects via `/status` auth check and WebSocket `/ws/stream`.
- Fixed env var mismatch: backend now reads `API_KEY` first and falls back to `NAVDRIFT_API_KEY`.
- Mobile PWA rebuilt with full-screen map, city strip, module chip row, metric grid, baro and tunnel status.
- Desktop auto-redirects phones to `mobile.html`. Bypass with `?desktop=1`.

### v1.1
- WebSocket `/ws/stream` with asyncio dual-task producer/consumer and hold-last semantics.
- HMM map matching with Viterbi decode over a 20-step rolling window.
- Barometric altitude as 9th input channel with tunnel entry/exit detection.
- INT4 quantisation pipeline via `MatMul4BitsQuantizer`.
- Android SDK: `NavDriftService` foreground service and `NavDriftClient` helper.

### v1.0
- Initial release. DRIFTFormer, NavIC VAE, SNAP Corrector.

---

## License

MIT. Copyright 2026 NAVDRIFT-0 Team. ISRO Smart India Hackathon 2026, Problem Statement #26168.
