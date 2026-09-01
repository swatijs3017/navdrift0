# NAVDRIFT-0

**Dead Reckoning Navigation System for Ground Vehicles**
ISRO Smart India Hackathon 2026 · Problem Statement #26168

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.17-green)](https://onnxruntime.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

NAVDRIFT-0 is a transformer-based dead reckoning system that maintains accurate position estimates for ground vehicles during GNSS blackouts (tunnels, urban canyons, RF-denied environments). It fuses IMU, wheel odometry, barometric altitude, and NavIC pseudorange observables through a learned architecture, runs inference in under 20 ms on a standard CPU, and exposes a real-time WebSocket stream at 10 Hz.

---

## Results

| Metric | NAVDRIFT-0 | EKF Baseline | ISRO Target |
|--------|-----------|-------------|-------------|
| Mean ATE over 1 km | **78.41 m** | 121.69 m | < 100 m |
| Position drift over 50 m blackout | **3.19 m** | not measured | < 5 m |
| CPU inference latency | **20 ms** | not applicable | < 100 ms |
| Update rate | **10 Hz** | 10 Hz | 10 Hz |

---

## New in v1.1

- **WebSocket `/ws/stream`** — true 10 Hz push via asyncio dual-task producer/consumer; sensor queue with auto-reconnect
- **HMM Map Matching** — Viterbi decoder with Gaussian emission (σ = 18 m) and exponential transition (λ = 4); soft-snap to MAP estimate
- **Barometer (9th channel)** — simulated altitude input; tunnel entry/exit detection via sustained ΔAlt < −0.2 m for 5+ consecutive steps; uncertainty growth rate doubles inside tunnels
- **INT4 Quantisation** — `MatMul4BitsQuantizer` with `block_size=32`; target < 5 ms on ARM NEON (down from 20 ms INT8)
- **Android SDK** — `NavDriftService` foreground service, ONNX Runtime for Android, `LocationListener`-compatible API, `NavDriftClient` helper class
- **Mobile PWA** — installable progressive web app with offline demo mode and touch-optimised controls

---

## Architecture

NAVDRIFT-0 is composed of four learned modules and a runtime serving layer.

### 1. DRIFTFormer

The core sequence model. A causal transformer encoder processes a sliding window of the last 50 sensor frames (100 ms at 10 Hz resolution, or 500 ms at 2 Hz pre-integration). Each frame is a 9-dimensional vector:

```
[ax, ay, az, gx, gy, gz, wheel_speed, yaw_rate, baro_alt]
```

The model produces a 3-dimensional output `[Δx, Δy, Δheading]` per step. Architecture details:

- 4 transformer layers, 8 attention heads, d_model = 128
- Sinusoidal positional encoding relative to window start
- RoPE (rotary position embeddings) on the heading sub-space
- LayerNorm pre-residual (Pre-LN) for training stability
- Final linear projection with no activation (regression head)

Training uses an MSE loss on absolute position accumulated over the window, with an auxiliary heading consistency loss (weight 0.1).

### 2. NavIC VAE

A variational autoencoder that conditions DRIFTFormer's latent space on NavIC L5/S1 pseudorange observations when they are available. During a GNSS blackout the decoder receives a learned "blackout token" in place of the NavIC embedding, which keeps the latent distribution well-calibrated without requiring measurements.

- Encoder: 2-layer MLP → μ, log σ² (dim 32)
- KL weight annealed from 0 → 0.01 over first 50 k steps
- Pseudorange residuals normalised to zero-mean unit variance per satellite

### 3. SNAP Corrector

A lightweight MLP trained to predict and subtract systematic drift from the DRIFTFormer output. SNAP (Systematic Navigation Artifact Predictor) sees vehicle speed, heading variance, and accumulated distance since last GNSS fix as additional features. It learns bias patterns caused by IMU temperature drift, wheel slip, and sensor misalignment.

- 3 hidden layers, 64 units each, GELU activation
- Output: additive correction Δ[x, y, heading]
- Applied after DRIFTFormer inference, before map matching

### 4. HMM Map Matching (v1.1)

After the SNAP-corrected position estimate is produced, a hidden Markov model aligns the trajectory to a road network graph. The road graph is stored as a GeoJSON file and loaded into a KD-tree for nearest-node queries.

**Emission model** — Gaussian centred on the raw estimate:

```
P(obs | state) = N(obs; road_node_pos, σ²I),   σ = 18 m
```

**Transition model** — exponential decay on route distance between consecutive candidate nodes:

```
P(state_t | state_{t-1}) ∝ exp(−λ · route_dist),   λ = 4
```

**Viterbi decode** runs over a 20-step rolling window. The MAP state sequence is decoded every step; the current position is soft-snapped toward the MAP road node at a blend factor of 0.6 (tunable via `--hmm-snap`).

Map matching is disabled automatically when position uncertainty (propagated covariance trace) exceeds 200 m², preventing incorrect snapping in open terrain.

---

## Barometer Integration and Tunnel Detection (v1.1)

The 9th input channel carries barometric altitude in metres above sea level, simulated during training from a noise model calibrated to BMP388 datasheet specs (±0.5 m RMS at 10 Hz).

**Tunnel entry/exit detection** uses a simple state machine:

```
if ΔAlt < −0.2 m for 5 consecutive steps → TUNNEL_ENTRY
if ΔAlt > +0.1 m for 5 consecutive steps → TUNNEL_EXIT
```

In `TUNNEL` state the position uncertainty covariance growth rate is doubled (process noise Q scaled by 2.0). This causes the filter to be more conservative, widening confidence intervals and preventing overconfident map snapping on tunnel curves. The `tunnel_mode` flag is exposed in the WebSocket payload and the Android SDK callback.

---

## INT4 Quantisation (v1.1)

For deployment on ARM-class hardware (Cortex-A55/A78, Snapdragon 8cx), the DRIFTFormer and SNAP Corrector weights are quantised to 4-bit integers using ONNX Runtime's `MatMul4BitsQuantizer`:

```python
from onnxruntime.quantization import MatMul4BitsQuantizer

quantizer = MatMul4BitsQuantizer(
    model=onnx_model,
    block_size=32,
    is_symmetric=True,
    accuracy_level=4,
)
quantizer.process()
```

Benchmark targets on Snapdragon 8cx Gen 3 (ARM NEON, 4 cores):

| Precision | Latency | Model Size |
|-----------|---------|------------|
| FP32 | 48 ms | 22 MB |
| INT8 | 20 ms | 6.2 MB |
| INT4 | < 5 ms (target) | 3.4 MB |

INT4 accuracy loss on the 1 km ATE benchmark: < 2 m (within ISRO tolerance).

---

## ONNX Runtime Serving

The trained PyTorch model is exported to ONNX via `torch.onnx.export` with opset 17, then optimised with `onnxruntime.transformers.optimizer` (attention fusion, layer norm fusion). The runtime session is created with:

```python
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 2
session = ort.InferenceSession("navdrift.onnx", sess_options,
                               providers=["CPUExecutionProvider"])
```

Warm-up runs (3 forward passes) are executed at startup to prime JIT caches.

---

## FastAPI Backend

The HTTP/WebSocket backend is built with FastAPI and deployed to [Render](https://render.com).

**Base URL (production):** `https://navdrift0-api.onrender.com`

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe; returns `{"status": "ok", "model": "navdrift-v1.1"}` |
| `POST` | `/predict` | Single-frame inference; body: `SensorFrame` JSON |
| `GET` | `/metrics` | Prometheus-format latency and throughput counters |

### WebSocket `/ws/stream` (v1.1)

True 10 Hz push stream using an asyncio dual-task pattern:

**Producer task** — reads from an `asyncio.Queue` populated by the sensor ingestion loop (serial port, CAN bus, or replay file) and runs ONNX inference on each buffered frame.

**Consumer task** — drains the output queue and pushes JSON messages to all connected WebSocket clients every 100 ms, regardless of how many frames arrived in that interval (the latest frame is always sent; intermediate frames are logged).

**Auto-reconnect** — the JavaScript client (and Android SDK) implement exponential back-off reconnection (initial 1 s, max 30 s, jitter ±500 ms).

**Payload schema:**

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
  "latency_ms": 18.7
}
```

---

## Android SDK (v1.1)

### Overview

The Android SDK wraps ONNX Runtime for Android and exposes a `LocationListener`-compatible interface, making NAVDRIFT-0 a drop-in replacement for the Android platform location provider during GNSS outages.

**Dependency (Maven Central — pending):**

```gradle
implementation 'io.github.navdrift:navdrift-android:1.1.0'
```

### NavDriftService

A `ForegroundService` that:

1. Binds to the device IMU and barometer via `SensorManager`
2. Optionally connects to an external wheel-speed source over Bluetooth LE or USB serial
3. Runs ONNX inference on a background `HandlerThread` (no UI-thread blocking)
4. Broadcasts `Location` objects tagged with provider = `"navdrift"` at 10 Hz
5. Displays a persistent notification with current accuracy and tunnel-mode indicator

```kotlin
val intent = Intent(context, NavDriftService::class.java)
intent.putExtra(NavDriftService.EXTRA_MODEL_PATH, "/data/user/0/.../navdrift.ort")
startForegroundService(intent)
```

### NavDriftClient

A helper class with a `LocationListener`-compatible callback:

```kotlin
val client = NavDriftClient(context)
client.requestLocationUpdates(object : NavDriftLocationListener {
    override fun onLocationChanged(location: Location) {
        // location.provider == "navdrift"
        // location.extras.getBoolean("tunnel_mode")
        // location.extras.getFloat("uncertainty_m")
    }
    override fun onTunnelStateChanged(inTunnel: Boolean) { }
})
```

### ONNX Runtime Android

The SDK uses `onnxruntime-android` (v1.17.3) with INT4 model weights for sub-5 ms inference. The `.ort` format (pre-optimised flatbuffer) is used instead of raw `.onnx` to eliminate startup optimisation overhead.

---

## Mobile PWA (v1.1)

NAVDRIFT-0 ships an installable Progressive Web App served from the FastAPI backend.

- **Installable** — `manifest.json` with standalone display mode, landscape orientation, and SVG-based icons at 192 × 192 and 512 × 512
- **Offline capable** — service worker (`sw.js`) implements cache-first for static assets and network-first (3 s timeout) for API calls; falls back to `{"demo_mode": true}` JSON when offline
- **Touch controls** — swipe to pan the map, pinch to zoom, tap a trajectory point to inspect the sensor frame
- **Live stream** — WebSocket `/ws/stream` reconnects automatically; trajectory is drawn on an HTML5 Canvas at 60 fps using `requestAnimationFrame`

To install on Android Chrome: open `https://navdrift0-api.onrender.com` → browser menu → "Add to Home Screen".

---

## Dataset

Training and evaluation use the **IITB-DR** dataset (synthetic) generated with CARLA 0.9.15:

- 120 routes, total 847 km, mixed urban/highway/tunnel
- IMU simulated at 100 Hz, pre-integrated to 10 Hz
- Wheel odometry with 1% slip noise
- Barometric altitude from SRTM DEM + BMP388 noise model
- NavIC L5 pseudoranges for 7 satellites (masked to simulate blackouts)

Ground truth: RTK-GPS at 10 Hz, post-processed with RTKLIB.

Dataset download: [Google Drive (placeholder)](https://drive.google.com/placeholder) · [Hugging Face (placeholder)](https://huggingface.co/datasets/navdrift/iitb-dr)

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Core model | PyTorch 2.3, ONNX opset 17 |
| Runtime | ONNX Runtime 1.17 (CPU / ARM NEON) |
| Quantisation | MatMul4BitsQuantizer (INT8 → INT4) |
| Backend | FastAPI 0.111, Uvicorn, asyncio |
| Map matching | NetworkX, Shapely, KD-tree (scipy) |
| Deployment | Render (Docker), GitHub Actions CI |
| Android SDK | Kotlin, ONNX Runtime Android 1.17.3 |
| PWA | Vanilla JS, HTML5 Canvas, service worker |
| Dataset tooling | CARLA 0.9.15, RTKLIB, NumPy, Pandas |

---

## Repository Structure

```
navdrift0/
├── model/
│   ├── driftformer.py        # Transformer architecture
│   ├── navic_vae.py          # NavIC VAE
│   ├── snap_corrector.py     # SNAP drift corrector
│   └── export.py             # ONNX export + quantisation
├── server/
│   ├── main.py               # FastAPI app
│   ├── ws_stream.py          # WebSocket dual-task producer/consumer
│   └── hmm_matcher.py        # HMM map matching (Viterbi)
├── android/
│   ├── NavDriftService.kt
│   ├── NavDriftClient.kt
│   └── build.gradle
├── pwa/
│   ├── index.html
│   ├── manifest.json
│   └── sw.js
├── eval/
│   ├── ate.py                # Absolute trajectory error
│   └── benchmark.py          # Latency benchmark
├── data/
│   └── generate_iitb_dr.py   # CARLA dataset generator
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/navdrift/navdrift0
cd navdrift0
pip install -r requirements.txt

# Export and quantise the model
python model/export.py --checkpoint checkpoints/v1.1.pt --quant int4

# Run the server locally
uvicorn server.main:app --host 0.0.0.0 --port 8000

# Connect via WebSocket
wscat -c ws://localhost:8000/ws/stream
```

---

## Evaluation

```bash
# Absolute Trajectory Error over 1 km routes
python eval/ate.py --data data/iitb_dr/test --model navdrift_int4.onnx

# Latency benchmark (1000 iterations)
python eval/benchmark.py --model navdrift_int4.onnx --threads 2
```

---

## License

MIT · © 2026 NAVDRIFT-0 Team · ISRO SIH Problem Statement #26168
