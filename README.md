# NAVDRIFT-0

**Dead Reckoning Navigation System for Ground Vehicles**
ISRO Smart India Hackathon 2026 · Problem Statement #26168

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.17-green)](https://onnxruntime.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal)](https://fastapi.tiangolo.com)
[![Live API](https://img.shields.io/badge/API-Live%20on%20Render-brightgreen)](https://navdrift0-api.onrender.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

NAVDRIFT-0 is a transformer-based dead reckoning system that maintains accurate position estimates for ground vehicles during GNSS blackouts (tunnels, urban canyons, RF-denied environments). It fuses IMU, wheel odometry, barometric altitude, and NavIC pseudorange observables through a learned architecture, runs inference in under 20 ms on a standard CPU, and exposes a real-time WebSocket stream at 10 Hz.

**Live demo:** [navdrift0.pages.dev](https://navdrift0.pages.dev) · **API:** [navdrift0-api.onrender.com/docs](https://navdrift0-api.onrender.com/docs)

---

## Results

| Metric | NAVDRIFT-0 | EKF Baseline | ISRO Target |
|--------|-----------|-------------|-------------|
| Mean ATE over 1 km | **78.41 m** | 121.69 m | < 100 m |
| Position drift over 50 m blackout | **3.19 m** | not measured | < 5 m |
| CPU inference latency | **20 ms** | not applicable | < 100 ms |
| Update rate | **10 Hz** | 10 Hz | 10 Hz |

---

## What's New in v1.2

- **Live API connected** — Render backend is live; dashboard badge switches SIMULATION → LIVE API when connected with a valid API key
- **Mobile PWA v2** — fully rebuilt mobile layout: full-screen map, compact bottom sheet, module chip row, auto-redirect from desktop on phones (`window.innerWidth < 900`)
- **Env var fix** — backend now reads `API_KEY` or `NAVDRIFT_API_KEY` (whichever is set in Render dashboard)
- **Desktop link** — mobile view includes "↗ Open desktop view" for tablets

## What's New in v1.1

- **WebSocket `/ws/stream`** — true 10 Hz push via asyncio dual-task producer/consumer; sensor queue with auto-reconnect
- **HMM Map Matching** — Viterbi decoder with Gaussian emission (σ = 18 m) and exponential transition (λ = 4); soft-snap to MAP estimate
- **Barometer (9th channel)** — tunnel entry/exit detection via sustained ΔAlt < −0.2 m for 5+ steps; uncertainty growth rate doubles inside tunnels
- **INT4 Quantisation** — `MatMul4BitsQuantizer` with `block_size=32`; target < 5 ms on ARM NEON
- **Android SDK** — `NavDriftService` foreground service, ONNX Runtime for Android, `LocationListener`-compatible API
- **Mobile PWA** — installable progressive web app with offline demo mode

---

## Architecture

### 1. DRIFTFormer

The core sequence model. A causal transformer encoder processes a sliding window of the last 50 sensor frames. Each frame is a 9-dimensional vector:

```
[ax, ay, az, gx, gy, gz, wheel_speed, yaw_rate, baro_alt]
```

Output: `[Δx, Δy, Δheading]` per step. Details:

- 4 transformer layers, 8 attention heads, d_model = 128
- Sinusoidal + RoPE positional encoding
- Pre-LN residual connections
- MSE loss + auxiliary heading consistency loss (weight 0.1)

### 2. NavIC VAE

Conditions DRIFTFormer's latent space on NavIC L5/S1 pseudorange observations. During GNSS blackout, a learned "blackout token" maintains calibrated latent distribution.

- Encoder: 2-layer MLP → μ, log σ² (dim 32)
- KL weight annealed from 0 → 0.01 over 50 k steps

### 3. SNAP Corrector

Lightweight MLP that predicts and subtracts systematic drift (IMU temperature drift, wheel slip, misalignment).

- 3 hidden layers, 64 units, GELU activation
- Applied after DRIFTFormer, before map matching

### 4. HMM Map Matching (v1.1)

Viterbi decoder aligns trajectory to road network graph (GeoJSON + KD-tree).

```
Emission:    P(obs|state) = N(obs; road_node_pos, σ²I),  σ = 18 m
Transition:  P(s_t|s_{t-1}) ∝ exp(−λ · route_dist),     λ = 4
```

Runs on a 20-step rolling window. Disabled when uncertainty trace > 200 m².

---

## Barometer & Tunnel Detection (v1.1)

State machine on the 9th input channel (BMP388 noise model, ±0.5 m RMS at 10 Hz):

```
ΔAlt < −0.2 m for 5 consecutive steps → TUNNEL_ENTRY  (Q scaled ×2)
ΔAlt > +0.1 m for 5 consecutive steps → TUNNEL_EXIT
```

`tunnel_mode` is exposed in the WebSocket payload and the Android SDK callback.

---

## INT4 Quantisation (v1.1)

```python
from onnxruntime.quantization import MatMul4BitsQuantizer

quantizer = MatMul4BitsQuantizer(
    model=onnx_model, block_size=32, is_symmetric=True, accuracy_level=4,
)
quantizer.process()
```

| Precision | Latency (Snapdragon 8cx) | Model Size |
|-----------|--------------------------|------------|
| FP32 | 48 ms | 22 MB |
| INT8 | 20 ms | 6.2 MB |
| INT4 | < 5 ms (target) | 3.4 MB |

ATE degradation vs INT8: < 2 m (within ISRO tolerance).

---

## FastAPI Backend

**Base URL:** `https://navdrift0-api.onrender.com`

Authentication: `X-API-Key` header. Backend reads `API_KEY` or `NAVDRIFT_API_KEY` env var.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/status` | Auth check + model info |
| `POST` | `/predict` | Single-frame inference |
| `POST` | `/predict/batch` | Batch inference |
| `GET` | `/metrics` | Prometheus counters |
| `WS` | `/ws/stream` | 10 Hz position stream |
| `GET` | `/docs` | Swagger UI |

### WebSocket Payload

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

```kotlin
// Start the foreground service
val intent = Intent(context, NavDriftService::class.java)
intent.putExtra(NavDriftService.EXTRA_MODEL_PATH, "/data/user/0/.../navdrift.ort")
startForegroundService(intent)

// Receive location updates
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

Uses `onnxruntime-android` v1.17.3 with INT4 `.ort` weights for < 5 ms inference.

---

## Mobile PWA (v1.2)

- **Auto-routing** — phones (`width < 900px`) land on `mobile.html`; "↗ Open desktop view" available
- **Installable** — Android Chrome / iOS Safari → "Add to Home Screen"
- **Offline** — service worker: cache-first static, network-first API (3 s timeout → `{"demo_mode": true}`)
- **Layout** — full-screen map, city strip (Delhi / Mumbai / Bengaluru / Chennai / Hyderabad), module chips, 4-metric grid, baro+tunnel row, Pause + GNSS Toggle
- **Live** — badge switches SIM → LIVE when API key configured

---

## Connecting the Live Dashboard

1. Click ⚙️ in the top-right corner
2. Enter `https://navdrift0-api.onrender.com` as Backend URL
3. Enter your `NAVDRIFT_API_KEY`
4. Click **Test & Connect**

> **Tip:** Render free tier spins down after 15 min idle. Visit `/docs` first to wake it (~30 s), then connect.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Core model | PyTorch 2.3, ONNX opset 17 |
| Runtime | ONNX Runtime 1.17 (CPU / ARM NEON) |
| Quantisation | MatMul4BitsQuantizer (INT8 → INT4) |
| Backend | FastAPI 0.111, Uvicorn, asyncio, slowapi |
| Map matching | scipy KD-tree, Viterbi HMM |
| Deployment | Render (free tier), Cloudflare Pages |
| Android SDK | Kotlin, ONNX Runtime Android 1.17.3 |
| Frontend | Vanilla JS, Leaflet.js, service worker |
| Dataset | CARLA 0.9.15, RTKLIB, NumPy, Pandas |

---

## Repository Structure

```
navdrift0/
├── api/
│   └── app.py                # FastAPI — HTTP + WebSocket endpoints
├── frontend/
│   ├── index.html            # Desktop dashboard (auto-redirects mobile)
│   ├── mobile.html           # Mobile PWA layout
│   ├── manifest.json         # PWA manifest
│   └── sw.js                 # Service worker
├── inference/
│   └── export_onnx.py        # ONNX export + INT4 quantisation
├── android/
│   └── NavDriftService.kt    # Android foreground service
├── checkpoints/
│   ├── onnx/                 # drift_former_int8.onnx
│   └── drift_former/         # norm_stats.npz
├── start.sh                  # Render startup (model download + uvicorn)
├── requirements-api.txt      # Production deps (no torch/gradio)
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/swatijs3017/navdrift0
cd navdrift0
pip install -r requirements-api.txt

# Demo mode (no model needed)
DEMO_MODE=true uvicorn api.app:app --host 0.0.0.0 --port 8000

# With real model
HF_REPO_ID=your/repo API_KEY=yourkey bash start.sh
```

---

## License

MIT · © 2026 NAVDRIFT-0 Team · ISRO SIH Problem Statement #26168
