# NAVDRIFT-0

**AI-ML based Intelligent Dead Reckoning for seamless vehicle navigation**  
ISRO Problem Statement #26168 · SIH 2026 · Theme: Smart Vehicles

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

---

## Problem

Vehicle logistics, ride-hailing, and emergency services rely on GNSS (GPS/NavIC/Galileo). When a vehicle enters a **tunnel, underpass, dense urban canyon, or multi-level parking**, GNSS drops entirely. Navigation apps freeze or jump erratically.

Worse: most Indian vehicles (trucks, two-wheelers, older cars) only have a **smartphone as a navigation device**. The phone's MEMS IMU is noisy, affected by vibrations, potholes, and engine harmonics — and without an ODB-II speedometer feed, raw dead reckoning drifts exponentially within seconds.

## NAVDRIFT-0 Solution

A lightweight, **edge-deployable AI/ML engine** that transforms a standalone smartphone into an Intelligent Dead Reckoning system with seamless GNSS fusion. Five technical modules:

| Module | Description |
|--------|-------------|
| **DRIFT-Former** | Causal Transformer (4L, 8H, d=256, RoPE positional encoding) — predicts SE(2) pose delta + full 3×3 covariance from IMU window |
| **NavIC VAE** | β-VAE encodes 60s GNSS history into 32-dim latent; constrains DR via product-of-Gaussians fusion |
| **SNAP Corrector** | Differentiable trajectory smoother — gradient descent warp that eliminates the position jump on GNSS reacquisition in <50ms CPU |
| **AI Speed Estimator** | CNN-GRU trained on IO-VNBD to predict forward vehicle speed from noisy IMU alone (no ODB-II), filtering out potholes and vibrations |
| **Map-Matching Filter** | Hidden Markov + Non-Holonomic Constraints snaps trajectory to OSM road network; vehicle can't slide sideways or fly upward |

## Performance Targets (per PS benchmark)

- **Dead Reckoning**: <5 m drift over 50 m GNSS-denied environment (<10% of distance)
- **GNSS+INS Fusion**: 10 Hz position update on smartphone; 200 Hz on edge hardware
- **SNAP correction**: <50 ms on CPU after GNSS reacquisition

## Architecture

```
IMU (accel/gyro/mag) + GNSS (when available)
         │
         ▼
┌─────────────────────┐
│ Alignment Calibrator │  ← auto-detects phone pitch/roll/yaw vs. vehicle frame
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  AI Speed Estimator  │  ← CNN-GRU, no ODB-II required
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│    DRIFT-Former      │  ← Causal Transformer, RoPE, SE(2)+Covariance
│    NavIC VAE         │  ← Product-of-Gaussians GNSS fusion
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Map-Matching + NHC   │  ← OSM + Hidden Markov + Non-Holonomic Constraints
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│   SNAP Corrector     │  ← Differentiable trajectory smoother
└─────────┬───────────┘
          ▼
    Live Position + Uncertainty
```

## Repo Structure

```
navdrift0/
├── api/                    # FastAPI backend (Render)
│   └── app.py              # All endpoints: /ingest /reacquire /status /trajectory
├── data/
│   └── loader.py           # IO-VNBD parser, GNSS masking, dataset splits
├── models/
│   ├── drift_former.py     # Causal Transformer + RoPE + covariance head
│   ├── navic_vae.py        # Motion prior VAE + Product-of-Gaussians fusion
│   └── snap_corrector.py   # Differentiable SNAP trajectory smoother
├── training/
│   ├── train_drift_former.py
│   └── train_navic_vae.py
├── inference/
│   ├── runtime.py          # NavDriftRuntime (EKF + ONNX + SNAP)
│   └── export_onnx.py      # ONNX export + INT8 quantization
├── eval/
│   └── metrics.py          # ATE, RTE, NLL, drift rate, EKF baseline
├── notebooks/
│   └── NAVDRIFT0_Training.ipynb  # Colab A100, 11 cells, anti-disconnect
├── frontend/
│   └── index.html          # Full dynamic web app (no build step)
├── render.yaml             # Render deploy config
├── requirements-api.txt    # Lean Render requirements (no torch)
├── requirements.txt        # Full training requirements
└── setup.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/init` | Set initial GNSS fix + coordinate frame |
| `POST` | `/ingest` | Feed IMU+odometry step → pose + uncertainty |
| `POST` | `/gnss_lost` | Notify system of GNSS loss |
| `POST` | `/reacquire` | New GNSS fix → SNAP-corrected trajectory |
| `GET` | `/trajectory` | Full trajectory history |
| `GET` | `/status` | Health check |

All endpoints require `X-API-Key` header.

## Dataset

**IO-VNBD** — Inertial and Odometry benchmark dataset for ground vehicle positioning  
https://github.com/onyekpeu/IO-VNBD

Training uses simulated GNSS outages: segments of 10–120s masked from GPS ground truth.

## Deployment

### Backend (Render — free tier)

1. Fork this repo
2. Connect to [render.com](https://render.com) → New Web Service → select this repo
3. Build command: `pip install -r requirements-api.txt && pip install -e .`  
   (or just connect — `render.yaml` has this baked in)
4. Set env vars: `NAVDRIFT_API_KEY` (generate), `DEMO_MODE=true`

### Frontend (Cloudflare Pages — free tier)

1. [pages.cloudflare.com](https://pages.cloudflare.com) → Create project → connect GitHub
2. Build output dir: `frontend`
3. Build command: (leave empty)
4. Live at `https://navdrift0.pages.dev`

## Training (Google Colab A100)

Open `notebooks/NAVDRIFT0_Training.ipynb`:
- Runtime → Change runtime type → **A100 GPU**
- Run Cell 0 first (anti-disconnect heartbeat)
- Cells train DRIFT-Former → NavIC VAE → export ONNX → benchmark vs EKF

## Team

Built for SIH 2026 · ISRO Problem Statement #26168  
**Swati** · swatijs3017@gmail.com
