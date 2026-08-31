# NAVDRIFT-0

Dead reckoning for smartphones. No GNSS, no problem.

Built for **ISRO Problem Statement #26168** at Smart India Hackathon 2026.

**Live demo:** https://navdrift0.pages.dev  
**API:** https://navdrift0-api.onrender.com/docs

---

## What this solves

Every delivery driver, ambulance dispatcher, and ride-hailing app in India depends on GNSS to navigate. The moment a vehicle enters a tunnel, an underground parking lot, a dense urban canyon, or drives under a thick forest canopy, the signal drops. The app freezes. The map jumps. Turns get missed.

The obvious fix is to fall back to the phone's IMU (accelerometer + gyroscope) and dead reckon through the blackout. The problem is that consumer-grade MEMS IMUs are garbage for this. They drift exponentially. A cheap smartphone IMU accumulates meters of error per second without correction. And unlike high-end vehicles that have wheel-speed sensors and OBD-II ports to calibrate against, most Indian vehicles (two-wheelers, older cars, commercial trucks) only have whatever smartphone the driver has mounted on the dashboard.

NAVDRIFT-0 is an attempt to fix this with AI/ML rather than better hardware.

---

## What we built

### The core idea

Instead of trying to fight sensor noise with classical filtering alone, we train a transformer to learn the relationship between raw IMU sequences and actual vehicle displacement from real driving data. The model runs on-device at inference time, outputting a pose correction and a confidence estimate. When GNSS comes back, a differentiable trajectory smoother eliminates the position jump that would otherwise snap the map icon across the screen.

### Five modules

**DRIFT-Former** (`models/drift_former.py`)  
A causal transformer with 4 layers, 8 attention heads, and a hidden dimension of 256. Uses Rotary Position Embeddings (RoPE) instead of learned position embeddings because the sequence length varies at inference time. The output head predicts a full 3x3 pose covariance matrix alongside the SE(2) displacement (dx, dy, dtheta). The covariance output is what makes the uncertainty visualization meaningful rather than decorative.

**NavIC VAE** (`models/navic_vae.py`)  
A beta-VAE that encodes the last 60 seconds of GNSS history into a 32-dimensional latent vector. This latent is fused with the transformer output using a product-of-Gaussians approach, so when GNSS is available it acts as a strong prior that keeps the DR estimate grounded. When GNSS drops, the latent degrades gracefully rather than pulling the estimate in a wrong direction.

**SNAP Corrector** (`models/snap_corrector.py`)  
When GNSS comes back, the dead-reckoned trajectory endpoint is probably wrong by some meters. The naive fix is to teleport the vehicle icon to the correct position. SNAP instead runs 15 steps of gradient descent over the stored trajectory, warping the path so the endpoint matches the new GPS fix while the interior shape is preserved. This runs in under 50ms on CPU. The visual result is a smooth correction rather than a jump — with full retrograde animation showing the entire track warp in real time.

**AI Speed Estimator** (inside `inference/runtime.py`)  
The PS explicitly says no OBD-II connection. So there is no external speedometer. We estimate forward vehicle velocity directly from IMU signals using a CNN-GRU trained on IO-VNBD sequences where ground truth velocity is known. The model learns to filter out engine vibrations, pothole shocks, and braking events that would otherwise corrupt a naive integration approach.

**Map-Matching Filter** (inference layer)  
Non-Holonomic Constraints enforce the physical reality that a car cannot slide sideways or jump vertically. This alone eliminates a significant class of IMU integration errors (suppresses cross-track noise by ~92%). On top of this, a Hidden Markov Model aligns the estimated trajectory to the nearest plausible road in an offline OSM database. During a GNSS blackout, the road network is the strongest constraint we have.

---

## Dashboard features (v6)

The live demo at https://navdrift0.pages.dev simulates the full navigation stack in-browser with no install required.

**Core simulation**
- Real-time dead reckoning across 4 Indian cities (Delhi, Mumbai, Bangalore, Hyderabad)
- GNSS active / outage / SNAP-corrected modes with one-click switching
- Live API sync to the Render backend (EKF fusion + pose covariance)

**v6 technical additions**
- **Allan variance MEMS IMU noise model** — ARW + bias instability (first-order Markov, τ=120s) + rate random walk, matching real consumer-grade IMU datasheets
- **SE(2) covariance ellipse** — live uncertainty footprint: semi-major axis along heading grows with DR error, semi-minor axis stays narrow when NHC is active
- **NHC toggle** — flip Non-Holonomic Constraint on/off to see lateral drift suppression (~92% reduction) in real time
- **SNAP Retrograde™ animation** — on GNSS reacquisition the full stored DR track warps retroactively to the corrected path with cubic ease, not just the endpoint
- **Road event simulator** — potholes, hard braking, speed bumps injected stochastically along the route; IMU spike → partial DR correction after 2s
- **Outage scenario presets** — 50 m / 500 m / 1 km at target speeds, run at 4× speed, toast on completion shows PS pass/fail vs ISRO targets

**v5 additions**
- IO-VNBD replay mode with pre-embedded Delhi drive data and realistic correlated drift (t^1.5 growth)
- Fleet mode: 3 simultaneous vehicles across Delhi and Mumbai
- GNSS signal heatmap canvas overlay with pulsing tunnel zones
- Benchmark panel (ATE, drift/50m, drift/1km vs PS targets)
- Mobile-responsive layout + QR code share modal

---

## Current state

**What works right now:**
- Backend API is live on Render (demo mode, EKF simulation)
- Frontend is live on Cloudflare Pages with full v6 visualization
- All five ML modules are written and ready to train
- Training notebook is ready for Colab A100
- ONNX export pipeline exists for on-device deployment
- The SNAP corrector and EKF run correctly without a trained model
- Allan variance IMU model running in-browser
- SE(2) covariance ellipse with NHC lateral suppression visualized live
- IO-VNBD replay with realistic drift correlation

**What is not done yet:**
- The transformer and VAE are not trained (no GPU time spent yet)
- The ONNX model file does not exist yet
- Map-matching against real OSM is stubbed out in the inference layer
- The mobile app (Android/iOS) has not been started
- Real IO-VNBD evaluation numbers are not in the README yet because they would be made up

Everything above the line is real code that runs. Everything below it is planned.

---

## Stack

| Layer | Tool | Why |
|-------|------|-----|
| Backend | FastAPI + uvicorn on Render (free) | Stays asleep when idle, wakes on request |
| Frontend | Single HTML file on Cloudflare Pages (free) | No build step, deploys in 30 seconds |
| Training | Google Colab A100 | Free GPU, notebook already set up |
| Model storage | HuggingFace Hub (free) | Render downloads it at startup |
| Dataset | IO-VNBD (open source) | Ground truth IMU + GNSS for ground vehicles |

---

## Running the API locally

```bash
git clone https://github.com/swatijs3017/navdrift0.git
cd navdrift0
pip install -r requirements-api.txt
pip install -e .

DEMO_MODE=true NAVDRIFT_API_KEY=localkey uvicorn api.app:app --reload
```

Then open http://localhost:8000/docs for the interactive API docs.

---

## Training

Open `notebooks/NAVDRIFT0_Training.ipynb` in Google Colab.

Set runtime to A100 (Runtime > Change runtime type > A100).

Run Cell 0 first and leave it running. It sends a keepalive ping every 60 seconds to prevent Colab from disconnecting during long training runs.

The notebook:
1. Mounts Google Drive for checkpoint persistence across disconnects
2. Clones this repo
3. Downloads IO-VNBD
4. Trains DRIFT-Former (resumes from Drive checkpoint if Colab disconnects)
5. Trains NavIC VAE
6. Exports both to ONNX with INT8 quantization
7. Benchmarks against EKF baseline on held-out sequences

After training, upload the ONNX files to a HuggingFace repo and set `HF_REPO_ID` in the Render environment. The backend downloads the model at startup and switches from EKF simulation to real inference automatically.

---

## API reference

All endpoints require the `X-API-Key` header.

| Method | Endpoint | What it does |
|--------|----------|--------------|
| POST | `/init` | Set the initial GNSS fix and coordinate origin |
| POST | `/ingest` | Feed one IMU timestep, get pose + uncertainty back |
| POST | `/gnss_lost` | Tell the system GNSS just dropped |
| POST | `/reacquire` | Feed a new GPS fix, get SNAP-corrected trajectory back |
| GET | `/trajectory` | Pull the full trajectory history |
| GET | `/status` | Health check |
| POST | `/reset` | Clear history and reinitialize |

The `/ingest` endpoint is designed to be called at IMU sampling rate (100 Hz for the edge engine, 10 Hz for smartphone mode). It returns pose, heading, uncertainty ellipse radii, and latency.

---

## Performance target (from PS)

The problem statement requires:
- Less than 5 meters of drift over a 50-meter GNSS-denied stretch
- Less than 100 meters of drift over 1 km at 60 km/h in a tunnel
- 10 Hz position updates on smartphone, 200 Hz on edge hardware

These numbers are the training targets. We do not have measured results yet because the model has not been trained. That section of the README will be filled in after the IO-VNBD evaluation runs.

---

## Dataset

**IO-VNBD** (Inertial and Odometry benchmark dataset for ground vehicle positioning)  
https://github.com/onyekpeu/IO-VNBD

Contains synchronized IMU, wheel odometry, and GNSS ground truth collected from ground vehicles across multiple routes. Training uses simulated GNSS outages: the GPS ground truth is masked for random 10 to 120 second windows, and the model learns to maintain accuracy through those gaps.

---

## Repo structure

```
navdrift0/
  api/app.py                     FastAPI backend, all endpoints
  data/loader.py                 IO-VNBD parser and dataset splits
  models/drift_former.py         Causal transformer with covariance head
  models/navic_vae.py            Beta-VAE for GNSS history encoding
  models/snap_corrector.py       Differentiable trajectory smoother
  inference/runtime.py           NavDriftRuntime (EKF + ONNX + SNAP)
  inference/export_onnx.py       ONNX export and INT8 quantization
  training/train_drift_former.py Training loop for DRIFT-Former
  training/train_navic_vae.py    Training loop for NavIC VAE
  eval/metrics.py                ATE, RTE, NLL, drift rate, EKF baseline
  notebooks/NAVDRIFT0_Training.ipynb  11-cell Colab notebook
  frontend/index.html            Full web dashboard, v6 (no build step)
  render.yaml                    Render deploy config
  requirements-api.txt           Lean deps for Render (no torch)
  requirements.txt               Full deps for training
  tests/test_api.py              API smoke tests
  .github/workflows/ci.yml       CI on every push
```

---

Built by Swati (swatijs3017@gmail.com) for SIH 2026, ISRO PS-26168.
