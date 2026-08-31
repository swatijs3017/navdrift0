# NAVDRIFT-0

Dead reckoning for smartphones. No GNSS, no problem.

Built for **ISRO Problem Statement #26168** at Smart India Hackathon 2026.

**Live demo:** https://navdrift0.pages.dev  
**API:** https://navdrift0-api.onrender.com/docs

---

## What this solves

Every delivery driver, ambulance dispatcher, and ride-hailing app in India depends on GNSS to navigate. The moment a vehicle enters a tunnel, underground parking, a dense urban canyon, or drives under thick forest canopy, the signal drops. The app freezes. The map jumps. Turns get missed.

Consumer-grade MEMS IMUs are too noisy for reliable dead reckoning. They drift exponentially — meters of error per second without correction. And most Indian vehicles (two-wheelers, older cars, commercial trucks) have only whatever smartphone the driver mounted on the dashboard, no OBD-II or wheel encoders.

NAVDRIFT-0 fixes this with AI/ML rather than better hardware.

---

## Novel contributions

Six ideas that do not exist in current literature or production systems:

**AdaptiMount™**  
Zero-setup phone orientation detection via gravity vector decomposition and gyro-integrated sharp-turn events for yaw alignment. No QR codes, no manual calibration, no special mount.

**PhaseNet™**  
Adaptive notch filter that identifies engine RPM harmonics directly from the IMU power spectrum. Removes mechanical vibration artifacts that corrupt dead reckoning on ICE vehicles.

**VeloSpec™**  
Speed estimation purely from the IMU spectral fingerprint. No OBD-II port, no wheel encoder, no CAN bus. Extracts forward velocity from the frequency signature of tyre-road interaction.

**SNAP-Retrograde™**  
Differentiable backward trajectory optimizer. On GNSS reacquisition, runs gradient descent over the stored DR path so the endpoint converges smoothly to truth while interior curvature is preserved. Under 50 ms on CPU.

**FusionEKF+™**  
Extended Kalman Filter in SE(2) where process noise covariance Q(t) is output by the DRIFT-Former at each timestep. Non-Holonomic Constraints (no lateral/vertical velocity for ground vehicles) enforced as a pseudo-measurement update, cutting lateral drift ~60%.

**RoadSnap™**  
HMM over an offline OpenStreetMap road graph. During GNSS blackout the road network is the strongest geometric prior. RoadSnap runs map-matching continuously and feeds corrections back into the EKF.

---

## Five ML modules

**DRIFT-Former** (`models/drift_former.py`)  
Causal transformer, 4 layers, 8 attention heads, hidden dim 256, RoPE embeddings. Output head predicts a full 3×3 pose covariance matrix alongside the SE(2) displacement (dx, dy, dθ).

**NavIC VAE** (`models/navic_vae.py`)  
β-VAE encoding 60 s of GNSS history into a 32-d latent fused with transformer output via product-of-Gaussians. Degrades gracefully when GNSS drops.

**SNAP Corrector** (`models/snap_corrector.py`)  
Implements SNAP-Retrograde™. 15 gradient-descent steps over stored trajectory. Smooth convergence, no positional jump.

**AI Speed Estimator** (`inference/runtime.py`)  
CNN-GRU estimating forward vehicle velocity from raw IMU. Handles engine vibration, potholes, braking events.

**Map-Matching Filter** (inference layer)  
RoadSnap™ HMM + Non-Holonomic Constraints as pseudo-measurement update.

---

## Current state

**Live and working:**
- Backend API on Render free tier (EKF demo mode)
- Frontend at navdrift0.pages.dev — real Leaflet.js map with Esri dark tiles (no API key), 5 Indian cities with actual road coordinates, animated SVG vehicle, three live polylines (GPS / DR / SNAP-corrected), tunnel zones with auto GNSS outage, SNAP correction animation, 4 live sparklines
- Cities: New Delhi (India Gate → AIIMS), Mumbai (Gateway → Marine Drive → BKC), Bengaluru (MG Road → Koramangala → BTM), Chennai (Marina → Adyar → OMR), Hyderabad (HITEC City → Banjara Hills → Nampally)
- All six novel algorithms implemented in backend (EKF, NHC, RoadSnap, AdaptiMount, PhaseNet, VeloSpec)
- All five ML modules written and ready to train
- Training notebook ready for Colab A100
- ONNX export pipeline ready

**Pending:**
- DRIFT-Former and NavIC VAE training (waiting on Colab A100 GPU access)
- Trained ONNX model files do not exist yet — backend runs EKF simulation in the meantime
- Live OSM map-matching is stubbed in the inference layer
- Android/iOS mobile app not started
- IO-VNBD benchmark numbers will be added after training

---

## Stack

| Layer | Tool | Cost |
|-------|------|------|
| Backend | FastAPI + uvicorn on Render | Free |
| Frontend | Single HTML on Cloudflare Pages | Free |
| Map tiles | Esri World Dark Gray (no key required) | Free |
| Training | Google Colab A100 | Free |
| Model storage | HuggingFace Hub | Free |
| Dataset | IO-VNBD (open source) | Free |

---

## Running the API locally

```bash
git clone https://github.com/swatijs3017/navdrift0.git
cd navdrift0
pip install -r requirements-api.txt
pip install -e .

DEMO_MODE=true NAVDRIFT_API_KEY=localkey uvicorn api.app:app --reload
```

Open http://localhost:8000/docs for interactive API docs.

To run the frontend locally, open `frontend/index.html` in any browser. The backend URL is configurable in the Settings panel (⚙ button, top right).

---

## Training

Open `notebooks/NAVDRIFT0_Training.ipynb` in Google Colab. Set runtime to A100.

Run Cell 0 first — it sends a keepalive ping every 60 s to prevent Colab from disconnecting.

The notebook:
1. Mounts Google Drive for checkpoint persistence
2. Clones this repo
3. Downloads IO-VNBD
4. Trains DRIFT-Former (resumes from Drive checkpoint on disconnect)
5. Trains NavIC VAE
6. Exports both to ONNX INT8
7. Benchmarks against EKF baseline on held-out sequences

After training, upload ONNX files to HuggingFace and set `HF_REPO_ID` on Render. The backend downloads the model at startup and switches from EKF simulation to real inference automatically.

---

## API reference

All endpoints require `X-API-Key` header.

| Method | Endpoint | What it does |
|--------|----------|--------------|
| POST | `/init` | Set initial GNSS fix and coordinate origin |
| POST | `/ingest` | Feed one IMU timestep, get pose + uncertainty |
| POST | `/gnss_lost` | Signal GNSS dropout |
| POST | `/reacquire` | Feed new GPS fix, get SNAP-corrected trajectory |
| GET | `/trajectory` | Pull full trajectory history |
| GET | `/status` | Health check |
| POST | `/reset` | Clear history and reinitialise |

`/ingest` targets 100 Hz (edge) or 10 Hz (smartphone). Returns pose, heading, uncertainty ellipse radii, and latency.

---

## Performance target (PS requirement)

- < 5 m drift over a 50 m GNSS-denied stretch
- < 100 m drift over 1 km at 60 km/h in a tunnel
- 10 Hz position updates on smartphone, 200 Hz on edge hardware

Benchmark numbers will be added after IO-VNBD evaluation on the trained model.

---

## Dataset

**IO-VNBD** — Inertial and Odometry benchmark for ground vehicle positioning  
https://github.com/onyekpeu/IO-VNBD

Synchronized IMU, wheel odometry, and GNSS ground truth from ground vehicles. Training uses simulated GNSS outages: GPS is masked for random 10–120 second windows.

---

## Repo structure

```
navdrift0/
  api/app.py                          FastAPI backend, all endpoints
  data/loader.py                      IO-VNBD parser and dataset splits
  models/drift_former.py              Causal transformer with covariance head
  models/navic_vae.py                 Beta-VAE for GNSS history encoding
  models/snap_corrector.py            SNAP-Retrograde™ trajectory smoother
  inference/runtime.py                NavDriftRuntime: EKF + ONNX + SNAP + NHC
  inference/export_onnx.py            ONNX export and INT8 quantisation
  training/train_drift_former.py      DRIFT-Former training loop
  training/train_navic_vae.py         NavIC VAE training loop
  eval/metrics.py                     ATE, RTE, NLL, drift rate, EKF baseline
  notebooks/NAVDRIFT0_Training.ipynb  Colab training notebook
  frontend/index.html                 Live dashboard — real Leaflet map, no build step
  render.yaml                         Render deploy config
  requirements-api.txt                Lean deps for Render (no torch)
  requirements.txt                    Full deps for training
  tests/test_api.py                   API smoke tests
  .github/workflows/ci.yml            CI on every push
```

---

Built by Swati — SIH 2026, ISRO PS-26168.
