# NAVDRIFT-0

Dead reckoning for ground vehicles using a trained causal transformer. No GNSS, no problem.

Built for **ISRO Problem Statement #26168**, Smart India Hackathon 2026.

**Live demo:** https://navdrift0.pages.dev  
**API docs:** https://navdrift0-api.onrender.com/docs  
**Trained models:** https://github.com/swatijs3017/navdrift0/releases/tag/v1.0.0

---

## The problem

India has 6.3 lakh kilometres of national and state highways, thousands of tunnels, and tens of millions of vehicles that rely on smartphone GPS for navigation. Consumer GNSS works until it does not. Tunnels, underpasses, dense urban canyons, multi-level parking structures, metro station approaches, tree-lined corridors in monsoon, military convoy routes through hilly terrain -- all of these cause GNSS signal loss that ranges from a few seconds to several minutes.

When signal drops, every navigation app currently does one of two things: it freezes the position marker, or it dead reckons using naive IMU integration. The first option is useless. The second option is worse than useless because cheap MEMS accelerometers and gyroscopes found in smartphones drift quadratically with time. A 10-second blackout with naive integration can accumulate 15 to 40 metres of error. A 60-second tunnel produces errors large enough to put the vehicle on the wrong road entirely.

ISRO PS-26168 asks for a system that can maintain position accuracy through GNSS-denied stretches using only the sensors already present in a smartphone or vehicle ECU: a 6-axis IMU and vehicle odometry (speed + steering angle). The targets are under 5 metres of drift over a 50-metre blackout and under 100 metres of drift over a 1 kilometre tunnel at 60 km/h.

---

## What NAVDRIFT-0 does

Instead of fighting sensor noise with hand-tuned Kalman filter parameters, NAVDRIFT-0 trains a transformer model on real driving data where ground truth is known. The model learns the statistical relationship between sequences of IMU readings and actual vehicle displacement. At inference time, it outputs both a pose estimate and a full covariance matrix that describes how confident it is. When GNSS returns, a differentiable trajectory smoother corrects the accumulated error without any visible position jump.

The whole inference stack runs on CPU at 20ms per step. No GPU required at deployment. The model is 14MB quantised to INT8.

---

## Results

Evaluated on the held-out test split of IO-VNBD dataset.

| Metric | NAVDRIFT-0 | EKF Baseline | ISRO Target |
|--------|-----------|-------------|-------------|
| Mean ATE over 1 km | 78.41 m | 121.69 m | under 100 m |
| Position drift over 50 m blackout | 3.19 m | not measured | under 5 m |
| CPU inference latency | 20 ms | not applicable | under 100 ms |
| Update rate | 10 Hz | 10 Hz | 10 Hz |

All three ISRO targets passed. NAVDRIFT-0 is 35.6% more accurate than the EKF baseline on mean ATE.

---

## Architecture

The system has five components that run in sequence on every sensor timestep.

### 1. DRIFT-Former

The core model. A causal transformer trained to map a sliding window of IMU and odometry readings to SE(2) pose deltas.

**Input:** a window of W=200 timesteps, each containing 8 channels: accel_x, accel_y, accel_z (m/s^2), gyro_x, gyro_y, gyro_z (rad/s), speed (m/s), steer_angle (radians).

**Architecture:**
- Linear input projection to d_model=256
- 4 transformer encoder layers, each with 8 attention heads and FFN expansion factor 4
- Rotary Position Embeddings (RoPE) instead of learned position embeddings -- this matters because the sequence length can vary at inference time without retraining
- Causal attention mask so the model cannot look ahead in time
- Two output heads: a mean head predicting (dx, dy, dtheta) and a covariance head predicting a 3x3 lower-triangular matrix for the full pose covariance

**Why a transformer and not an LSTM:** transformers can attend to any part of the input window, which lets them pick up on long-range patterns like a vehicle decelerating 5 seconds before a sharp turn. LSTMs compress history into a fixed hidden state and lose this.

**Why RoPE:** standard learned position embeddings are length-specific. If you train on W=200 and try to run inference on W=100 (shorter window at startup), the model breaks. RoPE encodes position as a rotation in the attention score space, which generalises to any length.

**Training objective:** MSE on pose deltas plus a covariance trace penalty. The trace penalty prevents the model from collapsing the covariance to near-zero (which it will do if trained purely on NLL, because minimising log det of a tiny matrix gives infinite reward). NLL is introduced after epoch 20 once the covariance has stabilised.

**Training details:** 60 epochs on IO-VNBD, A100 GPU on Google Colab, AdamW with lr=2e-4, cosine LR schedule, gradient clipping at 1.0. Checkpoints saved to Google Drive every epoch for resume on disconnection.

### 2. NavIC VAE

A variational autoencoder that encodes recent sensor history into a compact latent representation. This latent captures the motion prior -- what kind of driving is happening right now (highway at 80 km/h, city stop-and-go, sharp corners).

**Input:** last 60 timesteps of 4 channels: accel_x, accel_y, speed, steer_angle.

**Architecture:**
- Encoder: 3-layer MLP projecting to mu and logvar of a 32-dimensional Gaussian
- Reparameterisation trick: z = mu + eps * exp(logvar/2)
- Decoder: 3-layer MLP reconstructing the input sequence for the reconstruction loss

**Why:** the latent z from the VAE is available as an additional signal that could be used to condition the DRIFT-Former output or serve as an anomaly detector (large KL divergence means the current motion pattern is unusual). In the current deployment it is trained and exported but not yet fused with the main inference path, making it available for the next iteration.

### 3. SNAP Corrector

When GNSS reacquires, the dead-reckoned trajectory endpoint is wrong by some number of metres. The naive solution is to teleport the vehicle marker. SNAP avoids this.

SNAP buffers all raw SE(2) deltas and covariance matrices during the GNSS outage. On reacquisition, it runs 15 steps of gradient descent over the buffered trajectory, optimising correction factors applied to each delta such that the trajectory endpoint matches the new GPS fix while the interior trajectory shape is minimally disturbed.

The loss function is: squared distance between corrected endpoint and GPS fix, plus a regularisation term penalising large corrections (weighted by the inverse covariance of each delta, so high-confidence steps are corrected less). This runs in under 50ms on CPU.

The result is a smooth retroactive correction. The vehicle icon does not jump.

### 4. ONNX Runtime Wrapper

The NavDriftRuntime class handles everything at inference time:

- Loads the INT8-quantised ONNX model via ONNX Runtime (CPUExecutionProvider)
- Maintains the sliding window buffer as a deque of length 200
- Runs the ONNX session on every timestep, extracts the last-timestep prediction
- Integrates SE(2) deltas using the rotation-correct formula: new_x = x + cos(theta)*dx - sin(theta)*dy
- Buffers deltas and covariances during GNSS outage for SNAP
- Downloads the model from GitHub Releases on first startup (both drift_former.onnx and drift_former.onnx.data must be present side by side)

### 5. FastAPI Backend

Production-grade REST API deployed on Render free tier.

Security: API key authentication via X-API-Key header using constant-time comparison (secrets.compare_digest). CORS restricted to configured origins. 64KB request body limit. All inputs validated with Pydantic with strict bounds. No stack traces returned to clients. Rate limiting via slowapi.

Endpoints: /init, /ingest, /gnss_lost, /reacquire, /trajectory, /status, /reset.

---

## Dataset

**IO-VNBD** (Inertial and Odometry benchmark dataset for ground vehicle positioning)  
https://github.com/onyekpeu/IO-VNBD

IO-VNBD contains synchronised 6-axis IMU, wheel odometry, and GNSS ground truth collected from ground vehicles across multiple routes and driving conditions. It is the correct dataset for PS-26168 because it matches the sensor configuration specified in the problem statement (IMU + odometry, no camera, no lidar) and is collected from ground vehicles rather than drones or pedestrians.

Training uses simulated GNSS outages: GPS ground truth is masked for random 10 to 120 second windows, and the model learns to maintain accuracy through those gaps. The model never sees GPS during these windows at training time, only the IMU and odometry channels.

---

## Model storage

Models are stored on GitHub Releases (v1.0.0), not HuggingFace. Direct download URLs, versioned releases, no rate limits, no authentication required for public repos. The Render backend downloads both the .onnx and .onnx.data files on cold start via NAVDRIFT_DR_MODEL_URL environment variable.

Total model size: approximately 14MB for DRIFT-Former (INT8 quantised).

---

## Tech stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| ML model | PyTorch 2.1, transformer with RoPE | Causal attention, variable sequence length |
| Inference | ONNX Runtime 1.17, CPUExecutionProvider | No GPU needed at runtime, 20ms latency |
| Quantisation | INT8 dynamic quantisation on Linear layers | 2-4x speedup on ARM/x86 BLAS |
| API | FastAPI + uvicorn | Async, fast, built-in OpenAPI docs |
| Deployment | Render free tier | Stays alive with NAVDRIFT_DR_MODEL_URL auto-download |
| Frontend | Leaflet.js + OpenStreetMap | Real map tiles, zero API key required |
| Frontend hosting | Cloudflare Pages | Deploys on git push, global CDN, free |
| Dataset | IO-VNBD | Only open IMU+odometry dataset with GNSS ground truth for ground vehicles |
| Model hosting | GitHub Releases | Versioned, direct URLs, production-grade, completely free |
| Training compute | Google Colab A100 | Free GPU, notebook designed to resume from Drive on disconnect |

---

## Training pipeline

The full pipeline is in `notebooks/NAVDRIFT0_Training.ipynb`. It runs on Google Colab A100 and is designed to survive disconnections by checkpointing every epoch to Google Drive.

**Cell 1:** Mount Drive, clone repo, install dependencies, download IO-VNBD.

**Cell 2:** Parse and normalise IO-VNBD. Compute per-channel mean and std from training split. Save normalisation stats to Drive.

**Cell 3:** Dataset class. Generates sliding windows of 200 timesteps. Masks random GNSS blackout windows of 10 to 120 seconds. Returns (imu_odom_window, pose_delta_target) pairs.

**Cell 4:** Define DRIFTFormer. Load from Drive checkpoint if one exists, otherwise initialise from scratch.

**Cell 5:** Training loop. MSE + covariance trace penalty for epochs 0-19, adds mild NLL term from epoch 20. AdamW, lr=2e-4, cosine schedule. Saves checkpoint every epoch to Drive.

**Cell 6:** NavIC VAE training. Separate model, separate checkpoint. KL annealing over first 20 epochs to avoid posterior collapse.

**Cell 7:** ONNX export. Exports DRIFT-Former (opset 17, dynamic batch and sequence axes). Applies INT8 dynamic quantisation. Benchmarks CPU latency over 200 runs and reports mean, std, p95.

**Cell 8:** Benchmark against EKF baseline on held-out sequences. Reports ATE, 50m drift, latency. Serialises results to benchmark_results.json.

**Cell 9:** Upload models and benchmark results to GitHub Releases v1.0.0 via GitHub API.

**Cell 10:** Commit benchmark_results.json to repo.

---

## Running locally

```bash
git clone https://github.com/swatijs3017/navdrift0.git
cd navdrift0
pip install -r requirements.txt
pip install -e .

DEMO_MODE=true \
NAVDRIFT_DR_MODEL_URL=https://github.com/swatijs3017/navdrift0/releases/download/v1.0.0/drift_former.onnx \
uvicorn api.app:app --reload
```

The model downloads automatically (~14MB). Open http://localhost:8000/docs.

To use your own trained checkpoint instead:

```bash
NAVDRIFT_API_KEY=yourkey \
ONNX_PATH=./checkpoints/onnx/drift_former.onnx \
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

---

## API reference

All endpoints require `X-API-Key` header.

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| POST | /init | latitude, longitude, heading_deg, speed_m_s | Coordinate origin set |
| POST | /ingest | accel_x/y/z, gyro_x/y/z, speed, steer_angle, timestamp | pose_x, pose_y, heading_rad, uncertainty_major, uncertainty_minor, latency_ms |
| POST | /gnss_lost | none | Starts buffering for SNAP |
| POST | /reacquire | latitude, longitude, heading_deg | SNAP-corrected trajectory, endpoint error |
| GET | /trajectory | none | Full pose history as (x, y, theta) list |
| GET | /status | none | Model loaded, step count, current position |
| POST | /reset | latitude, longitude, heading_deg, speed_m_s | Clears history, reinitialises |

/ingest is designed to be called at IMU rate (10Hz smartphone, up to 100Hz edge hardware). Each call runs one ONNX inference pass and returns in under 25ms including Python overhead.

---

## Frontend

Single file at `frontend/index.html`. No build step, no npm, no bundler. Deploys to Cloudflare Pages on every git push.

**Map:** Leaflet.js with OpenStreetMap tiles filtered dark (brightness 25%, hue rotated to blue-green for mission-control aesthetic). Five cities with real lat/lon route waypoints: Delhi (Connaught Place area), Mumbai (Marine Drive), Bengaluru (Outer Ring Road), Chennai (Anna Salai), Hyderabad (ORR).

**Trajectories:** Three polylines drawn in real time: ground truth (cyan), NAVDRIFT-0 estimate (violet), raw IMU dead reckoning (red dashed). Vehicle marker follows NAVDRIFT-0 position. Uncertainty ellipse grows during GNSS blackout and shrinks on reacquisition.

**Live API mode:** Click the settings gear, enter your Render backend URL and API key. The frontend calls /ingest on every simulation step with synthetic IMU values derived from the route geometry. All displayed metrics -- ATE, latency, uncertainty -- come from real API responses. The mode badge switches from SIMULATION to LIVE API.

**EKF baseline:** A 3-state Kalman filter runs in JavaScript on every step. The benchmark table shows real EKF numbers, not a hardcoded multiplier.

**Simulation fallback:** Works entirely offline without an API key. The JS physics engine takes over so the demo runs anywhere.

---

## Repo structure

```
navdrift0/
  api/
    app.py                       FastAPI backend, all endpoints, auth, middleware
  data/
    loader.py                    IO-VNBD parser, normalisation stats, dataset class
  models/
    drift_former.py              Causal transformer with RoPE and covariance head
    navic_vae.py                 Beta-VAE for motion prior encoding
    snap_corrector.py            Differentiable trajectory smoother
  inference/
    runtime.py                   NavDriftRuntime: ONNX session, sliding window, SNAP
    export_onnx.py               Export to ONNX, INT8 quantisation, latency benchmark
  training/
    train_drift_former.py        Training loop for DRIFT-Former
    train_navic_vae.py           Training loop for NavIC VAE
  eval/
    metrics.py                   ATE, RTE, NLL, drift rate, EKF baseline implementation
    benchmark_results.json       Real numbers from IO-VNBD held-out evaluation
  notebooks/
    NAVDRIFT0_Training.ipynb     10-cell Colab notebook, full pipeline
  frontend/
    index.html                   Complete web dashboard, single file
  render.yaml                    Render service configuration
  requirements.txt               Full training dependencies
  setup.py                       Package install
```

---

## Potential next steps

**Map-matching HMM:** During a GNSS blackout, the road network is the strongest constraint available. A Hidden Markov Model that aligns the dead-reckoned trajectory to plausible roads in an offline OSM graph could cut ATE significantly in urban environments where roads constrain possible positions.

**Non-holonomic constraints:** Cars cannot slide sideways or jump vertically. Enforcing these as hard constraints on the pose delta output (zero lateral velocity, zero vertical velocity) removes a whole class of IMU integration errors for free.

**Android SDK:** Wrapping the ONNX Runtime in a Kotlin foreground service would make NAVDRIFT-0 callable from any navigation app on Android. The service exposes a LocationListener-compatible interface so apps need minimal changes.

**INT4 quantisation:** ONNX Runtime Mobile supports INT4 weight compression. Combined with ARM NEON SIMD on modern Snapdragon chips this could bring latency under 5ms and enable true 200Hz operation on-device.

**Multi-dataset training:** IO-VNBD is good but limited in geographic diversity. Training on KITTI, Oxford RobotCar, and KAIST Urban combined would give the model exposure to a wider range of road conditions and vehicle dynamics.

**WebSocket streaming:** The current API uses HTTP polling. A WebSocket /stream endpoint would let the frontend update at true 10Hz without repeated TCP handshakes.

**Barometer fusion:** Adding barometric altitude as a 9th input channel helps detect tunnel entry and exit, which could be used to modulate the uncertainty estimate (we know we are in a tunnel so uncertainty should grow faster).
