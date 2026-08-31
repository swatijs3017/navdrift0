"""
This file is the source for NAVDRIFT0_Training.ipynb.
Run: python notebooks/build_notebook.py to generate the .ipynb file.
Or use the cells below directly in Colab.

The notebook is structured as:
  Cell 0:  Anti-disconnect JS
  Cell 1:  Connect Google Drive
  Cell 2:  Install dependencies
  Cell 3:  Clone repo / mount code
  Cell 4:  Download IO-VNBD dataset
  Cell 5:  Parse and preprocess
  Cell 6:  Train DRIFT-Former
  Cell 7:  Train NavIC VAE
  Cell 8:  Export to ONNX + quantize
  Cell 9:  Evaluate on test set + baselines
  Cell 10: Visualize results
"""

NOTEBOOK_CELLS = [
    # -----------------------------------------------------------------------
    # Cell 0: Anti-disconnect (run this first, keep it running)
    # -----------------------------------------------------------------------
    {
        "type": "code",
        "source": """\
%%javascript
// Anti-disconnect: click the page every 60s to keep Colab alive
// Keep this cell running throughout your session.
function preventDisconnect() {
  var buttons = document.querySelectorAll("colab-toolbar-button");
  console.log("[NAVDRIFT-0] Anti-disconnect heartbeat");
  document.querySelector("#top-toolbar > paper-icon-button")
    && document.querySelector("#top-toolbar > paper-icon-button").click();
}
setInterval(preventDisconnect, 60000);
console.log("[NAVDRIFT-0] Anti-disconnect armed (60s interval)");
"""
    },

    # -----------------------------------------------------------------------
    # Cell 1: Connect Google Drive for persistent checkpoints
    # -----------------------------------------------------------------------
    {
        "type": "code",
        "source": """\
from google.colab import drive
import os

drive.mount('/content/drive', force_remount=False)

# Create a persistent output directory on Drive
DRIVE_DIR = '/content/drive/MyDrive/NAVDRIFT0'
os.makedirs(DRIVE_DIR, exist_ok=True)
os.makedirs(f'{DRIVE_DIR}/checkpoints', exist_ok=True)
os.makedirs(f'{DRIVE_DIR}/data', exist_ok=True)
os.makedirs(f'{DRIVE_DIR}/onnx', exist_ok=True)

print(f"Drive connected. Output directory: {DRIVE_DIR}")
print("All checkpoints will autosave here — safe across disconnects.")
"""
    },

    # -----------------------------------------------------------------------
    # Cell 2: Install dependencies
    # -----------------------------------------------------------------------
    {
        "type": "code",
        "source": """\
# Install all dependencies
# Using --quiet to keep output clean
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q onnx onnxruntime wandb folium matplotlib numpy scipy tqdm
!pip install -q fastapi uvicorn pydantic gradio

# Verify GPU is available
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
"""
    },

    # -----------------------------------------------------------------------
    # Cell 3: Clone repo and add to path
    # -----------------------------------------------------------------------
    {
        "type": "code",
        "source": """\
import sys
import os

# Clone the repo (replace with your actual repo URL after pushing)
REPO_URL = "https://github.com/YOUR_USERNAME/navdrift0.git"
REPO_DIR = "/content/navdrift0"

if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
else:
    print("Repo already exists, pulling latest...")
    !cd {REPO_DIR} && git pull

# Add to Python path
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
print(f"Repo at: {REPO_DIR}")
print("Python path updated.")
"""
    },

    # -----------------------------------------------------------------------
    # Cell 4: Download IO-VNBD dataset
    # -----------------------------------------------------------------------
    {
        "type": "code",
        "source": """\
import os
from pathlib import Path

DATA_DIR = Path('/content/drive/MyDrive/NAVDRIFT0/data')
IOVNBD_DIR = DATA_DIR / 'IO-VNBD'

if not IOVNBD_DIR.exists():
    print("Cloning IO-VNBD dataset from GitHub...")
    !git clone https://github.com/onyekpeu/IO-VNBD.git {IOVNBD_DIR}
    print("Dataset downloaded.")
else:
    print(f"Dataset already at {IOVNBD_DIR}")

# List what we have
!ls {IOVNBD_DIR}
"""
    },

    # -----------------------------------------------------------------------
    # Cell 5: Parse and preprocess
    # -----------------------------------------------------------------------
    {
        "type": "code",
        "source": """\
import numpy as np
from pathlib import Path
from navdrift0.data.loader import (
    IOVNBDParser, synchronize_sequence, compute_norm_stats, NormStats
)

IOVNBD_DIR    = Path('/content/drive/MyDrive/NAVDRIFT0/data/IO-VNBD')
CKPT_DIR      = Path('/content/drive/MyDrive/NAVDRIFT0/checkpoints')
NORM_STATS_PT = CKPT_DIR / 'norm_stats.npz'

parser  = IOVNBDParser()
raw_seqs = parser.parse_dataset(IOVNBD_DIR)
print(f"Loaded {len(raw_seqs)} raw sequences")

synced = []
for seq in raw_seqs:
    s = synchronize_sequence(seq)
    if s is not None:
        synced.append(s)
print(f"Synchronized sequences: {len(synced)}")

if NORM_STATS_PT.exists():
    print("Loading existing norm stats...")
    stats = NormStats.load(NORM_STATS_PT)
else:
    print("Computing norm stats...")
    stats = compute_norm_stats(synced)
    stats.save(NORM_STATS_PT)
    print(f"Norm stats saved to {NORM_STATS_PT}")

# Sanity check one sequence
s = synced[0]
print(f"\\nSample sequence shapes:")
print(f"  IMU:  {s['imu'].shape}")
print(f"  Odom: {s['odom'].shape}")
print(f"  GPS:  {s['gps'].shape}")
print(f"  GPS XY: {s['gps_xy'].shape}")
"""
    },

    # -----------------------------------------------------------------------
    # Cell 6: Train DRIFT-Former
    # -----------------------------------------------------------------------
    {
        "type": "code",
        "source": """\
import os
import torch
import wandb
from pathlib import Path
from navdrift0.data.loader import build_dataloaders, compute_norm_stats, NormStats
from navdrift0.models.drift_former import DRIFTFormer, count_parameters
from navdrift0.training.train_drift_former import (
    train_one_epoch, evaluate, WarmupCosineScheduler
)
import torch.optim as optim

# Config
CKPT_DIR    = Path('/content/drive/MyDrive/NAVDRIFT0/checkpoints')
CKPT_PATH   = CKPT_DIR / 'best_drift_former.pt'
NORM_STATS  = NormStats.load(CKPT_DIR / 'norm_stats.npz')
WANDB_KEY   = ""  # Set your wandb API key here (optional, free tier)
EPOCHS      = 100
BATCH_SIZE  = 64
LR          = 3e-4
WINDOW      = 200
STRIDE      = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Build DataLoaders
train_loader, val_loader, test_loader = build_dataloaders(
    synced, NORM_STATS,
    window=WINDOW, stride=STRIDE,
    batch_size=BATCH_SIZE, num_workers=2,
)
print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} | Test: {len(test_loader)}")

# Build Model
model = DRIFTFormer(
    input_dim=8, d_model=256, n_layers=4, n_heads=8,
    ffn_mult=4, dropout=0.1, max_seq_len=WINDOW + 32,
).to(device)
print(f"DRIFT-Former parameters: {count_parameters(model):,}")

# Load checkpoint if exists (resume training)
start_epoch = 1
best_ate = float('inf')
if CKPT_PATH.exists():
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    start_epoch = ckpt.get('epoch', 0) + 1
    best_ate    = ckpt.get('val_ate_m', float('inf'))
    print(f"Resumed from epoch {start_epoch - 1}, best ATE={best_ate:.3f}m")

# Optimizer + Scheduler
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
total_steps  = EPOCHS * len(train_loader)
warmup_steps = int(0.05 * total_steps)
scheduler    = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

# W&B (optional)
if WANDB_KEY:
    wandb.login(key=WANDB_KEY)
    wandb.init(project="navdrift0", name="drift-former",
               config={"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR})

# Training loop
for epoch in range(start_epoch, EPOCHS + 1):
    train_m = train_one_epoch(model, train_loader, optimizer, scheduler,
                               device, clip_grad=1.0, epoch=epoch)
    val_m   = evaluate(model, val_loader, device)

    ate = val_m.get('val/ate_m', float('nan'))
    print(f"Epoch {epoch:03d} | train_loss={train_m['train/loss']:.4f} | "
          f"val_ATE={ate:.3f}m | lr={train_m['train/lr']:.2e}")

    if WANDB_KEY:
        wandb.log({**train_m, **val_m, "epoch": epoch})

    # Save best + epoch checkpoint to Drive (safe against disconnects)
    if not __import__('math').isnan(ate) and ate < best_ate:
        best_ate = ate
        torch.save({
            'epoch':      epoch,
            'state_dict': model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'val_ate_m':  best_ate,
        }, CKPT_PATH)
        print(f"  Saved best: ATE={best_ate:.3f}m")

    # Always save epoch checkpoint (overwrite) for disconnect recovery
    torch.save({
        'epoch':      epoch,
        'state_dict': model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'val_ate_m':  ate,
    }, CKPT_DIR / 'latest_drift_former.pt')

print(f"\\nTraining complete. Best validation ATE: {best_ate:.3f}m")
if WANDB_KEY:
    wandb.finish()
"""
    },

    # -----------------------------------------------------------------------
    # Cell 7: Train NavIC VAE
    # -----------------------------------------------------------------------
    {
        "type": "code",
        "source": """\
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from navdrift0.data.loader import NavICVAEDataset
from navdrift0.models.navic_vae import NavICMotionPriorVAE, vae_loss
from navdrift0.training.train_navic_vae import train_one_epoch as vae_train_epoch
from navdrift0.training.train_navic_vae import evaluate_vae

CKPT_DIR  = Path('/content/drive/MyDrive/NAVDRIFT0/checkpoints')
VAE_CKPT  = CKPT_DIR / 'best_navic_vae.pt'
EPOCHS    = 50
BATCH_SIZE = 64
LR        = 3e-4
BETA      = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

full_ds = NavICVAEDataset(synced, imu_hz=100.0, window_s=60.0)
n_val   = max(1, int(0.15 * len(full_ds)))
n_train = len(full_ds) - n_val
train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                 generator=torch.Generator().manual_seed(42))
train_loader_vae = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=2, drop_last=True)
val_loader_vae   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=2)
print(f"VAE train: {n_train} | val: {n_val}")

model_vae = NavICMotionPriorVAE(
    input_dim=4, hidden_dim=128, n_layers=2, latent_dim=32, output_len=60
).to(device)

optimizer_vae = optim.AdamW(model_vae.parameters(), lr=LR, weight_decay=1e-4)
scheduler_vae = optim.lr_scheduler.CosineAnnealingLR(optimizer_vae,
                                                       T_max=EPOCHS, eta_min=1e-6)
best_val = float('inf')
start_epoch = 1

if VAE_CKPT.exists():
    vckpt = torch.load(VAE_CKPT, map_location=device)
    model_vae.load_state_dict(vckpt['state_dict'])
    start_epoch = vckpt.get('epoch', 0) + 1
    best_val    = vckpt.get('val_loss', float('inf'))
    print(f"Resumed VAE from epoch {start_epoch - 1}")

for epoch in range(start_epoch, EPOCHS + 1):
    tm = vae_train_epoch(model_vae, train_loader_vae, optimizer_vae, device, BETA)
    vm = evaluate_vae(model_vae, val_loader_vae, device, BETA)
    scheduler_vae.step()

    print(f"VAE Epoch {epoch:02d} | train={tm['train/total']:.4f} "
          f"recon={tm['train/recon']:.4f} kl={tm['train/kl']:.4f} | "
          f"val={vm['val/total']:.4f}")

    if vm['val/total'] < best_val:
        best_val = vm['val/total']
        torch.save({'epoch': epoch, 'state_dict': model_vae.state_dict(),
                    'val_loss': best_val}, VAE_CKPT)
        print(f"  VAE best saved: val_loss={best_val:.4f}")

    torch.save({'epoch': epoch, 'state_dict': model_vae.state_dict(),
                'val_loss': vm['val/total']},
               CKPT_DIR / 'latest_navic_vae.pt')

print(f"\\nVAE training complete. Best val loss: {best_val:.4f}")
"""
    },

    # -----------------------------------------------------------------------
    # Cell 8: ONNX export + quantization
    # -----------------------------------------------------------------------
    {
        "type": "code",
        "source": """\
import torch
from pathlib import Path
from navdrift0.models.drift_former import DRIFTFormer
from navdrift0.inference.export_onnx import export_to_onnx, quantize_dynamic, benchmark_latency, verify_onnx

CKPT_DIR  = Path('/content/drive/MyDrive/NAVDRIFT0/checkpoints')
ONNX_DIR  = Path('/content/drive/MyDrive/NAVDRIFT0/onnx')
ONNX_DIR.mkdir(exist_ok=True)

# Load best checkpoint
ckpt = torch.load(CKPT_DIR / 'best_drift_former.pt', map_location='cpu')
model = DRIFTFormer(
    input_dim=8, d_model=256, n_layers=4, n_heads=8,
    ffn_mult=4, dropout=0.0, max_seq_len=232,
)
model.load_state_dict(ckpt['state_dict'])
model.eval()

print(f"Loaded checkpoint: epoch {ckpt['epoch']}, val_ATE={ckpt['val_ate_m']:.3f}m")

# Export FP32 ONNX
onnx_fp32 = ONNX_DIR / 'drift_former_fp32.onnx'
export_to_onnx(model, onnx_fp32, window=200)
verify_onnx(model, onnx_fp32, window=200)

# Export INT8 quantized
model_q = quantize_dynamic(model)
onnx_int8 = ONNX_DIR / 'drift_former_int8.onnx'
export_to_onnx(model_q, onnx_int8, window=200)

# Benchmark
print("\\nBenchmarking FP32 ONNX (CPU):")
m, s, p95 = benchmark_latency(onnx_fp32)
print(f"  mean={m:.2f}ms | std={s:.2f}ms | p95={p95:.2f}ms")

print("\\nBenchmarking INT8 ONNX (CPU):")
m, s, p95 = benchmark_latency(onnx_int8)
print(f"  mean={m:.2f}ms | std={s:.2f}ms | p95={p95:.2f}ms")
if m < 10.0:
    print("  Target <10ms: MET")
else:
    print("  Target <10ms: NOT MET — consider model pruning")
"""
    },

    # -----------------------------------------------------------------------
    # Cell 9: Evaluate baselines + NAVDRIFT-0
    # -----------------------------------------------------------------------
    {
        "type": "code",
        "source": """\
import numpy as np
from pathlib import Path
from navdrift0.data.loader import NormStats
from navdrift0.eval.metrics import (
    compute_ate, compute_rte, compute_drift_rate,
    baseline_imu_integration, ConstantVelocityEKF,
    full_evaluation_report,
)
from navdrift0.inference.runtime import NavDriftRuntime
from navdrift0.data.loader import (
    simulate_gnss_outages, compute_se2_poses, compute_pose_deltas
)

CKPT_DIR = Path('/content/drive/MyDrive/NAVDRIFT0/checkpoints')
ONNX_DIR = Path('/content/drive/MyDrive/NAVDRIFT0/onnx')
NORM_STATS = NormStats.load(CKPT_DIR / 'norm_stats.npz')

# Use last 10% as test set
n_test = max(1, int(0.1 * len(synced)))
test_seqs = synced[-n_test:]

print(f"Evaluating on {n_test} test sequences...")
print("-" * 60)

# Load runtime
runtime = NavDriftRuntime(
    onnx_path       = str(ONNX_DIR / 'drift_former_int8.onnx'),
    norm_stats_path = str(CKPT_DIR / 'norm_stats.npz'),
    window=200, imu_hz=100.0,
)

results = {
    'navdrift0':  [],
    'raw_imu':    [],
    'ekf_cv':     [],
}

rng = __import__('random').Random(99)
ekf = ConstantVelocityEKF(dt=0.01)

for seq in test_seqs:
    imu     = seq['imu']
    gps_xy  = seq['gps_xy']
    gps_hdg = seq['gps_heading']
    odom    = seq['odom']
    N       = len(imu)

    poses = compute_se2_poses(gps_xy, gps_hdg)
    mask  = simulate_gnss_outages(N, rng=rng)

    # Outage window
    outage_steps = np.where(~mask)[0]
    if len(outage_steps) == 0:
        continue
    outage_start = int(outage_steps[0])
    outage_end   = int(outage_steps[-1]) + 1
    outage_dur_s = (outage_end - outage_start) / 100.0

    gt = poses[outage_start:outage_end]

    # NAVDRIFT-0
    runtime.set_initial_gnss_fix(0.0, 0.0, 0.0)
    runtime.current_pose = poses[outage_start].copy()
    runtime.notify_gnss_lost()
    nav_poses = [runtime.current_pose.copy()]
    for i in range(outage_start, outage_end):
        r = runtime.ingest(
            accel_xyz   = imu[i, :3],
            gyro_xyz    = imu[i, 3:],
            speed       = odom[i, 0],
            steer_angle = odom[i, 1],
        )
        nav_poses.append(np.array([r['pose_x'], r['pose_y'], r['heading_rad']]))
    nav_arr = np.array(nav_poses[1:], dtype=np.float32)

    # Raw IMU baseline
    raw_arr = baseline_imu_integration(imu[outage_start:outage_end],
                                        poses[outage_start])[1:]

    # EKF baseline
    ekf_arr = ekf.run_sequence(imu[outage_start:outage_end],
                                poses[outage_start:outage_end],
                                mask[outage_start:outage_end],
                                poses[outage_start])

    for name, arr in [('navdrift0', nav_arr), ('raw_imu', raw_arr), ('ekf_cv', ekf_arr)]:
        if len(arr) > 0 and len(gt) > 0:
            n = min(len(arr), len(gt))
            report = full_evaluation_report(gt[:n], arr[:n],
                                             outage_duration_s=outage_dur_s)
            results[name].append(report)

print("\\nResults on test set (GNSS-dark periods only):")
print(f"{'Method':<20} {'ATE (m)':>10} {'RTE (%)':>10} {'Drift (m/s)':>12}")
print("-" * 55)
for name, reps in results.items():
    if reps:
        ate  = np.mean([r['ate_m'] for r in reps])
        rte  = np.nanmean([r['rte_pct'] for r in reps])
        drift = np.mean([r['drift_m_per_s'] for r in reps])
        print(f"{name:<20} {ate:>10.3f} {rte:>10.2f} {drift:>12.4f}")
"""
    },

    # -----------------------------------------------------------------------
    # Cell 10: Visualization
    # -----------------------------------------------------------------------
    {
        "type": "code",
        "source": """\
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, HTML
import folium

# Use the first test sequence for visualization
seq = test_seqs[0]
N   = len(seq['imu'])
from navdrift0.data.loader import simulate_gnss_outages, compute_se2_poses
poses = compute_se2_poses(seq['gps_xy'], seq['gps_heading'])
mask  = simulate_gnss_outages(N, rng=__import__('random').Random(77))

# Pick outage window
outage_steps = np.where(~mask)[0]
os = int(outage_steps[0]); oe = int(outage_steps[-1]) + 1
gt    = poses[os:oe]

# Run NAVDRIFT-0 on full sequence
runtime.set_initial_gnss_fix(0.0, 0.0, 0.0)
runtime.current_pose = poses[os].copy()
runtime.notify_gnss_lost()
nav_poses = [poses[os].copy()]
for i in range(os, oe):
    r = runtime.ingest(seq['imu'][i,:3], seq['imu'][i,3:],
                        seq['odom'][i,0], seq['odom'][i,1])
    nav_poses.append([r['pose_x'], r['pose_y'], r['heading_rad']])
nav_arr = np.array(nav_poses[1:], dtype=np.float32)

# Raw DR
raw_arr = baseline_imu_integration(seq['imu'][os:oe], poses[os])[1:]
n = min(len(gt), len(nav_arr), len(raw_arr))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.plot(gt[:n,0], gt[:n,1], 'g-', lw=2, label='Ground Truth GPS')
ax.plot(raw_arr[:n,0], raw_arr[:n,1], 'r--', lw=1.5, alpha=0.7, label='Raw IMU DR')
ax.plot(nav_arr[:n,0], nav_arr[:n,1], 'b-', lw=2, label='NAVDRIFT-0')
ax.scatter([gt[0,0]], [gt[0,1]], c='orange', s=80, zorder=5, label='Outage start')
ax.scatter([gt[-1,0]], [gt[-1,1]], c='purple', s=80, zorder=5, label='Reacquisition')
ax.set_xlabel('X (metres)'); ax.set_ylabel('Y (metres)')
ax.set_title('Trajectory Comparison (GNSS-dark period)')
ax.legend(); ax.grid(True, alpha=0.3)

ax2 = axes[1]
errs_raw = np.linalg.norm(gt[:n,:2] - raw_arr[:n,:2], axis=-1)
errs_nav = np.linalg.norm(gt[:n,:2] - nav_arr[:n,:2], axis=-1)
t = np.arange(n) / 100.0
ax2.plot(t, errs_raw, 'r--', lw=1.5, label=f'Raw DR (ATE={errs_raw.mean():.2f}m)')
ax2.plot(t, errs_nav, 'b-',  lw=2,   label=f'NAVDRIFT-0 (ATE={errs_nav.mean():.2f}m)')
ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Position error (m)')
ax2.set_title('Position Error During GNSS Outage')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/NAVDRIFT0/navdrift0_results.png', dpi=150)
plt.show()
print("Plot saved to Drive")
"""
    },
]

if __name__ == "__main__":
    print(f"NAVDRIFT-0 notebook source: {len(NOTEBOOK_CELLS)} cells")
    print("Run: python notebooks/build_notebook.py to generate the .ipynb")
