"""
NAVDRIFT-0 | ISRO SIH 2026 PS #26168
File 00 — Setup, Drive mount, anti-disconnect, checkpointing
Run this first in every Colab session before any other script.
"""

import os, sys, time, json, threading, subprocess
from pathlib import Path

# ============================================================
# 1. INSTALL DEPENDENCIES
# ============================================================
def install_deps():
    pkgs = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "h5py scipy pandas numpy matplotlib tqdm",
        "onnx onnxruntime onnxruntime-gpu",
        "onnxmltools onnxconverter-common",
        "georinex",          # NavIC RINEX parsing
        "pymap3d",           # geodetic conversions
        "scikit-learn",
    ]
    for pkg in pkgs:
        print(f"Installing {pkg.split()[0]}...")
        subprocess.run(f"pip install -q {pkg}", shell=True, check=False)
    print("[DEPS] All packages installed.")

# ============================================================
# 2. GOOGLE DRIVE MOUNT + DIRECTORY STRUCTURE
# ============================================================
DRIVE_ROOT = Path("/content/drive/MyDrive/NAVDRIFT0")

PATHS = {
    "root":        DRIVE_ROOT,
    "datasets":    DRIVE_ROOT / "datasets",
    "kaist":       DRIVE_ROOT / "datasets" / "kaist",
    "euroc":       DRIVE_ROOT / "datasets" / "euroc",
    "tum_vi":      DRIVE_ROOT / "datasets" / "tum_vi",
    "nclt":        DRIVE_ROOT / "datasets" / "nclt",
    "navic":       DRIVE_ROOT / "datasets" / "navic_rinex",
    "processed":   DRIVE_ROOT / "processed",
    "checkpoints": DRIVE_ROOT / "checkpoints",
    "models":      DRIVE_ROOT / "models",
    "results":     DRIVE_ROOT / "results",
    "logs":        DRIVE_ROOT / "logs",
}

def mount_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        print("[DRIVE] Mounted at /content/drive")
    except ImportError:
        print("[DRIVE] Not in Colab — skipping mount, using local paths.")

def create_dirs():
    for key, path in PATHS.items():
        path.mkdir(parents=True, exist_ok=True)
    print(f"[DIRS] All directories ready under {DRIVE_ROOT}")

# ============================================================
# 3. ANTI-DISCONNECT — TWO LAYERS
# ============================================================
_keepalive_stop = threading.Event()

def start_keepalive(interval=90):
    """
    Layer 1: background thread prints every 90 s.
    Layer 2: inject JS click on Colab's connect button.
    Both together prevent the idle-timeout disconnect.
    """
    _keepalive_stop.clear()

    def _loop():
        start = time.time()
        tick = 0
        while not _keepalive_stop.is_set():
            _keepalive_stop.wait(interval)
            if _keepalive_stop.is_set():
                break
            tick += 1
            elapsed = (time.time() - start) / 60
            print(f"\r[ALIVE] {elapsed:.1f} min | tick {tick}", end="", flush=True)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()

    # JS layer — works in Colab notebook cells
    try:
        from IPython.display import Javascript, display
        display(Javascript("""
            var ka = setInterval(function() {
                var btns = document.querySelectorAll('colab-connect-button');
                btns.forEach(function(b){ b.click(); });
                console.log('[NAVDRIFT-ALIVE] keepalive tick');
            }, 55000);
            console.log('[NAVDRIFT-ALIVE] keepalive registered, id=' + ka);
        """))
    except Exception:
        pass  # not in a notebook cell

    print("[ALIVE] Keepalive started (background thread + JS).")

def stop_keepalive():
    _keepalive_stop.set()
    print("[ALIVE] Keepalive stopped.")

# ============================================================
# 4. CHECKPOINT HELPERS
# ============================================================
import torch

def save_checkpoint(model, optimizer, scheduler, epoch, loss_history, name, extra=None):
    ckpt_dir = PATHS["checkpoints"] / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optimizer_state":optimizer.state_dict(),
        "scheduler_state":scheduler.state_dict() if scheduler else None,
        "loss_history":   loss_history,
    }
    if extra:
        state.update(extra)

    epoch_path  = ckpt_dir / f"epoch_{epoch:04d}.pt"
    latest_path = ckpt_dir / "latest.pt"
    torch.save(state, epoch_path)
    torch.save(state, latest_path)
    print(f"[CKPT] {name} | epoch {epoch} saved -> {epoch_path.name}")
    return latest_path

def load_checkpoint(model, optimizer, scheduler, name, device="cpu"):
    latest = PATHS["checkpoints"] / name / "latest.pt"
    if not latest.exists():
        print(f"[CKPT] No checkpoint for '{name}' — starting fresh.")
        return 0, []

    state = torch.load(latest, map_location=device)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    if scheduler and state.get("scheduler_state"):
        scheduler.load_state_dict(state["scheduler_state"])

    epoch        = state["epoch"]
    loss_history = state.get("loss_history", [])
    print(f"[CKPT] Resumed '{name}' from epoch {epoch}  ({latest})")
    return epoch + 1, loss_history

def save_results(results, name):
    path = PATHS["results"] / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # make sure numpy scalars are serialisable
    def _fix(obj):
        if hasattr(obj, "item"):
            return obj.item()
        return obj
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_fix)
    print(f"[RESULTS] Saved -> {path}")

# ============================================================
# 5. DEVICE + GPU INFO
# ============================================================
def get_device():
    if torch.cuda.is_available():
        dev  = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU]  {name}  |  {mem:.1f} GB VRAM")
        torch.backends.cudnn.benchmark = True
    else:
        dev = torch.device("cpu")
        print("[GPU]  No CUDA — running on CPU (will be slow)")
    return dev

# ============================================================
# 6. GENERIC TRAIN / VAL EPOCH
# ============================================================
def run_epoch(model, loader, optimizer, loss_fn, device, train=True, grad_clip=1.0):
    model.train(train)
    ctx = torch.enable_grad if train else torch.no_grad
    total, n = 0.0, 0
    with ctx():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            if train:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total += loss.item() * x.size(0)
            n     += x.size(0)
    return total / max(n, 1)

def standard_train_loop(
    model, train_loader, val_loader,
    optimizer, scheduler, loss_fn,
    device, epochs, name,
    save_every=5, start_epoch=0, loss_history=None
):
    if loss_history is None:
        loss_history = []
    best_val = float("inf")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        tr  = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        val = run_epoch(model, val_loader,   optimizer, loss_fn, device, train=False)
        if scheduler:
            scheduler.step(val)

        loss_history.append({"epoch": epoch, "train": tr, "val": val})
        dt = time.time() - t0
        print(f"[{name}] Ep {epoch:03d}/{epochs}  "
              f"train={tr:.5f}  val={val:.5f}  {dt:.1f}s")

        if val < best_val:
            best_val = val
            save_checkpoint(model, optimizer, scheduler, epoch, loss_history,
                            name, extra={"best_val": best_val})
        elif epoch % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, loss_history, name)

    save_results({"name": name, "loss_history": loss_history, "best_val": best_val}, name)
    return loss_history

# ============================================================
# 7. ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  NAVDRIFT-0  |  ISRO SIH 2026 PS #26168  |  SETUP")
    print("=" * 60)
    install_deps()
    mount_drive()
    create_dirs()
    device = get_device()
    start_keepalive()
    print("\n[READY] Session is live. Run the next script.")
