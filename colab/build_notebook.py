"""
colab/build_notebook.py — generates NAVDRIFT_IOVNBD_SETUP.ipynb from a
plain list of (markdown|code, source) cells.

Kept as a build script (rather than hand-editing raw notebook JSON) so
the notebook's cells stay easy to review and modify as plain Python/
Markdown strings. Run:

    python colab/build_notebook.py

to regenerate colab/NAVDRIFT_IOVNBD_SETUP.ipynb.
"""
import json
from pathlib import Path

def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}

CELLS = []

CELLS.append(md("""# NAVDRIFT-0 — IO-VNBD Training on Colab A100

ISRO SIH 2026 PS #26168.

This notebook trains the NAVDRIFT speed-estimation LSTM on the real,
official IO-VNBD dataset (https://github.com/onyekpeu/IO-VNBD), using a
Colab GPU runtime (A100 recommended).

**What this notebook does NOT do:** it does not touch
`frontend/mobile.html`, `frontend/desktop.html`, or any live/deployed
NAVDRIFT model. Everything here is OFFLINE training and evaluation. The
resulting checkpoint/ONNX file is an artifact for you to bring back to
the repository and integrate later, as an explicit separate step — this
notebook does not do that integration.

**Before running:** open Runtime -> Change runtime type -> select a GPU
(A100 if available on your plan). Then run cells in order, top to
bottom. Cell 1 checks the GPU; every training cell below refuses to run
an expensive job on CPU and tells you why instead.
"""))

CELLS.append(md("## CELL 1 — Check GPU"))
CELLS.append(code(r"""import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout or
      "nvidia-smi not found / no GPU attached to this runtime.")

import torch
cuda_ok = torch.cuda.is_available()
print(f"torch.cuda.is_available() = {cuda_ok}")
if cuda_ok:
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("\n" + "="*78)
    print("  NO GPU DETECTED. Go to Runtime -> Change runtime type -> GPU, then")
    print("  Runtime -> Restart session, and re-run this notebook from Cell 1.")
    print("  Training cells below will refuse to run a real job without a GPU.")
    print("="*78)
"""))

CELLS.append(md("## CELL 2 — Install dependencies"))
CELLS.append(code(r"""!pip install -q onnx onnxruntime matplotlib numpy
# torch is preinstalled on Colab GPU runtimes with CUDA support already.
import torch
print("torch:", torch.__version__, " cuda build:", torch.version.cuda)
"""))

CELLS.append(md("""## CELL 3 — Mount Google Drive (optional)

Only needed if you're keeping the IO-VNBD ZIPs and/or the NAVDRIFT repo
on Google Drive. If you'd rather upload files directly to this Colab
session (Cell 4, option B), you can skip this cell.
"""))
CELLS.append(code(r"""MOUNT_DRIVE = True  # set False to skip

if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Drive mounted at /content/drive")
else:
    print("Skipping Drive mount.")
"""))

CELLS.append(md("""## CELL 4 — Locate the repository and the two IO-VNBD ZIP files

Two things are needed in this Colab session:

1. **The NAVDRIFT repository** (for `data/iovnbd.py`,
   `training/train_iovnbd_speed_lstm.py`, etc). Either clone it, or
   upload/copy it from Drive.
2. **The two official IO-VNBD ZIP files** — `Synchronised V abd S
   datasets.zip` and `Unsynchronised V and S Dataset.zip` — from
   https://github.com/onyekpeu/IO-VNBD. This notebook does NOT assume
   they already exist in this session; set exactly one of the two
   options below.
"""))
CELLS.append(code(r"""import os, shutil
from pathlib import Path

# ---- A) Repository location ----
# Option A1: repo already on Drive
REPO_DIR_ON_DRIVE = "/content/drive/MyDrive/navdrift0-main"  # change if different
# Option A2: upload the repo as a zip via the Colab file browser, then set:
REPO_ZIP_UPLOAD = None  # e.g. "/content/navdrift0-main.zip"

if REPO_ZIP_UPLOAD and Path(REPO_ZIP_UPLOAD).exists():
    import zipfile
    with zipfile.ZipFile(REPO_ZIP_UPLOAD) as z:
        z.extractall("/content/repo_extracted")
    REPO_DIR = "/content/repo_extracted/navdrift0-main"
elif Path(REPO_DIR_ON_DRIVE).exists():
    REPO_DIR = REPO_DIR_ON_DRIVE
else:
    raise FileNotFoundError(
        "Repository not found. Either mount Drive and set REPO_DIR_ON_DRIVE "
        "to the correct path, or upload the repo as a zip via the Colab file "
        "browser (left sidebar -> Files -> upload) and set REPO_ZIP_UPLOAD.")

print("Using repository at:", REPO_DIR)
os.chdir(REPO_DIR)
import sys
sys.path.insert(0, REPO_DIR)

# ---- B) IO-VNBD ZIP files ----
# Option B1: Google Drive path (already downloaded from the official repo)
SYNC_ZIP_DRIVE = "/content/drive/MyDrive/IO-VNBD/Synchronised V abd S datasets.zip"
UNSYNC_ZIP_DRIVE = "/content/drive/MyDrive/IO-VNBD/Unsynchronised V and S Dataset.zip"
# Option B2: uploaded directly into this Colab session (left sidebar -> Files -> upload)
SYNC_ZIP_UPLOAD = "/content/Synchronised V abd S datasets.zip"
UNSYNC_ZIP_UPLOAD = "/content/Unsynchronised V and S Dataset.zip"

def resolve(drive_path, upload_path, label):
    if Path(drive_path).exists():
        print(f"{label}: using Drive copy at {drive_path}")
        return drive_path
    if Path(upload_path).exists():
        print(f"{label}: using uploaded copy at {upload_path}")
        return upload_path
    raise FileNotFoundError(
        f"{label} not found at either:\n  Drive:  {drive_path}\n  Upload: {upload_path}\n"
        f"Download the official IO-VNBD ZIPs from https://github.com/onyekpeu/IO-VNBD "
        f"and either place them on Drive at the path above, or upload them to this "
        f"Colab session's /content/ folder via the Files sidebar, then re-run this cell.")

SYNC_ZIP = resolve(SYNC_ZIP_DRIVE, SYNC_ZIP_UPLOAD, "Synchronised IO-VNBD zip")
print("SYNC_ZIP =", SYNC_ZIP)
"""))

CELLS.append(md("""## CELL 5 — Inspect ZIPs

Confirms the ZIP is readable and reports real file counts before doing
anything else (this mirrors the inspection already done and documented
in `results/iovnbd_inspection_report.md`, re-run here so you can verify
it against your own copy of the files)."""))
CELLS.append(code(r"""import zipfile

for zp in [SYNC_ZIP]:
    with zipfile.ZipFile(zp) as z:
        bad = z.testzip()
        names = [n for n in z.namelist() if not n.endswith('/')]
        csvs = [n for n in names if n.lower().endswith('.csv')]
        print(f"{zp}")
        print(f"  testzip() (None=OK): {bad}")
        print(f"  total files: {len(names)}  csv files: {len(csvs)}")
"""))

CELLS.append(md("""## CELL 6 — Extract only what is required

The real parser (`data/iovnbd.py`) reads directly from the ZIP archive —
it does NOT require full extraction. This cell is a no-op by default;
set `FORCE_EXTRACT_ALL = True` only if you specifically want a flat
on-disk copy (e.g. for manual inspection), understanding this will use
several hundred MB of Colab disk."""))
CELLS.append(code(r"""FORCE_EXTRACT_ALL = False

if FORCE_EXTRACT_ALL:
    import zipfile
    extract_dir = "/content/iovnbd_extracted"
    with zipfile.ZipFile(SYNC_ZIP) as z:
        z.extractall(extract_dir)
    print("Extracted to", extract_dir)
else:
    print("Skipping full extraction — data/iovnbd.py reads directly from the ZIP.")
"""))

CELLS.append(md("""## CELL 7 — Parse real IO-VNBD data

Uses `data/iovnbd.py`'s real S/V-file parser. Reports exactly how many
of the real run pairs parsed successfully — if this is not 72/72 (or
close), stop and investigate before continuing; do not proceed on a
partially-broken parse."""))
CELLS.append(code(r"""from data.iovnbd import load_iovnbd_dataset

sequences, metas = load_iovnbd_dataset(SYNC_ZIP)
n_ok = sum(1 for m in metas if m.n_rows_used > 0)
print(f"Parsed {n_ok}/{len(metas)} real IO-VNBD run pairs successfully.")

total_hours = sum(m.duration_s for m in metas if m.n_rows_used > 0) / 3600.0
print(f"Total usable duration: {total_hours:.1f} hours")

n_mismatch = sum(1 for m in metas if m.row_count_mismatch)
n_resets = sum(m.n_time_resets for m in metas)
print(f"Runs with S/V row-count mismatch (trimmed to shorter): {n_mismatch}")
print(f"Total TIME-SINCE-START(ms) resets detected across all runs: {n_resets}")
"""))

CELLS.append(md("""## CELL 8 — Generate deterministic train/validation/test split

Sequence-level split (never row-level), seeded, stratified by driver
category. Matches `results/iovnbd/split_manifest.json` already committed
to the repo (same seed=42), regenerated here so this notebook is fully
self-contained and reproducible on its own."""))
CELLS.append(code(r"""from data.iovnbd_split import make_split
import json

manifest = make_split(metas, seed=42)
print(f"train={manifest['n_train']}  val={manifest['n_val']}  test={manifest['n_test']}")
print("small categories kept whole in train:", manifest["small_categories_kept_in_train"])

Path("results/iovnbd").mkdir(parents=True, exist_ok=True)
with open("results/iovnbd/split_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
"""))

CELLS.append(md("## CELL 9 — Print dataset statistics"))
CELLS.append(code(r"""from collections import Counter

by_name = {s.name: s for s in sequences}
cat_counts = Counter(m.driver_code for m in metas if m.n_rows_used > 0)
print("Runs per driver category:")
for cat, n in sorted(cat_counts.items()):
    print(f"  {cat:24s} {n:3d} runs")

durations = [m.duration_s for m in metas if m.n_rows_used > 0]
import numpy as np
print(f"\nRun duration (s): min={min(durations):.1f} median={np.median(durations):.1f} "
      f"max={max(durations):.1f} total={sum(durations)/3600:.1f}h")
"""))

CELLS.append(md("""## CELL 10 — One-batch smoke test

Runs a single forward+backward pass on ONE real batch, on whatever
device is available. This is a correctness check only — it proves the
training code runs against real data before committing GPU time to a
full run. It does NOT report this as a trained model."""))
CELLS.append(code(r"""!python -m training.train_iovnbd_speed_lstm \
    --zip "{SYNC_ZIP}" \
    --split_manifest results/iovnbd/split_manifest.json \
    --epochs 1 --max_train_runs 2 --max_val_runs 1 --max_test_runs 1 \
    --seq_len 50 --stride 25 --batch_size 128 \
    --out_dir /content/smoke_test --smoke_test
"""))

CELLS.append(md("""## CELL 11 — Train model (the real A100 run)

This is the actual training run. Refuses to proceed if no GPU is
detected (see the WARNING block in `train_iovnbd_speed_lstm.py`) rather
than silently training for hours on CPU. Adjust `--epochs` /
`--batch_size` as needed; defaults are reasonable starting points, not
guarantees of a particular accuracy."""))
CELLS.append(code(r"""!python -m training.train_iovnbd_speed_lstm \
    --zip "{SYNC_ZIP}" \
    --split_manifest results/iovnbd/split_manifest.json \
    --epochs 40 --seq_len 50 --stride 10 --batch_size 256 \
    --hidden_size 64 --num_layers 2 --patience 6 \
    --out_dir checkpoints/iovnbd_speed_lstm
"""))

CELLS.append(md("""## CELL 12 — Evaluate held-out sequences

Runs the offline classical-DR benchmark on the real, held-out test
split. (The AI-assisted comparison row is added automatically once a
real checkpoint exists at the path below — see Part E of the engineering
plan; this cell reports the classical baseline honestly either way.)"""))
CELLS.append(code(r"""!python -m eval.iovnbd_benchmark \
    --zip "{SYNC_ZIP}" \
    --split_manifest results/iovnbd/split_manifest.json \
    --out_dir results/iovnbd
"""))

CELLS.append(md("## CELL 13 — Trajectory / error plots (already generated by Cell 12)"))
CELLS.append(code(r"""import glob
from IPython.display import Image, display

for png in sorted(glob.glob("results/iovnbd/plots/*.png"))[:3]:
    print(png)
    display(Image(png))
"""))

CELLS.append(md("## CELL 14 — Export ONNX"))
CELLS.append(code(r"""!python -m training.export_iovnbd_lstm_onnx \
    --checkpoint checkpoints/iovnbd_speed_lstm/navdrift_lstm_best.pt \
    --out_dir checkpoints/iovnbd_speed_lstm
"""))

CELLS.append(md("""## CELL 15 — Validate ONNX against PyTorch

Already performed inside Cell 14's export script (max abs diff printed
there). This cell just re-displays the saved report."""))
CELLS.append(code(r"""import json
with open("checkpoints/iovnbd_speed_lstm/onnx_export_report.json") as f:
    print(json.dumps(json.load(f), indent=2))
"""))

CELLS.append(md("""## CELL 16 — Generate final benchmark JSON

Combines the training report, ONNX export report, and classical-DR
benchmark into one summary file for bringing back to the main repo."""))
CELLS.append(code(r"""import json
from pathlib import Path

final = {}
for name, path in [
    ("training_report", "checkpoints/iovnbd_speed_lstm/training_report.json"),
    ("onnx_export_report", "checkpoints/iovnbd_speed_lstm/onnx_export_report.json"),
    ("classical_dr_benchmark", "results/iovnbd/iovnbd_classical_dr_benchmark.json"),
    ("split_manifest", "results/iovnbd/split_manifest.json"),
]:
    p = Path(path)
    final[name] = json.load(open(p)) if p.exists() else f"MISSING: {path}"

out_path = "results/iovnbd/colab_final_summary.json"
with open(out_path, "w") as f:
    json.dump(final, f, indent=2)
print(f"Written: {out_path}")
print("\nBring these files back to the repository (see colab/README.md):")
for f in ["checkpoints/iovnbd_speed_lstm/navdrift_lstm_best.pt",
          "checkpoints/iovnbd_speed_lstm/navdrift_lstm.onnx",
          "checkpoints/iovnbd_speed_lstm/navdrift_lstm_meta.json",
          "checkpoints/iovnbd_speed_lstm/training_report.json",
          "checkpoints/iovnbd_speed_lstm/onnx_export_report.json",
          "results/iovnbd/iovnbd_classical_dr_benchmark.json",
          "results/iovnbd/split_manifest.json",
          "results/iovnbd/colab_final_summary.json",
          "results/iovnbd/plots/"]:
    print(" -", f)
"""))

notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "accelerator": "GPU",
        "colab": {"name": "NAVDRIFT_IOVNBD_SETUP.ipynb", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path(__file__).parent / "NAVDRIFT_IOVNBD_SETUP.ipynb"
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)
print(f"Wrote {out_path}  ({len(CELLS)} cells)")
