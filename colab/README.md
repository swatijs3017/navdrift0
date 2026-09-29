# NAVDRIFT-0 — Colab IO-VNBD Training Package

ISRO SIH 2026 PS #26168.

This is the self-contained package for training the NAVDRIFT speed LSTM
on the real, official IO-VNBD dataset using your Google Colab A100. It
does not require any local GPU. Everything the notebook calls into
already exists, tested, in this repository:

- `data/iovnbd.py` — real S/V-file parser (tested against the full real
  dataset: 72/72 run pairs parse successfully, see
  `results/iovnbd/run_inventory.json`)
- `data/iovnbd_split.py` — deterministic sequence-level train/val/test
  split (already generated once at `results/iovnbd/split_manifest.json`,
  seed 42; the notebook regenerates the same split independently so it
  is self-contained)
- `training/train_iovnbd_speed_lstm.py` — trains `SpeedLSTM`, matching
  `navdrift_engine.py`'s `LSTMSpeedEstimator` ONNX input/output contract
- `training/export_iovnbd_lstm_onnx.py` — exports and validates ONNX
  against the PyTorch checkpoint
- `eval/iovnbd_benchmark.py` — classical dead-reckoning offline
  benchmark on real, held-out test sequences

All five were smoke-tested on real IO-VNBD data in this session on CPU
(tiny subsets, 1 epoch) purely to confirm the code runs correctly end to
end — none of that is a real training result. No LSTM has actually been
trained yet; that is what this Colab notebook is for.

## 1. Which files you need to copy to Colab

The whole repository (or at minimum: `data/`, `training/`, `eval/`,
`colab/`, and `results/iovnbd/split_manifest.json` if you want to skip
regenerating it). The simplest approach: zip the repo and upload it, or
`git clone` it inside Colab if you have it on GitHub/a private remote.
The notebook's Cell 4 supports either a Drive copy or a direct upload.

## 2. Where to put the IO-VNBD ZIPs

Either:

- **Google Drive**: place `Synchronised V abd S datasets.zip` (and
  optionally `Unsynchronised V and S Dataset.zip`, not used by this
  notebook's training path but available for later work) somewhere on
  your Drive, and point Cell 4's `SYNC_ZIP_DRIVE` variable at it after
  mounting Drive in Cell 3.
- **Direct upload**: use the Colab file browser (left sidebar -> Files
  -> upload) to upload the ZIP straight into `/content/`, and Cell 4
  will find it there automatically if the Drive path doesn't exist.

The notebook does NOT assume the ZIPs are already present — Cell 4
fails loudly with instructions if neither location has them, rather than
silently continuing.

## 3. Which notebook to open

`colab/NAVDRIFT_IOVNBD_SETUP.ipynb` — upload it to Colab (File -> Upload
notebook) or open it directly if the repo is cloned into your Drive and
you open it from there.

## 4. Which cell starts the real training

**Cell 11** ("Train model (the real A100 run)"). Cells 1-10 are setup,
inspection, and a code-correctness smoke test — none of them train a
real model. Cell 11 refuses to run if no GPU is detected (checked in
Cell 1) and prints a warning instead of silently training for hours on
CPU.

## 5. Expected outputs

- Console output from Cell 11: per-epoch train MSE loss and validation
  MAE/RMSE, until early stopping or `--epochs` is reached.
- Cell 12: a JSON summary of the classical dead-reckoning benchmark on
  real held-out test sequences (ATE RMSE, mean drift, max drift), plus
  trajectory/error PNGs.
- Cell 14: a PyTorch-vs-ONNX validation report (max absolute
  difference between the two, and measured Colab-CPU-session ONNX
  inference latency — not a mobile-device measurement).
- Cell 16: one combined `colab_final_summary.json`.

## 6. Where the trained `.pt` checkpoint will appear

`checkpoints/iovnbd_speed_lstm/navdrift_lstm_best.pt` — the best
validation-MAE checkpoint (early stopping keeps this, not necessarily
the last epoch). Also `navdrift_lstm_meta.json` alongside it (seq_len,
feature_cols, scaler mean/scale) — the same meta format
`navdrift_engine.py`'s `LSTMSpeedEstimator` already expects.

## 7. Where `navdrift_lstm.onnx` will appear

`checkpoints/iovnbd_speed_lstm/navdrift_lstm.onnx`, produced by Cell 14.

## 8. Where benchmark plots will appear

`results/iovnbd/plots/*.png` — one trajectory + error-vs-time plot per
scored held-out test sequence, produced by Cell 12.

## 9. Bringing results back into the NAVDRIFT repository

Download these files from the Colab session (or save them to the same
Drive path you mounted) and copy them into your local repo at the
identical relative paths:

```
checkpoints/iovnbd_speed_lstm/navdrift_lstm_best.pt
checkpoints/iovnbd_speed_lstm/navdrift_lstm.onnx
checkpoints/iovnbd_speed_lstm/navdrift_lstm_meta.json
checkpoints/iovnbd_speed_lstm/training_report.json
checkpoints/iovnbd_speed_lstm/onnx_export_report.json
results/iovnbd/iovnbd_classical_dr_benchmark.json
results/iovnbd/split_manifest.json      (only if it differs from the repo's)
results/iovnbd/colab_final_summary.json
results/iovnbd/plots/                    (whole folder)
```

**None of this is wired into `frontend/mobile.html` or any live path by
copying these files in.** Integration is a separate, explicit decision
to make afterward, with the real numbers from `training_report.json` and
`iovnbd_classical_dr_benchmark.json` in hand — not before.

## What this package does NOT claim

No training has been run and no `.pt`/`.onnx` checkpoint exists in this
repository as of this package being built. `n_train_sequences`,
`test_mae`, etc. in `training_report.json` will only be real numbers
once you actually run Cell 11 on your A100. Nothing in this session
fabricated a trained model, a benchmark result, or a hardware validation
claim — the classical-DR benchmark in `results/iovnbd/` was run for
real, on real held-out data, without needing a GPU, and is already
committed with real numbers; the LSTM/ONNX artifacts are not.
