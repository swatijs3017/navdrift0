"""
NAVDRIFT0 — Improvements A through E
Run in order in Colab. Each cell is clearly marked.
All cells assume the previous session variables are still alive
(df, clean_idx, FEATURES, TARGET, EXPORT_DIR, etc.)
If runtime was reset, re-run Cells 1-9 from the original notebook first.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 1 — STRIDE=1 (5× more training windows, same BiLSTM)
# Paste ONLY the code inside the triple-quotes into a new Colab cell
# ═══════════════════════════════════════════════════════════════════════════════
IMP1 = """
import numpy as np, time
from sklearn.preprocessing import StandardScaler

SEQ_LEN = 50
STRIDE  = 1          # was 5 — now 5× more windows

df_clean = df.dropna(subset=FEATURES + [TARGET]).copy()
df_clean = df_clean[df_clean[TARGET] >= 0].copy()
clean_idx = df_clean.index.tolist()

X_raw = df_clean[FEATURES].values.astype(np.float32)
y_raw = df_clean[TARGET].values.astype(np.float32)

# Build windows
Xs, ys = [], []
for i in range(0, len(X_raw) - SEQ_LEN, STRIDE):
    Xs.append(X_raw[i:i+SEQ_LEN])
    ys.append(y_raw[i+SEQ_LEN-1])
X_all = np.array(Xs, dtype=np.float32)
y_all = np.array(ys, dtype=np.float32)

n = len(X_all)
n_train = int(n * 0.70)
n_val   = int(n * 0.15)
X_train, y_train = X_all[:n_train],            y_all[:n_train]
X_val,   y_val   = X_all[n_train:n_train+n_val], y_all[n_train:n_train+n_val]
X_test,  y_test  = X_all[n_train+n_val:],       y_all[n_train+n_val:]

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train.reshape(-1,7)).reshape(X_train.shape)
X_val_s   = scaler.transform(X_val.reshape(-1,7)).reshape(X_val.shape)
X_test_s  = scaler.transform(X_test.reshape(-1,7)).reshape(X_test.shape)

print(f"Windows — train: {len(X_train):,}  val: {len(X_val):,}  test: {len(X_test):,}")
print(f"(was ~{14047} total; now {len(X_all):,} — {len(X_all)/14047:.1f}× more)")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 2 — Attention-BiLSTM (new architecture)
# Run AFTER Improvement 1 (uses X_train_s etc. from above)
# ═══════════════════════════════════════════════════════════════════════════════
IMP2 = """
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
import os, json, numpy as np

# ── Attention layer ──────────────────────────────────────────────────────────
class BahdanauAttention(layers.Layer):
    def __init__(self, units=64, **kw):
        super().__init__(**kw)
        self.W = layers.Dense(units)
        self.V = layers.Dense(1)
    def call(self, seq):                          # seq: (B, T, D)
        score = self.V(tf.nn.tanh(self.W(seq)))  # (B, T, 1)
        weights = tf.nn.softmax(score, axis=1)   # (B, T, 1)
        ctx = tf.reduce_sum(weights * seq, axis=1)  # (B, D)
        return ctx, weights

# ── Build model ──────────────────────────────────────────────────────────────
inp = layers.Input(shape=(50, 7), name='imu')
x   = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inp)
x   = layers.Dropout(0.2)(x)
x   = layers.Bidirectional(layers.LSTM(64,  return_sequences=True))(x)
x   = layers.Dropout(0.2)(x)
ctx, attn_w = BahdanauAttention(64, name='attention')(x)
x   = layers.Dense(32, activation='relu')(ctx)
out = layers.Dense(1,  activation='relu', name='speed_mps')(x)

attn_model = Model(inp, out, name='NavdriftAttn')
attn_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.Huber(delta=1.0),
    metrics=['mae'])
attn_model.summary()

# ── Callbacks ────────────────────────────────────────────────────────────────
CKPT = f'{EXPORT_DIR}/attn_best.keras'
cbs = [
    callbacks.ModelCheckpoint(CKPT, save_best_only=True, monitor='val_loss', verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=1),
    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
]

# ── Train ────────────────────────────────────────────────────────────────────
hist = attn_model.fit(
    X_train_s, y_train,
    validation_data=(X_val_s, y_val),
    epochs=60, batch_size=256,
    callbacks=cbs, verbose=1)

# ── Quick eval ───────────────────────────────────────────────────────────────
y_pred_attn = attn_model.predict(X_test_s, batch_size=512).flatten()
mae_attn  = np.mean(np.abs(y_pred_attn - y_test)) * 3.6
rmse_attn = np.sqrt(np.mean((y_pred_attn - y_test)**2)) * 3.6
print(f"\\nAttention-BiLSTM  MAE: {mae_attn:.3f} km/h   RMSE: {rmse_attn:.3f} km/h")

# Save for downstream cells
y_pred = y_pred_attn   # replace global for eval cells
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 3 — Export Attention model to ONNX via PyTorch weight transfer
# Run AFTER Improvement 2
# ═══════════════════════════════════════════════════════════════════════════════
IMP3 = """
import torch, torch.nn as nn, numpy as np, subprocess, sys

# Install deps quietly
subprocess.run([sys.executable,'-m','pip','install','-q','onnxscript','onnxruntime'], check=True)
import onnxruntime as ort

# ── Extract weights from Keras attention model ────────────────────────────────
def get_bilstm_weights(layer):
    W = layer.get_weights()
    # Keras Bidirectional: [fwd_kernel, fwd_recurrent, fwd_bias,
    #                        bwd_kernel, bwd_recurrent, bwd_bias]
    n = len(W) // 2
    return W[:n], W[n:]

def split_gates(kernel, recurrent, bias, units):
    # Keras order: i, f, c, o  (not i,f,g,o like PyTorch)
    # PyTorch order: i, f, g, o — same as Keras (i,f,c,o) but 'g' = 'c'
    k = np.split(kernel,    4, axis=1)  # each (input_dim, units)
    r = np.split(recurrent, 4, axis=1)  # each (units, units)
    b = np.split(bias,      4, axis=0)  # each (units,)
    weight_ih = np.concatenate(k, axis=1).T   # (4*units, input_dim)
    weight_hh = np.concatenate(r, axis=1).T   # (4*units, units)
    bias_ih   = np.concatenate(b)              # (4*units,)
    bias_hh   = np.zeros_like(bias_ih)
    return weight_ih, weight_hh, bias_ih, bias_hh

def load_bilstm(pt_lstm, keras_fwd, keras_bwd, units):
    fwd_k, fwd_r, fwd_b = keras_fwd
    bwd_k, bwd_r, bwd_b = keras_bwd
    wih_f, whh_f, bih_f, bhh_f = split_gates(fwd_k, fwd_r, fwd_b, units)
    wih_b, whh_b, bih_b, bhh_b = split_gates(bwd_k, bwd_r, bwd_b, units)
    with torch.no_grad():
        pt_lstm.weight_ih_l0.copy_(torch.tensor(wih_f))
        pt_lstm.weight_hh_l0.copy_(torch.tensor(whh_f))
        pt_lstm.bias_ih_l0.copy_(torch.tensor(bih_f))
        pt_lstm.bias_hh_l0.copy_(torch.tensor(bhh_f))
        pt_lstm.weight_ih_l0_reverse.copy_(torch.tensor(wih_b))
        pt_lstm.weight_hh_l0_reverse.copy_(torch.tensor(whh_b))
        pt_lstm.bias_ih_l0_reverse.copy_(torch.tensor(bih_b))
        pt_lstm.bias_hh_l0_reverse.copy_(torch.tensor(bhh_b))

# ── PyTorch model with attention ─────────────────────────────────────────────
class NavdriftAttnPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bilstm1 = nn.LSTM(7,   128, batch_first=True, bidirectional=True)
        self.drop1   = nn.Dropout(0.0)   # no dropout at inference
        self.bilstm2 = nn.LSTM(256,  64, batch_first=True, bidirectional=True)
        self.drop2   = nn.Dropout(0.0)
        self.attn_W  = nn.Linear(128, 64)
        self.attn_V  = nn.Linear(64,  1)
        self.fc1     = nn.Linear(128, 32)
        self.fc2     = nn.Linear(32,  1)

    def forward(self, x):                         # x: (B, 50, 7)
        x, _ = self.bilstm1(x)                   # (B,50,256)
        x, _ = self.bilstm2(x)                   # (B,50,128)
        score  = self.attn_V(torch.tanh(self.attn_W(x)))  # (B,50,1)
        w      = torch.softmax(score, dim=1)               # (B,50,1)
        ctx    = (w * x).sum(dim=1)                        # (B,128)
        x = torch.relu(self.fc1(ctx))
        return self.fc2(x)

pt_model = NavdriftAttnPT().eval()

# Load BiLSTM weights
bilstm1 = attn_model.layers[1]  # Bidirectional(LSTM 128)
bilstm2 = attn_model.layers[3]  # Bidirectional(LSTM 64)
fwd1, bwd1 = get_bilstm_weights(bilstm1)
fwd2, bwd2 = get_bilstm_weights(bilstm2)
load_bilstm(pt_model.bilstm1, fwd1, bwd1, 128)
load_bilstm(pt_model.bilstm2, fwd2, bwd2, 64)

# Load attention weights
attn_layer = attn_model.get_layer('attention')
W_k, W_b, V_k, V_b = attn_layer.get_weights()   # W kernel, W bias, V kernel, V bias
with torch.no_grad():
    pt_model.attn_W.weight.copy_(torch.tensor(W_k.T))
    pt_model.attn_W.bias.copy_(torch.tensor(W_b))
    pt_model.attn_V.weight.copy_(torch.tensor(V_k.T))
    pt_model.attn_V.bias.copy_(torch.tensor(V_b))

# Load Dense weights
dense1 = attn_model.get_layer('dense') if 'dense' in [l.name for l in attn_model.layers] else attn_model.layers[-3]
dense2 = attn_model.layers[-2]
for keras_d, pt_d in [(attn_model.get_layer(index=-3), pt_model.fc1),
                       (attn_model.get_layer(index=-2), pt_model.fc1),
                       (attn_model.get_layer(index=-1), pt_model.fc2)]:
    pass  # handled below

# Simpler: grab by position
dense_layers = [l for l in attn_model.layers if 'dense' in l.name and l.name != 'attention']
for dl, pt_l in zip(dense_layers, [pt_model.fc1, pt_model.fc2]):
    k, b = dl.get_weights()
    with torch.no_grad():
        pt_l.weight.copy_(torch.tensor(k.T))
        pt_l.bias.copy_(torch.tensor(b))

# ── Verify ───────────────────────────────────────────────────────────────────
sample = torch.tensor(X_test_s[:32], dtype=torch.float32)
with torch.no_grad():
    pt_out = pt_model(sample).numpy().flatten()
tf_out  = attn_model.predict(X_test_s[:32], verbose=0).flatten()
max_diff = np.max(np.abs(pt_out - tf_out))
print(f"Max diff TF vs PyTorch: {max_diff:.5f} m/s  (target <0.005)")

# ── Export ONNX ──────────────────────────────────────────────────────────────
ONNX_PATH = f'{EXPORT_DIR}/navdrift_attn.onnx'
dummy = torch.zeros(1, 50, 7)
torch.onnx.export(
    pt_model, dummy, ONNX_PATH,
    input_names=['imu_window'],
    output_names=['speed_mps'],
    dynamic_axes={'imu_window': {0: 'batch'}, 'speed_mps': {0: 'batch'}},
    opset_version=17)

# Verify ONNX
sess = ort.InferenceSession(ONNX_PATH)
onnx_out = sess.run(None, {'imu_window': X_test_s[:32]})[0].flatten()
print(f"ONNX max diff vs TF: {np.max(np.abs(onnx_out - tf_out)):.5f} m/s")

import os
print(f"ONNX saved: {ONNX_PATH}  ({os.path.getsize(ONNX_PATH)/1024:.1f} KB)")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 4 — ONNX INT8 Quantization (no retraining needed)
# Run AFTER Improvement 3 (needs navdrift_attn.onnx)
# ═══════════════════════════════════════════════════════════════════════════════
IMP4 = """
import subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','-q','onnxruntime'], check=True)

from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort, numpy as np, os, time

FP32_PATH = f'{EXPORT_DIR}/navdrift_attn.onnx'
INT8_PATH = f'{EXPORT_DIR}/navdrift_attn_int8.onnx'

quantize_dynamic(FP32_PATH, INT8_PATH, weight_type=QuantType.QInt8)

fp32_size = os.path.getsize(FP32_PATH) / 1024
int8_size = os.path.getsize(INT8_PATH) / 1024
print(f"FP32: {fp32_size:.1f} KB")
print(f"INT8: {int8_size:.1f} KB  ({100*(1-int8_size/fp32_size):.0f}% smaller)")

# Accuracy check
sess_fp = ort.InferenceSession(FP32_PATH)
sess_i8 = ort.InferenceSession(INT8_PATH)
sample  = X_test_s[:500].astype(np.float32)
out_fp  = sess_fp.run(None, {'imu_window': sample})[0].flatten()
out_i8  = sess_i8.run(None, {'imu_window': sample})[0].flatten()
print(f"INT8 vs FP32 max diff: {np.max(np.abs(out_fp - out_i8))*3.6:.4f} km/h")
print(f"INT8 MAE on 500 samples: {np.mean(np.abs(out_i8 - y_test[:500]))*3.6:.3f} km/h")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 5 — Inference Speed Benchmark
# Run AFTER Improvement 3 (needs ONNX files)
# ═══════════════════════════════════════════════════════════════════════════════
IMP5 = """
import onnxruntime as ort, numpy as np, time

FP32_PATH = f'{EXPORT_DIR}/navdrift_attn.onnx'
INT8_PATH = f'{EXPORT_DIR}/navdrift_attn_int8.onnx'

def bench(path, name, batch_sizes=[1, 8, 32, 128], n_runs=500):
    sess = ort.InferenceSession(path,
           providers=['CPUExecutionProvider'])  # CPU-only for edge simulation
    print(f"\\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  {'Batch':>8}  {'Latency (ms)':>14}  {'Throughput (Hz)':>16}")
    print(f"  {'-'*45}")
    for bs in batch_sizes:
        x = np.random.randn(bs, 50, 7).astype(np.float32)
        # Warmup
        for _ in range(20):
            sess.run(None, {'imu_window': x})
        t0 = time.perf_counter()
        for _ in range(n_runs):
            sess.run(None, {'imu_window': x})
        elapsed = (time.perf_counter() - t0) / n_runs * 1000  # ms per call
        hz = bs / (elapsed / 1000)
        print(f"  {bs:>8}  {elapsed:>13.2f}ms  {hz:>14.0f} Hz")
    print(f"\\n  → System runs at 10 Hz. Need 1 inference per 100ms.")
    print(f"    Batch=1 latency above should be well under 100ms.")

bench(FP32_PATH, 'Attention-BiLSTM  FP32  (CPU)')
bench(INT8_PATH, 'Attention-BiLSTM  INT8  (CPU)')

# Real-time factor
sess = ort.InferenceSession(FP32_PATH, providers=['CPUExecutionProvider'])
x1 = np.random.randn(1, 50, 7).astype(np.float32)
times = []
for _ in range(1000):
    t0 = time.perf_counter()
    sess.run(None, {'imu_window': x1})
    times.append((time.perf_counter()-t0)*1000)
p50 = np.percentile(times, 50)
p99 = np.percentile(times, 99)
print(f"\\nSingle-window latency  p50: {p50:.2f}ms  p99: {p99:.2f}ms")
print(f"Real-time budget at 10Hz: 100ms  →  {100/p50:.0f}× headroom")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 6 — Re-run drift eval with new Attention model
# Run AFTER Improvement 2 (uses y_pred = y_pred_attn)
# Copy of Cell A from navdrift_eval_cells.py, works with new y_pred
# ═══════════════════════════════════════════════════════════════════════════════
IMP6 = """
R = 6371000  # Earth radius metres

# Rebuild test_df with new STRIDE=1 indices
# y_test and y_pred are already set from Improvement 2
# We need test rows from df to get heading + lat/lon
# Use last len(X_test) rows of clean_idx past train+val
n_train2 = int(len(X_all)*0.70)
n_val2   = int(len(X_all)*0.15)
test_start_window = n_train2 + n_val2
# Each window i ends at clean_idx[i*STRIDE + SEQ_LEN - 1]
test_end_idxs = [i*1 + SEQ_LEN - 1 for i in range(test_start_window, len(X_all))]
# But with STRIDE=1 window i starts at i, ends at i+SEQ_LEN-1
# We stored y = y_raw[i+SEQ_LEN-1] so the df row for window i is clean_idx[i+SEQ_LEN-1]
test_row_positions = [test_start_window + k + SEQ_LEN - 1 for k in range(len(X_test))]
# Cap at available
test_row_positions = [p for p in test_row_positions if p < len(clean_idx)]
n_cap = len(test_row_positions)
test_df2 = df.loc[[clean_idx[p] for p in test_row_positions]].copy()
test_df2['y_pred'] = y_pred_attn[:n_cap]
test_df2['y_true'] = y_test[:n_cap]

def trip_traj(spds, hdgs, lat0, lon0, dt=0.1):
    lats, lons = [lat0], [lon0]
    for i in range(len(spds)):
        h = np.radians(hdgs[i])
        dlat = spds[i]*np.cos(h)*dt/R
        dlon = spds[i]*np.sin(h)*dt/(R*np.cos(np.radians(lats[-1])))
        lats.append(lats[-1]+np.degrees(dlat))
        lons.append(lons[-1]+np.degrees(dlon))
    return np.array(lats), np.array(lons)

results = []
for src, grp in test_df2.groupby('source_file'):
    if len(grp) < 20: continue
    mae_  = np.mean(np.abs(grp['y_pred']-grp['y_true']))
    rmse_ = np.sqrt(np.mean((grp['y_pred']-grp['y_true'])**2))
    hdg   = grp[ORI_AZ].values
    lt, lnt = trip_traj(grp['y_true'].values, hdg, grp[LAT].iloc[0], grp[LON].iloc[0])
    lp, lnp = trip_traj(grp['y_pred'].values, hdg, grp[LAT].iloc[0], grp[LON].iloc[0])
    dlat_ = (lp[1:]-lt[1:])*R*np.pi/180
    dlon_ = (lnp[1:]-lnt[1:])*R*np.cos(np.radians(lt[1:]))*np.pi/180
    ate_  = np.mean(np.sqrt(dlat_**2+dlon_**2))
    dist_ = np.sum(grp['y_true'].values*0.1)
    drift_= ate_/dist_*100 if dist_>0 else 0
    results.append({'file': src.split('/')[-1][:30], 'n': len(grp),
                    'mae_kmh': mae_*3.6, 'rmse_kmh': rmse_*3.6,
                    'drift_pct': drift_})

import pandas as pd
res2 = pd.DataFrame(results)
print(res2[['file','n','mae_kmh','drift_pct']].to_string(index=False))
print(f"\\nMean drift:  {res2['drift_pct'].mean():.2f}% ± {res2['drift_pct'].std():.2f}%")
print(f"Mean MAE:    {res2['mae_kmh'].mean():.2f} km/h")
print(f"Trips <10%: {(res2['drift_pct']<10).sum()}/{len(res2)}")
"""

if __name__ == '__main__':
    steps = [
        ('IMPROVEMENT 1 — STRIDE=1 (more windows)', IMP1),
        ('IMPROVEMENT 2 — Attention-BiLSTM train',  IMP2),
        ('IMPROVEMENT 3 — ONNX export (attention)',  IMP3),
        ('IMPROVEMENT 4 — INT8 quantization',        IMP4),
        ('IMPROVEMENT 5 — Inference speed benchmark',IMP5),
        ('IMPROVEMENT 6 — Drift eval (new model)',   IMP6),
    ]
    for title, code in steps:
        print(f"\n{'='*70}")
        print(f"# {title}")
        print('='*70)
        print(code)
