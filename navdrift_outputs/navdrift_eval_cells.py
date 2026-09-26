"""
NAVDRIFT0 — Extra evaluation cells (A through E)
Paste each cell into Colab in order. Run Cell A first, paste output, then B, etc.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL A — Per-trip evaluation
# ═══════════════════════════════════════════════════════════════════════════════
CELL_A = """
results = []
test_orig_idx = clean_idx[n_train + len(X_val) + SEQ_LEN - 1 :
                           n_train + len(X_val) + SEQ_LEN - 1 + len(X_test)]
test_df = df.loc[test_orig_idx].copy()
test_df['y_pred'] = y_pred
test_df['y_true'] = y_test

def trip_traj(spds, hdgs, lat0, lon0, dt=0.1):
    lats, lons = [lat0], [lon0]
    for i in range(len(spds)):
        h = np.radians(hdgs[i])
        dlat = spds[i]*np.cos(h)*dt/R
        dlon = spds[i]*np.sin(h)*dt/(R*np.cos(np.radians(lats[-1])))
        lats.append(lats[-1]+np.degrees(dlat))
        lons.append(lons[-1]+np.degrees(dlon))
    return np.array(lats), np.array(lons)

for src, grp in test_df.groupby('source_file'):
    if len(grp) < 20:
        continue
    mae  = np.mean(np.abs(grp['y_pred'] - grp['y_true']))
    rmse = np.sqrt(np.mean((grp['y_pred'] - grp['y_true'])**2))
    hdg  = grp[ORI_AZ].values
    spd_t= grp['y_true'].values
    spd_p= grp['y_pred'].values
    lat0_= grp[LAT].iloc[0]; lon0_= grp[LON].iloc[0]
    lt, lnt = trip_traj(spd_t, hdg, lat0_, lon0_)
    lp, lnp = trip_traj(spd_p, hdg, lat0_, lon0_)
    dlat_ = (lp[1:]-lt[1:])*R*np.pi/180
    dlon_ = (lnp[1:]-lnt[1:])*R*np.cos(np.radians(lt[1:]))*np.pi/180
    ate_  = np.mean(np.sqrt(dlat_**2+dlon_**2))
    dist_ = np.sum(spd_t*0.1)
    drift_= ate_/dist_*100 if dist_>0 else 0
    results.append({'file': src.split('/')[-1][:30], 'n': len(grp),
                    'mae_kmh': mae*3.6, 'rmse_kmh': rmse*3.6,
                    'ate_m': ate_, 'dist_m': dist_, 'drift_pct': drift_})

res_df = pd.DataFrame(results)
print(res_df[['file','n','mae_kmh','rmse_kmh','drift_pct']].to_string(index=False))
print(f"\\nMean drift: {res_df['drift_pct'].mean():.2f}% ± {res_df['drift_pct'].std():.2f}%")
print(f"Trips <10%: {(res_df['drift_pct']<10).sum()}/{len(res_df)}")
print(f"Mean MAE:   {res_df['mae_kmh'].mean():.2f} ± {res_df['mae_kmh'].std():.2f} km/h")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL B — Baseline comparison (Linear Regression + 1-layer MLP)
# ═══════════════════════════════════════════════════════════════════════════════
CELL_B = """
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

# Flatten windows for sklearn
X_tr_flat = X_train.reshape(len(X_train), -1)
X_te_flat = X_test.reshape(len(X_test), -1)

# Baseline 1: Ridge regression
ridge = Ridge()
ridge.fit(X_tr_flat, y_train)
y_ridge = ridge.predict(X_te_flat)
mae_ridge = mean_absolute_error(y_test, y_ridge) * 3.6

# Baseline 2: Shallow MLP
mlp = MLPRegressor(hidden_layer_sizes=(64,), max_iter=200, random_state=42)
mlp.fit(X_tr_flat, y_train)
y_mlp = mlp.predict(X_te_flat)
mae_mlp = mean_absolute_error(y_test, y_mlp) * 3.6

mae_bilstm = np.mean(np.abs(y_pred - y_test)) * 3.6

print("=" * 40)
print(f"{'Model':<20} {'MAE (km/h)':>10}")
print("=" * 40)
print(f"{'Ridge Regression':<20} {mae_ridge:>10.3f}")
print(f"{'Shallow MLP (64)':<20} {mae_mlp:>10.3f}")
print(f"{'BiLSTM (ours)':<20} {mae_bilstm:>10.3f}")
print("=" * 40)
print(f"Improvement over Ridge: {(mae_ridge-mae_bilstm)/mae_ridge*100:.1f}%")
print(f"Improvement over MLP:   {(mae_mlp-mae_bilstm)/mae_mlp*100:.1f}%")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL C — GPS blackout simulation
# ═══════════════════════════════════════════════════════════════════════════════
CELL_C = """
import matplotlib.pyplot as plt

# Use test_df built in Cell A
# Simulate blackouts of 10s, 30s, 60s on the first 600 samples of test set
sample_slice = test_df.iloc[:600].copy()
sample_pred  = y_pred[:600]
sample_true  = sample_slice['y_true'].values
hdg_s        = sample_slice[ORI_AZ].values
lat0_s = sample_slice[LAT].iloc[0]
lon0_s = sample_slice[LON].iloc[0]
dt = 0.1   # 10 Hz

def dr_with_blackout(spd_true, spd_pred, hdg, lat0, lon0, blackout_start, blackout_len, dt=0.1):
    \"\"\"During blackout use predicted speed, outside use GPS speed.\"\"\"
    lats_gt, lons_gt   = [lat0], [lon0]
    lats_dr, lons_dr   = [lat0], [lon0]
    bo_end = blackout_start + blackout_len
    for i in range(len(spd_true)):
        h = np.radians(hdg[i])
        # GT always uses GPS speed
        dlat = spd_true[i]*np.cos(h)*dt/R
        dlon = spd_true[i]*np.sin(h)*dt/(R*np.cos(np.radians(lats_gt[-1])))
        lats_gt.append(lats_gt[-1]+np.degrees(dlat))
        lons_gt.append(lons_gt[-1]+np.degrees(dlon))
        # DR uses predicted speed during blackout
        spd = spd_pred[i] if blackout_start <= i < bo_end else spd_true[i]
        dlat = spd*np.cos(h)*dt/R
        dlon = spd*np.sin(h)*dt/(R*np.cos(np.radians(lats_dr[-1])))
        lats_dr.append(lats_dr[-1]+np.degrees(dlat))
        lons_dr.append(lons_dr[-1]+np.degrees(dlon))
    # Position error at end of blackout
    i = bo_end
    dlat_ = (lats_dr[i]-lats_gt[i])*R*np.pi/180
    dlon_ = (lons_dr[i]-lons_gt[i])*R*np.cos(np.radians(lats_gt[i]))*np.pi/180
    err_m = np.sqrt(dlat_**2+dlon_**2)
    return np.array(lats_gt), np.array(lons_gt), np.array(lats_dr), np.array(lons_dr), err_m

blackout_start = 100   # sample index ~10s in
durations = [10, 30, 60]   # seconds → samples at 10Hz = 100, 300, 600... use 10,30,60 samples
dur_samples = [int(d/dt) for d in durations]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, dur, dur_s in zip(axes, durations, dur_samples):
    if blackout_start + dur_s > len(sample_pred):
        dur_s = len(sample_pred) - blackout_start - 1
    lt, lnt, lp, lnp, err = dr_with_blackout(
        sample_true, sample_pred, hdg_s, lat0_s, lon0_s, blackout_start, dur_s)
    bo_end = blackout_start + dur_s
    ax.plot(lnt, lt, 'b-', linewidth=2, label='GT (GPS)')
    ax.plot(lnp, lp, 'r--', linewidth=1.5, label=f'DR (BiLSTM)')
    # Mark blackout region
    ax.axvspan(lnt[blackout_start], lnt[min(bo_end, len(lnt)-1)],
               alpha=0.15, color='red', label='GPS blackout')
    ax.set_title(f'{dur}s blackout — error: {err:.1f}m')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.legend(fontsize=8)

plt.suptitle('Dead Reckoning During GPS Blackout (BiLSTM Speed Estimation)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{EXPORT_DIR}/blackout_simulation.png', dpi=150)
plt.show()

# Summary table
print("\\nGPS Blackout Summary:")
print(f"{'Duration':>10} {'Samples':>8} {'Error (m)':>10} {'Drift %':>8}")
for dur, dur_s in zip(durations, dur_samples):
    if blackout_start + dur_s > len(sample_pred):
        dur_s = len(sample_pred) - blackout_start - 1
    *_, err = dr_with_blackout(sample_true, sample_pred, hdg_s, lat0_s, lon0_s, blackout_start, dur_s)
    dist_bo = np.sum(sample_true[blackout_start:blackout_start+dur_s])*dt
    pct = err/dist_bo*100 if dist_bo > 0 else 0
    print(f"{dur:>9}s {dur_s:>8} {err:>10.1f} {pct:>7.1f}%")
print(f"Plot saved → {EXPORT_DIR}/blackout_simulation.png")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL D — Error distribution histogram
# ═══════════════════════════════════════════════════════════════════════════════
CELL_D = """
import matplotlib.pyplot as plt

errors_kmh = (y_pred - y_test) * 3.6

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Histogram
axes[0].hist(errors_kmh, bins=60, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(0, color='black', linewidth=1.5, linestyle='--')
axes[0].axvline(np.mean(errors_kmh), color='red', linewidth=1.5,
                label=f'Mean bias: {np.mean(errors_kmh):.3f} km/h')
axes[0].axvline(np.percentile(errors_kmh, 5),  color='orange', linewidth=1, linestyle=':')
axes[0].axvline(np.percentile(errors_kmh, 95), color='orange', linewidth=1, linestyle=':',
                label=f'5th–95th pct: [{np.percentile(errors_kmh,5):.1f}, {np.percentile(errors_kmh,95):.1f}] km/h')
axes[0].set_xlabel('Speed Error (km/h)'); axes[0].set_ylabel('Count')
axes[0].set_title('Speed Prediction Error Distribution')
axes[0].legend()

# CDF
sorted_err = np.sort(np.abs(errors_kmh))
cdf = np.arange(1, len(sorted_err)+1) / len(sorted_err)
axes[1].plot(sorted_err, cdf, color='steelblue', linewidth=2)
axes[1].axhline(0.90, color='red', linestyle='--', alpha=0.7, label='90th percentile')
p90 = np.percentile(np.abs(errors_kmh), 90)
axes[1].axvline(p90, color='orange', linestyle='--', alpha=0.7, label=f'|error| < {p90:.2f} km/h')
axes[1].set_xlabel('|Speed Error| (km/h)'); axes[1].set_ylabel('CDF')
axes[1].set_title('Cumulative Error Distribution')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{EXPORT_DIR}/error_distribution.png', dpi=150)
plt.show()

print(f"Bias (mean error): {np.mean(errors_kmh):.4f} km/h")
print(f"Std of error:      {np.std(errors_kmh):.4f} km/h")
print(f"90th pct |error|:  {p90:.3f} km/h")
print(f"% within ±2 km/h:  {(np.abs(errors_kmh)<2).mean()*100:.1f}%")
print(f"% within ±5 km/h:  {(np.abs(errors_kmh)<5).mean()*100:.1f}%")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL E — Speed-binned MAE
# ═══════════════════════════════════════════════════════════════════════════════
CELL_E = """
import matplotlib.pyplot as plt

bins = [(0,2,'Stationary\\n(0–2 km/h)'),
        (2,30,'Urban\\n(2–30 km/h)'),
        (30,200,'Highway\\n(>30 km/h)')]

y_true_kmh = y_test * 3.6
y_pred_kmh = y_pred * 3.6

print(f"{'Regime':<22} {'N':>6} {'MAE (km/h)':>11} {'RMSE (km/h)':>12} {'% of data':>10}")
print("-"*65)
bin_maes, bin_labels, bin_ns = [], [], []
for lo, hi, label in bins:
    mask = (y_true_kmh >= lo) & (y_true_kmh < hi)
    if mask.sum() == 0:
        continue
    mae_  = np.mean(np.abs(y_pred_kmh[mask] - y_true_kmh[mask]))
    rmse_ = np.sqrt(np.mean((y_pred_kmh[mask] - y_true_kmh[mask])**2))
    pct_  = mask.mean()*100
    clean = label.replace('\\n', ' ')
    print(f"{clean:<22} {mask.sum():>6} {mae_:>11.3f} {rmse_:>12.3f} {pct_:>9.1f}%")
    bin_maes.append(mae_); bin_labels.append(label); bin_ns.append(mask.sum())

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(bin_labels, bin_maes, color=['#4c72b0','#dd8452','#55a868'], width=0.5)
ax.bar_label(bars, fmt='%.2f km/h', padding=4)
ax.set_ylabel('MAE (km/h)'); ax.set_title('Speed MAE by Driving Regime')
ax.axhline(np.mean(np.abs(y_pred_kmh - y_true_kmh)), color='red',
           linestyle='--', label=f'Overall MAE: {np.mean(np.abs(y_pred_kmh-y_true_kmh)):.2f} km/h')
ax.legend(); ax.set_ylim(0, max(bin_maes)*1.3)
plt.tight_layout()
plt.savefig(f'{EXPORT_DIR}/speed_binned_mae.png', dpi=150)
plt.show()
print(f"Plot saved → {EXPORT_DIR}/speed_binned_mae.png")
"""

if __name__ == '__main__':
    print("Copy each CELL_A through CELL_E string into separate Colab cells.")
    print("Run in order: A → B → C → D → E")
    for name, code in [('A',CELL_A),('B',CELL_B),('C',CELL_C),('D',CELL_D),('E',CELL_E)]:
        print(f"\n{'='*60}")
        print(f"# CELL {name}")
        print('='*60)
        print(code)
