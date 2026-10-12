# EKF / position-filter implementations in this repo

This repo contains **three independent, non-shared EKF implementations**. They were
written at different times for different runtimes (browser JS, offline Python
analysis tool, Python production inference service) and have never shared code.
This document records their actual behavior as implemented (verified by reading
the code, not comments) so that any future consolidation can be checked against
real baselines instead of assumptions.

1. `frontend/mobile.html` — inline JS `EKF` object / `ekfStep()` (~line 2159-2232).
   **This is what the live phone app actually uses.** Out of scope to modify for
   this task; documented here only for comparison.
2. `navdrift_engine.py` — `EKF` class (~line 159-184), used by the offline CLI
   analysis tool and by `NavdriftRTEngine` (~line 416-488), which is in turn used
   by `edge/external_imu.py`'s `engine_step_from_frame` and by
   `edge/vehicle_calibration.py`, `edge/replay_200hz_demo.py`,
   `edge/imu_replay_example.py`, `edge/requirement_evidence.py`.
3. `inference/runtime.py` — `_EKF` class (~line 28-45), used internally by
   `NavDriftRuntime` (same file), which is imported by `api/app.py` (the live
   backend API / `DEMO_MODE` path), `demo/demo.py`, and
   `edge/requirement_evidence.py`.

No code path currently calls across these — each caller only ever touches one
of the three.

## 1. State vector

| | mobile.html `EKF` | navdrift_engine.py `EKF` | inference/runtime.py `_EKF` |
|---|---|---|---|
| State | `EKF.x, EKF.y, EKF.th` (3 separate globals, not an array) | `self.x = [lat, lon, heading]` (np.float64 array) | `self.x = [x, y, heading]` (np.float64 array) |
| Position units | **degrees** lat/lon directly (`EKF.x`=lat, `EKF.y`=lon) | **degrees** lat/lon directly (`self.x[0]`=lat, `self.x[1]`=lon) | **local ENU meters**, relative to a reference fix set by `set_initial_gnss_fix()` (`_enu()` helper converts GPS lat/lon to local x/y meters before any EKF call ever sees them) |
| Heading units | radians, unwrapped (no `% 2π` wrap — can grow unbounded over a long session) | radians, unwrapped (`th_new = th + dheading`, no wrap) | radians, wrapped every update via `_wrap()` to `(-π, π]` |
| Heading convention | heading integrated as `EKF.th += dh` where `dh` comes from an IMU-derived yaw-rate-equivalent, used *half-stepped* in the predict (`th = EKF.th + dh*0.5`) for position propagation (a semi-implicit/midpoint integration), but the stored `EKF.th` itself is updated with the *full* `dh` afterward | heading used directly in the full step (`th_new = th + dheading`); no half-step | heading updated via `gz*dt` where `gz` is raw gyro-z passed straight into `predict()`; no half-step |

**Key divergence:** implementation 1 (JS) uses a **midpoint heading** for the
position update but a **full-step heading** for the stored state — the other
two use the *same* (non-midpoint) heading value for both the position update
and the stored state. This is a real algorithmic difference, not just a
units/representation difference, and would change numerical output even given
identical inputs.

## 2. Process / prediction model

- **mobile.html**: `ekfStep(spd, dh, dt, ...)` — inputs are forward speed
  (m/s), delta-heading for this tick (rad), dt (s). Position delta computed as
  `dl = spd*cos(th)/111000*dt` (lat delta in degrees, using a fixed
  111 000 m/deg constant — not WGS84-exact, not latitude-corrected for the
  lat-degree term), `dn = spd*sin(th)/(111000*cos(lat_rad))*dt` (lon delta,
  *is* corrected for latitude via `cos(EKF.x * PI/180)`). No Jacobian is
  computed — the covariance update is **not** an EKF linearized-propagation
  step at all; see §3.
- **navdrift_engine.py**: `EKF.predict(speed, dheading, dt)` — same inputs.
  Uses `R_EARTH = 6_371_000` m consistently for both lat and lon delta
  (`dlat = speed*cos(th)*dt/R_EARTH`, `dlon = speed*sin(th)*dt/(R_EARTH*cos(radians(lat)))`).
  **Does** compute and apply a proper Jacobian `F` (`F[0,2]`, `F[1,2]` partials
  of position w.r.t. heading) and propagates `P = F @ P @ F.T + Q` — a textbook
  EKF covariance propagation.
- **inference/runtime.py**: `_EKF.predict(speed, gz)` — inputs are speed (m/s)
  and raw gyro-z (rad/s); `dt` is fixed at construction (`self.dt`, default
  0.01s), **not** passed per-call. Position update is in local ENU **meters**
  directly: `x[0] += speed*cos(th)*dt`, `x[1] += speed*sin(th)*dt` (no
  lat/lon/Earth-radius conversion needed since state is already local-planar).
  Also computes a Jacobian `F` and propagates `P = F@P@F.T + Q`, same
  structure as navdrift_engine.py's.

## 3. Covariance structure

- **mobile.html**: `EKF.P` is a flat 9-element array laid out as a 3x3 but
  **only the diagonal terms `P[0]` (lat var) and `P[4]` (lon var) are ever
  read or written** by `ekfStep`; `P[8]` (heading variance) is initialized
  once (`.1`) and **never updated** by any predict or update step. All
  off-diagonal cross-covariance terms are hardcoded to `0` at
  init/reset and never populated — this is a **fully decoupled, 2-of-3-diagonal-only**
  filter, not a true 3-state EKF covariance. Growth is `P[0] += Q; P[4] += Q`
  (same scalar `Q` added to both), not a matrix propagation.
- **navdrift_engine.py**: `self.P` is a real 3x3 `np.diag([1e-8, 1e-8, 0.1])`
  at init, but becomes **fully dense** after the first `predict()` because
  `P = F @ P @ F.T + Q` mixes off-diagonal terms through the heading Jacobian
  terms `F[0,2]`/`F[1,2]`. Confirmed empirically (see regression test capture
  below): after 5 predict steps, `P[0,2]` and `P[1,2]` are non-zero
  (~`-1.3e-9`, `5.8e-8` in the captured run). This is a full/proper EKF
  covariance.
- **inference/runtime.py**: Same structure as navdrift_engine.py — `self.P`
  starts as `np.diag([1., 1., .1])` and becomes dense via the same
  `F @ P @ F.T + Q` propagation. Confirmed empirically: `P[0,1]`, `P[0,2]`,
  `P[1,2]` all non-zero after a handful of predict steps.

**Divergence:** mobile.html's filter is the only one of the three that is
*not* a real multi-state EKF covariance — it is two independent 1-D Kalman
filters (lat, lon) sharing nothing, plus an orphaned, never-updated heading
variance slot. The two Python implementations both do correct dense EKF
covariance propagation.

## 4. Measurement / correction model

- **mobile.html**: `ekfStep(..., gpsMeasured, gla, glo, gpsAccM)` — update
  runs **only if `gpsMeasured` is truthy and `gla`/`glo` are finite**. Measures
  **lat/lon only** — there is **no heading/course correction from GPS** at
  all (confirmed: no `EKF.th` write anywhere in the update branch). GPS
  accuracy (`gpsAccM`, meters) is converted to degrees
  (`accDeg = max(3, gpsAccM||8)/111000`) and squared into the per-axis `R`.
  Kalman gain computed per-axis independently (`Kx = P[0]/(P[0]+R)`,
  `Ky = P[4]/(P[4]+R)`) — consistent with the decoupled-covariance design in
  §3. No innovation gating/outlier rejection of any kind — any finite GPS fix
  is accepted and blended in directly, however far it is from the current
  estimate.
- **navdrift_engine.py**: `EKF.update_gps(lat_meas, lon_meas)` — measures
  **lat/lon only** (`H = [[1,0,0],[0,1,0]]`); heading is never corrected by
  GPS here either. Standard EKF update: `y = z - H@x`, `S = H@P@H.T + R`,
  `K = P@H.T@inv(S)`, `x += K@y`, `P = (I - K@H)@P`. `R` is a fixed
  `diag([R_GPS/R_EARTH, R_GPS/R_EARTH])` set once at construction
  (`R_GPS = 4.0` meters, converted to the position state's degree-ish scale by
  dividing by `R_EARTH` — note this reuses the same `R_EARTH` constant as a
  rough degrees-per-meter conversion, not a real unit transform). No
  innovation gating/outlier rejection.
- **inference/runtime.py**: `_EKF.update(ox, oy, ot)` — measures **x, y, AND
  heading** (`ot`), the only one of the three that corrects heading from an
  external fix at all. Innovation computed with heading wrapped
  (`_wrap(ot - x[2])`). Update uses a **simplified gain**
  `K = P @ inv(P + R)` (i.e. it assumes `H = I`, not a general
  `H @ P @ H.T + R`) and `P = (I - K) @ P` instead of the general Joseph/`(I-KH)P`
  form — algebraically equivalent only because `H` is implicitly identity
  here, which is correct for this call site (the caller, `reacquire()`,
  always supplies all three of x/y/heading) but means this `update()` cannot
  be reused as-is for a lat/lon-only correction the way the other two
  classes' update methods can. No innovation gating/outlier rejection.
  Called from `NavDriftRuntime.reacquire()`, not from the regular per-step
  `ingest()` path — i.e. the "GPS correction" here is specifically a
  reacquisition/snap-back event, paired with a separate `_SNAP` trajectory
  smoother that also warps the stored pose history, not a per-tick fusion
  step.

## 5. Process noise / measurement noise handling

- **mobile.html**: `Q = (EKF.Qp != null ? EKF.Qp : 2.2e-11) * (1 + dt*4) * shockBoost`.
  **This is the only one of the three that actually consumes an
  adaptive/learned noise model** — `EKF.Qp` and `EKF.Rp`, when non-null, come
  from the trained Adaptive-EKF ONNX model (referenced in the surrounding code
  as `NAVDRIFT_ONNX`, set elsewhere in mobile.html outside the excerpted
  range; `EKF.Qp`/`EKF.Rp` default to `null` meaning "fall back to the fixed
  classical constants" per the inline comment at the `EKF` object
  definition). `R` similarly becomes `(EKF.Rp != null ? EKF.Rp : 1) * accDeg^2`
  — i.e. `Rp` is a *scale factor* on the GPS-accuracy-derived variance, not a
  replacement for it. There's also a shock/pothole-triggered temporary Q
  multiplier (`shockBoost`, from `detectShock()`), a feature neither Python
  implementation has.
- **navdrift_engine.py**: `Q = diag([1e-10, 1e-10, Q_HEAD*0.01])` and
  `R = diag([R_GPS/R_EARTH, R_GPS/R_EARTH])`, both **fixed module-level
  constants** (`Q_SPD=0.1`, `Q_HEAD=0.05`, `R_GPS=4.0`, defined once near the
  top of the file) set once at `EKF.__init__` and never touched again — no
  adaptive noise, no ONNX noise predictor consumed anywhere in this class.
  (`Q_SPD` is defined but not actually referenced anywhere in the `EKF`
  class's Q — it appears unused dead config for this filter.)
- **inference/runtime.py**: `Q = diag([0.01, 0.01, 0.001])`,
  `R = diag([0.5, 0.5, 0.02])`, fixed at `_EKF.__init__`, never updated —
  no adaptive noise here either. The only "adaptivity" in this file is the
  separate ONNX **drift-correction** model (`self._sess`, loaded only when
  `DEMO_MODE=false`) used in `_onnx()` to directly nudge `self._ekf.x` — that
  model's output is **added directly to the state** (`x[0] += out[0]`, etc.),
  it is not consumed as a Q/R noise scale the way mobile.html's `EKF.Qp/Rp`
  are. When the ONNX session/buffer isn't ready, `ingest()` falls back to the
  plain `_EKF.predict()` kinematic step.

## 6. Known limitations per implementation

- **mobile.html `EKF`/`ekfStep`** (out of scope to change, documented for
  completeness):
  - Decoupled lat/lon covariance; heading variance (`P[8]`) is dead state,
    never updated.
  - No GPS→heading correction at all (course is never fused in).
  - No innovation gating — a wild GPS outlier is blended in unconditionally
    (bounded only by the Kalman gain itself, which shrinks as `P` shrinks).
  - Midpoint-heading-for-position but full-step-heading-for-state asymmetry
    (see §1) is unusual and undocumented in the original code; unclear if
    intentional.
  - Has NaN/Infinity input and output sanity guards (`ekfStep`'s top and
    bottom) that neither Python implementation has.
- **navdrift_engine.py `EKF`**:
  - Fixed noise constants only; no adaptive Q/R.
  - No GPS→heading correction (lat/lon-only measurement model).
  - No innovation gating/outlier rejection.
  - No NaN/Infinity guarding — a bad IMU sample will propagate NaN into
    `self.x` permanently, same failure mode mobile.html explicitly guards
    against.
  - Uses `R_EARTH` as a stand-in "meters per degree-ish" constant inside `R`
    (`R_GPS/R_EARTH`), which is not a correct unit conversion (111 320 m/deg
    at the equator vs. `R_EARTH`=6 371 000 — off by roughly 57x); this looks
    like a latent correctness bug inherited from treating degrees as if they
    were radians-on-a-unit-sphere. Pinned as-is by the regression test below;
    not fixed in this pass per the task's "preserve behavior" instruction.
- **inference/runtime.py `_EKF`**:
  - Fixed noise constants only; no adaptive Q/R; no ONNX-driven noise (the
    ONNX model here does direct state correction, a different mechanism).
  - `dt` is fixed at construction and not passable per predict() call, unlike
    the other two implementations — if the real sample rate drifts from the
    constructor's assumed `imu_hz`, the model silently integrates with the
    wrong dt.
  - `update()`'s simplified `K = P @ inv(P+R)` / `(I-K)@P` form is only valid
    because its one caller always supplies a full x/y/heading observation; it
    cannot be reused for a lat/lon-only GPS correction without changing its
    math, which matters if anyone later tries to unify this with the other
    two update() methods.
  - No innovation gating/outlier rejection; no NaN/Infinity guarding.
  - State is in local ENU meters relative to a reference fix set once via
    `set_initial_gnss_fix()` — mixing this engine's raw `x`/`y` with the other
    two engines' lat/lon state requires the `_enu()`/inverse conversion; they
    are not interchangeable without that step.

## 7. Real callers today

| Implementation | Real caller(s) |
|---|---|
| mobile.html `EKF`/`ekfStep` | The live phone app itself — `ekfStep()` is called from the app's per-frame navigation loop (not excerpted in full here, out of scope for this task) and its state (`EKF.x/y/th`) is read directly by `exportIMULog()` for `ekf_lat/ekf_lon/ekf_heading_rad` CSV columns and by `ekfUncertaintyM()` for the UI's displayed uncertainty. |
| navdrift_engine.py `EKF` | Directly: the file's own `__main__` CLI path (`navdrift_engine.py --imu ...`) at `navdrift_engine.py:289` (batch mode) and the real-time wrapper `NavdriftRTEngine` at `navdrift_engine.py:423` (`self.ekf = EKF(...)`, driven by `.step()`/`.gps_update()`). `NavdriftRTEngine` itself is used by `edge/external_imu.py`'s `engine_step_from_frame()` (confirmed via `edge/external_imu.py` module docstring referencing `navdrift_engine.py`'s `NavdriftRTEngine`), and by `edge/vehicle_calibration.py`, `edge/replay_200hz_demo.py`, `edge/imu_replay_example.py`, `edge/requirement_evidence.py` (confirmed by `grep -rl navdrift_engine`). No pytest file directly instantiates `navdrift_engine.EKF` prior to this task's new tests (`tests/test_butterworth_stability.py` imports `NavdriftRTEngine` but exercises the Butterworth filter, not the EKF class itself). |
| inference/runtime.py `_EKF` | Only reachable through `NavDriftRuntime` (same file), which is imported by `api/app.py` (confirmed via `grep -rl NavDriftRuntime` — this is the production/demo backend API), `demo/demo.py`, and `edge/requirement_evidence.py`. `_EKF.predict()` runs on every `NavDriftRuntime.ingest()` call that doesn't go through the ONNX drift-correction path; `_EKF.update()` runs only from `NavDriftRuntime.reacquire()`. No pytest file exercised `_EKF` directly prior to this task's new tests. |

## 8. Regression tests added

See `tests/test_ekf_implementations.py`. Both Python implementations
(`navdrift_engine.EKF` / `NavdriftRTEngine`, and `inference.runtime._EKF`) had
**zero existing direct test coverage** of their EKF classes (confirmed by
grepping `tests/` for `EKF`/`ekf` and for imports of either module — only
`navdrift_engine`'s Butterworth filter and `NavdriftRTEngine`'s high-level
`.step()` plumbing were covered, not the EKF math itself, and
`inference/runtime.py` had no test coverage at all). The new tests feed fixed,
deterministic input sequences into each filter and assert the resulting state
and covariance match the values captured from the current code (see values
embedded in the test file and in the captures below), so any future change to
either filter's math will be caught.

Captured baseline values (from running the current code, 2026-10-12):

```
navdrift_engine.EKF: start (12.9716, 77.5946, 0.0)
  after 5 predicts: x = [12.97160057, 77.5946, 0.035], P.diag = [1.05e-08, 1.050000338e-08, 0.1025]
  after update_gps(12.97165, 77.59465):
    x = [12.97160144, 77.59460076, 0.0350044846]

inference.runtime._EKF: dt=0.1, start [0,0,0]
  after 5 predicts: x = [3.59999235, 0.00649999, 0.0035]
  after update(10.0, 5.0, 0.05):
    x = [7.93503351, 3.63772496, 0.20232292]
```

## 9. Consolidation decision

**Not attempted in this pass.** Per §1-§6 above, `navdrift_engine.EKF` and
`inference.runtime._EKF` differ in: state representation (lat/lon degrees vs.
local ENU meters), per-call vs. fixed `dt`, and — most importantly —
`update()` semantics (lat/lon-only vs. full x/y/heading observation, and a
simplified `H=I` gain form vs. a general `H`). Making one delegate to the
other would require either (a) adding a heading-observation code path to
`navdrift_engine.EKF.update_gps` that no current caller uses (risk of
behavior change for `navdrift_engine.py`'s and `NavdriftRTEngine`'s existing
callers if done carelessly), or (b) generalizing `inference.runtime._EKF`'s
gain computation to a real `H` matrix (safe in isolation, but not actually
needed by its only caller and not validated against new/old numeric output
here). Given the task's explicit preference for safety over consolidation,
and that both filters are correctly covered by regression tests now, this is
left as a follow-up once someone wants to actually unify behavior rather than
code shape. The two Python implementations were left unmodified; mobile.html
was not touched at all per the task's instructions.
