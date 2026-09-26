"""
patch_imu.py  —  applies the IMU speed-fix to frontend/mobile.html in-place
Run from the repo root:  python patch_imu.py
"""
import sys, os

TARGET = os.path.join(os.path.dirname(__file__) if '__file__' in dir() else '.', 'frontend', 'mobile.html')
# Allow running from repo root or outputs folder
if not os.path.exists(TARGET):
    TARGET = os.path.join('frontend', 'mobile.html')
if not os.path.exists(TARGET):
    sys.exit("ERROR: cannot find frontend/mobile.html — run from repo root (D:\\navdrift0-main)")

with open(TARGET, 'rb') as f:
    raw = f.read()

# Work in text, preserve original line endings per-line via bytes round-trip
text = raw.decode('utf-8')

PATCHES = []

# ── PATCH 1: add fwdAccelBias + _gravFree to IMU object ──────────────────────
PATCHES.append((
    """  /* calibration status */
  calibrated: false
};""",
    """  /* calibration status */
  calibrated: false,
  /* forward-accel residual bias (captured during still calibration) */
  fwdAccelBias: 0,
  /* true when device provides gravity-free e.acceleration */
  _gravFree: false
};"""
))

# ── PATCH 2: onDeviceMotion — prefer gravity-free accel + fix raw readings ───
PATCHES.append((
    """function onDeviceMotion(e){
  const acc = e.accelerationIncludingGravity || e.acceleration;
  if(!acc) return;
  const now = performance.now();

  // Raw readings (m/s² on Android; iOS same)
  const rawAx = (acc.x || 0) - IMU.axOff;
  const rawAy = (acc.y || 0) - IMU.ayOff;
  const rawAz = (acc.z || 0) - IMU.azOff;""",
    """function onDeviceMotion(e){
  // Prefer gravity-free acceleration (e.acceleration) — already has gravity subtracted
  // Fall back to gravity-included only if gravity-free is unavailable/zero
  const accLin  = e.acceleration;
  const accGrav = e.accelerationIncludingGravity;
  const linMag = accLin ? Math.abs(accLin.x||0)+Math.abs(accLin.y||0)+Math.abs(accLin.z||0) : 0;
  const useLinear = linMag > 0.01;   // gravity-free available and non-trivial
  IMU._gravFree = useLinear;
  const acc = useLinear ? accLin : accGrav;
  if(!acc) return;
  const now = performance.now();

  // Raw readings (m/s²)
  const rawAx = (acc.x || 0) - IMU.axOff;
  const rawAy = (acc.y || 0) - IMU.ayOff;
  const rawAz = (acc.z || 0) - (useLinear ? IMU.azOff : (IMU.azOff + 9.81));"""
))

# ── PATCH 3: replace fwdAccel formula + speed integration ────────────────────
PATCHES.append((
    """  // Speed estimation: project filtered accel onto forward axis (phone lying flat: fwd = +Y)
  // Simple forward accel integration with gravity removal
  const dt = _lastDMT>0 ? (now-_lastDMT)/1000 : 0;
  _lastDMT = now;
  const cosB = Math.cos(IMU.beta*Math.PI/180);
  const sinB = Math.sin(IMU.beta*Math.PI/180);
  // forward acceleration = ay*cosB - az*sinB (phone held portrait, tilted)
  IMU.fwdAccel = clamp(fay*cosB - (faz - 9.81*Math.cos(IMU.beta*Math.PI/180))*sinB, -15, 15);
  if(dt>0 && dt<0.5){
    IMU.intSpeed = Math.max(0, IMU.intSpeed + IMU.fwdAccel*dt);
    IMU.intSpeed = Math.min(IMU.intSpeed, 40); // cap at 40 m/s (~144 km/h)
    // heading from gz integration
    IMU.intHeading += IMU.gz * dt;
  }""",
    """  // Speed estimation: project filtered accel onto forward axis
  const dt = _lastDMT>0 ? (now-_lastDMT)/1000 : 0;
  _lastDMT = now;

  let rawFwd;
  if(useLinear){
    // Gravity already removed by the OS — Y axis = phone's long axis = forward direction
    rawFwd = fay;
  } else {
    // Manual gravity removal using beta (front-back tilt) and gamma (left-right tilt)
    const beta  = IMU.beta  * Math.PI/180;
    const gamma = IMU.gamma * Math.PI/180;
    const cosB = Math.cos(beta), sinB = Math.sin(beta);
    const cosG = Math.cos(gamma), sinG = Math.sin(gamma);
    // Gravity projected onto device Y and Z axes (iPhone convention):
    //   gy_grav =  9.81 * sinB * cosG   (tilting forward puts gravity on +Y)
    //   gz_grav = -9.81 * cosB * cosG   (face-up: -9.81 on Z)
    const gravY =  9.81 * sinB * cosG;
    const gravZ = -9.81 * cosB * cosG;
    const linY = fay - gravY;
    const linZ = faz - gravZ;
    // Project onto car-forward axis: rotate by tilt angle beta
    rawFwd = linY * cosB - linZ * sinB;
  }

  // Subtract residual bias captured during stationary calibration
  const biasedFwd = rawFwd - IMU.fwdAccelBias;
  // Dead-band: small accel below 0.25 m/s² is noise (smooth road rolling)
  const deadFwd = Math.abs(biasedFwd) < 0.25 ? 0 : biasedFwd;
  IMU.fwdAccel = clamp(deadFwd, -15, 15);

  if(dt>0 && dt<0.5){
    IMU.intSpeed = Math.max(0, IMU.intSpeed + IMU.fwdAccel*dt);
    IMU.intSpeed = Math.min(IMU.intSpeed, 40); // cap at 40 m/s (~144 km/h)
    // heading from gz integration
    IMU.intHeading += IMU.gz * dt;
  }"""
))

# ── PATCH 4: replace 1-second calibration with 2-second averaging ─────────────
PATCHES.append((
    """    // Auto-calibrate: capture bias for 1 second then mark calibrated
    status.textContent = 'Calibrating bias… hold device still.';
    setTimeout(()=>{
      IMU.axOff = IMU.ax; IMU.ayOff = IMU.ay; IMU.azOff = (IMU.az - 9.81);
      IMU.gxOff = IMU.gx; IMU.gyOff = IMU.gy; IMU.gzOff = IMU.gz;
      IMU.calibrated = true;
      status.textContent = '✓ Calibrated. Live IMU active at ~' + (IMU.hz||30) + ' Hz.';
      activateSensorMode();
      setTimeout(closeSensorModal, 1200);
    }, 1200);""",
    """    // Auto-calibrate: accumulate samples for 2 seconds then compute mean bias
    status.textContent = 'Calibrating bias… hold device still (2 s).';
    let _calSamples = [];
    const _calHandler = (ev)=>{
      const a = (ev.acceleration && (Math.abs(ev.acceleration.x||0)+Math.abs(ev.acceleration.y||0)+Math.abs(ev.acceleration.z||0))>0.01)
                ? ev.acceleration : ev.accelerationIncludingGravity;
      if(a) _calSamples.push({ax:a.x||0,ay:a.y||0,az:a.z||0,gz:(ev.rotationRate?.gamma||0)*Math.PI/180,fwd:IMU.fwdAccel});
    };
    window.addEventListener('devicemotion', _calHandler, {passive:true});
    setTimeout(()=>{
      window.removeEventListener('devicemotion', _calHandler);
      if(_calSamples.length > 5){
        const n = _calSamples.length;
        const meanAx = _calSamples.reduce((s,r)=>s+r.ax,0)/n;
        const meanAy = _calSamples.reduce((s,r)=>s+r.ay,0)/n;
        const meanAz = _calSamples.reduce((s,r)=>s+r.az,0)/n;
        const meanGz = _calSamples.reduce((s,r)=>s+r.gz,0)/n;
        const meanFwd= _calSamples.reduce((s,r)=>s+r.fwd,0)/n;
        IMU.axOff = meanAx; IMU.ayOff = meanAy;
        IMU.azOff = IMU._gravFree ? meanAz : (meanAz - 9.81);
        IMU.gxOff = IMU.gx; IMU.gyOff = IMU.gy; IMU.gzOff = meanGz;
        IMU.fwdAccelBias = meanFwd;
        _bw_xp=[0,0,0]; _bw_yp1=[0,0,0]; _bw_yp2=[0,0,0];
      }
      IMU.calibrated = true;
      status.textContent = '✓ Calibrated. Live IMU active at ~' + (IMU.hz||30) + ' Hz.';
      activateSensorMode();
      setTimeout(closeSensorModal, 1200);
    }, 2000);"""
))

applied = 0
for old, new in PATCHES:
    if old in text:
        text = text.replace(old, new, 1)
        applied += 1
        print(f"  ✓ patch {applied} applied")
    else:
        print(f"  ✗ patch {applied+1} NOT FOUND — skipping (already applied?)")
        applied += 1

# Write back preserving original encoding
with open(TARGET, 'wb') as f:
    f.write(text.encode('utf-8'))

print(f"\nDone. {TARGET} updated.")
print("Now run:  git add -f frontend/mobile.html && git commit -m 'fix: IMU gravity-free accel + bias calibration' && git push origin isro-grade")
