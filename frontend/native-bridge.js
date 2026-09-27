/* ── NAVDRIFT-0 NATIVE BRIDGE ──
 * Loaded after the main inline script (so SensorManager/GPS/IMU/BaroSensor already exist as
 * globals). Does nothing at all in a plain browser — Capacitor.isNativePlatform() is only true
 * inside the actual native Android/iOS app shell, so the web deployment (Render, GitHub Pages,
 * any browser) is completely unaffected by this file's presence. This is what makes "one engine,
 * one UI, real native sensors" possible: the native plugin (native/android-plugin or
 * native/ios-plugin) reports real hardware events, and this file's only job is translating those
 * into the exact same SensorManager.setValue/setStatus calls the browser sensor code already
 * makes — so ekfStep(), RoadGraph, BaroSensor's UI rendering, and everything else downstream
 * never has to know or care which platform it's running on.
 *
 * If this ever runs somewhere Capacitor isn't present (window.Capacitor undefined), it exits
 * immediately and changes nothing — never a silent partial state.
 */
(function () {
  'use strict';
  if (typeof window.Capacitor === 'undefined' || !window.Capacitor.isNativePlatform || !window.Capacitor.isNativePlatform()) {
    return; // plain browser (or Capacitor not loaded yet) — web sensor code handles everything
  }
  if (typeof SensorManager === 'undefined') {
    console.warn('[NativeBridge] Capacitor native platform detected but SensorManager is not yet defined — bridge not attached.');
    return;
  }

  const NavdriftSensors = window.Capacitor.Plugins && window.Capacitor.Plugins.NavdriftSensors;
  if (!NavdriftSensors) {
    console.warn('[NativeBridge] Native platform detected but the NavdriftSensors plugin is not registered — falling back to whatever web sensor APIs this native WebView exposes.');
    return;
  }

  console.log('[NativeBridge] Native platform detected — routing real device sensors through NavdriftSensors plugin.');

  // ── GNSS ──
  NavdriftSensors.addListener('gnssFix', (fix) => {
    // Same discard-don't-fabricate rule as the web GPS handler (onFix in mobile.html) — a
    // malformed native fix must never reach GPS.lat/lon.
    if (!isFinite(fix.latitude) || !isFinite(fix.longitude)) {
      console.warn('[NativeBridge] non-finite native GNSS fix discarded', fix);
      return;
    }
    GPS.lat = fix.latitude; GPS.lon = fix.longitude; GPS.acc = fix.accuracy;
    GPS.lastLat = fix.latitude; GPS.lastLon = fix.longitude;
    GPS.lastFixTime = performance.now();
    GPS.fixConsumed = false;
    GPS.hardError = false;
    SensorManager.setValue('gnss', {
      latitude: fix.latitude, longitude: fix.longitude, altitude: fix.altitude ?? null,
      accuracy: fix.accuracy, speed: fix.speed ?? null, course: fix.course ?? null,
      timestamp: fix.timestamp
    }, fix.accuracy, 'NavdriftSensors(native-gnss)');
    if (!GPS.simMode && !RoadGraph.loading && RoadGraph.needsReload(fix.latitude, fix.longitude)) {
      RoadGraph.load(fix.latitude, fix.longitude);
    }
    if (!GPS.active) {
      GPS.active = true;
      S.gtLa = S.ndLa = S.iLa = fix.latitude;
      S.gtLo = S.ndLo = S.iLo = fix.longitude;
      EKF.x = fix.latitude; EKF.y = fix.longitude;
      S.totND = 0; S.n = 0; S.totEKF = 0; S.outN = 0; S.nhcN = 0; S.outt = 0;
      try {
        const lb = document.getElementById('btn-livegps');
        if (lb) { lb.textContent = '◉ Live'; lb.classList.add('on'); }
      } catch (e) {}
    }
  });
  NavdriftSensors.addListener('gnssStatus', (evt) => {
    SensorManager.setStatus('gnss', evt.status === 'permission-denied'
      ? SensorManager.STATUS.PERMISSION_DENIED : SensorManager.STATUS.NO_DATA, 'NavdriftSensors(native-gnss)');
    if (evt.status === 'permission-denied') GPS.hardError = true;
  });

  // ── IMU: accelerometer / gyroscope / orientation / magnetometer / barometer ──
  NavdriftSensors.addListener('imuAccel', (a) => {
    if (!isFinite(a.x) || !isFinite(a.y) || !isFinite(a.z)) return; // same guard as the web devicemotion handler
    if (!IMU.active) {
      IMU.active = true; IMU.state = 'ACTIVE'; IMU.firstEventAt = performance.now();
      try { const b = document.getElementById('imu-banner'); if (b) b.remove(); } catch (e) {}
    }
    IMU.evtCount++;
    // Native samples arrive already gravity-corrected when a.gravityRemoved is true (Android's
    // TYPE_LINEAR_ACCELERATION, iOS's CMDeviceMotion.userAcceleration) — same Butterworth filter
    // path the web code uses, so dead-reckoning math downstream is identical either way.
    const rawAx = a.x - IMU.axOff, rawAy = a.y - IMU.ayOff, rawAz = a.z - (a.gravityRemoved ? IMU.azOff : (IMU.azOff + 9.81));
    const fax = bw2(rawAx, _bw_xp[0], _bw_yp1[0], _bw_yp2[0]);
    const fay = bw2(rawAy, _bw_xp[1], _bw_yp1[1], _bw_yp2[1]);
    const faz = bw2(rawAz, _bw_xp[2], _bw_yp1[2], _bw_yp2[2]);
    _bw_yp2 = [_bw_yp1[0], _bw_yp1[1], _bw_yp1[2]];
    _bw_yp1 = [fax, fay, faz];
    _bw_xp = [rawAx, rawAy, rawAz];
    IMU.ax = fax; IMU.ay = fay; IMU.az = faz;
    IMU._gravFree = !!a.gravityRemoved;
    SensorManager.setValue('accelerometer', { x: fax, y: fay, z: faz }, null, 'NavdriftSensors(native-imu)');
    try {
      document.getElementById('sv-ax').textContent = fax.toFixed(2);
      document.getElementById('sv-ay').textContent = fay.toFixed(2);
      document.getElementById('sv-az').textContent = faz.toFixed(2);
    } catch (e) {}
  });

  NavdriftSensors.addListener('imuGyro', (g) => {
    if (!isFinite(g.x) || !isFinite(g.y) || !isFinite(g.z)) return;
    IMU.gx = g.x - IMU.gxOff; IMU.gy = g.y - IMU.gyOff; IMU.gz = g.z - IMU.gzOff;
    SensorManager.setValue('gyroscope', { x: IMU.gx, y: IMU.gy, z: IMU.gz }, null, 'NavdriftSensors(native-imu)');
    try {
      document.getElementById('sv-gx').textContent = IMU.gx.toFixed(3);
      document.getElementById('sv-gy').textContent = IMU.gy.toFixed(3);
      document.getElementById('sv-gz').textContent = IMU.gz.toFixed(3);
    } catch (e) {}
  });

  NavdriftSensors.addListener('orientationSample', (o) => {
    IMU.alpha = o.alpha || 0; IMU.beta = o.beta || 0; IMU.gamma = o.gamma || 0;
    SensorManager.setValue('orientation', { alpha: IMU.alpha, beta: IMU.beta, gamma: IMU.gamma }, null, 'NavdriftSensors(native-imu)');
    try {
      document.getElementById('sv-al').textContent = IMU.alpha.toFixed(1);
      document.getElementById('sv-be').textContent = IMU.beta.toFixed(1);
      document.getElementById('sv-ga').textContent = IMU.gamma.toFixed(1);
    } catch (e) {}
  });

  NavdriftSensors.addListener('magSample', (m) => {
    // Real heading from the OS's own sensor fusion (Android rotation vector / iOS CMDeviceMotion
    // heading) — the native equivalent of webkitCompassHeading / deviceorientationabsolute.
    SensorManager.setValue('magnetometer', { headingDeg: m.headingDeg }, null, 'NavdriftSensors(native-mag)');
  });

  NavdriftSensors.addListener('baroSample', (b) => {
    // Real hardware barometer via CMAltimeter (iOS) / TYPE_PRESSURE (Android) — this is exactly
    // the path that doesn't exist in a browser (WebKit blocks it on iOS entirely; see BaroSensor
    // in mobile.html). Feed S.baroAlt the same way BaroSensor.altitudeM does, so tunnel detection
    // and the Baro Alt tile behave identically to the web/active-barometer case.
    S.baroAlt = (typeof b.altitudeM === 'number') ? b.altitudeM
      : (typeof b.relativeAltitudeM === 'number' ? 220 + b.relativeAltitudeM : S.baroAlt);
    SensorManager.setValue('barometer', { altitudeM: S.baroAlt, pressureHPa: b.pressureHPa }, null, 'NavdriftSensors(native-baro)');
    try {
      const el = document.getElementById('v-baro');
      if (el) { el.textContent = S.baroAlt.toFixed(1) + ' m (live · native)'; el.style.color = 'var(--ng)'; }
    } catch (e) {}
  });

  // ── Kick everything off. Each call reports back which sensors the real hardware actually has
  // — never assumed, always read from the plugin's own honest per-device probe. ──
  NavdriftSensors.startGnss().catch((err) => {
    console.warn('[NativeBridge] startGnss failed:', err);
    SensorManager.setStatus('gnss', SensorManager.STATUS.PERMISSION_DENIED, 'NavdriftSensors(native-gnss)');
  });
  NavdriftSensors.startImu().then((avail) => {
    for (const key of ['accelerometer', 'gyroscope', 'orientation', 'magnetometer', 'barometer']) {
      if (avail && avail[key] === 'unavailable') {
        SensorManager.setStatus(key, SensorManager.STATUS.UNAVAILABLE, 'NavdriftSensors(native)');
      }
    }
  }).catch((err) => {
    console.warn('[NativeBridge] startImu failed:', err);
  });

  startHzTracker();
})();
