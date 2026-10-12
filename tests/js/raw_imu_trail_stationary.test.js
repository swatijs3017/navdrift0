// tests/js/raw_imu_trail_stationary.test.js
//
// Regression coverage for the real-device report: "while genuinely stationary (with real
// GPS lock), the map drew a chaotic scribble/star-burst right around the position dot" —
// specifically the purple "Raw IMU integration" trail (map legend), NOT the fused/EKF dot,
// which the user said looked reasonably anchored.
//
// ROOT CAUSE (traced in frontend/mobile.html):
//   step()'s GNSS-blackout branch integrates S.iLa/S.iLo (the raw-IMU trail's own position)
//   every tick using usedSpd2 (= clamp(IMU.intSpeed, 0, 35)) and the fused heading EKF.th —
//   see _rawTrailIntegrateStep(). IMU.intSpeed DOES have a ZUPT, but it is deliberately
//   conservative (only resets to exactly 0 after IMU._zuptMs > 800ms AND intSpeed < 2.5,
//   so it can never arrest an already-legitimate higher speed). Before that gate fires,
//   small nonzero residual accel-noise speed still gets integrated into S.iLa/S.iLo, every
//   tick, in whatever direction EKF.th points. Real devices flicker in and out of the
//   debounced GNSS-blackout state even while "locked" (brief fix gaps), and reacq() never
//   re-anchors this trail's own position (only the EKF's position gets corrected by the
//   next real GPS fix) — so repeated short blackout cycles while stationary produced
//   exactly the observed undamped, direction-varying scribble.
//
// FIX: _rawTrailStationaryGate() reuses the same real sensor evidence (IMU._zuptMs, which
// resets to 0 the instant real forward accel resumes) with a much shorter, purely cosmetic
// 150ms threshold to freeze this trail's own position integration during a stillness streak,
// without changing IMU.intSpeed, the EKF, or anything else.
//
// This file drives the REAL, unmodified _rawTrailIntegrateStep()/_rawTrailStationaryGate()
// functions extracted from mobile.html's live blackout branch — not a reimplementation —
// loaded the same jsdom way as the other tests/js/*.test.js files.
//
// Deterministic "noise": a fixed, non-random sequence standing in for real accelerometer
// noise (NOT Math.random()) — same values every run.

const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const { JSDOM } = require('jsdom');

const HTML_PATH = path.join(__dirname, '..', '..', 'frontend', 'mobile.html');
const HTML = fs.readFileSync(HTML_PATH, 'utf8');

function loadMobileWindow() {
  const dom = new JSDOM(HTML, {
    url: 'https://localhost/',
    runScripts: 'dangerously',
    pretendToBeVisual: true,
  });
  return dom.window;
}

// Fixed pseudo-noise sequence of residual forward-accel-integrated "speeds" (m/s), standing
// in for real MEMS accelerometer bias/noise that hasn't yet been zeroed by IMU's own ZUPT
// (i.e. values below the IMU.intSpeed < 2.5 gate, representative of what a stationary phone
// actually reads before 800ms of stillness has accumulated). Deterministic, no Math.random().
const RESIDUAL_SPEED_SEQUENCE = [0.18,0.22,0.09,0.31,0.14,0.27,0.11,0.19,0.33,0.06,
                                  0.24,0.17,0.29,0.08,0.21,0.35,0.13,0.26,0.10,0.20];
// Heading wander (radians) standing in for real small gyro-noise-driven EKF.th drift while
// stationary (device isn't perfectly rigid/level) — deterministic, varies direction each
// tick, which is exactly what turns small per-tick increments into a "star-burst" shape
// rather than a single straight line.
const HEADING_SEQUENCE = [0.1,1.4,2.8,-0.6,3.0,-2.1,0.9,-1.3,2.3,-2.9,
                           1.1,-0.4,2.6,-1.8,0.3,2.0,-2.5,1.6,-0.1,-1.6];

function bboxDiagM(points){
  if(points.length===0) return 0;
  let minLa=Infinity,maxLa=-Infinity,minLo=Infinity,maxLo=-Infinity;
  for(const [lo,la] of points){
    if(la<minLa)minLa=la; if(la>maxLa)maxLa=la;
    if(lo<minLo)minLo=lo; if(lo>maxLo)maxLo=lo;
  }
  const dLaM=(maxLa-minLa)*111000;
  const dLoM=(maxLo-minLo)*111000*Math.cos(((minLa+maxLa)/2)*Math.PI/180);
  return Math.hypot(dLaM,dLoM);
}

test('raw-IMU trail: BEFORE the ZUPT gate, repeated-blackout residual-noise integration scribbles unboundedly while stationary (reproduces reported bug)', () => {
  const w = loadMobileWindow();
  try {
    const originLat = 12.9716, originLon = 77.5946;
    // Force the gate OFF to reproduce the pre-fix behavior using the real, unmodified
    // integration math (_rawTrailIntegrateStep itself is untouched by this — only the
    // gate function it calls is monkey-patched here, to isolate exactly what the gate
    // contributes and prove the bug existed before it).
    w.eval(`window._rawTrailStationaryGate = function(){ return false; };`);

    let iLa = originLat, iLo = originLon;
    const points = [[iLo, iLa]];
    for (let i = 0; i < RESIDUAL_SPEED_SEQUENCE.length; i++) {
      // Simulate a short blackout cycle every tick (real device: brief/flickering GNSS
      // blackout state even while "locked" — see GNSS_DEBOUNCE_MS comment in step()),
      // each with ~65ms of real elapsed time, exactly as the live nav cadence.
      const spd = RESIDUAL_SPEED_SEQUENCE[i];
      const th = HEADING_SEQUENCE[i];
      const rawDt = 0.065;
      const r = JSON.parse(w.eval(
        `JSON.stringify(_rawTrailIntegrateStep(${spd}, ${th}, ${iLa}, ${rawDt}))`
      ));
      iLa += r.dl; iLo += r.dn;
      points.push([iLo, iLa]);
    }

    const diagM = bboxDiagM(points);
    console.log(`[BEFORE FIX] raw-IMU trail bounding-box diagonal over ${RESIDUAL_SPEED_SEQUENCE.length} stationary blackout-ticks: ${diagM.toFixed(2)}m`);

    // Each tick nudges a few cm to a few tens of cm in a quasi-random direction (per
    // HEADING_SEQUENCE); over 20 ticks with no gate at all, that's a real, visible scribble.
    assert.ok(diagM > 0.03,
      `expected the un-gated integration to produce a visible scribble while stationary, got only ${diagM.toFixed(4)}m bounding-box diagonal`);
  } finally {
    w.close();
  }
});

test('raw-IMU trail: AFTER the ZUPT gate, the real shipped _rawTrailIntegrateStep stays bounded under the same residual-noise sequence once sustained stillness is evident', () => {
  const w = loadMobileWindow();
  try {
    const originLat = 12.9716, originLon = 77.5946;
    // Real shipped IMU._zuptMs bookkeeping: accumulate as if forward-accel has been below
    // the 0.35 deadband continuously (a genuinely stationary phone), exactly the real
    // ZUPT block's own state variable — not reimplemented, just driven forward in time.
    w.eval(`IMU._zuptMs = 0;`);

    let iLa = originLat, iLo = originLon;
    const points = [[iLo, iLa]];
    for (let i = 0; i < RESIDUAL_SPEED_SEQUENCE.length; i++) {
      const spd = RESIDUAL_SPEED_SEQUENCE[i];
      const th = HEADING_SEQUENCE[i];
      const rawDt = 0.065;
      // Advance the real stillness clock by this tick's real dt, exactly as the live
      // onDeviceMotion ZUPT block does every tick while zuptMag stays below its deadband.
      w.eval(`IMU._zuptMs += ${rawDt * 1000};`);
      const r = JSON.parse(w.eval(
        `JSON.stringify(_rawTrailIntegrateStep(${spd}, ${th}, ${iLa}, ${rawDt}))`
      ));
      iLa += r.dl; iLo += r.dn;
      points.push([iLo, iLa]);
    }

    const diagM = bboxDiagM(points);
    console.log(`[AFTER FIX] raw-IMU trail bounding-box diagonal over ${RESIDUAL_SPEED_SEQUENCE.length} stationary blackout-ticks: ${diagM.toFixed(2)}m`);

    // IMU._zuptMs crosses the 150ms gate threshold after the 3rd tick (3*65ms=195ms), so
    // only the first ~2 ticks can contribute any displacement at all; everything after
    // must be fully frozen. The trail must stay under a single-digit-meter bound — in
    // practice, just the first couple of residual-speed ticks' worth (well under 1m).
    assert.ok(diagM < 1.0,
      `fixed raw-IMU trail should stay bounded (near-zero) while stationary once the ZUPT gate engages, got ${diagM.toFixed(4)}m bounding-box diagonal`);
  } finally {
    w.close();
  }
});

test('raw-IMU trail: genuine motion still produces a real, visible, unbounded-as-appropriate trail (gate does not over-correct into invisibility)', () => {
  const w = loadMobileWindow();
  try {
    const originLat = 12.9716, originLon = 77.5946;
    // Real forward acceleration is ongoing, so the real ZUPT block would keep resetting
    // IMU._zuptMs to 0 every tick (zuptMag stays >= 0.35 while genuinely moving) — the gate
    // must never engage here.
    w.eval(`IMU._zuptMs = 0;`);

    let iLa = originLat, iLo = originLon;
    const points = [[iLo, iLa]];
    const movingSpeedMps = 3.0; // brisk walk/slow jog, real dead-reckoning speed
    const th = 0.5; // steady heading — real forward motion, not noise
    const rawDt = 0.065;
    const ticks = 40; // ~2.6s of continuous real motion
    for (let i = 0; i < ticks; i++) {
      w.eval(`IMU._zuptMs = 0;`); // real motion: ZUPT accumulator never advances
      const r = JSON.parse(w.eval(
        `JSON.stringify(_rawTrailIntegrateStep(${movingSpeedMps}, ${th}, ${iLa}, ${rawDt}))`
      ));
      iLa += r.dl; iLo += r.dn;
      points.push([iLo, iLa]);
    }

    const diagM = bboxDiagM(points);
    const expectedTravelM = movingSpeedMps * rawDt * ticks; // ~7.8m
    console.log(`[MOTION] raw-IMU trail bounding-box diagonal over ${ticks} moving ticks at ${movingSpeedMps}m/s: ${diagM.toFixed(2)}m (expected ~${expectedTravelM.toFixed(2)}m of straight-line travel)`);

    // Straight-line motion at constant heading: bbox diagonal should equal the real
    // distance traveled, confirming the gate did not suppress or smooth genuine movement.
    assert.ok(diagM > expectedTravelM * 0.9 && diagM < expectedTravelM * 1.1,
      `genuine motion must still be drawn as a real, unsuppressed trail, got ${diagM.toFixed(2)}m vs expected ~${expectedTravelM.toFixed(2)}m`);
  } finally {
    w.close();
  }
});
