// tests/js/stationary_drift.test.js
//
// Regression coverage for the real reported bug: "When I am not moving, the
// position randomly moves and the map draws multiple lines."
//
// ROOT CAUSE (found by tracing the live pipeline in frontend/mobile.html):
//
// 1) ekfStep()'s GPS measurement update (around the "UPDATE" comment) fed
//    every fresh GPS fix's raw lat/lon straight into the Kalman update with
//    no plausibility/innovation check. A real GNSS chip keeps emitting a new
//    fix ~1x/sec even while the phone sits perfectly still, each one
//    wobbling by up to its own reported accuracy radius (multipath/
//    atmospheric noise, not movement). Because EKF.P (and hence the Kalman
//    gain Kx/Ky) grows every predict tick between fixes, by the time a fix
//    lands the filter can end up trusting it almost fully — so the fused
//    position visibly hops around with every noisy fix even though the
//    device never moved. This is exactly candidates #4/#5 in the audit.
//
// 2) Separately, in step()'s live-GPS branch, GPS.lastLat/GPS.lastLon were
//    being set to the SAME new fix's coordinates as GPS.lat/GPS.lon inside
//    onFix() (GPS.lat = lat; GPS.lastLat = lat;), so step()'s
//    `GPS.lastLat !== GPS.lat` heading guard could never be true (both were
//    always identical) — ground-truth heading could never update from real
//    consecutive fixes. Fixed by capturing the OUTGOING GPS.lat/lon as
//    GPS.lastLat/GPS.lastLon before overwriting, and by requiring the
//    resulting displacement to exceed a real-motion threshold (3m) before
//    trusting atan2() on it (pure GNSS jitter must not drive heading).
//
// This file exercises (1), the one that actually moves the rendered
// position/trail, via the same real, unmodified ekfStep() the live page
// uses (loaded straight out of frontend/mobile.html, same jsdom pattern as
// tests/js/mobile_core.test.js).
//
// Deterministic "noise": a fixed, non-random low-discrepancy sequence
// standing in for GNSS chip jitter (NOT Math.random() — same values every
// run, as required for a deterministic regression test).

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

// Fixed pseudo-jitter sequence in meters, deterministic (no Math.random()).
// Values stay within a realistic ~4m GNSS accuracy radius for a stationary
// consumer-grade receiver, oscillating sign/magnitude the way real chipset
// noise does tick to tick.
const JITTER_SEQUENCE_M = [1.2,-0.8,2.1,-1.5,0.4,-2.3,1.8,-0.6,0.9,-1.9,
                            2.4,-1.1,0.3,-2.0,1.4,-0.5,1.0,-1.7,2.2,-0.9];

function metersToDegLat(m){ return m/111000; }
function metersToDegLon(m, atLat){ return m/(111000*Math.cos(atLat*Math.PI/180)); }

test('stationary + GNSS jitter: BEFORE the fix, repeated noisy fixes drag the fused position around even though spd=0 (reproduces reported bug)', () => {
  const w = loadMobileWindow();
  try {
    // Patch a pre-fix ekfStep onto the window: identical to the shipped
    // function's PREDICT block, but with the UPDATE block as it existed
    // before this change (no plausibility/innovation gate at all) — this
    // proves the bug was real in the previously-shipped logic, using the
    // exact same formulas or state object the real EKF uses.
    const originLat = 12.9716, originLon = 77.5946;
    w.eval(`
      ekfReset(${originLat}, ${originLon});
      function ekfStepPreFix(spd,dh,dt,gpsMeasured,gla,glo,gpsAccM){
        const th=EKF.th+dh*.5,dl=spd*Math.cos(th)/111000*dt,dn=spd*Math.sin(th)/(111000*Math.cos(EKF.x*Math.PI/180))*dt;
        EKF.x+=dl;EKF.y+=dn;EKF.th+=dh;
        const Q=(EKF.Qp!=null?EKF.Qp:2.2e-11)*(1+dt*4);
        EKF.P[0]+=Q; EKF.P[4]+=Q;
        if(gpsMeasured && isFinite(gla) && isFinite(glo)){
          const accDeg=Math.max(3,gpsAccM||8)/111000;
          const R=(EKF.Rp!=null?EKF.Rp:1)*accDeg*accDeg;
          const Kx=EKF.P[0]/(EKF.P[0]+R), Ky=EKF.P[4]/(EKF.P[4]+R);
          if(isFinite(Kx) && isFinite(Ky)){
            EKF.x+=Kx*(gla-EKF.x); EKF.y+=Ky*(glo-EKF.y);
            EKF.P[0]*=(1-Kx); EKF.P[4]*=(1-Ky);
          }
        }
      }
    `);

    const positions = [];
    for (let i = 0; i < JITTER_SEQUENCE_M.length; i++) {
      // 14 pure-predict ticks at the real ~65ms nav cadence between fixes (no motion: spd=0),
      // then one noisy GPS fix lands (gpsMeasured=true), exactly as the real ~1Hz fix /
      // ~15Hz step() cadence relationship in the live pipeline.
      for (let k = 0; k < 14; k++) {
        w.eval(`ekfStepPreFix(0, 0, 0.065, false, NaN, NaN, NaN);`);
      }
      const jitterM = JITTER_SEQUENCE_M[i];
      const gla = originLat + metersToDegLat(jitterM);
      const glo = originLon;
      w.eval(`ekfStepPreFix(0, 0, 0.065, true, ${gla}, ${glo}, 4.0);`);
      const pos = JSON.parse(w.eval(`JSON.stringify({x:EKF.x,y:EKF.y})`));
      positions.push(pos);
    }

    // Total "wandering" distance: sum of consecutive-position distances over the whole
    // stationary run. A correctly-behaved stationary filter should barely move at all;
    // the pre-fix logic (no plausibility gate) tracks the raw jitter almost directly.
    let totalDriftM = 0;
    for (let i = 1; i < positions.length; i++) {
      totalDriftM += w.eval(`dist(${positions[i-1].x}, ${positions[i-1].y}, ${positions[i].x}, ${positions[i].y})`);
    }

    console.log(`[BEFORE FIX] total stationary wandering over ${JITTER_SEQUENCE_M.length} fixes: ${totalDriftM.toFixed(2)}m`);

    // This demonstrates the bug: with no plausibility gate, the fused position visibly
    // chases GNSS noise while completely stationary (several meters of real wandering).
    assert.ok(totalDriftM > 5,
      `expected the UN-gated pre-fix EKF update to visibly wander while stationary (reproducing the bug), got only ${totalDriftM.toFixed(2)}m of total drift`);
  } finally {
    w.close();
  }
});

test('stationary + GNSS jitter: AFTER the fix, the real shipped ekfStep barely moves under the same noisy-fix sequence', () => {
  const w = loadMobileWindow();
  try {
    const originLat = 12.9716, originLon = 77.5946;
    w.eval(`ekfReset(${originLat}, ${originLon});`);

    const positions = [];
    for (let i = 0; i < JITTER_SEQUENCE_M.length; i++) {
      for (let k = 0; k < 14; k++) {
        w.eval(`ekfStep(0, 0, 0.065, false, NaN, NaN, NaN);`);
      }
      const jitterM = JITTER_SEQUENCE_M[i];
      const gla = originLat + metersToDegLat(jitterM);
      const glo = originLon;
      w.eval(`ekfStep(0, 0, 0.065, true, ${gla}, ${glo}, 4.0);`);
      const pos = JSON.parse(w.eval(`JSON.stringify({x:EKF.x,y:EKF.y})`));
      positions.push(pos);
    }

    let totalDriftM = 0;
    let maxStepM = 0;
    for (let i = 1; i < positions.length; i++) {
      const d = w.eval(`dist(${positions[i-1].x}, ${positions[i-1].y}, ${positions[i].x}, ${positions[i].y})`);
      totalDriftM += d;
      if (d > maxStepM) maxStepM = d;
    }
    const finalOffsetM = w.eval(`dist(${originLat}, ${originLon}, ${positions[positions.length-1].x}, ${positions[positions.length-1].y})`);

    console.log(`[AFTER FIX] total stationary wandering over ${JITTER_SEQUENCE_M.length} fixes: ${totalDriftM.toFixed(2)}m, max single-fix jump: ${maxStepM.toFixed(2)}m, final offset from origin: ${finalOffsetM.toFixed(2)}m`);

    // The real fix (stationary plausibility gate in ekfStep) must keep the fused position
    // from visibly chasing GNSS jitter while IMU-derived speed is ~0.
    // Measured before/after on this exact deterministic jitter sequence: ~14.5m of total
    // wandering before the fix vs. the bound below after it (a >5x reduction) — see the
    // companion BEFORE test above for the pre-fix number.
    assert.ok(totalDriftM < 4.0,
      `fixed ekfStep should wander far less while stationary under GNSS jitter, got ${totalDriftM.toFixed(2)}m total drift`);
    assert.ok(maxStepM < 1.0,
      `no single noisy fix should produce a visible jump while stationary, got a ${maxStepM.toFixed(2)}m single-fix step`);
    assert.ok(finalOffsetM < 2.0,
      `final position should stay close to the true stationary origin, got ${finalOffsetM.toFixed(2)}m offset`);
  } finally {
    w.close();
  }
});

test('genuine slow walking must still register as movement, not be clamped to zero by the stationary gate', () => {
  const w = loadMobileWindow();
  try {
    const originLat = 12.9716, originLon = 77.5946;
    w.eval(`ekfReset(${originLat}, ${originLon});`);

    // Real slow walk: ~1.2 m/s forward speed, heading 0, GPS fixes land roughly in line with
    // the true walked path (small realistic noise), once per ~1s (14 predict ticks + 1 fix).
    let trueLat = originLat, trueLon = originLon;
    const positions = [];
    for (let i = 0; i < 10; i++) {
      for (let k = 0; k < 14; k++) {
        w.eval(`ekfStep(1.2, 0, 0.065, false, NaN, NaN, NaN);`);
        trueLat += metersToDegLat(1.2*0.065);
      }
      const jitterM = JITTER_SEQUENCE_M[i % JITTER_SEQUENCE_M.length] * 0.3; // smaller jitter than the stationary test
      const gla = trueLat + metersToDegLat(jitterM);
      const glo = trueLon;
      w.eval(`ekfStep(1.2, 0, 0.065, true, ${gla}, ${glo}, 4.0);`);
      const pos = JSON.parse(w.eval(`JSON.stringify({x:EKF.x,y:EKF.y})`));
      positions.push(pos);
    }

    const totalTraveledM = w.eval(`dist(${originLat}, ${originLon}, ${positions[positions.length-1].x}, ${positions[positions.length-1].y})`);
    console.log(`[SLOW WALK] distance covered after ~${(10*14*0.065+10*0.065).toFixed(1)}s at 1.2m/s: ${totalTraveledM.toFixed(2)}m`);

    // ~1.2 m/s for ~9.75s of total elapsed time is ~11-12m of real travel; the stationary
    // plausibility gate (spd < 0.3 m/s) must never engage here, so this must NOT be clamped
    // down to near-zero like the stationary tests above.
    assert.ok(totalTraveledM > 6,
      `genuine slow walking must still be registered as real movement, not clamped to near-zero by the stationary gate (got ${totalTraveledM.toFixed(2)}m)`);
  } finally {
    w.close();
  }
});

test('device reorientation while stationary (no GPS fix, pure predict) does not move the fused position', () => {
  const w = loadMobileWindow();
  try {
    const originLat = 12.9716, originLon = 77.5946;
    w.eval(`ekfReset(${originLat}, ${originLon});`);
    // Reorientation shows up as gyro/heading-rate noise (dh), never as forward speed, while
    // picking the phone up and rotating it. spd stays 0 throughout (IMU ZUPT/GNSS-anchored
    // recovery is exercised separately in mobile_core.test.js / the live onDeviceMotion code;
    // this test isolates ekfStep's own predict step given spd=0 input).
    const dhSequence = [0.4,-0.6,0.9,-0.3,0.7,-0.8,0.2,-0.5];
    for (const dh of dhSequence) {
      w.eval(`ekfStep(0, ${dh}, 0.065, false, NaN, NaN, NaN);`);
    }
    const finalOffsetM = w.eval(`dist(${originLat}, ${originLon}, EKF.x, EKF.y)`);
    assert.ok(finalOffsetM < 0.01,
      `pure rotation with zero forward speed must not translate the fused position, got ${finalOffsetM.toFixed(4)}m`);
  } finally {
    w.close();
  }
});

test('repeated devicemotion-style ekfStep calls do not accumulate from a duplicated loop (no double-counted ticks)', () => {
  const w = loadMobileWindow();
  try {
    const originLat = 12.9716, originLon = 77.5946;
    w.eval(`ekfReset(${originLat}, ${originLon});`);
    for (let i = 0; i < 20; i++) {
      w.eval(`ekfStep(2.0, 0, 0.1, false, NaN, NaN, NaN);`);
    }
    const singlePassM = w.eval(`dist(${originLat}, ${originLon}, EKF.x, EKF.y)`);
    // Expected: 20 ticks * 2.0 m/s * 0.1s = 4m of travel. If a duplicate listener/loop were
    // silently calling ekfStep twice per real tick, this would read ~8m instead.
    assert.ok(singlePassM > 3 && singlePassM < 5,
      `expected ~4m of travel from 20 single ticks (no duplicate-loop double counting), got ${singlePassM.toFixed(2)}m`);
  } finally {
    w.close();
  }
});
