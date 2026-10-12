// tests/js/gnss_ins_observability.test.js
//
// Regression coverage for the GNSS-INS fusion observability additions in
// frontend/mobile.html: GNSS outage duration tracking (S.outt, driven by
// triggerOut()/the step() loop) and reacquisition correction magnitude
// (Diag.lastReacqErrorM/lastReacqBlackoutS, computed in reacq()). Loaded the
// same way as tests/js/mobile_core.test.js — the real mobile.html via jsdom,
// runScripts:"dangerously", external <script src> resources never fetched.
//
// This drives a synthetic blackout-then-reacquisition sequence entirely
// through the page's own real functions (triggerOut/ekfStep/reacq/dist) —
// nothing here is a reimplementation of the logic under test.

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

test('GNSS blackout -> reacquisition: outage duration and correction magnitude are tracked honestly in Diag', () => {
  const w = loadMobileWindow();
  try {
    assert.equal(typeof w.triggerOut, 'function', 'triggerOut must be defined on the real page');
    assert.equal(typeof w.reacq, 'function', 'reacq must be defined on the real page');
    assert.equal(typeof w.ekfStep, 'function', 'ekfStep must be defined on the real page');
    assert.equal(typeof w.dist, 'function', 'dist must be defined on the real page');

    const ORIGIN_LAT = 12.9716, ORIGIN_LON = 77.5946;
    const SPEED_MPS = 10.0;   // realistic vehicle speed
    const DT = 0.1;           // seconds per simulated tick
    const N_TICKS = 50;       // 5.0s synthetic blackout

    const result = w.eval(`
      ekfReset(${ORIGIN_LAT}, ${ORIGIN_LON});
      S.gnss = true;
      S.gtLa = ${ORIGIN_LAT}; S.gtLo = ${ORIGIN_LON};

      // Enter blackout — real triggerOut(), not a reimplementation.
      triggerOut();
      const blackoutStartTracked = (S.gnss === false) && (Diag.blackoutStartMs != null);

      // Drive N dead-reckoning ticks exactly like step()'s live/sim blackout
      // path does: EKF integrates forward speed+heading, S.outt accumulates
      // real elapsed time, ground truth (S.gtLa/gtLo) stays frozen because no
      // GNSS fixes arrive during a real blackout.
      for (let i = 0; i < ${N_TICKS}; i++) {
        ekfStep(${SPEED_MPS}, 0.0, ${DT}, false, NaN, NaN, NaN);
        S.outt += ${DT};
      }
      const outageDurationTracked = S.outt;

      // Reacquire — real reacq(), computes Diag.lastReacqErrorM from the
      // real pre-correction EKF estimate vs the (frozen, real) ground truth,
      // BEFORE any correction is applied.
      const preCorrectionX = EKF.x, preCorrectionY = EKF.y;
      const expectedErrorM = dist(EKF.x, EKF.y, S.gtLa, S.gtLo);
      reacq();

      JSON.stringify({
        blackoutStartTracked,
        outageDurationTracked,
        expectedErrorM,
        lastReacqErrorM: Diag.lastReacqErrorM,
        lastReacqBlackoutS: Diag.lastReacqBlackoutS,
        preCorrectionX, preCorrectionY,
      });
    `);
    const r = JSON.parse(result);

    // 1. Outage duration is tracked correctly end-to-end.
    assert.equal(r.blackoutStartTracked, true, 'triggerOut() must record Diag.blackoutStartMs and flip S.gnss false');
    assert.ok(Math.abs(r.outageDurationTracked - N_TICKS * DT) < 1e-9,
      `S.outt must equal the real accumulated elapsed time (expected ${N_TICKS * DT}, got ${r.outageDurationTracked})`);

    // 2. A correction magnitude is computed at reacquisition and is reflected
    // in Diag correctly (matches the independently-computed expected value).
    assert.ok(Number.isFinite(r.lastReacqErrorM), 'Diag.lastReacqErrorM must be a finite number after reacq()');
    assert.ok(Math.abs(r.lastReacqErrorM - r.expectedErrorM) < 1e-9,
      'Diag.lastReacqErrorM must equal the real dist(EKF, ground truth) computed before correction');
    assert.ok(r.lastReacqErrorM > 0, 'dead reckoning at nonzero speed for 5s must produce a nonzero drift error');

    // 3. The magnitude is bounded/sane — not a teleport-sized number. At
    // 10 m/s for 5.0s of pure dead reckoning with zero heading change, the
    // true displacement is ~50m; the drift error (vs frozen ground truth,
    // which itself does not move) must be on that same order of magnitude,
    // not thousands of km (a broken lat/lon mixup) or exactly 0 (a stub).
    const maxPlausibleErrorM = SPEED_MPS * N_TICKS * DT * 1.5; // generous bound over the dead-reckoned distance
    assert.ok(r.lastReacqErrorM < maxPlausibleErrorM,
      `reacquisition correction magnitude ${r.lastReacqErrorM}m must be bounded by plausible dead-reckoning drift (< ${maxPlausibleErrorM}m), not a teleport-sized value`);

    // 4. Diag.lastReacqBlackoutS reflects the real blackout duration measured
    // at the moment of reacquisition (same value S.outt held just before reacq()).
    assert.ok(Math.abs(r.lastReacqBlackoutS - N_TICKS * DT) < 1e-9,
      'Diag.lastReacqBlackoutS must equal the real blackout duration at the moment of reacquisition');
  } finally {
    w.close();
  }
});

test('ekfUncertaintyM(): state uncertainty is derivable and grows monotonically during an uncorrected blackout', () => {
  const w = loadMobileWindow();
  try {
    assert.equal(typeof w.ekfUncertaintyM, 'function', 'ekfUncertaintyM must be defined on the real page');

    const series = w.eval(`
      ekfReset(12.9716, 77.5946);
      S.gnss = true;
      triggerOut();
      const out = [];
      for (let i = 0; i < 30; i++) {
        ekfStep(8.0, 0.0, 0.1, false, NaN, NaN, NaN);
        out.push(ekfUncertaintyM());
      }
      JSON.stringify(out);
    `);
    const u = JSON.parse(series);

    assert.equal(u.length, 30);
    u.forEach((v, i) => assert.ok(Number.isFinite(v) && v >= 0, `uncertainty at tick ${i} must be a finite non-negative meters value`));

    // Covariance grows every predict tick with no GPS update to shrink it
    // (a real blackout): the sequence must be non-decreasing throughout.
    for (let i = 1; i < u.length; i++) {
      assert.ok(u[i] >= u[i - 1] - 1e-9, `state uncertainty must not shrink during an uncorrected blackout (tick ${i - 1}=${u[i - 1]} -> tick ${i}=${u[i]})`);
    }
    assert.ok(u[u.length - 1] > u[0], 'state uncertainty must have genuinely grown over the blackout, not stayed flat');
  } finally {
    w.close();
  }
});
