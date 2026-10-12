// tests/js/mobile_core.test.js
//
// Regression coverage for the ACTUAL JS code frontend/mobile.html ships to
// phones (ekfStep / EKF), run headlessly via jsdom. This is new coverage —
// before this file, zero automated tests exercised mobile.html's JS; all
// existing Python tests cover the separate, phone-unused navdrift_engine.py.
//
// Approach: load the real frontend/mobile.html into jsdom with
// runScripts:"dangerously" and NO external resource loading (so the
// maplibre-gl/onnxruntime-web <script src> tags are parsed but never
// fetched/executed — they are not needed to exercise ekfStep/EKF, and
// fetching them would require network access this environment may not
// have). ekfStep/ekfReset/dist are `function` declarations, so they land on
// `window` directly; EKF/LiveHMM/RoadGraph/S are top-level `const`, which in
// a real browser are NOT window properties either (only `var`/function
// declarations are) — they live in the shared global lexical scope instead,
// so they are reached the same way a second <script> tag on the real page
// would reach them: via `window.eval(...)`, not `window.EKF`. This mirrors
// actual browser script-scoping, not a jsdom quirk being worked around.
//
// Scope: this file only exercises ekfStep/EKF (dead reckoning + GPS fusion).
// LiveHMM.update() depends on a populated RoadGraph (live Overpass fetch or
// IndexedDB cache) to produce candidates, neither of which is available/
// meaningful offline in this harness without fabricating fake road data —
// doing so would test the fabricated data, not real behavior. See the
// accompanying audit notes for why LiveHMM itself is left for a harness
// that can supply a real or recorded RoadGraph fixture.
//
// Note: mobile.html's map-init code retries on a setTimeout loop (it can't
// reach maplibre-gl here, since external <script src> resources are
// intentionally not fetched). Each window is explicitly closed at the end
// of its test (w.close()) to tear down jsdom's pending timers so the test
// process can exit promptly instead of idling until those retries give up.

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

test('ekfStep: dead reckoning keeps updating position across a simulated GNSS-absent tick sequence (no freeze)', () => {
  const w = loadMobileWindow();
  try {
    assert.equal(typeof w.ekfStep, 'function', 'ekfStep must be defined on the real page');

    // Reset to a known origin, then run 20 ticks of pure dead reckoning
    // (gpsMeasured=false throughout, as during a real GNSS blackout) with a
    // constant forward speed and zero heading change, at a realistic dt.
    const states = w.eval(`
      ekfReset(12.9716, 77.5946);
      const out = [];
      for (let i = 0; i < 20; i++) {
        ekfStep(5.0 /* m/s */, 0.0 /* dh */, 0.1 /* dt s */, false, NaN, NaN, NaN);
        out.push({ x: EKF.x, y: EKF.y, th: EKF.th });
      }
      JSON.stringify(out);
    `);
    const parsed = JSON.parse(states);

    assert.equal(parsed.length, 20);

    // Every tick must actually move the estimate (dead reckoning must not
    // freeze just because GNSS is absent) and every value must stay finite
    // (a non-finite state would be the EKF breaking, not "holding still").
    for (let i = 0; i < parsed.length; i++) {
      assert.ok(Number.isFinite(parsed[i].x) && Number.isFinite(parsed[i].y),
        `tick ${i}: EKF state must stay finite during dead reckoning`);
    }
    for (let i = 1; i < parsed.length; i++) {
      const moved = parsed[i].x !== parsed[i - 1].x || parsed[i].y !== parsed[i - 1].y;
      assert.ok(moved, `tick ${i}: position must keep changing tick-to-tick while moving with no GPS (dead reckoning must not freeze)`);
    }

    // Net displacement over 20 ticks at 5 m/s * 0.1s = 10m total distance
    // should be reflected in the final position (not zero, not wildly off).
    const totalDistM = w.eval(`dist(${parsed[0].x}, ${parsed[0].y}, ${parsed[19].x}, ${parsed[19].y})`);
    assert.ok(totalDistM > 5 && totalDistM < 20,
      `expected roughly ~10m of dead-reckoned travel, got ${totalDistM}m`);
  } finally {
    w.close();
  }
});

test('ekfStep: GNSS reacquisition after a blackout produces a bounded correction, not a teleport', () => {
  const w = loadMobileWindow();
  try {
    // Simulate: start at a known point, dead-reckon for a while (accumulating
    // some drift/uncertainty growth exactly as real blackout handling does),
    // then supply a single fresh GPS fix placed a known, deliberately large
    // distance away from the current dead-reckoned estimate (as if real drift
    // had accumulated), and confirm the correction is partial (Kalman blend),
    // not an instantaneous snap to the GPS coordinate.
    const result = w.eval(`
      ekfReset(12.9716, 77.5946);
      for (let i = 0; i < 30; i++) {
        ekfStep(5.0, 0.02, 0.1, false, NaN, NaN, NaN); // 30 ticks of blackout dead reckoning
      }
      const preX = EKF.x, preY = EKF.y;

      // A GPS fix offset ~50m from the current dead-reckoned estimate in the
      // lat direction (50m ~= 0.00045 deg lat), simulating real accumulated
      // drift being corrected by a fresh, trustworthy fix.
      const gla = preX + 0.00045, glo = preY;
      ekfStep(5.0, 0.0, 0.1, true, gla, glo, 5.0 /* gpsAccM */);
      const postX = EKF.x, postY = EKF.y;

      JSON.stringify({ preX, preY, gla, glo, postX, postY });
    `);
    const { preX, preY, gla, glo, postX, postY } = JSON.parse(result);

    const distPreToGps = w.eval(`dist(${preX}, ${preY}, ${gla}, ${glo})`);
    const distPostToGps = w.eval(`dist(${postX}, ${postY}, ${gla}, ${glo})`);
    const distPreToPost = w.eval(`dist(${preX}, ${preY}, ${postX}, ${postY})`);

    assert.ok(distPreToGps > 10, 'the fix must actually be meaningfully offset from the dead-reckoned estimate for this test to be meaningful');

    // Bounded correction: the post-fix estimate must move TOWARD the GPS fix...
    assert.ok(distPostToGps < distPreToGps,
      `reacquisition must reduce the gap to the GPS fix (pre=${distPreToGps}m, post=${distPostToGps}m)`);
    // ...but must NOT teleport exactly onto it in one measurement update (a
    // single-fix snap-to-GPS would defeat the point of fusing dead reckoning
    // with GPS, and would look like a visible jump on the map).
    assert.ok(distPostToGps > 0.001,
      `a single GPS update must not snap the estimate exactly onto the fix (post-fix distance to fix was ${distPostToGps}m)`);
    assert.ok(distPreToPost < distPreToGps,
      `the correction step itself (pre-fix -> post-fix) must be smaller than jumping straight to the GPS fix, i.e. a blend not a teleport`);
  } finally {
    w.close();
  }
});
