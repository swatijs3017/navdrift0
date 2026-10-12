// tests/js/live_hmm_viterbi.test.js
//
// Regression coverage for the bounded-lookback rolling-window Viterbi upgrade to LiveHMM
// (frontend/mobile.html, see docs/MAP_MATCHING_UPGRADE.md). Loads the REAL mobile.html code via
// jsdom (same pattern as tests/js/mobile_core.test.js) and exercises the REAL LiveHMM.update()
// implementation directly.
//
// Honesty note on what this does and does not prove (see the task's own acknowledgment that a
// full RoadGraph integration test is impractical to construct synthetically without fabricating
// data that doesn't exercise real code): RoadGraph.candidates() normally comes from a live
// Overpass fetch or an IndexedDB cache of real OSM data, neither available in this offline
// harness. Rather than fabricate a populated RoadGraph in a way that wouldn't actually exercise
// real candidate-generation logic, these tests override ONLY RoadGraph.candidates() (a plain,
// reassignable property on the object literal RoadGraph's IIFE returns -- not a frozen API) with
// an explicitly-labeled SYNTHETIC scripted candidate sequence, then drive the real, unmodified
// LiveHMM.update()/reset()/segIdx through it. This proves the real Viterbi scoring + backtrack
// logic inside LiveHMM behaves correctly given a controlled candidate stream; it does NOT prove
// RoadGraph.candidates() itself (the real grid ring-search over live/cached OSM segments) behaves
// correctly -- that remains unverified without a real device or a recorded real-world GNSS+OSM
// fixture (none was found under data/, eval/, or tests/ that pairs a road graph with a matching
// GPS track suitable for this harness; IO-VNBD-style datasets exist for other analyses in this
// repo but are not wired as a RoadGraph+track fixture).

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

// m -> degrees latitude, good enough at the few-tens-of-meters scale used here.
function mToLatDeg(m) { return m / 111320; }

test('LiveHMM (bounded-lookback Viterbi): discriminates a heading-aligned fork candidate from a closer-but-wrong-heading one', () => {
  const w = loadMobileWindow();
  try {
    assert.equal(typeof w.LiveHMM, 'undefined', 'LiveHMM is a top-level const, not a window property, in a real browser too');

    const baseLat = 12.9716, baseLon = 77.5946;
    // SYNTHETIC fork scenario: candidate A is the road the vehicle is actually on (heading-
    // aligned, bearing 0 == due north, matching fusedHeading=0 every tick). Candidate B is a
    // nearby near-parallel/crossing road that is slightly CLOSER in perpendicular distance but
    // whose bearing disagrees with the vehicle's real heading -- exactly the "parallel road /
    // service road / ramp" failure mode the design doc calls out. A naive nearest-distance-only
    // matcher would snap to B; LiveHMM's emission term must out-weigh that via heading agreement.
    const ticks = [];
    for (let t = 0; t < 8; t++) {
      ticks.push([
        { idx: 0, distM: 6, lat: baseLat + mToLatDeg(0.5 * t), lon: baseLon, bearing: 0 },
        { idx: 1, distM: 4, lat: baseLat + mToLatDeg(0.5 * t), lon: baseLon + 0.00002, bearing: Math.PI / 2 },
      ]);
    }

    const result = w.eval(`
      (function(){
        const ticks = ${JSON.stringify(ticks)};
        let i = -1;
        RoadGraph.candidates = function(lat, lon, k) { i++; return ticks[i]; };
        LiveHMM.reset();
        const reported = [];
        for (let t = 0; t < ticks.length; t++) {
          const r = LiveHMM.update(${baseLat}, ${baseLon}, 0 /* heading due north */, 5, 0.1);
          reported.push(r);
        }
        return JSON.stringify({ reported, finalSegIdx: LiveHMM.segIdx });
      })();
    `);
    const { reported, finalSegIdx } = JSON.parse(result);

    assert.equal(finalSegIdx, 0, 'LiveHMM must settle on the heading-aligned segment (idx 0), not the closer-but-wrong-heading one (idx 1)');
    for (let t = 0; t < reported.length; t++) {
      assert.equal(reported[t].bearing, 0, `tick ${t}: every tick's reported match must be the heading-aligned candidate (bearing 0), not the wrong-heading one (bearing PI/2)`);
    }
  } finally {
    w.close();
  }
});

test('LiveHMM (bounded-lookback Viterbi): hindsight backtrack on column eviction corrects an initial wrong greedy pick', () => {
  const w = loadMobileWindow();
  try {
    const baseLat = 12.9716, baseLon = 77.5946;

    // SYNTHETIC "ambiguous first tick, then unambiguous" scenario, purpose-built to separate the
    // new Viterbi path-accumulation behavior from the OLD pure-greedy algorithm this replaces:
    //   - Tick 0 (first tick post-reset -> backptr=-1, pure emission, no history to disambiguate
    //     with yet): candidate B is a plausible local pick (close, nearly-aligned heading) while
    //     the true road A is farther off at this instant -> a pure per-tick greedy argmax (the
    //     OLD LiveHMM behavior) picks B here, and there is nothing wrong with that decision in
    //     isolation.
    //   - Ticks 1-6: the real road is unambiguously A (perfectly heading-aligned, physically
    //     continuous with where A was) while B is now both badly heading-misaligned (perpendicular)
    //     and physically far from any plausible continuation -> every subsequent tick's emission
    //     and transition terms overwhelmingly favor a path that stayed on/returned to A.
    // Once the trellis window (W=6 in the implementation) fills and tick 0's column is evicted,
    // the lazy backtrack from the newest column's winner must resolve tick 0's segment as A, in
    // hindsight -- disagreeing with what was greedily reported for tick 0 at the time. This is
    // exactly the two-tier behavior from docs/MAP_MATCHING_UPGRADE.md section 3.3: the immediate
    // per-tick report for tick 0 (already consumed) is NOT rewritten, but Diag.roadMatchCorrectedSegIdx
    // must surface the hindsight-correct answer for use by future ticks.
    const farSouthLat = baseLat - mToLatDeg(80);
    const ticks = [
      [ // tick 0: B locally favored (close + only slightly off-heading); A locally weak (far perpendicular distance reading)
        { idx: 0, distM: 15, lat: baseLat, lon: baseLon, bearing: 0 },
        { idx: 1, distM: 1, lat: farSouthLat, lon: baseLon, bearing: 0.15 },
      ],
    ];
    for (let t = 1; t < 7; t++) {
      ticks.push([
        { idx: 0, distM: 2, lat: baseLat + mToLatDeg(0.5 * t), lon: baseLon, bearing: 0 },
        { idx: 1, distM: 2, lat: farSouthLat, lon: baseLon, bearing: Math.PI / 2 },
      ]);
    }

    const result = w.eval(`
      (function(){
        const ticks = ${JSON.stringify(ticks)};
        let i = -1;
        RoadGraph.candidates = function(lat, lon, k) { i++; return ticks[i]; };
        LiveHMM.reset();
        Diag.roadMatchCorrectedSegIdx = null; Diag.roadMatchBacktrackDeltaM = null;
        const reported = [];
        for (let t = 0; t < ticks.length; t++) {
          const r = LiveHMM.update(${baseLat}, ${baseLon}, 0 /* heading due north */, 5, 0.1);
          reported.push(r);
        }
        return JSON.stringify({
          reportedTick0Bearing: reported[0].bearing,
          correctedSegIdx: Diag.roadMatchCorrectedSegIdx,
          backtrackDeltaM: Diag.roadMatchBacktrackDeltaM,
        });
      })();
    `);
    const { reportedTick0Bearing, correctedSegIdx, backtrackDeltaM } = JSON.parse(result);

    // The immediate/greedy tier really did pick B (bearing 0.15) for tick 0 at the time --
    // confirming the test actually sets up the "initially wrong" premise, not a trivial case.
    assert.equal(reportedTick0Bearing, 0.15, 'tick 0 must have been greedily reported as candidate B (bearing 0.15) to make this a real test of hindsight correction');

    // Once the window evicts tick 0's column (7th update() call, trellis length W+1=7 > W=6),
    // the lazy backtrack must have resolved tick 0 as segment A (idx 0), disagreeing with the
    // greedy report above -- and must expose this as a real, non-fabricated diagnostic value.
    assert.equal(correctedSegIdx, 0, 'hindsight backtrack must identify segment A (idx 0) as the correct tick-0 match once later ticks disambiguate it');
    assert.ok(typeof backtrackDeltaM === 'number' && backtrackDeltaM > 0, `backtrackDeltaM must be a real positive correction distance, got ${backtrackDeltaM}`);
  } finally {
    w.close();
  }
});
