# Map-matching upgrade path: greedy LiveHMM -> bounded-lookback Viterbi

Status: design only. Nothing in this document has been implemented in
`frontend/mobile.html`. It describes exactly what would need to change to
give the live phone pipeline a small amount of backtracking, modeled after
the algorithm already proven offline in `edge/hmm_map_matcher.py`
(`ViterbiMapMatcher`), without turning it into that module's unbounded-batch
design.

## 1. What LiveHMM does today (baseline)

`LiveHMM` (frontend/mobile.html, ~line 4109) is a single-state, greedy,
per-tick selector:

- Each tick it asks `RoadGraph.candidates(fusedLat, fusedLon, K)` for up to
  `K=5` nearby segments.
- It scores each candidate with `emissionLogProb(cand, fusedHeading)` +
  `transitionLogProb(cand, speedMps, dtS)`, where the transition term
  compares the candidate against exactly one piece of state: `state`, the
  single previously *selected* segment (idx/lat/lon/bearing).
- It keeps only the argmax (`best`) and overwrites `state` with it.
- Nothing before the previous tick influences this tick's decision, and a
  bad local decision at tick N can never be corrected retroactively — there
  is no accumulated path score and no backpointer array, by the module's own
  header comment.

This is cheap: one argmax over <=5 candidates per tick, O(K) work, no
retained history beyond one previous selection.

## 2. What the offline module does (the other extreme)

`ViterbiMapMatcher` in `edge/hmm_map_matcher.py` keeps a trellis of the last
`history_window` columns (default 50), each column holding every candidate
considered at that step with an accumulated best-path score and a
backpointer into the previous column. `best_path()` does a full backward
traceback over the whole retained window. This is the correct, textbook
bounded-history Viterbi, but it is designed to be invoked as a batch
(`run_offline_match`) or stepped externally (`matcher.step(...)` /
`matcher.best_path()` called whenever the caller wants an answer) — it is
not wired to run as part of a 10 Hz live loop, and `best_path()`'s full
retrace is O(window x K) every time it's called, which is fine occasionally
but wasteful every single tick.

## 3. Proposed middle ground: bounded-lookback Viterbi for LiveHMM

Keep a short rolling trellis (`W` = 5-8 ticks, NOT 50) of
`{candidate, score, backptr}` columns, exactly like `_TrellisNode` in the
offline module, but:

- Only retrace the oldest column's winner once the window is full and
  about to evict it (an "amortized" or "lazy" traceback), not the whole
  window every tick.
- Reuse the exact same `emissionLogProb` / `transitionLogProb` formulas
  LiveHMM already has (same sigmas, same stay-bonus) — this is a structural
  change (keep more than one hypothesis, keep more than one step of
  history), not a re-tuning of the scoring model.

### 3.1 Data structures to add

```js
// Replace LiveHMM's single `state` with:
let trellis = [];      // ring buffer, length <= W, each entry:
                        //   { cands: [{idx,lat,lon,bearing,distM}, ...],
                        //     scores: Float64Array(cands.length),   // accumulated path score ending here
                        //     backptr: Int16Array(cands.length) }   // index into previous entry's cands, or -1
let committed = null;  // the segment LiveHMM has actually reported as `state` so far
                        // (what RoadGraph/NHC/the UI currently read) -- see 3.3
const W = 6;            // bounded lookback; tune per the budget in section 4
```

`RoadGraph.epoch` invalidation (already present) still fully resets
`trellis = []` and `committed = null`, same as today's `reset()`.

### 3.2 Per-tick update (replaces the current single-argmax block)

For each tick's `cands = RoadGraph.candidates(...)`:

1. Compute `emissionLogProb` for each candidate (unchanged).
2. If `trellis.length === 0` (first tick / post-reset): `scores[i] = emit[i]`,
   `backptr[i] = -1` for all i.
3. Else, for each candidate `i` in the new column, take the max over all
   candidates `j` in `trellis[trellis.length-1]` of
   `trellis[last].scores[j] + transitionLogProb(trellis[last].cands[j], cand_i, speed, dt) + emit[i]`.
   Record the argmax as `backptr[i]`.
4. Push the new column. If `trellis.length > W`, evict the oldest column
   — but first resolve what it decided (step 3.3) before dropping it.

This step is identical in spirit to `ViterbiMapMatcher.step()`, just capped
at `W` columns instead of 50, and the K is the same small `K=5` LiveHMM
already uses — so the per-tick cost is `O(W_inner * K^2)` only for the
**newest column's transition step** (`K x K = 25` score evaluations), which
is what step 3 above actually does (step 2/4 are O(K) or O(1)).

### 3.3 What LiveHMM actually reports each tick (the live-ness requirement)

The live pipeline (RoadGraph/NHC/soft road-pull, `Diag.roadMatchActive`,
etc.) needs a segment decision *every tick*, not just once every `W` ticks
when a traceback resolves. Two tiers:

- **Immediate/greedy report (every tick, unchanged latency):** continue
  reporting the current column's local argmax exactly as today — this is
  what `update()` returns and what downstream code (NHC, UI) consumes. No
  consumer-facing behavior changes tick to tick.
- **Corrected/backtracked report (lags by up to `W-1` ticks):** when a
  column is evicted from the trellis (step 3.2.4), trace back from the
  *current* newest column's current-best node through the stored
  backpointers to find what that evicted column's segment *should have
  been* in light of everything seen since. If it disagrees with what was
  already reported for that tick, this is logged/exposed as a correction
  signal (e.g. `Diag.roadMatchCorrectedSegIdx`, `Diag.roadMatchBacktrackDeltaM`)
  but does **not** retroactively move the live EKF/NHC state for a tick
  that has already been consumed — only future ticks benefit, by using the
  corrected segment id as `committed` going forward instead of whatever the
  greedy pass had chosen. This avoids ever "rewriting history" in a system
  where a GPS/EKF fusion decision for tick N has already been consumed by
  tick N+1's predict step.

This two-tier design is the key structural difference from both the current
LiveHMM (no lookback at all) and the offline matcher (full batch retrace,
answer only available after the whole stream is given) — it keeps the
existing per-tick API/latency contract (`update()` returns immediately,
every tick) while adding bounded hindsight that can only improve same-segment
persistence and short-glitch rejection, never block or delay the live loop.

### 3.4 Backpointer memory/array reuse

To avoid per-tick GC churn at 10 Hz: pre-allocate `W` reusable
`{cands, scores, backptr}` slots (circular buffer indices, not
push/shift on a JS array) rather than allocating a fresh object/array every
tick. `candidates()` already allocates a fresh array per tick (existing
behavior, unchanged) — only the *trellis columns* need the reuse to avoid
adding a second allocation source.

## 4. Performance budget (must stay well under 100 ms/tick at 10 Hz)

Target: this entire module's per-tick cost should stay under **2 ms**,
leaving a huge margin under the 100 ms/tick (10 Hz) budget for the rest of
the pipeline (EKF predict/update, ONNX inference, RoadGraph candidate
lookup, NHC, UI).

| Step | Cost per tick | Notes |
|---|---|---|
| `RoadGraph.candidates()` | unchanged (already running today) | grid ring-search, same as now |
| Emission scores, new column | O(K) = 5 evals | unchanged from today |
| Transition scores, new column vs. previous column | O(K x K) = 25 evals | **new** cost vs. today's O(K) |
| Column push/evict (ring buffer) | O(1) amortized | no array shift |
| Lazy traceback on eviction (every tick once window is full) | O(W) = 6 backpointer hops | one hop per evicted column, not a full retrace |
| **Total new work per tick** | ~25 extra log/exp evaluations + 6 pointer hops | microseconds on any phone CPU in scope |

Even a very conservative 1 microsecond per log-probability evaluation
(transcendental `exp`/`pow`-free, pure arithmetic — the existing
`emissionLogProb`/`transitionLogProb` are already this cheap) puts the new
O(K^2) + O(W) work at roughly 30 microseconds/tick, i.e. ~0.03% of the
100 ms/tick budget at 10 Hz. The dominant cost in the live loop remains
EKF/ONNX/RoadGraph work that already runs today, not this change. `W=6-8`
and `K=5` are deliberately small enough that this stays true even on a
low-end Android CPU; if a future revision wants a bigger `W` or `K`, the
O(K^2) and O(W) growth should be re-measured against this budget before
raising either constant.

## 5. What would actually need to change in `frontend/mobile.html`

(Not performed by this task — listed for the future implementation pass.)

1. Replace `LiveHMM`'s single `state` variable with the ring-buffer trellis
   described in 3.1.
2. Replace the single-argmax loop in `update()` (~line 4174-4178) with the
   two-step emission-then-transition-matrix evaluation in 3.2, keeping the
   existing fallback paths (non-finite input, epoch bump, `cands.length <
   MIN_CANDIDATES`) exactly as they are today — those are orthogonal to
   this change and should not be touched.
3. Add the lazy-traceback-on-eviction logic (3.2.4 / 3.3) and the new
   `Diag.roadMatchCorrectedSegIdx` / `Diag.roadMatchBacktrackDeltaM` fields,
   documented the same honest way the rest of `Diag` already is (real value
   or null, never fabricated).
4. Decide, with real device profiling (not just the budget estimate in
   section 4), whether `W` should be 6, 8, or smaller — this document's
   numbers are a budget ceiling, not a measured result.
5. Add regression coverage once `tests/js/` exists (see workstream 4 in the
   accompanying audit) — a bounded-lookback Viterbi is exactly the kind of
   logic that is easy to get subtly wrong (off-by-one on which column is
   "current" vs. "committed", stale backpointers after an eviction) and
   should not go live on `frontend/mobile.html` without tests, which is
   exactly why this task does not implement it there directly.

## 6. Non-goals / explicitly out of scope for this upgrade

- This is **not** a switch to a real road-network graph Viterbi (segments
  with OSM way/node adjacency). `RoadGraph` still has no adjacency data;
  "transition" here still means "did the real movement since last tick
  match moving from candidate A's point to candidate B's point", exactly as
  today — only the *lookback depth* changes, not the underlying graph
  model. A true adjacency-aware graph Viterbi is a separate, much larger
  project (building and maintaining a connected road graph from OSM ways,
  not just flat line segments) and is not implied or required by this
  upgrade.
- This does not change `RoadGraph.candidates()`, `RoadGraph.nearest()`, or
  any of the Overpass/IndexedDB caching logic.
- This does not touch the offline `edge/hmm_map_matcher.py` module, which
  stays exactly as documented in its own header: unrelated, unwired,
  batch-only.
