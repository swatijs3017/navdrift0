# Real map-matching — not implemented yet

`frontend/mobile.html`'s HMM map-matching block (search `HMM map-matching`) only runs in
simulation mode (`GPS.simMode`). It snaps the predicted position to `S.route`, which is a
hand-authored waypoint loop for a demo city (`CITIES[key].wp`), not a real road network. It is
gated off in live mode (`GPS.active && !GPS.simMode`) on purpose: running it against a live GPS
fix would silently pull your real predicted position toward a fake, unrelated route.

To make this real, in order of effort:

1. **Bundled regional extract (fastest to demo, no network dependency at judging time).**
   Pre-download an OSM extract (e.g. via `osmium`/`geofabrik`) for the specific
   city/campus/route you'll actually drive during evaluation. Build a KD-tree-indexed edge
   list from the `highway=*` ways (this matches the "KD-tree indexed GeoJSON road graph"
   description already in `navdrift_outputs/HANDOFF.md`, which was written but never wired to
   live GPS). Ship the extract as a static JSON/GeoJSON asset alongside `mobile.html`.

2. **Live Overpass API query (works anywhere, needs network).** On GPS lock, query
   `overpass-api.de` for `highway=*` ways within ~2km of the current fix, build the same
   KD-tree graph client-side. Cache it so blackout doesn't depend on a live connection at the
   exact moment GPS drops.

3. **Wire the existing Viterbi/HMM decode against real edges instead of `S.route`.** The
   emission (Gaussian, sigma≈18m) and transition (exponential, lambda≈4) model already in the
   code is reusable — only the candidate set needs to change from route waypoints to
   nearest-road-segment projections, plus a proper Non-Holonomic Constraint against the
   matched edge's bearing (the current NHC block constrains against `S.gtH`, the simulated
   ground-truth heading, which also needs to become the real IMU/GPS-derived heading in live
   mode — check that this is still correct once map-matching is real).

4. **Validate against a real recorded drive**, not the IO-VNBD offline set, before claiming
   this satisfies the PS's map-matching requirement.

None of this is implemented. Do not present the current HMM code as satisfying the PS's
map-matching requirement until one of the above is done and tested against a real drive.
