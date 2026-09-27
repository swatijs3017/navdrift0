# Map-matching against real roads

## Status: implemented for live mode, works anywhere OSM has coverage

`frontend/mobile.html` now fetches a real OpenStreetMap road graph from the Overpass API
(`https://overpass-api.de/api/interpreter`) centered on the phone's actual GPS fix, the moment
it locks (see the `RoadGraph` module, and the call to `RoadGraph.load()` inside `onFix()`).
This is not tied to Bengaluru, or to any of the demo cities in `CITIES` — it queries a ~2.2km
radius around wherever the real fix is, anywhere in the world, and refetches automatically once
the vehicle drifts near the edge of that cached radius (`RoadGraph.needsReload`).

During a GNSS blackout, the predicted position is snapped toward the nearest real road segment
in that graph (`RoadGraph.nearest`), the same way the old code snapped to the fake scripted
route, just against real data. This only runs when `isLive` (real GPS + real IMU) and
`RoadGraph.loaded` are both true. Simulation mode is untouched and still uses the scripted
city-loop HMM, which was never meant to represent real roads and still doesn't — that code path
exists for the on-screen demo cities only.

The debug panel's "Road Graph" row shows the live state: `fetching OSM…`, a segment count once
loaded, or the fetch-failure reason if Overpass could not be reached. If the fetch fails (no
signal at fix time, Overpass rate limit, firewall), map-matching simply stays off. It never
falls back to the fake route and never fabricates road data.

## Fixed since the first version of this file

**The live-mode map-matching and NHC code was dead code.** `step()` returns early the instant
`GPS.active` is true (a real GPS session), before ever reaching the block that called
`RoadGraph.nearest()`. That block only ever ran inside the simulation-mode code path further
down the same function, which a real GPS session can never reach. So a real GNSS blackout was
never actually NHC-constrained and never actually snapped to the OSM road graph, even though
`RoadGraph` itself really was loading real Overpass data and the debug panel's "Road Graph: N
segs" row was telling the truth about that — it just wasn't connected to the actual position
pipeline. Fixed by moving the real logic into the real live blackout branch, gated on
`GPS.active && !GPS.simMode && RoadGraph.loaded`.

**Non-Holonomic Constraint is now tied to the matched road's own bearing.** `RoadGraph.nearest()`
now also returns the matched segment's bearing. The live blackout branch uses that bearing for
the NHC "no sideways slide" constraint when a road match exists, and falls back to the last real
GPS-derived heading (`S.gtH`) only when Overpass hasn't returned a usable segment yet — never a
scripted or fabricated bearing.

## Also fixed since the first version of this file

**Offline cache added.** Every successful Overpass fetch now also gets written to an IndexedDB
store (`RoadGraphCache`, keyed by a coarse ~2km rounded lat/lon tile). If a later fetch fails —
no signal, Overpass down, or the classic case of GPS and network dropping together right at a
tunnel entrance — `RoadGraph.load()` falls back to the nearest cached tile within 1.5x the fetch
radius, if one exists from a previous visit near there. The debug panel's Road Graph row says
which source is actually in use (`live OSM` vs `offline cache`, with the cache's save date), so
this is never silently indistinguishable from a live fetch. A cache miss still just means
map-matching stays off, exactly as a live fetch failure always has — nothing here ever invents a
tile.

## What is NOT done yet, so nobody overclaims it

1. **The cache only helps on a repeat visit to the same area, or later in the same drive after
   the first successful fetch.** It cannot help the very first time a phone is ever used in a
   brand new area with a blackout starting before any fetch has ever completed there — there is
   nothing to have cached yet. That case still has no map-matching, which is correct: there is no
   real road data available to use.
2. **Overpass is a shared public rate-limited endpoint.** Fine for a single-phone demo. Not
   something to rely on for a fleet of vehicles hitting it simultaneously — a production
   version would run its own Overpass mirror or ship pre-extracted regional data.
3. **Not yet validated against a real recorded drive.** The graph fetch, nearest-segment
   projection, and the bearing-aware NHC have been checked for syntax and logic, not against a
   real GPS log through a real GNSS-denied stretch. Do that before presenting a drift number
   that depends on it.
