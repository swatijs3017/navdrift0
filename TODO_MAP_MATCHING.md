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

## What is NOT done yet, so nobody overclaims it

1. **No offline cache.** If the phone loses network exactly at the moment GPS also drops
   (tunnel entrance, common case), a road graph that hasn't been fetched yet won't get one.
   Fetching happens on GPS lock, before blackout, so this only bites if the vehicle enters a
   blackout zone before its first GPS fix ever lands. An IndexedDB cache keyed by rounded
   lat/lon would fix this for repeat visits to the same area, not yet built.
2. **Overpass is a shared public rate-limited endpoint.** Fine for a single-phone demo. Not
   something to rely on for a fleet of vehicles hitting it simultaneously — a production
   version would run its own Overpass mirror or ship pre-extracted regional data.
3. **Not yet validated against a real recorded drive.** The graph fetch, nearest-segment
   projection, and the bearing-aware NHC have been checked for syntax and logic, not against a
   real GPS log through a real GNSS-denied stretch. Do that before presenting a drift number
   that depends on it.
