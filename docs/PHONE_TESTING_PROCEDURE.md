# NAVDRIFT-0 — Physical Phone Testing Procedure

This procedure is for the one thing that cannot be verified in a development
environment: real sensor hardware, real GNSS, and a real moving vehicle.
Everything listed here as "verified in software" has already been checked by
automated tests and browser-level simulation (see test files under `tests/`
and `tests/js/`) — this document only covers what still needs a human with a
phone.

**Safety first.** Never let this testing interfere with driving. Use a
passenger seat, a stationary vehicle, or a slow, controlled, legal route with
someone else driving. Never test inside a tunnel unsafely, never contact
emergency services as part of a test, and never prioritize completing a test
step over safe driving.

## Before you start

- Use a phone with a modern browser (Safari on iOS 13+, Chrome on Android).
- Open the production URL: https://navdrift0.pages.dev/ — it should
  automatically route you to the mobile navigation interface (not the desktop
  Mission Control dashboard). If it doesn't, that's a routing bug worth
  reporting immediately, before going further.
- If your platform supports installing it as a PWA (Android Chrome's "Add to
  Home Screen", iOS Safari's "Add to Home Screen"), install it and open it
  from the home screen icon rather than the browser, since this is closer to
  how it would actually be used.

## 1. Opening and installing the PWA

- Confirm the page loads and shows the mobile navigation UI (map, position
  card, bottom nav bar), not an error page or the desktop dashboard.
- Try installing it (see above). Confirm it reopens correctly from the home
  screen icon afterward.

## 2. Granting sensor and location permissions

- Tap whatever "Enable sensors" / calibration-start control is shown (look
  for a button near the top or in the expandable technical panel).
- **On iOS specifically**: this tap should trigger the native "Motion &
  Orientation Access" permission prompt. If it does not appear at all, or
  appears only after a delay/second tap, that's a real bug — the code is
  supposed to request this synchronously inside the button's own click
  handler (iOS requires this).
- Grant motion/orientation access, then grant location access when prompted.
- If you deny either permission, confirm the app shows a clear message (not
  a silent failure or a frozen UI) explaining what was denied and that
  functionality will be degraded.

## 3. Verifying real accelerometer and gyroscope samples

- Open the technical/diagnostics panel (look for an expand control near the
  bottom nav card).
- Confirm you see live-updating accelerometer (Ax/Ay/Az) and gyroscope
  (Gx/Gy/Gz) numbers that visibly change as you move or tilt the phone.
- Confirm a measured sample-rate readout (labeled Hz) is present and
  nonzero. This is a real counter of arriving events, not a fixed label — if
  it reads 0 or stays frozen while you're moving the phone, sensors are not
  actually arriving and this needs investigation before any other step is
  meaningful.
- If the readout is present but shows "NO DATA" or similar per-sensor status
  text, that reflects a real staleness watchdog (no data for >3s), which is
  correct behavior if sensors genuinely aren't arriving — not itself a bug,
  but worth noting which sensor and under what circumstance.

## 4. Completing vehicle alignment and calibration

- Mount the phone in a vehicle (dashboard mount, cup holder, wherever you'd
  realistically use it) and start driving normally (or have a passenger hold
  it steady while someone else drives) for about 5 seconds of real forward
  motion.
- Confirm the app detects alignment completion (look for a calibration
  status indicator changing from "not calibrated"/"calibrating" to a locked
  state).
- **New in this pass**: if you then deliberately pick up the phone and
  reorient it significantly (e.g. rotate it 90°+ from its mounted position)
  while otherwise keeping the vehicle still or slow, a banner should appear
  after a few seconds prompting you to recalibrate. This is a new heuristic
  (25° drift over 4+ seconds of low motion) — it has not been tuned against
  a real holder-shift event, so note whether it triggers too eagerly, too
  late, or not at all, and roughly how much reorientation it took.

## 5. Confirming GNSS-aided navigation

- With a clear view of the sky, confirm the position card shows a real,
  updating latitude/longitude, a NavIC/GPS lock status, and that the map's
  vehicle marker moves consistently with your actual movement.
- Confirm speed and heading readouts update plausibly as you move.

## 6. Testing a controlled GNSS-denied scenario

**Only do this somewhere safe** — a stationary test, a passenger-seat test,
or (if your phone allows it) toggling Airplane Mode / disabling location
permission briefly while stationary or at very low, safe speed, with someone
else driving if the vehicle is moving at all. Do not use a real tunnel or any
maneuver that requires taking attention off the road.

- Trigger a simulated loss of GNSS (there is a GNSS-toggle control in the
  desktop dashboard for simulation purposes; on the real phone, the most
  realistic safe test is turning off location permission/airplane mode
  briefly, or simply driving into a location with naturally poor GPS like
  under a parking structure, if that is itself safe to do).
- Confirm the status chip changes to a dead-reckoning / blackout state
  (should be visually distinct from GNSS-locked, e.g. a different color or
  explicit "DEAD RECKONING" label).

## 7. Observing continuous dead-reckoning position updates

- While GNSS is unavailable, confirm the vehicle marker continues to move on
  the map (does not freeze) as long as the phone continues to receive real
  motion data. It should move based on your actual motion, not drift off in
  an obviously wrong direction or jump erratically.
- If you are genuinely stationary during this test, the marker should
  correctly stay essentially still (this is also correct behavior — dead
  reckoning from zero real motion should not fabricate movement).

## 8. Restoring GNSS and observing reacquisition

- Restore GNSS (turn location back on / exit airplane mode / return to open
  sky).
- Confirm the marker moves smoothly back toward the real GPS-indicated
  position rather than teleporting instantly, and that the status chip
  returns to a GNSS-locked state.
- If the new diagnostics panel additions from this pass are present, check
  for a "last blackout duration" and "reacquisition correction" readout and
  note the values — these are useful to report back even if you don't know
  what "correct" should look like, since they've never been exercised with
  real hardware.

## 9. Checking sensor health, sample rate, and navigation frequency

- In the diagnostics panel, note the measured sensor Hz (accelerometer/
  gyroscope) and the measured navigation update rate (labeled against a
  10Hz target). Real phone `devicemotion` rates commonly run lower than
  desktop-class sensors (often 30-60Hz) — this is expected and already
  accounted for in several of this pass's fixes (e.g. the map-matching and
  DriftFormer resampling logic), but worth recording the actual number you
  see.

## 10. Capturing diagnostics and logs if something fails

- If anything behaves unexpectedly, check the browser's developer console
  (Safari: Settings > Advanced > Web Inspector, then connect to a Mac; or
  use remote debugging from a desktop Chrome for Android) for JavaScript
  errors, and note the exact state shown in the diagnostics panel at the
  time (sensor Hz, nav Hz, GNSS status, calibration status).
- There is an "Export IMU Log" or CSV-export style control in the technical
  panel — use it if something looks wrong, since the raw log is far more
  useful for debugging than a description after the fact.
- Report back: what you did, what you expected, what actually happened, and
  the diagnostics panel's values at the time.

## What has already been verified without a phone (do not re-test from scratch)

- All EKF/dead-reckoning math, map-matching scoring, and the test suite
  logic itself (108 automated Python tests, 4+ automated JS tests covering
  the real `ekfStep`/`LiveHMM` code paths via a headless browser).
- iOS permission-request code is gesture-gated correctly per the spec (the
  code path is correct; only the actual native prompt appearing is
  unverified).
- Gravity compensation is applied exactly once, not twice.
- DriftFormer's AI correction is currently and deliberately disabled (shown
  honestly in the UI) because its required normalization statistics are not
  recoverable in this environment — expect to see dead reckoning driven by
  kinematic integration + EKF, not a neural correction, and that is the
  correct, intended current state, not a bug.
- The speed-estimation LSTM is validated but not integrated (same reason
  class as DriftFormer — a sample-rate mismatch that would need fixing
  first) — expect no speed-LSTM contribution to navigation currently.

## What specifically needs your real-device judgment

- Whether real accelerometer/gyroscope/GNSS data actually arrives at all on
  your specific phone/browser, and at what rate.
- Whether the iOS permission prompt itself behaves correctly end-to-end.
- Whether the new recalibration-drift banner's threshold feels right in
  practice.
- Whether the map-matching behavior (staying on the correct road, not
  snapping to a parallel road) holds up against real OSM data for your
  actual test route, not just the synthetic test cases already verified.
- Overall whether the dead-reckoning position estimate "feels" usably close
  to correct for your test route and speed — the per-sequence IO-VNBD
  evaluation found the classical baseline exceeds the SIH 10%-drift target
  on some trajectory types and stays within it on others, so expect
  variable real-world accuracy depending on your route and speed profile.
