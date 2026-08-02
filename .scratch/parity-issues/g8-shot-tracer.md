# Shot Tracer / ball-flight visualization missing from web app (API already exists)

## Gap (PyQt6 = model)

Desktop ships Shot Tracer (`src/launchers/_shot_tracer_gui.py` via `src/launchers/shot_tracer.py`): 3D ball-flight trajectories with multi-model comparison (overlay/side-by-side), a unified launch-conditions UI, and a flight-model registry. The backend exposes `POST /ball-flight/simulate` (`src/api/routes/ball_flight.py`), and there's a separate ball-flight GUI tool (`src/tools/ball_flight_gui/`) — but the web app has **no ball-flight page at all**.

## Proposed fix

1. Add a `BallFlight` web route: launch-condition form (speed, launch angle, spin, club — reuse `HelpfulField` with explicit units; the deg-vs-rad/RPM-vs-rad/s convention bugs of #7246 make unit labeling mandatory), flight-model multi-select from a model enumeration endpoint, and a Three.js trajectory overlay (trail rendering already exists in `Scene3D`/`GolferModel` trails — reuse, don't fork).
2. Extend the API if needed: enumerate available flight models (`GET /ball-flight/models`) so desktop registry and web selector share one list; simulate endpoint should accept a list of models and return per-model trajectories for overlay comparison, matching the desktop overlay feature.
3. Side-by-side metrics table (carry, apex, flight time) per model — same numbers the desktop tracer shows.
4. Add the tile to `launcher_manifest.json` with a `web_route` so it appears in both launchers (parity test will enforce).

## Caveat

The 2026-06-11 scientific audit found P0 lift/spin errors in the flight models (#7403–#7405). The web page must consume the corrected models; building the UI is independent, but golden-number tests should reference post-fix expectations.

## Acceptance criteria

- [ ] Web ball-flight page with multi-model overlay and unit-labeled launch conditions
- [ ] Flight-model list shared between desktop registry and web (single enumeration)
- [ ] Same trajectory numbers desktop vs web for identical inputs (golden test)
- [ ] Manifest tile with web_route present
