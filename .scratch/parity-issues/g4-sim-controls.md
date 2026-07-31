# Wire dead web SimulationControls affordances to their existing backend endpoints (camera, recording, export)

## Gap (PyQt6 = model)

`ui/src/components/simulation/SimulationControls.tsx` renders camera presets (side/front/top/follow_ball/follow_club), a recording toggle, and a trajectory-export button — but the callbacks are optional and **unwired**: clicking them does nothing. Meanwhile the backend endpoints already exist (`POST /simulation/camera`, `POST /simulation/recording` in `src/api/routes/physics.py`) and the PyQt6 viewers have a full camera-preset + recorder system. This is the worst kind of parity gap: visible-but-fake controls.

## Proposed fix

1. Camera presets: `Simulation.tsx` wires preset selection to `POST /simulation/camera` AND to the local Three.js camera in `Scene3D.tsx` (follow modes already partially exist there) so the WebGL view honors the same presets the desktop viewer has. Preset list should come from the API/capabilities, not be hardcoded.
2. Recording toggle: wire to `POST /simulation/recording` start/stop; surface recording state (armed/recording/saved id) via the simulation store; hand off the saved recording id to the recordings panel (export-parity issue in this epic).
3. Trajectory export: until the full recordings API lands, minimally export the in-browser frame buffer the LivePlot already accumulates as CSV/JSON download; switch to the server-side export once available.
4. If any control cannot be made functional for an engine, hide it via capability data instead of rendering a dead button (engine capability profiles exist at `/launcher/engines/{id}/capabilities`).

## Acceptance criteria

- [ ] Every control rendered in SimulationControls does something real or is hidden
- [ ] Camera presets affect both server-side state and the web 3D view
- [ ] Recording start/stop round-trips and produces a retrievable recording id
- [ ] Component tests cover the wired paths (no optional-callback no-ops left)

## References

- `src/api/routes/physics.py` (camera/recording/speed endpoints — already implemented)
- Related: #7424 (parameter reset), #7425 (speed-slider call-per-tick) from the UI/UX epic — coordinate, don't duplicate
