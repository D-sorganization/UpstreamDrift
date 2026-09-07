---
title: Swing to Flight Pipeline
tile_id: swing_flight_pipeline
status: complete
---

# Swing to Flight Pipeline

## Purpose

Follow one chain end to end: a clubhead arriving at the ball, the impact it
produces, the launch conditions that result, and the flight those launch
conditions fly. This is the tile to use when the question is "what does this
clubhead delivery actually produce", rather than "what does this launch produce"
(see [Ball Flight Simulator](ball_flight_simulator.md) for the latter).

## Inputs

| Control         | Unit | Range accepted | Default |
| --------------- | ---- | -------------- | ------- |
| Clubhead Speed  | m/s  | 20 to 60       | 45.0    |
| Loft Angle      | deg  | 5 to 60        | 10.5    |
| Clubhead Mass   | kg   | 0.100 to 0.400 | 0.200   |

- **Physics Engine Source** selects which provider builds the swing state.
  Entries are real providers, not labels:
  - `manual` - builds the state directly from the three numbers above. Always
    available, and the tile's default.
  - `mujoco` - sources the state from a MuJoCo forward-dynamics swing.
    Available only when the `mujoco` package and the in-repo golf MJCF asset
    are both importable.
  - `drake`, `pinocchio` - listed so the roadmap is visible, but permanently
    disabled with an honest tooltip reason ("engine sourcing not yet
    implemented" or "engine not installed").
  A provider is contractually forbidden from stamping another engine's name on
  its output, so the engine shown in the results is the engine that produced
  them.
- Club presets stamp speed and loft pairs: Driver (50.0 m/s, 10.5 deg),
  7-Iron (35.0 m/s, 34.0 deg), PW (28.0 m/s, 46.0 deg).

## Outputs

- Engine name that produced the swing state.
- Impact results: ball speed [m/s] and ball spin magnitude [rad/s].
- Launch conditions: speed [m/s], launch angle [deg], spin rate [rad/s].
- Flight results: carry [m] with a yards conversion, max height [m], flight
  time [s], landing angle [deg].
- Trajectory point count.
- Two headline values - Carry Distance [m] and Launch Speed [m/s] - rendered as
  provenance-carrying labels. Hovering one shows the formula, the named inputs,
  the engine, the run id, and the UTC timestamp behind that specific number. A
  headline value cannot be rendered without a non-empty engine attribution.
- A 3D flight line shaded by height (apex hot, ground cool), when `pyqtgraph`
  with OpenGL is installed.

## Method

`src/tools/swing_flight_pipeline/gui.py` turns the three controls into a
`SwingStateConfig`, asks the selected provider
(`src/shared/python/physics/swing_state_providers.py`) for a `SwingState`, and
runs `SwingBallFlightPipeline.run`
(`src/shared/python/physics/swing_ball_flight_pipeline.py`). That pipeline
executes five steps: build the pre-impact state, solve the impact, derive
launch conditions, integrate the ball flight, extract metrics.

The impact solve defaults to `ImpactModelType.RIGID_BODY` with default
`ImpactParameters` (COR, friction). Flight integration defaults to a 10 s cap
at a 0.01 s timestep with default `EnvironmentalConditions` and
`BallProperties`. None of those defaults are exposed on this tile.

Impact and flight physics, including COR ranges and the rigid-body,
spring-damper, and finite-time impact model tiers, are documented in
[../physics/BALL_FLIGHT_MODEL_DOCUMENTATION.md](../physics/BALL_FLIGHT_MODEL_DOCUMENTATION.md),
with the module mapping in
[../physics/GOLF_BALL_FLIGHT_IMPACT_SOURCE_MAP.md](../physics/GOLF_BALL_FLIGHT_IMPACT_SOURCE_MAP.md).

## Limitations

- **Carry, height, and flight time are model output, not measurement**, and so
  are the intermediate impact and launch numbers. The provenance tooltips tell
  you which run and engine produced a number; they do not make it measured.
- With `manual` selected - the default - the clubhead arrives with zero angular
  velocity and a fixed orientation. There is no face angle, no club path, no
  attack angle, no impact offset, and therefore no gear effect, no curvature
  from face-to-path, and no off-centre-strike penalty.
- Only one of the four engine sources is implemented beyond `manual`, and
  `mujoco` requires both the package and the in-repo MJCF asset to be present.
  `drake` and `pinocchio` are visible but non-functional by design.
- The impact model, its COR and friction parameters, the environment, and the
  ball properties are all fixed at their defaults with no control on this tile.
  No wind and no altitude.
- No ground interaction: carry only. No bounce, roll, or total distance.
- One shot per run. No dispersion, sweeps, or batches.
- Without `pyqtgraph` and OpenGL the 3D pane is absent; the numeric results
  still compute.

## See Also
- [Ball Flight Simulator](ball_flight_simulator.md) - flight only, with environment controls
- [Shot Tracer](shot_tracer.md) - the flight question across several models
- [Engine Selection Guide](engine_selection.md)
- [Golf Ball Physics: Scientific Basis and Model Documentation](../physics/BALL_FLIGHT_MODEL_DOCUMENTATION.md)
- [Simulation Controls](simulation_controls.md)
