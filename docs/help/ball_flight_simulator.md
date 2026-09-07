---
title: Ball Flight Simulator
tile_id: ball_flight_simulator
status: complete
---

# Ball Flight Simulator

## Purpose

Enter a set of launch conditions and an environment, press one button, and see
where the ball would go under this repository's aerodynamic flight model. The
tile is a single-shot "what does this launch produce" calculator with a 3D
trajectory view and a text summary; it does not model the swing or the strike
that created those launch conditions.

## Inputs

| Control        | Unit  | Range accepted     | Default |
| -------------- | ----- | ------------------ | ------- |
| Ball Speed     | mph   | 50 to 200          | 163     |
| Launch Angle   | deg   | -5 to 45           | 11      |
| Backspin       | rpm   | 0 to 12000         | 2500    |
| Sidespin       | rpm   | -5000 to 5000      | 0       |
| Wind Speed     | mph   | 0 to 50            | 0       |
| Wind Direction | deg   | 0 to 360           | 0       |
| Altitude       | ft    | 0 to 10000         | 0       |

- Sidespin sign convention: positive sidespin curves the ball to the right of
  the target line. Backspin and sidespin are combined into one total spin rate
  [rpm] plus a unit spin-axis vector before the simulator sees them.
- Wind Direction is the direction the wind blows *toward*, measured from
  downrange (+x) toward the left (+y): 0 deg is a pure tailwind, 180 deg a pure
  headwind. Wind is a horizontal vector [m/s]; its z-component is always zero.
- Altitude is converted to metres and used to derive air density [kg/m^3] via
  the ISA atmosphere model.
- Club presets stamp speed/angle/backspin triples only: Driver
  (163 mph, 11 deg, 2500 rpm), 7-Iron (118 mph, 16 deg, 7000 rpm),
  PW (94 mph, 23 deg, 9000 rpm).

## Outputs

- Carry: metres, with a yards conversion shown alongside.
- Max Height: metres, with a feet conversion shown alongside.
- Flight Time: seconds.
- Landing position: X, Y, Z components in metres.
- Trajectory point count (integration samples, not a physical quantity).
- Air density used for the run, echoed in kg/m^3.
- A 3D polyline of the trajectory, when `pyqtgraph` with OpenGL is installed.

## Method

`src/tools/ball_flight_gui/gui.py` builds a `LaunchConditions` and an
`EnvironmentalConditions` (`src/shared/python/physics/ball_launch_conditions.py`)
and hands them to `BallFlightSimulator.simulate_trajectory`
(`src/shared/python/physics/ball_simulator.py`) with `max_time = 10.0 s` and
`dt = 0.01 s`. The simulator integrates the ball state with RK4.

The forces the model includes, as stated in the GUI itself and by the simulator
module, are Reynolds-dependent drag, Magnus lift, gravity, wind, and
altitude-dependent air density. The underlying coefficient formulation and its
literature basis are documented in
[../physics/BALL_FLIGHT_MODEL_DOCUMENTATION.md](../physics/BALL_FLIGHT_MODEL_DOCUMENTATION.md);
the mapping from that literature to specific modules is in
[../physics/GOLF_BALL_FLIGHT_IMPACT_SOURCE_MAP.md](../physics/GOLF_BALL_FLIGHT_IMPACT_SOURCE_MAP.md).

## Limitations

- **The trajectory is model output, not measurement.** Nothing on this tile is
  measured, fitted to a launch monitor, or validated against a specific ball.
  Two flight models in this repository will disagree for the same launch
  conditions; see the Shot Tracer tile if you need that comparison.
- Dimple geometry and seam orientation are **not** modelled. The GUI says so
  explicitly, and the checkboxes that once implied otherwise were removed
  because no backing parameter existed.
- There is no swing, no clubface, and no impact solve. Launch conditions are an
  input, not a result. Use the
  [Swing to Flight Pipeline](swing_flight_pipeline.md) for that chain.
- Wind is uniform, steady, and horizontal. No gusts, shear, or vertical
  component.
- No ground interaction: the run ends at the trajectory's last integrated
  point. There is no bounce, no roll, and therefore no total distance.
- Integration is capped at 10 s of flight; a launch that would fly longer is
  truncated rather than flagged.
- Without `pyqtgraph` and OpenGL the 3D pane is simply absent; the numeric
  results still compute.

## See Also
- [Simulation Controls](simulation_controls.md)
- [Visualization Settings](visualization.md)
- [Shot Tracer](shot_tracer.md) - the same flight question across several models
- [Swing to Flight Pipeline](swing_flight_pipeline.md)
- [Golf Ball Physics: Scientific Basis and Model Documentation](../physics/BALL_FLIGHT_MODEL_DOCUMENTATION.md)
