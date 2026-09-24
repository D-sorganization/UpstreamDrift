---
title: Simulator
tile_id: golf_simulation_suite
status: stub
---

# Simulator

## Purpose

The registry presents this tile as a combined golf simulator covering both ball
flight and putting. In the code as it stands it is a placeholder: neither of
the two entry points that can back this tile runs a simulation. Treat it as
unimplemented and use the dedicated tiles instead -
[Ball Flight Simulator](ball_flight_simulator.md) for flight and
[Putting Green](putting_green.md) for putting.

## Inputs

None. There are no parameter controls of any kind on either code path - no
speed, angle, spin, distance, or surface input, and therefore no physical
quantities to document.

The only interactive elements, and only on the `__main__.py` path, are two
buttons:

| Control                | Effect                                             |
| ---------------------- | -------------------------------------------------- |
| "Simulate Ball Flight" | Draws a fixed three-point polyline and a sphere    |
| "Putting Green Mode"   | Draws a fixed plane, a cylinder, and a sphere      |

## Outputs

No computed outputs. No distances, speeds, times, or angles are produced or
reported.

What the `__main__.py` path draws is hardcoded geometry in metres:

- "Simulate Ball Flight": a `pv.Sphere` of radius 0.02 m at the origin and a
  polyline through the literal points (0, 0, 0), (50, 0, 20), (100, 0, 0).
- "Putting Green Mode": a 10 m x 10 m plane, a cylinder of radius 0.05 m and
  height 0.1 m at (3, 3, -0.05) standing in for the hole, and a 0.02 m sphere
  at (-3, -3, 0.02) standing in for the ball.

The `gui.py` path renders a single label reading "Golf Simulation Suite (GUI
placeholder)" and nothing else.

## Method

There is no method to document, because no physics is executed.

`src/tools/golf_simulation_suite/__main__.py` does import and instantiate
`EnhancedBallFlightSimulator`
(`src/shared/python/physics/ball_enhanced_simulator.py`) as `self.ball_sim`,
but **never calls it**. Both button handlers add fixed PyVista meshes to the
plotter and return. A terrain-engine import is present only as a commented-out
line. `src/tools/golf_simulation_suite/gui.py` builds a `QMainWindow` with one
`QLabel` and no logic at all.

## Limitations

- **Nothing this tile displays is a simulation, a model output, or a
  measurement.** The three-point "trajectory" is a literal list of coordinates
  typed into the source. It is not drag-free ballistics, not an analytic
  solution, and not the output of any flight model in this repository. No
  quantity may be read off it for any purpose.
- No inputs, no outputs, no physics, no persistence, no export.
- The registry capabilities `full_simulation`, `parameter_sweep`, and
  `batch_runs` are not implemented anywhere on either code path. There is no
  sweep and no batch facility.
- The registry description, "Complete golf simulation with ball flight physics
  and putting green", does not describe the current code.
- PyVista and `pyvistaqt` are hard imports on the `__main__.py` path, so that
  path fails to import outright if they are absent rather than degrading.

## Unclear

Which of two divergent code paths a user actually sees from this tile could not
be determined from the code:

- `src/config/launcher_manifest.json` (and `src/config/models.yaml`) give the
  tile's `path` as `src/tools/golf_simulation_suite/__main__.py`, whose
  `get_dockable_ui` returns the PyVista window with the two hardcoded-geometry
  buttons.
- `src/tools/golf_simulation_suite/_embed_adapter.py`, which implements the
  `EmbeddableTool` protocol the launcher resolves tiles through (ADR-0013),
  declares `EmbedCapabilities.NONE` and builds its widget from
  `src/tools/golf_simulation_suite/gui.py` - the bare placeholder label.

The two produce visibly different windows. Files inspected:
`src/tools/golf_simulation_suite/__main__.py`,
`src/tools/golf_simulation_suite/gui.py`,
`src/tools/golf_simulation_suite/_embed_adapter.py`,
`src/config/launcher_manifest.json`.

## See Also
- [Ball Flight Simulator](ball_flight_simulator.md) - a working flight simulation
- [Putting Green](putting_green.md) - a working putting simulation
- [ADR-0013: Launcher Composability](../adr/0013-launcher-composability.md)
- [Simulation Controls](simulation_controls.md)
