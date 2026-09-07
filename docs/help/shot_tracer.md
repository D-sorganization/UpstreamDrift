---
title: Shot Tracer
tile_id: shot_tracer
status: complete
---

# Shot Tracer

## Purpose

Run the same launch conditions through several different published ball-flight
models at once and see how far apart they land. This tile exists to make model
disagreement visible: it is a validation and comparison surface first, and a
shot-visualisation utility second. It can also load a saved trajectory record
and plot it beside the computed curves.

## Inputs

| Control        | Unit  | Range accepted | Default |
| -------------- | ----- | -------------- | ------- |
| Ball Speed     | mph   | 50 to 200      | 163     |
| Launch Angle   | deg   | -10 to 45      | 11      |
| Backspin       | rpm   | 0 to 12000     | 2500    |
| Direction      | deg   | -45 to 45      | 0       |
| Spin Axis Tilt | deg   | -45 to 45      | 0       |

- Direction is the horizontal launch azimuth. Spin Axis Tilt is the spin-axis
  roll: 0 deg is pure backspin, plus or minus 45 deg is a fade or draw axis.
- Model checkboxes select which models to run. Every entry in
  `FlightModelType` is offered and all are checked by default. Each checkbox
  tooltip carries the model's own description and citation string.
- Club presets stamp speed/angle/backspin and zero the two direction controls:
  Driver (163 mph, 11 deg, 2500 rpm), 7-Iron (118 mph, 16 deg, 7000 rpm),
  PW (94 mph, 23 deg, 9000 rpm).
- "Import Trajectory Record..." accepts a
  `swing_sim.ball_flight_trajectory/1` JSON file.

## Outputs

A comparison table with one row per selected model:

| Column         | Unit |
| -------------- | ---- |
| Model          | name |
| Carry          | yd   |
| Max Height     | m    |
| Time           | s    |
| Landing        | deg  |

Plus a 3D overlay of every trajectory, one colour per model, with an
independent colour set reserved for imported records so an imported curve can
never be confused with a computed one. Imported records are also listed by
provenance label (`model_family / model_name`) in their own panel, which stays
the source of truth for what was imported even when OpenGL is unavailable.

## Method

`src/launchers/_shot_tracer_gui.py` builds a single `UnifiedLaunchConditions`
from the controls and calls `compare_models`
(`src/shared/python/physics/flight_models.py`), which runs each selected model
over identical launch conditions. Models are resolved through
`FlightModelRegistry`. Two are distinct integrations and the rest share one
constant-coefficient implementation:

| Model             | Approach                                              | Citation string in code       |
| ----------------- | ----------------------------------------------------- | ----------------------------- |
| Waterloo/Penner   | Waterloo quadratic Cd with Penner spin-ratio lift fit | Penner (2003); McPhee et al.  |
| MacDonald-Hanzely | ODE model with exponential spin decay                 | MacDonald & Hanzely (1991)    |
| Nathan            | Constant Cd/Cl with spin decay (Cd 0.22, Cl 0.24)     | Nathan et al. (2018)          |
| Ballantyne        | Constant Cd/Cl, steady spin (Cd 0.20, Cl 0.18)        | Ballantyne et al. (2012)      |
| J. Cole           | Constant Cd/Cl, moderate decay (Cd 0.23, Cl 0.22)     | Cole (2016)                   |
| Rospie DL         | Constant Cd/Cl, driver launch (Cd 0.21, Cl 0.19)      | Rospie & Layton (2014)        |
| Charry L3         | Constant Cd/Cl, higher drag (Cd 0.24, Cl 0.21)        | Charry et al. (2017)          |

Drag and lift coefficients are dimensionless; spin decay rates are in 1/s.
Every `FlightResult` carries the coefficient set the producing model actually
integrated with, so a number can be traced to its coefficients.

The imported-record path and the rule that imported curves never reuse a native
model's colour come from
[ADR-0047](../adr/0047-trajectory-visualization-shared-wire-preserved-viewers.md).
Model background is in
[../physics/BALL_FLIGHT_MODEL_DOCUMENTATION.md](../physics/BALL_FLIGHT_MODEL_DOCUMENTATION.md).

## Limitations

- **Every curve on this plot is model output, not measurement**, including the
  imported records unless their own provenance says otherwise. The tile shows
  that models disagree; it does not adjudicate between them and nothing here
  identifies a "correct" answer.
- Five of the seven models are the same constant-Cd/Cl integration with
  different numbers. Agreement between two of those five is not independent
  corroboration.
- The citation strings are the ones written in the code. They name a source for
  the coefficient set; they are not a claim that this implementation has been
  validated against that paper's data.
- No environment controls at all: no wind, no altitude, no air-density or
  temperature input on this tile. Each model uses its own defaults.
- No swing, no impact solve. Launch conditions are entered by hand.
- No ground interaction: carry only, with no bounce, roll, or total distance.
- A record that fails wire validation is refused with a dialog naming the
  reason and never silently plotted, but the tile does not otherwise check that
  an imported record is comparable with the models beside it.
- Without `pyqtgraph` and OpenGL the 3D pane is replaced by an install hint;
  the comparison table still populates.

## See Also
- [ADR-0047: Trajectory Visualization - Shared Wire, Preserved Viewers](../adr/0047-trajectory-visualization-shared-wire-preserved-viewers.md)
- [Ball Flight Simulator](ball_flight_simulator.md) - one model, with environment controls
- [Golf Ball Physics: Scientific Basis and Model Documentation](../physics/BALL_FLIGHT_MODEL_DOCUMENTATION.md)
- [Visualization Settings](visualization.md)
- [Simulation Controls](simulation_controls.md)
