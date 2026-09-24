---
title: Putting Green
tile_id: putting_green
status: complete
---

# Putting Green

## Purpose

Roll a putt across a sloped green and see whether it drops. You set the stroke
speed, your aim, how far away the cup is, and how fast and how tilted the green
is; the tile runs the putting-green roll engine and shows the ball track, the
break, and whether the putt was holed. It is a green-authoring and roll-reading
surface, not a putter-fitting or impact-analysis tool.

## Inputs

| Control       | Unit  | Range accepted | Default |
| ------------- | ----- | -------------- | ------- |
| Putter Speed  | m/s   | 0.5 to 8.0     | 2.5     |
| Aim Angle     | deg   | -45 to 45      | 0       |
| Cup Distance  | ft    | 1 to 30        | 10      |
| Stimpmeter    | ft    | 6.0 to 14.0    | 10.0    |
| Cross Slope   | deg   | 0 to 5         | 1.0     |

- Every control is range-checked before the engine runs; an out-of-range or
  non-finite value raises rather than being silently clamped
  (`_validate` in `src/tools/putting_green_gui/_scene_builder.py`).
- Presets stamp speed and distance pairs only: Short (1.5 m/s, 5 ft),
  Medium (2.5 m/s, 15 ft), Long (4.0 m/s, 30 ft).
- Not exposed on this tile, but fixed in the scene builder: integrator `rk4`,
  timestep 0.002 s, terrain grid resolution 48. Cup distance is converted to
  metres; the green is sized around the putt with fixed margins.
- The Stimpmeter reading is the only turf control on the tile. Grass type,
  height of cut, grain direction, and condition exist in
  `TurfProperties` but are left at their defaults here.

## Outputs

- Result: holed, or missed by a lateral distance reported in cm.
- Total roll: metres, with a feet conversion alongside.
- Roll time: seconds.
- Peak break: cm, the largest perpendicular deviation of the path from the
  straight ball-to-cup line.
- Launch speed: m/s, taken from the first integrated velocity.
- **Roll model name**, printed verbatim in the metrics panel. Read this before
  you compare any number above with anything.
- A contour-shaded green mesh, an animated ball, a flagstick, the aim line, and
  a ball track coloured by roll mode: amber while skidding, green in pure roll,
  grey once stopped.

## Method

The GUI is a thin renderer. All domain logic is in
`src/tools/putting_green_gui/_scene_builder.py`, which builds a `GreenSurface`
with a uniform cross-slope heightmap, applies `TurfProperties(stimp_rating=...)`,
and runs `PuttingGreenSimulator.simulate_putt`
(`src/engines/physics_engines/putting_green/python/simulator.py`) with RK4 at a
0.002 s timestep. Roll dynamics come from
`src/engines/physics_engines/putting_green/python/ball_roll_physics.py`, which
resolves the slide-to-roll transition, applies the turf friction law, and
reports a `RollMode` per sample.

**This repository preserves two putting roll models, and this tile uses exactly
one of them.** Per
[ADR-0045](../adr/0045-putting-integration-one-experience-two-preserved-stacks.md):

- `ud-legacy-roll/1` (`UD_LEGACY_ROLL_MODEL`) - the agronomic law,
  mu approximately 0.196/stimp, scaled by height-of-cut, condition, and grain
  factors. This is the model `ball_roll_physics.py` implements, and therefore
  the model behind this tile. Hole capture is a radius test plus a 1.5 m/s
  lip-out heuristic.
- `usga-stimp-roll/1` (`USGA_STIMP_ROLL_MODEL`) - the preserved counterpart,
  mu approximately 0.559/stimp, derived from USGA stimpmeter geometry at a
  1.83 m/s release speed, with Holmes/Penner speed-dependent hole capture.
  Inside this repository that law lives in
  `src/shared/python/putting_dynamics` and is reached by the `/simulate-3d`
  route, not by this tile.

The two laws share the `1/stimp` form but assume different stimpmeter release
speeds, which pins the roll-out ratio between them at a constant of
approximately 2.854. The divergence is physics, not a bug: each model is
internally consistent. Because of that fixed factor, every result document the
engine emits carries a `roll_model` field, and readers refuse a payload that
does not name its model.

Turf and stimpmeter references, with provenance classes, are collected in
[../physics/PUTTING_KINEMATICS_KINETICS_REVIEW.md](../physics/PUTTING_KINEMATICS_KINETICS_REVIEW.md).

## Limitations

- **The roll is model output, not measurement.** No number on this tile is a
  measured putt.
- **Never compare a result from this tile with a result from the other roll
  model without both model names attached.** The two preserved models differ by
  a roll-out ratio of approximately 2.854; an unlabelled comparison is
  meaningless, and ADR-0045 makes naming the model mandatory rather than
  optional.
- There is no stroke or impact solve. Putter speed is the launch input; face
  angle, path, attack angle, impact offset, and putter MOI are not modelled
  here. Those belong to the analytics stack described in ADR-0045.
- No dispersion, no Monte Carlo, no repeated-putt statistics: one putt per run.
- The green geometry on this tile is a **uniform cross-slope only**. The
  engine's topography presets and its grid, `.npy`, GeoTIFF, and
  scattered-point importers are not reachable from these five controls.
- Grain direction, grass type, height of cut, and green condition are left at
  `TurfProperties` defaults and are not adjustable here, even though they scale
  the friction law.
- No wind, and no practice mode, though the engine supports both.
- Without `pyqtgraph` and OpenGL the 3D scene is absent; the metrics panel
  still computes.

## See Also
- [ADR-0045: Putting Integration - One Experience, Two Preserved Physics Stacks](../adr/0045-putting-integration-one-experience-two-preserved-stacks.md)
- [Putting Kinematics and Kinetics - Public-Data Review](../physics/PUTTING_KINEMATICS_KINETICS_REVIEW.md)
- [Terrain Engine](terrain_engine.md) - surface presets and material queries
- [Simulation Controls](simulation_controls.md)
- [Visualization Settings](visualization.md)
