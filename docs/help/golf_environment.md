---
title: Golf Environment
tile_id: golf_environment
status: complete
---

# Golf Environment

## Purpose

Put a ball flight in context. This tile draws a simple 3D scene - a driving
range with yardage lines, or a schematic golf hole with tee, fairway, green,
and pin - and overlays trajectory polylines on it. It is a viewer: it renders
geometry and paths handed to it, and computes no physics of its own.

## Inputs

Interactive controls on the standalone window:

| Control     | Values                                          |
| ----------- | ----------------------------------------------- |
| Environment | Driving Range, Par 3 (150y), Par 4 (400y)       |

Programmatic inputs, for callers embedding the renderer:

- `DrivingRange`: name, width [m] (default 100.0), length [m] (default 350.0),
  and a list of distance markers [yd] (defaults 50, 100, 150, 200, 250, 300).
- `CourseHole`: name, par, yardage [yd], tee position [m] as an (x, y, z)
  triple, pin position [m] as an (x, y, z) triple, fairway width [m]
  (default 40.0), green radius [m] (default 15.0).
- `add_trajectory(points, color)`: an (N, 3) array of positions [m], with an
  optional flat RGBA colour. Omit the colour and the line is shaded by height.
- `clear_trajectories()` removes every overlaid path.

Scene coordinates are metres. The two hole presets convert their yardages to
metres for pin placement (0.9144 m per yard), and the driving range does the
same for its marker lines.

## Outputs

- A 3D scene rendered with `pyqtgraph.opengl`: a coloured ground mesh, and
  either white yardage marker lines across the range or a tee box, fairway,
  circular green, and vertical pin line for a hole.
- Trajectory polylines, shaded by height when no explicit colour is given
  (apex hot, ground cool, via the shared `golf_viz` palette).

There are no numeric outputs. Nothing is measured or reported; the tile
produces a picture.

## Method

`src/tools/golf_environment/gui.py` builds meshes and line items directly with
`pyqtgraph.opengl`, using the shared geometry builders in
`src/shared/python/golf_viz` (`rect_vertices`, `circle_fan_vertices`,
`speed_colors`). Environments are the two dataclasses above; switching the
combo box replaces the environment and clears any overlaid trajectories.

There is no physics module behind this tile. Trajectories come from whatever
caller supplies them.

## Limitations

- **The standalone window shows a hardcoded demonstration curve, not a
  simulation.** On construction it adds one analytic parabola
  (`x = 50t`, `y = 0`, `z = 25t - 0.5 * 9.80665 * t^2`, truncated at ground
  level) purely so the view is not empty. It is a drag-free, spin-free,
  wind-free ballistic arc and it is not the output of any flight model in this
  repository. Do not read distance, height, or flight time off it.
- Nothing on this tile is measurement, and nothing on it is a validated
  trajectory either.
- The geometry is schematic, not a course model: the hole is a rectangular
  fairway plus a flat circular green, and boundaries are simplified. Terrain
  elevation, contour, and materials are absent - the ground is a flat plane.
  For real surface geometry and material queries use the
  [Terrain Engine](terrain_engine.md).
- The three combo entries are the only environments reachable from the UI.
  Custom `DrivingRange` and `CourseHole` definitions require code.
- No text labels in the 3D scene: yardage markers are unlabelled lines, because
  the renderer does not draw 3D text.
- `CourseHole.par` and `yardage` are carried but not used for rendering; pin
  placement comes from `pin_position`.
- No camera presets, no export, and no measurement tools.
- `pyqtgraph` is required. Without it the tile shows a single line of text
  saying so and renders nothing.

## See Also
- [Terrain Engine](terrain_engine.md) - real surface geometry and material queries
- [Ball Flight Simulator](ball_flight_simulator.md) - produces trajectories worth overlaying
- [Shot Tracer](shot_tracer.md)
- [Visualization Settings](visualization.md)
