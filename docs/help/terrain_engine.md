---
title: Terrain Engine
tile_id: terrain_engine
status: complete
---

# Terrain Engine

## Purpose

Browse the built-in golf terrain presets, resize and tilt one, and ask a point
on it what it is: how high, how steep, what surface type, and what friction and
rolling resistance a ball there would meet. It is an inspection tool for the
terrain layer that other simulations consume, not a simulation in itself and
not a terrain editor.

## Inputs

| Control     | Unit | Range accepted   | Default                          |
| ----------- | ---- | ---------------- | -------------------------------- |
| Environment | -    | preset list      | first preset                     |
| Width       | m    | 1 to 1000        | preset width                     |
| Length      | m    | 1 to 2000        | preset length                    |
| Slope       | deg  | -20 to 20        | 0                                |
| Direction   | deg  | 0 to 360         | 0                                |
| Query X     | m    | 0 to preset width  | half the width                 |
| Query Y     | m    | 0 to preset length | half the length                |

Environment presets, with their default extents and the terrain types they
contain (`ENVIRONMENT_PRESETS` in
`src/shared/python/physics/terrain_presets.py`):

| Preset          | Default size    | Terrain types                                 |
| --------------- | --------------- | --------------------------------------------- |
| `putting_green` | 10 m x 15 m     | green, fringe                                 |
| `fairway`       | 50 m x 200 m    | fairway, rough, bunker                        |
| `driving_range` | 80 m x 300 m    | tee, fairway, rough                           |
| `bunker`        | 20 m x 20 m     | bunker, green, fringe                         |
| `rough`         | 30 m x 40 m     | rough, fairway                                |
| `full_hole`     | 60 m x 340 m    | tee, fairway, rough, bunker, fringe, green    |

Selecting a preset resets Width and Length to its defaults and re-centres the
query point. Query controls stay disabled until a terrain has loaded
successfully.

## Outputs

- Summary line: terrain name, extents in m, grid resolution in m, patch count,
  region count.
- Sample table with four fixed sample points (at 25%, 50%, 75% of both extents,
  and one near the far edge): X [m], Y [m], Elevation [m], terrain type.
- Query result for the selected point: terrain type, elevation [m], slope angle
  [deg], friction coefficient [dimensionless], rolling resistance
  [dimensionless].

## Method

`src/tools/terrain_engine/gui.py` calls
`terrain_presets.build_environment_preset(preset, width, length, slope,
direction)`, which dispatches to the named builder for that preset and returns
a `Terrain` (`src/shared/python/physics/terrain.py`). Queries then go straight
to that object: `terrain.elevation.get_elevation(x, y)`,
`terrain.elevation.get_slope_angle(x, y)`, `terrain.get_terrain_type(x, y)`,
and `terrain.get_material(x, y)`.

The tile adds no physics of its own. It is a viewer over the preset builders
and the terrain query API; the elevation grid, patch and region decomposition,
and per-material coefficients are all defined by those modules.

## Limitations

- **The elevation, slope, friction, and rolling-resistance values are model
  parameters, not survey or measurement data.** They are the numbers the preset
  builders assign to a synthetic surface.
- No ball, no roll, no simulation. Nothing moves on this tile; it answers
  point queries only.
- Read-only with respect to terrain content. You can resize a preset and apply
  a single slope and direction, but you cannot edit the surface, place
  features, paint materials, or save a terrain from here.
- No import or export: the engine's grid, `.npy`, GeoTIFF, and
  scattered-point importers are not reachable from this tile.
- Slope is applied as one uniform tilt with one direction. Local undulation is
  whatever the preset builder generates.
- No 3D view. Output is a summary line, a four-row table, and one query string.
- Sample points are fixed and cannot be chosen.
- Elevation queries are resolved at the terrain's grid resolution, so a query
  between grid nodes is interpolated rather than exact.
- Failures - an unknown preset, a bad extent, a query outside the surface - are
  reported in the panel and a dialog rather than raising, so a stale reading can
  remain on screen next to an error message. Check the message before trusting
  the numbers.

## See Also
- [Putting Green](putting_green.md) - the roll simulation that consumes a green surface
- [Golf Environment](golf_environment.md) - the 3D range and hole viewer
- [Simulation Controls](simulation_controls.md)
- [Analysis Tools](analysis_tools.md)
