---
title: Ground Reaction Plots
tile_id: ground_reaction_plots
status: active
---

# Ground Reaction Plots

## Purpose

Show how the ground loads the body during a swing: the force under each foot and in total, how the vertical load shifts between the feet, the centre-of-pressure path, the free moment and the moment of the ground reaction about the whole-body centre of mass, over time. The same six panels are available in the desktop dock and in the web Analysis Tools page.

## Inputs

| Input                  | Units      | Description                                                                                              |
| ---------------------- | ---------- | -------------------------------------------------------------------------------------------------------- |
| Ground-reaction series | N, m, N\*m | Per-sample breakdown of the foot contacts (`GroundReactionSeries`, GCV-1), world axes, force on the body |
| Body weight            | N          | Optional; adds the force traces in body weights (from `body_weight_n` or `body_mass_kg` in the run data) |
| Events                 | s          | Optional address, top, impact and finish markers                                                         |

## Outputs

| Output                         | Units    | Description                                                                 |
| ------------------------------ | -------- | --------------------------------------------------------------------------- |
| Vertical Ground Reaction Force | BW or N  | Vertical force per foot and net; body weights when the body weight is known |
| Net Force Components           | N        | Net force x, y, z and magnitude                                             |
| Vertical Load Share            | fraction | Each foot's share of the summed vertical force                              |
| Centre of Pressure Path        | m        | Top view of the centre of pressure per foot and net                         |
| Free Moment About the Vertical | N\*m     | Free moment at the centre of pressure, per foot and net                     |
| Net Moment About CoM           | N\*m     | Moment of the net ground reaction about the centre of mass, x, y, z         |

## Method

`biomechanics.ground_reaction` (GCV-1) reduces each foot's contact forces to a force, centre of pressure, free moment and moment about the centre of mass. `biomechanics.ground_reaction_plot_model` turns the series into plot traces shared by every surface; `plotting.renderers.ground_reaction` draws the desktop and report sheet. The API route is `GET /api/analysis/ground-reaction?run_id=...`; the desktop dock opens a saved response JSON.

## Limitations

The centre of pressure and the load share are unavailable while the feet together carry less than 10 N. A run with no ground contact, or an engine that cannot report it (Simscape until GCV-3), is shown as "unavailable" with its reason, never as zero. Forces are software outputs of the contact model, not force-plate measurements.

## See Also

- [Grip Wrench Plots](grip_wrench_plots.md)
- [Analysis Tools](analysis_tools.md)
