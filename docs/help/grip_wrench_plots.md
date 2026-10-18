---
title: Grip Wrench Plots
tile_id: grip_wrench_plots
status: active
---

# Grip Wrench Plots

## Purpose

Show how the hands load the club: the force of each hand at its grip point, the net force at the grip midpoint and the equivalent couple about the midpoint, over time. The same plots are available in the desktop dock, in the web Analysis Tools page and the matched-swing results view, and as 3D glyphs in every viewer and exported video.

## Inputs

| Input              | Units         | Description                                                                                                                  |
| ------------------ | ------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Grip wrench series | N, N\*m       | Per-sample hand wrenches exerted by the hand on the club, world axes, from the engine's constraint multiplier or `efc_force` |
| Couple frame       | world or club | Frame of the equivalent couple panel; the club frame needs the club orientation                                              |
| Impact time        | s             | Optional event marker                                                                                                        |

## Outputs

| Output                              | Units | Description                                                                                                           |
| ----------------------------------- | ----- | --------------------------------------------------------------------------------------------------------------------- |
| Hand force magnitude                | N     | Left and right hand force on the club                                                                                 |
| Net force at midpoint               | N     | Sum of both hand forces, drawn at the grip midpoint                                                                   |
| Equivalent couple                   | N\*m  | Moment of the hand forces about the midpoint plus both free torques                                                   |
| Contact force moment vs free torque | N\*m  | The two parts of the couple                                                                                           |
| Split method                        | label | How the left/right split was obtained: `constraint_multiplier`, `efc_force`, `allocation`, `bushing` or `unavailable` |

Glyph colours: net `#56B4E9`; the left hand is lighter and the right hand darker. The couple is a curved arrow about the midpoint whose axis is the couple direction; each hand's free torque is a separate arc at its grip point. Use the `grip_per_hand`, `grip_net`, `grip_couple` and `grip_mof` toggles to show or hide each class.

## Null-Space Exploration Tabs

The window is tabbed. The grip wrench view above is the first tab. The Invariants, Envelopes, Hand Split and Diagnostics tabs show the closed-loop null-space analysis (NSX, #12575). They draw only the NSX plot model: every value, band, hatched or shaded segment, withheld quantity and reason is the display decision recorded in that model, and a quantity the model does not mark shown is listed with its reason and never plotted. Each tab carries an evidence badge (action, label, model and validation badges, hypothesis and conditioning chips). A representative allocation is drawn only as a dashed overlay on its band.

Pick a registry preset, a registered trajectory source and the frame indices, then press Run. The analysis runs on a worker thread with a progress bar; Cancel stops it at the next stage boundary. Nothing here is a swing measurement: the bands are static feasible-set envelopes under the declared hypotheses, and the representative is not an estimate of the true hand loads.

## Method

`biomechanics.grip_wrench.analyze_grip` reduces the two hand wrenches to the midpoint with the shared wrench-transport helper. `force_overlay.grip_frame` turns the analysis into an overlay frame, `biomechanics.grip_plot_model` into plot series. The API route is `GET /api/analysis/grip-wrench?run_id=...`. The video export draws the same glyphs and offers a `hands_closeup` camera that follows the grip midpoint.

## Limitations

The left/right split of two rigid welds is set by the solver, not by physics; read it together with the split method. A quantity that cannot be computed (OpenSim, a run with no grip data, an allocation that yields only the net wrench, an engine that supplies no free torque) is shown as "unavailable" with its reason, never as zero. Wrenches are software outputs, not measured hand forces.

## See Also

- [Impact Parameters](impact_parameters.md)
- [Analysis Tools](analysis_tools.md)
