---
title: Gait Model
tile_id: biomech_gait
status: stub
---

# Gait Model

## Purpose

Gait Model is an exercise preset, not a separate application. It is a virtual
tile: the registry gives it the pseudo-path
`virtual/biomech_exercise/biomech_gait` and the field `exercise: gait`, and
`VIRTUAL_PREFIXES` in `src/shared/python/config/tile_target_resolution.py` maps
the `virtual/biomech_exercise/` namespace onto the backing script
`src/launchers/exercise_dashboard.py`. Opening the tile therefore opens the
[Exercise Dashboard](biomech_exercise.md) with the gait exercise selected.

The registry describes it as a "Standardized human gait model across engines"
and gives it the capabilities `biomechanics`, `exercise_preset` and `gait`.

## Inputs

| Input | How it is supplied | Unit or values |
| --- | --- | --- |
| Exercise name | fixed to `gait` by the tile's `exercise` field | identifier string |
| Engine | `BIOMECH_ENGINE`, or the dashboard's toolbar combo box | `MuJoCo_Models`, `Drake_Models`, `Pinocchio_Models`, `JaxSim_Models`, `OpenSim_Models` |
| Gait model definition | the selected engine's sibling model checkout, under `exercises/gait` | engine-native model format |

`BiomechExerciseHandler` in `src/launchers/launcher_model_handlers.py` reads
`model.exercise` (defaulting to `gait`) and launches the dashboard script.
`discover_exercise("gait")` in `src/shared/python/biomech/exercise_registry.py`
decides which engines are offered by testing for an `exercises/gait` directory
under each registered model source.

## Outputs

| Output | Description |
| --- | --- |
| Window title | `Biomechanics Exercise: Gait` |
| Embedded dashboard | the selected engine's dashboard, constructed with `exercise_filter="gait"` |

Everything numeric that a gait analysis would produce is computed by the
embedded engine dashboard and by the gait model in the sibling checkout, not by
any module in this repository. See [Unclear](#unclear).

## Method

The tile carries no code of its own. Launching it is exactly the Exercise
Dashboard launch path with `exercise = "gait"`; see the
[Exercise Dashboard](biomech_exercise.md) page for the engine-selection and
widget-swapping mechanics.

The gait models themselves live in the sibling engine repositories. As an
example of what "standardized gait model" means there, the MuJoCo builder
(`exercises/gait/gait_model.py` in the MuJoCo model checkout) documents a
bipedal sagittal-plane model with no barbell, an initial pose at right heel
strike with slight forward lean and asymmetric hip, knee and ankle angles, and
the MuJoCo Z-up gravity convention. That is the sibling repository's
documentation, not this repository's implementation.

## Limitations

- No gait analysis code exists in this repository. This tile is a preset that
  selects an exercise name; the model and its analysis are supplied by the
  sibling engine checkouts.
- The ZMP and locomotion stack is not involved. `src/robotics/locomotion/`
  (ZMP computer, footstep planner, gait state machine, gait types) is
  documented in [zmp_gait_scope.md](../architecture/zmp_gait_scope.md) as dead
  code with respect to the golf-swing use case: nothing outside that subtree
  constructs any of its classes, its only runtime consumers are its own unit
  tests, and the audit recommends archiving it. Do not expect this tile to use
  it, and do not read its ZMP output as this tile's output.
- Nothing is available without a sibling checkout. If no model source exposes
  `exercises/gait`, the dashboard falls back to a static engine list whose
  entries will then fail to load.
- OpenSim shows a placeholder. Selecting `OpenSim_Models` yields the literal
  text "OpenSim dashboard not yet available."
- The exercise cannot be changed in the open window; only the engine can.

## Unclear

The template's Outputs section cannot be completed from this repository.

- What gait-specific quantities the dashboards report, and in which units.
  Checked `src/launchers/exercise_dashboard.py`,
  `src/shared/python/biomech/exercise_registry.py`,
  `src/shared/python/config/model_source_providers.py` and
  `src/shared/python/config/tile_target_resolution.py`. None of them names a
  gait metric; the shell only forwards `exercise_filter="gait"` into
  `MuJoCoDashboard`, `DrakeDashboard`, `PinocchioDashboard` or
  `JaxSimDashboard`, and the gait models are in sibling checkouts outside this
  repository.
- Whether the four engines produce comparable gait results. The tile is
  described as "standardized ... across engines", but no cross-engine gait
  parity contract or test was found in this checkout.
- Which gait cycle events, if any, are detected. `src/shared/python/analysis/`
  contains `phase_detection.py` for swing phases; no gait-cycle equivalent was
  found.

## See Also
- [Exercise Dashboard](biomech_exercise.md)
- [Sit-to-Stand Model](biomech_sit_to_stand.md)
- [ZMP and gait scope audit](../architecture/zmp_gait_scope.md)
- [Biomechanics workspace architecture](../architecture/biomech_workspace.md)
