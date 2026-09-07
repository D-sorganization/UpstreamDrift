---
title: Sit-to-Stand Model
tile_id: biomech_sit_to_stand
status: stub
---

# Sit-to-Stand Model

## Purpose

Sit-to-Stand Model is an exercise preset, not a separate application. It is a
virtual tile: the registry gives it the pseudo-path
`virtual/biomech_exercise/biomech_sit_to_stand` and the field
`exercise: sit_to_stand`, and `VIRTUAL_PREFIXES` in
`src/shared/python/config/tile_target_resolution.py` maps the
`virtual/biomech_exercise/` namespace onto the backing script
`src/launchers/exercise_dashboard.py`. Opening the tile opens the
[Exercise Dashboard](biomech_exercise.md) with the sit-to-stand exercise
selected.

The registry describes it as a "Standardized sit-to-stand motion model" with
the capabilities `biomechanics`, `exercise_preset` and `sit_to_stand`.

## Inputs

| Input | How it is supplied | Unit or values |
| --- | --- | --- |
| Exercise name | fixed to `sit_to_stand` by the tile's `exercise` field | identifier string |
| Engine | `BIOMECH_ENGINE`, or the dashboard's toolbar combo box | `MuJoCo_Models`, `Drake_Models`, `Pinocchio_Models`, `JaxSim_Models`, `OpenSim_Models` |
| Sit-to-stand model definition | the selected engine's sibling model checkout, under `exercises/sit_to_stand` | engine-native model format |

`BiomechExerciseHandler` in `src/launchers/launcher_model_handlers.py` reads
`model.exercise` and launches the dashboard script.
`discover_exercise("sit_to_stand")` in
`src/shared/python/biomech/exercise_registry.py` decides which engines are
offered by testing for an `exercises/sit_to_stand` directory under each
registered model source.

## Outputs

| Output | Description |
| --- | --- |
| Window title | `Biomechanics Exercise: Sit_To_Stand` (the shell applies `str.title()` to the raw exercise name) |
| Embedded dashboard | the selected engine's dashboard, constructed with `exercise_filter="sit_to_stand"` |

Everything numeric that a sit-to-stand analysis would produce is computed by
the embedded engine dashboard and by the model in the sibling checkout, not by
any module in this repository. See [Unclear](#unclear).

## Method

The tile carries no code of its own. Launching it is exactly the Exercise
Dashboard launch path with `exercise = "sit_to_stand"`; see the
[Exercise Dashboard](biomech_exercise.md) page for the engine-selection and
widget-swapping mechanics.

The models live in the sibling engine repositories. As an example of what the
preset means there, the MuJoCo builder
(`exercises/sit_to_stand/sit_to_stand_model.py` in the MuJoCo model checkout)
documents a chair body welded to the ground with a seat at about 0.45 m, an
initial seated pose at roughly 90 degrees of hip and knee flexion, no barbell,
and the MuJoCo Z-up gravity convention. That is the sibling repository's
documentation, not this repository's implementation.

## Limitations

- No sit-to-stand analysis code exists in this repository. This tile is a
  preset that selects an exercise name.
- Nothing is available without a sibling checkout. If no model source exposes
  `exercises/sit_to_stand`, the dashboard falls back to a static engine list
  whose entries will then fail to load.
- OpenSim shows a placeholder. Selecting `OpenSim_Models` yields the literal
  text "OpenSim dashboard not yet available."
- The exercise cannot be changed in the open window; only the engine can.
- The window title is derived mechanically from the identifier, so it reads
  `Sit_To_Stand` rather than a properly formatted display name.

## Unclear

The template's Outputs section cannot be completed from this repository.

- What sit-to-stand quantities the dashboards report, and in which units.
  Checked `src/launchers/exercise_dashboard.py`,
  `src/shared/python/biomech/exercise_registry.py` and
  `src/shared/python/config/tile_target_resolution.py`; none names a
  sit-to-stand metric. The shell only forwards
  `exercise_filter="sit_to_stand"` into the engine dashboards.
- Whether chair geometry, seat height or rise strategy is configurable from
  this tile. No such parameter is exposed by the dashboard shell or by the
  registry entry.
- Whether the four engines produce comparable results. The tile is described
  as "standardized", but no cross-engine parity contract or test for this
  exercise was found in this checkout.

## See Also
- [Exercise Dashboard](biomech_exercise.md)
- [Gait Model](biomech_gait.md)
- [Biomechanics workspace architecture](../architecture/biomech_workspace.md)
- [Launchers](../user_guide/launchers.md)
