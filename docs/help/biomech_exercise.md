---
title: Exercise Dashboard
tile_id: biomech_exercise
status: complete
---

# Exercise Dashboard

## Purpose

The Exercise Dashboard is a cross-engine shell for biomechanics exercise
workflows. It is a single window with an engine selector in its toolbar and an
engine-specific dashboard in its body; changing the selector swaps the inner
dashboard while keeping the chosen exercise fixed. Entry point:
`src/launchers/exercise_dashboard.py` (`ExerciseDashboard`, plus
`get_dockable_ui()` for embedding as a launcher tab).

The tile declares the capabilities `injury_risk`, `joint_stress`,
`exercise_presets` and `swing_modification`. Those analyses are implemented by
the per-engine dashboards this shell hosts, not by the shell itself.

## Inputs

| Input | How it is supplied | Unit or values |
| --- | --- | --- |
| Exercise name | `--exercise` argument, else the `BIOMECH_EXERCISE` environment variable, else `gait` | identifier string, for example `gait`, `sit_to_stand` |
| Preferred engine | `BIOMECH_ENGINE` environment variable, or the `preferred_engine` constructor argument | one of the discovered engine names |
| Available engines | `discover_exercise(exercise)` in `src/shared/python/biomech/exercise_registry.py` | list of engine names |
| Engine selection | toolbar combo box, at runtime | one of the listed engines |

`discover_exercise` asks each registered model source (`_MODEL_SOURCES` in
`src/shared/python/config/model_source_providers.py`) for its root and reports
the engine when either `<root>/exercises/<exercise>` or
`<root>/src/<provider>/exercises/<exercise>` exists. If discovery returns
nothing, the shell falls back to offering MuJoCo, Drake, Pinocchio and JaxSim.
`JaxSim_Models` is always appended even when not discovered on disk, because it
is a dependency-light analysis backend with no sibling model repository
(issue #6658).

## Outputs

| Output | Description |
| --- | --- |
| Window title | `Biomechanics Exercise: <Exercise>` |
| Embedded dashboard widget | `MuJoCoDashboard`, `DrakeDashboard`, `PinocchioDashboard` or `JaxSimDashboard`, each constructed with `exercise_filter=<exercise>` |
| Engine-load error panel | a word-wrapped `QLabel` with object name `engine-load-error` when the selected engine cannot be imported or constructed |
| OpenSim placeholder | the literal text "OpenSim dashboard not yet available." |

All numeric results, plots and risk scores come from the embedded dashboard.
This module produces no metrics of its own.

## Method

`ExerciseDashboard.__init__` builds the toolbar, populates the engine list,
selects `preferred_engine` when it is present in that list (otherwise the first
entry) and calls `_on_engine_changed`.

`_on_engine_changed(name)` deletes the current inner widget, then lazily
imports and constructs the dashboard for the requested engine from
`src/launchers/`. Each import is deferred to the branch that needs it, so a
missing optional engine dependency does not prevent the window from opening. If
the embedded widget is itself a `QMainWindow`, its `Qt.WindowType.Window` flag
is cleared so it docks without a title bar; the in-body identity strip
(`ModelLoadStatus`, issue #8829) exists precisely because that removes the
title bar as an identity cue.

Construction failures are caught at the optional-engine boundary, logged with a
traceback, and replaced by `_engine_load_error_widget`. That helper special-cases
a MuJoCo DLL failure with an actionable message pointing the user at
`JaxSim_Models` and including the technical detail.

The virtual tiles `biomech_gait` and `biomech_sit_to_stand` dispatch into this
same script through `BiomechExerciseHandler`
(`src/launchers/launcher_model_handlers.py`), which reads the tile's `exercise`
field and launches `src/launchers/exercise_dashboard.py`. The registry mapping
lives in `VIRTUAL_PREFIXES` in
`src/shared/python/config/tile_target_resolution.py`.

## Limitations

- The shell computes nothing. Injury risk, joint stress and swing modification
  are the responsibility of the embedded engine dashboard.
- Exercise content lives outside this repository. The MuJoCo, Drake, Pinocchio
  and OpenSim exercise models come from sibling checkouts. If those checkouts
  are absent, `discover_exercise` returns an empty list and the fallback engine
  list will offer engines that then fail to load.
- OpenSim is a stub. Selecting `OpenSim_Models` shows a placeholder label; no
  OpenSim dashboard exists.
- No exercise switching at runtime. The exercise is fixed for the lifetime of
  the window; only the engine can be changed. Re-launch with a different
  `--exercise` or `BIOMECH_EXERCISE` value to switch.
- No persistence. Engine choice, window geometry and dashboard state are not
  saved between runs.
- Exercise names are not validated. An unknown name simply discovers no engines
  and lands on the fallback list.

## See Also
- [Gait Model](biomech_gait.md)
- [Sit-to-Stand Model](biomech_sit_to_stand.md)
- [Engine Selection](engine_selection.md)
- [Biomechanics workspace architecture](../architecture/biomech_workspace.md)
- [Launchers](../user_guide/launchers.md)
