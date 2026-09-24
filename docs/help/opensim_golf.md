---
title: OpenSim
tile_id: opensim_golf
status: complete
---

# OpenSim

## Purpose

The OpenSim tile loads an OpenSim musculoskeletal model (`.osim`), runs a
forward-dynamics simulation of it, and plots joint angles, joint torques and the
hand and club-head paths. It is the tile to use when your model is a
musculoskeletal one authored in OpenSim and you want OpenSim itself, rather than
an approximation, to integrate it.

## Inputs

| Input | Unit / type | Notes |
| --- | --- | --- |
| Model file | `.osim` | Passed as a command-line argument to `opensim_gui.py`, or chosen with `Load Model`. There is no default model. |
| Integration time step | s | 0.001 s (`GolfSwingModel.dt`). |
| Simulation duration | s | 1.5 s (`GolfSwingModel.duration`). |
| Gravity | m/s^2 | `-constants.GRAVITY_M_S2`. |
| Arm length | m | 0.7 m. |
| Club length | m | 1.1 m. |
| Arm mass | kg | 5.0 kg. |
| Club mass | kg | 0.4 kg. |
| Peak shoulder torque | N*m | 50.0 N*m. |
| Passive wrist torque | N*m | 1.0 N*m. |

The numeric parameters above are constructor defaults on `GolfSwingModel`
(`src/engines/physics_engines/opensim/python/opensim_golf/core.py`). The tile's
UI does not expose editors for them; they are not user inputs from the GUI.

## Outputs

| Output | Unit / type | Notes |
| --- | --- | --- |
| Joint angles vs time | rad | Plot 1, axis labelled "Joint Angles (rad)"; shoulder and wrist traces. |
| Joint torques vs time | N*m | Plot 2, axis labelled "Joint Torques (Nm)". |
| Swing trajectory | m (2D) | Plot 3: hand path and club-head path in the XZ plane, equal aspect. |
| Muscle forces | N | `SimulationResult.muscle_forces`, one column per muscle in the model. |
| Control signals | dimensionless activations | One column per model control. |
| Marker positions | m (3-vector per marker per step) | One entry per marker in the model's marker set. |
| Simulation summary | s and step count | Status line: duration and number of steps. |
| Status / error dialogs | text | Explicit dialogs for "OpenSim Not Installed", "Model Load Failed", "Model File Not Found". |

## Method

OpenSim performs the integration. `GolfSwingModel._run_opensim_simulation`
builds the storage arrays from the model's own counts
(`getNumCoordinates`, `getNumSpeeds`, `getMuscles`, `getNumControls`,
`getMarkerSet`), calls `initializeState()` and `equilibrateMuscles(state)`, then
steps an `opensim.Manager` for `int(duration / dt)` steps, recording Q, U,
muscle forces, controls and marker positions each step.

The GUI is `MainWidget` in
`src/engines/physics_engines/opensim/python/opensim_gui.py`, hosted by the thin
`OpenSimGolfGUI` `QMainWindow` for standalone launch. Plots are matplotlib on a
`FigureCanvasQTAgg`. A *Screw Kinematics* tab is added when
`src.shared.python.screw_theory.ui.ScrewVisualizationTab` imports successfully.

The engine adapter used elsewhere in the suite is `OpenSimPhysicsEngine` - see
[OpenSim engine reference](../engines/opensim.md).

## Limitations

- **There is no demo or fallback mode.** The module docstring says so
  explicitly. Without an OpenSim install the tile opens, reports "OpenSim Not
  Installed - Setup Required", and the Run button stays disabled.
- No bundled model. `Run Simulation` is disabled until you load a `.osim` file
  yourself.
- `run_simulation` can raise `NotImplementedError`, which the GUI surfaces as
  "OpenSim simulation is not yet fully implemented". Coverage depends on the
  loaded model and the installed OpenSim version.
- The joint-angle and joint-torque plots hard-code the first two state and
  torque columns and label them "Shoulder" and "Wrist". On a model whose
  coordinate ordering differs, those legends are wrong.
- The trajectory plot requires markers literally named `Hand` and `ClubHead`;
  it raises if the loaded model lacks them.
- Joint torques are recorded as an approximation - the array is commented
  `# Approx` in `core.py`.
- Duration, time step and the segment/torque parameters are not editable from
  the GUI.
- No recording/export surface and no live 3D viewport: output is the four
  matplotlib panels only.

## See Also
- [OpenSim engine reference](../engines/opensim.md)
- [Engine capabilities matrix](../engines/engine_capabilities.md)
- [Engine support tiers](../engines/support_tiers.md)
- [Engine selection guide](engine_selection.md)
- [Analysis tools](analysis_tools.md)
- [OpenSim Models library tile](opensim_models_shared.md)
