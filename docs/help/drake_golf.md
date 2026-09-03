---
title: Drake
tile_id: drake_golf
status: complete
---

# Drake

## Purpose

The Drake tile runs a golf-swing multibody model through Drake's
`MultibodyPlant`, visualises it in a MeshCat browser window, and gives you
post-hoc analyses that Drake's exact dynamics make cheap: induced
accelerations, zero-torque and zero-velocity counterfactuals, swing-plane
deviation, and Jacobian conditioning. Use it when you care about rigorous
multibody dynamics and optimisation-adjacent analysis rather than contact-heavy
simulation.

## Inputs

| Input | Unit / type | Notes |
| --- | --- | --- |
| Model selection | enumeration | `Default Golf Model`, plus every `*.urdf` found by the URDF scan (see Method). |
| Operating mode | `Dynamic (Physics)` or `Kinematic (Pose)` | Dynamic integrates; kinematic poses the plant. |
| Joint position | rad (revolute) / m (prismatic) | One slider plus spin box per single-DOF joint; limits are read from the plant, falling back to a UI default range. Multi-DOF joints and welds are skipped. |
| Integration time step | s | Fixed at 1e-3 s (`TIME_STEP_S`); the Qt timer ticks at the same period. |
| Initial pelvis height | m | 1.0 m (`INITIAL_PELVIS_HEIGHT_M`). |
| `MESHCAT_HOST` | env var | Host for the MeshCat server; when set, the browser is not auto-opened (container / headless use). |
| Visualization toggles | boolean | Forces, torques, mobility ellipsoid, force ellipsoid, induced-acceleration vectors, counterfactual vectors, live analysis. |
| Manipulability target bodies | selection | Checkbox grid of plant bodies. |
| Induced-acceleration source | enumeration | `gravity`, `velocity`, `total`, or a named single-DOF joint. |
| Counterfactual type | enumeration | `ztcf_accel` or `zvcf_torque`. |

## Outputs

| Output | Unit / type | Notes |
| --- | --- | --- |
| MeshCat 3D scene | rendered frames in a browser | URL is logged on start-up. |
| Simulation time | s | Status bar. |
| Recorded joint positions / velocities | rad, rad/s (revolute) | `DrakeRecorder.q_history`, `v_history`. |
| Club-head position | m (3-vector) | `club_head_pos_history`. |
| Centre-of-mass position | m (3-vector) | `com_position_history`. |
| Angular momentum | kg*m^2/s (3-vector) | `angular_momentum_history`. |
| Ground reaction forces | N (3-vector) | `ground_forces_history`. |
| Centre of pressure | m (3-vector) | `cop_position_history`. |
| Induced accelerations | rad/s^2 per generalized coordinate | Per source, per frame. |
| Counterfactuals | rad/s^2 (ZTCF) and N*m (ZVCF) | Stored per frame. |
| Jacobian condition number | dimensionless | *Matrix Analysis* readout. |
| Constraint rank | count | *Matrix Analysis* readout. |
| Recorded frame count | count | `Frames: N` label. |
| Exported recording | `.json`, `.csv`, `.mat`, `.hdf5` | The "Export Analysis Data (CSV)" button calls `export_recording_all_formats`, which writes all four formats by default. |

## Method

Drake does the dynamics. `DrakeSimApp`
(`src/engines/physics_engines/drake/python/src/drake_gui_app.py`) builds a
`DiagramBuilder` / `AddMultibodyPlantSceneGraph` pair at `time_step=1e-3`,
parses the selected URDF with `Parser.AddModels` (or calls
`build_golf_swing_diagram` from `drake_golf_model.py` for the default model),
attaches a `MeshcatVisualizer`, and drives a `Simulator` with
`set_target_realtime_rate(1.0)` from a Qt timer.

The GUI is assembled from mixins in the same directory: `drake_gui_ui.py`
(layout), `drake_gui_sim.py` (loop, recording, export), `drake_gui_analysis.py`
(induced accelerations, counterfactuals, swing plane, advanced plots),
`drake_gui_viz.py` (MeshCat overlays), and `manipulability.py`
(`DrakeManipulabilityAnalyzer`). Recording is handled by `DrakeRecorder` in
`drake_analysis.py`.

The engine adapter used elsewhere in the suite is `DrakePhysicsEngine` in
`src/engines/physics_engines/drake/python/drake_physics_engine.py` - see
[Drake engine reference](../engines/drake.md).

## Limitations

- Requires `pydrake`. Every Drake symbol falls back to `None` on import
  failure, so the tile cannot simulate without it.
- Visualisation is MeshCat in a separate browser tab, not an in-Qt viewport.
  If `Meshcat()` fails to start, the tile logs the failure and runs with the
  visualiser disabled.
- Kinematic sliders only cover joints with exactly one position coordinate.
  Welds and multi-DOF joints are explicitly skipped and cannot be posed here.
- Model discovery is narrow: the tile globs `*.urdf` from `/shared/urdf` (when
  present) or `<repo>/shared/urdf`. It does not read the biomech sibling model
  repositories.
- The plotting and counterfactual buttons are disabled without matplotlib.
- No trajectory optimisation surface. Drake's solvers are not exposed by this
  tile even though the registry lists an `optimization` capability for it.
- If `drake_golf_model` cannot be imported, `build_golf_swing_diagram` falls
  back to a stub that returns `(None, None, None)` and the tile builds an empty
  finalised plant instead of a golfer.

## See Also
- [Drake engine reference](../engines/drake.md)
- [Engine capabilities matrix](../engines/engine_capabilities.md)
- [Engine support tiers](../engines/support_tiers.md)
- [Engine selection guide](engine_selection.md)
- [Analysis tools](analysis_tools.md)
- [Visualization](visualization.md)
- [Drake Models library tile](drake_models_shared.md)
