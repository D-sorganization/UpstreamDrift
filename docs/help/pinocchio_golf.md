---
title: Pinocchio
tile_id: pinocchio_golf
status: complete
---

# Pinocchio

## Purpose

The Pinocchio tile loads a URDF golfer, poses or free-runs it, and reports the
rigid-body dynamics quantities Pinocchio computes very fast: mass matrix,
Jacobians, manipulability ellipsoids, kinetic and potential energy, induced
accelerations and counterfactuals. Use it for quick kinematic and dynamic
introspection of a model, not for a driven, controlled swing.

## Inputs

| Input | Unit / type | Notes |
| --- | --- | --- |
| URDF model | file (`*.urdf`, `*.xml`) | `Load File...` opens a file dialog; a default `models/generated/golfer.urdf` is loaded at start-up when present. |
| Discovered models | enumeration | Every `*.urdf` under the shared URDF directory returned by `get_shared_urdf_path()`. |
| Operating mode | dynamic / kinematic | `operating_mode` starts as `"dynamic"`. |
| Joint position | rad | One slider plus spin box per joint; slider span is +/- 10.0 rad (`SLIDER_RANGE_RAD`) with a 100 counts/rad scale (`SLIDER_SCALE`). |
| Physics time step | s | 0.01 s (`DT_DEFAULT`). |
| Visualization toggles | boolean | Reference frames, centre-of-mass spheres (radius 0.02 m, `COM_SPHERE_RADIUS`), manipulability ellipsoids. |
| Points of interest | selection | Checkbox list of frames for manipulability analysis. |
| Live analysis | boolean | Computes induced accelerations and counterfactuals every recorded frame. |

## Outputs

| Output | Unit / type | Notes |
| --- | --- | --- |
| MeshCat 3D scene | rendered frames in a browser | Server started on port 7000 where supported; the host URL is printed into the log panel. |
| Model size readout | `nq`, `nv` counts | Logged on load. |
| Simulation time | s | `sim_time`. |
| Joint positions / velocities | rad, rad/s | Recorded per frame. |
| Joint torques | N*m | Recorded per frame - always zero in the current loop (see Limitations). |
| Kinetic energy | J | `data.kinetic_energy`. |
| Potential energy | J | `data.potential_energy`. |
| Club-head position | m (3-vector) | From the first frame whose name contains `club` or `head`, else the last frame. |
| Club-head velocity | m/s (3-vector) | `pin.getFrameVelocity` on the same frame. |
| Induced accelerations | rad/s^2 per generalized coordinate | Per source. |
| Counterfactuals | rad/s^2 (`ztcf_accel`) and N*m (`zvcf_torque`) | Per frame. |
| Jacobian condition number | dimensionless | *Matrix Analysis* readout. |
| Constraint rank | count | *Matrix Analysis* readout. |
| Recorded frame count | count | `Frames: N` label. |
| Exported statistics | CSV | `Export CSV` in the Analysis tab. |

## Method

Pinocchio's Featherstone-family algorithms do the work, called directly from the
Qt layer:

- Forward dynamics: `pin.aba(model, data, q, v, tau)`, then explicit Euler on
  velocity (`v += a * dt`) and `pin.integrate(model, q, v * dt)` on
  configuration - see `_advance_physics` in
  `src/engines/physics_engines/pinocchio/python/pinocchio_golf/gui_simulation.py`.
- Mass matrix: `pin.crba`. Inverse dynamics: `pin.rnea`. Kinematics:
  `pin.forwardKinematics` plus `pin.computeJointJacobians`
  (`pinocchio_visualization_mixin.py`, `manipulability.py`).
- Energy: `pin.computeKineticEnergy` and `pin.computePotentialEnergy` are called
  **separately** and read back off `data.kinetic_energy` / `data.potential_energy`.
  There is no `computeTotalEnergy` call; any total is the sum of the two.
- Induced accelerations and ZTCF/ZVCF: repeated `pin.aba` evaluations with
  gravity or torque zeroed, in `induced_acceleration.py`.

`PinocchioGUI` (`gui.py`) composes `UISetupMixin`, `SimulationMixin`,
`PinocchioAnalysisMixin`, `PinocchioVisualizationMixin` and the shared
`SimulationGUIBase`. The engine adapter used elsewhere is
`PinocchioPhysicsEngine` - see
[Pinocchio engine reference](../engines/pinocchio.md).

## Limitations

- Requires the `pinocchio` wheel; `PINOCCHIO_AVAILABLE` goes false without it.
  Visualisation additionally requires `meshcat`, and degrades to "Model loaded
  without 3D visualization" when it is missing.
- **The dynamic loop is passive.** `_advance_physics` and `_record_frame` both
  set `tau = np.zeros(model.nv)`. There is no actuator, controller or torque
  input in the simulation path, so a "run" is free motion under gravity from
  the current pose. Recorded joint torques are therefore always zero.
- Integration is explicit (semi-implicit) Euler at a fixed 0.01 s step. No
  adaptive stepping, no energy-conserving integrator.
- No contact and no collision response: geometry is loaded for visualisation,
  and there is no ground or ball interaction.
- Club-head identification is a name heuristic. If no frame name contains
  `club` or `head`, the last frame in the model is used instead, silently.
- Joint slider limits are a flat +/- 10 rad UI range, not the URDF's joint
  limits.

## See Also
- [Pinocchio engine reference](../engines/pinocchio.md)
- [Engine capabilities matrix](../engines/engine_capabilities.md)
- [Engine support tiers](../engines/support_tiers.md)
- [Engine selection guide](engine_selection.md)
- [Analysis tools](analysis_tools.md)
- [Visualization](visualization.md)
- [Pinocchio Models library tile](pinocchio_models_shared.md)
