# MDL reference copies — agent-readable model surface

These are text-format (MDL) exports of the binary `.slx` Simulink models in
`../`. They exist solely so that LLM-based agents (and humans without MATLAB
open) can grep / read the underlying model structure: block names, parameter
expressions, model-workspace declarations, and Stateflow chart bodies.

**Authoritative model is still the `.slx`.** These MDLs are a snapshot for
inspection — never load them with `sim()` and never edit them and expect the
edits to round-trip. If the .slx is changed, regenerate these by exporting via
`Simulink.MDLInfo` / "Save As" → MDL. The MDL XML format is documented at
<https://www.mathworks.com/help/simulink/slref/save-system.html>.

| File                                     | Maps to                                                                        | Lines   |
| ---------------------------------------- | ------------------------------------------------------------------------------ | ------- |
| `GolfSwing3D_Kinetic.mdl`                | `../GolfSwing3D_Kinetic.slx` (top-level model)                                 | ~82,000 |
| `Kinetically_Driven_Gimbal_Joint.mdl`    | `../Kinetically_Driven_Gimbal_Joint.slx` (3-DOF rotational joint subsystem)    | ~3,500  |
| `Kinetically_Driven_Revolute_Joint.mdl`  | `../Kinetically_Driven_Revolute_Joint.slx` (1-DOF rotational joint subsystem)  | ~2,300  |
| `Kinetically_Driven_Universal_Joint.mdl` | `../Kinetically_Driven_Universal_Joint.slx` (2-DOF rotational joint subsystem) | ~3,000  |

## Things worth grepping for

```bash
# Confirm Simscape full-block logging is persistent in the model
grep "SimscapeLogType" GolfSwing3D_Kinetic.mdl
# → "all" at line 4256 — the home-license workaround that surfaces every
#   block's state in CombinedSignalBus / simlog without virtual signal markers.

# List every model-workspace parameter (segment lengths, masses, etc.)
grep -E "<Identifier>(.*Length|.*Width|.*Mass)</Identifier>" GolfSwing3D_Kinetic.mdl

# Find Stateflow-chart equations (control law, dampening selector, ...)
grep -E "JointTorque[A-Z]" GolfSwing3D_Kinetic.mdl

# Confirm a joint subsystem's output bus contents
grep '<P Name="Name">' Kinetically_Driven_Revolute_Joint.mdl
# → AngularPosition / GlobalPosition / Rotation Transform / SignalBus / etc.
```

## Architecture notes derived from these files

- **`SimscapeLogType='all'`** is set at the model level (line 4256). This is
  the home-license workaround documented in
  [src/functions/dataset_generator/setModelParameters.m](../../functions/dataset_generator/setModelParameters.m):
  it makes Simscape automatically log every block's state into `simlog`
  without spending virtual signals.

- **`LocalDampeningEnable`** is a Stateflow PARAMETER_DATA (SSID 98) declared
  in the model workspace (line ~6883) and referenced as a 1/0 selector inside
  every joint Stateflow chart and inside every kinetic-driver subsystem's
  damping-coefficient parameter expression
  (`Dampening*LocalDampeningEnable*DampeningGlobalGain`). It must **never** be
  overwritten by the per-trial input MAT files — those contain only the 595
  variable inputs (torques, gains, set-points, dampening tuning) and do not
  include structural parameters like `LocalDampeningEnable` or segment
  lengths.

- **Body-landmark world positions** flow from each joint subsystem's
  `SignalBus` output port (the joint Subsystem MDLs list `GlobalPosition`,
  `GlobalVelocity`, `GlobalAcceleration`, `Rotation Transform`,
  `AngularPosition`, `AngularVelocity`, `ConstraintForceLocal`,
  `ConstraintTorqueLocal`, `ForceLocal`, `TorqueLocal` as bus elements). The
  parent model bundles those into the named groups visible at
  `simOut.CombinedSignalBus.LSLogs`, `RSLogs`, `LFLogs`, `RFLogs`, etc.

- **Segment lengths** (e.g. `UpperTorsoLength`, `LeftUpperArmLength`,
  `LowerArmLength`, `LeftShoulderWidth`, `HubtoSLength`,
  `LeftWristStandoffLength`) are model-workspace parameters declared around
  lines 6800–8400 of the top-level MDL. They are NOT in the input MAT files
  — the FK helper
  [`motion_matching/shared/compute_skeleton_fk.m`](../../../motion_matching/shared/compute_skeleton_fk.m)
  reads them from the model workspace at runtime.

## Updating the snapshot

```matlab
% From within MATLAB, after editing the .slx:
load_system('GolfSwing3D_Kinetic')
save_system('GolfSwing3D_Kinetic', 'GolfSwing3D_Kinetic.mdl', 'ExportToVersion', 'R2025b')
% Then move the .mdl into this folder, overwriting the snapshot.
```

`save_system` exports MDL only when called explicitly with the `.mdl`
extension; the `.slx` remains the working copy.
