---
title: Pose Studio
tile_id: pose_studio
status: complete
---

# Pose Studio

## Purpose

Hand-build a single body pose with per-joint sliders and see it rendered
through whichever physics engine you select - Drake, MuJoCo, Pinocchio,
OpenSim or Simscape - without restarting. The pose you author is a
`CanonicalPose`, the engine-agnostic interchange type that also seeds
the [Motion-Match Preview](motion_target_preview.md) and the motion
matcher's target.

## Inputs

| Input | Unit / type | Notes |
| --- | --- | --- |
| Active engine | one of `drake`, `mujoco`, `pinocchio`, `opensim`, `simscape` | The picker only offers engines present in *both* the adapter registry and the kinematics-service registry. Missing wheels fall back to a mock, flagged by a yellow status pill. |
| Joint angles | **degrees**, range -180.0 to +180.0 | Accordion of per-joint spinboxes and sliders, grouped by body region. Sliders tick in tenths of a degree. |
| Angle display unit | degrees (default) or radians | A display toggle only: internal state is always kept in degrees, and the sliders always tick in tenths of a degree. |
| Pelvis translation | **metres**, 3-vector | Part of `CanonicalPose`. |
| Pelvis rotation | **degrees**, intrinsic XYZ Euler, 3-vector | Part of `CanonicalPose`. |
| Pose Library selection | `canonical_zero_pose` or `canonical_from_reference_setup` | The two starting points offered. |
| 3D landmark click | pointer input | Highlights the picked landmark. |
| Undo / redo | Ctrl+Z / Ctrl+Shift+Z | Stack of `CanonicalPose` snapshots. |

## Outputs

| Output | Unit / type | Notes |
| --- | --- | --- |
| `CanonicalPose` (in-process) | `pelvis_translation_m` in metres, `pelvis_rotation_xyz_deg` in degrees, `joint_angles_deg` in degrees, `convention_tag` = `"canonical-v1"` | The edited pose. Joint-angle keys must be a subset of `REFERENCE_GOLFER_FIELDS`; unknown names raise. Missing keys default to 0.0 degrees on round-trip. |
| 3D skeleton view | metres | Matplotlib 3D rendering of the current pose under the active engine's forward kinematics. |
| Engine status pill | enumerated status | Whether the selected engine is live or mocked. |
| Units badge | text | The active engine's *native* convention, for information only: Drake URDF / RPY (rad), MuJoCo MJCF / Euler (rad), Pinocchio URDF / RPY (rad), OpenSim `.osim` / coordinates (rad), Simscape Parameters / RPY (deg). |

There is no file output. See Limitations.

## Method

The tool separates pure data from Qt. `core.py` holds the engine list
and the joint-region layout with no Qt import; `controllers/` holds the
`EngineController` (which owns the active `LiveKinematicsService` and
adapter) and the `HistoryController` (the undo/redo snapshot stack);
`widgets/` holds the engine picker, joint accordion, matplotlib 3D view
and units badge; `gui.py` is layout and signal wiring only. The
structure and its test split are documented in the tool's
[README](../../src/tools/pose_studio/README.md).

The pose type itself is `CanonicalPose` in
[`canonical.py`](../../src/shared/python/pose_interchange/canonical.py),
which validates shapes and field names on construction and marks its
arrays non-writeable. Angles in the joint panel are *always* displayed
in the canonical convention regardless of the engine's own convention;
the units badge exists to tell you what that engine reports natively,
and does not change any value you see.

## Limitations

- **Save and load are stubs.** The buttons carry a tooltip saying so:
  formats are deferred to issue #4900. You cannot persist a pose from
  this tile, and you cannot reload one - only pick from the two-entry
  Pose Library.
- **No inverse kinematics and no drag handles.** You cannot grab a
  clubhead and have the chain solve back. That is explicitly out of
  scope for v1.
- **No mocap scrubbing.** It edits one static pose. Playing a captured
  trajectory is a deferred follow-up (#4901); use
  [Motion-Match Preview](motion_target_preview.md) for that.
- **An engine may silently be a mock.** When the engine's wheel is not
  installed the picker still offers it and falls back to a mock service,
  indicated only by a yellow pill. Kinematics you see in that state are
  not that engine's kinematics. Check the pill.
- **The engine list is not fixed.** It is the runtime intersection of
  two registries, so an engine can be absent from the picker entirely on
  a given install.
- **Not a simulation.** No dynamics, no contact, no time. It is a pose
  editor.
- The tile is registered at maturity `beta`.

## See Also
- [Pose Studio README](../../src/tools/pose_studio/README.md)
- [Motion-Match Preview](motion_target_preview.md) - consumes the canonical pose as a starting-pose seed
- [Motion matching](../motion_matching/README.md)
- [Motion pipeline workflow guide](../motion_pipeline/README.md) - which engine to pick for what
- [ADR-0007: motion pipeline architecture](../adr/0007-motion-pipeline-architecture.md)
- [Engine selection](engine_selection.md), [Visualization](visualization.md)
