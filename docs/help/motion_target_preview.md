---
title: Motion-Match Preview
tile_id: motion_target_preview
status: complete
---

# Motion-Match Preview

## Purpose

Load one or more motion-capture targets - a club trajectory, body
markers, or both - onto a shared time grid, play them back against a
physics model's skeleton, and hand-align the model's starting pose to
the target frame you care about. The point is to stop a gradient-based
motion-matching optimiser from starting at a zero-theta pose nowhere
near top-of-backswing and falling into a bad local minimum.

The tile is named "Motion-Match Preview" but it launches the
**Starting-Pose Matcher**
([`src/tools/starting_pose_matcher`](../../src/tools/starting_pose_matcher/README.md)).

## Inputs

| Input | Unit / type | Notes |
| --- | --- | --- |
| Club source | `.xlsx`, `.mat` or `.c3d` | Toggleable between *Club only* and *Club + ball*. The bundled `Wiffle_ProV1_club_3D_data.xlsx` holds positions in **centimetres**, despite its own "Definitions" tab claiming inches. |
| Body-markers source | `.c3d` | With a marker-set combo: `Anatomical 28` (default), `Lower body only`, `Upper body only`, `All markers`. |
| Skeleton definition | `simscape_skeleton_<pose>.json` | Exported once from MATLAB by `export_default_skeleton.m` for Address / Top of Backswing / Impact. When absent, a fallback skeleton is derived by forward kinematics from the canonical reference-golfer pose. |
| Forward-dynamics trajectory | `.csv` (optional) | Either the short-form `<joint>_X/Y/Z` schema or the long-form `<Joint>Logs_...GlobalPosition_1/2/3` raw-bus schema. |
| Sample rate | Hz, integer, range 50-10000, default 1000, step 100 | Shared `AlignOptions` row. |
| Duration | seconds, range 0.05-5.0, default 0.300, 3 decimal places, step 0.05 | Shared `AlignOptions` row. |
| Time alignment | `impact` (default) or `address` | Radio pair. Decides which instant the sources are pinned to. |
| Tx / Ty / Tz | **metres**, range -1.500 to +1.500, step 0.001 | Translation sliders. |
| Rx / Ry / Rz | **degrees**, range -180.0 to +180.0, step 0.1 | Rotation sliders. Applied as `Rz @ Ry @ Rx`. |
| Scale | dimensionless multiplier, range 0.50 to 2.00, step 0.01, default 1.00 | Uniform scale about the pivot. |
| Frame | integer frame index | Scrubber plus spinbox. |
| Playback frame rate | Hz (frames per second), range 1-240 | Playback spinbox, labelled `fps`. |
| Playback speed multiplier | dimensionless | One of 0.1, 0.25, 0.5, 1.0, 2.0, 4.0. |
| Trail length | frames, range 0-600, default 30 | Show-trail layer. |

## Outputs

| Output | Unit / type | Notes |
| --- | --- | --- |
| `starting_pose_offsets.json` | 7 degrees of freedom: Tx/Ty/Tz in metres, Rx/Ry/Rz in degrees, Scale dimensionless | The deliverable. Consumed by `solve_starting_pose.m`, `fit_swing_full_pipeline.m` and `simulate_with_coefficients.m` through `opts.stage2_opts.sim.input_overrides`. |
| `<sheet>_<timestamp>.session.json` | JSON | Full UI snapshot, so an alignment session can be resumed. |
| On-screen 3D view | metres | Mocap markers, model skeleton, or both, with optional traces and auto-fitted axes. |
| `MultiSourceTarget` (in-process) | dataclass | Emitted by the data-sources panel whenever a slot changes, or `None` when no slot is loaded. |

## Method

The transform is a 7-DOF similarity applied about a pivot:
`P' = scale * R * (P - pivot) + pivot + t` with `R = Rz @ Ry @ Rx`,
implemented as `RigidTransform` in
[`core.py`](../../src/tools/starting_pose_matcher/core.py). Translations
are metres and rotations are degrees, converted internally with
`deg2rad`.

Fallback Address and Top-of-Backswing skeletons are not hand-tuned
Cartesian dictionaries: they are evaluated from
`motion_matching.diagnostics.reference_pose.reference_golfer_setup`
(canonical joint angles in degrees) through
`motion_matching.diagnostics.forward_kinematics`. Multi-source targets
are validated by the data-sources panel, which requires at least one
loaded slot and rejects sources whose time grids do not match.

The wider matching context is in
[Motion matching](../motion_matching/README.md); the tool's own design
notes and issue history are in its
[README](../../src/tools/starting_pose_matcher/README.md).

## Limitations

- **It does not run the optimiser.** It produces the starting-pose
  offsets that a later stage reads. Nothing is fitted here.
- **Alignment is manual (with an assist).** There is a shaft-snap
  auto-align, but the seven degrees of freedom are otherwise yours to
  set by hand. There is no global pose-fitting solve.
- **The skeleton provider is fixed in code.** The tool's own README
  states that a runtime combo to pick Drake / MuJoCo / Pinocchio /
  OpenSim / MediaPipe / OpenPose has not landed (issue #4367); today
  the JSON-based Simscape provider is configured in code. The README
  also names the file as `skeleton_provider.py`, whereas the package
  ships `skeleton_extractor.py` and a `skeleton_extractors/` package.
- **Sources must share a time grid.** Mismatched grids are rejected
  outright rather than resampled for you.
- **Unit handling is a known trap.** The bundled club spreadsheet is in
  centimetres while its own documentation says inches; the tool
  compensates, but a different spreadsheet may not behave.
- **The offsets file is consumed by MATLAB**, so the downstream half of
  this workflow needs MATLAB with Simscape Multibody.
- Panels referenced by the issue tracker but not necessarily present in
  a given build: source-toggle refinements (#4482), animated-preview
  polish (#4481), input-MAT editor (#4366), layer-visibility
  refinements (#4486).

## See Also
- [Starting-Pose Matcher README](../../src/tools/starting_pose_matcher/README.md)
- [Motion matching](../motion_matching/README.md)
- [Surrogate training guide](../motion_matching/SURROGATE_TRAINING_GUIDE.md)
- [Motion pipeline workflow guide](../motion_pipeline/README.md)
- [Motion pipeline format matrix](../motion_pipeline/formats.md)
- [C3D Viewer](c3d_viewer.md) - inspect a `.c3d` target before loading it here
- [Pose Studio](pose_studio.md) - author the canonical pose this tool seeds from
