# Starting-Pose Matcher

A PyQt6 desktop tool for aligning a physics-engine model's starting pose
to a motion-capture target frame **before** any motion-matching optimiser
runs.

Why: zero-theta starting poses sit nowhere near top-of-backswing, which
sends gradient-based optimisers into bad local minima. This tool
produces the rigid transform + scale (saved as JSON) that
`fit_swing_full_pipeline` reads as `input_overrides` for the model
workspace.

## Run

```bash
python -m src.tools.starting_pose_matcher
```

Or from the **Starting-Pose Matcher** tile in the UpstreamDriftLauncher (registered
in `src/config/models.yaml`).

Requires the `gui-tools` extra:

```bash
pip install upstream-drift[gui-tools]
```

## What's in this package

| File                   | Purpose                                                                                                                                                                                                |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `core.py`              | Pure-data + math (no Qt). Dataclasses, xlsx loader, trajectory CSV loader, shaft-snap math, FK-derived fallback skeletons.                                                                             |
| `gui.py`               | PyQt6 `QMainWindow`. ~2,400 lines: pose slots, frame scrubber, playback (mocap / skeleton / both), shaft-snap auto-align, transform sliders, traces, session save/load, help popups, resizable layout. |
| `skeleton_provider.py` | Abstract source of skeleton joints. Today: JSON (Simscape). Future: MuJoCo / Drake / Pinocchio / OpenSim (issue #4367).                                                                                |
| `__main__.py`          | `python -m` entry point.                                                                                                                                                                               |
| `README.md`            | This file.                                                                                                                                                                                             |

## Shared infrastructure used (per AGENTS.md)

- `src.shared.python.motion_matching.diagnostics.forward_kinematics` —
  evaluates the canonical reference-golfer joint angles into Cartesian
  joint positions. Used to BUILD the fallback Address + Top-of-
  Backswing skeletons; we no longer hand-tune Cartesian dicts.
- `src.shared.python.motion_matching.diagnostics.reference_pose.reference_golfer_setup` —
  canonical address-pose joint angles (degrees). Single source of truth.
- `src.shared.python.motion_matching.diagnostics._skeleton_render.equalize_3d_axes` —
  fits the 3D axis bounds to the data; toggle via the "Auto-fit axes"
  checkbox in the View / Mocap Traces group.

## Inputs

- **Wiffle xlsx** (`Wiffle_ProV1_club_3D_data.xlsx`). Positions are in
  **centimetres** (despite the spreadsheet's "Definitions" tab claiming
  inches — see `MATLAB_GOLF_MODEL_GUIDE.md` line 354).
- **`simscape_skeleton_<pose>.json`** produced once by
  `export_default_skeleton.m` (MATLAB-side; runs a 5 ms simulation and
  dumps joint positions for Address / Top of Backswing / Impact).
  Fallback: FK-derived from the canonical reference pose.
- **Simscape forward-dynamics CSV** (optional). Either the short-form
  `<joint>_X/Y/Z` schema used by `motion_capture_plotter_data` or the
  long-form `<Joint>Logs_…GlobalPosition_1/2/3` raw-bus schema.

## Outputs

- `starting_pose_offsets.json` — the 7-DOF rigid transform (Tx/Ty/Tz/
  Rx/Ry/Rz/Scale). Consumed by `solve_starting_pose.m` ↔
  `fit_swing_full_pipeline.m` ↔ `simulate_with_coefficients.m` via
  `opts.stage2_opts.sim.input_overrides`.
- `<sheet>_<timestamp>.session.json` — full UI snapshot for resumable
  alignment sessions.

## Tests

`tests/unit/engines/simscape/three_d_gui/test_starting_pose_matcher.py`
— 45+ tests covering RigidTransform algebra, shaft-snap solver, fallback
skeleton modelling, xlsx unit handling, phase windowing, session
round-trip, FK-derived fallback (this iteration). UI smoke tests skip
cleanly when PyQt6 fails to import in the test environment.

## Help text

Every interactive control in the matcher exposes a non-empty Qt
`toolTip()`. A coverage test (`tests/ui/test_help_coverage.py`) walks
the widget tree under `QT_QPA_PLATFORM=offscreen` and asserts that a
curated whitelist of widgets all have hover-text. See issue #4492 for
the full sweep scope.

A "?" help button at the top-right of every section opens a topic
panel describing what the section does and what the controls mean.

## Coming soon

The following panels are referenced by the issue tracker but may not
have landed in this branch yet. The matcher is intended to grow them in
sequence:

- **Source-toggle panel** (#4482) — switches between motion-target
  sources (xlsx, parquet, live-streamed mocap) without reloading the
  whole tool. The current build assumes xlsx via the **Mocap Source**
  group.
- **Animated preview** (#4481) — refines the playback panel with
  better speed / loop / trail controls and a more compact timeline.
  Today's playback box already supports frame, slider, play/pause,
  speed, loop, target switching, and event marking.
- **Skeleton-provider combo** (#4367) — extends the fixed JSON-based
  Simscape provider so users can pick Drake / MuJoCo / Pinocchio /
  OpenSim / Simscape / MediaPipe / OpenPose at runtime. Until that
  lands, the provider is fixed and is configured in code.

## Related issues

- #4363 frame scrubber + playback ✅ closed
- #4364 swing-phase windowing ✅ closed
- #4365 session save / load ✅ closed
- #4366 input-MAT editor with constraint-resolved overlay (open)
- #4367 port to MuJoCo / Drake / Pinocchio (open) — uses
  `skeleton_provider.SkeletonProvider`
- #4376 relocate + adopt shared FK + declare deps (this PR)
- #4377 AGENTS.md — shared-infra directory + workflow
- #4481 animated preview / timeline polish (open)
- #4482 source-toggle panel (open)
- #4486 layer-visibility refinements (open)
- #4492 in-app help-text sweep (this PR)
