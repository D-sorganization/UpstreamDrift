# Handoff Archive: Completed Visuals Lane

Visuals plan steps 1, 3, and 4. Moved verbatim from the [full-body handoff](../HANDOFF.md).

## Visual Skeleton Layer Completed (Step 1 of the Visuals Plan)

- `src/shared/python/motion_matching/visual_skeleton.py`: one engine-agnostic
  visual description derived from any native or full-body spec (capsule per
  body, COM and frame spheres, ground plane opposite gravity with a
  `calibrated` flag, world-segment mapping for viewers); 5 unit tests.
- `src/engines/physics_engines/mujoco/python/visual_layer.py` and
  `export_full_body_mjcf(..., visual=True)`: massless non-colliding visual
  geoms, floor, lights, camera; physics proven identical to the plain export
  (masses, inertias, xpos, qacc) in `tests/unit/motion_matching/test_mujoco_visual_layer.py`.
- Evidence `evidence/visual_layer/` (default pose PNG, returned81 kinematic
  playback GIF, MJCF, receipt). Steps 3 to 6 (viewer tile, cross-engine replay
  videos, OpenSim real horizon, FB-3 re-checks) are specified for lower-level
  agents in [VISUALS_HANDOFF.md](../VISUALS_HANDOFF.md).
- Capsule radius policy (user review 2026-09-13: club and segments looked
  uniformly thick): radius of a uniform-density cylinder,
  `sqrt(m / (pi * 1500 kg/m^3 * L))`, clamped to 6 to 50 mm, and one capsule per
  child joint plus one to the farthest centre of mass. Club shaft 17 mm,
  forearms 36 to 40 mm, torso and thighs 50 mm (`address_segments.json`).
- Address-pose evidence at the returned81 candidate's q0 (the Simscape
  qualified start, legs at zero): `address_{front,side,top}.png` with frame-0
  capture markers (black) and native marker predictions (blue) overlaid
  (`render_address_views.py`, frame-0 valid marker RMS 2.9e-13 m), the
  labelled joint-to-joint Simscape skeleton `simscape_skeleton_address.png`
  with `address_posture.json` (`render_simscape_skeleton.py`), and
  `render_playback.py` for the default pose and playback GIF. No MATLAB is
  runnable on DeskComputer (`C:/Program Files/MATLAB/R2025b` has no
  `bin/matlab.exe`), so the Simscape skeleton is drawn from the ported
  geometry whose frame parity against R2025b is receipted at 2.1e-13 m
  (`native_frame_parity_r2025b.json`).
- Posture facts the user should decide on (model and start pose, not
  rendering): Simscape geometry inputs UpperArmLength 14.5 in (0.368 m) and
  LowerArmLength 12 in (0.305 m elbow to wrist); hub to shoulder 0.254 m;
  the hip, torso and spine joints coincide, so the pelvis-to-rod bend of
  -16.9 deg (SpineInputX) at q0 is what the frame-0 marker fit needed; the
  shoulders sit 0.09 m (L) and 0.19 m (R) below the hub at q0 (scapula
  coordinates 30/-40 deg and 35/38 deg). A straight spine or higher
  scapulae at address means refitting q0 (or re-calibrating attachments)
  with those coordinates constrained, then re-qualifying.

## Tour Matching Viewer Launcher Tile Completed (Step 3 of the Visuals Plan)

- Implemented pure-data engine-agnostic replay and forward kinematics reader
  `src/tools/tour_matching_viewer/core.py` (`load_replay`, `body_poses_from_state`,
  `viewer_frame`). Supports native `returned-replay.npz` and OpenSim IK `.mot`
  formats. Forward kinematics derived pure-Python from `full_body_spec_v1.json`
  evaluates to $< 10^{-12}$ error against MuJoCo forward kinematics across all 24
  bodies over 20 random states without importing `mujoco`.
- Implemented PyQt6/matplotlib 3D viewer `TourMatchingViewerWidget` and
  `TourMatchingViewerWindow` (`src/tools/tour_matching_viewer/gui.py`) with scrub
  slider, playback controls, target/model markers, visual skeleton segments,
  ground wireframe, and per-frame valid marker RMS readout.
- Embedded launcher tile via `_TourMatchingViewerEmbedAdapter` (`_embed_adapter.py`)
  registered into `EMBEDDABLE_TOOL_REGISTRY` and configured in `src/config/models.yaml`
  (`tool_id: tour_matching_viewer`, category `tool`, status `beta`).
- 10 unit tests passing in `tests/unit/tools/test_tour_matching_viewer_core.py` and
  `tests/unit/tools/test_tour_matching_viewer_adapter.py`.
- Rendered offscreen verification evidence archived under
  `docs/development/full_body_models/evidence/viewer/`:
  - `screenshot_returned81.png`
  - `screenshot_mot.png`
  - `receipt.json`
  - `returned81_replay.npz`
  - `opensim_os3b_ik.mot`

## Visuals Handoff Step 4 Completed: Same-Input Replays in MuJoCo, Pinocchio, and Drake (#10062)

- Replayed returned81 candidate trajectory (`returned-candidate.json`, SHA256 `dfafdff1cdec1a7fa15c41a34d41f054ef45d8a7df0a1faf8fab7898ab855776`) across all three physics engines (Pinocchio, MuJoCo, Drake).
- Shared 5-metric evaluator and standardized NPZ reader/writer implemented in `src/shared/python/motion_matching/replay_metrics.py`.
- Evaluated 307 frames on candidate coordinates:
  - MuJoCo: whole_rms_m 0.026366, early_rms_m 0.011427, terminal_rms_m 0.046305, club_cluster_rms_m 0.015955, pelvis_yaw_error_pct 13.923%. Max marker error vs Pinocchio: $2.35 \times 10^{-15}$ m.
  - Pinocchio: whole_rms_m 0.026366, early_rms_m 0.011427, terminal_rms_m 0.046305, club_cluster_rms_m 0.015955, pelvis_yaw_error_pct 13.923%.
  - Drake: whole_rms_m 0.026365, early_rms_m 0.011426, terminal_rms_m 0.046305, club_cluster_rms_m 0.015953, pelvis_yaw_error_pct 13.923%. Max marker error vs Pinocchio: $1.62 \times 10^{-5}$ m.
- Evidence archived under `docs/development/full_body_models/evidence/replays/`:
  - Three animated visual GIFs: `mujoco_returned81.gif`, `pinocchio_returned81.gif`, `drake_returned81.gif`.
  - Three standardized NPZ replays: `mujoco_returned81_replay.npz`, `pinocchio_returned81_replay.npz`, `drake_returned81_replay.npz`.
  - Engine receipts: `mujoco_receipt.json`, `pinocchio_receipt.json`, `drake_receipt.json`.
  - Combined cross-engine receipt: `receipt.json`.
- Tests: `tests/unit/motion_matching/test_returned81_cross_engine_replay.py` (3 passed).
