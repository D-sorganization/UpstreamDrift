# Markerless Motion Capture: Turnover

Issue #9732. Keep this current with every PR that touches
`src/motion_capture/`, `src/tools/capture_rig/` or the pose estimators. Last
update: 2026-09-08 (Simscape axis validation, #9714: `golfer-scapula/2.0`).

## Stage Map

| Stage                | Command / tile action                                                                        | Reads                               | Writes                                                                          | Evidence                                                               |
| -------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Plan and cameras     | `rig plan-check`, `rig record`                                                               | plan.json, USB cameras              | `plan.json`, `recordings.json`, `session_manifest.json`, MJPEG                  | soak #9613, mode sweeps; one camera per USB 2.0 root port              |
| Import files         | `rig import`                                                                                 | video files                         | same bundle documents                                                           | `test_importer.py`                                                     |
| Intrinsics           | `rig calibrate-intrinsics`, `rig board`                                                      | chessboard / ChArUco recordings     | `intrinsics.json`                                                               | synthetic lens tests within 2 % focal; **no real board recording yet** |
| Detect               | `rig ingest`, `rig compare`                                                                  | recordings                          | `observations*/<view>.json`, comparison                                         | MediaPipe + OpenPose BODY_25 on take 2                                 |
| Reliability          | `rig reliability`                                                                            | observation sets, clean report      | `reliability.json`, `.md`                                                       | take 2 grades                                                          |
| Reconstruct          | `rig reconstruct --anchor ...`                                                               | observations, intrinsics or cameras | `reconstruct/` joints, cameras, swing summary                                   | synthetic: bones < 3 %, RMS < 2 px; **no three-view real take yet**    |
| Fit model            | `rig fit-model --model NAME [--fit-lengths] [--variant V] [--from-views a --cameras-from W]` | `reconstruct/joints_3d_m.npy`       | `model/[NAME/]joint_angles.json`, `fit_report.json`, `landmarks_fit.npy`        | synthetic: 6 mm RMS, jump rejected, scapular elevation within 4°       |
| Compare models       | `rig compare-models`                                                                         | reconstruction                      | `model/comparison.json`, `.md`, `model/<name>/`                                 | golfer vs pendulums on synthetic                                       |
| Kinetics             | `rig kinetics --model NAME --body-mass KG`                                                   | `model/joint_angles.json`           | `model/kinetics.json`                                                           | gravity statics exact; replay drift < 0.05 rad on a swung pendulum     |
| 2-D analysis (1 cam) | `rig analyze`                                                                                | observations                        | `analysis_2d/<view>.json`                                                       | take 2: address 981 → top 1061 → peak 1075                             |
| Clips and comparison | `rig clip`, `rig compare-takes`                                                              | recordings + observations + events  | mp4 + json                                                                      | take 2 clip                                                            |
| Export               | `rig export [--variant V]`                                                                   | reconstruct, model                  | `reconstruction.trc`, `reconstruction_export.json`, `joint_angles_simscape.csv` | TRC round-trips through `TRCAdapter`; Simscape axes validated (#9714)  |
| Overlay              | `rig overlay --view v --variant a --variant b`                                               | recordings, variants                | `<out>.mp4` + `.json` sidecar (reprojection RMS per track, held-out flag)       | legend + projected pixels asserted on the MJPEG fixture                |
| Compare variants     | `rig compare-variants [--reference V]`                                                       | variants                            | `variants/comparison.{json,md}`                                                 | `evidence/camera_subsets.md` (synthetic 3/2/1 cameras)                 |
| Annotate             | tile _Annotate / edit points_, `rig annotations-to-observations [--merge-with SET]`          | recordings, observation sets        | `annotations/<view>.json`, `observations_manual/`, `observations_<set>_edited/` | `evidence/sparse_annotations.md`; edit-mode dialog test                |
| Lineage              | `rig lineage --path FILE`                                                                    | any pipeline JSON                   | stdout                                                                          | `test_provenance.py`; every writer stamped                             |

The Capture Rig tile (`python3 -m src.tools.capture_rig`) runs every command
as a child process and shows the guided workflow; the user guide is generated
from the same step model (`scripts/generate_mocap_user_guide.py`).

## Models

Variants (#9793): every command above takes `--variant NAME`; outputs go to
`variants/NAME/` and `variants/index.json` lists them (`src/motion_capture/variants.py`).
Every JSON carries `provenance` (`src/motion_capture/provenance.py`).

`reconstruct/model/registry.py`: `golfer` (default, `golfer-scapula/2.0`:
body frames are the Simscape sensor frames, scapula struts after the MATLAB
3-D golf model), `double_pendulum`, `triple_pendulum`. Add a model by
registering a `ModelSpec` + `LandmarkMap`; nothing else changes. Design and
evidence: `articulated_model.md`.

## Open Validations (Owner-Side or Next PR)

1. **Real data.** Board recording per camera, tape-measured shank/forearm/
   upper arm/thigh, a three-view take. Everything downstream of ingest has
   only synthetic and single-camera evidence.
2. **Simscape axis conventions: done** (`evidence/simscape_axes.md`). Every
   logged joint identified to ≤ 2.7e-4 rad and encoded in
   `golfer-scapula/2.0`; replaying the logs through the Python model lands
   on the sensors to ≤ 1.7e-3 rad. Still open: the wrist universal (not
   logged) and the Simscape world orientation (affects only the root).
   Owner-side check: load one exported `joint_angles_simscape.csv` row as
   start positions in `GolfSwing3D_Kinetic` and compare the pose.
3. **Kinetics fidelity.** Point-mass segments, no rod inertia, linearised
   replay. A free forward integration in an engine (MuJoCo/Drake via the
   existing URDF builders) is the next step.
4. **Real-take camera-subset study.** The 3/2/1-camera comparison and the
   sparse-annotation accuracy are synthetic (`evidence/camera_subsets.md`,
   `evidence/sparse_annotations.md`); repeat both on a real three-view take.
5. **Pendulum club link.** The detectors do not observe the club; the
   pendulums end at the hands. A clubhead detector (or the ball line) would
   add the last link.

## Where Things Are Tracked

- #9619 capture → reconstruction chain; #9658 guided workflow; #9677 video
  tools and ecosystem decisions (ADR-0049); #9709 articulated model, registry,
  kinetics; #9648 more 2-D detectors; #9683 monocular 3-D (evaluated, not
  adopted).
- Runbook: `camera_rig_runbook.md`. Design: `self_calibrating_pipeline.md`,
  `articulated_model.md`. Guide: `user_guide.md` (generated).

## Host Notes

- One ELP camera per USB 2.0 root port; 30 ft powered cables are fine.
- Windows dev host: run GUI tests with the QApplication held at module level;
  the WSL lane (`scripts/dev/wsl_qt_tests.sh`) gives CI parity.
- Worktrees: remove the `vendor/ud-tools` junction before `git worktree remove`.
