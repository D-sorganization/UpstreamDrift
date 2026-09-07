# Self-Calibrating Markerless Pipeline

Version: 1.0.0

Issues: #9619 (epic), children #9621 to #9630; builds on #9599 and #9422

This is the path from the three-camera rig to a 3-D golf swing that respects
the golfer's bone lengths and the dynamics of the motion, without a precise
camera setup. It records what exists, what is missing, the algorithmic
choices, and the acceptance gates, so each child issue can land independently.

## What MediaPipe Does and Does Not Do

MediaPipe Pose is a per-image detector. Given one frame it returns 33 2-D
landmarks with a visibility score and a model-conditioned depth guess. It has
no notion of a second camera, of a previous frame, or of the subject's limb
lengths, and it never reports why a landmark is wrong. Take 2 on 2026-09-06
(1280x720 @ 120 fps, down-the-line view) shows the consequence: torso, hips and
legs at 0.7 to 0.9 visibility through the swing, hands and wrists below 0.4 for
most frames, head lost at the finish, and a per-frame skeleton that is
plausible but unconstrained. Fusing views, enforcing bone lengths and
rejecting outliers are separate stages that sit after every detector.

## Inventory (2026-09-06)

| Stage                                                       | Status  | Where                                                                                           |
| ----------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------- |
| Capture, recording, time sync, proxies                      | exists  | `src/motion_capture/rig`                                                                        |
| Per-view 2-D detection (MediaPipe; OpenPose via OpenCV DNN) | exists  | `rig/ingest.py`, `pose_estimation/{mediapipe_estimator,openpose_dnn_estimator}.py`              |
| Calibration records (K, distortion, `T_world_from_camera`)  | exists  | `pose_estimation/observations.py`                                                               |
| Intrinsic and extrinsic solvers                             | missing | ADR-0041 assigns reference algorithms to Tools `sidekick.lab.mocap` (contracts only in the pin) |
| Multi-view triangulation with residuals                     | missing |                                                                                                 |
| 3-D points to rigid skeleton (fixed segment lengths)        | exists  | `motion_pipeline/ik/geometric_backend.py`, `ik/pinocchio_backend.py`                            |
| Subject segment lengths from data                           | exists  | `motion_pipeline/scaling/anthropometric.py`                                                     |
| Per-channel temporal filters                                | exists  | `motion_pipeline/preprocessing/filter.py`                                                       |
| Skeleton-constrained temporal estimator                     | missing |                                                                                                 |
| Ball detection                                              | missing |                                                                                                 |
| Rig observations into `motion_pipeline`                     | missing | orchestrator starts from a 3-D file                                                             |

## Design

### One Unknown Vector, Many Takes

Everything the system "learns" is a parameter of one least-squares problem:

- per camera: rotation, translation (intrinsics fixed from the one-time board
  calibration, #9622);
- per subject: segment lengths with left/right symmetry (#9624);
- per frame: joint angles of the rigid skeleton (root pose plus rotations);
- per take: the ball position (#9621) and a small camera correction, because
  the cameras move a little between takes.

The residuals are reprojection errors of the skeleton's joints into every
view, weighted by detector confidence, plus priors: gravity is up, the feet and
the ball lie on the ground plane, segment lengths follow anthropometric
priors, and consecutive frames obey velocity and acceleration bounds. Bone
lengths are never optimised per frame; the skeleton is rigid by construction
(the same `tpose_offset` model the geometric IK backend already uses).

Every take of the same subject re-uses the previous solution as its starting
point and adds its frames to the pool that constrains the shared parameters,
so camera placement and bone lengths sharpen over time instead of being
re-estimated from scratch.

### Outliers Are Rejected, Not Averaged

The cost uses robust kernels (Huber for the first pass, Geman-McClure once the
solution is near) so a wrong detection cannot pull the fit, and three explicit
gates reject points: a joint that disagrees with the other views beyond its
residual threshold for that frame; a left/right swap detected by consistency
with the neighbouring frames and the other views; and, after convergence, any
observation whose reprojection residual exceeds the acceptance threshold. A
rejected observation is written to the bundle with its reason (view, frame,
joint, residual). Nothing is imputed; a joint with no surviving observation in
a frame is estimated from the dynamics prior and reported with its uncertainty
(#9625).

### Ball as the Shared Anchor

`reconstruct/ball.py` (#9621) finds bright, pale, near-circular blobs of a
plausible radius and ranks them; it returns candidates with scores, an
optional operator hint steers the choice, and `ball_at_rest` requires a
stable run of frames before it reports a position. On the real down-the-line
frame the spare balls on the mat are found once the value gate matches the
bay lighting; the addressed ball is hidden under the club head at that
instant, so the rest phase before takeaway is where the anchor is read.

### Per-View Cleaning

`reconstruct/clean.py` applies the smoother joint by joint to a view's
observation file: rejected detections keep their coordinates but drop to
confidence 0, the fitted track and its uncertainty go to separate `fit_px`
fields, and every rejection is listed with the residual that condemned it.
On the synthetic harness (1 px noise, 3 % occlusion, 3 % gross outliers of
up to 120 px) it flags injected outliers with recall 0.94-0.96 and precision
0.92-0.96 across seeds; the misses are gross points in the first or last
frames and offsets under about 10 px, which the multi-view gate (C5) is
expected to catch.

### Dynamics Prior

After the geometric fit, joint trajectories are re-estimated in joint space
with a temporal factor graph: measurement factors from the fit, smoothness
factors between consecutive frames, and per-joint velocity and acceleration
bounds tuned on golf swings. This removes single-frame spikes without
flattening the peak of the downswing; the acceptance test injects one-frame
spikes into a synthetic swing and requires club-head speed within 2 % of the
truth (#9626).

The first implementation is `reconstruct/temporal.py` (#9626): a penalised
least-squares smoother in physical units — measurement noise estimated
robustly from second differences, an acceleration prior `acceleration_sigma`
in units per second squared, Huber reweighting, a residual gate that rejects
and _lists_ each outlier, explicit velocity/acceleration bound checks that
report every violation instead of clipping, and per-frame posterior
uncertainty. It runs on any `(T, D)` series (pixels, metres, radians), so it
serves both per-view cleaning and the joint-space stage after IK.

### Initialisation Without a Calibration Object

The first take of a new camera placement has no extrinsics. Confident 2-D
joints on strobe-aligned frames across two views give correspondences; with
known intrinsics the essential matrix (RANSAC) yields relative rotation and a
direction of translation. Scale comes from the subject's prior segment lengths
and the ball; orientation from gravity (the subject's vertical at address) and
the ground plane. Bundle adjustment then refines everything jointly (#9623).
Later takes skip this step and start from the stored placement.

### Evaluation Before Trust

A synthetic harness renders a known skeleton on a swing-like trajectory
through known cameras with noise, occlusion and gross outliers, runs the full
chain, and asserts camera pose, bone-length, joint-position and outlier-flag
bounds. It is the CI gate for every algorithm above; real takes add a
take-over-take consistency metric (#9629). Thresholds live in
[`markerless_mocap_acceptance.md`](markerless_mocap_acceptance.md).

## Synthetic Evaluation Harness

`src/motion_capture/reconstruct` (#9629) makes every later algorithm testable
before it exists:

```bash
python3 -m motion_capture.reconstruct synth --out sessions/synthetic --frames 120   --noise-px 1.0 --occlusion 0.05 --outliers 0.02 --seed 0
```

writes `observations/<view>.json` in the exact schema `rig ingest` produces
for three cameras around a rigid 15-joint skeleton on a swing-like motion,
plus `truth.json`: 3-D joints per frame, the camera records
(`T_world_from_camera`, K), the bone lengths, and the list of every
observation that was occluded or turned into a gross outlier (moved by
~120 px while keeping a high confidence — the case a robust fitter must
catch). `PinholeCamera` bridges to `pose_estimation.observations`
records in both directions, so a fitter tested here runs unchanged on a
calibrated real rig. `metrics.py` scores camera pose (degrees, metres),
relative bone-length error, joint-position error with missing counts, and
outlier-flag precision/recall; the thresholds live in the acceptance document.

## Detector Comparison

`python3 -m motion_capture.rig compare --session S --estimators mediapipe,openpose_dnn`
ingests one bundle with each detector and writes per-view coverage, mean
confidence, frame-to-frame jitter (normalised by subject height) and
cross-detector agreement. `openpose_dnn` runs the published BODY_25 Caffe
network through OpenCV on the CPU, because `pyopenpose` cannot be built for
this host's GPU; fetch its files with
`python3 -m src.shared.python.pose_estimation.openpose_models`. Neither
metric is accuracy; without 3-D ground truth the honest statements are
coverage, self-consistency and agreement (#9628).

## Ownership

ADR-0041 gives Tools authority over calibration and reconstruction records
and their reference algorithms. The generic geometry (intrinsics solve,
essential-matrix RANSAC, triangulation, bundle-adjustment core) belongs there;
the golf-specific subject and scene model, the dynamics prior and the
take-over-take learning are UpstreamDrift orchestration. Whether that split
holds, or the ADR is amended so this repository may host the fitter while the
vendored pin lacks algorithms, is decided in #9630 before #9623 merges.

## Order of Work

1. #9630 ownership decision and #9629 synthetic harness (the gate comes first).
2. #9622 intrinsics, #9621 ball, #9628 detector comparison (parallel).
3. #9623 extrinsic self-calibration, #9624 skeleton learning, #9625 robust cost.
4. #9626 dynamics prior, #9627 wiring into `motion_pipeline`.
