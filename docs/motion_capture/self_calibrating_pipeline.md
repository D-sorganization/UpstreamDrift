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

### The Joint Fit, as Built

`reconstruct/geometry.py` triangulates one point from any number of
calibrated views (weighted DLT), reports the reprojection residual per view
and a first-order covariance, and assigns blame to a wrong view by leaving
each view out in turn — a wrong view contaminates the joint solution, so its
own residual is not reliably the largest. Two views cannot assign blame: an
error along the epipolar line is absorbed by depth, which is the reason the
acceptance program requires a third useful view.

`reconstruct/bundle.py` is the joint least squares itself: camera rotation
and position for every camera but the first (the gauge), every joint in
every frame, and one length per segment shared by all frames, with rigid
segments, an anthropometric prior, left/right symmetry and **one measured
length on the subject as the scale anchor**. With one camera fixed, a global
scale about its centre leaves every reprojection unchanged, so scale must
come from the subject; a soft anchor gets traded away by surviving outliers,
which is why the anchor is treated as known. Robustness is iteratively
reweighted least squares on the reprojection and segment terms only (a
library-wide robust loss would also flatten the anchor). After convergence
each offending point is judged by the triangulation rule, points left with a
single view are declared unobservable and placed by the segments alone, and
the cameras and lengths are refitted without the rejected observations. The
Jacobian is analytic and sparse; 40 frames of three views fit in about two
seconds.

On the synthetic harness (1 px noise, 2 % occlusion, cameras started 3
degrees and 15 cm off, a subject 6 % taller than the prior): camera rotation
within 0.5 degrees and position within 3 cm, every bone length within 2 %,
joints within about 1.4 cm median. With 3 % gross outliers: injected outliers
flagged with recall 0.9 and precision 0.85 or better, the same camera and
length accuracy, and the handful of points every view got wrong reported as
unobservable rather than invented.

```bash
python3 -m motion_capture.reconstruct synth --out sessions/synthetic --frames 120
python3 -m motion_capture.reconstruct fit --bundle sessions/synthetic --anchor neck=0.53
```

`fit` reads `observations/<view>.json` (synthetic or from `rig ingest`),
starts from the cameras in `truth.json` or from `--cameras records.json`
(the previous take's solution), and writes `reconstruction.json`: refined
camera records, learned bone lengths, per-view residual statistics, every
rejected observation with its residual, the count of unobservable points,
and — when truth is present — the metrics. `joints_3d_m.npy` holds the
fitted trajectory for the IK stage.

On a real take the chain is one command:

```bash
python3 -m motion_capture.rig reconstruct --session sessions/<take>   --cameras sessions/<previous-take>/reconstruct/reconstruction.json --anchor neck=0.53
```

It maps the ingested views onto the reconstruct skeleton, cleans each view
with the dynamics prior (`reconstruct/clean_report.json` lists every
rejection), and runs the joint fit from the previous take's cameras — which
is how placement is learned across takes — writing
`reconstruct/reconstruction.json`, `joints_3d_m.npy` and a summary. The first
take of a new placement starts from a rough camera file; later takes start
from the last solution.

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

### Intrinsics Once per Camera, as Built

`reconstruct/intrinsics.py` and `rig calibrate-intrinsics --session S --board
9x6 --square 0.025` find a printed, asymmetric chessboard in the sampled
frames of each recording of a bundle and run Zhang calibration into an
`intrinsics.json` keyed by view: `K`, distortion, reprojection RMS, frames
used and frames where the board was missing. Fewer than eight usable frames
refuses to calibrate and an RMS above 1 px is reported as below standard,
never silently accepted. The synthetic test renders a board through a known
lens (including distortion) and recovers focal lengths within 2 %, the
principal point within 15 px and the first distortion term within 0.05.
The file is the `--intrinsics` input of a first-take reconstruction.

### Initialisation Without a Calibration Object, as Built

`reconstruct/initialize.py` needs only each camera's intrinsics and the
golfer: confident joints seen by two cameras in the same frame are
correspondences, the essential matrix under RANSAC gives each camera's
rotation and the direction to it from the first camera, the anchor segment
triangulated with the unit baseline sets the scale, and the hip-to-neck
direction at address with the first mid-hip as origin sets the world frame
(yaw stays free until the ball line). A pair below 40 inliers is refused,
not guessed. On the synthetic harness the start is coarse (optical-axis
angles within about 10 degrees, baselines within 30 %) and the joint fit
from that start reaches the same accuracy as from a good previous take.
`rig reconstruct --intrinsics records.json` uses it for the first take of a
new setup; later takes pass `--cameras` with the previous solution.

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

## What the Golfer Gets Back

`reconstruct/analytics.py` turns the fitted joints into the numbers a coach
reads first, each with the frame it happened in: pelvis and shoulder turn
about the vertical relative to address (unwrapped, so a full backswing does
not fold at 180 degrees), their difference (the X-factor), hand speed from
the robust smoother with its uncertainty (the club is not tracked yet, so
hands are the proxy), the swing events the speed profile implies (address,
top, peak speed, finish) and the tempo ratio. `rig reconstruct` writes them
as `reconstruct/swing_summary.json`. On the synthetic swing the turn angles
match the motion's known rotations within half a degree, and a one-frame
30 cm jump of the wrists changes the peak hand speed by less than 5 %
because the smoother rejects it and says so.

The existing `shared.python.analysis` package (phases, tempo, X-factor
stretch, reports) consumes joint-angle and club-speed series from
simulations; these series are the mocap-side input to it, not a second
implementation of it.

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
