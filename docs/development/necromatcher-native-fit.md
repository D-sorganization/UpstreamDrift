# Necromatcher Native Fitting Turnover

## Scope and Current State

Issue #11235 belongs to the integrated Necromatcher epic #11232. The historical
captures are stored and reviewable; image observations do not establish physical
time, scale, joint torques or an identifiable three-dimensional reconstruction.
Continue through actual native fitting, measured residuals, independent replay
and simulation/impact/analysis handoffs. No historical fit is accepted yet.

Owned branch: `feat/necromatcher-native-fit-11235`, based on workspace branch
`feat/necromatcher-workspace-11234`. Lease uses the current Codex session.
Draft [PR #11240](https://github.com/D-sorganization/UpstreamDrift/pull/11240)
records this prerequisite and remains open for the fitting work.

## Native Closure Repair

An actual call to the public `MatchingPlant.closure_residuals(q)` failed because
the MuJoCo implementation constructed an IK adapter with an empty attachment
mapping. That adapter rejects empty markers; its inherited position-residual
method is also unimplemented. Closure must be available independently of image
marker observations.

The native model now evaluates a detached world-space grip separation vector
after forward kinematics. The plant validates one finite coordinate per declared
DOF and delegates to this native boundary. Units remain metres; this is a position
residual, not a weld velocity, orientation residual or dynamics qualification.

Five regression cases failed before the repair. Neutral and bent-elbow poses
are compared with independently compiled MuJoCo MJCF and named native grip sites;
invalid size, nonfinite values and matrix-shaped coordinates are rejected. The
test also mutates a returned array and verifies native state remains unchanged.
The broader plant/model run passes 12 tests; one real-Drake test skips because
the installed Drake package is mocked rather than a native runtime.

The actual driver anthropometry specification loads 44 coordinates and model
SHA-256 `da181bae388e8e4ac1e5f905b11d5c80195bbe177bc685cbb6083a5815053ae8`.
Its neutral grip displacement is approximately `[-0.014840, -0.373188, -0.058067]`
metres. This nonzero displacement is reported as measured; no closure success
is inferred. This generic specimen is not a fitted Tiger or Hogan model.

## Reusable Boundaries and Next Steps

- Use public `motion_matching.pipeline.plant.get_plant` and its engine protocol.
  The old motion-pipeline MuJoCo IK backend is retired and raises explicitly.
- Reuse `estimation.project_pinhole` and `reprojection_residual_from_points`
  for image evidence. Preserve missingness and unknown visibility. Camera,
  metric scale, depth and physical clock assumptions must be recorded separately
  from detector observations.
- The existing Shadow Tracker control fitter consumes actual masks. Do not
  create masks from landmarks and call them observed segmentation; #11227 tracks
  the synthetic segmentation fallback.
- Do not use `ModelMatchHandoffCoordinator` fixed fit numbers as evidence.
  The old torque-matching implementation comparing a reference to itself does
  not establish independent forward replay.
- Fit and save an explicitly qualified research hypothesis for each player,
  with original capture/model/code hashes and measured projection/replay errors.
  Validate engine joint order and external resources before importing a native
  version; library storage alone does not perform this validation.
- Source `tiger-usga-range-capture-v1` has 210 observed frames from the official
  USGA 2000 broadcast. `hogan-practice-capture-v1` has 750 original practice
  frames. Both remain image-observation evidence with physical time unknown.
- MATLAB scientific acceptance requires R2025b. Manual governance currently
  reports `blocked-inventory-required`; this repair does not change that status.

## Validation Procedure

Initialize the exact pinned Tools dependency, set repository `src` on
`PYTHONPATH`, and run:

```powershell
python3 -m pytest tests/unit/motion_matching/pipeline/test_mujoco_closure.py tests/unit/motion_matching/pipeline/test_plant_protocol.py tests/unit/motion_matching/test_full_body_mujoco.py -q --no-cov
python3 scripts/ci/check_lod.py src --baseline scripts/ci/lod_baseline.txt
python3 -m scripts.check_design_manual_governance
```

Keep the full goal active until the player fits and downstream handoffs are
implemented and verified. Native closure repair is a prerequisite only.

## Image Fitting and Source Evaluation

The public `motion_matching.historical_fit` package now converts immutable
capture observations to pixels, fits native coordinates through the canonical
Hermite-spline MAP estimator, and preserves its coefficients for evaluation at
every original source PTS. Camera projection and weighted residuals reuse the
shared estimation facade. Missing landmarks have zero objective weight and null
reported errors; unknown visibility requires an explicit caller weight. Evidence
and fitted arrays are detached and read-only.

`initialize_camera_hypothesis` uses OpenCV SQPnP followed by LM refinement,
conditional on the caller's assumed pose, geometry and optics. It validates six
nondegenerate correspondences, proper rotations and positive camera depth.
This is an initial research hypothesis, not camera calibration. See the
[OpenCV Pose Computation Documentation](https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html).

Eight red-to-green tests cover native projection recovery, missing-point
exclusion, evidence clocks/identity/immutability, invalid camera geometry,
rotations and out-of-interval spline evaluation. The broader native run passes
20 tests with one real-Drake skip. Scoped Ruff and mypy pass; global LoD remains
within its existing baseline. Manual governance remains blocked for release.

## Measured Historical Research Runs

Both actual archives were fitted using 25 evenly selected source frames, four
Hermite knots, 17 free upper-body/root coordinates, and all 13 anatomical
associations (nose, shoulders, elbows, wrists, hips, knees and ankles). Geometry
is the generic driver specimen; lower-limb coordinates remain locked. Focal
length is assumed equal to image width, principal point at image center, and
distortion zero. Camera initialization is conditioned on the authored address
seed. Prior/smoothness/closure weights are 3/1/100; the budget is 30 evaluations.
These choices are experiment inputs, not qualified player properties.

| Source                                                | Initial Landmark RMS | Fitted Landmark RMS | Optimizer Status                |
| ----------------------------------------------------- | -------------------- | ------------------- | ------------------------------- |
| Hogan Practice, 110–134.967 Source Seconds            | 23.842 px            | 13.168 px           | Evaluation Limit, Not Converged |
| Official USGA Tiger Range, 15.015–21.989 Clip Seconds | 130.198 px           | 28.526 px           | Evaluation Limit, Not Converged |

RMS is confidence-weighted Euclidean distance per landmark, not per-axis RMS.
The preserved splines were subsequently evaluated at all 750 Hogan and 210 Tiger
source frames. Excluding the 25 fitting frames, measured held-out landmark RMS
is 13.683 px across 725 Hogan frames and 28.968 px across 185 Tiger frames.
The `*-dense-source-evaluation-v1.json` artifacts bind the spline-fit hash and
each original frame identity/hash, all 44-coordinate poses and per-point errors.
This is interpolation and held-out image evaluation, not independent dynamics
replay; no physical rates or torque acceptance are inferred.
Raw fit records and four-frame overlays are retained outside Git under the
Downloads `historical-capture/native-fit-research` directory. Each record binds
the source archive, native coordinate order/model bytes, original frame IDs and
hashes, camera assumptions, q samples and actual errors. The first prototype
records are named `*-sparse-fit.json`; the subsequent `*-spline-fit-v1.json`
records additionally preserve spline knots and coefficients. Do not overwrite
or promote these failed-to-converge candidates as accepted models.

Visual review confirms residual leg motion in Hogan that cannot be represented
by locked lower-limb coordinates. Tiger framing changes near the end of the
window; a fixed camera across that interval is an inadequate assumption.
Next passes need explicit reviewed swing/shot intervals, full-body motion,
camera-change handling, denser knots and held-out-frame residuals. Increasing
iteration count alone does not fix these model limitations. Native finite
differences currently repeat FK for each spline coefficient; reuse analytic
native derivatives and spline bases for efficient future-player fitting.

## Repeatable Research Procedure

1. Open the source through `CaptureReview` after the library verifies its hash.
   Select reviewed continuous shot/swing indices; retain source and frame hashes.
2. Call `read_capture_evidence` with attachment labels in declared order and an
   explicit unknown-visibility weight. Preserve the archive's original evidence.
3. Load original model-spec bytes with public `get_plant`; supply a documented
   seed, coordinate scales, free coordinates, attachments and camera assumptions.
   Use `ImageFitInputs` and `ImageFitConfig`, then `fit_image_trajectory`.
4. Store configuration, source/model/code hashes, convergence, per-point errors,
   fitted coefficients/knots and coordinate order as a new research version.
   `evaluate_source_times` evaluates inside that interval without extrapolation;
   measure held-out original-frame errors and inspect overlays.
5. Keep physical time unknown until source-speed evidence supports an explicit
   conversion. Source-clock velocities do not certify joint torques. Require
   bounded anatomical motion, grip/contact checks, sensitivity/identifiability
   analysis, actual native dynamics and independent forward replay before
   publishing simulation/impact/analysis handoffs.

Gemini Flash 3.8 was used through `agy` for supplied-source, tool-free audits.
The evidence-mutation finding was reproduced with a failing test and repaired.
Its suggested factor-of-two RMS change was rejected because this metric is
per-landmark distance; off-image coordinates remain valid detector observations.

## Full-Body Pass and Efficient Derivatives

The fitter now differentiates each native frame with the public central-difference
helper and applies exact canonical spline bases. Pose and source-speed prior
derivatives are analytic. Native pose evaluations therefore depend on the free
coordinate count rather than the number of spline coefficients. A new numerical
test compares every residual block with independent coefficient differences at
interior and endpoint samples, includes a missing landmark and grip closure,
and verifies fewer than half as many native marker evaluations. Nine fitting
tests pass. Explicit ndarray list annotations repair the NumPy shape inference
failure observed in the Python 3.11 CI lane; scoped mypy passes afterward.

The second actual research pass uses 60 frames, 12 knots and 27 free coordinates,
including hip, knee and ankle motion, while retaining the same camera assumptions
and prior/smoothness/closure weights. It evaluates every source frame afterward.

| Source                    | Fitted Landmark RMS | Held-Out Landmark RMS | Maximum Grip Separation | Declared ROM Violations                         |
| ------------------------- | ------------------- | --------------------- | ----------------------- | ----------------------------------------------- |
| Hogan Practice            | 7.210 px            | 7.694 px, 690 Frames  | 0.1258 m                | None in the Available Declared Ranges           |
| Official USGA Tiger Range | 11.636 px           | 11.666 px, 150 Frames | 0.2571 m                | LEInput: 4 Frames; REInput: 9; SpineInputY: 156 |

Both solves still reach the evaluation limit. Better pixel agreement is not
closure or anatomical acceptance. The available range inventory is incomplete
for lower limbs; absence of declared violations does not certify physiology.
The new `*-full-body-fit-v2.json` and `*-dense-full-body-evaluation-v2.json`
artifacts remain outside Git. The tracked second-pass receipt binds their bytes
and records actual rejection evidence. Execution-time implementation stamps were
not captured for these exploratory runs; the reviewed source digest is recorded
separately and must not be substituted for qualified run provenance.

Next: preserve launch-time source/model/capture fingerprints in the production
fit job, support warm starts, enforce anatomical ranges and grip closure, handle
camera changes and store source-bound fit versions in the player library. Joint
effort exports also need per-coordinate units: native root translations use N,
while rotational joint torques use N*m. The existing all-N*m authored-profile
schema must not be used as a 44-coordinate full-body effort qualification.

## Launcher Inventory Repair

The full CI lane reproduced eight Necromatcher logo, migration and companion
inventory failures. The native and web registry now reference a distinct,
Qt-validated SVG; tile/feature migration entries point to actual implementation
and acceptance tests. A governed pending screenshot record uses the canonical
pending reason; no capture is invented. Companion counts reflect the new
program, feature and surfaces. Generated baseline, atlas and agent-context views
are refreshed from their authorities, with the launcher boundary re-reviewed.
The separate shallow-checkout origin/main prerequisite remains a CI concern.

## Persistent Research Fit Versions

`NecromatcherLibrary.add_fit(fit_id, swing_id, source)` stores immutable JSON
under `necromatcher/kinematic-fit/1`. `load_fit(fit_id)` rechecks the fit bytes,
both parent hashes, same-swing ownership, model coordinate order and every exact
source frame identity. Sample matrices must be finite numeric values; declared
coordinate units are `rad` or `m`. This is storage validation, not native model
compilation or a physical calibration certificate. Fit exports use the existing
portable swing package and revalidate bindings before publication.

The real `hogan-full-body-research-fit-v2` and
`tiger-full-body-research-fit-v2` versions now retain 750 and 210 source-frame
samples in the configured persistent library. Their corresponding
`hogan-generic-native-model-v2` and `tiger-generic-native-model-v2` assets contain
actual compiled-export MuJoCo XML, with the original generic definition and its
digest retained in fit provenance. The producer derived coordinate units from
compiled MuJoCo hinge/slide types. These are generic geometry candidates.
[Storage Receipt](historical_capture/native-fit-library-receipt-v2.json) records
the immutable model, capture and fit hashes without local filesystem paths.

For Future Players:

1. Import a verified capture and immutable native model under the same swing.
2. Produce finite native `q` samples in the model's ordered coordinates.
3. Bind `model_id`, `model_hash`, `capture_id` and `capture_hash` to those exact
   library versions. Copy complete `CaptureReview.frame(index)["frame"]`
   identities, retaining rational PTS, rather than rounding source timestamps.
4. Include explicit provenance, camera/geometry assumptions, fit evidence and
   rejection reasons. Keep `qualification=monocular_research_hypothesis`,
   `physical_time_qualified=false` and `dynamics_replayed=false`.
5. Call `add_fit`, then reopen the library and call `load_fit`; export through
   `export_swing` for portable transfer. Never overwrite a prior version.

The web workspace distinguishes these versions as Kinematic Research Fits and
shows their qualification. Its existing import form accepts fit JSON through
`POST /necromatcher/swings/{swing_id}/fits`. Verified fit summaries and individual
native samples are available at `GET /necromatcher/fits/{fit_id}` and
`GET /necromatcher/fits/{fit_id}/frames/{source_frame_index}`. Missing sparse
samples return 404; stale model/capture bindings reject recall. The frame response
retains exact source identity, coordinate order/units and research qualification.
Seventeen fit-storage/API tests and fourteen web page/form tests pass.
Native trajectory overlays, fit-job submission,
resource-validated model import, physical-clock qualification, generalized
effort profiles and independently verified simulation/impact handoffs remain
required. Preserve the existing rejection evidence during that work.
