# Necromatcher Native Fitting Turnover

## Unit-Preserving Authored Controls

The new effort-profile/2 format binds exact model and research-fit hashes and
preserves ordered m/rad coordinates with N/N\*m generalized efforts. Import,
recall and export reject incompatible units and legacy torque profiles for known
translation coordinates. Bounded evaluation rejects extrapolation and overflow;
coefficients use immutable backing bytes. The existing native/web import route
accepts the format. See [Authored Effort Procedure](necromatcher-effort-profiles.md).
54 focused profile/library/fit/refit checks and two-module mypy pass.
Native unit qualification, actual historical driving controls and independent
replay remain open; source timing and scientific acceptance remain unqualified.

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
Fit-job UI/API submission,
resource-validated model import, physical-clock qualification, generalized
effort profiles and independently verified simulation/impact handoffs remain
required. Preserve the existing rejection evidence during that work.

## Verified Native and Web Fit Review

Saved Hogan and Tiger fits now project their bound native model onto their exact
source frames in both review interfaces. The shared projection validates the
capture/model hashes, reproduces the stored XML from its generic definition,
checks coordinate order and uses the stored camera hypothesis. The API exposes
`GET /necromatcher/fits/{fit_id}/frames/{source_frame_index}/projection`.
Selecting Review Fit in the web workspace or a fit asset in the native workspace
loads its bound capture. Slider changes discard stale projections.

A fresh Windows Qt application reproduced MuJoCo plugin DLL initialization error
1114 when projection ran in a thread or multiprocessing spawn child. The native
review now reuses a clean interpreter through the shared secure subprocess facade.
Its JSON-line worker imports no Qt application main; requests remain serialized,
timeouts terminate the owned process, and cleanup releases pipes and the worker.
A fresh Qt-parent regression checks actual native projection independently of
warm SDK import order. The UI responsiveness regression checks that an old frame
cannot paint after the slider advances.

Actual review loaded both stored versions: Hogan 750 frames and Tiger 210 frames,
with thirteen native attachment projections per reviewed frame. Local offscreen
PNG renders live beside the research artifacts, outside Git. The harness loads
an installed Windows font explicitly because the offscreen platform does not
resolve its font database normally; this is a harness accommodation. Browser
review independently displayed each source image, observed landmarks, native
projections and the unqualified camera/time labels.

The 25 fit-storage/API/projection/GUI/overlay checks pass. Seventeen web page/form/overlay
checks passed, with TypeScript and ESLint. These verify storage and review only;
the original nonconvergence, grip separation, ROM and timing rejection evidence
still governs the fits. Production fit jobs, bounded closed motion, generalized
effort profiles and qualified downstream replay remain open.

## Source-Stamped Research Refit Jobs

`start_native_refit(library, source_fit_id, new_fit_id, options, service)` now
runs actual native fitting through `MatchingJobService`. `NativeRefitOptions`
requires increasing source indices, explicit coordinate scales, knot count,
visibility prior, solver configuration and wall budget. The source fit must
contain every selected sample; the new fit identity must be unused.

A clean interpreter verifies the source-fit hash, compiled model binding,
camera/attachment assumptions and launch fingerprints before native execution.
Warm starts copy saved full native samples and reject changes to locked
coordinates. Source PTS remains the trajectory clock. Dense and held-out pixel
RMS use the original observation confidence; held-out frames exclude training
indices, and missing observations contribute no fabricated target.

Execution stamps record selected workspace, matching, estimation, numerical and
MuJoCo source-file hashes, their combined digest, the Git context, Python and
installed numerical/SDK versions, platform and Tools checkout commit. The
worker verifies the launch source/runtime digests and checks source freshness
again after solving. These are scoped reproduction identities, not approval of
all runtime dependencies or evidence of camera/physical-clock calibration.
The original exploratory v2 results remain unchanged.

`MatchingWorkOutcome` separates computational completion from qualification.
Necromatcher jobs return succeeded/rejected and retain their actual rejection
reasons in the canonical run manifest. Explicit accepted outcomes cannot bypass
job blockers. Existing legacy work-return semantics remain compatible and must
not be cited as scientific acceptance. Publication runs under the same completion
gate as cancellation: cancellation before commit prevents a new library version;
a late cancellation cannot relabel a committed version as interrupted.
Unpublished candidates and diagnostic manifests may remain in the owned run
folder for inspection. The worker is terminated on cancellation or wall timeout.

To Refit a Saved Player Version:

1. Open the configured library and call `load_fit` on the immutable prior version.
2. Choose reviewed continuous-shot source indices and a new version identity.
3. Build `NativeRefitOptions` with explicit native coordinate scales and priors.
4. Reuse one `MatchingJobService`, call `start_native_refit`, retain the returned
   handle and run folder, and use `request_cancel` or `join` as appropriate.
5. Inspect `request.json`, `run_manifest.json` and rejection evidence; reopen the
   stored fit with `load_fit` and review its model projections before export.
6. Call `service.close()` after owned jobs finish. Preserve previous versions.

Seventy-three fitting/job/storage/API/native review checks passed at `5038b0748b`, including red-first
outcome, warm-start, cancellation, publication, input-contract and held-out-metric
regressions. Job submission through native/web controls, anatomical bounds,
closed motion, physical-clock qualification, mixed effort units and independently
verified downstream replay remain open. The full goal and #11235 remain active.

## Committed V4 Research Runs and Clean Worker Launches

[V4 Run Receipt](historical_capture/native-refit-job-receipt-v4.json) records jobs
executed from exact committed source `5038b0748bec967931ab76a07593aa1af14ca2d8`.
Both used 60 source samples, 12 knots and a ten-evaluation budget, preserving
all 750 Hogan and 210 Tiger frames. Hogan training RMS changed from 9.672 to
7.252 px and held-out RMS is 7.635 px. Tiger training RMS changed from 21.456
to 11.350 px and held-out RMS is 11.861 px. Initial errors evaluate the actual
warm-start spline, rather than the unsmoothed seed samples. Hogan held-out error
improved slightly versus v2; Tiger held-out error worsened. Both jobs exhausted
the evaluation budget. Maximum grip gaps are 0.123 m and 0.257 m, respectively.
Neither computation establishes anatomical, physical-clock or dynamics acceptance.

A subsequent launch regression removed inherited PYTHONPATH and reproduced
`ModuleNotFoundError: bunkershot3d` in the native projection child. Both owned
workers now use `core.repo_python_environment`, extracted from the existing
Capture Rig environment builder. It puts this checkout's src first, preserves
all other settings and does not alter SDK plugin or rendering options. Launch
module commands from the repository root. Current execution stamps include core
sources; the historical v4 receipt retains its original, narrower source scope.
The v4 jobs were not executed from this subsequent launch-fix revision.

Validation of the follow-up: 84 focused fitting, job, storage, API, native
review, shared environment and Capture Rig checks passed with inherited
PYTHONPATH removed. Ruff, file budgets, title case and manual governance pass;
manual release remains blocked pending the required calculation inventory.

Both v4 fits reopened through the verified library after the clean-launch fix.
Actual native projections at Hogan frames 0/375/749 and Tiger frames 0/105/209
returned all 13 stored attachments with PYTHONPATH removed from the parent.
Portable swing exports retained each exact v4 fit hash recorded in the receipt.
These are recall/export checks, not independent physical replay.

## Native and Web Refit Controls

Select a stored research fit, then use Refit Selected Version in the native
launcher or Research Refit in the web Models and Controls panel. Enter a new
version identity, reviewed source indices, knot count, one positive prior scale
per ordered native coordinate, weights and budgets. Recorded v4 settings can be
reused explicitly; older fits without recorded scales require operator input.
Source PTS remains the clock. These controls do not infer calibrated anatomy or
physical timing. Native submission runs in the canonical background adapter.

Both shells use NativeRefitSession over MatchingJobService; no second scheduler
is introduced. One active run per host prevents queued duplicates. Status and
cancellation use the owned public handle; terminal history reopens canonical
request/run manifests. A request is persisted before scheduling; execution_started
becomes true only at worker execution. A saved running record without an owned
handle remains unverified, since the absence of a control handle cannot prove
worker termination. Verify the prior host before starting replacement work.

Local-client API routes are POST fits/{fit_id}/refits, GET fits/{fit_id}/refit-plan,
GET refits/{run_id}, and POST refits/{run_id}/cancel under the Necromatcher prefix.
Body contracts reject extra fields, invalid samples/scales and unbounded HTTP
budgets. Host shutdown cancels owned jobs and drains the canonical executor.
The web run query preserves job recall across refresh and source-frame changes;
completion for an old source does not update a newly selected player. Reopened
terminal runs do not trigger repeated library refreshes. Computational success
and rejected qualification are displayed independently, with rejection reasons.

Validation includes 90 focused Python checks and five new controller/API/native
tests, actual rejected clean
worker HTTP submission, cancellation and saved-manifest recall, Qt responsiveness,
19 web form/page checks, generated request types, TypeScript and scoped ESLint.
Actual Hogan native-dialog and Tiger browser submissions now saved 750/210-frame v5 versions with source/runtime stamps and explicit rejected qualification. Canonical status reopened through the API after native host shutdown. The tracked V5 Interface Receipt records exact IDs, input options and source scope; the one-evaluation budget validates interfaces, not motion acceptance.

[V5 Interface Receipt](historical_capture/native-web-control-receipt-v5.json) records both runs. A red-first web URL-completion race regression prevents suppression of the new-version refresh when a newly submitted job finishes during URL persistence. Archived terminal jobs still reopen without repeated refreshes.
