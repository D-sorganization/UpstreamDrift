# Common-Reference Calibration Implementation

## Scope and Delivery State

Epic #9897; children #9898 and #9900. PR #9946 merged as `126158943` before the
installed-package repairs below. Issue #9949 delivers those repairs from branch
`fix/9949-installed-capture` in the isolated `UpstreamDrift-common-calibration`
worktree; merge `e623b3c69` integrates current main without application changes.
The calibration delivery integrates published
main `08c8529ef` through merge `732553479`, including the combined Tools provider
`e83bd2e4a7a29a2dcd8145ef2d1efa07123324f0`. The checked-out provider, Gitlink,
Python requirement and Rust dependency agree. Twenty-eight pin/context/atlas/
governance tests pass. All 660 capture/reconstruction tests pass against this provider. The package
probe found the installed-worker issue described below.

The context owner qualified a frontend-inclusive installed wheel at source
`6dab98fce`: launcher, dependency and 31 calibration/reference checks passed.
The impact owner independently verified wheel/test-report hashes and zero
failures/skips. This is provider/installation evidence, not qualification of the
full current #9946 wheel or physical camera accuracy. Historical failures remain
failures. Live test applications still use their frozen earlier checkouts.

## Implemented Workflow

- First-capture library registration (#9950): successful completion now creates
  the catalog entry even if Library has never been opened. The capture header
  refreshes after registration to show the persisted identity immediately.
  Both missing registration and stale identity were reproduced by regression
  tests; 29 library/capture/wizard/journey checks now pass. The first installed
  registration probe passes; the final identity-refresh wheel remains pending.

- Installed storage (#9949/#9950): new recordings resolve to unique destinations
  in the player's selected library rather than the application installation.
  Empty explicit destinations remain usable; existing takes select a fresh
  destination without modifying their media. Fourteen GUI/library/wizard tests
  and configured mypy for both changed modules pass. The rebuilt installed
  window/wizard selects the configured player-library destination outside the
  application, creates no capture prematurely, and passes worker catalog and
  pip checks. Frontend-inclusive wheel SHA256:
  `55aacd817e75004f9355fd46aa9903e2de3bed47f65d3dbad239e799214f2310`.
  Prior numerical wheel evidence below retains its own scope.
  The broader post-storage run passed665 tests and failed one obsolete preview
  assertion expecting a checkout `sessions` directory. Its replacement verifies
  the configured library destination; all eight preview tests then pass.

- Installed-worker qualification: the normal frontend-inclusive wheel built from
  c432788c3 failed its actual installed catalog request because it required a
  repository vendor directory. Resolution now accepts the provider recorded in
  the owning application distribution, while rejecting unrelated distributions
  and incomplete source checkouts. Eight source-worker/ownership checks pass. Metadata paths are normalized for
  the standard-library path protocol. The corrected installed worker returns all
  four reference targets; 29 installed persistence/solve/reuse tests pass.
  The subsequent GUI probe exposed a second packaging defect: the unanchored
  motion-matching scratch ignore removed the tracked reference-loader package.
  A source-package exception fixes that omission; the artifact regression fails
  against the old wheel and passes against the rebuilt wheel. The final installed
  capture window and edit wizard open with `No Capture Selected`; all 29 installed
  calibration checks pass again. Offscreen visual inspection uses the system
  Segoe UI font because the offscreen platform has no automatic font discovery.
  Frontend-inclusive wheel source: 2d84f2fe0 plus this ignore exception (SELF).
  SHA256: `12705ae63e13131d52eef90712f18421e6a09221ee148172ca79febf7367e637`.
  Logs, JUnit, module origins and screenshot are in the task's temporary
  `capture-9946-installed-qualification` directory. These checks establish package
  startup and synthetic calibration behavior, not physical-camera accuracy.
  Original failing wheel SHA256: `5be3ecb354963958e44016e6de34ecd1677df3437e600f973b54a19875710fca`.

- Comparison journey qualification: thirteen wizard-evidence tests pass, including
  a persisted, capture-bound expert-video registration that completes the entire
  comparison route despite an unrelated damaged comparison file. The damaged
  file remains untouched. These tests do not claim a calibrated 3-D viewpoint.

- CI repair: the developer log now links to this detailed checkpoint within its
  50 KiB budget. Projection pixels are explicitly floating point before camera
  skew is applied. Club and measured-reference editors share native Save/Cancel
  wiring. Seventeen projection/editor/calibration tests pass; the calculation
  and editor validation behavior are unchanged. The registry retains the primary
  inventory blocker first, as required by its existing contract; all calibration
  and peer blockers remain present and scientific release remains blocked.

- Missing-output recovery (#9909) records a removed recording or pose-output
  path in the input revision instead of failing the whole wizard inspection.
  Thirty-one planner/wizard tests pass, including removed pose output with editing
  still available. A fresh Capture Rig window also resumes a saved editing route
  without changing its edits or progress file; this is an application-instance
  test, not a physical camera journey.

- Calibration type qualification preserves uint8 preview images after downscaling
  and accepts Qt's optional close-event argument. Eleven decoder/frame-selection
  and layout-reuse tests pass, including a 2560×1440 source resized to 1280×720.
  The broader import-following check previously reported eleven errors in eight
  modules; unrelated imported-module findings remain separate from this repair.

- Saved comparison recovery (#9909): malformed JSON or unreadable text now
  produces a file-specific review action without aborting capture status checks.
  Existing valid alignments remain eligible and original files are preserved.
  Both corruption regressions and the focused wizard suite pass.

- Wizard journey qualification #9909 found that a missing reviewed calibration
  aborted unrelated status checks. Editing-only routes now skip calibration
  inspection; dependent routes retain a local blocked calibration status and a
  Review/Repeat recovery action.19 evidence/planner/native-wizard tests pass,
  including switching from reconstruction to editing and finishing after the
  reviewed file disappears. Configured mypy and architecture checks pass.
  Readiness now retains prerequisite IDs separately from user-facing titles;
  blocked pages offer named Go to buttons inside their scrollable instructions.
  Links route only to declared prerequisite pages, disable while busy and hide
  after satisfaction.28 planner/wizard tests, configured mypy and a native660×560
  screenshot/click check passed (`capture-wizard-links-native-tn3p0v9p` in TEMP).

- Precision marking adds Pan Image, middle-button drag and Fit Image. Panning
  uses the existing source-pixel transform, clamps to image edges and never emits
  a point mark. The shared ImageCanvas caches the original pixmap and lets Qt clip
  its paint transform, avoiding allocation of a zoom-expanded bitmap.25 focused
  pan/point/selector/coaching/annotation tests and12 crop/comparison consumer checks
  passed on Python3.12; configured mypy passed. Native640×580 fit/pan views were
  inspected in the temporary `capture-point-pan-native-k4m5afs8` directory. The
  final calibration-dialog/atlas group16 passed and all normal push hooks passed.

- Original-video frame selection now uses the existing VideoReader and ImageCanvas,
  with timeline, Play/Pause, source clock, frame stepping, numeric selection and
  F11/double-click full screen. One background decoder retains only the latest
  pending seek. Closing schedules decoder cleanup after an active read without
  blocking Qt. Previewing does not archive evidence, and an undisplayed seek
  cannot be accepted. The selected original frame enters the existing worker
  archive/point-edit path; swing crop/trim metadata remains irrelevant to these pixels.
  Fourteen Python3.12 selector/dialog/point-editor checks and the configured mypy
  hook passed, including real-video selection through saved observations. Native
  synthetic video was visually reviewed at640×560 and full screen, then restored
  and accepted at the same source frame. Artifacts are in the local temporary
  `capture-frame-selector-native-jmapf0rl` directory; no camera was opened.

- Capture-owned immutable reference sessions reuse canonical Tools US Letter,
  A4, yardstick and metre-stick geometry. Measured dimensions create distinct
  targets. Camera selections reuse existing lens profiles, declared optical
  settings and physical identity. Moving a camera or changing declared optics
  disables further marking/solving until a new session is created.
- Actual original video frames are decoded outside the UI process and retained
  as lossless PNGs with bounded metadata and SHA-256. Existing trim/crop selections
  never change calibration coordinates. Repeated observations retain physical
  point IDs, placement names, notes, inclusion and held-out status.
- Standard Qt controls, ImageCanvas, QUndoStack and RigProcessRunner provide
  point editing, keyboard entry, undo/redo, worker progress, cancellation and
  actionable errors. A separate loader verifies/decompresses archived frames
  without blocking Qt. Saved capture profiles are restored without retaining
  an unrelated previous combo-box selection.
- Canonical Tools fixed-intrinsics estimation consumes reviewed lens settings,
  explicit world-anchor translation/orientation and identified observations.
  The adapter converts camera-from-world into existing world-from-camera records
  once, preserving all lens coefficients. Camera/placement residuals and Tools
  limitations are shown before explicit operator acceptance.
- Saved estimates can be reopened against their matching observation revision.
  Acceptance rechecks frames, profiles, anchor, revision hash and the exact result
  digest shown to the player. Reviewed copies do not overwrite numerical results.
  Source evidence hashes provider/adapter files and records numerical runtime
  versions. It is source provenance, not a full SBOM or physical approval.
- Reconstruction corrects original distorted detections into ideal pixels before
  layout averaging and fitting, with a calibration signature preventing repeated
  correction. Camera records retain distortion for downstream video overlays.
  OpenCV supplies the numerical operation; no new lens or camera solver is copied.
- Native Help and the generated operator guide share `guidance.py`. Regenerate
  with `python3 scripts/generate_reference_calibration_guide.py`; `--check` detects
  drift. The calibration inventory remains explicitly blocked under #9900 in the
  engineering manual registry.

## Provider Boundary and Responsiveness

UpstreamDrift owns its local Sidekick cluster. The isolated worker launches with
`python -I`, resolves the selected Tools family first and leaves the parent's
namespace unchanged. Do not remove the fallback guards or add global aliases.
This checkout-based worker still requires exact packaged-runtime qualification.

Import tracing found calibration bookkeeping loading reconstruction, pose and
physics modules. IntrinsicsRecord and swing-export rendering imports are now
lazy. Separate validation samples changed from about2.85s to1.26s; catalog samples
varied0.39–0.97s under shared host load. These are diagnostic observations, not
controlled speedup benchmarks. Native OpenCV/SciPy already run the numerical work;
there is no evidence here supporting a Rust rewrite as the next improvement.

## Validation Checkpoint

The broad Capture Rig and reconstruction run completed successfully:594 tests
on Python3.13, including export, wizard, body-model and reference cases. Log:
`TEMP/capture-reference-integrated-regression.log`. Later history/Qt rerun7 passed,
and projection/Qt refinement25 passed. The four new distortion overlay cases
first failed against ideal projection and then passed after the shared adapter
fix. Actual OpenCV5.0.0 passed20 lens/reference-projection tests, including
5/8/12/14-term models. Atlas/parity49 passed. These are synthetic/software results.

Full Ruff and format pass (6941 files), mypy28 source modules plus20 projection
modules pass, and LoD reports no growth across3050 source files. The actual
working-tree architecture check covered41 changed Python paths and passed.
Normal commit and push hooks passed on d15013a2c; normal merge hooks passed on f4da52c65. Reuse checkpoint f6298fb0c passed normal commit hooks; the foreground verification refinement needs final push hooks. Generated help/atlas freshness and
manual governance pass; the calculation inventory remains release-blocked.

Native Python3.12 setup, placement, solve and Help pages were inspected at850×720
and640×560. Qt scroll areas and a two-column footer keep actions reachable;
Calibrate Again and Add Another Placement are explicit. Latest native screenshots:
`TEMP/capture-reference-dialog-visual-hn7cyubr`. Point editor was previously
reviewed at640×580. These are synthetic UI fixtures, not physical camera evidence.

The first broader Python3.12 environment lacked declared pydantic-settings and
pytest-qt dependencies; installed them in an isolated temporary environment.
That run ended581 passed,2 failed,12 errors, including a native Help crash and
a180s existing Simscape real-log calculation timeout. The numeric timeout was
reproduced with a stack in existing model/simscape.py and SciPy Rotation; no
calibration code was involved. Do not relabel this broad run as passing.

Subsequent scoped checks identified an excessively wide footer, now corrected,
and repeated PNG decompression during evidence checks. Saved-evidence validation
now rechecks capture identity, dimensions and original byte hashes without
re-decoding an already-archived frame; actual point inspection still decodes it.
Measured-target operations also defer player/media imports until actual frame
extraction. This removes work rather than increasing test timeouts. Production
Python3.12 scoped rerun passes51 tests in37.11s, including real video, revision
history, the isolated numerical worker, standard Help navigation and distortion
projection. Log: `TEMP/capture-reference-py312-lazy.log`. The fixture timeout
remains20s; no test limit was raised to obtain this pass.

## Cross-Capture Reuse Checkpoint

The new review uses the existing asynchronous worker and standard Qt controls.
It shows source swing/settings, requires two fresh confirmations and retains a
separate `capture-reference-assignment/1` with copied original revision, reviewed
result and archived frames. The camera reader rechecks source evidence before
processing, including outside the wizard; old manual camera files retain their
existing contract. No new camera solve or physical accuracy claim is implied.

Red tests:9 failed before reuse existed. The isolated Python3.12 provider suite
then passed27 checks, including portable evidence after moving the source,
tamper rejection and capture identity at the CLI. Seven dialog/profile tests and
18 reconstruction/lens/dialog boundary tests passed. Native actual-worker review
and assignment were exercised at640×560; screenshots and synthetic assignment
are in `TEMP/capture-reuse-native-ugzr1gl3`. The first small-window test exposed
an814px minimum width; shorter controls with wrapped guidance fixed it.
The repository configured isolated mypy hook passes8 changed source files. The final worker/Qt/atlas group passes19 tests (including27 nested provider checks). Final audit found full archived-frame reads in foreground confirmation. A red test reproduced the pause risk; full image checks now run in the worker, background wizard inspection and camera reader. Foreground confirmation verifies bounded documents only. The updated isolated suite passes28 checks in27.82s; every normal push hook passed on1e4984765. The broader global mypy invocation was stopped before producing a result and is not passing evidence.

Named library selection now uses capture names, review dates and scene labels,
including archived captures. `CaptureLibrary.catalog_entries` reads metadata
without walking storage or requiring source video availability; the existing
library manager still adds its storage diagnostics through `list`. The worker
lists bounded result metadata and verifies complete evidence only after selection.
Thirteen catalog/library/dialog tests,29 isolated provider checks and19 library/Qt/atlas integration tests pass. The configured isolated mypy hook passes five source files. Native library selection, full review and assignment were exercised at640×560 in `TEMP/capture-reuse-native-3l2a4f2e`, including an archived capture whose original video is offline. Earlier running test apps remain untouched.

## Limits and Remaining Acceptance

- Profiles and settings are operator declarations. Unreported camera/ring movement
  cannot be detected. Paper does not replace lens calibration. Ruler endpoints
  alone cannot establish camera poses. A wholly held-out placement cannot seed
  its own transform. Pixel residuals do not certify metric physical accuracy.
- Guided reuse now copies original reviewed results, revisions and archived frames into a new capture. It requires matching view names, identities and recorded sizes, plus fresh manual optics/scene confirmation. View remapping and subsets are deliberately unsupported; calibration/recording orchestration still needs end-to-end acceptance.
- Saved revisions assume a single dialog writer. Cancelling marking may retain
  unused archived frames. Do not delete evidence automatically. Referenced lens
  files must stay available; portable profile packaging remains to be qualified.
- #9900 manual calculation inventory, physical multi-camera/zoom evidence and
  packaged-runtime qualification are not approved by synthetic tests.
- The published provider is integrated; full current-wheel qualification, PR/CI/protected merge and physical-camera acceptance remain pending. Current main and generated metadata are integrated; named reuse, visual frame selection and precision panning are implemented and locally qualified.

## Concurrent Work and Live Applications

Session `capture-product-01a08427-common-calibration` owns the reference package,
calibration actions/profiles/evidence, scoped lazy swing-export imports and
reconstruction lens boundary. Presence renewed through08:29UTC, #9898 lease through
08:11UTC and #9900 lease through08:29UTC on2026-09-10. Impact sequencing message35dd44d5 was
acknowledged; there are no active scope conflicts in the last complete inbox.

Preserve live app PID54812 (`Capture Rig — main 56552f245`), older app61500 and
atlas2963. They serve frozen earlier checkouts, not this branch. The analysis
owner controls comparison/coaching/model work (#9932/#9942); do not edit it.
The context owner controls #9915/catalog/provider alignment. Continue only in this
isolated tree and merge normally after compatible provider qualification.
