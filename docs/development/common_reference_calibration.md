# Common-Reference Calibration Implementation

## Scope and Delivery State

Epic #9897; children #9898 and #9900. Branch
`feat/9898-common-reference-sessions` in the isolated
`UpstreamDrift-common-calibration` worktree. PR #9946 checkpoint 457521222 passed every normal push hook and integrates main276998030 (#9945/#9947 analysis delivery). It is reachable through Calibration in this worktree,
but the running test application still uses the earlier merged wizard source.

Tools #5141 published the combined provider as
`e83bd2e4a7a29a2dcd8145ef2d1efa07123324f0`, with both Linux aggregates and
required quality checks passing. The context owner retains final consumer
Gitlink/Cargo/pip/catalog alignment and installed-launcher qualification.
The local vendor checkout remains 0a561daff for development; do not commit it
alone because it lacks the launcher correction. Integrate the owner's published
consumer delivery before final PR qualification. Historical failures remain failures.

## Implemented Workflow

- CI repair: the developer log now links to this detailed checkpoint within its
  50 KiB budget. Projection pixels are explicitly floating point before camera
  skew is applied. Club and measured-reference editors share native Save/Cancel
  wiring. Seventeen projection/editor/calibration tests pass; the calculation
  and editor validation behavior are unchanged.

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
- Final provider pin, packaged qualification, PR/CI/protected merge and physical-camera acceptance remain pending. Current main and generated metadata are integrated; named reuse, visual frame selection and precision panning are implemented and locally qualified.

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
