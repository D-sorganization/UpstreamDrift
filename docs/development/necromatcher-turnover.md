# Necromatcher Turnover and Integration Procedure

## Active Objective

Owner priority: integrate historical footage matching as **Necromatcher**, with player tiles, persistent swing/model/control versions and downstream simulation/impact/analysis handoffs. Epic #11232 supersedes the narrower capture-only delivery scope. Tiger #11226 and Hogan #11229 remain open until reconstruction and real native replay qualify.

## Shared Authorities

- `SessionProjectStore`: durable player subjects, swing sessions and asset datasets; reused rather than creating a parallel result store.
- `ArtifactReference` and `compute_file_sha256`: checked recall of immutable bytes.
- `PiecewisePolynomialTorque` and `PolynomialSegment`: canonical control evaluation. A profile binds one model revision, ordered DOFs, N\*m units and physical seconds. Authored controls are identified as authored; detector playback timestamps cannot certify physical control time.
- `historical_capture`: rational container PTS, normalized image observations, frame hashes and source/model/code receipts.
- Canonical `user_config_path`: persistent local library location; `NECROMATCHER_LIBRARY_ROOT` can select another root.
- API route discovery: shared by server and packaged desktop; new Necromatcher routes require local evidence access, like the existing matched-swing browser.

## Current Delivery

### Sequential Repair and Review Exports

The current uncommitted `repair_native_motion` change reuses the canonical
`solve_trajectory` with native bounded TRF. The previous repaired pose supplies
the next initialization and weak prior; the first source pose initializes the
first solve. Reports identify `initialization=previous_repaired_pose` and
`prior_target=previous_repaired_pose_or_first_source_pose`. Marker targets still
come from the original inferred native world positions. They are not measured
3D landmarks. Twelve repair tests pass, but this method change has no new formal
whole-track execution receipt yet. Earlier LM and independent-TRF receipts retain
their actual algorithms, budgets, source hashes and failed midpoint evidence.

The public workspace facade now provides `export_fit_video(library, fit_id,
destination, selected_frames=...)`. It uses the bound native model and original
source-sized capture PNGs as backgrounds. The blue projected rig is a native
body-origin joint tree and attachment markers, not a rendered body mesh or an
observed silhouette. Green observations and yellow residuals retain their
distinct meanings. Original capture bytes remain unchanged. Export admits only
contiguous uniformly timed source frames; sparse or irregular timing requires a
separate export method instead of a fabricated constant physical clock.

Each new output directory contains a verified MP4, selected lossless overlay
PNGs and a completion manifest. The manifest binds fit/model/capture hashes,
original PNG hashes, exact rational frame PTS, output hashes, missing rig data,
physical-time qualification false, unqualified camera/anatomy flags and research
qualification. MP4 presentation uses the
source frame rate; the manifest is the authority for original PTS. Codec
readback checks decoded frame count and source dimensions. Existing destinations
are rejected; the manifest publishes last after staging and parent revalidation.

The actual Desktop review package is
`C:/Users/diete/Desktop/Necromatcher Review 2026-10-01`. New `Tiger Overlay` and
`Hogan Overlay` subfolders contain the v6 research overlay MP4s, selected PNGs
(Tiger 0/103/209; Hogan 0/549/749), and manifests. These depict the original
authored-ground hypotheses, not the unpublished repaired candidates. The
`Report` subfolder contains the compiled methods PDF and its LaTeX source;
the repository source is [Necromatcher Methods](necromatcher-methods.tex).
Built-in compilation failed with `Unable to find standard directories for
platform`; the installed MiKTeX PDF export was used without installing packages.
The standalone report is a methods/turnover artifact; canonical manual authority
and its blocked release inventory remain unchanged.

For repeatable export, follow the clean native-SDK process and new-directory
example in [Historical Capture Procedure](historical-capture-procedure.md).
API/UI export execution controls remain tracked in #11246; compiled report and
Desktop delivery are tracked in #11247. Physical-clock qualification, usable
continuous constrained motion, authored historical control recovery, downstream
simulation/impact/analysis and AffineDrift acceptance remain open under #11235
and #11232. No local artifact implies remote green CI or scientific acceptance.

Workspace CI completed three story remediation cycles: cleanup LoD, source-bound
launcher context/atlas freshness, and the GUI-thread heuristic. A new test
verifies a blocked background operation leaves Qt responsive and applies its
completion on the owner thread. Five native GUI/overlay tests pass; no GUI
ratchet baseline was increased. The remaining unit-lane failure is a runner Rust
toolchain install conflict before tests. See [Workspace CI Report](../ci-failures/11234-20261001.md);
keep delivery in progress. The generated API contract is now refreshed through
the library dependency, all six freshness tests pass, and web forms reuse the
canonical request types.

CI follow-up for draft #11237: its full unit lane reported 19,886 passing,
241 skipped and two failures. The API type freshness failure was reproduced
locally before regenerating `ui/src/api/generated/types.ts`; the added request
models require their canonical TypeScript projection. The second failure is the
existing child-copy contract requiring `origin/main` in CI; its shallow checkout
does not provide that ref. Keep this infrastructure failure separate from
Necromatcher acceptance and do not weaken the test. The SPEC changelog row now
references actual PR #11237.

Library issue #11233 is implemented in the owned `feat/necromatcher-library-11233` worktree. Code is published in [Draft PR #11237](https://github.com/D-sorganization/UpstreamDrift/pull/11237), based on capture PR #11231. This branch currently depends on the capture foundation PR #11231.

Players and swings survive process restart. Model bytes are copied and stored as unqualified candidates. Immutable version IDs reject overwrite. Recall checks bytes; profiles reject incompatible model revisions, joint order, units, physical clock and cross-player sessions. Library writers acquire an exclusive lock; concurrent writers fail visibly instead of losing metadata. Do not modify the same project through a separate raw store writer concurrently. A crashed writer can leave a lock: inspect its recorded PID and confirm no writer is running before removing that one lock file.

Capture archives retain receipt JSON, original JSONL bytes and lossless PNGs. Import validates every image hash, source identity, exact increasing PTS, bounded interval, counts and normalized observation validity. Only image-observation qualification is accepted. Missing source evidence or inferred depth cannot be promoted by saving an archive.

The export-mutation test failed before final ZIP byte verification was added. Exports are now verified after ZIP closure and atomically published without overwriting an existing file. Portable swing exports carry player/session identities, relative asset paths, hashes and original qualification metadata. Native model storage does not yet validate joint names against a running engine or assemble external model resources. Downstream qualified simulation handoff remains a separate acceptance requirement.

## TDD Evidence

- Initial library tests failed with missing module; implementation passed five persistence/compatibility tests.
- Capture import test failed with missing method; malformed capture then exposed KeyError handling, corrected to actionable payload rejection. Tiny generated test-video archive preserves exact receipt/JSONL bytes and PNG count; it is test evidence only.
- Export test failed with missing method before implementation.
- HTTP tests failed with missing route module before implementation; three tests now pass using the real persistent store, including import/export, duplicate/unknown identities and rejection of remote access to local evidence.
- The workspace and API suite passed 43 tests; scoped mypy passed for both library modules. A malformed-profile shape test then failed before an explicit JSON-object guard was added; final suite rerun passed.

## Remaining Work

Workspace #11234 now has a registered Necromatcher tile, React route, native
entry point and lazy embeddable adapter. A public default-library factory keeps
both hosts on one configured store. Shared `CaptureReview` checks the archive
hash once per opened version, retains original observation rows, verifies PNG
ZIP CRCs and rejects changes to the opened file's size/mtime. HTTP previews cache
up to four open captures; image reads never extract arbitrary archive paths.
Web forms save players/swings and import capture/model/profile versions.
The shared image overlay also replaces Video Analyzer's duplicate SVG overlay.
Frame requests hide old imagery while loading and retain source PTS/missingness.

Native review supports player/swing creation, recalled original PNGs and portable
export. Shared background workers perform costly verification outside the Qt
thread; a timer applies results on the UI thread. A failing KeyError test exposed
the worker adapter's limited exception contract; the adapter boundary now
translates expected lookup/type errors to a reported ValueError.
Thirteen UI tests and native recall/failure/import/overlay tests pass. Native
import forms and landmark overlays are implemented. A real Hogan practice frame
was rendered from the persistent capture ZIP, with PTS 3300000 × 1/30000 seconds,
750 source frames and physical time unknown. Preview outside Git:
`C:/Users/diete/Downloads/historical-capture/necromatcher-native-hogan-font-20261001.png`.
The Windows offscreen Qt plugin exposes no system font families; the verification
harness explicitly loaded the installed Segoe UI font. No product font fallback
or user preference was changed to accommodate that headless renderer.
Real web review on `http://127.0.0.1:5191/tools/necromatcher` against the owned
local API on port 8019 rendered both Hogan and Tiger PNGs and landmark overlays.
Hogan navigation reached source frame 750/750 at PTS 134.967 s. Physical time
remained unknown. Both player/swing selections recalled their proper captures.
The in-app browser review tab is marked for continuation. Backend/Vite process
sessions are 46801/73969; inspect their current state before reuse or shutdown.
Further form tests and final validation remain active. Navigation regressions
were reproduced before fixing the shared scoped request state: recalling another
player URL must hide the previous assets and swings immediately, failed frame
loads must stop their loading message and allow retry, and mismatched player/swing
URLs must not expose imports or exports. Sixteen UI tests now pass. Cleanup LoD
fix `d762816466` passes native tests, the global baseline and its CI gate.
Form verification now passes four additional tests: swing drafts reset when the
selected player changes, import drafts reset when the selected swing changes,
failed saves retain values for correction, and a pending model import retains
its original engine/joint payload. Gemini Flash 3.8 through `agy` supplied a
tool-free source audit; its potential asynchronous payload mutation claim was
rejected by the actual request-snapshot test. Twenty focused UI tests pass.
The official USGA source capture also renders its actual original PNG and
210-frame review in the browser at source PTS 15.015 s, physical time unknown.

CI cycle 2 identified stale agent-context generated views and launcher boundary
review. Reviewed the existing launcher-to-atlas contract, regenerated capability
atlas outputs and agent-context views, and renewed the source-bound review with
its explicit limitation: registry membership does not establish scientific
qualification. The atlas freshness test failed before regeneration.

Current checks: 31 library/API/native/launcher tests, 10 generated inventory tests,
13 UI tests; scoped mypy passes eight production files. Route-producer,
architecture, document title, catalog and design-manual governance checks pass.
Capture-cache retry testing failed before invalidation was added: a changed
archive returns 409 and clears the cached review; retry must hash-check anew.
Archive byte corruption remains rejected. Source PNGs and observations are
never changed by the desktop's detached overlay rendering.

The official USGA broadcast source now has a downloaded 2981–3267-second
excerpt outside Git, 1280×720 AV1 with audio, 30000/1001 presentation FPS,
clip duration 286.031 seconds and SHA-256
`6618fd8caf6a3c17ba4121d23b9bcdd576eff868a73045166f0a260705c30dde`.
Path: `C:/Users/diete/Downloads/historical-capture/tiger_2000/usga/bado2QdgD3c-teeoff-2981-3267.mp4`.
Its chapter label is insufficient golfer identity evidence: the contact sheet
mostly shows other players. The subsequent 3267–3616-second window is downloaded
(349.013 s, SHA-256 `e6b524474db4b800bafe8b98e754c8fc6d2fe7e43feec86b15db45adbac7889b`)
and shows Tiger warming up at its beginning, with Tiger/Ernie Els tee-time graphics.
A single 3250–3295-second excerpt now contains the full-body practice swing around
clip PTS 15–22 s; file `bado2QdgD3c-range-3250-3295.mp4`, duration 45.025 s,
SHA-256 `99e61d182c901548f3857d0325e23747d66b68db24610e16622de7e45d1dd673`.
The contact sheet shows address, follow-through and subsequent camera zoom;
dense continuous-shot review is next. Do not promote broadcast playback time to
physical swing time without independent timing review.

The 15–22 s range-swing extraction is complete: 210 frames, 210 detections using
the same full MediaPipe model and strict source/PTS contracts. It is recalled
from the persistent library as swing `tiger-usopen-2000-range`, capture version
`tiger-usga-range-capture-v1`, archive hash
`sha256:d4c10da4ac450d1e555ef08c04b1fc2e4688a0289c82afa70bcb9140ac4a7357`.
The original generic receipt retains unqualified timing/year fields; separate
source-catalog evidence attributes the official archive event to 2000.
No receipt fields were rewritten to manufacture scientific qualification.
Library PR #11237 is published as a draft over capture PR #11231.

Capture PR #11231's generated inventory correction passes unit/structure checks.
Its documentation check fails on `.jules/bolt.md:208`, inherited from main's
unrelated quaternion optimization. Record this external failure; the CI skill
forbids modifying pre-existing failures outside the story.

Real imports are complete and verified after reopening: Hogan practice 750 frames, perfection 899, compilation 839; Tiger practice 2,000. Library root: `C:/Users/diete/AppData/Local/upstream-drift/upstream-drift/launcher/necromatcher`. Media stays outside Git. Library validation passed: 43 workspace/API tests, 13 library tests after the ZIP typing correction, scoped mypy for all three production modules, repo-wide Ruff lint and format (8,269 files). Draft PR #11237 is published. Keep it draft until the base capture PR is accepted and dependency tracking is resolved. The public workspace facade and native-model/driving-profile/image-capture artifact contracts are registered. Implement the tile/web/desktop child #11234 under #11232, then wire real dense fitting and native downstream adapters under #11235 with evidence.

The existing `ModelMatchHandoffCoordinator` currently generates fixed output artifacts and hard-coded fit metrics. Do not call those results real matching or reuse that coordinator as Necromatcher scientific evidence. The shadow-tracker segmentation fallback issue #11227 also remains open. Physical-time calibration, shot continuity, club visibility, camera calibration and source-year lineage remain unresolved for the current clips.

## Capture Foundation CI

PR #11231 failed the 100-line function budget. Receipt and source-identity helpers reduce the function below the limit; local architecture check and 12 capture tests pass. Capture commit `c029a23e6b` also regenerates the required divergence inventory after the full unit gate exposed the missing capture entry (19,873 passed, one inventory failure). The inventory suite passed after regeneration; new CI is running. Earlier saved capture receipts correctly retain the pre-refactor implementation hash. The prior head passed repository-structure validation; current capture CI is running after the inventory update.

## Owned Native and Web Video Export Controls

Issue #11246 now uses NativeVideoSession over the existing matching-job service,
a clean SDK subprocess and guarded ZIP publication. Shared API routes admit,
poll, cancel and download by fit/run identity. Native and web controls bind the
selected source version; web URLs retain export_run during frame navigation.
Downloads require succeeded computation and execution_verified plus
download_available; scientific acceptance remains rejected research. Native
saves are exclusive, outside the library, and checked against source SHA-256.

Red-first API and native tests prove unknown identities, duplicate admission,
remote-client rejection, guarded download, response ownership, asynchronous
submission, cancellation, closed-state guards and corrupted-transfer cleanup.
The frontend extracts common refit/export polling and preserves prior refit
regressions; 37 related UI tests, TypeScript and scoped ESLint pass.

A wider run first exposed an intermittent request-reader failure whose exact
exception was truncated; subsequent focused and broad repetitions passed.
Separately, a held Windows reader decisively reproduced atomic promotion
WinError 5. Canonical io_atomic now retries only Windows permission/sharing
errors 5/32/33 for six attempts, with 0.31 seconds total backoff; permanent and
unrelated errors remain visible and owned stages are cleaned. Fifty focused
atomic-I/O, service and video-job cases pass. This evidence does not claim to
identify the original unknown reader exception or cure every filesystem race.

Gemini Flash 3.8 via tool-free agy audited supplied native/worker source. Its
transfer-integrity finding led to a failing corrupted-copy regression and
SHA-256 postcondition. Its claims about uncontrolled worker cancellation and
missing cleanup guards were checked against canonical ProcessGuard ownership
tests and the existing closed/worker guards; they were not treated as proven
failures. Actual UI exports and current committed-source identities will be
recorded below after the live journey. See necromatcher-video-export.md for the
repeatable control contract.

## Live Video Export Review — October 2, 2026

Source commit `3c73001a7c` was committed and pushed with normal commit and
pre-push gates passing. The live local desktop API was restarted from that
source before both exports. Actual web buttons started Tiger run
`165988043c994250aed72d3a57ebf027` and Hogan run
`f6daf6efec0d4dd8aa36a8a2424036cd`. Both reached succeeded execution with rejected
scientific acceptance and exposed verified downloads. Tiger frame navigation
preserved the export-run query. Hogan page reload and reopening Models and
Controls recalled the same completed run and download link.

The actual visible links downloaded both ZIPs into Downloads. Hash-checked
copies and expanded MP4/manifest/first-middle-last PNG sets were saved in
`Desktop/Necromatcher Review 2026-10-01/Tiger Web Overlay` and `Hogan Web Overlay`.
Independent OpenCV decode counted all 210 Tiger and 750 Hogan frames; every
video and PNG hash matched its manifest. Exact ZIP and manifest hashes are in
`historical_capture/live-web-export-review.json`. The UI proof screenshot is
`hogan-web-overlay-export-proof.png` in the Desktop review folder.

Final validation included pinned mypy, scoped Ruff, 50 atomic writer/job/export
tests, two combined 43-test Python runs, 37 UI tests, TypeScript and scoped
ESLint, API type freshness, manual governance and all normal Git hooks. These
are engineering checks; continuous grip/ground, anatomy, camera, physical time,
control replay and downstream acceptance remain unresolved.

### Compiled Methods Report Refresh

The polished standalone report now has 17 pages (490,594 bytes). Two existing
MiKTeX passes with installer disabled succeeded after removing a forced appendix
break and suppressing title-page anchors. Visual review of all rendered pages
found no clipping or overlap. Duplicate-anchor and overfull warnings are gone;
underfull paragraph warnings remain. Built-in compiler infrastructure remains
unavailable as recorded earlier; this is a verified fallback compilation.

Final Desktop PDF SHA-256:
`ab167d5a8571cbb0ed6d3f29a9bf10f6385cc2a370e24df140c354281e7d8650`.
Source SHA-256:
`867d90bbdba0b3ce50fb92b68449e505cd4c981b776a3490e0252d59085964c6`.
`Report/render-review-20261002-inventory-final` retains the 17 page renders.
README and `review-manifest.json` now cover 41 verified review artifacts,
including the live web export packages and proof. Render PNGs and compiler
intermediates are deliberately excluded from that review inventory.

## Interior Constraints and Spline Extrema — October 2, 2026

The public native constraint boundary now returns fixed 6D grip and declared
sphere-ground rows with analytic pose Jacobians, explicit sqrt(weight)/m-or-rad
scaling, ordered coordinates and immutable labels. Unsupported providers fail
explicitly. The historical solver opts into authored knot-interior fractions,
evaluates the union with original source PTS, and chains native constraints
through exact canonical spline bases. Image, prior, speed, RMS and point counts
continue to use original observations only. Default fits retain prior behavior.

Validated JSON configuration decoding preserves nested ground/options and
fractions across worker transport. The worker stores `constraint_assessment`
with `tested_times` (the source-clock union, including authored probes), row
labels, dimensionless residuals, a maximum and `continuous_certified=false`.
These timestamps do not create observed frame identities.

A separate typed helper assesses each scalar Hermite segment at endpoints and
real derivative roots. It reports extrema/worst times and finite two-sided
bound violations, including equal locked limits. Source samples must agree with
the stored spline at atol 1e-10 and rtol 1e-8. One-sided/infinite limits are
explicitly unsupported by this assessment and must remain separately recorded;
they are not silently cleared. Grip and ground remain `not_assessed` by the
coordinate helper. Neither assessment nor soft constraints enforce hard bounds
or establish accepted continuous historical motion.

Red-first suites cover overshoot, malformed/unsupported capabilities, fixed
contact rows, analytic coefficient derivatives, immutable/time-bounded reports,
unchanged image evidence, JSON/default compatibility and worker receipt
persistence. The combined new assessment/native/probe/image/worker/refit lane
passed 108 cases. Scoped Ruff and the repository-pinned mypy hook passed; a
separate CI-pinned direct native mypy attempt remains unverified and does not
supply a remote-green verdict. Procedures are in necromatcher-continuous-
assessment.md, necromatcher-native-constraint-boundary.md and
necromatcher-constraint-probes.md. Actual whole-track fitting must follow from
committed source, with independent feasibility and original-image review.

## Independent V7 Audit and Final Review Package

Producer source b92732e72579bfb72ba77d16f671f5f657a1b37c generated both v7 candidates and overlays. The independent clean-native audit verified all 500 implementation hashes before and after evaluation. Tiger: 419 source/midpoint poses, maximum grip gap 0.0385140 m, angle 32.8285 degrees, ten authored coordinate violations; zero penetration reflects floating geometry, not planted contact. Hogan: 1499 poses, gap 0.00629234 m, angle 5.98462 degrees, penetration 0.0172919 m and five authored coordinate violations. Both optimizers exhausted their budget and both candidates remain rejected. Coordinate extrema are assessed analytically; sampled grip/ground checks do not certify continuous feasibility.

Desktop `Necromatcher Review 2026-10-01` retains 60 hash-verified artifacts, including both v7 fit records, full independent assessment and reproducibility script, overlay ZIPs, source-sized MP4s and first/middle/last PNGs. All 960 v7 overlay frames were independently decoded. The compiled methods report has 19 pages; PDF SHA-256 is `7c2755f29cbf289c351eadf38eb823e49774491e8bbcca5cd0f6df52d7a1b0f6`. All pages were visually reviewed. Existing MiKTeX compilation succeeded with installer disabled; built-in compilation remains unavailable and #11247 remains open.

The branch subsequently advanced to b50e8fd91e through main merges and external CI fixes. These do not retrospectively qualify b927-produced artifacts or establish scientific acceptance for the newer source. Stash 832eac595f preserved the final report and two assessment summaries; only those files were restored, without dropping the stash. DL-#11235, DL-#11246 and DL-#11247 now track current feature state. Repository-wide development-log validation still reports pre-existing duplicate entries, missing verification fields and WIP/size ceilings; no global-green claim is made. Next implement hard whole-Hermite native coordinate bounds and explicit stance constraints, then repeat image/closure/contact audits before control and simulation qualification.

## Hard Coordinate Bounds and Truthful Overlay Readiness

The shared single-trial MAP estimator now accepts an opt-in immutable HermiteBoundsDomain. Bounded knot positions and normalized slopes decode to ordinary physical q/v coefficients whose segment Bernstein controls stay inside finite two-sided authored ranges. Nonuniform incident segment intervals preserve C1 continuity. Fixed coordinates remove decisions; all-fixed objectives bypass SciPy's zero-variable path. Physical callback Jacobians are chained through the decoder and shared prior columns remain intact. Active interval ties use a documented symmetric generalized derivative, not a classical smooth derivative. Unsupported multi-trial use rejects explicitly.

ImageFitConfig.coordinate_bounds transports immutable named native limits through validated JSON. Unknown/duplicate names, inconsistent locked seeds and infeasible initial Bernstein controls fail before optimization. Existing v7 fits are neither clipped nor retroactively accepted. Actual native image-target tests independently assess whole-cubic extrema and retain original point counts. Red-first domain, canonical/full-objective, JSON/native and compatibility checks passed a combined 106-case Python lane; scoped pinned mypy, Ruff, function/file/argument budgets and blocked manual governance passed.

Video completion now preserves historical execution separately from present downloadable artifact readiness. Worker completion brackets full authoritative output/parent verification with a bounded metadata/stat baseline; cheap polls fail closed on missing, extra, symlinked or changed outputs and source assets. Full ZIP/hash verification remains authoritative. Legacy completion records gain a baseline only through explicit guarded download, accessible as Verify Stored Overlay Package in both interfaces. Producer commit is disclosed; current-source equality is not claimed. Twenty job cases, ten native control cases and 28 web cases passed, with TypeScript, scoped ESLint, Ruff and pinned mypy.

The next historical trial must use an explicitly authored feasible initialization and exact native range/unit identities, then independently audit dense image fidelity, preserved-spline extrema, grip closure and planted stance. Coordinate bounds alone do not qualify contact, camera, anatomy, physical timing, dynamics, golf simulation or impact consumers. DL-#11235/#11246 remain active; release/manual and existing repository development-log findings remain open.

## Committed Bounds, Live Recall and Expanded Methods Report

Implementation commit 73ace83145 adds continuous coordinate bounds and artifact-readiness verification. The final compiled report now has 21 pages, including full Bernstein/slope/Jacobian equations, strict feasible initialization, tie limitations, and historical-execution versus present-readiness semantics. Two final installer-disabled MiKTeX passes succeeded; all 21 pages were visually reviewed with no clipping/overlap/overfull/rerun/duplicate-anchor warnings. PDF SHA-256: `a1c7c55fc196b199088fb7a28f5aa006a2da780480e85d1bd18ded1c516e2f8c`. V7 scientific results retain producer b92732e725 and rejected status.

The owned API was restarted on port 8020 (PID 39236; exec session 28022). A live Hogan legacy export exposed Verify Stored Overlay Package; its full guarded download produced SHA-256 `49aa8296d86e9414a3683e40237dd89e045de32560d99d2d0866ffe061456418`, identical to the previously checked package. Reload and drawer recall restored Download Overlay Package while continuing to disclose producer 3c73001a7c and current-source equality unverified. Root saved a proof PNG and structured live receipt to Desktop; the tracked receipt is historical_capture/guarded-overlay-recall-review.json. The latest Desktop inventory contains 62 reviewed artifact files. Native migration is unit-tested but actual player-package native UI exercise remains open.

Source and report commits progress #11235/#11246/#11247, close no issues and establish no remote-green or scientific-motion acceptance. The wider goal remains active: new feasible bounded player trials, explicit stance/closure, authored controls and independent replay, simulation/impact/whole-analysis consumers and qualified AffineDrift publication.

## Resume From Published Feature Evidence

Remote branch now contains merged HEAD c34fc1b968, preserving concurrent external test update fab9350c56. Normal commit/pre-push checks passed, including pinned mypy, Bandit and repository pre-push unit lane; the two changed native IK tests independently passed. No remote CI verdict or merge is claimed. Next implementation is documented in necromatcher-feasible-initialization-plan.md. Important range provenance: exact model definitions author coordinate_ranges_deg, while exported native joints use limited=false; the bounded optimizer enforces the authored hypotheses, not pre-existing compiled joint limits. No new bounded player trial has run.

## Explicit Authored Initialization and Native Range Identities

The canonical estimation facade now exposes a pure authored initializer with immutable coefficients, original/initialized little-endian float64 SHA256 identities, every changed knot/source time and per-coordinate displacement. The strict image-fitting default still rejects infeasible initial Bernstein controls. The explicit policy projects bounded free knot positions once and zeros all free slopes; original pixels/confidence and the original seed prior remain unchanged. Shared fit preparation/result assembly measures the actual initialized image RMS, while a separate initializer-only result reports optimizer_ran=false and converged=false.

NativeFitBinding captures the exact definition serialization used to bind the resource. Its public authored_coordinate_bounds method verifies exported XML identity, scalar native names/order/units and limited=false, then converts declared angular degree ranges to immutable radian bounds. Missing limits are explicitly unbounded. Changing the detached fit dictionary cannot substitute a different range policy. This is authored model provenance, not historical anatomical measurement.

The owned canonical research job path adds an author_initialization operation that publishes a separate immutable seed fit version with original parent lineage and fresh source-frame/held-out metrics. No second scheduler or optimization implementation is introduced. Exact authored ranges and custom limits remain distinct; historical anatomy and nonlinear continuous constraints remain unqualified. Pure/domain/image suites passed 67 cases; canonical estimator/extrema/native binding compatibility passed 66 cases, with pinned mypy and scoped Ruff. Actual saved bounded Tiger/Hogan trials have not yet run. The report draft compiles to 23 pages with the new method equations and receipt procedure; final evidence publication and page review follow the new trials.

Next freeze source, author separate Tiger/Hogan seeds, run finite-budget bounded fits with explicitly recorded heel-contact hypotheses, audit original frames/midpoints/cubic extrema and image fidelity, export source overlays, then update the Desktop manifest and compiled report. Existing V7 results retain their original producer and rejected qualification.

## Committed V8 Bounded Trials and Desktop Review

Producer 2ad99b870a12b1683fe19a82100ebcf22aefa237 ran four owned jobs: separate Tiger/Hogan authored seeds (optimizer_ran=false) and subsequent bounded heel-hypothesis fits (30 evaluations, 300-second wall guards, twelve knots). All computations succeeded and all remain scientifically rejected. Both fit jobs re-sample the parent's source poses and reinitialize slopes; they do not restart exact preserved Hermite coefficients. Each new fit prior uses its seed parent's first pose, while pixels and prior are unchanged within the job. Exact coefficient restart and preserving complete bound/contact recipes through both forms are next implementation work.

The independent audit reloads exact parents, brackets source/runtime/script identities, and evaluates every original source PTS, adjacent rational midpoint and canonical scalar global extremum without optimizing or adding image targets. Both stored splines pass all 32 authored scalar ranges, with 12 coordinates unbounded. Tiger has 507 evaluation records at 465 distinct times and Hogan 1587 at 1547. Maximum sampled grip gaps are 50.778827mm/15.424154mm, angles 27.785302/17.021964degrees and penetration 31.379098/6.523293mm. Finite nonlinear sampling remains uncertified. Tiger's image fit worsened; constant two-heel pins across follow-through/walking are an inadequate whole-interval assumption. Next review swing/stance intervals, add phase-dependent contact hypotheses and improve camera/landmark/seed evidence.

Both V8 overlay jobs retained the producer, succeeded with verified guarded downloads and rejected scientific qualification. Root decoded all 960 source-sized MP4 frames, checked PNG/video hashes and visually reviewed six stills. Live Hogan web recall listed both new versions, projected the source frame, reopened the completed export and downloaded ZIP SHA-256: 567e77317ab777c05d80726a3301442479be2a9b5872b601030cb0ad1a6d3c65, identical to the CLI package. Current API source equality remains unverified. Desktop proof and tracked bounded-v8-web-recall-review.json retain this result.

The final polished report has 25 pages, compiled twice with the existing MiKTeX installer disabled. Every page was visually reviewed, new methods/trial pages also full size; no clipping, overlap, table overflow, overfull boxes, reference/rerun/duplicate-anchor warnings. PDF SHA-256: e40bc030acce622ee5af22abbc4be967e720ab53c5a779273a7d7f1cc3d1671d; exact source SHA-256: f0ebfbf88bb681a4a49f1b0d9bf3b0e39357a570e6eee6f44af38a05fc8e1af8. Desktop inventory now contains 89 independently hash-verified artifacts; previous 21-page PDF/source remain under Report/snapshots/hard-bounds-readiness-21-pages. Built-in compiler setup, remote CI/merge, qualified historical fitting, native player-package UI exercise, replay controls and downstream/AffineDrift acceptance remain open. No issue or full-goal completion is claimed.

## Exact Restart and Complete Recipe Transport

The public ImageSplineStart and strict preserved_spline job mode now retain exact physical coefficients and the source knot clock. Shared SDK-free recall rejects malformed declared identities; native execution independently verifies model/order, interval, knot count and all parent poses. Both forms retain the authoritative full ImageFitConfig, with canonical nested API transport and explicit rejection of ambiguous legacy scalar mixing. Tests cover exact q/v/a equality, no sampling/reauthoring, malformed identities and recipe retention. Actual committed Tiger/Hogan restart receipts and report updates follow source freeze. See [Saved-Spline Procedure](necromatcher-saved-spline-restart.md). Qualified historical contact, camera, dynamics and downstream handoffs remain open.
