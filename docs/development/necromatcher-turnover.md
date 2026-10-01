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
