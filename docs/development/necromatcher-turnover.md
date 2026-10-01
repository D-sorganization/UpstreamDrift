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

Real imports are complete and verified after reopening: Hogan practice 750 frames, perfection 899, compilation 839; Tiger practice 2,000. Library root: `C:/Users/diete/AppData/Local/upstream-drift/upstream-drift/launcher/necromatcher`. Media stays outside Git. Library validation passed: 43 workspace/API tests, 13 library tests after the ZIP typing correction, scoped mypy for all three production modules, repo-wide Ruff lint and format (8,269 files). Draft PR #11237 is published. Keep it draft until the base capture PR is accepted and dependency tracking is resolved. The public workspace facade and native-model/driving-profile/image-capture artifact contracts are registered. Implement the tile/web/desktop child #11234 under #11232, then wire real dense fitting and native downstream adapters under #11235 with evidence.

The existing `ModelMatchHandoffCoordinator` currently generates fixed output artifacts and hard-coded fit metrics. Do not call those results real matching or reuse that coordinator as Necromatcher scientific evidence. The shadow-tracker segmentation fallback issue #11227 also remains open. Physical-time calibration, shot continuity, club visibility, camera calibration and source-year lineage remain unresolved for the current clips.

## Capture Foundation CI

PR #11231 failed the 100-line function budget. Receipt and source-identity helpers reduce the function below the limit; local architecture check and 12 capture tests pass. Capture commit `c029a23e6b` also regenerates the required divergence inventory after the full unit gate exposed the missing capture entry (19,873 passed, one inventory failure). The inventory suite passed after regeneration; new CI is running. Earlier saved capture receipts correctly retain the pre-refactor implementation hash. The prior head passed repository-structure validation; current capture CI is running after the inventory update.
