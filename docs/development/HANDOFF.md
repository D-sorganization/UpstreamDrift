# Implementation Handoff

## Identity

- Repository: `D-sorganization/UpstreamDrift`
- Working directory: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-coaching-drawings`
- Branch: `feat/9862-coaching-drawings`
- Baseline commit: `156615443` (editing/library PR #9868 with concurrent navigation fix preserved)
- Implementation commit: `SELF`
- Pull request: #9869 (draft)
- Governing issue/epic: #9862; #9849

## Objective and Status

- Objective: Give instructors saved visual references over original camera video, with intuitive editing and consistent exports.
- Status: draft review; dependency integration pending
- Completed: Five shape tools, source-pixel/frame document, validation, bounded undo/redo, draw/select/move/resize, numeric and keyboard editing, style/visibility, atomic save/reopen, library/editor access, common preview/PNG/video renderer and portable sidecars.
- Remaining: Final focused validation, visual review, protected PR and current-head CI. Advanced external reference epic #9863 and fleet adoption remain separate open work.

## Files and Decisions

- Files changed: src/motion_capture/coaching, coaching_canvas.py, coaching_dialog.py, coaching_export.py, existing clip/export/editor/library adapters, tests, generated feature/workflow maps and user documentation.
- Key decisions: Use the native ImageCanvas coordinate transform and OpenCV export renderer. Visual reference layers never enter scientific landmark observations. Draw before crop and retain original frames. Reuse existing export worker and exclusive publication helper. Tools Fabric is a workflow reference, not falsely compatible serialization; no vendor edits.
- User-owned or unrelated worktree changes: none; concurrent editing branch update c401d006e preserved through baseline 156615443. Shared clones and other agents' branches untouched.

## Validation

- Initial RED: new coaching test failed because the module did not exist. Nine backend shape/history/persistence tests then passed.
- Initial integrated drawing/export suite: 18 passed. Ten source modules pass mypy with repository follow-imports=silent settings.
- Broader UI integration found a fixed-row minimum width of 952 px after adding Draw References; switched to existing FlowLayout. Repaired integrated drawing/library/editor/clip suite: 28 passed.
- Initial visual review at 900x720 showed all controls and a 496-px minimum width. Reduced selection handles to screen-relative size and added a visible stroke-unit label/colour value.
- Subsequent final checks are recorded before PR submission; synthetic media do not qualify physical camera calibration or instructor usability.

## Blockers and Risks

- Blockers: none for implementation; protected CI remains required before merge.
- Risks/assumptions: Video exports are silent. Media/sidecar publication rolls back ordinary errors but is not atomic across power loss. Imported Fabric documents, freehand/text tools and external expert projection are outside this shape layer; #9863 remains open.

## Next Steps

1. Integrate dependency PR #9868 after its current-head protected checks pass.
2. Qualify the focused #9869 diff against merged main, then mark ready; respect normal branch protection.
3. Continue advanced reference imports/registration/comparison and verify fleet adoption.

## Change Log

- SELF — Add native saved coaching references and shared image/video rendering, preserving source media and scientific observations. Reuse existing layout and export lifecycle primitives; correct the laptop-width regression discovered by integration tests.

- SELF — Final qualification: registry/atlas 49 passed; latest library/drawing/export suite 19 passed, including immutable worker snapshot, output sidecar race and copied reference layers. Twelve-module mypy plus storage/library follow-up pass. Full LoD scan is clean across 3,012 files. Generated atlas, parity matrix and guide are fresh. Final visual review: drawing dialog 900x720/minimum 496 px; swing editor 850x650/minimum 465 px. Both remain readable.

- 2d92e179d — Integrated protected main with atlas and performance improvements; preserved both development-log entries and all SPEC rows.

- SELF — Add cancellable swing export with provenance, source-timeline initialization and analytics regressions, and regenerated workflow/capability references. Export/editor/clips 12 passed, reconstruction/analytics 11 passed, registry/atlas 49 passed, repaired guide/timing 10 passed, five-module mypy passed.

- SELF — Record PR #9868 and passing normal push hooks (including mypy, Bandit and unit tests); place its single SPEC row inside the canonical change-log table.

- SELF — CI identified two library navigation calls reaching through CapturePanel into its text control. Added a panel-owned set_session_path API and exercised the real library-open callback in the integration test; no baseline weakening.

- Verification: full LoD scan passes (3,005 source files), six GUI/library integration tests pass, and gui.py passes mypy after the navigation API correction.

- SELF — Integrated concurrent agent commit c401d006e through a normal merge. Preserved its panel-owned set_session_path API and retained the callback integration tests; removed the equivalent duplicate accessor.

- SELF — Full CI #9868 exposed a sparse-manual regression (unit job 102370002003): summary availability must not block reconstruction/model fitting. Added a specific SwingDataUnavailable result path, recorded the reason in session_reconstruction.json, and removed stale dense summaries before retaining sparse output. Existing every-fifth-frame model-fit qualification now passes along with nine pipeline/analytics checks (10 total). Scientific calculations and interpolation policies remain unchanged.

- SELF — Integrated latest main dba24ceb7 (Tools canonical theme/notes migration) in merge 3a10d4986. The corrected sparse-fit, trim-timeline and native GUI/library/editor suite passes 15 tests with the pinned Tools theme; atlas freshness passes. Two-module mypy and design-manual governance pass.

- SELF — Integrated merged dependency PR #9868 on main into #9869. Ready for protected PR review and CI.

- SELF — Integrated editing fix 951c94ed8, latest Tools-canonical migration from main dba24ceb7 and qualification commit 1e2469296. Sparse manual captures now retain reconstruction/model fit while recording summary unavailability and removing stale metrics. Dependency validation passes 15 sparse/timeline/GUI/editor tests; drawings remain separate additional behavior.

- SELF — Record PR #9869; post-dependency integration passes 22 drawing/editor/library tests and atlas freshness. Normal commit and push hooks passed. External editing-branch architecture extraction 4909ee460 is integrated cleanly.
