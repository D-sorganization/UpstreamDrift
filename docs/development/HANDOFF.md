# Reference Asset Implementation Handoff

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: C:/Users/diete/Repositories/Worktrees/UpstreamDrift-reference-assets
- Branch: feat/9864-reference-assets
- Baseline commit: db4862fb8 (drawings stack; dependency #9868 now merged)
- Implementation commit: SELF
- Pull request: #9870 (draft)
- Governing issue/epic: #9864, #9863

## Objective and Status

Validated expert C3D/model marker/video imports with a visible native library.
Implementation is in progress; registration #9865 and projection/sync UI #9866
remain separate work. Drawing #9869 and fleet rollout #1588 are in protected CI.

## Files and Decisions

New reference model/importers/storage and native mapping/library dialogs reuse
C3DAdapter, BodyTarget JSON loader, CIR MarkerTrajectory, VideoReader, atomic
write_document, provenance hashing and FlowLayout. No vendor edits. Explicit
source-unit/axis/name choices precede 3D registration. Missing points remain null;
zero coordinates remain valid. Video clocks are nominal-frame-rate only.

The shared C3D adapter retains complete labels and rejects duplicates. ezc3d
residuals come from meta_points, not the XYZ1 fourth row. Source arrays are never
filled or resampled. Existing pipeline.py helper is aligned to merged editing
head 4909ee460; this is a dependency update, not a new analytics change.

## Validation

- 21 backend/import/C3D tests pass, including a real ezc3d-written fixture.
- Two initial C3D fidelity regressions failed before the adapter correction.
- Four native tests pass (notes/archive, mapping confirmation, video source
  immutability, unsaved close). Initial 808-px minimum-width failure was corrected
  by a scrollable source-path field and existing FlowLayout.
- Eight modules pass mypy with --follow-imports=silent. Architecture budget passes.
- Design-manual governance passes; release remains blocked-inventory-required.
- Remaining: expanded integration/negative cases, map freshness, full LoD, visual
  review, final formatting and protected PR checks. No physical calibration claim.

## Risks and Compatibility

Worker lifecycle prevents closing while a bounded import/write runs. No immediate
native-parser cancellation is claimed. Sources are linked, not copied or deleted.
CIR/model inputs must contain Cartesian marker trajectories; arbitrary joint-angle
or mesh files are not silently converted. Shared clones and peers' worktrees intact.

## Next Steps

1. Complete reference-library integration, visual review and generated checks.
2. Commit with DL-#9864/SPEC evidence; integrate merged drawings/main and submit PR.
3. Implement #9865 registration/time mapping and #9866 comparison/export controls.
4. Verify fleet #1579 adoption after #1588 runner fix merges and live sync executes.

## Change Log

- SELF — Build reference import contracts, native management and shared C3D fidelity fix.

- SELF — Expanded qualification: 30 backend/import/C3D/reference UI/existing-library tests pass. Fresh-process body JSON isolation reproduced an eager legacy C3D import failure; the shared loaders facade now lazily resolves optional formats, and the regression plus import tests pass. Standalone previews opened: library 740×660 (184-px minimum), mapping 640×650 (441-px minimum). Mapping now applies the shared theme. Atlas/parity freshness passes. Full unbaselined LoD reports existing repository findings; use the configured no-growth baseline and fix the new source-path access through the asset-owned property.

- SELF — Final local checks: 49 registry/atlas tests pass; atlas and parity artifacts are fresh; eight-module mypy passes; configured LoD no-growth check passes (2,987 source files, existing baseline retained). Visually inspected both screens including themed mapping. A cancelled archive-filter change now restores the visible filter and preserves unsaved notes (RED then four UI tests GREEN). Scoped Ruff/format pass. Keep the single current #9868 SPEC row while recording the new reference change.

- SELF — Merge reviewed drawings head 3f8b592bf through a normal merge after reference implementation 269354d07. Preserve the complete reference additions and the same drawing/editor behavior; the extracted sparse-summary helper already matches merged #9868. Reference PR is now a focused addition to this dependency.

- SELF — Published draft PR #9870 after normal push hooks. Local import and library qualification is complete. Integrate drawing theme correction before readying this PR. Fleet runner repair #1588 merged; live rollout 34330895204 exposed Git credential routing and a stale clone-root handoff, being corrected in an isolated RM branch.
