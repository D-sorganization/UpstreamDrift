# C3D Reference Fitting Handoff

- Repository/worktree: `D-sorganization/UpstreamDrift`, `../UpstreamDrift-reference-9914`.
- Branch: `feat/c3d-reference-overlay-9914`; baseline `7c09642df`; commit SELF.
- Governing epic: #9914; PR not created; development entry DL-#9914.
- Complete: marker profiles, URDF and compiled-MJCF tree adapters, root seed,
  existing continuous fit orchestration, saved jobs and library assets,
  fixed placement estimator, standalone keyframe graphic, operator guide.
- Validation: RED observed before each new module; 80 reference tests pass;
  new two-camera renderer, preview, identity and custom-model contracts pass
  separately. Scoped Ruff passes; one mypy matrix typing issue corrected,
  recheck pending. Exact commands are in the epic document.
- Evidence: provisional model surveys under `../reference-fit-artifacts-9914`.
  Full driver and iron runs are still active; do not treat provisional artifacts
  as final source-version evidence. Native OpenSim adapter and MyoSuite anatomy
  are unavailable, explicitly recorded rather than replaced with a fallback.
- Coordination: `codex-reference-9914-20260909`, issue lease and central presence
  registered. #9913 owns capture GUI changes; no edits to those files here.
- Next: finish evidence, rerun final checks, commit/push with normal hooks and
  create a protected PR. Do not close the epic without implemented acceptance.

# Guided Capture Setup Handoff

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-capture-setup`
- Branch: `feat/capture-guided-setup`
- Baseline commit: `54b9b0586` (main after qualified product merge #9896)
- Implementation commit: `SELF`
- Pull request: #9910 (draft): https://github.com/D-sorganization/UpstreamDrift/pull/9910
- Governing issue: #9898; epics #9897, #9902, #9906; development entry DL-#9898
- Session: `capture-product-01a08427-guided-setup`, issue lease and presence active.

## Objective and Status

Complete the newly requested everyday calibration, sourced club database/player bag,
and capability-driven setup wizard. The full prior fleet rollout also remains required.
Detailed execution children and dependency/acceptance boundaries are recorded in
`docs/development/capture_setup_execution_plan.md` and the three linked GitHub epics.

The current change includes intrinsic profiles and visible Capture Rig header/dialog
controls. Common-reference observations, downstream stale-selection guards, multi-view
geometry, club data and wizard work remain open. Do not close #9898 or its epic yet.

## Files and Decisions

- `calibration_profiles.py` references existing IntrinsicsRecord artifacts; it creates
  no reference solver or competing camera contract. Manual lens settings need fresh
  confirmation; camera/lens/zoom/focus/image size/sensor-crop changes reject reuse.
- Saving a profile preserves an atomic calibration artifact revision separately from
  the mutable input filename, with source provenance and digests. Old captures can
  restore a prior revision after recalibration. Corrupt history/revisions are errors.
- `test_calibration_profiles.py` covers compatibility, invalid quality/geometry,
  changed/missing files, idempotency and preserved revisions. The plan records all
  ten execution issues across the three new epics.
- Existing #9621/#9622/#9623/#9630/#9554 and ADR-0041 remain authorities/dependencies.
  Common paper/yardstick observations cannot silently certify full 3-D geometry.
- Preserve existing club-data and setup/workflow/registry infrastructure. Unknown
  club properties remain unknown, with published/measured/estimated distinctions.
- Reconciled this task's completed DL-#9894/#9892/#9883 entries against the qualified
  #9896 merge; earlier product epics #9849/#9850/#9863 are complete. Root handoff
  retains standing UP-D0/UP-D1 governance and other agents' historical context.
- User-owned changes: none in this new isolated worktree. The launched application
  remains in the separate UpstreamDrift-ubuntu-ci checkout.

## Validation

- Added rig-wide verified export and per-camera profile review, with confirmation reset
  on edits, saved revision selection and an existing-board recalibration route.
- Entire `tests/tools/capture_rig` suite passes (388 tests). Focused UI tests cover
  explicit confirmation, edited settings, export provenance and repeat-without-export.

- TDD: profile module initially absent; revision-preservation regression then failed
  before implementing archived snapshots. Existing normal repo conftest is enabled.
- `python3 -m pytest tests/tools/capture_rig/test_calibration_profiles.py tests/motion_capture/reconstruct/test_intrinsics.py -q --timeout=60`: 31 passed, including archive idempotency/corruption, geometry rejection and existing solver/CLI behavior.
- Changed-file Ruff lint/format pass after reviewed dict-literal fixes. Focused mypy (`--follow-imports=silent --ignore-missing-imports`) passes for the new source module. Catalog, title, SPEC and tracked file-size checks pass; staged checks and all normal commit/pre-push hooks passed. The default shared mypy cache had incompatible NumPy stubs; the unchanged hooks passed in the task-qualified PRE_COMMIT_HOME cache.
- Pinned Tools submodule initialized at eab74a901a7c8467e1997049a73e2cfd2df74428.

## Blockers and Risks

No blocker prevents continued feature implementation. Profile review is wired into
Capture Rig; downstream session/geometry qualification remains required before shipment.
Do not claim physical calibration accuracy from synthetic tests alone. Generic
calibration/reference geometry belongs to Tools under ADR-0041.

Fleet context: central handoff PR #1628 merged as 537f9ad087dd3afdda60d28dd2e54d1ac7583864. Half-ton-controls #3 merged as
305a21307ed6f6e2a3c500da6a3c4a50cdbb73db after run34391771678 passed; lease released.
New audit at19:53UTC confirms39/41, including Tools_Private. Tools and Gasification_Model
still lack adoption; direct notices request the canonical replacement queue. Do not race closures.

## Next Steps

1. Continue #9910 with the player workflow; the foundation at b4abd67bb7b6629e0992364037b4f3fd0d0e816e is locally qualified and published.
2. Implement common-reference observation sessions and compatible profile UI, then
   qualify consumers before closing #9898/#9900. Follow Tools geometry ownership.
3. Execute #9899/#9901, club children #9903-#9905, and wizard children #9907-#9909.
4. Continue actual fleet adoption and audit every required result before goal closure.

## Change Log

- `SELF`: start DL-#9898, preserve three new goal epics, and implement tested optical
  profile compatibility with durable calibration revisions; UI integration remains open.

- `SELF`: add visible calibration revision review and verified per-view exports.
  Test launch PID49580 exited: missing imageio_ffmpeg escaped automatic preview.
  Issue #9911 adds import/timeout recovery; 11 preview regressions pass. An isolated
  TEMP/upstreamdrift-capture-test-runtime now supplies the declared FFmpeg dependency.
  Verify the current launch before claiming the application is still open.
