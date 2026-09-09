# Guided Capture Setup Handoff

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-capture-setup`
- Branch: `feat/capture-guided-setup`
- Baseline commit: `54b9b0586` (main after qualified product merge #9896)
- Implementation commit: `SELF`
- Pull request: not created.
- Governing issue: #9898; epics #9897, #9902, #9906; development entry DL-#9898
- Session: `capture-product-01a08427-guided-setup`, issue lease and presence active.

## Objective and Status

Complete the newly requested everyday calibration, sourced club database/player bag,
and capability-driven setup wizard. The full prior fleet rollout also remains required.
Detailed execution children and dependency/acceptance boundaries are recorded in
`docs/development/capture_setup_execution_plan.md` and the three linked GitHub epics.

The current change is a tested intrinsic-profile foundation, not a completed player
calibration workflow. Common-reference observations, actual UI integration, multi-view
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

- TDD: profile module initially absent; revision-preservation regression then failed
  before implementing archived snapshots. Existing normal repo conftest is enabled.
- `python3 -m pytest tests/tools/capture_rig/test_calibration_profiles.py tests/motion_capture/reconstruct/test_intrinsics.py -q --timeout=60`: 31 passed, including archive idempotency/corruption, geometry rejection and existing solver/CLI behavior.
- Changed-file Ruff lint/format pass after reviewed dict-literal fixes. Focused mypy (`--follow-imports=silent --ignore-missing-imports`) passes for the new source module. Catalog, title, SPEC and tracked file-size checks pass; staged checks and normal hooks remain required.
- Pinned Tools submodule initialized at eab74a901a7c8467e1997049a73e2cfd2df74428.

## Blockers and Risks

No blocker prevents continued feature implementation. The profile API is not yet
wired into Capture Rig or downstream geometry, so it is not a shipped zoom guard.
Do not claim physical calibration accuracy from synthetic tests alone. Generic
calibration/reference geometry belongs to Tools under ADR-0041.

Fleet context: central handoff PR #1628 merged as 537f9ad087dd3afdda60d28dd2e54d1ac7583864. Half-ton-controls #3 candidate
f17766c5eb0ced58261f85a0869796807c6f401d awaits quality job 102603927666 in run
34391771678 (live watch 87268). Tools#5127, Tools_Private#1475 and Gasification_Model#4942
have direct conflict notices requesting their claimed canonical replacement queue.
Last complete adoption inventory: 37/41. Do not restart live jobs or race closures.

## Next Steps

1. Complete final tests/type checks and normal hooks for this foundation.
2. Implement common-reference observation sessions and compatible profile UI, then
   qualify consumers before closing #9898/#9900. Follow Tools geometry ownership.
3. Execute #9899/#9901, club children #9903-#9905, and wizard children #9907-#9909.
4. Continue actual fleet adoption and audit every required result before goal closure.

## Change Log

- `SELF`: start DL-#9898, preserve three new goal epics, and implement tested optical
  profile compatibility with durable calibration revisions; UI integration remains open.
