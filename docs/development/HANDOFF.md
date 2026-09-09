# Impact Shaft Provider Integration (#9912)

## Current Continuation State

- Working directory: C:/Users/diete/Repositories/UpstreamDrift-impact-provider-pin.
- Branch: feat/9912-impact-provider-pin; implementation commit b6107f8e2; PR #9916.
- Original base: 6e3610a9b; incoming main: 18c8f922e; Tools candidate: 00d17e7f91fe8541bc8882ee745fda58ee2ad7af.
- Governing issue #9912, development entry DL-#9912, parent #9703/#9701/#9700.
- Pair: Tools #5133. No shared source is copied or modified in this consumer.

The new tests/shared_contracts/test_impact_shaft_provider.py exercises the strict
golf_club.distributed_shaft/1 public input format and canonical theme API through
the existing provider-resolution harness. The synthetic fixture verifies coupled
stiffness, integrated mass, canonical roundtrip/digest, exact source-byte checks
and preserved unqualified status. Adverse cases reject version changes, missing
calibration fields, numeric strings and attempted qualification promotion.
Theme coverage checks resolved defaults, custom tokens and independent output.

Before changing the old eab74a901 pin, all six tests failed: five missing shaft
module cases and one missing resolved-theme method. After selecting the exact
candidate, all six pass; the whole provider suite passes all 24 cases with five
existing import-alias deprecation warnings. Run with the established Python 3.12
environment, REQUIRE_REAL_TOOLS_REPO=1, TOOLS_REPO_PATH pointing to this worktree's
vendor/ud-tools, and pytest tests/shared_contracts --tools-mode=vendored -n 0
--no-cov. Numerical libraries use one native thread; Qt is offscreen.
Pinned Ruff 0.15.17 check/format passes all 6,840 files, along with manual,
document-catalog, size and title checks. Ten existing packaging/provenance tests
pass. The local development-log validator file is absent despite synced policy;
the central validator at ad9bcb885 reports 40 inherited findings versus 41 on
base, with no new findings after normalizing shifted diagnostic line numbers.
DL-#9830 and DL-#9825 now concisely record their verified merged results; detailed
evidence remains in their existing turnover documents. Other owners' entries
are preserved. The Python-only wheel built from b6107f8e2 installs with its full
declared core dependencies into a clean environment outside the checkout. Both
shaft and theme imports resolve under site-packages. Canonical wire/digest,
coupled inputs, tamper refusal and theme ownership checks pass; pip check passes.
Wheel SHA256: 6f6f259ff9679f921a5a05e515abcb4bb466589221ce6c7aefdc43b9ca656653.
Runtime: Python 3.12.10, NumPy 2.5.3, SciPy 1.18.1, Pydantic 2.13.5.
SKIP_UI_BUILD=1 is the existing Python-provider build path; this evidence does
not qualify a UI/release artifact. The final reviewed provider pin remains pending.

Main 18c8f922e is integrated with only four shared-document conflicts. Both
owners' entries and all incoming implementation are preserved. The provider
suite passes all 24 cases after merging (five existing alias warnings); manual,
doc-size and scoped Ruff checks pass. Source diff from main consists solely of
the candidate vendor pin and its two test/fixture files. Final reviewed Tools
repinning and protected provider/consumer delivery remain pending.

Post-merge audit removed one empty conflict-created heading while retaining
the complete shipped #9894 entry. SPEC rows exactly equal the union of both
parents, with no extras or omissions. The current main development-log audit
has 36 inherited findings versus 37 on main, with no new findings; earlier
40/41 counts above belong to the original base. Merge commit: 70243adfe.

## Remaining Work

1. Incorporate the reviewed Tools repair revision and rerun the provider checks.
2. Repeat installed-consumer validation when the final provider pin changes.
3. Finish protected checks on PR #9916 and its Tools #5133 pair before
   merging in provider-then-consumer order.
4. Continue the full #9703 engine adapters and registered #9704 studies.

This pin does not close physical calibration, flexible impact, acoustic radiation
or blinded sweetness qualification. Tools #5133 still has protected CI failures;
private Gasification checkout access is separately unresolved. Current dirty
files belong to this branch; other worktrees and user-owned source are untouched.
The complete presence read found common handoff/SPEC/development-log overlap
with capture-product sessions #9898/#9913, but no implementation-path overlap.
Each session uses its own worktree; preserve their incoming metadata during
merges. An unrelated rejected identity-change warning is retained in inbox evidence.

## Preserved Incoming Handoff

The following incoming handoff is preserved from main 18c8f922e, including its original ownership and historical status. The prior provider-branch handoff is retained at the immutable c315c4b74 revision.

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
