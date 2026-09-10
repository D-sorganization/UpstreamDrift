# Player Bag and Capture Equipment Handoff

## Identity

- Repository: D-sorganization/UpstreamDrift.
- Working directory: C:/Users/diete/Repositories/Worktrees/UpstreamDrift-player-bag.
- Branch: feat/9905-player-club-bag; main integration includes6f2d63325 (reference PR #9918).
- Implementation: b206ae943; draft PR #9923. Main/reference integration is being qualified.
- Governing issue: #9905, epic #9902; development entry DL-#9905.
- Session: capture-product-01a08427-player-bag; lease through01:00UTC; scoped presence renewed through01:45UTC.

## Current Work

PlayerClub/PlayerBag/CaptureClubSnapshot extend the existing club-data authority.
Catalog bases and player overrides stay separately inspectable. Capture snapshots
bind to the portable capture ID and verify a content revision. The equipment adapter
uses the existing atomic document writer, keeps prior capture selections in
`equipment_revisions`, rejects stale bag saves and reports corrupt records.

My Clubs now opens from the header and selected library capture. Catalog search
exposes source links; custom clubs, measured/estimated/unknown quantities, canonical
unit controls, notes, archive/restore and assignment have visible outcomes and help.
Editable copies rebind equipment to their new capture ID. Portable notes moved to
rig/capture_notes.py with compatible library re-exports; rig/equipment.py is the
headless reader. Model session write_fit stores the exact selected club, eligible
SI context and withheld reasons in hashed provenance, with no applied club constraint.

652 capture/model/catalog/inventory tests pass. Initial editor TDD failed on the
missing module before implementation. Visual review passed at760x600 /640x500;
Windows offscreen QA required explicitly loading Segoe UI, with no production font
change. Evidence: TEMP/player-bag-visual-katpe_dp/\*-font.png. Full Ruff lint/format
(6880 files), scoped mypy and architecture/doc budgets pass. Five new LoD chains
were repaired using the player identity facade and local evidence values. The3020-file
LoD scan passes with490 baseline occurrences and60 reductions. Final69 focused tests
plus map/parity checks pass (11 existing warnings). Wizard entry remains
required under #9906 before #9905 closes. Normal commit and push hooks passed for b206ae943.

Only byte-identical repeated Field Reference tables were removed from the development
log to make room for the new entry: all four original tables had SHA256
43401899f2017f08b0f33e6d9c818eb210f76476f5874f176b242f7d2dbcc725.
The first remains; every feature entry and unique description is preserved.

## Completed Parent Work

Capture UX #9917 merged to remote main8fce9f238ce89876dd363fb41b4ba1169a87d1b6
at2026-09-09T22:53:40Z. All445 capture/parity tests and protected checks passed.
The live test app is childPID61500, venv launcherPID30900, running in the separate
UpstreamDrift-capture-setup checkout. It is source-identical to the merged capture
code. Initial source launch needed PYTHONPATH pointing at that checkout's root,
src and src/shared/python. The oldPID50860 app exited. Preserve the current window.

Catalog/source PR #9919 merged2026-09-09T23:35:01Z atmain01831aa4c580ecb5065477c8219168064b629443.
Its original unit gate passed14929 tests but failed only generated divergence
inventory freshness. Concurrent remote commit ecab78b11 regenerated that inventory;
it was preserved by fast-forward and protected CI passed before merge. Auxiliary
manufactured authority/rolling jobs were still queued when inspected. Catalog and
source presence sessions were released; the issues remain open pending the bag/wizard.

## Remaining Goal and Coordination

- Everyday reference calibration #9897 (#9898-#9901), club bag #9902, wizard #9906
  (#9907-#9909) and fleet adoption remain active. Existing product/editing/drawing/
  overlay epics shipped through #9896. Gasification mapping is planned for future
  cheaper agents per user direction; do not implement that mapping now.
- Reference task #9914/#9918 owns headless fitting and a positive-length solver fix
  in reconstruct/model/fit.py. Do not edit that path without coordination.
- Current fitting models end at wrists/hands. #9914 owner explicitly confirmed no
  overlap with our session.py/write_fit changes. Their fit.py and new reference
  appearance/volumes/control files remain theirs. No invented club constraint.
- Tools #5136 numerical repair and #5140 moving-reference solver remain unmerged.
  Rust pre-checkout retry passed. Both rate shards in run34407390506 timed out at99%;
  Python3.11 leaves TestHoldFraction::test_matches_the_hand_counted_fixture unreported.
  Evidence is on Tools#5114; its prior Qt-cleanup candidate remains unqualified.
- Tools private Gasification checkout fails; user was asked via async input to have
  the Actions credential owner restore read access. Current App cannot inspect/update
  secrets (403). Never bypass the contract check or paste/mint a temporary secret.
- Fleet audit remains39/41. Context/Obsidian task owns replacements Tools#5138 and
  Gasification#4944 and agreed to sync the central agent-communication block in both
  AGENTS/CLAUDE. Authority537f9ad087dd3afdda60d28dd2e54d1ac7583864. Verify after merge.
- Goal stays active; no scientific accuracy/publication approval is implied by tests.

## Next Steps

1. Commit/publish the qualified #9905 change with normal hooks; merge only green protected CI.
2. Connect My Clubs to the goal wizard under #9906; retain #9905 open until that entry exists.
3. Continue everyday-reference calibration consumers and qualification after Tools gates clear.
4. Verify final Tools/Gas fleet policy replacements after their owner merges them.
5. Keep the full goal active; no unsupported equipment model constraint is claimed.

## Integrated Reference Fitting Handoff

- Repository/worktree: `D-sorganization/UpstreamDrift`, `../UpstreamDrift-reference-9914`.
- Branch: `feat/c3d-reference-overlay-9914`; baseline `7c09642df`; commit SELF.
- Governing epic: #9914; PR #9918; development entry DL-#9914.
- Complete: marker profiles, URDF and compiled-MJCF tree adapters, root seed,
  existing continuous fit orchestration, saved jobs and library assets,
  fixed placement estimator, standalone keyframe graphic, operator guide.
  Expanded scope adds measured club edges, shared 3D ellipsoid projection with
  adjustable alpha/radius, and saved reversible handedness before scene placement.
- Validation: RED observed before each new module; 177 combined reference/UI/solver tests pass;
  new two-camera renderer, preview, identity and custom-model contracts pass
  on combined main 18c8f922e. Normal pre-push gates pass on 74e867786.
  Exact commands are in the epic document.
- Evidence: Corrected positive-length model survey bundles under `../reference-fit-artifacts-9914`.
  Twenty corrected bundles verified; first driver fit was withdrawn for a negative length.
  Tracked survey evidence: docs/development/reference_fit_qualification.json. Native OpenSim adapter and MyoSuite anatomy
  are unavailable, explicitly recorded rather than replaced with a fallback.
- Display evidence: `reference_display_qualification.json` records twenty club
  assets derived from exactly unchanged qualified body coordinates. The external
  `reference-display-library.zip` contains assets and a reproduction script.
  Full reference/Capture Rig selection and targeted missing-club preview tests pass.
- Coordination: `codex-reference-9914-20260909`, lease and reference UI paths
  registered. #9917 integrated, concurrent branch fixes preserved in d874062a5.
- Merged: PR #9918 at6f2d63325f6260de99527a08551f7e116abdec28; owner completing documentation.

## Publication Checkpoint

Player bag implementation b206ae943 published through all normal hooks as draft
PR#9923. Current main merge retains #9918 reference/club/volume/handedness work,
its unique DL-#9914 entry and handoff evidence; generated inventories/maps were
rebuilt.3031-file LoD and architecture/doc budgets pass. Combined capture/model/
reference/map regression passes741 tests with2 normal deselections and5488 existing
warnings in515.50s. Keep the PR draft until the integration commit/push completes.
Local warm Qt smoke:1,000 catalog-backed clubs,3,466,962bytes, dialog156.88ms,
filter1.32ms. No production Rust or font change is justified by this measurement.

Tools#5136 main integration committed45f3bd8b9 after normal hooks;19 calibration
contracts/numerics and12 OpenCV5 numerical tests pass. Its normal push passed
(log TEMP/tools-5136-merged-push.log) and fresh CI is running. Tools#5140 merged
the same baseline locally;101 mocap/API and24 OpenCV5 checks pass. Its commit
hook caught a duplicate #5132 SPEC row from the merge; consolidate both meanings
into one row, then resume normal hooks.
Live Capture Rig PID61500 remains open and unchanged in capture-setup.

## Expanded Slow-Test Finding

The first combined command overrode addopts to obtain a summary, unintentionally
including the repository-marked slow real-log Simscape test. It exceeded the
unchanged60-second limit in test_shoulder_gimbal_and_strut_validate_on_the_real_logs
at test_simscape.py:120. A separate one-BLAS-thread diagnostic also timed out,
so thread oversubscription alone is not an established cause. No marker, timeout
or numerical tolerance changed. Logs: TEMP/player-bag-merged-tests.log and
TEMP/player-bag-simscape-thread-test.log. Standard repository selection passed741 tests (2 deselected)
with an explicit one-thread native budget and JUnit at
TEMP/player-bag-merged-standard.xml; do not claim that expanded run passed.

Wizard discovery: workflow.py already owns pure Step requirements/readiness and
SessionMedia rules. The simulation config SetupWizardViewModel serves a separate
canonical-core configuration contract. #9907 should add goal/dependency metadata
to the graph authority and reuse capture rules; #9908 should use standard Qt
Back/Next and existing action adapters. No wizard code has been written yet.

## Qualified Integration

The main merge preserves #9918 and all other owners.741 standard regressions pass;
3031-file LoD, architecture and document budgets pass. No source behavior was
changed to satisfy qualification. Normal integration commit/push is next, then
restore PR#9923 to ready and follow protected CI.
