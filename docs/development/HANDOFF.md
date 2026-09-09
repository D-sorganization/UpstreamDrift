# Player Bag and Capture Equipment Handoff

## Identity

- Repository: D-sorganization/UpstreamDrift.
- Working directory: C:/Users/diete/Repositories/Worktrees/UpstreamDrift-player-bag.
- Branch: feat/9905-player-club-bag; baseecab78b11 (catalog PR #9919).
- Implementation: uncommitted working tree; PR not created.
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
required under #9906 before #9905 closes. No implementation commit yet.

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
