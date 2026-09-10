# Guided Capture Workflow Handoff

## Identity

- Repository: D-sorganization/UpstreamDrift.
- Worktree: C:/Users/diete/Repositories/Worktrees/UpstreamDrift-capture-wizard.
- Branch: feat/9907-capture-goal-wizard; base fd9434ce9b98a008741b2afca62f8654fd34c6e1.
- Issues: #9907/#9908; epic #9906; bag completion entry #9905.
- Session: capture-product-01a08427-goal-wizard; presence through02:51UTC, leases02:22/02:50UTC.
- Implementation is uncommitted; no wizard PR yet. See DL-#9907.

## Current Work

The existing capability_connections.json now owns typed executable goal metadata,
separate from architecture data-flow edges. The standard-library DAG planner
rejects cycles, unknown bindings and incompatible camera routes, deduplicates
shared prerequisites and rejects changed capture/input/graph resumes. The atlas
renders goal choices, downloads validated portable plan JSON and generates a
separate capture-goals.mmd from the same source.

Capture Wizard is reachable in the app header. Standard Qt Classic Back/Next/
Finish/Cancel keeps navigation inside the window; Aero clipped the smaller layout
and was replaced after visual review. Existing Library, Edit Swing, Drawing,
Expert Library/Comparison, My Clubs, Calibration and workflow controls are reused.
Opening a step never starts a hidden analysis job. Status inspection runs in a
single standard-library worker; Qt updates are polled on the UI thread. Bookmarks
are capture-owned, atomic and preserve malformed existing files. Optional skips
persist; lens confirmation is renewed after closing the wizard.

The existing workflow.evaluate remains the readiness authority. Supplemental
checks inspect saved edits, explicit detector/edit association, model provenance,
reference bindings, drawings and equipment snapshots. The calibration profile
consumer rechecks intrinsic quality, camera IDs and recorded dimensions. No
physical accuracy claim is made. Guided model paths presently use the default
triangulated match/all views/default observations; unsupported advanced selections
receive an explicit explanation. Named variants/image-space matching remain in
the existing controls. User guide: docs/motion_capture/capture_wizard.md.

## Validation

Initial TDD failed on missing planner/catalog/wizard modules, then implementation
passed. Planner/catalog15, Qt navigation5, evidence/calibration34 and real host
navigation/resume4 checks pass in their recorded focused runs. Existing detector
activity tests remain compatible. Scoped mypy with the repository hook's
--follow-imports=silent passes9 source files. An expanded import-following run
found8 unrelated pre-existing dependency errors plus one local variable type
error; the local error was repaired. Full Ruff lint/format (6917 files), architecture and document budgets pass.
The3037-file LoD scan passes with490 baseline occurrences and60 reductions;
20 new chains were removed through component methods/local delegates, no waivers.
Broad tests/tools/capture_rig + parity + atlas regressions:510 passed,5893 existing
warnings in149.17s; TEMP/capture-wizard-regression.log and XML. First invocation had an invalid PowerShell
JUnit argument and did not run tests; the corrected run passed.
Visual evidence: TEMP/capture-wizard-visual-iowk4w5o/\*-classic.png,760x610 and
660x560, inspected after loading Segoe UI for the Windows offscreen environment.
No production font override was introduced. Final docs/CI qualification remains.

## Completed Parents and Remaining Goal

- Capture UX #9913/#9917 merged8fce9f238; club catalog #9919 merged01831aa4c.
- Player bag #9923 merged5ada5e6a6827bfb94a00d1801f6b475afb885885 at2026-09-10T00:19:44Z.
  Its merged implementation is byte-identical to the741-test qualified source.
  A concurrent remote branch rewrite was reconciled by normal merge/push f7e2f27af,
  preserving completion docs. No force-push. #9905 remains open for wizard entry.
- Reference #9918 merged6f2d63325; completion docs #9922 merged90c3d0b77.
  Do not edit reference fit.py/appearance/volumes without its owner's coordination.
- Tools numerical PR5136 head45f3bd8b9 and moving-reference PR5140 headc98402cb1
  include main2c9a8d6c9 and passed normal commit/push hooks. Qualification:19/12
  numerical tests and101/24 moving-reference/API/OpenCV5 tests. Private Gasification
  checkout still fails; rate shards are running. User has been asked to have the
  Actions credential owner restore private-repo access; no response yet. Do not
  bypass checks or mint/store a temporary secret. Claims through01:49/02:03UTC.
- Everyday-reference calibration #9897/#9898-9901 remains open, including geometry,
  repeatable player UI and hardware qualification. Club epic9902 and wizard9906
  must close only after remaining acceptance and remote merges.
- Fleet adoption39/41 remains pending owner replacements Tools5138/Gas4944. The
  Obsidian/context task owns those changes and agreed to sync canonical blocks in
  AGENTS and CLAUDE. Do not duplicate its work. Gas mapping planning is complete;
  implementation is deferred to future cheaper agents per user instruction.
- Preserve live Capture Rig PID61500 and launcher30900, source capture-setup501092b27
  (matches9917), runtime TEMP/upstreamdrift-capture-test-runtime. It does not include
  the new bag or wizard. Do not kill or edit that live source while the user tests.
- Automatic approval review rejected deleting a temporary clean-export folder
  with 'blocked by policy'; it remains in place. Do not retry by another route.

## Next Steps

Finish broad regressions, fresh map/inventory/docs and visual qualification. Commit
and push normally, open a focused PR, update SPEC to its actual PR number, follow
protected CI and merge only when qualified. Connect #9905 closure to this PR. Keep
all remaining calibration/fleet work active. The historical parent records below
are retained for their unique qualification evidence, not current wizard status.

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
Back/Next and existing action adapters. That parent checkpoint preceded the wizard implementation above.

## Qualified Integration

The main merge preserves #9918 and all other owners.741 standard regressions pass;
3031-file LoD, architecture and document budgets pass. No source behavior was
changed to satisfy qualification. Normal integration commit/push is next, then
restore PR#9923 to ready and follow protected CI.
