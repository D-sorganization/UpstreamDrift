# Native Multi-Engine Matching Checkpoint

## Current Bound-Audit Correction (SELF)

Latest initial-state experiment SELF: retracted_tracking_9967_29.json uses a
clamped initial spline derivative equal to candidate qd0, with natural final
second derivative. Sensitivity maps use homogeneous initial-rate conditions;
TDD verifies the fixed derivative and finite-difference node sensitivity.
ControlTower eight evaluations yield 0.578439 mm marker RMS, exactly zero
initial-rate error, and initial combined Jacobian scaled discrepancy 4.20991e-6.
Dense acceleration closure remains 0.200508 and is NOT accepted. This replaces
the previous missing-initial-velocity limitation. Next enforce closure with
explicit scales or hard constraints while retaining marker objective and q0/qd0.

Latest marker-aware experiment SELF supersedes closure-only initialization:
retracted_tracking_9967_28.json uses run19 candidate attachments and capture
payload, checks node-time alignment and observation masks, and removes the
first chart from optimization. Nine evaluations take 0.89997 seconds, marker
RMS is 0.578684 mm, initial-pose error 2.78888e-13, and chart maximum 0.00266590.
The full combined initial Jacobian check has scaled error 8.84884e-6.
Dense acceleration closure is 0.118483: optimizer success does NOT qualify
this path. Closure/marker tradeoffs need explicit physical scales or hard
closure constraints. Initial velocity is explicitly not enforced yet.
Next implement initial-velocity conditions and scaled closure enforcement;
preserve marker masks and q0. No global torque or full-swing claim is made.

Latest experiment SELF: runner27 adds an explicit bounded least-squares option
and automatically audits 61 samples between nodes. Receipt
simscape_tour_matching/native_evidence/retracted_ls_9967_27.json records
100 function evaluations in 4.03187 seconds, chart maximum 0.00999886834,
dense pose 1.10441e-8, rate 7.98757e-7 and acceleration 4.64174e-5.
The evaluation budget was reached; this is not convergence or swing acceptance.
Least-squares minimizes closure only, unlike the displacement objective of
trust-constr; the results are feasibility experiments, not equal-objective
solver benchmarks. Next integrate marker tracking and initial-state conditions
before extending the path toward transition. Preserve the saved candidate.

This section supersedes the historical solver claims below. Run20 is terminal.
The old Jacobian comparison returned 59 chart components outside ±0.01, with
maximum 0.07306001724; it was not a valid bounded candidate. The finite-difference
candidate stayed within bounds (maximum 0.00483002077). Initial-point derivative
agreement does not establish trajectory feasibility or full-engine equivalence.

The runner now uses keep_feasible=True and independently rejects nonfinite or
out-of-bound returned charts. Its regression test failed before implementation
and passes afterward. New ControlTower evidence:
simscape_tour_matching/native_evidence/retracted_bounds_9967_26.json.
Forty iterations took 1.35135 seconds, returned chart maximum 0.00997851131,
and reduced node residual from 0.4876762 to 0.03435334. It is not converged.
A separate 61-sample spline audit found pose 4.84004e-6, rate 4.22302e-4,
and acceleration 0.03435334 maximum absolute residuals. Thus the path is not
qualified for torque identification. Runtime25 was reused without modification;
runner26 is distinct from historical scripts.

Next: include between-node closure and marker tracking in the path objective;
retain actual initial-state requirements, chart bounds and full forward gates.
Do not compare different horizons or two unfinished iterates as solver verdicts.
Validation: test_collocation_bound_guard.py passes; native receipt above is
physical-runtime evidence only for the four-node window. Full swing remains open.

Current override SELF: run20 is TERMINAL and independently rejected; no optimizer
is live. Exact final evidence and next action are at the top of the native
checkpoint. The turnover's Third Task is complete: the shared batch seam,
trusted worker adapter, and a frozen two-evaluation full solver assembly receipt
match exactly for workers=0 and workers=2 apart from timing telemetry. Read
`native_parallel_performance/batched-solver-qualification-9967-24.json` before
use. The four-node retracted collocation preflight is also complete and remains
unqualified: it preserves weld position by construction and reduces acceleration
closure from1.3366819 to0.1041854 after two bounded iterations. This commit adds
the exact cubic-spline chain rule from each retracted node chart into qd/qdd;
it has no new optimizer or physical-match claim. Next, qualify structured
node-level residual derivatives before any further bounded fit. The present
increment implements local centered q/v residual derivatives, exact
acceleration-J blocks, and their composition into a trust-constr Jacobian; it
has unit tests and a ControlTower two-iteration smoke receipt at
`simscape_tour_matching/native_evidence/retracted_collocation_jacobian_probe_9967_25.json`.
The residual is0.1490535 and remains unqualified; accepted chart bounds remain
±0.01 while a0.2 trial retraction radius prevents internal trust-constr probes
from crashing. A complete 84-column centered chart check now records maximum
absolute error7.90135e-6 and scaled relative error6.23290e-6; this qualifies
the local derivative assembly, not the fitted trajectory. Preserve older
live-job text as historical only.

The current matched ControlTower comparison is saved as
`native_evidence/retracted_collocation_compare_{fd,jac}_9967_25.json`. Both
start at0.4876762. With two evaluations, SciPy's internal differences reach
0.1041854, while the qualified supplied Jacobian reaches0.1490535. Diagnose
the trust-constr formulation before increasing its budget; retain the supplied
Jacobian as the validated fast derivative source.

- Worktree: C:/Users/diete/Repositories/Worktrees/UpstreamDrift-pinocchio-native.
- Branch: feat/9967-native-simscape-pinocchio; checkpoint SELF; PR not created.
- Issues: #9967, #10021, #10022; parent #9921; OpenSim planning #10003.
- Development entry: DL-#9967. MATLAB release remains R2025b.
- Current evidence and exact continuation: [Native Checkpoint](simscape_tour_matching/NATIVE_PORT_CHECKPOINT_20260911.md).
- Run19 is terminal and rejected; run20 session43223/PID2348439 is the sole
  optimizer on ControlTower. Never duplicate it or modify its runtime.
- Integrated reaction-identification feasibility and parallel-window benchmark
  receipts. Root reran six identification tests and two comparator tests: pass.
  Strict scalar-rate reconstruction still fails; full swing is not matched.
- Detailed executable next-agent plan: [Lower-Cost Agent Turnover](simscape_tour_matching/LOW_COST_AGENT_TURNOVER_20260912.md).
- Next: qualify the composed chart Jacobian against direct chart perturbations,
  then run only the four-node bounded jacobian-enabled probe and independently
  audit its trajectory. Optional
  window-executor module/tests are integrated from621efab7a; root reran five
  tests successfully. No other agent-owned worktree was edited.
- Preserved all earlier handoffs below. Archives retain raw identity-bearing
  bytes; formatted JSON is for review. Current commit introduces no active
  runtime or physics changes.

# Architecture Map Contract Handoff Checkpoint — 2026-09-10

- Worktree: C:/Users/diete/Repositories/UpstreamDrift
- Branch: docs/1616-c4-architecture-map; checkpoint SELF; PR pending.
- Governing issue: Repository_Management #1616 (Parent epic #1594).
- Objectives: Adopt Mermaid C4 architecture-map contract (C4Context, C4Container, Feature Map, Change Log) in UpstreamDrift.
- Verification: scripts/architecture_map_contract.py passes; pytest tests/test_architecture_map_contract.py passes 4/4; SPEC.md change log passes 642 rows.
- Preserved peer handoffs below.

# Impact Program Handoff Checkpoint — 2026-09-10

- Worktree: C:/Users/diete/Repositories/UpstreamDrift-impact-provider.
- Branch: docs/9700-impact-handoff; checkpoint SELF; ready PR #9962.
- Root handoff impact summary was compacted after CI found its 50 KB budget exceeded; detailed receipts remain linked.
- Governing epic #9700; this change updates turnover only. Main baseline:
  9c8afeaabf60f2751ebbd61b32dac98d32546c3e. Preserve peer capture work below.
- Provider PR #9916 merged as c487265f1ebc9c61a2e124267ad9cd1c96a6c007;
  follow-up #9920 merged as 08c8529ef78b9d7c336e0598721fb01bf814934d.
- Current main gitlink, requirements-tools.txt and Cargo.toml agree on Tools
  e83bd2e4a7a29a2dcd8145ef2d1efa07123324f0. The clean submodule was checked
  out at that existing pin after switching branches. No pin or runtime changed.
  Earlier PROVIDER_PIN_RESULTS.json is historical evidence for its recorded
  source; it does not qualify this newer main pin or the friction candidate.
- Tools normal contact #5146/#5149/#5152, prescribed loads #5154, complex FRF
  #5156 and calibration #5159 are merged. Friction #5162 is pending final
  integration/hosted qualification at this checkpoint. Coordinate the next
  combined pin with the context/capture owner, then qualify exact consumer
  contracts, built wheel and installed runtime. Do not alter installed CaptureRig.
- AffineDrift force-regularity #4356 merged as
  963867d7c78e544799ef4b6070eb1779e64c0452 after all 15 checks passed.
- Resume: inspect the live Tools #5162 and canonical HANDOFF, update local main
  without discarding work, read policy/context inventory, claim the next child
  issue and register presence. Preserve #8557 authority, reviewed claim outcomes,
  numerical ceilings and manufactured-data limits. No new solver qualification
  is claimed by this documentation-only checkpoint.

The program remains open. Next work must retain TDD, explicit contracts, shared
provider mechanics and independent reference controls. Prioritize event-resolved
force/work accuracy, general sliding/reversal/recontact, spatial and modal
convergence, measured shaft/grip/contact identification, calibrated structural
and acoustic transfer, then controlled blinded perception. Force spectrum,
radiated pressure and perceived sweetness are separate quantities. Synthetic
convergence cannot establish a player-dependent sound or heavy-hit effect.

## Preserved Capture Handoff

# Common-Reference Calibration Handoff

**User-requested transfer:** read [Capture Product Turnover](capture_product_turnover.md)
first. It supersedes older in-flight statements below and identifies the exact
remaining work, validation, branches and live application. Source and turnover
changes are being committed for the successor; do not start a duplicate epic.

Issue #8365 (claude, worktree `_issue_worktrees/UpstreamDrift-conductor-issue-8365`,
branch `conductor/issue-8365`, PR #9960, commit SELF): added
`src/tools/launch_monitor_model/launch_monitor_data.py`, importing the public
Launch-Monitor-Data shot and aggregate exports with declared units, corpus
identity and per-cell lineage, exported through the façade and registered
app-local in the ADR-0046 Stage 2 parity gate and ADR-0048. Validation:
`tests/unit/launch_monitor/test_launch_monitor_data_exports.py` 17 pass,
`test_canonical_layer_parity.py` 40 pass, ruff/mypy/pre-commit clean on changed
files. Workbench dialog wiring for the public exports is a follow-up. See DL-#8365.

Active #9899: `feat/9899-calibration-revision-status`, draft PR #9959,
builds on #9954 (which merged to main). Camera source hashing uses shared provenance;
command and wizard regression tests cover same-path changes, absent source,
matching/legacy outputs and downstream blocking. See
`capture_calibration_revision_delivery.md`. The numerical ruler provider is
Tools #5169 at 3d7beb203; its worker/editor integration is still outstanding.
Preserve the installed candidate runtime and the peer camera/impact work.

Active #9952/#9909: `feat/9952-camera-setup`, PR #9954, added the native camera
editor and guided acceptance (merged to main). All 566 Capture Rig and 30 isolated calibration
checks pass; searchable player examples were visually reviewed with the app theme.
See `camera_setup_delivery.md`. Runtime prerequisite #9950 merged as
9c8afeaabf60f2751ebbd61b32dac98d32546c3e.

Current #9949: `fix/9949-installed-capture`, PR #9950. Installed worker,
packaging, new-take identity and library indexing repairs remain intact. PR #9946
merged as126158943 before these installed-package repairs; installed startup,
wizard and29 calibration checks passed, as did pip check.

PR #9955 merged at32410babfd1e4741fa0c53bf05dd8403a51bf233 after all CI checks
passed, including actual Bioptim OCP tests and both manufactured authority checks.
Its scalar-bound API fix is now integrated into #9950 without changing numerical
bounds or tolerances. The only merge conflict was this handoff; both delivery
records are preserved here. Next: validate and publish the integrated #9950 head,
then advance native camera setup and guided acceptance in #9954.

See [active checkpoint](common_reference_calibration.md), DL-#9898 and the
[operator guide](../motion_capture/common_reference_calibration.md).

Implemented original-frame marking, revisions, isolated Tools solve, result
history/review, source hashes, distortion-preserving reconstruction/overlays and
shared native/generated help. Broad Python3.13 regression594 passed; history/Qt7,
projection/Qt25, OpenCV5 projection20 and atlas/parity49 passed. Production
Python3.12 scoped51 now passes after installing missing declared dependencies
and eliminating repeated PNG decode/player imports. The separate unchanged
Simscape real-log test still times out at180s; its broad Python3.12 run remains
failed. See the active checkpoint for exact logs. Native small layout fits640×560;
Calibrate Again and Add Another Placement remain visible. Main276998030 is
integrated; final provider qualification remains pending. No physical accuracy claim is made.

Tools #5140 merged0a561daff (tree identical to candidatec984, includes #5136).
Local vendor0a is development-only: it would regress the launcher compared with
main's interim4dabe900c. Context owner owns final gitlink/Cargo/pip/catalog alignment
after Tools #5144; impact owner qualifies the exact installed consumer. No bypass
of historical private404/rate-shard failures. Update final metadata and integrate
main normally before PR/CI/protected merge. Cross-capture reuse is now implemented with portable source evidence and native review;27 isolated checks and18 pipeline/lens/Qt boundary tests pass. Reuse committed f6298fb0c; types passed. Foreground image reads are now deferred to background/processing checks, with28 isolated tests passing. All push hooks passed on1e4984765. Named catalog selection passes13 library/dialog tests,29 isolated provider checks and19 integration tests; native archived-source selection/review/assignment at640×560 was inspected. Packaged-runtime and physical accuracy acceptance remain open.

Preserve live app54812 (`Capture Rig — main 56552f245`), older61500 and atlas2963.
They use frozen earlier checkouts, not this branch. Analysis owner controls
comparison/coaching/model work; context owner controls catalog/provider alignment.
Wizard #9931 merged56552f245; #9907/#9905 closed. Additional owners' delivery states
are preserved below. The canonical calibration checkpoint records visual frame
selection, Pan Image/Fit Image, their tests and native review. Catalog #9903/#9904
closed after87 contract/source/equipment tests; parent #9902 remains open for journey acceptance.

## Preserved Main Integration Context

# Shared Analysis Delivery Handoff

## Bounded Launcher Splash (#8360)

- Repository: D-sorganization/UpstreamDrift.
- Worktree: C:/Users/diete/Repositories/\_issue_worktrees/UpstreamDrift-conductor-issue-8360.
- Branch: conductor/issue-8360; record commit SELF. PR: not created.
- Issue: #8360 (splash stalls when an optional provider fails); development-log entry DL-#8360.
- Completed: `startup_phases.py` (pure-Python `StartupTimeline`, `run_bounded`,
  closed outcome/category sets, `probe_tools_provider` resolving Tools through
  `tools_repo_path.resolve_tools_repo` and checking the Rate standalone entry
  point `src/rate_of_closure/launch_pyqt6.py` without importing it);
  `AsyncStartupWorker` runs registry/engines/Docker/Tools-provider phases under
  explicit timeouts and only the registry is required; `StartupSession` owns the
  splash <-> worker <-> shell handshake with a `STARTUP_TIMEOUT_SEC` watchdog,
  worker-generation isolation and `sip.isdeleted` guards; `StartupFailureDialog`
  offers Retry / Continue without provider / Copy diagnostics / Close;
  `UpstreamDriftLauncher(loading=True)` no longer lazy-loads the registry and
  engine manager on the GUI thread; `LauncherOrchestrator` treats supplied
  results as authoritative. `STARTUP_TIMEOUT_SEC` moved to `launcher_constants`
  (re-exported unchanged from `upstream_drift_launcher`).
- Validation: `tests/unit/launchers/test_startup_phases.py` (30 pass),
  `tests/launchers/test_startup_session.py` (23 pass), orchestrator/startup/
  launcher suites pass except pre-existing unrelated failures
  (`test_golf_launcher_startup_timeout.py` patches a non-existent
  `_lazy_load_model_registry` attribute; theme-colour assertions; vendored
  theme/API import mismatches). `pre-commit run --files`, ruff, error-handling
  ratchet and file-size budget pass. mypy was run on the changed modules.
- Assumptions: the vendored `vendor/ud-tools` submodule was initialized at the
  pinned SHA to run tests; no tracked content changed.
- Next: open the PR with `Fixes #8360`; run the clean-checkout Windows smoke
  matrix (editable-Tools, vendored-Tools, absent-Tools, broken-Rate) and record
  the outcome in DL-#8360.

## Completed Runtime (#9926)

- Repository: D-sorganization/UpstreamDrift.
- Worktree: C:/Users/diete/Repositories/UpstreamDrift-analysis-metrics-9942.
- Branch: docs/9926-analysis-completion; record commit SELF.
- Owner session: codex-unified-metrics-9942; preserve peer worktrees and Tools pin4dabe900.
- Foundation #9933 merged at f04aa1a570e64c3db0b3d009351ab222656175b2.
- Comparison drawings #9943 merged at 3fbd4b2da5f8661ec1f1e91b97884f781d83f1fd.
- Metrics, Trace and native references #9945 merged at
  2d41aba4159162f91c0cd1cc6919d0341755b492 on 2026-09-10T05:20:06Z.
  Required quality-gate passed in CI Standard run34440058289; the run succeeded.
  Children #9929/#9930/#9932/#9942 are closed. Optional queued jobs are not passing evidence.
- Runtime includes common drawing/geometry/measurement controls, explicit model
  placement and handedness, club/ellipsoid appearance, Trace marker import and
  Pose Studio reference meshes. See [ledger](unified_analysis_9926.md) and
  [reviewed contracts](../architecture/SHARED_ANALYSIS_CONTRACTS.md).
- Qualification: all20 verified Tour Average assets passed actual shared analysis,
  save/reopen and export; native30 and Trace17 tests passed. After integrating
  main300d96a1 and biomechanics helper f58ee5b3, all88 combined tests passed.
  Earlier broad suites, source typing, Ruff, LoD, budgets and generated-map
  evidence are recorded in the ledger. No physical camera or anatomical accuracy
  qualification is implied; unsupported native viewers and marker gaps are explicit.
- External reproducible visual evidence: analysis-9926-artifacts/catalog-analysis-zcgqk8ta
  and qualify_catalog_analysis.py; native-references-saep77qb and trace-appearance-l276j6ua.
- Repair47bd32c11 was pushed after #9945 merged, so this branch preserves it via
  417fd83ae: restored DL-#9926, shipped DL-#9907 and lost SPEC#9931/#9932/#9933
  without modifying peer #9934 records. This closure changes documentation only.
- Next: merge this final documentation record through normal protected CI, then
  verify epic #9926 closure. No runtime implementation remains. The independently
  owned #9915 catalog may index current contracts later; it is not a runtime dependency.

## Preserved Wizard Context

## Identity

- Repository: D-sorganization/UpstreamDrift.
- Worktree: C:/Users/diete/Repositories/Worktrees/UpstreamDrift-capture-wizard.
- Branch: feat/9907-capture-goal-wizard; base fd9434ce9b98a008741b2afca62f8654fd34c6e1.
- Issues: #9907/#9908; epic #9906; bag completion entry #9905.
- Session: capture-product-01a08427-goal-wizard; presence through02:51UTC, leases02:22/02:50UTC.
- Implementation455022ede integrates main c487265f1 via a74d5ebd2; Tools pin4dabe900c.
  PR #9931 is draft while remote CI runs. Normal commit/push hooks passed.

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
No production font override was introduced. Integrated main regression passes
600 checks,6113 existing warnings in149.75s; TEMP/capture-wizard-main-tests.log/XML.
The integrated3032-file LoD scan passes with490 baseline occurrences and60
reductions; architecture/document budgets pass. Chromium verified empty-selection
feedback and an actual capture-plan.json download with the selected drawing goal.
Remote CI qualification remains.

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

## Preserved Incoming Main Handoff

The following text records the incoming provider and earlier capture checkpoints.
Their current-status wording describes those checkpoints; the wizard state above
is authoritative for this branch. Preserve their source and qualification details.

# Impact Shaft Provider Integration (#9912)

## Current Continuation State

Concurrent eacd69858 is preserved, including player/equipment main 5ada5e6a6.
An inherited inventory mismatch was reproduced by the existing test; regeneration
from the actual pinned checkout with full authorship restores the omitted
shaft/provider entries. All 271 integration controls pass. Provider source,
Cargo/gitlink and seam repairs are unchanged; #5144 reviewed-pin qualification
remains pending. Evidence is in PROVIDER_PIN_RESULTS.json.

Main 90c3d0b77 brings reviewed reference fitting/display #9918/#9922 and club
catalog #9919. Six conflicts are documentation/inventory only; both task
scopes are retained and the inventory is regenerated from the real trees.
All 271 source controls pass after this merge. Tools numerical PR #5133
was already squash-merged as 2c9a8d6c; launcher correction #5143 is a separate
follow-up. Candidate 4dabe900c and the 32a8b36ec wheel remain accurately
identified interim evidence, not that future reviewed revision.

The latest candidate is Tools 4dabe900c6ef7767b565c778cda9d9449bed28cf. Six protected unit-gate
failures at d2760f05c are repaired locally: Cargo/gitlink consistency, current
inventory metadata, immutable catalog provenance, obsolete color waiver,
explicit pre-existing realtime migration debt and the moved launcher target.
The corrected manifest exposed a real UI import failure. Two new tests fail
before retiring four byte-identical UI copies through the existing namespace
resolver. All 27 UD-only widgets remain; 105 UI controls and real offscreen
widget construction pass. Before that UI repair, all 106 migration controls
passed. Exact new-pin combined source suite passes 271 tests in 71.66 s (20 existing deprecation warnings); the isolated 32a8b36ec wheel now passes provider, namespace and Qt construction checks with gui-tools installed. Its web assets are omitted.
Realtime #8942 remains pending under a deliberate review exception expiring
2026-10-09; no guard logic was weakened and no migration completion is claimed.
Earlier source/wheel records below retain their actual revision identities.

Concurrent remote repairs through 7141feb79 are preserved by normal merge.
The same provider pin and theme retirement are retained. Reviewed main
use_start_file preserves the concurrent process-action fix without adding
a second equivalent API. Realtime is split/pending-cleanup because both trees
contain different transports and UD retains a distinct facade. Source and
installed-wheel evidence keeps its original revision identity.

Main 8fce9f238 (capture PR #9917) is integrated with its reviewed LoD fixes.
Only root handoff and development-log conflicts needed resolution; both scopes
are retained. All 79 provider/theme/fallback/manual checks pass again after the merge.

Current repair: 79 provider/theme/fallback/manual tests pass against 00d17e7f9,
with eight existing deprecation warnings. The seam gate passes after actual
retirement of the already-shadowed color child and an explicit realtime
split/pending-cleanup ruling linked to #8942. Its facade remains local.
No byte-identity or private non-string equivalence is claimed for the removed
color child. The 608e85b24 wheel below is historical. The current 00d17e7f9 pin was built
from efe44846e and installed into a fresh environment with all core dependencies.
Shaft wire/digest/tamper and theme/layout/realtime import checks pass under
site-packages; pip check passes. Wheel SHA256:
8e09cd2c6e6c6ddb008c0bff9839514aa3e07f7c95f031da629bbe34ad4c3ce8.
This is Python-provider evidence; UI assets and physical qualification remain
excluded. Source checks after capture merge pass all 79 tests; development-log
audit retains 36 inherited findings against main's 37, with no new findings
and the exact union of both parents' SPEC issue rows.
Root handoff again carries required UP-D0/UP-D1 references. The PR title now
accurately uses chore for dependency qualification. Provider CI is still open.

- Working directory: C:/Users/diete/Repositories/UpstreamDrift-impact-provider-pin.
- Branch: feat/9912-impact-provider-pin; implementation commit b6107f8e2; PR #9916.
- Original base: 6e3610a9b; incoming main: 18c8f922e; Tools candidate: 00d17e7f91fe8541bc8882ee745fda58ee2ad7af.
- Governing issue #9912, development entry DL-#9912, parent #9703/#9701/#9700.
- Pair: Tools #5133. Canonical shared source stays in Tools; the obsolete theme color child is retired.

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

# C3D Reference Fitting Handoff

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

## Verified Agent Context: #9915

- Identity: UpstreamDrift, `C:/Users/diete/Repositories/.context-implementation/UpstreamDrift`; branch `feat/issue-9915-agent-context`; commit `SELF`; PR #9920; DL-#9915; session `context-01a0879e-ud`; fleet epic #1629.
- Implemented: twelve components, five reviewed integration contracts, twelve navigation tasks, generated module graph/offline browser and required source/review/provider checks. Existing atlas, scientific authorities and peer handoffs are preserved.
- Published provider: Tools #5141 merged as `e83bd2e4a7a29a2dcd8145ef2d1efa07123324f0`, byte-identical to qualified34b142e28. Both Linux aggregates and quality passed. The gitlink, Cargo.toml and requirements-tools.txt now use this same revision. The new required installation regression failed against the old v1.15.0 wheel and passes after alignment.
- Source integration: main276998030 is integrated. Its registry changes invalidated two atlas reviews; inspected shared-analysis contracts, preserved pixel/frame/scene limits and updated both reviews after31 atlas/workflow/goal tests passed. Existing documentation/CI regression repairs remain unchanged.
- Final-pin validation:107 provider/catalog/seam tests, six context tests and twelve navigation tasks pass (median2423.645ms, max15593 characters). Source/view checks and all three pin surfaces pass. Full-authority divergence inventory regenerated. The source graph verifies3543 files/39589 symbols and two MotionPipeline caller candidates. Tools wheel builds and installed CodeMap verifies the same graph in an isolated interpreter; candidate/squash source trees are identical.
- Concurrent integration: remote60318e4c4 selects the same provider and adds the atlas-required NumPy dependency to the context CI environment. The minimal validation environment reproduces successful atlas/source checks after that dependency is installed. Retained the more specific reviewed contracts and regenerated views; all application source files remain identical to wheel source6dab98fce.
- Installed consumer: normal Python3.12 wheel build at6dab98fce includes the frontend bundle; SHA256 `8f5b62979229ef0c2372ea9080cc0a34a51105604bf170c94828f5e6f0d06e42`. In an isolated declared gui-tools environment, the actual manifest resolves and constructs FunctionGeneratorWidget, installed aliases/origins pass, synthetic shaft round-trip and altered-source rejection pass, and frontend index presence is verified. All31 unmodified calibration/reference tests pass against installed modules with OpenCV5.0.0.93; pip check passes. Later merge changes only CI/docs, so wheel source and provider are unchanged. Local exact records: `.context-implementation/ud-e83-installed-qualification.json` and `ud-e83-installed-calibration.xml`.
- Limits: protected application CI remains pending. Lexical graphs are scoped navigation evidence; scientific/physical approval is separate. No subscription is needed. No peer-owned worktree was modified.
- Merge metadata: the concurrent merge retained two versions of our own #9920 SPEC row; CI rejected the duplicate. Kept the complete current row and preserved all peer rows; concurrent equivalent repair4ded98ad6 is integrated and the unchanged SPEC checker passes.
- Documentation budget: shortened only the context log entry; peer records and the51,200-byte limit are preserved.
- Next: publish through normal hooks, require green protected CI, then reconcile the fleet delivery audit and epic.

# Unified Biomechanics Analysis (#9934)

## Current Continuation State

Branch `feat/9934-biomechanical-analysis` is implementing the shared
calibrated trajectory contract, source-qualified orientation conversion,
Cheetham-labelled golf channels, body/club COM, API conversion/compute/display
routes, and desktop/web plot explorers. Focused biomechanics/API/display tests
pass; Ruff lint and formatting pass after formatting. The work is not yet
qualified for merge: native model transform adapters and full repository gates
remain. Missing anatomy, mass, calibration or engine capability must remain
unavailable rather than being inferred.

- Epic: #9934; child issues: #9935–#9939.
- Branch: `feat/9934-biomechanical-analysis`; base: `c487265f1`.
- API: `/api/biomechanics/compute`, `/display`, `/convert`, `/bindings`, `/results`.
- Next step: add/qualify native transform adapters, then run full CI gates and open PR.
