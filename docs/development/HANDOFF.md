# Current Matching Continuation Handoff

## ORG-06 Apply the Same Workspace Navigation to React and Tauri (#10516)

- Worktree: main checkout, branch `feat/issue-10516-org06-react-workspace-navigation`, lease `antigravity-ud-10516`, DL-#10516.
- Changes:
  - `workspaceNavigation.ts`: Authoritative shared catalog definitions for the five primary workspaces (`Capture & Analyze`, `Model & Match`, `Shot & Course Lab`, `Optimize & Train`, `Results & Compare`) and secondary navigation (`Developer & Research`, `All Tools`, `Favorites`, `History`).
  - `capabilityAdapter.ts`: `resolveWorkspaceToolAction` handles in-app web routes, desktop-window launches under Tauri, and actionable explanations with web alternatives for browser users.
  - `WorkspaceNavigation.tsx`: Accessible `WorkspaceSidebar` with visible focus and focus recovery to `#main-content`, `WorkspaceBreadcrumb`, and `WorkspaceView` listing workspace member tools.
  - `WorkspacePage.tsx`: Integrates `WorkspaceShell` with `WorkspaceSidebar` and `WorkspaceView` for bookmarkable task URLs (`/workspaces/:slug`).
  - `LauncherDashboard.tsx`: Added task workspaces bar linking directly to each workspace destination.
  - `routeTitles.ts`: Centralized page titles for all `/workspaces/:slug` routes.
- Reproduction: `npm run test:run -- src/components/layout/WorkspaceNavigation.test.tsx` (from `ui/`).
- Next: Open PR referencing Fixes #10516, enable auto-merge, release lease.

## ORG-05 Build Task-Oriented Desktop Navigation Over Existing Embedded Tools (#10515)

- Worktree: main checkout, branch `feat/issue-10515-org05-desktop-navigation`, lease `antigravity-ud-10515`, DL-#10515.
- Changes:
  - `workspace_navigation.py`: Defined 5 primary task workspaces (`Capture & Analyze`, `Model & Match`, `Shot & Course Lab`, `Optimize & Train`, `Results & Compare`) and secondary navigation (`Developer & Research`, `All Tools`, `Favorites`, `History`); implemented `ALIAS_MAP` canonical migration utilities (`migrate_model_order`, `migrate_favorites`, `migrate_saved_layout`); implemented single-instance tool reuse policy (`find_existing_tool_tab`, `focus_or_open_tool_tab`); guarded dirty tab close (`close_tool_tab_guarded`); created accessible return-to-workspace breadcrumb bar (`WorkspaceBreadcrumbBar`); implemented discoverability status explanations (`explain_tool_status`).
  - `launcher_layout_manager.py`: Integrated `migrate_saved_layout` in `load_layout()` preserving user tile scaling, view mode, and dock state; updated `get_filtered_order()` to support workspace filters.
  - `_launcher_navigation_ui.py`: Updated `_build_sidebar_filter_buttons()` with workspace destinations; added workspace routing to `_on_sidebar_routed()`; added `navigate_to_workspace()`.
  - `launcher_ui_setup.py`: Enforced single-instance tool reuse in `dock_widget_as_tab()` before adding duplicate tabs.
  - `test_workspace_navigation.py`: 22 comprehensive unit tests covering all requirements.
- Reproduction: `pytest tests/launchers/test_workspace_navigation.py tests/launchers/test_workspace_tabs.py tests/launchers/test_launcher_layout_manager.py tests/launchers/test_launcher_ui_setup.py -v`.
- Next: Open PR referencing Fixes #10515, enable auto-merge, release lease.

## ORG-17 Replace Canonical Estimation Shell With Bounded Estimator Coordinator (#10526)

- Branch: `feat/issue-10526-org17-estimation-workflow`, lease `antigravity-org17`, DL-#10526.
- Changes:
  - `src/shared/python/workspace/estimation_workspace.py`: Implemented `EstimationWorkspaceCoordinator`, `EstimationRunConfig`, `EstimationRunResult`, `ParameterPriorConfig`, and `IdentifiabilityGateConfig`.
  - Wires `solve_single_trial_map`, `IdentifiabilityGateOptions`, parameter priors and bounds, and spline trajectory evaluation into an application service.
  - Provides fail-closed validation for non-finite cost/trajectories and ill-conditioned systems, with complete provenance persistence round-trips.
  - `src/shared/python/workspace/__init__.py`: Exported coordinator and dataclasses.
  - `tests/integration/test_estimation_workspace.py`: 8 integration tests validating capability availability, synthetic parameter recovery, prior regularization, identifiability gating, non-finite cost gating, parameter bounds enforcement, and run provenance serialization roundtrips.
- Reproduction: `pytest tests/integration/test_estimation_workspace.py --timeout=60`.
- Next: PR auto-merge and release lease.

## ORG-03 Replace Misleading Launches With Real Tasks or Explicit Nonlaunchable States (#10512)

- Branch: `feat/issue-10512-org03-launch-truthfulness`, lease `issue-10512`, DL-#10512.
- Changes:
  - `src/launchers/task_launch_truthfulness.py`: Authoritative launch disposition audit table (`LaunchDisposition`) covering production solvers, prototype demos, library-only algorithms, parametric CLIs, provider-required tools, and service previews. Implemented pre-flight parameter validation contract for FreeMoCap sidecar runner (`launch_freemocap_with_validation`), rejecting zero-arg headless launches and dialog cancellations without spawning subprocesses.
  - `src/launchers/launcher_model_handlers.py`: Guarded `SpecialAppHandler` to reject direct launches of library-only components (`swing_optimizer`, `injury_analysis`) and parametric CLIs (`motion_capture`), exposing actionable `status_message()` with tracking issue references and remediation. Marked `GolfSimulationSuiteHandler` as prototype demo with truthful status disclosure.
  - `src/launchers/external_tools_adapter.py`: Removed blank placeholder `VideoAnalyzerWindow` fallback from `_import_video_analyzer()`, allowing missing provider / import errors to surface through the shared `_UnavailableToolWindow` diagnostic.
  - `src/launchers/launcher_process_manager.py`: Added `launch_script_with_args()` and safe argument forwarding in `launch_script()`. Hardened `_assign_to_job()` on Windows to safely handle non-integer or mock process PIDs without crashing.
  - `src/config/launcher_manifest.json` & `src/config/models.yaml`: Updated metadata and status chips for `golf_simulation_suite` (`prototype`), `swing_optimizer` (`experimental`/library), `injury_analysis` (`experimental`/library), removing qualified engine claims.
  - `ui/public/capability-atlas/`: Regenerated `graph.json` and `index.html` via `python -m scripts.generate_capability_atlas`.
  - `tests/launchers/test_task_launch_truthfulness.py`: TDD acceptance suite verifying all RED/GREEN cases: problematic tiles cannot claim ready/gui_ready; missing video analyzer yields diagnostic window; FreeMoCap zero args/cancellation performs no spawn and valid args pass through unchanged; simulator prototype is marked demo-only.
- Reproduction: `pytest tests/launchers/test_task_launch_truthfulness.py tests/launchers/test_simulation_guis.py tests/launchers/test_launcher_process_manager.py tests/scripts/test_capability_atlas.py --timeout=60`.
- Next: Validate against rebased main, enable auto-merge on #10536, release lease on #10512, claim #10513 (ORG-04).

## ORG-02 Separate Capability Identity, Maturity, Availability, and Qualification (#10511)

- Branch: `feat/issue-10511-org02-capability-state-contract`, lease `antigravity-ud-10511`, DL-#10511.
- Changes:
  - `capability_state.py`: Implemented orthogonal typed models: `SurfaceAvailability` (with DbC invariant demanding non-empty reason and remediation when unavailable), `CapabilityAvailability` (desktop, web, api, cli), and `CapabilityQualification` (with status, is_qualified, receipt, failure reasons). Implemented `adapt_engine_matrix_qualification` consuming #10351 engine matrix contract (non-engine tools are exempt). Implemented `RuntimeProbeKey` and `RuntimeProbeCache` for lazy cached probing tied to provider pin / runtime identity. Established `CANONICAL_TILE_DISPLAY_NAMES` and `resolve_canonical_display_name`.
  - `launcher_manifest_loader.py`: Extended `LauncherTile` with `maturity`, `availability`, and `qualification` fields. Implemented `_build_default_availability` deriving surface states from provider and web contracts. Updated `LauncherTile.to_dict()` to serialize orthogonal fields while maintaining backward compatibility for legacy `status`. Enforced that unqualified engines never claim `ready` or `stable`.
  - `models.yaml`: Synchronized canonical display names for `matlab_suite` ("Matlab Models") and `golf_simulation_suite` ("Golf Simulation Suite").
- Reproduction: `pytest tests/config/launcher_manifest/test_capability_state_contract.py tests/config/launcher_manifest/ --timeout=60` and `pytest tests/config/test_launcher_registry_parity.py --timeout=60`.
- Next: Open PR, arm auto-merge, release lease.

## ORG-01 Baseline Every Capability and Preserve Tile, Layout, and Artifact Identity (#10510)

- Worktree: main checkout, branch `feat/issue-10510-org01-capability-baseline`, lease `antigravity-ud-10510`, DL-#10510.
- Changes:
  - `src/config/capability_migration.py`: Machine-checkable capability migration data model and validation engine (`CapabilityMigrationInventory`, `MigrationEntry`, `FixtureEntry`). Enforces uniqueness, acyclic alias resolution, primary workspace exclusivity, provider availability decoupling, saved layout validation, and fixture hash verification.
  - `src/config/capability_migration.json`: Complete canonical baseline inventory cataloging all 104 launcher tiles/models (61 base desktop models + 29 provider models + 14 web catalog tiles), 45 feature parity contracts, 9 excluded tool packages, and 15 golden preservation fixtures with SHA-256 integrity hashes.
  - `scripts/generate_capability_baseline.py`: Markdown generator and freshness validator for `docs/development/ORG01_CAPABILITY_BASELINE.md`.
  - `docs/development/ORG01_CAPABILITY_BASELINE.md`: Architecture decision document recording baseline inventory, workspace domains, alias resolution graph, and preservation of ADR-0047 viewer identity and provider seams.
  - `tests/config/test_capability_migration_coverage.py`: 18 RED and GREEN regression tests covering unclassified local/provider/manifest/CLI detection, alias cycle detection, duplicate identity prevention, missing provenance failure, legacy alias resolution, and fixture hash preservation.
  - `scripts/capability_atlas/render.py` & `docs/architecture/CAPABILITY_ATLAS.md`: Linked ORG-01 baseline from the generated capability atlas.
- Reproduction: `pytest tests/config/test_capability_migration_coverage.py --timeout=60` and `python -m scripts.generate_capability_baseline --check`.
- Next: PR auto-merge, complete lease on #10510, proceed to downstream issues in Epic #10508.

## ORG-13 Results Workspace Handoff (#10521)

- Branch: `feat/issue-10521-org13-results-workspace-handoff`, lease `antigravity-ud-10521`, DL-#10521.
- Changes:
  - `results_workspace.py`: Implemented `ResultsWorkspaceCoordinator`, `ResultArtifactItem`, `ResultCategory`, `WorkspaceActionType`, `ActionAvailability`, `ComparisonResult`, `MissingAssetDiagnosticError`, `UnitMismatchDiagnosticError`, and `HandoffDispatchPayload`.
  - Consumed public contract of #10353 (`MatchedSwingBrowserModel` / `MatchedSwingFilter`) and canonical `ResultsBrowser` indexing, without creating a competing browser.
  - Enforced selected run isolation in tool handoffs, preventing fallback to global active sessions.
  - Implemented artifact categorization (source data, processed recipes, kinematic replay, dynamic run, measurements, flight trajectory, qualification verdict).
  - Validated missing assets and unit mismatches during run comparison (never guessing similarly named files or silently comparing mismatched units).
  - Verified provenance retention (#8820) across export and reimport round-trips.
- Reproduction: `python3 -m pytest tests/integration/test_results_workspace_handoff.py tests/tools/matched_swing_browser/test_model.py --timeout=60`.
- Next: PR auto-merge and release lease.

## Docker Audit Repair (#10472)

The Docker image now pins Tornado 6.5.8, matching the generated runtime and
development locks and the declared runtime floor. This resolves the in-image
pip-audit findings GHSA-wwv5-g3v4-889x and GHSA-8423-8fgw-73vq (plus the third
Tornado 6.5.7 advisory) without an audit waiver. CI must confirm the complete
Docker build and dependency-artifact regeneration before merge.

## ORG-04 Validate Every Browser, Tauri, and Native Launch Destination (#10513)

- Branch: `feat/issue-10513-org04-launch-destinations`, lease `antigravity-ud-10513`, DL-#10513.
- Changes:
  - `ui/src/routes.ts`: Declared canonical `KNOWN_APP_ROUTES` array and `isKnownAppRoute(route: string): boolean` helper.
  - `ui/src/api/webLaunch.ts`: Updated `resolveTileLaunchAction` to reject unknown/unregistered routes with `'Unavailable'` badge and reason instead of silently navigating to 404.
  - `ui/src/api/launcherReachability.ts`: Implemented `evaluateTileReachability` and `generateReachabilityMatrix` covering browser and Tauri contexts.
  - `ui/src/api/launcherReachability.test.tsx`: Comprehensive Vitest suite for reachability matrix and unknown route rejection.
  - `src/config/launcher_manifest_loader.py`: Sanitized `web_route` for Movement Optimizer in `_build_provider_tile`, deriving `native-window` cleanly.
  - `src/shared/python/movement_optimizer/model_pack.yaml`: Removed unmapped `web_route: "/tools/movement-optimizer"`.
  - `tests/config/launcher_manifest/test_parity.py`: Added checks for loaded tiles in `test_route_mode_routes_exist_in_react_router`, and added `test_every_tile_destination_resolves_authoritatively`.
- Reproduction: `npm run test:run` in `ui/` (874 tests pass); `pytest tests/config/launcher_manifest/ tests/config/test_launcher_registry_parity.py` (95 tests pass).
- Next: Land PR, arm auto-merge, release lease.

=======

> > > > > > > bf6ec7343 (feat(ui): apply task-oriented workspace navigation to React and Tauri (#10516))

## MV-06 Expose Real Forces, Torques, and Explicit Counterfactual Semantics (#10482)

- Worktree: `UpstreamDrift-10482-forces`, branch `feat/10482-forces-torques-counterfactual`, lease `antigravity-10482-mv06`, DL-#10482.
- Changes:
  - `force_torque.py`: `SpatialWrench`, `transform_wrench` with moment arm calculation, `compute_center_of_pressure` with strict threshold semantics (Fz <= 5.0 N returns None), and `ContactReaction`.
  - `counterfactual.py`: `AccelerationDecomposition` (gravity, drift, control, ZTCF, ZVCF), `CounterfactualStrategy`, `CounterfactualFork`, and `create_counterfactual_rollout` with cryptographic baseline immutability check.
  - `candidate_session.py`: Added `get_wrench_at`, `get_center_of_pressure`, `get_joint_torques_at`, `get_closure_residual_at`, and `create_counterfactual_fork`.
  - `requests.py`, `analysis.py`, `simulation_service.py`: `GET /analysis/candidate/forces` and `POST /analysis/candidate/counterfactual` endpoints (failing closed with 409 when no session is loaded).
  - `force_inspection.py`, `gui.py`: `ForceInspectionWidget` synchronized with physical playback time embedded in Tour Matching Viewer.
- Reproduction: `pytest tests/unit/motion_matching/test_force_torque.py tests/unit/motion_matching/test_counterfactual.py tests/unit/motion_matching/test_candidate_session_forces.py tests/unit/api/test_candidate_session_analysis_routes.py tests/unit/tools/test_tour_matching_viewer_forces.py`.
- Next: PR auto-merge; epic #10476 completion!

## MV-05 Manage MeshCat and Gepetto Launch Lifecycle and URDF Loading (#10481)

- Completed in PR #10501 (merged). Managed subprocess lifecycle in `ViewerProcessManager`, active port polling, existing listener detection, native viewer backends (`open_in_native_viewer`), `ModelBundle.extract_to`, CLI and GUI integration.

## MV-04 Reuse Shared Physical-Time Playback Across Viewers (#10480)

- Completed in PR #10496 (merged). PhysicalTimePlayback, quaternion SLERP, Euclidean LERP, playback adapters matrix, Tour Matching Viewer transport controls.

## MV-03 Bind Saved Candidates to Viewer & Analysis Sessions (#10479)

- Completed in PR #10492 (merged). CandidateSession ingestion, WSL host probe, multi-candidate replay overlay with ENGINE_COLORS, rejected fit banner, GIF export.

## MV-02 Anatomical Visual Assets and Skin Toggling (#10478)

- Completed in PR #10488 (merged). Added visual skin modes, visible diagnostic fallback (magenta), cadence pacing, and verified physics immutability under visual skin toggles.

## MV-01 Qualify Shared URDF Bundles and Numeric Precision (#10477)

- # Completed in PR #10485 (merged). 17g float serialization, ModelBundleManifest, ModelBundle zip export/import, Drake export integration, Pinocchio parity verified.

## MS-21 MuJoCo Replay Continuation (#10336)

Working directory: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-codex-10336`.
Branch: `feat/10336-mujoco-candidate-replay`. Base HEAD:
`5347cba0f4378cd72a6e8afea9fb27c8bfe5db75`. Implementation `94ccb1825`;
PR #10448 open. User authorized commit, push, PR, CI repair and merge.
Development-log entry: DL-#10336. Ownership was checked free and leased as
`codex-mujoco-replay-10336-20260918` before editing. The original checkout's
Pinocchio changes and untracked evidence were preserved.

Implemented name-mapped armature and saved-control replay through the existing
MuJoCo rigid-grip/shared-contact plant, exact G1 slicing and fail-closed checks.
See [Reproduction and Results](../../evidence/matched/driver_full_mujoco_replay/README.md)
for commands, candidate identity and receipt. Historical source evidence is
unchanged. Diagnostic armature 0.005 kg·m² follows turnover guidance but cannot
be certified as source-identical: the merged analytic producer never applies
armature and its receipt omits it, contact configuration and interpolation.
Missing root history and independent source dynamics also block qualification.

Validation: 40 focused replay/acceptance/schema/full-body/contact tests pass;
scoped Ruff and mypy pass; agent-context and architecture/file-size budgets pass.
PR CI found CVE-2026-63374 and CVE-2026-64847 in the inherited AnyIO 4.12.1
pins. SELF updates both runtime/development locks to the reported fixed 4.14.2;
the audit remains enforced. SELF refreshes the matched-swing ledger for the new
receipt and repairs the existing cross-engine gravity fixture (#4249): MuJoCo
and Drake now consume the same collision-free canonical URDF and right-hand
anchor rather than unlike demo plants. The unchanged 5 mm gate passes with
real MuJoCo 3.8.0 / Drake 1.51.1 on Linux (15 passed, 2 unavailable-engine skips),
including new analytic free-fall checks. This fixture does not qualify contact
or bilateral grip closure. Replay plus ledger tests: 28 passed. The fixture uses
defusedxml for both reading and serialization to satisfy the XML security gate.
Continue PR CI on the branch updated with main's shadow-tracker merge.
Commit/push hooks pass, including security and bounded unit checks. All 21
replay tests also pass with CI-pinned MuJoCo 3.8.0 in an isolated environment.
The central development-log validator flags inherited missing verifying SHAs,
a missing PR field in DL-#9967, and the existing portfolio WIP excess. These
unrelated entries are not repaired in this diff. The repo-local validator
script is absent, so the central Repository_Management copy was used.
Next: review this focused diff, then obtain a provenance-complete source
candidate and independent native replay. Do not close #10336 or promote G2/G3
from these diagnostic artifacts. Earlier lane handoffs follow unchanged.

---

## Shadow Tracker #10233 Handoff

- Workspace: `C:/Users/diete/Repositories/UpstreamDrift-10233-pr`.
- Branch: `fix/shadow-tracker-10233-pr`; HEAD/base:
  `5347cba0f` (origin/main); implementation `c76d6f02c`; handoff update SELF; PR #10450 open.
- CI repair SELF: upgrade AnyIO locks 4.12.1 -> 4.14.2 for
  CVE-2026-63374 and CVE-2026-64847; no audit waiver or gate weakening.
- Entry DL-#10233: provider implementation and tests complete; renderer and
  unrelated original-checkout work preserved.
- Contracts, RED/GREEN commands, compatibility and limitations:
  [Shadow Tracker turnover](../plans/shadow_tracker/TURNOVER_CURRENT.md).
- Next action: validate PR #10450 checks, repair any actionable CI failures,
  then merge through normal branch protection.

## OpenSim Epic #10394 Status (OG-01 Through OG-09 Completed)

- Governing epic #10394 / #10363. All 9 child tasks completed and qualified:
  - OG-01 (#10404, merged): Baseline model geometry and structural qualification audit.
  - OG-02 (#10407, merged): Consistent OpenSim segment scaling module and acromion proxy reconstruction.
  - OG-03 (#10405, merged): Parameterized visual club geometry and grip offset frames.
  - OG-04 (#10408, merged): Pure 3D rigid capture registration with Kabsch alignment and camera presets.
  - OG-05 (#10409, merged): Two-handed address pose calibration, grip closure verification, and tolerance profile.
  - OG-06 (#10410, merged): Full-swing dynamic tracking qualification and multi-stage ladder progression.
  - OG-07 (#10412, merged): Versioned OpenSim model variants, actuation profiles, and adapter boundaries.
  - OG-08 (#10413, merged): Muscle and tendon extension qualification, moment arm derivative check, and static tendon equilibrium.
  - OG-09 (#10403, merged): Native viewer package packaging, timeline scrubbing, visual layer toggles, and release evidence export.

---

# Native Multi-Engine Matching Handoff

## Matched Swing Program: Pinocchio Lane (MS-31, #10338)

Updated 2026-09-18 (local/claude). Full 654-frame Driver (`C3D_TA_Driver.c3d`, 1.814 s) and 657-frame 7-Iron (`C3D_TA_Iron.c3d`, 1.827 s) swings successfully matched in ~8.3 s using the decoupled kinematic tracking + analytic inverse dynamics pipeline (`scripts/match_pinocchio_c3d.py`).
Delivered PR #10411 on `feat/10338-crocoddyl-native-fit`. Both Optimum (minimum 2-norm, 3.9 ms) and Trail-Side Zero (tau_trail == 0, 168-215 ms, 436-616 N grip force transfer) torque solutions computed with exact forward dynamics acceleration parity.
Current turnover: `docs/development/matched_swing_program/MS31_PINOCCHIO_CROCODDYL_TURNOVER.md` and `docs/development/PINOCCHIO_C3D_MOTION_MATCHING_GUIDE.md`. Program epic #10363.

## Current Status

Run101 is a **rejected 0–0.85 s prefix**, not a completed full-swing match.
The independent R2025b replay passes four of five marker/yaw gates; terminal
RMS is **40.3115 mm**, above the 35 mm limit. The optimizer hit its iteration
limit and returned accepted=false / optimizer_converged=false.

The 2026-09-15 review recomputed raw NPZ/MAT results: maximum Euclidean
Simscape–Pinocchio marker discrepancy is 0.0604935 mm; overall RMS 20.26494 mm,
early RMS 9.99517 mm, club RMS 8.42248 mm and yaw error 0.54345%.
Required MATLAB settings: R2025b Update 5, ode15s, RelTol 1e-6, AbsTol 1e-9,
MaxStep 1/1440 s. Preserve the corrected seed geometry and world-force convention.
This is measured prefix agreement, not proof of universal numerical convergence.

Read [Run101 Review and Completion Turnover](simscape_tour_matching/RUN101_REVIEW_AND_TURNOVER.md)
for the current evidence, ordered work packages and copy-ready agent prompt.
It supersedes earlier claims that all fitting gates are certified. Three head
markers account for 38.61% of terminal squared error and LUArmHigh for 15.40%.
Next: bounded refinement verification, terminal body/attachment feasibility,
one justified fitting trial, then progressive full-capture extension.

Branch: feat/9967-native-simscape-pinocchio. Issues: #9967 / #9921.
Review source checkpoint: ca750f7d7; SELF adds the current turnover review.
Workspace: C:/Users/diete/Repositories/Worktrees/UpstreamDrift-pinocchio-native.
Remote hosts: DeskComputer (R2025b) and ControlTower (Pinocchio WSL).
No new numerical jobs were launched during this review; check live processes
before resuming. Full capture and other-engine qualification remain incomplete.

## Representation Qualification

Run58 is terminal (handle2605 exited0). Local-chart DOP853 on native spherical
Pinocchio PASSES all existing parity gates over the run19 0.85-second fixture:
marker maximum2.62982e-8 m; native q maximum1.28599e-7; native rate
maximum1.15849e-5; closure pose3.74372e-11 and rate9.40426e-11.
Elapsed37.6895 s/9972 RHS calls versus7.2147 s for the tighter scalar reference.
This is a same-input representation result, NOT a C3D fit or full R2025b parity.

Run59 is TERMINAL, original handle97788 exited0: same setup with smaller maximum
step. It FAILS native velocity parity (0.000587270); marker maximum1.15187e-7 m,
native q6.06676e-7, closure pose6.60249e-11/rate1.61521e-10. Elapsed61.2725 s.
Therefore58's isolated pass is not robust convergence or representation acceptance.
Those representation runs are terminal. The pointwise acceleration/history audit
found no sampled history dependence or substantial actuator-route mismatch:
maximum routed native acceleration difference2.18034e-6 rad/s² at a reference
539073.49 rad/s²; effort roundtrip1.25056e-12. Sampled unscaled inertia condition
is about6.7e7 in both representations. These results support investigating
conditioning/trajectory sensitivity, not declaring a physical-model mismatch.
They do not establish global convergence. Detailed receipt:
simscape_tour_matching/native_evidence/manifold_acceleration_10043_19.
Remote runtime /home/dieterolson/native-manifold-10043-18; driver
/mnt/c/Users/diete/compare_native_manifold_replay_9967_59.py; output
/mnt/c/Users/diete/native-manifold-replay-9967-59. Horizon0.85, methoddop853,
rtol1e-12, atol1e-14, max_step1/1440, max_evaluations100000. Scalar reference
rtol1e-12, atol1e-14, max_step0.000125. Poll the existing handle/process before
launching another run. All root runs through59 are terminal; none should restart.

## Transition Sensitivity and Regularized Fitting

Run60 completes six original-state replays. Repeated baseline results are exactly
identical. A late LSInputX B6 perturbation of1e-6 Nm changes native rates by
0.00750466 rad/s near0.786111 s but markers by only0.5386 micrometres. The
1e-4 Nm trials produce about0.765 rad/s and50.8 micrometres. Central derivative
estimates retain amplitude dependence; these are measured sensitivity evidence,
not a waiver of parity gates. Exact inputs, executed source and raw trajectories
are archived in native_evidence/perturbation_9967_60.

Shared prefix and multiple-shooting fitters now accept checked analytic penalty
Jacobians. NativeEffortPenalty reuses the native actuator/frame provider and
seven-point quadrature to evaluate exact mean-square total degree-six primitive
efforts. Twelve new tests passed after RED; combined fitter/effort/control regression77 tests
passes, as do Ruff and four-module mypy. Immutable ControlTower runtime61 passes
96 focused/real Pinocchio tests; source archives/hashes are preserved in
native_evidence/regularized_fit_runtime_9967_61. Existing runner behavior remains
the default with zero penalty. Trial61 is terminal0 in75.40 s: whole RMS29.813 mm, early10.707 mm,
terminal75.846 mm, club23.930 mm, versus baseline30.791/10.667/99.989/56.557 mm.
All three primal sensitivity gates pass. Max_nfev3 was reached; no numerical
acceptance or convergence is claimed. Native effort cost barely changes
(1.55060 to1.55026), so improvement cannot be attributed to the penalty alone.
Exact evidence and convenient candidate: native_evidence/regularized_fit_9967_61.

Run62 is terminal0 after468.965 s: whole28.6353 mm, early10.8248 mm,
terminal66.7134 mm, club31.0615 mm. It reaches max_nfev20 with four active
correction bounds; accepted/converged both false. Returned canonical candidate
7467c5d82817251858255bdf0e560a712f0d20a366ee0094c364a6c14481e1d5.
Native evidence folder regularized_fit_9967_62 contains the terminal receipt,
measured replay arrays and marker-comparison.png (root visually inspected).
Run63 is terminal1 after77.672 s and three forward evaluations. The B4/B5/B6
trial fails the unchanged sensitivity-primal marker agreement gate on its third
candidate, hash e057a7e977cfe2a163fe4c5e7e5767c17d5025b4ef7f33166672afbf658c358d.
There is no returned accepted candidate. First two Jacobians passed. Diagnostic64
PASSES at that exact candidate before its600 s wall budget: augmented solve
409.086 s/1,222,535 evaluations, primal marker discrepancy8.9147861e-9 m against
the unchanged1e-7 gate. Total launch466.239 s; diagnostic call/save419.190 s.
The augmented settings were rtol1e-12/atol1e-14/max_step0.000125. The provider's
independent primal remained rtol1e-11/atol1e-13; max_step applies to both. This
supports an accuracy-dependent63 failure, not a derivative correctness or C3D
acceptance claim. Cost is far above the default sensitivity calls. All jobs through69 are terminal; diagnostic70 is the next authorized experiment.
Runtime65 adds the checked call-budget API and passes36 tests including nine
real Pinocchio tests. Diagnostic65 changes only max_step to0.000125 relative
to63, retaining augmented tolerances1e-10/1e-12: marker difference3.88459e-8 m,
82931 sensitivity calls,27.0993 s (versus64's409.086 s). Gates are unchanged.
Audit66 checks saved64 LSInputX B6 column41 against six original-state replays:
relative errors9.11e-5/4.94e-4/9.53e-6 for1e-5/1e-4/1e-3 Nm, all below the
existing1e-3 threshold. This verifies one direction, not the entire Jacobian.
Raw sources/inputs/replays are preserved in sensitivity_9967_65 and derivative_9967_66.

Runner now accepts --max-step for both forward and sensitivity paths and
--max-sensitivity-evaluations; defaults preserve prior behavior. Five new parser
checks went RED/GREEN;13 runner tests pass. Runtime68 qualifies32 tests including nine real Pinocchio tests, plus runner
--help. Fit68 is TERMINAL1 after425.444 s, former PID2842183 / handle54657;
output /mnt/c/Users/diete/native-regularized-fit-9967-68. Authorized fit68 retains original
baseline19, restarts returned62, uses B4/B5/B6,max_nfev10,max_step0.000125,
sensitivity budget100000, and unchanged weight0.01/scales/amplitude/bounds.
Eight Jacobian checks pass, then the ninth candidate fails unchanged marker
agreement (not its evaluation budget). Candidate canonical hash:
c8885ace0b1021dea6e2fbd824286fd37eeff8cb91498147d2d585cb64bed039.
The last unqualified checkpoint reports28.138 mm whole/65.565 mm terminal;
it is not a returned solution. Exact evidence: native_evidence/regularized_fit_9967_68.
All previous numerical runtimes remain immutable.

Manufactured diagnostic67 confirms inactive sensitivity-column padding changes
physical integration error under the existing combined error norm. This suggests
an architecture improvement, not a proof of native model error. Keep it as a
regression case for opt-in grouped DOP853 error control, now implemented through
shared integrate_forward/integrate_sensitivities and the native provider/runner.
The maximum of SciPy's existing physical and sensitivity block norms controls
steps; default behavior is unchanged. The protected SciPy hook requires runtime
qualification. Initial-step selection remains unchanged.57 focused tests pass
after RED/GREEN, plus Ruff; no global accuracy guarantee is inferred.
Diagnostic70 is TERMINAL1 on the exact failed68 candidate, max_step0.000125,
rtol1e-10/atol1e-12,100000 sensitivity calls, separate_error_control=True.
Runtime70 passes49 focused tests including nine real Pinocchio cases. Agreement
fails narrowly at1.02308673e-7 m, t0.85, WaistRBack axis1; limit remains1e-7.
Call35.969 s, launch40.103 s. Diagnostic71 is TERMINAL1: tighter augmented
rtol3e-11/atol3e-13 yields1.02308684e-7 m, effectively unchanged, in34.025 s.
Diagnostic72 is TERMINAL0: max_step0.0000625 with original70 tolerances yields
2.506897942e-8 m agreement,164339 calls/53.916 s sensitivity (71.554 s call/save).
Marker samples and state/marker sensitivity Jacobians are archived; primal q/qd
trajectories were not saved. This qualifies one candidate, not all derivatives.
Fit73 is TERMINAL0 after989.414 s, accepted/converged false, max_nfev10.
All ten Jacobian checks pass; final whole28.10547 mm, terminal65.39791 mm,
early10.86040 mm and club30.04069 mm. Settings: original19 baseline, B456,
grouped control, max_step0.0000625,200000 sensitivity calls.
Runtime73 qualifies62 tests plus runner help. Former PID2891462/handle55242
are terminal. Output /mnt/c/Users/diete/native-regularized-fit-9967-73. A separate
post-return replay saves actual q/qd (not qdd), markers and a rejected overlay.
Final candidate canonical SHA256:
786522cd5380a9f62602920b2cff6e6b404f7ceb99bfc6783f1f359b98a6457a. Four changed source modules pass mypy.
Marker agreement failures now report discrepancy, time, marker and axis.

The shared native motion sequence API is implemented and publicly exported.
It converts complete batches with immutable time/model/frame/branch metadata
and original per-sample references, reusing NativeJointStateAdapter. Root passes
101 related tests including12 new strict atomic file-I/O tests. Public
save_native_motion/load_native_motion preserve the validated sequence plus an
optional distinct raw-model hash. UI and complete dynamic frame transport remain;
Five new consumer checks pass, including actual MuJoCo all16 frames at three
samples; Drake/Pinocchio consumer tests skip locally for missing runtimes.
These are conversion checks, not alternate MuJoCo/Drake dynamics qualification. See dedicated
REPRESENTATION_HANDOFF for runnable usage and the remaining ordered plan.

## MuJoCo Spherical Export and Fitting Conditioning

The opt-in native_spherical_mjcf exporter now reuses canonical MJCF geometry and
shared group inventory to replace three XYZ triples with ball joints. Real local
MuJoCo3.3.4 compiles nq30/nv27; five tests verify unchanged inertias/closure/scalars
and all body/site poses across three manufactured states. This is kinematics
qualification, not a dynamics adapter or stock mj_step acceptance. Next gates:
tangent conventions, dual effort/convective acceleration, reused rigid-closure
solve, then same-input trajectories. Dedicated turnover:
../development/mujoco_native_matching/MUJOCO_MANIFOLD_HANDOFF.md.

Offline audit74 uses saved72 Jacobians, existing valid masks, amplitude10 and
terminal weight10. Marker-only23100-by-81 condition number is7.452e7; column
normalization leaves3.767e7. Penalty rows/bounds are excluded. This supports
checking correlated sensitivities, not claiming a proven cause of stagnation.
Fit73 ended with only modest matching improvement. Inspect
actual/predicted reduction and a mixed-control directional derivative before
more compute. See native_evidence/sensitivity_condition_9967_74.

## Fixed-Attachment Feasibility

Audit69 uses existing rigid-body relaxation on all654 frames. Three head markers
are attached to Hub; no independent native head joint exists. Fixed-offset lower
bounds are17.393 mm whole swing/24.279 mm terminal, even with independent body
poses and no dynamics/connectivity. Separating head as a hypothetical independent
six-DOF body lowers this to4.471/9.177 mm, but changes the model. No marker/model
changes were made. Current torque errors exceed the rigidity floor, so both
optimization and physical approximation matter. Preserve original25/35 mm gates;
never silently omit head markers. See native_evidence/rigidity_9967_69.

## Reproducible Compute Limits

`integrate_forward` and `integrate_sensitivities` now accept optional positive
integer max_evaluations, checked before excess derivative/linearization calls.
`replay_marker_sensitivities` exposes max_sensitivity_evaluations for the augmented
solve only; its independent primal replay remains separate. DefaultsNone retain
existing behavior. Seven new tests and native forwarding checks went RED/GREEN;
27 related tests, Ruff and pinned mypy pass. This source update is NOT in immutable
runtime61; qualify a new runtime before using it remotely. Budgets are not a
numerical accuracy or wall-time guarantee. Exhaustion raises without partial data.

## Why Earlier Attempts Stalled

- Static pose fitting and feedback tracking did not produce an open-loop sextic
  motion. Run45 feedback whole RMS35.2 mm becomes346.5 mm in time-only replay,
  and655.5 mm after global-sixth-order compression. Do not promote the initializer.
- An actual native gimbal singularity stopped run41. An independent inverse-angle
  branch bug caused run49 to apply a different native effort map. Branch metadata
  is now explicit and tested; true singular native actuator inverses remain rejected.
- Native rate errors are amplified near the shoulder chart. Run52 physical
  angular error0.000208 rad/s became native rate error0.00465 rad/s. Tight scalar
  reference54 and manifold tolerance refinement led to passed58. Gates were not
  relaxed. Run54/55 second-level budgets were insufficient even for their selected
  maximum step; those failures are not proof of solver nonconvergence.
- The Python manifold implementation remains slower than scalar Pinocchio. Keep
  scalar native dynamics available as the fitting baseline while qualifying
  alternate representations. Do not block fitting on all-engine alternate builders.

## Two-Window Preflight and Direct Chart Derivatives

Stage1 diagnostic75 is TERMINAL0. Window0 starts at original q0/qd0 and window1
at the saved uninterrupted73 sample216 at0.6 s. Against uninterrupted73, maximum
marker distance is5.369e-12 m, native q4.422e-11 and qd8.804e-9. First endpoint
minus saved node is9.77e-14. Full pose/rate closure and the42-dimensional q/v
node chart pass. Zero retraction shifts physical state by4.75e-12. Exact evidence:
native_evidence/two_window_preflight_9967_75. This is fixture qualification, not
an optimized segmented swing or a derivative pass.

MultipleShootingOptions now accepts window_jacobian_state_coordinates="node"
(default "physical"). Node mode takes theta followed by direct chart columns,
retaining physical endpoint rows and the negative next-node transform derivative
in continuity. No pseudoinverse extension is needed. Eleven new nonlinear tests
went RED/GREEN; root passes44 combined shooting/node tests and source mypy/Ruff.

Diagnostic76 is TERMINAL (former PID2923511/handle25121), on frozen runtime73. It checks
81 control columns in window0 and81+42 in window1, then at most16 signed
perturbation trials for control, node-position, node-velocity and mixed directions.
Marker, q, qd and scaled endpoint/continuity checks retain1e-3 derivative gates
with explicit weak-block reporting. Process exit0 is not scientific acceptance:
status is failed_derivative_gates. LS B6 h1e-4 and mixed h1e-6 pass all blocks;
smaller steps fail some resolved checks. Both node directions pass resolved blocks
but near-zero cross-continuity derivatives fail relative checks (analytic norms
about5e-14/2e-17 versus central estimates about1e-10/1e-11). Establish physical
absolute-error floors/structural-zero treatment before declaring full qualification;
do not widen C3D or replay gates. Call225.112 s, launch250.611 s, all16 trials
complete. No optimizer or other numerical job remains active. Evidence:
native_evidence/two_window_derivatives_9967_76.

## Derivative Resolution Floors (Audit 77, Terminal)

Local analysis of the archived run76 arrays already resolves the two failure
classes. (a) Node-direction cross-continuity blocks: the zero-retraction
Jacobian equals diag(scales)\*basis to1.6e-10 with scaled orthonormality defect
3.1e-15, so the velocity response of the pure-position singular direction and
the position response of the Dq null direction are structurally zero up to
that defect; the archived retracted nodes match the linear prediction to
1.2e-14 (h1e-5) and1.3e-12 (h1e-4), so the central estimates (1.7e-10/1.5e-11)
are retraction roundoff divided by2h, not derivative error. (b) LSInputX h1e-5
and mixed h1e-7 absolute errors are consistent with replay noise divided by
step (endpoint q about4e-10 to2e-9 per replay); the archived perturbed
coefficients reproduce the intended Bernstein increments to4.8e-10 (control)
and4.3e-7 (mixed h1e-7) relative, and effort evaluation error is at most
2.3e-6 relative to h, far below the observed4e-3 failures.

New shared providers: `motion_matching/derivative_resolution.py`
(central_difference_floor, classify_derivative_block with verdicts passed /
structural_zero / unresolved_at_step / failed, qualify_direction and the
orthonormality cross_block_norm_bound) and
`pinocchio/python/native_node_chart.py` (closure/Jacobian/basis/retraction
wrapper reused by drivers). Twenty new unit tests pass with Ruff and mypy.
A pass now additionally requires the measured floor to be below the gate times
the analytic norm; agreement inside noise is reported unresolved, never passed.

Audit77 is TERMINAL0 (202.6 s launch,26 of32 budgeted window replays) on
runtime77 (frozen73 file set refreshed from f96766c8e plus five new files;122
qualification tests). Measured per-block replay error by tolerance variation at
unchanged max_step: markers8.2e-10 m, endpoint q9.3e-10, qd1.6e-8. Loose and
working replays coincide while both differ from tight by that amount: the
integration is max-step limited and the non-reproducible part is step-sequence
roundoff/constraint-solver noise, so tolerance alone does not shrink it. Base
replays reproduce run76's primal arrays exactly. Every factor-one failure sits
at the smallest step with absolute error1.02x–1.14x the single-pair floor;
`reclassify.py` applies the tested `floor_safety_factor=2` to the same archived
arrays and every direction/block then has a resolved pass (LSInputX h1e-4;
mixed h1e-6 and h1e-5) or an orthonormality-verified structural zero, with no
unexplained failure. Gates are unchanged. Evidence and turnover:
native_evidence/two_window_floor_9967_77 (raw ZIP SHA2563d5768fc…).

MultipleShootingOptions now also accepts shared_boundary_policy="once"
(default "both"), which observes the capture sample shared by adjacent windows
only in the earlier window so marker rows, Jacobian rows, segmented RMS and
equality offsets match an uninterrupted single-window objective at zero
defect. Four RED/GREEN tests;46 combined shooting tests, mypy and Ruff pass.

## Two-Window Direct-Node SLSQP Trials 78–81 (Terminal, Historical)

Runs 78–81 explored direct-node SLSQP optimizations restarting from run 73/79/80 (see `native_evidence/two_window_fit_9967_78` through `81`). All terminated at bounds or iteration limits with continuity defects remaining between 3.99e-4 and 1.61e-3, leaving candidates rejected against the 25/35 mm gates. Continuations only reshaped the last 0.1 s; error growth 0.4–0.75 s remained unchanged from run 73. Archived rotation-chart audits confirm Jacobian conditioning issues. See native evidence for archived raw ZIPs.

The archived run73 sampled rotation-chart audit gives peak condition2.680/17.886/
2.889 for hip/left/right shoulder. These samples do not bound between-sample
extrema or explain the full7.45e7 trajectory-control Jacobian condition by
coordinate maps alone. See native_evidence/returned73_chart_audit.

## Ordered Next Work

Use [Next Agent Execution Plan](simscape_tour_matching/NEXT_AGENT_CONVERGENCE_EXECUTION.md)
for the existing-provider, two-window derivative preflight now that fit73 is terminal.
It specifies integrated nodes, retraction/continuity chain-rule checks and bounded
SLSQP acceptance without recreating the solver or accepting state-reset motion.

1. Decide, one receipt each, on rebalancing terminal/early weighting (early RMS
   and pelvis yaw regress under the100x terminal weight), relaxing the ±0.05 node
   box, or extending the horizon beyond0.85 s with additional integrated nodes on
   actual capture samples (same-input replay check first). Continue from exact
   returned81 with the run81 driver pattern; report early-motion and pelvis-yaw
   regressions alongside any terminal gain; compare on uninterrupted metrics.

2. Qualify the variant over further representative trajectories and against
   MATLAB R2025b before claiming full equivalence. Qualify tangent derivatives
   before using the manifold variant in an optimizer. Preserve existing scalar API.
3. Resume trajectory/control co-optimization using native constrained forward
   dynamics. Reuse native profile, sensitivity, marker and shooting providers.
   Start from integrated states, permit trajectory changes, and optimize a global
   degree-six native effort basis. Do not repeat tight static-node boxes or simply
   compress an arbitrary feedback torque trace. Existing seven-DOF bioptim tracking
   is a different physical model and cannot silently replace this native model.
4. Require uninterrupted original-state replay through1.813888889 s, all654 capture
   frames and valid observations, closure, effort audit and independent R2025b
   validation before accepting the final swing. Preserve per-marker/time residuals,
   overlay animation, coefficient/time-basis metadata and exact input hashes.
5. Implement/qualify remaining engine variants sequentially using canonical
   pose_interchange providers, without inferring dynamics parity from pose roundtrips.

## Engine and Representation Handoffs

- [Pinocchio Manifold](simscape_tour_matching/PINOCCHIO_MANIFOLD_HANDOFF.md): native
  spherical variant nq30/nv27, exact fixed transforms/inertia/weld reused. DOP853
  integrates local tangent coordinates and carries physical endpoints unchanged;
  no quaternion projection or target-state reset. Runtime18 passes96 tests with
  one optional MuJoCo skip; root independently verified generic and real Pin tests.
  Remote receipts qualify the numerical runtime bundle, not full GUI/application
  deployment; retain the documented package boundaries when reproducing.
- [Canonical Representations](simscape_tour_matching/REPRESENTATION_HANDOFF.md):
  angle/quaternion/rate/convective-acceleration/effort conversions and fixed-frame
  transport. Preserve units, named frames, branch/winding and model identity.
  Moving-frame acceleration transport and #8867 convention consolidation remain open.
- [MuJoCo](mujoco_native_matching/HANDOFF.md) and
  [Drake](drake_native_matching/HANDOFF.md): native adapters and narrower receipts
  exist; alternate quaternion builders/full trajectory equivalence remain open.
  Drake's projected accelerations are not derivative-consistent trajectories.
- OpenSim epic #10003: plan/handoff branch docs/10003-opensim-matching-epic,
  documented plan commit1a68091b6. Implementation/runtime qualification is not
  accepted. Inspect its branch and requested planning check-in before proceeding.

The documented development-log validator path is absent in this checkout; no
validator pass is claimed. Normal configured commit/push hooks still apply.

## Evidence and Reproduction

Evidence root: simscape_tour_matching/native_evidence. Preserve raw ZIP archives. Original native model SHA256 `b817fea76407f71b29e42aeb188890df53d2d86874c7e5875d14d353f4a1a248`. Run19 canonical candidate SHA256 `b5b1c3823c86a21df323dc4e430366dc093495b069b3d25c361b0d007b8ff24f` (rejected 0.85 s fit).

ControlTower: ssh alias controltower; WSL ControlTower-Runner. Raw run receipts identify exact archived source and inputs. Never overwrite runs.

[Convergence Review](simscape_tour_matching/CONVERGENCE_REVIEW_20260912.md) gives strategy and delegation gates. [Historical Handoff](HANDOFF_HISTORY_20260912.md) preserves earlier matching history. Update this concise handoff and DEVELOPMENT_LOG with each commit.
