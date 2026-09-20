# Development Log — UpstreamDrift

State table for every feature in flight in this repository. Update entries
**in place**; never append dated sections. One entry per feature, from proposal
to ship. See the `development-logs` section of `AGENTS.md` for the binding rules
and `shared_scripts/development_log.py` for the validator.

- **Portfolio:** golf
- **WIP limit:** 8
- **Last audited:** 2026-09-12 by claude

## States

`proposed` → `in_progress` → `in_review` → `shipped`, with `parked` reachable
from any live state and `abandoned` from `parked`. `shipped` never returns to
`in_progress`; open a new entry instead.

## Active

### DL-#10432 · Calibrate and Smooth Full-Swing Pinocchio Kinematics With Exact Grip Compatibility

- **State:** in_review
- **Owner:** local
- **Issue:** #10432 (PF-02, epic #10427)
- **Branch:** feat/issue-10432-pf02-pinocchio-kinematics-grip-calibration
- **PR:** #10497 (auto-merge enabled)
- **Paths:** src/engines/physics_engines/pinocchio/python/marker_kinematics.py; src/shared/python/motion_matching/kinematic_smoothing.py; tests/unit/motion_matching/test_pinocchio_kinematics_calibration.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (10 unit tests in test_pinocchio_kinematics_calibration.py pass 100%; 41 related motion-matching tests pass; 979 pre-push tests pass; ruff check, ruff format, bandit, and mypy clean).
- **Summary:** Implemented SolveDiagnostics for Pinocchio MarkerIkSolver capturing convergence, projected gradient norm, cost decrease, and active bound counts. Added multi-start resolution solve_frame_multi_start evaluating geometric tracking floors with and without weld closure. Added refine_overlapping_window with bounded temporal regularization. Implemented kinematic_smoothing module providing joint trajectory smoothing with analytical/numerical derivative compatibility (q_dot ≈ v, v_dot ≈ a), boundary spike auditing, and cutoff frequency sensitivity analysis. Verified human range of motion wrist compliance (MM-2, #10104) and address left elbow pit up-and-inward alignment (MM-5, #10107).
- **Next step:** Land PR #10497 via CI and proceed to PF-03.
- **Evidence:** tests/unit/motion_matching/test_pinocchio_kinematics_calibration.py; src/shared/python/motion_matching/kinematic_smoothing.py; src/engines/physics_engines/pinocchio/python/marker_kinematics.py.

### DL-#10528 · Unify Sidekick, Setup, Help, and Library as Global Utilities

- **State:** in_progress
- **Owner:** local
- **Issue:** #10528 (ORG-19, epic #10508)
- **Branch:** feat/issue-10528-org19-global-utilities
- **PR:**
- **Paths:** src/shared/python/workspace/global_utilities.py; src/shared/python/workspace/**init**.py; tests/launchers/test_global_workspace_utilities.py; docs/development/HANDOFF.md; docs/development/DEVELOPMENT_LOG.md; SPEC.md
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 at HEAD (7 unit tests pass in tests/launchers/test_global_workspace_utilities.py covering all RED and GREEN criteria: switching workspaces updates assistant context without duplicate sessions or stale run references; canonical alias resolution for old assistant/library/setup IDs; dismissed onboarding persistence across session migration; keyboard open/close restores focus to prior widget; browser platform environment refuses native-only controls fail-closed; assistant conversation history persists across workspace navigation without deletion; divergence inventory updated; architecture budget and DRY duplication gate clean).
- **Summary:** Implemented `GlobalWorkspaceUtilitiesCoordinator` unifying Sidekick, Setup, Help, and Library utilities into global, workspace-agnostic overlays per ADR-0047 and issue #10528. Enforced canonical alias resolution mapping legacy utility IDs (`legacy_assistant`, `setup_wizard`, `library_browser`, `help_center`) to canonical names. Preserved conversation history across workspace transitions while synchronizing run and project context. Maintained sticky onboarding dismissal across session reload and migration. Enforced fail-closed native action behavior in browser environments.
- **Next step:** Push branch, open PR with auto-merge, release lease on #10528.
- **Evidence:** tests/launchers/test_global_workspace_utilities.py; src/shared/python/workspace/global_utilities.py.

### DL-#10525 · Consolidate Optimization and Training Launchers Under Shared Project Workspace and Controller Authority

- **State:** in_progress
- **Owner:** local
- **Issue:** #10525 (ORG-16, epic #10508)
- **Branch:** feat/issue-10525-org16-optimization-training
- **PR:** #10563
- **Paths:** src/shared/python/workspace/optimization_training_workspace.py; src/shared/python/workspace/**init**.py; tests/integration/test_optimization_training_workspace.py; docs/development/HANDOFF.md; docs/development/DEVELOPMENT_LOG.md; SPEC.md
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 at HEAD (9 integration tests pass in test_optimization_training_workspace.py covering all RED and GREEN acceptance criteria: invalid objectives/constraints/model compatibility prevent start, cancel/pause/resume and dependency failure maintain state integrity, duplicate submissions do not duplicate jobs, small deterministic optimization changes output for changed input, controller publishes metrics and registers result, and dataset selection with provenance survives reopen; check_file_size_budget, ruff check, ruff format all pass).
- **Summary:** Implemented `OptimizationTrainingWorkspaceCoordinator` consolidating optimization and training under shared workspace and scheduler authority. Bounded job form over the public optimizer and training controller authority, validating objectives, constraints, and model compatibility prior to dispatch. Enforced fail-closed handling for unsupported/uninstalled backends. Enforced cancel/pause/resume lifecycle invariants and deduplication of active submissions. Connected dataset selection with provenance directly to durable project sessions in `SessionProjectStore`.
- **Next step:** Push branch, open PR with auto-merge, complete lease on #10525.
- **Evidence:** tests/integration/test_optimization_training_workspace.py; src/shared/python/workspace/optimization_training_workspace.py.

### DL-#10524 · Compose Terrain, Putting, Scene, Bunker, and Simulator Delivery Modes

- **State:** in_progress
- **Owner:** local
- **Issue:** #10524 (ORG-15, epic #10508)
- **Branch:** feat/issue-10524-org15-scene-delivery-modes
- **PR:** #10561
- **Paths:** src/shared/python/workspace/shot_course_workspace.py; src/shared/python/workspace/**init**.py; tests/integration/test_shot_course_workspace.py
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 at HEAD (7 integration tests pass in tests/integration/test_shot_course_workspace.py covering all RED and GREEN acceptance criteria: incompatible ground/flight record rejected, terrain edit invalidates dependent run rather than silently mutating history, scene-only view fails closed with SceneNonPhysicsError, unsupported simulator destination disabled, putting fixture save/reopen round-trip, bunker multi-fidelity export round-trip, simulator network failure and cancellation flows; ruff check, ruff format, and check_file_size_budget pass cleanly).
- **Summary:** Implemented `ShotCourseWorkspaceCoordinator` composing Terrain, Putting, Scene, Bunker, and Simulator Delivery modes per ADR-0047 and issue #10524. Enforced explicit model boundaries: scene view is visual inspection only; bunker preserves F0-F3 fidelity tiers; putting conforms to rolling/ground contracts; terrain mutation increments revision and invalidates prior runs; simulator delivery verifies destination capabilities and produces explicit submission receipts.
- **Next step:** Push branch, verify CI, enable auto-merge, release lease on #10524.
- **Evidence:** tests/integration/test_shot_course_workspace.py

### DL-#10523 · Connect Swing, Impact, Flight, and Preserved Trajectory Viewers

- **State:** in_progress
- **Owner:** local
- **Issue:** #10523 (ORG-14, epic #10508)
- **Branch:** feat/issue-10523-org14-trajectory-viewers
- **PR:** #10560
- **Paths:** `src/shared/python/workspace/trajectory_handoff.py`; `src/shared/python/workspace/results_workspace.py`; `src/shared/python/workspace/__init__.py`; `src/shared/python/physics/flight_trajectory_export.py`; `src/shared/python/physics/swing_state_providers.py`; `tests/integration/test_shot_trajectory_handoff.py`
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (All 3 integration tests pass in test_shot_trajectory_handoff.py; 65 regression tests pass; ruff clean; DRY duplication clean).
- **Summary:** Implemented `ShotTrajectoryHandoffCoordinator` connecting swing-state providers, impact solvers, and aerodynamic ball flight simulation with specialized viewers per ADR-0047. Preserves viewer identities (Shot Tracer Qt, web BallFlight, and Impact Explorer ROC) without retiring viewers or merging distinct flight model families (`ud.flight_models` vs `swing_sim.flight`). Bridges `PipelineResult` to `swing_sim.ball_flight_trajectory/1` wire contract with immutable SI sample positions and timestamps. Enforces honest engine sourcing with fail-closed diagnostics (`UnsupportedEngineSourceError`, `ExtractionAdapterError`, `FrameUnitMismatchError`, `InvalidTrajectoryHashError`). Extends ResultsWorkspace with `COMPARE_FLIGHT_MODELS` and `OPEN_IN_IMPACT_EXPLORER` actions and provides atomic transaction staging/rollback.
- **Next step:** Push branch, verify CI, enable auto-merge, release lease.
- **Evidence:** tests/integration/test_shot_trajectory_handoff.py.

### DL-#10522 · Connect Subject, Club, Model, Pose, Fit, and Dynamics Stages

- **State:** in_progress
- **Owner:** local
- **Issue:** #10522 (ORG-12, epic #10508)
- **Branch:** feat/issue-10522-org12-model-match-handoff
- **PR:** #10547
- **Paths:** src/shared/python/workspace/**init**.py; src/shared/python/workspace/model_match_handoff.py; tests/integration/test_model_match_handoff.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (All 3 integration tests in test_model_match_handoff.py pass, 37 workspace regression tests pass; ruff check and format clean; check_file_size_budget clean).
- **Summary:** Implemented task adapters over existing public APIs for model selection/generation, subject parameters, club snapshot, and initial canonical pose bound to active project/session (`SessionProjectStore`). Separated general-input motion pipeline and tour driver/7-iron matching routes with explicit routing refusing unsupported arbitrary video observations. Passed validated target/model/pose references into fit jobs and recorded outputs/receipts in `SessionProjectStore`. Exposed Fit Kinematics and Run Dynamics as distinct steps with explicit backend choices across all 6 engines (`mujoco`, `drake`, `pinocchio`, `opensim`, `myosuite`, `simscape`) without silent substitutions. Enforced that kinematic outputs cannot be marked as dynamic qualified. Implemented downstream state invalidation on model change, failed-fit diagnostics, cancellation preserving prior runs, and reopen descriptor linking to Results/Replay seam.
- **Next step:** Push branch, open PR, enable auto-merge, release lease on #10522, and notify parent orchestrator.
- **Evidence:** tests/integration/test_model_match_handoff.py; tests/unit/workspace/test_workflow_transitions.py; tests/unit/workspace/test_artifact_handoff.py.

### DL-#10531 · Generate Accurate Atlas, Help, Parity, and Completion Records

- **State:** in_review
- **Owner:** local
- **Issue:** #10531 (ORG-21, epic #10508)
- **Branch:** `feat/issue-10531-org21-accurate-atlas-parity`
- **PR:** #10576
- **Paths:** `tests/scripts/test_workspace_documentation_freshness.py`, `src/config/industrial_readiness.json`, `docs/operations/industrial-readiness-index.md`, `src/tools/training_controller/README.md`
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 (`2c9f9fcd3`; all 8 tests pass in `test_workspace_documentation_freshness.py`; all 11 tests pass in `test_capability_atlas.py`; all 25 tests pass in `tests/config/industrial_readiness/`; ruff check and format clean; file size budget passed).
- **Summary:** Implemented comprehensive regression tests in `test_workspace_documentation_freshness.py` covering workspace membership drift, undocumented/dangling aliases, broken source/help links, stale generated views, shell-only parity vs compute-complete separation, deterministic generators, training controller README accuracy, and reconciled industrial readiness item U3 (#8820 / PR #9995) with merge SHA and verified implementation/test paths.
- **Next step:** Await CI completion and auto-merge on PR #10576.
- **Evidence:** `tests/scripts/test_workspace_documentation_freshness.py`, `tests/scripts/test_capability_atlas.py`.

### DL-#10353 · Results Browser Tile for Matched Swing Program

- **State:** in_progress
- **Owner:** claude
- **Issue:** #10353 (MS-80, epic #10363)
- **Branch:** feat/10353-results-browser
- **PR:** #10543
- **Paths:** src/tools/matched_swing_browser/**init**.py; src/tools/matched_swing_browser/**main**.py; src/tools/matched_swing_browser/gui.py; src/tools/matched_swing_browser/model.py; src/tools/matched_swing_browser/\_embed_adapter.py; src/config/models.yaml; src/config/launcher_manifest.json; src/config/feature_parity.json; src/launchers/embedded_tool_bootstrap.py; pyproject.toml; tests/tools/matched_swing_browser/test_matched_swing_browser_gui.py; tests/tools/matched_swing_browser/test_model.py; docs/development/matched_swing_program/evidence/browser/screenshot.png
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 at HEAD (21 unit and GUI tests pass; 10 registry and feature parity tests pass; architecture and file size budgets pass; DRY duplication gate clean; ruff and mypy clean; screenshot evidence recorded).
- **Summary:** Implemented `matched_swing_browser` embeddable PyQt6 desktop tool and data model. Left pane provides filterable table over 98 ledger receipts (by engine, capture, lane, verdict, search text). Right pane displays receipt summary, acceptance badge, five standardized metrics with units, physical gates breakdown, and lazy-loaded animated QMovie playback for rows with visual GIF artifacts. Provides action buttons to launch Tour Matching Viewer, Native Viewer (MS-83), and parity reports. Reuses `ResultFilter` lineage resolving #8824 and establishing contract for #10521 (ORG-13). Registered across all 5 canonical surfaces (`models.yaml`, `launcher_manifest.json`, `pyproject.toml`, `embedded_tool_bootstrap.py`, `feature_parity.json`).
- **Next step:** Update PR #10543, enable auto-merge, monitor remote CI to green merge.
- **Evidence:** tests/tools/matched_swing_browser/test_model.py; tests/tools/matched_swing_browser/test_matched_swing_browser_gui.py; docs/development/matched_swing_program/evidence/browser/screenshot.png.

### DL-#10529 · Consume Provider Ownership Decisions and Verify Runtime Import Authority

- **State:** in_progress
- **Owner:** local
- **Issue:** #10529 (ORG-20, epic #10508)
- **Branch:** feat/issue-10529-org20-provider-ownership
- **Paths:** src/shared/python/config/tools_vendor_authority.py; tests/integration/test_installed_provider_authority.py
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 at HEAD (8 integration tests pass in test_installed_provider_authority.py; 221 launcher/manifest tests pass with zero regressions; ruff check & format clean; file line budget passed).
- **Summary:** Consumed provider ownership decisions and implemented runtime import authority and provenance verification across repository, installed, and packaged execution environments. Added `assert_runtime_provenance_parity` failing closed upon root divergence between pytest and packaged app contexts. Added `verify_provider_provenance` asserting module paths resolve within canonical provider roots. Implemented `inspect_provider_authority` handling pinned gitlinks, clean installed wheel distributions (`ud-tools`), and probe import failures without silent fallback. Verified public seams for Sidekick, Movement Optimizer (`tools_movement_optimizer` via `ALIAS_MAP`), Pendulum (`swing_objective_lab`), and backward compatibility import delegation (`upstream_drift_tools` -> `sidekick`).
- **Next step:** Push branch, open PR with auto-merge, update issue.
- **Evidence:** tests/integration/test_installed_provider_authority.py.

### DL-#10520 · Move Tour Matching Execution Out of Documentation Without Changing Results

- **State:** in_progress
- **Owner:** local
- **Issue:** #10520 (ORG-11, epic #10508)
- **Branch:** feat/issue-10520-org11-motion-matching-packaging
- **PR:** #10546
- **Paths:** src/shared/python/motion_matching/execution/**init**.py; src/shared/python/motion_matching/execution/assets.py; src/shared/python/motion_matching/execution/spec_builder.py; src/shared/python/motion_matching/execution/downswing.py; src/shared/python/motion_matching/execution/mjx_export.py; src/shared/python/motion_matching/execution/driver.py; src/tools/motion_matching/pipeline.py; docs/development/full_body_models/build_anthropometric_spec.py; docs/development/full_body_models/evidence/ground_support/run_ground_support.py; docs/development/full_body_models/evidence/ground_support/downswing_experiment.py; docs/development/full_body_models/evidence/ground_support/export_mjx_package.py; tests/integration/test_installed_motion_matching.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (17 tests pass across tests/integration/test_installed_motion_matching.py, tests/tools/motion_matching/test_pipeline.py, tests/tools/motion_matching/test_motion_matching_gui.py; pre-commit checks pass; ruff format clean; file size budget clean).
- **Summary:** Extracted reusable execution out of documentation into `src/shared/python/motion_matching/execution/` (`spec_builder.py`, `downswing.py`, `mjx_export.py`, `driver.py`, `assets.py`). Preserved algorithms, numerical precision, parameters, coordinate frames, and output schemas. Resolved reference assets through standard resource resolution functions (`get_native_geometry_spec`, `get_opensim_model`, `get_candidate_geometry_spec`, `get_capture_c3d`, `resolve_output_root`) with environment variable overrides and clear error explanations for unavailable assets. Kept legacy script paths in `docs/development/full_body_models/` as thin compatibility wrappers issuing `DeprecationWarning` while delegating to packaged entry points and preserving CLI schemas and exit codes. Updated `pipeline.py` command constants (`BUILDER`, `DRIVER_SCRIPT`, `DOWNSWING_SCRIPT`, `EXPORT_MJX_SCRIPT`) to point to packaged entry points and write outputs outside docs/package.
- **Next step:** Push branch, open PR, enable auto-merge, release lease on #10520, and notify parent orchestrator.
- **Evidence:** tests/integration/test_installed_motion_matching.py; tests/tools/motion_matching/test_pipeline.py; tests/tools/motion_matching/test_motion_matching_gui.py.

### DL-#10519 · Connect Capture Rig, Optical Import, Pose Inspection, and Model Calibration Workspaces

- **State:** in_progress
- **Owner:** local
- **Issue:** #10519 (ORG-10, epic #10508)
- **Branch:** feat/issue-10519-org10-capture-inspection-handoff
- **PR:** #10545
- **Paths:** src/shared/python/workspace/**init**.py; src/shared/python/workspace/capture_inspection_handoff.py; src/tools/capture_rig/gui.py; src/tools/capture_rig/journey_actions.py; tests/integration/test_capture_target_handoff.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (32 tests pass across workspace unit suite and integration test_capture_target_handoff.py; ruff check and format clean; mypy pre-push clean; file size budget clean).
- **Summary:** Implemented `CaptureInspectionHandoff` and `FreeMoCapJobAdapter` bridging Capture Rig to Inspect Targets and Model Calibration. Enforced that trim, crop, and time offsets survive handoffs with explicit clock calculations; maintained MediaPipe and OpenPose as explicit estimator choices with separate observation sets, confidence scores, and source pixels; provided FreeMoCap CLI input/output validation before process spawn and cancellation leaving sources untouched with preserved HMR2/AGPL license isolation; opened C3D and optical imports keeping missing samples masked (NaN) and rejecting incompatible units and frames; rejected pretending 2-D coordinates are metric 3-D; and registered targets into `SessionProjectStore` automatically preserving annotations, calibration, and club metadata. Added "Open in Inspect Targets" action to Capture Rig GUI and JourneyActions.
- **Next step:** Push branch, open PR, enable auto-merge, release lease on #10519, and notify parent orchestrator.
- **Evidence:** tests/integration/test_capture_target_handoff.py; tests/tools/capture_rig/test_workflow.py; tests/unit/workspace/test_workflow_transitions.py.

### DL-#10518 · Guided Workflow Transitions Across Unified Workspaces

- **State:** in_progress
- **Owner:** local
- **Issue:** #10518 (ORG-09, epic #10508)
- **Branch:** feat/issue-10518-org09-workflow-transitions
- **PR:** #10544
- **Paths:** src/shared/python/workspace/**init**.py; src/shared/python/workspace/workflow_coordinator.py; tests/unit/workspace/test_workflow_transitions.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (26 unit tests pass across test_workflow_transitions.py, test_artifact_handoff.py, test_project_store.py, test_results_browser.py; ruff check and format clean; pre-commit mypy passed; check_file_size_budget clean).
- **Summary:** Implemented WorkflowCoordinator coordinating the 7-stage workflow pipeline (`Capture/Import -> Inspect Targets -> Configure Model -> Fit -> Dynamics -> Compare -> Export`) over typed ArtifactReference inputs and outputs. Evaluates live step readiness and diagnostics directly from cryptographic sha256 hashes and on-disk artifact existence rather than superficial flags. Added support for single-view coaching mode skipping 3-D dynamics when physics engines are unavailable, enforced contract distinction preventing dynamics from inheriting purely kinematic passes, tracked cancellation reasons and retry attempts, supported later-stage entry from imported artifacts, and exposed a pure state projection with to_dict() for Qt and React/Tauri parity.
- **Next step:** Push branch, open PR, enable auto-merge, release lease on #10518, and notify parent orchestrator.
- **Evidence:** tests/unit/workspace/test_workflow_transitions.py; tests/unit/workspace/test_artifact_handoff.py; tests/unit/workspace/test_project_store.py; tests/unit/workspace/test_results_browser.py.

### DL-#10517 · Unified Artifact and Project Context Handoff Between Workspaces

- **State:** in_progress
- **Owner:** local
- **Issue:** #10517 (ORG-08, epic #10508)
- **Branch:** feat/issue-10517-org08-workspace-handoff
- **PR:** #10542
- **Paths:** src/shared/python/workspace/**init**.py; src/shared/python/workspace/artifact_handoff.py; src/shared/python/workspace/project_store.py; tests/unit/workspace/test_artifact_handoff.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (18 unit tests pass across test_artifact_handoff.py, test_project_store.py, test_results_browser.py; ruff check and format clean; check_file_size_budget clean).
- **Summary:** Extended SessionProjectStore and ProjectMetadata with typed, versioned artifact handoffs (WorkspaceHandoff, ArtifactReference, ArtifactKind, RunMetadata). Enforced Design-by-Contract boundary preconditions (cross-session subject mismatch rejection, frame and schema validation, artifact existence and cryptographic sha256 hash checks before disk write, canceled/failed job qualification invariant). Added migration handling preserving unknown supported fields in older project.json files, atomic durability under interrupted writes, active context selection, and non-destructive run cloning without falsified output evidence. Implemented registered named artifact adapter conversion preserving provenance.
- **Next step:** Push branch, open PR, enable auto-merge, release lease on #10517, and notify parent orchestrator.
- **Evidence:** tests/unit/workspace/test_artifact_handoff.py; tests/unit/workspace/test_project_store.py; tests/unit/workspace/test_results_browser.py.

### DL-#10527 · Surface Cross-Engine Comparison and Injury Indicators in Dedicated Workspaces

- **State:** shipped
- **Owner:** local
- **Issue:** #10527 (ORG-18, epic #10508)
- **Branch:** feat/issue-10527-org18-comparison-indicator-workspace
- **PR:** #10568 (merged)
- **Paths:** src/shared/python/workspace/**init**.py; src/shared/python/workspace/comparison_indicator_workspace.py; tests/integration/test_comparison_indicator_workspace.py
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 at HEAD (10 integration tests pass in test_comparison_indicator_workspace.py; 23 total workspace integration tests pass; ruff check & format clean; mypy 0 issues; LoD clean; file line budget 447 <= 500 LOC; merged to main at 8e6cadb5c).
- **Summary:** Implemented `ComparisonIndicatorWorkspaceCoordinator` surfacing `canonical_core_comparison` and `injury_analysis` capabilities across `results_and_compare` and `exercise_analysis` workspace shells. Provides fail-closed validation on physical units, coordinate frame identifiers, timebase sample interval alignment, channel names, and model fidelity invariance (`ModelFidelityLevel` enum rejecting cross-tier comparisons with `IncompatibleArtifactError`). Integrates `CrossEngineComparisonAdapter` delegating trace alignment to `compare_traces` with SHA-256 provenance hashes, and `InjuryIndicatorAdapter` requiring physical load channels (`peak_compression_bw`, `peak_lateral_shear_bw`, `x_factor_stretch`) normalized by body weight without mock fallbacks, stamping every output with non-clinical disclaimers.
- **Next step:** Shipped.
- **Evidence:** tests/integration/test_comparison_indicator_workspace.py.

### DL-#10515 · Build Task-Oriented Desktop Navigation Over Existing Embedded Tools

- **State:** completed
- **Owner:** local
- **Issue:** #10515 (ORG-05, epic #10508)
- **Branch:** feat/issue-10515-org05-desktop-navigation
- **PR:** #10539
- **Paths:** src/launchers/workspace_navigation.py; src/launchers/launcher_layout_manager.py; src/launchers/\_launcher_navigation_ui.py; src/launchers/launcher_ui_setup.py; tests/launchers/test_workspace_navigation.py; docs/development/DEVELOPMENT_LOG.md; docs/development/HANDOFF.md
- **Started:** 2026-09-19
- **Last verified:** 2026-09-20 at HEAD (SELF; 22 unit tests pass in test_workspace_navigation.py; all 42 tests in test_launcher_ui_setup.py pass; all 35 tests in test_launcher_layout_manager.py pass; all 19 tests in test_workspace_tabs.py pass; ruff check, ruff format --check, mypy, check_file_size_budget clean)
- **Summary:** Built task-oriented desktop navigation over embedded tools (ORG-05). Added 5 primary task workspaces (`Capture & Analyze`, `Model & Match`, `Shot & Course Lab`, `Optimize & Train`, `Results & Compare`) and secondary navigation (`Developer & Research`, `All Tools`, `Favorites`, `History`). Integrated `ALIAS_MAP` layout migration into `LayoutManager.load_layout` preserving user custom tile scaling, view mode, and dock state. Enforced single-instance tool reuse in `dock_widget_as_tab` and `focus_or_open_tool_tab`. Provided accessible names, keyboard navigation, narrow-window scrolling via `QScrollArea`, return-to-workspace breadcrumbs (`WorkspaceBreadcrumbBar`), and actionable status explanations for missing/unconfigured capabilities (`explain_tool_status`).
- **Next step:** Pass CI, auto-merge into main, release lease on #10515.
- **Evidence:** tests/launchers/test_workspace_navigation.py; src/launchers/workspace_navigation.py.

### DL-#10355 · Motion Matching Tile Visual Playback, Standardized Metrics, and Navigation Handoff (MS-82)

- **State:** in_progress
- **Owner:** local
- **Issue:** #10355 (MS-82, closes #10106 gap)
- **Branch:** feat/10355-motion-matching-tile-playback
- **PR:** pending
- **Paths:** src/tools/motion_matching/gui.py; src/tools/motion_matching/pipeline.py; src/config/feature_parity.json; docs/development/feature_parity_matrix.md; tests/tools/motion_matching/test_motion_matching_gui.py
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 at HEAD (18 unit tests pass in tests/tools/motion_matching/; 39 parity tests pass in tests/config/feature_parity/; ruff check & format clean).
- **Summary:** Enhanced the Motion Matching PyQt6 tile to display asynchronous animation playback (`ik_playback.gif` and `tracking_playback.gif`) via `QMovie`, populated five standardized headline metrics (`full_capture_ik_rms_mm`, `address_marker_rms_mm`, `backswing_root_error_max_mm`, `whole_run_root_rms_mm`, `inside_support_polygon_fraction`), acceptance verdict badge, and navigation handoffs to the Matched Swing Results Browser and Tour Matching Viewer. Upgraded `tools.motion_matching` in `feature_parity.json` from `gap` to `parity`.
- **Next step:** Create PR, enable auto-merge, verify merge, and release lease.
- **Evidence:** tests/tools/motion_matching/test_motion_matching_gui.py, tests/config/feature_parity/test_matrix_freshness.py.

### DL-#10526 · Replace Canonical Estimation Shell With Bounded Estimator Coordinator

- **State:** shipped
- **Owner:** local
- **Issue:** #10526 (ORG-17, epic #10508)
- **Branch:** feat/issue-10526-org17-estimation-workflow
- **PR:** #10565 (merged)

- **Paths:** src/shared/python/workspace/**init**.py; src/shared/python/workspace/estimation_workspace.py; tests/integration/test_estimation_workspace.py
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 at HEAD (8 integration tests pass in test_estimation_workspace.py; ruff check & format clean; file line budget 447 <= 500 LOC; merged to main at 760a8470e).
- **Summary:** Implemented `EstimationWorkspaceCoordinator` replacing placeholder canonical estimation shells with a bounded application service coordinator. Integrates `solve_single_trial_map`, `IdentifiabilityGateOptions`, parameter priors and bounds, and spline trajectory evaluation. Provides fail-closed validation for non-finite cost/trajectories and ill-conditioned systems, with complete provenance persistence round-trips.
- **Next step:** Shipped.
- **Evidence:** tests/integration/test_estimation_workspace.py.

### DL-#10512 · Replace Misleading Launches With Real Tasks or Explicit Nonlaunchable States

- **State:** completed
- **Owner:** local
- **Issue:** #10512 (ORG-03, epic #10508)
- **Branch:** feat/issue-10512-org03-launch-truthfulness
- **PR:** #10536
- **Paths:** src/config/launcher_manifest.json; src/config/models.yaml; src/launchers/external_tools_adapter.py; src/launchers/launcher_model_handlers.py; src/launchers/launcher_process_manager.py; src/launchers/task_launch_truthfulness.py; tests/launchers/test_simulation_guis.py; tests/launchers/test_task_launch_truthfulness.py; ui/public/capability-atlas/graph.json; ui/public/capability-atlas/index.html
- **Started:** 2026-09-19
- **Last verified:** 2026-09-20 at HEAD (84 launcher unit tests pass; architecture budget clean; file size budget clean; DRY duplication gate clean; agent_context clean; SPEC.md updated).
- **Summary:** Audited capability launches and replaced misleading launch actions. Simulator prototype marked inspection/demo-only and distinct from qualified solvers; FreeMoCap parametric CLI guarded against zero-argument headless launch with parameter validation and cancellation safety; Video Analyzer placeholder fallback replaced with explicit unavailable provider diagnostic window (\_UnavailableToolWindow); library-only components (swing_optimizer, injury_analysis, pinn_pure_rigid, pinn_hybrid) and dual-shell service previews (canonical_core_estimation, canonical_core_comparison) audited with truthful dispositions, status messages, and next actions; fixed process assignment to Windows job objects for mock/invalid pids; regenerated capability atlas with verified freshness.
- **Next step:** Pass CI, auto-merge into main, release lease on #10512.
- **Evidence:** tests/launchers/test_task_launch_truthfulness.py; tests/launchers/test_simulation_guis.py; tests/launchers/test_launcher_process_manager.py; tests/scripts/test_capability_atlas.py.

### DL-#10511 · Separate Capability Identity, Maturity, Availability, and Qualification (ORG-02)

- **State:** completed
- **Owner:** local
- **Issue:** #10511 (ORG-02, epic #10508)
- **Branch:** feat/issue-10511-org02-capability-state-contract
- **PR:** #10535
- **Paths:** src/config/capability_state.py; src/config/launcher_manifest_loader.py; src/config/models.yaml; tests/config/launcher_manifest/test_capability_state_contract.py; docs/development/DEVELOPMENT_LOG.md; docs/development/HANDOFF.md
- **Started:** 2026-09-19
- **Last verified:** 2026-09-20 at HEAD (All 12 capability state contract regression tests pass; all 99 launcher_manifest tests pass; 7 launcher registry parity tests pass; ruff check clean; ruff format clean; black clean; mypy clean; SPEC.md updated).
- **Summary:** Disentangled overloaded capability status into four orthogonal typed dimensions: capability identity, lifecycle maturity (prototype/experimental/stable/deprecated), per-surface availability (desktop, web, api, cli) with actionable reason and remediation when unavailable, and evidence-backed qualification conforming to #10351 engine matrix contract (exempt for non-engine tools). Enforced invariant that installed engines without a qualified receipt never serialize as release-ready. Resolved Simscape/Matlab Models and Simulator/Golf Simulation Suite display names from a single canonical authority across native registry and web manifest views. Added lazy probing cache keyed on runtime identity without synchronous engine imports. Preserved backward-compatible legacy status field for API consumers.
- **Next step:** Pass CI, auto-merge into main, release lease on #10511.
- **Evidence:** tests/config/launcher_manifest/test_capability_state_contract.py; src/config/capability_state.py; src/config/launcher_manifest_loader.py.

### DL-#10510 · Baseline Every Capability and Preserve Tile, Layout, and Artifact Identity

- **State:** completed
- **Owner:** local
- **Issue:** #10510 (ORG-01, epic #10508)
- **Branch:** feat/issue-10510-org01-capability-baseline
- **PR:** #10534
- **Paths:** src/config/capability_migration.py; src/config/capability_migration.json; scripts/generate_capability_baseline.py; docs/development/ORG01_CAPABILITY_BASELINE.md; tests/config/test_capability_migration_coverage.py; scripts/capability_atlas/render.py; docs/architecture/CAPABILITY_ATLAS.md
- **Started:** 2026-09-19
- **Last verified:** 2026-09-20 at HEAD (18 unit tests pass in test_capability_migration_coverage.py covering all RED and GREEN acceptance cases; all 104 observed tiles, 45 parity features, 9 excluded tool packages cataloged with explicit migration metadata; legacy aliases starting_pose_matcher and putting_green_gui resolve acyclically; 15 golden fixtures pass hash checks; ruff/black/mypy clean).
- **Summary:** Built the canonical machine-checkable capability baseline inventory (`capability_migration.json`) and schema/validation engine (`capability_migration.py`) for Epic #10508. Enforced explicit contracts: IDs are unique, aliases are acyclic and resolve to retained targets, every capability has exactly one primary workspace domain, provider absence changes availability rather than identity, and preserved test fixtures retain golden byte hashes. Preserved ADR-0047 viewer identity and provider seam rulings. Created `scripts/generate_capability_baseline.py` producing `docs/development/ORG01_CAPABILITY_BASELINE.md` with freshness validation.
- **Next step:** Push branch, open PR with auto-merge, complete lease on #10510.
- **Evidence:** tests/config/test_capability_migration_coverage.py; docs/development/ORG01_CAPABILITY_BASELINE.md; src/config/capability_migration.json.

### DL-#10521 · Integrate Existing Results Browser Work With Replay, Data, and Export

- **State:** shipped
- **Owner:** local
- **Issue:** #10521 (ORG-13, epic #10508)
- **Branch:** feat/issue-10521-org13-results-workspace-handoff
- **PR:** #10548
- **Paths:** `src/shared/python/workspace/__init__.py`; `src/shared/python/workspace/artifact_handoff.py`; `src/shared/python/workspace/results_workspace.py`; `src/tools/matched_swing_browser/__init__.py`; `src/tools/matched_swing_browser/model.py`; `tests/integration/test_results_workspace_handoff.py`; `tests/tools/matched_swing_browser/test_model.py`
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (5 integration tests pass in test_results_workspace_handoff.py, 13 unit tests pass in test_model.py; ruff clean; mypy clean; line budget clean).
- **Summary:** Implemented `ResultsWorkspaceCoordinator` integrating canonical `ResultsBrowser` and #10353 `MatchedSwingBrowserModel` with Replay, Data Explorer, Plot, Compare, and Export actions. Enforces selected run context isolation (preventing global state leakage), artifact-type-aware action availability with diagnostic reasons, missing asset and unit mismatch validation (never guessing substitute files or silently comparing disparate units), and complete provenance retention during export and reimport (#8820).
- **Next step:** Merged in PR #10548.
- **Evidence:** tests/integration/test_results_workspace_handoff.py; tests/tools/matched_swing_browser/test_model.py.

### DL-#10513 · Validate Every Browser, Tauri, and Native Launch Destination

- **State:** completed
- **Owner:** local
- **Issue:** #10513 (ORG-04, epic #10508)
- **Branch:** feat/issue-10513-org04-launch-destinations
- **PR:** #10537
- **Paths:** ui/src/routes.ts; ui/src/api/launcherReachability.ts; ui/src/api/launcherReachability.test.tsx; ui/src/App.tsx; ui/src/api/webLaunch.ts; src/config/launcher_manifest_loader.py; src/shared/python/movement_optimizer/model_pack.yaml; tests/config/launcher_manifest/test_parity.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (SELF; Vitest 94 test files, 874 tests pass; pytest 95 launcher manifest and registry parity tests pass; ruff, black, mypy clean; capability atlas up-to-date; file-size budget clean).
- **Summary:** Established canonical route table `KNOWN_APP_ROUTES` and `isKnownAppRoute` in React UI; implemented browser and Tauri reachability evaluation and matrix generator (`evaluateTileReachability`, `generateReachabilityMatrix`); hardened `resolveTileLaunchAction` to reject unmapped routes with honest blocked/unavailable states; removed invalid web_route from Movement Optimizer model packs and sanitized in `LauncherManifestLoader` so Movement Optimizer resolves cleanly to `native-window`; expanded `test_route_mode_routes_exist_in_react_router` to inspect all loaded tiles from `LauncherManifest.load()` and added `test_every_tile_destination_resolves_authoritatively`.
- **Next step:** Commit, push, enable auto-merge, and release lease.
- **Evidence:** ui/src/api/launcherReachability.test.tsx; tests/config/launcher_manifest/test_parity.py.

### DL-#10516 · Apply the Same Workspace Navigation to React and Tauri

- **State:** completed
- **Owner:** local
- **Issue:** #10516 (ORG-06, epic #10508)
- **Branch:** feat/issue-10516-org06-react-workspace-navigation
- **PR:** #10540
- **Paths:** ui/src/types/workspaceNavigation.ts; ui/src/api/capabilityAdapter.ts; ui/src/components/layout/WorkspaceNavigation.tsx; ui/src/components/layout/WorkspaceNavigation.test.tsx; ui/src/pages/WorkspacePage.tsx; ui/src/components/simulation/LauncherDashboard.tsx; ui/src/App.tsx; ui/src/utils/routeTitles.ts; docs/development/DEVELOPMENT_LOG.md; docs/development/HANDOFF.md
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (SELF; 13 unit tests pass in WorkspaceNavigation.test.tsx; full 94-file 878-test Vitest suite passes; tsc type-check, eslint, vite build pass; WCAG contrast guard passes; file budget clean)
- **Summary:** Applied task-oriented workspace navigation across React and Tauri (ORG-06). Reused shared catalog metadata for the five primary workspaces (`Capture & Analyze`, `Model & Match`, `Shot & Course Lab`, `Optimize & Train`, `Results & Compare`) and secondary navigation (`Developer & Research`, `All Tools`, `Favorites`, `History`). Integrated `WorkspaceSidebar` and `WorkspaceBreadcrumb` within `WorkspaceShell` preserving browser history, bookmarkable task URLs (`/workspaces/:slug`), centralized route titles, and keyboard focus recovery. Created shared `resolveWorkspaceToolAction` capability adapter opening native tools under Tauri/desktop while providing actionable explanations and web alternatives for browser-only users.
- **Next step:** Commit, push, open PR referencing Fixes #10516, enable auto-merge, and release lease.
- **Evidence:** ui/src/components/layout/WorkspaceNavigation.test.tsx; ui/src/components/layout/WorkspaceNavigation.tsx.

### DL-#10514 · Group Engine Dashboards, Exercise Variants, and Repository Shortcuts

- **State:** in_progress
- **Owner:** local
- **Issue:** #10514 (ORG-07, epic #10508)
- **Branch:** feat/issue-10514-org07-model-variant-grouping
- **PR:** #10541
- **Paths:** src/config/models.yaml; src/launchers/exercise_dashboard.py; src/launchers/launcher_model_handlers.py; src/shared/python/config/**init**.py; src/shared/python/config/model_pack_manifest.py; src/shared/python/config/model_registry.py; src/shared/python/config/model_variant_grouping.py; tests/config/test_model_variant_grouping.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (6 unit tests pass in test_model_variant_grouping.py, 183 tests pass across adjacent suites; ruff check clean; ruff format clean; line budget check clean).
- **Summary:** Implemented `ModelVariant`, `LogicalModelIdentity`, `LogicalModelChoice`, and `ModelGroupingProjection` in `src/shared/python/config/model_variant_grouping.py` to project 28 exercise variants across 4 providers (`MuJoCo_Models`, `Drake_Models`, `Pinocchio_Models`, `OpenSim_Models`) into 7 logical choices without dropping any provider assets underneath. Preserved strict DbC on engine selection (never silently substitutes another engine). Added name collision protection across distinct canonical identities. Implemented `resolve_shortcut` mapping `biomech_sit_to_stand` (never falls back to gait) and `biomech_gait` to `biomech_exercise`, engine dashboards (`drake_dashboard`, `mujoco_dashboard`, `pinocchio_dashboard`) to engine advanced modes, and `movement_optimizer` / `tools_movement_optimizer` to unified task with #9406 authority resolution. Updated `SharedRepoHandler` with `get_missing_checkout_diagnostic` to emit actionable diagnostics when sibling repos are missing. Dynamicized exercise names in `exercise_dashboard.py`.
- **Next step:** Commit with conventional commit, push branch, open PR, and arm auto-merge.
- **Evidence:** tests/config/test_model_variant_grouping.py; tests/config/test_tile_paths_resolve.py; tests/unit/config/test_model_pack_manifest.py; tests/launchers/test_launcher_model_handlers.py.

### DL-#10482 · Expose Real Forces, Torques, and Explicit Counterfactual Semantics

- **State:** in_progress
- **Owner:** local
- **Issue:** #10482 (MV-06, epic #10476)
- **Branch:** feat/10482-forces-torques-counterfactual
- **PR:** #10504
- **Paths:** src/api/models/requests.py; src/api/routes/analysis.py; src/api/services/simulation_service.py; src/shared/python/motion_matching/candidate_session.py; src/shared/python/motion_matching/counterfactual.py; src/shared/python/motion_matching/force_torque.py; src/tools/tour_matching_viewer/force_inspection.py; src/tools/tour_matching_viewer/gui.py; tests/unit/api/test_candidate_session_analysis_routes.py; tests/unit/motion_matching/test_candidate_session_forces.py; tests/unit/motion_matching/test_counterfactual.py; tests/unit/motion_matching/test_force_torque.py; tests/unit/tools/test_tour_matching_viewer_forces.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (30 unit tests pass across force_torque, counterfactual, candidate_session_forces, candidate_session_analysis_routes, and tour_matching_viewer_forces; ruff clean; mypy 0 errors; Law of Demeter zero-growth clean; DRY gate clean; divergence inventory updated).
- **Summary:** Implemented `SpatialWrench` and rigid-body `transform_wrench` with rotational mapping and cross-product moment arm calculation ($\tau_B = R \tau_A + r \times (R F_A)$). Implemented `compute_center_of_pressure` with strict threshold semantics ($F_z \le 5.0$ N returns `None` rather than fabricated zeros). Implemented `AccelerationDecomposition` (gravity, drift, control, ZTCF, ZVCF) and `create_counterfactual_rollout` with cryptographic baseline immutability assertion (SHA-256 byte check before and after execution). Enhanced `CandidateSession` with `get_wrench_at`, `get_center_of_pressure`, `get_joint_torques_at`, `get_closure_residual_at`, and `create_counterfactual_fork`. Added API endpoints `GET /analysis/candidate/forces` and `POST /analysis/candidate/counterfactual` (failing closed with 409 Conflict when session is absent or kinematic-only). Integrated `ForceInspectionWidget` into Tour Matching Viewer GUI synchronized with physical playback time.
- **Next step:** Create PR #10504, enable auto-merge, and monitor until merged into main.
- **Evidence:** tests/unit/motion_matching/test_force_torque.py; tests/unit/motion_matching/test_counterfactual.py; tests/unit/motion_matching/test_candidate_session_forces.py; tests/unit/api/test_candidate_session_analysis_routes.py; tests/unit/tools/test_tour_matching_viewer_forces.py.

### DL-#10481 · Manage MeshCat and Gepetto Launch Lifecycle and URDF Loading

- **State:** shipped
- **Owner:** local
- **Issue:** #10481 (MV-05, epic #10476)
- **Branch:** feat/10481-meshcat-gepetto-lifecycle
- **PR:** #10501 (merged)
- **Paths:** scripts/launch_simulation_viewer.py; src/shared/python/model_generation/export/model_bundle.py; src/shared/python/motion_matching/native_viewers.py; src/shared/python/motion_matching/viewer_lifecycle.py; src/tools/tour_matching_viewer/gui.py; tests/unit/model_generation/test_urdf_precision_bundle.py; tests/unit/motion_matching/test_launch_simulation_viewer_cli.py; tests/unit/motion_matching/test_native_viewers_registry.py; tests/unit/motion_matching/test_viewer_lifecycle.py; tests/unit/tools/test_tour_matching_viewer_native_button.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (28 unit tests pass across viewer_lifecycle, native_viewers_registry, launch_simulation_viewer_cli, tour_matching_viewer_native_button, and urdf_precision_bundle; ruff clean; mypy 0 errors in 10 source files; Law of Demeter zero-growth clean; DRY gate clean; divergence inventory updated).
- **Summary:** Implemented `ViewerProcessManager` with managed background process lifecycle, active port polling, existing listener detection without collision (`reuse_existing`), process ownership tracking (never terminates unowned processes), CORBA port 12321 protection, dynamic MeshCat URL parsing, and startup crash diagnostics. Registered native viewer backends (`open_in_native_viewer` for MuJoCo, MeshCat, Gepetto, OpenSim, and MATLAB) with fail-closed missing dependency handlers and explicit install hints. Enhanced `ModelBundle` with `extract_to` and direct directory bundle loading with mesh asset inventory parsing. Extended `launch_simulation_viewer.py` CLI to support `--model-bundle`, `--urdf`, `--view-mode` (static, fitted, native), `--speed`, `--stride`, `--loop`, and `--output-html`. Added `_open_native_btn` in Tour Matching Viewer GUI.
- **Next step:** Merged into main in PR #10501.
- **Evidence:** tests/unit/motion_matching/test_viewer_lifecycle.py; tests/unit/motion_matching/test_native_viewers_registry.py; tests/unit/motion_matching/test_launch_simulation_viewer_cli.py; tests/unit/tools/test_tour_matching_viewer_native_button.py; tests/unit/model_generation/test_urdf_precision_bundle.py.

### DL-#10336 · Gate Ladder G1 -> G2 -> G3 for MuJoCo: Replay Pinocchio B100 Candidate

- **State:** shipped
- **Owner:** claude
- **Issue:** #10336 (MS-21, epic #10363)
- **Branch:** feat/10336-mujoco-candidate-replay-g1
- **PR:** #10500 (merged)
- **Paths:** src/engines/physics_engines/mujoco/python/candidate_replay.py; src/engines/physics_engines/mujoco/python/full_body_model.py; src/engines/physics_engines/mujoco/python/replay_contract.py; src/engines/physics_engines/mujoco/python/replay_evidence.py; tests/unit/motion_matching/test_mujoco_candidate_replay.py; evidence/matched/driver_g1_crocoddyl_rk45_b100_mujoco_replay/
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (22 unit tests pass in test_mujoco_candidate_replay.py under native MuJoCo 3.13.0 on ControlTower; full 307-frame forward replay generated; same-state FK error 2.51e-15 m; fail-closed G1 rejection recorded).
- **Summary:** Upgraded MuJoCo candidate replay pipeline and full-body model to support MuJoCo 3.x mj_fullM(model, data, mass) signature alongside MuJoCo 2.x (model, mass, data.qM). Relaxed control array shape validation to support (N-1, n_act) control intervals standard in optimal control solvers. Executed full 0.85s (307 frames) uninterrupted forward replay in MuJoCo for the Crocoddyl b100 candidate without numerical failure, recording candidate receipt, playback GIF, and fail-closed MS-100 verdict.
- **Next step:** PR #10500 created; auto-merge enabled.
- **Evidence:** tests/unit/motion_matching/test_mujoco_candidate_replay.py; evidence/matched/driver_g1_crocoddyl_rk45_b100_mujoco_replay/receipt.json.

### DL-#10340 · OpenSim IK on Shared Document With Validity Policy and Shared Receipt

- **State:** shipped
- **Owner:** claude
- **Issue:** #10340 (MS-41, epic #10363)
- **Branch:** feat/10340-opensim-document-ik
- **PR:** #10489 (merged)
- **Paths:** src/engines/physics_engines/opensim/python/full_body_osim.py; src/engines/physics_engines/opensim/python/tour_matching/marker_map.py; src/engines/physics_engines/opensim/python/tour_matching/document_ik.py; src/engines/physics_engines/opensim/python/tour_matching/cli.py; src/shared/python/motion_matching/pipeline/plants/opensim.py; src/shared/python/motion_matching/pipeline/plant.py; src/shared/python/motion_matching/pipeline/plants/**init**.py; docs/development/full_body_models/evidence/ground_support/anthro_driver_opensim/; reports/matched_swing_ledger.json; tests/opensim/test_document_ik.py; tests/unit/motion_matching/test_full_body_osim.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (Full 654-frame OpenSim IK executed in 50s on ControlTower; receipt.json validated; 6/6 native tests pass; 7 unit tests pass; architecture budget clean; ruff/black clean).
- **Summary:** Executed OpenSim InverseKinematicsTool on the exported anthropometric document model (full_body_spec_anthro_driver.json / full_body_anthro_driver.osim) with marker weights from MARKER_VALIDITY_POLICY (MS-04). Implemented OpensimMatchingPlant (reporting dynamics not_run: use moco) and registered in plant registry. Generated evidence package {receipt.json, ik.mot, candidate.npz, ik_playback.gif} under docs/development/full_body_models/evidence/ground_support/anthro_driver_opensim/ and indexed in reports/matched_swing_ledger.json.
- **Next step:** PR #10489 merged.
- **Evidence:** docs/development/full_body_models/evidence/ground_support/anthro_driver_opensim/receipt.json; reports/matched_swing_ledger.json.

### DL-#10480 · Reuse Shared Physical-Time Playback Across Qt React and Native Viewers

- **State:** shipped
- **Owner:** local
- **Issue:** #10480 (MV-04, epic #10476)
- **Branch:** feat/10480-physical-time-playback
- **PR:** #10496 (merged)
- **Paths:** src/shared/python/motion_matching/playback.py; src/shared/python/motion_matching/playback_adapters.py; src/tools/tour_matching_viewer/gui.py; tests/unit/motion_matching/test_physical_playback.py; tests/unit/motion_matching/test_playback_adapters.py; tests/unit/tools/test_tour_matching_viewer_playback.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (22 unit tests pass across physical_playback, playback_adapters, tour_matching_viewer_playback, and combo; ruff clean; Law of Demeter zero new violations; DRY gate clean; divergence inventory updated; architecture budget OK).
- **Summary:** Reuses shared Tools playback transport (`rate_of_closure.simulation.playback_transport`) in `PhysicalTimePlayback` so continuous physical time is the evaluation authority instead of naive tick timers. Implemented quaternion SLERP with antipodal sign continuity, Euclidean coordinate LERP for markers and forces, non-uniform timestamp support, dropped-draw handling without timescale drift, and discrete knot stepping. Implemented PlaybackAdapter capabilities matrix across Qt, React (web JSON payload), MeshCat, Gepetto, and MediaVideo (with media-time offset and documented mute reason). Integrated `PlaybackTransportControls` in `TourMatchingViewerWidget` while preserving paused camera manipulation.
- **Next step:** PR #10496 merged.
- **Evidence:** tests/unit/motion_matching/test_physical_playback.py; tests/unit/motion_matching/test_playback_adapters.py; tests/unit/tools/test_tour_matching_viewer_playback.py.

### DL-#10479 · Bind Saved Candidates to Viewer and Analysis Sessions

- **State:** shipped
- **Owner:** local
- **Issue:** #10479 (MV-03, epic #10476)
- **Branch:** feat/10479-viewer-analysis-sessions
- **PR:** #10492 (merged)
- **Paths:** src/api/routes/capabilities.py; src/api/services/simulation_service.py; src/shared/python/engine_core/wsl_probe.py; src/shared/python/motion_matching/candidate_session.py; src/tools/tour_matching_viewer/core.py; src/tools/tour_matching_viewer/gui.py; tests/unit/api/test_candidate_session_routes.py; tests/unit/engine_core/test_wsl_probe.py; tests/unit/motion_matching/test_candidate_session.py; tests/unit/tools/test_tour_matching_viewer_combo.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (23 unit tests pass across candidate_session, wsl_probe, viewer combo, and API routes; ruff clean; Law of Demeter zero new violations; DRY gate clean; divergence inventory updated; architecture budget OK).
- **Summary:** Ingests saved candidate trajectories and model specifications into immutable CandidateSession objects with SHA-256 verification and coordinate order remapping. Missing force channels remain None without fabrication. Probes WSL physics engine environment so Linux-native SDKs are accurately reported. Upgrades Tour Matching Viewer with MultiCandidateReplay supporting up to 4 candidates overlaid with ENGINE_COLORS, runs ledger combo selection, conspicuous rejected fit banner, capability indicators, and animation GIF export.
- **Next step:** PR #10492 merged.
- **Evidence:** tests/unit/motion_matching/test_candidate_session.py; tests/unit/engine_core/test_wsl_probe.py; tests/unit/tools/test_tour_matching_viewer_combo.py; tests/unit/api/test_candidate_session_routes.py.

### DL-#10478 · Anatomical Visual Assets and Skin Toggling Without Physics Mutation

- **State:** shipped
- **Owner:** local
- **Issue:** #10478 (MV-02, epic #10476)
- **Branch:** feat/10478-anatomical-visuals
- **PR:** #10488 (merged)
- **Paths:** src/engines/physics_engines/pinocchio/python/native_candidate_viewer.py; src/engines/physics_engines/pinocchio/python/viewer_presentation.py; src/shared/python/body_part_viz/anatomical_visuals.py; src/shared/python/model_generation/export/model_bundle.py; tests/unit/body_part_viz/test_anatomical_visuals.py; tests/unit/motion_matching/test_native_candidate_viewer.py; tests/unit/motion_matching/test_native_viewer_presentation.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (7 visuals tests pass, 4 presentation tests pass, 6 candidate viewer tests pass including native Pinocchio 4.1.0 in WSL Ubuntu-24.04 verifying physics immutability under visual skin toggles; ruff clean).
- **Summary:** Added anatomical visual asset bindings, visual skin modes (NONE, INERTIA_ELLIPSOIDS, ANATOMICAL_MESH, COLLISION), and deterministic visible diagnostic fallback (magenta). Implemented viewer presentation cadence pacing and cross-platform gepetto playback lock. Verified that visual skin toggling in Pinocchio leaves mass, inertia, generalized coordinates, forward kinematics, and contact sphere positions strictly invariant.
- **Next step:** PR #10488 merged.
- **Evidence:** tests/unit/body_part_viz/test_anatomical_visuals.py; tests/unit/motion_matching/test_native_viewer_presentation.py; tests/unit/motion_matching/test_native_candidate_viewer.py.

### DL-#10477 · Qualify Shared URDF Bundles and Preserve Numeric Precision

- **State:** shipped
- **Owner:** local
- **Issue:** #10477 (MV-01, epic #10476)
- **Branch:** feat/10477-urdf-bundle-precision
- **PR:** #10485 (merged)
- **Paths:** src/engines/physics_engines/drake/python/full_body_urdf.py; src/shared/python/model_generation/\_lazy_map.py; src/shared/python/model_generation/builders/urdf_writer.py; src/shared/python/model_generation/export/**init**.py; src/shared/python/model_generation/export/bundle_manifest.py; src/shared/python/model_generation/export/model_bundle.py; tests/integration/test_pinocchio_urdf_bundle_parity.py; tests/unit/model_generation/test_urdf_precision_bundle.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (7 unit tests pass; Pinocchio 4.1.0 parity verified in WSL native Ubuntu environment with < 1e-12 m transform and < 1e-11 mass matrix error; ruff clean).
- **Summary:** Upgraded URDF numeric serialization from lossy 6g/4g to deterministic 17g representation. Implemented ModelBundle and ModelBundleManifest with SHA-256 integrity verification, canonical coordinate ordering, and zip archive export/import. Integrated with Drake full_body_urdf export. Verified numeric round-trip parity with native Pinocchio.
- **Next step:** PR #10485 merged.
- **Evidence:** tests/unit/model_generation/test_urdf_precision_bundle.py; tests/integration/test_pinocchio_urdf_bundle_parity.py.

### DL-#10460 · Consume Shared GSPro Open Connect V1 Codec From Tools

- **State:** in_progress
- **Owner:** local
- **Issue:** #10460
- **Branch:** feat/10460-consume-tools-gspro-codec
- **PR:** not created
- **Paths:** src/shared/python/golf_simulator/adapters/gspro/codec.py; tests/unit/golf_simulator/test_gspro_codec.py; vendor/ud-tools; Cargo.toml; requirements-tools.txt; docs/shared_tools/divergence_inventory.v1.json; docs/shared_tools/divergence_inventory.md
- **Started:** 2026-09-18
- **Last verified:** 2026-09-18 at HEAD (SELF; TDD RED captured then green verified; 106 golf simulator unit and integration tests pass; ruff, black, mypy clean with 0 errors).
- **Summary:** Upgraded vendor/ud-tools pin to Tools commit a9ed0e7c5c6905b1164082659051d6381068052d carrying shared GSPro Open Connect v1 codec (Tools#5228). Refactored UpstreamDrift's gspro adapter codec to retain ShotEnvelope canonical SI/radian unit conversion and profile handling, but delegate wire payload encoding and response decoding to shared.python.launch_monitor.gspro_connect.
- **Next step:** Commit, push, open PR referencing Closes #10460, and arm auto-merge.
- **Evidence:** tests/unit/golf_simulator/test_gspro_codec.py.

### DL-#10336 · MuJoCo Replay of the Merged Pinocchio Driver Candidate

- **State:** in_review
- **Owner:** codex
- **Issue:** #10336 (MS-21, epic #10363)
- **Branch:** feat/10336-mujoco-candidate-replay
- **PR:** #10448 (open)
- **Paths:** src/engines/physics_engines/mujoco/python/candidate_replay.py; src/engines/physics_engines/mujoco/python/replay_contract.py; src/engines/physics_engines/mujoco/python/replay_evidence.py; scripts/replay_pinocchio_in_mujoco.py; src/shared/python/motion_matching/pipeline/receipt_schema.py; tests/unit/motion_matching/test_mujoco_candidate_replay.py; evidence/matched/driver_full_mujoco_replay/
- **Started:** 2026-09-18
- **Last verified:** 2026-09-18 at 94ccb1825; 40 focused tests pass and 21 replay tests pass on MuJoCo 3.8.0; commit/push hooks, scoped Ruff/mypy, architecture/file-size budgets and agent-context checks pass. SELF fixes PR CI's inherited AnyIO vulnerabilities with 4.14.2 in both locks; dependency audit remains enforced.
- **Summary:** Hash-checked saved-control replay with name-mapped non-root armature, shared contact law, rigid KKT grip, no feedback or pose resets, exact G1 windows, fail-closed source/parity/physical evidence checks, and separate rejected dynamics versus IK playback. Original candidate and receipts remain unchanged. The source omits plant/control provenance and root history, so identical-plant parity and physical acceptance remain unverified/rejected.
- **CI continuation:** SELF refreshes the receipt ledger and fixes #4249's gravity fixture to use the same collision-free URDF/right-hand anchor in MuJoCo and Drake. The 5 mm gate is unchanged; analytic free-fall assertions prevent shared wrong/stationary outputs. Real Linux engines: 15 passed, 2 unavailable-engine skips. Ledger plus replay: 28 passed.
- **Next step:** Finish PR CI and merge; regenerate a source candidate with recorded armature/contact/control provenance, root history and an independent uninterrupted native replay before advancing G1.
- **Evidence:** evidence/matched/driver_full_mujoco_replay/receipt.json; evidence/matched/driver_full_mujoco_replay/README.md.

### DL-#10233 · Shadow Tracker Revision Integrity and Persistence

- **State:** in_review
- **Owner:** codex
- **Issue:** #10233 (ST-04 / epic #10122)
- **Branch:** fix/shadow-tracker-10233-pr
- **PR:** #10450 (https://github.com/D-sorganization/UpstreamDrift/pull/10450)
- **Paths:** src/shared/python/shadow_tracker/segmentation.py; tests/unit/shadow_tracker/test_revision_persistence.py; docs/plans/shadow_tracker/
- **Started:** 2026-09-18
- **Last verified:** 2026-09-18 on c76d6f02c (remote-main base); 306 tests pass; CI run 35376242857 identifies AnyIO 4.12.1 vulnerabilities; SELF upgrades both locks to 4.14.2; scoped Ruff, file-size and context checks pass.
- **Summary:** Complete-record idempotence, same-observation parents, current selection and strict atomic provider snapshots with legacy reading. Renderer preserved.
- **Next step:** Validate PR #10450 CI and merge through branch protection.
- **Evidence:** docs/plans/shadow_tracker/TURNOVER_CURRENT.md; tests/unit/shadow_tracker/test_revision_persistence.py.

### DL-#10403 · OpenSim Package a Golf-Like Native Viewer and Release Evidence

- **State:** in_progress
- **Owner:** local
- **Issue:** #10403 (epic #10394 / #10363, OG-09)
- **Branch:** feat/og09-golf-native-viewer-package-10403
- **PR:** not created
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/view_package.py; src/engines/physics_engines/opensim/python/tour_matching/**init**.py; tests/opensim/test_golf_view_package.py; docs/development/DEVELOPMENT_LOG.md; docs/development/HANDOFF.md
- **Started:** 2026-09-18
- **Last verified:** 2026-09-18 at HEAD (SELF; TDD RED fixtures established; blank motion name raises InvalidMotionSpecificationError; invalid/inverted frame range raises InvalidMotionSpecificationError; missing club asset raises MissingClubAssetError; model/motion hash mismatch raises ModelMotionHashMismatchError; torque baseline truthfully sets muscles_available=False; muscle variant sets muscles_available=True; reset to address verifies bilateral grip closure <= 5 mm and canonical face-on viewpoint; scrub to time linearly interpolates coordinates across swing horizon; motion status explicitly distinguishes IK_PLAYBACK, REJECTED_REPLAY, and ACCEPTED_DYNAMIC; 4 canonical milestone stills generated [address, top, impact, finish]; reproducible video exported; deterministic SHA-256 package digest; all 12 view package tests pass; all 139 opensim unit tests pass; ruff, mypy, lod clean)
- **Summary:** Packaged native viewer artifacts and release evidence (`view_package.py`) for the OpenSim golf humanoid. Provides visual layer options, truth-in-advertising muscle toggles, swing milestone stills, reproducible video animation export, reset-to-address with grip closure verification, continuous scrubbing, launcher entry generation, and deterministic package hashing. Completes all 9 child issues of OpenSim epic #10394.
- **Next step:** Commit, push, open PR referencing Closes #10403, release lease, conclude OpenSim epic #10394.
- **Evidence:** tests/opensim/test_golf_view_package.py; src/engines/physics_engines/opensim/python/tour_matching/view_package.py.

### DL-#10402 · OpenSim Qualify Muscle and Tendon Extensions Without Replacing Baseline

- **State:** in_review
- **Owner:** local
- **Issue:** #10402 (epic #10394 / #10363, OG-08)
- **Branch:** feat/og08-qualify-muscle-tendon-extensions-10402
- **PR:** #10413
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/muscle_qualification.py; src/engines/physics_engines/opensim/python/tour_matching/**init**.py; tests/opensim/test_muscle_cmc.py; src/engines/physics_engines/opensim/python/POST_MVP_MUSCLES.md; docs/development/DEVELOPMENT_LOG.md; docs/development/HANDOFF.md
- **Started:** 2026-09-18
- **Last verified:** 2026-09-18 at HEAD (SELF; TDD RED fixtures established; lower-limb-only claiming full-body golf swing raises UnsupportedAnatomyClaimError; invalid parameters raise InvalidMuscleParameterError; invalid MTU path raises InvalidMusclePathError; uninitialized tendon state raises UninitializedTendonStateError; moment arm discrepancy with finite-difference path-length derivative raises MomentArmDerivativeMismatchError; continuous Hill tendon model equilibrates; receipt reports reserve actuator torques and pelvic residuals; torque baseline preserved; all 30 muscle tests pass; all 127 opensim tests pass; ruff, mypy, lod, file-budget clean)
- **Summary:** Qualified muscle and tendon extensions (`muscle_qualification.py`) without replacing the torque baseline. Provides explicit anatomy coverage audit (disallowing lower-extremity-only models from claiming full golf capability), parameter provenance and licensing audits, MTU path and wrapping verification, moment arm vs. finite-difference path-length derivative validation, activation dynamics, initial tendon equilibrium, and short replay receipts reporting reserve torques and pelvic residuals. Keeps epic muscle-complete status open pending independent #10375 validation.
- **Next step:** Land PR #10413 referencing Closes #10402.
- **Evidence:** tests/opensim/test_muscle_cmc.py; src/engines/physics_engines/opensim/python/tour_matching/muscle_qualification.py.

### DL-#10401 · OpenSim Introduce Versioned Model Variants and Actuation Capabilities

- **State:** in_review
- **Owner:** local
- **Issue:** #10401 (epic #10394 / #10363, OG-07)
- **Branch:** feat/og07-versioned-model-variants-10401
- **PR:** #10412
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/model_variants.py; src/engines/physics_engines/opensim/python/tour_matching/**init**.py; tests/opensim/test_golf_model_variants.py; docs/development/DEVELOPMENT_LOG.md; docs/development/HANDOFF.md
- **Started:** 2026-09-18
- **Last verified:** 2026-09-18 at HEAD (SELF; TDD RED fixtures established; loading torque controls into muscle variant raises IncompatibleActuationError; querying unknown state raises UnknownStateError; missing geometry mesh raises MissingGeometryAssetError; stale/mismatched model hash raises StaleModelHashError; unsupported capability raises UnsupportedCapabilityError; identity variant preserves forward kinematics; torque and muscle variants share identical GolfModelAdapter API; adapter shields callers from OpenSim C++ SDK objects; 8 variant tests pass; all 114 opensim unit tests pass; ruff, mypy, lod, file-budget clean)
- **Summary:** Implemented versioned OpenSim model variants and explicit actuation capabilities (`model_variants.py`) via composition: AnatomicalSkeletonSpec + GolfEquipmentSpec + Calibration + ActuationProfile (torque vs. muscle/tendon). Provides typed error boundaries and a clean adapter API without SDK leakage.
- **Next step:** Land PR #10412 referencing Closes #10401.
- **Evidence:** tests/opensim/test_golf_model_variants.py; src/engines/physics_engines/opensim/python/tour_matching/model_variants.py.

### DL-#10400 · OpenSim Rebuild Full-Swing Tracking From Qualified Address

- **State:** in_review
- **Owner:** local
- **Issue:** #10400 (epic #10394 / #10363, OG-06)
- **Branch:** feat/og06-rebuild-full-swing-tracking-10400
- **PR:** #10410
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/full_swing_tracking.py; src/engines/physics_engines/opensim/python/tour_matching/**init**.py; tests/opensim/test_moco_g1_ladder.py; docs/development/DEVELOPMENT_LOG.md; docs/development/HANDOFF.md
- **Started:** 2026-09-18
- **Last verified:** 2026-09-18 at HEAD (SELF; TDD RED fixtures established; model checkpoint verification raises ModelCheckpointMismatchError; truncated capture claim raises TruncatedCaptureClaimError; control/state naming mismatch raises ControlStateNamingMismatchError; dynamic bilateral grip violation raises DynamicGripViolationError; continuity violation raises ContinuityViolationError; ladder stages from static address through G1, G2, G3 full capture; distinct statuses for IK playback, solver convergence, and replay acceptance; separate tracking and forward replay receipts; 9 ladder tests pass; all 106 opensim unit tests pass; ruff, mypy, lod, file-budget clean)
- **Summary:** Rebuilt OpenSim full-swing tracking qualification module (`full_swing_tracking.py`) reinitialized from qualified address pose $q_0$ and model hash with frozen marker calibration. Implements ladder progression, continuity, full-swing coordinate limits, dynamic bilateral grip closure, ground contact mechanics, and separate receipts under MS-100 / MS-104.
- **Next step:** Land PR #10410 referencing Closes #10400.
- **Evidence:** tests/opensim/test_moco_g1_ladder.py; src/engines/physics_engines/opensim/python/tour_matching/full_swing_tracking.py.

### DL-#10399 · OpenSim Calibrate and Match Two-Handed Address Pose

- **State:** in_review
- **Owner:** local
- **Issue:** #10399 (epic #10394 / #10363, OG-05)
- **Branch:** feat/og05-match-two-handed-address-10399
- **PR:** #10409
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/address.py; src/engines/physics_engines/opensim/python/tour_matching/marker_calibration.py; src/engines/physics_engines/opensim/python/tour_matching/**init**.py; tests/opensim/test_golf_address.py; tests/opensim/test_marker_calibration.py; SPEC.md
- **Started:** 2026-09-18
- **Last verified:** 2026-09-18 at HEAD (SELF; TDD RED fixtures established; frozen acceptance profile hash verified; quasi-static address window detection; dual-arm grip closure <= 5 mm; ground support clearance <= 15 mm; coordinate range limits audited; holdout validation; all 15 address and calibration tests pass; ruff, mypy, lod, file-budget clean)
- **Summary:** Calibrated and qualified two-handed golf address pose (`address.py`) on OpenSim humanoid model against canonical tour capture. Validates bilateral grip closure between lead hand and club shaft, foot ground support, posture metrics, and coordinate limits against model XML ranges.
- **Next step:** Land PR #10409 referencing Closes #10399.
- **Evidence:** tests/opensim/test_golf_address.py; tests/opensim/test_marker_calibration.py.

### DL-#10398 · OpenSim Capture Registration and Golf Camera Views

- **State:** in_review
- **Owner:** local
- **Issue:** #10398 (epic #10394 / #10363, OG-04)
- **Branch:** feat/og04-qualify-registration-camera-10398
- **PR:** #10408
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/registration.py; src/engines/physics_engines/opensim/python/tour_matching/visualization.py; src/engines/physics_engines/opensim/python/tour_matching/**init**.py; tests/opensim/test_golf_registration.py; SPEC.md
- **Started:** 2026-09-18
- **Last verified:** 2026-09-18 at HEAD (SELF; 3D rigid transform with Kabsch SVD and proper rotation constraint det(R)==+1.0; round-trip identity error <= 1e-8 m; stance ground support registration to Y=0 and target line yaw alignment to +X; golf camera view presets FRONT_VIEW, SIDE_VIEW, DOWN_THE_LINE, OVERHEAD; camera viewpoint adjustments proven invariant over model states and kinematic metrics; 30 unit tests pass; ruff, mypy, lod, file-budget clean)
- **Summary:** Implemented capture registration and qualified golf camera viewpoints (`registration.py`). Provides rigid landmark alignment without scaling or shearing, ground support plane alignment, and down-the-line / front / side / overhead camera views in `visualization.py`. Validated invariant over model states and simulation outputs.
- **Next step:** Land PR #10408 referencing Closes #10398.
- **Evidence:** tests/opensim/test_golf_registration.py.

### DL-#10396 · OpenSim Anatomically and Physically Consistent Segment Scaling

- **State:** in_review
- **Owner:** local
- **Issue:** #10396 (epic #10394 / #10363, OG-02)
- **Branch:** feat/og02-consistent-segment-scaling-10396
- **PR:** #10407
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/segment_scaling.py; src/engines/physics_engines/opensim/python/tour_matching/scale.py; src/engines/physics_engines/opensim/python/tour_matching/**init**.py; src/engines/physics_engines/opensim/models/golf_humanoid_scaled.osim; docs/development/opensim_tour_matching/os3b_scale_and_full_ik.py; tests/opensim/test_segment_scale.py; tests/opensim/test_opensim_os0_qualification.py; SPEC.md
- **Started:** 2026-09-18
- **Last verified:** 2026-09-18 at HEAD (SELF; TDD RED fixtures established; acromion proxy reconstruction removes artificial humerus inflation bringing bilateral ratio to 1.0807; apply_segment_scaling scales joint frames, bone meshes, COM, and inertia under fixed_mass and density_preserving policies; repeat scaling protection verified; 80 unit tests passed; ruff, mypy, lod, file-budget clean)
- **Summary:** Implemented consistent segment scaling module (`segment_scaling.py`) and acromion proxy reconstruction for unilateral marker occlusions (`scale.py`). Joint frames, visual meshes, COM, and inertia are scaled in lockstep. Regenerated qualified `golf_humanoid_scaled.osim` with attached visual club and consistent scaling, passing qualification gates without unscaled arm mesh defects.
- **Next step:** Land PR #10407 referencing Closes #10396.
- **Evidence:** tests/opensim/test_segment_scale.py; tests/opensim/test_opensim_os0_qualification.py; src/engines/physics_engines/opensim/models/golf_humanoid_scaled.osim.

### DL-#10397 · OpenSim Visible Parameterized Golf Club and Grip Frames

- **State:** in_review
- **Owner:** local
- **Issue:** #10397 (epic #10394 / #10363, OG-03)
- **Branch:** feat/og03-visible-golf-club-10397
- **PR:** #10405
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/club_geometry.py; src/engines/physics_engines/opensim/python/tour_matching/**init**.py; tests/opensim/test_golf_club_geometry.py; SPEC.md
- **Started:** 2026-09-18
- **Last verified:** 2026-09-18 at HEAD (SELF; TDD RED fixtures established; parameterized club visual geometry generator implemented; shaft and head mesh elements attached with ClubSpec scaling; mass, com, and inertia preserved; grip and clubhead offset frames aligned with OpenSim conventions; 13 pure unit tests passed; ruff, mypy, lod, file-budget clean)
- **Summary:** Added parameterized visual geometry attachment (`attach_visual_club`) and frame calculation (`get_club_frame_offsets`) for the OpenSim Club body according to shared ClubSpec (Driver and 7-iron). Solves the missing visual club defect while strictly preserving physical mass properties (0.320 kg, COM, inertia). Fully verified against anatomical baseline fixtures and qualification gates.
- **Next step:** Land PR #10405 referencing Closes #10397.
- **Evidence:** tests/opensim/test_golf_club_geometry.py; tests/opensim/test_anatomical_baseline_fixtures.py.

### DL-#10395 · OpenSim Anatomical Baseline Freeze and Qualification Fixtures

- **State:** in_review
- **Owner:** local
- **Issue:** #10395 (epic #10394 / #10363, OG-01)
- **Branch:** feat/og01-freeze-anatomical-baseline-10395
- **PR:** #10404
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/model_audit.py; src/engines/physics_engines/opensim/python/tour_matching/**init**.py; src/engines/physics_engines/opensim/python/tour_matching/cli.py; tests/opensim/test_anatomical_baseline_fixtures.py; tests/opensim/test_opensim_os0_qualification.py
- **Started:** 2026-09-18
- **Last verified:** 2026-09-18 at 94d593cf1 (SELF; pure model audit implemented test-first; baseline SHA-256 digests and structural counts verified; missing club and unscaled mesh defects reproduced with RED fixtures; fail-closed qualification gates verified; 22 pure tests passed; ruff, mypy, lod, file-budget clean)
- **Summary:** Delivered pure-Python OpenSim model geometry and structural qualification audit under OG-01. Verifies SHA-256 digests against pinned baseline models, audits body/coordinate/actuator counts (23/39/39/0), detects empty attached_geometry on equipment bodies (Club body) and unscaled arm meshes (scale factors 1 1 1 on humerus). Implements fail-closed `verify_model_qualification` gate with DbC assertions and structured receipts via `cli.py qualify`.
- **Next step:** Land PR #10404 referencing Closes #10395.
- **Evidence:** tests/opensim/test_anatomical_baseline_fixtures.py; tests/opensim/test_opensim_os0_qualification.py; docs/development/opensim_tour_matching/evidence/anatomical_review_20260918/inspection.json.

### DL-#10323 · Matched-Swing Run Ledger

- **State:** in_progress
- **Owner:** local
- **Issue:** #10323 (epic #10363 MS-02)
- **Branch:** feat/ms02-matched-swing-run-ledger-10323
- **PR:** open
- **Paths:** src/shared/python/motion_matching/ledger.py; src/shared/python/motion_matching/ledger_schema.py; src/shared/python/motion_matching/**main**.py; src/shared/python/motion_matching/leaderboard.py; src/tools/motion_matching/pipeline.py; reports/matched_swing_ledger.json; tests/unit/motion_matching/test_ledger.py
- **Started:** 2026-09-17
- **Last verified:** 2026-09-17 at HEAD (SELF; scan discovers and classifies all 85 committed receipts across evidence roots; tests/unit/motion_matching/test_ledger.py 7 passed; architecture budget OK, ruff and ruff format clean)
- **Summary:** Added matched-swing run ledger discovering, classifying, and indexing execution receipts across ground-support, native Simscape, OpenSim, calibration, and parity evidence trees. Deterministic serialization into reports/matched_swing_ledger.json, CLI subcommand `ledger --write`, and `list_runs()` API for tools.
- **Next step:** Open PR referencing Closes #10323, enable auto-merge.
- **Evidence:** reports/matched_swing_ledger.json; tests/unit/motion_matching/test_ledger.py

### DL-#10362 · Tools Dependency Gate for Matched Swing Program

- **State:** in_progress
- **Owner:** local
- **Issue:** #10362 (epic #10363)
- **Branch:** feat/ms95-tools-dependency-gate-10362
- **PR:** open
- **Paths:** vendor/ud-tools; Cargo.toml; requirements-tools.txt; docs/shared_tools/divergence_inventory.md; docs/shared_tools/divergence_inventory.v1.json; docs/agent_context/README.md; docs/agent_context/index.html
- **Started:** 2026-09-17
- **Last verified:** 2026-09-17 at 62e8cdbf9 (SELF; Tools main green, Tools #4494 and #4262 closed, Tools #5227 landed; four-way pin bumped to 62e8cdbf9; divergence inventory regenerated; agent context verified; test_no_shadow_of_tools_shared and run_checks pass)
- **Summary:** UpstreamDrift ownership of humanoid_character_builder and model_generation ruled per Tools #4494; Tools #4262 and #4494 closed; vendor/ud-tools, requirements-tools.txt, Cargo.toml, and divergence inventory repinned to Tools main 62e8cdbf9.
- **Next step:** Open PR referencing Closes #10362, enable auto-merge, release lease.
- **Evidence:** docs/shared_tools/divergence_inventory.v1.json; tests/unit/repo_hygiene/test_no_shadow_of_tools_shared.py; tests/fixtures/reference_calibration/run_checks.py.

### DL-#10334 · Versioned Matched Swing Candidate (MS-15)

- **State:** in_progress
- **Owner:** claude
- **Issue:** #10334 (MS-15, epic #10363)
- **Branch:** feat/10334-matched-swing-candidate
- **PR:** not created
- **Paths:** src/shared/python/motion_matching/candidate.py; src/shared/python/motion_matching/candidate_io.py; src/shared/python/motion_matching/candidate_convert.py; src/tools/tour_matching_viewer/core.py; src/shared/python/motion_matching/cross_engine_replay.py; docs/development/full_body_models/CANDIDATES.md; tests/unit/motion_matching/test_candidate.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (SELF; candidate package defined with kinematic and dynamic profiles, immutable arrays, and SHA-256 tamper-proof checksumming; lossless converters for returned81 replays, OpenSim MOT files, and ground-support IK/dynamics; 24 candidate and viewer unit tests pass 100%; architecture budget, ruff, mypy clean)
- **Summary:** Implemented unified MatchedSwingCandidate versioned package (matched-swing-candidate-v1) supporting distinct kinematic and dynamic profiles, full-body generalized coordinates and tangent velocities (nq != nv), virtual work power consistency verification, immutable arrays, and SHA-256 array tamper detection. Implemented converters for legacy returned81 replays and OpenSim MOT files, updated viewer loader, and authored complete CANDIDATES.md schema document.
- **Next step:** Run CI checks, commit, push, open PR referencing Closes #10334, enable auto-merge, release lease.
- **Evidence:** docs/development/full_body_models/CANDIDATES.md; tests/unit/motion_matching/test_candidate.py; docs/development/full_body_models/evidence/replays/mujoco_returned81_candidate.npz; docs/development/full_body_models/evidence/viewer/opensim_os3b_candidate.npz.

### DL-#10339 · Pure-XML OpenSim Full-Body Exporter (MS-40)

- **State:** shipped
- **Owner:** claude
- **Issue:** #10339 (MS-40, epic #10363)
- **Branch:** feat/10339-full-body-osim
- **PR:** #10474 (merged)
- **Paths:** src/engines/physics_engines/opensim/python/full_body_osim.py; src/engines/physics_engines/opensim/models/generated/full_body_anthro_driver.osim; src/engines/physics_engines/opensim/models/generated/full_body_anthro_iron7.osim; src/engines/physics_engines/opensim/models/generated/export_receipt.json; src/engines/physics_engines/opensim/models/README.md; tests/unit/motion_matching/test_full_body_osim.py; tests/opensim/test_full_body_osim_native.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (SELF; pure XML exporter generates 44-coordinate full-body .osim models from anthro specs with dual-grip weld closure, foot contact spheres, and 34 tour markers; unit tests pass without OpenSim SDK; native test tests/opensim/test_full_body_osim_native.py skips cleanly when opensim is not installed; architecture budget, ruff, mypy clean)
- **Summary:** Implemented pure-XML ElementTree OpenSim exporter producing 44-coordinate full-body models (full_body_anthro_driver.osim and full_body_anthro_iron7.osim) from anthropometric specs without OpenSim runtime dependencies. Enforces 6-DOF dual-grip weld closure, Hunt-Crossley compliant foot contact spheres, 38 internal coordinate actuators, and 34 tour marker attachments with hash-verified provenance receipt.
- **Next step:** Completed; PR #10474 merged into main.
- **Evidence:** src/engines/physics_engines/opensim/models/generated/export_receipt.json; tests/unit/motion_matching/test_full_body_osim.py; tests/opensim/test_full_body_osim_native.py.

### DL-#10352 · Shared Contact Law and Grip Closure Conformance (MS-72)

- **State:** shipped
- **Owner:** claude
- **Issue:** #10352 (MS-72, epic #10363)
- **Branch:** feat/10352-contact-closure-conformance
- **PR:** #10465 (merged)
- **Paths:** src/shared/python/motion_matching/contact_law.py; docs/development/matched_swing_program/CONTACT_CLOSURE_CONFORMANCE.md; tests/integration/cross_engine/test_contact_closure_conformance.py; tests/integration/cross_engine/divergence_registry.yaml
- **Started:** 2026-09-18
- **Last verified:** 2026-09-18 at 01c5ef0c6 (SELF; 10 contact closure conformance tests pass; frontmatter tolerance loading verified; ruff and mypy clean)
- **Summary:** Formalized shared contact law (Hunt-Crossley compliant normal force with non-tensile clipping and regularized friction) and 6-DOF dual-grip spatial weld closure contracts across engines. Versioned via CONFORMANCE_VERSION 1.0.0 and registered divergences in divergence_registry.yaml.
- **Next step:** Completed; PR #10465 merged into main.
- **Evidence:** tests/integration/cross_engine/test_contact_closure_conformance.py; docs/development/matched_swing_program/CONTACT_CLOSURE_CONFORMANCE.md.

### DL-#10381 · Pinocchio G1 Qualification and Program Truth Reset (MS-107)

- **State:** in_progress
- **Owner:** claude
- **Issue:** #10381 (epic #10363)
- **Branch:** evidence/10381-g1-b100
- **Paths:** src/shared/python/motion_matching/ledger.py; evidence/matched; docs/development/full_body_models/evidence/acceptance/verdicts_2026-09.json; docs/development/matched_swing_program
- **Started:** 2026-09-18
- **Last verified:** 2026-09-19 (SELF; barrier-reduced continuation converged at 46.8 mm replay-identical, committed as rejected evidence; ledger regenerated)
- **Summary:** Ledger is fail-closed. Same-integrator continuation (rtol 1e-6) gave 123.5 mm, barrier-dominated; with range-barrier weight 100 and raised trail-side effort bounds (now defaults, `--range-barrier-weight` flag) the 0.85 s stage converges at 46.8 mm whole, rollout == replay, 1.49 BW. Still REJECTED at G1 (early 25.7, terminal 64.3, yaw 9.3 deg, penetration 17 mm, weight-fraction floor).
- **Next step:** From `stage_0.85s.npz`: 300 more iterations; terminal/club weight x3; MS-20 contact identification for the 17 mm penetration; pelvis-yaw cost term. One receipt per lever.

### DL-#10338 · Native Crocoddyl Full-Body Fit & Balanced Contact Kinetics (Matched Swing Program MS-31 / #10415)

- **State:** shipped
- **Owner:** claude
- **Issue:** #10338 (epic #10363, child epic #10415)
- **Branch:** feat/10338-crocoddyl-native-fit
- **PR:** #10411
- **Paths:** src/engines/physics_engines/pinocchio/python/{crocoddyl_problem,crocoddyl_action,marker_kinematics,full_body_fit}.py; src/shared/python/motion_matching/{contact_force_allocator,swing_evaluator}.py; scripts/match_pinocchio_c3d.py; tests/unit/motion_matching/{test_match_pinocchio_c3d,test_contact_force_allocator,test_swing_evaluator}.py; docs/development/PINOCCHIO_C3D_MOTION_MATCHING_GUIDE.md; evidence/matched/{driver_full_pinocchio,iron_full_pinocchio}
- **Started:** 2026-09-17
- **Last verified:** 2026-09-18 (SELF; MS-31 closed on implementation scope via #10371/#10411; qualification continues under DL-#10381)
- **Summary:** Solved full-swing (654-frame driver, 657-frame 7-iron) decoupled kinematic tracking via `MarkerIkSolver` with category weighting (club 50x, feet 20x), analytical foot non-penetration barrier, and constant-velocity extrapolation prior. Replaced algebraic trail-zero overwrite with rigorous QP-based `ContactForceAllocator` satisfying $M \ddot{q} + b = S^T \tau + J_{\text{ground}}^T f + J_{\text{grip}}^T \lambda + S_{\text{root}}^T \delta \tau_{\text{root}}$ under unilateral contact ($f_z \ge 0$) and exact dynamic equilibrium. Built `SwingEvaluator` for audit-grade segment and phase reporting. Driver club RMSE drops from 425.3 mm to 50.2 mm (G1 gate <= 60 mm met); max foot penetration drops from 111.2 mm to 10.1 mm; ABA acceleration parity residual verified to 0.00155 m/s². Continuous forward simulation replay verified stable without pose resets. Artifacts committed under `evidence/matched/driver_full_pinocchio/` and `evidence/matched/iron_full_pinocchio/`.
- **Next step:** None here; continue in DL-#10381.

### DL-#9967 · Native Simscape Tour Matching

- **State:** in_progress
- **Owner:** codex (turnover review; execution ownership by next lease)
- **Issue:** #9967 (parent #9921)
- **Branch:** feat/9967-native-simscape-pinocchio
- **Paths:** src/shared/python/motion_matching; docs/development/simscape_tour_matching
- **Started:** 2026-09-09
- **Last verified:** 2026-09-15 (SELF; raw run101 MAT/NPZ metrics independently recomputed; seven focused yaw/replay tests passed)
- **Summary:** Run101 improves yaw and has measured R2025b–Pinocchio prefix agreement of 0.0605 mm maximum. Terminal RMS 40.31 mm fails the 35 mm gate; full 1.814 s capture is incomplete. No optimizer launched by this review.
- **Next step:** Follow RUN101_REVIEW_AND_TURNOVER.md: bounded refinement check, head/left-arm terminal feasibility, justified bounded fitting trial and horizon extension.

### DL-#10204 · Capture Rig Shared Camera Layer

- **State:** in_review
- **Owner:** claude
- **Issue:** #10204 (part of D-sorganization/Tools#5218)
- **Branch:** claude/10204-shared-camera
- **PR:** #10211 (open; auto-merge squash armed)
- **Paths:** src/motion_capture/rig/preview_source.py; src/motion_capture/rig/recorder.py; tests/fixtures/reference_calibration/preview_source_checks.py; vendor/ud-tools; requirements-tools.txt; Cargo.toml; docs/shared_tools/divergence_inventory.v1.json; docs/shared_tools/seam_rulings.v1.json; docs/agent_context
- **Started:** 2026-09-15
- **Last verified:** 2026-09-15 (SELF; seam-drift and agent-context gates pass locally; `run_checks.py` 38 passed incl. 8 adapter checks; `tests/motion_capture/rig` + shadow/fallback hygiene 140 passed; `check_tools_pins` consistent at 1ac89c18e; ruff, ruff-format, mypy clean on changed files)
- **Summary:** Pin `vendor/ud-tools` to Tools 1ac89c18e, replace the rig's own ffmpeg preview decode with one adapter over Tools `shared.python.camera.FfmpegDirectShowSource` (seam points rig → Tools), delegate `dshow_device_ref` to the shared builder, and pin the launched command token-for-token against the pre-port list. The shared package is a `sidekick.lab.mocap` consumer, so it imports at launcher runtime and in the isolated provider harness, not in the root test process.
- **Next step:** Operator verifies on the rig that the preview binds all three cameras at 60 fps and Record still hands off, then merges the PR.
- **Evidence:** tests/fixtures/reference_calibration/preview_source_checks.py; scripts/shared_tools/check_tools_pins.py.

### DL-#10188 · Model-Driven Golf Simulator Integration

- **State:** in_progress
- **Owner:** local
- **Issue:** #10188 (child #10200 active; children #10189–#10200)
- **Branch:** feat/issue-10200-native-avatar-course-feedback
- **PR:** #10227 (merged; GS-10 #10199); #10226 (merged; GS-09 #10198); #10225 (merged; GS-08 #10197); #10222 (merged; GS-07 #10196); #10220 (merged; GS-06 #10195); #10217 (merged; GS-05 #10194); #10215 (merged; GS-04 #10193); #10213 (merged; GS-03 #10192); #10208 (merged; GS-00 #10189, GS-01 #10190, GS-02 #10191); #10201 (merged; planning)
- **Paths:** `docs/plans/golf_simulator_integration/NATIVE_AVATAR_COURSE_FEEDBACK_RESEARCH.md; src/shared/python/golf_simulator/contracts.py; src/shared/python/golf_simulator/__init__.py; tests/unit/golf_simulator/test_avatar_course_feedback_research.py`
- **Started:** 2026-09-15
- **Last verified:** 2026-09-16 (SELF; GS-00 through GS-11 implemented test-first; all 100 golf_simulator unit tests pass locally, verified under python -O; ruff, black, mypy, and architecture budgets clean)
- **Summary:** GS-11 (#10200) Native GSPro model animation and autonomous course feedback research delivered. Documents Unity runtime constraints and confirms Course Designer produces static AssetBundles with zero dynamic skeletal mesh hooks; confirms Open Connect v1 is strictly unidirectional shot input lacking ball landing, lie, surface, wind, elevation, hazard, or aim feedback; prohibits memory scraping / DLL injection / packet sniffing; establishes turnkey vendor inquiry templates; formalizes Synchronized Companion Presentation Architecture; implements `UnsupportedCapabilityError` and `assert_capability_supported()` in `contracts.py`.
- **Next step:** Commit, open PR referencing Closes #10200, merge via auto-squash, and close parent Epic #10188.
- **Evidence:** tests/unit/golf_simulator/test_avatar_course_feedback_research.py; docs/plans/golf_simulator_integration/NATIVE_AVATAR_COURSE_FEEDBACK_RESEARCH.md.

### DL-#10003 · OpenSim Tour-Average Full-Body Matching

- **State:** in_progress
- **Owner:** claude
- **Issue:** #10003 (parent #9921; sibling full-body epic #10062)
- **Branch:** feat/full-body-opensim-epic
- **PR:** #10071
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching; src/shared/python/motion_matching/tour_capture_contract.py; tests/opensim; docs/development/opensim_tour_matching
- **Started:** 2026-09-12
- **Last verified:** 2026-09-13 (SELF; 17 unit tests in tests/opensim pass locally with Ruff; OpenSim 4.6 runtime qualified on ControlTower; OS-3 IK on 33 frames reaches 6.5 cm marker RMS, unscaled model)
- **Summary:** Frozen capture contract, marker-to-body map, TRC export verified by OpenSim, MarkerSet authoring with coordinate unlocking, and alternating placement/IK calibration are implemented test-first; runtime qualified; OS-3 kinematic feasibility measured. Moco tracking and sextic effort fitting not started.
- **Next step:** Execute OS-2b/OS-3b from docs/development/opensim_tour_matching/NEXT_AGENT_PROMPT.md: golf model variant in the builder (unlocks, clamp ranges, club length), segment scaling, keep-best iteration, full 654-frame IK with per-frame RMS and overlay.
- **Evidence:** docs/development/opensim_tour_matching/HANDOFF.md and evidence/os1_trc_receipt.json, os2_runtime_receipt.json, os3_stride20/, os3_unlocked_stride20/.

### DL-#10062 · Full-Body Models With Lower Limbs and Ground Contact

- **IK & Forward Dynamics Consolidation (MS-11 #10330; 2026-09-17):**
  Consolidated ground-support marker IK and forward dynamics into shared modules, retiring
  `src/engines/physics_engines/mujoco/python/full_body_markers.py` (813 -> 44 lines) and
  `full_body_simulation.py` (709 -> 55 lines) into backward-compatible deprecation shims
  and eliminating 1,423 duplicate lines across engine files. Shared solver and dataclasses live
  in `src/shared/python/motion_matching/full_body_ik.py` (`BaseFullBodyIK`) and
  `src/shared/python/motion_matching/full_body_forward_dynamics.py` with zero MuJoCo imports in
  shared motion matching. MuJoCo adapter preserved as subclass in
  `src/engines/physics_engines/mujoco/python/full_body_ik.py`. Added comprehensive consolidation
  test suite `tests/unit/motion_matching/test_full_body_consolidation.py`.

- **Pink Displaced Targets & Both-Club Smoke Qualification (2026-09-17):**
  Added tests verifying that reachable displaced marker targets produce nonzero motion
  and measurable residual reduction, infeasible hard constraints fail qualification closed
  with structured failure reasons, and both driver and 7-iron smoke journeys produce valid
  conforming receipts through `MatchRequest` and `ConstrainedIkReceipt`.

- **Pink Integration Defect Repair & Fail-Closed Qualification (#10318; 2026-09-17):**
  Repaired three blocking defects in the Pink constrained IK pipeline:

  1. Driver interface: Replaced nonexistent module reference in `run_ground_support.py`
     with `PinkTrajectoryService`, `IKTrajectoryRequest`, and `IKOptions`, generating
     conforming `ConstrainedIkReceipt` records.
  2. Fail-closed native execution: Eliminated silent no-op fallback in
     `PinkTrajectoryService._execute_qp_step`; missing native stack or model fails closed
     with explicit failure reasons.
  3. Honest constraint evaluation: Enforced `NaN` residuals for unevaluated marker and
     weld constraints, and gated qualification on rate limit violations and named
     physical tolerances (`weld_translation_tolerance_m`, `weld_rotation_tolerance_rad`,
     `marker_tolerance_m`).

- **Pink Pipeline & Receipt Exposure (#10278; 2026-09-17):** Exposed Pink
  constrained inverse kinematics backend through `MatchRequest` (`backend="pink"`,
  `step_mode`, `solver`), CLI `--backend pink`, and GUI dropdown. Structured
  diagnostics and provenance recorded via `ConstrainedIkReceipt` in receipt schema.
  Fail-closed capability probe `probe_pink_capability()` prevents silent fallback.
  Regenerated `RECEIPTS.md`.

- **Viewer Adapters (#10256; 2026-09-16):** Replaced no-op display paths with
  persistent Pinocchio visualizers, explicit validation and scoped cleanup.
  Real MeshCat probing exposed and corrected shared-root deletion and owned
  process cleanup defects. Live Gepetto qualification and production replay
  integration remain outstanding under #10254.

- **Runtime Qualification (#10262; 2026-09-16):** Runtime slice #10262 adds a consistent conda-forge numerical manifest and exact
  Linux lock plus isolated capability probes. Receipt success is scoped to
  runtime behavior; model, full-body fitting and renderer acceptance stay open.

- **Pink Adapters (#10257; 2026-09-16):** Shared solve path validates state,
  time and outputs; forwards hard constraints/limits; retains collision
  geometry; refreshes cached FK and propagates solver failures. Real native
  contracts include free-flyer dimensions and infeasible equality/limit
  combinations. Full-body task assembly and runtime packaging (#10262) remain
  separate; no fitted trajectories were regenerated.

- **LoD Regression (#10254; 2026-09-16):** Reproduced the main-derived
  `inputs.calibration2.offsets.items()` architecture failure before the change.
  Resolve the owned offset mapping once before formatting the report; preserve
  explicit/calibrated/attachment precedence. Seven reference-stage tests and
  the full 3,221-file LoD no-growth scan pass; the baseline was not changed.

- **Integration Turnover (#10254; 2026-09-16):** Post-compaction Crocoddyl ABI
  and Pink feasible/infeasible QP probes pass on the preserved WSL environment.
  Added bounded TDD/DbC/LoD/DRY worker contracts and three Pink pipeline packets.
  Main authority is 0ec64e45; #10250/#10251 are closed. Native action assembly,
  production Pink integration, CI and full physical qualification remain open.
  See `docs/plans/qualified_motion_integration/TURNOVER.md`.

Slice #10265 preserves global degree-six controls with a checked unactuated
root mapping and exact RK4 state/coefficient sensitivities. It is a prerequisite
to coefficient-lift Crocoddyl actions; optimizer and physical acceptance remain
open. Preserve explicit ground configuration in independent replay.

- **Crocoddyl Actions (#10269; 2026-09-16):** Implemented the lift/flow/terminal
  models, exact RK4 chain-rule derivatives, cost scaling, coefficient bounds, warm starts
  and independent replay diagnostics. Cases tested in isolation include real FDDP, BoxFDDP
  bounded polynomial effort and full-body active/offground contact with canonical/reversed
  coordinates.

- **Weld Linearization (#10260; 2026-09-16):** Corrected the finite weld pose
  Jacobian and preserved the distinct acceleration-constraint partial. Real
  Pinocchio directional checks fail before correction for displaced wrists;
  all 11 closure and 11 contact integration tests pass afterward, including
  explicit rejection of undefined derivatives at the rotation-pi log branch. Next:
  merge numerical prerequisites before constrained Pink task assembly.

- **Solver Integration (#10254, #10255; 2026-09-16):** Reproduced missing
  ground-contact state terms in inherited Pinocchio acceleration derivatives;
  added exact local contact-force partials and constrained chain-rule composition.
  Real Pinocchio 3.8 integration tests cover active/no contact, moving joints,
  reversed coordinate order, nonfinite input, contact kinks and cache isolation.
  Plan: `docs/plans/qualified_motion_integration/README.md`. Next: review/merge
  derivative boundary, then integrate #10257 Pink and #10256 viewer adapters;
  preserve #10250 evidence refresh and all full-horizon qualification gates.

- **State:** in_progress
- **Owner:** local
- **Issue:** #10062 (children #10063 to #10070); continued by epic #10162 (MM-1 to MM-10, HO-1 to HO-10)
- **Branch:** refactor/10251-pipeline-stages-run-ground-support
- **PR:** #10092 (FB-5, #10069); #10090 (Step 3 merged); #10089 (FB-4, #10068 merged); #10203 (Step 4 cross-engine replays merged); #10218 (HO-1 #10155 merged); #10224 (HO-2 #10156 merged); #10235 (HO-7 #10161 merged); #10236 (HO-4 #10158 merged); #10249 (HO-9 #10111 merged); #10228 (HO-3 #10157 merged); #10261 (HO-11 #10250 merged); #10258 (HO-8 #10108 merged)
- **Paths:** docs/development/full_body_models; src/shared/python/motion_matching/full_body_spec.py; src/shared/python/motion_matching/contact_law.py; src/shared/python/motion_matching/tour_capture_contract.py; src/shared/python/motion_matching/marker_calibration.py; src/shared/python/motion_matching/full_body_ik.py; src/shared/python/motion_matching/visual_skeleton.py; src/shared/python/motion_matching/derivative_resolution.py; src/shared/python/motion_matching/full_body_forward_dynamics.py; src/shared/python/motion_matching/anthropometry.py; src/shared/python/motion_matching/hip_calibration.py; src/shared/python/motion_matching/pipeline; src/tools/motion_matching; tests/unit/motion_matching/pipeline; tests/unit/motion_matching; tests/unit/tools; tests/tools/motion_matching; scripts/config/mjx_env_pins.json; scripts/setup_mjx_env.ps1; scripts/setup_mjx_env.sh; tests/unit/motion_matching/test_document_freshness.py
- **Started:** 2026-09-13
- **Last verified:** 2026-09-17 (SELF; MS-03 #10324 reconciled headline tour numbers with primary receipts on main, established canonical calibrated reference runs vs baselines, added CANONICAL_RUN.md and bisect_receipt.json; MS-06 #10327 delivered matched swing program tracker docs, physical acceptance ladder, waves plan, status generator script, freshness tests, and retired stale claims; MS-11 #10330 finished HO-1 consolidation into shared modules, retiring shims with -1,423 lines; tests/unit/motion_matching/test_full_body_consolidation.py passed)
- **Summary:** Full-body pipeline handoff (epic #10162, matched-swing program epic #10363). MS-03 (#10324) reconciled headline tour numbers with primary receipts on main with bisect attribution in `CANONICAL_RUN.md` and `bisect_receipt.json`. MS-06 (#10327) created `docs/development/matched_swing_program/README.md`, `GATES.md`, and `WAVES.md`, delivered `scripts/generate_matched_swing_status.py` rendering the cross-engine status matrix from `reports/matched_swing_ledger.json`, marked legacy stale parity documents with dated `SUPERSEDED` banners, and added `tests/docs/test_matched_swing_status_freshness.py`. MS-11 (#10330) consolidated ground-support marker IK and forward dynamics into shared modules, retiring duplicate implementations to backward-compatible deprecation shims.
- **Next step:** Commit MS-11 (#10330), open PR, auto-merge, and proceed to next Wave 1 task (MS-10 #10329).
- **Evidence:** docs/development/full_body_models/evidence/ground_support/CANONICAL_RUN.md, docs/development/full_body_models/evidence/ground_support/bisect_receipt.json, tests/unit/motion_matching/test_handoff_numbers_match_receipts.py, tests/unit/motion_matching/test_full_body_consolidation.py, docs/development/matched_swing_program/README.md, docs/development/matched_swing_program/GATES.md, docs/development/matched_swing_program/WAVES.md, scripts/generate_matched_swing_status.py, tests/docs/test_matched_swing_status_freshness.py.

### DL-#8766 · Unit-Test-Gate Debt Ledger Burndown

- **State:** in_progress
- **Owner:** antigravity
- **Issue:** #8766
- **Branch:** fix/8766-burndown-launcher-67
- **PR:** #10035
- **Paths:** scripts/config/unit_gate_quarantine.json, SPEC.md, docs/development/DEVELOPMENT_LOG.md, tests/launchers/test_golf_launcher.py, docs/agent_context/
- **Started:** 2026-09-12
- **Last verified:** 2026-09-12 (`f712806df`)
- **Summary:** Progressive burndown of the quarantine ledger (#8766). Prior tranches retired 43 packaging/governance tests (#10010), 11 deployment tests (#10012), 57 bunker shot and API route tests (#10013), 29 shared Python / physics tests (#10015), 16 AI adapter / launcher tests (#10026), 20 safe launcher / pipeline / model sources tests (#10031), 13 CORS tests (#10033), and 32 security and module docstring tests (#10034). This tranche burns down 67 quarantined tests across tests/launchers/test_golf_launcher.py (25), tests/launchers/test_launcher_ui_setup.py (21), tests/launchers/test_launcher_process_manager.py (17), and tests/launchers/test_library_widget.py (4), ratcheting debt down from 298 to 231.
- **Next step:** Open PR, monitor CI checks, and merge.

## Shipped (Last 90 Days)

### DL-#8875 · Motion Pipeline Formats Documentation Reconcile

- **State:** shipped
- **Owner:** antigravity
- **Issue:** #8875
- **PR:** #10018 (merged)
- **Paths:** src/shared/python/motion_pipeline/api.py, docs/motion_pipeline/formats.md, tests/unit/motion_pipeline/orchestrator/test_api.py, docs/development/DEVELOPMENT_LOG.md, SPEC.md, docs/agent_context/
- **Started:** 2026-09-12
- **Last verified:** 2026-09-12 (`3c17225ef`)
- **Summary:** Reconcile motion pipeline API docstrings and OpenAPI schemas to advertise registered source formats and auto/passthrough instead of rejected formats (mat, fbx, generic json); remove the misleading 'Auto-generated' claim from formats.md; and add unit test coverage asserting format validation and schema accuracy.
- **Evidence:** All CI passed including quality-gate and unit-test-gate; merged to main at 3c17225ef.

### DL-#8695 · DRY Duplication Quarantine Tightening

- **State:** shipped
- **Owner:** claude
- **Issue:** #8695
- **PR:** #10005 (merged)
- **Paths:** scripts/config/dry_duplication_quarantine.json, SPEC.md, docs/development/DEVELOPMENT_LOG.md
- **Started:** 2026-09-11
- **Last verified:** 2026-09-12 (`c578ca942`)
- **Summary:** Pruned 72 dead quarantined fingerprints across supported scanner runtimes (666 -> 594); no entry raised, baseline not regenerated.
- **Evidence:** All CI passed; merged to main at c578ca942.

### DL-#9747 · Signed Release Tag Enforcement and Verification

- **State:** shipped
- **Owner:** claude
- **Issue:** #9747
- **PR:** #10008 (merged)
- **Paths:** .github/workflows/release.yml, docs/operations/release-runbook.md, tests/ci/test_ci_infrastructure.py, SPEC.md, docs/development/DEVELOPMENT_LOG.md
- **Started:** 2026-09-12
- **Last verified:** 2026-09-12 (`c893a73ad`)
- **Summary:** Enforce cryptographic signature verification on production release tags in release.yml and update release runbook.
- **Evidence:** All CI passed including quality-gate and unit-test-gate; merged to main at c893a73ad.

### DL-#9953 · Scalar Parameter Bounds

- **State:** shipped
- **Owner:** codex
- **Issue:** #9953
- **PR:** #9955 (merged)
- **Paths:** src/shared/python/optimization/ocp/parameter_ocp.py; parameter OCP tests; calculation inventory.
- **Started:** 2026-09-10
- **Last verified:** 2026-09-10 (`32410babfd`)
- **Summary:** Explicit constant interpolation gives each shared parameter a single-column bound, preserving limits and locked values.
- **Evidence:** All CI passed, including Linux Bioptim OCP and both manufactured checks. Merged as32410babfd1e4741fa0c53bf05dd8403a51bf233.

### DL-#9952 · Native Camera Setup

- **State:** shipped
- **Owner:** codex
- **Issue:** #9952; parent #9906
- **PR:** #9954 (merged)
- **Paths:** src/tools/capture_rig/camera_setup\*.py; wizard/header; capability registry; tests and camera setup guide.
- **Started:** 2026-09-10
- **Last verified:** 2026-09-10 (`4b305df1d1`)
- **Summary:** Background discovery, stable named bindings, immutable plan revisions and optional wizard entry reuse the rig pipeline.

### DL-#9934 · Cross-Model Biomechanics Analysis

- **State:** shipped
- **Owner:** codex
- **Issue:** #9934
- **PR:** #9941 (merged)
- **Paths:** src/shared/python/biomechanics, src/api/routes/biomechanics.py, src/shared/python/analysis/biomechanics_display.py, src/shared/python/dashboard, ui/src/components/analysis
- **Started:** 2026-09-09
- **Last verified:** 2026-09-10 (`300d96a1b2`)
- **Summary:** Shared calibrated conventions and golf metrics across model inputs with explicit availability and configurable displays.
- **Evidence:** Focused 63-test suite passes; API compute/convert/display and web plot tests pass.

### DL-#9926 · Unified Model and Video Analysis

- **State:** shipped
- **Owner:** codex
- **Issue:** #9926; children #9929, #9930, #9932, #9942
- **PR:** #9933 (merged)
- **Paths:** src/motion_capture/coaching; src/tools/capture_rig; src/tools/pose_studio
- **Started:** 2026-09-09
- **Last verified:** 2026-09-10 (`f04aa1a570`)
- **Summary:** Shared geometry and model-only coaching; see [ledger](unified_analysis_9926.md).
- **Evidence:** Comparison launch/save, PNG parity, snapshot video, cancellation and camera-evidence tests pass; Ruff/format, budgets and LoD pass; Driver comparison drawing UI and PNG inspected.

### DL-#9921 · Native Simscape Tour-Average Matching

- **State:** shipped
- **Owner:** codex
- **Issue:** #9921; implementation #9924, #9925, #9927
- **PR:** #9948 (merged)
- **Paths:** Simscape MATLAB motion_matching/shared, model initialization, shared Python prefix_fit, tests and simscape_tour_matching docs
- **Started:** 2026-09-09
- **Last verified:** 2026-09-11 (`156fcf2fa5`)
- **Summary:** Reproducible forward-dynamics matching with fixed geometry and continuous polynomial torques; native starting pose verified, full swing fit outstanding.

### DL-#9915 · Verified Agent Context

- **State:** shipped
- **Owner:** codex
- **Issue:** #9915
- **PR:** #9920 (merged)
- **Paths:** `docs/agent_context`, `.github/workflows/ci-standard.yml`, `scripts/check_doc_size_budget.py`, `tests/ci`, `tests/scripts`
- **Started:** 2026-09-09
- **Last verified:** 2026-09-10 (final e83bd2e4 pins pass provider/seam, context, atlas and navigation controls; wheel passes launcher and calibration tests).
- **Summary:** Twelve components and five reviewed integrations reuse the atlas and capture goals. Main276998030 is integrated; a required regression rejects divergent pip/source/Rust providers.

### DL-#9914 · C3D Reference Fitting

- **State:** shipped
- **Owner:** codex
- **Issue:** #9914
- **PR:** #9918 (merged)
- **Paths:** src/motion_capture
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`6f2d63325` merge; protected CI, 20 club assets)
- **Summary:** Fits, club/volume/handedness display; see [evidence](reference_fitting_epic.md).

### DL-#9913 · Capture Journey Feedback and Detachable Views

- **State:** shipped
- **Owner:** codex
- **Issue:** #9913; epic #9906
- **PR:** #9917 (merged)
- **Paths:** capture_rig source/tests, guide and parity registry
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`8fce9f238` merge;445 local tests and protected CI pass)
- **Summary:** Identity/history, linked help/provenance and retained detachable Qt views.

### DL-#9912 · Impact Shaft Provider Integration

- **State:** shipped
- **Owner:** codex
- **Issue:** #9912; parent #9703
- **PR:** #9916 (merged)
- **Paths:** vendor/ud-tools, tests/shared_contracts, docs/development/impact-acoustics
- **Started:** 2026-09-09
- **Last verified:** 2026-09-10 (PR #9916 and #9920 merged). Main pin e83bd2e4a7a29a2dcd8145ef2d1efa07123324f0 consistent across pins.
- **Summary:** Qualify Tools shaft/theme provider; see PROVIDER_PIN_RESULTS.json.

### DL-#9911 · Preview Discovery Failure Recovery

- **State:** shipped
- **Owner:** codex
- **Issue:** #9911
- **PR:** #9910 (merged)
- **Paths:** src/tools/capture_rig/preview.py, tests/tools/capture_rig/test_preview.py
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`18c8f922e` merge; recovery tests pass)
- **Summary:** Report discovery imports/timeouts through preview status.

### DL-#9907 · Guided Capture Outcomes

- **State:** shipped
- **Owner:** codex
- **Issue:** #9907; #9908; epic #9906
- **PR:** #9931 (merged)
- **Paths:** `src/tools/capture_rig`; capability graph/generator; matching tests and guide.
- **Started:** 2026-09-09
- **Last verified:** 2026-09-10 (a74d5ebd2; 600 integrated regressions, normal push hooks, scoped mypy and Qt/browser review pass)
- **Summary:** Standard Qt outcome wizard shares typed map metadata and existing editors/readiness; capture-owned resume, optional My Clubs, background status and safe map-plan import.

### DL-#9905 · Player Bag and Capture Equipment

- **State:** shipped
- **Owner:** codex
- **Issue:** #9905; epic #9902
- **PR:** #9923 (merged)
- **Paths:** club_data/player_clubs.py; rig/capture_notes.py and equipment.py; Capture Rig bag/editor/library; model/session.py; matching tests, guide and generated maps.
- **Started:** 2026-09-09
- **Last verified:** 2026-09-10 (`5ada5e6a68`)
- **Summary:** My Clubs supports catalog/custom entries, partial measurements, notes, archive and capture assignment. Captures preserve club snapshots and editable-copy lineage; fit provenance retains evidence.

### DL-#9904 · Offline Club Source Catalog

- **State:** shipped
- **Owner:** codex
- **Issue:** #9904; epic #9902
- **PR:** #9919 (merged)
- **Paths:** club_data/catalog_sources.py, public_clubs.json, scripts/review_club_catalog.py and tests
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`01831aa4c5`)
- **Summary:** Three sourced builds, review diffs and preserved player overrides.

### DL-#9903 · Attributed Club Catalog

- **State:** shipped
- **Owner:** codex
- **Issue:** #9903; epic #9902
- **PR:** #9919 (merged)
- **Paths:** club_data/, test_club_catalog.py and club_catalog.md
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`01831aa4c5`)
- **Summary:** Attributed optional properties, explicit inference gates and lossless exchange.

### DL-#9899 · Calibration Revision Status

- **State:** shipped
- **Owner:** codex
- **Issue:** #9899
- **PR:** #9959 (merged)
- **Paths:** reconstruct; rig command; capture_rig result evidence; tests.
- **Started:** 2026-09-10
- **Last verified:** 2026-09-10 (`fff95d5780`)
- **Summary:** Calibration and reconstruction fingerprints invalidate stale outputs; preserve results.

### DL-#9898 · Common Reference Calibration

- **State:** shipped
- **Owner:** codex
- **Issue:** #9898/#9900/#9909
- **PR:** #9946 (merged)
- **Paths:** reference_calibration, wizard, lens adapter
- **Started:** 2026-09-09
- **Last verified:** 2026-09-10 (732553479;28 pin/context checks pass)
- **Summary:** Reviewed paper/ruler calibration and guided recovery. [Evidence and limits](common_reference_calibration.md).

### DL-#9894 · Scoped Ubuntu CI Dependencies

- **State:** shipped
- **Owner:** codex
- **Issue:** #9894
- **PR:** #9896 (merged)
- **Paths:** .github/workflows/ci-standard.yml, scripts/ci/, tests/scripts/
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`5ef615e6fe22a046aec0bac4f80a1c241d03fef3`)
- **Summary:** Signed per-job APT sources preserve shared-runner configuration; six Bash regressions and standard CI pass.

### DL-#9892 · Fleet Guide Compatibility

- **State:** shipped
- **Owner:** codex
- **Issue:** #9892
- **PR:** #9896 (merged)
- **Paths:** scripts/check_agent_docs_consistency.py, tests/architecture/test_check_agent_docs_consistency.py
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`5ef615e6fe22a046aec0bac4f80a1c241d03fef3`)
- **Summary:** Central managed guidance and legitimate external/optional paths pass without hiding real missing-file failures.

### DL-#9883 · Instructor Reference Alignment Workspace

- **State:** shipped
- **Owner:** codex
- **Issue:** #9883
- **PR:** #9896 (merged)
- **Paths:** src/tools/capture*rig/reference*\*.py, styling.py, tests/tools/capture_rig/
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`5ef615e6fe22a046aec0bac4f80a1c241d03fef3`)
- **Summary:** Responsive placement/timing/notes controls, event alignment, revision checks, preview/export parity and native layout evidence are delivered.
- **Evidence:** Qualified candidate equals merged tree; standard unit gate passed 14,821 tests.

### DL-#9882 · Comparison Rendering and Export Qualification

- **State:** shipped
- **Owner:** codex
- **Issue:** #9882
- **PR:** #9896 (merged)
- **Paths:** src/tools/capture_rig/reference_rendering.py, reference_export.py
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`5ef615e6fe22a046aec0bac4f80a1c241d03fef3`)
- **Summary:** Shared compositor, coverage-aware expert homography and staged exports delivered with qualified product #9896.

### DL-#9881 · Reference Timing and Camera Evidence

- **State:** shipped
- **Owner:** codex
- **Issue:** #9881 (advanced reference epic #9863)
- **PR:** #9885 (merged)
- **Paths:** src/motion_capture/reference, src/motion_capture/reconstruct/overlay3d.py, src/tools/capture_rig/reference_comparison.py, src/tools/capture_rig/reference_export.py, related tests and benchmark
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`8901f5804f`)
- **Summary:** Immutable bounded event anchors, binary-search gap-aware sampling, actual camera/clock snapshots and stale-registration checks replace unsupported calibration assumptions.
- **Evidence:** 12 adverse regressions failed before repair; 46 combined tests and 14-module mypy pass. Sampling medians: 0.157/0.093/0.304 ms for120/1200/12000 frames. CI typing/budget corrections pass16 tests.

### DL-#9879 · Comparison State and Export Lifetime

- **State:** shipped
- **Owner:** codex
- **Issue:** #9879 (advanced reference epic #9863)
- **PR:** #9884 (merged)
- **Paths:** src/motion_capture/reference/comparison.py, src/tools/capture_rig/reference_comparison.py, src/tools/capture_rig/swing_export_actions.py, tests/tools/capture_rig/test_reference_comparison_state.py
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`2c99bc83ea`)
- **Summary:** Preserve exact unrelated layer/registration fields, reject bad saved records, and reuse the existing export controller for safe thread ownership and deferred close.
- **Evidence:** Nine adverse regressions preceded repair;31 comparison/cancellation/swing/coaching tests and three-module mypy pass.

### DL-#9865 · Reference Scene Registration & Synchronization

- **State:** shipped
- **Owner:** codex
- **Issue:** #9865 (advanced reference epic #9863)
- **PR:** #9871 (merged)
- **Paths:** src/motion_capture/reference/registration.py, src/motion_capture/reconstruct/overlay3d.py, related tests/docs
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`10caddd219`)
- **Summary:** Calibrated scene registration, event-anchor and offset time synchronization, missing-joint gap mask preservation across bounded interpolation, distortion-aware camera projection, and 2D expert video homography without 3D claims.
- **Evidence:** 7 focused registration tests pass in tests/motion_capture/test_reference_registration.py. Strict round-trip serialization/deserialization validated. Projection tested with both pinhole and Brown-Conrady distortion. Ruff checks pass cleanly.

### DL-#9864 · Expert Reference Asset Imports

- **State:** shipped
- **Owner:** codex
- **Issue:** #9864 (advanced reference epic #9863)
- **PR:** #9870 (merged)
- **Paths:** src/motion_capture/reference, src/tools/capture_rig/reference_import.py, src/tools/capture_rig/reference_library_dialog.py, src/tools/capture_rig/library_dialog.py, src/shared/python/motion_pipeline/sources/c3d_adapter.py and related tests/docs/maps
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`90b147e10`; local qualification complete)
- **Summary:** Versioned portable reference assets retain explicit mapping, source hashes, timestamps and missing points; native library adds imports, notes/archive and background I/O. Expert videos remain linked 2D assets.
- **Evidence:** 30 integration tests pass, including real C3D and fresh-process isolation. Four native UI tests, eight-module mypy and architecture checks pass after layout/helper corrections.

### DL-#9862 · Saved Coaching References

- **State:** shipped
- **Owner:** codex
- **Issue:** #9862 (product #9849)
- **PR:** #9869 (merged)
- **Paths:** src/motion_capture/coaching, src/tools/capture_rig/coaching_canvas.py, coaching_dialog.py, coaching_export.py and related integration/tests/docs
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`77e6bca88`; protected PR merged)
- **Summary:** Saved source-coordinate shapes; draw/edit/style/frame visibility, undo/redo, library/editor access and cancellable PNG/video export.
- **Evidence:** 49 registry/atlas and19 drawing/export tests;12-module mypy;3012-file LoD clean. Visual minimums:496px references,465px editor.

### DL-#9860 · Capture Editing and Library

- **State:** shipped
- **Owner:** codex
- **Issue:** #9860, #9861 (product #9849)
- **PR:** #9868 (merged)
- **Paths:** src/motion_capture/rig/edits.py, ingest.py, src/tools/capture_rig/swing_editor.py, related tests and docs/development/capture_editing_integration.md
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`4c89892d7`; protected PR merged)
- **Summary:** Source-preserving trim/crop, capture notes/library, archive/storage/rename rollback, editable copies, cancellable export and timeline guards.
- **Evidence:** 300 integrated, 12 library/UI and 5 editor tests; eight-module mypy. Visual QA: 850x650, minimum492px.

### DL-#9851 · Capture Responsiveness and Recovery

- **State:** shipped
- **Owner:** codex
- **Issue:** #9851, #9857 (epic #9849)
- **PR:** #9859 (merged)
- **Paths:** src/tools/capture_rig/player.py, process_runner.py, benchmark_capture_responsiveness.py and cache/process tests; docs/development/capture_product_review.md
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`eaf8503ce`)
- **Summary:** Capture Responsiveness and Recovery
- **Acceptance:** Repeated frames decode once with isolated pixels; failed starts restore lifecycle/retry; benchmark limits documented.
- **Evidence:** 241 camera tests after cache; six focused tests after recovery; duplicate median 196.788 to 12.613 ms.

### DL-#9850 · Generated Capability Atlas

- **State:** shipped
- **Owner:** codex
- **Issue:** #9850 (children #9852, #9853; product #9849)
- **PR:** #9856 (merged)
- **Paths:** `scripts/capability_atlas/`, `scripts/generate_capability_atlas.py`,
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`9d6e6a872`)
- **Summary:** Generated C4-style context, workflow/artifact maps, searchable capabilities and Mermaid from existing registries.

### DL-#9830 · Independent Shooting Accuracy

- **State:** shipped
- **Owner:** codex
- **Issue:** #9830
- **PR:** #9841 (merged)
- **Paths:** src/shared/python/optimization; docs/development/shooting_convergence_9830_turnover.md
- **Started:** 2026-09-08
- **Last verified:** 2026-09-09 (merged 28d9bf79e)
- **Summary:** Adaptive reference defects; 21 native Bioptim/Casadi 3.6.7 passes. Casadi 3.8 failure and physical limits remain in the linked turnover.

### DL-#9825 · Preserve Reviewed Manufactured Claims in Actual Registration

- **State:** shipped
- **Owner:** codex
- **Issue:** #9825
- **PR:** #9826 (merged)
- **Paths:** docs/development/claim_preservation_9825_turnover.md; manufactured registration and evidence
- **Started:** 2026-09-08
- **Last verified:** 2026-09-09 (merged a410ae705)
- **Summary:** Preserves 328 reviewed outcomes; 128 strict contracts and 11 publication controls pass. Linked turnover retains full provenance and physical limits.

### DL-#9787 · Manufactured Authority Runtime and Provenance

- **State:** shipped
- **Owner:** codex
- **Issue:** #9787
- **PR:** #9804 (merged)
- **Paths:** authority runtime pins, native provenance/CI contracts and manufactured_authority_9787_turnover.md.
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (PR merged as 736ec2189 from b8da0c024)
- **Summary:** Compatible native pins and runtime support merged. The merged revision differs from locally validated 6235789dc; its actual registration bypass and stale evidence require follow-up #9825. Historical test results do not certify differing merged bytes.

### DL-#9783 · Reviewed Renderer Provider Compatibility

- **State:** shipped
- **Owner:** codex
- **Issue:** #9783
- **PR:** #9784 (merged)
- **Paths:** `tests/shared_contracts/test_tools_provider_contracts.py`, `docs/development/renderer_reference_9783_turnover.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`9aa26e4f83`)
- **Summary:** Reproduced the candidate's exact old-hash failure before accepting the two reviewed source/hash pairs. Tolerances, immutable provider origin and the current vendor pin remain strict.

### DL-#9762 · `bioptim` Optimal-Control Backend and the Swing-Dynamics Fixes

- **State:** shipped
- **Owner:** claude
- **Issue:** #9762 (epic); prerequisites #9755, #9756, #9757, #9758, #9759, #9760, #9761
- **PR:** #9768 (merged)
- **Paths:** `src/shared/python/optimization/ocp/`,
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`af6923bc70`)
- **Summary:** Adopts `pyomeca/bioptim` as an opt-in optimal-control layer

### DL-#9733 · Fail Fast on the Uninitialized Vendored Tools Fallback

- **State:** shipped
- **Owner:** claude
- **Issue:** #9733
- **PR:** #9743 (merged)
- **Paths:** `src/__init__.py`, `tests/unit/repo_hygiene/test_src_fallback_fail_fast_9733.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`930543b24d`)
- **Summary:** Uninitialized `vendor/ud-tools` raises actionable ImportError naming remediation command before finder installation.

### DL-#9648 · RTMPose ONNX Pose Estimator Behind the Registry

- **State:** shipped
- **Owner:** claude
- **Issue:** #9648
- **PR:** #9739 (merged)
- **Paths:** `src/shared/python/pose_estimation/rtmpose_onnx_estimator.py`, `src/shared/python/pose_estimation/rtmpose_models.py`, `src/shared/python/pose_estimation/registry.py`, `src/motion_capture/rig/ingest.py`, `src/motion_capture/reconstruct/layouts.py`, `pyproject.toml`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`197e0a942`)
- **Summary:** Registers `rtmpose_onnx` (SimCC decode via onnxruntime, COCO-17/Halpe-26, whole-frame letterbox) with `capture_source=False`; adds the optional `pose-onnx` extra; pins the official OpenMMLab ONNX model URLs/sizes with digests PENDING OWNER APPROVAL; teaches `RegisteredFrameEstimator` to honour instance-level `LANDMARK_MAP`/`LAYOUT_NAME`; extends `layouts.py` with the Halpe-26 `hip`→`mid_hip` alias.

### DL-#9631 · Vendor Pin Carries the Tools#5048 Alias-Predicate Fix

- **State:** shipped
- **Owner:** claude
- **Issue:** #9631
- **PR:** #9722 (merged)
- **Paths:** `vendor/ud-tools`, `tests/unit/repo_hygiene/test_pinned_import_alias_contract.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`648d4c9213`)
- **Summary:** Pinned `vendor/ud-tools` carries Tools#5049 flattened-install fix; added TDD contract test asserting pinned predicate in both layouts.

### DL-#9612 · Video Upload Suffix Derived From Filename Allow-List

- **State:** shipped
- **Owner:** claude
- **Issue:** `#9612`
- **PR:** #9720 (merged)
- **Paths:** `src/api/routes/video.py`, `tests/unit/api/test_routes_video.py`
- **Started:** 2026-09-07
- **Last verified:** 2026-09-08 (`4f705f508c`)
- **Summary:** Video analysis uploads no longer default temp files to `.mp4`;

### DL-#9607 · Authority Runtime Native Library Bootstrap for Cmeel Pinocchio Wheels

- **State:** shipped
- **Owner:** claude
- **Issue:** #9607
- **PR:** #9726 (merged)
- **Paths:** `scripts/research/proximal_distal_energy/articulated_native_runtime.py`, `scripts/research/proximal_distal_energy/run_articulated_manufactured_solution.py`, `tests/research/test_articulated_native_runtime.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`474ea6bf56`)
- **Summary:** Resolves the authority lane's `liburdfdom_sensor.so.4.0` import failure by resolving `cmeel.prefix/lib` from the live venv, verifying the locked sonames with an explicit DbC diagnostic, and re-execing the authority profile with `LD_LIBRARY_PATH` prepended before `import pinocchio`.

### DL-#9542 · Bunker Exit State Consistency, Provenance, and Result Envelope

- **State:** shipped
- **Owner:** claude
- **Issue:** #9542
- **PR:** #9728 (merged)
- **Paths:** `src/bunkershot3d/ball/**`
- **Started:** 2026-09-07
- **Last verified:** 2026-09-08 (`ba6e6e2a30`)
- **Summary:** `SandDelivery` now refuses contradictory exit speed/vector pairs and owns copies of list-supplied exit vectors so post-construction mutation cannot invalidate the frozen record; the `to_post_impact_state` boundary carries explicit `ExitVectorProvenance` labels, and `PostImpactEnvelope` wraps the flight handoff with the validity verdict, F0 tier, per-group frames, the proper `HEAD_FRAME_TO_FLIGHT_TRANSFORM`, a schema version, and a SHA-256 source digest with JSON round trip. Reflection rejection itself was already delivered by PR #9574 and is not redone.

### DL-#9533 · Test-Only Extras Reachable From the Dev Lock

- **State:** shipped
- **Owner:** claude
- **Issue:** `#9533`
- **PR:** #9716 (merged)
- **Paths:** `pyproject.toml`, `.github/workflows/lock-refresh.yml`, `requirements*.lock`, `environment.yml`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`bae85ae4c4`)
- **Summary:** `openpyxl` and `imageio` resolve through `dev` extra; lock regeneration delegated to `lock-refresh.yml`.

### DL-#9499 · Spec Check Reminder Fail-Safe Extraction

- **State:** shipped
- **Owner:** claude
- **Issue:** `#9499`
- **PR:** #9719 (merged)
- **Paths:** `.github/workflows/spec-check.yml`, `scripts/post_spec_reminder.py`, `tests/ci/test_spec_check_workflow.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`e214fae7bd`)
- **Summary:** The `Verify SPEC.md freshness` job posts its SPEC reminder

### DL-#9494 · Resolve the CLAUDE.md `--no-verify` Contradiction by Fixing the Windows Hook Environment

- **State:** shipped
- **Owner:** claude
- **Issue:** #9494
- **PR:** #9744 (merged)
- **Paths:** `CLAUDE.md`, `AGENT_HANDOFF.md`, `docs/development/DEVELOPMENT_LOG.md`, `SPEC.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`dbc6727aa`)
- **Summary:** CLAUDE.md forbade `git commit --no-verify` while agents on

### DL-#9484 · Impact Explorer Web Route Has a CI Bundle Producer

- **State:** shipped
- **Owner:** W4_9484 (agent claude)
- **Issue:** #9484
- **PR:** #9724 (merged)
- **Paths:** `.github/workflows/ci-standard.yml`, `scripts/check_declared_route_producers.py`, `tests/scripts/test_declared_route_producers.py`, `docs/workflows/WORKFLOW_TRACKING.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`ba841ae524`)
- **Summary:** The `rate_of_closure` tile declared `web.mode: route` for `/tools/impact-explorer` but no pipeline built `vendor/ud-tools/src/rate_of_closure/web/dist`, so a clean checkout served the honest fallback. CI Standard now builds the bundle from the pinned Tools tree with `npm run build -- --base=/impact-explorer-app/`, and `scripts/check_declared_route_producers.py` fails any declared route that no pipeline produces. Shipping the bundle inside the wheel/image remains an open maintainer decision (#9417).

### DL-#9482 · Launcher Tile Logo Families and Registry Gate

- **State:** shipped
- **Owner:** claude
- **Issue:** #9482
- **PR:** #9725 (merged)
- **Paths:** `src/config/launcher_manifest.json`, `assets/logos/**`,
- **Started:** 2026-09-07
- **Last verified:** 2026-09-07 (`191351bdf`)
- **Summary:** Broke the launcher grid's worst logo reuse (data_explorer x9,

### DL-#9478 · Launcher Registry Truth: `tools://` Provenance Scheme and Ready/Beta Maturity Gate

- **State:** shipped
- **Owner:** claude
- **Issue:** #9478
- **PR:** #9729 (merged)
- **Paths:** `src/config/models.yaml`, `src/config/launcher_manifest.json`,
- **Started:** 2026-09-07
- **Last verified:** 2026-09-08 (`8c2c6fa4a1`)
- **Summary:** `provider: tools` entries in `src/config/models.yaml` and

### DL-#9476 · Re-Vendor the Corrected Spec Merge Driver and Pin Drift

- **State:** shipped
- **Owner:** claude
- **Issue:** `#9476`
- **PR:** #9736 (merged)
- **Paths:** `scripts/install_spec_merge_driver.py`,
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`875cd501e`)
- **Summary:** The vendored installer still stamped the withdrawn

### DL-#9470 · Launch-Monitor Analysis Handlers Onto the Async_Action Worker

- **State:** shipped
- **Owner:** claude
- **Issue:** #9470
- **PR:** #9742 (merged)
- **Paths:** `src/tools/launch_monitor_analytics/gui.py`, `src/tools/launch_monitor_analytics/_embed_adapter.py`, `tests/ui/tools/launch_monitor/test_async_actions.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`af0aed1d57`)
- **Summary:** All seven analysis handlers (`treatment`, `relationship`, `multivariate`, `model`, `comparison`, `dispersion`, `trend`) now run their compute on the #8880 `async_action` worker via one shared `AsyncActionBar`; synchronous `present(compute())` paths kept; embed adapter `cleanup()` cancels and joins the worker. First slice of the #9470 tool checklist; the remaining tools are follow-ups.

### DL-#9409 · Always-On Quality Gate Lane and Conftest Src-Pivot Guard

- **State:** shipped
- **Owner:** `claude`
- **Issue:** [#9409](https://github.com/D-sorganization/UpstreamDrift/issues/9409)
- **PR:** #9723 (merged)
- **Paths:** `.github/workflows/ci-standard.yml`, `tests/unit/repo_hygiene/test_no_conftest_src_module_pivot.py`, `docs/workflows/WORKFLOW_TRACKING.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`6428062284`)
- **Summary:** CI Standard gains an always-on, ≤10-minute `always-on-unit-lane` (verify_installation import smoke over the shared Tools alias roots, top-level smoke tests, contract tests) that `quality-gate` requires `success` on every PR including docs-only ones; a repo-hygiene guard forbids any conftest from pivoting `sys.modules["src"]` directly (must use `EngineSrcPivot`). Deferred on #9409: main-branch cancel exemption (RM campaign) and nightly cross-engine dedupe (#8725/#9002).

### DL-#9387 · Unit-Gate Worker Corruption: `src`-Identity Sentinel and Leak Fixes

- **State:** shipped
- **Owner:** claude
- **Issue:** #9387
- **PR:** #9741 (merged)
- **Paths:** `tests/unit/repo_hygiene/test_src_identity_sentinel.py`,
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`c70d5ddae`)
- **Summary:** Static audit of `tests/` (conftests excluded) found 83

### DL-#9249 · UI: Pin @vitejs/Plugin-React to ^5 Until Vite 8

- **State:** shipped
- **Owner:** claude
- **Issue:** #9249
- **PR:** #9718 (merged)
- **Paths:** `.github/dependabot.yml`, `ui/README.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`7cdbb0a3d`)
- **Summary:** Dependabot ignores `@vitejs/plugin-react` major updates

### DL-#9091 · Phantom-Guard Rule-3 False Positive on Shallow Base Fetch

- **State:** shipped
- **Owner:** claude
- **Issue:** #9091
- **PR:** #9717 (merged)
- **Paths:** `.github/workflows/anti-phantom-merge.yml`, `scripts/ci/check_phantom_guard_paths.py`, `tests/scripts/test_check_phantom_guard_paths.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`e55c4f3e19`)
- **Summary:** The phantom-guard rule-3 closes-issue path-membership check

### DL-#8943 · Cache API CPU Work Off the Event Loop

- **State:** shipped
- **Owner:** claude
- **Issue:** #8943
- **PR:** #9727 (merged)
- **Paths:** `src/api/routes/analysis_plots.py`, `src/api/routes/model_explorer.py`, `src/api/routes/models.py`, `src/api/routes/launch_monitor_analytics.py`, `src/api/routes/_route_utils.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`17beca5537`)
- **Summary:** `GET /analysis/plot-data/{plot_type}` builds orchestrator once per recorder identity and serves results from an LRU off the event loop. Model explorer and models URDF handlers use LRU cache in worker threads.

### DL-#8901 · Accessible Model Card Actions and Grid Navigation

- **State:** shipped
- **Owner:** claude
- **Issue:** #8901
- **PR:** #10006 (merged)
- **Paths:** src/launchers/model_card.py, tests/launchers/test_model_card_accessibility.py
- **Started:** 2026-09-12
- **Last verified:** 2026-09-12 (`eb4a042cbb`)
- **Summary:** Harden model card touch and keyboard accessibility, ensure WCAG target sizes, and add arrow-key grid navigation in launcher.

### DL-#8360 · Bounded Launcher Splash and Optional-Provider Degradation

- **State:** shipped
- **Owner:** claude
- **Issue:** #8360 (related #8339, #8358, #8359)
- **PR:** #9951 (merged)
- **Paths:** src/launchers/startup.py, src/launchers/startup_phases.py, src/launchers/startup_session.py, src/launchers/startup_failure_dialog.py, src/launchers/upstream_drift_launcher_main.py, src/launchers/launcher_orchestrator.py
- **Started:** 2026-09-10
- **Last verified:** 2026-09-11 (`35c69c2894`)
- **Summary:** Bounded, timestamped startup phases with a StartupSession watchdog; optional Tools/Rate provider degrades the shell instead of stalling the splash; Retry / Continue / Copy diagnostics / Close dialog replaces quit-on-error.
- **Evidence:** Deterministic tests inject successful, missing, exception-raising and never-completing providers and prove bounded splash lifetime, degraded shell startup, stale-generation isolation and deleted-widget guards.

### DL-#1616 · Mermaid C4 Architecture Maps

- **State:** shipped
- **Owner:** local
- **Issue:** #1616
- **PR:** #9963 (merged)
- **Paths:** docs/architecture/C4.md, scripts/architecture_map_contract.py
- **Started:** 2026-09-10
- **Last verified:** 2026-09-10 (`09f6d22da3`)
- **Summary:** Baseline adoption of Mermaid C4 architecture maps in UpstreamDrift.

## Archive

Older entries live in `DEVELOPMENT_LOG_ARCHIVE_<year>.md`.

## Field Reference

| Field           | Required                   | Notes                                                          |
| --------------- | -------------------------- | -------------------------------------------------------------- |
| `State`         | Always                     | One of the six states above                                    |
| `Owner`         | Always                     | Agent id from the fleet roster, or `unassigned`                |
| `Issue`         | While live                 | Governing GitHub issue; enforces the entry/issue join          |
| `Branch`        | `in_progress`, `in_review` | Enforces the entry/branch join                                 |
| `PR`            | Always                     | Number and state, or `not created`                             |
| `Paths`         | Always                     | Globs; drives silent-entry detection                           |
| `Started`       | Always                     | Drives cycle time                                              |
| `Last verified` | Always                     | Date plus SHA — the liveness signal                            |
| `Summary`       | Always                     | One or two sentences                                           |
| `Next step`     | While live                 | Exactly one action; if it needs two sentences, split the entry |
| `Parked`        | When `parked`              | Date plus reason                                               |

Never place credentials, tokens, or customer data in a development log.
