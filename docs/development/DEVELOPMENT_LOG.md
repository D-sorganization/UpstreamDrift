# Development Log — UpstreamDrift

State table for every feature in flight in this repository. Update entries
**in place**; never append dated sections. One entry per feature, from proposal
to ship. See the `development-logs` section of `AGENTS.md` for the binding rules
and `shared_scripts/development_log.py` for the validator.

- **Portfolio:** golf
- **WIP limit:** 8
- **Last audited:** 2026-09-12 by claude
- **Current metadata review:** Last verified commit identifiers below bind path inventory only to e68baeabde8ff639f846a167ca7703ace43bd0a9; no tests or native runs were performed. Full original row text before narrative condensation is retained in [the pre-compression snapshot](evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_PRE_COMPRESSION_20261011.md.snapshot). Prior verification wording and per-path Git blob identities are preserved in [the source metadata audit](evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json).

## States

`proposed` → `in_progress` → `in_review` → `shipped`, with `parked` reachable
from any live state and `abandoned` from `parked`. `shipped` never returns to
`in_progress`; open a new entry instead.

## Active

### DL-#12784 · Nightly motion-matching-smoke.yml Runs Tests/Integration/Motion_Matching (Slow, --Timeout=0) via Scripts/Ci/Run_Motion_Matching_Smoke.Sh on the Fleet; a Failure Opens or Updates an Issue and the Red Scheduled Run Reaches Fleet-Main-Health; Not Coupled to Heavy-Tests-Opt-In Parity

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12784
- **Branch:** ci/12784-motion-matching-smoke-lane
- **PR:** #12786
- **Paths:** see #12786
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`c4b15957`; collated from changes/12784-motion-matching-smoke-lane.md)
- **Summary:** Nightly motion-matching-smoke.yml runs tests/integration/motion_matching (slow, --timeout=0) via scripts/ci/run_motion_matching_smoke.sh on the fleet; a failure opens or updates an issue and the red scheduled run reaches fleet-main-health; not coupled to heavy-tests-opt-in parity
- **Next step:** Merge the PR; the first scheduled run needs #12776 and #12783 on main to be green.

### DL-#12702 · Fix the 'Spec Has No Club Body' Crash on the MuJoCo-Native Driver Run: Assembly_From_Spec Now Recognises the Native Simscape Club (Segmented 2.5in Butt Grip, No Grip Solid) so the Default V2 Spec Carries a Club Body; Lane.Set_Club_Targets Fails Fast via Club_Face_Target.Require_Club_Assembly Naming the Spec Source. the Smoke Test Now Gets Past Set_Club_Targets but Still Fails on Dynamics Marker RMS (0.51 m vs 0.15 m Bound), a Separate Regression; the Smoke Is Marked Slow and Runs in No CI Lane

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12702
- **Branch:** fix/12702-mujoco-smoke-club-body
- **PR:** #12776, #12783
- **Paths:** see #12776
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`789e583f`; collated from changes/12702-dynamics-rms-smoke-spec.md)
- **Summary:** Fix the 'spec has no club body' crash on the MuJoCo-native driver run: assembly_from_spec now recognises the native Simscape club (segmented 2.5in Butt grip, no Grip solid) so the default v2 spec carries a club body; Lane.set_club_targets fails fast via club_face_target.require_club_assembly naming the spec source. The smoke test now gets past set_club_targets but still fails on dynamics marker RMS (0.51 m vs 0.15 m bound), a separate regression; the smoke is marked slow and runs in no CI lane
- **Next step:** Merge the PR; put the slow smoke in a nightly lane (it runs in no CI lane today).

### DL-#12683 · R3: Trail-Foot Toe-Out Owner Style Target (TRAIL_TOE_OUT_STYLE_DEG = 20 Deg, Single-Sourced From DEFAULT_TOE_OUT_DEG) via a New --Foot-Progression Style Mode, Now the Canonical Default; Capture Mode Still Lets the Measurement Win; Per-Foot Target_Source in Receipts; Driver Trail -2.9 -> 19.8 Deg, 7-Iron -9.8 -> 13.5 Deg (Hip_Rotation_R at Its -40 Deg Limit)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12683
- **Branch:** fix/12683-trail-toe-out
- **PR:** #12731
- **Paths:** see #12731
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`f6e69608`; collated from changes/12683-r3-trail-foot-toe-out-owner-style-target.md)
- **Summary:** R3: trail-foot toe-out owner style target (TRAIL_TOE_OUT_STYLE_DEG = 20 deg, single-sourced from DEFAULT_TOE_OUT_DEG) via a new --foot-progression style mode, now the canonical default; capture mode still lets the measurement win; per-foot target_source in receipts; driver trail -2.9 -> 19.8 deg, 7-iron -9.8 -> 13.5 deg (hip_rotation_r at its -40 deg limit)
- **Next step:** Owner decision on the 7-iron shortfall (hip external-rotation range vs accept ~13.5 deg), then regenerate anthro_driver_seeds/anthro_iron_seeds_zmp receipts before R4 (#12684).

### DL-#12682 · R5: One Clubhead Measurement Point (Face Centre via Club_Face_Target.Face_Centre_Path) for the Fd_Phase Impact Split, the Exporter Impact Detector and a New Dynamics.Fd_Phase.Clubhead_Speed Receipt Block (Simulated/Reference/Capture Speed, % Error, Peak-Timing Offset); Scripts/diagnose_clubhead_speed.py Decomposition

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12682
- **Branch:** fix/12682-clubhead-speed-parity
- **PR:** #12695
- **Paths:** see #12695
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`0081a09b`; collated from changes/12682-r5-one-clubhead-measurement-point-face-c.md)
- **Summary:** R5: one clubhead measurement point (face centre via club_face_target.face_centre_path) for the fd_phase impact split, the exporter impact detector and a new dynamics.fd_phase.clubhead_speed receipt block (simulated/reference/capture speed, % error, peak-timing offset); scripts/diagnose_clubhead_speed.py decomposition
- **Next step:** Merge the PR; R10 regenerates canonical receipts with the new block; R6 sizes the shaft share (about 2 % on the 7-iron).

### DL-#12413 · \ Restore DBC Level in Test_Ud_Contract_Level_Alias_Sync to Prevent Xdist Pollution\

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12413
- **Branch:** fix/12413-flaky-test-missing-columns-contract-level
- **PR:** #12414
- **Paths:** see #12414
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`2202a2b3`; collated from changes/12413-restore-dbc-level-in-test-ud-contract-le.md)
- **Summary:** \ Restore DBC level in test_ud_contract_level_alias_sync to prevent xdist pollution\
- **Next step:** \Merge the PR.\

### DL-#12405 · Reconcile Package Collision Gate With Initialized Tools Provider Layout, Distinguishing Verified Canonical Aliases and Seam Counterparts While Eliminating Redundant Nested Tools Roots From Pythonpath.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12405
- **Branch:** fix/12405-reconcile-package-collision-gate
- **PR:** #12419
- **Paths:** see #12419
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`7b74513b`; collated from changes/12405-reconcile-package-collision-gate-with-in.md)
- **Summary:** Reconcile package collision gate with initialized Tools provider layout, distinguishing verified canonical aliases and seam counterparts while eliminating redundant nested Tools roots from pythonpath.
- **Next step:** Merge the PR.

### DL-#12404 · Reconcile Optimized Contract Defaults Across Provider Boundaries in Test Probes Without Creating an Unapproved Public Contracts Fork.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12404
- **Branch:** fix/12404-reconcile-optimized-contract-defaults
- **PR:** #12426
- **Paths:** see #12426
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`d56b30b8`; collated from changes/12404-reconcile-optimized-contract-defaults-ac.md)
- **Summary:** Reconcile optimized contract defaults across provider boundaries in test probes without creating an unapproved public contracts fork.
- **Next step:** Merge the PR.

### DL-#12353 · Allow Native Moco Preparation to Skip Only the Short Capability Probe While Recording Runtime as Not Attempted and Keeping Solve Readiness False

- **State:** in_progress
- **Owner:** unassigned
- **Issue:** #12353
- **Branch:** feat/native-no-probe-12353-20261012
- **PR:** #12367
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/native_moco_preparation.py,src/engines/physics_engines/opensim/python/tour_matching/native_body_inertia.py,src/engines/physics_engines/opensim/python/tour_matching/native_passive_receipt.py,src/engines/physics_engines/opensim/python/tour_matching/cli.py,tests/opensim/test_native_passive_receipt.py,tests/opensim/test_native_moco_inertia_preflight.py,tests/unit/engines/opensim/test_native_moco_probe_mode.py,tests/unit/engines/opensim/test_native_path_optimizer_domains.py,manuals/upstreamdrift/chapters/56-native-diagnostic-execution-contract.qmd,manuals/upstreamdrift/calculation-registry.json,SPEC.md,AGENT_HANDOFF.md,docs/development/feedback_controls/F07_NATIVE_NO_PROBE_PREPARATION_TURNOVER.md,docs/agent_context/README.md,docs/agent_context/index.html
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`56c9b40e`; collated from changes/12353-native-moco-skip-capability-probe.md)
- **Summary:** Allow native Moco preparation to skip only the short capability probe while recording runtime as not attempted and keeping solve readiness false
- **Next step:** Root review of implementation, focused tests, and manual-governance updates.

### DL-#12319 · Putt Scene Export for Public Explorer

- **State:** in_review
- **Owner:** claude
- **Issue:** #12319
- **Branch:** feat/12319-putt-export
- **PR:** #12335
- **Paths:** `src/tools/putting_green_gui/_scene_export.py`
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`178b32b6`; collated from changes/12319-export-the-3-d-putt-simulation-as-the-ud.md)
- **Summary:** Export the 3-D putt simulation as the ud.putt_scene/1 mirror artifact for the public explorer Putting tab
- **Next step:** Merge the PR; sync the artifact into the public explorer mirror.

### DL-#12317 · Force Redistribution Reports Optimizer Convergence Only When the Optimizer Reports Success, Independently of the Candidate Feasibility Audit.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12317
- **Branch:** fix/force-allocation-convergence-12317
- **PR:** #12322
- **Paths:** src/shared/python/motion_matching/force_nullspace.py,tests/unit/motion_matching/test_force_nullspace.py,docs/design/force_redistribution_convergence.md,docs/motion_matching/README.md,docs/development/force_allocation_convergence_12317_turnover.md
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ed672ef3`; collated from changes/12317-force-allocation-convergence.md)
- **Summary:** Force redistribution reports optimizer convergence only when the optimizer reports success, independently of the candidate feasibility audit.
- **Next step:** Root review of the committed implementation and validation.

### DL-#12315 · Widen the Golf Models' Forearm Pronation Range to the AAOS Normative 80 Deg From Neutral, From One Shared Source

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12315
- **Branch:** fix/12315-forearm-pronation-range
- **PR:** #12484
- **Paths:** see #12484
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`6d665f59`; collated from changes/12315-widen-the-golf-models-forearm-pronation.md)
- **Summary:** Widen the golf models' forearm pronation range to the AAOS normative 80 deg from neutral, from one shared source
- **Next step:** Regenerate the OSV-9 calibration and re-render after merge.

### DL-#12309 · Migrate Final Figure Builder Calls to FigureOptions and Verify Real Writer Integration and Intended-Output Parity

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12309
- **Branch:** codex/shot-pattern-safe-followup
- **PR:** #12418, #12445
- **Paths:** scripts/research/shot_pattern_analysis/build_final_figures.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`9f60d7c4`; collated from changes/12309-single-figureoptions-mapping-helper-for.md)
- **Summary:** Migrate final figure builder calls to FigureOptions and verify real writer integration and intended-output parity
- **Next step:** Include validated fix in the shot-pattern integration PR; merge remains pending

### DL-#12302 · Refactor Shot-Pattern Duplication and Architecture Debt While Preserving Frozen Research Evidence

- **State:** in_progress
- **Owner:** unassigned
- **Issue:** #12302
- **Branch:** codex/shot-pattern-refinement-integration
- **PR:** #12418
- **Paths:** see #12418
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/12302-refactor-shot-pattern-duplication-and-ar.md)
- **Summary:** Refactor shot-pattern duplication and architecture debt while preserving frozen research evidence
- **Next step:** Complete remaining decomposition and unchanged architecture/DRY gates before PR integration

### DL-#12299 · Reconcile Shot-Pattern Integration and Repair Inherited Validation Blockers While Preserving Frozen Science and the Approved DRY Baseline

- **State:** in_progress
- **Owner:** unassigned
- **Issue:** #12299
- **Branch:** codex/shot-pattern-safe-followup
- **PR:** #12418
- **Paths:** see #12418
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/12299-restore-multi-runtime-dry-baseline.md)
- **Summary:** Reconcile shot-pattern integration and repair inherited validation blockers while preserving frozen science and the approved DRY baseline
- **Next step:** Finish provider repairs and merged Tools dependency, then pass complete canonical validation and protected integration

### DL-#12280 · Admit Exact Zero-Coordinate Structural WeldJoint in Constrained Muscle Replay

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12280
- **Branch:** feat/opensim-weldjoint-12280
- **PR:** #12282
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/native_constrained_muscle.py;tests/opensim/test_native_constrained_weld_policy.py;tests/opensim/test_native_constrained_muscle.py;manuals/upstreamdrift/chapters/59-native-structural-weld-admission.qmd;manuals/upstreamdrift/index.qmd;docs/development/feedback_controls/DESIGN.md;docs/development/feedback_controls/F07_WELDJOINT_ADMISSION_TURNOVER.md;docs/development/feedback_controls/evidence/native_weldjoint_12280/child-tests.json;docs/development/feedback_controls/evidence/native_weldjoint_12280/junit.xml;docs/development/feedback_controls/evidence/native_weldjoint_12280/manifest.json
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`55f93c42`; collated from changes/12280-native-weldjoint-admission.md)
- **Summary:** Admit exact zero-coordinate structural WeldJoint in constrained muscle replay
- **Next step:** Complete aggregate pre-PR checks, then validate the source-specific replay path separately from this synthetic fixture.

### DL-#12275 · Retain Hash-Bound Manufactured Native Diagnostics

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12275
- **Branch:** docs/manufactured-feedback-evidence-12275
- **PR:** #12277
- **Paths:** see #12277
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`7016d485`; collated from changes/12275-retain-manufactured-native-diagnostic-evidence.md)
- **Summary:** Retain hash-bound manufactured native diagnostics
- **Next step:** Review byte-preserved evidence and documentation; keep all production qualification gates open.

### DL-#12271 · Add Hash-Bound One-Pass Nonuniform Moco Mesh Refinement Planning

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12271
- **Branch:** feat/f07-native-component-audit-budget-12244
- **PR:** #12273
- **Paths:** see #12273
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`60598915`; collated from changes/12271-native-moco-history-appendices.md)
- **Summary:** Add hash-bound one-pass nonuniform Moco mesh refinement planning
- **Next step:** Preserve source-bound failures and strict replay limits; do not infer production or physical qualification from synthetic solver results.

### DL-#12249 · MyoSuite Native Muscle Actuator Classification

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12249
- **Branch:** fix/12249-native-muscle-actuator-enum
- **PR:** #12252
- **Paths:** src/engines/physics_engines/myosuite/python/muscle_analysis.py;tests/unit/engines/myosuite/test_muscle_analysis.py;manuals/upstreamdrift/chapters/31-myosuite-native-excitation-replay.qmd;docs/development/feedback_controls/F09Q-MYOSUITE-ACTUATOR-CLASSIFICATION-TURNOVER.md
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0870e093`; collated from changes/12249-native-muscle-actuator-classification.md)
- **Summary:** Use MuJoCo's named muscle enum for native MyoSuite actuator classification.
- **Next step:** Root will publish the reviewed branch; retain physiology and replay qualification gates.

### DL-#12243 · Repair Matched-Supplement Density Figure Clearance

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12243
- **Branch:** codex/shot-pattern-main-integration
- **PR:** #12418
- **Paths:** src/tools/shot_pattern_analysis/matched_supplement_figures.py,tests/tools/shot_pattern_analysis/test_matched_supplement_pipeline.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/12243-increase-matched-supplement-density-figu.md)
- **Summary:** Increase matched-supplement density figure gutters to clear legends, panel titles, and colorbars.
- **Next step:** Complete the final canonical gate and live publication verification, then deliver through protected merge; no merge claimed yet.

### DL-#12242 · Keep Research PDF Headings With Their Following Tables and Paragraphs

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12242
- **Branch:** codex/shot-pattern-main-integration
- **PR:** #12418
- **Paths:** see #12418
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/12242-keep-research-pdf-headings-with-their-fo.md)
- **Summary:** Keep Research PDF Headings With Their Following Tables and Paragraphs
- **Next step:** Complete the final canonical gate and live publication verification, then deliver through protected merge; no merge claimed yet.

### DL-#12236 · Add Source-Bound Native Diagnostic Admission, Runtime, and Qualification Contract.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12236
- **Branch:** feat/native-diagnostic-contract-12236
- **PR:** #12250
- **Paths:** see #12250
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`f131464b`; collated from changes/12236-add-source-bound-native-diagnostic-admis.md)
- **Summary:** Add source-bound native diagnostic admission, runtime, and qualification contract.
- **Next step:** Complete native regression and governed checks, then integrate the validated diagnostic contract.

### DL-#12233 · Bound Moco Optimizer States to Source Moving-Path Function Domains

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12233
- **Branch:** fix/native-path-connection-finalization-12233-20261012
- **PR:** #12318, #12228
- **Paths:** see #12318
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`49ef898b`; collated from changes/12233-source-bound-moving-path-chart-audit.md)
- **Summary:** Bound Moco optimizer states to source moving-path function domains
- **Next step:** Complete repository gates and review bounded diagnostic scope; no replay or global workless qualification is granted.

### DL-#12231 · Bind Native OpenSim Visual Meshes to a Strict Source-Owned Resource Closure

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12231
- **Branch:** fix/12231-source-bound-visual-assets
- **PR:** #12232
- **Paths:** .prettierignore,src/engines/physics_engines/opensim/python/tour_matching/native_visual_resources.py,src/engines/physics_engines/opensim/python/tour_matching/native_muscle_bundle.py,tests/opensim/test_native_muscle_bundle.py,manuals/upstreamdrift/chapters/13-native-muscle-replay.qmd,docs/agent_context/README.md,docs/agent_context/index.html,src/engines/native_replay_contracts.py,src/engines/physics_engines/myosuite/python/native_direct_model_replay.py,tests/unit/engines/test_native_replay_contracts.py,docs/development/feedback_controls/F07_VISUAL_RESOURCE_CLOSURE_TURNOVER.md
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`b2ff2d1c`; collated from changes/12231-bind-native-opensim-visual-meshes-to-a-s.md)
- **Summary:** Bind native OpenSim visual meshes to a strict source-owned resource closure
- **Next step:** Complete native and repository gates, then review the draft PR.

### DL-#12225 · Explicitly Unqualified Native Moco Feasibility

- **State:** in_review
- **Owner:** codex
- **Issue:** #12225
- **Branch:** feat/f07-unqualified-moco-feasibility-12225
- **PR:** #12228
- **Paths:** see #12228
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`49ef898b`; collated from changes/12225-unqualified-moco-feasibility.md)
- **Summary:** Keep explicitly unqualified native Moco feasibility separate from solve convergence and failed or incomplete independent replay
- **Next step:** Analyze the retained source-bound activation-law residual and continuous control representation without changing replay limits; then validate controller-to-independent-open-loop replay before binding full capture inputs

### DL-#12222 · Club-Face Fixtures Keep the Capture Address Toe-Out: The Swing IK and Every Re-Solve Plant the Stance at the Calibrated Address When Foot Progression Is on; Fixtures Regenerated With --Foot-Progression Capture (Driver 16.64/3.57 vs 16.37/4.03 Deg, 7-Iron 15.43/-0.96 vs 15.00/-0.37 Deg)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12222
- **Branch:** claude/12222-fixture-toe-out
- **PR:** #12230
- **Paths:** see #12230
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`9e5f87c4`; collated from changes/12222-club-face-fixtures-keep-the-capture-addr.md)
- **Summary:** Club-face fixtures keep the capture address toe-out: the swing IK and every re-solve plant the stance at the calibrated address when foot progression is on; fixtures regenerated with --foot-progression capture (driver 16.64/3.57 vs 16.37/4.03 deg, 7-iron 15.43/-0.96 vs 15.00/-0.37 deg)
- **Next step:** Merge after #12223; then #12221 grip calibration on the new fixtures

### DL-#12221 · Adopt the Per-Capture Grip Rotation on the Fixed Driver Club Cluster and Regenerate the Three Driver Ground-Support Receipts, the Driver OpenSim Export, the Driver Club-Face Fixture (Meets GCV-20) and the MSK Driver Grip Calibration; the 7-Iron Refit Is Deferred Because Its Same-Input Export Fails GCV-20 and OSV-10

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12221
- **Branch:** claude/12221-grip-adopt
- **PR:** #12274, #12265, #12270, #12251
- **Paths:** see #12274
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0f0d7ee7`; collated from changes/12221-opt-in-fix-club-markers-keeps-the-club-c.md)
- **Summary:** Adopt the per-capture grip rotation on the fixed driver club cluster and regenerate the three driver ground-support receipts, the driver OpenSim export, the driver club-face fixture (meets GCV-20) and the MSK driver grip calibration; the 7-iron refit is deferred because its same-input export fails GCV-20 and OSV-10
- **Next step:** Owner decision on --fix-club-markers, then apply the per-capture grip receipts and regenerate receipts

### DL-#12211 · Generic Shoulder Triangle-Lamina Candidate

- **State:** in_review
- **Owner:** codex
- **Issue:** #12211
- **Branch:** fix/f07-parent-receipt-12198
- **PR:** #12198
- **Paths:** scripts/opensim/triangle_lamina_provider_worker.py,scripts/opensim/derive_generic_shoulder_lamina.py,src/engines/physics_engines/opensim/python/tour_matching/generic_shoulder_lamina.py,src/engines/physics_engines/opensim/python/tour_matching/native_body_inertia.py,src/shared/python/core/physical_inertia.py,src/shared/python/estimation/dime_global_calibration.py,tests/opensim/test_generic_shoulder_lamina.py,tests/unit/core/test_physical_inertia.py,docs/development/feedback_controls/F07_GENERIC_SHOULDER_LAMINA_TURNOVER.md,manuals/upstreamdrift/chapters/54-generic-shoulder-triangle-lamina.qmd,changes/12211-add-a-separately-identified-source-bound.md,src/config/impact_acceptance.json,docs/development/impact_acceptance_matrix.md,docs/development/feedback_controls/F07_PROVIDER_12246_IMPACT_SERVED_RECEIPT.json
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`7b925718`; collated from changes/12211-add-a-separately-identified-source-bound.md)
- **Summary:** Add a separately identified source-bound generic shoulder triangle-lamina candidate with native physical and sampled kinematic checks
- **Next step:** Keep the candidate diagnostic while remaining force/work, replay, and muscular capture gates are resolved

### DL-#12210 · Give the 7-Iron Anthropometric Document Its Own Clubhead Marker Cluster (Per-Club Club-Markers File Read by the Builder); Head-Triad Residual on the 7-Iron Capture 42.4 to 0.24 mm; Both 7-Iron Receipts Regenerated (Nominal IK 67.9 to 45.2 mm; Calibrated Dynamics 54.5 to 109.9 mm, Owner Decision Pending)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12210
- **Branch:** claude/12210-iron-head-triad
- **PR:** #12219
- **Paths:** see #12219
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`4cd2991b`; collated from changes/12210-give-the-7-iron-anthropometric-document.md)
- **Summary:** Give the 7-iron anthropometric document its own clubhead marker cluster (per-club club-markers file read by the builder); head-triad residual on the 7-iron capture 42.4 to 0.24 mm; both 7-iron receipts regenerated (nominal IK 67.9 to 45.2 mm; calibrated dynamics 54.5 to 109.9 mm, owner decision pending)
- **Next step:** Owner decides which calibrated 7-iron receipt is canonical; static-trial calibration depends on its club priors

### DL-#12205 · Require Native Source-Body Physical Admission Before OpenSim Moco Solve

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12205
- **Branch:** feat/f07-moco-inertia-preflight-12205
- **PR:** #12198
- **Paths:** see #12198
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`7b925718`; collated from changes/12205-native-moco-physical-preflight.md)
- **Summary:** Require native source-body physical admission before OpenSim Moco solve
- **Next step:** Qualify an authoritative physical source, then anatomy, passive forces, contact, capture registration and full-horizon independent replay.

### DL-#12203 · Make the Engine-Module Teardown Guard Tolerate a Concurrently Removed Package Binding

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12203
- **Branch:** fix/f09-drake-import-laziness-12203
- **PR:** #12202
- **Paths:** see #12202
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`e42e9307`; collated from changes/12203-make-the-engine-module-teardown-guard-to.md)
- **Summary:** Make the engine-module teardown guard tolerate a concurrently removed package binding
- **Next step:** Merge the PR.

### DL-#12191 · Bind Exact OpenSim Body Properties to Native Physical-Inertia Admission

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12191
- **Branch:** feat/f07-native-body-inertia-12191
- **PR:** #12198
- **Paths:** see #12198
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`7b925718`; collated from changes/12191-native-source-body-inertia.md)
- **Summary:** Bind exact OpenSim Body properties to native physical-inertia admission
- **Next step:** Obtain an authoritative physically valid source, then qualify anatomy, passive forces, contact and full-horizon excitation replay.

### DL-#12178 · Exercise Calibrated Native Marker Replay Through Existing Observation Scoring

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12178
- **Branch:** feat/f09-calibrated-marker-score-12178
- **PR:** #12288
- **Paths:** see #12288
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`9bc8e05f`; collated from changes/12178-calibrated-marker-replay-score.md)
- **Summary:** Exercise calibrated native marker replay through existing observation scoring
- **Next step:** Review the synthetic native replay-to-score handoff without promoting qualification or production attachments.

### DL-#12176 · Verify Exact Zero-MTP CustomJoint Reduction on Assembled 557-Muscle Native Source

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12176
- **Branch:** feat/f07-buet-zero-custom-mtp-12176
- **PR:** #12181
- **Paths:** see #12181
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`cd25e8ab`; collated from changes/12176-source-bound-zero-custom-mtp-reduction.md)
- **Summary:** Verify exact zero-MTP CustomJoint reduction on assembled 557-muscle native source
- **Next step:** Merge after subtalar and assembly prerequisites; independently qualify original-to-final composition and Moco initialization.

### DL-#12173 · Replay Constrained Mixed OpenSim Moco Inputs Without Hiding Native State Projection

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12173
- **Branch:** feat/f07-constrained-mixed-native-12173
- **PR:** #12184
- **Paths:** see #12184
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`6af5715f`; collated from changes/12173-constrained-mixed-native-moco.md)
- **Summary:** Replay Constrained Mixed OpenSim Moco Inputs Without Hiding Native State Projection
- **Next step:** Review native regression and retain production scientific blockers.

### DL-#12167 · Verify Exact Zero-Subtalar CustomJoint Reduction With Sampled Native 557-Muscle Mechanics

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12167
- **Branch:** feat/f07-buet-zero-subtalar-12167
- **PR:** #12172
- **Paths:** see #12172
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`1bef5534`; collated from changes/12167-verify-exact-zero-subtalar-customjoint-r.md)
- **Summary:** Verify exact zero-subtalar CustomJoint reduction with sampled native 557-muscle mechanics
- **Next step:** Merge after BUET-Hamner and zero-MTP prerequisites; compose locks separately.

### DL-#12164 · Add Source-Bound Provisional Marker Calibration Artifacts

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12164
- **Branch:** feat/f09-marker-calibration-12164
- **PR:** #12288
- **Paths:** see #12288
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`9bc8e05f`; collated from changes/12164-add-source-bound-provisional-marker-cali.md)
- **Summary:** Add source-bound provisional marker calibration artifacts
- **Next step:** Review and merge the stacked marker calibration artifact PR; preserve all heldout and qualification gates.

### DL-#12160 · Qualify Shot-Pattern Validation Helpers for Repository Gates

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12160
- **Branch:** codex/shot-pattern-main-integration
- **PR:** #12418
- **Paths:** see #12418
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/12160-qualify-shot-pattern-validation-helpers.md)
- **Summary:** Qualify shot-pattern validation helpers for repository gates
- **Next step:** Complete the final canonical gate and live publication verification, then deliver through protected merge; no merge claimed yet.

### DL-#12159 · Correct Phase-Two Article Source Fidelity and Unsupported Review Claims

- **State:** in_progress
- **Owner:** unassigned
- **Issue:** #12159
- **Branch:** codex/shot-pattern-matched-delivery
- **PR:** #12418, #12480
- **Paths:** see #12418
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`639f5570`; collated from changes/12159-reconcile-shot-pattern-analysis-paper-pr.md)
- **Summary:** Correct phase-two article source fidelity and unsupported review claims
- **Next step:** Numerical article repair independently verified; resolve external compiler issue #12120 and qualify rendered publication before ready PR delivery

### DL-#12158 · Bind Observation Scores to Native Marker Replay Evidence

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12158
- **Branch:** feat/f09-observation-receipts-12158
- **PR:** #12288
- **Paths:** see #12288
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`9bc8e05f`; collated from changes/12158-f09-native-marker-score-binding.md)
- **Summary:** Bind observation scores to native marker replay evidence
- **Next step:** Keep qualification unpromoted; add further native consumers only with their own source-bound replay evidence.

### DL-#12148 · Report OpenSim Golf Muscle Support Only From Source-Bound Native Readback

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12148
- **Branch:** feat/f09-opensim-native-muscle-capability-12148
- **PR:** #12288
- **Paths:** see #12288
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`9bc8e05f`; collated from changes/12148-report-opensim-golf-muscle-support-only.md)
- **Summary:** Report OpenSim golf muscle support only from source-bound native readback
- **Next step:** Complete focused CI and review native capability contract

### DL-#12143 · Bind Declared Native Constrained Muscle States to Moco and Independent Excitation Replay

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12143
- **Branch:** feat/f07-constrained-muscle-replay-12143
- **PR:** #12163
- **Paths:** see #12163
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5e5757ef`; collated from changes/12143-constrained-native-muscle-moco-replay.md)
- **Summary:** Bind declared native constrained muscle states to Moco and independent excitation replay
- **Next step:** Review stacked child after its declared cold-start prerequisite and retain locked-source and production scientific blockers.

### DL-#12136 · Declare and Verify Native OpenSim Constrained Cold Starts With Exact Mechanical Replay Diagnostics

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12136
- **Branch:** feat/f07-constrained-cold-start-12136
- **PR:** #12163
- **Paths:** see #12163
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5e5757ef`; collated from changes/12136-declare-and-verify-native-opensim-constr.md)
- **Summary:** Declare and verify native OpenSim constrained cold starts with exact mechanical replay diagnostics
- **Next step:** Review the focused child PR and preserve production model blockers.

### DL-#12132 · Covariance Caption Containment and Verified Publication

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12132
- **Branch:** codex/shot-pattern-main-integration
- **PR:** #12418
- **Paths:** src/tools/shot_pattern_analysis/matched_supplement_figures.py,scripts/research/shot_pattern_analysis/supplement_publication.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/12132-shot-pattern-caption-publication.md)
- **Summary:** Wrap complete covariance captions and add selective, provenance-checked figure revision publication.
- **Next step:** Complete the final canonical gate and live publication verification, then deliver through protected merge; no merge claimed yet.

### DL-#12131 · Paired Density Row Spacing

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12131
- **Branch:** codex/shot-pattern-main-integration
- **PR:** #12418
- **Paths:** src/tools/shot_pattern_analysis/matched_supplement_figures.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/12131-shot-pattern-density-row-spacing.md)
- **Summary:** Increase paired density row spacing while preserving scientific arrays, limits and population identities.
- **Next step:** Complete the final canonical gate and live publication verification, then deliver through protected merge; no merge claimed yet.

### DL-#12129 · Keep Report Numerical Tokens Intact Using Compact Condition Keys and Separate Scoring and Population Tables

- **State:** in_progress
- **Owner:** unassigned
- **Issue:** #12129
- **Branch:** codex/shot-pattern-matched-delivery
- **PR:** #12418
- **Paths:** see #12418
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/12129-keep-report-numerical-tokens-intact-usin.md)
- **Summary:** Keep report numerical tokens intact using compact condition keys and separate scoring and population tables
- **Next step:** Review complete assembled report PDF after supplemental rendering and publish ready PR

### DL-#12128 · Derive Final Shot-Pattern Conclusions From Qualified Selected Results and Preserve Protocol and Population Identities

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12128
- **Branch:** codex/shot-pattern-main-integration
- **PR:** #12418
- **Paths:** see #12418
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/12128-derive-final-shot-pattern-conclusions-fr.md)
- **Summary:** Derive final shot-pattern conclusions from qualified selected results and preserve protocol and population identities
- **Next step:** Complete the final canonical gate and live publication verification, then deliver through protected merge; no merge claimed yet.

### DL-#12117 · Regenerate the Club-Face Swing Fixtures on the Current Spec (Feet Flat Instead of 9-41 cm up), Fix Impact_Frame's Missing Ball-Radius Check, Check the Address Face Against the Capture's Own Face, Regenerate the OSV-9 MSK Grip Calibration, and Strict-Xfail the 7-Iron Impact Face (#12221)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12117
- **Branch:** claude/12117-club-face-fixtures
- **PR:** #12223, #12145
- **Paths:** see #12223
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`3a63d61c`; collated from changes/12117-split-the-reference-zmp-rates-at-the-bal.md)
- **Summary:** Regenerate the club-face swing fixtures on the current spec (feet flat instead of 9-41 cm up), fix impact_frame's missing ball-radius check, check the address face against the capture's own face, regenerate the OSV-9 MSK grip calibration, and strict-xfail the 7-iron impact face (#12221)
- **Next step:** Diagnose the computed-torque replay's speed loss through the release (reference +0.9 %, replay -9.2 %); regenerate club-face fixtures once #11960 and #12125 are on main

### DL-#12064 · Correct SG Optimizer CLI Integration Invocation

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12064
- **Branch:** fix/12064-sg-optimizer-cli-test
- **PR:** #12144
- **Paths:** see #12144
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`7eac477a`; collated from changes/12064-correct-sg-optimizer-cli-integration-inv.md)
- **Summary:** Correct SG optimizer CLI integration invocation
- **Next step:** Merge the test-only CLI invocation fix.

### DL-#12052 · MyoSuite Leg Retarget Maps the Hip Orientation Through the Calibrated Hip Frames and Fixes the Left Knee Sign; MyoSuite Address Feet Within 2 Deg on Both Captures

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12052
- **Branch:** claude/osv-6e-myosuite-hip-retarget
- **PR:** #12145
- **Paths:** see #12145
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`3a63d61c`; collated from changes/12052-myosuite-leg-retarget-maps-the-hip-orien.md)
- **Summary:** MyoSuite leg retarget maps the hip orientation through the calibrated hip frames and fixes the left knee sign; MyoSuite address feet within 2 deg on both captures
- **Next step:** Merge after #12051; #12057 (right knee range) is open

### DL-#12039 · Drake IK Uses MuJoCo's Principal-Log Weld Rotation Row and Exact Jacobian; Drake and MuJoCo Address Stages Now Agree Within 0.02 mm on Both Tour Captures and the Capture-A Drake Receipt Is Regenerated

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12039
- **Branch:** claude/12039-drake-address-parity
- **PR:** #12216
- **Paths:** see #12216
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`7da0a298`; collated from changes/12039-drake-ik-uses-mujoco-s-principal-log-wel.md)
- **Summary:** Drake IK uses MuJoCo's principal-log weld rotation row and exact Jacobian; Drake and MuJoCo address stages now agree within 0.02 mm on both tour captures and the capture-A Drake receipt is regenerated
- **Next step:** Merge; Drake now tracks MuJoCo through IK and dynamics on capture-A

### DL-#12030 · Fail Closed When the Clubhead Triad Offsets Do Not Match the Captured Triad; the Nominal 7-Iron IK Drops From 264.8 to 67.9 mm (Frame 0 70.2 mm) and Its Dynamics From 1355 to 79.2 mm, With Both Nominal Receipts Regenerated

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12030
- **Branch:** claude/12030-nominal-iron-ik
- **PR:** #12208
- **Paths:** see #12208
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`7e105580`; collated from changes/12030-fail-closed-when-the-clubhead-triad-offs.md)
- **Summary:** Fail closed when the clubhead triad offsets do not match the captured triad; the nominal 7-iron IK drops from 264.8 to 67.9 mm (frame 0 70.2 mm) and its dynamics from 1355 to 79.2 mm, with both nominal receipts regenerated
- **Next step:** Correct the 7-iron document's nominal head triad (it is the driver's) in a separate spec-hash change

### DL-#12023 · Add Frozen Source-Bound Native Tracking Costs for Mixed-Command Search.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12023
- **Branch:** feat/f09m-native-tracking-objective-12023
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/12023-native-tracking-objective.md)
- **Summary:** Add frozen source-bound native tracking costs for mixed-command search.
- **Next step:** Integrate bounded command search after stacked replay and contract dependencies.

### DL-#12021 · Extract One Bounded Candidate Search Loop Behind Finite Cost and Signed Hard Margins

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12021
- **Branch:** feat/f02-bounded-callback-12021
- **PR:** #12024
- **Paths:** see #12024
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`e47a21e2`; collated from changes/12021-extract-one-bounded-candidate-search-loo.md)
- **Summary:** Extract one bounded candidate search loop behind finite cost and signed hard margins
- **Next step:** Validate stacked callback PR, then integrate native iron adapter using declared 20 ms source-only reference and fresh promotion.

### DL-#12015 · Use SciPy's Typed Vector Inequality Constraint for F02 SLSQP and Count Its Preparatory Evaluations Within the Existing Budget

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12015
- **Branch:** fix/f02-scipy-typing-12015
- **PR:** #11816
- **Paths:** see #11816
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`1fe02740`; collated from changes/12015-type-f02-slsqp-options-and-inequality-co.md)
- **Summary:** Use SciPy's typed vector inequality constraint for F02 SLSQP and count its preparatory evaluations within the existing budget
- **Next step:** Merge stacked PR after F02 #11816 reaches main and its scoped CI passes.

### DL-#11976 · Add Matched Delivery, Correlated Noise, Density Graphics and Finite-Head Contact Study

- **State:** in_review
- **Owner:** codex
- **Issue:** #11976
- **Branch:** codex/shot-pattern-matched-delivery
- **PR:** #12418
- **Paths:** see #12418
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/11976-shot-pattern-remote-restoration.md)
- **Summary:** Add matched delivery, correlated noise, density graphics and finite-head contact study
- **Next step:** Review the delivery PR; resolve documented repository gates before merge

### DL-#11962 · Observe Exact Native Source Reference Frames and Muscle Path Conventions

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #11962
- **Branch:** feat/f07-reference-conventions-11962
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11962-native-reference-conventions.md)
- **Summary:** Observe exact native source reference frames and muscle path conventions
- **Next step:** Resolve donor release, frame correspondence and resource closure before muscle-model calibration.

### DL-#11955 · Bind Compiled Actuator Command Profiles to Native Replay Execution.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #11955
- **Branch:** feat/f01f-compiled-command-admission-11955
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11955-compiled-actuator-command-admission.md)
- **Summary:** Bind compiled actuator command profiles to native replay execution.
- **Next step:** Merge the stacked F09 compiled-command admission PR after its dependency chain.

### DL-#11950 · Add Bounded Native MyoSuite Model Resource Closure Admission.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #11950
- **Branch:** feat/f09h-native-muscle-replay-11950
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11950-add-bounded-native-myosuite-model-resour.md)
- **Summary:** Add bounded native MyoSuite model resource closure admission.
- **Next step:** Merge the stacked F09h provider and closure PR.

### DL-#11949 · Native Moco Initial-State Bindings

- **State:** in_review
- **Owner:** codex
- **Issue:** #11949
- **Branch:** feat/feedback-moco-state-binding-11791
- **PR:** #12141
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/moco_initial_bindings.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11949-native-moco-initial-bindings.md)
- **Summary:** Apply complete continuous initial-state and scalar-control bounds before native Moco guess creation
- **Next step:** Review the ready stacked PR against native muscle bundle parent #11917; keep unarmed until safe retargeting to main.

### DL-#11942 · Bind Owned Tools Native Envelopes to Actual R2025b Diagnostic Restart and Changed Future-Force Replay

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #11942
- **Branch:** feat/feedback-simscape-owned-native-11942
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11942-owned-native-simscape-replay.md)
- **Summary:** Bind owned Tools native envelopes to actual R2025b diagnostic restart and changed future-force replay
- **Next step:** Publish and validate the owned native adapter, then implement production variants while retaining capture and parity gates

### DL-#11939 · Require Explicit Native Passive-Load Readiness Before Muscle Matching

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #11939
- **Branch:** feat/feedback-passive-readiness-11939
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11939-require-explicit-native-passive-load-rea.md)
- **Summary:** Require explicit native passive-load readiness before muscle matching
- **Next step:** Land focused passive-readiness gate after native constraint observer dependency; retain physiological policy and donor ancestry blockers.

### DL-#11935 · Reconcile Development Log Provenance, Legacy Headings, and Source-Backed Delivery States Without Qualifying Scientific Work.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #11935
- **Branch:** chore/development-log-collation-11935
- **PR:** #12337
- **Paths:** see #12337
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`d8a6e303`; collated from changes/11935-reconcile-development-log-provenance-leg.md)
- **Summary:** Reconcile Development Log provenance, legacy headings, and source-backed delivery states without qualifying scientific work.
- **Next step:** Review the evidence inventory and validator results; merge only after approved review.

### DL-#11933 · Add Actual R2025b Physical and Discrete Saved-Operating-Point Restart Prerequisite

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #11933
- **Branch:** feat/feedback-simscape-replay-11921
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11933-add-actual-r2025b-physical-and-discrete.md)
- **Summary:** Add actual R2025b physical and discrete saved-operating-point restart prerequisite
- **Next step:** Integrate immutable Tools native artifact consumer and production model bindings; retain all parity blockers

### DL-#11923 · Native Constrained-Source Readiness

- **State:** in_progress
- **Owner:** codex
- **Issue:** #11923
- **Branch:** feat/feedback-constrained-source-11923
- **PR:** #11931
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/native_constraint_state.py,tests/opensim/test_native_constraint_state.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`2e3c79b4`; collated from changes/11923-native-constrained-state.md)
- **Summary:** Observe actual native constrained state and preserve unverified full-model restart, lock-target and physiology gates.
- **Next step:** Complete protected review; qualify fixed-lock source policy before full-model replay.

### DL-#11912 · Frozen Training Capture Registration

- **State:** in_review
- **Owner:** claude
- **Issue:** #11912
- **Branch:** fix/11912-reland-frozen-registration
- **PR:** #12363
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/frozen_registration.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`4a1809f9`; collated from changes/11912-frozen-training-registration.md)
- **Summary:** Freeze training-only native capture registration with explicit geometry lineage
- **Next step:** Qualify anatomical correspondences and ground registration before independent muscle-driven capture replay.

### DL-#11908 · Native Muscle Experiment Bundle Development Boundary

- **State:** in_progress
- **Owner:** codex
- **Issue:** #11908
- **Branch:** feat/feedback-opensim-bundle-11908
- **PR:** #12141
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/native_muscle_bundle.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11908-native-muscle-bundle.md)
- **Summary:** Bind native OpenSim muscle replay to the shared experiment bundle
- **Next step:** Complete native state policy review, negative coverage and qualification gates.

### DL-#11903 · Native OpenSim Geometry Admission

- **State:** in_review
- **Owner:** codex
- **Issue:** #11903
- **Branch:** feat/feedback-native-markers-11903
- **PR:** #11911
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/native_marker_geometry.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b21b80a`; collated from changes/11903-native-opensim-geometry.md)
- **Summary:** Bind explicit native OpenSim marker geometry and reject placeholder physical evidence
- **Next step:** Freeze source-backed frame registration and anatomical offsets before independent full-body capture fitting.

### DL-#11900 · Native Pinocchio Replay Development Boundary

- **State:** in_progress
- **Owner:** codex
- **Issue:** #11900
- **Branch:** feat/feedback-pinocchio-native-11900
- **PR:** #12141
- **Paths:** src/engines/physics_engines/pinocchio/python/native_torque_replay.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11900-native-pinocchio-torque-replay.md)
- **Summary:** Add native Pinocchio frozen torque replay through the existing engine
- **Next step:** Qualify shared-adapter regressions and required production model bindings.

### DL-#11867 · Reject Unqualified Concrete Muscle Laws Before Native Initialization

- **State:** in_review
- **Owner:** codex
- **Issue:** #11867
- **Branch:** fix/feedback-exact-law-11867
- **PR:** #11897
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/muscle_replay.py,tests/opensim/test_native_muscle_replay.py,manuals/upstreamdrift/chapters/13-native-muscle-replay.qmd
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`da8228db`; collated from changes/11867-reject-unqualified-concrete-muscle-laws.md)
- **Summary:** Reject unqualified concrete muscle laws before native initialization
- **Next step:** Merge Thelen dependency and retain full-model native qualification gates.

### DL-#11856 · Reproduce Native Moving-Path Force-Length Mismatch and Reject Nonfinite Derivative Evidence

- **State:** in_review
- **Owner:** codex
- **Issue:** #11856
- **Branch:** feat/feedback-path-work-11856
- **PR:** #12228
- **Paths:** scripts/diagnostics/native_path_work.py,src/engines/physics_engines/opensim/python/tour_matching/muscle_qualification.py,tests/opensim/test_native_path_work.py,manuals/upstreamdrift/chapters/24-native-path-work.qmd
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`49ef898b`; collated from changes/11856-reproduce-native-moving-path-force-lengt.md)
- **Summary:** Reproduce native moving-path force-length mismatch and reject nonfinite derivative evidence
- **Next step:** Review force-law alternatives before donor or runtime changes; full F07 qualification remains open.

### DL-#11851 · Cast Numeric Refinement Rows Explicitly in numerics.py to Resolve Mypy Static Type Errors.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #11851
- **Branch:** fix/11851-correct-static-types-shot-pattern
- **PR:** #12452
- **Paths:** see #12452
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`c6ad9cc2`; collated from changes/11851-correct-static-types-shot-pattern.md)
- **Summary:** Cast numeric refinement rows explicitly in numerics.py to resolve mypy static type errors.
- **Next step:** Merge the PR.

### DL-#11840 · Correct Central Impact and Add Three-Club Shot Pattern Analysis

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #11840
- **Branch:** codex/shot-pattern-analysis
- **PR:** #12418
- **Paths:** see #12418
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/11840-correct-central-impact-and-add-three-clu.md)
- **Summary:** Correct Central Impact and Add Three-Club Shot Pattern Analysis
- **Next step:** Complete final scientific gates and publish the ready analysis PR.

### DL-#11838 · Native Drake Replay Development Boundary

- **State:** in_review
- **Owner:** codex
- **Issue:** #11838
- **Branch:** feat/feedback-drake-native-11838
- **PR:** #12141
- **Paths:** src/engines/physics_engines/drake/python/native_torque_replay.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11838-native-drake-torque-replay.md)
- **Summary:** Add restricted native Drake frozen torque replay
- **Next step:** Integrate native candidates and qualify refinement, contact and full-model parity.

### DL-#11837 · Score Native Output Against Frozen Observations With Full Replay Identity

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #11837
- **Branch:** feat/11837-f09b-native-observation-qualification
- **PR:** #12141
- **Paths:** src/engines/feedback_observation_qualification.py,tests/unit/engines/test_feedback_observation_qualification.py,tests/motion_capture/rig/test_tools_session_export.py,docs/development/feedback_controls/IMPLEMENTATION_PLAN.md,docs/development/feedback_controls/F09B-OBSERVATION-QUALIFICATION.md,docs/development/feedback_controls/TURNOVER.md,manuals/upstreamdrift/chapters/13-feedback-comparison.qmd,vendor/ud-tools
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11837-f09b-native-observation-scoring.md)
- **Summary:** Score native output against frozen observations with full replay identity
- **Next step:** Integrate per-engine native outputs and qualification evidence without relaxing gates.

### DL-#11834 · Observe Pinned Native Coupled-Wrist Geometry With Explicit Assembly and Derivative Gates

- **State:** in_review
- **Owner:** claude
- **Issue:** #11834
- **Branch:** fix/11834-reland-wrist-gate
- **PR:** #12362
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/wrist_compatibility.py,tests/opensim/test_native_wrist_compatibility.py,manuals/upstreamdrift/chapters/29-native-wrist-compatibility.qmd
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ba06c0da`; collated from changes/11834-native-wrist-compatibility.md)
- **Summary:** Observe pinned native coupled-wrist geometry with explicit assembly and derivative gates
- **Next step:** Resolve native donor derivative and anatomical registration gates before F07/F08 qualification.

### DL-#11831 · Sample Native Replay Marker Outputs on the Observation Clock

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #11831
- **Branch:** feat/11793-f09a-output-sampling
- **PR:** #11830
- **Paths:** src/shared/python/motion_matching/replay_metrics.py,src/shared/python/motion_matching/**init**.py,tests/unit/motion_matching/test_replay_observation_sampling.py,docs/development/feedback_controls/F09A-OBSERVATION-SAMPLING.md,docs/development/feedback_controls/TURNOVER.md,manuals/upstreamdrift/chapters/13-feedback-comparison.qmd
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`fb692749`; collated from changes/11831-f09a-observation-output-sampling.md)
- **Summary:** Sample native replay marker outputs on the observation clock
- **Next step:** Integrate sampled native outputs with F09 comparison levels without changing frozen acceptance criteria.

### DL-#11826 · Native Thelen Replay Boundary

- **State:** in_review
- **Owner:** codex
- **Issue:** #11826
- **Branch:** feat/feedback-native-thelen-11826
- **PR:** #11829
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/muscle_replay.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`cde5a1ec`; collated from changes/11826-native-thelen-replay.md)
- **Summary:** Admit compliant native Thelen muscle replay with explicit subtype and state policy
- **Next step:** Qualify ignored-state modes, native full-body resources and anatomy; retain F07/F08 scientific gates.

### DL-#11823 · Native Torque Replay Development Boundary

- **State:** in_review
- **Owner:** codex
- **Issue:** #11823
- **Branch:** feat/feedback-native-replay-11790
- **PR:** #11836
- **Paths:** src/engines/physics_engines/mujoco/python/native_torque_replay.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`9d336118`; collated from changes/11823-native-torque-replay.md)
- **Summary:** Add a full-state native MuJoCo consumer of the Tools torque replay bundle
- **Next step:** Pin merged Tools T01 and qualify remaining engine and native muscle/contact replay paths.

### DL-#11821 · Add Synthetic Sparse Inverse-Dynamics Collocation Spike and Benchmark Existing Multiple Shooting With Fresh Replay Diagnostics

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #11821
- **Branch:** feat/f03-constrained-ocp-11787
- **PR:** #11822
- **Paths:** src/shared/python/motion_matching/sparse_collocation_spike.py,tests/unit/motion_matching/test_sparse_collocation_spike.py,manuals/upstreamdrift/chapters/17-sparse-collocation-benchmark.qmd
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`72a58405`; collated from changes/11821-sparse-collocation-spike.md)
- **Summary:** Add synthetic sparse inverse-dynamics collocation spike and benchmark existing multiple shooting with fresh replay diagnostics
- **Next step:** Integrate early F06 native replay validator and D02 frozen private protocol before production backend selection

### DL-#11819 · Add Pinned Native OpenSim Muscle-Model Structural Admission Without Anatomical Qualification

- **State:** in_review
- **Owner:** codex
- **Issue:** #11819
- **Branch:** feat/feedback-native-admission-11819
- **PR:** #11825
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/model_admission.py; tests/opensim/test_model_admission.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`1b149c36`; collated from changes/11819-native-model-admission.md)
- **Summary:** Add pinned native OpenSim muscle-model structural admission without anatomical qualification
- **Next step:** Review native inventory; continue licensed F07 model provisioning and independent qualification.

### DL-#11815 · Native Muscle Contact Replay

- **State:** in_review
- **Owner:** codex
- **Issue:** #11815
- **Branch:** feat/feedback-native-contact-11815
- **PR:** #11820
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/muscle_replay.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`194cf468`; collated from changes/11815-native-muscle-contact.md)
- **Summary:** Add explicit model-owned native contact policies and complete wrench evidence to muscle replay
- **Next step:** Qualify full-body anatomy, calibrated ground/grip contact and full-horizon private mocap replay.

### DL-#11810 · Prevent Parameter-Only Muscle Audits From Fabricating Native Qualification Evidence

- **State:** in_review
- **Owner:** codex
- **Issue:** #11810
- **Branch:** fix/feedback-muscle-evidence-11791
- **PR:** #11814
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/muscle_qualification.py; tests/opensim/test_muscle_cmc.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5ef39d92`; collated from changes/11810-muscle-evidence-guard.md)
- **Summary:** Prevent parameter-only muscle audits from fabricating native qualification evidence
- **Next step:** Review evidence guard; continue genuine native qualification under F07 #11791.

### DL-#11809 · Native Muscle Replay Development Boundary

- **State:** in_review
- **Owner:** codex
- **Issue:** #11809
- **Branch:** feat/feedback-opensim-11791
- **PR:** #11812
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/muscle_replay.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`45615364`; collated from changes/11809-native-muscle-replay.md)
- **Summary:** Add early native continuous-state muscle replay and reconcile reviewed epic milestone dependencies
- **Next step:** Qualify explicit native contact/state policies and the full-body muscle model; retain F07/F08 qualification gates.

### DL-#11786 · Add Phase-Gated Tangent Feedback With Constrained Actuator Allocation and Auditable Applied-Torque Evidence

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #11786
- **Branch:** feat/f02-distributed-feedback-11786
- **PR:** #11816
- **Paths:** src/shared/python/motion_matching/distributed_feedback.py,tests/unit/motion_matching/test_distributed_feedback.py,manuals/upstreamdrift/chapters/16-distributed-feedback.qmd
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`1fe02740`; collated from changes/11786-distributed-feedback.md)
- **Summary:** Add phase-gated tangent feedback with constrained actuator allocation and auditable applied-torque evidence
- **Next step:** Integrate native tangent/task adapters and independently replay exact applied torque under F09 gates

### DL-#11785 · Freeze Six-Engine Feedback Comparison Admission and Independent Replay Prerequisites

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #11785
- **Branch:** feat/f01-six-engine-contracts-11785
- **PR:** #11808
- **Paths:** src/engines/feedback_comparison.py,src/engines/model_inventory.py,src/config/engine_model_inventory.json,manuals/upstreamdrift/chapters/13-feedback-comparison.qmd
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`a0a29db1`; collated from changes/11785-feedback-comparison.md)
- **Summary:** Freeze six-engine feedback comparison admission and independent replay prerequisites
- **Next step:** F09 maps versioned Tools T01 bundles to native full-horizon replay receipts before numerical qualification.

### DL-#11784 · Document Researched Distributed Feedback Epic, Six-Engine Parity and Muscle-Driven OpenSim Target

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #11784
- **Branch:** docs/feedback-controls-11784
- **PR:** #11798
- **Paths:** docs/development/feedback_controls
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`b488dbbd`; collated from changes/11784-feedback-controls-design.md)
- **Summary:** Document researched distributed feedback epic, six-engine parity and muscle-driven OpenSim target
- **Next step:** Start F01 inventory and gate freeze; preserve MOSAIC and parity authorities

### DL-#11745 · LIFT-5 Slice 1: Back-Squat Same-Input Bundle From the Movement Optimizer via Full-Body Marker IK, With Provenance, Regeneration Script and MuJoCo Replay Smoke Test

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #11745
- **Branch:** claude/lift-5-squat-bundle
- **PR:** #12245, #12340
- **Paths:** see #12245
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ba48dafa`; collated from changes/11745-lift-5-slice-cross-engine-open-loop-bar.md)
- **Summary:** LIFT-5 slice 1: back-squat same-input bundle from the Movement Optimizer via full-body marker IK, with provenance, regeneration script and MuJoCo replay smoke test
- **Next step:** Cross-engine bar-path replay parity of the back-squat bundle in MuJoCo, Pinocchio and Drake; then the other four lifts.

### DL-#11737 · OSV-6 Lead Foot: Lateral-Marker Zero-Twist Correction (Knee_Flexion_Axis) and Leg-Marker Calibration Constraints Under --Foot-Progression; MuJoCo Address Toe-Out Within 2 Deg on Both Captures (Driver +0.2/-0.3, Iron -0.0/-1.4 Deg); Canonical Receipts Unchanged (Default-On Correction Is a Recorded Failed Experiment)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #11737
- **Branch:** claude/osv-6b-leg-azimuth
- **PR:** #12049, #12051
- **Paths:** see #12049
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b2ef7a6`; collated from changes/11737-osv-6-record-the-full-address-coordinate.md)
- **Summary:** OSV-6 lead foot: lateral-marker zero-twist correction (knee_flexion_axis) and leg-marker calibration constraints under --foot-progression; MuJoCo address toe-out within 2 deg on both captures (driver +0.2/-0.3, iron -0.0/-1.4 deg); canonical receipts unchanged (default-on correction is a recorded failed experiment)
- **Next step:** Lead foot still 15-32 deg out at address in MuJoCo/Drake: trade hip adduction against stance width; heel-proxy malleolus correction; Pinocchio/MyoSuite stills

### DL-#11729 · MyoSuite Anthro Map: NeckInputY Drives Neck_Flexion (Sign -1 by FK), NeckInputX (Lateral Bending) Unmapped, Neck_Rotation No Longer Overwritten

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #11729
- **Branch:** claude/osv-3-myosuite-neck-flexion
- **PR:** #11768, #11772, #11979, #12154, #12194
- **Paths:** see #11768
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`31b4252b`; collated from changes/11729-osv-3d-closed-loop-gaze-neck-fd-neck-gaz.md)
- **Summary:** MyoSuite anthro map: NeckInputY drives neck_flexion (sign -1 by FK), NeckInputX (lateral bending) unmapped, neck_rotation no longer overwritten
- **Next step:** Merge after #11768; FK-fit scapula absorption and pelvis root orientation for MyoSuite replay

### DL-#11659 · Typed Attachment Ports, Part Catalog and Drag-and-Drop Assembly Canvas in the Frankenstein Editor (CMB-8, CMB-9)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #11659
- **Branch:** feat/frankenstein-ports-dnd-cmb-11659
- **PR:** #11702
- **Paths:** see #11702
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b607519`; collated from changes/11659-typed-attachment-ports-part-catalog-and.md)
- **Summary:** Typed attachment ports, part catalog and drag-and-drop assembly canvas in the Frankenstein editor (CMB-8, CMB-9)
- **Next step:** Web drag-and-drop (CMB-10, #11661)

### DL-#11792 — F08: Match Mocap With Muscle Excitations and Verify Pure OpenSim Replay

- **State:** in_progress
- **Owner:** codex
- **Issue:** #11792 (parent #11784; children #12342, #12343, #12353)
- **PR:** #12344 (draft); related #12355, #12318, #12402, #12407, #12386, #12442, #12401
- **Branch:** `fix/native-path-connection-finalization-12233-20261012`
- **Started:** 2026-10-09 (parent issue opened)
- **Paths:** `changes/12233-bound-moco-optimizer-states-to-source-mo.md`; `docs/agent_context/README.md`; `docs/agent_context/index.html`; `docs/development/feedback_controls/F07_PATH_WORK_TURNOVER.md`; `manuals/upstreamdrift/chapters/24-native-path-work.qmd`; `src/engines/physics_engines/opensim/python/tour_matching/native_moco_preparation.py`; `tests/unit/engines/opensim/test_native_path_optimizer_domains.py`
- **Summary:** This software/readiness work moves native connection finalization before optimizer-domain and passive audits. It does not claim a solve, optimum, learned replay, or physical qualification; all 17 registered model/variant rows remain unqualified.
- **Next step:** Review and integrate the stateless policy extraction, then implement an explicitly mapped muscle-allocation bridge and validate controller-off replay; native model qualification remains separate.
- **Last verified:** 2026-10-14 (`85b810d4`; collated from changes/11792-stateless-feedback-policy-12390.md)

### DL-#11719 · GCV-13 Slice 3: Decorative Address Ball in the MuJoCo, OpenSim and MyoSuite Native Export

- **State:** in_review
- **Owner:** claude
- **Issue:** #11719 (epic #11706)
- **Branch:** claude/gcv-13c-native-ball-backends
- **PR:** not created (stacked on #12114, which is stacked on #11992), #12140, #12332, #12283, #12297, #12481
- **Paths:** src/tools/native_viewer_export/backends/\_ball.py, src/tools/native_viewer_export/backends/\_worker_job.py, src/tools/native_viewer_export/backends/\_subprocess.py, src/tools/native_viewer_export/backends/mujoco_native.py, src/tools/native_viewer_export/backends/mujoco_worker.py, src/tools/native_viewer_export/backends/myosuite_arena.py, src/tools/native_viewer_export/backends/myosuite_worker.py, src/tools/native_viewer_export/backends/opensim_simbody.py, src/tools/native_viewer_export/backends/opensim_worker.py, src/tools/native_viewer_export/README.md, src/config/feature_parity.json, tests/unit/tools/native_viewer_export/test_ball_native_backends.py
- **Started:** 2026-10-10
- **Last verified:** 2026-10-14 (`c0b6c94b`; collated from changes/11719-promote-render-club-head-and-ball-to-par.md)
- **Summary:** GCV-13 adds the resolved decorative address ball to remaining engine backends and worker transport; OpenSim attaches build-time ground geometry. The web ClubHead follow-up remains open; this presentation change does not alter physics.
- **Next step:** Push `claude/gcv-13c-native-ball-backends` and open the PR stacked on #12114.

### DL-#10381 · Reconciled Scopes From Duplicate Source Rows

- **State:** in_progress
- **Owner:** claude (PR #10599; branch feat/10381-crocoddyl-pelvis-yaw-g1) | claude (PR not recorded in source branch evidence/10381-g1-b100)
- **Issue:** #10381
- **PR:** #10599
- **Branch:** feat/10381-crocoddyl-pelvis-yaw-g1; evidence/10381-g1-b100
- **Paths:** src/engines/physics_engines/pinocchio/python/crocoddyl_problem.py; src/engines/physics_engines/pinocchio/python/crocoddyl_action.py; src/engines/physics_engines/pinocchio/python/full_body_fit.py; tests/unit/motion_matching/test_crocoddyl_pelvis_yaw.py; SPEC.md; docs/development/DEVELOPMENT_LOG.md; src/shared/python/motion_matching/ledger.py; evidence/matched; docs/development/full_body_models/evidence/acceptance/verdicts_2026-09.json; docs/development/matched_swing_program
- **Started:** 2026-09-18
- **Summary:** Added analytic pelvis-yaw cost/gradient to Crocoddyl full-body fitting, CLI and receipts. The recorded candidate remains G1 REJECTED (early/terminal error, yaw, penetration and weight-fraction limits); no physical qualification.
- **Next step:** G1 remains REJECTED. Continue the recorded 300-iteration continuation, terminal/club weight x3, MS-20 contact identification for 17 mm penetration, and pelvis-yaw cost investigation; one receipt per lever.
- **Evidence:** tests/unit/motion_matching/test_crocoddyl_pelvis_yaw.py.
- **Last verified:** 2026-10-11 (`944d798e`; collated from docs/development/archive/DEVELOPMENT_LOG_RECONCILIATION_SOURCE_20261011.md)

### DL-#11285 · Force and Torque Overlays for Every Engine

- **Started:** not recorded in source
- **State:** in_review
- **Owner:** antigravity
- **Issue:** #11285; children #11286–#11315
- **PR:** #11397
- **Branch:** `feat/fto-30-gallery-docs-11315`
- **Paths:** `scripts/render_force_overlay_gallery.py`, `tests/visual/force_overlay/**`, `docs/user_guide/force_overlay.md`, `src/config/feature_parity.json`
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** One engine-agnostic force/torque contract, glyph builder and renderer adapters; real providers for MuJoCo, Drake, Pinocchio, OpenSim and Simscape; tension/compression producers for every engine; arrows, shaded model and legend over source footage. [Plan](force_torque_overlay_epic.md).
- **Next step:** FTO core complete and shipped. Follow-up #11313 is tracked in DL-#11313.

### DL-#11614 · Same-Input Parity LaTeX Research Reference (P-1 Pointwise Result)

- **State:** in_progress
- **Owner:** unassigned
- **Issue:** #11614
- **Branch:** docs/sip-9-parity-reference-11614
- **PR:** #11616, #11625, #12485, #11806, #12493
- **Paths:** see #11616
- **Started:** 2026-10-07
- **Last verified:** 2026-10-14 (`4aca3f1a`; collated from changes/11614-synchronize-same-input-parity-calculatio.md)
- **Summary:** Same-input parity LaTeX research reference (P-1 pointwise result)
- **Next step:** Extend with P-2..P-5 trajectory and closed-loop results

### DL-#11532 · MOSAIC Model-Aware Multi-Trial Matching Kernels, Methods Reference and DIME (#11421) Audit

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #11532
- **Branch:** feat/mosaic-model-aware-matching
- **PR:** #11555
- **Paths:** see #11555
- **Started:** 2026-10-05
- **Last verified:** 2026-10-05 (`90df1e03`; collated from changes/11532-mosaic-model-aware-multi-trial-matching.md)
- **Summary:** MOSAIC model-aware multi-trial matching kernels, methods reference and DIME (#11421) audit
- **Next step:** Merge PR; start MOSAIC-01/02/13 (benchmark freeze, Pinocchio regressor provider, DIME defect remediation).

### DL-#10950 · Exploratory GS3DX Simscape Model: Quaternion Joints and Full-Body Under the 1,000-Block Budget

- **State:** in_progress
- **Owner:** codex (Gemini 3.8 CLI delegates)
- **Issue:** #10950 (children #10951–#10959, #10979, #10985, #10986, #11011)
- **PR:** #11256 (draft); original #11179 remains separate.
- **Branch:** `feat/simscape-matching-review-main-20261002` (review continuation of `feat/simscape-gs3dx-exploratory`)
- **Paths:** `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/`, `docs/research/simscape_matching_reference/`, `AGENTS.md`
- **Started:** 2026-09-26
- **Last verified:** 2026-10-03: production observation-export checkpoint passes 39 native contracts plus actual two-frame owner cache identity/reuse integration. Private rotational continuity helper passes 16 RED/GREEN tests; exact inactive whole-output parity passes on eight owner and 53 tour frames. Sparse/dense experiments recheck all 305 native poses and exit naturally; marker/step tradeoffs remain adverse and unpromoted, including owner root motion and tour wrist/root increments. Six owner-session originals and archive are hash-verified on a local machine, with deterministic supplemental acquisition provenance. Source-only LaTeX review reconciles current versus historical candidates, source-mask tensor indexing and source-verified balance units. See continuity_experiment_checkpoint_20261003.json. Normal current-main integration includes 4eb3051c3; optional-engine verification records five MuJoCo passes/four missing-mj_copyData failures and two OpenSim skips on this host. MATLAB sources are unchanged; protected CI/review and latest PDF qualification remain open.
- **Summary:** Agent-editable GS3DX clones, LaTeX reference and documentation policy, truthful serialized execution, Human ellipsoid geometry adapter, joint-role mapping and calibrated full foot orientation. A is the tour reference; O is the owner capture. Marker residuals, offsets, orientation and provenance remain independently visible; videos are IK, not qualified forward dynamics. Additional recorded 2026-10-02 trial evidence: controlled 0-50 ms tour rejected at peak 2.000907 BW; separate 20-50 ms ramp runs met only five one-second hold gates (tour 3.176 mm / 0.836 deg; owner 2.321 mm / 0.699 deg). These limited checks do not qualify independent full-body dynamics.
- **Next step:** Qualify full-rate whole-capture A/O paths with proper native derivatives, seed transitions, gap/branch diagnostics and independently frozen marker/anatomical calibration; retain failed trials without promotion. Bind accepted motion to physical stance/contact, recover all 35 actuator torques including neck and verify fresh independent full-swing replay. Complete #11268 catalog/backend/pairing gates through shared authorities; local acquisition and visible placement review are done, while #11269 additional acceptance remains open. Preserve peer #11351 work and existing Desktop media. Verify protected CI and built-in LaTeX compilation before delivery.

Root-availability review (2026-10-03 UTC): saved-array analysis reproduces the legacy owner-window root-step maxima and associates the largest approximately 40 mm increment with restoration of seven derived lower-body targets. Waist-marker gaps can propagate through the pelvis-axis joint-centre estimator despite visible raw leg markers. This is association, not causation. The current refined preview uses a different head-axis, back-marker, spine/scapula and gap-weight objective; the legacy-window maximum does not establish its root error. The private translation-option TDD completed naturally with eight RED failures and eight GREEN passes, no incomplete tests and no model/simulation. The aligned refined port is now natively evaluated but remains private and unpromoted. See root_availability_checkpoint_20261003.json and the maintained LaTeX calculations. Full moving references, contact, all 35 actuators, independent replay and PDF qualification remain open.

Original-fit regularity review (2026-10-03 UTC): the separate LaTeX reference and `tangent_c2_checkpoint_20261003.json` contain the equations, source hashes, actual receipts and rejected experiments. Full-clock rate fitting has an independent 2,042-pose audit; adding low/moderate acceleration regularization produces another 2,042 poses. The acceleration fitter saves its results but hangs at shutdown (watchdog 125); a fresh independent native process rechecks all new poses and exits naturally with process/wrapper 0 at 17:47:42 UTC. Stronger smoothing reduces the owner's maximum wrist step from 13.24 to 9.53 degrees, but increases root motion from 6.54 to 9.39 mm and worsens marker error. The tour also trades better shoulder continuity for worse wrist motion. No new best video is promoted. These sampled position checks and finite-window derivative proxies do not qualify continuous motion, contact or independent dynamics.

The current 14-position-target objective includes the clubhead, not the grip centroid. Raw club triads provide an orientation observation with address fixture alignment, while changing relative head/grip orientation prevents an unsupported rigid observation assumption. Ten native input-contract tests verify scalar validation and mask isolation without model loading. Completed club-orientation window evaluation tests baseline, zero, and two positive weights across 536 window poses (four trials each, 75 tour poses and 59 owner poses per trial), establishing exact whole-output parity at zero weight. The original window fitter saves results and hangs at shutdown (watchdog 125); a fresh independent native audit rechecks all 536 new poses and exits naturally with process/wrapper 0 at 18:18:38 UTC. Mean club orientation error improves from 10.30 to 5.92 degrees for Tour (capture_A) and 19.20 to 10.73 degrees for Owner (capture_O), while position target error shifts from 27.242 to 27.248 mm (Tour) and 37.989 to 38.014 mm (Owner), with boundary transition tradeoffs from sparse dt 1/30 s into dense steps.

A subsequent full source-clock club comparison evaluates 3,063 trial poses across both captures (three trials each, with 654 tour poses at 360 Hz and 367 owner poses at 240 Hz per trial), producing 2,042 new poses from positive weights (0.025 and 0.075) and retaining 1,021 rate-control poses without refitting. Observed club triads are available at 618 tour and 345 owner samples; missing frames remain excluded rather than counted as agreement. The original fitting process saved all poses before hanging at shutdown (receipt 125 at 19:13:08 UTC); a separate fresh native R2025b audit verified all 3,063 saved poses, club rotation matrices, observation masks, metrics, and source hashes with maximum physical joint residual <= 9.93e-16, exiting naturally with process/wrapper 0 at 19:15:18 UTC. For private video previews at 30 fps (55 tour display frames, 46 owner display frames), owner weight 0.025 and tour weight 0.075 are selected as a documented tradeoff: tour mean club error improves from 11.859 to 5.476 degrees with derived 14-target position RMS shifting from 16.887 to 16.944 mm and near-unchanged wrist and root steps, while owner weight 0.025 improves mean club error from 21.143 to 19.287 degrees, reduces derived position RMS from 20.800 to 20.762 mm, and reduces maximum wrist step from 13.240 to 13.060 degrees as well as shoulder geodesic and root displacement. The stronger owner weight 0.075 worsens maximum wrist step to 17.536 degrees (shoulder step to 12.534 degrees, root displacement to 7.006 mm) and is not selected for default preview.

Eight Human ellipsoid H.264 MP4 previews are delivered on both Desktops in Simscape_Matches_20261003_Club_Refined_1080p, with a verified ZIP. Native render-to-pose parity, independent full decoding of 404 frames, 1080p/30fps/PTS checks, hashes and sampled visual review pass; the render process exits naturally with code 0 at 19:32:57 UTC. These previews remain inverse-kinematic (IK) visualizations, not forward dynamics (FD). Native actuator discovery on the nominal 80 kg model identifies 25 joint blocks (19 actuated spanning 35 active actuator axes, 6 passive/unactuated blocks) and 99 logged leaf series in an independent 0.02 s pilot with verified neck torque (+/-0.25 N*m) exiting naturally with code 0 at 18:48:51 UTC, but leaves expose state and acceleration quantities rather than measured actuator torque. Native peer and input-graph audits across 267 connection ports and 27 driving converters document native input parameter units in aggregate: 21 scalar converters with Unit 1 (7 revolute axes, 14 universal axes across 7 body joints), 2 neck converters with N*m (adapted torque drive), and 4 spherical vector converters with N\*m (FollowerFrame 3D torque vectors across 4 spherical joints). Input Unit 1 applies no scaling and does not by itself imply a torque-unit error; physical output units are inferred from the destination. This is a parameter inventory, not physical torque measurement. Moving swing, contact, and feedback-off replay across all 35 axes remain outstanding without speculative all-35 measurement or anatomical truth.

The controller XYZ chart remains a separate concern: the owner's left-shoulder peak condition number rises from 21.2 to 54.2 after rate smoothing. Native joint inventory identifies 33 input-torque components plus two prescribed neck axes; only the bounded two-neck pilot has measured primitive-actuator evidence. The complete 35-axis moving/contact/open-loop goal remains open. Six owner session videos are locally downloaded and decoded; exact capture-system trial pairing and anatomical calibration remain unresolved. Existing Human ellipsoid Desktop previews remain kinematic fits. Canonical manual governance still reports two QMD sources, zero registered calculations and an owned inventory blocker. The existing LaTeX editor continues to report its standard-directories platform error; no compiled-PDF claim is made.

Closed-arm regularity and foot review (2026-10-03 UTC): six additional assembled support poses retain all 101 original anchors and 11 prior projections. Native dense acceptance improves to A 1,295/1,297 and O 721/721; remaining two tour failures are midpoints. The returned velocities nevertheless reach right-wrist peaks of 113,025 and 169,742 degrees/s, and the owner has a 99.72-degree wrapped scalar step and 115.22-degree shoulder SO(3) step across consecutive accepted samples. Both curves remain rejected for moving-reference/dynamics promotion; pointwise compatibility is insufficient. A fresh native styled-foot audit shows forefoot-minus-rearfoot direction aligned with sole forward axis (dot above 0.9937), with small address yaw errors. Later right-foot horizontal reversals occur when projection lengths are small; their 3D marker-to-visual angles remain about 24 degrees for tour and 11 degrees for owner despite near-180-degree projected yaw. Preserve the documented ankle-marker-height/flat-sole calibration assumptions; no blind shoe flip is justified. Initial World-Frame uniqueness probe exit-one is retained alongside corrected natural-zero audits. See tangent_c2_checkpoint_20261003.json and the maintained LaTeX. Alternate coordinate-target allocation is a private unqualified hypothesis; no physical model/actuator authority change is claimed.

Denser C2/fit review (2026-10-03 UTC): evaluating the SAME candidate curves at 2,018 source and midpoint samples reproduces all 1,010 source samples exactly, but native acceptance is A 1,294/1,297 and O 718/721: three new midpoint assembly failures per capture. Original source samples and all 101 anchors still pass; full-path acceptance is rejected. The new failure diagnostic finds six nearby assembled states with unchanged root translation and all 11 controls passing. A separate native dense fit audit exits naturally zero with original anchor point/RMS parity: mean/peak derived target RMS is 16.98/38.02 mm for A and 21.06/59.60 mm for O; mean/peak directly measured three-back-marker RMS is 26.77/54.66 mm and 38.41/88.12 mm respectively. Valid derived target coverage is 7..14 per frame, with gaps excluded. These mean per-frame RMS scores use different sampling from sparse previews and are not direct model/golfer rankings. No curve or video is promoted to full-path or forward-dynamics acceptance. See tangent_c2_checkpoint_20261003.json and the maintained LaTeX.

Tangent/C2 review (2026-10-03 UTC): fresh native checking passes all 48 saved q/v states; original timeout 124 and tour shutdown 125 remain recorded. Private C2 TDD passes 19 tests. Closed-arm guess correction eliminates all 11 owner anchor mismatches while preserving independent targets bitwise; 6 tour and 5 owner assembly failures remain. Nearby assembled states pass fresh position gates, but a same-target closest-seed retry recovers none of the 11 failures (six controls pass). A controlled projected C2 curve adds eleven upper-limb support poses, preserves original 55/46 fitted anchors, root motion, lower body and unaffected curves, and naturally exits zero with ALL 649 tour and 361 owner source-rate q/v/fresh-state checks accepted. Maximum scalar/spherical target deformation is 0.112/0.173 degrees for A and 0.338/0.343 degrees for O; dense raw-marker RMS is not evaluated. Three axis-contract tests pass after real RED/GREEN correction. The candidate remains unpromoted pending between-state closure, acceleration, full raw-clock coverage, all 35 input torques including neck, contact and independent full-swing forward replay. Parent verifies 104 Desktop video hashes and 5,252 decoded 1080p frames; comparison topologies differ and Human ellipsoids remain the new matching standard. See tangent_c2_checkpoint_20261003.json and the maintained LaTeX. PDF compilation remains unavailable due to missing platform standard directories.

Aligned refined-objective review (2026-10-03 UTC): actual sparse refined previews have zero consecutive source-frame transitions; largest root increments are 17.827 mm (A) and 30.166 mm (O) over 1/30 s. Address-prefix controlled eight-pose windows preserve the spine prior. Private parameter TDD has eight RED failures/eight GREEN passes. The original 344-pose fit wrapper returned 125 after completion; a fresh independent saved-pose recheck exited naturally with zero and verified all 344 poses plus exact whole-output zero parity. Small root improvements accompany worse wrist increments; no candidate is promoted. Native counts distinguish 48 position variables, 43 velocity variables, 37 floating IK parameters and 35 requested control axes. See aligned_refined_checkpoint_20261003.json and the maintained LaTeX. Full continuous references, contact, all 35 input torques including the neck, independent replay and PDF qualification remain open.

### DL-#11161 — Capture Registry, Capture Export and Swing Comparison (Owner Swing vs Tour Average)

- **Next step:** Coordinate the separately tracked engine deliveries in #11165–#11169 and cross-engine comparison #11170; keep this parent active until its scoped acceptance is met.
- **PR:** not recorded in source
- **State:** in_review
- **Owner:** claude
- **Issue:** #11161
- **Branch:** `feat/capture-registry-11161`
- **Paths:** `data/capture_registry.json`, `src/motion_capture/capture_registry.py`, `src/motion_capture/capture_export.py`, `src/shared/python/swing_comparison/`, `src/shared/python/motion_matching/leaderboard.py`, `src/shared/python/motion_matching/pipeline/`
- **Started:** 2026-09-29
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Capture-O parent coordinates a neutral capture registry/export and engine-independent swing comparison. Engine slices #11165 through #11169 and cross-engine comparison #11170 remain separately tracked.

### DL-#11421 · Dynamics-Informed Mocap Matching With ZTCF Prediction and Continuous Forward Replay

- **State:** in_progress
- **Owner:** antigravity
- **Issue:** #11421 (children #11422–#11430, #11434, #11435, #11437 closed; #11436 active)
- **PR:** #11436
- **Branch:** `feat/dime-11436-learned-initializers`
- **Paths:** `src/shared/python/estimation/dime_learned_initializers.py`, `src/shared/python/estimation/__init__.py`, `tests/unit/estimation/test_dime_learned_initializers.py`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Epic #11421: ZTCF drift prediction, kinematics-informed mocap matching, calibrated uncertainty weighting, uninterrupted forward simulation replay without frame-to-frame resetting.
- **Next step:** Create PR, verify CI Standard, squash-merge into origin/main.

### DL-#11400 · OpenCap to OpenSim Integration

- **State:** in_progress
- **Owner:** antigravity
- **Issue:** #11400; child #11408 (OpenCap: golf accuracy qualification against a marker reference; follows #11401–#11403, #11406, #11407, #11409, #11405)
- **PR:** #11451
- **Branch:** `docs/opencap-golf-accuracy-deferred-11408`
- **Paths:** `docs/development/planning/catalog.json`, `docs/development/planning/DV-11408.md`, `docs/development/planning/source-11408.json`
- **Started:** 2026-10-03
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** OpenCap to OpenSim integration complete across sidecar launcher, session import, marker augmentation, UI, and hosted download; golf accuracy qualification against physical marker reference recorded in deferred-validation catalog (DV-11408).
- **Next step:** Merge PR, publish DV-11408 on main, close issue #11408, and close epic #11400.

### DL-#11329 · Scapula and Quiet Torso Matching With Neutral 1080P Previews

- **State:** in_review
- **Owner:** codex (Gemini 3.8 Flash CLI reviews)
- **Issue:** #11329
- **PR:** #11351; depends on draft #11256 and shared-capture #11172.
- **Branch:** `feat/simscape-scapula-protraction-20261002`
- **Paths:** `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/tools/gs3dx_whole_body_ik.m`, `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/tools/gs3dx_spine_bounds.m`, `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/tools/gs3dx_render.m`, `docs/research/simscape_matching_reference/simscape_matching_reference.tex`
- **Started:** 2026-10-02
- **Last verified:** 2026-10-03: 95 native R2025b contracts/regression tests pass with no failures/skips and natural exit 0. Final independent native replay checks all 1,313 sampled poses across 26 complete trajectories; final replay/export batches exit naturally with 0. All 104 neutral 1920x1080/30-fps videos across 13 moving-scapula models are delivered on Desktop, with both views and marker/clean modes. All 5,252 frames decoded; Desktop hashes and ZIP CRC/member hashes verified. Fixed-foot FullBody public-source fits reduce mean target RMS from 532/377 to 27/34 mm while preserving original offsets/model bytes; back RMS 52/61 mm remains a limitation. Final complete Human legacy/default parity passes. Source head bbe21592b6 preserves the qualified native tree after accepted-main sync; current documentation head requires CI verification. Built-in LaTeX compilation remains unavailable (platform directories missing).
- **Summary:** Scapula address/backswing bands, direct measured back-marker fitting, native spine excursion bands through top with soft continuation afterwards, two-coordinate-neck head-axis guide, complete measured-marker overlays, cyan back/waist highlights, neutral Model Swing labels and undistorted native 1080P pixels. Full 3D head yaw remains imperfect and is separately reported. Fixed dimensions/position offsets and private source identities are preserved. Independent dynamics remains outside this scope and unqualified.
- **Next step:** Verify current documentation-head CI and await scientific acceptance of parent drafts #11256/#11172 before protected source integration. Desktop matching/media deliverables are complete. Do not merge unqualified parent work through dependent draft #11351; no protected-main integration is claimed.

### DL-#11313 · Necromatcher Video Export Force/Torque Layer

- **State:** in_review
- **Owner:** claude (Sonnet 5 CLI delegate)
- **Issue:** #11313; parent #11285
- **PR:** draft from `claude/issue-11313` (number on the PR; `Fixes #11313`)
- **Branch:** `claude/issue-11313`
- **Paths:** `src/shared/python/workspace/necromatcher_video_forces.py`, `src/shared/python/workspace/necromatcher_video.py`, `src/shared/python/workspace/necromatcher_video_worker.py`, `src/shared/python/workspace/necromatcher_video_jobs.py`, `src/api/routes/necromatcher.py`, `src/tools/necromatcher/video_dialog.py`, `ui/src/components/necromatcher/VideoExportControls.tsx`, `tests/unit/workspace/test_necromatcher_video_forces.py`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Off-by-default `force_layer {enabled, kinds, scale, segment_shading}` drawn through the shared FTO-8 renderer with the fit's camera; analytic v/a from the preserved spline (kinematic-only fits report the layer unavailable); legend states research-fit, not measured forces; manifest and per-frame receipts are deterministic.
- **Next step:** Frontier review of the draft PR, then rebase against paused draft #11359 (also edits `necromatcher_video.py`) before merge.

### DL-#1900 · Merge-Queue Scoping for Core Test Lanes

- **State:** in_review
- **Owner:** claude
- **Issue:** D-sorganization/Repository_Management#1900
- **Branch:** `claude/1900-ud-merge-group-tests`
- **PR:** #11510
- **Paths:** `.github/workflows/ci-standard.yml`, `tests/ci/test_merge_group_test_lanes.py`, `tests/ci/test_ci_infrastructure.py`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** `merge_group` runs diff the core lane against `merge_group.base_sha` and fetch the default branch for the Tools child-copy guard, matching pull-request behaviour.
- **Next step:** Merge through the queue and confirm the next queued UpstreamDrift PR passes `tests` and `unit-test-gate`.

### DL-#11278 · COV-10 Model-Candidate Video Comparison

- **State:** in_progress
- **Owner:** claude
- **Issue:** #11278; parent #11268; blocked (real data) on #11165
- **PR:** not created (Refs #11278)
- **Branch:** `claude/issue-11278`
- **Paths:** `src/motion_capture/reference/model_candidate_source.py`, `tests/unit/motion_capture/test_cov10_model_candidate_source.py`, `docs/development/capture-o-video/procedure.md`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Python-only engine-generic path projecting any model candidate into video views and comparing it through the COV-7 L2 receipt; real Simscape evidence needs a stored `capture-O` candidate from #11165 and an R2025b host.
- **Next step:** After #11165 stores a candidate, run `compare_candidate_2d` on an R2025b host for every paired swing and record aggregate results on #11278.

### DL-#11268 · Capture-O Video Companion

- **State:** in_progress
- **Owner:** local
- **Issue:** #11268; children #11269–#11279; parent program #11161
- **PR:** #11502 (Closes #11271, Refs #11268)
- **Branch:** `feat/cov-3-protocol-decision-11271`
- **Paths:** `docs/development/capture-o-video/comparison-protocol.md`, `src/config/cov_comparison_profile.v1.json`, `src/config/cov_landmark_correspondence.json`, `tests/unit/motion_capture/test_cov3_protocol_and_correspondence.py`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Ratify comparison protocol L0-L3, frozen comparison profile, and landmark correspondence table.
- **Next step:** Merge PR for COV-3, complete Epic #11268.

- **PR:** #11499 (Closes #11272, #11274, #11275, #11276, #11277, #11279, Refs #11268)
- **Branch:** `feat/cov-benchmark-suite-11268`
- **Paths:** `src/motion_capture/reference/**`, `docs/development/capture-o-video/**`, `tests/unit/motion_capture/test_cov*`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-11 at e68baeabde8f
- **Summary:** Complete implementation and verification of Capture-O Video benchmark suite spanning levels L0–L3, virtual camera projection, pairing confidence, marker-anchored subject scaling, 2D/3D residuals, and error budget reporting.
- **Next step:** Merge consolidated PR, close superseded individual draft PRs, and close Epic #11268.

- **PR:** #11416 (Closes #11270, Refs #11268)
- **Branch:** `feat/cov-2-register-sources-11270`
- **Started:** 2026-10-02
- **Last verified:** 2026-10-11 at e68baeabde8f
- **Summary:** Bring the owner's capture-session video album into the markerless, Necromatcher and engine-matching pipelines, compare against the `capture-O` marker capture at graded levels L0–L3, and publish a single-view reconstruction error budget. [Procedure](capture-o-video/procedure.md).
- **Next step:** Advance to COV-3 comparison protocol specification and level definitions (#11271).

- **PR:** #11413 (Closes #11269, Refs #11268)
- **Branch:** `feat/cov-1-acquisition-11269`
- **Started:** 2026-10-02
- **Last verified:** 2026-10-11 at e68baeabde8f
- **Summary:** Bring the owner's capture-session video album into the markerless, Necromatcher and engine-matching pipelines, compare against the `capture-O` marker capture at graded levels L0–L3, and publish a single-view reconstruction error budget. [Procedure](capture-o-video/procedure.md).
- **Next step:** Advance to COV-2 (#11270) video source registration and swing window grading.

- **PR:** #11417 (Closes #11273, Refs #11268)
- **Branch:** `feat/cov-5-runner-matrix-11273`
- **Started:** 2026-10-02
- **Last verified:** 2026-10-11 at e68baeabde8f
- **Summary:** Bring the owner's capture-session video album into the markerless, Necromatcher and engine-matching pipelines, compare against the `capture-O` marker capture at graded levels L0–L3, and publish a single-view reconstruction error budget. [Procedure](capture-o-video/procedure.md).
- **Next step:** Advance to COV-6 validation harness and error budget report (#11274).

### DL-#11235 · Necromatcher Native Fit

- **Local Research Golf Checkpoint — #11493:** API, native and React now provide explicit Connect → Prepare Research → Arm → Submit → Recall through the actual local session. The authenticated bridge retains exact saved impact vectors and a declared proper Aim rotation; MODEL_CONTACT and contact/numerical/scientific qualification remain unverified. React passed 106 cases; one actual temporary-Library MuJoCo/clean-worker/Rust API and native case passed in 23.07 s with seven exact retained samples; configured mypy passed on seven owners. The final root cohort passed 209 cases with 12 warnings and no skips in 58.64 s; the actual native case is one additional pass. All 13 modified Python owners pass pinned Ruff check/format. The capability atlas was refreshed without claiming overall gap closure. Core V1 is superseded only by the root delivery_status unknown-ID KeyError→None contract fix. The three-page local report and procedure document the new local flight, saved assumptions and explicit lifecycle. No fit improvement, avatar/course capability, GL accuracy, ControlTower delivery or new remote CI claim is established. Tiger remains [0,191), Hogan [0,750); the full goal remains ACTIVE. [Procedure](necromatcher-local-research-golf.md); [Three-Page Report Source](necromatcher-local-research-golf-methods.tex).

- **Authored Replay Impact Checkpoint:** SELF progresses #11464 with authenticated explicit head/face/sample extraction and full-vector declared-flight handoff. Earlier root 83 integrated app/impact cases pass. Portable full-state receipt export/recall now binds exact canonical trajectory bytes without changing the viewer wire; six integration cases pass, including tamper rejection. The supplemental five-page LaTeX methods report is compiled and visually reviewed on the local Desktop. [Procedure](necromatcher-replay-impact.md). Normal app impact execution remains open; authored seconds are not calibrated historical time and no historical replay or qualification is claimed.

- **App Boundary Checkpoint:** SELF progresses #11450/#11414 with explicit strict lossless seeds and exact ordered saved-frame review in native/React hosts. Root 45 Python/45 UI cases pass; edited domains survive mode changes, excluded/sparse indices are covered, original source-only browsing remains available. [Procedure](necromatcher-app-restriction-and-frame-review.md). Historical results and Methods V8 remain unchanged; no physical qualification or CI-green claim.

- **State:** in_progress
- **Owner:** codex
- **Issue:** #11235; parent #11232; explicit coordinate seeds #11281; probe/generalization trials #11284; qualification child #11357
- **PR:** [#11240](https://github.com/D-sorganization/UpstreamDrift/pull/11240) (merged); [#11359](https://github.com/D-sorganization/UpstreamDrift/pull/11359) (open)
- **Arm Morphology Documentation Checkpoint:** Four unoptimized arm ±5% admissions and twelve selected overlays are independently provenance-verified; exact twelve-page AN1 report retained. Source 14,800 preserved, library 383→391; grip gaps worsen to 18.6–22.1 mm and anatomy remains unadopted. [Procedure](necromatcher-arm-morphology-admission.md). Documentation SELF; no behavior, evidence-role or release-qualification change.
- **Scoped Camera Checkpoint:** At published producer 55d17, native session 74740 and independent session 92074 closed with exit 0; full 14,805-source-file and 391-library-file maps preserved. Tiger [0,191) body RMS 24.0827→24.7803 px worsens; Hogan 10.1364→10.0910 px has mixed shaft outcomes. Keep saved cameras; 17 root checker tests and all six compiled report pages reviewed. [Procedure](necromatcher-conditional-camera-and-scope.md). #11414 durable scope integration is implemented; root 294 Python/39 UI cases, Ruff, pinned mypy, TypeScript and ESLint pass. HTTP receipt registration runs off the event loop; 13 focused root follow-up cases pass. Restricted trial at 669508 saved an authored seed; strict fitting exceeded unchanged 300-second wall cap, exit 1, no final fit. Original assets/index records preserved; ten allowed files added. Solver evaluations/elapsed unknown. Independent partial-stage verification is accepted (V4 partial checker, actual root release V2). Setup profile V1 failed on an ambiguous ndarray adapter/receipt JSON and is preserved; V2 measured five public stages successfully, with independent serialized acceptance. At published `0143b39f571cbe67c57091fb2ae19e7d7eadae3b`, both independent audits preserved complete 14,826-source-file/401-library-file maps. The same 191-frame confidence-weighted dense body RMS is 35.102046 px for the seed versus 24.082673 px for its parent; retain the parent. Actual selected seed stills 0/150/190 show opacity 0.35 proxy shapes, skeleton and an explicit unoptimized authored research seed caption. Independent serialized source/image verification is accepted; unweighted display-marker RMS is approximately 6.45/44.14/18.27 px, separate from weighted body metrics. Native projection and complete overlay-union pixel parity were not independently recomputed. No timeout cause, qualified match, new full video or ControlTower delivery is established. Tiger matching remains [0,191), exact physical release is unmeasured, and Hogan is unchanged. Methods V7 is the latest compiled supplement: root reviewed all eighteen pages in contact sheets and new pages 17 and 18 full-size; V2/V3/V4/V5/V6 remain preserved. The root V7 acceptance SHA256 is 51a8da1aaa9b590100354d8db1956431450b955d87ba57894b97da0df673c810. The two new pages describe the reusable restriction workflow and synthetic native end-to-end evidence at 5197. Its three pinned external Desktop PNGs remain earlier V21 authored-seed figures and local repeatability dependencies, not a portable standalone bundle or newly rendered restricted fit. No scientific qualification or new ControlTower transfer. Performance child #11445 now implements public authenticated_read: one canonical full PNG/BGR/clock validation per capture per operation, fresh complete file hashes and metadata checks on reuse/close, cross-library/thread/task and nested-context rejection, immutable DTOs and failure reset. Final root validation passed 93 cases across two lanes (86 plus seven public-owner/fingerprint cases; one Windows symlink-privilege skip and inherited import warnings). Canonical public SHA256 validation is reused, and execution fingerprints cover shadow_tracker; this API checkpoint is separate from the accepted artifacts at 0143. This integration checkpoint implements short authentication boundaries for queue admission, worker setup and fresh output checks, candidate/delayed publication and storage. Contexts close before scheduling, computation, delivery or persistence; delayed publication closes before add_fit, which uses its own fresh validation context. Root passed 188 distinct cases (185 SDK-free plus three native temporary-fixture cases; one Windows symlink-privilege skip, three native cases deselected in the SDK-free lane and nine inherited warnings). Published integration f2dfe completed read-only Profile V3: five stages took 0.672506/1.464802/9.053819/1.511551/0.460402 s; only seed binding decoded 210 PNGs, once in one canonical full authentication. Source 14,835/Library 401, project/runtime/config were preserved; assessment is byte-identical to V2. Root acceptance SHA256 b8469dc86d8aa72e1be2499786a107c847dd2e11d32f20567aa1effb1947a8e4 records this invocation; differing instrumentation/context/hardware prevents an isolated speedup or timeout-cause claim. No strict retry or new ControlTower delivery is authorized; accepted 0143 artifacts are unchanged. Pure restriction #11450 has a frozen V2 core (SHA256 b3504e6c64c7f97116a108b97a06d6ecba8732f3d38dd004751789e0bc93ce58) with 23 plus 53 agent cases. Root accepted 91 distinct cases (23 core plus 68 disjoint compatibility/bounds/initialization cases; synthetic MuJoCo-first fixtures only, nine inherited warnings and 38 unmarked cases reported only). Root acceptance SHA256 c0c2c44509650d37dc9516aa4185a54b828a49767bcca4ab8d8b2af64786b2ff records Ruff/format, pinned mypy and governance checks; manual release remains inventory-blocked. Pure q/v/a roundoff and Bernstein encoding evidence 17ddcd74c27276a376174c202500baa10e389d09b97ca27c3c42e9d1e4b778b5 is not native feasibility, pixel agreement, seed adoption or queue integration. Subsequently, native session 50689 closed exit0 at published 81aa448794a0ce763cdc2dd6e503d3f9b42200e9; root serialized closure accepted ce833a420c5e208b82bc430fe4be4c4c16830aee54d9c8c357bdb1af9e775e9c. Full source 14,837/Library 401/project/runtime/config/helpers matched the fresh baseline. Restricted-parent body RMS exactly equals saved parent (dense 24.082673308003265, remaining 24.337344114034266, training 21.776977995696026 px); 381-point q/v/a roundoff and typed receipt equality hold. The full 176 x 12 dimensionless scaled constraint matrix has maximum 17.023155533359613 at ground:heel_l, source 18.501816666666667; continuous contact is uncertified and nodes differ from V21, so no V21 656 comparison applies. Retained frame 0 raw shaft RMS 66.69082241158017 px/angle 28.301292322830996deg remains mismatched. No optimizer/convergence/physical qualification, telemetry null; no adopted seed, new fit, overlay rendering/pixel-parity evaluation or ControlTower delivery. Subsequent backend software checkpoint: root accepted paired restrict_initialization/restricted_spline operations with canonical SDK-free restriction receipts, preserved prior/bounds, parent agreement within 1e-12, scope/hash admission and worker/storage/recall/delayed-publication checks. Legacy semantics and public counts (eight option fields, seven start_native_refit arguments, eight builder arguments) are unchanged. Root passed 87 distinct SDK-free cases with nine inherited warnings; pinned mypy on all five owners and Ruff/format passed. Root acceptance SHA256 dc43749d0e69c6c43051712c5d37b819fbbe950edd3cf5dfb12f426a61b80bed authenticates the software freezes. Backend software was published at 5197f8639818321b7cf8f86d4748a25ce79aa278 with all normal hooks passing. Synthetic native end-to-end session 4534 then closed exit 0: the public queue, clean worker, storage and recall preserved the nonuniform restricted curve on [1,6), with four training/five dense frames, 43 locked coordinates and five rejected queue/storage counterexamples. Root closure SHA256 a26073a9c7bef6c8d5f2dd12b249c8e5f848be49c9fb0ba63d0a5c98f011d9bc authenticates the independently closed SDK-free audit (session 97412): fresh 14,844-source-file/401-historical-file/runtime maps are unchanged. Independent pure q/v/a errors are 3.47e-18/2.78e-17/1.78e-15; saved native pixel metrics agree with the actual run, but were not independently reprojected by the serialized audit. Optimizer counts remain null; no optimization, historical seed registration, new overlay or ControlTower delivery occurred. This synthetic software acceptance does not alter the earlier Methods V6 historical native-evaluation checkpoint or establish physical qualification. Subsequent caption software adds UNOPTIMIZED RESTRICTED RESEARCH SEED through the shared still/video composer and authenticated manifest, with eight strict DTO fields and mutually exclusive authored/restricted flags. Root passed 80 distinct cases in MuJoCo-first temporary fixtures: canonical exports and tampered lineage, manifest claims and exact legacy/authored pixels; configured pinned mypy, Ruff and format pass. Caption root acceptance SHA256 12a5e9e99d0b5d3827fcddc229324bf9e5525e448ae430acb4a3e0881a1f73d0 records this later software checkpoint, separate from the Methods V7 appendix at 5197. No new historical seed or overlay is claimed. Next: publish through normal hooks, then separately reviewed historical admission and the unchanged strict restart. The full goal remains active. Subsequent V22 checkpoint: Tiger remains within reviewed source frames 0-190. The accepted lossless seed has a local 191-frame shape/skeleton video; the ordinary strict fit modestly reduces same-domain weighted dense RMS from 24.08 to 23.57 px but remains nonconverged and REJECTED. Independent acceptance SHA256 8bb98838915c914c4dab8b9c9b9fff871ec7b02294a8cfd155d57657852aa72d records preserved source and original non-index Library bytes. Methods V8 is the earlier 21-page seed/stills report, not this later fit/video checkpoint. Three optimized research stills are accepted locally (proof 0d244648c2f76bb2c4be9f232d81215a58c5a1779963f50bcb575f76ff402d13); ControlTower delivery remains pending. Hogan is unchanged; physical/anatomical/contact qualification is false and the goal remains active. Detailed evidence is retained in the camera-and-scope procedure.
- **Branch:** `feat/necromatcher-native-fit-11235`
- **Paths:** `src/shared/python/motion_matching/historical_fit/**`, `src/shared/python/workspace/necromatcher_*`, `src/engines/physics_engines/mujoco/python/**`, `src/shared/python/motion_matching/historical_fit/shaft_*`, `src/shared/python/workspace/necromatcher_shaft_evidence.py`
- **Started:** 2026-10-01
- **Last verified:** 2026-10-03 — Actual producer `3ab5c2ac87dd6aa97872b49596fbcad4f9a6eeb5` completed all four matched shaft control/variant jobs. Fresh assessment `b1f3898d3d9e2cac5c460ceb33fd2e6c5b85371158b4eb2a823e4f027ede002e` verifies identical starts/body/geometry recipes and all five fixed finite targets; every fit remains budget-exhausted/nonconverged/rejected. Raw training improves, Tiger holdout worsens and Hogan holdout is essentially unchanged. Actual V2 admission and display-only exports preserve source/clock/roles without changing objectives; see the shaft trial procedure. Earlier verification retained: 2026-10-02 — producer6654881b0788c77620a0afb8a9a6763f3ee1ccb9 completed two V15 exact-six-knot normalized-coverage continuations under #11284. Independent native audits verify identical start/evidence/prior,32 finite authored ranges with12 unbounded coordinates,210 source frames,209 midpoints,88 scalar extrema and10 phase probes. Control78nodes/dense25.048392px/grip1.905599mm/angle0.703158degrees/ground3.437775mm fails ground; variant178nodes/dense25.160335px/grip1.955344mm/angle0.715385degrees/ground1.852699mm passes all five fixed finite research targets. Geometry weights4000→1752.808989 preserve potential family rowmass, not physical-time quadrature or continuous nonlinear qualification. Both exhaust30evaluations, remain nonconverged/rejected and have unknown physical time. Exact assessment and repeatability inputs are preserved in turnover; preceding V12–V14 evidence remains unchanged. Actual V15 overlay and selected-still review are published, with rejected qualification retained; final expanded source regression passed 476 cases.
- **Slice A verification:** Source-bound club evidence Slice A (#11318) is committed at `605c695259b13cb98a6d6e379e7f7fee51f27443`: immutable reviewed shaft-axis segment/source-frame DTOs, distinct original decoded-pixel and PNG-byte hashes, exact source clock/capture binding, explicit abstention and authored-uncalibrated weighting, plus an exact-definition-bound infinite authored axis resolved from declared coaxial shaft/grip solid centers. Public receipt counts/digests and strict real-coordinate guards reject malformed records. Initial tests were red; the final focused suite passed 57 cases, with Ruff/format, pinned mypy and budget checks. Actual source-bound admission subsequently preserved two Tiger fragments, one faint Hogan fragment and a blurred-frame abstention; receipt `bb36ef2233e39c829b341046d2555fcbc245769dccbaa9489933f09ddb5fadd7`. Root independently inspected all four plots; separate review `67f8ace21806ea5c62e0f1da5635c3e3e19baf614e9989c9fa935ee48201d7ac`. The first TEMP driver failed before output creation because Ruff removed imports used only through locals; preserved v1 and traceback remain distinct from v2, whose explicit registry repair passed 10 protocol tests before successful execution. No optimization, library mutation or physical endpoint/shaft-length qualification occurred.
- **Summary:** Source-bound native fits retain canonical Hermite coefficients and optional analytic interior constraints. Opt-in named hard bounds use a canonical Bernstein-admissible domain with explicit logged authored initialization and immutable native definition/range identities; v7 candidates remain rejected and require new feasible trials plus grip/contact qualification. [Turnover](necromatcher-turnover.md).
- **Slice B verification:** Committed8989 generic additional-image residuals preserve ImageFitInputs and independent body/geometry timelines. Separate sparse infinite-axis residuals and raw perpendicular RMS/unoriented angles passed112 combined and33 independent cases. No historical shaft fit occurred at that checkpoint.
- **Slice C verification:** Committed8989 canonical capture/camera/PNG/full-clock/native-axis rebind returns one reusable native binding. Tests passed96 combined and28 independent cases; compilation-count compatibility is retained.
- **Slice D verification:** Current reviewed queue/worker/persistence source passed102 combined and19 independent cases; strict receipt parsing rejects bool/float counts before native/source I/O. Native evidence controls passed32 offscreen tests independently; web30 plus type checking/ESLint passed. Stable original-video/job/API integration passed41; renderer20/80 combined and20 independent passed. Ruff/format and pinned mypy passed. [Shaft Trial Procedures](necromatcher-shaft-trials.md) records exact freezes and the independently observed unchanged five-second test deadline overrun, followed by successful focused rerun; no speculative cause is asserted.
- **Next step:** Integrate the source-bound continuation in a separate follow-up PR; retain all historical receipts and failed shaft generalization. Head observations/errors and physical endpoints remain unmeasured; camera/anatomy/physical timing, continuous dynamics and impact acceptance remain open. Use the corrected preparation-only R,t geometry protocol before forearm/camera/morphology trials; no new native diagnostic has run. See necromatcher-camera-morphology-shutter-review.md.

### DL-#11317 · Sanitized Historical Research Provider

- **State:** in_progress
- **Owner:** codex
- **PR:** [#11240](https://github.com/D-sorganization/UpstreamDrift/pull/11240) (merged); Affine #4834 merged externally at 2026-10-03T02:38:02Z; actual consumer receipt remains 8fdebad
- **Issue:** #11317; consumer AffineDrift #4828; parent #11232 / #11235
- **Branch:** feat/necromatcher-native-fit-11235
- **Paths:** src/shared/python/workspace/historical_research.py, docs/api/contracts/historical-player-research-v1.schema.json, tests/unit/workspace/test_historical_research_provider.py, docs/development/necromatcher-research-package.md
- **Started:** 2026-10-02
- **Last verified:** 2026-10-02 — Latest Tiger V15/Hogan V12 sanitized cohort exported from committed/pushed/remote-verified provider `605c695259b13cb98a6d6e379e7f7fee51f27443`; manifest `d68830b14e2704466d4dd1795342dffd96f6f4d9b238b1fccd7f27117a9c10ed` (6,852 bytes), export receipt `65c4e775a501c0719e06a40aefd6b8be73a59a031298b96cf1e8a453a605debd`. Committed Affine consumer `8fdebad14db0f62b719942fa5d64f14878b54fd5` passed no-write inspect, exclusive draft installation, exact/idempotent recall, deterministic QMD and standalone Quarto --no-execute; admission receipt `184907af70181af6bd7e94cd5526139fdec828e57fe540c69f87c5c7a80b7d2b`, wrapper `d94e28062f090f00b45e09bb550d99903926129c9b29d70b7d702ec46057aba6`. All source/input/helper/config/registry brackets remained unchanged. Root separately browser-reviewed three natural 1234-by-712 views; browser receipt `046cb332516253fe0592be36a20ceb148cc7496ec1e299a7dbd1041d9e0f10d6` records bounded review with full-page/manual acceptance false. V15 adapter plus selection regressions:64 passed, scoped Ruff/format/pinned mypy passed; unchanged final consumer suite:6,685 passed/26 skipped/187 existing-marker deselected. The earlier V14/V12 bundle and its exact receipts remain preserved in procedure/turnover. Both cohorts remain rejected with unknown timing and no original-media import, optimizer run, public deployment or scientific promotion.
- **Summary:** Reuse canonical immutable library recall to cross-check external independent assessments, exact numerical producer and source/model/capture/fit hashes; rehash original downloaded footage without copying media. Export deterministic generic-player JSON under a separate rejected local research contract, preserving unknown physical timing and unauthorized distribution. No estimator, scheduler or qualified publication pathway is duplicated.
- **Next step:** Preserve the new committed local research handoff and 18 checked repeatability-input copies without altering historical cohorts. Advance source-bound club evidence under #11318 and qualify camera/anatomy/physical timing and continuous behavior before scientific/public release; finite target passes and local browser review grant no qualification or publication authority.

### DL-#11246 · Necromatcher Source Video Overlays

- **State:** in_review
- **Owner:** codex
- **Issue:** #11246; parent #11232; native stored-run recall #11282; translucent model-surface child #11356
- **PR:** [#11240](https://github.com/D-sorganization/UpstreamDrift/pull/11240) (merged); [#11359](https://github.com/D-sorganization/UpstreamDrift/pull/11359) (open)
- **Branch:** `feat/necromatcher-native-fit-11235`
- **Paths:** `src/shared/python/workspace/necromatcher_video*`, `src/tools/necromatcher/**`, `ui/src/components/necromatcher/**`
- **Started:** 2026-10-01
- **Last verified:** 2026-10-03 — Producer `3ab5c2ac87dd6aa97872b49596fbcad4f9a6eeb5` exported all four original-footage shaft displays; independent receipt `f60d277ce81e7d1d3dc2ddace626a1bacd9bd3a83708375ffefe1e7dcdd39dd0` verifies 1,920 source-sized frames, original PNG/PTS bindings, projections/metrics, immutable source/library brackets and twelve selected overlays/six originals. V1/V2 union is display-only; all-frame visual and manual acceptance remain false. Exact manifest PTS are distinct from Tiger encoder rounding (maximum 6.9736403 microseconds relative drift); Hogan 30/1 is exact. Earlier verification retained: 2026-10-02 — V15 selected Tiger variant export run `97b1b96fea0c4a2da403004a4985f390` at producer `6654881b0788c77620a0afb8a9a6763f3ee1ccb9` verifies all 210 original 1280x720 frames, original PNG/PTS bindings, ZIP CRC/member hashes and immutable source/runtime/audit brackets. Actual stills 0/105/209 were separately viewed; club-seed disagreement, follow-through upper-body residuals and cropped feet remain. Combined still-review SHA-256 `ca46a15129d238c06391f6ee1b05c0cfa6bc14cd6be95075d738841798d4b369`; the execution journal retains visual_review_completed=false. All-frame visual and manual acceptance remain false. Earlier V13/V12/native stored-run review receipts remain preserved; canonical sharing-race repair and final expanded source regression passed 476 cases.
- **Summary:** Shared owned jobs export source-clock native joint-tree wireframes, PNGs and verified ZIP packages through native and web controls. Export success remains separate from scientific acceptance. [Turnover](necromatcher-turnover.md).
- **Next step:** Retain guarded recall and distinct machine/selected-image/scientific outcomes. Infinite authored axes and interior fragments do not establish measured head/physical shaft endpoints or validated historical club motion. Complete independent surface-renderer review and clean published producer before actual translucent exports; disabled rendering preserves legacy behavior and displayed surfaces remain model proxies.

### DL-#11247 · Necromatcher Reproducible Methods Report

- **State:** in_progress
- **Owner:** codex
- **Issue:** #11247; parent #11232; source audit #11357
- **PR:** [#11240](https://github.com/D-sorganization/UpstreamDrift/pull/11240) (merged); [#11359](https://github.com/D-sorganization/UpstreamDrift/pull/11359) (open)
- **Branch:** `feat/necromatcher-native-fit-11235`
- **Paths:** `docs/development/necromatcher-methods.tex`, `docs/development/historical_capture/*summary.json`, `docs/development/necromatcher-turnover.md`
- **Started:** 2026-10-01
- **Last verified:** 2026-10-03 at `b8b280ce2f5ed706d1778fd809250cf18eac1fb7` — Published 74-page actual matched-shaft report: PDF `36319afeb6ec6132f7684387d04c1b8cde8d2854a38cf718354325334e4fdffd`, source `41c61a755f72a3b8928c5938f545e0a8a32aa19e4f84437ffc5ac07c92d7b6e9`, independent review `8b4fb44f2b0cd552c7a443f5e90ada9d95f624ff3711d4514f3f70e6708db6ee`, root review `6fe30a6978fcc61fbb3192df6641b3da93515155e21bec6cc3852ef832854c5c`. Three installer-disabled passes completed without actual warnings/overfull/unresolved references; all page layouts reviewed. Desktop manifest 1,504 artifacts `49868b903a729a147d188b03a0fc5ecf3f68e4973345f08b0191d79de2b85228`; exact prior 66 publication and1,260 index/README preserved under Report/handoff-shaft-66-page-publication. Earlier verification retained: 2026-10-02 — Expanded 66-page latest-handoff/sparse-shaft report compiled in three installer-disabled MiKTeX passes and was independently reviewed on all pages, with full-size contents/new chapter/figures/source-map checks. No overfull boxes or unresolved references; 17 contained table-wrapping warnings were visually reviewed. Published PDF SHA-256 `f31f1fbd6c0917254b358bae3852e42de8e5d898b1f60d47b3b08c63427f3228`; exact source `1702e35d4a04b95d0c5de13b04db34325cfef7b8e2ad47f50ea47da45354991c`; independent review `0c7a85f91018a01d117e0d7b0e41dbec61087f6c52019b780ade7bbca4e27481`; separate root review `9de379e2b3a32089bff17fd1086eec9430f0f953240717bfa6093bf02e57446d`. Refreshed Desktop index verifies 1,216 retained artifacts, SHA-256 `bd89c17e27de7b7849986f6e00d36debcd0e3c988ab8530b687a0f92ff1a6e9d`. Exact prior 60-page publication, 49-page edition and historical snapshots remain preserved. No scientific acceptance or physical motion claim; built-in compiler infrastructure criterion remains open.
- **Summary:** Polished methods report records equations, provenance, assumptions, actual interim failures and reproducibility procedures. Built-in compiler infrastructure remains unavailable, so that issue criterion remains open. [Turnover](necromatcher-turnover.md).
- **Next step:** Preserve publication and evidence snapshots; document subsequent work only after independent execution. Built-in compiler infrastructure and scientific/manual release criteria remain open. Nine-page source-audit supplement compiled and all pages reviewed; 77 source-audit files are verified on ControlTower separately from the unchanged 74-page/1,504-artifact publication.

### DL-#11230 — Historical Player Capture

- **Next step:** Complete dense review; camera/time calibration, dense 3D fitting, native parity, dynamics, held-out evaluation, and website integration remain open. Preserve observations as observations until their gates pass.
- **Started:** not recorded in source
- **State:** in_progress
- **Owner:** codex
- **Issue:** #11230; epics #11226 and #11229
- **Branch:** `feat/historical-player-capture-11226`
- **PR:** [#11231](https://github.com/D-sorganization/UpstreamDrift/pull/11231)
- **Paths:** `src/shared/python/shadow_tracker/historical_capture.py`, `scripts/historical_capture.py`, `docs/development/historical-capture-procedure.md`
- **Last verified:** 2026-10-01 at ab2c869813a2b2be2f14b512ee34647694614377 — 364 Shadow Tracker tests passed; final Hogan 750/739 detected and Tiger 2000/1994 detected; observations only.
- **Summary:** Streaming source-bound detector observations; final reproducibility runs completed; dense review pending. Camera/time calibration, dense 3D fitting, native parity, dynamics, held-out evaluation and website integration remain open.

### DL-#11095 — Qualify OpenSim Native Dual-Club Dynamics and Replay

- **Next step:** Native qualification remains unachieved. Run only on the corresponding pinned native lane with complete source-bound evidence; this metadata review did not run tests or native code.
- **Started:** not recorded in source
- **PR:** not recorded in source
- **Owner:** not recorded in source
- **State:** in_review
- **Issue:** #11095
- **Branch:** `feat/mmr-10o-opensim-dual-club-11095`
- **Paths:** `src/engines/physics_engines/opensim/python/native_qualification.py`, `tests/unit/engines/opensim/test_opensim_dual_club_qualification.py`, `docs/development/matched_swing_program/evidence/opensim/`
- **Last verified:** 2026-09-29 at d5dd606ffa6dafc11408614676a2576764a5e092 — 17 unit tests passed; red evidence recorded for the 7 new fail-closed tests against the pre-fix placeholder path; ruff check and format clean on changed files. Native OpenSim qualification NOT achieved: opensim bindings unavailable on all reachable hosts.
- **Summary:** OpenSim receipts now fail closed on missing runtime/evidence; fabricated marker metrics, unenforced tolerances and placeholder hashes were removed. Native qualification remains blocked without the pinned bindings.

### DL-#11096 — Qualify MyoSuite Native Dual-Club Dynamics and Replay

- **Next step:** Native qualification remains unachieved. Run only on the corresponding pinned native lane with complete source-bound evidence; this metadata review did not run tests or native code.
- **Started:** not recorded in source
- **PR:** not recorded in source
- **Owner:** not recorded in source
- **State:** in_review
- **Issue:** #11096
- **Branch:** `feat/mmr-10m-myosuite-dual-club-11096`
- **Paths:** `src/engines/physics_engines/myosuite/python/native_qualification.py`, `tests/unit/engines/myosuite/test_myosuite_dual_club_qualification.py`, `docs/development/matched_swing_program/evidence/myosuite/`
- **Last verified:** 2026-09-29 at 613b0e28ed88b5af4719ecccfe876d5d9db90419 — 19 unit tests passed; red evidence recorded for the 7 new fail-closed tests against the pre-fix placeholder path; ruff check and format clean on changed files. Native MyoSuite qualification NOT achieved: myosuite/MuJoCo unavailable on all reachable hosts.
- **Summary:** MyoSuite receipts fail closed on unavailable runtime and missing/invalid rollout, marker, derivative or quaternion evidence; fabricated metrics and unenforced tolerances were removed. Native qualification remains unavailable.

### DL-#11094 — Qualify Drake Native Dual-Club Dynamics and Replay

- **Started:** not recorded in source
- **PR:** not recorded in source
- **State:** in_review
- **Owner:** UDFixTrio10x
- **Issue:** #11094
- **Branch:** `feat/mmr-10d-drake-dual-club-11094`
- **Paths:** `src/engines/physics_engines/drake/python/native_qualification.py`, `tests/unit/engines/drake/test_drake_dual_club_qualification.py`, `scripts/ci/run_native_engine_lane.py`, `scripts/ci/run_native_engine_lane.sh`, `tests/scripts/test_run_native_engine_lane.py`, `docs/development/matched_swing_program/evidence/nightly/drake_receipt.json`, `docs/development/matched_swing_program/evidence/drake/`
- **Last verified:** 2026-09-29 at 3b23bd41502a7b7d028f291564997b6460f14cbc — 36 focused tests passed (test_drake_dual_club_qualification.py 16, test_run_native_engine_lane.py + test_native_lane_freshness.py 20); red evidence recorded for the 7 new fail-closed tests against the pre-fix placeholder path; ruff check and format clean on changed files. Native Drake qualification NOT achieved: pydrake unavailable on all reachable hosts.
- **Summary:** Drake receipts now fail closed on missing runtime/evidence; fabricated marker metrics and placeholder hashes were removed, and Drake entered the nightly lane. Native qualification remains blocked without pydrake.
- **Next step:** Merge drivers follow; native qualification remains blocked on engine availability (recorded in receipts, not silently).

### DL-#11080 · Product Review for the Expert Panel

- **State:** in_review
- **Owner:** codex
- **Issue:** #11080 (owner-requested documentation review tracked by this PR)
- **PR:** #11080
- **Branch:** `docs/product-review-20260928`
- **Paths:** `docs/development/2026-09-28-product-review-board-proposals.md`
- **Started:** 2026-09-28
- **Last verified:** 2026-09-28 at 599cce5d5cb7d6972dfa2ca4e5770ea09cfaa309 — 108 focused tests and safe counterexample probes; Ruff lint/format and documentation hooks passed.
- **Summary:** Twelve prioritized proposals across UI/UX, error handling, performance, scientific validity and shared Tools wiring, with bounded evidence, acceptance criteria and current backlog reconciliation. Product code and qualification status are unchanged.
- **Next step:** Review R01–R12 using the report's RunnerDashboard panel brief.

### DL-#11003 · Restate the #9243 BunkerShot Uncertainty Claims Under the Corrected F0 Model

- **State:** in_review
- **Owner:** claude
- **Issue:** #11003
- **PR:** #11010
- **Branch:** `claude/ud-9243-claims-restated`
- **Paths:** `tests/unit/tools/bunker_shot_gui/test_uncertainty_propagation_9243.py`
- **Started:** 2026-09-26
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Two #9243 claims went stale while the kernel-less gate skipped them. The window-band fixture now aims at the nominal shot's carry (0.690 m) instead of an unreachable 2 m and asserts a non-empty window; band (0.0, 1.141, 2.282). The mass interval's share is 0.676, so the test now says it dominates without swamping; `DOMINANCE_SHARE` stays 0.75.
- **Next step:** Let the armed auto-merge land it, then replay UD#10997 onto main.

### DL-#1755 · Retire the Review-Comment-to-Issue Converter

- **State:** in_review
- **Owner:** claude
- **Issue:** Repository_Management#1755
- **Branch:** `chore/retire-comment-converter-v3`
- **PR:** draft PR from `chore/retire-comment-converter-v3` (supersedes #10978)
- **Paths:** `.github/workflows/Comment-to-Issue-Converter.yml`, `scripts/ci/process_review_comments.py`, `tests/ci/test_process_review_comments.py`, `tests/ci/test_ci_infrastructure.py`, `.github/workflows/Nightly-Doc-Organizer.yml`, `docs/development/ci_workflow_map.md`, `docs/development/`
- **Started:** 2026-09-25
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Remove the retired Convert-Review-Comments-to-Issues workflow and its processor from this repository per the fleet-wide Repository_Management#1755 campaign, and drop the stale references in the Nightly Doc Organizer and the CI workflow map. Re-lands #10978 (itself superseding #10941) fresh off `origin/main` after its branch conflicted.
- **Next step:** CI green on the v3 draft PR, then mark ready and arm auto-merge.

### DL-#10286 · Native Swing ZTCF/ZVCF Fail-Closed Dynamics and DTACK Semantics

- **State:** in_review
- **Owner:** claude (agy executor)
- **Issue:** #10286
- **PR:** draft PR from `agy/ud-10286-fail-closed`
- **Branch:** `agy/ud-10286-fail-closed`
- **Paths:** `src/engines/physics_engines/myosuite/python/_drift_control.py`, `src/engines/physics_engines/pendulum/python/golf_swing_physics_engine.py`, `src/engines/physics_engines/pinocchio/python/dtack/gui/main_window.py`, `src/engines/physics_engines/pinocchio/python/dtack/sim/dynamics.py`, `tests/engines/physics_engines/test_golf_swing_pendulum.py`, `tests/engines/physics_engines/test_myosuite_engine.py`, `tests/unit/engines/pinocchio/dtack/sim/test_dynamics.py`
- **Started:** 2026-09-25
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Remove success-shaped empty/zero array returns from uninitialized ZTCF/ZVCF counterfactual methods (raising StateError and ValueError for invalid dimensions), update dtack compute_zvcf to canonical (v=0, tau=0) semantics, add compute_zero_velocity_controlled for control-preserved dynamics, and update GUI caller and button label to "Zero-velocity (control kept)".
- **Next step:** CI green, review, merge.

### DL-#9619 · Amend ADR-0041 to Record Consumer-Side Fitter Decision

- **PR:** not recorded in source
- **State:** in_review
- **Owner:** local (session `claude-deskcomputer-20260925-ud`)
- **Issue:** #9619 (companion #9630)
- **Branch:** `agy/ud-9619-adr0041`
- **Paths:** `docs/adr/0041-markerless-mocap-consumer-authority.md`, `tests/architecture/test_markerless_mocap_authority.py`, `docs/development/DEVELOPMENT_LOG.md`
- **Started:** 2026-09-25
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Amended ADR-0041 to record that consumer-side self-calibration fitters live in UpstreamDrift (`src/motion_capture/reconstruct/`) while Tools maintains vendor-neutral reference geometry (#9630, #9619).
- **Next step:** CI green, review, merge.

### DL-#9549 · Guard Contact-Interval Provider Contract for Interval Tab Owner Ruling

- **State:** in_review
- **Owner:** claude
- **Issue:** #9549 (epic #9546)
- **PR:** draft PR from `agy/ud-9549-ruling-guard`, #12538
- **Branch:** `agy/ud-9549-ruling-guard`
- **Paths:** `tests/shared_contracts/test_impact_interval_provider.py`, `docs/development/DEVELOPMENT_LOG.md`
- **Started:** 2026-09-25
- **Last verified:** 2026-10-14 (`b8bb030b`; collated from changes/9549-add-versioned-pre-impact-kernel-bundle-a.md)
- **Summary:** Replaced obsolete strict-xfail pin gate in test_impact_interval_provider.py with test_provider_honours_interval_tab_ruling guarding the owner ruling from Tools #4946 / Tools PR #5289 (IMPACT_INTERVAL_TAB_RULING.md): standalone interval tab dropped, ImpactModelType gains no INTERVAL member, and contact-interval solver remains headless (#9549).
- **Next step:** CI green, review, merge.

### DL-#9688 · Bunker P1: Decide and Benchmark a Genuine 3D Sand-Motion Tier

- **State:** in_review
- **Owner:** antigravity
- **Issue:** #9688
- **PR:** draft PR from `agy/ud-9688-capability-register`
- **Branch:** `agy/ud-9688-capability-register`
- **Paths:** `src/bunkershot3d/solvers/capability.py`, `src/bunkershot3d/solvers/exceptions.py`, `src/bunkershot3d/solvers/__init__.py`, `docs/adr/0044-out-of-plane-fidelity-for-bunkershot3d.md`, `tests/bunkershot3d/solvers/test_capability_9688.py`
- **Started:** 2026-09-25
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Added fail-closed sand-motion capability register (SandMotionKind, SandMotionCapability, register mapping, and require\_\* guards) in bunkershot3d.solvers to prevent presenting F0/F1/proxy/tracer outputs as genuine 3-D individual-grain trajectories or spherical ball spin.
- **Next step:** CI green, review, merge.

### DL-#9613 · Rig Soak: Multi-Camera Repeat Record Mode

- **State:** in_review
- **Owner:** antigravity
- **Issue:** #9613
- **PR:** draft PR from `agy/ud-9613-record-repeat`
- **Branch:** `agy/ud-9613-record-repeat`
- **Paths:** `src/motion_capture/rig/__main__.py`, `src/motion_capture/rig/soak.py`, `tests/motion_capture/rig/test_record_repeat.py`, `docs/development/DEVELOPMENT_LOG.md`
- **Started:** 2026-09-25
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Added `record --repeat N [--pause S]` soak mode to camera rig CLI, supporting N back-to-back takes in take subdirectories, machine-readable soak summary, and worst-take exit code.
- **Next step:** CI green, review, merge.

### DL-#9411 · Keep MyPy Exclusion Budget and Unit-Gate Quarantine Ratchets Green

- **State:** in_review
- **Owner:** claude (agy executor)
- **Issue:** #9411
- **PR:** draft PR from `claude/ud-9411-unquarantine-passing`
- **Branch:** `claude/ud-9411-unquarantine-passing`
- **Paths:** `scripts/config/unit_gate_quarantine.json`, quarantined unit test files, `src/shared/python/validation_pkg/data_fitting.py`, `src/tools/model_explorer/mujoco_viewer.py`, `src/shared/python/ai/adapters/`, `src/api/routes/__init__.py`, `src/shared/python/physics/impact_model/utils.py`, `src/shared/python/pendulum_simulator/physics.py`
- **Started:** 2026-09-25
- **Last verified:** 2026-09-26 — #10997 Linux run: kernel built and imported, none of the 12 retired BunkerShot IDs failed; the tests the kernel un-skipped exposed #11000 (fixed in #11004) and #11003 (fixed in #11010), and an untracked root `Cargo.lock` (fixed); replayed onto main d716c698f, ledger contract passes at 55 IDs.
- **Summary:** Retire unit-gate quarantine IDs whose tests pass on the Linux CI lane after fixing real defects (shoulder FK origin), deduplicating the `data_fitting` and `mujoco_viewer` coordinators onto their existing helper modules, and retargeting stale tests. Tools-owned `ai` adapter changes and the gear-effect sign question are out of scope here (Tools#5362 / a physics-convention decision). Earlier slice (#10973) landed the mypy budget and 43 retirements. Rust slice (#10997): unit-test-gate maturin-builds upstream-physics into its venv (ball_simulator enforces strict Rust parity), retiring the 12 BunkerShot workbench/GUI IDs. Consolidated delivered subscopes from the preserved source rows: Claude #10995 (claude/ud-9411-vendor-bump-safe-eval) advanced the Tools pin and retired 10 quarantined tests; Antigravity #10999 (feat/ud-9411-quarantine-burndown-batch1) retired 6; #11005 (feat/ud-9411-ai-adapters-convergence) converged Gemini/BitNet adapters and retired 5; #11013 (feat/ud-9411-quarantine-batch2) retired 1 and narrowed optional-dependency mock scope. The source snapshot retains original owners, branches, paths, verification text and next steps. Broader parent mypy/quarantine readiness remains in review; no full ratchet completion is claimed.
- **Next step:** Complete parent-level mypy-exclusion and quarantine acceptance against a source-bound Linux unit-gate run, then reconcile the remaining eligible IDs. The delivered child tranches do not establish full parent readiness; the original child next steps are preserved in the source snapshot.

### DL-#9703 · Versioned Pre-Impact Bundle for Tools Impact Kernels

- **State:** in_review
- **Owner:** claude
- **Issue:** #9703 (IA-U2; parent #9700)
- **PR:** draft (see branch)
- **Branch:** `feat/9703-pre-impact-bundle`
- **Paths:** `src/shared/python/physics/pre_impact_bundle.py`, `src/shared/python/physics/_pre_impact_contracts.py`, `src/shared/python/physics/_pre_impact_frames.py`, `tests/shared_contracts/test_pre_impact_bundle.py`, `docs/development/impact_acoustics_program.md`
- **Started:** 2026-09-25
- **Last verified:** 2026-09-25 — 48/48 tests pass against vendor pin a9ed0e7c5 (incl. python -O); pre-push mypy and unit hooks pass (Rust wheel hidden, #10946)
- **Summary:** Consumer-side PreImpactBundle v1 composing Tools conventions: provenance, bounded timebase, world/head/grip poses, head COM inertia and contact, ball, reduced shaft modal state and prestress field, per-hand wrench/impedance, per-field origin; absent fields raise; power-invariant transforms; energy-reporting modal projection. Engine adapters and installed-wheel fixtures remain.
- **Next step:** Review the draft PR, then implement the first engine adapter against the bundle under #9703.

### DL-#9415 · Repository Root Allowlist Check in Docs Governance

- **State:** in_review
- **Owner:** claude
- **Issue:** #9415
- **PR:** draft PR from `agy/ud-9415-root-allowlist`, #12334, #12464, #12472, #12393
- **Branch:** `agy/ud-9415-root-allowlist`
- **Paths:** `scripts/check_docs_governance.py`, `scripts/config/root_allowlist.json`, `docs/governance/DOCS_GOVERNANCE.md`, `tests/scripts/test_doc_governance_checks.py`, `tests/unit/scripts/test_check_docs_governance_root_allowlist.py`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`
- **Started:** 2026-09-26
- **Last verified:** 2026-10-14 (`b686ab17`; collated from changes/9415-spec-archive.md)
- **Summary:** Committed root entries (`git ls-tree --name-only HEAD`) must match `scripts/config/root_allowlist.json`; unlisted and stale entries both fail, and a failed git call or malformed config fails closed. Closes the last software checkbox of #9415; the history rewrite, Jules workflows and SPEC cap remain owner decisions.
- **Next step:** CI green, review, merge.

### DL-#8907 · One User Config Root and One QSettings Namespace for the Launcher

- **State:** in_review
- **Owner:** claude
- **Issue:** #8907
- **Branch:** fix/8907-settings-root
- **PR:** #10742 (open)
- **Paths:** src/shared/python/data_io/user_config_root.py; src/launchers/launcher_settings_store.py; src/launchers/launcher_constants.py; tests/unit/data_io/test_user_config_root.py; tests/launchers/test_launcher_settings_store.py
- **Started:** 2026-09-22
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Launcher writers (preferences, recent models, library, onboarding, process/launcher logs, layout reset/diagnostics) resolve through `user_config_path()` under the platformdirs root with a one-time copy from the two legacy dot-dirs; QSettings consolidated on `(UpstreamDrift, Launcher)` with legacy aliasing; three secondary windows persist geometry.
- **Next step:** Relocate the Tools-owned `~/.upstreamdrift/mcp_servers.json` contract in Tools, then point `McpServersConfig.default_path()` at the shared constant.

### DL-#9225 · GolfSwingVisualizer MATLAB Duplication Consolidation

- **State:** in_review
- **Owner:** local
- **Issue:** #9225 (source:assessment P2; DRY PP1)
- **Branch:** bot/issue-9225-golfviz-consolidation
- **PR:** #10715
- **Paths:** src/engines/Simscape_Multibody_Models/shared/+golfviz/GolfSwingVisualizer.m; src/engines/Simscape_Multibody_Models/2D_Golf_Model/matlab/2D GUI/launch_gui.m; src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/2D GUI/launch_gui.m; the two `2D GUI/main_scripts/golf_swing_analysis_gui.m` call sites; four deleted per-tree copies
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 at f33b40c9c (#10715) — MATLAB R2025b headless: package resolves to the single shared file from a bare `addpath`, class parses (75 methods), DbC precondition `GolfSwingVisualizer:InvalidInput` fires, and both launchers' relative path setups resolve the package with no shadow copy.
- **Summary:** Four near-identical `GolfSwingVisualizer.m` copies (1180–1183 lines each, 323 shared 8-line blocks between the worst pair) consolidated into one fleet-shared `+golfviz` package class; the two launchers add the shared directory to the MATLAB path and all four call sites use `golfviz.GolfSwingVisualizer`. Canonical behavior is the 2D variant superset (reproducible ground texture via `rng(1)` seeding that the 3D copies had silently lost); no genuine 2D-vs-3D behavioural divergence existed, so no parameter was needed.
- **Next step:** Confirm `quality-gate` green on the PR and allow squash auto-merge to land.
- **Evidence:** MATLAB R2025b `-batch` verification transcript in the PR body; `git ls-files` shows one `GolfSwingVisualizer.m`.

### DL-#8941 · Simulation-Page Polling Complete (Server + Client)

- **State:** in_review
- **Owner:** local
- **Issue:** #8941 (server via #10736 merged; client polling via #10748 merged; ActuatorPanel and SimulationToolbar loops onto usePolling here; WS frames stay with #8936/#8940)
- **Branch:** fix/8941-use-polling-toolbar-actuator
- **PR:** #10881 (open)
- **Paths:** ui/src/components/simulation/ActuatorPanel.tsx; ui/src/components/simulation/ActuatorPanel.test.tsx; ui/src/components/simulation/SimulationToolbar.tsx; ui/src/components/simulation/SimulationToolbar.test.tsx
- **Started:** 2026-09-24
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Complete the client REST polling migration for #8941: both ActuatorPanel (1000 ms) and SimulationToolbar (1000 ms) now use the shared usePolling hook with visibility and simulation-running gating, eliminating background resource contention when tabs are hidden.
- **Next step:** Publish force and analysis frames over /ws/simulate in #8936/#8940 to deprecate REST polling entirely.

### DL-#10385 · Pinocchio Driver/Iron G2-G3 Continuation and Independent Replay (MS-111)

- **State:** in_progress
- **Owner:** local
- **Issue:** #10385 (MS-111, epic #10363; folded PF-07 #10437)
- **Branch:** feat/10385-ms111-pinocchio-g2-g3-replay
- **PR:** #10723
- **Paths:** src/shared/python/motion_matching/pinocchio_g2_g3.py; src/engines/physics_engines/pinocchio/python/full_body_fit.py; tests/unit/motion_matching/test_pinocchio_g2_g3.py; docs/development/matched_swing_program/evidence/ms111/; docs/development/matched_swing_program/README.md; docs/development/matched_swing_program/MS31_PINOCCHIO_CROCODDYL_TURNOVER.md; docs/development/DEVELOPMENT_LOG.md; SPEC.md
- **Started:** 2026-09-22
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Software contracts for Pinocchio driver/iron G2/G3 continuation schedules, same-integrator solve/replay parity (reject integrator-specific solutions), armature propagation to equivalent-model independent replay packages (save/reopen), failed-continuation evidence preservation, open-loop q0/v0 control feed (no per-frame poses), and PF-07 robustness roster. Fitter exposes `--ms111-schedule` and fail-closed integrator parity. Native G2/G3 qualification remains blocked on MS-107 accepted G1 + MS-100 receipts.
- **Next step:** Confirm CI green on PR #10723 and allow squash auto-merge to land.
- **Evidence:** docs/development/matched_swing_program/evidence/ms111/continuation_contract_status.json; tests/unit/motion_matching/test_pinocchio_g2_g3.py.

### DL-#10271 · Restore End-to-End Provenance for Hip-Calibrated Motion Evidence

- **State:** in_review
- **Owner:** local
- **Issue:** #10271 (child of #10254; blocks #10162 physical acceptance provenance)
- **Branch:** fix/10271-hipcal-provenance
- **PR:** #10722
- **Paths:** src/shared/python/motion_matching/pipeline/receipt_provenance.py; src/shared/python/motion_matching/pipeline/receipt.py; src/shared/python/motion_matching/pipeline/receipt_schema.py; tests/unit/motion_matching/pipeline/test_receipt_provenance_chain.py; docs/development/full_body_models/evidence/ground_support/anthro_driver/receipt.json; docs/development/full_body_models/evidence/ground_support/anthro_iron/receipt.json; docs/development/full_body_models/RECEIPTS.md; docs/development/full_body_models/evidence/ground_support/CANONICAL_RUN.md
- **Started:** 2026-09-22
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Shared `validate_receipt_provenance_chain` (`receipt-provenance-chain/1`) fail-closes missing/stale de Leva, base canonical, and final raw/canonical digests; CI gates `anthro_driver`/`anthro_iron`; producer emits `spec_canonical_sha256`; receipts re-anchored to committed bases/scaled specs that already embed the current table; intermediate hipcal docs not fabricated; physical RMS unchanged/unqualified.
- **Next step:** Confirm CI green on PR #10722 and squash-merge; schedule native MuJoCo regen when disk is stable.

### DL-#10378 · Full-Swing Qualification for All Six Engines (MS-104)

- **State:** in_review
- **Owner:** local
- **Issue:** #10378 (MS-104, epic #10363; folded MS-109/110/112 owner blockers)
- **Branch:** feat/ms104-full-swing-qualification
- **PR:** #10707
- **Paths:** src/shared/python/motion_matching/full_swing_qualification.py; tests/unit/motion_matching/test_full_swing_qualification.py; docs/plans/matched_swing/evidence/ms104_full_swing_qualification.json; docs/development/HANDOFF.md; docs/development/DEVELOPMENT_LOG.md; SPEC.md
- **Started:** 2026-09-22
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Software-contract qualification matrix over 6 engines × driver/iron × G1/G2/G3. Rows require MS-100 acceptance, MS-72 conformance, native replay, and numerical-convergence links; reduced Simscape oracle stays partial. Release stays blocked until every required cell is fully linked — no invented six-engine native pass.
- **Next step:** Confirm CI green on PR #10707 after SPEC duplicate-key repair and allow squash auto-merge to land.

### DL-#8880 · GUI Thread-Blocking Simulation Migration to Async Action

- **State:** in_review
- **Owner:** local
- **Issue:** #8880
- **Branch:** fix/8880-gui-thread-blocking-sims
- **PR:** #10656
- **Paths:** src/tools/bunker_shot_gui/gui.py; src/tools/ball_flight_gui/gui.py; src/tools/swing_flight_pipeline/gui.py; src/tools/motion_matching/gui.py; src/shared/python/theme/tool_stylesheet.py; scripts/ci/check_gui_thread_blocking_ratchet.py; scripts/config/gui_thread_blocking_baseline.json; tests/tools/bunker_shot_gui/test_async_actions.py; tests/tools/ball_flight_gui/test_async_actions.py; tests/tools/swing_flight_pipeline/test_async_actions.py; tests/unit/scripts/test_gui_thread_blocking_ratchet.py
- **Started:** 2026-09-21
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Migrated `bunker_shot_gui`, `ball_flight_gui`, and `swing_flight_pipeline` onto `src/tools/async_action.py`; added lower-only GUI-thread-blocking ratchet (baseline 12); primary run buttons use shared theme wiring (#10654). ~9 tools remain un-migrated (see PR #10656 Deferred).
- **Next step:** Merge PR #10656 after CI green; follow-up PRs for remaining inline tools.

### DL-#10333 · Pinocchio MatchingPlant Full Lane (MS-14)

- **State:** in_review
- **Owner:** local
- **Issue:** #10333 (MS-14, epic #10363)
- **Branch:** feat/10333-pinocchio-matching-plant
- **PR:** #10684
- **Paths:** src/shared/python/motion*matching/pipeline/plants/pinocchio_plant.py; pinocchio_lane_receipts.py; receipt_components.py; cli.py; src/engines/physics_engines/pinocchio/python/full_body_fit.py; tests/unit/motion_matching/pipeline/test_pinocchio_plant.py; docs/development/full_body_models/evidence/ground_support/anthro_driver*{pinocchio,pink}/
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 at SELF — rematched onto origin/main after CO-06 #10687 merge (`f191dd09d`); regenerated matched_swing status README + divergence inventory; blocked native G1 receipts unchanged.
- **Summary:** Wire Pinocchio MatchingPlant derivatives + Pink `create_constrained_ik`, fitter `resolve_fit_native_plant` LoD bridge, ConstrainedIkReceipt `closure_residual_m` budget, and honest blocked evidence until ControlTower native run.
- **Next step:** Confirm quality-gate green + squash auto-merge of PR #10684; ControlTower native pink receipts when scheduled (MS-107 owns G1).
- **Evidence:** docs/development/full_body_models/evidence/ground_support/anthro_driver_pinocchio/; docs/development/full_body_models/evidence/ground_support/anthro_driver_pink/

### DL-#10345 · MyoSuite Kinematic Replay With Coordinate Map and Marker Parity Receipt

- **State:** in_review
- **Owner:** local
- **Issue:** #10345 (epic #10363, MS-52)
- **Branch:** fix/issue-10345-ms-52-local
- **PR:** #10666
- **Paths:** src/engines/physics_engines/myosuite/python/{retarget,replay,golfer_scene,coordinate_map_anthro.json,viz/render_replay.py}; tests/unit/engines/myosuite/test_retarget.py; tests/myosuite/test_replay_native.py; evidence/matched/driver_g1_myosuite/
- **Started:** 2026-09-21
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Added pure-numpy retarget map, kinematic replay CLI, marker parity receipt (`stage=replay`), MyoSuite registration in `cross_engine_replay.VALID_ENGINES` as kinematic-only, and viewer colour. Evidence committed under `evidence/matched/driver_g1_myosuite/`.
- **Next step:** Merge PR after CI green; drive 15 mm parity on the MS-51 golfer scene (still unqualified).

### DL-#9545 · Versioned BunkerShot3D Workbench API Route

- **State:** in_review
- **Owner:** claude
- **Issue:** #9545 (epic #9541)
- **Branch:** claude/issue-9545-api
- **PR:** #11516 (draft), #12449, #12495, #12440, #12506, #12515, #12519, #12522, #12535, #12541
- **Paths:** src/api/routes/bunker_workbench.py; src/tools/bunker_shot_gui/input_ranges.py; src/tools/bunker_shot_gui/design.py; src/tools/bunker_shot_gui/widgets.py; src/tools/bunker_shot_gui/report.py; src/tools/bunker_shot_gui/render.py; src/config/feature_parity.json; tests/bunkershot3d/test_bunker_workbench_api.py
- **Started:** 2026-10-04
- **Last verified:** 2026-10-15 (`09b41bd3`; collated from changes/9545-bunkershot3d-particle-playback-panel-emb.md)
- **Summary:** The headless workbench API enforces request ranges, handedness and source-stamped verdicts; predictive playability is refused or reported without ranking. UI and API share input-range authority.
- **Next step:** Merge the PR; producer (pychrono F3) stays deferred under #12536.

### DL-#9544 · Bunker Contact Regimes and Coupled Club Rotation Across Fidelity Tiers

- **State:** in_review
- **Owner:** claude
- **Issue:** #9544 (epic #9541)
- **Branch:** conductor/issue-9544
- **PR:** #10455 (open)
- **Paths:** src/bunkershot3d/ball/regimes.py; src/bunkershot3d/ball/pipeline.py; src/bunkershot3d/ball/splash.py; src/bunkershot3d/solvers/shot.py; src/bunkershot3d/solvers/mpm/wholeshot.py; src/bunkershot3d/vandv/conservation.py; src/tools/bunker_shot_gui/model.py; tests/bunkershot3d/ball/test_contact_regimes_9544.py; tests/bunkershot3d/solvers/test_rotation_coupling_9544.py; docs/bunkershot3d/contact-regimes.md
- **Started:** 2026-09-18
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Four contact regimes (no hit, direct/thin strike, splash, buried no-release) classified from the F0 sole path and divot stations; launch refused for every regime but splash. Prescribed rotation is a named `RotationMode`; an optional `RotationCoupling` boundary receives the sand wrench (about the body origin, world frame) and owns the angular velocity. V&V ledger adds support angular impulse and prescribed-driver work. F1 launch/out-of-plane refusals re-asserted; F2 requirements recorded in docs/bunkershot3d/contact-regimes.md.
- **Next step:** Open the PR with `Closes #9544` and record the merge SHA plus pinned Tools `62e8cdbf9c9f5f8a43a0342059f825e8fa78f8e1` in the completion comment.
- **Evidence:** tests/bunkershot3d/ball/test_contact_regimes_9544.py; tests/bunkershot3d/solvers/test_rotation_coupling_9544.py; docs/bunkershot3d/contact-regimes.md.

### DL-#9543 · Bunker Sand-to-Ball Transfer Calibration and Held-Out Qualification

- **State:** in_review
- **Owner:** claude
- **Issue:** #9543 (epic #9541)
- **Branch:** conductor/issue-9543
- **PR:** #10457 (open)
- **Paths:** src/bunkershot3d/ball/qualification.py; src/bunkershot3d/ball/qualification_fit.py; src/bunkershot3d/ball/rig_capability.py; src/bunkershot3d/ball/splash.py; src/bunkershot3d/ball/**init**.py; src/bunkershot3d/vandv/validation.py; src/bunkershot3d/vandv/measurement_intake.py; src/bunkershot3d/sand/provenance.py; tests/bunkershot3d/ball/test_transfer_qualification.py; docs/bunkershot3d/transfer-qualification.md
- **Started:** 2026-09-18
- **Last verified:** 2026-09-18 at SELF (base 49d94788a; 55 new tests pass; 1018 tests across tests/bunkershot3d ball, vandv, sand, study and public-API suites pass; scoped ruff, ruff format, mypy, LoD, file-size, architecture and error-handling ratchet checks pass; pinned Tools tree 62e8cdbf9c9f5f8a43a0342059f825e8fa78f8e1 materialised read-only for the run)
- **Summary:** Software intake/protocol and leakage controls support bounded fits, sensitivity checks and held-out V&V. No strokes are on file; physical qualification remains blocked and the issue stays open.
- **Next step:** Open the PR referencing #9543 (not `Closes`), then acquire measured strokes under `MEASUREMENT_PROTOCOL` before any qualification is attempted.
- **Evidence:** tests/bunkershot3d/ball/test_transfer_qualification.py; docs/bunkershot3d/transfer-qualification.md.

### DL-#9546 · Impact Zone Readiness Execution Index and I1/I2 Pin Consumption

- **Started:** not recorded in source
- **Owner:** not recorded in source
- **State:** in_review
- **Issue:** #9546 (children #9547, #9548, #9549, #9550; reused #9484, #9349)
- **Branch:** conductor/issue-9546
- **PR:** #10446
- **Paths:** src/config/impact_zone_readiness.json; src/config/industrial_readiness_loader.py; scripts/generate_industrial_readiness_index.py; docs/operations/impact-zone-readiness-index.md; tests/config/industrial_readiness/; tests/shared_contracts/test_impact_interval_provider.py
- **Last verified:** 2026-09-18 (SELF; ledger + freshness gates 45 passed; consumer contract 3 passed with `--tools-mode=vendored` against Tools pin `62e8cdbf9`)
- **Summary:** Epic #9546 gets the same machine-readable execution index as #9539: `impact_zone_readiness.json` reconciles I1–I4 and the two reused issues against `5347cba0f` and the vendored Tools pin, under the existing loader contract (keys `I-stage`/`R-stage` admitted, nothing else relaxed). The Tools fixes for I1 (#5088) and I2 (#5079) are consumed by a UD consumer contract that drives the audit probe through the vendored solver; I3, I4 and the Tools half of #9349 remain open with ordered plans, and `release_status` is `blocked`.
- **Next step:** When Tools #4946 lands the live interval run record, bump the pin and mark I1/I2/I3 in the ledger with merge SHAs, tests and acceptance evidence.
- **Evidence:** tests/config/industrial_readiness/; tests/shared_contracts/test_impact_interval_provider.py; docs/operations/impact-zone-readiness-index.md.

### DL-#10590 · TB-05: Fit and Replay the Hub–Arm–Club Triple Pendulum

- **State:** in_progress
- **Owner:** local
- **Issue:** #10590 (parent #10584)
- **Branch:** feat/tb05-triple-pendulum-fit-10590
- **PR:** #10644
- **Paths:** src/engines/physics_engines/pendulum/python/motion_matching/adapters_triple.py; src/engines/physics_engines/pendulum/python/motion_matching/torque_optimization_triple.py; src/engines/physics_engines/pendulum/python/motion_matching/provider_triple.py; src/engines/physics_engines/pendulum/python/motion_matching/qualification_triple.py; tests/unit/engines/physics_engines/pendulum/test_triple_pendulum_fit.py; docs/plans/tour_baselines/evidence/tb05_driver_qualification_receipt.json; docs/plans/tour_baselines/evidence/tb05_iron_qualification_receipt.json
- **Started:** 2026-09-21
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Driver/iron triple-pendulum fits use bounded controls, positive geometry and independent tighter replay; generated baseline packages and receipts remain the recorded evidence boundary.
- **Next step:** Open PR referencing #10590, await green CI, merge and release lease.
- **Evidence:** docs/plans/tour_baselines/evidence/tb05_driver_qualification_receipt.json; docs/plans/tour_baselines/evidence/tb05_iron_qualification_receipt.json; tests/unit/engines/physics_engines/pendulum/test_triple_pendulum_fit.py.

### DL-#9548 · Impact-Interval Energy Audit Consumer Gate

- **State:** in_review
- **Owner:** claude
- **Issue:** #9548 (parent #9546; provider Tools #4130 / #5079 / #5088)
- **Branch:** conductor/issue-9548
- **PR:** #10311
- **Paths:** src/shared/python/physics/impact_interval_audit.py; tests/unit/physics/test_impact_interval_audit.py; tests/shared_contracts/test_impact_interval_provider.py
- **Started:** 2026-09-17
- **Last verified:** 2026-09-17 (SELF; `tests/shared_contracts/` 33 passed, 0 skipped under `--tools-mode=vendored` at pin 1ac89c18e6280752d949e520c2143d2fb584d31e; 7 gate unit tests passed; pre-commit on changed files)
- **Summary:** The pinned Tools solver already integrates release, dashpot/friction, torsional damping and boundary storage independently of the residual (Tools #5079). UD consumes that pin through a fail-closed gate that recomputes the residual from the ledger identity, audits free vs supported momentum separately, reports evidence plus limitations as a JSON-ready record, and refuses `to_post_impact_state()` on unseparated contact or a failed audit. No UI consumer of the interval solver exists yet; the report record is the surface for one.
- **Next step:** Open the PR with `Closes #9548`, then wire the verdict report into the first UI/report consumer of the interval solver when one lands.
- **Evidence:** tests/shared_contracts/test_impact_interval_provider.py (interrupted compression 32.54 J stored / 0 J release / −0.063 J signed residual; clipping release 1.71 J; perturbed law residual > 0.5 J blocked; halving dt lowers both residuals).
  > > > > > > > origin/main

### DL-#10349 · Simscape 44-to-27 Coordinate Slice and Boundary-Load Validation

- **State:** in_review
- **Owner:** local
- **Issue:** #10349 (MS-62, epic #10363)
- **Branch:** fix/issue-10349-ms-62-local
- **PR:** #10665 (open)
- **Paths:** src/shared/python/motion_matching/coordinate_slice.py; tests/unit/motion_matching/test_coordinate_slice.py; evidence/matched/driver_g1_simscape_slice/; src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/align_measured_to_model.m
- **Started:** 2026-09-21
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Added JSON-backed 44-to-27 coordinate slice with name-based kinematic projection, virtual-work decomposition tests, boundary-wrench derivation for omitted neck/leg DOFs, CLI, and Simscape workspace overrides from anthropometric geometry documents in `align_measured_to_model.m`.
- **Next step:** R2025b Simscape native replay of `candidate27.npz` with boundary-load validation.
- **Evidence:** evidence/matched/driver_g1_simscape_slice/{slice_map.json,receipt.json,parity.json,run_manifest.json,candidate27.npz}; tests/unit/motion_matching/test_coordinate_slice.py.

### DL-#10342 · OpenSim/MyoSuite Native Nightly Lane Receipts

- **PR:** not recorded in source
- **State:** in_review
- **Owner:** local
- **Issue:** #10342 (MS-43, epic #10363)
- **Branch:** fix/issue-10342-ms-43-native-lane-local
- **Paths:** scripts/ci/run_native_engine_lane.py; scripts/ci/run_native_engine_lane.sh; docs/development/matched_swing_program/evidence/nightly/; tests/docs/test_native_lane_freshness.py; tests/scripts/test_run_native_engine_lane.py
- **Started:** 2026-09-21
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Added idempotent native-engine lane runner emitting hashed nightly receipts for OpenSim and MyoSuite (`requires_opensim` / `requires_myosuite` pytest markers), freshness gate warning at seven days and failing at thirty days, and ControlTower verification docs without editing `.github/workflows`.
- **Next step:** Refresh receipts on ControlTower opensim-10003 venv; merge PR closing #10342.
- **Evidence:** docs/development/matched_swing_program/evidence/nightly/opensim_receipt.json; docs/development/matched_swing_program/evidence/nightly/myosuite_receipt.json; tests/docs/test_native_lane_freshness.py.

### DL-#9422 · Rig Capture Sessions Through the Tools MocapSession Contract

- **State:** in_review
- **Owner:** claude
- **Issue:** #9422 (readiness P6; Tools #4706 M-track consumer)
- **Branch:** conductor/issue-9422
- **PR:** #10466 (open)
- **Paths:** src/motion_capture/rig/tools_bridge.py; src/motion_capture/rig/**main**.py; tests/motion_capture/rig/test_tools_session_export.py; tests/fixtures/mocap_session_export/; docs/motion_capture/capture_rig.md
- **Started:** 2026-09-19
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Rig capture export uses the pinned Tools MocapSessionManifest builders, consent policy and camera identity records. This is one consumer slice; D-track and prerequisite issues remain open.
- **Next step:** Open the PR, then route the C3D upload path (#8865) through the same pinned contract as the next consumer slice.
- **Evidence:** tests/fixtures/mocap_session_export/export_checks.py; tests/motion_capture/rig/test_tools_session_export.py.

### DL-#9162 · Local Branch Triage

- **PR:** not recorded in source
- **State:** in_progress
- **Owner:** claude
- **Issue:** #9162 (rollout Repository_Management#1460; no-seed bootstrap #9161)
- **Branch:** conductor/issue-9162
- **Paths:** docs/development/branch_triage_9162.md
- **Started:** 2026-09-15
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Disposition ledger for every local branch in the primary checkout: 3 protected, 27 deferred to the worktree pass, 34 live (merged sweep then DL entry), 178 stale and 144 abandoned scratch branches (bundle snapshot then delete). The runbook in the ledger executes the deletions and the coordination notice for agent-owned branches.
- **Next step:** Execute the runbook in docs/development/branch_triage_9162.md from the primary checkout and fill in its Outcome section.

### DL-#9410 · Adversarial Product Review Remediation Epic

- **PR:** not recorded in source
- **State:** in_progress
- **Owner:** claude
- **Issue:** #9410 (program Repository_Management#1505; children #8820–#8943, #8360, #8641, #8843, #8846, #8853, #8861–#8870, #8874–#8876, #8894)
- **Branch:** conductor/issue-9410
- **Paths:** docs/development/adversarial_review_remediation_9410.md
- **Started:** 2026-09-16
- **Last verified:** 2026-09-16 (SELF; every child re-checked against `db4fe88c4` by first-parent history search plus source reads of each residual)
- **Summary:** Reconciliation ledger for the 2026-08-21 adversarial review: 34 of 61 children landed on `main` (SHAs recorded), 27 residual. Three residual fixes already exist on unmerged branches (`readiness/p0-9412-one-tile-registry`, `conductor/issue-8865`, `conductor/issue-8866`). Epic acceptance (one registry, one API factory, one C3D reader, one pose type, no GUI-thread simulation) is not met on `main`.
- **Next step:** Rebase and merge `readiness/p0-9412-one-tile-registry`, then re-verify the Cluster B rows in the ledger.

### DL-#10602 · Club-Only Motion Matching Plan

- **State:** in_review
- **Owner:** codex
- **Issue:** #10602
- **Branch:** `agy/ud-10602-real-matrix`
- **PR:** draft PR from `agy/ud-10602-real-matrix`
- **Paths:** docs/plans/club_neural_review/; docs/plans/club_only_matching/; docs/plans/neural_motion_matching/; src/shared/python/motion_matching/club_only/
- **Started:** 2026-09-20
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Published bounded implementation issues with TDD/DbC/LoD/DRY prompts, dependency ordering, native validation gates and shared technical review. Planning artifacts do not qualify physical results or speedup.
- **Next step:** CI green, review, merge.
- **Evidence:** docs/plans/club_neural_review/REVIEW.md; docs/plans/club_neural_review/excel_audit.json.

### DL-#10603 · Neural Motion Matching Plan

- **State:** proposed
- **Owner:** codex
- **Issue:** #10603
- **Branch:** docs/club-neural-matching-plans-20260920
- **PR:** #10628
- **Paths:** docs/plans/club_neural_review/; docs/plans/club_only_matching/; docs/plans/neural_motion_matching/
- **Started:** 2026-09-20
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Published bounded implementation issues with TDD/DbC/LoD/DRY prompts, dependency ordering, native validation gates and shared technical review. Planning artifacts do not qualify physical results or speedup.
- **Next step:** Land NM-02 (#10617) PR; do not start NM-03+.
- **Evidence:** docs/plans/club_neural_review/REVIEW.md; docs/plans/club_neural_review/excel_audit.json.

### DL-#9550 · Impact Explorer Acceptance Matrix and Served-Bundle Verification

- **State:** in_review
- **Owner:** claude
- **Issue:** #9550 (epic #9546)
- **Branch:** conductor/issue-9550
- **PR:** #10441, #12471
- **Paths:** src/config/impact_acceptance.json; tests/config/impact_acceptance/test_impact_acceptance_matrix.py; scripts/ci/verify_impact_explorer_bundle.py; tests/ci/test_verify_impact_explorer_bundle.py; tests/api/test_impact_explorer_mount.py; .github/workflows/ci-standard.yml; src/config/feature_parity.json; docs/development/impact_acceptance_matrix.md
- **Started:** 2026-09-18
- **Last verified:** 2026-10-14 (`126b0964`; collated from changes/9550-add-frozen-impactmodelcapabilitymatrix-a.md)
- **Summary:** Froze the model-capability matrix for `src/shared/python/physics/impact_model` (spin support stated as unavailable where absent), recorded the six acceptance items with evidence or explicit open state and a `code_verified_only` claim, made CI stamp Tools' release artifacts with the pinned gitlink and verify that `/impact-explorer-app/` serves that revision's real JavaScript, and downgraded `tools.rate_of_closure` parity to an evidence-based gap.
- **Next step:** Open the PR and, once #9417 fixes the artifact set, run the served-bundle verifier against the installed artifact with restart/reload/offline checks.
- **Evidence:** docs/development/impact_acceptance_matrix.md; tests/config/impact_acceptance/test_impact_acceptance_matrix.py.

### DL-#9541 · BunkerShot3D Product Acceptance Matrix

- **State:** in_review
- **Owner:** claude
- **Issue:** #9541 (epic; children #9542–#9545, #9239, #9286, #8733, #8880, #9688–#9695)
- **Branch:** conductor/issue-9541
- **PR:** #10459 (open)
- **Paths:** src/config/bunkershot3d_qualification.json; tests/config/bunkershot3d_qualification/test_bunkershot3d_qualification_ledger.py; CLAUDE.md
- **Started:** 2026-09-18
- **Last verified:** 2026-09-18 at 9af408974 (SELF); 16 matrix tests plus the 21 sibling readiness-ledger tests pass on Python 3.12 with the Tools pin 62e8cdbf materialized; ruff check and format clean.
- **Summary:** Machine-readable product acceptance matrix for epic #9541, reusing the #9539 readiness-ledger loader unchanged: all sixteen checklist children recorded open with owner, dependency order and narrow TDD plan; eight acceptance criteria from the epic's prediction protocol (one met, two partial, five unmet); `release_status` bound to `shipped_register()` and `credibility_assessment()` so it cannot be greened by editing JSON. No physics, calibration or GUI change; the tool remains an exploratory simulator.
- **Next step:** Land the matrix, then start U1 (#9286) and U2 (#9542) with their RED tests and update their entries with merge SHAs.
- **Evidence:** src/config/bunkershot3d_qualification.json; tests/config/bunkershot3d_qualification/test_bunkershot3d_qualification_ledger.py.

### DL-#10363 · Matched Swing Continuation Review & Drake G1 Retraction

- **State:** in_review
- **Owner:** claude-deskcomputer-20260925-ud
- **Issue:** #10363
- **Branch:** agy/ud-10363-drake-retraction
- **PR:** draft PR from `agy/ud-10363-drake-retraction`
- **Paths:** docs/development/matched_swing_program; src/shared/python/motion_matching/acceptance.py; src/engines/physics_engines/drake/python/full_body_fit.py; evidence/matched/driver_g1_drake/reevaluation.json; docs/development/full_body_models/evidence/acceptance/verdicts_2026-09.json; reports/matched_swing_ledger.json
- **Started:** 2026-09-18
- **Last verified:** 2026-09-25 at ff9fbee62 (retracted Drake G1 fabricated PASS via reevaluation.json; closed synthesis paths in full_body_fit.py; added fail-closed integrity gates in acceptance.py; 0 PASSED rows in ledger; motion_matching/tour_baselines/opensim ladder/browser suites green except 13 pre-existing local failures)
- **Summary:** 2026-09-18 continuation review (codex) preserved checkpoint/source recovery evidence and bounded the cheaper-agent continuation, with no physical acceptance claimed. Then: retracted the fabricated Drake G1 receipt (PR #10506) under MS-100; added fail-closed evidence integrity gates (\_evaluate_evidence_integrity) rejecting placeholder hashes and zero residuals without replay evidence; removed physical_audit numeric literals, np.zeros target fallback, and silent warm-start fallback in Drake full_body_fit.py; regenerated ledger and status matrices confirming 0 PASSED rows.
- **Next step:** CI green, review, merge.
- **Evidence:** docs/development/matched_swing_program/evidence/continuation_20260918/matching-handoff-snapshot.json; evidence/matched/driver_g1_drake/reevaluation.json; reports/matched_swing_ledger.json; tests/unit/motion_matching/test_acceptance.py; tests/unit/motion_matching/test_drake_full_body_fit.py.

### DL-#10355 · Motion Matching Tile Visual Playback, Standardized Metrics, and Navigation Handoff (MS-82)

- **State:** in_progress
- **Owner:** local
- **Issue:** #10355 (MS-82, closes #10106 gap)
- **Branch:** feat/10355-motion-matching-tile-playback
- **PR:** #10995
- **Paths:** src/tools/motion_matching/gui.py; src/tools/motion_matching/pipeline.py; src/config/feature_parity.json; docs/development/feature_parity_matrix.md; tests/tools/motion_matching/test_motion_matching_gui.py
- **Started:** 2026-09-20
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Enhanced the Motion Matching PyQt6 tile to display asynchronous animation playback (`ik_playback.gif` and `tracking_playback.gif`) via `QMovie`, populated five standardized headline metrics (`full_capture_ik_rms_mm`, `address_marker_rms_mm`, `backswing_root_error_max_mm`, `whole_run_root_rms_mm`, `inside_support_polygon_fraction`), acceptance verdict badge, and navigation handoffs to the Matched Swing Results Browser and Tour Matching Viewer. Upgraded `tools.motion_matching` in `feature_parity.json` from `gap` to `parity`.
- **Next step:** Create PR, enable auto-merge, verify merge, and release lease.
- **Evidence:** tests/tools/motion_matching/test_motion_matching_gui.py, tests/config/feature_parity/test_matrix_freshness.py.

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

### DL-#10402 · OpenSim Qualify Muscle and Tendon Extensions Without Replacing Baseline

- **State:** in_review
- **Owner:** local
- **Issue:** #10402 (epic #10394 / #10363, OG-08)
- **Branch:** feat/og08-qualify-muscle-tendon-extensions-10402
- **PR:** #10413
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/muscle_qualification.py; src/engines/physics_engines/opensim/python/tour_matching/**init**.py; tests/opensim/test_muscle_cmc.py; src/engines/physics_engines/opensim/python/POST_MVP_MUSCLES.md; docs/development/DEVELOPMENT_LOG.md; docs/development/HANDOFF.md
- **Started:** 2026-09-18
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Qualified muscle and tendon extensions (`muscle_qualification.py`) without replacing the torque baseline. Provides explicit anatomy coverage audit (disallowing lower-extremity-only models from claiming full golf capability), parameter provenance and licensing audits, MTU path and wrapping verification, moment arm vs. finite-difference path-length derivative validation, activation dynamics, initial tendon equilibrium, and short replay receipts reporting reserve torques and pelvic residuals. Keeps epic muscle-complete status open pending independent #10375 validation.
- **Next step:** Land PR #10413 referencing Closes #10402.
- **Evidence:** tests/opensim/test_muscle_cmc.py; src/engines/physics_engines/opensim/python/tour_matching/muscle_qualification.py.

### DL-#10400 · OpenSim Rebuild Full-Swing Tracking From Qualified Address

- **State:** in_review
- **Owner:** local
- **Issue:** #10400 (epic #10394 / #10363, OG-06)
- **Branch:** feat/og06-rebuild-full-swing-tracking-10400
- **PR:** #10410
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/full_swing_tracking.py; src/engines/physics_engines/opensim/python/tour_matching/**init**.py; tests/opensim/test_moco_g1_ladder.py; docs/development/DEVELOPMENT_LOG.md; docs/development/HANDOFF.md
- **Started:** 2026-09-18
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
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
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
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
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
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
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Implemented consistent segment scaling module (`segment_scaling.py`) and acromion proxy reconstruction for unilateral marker occlusions (`scale.py`). Joint frames, visual meshes, COM, and inertia are scaled in lockstep. Regenerated qualified `golf_humanoid_scaled.osim` with attached visual club and consistent scaling, passing qualification gates without unscaled arm mesh defects.
- **Next step:** Land PR #10407 referencing Closes #10396.
- **Evidence:** tests/opensim/test_segment_scale.py; tests/opensim/test_opensim_os0_qualification.py; src/engines/physics_engines/opensim/models/golf_humanoid_scaled.osim.

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

### DL-#9967 · Native Simscape Tour Matching

- **PR:** not recorded in source
- **State:** in_progress
- **Owner:** codex (turnover review; execution ownership by next lease)
- **Issue:** #9967 (parent #9921)
- **Branch:** feat/9967-native-simscape-pinocchio
- **Paths:** src/shared/python/motion_matching; docs/development/simscape_tour_matching
- **Started:** 2026-09-09
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Run101 improves yaw and has measured R2025b–Pinocchio prefix agreement of 0.0605 mm maximum. Terminal RMS 40.31 mm fails the 35 mm gate; full 1.814 s capture is incomplete. No optimizer launched by this review.
- **Next step:** Follow COMPLETION_HANDOFF_20260916.md: restore clean-runtime native providers, coordinate #10260 finite-weld derivative qualification, produce articulated feasibility evidence, fit0.90 s and deliver repeatable reports. Run102 remains rejected; no new compute launched by this review.

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
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
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
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
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
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Full-body handoff aligns headline values to primary receipts, adds matched-swing status/docs freshness, and centralizes ground-support IK/dynamics behind shared modules with compatibility shims.
- **Next step:** Commit MS-11 (#10330), open PR, auto-merge, and proceed to next Wave 1 task (MS-10 #10329).
- **Evidence:** docs/development/full_body_models/evidence/ground_support/CANONICAL_RUN.md, docs/development/full_body_models/evidence/ground_support/bisect_receipt.json, tests/unit/motion_matching/test_handoff_numbers_match_receipts.py, tests/unit/motion_matching/test_full_body_consolidation.py, docs/development/matched_swing_program/README.md, docs/development/matched_swing_program/GATES.md, docs/development/matched_swing_program/WAVES.md, scripts/generate_matched_swing_status.py, tests/docs/test_matched_swing_status_freshness.py.

### DL-#8869 · Realtime Pub/Sub: Wire the WS Transport Instead of a Silent No-Op

- **State:** in_review
- **Owner:** claude
- **Issue:** #8869 (folds in #8868, #8942A; seam #9406 ruling: `realtime` is `split pending`, UD keeps this facade)
- **Branch:** fix/8869-realtime-pubsub-decision
- **PR:** #10655 (open)
- **Paths:** `src/shared/python/realtime/api.py`; `src/shared/python/realtime/channels.py`; `src/shared/python/realtime/__init__.py`; `tests/unit/realtime/test_facade.py`; `tests/unit/realtime/test_channels.py`; `tests/shared/realtime/test_api.py`; `tests/shared/realtime/test_channels.py`; `docs/config/pydantic-settings-migration.md` (removed `file_pubsub.py` mention)
- **Started:** 2026-09-21
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** WebSocket transport is opt-in; invalid settings fail immediately. The channel-name collision and redundant file transport were corrected. Automatic default routing stays off to avoid an implicit server; #8868 and #8942A remain separate follow-ups.
- **Next step:** Push rebase merge commit, await green CI, merge, release lease.
- **Evidence:** tests/unit/realtime/test_facade.py (`test_publish_with_ws_transport_routes_to_ws`, `test_publish_with_unsupported_transport_raises`, `test_subscribe_with_ws_transport_routes_to_ws`); tests/shared/realtime/test_api.py (`test_publish_ws_transport_routes_to_ws`, `test_publish_unsupported_transport_raises`, `test_subscribe_ws_transport_routes_to_ws`).

### DL-#9193 · Companion Documentation and Capability Evidence Authority

- **State:** in_review
- **Owner:** claude
- **Issue:** #9193 (parent #9174)
- **Branch:** conductor/issue-9193
- **PR:** #10668
- **Paths:** scripts/companion_evidence.py; scripts/companion_catalog.py; scripts/config/companion_documentation.v1.json; scripts/config/companion_capability_evidence.v1.json; docs/api/contracts/upstreamdrift-companion-v1.schema.json; docs/engines/engine_capability_evidence.md; tests/companion/test_companion_evidence.py
- **Started:** 2026-09-17
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Replace the empty documentation inventory and unqualified engine placeholders with two hashed registries parsed by one module: exact-commit, hash-bound, immutable-URL documentation records with derived freshness; engine capabilities qualified only by exact test/artifact evidence with an executing CI gate; per-program documentation routes; known gaps with owning issues; derived publication blockers; generated, freshness-checked provider page.
- **Next step:** Open the non-draft PR with `Closes #9193`, then record reviews for the sixteen `unknown` documentation records in follow-up PRs.

## Shipped (Last 90 Days)

### DL-#12748 · SPN-6a: Spine Model Asset Manifest and Local-Source SHA-256 Admission

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12748
- **Branch:** merged via #12781
- **PR:** #12781
- **Paths:** see #12781
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`7dea9b22`; collated from changes/12748-spn-6a-spine-model-asset-manifest-and-lo.md)
- **Summary:** SPN-6a: spine model asset manifest and local-source SHA-256 admission
- **Next step:** Shipped in PR #12781.

### DL-#12747 · SPN-5 Spinal Load Series, Sign Conventions and Tension-Positive Axial Bridge

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12747
- **Branch:** merged via #12781
- **PR:** #12781
- **Paths:** see #12781
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`7dea9b22`; collated from changes/12747-spn-5-spinal-load-series-sign-convention.md)
- **Summary:** SPN-5 spinal load series, sign conventions and tension-positive axial bridge
- **Next step:** Shipped in PR #12781.

### DL-#12740 · SPN-1B: Segmental Spine Per-Level Transforms and Ordered Composition (Spine.Segmental)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12740
- **Branch:** merged via #12781
- **PR:** #12781
- **Paths:** see #12781
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`7dea9b22`; collated from changes/12740-spn-1b-segmental-spine-per-level-transfo.md)
- **Summary:** SPN-1b: segmental spine per-level transforms and ordered composition (spine.segmental)
- **Next step:** Shipped in PR #12781.

### DL-#12739 · SPN-1a: Spinal Level Enumeration, Regions and Adjacency (Spine.Levels)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12739
- **Branch:** merged via #12781
- **PR:** #12781
- **Paths:** see #12781
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`7dea9b22`; collated from changes/12739-spn-1a-spinal-level-enumeration-regions.md)
- **Summary:** SPN-1a: spinal level enumeration, regions and adjacency (spine.levels)
- **Next step:** Shipped in PR #12781.

### DL-#12738 · Segmental Spine Research Reference Scaffold, Provisional Manual Chapter 66 and Registry Blocker

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12738
- **Branch:** merged via #12781
- **PR:** #12781
- **Paths:** see #12781
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`7dea9b22`; collated from changes/12738-segmental-spine-research-reference-scaff.md)
- **Summary:** Segmental spine research reference scaffold, provisional manual chapter 66 and registry blocker
- **Next step:** Shipped in PR #12781.

### DL-#12737 · Primary Data Extraction for Segmental Spine (Bruno, Fujii, Wilke, Golf Loads) With Provenance Flags

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12737
- **Branch:** merged via #12781
- **PR:** #12781
- **Paths:** see #12781
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`7dea9b22`; collated from changes/12737-primary-data-extraction-for-segmental-sp.md)
- **Summary:** Primary data extraction for segmental spine (Bruno, Fujii, Wilke, golf loads) with provenance flags
- **Next step:** Shipped in PR #12781.

### DL-#12736 · ADR-0055: Segmental Spine Kinematics, Load Observables and Golf-Term Definitions (SPN-0)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12736
- **Branch:** merged via #12781
- **PR:** #12781
- **Paths:** see #12781
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`7dea9b22`; collated from changes/12736-adr-0055-segmental-spine-kinematics-load.md)
- **Summary:** ADR-0055: segmental spine kinematics, load observables and golf-term definitions (SPN-0)
- **Next step:** Shipped in PR #12781.

### DL-#12775 · OpenSim MSK Renders Draw the Wrapped-Finger Grip Hands on the Club and Hide the Flat Rajagopal Hand Bones

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12775
- **Branch:** merged via #12779
- **PR:** #12779
- **Paths:** see #12779
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`cf26f3db`; collated from changes/12775-opensim-msk-renders-draw-the-wrapped-fin.md)
- **Summary:** OpenSim MSK renders draw the wrapped-finger grip hands on the club and hide the flat Rajagopal hand bones
- **Next step:** Shipped in PR #12779.

### DL-#12710 · NSX-81: Add Orthogonal Conditioning Axis to QuantityRecord so INVARIANT Quantities Can Declare Hypotheses They Are Conditioned On

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12710
- **Branch:** merged via #12733
- **PR:** #12733
- **Paths:** see #12733
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`b3048176`; collated from changes/12710-nsx-81-add-orthogonal-conditioning-axis.md)
- **Summary:** NSX-81: add orthogonal conditioning axis to QuantityRecord so INVARIANT quantities can declare hypotheses they are conditioned on
- **Next step:** Shipped in PR #12733.

### DL-#12680 · R1: Trunk, Pelvis and Shoe Appearance Cross-Sections Now Oriented From Spec Landmarks (ML > AP); Lofted_Segment Requires an Explicit Width Axis for Non-Round Sections; New Model_Appearance.Cross_Section; Cross-Engine Trunk Guard; golfer_appearance_reference.tex Master + Trunk_Cross_Section Section

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12680
- **Branch:** merged via #12696
- **PR:** #12696
- **Paths:** see #12696
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`cf25212b`; collated from changes/12680-r1-trunk-pelvis-and-shoe-appearance-cros.md)
- **Summary:** R1: trunk, pelvis and shoe appearance cross-sections now oriented from spec landmarks (ML > AP); lofted_segment requires an explicit width axis for non-round sections; new model_appearance.cross_section; cross-engine trunk guard; golfer_appearance_reference.tex master + trunk_cross_section section
- **Next step:** Shipped in PR #12696.

### DL-#12700 · Pinocchio Reference Adapter Finite-Differences the Trajectory on the Base Lie Group (World Linear, Body Angular via the Quaternion Log) and Feeds the Result Through the NSX-50 Velocity Map, Instead of Differentiating Raw Native Q

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12700
- **Branch:** merged via #12704
- **PR:** #12704
- **Paths:** see #12704
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`12f14c0b`; collated from changes/12700-pinocchio-fd-frames.md)
- **Summary:** Pinocchio reference adapter finite-differences the trajectory on the base Lie group (world linear, body angular via the quaternion log) and feeds the result through the NSX-50 velocity map, instead of differentiating raw native q
- **Next step:** Shipped in PR #12704.

### DL-#12669 · Correct the Club Y-Sign, Hub Direction and Club Jacobians in the JAX Golfer Path; Club Points and Club-Angle Jacobian Columns Now Share Backend-Agnostic Formulas With the NumPy Kinematics, Pinned by NumPy-vs-JAX Parity Tests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12669
- **Branch:** merged via #12679
- **PR:** #12679
- **Paths:** see #12679
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`3ec138c7`; collated from changes/12669-correct-the-club-y-sign-hub-direction-an.md)
- **Summary:** Correct the club y-sign, hub direction and club Jacobians in the JAX golfer path; club points and club-angle Jacobian columns now share backend-agnostic formulas with the NumPy kinematics, pinned by NumPy-vs-JAX parity tests
- **Next step:** Shipped in PR #12679.

### DL-#12592 · NSX-74 Publication Gate Checklist for Loop Null-Space Results

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12592
- **Branch:** merged via #12664
- **PR:** #12664
- **Paths:** see #12664
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`b24d4235`; collated from changes/12592-nsx-74-publication-gate-checklist-for-lo.md)
- **Summary:** NSX-74 publication gate checklist for loop null-space results
- **Next step:** Shipped in PR #12664.

### DL-#12562 · NSX-17 Invariant Quantities With Validity Gates

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12562
- **Branch:** merged via #12638
- **PR:** #12638
- **Paths:** see #12638
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`d497012c`; collated from changes/12562-nsx-17-invariant-quantities-with-validit.md)
- **Summary:** NSX-17 invariant quantities with validity gates
- **Next step:** Shipped in PR #12638.

### DL-#12584 · Two-Track Cross-Engine Closed-Loop Conformance Harness (Numerical Snapshot Equality Plus Propagated Scientific Envelopes), Registry Entries per Engine Pair, Native OpenSim Adapter Registered in the Engine Factory

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12584
- **Branch:** merged via #12639
- **PR:** #12639
- **Paths:** see #12639
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`16e88462`; collated from changes/12584-two-track-cross-engine-closed-loop-confo.md)
- **Summary:** Two-track cross-engine closed-loop conformance harness (numerical snapshot equality plus propagated scientific envelopes), registry entries per engine pair, native OpenSim adapter registered in the engine factory
- **Next step:** Shipped in PR #12639.

### DL-#12681 · Pelvis Wears the Active Outfit's Pants Material: Body Classification Is Order-Independent (Leaf Segment Before Parent Folder, Exact Before Prefix Before Substring, Equal-Specificity Ties Raise), GolfSwing3D LowerTorso Maps to Pelvis, and Check_Clothing Enforces Pelvis Garment == Pants for Every Preset

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12681
- **Branch:** merged via #12694
- **PR:** #12694
- **Paths:** see #12694
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`7d59de0d`; collated from changes/12681-pelvis-wears-the-active-outfit-s-pants-m.md)
- **Summary:** Pelvis wears the active outfit's pants material: body classification is order-independent (leaf segment before parent folder, exact before prefix before substring, equal-specificity ties raise), GolfSwing3D LowerTorso maps to pelvis, and check_clothing enforces pelvis garment == pants for every preset
- **Next step:** Shipped in PR #12694.

### DL-#12580 · NSX-52: MyoSuite CLOSED_LOOP_SNAPSHOT via MuJoCo Adapter Reuse; Kkt_Regularization Recorded, FULL Only At 0

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12580
- **Branch:** merged via #12636
- **PR:** #12636
- **Paths:** see #12636
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`91feef10`; collated from changes/12580-nsx-52-myosuite-closed-loop-snapshot-via.md)
- **Summary:** NSX-52: MyoSuite CLOSED_LOOP_SNAPSHOT via MuJoCo adapter reuse; kkt_regularization recorded, FULL only at 0
- **Next step:** Shipped in PR #12636.

### DL-#12561 · NSX-16: Grip-Frame Split Coordinates, Grasp Map Kernel and Bare-Club Net Wrench

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12561
- **Branch:** merged via #12632
- **PR:** #12632
- **Paths:** see #12632
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`54f33667`; collated from changes/12561-nsx-16-grip-frame-split-coordinates-gras.md)
- **Summary:** NSX-16: grip-frame split coordinates, grasp map kernel and bare-club net wrench
- **Next step:** Shipped in PR #12632.

### DL-#12605 · Pinocchio Reference Adapter Maps the Free-Flyer World Linear Velocity/Acceleration Into Pinocchio's LOCAL Frame via the NSX-50 Velocity Map

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12605
- **Branch:** merged via #12699
- **PR:** #12699
- **Paths:** see #12699
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`30a72e42`; collated from changes/12605-pinocchio-ff-velocity.md)
- **Summary:** Pinocchio reference adapter maps the free-flyer world linear velocity/acceleration into Pinocchio's LOCAL frame via the NSX-50 velocity map
- **Next step:** Shipped in PR #12699.

### DL-#12560 · NSX-15 Closed-Loop Decomposition: Dimension Accounting, Zero-Side Three-Case Solve, Minimum-Metric Torque

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12560
- **Branch:** merged via #12631
- **PR:** #12631
- **Paths:** see #12631
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`a0b7f978`; collated from changes/12560-nsx-15-closed-loop-decomposition-dimensi.md)
- **Summary:** NSX-15 closed-loop decomposition: dimension accounting, zero-side three-case solve, minimum-metric torque
- **Next step:** Shipped in PR #12631.

### DL-#12581 · NSX-53 Native Drake Force Adapter With Closed-Loop Snapshot; Synthetic Drake Route Quarantined to Tests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12581
- **Branch:** merged via #12630
- **PR:** #12630
- **Paths:** see #12630
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`3b2357c0`; collated from changes/12581-nsx-53-native-drake-force-adapter-with-c.md)
- **Summary:** NSX-53 native Drake force adapter with closed-loop snapshot; synthetic Drake route quarantined to tests
- **Next step:** Shipped in PR #12630.

### DL-#12559 · NSX-14 Kinematic and Contact Admission Gate (Admit, AdmittedTrajectory)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12559
- **Branch:** merged via #12625
- **PR:** #12625
- **Paths:** see #12625
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`caf2cfb0`; collated from changes/12559-nsx-14-kinematic-and-contact-admission-g.md)
- **Summary:** NSX-14 kinematic and contact admission gate (admit, AdmittedTrajectory)
- **Next step:** Shipped in PR #12625.

### DL-#12675 · Guard MOSAIC Inertial Parameter Layouts: Audit Found No Live Pinocchio/inertial.py Mix; Tests Pin Each Module's Ordering and Block Unconverted Coupling

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12675
- **Branch:** merged via #12676
- **PR:** #12676
- **Paths:** see #12676
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`f9f46844`; collated from changes/12675-guard-mosaic-inertial-parameter-layouts.md)
- **Summary:** Guard MOSAIC inertial parameter layouts: audit found no live Pinocchio/inertial.py mix; tests pin each module's ordering and block unconverted coupling
- **Next step:** Shipped in PR #12676.

### DL-#12582 · NSX-54: OpenSim Closed-Loop Snapshot Adapter Delegating to OpenSimFullBodyParityAdapter, Equal to Pinocchio

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12582
- **Branch:** merged via #12627
- **PR:** #12627
- **Paths:** see #12627
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`bb5fca69`; collated from changes/12582-nsx-54-opensim-closed-loop-snapshot-adap.md)
- **Summary:** NSX-54: OpenSim closed-loop snapshot adapter delegating to OpenSimFullBodyParityAdapter, equal to Pinocchio
- **Next step:** Shipped in PR #12627.

### DL-#12579 · MuJoCo Closed-Loop Snapshot by Hard-KKT Evaluation (CLOSED_LOOP_SNAPSHOT FULL), Separate Stock-Soft-Weld Divergence Entry

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12579
- **Branch:** merged via #12629
- **PR:** #12629
- **Paths:** see #12629
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`67eeb469`; collated from changes/12579-mujoco-closed-loop-snapshot-by-hard-kkt.md)
- **Summary:** MuJoCo closed-loop snapshot by hard-KKT evaluation (CLOSED_LOOP_SNAPSHOT FULL), separate stock-soft-weld divergence entry
- **Next step:** Shipped in PR #12629.

### DL-#12608 · Correct the Club Y-Sign in Golfer Forward Kinematics, Club Jacobians and the Native Grip_Left Wrapper; Closure Is a Wrist-Pin PLANAR_POINT2 With Rank 2 on the Manifold (TB-00 Registry: 6 Independent DOFs); Planar-Model and Central-Difference Cross-Checks

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12608
- **Branch:** merged via #12668
- **PR:** #12668
- **Paths:** see #12668
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`41f73696`; collated from changes/12608-correct-the-club-y-sign-in-golfer-forwar.md)
- **Summary:** Correct the club y-sign in golfer forward kinematics, club Jacobians and the native grip_left wrapper; closure is a wrist-pin PLANAR_POINT2 with rank 2 on the manifold (TB-00 registry: 6 independent DOFs); planar-model and central-difference cross-checks
- **Next step:** Shipped in PR #12668.

### DL-#12558 · NSX-13: Snapshot_From_Adapter With CLOSED_LOOP_SNAPSHOT Capability (Pinocchio FULL), Canonical Closure in the Torque Allocator, constraintDynamics Parity

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12558
- **Branch:** merged via #12623
- **PR:** #12623
- **Paths:** see #12623
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`8c56e217`; collated from changes/12558-nsx-13-snapshot-from-adapter-with-closed.md)
- **Summary:** NSX-13: snapshot_from_adapter with CLOSED_LOOP_SNAPSHOT capability (Pinocchio FULL), canonical closure in the torque allocator, constraintDynamics parity
- **Next step:** Shipped in PR #12623.

### DL-#12556 · NSX-11 Typed Loop-Nullspace Registry Loader With CI Citation Gate

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12556
- **Branch:** merged via #12613
- **PR:** #12613
- **Paths:** see #12613
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`bd34dcce`; collated from changes/12556-nsx-11-typed-loop-nullspace-registry-loa.md)
- **Summary:** NSX-11 typed loop-nullspace registry loader with CI citation gate
- **Next step:** Shipped in PR #12613.

### DL-#12578 · NSX-50: Pose_Interchange Dual Velocity/Acceleration/Force Maps and Closure Wrench Duals

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12578
- **Branch:** merged via #12615
- **PR:** #12615
- **Paths:** see #12615
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`696113fc`; collated from changes/12578-nsx-50-pose-interchange-dual-velocity-ac.md)
- **Summary:** NSX-50: pose_interchange dual velocity/acceleration/force maps and closure wrench duals
- **Next step:** Shipped in PR #12615.

### DL-#12557 · NSX-12: Analytic Closed-Loop Oracle Fixtures (Planar Two-Arm Golfer, Straight Arm, Envelope, Spatial Pinocchio Chain) Reproducing NSX-00 Goldens

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12557
- **Branch:** merged via #12614
- **PR:** #12614
- **Paths:** see #12614
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`3a97285d`; collated from changes/12557-nsx-12-analytic-closed-loop-oracle-fixtu.md)
- **Summary:** NSX-12: analytic closed-loop oracle fixtures (planar two-arm golfer, straight arm, envelope, spatial Pinocchio chain) reproducing NSX-00 goldens
- **Next step:** Shipped in PR #12614.

### DL-#12555 · NSX-10: ClosedLoopSnapshot, LoopTopology and Evidence-Axis Contracts

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12555
- **Branch:** merged via #12612
- **PR:** #12612
- **Paths:** see #12612
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`3e0aa085`; collated from changes/12555-nsx-10-closedloopsnapshot-looptopology-a.md)
- **Summary:** NSX-10: ClosedLoopSnapshot, LoopTopology and evidence-axis contracts
- **Next step:** Shipped in PR #12612.

### DL-#12551 · NSX-02 Canonical Closure Jacobian Convention (Frame, Row Order, Sign) With Per-Engine Declared Conventions, Golden and Cross-Engine Tests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12551
- **Branch:** merged via #12610
- **PR:** #12610
- **Paths:** see #12610
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`a3442c7d`; collated from changes/12551-nsx-02-canonical-closure-jacobian-conven.md)
- **Summary:** NSX-02 canonical closure Jacobian convention (frame, row order, sign) with per-engine declared conventions, golden and cross-engine tests
- **Next step:** Shipped in PR #12610.

### DL-#12635 · Port MyoSuite Runtime Loading to a Version-Tolerant Shim (Env_Base Removed in 3.0.0) so the Myosuite Extras Stay Unpinned

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12635
- **Branch:** merged via #12650
- **PR:** #12650
- **Paths:** see #12650
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`96b6f29c`; collated from changes/12635-port-myosuite-runtime-loading-to-a-versi.md)
- **Summary:** Port MyoSuite runtime loading to a version-tolerant shim (env_base removed in 3.0.0) so the myosuite extras stay unpinned
- **Next step:** Shipped in PR #12650.

### DL-#12553 · NSX-04: Per-Slot Torque-Capacity Registry and Capacity_for Loader With Verified and Assumed Sources

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12553
- **Branch:** merged via #12611
- **PR:** #12611
- **Paths:** see #12611
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`c19af88d`; collated from changes/12553-nsx-04-per-slot-torque-capacity-registry.md)
- **Summary:** NSX-04: per-slot torque-capacity registry and capacity_for loader with verified and assumed sources
- **Next step:** Shipped in PR #12611.

### DL-#12550 · Hand-on-Club Grip Wrench Uses the Bare Club Inertia (Club_Solids_Partition); Legacy Composite Kept as Wrist_on_Hands_Plus_Club

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12550
- **Branch:** merged via #12603
- **PR:** #12603
- **Paths:** see #12603
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`a9f98220`; collated from changes/12550-hand-on-club-grip-wrench-uses-the-bare-c.md)
- **Summary:** Hand-on-club grip wrench uses the bare club inertia (club_solids_partition); legacy composite kept as wrist_on_hands_plus_club
- **Next step:** Shipped in PR #12603.

### DL-#12552 · Split force_nullspace.py Into a Force_Nullspace Package With Re-Export Shims and Public API Contract Test (NSX-03, No Behavior Change)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12552
- **Branch:** merged via #12595
- **PR:** #12595
- **Paths:** see #12595
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`979b12e0`; collated from changes/12552-split-force-nullspace-py-into-a-force-nu.md)
- **Summary:** Split force_nullspace.py into a force_nullspace package with re-export shims and public API contract test (NSX-03, no behavior change)
- **Next step:** Shipped in PR #12595.

### DL-#12544 · Agent-Context Views Rendered on Demand

- **State:** shipped
- **Owner:** claude
- **Issue:** #12544
- **Branch:** fix/12544-agent-context-views-generated
- **PR:** #12547
- **Paths:** .gitignore; .pre-commit-config.yaml; AGENTS.md; docs/agent_context; docs/index.md; docs/README.md; scripts/generate_docs_map.py; tests/agent_context/test_generated_views_wiring.py; tests/unit/ci/test_generate_docs_map.py
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`fc47c9a9`; collated from changes/12544-stop-committing-the-generated-agent-cont.md)
- **Summary:** Stop committing the generated agent-context views (README.md, index.html); render them on demand in CI and locally; reviews, catalog and contracts stay committed
- **Next step:** Shipped in PR #12547.

### DL-#12563 · NSX-20: Convex Solver Backends (Highspy LP Session With Basis Reuse, Clarabel SOCP), Loop-Nullspace Extra, Benchmark

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12563
- **Branch:** merged via #12597
- **PR:** #12597
- **Paths:** see #12597
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`fe6d354d`; collated from changes/12563-nsx-20-convex-solver-backends-highspy-lp.md)
- **Summary:** NSX-20: convex solver backends (highspy LP session with basis reuse, Clarabel SOCP), loop-nullspace extra, benchmark
- **Next step:** Shipped in PR #12597.

### DL-#12554 · Add NSX-05 Deferred-Validation Record DV-12554 and External Validation Protocol

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12554
- **Branch:** merged via #12596
- **PR:** #12596
- **Paths:** see #12596
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`c50ced25`; collated from changes/12554-add-nsx-05-deferred-validation-record-dv.md)
- **Summary:** Add NSX-05 deferred-validation record DV-12554 and external validation protocol
- **Next step:** Shipped in PR #12596.

### DL-#12549 · Add NSX-00 Mathematical Specification, SymPy Golden Fixtures, ADR-0054, Hypothesis Registry Seed, and Contract Appendix

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12549
- **Branch:** merged via #12596
- **PR:** #12596, #12598
- **Paths:** see #12596
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`f6b7efb7`; collated from changes/12549-nsx-00-loop-null-space-specification-sym.md)
- **Summary:** Add NSX-00 mathematical specification, SymPy golden fixtures, ADR-0054, hypothesis registry seed, and contract appendix
- **Next step:** Shipped in PR #12596.

### DL-#12536 · Complete BunkerShot3D Workbench Export Views, Save/Reload, VTK Stamped Export, and API Docs (#12536)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12536
- **Branch:** merged via #12542
- **PR:** #12542
- **Paths:** see #12542
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`477af06c`; collated from changes/12536-complete-bunkershot3d-workbench-export-v.md)
- **Summary:** Complete BunkerShot3D workbench export views, save/reload, VTK stamped export, and API docs (#12536)
- **Next step:** Shipped in PR #12542.

### DL-#12399 · Add Transactional Muscle Allocation Stepper, Fresh Controller-Off Replay, R2025b T01 Binding, and F10 Turnover Packet (#12399, #11898, #11921, #11784)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12399
- **Branch:** merged via #12538
- **PR:** #12538
- **Paths:** see #12538
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`b8bb030b`; collated from changes/12399-add-transactional-muscle-allocation-step.md)
- **Summary:** Add transactional muscle allocation stepper, fresh controller-off replay, R2025b T01 binding, and F10 turnover packet (#12399, #11898, #11921, #11784)
- **Next step:** Shipped in PR #12538.

### DL-#12398 · Add Frozen Research Replay Provenance Auditor and Source-Authority Amendment Ledger (#12398)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12398
- **Branch:** merged via #12538
- **PR:** #12538
- **Paths:** see #12538
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`b8bb030b`; collated from changes/12398-add-frozen-research-replay-provenance-au.md)
- **Summary:** Add frozen research replay provenance auditor and source-authority amendment ledger (#12398)
- **Next step:** Shipped in PR #12538.

### DL-#11750 · Add Five Additional Weightlifting Movement Specifications With Split Stance

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11750
- **Branch:** merged via #12538
- **PR:** #12538
- **Paths:** see #12538
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`b8bb030b`; collated from changes/11750-add-five-additional-weightlifting-moveme.md)
- **Summary:** Add five additional weightlifting movement specifications with split stance
- **Next step:** Shipped in PR #12538.

### DL-#11749 · \ Add Lifting Showcase Video Planner and Headless Smoke Renderer\

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11749
- **Branch:** merged via #12538
- **PR:** #12538
- **Paths:** see #12538
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`b8bb030b`; collated from changes/11749-add-lifting-showcase-video-planner-and-h.md)
- **Summary:** \ Add lifting showcase video planner and headless smoke renderer\
- **Next step:** Shipped in PR #12538.

### DL-#11746 · Add OpenSim Full-Body Musculoskeletal Lift Model Composer and Kinetics Analyzer

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11746
- **Branch:** merged via #12538
- **PR:** #12538
- **Paths:** see #12538
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`b8bb030b`; collated from changes/11746-add-opensim-full-body-musculoskeletal-li.md)
- **Summary:** Add OpenSim full-body musculoskeletal lift model composer and kinetics analyzer
- **Next step:** Shipped in PR #12538.

### DL-#11541 · Add MOSAIC-09 Leave-One-Trial-Out Cross-Validation Study and Update Model-Aware Matching Reference (#11541, #11532)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11541
- **Branch:** merged via #12538
- **PR:** #12538
- **Paths:** see #12538
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`b8bb030b`; collated from changes/11541-add-mosaic-09-leave-one-trial-out-cross.md)
- **Summary:** Add MOSAIC-09 leave-one-trial-out cross-validation study and update model-aware matching reference (#11541, #11532)
- **Next step:** Shipped in PR #12538.

### DL-#11195 · Qualify Spatial Carriage Damper Angular-Momentum Closure With Central-Force and Closed-Wrench Laws (#11195)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11195
- **Branch:** merged via #12538
- **PR:** #12538
- **Paths:** see #12538
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`b8bb030b`; collated from changes/11195-qualify-spatial-carriage-damper-angular.md)
- **Summary:** Qualify spatial carriage damper angular-momentum closure with central-force and closed-wrench laws (#11195)
- **Next step:** Shipped in PR #12538.

### DL-#11961 · Adopt the #11671 Foot Torsional-Friction Patch (0.05 m) in Canonical Club-Face Fixture Runs Against Ground Yaw Slip

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11961
- **Branch:** merged via #12532
- **PR:** #12532
- **Paths:** see #12532
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`8c71f972`; collated from changes/11961-adopt-the-11671-foot-torsional-friction.md)
- **Summary:** Adopt the #11671 foot torsional-friction patch (0.05 m) in canonical club-face fixture runs against ground yaw slip
- **Next step:** Shipped in PR #12532.

### DL-#11672 · Add Root-Yaw Target Tracking Support and Evaluate Finish-Window Root Yaw Feasibility for Shooting Fit

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11672
- **Branch:** merged via #12532
- **PR:** #12532
- **Paths:** see #12532
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`8c71f972`; collated from changes/11672-add-root-yaw-target-tracking-support-and.md)
- **Summary:** Add root-yaw target tracking support and evaluate finish-window root yaw feasibility for shooting fit
- **Next step:** Shipped in PR #12532.

### DL-#11731 · Add OSV-5 Golfer Realism Showcase Render Script, Headless Smoke Test, and Modelling Reference Updates (#11731, #11726)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11731
- **Branch:** merged via #12527
- **PR:** #12527
- **Paths:** see #12527
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`9aa4f2ea`; collated from changes/11731-add-osv-5-golfer-realism-showcase-render.md)
- **Summary:** Add OSV-5 golfer realism showcase render script, headless smoke test, and modelling reference updates (#11731, #11726)
- **Next step:** Shipped in PR #12527.

### DL-#11681 · Add MATLAB R2025b Simscape Mechanics Explorer 4-View Export Script and Deferred Catalog Entry (NV-8) (#11681, #11673)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11681
- **Branch:** merged via #12523
- **PR:** #12523
- **Paths:** see #12523
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`8aa6be83`; collated from changes/11681-add-matlab-r2025b-simscape-mechanics-exp.md)
- **Summary:** Add MATLAB R2025b Simscape Mechanics Explorer 4-view export script and deferred catalog entry (NV-8) (#11681, #11673)
- **Next step:** Shipped in PR #12523.

### DL-#11613 · Add MATLAB R2025b Simscape Same-Input Torque Bundle Replay Harness and Update Parity Reference (SIP P-8) (#11613, #11605)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11613
- **Branch:** merged via #12525
- **PR:** #12525
- **Paths:** see #12525
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`59d0e9b7`; collated from changes/11613-add-matlab-r2025b-simscape-same-input-to.md)
- **Summary:** Add MATLAB R2025b Simscape same-input torque bundle replay harness and update parity reference (SIP P-8) (#11613, #11605)
- **Next step:** Shipped in PR #12525.

### DL-#9700 · Add MockEngineQuarantineError, Assert_Not_Mock_Engine Guard, and GOLF_FORBID_MOCK_ENGINE Enforcement

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #9700
- **Branch:** merged via #12476
- **PR:** #12476
- **Paths:** see #12476
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`256b5859`; collated from changes/9700-add-mockenginequarantineerror-assert-not.md)
- **Summary:** Add MockEngineQuarantineError, assert_not_mock_engine guard, and GOLF_FORBID_MOCK_ENGINE enforcement
- **Next step:** Shipped in PR #12476.

### DL-#9547 · Add Fail-Closed Impact-Interval Export Schema and Mutation Tests for Residual Zeroing and Unfinished Contact

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #9547
- **Branch:** merged via #12466
- **PR:** #12466
- **Paths:** see #12466
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`6561420c`; collated from changes/9547-add-fail-closed-impact-interval-export-s.md)
- **Summary:** Add fail-closed impact-interval export schema and mutation tests for residual zeroing and unfinished contact
- **Next step:** Shipped in PR #12466.

### DL-#9417 · Extend Rust Optional Extra to All 10 Maturin Crates and Populate Deploy/README.md

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #9417
- **Branch:** merged via #12489
- **PR:** #12489
- **Paths:** see #12489
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`fe92399b`; collated from changes/9417-extend-rust-optional-extra-to-all-10-mat.md)
- **Summary:** Extend rust optional extra to all 10 Maturin crates and populate deploy/README.md
- **Next step:** Shipped in PR #12489.

### DL-#9406 · Add Per-Engine Optional Extras in Pyproject.Toml and Enforce Requirements-Tools.Txt Pin Contract

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #9406
- **Branch:** merged via #12467
- **PR:** #12467
- **Paths:** see #12467
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`471e2f65`; collated from changes/9406-add-per-engine-optional-extras-in-pyproj.md)
- **Summary:** Add per-engine optional extras in pyproject.toml and enforce requirements-tools.txt pin contract
- **Next step:** Shipped in PR #12467.

### DL-#2064 · Bootstrap Rustup With --No-Modify-Path Before Dtolnay/Rust-Toolchain so CI Stops Appending CARGO_HOME Env Lines to Runner Shell Profiles (Repository_Management#2064)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #2064
- **Branch:** merged via #12218
- **PR:** #12218
- **Paths:** see #12218
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ed91c702`; collated from changes/2064-bootstrap-rustup-with-no-modify-path-bef.md)
- **Summary:** Bootstrap rustup with --no-modify-path before dtolnay/rust-toolchain so CI stops appending CARGO_HOME env lines to runner shell profiles (Repository_Management#2064)
- **Next step:** Shipped in PR #12218.

### DL-#1917 · Sync the Agent-Identity Managed Block: The Owner's Personal Token Is an Approved Agent Identity

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #1917
- **Branch:** merged via #12264
- **PR:** #12264
- **Paths:** see #12264
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`7b40a7fa`; collated from changes/1917-sync-the-agent-identity-managed-block-th.md)
- **Summary:** Sync the agent-identity managed block: the owner's personal token is an approved agent identity
- **Next step:** Shipped in PR #12264.

### DL-#1893 · Re-Archive HANDOFF.md Under the 200-Line Target

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #1893
- **Branch:** merged via #12359
- **PR:** #12359
- **Paths:** see #12359
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`77b9e6ea`; collated from changes/1893-re-archive-handoff-md-under-the-200-line.md)
- **Summary:** Re-archive HANDOFF.md under the 200-line target
- **Next step:** Shipped in PR #12359.

### DL-#12510 · Accommodate Collated Change-Log Rows in SPEC Cap and Test_Spec_Archive

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12510
- **Branch:** merged via #12512
- **PR:** #12512
- **Paths:** see #12512
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`40cc066c`; collated from changes/12510-accommodate-collated-rows-in-spec-cap.md)
- **Summary:** accommodate collated change-log rows in SPEC cap and test_spec_archive
- **Next step:** Shipped in PR #12512.

### DL-#12504 · Pack-Audit Adapters Track the Packs: Recognise Leaf/Loop-Closure Hand_R Weld to the Bar (Shared Hand_Attachment Helper), Measure Grip Width at the Start Pose, Read Start GRF Through the Drake/Pinocchio Pack Contact APIs

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12504
- **Branch:** merged via #12509
- **PR:** #12509
- **Paths:** see #12509
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`b7661855`; collated from changes/12504-pack-audit-adapters-track-the-packs-reco.md)
- **Summary:** Pack-audit adapters track the packs: recognise leaf/loop-closure hand_r weld to the bar (shared hand_attachment helper), measure grip width at the start pose, read start GRF through the Drake/Pinocchio pack contact APIs
- **Next step:** Shipped in PR #12509.

### DL-#12503 · Regenerate the Grip-Parity OpenSim Reference Series and Engine Series From the Current Club-Face Fixtures, With a Regeneration Receipt; MuJoCo, Drake and Pinocchio Pass Again on Both Clubs

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12503
- **Branch:** merged via #12505
- **PR:** #12505
- **Paths:** see #12505
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`686b0c61`; collated from changes/12503-regenerate-the-grip-parity-opensim-refer.md)
- **Summary:** Regenerate the grip-parity OpenSim reference series and engine series from the current club-face fixtures, with a regeneration receipt; MuJoCo, Drake and Pinocchio pass again on both clubs
- **Next step:** Shipped in PR #12505.

### DL-#12492 · Collate-Changes Keeps Going Past a Refused Fragment and Fails Only After Proposing the Rest (Port of Repository_Management #2046)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12492
- **Branch:** merged via #12490
- **PR:** #12490, #12511
- **Paths:** see #12490
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`d13d5efc`; collated from changes/12492-re-sync-shared-scripts-collator-with-rep.md)
- **Summary:** collate-changes keeps going past a refused fragment and fails only after proposing the rest (port of Repository_Management #2046)
- **Next step:** Shipped in PR #12490.

### DL-#12473 · Remove Private Native Capture Files, Original Capture Exports and Club Workbooks From the Public Tree; Workbooks Resolve Only Through the Capture Registry (CAPTURE_DATA_DIR); Add a Hash-Denylist Hygiene Guard

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12473
- **Branch:** merged via #12474
- **PR:** #12474
- **Paths:** see #12474
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`18497550`; collated from changes/12473-move-private-captures-out-of-public-tree.md)
- **Summary:** Remove private native capture files, original capture exports and club workbooks from the public tree; workbooks resolve only through the capture registry (CAPTURE_DATA_DIR); add a hash-denylist hygiene guard
- **Next step:** Shipped in PR #12474.

### DL-#12435 · Unblock Collate Change Fragments: The MOSAIC-13 Multi-Trial #11545 Fragment No Longer Reopens Shipped DL-#11545 as in_Review; It Records Against the Shipped Entry

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12435
- **Branch:** claude/fix-dl-11545-shipped-fragments
- **PR:** #12436
- **Paths:** changes/11545-mosaic-13-closed-form-multi-trial-benchm.md
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`3bbd5d09`; collated from changes/12435-collate-shipped-dl-11545.md)
- **Summary:** Unblock Collate Change Fragments: the MOSAIC-13 multi-trial #11545 fragment no longer reopens shipped DL-#11545 as in_review; it records against the shipped entry
- **Next step:** Shipped in PR #12436.

### DL-#12422 · Make UI API Type Generation Deterministic: Share One GroupColumn Literal so ModelPayloadV2.Group_Column No Longer Flips Order by Import Order

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12422
- **Branch:** merged via #12423
- **PR:** #12423
- **Paths:** see #12423
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`16ad31e9`; collated from changes/12422-make-ui-api-type-generation-deterministi.md)
- **Summary:** Make UI API type generation deterministic: share one GroupColumn Literal so ModelPayloadV2.group_column no longer flips order by import order
- **Next step:** Shipped in PR #12423.

### DL-#12416 · Move Per-Slice AGENT_HANDOFF.md History Verbatim to Docs/Development/AGENT_HANDOFF_ARCHIVE.md; Live Handoff Is Current-State Only (96 Lines), the Expiring Doc-Size Exception Is Dropped.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12416
- **Branch:** merged via #12417
- **PR:** #12417
- **Paths:** see #12417
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`01c4da6f`; collated from changes/12416-move-per-slice-agent-handoff-md-history.md)
- **Summary:** Move per-slice AGENT_HANDOFF.md history verbatim to docs/development/AGENT_HANDOFF_ARCHIVE.md; live handoff is current-state only (96 lines), the expiring doc-size exception is dropped.
- **Next step:** Shipped in PR #12417.

### DL-#12409 · Shot-Pattern Article: 2,000-Resample Headline Statistics, Clubhead-Assumption Band (136 Runs), Generated Hash-Bound Tables in SG per 100 Shots and m (Yd); Shared Figure Palette With Reproducible PDF/SVG Metadata; Preset-Modification and Driver-Lean Cautions; Matched Physics Rejects an Ignored Nominal Face

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12409
- **Branch:** merged via #12418
- **PR:** #12418
- **Paths:** see #12418
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/12409-shot-pattern-study-hardening-fail-fast-s.md)
- **Summary:** Shot-pattern article: 2,000-resample headline statistics, clubhead-assumption band (136 runs), generated hash-bound tables in SG per 100 shots and m (yd); shared figure palette with reproducible PDF/SVG metadata; preset-modification and driver-lean cautions; matched physics rejects an ignored nominal face
- **Next step:** Shipped in PR #12418.

### DL-#12391 · Restore Development Log Capacity With Source-Backed DL-#10783 Shipped Metadata, the Official Zero-Day Archive Routine, and Two Narrowly Scoped Title-Case Corrections; Preserve All Other Metadata and Canonical-Function Exception Deadlines.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12391
- **Branch:** merged via #12396
- **PR:** #12396
- **Paths:** see #12396
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`8a22fd99`; collated from changes/12391-development-log-capacity-maintenance.md)
- **Summary:** Restore Development Log capacity with source-backed DL-#10783 shipped metadata, the official zero-day archive routine, and two narrowly scoped title-case corrections; preserve all other metadata and canonical-function exception deadlines.
- **Next step:** Shipped in PR #12396.

### DL-#12388 · Add Bounded Invocation-Local Muscle Path Observation Caching to Avoid Redundant Solves During Dense Finite-Difference Probes; Preserve Complete Allocation Results, Candidate Accounting, and Error Propagation.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12388
- **Branch:** merged via #12433
- **PR:** #12433
- **Paths:** see #12433
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ef7f1daf`; collated from changes/12388-bounded-muscle-path-caching.md)
- **Summary:** Add bounded invocation-local muscle path observation caching to avoid redundant solves during dense finite-difference probes; preserve complete allocation results, candidate accounting, and error propagation.
- **Next step:** Shipped in PR #12433.

### DL-#12387 · Expose a Shared Immutable Quasistatic Hill Force Snapshot for Allocator and Future Validated Replay Consumers; Preserve Allocator Behavior and Keep Model Qualification Unchanged.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12387
- **Branch:** merged via #12389
- **PR:** #12389
- **Paths:** see #12389
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`a014437c`; collated from changes/12387-expose-a-shared-immutable-quasistatic-hi.md)
- **Summary:** Expose a shared immutable quasistatic Hill force snapshot for allocator and future validated replay consumers; preserve allocator behavior and keep model qualification unchanged.
- **Next step:** Shipped in PR #12389.

### DL-#12347 · Back-Squat Bar Geometry Slice 1: Torso Bar Shelf, Standard Grip Width and Medio-Lateral Bar Axis From the Parity Standard (back_bar.py, Lift_to_Full_Body Geometry Option, Generate_Lift_Bundle --Geometry Back_Bar); Committed Bundle and Evidence Not Regenerated

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12347
- **Branch:** merged via #12488
- **PR:** #12488, #12534
- **Paths:** see #12488
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`aca30fca`; collated from changes/12347-back-squat-lift-body.md)
- **Summary:** Back-squat bar geometry slice 1: torso bar shelf, standard grip width and medio-lateral bar axis from the parity standard (back_bar.py, lift_to_full_body geometry option, generate_lift_bundle --geometry back_bar); committed bundle and evidence not regenerated
- **Next step:** Shipped in PR #12488.

### DL-#12331 · Add Pinocchio OpenSim and MATLAB 3D Adapters for Swing Report Parity

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12331
- **Branch:** merged via #12430
- **PR:** #12430
- **Paths:** see #12430
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`6685e235`; collated from changes/12331-add-pinocchio-opensim-and-matlab-3d-adap.md)
- **Summary:** Add Pinocchio OpenSim and MATLAB 3D adapters for swing report parity
- **Next step:** Shipped in PR #12430.

### DL-#12330 · Swing Report React/Tauri Parity: Web Full/Custom Dialog Over POST /Api/Swing-Report/View (PyQt ReportViewModel Rows) and Zip Package Export via POST /Api/Swing-Report/Export

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12330
- **Branch:** merged via #12421
- **PR:** #12421
- **Paths:** see #12421
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`fe7aa816`; collated from changes/12330-swing-report-web-parity.md)
- **Summary:** Swing Report React/Tauri parity: web Full/Custom dialog over POST /api/swing-report/view (PyQt ReportViewModel rows) and zip package export via POST /api/swing-report/export
- **Next step:** Shipped in PR #12421.

### DL-#12329 · Swing Report S6: PyQt Full/Custom Report Dialog and Qt-Free View-Model Over the Validated Report

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12329
- **Branch:** merged via #12406
- **PR:** #12406
- **Paths:** see #12406
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ded03ad8`; collated from changes/12329-swing-report-s6-pyqt-full-custom-report.md)
- **Summary:** Swing Report S6: PyQt Full/Custom report dialog and Qt-free view-model over the validated report
- **Next step:** Shipped in PR #12406.

### DL-#12328 · Swing Report S5: MuJoCo and Drake Adapters With Cross-Engine Speed Parity Harness

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12328
- **Branch:** merged via #12395
- **PR:** #12395
- **Paths:** see #12395
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`9d9e05d6`; collated from changes/12328-swing-report-s5-mujoco-and-drake-adapter.md)
- **Summary:** Swing report S5: MuJoCo and Drake adapters with cross-engine speed parity harness
- **Next step:** Shipped in PR #12395.

### DL-#12327 · Swing Report S4: Package Exporter and CLI (Report.Json, Series CSVs, Value-Only Plots, MANIFEST.Json Sha256s, Markdown Summary) With Fail-Closed Round-Trip Tests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12327
- **Branch:** merged via #12394
- **PR:** #12394
- **Paths:** see #12394
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`51f56377`; collated from changes/12327-swing-report-s4-package-exporter-and-cli.md)
- **Summary:** Swing report S4: package exporter and CLI (report.json, series CSVs, value-only plots, MANIFEST.json sha256s, markdown summary) with fail-closed round-trip tests
- **Next step:** Shipped in PR #12394.

### DL-#12326 · Swing Report S3: Plot Registry and Full/Custom Plot Selection With Unavailable Reasons

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12326
- **Branch:** merged via #12366
- **PR:** #12366
- **Paths:** see #12366
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`7e9e8d63`; collated from changes/12326-swing-report-s3-plot-registry-and-full-c.md)
- **Summary:** Swing report S3: plot registry and Full/Custom plot selection with unavailable reasons
- **Next step:** Shipped in PR #12366.

### DL-#12325 · Swing Report S2: Build_Swing_Report Assembles Runs, Quantities and Provenance From Engine Results Under the Fail-Closed #10382 Policy (Withheld Stubs Always Value-Free, Advertised Quantities Need Two-Engine Spread, Nulls Never Zero-Filled); Reuses the S1 Schema, Policy and Validator

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12325
- **Branch:** merged via #12354
- **PR:** #12354
- **Paths:** see #12354
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`c68d5a4c`; collated from changes/12325-swing-report-s2-build-swing-report-assem.md)
- **Summary:** Swing Report S2: build_swing_report assembles runs, quantities and provenance from engine results under the fail-closed #10382 policy (withheld stubs always value-free, advertised quantities need two-engine spread, nulls never zero-filled); reuses the S1 schema, policy and validator
- **Next step:** Shipped in PR #12354.

### DL-#12324 · Swing Report Contract V1 (S1): Schemas/Swing-Report-V1.Json, Quantity Policy Table, Fail-Closed Validate_Swing_Report With Named I1-I12 Errors, Valid/Invalid Fixtures

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12324
- **Branch:** merged via #12336
- **PR:** #12336
- **Paths:** see #12336
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`fb1f4183`; collated from changes/12324-swing-report-contract-v1-s1-schemas-swin.md)
- **Summary:** Swing report contract v1 (S1): schemas/swing-report-v1.json, quantity policy table, fail-closed validate_swing_report with named I1-I12 errors, valid/invalid fixtures
- **Next step:** Shipped in PR #12336.

### DL-#12320 · R05: Neural Promotion Gate Fails Closed on Missing, Zero, or Non-Comparable Evidence

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12320
- **Branch:** merged via #12333
- **PR:** #12333
- **Paths:** see #12333
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`713b5242`; collated from changes/12320-r05-neural-promotion-gate-fails-closed-o.md)
- **Summary:** R05: neural promotion gate fails closed on missing, zero, or non-comparable evidence
- **Next step:** Shipped in PR #12333.

### DL-#12308 · Frame the OpenSim Same-Input View From the Trajectory so Feet and Clubhead Stay in the Face-On Image

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12308
- **Branch:** merged via #12313
- **PR:** #12313, #12496
- **Paths:** see #12313
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`f56f18f3`; collated from changes/12308-msk-address-posture-and-impact-offset.md)
- **Summary:** Frame the OpenSim same-input view from the trajectory so feet and clubhead stay in the face-on image
- **Next step:** Shipped in PR #12313.

### DL-#12307 · OSV-9 MSK Turn_Feet Mode Drives Per-Frame Foot Targets (Heel and Toe Lift) From the Capture Foot Markers

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12307
- **Branch:** merged via #12311
- **PR:** #12311
- **Paths:** see #12311
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`b1edc27e`; collated from changes/12307-osv-9-msk-turn-feet-mode-drives-per-fram.md)
- **Summary:** OSV-9 MSK turn_feet mode drives per-frame foot targets (heel and toe lift) from the capture foot markers
- **Next step:** Shipped in PR #12311.

### DL-#12306 · OSV-9 MSK Grip: Specified Hand Grip Rotation (Trail Palm to Target, Lead Palm Away, Thumb Down the Shaft) Replaces the IK-Inherited Frame; Calibration, Models and MSK Renders Regenerated

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12306
- **Branch:** merged via #12314
- **PR:** #12314
- **Paths:** see #12314
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`14ba9306`; collated from changes/12306-osv-9-msk-grip-specified-hand-grip-rotat.md)
- **Summary:** OSV-9 MSK grip: specified hand grip rotation (trail palm to target, lead palm away, thumb down the shaft) replaces the IK-inherited frame; calibration, models and MSK renders regenerated
- **Next step:** Shipped in PR #12314.

### DL-#12304 · Harden CI Shot-Pattern Figure Restore: Strict Path Charset, Symlink/Alias/Duplicate ZIP Member Rejection, Size Caps, Atomic Non-Following Writes

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12304
- **Branch:** merged via #12446
- **PR:** #12446, #12418
- **Paths:** see #12446
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/12304-protect-revised-latex-sources-by-verifyi.md)
- **Summary:** Harden CI shot-pattern figure restore: strict path charset, symlink/alias/duplicate ZIP member rejection, size caps, atomic non-following writes
- **Next step:** Shipped in PR #12446.

### DL-#12303 · Batch Matplotlib Face Vertex Gathering to Preserve the 1 ms Frame Update Contract

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12303
- **Branch:** merged via #12418
- **PR:** #12418, #12305
- **Paths:** see #12418
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`e1429bde`; collated from changes/12303-vectorize-matplotlibrenderer-face-gather.md)
- **Summary:** Batch Matplotlib face vertex gathering to preserve the 1 ms frame update contract
- **Next step:** Shipped in PR #12418.

### DL-#12300 · Scale Native Hand Masses With Subject Mass When Match_Subject_Mass Is Enabled and Document Reference Capture Hand Mass Policy

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12300
- **Branch:** merged via #12486
- **PR:** #12486
- **Paths:** see #12486
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`67fd3188`; collated from changes/12300-scale-native-hand-masses-with-subject-ma.md)
- **Summary:** Scale native hand masses with subject mass when match_subject_mass is enabled and document reference capture hand mass policy
- **Next step:** Shipped in PR #12486.

### DL-#12286 · Realtime Soak Nightly Skips Cleanly Unless SOAK_RUNNER_AVAILABLE Is True

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12286
- **Branch:** merged via #12298
- **PR:** #12298
- **Paths:** see #12298
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`944d798e`; collated from changes/12286-realtime-soak-nightly-skips-cleanly-unle.md)
- **Summary:** Realtime soak nightly skips cleanly unless SOAK_RUNNER_AVAILABLE is true
- **Next step:** Shipped in PR #12298.

### DL-#12285 · Cross-Engine Equivalence Keeps One Running Plus the Newest Pending Main Run Instead of Queuing One per Push

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12285
- **Branch:** merged via #12296
- **PR:** #12296
- **Paths:** see #12296
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`06804d56`; collated from changes/12285-cross-engine-equivalence-keeps-one-runni.md)
- **Summary:** Cross-Engine Equivalence keeps one running plus the newest pending main run instead of queuing one per push
- **Next step:** Shipped in PR #12296.

### DL-#12279 · Add Explicit Effort Hold Rate Manifest Policy for Same-Input Bundles and Document 0.125 ms ZOH Findings

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12279
- **Branch:** merged via #12502
- **PR:** #12502, #12517
- **Paths:** see #12502
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`79a83078`; collated from changes/12279-regenerate-the-7-iron-club-face-fixture.md)
- **Summary:** Add explicit effort hold rate manifest policy for same-input bundles and document 0.125 ms ZOH findings
- **Next step:** Shipped in PR #12502.

### DL-#12272 · Record OpenSim Bushing Grip Ball-Impact Transient Bounds Decision 21 in DESIGN_DECISIONS_2.md

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12272
- **Branch:** merged via #12500
- **PR:** #12500
- **Paths:** see #12500
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`e8acf52b`; collated from changes/12272-record-bushing-impact-transient-bounds.md)
- **Summary:** Record OpenSim bushing grip ball-impact transient bounds decision 21 in DESIGN_DECISIONS_2.md
- **Next step:** Shipped in PR #12500.

### DL-#12267 · Retire the Drifted In-Tree Movement Optimizer Copy; the Legacy Src.Shared.Python.Movement_Optimizer Spelling Now Binds to the Pinned Tools Package With a DeprecationWarning, and the Maturin Parity Gate Builds the Pinned Crate.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12267
- **Branch:** merged via #12268
- **PR:** #12268
- **Paths:** see #12268
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ceffab6c`; collated from changes/12267-retire-the-drifted-in-tree-movement-opti.md)
- **Summary:** Retire the drifted in-tree Movement Optimizer copy; the legacy src.shared.python.movement_optimizer spelling now binds to the pinned Tools package with a DeprecationWarning, and the maturin parity gate builds the pinned crate.
- **Next step:** Shipped in PR #12268.

### DL-#12224 · CI: Heavy Cross-Engine Equivalence, Optional-Stack and PR Leaderboard-Artifact Lanes Run Nightly, on Dispatch or on the Ci:Full PR Label Instead of Every PR Push; Non-PR Cross-Engine Runs Are Never Cancelled and a Nightly Failure Files an Issue

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12224
- **Branch:** merged via #12226
- **PR:** #12226
- **Paths:** see #12226
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`d54e57c3`; collated from changes/12224-ci-heavy-cross-engine-equivalence-option.md)
- **Summary:** CI: heavy cross-engine equivalence, optional-stack and PR leaderboard-artifact lanes run nightly, on dispatch or on the ci:full PR label instead of every PR push; non-PR cross-engine runs are never cancelled and a nightly failure files an issue
- **Next step:** Shipped in PR #12226.

### DL-#12217 · Design Manual Index Integrity: Restore Lost Chapter 50, Include Orphaned Chapter 31, Drop Three Duplicate Includes, Renumber OpenSim Geometry Chapter 26 to 60, and Add a Ratchet Test Against New Collisions

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12217
- **Branch:** merged via #12293
- **PR:** #12293
- **Paths:** see #12293
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`40c319d0`; collated from changes/12217-design-manual-index-integrity-restore-lo.md)
- **Summary:** Design manual index integrity: restore lost chapter 50, include orphaned chapter 31, drop three duplicate includes, renumber OpenSim geometry chapter 26 to 60, and add a ratchet test against new collisions
- **Next step:** Shipped in PR #12293.

### DL-#12204 · Real-Engine Tests Skip on Conftest MagicMock Engines Through One Shared Tests.Support.Real_Engines.Real_Module Helper

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12204
- **Branch:** merged via #12295
- **PR:** #12295
- **Paths:** see #12295
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`522260f6`; collated from changes/12204-real-engine-tests-skip-on-conftest-magic.md)
- **Summary:** Real-engine tests skip on conftest MagicMock engines through one shared tests.support.real_engines.real_module helper
- **Next step:** Shipped in PR #12295.

### DL-#12199 · Character Builder Mass_kg Now Sets the Compiled Body Mass: The Rajagopal Legs Scale to Carry What the Upper Body Does Not.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12199
- **Branch:** merged via #12301
- **PR:** #12301
- **Paths:** see #12301
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`f0408393`; collated from changes/12199-character-builder-mass-kg-now-sets-the-c.md)
- **Summary:** Character Builder mass_kg now sets the compiled body mass: the Rajagopal legs scale to carry what the upper body does not.
- **Next step:** Shipped in PR #12301.

### DL-#12187 · Preserve Canonical Ground-Support Numerical Receipt Bytes Through Formatting Hooks

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12187
- **Branch:** fix/f10-inertia-principal-moments-12183
- **PR:** #12163
- **Paths:** .prettierignore, docs/development/full_body_models/HANDOFF.md
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5e5757ef`; collated from changes/12187-canonical-receipt-formatting.md)
- **Summary:** Preserve canonical ground-support numerical receipt bytes through formatting hooks
- **Next step:** Shipped in PR #12163.

### DL-#12183 · Reject Impossible Rotated Inertia Tensors and Scale-Masked Asymmetry

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12183
- **Branch:** fix/f10-inertia-principal-moments-12183
- **PR:** #12188
- **Paths:** src/shared/python/estimation/dime_global_calibration.py, manuals/upstreamdrift/chapters/51-physical-inertia-validation.qmd
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`c30934fc`; collated from changes/12183-physical-inertia-principal-moments.md)
- **Summary:** Reject impossible rotated inertia tensors and scale-masked asymmetry
- **Next step:** Shipped in PR #12188.

### DL-#12182 · Re-Land the Exact 557-Muscle Diagnostic Replay Profile (Native_Exact_557_Profile, V3 Opt-In) That Stack Consolidation Dropped From Main

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12182
- **Branch:** merged via #12361
- **PR:** #12361
- **Paths:** see #12361
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`f66a30ad`; collated from changes/12182-re-land-the-exact-557-muscle-diagnostic.md)
- **Summary:** Re-land the exact 557-muscle diagnostic replay profile (native_exact_557_profile, v3 opt-in) that stack consolidation dropped from main
- **Next step:** Shipped in PR #12361.

### DL-#12177 · Use the Src.Shared.Python Prefix for Ud-Canonical Model_Generation/Humanoid_Character_Builder Self-Imports so the Drake Path Works Without PYTHONPATH=Src

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12177
- **Branch:** merged via #12179
- **PR:** #12179
- **Paths:** see #12179
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5bb9f688`; collated from changes/12177-use-the-src-shared-python-prefix-for-ud.md)
- **Summary:** Use the src.shared.python prefix for ud-canonical model_generation/humanoid_character_builder self-imports so the Drake path works without PYTHONPATH=src
- **Next step:** Shipped in PR #12179.

### DL-#12169 · Pin the Paired-Figure Renderer Protocol Precondition: Protocol_Id-Only References Raise ValueError Before Any I/O or Figure Allocation for Every Selectable Kind

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12169
- **Branch:** merged via #12447
- **PR:** #12447, #12418
- **Paths:** see #12447
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/12169-reject-missing-pair-protocols-before-ren.md)
- **Summary:** Pin the paired-figure renderer protocol precondition: protocol_id-only references raise ValueError before any I/O or figure allocation for every selectable kind
- **Next step:** Shipped in PR #12447.

### DL-#12161 · Guard Partial BUET Abdominal Zero-Translation Reduction With Native Source and State Verification.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12161
- **Branch:** merged via #12171
- **PR:** #12171
- **Paths:** see #12171
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`f192ead7`; collated from changes/12161-native-abdominal-reduction.md)
- **Summary:** Guard partial BUET abdominal zero-translation reduction with native source and state verification.
- **Next step:** Shipped in PR #12171.

### DL-#12157 · Compose Exact BUET Upper and Hamner Distal Native OpenSim Sources With Fresh-Reload Sampled Mechanical Checks

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12157
- **Branch:** feat/f07-buet-hamner-assembly-12157
- **PR:** #12165
- **Paths:** src/engines/physics_engines/opensim/python/native_buet_hamner_assembly.py, manuals/upstreamdrift/chapters/44-native-buet-hamner-assembly.qmd
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`fa7ee92f`; collated from changes/12157-native-buet-hamner-assembly.md)
- **Summary:** Compose exact BUET upper and Hamner distal native OpenSim sources with fresh-reload sampled mechanical checks
- **Next step:** Shipped in PR #12165.

### DL-#12151 · Add Assistance-Explicit OpenSim Scalar Replay With Native Force and Work Accounting

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12151
- **Branch:** merged via #12152
- **PR:** #12152
- **Paths:** see #12152
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5482cf8c`; collated from changes/12151-opensim-mixed-actuation-replay.md)
- **Summary:** Add assistance-explicit OpenSim scalar replay with native force and work accounting
- **Next step:** Shipped in PR #12152.

### DL-#12150 · Add a Source-Bound Native Zero-MTP Reduction With Exact Rajagopal and Scaled-Club Mechanical Checks

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12150
- **Branch:** feat/f07-rajagopal-mtp-reduction-12150
- **PR:** #12153
- **Paths:** src/engines/physics_engines/opensim/python/native_mtp_reduction.py, manuals/upstreamdrift/chapters/43-native-zero-mtp-reduction.qmd
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`9df5191b`; collated from changes/12150-native-zero-mtp-reduction.md)
- **Summary:** Add a source-bound native zero-MTP reduction with exact Rajagopal and scaled-club mechanical checks
- **Next step:** Shipped in PR #12153.

### DL-#12147 · Link Assistance-Explicit Moco Controls to Independent Native Replay

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12147
- **Branch:** merged via #12163
- **PR:** #12163
- **Paths:** see #12163
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5e5757ef`; collated from changes/12147-link-mixed-moco-to-native-replay.md)
- **Summary:** Link Assistance-Explicit Moco Controls to Independent Native Replay
- **Next step:** Shipped in PR #12163.

### DL-#12139 · Prove Paired-Figure Stems Are Unique and Unchanged Over the Saved 384-Comparison Inventory and Cover Full-Precision Target, Delimiter and Fractional-Label Collision Cases

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12139
- **Branch:** merged via #12447
- **PR:** #12447, #12418
- **Paths:** see #12447
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/12139-shot-pattern-protocol-figure-identity.md)
- **Summary:** Prove paired-figure stems are unique and unchanged over the saved 384-comparison inventory and cover full-precision target, delimiter and fractional-label collision cases
- **Next step:** Shipped in PR #12447.

### DL-#12137 · Fix 11 Motion-Matching Unit Failures: DrakeFullBodyIK Reads Spec Offset_m and Keeps Every Spec-Referenced Body in Pose_Fn; Event-Alignment and Drake Fit Tests Repointed to the Current Reader and Dispatch Seams

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12137
- **Branch:** merged via #12138
- **PR:** #12138
- **Paths:** see #12138
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`fa5eef0e`; collated from changes/12137-fix-11-motion-matching-unit-failures-dra.md)
- **Summary:** Fix 11 motion-matching unit failures: DrakeFullBodyIK reads spec offset_m and keeps every spec-referenced body in pose_fn; event-alignment and Drake fit tests repointed to the current reader and dispatch seams
- **Next step:** Shipped in PR #12138.

### DL-#12109 · Hip Rewrite Uses the Spec's Own Pelvis Alignment; Anthro Hips No Longer Start 24-66 Deg Yawed, Hip_Rotation Leaves Its +-40 Deg Limits and Both Feet Meet the 2 Deg Toe-Out Target at Address

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12109
- **Branch:** merged via #12145
- **PR:** #12145
- **Paths:** see #12145
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`3a63d61c`; collated from changes/12109-hip-rewrite-uses-the-spec-s-own-pelvis-a.md)
- **Summary:** Hip rewrite uses the spec's own pelvis alignment; anthro hips no longer start 24-66 deg yawed, hip_rotation leaves its +-40 deg limits and both feet meet the 2 deg toe-out target at address
- **Next step:** Shipped in PR #12145.

### DL-#12073 · LIFT-1 Drake Pack-Audit Adapter Reports Segment Inertia About the COM, Not Bo; Regenerate Parity Baseline

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12073
- **Branch:** merged via #12105
- **PR:** #12105
- **Paths:** see #12105
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`2dc710d2`; collated from changes/12073-lift-1-drake-pack-audit-adapter-reports.md)
- **Summary:** LIFT-1 Drake pack-audit adapter reports segment inertia about the COM, not Bo; regenerate parity baseline
- **Next step:** Shipped in PR #12105.

### DL-#12065 · Add Shared Ezc3d Import Guard That Diagnoses the OpenSim/Ezc3d DLL Conflict With an Explicit NativeDllConflictError and Process-Boundary Guidance

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12065
- **Branch:** merged via #12350
- **PR:** #12350, #12454, #12437, #12451, #12470, #12499, #12475, #12365, #12482, #12513
- **Paths:** see #12350
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`1019e19f`; collated from changes/12065-route-the-remaining-dynamic-opensim-load.md)
- **Summary:** Add shared ezc3d import guard that diagnoses the OpenSim/ezc3d DLL conflict with an explicit NativeDllConflictError and process-boundary guidance
- **Next step:** Shipped in PR #12350.

### DL-#12060 · Restore Shared Native Feedback Admission After Platform Integration

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12060
- **Branch:** merged via #12141
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/12060-native-feedback-platform-integration.md)
- **Summary:** Restore shared native feedback admission after platform integration
- **Next step:** Shipped in PR #12141.

### DL-#12057 · Right Knee Flexes Negative Like the Left (Spec Builder + Committed Specs); Canonical Ground-Support Receipts Regenerated: Calibrated IK Improves, Dynamics Replay Regresses (#12110), Trail Toe-Out Misses 2 Deg (#12109)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12057
- **Branch:** merged via #12113
- **PR:** #12113
- **Paths:** see #12113
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`d2070536`; collated from changes/12057-right-knee-flexes-negative-like-the-left.md)
- **Summary:** Right knee flexes negative like the left (spec builder + committed specs); canonical ground-support receipts regenerated: calibrated IK improves, dynamics replay regresses (#12110), trail toe-out misses 2 deg (#12109)
- **Next step:** Shipped in PR #12113.

### DL-#12055 · Native Prediction Verifies Complete Live/Source/Model/Handle/Callback Identity After Bundle Preparation Before Stepping, Closing a Late Handle-Mutation Gap, and the Execution Validator Reuses That Verification, Cutting Repeated Library-Byte Reads by 25 Percent in Controlled Short Native Fixtures; Full Matching and Parity Acceptance Remain Open

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12055
- **Branch:** merged via #12141
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/12055-prediction-integrity-boundaries.md)
- **Summary:** Native prediction verifies complete live/source/model/handle/callback identity after bundle preparation before stepping, closing a late handle-mutation gap, and the execution validator reuses that verification, cutting repeated library-byte reads by 25 percent in controlled short native fixtures; full matching and parity acceptance remain open
- **Next step:** Shipped in PR #12141.

### DL-#12046 · Move the Expired Realtime Shadow-Ledger Entry to #12046 With a 30-Day Sunset to Unblock the Merge Queue

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12046
- **Branch:** merged via #12048
- **PR:** #12048
- **Paths:** see #12048
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`d50b9317`; collated from changes/12046-move-the-expired-realtime-shadow-ledger.md)
- **Summary:** Move the expired realtime shadow-ledger entry to #12046 with a 30-day sunset to unblock the merge queue
- **Next step:** Shipped in PR #12048.

### DL-#12044 · Integrate Admitted Tools/Private Replay Contracts Into Native Command Solving

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12044
- **Branch:** merged via #12141
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/12044-native-tools-integration.md)
- **Summary:** Integrate admitted Tools/private replay contracts into native command solving
- **Next step:** Shipped in PR #12141.

### DL-#12042 · Commit the Golf_Humanoid and Golf_Humanoid_Scaled 7-Iron MSK Grip Calibrations, so the OpenSim MSK Retarget and Render Run for the 7-Iron

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12042
- **Branch:** merged via #12248
- **PR:** #12248, #12108, #12145, #12078, #12116, #12166, #12247
- **Paths:** see #12248
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`8ed0f40a`; collated from changes/12042-the-osv-9-msk-club-render-can-track-with.md)
- **Summary:** Commit the golf_humanoid and golf_humanoid_scaled 7-iron MSK grip calibrations, so the OpenSim MSK retarget and render run for the 7-iron
- **Next step:** Shipped in PR #12248.

### DL-#12036 · Connect Native Tracking to Shared Bounded Command Search With Independently Replayed Fallback and Guarded Promotion.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12036
- **Branch:** merged via #12141
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/12036-native-command-solve.md)
- **Summary:** Connect native tracking to shared bounded command search with independently replayed fallback and guarded promotion.
- **Next step:** Shipped in PR #12141.

### DL-#12018 · Keep F09 Replay Contracts on One Private Tools Module Identity

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12018
- **Branch:** merged via #12141
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/12018-f09-private-tools-contract-consumers.md)
- **Summary:** Keep F09 replay contracts on one private Tools module identity
- **Next step:** Shipped in PR #12141.

### DL-#12016 · Align Tools Dependency Pins to Merged Main

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12016
- **Branch:** merged via #12141
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/12016-align-tools-dependency-pins-to-merged-ma.md)
- **Summary:** Align Tools dependency pins to merged main
- **Next step:** Shipped in PR #12141.

### DL-#12013 · Parity Harness Records the Producing Engine of Each Baseline and Rejects a MuJoCo-Produced Baseline Labelled Simscape (Fail-Closed); Real Simscape Regeneration Remains Deferred

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12013
- **Branch:** merged via #12349
- **PR:** #12349
- **Paths:** see #12349
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`4bca4ce7`; collated from changes/12013-parity-harness-records-the-producing-eng.md)
- **Summary:** Parity harness records the producing engine of each baseline and rejects a MuJoCo-produced baseline labelled Simscape (fail-closed); real Simscape regeneration remains deferred
- **Next step:** Shipped in PR #12349.

### DL-#12006 · Fix Reversed Toe/Heel Sign in Legacy Compute_Gear_Effect_Spin; Toe Now Curves Left (Draw) for Right-Handed +X Shots

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12006
- **Branch:** merged via #12346
- **PR:** #12346
- **Paths:** see #12346
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`e4b3dfea`; collated from changes/12006-fix-reversed-toe-heel-sign-in-legacy-com.md)
- **Summary:** Fix reversed toe/heel sign in legacy compute_gear_effect_spin; toe now curves left (draw) for right-handed +X shots
- **Next step:** Shipped in PR #12346.

### DL-#11997 · MyoSuite Native Export Worker No Longer Imports the Removed Myosuite.Envs.Env_Base: It Builds a Raw Mujoco Model, Takes MJRenderer Through Myosuite_Compat (3.X Then 2.X), and Unavailable_Reason() Probes That Same Import

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11997
- **Branch:** merged via #12348
- **PR:** #12348, #12140
- **Paths:** see #12348
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`e68baeab`; collated from changes/11997-myosuite-native-export-worker-supports-m.md)
- **Summary:** MyoSuite native export worker no longer imports the removed myosuite.envs.env_base: it builds a raw mujoco model, takes MJRenderer through myosuite_compat (3.x then 2.x), and unavailable_reason() probes that same import
- **Next step:** Shipped in PR #12348.

### DL-#11995 · Authenticate Physics API Endpoint Tests via Scoped Dependency Override and Add Explicit 401 Contract Coverage

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11995
- **Branch:** merged via #12321
- **PR:** #12321
- **Paths:** see #12321
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`7e14eeeb`; collated from changes/11995-authenticate-physics-api-endpoint-tests.md)
- **Summary:** Authenticate physics API endpoint tests via scoped dependency override and add explicit 401 contract coverage
- **Next step:** Shipped in PR #12321.

### DL-#11994 · Load the Pinned Tools Mocap Package Under a Private Name so Native Torque Replay No Longer Grafts It Onto Sidekick.Lab, Fixing the Order-Dependent Vendored-Fallback Test

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11994
- **Branch:** merged via #12002
- **PR:** #12002
- **Paths:** see #12002
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`d3b2da07`; collated from changes/11994-load-the-pinned-tools-mocap-package-unde.md)
- **Summary:** Load the pinned Tools mocap package under a private name so native torque replay no longer grafts it onto sidekick.lab, fixing the order-dependent vendored-fallback test
- **Next step:** Shipped in PR #12002.

### DL-#11990 · Materialize a Source-Bound Numerical Moco Guess on the Frozen Observation Clock and Reject Incomplete Native Guess State Tables.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11990
- **Branch:** merged via #12141
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11990-native-moco-numerical-guess.md)
- **Summary:** Materialize a source-bound numerical Moco guess on the frozen observation clock and reject incomplete native guess state tables.
- **Next step:** Shipped in PR #12141.

### DL-#11987 · Add LaunchMonitorReportsPanel and V2 Report/Export API Client Functions for Web Parity

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11987
- **Branch:** merged via #12479
- **PR:** #12479, #12508, #12076, #12130, #12155, #12168, #12063, #12001, #12061, #12112, #12028, #12037, #12118, #12019, #12072, #12054, #12140, #12079, #12008, #12106, #12025, #12469
- **Paths:** see #12479
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`a2e56098`; collated from changes/11987-wire-matchedswings-native-viewer-handoff.md)
- **Summary:** Add LaunchMonitorReportsPanel and v2 report/export API client functions for web parity
- **Next step:** Shipped in PR #12479.

### DL-#11986 · Explain the 4X Per-Hand Contact-Grip Force: Input Drift Is Zero, Internal Force Is a Hyperstatic Ring Redundant (Diagnostics and Evidence)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11986
- **Branch:** merged via #12022
- **PR:** #12022
- **Paths:** see #12022
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`43385ba6`; collated from changes/11986-explain-the-4x-per-hand-contact-grip-for.md)
- **Summary:** Explain the 4x per-hand contact-grip force: input drift is zero, internal force is a hyperstatic ring redundant (diagnostics and evidence)
- **Next step:** Shipped in PR #12022.

### DL-#11982 · Reconcile Twelve Stale Feature-Parity Gap Entries That Cited Ten Closed Issues: Eight Flipped to Parity With Verified Api/Web Paths, Frankenstein Canvas Re-Pointed to Epic #11651, Three Real Web Gaps Re-Pointed to New Tracker #11987

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11982
- **Branch:** merged via #11981
- **PR:** #11981
- **Paths:** see #11981
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`c2952e1d`; collated from changes/11982-reconcile-ten-stale-feature-parity-gap-e.md)
- **Summary:** Reconcile twelve stale feature-parity gap entries that cited ten closed issues: eight flipped to parity with verified api/web paths, frankenstein canvas re-pointed to epic #11651, three real web gaps re-pointed to new tracker #11987
- **Next step:** Shipped in PR #11981.

### DL-#11968 · Native Offline Muscle Matching Runner

- **State:** shipped
- **Owner:** codex
- **Issue:** #11968
- **Branch:** feat/f07-native-offline-matching-11968
- **PR:** #12141
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11968-native-offline-moco-runner.md)
- **Summary:** Add file-driven native OpenSim Moco preparation, exact-knot excitation export, independent replay and masked scoring
- **Next step:** Shipped in PR #12141.

### DL-#11958 · Add an Activation-Aware Native Floating-Root Manifold and Complete-State Command Replay Boundary.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11958
- **Branch:** feat/f05e-activation-manifold-11958
- **PR:** #11816
- **Paths:** see #11816
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`1fe02740`; collated from changes/11958-native-activation-manifold.md)
- **Summary:** Add an activation-aware native floating-root manifold and complete-state command replay boundary.
- **Next step:** Shipped in PR #11816.

### DL-#11952 · Exercise Bounded F04 Feedback Tuning and Paired Motor Interventions on a Native Floating-Root Fixture.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11952
- **Branch:** feat/f04-native-coupling-11788
- **PR:** #11816
- **Paths:** see #11816
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`1fe02740`; collated from changes/11952-native-coupling.md)
- **Summary:** Exercise bounded F04 feedback tuning and paired motor interventions on a native floating-root fixture.
- **Next step:** Shipped in PR #11816.

### DL-#11948 · Fit Exact-Clock Masked Native Site Markers With Bounded Manifold Control and Complete-State Replay.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11948
- **Branch:** feat/f03-native-marker-fit-11787
- **PR:** #11816
- **Paths:** see #11816
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`1fe02740`; collated from changes/11948-native-marker-fit.md)
- **Summary:** Fit exact-clock masked native site markers with bounded manifold control and complete-state replay.
- **Next step:** Shipped in PR #11816.

### DL-#11946 · Connect F02 Feedback to Native Floating-Root Manifold State and Independently Replay Post-Limit Torques.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11946
- **Branch:** feat/f02-native-manifold-11946
- **PR:** #11816
- **Paths:** see #11816
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`1fe02740`; collated from changes/11946-native-manifold-feedback.md)
- **Summary:** Connect F02 feedback to native floating-root manifold state and independently replay post-limit torques.
- **Next step:** Shipped in PR #11816.

### DL-#11945 · Label the MyoSuite Inventory MJCF Compile Probe as Bare MuJoCo Evidence and Reject It as Gym-Provider Availability

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11945
- **Branch:** merged via #12360
- **PR:** #12360
- **Paths:** see #12360
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ba57e270`; collated from changes/11945-label-the-myosuite-inventory-mjcf-compil.md)
- **Summary:** Label the MyoSuite inventory MJCF compile probe as bare MuJoCo evidence and reject it as Gym-provider availability
- **Next step:** Shipped in PR #12360.

### DL-#11940 · Regenerate Stale Articulated-Ground Post-Hoc Sensitivity Provenance (Atlas SHA256) With Its Generator; Add Regenerate-Equality Test

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11940
- **Branch:** merged via #12323
- **PR:** #12323
- **Paths:** see #12323
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`439ee710`; collated from changes/11940-regenerate-stale-articulated-ground-post.md)
- **Summary:** Regenerate stale articulated-ground post-hoc sensitivity provenance (atlas SHA256) with its generator; add regenerate-equality test
- **Next step:** Shipped in PR #12323.

### DL-#11934 · Fix(Drake): Prevent Offscreen Widget Abort During Induced Source Combo Clear

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11934
- **Branch:** merged via #12441
- **PR:** #12441
- **Paths:** see #12441
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`54121413`; collated from changes/11934-fix-drake-prevent-offscreen-widget-abort.md)
- **Summary:** fix(drake): prevent offscreen widget abort during induced source combo clear
- **Next step:** Shipped in PR #12441.

### DL-#11932 · Add a Bounded Native Multi-DOF Manifold BoxFDDP Candidate With Complete-State Torque Replay and a Matched SciPy Comparison.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11932
- **Branch:** feat/f05d-multidof-boxfddp-11932
- **PR:** #11816
- **Paths:** see #11816
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`1fe02740`; collated from changes/11932-native-multidof-box-fddp.md)
- **Summary:** Add a bounded native multi-DOF manifold BoxFDDP candidate with complete-state torque replay and a matched SciPy comparison.
- **Next step:** Shipped in PR #11816.

### DL-#11920 · Add a Strict Floating-Root, Two-Hinge Native MuJoCo Tangent-Derivative Fixture With Ordered Bounded Motors, Quaternion-Aware Finite-Difference Checks and Independent Full-State Replay of Frozen ZOH Torques, Recording Source-Hashed Supported-Provider Timing; Optimal-Control, Contact, Muscle and Capture Qualification Remain Open

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11920
- **Branch:** merged via #11816
- **PR:** #11816
- **Paths:** see #11816
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`1fe02740`; collated from changes/11920-native-manifold-derivative-replay.md)
- **Summary:** Add a strict floating-root, two-hinge native MuJoCo tangent-derivative fixture with ordered bounded motors, quaternion-aware finite-difference checks and independent full-state replay of frozen ZOH torques, recording source-hashed supported-provider timing; optimal-control, contact, muscle and capture qualification remain open
- **Next step:** Shipped in PR #11816.

### DL-#11918 · Add an F09 Native MyoSuite Muscle-Excitation Consumer for Versioned T01 Bundles That Applies Exact Normalized Excitation Directly to Native MuJoCo Controls With Full State Restore and Strict Provider/Model Identity, and Share Wrapper-State Capture and Restoration in One Chain-Walk Implementation; the Elbow Fixture Remains Test-Only and the Six-Engine Parity Denominator Stays Unqualified

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11918
- **Branch:** merged via #12141
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11918-f09f-myosuite-native-excitation.md)
- **Summary:** Add an F09 native MyoSuite muscle-excitation consumer for versioned T01 bundles that applies exact normalized excitation directly to native MuJoCo controls with full state restore and strict provider/model identity, and share wrapper-state capture and restoration in one chain-walk implementation; the elbow fixture remains test-only and the six-engine parity denominator stays unqualified
- **Next step:** Shipped in PR #12141.

### DL-#11914 · F09e Adds Pinocchio Native Replay Outputs to the F09 Marker Forward-Kinematics Path Using the Actual Full-State Q/V Trajectory and Explicit Frame-Local Offsets, Preserving All Registered Rows and Six Required Engines; Native Receipts Remain Unqualified and MyoSuite, OpenSim and Simscape Gaps Stay Visible

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11914
- **Branch:** merged via #12141
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11914-f09e-pinocchio-marker-fk.md)
- **Summary:** F09e adds Pinocchio native replay outputs to the F09 marker forward-kinematics path using the actual full-state q/v trajectory and explicit frame-local offsets, preserving all registered rows and six required engines; native receipts remain unqualified and MyoSuite, OpenSim and Simscape gaps stay visible
- **Next step:** Shipped in PR #12141.

### DL-#11910 · Add a Strict Native RK4 One-Hinge BoxFDDP Candidate With Verified Discrete Derivatives, Loaded-Model Identity, Bounded Post-Solve Admission and Fail-Closed Fallback, Compared With SciPy Shooting on Predeclared Supported-MuJoCo Trials Using Exact Applied-Torque Replay and Total Measured Time; F05 Full-Body and Capture Qualification Remain Open

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11910
- **Branch:** merged via #11816
- **PR:** #11816
- **Paths:** see #11816
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`1fe02740`; collated from changes/11910-native-box-fddp-derivative-candidate.md)
- **Summary:** Add a strict native RK4 one-hinge BoxFDDP candidate with verified discrete derivatives, loaded-model identity, bounded post-solve admission and fail-closed fallback, compared with SciPy shooting on predeclared supported-MuJoCo trials using exact applied-torque replay and total measured time; F05 full-body and capture qualification remain open
- **Next step:** Shipped in PR #11816.

### DL-#11907 · Add a Fail-Closed Native Replay to Marker-Position Bridge for MuJoCo and Drake With Explicit Ordered Body/Frame Attachments, Local Offsets, Source/Model/Provider Identities and Output Hashes; Unsupported Engines Stay in the Required Denominator and No Qualification Is Inferred

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11907
- **Branch:** merged via #12141
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11907-f09d-native-marker-fk.md)
- **Summary:** Add a fail-closed native replay to marker-position bridge for MuJoCo and Drake with explicit ordered body/frame attachments, local offsets, source/model/provider identities and output hashes; unsupported engines stay in the required denominator and no qualification is inferred
- **Next step:** Shipped in PR #12141.

### DL-#11904 · Add Bounded Robust Shooting NMPC With Fail-Closed Actuator Fallback, Native MuJoCo Step Execution and Independent Frozen-Torque Replay, Benchmarked Against F02 TVLQR on Two Predeclared Perturbed One-Hinge Trials; TVLQR Is Retained for Native-Step Operation Because the Shooting Prototype Cannot Meet the 10 ms Step Budget, and F05 Full-Swing and Capture Qualification Remain Open

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11904
- **Branch:** merged via #11816
- **PR:** #11816
- **Paths:** see #11816
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`1fe02740`; collated from changes/11904-native-bounded-nmpc.md)
- **Summary:** Add bounded robust shooting NMPC with fail-closed actuator fallback, native MuJoCo step execution and independent frozen-torque replay, benchmarked against F02 TVLQR on two predeclared perturbed one-hinge trials; TVLQR is retained for native-step operation because the shooting prototype cannot meet the 10 ms step budget, and F05 full-swing and capture qualification remain open
- **Next step:** Shipped in PR #11816.

### DL-#11899 · Add Frozen Marker Scoring Without Fitting Withheld Observations

- **State:** shipped
- **Owner:** codex
- **Issue:** #11899
- **Branch:** fix/feedback-frozen-holdout-11899
- **PR:** #11902
- **Paths:** see #11902
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`009eb282`; collated from changes/11899-frozen-marker-holdout.md)
- **Summary:** Add frozen marker scoring without fitting withheld observations
- **Next step:** Review scoped scoring boundary; keep private D02 and full F07/F08 qualification open

### DL-#11898 · Instrument Native Moco Solve Progress and Solver-Stage Identity Without Changing Model, Request, Physics, or Solver Policy

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11898
- **Branch:** merged via #12318
- **PR:** #12318, #12269, #12141
- **Paths:** see #12318
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`9410bb19`; collated from changes/11898-same-input-lazy-exports.md)
- **Summary:** Instrument native Moco solve progress and solver-stage identity without changing model, request, physics, or solver policy
- **Next step:** Shipped in PR #12318.

### DL-#11859 · Add Bounded Coupled F02 Loop Tuning, Frozen-Controller Sensitivity and Honest Heldout/Perturbation Reports on a Two-Coordinate Synthetic Forward Plant

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11859
- **Branch:** merged via #11816
- **PR:** #11816
- **Paths:** see #11816
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`1fe02740`; collated from changes/11859-coupled-control-loop-tuning.md)
- **Summary:** Add bounded coupled F02 loop tuning, frozen-controller sensitivity and honest heldout/perturbation reports on a two-coordinate synthetic forward plant
- **Next step:** Shipped in PR #11816.

### DL-#11854 · Bind Actual Initial-State Payload Identity in Versioned Independent Replay and Same-Input Comparison Admission; Retain Legacy Identity/Transcription Receipts

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11854
- **Branch:** merged via #12141
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11854-initial-state-replay-binding.md)
- **Summary:** Bind actual initial-state payload identity in versioned independent replay and same-input comparison admission; retain legacy identity/transcription receipts
- **Next step:** Shipped in PR #12141.

### DL-#11847 · Bound Impact Acceptance Claims to the Tested Central-Contact Model

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11847
- **Branch:** merged via #12357
- **PR:** #12357
- **Paths:** see #12357
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`20a661f2`; collated from changes/11847-bound-impact-acceptance-claims-to-the-te.md)
- **Summary:** Bound impact acceptance claims to the tested central-contact model
- **Next step:** Shipped in PR #12357.

### DL-#11828 · F01 Docs: Replay Admission Requires Full_Horizon and a Separate Observation_Time_Grid_Sha256 (Contract Repair From PR #12141, Not a Numerical Qualification)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11828
- **Branch:** merged via #12356
- **PR:** #12356, #12141
- **Paths:** see #12356
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11828-replay-horizon-observation-clock.md)
- **Summary:** F01 docs: replay admission requires full_horizon and a separate observation_time_grid_sha256 (contract repair from PR #12141, not a numerical qualification)
- **Next step:** Shipped in PR #12356.

### DL-#11824 · Add a Bounded Native MuJoCo Candidate Benchmark for the Synthetic Rotary Optimal-Control Fixture, With Independent Exact Held-Input Truth, a Fresh F06 Replay, Fail-Closed Torque/Slew and Numerical Gates, and Serial Failure-Retaining Timing and Receipt Evidence

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11824
- **Branch:** merged via #11816
- **PR:** #11816
- **Paths:** see #11816
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`1fe02740`; collated from changes/11824-native-candidate-benchmark.md)
- **Summary:** Add a bounded native MuJoCo candidate benchmark for the synthetic rotary optimal-control fixture, with independent exact held-input truth, a fresh F06 replay, fail-closed torque/slew and numerical gates, and serial failure-retaining timing and receipt evidence
- **Next step:** Shipped in PR #11816.

### DL-#11818 · Raise Canonical Golfer Root to 0.98 m so the Feet Rest on the Floor; Regenerate MJCF/Simscape, Update Inventory Hashes

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11818
- **Branch:** merged via #12351
- **PR:** #12351
- **Paths:** see #12351
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`e710cd41`; collated from changes/11818-raise-canonical-golfer-root-to-0-98-m-so.md)
- **Summary:** Raise canonical golfer root to 0.98 m so the feet rest on the floor; regenerate MJCF/Simscape, update inventory hashes
- **Next step:** Shipped in PR #12351.

### DL-#11811 · Generated MuJoCo Golfer: Segment Geoms Use Contype 2 / Conaffinity 1 so They Touch the Floor (And a Ball) but Never Each Other; Golfer.Xml Regenerated and Its Inventory Hash Updated; the Four-Engine MuJoCo Parity Tests Are Strict-Xfail Because Their 'Simscape' Baseline Is a Rollout of the Self-Colliding MuJoCo Golfer (#12013)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11811
- **Branch:** merged via #11817
- **PR:** #11817
- **Paths:** see #11817
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`924cdc77`; collated from changes/11811-golfer-mjcf-self-collision.md)
- **Summary:** Generated MuJoCo golfer: segment geoms use contype 2 / conaffinity 1 so they touch the floor (and a ball) but never each other; golfer.xml regenerated and its inventory hash updated; the four-engine MuJoCo parity tests are strict-xfail because their 'Simscape' baseline is a rollout of the self-colliding MuJoCo golfer (#12013)
- **Next step:** Shipped in PR #11817.

### DL-#11796 · OpenSim FK No Longer Calls the Missing State.isValid Binding; an Uninitialised State Is Rejected by Its Empty Subsystem List

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11796
- **Branch:** merged via #11833
- **PR:** #11833
- **Paths:** see #11833
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`3effe852`; collated from changes/11796-opensim-fk-no-longer-calls-the-missing-s.md)
- **Summary:** OpenSim FK no longer calls the missing State.isValid binding; an uninitialised state is rejected by its empty subsystem list
- **Next step:** Shipped in PR #11833.

### DL-#11778 · Simscape Grip Fixture Now Comes From the Qualified Run-102 Replay; Model-Workspace Coefficients Diagnosed as Out of Bounds (Max 1.2E10) and Divergent

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11778
- **Branch:** merged via #11783
- **PR:** #11783
- **Paths:** see #11783
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`44f09127`; collated from changes/11778-simscape-grip-fixture-now-comes-from-the.md)
- **Summary:** Simscape grip fixture now comes from the qualified run-102 replay; model-workspace coefficients diagnosed as out of bounds (max 1.2e10) and divergent
- **Next step:** Shipped in PR #11783.

### DL-#11777 · Prettier Ignores the Generated Matched-Swing Ledger and the Budget-Held DESIGN_DECISIONS.md so the Hook No Longer Fights Test_Ledger_Freshness and the Doc Size Budget

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11777
- **Branch:** merged via #11805
- **PR:** #11805
- **Paths:** see #11805
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`8f4a2d73`; collated from changes/11777-prettier-ignores-the-generated-matched-s.md)
- **Summary:** Prettier ignores the generated matched-swing ledger and the budget-held DESIGN_DECISIONS.md so the hook no longer fights test_ledger_freshness and the doc size budget
- **Next step:** Shipped in PR #11805.

### DL-#11767 · Clubhead Speed Timing Around Impact

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11767
- **Branch:** claude/gcv-20-speed-timing
- **PR:** #12145
- **Paths:** see #12145
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`3a63d61c`; collated from changes/11767-gcv-20-clubhead-speed-timing.md)
- **Summary:** GCV-20: ball contact impulse in every replay, release-preserving pre-contact cutoff and grip-weld projection of the tracked reference; peak clubhead-speed timing now within 5 ms of the capture, impact speed still 4-7 % low (strict xfail with measured numbers)
- **Next step:** Shipped in PR #12145.

### DL-#11759 · Club-Face Orientation Residual in the Shared IK

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11759
- **Branch:** claude/osv-10-face-roll-refit
- **PR:** #11772
- **Paths:** see #11772
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`509c6933`; collated from changes/11759-osv-10-face-roll-refit.md)
- **Summary:** OSV-10: shared marker IK gains an explicit club-face orientation residual (weight 3) from the capture head triad; club-face fixtures regenerated by script with provenance; each engine's face tracks the capture within 5 degrees at address, top and impact
- **Next step:** Shipped in PR #11772.

### DL-#11756 · OSV-9: Rajagopal Golf Models and the Muscle Model Hold the Shared Club in Both Hands (Lead Weld + Trail Constraint or Bushings), Calibrated to the Captured Address; Swing Tracking and Offscreen Renders

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11756
- **Branch:** merged via #11795
- **PR:** #11795, #12145
- **Paths:** see #11795
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`3a63d61c`; collated from changes/11756-osv-9-slice-2-tour-matching-variants-rea.md)
- **Summary:** OSV-9: Rajagopal golf models and the muscle model hold the shared club in both hands (lead weld + trail constraint or bushings), calibrated to the captured address; swing tracking and offscreen renders
- **Next step:** Shipped in PR #11795.

### DL-#11755 · Square the Clubface at Address in Every Engine: One Shared Head Roll About the Shaft (OSV-8)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11755
- **Branch:** merged via #11772
- **PR:** #11772
- **Paths:** src/shared/python/model_appearance/club_assembly.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`509c6933`; collated from changes/11755-square-the-clubface-at-address-in-every.md)
- **Summary:** Square the clubface at address in every engine: one shared head roll about the shaft (OSV-8)
- **Next step:** Shipped in PR #11772.

### DL-#11753 · Scrub Capture-System Identifiers From Docs, Research Checkpoints and Exploratory MATLAB Comments per Private Data Policy

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11753
- **Branch:** merged via #11754
- **PR:** #11754
- **Paths:** see #11754
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5275f770`; collated from changes/11753-scrub-capture-system-identifiers.md)
- **Summary:** Scrub capture-system identifiers from docs, research checkpoints and exploratory MATLAB comments per private data policy
- **Next step:** Shipped in PR #11754.

### DL-#11751 · Design Manual: LIFT-4 Static-Hold Per-Hand Bar Wrench Section in Chapter 15 and Its Registry Blocker

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11751
- **Branch:** merged via #12215
- **PR:** #12215, #11978, #11835, #12010
- **Paths:** see #12215
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`f781a7e1`; collated from changes/11751-standalone-latex-research-reference-for.md)
- **Summary:** Design manual: LIFT-4 static-hold per-hand bar wrench section in chapter 15 and its registry blocker
- **Next step:** Shipped in PR #12215.

### DL-#11748 · PyQt Cross-Engine Lift Baseline Panel in the Exercise Dashboard

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11748
- **Branch:** merged via #12140
- **PR:** #12140, #12070, #11971, #12075
- **Paths:** see #12140
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`bb5ee162`; collated from changes/11748-web-lift-baseline-page-tools-lift-baseli.md)
- **Summary:** PyQt cross-engine lift baseline panel in the Exercise Dashboard
- **Next step:** Shipped in PR #12140.

### DL-#11747 · LIFT-7 Slice: Back-Squat GRF Breakdown (Per Foot, Net, CoP, Free Moment) and Per-Hand Grip Wrench Parity Across MuJoCo, Drake, Pinocchio and OpenSim From the LIFT-5 Bundle; KINETICS_PARITY.md Report

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11747
- **Branch:** merged via #12341
- **PR:** #12341
- **Paths:** see #12341
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`e76b1f50`; collated from changes/11747-lift-7-slice-back-squat-grf-breakdown-pe.md)
- **Summary:** LIFT-7 slice: back-squat GRF breakdown (per foot, net, CoP, free moment) and per-hand grip wrench parity across MuJoCo, Drake, Pinocchio and OpenSim from the LIFT-5 bundle; KINETICS_PARITY.md report
- **Next step:** Shipped in PR #12341.

### DL-#11744 · LIFT-4: Barbell Held at Both Hands Through the OSV-7 Compliant Bushing on the Back-Squat Same-Input Bundle, With Per-Hand Bushing Wrench, a Recorded Quasi-Static Hold and Four-Engine Parity

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11744
- **Branch:** merged via #12345
- **PR:** #12345, #12189
- **Paths:** see #12345
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`7d05b4f1`; collated from changes/11744-lift-4-per-hand-bar-wrench-static-hold-f.md)
- **Summary:** LIFT-4: barbell held at both hands through the OSV-7 compliant bushing on the back-squat same-input bundle, with per-hand bushing wrench, a recorded quasi-static hold and four-engine parity
- **Next step:** Shipped in PR #12345.

### DL-#11743 · LIFT-3: Per-Engine and Per-Lift Start-Pose Conformance Checks (Bar, Feet, Closure, Joint Limits, Static GRF) With a Recorded Matrix

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11743
- **Branch:** merged via #12241
- **PR:** #12241, #12477, #12498, #12507
- **Paths:** see #12241
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`656f16f2`; collated from changes/11743-lift-3-slice-3-re-measure-the-lift-pack.md)
- **Summary:** LIFT-3: per-engine and per-lift start-pose conformance checks (bar, feet, closure, joint limits, static GRF) with a recorded matrix
- **Next step:** Shipped in PR #12241.

### DL-#11741 · Add Lift Model Pack Audit and Cross-Engine Parity Baseline (LIFT-1): Baseline Doc, Receipt, Audit Package, Stills Script

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11741
- **Branch:** merged via #11771
- **PR:** #11771
- **Paths:** see #11771
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`071eb55b`; collated from changes/11741-add-lift-model-pack-audit-and-cross-engi.md)
- **Summary:** Add lift model pack audit and cross-engine parity baseline (LIFT-1): baseline doc, receipt, audit package, stills script
- **Next step:** Shipped in PR #11771.

### DL-#11739 · OpenSim Contact Grip (ElasticFoundationForce on Closed Meshes): Driver and 7-Iron Full Swings, Net Force Within 3 Percent of Bushing

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11739
- **Branch:** merged via #12022
- **PR:** #12022, #12119, #11774, #11963, #11985, #11772
- **Paths:** see #12022
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`509c6933`; collated from changes/11739-osv7-bushing-grip.md)
- **Summary:** OpenSim contact grip (ElasticFoundationForce on closed meshes): driver and 7-iron full swings, net force within 3 percent of bushing
- **Next step:** Shipped in PR #12022.

### DL-#11730 · OSV-4: Address Foot Progression (Toe-Out) Measured From Markers and Fitted in the Shared Address Seed, Explicit --Foot-Progression Option With Receipt Fields

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11730
- **Branch:** merged via #11736
- **PR:** #11736, #12175
- **Paths:** see #11736
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ee8920e5`; collated from changes/11730-osv-4-final-evidence-per-engine-address.md)
- **Summary:** OSV-4: address foot progression (toe-out) measured from markers and fitted in the shared address seed, explicit --foot-progression option with receipt fields
- **Next step:** Shipped in PR #11736.

### DL-#11728 · OSV-2 Item 4: Hand-on-Grip Orientation (Palm Diagonal, Palm Facing) Moves From the OpenSim Calibration Into the Shared Grip_Pose Definition; No Per-Engine Grip Angle

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11728
- **Branch:** merged via #12439
- **PR:** #12439, #12145, #12240, #12339
- **Paths:** see #12439
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`88b2d9eb`; collated from changes/11728-osv-2-slice-2b-both-curled-finger-grip-h.md)
- **Summary:** OSV-2 item 4: hand-on-grip orientation (palm diagonal, palm facing) moves from the OpenSim calibration into the shared grip_pose definition; no per-engine grip angle
- **Next step:** Shipped in PR #12439.

### DL-#11727 · Visible Shaft, Grip and Head Meshes in Every Generated OpenSim Model and the Tour-Matching Golf Humanoid

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11727
- **Branch:** merged via #11772
- **PR:** #11772
- **Paths:** see #11772
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`509c6933`; collated from changes/11727-visible-shaft-grip-and-head-meshes-in-ev.md)
- **Summary:** Visible shaft, grip and head meshes in every generated OpenSim model and the tour-matching golf humanoid
- **Next step:** Shipped in PR #11772.

### DL-#11725 · Add Scripts/render_forces_showcase.py: Reproducible Owner Showcase Driver (Clean, Club Forces, GRF Clips) Over Native_Viewer_Export, With Manifest and Smoke Test

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11725
- **Branch:** merged via #12237
- **PR:** #12237, #12287, #12292, #12312
- **Paths:** see #12237
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`3e683941`; collated from changes/11725-gcv-19-myosuite-evidence.md)
- **Summary:** Add scripts/render_forces_showcase.py: reproducible owner showcase driver (clean, club forces, GRF clips) over native_viewer_export, with manifest and smoke test
- **Next step:** Shipped in PR #12237.

### DL-#11724 · GCV-18 Slice 1: Provisional Design-Manual Chapter and Registry Blocker for the GCV-1 Ground-Reaction Breakdown; User Manual 12.4/12.7 Corrected (CoP and Free-Moment Equations, Metrics Not Computed)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11724
- **Branch:** merged via #11782
- **PR:** #11782, #11801, #11803, #11804
- **Paths:** see #11782
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`d2d0e1c4`; collated from changes/11724-gcv-18-slice-4-adr-0052-grip-derived-lab.md)
- **Summary:** GCV-18 slice 1: provisional design-manual chapter and registry blocker for the GCV-1 ground-reaction breakdown; user manual 12.4/12.7 corrected (CoP and free-moment equations, metrics not computed)
- **Next step:** Shipped in PR #11782.

### DL-#11723 · Impact Parameters Panel: API Route, PyQt Dock and Web Panel With Parity Registration (GCV-17)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11723
- **Branch:** merged via #11772
- **PR:** #11772
- **Paths:** see #11772
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`509c6933`; collated from changes/11723-impact-parameters-panel-api-route-pyqt-d.md)
- **Summary:** Impact parameters panel: API route, PyQt dock and web panel with parity registration (GCV-17)
- **Next step:** Shipped in PR #11772.

### DL-#11722 · ClubheadSeries Adapters for MuJoCo, MyoSuite, Drake, Pinocchio, OpenSim and Simscape With Cross-Engine Parity (GCV-16)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11722
- **Branch:** merged via #11772
- **PR:** #11772, #11970
- **Paths:** see #11772
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`c15a386b`; collated from changes/11722-run-the-gcv-16-clubhead-series-adapter-a.md)
- **Summary:** ClubheadSeries adapters for MuJoCo, MyoSuite, Drake, Pinocchio, OpenSim and Simscape with cross-engine parity (GCV-16)
- **Next step:** Shipped in PR #11772.

### DL-#11721 · GCV-15: Add Impact_Parameters Package (Speed, AoA, Path, Face, Face-to-Path, Dynamic/Spin Loft, Low Point) With Fail-Closed Tools Delivery Gateway

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11721
- **Branch:** merged via #11735
- **PR:** #11735, #11861
- **Paths:** see #11735
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`18123597`; collated from changes/11721-gcv-15-engine-free-adapter-contract-test.md)
- **Summary:** GCV-15: add impact_parameters package (speed, AoA, path, face, face-to-path, dynamic/spin loft, low point) with fail-closed Tools delivery gateway
- **Next step:** Shipped in PR #11735.

### DL-#11720 · GCV-14: Full-, Half- and Quarter-Speed (1X, 0.5X, 0.25X at 60 Fps) Are Now the Default Native Export Variants, and Motion_Matching.Export.Export_Video_Variants Writes the Same Three Clips for Any Engine

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11720
- **Branch:** merged via #12026
- **PR:** #12026, #11734, #12124, #12004, #12043, #12133
- **Paths:** see #12026
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`eeefefbc`; collated from changes/11720-playbacktiming-groups-render-playback-ti.md)
- **Summary:** GCV-14: full-, half- and quarter-speed (1x, 0.5x, 0.25x at 60 fps) are now the default native export variants, and motion_matching.export.export_video_variants writes the same three clips for any engine
- **Next step:** Shipped in PR #12026.

### DL-#11718 · GCV-12: Visible Parametric Head, Face and Neck on the Spec Head Body in MuJoCo, Drake, Pinocchio, MyoSuite and OpenSim Renders; Visual-Only Gaze Orientation Channel

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11718
- **Branch:** merged via #11772
- **PR:** #11772
- **Paths:** see #11772
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`509c6933`; collated from changes/11718-gcv-12-visible-parametric-head-face-and.md)
- **Summary:** GCV-12: visible parametric head, face and neck on the spec Head body in MuJoCo, Drake, Pinocchio, MyoSuite and OpenSim renders; visual-only gaze orientation channel
- **Next step:** Shipped in PR #11772.

### DL-#11717 · Realistic Parametric Clubhead Meshes in the MuJoCo, OpenSim, Drake, Pinocchio and MyoSuite Visual Layers via One Shared Adapter, With Committed STL Fallback Assets

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11717
- **Branch:** merged via #11772
- **PR:** #11772, #12140
- **Paths:** see #11772
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`e68baeab`; collated from changes/11717-web-clubhead-stl-heads-at-spec-loft-lie.md)
- **Summary:** Realistic parametric clubhead meshes in the MuJoCo, OpenSim, Drake, Pinocchio and MyoSuite visual layers via one shared adapter, with committed STL fallback assets
- **Next step:** Shipped in PR #11772.

### DL-#11716 · GCV-10 Club Force and Torque Overlays and Plots: Per-Hand, Net and Couple Glyphs, Grip Plot Route, PyQt Tile and Web Charts, Hands Close-Up Camera, Grip Export in Native Viewer

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11716
- **Branch:** merged via #11775
- **PR:** #11775
- **Paths:** see #11775
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`c9fd0bed`; collated from changes/11716-gcv-10-club-force-and-torque-overlays-an.md)
- **Summary:** GCV-10 club force and torque overlays and plots: per-hand, net and couple glyphs, grip plot route, PyQt tile and web charts, hands close-up camera, grip export in native viewer
- **Next step:** Shipped in PR #11775.

### DL-#11715 · Simscape Per-Hand Grip Wrench Channels (Grip:Hand_Left/Right) and an R2025b Grip Fixture With a Reconstruction Receipt (GCV-9)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11715
- **Branch:** merged via #11779
- **PR:** #11779
- **Paths:** see #11779
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`50a68358`; collated from changes/11715-simscape-per-hand-grip-wrench-channels-g.md)
- **Summary:** Simscape per-hand grip wrench channels (grip:hand_left/right) and an R2025b grip fixture with a reconstruction receipt (GCV-9)
- **Next step:** Shipped in PR #11779.

### DL-#11714 · GCV-8 Gaps: Grip-Modelling Tab Reuses Grip_Efc, Drake URDF Right-Hand Grip Wrench With Left Unavailable, Pinocchio Lambda_Grip Routed Through Grip_From_Allocation

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11714
- **Branch:** merged via #12035
- **PR:** #12035, #11757
- **Paths:** see #12035
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`f13bad3b`; collated from changes/11714-gcv-8-per-hand-grip-wrench-extraction-fo.md)
- **Summary:** GCV-8 gaps: grip-modelling tab reuses grip_efc, Drake URDF right-hand grip wrench with left unavailable, Pinocchio lambda_grip routed through grip_from_allocation
- **Next step:** Shipped in PR #12035.

### DL-#11713 · Add Shared Grip Wrench Core (Per-Hand Wrench, Midpoint Net Force and Couple, Contact-Moment/Free-Torque Split) for GCV-7

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11713
- **Branch:** merged via #11732
- **PR:** #11732
- **Paths:** see #11732
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ca8bab43`; collated from changes/11713-add-shared-grip-wrench-core-per-hand-wre.md)
- **Summary:** Add shared grip wrench core (per-hand wrench, midpoint net force and couple, contact-moment/free-torque split) for GCV-7
- **Next step:** Shipped in PR #11732.

### DL-#11712 · Route Every Centre-of-Pressure Computation Through the Shared Ground-Reaction Core; Legacy Helpers Become Deprecation Shims (GCV-6)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11712
- **Branch:** merged via #11802
- **PR:** #11802
- **Paths:** see #11802
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`b7d4189c`; collated from changes/11712-route-every-centre-of-pressure-computati.md)
- **Summary:** Route every centre-of-pressure computation through the shared ground-reaction core; legacy helpers become deprecation shims (GCV-6)
- **Next step:** Shipped in PR #11802.

### DL-#11711 · GCV-5 Ground-Reaction Plots: Shared Plot Series, GET /Api/Analysis/Ground-Reaction, PyQt Ground Reaction Plots Tile and Web Charts With Per-Foot Force, Load Share, CoP Path, Free Moment and Moment About CoM

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11711
- **Branch:** merged via #11781
- **PR:** #11781, #11802
- **Paths:** see #11781
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`b7d4189c`; collated from changes/11711-gcv-5-mujoco-run-ground-reaction-series.md)
- **Summary:** GCV-5 ground-reaction plots: shared plot series, GET /api/analysis/ground-reaction, PyQt Ground Reaction Plots tile and web charts with per-foot force, load share, CoP path, free moment and moment about CoM
- **Next step:** Shipped in PR #11781.

### DL-#11710 · Commit the Canonical Driver Per-Frame GRF Series and Assert the Native-Export Arrows Never Clamp (GCV-4)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11710
- **Branch:** merged via #12180
- **PR:** #12180, #11981, #11772, #11969
- **Paths:** see #12180
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec337448`; collated from changes/11710-matplotlib-and-qpainter-force-renderers.md)
- **Summary:** Commit the canonical driver per-frame GRF series and assert the native-export arrows never clamp (GCV-4)
- **Next step:** Shipped in PR #12180.

### DL-#11709 · Define 3-Tier Mathematical Engine Equivalency Contract and Verify Simscape Unavailable Contact Channel

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11709
- **Branch:** merged via #12468
- **PR:** #12468, #11780, #12483
- **Paths:** see #12468
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0829f00f`; collated from changes/11709-simscape-tier-registry.md)
- **Summary:** Define 3-tier mathematical engine equivalency contract and verify Simscape unavailable contact channel
- **Next step:** Shipped in PR #12468.

### DL-#11708 · GCV-2 Ground-Reaction Parity Test Loads Each Engine Module Once (Module-Scoped Breakdown): Re-Executing the Drake/Pinocchio Test Modules Broke Their Native Imports, so Only the First Drake/Pinocchio Row Ran

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11708
- **Branch:** merged via #12050
- **PR:** #12050, #11991, #11802
- **Paths:** see #12050
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`b7d4189c`; collated from changes/11708-wire-the-shared-ground-reaction-core-int.md)
- **Summary:** GCV-2 ground-reaction parity test loads each engine module once (module-scoped breakdown): re-executing the Drake/Pinocchio test modules broke their native imports, so only the first Drake/Pinocchio row ran
- **Next step:** Shipped in PR #12050.

### DL-#11707 · GCV-1: Shared Ground-Reaction Analysis Core (Per-Foot and Net GRF, CoP, Free Moment, Moments About CoM); Bundle Provider Consumes It

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11707
- **Branch:** merged via #11772
- **PR:** #11772
- **Paths:** see #11772
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`509c6933`; collated from changes/11707-gcv-1-shared-ground-reaction-analysis-co.md)
- **Summary:** GCV-1: shared ground-reaction analysis core (per-foot and net GRF, CoP, free moment, moments about CoM); bundle provider consumes it
- **Next step:** Shipped in PR #11772.

### DL-#11697 · NV-9: Shared Viewer FOV (0.7 rad) for MeshCat Exports, Projected-Extent and Fitted-Distance Framing Helpers

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11697
- **Branch:** merged via #11813
- **PR:** #11813
- **Paths:** see #11813
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`548a068f`; collated from changes/11697-nv-9-shared-viewer-fov-0-7-rad-for-meshc.md)
- **Summary:** NV-9: shared viewer FOV (0.7 rad) for MeshCat exports, projected-extent and fitted-distance framing helpers
- **Next step:** Shipped in PR #11813.

### DL-#11689 · MyoFullBody Reserve Attribution, Share IK and Torque Neck; Reserves Still Above 10 Percent (Not Qualified)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11689
- **Branch:** merged via #11702
- **PR:** #11702
- **Paths:** see #11702
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b607519`; collated from changes/11689-myofullbody-reserve-attribution-share-ik.md)
- **Summary:** MyoFullBody reserve attribution, share IK and torque neck; reserves still above 10 percent (not qualified)
- **Next step:** Shipped in PR #11702.

### DL-#11679 · Labelled 2X2 Multi-View Compositor and Clip Writer for Native Viewer Export

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11679
- **Branch:** merged via #11702
- **PR:** #11702
- **Paths:** see #11702
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b607519`; collated from changes/11679-labelled-2x2-multi-view-compositor-and-c.md)
- **Summary:** Labelled 2x2 multi-view compositor and clip writer for native viewer export
- **Next step:** Shipped in PR #11702.

### DL-#11678 · Native Viewer Export CLI With Drake MeshCat, Pinocchio MeshCat, OpenSim Simbody (Xvfb) and MyoSuite Backends

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11678
- **Branch:** merged via #11702
- **PR:** #11702
- **Paths:** see #11702
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b607519`; collated from changes/11678-native-viewer-export-cli-with-drake-mesh.md)
- **Summary:** Native viewer export CLI with Drake MeshCat, Pinocchio MeshCat, OpenSim simbody (xvfb) and MyoSuite backends
- **Next step:** Shipped in PR #11702.

### DL-#11677 · MeshCat Overlays for the Drake and Pinocchio Native Exports; Fix Drake Sink MeshcatCone Argument Order

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11677
- **Branch:** merged via #11702
- **PR:** #11702
- **Paths:** see #11702
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b607519`; collated from changes/11677-meshcat-overlays-for-the-drake-and-pinoc.md)
- **Summary:** MeshCat overlays for the Drake and Pinocchio native exports; fix Drake sink MeshcatCone argument order
- **Next step:** Shipped in PR #11702.

### DL-#11676 · Add Engine-Agnostic Bundle Overlay Frame Provider (Joint Torque Arcs, Per-Foot GRF at the CoP, Weight at the CoM) and a MuJoCo Overlay Source

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11676
- **Branch:** merged via #11702
- **PR:** #11702
- **Paths:** see #11702
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b607519`; collated from changes/11676-add-engine-agnostic-bundle-overlay-frame.md)
- **Summary:** Add engine-agnostic bundle overlay frame provider (joint torque arcs, per-foot GRF at the CoP, weight at the CoM) and a MuJoCo overlay source
- **Next step:** Shipped in PR #11702.

### DL-#11675 · Initialise MjvGeom Glyph Slots With Mjv_initGeom so Renderer.Render No Longer Segfaults With Arrows or Arcs

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11675
- **Branch:** merged via #11702
- **PR:** #11702
- **Paths:** see #11702
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b607519`; collated from changes/11675-initialise-mjvgeom-glyph-slots-with-mjv.md)
- **Summary:** Initialise MjvGeom glyph slots with mjv_initGeom so Renderer.render no longer segfaults with arrows or arcs
- **Next step:** Shipped in PR #11702.

### DL-#11674 · Add Shared Golf Camera View Presets (Face-On, Down-the-Line, Overhead, Oblique) With MuJoCo, Drake MeshCat, MeshCat and Simbody Camera Adapters

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11674
- **Branch:** merged via #11702
- **PR:** #11702
- **Paths:** see #11702
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b607519`; collated from changes/11674-add-shared-golf-camera-view-presets-face.md)
- **Summary:** Add shared golf camera view presets (face-on, down-the-line, overhead, oblique) with MuJoCo, Drake MeshCat, MeshCat and simbody camera adapters
- **Next step:** Shipped in PR #11702.

### DL-#11671 · Foot Contact: Torsional Friction (Opt-In --Torsional-Patch-m) and Foot-Width Spheres (Opt-In --Foot-Half-Width-m)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11671
- **Branch:** merged via #11772
- **PR:** #11772
- **Paths:** see #11772
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`509c6933`; collated from changes/11671-foot-contact-torsional-friction-opt-in-t.md)
- **Summary:** Foot contact: torsional friction (opt-in --torsional-patch-m) and foot-width spheres (opt-in --foot-half-width-m)
- **Next step:** Shipped in PR #11772.

### DL-#11670 · Contact-Wrench Tracking QP as Opt-In --Tracking Wrench-Qp; Negative Result Recorded

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11670
- **Branch:** merged via #11772
- **PR:** #11772
- **Paths:** see #11772
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`509c6933`; collated from changes/11670-contact-wrench-tracking-qp-as-opt-in-tra.md)
- **Summary:** Contact-wrench tracking QP as opt-in --tracking wrench-qp; negative result recorded
- **Next step:** Shipped in PR #11772.

### DL-#11669 · Add Centroidal Feasibility Filter V2 (Full ZMP, Vertical Force and Friction Cone Over the Finish) as the Opt-In --Centroidal-Filter Pipeline Stage

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11669
- **Branch:** merged via #11772
- **PR:** #11772
- **Paths:** see #11772
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`509c6933`; collated from changes/11669-add-centroidal-feasibility-filter-v2-ful.md)
- **Summary:** Add centroidal feasibility filter v2 (full ZMP, vertical force and friction cone over the finish) as the opt-in --centroidal-filter pipeline stage
- **Next step:** Shipped in PR #11772.

### DL-#11668 · Add Finish-Feasibility Metrics (ZMP-Inside Fraction, Friction-Cone Utilisation, Foot Slide and Yaw, Pelvis Yaw Error, Vertical Force Range) to the Dynamics Receipt, With a Post-Processing CLI and a Regression Ratchet

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11668
- **Branch:** merged via #11690
- **PR:** #11690
- **Paths:** see #11690
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`35464009`; collated from changes/11668-add-finish-feasibility-metrics-zmp-insid.md)
- **Summary:** Add finish-feasibility metrics (ZMP-inside fraction, friction-cone utilisation, foot slide and yaw, pelvis yaw error, vertical force range) to the dynamics receipt, with a post-processing CLI and a regression ratchet
- **Next step:** Shipped in PR #11690.

### DL-#11663 · Character Builder Docs: Web Builder and Appearance-V1 Sections, and a Reproducible Build-Dress-Export Tutorial (CMB-12)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11663
- **Branch:** merged via #12200
- **PR:** #12200, #12235, #12294, #12206
- **Paths:** see #12200
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5812c63d`; collated from changes/11663-desktop-character-builder-walkthrough-wi.md)
- **Summary:** Character Builder docs: web builder and appearance-v1 sections, and a reproducible build-dress-export tutorial (CMB-12)
- **Next step:** Shipped in PR #12200.

### DL-#11662 · Add CMB-11 Cross-Engine L0 Export Parity Verification Across MJCF, URDF, OpenSim, and Pinocchio

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11662
- **Branch:** merged via #12497
- **PR:** #12497
- **Paths:** see #12497
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`d388660f`; collated from changes/11662-add-cmb-11-cross-engine-l0-export-parity.md)
- **Summary:** Add CMB-11 cross-engine L0 export parity verification across MJCF, URDF, OpenSim, and Pinocchio
- **Next step:** Shipped in PR #12497.

### DL-#11661 · CMB-10 Slice 1: Stateless Frankenstein Assembly REST API (Parts, Plan Replay With Drop Evaluation, URDF/MJCF Export) Over AssemblySession

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11661
- **Branch:** merged via #11799
- **PR:** #11799
- **Paths:** see #11799
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`17d1fd9f`; collated from changes/11661-cmb-10-slice-1-stateless-frankenstein-as.md)
- **Summary:** CMB-10 slice 1: stateless Frankenstein assembly REST API (parts, plan replay with drop evaluation, URDF/MJCF export) over AssemblySession
- **Next step:** Shipped in PR #11799.

### DL-#11658 · Character Appearance API: Material Library and Validated Appearance-V1 Sidecar Bound to the Spec (CMB-7B)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11658
- **Branch:** merged via #12186
- **PR:** #12186, #12220, #12213, #12193, #12174
- **Paths:** see #12186
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5e7903c2`; collated from changes/11658-web-character-builder-drives-the-spec-na.md)
- **Summary:** Character appearance API: material library and validated appearance-v1 sidecar bound to the spec (CMB-7b)
- **Next step:** Shipped in PR #12186.

### DL-#11657 · CMB-6 Slice 1: Engine-Free Spec FK and a Rigidly Skinned Procedural glTF Body Export (One Bone per Spec Body, Y-Up), Wired Into the Character-Builder API, Desktop and Web Export Buttons

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11657
- **Branch:** merged via #12220
- **PR:** #12220
- **Paths:** see #12220
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`9b93fb2a`; collated from changes/11657-cmb-6-slice-2-gltf-animation-of-the-skin.md)
- **Summary:** CMB-6 slice 1: engine-free spec FK and a rigidly skinned procedural glTF body export (one bone per spec body, Y-up), wired into the character-builder API, desktop and web export buttons
- **Next step:** Shipped in PR #12220.

### DL-#11656 · Add Game-Quality MuJoCo Appearance: Materials, Textures, Skybox, Smooth Meshes

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11656
- **Branch:** merged via #11702
- **PR:** #11702
- **Paths:** see #11702
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b607519`; collated from changes/11656-add-game-quality-mujoco-appearance-mater.md)
- **Summary:** Add game-quality MuJoCo appearance: materials, textures, skybox, smooth meshes
- **Next step:** Shipped in PR #11702.

### DL-#11655 · Add the Engine-Agnostic Appearance-V1 Schema and Material Library

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11655
- **Branch:** merged via #11702
- **PR:** #11702
- **Paths:** see #11702
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b607519`; collated from changes/11655-add-the-engine-agnostic-appearance-v1-sc.md)
- **Summary:** Add the engine-agnostic appearance-v1 schema and material library
- **Next step:** Shipped in PR #11702.

### DL-#11654 · Character Builder Launcher Tile, Embed Adapter and Working Spec API (Build, Preview, Export)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11654
- **Branch:** merged via #11702
- **PR:** #11702
- **Paths:** see #11702
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b607519`; collated from changes/11654-character-builder-launcher-tile-embed-ad.md)
- **Summary:** Character Builder launcher tile, embed adapter and working spec API (build, preview, export)
- **Next step:** Shipped in PR #11702.

### DL-#11653 · Character Presets Library (JSON, Schema-Validated, Compiled via CMB-1)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11653
- **Branch:** merged via #11702
- **PR:** #11702
- **Paths:** see #11702
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b607519`; collated from changes/11653-character-presets-library-json-schema-va.md)
- **Summary:** Character presets library (JSON, schema-validated, compiled via CMB-1)
- **Next step:** Shipped in PR #11702.

### DL-#11652 · Spec-Native Character Parameters Compile to Full_Body_Spec via the Shared De Leva Builder

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11652
- **Branch:** merged via #11702
- **PR:** #11702
- **Paths:** see #11702
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b607519`; collated from changes/11652-spec-native-character-parameters-compile.md)
- **Summary:** Spec-native character parameters compile to full_body_spec via the shared De Leva builder
- **Next step:** Shipped in PR #11702.

### DL-#11651 · Consolidate Appearance, Character Builder, Frankenstein Assembly, Native Viewer Export and MyoFullBody PRs; Reconcile Registries

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11651
- **Branch:** merged via #11702
- **PR:** #11702
- **Paths:** see #11702
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b607519`; collated from changes/11651-consolidate-appearance-character-builder.md)
- **Summary:** Consolidate appearance, character builder, Frankenstein assembly, native viewer export and MyoFullBody PRs; reconcile registries
- **Next step:** Shipped in PR #11702.

### DL-#11647 · LaTeX Reference for the MyoFullBody Full-Body Muscle-Driven Swing Study

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11647
- **Branch:** merged via #11702
- **PR:** #11702
- **Paths:** see #11702
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b607519`; collated from changes/11647-latex-reference-for-the-myofullbody-full.md)
- **Summary:** LaTeX reference for the MyoFullBody full-body muscle-driven swing study
- **Next step:** Shipped in PR #11702.

### DL-#11646 · Activation-Coloured Four-View MyoFullBody Swing Renderer at Quarter Speed With Activation and Reserve Plots

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11646
- **Branch:** merged via #11702
- **PR:** #11702
- **Paths:** see #11702
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b607519`; collated from changes/11646-activation-coloured-four-view-myofullbod.md)
- **Summary:** Activation-coloured four-view MyoFullBody swing renderer at quarter speed with activation and reserve plots
- **Next step:** Shipped in PR #11702.

### DL-#11645 · Per-Frame Muscle Static Optimisation on MyoFullBody With Reserve Actuators, ID-Consistency Check and Fail-Closed Receipt

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11645
- **Branch:** merged via #11702
- **PR:** #11702
- **Paths:** see #11702
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b607519`; collated from changes/11645-per-frame-muscle-static-optimisation-on.md)
- **Summary:** Per-frame muscle static optimisation on MyoFullBody with reserve actuators, ID-consistency check and fail-closed receipt
- **Next step:** Shipped in PR #11702.

### DL-#11644 · Map the 44 Spec Coordinates Onto MyoFullBody by Segment-Orientation IK With Exact Joint-Equality Couplings, ROM Clamp and Muscle-Length Receipts

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11644
- **Branch:** merged via #11702
- **PR:** #11702
- **Paths:** see #11702
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b607519`; collated from changes/11644-map-the-44-spec-coordinates-onto-myofull.md)
- **Summary:** Map the 44 spec coordinates onto MyoFullBody by segment-orientation IK with exact joint-equality couplings, ROM clamp and muscle-length receipts
- **Next step:** Shipped in PR #11702.

### DL-#11643 · Pinned Hash-Verified MyoFullBody Asset Fetch With Licence Record

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11643
- **Branch:** merged via #11702
- **PR:** #11702
- **Paths:** see #11702
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b607519`; collated from changes/11643-pinned-hash-verified-myofullbody-asset-f.md)
- **Summary:** Pinned hash-verified MyoFullBody asset fetch with licence record
- **Next step:** Shipped in PR #11702.

### DL-#11587 · Guard Fsspec Lock Floor for CVE-2026-104851 (#11587)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11587
- **Branch:** merged via #12465
- **PR:** #12465
- **Paths:** see #12465
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`47c49436`; collated from changes/11587-bump-fsspec-to-2026-6-0-for-cve-2026-104.md)
- **Summary:** Guard fsspec lock floor for CVE-2026-104851 (#11587)
- **Next step:** Shipped in PR #12465.

### DL-#11570 · Consume Spec-Exported URDF Bundle Across Engines and Ledger MyoSuite Topology Divergence

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11570
- **Branch:** merged via #12501
- **PR:** #12501
- **Paths:** see #12501
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5c8d6434`; collated from changes/11570-consume-spec-exported-urdf-bundle-across.md)
- **Summary:** Consume spec-exported URDF bundle across engines and ledger MyoSuite topology divergence
- **Next step:** Shipped in PR #12501.

### DL-#11569 · Marker-Overlay Video Export Takes a Frame Size (Size_Px), so the Run-102 Simscape Playback Renders at 1920X1080 60 Fps (1X, 0.5X, 0.1X).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11569
- **Branch:** merged via #11901
- **PR:** #11901, #11868
- **Paths:** see #11901
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`82946677`; collated from changes/11569-simscape-urdf-exchange-receipt-r2025b-sm.md)
- **Summary:** Marker-overlay video export takes a frame size (size_px), so the run-102 Simscape playback renders at 1920x1080 60 fps (1x, 0.5x, 0.1x).
- **Next step:** Shipped in PR #11901.

### DL-#11554 · DIME Dynamics Window: Multiple Shooting With States and Controls as Decision Variables, Real Transition Defects D*K = X*{K+1} - Phi(X_K, U_K, Dt) From the Provider Step, and Real Costs Evaluated on the Returned Trajectory

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11554
- **Branch:** fix/11554-multiple-shooting
- **PR:** #11699
- **Paths:** see #11699
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`e50a9ba1`; collated from changes/11554-dime-dynamics-window-multiple-shooting-w.md)
- **Summary:** DIME dynamics window: multiple shooting with states and controls as decision variables, real transition defects d*k = x*{k+1} - Phi(x_k, u_k, dt) from the provider step, and real costs evaluated on the returned trajectory
- **Next step:** Shipped in PR #11699.

### DL-#11546 · Vectorised Multi-Trial Batching (Block-Arrow Schur, Time-Parallel Riccati) and Runtime Receipts (MOSAIC-14)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11546
- **Branch:** merged via #12456
- **PR:** #12456
- **Paths:** see #12456
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`e2e1416d`; collated from changes/11546-mosaic-14-multi-trial-batching.md)
- **Summary:** Vectorised multi-trial batching (block-arrow Schur, time-parallel Riccati) and runtime receipts (MOSAIC-14)
- **Next step:** Shipped in PR #12456.

### DL-#11545 · Document the DIME Manifold Chart, Retraction and Local-Coordinates Jacobians (Closed Forms From #11560) in the Model-Aware Matching Reference, With FD Acceptance Evidence

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11545
- **Branch:** fix/11545-dime-manifold-jacobians
- **PR:** #11700, #11693, #11698, #11695, #11694
- **Paths:** docs/research/model_aware_matching/model_aware_matching.tex
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`d867e857`; collated from changes/11545-mosaic-13-closed-form-multi-trial-benchm.md)
- **Summary:** Document the DIME manifold chart, retraction and local-coordinates Jacobians (closed forms from #11560) in the model-aware matching reference, with FD acceptance evidence
- **Next step:** Shipped in PR #11700.

### DL-#11544 · Humanoid/Manipulation Transfer: Known-Payload Anchor and Reference-Plus-Gains WBC Export (MOSAIC-12)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11544
- **Branch:** merged via #12461
- **PR:** #12461
- **Paths:** see #12461
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`7a652440`; collated from changes/11544-mosaic-12-humanoid-transfer.md)
- **Summary:** Humanoid/manipulation transfer: known-payload anchor and reference-plus-gains WBC export (MOSAIC-12)
- **Next step:** Shipped in PR #12461.

### DL-#11543 · Actuation Layer: Declared Joint-Level Activation Model and Replay Comparison

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11543
- **Branch:** merged via #12460
- **PR:** #12460, #12364
- **Paths:** see #12460
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`23817951`; collated from changes/11543-mosaic-11-first-slice-declared-joint-lev.md)
- **Summary:** Actuation layer: declared joint-level activation model and replay comparison
- **Next step:** Shipped in PR #12460.

### DL-#11542 · Multi-Swing Acquisition and Phase Registration With Event Anchoring (MOSAIC-10)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11542
- **Branch:** merged via #12462
- **PR:** #12462
- **Paths:** see #12462
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`425da9f3`; collated from changes/11542-mosaic-10-swing-registration.md)
- **Summary:** Multi-swing acquisition and phase registration with event anchoring (MOSAIC-10)
- **Next step:** Shipped in PR #12462.

### DL-#11540 · Analytic Dynamics Derivatives Replacing Finite Differences (MOSAIC-08)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11540
- **Branch:** merged via #12458
- **PR:** #12458
- **Paths:** see #12458
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`088511f2`; collated from changes/11540-mosaic-08-analytic-derivatives.md)
- **Summary:** Analytic dynamics derivatives replacing finite differences (MOSAIC-08)
- **Next step:** Shipped in PR #12458.

### DL-#11539 · MOSAIC-07: Kinematic Jerk Prior, Whiteness Gate and Declared-Discrepancy Calculator

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11539
- **Branch:** merged via #12444
- **PR:** #12444
- **Paths:** see #12444
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`db2c6fe2`; collated from changes/11539-mosaic-07-kinematic-jerk-prior-whiteness.md)
- **Summary:** MOSAIC-07: kinematic jerk prior, whiteness gate and declared-discrepancy calculator
- **Next step:** Shipped in PR #12444.

### DL-#11538 · Observation Factors: 2-D Keypoint Projection, Robust Kernels, Marker Attachment Offsets (MOSAIC-06)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11538
- **Branch:** merged via #12455
- **PR:** #12455
- **Paths:** see #12455
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`67b0a4ec`; collated from changes/11538-mosaic-06-observation-factors.md)
- **Summary:** Observation factors: 2-D keypoint projection, robust kernels, marker attachment offsets (MOSAIC-06)
- **Next step:** Shipped in PR #12455.

### DL-#11537 · Contact Wrenches in the Inner Solve: Unilateral/Friction-Cone SOCP, Measured GRF (MOSAIC-05)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11537
- **Branch:** merged via #12459
- **PR:** #12459
- **Paths:** see #12459
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`57085018`; collated from changes/11537-mosaic-05-contact-wrenches.md)
- **Summary:** Contact wrenches in the inner solve: unilateral/friction-cone SOCP, measured GRF (MOSAIC-05)
- **Next step:** Shipped in PR #12459.

### DL-#11536 · MOSAIC-04: Floating-Base Manifold Spline on SE(3) With Retraction, Exact Jacobians, and Outer Gauss-Newton Solve From Marker Observations

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11536
- **Branch:** merged via #12457
- **PR:** #12457
- **Paths:** see #12457
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5067c0bd`; collated from changes/11536-mosaic-04-manifold-spline.md)
- **Summary:** MOSAIC-04: Floating-base manifold spline on SE(3) with retraction, exact Jacobians, and outer Gauss-Newton solve from marker observations
- **Next step:** Shipped in PR #12457.

### DL-#11535 · MuJoCo/MJX Regressor Provider With Inertia Mapping and Planar Conformance MOSAIC-03

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11535
- **Branch:** merged via #12463
- **PR:** #12463
- **Paths:** see #12463
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`078f6e1d`; collated from changes/11535-mosaic-03-mujoco-regressor.md)
- **Summary:** MuJoCo/MJX regressor provider with inertia mapping and planar conformance MOSAIC-03
- **Next step:** Shipped in PR #12463.

### DL-#11534 · Provide the Pinocchio RegressorModel Provider (computeJointTorqueRegressor, Batched) With Planar Oracle Conformance (MOSAIC-02)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11534
- **Branch:** merged via #12450
- **PR:** #12450
- **Paths:** see #12450
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`9abae67c`; collated from changes/11534-mosaic-02-pinocchio-regressor.md)
- **Summary:** Provide the Pinocchio RegressorModel provider (computeJointTorqueRegressor, batched) with planar oracle conformance (MOSAIC-02)
- **Next step:** Shipped in PR #12450.

### DL-#11533 · Freeze Benchmark Protocol, Tolerances and Planar Conformance Oracle

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11533
- **Branch:** merged via #12453
- **PR:** #12453
- **Paths:** see #12453
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`9b60143d`; collated from changes/11533-freeze-benchmark-protocol-tolerances-and.md)
- **Summary:** Freeze benchmark protocol, tolerances and planar conformance oracle
- **Next step:** Shipped in PR #12453.

### DL-#11026 · Add Neutral Swing Posture Metrics (Pelvis Shift, Trunk Tilt Change) With Caller-Supplied Thresholds and Provenance (#11007 Package 3)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11026
- **Branch:** merged via #12358
- **PR:** #12358
- **Paths:** see #12358
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`6863fca6`; collated from changes/11026-add-neutral-swing-posture-metrics-pelvis.md)
- **Summary:** Add neutral swing posture metrics (pelvis shift, trunk tilt change) with caller-supplied thresholds and provenance (#11007 package 3)
- **Next step:** Shipped in PR #12358.

### DL-#10382 · Audit Club Face-Angle Observability (Sparse, Collinear, Uncaptured, Gapped or Mismatched Head Markers) and Keep Face Angle Unavailable in the Output Policy When It Is Unobservable

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #10382
- **Branch:** merged via #12420
- **PR:** #12420, #12425, #12338, #12431, #12434, #12424
- **Paths:** see #12420
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`989890a8`; collated from changes/10382-wire-the-face-angle-observability-audit.md)
- **Summary:** Audit club face-angle observability (sparse, collinear, uncaptured, gapped or mismatched head markers) and keep face angle unavailable in the output policy when it is unobservable
- **Next step:** Shipped in PR #12420.

### DL-#9507 · Cone-Mode Sparse Checkout in Sidekick Wheel Smoke Job

- **State:** shipped
- **Owner:** claude
- **Issue:** #9507
- **Branch:** `claude/issue-9507`
- **Paths:** `.github/workflows/package-standalone-sidekick.yml`, `tests/unit/packaging/test_standalone_sidekick_workflows.py`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/9507-metadata-only-completed-checkout-reconci.md)
- **Summary:** `smoke-test-wheel` used non-cone sparse checkout, which leaves `core.sparseCheckout=true` in `.git/config` and poisons later jobs on reused self-hosted runners. Switched to cone mode on `tests/fixtures` and added a guard test over all workflows.
- **Next step:** Implementation completed in merged #11522 (43d63ce98bcb6d4318bf5ab7ba8a6adf87071caf), issue #9507 closed completed. Metadata-only reconciliation: 24 checkout/workflow contract tests pass on 0c612812cb; no new fleet-runtime, scientific or physical qualification is claimed. Continue routine runner monitoring.
- **PR:** #11522, #12278, #12418

### DL-#10783 · Enforce the Deferred Validation Catalog

- **State:** shipped
- **Owner:** codex
- **Issue:** #10783; full rollout Repository_Management#1687
- **PR:** #10784
- **Branch:** `chore/10783-deferred-catalog-guard`
- **Paths:** `shared_scripts/`, `tests/unit/repo_hygiene/test_deferred_catalog_hook.py`, `.pre-commit-config.yaml`, `docs/development/`, `AGENT_HANDOFF.md`, `README.md`, `SPEC.md`
- **Started:** 2026-09-23
- **Last verified:** 2026-10-12 (`8e9b4b97`; collated from scoped #10783 verification on UpstreamDrift main)
- **Summary:** Install the exact central validator and always-run hook while preserving six published v1 plans, executable software and unavailable empirical acceptance.
- **Next step:** The UpstreamDrift software guard is complete; the three canonical-function exceptions expire 2026-10-23. Broader fleet rollout remains separately tracked in Repository_Management #1687.

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
No material development-log change — Bolt `np.linalg.norm` → `einsum` consolidation (#11073, #11074, #11076) is a behaviour-preserving micro-optimisation with no feature entry.
No material development-log change — Bolt `np.linalg.norm` → `sqrt(einsum)`/`math.sqrt(np.vdot)` consolidation (#11112, #11128, #11129) is a behaviour-preserving micro-optimisation with no feature entry.
No material development-log change — PyJWT floor/lock bump to 2.14.0 for OSV GHSA-w6j9-cwv2-h6wq (#11153) is a dependency-only change with no feature entry.
No material development-log change — urllib3 (2.8.0, CVE-2026-97687) and PyJWT (2.15.0, CVE-2026-101918) floor/lock bumps (#11191) are dependency-only changes with no feature entry.
No material development-log change — Bolt np.linalg.norm → sqrt(einsum) in rigidity.py (#11189) is a behaviour-preserving micro-optimisation with no feature entry.

## DL-#11357 — Pure Raw Shaft Fragment Derivative Diagnostics

- **Change:** Canonical immutable options/check/result contracts and public
  raw fragment analytic/central FD diagnostics; canonical camera/marker reuse.
- **Validation:** Repository fail-first scaffold regression followed by focused
  native-free synthetic tests, curated exports and fingerprint inclusion.
- **Limits:** No historical native/Library/render/fit operation, no objective
  or evidence-role change, and no clinical/scientific qualification.
- **Next Step:** Review typed core, then separately design authenticated
  workspace consumers using existing job/store/renderer boundaries.

## Authored Replay Impact Application Checkpoint — #11464

The normal React journey now previews impact from a verified authored replay,
with exact geometry/sample/flight declarations, bounded native execution,
cancellation, persisted run-ID recall and authenticated four-file download.
Saved results display their own assumptions and metrics. Canonical receipt and
trajectory owners remain unchanged; source/runtime/parent and artifact checks
fail closed. Research acceptance remains rejected, with authored seconds and
false physical/scientific qualification.

Root final validation: 125 focused Python cases pass with one Windows symlink
capability skip; one additional opt-in native case passes with actual MuJoCo,
clean process transport, impact solver and Rust flight (no substitutions).
The React cohort passes 104 cases; configured six-owner mypy, scoped lint and
format checks pass. Earlier failures and immutable agent freezes are retained
in the application review. Four-page LaTeX Application Supplement 1 is compiled
without final layout warnings and all pages visually reviewed; editable source,
PDF and the explicitly synthetic native bundle are saved on the local Desktop.
Earlier Methods V8 and the five-page extraction report remain unchanged.

See [Application Procedure](necromatcher-replay-impact.md) and its linked review/report. Tiger stays
within original frames 0–190 ([0,191)); Hogan stays [0,750). No historical
Library mutation, qualified match or new ControlTower delivery is established.
Remaining goal work includes improved historical matching, authored dynamics,
simulation golf integration and desktop parity. The full goal remains active.

## Native Desktop Replay Impact Checkpoint — #11487

The native Necromatcher tile now imports registered authored replays, runs the
shared bounded replay-impact session off the GUI thread, cancels owned jobs,
recalls retained run IDs after restart and exports an authenticated four-file
ZIP exclusively. Saved assumptions remain separate from editable declarations.
The public Shot Tracer handoff retains exact detached positions and provenance.

Root validation passes 146 focused Qt/viewer/parity cases without skips and
configured mypy on three production owners. One additional opt-in actual
MuJoCo/HDF5/clean-worker/impact/Rust native Qt case passes with Python exit 0.
The three final synthetic screenshots are readable and visually reviewed;
retained positions are verified, while OpenGL pixel accuracy is unverified.
Test-only font bootstrap uses existing Windows Segoe UI when the offscreen
font database is empty. The 120-second whole-test cap does not change the
60-second native budget; earlier timeout evidence is preserved.

The three-page Native Desktop Supplement 1 and editable LaTeX, screenshots and
intact synthetic bundle are saved on the local Desktop. See [Procedure](necromatcher-replay-impact.md)
and [Frozen Review](historical_capture/native-desktop-replay-impact-review.json). Tiger remains [0,191), Hogan [0,750).
Historical fits, scientific qualification, simulation golf and final overall
desktop parity acceptance remain open. No new ControlTower delivery or remote
CI-green claim is made. The full goal remains active.

## Local Research Admission Race Follow-Up — #11493

A concurrent request regression reproduced reused shot-ID admission after another
request prepared and cancelled during authentication. The API now applies the
same identity guard before and immediately after authentication. All 10 API
cases pass; a fresh additional actual MuJoCo/clean-worker/Rust API and Qt
acceptance passes in 24.56 s, with seven exact retained samples and no
re-simulation on recall. Configured mypy passes for the changed API owner.
The earlier 209-case cohort and 106 React cases remain separate checkpoint
evidence. The follow-up preserves unverified qualification and all historical
limits; the full goal remains active.

See [Follow-Up Review](historical_capture/local-research-golf-admission-race-review.json). Source inventory covers 6,697 src/test
files, with equal before/after maps; this is not a whole-repository baseline.

## User-Requested Handoff and Stop Checkpoint

The user requested a committed checkpoint and instructed this agent to stop
pursuing the goal after turnover. See [Next-Agent Turnover](necromatcher-next-agent-turnover.md) for the
exact implementation state at f0e14944, scoped work packets, public owners,
validation commands, historical evidence boundaries and remaining acceptance.
No proposed optimized video export or new fit was started. This documentation
checkpoint follows the published local research integration; the full goal is
unfinished and is to be paused after normal commit/push and handoff publication.

## Capture Registry Parent Integration - #11162 / #11172

Accepted-main synchronization preserves capture-registry, export and swing-comparison source changes. Record conflicts retain distinct entries from both branches. All 133 ordered matched-swing ledger rows and non-timestamp metadata are identical; retain accepted-main timestamp. Regenerate the monolith register from merged source. Scientific matching/dynamics acceptance remains separate.

Record validation: all five monolith-register tests pass; no duplicate primary headings or conflict markers; all 133 ledger rows and non-timestamp metadata preserved exactly; registry/export/comparison source unchanged. Broader capture tests are pending pinned Tools initialization; first collection attempt stopped on the missing submodule. Protected parent CI and full scoped rerun remain required before merge.

## Capture Comparison DRY Gate Remediation - #11162 / #11172

CI identified duplicated club endpoint selection in angular-speed and wrist metrics. A shared private helper preserves explicit endpoint precedence, measured-marker aliases, array identity and missing-endpoint/shaft-axis behavior. AGY Gemini Flash supplied the bounded extraction; root reviewed the exact diff. Existing focused regression suite passes 190 tests with 15 unavailable-private-workbook skips, zero errors/failures, natural exit 0. Ruff format/check and full DRY duplication gate pass without changing baselines. Regenerate the monolith register for the revised file length. Current-head protected CI remains required.

## Capture Leaderboard Suite Classification - #11162 / #11172

Protected CI verified the DRY extraction, then identified 14 net-new unmarked tests in TestPerCaptureLeaderboard. Mark this pure-Python class as unit; all 14 execute and pass under unit selection with no skips/errors. Full suite-marker ratchet passes without baseline changes. Broader focused capture/comparison qualification remains 190 passed and 15 unavailable-private-workbook skips. Protected current-head CI remains required before merge.

## Capture Comparison Inventory Integration - #11162 / #11172

CI at bd1ee9a8e passes repository structure and 20,546 unit cases; its only unit failure identifies five new swing_comparison paths missing from the shared-tools divergence inventory. Regenerate JSON and Markdown with the authoritative generator against the pinned Tools tree. All five paths classify as UD-only; existing overlap totals remain unchanged. All ten inventory tests pass with no errors/skips, including the committed-tree freshness test, and generator --check succeeds. Protected current-head CI remains required before merging.

### Selected Club Continuity Update (2026-10-03)

PR #11256 merged at `7db86ff173653a235ea63bb4af347af0a8a5ab83`. The eight verified Human ellipsoid MP4s remain on both Desktops in `Simscape_Matches_20261003_Club_Refined_1080p`; these are IK reconstructions. The source reference now presents this current selection before historical experiments.

The selected club fits were tested at all anchors and midpoints using independent native position, velocity and fresh 91-state checks. Sequential seeding accepted 1,306/1,307 tour states and 732/733 owner states. A targeted diagnostic recovered the owner's rejected original anchor from its saved closed-arm seed; the previous accepted pose reproduced an alternate branch. A complete anchor-seeded rerun exited naturally at 20:31:23 UTC and accepted all 733 owner states (367 anchors and 366 midpoints), while the tour still rejected midpoint 998 at 1.3847222 s. All original anchors matched within 1.73e-15 physical residual. The peak accepted dependent angular rates remain approximately 1,639 and 4,118 deg/s; sampled-state acceptance does not qualify between-sample branch continuity, dependent acceleration, anatomy, contact or forward dynamics.

The private production integration candidate passed 17 native pre-model contract cases (parent RED, candidate GREEN, 51 recorded checks) at 20:20:58 UTC. Numerical whole-output parity and positive-fit validation remain separate pending gates. Original failed runner receipts are preserved. A custom vector measurement block built successfully, but the eight-axis instrumented simulation hit the Home license's 1,000-nonvirtual-block limit. No all-35 measured torque or moving feedback-off replay claim follows from these tests. Original physical model files were not saved.

See the separate editable research reference and the extended aggregate `tangent_c2_checkpoint_20261003.json` for scope and provenance. The built-in LaTeX compiler still fails with `Unable to find standard directories for platform`; PDF compilation/page review are unverified. Canonical calculation inventory and governed manual release remain blocked; this update grants neither a release exemption nor scientific approval.

## F05d Native Manifold BoxFDDP Candidate (#11932; 2026-10-09)

A test-first MuJoCo 3.8/Crocoddyl 3.2.1 child now solves the same
contact-free floating-root/two-hinge physical state with a 16-dimensional
configuration/velocity tangent, exact native Euler action derivative,
bounded direct motor input and nonlinear native fallback admission. The
paired source-hashed receipt measures BoxFDDP against SciPy native shooting
under identical reference, truth, input box, horizon, iteration and
cooperative wall budget, alternates solver order and charges a common
Tools T01 bootstrap to both. Every applied input independently replays from
complete native integration state. This is synthetic controller evidence,
not a hard deadline, measured capture fit or full-body qualification;
the F05 parent and canonical manual publication remain blocked. Read
chapter 26 and `F05D_NATIVE_MANIFOLD_BOX_FDDP_TURNOVER.md`.

## Bounded Nonuniform Moco Refinement Plan (#12244; 2026-10-10)

The 20-cell synthetic Moco parent inputs are retained byte-for-byte and the
offline activation report reproduces from those committed source, request,
solution, control and replay-audit artifacts. A pure source-bound planner marks
the observed full Hermite-Simpson cells, caps one child pass at four marked
cells and 24 intervals, and rejects evidence/hash or grid mismatches. The worker
uses the normalized custom mesh and parent state seed, then linearly samples
the persisted controls at every child endpoint and midpoint without control
trajectory resampling. Pure planner/interpolation regressions pass. No native
child solve or replay was run; solver API/time-grid acceptance and all physical
qualification remain open. The context catalog could not be refreshed here:
its `vendor/ud-tools` junction escapes this checkout to the frozen
`feedback-drake-native-11838` donor, and the context CLI correctly rejects that
path. See the #12244 change fragment, chapter 13 and F07 ladder turnover.

The follow-up recomputes and compares the full plan before solver setup and
builds the post-solve mesh audit from the seeded grid in the child request.
Pure JSON-reload and tampering regressions pass; there is still no native child
solve or replay evidence.

The next correction uses the OpenSim 4.6-compatible states-table insertion
signature and restores exact declared initial values after interpolation; a
strict fake-SWIG test covers the behavior without running the native solver.

An installed OpenSim 4.6 API smoke also passed for the committed parent STOs:
the states table inserted into a native 43-row `MocoTrajectory`, controls were
PWL-seeded, and cold initial values matched exactly. It ran under the 30-second
owned-tree supervisor with 140 MB sampled peak, and called no model init, solve,
or replay. A first malformed empty-trajectory smoke crashed in a SWIG setter;
the corrected data-bearing constructor passed. Actual solver acceptance remains
unverified.
