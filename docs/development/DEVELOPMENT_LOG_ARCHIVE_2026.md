# Development Log Archive — 2026

### DL-#11719 · GCV-13 Slice 3: Decorative Address Ball in the MuJoCo, OpenSim and MyoSuite Native Export

- **State:** shipped
- **Shipped:** 2026-10-14
- **Owner:** claude
- **Issue:** #11719 (epic #11706)
- **Branch:** claude/gcv-13c-native-ball-backends
- **PR:** not created (stacked on #12114, which is stacked on #11992), #12140, #12332, #12283, #12297, #12481
- **Paths:** src/tools/native_viewer_export/backends/\_ball.py, src/tools/native_viewer_export/backends/\_worker_job.py, src/tools/native_viewer_export/backends/\_subprocess.py, src/tools/native_viewer_export/backends/mujoco_native.py, src/tools/native_viewer_export/backends/mujoco_worker.py, src/tools/native_viewer_export/backends/myosuite_arena.py, src/tools/native_viewer_export/backends/myosuite_worker.py, src/tools/native_viewer_export/backends/opensim_simbody.py, src/tools/native_viewer_export/backends/opensim_worker.py, src/tools/native_viewer_export/README.md, src/config/feature_parity.json, tests/unit/tools/native_viewer_export/test_ball_native_backends.py
- **Started:** 2026-10-10
- **Last verified:** 2026-10-14 (`c0b6c94b`; collated from changes/11719-promote-render-club-head-and-ball-to-par.md)
- **Summary:** GCV-13 adds the resolved decorative address ball to remaining engine backends and worker transport; OpenSim attaches build-time ground geometry. The web ClubHead follow-up remains open; this presentation change does not alter physics.

### DL-#10950 · Exploratory GS3DX Simscape Model: Quaternion Joints and Full-Body Under the 1,000-Block Budget

- **State:** shipped
- **Shipped:** 2026-10-03
- **Owner:** codex (Gemini 3.8 CLI delegates)
- **Issue:** #10950 (children #10951–#10959, #10979, #10985, #10986, #11011)
- **PR:** #11256 (draft); original #11179 remains separate.
- **Branch:** `feat/simscape-matching-review-main-20261002` (review continuation of `feat/simscape-gs3dx-exploratory`)
- **Paths:** `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/`, `docs/research/simscape_matching_reference/`, `AGENTS.md`
- **Started:** 2026-09-26
- **Last verified:** 2026-10-03: production observation-export checkpoint passes 39 native contracts plus actual two-frame owner cache identity/reuse integration. Private rotational continuity helper passes 16 RED/GREEN tests; exact inactive whole-output parity passes on eight owner and 53 tour frames. Sparse/dense experiments recheck all 305 native poses and exit naturally; marker/step tradeoffs remain adverse and unpromoted, including owner root motion and tour wrist/root increments. Six owner-session originals and archive are hash-verified on a local machine, with deterministic supplemental acquisition provenance. Source-only LaTeX review reconciles current versus historical candidates, source-mask tensor indexing and source-verified balance units. See continuity_experiment_checkpoint_20261003.json. Normal current-main integration includes 4eb3051c3; optional-engine verification records five MuJoCo passes/four missing-mj_copyData failures and two OpenSim skips on this host. MATLAB sources are unchanged; protected CI/review and latest PDF qualification remain open.
- **Summary:** Agent-editable GS3DX clones, LaTeX reference and documentation policy, truthful serialized execution, Human ellipsoid geometry adapter, joint-role mapping and calibrated full foot orientation. A is the tour reference; O is the owner capture. Marker residuals, offsets, orientation and provenance remain independently visible; videos are IK, not qualified forward dynamics. Additional recorded 2026-10-02 trial evidence: controlled 0-50 ms tour rejected at peak 2.000907 BW; separate 20-50 ms ramp runs met only five one-second hold gates (tour 3.176 mm / 0.836 deg; owner 2.321 mm / 0.699 deg). These limited checks do not qualify independent full-body dynamics.

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

### DL-#10188 · Model-Driven Golf Simulator Integration

- **State:** shipped
- **Shipped:** 2026-09-16
- **Owner:** local
- **Issue:** #10188 (child #10200 active; children #10189–#10200)
- **Branch:** feat/issue-10200-native-avatar-course-feedback
- **PR:** #10227 (merged; GS-10 #10199); #10226 (merged; GS-09 #10198); #10225 (merged; GS-08 #10197); #10222 (merged; GS-07 #10196); #10220 (merged; GS-06 #10195); #10217 (merged; GS-05 #10194); #10215 (merged; GS-04 #10193); #10213 (merged; GS-03 #10192); #10208 (merged; GS-00 #10189, GS-01 #10190, GS-02 #10191); #10201 (merged; planning)
- **Paths:** `docs/plans/golf_simulator_integration/NATIVE_AVATAR_COURSE_FEEDBACK_RESEARCH.md; src/shared/python/golf_simulator/contracts.py; src/shared/python/golf_simulator/__init__.py; tests/unit/golf_simulator/test_avatar_course_feedback_research.py`
- **Started:** 2026-09-15
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** GS-11 (#10200) Native GSPro model animation and autonomous course feedback research delivered. Documents Unity runtime constraints and confirms Course Designer produces static AssetBundles with zero dynamic skeletal mesh hooks; confirms Open Connect v1 is strictly unidirectional shot input lacking ball landing, lie, surface, wind, elevation, hazard, or aim feedback; prohibits memory scraping / DLL injection / packet sniffing; establishes turnkey vendor inquiry templates; formalizes Synchronized Companion Presentation Architecture; implements `UnsupportedCapabilityError` and `assert_capability_supported()` in `contracts.py`.
- **Evidence:** tests/unit/golf_simulator/test_avatar_course_feedback_research.py; docs/plans/golf_simulator_integration/NATIVE_AVATAR_COURSE_FEEDBACK_RESEARCH.md.

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

- **State:** shipped
- **Shipped:** 2026-09-16
- **Owner:** local
- **Issue:** #10062 (children #10063 to #10070); continued by epic #10162 (MM-1 to MM-10, HO-1 to HO-10)
- **Branch:** refactor/10251-pipeline-stages-run-ground-support
- **PR:** #10092 (FB-5, #10069); #10090 (Step 3 merged); #10089 (FB-4, #10068 merged); #10203 (Step 4 cross-engine replays merged); #10218 (HO-1 #10155 merged); #10224 (HO-2 #10156 merged); #10235 (HO-7 #10161 merged); #10236 (HO-4 #10158 merged); #10249 (HO-9 #10111 merged); #10228 (HO-3 #10157 merged); #10261 (HO-11 #10250 merged); #10258 (HO-8 #10108 merged)
- **Paths:** docs/development/full_body_models; src/shared/python/motion_matching/full_body_spec.py; src/shared/python/motion_matching/contact_law.py; src/shared/python/motion_matching/tour_capture_contract.py; src/shared/python/motion_matching/marker_calibration.py; src/shared/python/motion_matching/full_body_ik.py; src/shared/python/motion_matching/visual_skeleton.py; src/shared/python/motion_matching/derivative_resolution.py; src/shared/python/motion_matching/full_body_forward_dynamics.py; src/shared/python/motion_matching/anthropometry.py; src/shared/python/motion_matching/hip_calibration.py; src/shared/python/motion_matching/pipeline; src/tools/motion_matching; tests/unit/motion_matching/pipeline; tests/unit/motion_matching; tests/unit/tools; tests/tools/motion_matching; scripts/config/mjx_env_pins.json; scripts/setup_mjx_env.ps1; scripts/setup_mjx_env.sh; tests/unit/motion_matching/test_document_freshness.py
- **Started:** 2026-09-13
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Full-body handoff aligns headline values to primary receipts, adds matched-swing status/docs freshness, and centralizes ground-support IK/dynamics behind shared modules with compatibility shims.
- **Evidence:** docs/development/full_body_models/evidence/ground_support/CANONICAL_RUN.md, docs/development/full_body_models/evidence/ground_support/bisect_receipt.json, tests/unit/motion_matching/test_handoff_numbers_match_receipts.py, tests/unit/motion_matching/test_full_body_consolidation.py, docs/development/matched_swing_program/README.md, docs/development/matched_swing_program/GATES.md, docs/development/matched_swing_program/WAVES.md, scripts/generate_matched_swing_status.py, tests/docs/test_matched_swing_status_freshness.py.

### DL-#13577 · Add Fast PR/Push and Full Merge-Queue/Scheduled Tiers With a Fail-Closed Required Aggregate

- **State:** shipped
- **Shipped:** 2026-10-21
- **Owner:** unassigned
- **Issue:** #13577
- **Branch:** codex/13577-ci-fast-full
- **PR:** #13612
- **Paths:** .github/workflows/ci-standard.yml,scripts/ci/classify_event_tier.py,tests/ci/test_ci_standard_workflow.py,tests/ci/test_required_check_contract.py
- **Started:** 2026-10-21
- **Last verified:** 2026-10-21 (`ac77271c`; collated from changes/13577-add-fast-and-full-ci-event-tiers-with-a.md)
- **Summary:** Add fast PR/push and full merge-queue/scheduled tiers with a fail-closed required aggregate

### DL-#13595 · CI Standard Lint Gate

- **State:** shipped
- **Shipped:** 2026-10-21
- **Owner:** codex
- **Issue:** #13595
- **Branch:** codex/13595-ci-lint
- **PR:** #13599
- **Paths:** .github/actionlint.yaml,.github/WORKFLOWS.md,.github/workflows/ci-standard.yml,.github/workflows/fleet-policy-gate.yml,.github/workflows/architecture-map-contract.yml,scripts/pre_push_hook.py,tests/ci/test_ci_infrastructure.py,tests/ci/test_ci_drift_gate_arguments.py,tests/ci/test_devlog_repair_workflow.py,tests/ci/test_publication_quality_workflow.py,AGENT_HANDOFF.md
- **Started:** 2026-10-21
- **Last verified:** 2026-10-21 (`d5d8ce1c`; collated from changes/13595-ci-standard-lint.md)
- **Summary:** Repair CI lint, trusted governed policy consumption, publication detection coverage, and pre-push failure guidance

### DL-#13579 · Engine Docker Cache Steps per Fleet Contract

- **State:** shipped
- **Shipped:** 2026-10-21
- **Owner:** unassigned
- **Issue:** #13579
- **Branch:** feat/engine-docker-cache-13579
- **PR:** #13598
- **Paths:** see #13598
- **Started:** 2026-10-21
- **Last verified:** 2026-10-21 (`d5112d55`; collated from changes/13579-engine-docker-cache-steps-per-fleet-cont.md)
- **Summary:** engine docker cache steps per fleet contract

### DL-#13580 · Quarantine Pilot for Simscape Grip Contact Tests and Audit Validator (UD-CI-030)

- **State:** shipped
- **Shipped:** 2026-10-21
- **Owner:** gemini
- **Issue:** #13580
- **Branch:** feat/quarantine-pilot-13580
- **PR:** #13596
- **Paths:** see #13596
- **Started:** 2026-10-21
- **Last verified:** 2026-10-21 (`1e652a65`; collated from changes/13580-quarantine-pilot-for-simscape-grip-conta.md)
- **Summary:** Quarantine pilot for Simscape grip contact tests and audit validator (UD-CI-030)

### DL-#13574 · Record Merge-Queue Batch-Size Experiment Snapshots and Fleet Merge Settings Checker

- **State:** shipped
- **Shipped:** 2026-10-21
- **Owner:** unassigned
- **Issue:** #13574
- **Branch:** cursor/merge-queue-batch-size-db92
- **PR:** #13588
- **Paths:** see #13588
- **Started:** 2026-10-21
- **Last verified:** 2026-10-21 (`1c6736d7`; collated from changes/13574-merge-queue-batch-size-snapshot.md)
- **Summary:** Record merge-queue batch-size experiment snapshots and fleet merge settings checker

### DL-#13571 · Gate OpenSim CI on Paths or Ci:Opensim Label With Classify_Ci_Changes Job

- **State:** shipped
- **Shipped:** 2026-10-21
- **Owner:** unassigned
- **Issue:** #13571
- **Branch:** cursor/opensim-ci-path-label-gating-eee2
- **PR:** #13585
- **Paths:** see #13585
- **Started:** 2026-10-21
- **Last verified:** 2026-10-21 (`31f4a29e`; collated from changes/13571-gate-opensim-ci-on-paths-or-ci-opensim-l.md)
- **Summary:** Gate OpenSim CI on paths or ci:opensim label with classify_ci_changes job

### DL-#13038 · Record the OSL Epic Handoff State (Open PRs, Owner Decisions, Rejected Video Work)

- **State:** shipped
- **Shipped:** 2026-10-20
- **Owner:** unassigned
- **Issue:** #13038
- **Branch:** docs/osl-handoff-2026-10-20
- **PR:** #13549
- **Paths:** see #13549
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`12378aad`; collated from changes/13038-record-the-osl-epic-handoff-state-open-p.md)
- **Summary:** Record the OSL epic handoff state (open PRs, owner decisions, rejected video work)

### DL-#13190 · HRG-H2 Diagnostic Palm-Thumb Pad: Drift-Gated Physical Trail_Palm_Lead_Thumb Pair on the Shared G2 Law in Separated_V2, Impact Timestep and Preload Sensitivity Reports, and a New Spec Revision With Its Own Spec_Sha256; Contract Pinned by Tests.

- **State:** shipped
- **Shipped:** 2026-10-20
- **Owner:** unassigned
- **Issue:** #13190
- **Branch:** feat/13190-palm-thumb-pad
- **PR:** #13525
- **Paths:** src/shared/python/grip_contact/palm_thumb_pad.py,tests/unit/grip_contact/test_palm_thumb_pad.py
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`c93840a6`; collated from changes/13190-hrg-h2-diagnostic-palm-thumb-pad-drift-g.md)
- **Summary:** HRG-H2 diagnostic palm-thumb pad: drift-gated physical trail_palm_lead_thumb pair on the shared G2 law in separated_v2, impact timestep and preload sensitivity reports, and a new spec revision with its own spec_sha256; contract pinned by tests.

### DL-#13171 · HRG-E3 Slice: Pelvis and Thorax ISB Lab-Frame Turn, Tilt, Obliquity and Side-Bend Golf Terms With Golf-Word Aliases; Thoracic/Lumbar Split Marked Unavailable

- **State:** shipped
- **Shipped:** 2026-10-20
- **Owner:** unassigned
- **Issue:** #13171
- **Branch:** claude/ud-13171-trunk-pelvis-terms
- **PR:** #13498
- **Paths:** see #13498
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`49f40152`; collated from changes/13171-hrg-e3-slice-pelvis-and-thorax-isb-lab-f.md)
- **Summary:** HRG-E3 slice: pelvis and thorax ISB lab-frame turn, tilt, obliquity and side-bend golf terms with golf-word aliases; thoracic/lumbar split marked unavailable

### DL-#13490 · Hang Watchdog Dumps Stacks From a GIL-Held Python Handler so SIGUSR1 No Longer Segfaults the Xdist Controller (Exit 139)

- **State:** shipped
- **Shipped:** 2026-10-20
- **Owner:** unassigned
- **Issue:** #13490
- **Branch:** fix/qt-teardown-segv
- **PR:** #13497
- **Paths:** see #13497
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`76b42c4d`; collated from changes/13490-hang-watchdog-dumps-stacks-from-a-gil-he.md)
- **Summary:** Hang watchdog dumps stacks from a GIL-held Python handler so SIGUSR1 no longer segfaults the xdist controller (exit 139)

### DL-#13256 · Re-Lands the Salvaged Source-Bound Native OpenSim Muscle-Effort Adapter and Runtime With SDK-Free Contract Tests; Native Tests Skip Without the SDK and No Manager Advance Has Been Executed.

- **State:** shipped
- **Shipped:** 2026-10-20
- **Owner:** unassigned
- **Issue:** #13256
- **Branch:** feat/ud-13256-native-muscle-adapter
- **PR:** #13478
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/native_muscle_effort_adapter.py,src/engines/physics_engines/opensim/python/tour_matching/native_muscle_effort_runtime.py,src/engines/physics_engines/opensim/python/tour_matching/muscle_replay.py,tests/unit/engines/opensim/test_native_muscle_effort_adapter.py,tests/opensim/test_native_muscle_effort_adapter.py,tests/opensim/test_native_muscle_effort_native.py
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`bf10d600`; collated from changes/13256-native-muscle-effort-adapter.md)
- **Summary:** Re-lands the salvaged source-bound native OpenSim muscle-effort adapter and runtime with SDK-free contract tests; native tests skip without the SDK and no Manager advance has been executed.

### DL-#13176 · HRG-Q1 Slice: Full-Body-V2 Mass Matrix and Gravity Qualified Against a Spec-Only Geometric-Jacobian Reference on MuJoCo (Measured Rel. Error <=4E-16, Tol 1E-12) With a Pinocchio Twin; Drake, OpenSim, MyoSuite, Limits, Closure, Contact and ID Remain Open

- **State:** shipped
- **Shipped:** 2026-10-20
- **Owner:** unassigned
- **Issue:** #13176
- **Branch:** claude/ud-13176-hrg-q1-qualification
- **PR:** #13491
- **Paths:** see #13491
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`8000133e`; collated from changes/13176-hrg-q1-slice-full-body-v2-mass-matrix-an.md)
- **Summary:** HRG-Q1 slice: full-body-v2 mass matrix and gravity qualified against a spec-only geometric-Jacobian reference on MuJoCo (measured rel. error <=4e-16, tol 1e-12) with a Pinocchio twin; Drake, OpenSim, MyoSuite, limits, closure, contact and ID remain open

### DL-#13112 · Forward-Dynamics Replay Reproducibility: Deterministic Replay Test, Seeded Lyapunov-Style Sensitivity Probe (--Replay-Sensitivity), Bounded 0.1 s Post-Impact FD Metric, and a Base-of-Support Bound on the ZMP Filter Centre-of-Mass Shift

- **State:** shipped
- **Shipped:** 2026-10-20
- **Owner:** unassigned
- **Issue:** #13112
- **Branch:** claude/ud-13112-fd-replay-repro
- **PR:** #13501
- **Paths:** see #13501
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`67c97461`; collated from changes/13112-replay-sensitivity-zmp-bound.md)
- **Summary:** Forward-dynamics replay reproducibility: deterministic replay test, seeded Lyapunov-style sensitivity probe (--replay-sensitivity), bounded 0.1 s post-impact FD metric, and a base-of-support bound on the ZMP filter centre-of-mass shift

### DL-#12985 · Align Default Engine Models to Canonical Golfer and Raise KeyError on Unknown Body in Compute_Jacobian

- **State:** shipped
- **Shipped:** 2026-10-19
- **Owner:** unassigned
- **Issue:** #12985
- **Branch:** orch/12985-agy
- **PR:** #13312
- **Paths:** see #13312
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`23dbe141`; collated from changes/12985-align-default-engine-models-to-canonical.md)
- **Summary:** Align default engine models to canonical golfer and raise KeyError on unknown body in compute_jacobian

### DL-#12990 · Standardize Get_Joint_Names Across Physics Engines to Match Generalized Velocity Dimensions (Nv) With Per-DOF Expansion for Multi-DOF Joints

- **State:** shipped
- **Shipped:** 2026-10-19
- **Owner:** unassigned
- **Issue:** #12990
- **Branch:** orch/12990-agy
- **PR:** #13345
- **Paths:** see #13345
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`8115b862`; collated from changes/12990-standardize-get-joint-names-across-physi.md)
- **Summary:** Standardize get_joint_names across physics engines to match generalized velocity dimensions (nv) with per-DOF expansion for multi-DOF joints

### DL-#13364 · Ci(Quality-Gate): Aggregate Frontend-Tests and the Other Gating CI Standard Jobs so a Red Job Blocks the Merge Queue

- **State:** shipped
- **Shipped:** 2026-10-19
- **Owner:** unassigned
- **Issue:** #13364
- **Branch:** cursor/merge-queue-gate-13364
- **PR:** #13366
- **Paths:** see #13366
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`65857013`; collated from changes/13364-quality-gate-aggregates-gating-jobs.md)
- **Summary:** ci(quality-gate): aggregate frontend-tests and the other gating CI Standard jobs so a red job blocks the merge queue

### DL-#13355 · Fix(Ui): Include Physical Style in resolveGrip to Resolve Red Main on Frontend-Tests

- **State:** shipped
- **Shipped:** 2026-10-19
- **Owner:** unassigned
- **Issue:** #13355
- **Branch:** fix/main-grip-selection-style
- **PR:** #13361
- **Paths:** see #13361
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`3e195a56`; collated from changes/13355-fix-grip-selection-style.md)
- **Summary:** fix(ui): include physical style in resolveGrip to resolve red main on frontend-tests

### DL-#13350 · Fix(Launcher): Renumber Jaxsim_Dashboard to Order 105 to Resolve Duplicate Tile Order 33

- **State:** shipped
- **Shipped:** 2026-10-19
- **Owner:** unassigned
- **Issue:** #13350
- **Branch:** fix/13350-launcher-tile-order
- **PR:** #13358
- **Paths:** see #13358
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`deec5362`; collated from changes/13350-renumber-jaxsim-dashboard.md)
- **Summary:** fix(launcher): renumber jaxsim_dashboard to order 105 to resolve duplicate tile order 33

### DL-#13348 · Fix(Data_Io): Make C3d_Loader Resilient to Sidekick.Lab.Bio Test Pollution

- **State:** shipped
- **Shipped:** 2026-10-19
- **Owner:** unassigned
- **Issue:** #13348
- **Branch:** fix/13348-c3d-marker-set-import
- **PR:** #13359
- **Paths:** see #13359
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`100b1885`; collated from changes/13348-c3d-loader-bio-resilience.md)
- **Summary:** fix(data_io): make c3d_loader resilient to sidekick.lab.bio test pollution

### DL-#13001 · Reject Unauthenticated Diagnostics in Remote Mode, Redact Absolute Repo Root, and Measure Runtime Probes

- **State:** shipped
- **Shipped:** 2026-10-18
- **Owner:** unassigned
- **Issue:** #13001
- **Branch:** orch/13001-agy
- **PR:** #13268
- **Paths:** see #13268
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`81b34684`; collated from changes/13001-reject-unauthenticated-diagnostics-in-re.md)
- **Summary:** Reject unauthenticated diagnostics in remote mode, redact absolute repo root, and measure runtime probes

### DL-#13104 · OSL-16: Bar-Loaded Lifts Track at 64 RK4 Substeps With a 0.25 ms Effort Hold; the Loaded Feet Stay Within the LIFT-5 Ankle Bounds and the Re-Track and Full-Window Bar-Path Parity Close

- **State:** shipped
- **Shipped:** 2026-10-18
- **Owner:** unassigned
- **Issue:** #13104
- **Branch:** fix/13104-loaded-foot-stability
- **PR:** #13047
- **Paths:** see #13047
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`7fed54d1`; collated from changes/13104-osl-16-bar-loaded-lifts-track-at-64-rk4.md)
- **Summary:** OSL-16: bar-loaded lifts track at 64 RK4 substeps with a 0.25 ms effort hold; the loaded feet stay within the LIFT-5 ankle bounds and the re-track and full-window bar-path parity close

### DL-#13070 · Add the Family-Wise Plate Consistency Screen and Plate Observation Provider That Eliminate Candidate Swing Solutions Inconsistent With Vertical Force-Plate Data (OF-12)

- **State:** shipped
- **Shipped:** 2026-10-17
- **Owner:** unassigned
- **Issue:** #13070
- **Branch:** feat/13070-of-12-plate-screen
- **PR:** #13079
- **Paths:** see #13079
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`a9998da0`; collated from changes/13070-add-the-family-wise-plate-consistency-sc.md)
- **Summary:** Add the family-wise plate consistency screen and plate observation provider that eliminate candidate swing solutions inconsistent with vertical force-plate data (OF-12)

### DL-#13088 · Add Optional Live Open-Fp Bridge to Realtime IPC (OF-09)

- **State:** shipped
- **Shipped:** 2026-10-17
- **Owner:** unassigned
- **Issue:** #13088
- **Branch:** feat/13088-of-09-live-bridge
- **PR:** #13097
- **Paths:** see #13097
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`cd301b48`; collated from changes/13088-add-optional-live-open-fp-bridge-to-real.md)
- **Summary:** Add optional live open-fp bridge to realtime IPC (OF-09)

### DL-#13087 · Overlay Measured Plate Trials on the Ground Reaction Tile (OF-07)

- **State:** shipped
- **Shipped:** 2026-10-17
- **Owner:** unassigned
- **Issue:** #13087
- **Branch:** feat/13087-of-07-plate-overlay
- **PR:** #13100, #13124
- **Paths:** see #13100
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`c1e4d044`; collated from changes/13087-ground-reaction-tile-asks-for-plate-decl.md)
- **Summary:** Overlay measured plate trials on the ground reaction tile (OF-07)

### DL-#13085 · Open-Fp LAN Client

- **State:** shipped
- **Shipped:** 2026-10-17
- **Owner:** claude
- **Issue:** #13085
- **Branch:** feat/13085-of-06-lan-client
- **PR:** #13094
- **Paths:** `src/shared/python/force_plates/open_fp_format/client.py`
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`b0c20bdc`; collated from changes/13085-add-read-only-open-fp-lan-client-of-06.md)
- **Summary:** Add read-only open-fp LAN client (OF-06)

### DL-#13084 · Add Force-Event Namespace and Sourced Impact Anchor (OF-04)

- **State:** shipped
- **Shipped:** 2026-10-17
- **Owner:** unassigned
- **Issue:** #13084
- **Branch:** feat/13084-of-04-force-events
- **PR:** #13100
- **Paths:** see #13100
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`181406be`; collated from changes/13084-add-force-event-namespace-and-sourced-im.md)
- **Summary:** Add force-event namespace and sourced impact anchor (OF-04)

### DL-#13106 · Src-Pivot Hygiene Test Accepts the Pivot's Documented Parent Healing, Removing an Order-Dependent Unit-Test-Gate Failure

- **State:** shipped
- **Shipped:** 2026-10-17
- **Owner:** unassigned
- **Issue:** #13106
- **Branch:** fix/13106-src-pivot-healed-parent
- **PR:** #13107, #13118
- **Paths:** see #13107
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`2c2b6698`; collated from changes/13106-fix-order-dependent-keyerror-in-the-pivo.md)
- **Summary:** Src-pivot hygiene test accepts the pivot's documented parent healing, removing an order-dependent unit-test-gate failure

### DL-#13089 · Add Research Proposal for Measured Plate Rows in the Contact-Wrench QP (OF-10)

- **State:** shipped
- **Shipped:** 2026-10-17
- **Owner:** unassigned
- **Issue:** #13089
- **Branch:** docs/13089-of-10-contact-qp-proposal
- **PR:** #13098
- **Paths:** see #13098
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`2bcc5df8`; collated from changes/13089-add-research-proposal-for-measured-plate.md)
- **Summary:** Add research proposal for measured plate rows in the contact-wrench QP (OF-10)

### DL-#13086 · Add Model-Versus-Measured Vertical Comparison in Sigma Units (OF-08)

- **State:** shipped
- **Shipped:** 2026-10-17
- **Owner:** unassigned
- **Issue:** #13086
- **Branch:** feat/13086-of-08-vertical-comparison
- **PR:** #13096
- **Paths:** see #13096
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`ca442c6e`; collated from changes/13086-add-model-versus-measured-vertical-compa.md)
- **Summary:** Add model-versus-measured vertical comparison in sigma units (OF-08)

### DL-#13083 · Derive Plate Placement From a Model Target Frame Without Mirroring (OF-03)

- **State:** shipped
- **Shipped:** 2026-10-17
- **Owner:** unassigned
- **Issue:** #13083
- **Branch:** feat/13083-of-03-model-frame-placement
- **PR:** #13090
- **Paths:** see #13090
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`cb63d205`; collated from changes/13083-derive-plate-placement-from-a-model-targ.md)
- **Summary:** Derive plate placement from a model target frame without mirroring (OF-03)

### DL-#13069 · Add Plate Readings at Model Time With an Interpolation Bias Estimate, Skew Term, World-Axis CoP Uncertainty and Common-Mode Clock-Offset Alignment (OF-05)

- **State:** shipped
- **Shipped:** 2026-10-17
- **Owner:** unassigned
- **Issue:** #13069
- **Branch:** feat/13069-of-05-plate-readings
- **PR:** #13096
- **Paths:** see #13096
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`ca442c6e`; collated from changes/13069-add-plate-readings-at-model-time-with-in.md)
- **Summary:** Add plate readings at model time with an interpolation bias estimate, skew term, world-axis CoP uncertainty and common-mode clock-offset alignment (OF-05)

### DL-#13067 · Add Per-Foot Vertical-Cell Force-Plate Observation Rows With the Exact Sensing-Plane Shear Lever to the Ground Null Space, With Sympy Goldens and Spec (OF-11)

- **State:** shipped
- **Shipped:** 2026-10-17
- **Owner:** unassigned
- **Issue:** #13067
- **Branch:** feat/13067-of-11-vertical-plate-rows
- **PR:** #13077
- **Paths:** see #13077
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`4aa8f701`; collated from changes/13067-add-per-foot-vertical-cell-force-plate-o.md)
- **Summary:** Add per-foot vertical-cell force-plate observation rows with the exact sensing-plane shear lever to the ground null space, with sympy goldens and spec (OF-11)

### DL-#13066 · Add Fail-Closed Open-Fp Swing File Reader (JSON+NPZ, CSV) Producing VerticalPlateTrial (OF-02)

- **State:** shipped
- **Shipped:** 2026-10-17
- **Owner:** unassigned
- **Issue:** #13066
- **Branch:** feat/13066-of-02-open-fp-reader
- **PR:** #13096
- **Paths:** see #13096
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`ca442c6e`; collated from changes/13066-add-fail-closed-open-fp-swing-file-reade.md)
- **Summary:** Add fail-closed open-fp swing file reader (JSON+NPZ, CSV) producing VerticalPlateTrial (OF-02)

### DL-#13065 · Add Force_Plates Package: Vertical-Only Two-Plate Trial Contract, Ground-Aligned Placement and Observed Ground-Reaction View (OF-01)

- **State:** shipped
- **Shipped:** 2026-10-17
- **Owner:** unassigned
- **Issue:** #13065
- **Branch:** feat/13065-of-01-vertical-plate-trial
- **PR:** #13090
- **Paths:** see #13090
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`cb63d205`; collated from changes/13065-add-force-plates-package-vertical-only-t.md)
- **Summary:** Add force_plates package: vertical-only two-plate trial contract, ground-aligned placement and observed ground-reaction view (OF-01)

### DL-#13061 · Evaluate Open-Fp Force Plates and Add the Vertical-Only Integration Plan (Licence Boundary, Data Contract, Frames, Timing, Backlog)

- **State:** shipped
- **Shipped:** 2026-10-17
- **Owner:** unassigned
- **Issue:** #13061
- **Branch:** claude/upstreamdrift-integration-plans-nlivq0
- **PR:** #13062, #13080
- **Paths:** see #13062
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`73a62ea5`; collated from changes/13061-of-plan-update.md)
- **Summary:** Evaluate open-fp force plates and add the vertical-only integration plan (licence boundary, data contract, frames, timing, backlog)

### DL-#12691 · R11: The R6 Chain Shaft Runs in OpenSim (SpringGeneralizedForce) and MyoSuite (MuJoCo Export) With Static, Modal, Topology and Same-Input Parity Gates; Chain Bundles Export With --Shaft-Model Chain

- **State:** shipped
- **Shipped:** 2026-10-17
- **Owner:** unassigned
- **Issue:** #12691
- **Branch:** claude/12691-opensim-flex-shaft
- **PR:** #12951
- **Paths:** see #12951
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`aa0c176b`; collated from changes/12691-r11-the-r6-chain-shaft-runs-in-opensim-s.md)
- **Summary:** R11: the R6 chain shaft runs in OpenSim (SpringGeneralizedForce) and MyoSuite (MuJoCo export) with static, modal, topology and same-input parity gates; chain bundles export with --shaft-model chain

### DL-#12960 · Adversarial Review 2026-10-16: 43 Findings With Draft PR Pitches Under Docs/Development/Adversarial-Review/

- **State:** shipped
- **Shipped:** 2026-10-16
- **Owner:** unassigned
- **Issue:** #12960
- **Branch:** claude/adversarial-review-2026-10-16
- **PR:** #12960
- **Paths:** docs/development/adversarial-review/2026-10-16
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`f249f9c5`; collated from changes/12960-adversarial-review-2026-10-16.md)
- **Summary:** Adversarial review 2026-10-16: 43 findings with draft PR pitches under docs/development/adversarial-review/

### DL-#12686 · R6: Opt-In Flexible Golf Shaft as a Rigid-Link Spring-Damper Chain (MuJoCo); Generic STIFF Profile Flagged Default_Not_Measured; Rigid Default Byte-Identical; Other Engines Refuse Chain Specs

- **State:** shipped
- **Shipped:** 2026-10-20
- **Owner:** unassigned
- **Issue:** #12686
- **Branch:** feat/12686-flexible-shaft-mujoco
- **PR:** #12734, #13486
- **Paths:** see #12734
- **Started:** 2026-10-16
- **Last verified:** 2026-10-20 (`069023b1`; collated from changes/12686-r6-criterion-6-mujoco-visual-and-appeara.md)
- **Summary:** R6: opt-in flexible golf shaft as a rigid-link spring-damper chain (MuJoCo); generic STIFF profile flagged default_not_measured; rigid default byte-identical; other engines refuse chain specs

### DL-#12784 · Nightly motion-matching-smoke.yml Runs Tests/Integration/Motion_Matching (Slow, --Timeout=0) via Scripts/Ci/Run_Motion_Matching_Smoke.Sh on the Fleet; a Failure Opens or Updates an Issue and the Red Scheduled Run Reaches Fleet-Main-Health; Not Coupled to Heavy-Tests-Opt-In Parity

- **State:** shipped
- **Shipped:** 2026-10-15
- **Owner:** unassigned
- **Issue:** #12784
- **Branch:** ci/12784-motion-matching-smoke-lane
- **PR:** #12786
- **Paths:** see #12786
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`c4b15957`; collated from changes/12784-motion-matching-smoke-lane.md)
- **Summary:** Nightly motion-matching-smoke.yml runs tests/integration/motion_matching (slow, --timeout=0) via scripts/ci/run_motion_matching_smoke.sh on the fleet; a failure opens or updates an issue and the red scheduled run reaches fleet-main-health; not coupled to heavy-tests-opt-in parity

### DL-#12702 · Fix the 'Spec Has No Club Body' Crash on the MuJoCo-Native Driver Run: Assembly_From_Spec Now Recognises the Native Simscape Club (Segmented 2.5in Butt Grip, No Grip Solid) so the Default V2 Spec Carries a Club Body; Lane.Set_Club_Targets Fails Fast via Club_Face_Target.Require_Club_Assembly Naming the Spec Source. the Smoke Test Now Gets Past Set_Club_Targets but Still Fails on Dynamics Marker RMS (0.51 m vs 0.15 m Bound), a Separate Regression; the Smoke Is Marked Slow and Runs in No CI Lane

- **State:** shipped
- **Shipped:** 2026-10-15
- **Owner:** unassigned
- **Issue:** #12702
- **Branch:** fix/12702-mujoco-smoke-club-body
- **PR:** #12776, #12783
- **Paths:** see #12776
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`789e583f`; collated from changes/12702-dynamics-rms-smoke-spec.md)
- **Summary:** Fix the 'spec has no club body' crash on the MuJoCo-native driver run: assembly_from_spec now recognises the native Simscape club (segmented 2.5in Butt grip, no Grip solid) so the default v2 spec carries a club body; Lane.set_club_targets fails fast via club_face_target.require_club_assembly naming the spec source. The smoke test now gets past set_club_targets but still fails on dynamics marker RMS (0.51 m vs 0.15 m bound), a separate regression; the smoke is marked slow and runs in no CI lane

### DL-#12683 · R3: Trail-Foot Toe-Out Owner Style Target (TRAIL_TOE_OUT_STYLE_DEG = 20 Deg, Single-Sourced From DEFAULT_TOE_OUT_DEG) via a New --Foot-Progression Style Mode, Now the Canonical Default; Capture Mode Still Lets the Measurement Win; Per-Foot Target_Source in Receipts; Driver Trail -2.9 -> 19.8 Deg, 7-Iron -9.8 -> 13.5 Deg (Hip_Rotation_R at Its -40 Deg Limit)

- **State:** shipped
- **Shipped:** 2026-10-20
- **Owner:** unassigned
- **Issue:** #12683
- **Branch:** fix/12683-trail-toe-out
- **PR:** #12731, #13482
- **Paths:** see #12731
- **Started:** 2026-10-15
- **Last verified:** 2026-10-20 (`1562f63d`; collated from changes/12683-pin-the-trail-foot-toe-out-on-the-commit.md)
- **Summary:** R3: trail-foot toe-out owner style target (TRAIL_TOE_OUT_STYLE_DEG = 20 deg, single-sourced from DEFAULT_TOE_OUT_DEG) via a new --foot-progression style mode, now the canonical default; capture mode still lets the measurement win; per-foot target_source in receipts; driver trail -2.9 -> 19.8 deg, 7-iron -9.8 -> 13.5 deg (hip_rotation_r at its -40 deg limit)

### DL-#12682 · R5: One Clubhead Measurement Point (Face Centre via Club_Face_Target.Face_Centre_Path) for the Fd_Phase Impact Split, the Exporter Impact Detector and a New Dynamics.Fd_Phase.Clubhead_Speed Receipt Block (Simulated/Reference/Capture Speed, % Error, Peak-Timing Offset); Scripts/diagnose_clubhead_speed.py Decomposition

- **State:** shipped
- **Shipped:** 2026-10-15
- **Owner:** unassigned
- **Issue:** #12682
- **Branch:** fix/12682-clubhead-speed-parity
- **PR:** #12695
- **Paths:** see #12695
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`0081a09b`; collated from changes/12682-r5-one-clubhead-measurement-point-face-c.md)
- **Summary:** R5: one clubhead measurement point (face centre via club_face_target.face_centre_path) for the fd_phase impact split, the exporter impact detector and a new dynamics.fd_phase.clubhead_speed receipt block (simulated/reference/capture speed, % error, peak-timing offset); scripts/diagnose_clubhead_speed.py decomposition

### DL-#12413 · \ Restore DBC Level in Test_Ud_Contract_Level_Alias_Sync to Prevent Xdist Pollution\

- **State:** shipped
- **Shipped:** 2026-10-12
- **Owner:** unassigned
- **Issue:** #12413
- **Branch:** fix/12413-flaky-test-missing-columns-contract-level
- **PR:** #12414
- **Paths:** see #12414
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`2202a2b3`; collated from changes/12413-restore-dbc-level-in-test-ud-contract-le.md)
- **Summary:** \ Restore DBC level in test_ud_contract_level_alias_sync to prevent xdist pollution\

### DL-#12405 · Reconcile Package Collision Gate With Initialized Tools Provider Layout, Distinguishing Verified Canonical Aliases and Seam Counterparts While Eliminating Redundant Nested Tools Roots From Pythonpath.

- **State:** shipped
- **Shipped:** 2026-10-13
- **Owner:** unassigned
- **Issue:** #12405
- **Branch:** fix/12405-reconcile-package-collision-gate
- **PR:** #12419
- **Paths:** see #12419
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`7b74513b`; collated from changes/12405-reconcile-package-collision-gate-with-in.md)
- **Summary:** Reconcile package collision gate with initialized Tools provider layout, distinguishing verified canonical aliases and seam counterparts while eliminating redundant nested Tools roots from pythonpath.

### DL-#12404 · Reconcile Optimized Contract Defaults Across Provider Boundaries in Test Probes Without Creating an Unapproved Public Contracts Fork.

- **State:** shipped
- **Shipped:** 2026-10-13
- **Owner:** unassigned
- **Issue:** #12404
- **Branch:** fix/12404-reconcile-optimized-contract-defaults
- **PR:** #12426
- **Paths:** see #12426
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`d56b30b8`; collated from changes/12404-reconcile-optimized-contract-defaults-ac.md)
- **Summary:** Reconcile optimized contract defaults across provider boundaries in test probes without creating an unapproved public contracts fork.

### DL-#12353 · Allow Native Moco Preparation to Skip Only the Short Capability Probe While Recording Runtime as Not Attempted and Keeping Solve Readiness False

- **State:** shipped
- **Shipped:** 2026-10-13
- **Owner:** unassigned
- **Issue:** #12353
- **Branch:** feat/native-no-probe-12353-20261012
- **PR:** #12367
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/native_moco_preparation.py,src/engines/physics_engines/opensim/python/tour_matching/native_body_inertia.py,src/engines/physics_engines/opensim/python/tour_matching/native_passive_receipt.py,src/engines/physics_engines/opensim/python/tour_matching/cli.py,tests/opensim/test_native_passive_receipt.py,tests/opensim/test_native_moco_inertia_preflight.py,tests/unit/engines/opensim/test_native_moco_probe_mode.py,tests/unit/engines/opensim/test_native_path_optimizer_domains.py,manuals/upstreamdrift/chapters/56-native-diagnostic-execution-contract.qmd,manuals/upstreamdrift/calculation-registry.json,SPEC.md,AGENT_HANDOFF.md,docs/development/feedback_controls/F07_NATIVE_NO_PROBE_PREPARATION_TURNOVER.md,docs/agent_context/README.md,docs/agent_context/index.html
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`56c9b40e`; collated from changes/12353-native-moco-skip-capability-probe.md)
- **Summary:** Allow native Moco preparation to skip only the short capability probe while recording runtime as not attempted and keeping solve readiness false

### DL-#12319 · Putt Scene Export for Public Explorer

- **State:** shipped
- **Shipped:** 2026-10-12
- **Owner:** claude
- **Issue:** #12319
- **Branch:** feat/12319-putt-export
- **PR:** #12335
- **Paths:** `src/tools/putting_green_gui/_scene_export.py`
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`178b32b6`; collated from changes/12319-export-the-3-d-putt-simulation-as-the-ud.md)
- **Summary:** Export the 3-D putt simulation as the ud.putt_scene/1 mirror artifact for the public explorer Putting tab

### DL-#12317 · Force Redistribution Reports Optimizer Convergence Only When the Optimizer Reports Success, Independently of the Candidate Feasibility Audit.

- **State:** shipped
- **Shipped:** 2026-10-12
- **Owner:** unassigned
- **Issue:** #12317
- **Branch:** fix/force-allocation-convergence-12317
- **PR:** #12322
- **Paths:** src/shared/python/motion_matching/force_nullspace.py,tests/unit/motion_matching/test_force_nullspace.py,docs/design/force_redistribution_convergence.md,docs/motion_matching/README.md,docs/development/force_allocation_convergence_12317_turnover.md
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ed672ef3`; collated from changes/12317-force-allocation-convergence.md)
- **Summary:** Force redistribution reports optimizer convergence only when the optimizer reports success, independently of the candidate feasibility audit.

### DL-#12315 · Widen the Golf Models' Forearm Pronation Range to the AAOS Normative 80 Deg From Neutral, From One Shared Source

- **State:** shipped
- **Shipped:** 2026-10-14
- **Owner:** unassigned
- **Issue:** #12315
- **Branch:** fix/12315-forearm-pronation-range
- **PR:** #12484
- **Paths:** see #12484
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`6d665f59`; collated from changes/12315-widen-the-golf-models-forearm-pronation.md)
- **Summary:** Widen the golf models' forearm pronation range to the AAOS normative 80 deg from neutral, from one shared source

### DL-#12309 · Migrate Final Figure Builder Calls to FigureOptions and Verify Real Writer Integration and Intended-Output Parity

- **State:** shipped
- **Shipped:** 2026-10-13
- **Owner:** unassigned
- **Issue:** #12309
- **Branch:** codex/shot-pattern-safe-followup
- **PR:** #12418, #12445
- **Paths:** scripts/research/shot_pattern_analysis/build_final_figures.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`9f60d7c4`; collated from changes/12309-single-figureoptions-mapping-helper-for.md)
- **Summary:** Migrate final figure builder calls to FigureOptions and verify real writer integration and intended-output parity

### DL-#12302 · Refactor Shot-Pattern Duplication and Architecture Debt While Preserving Frozen Research Evidence

- **State:** shipped
- **Shipped:** 2026-10-13
- **Owner:** unassigned
- **Issue:** #12302
- **Branch:** codex/shot-pattern-refinement-integration
- **PR:** #12418
- **Paths:** see #12418
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/12302-refactor-shot-pattern-duplication-and-ar.md)
- **Summary:** Refactor shot-pattern duplication and architecture debt while preserving frozen research evidence

### DL-#12299 · Reconcile Shot-Pattern Integration and Repair Inherited Validation Blockers While Preserving Frozen Science and the Approved DRY Baseline

- **State:** shipped
- **Shipped:** 2026-10-13
- **Owner:** unassigned
- **Issue:** #12299
- **Branch:** codex/shot-pattern-safe-followup
- **PR:** #12418
- **Paths:** see #12418
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/12299-restore-multi-runtime-dry-baseline.md)
- **Summary:** Reconcile shot-pattern integration and repair inherited validation blockers while preserving frozen science and the approved DRY baseline

### DL-#12280 · Admit Exact Zero-Coordinate Structural WeldJoint in Constrained Muscle Replay

- **State:** shipped
- **Shipped:** 2026-10-11
- **Owner:** unassigned
- **Issue:** #12280
- **Branch:** feat/opensim-weldjoint-12280
- **PR:** #12282
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/native_constrained_muscle.py;tests/opensim/test_native_constrained_weld_policy.py;tests/opensim/test_native_constrained_muscle.py;manuals/upstreamdrift/chapters/59-native-structural-weld-admission.qmd;manuals/upstreamdrift/index.qmd;docs/development/feedback_controls/DESIGN.md;docs/development/feedback_controls/F07_WELDJOINT_ADMISSION_TURNOVER.md;docs/development/feedback_controls/evidence/native_weldjoint_12280/child-tests.json;docs/development/feedback_controls/evidence/native_weldjoint_12280/junit.xml;docs/development/feedback_controls/evidence/native_weldjoint_12280/manifest.json
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`55f93c42`; collated from changes/12280-native-weldjoint-admission.md)
- **Summary:** Admit exact zero-coordinate structural WeldJoint in constrained muscle replay

### DL-#12275 · Retain Hash-Bound Manufactured Native Diagnostics

- **State:** shipped
- **Shipped:** 2026-10-11
- **Owner:** unassigned
- **Issue:** #12275
- **Branch:** docs/manufactured-feedback-evidence-12275
- **PR:** #12277
- **Paths:** see #12277
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`7016d485`; collated from changes/12275-retain-manufactured-native-diagnostic-evidence.md)
- **Summary:** Retain hash-bound manufactured native diagnostics

### DL-#12271 · Add Hash-Bound One-Pass Nonuniform Moco Mesh Refinement Planning

- **State:** shipped
- **Shipped:** 2026-10-11
- **Owner:** unassigned
- **Issue:** #12271
- **Branch:** feat/f07-native-component-audit-budget-12244
- **PR:** #12273
- **Paths:** see #12273
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`60598915`; collated from changes/12271-native-moco-history-appendices.md)
- **Summary:** Add hash-bound one-pass nonuniform Moco mesh refinement planning

### DL-#12249 · MyoSuite Native Muscle Actuator Classification

- **State:** shipped
- **Shipped:** 2026-10-11
- **Owner:** unassigned
- **Issue:** #12249
- **Branch:** fix/12249-native-muscle-actuator-enum
- **PR:** #12252
- **Paths:** src/engines/physics_engines/myosuite/python/muscle_analysis.py;tests/unit/engines/myosuite/test_muscle_analysis.py;manuals/upstreamdrift/chapters/31-myosuite-native-excitation-replay.qmd;docs/development/feedback_controls/F09Q-MYOSUITE-ACTUATOR-CLASSIFICATION-TURNOVER.md
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0870e093`; collated from changes/12249-native-muscle-actuator-classification.md)
- **Summary:** Use MuJoCo's named muscle enum for native MyoSuite actuator classification.

### DL-#12243 · Repair Matched-Supplement Density Figure Clearance

- **State:** shipped
- **Shipped:** 2026-10-13
- **Owner:** unassigned
- **Issue:** #12243
- **Branch:** codex/shot-pattern-main-integration
- **PR:** #12418
- **Paths:** src/tools/shot_pattern_analysis/matched_supplement_figures.py,tests/tools/shot_pattern_analysis/test_matched_supplement_pipeline.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/12243-increase-matched-supplement-density-figu.md)
- **Summary:** Increase matched-supplement density figure gutters to clear legends, panel titles, and colorbars.

### DL-#12242 · Keep Research PDF Headings With Their Following Tables and Paragraphs

- **State:** shipped
- **Shipped:** 2026-10-13
- **Owner:** unassigned
- **Issue:** #12242
- **Branch:** codex/shot-pattern-main-integration
- **PR:** #12418
- **Paths:** see #12418
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/12242-keep-research-pdf-headings-with-their-fo.md)
- **Summary:** Keep Research PDF Headings With Their Following Tables and Paragraphs

### DL-#12236 · Add Source-Bound Native Diagnostic Admission, Runtime, and Qualification Contract.

- **State:** shipped
- **Shipped:** 2026-10-11
- **Owner:** unassigned
- **Issue:** #12236
- **Branch:** feat/native-diagnostic-contract-12236
- **PR:** #12250
- **Paths:** see #12250
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`f131464b`; collated from changes/12236-add-source-bound-native-diagnostic-admis.md)
- **Summary:** Add source-bound native diagnostic admission, runtime, and qualification contract.

### DL-#12233 · Bound Moco Optimizer States to Source Moving-Path Function Domains

- **State:** shipped
- **Shipped:** 2026-10-12
- **Owner:** unassigned
- **Issue:** #12233
- **Branch:** fix/native-path-connection-finalization-12233-20261012
- **PR:** #12318, #12228
- **Paths:** see #12318
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`49ef898b`; collated from changes/12233-source-bound-moving-path-chart-audit.md)
- **Summary:** Bound Moco optimizer states to source moving-path function domains

### DL-#12231 · Bind Native OpenSim Visual Meshes to a Strict Source-Owned Resource Closure

- **State:** shipped
- **Shipped:** 2026-10-11
- **Owner:** unassigned
- **Issue:** #12231
- **Branch:** fix/12231-source-bound-visual-assets
- **PR:** #12232
- **Paths:** .prettierignore,src/engines/physics_engines/opensim/python/tour_matching/native_visual_resources.py,src/engines/physics_engines/opensim/python/tour_matching/native_muscle_bundle.py,tests/opensim/test_native_muscle_bundle.py,manuals/upstreamdrift/chapters/13-native-muscle-replay.qmd,docs/agent_context/README.md,docs/agent_context/index.html,src/engines/native_replay_contracts.py,src/engines/physics_engines/myosuite/python/native_direct_model_replay.py,tests/unit/engines/test_native_replay_contracts.py,docs/development/feedback_controls/F07_VISUAL_RESOURCE_CLOSURE_TURNOVER.md
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`b2ff2d1c`; collated from changes/12231-bind-native-opensim-visual-meshes-to-a-s.md)
- **Summary:** Bind native OpenSim visual meshes to a strict source-owned resource closure

### DL-#12225 · Explicitly Unqualified Native Moco Feasibility

- **State:** shipped
- **Shipped:** 2026-10-11
- **Owner:** codex
- **Issue:** #12225
- **Branch:** feat/f07-unqualified-moco-feasibility-12225
- **PR:** #12228
- **Paths:** see #12228
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`49ef898b`; collated from changes/12225-unqualified-moco-feasibility.md)
- **Summary:** Keep explicitly unqualified native Moco feasibility separate from solve convergence and failed or incomplete independent replay

### DL-#12222 · Club-Face Fixtures Keep the Capture Address Toe-Out: The Swing IK and Every Re-Solve Plant the Stance at the Calibrated Address When Foot Progression Is on; Fixtures Regenerated With --Foot-Progression Capture (Driver 16.64/3.57 vs 16.37/4.03 Deg, 7-Iron 15.43/-0.96 vs 15.00/-0.37 Deg)

- **State:** shipped
- **Shipped:** 2026-10-11
- **Owner:** unassigned
- **Issue:** #12222
- **Branch:** claude/12222-fixture-toe-out
- **PR:** #12230
- **Paths:** see #12230
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`9e5f87c4`; collated from changes/12222-club-face-fixtures-keep-the-capture-addr.md)
- **Summary:** Club-face fixtures keep the capture address toe-out: the swing IK and every re-solve plant the stance at the calibrated address when foot progression is on; fixtures regenerated with --foot-progression capture (driver 16.64/3.57 vs 16.37/4.03 deg, 7-iron 15.43/-0.96 vs 15.00/-0.37 deg)

### DL-#12221 · Adopt the Per-Capture Grip Rotation on the Fixed Driver Club Cluster and Regenerate the Three Driver Ground-Support Receipts, the Driver OpenSim Export, the Driver Club-Face Fixture (Meets GCV-20) and the MSK Driver Grip Calibration; the 7-Iron Refit Is Deferred Because Its Same-Input Export Fails GCV-20 and OSV-10

- **State:** shipped
- **Shipped:** 2026-10-11
- **Owner:** unassigned
- **Issue:** #12221
- **Branch:** claude/12221-grip-adopt
- **PR:** #12274, #12265, #12270, #12251
- **Paths:** see #12274
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0f0d7ee7`; collated from changes/12221-opt-in-fix-club-markers-keeps-the-club-c.md)
- **Summary:** Adopt the per-capture grip rotation on the fixed driver club cluster and regenerate the three driver ground-support receipts, the driver OpenSim export, the driver club-face fixture (meets GCV-20) and the MSK driver grip calibration; the 7-iron refit is deferred because its same-input export fails GCV-20 and OSV-10

### DL-#12211 · Generic Shoulder Triangle-Lamina Candidate

- **State:** shipped
- **Shipped:** 2026-10-11
- **Owner:** codex
- **Issue:** #12211
- **Branch:** fix/f07-parent-receipt-12198
- **PR:** #12198
- **Paths:** scripts/opensim/triangle_lamina_provider_worker.py,scripts/opensim/derive_generic_shoulder_lamina.py,src/engines/physics_engines/opensim/python/tour_matching/generic_shoulder_lamina.py,src/engines/physics_engines/opensim/python/tour_matching/native_body_inertia.py,src/shared/python/core/physical_inertia.py,src/shared/python/estimation/dime_global_calibration.py,tests/opensim/test_generic_shoulder_lamina.py,tests/unit/core/test_physical_inertia.py,docs/development/feedback_controls/F07_GENERIC_SHOULDER_LAMINA_TURNOVER.md,manuals/upstreamdrift/chapters/54-generic-shoulder-triangle-lamina.qmd,changes/12211-add-a-separately-identified-source-bound.md,src/config/impact_acceptance.json,docs/development/impact_acceptance_matrix.md,docs/development/feedback_controls/F07_PROVIDER_12246_IMPACT_SERVED_RECEIPT.json
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`7b925718`; collated from changes/12211-add-a-separately-identified-source-bound.md)
- **Summary:** Add a separately identified source-bound generic shoulder triangle-lamina candidate with native physical and sampled kinematic checks

### DL-#12210 · Give the 7-Iron Anthropometric Document Its Own Clubhead Marker Cluster (Per-Club Club-Markers File Read by the Builder); Head-Triad Residual on the 7-Iron Capture 42.4 to 0.24 mm; Both 7-Iron Receipts Regenerated (Nominal IK 67.9 to 45.2 mm; Calibrated Dynamics 54.5 to 109.9 mm, Owner Decision Pending)

- **State:** shipped
- **Shipped:** 2026-10-11
- **Owner:** unassigned
- **Issue:** #12210
- **Branch:** claude/12210-iron-head-triad
- **PR:** #12219
- **Paths:** see #12219
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`4cd2991b`; collated from changes/12210-give-the-7-iron-anthropometric-document.md)
- **Summary:** Give the 7-iron anthropometric document its own clubhead marker cluster (per-club club-markers file read by the builder); head-triad residual on the 7-iron capture 42.4 to 0.24 mm; both 7-iron receipts regenerated (nominal IK 67.9 to 45.2 mm; calibrated dynamics 54.5 to 109.9 mm, owner decision pending)

### DL-#12205 · Require Native Source-Body Physical Admission Before OpenSim Moco Solve

- **State:** shipped
- **Shipped:** 2026-10-11
- **Owner:** unassigned
- **Issue:** #12205
- **Branch:** feat/f07-moco-inertia-preflight-12205
- **PR:** #12198
- **Paths:** see #12198
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`7b925718`; collated from changes/12205-native-moco-physical-preflight.md)
- **Summary:** Require native source-body physical admission before OpenSim Moco solve

### DL-#12203 · Make the Engine-Module Teardown Guard Tolerate a Concurrently Removed Package Binding

- **State:** shipped
- **Shipped:** 2026-10-11
- **Owner:** unassigned
- **Issue:** #12203
- **Branch:** fix/f09-drake-import-laziness-12203
- **PR:** #12202
- **Paths:** see #12202
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`e42e9307`; collated from changes/12203-make-the-engine-module-teardown-guard-to.md)
- **Summary:** Make the engine-module teardown guard tolerate a concurrently removed package binding

### DL-#12191 · Bind Exact OpenSim Body Properties to Native Physical-Inertia Admission

- **State:** shipped
- **Shipped:** 2026-10-11
- **Owner:** unassigned
- **Issue:** #12191
- **Branch:** feat/f07-native-body-inertia-12191
- **PR:** #12198
- **Paths:** see #12198
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`7b925718`; collated from changes/12191-native-source-body-inertia.md)
- **Summary:** Bind exact OpenSim Body properties to native physical-inertia admission

### DL-#12178 · Exercise Calibrated Native Marker Replay Through Existing Observation Scoring

- **State:** shipped
- **Shipped:** 2026-10-11
- **Owner:** unassigned
- **Issue:** #12178
- **Branch:** feat/f09-calibrated-marker-score-12178
- **PR:** #12288
- **Paths:** see #12288
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`9bc8e05f`; collated from changes/12178-calibrated-marker-replay-score.md)
- **Summary:** Exercise calibrated native marker replay through existing observation scoring

### DL-#12176 · Verify Exact Zero-MTP CustomJoint Reduction on Assembled 557-Muscle Native Source

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** unassigned
- **Issue:** #12176
- **Branch:** feat/f07-buet-zero-custom-mtp-12176
- **PR:** #12181
- **Paths:** see #12181
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`cd25e8ab`; collated from changes/12176-source-bound-zero-custom-mtp-reduction.md)
- **Summary:** Verify exact zero-MTP CustomJoint reduction on assembled 557-muscle native source

### DL-#12173 · Replay Constrained Mixed OpenSim Moco Inputs Without Hiding Native State Projection

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** unassigned
- **Issue:** #12173
- **Branch:** feat/f07-constrained-mixed-native-12173
- **PR:** #12184
- **Paths:** see #12184
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`6af5715f`; collated from changes/12173-constrained-mixed-native-moco.md)
- **Summary:** Replay Constrained Mixed OpenSim Moco Inputs Without Hiding Native State Projection

### DL-#12167 · Verify Exact Zero-Subtalar CustomJoint Reduction With Sampled Native 557-Muscle Mechanics

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** unassigned
- **Issue:** #12167
- **Branch:** feat/f07-buet-zero-subtalar-12167
- **PR:** #12172
- **Paths:** see #12172
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`1bef5534`; collated from changes/12167-verify-exact-zero-subtalar-customjoint-r.md)
- **Summary:** Verify exact zero-subtalar CustomJoint reduction with sampled native 557-muscle mechanics

### DL-#12164 · Add Source-Bound Provisional Marker Calibration Artifacts

- **State:** shipped
- **Shipped:** 2026-10-11
- **Owner:** unassigned
- **Issue:** #12164
- **Branch:** feat/f09-marker-calibration-12164
- **PR:** #12288
- **Paths:** see #12288
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`9bc8e05f`; collated from changes/12164-add-source-bound-provisional-marker-cali.md)
- **Summary:** Add source-bound provisional marker calibration artifacts

### DL-#12160 · Qualify Shot-Pattern Validation Helpers for Repository Gates

- **State:** shipped
- **Shipped:** 2026-10-13
- **Owner:** unassigned
- **Issue:** #12160
- **Branch:** codex/shot-pattern-main-integration
- **PR:** #12418
- **Paths:** see #12418
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/12160-qualify-shot-pattern-validation-helpers.md)
- **Summary:** Qualify shot-pattern validation helpers for repository gates

### DL-#12159 · Correct Phase-Two Article Source Fidelity and Unsupported Review Claims

- **State:** shipped
- **Shipped:** 2026-10-14
- **Owner:** unassigned
- **Issue:** #12159
- **Branch:** codex/shot-pattern-matched-delivery
- **PR:** #12418, #12480
- **Paths:** see #12418
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`639f5570`; collated from changes/12159-reconcile-shot-pattern-analysis-paper-pr.md)
- **Summary:** Correct phase-two article source fidelity and unsupported review claims

### DL-#12158 · Bind Observation Scores to Native Marker Replay Evidence

- **State:** shipped
- **Shipped:** 2026-10-11
- **Owner:** unassigned
- **Issue:** #12158
- **Branch:** feat/f09-observation-receipts-12158
- **PR:** #12288
- **Paths:** see #12288
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`9bc8e05f`; collated from changes/12158-f09-native-marker-score-binding.md)
- **Summary:** Bind observation scores to native marker replay evidence

### DL-#12148 · Report OpenSim Golf Muscle Support Only From Source-Bound Native Readback

- **State:** shipped
- **Shipped:** 2026-10-11
- **Owner:** unassigned
- **Issue:** #12148
- **Branch:** feat/f09-opensim-native-muscle-capability-12148
- **PR:** #12288
- **Paths:** see #12288
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`9bc8e05f`; collated from changes/12148-report-opensim-golf-muscle-support-only.md)
- **Summary:** Report OpenSim golf muscle support only from source-bound native readback

### DL-#12143 · Bind Declared Native Constrained Muscle States to Moco and Independent Excitation Replay

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** unassigned
- **Issue:** #12143
- **Branch:** feat/f07-constrained-muscle-replay-12143
- **PR:** #12163
- **Paths:** see #12163
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5e5757ef`; collated from changes/12143-constrained-native-muscle-moco-replay.md)
- **Summary:** Bind declared native constrained muscle states to Moco and independent excitation replay

### DL-#12136 · Declare and Verify Native OpenSim Constrained Cold Starts With Exact Mechanical Replay Diagnostics

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** unassigned
- **Issue:** #12136
- **Branch:** feat/f07-constrained-cold-start-12136
- **PR:** #12163
- **Paths:** see #12163
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5e5757ef`; collated from changes/12136-declare-and-verify-native-opensim-constr.md)
- **Summary:** Declare and verify native OpenSim constrained cold starts with exact mechanical replay diagnostics

### DL-#12132 · Covariance Caption Containment and Verified Publication

- **State:** shipped
- **Shipped:** 2026-10-13
- **Owner:** unassigned
- **Issue:** #12132
- **Branch:** codex/shot-pattern-main-integration
- **PR:** #12418
- **Paths:** src/tools/shot_pattern_analysis/matched_supplement_figures.py,scripts/research/shot_pattern_analysis/supplement_publication.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/12132-shot-pattern-caption-publication.md)
- **Summary:** Wrap complete covariance captions and add selective, provenance-checked figure revision publication.

### DL-#12131 · Paired Density Row Spacing

- **State:** shipped
- **Shipped:** 2026-10-13
- **Owner:** unassigned
- **Issue:** #12131
- **Branch:** codex/shot-pattern-main-integration
- **PR:** #12418
- **Paths:** src/tools/shot_pattern_analysis/matched_supplement_figures.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/12131-shot-pattern-density-row-spacing.md)
- **Summary:** Increase paired density row spacing while preserving scientific arrays, limits and population identities.

### DL-#12129 · Keep Report Numerical Tokens Intact Using Compact Condition Keys and Separate Scoring and Population Tables

- **State:** shipped
- **Shipped:** 2026-10-13
- **Owner:** unassigned
- **Issue:** #12129
- **Branch:** codex/shot-pattern-matched-delivery
- **PR:** #12418
- **Paths:** see #12418
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/12129-keep-report-numerical-tokens-intact-usin.md)
- **Summary:** Keep report numerical tokens intact using compact condition keys and separate scoring and population tables

### DL-#12128 · Derive Final Shot-Pattern Conclusions From Qualified Selected Results and Preserve Protocol and Population Identities

- **State:** shipped
- **Shipped:** 2026-10-13
- **Owner:** unassigned
- **Issue:** #12128
- **Branch:** codex/shot-pattern-main-integration
- **PR:** #12418
- **Paths:** see #12418
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/12128-derive-final-shot-pattern-conclusions-fr.md)
- **Summary:** Derive final shot-pattern conclusions from qualified selected results and preserve protocol and population identities

### DL-#12117 · Regenerate the Club-Face Swing Fixtures on the Current Spec (Feet Flat Instead of 9-41 cm up), Fix Impact_Frame's Missing Ball-Radius Check, Check the Address Face Against the Capture's Own Face, Regenerate the OSV-9 MSK Grip Calibration, and Strict-Xfail the 7-Iron Impact Face (#12221)

- **State:** shipped
- **Shipped:** 2026-10-11
- **Owner:** unassigned
- **Issue:** #12117
- **Branch:** claude/12117-club-face-fixtures
- **PR:** #12223, #12145
- **Paths:** see #12223
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`3a63d61c`; collated from changes/12117-split-the-reference-zmp-rates-at-the-bal.md)
- **Summary:** Regenerate the club-face swing fixtures on the current spec (feet flat instead of 9-41 cm up), fix impact_frame's missing ball-radius check, check the address face against the capture's own face, regenerate the OSV-9 MSK grip calibration, and strict-xfail the 7-iron impact face (#12221)

### DL-#12064 · Correct SG Optimizer CLI Integration Invocation

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** unassigned
- **Issue:** #12064
- **Branch:** fix/12064-sg-optimizer-cli-test
- **PR:** #12144
- **Paths:** see #12144
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`7eac477a`; collated from changes/12064-correct-sg-optimizer-cli-integration-inv.md)
- **Summary:** Correct SG optimizer CLI integration invocation

### DL-#12052 · MyoSuite Leg Retarget Maps the Hip Orientation Through the Calibrated Hip Frames and Fixes the Left Knee Sign; MyoSuite Address Feet Within 2 Deg on Both Captures

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** unassigned
- **Issue:** #12052
- **Branch:** claude/osv-6e-myosuite-hip-retarget
- **PR:** #12145
- **Paths:** see #12145
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`3a63d61c`; collated from changes/12052-myosuite-leg-retarget-maps-the-hip-orien.md)
- **Summary:** MyoSuite leg retarget maps the hip orientation through the calibrated hip frames and fixes the left knee sign; MyoSuite address feet within 2 deg on both captures

### DL-#12039 · Drake IK Uses MuJoCo's Principal-Log Weld Rotation Row and Exact Jacobian; Drake and MuJoCo Address Stages Now Agree Within 0.02 mm on Both Tour Captures and the Capture-A Drake Receipt Is Regenerated

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** unassigned
- **Issue:** #12039
- **Branch:** claude/12039-drake-address-parity
- **PR:** #12216
- **Paths:** see #12216
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`7da0a298`; collated from changes/12039-drake-ik-uses-mujoco-s-principal-log-wel.md)
- **Summary:** Drake IK uses MuJoCo's principal-log weld rotation row and exact Jacobian; Drake and MuJoCo address stages now agree within 0.02 mm on both tour captures and the capture-A Drake receipt is regenerated

### DL-#12030 · Fail Closed When the Clubhead Triad Offsets Do Not Match the Captured Triad; the Nominal 7-Iron IK Drops From 264.8 to 67.9 mm (Frame 0 70.2 mm) and Its Dynamics From 1355 to 79.2 mm, With Both Nominal Receipts Regenerated

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** unassigned
- **Issue:** #12030
- **Branch:** claude/12030-nominal-iron-ik
- **PR:** #12208
- **Paths:** see #12208
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`7e105580`; collated from changes/12030-fail-closed-when-the-clubhead-triad-offs.md)
- **Summary:** Fail closed when the clubhead triad offsets do not match the captured triad; the nominal 7-iron IK drops from 264.8 to 67.9 mm (frame 0 70.2 mm) and its dynamics from 1355 to 79.2 mm, with both nominal receipts regenerated

### DL-#12023 · Add Frozen Source-Bound Native Tracking Costs for Mixed-Command Search.

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** unassigned
- **Issue:** #12023
- **Branch:** feat/f09m-native-tracking-objective-12023
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/12023-native-tracking-objective.md)
- **Summary:** Add frozen source-bound native tracking costs for mixed-command search.

### DL-#12021 · Extract One Bounded Candidate Search Loop Behind Finite Cost and Signed Hard Margins

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** unassigned
- **Issue:** #12021
- **Branch:** feat/f02-bounded-callback-12021
- **PR:** #12024
- **Paths:** see #12024
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`e47a21e2`; collated from changes/12021-extract-one-bounded-candidate-search-loo.md)
- **Summary:** Extract one bounded candidate search loop behind finite cost and signed hard margins

### DL-#12015 · Use SciPy's Typed Vector Inequality Constraint for F02 SLSQP and Count Its Preparatory Evaluations Within the Existing Budget

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** unassigned
- **Issue:** #12015
- **Branch:** fix/f02-scipy-typing-12015
- **PR:** #11816
- **Paths:** see #11816
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`1fe02740`; collated from changes/12015-type-f02-slsqp-options-and-inequality-co.md)
- **Summary:** Use SciPy's typed vector inequality constraint for F02 SLSQP and count its preparatory evaluations within the existing budget

### DL-#11976 · Add Matched Delivery, Correlated Noise, Density Graphics and Finite-Head Contact Study

- **State:** shipped
- **Shipped:** 2026-10-13
- **Owner:** codex
- **Issue:** #11976
- **Branch:** codex/shot-pattern-matched-delivery
- **PR:** #12418
- **Paths:** see #12418
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/11976-shot-pattern-remote-restoration.md)
- **Summary:** Add matched delivery, correlated noise, density graphics and finite-head contact study

### DL-#11962 · Observe Exact Native Source Reference Frames and Muscle Path Conventions

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** unassigned
- **Issue:** #11962
- **Branch:** feat/f07-reference-conventions-11962
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11962-native-reference-conventions.md)
- **Summary:** Observe exact native source reference frames and muscle path conventions

### DL-#11955 · Bind Compiled Actuator Command Profiles to Native Replay Execution.

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** unassigned
- **Issue:** #11955
- **Branch:** feat/f01f-compiled-command-admission-11955
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11955-compiled-actuator-command-admission.md)
- **Summary:** Bind compiled actuator command profiles to native replay execution.

### DL-#11950 · Add Bounded Native MyoSuite Model Resource Closure Admission.

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** unassigned
- **Issue:** #11950
- **Branch:** feat/f09h-native-muscle-replay-11950
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11950-add-bounded-native-myosuite-model-resour.md)
- **Summary:** Add bounded native MyoSuite model resource closure admission.

### DL-#11949 · Native Moco Initial-State Bindings

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** codex
- **Issue:** #11949
- **Branch:** feat/feedback-moco-state-binding-11791
- **PR:** #12141
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/moco_initial_bindings.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11949-native-moco-initial-bindings.md)
- **Summary:** Apply complete continuous initial-state and scalar-control bounds before native Moco guess creation

### DL-#11942 · Bind Owned Tools Native Envelopes to Actual R2025b Diagnostic Restart and Changed Future-Force Replay

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** unassigned
- **Issue:** #11942
- **Branch:** feat/feedback-simscape-owned-native-11942
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11942-owned-native-simscape-replay.md)
- **Summary:** Bind owned Tools native envelopes to actual R2025b diagnostic restart and changed future-force replay

### DL-#11939 · Require Explicit Native Passive-Load Readiness Before Muscle Matching

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** unassigned
- **Issue:** #11939
- **Branch:** feat/feedback-passive-readiness-11939
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11939-require-explicit-native-passive-load-rea.md)
- **Summary:** Require explicit native passive-load readiness before muscle matching

### DL-#11935 · Reconcile Development Log Provenance, Legacy Headings, and Source-Backed Delivery States Without Qualifying Scientific Work.

- **State:** shipped
- **Shipped:** 2026-10-12
- **Owner:** unassigned
- **Issue:** #11935
- **Branch:** chore/development-log-collation-11935
- **PR:** #12337
- **Paths:** see #12337
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`d8a6e303`; collated from changes/11935-reconcile-development-log-provenance-leg.md)
- **Summary:** Reconcile Development Log provenance, legacy headings, and source-backed delivery states without qualifying scientific work.

### DL-#11933 · Add Actual R2025b Physical and Discrete Saved-Operating-Point Restart Prerequisite

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** unassigned
- **Issue:** #11933
- **Branch:** feat/feedback-simscape-replay-11921
- **PR:** #12141
- **Paths:** see #12141
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11933-add-actual-r2025b-physical-and-discrete.md)
- **Summary:** Add actual R2025b physical and discrete saved-operating-point restart prerequisite

### DL-#11923 · Native Constrained-Source Readiness

- **State:** shipped
- **Shipped:** 2026-10-09
- **Owner:** codex
- **Issue:** #11923
- **Branch:** feat/feedback-constrained-source-11923
- **PR:** #11931
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/native_constraint_state.py,tests/opensim/test_native_constraint_state.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`2e3c79b4`; collated from changes/11923-native-constrained-state.md)
- **Summary:** Observe actual native constrained state and preserve unverified full-model restart, lock-target and physiology gates.

### DL-#11912 · Frozen Training Capture Registration

- **State:** shipped
- **Shipped:** 2026-10-12
- **Owner:** claude
- **Issue:** #11912
- **Branch:** fix/11912-reland-frozen-registration
- **PR:** #12363
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/frozen_registration.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`4a1809f9`; collated from changes/11912-frozen-training-registration.md)
- **Summary:** Freeze training-only native capture registration with explicit geometry lineage

### DL-#11908 · Native Muscle Experiment Bundle Development Boundary

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** codex
- **Issue:** #11908
- **Branch:** feat/feedback-opensim-bundle-11908
- **PR:** #12141
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/native_muscle_bundle.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11908-native-muscle-bundle.md)
- **Summary:** Bind native OpenSim muscle replay to the shared experiment bundle

### DL-#11903 · Native OpenSim Geometry Admission

- **State:** shipped
- **Shipped:** 2026-10-09
- **Owner:** codex
- **Issue:** #11903
- **Branch:** feat/feedback-native-markers-11903
- **PR:** #11911
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/native_marker_geometry.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b21b80a`; collated from changes/11903-native-opensim-geometry.md)
- **Summary:** Bind explicit native OpenSim marker geometry and reject placeholder physical evidence

### DL-#11900 · Native Pinocchio Replay Development Boundary

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** codex
- **Issue:** #11900
- **Branch:** feat/feedback-pinocchio-native-11900
- **PR:** #12141
- **Paths:** src/engines/physics_engines/pinocchio/python/native_torque_replay.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11900-native-pinocchio-torque-replay.md)
- **Summary:** Add native Pinocchio frozen torque replay through the existing engine

### DL-#11867 · Reject Unqualified Concrete Muscle Laws Before Native Initialization

- **State:** shipped
- **Shipped:** 2026-10-09
- **Owner:** codex
- **Issue:** #11867
- **Branch:** fix/feedback-exact-law-11867
- **PR:** #11897
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/muscle_replay.py,tests/opensim/test_native_muscle_replay.py,manuals/upstreamdrift/chapters/13-native-muscle-replay.qmd
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`da8228db`; collated from changes/11867-reject-unqualified-concrete-muscle-laws.md)
- **Summary:** Reject unqualified concrete muscle laws before native initialization

### DL-#11856 · Reproduce Native Moving-Path Force-Length Mismatch and Reject Nonfinite Derivative Evidence

- **State:** shipped
- **Shipped:** 2026-10-11
- **Owner:** codex
- **Issue:** #11856
- **Branch:** feat/feedback-path-work-11856
- **PR:** #12228
- **Paths:** scripts/diagnostics/native_path_work.py,src/engines/physics_engines/opensim/python/tour_matching/muscle_qualification.py,tests/opensim/test_native_path_work.py,manuals/upstreamdrift/chapters/24-native-path-work.qmd
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`49ef898b`; collated from changes/11856-reproduce-native-moving-path-force-lengt.md)
- **Summary:** Reproduce native moving-path force-length mismatch and reject nonfinite derivative evidence

### DL-#11851 · Cast Numeric Refinement Rows Explicitly in numerics.py to Resolve Mypy Static Type Errors.

- **State:** shipped
- **Shipped:** 2026-10-13
- **Owner:** unassigned
- **Issue:** #11851
- **Branch:** fix/11851-correct-static-types-shot-pattern
- **PR:** #12452
- **Paths:** see #12452
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`c6ad9cc2`; collated from changes/11851-correct-static-types-shot-pattern.md)
- **Summary:** Cast numeric refinement rows explicitly in numerics.py to resolve mypy static type errors.

### DL-#11840 · Correct Central Impact and Add Three-Club Shot Pattern Analysis

- **State:** shipped
- **Shipped:** 2026-10-13
- **Owner:** unassigned
- **Issue:** #11840
- **Branch:** codex/shot-pattern-analysis
- **PR:** #12418
- **Paths:** see #12418
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`0644635c`; collated from changes/11840-correct-central-impact-and-add-three-clu.md)
- **Summary:** Correct Central Impact and Add Three-Club Shot Pattern Analysis

### DL-#11838 · Native Drake Replay Development Boundary

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** codex
- **Issue:** #11838
- **Branch:** feat/feedback-drake-native-11838
- **PR:** #12141
- **Paths:** src/engines/physics_engines/drake/python/native_torque_replay.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11838-native-drake-torque-replay.md)
- **Summary:** Add restricted native Drake frozen torque replay

### DL-#11837 · Score Native Output Against Frozen Observations With Full Replay Identity

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** unassigned
- **Issue:** #11837
- **Branch:** feat/11837-f09b-native-observation-qualification
- **PR:** #12141
- **Paths:** src/engines/feedback_observation_qualification.py,tests/unit/engines/test_feedback_observation_qualification.py,tests/motion_capture/rig/test_tools_session_export.py,docs/development/feedback_controls/IMPLEMENTATION_PLAN.md,docs/development/feedback_controls/F09B-OBSERVATION-QUALIFICATION.md,docs/development/feedback_controls/TURNOVER.md,manuals/upstreamdrift/chapters/13-feedback-comparison.qmd,vendor/ud-tools
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ec109fa1`; collated from changes/11837-f09b-native-observation-scoring.md)
- **Summary:** Score native output against frozen observations with full replay identity

### DL-#11834 · Observe Pinned Native Coupled-Wrist Geometry With Explicit Assembly and Derivative Gates

- **State:** shipped
- **Shipped:** 2026-10-12
- **Owner:** claude
- **Issue:** #11834
- **Branch:** fix/11834-reland-wrist-gate
- **PR:** #12362
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/wrist_compatibility.py,tests/opensim/test_native_wrist_compatibility.py,manuals/upstreamdrift/chapters/29-native-wrist-compatibility.qmd
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`ba06c0da`; collated from changes/11834-native-wrist-compatibility.md)
- **Summary:** Observe pinned native coupled-wrist geometry with explicit assembly and derivative gates

### DL-#11831 · Sample Native Replay Marker Outputs on the Observation Clock

- **State:** shipped
- **Shipped:** 2026-10-09
- **Owner:** unassigned
- **Issue:** #11831
- **Branch:** feat/11793-f09a-output-sampling
- **PR:** #11830
- **Paths:** src/shared/python/motion_matching/replay_metrics.py,src/shared/python/motion_matching/**init**.py,tests/unit/motion_matching/test_replay_observation_sampling.py,docs/development/feedback_controls/F09A-OBSERVATION-SAMPLING.md,docs/development/feedback_controls/TURNOVER.md,manuals/upstreamdrift/chapters/13-feedback-comparison.qmd
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`fb692749`; collated from changes/11831-f09a-observation-output-sampling.md)
- **Summary:** Sample native replay marker outputs on the observation clock

### DL-#11826 · Native Thelen Replay Boundary

- **State:** shipped
- **Shipped:** 2026-10-09
- **Owner:** codex
- **Issue:** #11826
- **Branch:** feat/feedback-native-thelen-11826
- **PR:** #11829
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/muscle_replay.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`cde5a1ec`; collated from changes/11826-native-thelen-replay.md)
- **Summary:** Admit compliant native Thelen muscle replay with explicit subtype and state policy

### DL-#11823 · Native Torque Replay Development Boundary

- **State:** shipped
- **Shipped:** 2026-10-09
- **Owner:** codex
- **Issue:** #11823
- **Branch:** feat/feedback-native-replay-11790
- **PR:** #11836
- **Paths:** src/engines/physics_engines/mujoco/python/native_torque_replay.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`9d336118`; collated from changes/11823-native-torque-replay.md)
- **Summary:** Add a full-state native MuJoCo consumer of the Tools torque replay bundle

### DL-#11821 · Add Synthetic Sparse Inverse-Dynamics Collocation Spike and Benchmark Existing Multiple Shooting With Fresh Replay Diagnostics

- **State:** shipped
- **Shipped:** 2026-10-09
- **Owner:** unassigned
- **Issue:** #11821
- **Branch:** feat/f03-constrained-ocp-11787
- **PR:** #11822
- **Paths:** src/shared/python/motion_matching/sparse_collocation_spike.py,tests/unit/motion_matching/test_sparse_collocation_spike.py,manuals/upstreamdrift/chapters/17-sparse-collocation-benchmark.qmd
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`72a58405`; collated from changes/11821-sparse-collocation-spike.md)
- **Summary:** Add synthetic sparse inverse-dynamics collocation spike and benchmark existing multiple shooting with fresh replay diagnostics

### DL-#11819 · Add Pinned Native OpenSim Muscle-Model Structural Admission Without Anatomical Qualification

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** codex
- **Issue:** #11819
- **Branch:** feat/feedback-native-admission-11819
- **PR:** #11825
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/model_admission.py; tests/opensim/test_model_admission.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`1b149c36`; collated from changes/11819-native-model-admission.md)
- **Summary:** Add pinned native OpenSim muscle-model structural admission without anatomical qualification

### DL-#11815 · Native Muscle Contact Replay

- **State:** shipped
- **Shipped:** 2026-10-09
- **Owner:** codex
- **Issue:** #11815
- **Branch:** feat/feedback-native-contact-11815
- **PR:** #11820
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/muscle_replay.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`194cf468`; collated from changes/11815-native-muscle-contact.md)
- **Summary:** Add explicit model-owned native contact policies and complete wrench evidence to muscle replay

### DL-#11810 · Prevent Parameter-Only Muscle Audits From Fabricating Native Qualification Evidence

- **State:** shipped
- **Shipped:** 2026-10-09
- **Owner:** codex
- **Issue:** #11810
- **Branch:** fix/feedback-muscle-evidence-11791
- **PR:** #11814
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/muscle_qualification.py; tests/opensim/test_muscle_cmc.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5ef39d92`; collated from changes/11810-muscle-evidence-guard.md)
- **Summary:** Prevent parameter-only muscle audits from fabricating native qualification evidence

### DL-#11809 · Native Muscle Replay Development Boundary

- **State:** shipped
- **Shipped:** 2026-10-09
- **Owner:** codex
- **Issue:** #11809
- **Branch:** feat/feedback-opensim-11791
- **PR:** #11812
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/muscle_replay.py
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`45615364`; collated from changes/11809-native-muscle-replay.md)
- **Summary:** Add early native continuous-state muscle replay and reconcile reviewed epic milestone dependencies

### DL-#11786 · Add Phase-Gated Tangent Feedback With Constrained Actuator Allocation and Auditable Applied-Torque Evidence

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** unassigned
- **Issue:** #11786
- **Branch:** feat/f02-distributed-feedback-11786
- **PR:** #11816
- **Paths:** src/shared/python/motion_matching/distributed_feedback.py,tests/unit/motion_matching/test_distributed_feedback.py,manuals/upstreamdrift/chapters/16-distributed-feedback.qmd
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`1fe02740`; collated from changes/11786-distributed-feedback.md)
- **Summary:** Add phase-gated tangent feedback with constrained actuator allocation and auditable applied-torque evidence

### DL-#11785 · Freeze Six-Engine Feedback Comparison Admission and Independent Replay Prerequisites

- **State:** shipped
- **Shipped:** 2026-10-09
- **Owner:** unassigned
- **Issue:** #11785
- **Branch:** feat/f01-six-engine-contracts-11785
- **PR:** #11808
- **Paths:** src/engines/feedback_comparison.py,src/engines/model_inventory.py,src/config/engine_model_inventory.json,manuals/upstreamdrift/chapters/13-feedback-comparison.qmd
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`a0a29db1`; collated from changes/11785-feedback-comparison.md)
- **Summary:** Freeze six-engine feedback comparison admission and independent replay prerequisites

### DL-#11784 · Document Researched Distributed Feedback Epic, Six-Engine Parity and Muscle-Driven OpenSim Target

- **State:** shipped
- **Shipped:** 2026-10-09
- **Owner:** unassigned
- **Issue:** #11784
- **Branch:** docs/feedback-controls-11784
- **PR:** #11798
- **Paths:** docs/development/feedback_controls
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`b488dbbd`; collated from changes/11784-feedback-controls-design.md)
- **Summary:** Document researched distributed feedback epic, six-engine parity and muscle-driven OpenSim target

### DL-#11745 · LIFT-5 Slice 1: Back-Squat Same-Input Bundle From the Movement Optimizer via Full-Body Marker IK, With Provenance, Regeneration Script and MuJoCo Replay Smoke Test

- **State:** shipped
- **Shipped:** 2026-10-19
- **Owner:** unassigned
- **Issue:** #11745
- **Branch:** claude/lift-5-squat-bundle
- **PR:** #12245, #12340, #13052, #13047, #13263, #13276
- **Paths:** see #12245
- **Started:** 2026-10-14
- **Last verified:** 2026-10-19 (`c0d8142c`; collated from changes/11745-lift-5-clean-and-jerk-is-the-whole-lift.md)
- **Summary:** LIFT-5 slice 1: back-squat same-input bundle from the Movement Optimizer via full-body marker IK, with provenance, regeneration script and MuJoCo replay smoke test

### DL-#11737 · OSV-6 Lead Foot: Lateral-Marker Zero-Twist Correction (Knee_Flexion_Axis) and Leg-Marker Calibration Constraints Under --Foot-Progression; MuJoCo Address Toe-Out Within 2 Deg on Both Captures (Driver +0.2/-0.3, Iron -0.0/-1.4 Deg); Canonical Receipts Unchanged (Default-On Correction Is a Recorded Failed Experiment)

- **State:** shipped
- **Shipped:** 2026-10-10
- **Owner:** unassigned
- **Issue:** #11737
- **Branch:** claude/osv-6b-leg-azimuth
- **PR:** #12049, #12051
- **Paths:** see #12049
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b2ef7a6`; collated from changes/11737-osv-6-record-the-full-address-coordinate.md)
- **Summary:** OSV-6 lead foot: lateral-marker zero-twist correction (knee_flexion_axis) and leg-marker calibration constraints under --foot-progression; MuJoCo address toe-out within 2 deg on both captures (driver +0.2/-0.3, iron -0.0/-1.4 deg); canonical receipts unchanged (default-on correction is a recorded failed experiment)

### DL-#11729 · MyoSuite Anthro Map: NeckInputY Drives Neck_Flexion (Sign -1 by FK), NeckInputX (Lateral Bending) Unmapped, Neck_Rotation No Longer Overwritten

- **State:** shipped
- **Shipped:** 2026-10-11
- **Owner:** unassigned
- **Issue:** #11729
- **Branch:** claude/osv-3-myosuite-neck-flexion
- **PR:** #11768, #11772, #11979, #12154, #12194
- **Paths:** see #11768
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`31b4252b`; collated from changes/11729-osv-3d-closed-loop-gaze-neck-fd-neck-gaz.md)
- **Summary:** MyoSuite anthro map: NeckInputY drives neck_flexion (sign -1 by FK), NeckInputX (lateral bending) unmapped, neck_rotation no longer overwritten

### DL-#11792 — F08: Match Mocap With Muscle Excitations and Verify Pure OpenSim Replay

- **State:** shipped
- **Shipped:** 2026-10-13
- **Owner:** codex
- **Issue:** #11792 (parent #11784; children #12342, #12343, #12353)
- **PR:** #12344 (draft); related #12355, #12318, #12402, #12407, #12386, #12442, #12401
- **Branch:** `fix/native-path-connection-finalization-12233-20261012`
- **Started:** 2026-10-09 (parent issue opened)
- **Paths:** `changes/12233-bound-moco-optimizer-states-to-source-mo.md`; `docs/agent_context/README.md`; `docs/agent_context/index.html`; `docs/development/feedback_controls/F07_PATH_WORK_TURNOVER.md`; `manuals/upstreamdrift/chapters/24-native-path-work.qmd`; `src/engines/physics_engines/opensim/python/tour_matching/native_moco_preparation.py`; `tests/unit/engines/opensim/test_native_path_optimizer_domains.py`
- **Summary:** This software/readiness work moves native connection finalization before optimizer-domain and passive audits. It does not claim a solve, optimum, learned replay, or physical qualification; all 17 registered model/variant rows remain unqualified.
- **Last verified:** 2026-10-14 (`85b810d4`; collated from changes/11792-stateless-feedback-policy-12390.md)

### DL-#11614 · Same-Input Parity LaTeX Research Reference (P-1 Pointwise Result)

- **State:** shipped
- **Shipped:** 2026-10-14
- **Owner:** unassigned
- **Issue:** #11614
- **Branch:** docs/sip-9-parity-reference-11614
- **PR:** #11616, #11625, #12485, #11806, #12493
- **Paths:** see #11616
- **Started:** 2026-10-07
- **Last verified:** 2026-10-14 (`4aca3f1a`; collated from changes/11614-synchronize-same-input-parity-calculatio.md)
- **Summary:** Same-input parity LaTeX research reference (P-1 pointwise result)

### DL-#11235 · Necromatcher Native Fit

- **Local Research Golf Checkpoint — #11493:** API, native and React now provide explicit Connect → Prepare Research → Arm → Submit → Recall through the actual local session. The authenticated bridge retains exact saved impact vectors and a declared proper Aim rotation; MODEL_CONTACT and contact/numerical/scientific qualification remain unverified. React passed 106 cases; one actual temporary-Library MuJoCo/clean-worker/Rust API and native case passed in 23.07 s with seven exact retained samples; configured mypy passed on seven owners. The final root cohort passed 209 cases with 12 warnings and no skips in 58.64 s; the actual native case is one additional pass. All 13 modified Python owners pass pinned Ruff check/format. The capability atlas was refreshed without claiming overall gap closure. Core V1 is superseded only by the root delivery_status unknown-ID KeyError→None contract fix. The three-page local report and procedure document the new local flight, saved assumptions and explicit lifecycle. No fit improvement, avatar/course capability, GL accuracy, <host> delivery or new remote CI claim is established. Tiger remains [0,191), Hogan [0,750); the full goal remains ACTIVE. [Procedure](necromatcher-local-research-golf.md); [Three-Page Report Source](necromatcher-local-research-golf-methods.tex).

- **Authored Replay Impact Checkpoint:** SELF progresses #11464 with authenticated explicit head/face/sample extraction and full-vector declared-flight handoff. Earlier root 83 integrated app/impact cases pass. Portable full-state receipt export/recall now binds exact canonical trajectory bytes without changing the viewer wire; six integration cases pass, including tamper rejection. The supplemental five-page LaTeX methods report is compiled and visually reviewed on the local Desktop. [Procedure](necromatcher-replay-impact.md). Normal app impact execution remains open; authored seconds are not calibrated historical time and no historical replay or qualification is claimed.

- **App Boundary Checkpoint:** SELF progresses #11450/#11414 with explicit strict lossless seeds and exact ordered saved-frame review in native/React hosts. Root 45 Python/45 UI cases pass; edited domains survive mode changes, excluded/sparse indices are covered, original source-only browsing remains available. [Procedure](necromatcher-app-restriction-and-frame-review.md). Historical results and Methods V8 remain unchanged; no physical qualification or CI-green claim.

- **State:** shipped
- **Shipped:** 2026-10-12
- **Owner:** codex
- **Issue:** #11235; parent #11232; explicit coordinate seeds #11281; probe/generalization trials #11284; qualification child #11357
- **PR:** [#11240](https://github.com/D-sorganization/UpstreamDrift/pull/11240) (merged); [#11359](https://github.com/D-sorganization/UpstreamDrift/pull/11359) (open)
- **Arm Morphology Documentation Checkpoint:** Four unoptimized arm ±5% admissions and twelve selected overlays are independently provenance-verified; exact twelve-page AN1 report retained. Source 14,800 preserved, library 383→391; grip gaps worsen to 18.6–22.1 mm and anatomy remains unadopted. [Procedure](necromatcher-arm-morphology-admission.md). Documentation SELF; no behavior, evidence-role or release-qualification change.
- **Scoped Camera Checkpoint:** At published producer 55d17, native session 74740 and independent session 92074 closed with exit 0; full 14,805-source-file and 391-library-file maps preserved. Tiger [0,191) body RMS 24.0827→24.7803 px worsens; Hogan 10.1364→10.0910 px has mixed shaft outcomes. Keep saved cameras; 17 root checker tests and all six compiled report pages reviewed. [Procedure](necromatcher-conditional-camera-and-scope.md). #11414 durable scope integration is implemented; root 294 Python/39 UI cases, Ruff, pinned mypy, TypeScript and ESLint pass. HTTP receipt registration runs off the event loop; 13 focused root follow-up cases pass. Restricted trial at 669508 saved an authored seed; strict fitting exceeded unchanged 300-second wall cap, exit 1, no final fit. Original assets/index records preserved; ten allowed files added. Solver evaluations/elapsed unknown. Independent partial-stage verification is accepted (V4 partial checker, actual root release V2). Setup profile V1 failed on an ambiguous ndarray adapter/receipt JSON and is preserved; V2 measured five public stages successfully, with independent serialized acceptance. At published `0143b39f571cbe67c57091fb2ae19e7d7eadae3b`, both independent audits preserved complete 14,826-source-file/401-library-file maps. The same 191-frame confidence-weighted dense body RMS is 35.102046 px for the seed versus 24.082673 px for its parent; retain the parent. Actual selected seed stills 0/150/190 show opacity 0.35 proxy shapes, skeleton and an explicit unoptimized authored research seed caption. Independent serialized source/image verification is accepted; unweighted display-marker RMS is approximately 6.45/44.14/18.27 px, separate from weighted body metrics. Native projection and complete overlay-union pixel parity were not independently recomputed. No timeout cause, qualified match, new full video or <host> delivery is established. Tiger matching remains [0,191), exact physical release is unmeasured, and Hogan is unchanged. Methods V7 is the latest compiled supplement: root reviewed all eighteen pages in contact sheets and new pages 17 and 18 full-size; V2/V3/V4/V5/V6 remain preserved. The root V7 acceptance SHA256 is 51a8da1aaa9b590100354d8db1956431450b955d87ba57894b97da0df673c810. The two new pages describe the reusable restriction workflow and synthetic native end-to-end evidence at 5197. Its three pinned external Desktop PNGs remain earlier V21 authored-seed figures and local repeatability dependencies, not a portable standalone bundle or newly rendered restricted fit. No scientific qualification or new <host> transfer. Performance child #11445 now implements public authenticated_read: one canonical full PNG/BGR/clock validation per capture per operation, fresh complete file hashes and metadata checks on reuse/close, cross-library/thread/task and nested-context rejection, immutable DTOs and failure reset. Final root validation passed 93 cases across two lanes (86 plus seven public-owner/fingerprint cases; one Windows symlink-privilege skip and inherited import warnings). Canonical public SHA256 validation is reused, and execution fingerprints cover shadow_tracker; this API checkpoint is separate from the accepted artifacts at 0143. This integration checkpoint implements short authentication boundaries for queue admission, worker setup and fresh output checks, candidate/delayed publication and storage. Contexts close before scheduling, computation, delivery or persistence; delayed publication closes before add_fit, which uses its own fresh validation context. Root passed 188 distinct cases (185 SDK-free plus three native temporary-fixture cases; one Windows symlink-privilege skip, three native cases deselected in the SDK-free lane and nine inherited warnings). Published integration f2dfe completed read-only Profile V3: five stages took 0.672506/1.464802/9.053819/1.511551/0.460402 s; only seed binding decoded 210 PNGs, once in one canonical full authentication. Source 14,835/Library 401, project/runtime/config were preserved; assessment is byte-identical to V2. Root acceptance SHA256 b8469dc86d8aa72e1be2499786a107c847dd2e11d32f20567aa1effb1947a8e4 records this invocation; differing instrumentation/context/hardware prevents an isolated speedup or timeout-cause claim. No strict retry or new <host> delivery is authorized; accepted 0143 artifacts are unchanged. Pure restriction #11450 has a frozen V2 core (SHA256 b3504e6c64c7f97116a108b97a06d6ecba8732f3d38dd004751789e0bc93ce58) with 23 plus 53 agent cases. Root accepted 91 distinct cases (23 core plus 68 disjoint compatibility/bounds/initialization cases; synthetic MuJoCo-first fixtures only, nine inherited warnings and 38 unmarked cases reported only). Root acceptance SHA256 c0c2c44509650d37dc9516aa4185a54b828a49767bcca4ab8d8b2af64786b2ff records Ruff/format, pinned mypy and governance checks; manual release remains inventory-blocked. Pure q/v/a roundoff and Bernstein encoding evidence 17ddcd74c27276a376174c202500baa10e389d09b97ca27c3c42e9d1e4b778b5 is not native feasibility, pixel agreement, seed adoption or queue integration. Subsequently, native session 50689 closed exit0 at published 81aa448794a0ce763cdc2dd6e503d3f9b42200e9; root serialized closure accepted ce833a420c5e208b82bc430fe4be4c4c16830aee54d9c8c357bdb1af9e775e9c. Full source 14,837/Library 401/project/runtime/config/helpers matched the fresh baseline. Restricted-parent body RMS exactly equals saved parent (dense 24.082673308003265, remaining 24.337344114034266, training 21.776977995696026 px); 381-point q/v/a roundoff and typed receipt equality hold. The full 176 x 12 dimensionless scaled constraint matrix has maximum 17.023155533359613 at ground:heel_l, source 18.501816666666667; continuous contact is uncertified and nodes differ from V21, so no V21 656 comparison applies. Retained frame 0 raw shaft RMS 66.69082241158017 px/angle 28.301292322830996deg remains mismatched. No optimizer/convergence/physical qualification, telemetry null; no adopted seed, new fit, overlay rendering/pixel-parity evaluation or <host> delivery. Subsequent backend software checkpoint: root accepted paired restrict_initialization/restricted_spline operations with canonical SDK-free restriction receipts, preserved prior/bounds, parent agreement within 1e-12, scope/hash admission and worker/storage/recall/delayed-publication checks. Legacy semantics and public counts (eight option fields, seven start_native_refit arguments, eight builder arguments) are unchanged. Root passed 87 distinct SDK-free cases with nine inherited warnings; pinned mypy on all five owners and Ruff/format passed. Root acceptance SHA256 dc43749d0e69c6c43051712c5d37b819fbbe950edd3cf5dfb12f426a61b80bed authenticates the software freezes. Backend software was published at 5197f8639818321b7cf8f86d4748a25ce79aa278 with all normal hooks passing. Synthetic native end-to-end session 4534 then closed exit 0: the public queue, clean worker, storage and recall preserved the nonuniform restricted curve on [1,6), with four training/five dense frames, 43 locked coordinates and five rejected queue/storage counterexamples. Root closure SHA256 a26073a9c7bef6c8d5f2dd12b249c8e5f848be49c9fb0ba63d0a5c98f011d9bc authenticates the independently closed SDK-free audit (session 97412): fresh 14,844-source-file/401-historical-file/runtime maps are unchanged. Independent pure q/v/a errors are 3.47e-18/2.78e-17/1.78e-15; saved native pixel metrics agree with the actual run, but were not independently reprojected by the serialized audit. Optimizer counts remain null; no optimization, historical seed registration, new overlay or <host> delivery occurred. This synthetic software acceptance does not alter the earlier Methods V6 historical native-evaluation checkpoint or establish physical qualification. Subsequent caption software adds UNOPTIMIZED RESTRICTED RESEARCH SEED through the shared still/video composer and authenticated manifest, with eight strict DTO fields and mutually exclusive authored/restricted flags. Root passed 80 distinct cases in MuJoCo-first temporary fixtures: canonical exports and tampered lineage, manifest claims and exact legacy/authored pixels; configured pinned mypy, Ruff and format pass. Caption root acceptance SHA256 12a5e9e99d0b5d3827fcddc229324bf9e5525e448ae430acb4a3e0881a1f73d0 records this later software checkpoint, separate from the Methods V7 appendix at 5197. No new historical seed or overlay is claimed. Next: publish through normal hooks, then separately reviewed historical admission and the unchanged strict restart. The full goal remains active. Subsequent V22 checkpoint: Tiger remains within reviewed source frames 0-190. The accepted lossless seed has a local 191-frame shape/skeleton video; the ordinary strict fit modestly reduces same-domain weighted dense RMS from 24.08 to 23.57 px but remains nonconverged and REJECTED. Independent acceptance SHA256 8bb98838915c914c4dab8b9c9b9fff871ec7b02294a8cfd155d57657852aa72d records preserved source and original non-index Library bytes. Methods V8 is the earlier 21-page seed/stills report, not this later fit/video checkpoint. Three optimized research stills are accepted locally (proof 0d244648c2f76bb2c4be9f232d81215a58c5a1779963f50bcb575f76ff402d13); <host> delivery remains pending. Hogan is unchanged; physical/anatomical/contact qualification is false and the goal remains active. Detailed evidence is retained in the camera-and-scope procedure.
- **Branch:** `feat/necromatcher-native-fit-11235`
- **Paths:** `src/shared/python/motion_matching/historical_fit/**`, `src/shared/python/workspace/necromatcher_*`, `src/engines/physics_engines/mujoco/python/**`, `src/shared/python/motion_matching/historical_fit/shaft_*`, `src/shared/python/workspace/necromatcher_shaft_evidence.py`
- **Started:** 2026-10-01
- **Last verified:** 2026-10-03 — Actual producer `3ab5c2ac87dd6aa97872b49596fbcad4f9a6eeb5` completed all four matched shaft control/variant jobs. Fresh assessment `b1f3898d3d9e2cac5c460ceb33fd2e6c5b85371158b4eb2a823e4f027ede002e` verifies identical starts/body/geometry recipes and all five fixed finite targets; every fit remains budget-exhausted/nonconverged/rejected. Raw training improves, Tiger holdout worsens and Hogan holdout is essentially unchanged. Actual V2 admission and display-only exports preserve source/clock/roles without changing objectives; see the shaft trial procedure. Earlier verification retained: 2026-10-02 — producer6654881b0788c77620a0afb8a9a6763f3ee1ccb9 completed two V15 exact-six-knot normalized-coverage continuations under #11284. Independent native audits verify identical start/evidence/prior,32 finite authored ranges with12 unbounded coordinates,210 source frames,209 midpoints,88 scalar extrema and10 phase probes. Control78nodes/dense25.048392px/grip1.905599mm/angle0.703158degrees/ground3.437775mm fails ground; variant178nodes/dense25.160335px/grip1.955344mm/angle0.715385degrees/ground1.852699mm passes all five fixed finite research targets. Geometry weights4000→1752.808989 preserve potential family rowmass, not physical-time quadrature or continuous nonlinear qualification. Both exhaust30evaluations, remain nonconverged/rejected and have unknown physical time. Exact assessment and repeatability inputs are preserved in turnover; preceding V12–V14 evidence remains unchanged. Actual V15 overlay and selected-still review are published, with rejected qualification retained; final expanded source regression passed 476 cases.
- **Slice A verification:** Source-bound club evidence Slice A (#11318) is committed at `605c695259b13cb98a6d6e379e7f7fee51f27443`: immutable reviewed shaft-axis segment/source-frame DTOs, distinct original decoded-pixel and PNG-byte hashes, exact source clock/capture binding, explicit abstention and authored-uncalibrated weighting, plus an exact-definition-bound infinite authored axis resolved from declared coaxial shaft/grip solid centers. Public receipt counts/digests and strict real-coordinate guards reject malformed records. Initial tests were red; the final focused suite passed 57 cases, with Ruff/format, pinned mypy and budget checks. Actual source-bound admission subsequently preserved two Tiger fragments, one faint Hogan fragment and a blurred-frame abstention; receipt `bb36ef2233e39c829b341046d2555fcbc245769dccbaa9489933f09ddb5fadd7`. Root independently inspected all four plots; separate review `67f8ace21806ea5c62e0f1da5635c3e3e19baf614e9989c9fa935ee48201d7ac`. The first TEMP driver failed before output creation because Ruff removed imports used only through locals; preserved v1 and traceback remain distinct from v2, whose explicit registry repair passed 10 protocol tests before successful execution. No optimization, library mutation or physical endpoint/shaft-length qualification occurred.
- **Summary:** Source-bound native fits retain canonical Hermite coefficients and optional analytic interior constraints. Opt-in named hard bounds use a canonical Bernstein-admissible domain with explicit logged authored initialization and immutable native definition/range identities; v7 candidates remain rejected and require new feasible trials plus grip/contact qualification. [Turnover](necromatcher-turnover.md).
- **Slice B verification:** Committed8989 generic additional-image residuals preserve ImageFitInputs and independent body/geometry timelines. Separate sparse infinite-axis residuals and raw perpendicular RMS/unoriented angles passed112 combined and33 independent cases. No historical shaft fit occurred at that checkpoint.
- **Slice C verification:** Committed8989 canonical capture/camera/PNG/full-clock/native-axis rebind returns one reusable native binding. Tests passed96 combined and28 independent cases; compilation-count compatibility is retained.
- **Slice D verification:** Current reviewed queue/worker/persistence source passed102 combined and19 independent cases; strict receipt parsing rejects bool/float counts before native/source I/O. Native evidence controls passed32 offscreen tests independently; web30 plus type checking/ESLint passed. Stable original-video/job/API integration passed41; renderer20/80 combined and20 independent passed. Ruff/format and pinned mypy passed. [Shaft Trial Procedures](necromatcher-shaft-trials.md) records exact freezes and the independently observed unchanged five-second test deadline overrun, followed by successful focused rerun; no speculative cause is asserted.

### DL-#11246 · Necromatcher Source Video Overlays

- **State:** shipped
- **Shipped:** 2026-10-12
- **Owner:** codex
- **Issue:** #11246; parent #11232; native stored-run recall #11282; translucent model-surface child #11356
- **PR:** [#11240](https://github.com/D-sorganization/UpstreamDrift/pull/11240) (merged); [#11359](https://github.com/D-sorganization/UpstreamDrift/pull/11359) (open)
- **Branch:** `feat/necromatcher-native-fit-11235`
- **Paths:** `src/shared/python/workspace/necromatcher_video*`, `src/tools/necromatcher/**`, `ui/src/components/necromatcher/**`
- **Started:** 2026-10-01
- **Last verified:** 2026-10-03 — Producer `3ab5c2ac87dd6aa97872b49596fbcad4f9a6eeb5` exported all four original-footage shaft displays; independent receipt `f60d277ce81e7d1d3dc2ddace626a1bacd9bd3a83708375ffefe1e7dcdd39dd0` verifies 1,920 source-sized frames, original PNG/PTS bindings, projections/metrics, immutable source/library brackets and twelve selected overlays/six originals. V1/V2 union is display-only; all-frame visual and manual acceptance remain false. Exact manifest PTS are distinct from Tiger encoder rounding (maximum 6.9736403 microseconds relative drift); Hogan 30/1 is exact. Earlier verification retained: 2026-10-02 — V15 selected Tiger variant export run `97b1b96fea0c4a2da403004a4985f390` at producer `6654881b0788c77620a0afb8a9a6763f3ee1ccb9` verifies all 210 original 1280x720 frames, original PNG/PTS bindings, ZIP CRC/member hashes and immutable source/runtime/audit brackets. Actual stills 0/105/209 were separately viewed; club-seed disagreement, follow-through upper-body residuals and cropped feet remain. Combined still-review SHA-256 `ca46a15129d238c06391f6ee1b05c0cfa6bc14cd6be95075d738841798d4b369`; the execution journal retains visual_review_completed=false. All-frame visual and manual acceptance remain false. Earlier V13/V12/native stored-run review receipts remain preserved; canonical sharing-race repair and final expanded source regression passed 476 cases.
- **Summary:** Shared owned jobs export source-clock native joint-tree wireframes, PNGs and verified ZIP packages through native and web controls. Export success remains separate from scientific acceptance. [Turnover](necromatcher-turnover.md).

### DL-#11247 · Necromatcher Reproducible Methods Report

- **State:** shipped
- **Shipped:** 2026-10-12
- **Owner:** codex
- **Issue:** #11247; parent #11232; source audit #11357
- **PR:** [#11240](https://github.com/D-sorganization/UpstreamDrift/pull/11240) (merged); [#11359](https://github.com/D-sorganization/UpstreamDrift/pull/11359) (open)
- **Branch:** `feat/necromatcher-native-fit-11235`
- **Paths:** `docs/development/necromatcher-methods.tex`, `docs/development/historical_capture/*summary.json`, `docs/development/necromatcher-turnover.md`
- **Started:** 2026-10-01
- **Last verified:** 2026-10-03 at `b8b280ce2f5ed706d1778fd809250cf18eac1fb7` — Published 74-page actual matched-shaft report: PDF `36319afeb6ec6132f7684387d04c1b8cde8d2854a38cf718354325334e4fdffd`, source `41c61a755f72a3b8928c5938f545e0a8a32aa19e4f84437ffc5ac07c92d7b6e9`, independent review `8b4fb44f2b0cd552c7a443f5e90ada9d95f624ff3711d4514f3f70e6708db6ee`, root review `6fe30a6978fcc61fbb3192df6641b3da93515155e21bec6cc3852ef832854c5c`. Three installer-disabled passes completed without actual warnings/overfull/unresolved references; all page layouts reviewed. Desktop manifest 1,504 artifacts `49868b903a729a147d188b03a0fc5ecf3f68e4973345f08b0191d79de2b85228`; exact prior 66 publication and1,260 index/README preserved under Report/handoff-shaft-66-page-publication. Earlier verification retained: 2026-10-02 — Expanded 66-page latest-handoff/sparse-shaft report compiled in three installer-disabled MiKTeX passes and was independently reviewed on all pages, with full-size contents/new chapter/figures/source-map checks. No overfull boxes or unresolved references; 17 contained table-wrapping warnings were visually reviewed. Published PDF SHA-256 `f31f1fbd6c0917254b358bae3852e42de8e5d898b1f60d47b3b08c63427f3228`; exact source `1702e35d4a04b95d0c5de13b04db34325cfef7b8e2ad47f50ea47da45354991c`; independent review `0c7a85f91018a01d117e0d7b0e41dbec61087f6c52019b780ade7bbca4e27481`; separate root review `9de379e2b3a32089bff17fd1086eec9430f0f953240717bfa6093bf02e57446d`. Refreshed Desktop index verifies 1,216 retained artifacts, SHA-256 `bd89c17e27de7b7849986f6e00d36debcd0e3c988ab8530b687a0f92ff1a6e9d`. Exact prior 60-page publication, 49-page edition and historical snapshots remain preserved. No scientific acceptance or physical motion claim; built-in compiler infrastructure criterion remains open.
- **Summary:** Polished methods report records equations, provenance, assumptions, actual interim failures and reproducibility procedures. Built-in compiler infrastructure remains unavailable, so that issue criterion remains open. [Turnover](necromatcher-turnover.md).

### DL-#1755 · Retire the Review-Comment-to-Issue Converter

- **State:** abandoned
- **Owner:** claude
- **Issue:** Repository_Management#1755
- **Branch:** `chore/retire-comment-converter-v3`
- **PR:** draft PR from `chore/retire-comment-converter-v3` (supersedes #10978)
- **Paths:** `.github/workflows/Comment-to-Issue-Converter.yml`, `scripts/ci/process_review_comments.py`, `tests/ci/test_process_review_comments.py`, `tests/ci/test_ci_infrastructure.py`, `.github/workflows/Nightly-Doc-Organizer.yml`, `docs/development/ci_workflow_map.md`, `docs/development/`
- **Started:** 2026-09-25
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Remove the retired Convert-Review-Comments-to-Issues workflow and its processor from this repository per the fleet-wide Repository_Management#1755 campaign, and drop the stale references in the Nightly Doc Organizer and the CI workflow map. Re-lands #10978 (itself superseding #10941) fresh off `origin/main` after its branch conflicted.

### DL-#9549 · Guard Contact-Interval Provider Contract for Interval Tab Owner Ruling

- **State:** shipped
- **Shipped:** 2026-10-14
- **Owner:** claude
- **Issue:** #9549 (epic #9546)
- **PR:** draft PR from `agy/ud-9549-ruling-guard`, #12538
- **Branch:** `agy/ud-9549-ruling-guard`
- **Paths:** `tests/shared_contracts/test_impact_interval_provider.py`, `docs/development/DEVELOPMENT_LOG.md`
- **Started:** 2026-09-25
- **Last verified:** 2026-10-14 (`b8bb030b`; collated from changes/9549-add-versioned-pre-impact-kernel-bundle-a.md)
- **Summary:** Replaced obsolete strict-xfail pin gate in test_impact_interval_provider.py with test_provider_honours_interval_tab_ruling guarding the owner ruling from Tools #4946 / Tools PR #5289 (IMPACT_INTERVAL_TAB_RULING.md): standalone interval tab dropped, ImpactModelType gains no INTERVAL member, and contact-interval solver remains headless (#9549).

### DL-#9415 · Repository Root Allowlist Check in Docs Governance

- **State:** shipped
- **Shipped:** 2026-10-14
- **Owner:** claude
- **Issue:** #9415
- **PR:** draft PR from `agy/ud-9415-root-allowlist`, #12334, #12464, #12472, #12393
- **Branch:** `agy/ud-9415-root-allowlist`
- **Paths:** `scripts/check_docs_governance.py`, `scripts/config/root_allowlist.json`, `docs/governance/DOCS_GOVERNANCE.md`, `tests/scripts/test_doc_governance_checks.py`, `tests/unit/scripts/test_check_docs_governance_root_allowlist.py`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`
- **Started:** 2026-09-26
- **Last verified:** 2026-10-14 (`b686ab17`; collated from changes/9415-spec-archive.md)
- **Summary:** Committed root entries (`git ls-tree --name-only HEAD`) must match `scripts/config/root_allowlist.json`; unlisted and stale entries both fail, and a failed git call or malformed config fails closed. Closes the last software checkbox of #9415; the history rewrite, Jules workflows and SPEC cap remain owner decisions.

### DL-#10385 · Pinocchio Driver/Iron G2-G3 Continuation and Independent Replay (MS-111)

- **State:** abandoned
- **Owner:** local
- **Issue:** #10385 (MS-111, epic #10363; folded PF-07 #10437)
- **Branch:** feat/10385-ms111-pinocchio-g2-g3-replay
- **PR:** #10723
- **Paths:** src/shared/python/motion_matching/pinocchio_g2_g3.py; src/engines/physics_engines/pinocchio/python/full_body_fit.py; tests/unit/motion_matching/test_pinocchio_g2_g3.py; docs/development/matched_swing_program/evidence/ms111/; docs/development/matched_swing_program/README.md; docs/development/matched_swing_program/MS31_PINOCCHIO_CROCODDYL_TURNOVER.md; docs/development/DEVELOPMENT_LOG.md; SPEC.md
- **Started:** 2026-09-22
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Software contracts for Pinocchio driver/iron G2/G3 continuation schedules, same-integrator solve/replay parity (reject integrator-specific solutions), armature propagation to equivalent-model independent replay packages (save/reopen), failed-continuation evidence preservation, open-loop q0/v0 control feed (no per-frame poses), and PF-07 robustness roster. Fitter exposes `--ms111-schedule` and fail-closed integrator parity. Native G2/G3 qualification remains blocked on MS-107 accepted G1 + MS-100 receipts.
- **Evidence:** docs/development/matched_swing_program/evidence/ms111/continuation_contract_status.json; tests/unit/motion_matching/test_pinocchio_g2_g3.py.

### DL-#10271 · Restore End-to-End Provenance for Hip-Calibrated Motion Evidence

- **State:** abandoned
- **Owner:** local
- **Issue:** #10271 (child of #10254; blocks #10162 physical acceptance provenance)
- **Branch:** fix/10271-hipcal-provenance
- **PR:** #10722
- **Paths:** src/shared/python/motion_matching/pipeline/receipt_provenance.py; src/shared/python/motion_matching/pipeline/receipt.py; src/shared/python/motion_matching/pipeline/receipt_schema.py; tests/unit/motion_matching/pipeline/test_receipt_provenance_chain.py; docs/development/full_body_models/evidence/ground_support/anthro_driver/receipt.json; docs/development/full_body_models/evidence/ground_support/anthro_iron/receipt.json; docs/development/full_body_models/RECEIPTS.md; docs/development/full_body_models/evidence/ground_support/CANONICAL_RUN.md
- **Started:** 2026-09-22
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Shared `validate_receipt_provenance_chain` (`receipt-provenance-chain/1`) fail-closes missing/stale de Leva, base canonical, and final raw/canonical digests; CI gates `anthro_driver`/`anthro_iron`; producer emits `spec_canonical_sha256`; receipts re-anchored to committed bases/scaled specs that already embed the current table; intermediate hipcal docs not fabricated; physical RMS unchanged/unqualified.

### DL-#10378 · Full-Swing Qualification for All Six Engines (MS-104)

- **State:** abandoned
- **Owner:** local
- **Issue:** #10378 (MS-104, epic #10363; folded MS-109/110/112 owner blockers)
- **Branch:** feat/ms104-full-swing-qualification
- **PR:** #10707
- **Paths:** src/shared/python/motion_matching/full_swing_qualification.py; tests/unit/motion_matching/test_full_swing_qualification.py; docs/plans/matched_swing/evidence/ms104_full_swing_qualification.json; docs/development/HANDOFF.md; docs/development/DEVELOPMENT_LOG.md; SPEC.md
- **Started:** 2026-09-22
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Software-contract qualification matrix over 6 engines × driver/iron × G1/G2/G3. Rows require MS-100 acceptance, MS-72 conformance, native replay, and numerical-convergence links; reduced Simscape oracle stays partial. Release stays blocked until every required cell is fully linked — no invented six-engine native pass.

### DL-#9545 · Versioned BunkerShot3D Workbench API Route

- **State:** shipped
- **Shipped:** 2026-10-15
- **Owner:** claude
- **Issue:** #9545 (epic #9541)
- **Branch:** claude/issue-9545-api
- **PR:** #11516 (draft), #12449, #12495, #12440, #12506, #12515, #12519, #12522, #12535, #12541
- **Paths:** src/api/routes/bunker_workbench.py; src/tools/bunker_shot_gui/input_ranges.py; src/tools/bunker_shot_gui/design.py; src/tools/bunker_shot_gui/widgets.py; src/tools/bunker_shot_gui/report.py; src/tools/bunker_shot_gui/render.py; src/config/feature_parity.json; tests/bunkershot3d/test_bunker_workbench_api.py
- **Started:** 2026-10-04
- **Last verified:** 2026-10-15 (`09b41bd3`; collated from changes/9545-bunkershot3d-particle-playback-panel-emb.md)
- **Summary:** The headless workbench API enforces request ranges, handedness and source-stamped verdicts; predictive playability is refused or reported without ranking. UI and API share input-range authority.

### DL-#10590 · TB-05: Fit and Replay the Hub–Arm–Club Triple Pendulum

- **State:** abandoned
- **Owner:** local
- **Issue:** #10590 (parent #10584)
- **Branch:** feat/tb05-triple-pendulum-fit-10590
- **PR:** #10644
- **Paths:** src/engines/physics_engines/pendulum/python/motion_matching/adapters_triple.py; src/engines/physics_engines/pendulum/python/motion_matching/torque_optimization_triple.py; src/engines/physics_engines/pendulum/python/motion_matching/provider_triple.py; src/engines/physics_engines/pendulum/python/motion_matching/qualification_triple.py; tests/unit/engines/physics_engines/pendulum/test_triple_pendulum_fit.py; docs/plans/tour_baselines/evidence/tb05_driver_qualification_receipt.json; docs/plans/tour_baselines/evidence/tb05_iron_qualification_receipt.json
- **Started:** 2026-09-21
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Driver/iron triple-pendulum fits use bounded controls, positive geometry and independent tighter replay; generated baseline packages and receipts remain the recorded evidence boundary.
- **Evidence:** docs/plans/tour_baselines/evidence/tb05_driver_qualification_receipt.json; docs/plans/tour_baselines/evidence/tb05_iron_qualification_receipt.json; tests/unit/engines/physics_engines/pendulum/test_triple_pendulum_fit.py.

### DL-#9550 · Impact Explorer Acceptance Matrix and Served-Bundle Verification

- **State:** shipped
- **Shipped:** 2026-10-14
- **Owner:** claude
- **Issue:** #9550 (epic #9546)
- **Branch:** conductor/issue-9550
- **PR:** #10441, #12471
- **Paths:** src/config/impact_acceptance.json; tests/config/impact_acceptance/test_impact_acceptance_matrix.py; scripts/ci/verify_impact_explorer_bundle.py; tests/ci/test_verify_impact_explorer_bundle.py; tests/api/test_impact_explorer_mount.py; .github/workflows/ci-standard.yml; src/config/feature_parity.json; docs/development/impact_acceptance_matrix.md
- **Started:** 2026-09-18
- **Last verified:** 2026-10-14 (`126b0964`; collated from changes/9550-add-frozen-impactmodelcapabilitymatrix-a.md)
- **Summary:** Froze the model-capability matrix for `src/shared/python/physics/impact_model` (spin support stated as unavailable where absent), recorded the six acceptance items with evidence or explicit open state and a `code_verified_only` claim, made CI stamp Tools' release artifacts with the pinned gitlink and verify that `/impact-explorer-app/` serves that revision's real JavaScript, and downgraded `tools.rate_of_closure` parity to an evidence-based gap.
- **Evidence:** docs/development/impact_acceptance_matrix.md; tests/config/impact_acceptance/test_impact_acceptance_matrix.py.

### DL-#10402 · OpenSim Qualify Muscle and Tendon Extensions Without Replacing Baseline

- **State:** abandoned
- **Owner:** local
- **Issue:** #10402 (epic #10394 / #10363, OG-08)
- **Branch:** feat/og08-qualify-muscle-tendon-extensions-10402
- **PR:** #10413
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/muscle_qualification.py; src/engines/physics_engines/opensim/python/tour_matching/**init**.py; tests/opensim/test_muscle_cmc.py; src/engines/physics_engines/opensim/python/POST_MVP_MUSCLES.md; docs/development/DEVELOPMENT_LOG.md; docs/development/HANDOFF.md
- **Started:** 2026-09-18
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Qualified muscle and tendon extensions (`muscle_qualification.py`) without replacing the torque baseline. Provides explicit anatomy coverage audit (disallowing lower-extremity-only models from claiming full golf capability), parameter provenance and licensing audits, MTU path and wrapping verification, moment arm vs. finite-difference path-length derivative validation, activation dynamics, initial tendon equilibrium, and short replay receipts reporting reserve torques and pelvic residuals. Keeps epic muscle-complete status open pending independent #10375 validation.
- **Evidence:** tests/opensim/test_muscle_cmc.py; src/engines/physics_engines/opensim/python/tour_matching/muscle_qualification.py.

### DL-#10400 · OpenSim Rebuild Full-Swing Tracking From Qualified Address

- **State:** abandoned
- **Owner:** local
- **Issue:** #10400 (epic #10394 / #10363, OG-06)
- **Branch:** feat/og06-rebuild-full-swing-tracking-10400
- **PR:** #10410
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/full_swing_tracking.py; src/engines/physics_engines/opensim/python/tour_matching/**init**.py; tests/opensim/test_moco_g1_ladder.py; docs/development/DEVELOPMENT_LOG.md; docs/development/HANDOFF.md
- **Started:** 2026-09-18
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Rebuilt OpenSim full-swing tracking qualification module (`full_swing_tracking.py`) reinitialized from qualified address pose $q_0$ and model hash with frozen marker calibration. Implements ladder progression, continuity, full-swing coordinate limits, dynamic bilateral grip closure, ground contact mechanics, and separate receipts under MS-100 / MS-104.
- **Evidence:** tests/opensim/test_moco_g1_ladder.py; src/engines/physics_engines/opensim/python/tour_matching/full_swing_tracking.py.

### DL-#10399 · OpenSim Calibrate and Match Two-Handed Address Pose

- **State:** abandoned
- **Owner:** local
- **Issue:** #10399 (epic #10394 / #10363, OG-05)
- **Branch:** feat/og05-match-two-handed-address-10399
- **PR:** #10409
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/address.py; src/engines/physics_engines/opensim/python/tour_matching/marker_calibration.py; src/engines/physics_engines/opensim/python/tour_matching/**init**.py; tests/opensim/test_golf_address.py; tests/opensim/test_marker_calibration.py; SPEC.md
- **Started:** 2026-09-18
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Calibrated and qualified two-handed golf address pose (`address.py`) on OpenSim humanoid model against canonical tour capture. Validates bilateral grip closure between lead hand and club shaft, foot ground support, posture metrics, and coordinate limits against model XML ranges.
- **Evidence:** tests/opensim/test_golf_address.py; tests/opensim/test_marker_calibration.py.

### DL-#10398 · OpenSim Capture Registration and Golf Camera Views

- **State:** abandoned
- **Owner:** local
- **Issue:** #10398 (epic #10394 / #10363, OG-04)
- **Branch:** feat/og04-qualify-registration-camera-10398
- **PR:** #10408
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/registration.py; src/engines/physics_engines/opensim/python/tour_matching/visualization.py; src/engines/physics_engines/opensim/python/tour_matching/**init**.py; tests/opensim/test_golf_registration.py; SPEC.md
- **Started:** 2026-09-18
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Implemented capture registration and qualified golf camera viewpoints (`registration.py`). Provides rigid landmark alignment without scaling or shearing, ground support plane alignment, and down-the-line / front / side / overhead camera views in `visualization.py`. Validated invariant over model states and simulation outputs.
- **Evidence:** tests/opensim/test_golf_registration.py.

### DL-#10396 · OpenSim Anatomically and Physically Consistent Segment Scaling

- **State:** abandoned
- **Owner:** local
- **Issue:** #10396 (epic #10394 / #10363, OG-02)
- **Branch:** feat/og02-consistent-segment-scaling-10396
- **PR:** #10407
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/segment_scaling.py; src/engines/physics_engines/opensim/python/tour_matching/scale.py; src/engines/physics_engines/opensim/python/tour_matching/**init**.py; src/engines/physics_engines/opensim/models/golf_humanoid_scaled.osim; docs/development/opensim_tour_matching/os3b_scale_and_full_ik.py; tests/opensim/test_segment_scale.py; tests/opensim/test_opensim_os0_qualification.py; SPEC.md
- **Started:** 2026-09-18
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Implemented consistent segment scaling module (`segment_scaling.py`) and acromion proxy reconstruction for unilateral marker occlusions (`scale.py`). Joint frames, visual meshes, COM, and inertia are scaled in lockstep. Regenerated qualified `golf_humanoid_scaled.osim` with attached visual club and consistent scaling, passing qualification gates without unscaled arm mesh defects.
- **Evidence:** tests/opensim/test_segment_scale.py; tests/opensim/test_opensim_os0_qualification.py; src/engines/physics_engines/opensim/models/golf_humanoid_scaled.osim.

### DL-#10003 · OpenSim Tour-Average Full-Body Matching

- **State:** abandoned
- **Owner:** claude
- **Issue:** #10003 (parent #9921; sibling full-body epic #10062)
- **Branch:** feat/full-body-opensim-epic
- **PR:** #10071
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching; src/shared/python/motion_matching/tour_capture_contract.py; tests/opensim; docs/development/opensim_tour_matching
- **Started:** 2026-09-12
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Frozen capture contract, marker-to-body map, TRC export verified by OpenSim, MarkerSet authoring with coordinate unlocking, and alternating placement/IK calibration are implemented test-first; runtime qualified; OS-3 kinematic feasibility measured. Moco tracking and sextic effort fitting not started.
- **Evidence:** docs/development/opensim_tour_matching/HANDOFF.md and evidence/os1_trc_receipt.json, os2_runtime_receipt.json, os3_stride20/, os3_unlocked_stride20/.

### DL-#13011 · Stop Realtime File-Tail Poll Thread When Last Subscriber Unsubscribes

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13011
- **Branch:** merged via #13600
- **PR:** #13600
- **Paths:** see #13600
- **Started:** 2026-10-21
- **Last verified:** 2026-10-21 (`93a3db81`; collated from changes/13011-stop-realtime-file-tail-poll-thread-when.md)
- **Summary:** Stop realtime file-tail poll thread when last subscriber unsubscribes
- **Next step:** Shipped in PR #13600.

### DL-#13573 · Merge-Queue-Safe Workflow Cancel-in-Progress Audit and Gh-Readonly-Queue Push Ignores (UD-CI-032).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13573
- **Branch:** merged via #13584
- **PR:** #13584
- **Paths:** see #13584
- **Started:** 2026-10-21
- **Last verified:** 2026-10-21 (`e30806f5`; collated from changes/13573-merge-queue-safe-workflow-cancel-in-prog.md)
- **Summary:** Merge-queue-safe workflow cancel-in-progress audit and gh-readonly-queue push ignores (UD-CI-032).
- **Next step:** Shipped in PR #13584.

### DL-#13575 · Retire Jules Push/PR Noise Workflows Batch 1 With Archived Rollback Copies and Post-Merge Gh Disable.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13575
- **Branch:** merged via #13586
- **PR:** #13586
- **Paths:** see #13586
- **Started:** 2026-10-21
- **Last verified:** 2026-10-21 (`483b7813`; collated from changes/13575-retire-jules-push-pr-noise-batch-1.md)
- **Summary:** Retire Jules push/PR noise workflows batch 1 with archived rollback copies and post-merge gh disable.
- **Next step:** Shipped in PR #13586.

### DL-#13582 · Add OpenSim CI Failure Triage Snapshot, Doc, and Classifier Script for UD-CI-002T.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13582
- **Branch:** merged via #13589
- **PR:** #13589
- **Paths:** see #13589
- **Started:** 2026-10-21
- **Last verified:** 2026-10-21 (`c9caab9c`; collated from changes/13582-add-opensim-ci-failure-triage-snapshot-d.md)
- **Summary:** Add OpenSim CI failure triage snapshot, doc, and classifier script for UD-CI-002t.
- **Next step:** Shipped in PR #13589.

### DL-#13572 · Move Cross-Engine Leaderboard JSON Publish Off PR Triggers (Nightly and Main Push Only).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13572
- **Branch:** merged via #13583
- **PR:** #13583
- **Paths:** see #13583
- **Started:** 2026-10-21
- **Last verified:** 2026-10-21 (`9eae13f1`; collated from changes/13572-move-cross-engine-leaderboard-json-publi.md)
- **Summary:** Move cross-engine leaderboard JSON publish off PR triggers (nightly and main push only).
- **Next step:** Shipped in PR #13583.

### DL-#13563 · Bump Pillow to 13.0.0 and Certifi to 2026.10.19 to Clear Pip-Audit CVEs.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13563
- **Branch:** merged via #13587
- **PR:** #13587
- **Paths:** see #13587
- **Started:** 2026-10-21
- **Last verified:** 2026-10-21 (`e73f47e2`; collated from changes/13563-bump-pillow-to-13-0-0-and-certifi-to-202.md)
- **Summary:** Bump pillow to 13.0.0 and certifi to 2026.10.19 to clear pip-audit CVEs.
- **Next step:** Shipped in PR #13587.

### DL-#13557 · Grip Topology Parity Compares Numerically-Zero Reference Channels (Peak <= 1E-8 N or N·m) Absolutely Instead of Relatively; Fixes myoHand Frozen-Finger Law-Level Parity on Roundoff Spin Channels

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13557
- **Branch:** merged via #13558
- **PR:** #13558
- **Paths:** see #13558
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`b53903ee`; collated from changes/13557-grip-parity-numerical-zero-reference.md)
- **Summary:** Grip topology parity compares numerically-zero reference channels (peak <= 1e-8 N or N·m) absolutely instead of relatively; fixes myoHand frozen-finger law-level parity on roundoff spin channels
- **Next step:** Shipped in PR #13558.

### DL-#13544 · Grip_Contact Exporter --Check Now Logs the Drifted File and a Bounded Unified Diff (Flake Diagnostics)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13544
- **Branch:** merged via #13546
- **PR:** #13546, #13602
- **Paths:** see #13546
- **Started:** 2026-10-20
- **Last verified:** 2026-10-21 (`9fb90ed9`; collated from changes/13544-serialize-simscape-grip-committed-path-g.md)
- **Summary:** grip_contact exporter --check now logs the drifted file and a bounded unified diff (flake diagnostics)
- **Next step:** Shipped in PR #13546.

### DL-#13334 · Add Stateless Simscape Physical Grip Contact Codegen Kernel, GS3DX_GripContact Builder/Runner, and M12 Parity Harness

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13334
- **Branch:** merged via #13529
- **PR:** #13529, #13587
- **Paths:** see #13529
- **Started:** 2026-10-20
- **Last verified:** 2026-10-21 (`e73f47e2`; collated from changes/13334-canonicalize-grip-contact-codegen-floats.md)
- **Summary:** Add stateless Simscape physical grip contact codegen kernel, GS3DX_GripContact builder/runner, and M12 parity harness
- **Next step:** Shipped in PR #13529.

### DL-#13466 · Drake Native Golfer Grip Tests Pass With Real Pydrake: Default Loader Parses the Generator Humanoid (Not the #13312 Canonical Golfer), Actuator Retrofit No Longer Calls Num_Velocities() Before Finalize, humanoid_urdf.py Split Under 1200 Lines and Its Size Exceptions Removed

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13466
- **Branch:** merged via #13520
- **PR:** #13520
- **Paths:** see #13520
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`b9a397d6`; collated from changes/13466-drake-native-golfer-grip-tests-pass-with.md)
- **Summary:** Drake native golfer grip tests pass with real pydrake: default loader parses the generator humanoid (not the #13312 canonical golfer), actuator retrofit no longer calls num_velocities() before Finalize, humanoid_urdf.py split under 1200 lines and its size exceptions removed
- **Next step:** Shipped in PR #13520.

### DL-#13508 · Symmetric Capsule Line-Contact Quadrature Integrates the Shorter Capsule Axis (#13508)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13508
- **Branch:** merged via #13517
- **PR:** #13517
- **Paths:** see #13517
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`f7d5547a`; collated from changes/13508-symmetric-capsule-line-contact-quadratur.md)
- **Summary:** Symmetric capsule line-contact quadrature integrates the shorter capsule axis (#13508)
- **Next step:** Shipped in PR #13517.

### DL-#13346 · Reclassify Grip_Style as a Physical Pick Across CLI, API, Desktop and Web and Migrate Legacy Sidecars

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13346
- **Branch:** merged via #13472
- **PR:** #13472
- **Paths:** see #13472
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`53919d4d`; collated from changes/13346-reclassify-grip-style-as-a-physical-pick.md)
- **Summary:** Reclassify grip_style as a physical pick across CLI, API, desktop and web and migrate legacy sidecars
- **Next step:** Shipped in PR #13472.

### DL-#13510 · Ci(Opensim): Real OpenSim Job Reuses a Validated Per-Runner Venv and Opensim-Models Clone and Stops Uploading the GitHub Pip Cache From Self-Hosted Runners

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13510
- **Branch:** merged via #13511
- **PR:** #13511
- **Paths:** see #13511
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`d8d85a52`; collated from changes/13510-ci-opensim-real-opensim-job-reuses-a-val.md)
- **Summary:** ci(opensim): Real OpenSim job reuses a validated per-runner venv and opensim-models clone and stops uploading the GitHub pip cache from self-hosted runners
- **Next step:** Shipped in PR #13511.

### DL-#13500 · Fix Duplicate Tex_Skin_2 Asset in MuJoCo With_Head Exports so the MJCF Compiles

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13500
- **Branch:** merged via #13505
- **PR:** #13505
- **Paths:** see #13505
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`94b8e96b`; collated from changes/13500-fix-duplicate-tex-skin-2-asset-in-mujoco.md)
- **Summary:** Fix duplicate tex_skin_2 asset in MuJoCo with_head exports so the MJCF compiles
- **Next step:** Shipped in PR #13505.

### DL-#13175 · HRG-E4 Slice: Anatomical-Angle Event Table (Address/Top/Impact/Finish) With CSV/JSON Export Carrying Units, Sign Conventions, Observability and Receipts; JSON Schema Validated

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13175
- **Branch:** merged via #13503
- **PR:** #13503, #13538, #13541
- **Paths:** see #13503
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`527c99ad`; collated from changes/13175-hrg-e4-pelvis-reversal-trunk-terms.md)
- **Summary:** HRG-E4 slice: anatomical-angle event table (address/top/impact/finish) with CSV/JSON export carrying units, sign conventions, observability and receipts; JSON schema validated
- **Next step:** Shipped in PR #13503.

### DL-#13174 · MuJoCo Visual Layer for Full-Body-V2: One Rigid Pelvis, Abdomen and Thorax Visual per Trunk Segment, Seam No-Gap and Physics-Unchanged Tests (HRG-D3 Slice; Renders Deferred)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13174
- **Branch:** merged via #13499
- **PR:** #13499
- **Paths:** see #13499
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`396a999f`; collated from changes/13174-mujoco-visual-layer-for-full-body-v2-one.md)
- **Summary:** MuJoCo visual layer for full-body-v2: one rigid pelvis, abdomen and thorax visual per trunk segment, seam no-gap and physics-unchanged tests (HRG-D3 slice; renders deferred)
- **Next step:** Shipped in PR #13499.

### DL-#13494 · Stop CI Requires OpenSim Queueing One Run per Main Push

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13494
- **Branch:** ci/opensim-main-concurrency
- **PR:** #13496
- **Paths:** see #13496
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`2e64f840`; collated from changes/13494-stop-ci-requires-opensim-queueing-one-ru.md)
- **Summary:** Stop CI Requires OpenSim queueing one run per main push
- **Next step:** Shipped in PR #13496.

### DL-#13017 · Heavy Lanes: Install Real Pin/Pin-Pink Wheels and Isolate Each Lane in a Venv

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13017
- **Branch:** merged via #13476
- **PR:** #13476
- **Paths:** see #13476
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`1aadc830`; collated from changes/13017-heavy-lanes-install-real-pin-pin-pink-wh.md)
- **Summary:** Heavy lanes: install real pin/pin-pink wheels and isolate each lane in a venv
- **Next step:** Shipped in PR #13476.

### DL-#13454 · Remove Dead Pinocchio Default Branch From Drake Simulate DEFAULT_GOLFER_URDF

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13454
- **Branch:** fix/ud-drake-bugs
- **PR:** #13468
- **Paths:** see #13468
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`e541e40b`; collated from changes/13454-remove-dead-pinocchio-default-branch-fro.md)
- **Summary:** Remove dead Pinocchio default branch from Drake simulate DEFAULT_GOLFER_URDF
- **Next step:** Shipped in PR #13468.

### DL-#13451 · Guard DrakePhysicsEngine.Get_Joint_Names Against Unfinalized Plants

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13451
- **Branch:** fix/ud-drake-bugs
- **PR:** #13468
- **Paths:** see #13468
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`e541e40b`; collated from changes/13451-guard-drakephysicsengine-get-joint-names.md)
- **Summary:** Guard DrakePhysicsEngine.get_joint_names against unfinalized plants
- **Next step:** Shipped in PR #13468.

### DL-#13450 · Resolve the Drake Autodiff Fit Grip Body Through the Shared Simulate Fallback

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13450
- **Branch:** fix/ud-drake-bugs
- **PR:** #13468
- **Paths:** see #13468
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`e541e40b`; collated from changes/13450-resolve-the-drake-autodiff-fit-grip-body.md)
- **Summary:** Resolve the Drake autodiff fit grip body through the shared simulate fallback
- **Next step:** Shipped in PR #13468.

### DL-#13449 · Size Drake-Simscape Equivalence Fixture Theta From the Plant Actuator Count

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13449
- **Branch:** fix/ud-drake-bugs
- **PR:** #13468, #13515
- **Paths:** see #13468
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`f0d94751`; collated from changes/13449-add-headless-regression-test-for-drake-s.md)
- **Summary:** Size Drake-Simscape equivalence fixture theta from the plant actuator count
- **Next step:** Shipped in PR #13468.

### DL-#13043 · Add OpenSim 3D Joint Reaction Analysis for Barbell Lifts (OSL-5)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13043
- **Branch:** merged via #13415
- **PR:** #13415
- **Paths:** see #13415
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`b030576f`; collated from changes/13043-add-opensim-3d-joint-reaction-analysis-f.md)
- **Summary:** Add OpenSim 3D joint reaction analysis for barbell lifts (OSL-5)
- **Next step:** Shipped in PR #13415.

### DL-#13460 · Relax Last-Ulp Geometry Assertions in Batched Grip Contact Tests (Linux BLAS Dot Rounding)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13460
- **Branch:** merged via #13473
- **PR:** #13473, #13469
- **Paths:** see #13473
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`844aea4a`; collated from changes/13460-mark-the-torch-cuda-wheel-index-pip-prob.md)
- **Summary:** Relax last-ulp geometry assertions in batched grip contact tests (Linux BLAS dot rounding)
- **Next step:** Shipped in PR #13473.

### DL-#13453 · Test the MyoSuite Gymnasium-First Import Behaviourally in _Engine_Init

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13453
- **Branch:** fix/ud-myosuite-gym-tests
- **PR:** #13471
- **Paths:** see #13471
- **Started:** 2026-10-20
- **Last verified:** 2026-10-20 (`1402b709`; collated from changes/13453-test-myosuite-gymnasium-import-behaviour.md)
- **Summary:** Test the MyoSuite gymnasium-first import behaviourally in _engine_init
- **Next step:** Shipped in PR #13471.

### DL-#13396 · HRG Reference: Record That an Analytic Radau Jacobian Is Not Worth Building for the Pinocchio Grip Sim (Num_Jac Is 5-15% of CPU; Newton Iterations Dominate)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13396
- **Branch:** merged via #13422
- **PR:** #13422, #13480
- **Paths:** see #13422
- **Started:** 2026-10-19
- **Last verified:** 2026-10-20 (`9c1c578f`; collated from changes/13396-restore-the-13396-grip-follow-ups-lost-i.md)
- **Summary:** HRG reference: record that an analytic Radau Jacobian is not worth building for the Pinocchio grip sim (num_jac is 5-15% of CPU; Newton iterations dominate)
- **Next step:** Shipped in PR #13422.

### DL-#13384 · Batch One Side of the Grip Topology Contact Law: Validate Once, Python-Float Row Kernel, 9.9X Faster Topology_Side_Wrench With Bit-Identical Parity

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13384
- **Branch:** merged via #13422
- **PR:** #13422
- **Paths:** see #13422
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`2605c9ef`; collated from changes/13384-batch-one-side-of-the-grip-topology-cont.md)
- **Summary:** Batch one side of the grip topology contact law: validate once, Python-float row kernel, 9.9x faster topology_side_wrench with bit-identical parity
- **Next step:** Shipped in PR #13422.

### DL-#13336 · Extract Single Array-Namespace Contact Kernel in Motion_Matching Shared by Contact_Law, Jax_Contact, and Grip_Contact

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13336
- **Branch:** feat/13336-array-namespace-contact-kernel
- **PR:** #13422
- **Paths:** see #13422
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`2605c9ef`; collated from changes/13336-extract-pure-array-namespace-contact-ker.md)
- **Summary:** Extract single array-namespace contact kernel in motion_matching shared by contact_law, jax_contact, and grip_contact
- **Next step:** Shipped in PR #13422.

### DL-#13335 · Emit Grip-Binding/V1 Metadata on Club-Composed Character Builder Exports, Fail Closed on Clubless Exports, and Enforce 100% Model-Library Grip Binding Admission

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13335
- **Branch:** merged via #13422
- **PR:** #13422
- **Paths:** see #13422
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`2605c9ef`; collated from changes/13335-emit-grip-binding-v1-metadata-on-club-co.md)
- **Summary:** Emit grip-binding/v1 metadata on club-composed character builder exports, fail closed on clubless exports, and enforce 100% model-library grip binding admission
- **Next step:** Shipped in PR #13422.

### DL-#13333 · Export Deterministic Simscape GolfSwing3D Grip Topology and Allocate Pad and Hand-Hand Forces From Recorded States With Zero Injected Forces

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13333
- **Branch:** feat/13333-simscape-grip-topology-export
- **PR:** #13422
- **Paths:** see #13422
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`2605c9ef`; collated from changes/13333-export-deterministic-simscape-golfswing3.md)
- **Summary:** Export deterministic Simscape GolfSwing3D grip topology and allocate pad and hand-hand forces from recorded states with zero injected forces
- **Next step:** Shipped in PR #13422.

### DL-#13332 · Add Explicit Bilateral Lead-Hand Bushing Attachment and NativeDrakeGolferGripAdapter for the Native Drake Golfer (HRG-GS-M10).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13332
- **Branch:** merged via #13422
- **PR:** #13422
- **Paths:** see #13422
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`2605c9ef`; collated from changes/13332-add-explicit-bilateral-lead-hand-bushing.md)
- **Summary:** Add explicit bilateral lead-hand bushing attachment and NativeDrakeGolferGripAdapter for the native Drake golfer (HRG-GS-M10).
- **Next step:** Shipped in PR #13422.

### DL-#13331 · Add Native Pinocchio Golfer Separate-Hand Variant and NativePinocchioGolferGripAdapter (HRG-GS-M9).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13331
- **Branch:** merged via #13422
- **PR:** #13422
- **Paths:** see #13422
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`2605c9ef`; collated from changes/13331-add-native-pinocchio-golfer-separate-han.md)
- **Summary:** Add native Pinocchio golfer separate-hand variant and NativePinocchioGolferGripAdapter (HRG-GS-M9).
- **Next step:** Shipped in PR #13422.

### DL-#13330 · Add MyoHub myoHand Articulated Bilateral Grip Fixture and Segment-Mapped Phalanx Adapter (HRG-GS-M8)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13330
- **Branch:** merged via #13422
- **PR:** #13422, #13445
- **Paths:** see #13422
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`2717ba32`; collated from changes/13330-regulate-bilateral-myohand-grip-trajecto.md)
- **Summary:** Add MyoHub myoHand articulated bilateral grip fixture and segment-mapped phalanx adapter (HRG-GS-M8)
- **Next step:** Shipped in PR #13422.

### DL-#13329 · Add MyoSuiteGolferGripAdapter, Grip-Binding Manifests, and Fix MyoSuite Club Shaft Geometry

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13329
- **Branch:** merged via #13422
- **PR:** #13422
- **Paths:** see #13422
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`2605c9ef`; collated from changes/13329-add-myosuitegolfergripadapter-grip-bindi.md)
- **Summary:** Add MyoSuiteGolferGripAdapter, grip-binding manifests, and fix MyoSuite club shaft geometry
- **Next step:** Shipped in PR #13422.

### DL-#13328 · Add MuJoCo Shadow/Allegro Grip Laboratory Adapter With Phalanx-Mapped Sites and Physical Hand-Hand Contact

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13328
- **Branch:** merged via #13422
- **PR:** #13422
- **Paths:** see #13422
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`2605c9ef`; collated from changes/13328-add-mujoco-shadow-allegro-grip-laborator.md)
- **Summary:** Add MuJoCo Shadow/Allegro grip laboratory adapter with phalanx-mapped sites and physical hand-hand contact
- **Next step:** Shipped in PR #13422.

### DL-#13327 · Add Native MuJoCo Humanoid Golf Grip Adapter for Rigid-Hand MJCFs (HRG-GS-M5)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13327
- **Branch:** feat/13327-mujoco-humanoid-grip-adapter
- **PR:** #13422
- **Paths:** see #13422
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`2605c9ef`; collated from changes/13327-add-native-mujoco-humanoid-golf-grip-ada.md)
- **Summary:** Add native MuJoCo humanoid golf grip adapter for rigid-hand MJCFs (HRG-GS-M5)
- **Next step:** Shipped in PR #13422.

### DL-#13326 · Add Native OpenSim MSK Grip Adapter With Weld/Bushing Binding, Bruno 2015 Admission, and M3 Qualification

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13326
- **Branch:** merged via #13422
- **PR:** #13422
- **Paths:** see #13422
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`2605c9ef`; collated from changes/13326-add-native-opensim-msk-grip-adapter-with.md)
- **Summary:** Add native OpenSim MSK grip adapter with weld/bushing binding, Bruno 2015 admission, and M3 qualification
- **Next step:** Shipped in PR #13422.

### DL-#13325 · Add Cross-Model Grip Qualification Harness With 3-State Availability Envelope and Pair-Level Parity (HRG-GS-M3)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13325
- **Branch:** feat/13325-grip-qualification-harness
- **PR:** #13422
- **Paths:** see #13422
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`2605c9ef`; collated from changes/13325-add-cross-model-grip-qualification-harne.md)
- **Summary:** Add cross-model grip qualification harness with 3-state availability envelope and pair-level parity (HRG-GS-M3)
- **Next step:** Shipped in PR #13422.

### DL-#13324 · Add Versioned Per-Model Grip Binding Manifests and Frame/Unit Validator (HRG-GS-M2)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13324
- **Branch:** merged via #13422
- **PR:** #13422
- **Paths:** see #13422
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`2605c9ef`; collated from changes/13324-add-versioned-per-model-grip-binding-man.md)
- **Summary:** Add versioned per-model grip binding manifests and frame/unit validator (HRG-GS-M2)
- **Next step:** Shipped in PR #13422.

### DL-#13323 · Add Shared Model Grip Binding Contract, GripProvider Protocol, and PinocchioGripAdapter (HRG-GS-M1)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13323
- **Branch:** merged via #13422
- **PR:** #13422
- **Paths:** see #13422
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`2605c9ef`; collated from changes/13323-add-shared-model-grip-binding-contract-g.md)
- **Summary:** Add shared model grip binding contract, GripProvider protocol, and PinocchioGripAdapter (HRG-GS-M1)
- **Next step:** Shipped in PR #13422.

### DL-#13309 · Separate LHand and RHand in Full-Body-V2 (Opt-In) and Evaluate Physical Hand-Hand Contact Pairs

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13309
- **Branch:** merged via #13422
- **PR:** #13422
- **Paths:** see #13422
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`2605c9ef`; collated from changes/13309-separate-lhand-and-rhand-in-full-body-v2.md)
- **Summary:** Separate LHand and RHand in full-body-v2 (opt-in) and evaluate physical hand-hand contact pairs
- **Next step:** Shipped in PR #13422.

### DL-#13308 · Extend Grip_Contact.Parity With Per-Pair Forces, Contact-Point Drift, Determinate/Indeterminate Classification, and Cross-Engine Style Parity Report (HRG-GS-G10)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13308
- **Branch:** feat/13308-grip-style-parity-report
- **PR:** #13422, #13442, #13457
- **Paths:** see #13422
- **Started:** 2026-10-19
- **Last verified:** 2026-10-20 (`5884e94a`; collated from changes/13308-continuous-capsule-line-contact.md)
- **Summary:** Extend grip_contact.parity with per-pair forces, contact-point drift, determinate/indeterminate classification, and cross-engine style parity report (HRG-GS-G10)
- **Next step:** Shipped in PR #13457.

### DL-#13307 · Refactor Drake ClubInHands to Two Hand Bodies and Add TopologyContactModel Adapter

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13307
- **Branch:** feat/13307-drake-two-hand-grip
- **PR:** #13422
- **Paths:** see #13422
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`2605c9ef`; collated from changes/13307-refactor-drake-clubinhands-to-two-hand-b.md)
- **Summary:** Refactor Drake ClubInHands to two hand bodies and add TopologyContactModel adapter
- **Next step:** Shipped in PR #13422.

### DL-#13306 · Add MuJoCo ContactTopology Grip-Contact Adapter and Parity Tests (HRG-GS-G8)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13306
- **Branch:** merged via #13422
- **PR:** #13422, #13442, #13504
- **Paths:** see #13422
- **Started:** 2026-10-19
- **Last verified:** 2026-10-20 (`ec754166`; collated from changes/13306-mujoco-friction-secant-schedule.md)
- **Summary:** Add MuJoCo ContactTopology grip-contact adapter and parity tests (HRG-GS-G8)
- **Next step:** Review and merge; overlap/interlock 0.10 s trail-hand deflection (4.60% hold, 3.39% swing RMS) stays a strict xfail; OpenSim ten_finger 0.04 s gaps remain

### DL-#13305 · Calibrate OpenSim Topology Grip Foundation at 2X Preload With Hollars Friction Mapping and Preserve SimmSpline Horizon Across Short Swing Windows

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13305
- **Branch:** merged via #13422
- **PR:** #13422, #13458
- **Paths:** see #13422
- **Started:** 2026-10-19
- **Last verified:** 2026-10-20 (`ac3676bb`; collated from changes/13305-fix-opensim-capsule-contact-mesh-ring-al.md)
- **Summary:** Calibrate OpenSim topology grip foundation at 2x preload with Hollars friction mapping and preserve SimmSpline horizon across short swing windows
- **Next step:** Shipped in PR #13422.

### DL-#13045 · Kinetics Parity for All Registered Lifts

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13045
- **Branch:** merged via #13443
- **PR:** #13443
- **Paths:** see #13443
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`dcd7e47e`; collated from changes/13045-kinetics-parity-for-all-registered-lifts.md)
- **Summary:** kinetics parity for all registered lifts
- **Next step:** Shipped in PR #13443.

### DL-#13423 · Declare Websockets for Realtime Python Backend and Document Qualisys C3D/TRC Export Path in formats.md

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13423
- **Branch:** merged via #13441
- **PR:** #13441
- **Paths:** see #13441
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`bf993892`; collated from changes/13423-declare-websockets-for-realtime-python-b.md)
- **Summary:** Declare websockets for realtime Python backend and document Qualisys C3D/TRC export path in formats.md
- **Next step:** Shipped in PR #13441.

### DL-#13421 · Retrofit Drake Golfer Joint Actuators When URDF Omits Transmissions; Fail Closed on Unwired Torque (#13421).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13421
- **Branch:** merged via #13435
- **PR:** #13435
- **Paths:** see #13435
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`9f93fa1e`; collated from changes/13421-retrofit-drake-golfer-joint-actuators-wh.md)
- **Summary:** Retrofit Drake golfer joint actuators when URDF omits transmissions; fail closed on unwired torque (#13421).
- **Next step:** Shipped in PR #13435.

### DL-#13304 · Pinocchio Grip-Contact Adapter From Shared Topology (HRG-GS-G6)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13304
- **Branch:** merged via #13414
- **PR:** #13414, #13422
- **Paths:** see #13414
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`2605c9ef`; collated from changes/13304-add-pinocchio-grip-adapter-driven-by-con.md)
- **Summary:** Pinocchio grip-contact adapter from shared topology (HRG-GS-G6)
- **Next step:** Shipped in PR #13414.

### DL-#12941 · Repo-Manage Docker-Cache-Prune Host Script With Keep-Label Spare and Contract Tests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12941
- **Branch:** merged via #13424
- **PR:** #13424
- **Paths:** see #13424
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`c49b57a6`; collated from changes/12941-repo-manage-docker-cache-prune-host-scri.md)
- **Summary:** Repo-manage docker-cache-prune host script with keep-label spare and contract tests
- **Next step:** Shipped in PR #13424.

### DL-#12940 · Resolve Tools Gitlink for Docker Explorer on WSL Windows Worktrees

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12940
- **Branch:** merged via #13424
- **PR:** #13424
- **Paths:** see #13424
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`c49b57a6`; collated from changes/12940-resolve-tools-gitlink-for-docker-explore.md)
- **Summary:** Resolve Tools gitlink for Docker explorer on WSL Windows worktrees
- **Next step:** Shipped in PR #13424.

### DL-#12938 · Gate Compose Training Service Behind Profile; Pin PyTorch Cu128 Wheels for Training Stage

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12938
- **Branch:** merged via #13424
- **PR:** #13424
- **Paths:** see #13424
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`c49b57a6`; collated from changes/12938-gate-compose-training-service-behind-pro.md)
- **Summary:** Gate compose training service behind profile; pin PyTorch cu128 wheels for training stage
- **Next step:** Shipped in PR #13424.

### DL-#13416 · Fix Three Drake Continuous-Torque Tests That Failed With Pydrake Installed: Hide Every Cached Pydrake Module and Size the Live Theta via _Resolve_N_Actuators

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13416
- **Branch:** merged via #13418
- **PR:** #13418
- **Paths:** see #13418
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`648fd33e`; collated from changes/13416-fix-three-drake-continuous-torque-tests.md)
- **Summary:** Fix three Drake continuous-torque tests that failed with pydrake installed: hide every cached pydrake module and size the live theta via _resolve_n_actuators
- **Next step:** Shipped in PR #13418.

### DL-#13208 · Add Scripts/Ci/shared_buildkit.py: Idempotent, Flock-Serialised Manager for One Persistent GC-Bounded BuildKit Builder per Host (Create/Upgrade/Repair, Cold-Build Fallback)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13208
- **Branch:** merged via #13277
- **PR:** #13277, #13430
- **Paths:** see #13277
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`dbbc5eb4`; collated from changes/13208-docker-ci-builds-on-the-host-s-shared-ud.md)
- **Summary:** Add scripts/ci/shared_buildkit.py: idempotent, flock-serialised manager for one persistent GC-bounded BuildKit builder per host (create/upgrade/repair, cold-build fallback)
- **Next step:** Shipped in PR #13277.

### DL-#13044 · Predictive Moco Simulation of Deadlift Lift (OSL-7)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13044
- **Branch:** merged via #13406
- **PR:** #13406
- **Paths:** see #13406
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`54a19254`; collated from changes/13044-predictive-moco-simulation-of-deadlift-l.md)
- **Summary:** Predictive Moco simulation of deadlift lift (OSL-7)
- **Next step:** Shipped in PR #13406.

### DL-#13392 · Fix(Pinocchio): CMB-11 Parity Evaluates the Full Neutral Coordinate Set

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13392
- **Branch:** merged via #13397
- **PR:** #13397
- **Paths:** see #13397
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`7da2a46d`; collated from changes/13392-fix-pinocchio-cmb-11-parity-evaluates-th.md)
- **Summary:** fix(pinocchio): CMB-11 parity evaluates the full neutral coordinate set
- **Next step:** Shipped in PR #13397.

### DL-#13303 · Named Grip Presets: Palmer Overlap, Woods Interlock, Ten-Finger With Photo_Estimate Provenance (HRG-GS-G5)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13303
- **Branch:** merged via #13343
- **PR:** #13343
- **Paths:** see #13343
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`fc49cd6e`; collated from changes/13303-named-grip-presets-palmer-overlap-woods.md)
- **Summary:** Named grip presets: Palmer overlap, Woods interlock, ten-finger with photo_estimate provenance (HRG-GS-G5)
- **Next step:** Shipped in PR #13343.

### DL-#13260 · Add Run Study Panel and Worker Across PyQt and React Spine Viewers

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13260
- **Branch:** merged via #13279
- **PR:** #13279
- **Paths:** see #13279
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`38eebd22`; collated from changes/13260-add-run-study-panel-and-worker-across-py.md)
- **Summary:** Add Run Study panel and worker across PyQt and React spine viewers
- **Next step:** Shipped in PR #13279.

### DL-#13302 · Drive Finger Pose From Contact Topology With Per-Hand Roll, Absorbing #13166

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13302
- **Branch:** merged via #13338
- **PR:** #13338
- **Paths:** see #13338
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`366e8614`; collated from changes/13302-drive-finger-pose-from-contact-topology.md)
- **Summary:** Drive finger pose from contact topology with per-hand roll, absorbing #13166
- **Next step:** Shipped in PR #13338.

### DL-#13391 · Drop the Removed Simulator.Set_Publish_Every_Time_Step Call From Drake Simulate and Fit_Swing_Autodiff; Drake Already Defaults to No Per-Step Publish

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13391
- **Branch:** merged via #13395
- **PR:** #13395
- **Paths:** see #13395
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`d279b256`; collated from changes/13391-drop-the-removed-simulator-set-publish-e.md)
- **Summary:** Drop the removed Simulator.set_publish_every_time_step call from Drake simulate and fit_swing_autodiff; Drake already defaults to no per-step publish
- **Next step:** Shipped in PR #13395.

### DL-#13390 · Move the MOSAIC Analytic-vs-Finite-Difference Wall-Clock Speedup Claim to the Benchmark Lane; Unit Lane Asserts It Deterministically via an Injected Clock and Regressor-Evaluation Counts

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13390
- **Branch:** merged via #13394
- **PR:** #13394
- **Paths:** see #13394
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`a491a08e`; collated from changes/13390-move-the-mosaic-analytic-vs-finite-diffe.md)
- **Summary:** Move the MOSAIC analytic-vs-finite-difference wall-clock speedup claim to the benchmark lane; unit lane asserts it deterministically via an injected clock and regressor-evaluation counts
- **Next step:** Shipped in PR #13394.

### DL-#13014 · DomainRandomizer Fails Loud on Unsupported Engines; MuJoCo Parameter Surface; Remove Dead Config Ranges

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13014
- **Branch:** merged via #13345
- **PR:** #13345
- **Paths:** see #13345
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`8115b862`; collated from changes/13014-domainrandomizer-fails-loud-on-unsupport.md)
- **Summary:** DomainRandomizer fails loud on unsupported engines; MuJoCo parameter surface; remove dead config ranges
- **Next step:** Shipped in PR #13345.

### DL-#13012 · Learning/Imitation: Adopt He/Xavier Init, Adam Optimizer, and Shared _Mlp Helper in BehaviorCloning and DAgger

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13012
- **Branch:** merged via #13345
- **PR:** #13345
- **Paths:** see #13345
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`8115b862`; collated from changes/13012-learning-imitation-adopt-he-xavier-init.md)
- **Summary:** learning/imitation: adopt He/Xavier init, Adam optimizer, and shared _mlp helper in BehaviorCloning and DAgger
- **Next step:** Shipped in PR #13345.

### DL-#13008 · Stop C3d_Simscape_Preview False Match Success; Load C3D via Shared Reader With Unit Conversion

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13008
- **Branch:** merged via #13345
- **PR:** #13345
- **Paths:** see #13345
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`8115b862`; collated from changes/13008-stop-c3d-simscape-preview-false-match-su.md)
- **Summary:** Stop c3d_simscape_preview false match success; load C3D via shared reader with unit conversion
- **Next step:** Shipped in PR #13345.

### DL-#13000 · Restore Capture-Contract CI Coverage and Guard Addopts Marker Exclusions (#13000).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13000
- **Branch:** merged via #13345
- **PR:** #13345
- **Paths:** see #13345
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`8115b862`; collated from changes/13000-restore-capture-contract-ci-coverage-and.md)
- **Summary:** Restore capture-contract CI coverage and guard addopts marker exclusions (#13000).
- **Next step:** Shipped in PR #13345.

### DL-#12997 · Fix(Model_Explorer): Per-Engine Export Converts MJCF/Drake/Pinocchio and Viewer Handles Errors Visibly

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12997
- **Branch:** merged via #13345
- **PR:** #13345
- **Paths:** see #13345
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`8115b862`; collated from changes/12997-fix-model-explorer-per-engine-export-con.md)
- **Summary:** fix(model_explorer): per-engine export converts MJCF/Drake/Pinocchio and viewer handles errors visibly
- **Next step:** Shipped in PR #13345.

### DL-#12996 · Add Results Playback UI and Real Engine Detection in Model Explorer (#12996)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12996
- **Branch:** merged via #13345
- **PR:** #13345
- **Paths:** see #13345
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`8115b862`; collated from changes/12996-add-results-playback-ui-and-real-engine.md)
- **Summary:** Add results playback UI and real engine detection in Model Explorer (#12996)
- **Next step:** Shipped in PR #13345.

### DL-#12989 · Fix(Mujoco): Eliminate Phantom World-Fused Contact on Double-Pendulum Fixture (#12989)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12989
- **Branch:** merged via #13345
- **PR:** #13345
- **Paths:** see #13345
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`8115b862`; collated from changes/12989-fix-mujoco-eliminate-phantom-world-fused.md)
- **Summary:** fix(mujoco): eliminate phantom world-fused contact on double-pendulum fixture (#12989)
- **Next step:** Shipped in PR #13345.

### DL-#12988 · Align Engine Capability Flags and Interfaces Across Registered Physics Engines

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12988
- **Branch:** merged via #13345
- **PR:** #13345
- **Paths:** see #13345
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`8115b862`; collated from changes/12988-align-engine-capability-flags-and-interf.md)
- **Summary:** Align engine capability flags and interfaces across registered physics engines
- **Next step:** Shipped in PR #13345.

### DL-#12979 · Fix CentroidalMPC Contact Count and Mass From Model

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12979
- **Branch:** merged via #13345
- **PR:** #13345
- **Paths:** see #13345
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`8115b862`; collated from changes/12979-fix-centroidalmpc-contact-count-and-mass.md)
- **Summary:** Fix CentroidalMPC contact count and mass from model
- **Next step:** Shipped in PR #13345.

### DL-#12974 · Enforce Workspace, Forbidden Zone, Cartesian Velocity, and Contact Force Limits in SafetyMonitor

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12974
- **Branch:** merged via #13345
- **PR:** #13345
- **Paths:** see #13345
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`8115b862`; collated from changes/12974-enforce-workspace-forbidden-zone-cartesi.md)
- **Summary:** Enforce workspace, forbidden zone, Cartesian velocity, and contact force limits in SafetyMonitor
- **Next step:** Shipped in PR #13345.

### DL-#12971 · Enforce OpenCap Learned Augmenter Sidecar Execution, Fail-Closed Checks, and Correct Synthetic/Model-Conditioned Provenance Labeling

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12971
- **Branch:** merged via #13345
- **PR:** #13345
- **Paths:** see #13345
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`8115b862`; collated from changes/12971-enforce-opencap-learned-augmenter-sideca.md)
- **Summary:** Enforce OpenCap learned augmenter sidecar execution, fail-closed checks, and correct synthetic/model-conditioned provenance labeling
- **Next step:** Shipped in PR #13345.

### DL-#12970 · Pipeline: Wire Ground-Support CLI Neural Flags Through MotionMatchingController and Record Neural Disposition on Receipts

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12970
- **Branch:** orch/12970-cursor
- **PR:** #13345
- **Paths:** see #13345
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`8115b862`; collated from changes/12970-pipeline-neural-cli-flags-wired.md)
- **Summary:** pipeline: wire ground-support CLI neural flags through MotionMatchingController and record neural disposition on receipts
- **Next step:** Shipped in PR #13345.

### DL-#12967 · Return Dataset_Id, Name, Rows, Columns From Generate; Add Export Stream Route

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12967
- **Branch:** merged via #13345
- **PR:** #13345
- **Paths:** see #13345
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`8115b862`; collated from changes/12967-return-dataset-id-name-rows-columns-from.md)
- **Summary:** Return dataset_id, name, rows, columns from generate; add export stream route
- **Next step:** Shipped in PR #13345.

### DL-#12939 · Document Ud-Sim:Latest Compose Source of Truth and Enforce Ct-Run Contract in CI

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12939
- **Branch:** merged via #13345
- **PR:** #13345
- **Paths:** see #13345
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`8115b862`; collated from changes/12939-document-ud-sim-latest-compose-source-of.md)
- **Summary:** Document ud-sim:latest compose source of truth and enforce ct-run contract in CI
- **Next step:** Shipped in PR #13345.

### DL-#13301 · Allocated Hand-Hand Contact Reporting for All Style Pairs (HRG-GS-G3)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13301
- **Branch:** merged via #13342
- **PR:** #13342
- **Paths:** see #13342
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`15ef79e2`; collated from changes/13301-allocated-hand-hand-contact-reporting-fo.md)
- **Summary:** Allocated hand-hand contact reporting for all style pairs (HRG-GS-G3)
- **Next step:** Shipped in PR #13342.

### DL-#13377 · Bump Vendor/Ud-Tools to Tools 96A1684 (Coriolis Christoffel Term, Tools #5664; Standard Gravity, Tools #5668)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13377
- **Branch:** merged via #13378
- **PR:** #13378
- **Paths:** see #13378
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`3462c62b`; collated from changes/13377-bump-vendor-ud-tools-to-tools-96a1684-co.md)
- **Summary:** Bump vendor/ud-tools to Tools 96a1684 (Coriolis Christoffel term, Tools #5664; standard gravity, Tools #5668)
- **Next step:** Shipped in PR #13378.

### DL-#13159 · Fix(Lifting): Resolve Unstable Open-Loop Wrist Mode on High-Bar Rack

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13159
- **Branch:** merged via #13344
- **PR:** #13344
- **Paths:** see #13344
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`ff23d322`; collated from changes/13159-fix-lifting-resolve-unstable-open-loop-w.md)
- **Summary:** fix(lifting): resolve unstable open-loop wrist mode on high-bar rack
- **Next step:** Shipped in PR #13344.

### DL-#13158 · Feat(Lifting): High-Bar Rack Reach Sweep and Wrist Mode Resolution

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13158
- **Branch:** merged via #13344
- **PR:** #13344
- **Paths:** see #13344
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`ff23d322`; collated from changes/13158-feat-lifting-high-bar-rack-reach-sweep-a.md)
- **Summary:** feat(lifting): high-bar rack reach sweep and wrist mode resolution
- **Next step:** Shipped in PR #13344.

### DL-#13298 · Unify Grip Frames Across Model_Appearance and Grip_Contact, Fix Drake Contact Normal Recording

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13298
- **Branch:** fix/13298-grip-frame-unify
- **PR:** #13318
- **Paths:** see #13318
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`24633553`; collated from changes/13298-unify-grip-frames-across-model-appearanc.md)
- **Summary:** Unify grip frames across model_appearance and grip_contact, fix Drake contact normal recording
- **Next step:** Shipped in PR #13318.

### DL-#13322 · Add Cross-Model Grip Capability Inventory Manifest, DbC Loader, and Verification Suite

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13322
- **Branch:** merged via #13337
- **PR:** #13337
- **Paths:** see #13337
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`a9170eb8`; collated from changes/13322-add-cross-model-grip-capability-inventor.md)
- **Summary:** Add cross-model grip capability inventory manifest, DbC loader, and verification suite
- **Next step:** Shipped in PR #13337.

### DL-#13189 · Report the Allocated Trail-Palm-on-Lead-Thumb Force (Patch Frame, Allocation, Provider Metadata); Physics Unchanged

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13189
- **Branch:** feat/13189-palm-thumb-contact
- **PR:** #13217
- **Paths:** see #13217
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`15cafedb`; collated from changes/13189-report-the-allocated-trail-palm-on-lead.md)
- **Summary:** Report the allocated trail-palm-on-lead-thumb force (patch frame, allocation, provider metadata); physics unchanged
- **Next step:** Shipped in PR #13217.

### DL-#13049 · Run Real-OpenSim Lift Tests in a Dedicated CI Job (OSL-13)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13049
- **Branch:** merged via #13284
- **PR:** #13284
- **Paths:** see #13284
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`aefee65b`; collated from changes/13049-run-real-opensim-lift-tests-in-a-dedicat.md)
- **Summary:** Run real-OpenSim lift tests in a dedicated CI job (OSL-13)
- **Next step:** Shipped in PR #13284.

### DL-#13299 · Engine-Agnostic Grip-Style Contact Topology: GripStyle and Trigger Finger Join the Physical GripSpec; Per-Style Contact Pair Tables With Anatomy Scaling.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13299
- **Branch:** merged via #13320
- **PR:** #13320
- **Paths:** see #13320
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`10e75c0b`; collated from changes/13299-engine-agnostic-grip-style-contact-topol.md)
- **Summary:** Engine-agnostic grip-style contact topology: GripStyle and trigger finger join the physical GripSpec; per-style contact pair tables with anatomy scaling.
- **Next step:** Shipped in PR #13320.

### DL-#13339 · Test_Check_Windows_Deps No Longer Races Real Engine Imports Against a 3s Wait; the Probe Is Tested Synchronously With Optional Engines Forced Absent

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13339
- **Branch:** merged via #13341
- **PR:** #13341
- **Paths:** see #13341
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`588068ae`; collated from changes/13339-test-check-windows-deps-no-longer-races.md)
- **Summary:** test_check_windows_deps no longer races real engine imports against a 3s wait; the probe is tested synchronously with optional engines forced absent
- **Next step:** Shipped in PR #13341.

### DL-#13258 · Route Web Launcher to /Tools/Spine-Viewer, Update Ch68 Manual and Ignore Licensed_Assets

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13258
- **Branch:** merged via #13269
- **PR:** #13269
- **Paths:** see #13269
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`ac929b64`; collated from changes/13258-route-web-launcher-to-tools-spine-viewer.md)
- **Summary:** Route web launcher to /tools/spine-viewer, update Ch68 manual and ignore licensed_assets
- **Next step:** Shipped in PR #13269.

### DL-#13042 · Compute Muscle Forces and Activations for Barbell Squat (OSL-4)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13042
- **Branch:** merged via #13288
- **PR:** #13288, #13419, #13532
- **Paths:** see #13288
- **Started:** 2026-10-19
- **Last verified:** 2026-10-20 (`7a963f7b`; collated from changes/13042-complete-osl-4-muscle-solve-and-residual.md)
- **Summary:** Compute muscle forces and activations for barbell squat (OSL-4)
- **Next step:** Shipped in PR #13288.

### DL-#13314 · Shard Unit-Test-Gate Into a 2-Leg File-Hash Matrix so Each Leg Stays Well Under the 35 Min Limit Without Raising It

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13314
- **Branch:** ci/13314-shard-unit-test-gate
- **PR:** #13316
- **Paths:** .github/workflows/ci-standard.yml,scripts/ci/pytest_unit_gate_shard.py,tests/ci/test_unit_gate_shard.py
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`3a8b5173`; collated from changes/13314-shard-unit-test-gate-into-a-2-leg-file-h.md)
- **Summary:** Shard unit-test-gate into a 2-leg file-hash matrix so each leg stays well under the 35 min limit without raising it
- **Next step:** Shipped in PR #13316.

### DL-#13300 · Generalised Shared Contact Law: Sphere-Capsule and Capsule-Capsule

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13300
- **Branch:** feat/13300-capsule-contact-law
- **PR:** #13319, #13320
- **Paths:** see #13319
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`10e75c0b`; collated from changes/13300-generalised-shared-contact-law-sphere-ca.md)
- **Summary:** generalised shared contact law: sphere-capsule and capsule-capsule
- **Next step:** Shipped in PR #13319.

### DL-#13261 · Feed Finished Spine Study Receipts Into Injury Analysis via ModelSpinalLoadProvider; Spine Models Stay Out of the Feedback Inventory

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13261
- **Branch:** merged via #13274
- **PR:** #13274
- **Paths:** see #13274
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`512400ae`; collated from changes/13261-feed-spine-study-loads-to-injury-analysis.md)
- **Summary:** Feed finished spine study receipts into injury analysis via ModelSpinalLoadProvider; spine models stay out of the feedback inventory
- **Next step:** Shipped in PR #13274.

### DL-#13259 · Add Spine Study Execution and Model Asset Listing API

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13259
- **Branch:** merged via #13275
- **PR:** #13275
- **Paths:** see #13275
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`a03af52f`; collated from changes/13259-add-spine-study-execution-and-model-asse.md)
- **Summary:** Add spine study execution and model asset listing API
- **Next step:** Shipped in PR #13275.

### DL-#13253 · Lazy Plot Style Qt Widget Re-Exports

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13253
- **Branch:** fix/plot-style-lazy-widgets
- **PR:** #13257
- **Paths:** see #13257
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`9b492271`; collated from changes/13253-plot-style-re-exports-qt-widgets-and-the.md)
- **Summary:** plot_style re-exports Qt widgets and the matplotlib renderer lazily so headless force_overlay/loop_nullspace imports no longer load PyQt6 or mplot3d
- **Next step:** Shipped in PR #13257.

### DL-#13205 · HRG-GR4: Grip Presets and Sliders in the Spec Builder, Native Viewer Export, Desktop and Web, With One Shared Resolver

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13205
- **Branch:** feat/13205-grip-controls
- **PR:** #13198
- **Paths:** see #13198
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`5272fa8b`; collated from changes/13205-hrg-gr4-grip-presets-and-sliders-in-the.md)
- **Summary:** HRG-GR4: grip presets and sliders in the spec builder, native viewer export, desktop and web, with one shared resolver
- **Next step:** Shipped in PR #13198.

### DL-#13204 · Grip Fitting Modes (User_Fixed, Bounded_Prior, Free_Fit), Rom_Set_Identified Provenance, Golf-Language Grip Report With Profile Interval and Multistart Sensitivity (HRG-GR3)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13204
- **Branch:** feat/13204-grip-fit-modes
- **PR:** #13198
- **Paths:** see #13198
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`5272fa8b`; collated from changes/13204-grip-fitting-modes-user-fixed-bounded-pr.md)
- **Summary:** Grip fitting modes (user_fixed, bounded_prior, free_fit), rom_set_identified provenance, golf-language grip report with profile interval and multistart sensitivity (HRG-GR3)
- **Next step:** Shipped in PR #13198.

### DL-#13203 · HRG-GR2: Physical Grip Configuration as an Immutable Spec Revision With Hand Depth/Spacing on the Closure and Grip Provenance; Fitted_Grip Uses Provenance

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13203
- **Branch:** feat/13203-physical-grip-revision
- **PR:** #13198
- **Paths:** see #13198
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`5272fa8b`; collated from changes/13203-hrg-gr2-physical-grip-configuration-as-a.md)
- **Summary:** HRG-GR2: physical grip configuration as an immutable spec revision with hand depth/spacing on the closure and grip provenance; fitted_grip uses provenance
- **Next step:** Shipped in PR #13198.

### DL-#13202 · Add the Anatomical Grip Contract (GripSpec, Exact Euler Bijection) and Fix the Trail Grip-Roll Axis

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13202
- **Branch:** feat/13202-anatomical-grip-contract
- **PR:** #13198
- **Paths:** see #13198
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`5272fa8b`; collated from changes/13202-add-the-anatomical-grip-contract-gripspe.md)
- **Summary:** Add the anatomical grip contract (GripSpec, exact Euler bijection) and fix the trail grip-roll axis
- **Next step:** Shipped in PR #13198.

### DL-#13201 · Subject Handedness in Spec, Receipts and GripPose (HRG-GR0)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13201
- **Branch:** feat/13201-subject-handedness
- **PR:** #13198
- **Paths:** see #13198
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`5272fa8b`; collated from changes/13201-subject-handedness-in-spec-receipts-and.md)
- **Summary:** Subject handedness in spec, receipts and GripPose (HRG-GR0)
- **Next step:** Shipped in PR #13198.

### DL-#13186 · HRG-G1: Skin-Coloured Bare Grip Hands by Default and an Optional White Glove (Glove None/Lead/Trail/Both) Through CLI, Desktop and Web; One Hand-Material Resolver; No Duplicate RHandStandoff Loft; Shared Core.Handedness

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13186
- **Branch:** feat/13186-bare-hands-lead-glove
- **PR:** #13198
- **Paths:** see #13198
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`5272fa8b`; collated from changes/13186-hrg-g1-skin-coloured-bare-grip-hands-by.md)
- **Summary:** HRG-G1: skin-coloured bare grip hands by default and an optional white glove (glove none/lead/trail/both) through CLI, desktop and web; one hand-material resolver; no duplicate RHandStandoff loft; shared core.handedness
- **Next step:** Shipped in PR #13198.

### DL-#13173 · Full-Body-V2 Landmark Lumbar (L5/S1) and Thoracic (T12/L1) Joints: C3 L5/S1, Gray T12/L1 Fraction, Yukawa Curvature Check, Mass Ledger, Soft-Limit ROM, Back-Marker Observability (Rank 5, Split Regularised); Fixes OpenSim CustomJoint Rotation Order

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13173
- **Branch:** feat/13173-regional-trunk-joints
- **PR:** #13247
- **Paths:** see #13247
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`b2aec7d0`; collated from changes/13173-full-body-v2-landmark-lumbar-l5-s1-and-t.md)
- **Summary:** full-body-v2 landmark lumbar (L5/S1) and thoracic (T12/L1) joints: C3 L5/S1, Gray T12/L1 fraction, Yukawa curvature check, mass ledger, soft-limit ROM, back-marker observability (rank 5, split regularised); fixes OpenSim CustomJoint rotation order
- **Next step:** Shipped in PR #13247.

### DL-#13169 · Engine-Neutral Anatomical Angles Package With Wrist and Forearm Terms (HRG-E2)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13169
- **Branch:** feat/13169-anatomical-angles
- **PR:** #13198, #13487
- **Paths:** see #13198
- **Started:** 2026-10-19
- **Last verified:** 2026-10-20 (`471e868e`; collated from changes/13169-hrg-e2-full-body-v1-wrist-forearm-adapte.md)
- **Summary:** Engine-neutral anatomical angles package with wrist and forearm terms (HRG-E2)
- **Next step:** Shipped in PR #13198.

### DL-#13167 · HRG-B2: Appearance CLI Flags, Precedence Resolver, Shared Qt Colour Field, Desktop and Web Hair/Skin/Clothing Colour Pickers

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13167
- **Branch:** feat/13167-appearance-colour-controls
- **PR:** #13198
- **Paths:** see #13198
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`5272fa8b`; collated from changes/13167-hrg-b2-appearance-cli-flags-precedence-r.md)
- **Summary:** HRG-B2: appearance CLI flags, precedence resolver, shared Qt colour field, desktop and web hair/skin/clothing colour pickers
- **Next step:** Shipped in PR #13198.

### DL-#13166 · HRG-A2: Parameterise Visual Grip Strength and Per-Hand Roll Offsets in GripPose and Grip Selection

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13166
- **Branch:** feat/13166-visual-grip-strength
- **PR:** #13311
- **Paths:** see #13311
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`9dfb65ab`; collated from changes/13166-hrg-a2-parameterise-visual-grip-strength.md)
- **Summary:** HRG-A2: parameterise visual grip strength and per-hand roll offsets in GripPose and grip selection
- **Next step:** Shipped in PR #13311.

### DL-#13019 · Align Native Drake Torque Replay With Reviewed 1.57/1.58 Versions and Skip Unreviewed Stacks in Tests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13019
- **Branch:** merged via #13286
- **PR:** #13286
- **Paths:** see #13286
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`e66abba7`; collated from changes/13019-align-native-drake-torque-replay-with-re.md)
- **Summary:** Align native Drake torque replay with reviewed 1.57/1.58 versions and skip unreviewed stacks in tests
- **Next step:** Shipped in PR #13286.

### DL-#13013 · Enforce Imitation Learning Contracts: Seed-Based Determinism, Validation Split Bounds, Activation Execution, and DAgger Dataset Immutability

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13013
- **Branch:** merged via #13254
- **PR:** #13254
- **Paths:** see #13254
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`fa374264`; collated from changes/13013-imitation-contracts-and-determinism.md)
- **Summary:** Enforce imitation learning contracts: seed-based determinism, validation split bounds, activation execution, and DAgger dataset immutability
- **Next step:** Shipped in PR #13254.

### DL-#12986 · Abort RealTimeController on Unexpected Callback Errors With Zero Torque; Wire SafetyMonitor on Non-Simulation Backends.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12986
- **Branch:** merged via #13272
- **PR:** #13272
- **Paths:** see #13272
- **Started:** 2026-10-19
- **Last verified:** 2026-10-19 (`e40c9962`; collated from changes/12986-abort-realtimecontroller-on-unexpected-c.md)
- **Summary:** Abort RealTimeController on unexpected callback errors with zero torque; wire SafetyMonitor on non-simulation backends.
- **Next step:** Shipped in PR #13272.

### DL-#13157 · OSL-18: OpenSim Lift Foot Carries the Bundle's 10 Deg Tibial Torsion and Is Sized From the Bundle's Calcaneus Stations; Squat IK Marker RMS 1.93 cm / Max 3.55 cm, Inside the Unchanged 2/4 cm Gate; Marker-Gate Xfails Removed

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13157
- **Branch:** merged via #13264
- **PR:** #13264
- **Paths:** see #13264
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`9834a7c7`; collated from changes/13157-osl-18-opensim-lift-foot-carries-the-bun.md)
- **Summary:** OSL-18: OpenSim lift foot carries the bundle's 10 deg tibial torsion and is sized from the bundle's calcaneus stations; squat IK marker RMS 1.93 cm / max 3.55 cm, inside the unchanged 2/4 cm gate; marker-gate xfails removed
- **Next step:** Shipped in PR #13264.

### DL-#12998 · Add JaxSim Launcher Dashboard Tile and Engine Tile Coverage Test

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12998
- **Branch:** merged via #13224
- **PR:** #13224, #13603
- **Paths:** see #13224
- **Started:** 2026-10-18
- **Last verified:** 2026-10-21 (`fef8efa8`; collated from changes/12998-register-jaxsim-dashboard-on-engines-das.md)
- **Summary:** Add JaxSim launcher dashboard tile and engine tile coverage test
- **Next step:** Shipped in PR #13224.

### DL-#12977 · Pipeline: FreeMoCap Adapter Deterministic CSV Selection, NaN Confidence Handling, Non-Monotonic Time Rejection, and Configurable Min_Confidence

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12977
- **Branch:** fix/12977-freemocap-adapter-integrity
- **PR:** #13252
- **Paths:** see #13252
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`9843b55e`; collated from changes/12977-freemocap-adapter-integrity.md)
- **Summary:** pipeline: FreeMoCap adapter deterministic CSV selection, NaN confidence handling, non-monotonic time rejection, and configurable min_confidence
- **Next step:** Shipped in PR #13252.

### DL-#12968 · Robotics: Eliminate Silent Velocity, Acceleration, and Mass Fallbacks in ZMPComputer and Support Finite-Difference Acceleration

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12968
- **Branch:** fix/12968-zmp-computer-velocity-fallback
- **PR:** #13250
- **Paths:** see #13250
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`77c7b1d9`; collated from changes/12968-zmp-computer-velocity-fallback.md)
- **Summary:** robotics: eliminate silent velocity, acceleration, and mass fallbacks in ZMPComputer and support finite-difference acceleration
- **Next step:** Shipped in PR #13250.

### DL-#13179 · Close the Segment Axial-Load Evaluation Over the Grip Weld: Physical KKT Closure Wrench Replaces the Soft Weld in the Prescribed Inverse, Ball-Impact Frames Are Unavailable, Ground Free Moment Is Carried

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13179
- **Branch:** fix/13179-closure-wrench-overlay
- **PR:** #13200
- **Paths:** see #13200
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`33b2d5eb`; collated from changes/13179-close-the-segment-axial-load-evaluation.md)
- **Summary:** Close the segment axial-load evaluation over the grip weld: physical KKT closure wrench replaces the soft weld in the prescribed inverse, ball-impact frames are unavailable, ground free moment is carried
- **Next step:** Shipped in PR #13200.

### DL-#13170 · Full-Body-V2 Schema (Regional Lumbar L5/S1 and Thoracic T12/L1 Trunk), One Schema Registry for Exporters and Consumers, and a Constrained V1->V2 Trunk Migration With Residual and Non-Uniqueness Report; V1 Stays Default and Byte-Identical

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13170
- **Branch:** feat/13170-full-body-v2-schema
- **PR:** #13244, #13224
- **Paths:** see #13244
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`3dec2e7c`; collated from changes/13170-full-body-v2-schema-regional-lumbar-l5-s.md)
- **Summary:** full-body-v2 schema (regional lumbar L5/S1 and thoracic T12/L1 trunk), one schema registry for exporters and consumers, and a constrained v1->v2 trunk migration with residual and non-uniqueness report; v1 stays default and byte-identical
- **Next step:** Shipped in PR #13244.

### DL-#13146 · Translucent Tension/Compression Segment Shading With Independent Force/Torque Vector Toggles, User Colours and Opacity Across MuJoCo GUI, Drake/Pinocchio GUIs, Web and Native_Viewer_Export (MuJoCo, Drake, Pinocchio, MyoSuite)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13146
- **Branch:** feat/tension-compression-overlay
- **PR:** #13200
- **Paths:** see #13200
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`33b2d5eb`; collated from changes/13146-translucent-tension-compression-segment.md)
- **Summary:** Translucent tension/compression segment shading with independent force/torque vector toggles, user colours and opacity across MuJoCo GUI, Drake/Pinocchio GUIs, web and native_viewer_export (MuJoCo, Drake, Pinocchio, MyoSuite)
- **Next step:** Shipped in PR #13200.

### DL-#13004 · Fail Closed With Ok=False and Rejected Cameras on Degenerate or Collinear Camera Pairs

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13004
- **Branch:** merged via #13239
- **PR:** #13239, #13224
- **Paths:** see #13239
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`3dec2e7c`; collated from changes/13004-triangulate-degenerate-cameras.md)
- **Summary:** Fail closed with ok=False and rejected cameras on degenerate or collinear camera pairs
- **Next step:** Shipped in PR #13239.

### DL-#13003 · Penalize Behind-Camera Observations in Bundle Adjustment and Propagate Optimizer Convergence

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13003
- **Branch:** merged via #13242
- **PR:** #13242
- **Paths:** see #13242
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`0a104862`; collated from changes/13003-bundle-behind-camera-convergence.md)
- **Summary:** Penalize behind-camera observations in bundle adjustment and propagate optimizer convergence
- **Next step:** Shipped in PR #13242.

### DL-#12975 · Verify All Participating Swing Receipt Hashes in Leakage Guard, Preserve All Participant Hashes, and Abstain on Single Candidates

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12975
- **Branch:** merged via #13249
- **PR:** #13249
- **Paths:** see #13249
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`6f3a0ad9`; collated from changes/12975-swing-pairing-receipt-hash-guard.md)
- **Summary:** Verify all participating swing receipt hashes in leakage guard, preserve all participant hashes, and abstain on single candidates
- **Next step:** Shipped in PR #13249.

### DL-#13188 · Procedural Shoe-Last Feet (HRG-F1): Heel Counter, Medial Arch, Instep Ramp, Asymmetric Toe Box, Flat Sole Sized From Spec Foot Length and a Cited Breadth Ratio; Foot.Model Option

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13188
- **Branch:** feat/13188-procedural-feet
- **PR:** #13196
- **Paths:** see #13196
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`b0aba425`; collated from changes/13188-procedural-shoe-last-feet-hrg-f1-heel-co.md)
- **Summary:** Procedural shoe-last feet (HRG-F1): heel counter, medial arch, instep ramp, asymmetric toe box, flat sole sized from spec foot length and a cited breadth ratio; foot.model option
- **Next step:** Shipped in PR #13196.

### DL-#13002 · Align Multi-View Observations by Physical Time Across Different Frame Rates and Report Dropped Frames

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13002
- **Branch:** merged via #13243
- **PR:** #13243
- **Paths:** see #13243
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`412e4fbf`; collated from changes/13002-observations-from-views-fps.md)
- **Summary:** Align multi-view observations by physical time across different frame rates and report dropped frames
- **Next step:** Shipped in PR #13243.

### DL-#13163 · HRG-B1: Default Golfer Is Light Skin With Mid Brown Hair From One Serialized Appearance-Defaults Contract (Python + Generated TS), Skin_1..8 and Hair Palettes

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13163
- **Branch:** feat/13163-appearance-default-contract
- **PR:** #13182
- **Paths:** see #13182
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`9ef639a9`; collated from changes/13163-hrg-b1-default-golfer-is-light-skin-with.md)
- **Summary:** HRG-B1: default golfer is light skin with mid brown hair from one serialized appearance-defaults contract (Python + generated TS), skin_1..8 and hair palettes
- **Next step:** Shipped in PR #13182.

### DL-#13009 · Export NaN Markers With Negative Residual, Homogeneous Coordinates, and Axis Convention in Canonical_C3d_Exporter

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13009
- **Branch:** merged via #13232
- **PR:** #13232
- **Paths:** see #13232
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`c245e9c9`; collated from changes/13009-c3d-exporter-nan-residuals-and-axis.md)
- **Summary:** Export NaN markers with negative residual, homogeneous coordinates, and axis convention in canonical_c3d_exporter
- **Next step:** Shipped in PR #13232.

### DL-#13006 · Map Reconstruct and Legacy Joint Names in L3 Paired 3D Comparisons, Fail Closed on Missing or Swapped Bilateral Joints, Compute Kinematic Sequence Order and Timing Error From Angular Velocity Peaks With Explicit Unmeasured None, and Isolate Depth Axis According to Camera Pose and View.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13006
- **Branch:** merged via #13237
- **PR:** #13237
- **Paths:** see #13237
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`da3a7975`; collated from changes/13006-l3-comparison-laterality-angles.md)
- **Summary:** Map reconstruct and legacy joint names in L3 paired 3D comparisons, fail closed on missing or swapped bilateral joints, compute kinematic sequence order and timing error from angular velocity peaks with explicit unmeasured None, and isolate depth axis according to camera pose and view.
- **Next step:** Shipped in PR #13237.

### DL-#12972 · Reject Non-Finite Inputs and Validate Shapes in SafetyMonitor

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12972
- **Branch:** merged via #13229
- **PR:** #13229
- **Paths:** see #13229
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`74de6533`; collated from changes/12972-reject-non-finite-inputs-and-validate-sh.md)
- **Summary:** Reject non-finite inputs and validate shapes in SafetyMonitor
- **Next step:** Shipped in PR #13229.

### DL-#13187 · HRG-C4: Pelvis Skin Width From Measured Hip Breadth or Stature Prior With Femoral-Head Containment

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13187
- **Branch:** fix/13187-pelvis-width-prior
- **PR:** #13193
- **Paths:** see #13193
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`bb7ad6f3`; collated from changes/13187-hrg-c4-pelvis-skin-width-from-measured-h.md)
- **Summary:** HRG-C4: pelvis skin width from measured hip breadth or stature prior with femoral-head containment
- **Next step:** Shipped in PR #13193.

### DL-#13007 · Handle Numeric Array Wrench Point Attributes Without Boolean Evaluation Error and Log Warnings for Unparseable Points in Load_Trace_Forces

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13007
- **Branch:** merged via #13234
- **PR:** #13234
- **Paths:** see #13234
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`03bd87ab`; collated from changes/13007-force-alignment-numeric-attr.md)
- **Summary:** Handle numeric array wrench point attributes without boolean evaluation error and log warnings for unparseable points in load_trace_forces
- **Next step:** Shipped in PR #13234.

### DL-#13005 · Enforce Unit, Metric, and Level Preconditions in Error-Budget Guidance Classification

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13005
- **Branch:** merged via #13236
- **PR:** #13236
- **Paths:** see #13236
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`4198982e`; collated from changes/13005-error-budget-guidance-unit-level-matching.md)
- **Summary:** Enforce unit, metric, and level preconditions in error-budget guidance classification
- **Next step:** Shipped in PR #13236.

### DL-#12730 · NSX-101: Variability Interpretation Guide, Ten Publication-Gate Rows (All Blocked or Deferred), Five Preregistered External Experiments in DV-12554, and a Variability Sensitivity Section in the Design-Manual QMD Chapter; No Coaching Prescription Validated

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12730
- **Branch:** merged via #13233
- **PR:** #13233
- **Paths:** see #13233
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`913b5eac`; collated from changes/12730-nsx101-variability-docs-gates.md)
- **Summary:** NSX-101: variability interpretation guide, ten publication-gate rows (all blocked or deferred), five preregistered external experiments in DV-12554, and a Variability Sensitivity section in the design-manual QMD chapter; no coaching prescription validated
- **Next step:** Shipped in PR #13233.

### DL-#12973 · OpenCap Augmenter Non-Zero Base Joint Derivation, NaN Marker Propagation, and Gap Counting

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12973
- **Branch:** merged via #13230
- **PR:** #13230
- **Paths:** see #13230
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`3f7b262f`; collated from changes/12973-opencap-augmenter-impute-nan-handling.md)
- **Summary:** OpenCap augmenter non-zero base joint derivation, NaN marker propagation, and gap counting
- **Next step:** Shipped in PR #13230.

### DL-#13131 · OSL-17: Back-Squat Bar on the Cited High-Bar Rack (One Shaft Radius Behind the C7 Landmark, Glassbrook 2019 / Wretenberg 1996) and a 0.1 s Lead-In so the Bundle and Barbell Re-Track Start Settled; OpenSim Squat Hicks Gate Passes on All 41 States

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13131
- **Branch:** merged via #13047
- **PR:** #13047
- **Paths:** see #13047
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`7fed54d1`; collated from changes/13131-osl-17-back-squat-bar-on-the-cited-high.md)
- **Summary:** OSL-17: back-squat bar on the cited high-bar rack (one shaft radius behind the C7 landmark, Glassbrook 2019 / Wretenberg 1996) and a 0.1 s lead-in so the bundle and barbell re-track start settled; OpenSim squat Hicks gate passes on all 41 states
- **Next step:** Shipped in PR #13047.

### DL-#13093 · OSL-15: Opt-In Bar-Load Balance of the Back-Squat IK Over Mid-Foot (0.190 m to 6.4E-5 m); Loaded Kkt Tracking Loses the Feet, so the Committed Source and Evidence Stay Unchanged

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13093
- **Branch:** merged via #13047
- **PR:** #13047
- **Paths:** see #13047
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`7fed54d1`; collated from changes/13093-osl-15-opt-in-bar-load-balance-of-the-ba.md)
- **Summary:** OSL-15: opt-in bar-load balance of the back-squat IK over mid-foot (0.190 m to 6.4e-5 m); loaded kkt tracking loses the feet, so the committed source and evidence stay unchanged
- **Next step:** Shipped in PR #13047.

### DL-#13072 · LIFT-4 Barbell Re-Track: Track the Loaded Plant With a Rigid-Bar Controller Model so the Bar's Floating-Base Load Is on Every Row (Drift Removed); Record Per-Landmark Re-Track Deviation; Re-Tighten the Pelvis Gate to 0; Split Build_Lift_Model; Source-Squat Balance (Bar Ahead of the Forefoot) Still Blocks the LIFT-5 Bounds

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13072
- **Branch:** merged via #13047
- **PR:** #13047
- **Paths:** see #13047
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`7fed54d1`; collated from changes/13072-lift-4-barbell-re-track-track-the-loaded.md)
- **Summary:** LIFT-4 barbell re-track: track the loaded plant with a rigid-bar controller model so the bar's floating-base load is on every row (drift removed); record per-landmark re-track deviation; re-tighten the pelvis gate to 0; split build_lift_model; source-squat balance (bar ahead of the forefoot) still blocks the LIFT-5 bounds
- **Next step:** Shipped in PR #13047.

### DL-#13057 · OSL-14: Lift Bundles Rebuilt on the AN-1 Lift Body (Lumbar Joint, Cited-Range TRF Bounds, de Leva Hips, Wrist Grip Fix); OpenSim Torso Width Scaling and Pelvis Tracking; LIFT-4 Barbell Re-Track Divergence Measured as Pre-Existing

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13057
- **Branch:** merged via #13047
- **PR:** #13047
- **Paths:** see #13047
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`7fed54d1`; collated from changes/13057-osl-14-lift-bundles-rebuilt-on-the-an-1.md)
- **Summary:** OSL-14: lift bundles rebuilt on the AN-1 lift body (lumbar joint, cited-range TRF bounds, de Leva hips, wrist grip fix); OpenSim torso width scaling and pelvis tracking; LIFT-4 barbell re-track divergence measured as pre-existing
- **Next step:** Shipped in PR #13047.

### DL-#13048 · OSL-12: OpenSim Lift IK/ID Credibility: Pelvis Width Scaling and Verified Virtual ASIS/PSIS Landmarks, 1E7 Pa Contact (PR #476) Within 8 mm, Ankle-Lean Balanced Holds at Zero Pelvis Residual, In-Range Squat IK

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13048
- **Branch:** merged via #13047
- **PR:** #13047, #13456
- **Paths:** see #13047
- **Started:** 2026-10-18
- **Last verified:** 2026-10-20 (`dd1b1650`; collated from changes/13048-keep-the-squat-ik-inside-every-coordinat.md)
- **Summary:** OSL-12: OpenSim lift IK/ID credibility: pelvis width scaling and verified virtual ASIS/PSIS landmarks, 1e7 Pa contact (PR #476) within 8 mm, ankle-lean balanced holds at zero pelvis residual, in-range squat IK
- **Next step:** Shipped in PR #13047.

### DL-#13041 · Lift Body and Every Derived Lift Input (Squat on the OSL-17 Rack, Deadlift, Clean, Snatch, Bench) Regenerated on the Whole-Row de Leva Table (#13037) With Main's Tibial Torsion; Regression-Snapshot IK Bounds Re-Derived; Bundle Builder Picks RK4-Stable Substeps (Deadlift 8 -> 16, Parity Xfail Removed); Follow-Ups OSL-18 #13157 (Foot Torsion Marker Gate, Strict Xfail), OSL-19 #13158 (Arm Reach), OSL-20 #13159 (Open-Loop Wrist Mode)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13041
- **Branch:** merged via #13047
- **PR:** #13047, #13404, #13431, #13536
- **Paths:** see #13047
- **Started:** 2026-10-18
- **Last verified:** 2026-10-20 (`702d19d6`; collated from changes/13041-count-bench-support-reaction-in-hicks-re.md)
- **Summary:** Lift body and every derived lift input (squat on the OSL-17 rack, deadlift, clean, snatch, bench) regenerated on the whole-row de Leva table (#13037) with main's tibial torsion; regression-snapshot IK bounds re-derived; bundle builder picks RK4-stable substeps (deadlift 8 -> 16, parity xfail removed); follow-ups OSL-18 #13157 (foot torsion marker gate, strict xfail), OSL-19 #13158 (arm reach), OSL-20 #13159 (open-loop wrist mode)
- **Next step:** Shipped in PR #13047.

### DL-#12729 · NSX-100: Variability & Consistency Tab in the PyQt6 Grip Wrench Window and a Matching React Section, Driven by the Plot Model 2.0.0 Sensitivity Panels, With Catalogue-Backed Request Controls and Parity Entry Loop_Nullspace_Variability

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12729
- **Branch:** merged via #13227
- **PR:** #13227
- **Paths:** see #13227
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`62be7661`; collated from changes/12729-nsx-100-variability-consistency-tab-in-t.md)
- **Summary:** NSX-100: Variability & Consistency tab in the PyQt6 Grip Wrench window and a matching React section, driven by the plot model 2.0.0 sensitivity panels, with catalogue-backed request controls and parity entry loop_nullspace_variability
- **Next step:** Shipped in PR #13227.

### DL-#12978 · Reject Duplicate Source Labels and Handle Inf Values in Capture Export

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12978
- **Branch:** merged via #13225
- **PR:** #13225
- **Paths:** see #13225
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`6e9291cc`; collated from changes/12978-reject-duplicate-source-labels-and-handl.md)
- **Summary:** Reject duplicate source labels and handle inf values in capture export
- **Next step:** Shipped in PR #13225.

### DL-#13023 · Surface Searched Paths and Submodule Remedy in Tile Launch Failure Toast

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13023
- **Branch:** merged via #13219
- **PR:** #13219
- **Paths:** see #13219
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`b997dddd`; collated from changes/13023-surface-searched-paths-and-submodule-rem.md)
- **Summary:** Surface searched paths and submodule remedy in tile launch failure toast
- **Next step:** Shipped in PR #13219.

### DL-#12984 · Resolve MATLAB Default Slx Path Under Production Src Root and Raise GolfModelingError on Missing Default Model

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12984
- **Branch:** merged via #13222
- **PR:** #13222
- **Paths:** see #13222
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`9cf31a81`; collated from changes/12984-resolve-matlab-default-slx-path-under-pr.md)
- **Summary:** Resolve MATLAB default slx path under production src root and raise GolfModelingError on missing default model
- **Next step:** Shipped in PR #13222.

### DL-#12576 · NSX-44: React Loop Null-Space Panel Rendering the Shared Plot-Model Golden

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12576
- **Branch:** merged via #13215
- **PR:** #13215
- **Paths:** see #13215
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`b13bbb0e`; collated from changes/12576-nsx44-react-panel.md)
- **Summary:** NSX-44: React loop null-space panel rendering the shared plot-model golden
- **Next step:** Shipped in PR #13215.

### DL-#13161 · HRG-00: Measured Coordinate Sign Map, ISB Basis Map, Human-Range Characterisation and HRG Calculation Reference

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13161
- **Branch:** chore/13161-hrg-baseline-sign-map
- **PR:** #13184
- **Paths:** see #13184
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`e487e2c5`; collated from changes/13161-hrg-00-measured-coordinate-sign-map-isb.md)
- **Summary:** HRG-00: measured coordinate sign map, ISB basis map, human-range characterisation and HRG calculation reference
- **Next step:** Shipped in PR #13184.

### DL-#12728 · NSX-99: Sensitivity Stage, Report Schema V2 and Variability Panels

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12728
- **Branch:** merged via #13195
- **PR:** #13195
- **Paths:** see #13195
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`878ab14f`; collated from changes/12728-nsx99-variability-service.md)
- **Summary:** NSX-99: sensitivity stage, report schema v2 and variability panels
- **Next step:** Shipped in PR #13195.

### DL-#13165 · HRG-E1: Wrist Lever Angle Renamed With NaN-Plus-Reason on Missing Markers; Joint_Stress Requires Signed, Side-Tagged WristDeviation

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13165
- **Branch:** fix/13165-legacy-angle-validity
- **PR:** #13180
- **Paths:** see #13180
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`610d9a0b`; collated from changes/13165-hrg-e1-wrist-lever-angle-renamed-with-na.md)
- **Summary:** HRG-E1: wrist lever angle renamed with NaN-plus-reason on missing markers; joint_stress requires signed, side-tagged WristDeviation
- **Next step:** Shipped in PR #13180.

### DL-#12575 · NSX-43: Tabbed Grip Wrench Window With NSX Plot-Model Tabs, Preset Selector and QThreadPool Runner

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12575
- **Branch:** merged via #13197
- **PR:** #13197
- **Paths:** see #13197
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`c3ee8732`; collated from changes/12575-nsx-43-pyqt-tabbed-nsx-window.md)
- **Summary:** NSX-43: tabbed Grip Wrench window with NSX plot-model tabs, preset selector and QThreadPool runner
- **Next step:** Shipped in PR #13197.

### DL-#13172 · HRG-C3: Pelvis Frame Re-Expression (Invariant) and Documented HJC/COM/L5S1 Relocation as a New Spec Revision

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13172
- **Branch:** feat/13172-pelvis-reexpression
- **PR:** #13194
- **Paths:** see #13194
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`d2777a49`; collated from changes/13172-hrg-c3-pelvis-frame-re-expression-invari.md)
- **Summary:** HRG-C3: pelvis frame re-expression (invariant) and documented HJC/COM/L5S1 relocation as a new spec revision
- **Next step:** Shipped in PR #13194.

### DL-#13168 · HRG-C2: MAP Hip Joint Centre Estimator With Covariance, Status and Stature Prior (Estimation Only)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13168
- **Branch:** feat/13168-hjc-map-estimation
- **PR:** #13192
- **Paths:** see #13192
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`8eee78d6`; collated from changes/13168-hrg-c2-map-hip-joint-centre-estimator-wi.md)
- **Summary:** HRG-C2: MAP hip joint centre estimator with covariance, status and stature prior (estimation only)
- **Next step:** Shipped in PR #13192.

### DL-#13162 · Grip Hand Frame Rolled -90 Deg About the Shaft so Azimuth 0 Is the Top of the Grip (HRG-A1)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13162
- **Branch:** fix/13162-grip-hand-frame
- **PR:** #13206
- **Paths:** see #13206
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`fa401cf2`; collated from changes/13162-grip-hand-frame-rolled-90-deg-about-the.md)
- **Summary:** Grip hand frame rolled -90 deg about the shaft so azimuth 0 is the top of the grip (HRG-A1)
- **Next step:** Shipped in PR #13206.

### DL-#13164 · HRG-C1: Loft the Pelvis Visual on the Hip Landmarks (Centred on the HJC Midpoint, Ends at the Lumbar Plane) and Shrink the Hip Blend Ball; Visual Only

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13164
- **Branch:** fix/13164-pelvis-visual-loft
- **PR:** #13183
- **Paths:** see #13183
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`075d5542`; collated from changes/13164-hrg-c1-loft-the-pelvis-visual-on-the-hip.md)
- **Summary:** HRG-C1: loft the pelvis visual on the hip landmarks (centred on the HJC midpoint, ends at the lumbar plane) and shrink the hip blend ball; visual only
- **Next step:** Shipped in PR #13183.

### DL-#13156 · Plate Import Forms (PyQt and Web) and the Plate-Overlay Route Declare Channel Timing Skew and the CoP Skew Relation (DeclaredReading); Only Independent Allows the CoP Covariance

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13156
- **Branch:** merged via #13156
- **PR:** #13156
- **Paths:** see #13156
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`ef519eca`; collated from changes/13156-plate-import-forms-pyqt-and-web-and-the.md)
- **Summary:** Plate import forms (PyQt and web) and the plate-overlay route declare channel timing skew and the CoP skew relation (DeclaredReading); only independent allows the CoP covariance
- **Next step:** Shipped in PR #13156.

### DL-#13152 · Ground-Reaction Overlay Draws Measured CoP 1-Sigma Covariance Ellipses, Tilted With the Plate Yaw (Cop_Cov_Random_Xy_M2 = R_Xy Diag R_Xy^T)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13152
- **Branch:** merged via #13152
- **PR:** #13152
- **Paths:** see #13152
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`8582cf13`; collated from changes/13152-ground-reaction-overlay-draws-measured-c.md)
- **Summary:** Ground-reaction overlay draws measured CoP 1-sigma covariance ellipses, tilted with the plate yaw (cop_cov_random_xy_m2 = R_xy diag R_xy^T)
- **Next step:** Shipped in PR #13152.

### DL-#12965 · Infra: Handle_Api_Errors Leaks Absolute Server Paths and Internals via Str(E) in 400/403/404 Bodies

- **State:** shipped
- **Owner:** antigravity
- **Issue:** #12965
- **Branch:** fix/12965-sanitize-api-errors
- **PR:** #13155
- **Paths:** src/api/middleware/error_handler.py,src/api/utils/error_codes.py,tests/unit/api/test_error_handler.py
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`6c4b5e9f`; collated from changes/12965-sanitize-400-403-404-error-details-in-ha.md)
- **Summary:** Sanitize 400/403/404 error details in handle_api_errors to prevent path leaks
- **Next step:** Shipped in PR #13155.

### DL-#12574 · NSX-42: Framework-Free NSX Plot Model and Cross-UI Golden

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12574
- **Branch:** merged via #13154
- **PR:** #13154
- **Paths:** see #13154
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`bfc324fd`; collated from changes/12574-nsx-42-framework-free-nsx-plot-model-and.md)
- **Summary:** NSX-42: framework-free NSX plot model and cross-UI golden
- **Next step:** Shipped in PR #13154.

### DL-#13113 · Label the Dockerfile.Modular Runtime Stage Keep=True so Host Prune Jobs Spare Every Profile Image; Contract Test Guards It

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13113
- **Branch:** merged via #13138
- **PR:** #13138
- **Paths:** see #13138
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`dea902bb`; collated from changes/13113-label-the-dockerfile-modular-runtime-sta.md)
- **Summary:** Label the Dockerfile.modular runtime stage keep=true so host prune jobs spare every profile image; contract test guards it
- **Next step:** Shipped in PR #13138.

### DL-#12573 · NSX-41: Display_Decision Output-Policy Integration (ADR-0054 D7 Option A)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12573
- **Branch:** merged via #13149
- **PR:** #13149
- **Paths:** see #13149
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`551c914c`; collated from changes/12573-nsx-41-display-decision-output-policy-in.md)
- **Summary:** NSX-41: display_decision output-policy integration (ADR-0054 D7 Option A)
- **Next step:** Shipped in PR #13149.

### DL-#13147 · Ground Reaction Overlay Draws Measured CoP Uncertainty as Per-World-Axis 1-Sigma Ellipses

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13147
- **Branch:** merged via #13147
- **PR:** #13147
- **Paths:** see #13147
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`35a56c55`; collated from changes/13147-ground-reaction-overlay-draws-measured-c.md)
- **Summary:** Ground reaction overlay draws measured CoP uncertainty as per-world-axis 1-sigma ellipses
- **Next step:** Shipped in PR #13147.

### DL-#13141 · Ground Reaction Overlay Rings Measured CoP Points by Their 1-Sigma per Axis

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13141
- **Branch:** merged via #13141
- **PR:** #13141
- **Paths:** see #13141
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`2dc3724e`; collated from changes/13141-ground-reaction-overlay-rings-measured-c.md)
- **Summary:** Ground reaction overlay rings measured CoP points by their 1-sigma per axis
- **Next step:** Shipped in PR #13141.

### DL-#13139 · Ground Reaction Overlay Shades Measured Plate Load by Its 1-Sigma Uncertainty and Names the Declared Uncertainty Source

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13139
- **Branch:** merged via #13139
- **PR:** #13139
- **Paths:** see #13139
- **Started:** 2026-10-18
- **Last verified:** 2026-10-18 (`e0f06035`; collated from changes/13139-ground-reaction-overlay-shades-measured.md)
- **Summary:** Ground reaction overlay shades measured plate load by its 1-sigma uncertainty and names the declared uncertainty source
- **Next step:** Shipped in PR #13139.

### DL-#13134 · Plate Declarations Dialog and Web Form Offer the Cross_Correlation Alignment and Require the Model It Was Fitted Against

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13134
- **Branch:** merged via #13134
- **PR:** #13134
- **Paths:** see #13134
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`2f95b7c6`; collated from changes/13134-plate-declarations-dialog-and-web-form-o.md)
- **Summary:** Plate declarations dialog and web form offer the cross_correlation alignment and require the model it was fitted against
- **Next step:** Shipped in PR #13134.

### DL-#13122 · Web Analysis Tools Ground Reaction Charts Gain the Plate Declarations Form (Same Fields as the PyQt6 Dialog, Guarded by a Cross-Language Parity Test); Analysis.Ground_Reaction Back to Parity

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13122
- **Branch:** merged via #13125
- **PR:** #13125
- **Paths:** see #13125
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`00019c88`; collated from changes/13122-web-analysis-tools-ground-reaction-chart.md)
- **Summary:** Web Analysis Tools ground reaction charts gain the plate declarations form (same fields as the PyQt6 dialog, guarded by a cross-language parity test); analysis.ground_reaction back to parity
- **Next step:** Shipped in PR #13125.

### DL-#13082 · Add DV-13082 Deferred-Validation Record for OF-HW Open-Fp Device Qualification; Refresh DV-13061 Now That the Software Children Have Landed

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13082
- **Branch:** merged via #13129
- **PR:** #13129
- **Paths:** see #13129
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`e98259fe`; collated from changes/13082-add-dv-13082-deferred-validation-record.md)
- **Summary:** Add DV-13082 deferred-validation record for OF-HW open-fp device qualification; refresh DV-13061 now that the software children have landed
- **Next step:** Shipped in PR #13129.

### DL-#13110 · Sidekick Smoke-Test Checks Fixtures Out Into an Isolated Path so the Shared Self-Hosted Workspace Is Never Left a Sparse Partial Clone

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13110
- **Branch:** merged via #13111
- **PR:** #13111
- **Paths:** see #13111
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`9fd898a6`; collated from changes/13110-sidekick-smoke-test-checks-fixtures-out.md)
- **Summary:** Sidekick smoke-test checks fixtures out into an isolated path so the shared self-hosted workspace is never left a sparse partial clone
- **Next step:** Shipped in PR #13111.

### DL-#13037 · Derive the Fit-Path de Leva Table From Whole Table 4 Rows, Keep the Capture Trunk Length by Rescaling Trunk_Scale, and Measure the Head and Trunk-Part CM From the Cranial Row Origin; Canonical Receipts Regenerated

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13037
- **Branch:** merged via #13102
- **PR:** #13102
- **Paths:** see #13102
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`66d3c25a`; collated from changes/13037-derive-the-fit-path-de-leva-table-from-w.md)
- **Summary:** Derive the fit-path de Leva table from whole Table 4 rows, keep the capture trunk length by rescaling trunk_scale, and measure the head and trunk-part CM from the cranial row origin; canonical receipts regenerated
- **Next step:** Shipped in PR #13102.

### DL-#12789 · Static 10 Deg External Tibial-Torsion Offset in the Shared Full-Body Spec (Ankle Base Frames, Cited Seber 2000 Range); 7-Iron Trail Toe-Out Reaches 19.97 Deg With Hip_Rotation_R Off Its Limit; the Four Canonical Ground-Support Receipts Are Regenerated on the Default Style Feet (Calibrated 7-Iron Dynamics 64.35 mm, G3 Passes)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12789
- **Branch:** merged via #12814
- **PR:** #12814, #13436
- **Paths:** see #12814
- **Started:** 2026-10-17
- **Last verified:** 2026-10-19 (`148e34f5`; collated from changes/12789-lock-baseline-receipt-provenance-to-tibi.md)
- **Summary:** Static 10 deg external tibial-torsion offset in the shared full-body spec (ankle base frames, cited Seber 2000 range); 7-iron trail toe-out reaches 19.97 deg with hip_rotation_r off its limit; the four canonical ground-support receipts are regenerated on the default style feet (calibrated 7-iron dynamics 64.35 mm, G3 passes)
- **Next step:** Shipped in PR #12814.

### DL-#13095 · Rotate the 315 Oldest SPEC.md Section 12 Rows (Dated up to 2026-10-14) Verbatim Into a Second Pinned Block of Docs/Spec/CHANGELOG_ARCHIVE.md; SPEC.md 349,957 to 276,739 Bytes Under the Unchanged 350,000-Byte Cap; the Change-Log Gate Reads Every Archived Row Block

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13095
- **Branch:** merged via #13099
- **PR:** #13099
- **Paths:** see #13099
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`3ebdc456`; collated from changes/13095-rotate-the-315-oldest-spec-md-section-12.md)
- **Summary:** Rotate the 315 oldest SPEC.md section 12 rows (dated up to 2026-10-14) verbatim into a second pinned block of docs/spec/CHANGELOG_ARCHIVE.md; SPEC.md 349,957 to 276,739 bytes under the unchanged 350,000-byte cap; the change-log gate reads every archived row block
- **Next step:** Shipped in PR #13099.

### DL-#13068 · Capture Rig No Longer Aborts When Its Panel Is Destroyed With Help or Status Dialogs Alive

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13068
- **Branch:** merged via #13071
- **PR:** #13071
- **Paths:** see #13071
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`13a1d07f`; collated from changes/13068-capture-rig-no-longer-aborts-when-its-pa.md)
- **Summary:** Capture Rig no longer aborts when its panel is destroyed with Help or Status dialogs alive
- **Next step:** Shipped in PR #13071.

### DL-#13058 · Sole-Load View Note Names Its Matplotlib Renderer When a 3-D Provider Is Installed; Adapter Golden Accepts Verified Numpy 2.5.3 Runtime

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13058
- **Branch:** merged via #13059
- **PR:** #13059
- **Paths:** see #13059
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`7f83e101`; collated from changes/13058-sole-load-view-note-names-its-matplotlib.md)
- **Summary:** Sole-load view note names its matplotlib renderer when a 3-D provider is installed; adapter golden accepts verified numpy 2.5.3 runtime
- **Next step:** Shipped in PR #13059.

### DL-#13056 · OS-8 Branch-Continuous Capture-A IK: Constant Marker Set, Warm-Started Tie-Break IK, Continuous Through Impact

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13056
- **Branch:** merged via #13060
- **PR:** #13060, #13076, #13101, #13117
- **Paths:** see #13060
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`bb5df06b`; collated from changes/13056-shoulder-orientation-solve-keeps-its-eul.md)
- **Summary:** OS-8 branch-continuous capture-A IK: constant marker set, warm-started tie-break IK, continuous through impact
- **Next step:** Shipped in PR #13060.

### DL-#13054 · Make Tests/Tools a Package so Its Test Packages Stop Shadowing Src/Tools; Collision Gate Green and Unquarantined

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13054
- **Branch:** merged via #13055
- **PR:** #13055
- **Paths:** see #13055
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`dd723779`; collated from changes/13054-make-tests-tools-a-package-so-its-test-p.md)
- **Summary:** Make tests/tools a package so its test packages stop shadowing src/tools; collision gate green and unquarantined
- **Next step:** Shipped in PR #13055.

### DL-#13050 · Bump Vendor/Ud-Tools to Tools 46C78180, Carrying the Rust Golfer Solver Closure-Row Parity With Python (Tools#5644) and the ChatDockWidget Mixin Split (Tools#5642); Pins and Divergence Inventory Kept in Step

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13050
- **Branch:** merged via #13050
- **PR:** #13050
- **Paths:** see #13050
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`8db3d63d`; collated from changes/13050-bump-vendor-ud-tools-to-tools-46c7818.md)
- **Summary:** Bump vendor/ud-tools to Tools 46c78180, carrying the Rust golfer solver closure-row parity with Python (Tools#5644) and the ChatDockWidget mixin split (Tools#5642); pins and divergence inventory kept in step
- **Next step:** Shipped in PR #13050.

### DL-#13040 · OSL-2: Compose Real OpenSim Musculoskeletal Lift Models (Rajagopal-Lai-Uhlrich Body, Hand-Held Barbell, Smooth Sphere Contact) Replacing the LIFT-6 Stub

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13040
- **Branch:** merged via #13046
- **PR:** #13046
- **Paths:** see #13046
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`728522aa`; collated from changes/13040-osl-2-compose-real-opensim-musculoskelet.md)
- **Summary:** OSL-2: compose real OpenSim musculoskeletal lift models (Rajagopal-Lai-Uhlrich body, hand-held barbell, smooth sphere contact) replacing the LIFT-6 stub
- **Next step:** Shipped in PR #13046.

### DL-#13015 · Resolve the Club-Head Body Across Engine Naming Schemes via One Shared Alias Table; Add Strict Mode

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13015
- **Branch:** merged via #13028
- **PR:** #13028
- **Paths:** see #13028
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`47143bbc`; collated from changes/13015-resolve-the-club-head-body-across-engine.md)
- **Summary:** Resolve the club-head body across engine naming schemes via one shared alias table; add strict mode
- **Next step:** Shipped in PR #13028.

### DL-#13010 · de Leva 1996 Table 4 Transcribed Whole (Both Sexes, All Rows) With Load-Time Invariants; the Humanoid Table Derives From It and Closes to Body Mass Without Rescaling; Fit-Path Mixed Rows Ratcheted (#13037)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13010
- **Branch:** merged via #13039
- **PR:** #13039
- **Paths:** see #13039
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`826e1c8e`; collated from changes/13010-de-leva-1996-table-4-transcribed-whole-b.md)
- **Summary:** de Leva 1996 Table 4 transcribed whole (both sexes, all rows) with load-time invariants; the humanoid table derives from it and closes to body mass without rescaling; fit-path mixed rows ratcheted (#13037)
- **Next step:** Shipped in PR #13039.

### DL-#12992 · Fix(Gui): Restore Dead Settings Menu Actions, Vendored Tile Fallback, Embedded C3D Viewer (#12992, #12993, #12995)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12992
- **Branch:** merged via #13028
- **PR:** #13028
- **Paths:** see #13028
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`47143bbc`; collated from changes/12992-fix-gui-restore-dead-settings-menu-actio.md)
- **Summary:** fix(gui): restore dead settings menu actions, vendored tile fallback, embedded C3D viewer (#12992, #12993, #12995)
- **Next step:** Shipped in PR #13028.

### DL-#12983 · OpenSim 4.6 Engine: Inverse Dynamics, Bias, Gravity, Drift, Counterfactuals and Reset Work and Raise on Failure, Joint Names Follow the State; Committed Golfer Models Load Their Club Meshes (#13016)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12983
- **Branch:** merged via #13034
- **PR:** #13034
- **Paths:** see #13034
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`5472818a`; collated from changes/12983-opensim-4-6-engine-inverse-dynamics-bias.md)
- **Summary:** OpenSim 4.6 engine: inverse dynamics, bias, gravity, drift, counterfactuals and reset work and raise on failure, joint names follow the state; committed golfer models load their club meshes (#13016)
- **Next step:** Shipped in PR #13034.

### DL-#12980 · Fix(Drake): Real Pydrake Forces/Reset APIs and Gravity Sign (#12980, #12981, #12982)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12980
- **Branch:** merged via #13028
- **PR:** #13028
- **Paths:** see #13028
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`47143bbc`; collated from changes/12980-fix-drake-real-pydrake-forces-reset-apis.md)
- **Summary:** fix(drake): real pydrake forces/reset APIs and gravity sign (#12980, #12981, #12982)
- **Next step:** Shipped in PR #13028.

### DL-#12963 · Ci: Propagate Pytest Exit Status Through Tee in Heavy-Test, Soak and Optional-Stack Lanes; Add Workflow Pipefail Lint Test

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12963
- **Branch:** merged via #13018
- **PR:** #13018
- **Paths:** see #13018
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`71884f8e`; collated from changes/12963-ci-propagate-pytest-exit-status-through.md)
- **Summary:** ci: propagate pytest exit status through tee in heavy-test, soak and optional-stack lanes; add workflow pipefail lint test
- **Next step:** Shipped in PR #13018.

### DL-#12948 · Results Playback: Wall-Clock Play With 0.25X/0.5X/1X Speed Selector, Viewport Scales to the Dock, Y-Label No Longer Clipped

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12948
- **Branch:** merged via #13028
- **PR:** #13028
- **Paths:** see #13028
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`47143bbc`; collated from changes/12948-results-playback-wall-clock-play-with-0.md)
- **Summary:** Results Playback: wall-clock play with 0.25x/0.5x/1x speed selector, viewport scales to the dock, y-label no longer clipped
- **Next step:** Shipped in PR #13028.

### DL-#12925 · OpenSim Preview: Orient by Model Gravity (Z-Up Golf Models No Longer Lie on Their Side), Capsules Only for Bodies With Missing Meshes

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12925
- **Branch:** merged via #13028
- **PR:** #13028
- **Paths:** see #13028
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`47143bbc`; collated from changes/12925-opensim-preview-orient-by-model-gravity.md)
- **Summary:** OpenSim preview: orient by model gravity (Z-up golf models no longer lie on their side), capsules only for bodies with missing meshes
- **Next step:** Shipped in PR #13028.

### DL-#12919 · OpenSim Explorer: Resolve Meshes From Repo/Package/OPENSIM_GEOMETRY, Geometry-Missing Banner With Capsule Fallback, Anthro Golfer Default

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12919
- **Branch:** merged via #13028
- **PR:** #13028
- **Paths:** see #13028
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`47143bbc`; collated from changes/12919-opensim-explorer-resolve-meshes-from-rep.md)
- **Summary:** OpenSim explorer: resolve meshes from repo/package/OPENSIM_GEOMETRY, geometry-missing banner with capsule fallback, anthro golfer default
- **Next step:** Shipped in PR #13028.

### DL-#12915 · Model Explorer Opens Swing-Datastore Results (MuJoCo, Drake, OpenSim, Pinocchio) and Plays Them on the Shared-Appearance Golfer With Timeline, Plots and MP4 Export

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12915
- **Branch:** merged via #13028
- **PR:** #13028
- **Paths:** see #13028
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`47143bbc`; collated from changes/12915-model-explorer-opens-swing-datastore-res.md)
- **Summary:** Model Explorer opens swing-datastore results (MuJoCo, Drake, OpenSim, Pinocchio) and plays them on the shared-appearance golfer with timeline, plots and MP4 export
- **Next step:** Shipped in PR #13028.

### DL-#12912 · OpenSim Run Simulation No Longer Calls the Removed Manager.setInitialTime.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12912
- **Branch:** merged via #13028
- **PR:** #13028
- **Paths:** see #13028
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`47143bbc`; collated from changes/12912-opensim-run-simulation-no-longer-calls-t.md)
- **Summary:** OpenSim Run Simulation no longer calls the removed Manager.setInitialTime.
- **Next step:** Shipped in PR #13028.

### DL-#12901 · Launcher Theme Imports Resolve via Launcher_Compat; Sidekick API Child Runs Without Uvicorn Auto-Reload (Ready in ~10s); Closing an Engine Tab Resets Its Running Status (#12962)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12901
- **Branch:** merged via #13028
- **PR:** #13028, #13345
- **Paths:** see #13028
- **Started:** 2026-10-17
- **Last verified:** 2026-10-19 (`8115b862`; collated from changes/12901-defer-heavy-route-and-backend-imports-so.md)
- **Summary:** Launcher theme imports resolve via launcher_compat; Sidekick API child runs without uvicorn auto-reload (ready in ~10s); closing an engine tab resets its Running status (#12962)
- **Next step:** Shipped in PR #13028.

### DL-#12872 · Keep the Runtime Cancel Event Out of Persisted Chat Stream Metadata and Validate JSON-Serialisability Before Persisting

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12872
- **Branch:** merged via #13028
- **PR:** #13028
- **Paths:** see #13028
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`47143bbc`; collated from changes/12872-keep-the-runtime-cancel-event-out-of-per.md)
- **Summary:** Keep the runtime cancel event out of persisted chat stream metadata and validate JSON-serialisability before persisting
- **Next step:** Shipped in PR #13028.

### DL-#12855 · Motion-Matching Smoke: Headless GL Probe and Best-Effort Playback Render (Render_Skipped in Receipt)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12855
- **Branch:** merged via #13029
- **PR:** #13029
- **Paths:** see #13029
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`f565f4a7`; collated from changes/12855-motion-matching-smoke-headless-gl-probe.md)
- **Summary:** Motion-matching smoke: headless GL probe and best-effort playback render (render_skipped in receipt)
- **Next step:** Shipped in PR #13029.

### DL-#12849 · Split Oversized Development and Agent Narrative Docs Into Index Plus Linked Parts; Drop Four Doc-Size Exceptions

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12849
- **Branch:** merged via #13026
- **PR:** #13026
- **Paths:** see #13026
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`c611ab8d`; collated from changes/12849-split-oversized-development-and-agent-na.md)
- **Summary:** Split oversized development and agent narrative docs into index plus linked parts; drop four doc-size exceptions
- **Next step:** Shipped in PR #13026.

### DL-#12845 · Group IK/Eval Tuning Parameters Into Option Objects (TrajectoryIkOptions, DualFrameIkOptions, EvaluationOptions) and Delete the Three Expiring Architecture-Budget Parameter Exceptions

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12845
- **Branch:** merged via #13026
- **PR:** #13026
- **Paths:** see #13026
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`c611ab8d`; collated from changes/12845-group-ik-eval-tuning-parameters-into-opt.md)
- **Summary:** Group IK/eval tuning parameters into option objects (TrajectoryIkOptions, DualFrameIkOptions, EvaluationOptions) and delete the three expiring architecture-budget parameter exceptions
- **Next step:** Shipped in PR #13026.

### DL-#12844 · Introduce Shared Frozen SimulationConfig for Pendulum Run_Simulation Variants and BatchSwingConfig for BatchSwingOptimizer; Delete 5 Expiring Architecture-Budget Exceptions

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12844
- **Branch:** merged via #13026
- **PR:** #13026
- **Paths:** see #13026
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`c611ab8d`; collated from changes/12844-introduce-shared-frozen-simulationconfig.md)
- **Summary:** Introduce shared frozen SimulationConfig for pendulum run_simulation variants and BatchSwingConfig for BatchSwingOptimizer; delete 5 expiring architecture-budget exceptions
- **Next step:** Shipped in PR #13026.

### DL-#12843 · Make Pinocchio/Drake SimOut Real Dataclasses and Decompose Simulate_With_Coefficients, _Record_Rollout and Fit_Swing_Drake Under the Architecture Budget; Delete 8 Expiring Exceptions

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12843
- **Branch:** merged via #13026
- **PR:** #13026
- **Paths:** see #13026
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`c611ab8d`; collated from changes/12843-make-pinocchio-drake-simout-real-datacla.md)
- **Summary:** Make pinocchio/drake SimOut real dataclasses and decompose simulate_with_coefficients, _record_rollout and fit_swing_drake under the architecture budget; delete 8 expiring exceptions
- **Next step:** Shipped in PR #13026.

### DL-#12842 · Split MuJoCo Motion-Matching Simulate_With_Coefficients, Compute_Qpos_Jacobian and Compute_Cost_Gradient_Analytical Into Helpers Under the Function-Lines Budget (Bit-Identical Outputs); Drop Their Expiring Exceptions

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12842
- **Branch:** merged via #13026
- **PR:** #13026
- **Paths:** see #13026
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`c611ab8d`; collated from changes/12842-split-mujoco-motion-matching-simulate-wi.md)
- **Summary:** Split MuJoCo motion-matching simulate_with_coefficients, compute_qpos_jacobian and compute_cost_gradient_analytical into helpers under the function-lines budget (bit-identical outputs); drop their expiring exceptions
- **Next step:** Shipped in PR #13026.

### DL-#12840 · Split Launch_Monitor_Analytics gui.py Into Cohesive _Gui_* Mixin Modules and Decompose TerrainExplorerWidget._Build_Ui; Remove Expiring Size/Architecture Budget Exceptions.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12840
- **Branch:** merged via #13026
- **PR:** #13026
- **Paths:** see #13026
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`c611ab8d`; collated from changes/12840-split-launch-monitor-analytics-gui-py-in.md)
- **Summary:** Split launch_monitor_analytics gui.py into cohesive _gui_* mixin modules and decompose TerrainExplorerWidget._build_ui; remove expiring size/architecture budget exceptions.
- **Next step:** Shipped in PR #13026.

### DL-#12839 · Decompose Launcher UI Widgets (Rebuild_Grid, Init_Ui, LibraryWidget._Setup_Ui, CustomTitleBar) Under the Architecture Budget and Drop the Expiring Exceptions

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12839
- **Branch:** merged via #13026
- **PR:** #13026
- **Paths:** see #13026
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`c611ab8d`; collated from changes/12839-decompose-launcher-ui-widgets-rebuild-gr.md)
- **Summary:** Decompose launcher UI widgets (rebuild_grid, init_ui, LibraryWidget._setup_ui, CustomTitleBar) under the architecture budget and drop the expiring exceptions
- **Next step:** Shipped in PR #13026.

### DL-#12838 · Re-Vendor Refactored Deferred_Validation/Handoff_Validator (RM#2112) With Handoff_Schema and Archive_Policy; Delete 3 Expiring Function-Lines Exceptions

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12838
- **Branch:** merged via #12862
- **PR:** #12862
- **Paths:** see #12862
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`b1bf9306`; collated from changes/12838-re-vendor-refactored-deferred-validation.md)
- **Summary:** Re-vendor refactored deferred_validation/handoff_validator (RM#2112) with handoff_schema and archive_policy; delete 3 expiring function-lines exceptions
- **Next step:** Shipped in PR #12862.

### DL-#12807 · Cross-Engine _Safe_Cv Returns NaN at Near-Zero Mean; Robustness_Score Fails Closed and the API Reports Null With a Reason Instead of Perfect Robustness

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12807
- **Branch:** fix/12807-cross-engine-zero-mean-cv
- **PR:** #13029
- **Paths:** see #13029
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`f565f4a7`; collated from changes/12807-cross-engine-safe-cv-returns-nan-at-near.md)
- **Summary:** Cross-engine _safe_cv returns NaN at near-zero mean; robustness_score fails closed and the API reports null with a reason instead of perfect robustness
- **Next step:** Shipped in PR #13029.

### DL-#12803 · Re-Synced the Chat Child Copy With the Pinned Tools Tree: ChatDockWidget.**Init** Now Takes ChatConnectionConfig/ChatPresentationConfig/ChatIntegrationHooks (Tools#4897), Call Sites Updated, and the Expiring Architecture-Budget Exception Removed.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12803
- **Branch:** merged via #12858
- **PR:** #12858
- **Paths:** see #12858
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`ad09ed35`; collated from changes/12803-re-synced-the-chat-child-copy-with-the-p.md)
- **Summary:** Re-synced the chat child copy with the pinned Tools tree: ChatDockWidget.**init** now takes ChatConnectionConfig/ChatPresentationConfig/ChatIntegrationHooks (Tools#4897), call sites updated, and the expiring architecture-budget exception removed.
- **Next step:** Shipped in PR #12858.

### DL-#12758 · SPN-9C: Full-Body Driving of the Bruno Model (Coordinate Map, Shoulder Orientation Match, Club Hand Wrench and Ground Reaction Loads, Study Option and Receipt Record)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12758
- **Branch:** merged via #12917
- **PR:** #12917
- **Paths:** see #12917
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`5c015d16`; collated from changes/12758-spn-9c-full-body-driving-of-the-bruno-mo.md)
- **Summary:** SPN-9c: full-body driving of the Bruno model (coordinate map, shoulder orientation match, club hand wrench and ground reaction loads, study option and receipt record)
- **Next step:** Shipped in PR #12917.

### DL-#12727 · NSX-98: Two-Track Conformance Rows for Closed-Loop Sensitivities and Variance Across Pinocchio, Drake, MuJoCo, MyoSuite and OpenSim

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12727
- **Branch:** merged via #13033
- **PR:** #13033
- **Paths:** see #13033
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`73738367`; collated from changes/12727-nsx-98-two-track-conformance-rows-for-cl.md)
- **Summary:** NSX-98: two-track conformance rows for closed-loop sensitivities and variance across Pinocchio, Drake, MuJoCo, MyoSuite and OpenSim
- **Next step:** Shipped in PR #13033.

### DL-#12726 · NSX-97: MOSAIC Multi-Swing Variability Consumer With UCM/ORT, Muller-Sternad TNC, Direction Comparison, Phase and Physical-Time Residuals and Session-Level Train/Test Split (Real-Capture Falsification Deferred)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12726
- **Branch:** merged via #13033
- **PR:** #13033
- **Paths:** see #13033
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`73738367`; collated from changes/12726-nsx-97-mosaic-multi-swing-variability-co.md)
- **Summary:** NSX-97: MOSAIC multi-swing variability consumer with UCM/ORT, Muller-Sternad TNC, direction comparison, phase and physical-time residuals and session-level train/test split (real-capture falsification deferred)
- **Next step:** Shipped in PR #13033.

### DL-#12725 · NSX-96: Saltelli/Jansen Sobol and PRCC Global Sensitivity Screening With Grouped Indices on the Planar Oracle and Low-Rank Basis

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12725
- **Branch:** merged via #13033
- **PR:** #13033
- **Paths:** see #13033
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`73738367`; collated from changes/12725-nsx-96-saltelli-jansen-sobol-and-prcc-gl.md)
- **Summary:** NSX-96: Saltelli/Jansen Sobol and PRCC global sensitivity screening with grouped indices on the planar oracle and low-rank basis
- **Next step:** Shipped in PR #13033.

### DL-#12724 · NSX-95 Strategy Catalogue S1-S12 With Matched-CHS/Matched-Effort Campaigns, Prior Reversals, Held-Out Stability and Variance Budget

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12724
- **Branch:** merged via #13033
- **PR:** #13033
- **Paths:** see #13033
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`73738367`; collated from changes/12724-nsx-95-strategy-catalogue-s1-s12-with-ma.md)
- **Summary:** NSX-95 strategy catalogue S1-S12 with matched-CHS/matched-effort campaigns, prior reversals, held-out stability and variance budget
- **Next step:** Shipped in PR #13033.

### DL-#12723 · NSX-94: Muscle-Level Co-Contraction Experiments (Activation Fibre, Signal-Dependent Muscle Noise, Short-Range Stiffness) for OpenSim and MuJoCo-Muscle, Fail-Closed Capability Registry

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12723
- **Branch:** merged via #13033
- **PR:** #13033
- **Paths:** see #13033
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`73738367`; collated from changes/12723-nsx-94-muscle-level-co-contraction-exper.md)
- **Summary:** NSX-94: muscle-level co-contraction experiments (activation fibre, signal-dependent muscle noise, short-range stiffness) for OpenSim and MuJoCo-muscle, fail-closed capability registry
- **Next step:** Shipped in PR #13033.

### DL-#12722 · NSX-93: Declared Joint-Impedance and Delayed Minimum-Intervention Feedback Layer for Torque-Driven Models

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12722
- **Branch:** merged via #13033
- **PR:** #13033
- **Paths:** see #13033
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`73738367`; collated from changes/12722-nsx-93-declared-joint-impedance-and-dela.md)
- **Summary:** NSX-93: declared joint-impedance and delayed minimum-intervention feedback layer for torque-driven models
- **Next step:** Shipped in PR #13033.

### DL-#12721 · NSX-92 Noise-Aware Null-Space Allocation: Minimum Face-Angle Variance Under Declared M1/M2 With Convex and Local-Certified Solves, Fronts and Held-Out MC Verification

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12721
- **Branch:** merged via #13033
- **PR:** #13033
- **Paths:** see #13033
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`73738367`; collated from changes/12721-nsx-92-noise-aware-null-space-allocation.md)
- **Summary:** NSX-92 noise-aware null-space allocation: minimum face-angle variance under declared M1/M2 with convex and local-certified solves, fronts and held-out MC verification
- **Next step:** Shipped in PR #13033.

### DL-#12720 · NSX-91 Paired Nonlinear Monte Carlo on Constrained Replay (CRN, Hold-Out Seeds, Retained Failures)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12720
- **Branch:** merged via #13033
- **PR:** #13033
- **Paths:** see #13033
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`73738367`; collated from changes/12720-nsx-91-paired-nonlinear-monte-carlo-on-c.md)
- **Summary:** NSX-91 paired nonlinear Monte Carlo on constrained replay (CRN, hold-out seeds, retained failures)
- **Next step:** Shipped in PR #13033.

### DL-#12715 · NSX-86: Drake AutoDiff Constrained-Dynamics Derivatives (Closed_Loop_Derivatives FULL) Through the NSX-84 Seam, Canonical Closure Convention, Capability Rejection Instead of FD Fallback

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12715
- **Branch:** merged via #13033
- **PR:** #13033
- **Paths:** see #13033
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`73738367`; collated from changes/12715-nsx-86-drake-autodiff-constrained-dynami.md)
- **Summary:** NSX-86: Drake AutoDiff constrained-dynamics derivatives (closed_loop_derivatives FULL) through the NSX-84 seam, canonical closure convention, capability rejection instead of FD fallback
- **Next step:** Shipped in PR #13033.

### DL-#12706 · Humanoid_Character_Builder: One Primitive and One SMPL-X Mesh Generator Class, URDF Generator and URDFWriter Collision Exclusions Restored After #8322, Precondition Binding by Name, and the Builder Unit Tests Now Run in the Unit Gate

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12706
- **Branch:** merged via #13029
- **PR:** #13029
- **Paths:** see #13029
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`f565f4a7`; collated from changes/12706-humanoid-builder-dedupe-hidden-tests.md)
- **Summary:** humanoid_character_builder: one Primitive and one SMPL-X mesh generator class, URDF generator and URDFWriter collision exclusions restored after #8322, precondition binding by name, and the builder unit tests now run in the unit gate
- **Next step:** Shipped in PR #13029.

### DL-#12698 · Dedupe MakeHumanMeshGenerator: One Class and One _Convert_Params_to_Makehuman in _Mesh_Makehuman; Mesh-Generator Tests Marked Unit

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12698
- **Branch:** merged via #13029
- **PR:** #13029
- **Paths:** see #13029
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`f565f4a7`; collated from changes/12698-dedupe-makehumanmeshgenerator-one-class.md)
- **Summary:** Dedupe MakeHumanMeshGenerator: one class and one _convert_params_to_makehuman in _mesh_makehuman; mesh-generator tests marked unit
- **Next step:** Shipped in PR #13029.

### DL-#12689 · R9b: OpenSim and MyoSuite Viewers Render the Shared Appearance (OpenSim Registers R8 Body Meshes in the Spec Body Frame; MyoSuite Uses the MuJoCo Appearance Layer)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12689
- **Branch:** merged via #12910
- **PR:** #12910
- **Paths:** see #12910
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`34968337`; collated from changes/12689-r9b-opensim-and-myosuite-viewers-render.md)
- **Summary:** R9b: OpenSim and MyoSuite viewers render the shared appearance (OpenSim registers R8 body meshes in the spec body frame; MyoSuite uses the MuJoCo appearance layer)
- **Next step:** Shipped in PR #12910.

### DL-#12590 · NSX-72: True Closure Alternatives With Kinematics Re-Derived per Grip Model (Research Report, Per-Model Admission)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12590
- **Branch:** merged via #13033
- **PR:** #13033
- **Paths:** see #13033
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`73738367`; collated from changes/12590-nsx-72-true-closure-alternatives-with-ki.md)
- **Summary:** NSX-72: true closure alternatives with kinematics re-derived per grip model (research report, per-model admission)
- **Next step:** Shipped in PR #13033.

### DL-#12589 · NSX-71 Inverse Optimal Control of Allocation Weights: Pre-Registered Held-Out Report, KKT-Residual Module and Synthetic Recovery Tests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12589
- **Branch:** merged via #13033
- **PR:** #13033
- **Paths:** see #13033
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`73738367`; collated from changes/12589-nsx-71-inverse-optimal-control-of-alloca.md)
- **Summary:** NSX-71 inverse optimal control of allocation weights: pre-registered held-out report, KKT-residual module and synthetic recovery tests
- **Next step:** Shipped in PR #13033.

### DL-#12588 · NSX-70 Hit-and-Run Sampling Research Spike on a Low-Rank Temporal Basis With Mixing Diagnostics

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12588
- **Branch:** merged via #13033
- **PR:** #13033
- **Paths:** see #13033
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`73738367`; collated from changes/12588-nsx-70-hit-and-run-sampling-research-spi.md)
- **Summary:** NSX-70 hit-and-run sampling research spike on a low-rank temporal basis with mixing diagnostics
- **Next step:** Shipped in PR #13033.

### DL-#12587 · NSX-62: Inertial Parameter Ensemble Over Identifiability-Gated Parameters With LMI Rejection, Sensitivity Report and Anthro Acceptance

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12587
- **Branch:** merged via #13033
- **PR:** #13033
- **Paths:** see #13033
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`73738367`; collated from changes/12587-nsx-62-inertial-parameter-ensemble-over.md)
- **Summary:** NSX-62: inertial parameter ensemble over identifiability-gated parameters with LMI rejection, sensitivity report and anthro acceptance
- **Next step:** Shipped in PR #13033.

### DL-#12585 · Lower-Body Loop: Two-Foot Ground Contact Null Space, GRF Observations, Real Anthro Envelope Benchmark (NSX-60, NSX-22)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12585
- **Branch:** merged via #13033
- **PR:** #13033
- **Paths:** see #13033
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`73738367`; collated from changes/12585-lower-body-loop-two-foot-ground-contact.md)
- **Summary:** Lower-body loop: two-foot ground contact null space, GRF observations, real anthro envelope benchmark (NSX-60, NSX-22)
- **Next step:** Shipped in PR #13033.

### DL-#12572 · NSX-40: Loop-Nullspace Report Schema V1, Facade, Service, Async Cancellable Job Routes With Result Cache

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12572
- **Branch:** merged via #13033
- **PR:** #13033
- **Paths:** see #13033
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`73738367`; collated from changes/12572-nsx-40-loop-nullspace-report-schema-v1-f.md)
- **Summary:** NSX-40: loop-nullspace report schema v1, facade, service, async cancellable job routes with result cache
- **Next step:** Shipped in PR #13033.

### DL-#12571 · NSX-31: Open-Loop Versus Feedback-Assisted Replay of Allocated Torques With a Separate Feedback Budget (Loop_Nullspace/replay.py)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12571
- **Branch:** merged via #13033
- **PR:** #13033
- **Paths:** see #13033
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`73738367`; collated from changes/12571-nsx-31-open-loop-versus-feedback-assiste.md)
- **Summary:** NSX-31: open-loop versus feedback-assisted replay of allocated torques with a separate feedback budget (loop_nullspace/replay.py)
- **Next step:** Shipped in PR #13033.

### DL-#12569 · NSX-26: Reaction-Restriction Sensitivity (WELD6 to SHAFT_FREE5 to BALL3) With Anthro Report

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12569
- **Branch:** merged via #13033
- **PR:** #13033
- **Paths:** see #13033
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`73738367`; collated from changes/12569-nsx-26-reaction-restriction-sensitivity.md)
- **Summary:** NSX-26: reaction-restriction sensitivity (WELD6 to SHAFT_FREE5 to BALL3) with anthro report
- **Next step:** Shipped in PR #13033.

### DL-#12568 · NSX-25: Epsilon-Constraint Pareto Fronts and Dual-Certified Affine Probes Over the Trajectory Feasible Set

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12568
- **Branch:** merged via #13033
- **PR:** #13033
- **Paths:** see #13033
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`73738367`; collated from changes/12568-nsx-25-epsilon-constraint-pareto-fronts.md)
- **Summary:** NSX-25: epsilon-constraint Pareto fronts and dual-certified affine probes over the trajectory feasible set
- **Next step:** Shipped in PR #13033.

### DL-#12567 · Consolidate ContactForceAllocator and Redistribute_Forces/Trajectory Onto the Conic Core (Qp_Core); DeprecationWarning Shims, No SLSQP on Convex Allocation Paths

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12567
- **Branch:** merged via #13033
- **PR:** #13033
- **Paths:** see #13033
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`73738367`; collated from changes/12567-consolidate-contactforceallocator-and-re.md)
- **Summary:** Consolidate ContactForceAllocator and redistribute_forces/trajectory onto the conic core (qp_core); DeprecationWarning shims, no SLSQP on convex allocation paths
- **Next step:** Shipped in PR #13033.

### DL-#12566 · NSX-23 Anthro Trajectory Allocation and NSX-61 (#12586) Anthro Kinematic Ensemble Acceptance Reports on the NSX-60 Foot-Contact Fixture

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12566
- **Branch:** merged via #13033
- **PR:** #13033
- **Paths:** see #13033
- **Started:** 2026-10-17
- **Last verified:** 2026-10-17 (`73738367`; collated from changes/12566-nsx-23-trajectory-conic-allocation-capac.md)
- **Summary:** NSX-23 anthro trajectory allocation and NSX-61 (#12586) anthro kinematic ensemble acceptance reports on the NSX-60 foot-contact fixture
- **Next step:** Shipped in PR #13033.

### DL-#12815 · Web Spine Viewer: /Api/Tools/Spine-Viewer Routes and React Page With Always-Visible Scientific Status and Distribution Source (SPN-15C)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12815
- **Branch:** merged via #12897
- **PR:** #12897
- **Paths:** see #12897
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`883e962d`; collated from changes/12815-web-spine-viewer-api-tools-spine-viewer.md)
- **Summary:** Web spine viewer: /api/tools/spine-viewer routes and React page with always-visible scientific status and distribution source (SPN-15c)
- **Next step:** Shipped in PR #12897.

### DL-#12828 · Correct the Golf-Term Evaluate Docstring for Side_Bend_Tilt Input and Output Shapes

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12828
- **Branch:** merged via #12867
- **PR:** #12867
- **Paths:** see #12867
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`303a81b6`; collated from changes/12828-side-bend-tilt-docstring.md)
- **Summary:** Correct the golf-term evaluate docstring for side_bend_tilt input and output shapes
- **Next step:** Shipped in PR #12867.

### DL-#12827 · Static Optimization Holds Zero-Capacity Muscles at Zero, Fixing the Early-Sample Failure on the Synthetic Fixture

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12827
- **Branch:** merged via #12867
- **PR:** #12867
- **Paths:** see #12867
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`303a81b6`; collated from changes/12827-so-zero-capacity-muscles.md)
- **Summary:** Static optimization holds zero-capacity muscles at zero, fixing the early-sample failure on the synthetic fixture
- **Next step:** Shipped in PR #12867.

### DL-#12826 · A Missing ExternalLoads Data File Raises Before Any Study Stage Runs; Relative Datafiles Resolve Against the XML Directory

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12826
- **Branch:** merged via #12867
- **PR:** #12867
- **Paths:** see #12867
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`303a81b6`; collated from changes/12826-external-loads-fail-closed.md)
- **Summary:** A missing ExternalLoads data file raises before any study stage runs; relative datafiles resolve against the XML directory
- **Next step:** Shipped in PR #12867.

### DL-#12825 · Movement-Variant Undeclared-Term Tolerance Is Enforced at Study Run Time; a Violation Makes the Run Unqualified

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12825
- **Branch:** merged via #12867
- **PR:** #12867
- **Paths:** see #12867
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`303a81b6`; collated from changes/12825-variant-undeclared-runtime.md)
- **Summary:** Movement-variant undeclared-term tolerance is enforced at study run time; a violation makes the run unqualified
- **Next step:** Shipped in PR #12867.

### DL-#12824 · Pelvis Residual Height Is the Model Extent Along Gravity at the First Frame, With Its Source Recorded in the Budget and Receipt

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12824
- **Branch:** merged via #12867
- **PR:** #12867
- **Paths:** see #12867
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`303a81b6`; collated from changes/12824-residual-height-first-frame.md)
- **Summary:** Pelvis residual height is the model extent along gravity at the first frame, with its source recorded in the budget and receipt
- **Next step:** Shipped in PR #12867.

### DL-#12823 · Study Requests Carry the Drive Torso Alignment and Its Calibration Source; Receipts Record and Hash It, and an Unsupplied Alignment Makes the Study Unqualified

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12823
- **Branch:** merged via #12867
- **PR:** #12867
- **Paths:** see #12867
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`303a81b6`; collated from changes/12823-drive-alignment-receipt.md)
- **Summary:** Study requests carry the drive torso alignment and its calibration source; receipts record and hash it, and an unsupplied alignment makes the study unqualified
- **Next step:** Shipped in PR #12867.

### DL-#12822 · Golf-Term Contributions Attribute Base, Terminal and Alignment Rotations as Named Entries (Segmental_Contribution_Breakdown) and Close on the Full Pelvis-to-Thorax Term

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12822
- **Branch:** merged via #12867
- **PR:** #12867
- **Paths:** see #12867
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`303a81b6`; collated from changes/12822-contribution-breakdown.md)
- **Summary:** Golf-term contributions attribute base, terminal and alignment rotations as named entries (segmental_contribution_breakdown) and close on the full pelvis-to-thorax term
- **Next step:** Shipped in PR #12867.

### DL-#12821 · Fujii 2007 Coupled Flexion Is Even in Trunk Rotation: New ParityCouplingLaw, Fujii2007_Lumbar Law Version 2, Row Numbers Unchanged

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12821
- **Branch:** merged via #12867
- **PR:** #12867
- **Paths:** see #12867
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`303a81b6`; collated from changes/12821-fujii-parity-law.md)
- **Summary:** Fujii 2007 coupled flexion is even in trunk rotation: new ParityCouplingLaw, fujii2007_lumbar law version 2, row numbers unchanged
- **Next step:** Shipped in PR #12867.

### DL-#12755 · SPN-8C: Translational DOFs in SpineInventory, Extract_Passive_Elements Bushing Reader, Bruno Names-Only Level Map, Asset-Gated Tests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12755
- **Branch:** merged via #12896
- **PR:** #12896
- **Paths:** see #12896
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`816fcac1`; collated from changes/12755-spn-8c-translational-dofs-in-spineinvent.md)
- **Summary:** SPN-8c: translational DOFs in SpineInventory, extract_passive_elements bushing reader, Bruno names-only level map, asset-gated tests
- **Next step:** Shipped in PR #12896.

### DL-#12749 · Pin the Bruno 2015 Full-Body Model Hash With the Owner's 2026-10-16 SimTK Licence Acceptance; Senteler Stays Unpinned

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12749
- **Branch:** merged via #13021
- **PR:** #13021, #13029
- **Paths:** see #13021
- **Started:** 2026-10-16
- **Last verified:** 2026-10-17 (`f565f4a7`; collated from changes/12749-pin-the-bruno-2015-full-body-model-hash.md)
- **Summary:** Pin the Bruno 2015 full-body model hash with the owner's 2026-10-16 SimTK licence acceptance; Senteler stays unpinned
- **Next step:** Shipped in PR #13021.

### DL-#12964 · Confine Launcher Logo Route to Its Asset Directory (Fixes Unauthenticated Arbitrary File Read)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12964
- **Branch:** merged via #13020
- **PR:** #13020
- **Paths:** see #13020
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`286f7191`; collated from changes/12964-confine-launcher-logo-route-to-its-asset.md)
- **Summary:** Confine launcher logo route to its asset directory (fixes unauthenticated arbitrary file read)
- **Next step:** Shipped in PR #13020.

### DL-#12955 · Render the Address Ball at the Club's Own Face on the Ground for Every Club (Drop the Stale Capture Ball and the Duplicate Static-Pose MuJoCo Ball)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12955
- **Branch:** merged via #12966
- **PR:** #12966
- **Paths:** see #12966
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`c9a8612e`; collated from changes/12955-render-the-address-ball-at-the-club-s-ow.md)
- **Summary:** Render the address ball at the club's own face on the ground for every club (drop the stale capture ball and the duplicate static-pose MuJoCo ball)
- **Next step:** Shipped in PR #12966.

### DL-#12949 · CC0 Short-Hair Head Is Now the Default HeadSettings.Model in Every Engine; Parametric Stays Selectable

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12949
- **Branch:** merged via #12956
- **PR:** #12956
- **Paths:** see #12956
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`5226cbd6`; collated from changes/12949-cc0-short-hair-head-is-now-the-default-h.md)
- **Summary:** CC0 short-hair head is now the default HeadSettings.model in every engine; parametric stays selectable
- **Next step:** Shipped in PR #12956.

### DL-#12774 · Record Segmental Spine Contract Ownership; Open Tools#5630 and OpenSim_Models#464

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12774
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12774-record-segmental-spine-contract-ownershi.md)
- **Summary:** Record segmental spine contract ownership; open Tools#5630 and OpenSim_Models#464
- **Next step:** Shipped in PR #12796.

### DL-#12773 · Rewrite Segmental Spine Manual Chapter 66 as a Calculation-Level Reference After Panel Review; Declare the Fujii Flexion Parity Defect; Correct the Gravity-Only Residual Wording

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12773
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12773-spn-16-consolidate-segmental-spine-manua.md)
- **Summary:** Rewrite segmental spine manual chapter 66 as a calculation-level reference after panel review; declare the Fujii flexion parity defect; correct the gravity-only residual wording
- **Next step:** Shipped in PR #12796.

### DL-#12772 · Add PyQt6 Spine Viewer Tool With Embed Adapter, Tile and Feature-Parity Entry (SPN-15B)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12772
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12772-add-pyqt6-spine-viewer-tool-with-embed-a.md)
- **Summary:** Add PyQt6 Spine Viewer tool with embed adapter, tile and feature-parity entry (SPN-15b)
- **Next step:** Shipped in PR #12796.

### DL-#12771 · SPN-15a: Pure-Data Spine Angle-Stack and Load-Heatmap Adapter Plus Agg Report Figures

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12771
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12771-spn-15a-pure-data-spine-angle-stack-and.md)
- **Summary:** SPN-15a: pure-data spine angle-stack and load-heatmap adapter plus Agg report figures
- **Next step:** Shipped in PR #12796.

### DL-#12769 · SPN-14a: Distribution Sensitivity Within One Model Across Shipped Coupling Laws, With Recorded Law Restriction to the Model's Levels

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12769
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12769-spn-14a-distribution-sensitivity-within.md)
- **Summary:** SPN-14a: distribution sensitivity within one model across shipped coupling laws, with recorded law restriction to the model's levels
- **Next step:** Shipped in PR #12796.

### DL-#12768 · SPN-13C: Spine Study Aggregation Tables

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12768
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12768-spn-13c-spine-study-aggregation-tables.md)
- **Summary:** SPN-13c: spine study aggregation tables
- **Next step:** Shipped in PR #12796.

### DL-#12767 · Spine Study Requests Can Run a Movement Variant With Fingerprint and Receipt Provenance (SPN-13D)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12767
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12767-spn-13b-movement-variant-generator-five.md)
- **Summary:** Spine study requests can run a movement variant with fingerprint and receipt provenance (SPN-13d)
- **Next step:** Shipped in PR #12796.

### DL-#12766 · SPN-13a: Deterministic Hash-Pinned Spine Study Request -> Receipt (Spine.Study)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12766
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12766-spn-13a-deterministic-hash-pinned-spine.md)
- **Summary:** SPN-13a: deterministic hash-pinned spine study request -> receipt (spine.study)
- **Next step:** Shipped in PR #12796.

### DL-#12765 · SPN-12: SpinalLoadAnalyzer Heuristic/Model Provider Split

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12765
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12765-spn-12-spinalloadanalyzer-heuristic-mode.md)
- **Summary:** SPN-12: SpinalLoadAnalyzer heuristic/model provider split
- **Next step:** Shipped in PR #12796.

### DL-#12764 · SPN-11B: Golf Literature Context Bands (Hosea 1990, Lim 2012) Annotated Beside Model Peaks, Never Acceptance

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12764
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12764-spn-11b-golf-literature-context-bands-ho.md)
- **Summary:** SPN-11b: golf literature context bands (Hosea 1990, Lim 2012) annotated beside model peaks, never acceptance
- **Next step:** Shipped in PR #12796.

### DL-#12762 · SPN-10C: Per-Level Joint Reactions With SO Muscle Forces as a JOINT_REACTION SpinalLoadSeries

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12762
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12762-spn-10c-per-level-joint-reactions-with-s.md)
- **Summary:** SPN-10c: per-level joint reactions with SO muscle forces as a JOINT_REACTION SpinalLoadSeries
- **Next step:** Shipped in PR #12796.

### DL-#12761 · SPN-10B: Trunk Static Optimisation With Cost Exponent P in {2,3}, Lexicographic Reserves With Flagging, Shared Numeric Core With the Leg SO, Per-Coordinate ID Accessor

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12761
- **Branch:** merged via #12796
- **PR:** #12796, #12958, #13429
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-19 (`9969c0aa`; collated from changes/12761-record-native-bruno-capture-a-runs-pelvi.md)
- **Summary:** SPN-10b: trunk static optimisation with cost exponent p in {2,3}, lexicographic reserves with flagging, shared numeric core with the leg SO, per-coordinate ID accessor
- **Next step:** Shipped in PR #12796.

### DL-#12760 · Spinal Inverse Dynamics With External Loads: Generalized Effort per Level and Pelvis Residual Budget (SPN-10A)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12760
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12760-spinal-inverse-dynamics-with-external-lo.md)
- **Summary:** Spinal inverse dynamics with external loads: generalized effort per level and pelvis residual budget (SPN-10a)
- **Next step:** Shipped in PR #12796.

### DL-#12759 · SPN-9D: Read OpenSim Motions Into SegmentalKinematicsSeries (Inverse of Export_Mot)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12759
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12759-spn-9d-read-opensim-motions-into-segment.md)
- **Summary:** SPN-9d: read OpenSim motions into SegmentalKinematicsSeries (inverse of export_mot)
- **Next step:** Shipped in PR #12796.

### DL-#12757 · SPN-9B: Drive a Coupled Segmental Model From Golf Trunk Input With Per-Frame Residuals and .mot Export

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12757
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12757-spn-9b-drive-a-coupled-segmental-model-f.md)
- **Summary:** SPN-9b: drive a coupled segmental model from golf trunk input with per-frame residuals and .mot export
- **Next step:** Shipped in PR #12796.

### DL-#12756 · SPN-9a: Golf Humanoid Motion Reader With Native Pelvis-to-Thorax Rotation and SLERP Resampling

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12756
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12756-spn-9a-golf-humanoid-motion-reader-with.md)
- **Summary:** SPN-9a: golf humanoid motion reader with native pelvis-to-thorax rotation and SLERP resampling
- **Next step:** Shipped in PR #12796.

### DL-#12754 · SPN-8B: Extract OpenSim CoordinateCouplerConstraints Into a CouplingLaw

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12754
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12754-spn-8b-extract-opensim-coordinatecoupler.md)
- **Summary:** SPN-8b: extract OpenSim CoordinateCouplerConstraints into a CouplingLaw
- **Next step:** Shipped in PR #12796.

### DL-#12753 · SPN-8a: OpenSim Spine Inventory, Level Maps and ISB Axis Check

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12753
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12753-spn-8a-opensim-spine-inventory-level-map.md)
- **Summary:** SPN-8a: OpenSim spine inventory, level maps and ISB axis check
- **Next step:** Shipped in PR #12796.

### DL-#12752 · SPN-7C: OpenSim Lane Guard Turns Absent-OpenSim Spine Skips Into Failures (UD_REQUIRE_OPENSIM=1)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12752
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12752-spn-7c-opensim-lane-guard-turns-absent-o.md)
- **Summary:** SPN-7c: OpenSim lane guard turns absent-OpenSim spine skips into failures (UD_REQUIRE_OPENSIM=1)
- **Next step:** Shipped in PR #12796.

### DL-#12751 · SPN-7B: Fixture Coupler Constraints, Massless Driver and Lumped Trunk Muscles

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12751
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12751-spn-7b-fixture-coupler-constraints-massl.md)
- **Summary:** SPN-7b: fixture coupler constraints, massless driver and lumped trunk muscles
- **Next step:** Shipped in PR #12796.

### DL-#12750 · SPN-7a: Synthetic Segmental Spine OpenSim Fixture Generator and Native Composition Check

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12750
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12750-spn-7a-synthetic-segmental-spine-opensim.md)
- **Summary:** SPN-7a: synthetic segmental spine OpenSim fixture generator and native composition check
- **Next step:** Shipped in PR #12796.

### DL-#12746 · SPN-4C: Swing-Metrics and API X-Factor Delegate to Golf-Term Registry

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12746
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12746-spn-4c-swing-metrics-and-api-x-factor-de.md)
- **Summary:** SPN-4c: swing-metrics and API X-factor delegate to golf-term registry
- **Next step:** Shipped in PR #12796.

### DL-#12745 · Migrate SpinalLoadAnalyzer X-Factor and Crunch Factor to the Golf-Term Registry (SPN-4B), Behaviour-Preserving

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12745
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12745-migrate-spinalloadanalyzer-x-factor-and.md)
- **Summary:** Migrate SpinalLoadAnalyzer X-factor and crunch factor to the golf-term registry (SPN-4b), behaviour-preserving
- **Next step:** Shipped in PR #12796.

### DL-#12744 · Add Spine.Golf_Terms Versioned Golf-Term Registry (SPN-4A)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12744
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12744-add-spine-golf-terms-versioned-golf-term.md)
- **Summary:** Add spine.golf_terms versioned golf-term registry (SPN-4a)
- **Next step:** Shipped in PR #12796.

### DL-#12743 · SPN-3 Partial: Cited Coupling-Law Data and Loader; Verified Laws Only (Bruno2015_Regional, Fujii2007_Lumbar, Uniform_Reference); Per-Level Bruno and Pearcy Pending

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12743
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12743-spn-3-partial-cited-coupling-law-data-an.md)
- **Summary:** SPN-3 partial: cited coupling-law data and loader; verified laws only (bruno2015_regional, fujii2007_lumbar, uniform_reference); per-level Bruno and Pearcy pending
- **Next step:** Shipped in PR #12796.

### DL-#12742 · Spine Distribution Problem: Geodesic Least-Squares Distribution of Pelvis-Thorax Rotation Over Levels Under a Fixed Coupling Law (SPN-2B)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12742
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12742-spine-distribution-problem-geodesic-leas.md)
- **Summary:** Spine distribution problem: geodesic least-squares distribution of pelvis-thorax rotation over levels under a fixed coupling law (SPN-2b)
- **Next step:** Shipped in PR #12796.

### DL-#12741 · Add Spine Coupling Laws (Linear, State-Dependent, MyoFullBody Adapter) and Regional Fraction (SPN-2A)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12741
- **Branch:** merged via #12796
- **PR:** #12796
- **Paths:** see #12796
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12741-add-spine-coupling-laws-linear-state-dep.md)
- **Summary:** Add spine coupling laws (linear, state-dependent, MyoFullBody adapter) and regional fraction (SPN-2a)
- **Next step:** Shipped in PR #12796.

### DL-#12945 · Explorer: OpenSim Plot Resolves Markers From the Model (No KeyError 'Hand'), MuJoCo Export Video Uses the Viewport Camera, App Exports Write H.264 (#12946, #12947)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12945
- **Branch:** merged via #12950
- **PR:** #12950
- **Paths:** see #12950
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`2be01520`; collated from changes/12945-explorer-opensim-plot-resolves-markers-f.md)
- **Summary:** Explorer: OpenSim plot resolves markers from the model (no KeyError 'Hand'), MuJoCo Export Video uses the viewport camera, app exports write H.264 (#12946, #12947)
- **Next step:** Shipped in PR #12950.

### DL-#12718 · NSX-89: Simscape Closed-Loop Derivative Capability NONE With UNAVAILABLE Reason, R2025b-Only Linearization Payload Gate, DV-12718 Deferred Record

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12718
- **Branch:** merged via #12880
- **PR:** #12880
- **Paths:** see #12880
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`9455bfa1`; collated from changes/12718-nsx-89-simscape-closed-loop-derivative-c.md)
- **Summary:** NSX-89: Simscape closed-loop derivative capability NONE with UNAVAILABLE reason, R2025b-only linearization payload gate, DV-12718 deferred record
- **Next step:** Shipped in PR #12880.

### DL-#12714 · NSX-85: Pinocchio Analytic Closed-Loop Derivatives, FULL Reference for CLOSED_LOOP_DERIVATIVES

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12714
- **Branch:** merged via #12853
- **PR:** #12853
- **Paths:** see #12853
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`cd1a8e6c`; collated from changes/12714-nsx-85-pinocchio-analytic-closed-loop-de.md)
- **Summary:** NSX-85: Pinocchio analytic closed-loop derivatives, FULL reference for CLOSED_LOOP_DERIVATIVES
- **Next step:** Shipped in PR #12853.

### DL-#12719 · NSX-90 Covariance Propagation, Adjoint Sensitivity Kernels and Nonlinearity Gate

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12719
- **Branch:** merged via #12881
- **PR:** #12881
- **Paths:** see #12881
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`b6600a30`; collated from changes/12719-nsx-90-covariance-propagation-adjoint-se.md)
- **Summary:** NSX-90 covariance propagation, adjoint sensitivity kernels and nonlinearity gate
- **Next step:** Shipped in PR #12881.

### DL-#12586 · Kinematic Ensemble (Admitted Samples, Rejection Accounting, Envelope Union/Intersection) and Reproject_Onto_Closure Delegating to Project_to_Closure (NSX-61)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12586
- **Branch:** merged via #12645
- **PR:** #12645
- **Paths:** see #12645
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`a934b4aa`; collated from changes/12586-kinematic-ensemble-admitted-samples-reje.md)
- **Summary:** Kinematic ensemble (admitted samples, rejection accounting, envelope union/intersection) and reproject_onto_closure delegating to project_to_closure (NSX-61)
- **Next step:** Shipped in PR #12645.

### DL-#12914 · Closing the Launcher After a MuJoCo Session Releases Renderers Before Widgets Die.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12914
- **Branch:** merged via #12922
- **PR:** #12922
- **Paths:** see #12922
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`bd1fa54e`; collated from changes/12914-closing-the-launcher-after-a-mujoco-sess.md)
- **Summary:** Closing the launcher after a MuJoCo session releases renderers before widgets die.
- **Next step:** Shipped in PR #12922.

### DL-#12884 · MuJoCo Explorer: 3D View No Longer Renders Black in the Launcher (Renderer Freed Under Its Own GL Context), Side Quick-Camera No Longer Crashes, Visualization Tab Scrolls Instead of Overlapping (#12884, #12885, #12889)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12884
- **Branch:** merged via #12902
- **PR:** #12902
- **Paths:** see #12902
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`a4333859`; collated from changes/12884-mujoco-explorer-3d-view-no-longer-render.md)
- **Summary:** MuJoCo explorer: 3D view no longer renders black in the launcher (renderer freed under its own GL context), Side quick-camera no longer crashes, Visualization tab scrolls instead of overlapping (#12884, #12885, #12889)
- **Next step:** Shipped in PR #12902.

### DL-#12717 · NSX-88: OpenSim Closed-Loop FD Derivatives via NSX-84 Routine, Capability PARTIAL, Muscle-Insufficiency Flag

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12717
- **Branch:** merged via #12852
- **PR:** #12852
- **Paths:** see #12852
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`43baeba2`; collated from changes/12717-nsx-88-opensim-closed-loop-fd-derivative.md)
- **Summary:** NSX-88: OpenSim closed-loop FD derivatives via NSX-84 routine, capability PARTIAL, muscle-insufficiency flag
- **Next step:** Shipped in PR #12852.

### DL-#12716 · NSX-87: MuJoCo/MyoSuite Hard-KKT Finite-Difference Closed-Loop Derivatives (PARTIAL), Reusing the NSX-84 Routine

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12716
- **Branch:** merged via #12852
- **PR:** #12852
- **Paths:** see #12852
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`43baeba2`; collated from changes/12716-nsx-87-mujoco-myosuite-hard-kkt-finite-d.md)
- **Summary:** NSX-87: MuJoCo/MyoSuite hard-KKT finite-difference closed-loop derivatives (PARTIAL), reusing the NSX-84 routine
- **Next step:** Shipped in PR #12852.

### DL-#12565 · NSX-22 Static Torque and Hand-Split Envelopes With Chebyshev Margin and Sign Classes

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12565
- **Branch:** merged via #12640
- **PR:** #12640
- **Paths:** see #12640
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`66c8da74`; collated from changes/12565-nsx-22-static-torque-and-hand-split-enve.md)
- **Summary:** NSX-22 static torque and hand-split envelopes with Chebyshev margin and sign classes
- **Next step:** Shipped in PR #12640.

### DL-#12913 · Club Head Speed No Longer Reads 0.0 for Models Naming the Body Clubhead.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12913
- **Branch:** merged via #12921
- **PR:** #12921
- **Paths:** see #12921
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`1e013045`; collated from changes/12913-club-head-speed-no-longer-reads-0-0-for.md)
- **Summary:** Club Head Speed no longer reads 0.0 for models naming the body clubhead.
- **Next step:** Shipped in PR #12921.

### DL-#12886 · Launcher Survives Slot Errors, Accurate Engine Status/Hints, Shutdown and First-Click Launch Fixes (#12890 #12891 #12892 #12894)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12886
- **Branch:** merged via #12904
- **PR:** #12904
- **Paths:** see #12904
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`fc12df4c`; collated from changes/12886-launcher-survives-slot-errors-accurate-e.md)
- **Summary:** Launcher survives slot errors, accurate engine status/hints, shutdown and first-click launch fixes (#12890 #12891 #12892 #12894)
- **Next step:** Shipped in PR #12904.

### DL-#12878 · Offscreen Per-Engine Explorer Contract Tests; Explorer Opens MJCF, Keeps Every MJCF DOF, Loads Results, Scrubs and Exports MP4 via One Shared Frame Source; OSIM Loader Accepts Class::Nested Tags; Results Browser Export Button Follows Playability; MuJoCo GUI Gains Export Video (#12888) and Plots No Longer Crash on Unrecorded Channels (#12887)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12878
- **Branch:** merged via #12899
- **PR:** #12899, #13028
- **Paths:** see #12899
- **Started:** 2026-10-16
- **Last verified:** 2026-10-17 (`47143bbc`; collated from changes/12878-docker-explorer.md)
- **Summary:** Offscreen per-engine explorer contract tests; explorer opens MJCF, keeps every MJCF DOF, loads results, scrubs and exports MP4 via one shared frame source; OSIM loader accepts Class::Nested tags; results browser Export button follows playability; MuJoCo GUI gains Export Video (#12888) and plots no longer crash on unrecorded channels (#12887)
- **Next step:** Shipped in PR #12899.

### DL-#12846 · Refactor Chat_Service Stream Worker, Motion_Pipeline API Routes and Model-Pack Manifest Parsing Under the Architecture Budget; Remove Six Expiring Exceptions

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12846
- **Branch:** merged via #12869
- **PR:** #12869
- **Paths:** see #12869
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`3c347de9`; collated from changes/12846-refactor-chat-service-stream-worker-moti.md)
- **Summary:** Refactor chat_service stream worker, motion_pipeline API routes and model-pack manifest parsing under the architecture budget; remove six expiring exceptions
- **Next step:** Shipped in PR #12869.

### DL-#12684 · R4: Address Knee Flexion (Owner 15 +-5 Deg Prior, Reliable Capture Wins, Hard No-Hyperextension Bound, Per-Knee Target_Source in the Receipt; --Knee-Flexion Defaults to Capture)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12684
- **Branch:** merged via #12788
- **PR:** #12788, #13527
- **Paths:** see #12788
- **Started:** 2026-10-16
- **Last verified:** 2026-10-20 (`19f8fef3`; collated from changes/12684-r4-the-dynamics-receipt-records-ik-and-f.md)
- **Summary:** R4: address knee flexion (owner 15 +-5 deg prior, reliable capture wins, hard no-hyperextension bound, per-knee target_source in the receipt; --knee-flexion defaults to capture)
- **Next step:** Shipped in PR #12788.

### DL-#12688 · Drake and Pinocchio MeshCat Backends Draw the Shared Model_Appearance Spec Through One Engine-Neutral Scene Builder; Local Grey Palettes Removed (R9a)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12688
- **Branch:** merged via #12908
- **PR:** #12908
- **Paths:** see #12908
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`29cacd85`; collated from changes/12688-drake-and-pinocchio-meshcat-backends-dra.md)
- **Summary:** Drake and Pinocchio MeshCat backends draw the shared model_appearance spec through one engine-neutral scene builder; local grey palettes removed (R9a)
- **Next step:** Shipped in PR #12908.

### DL-#12711 · Add DisturbanceSpec Timestep-Consistent Disturbance Contract, Wrapped OutcomeDispersion Statistics, and NaN (Not 0) MetricStatistics.Cv at Near-Zero Mean (NSX-82)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12711
- **Branch:** merged via #12799
- **PR:** #12799
- **Paths:** see #12799
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`d166ace5`; collated from changes/12711-add-disturbancespec-timestep-consistent.md)
- **Summary:** Add DisturbanceSpec timestep-consistent disturbance contract, wrapped OutcomeDispersion statistics, and NaN (not 0) MetricStatistics.cv at near-zero mean (NSX-82)
- **Next step:** Shipped in PR #12799.

### DL-#12690 · R9c: GS3DX Simscape Golfer Takes Its Colours and Trunk Size From the Shared Model_Appearance Spec via a Versioned JSON Exporter (Fail-Closed MATLAB Reader)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12690
- **Branch:** merged via #12909
- **PR:** #12909
- **Paths:** see #12909
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`cf095b75`; collated from changes/12690-r9c-gs3dx-simscape-golfer-takes-its-colo.md)
- **Summary:** R9c: GS3DX Simscape golfer takes its colours and trunk size from the shared model_appearance spec via a versioned JSON exporter (fail-closed MATLAB reader)
- **Next step:** Shipped in PR #12909.

### DL-#12687 · R7: Realistic CC0 Head for the Shared Parametric Head Slot - Head.Model Cc0_Mesh Loads a MakeHuman-Derived CC0 Skin and Short-Hair Mesh (Assets/Head_Cc0, Provenance + Sha256 Verified, Fail Closed) Through Build_Head_Parts, so MuJoCo and the OBJ Export (R9a/B/C) Get It With No Per-Engine Code; Default Stays the Procedural Head

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12687
- **Branch:** merged via #12907
- **PR:** #12907
- **Paths:** see #12907
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`c5487058`; collated from changes/12687-r7-realistic-cc0-head-for-the-shared-par.md)
- **Summary:** R7: realistic CC0 head for the shared parametric head slot - head.model cc0_mesh loads a MakeHuman-derived CC0 skin and short-hair mesh (assets/head_cc0, provenance + sha256 verified, fail closed) through build_head_parts, so MuJoCo and the OBJ export (R9a/b/c) get it with no per-engine code; default stays the procedural head
- **Next step:** Shipped in PR #12907.

### DL-#12883 · OpenSim Loads Alongside Ezc3d: C3D Stack Imported Lazily at Startup, Structured OpenSim Probe Reports the Real Import Failure Instead of Not Installed

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12883
- **Branch:** merged via #12903
- **PR:** #12903
- **Paths:** see #12903
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`155bcc47`; collated from changes/12883-opensim-loads-alongside-ezc3d-c3d-stack.md)
- **Summary:** OpenSim loads alongside ezc3d: C3D stack imported lazily at startup, structured OpenSim probe reports the real import failure instead of Not Installed
- **Next step:** Shipped in PR #12903.

### DL-#12882 · Fix Classic Launcher Startup: Parent Contract Guard, ChatService ToolRegistry Identity, Data_Explorer Launcher_Embed Alias (#12893, #12895)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12882
- **Branch:** merged via #12900
- **PR:** #12900
- **Paths:** see #12900
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`d03af3a0`; collated from changes/12882-fix-classic-launcher-startup-parent-cont.md)
- **Summary:** Fix classic launcher startup: parent contract guard, ChatService ToolRegistry identity, data_explorer launcher_embed alias (#12893, #12895)
- **Next step:** Shipped in PR #12900.

### DL-#12817 · NSX-84B: Tangent State Space for Multi-DOF Closed Loops (K > 1), Five-Bar Golden and Gauge-Law Tests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12817
- **Branch:** merged via #12876
- **PR:** #12876
- **Paths:** see #12876
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`a26b27f9`; collated from changes/12817-nsx-84b-tangent-state-space-for-multi-do.md)
- **Summary:** NSX-84b: tangent state space for multi-DOF closed loops (k > 1), five-bar golden and gauge-law tests
- **Next step:** Shipped in PR #12876.

### DL-#12713 · NSX-84: Constrained Tangent Linearization (M2 Preload) and Closed-Loop Derivative Capability Seam With Engine-Free FD Reference

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12713
- **Branch:** merged via #12876
- **PR:** #12876
- **Paths:** see #12876
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`a26b27f9`; collated from changes/12713-nsx-84-constrained-tangent-linearization.md)
- **Summary:** NSX-84: constrained tangent linearization (M2 preload) and closed-loop derivative capability seam with engine-free FD reference
- **Next step:** Shipped in PR #12876.

### DL-#12671 · Make the Vendored Tools Double_Pendulum_Golf the Canonical Golfer Kinematics and Closure Jacobian; Golfer_Kinematics/Golfer_Constraints Become Thin Adapters Pinned by a CI Parity Test (ADR-0056)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12671
- **Branch:** merged via #12801
- **PR:** #12801
- **Paths:** see #12801
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`ef956604`; collated from changes/12671-make-the-vendored-tools-double-pendulum.md)
- **Summary:** Make the vendored Tools double_pendulum_golf the canonical golfer kinematics and closure Jacobian; golfer_kinematics/golfer_constraints become thin adapters pinned by a CI parity test (ADR-0056)
- **Next step:** Shipped in PR #12801.

### DL-#12685 · R8: One Shared Appearance and Anthropometry Spec (Model_Appearance.Body_Meshes) - Engine-Neutral Skin/Garment/Shoe Meshes in the Spec Body Frame, Trunk Sized From Spec Visual_Hints as One Solid, Default Charcoal Trousers Hip to Ankle, One Plain Palette, Shared OBJ Export; MuJoCo Is Now a Thin Translator

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12685
- **Branch:** merged via #12790
- **PR:** #12790
- **Paths:** see #12790
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`24fc23a8`; collated from changes/12685-r8-one-shared-appearance-and-anthropomet.md)
- **Summary:** R8: one shared appearance and anthropometry spec (model_appearance.body_meshes) - engine-neutral skin/garment/shoe meshes in the spec body frame, trunk sized from spec visual_hints as one solid, default charcoal trousers hip to ankle, one plain palette, shared OBJ export; MuJoCo is now a thin translator
- **Next step:** Shipped in PR #12790.

### DL-#12866 · Generate the User Manual TOC From Real Chapter Headings and Test That Every Manual Link Resolves

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12866
- **Branch:** merged via #12877
- **PR:** #12877
- **Paths:** see #12877
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`94fb252b`; collated from changes/12866-generate-the-user-manual-toc-from-real-c.md)
- **Summary:** Generate the user manual TOC from real chapter headings and test that every manual link resolves
- **Next step:** Shipped in PR #12877.

### DL-#12833 · Divergence Inventory: Commit Only Stable Per-Path Facts (V2 JSONL, One Sorted Record per Line), Drop the Committed Markdown, Move Totals/Bytes/Authorship to --Report; Different-Path PRs No Longer Conflict

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12833
- **Branch:** merged via #12835
- **PR:** #12835
- **Paths:** see #12835
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`aeb6b9a0`; collated from changes/12833-divergence-inventory-commit-only-stable.md)
- **Summary:** Divergence inventory: commit only stable per-path facts (v2 JSONL, one sorted record per line), drop the committed markdown, move totals/bytes/authorship to --report; different-path PRs no longer conflict
- **Next step:** Shipped in PR #12835.

### DL-#12850 · Split the User Manual and Functional Test Plan Into an Index Plus Chapter Files Under the Doc Size Budget; Delete Both Expiring Exceptions.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12850
- **Branch:** merged via #12865
- **PR:** #12865
- **Paths:** see #12865
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`227e255c`; collated from changes/12850-split-the-user-manual-and-functional-tes.md)
- **Summary:** Split the user manual and functional test plan into an index plus chapter files under the doc size budget; delete both expiring exceptions.
- **Next step:** Shipped in PR #12865.

### DL-#12848 · Split Generated ORG-01 Capability Baseline Into Index Plus Linked Parts; Drop Three Doc-Size Exceptions

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12848
- **Branch:** merged via #12854
- **PR:** #12854
- **Paths:** see #12854
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`fb3fbfb3`; collated from changes/12848-split-generated-org-01-capability-baseli.md)
- **Summary:** Split generated ORG-01 capability baseline into index plus linked parts; drop three doc-size exceptions
- **Next step:** Shipped in PR #12854.

### DL-#12847 · Split Scripts/_urdf_campaign_issues.py Into a Domain Package and Drop the Stale/Expiring File-Size Exceptions; Research Scripts Left Pinned.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12847
- **Branch:** merged via #12863
- **PR:** #12863
- **Paths:** see #12863
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`73527918`; collated from changes/12847-split-scripts-urdf-campaign-issues-py-in.md)
- **Summary:** Split scripts/_urdf_campaign_issues.py into a domain package and drop the stale/expiring file-size exceptions; research scripts left pinned.
- **Next step:** Shipped in PR #12863.

### DL-#12841 · Refactor Model_Converter Exporters and Schema Validator Under the Architecture Budget; Delete 5 Expiring Exceptions; Add Golden-Output Tests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12841
- **Branch:** merged via #12857
- **PR:** #12857
- **Paths:** see #12857
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`cd3227b3`; collated from changes/12841-refactor-model-converter-exporters-and-s.md)
- **Summary:** Refactor model_converter exporters and schema validator under the architecture budget; delete 5 expiring exceptions; add golden-output tests
- **Next step:** Shipped in PR #12857.

### DL-#12670 · Golfer Generator Adapter Derives Coordinate and Control Widths From the TB-00 Registry (U_Dim=6) Instead of a Literal 5, and Rejects a Sealed U_Dim That Disagrees; Sealed NM-01/NM-09 Records Untouched Pending Owner Decision

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12670
- **Branch:** merged via #12834
- **PR:** #12834, #13440
- **Paths:** see #12834
- **Started:** 2026-10-16
- **Last verified:** 2026-10-19 (`42f2f673`; collated from changes/12670-add-dated-nm-addendum-superseding-sealed.md)
- **Summary:** Golfer generator adapter derives coordinate and control widths from the TB-00 registry (u_dim=6) instead of a literal 5, and rejects a sealed u_dim that disagrees; sealed NM-01/NM-09 records untouched pending owner decision
- **Next step:** Shipped in PR #12834.

### DL-#12829 · Perturbation Panel Renders the Worker's PerturbationSummary on Batch Finish and Shows an Undefined CV as N/A

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12829
- **Branch:** merged via #12837
- **PR:** #12837
- **Paths:** see #12837
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`0267bb78`; collated from changes/12829-perturbation-panel-renders-the-worker-s.md)
- **Summary:** Perturbation panel renders the worker's PerturbationSummary on batch finish and shows an undefined CV as n/a
- **Next step:** Shipped in PR #12837.

### DL-#12820 · Receipt Base_Spec_Sha256 Hashes the Committed Spec Before Foot-Progression Seeding; Foot_Progression_Mode and Foot_Seed_Deg Recorded as Explicit Receipt Fields so Style-Feet Canonical Receipts Chain to the Committed Spec

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12820
- **Branch:** merged via #12830
- **PR:** #12830
- **Paths:** see #12830
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`2247317e`; collated from changes/12820-receipt-base-spec-sha256-hashes-the-comm.md)
- **Summary:** Receipt base_spec_sha256 hashes the committed spec before foot-progression seeding; foot_progression_mode and foot_seed_deg recorded as explicit receipt fields so style-feet canonical receipts chain to the committed spec
- **Next step:** Shipped in PR #12830.

### DL-#12570 · Constrained ZTCF/ZVCF via the KKT Projector, ConstrainedEquationsOfMotionProvider; Counterfactual Rollouts Integrate Real Dynamics (m_Eff Mock Removed)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12570
- **Branch:** merged via #12641
- **PR:** #12641
- **Paths:** see #12641
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`8e7782a4`; collated from changes/12570-constrained-ztcf-zvcf-via-the-kkt-projec.md)
- **Summary:** Constrained ZTCF/ZVCF via the KKT projector, ConstrainedEquationsOfMotionProvider; counterfactual rollouts integrate real dynamics (m_eff mock removed)
- **Next step:** Shipped in PR #12641.

### DL-#12712 · NSX-83: Event-Triggered Impact Outcome Map With Saltation Correction, Face-Angle Availability and Tools Ball-Flight Bridge

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12712
- **Branch:** merged via #12800
- **PR:** #12800
- **Paths:** see #12800
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`4c3a8064`; collated from changes/12712-nsx-83-event-triggered-impact-outcome-ma.md)
- **Summary:** NSX-83: event-triggered impact outcome map with saltation correction, face-angle availability and Tools ball-flight bridge
- **Next step:** Shipped in PR #12800.

### DL-#12583 · NSX-55: Simscape Closed-Loop Export Format, Loader and R2025b Exporter (Planar Rig Validated; Golf Model Deferred as DV-12583)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12583
- **Branch:** merged via #12644
- **PR:** #12644
- **Paths:** see #12644
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`efe8489a`; collated from changes/12583-nsx-55-simscape-closed-loop-export-forma.md)
- **Summary:** NSX-55: Simscape closed-loop export format, loader and R2025b exporter (planar rig validated; golf model deferred as DV-12583)
- **Next step:** Shipped in PR #12644.

### DL-#12811 · Fix Main Breakage From #12704: Pinocchio JointModelRZ Test Lane and Pinocchio_Reference Row Typing

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12811
- **Branch:** merged via #12812
- **PR:** #12812, #12796
- **Paths:** see #12812
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12811-fix-main-breakage-from-12704-pinocchio-j.md)
- **Summary:** Fix main breakage from #12704: pinocchio JointModelRZ test lane and pinocchio_reference row typing
- **Next step:** Shipped in PR #12812.

### DL-#12808 · Run the Prettier Pre-Commit Hook at Commit Time Only so Rebased Branches Can Push

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12808
- **Branch:** merged via #12809
- **PR:** #12809, #12796
- **Paths:** see #12809
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12808-run-the-prettier-pre-commit-hook-at-comm.md)
- **Summary:** Run the Prettier pre-commit hook at commit time only so rebased branches can push
- **Next step:** Shipped in PR #12809.

### DL-#12709 · NSX-80: Variability-Sensitivity Specification, Symbolic Goldens, Hypotheses and Unverified Citations

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12709
- **Branch:** merged via #12780
- **PR:** #12780, #12800
- **Paths:** see #12780
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`4c3a8064`; collated from changes/12709-nsx-80-variability-sensitivity-specifica.md)
- **Summary:** NSX-80: variability-sensitivity specification, symbolic goldens, hypotheses and unverified citations
- **Next step:** Shipped in PR #12780.

### DL-#12591 · NSX-73 Research Report Scoping a Muscle-Redundancy Layer Above NSX Envelopes, With Elbow Null-Space Example Test

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12591
- **Branch:** merged via #12663
- **PR:** #12663, #12796
- **Paths:** see #12663
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12591-nsx-73-research-report-scoping-a-muscle.md)
- **Summary:** NSX-73 research report scoping a muscle-redundancy layer above NSX envelopes, with elbow null-space example test
- **Next step:** Shipped in PR #12663.

### DL-#12564 · NSX-21: FeasibleSet With Declared Torque, Friction-Cone and Grip Constraints in Null-Space Coordinates

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12564
- **Branch:** merged via #12637
- **PR:** #12637, #12796
- **Paths:** see #12637
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`983e46ab`; collated from changes/12564-nsx-21-feasibleset-with-declared-torque.md)
- **Summary:** NSX-21: FeasibleSet with declared torque, friction-cone and grip constraints in null-space coordinates
- **Next step:** Shipped in PR #12637.

### DL-#12791 · Fix garment_classification.tex Row Terminators so the Golfer Appearance Reference Compiles; Latex-References CI Now Builds the Full-Body Model Reference Masters

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12791
- **Branch:** merged via #12798
- **PR:** #12798
- **Paths:** see #12798
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`59556187`; collated from changes/12791-fix-garment-classification-tex-row-termi.md)
- **Summary:** Fix garment_classification.tex row terminators so the golfer appearance reference compiles; latex-references CI now builds the full-body model reference masters
- **Next step:** Shipped in PR #12798.

### DL-#12802 · Split Full-Body HANDOFF and Anthropometry REVIEW Under the Doc-Size Budget; Renew Only the Stale Chat Child-Copy Architecture Exception for 14 Days (#12803)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12802
- **Branch:** merged via #12804
- **PR:** #12804
- **Paths:** see #12804
- **Started:** 2026-10-16
- **Last verified:** 2026-10-16 (`3fb64eff`; collated from changes/12802-split-full-body-handoff-and-anthropometr.md)
- **Summary:** Split full-body HANDOFF and anthropometry REVIEW under the doc-size budget; renew only the stale chat child-copy architecture exception for 14 days (#12803)
- **Next step:** Shipped in PR #12804.

### DL-#12792 · Bump Vendor/Ud-Tools to Tools E3cf5a43, Carrying the Corrected Golfer Club Jacobians (Tools#5627 Rust Pendulum-Core, Tools#5628 Python/JAX Forward Kinematics Y-Sign, Tools#5626); Pins and Divergence Inventory Kept in Step

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12792
- **Branch:** merged via #12792
- **PR:** #12792
- **Paths:** see #12792
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`513cfff7`; collated from changes/12792-bump-vendor-ud-tools-to-tools-e3cf5a43-c.md)
- **Summary:** Bump vendor/ud-tools to Tools e3cf5a43, carrying the corrected golfer club Jacobians (Tools#5627 Rust pendulum-core, Tools#5628 Python/JAX forward kinematics y-sign, Tools#5626); pins and divergence inventory kept in step
- **Next step:** Shipped in PR #12792.

### DL-#12777 · Re-Solve the MSK Address Calibration on the Widened Forearm Range so the Lead Pronation Leaves Its Old 90 Deg Bound

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12777
- **Branch:** merged via #12782
- **PR:** #12782
- **Paths:** see #12782
- **Started:** 2026-10-15
- **Last verified:** 2026-10-15 (`4f24ae09`; collated from changes/12777-re-solve-the-msk-address-calibration-on.md)
- **Summary:** Re-solve the MSK address calibration on the widened forearm range so the lead pronation leaves its old 90 deg bound
- **Next step:** Shipped in PR #12782.

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
- **PR:** #12488, #12534, #13063
- **Paths:** see #12488
- **Started:** 2026-10-14
- **Last verified:** 2026-10-17 (`3b1ef961`; collated from changes/12347-prettier-format-the-back-squat-barbell-h.md)
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
- **PR:** #12345, #12189, #13047
- **Paths:** see #12345
- **Started:** 2026-10-14
- **Last verified:** 2026-10-18 (`7fed54d1`; collated from changes/11744-lift-4-the-barbell-hold-starts-in-contac.md)
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

### DL-#11659 · Typed Attachment Ports, Part Catalog and Drag-and-Drop Assembly Canvas in the Frankenstein Editor (CMB-8, CMB-9)

- **State:** shipped
- **Shipped:** 2026-10-07
- **Owner:** unassigned
- **Issue:** #11659
- **Branch:** feat/frankenstein-ports-dnd-cmb-11659
- **PR:** #11702
- **Paths:** see #11702
- **Started:** 2026-10-14
- **Last verified:** 2026-10-14 (`5b607519`; collated from changes/11659-typed-attachment-ports-part-catalog-and.md)
- **Summary:** Typed attachment ports, part catalog and drag-and-drop assembly canvas in the Frankenstein editor (CMB-8, CMB-9)

### DL-#10381 · Reconciled Scopes From Duplicate Source Rows

- **State:** shipped
- **Shipped:** 2026-09-20
- **Owner:** claude (PR #10599; branch feat/10381-crocoddyl-pelvis-yaw-g1) | claude (PR not recorded in source branch evidence/10381-g1-b100)
- **Issue:** #10381
- **PR:** #10599
- **Branch:** feat/10381-crocoddyl-pelvis-yaw-g1; evidence/10381-g1-b100
- **Paths:** src/engines/physics_engines/pinocchio/python/crocoddyl_problem.py; src/engines/physics_engines/pinocchio/python/crocoddyl_action.py; src/engines/physics_engines/pinocchio/python/full_body_fit.py; tests/unit/motion_matching/test_crocoddyl_pelvis_yaw.py; SPEC.md; docs/development/DEVELOPMENT_LOG.md; src/shared/python/motion_matching/ledger.py; evidence/matched; docs/development/full_body_models/evidence/acceptance/verdicts_2026-09.json; docs/development/matched_swing_program
- **Started:** 2026-09-18
- **Summary:** Added analytic pelvis-yaw cost/gradient to Crocoddyl full-body fitting, CLI and receipts. The recorded candidate remains G1 REJECTED (early/terminal error, yaw, penetration and weight-fraction limits); no physical qualification.
- **Evidence:** tests/unit/motion_matching/test_crocoddyl_pelvis_yaw.py.
- **Last verified:** 2026-10-11 (`944d798e`; collated from docs/development/archive/DEVELOPMENT_LOG_RECONCILIATION_SOURCE_20261011.md)

### DL-#11285 · Force and Torque Overlays for Every Engine

- **Started:** not recorded in source
- **State:** shipped
- **Shipped:** 2026-10-03
- **Owner:** antigravity
- **Issue:** #11285; children #11286–#11315
- **PR:** #11397
- **Branch:** `feat/fto-30-gallery-docs-11315`
- **Paths:** `scripts/render_force_overlay_gallery.py`, `tests/visual/force_overlay/**`, `docs/user_guide/force_overlay.md`, `src/config/feature_parity.json`
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** One engine-agnostic force/torque contract, glyph builder and renderer adapters; real providers for MuJoCo, Drake, Pinocchio, OpenSim and Simscape; tension/compression producers for every engine; arrows, shaded model and legend over source footage. [Plan](force_torque_overlay_epic.md).

### DL-#11532 · MOSAIC Model-Aware Multi-Trial Matching Kernels, Methods Reference and DIME (#11421) Audit

- **State:** shipped
- **Shipped:** 2026-10-05
- **Owner:** unassigned
- **Issue:** #11532
- **Branch:** feat/mosaic-model-aware-matching
- **PR:** #11555
- **Paths:** see #11555
- **Started:** 2026-10-05
- **Last verified:** 2026-10-05 (`90df1e03`; collated from changes/11532-mosaic-model-aware-multi-trial-matching.md)
- **Summary:** MOSAIC model-aware multi-trial matching kernels, methods reference and DIME (#11421) audit

### DL-#11400 · OpenCap to OpenSim Integration

- **State:** shipped
- **Shipped:** 2026-10-04
- **Owner:** antigravity
- **Issue:** #11400; child #11408 (OpenCap: golf accuracy qualification against a marker reference; follows #11401–#11403, #11406, #11407, #11409, #11405)
- **PR:** #11451
- **Branch:** `docs/opencap-golf-accuracy-deferred-11408`
- **Paths:** `docs/development/planning/catalog.json`, `docs/development/planning/DV-11408.md`, `docs/development/planning/source-11408.json`
- **Started:** 2026-10-03
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** OpenCap to OpenSim integration complete across sidecar launcher, session import, marker augmentation, UI, and hosted download; golf accuracy qualification against physical marker reference recorded in deferred-validation catalog (DV-11408).

### DL-#11329 · Scapula and Quiet Torso Matching With Neutral 1080P Previews

- **State:** shipped
- **Shipped:** 2026-10-03
- **Owner:** codex (Gemini 3.8 Flash CLI reviews)
- **Issue:** #11329
- **PR:** #11351; depends on draft #11256 and shared-capture #11172.
- **Branch:** `feat/simscape-scapula-protraction-20261002`
- **Paths:** `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/tools/gs3dx_whole_body_ik.m`, `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/tools/gs3dx_spine_bounds.m`, `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/tools/gs3dx_render.m`, `docs/research/simscape_matching_reference/simscape_matching_reference.tex`
- **Started:** 2026-10-02
- **Last verified:** 2026-10-03: 95 native R2025b contracts/regression tests pass with no failures/skips and natural exit 0. Final independent native replay checks all 1,313 sampled poses across 26 complete trajectories; final replay/export batches exit naturally with 0. All 104 neutral 1920x1080/30-fps videos across 13 moving-scapula models are delivered on Desktop, with both views and marker/clean modes. All 5,252 frames decoded; Desktop hashes and ZIP CRC/member hashes verified. Fixed-foot FullBody public-source fits reduce mean target RMS from 532/377 to 27/34 mm while preserving original offsets/model bytes; back RMS 52/61 mm remains a limitation. Final complete Human legacy/default parity passes. Source head bbe21592b6 preserves the qualified native tree after accepted-main sync; current documentation head requires CI verification. Built-in LaTeX compilation remains unavailable (platform directories missing).
- **Summary:** Scapula address/backswing bands, direct measured back-marker fitting, native spine excursion bands through top with soft continuation afterwards, two-coordinate-neck head-axis guide, complete measured-marker overlays, cyan back/waist highlights, neutral Model Swing labels and undistorted native 1080P pixels. Full 3D head yaw remains imperfect and is separately reported. Fixed dimensions/position offsets and private source identities are preserved. Independent dynamics remains outside this scope and unqualified.

### DL-#1900 · Merge-Queue Scoping for Core Test Lanes

- **State:** shipped
- **Shipped:** 2026-10-04
- **Owner:** claude
- **Issue:** D-sorganization/Repository_Management#1900
- **Branch:** `claude/1900-ud-merge-group-tests`
- **PR:** #11510
- **Paths:** `.github/workflows/ci-standard.yml`, `tests/ci/test_merge_group_test_lanes.py`, `tests/ci/test_ci_infrastructure.py`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** `merge_group` runs diff the core lane against `merge_group.base_sha` and fetch the default branch for the Tools child-copy guard, matching pull-request behaviour.

### DL-#11317 · Sanitized Historical Research Provider

- **State:** shipped
- **Shipped:** 2026-10-02
- **Owner:** codex
- **PR:** [#11240](https://github.com/D-sorganization/UpstreamDrift/pull/11240) (merged); Affine #4834 merged externally at 2026-10-03T02:38:02Z; actual consumer receipt remains 8fdebad
- **Issue:** #11317; consumer AffineDrift #4828; parent #11232 / #11235
- **Branch:** feat/necromatcher-native-fit-11235
- **Paths:** src/shared/python/workspace/historical_research.py, docs/api/contracts/historical-player-research-v1.schema.json, tests/unit/workspace/test_historical_research_provider.py, docs/development/necromatcher-research-package.md
- **Started:** 2026-10-02
- **Last verified:** 2026-10-02 — Latest Tiger V15/Hogan V12 sanitized cohort exported from committed/pushed/remote-verified provider `605c695259b13cb98a6d6e379e7f7fee51f27443`; manifest `d68830b14e2704466d4dd1795342dffd96f6f4d9b238b1fccd7f27117a9c10ed` (6,852 bytes), export receipt `65c4e775a501c0719e06a40aefd6b8be73a59a031298b96cf1e8a453a605debd`. Committed Affine consumer `8fdebad14db0f62b719942fa5d64f14878b54fd5` passed no-write inspect, exclusive draft installation, exact/idempotent recall, deterministic QMD and standalone Quarto --no-execute; admission receipt `184907af70181af6bd7e94cd5526139fdec828e57fe540c69f87c5c7a80b7d2b`, wrapper `d94e28062f090f00b45e09bb550d99903926129c9b29d70b7d702ec46057aba6`. All source/input/helper/config/registry brackets remained unchanged. Root separately browser-reviewed three natural 1234-by-712 views; browser receipt `046cb332516253fe0592be36a20ceb148cc7496ec1e299a7dbd1041d9e0f10d6` records bounded review with full-page/manual acceptance false. V15 adapter plus selection regressions:64 passed, scoped Ruff/format/pinned mypy passed; unchanged final consumer suite:6,685 passed/26 skipped/187 existing-marker deselected. The earlier V14/V12 bundle and its exact receipts remain preserved in procedure/turnover. Both cohorts remain rejected with unknown timing and no original-media import, optimizer run, public deployment or scientific promotion.
- **Summary:** Reuse canonical immutable library recall to cross-check external independent assessments, exact numerical producer and source/model/capture/fit hashes; rehash original downloaded footage without copying media. Export deterministic generic-player JSON under a separate rejected local research contract, preserving unknown physical timing and unauthorized distribution. No estimator, scheduler or qualified publication pathway is duplicated.

### DL-#11230 — Historical Player Capture

- **Started:** not recorded in source
- **State:** shipped
- **Shipped:** 2026-10-02
- **Owner:** codex
- **Issue:** #11230; epics #11226 and #11229
- **Branch:** `feat/historical-player-capture-11226`
- **PR:** [#11231](https://github.com/D-sorganization/UpstreamDrift/pull/11231)
- **Paths:** `src/shared/python/shadow_tracker/historical_capture.py`, `scripts/historical_capture.py`, `docs/development/historical-capture-procedure.md`
- **Last verified:** 2026-10-01 at ab2c869813a2b2be2f14b512ee34647694614377 — 364 Shadow Tracker tests passed; final Hogan 750/739 detected and Tiger 2000/1994 detected; observations only.
- **Summary:** Streaming source-bound detector observations; final reproducibility runs completed; dense review pending. Camera/time calibration, dense 3D fitting, native parity, dynamics, held-out evaluation and website integration remain open.

### DL-#11080 · Product Review for the Expert Panel

- **State:** shipped
- **Shipped:** 2026-09-29
- **Owner:** codex
- **Issue:** #11080 (owner-requested documentation review tracked by this PR)
- **PR:** #11080
- **Branch:** `docs/product-review-20260928`
- **Paths:** `docs/development/2026-09-28-product-review-board-proposals.md`
- **Started:** 2026-09-28
- **Last verified:** 2026-09-28 at 599cce5d5cb7d6972dfa2ca4e5770ea09cfaa309 — 108 focused tests and safe counterexample probes; Ruff lint/format and documentation hooks passed.
- **Summary:** Twelve prioritized proposals across UI/UX, error handling, performance, scientific validity and shared Tools wiring, with bounded evidence, acceptance criteria and current backlog reconciliation. Product code and qualification status are unchanged.

### DL-#11003 · Restate the #9243 BunkerShot Uncertainty Claims Under the Corrected F0 Model

- **State:** shipped
- **Shipped:** 2026-09-27
- **Owner:** claude
- **Issue:** #11003
- **PR:** #11010
- **Branch:** `claude/ud-9243-claims-restated`
- **Paths:** `tests/unit/tools/bunker_shot_gui/test_uncertainty_propagation_9243.py`
- **Started:** 2026-09-26
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Two #9243 claims went stale while the kernel-less gate skipped them. The window-band fixture now aims at the nominal shot's carry (0.690 m) instead of an unreachable 2 m and asserts a non-empty window; band (0.0, 1.141, 2.282). The mass interval's share is 0.676, so the test now says it dominates without swamping; `DOMINANCE_SHARE` stays 0.75.

### DL-#8907 · One User Config Root and One QSettings Namespace for the Launcher

- **State:** shipped
- **Shipped:** 2026-09-23
- **Owner:** claude
- **Issue:** #8907
- **Branch:** fix/8907-settings-root
- **PR:** #10742 (open)
- **Paths:** src/shared/python/data_io/user_config_root.py; src/launchers/launcher_settings_store.py; src/launchers/launcher_constants.py; tests/unit/data_io/test_user_config_root.py; tests/launchers/test_launcher_settings_store.py
- **Started:** 2026-09-22
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Launcher writers (preferences, recent models, library, onboarding, process/launcher logs, layout reset/diagnostics) resolve through `user_config_path()` under the platformdirs root with a one-time copy from the two legacy dot-dirs; QSettings consolidated on `(UpstreamDrift, Launcher)` with legacy aliasing; three secondary windows persist geometry.

### DL-#9225 · GolfSwingVisualizer MATLAB Duplication Consolidation

- **State:** abandoned
- **Owner:** local
- **Issue:** #9225 (source:assessment P2; DRY PP1)
- **Branch:** bot/issue-9225-golfviz-consolidation
- **PR:** #10715
- **Paths:** src/engines/Simscape_Multibody_Models/shared/+golfviz/GolfSwingVisualizer.m; src/engines/Simscape_Multibody_Models/2D_Golf_Model/matlab/2D GUI/launch_gui.m; src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/2D GUI/launch_gui.m; the two `2D GUI/main_scripts/golf_swing_analysis_gui.m` call sites; four deleted per-tree copies
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 at f33b40c9c (#10715) — MATLAB R2025b headless: package resolves to the single shared file from a bare `addpath`, class parses (75 methods), DbC precondition `GolfSwingVisualizer:InvalidInput` fires, and both launchers' relative path setups resolve the package with no shadow copy.
- **Summary:** Four near-identical `GolfSwingVisualizer.m` copies (1180–1183 lines each, 323 shared 8-line blocks between the worst pair) consolidated into one fleet-shared `+golfviz` package class; the two launchers add the shared directory to the MATLAB path and all four call sites use `golfviz.GolfSwingVisualizer`. Canonical behavior is the 2D variant superset (reproducible ground texture via `rng(1)` seeding that the 3D copies had silently lost); no genuine 2D-vs-3D behavioural divergence existed, so no parameter was needed.
- **Evidence:** MATLAB R2025b `-batch` verification transcript in the PR body; `git ls-files` shows one `GolfSwingVisualizer.m`.

### DL-#8941 · Simulation-Page Polling Complete (Server + Client)

- **State:** shipped
- **Shipped:** 2026-09-24
- **Owner:** local
- **Issue:** #8941 (server via #10736 merged; client polling via #10748 merged; ActuatorPanel and SimulationToolbar loops onto usePolling here; WS frames stay with #8936/#8940)
- **Branch:** fix/8941-use-polling-toolbar-actuator
- **PR:** #10881 (open)
- **Paths:** ui/src/components/simulation/ActuatorPanel.tsx; ui/src/components/simulation/ActuatorPanel.test.tsx; ui/src/components/simulation/SimulationToolbar.tsx; ui/src/components/simulation/SimulationToolbar.test.tsx
- **Started:** 2026-09-24
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Complete the client REST polling migration for #8941: both ActuatorPanel (1000 ms) and SimulationToolbar (1000 ms) now use the shared usePolling hook with visibility and simulation-running gating, eliminating background resource contention when tabs are hidden.

### DL-#8880 · GUI Thread-Blocking Simulation Migration to Async Action

- **State:** shipped
- **Shipped:** 2026-09-22
- **Owner:** local
- **Issue:** #8880
- **Branch:** fix/8880-gui-thread-blocking-sims
- **PR:** #10656
- **Paths:** src/tools/bunker_shot_gui/gui.py; src/tools/ball_flight_gui/gui.py; src/tools/swing_flight_pipeline/gui.py; src/tools/motion_matching/gui.py; src/shared/python/theme/tool_stylesheet.py; scripts/ci/check_gui_thread_blocking_ratchet.py; scripts/config/gui_thread_blocking_baseline.json; tests/tools/bunker_shot_gui/test_async_actions.py; tests/tools/ball_flight_gui/test_async_actions.py; tests/tools/swing_flight_pipeline/test_async_actions.py; tests/unit/scripts/test_gui_thread_blocking_ratchet.py
- **Started:** 2026-09-21
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Migrated `bunker_shot_gui`, `ball_flight_gui`, and `swing_flight_pipeline` onto `src/tools/async_action.py`; added lower-only GUI-thread-blocking ratchet (baseline 12); primary run buttons use shared theme wiring (#10654). ~9 tools remain un-migrated (see PR #10656 Deferred).

### DL-#10333 · Pinocchio MatchingPlant Full Lane (MS-14)

- **State:** shipped
- **Shipped:** 2026-09-22
- **Owner:** local
- **Issue:** #10333 (MS-14, epic #10363)
- **Branch:** feat/10333-pinocchio-matching-plant
- **PR:** #10684
- **Paths:** src/shared/python/motion*matching/pipeline/plants/pinocchio_plant.py; pinocchio_lane_receipts.py; receipt_components.py; cli.py; src/engines/physics_engines/pinocchio/python/full_body_fit.py; tests/unit/motion_matching/pipeline/test_pinocchio_plant.py; docs/development/full_body_models/evidence/ground_support/anthro_driver*{pinocchio,pink}/
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 at SELF — rematched onto origin/main after CO-06 #10687 merge (`f191dd09d`); regenerated matched_swing status README + divergence inventory; blocked native G1 receipts unchanged.
- **Summary:** Wire Pinocchio MatchingPlant derivatives + Pink `create_constrained_ik`, fitter `resolve_fit_native_plant` LoD bridge, ConstrainedIkReceipt `closure_residual_m` budget, and honest blocked evidence until <host> native run.
- **Evidence:** docs/development/full_body_models/evidence/ground_support/anthro_driver_pinocchio/; docs/development/full_body_models/evidence/ground_support/anthro_driver_pink/

### DL-#10345 · MyoSuite Kinematic Replay With Coordinate Map and Marker Parity Receipt

- **State:** shipped
- **Shipped:** 2026-09-21
- **Owner:** local
- **Issue:** #10345 (epic #10363, MS-52)
- **Branch:** fix/issue-10345-ms-52-local
- **PR:** #10666
- **Paths:** src/engines/physics_engines/myosuite/python/{retarget,replay,golfer_scene,coordinate_map_anthro.json,viz/render_replay.py}; tests/unit/engines/myosuite/test_retarget.py; tests/myosuite/test_replay_native.py; evidence/matched/driver_g1_myosuite/
- **Started:** 2026-09-21
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Added pure-numpy retarget map, kinematic replay CLI, marker parity receipt (`stage=replay`), MyoSuite registration in `cross_engine_replay.VALID_ENGINES` as kinematic-only, and viewer colour. Evidence committed under `evidence/matched/driver_g1_myosuite/`.

### DL-#9544 · Bunker Contact Regimes and Coupled Club Rotation Across Fidelity Tiers

- **State:** shipped
- **Shipped:** 2026-09-21
- **Owner:** claude
- **Issue:** #9544 (epic #9541)
- **Branch:** conductor/issue-9544
- **PR:** #10455 (open)
- **Paths:** src/bunkershot3d/ball/regimes.py; src/bunkershot3d/ball/pipeline.py; src/bunkershot3d/ball/splash.py; src/bunkershot3d/solvers/shot.py; src/bunkershot3d/solvers/mpm/wholeshot.py; src/bunkershot3d/vandv/conservation.py; src/tools/bunker_shot_gui/model.py; tests/bunkershot3d/ball/test_contact_regimes_9544.py; tests/bunkershot3d/solvers/test_rotation_coupling_9544.py; docs/bunkershot3d/contact-regimes.md
- **Started:** 2026-09-18
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Four contact regimes (no hit, direct/thin strike, splash, buried no-release) classified from the F0 sole path and divot stations; launch refused for every regime but splash. Prescribed rotation is a named `RotationMode`; an optional `RotationCoupling` boundary receives the sand wrench (about the body origin, world frame) and owns the angular velocity. V&V ledger adds support angular impulse and prescribed-driver work. F1 launch/out-of-plane refusals re-asserted; F2 requirements recorded in docs/bunkershot3d/contact-regimes.md.
- **Evidence:** tests/bunkershot3d/ball/test_contact_regimes_9544.py; tests/bunkershot3d/solvers/test_rotation_coupling_9544.py; docs/bunkershot3d/contact-regimes.md.

### DL-#9543 · Bunker Sand-to-Ball Transfer Calibration and Held-Out Qualification

- **State:** shipped
- **Shipped:** 2026-09-21
- **Owner:** claude
- **Issue:** #9543 (epic #9541)
- **Branch:** conductor/issue-9543
- **PR:** #10457 (open)
- **Paths:** src/bunkershot3d/ball/qualification.py; src/bunkershot3d/ball/qualification_fit.py; src/bunkershot3d/ball/rig_capability.py; src/bunkershot3d/ball/splash.py; src/bunkershot3d/ball/**init**.py; src/bunkershot3d/vandv/validation.py; src/bunkershot3d/vandv/measurement_intake.py; src/bunkershot3d/sand/provenance.py; tests/bunkershot3d/ball/test_transfer_qualification.py; docs/bunkershot3d/transfer-qualification.md
- **Started:** 2026-09-18
- **Last verified:** 2026-09-18 at SELF (base 49d94788a; 55 new tests pass; 1018 tests across tests/bunkershot3d ball, vandv, sand, study and public-API suites pass; scoped ruff, ruff format, mypy, LoD, file-size, architecture and error-handling ratchet checks pass; pinned Tools tree 62e8cdbf9c9f5f8a43a0342059f825e8fa78f8e1 materialised read-only for the run)
- **Summary:** Software intake/protocol and leakage controls support bounded fits, sensitivity checks and held-out V&V. No strokes are on file; physical qualification remains blocked and the issue stays open.
- **Evidence:** tests/bunkershot3d/ball/test_transfer_qualification.py; docs/bunkershot3d/transfer-qualification.md.

### DL-#9546 · Impact Zone Readiness Execution Index and I1/I2 Pin Consumption

- **Started:** not recorded in source
- **Owner:** not recorded in source
- **State:** shipped
- **Shipped:** 2026-09-22
- **Issue:** #9546 (children #9547, #9548, #9549, #9550; reused #9484, #9349)
- **Branch:** conductor/issue-9546
- **PR:** #10446
- **Paths:** src/config/impact_zone_readiness.json; src/config/industrial_readiness_loader.py; scripts/generate_industrial_readiness_index.py; docs/operations/impact-zone-readiness-index.md; tests/config/industrial_readiness/; tests/shared_contracts/test_impact_interval_provider.py
- **Last verified:** 2026-09-18 (SELF; ledger + freshness gates 45 passed; consumer contract 3 passed with `--tools-mode=vendored` against Tools pin `62e8cdbf9`)
- **Summary:** Epic #9546 gets the same machine-readable execution index as #9539: `impact_zone_readiness.json` reconciles I1–I4 and the two reused issues against `5347cba0f` and the vendored Tools pin, under the existing loader contract (keys `I-stage`/`R-stage` admitted, nothing else relaxed). The Tools fixes for I1 (#5088) and I2 (#5079) are consumed by a UD consumer contract that drives the audit probe through the vendored solver; I3, I4 and the Tools half of #9349 remain open with ordered plans, and `release_status` is `blocked`.
- **Evidence:** tests/config/industrial_readiness/; tests/shared_contracts/test_impact_interval_provider.py; docs/operations/impact-zone-readiness-index.md.

### DL-#9548 · Impact-Interval Energy Audit Consumer Gate

- **State:** shipped
- **Shipped:** 2026-09-21
- **Owner:** claude
- **Issue:** #9548 (parent #9546; provider Tools #4130 / #5079 / #5088)
- **Branch:** conductor/issue-9548
- **PR:** #10311
- **Paths:** src/shared/python/physics/impact_interval_audit.py; tests/unit/physics/test_impact_interval_audit.py; tests/shared_contracts/test_impact_interval_provider.py
- **Started:** 2026-09-17
- **Last verified:** 2026-09-17 (SELF; `tests/shared_contracts/` 33 passed, 0 skipped under `--tools-mode=vendored` at pin 1ac89c18e6280752d949e520c2143d2fb584d31e; 7 gate unit tests passed; pre-commit on changed files)
- **Summary:** The pinned Tools solver already integrates release, dashpot/friction, torsional damping and boundary storage independently of the residual (Tools #5079). UD consumes that pin through a fail-closed gate that recomputes the residual from the ledger identity, audits free vs supported momentum separately, reports evidence plus limitations as a JSON-ready record, and refuses `to_post_impact_state()` on unseparated contact or a failed audit. No UI consumer of the interval solver exists yet; the report record is the surface for one.
- **Evidence:** tests/shared_contracts/test_impact_interval_provider.py (interrupted compression 32.54 J stored / 0 J release / −0.063 J signed residual; clipping release 1.71 J; perturbed law residual > 0.5 J blocked; halving dt lowers both residuals).
  > > > > > > > origin/main

### DL-#10349 · Simscape 44-to-27 Coordinate Slice and Boundary-Load Validation

- **State:** shipped
- **Shipped:** 2026-09-21
- **Owner:** local
- **Issue:** #10349 (MS-62, epic #10363)
- **Branch:** fix/issue-10349-ms-62-local
- **PR:** #10665 (open)
- **Paths:** src/shared/python/motion_matching/coordinate_slice.py; tests/unit/motion_matching/test_coordinate_slice.py; evidence/matched/driver_g1_simscape_slice/; src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/align_measured_to_model.m
- **Started:** 2026-09-21
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Added JSON-backed 44-to-27 coordinate slice with name-based kinematic projection, virtual-work decomposition tests, boundary-wrench derivation for omitted neck/leg DOFs, CLI, and Simscape workspace overrides from anthropometric geometry documents in `align_measured_to_model.m`.
- **Evidence:** evidence/matched/driver_g1_simscape_slice/{slice_map.json,receipt.json,parity.json,run_manifest.json,candidate27.npz}; tests/unit/motion_matching/test_coordinate_slice.py.

### DL-#9422 · Rig Capture Sessions Through the Tools MocapSession Contract

- **State:** shipped
- **Shipped:** 2026-09-21
- **Owner:** claude
- **Issue:** #9422 (readiness P6; Tools #4706 M-track consumer)
- **Branch:** conductor/issue-9422
- **PR:** #10466 (open)
- **Paths:** src/motion_capture/rig/tools_bridge.py; src/motion_capture/rig/**main**.py; tests/motion_capture/rig/test_tools_session_export.py; tests/fixtures/mocap_session_export/; docs/motion_capture/capture_rig.md
- **Started:** 2026-09-19
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Rig capture export uses the pinned Tools MocapSessionManifest builders, consent policy and camera identity records. This is one consumer slice; D-track and prerequisite issues remain open.
- **Evidence:** tests/fixtures/mocap_session_export/export_checks.py; tests/motion_capture/rig/test_tools_session_export.py.

### DL-#10603 · Neural Motion Matching Plan

- **State:** shipped
- **Shipped:** 2026-09-21
- **Owner:** codex
- **Issue:** #10603
- **Branch:** docs/club-neural-matching-plans-20260920
- **PR:** #10628
- **Paths:** docs/plans/club_neural_review/; docs/plans/club_only_matching/; docs/plans/neural_motion_matching/
- **Started:** 2026-09-20
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Published bounded implementation issues with TDD/DbC/LoD/DRY prompts, dependency ordering, native validation gates and shared technical review. Planning artifacts do not qualify physical results or speedup.
- **Evidence:** docs/plans/club_neural_review/REVIEW.md; docs/plans/club_neural_review/excel_audit.json.

### DL-#9541 · BunkerShot3D Product Acceptance Matrix

- **State:** shipped
- **Shipped:** 2026-09-21
- **Owner:** claude
- **Issue:** #9541 (epic; children #9542–#9545, #9239, #9286, #8733, #8880, #9688–#9695)
- **Branch:** conductor/issue-9541
- **PR:** #10459 (open)
- **Paths:** src/config/bunkershot3d_qualification.json; tests/config/bunkershot3d_qualification/test_bunkershot3d_qualification_ledger.py; CLAUDE.md
- **Started:** 2026-09-18
- **Last verified:** 2026-09-18 at 9af408974 (SELF); 16 matrix tests plus the 21 sibling readiness-ledger tests pass on Python 3.12 with the Tools pin 62e8cdbf materialized; ruff check and format clean.
- **Summary:** Machine-readable product acceptance matrix for epic #9541, reusing the #9539 readiness-ledger loader unchanged: all sixteen checklist children recorded open with owner, dependency order and narrow TDD plan; eight acceptance criteria from the epic's prediction protocol (one met, two partial, five unmet); `release_status` bound to `shipped_register()` and `credibility_assessment()` so it cannot be greened by editing JSON. No physics, calibration or GUI change; the tool remains an exploratory simulator.
- **Evidence:** src/config/bunkershot3d_qualification.json; tests/config/bunkershot3d_qualification/test_bunkershot3d_qualification_ledger.py.

### DL-#10355 · Motion Matching Tile Visual Playback, Standardized Metrics, and Navigation Handoff (MS-82)

- **State:** shipped
- **Shipped:** 2026-09-27
- **Owner:** local
- **Issue:** #10355 (MS-82, closes #10106 gap)
- **Branch:** feat/10355-motion-matching-tile-playback
- **PR:** #10995
- **Paths:** src/tools/motion_matching/gui.py; src/tools/motion_matching/pipeline.py; src/config/feature_parity.json; docs/development/feature_parity_matrix.md; tests/tools/motion_matching/test_motion_matching_gui.py
- **Started:** 2026-09-20
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Enhanced the Motion Matching PyQt6 tile to display asynchronous animation playback (`ik_playback.gif` and `tracking_playback.gif`) via `QMovie`, populated five standardized headline metrics (`full_capture_ik_rms_mm`, `address_marker_rms_mm`, `backswing_root_error_max_mm`, `whole_run_root_rms_mm`, `inside_support_polygon_fraction`), acceptance verdict badge, and navigation handoffs to the Matched Swing Results Browser and Tour Matching Viewer. Upgraded `tools.motion_matching` in `feature_parity.json` from `gap` to `parity`.
- **Evidence:** tests/tools/motion_matching/test_motion_matching_gui.py, tests/config/feature_parity/test_matrix_freshness.py.

### DL-#10336 · MuJoCo Replay of the Merged Pinocchio Driver Candidate

- **State:** shipped
- **Shipped:** 2026-09-18
- **Owner:** codex
- **Issue:** #10336 (MS-21, epic #10363)
- **Branch:** feat/10336-mujoco-candidate-replay
- **PR:** #10448 (open)
- **Paths:** src/engines/physics_engines/mujoco/python/candidate_replay.py; src/engines/physics_engines/mujoco/python/replay_contract.py; src/engines/physics_engines/mujoco/python/replay_evidence.py; scripts/replay_pinocchio_in_mujoco.py; src/shared/python/motion_matching/pipeline/receipt_schema.py; tests/unit/motion_matching/test_mujoco_candidate_replay.py; evidence/matched/driver_full_mujoco_replay/
- **Started:** 2026-09-18
- **Last verified:** 2026-09-18 at 94ccb1825; 40 focused tests pass and 21 replay tests pass on MuJoCo 3.8.0; commit/push hooks, scoped Ruff/mypy, architecture/file-size budgets and agent-context checks pass. SELF fixes PR CI's inherited AnyIO vulnerabilities with 4.14.2 in both locks; dependency audit remains enforced.
- **Summary:** Hash-checked saved-control replay with name-mapped non-root armature, shared contact law, rigid KKT grip, no feedback or pose resets, exact G1 windows, fail-closed source/parity/physical evidence checks, and separate rejected dynamics versus IK playback. Original candidate and receipts remain unchanged. The source omits plant/control provenance and root history, so identical-plant parity and physical acceptance remain unverified/rejected.
- **CI continuation:** SELF refreshes the receipt ledger and fixes #4249's gravity fixture to use the same collision-free URDF/right-hand anchor in MuJoCo and Drake. The 5 mm gate is unchanged; analytic free-fall assertions prevent shared wrong/stationary outputs. Real Linux engines: 15 passed, 2 unavailable-engine skips. Ledger plus replay: 28 passed.
- **Evidence:** evidence/matched/driver_full_mujoco_replay/receipt.json; evidence/matched/driver_full_mujoco_replay/README.md.

### DL-#10204 · Capture Rig Shared Camera Layer

- **State:** shipped
- **Shipped:** 2026-09-15
- **Owner:** claude
- **Issue:** #10204 (part of D-sorganization/Tools#5218)
- **Branch:** claude/10204-shared-camera
- **PR:** #10211 (open; auto-merge squash armed)
- **Paths:** src/motion_capture/rig/preview_source.py; src/motion_capture/rig/recorder.py; tests/fixtures/reference_calibration/preview_source_checks.py; vendor/ud-tools; requirements-tools.txt; Cargo.toml; docs/shared_tools/divergence_inventory.v1.json; docs/shared_tools/seam_rulings.v1.json; docs/agent_context
- **Started:** 2026-09-15
- **Last verified:** 2026-09-15 (SELF; seam-drift and agent-context gates pass locally; `run_checks.py` 38 passed incl. 8 adapter checks; `tests/motion_capture/rig` + shadow/fallback hygiene 140 passed; `check_tools_pins` consistent at 1ac89c18e; ruff, ruff-format, mypy clean on changed files)
- **Summary:** Pin `vendor/ud-tools` to Tools 1ac89c18e, replace the rig's own ffmpeg preview decode with one adapter over Tools `shared.python.camera.FfmpegDirectShowSource` (seam points rig → Tools), delegate `dshow_device_ref` to the shared builder, and pin the launched command token-for-token against the pre-port list. The shared package is a `sidekick.lab.mocap` consumer, so it imports at launcher runtime and in the isolated provider harness, not in the root test process.
- **Evidence:** tests/fixtures/reference_calibration/preview_source_checks.py; scripts/shared_tools/check_tools_pins.py.

### DL-#8869 · Realtime Pub/Sub: Wire the WS Transport Instead of a Silent No-Op

- **State:** shipped
- **Shipped:** 2026-09-22
- **Owner:** claude
- **Issue:** #8869 (folds in #8868, #8942A; seam #9406 ruling: `realtime` is `split pending`, UD keeps this facade)
- **Branch:** fix/8869-realtime-pubsub-decision
- **PR:** #10655 (open)
- **Paths:** `src/shared/python/realtime/api.py`; `src/shared/python/realtime/channels.py`; `src/shared/python/realtime/__init__.py`; `tests/unit/realtime/test_facade.py`; `tests/unit/realtime/test_channels.py`; `tests/shared/realtime/test_api.py`; `tests/shared/realtime/test_channels.py`; `docs/config/pydantic-settings-migration.md` (removed `file_pubsub.py` mention)
- **Started:** 2026-09-21
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** WebSocket transport is opt-in; invalid settings fail immediately. The channel-name collision and redundant file transport were corrected. Automatic default routing stays off to avoid an implicit server; #8868 and #8942A remain separate follow-ups.
- **Evidence:** tests/unit/realtime/test_facade.py (`test_publish_with_ws_transport_routes_to_ws`, `test_publish_with_unsupported_transport_raises`, `test_subscribe_with_ws_transport_routes_to_ws`); tests/shared/realtime/test_api.py (`test_publish_ws_transport_routes_to_ws`, `test_publish_unsupported_transport_raises`, `test_subscribe_ws_transport_routes_to_ws`).

### DL-#9193 · Companion Documentation and Capability Evidence Authority

- **State:** shipped
- **Shipped:** 2026-09-21
- **Owner:** claude
- **Issue:** #9193 (parent #9174)
- **Branch:** conductor/issue-9193
- **PR:** #10668
- **Paths:** scripts/companion_evidence.py; scripts/companion_catalog.py; scripts/config/companion_documentation.v1.json; scripts/config/companion_capability_evidence.v1.json; docs/api/contracts/upstreamdrift-companion-v1.schema.json; docs/engines/engine_capability_evidence.md; tests/companion/test_companion_evidence.py
- **Started:** 2026-09-17
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Replace the empty documentation inventory and unqualified engine placeholders with two hashed registries parsed by one module: exact-commit, hash-bound, immutable-URL documentation records with derived freshness; engine capabilities qualified only by exact test/artifact evidence with an executing CI gate; per-program documentation routes; known gaps with owning issues; derived publication blockers; generated, freshness-checked provider page.

### DL-#10965 · Reconciled Scopes From Duplicate Source Rows

- **State:** shipped
- **Owner:** claude (PR not recorded in source branch `bot/issue-10965-coverage-gate-authority`) | claude (PR merged in consolidated #10988; follow-up fix for #10989 on `fix/10989-coverage-gates-defusedxml`; branch `agy/ud-10965-coverage-gates`)
- **Issue:** #10965
- **PR:** #10988, #10994, #11113
- **Branch:** `bot/issue-10965-coverage-gate-authority`; `agy/ud-10965-coverage-gates`
- **Paths:** `scripts/check_coverage_gates.py`, `scripts/check_mypy_exclusion_budget.py`, `scripts/config/mypy_exclusion_budget.json`, `.github/workflows/ci-standard.yml`, `tests/unit/scripts/test_check_coverage_gates.py`, `tests/unit/scripts/test_check_mypy_exclusion_budget.py`; `scripts/check_coverage_gates.py`, `tests/unit/scripts/test_check_coverage_gates.py`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`
- **Started:** 2026-09-26
- **Shipped:** 2026-09-29
- **Summary:** Coverage gates use one JSON authority, strict XML/JSON parsing and CI enforcement; merged #10994 covers the parser-security follow-up. This is software enforcement only, not model qualification.
- **Next step:** No further #10965 coverage-gate delivery remains. This is a software enforcement record, not evidence that engine coverage constitutes model or physical qualification.
- **Last verified:** 2026-10-11 (`944d798e`; collated from docs/development/archive/DEVELOPMENT_LOG_RECONCILIATION_SOURCE_20261011.md)

### DL-#11034 · Reconciled Scopes From Duplicate Source Rows

- **State:** shipped
- **Owner:** local (PR #11069; branch `fix/force-plate-test-module-identity-11034`) | claude (PR #11042; branch `claude/ud-11034-force-plate-flake`)
- **Issue:** #11034
- **PR:** #11069, #11042
- **Branch:** `fix/force-plate-test-module-identity-11034`; `claude/ud-11034-force-plate-flake`
- **Paths:** `src/launchers/sidekick_extension_overlay.py`, `tests/unit/launcher/test_sidekick_extension_overlay.py`, `tests/unit/sidekick/lab/bio/test_force_plate_stitching.py`; `tests/unit/sidekick/lab/bio/test_force_plate_stitching.py`
- **Started:** 2026-09-27
- **Shipped:** 2026-09-28
- **Summary:** Merged #11069 fixes canonical test imports and uninstall parent-package cleanup; #11042 preserves diagnostic context for the prior contamination symptom. These are distinct software scopes; no native qualification is claimed.
- **Next step:** The canonical-import and uninstall-detachment software correction is shipped. This record makes no native or physical qualification claim.
- **Last verified:** 2026-10-11 (`944d798e`; collated from docs/development/archive/DEVELOPMENT_LOG_RECONCILIATION_SOURCE_20261011.md)

### DL-#11596 · Commit Recovered Crocoddyl G1 Warm-Start IK and 0.60 s FDDP Stage Inputs Beside the Rk45, Rtol6 and B100 Receipts, Plus the W030r Chain-Root Candidate, With Sha256 Provenance

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11596
- **Branch:** chore/crocoddyl-warm-start-evidence-11596
- **PR:** #11598 (merged 2026-10-07; SHA d4a763d822464b53515bcbc6c798af37af164330)
- **Paths:** see #11598
- **Started:** 2026-10-07
- **Shipped:** 2026-10-07
- **Last verified:** 2026-10-11 (e68baeabde8ff639f846a167ca7703ace43bd0a9; six saved input hashes/source paths and tracked W030r candidate/receipt identity inspected, no test or solver rerun)
- **Summary:** The six recovered warm-start/stage inputs are tracked beside their receipts with SHA-256 provenance. The tracked W030r chain-root candidate SHA-256 90b80cf50f8b86bf72bc8a5a85610227820a355bb32d7b7df26ab19af5568599 matches both its receipt and recovered_inputs.json source-run identity. All recorded G1 results remain rejected; input recovery does not change that disposition.
- **Next step:** No further #11596 input-recovery delivery remains. Any new Crocoddyl G1 run is a separate, explicitly reviewed experiment; existing rejected receipts remain unchanged.

### DL-#11553 · Fix ZTCF/Drift Sign in Engine Contract Docs and Unify ZVCF on Simulation_Backends.Ztcf_Zvcf (Counterfactual Delegates)

- **State:** shipped
- **Owner:** claude
- **Issue:** #11553
- **Branch:** fix/11553-zvcf-sign-and-definition
- **PR:** #11638 (merged 2026-10-07; SHA 7300b4c4a91c9de5883dbd1e89950eaa1fb4116d), #11687 (merged 2026-10-07; SHA 71cc60b28b30cfda52391ff5610feb493dffa707)
- **Paths:** see #11638
- **Started:** 2026-10-07
- **Shipped:** 2026-10-07
- **Last verified:** 2026-10-11 (e68baeabde8ff639f846a167ca7703ace43bd0a9; source inspection only, no tests rerun)
- **Summary:** The ZTCF/drift sign and canonical ZVCF contract were corrected, with protocol aliases and compatibility coverage delivered by the two merged PRs. Software/API delivery only; dynamics and physical qualification remain separate and unqualified.
- **Next step:** No further #11553 software/API delivery remains. Any dynamics or physical qualification requires its own evidence and is not established by these PRs.

### DL-#11234 — Necromatcher Historical Player Workspace

- **Started:** not recorded in source
- **State:** shipped
- **Owner:** codex
- **Issue:** #11234; parent #11232
- **PR:** [#11239](https://github.com/D-sorganization/UpstreamDrift/pull/11239)
- **Branch:** `feat/necromatcher-workspace-11234`
- **Paths:** `src/tools/necromatcher`, `ui/src/pages/Necromatcher.tsx`, `src/shared/python/workspace/necromatcher_review.py`, launcher registries
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Shared persistent library, historical tiles, source-frame review, native/web version imports and portable exports. Reconciled launcher manifest and capability migration inventories. Further form tests, final parity acceptance and qualified fitting remain active. [Turnover](necromatcher-turnover.md).

### DL-#11083 — Motion Matching Board Review

- **State:** shipped
- **Owner:** codex
- **Issue:** owner-requested documentation review; tracked by PR #11083
- **PR:** #11083
- **Branch:** `docs/motion-matching-board-review`
- **Paths:** `docs/development/2026-09-28-motion-matching-board-review.md`
- **Started:** 2026-09-28
- **Shipped:** 2026-09-29
- **Last verified:** 2026-10-12 (merged PR #11083 at 64962e8d6b6263d5fb32bb1444712b28a9df4433; remote merge metadata only, no tests or native runs)
- **Summary:** Evidence-backed engine/model review, recent GS3DX improvements, 18 issue proposals, anatomical Home-budget options and historical-video readiness; no implementation/qualification claim.
- **Next step:** The documentary review is shipped; its implementation proposals remain separately tracked and open. No qualification claim.

### DL-#11058 — MJX Knot Optimiser Head-to-Head Benchmark and Promotion Decision

- **State:** shipped
- **Owner:** claude
- **Issue:** #11058
- **PR:** #11072
- **Branch:** `claude/ud-11058-benchmark`
- **Paths:** `src/shared/python/motion_matching/solver_benchmark.py`, `scripts/benchmark_mjx_knot_solvers.py`, `src/shared/python/motion_matching/knot_gradient_optimiser.py`, `src/engines/physics_engines/mujoco/python/motion_matching/mjx_knot_optimiser.py`, `src/shared/python/motion_matching/pipeline/`, `docs/development/full_body_models/evidence/mjx_benchmark/`
- **Started:** 2026-09-27
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Head-to-head of shooting fit, MJX Adam and MJX L-BFGS-B, each scored through the shared simulator, with promotion gated on accuracy parity, convergence and ground contact. Decision: keep `none` as the default. No MJX run converged in 10 iterations, and Adam loses iron downswing contact. L-BFGS is the most accurate opt-in (47.2 / 52.9 mm).
- **Next step:** none; merged as PR #11072 (follow-up DL-#11071).

### DL-#11059 — Shooting Fit No Longer Crashes on an Unimported `fs`

- **State:** shipped
- **Owner:** claude
- **Issue:** #11059
- **PR:** #11060
- **Branch:** `claude/ud-11059-shooting-fs`
- **Paths:** `src/shared/python/motion_matching/pipeline/dynamics.py`, `tests/unit/motion_matching/pipeline/test_dynamics.py`
- **Started:** 2026-09-27
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** `shooting_fit` used `fs` at run time without importing it, so every `--shooting-fit` run crashed; it now imports it like its sibling functions.
- **Next step:** None — merged as PR #11060.

### DL-#11051 — Opt-In MJX Knot Trajectory Optimiser Stage in the Matching Pipeline

- **State:** shipped
- **Owner:** claude (agy executor, Gemini 3.8 Flash)
- **Issue:** #11051
- **PR:** #11057
- **Branch:** `claude/ud-11051-mjx-pipeline`
- **Paths:** `src/shared/python/motion_matching/pipeline/trajectory_optimiser.py`, `src/shared/python/motion_matching/pipeline/cli.py`, `src/shared/python/motion_matching/pipeline/constants.py`, `src/engines/physics_engines/mujoco/python/motion_matching/mjx_knot_optimiser.py`, `tests/unit/motion_matching/test_trajectory_optimiser_selection.py`, `tests/unit/motion_matching/test_mjx_optimisation.py`
- **Started:** 2026-09-27
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** The matching pipeline can select the MJX knot optimiser as an opt-in trajectory stage; the default `none` leaves the pipeline output unchanged, and a missing JAX/MJX install is a named error.
- **Next step:** None — merged as PR #11057.

### DL-#11055 · Every `mj_fullM` Call Routed Through One MuJoCo 3.13-Safe Helper

- **State:** shipped
- **Owner:** claude (agy executor, Gemini 3.8 Flash)
- **Issue:** #11055
- **PR:** #11056
- **Branch:** `claude/ud-11055-fullm-helper`
- **Paths:** `src/shared/python/engine_core/mujoco_compat.py`, `tests/unit/engine_core/test_mujoco_compat.py`, every `src` module that called `mj_fullM` with `data.qM`
- **Started:** 2026-09-27
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** MuJoCo 3.13 (inside the declared range) removed `MjData.qM`; one helper selects the right `mj_fullM` signature and replaces every single-path and hand-written dual-path call.
- **Next step:** None — merged as PR #11056.

### DL-#11049 · MJX Knot Optimiser Core Moved From the Evidence CLI Into `src`

- **State:** shipped
- **Owner:** claude (agy executor, Gemini 3.8 Flash)
- **Issue:** #11049 (epic #11006 package P3)
- **PR:** #11054
- **Branch:** `claude/ud-11049-mjx-optimiser`
- **Paths:** `src/engines/physics_engines/mujoco/python/motion_matching/mjx_knot_optimiser.py`, `docs/development/full_body_models/evidence/ground_support/mjx_trajectory_optimisation.py`, `tests/unit/engines/mujoco/test_mjx_knot_optimiser.py`, `tests/unit/engines/mujoco/mjx_toy_package.py`, `tests/unit/engines/mujoco/test_mjx_evidence_cli.py`
- **Started:** 2026-09-27
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Loader, settings, knot-optimisation and diagnose logic move into a tested `src` module with validated inputs and no file or JAX-config side effects; the evidence script becomes a thin CLI over it.
- **Next step:** None — merged as PR #11054.

### DL-#11046 · MJX Evidence Prototype Rewired Onto the Tested `src` Plant, Knots and Adam Driver

- **State:** shipped
- **Owner:** claude (agy executor, Gemini 3.8 Flash)
- **Issue:** #11046 (epic #11006 package 2c)
- **PR:** #11048
- **Branch:** `claude/ud-11046-mjx-rewire`
- **Paths:** `docs/development/full_body_models/evidence/ground_support/mjx_trajectory_optimisation.py`, `tests/unit/engines/mujoco/test_mjx_evidence_cli.py`
- **Started:** 2026-09-27
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** The evidence prototype drops its private copies of the knot basis, Adam loop, contact law, weld and plant and calls the merged `src` modules; the root vertical coordinate is read by name.
- **Next step:** None — merged as PR #11048.

### DL-#11052 · PR-Scoped Tests That All Skip Report Not Executed Instead of Failing Coverage

- **State:** shipped
- **Owner:** claude
- **Issue:** #11052
- **PR:** #11053
- **Branch:** `claude/ud-11052-ci-exit5`
- **Paths:** `.github/workflows/ci-standard.yml`, `tests/ci/test_ci_infrastructure.py`
- **Started:** 2026-09-27
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** A PR-scoped pytest run that collects nothing (exit 5) with no changed `src` or dependency file now reports "Core test suite NOT EXECUTED" and exits 0, instead of falling back to a whole-`src` coverage lane whose 75 % floor the dependency-light lane cannot reach.
- **Next step:** None — merged as PR #11053.

### DL-#11044 · Canonical Calibrated Runs Regenerated on Current Code

- **State:** shipped
- **Owner:** claude
- **Issue:** #11044
- **PR:** #11050
- **Branch:** `claude/ud-11044-canonical`
- **Paths:** `docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds/`, `docs/development/full_body_models/evidence/ground_support/anthro_iron_seeds_zmp/`, `docs/development/full_body_models/evidence/ground_support/bisect_11044_receipt.json`, `docs/development/full_body_models/evidence/ground_support/CANONICAL_RUN.md`, `docs/development/matched_swing_program/README.md`, `scripts/generate_matched_swing_status.py`, `reports/matched_swing_ledger.json`
- **Started:** 2026-09-27
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** The canonical calibrated runs are regenerated on current code (driver 7.9 / 34.1 / 84.5 mm, 7-iron 6.6 / 31.6 / 88.6 mm); the pre-HO-8 receipts are kept as history because widened leg bounds explain their lower IK and both record range-of-motion flags, so G1 is not met.
- **Next step:** None — merged as PR #11050.

### DL-#11043 · Replays Start on the Dual-Grip Weld

- **State:** shipped
- **Owner:** claude
- **Issue:** #11043
- **PR:** #11047
- **Branch:** `claude/ud-11043-dls`
- **Paths:** `src/shared/python/motion_matching/weld_manifold.py`, `src/shared/python/motion_matching/full_body_forward_dynamics.py`, `src/shared/python/motion_matching/pipeline/dynamics.py`, `src/shared/python/motion_matching/execution/downswing.py`, `tests/unit/motion_matching/test_weld_manifold.py`
- **Started:** 2026-09-27
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Tracked replays start from the reference velocity projected onto the weld (mass-weighted, the weld's inelastic impulse), because the acceleration-level KKT conserves any initial weld violation and the fitted closure's 187 mm/s start opened the grip and drove the controller through a truncated-SVD singularity.
- **Next step:** None — merged as PR #11047.

### DL-#11039 · MJX Tracking Plant and Differentiable Rollout

- **State:** shipped
- **Owner:** claude (agy executor, Gemini 3.8 Flash)
- **Issue:** #11039 (epic #11006 package 2b)
- **PR:** #11040
- **Branch:** `claude/ud-11039-mjx-plant`
- **Paths:** `src/engines/physics_engines/mujoco/python/motion_matching/mjx_tracking_plant.py`, `tests/unit/engines/mujoco/test_mjx_tracking_plant.py`
- **Started:** 2026-09-27
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** MJX tracking plant with computed-torque control, JAX contact and grip-weld wrenches, and a differentiable rollout. Measured on the toy model: torque residual 7.1e-15, marker RMS 0.28 mm without contact, rollout gradient 2.2e-9 relative to central differences. Packages with a grip closure must supply weld gains; the caller's model is never mutated.
- **Next step:** None — merged as PR #11040.

### DL-#11037 · Differentiable JAX Contact Law and Grip Weld With Measured Parity

- **State:** shipped
- **Owner:** claude (agy executor, Gemini 3.8 Flash)
- **Issue:** #11037 (epic #11006 package 2a)
- **PR:** #11038
- **Branch:** `claude/ud-11037-jax-contact`
- **Paths:** `src/shared/python/motion_matching/jax_contact.py`, `tests/unit/motion_matching/test_jax_contact.py`
- **Started:** 2026-09-27
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** JAX port of the shared Hunt-Crossley plus regularised-Coulomb contact law and the spring-damper grip weld. Parity over 2000 seeded states is 0.0 N normal and 1.0e-11 N friction; normal-force gradients match central differences to 1e-6 relative and stay finite at zero tangential velocity; weld forces are equal and opposite and the net moment equals (p_b − p_a) × F_b exactly.
- **Next step:** None — merged as PR #11038.

### DL-#11032 · JAX-Free Knot Basis, Horizon Mask and Adam Driver for the MJX Solver

- **State:** shipped
- **Owner:** claude (agy executor, Gemini 3.8 Flash)
- **Issue:** #11032 (epic #11006 package 1)
- **PR:** #11035
- **Branch:** `claude/ud-11032-knot-adam`
- **Paths:** `src/shared/python/motion_matching/knot_gradient_optimiser.py`, `tests/unit/motion_matching/test_knot_gradient_optimiser.py`
- **Started:** 2026-09-27
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Knot grid with an integer knot count, vectorised hat basis that is exact at knots and refuses untouched knots, horizon mask, and a bias-corrected Adam loop generic over the array namespace that returns the best iterate by objective and stops on a non-finite cost or gradient. The first iterates match a hand-written Adam recurrence to 1e-12.
- **Next step:** None — merged as PR #11035.

### DL-#11029 · White-Jerk RTS Kinematic Smoother With Posterior Uncertainty

- **State:** shipped
- **Owner:** claude (agy executor, Gemini 3.8 Flash)
- **Issue:** #11029 (epic #11007 package 5)
- **PR:** #11031
- **Branch:** `claude/ud-11029-rts-smoother`
- **Paths:** `src/shared/python/estimation/kinematic_smoother.py`, `src/shared/python/motion_matching/pipeline/reference.py`, `src/shared/python/motion_matching/pipeline/__init__.py`, `tests/unit/estimation/test_kinematic_smoother.py`, `tests/unit/motion_matching/test_smooth_reference_bayesian_11029.py`
- **Started:** 2026-09-27
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** State [q, q̇, q̈] under a white-jerk prior, discretised exactly; the RTS posterior matches an independently built dense-batch Gaussian posterior to 1e-8, 95% bands cover the truth at the nominal rate with the true noise, sigma widens inside NaN gaps, and the ML fit recovers r within 15% and q_c within a factor of 2 on seeded synthetic data. `smooth_reference` stays bit-for-bit unchanged.
- **Next step:** None — merged as PR #11031.

### DL-#11024 · Fit the Physics-Structured Surrogate Residual by Sparse Regression

- **State:** shipped
- **Owner:** claude (agy executor, Gemini 3.8 Flash)
- **Issue:** #11024 (epic #11007 package 2, Board RM#1795)
- **PR:** #11025
- **Branch:** `claude/ud-11024-sparse-residual`
- **Paths:** `src/shared/python/neural_motion/surrogates/sparse_residual.py`, `src/shared/python/neural_motion/surrogates/physics_structured.py`, `src/shared/python/neural_motion/surrogates/__init__.py`, `tests/unit/neural_motion/test_sparse_residual.py`, `tests/unit/neural_motion/test_forward_surrogates_nm07.py`
- **Started:** 2026-09-27
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** STLSQ (Brunton, Proctor & Kutz 2016) over a polynomial plus optional sin/cos library recovers the damped-pendulum and Lorenz supports exactly on synthetic data; the physics-structured surrogate uses a fitted residual or refuses, replacing the invented constant.
- **Next step:** None — merged as PR #11025.

### DL-#11021 · Parameter Covariance on the Shared Least-Squares Motion-Matching Fits

- **State:** shipped
- **Owner:** claude (agy executor, Gemini 3.8 Flash)
- **Issue:** #11021 (MS-101 #10375, Board RM#1793)
- **PR:** #11023
- **Branch:** `claude/ud-11021-fit-covariance`
- **Paths:** `src/shared/python/estimation/fit_uncertainty.py`, `src/shared/python/motion_matching/prefix_fit.py`, `src/shared/python/motion_matching/multi_shooting_fit.py`, `src/shared/python/motion_matching/residual_regularization.py`, `tests/unit/estimation/test_fit_uncertainty.py`, `tests/unit/motion_matching/test_fit_uncertainty_wiring_11021.py`
- **Started:** 2026-09-27
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Least-squares uncertainty reports marginal free-parameter covariance; bound parameters are NaN and rank-deficient/nonfinite Jacobians yield no estimate. Equality-constrained shooting leaves uncertainty unset.
- **Next step:** None — merged as PR #11023.

### DL-#11014 · Consolidate the Swing Phase Detectors Into One Canonical Event Detector

- **State:** shipped
- **Owner:** claude (agy executor, Gemini 3.8 Flash)
- **Issue:** #11014
- **PR:** #11020
- **Branch:** `claude/ud-11014-swing-events`
- **Paths:** `src/shared/python/analysis/swing_events.py`, `src/motion_capture/reconstruct/analytics.py`, `src/shared/python/analysis/phase_detection.py`, `src/shared/python/data_io/swing_capture_import.py`, `src/shared/python/motion_matching/loaders/_align.py`, `tests/unit/analysis/test_swing_events.py`, `tests/unit/analysis/test_swing_event_parity_11014.py`
- **Started:** 2026-09-27
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** One frame-rate-aware swing-event detector now serves analytics, phase detection, import and alignment; ordering invariants and cross-site parity are tested.
- **Next step:** None — merged as PR #11020; package 2 continues as #11024.

### DL-#10960 · Fabricated-Evidence Audit: Fail-Closed Remediation of Literal Receipts and Self-Scored Fits

- **State:** shipped
- **Owner:** claude (agy executors on <host> and OG Laptop)
- **Issue:** #10960
- **PR:** #11017
- **Branch:** `claude/ud-consolidated2-20260926`
- **Paths:** `src/shared/python/motion_matching/provider.py`, `src/engines/physics_engines/*/python/motion_matching/provider.py`, `src/shared/python/neural_motion/surrogates/`, `src/shared/python/neural_motion/matrix/`, `src/shared/python/neural_motion/turnover/`, `src/shared/python/neural_motion/benchmark/runner.py`, `src/shared/python/neural_motion/inference/orchestration.py`, `src/shared/python/motion_matching/contact_identification.py`, `src/engines/physics_engines/opensim/python/tour_matching/full_swing_tracking.py`, `docs/plans/neural_motion_matching/evidence/`, `docs/development/full_body_models/evidence/contact_id/receipt.json`, `src/shared/python/motion_matching/parity_report.py`, `src/shared/python/motion_matching/parity_schema.py`, `src/shared/python/motion_matching/cross_engine_replay.py`, `src/shared/python/motion_matching/full_body_forward_dynamics.py`, `src/shared/python/motion_matching/candidate_session.py`, `docs/development/full_body_models/evidence/fb5_matching/`, `docs/development/full_body_models/evidence/fb6_parity/`, `evidence/matched/driver_g1/parity_report.json`, `docs/development/full_body_models/evidence/_gates.py`, `docs/development/full_body_models/evidence/fb3_drake/`, `docs/development/full_body_models/evidence/fb4_calibration/`, `evidence/matched/driver_g1_pinocchio/`, `src/tools/matched_swing_browser/model.py`, `src/engines/physics_engines/opensim/python/tour_matching/document_ik.py`, `docs/development/full_body_models/evidence/ground_support/anthro_driver_opensim/receipt.json`, `src/shared/python/motion_matching/matching_strategy.py`, `src/shared/python/motion_matching/ledger.py`, `reports/matched_swing_ledger.json`, `tests/fixtures/motion_matching_strategy.py`, `src/shared/python/motion_matching/club_only/`, `src/tools/motion_matching/gui.py`, `src/shared/python/motion_matching/acceptance.py`, `src/shared/python/motion_matching/kinematic_smoothing.py`, `src/shared/python/motion_matching/loaders/body_json.py`, `src/engines/physics_engines/drake/python/motion_matching/simulate.py`, `src/engines/physics_engines/pendulum/python/motion_matching/club_pendulum_match.py`, `src/shared/python/tour_baselines/calibration.py`, `src/shared/python/pose_interchange/pose_io.py`, `docs/plans/club_only_matching/evidence/club_body_candidates.json`
- **Started:** 2026-09-25
- **Shipped:** 2026-09-27
- **Last verified:** 2026-09-26 — consolidated branch on `71610a4ff`: motion_matching, neural_motion, opensim, tools/motion_matching, config and every changed test file: 3034 passed, 21 failed, all baseline or host-only (16 known motion_matching failures, OpenSim binding absent, Rajagopal asset absent, local 10k corpus present); ledger regenerated and fresh.
- **Summary:** Fail-closed software fixes span motion matching, neural-motion receipts, parity and rollout evidence, native-lane reporting, and UI seed guards. Missing data remains unmeasured or unverified; no synthetic evidence or model qualification is emitted.
- **Next step:** The fail-closed receipt correction is shipped. The checked receipts retain empty qualified/promoted sets and unmeasured status; no native or physical qualification is claimed.

### DL-#12316 — Fix the PreconditionError Exception-Identity Split at the Shared Contracts Seam

- **State:** shipped
- **Owner:** fleet-orchestrator (agent: claude)
- **Issue:** #12316 (retrospective governance created 2026-10-11; the original source recorded no governing issue)
- **Branch:** `bot/contracts-exception-identity`; PR #11175
- **Paths:** `src/shared/python/_contracts_exceptions.py`, `tests/unit/motion_matching/test_stability_matrix.py`
- **Started:** 2026-09-30
- **Shipped:** 2026-09-30
- **Last verified:** 2026-10-11 (`0446a2b5`; collated from verified merged PR metadata)
- **Summary:** `_contracts_exceptions.py` redefined the DbC exceptions in parallel with the public `contracts` module, so `@precondition` violations raised via `contracts` could not be caught as the shard-imported `PreconditionError` — `TestStabilityMatrix::test_get_canonical_test_invalid` stayed red on main. The shard now re-exports the same class objects; both import paths remain usable and referentially identical, asserted in the test.
- **Next step:** PR #11175 merged 2026-09-30 (0446a2b5e46a1192ff58c84f614975bd0be2b2a4). The original source recorded no governing issue; retrospective governance was added as #12316 on 2026-10-11. This records the exception-identity delivery only.
- **PR:** #11175

### DL-#9191 · Govern Screenshot Schema, Capture Metadata, and Qualified Assets

- **State:** shipped
- **Owner:** local
- **Issue:** #9191
- **Branch:** `feat/comp-b3-screenshot-governance-9191`
- **Paths:** `docs/api/contracts/upstreamdrift-companion-screenshots-v1.schema.json`, `docs/api/contracts/upstreamdrift-companion-v1.schema.json`, `scripts/companion_screenshots.py`, `scripts/companion_catalog.py`, `scripts/config/companion_screenshots.v1.json`, `docs/screenshots/`, `tests/companion/test_companion_screenshots.py`, `tests/companion/test_companion_catalog.py`, `tests/companion/test_companion_publication.py`, `tests/fixtures/companion/current-v1.0.0.json`, `SPEC.md`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`
- **Started:** 2026-09-29
- **Last verified:** 2026-10-11 (`27af7ffe`; collated from verified GitHub merge metadata (PR #11140, 2026-09-30))
- **Shipped:** 2026-09-30
- **Summary:** COMP-B3 records screenshot provenance/hashes, generates six representative assets, and registers the remaining captures as pending in the companion catalog and publication inventory.
- **Next step:** Shipped via PR #11140 on 2026-09-30 (merge SHA 27af7ffe41f1fa8e26c367f27f9cdd0fe79da866). Preserved limitation: Only 6 of 74 screenshots exist; documentation completion cannot imply full visual inventory.
- **PR:** #11140

### DL-#11008 · Make Ball-Flight Parity Fixture Export Opt-In

- **State:** shipped
- **Owner:** claude
- **Issue:** #11008
- **PR:** #11012 (ready; auto-merge armed)
- **Branch:** `fix/ball-flight-parity-fixture-opt-in`
- **Paths:** `tests/parity/test_ball_flight_parity.py`
- **Started:** 2026-09-26
- **Last verified:** 2026-10-11 (`0aa8bb66`; collated from verified GitHub merge metadata (PR #11012, 2026-09-27))
- **Shipped:** 2026-09-27
- **Summary:** The export test writes to `tmp_path` and asserts the vector schema. It rewrites the committed golden only when `UPSTREAMDRIFT_REGENERATE_PARITY_FIXTURES=1`. A new test checks read-only that the committed fixture has the schema and the path pinned in `src/config/capability_migration.json`.
- **Next step:** Shipped via PR #11012 on 2026-09-27 (merge SHA 0aa8bb660d138c9255a4f224902dd6133f21a3c0).

### DL-#11071 — Run the MJX L-BFGS Arm Toward Convergence

- **State:** shipped
- **Owner:** claude
- **Issue:** #11071
- **PR:** #11075
- **Branch:** `claude/ud-11071-lbfgs-converge`
- **Paths:** `scripts/benchmark_mjx_knot_solvers.py`, `docs/development/full_body_models/evidence/mjx_benchmark_lbfgs60/`
- **Started:** 2026-09-28
- **Last verified:** 2026-10-11 (`0206717f`; collated from verified GitHub merge metadata (PR #11075, 2026-09-28))
- **Shipped:** 2026-09-28
- **Summary:** Re-runs only `mjx-lbfgs` at 60 iterations against the #11058 incumbents. Replay RMS falls to 40.8 / 42.6 mm (driver / iron) and the iron downswing weight fraction rises to 0.29, but neither capture converges, so `none` stays the default.
- **Next step:** Shipped via PR #11075 on 2026-09-28 (merge SHA 0206717f70a654744165990fe4b229fcdd4f3dd7). Preserved limitation: Both captures did not converge; the no-MJX default decision is not an accepted physical result.

### DL-#11102 — Publish a Best-Candidate Viewer With Honest Residuals

- **State:** shipped
- **Owner:** local
- **Issue:** #11102
- **Branch:** `feat/mmr-16-best-candidate-viewer-11102`
- **Paths:** `src/tools/tour_matching_viewer/core.py`, `src/tools/tour_matching_viewer/gui.py`, `src/tools/matched_swing_browser/model.py`, `src/tools/matched_swing_browser/gui.py`, `tests/unit/tools/test_tour_matching_viewer_residuals.py`, `tests/unit/tools/test_matched_swing_browser_best_candidate.py`, `tests/unit/tools/test_tour_matching_viewer_combo.py`
- **Started:** 2026-09-29
- **Last verified:** 2026-10-11 (`9c23f1a2`; collated from verified GitHub merge metadata (PR #11200, 2026-10-01))
- **Shipped:** 2026-10-01
- **Summary:** Best-candidate viewer preserves observations and rejected verdicts while adding comparable RMS ranking, residual navigation, metadata-bearing exports and missing-engine recovery. Viewer output does not qualify candidates.
- **Remaining scope on #11102:** web/API surface parity, accessibility/native visual review and the remaining acceptance checkboxes; desktop PyQt fixes above do not close them.
- **Next step:** Shipped via PR #11200 on 2026-10-01 (merge SHA 9c23f1a2efcbc722de1bd86299b8351c5e96c41e).
- **PR:** #11200

### DL-#11097 — Repair Reduced-Model and Club-Only Product Claims (MMR-11)

- **State:** shipped
- **Owner:** local
- **Issue:** #11097
- **Branch:** `feat/mmr-11-reduced-model-claims-11097`
- **Paths:** `src/shared/python/tour_baselines/coverage.py`, `src/shared/python/tour_baselines/baseline_package.py`, `src/shared/python/tour_baselines/qualification.py`, `src/shared/python/motion_matching/provider.py`, `src/engines/physics_engines/pendulum/python/motion_matching/provider.py`, `src/shared/python/motion_matching/club_only/ui_integration.py`, `src/shared/python/motion_matching/club_only/matrix_qualification.py`, `tests/unit/motion_matching/test_fit_options_dbc.py`
- **Started:** 2026-09-29
- **Last verified:** 2026-10-11 (`2cb3d58c`; collated from verified GitHub merge metadata (PR #11202, 2026-10-01))
- **Shipped:** 2026-10-01
- **Summary:** Added code-level, fail-closed claims gates: four driven-triple pendulum receipts (`driver_amateur`, `driver_elite`, `iron_amateur`, `iron_elite`) disqualified at projection time (evidence files on disk unchanged); promoted-package hash and out-of-plane-residual integrity checks; club-only fresh continuous replay + labeled inferred posture; matrix all-complete claims fail closed on unresolved cells; pendulum planar-floor early rejection now driven by the CURRENT target's plane distances with a DbC-validated `max_marker_rmse_m`.
- **PR:** #11202
- **Next step:** Shipped via PR #11202 on 2026-10-01 (merge SHA 2cb3d58cfe3b181c12c26015d236dcdb234cf574).
- **Open (per MMR-11 acceptance, not satisfied here):** raw-to-package reproduction/regeneration of the reported baselines, Board-selected required club-only cells, and native qualification runs. PR references (not "Closes") #11097 for these.

### DL-#11098 — Historical-Video Evidence Review Workflow

- **State:** shipped
- **Owner:** local
- **Issue:** #11098
- **Branch:** `feat/mmr-12-historical-video-review-11098`
- **Paths:** `src/shared/python/shadow_tracker/service.py`, `src/tools/shadow_tracker/gui.py`, `tests/unit/shadow_tracker/test_service.py`, `tests/tools/shadow_tracker/test_shadow_tracker_gui.py`
- **Started:** 2026-09-29
- **Last verified:** 2026-10-11 (`add9a1bd`; collated from verified GitHub merge metadata (PR #11197, 2026-10-01))
- **Shipped:** 2026-10-01
- **Summary:** Shadow Tracker preserves shot isolation/lineage, PTS timing, bounded cancellable decode and corrupt-media recovery; imports validate before mutation and unknown physical time remains null.
- **Next step:** Shipped via PR #11197 on 2026-10-01 (merge SHA add9a1bd59ba21d99e6c65d04058b6db8a870d91).
- **PR:** #11197

### DL-#11107 — Implement Simscape Continuous-Replay Qualification Harness

- **State:** shipped
- **Owner:** local
- **Issue:** #11107
- **Branch:** `feat/mmr-07-simscape-continuous-replay-11107`
- **Paths:** `src/shared/python/motion_matching/simscape_replay_harness.py`, `tests/unit/motion_matching/test_simscape_continuous_replay_harness.py`, `scripts/matlab/run_simscape_candidate.ps1`, `docs/development/simscape_tour_matching/native_evidence/two_window_fit_9967_102/simscape_replay_qualification.json`, `SPEC.md`
- **Started:** 2026-09-29
- **Last verified:** 2026-10-11 (`ee806b50`; collated from verified GitHub merge metadata (PR #11198, 2026-10-01))
- **Shipped:** 2026-10-01
- **Summary:** Added fail-closed continuous replay checks for source, time, samples, runtime and frozen G1 metrics. Native run 102 remains REJECTED at 40.3 mm terminal RMS against a 35 mm ceiling; no new run or qualification is claimed.
- **Next step:** Shipped via PR #11198 on 2026-10-01 (merge SHA ee806b50cb7db3e7001ce25b94c536895712625a). Preserved limitation: Run 102 was rejected; keep rejection visible and do not claim physical acceptance.
- **PR:** #11198

### DL-#11124 — Anti-Phantom-Merge Path Extraction for Scripts, Workflows, and Parentheticals

- **State:** shipped
- **Owner:** local
- **Issue:** #11124
- **Branch:** `fix/phantom-guard-scripts-paths-11124`
- **Paths:** `scripts/ci/check_phantom_guard_paths.py`, `tests/scripts/test_check_phantom_guard_paths.py`
- **Started:** 2026-09-29
- **Last verified:** 2026-10-11 (`657d9528`; collated from verified GitHub merge metadata (PR #11131, 2026-09-29))
- **Shipped:** 2026-09-29
- **Summary:** Fixes false positive in anti-phantom-merge Rule 3 by expanding `ISSUE_PATH_PATTERN` to recognize `scripts/` and `.github/workflows/` (and `.github/`) paths, adding `(?<![\w/.-])` boundary protection to reject non-path prefix substrings (e.g. `engines/api`), stripping trailing punctuation from extracted paths, and dropping parenthetical prose fragments lacking valid file extensions.
- **Next step:** Shipped via PR #11131 on 2026-09-29 (merge SHA 657d95287e7848dfc568ca4e70236bd5e982393a).
- **PR:** #11131

### DL-#11184 — Restore the High-Severity UI Npm Audit Gate

- **State:** shipped
- **Owner:** codex
- **Issue:** #11184
- **Branch:** `fix/main-npm-audit-11184`
- **Paths:** `ui/package-lock.json`, `ui/src/test/dependencySecurityContract.test.ts`, `SPEC.md`
- **Started:** 2026-09-30
- **Last verified:** 2026-10-11 (`1a7f7f5e`; collated from verified GitHub merge metadata (PR #11187, 2026-09-30))
- **Shipped:** 2026-09-30
- **Summary:** Raised only the compatible transitive resolutions for `brace-expansion` (5.0.9 → 5.0.12) and `undici` (8.10.0 → 8.11.2); added a lockfile security contract. Manifest dependencies, overrides, audit threshold, and the two moderate findings are unchanged.
- **Next step:** Shipped via PR #11187 on 2026-09-30 (merge SHA 1a7f7f5e8fdbd3f82b69b0f8bb0fc43df7557791).
- **PR:** #11187

### DL-#11233 — Necromatcher Persistent Library

- **Started:** not recorded in source
- **State:** shipped
- **Owner:** codex
- **PR:** [#11237](https://github.com/D-sorganization/UpstreamDrift/pull/11237)
- **Issue:** #11233; parent #11232; player epics #11226 and #11229
- **Branch:** `feat/necromatcher-library-11233`
- **Paths:** `src/shared/python/workspace/necromatcher.py`, `src/shared/python/workspace/necromatcher_capture.py`, `src/api/routes/necromatcher.py`
- **Last verified:** 2026-10-11 (`93b5aa1a`; collated from verified GitHub merge metadata (PR #11237, 2026-10-02))
- **Shipped:** 2026-10-02
- **Summary:** Reuse project/session spine, artifact hashes and canonical torque evaluation for immutable historical-player library; captures remain image observations and model bytes remain unqualified candidates. [Turnover](necromatcher-turnover.md).
- **Next step:** Shipped via PR #11237 on 2026-10-02 (merge SHA 93b5aa1a5a2bd0d6ce4beceaf57904fc903780fc). Preserved limitation: Model status remains unqualified; preserve that state and all downstream blockers.

### DL-#11190 · Disable FastRestart for Solve_Starting_Pose and Resolve Model Path in Run_Leaderboard

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11190
- **Branch:** fix/11190-motion-matching-fast-restart
- **PR:** #11518
- **Paths:** see #11518
- **Started:** 2026-10-05
- **Last verified:** 2026-10-11 (`d816c998`; collated from verified GitHub merge metadata (PR #11518, 2026-10-05))
- **Shipped:** 2026-10-05
- **Summary:** Disable FastRestart for solve_starting_pose and resolve model path in run_leaderboard
- **Next step:** Shipped via PR #11518 on 2026-10-05 (merge SHA d816c9985835ba78f275ee4f6ae5efc7c6ae152c).

### DL-#11513 · Point Contributors at Change Fragments in Post_Spec_Reminder

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11513
- **Branch:** fix/11513-post-spec-reminder-fragments
- **PR:** #11519
- **Paths:** see #11519
- **Started:** 2026-10-05
- **Last verified:** 2026-10-11 (`739df607`; collated from verified GitHub merge metadata (PR #11519, 2026-10-05))
- **Shipped:** 2026-10-05
- **Summary:** Point contributors at change fragments in post_spec_reminder
- **Next step:** Shipped via PR #11519 on 2026-10-05 (merge SHA 739df607f6e658ff966f33010f67d5fb6cc1e3c3).

### DL-#11589 · Retire Stale Shared-Code Header Test

- **State:** shipped
- **Owner:** claude
- **Issue:** #11589
- **Branch:** chore/retire-unflagged-header-test-11589
- **PR:** #11590
- **Paths:** see #11590
- **Started:** 2026-10-06
- **Last verified:** 2026-10-11 (`a4031aa0`; collated from verified GitHub merge metadata (PR #11590, 2026-10-06))
- **Shipped:** 2026-10-06
- **Summary:** chore(tests): retire stale test_no_unflagged_shared_code header test and its unit-gate quarantine entry
- **Next step:** Shipped via PR #11590 on 2026-10-06 (merge SHA a4031aa02ccb410936de6976090dc63abe2d7887).

### DL-#11691 · Protect `.yml` and Related Filenames in the Title-Case Checker

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11691
- **Branch:** merged via #11692
- **PR:** #11692
- **Paths:** see #11692
- **Started:** 2026-10-07
- **Last verified:** 2026-10-07 (`de0e48a1`; collated from changes/11691-protect-yml-and-related-filenames-in-the.md)
- **Summary:** Protect `.yml` and Related Filenames in the Title-Case Checker
- **Next step:** Shipped in PR #11692.

### DL-#11641 · Register latex-references.yml in the Workflow Inventory so the Inventory Gate Passes on Main

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11641
- **Branch:** merged via #11666
- **PR:** #11666
- **Paths:** see #11666
- **Started:** 2026-10-07
- **Last verified:** 2026-10-07 (`ee47b1ee`; collated from changes/11641-register-latex-references-yml-in-the-wor.md)
- **Summary:** Register latex-references.yml in the workflow inventory so the inventory gate passes on main
- **Next step:** Shipped in PR #11666.

### DL-#11574 · Ci: Compile Docs/Research LaTeX References With Pinned Tectonic and Check Changed Title Case

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11574
- **Branch:** merged via #11641
- **PR:** #11641
- **Paths:** see #11641
- **Started:** 2026-10-07
- **Last verified:** 2026-10-07 (`18a4a5bd`; collated from changes/11574-ci-compile-docs-research-latex-reference.md)
- **Summary:** ci: compile docs/research LaTeX references with pinned Tectonic and check changed title case
- **Next step:** Shipped in PR #11641.

### DL-#11554 · Fixed-Size Window Residual: Typed DimeResidualEvaluationError Replaces the Np.Full(100, 1E8) Fake Residual; Bound Residuals Fixed-Length; DbC Postcondition on Residual Length. Single Shooting / Real Defects Deferred to MOSAIC-13 (#11545).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11554
- **Branch:** fix/11554-fixed-size-residual
- **PR:** #11637
- **Paths:** src/shared/python/estimation/dime_dynamics_window.py
- **Started:** 2026-10-07
- **Last verified:** 2026-10-07 (`feb58aa1`; collated from changes/11554-fixed-size-window-residual-typed-dimeres.md)
- **Summary:** Fixed-size window residual: typed DimeResidualEvaluationError replaces the np.full(100, 1e8) fake residual; bound residuals fixed-length; DbC postcondition on residual length. Single shooting / real defects deferred to MOSAIC-13 (#11545).
- **Next step:** Shipped in PR #11637.

### DL-#11635 · Bump Tornado to 6.5.10 and Remove the Three Stale Tornado Pip-Audit Waivers

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11635
- **Branch:** merged via #11636
- **PR:** #11636
- **Paths:** see #11636
- **Started:** 2026-10-07
- **Last verified:** 2026-10-07 (`019a42d3`; collated from changes/11635-bump-tornado-to-6-5-10-and-remove-the-th.md)
- **Summary:** Bump tornado to 6.5.10 and remove the three stale tornado pip-audit waivers
- **Next step:** Shipped in PR #11636.

### DL-#11617 · OpenSim Musculoskeletal Swing: GRF Estimate, StaticOptimization Pipeline, Receipt

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11617
- **Branch:** merged via #11621
- **PR:** #11621, #11627
- **Paths:** see #11621
- **Started:** 2026-10-07
- **Last verified:** 2026-10-07 (`976b9ce4`; collated from changes/11617-phase-2-spec-driven-musculoskeletal-mode.md)
- **Summary:** OpenSim musculoskeletal swing: GRF estimate, StaticOptimization pipeline, receipt
- **Next step:** Shipped in PR #11621.

### DL-#11612 · P-7: Spec Full-Body Model Loaded Through the MyoSuite Runtime Reproduces MuJoCo Bit-Exactly Under the Same Inputs (L0/L1/L2 Parity Receipt)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11612
- **Branch:** merged via #11628
- **PR:** #11628
- **Paths:** see #11628
- **Started:** 2026-10-07
- **Last verified:** 2026-10-07 (`cf9fb216`; collated from changes/11612-p-7-spec-full-body-model-loaded-through.md)
- **Summary:** P-7: spec full-body model loaded through the MyoSuite runtime reproduces MuJoCo bit-exactly under the same inputs (L0/L1/L2 parity receipt)
- **Next step:** Shipped in PR #11628.

### DL-#11611 · OpenSim Same-Input Parity Adapter (Exact Weld KKT, Shared Contact Law); Fix OSIM Exporter XYZ Euler Convention and 41-Coordinate Validation

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11611
- **Branch:** merged via #11628
- **PR:** #11628
- **Paths:** see #11628
- **Started:** 2026-10-07
- **Last verified:** 2026-10-07 (`cf9fb216`; collated from changes/11611-opensim-same-input-parity-adapter-exact.md)
- **Summary:** OpenSim same-input parity adapter (exact weld KKT, shared contact law); fix OSIM exporter XYZ Euler convention and 41-coordinate validation
- **Next step:** Shipped in PR #11628.

### DL-#11607 · Same-Input Parity P-2: Same-Input-Bundle/V1, Shared ZOH RK4 Integrator (8 Substeps for Stiff Contact Modes), Per-Step Closure Projection, MuJoCo Reference Generation, Replay Scoring; Pipeline Persists Q_Track

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11607
- **Branch:** merged via #11628
- **PR:** #11628
- **Paths:** see #11628
- **Started:** 2026-10-07
- **Last verified:** 2026-10-07 (`cf9fb216`; collated from changes/11607-same-input-parity-p-2-same-input-bundle.md)
- **Summary:** Same-input parity P-2: same-input-bundle/v1, shared ZOH RK4 integrator (8 substeps for stiff contact modes), per-step closure projection, MuJoCo reference generation, replay scoring; pipeline persists q_track
- **Next step:** Shipped in PR #11628.

### DL-#11606 · Same-Input Parity P-1: VectorPlant + Convention-Free Closure Projection; MuJoCo Exact-KKT Option; Pointwise MuJoCo/Drake/Pinocchio Acceleration Parity Gate and Receipt (Worst 1.1E-9 Relative Over 31 Frames)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11606
- **Branch:** merged via #11615
- **PR:** #11615
- **Paths:** see #11615
- **Started:** 2026-10-07
- **Last verified:** 2026-10-07 (`f2def974`; collated from changes/11606-same-input-parity-p-1-vectorplant-conven.md)
- **Summary:** Same-input parity P-1: VectorPlant + convention-free closure projection; MuJoCo exact-KKT option; pointwise MuJoCo/Drake/Pinocchio acceleration parity gate and receipt (worst 1.1e-9 relative over 31 frames)
- **Next step:** Shipped in PR #11615.

### DL-#11623 · Per-Workspace CARGO_HOME for Rust-Quickstart and Realtime-Soak; Contract Test Pins RUSTUP_HOME and CARGO_HOME for Every Rust Job (RM#2021)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11623
- **Branch:** merged via #11624
- **PR:** #11624
- **Paths:** see #11624
- **Started:** 2026-10-07
- **Last verified:** 2026-10-07 (`17b3b95c`; collated from changes/11623-per-workspace-cargo-home-for-rust-quicks.md)
- **Summary:** Per-workspace CARGO_HOME for rust-quickstart and realtime-soak; contract test pins RUSTUP_HOME and CARGO_HOME for every Rust job (RM#2021)
- **Next step:** Shipped in PR #11624.

### DL-#1893 · Archive HANDOFF.md and DEVELOPMENT_LOG.md Verbatim (RM#1893)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #1893
- **Branch:** merged via #11620
- **PR:** #11620
- **Paths:** see #11620
- **Started:** 2026-10-07
- **Last verified:** 2026-10-07 (`7a4e3436`; collated from changes/1893-archive-handoff-devlog.md)
- **Summary:** Archive HANDOFF.md And DEVELOPMENT_LOG.md Verbatim (RM#1893)
- **Next step:** Shipped in PR #11620.

### DL-#11584 · Settings_Dialog: Ignore Late Dependency-Check Results After Close; Worker Emits Failed for Any Exception

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11584
- **Branch:** merged via #11618
- **PR:** #11618
- **Paths:** see #11618
- **Started:** 2026-10-07
- **Last verified:** 2026-10-07 (`1696354a`; collated from changes/11584-settings-dialog-ignore-late-dependency-c.md)
- **Summary:** settings_dialog: ignore late dependency-check results after close; worker emits failed for any exception
- **Next step:** Shipped in PR #11618.

### DL-#11601 · Ci: Deleted-Test Guard Honours a Reviewed Allowlist (Scripts/Config/Reviewed_Test_Deletions.Json: Path + Issue + Reason, Fails Closed When Malformed) so an Approved Test Retirement Can Pass the Merge Queue; Unapproved Deletions Still Fail.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11601
- **Branch:** merged via #11602
- **PR:** #11602
- **Paths:** see #11602
- **Started:** 2026-10-06
- **Last verified:** 2026-10-06 (`cf9f9ad8`; collated from changes/11601-ci-deleted-test-guard-honours-a-reviewed.md)
- **Summary:** ci: deleted-test guard honours a reviewed allowlist (scripts/config/reviewed_test_deletions.json: path + issue + reason, fails closed when malformed) so an approved test retirement can pass the merge queue; unapproved deletions still fail.
- **Next step:** Shipped in PR #11602.

### DL-#1976 · Make the Change-Fragment Round-Trip Test Hermetic so Collating a Real DL-#1976 Entry Cannot Break It

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #1976
- **Branch:** merged via #11592
- **PR:** #11592
- **Paths:** see #11592
- **Started:** 2026-10-06
- **Last verified:** 2026-10-06 (`d2086034`; collated from changes/1976-make-the-change-fragment-round-trip-test.md)
- **Summary:** Make the change-fragment round-trip test hermetic so collating a real DL-#1976 entry cannot break it
- **Next step:** Shipped in PR #11592.

### DL-#11595 · Ci: Give Every Dtolnay/Rust-Toolchain Job a Per-Job RUSTUP_HOME (${{ Github.Workspace }}/.rustup-home) so Rustup Never Upgrades the Runner Image's Partially Installed ~/.rustup Toolchain in Place; Contract Test in Tests/Ci/test_unit_gate_rust_kernel.py.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11595
- **Branch:** merged via #11597
- **PR:** #11597
- **Paths:** see #11597
- **Started:** 2026-10-06
- **Last verified:** 2026-10-06 (`39fbd017`; collated from changes/11595-ci-give-every-dtolnay-rust-toolchain-job.md)
- **Summary:** ci: give every dtolnay/rust-toolchain job a per-job RUSTUP_HOME (${{ github.workspace }}/.rustup-home) so rustup never upgrades the runner image's partially installed ~/.rustup toolchain in place; contract test in tests/ci/test_unit_gate_rust_kernel.py.
- **Next step:** Shipped in PR #11597.

### DL-#2018 · Vendor Automerge_Guard and Requeue_Stalled_Merges (Stalled-Merge Requeue Tooling) From Repository_Management With Canonical Tests (Refs Repository_Management#2018)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #2018
- **Branch:** merged via #11577
- **PR:** #11577, #11586
- **Paths:** see #11577
- **Started:** 2026-10-06
- **Last verified:** 2026-10-06 (`1943a1b7`; collated from changes/2018-resync-requeue.md)
- **Summary:** Vendor automerge_guard and requeue_stalled_merges (stalled-merge requeue tooling) from Repository_Management with canonical tests (Refs Repository_Management#2018)
- **Next step:** Shipped in PR #11577.

### DL-#11593 · Clear the Security Findings Main Carries: Werkzeug 3.1.8 to 3.1.9 (CVE-2026-102598) and Fsspec 2026.1.0 to 2026.6.0 (CVE-2026-104851) in Both Python Lockfiles, and Source-Map-Js 1.2.1 to 1.2.2 (CVE-2026-93749 / GHSA-68Fv-2Mgg-Jv7q) in Ui/Package-Lock.Json; Lockfile-Only Upgrades, No New Waiver.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11593
- **Branch:** merged via #11594
- **PR:** #11594
- **Paths:** see #11594
- **Started:** 2026-10-06
- **Last verified:** 2026-10-06 (`2d183206`; collated from changes/11593-bump-werkzeug-3-1-8-to-3-1-9-in-both-loc.md)
- **Summary:** Clear the security findings main carries: werkzeug 3.1.8 to 3.1.9 (CVE-2026-102598) and fsspec 2026.1.0 to 2026.6.0 (CVE-2026-104851) in both Python lockfiles, and source-map-js 1.2.1 to 1.2.2 (CVE-2026-93749 / GHSA-68fv-2mgg-jv7q) in ui/package-lock.json; lockfile-only upgrades, no new waiver.
- **Next step:** Shipped in PR #11594.

### DL-#11577 · Fix(Tests): Wait for Settings-Dialog Dependency-Check Workers Before Teardown

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11577
- **Branch:** merged via #11581
- **PR:** #11581
- **Paths:** see #11581
- **Started:** 2026-10-06
- **Last verified:** 2026-10-06 (`843e3e82`; collated from changes/11577-fix-tests-wait-for-settings-dialog-depen.md)
- **Summary:** fix(tests): wait for settings-dialog dependency-check workers before teardown
- **Next step:** Shipped in PR #11581.

### DL-#11571 · Reconcile Engine Capability Matrix With Get_Capabilities and Add Consistency Test

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11571
- **Branch:** merged via #11576
- **PR:** #11576, #11580
- **Paths:** see #11576
- **Started:** 2026-10-05
- **Last verified:** 2026-10-05 (`a40295cc`; collated from changes/11571-launcher-capability-tests.md)
- **Summary:** Reconcile engine capability matrix with get_capabilities and add consistency test
- **Next step:** Shipped in PR #11576.

### DL-#11573 · Report NaN for Unmeasured Replay Metrics and Fail Closed in Acceptance Evaluation

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11573
- **Branch:** merged via #11575
- **PR:** #11575
- **Paths:** see #11575
- **Started:** 2026-10-05
- **Last verified:** 2026-10-05 (`34cf0eb8`; collated from changes/11573-unmeasured-replay-metrics-nan.md)
- **Summary:** Report NaN for unmeasured replay metrics and fail closed in acceptance evaluation
- **Next step:** Shipped in PR #11575.

### DL-#11567 · Drake Full-Body Fit Scores Forward Kinematics of the Drake Rollout Instead of the Warm-Start Candidate's Markers

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11567
- **Branch:** merged via #11568
- **PR:** #11568
- **Paths:** see #11568
- **Started:** 2026-10-05
- **Last verified:** 2026-10-05 (`80dde80a`; collated from changes/11567-drake-full-body-fit-scores-forward-kinem.md)
- **Summary:** Drake full-body fit scores forward kinematics of the Drake rollout instead of the warm-start candidate's markers
- **Next step:** Shipped in PR #11568.

### DL-#11556 · Ci(Security): Re-Vendor fork_pr_runner_guard.py and New fork_pr_guard_analysis.py From Repository_Management (RM#1996): Same-Repo `if:` Exemption and Sink-Based Head-Checkout Analysis.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11556
- **Branch:** merged via #11556
- **PR:** #11556
- **Paths:** see #11556
- **Started:** 2026-10-05
- **Last verified:** 2026-10-05 (`b081650e`; collated from changes/11556-ci-security-re-vendor-fork-pr-runner-gua.md)
- **Summary:** ci(security): re-vendor fork_pr_runner_guard.py and new fork_pr_guard_analysis.py from Repository_Management (RM#1996): same-repo if: exemption and sink-based head-checkout analysis.
- **Next step:** Shipped in PR #11556.

### DL-#11552 · Fix(Estimation): DIME Initialisers, Ablations and Manifest Report Not-Measured Instead of Constants or Ground-Truth-Derived Figures

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11552
- **Branch:** merged via #11559
- **PR:** #11559
- **Paths:** see #11559
- **Started:** 2026-10-05
- **Last verified:** 2026-10-05 (`3637c80e`; collated from changes/11552-fix-estimation-dime-initialisers-ablatio.md)
- **Summary:** fix(estimation): DIME initialisers, ablations and manifest report not-measured instead of constants or ground-truth-derived figures
- **Next step:** Shipped in PR #11559.

### DL-#11551 · DIME Replay and Engine Qualification Fail Closed on Missing Reference/Metrics; Replay Provenance Read From the Environment

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11551
- **Branch:** merged via #11558
- **PR:** #11558
- **Paths:** see #11558
- **Started:** 2026-10-05
- **Last verified:** 2026-10-05 (`8dfd58bc`; collated from changes/11551-dime-replay-and-engine-qualification-fai.md)
- **Summary:** DIME replay and engine qualification fail closed on missing reference/metrics; replay provenance read from the environment
- **Next step:** Shipped in PR #11558.

### DL-#11549 · Drift Prediction Propagates Covariance Through F = Df/Dx per Step, Steps the Full Horizon as H Steps of Dt, and Never Actuates the Floating-Base Root by Default

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11549
- **Branch:** fix/11549-drift-covariance-propagation
- **PR:** #11562
- **Paths:** src/shared/python/estimation/drift_prediction.py
- **Started:** 2026-10-05
- **Last verified:** 2026-10-05 (`dc22a363`; collated from changes/11549-drift-prediction-propagates-covariance-t.md)
- **Summary:** drift prediction propagates covariance through F = df/dx per step, steps the full horizon as H steps of dt, and never actuates the floating-base root by default
- **Next step:** Shipped in PR #11562.

### DL-#11548 · Multi-Trial MAP: Residual and Jacobian Prior Rows Share One Layout (Locked Priors Excluded From Both); Posterior Covariance Is the Schur Marginal Over the Shared Parameters, Reported as Unavailable (`covariance_status="rank_deficient"`, NaN) When the Fisher Matrix Is Singular; Calculation Documented in Docs/Conventions/canonical-v2.md §6.1

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11548
- **Branch:** merged via #11557
- **PR:** #11557
- **Paths:** see #11557
- **Started:** 2026-10-05
- **Last verified:** 2026-10-05 (`efa445dd`; collated from changes/11548-multi-trial-map-residual-and-jacobian-pr.md)
- **Summary:** Multi-trial MAP: residual and Jacobian prior rows share one layout (locked priors excluded from both); posterior covariance is the Schur marginal over the shared parameters, reported as unavailable (`covariance_status="rank_deficient"`, NaN) when the Fisher matrix is singular; calculation documented in docs/conventions/canonical-v2.md §6.1
- **Next step:** Shipped in PR #11557.

### DL-#11547 · DIME Solver Cache Key Now Covers Targets, Weights and Solver Options; Cost Breakdown Reports Measured Values or Not-Measured (None) Instead of Wall-Time Fractions

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11547
- **Branch:** merged via #11561
- **PR:** #11561
- **Paths:** see #11561
- **Started:** 2026-10-05
- **Last verified:** 2026-10-05 (`d7b122a1`; collated from changes/11547-dime-solver-cache-key-now-covers-targets.md)
- **Summary:** DIME solver cache key now covers targets, weights and solver options; cost breakdown reports measured values or not-measured (None) instead of wall-time fractions
- **Next step:** Shipped in PR #11561.

### DL-#11550 · DIME Manifold Retract and Local-Coordinate Jacobians Are Now the Exact Derivatives of the Right Quaternion Chart at Any Pose and Velocity (J_R^-1, dExp/Dv), Verified Against Central Finite Differences

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11550
- **Branch:** merged via #11560
- **PR:** #11560
- **Paths:** see #11560
- **Started:** 2026-10-05
- **Last verified:** 2026-10-05 (`a9f33f14`; collated from changes/11550-dime-manifold-retract-and-local-coordina.md)
- **Summary:** DIME manifold retract and local-coordinate Jacobians are now the exact derivatives of the right quaternion chart at any pose and velocity (J_r^-1, dExp/dv), verified against central finite differences
- **Next step:** Shipped in PR #11560.

### DL-#11525 · CLAUDE.md Is Prettier-Clean: Restored the Mangled Where-to-Edit Code Spans and Spaced the @AGENTS.md Import

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #11525
- **Branch:** merged via #11526
- **PR:** #11526
- **Paths:** see #11526
- **Started:** 2026-10-05
- **Last verified:** 2026-10-05 (`45110b2b`; collated from changes/11525-claude-md-is-prettier-clean-restored-the.md)
- **Summary:** CLAUDE.md is prettier-clean: restored the mangled Where-to-edit code spans and spaced the @AGENTS.md import
- **Next step:** Shipped in PR #11526.

### DL-#1989 · Keep Fork PRs Off the Self-Hosted Fleet: Fork Guard on PR-Triggered Jobs, Hosted Route for the Quality-Gate Lane, and a Static Checker (RM#1989)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #1989
- **Branch:** merged via #11523
- **PR:** #11523
- **Paths:** see #11523
- **Started:** 2026-10-05
- **Last verified:** 2026-10-05 (`c20fb0b7`; collated from changes/1989-fork-pr-self-hosted-guard.md)
- **Summary:** Keep fork PRs off the self-hosted fleet: fork guard on PR-triggered jobs, hosted route for the quality-gate lane, and a static checker (RM#1989)
- **Next step:** Shipped in PR #11523.

### DL-#9700 · Deferred External Validation Plans

- **State:** shipped
- **Owner:** codex (session `codex-validation-planning-20260922-ud`)
- **Issue:** #9700; #10375; #10382; #9619; #9613; #9546
- **Branch:** `docs/deferred-validation-planning`
- **PR:** #10741 (merged 2026-09-22; SHA 60e63ae114e1841fed9ece43bf1dd42895e94fcd)
- **Paths:** `docs/development/planning/`, `docs/development/HANDOFF.md`, `AGENT_HANDOFF.md`
- **Started:** 2026-09-22
- **Shipped:** 2026-09-22
- **Last verified:** 2026-10-11 (e68baeabde8ff639f846a167ca7703ace43bd0a9; source/path inspection only, no tests or experiments rerun)
- **Summary:** Delivered six external-validation planning documents only: DV-10375, DV-10382, DV-9619, DV-9613, DV-9546 and DV-9700. Current source retains the six deferred plan records; no experiments, approvals, provider changes or scientific results are claimed.
- **Next step:** Keep parent epic #9700 open. Evidence and approval for execution scopes #10375, #10382, #9619, #9613 and #9546 remain catalog-deferred; #9546 was historically closed via #10446, and that GitHub closure does not establish acceptance. Do not reopen #9546 or treat planning documents as execution results.

### DL-#9349 · ADR-0046 G2 Workbench Re-Point Closure

- **State:** shipped
- **Owner:** not recorded in source
- **Issue:** #9349 (ADR-0046 Stage 2; module retirement landed under #9348)
- **Branch:** conductor/issue-9349
- **PR:** #10178
- **Paths:** src/config/launcher_manifest.json, src/config/models.yaml, tests/config/launcher_manifest/test_launch_monitor_tiles_share_one_engine.py, docs/adr/0046-launch-monitor-analytics-single-model-layer.md, ui/public/capability-atlas, docs/architecture/CAPABILITY_ATLAS.md
- **Started:** 2026-09-14
- **Shipped:** 2026-09-23
- **Last verified:** 2026-09-14 (SELF; new manifest test 4 pass; test_canonical_layer_parity.py and tests/ui/tools/launch_monitor pass against the vendored canonical layer at pin e83bd2e4; capability atlas regenerated)
- **Summary:** Closes the last Stage 2 deliverable this repository owns: both launch-monitor tiles (UD workbench and Rate of Closure Impact Explorer) now state the "same analytics engine" relationship in models.yaml (desktop launcher) and launcher_manifest.json (web launcher), pinned by a test; ADR-0046 follow-ups record G2 as landed. No workbench code changed — both UIs keep their identity.
- **Next step:** No additional #9349 software delivery remains. This record does not claim workbench behavior or physical qualification beyond the documented manifest/test change.
- **Evidence:** tests/config/launcher_manifest/test_launch_monitor_tiles_share_one_engine.py; tests/unit/launch_monitor/test_canonical_layer_parity.py.

### DL-#10799 · Fail-Closed Qualification for Unsigned Legacy Evidence & Dynamic Inertia Digest (#10799, #10800)

- **State:** shipped
- **Owner:** antigravity
- **Issue:** #10799, #10800
- **Branch:** `fix/10799-10800-qualification-integrity`
- **Paths:** `src/shared/python/tour_baselines/qualification.py`; `src/engines/physics_engines/pendulum/python/motion_matching/qualification.py`; `tests/unit/tour_baselines/test_qualification.py`; `SPEC.md`
- **Started:** 2026-09-24
- **Shipped:** 2026-09-24
- **Last verified:** 2026-10-11 (`6ad8b987`; collated from verified merged PR metadata)
- **Summary:** Remediated bot review feedback on PR #10798:
  1. Kept qualification strictly fail-closed for unsigned legacy evidence (#10799): removed silent auto-migration from `IndependentBaselineQualifier.qualify()`, ensuring packages without recorded controls, time, or identity hashes are rejected at the integrity gate with `IntegrityViolation`. Explicit `migrate_legacy_package` compatibility operations leave package status as `UNVERIFIED` with `has_native_replay=False` rather than silently passing qualification.
  2. Built `fixed_inertia_hash` from actual dynamics parameters (#10800): introduced `compute_pendulum_inertia_hash` in pendulum qualification to digest segment masses, center-of-mass ratios, and rotational inertias from the actual `DoublePendulumDynamics` rollout parameters, ensuring changes to dynamics physics invalidate the inertia digest even when link lengths remain identical.
- **Next step:** PR #10841 merged 2026-09-24 (6ad8b9877f1c6280f7d424c5b890ab576eb8a4ad), closing #10799 and #10800. This ships the software integrity change only; downstream qualification remains unmet.
- **PR:** #10841

### DL-#10943 · Reconciled Scopes From Duplicate Source Rows

- **State:** shipped
- **Owner:** antigravity (PR #10968; branch `agy/issue-10943`) | antigravity (PR not created; branch `agy/issue-10943`)
- **Issue:** #10943
- **PR:** #10968
- **Branch:** `agy/issue-10943`
- **Paths:** `knowledge/pack.yml`, `knowledge/wizard.yml`, `sidekick.spec`, `scripts/packaging/build_sidekick_binary.py`, `scripts/ci/lod_baseline.txt`, `.github/workflows/wizard-pack.yml`, `.github/WORKFLOWS.md`, `tests/unit/ai/test_drift_wizard.py`, `SPEC.md`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`; `knowledge/wizard.yml`, `knowledge/pack.yml`, `tests/unit/ai/test_drift_wizard.py`, `.github/workflows/wizard-pack.yml`, `.github/WORKFLOWS.md`, `sidekick.spec`
- **Started:** 2026-09-25
- **Summary:** [antigravity (PR #10968; branch `agy/issue-10943`)] Define source catalog knowledge/pack.yml and wizard configuration knowledge/wizard.yml for UpstreamDrift product documentation and reference. Wire Sidekick packaging to compile and bundle the SQLite knowledge pack. Add wizard-pack CI workflow and test suite. Baseline child-copy LOD finding to preserve child-copy contract. [antigravity (PR not created; branch `agy/issue-10943`)] Author product knowledge pack definition and CI packaging for Drift Wizard in Sidekick chat.
- **Next step:** Shipped knowledge-pack and wizard-pack CI delivery via PR #10968 on 2026-09-26 (merge SHA 28b37bd47eb85d977e530e60659c6f54ca0a1a2b). The PR-to-issue association is manually reconciled, not a direct closedBy link: exact PR title/timing, changed-file scope, and current-main ancestry match the canonical delivery. The contradictory duplicate-row claim that the wizard files still needed authoring is preserved in the source snapshot; current merged files resolve it. The baseline child-copy contract remains documented in the shipped scope.
- **Last verified:** 2026-10-11 (`28b37bd4`; collated from manual reconciliation of merged PR #10968 metadata; association is not direct issue-closure evidence)
- **Shipped:** 2026-09-26

### DL-#8766 · Unit-Test-Gate Debt Ledger Burndown

- **State:** shipped
- **Owner:** antigravity
- **Issue:** #8766
- **Branch:** fix/8766-burndown-launcher-67
- **PR:** #10035, #8997
- **Paths:** scripts/config/unit_gate_quarantine.json, SPEC.md, docs/development/DEVELOPMENT_LOG.md, tests/launchers/test_golf_launcher.py, docs/agent_context/
- **Started:** 2026-09-12
- **Last verified:** 2026-10-11 (`fe411260`; collated from verified GitHub merge metadata (PR #8997, 2026-08-23))
- **Shipped:** 2026-08-23
- **Summary:** Progressive burndown of the quarantine ledger (#8766). Prior tranches retired 43 packaging/governance tests (#10010), 11 deployment tests (#10012), 57 bunker shot and API route tests (#10013), 29 shared Python / physics tests (#10015), 16 AI adapter / launcher tests (#10026), 20 safe launcher / pipeline / model sources tests (#10031), 13 CORS tests (#10033), and 32 security and module docstring tests (#10034). This tranche burns down 67 quarantined tests across tests/launchers/test_golf_launcher.py (25), tests/launchers/test_launcher_ui_setup.py (21), tests/launchers/test_launcher_process_manager.py (17), and tests/launchers/test_library_widget.py (4), ratcheting debt down from 298 to 231.
- **Next step:** Shipped via PR #8997 on 2026-08-23 (merge SHA fe411260cab577dd197b54ad3eab4d9d9ffa4213).

### DL-#10334 · Versioned Matched Swing Candidate (MS-15)

- **State:** shipped
- **Owner:** claude
- **Issue:** #10334 (MS-15, epic #10363)
- **Branch:** feat/10334-matched-swing-candidate
- **PR:** #10668, #10475
- **Paths:** src/shared/python/motion_matching/candidate.py; src/shared/python/motion_matching/candidate_io.py; src/shared/python/motion_matching/candidate_convert.py; src/tools/tour_matching_viewer/core.py; src/shared/python/motion_matching/cross_engine_replay.py; docs/development/full_body_models/CANDIDATES.md; tests/unit/motion_matching/test_candidate.py
- **Started:** 2026-09-19
- **Last verified:** 2026-10-11 (`8d452e09`; collated from verified GitHub merge metadata (PR #10475, 2026-09-19))
- **Shipped:** 2026-09-19
- **Summary:** Implemented unified MatchedSwingCandidate versioned package (matched-swing-candidate-v1) supporting distinct kinematic and dynamic profiles, full-body generalized coordinates and tangent velocities (nq != nv), virtual work power consistency verification, immutable arrays, and SHA-256 array tamper detection. Implemented converters for legacy returned81 replays and OpenSim MOT files, updated viewer loader, and authored complete CANDIDATES.md schema document.
- **Next step:** Shipped via PR #10475 on 2026-09-19 (merge SHA 8d452e09f8443b540c0de8fe9cd324f99e76381c).
- **Evidence:** docs/development/full_body_models/CANDIDATES.md; tests/unit/motion_matching/test_candidate.py; docs/development/full_body_models/evidence/replays/mujoco_returned81_candidate.npz; docs/development/full_body_models/evidence/viewer/opensim_os3b_candidate.npz.

### DL-#10323 · Matched-Swing Run Ledger

- **State:** shipped
- **Owner:** local
- **Issue:** #10323 (epic #10363 MS-02)
- **Branch:** feat/ms02-matched-swing-run-ledger-10323
- **PR:** open, #10387
- **Paths:** src/shared/python/motion_matching/ledger.py; src/shared/python/motion_matching/ledger_schema.py; src/shared/python/motion_matching/**main**.py; src/shared/python/motion_matching/leaderboard.py; src/tools/motion_matching/pipeline.py; reports/matched_swing_ledger.json; tests/unit/motion_matching/test_ledger.py
- **Started:** 2026-09-17
- **Last verified:** 2026-10-11 (`94d593cf`; collated from verified GitHub merge metadata (PR #10387, 2026-09-17))
- **Shipped:** 2026-09-17
- **Summary:** Added matched-swing run ledger discovering, classifying, and indexing execution receipts across ground-support, native Simscape, OpenSim, calibration, and parity evidence trees. Deterministic serialization into reports/matched_swing_ledger.json, CLI subcommand `ledger --write`, and `list_runs()` API for tools.
- **Next step:** Shipped via PR #10387 on 2026-09-17 (merge SHA 94d593cf119ffeceaad7658ef0502c9c22e6ea89).
- **Evidence:** reports/matched_swing_ledger.json; tests/unit/motion_matching/test_ledger.py

### DL-#10395 · OpenSim Anatomical Baseline Freeze and Qualification Fixtures

- **State:** shipped
- **Owner:** local
- **Issue:** #10395 (epic #10394 / #10363, OG-01)
- **Branch:** feat/og01-freeze-anatomical-baseline-10395
- **PR:** #10404
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/model_audit.py; src/engines/physics_engines/opensim/python/tour_matching/**init**.py; src/engines/physics_engines/opensim/python/tour_matching/cli.py; tests/opensim/test_anatomical_baseline_fixtures.py; tests/opensim/test_opensim_os0_qualification.py
- **Started:** 2026-09-18
- **Last verified:** 2026-10-11 (`1d302ba8`; collated from verified GitHub merge metadata (PR #10404, 2026-09-18))
- **Shipped:** 2026-09-18
- **Summary:** Delivered pure-Python OpenSim model geometry and structural qualification audit under OG-01. Verifies SHA-256 digests against pinned baseline models, audits body/coordinate/actuator counts (23/39/39/0), detects empty attached_geometry on equipment bodies (Club body) and unscaled arm meshes (scale factors 1 1 1 on humerus). Implements fail-closed `verify_model_qualification` gate with DbC assertions and structured receipts via `cli.py qualify`.
- **Next step:** Shipped via PR #10404 on 2026-09-18 (merge SHA 1d302ba81e5f63dbc88112e2e80c088f09adcdfc).
- **Evidence:** tests/opensim/test_anatomical_baseline_fixtures.py; tests/opensim/test_opensim_os0_qualification.py; docs/development/opensim_tour_matching/evidence/anatomical_review_20260918/inspection.json.

### DL-#10397 · OpenSim Visible Parameterized Golf Club and Grip Frames

- **State:** shipped
- **Owner:** local
- **Issue:** #10397 (epic #10394 / #10363, OG-03)
- **Branch:** feat/og03-visible-golf-club-10397
- **PR:** #10405, #10414
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/club_geometry.py; src/engines/physics_engines/opensim/python/tour_matching/**init**.py; tests/opensim/test_golf_club_geometry.py; SPEC.md
- **Started:** 2026-09-18
- **Last verified:** 2026-10-11 (`9091b3de`; collated from verified GitHub merge metadata (PR #10414, 2026-09-18))
- **Shipped:** 2026-09-18
- **Summary:** Added parameterized visual geometry attachment (`attach_visual_club`) and frame calculation (`get_club_frame_offsets`) for the OpenSim Club body according to shared ClubSpec (Driver and 7-iron). Solves the missing visual club defect while strictly preserving physical mass properties (0.320 kg, COM, inertia). Fully verified against anatomical baseline fixtures and qualification gates.
- **Next step:** Shipped via PR #10414 on 2026-09-18 (merge SHA 9091b3deebd90df90f400e089c32bd29d2100016).
- **Evidence:** tests/opensim/test_golf_club_geometry.py; tests/opensim/test_anatomical_baseline_fixtures.py.

### DL-#10401 · OpenSim Introduce Versioned Model Variants and Actuation Capabilities

- **State:** shipped
- **Owner:** local
- **Issue:** #10401 (epic #10394 / #10363, OG-07)
- **Branch:** feat/og07-versioned-model-variants-10401
- **PR:** #10412, #10414
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/model_variants.py; src/engines/physics_engines/opensim/python/tour_matching/**init**.py; tests/opensim/test_golf_model_variants.py; docs/development/DEVELOPMENT_LOG.md; docs/development/HANDOFF.md
- **Started:** 2026-09-18
- **Last verified:** 2026-10-11 (`9091b3de`; collated from verified GitHub merge metadata (PR #10414, 2026-09-18))
- **Shipped:** 2026-09-18
- **Summary:** Implemented versioned OpenSim model variants and explicit actuation capabilities (`model_variants.py`) via composition: AnatomicalSkeletonSpec + GolfEquipmentSpec + Calibration + ActuationProfile (torque vs. muscle/tendon). Provides typed error boundaries and a clean adapter API without SDK leakage.
- **Next step:** Shipped via PR #10414 on 2026-09-18 (merge SHA 9091b3deebd90df90f400e089c32bd29d2100016).
- **Evidence:** tests/opensim/test_golf_model_variants.py; src/engines/physics_engines/opensim/python/tour_matching/model_variants.py.

### DL-#10403 · OpenSim Package a Golf-Like Native Viewer and Release Evidence

- **State:** shipped
- **Owner:** local
- **Issue:** #10403 (epic #10394 / #10363, OG-09)
- **Branch:** feat/og09-golf-native-viewer-package-10403
- **PR:** #10668, #10414
- **Paths:** src/engines/physics_engines/opensim/python/tour_matching/view_package.py; src/engines/physics_engines/opensim/python/tour_matching/**init**.py; tests/opensim/test_golf_view_package.py; docs/development/DEVELOPMENT_LOG.md; docs/development/HANDOFF.md
- **Started:** 2026-09-18
- **Last verified:** 2026-10-11 (`9091b3de`; collated from verified GitHub merge metadata (PR #10414, 2026-09-18))
- **Shipped:** 2026-09-18
- **Summary:** Packaged native viewer artifacts and release evidence (`view_package.py`) for the OpenSim golf humanoid. Provides visual layer options, truth-in-advertising muscle toggles, swing milestone stills, reproducible video animation export, reset-to-address with grip closure verification, continuous scrubbing, launcher entry generation, and deterministic package hashing. Completes all 9 child issues of OpenSim epic #10394.
- **Next step:** Shipped via PR #10414 on 2026-09-18 (merge SHA 9091b3deebd90df90f400e089c32bd29d2100016). Preserved limitation: Viewer completion does not establish muscle qualification.
- **Evidence:** tests/opensim/test_golf_view_package.py; src/engines/physics_engines/opensim/python/tour_matching/view_package.py.

### DL-#10233 · Shadow Tracker Revision Integrity and Persistence

- **State:** shipped
- **Owner:** codex
- **Issue:** #10233 (ST-04 / epic #10122)
- **Branch:** fix/shadow-tracker-10233-pr
- **PR:** #10450 (https://github.com/D-sorganization/UpstreamDrift/pull/10450), #10283
- **Paths:** src/shared/python/shadow_tracker/segmentation.py; tests/unit/shadow_tracker/test_revision_persistence.py; docs/plans/shadow_tracker/
- **Started:** 2026-09-18
- **Last verified:** 2026-10-11 (`dd243d39`; collated from verified GitHub merge metadata (PR #10283, 2026-09-17))
- **Shipped:** 2026-09-17
- **Summary:** Complete-record idempotence, same-observation parents, current selection and strict atomic provider snapshots with legacy reading. Renderer preserved.
- **Next step:** Shipped via PR #10283 on 2026-09-17 (merge SHA dd243d392232003b15db389d54d962cffe6fad14).
- **Evidence:** docs/plans/shadow_tracker/TURNOVER_CURRENT.md; tests/unit/shadow_tracker/test_revision_persistence.py.

### DL-#10460 · Consume Shared GSPro Open Connect V1 Codec From Tools

- **State:** shipped
- **Owner:** local
- **Issue:** #10460
- **Branch:** feat/10460-consume-tools-gspro-codec
- **PR:** #10668, #10463
- **Paths:** src/shared/python/golf_simulator/adapters/gspro/codec.py; tests/unit/golf_simulator/test_gspro_codec.py; vendor/ud-tools; Cargo.toml; requirements-tools.txt; docs/shared_tools/divergence_inventory.v1.json; docs/shared_tools/divergence_inventory.md
- **Started:** 2026-09-18
- **Last verified:** 2026-10-11 (`db31d1f3`; collated from verified GitHub merge metadata (PR #10463, 2026-09-19))
- **Shipped:** 2026-09-19
- **Summary:** Upgraded vendor/ud-tools pin to Tools commit a9ed0e7c5c6905b1164082659051d6381068052d carrying shared GSPro Open Connect v1 codec (Tools#5228). Refactored UpstreamDrift's gspro adapter codec to retain ShotEnvelope canonical SI/radian unit conversion and profile handling, but delegate wire payload encoding and response decoding to shared.python.launch_monitor.gspro_connect.
- **Next step:** Shipped via PR #10463 on 2026-09-19 (merge SHA db31d1f364903c4bb0451b049f3e9e98a3246b1f).
- **Evidence:** tests/unit/golf_simulator/test_gspro_codec.py.

### DL-#10482 · Expose Real Forces, Torques, and Explicit Counterfactual Semantics

- **State:** shipped
- **Owner:** local
- **Issue:** #10482 (MV-06, epic #10476)
- **Branch:** feat/10482-forces-torques-counterfactual
- **PR:** #10504, #10505
- **Paths:** src/api/models/requests.py; src/api/routes/analysis.py; src/api/services/simulation_service.py; src/shared/python/motion_matching/candidate_session.py; src/shared/python/motion_matching/counterfactual.py; src/shared/python/motion_matching/force_torque.py; src/tools/tour_matching_viewer/force_inspection.py; src/tools/tour_matching_viewer/gui.py; tests/unit/api/test_candidate_session_analysis_routes.py; tests/unit/motion_matching/test_candidate_session_forces.py; tests/unit/motion_matching/test_counterfactual.py; tests/unit/motion_matching/test_force_torque.py; tests/unit/tools/test_tour_matching_viewer_forces.py
- **Started:** 2026-09-19
- **Last verified:** 2026-10-11 (`4cfdba7c`; collated from verified GitHub merge metadata (PR #10505, 2026-09-20))
- **Shipped:** 2026-09-20
- **Summary:** Implemented `SpatialWrench` and rigid-body `transform_wrench` with rotational mapping and cross-product moment arm calculation ($\tau_B = R \tau_A + r \times (R F_A)$). Implemented `compute_center_of_pressure` with strict threshold semantics ($F_z \le 5.0$ N returns `None` rather than fabricated zeros). Implemented `AccelerationDecomposition` (gravity, drift, control, ZTCF, ZVCF) and `create_counterfactual_rollout` with cryptographic baseline immutability assertion (SHA-256 byte check before and after execution). Enhanced `CandidateSession` with `get_wrench_at`, `get_center_of_pressure`, `get_joint_torques_at`, `get_closure_residual_at`, and `create_counterfactual_fork`. Added API endpoints `GET /analysis/candidate/forces` and `POST /analysis/candidate/counterfactual` (failing closed with 409 Conflict when session is absent or kinematic-only). Integrated `ForceInspectionWidget` into Tour Matching Viewer GUI synchronized with physical playback time.
- **Next step:** Shipped via PR #10505 on 2026-09-20 (merge SHA 4cfdba7ce448ee0d236c1d8082052c10f4f3ad07).
- **Evidence:** tests/unit/motion_matching/test_force_torque.py; tests/unit/motion_matching/test_counterfactual.py; tests/unit/motion_matching/test_candidate_session_forces.py; tests/unit/api/test_candidate_session_analysis_routes.py; tests/unit/tools/test_tour_matching_viewer_forces.py.

### DL-#10514 · Group Engine Dashboards, Exercise Variants, and Repository Shortcuts

- **State:** shipped
- **Owner:** local
- **Issue:** #10514 (ORG-07, epic #10508)
- **Branch:** feat/issue-10514-org07-model-variant-grouping
- **PR:** #10541
- **Paths:** src/config/models.yaml; src/launchers/exercise_dashboard.py; src/launchers/launcher_model_handlers.py; src/shared/python/config/**init**.py; src/shared/python/config/model_pack_manifest.py; src/shared/python/config/model_registry.py; src/shared/python/config/model_variant_grouping.py; tests/config/test_model_variant_grouping.py
- **Started:** 2026-09-19
- **Last verified:** 2026-10-11 (`3ef6f91c`; collated from verified GitHub merge metadata (PR #10541, 2026-09-20))
- **Shipped:** 2026-09-20
- **Summary:** Implemented `ModelVariant`, `LogicalModelIdentity`, `LogicalModelChoice`, and `ModelGroupingProjection` in `src/shared/python/config/model_variant_grouping.py` to project 28 exercise variants across 4 providers (`MuJoCo_Models`, `Drake_Models`, `Pinocchio_Models`, `OpenSim_Models`) into 7 logical choices without dropping any provider assets underneath. Preserved strict DbC on engine selection (never silently substitutes another engine). Added name collision protection across distinct canonical identities. Implemented `resolve_shortcut` mapping `biomech_sit_to_stand` (never falls back to gait) and `biomech_gait` to `biomech_exercise`, engine dashboards (`drake_dashboard`, `mujoco_dashboard`, `pinocchio_dashboard`) to engine advanced modes, and `movement_optimizer` / `tools_movement_optimizer` to unified task with #9406 authority resolution. Updated `SharedRepoHandler` with `get_missing_checkout_diagnostic` to emit actionable diagnostics when sibling repos are missing. Dynamicized exercise names in `exercise_dashboard.py`.
- **Next step:** Shipped via PR #10541 on 2026-09-20 (merge SHA 3ef6f91c33057fb1e6dd4817996e82bc834d8299).
- **Evidence:** tests/config/test_model_variant_grouping.py; tests/config/test_tile_paths_resolve.py; tests/unit/config/test_model_pack_manifest.py; tests/launchers/test_launcher_model_handlers.py.

### DL-#10517 · Unified Artifact and Project Context Handoff Between Workspaces

- **State:** shipped
- **Owner:** local
- **Issue:** #10517 (ORG-08, epic #10508)
- **Branch:** feat/issue-10517-org08-workspace-handoff
- **PR:** #10542
- **Paths:** src/shared/python/workspace/**init**.py; src/shared/python/workspace/artifact_handoff.py; src/shared/python/workspace/project_store.py; tests/unit/workspace/test_artifact_handoff.py
- **Started:** 2026-09-19
- **Last verified:** 2026-10-11 (`2a854627`; collated from verified GitHub merge metadata (PR #10542, 2026-09-20))
- **Shipped:** 2026-09-20
- **Summary:** Extended SessionProjectStore and ProjectMetadata with typed, versioned artifact handoffs (WorkspaceHandoff, ArtifactReference, ArtifactKind, RunMetadata). Enforced Design-by-Contract boundary preconditions (cross-session subject mismatch rejection, frame and schema validation, artifact existence and cryptographic sha256 hash checks before disk write, canceled/failed job qualification invariant). Added migration handling preserving unknown supported fields in older project.json files, atomic durability under interrupted writes, active context selection, and non-destructive run cloning without falsified output evidence. Implemented registered named artifact adapter conversion preserving provenance.
- **Next step:** Shipped via PR #10542 on 2026-09-20 (merge SHA 2a854627054fb272167d8de67504c4fe25ebf71f).
- **Evidence:** tests/unit/workspace/test_artifact_handoff.py; tests/unit/workspace/test_project_store.py; tests/unit/workspace/test_results_browser.py.

### DL-#10518 · Guided Workflow Transitions Across Unified Workspaces

- **State:** shipped
- **Owner:** local
- **Issue:** #10518 (ORG-09, epic #10508)
- **Branch:** feat/issue-10518-org09-workflow-transitions
- **PR:** #10544
- **Paths:** src/shared/python/workspace/**init**.py; src/shared/python/workspace/workflow_coordinator.py; tests/unit/workspace/test_workflow_transitions.py
- **Started:** 2026-09-19
- **Last verified:** 2026-10-11 (`39092e54`; collated from verified GitHub merge metadata (PR #10544, 2026-09-20))
- **Shipped:** 2026-09-20
- **Summary:** Implemented WorkflowCoordinator coordinating the 7-stage workflow pipeline (`Capture/Import -> Inspect Targets -> Configure Model -> Fit -> Dynamics -> Compare -> Export`) over typed ArtifactReference inputs and outputs. Evaluates live step readiness and diagnostics directly from cryptographic sha256 hashes and on-disk artifact existence rather than superficial flags. Added support for single-view coaching mode skipping 3-D dynamics when physics engines are unavailable, enforced contract distinction preventing dynamics from inheriting purely kinematic passes, tracked cancellation reasons and retry attempts, supported later-stage entry from imported artifacts, and exposed a pure state projection with to_dict() for Qt and React/Tauri parity.
- **Next step:** Shipped via PR #10544 on 2026-09-20 (merge SHA 39092e54caccd83edc00e514cade2014f25c866c).
- **Evidence:** tests/unit/workspace/test_workflow_transitions.py; tests/unit/workspace/test_artifact_handoff.py; tests/unit/workspace/test_project_store.py; tests/unit/workspace/test_results_browser.py.

### DL-#10519 · Connect Capture Rig, Optical Import, Pose Inspection, and Model Calibration Workspaces

- **State:** shipped
- **Owner:** local
- **Issue:** #10519 (ORG-10, epic #10508)
- **Branch:** feat/issue-10519-org10-capture-inspection-handoff
- **PR:** #10545
- **Paths:** src/shared/python/workspace/**init**.py; src/shared/python/workspace/capture_inspection_handoff.py; src/tools/capture_rig/gui.py; src/tools/capture_rig/journey_actions.py; tests/integration/test_capture_target_handoff.py
- **Started:** 2026-09-19
- **Last verified:** 2026-10-11 (`15bce228`; collated from verified GitHub merge metadata (PR #10545, 2026-09-20))
- **Shipped:** 2026-09-20
- **Summary:** Implemented `CaptureInspectionHandoff` and `FreeMoCapJobAdapter` bridging Capture Rig to Inspect Targets and Model Calibration. Enforced that trim, crop, and time offsets survive handoffs with explicit clock calculations; maintained MediaPipe and OpenPose as explicit estimator choices with separate observation sets, confidence scores, and source pixels; provided FreeMoCap CLI input/output validation before process spawn and cancellation leaving sources untouched with preserved HMR2/AGPL license isolation; opened C3D and optical imports keeping missing samples masked (NaN) and rejecting incompatible units and frames; rejected pretending 2-D coordinates are metric 3-D; and registered targets into `SessionProjectStore` automatically preserving annotations, calibration, and club metadata. Added "Open in Inspect Targets" action to Capture Rig GUI and JourneyActions.
- **Next step:** Shipped via PR #10545 on 2026-09-20 (merge SHA 15bce2283fb6954b2b285585d370b0b7d03ed545).
- **Evidence:** tests/integration/test_capture_target_handoff.py; tests/tools/capture_rig/test_workflow.py; tests/unit/workspace/test_workflow_transitions.py.

### DL-#10520 · Move Tour Matching Execution Out of Documentation Without Changing Results

- **State:** shipped
- **Owner:** local
- **Issue:** #10520 (ORG-11, epic #10508)
- **Branch:** feat/issue-10520-org11-motion-matching-packaging
- **PR:** #10546
- **Paths:** src/shared/python/motion_matching/execution/**init**.py; src/shared/python/motion_matching/execution/assets.py; src/shared/python/motion_matching/execution/spec_builder.py; src/shared/python/motion_matching/execution/downswing.py; src/shared/python/motion_matching/execution/mjx_export.py; src/shared/python/motion_matching/execution/driver.py; src/tools/motion_matching/pipeline.py; docs/development/full_body_models/build_anthropometric_spec.py; docs/development/full_body_models/evidence/ground_support/run_ground_support.py; docs/development/full_body_models/evidence/ground_support/downswing_experiment.py; docs/development/full_body_models/evidence/ground_support/export_mjx_package.py; tests/integration/test_installed_motion_matching.py
- **Started:** 2026-09-19
- **Last verified:** 2026-10-11 (`2715f5c8`; collated from verified GitHub merge metadata (PR #10546, 2026-09-20))
- **Shipped:** 2026-09-20
- **Summary:** Extracted reusable execution out of documentation into `src/shared/python/motion_matching/execution/` (`spec_builder.py`, `downswing.py`, `mjx_export.py`, `driver.py`, `assets.py`). Preserved algorithms, numerical precision, parameters, coordinate frames, and output schemas. Resolved reference assets through standard resource resolution functions (`get_native_geometry_spec`, `get_opensim_model`, `get_candidate_geometry_spec`, `get_capture_c3d`, `resolve_output_root`) with environment variable overrides and clear error explanations for unavailable assets. Kept legacy script paths in `docs/development/full_body_models/` as thin compatibility wrappers issuing `DeprecationWarning` while delegating to packaged entry points and preserving CLI schemas and exit codes. Updated `pipeline.py` command constants (`BUILDER`, `DRIVER_SCRIPT`, `DOWNSWING_SCRIPT`, `EXPORT_MJX_SCRIPT`) to point to packaged entry points and write outputs outside docs/package.
- **Next step:** Shipped via PR #10546 on 2026-09-20 (merge SHA 2715f5c87834c8efec84795e149cc69525cee489).
- **Evidence:** tests/integration/test_installed_motion_matching.py; tests/tools/motion_matching/test_pipeline.py; tests/tools/motion_matching/test_motion_matching_gui.py.

### DL-#10529 · Consume Provider Ownership Decisions and Verify Runtime Import Authority

- **State:** shipped
- **Owner:** local
- **Issue:** #10529 (ORG-20, epic #10508)
- **Branch:** feat/issue-10529-org20-provider-ownership
- **Paths:** src/shared/python/config/tools_vendor_authority.py; tests/integration/test_installed_provider_authority.py
- **Started:** 2026-09-20
- **Last verified:** 2026-10-11 (`57ed7523`; collated from verified GitHub merge metadata (PR #10572, 2026-09-20))
- **Shipped:** 2026-09-20
- **Summary:** Consumed provider ownership decisions and implemented runtime import authority and provenance verification across repository, installed, and packaged execution environments. Added `assert_runtime_provenance_parity` failing closed upon root divergence between pytest and packaged app contexts. Added `verify_provider_provenance` asserting module paths resolve within canonical provider roots. Implemented `inspect_provider_authority` handling pinned gitlinks, clean installed wheel distributions (`ud-tools`), and probe import failures without silent fallback. Verified public seams for Sidekick, Movement Optimizer (`tools_movement_optimizer` via `ALIAS_MAP`), Pendulum (`swing_objective_lab`), and backward compatibility import delegation (`upstream_drift_tools` -> `sidekick`).
- **Next step:** Shipped via PR #10572 on 2026-09-20 (merge SHA 57ed7523df51627b2c66b9e89c502a97efe44d3b).
- **Evidence:** tests/integration/test_installed_provider_authority.py.
- **PR:** #10572

### DL-#10358 · Web Matched-Swing Results API and Results Page

- **State:** shipped
- **Owner:** local
- **Issue:** #10358 (MS-85, epic #10363)
- **Branch:** fix/issue-10358-ms-85-local
- **Paths:** src/api/routes/matched_swings.py; src/api/services/matched_swings_service.py; tests/api/test_matched_swings.py; ui/src/pages/MatchedSwings.tsx; ui/src/api/matchedSwings.ts; ui/src/App.tsx; src/config/launcher_manifest.json; src/api/route_registry.py
- **Started:** 2026-09-21
- **Last verified:** 2026-10-11 (`e60228fa`; collated from verified GitHub merge metadata (PR #10664, 2026-09-21))
- **Shipped:** 2026-09-21
- **Summary:** Added local-only read-only `/api/matched-swings` routes (ledger, receipt, candidate NPZ/preview, parity, GIF) keyed by receipt SHA-256 without leaking absolute paths. Web Results page at `/tools/matched-swings` lists runs with verdict badges, GIF playback, and MocapSkeleton3D marker preview; mounted CrossEngineDashboard at `/tools/cross-engine` with launcher manifest web routes for both tiles.
- **Next step:** Shipped via PR #10664 on 2026-09-21 (merge SHA e60228fadfda763f10c5f4e095172bbca8f777e8).
- **Evidence:** tests/api/test_matched_swings.py; ui/src/pages/MatchedSwings.test.tsx.
- **PR:** #10664

### DL-#10353 · Results Browser Tile for Matched Swing Program

- **State:** shipped
- **Owner:** claude
- **Issue:** #10353 (MS-80, epic #10363)
- **Branch:** feat/10353-results-browser
- **PR:** #10543
- **Paths:** src/tools/matched_swing_browser/**init**.py; src/tools/matched_swing_browser/**main**.py; src/tools/matched_swing_browser/gui.py; src/tools/matched_swing_browser/model.py; src/tools/matched_swing_browser/\_embed_adapter.py; src/config/models.yaml; src/config/launcher_manifest.json; src/config/feature_parity.json; src/launchers/embedded_tool_bootstrap.py; pyproject.toml; tests/tools/matched_swing_browser/test_matched_swing_browser_gui.py; tests/tools/matched_swing_browser/test_model.py; docs/development/matched_swing_program/evidence/browser/screenshot.png
- **Started:** 2026-09-20
- **Last verified:** 2026-10-11 (`a1c13ed0`; collated from verified GitHub merge metadata (PR #10543, 2026-09-20))
- **Shipped:** 2026-09-20
- **Summary:** Implemented `matched_swing_browser` embeddable PyQt6 desktop tool and data model. Left pane provides filterable table over 98 ledger receipts (by engine, capture, lane, verdict, search text). Right pane displays receipt summary, acceptance badge, five standardized metrics with units, physical gates breakdown, and lazy-loaded animated QMovie playback for rows with visual GIF artifacts. Provides action buttons to launch Tour Matching Viewer, Native Viewer (MS-83), and parity reports. Reuses `ResultFilter` lineage resolving #8824 and establishing contract for #10521 (ORG-13). Registered across all 5 canonical surfaces (`models.yaml`, `launcher_manifest.json`, `pyproject.toml`, `embedded_tool_bootstrap.py`, `feature_parity.json`).
- **Next step:** Shipped via PR #10543 on 2026-09-20 (merge SHA a1c13ed0d56fd622f138961477651310ea254b13).
- **Evidence:** tests/tools/matched_swing_browser/test_model.py; tests/tools/matched_swing_browser/test_matched_swing_browser_gui.py; docs/development/matched_swing_program/evidence/browser/screenshot.png.

### DL-#10531 · Generate Accurate Atlas, Help, Parity, and Completion Records

- **State:** shipped
- **Owner:** local
- **Issue:** #10531 (ORG-21, epic #10508)
- **Branch:** `feat/issue-10531-org21-accurate-atlas-parity`
- **PR:** #10576
- **Paths:** `tests/scripts/test_workspace_documentation_freshness.py`, `src/config/industrial_readiness.json`, `docs/operations/industrial-readiness-index.md`, `src/tools/training_controller/README.md`
- **Started:** 2026-09-20
- **Last verified:** 2026-10-11 (`777033b5`; collated from verified GitHub merge metadata (PR #10576, 2026-09-20))
- **Shipped:** 2026-09-20
- **Summary:** Implemented comprehensive regression tests in `test_workspace_documentation_freshness.py` covering workspace membership drift, undocumented/dangling aliases, broken source/help links, stale generated views, shell-only parity vs compute-complete separation, deterministic generators, training controller README accuracy, and reconciled industrial readiness item U3 (#8820 / PR #9995) with merge SHA and verified implementation/test paths.
- **Next step:** Shipped via PR #10576 on 2026-09-20 (merge SHA 777033b5ef335c6c8b27ce568018b06de5a17860).
- **Evidence:** `tests/scripts/test_workspace_documentation_freshness.py`, `tests/scripts/test_capability_atlas.py`.

### DL-#10522 · Connect Subject, Club, Model, Pose, Fit, and Dynamics Stages

- **State:** shipped
- **Owner:** local
- **Issue:** #10522 (ORG-12, epic #10508)
- **Branch:** feat/issue-10522-org12-model-match-handoff
- **PR:** #10547
- **Paths:** src/shared/python/workspace/**init**.py; src/shared/python/workspace/model_match_handoff.py; tests/integration/test_model_match_handoff.py
- **Started:** 2026-09-19
- **Last verified:** 2026-10-11 (`81d32c30`; collated from verified GitHub merge metadata (PR #10547, 2026-09-20))
- **Shipped:** 2026-09-20
- **Summary:** Implemented task adapters over existing public APIs for model selection/generation, subject parameters, club snapshot, and initial canonical pose bound to active project/session (`SessionProjectStore`). Separated general-input motion pipeline and tour driver/7-iron matching routes with explicit routing refusing unsupported arbitrary video observations. Passed validated target/model/pose references into fit jobs and recorded outputs/receipts in `SessionProjectStore`. Exposed Fit Kinematics and Run Dynamics as distinct steps with explicit backend choices across all 6 engines (`mujoco`, `drake`, `pinocchio`, `opensim`, `myosuite`, `simscape`) without silent substitutions. Enforced that kinematic outputs cannot be marked as dynamic qualified. Implemented downstream state invalidation on model change, failed-fit diagnostics, cancellation preserving prior runs, and reopen descriptor linking to Results/Replay seam.
- **Next step:** Shipped via PR #10547 on 2026-09-20 (merge SHA 81d32c3057c6ad8c4977e803df389031d9a789fb).
- **Evidence:** tests/integration/test_model_match_handoff.py; tests/unit/workspace/test_workflow_transitions.py; tests/unit/workspace/test_artifact_handoff.py.

### DL-#10523 · Connect Swing, Impact, Flight, and Preserved Trajectory Viewers

- **State:** shipped
- **Owner:** local
- **Issue:** #10523 (ORG-14, epic #10508)
- **Branch:** feat/issue-10523-org14-trajectory-viewers
- **PR:** #10560
- **Paths:** `src/shared/python/workspace/trajectory_handoff.py`; `src/shared/python/workspace/results_workspace.py`; `src/shared/python/workspace/__init__.py`; `src/shared/python/physics/flight_trajectory_export.py`; `src/shared/python/physics/swing_state_providers.py`; `tests/integration/test_shot_trajectory_handoff.py`
- **Started:** 2026-09-19
- **Last verified:** 2026-10-11 (`2da04559`; collated from verified GitHub merge metadata (PR #10560, 2026-09-20))
- **Shipped:** 2026-09-20
- **Summary:** Implemented `ShotTrajectoryHandoffCoordinator` connecting swing-state providers, impact solvers, and aerodynamic ball flight simulation with specialized viewers per ADR-0047. Preserves viewer identities (Shot Tracer Qt, web BallFlight, and Impact Explorer ROC) without retiring viewers or merging distinct flight model families (`ud.flight_models` vs `swing_sim.flight`). Bridges `PipelineResult` to `swing_sim.ball_flight_trajectory/1` wire contract with immutable SI sample positions and timestamps. Enforces honest engine sourcing with fail-closed diagnostics (`UnsupportedEngineSourceError`, `ExtractionAdapterError`, `FrameUnitMismatchError`, `InvalidTrajectoryHashError`). Extends ResultsWorkspace with `COMPARE_FLIGHT_MODELS` and `OPEN_IN_IMPACT_EXPLORER` actions and provides atomic transaction staging/rollback.
- **Next step:** Shipped via PR #10560 on 2026-09-20 (merge SHA 2da04559b07b87c73c2a46e41d4134fe12e9088a).
- **Evidence:** tests/integration/test_shot_trajectory_handoff.py.

### DL-#10524 · Compose Terrain, Putting, Scene, Bunker, and Simulator Delivery Modes

- **State:** shipped
- **Owner:** local
- **Issue:** #10524 (ORG-15, epic #10508)
- **Branch:** feat/issue-10524-org15-scene-delivery-modes
- **PR:** #10561
- **Paths:** src/shared/python/workspace/shot_course_workspace.py; src/shared/python/workspace/**init**.py; tests/integration/test_shot_course_workspace.py
- **Started:** 2026-09-20
- **Last verified:** 2026-10-11 (`41618f3a`; collated from verified GitHub merge metadata (PR #10561, 2026-09-20))
- **Shipped:** 2026-09-20
- **Summary:** Implemented `ShotCourseWorkspaceCoordinator` composing Terrain, Putting, Scene, Bunker, and Simulator Delivery modes per ADR-0047 and issue #10524. Enforced explicit model boundaries: scene view is visual inspection only; bunker preserves F0-F3 fidelity tiers; putting conforms to rolling/ground contracts; terrain mutation increments revision and invalidates prior runs; simulator delivery verifies destination capabilities and produces explicit submission receipts.
- **Next step:** Shipped via PR #10561 on 2026-09-20 (merge SHA 41618f3a41c78f58b111c64e57efdfe20c64813e).
- **Evidence:** tests/integration/test_shot_course_workspace.py

### DL-#10525 · Consolidate Optimization and Training Launchers Under Shared Project Workspace and Controller Authority

- **State:** shipped
- **Owner:** local
- **Issue:** #10525 (ORG-16, epic #10508)
- **Branch:** feat/issue-10525-org16-optimization-training
- **PR:** #10563
- **Paths:** src/shared/python/workspace/optimization_training_workspace.py; src/shared/python/workspace/**init**.py; tests/integration/test_optimization_training_workspace.py; docs/development/HANDOFF.md; docs/development/DEVELOPMENT_LOG.md; SPEC.md
- **Started:** 2026-09-20
- **Last verified:** 2026-10-11 (`5b1e5757`; collated from verified GitHub merge metadata (PR #10563, 2026-09-20))
- **Shipped:** 2026-09-20
- **Summary:** Implemented `OptimizationTrainingWorkspaceCoordinator` consolidating optimization and training under shared workspace and scheduler authority. Bounded job form over the public optimizer and training controller authority, validating objectives, constraints, and model compatibility prior to dispatch. Enforced fail-closed handling for unsupported/uninstalled backends. Enforced cancel/pause/resume lifecycle invariants and deduplication of active submissions. Connected dataset selection with provenance directly to durable project sessions in `SessionProjectStore`.
- **Next step:** Shipped via PR #10563 on 2026-09-20 (merge SHA 5b1e57578886f5dd52875786209b1f811b1d3c53).
- **Evidence:** tests/integration/test_optimization_training_workspace.py; src/shared/python/workspace/optimization_training_workspace.py.

### DL-#10528 · Unify Sidekick, Setup, Help, and Library as Global Utilities

- **State:** shipped
- **Owner:** local
- **Issue:** #10528 (ORG-19, epic #10508)
- **Branch:** feat/issue-10528-org19-global-utilities
- **PR:** #10578
- **Paths:** src/shared/python/workspace/global_utilities.py; src/shared/python/workspace/**init**.py; tests/launchers/test_global_workspace_utilities.py; docs/development/HANDOFF.md; docs/development/DEVELOPMENT_LOG.md; SPEC.md
- **Started:** 2026-09-20
- **Last verified:** 2026-10-11 (`e60d0927`; collated from verified GitHub merge metadata (PR #10578, 2026-09-20))
- **Shipped:** 2026-09-20
- **Summary:** Implemented `GlobalWorkspaceUtilitiesCoordinator` unifying Sidekick, Setup, Help, and Library utilities into global, workspace-agnostic overlays per ADR-0047 and issue #10528. Enforced canonical alias resolution mapping legacy utility IDs (`legacy_assistant`, `setup_wizard`, `library_browser`, `help_center`) to canonical names. Preserved conversation history across workspace transitions while synchronizing run and project context. Maintained sticky onboarding dismissal across session reload and migration. Enforced fail-closed native action behavior in browser environments.
- **Next step:** Shipped via PR #10578 on 2026-09-20 (merge SHA e60d09274f77e86c9bfb1beb3afe5e0e3a4324b5).
- **Evidence:** tests/launchers/test_global_workspace_utilities.py; src/shared/python/workspace/global_utilities.py.

### DL-#10530 · Reconcile and Document Intentionally Excluded, Research-Only, and Incomplete Workflows

- **State:** shipped
- **Owner:** local
- **Issue:** #10530 (ORG-22, epic #10508)
- **Branch:** feat/issue-10530-org22-research-lifecycle
- **PR:** #10579
- **Paths:** src/config/research_capability_lifecycle.py; src/config/**init**.py; src/config/capability_migration.py; src/tools/model_converter/**main**.py; tests/config/test_research_capability_lifecycle.py; docs/development/HANDOFF.md; docs/development/DEVELOPMENT_LOG.md; SPEC.md
- **Started:** 2026-09-20
- **Last verified:** 2026-10-11 (`f55cdd75`; collated from verified GitHub merge metadata (PR #10579, 2026-09-20))
- **Shipped:** 2026-09-20
- **Summary:** Implemented `ResearchCapabilityLifecycleManager` and `IncompleteCapabilityRecord` reconciling intentionally excluded, research-only, and incomplete workflows per ADR-0047 and issue #10530. Audited every excluded package under `src/tools/` against launcher tiles and `src/config/registry_exclusions.yaml`. Enforced fail-closed prevention of adapting CLI-only tools as GUI tiles via `CLINotInteractiveGUIError`. Verified and standardized CLI entry points for retained tools (`contraction`, `drift_control`, `model_converter`, `sg_optimizer`). Disclosed structured owner/issue/next-action metadata for incomplete capabilities, honestly documenting the SG optimizer Phase 3 PyQt6 UI follow-up tied to #6272 without fake GUIs or premature claims of completion.
- **Next step:** Shipped via PR #10579 on 2026-09-20 (merge SHA f55cdd752f72992d0714294580a743eda659bc64).
- **Evidence:** tests/config/test_research_capability_lifecycle.py; src/config/research_capability_lifecycle.py.

### DL-#10532 · Validate and Accept Every Enabled Recommended Task Journey Across Shipped Surfaces

- **State:** shipped
- **Owner:** local
- **Issue:** #10532 (ORG-23, epic #10508)
- **Branch:** feat/issue-10532-org23-installed-workspace-journeys
- **PR:** #10580
- **Paths:** src/shared/python/workspace/installed_journeys.py; src/shared/python/workspace/**init**.py; tests/integration/test_installed_workspace_journeys.py; src/tools/capture_rig/gui.py
- **Started:** 2026-09-20
- **Last verified:** 2026-10-11 (`70378c96`; collated from verified GitHub merge metadata (PR #10580, 2026-09-20))
- **Shipped:** 2026-09-20
- **Summary:** Implemented `InstalledWorkspaceJourneysCoordinator` validating and accepting all enabled recommended task journeys across shipped surfaces per ADR-0047 and issue #10532. Covered optical/video import to inspection, model/pose to supported fit, shot to named flight comparison, optimization/training to result, bounded estimation to cross-engine comparison, and global utilities navigation. Enforced failure recovery preserving source media, actionable dependency diagnostics, and rejection of corrupted artifact schemas. Decoupled `CaptureRigWidget.open_in_inspect_targets()` from StepRail action set to preserve contract and fix test suites.
- **Next step:** Shipped via PR #10580 on 2026-09-20 (merge SHA 70378c9602c68080efe942ad3f2affaae7d770e8).
- **Evidence:** tests/integration/test_installed_workspace_journeys.py; src/shared/python/workspace/installed_journeys.py.

### DL-#10533 · Reconcile, Audit, and Freeze Feature Preservation Across All Historical Boundaries

- **State:** shipped
- **Owner:** local
- **Issue:** #10533 (ORG-24, epic #10508)
- **Branch:** feat/issue-10533-org24-feature-preservation-audit
- **PR:** #10581
- **Paths:** src/shared/python/workspace/feature_preservation_audit.py; src/shared/python/workspace/**init**.py; tests/integration/test_feature_preservation_audit.py; docs/development/ORG24_FEATURE_PRESERVATION_AUDIT.md; docs/development/HANDOFF.md; docs/development/DEVELOPMENT_LOG.md
- **Started:** 2026-09-20
- **Last verified:** 2026-10-11 (`5d73e455`; collated from verified GitHub merge metadata (PR #10581, 2026-09-20))
- **Shipped:** 2026-09-20
- **Summary:** Implemented `FeaturePreservationAuditor` providing comprehensive reconciliation, integrity auditing, and disposition freezing for Epic #10508 closeout per issue #10533. Validated that all 159 capabilities from the ORG-01 baseline are preserved without silent removals. Verified byte-exact SHA-256 and file size integrity across all supported preservation fixtures. Confirmed acyclic transitive resolution of legacy aliases (`starting_pose_matcher` -> `motion_target_preview`, `putting_green_gui` -> `putting_green`). Audited 5 core workspaces and confirmed honest physics engine qualifications (#10351, #10353). Generated and published frozen audit disposition report at `docs/development/ORG24_FEATURE_PRESERVATION_AUDIT.md`.
- **Next step:** Shipped via PR #10581 on 2026-09-20 (merge SHA 5d73e45545c9498e1904eab8d7355cce7ef353c3).
- **Evidence:** tests/integration/test_feature_preservation_audit.py; src/shared/python/workspace/feature_preservation_audit.py; docs/development/ORG24_FEATURE_PRESERVATION_AUDIT.md.

### DL-#10432 · Calibrate and Smooth Full-Swing Pinocchio Kinematics With Exact Grip Compatibility

- **State:** shipped
- **Owner:** local
- **Issue:** #10432 (PF-02, epic #10427)
- **Branch:** feat/issue-10432-pf02-pinocchio-kinematics-grip-calibration
- **PR:** #10497 (auto-merge enabled)
- **Paths:** src/engines/physics_engines/pinocchio/python/marker_kinematics.py; src/shared/python/motion_matching/kinematic_smoothing.py; tests/unit/motion_matching/test_pinocchio_kinematics_calibration.py
- **Started:** 2026-09-19
- **Last verified:** 2026-10-11 (`a80290b4`; collated from verified GitHub merge metadata (PR #10497, 2026-09-20))
- **Shipped:** 2026-09-20
- **Summary:** Implemented SolveDiagnostics for Pinocchio MarkerIkSolver capturing convergence, projected gradient norm, cost decrease, and active bound counts. Added multi-start resolution solve_frame_multi_start evaluating geometric tracking floors with and without weld closure. Added refine_overlapping_window with bounded temporal regularization. Implemented kinematic_smoothing module providing joint trajectory smoothing with analytical/numerical derivative compatibility (q_dot ≈ v, v_dot ≈ a), boundary spike auditing, and cutoff frequency sensitivity analysis. Verified human range of motion wrist compliance (MM-2, #10104) and address left elbow pit up-and-inward alignment (MM-5, #10107).
- **Next step:** Shipped via PR #10497 on 2026-09-20 (merge SHA a80290b4d458159d1587eb354aa12de0b9adf652).
- **Evidence:** tests/unit/motion_matching/test_pinocchio_kinematics_calibration.py; src/shared/python/motion_matching/kinematic_smoothing.py; src/engines/physics_engines/pinocchio/python/marker_kinematics.py.

### DL-#10622 · NM-07 Compare Forward Surrogates and Physics-Structured Alternatives

- **State:** shipped
- **Owner:** local
- **Issue:** #10622 (epic #10603)
- **Branch:** feat/nm07-forward-surrogates-10622
- **PR:** #10768
- **Paths:** src/shared/python/motion_matching/surrogate/validate.py; src/shared/python/motion_matching/surrogate/nm07_comparison.py; src/shared/python/neural_motion/surrogates/; tests/unit/motion_matching/test_forward_surrogates_nm07.py; tests/unit/neural_motion/test_forward_surrogates_nm07.py; tests/unit/neural_motion/test_surrogate_nm07_discovery.py; docs/plans/neural_motion_matching/forward_surrogates.md; docs/plans/neural_motion_matching/evidence/nm07_forward_surrogates_receipt.json
- **Started:** 2026-09-23
- **Last verified:** 2026-10-11 (`c79f9772`; collated from verified GitHub merge metadata (PR #10768, 2026-09-23))
- **Shipped:** 2026-09-23
- **Summary:** Compare forward surrogate inversion, hybrid polish, physics-structured residual dynamics, and diffusion fallback under schema neural-surrogate-comparison/1.0.0. Added real-clock timegrid resampling, antipodal quaternion geodesic distance, trust-region validation, directional derivative gradient fidelity check, and contact-boundary failure rejection to validate.py. Documented adversarial exploitation risk of unconstrained forward surrogate inversion and high latency/sample inefficiency of diffusion fallback.
- **Next step:** Shipped via PR #10768 on 2026-09-23 (merge SHA c79f97724445d24a403d932bbd8bd8919cf81bac).
- **Evidence:** docs/plans/neural_motion_matching/forward_surrogates.md; docs/plans/neural_motion_matching/evidence/nm07_forward_surrogates_receipt.json

### DL-#10623 · NM-08 Add Native Verification, Distribution Checks and Safe Fallback

- **State:** shipped
- **Owner:** local
- **Issue:** #10623 (epic #10603)
- **Branch:** feat/nm08-native-verification-10623
- **PR:** #10770
- **Paths:** src/shared/python/neural_motion/inference/; src/shared/python/motion_matching/hybrid.py; tests/unit/neural_motion/test_verified_inference_nm08.py; tests/unit/motion_matching/test_verified_inference_nm08.py; docs/plans/neural_motion_matching/native_verification.md; docs/plans/neural_motion_matching/evidence/nm08_native_verification_receipt.json
- **Started:** 2026-09-23
- **Last verified:** 2026-10-11 (`a0beddd8`; collated from verified GitHub merge metadata (PR #10770, 2026-09-23))
- **Shipped:** 2026-09-23
- **Summary:** Added VerifiedInferenceOrchestrator, DistributionBounds, and check_target_distribution under neural_motion/inference/ (schema neural-verified-inference/1.0.0). Enforces fail-closed validation of non-finite inputs, geometry/engine/control dimension mismatch, and incompatible checkpoint contracts. Validates empirical coverage (durations, peak velocities, contact regimes) with confidence scoring recorded as domain metrics, not golfer truth probability. Enforces mandatory independent replay before dynamic acceptance; on failed proposal or missing checkpoint, falls back to classical/retrieval solver with shared remaining wall-clock budget and retains all attempts with auditable statuses (NEURAL_ACCEPTED, CLASSICAL_FALLBACK, REJECTED). Integrated fit_swing_verified_inference into hybrid.py facade.
- **Next step:** Shipped via PR #10770 on 2026-09-23 (merge SHA a0beddd8ecc8d53976a49179785712b3636a7ec7).
- **Evidence:** docs/plans/neural_motion_matching/native_verification.md; docs/plans/neural_motion_matching/evidence/nm08_native_verification_receipt.json

### DL-#10439 · Replace Synthetic Force Adapters With Native Model-Conformant Bridges

- **State:** shipped
- **Owner:** local
- **Issue:** #10439 (PF-09, epic #10430, program #10363)
- **Branch:** feat/issue-10439-pf09-native-force-bridges
- **PR:** #10507
- **Paths:** scripts/allocate_swing_torques.py; src/engines/physics_engines/pinocchio/python/force_adapter.py; src/engines/physics_engines/pinocchio/python/native_model.py; src/shared/python/motion_matching/multi_engine_torque_allocator.py; tests/integration/engines/pinocchio/test_force_adapter.py; tests/integration/engines/pinocchio/test_force_mapping.py; tests/unit/motion_matching/test_force_bridges_pf09.py; tests/unit/motion_matching/test_multi_engine_torque_allocator.py; tests/unit/motion_matching/test_native_force_equations.py
- **Started:** 2026-09-19
- **Last verified:** 2026-10-11 (`20036a7c`; collated from verified GitHub merge metadata (PR #10507, 2026-09-21))
- **Shipped:** 2026-09-21
- **Summary:** Quarantined `_AnalyticalMultibodyBase` production routes as `SyntheticMultibodyFixture` requiring explicit `allow_synthetic=True`, failing closed with `RuntimeError` on unbridged engines (Drake, OpenSim, Simscape). Extended `BaseEngineForceAdapter` protocol with `model_hash`, `coordinate_order`, `contact_names`, and `compute_mass_and_bias`. Corrected `MujocoForceAdapter` to compute raw unconstrained generalized dynamic forces M a + bias eliminating `qfrc_inverse` passive/constraint force double counting, added input validation and state refresh before mutation, and verified exact acceleration parity. Implemented `PinocchioForceAdapter` with fresh constraint kinematics refresh (`_refresh_constraint_data` and `closure_force_jacobian`). Updated `allocate_swing_torques.py` CLI to support Pinocchio and enforce `--allow-synthetic` gate.
- **Next step:** Shipped via PR #10507 on 2026-09-21 (merge SHA 20036a7c5fd7e9a38a342bb1c1a0c5b968cc8223).
- **Evidence:** tests/unit/motion_matching/test_native_force_equations.py; tests/unit/motion_matching/test_multi_engine_torque_allocator.py; tests/unit/motion_matching/test_force_bridges_pf09.py.

### DL-#8684 · Coupled Grip, Shaft, Ground Rollup

- **State:** shipped
- **Owner:** claude
- **Issue:** #8684
- **Branch:** conductor/issue-8684
- **PR:** #10668, #9998
- **Paths:** docs/research/proximal_distal_energy_transfer/COMPREHENSIVE_RESEARCH_PROGRAM.md, MODEL_COMPLETION_FALSIFICATION_MATRIX.md
- **Started:** 2026-09-11
- **Last verified:** 2026-10-11 (`70560607`; collated from verified GitHub merge metadata (PR #9998, 2026-09-21))
- **Shipped:** 2026-09-21
- **Summary:** Parent rollup of child tiers #8685/#8797/#8715/#8723 answering the four #8684 questions; manifests re-pinned.
- **Next step:** Shipped via PR #9998 on 2026-09-21 (merge SHA 705606076622b0f886952401e15de6c8692999af).

### DL-#10440 · Connect Qualified Matching Strategies to Existing Results and Engine Feature Contracts

- **State:** shipped
- **Owner:** local
- **Issue:** #10440 (PF-10, epic #10430, program #10363)
- **Branch:** feat/issue-10440-pf10-matching-strategies-contracts
- **PR:** #10509
- **Paths:** src/shared/python/motion_matching/matching_strategy.py; src/shared/python/motion_matching/**init**.py; tests/unit/motion_matching/test_matching_strategy.py
- **Started:** 2026-09-19
- **Last verified:** 2026-10-11 (`b633007c`; collated from verified GitHub merge metadata (PR #10509, 2026-09-21))
- **Shipped:** 2026-09-21
- **Summary:** Implemented versioned matching strategy schema (`STRATEGY_SCHEMA_VERSION = "matched-strategy-v1"`), stage-separated qualification matrix tracking all 6 stages (`model_available`, `kinematic_fit`, `force_feasible`, `replay_accepted`, `runtime_budget_met`, `muscle_qualified`), strategy presets, controller specifications, and contact reaction history containers. Created `CandidateStrategyPackage` supporting lossless .npz serialization without pickle, and fail-closed name-permuted coordinate remapping. Implemented `StrategyComparisonService` exposing cross-strategy torque profiles, kinematics/closure errors, and capability auditing invalidating supported status on missing SDKs. Provided sample package generator for concrete handoff to MV-03 #10479 without modifying viewer files.
- **Next step:** Shipped via PR #10509 on 2026-09-21 (merge SHA b633007c6ab323d71529ffc1109fca1c02218bc1).
- **Evidence:** tests/unit/motion_matching/test_matching_strategy.py.

### DL-#10433 · Enforce Contact, Actuator and Root Constraints in Force Allocation

- **State:** shipped
- **Owner:** local
- **Issue:** #10433 (PF-03, epic #10430)
- **Branch:** feat/issue-10433-pf03-contact-actuator-root-constraints
- **PR:** #10498 (auto-merge enabled)
- **Paths:** src/shared/python/motion_matching/contact_force_allocator.py; tests/unit/motion_matching/test_contact_force_allocator.py; tests/unit/motion_matching/test_contact_force_allocator_pf03.py
- **Started:** 2026-09-19
- **Last verified:** 2026-10-11 (`7e6b3989`; collated from verified GitHub merge metadata (PR #10498, 2026-09-21))
- **Shipped:** 2026-09-21
- **Summary:** Upgrades ContactForceAllocator with constrained QP inverse dynamics. Enforces 8-faceted polyhedral friction pyramid, non-negative normal ground forces along arbitrary terrain normals, exact contact separation masks, and strict actuator bounds without post-projection. Separates diagnostic root slack so ungrounded reactions never create false physical success. Introduces FeasibilityStatus, HARD_ZERO_TRAIL mode, and verify_torque_and_rate_bounds.
- **Next step:** Shipped via PR #10498 on 2026-09-21 (merge SHA 7e6b39893674af3de31fb79efc6a2710f30c36de).
- **Evidence:** tests/unit/motion_matching/test_contact_force_allocator_pf03.py; tests/unit/motion_matching/test_contact_force_allocator.py.

### DL-#10589 · TB-04: Fit and Independently Replay the Actual Driven Double Pendulum

- **State:** shipped
- **Owner:** local
- **Issue:** #10589 (TB-04, parent #10584, program #10363)
- **Branch:** feat/tb04-double-pendulum-fit-10589
- **PR:** #10638
- **Paths:** src/engines/physics_engines/pendulum/python/motion_matching/adapters.py; src/engines/physics_engines/pendulum/python/motion_matching/torque_optimization.py; src/engines/physics_engines/pendulum/python/motion_matching/provider.py; src/engines/physics_engines/pendulum/python/motion_matching/qualification.py; src/engines/physics_engines/pendulum/python/motion_matching/**init**.py; tests/unit/engines/physics_engines/pendulum/test_double_pendulum_fit.py; tests/unit/engines/physics_engines/pendulum/test_motion_matching_provider.py; docs/plans/tour_baselines/evidence/tb04_driver_qualification_receipt.json; docs/plans/tour_baselines/evidence/tb04_iron_qualification_receipt.json; docs/plans/tour_baselines/evidence/tb04_driver_baseline_package.npz; docs/plans/tour_baselines/evidence/tb04_iron_baseline_package.npz; docs/plans/tour_baselines/coverage_matrix.md
- **Started:** 2026-09-20
- **Last verified:** 2026-10-11 (`c04a22e0`; collated from verified GitHub merge metadata (PR #10638, 2026-09-21))
- **Shipped:** 2026-09-21
- **Summary:** Built bidirectional mapping and verified mathematical & numerical acceleration parity (< 1e-14) between DoublePendulumDynamics and Tools physics.py. Formulated continuous smooth bounded joint torques via degree-6 Bernstein polynomials strictly bounded in [tau_min, tau_max] with curvature and effort regularization. Fixed frame-0 off-by-one initial state evaluation bug, implemented non-uniform timestep integration, and added independent 4x tighter substep replay verification. Produced authoritative qualification receipts and baseline packages for Driver and 7-Iron.
- **Next step:** Shipped via PR #10638 on 2026-09-21 (merge SHA c04a22e0e0435583d31ea56f4c3167dc79b1f5fa). Preserved limitation: Only the small baseline is evidenced; avoid a broader production claim.
- **Evidence:** docs/plans/tour_baselines/evidence/tb04_driver_qualification_receipt.json; docs/plans/tour_baselines/evidence/tb04_iron_qualification_receipt.json; tests/unit/engines/physics_engines/pendulum/test_double_pendulum_fit.py; tests/unit/engines/physics_engines/pendulum/test_motion_matching_provider.py.

### DL-#10434 · Qualify Contact Modes and Native Pinocchio Force Feasibility

- **State:** shipped
- **Owner:** local
- **Issue:** #10434 (epic #10363, PF-04)
- **Branch:** feat/issue-10434-pf04-qualify-contact-modes-pinocchio-forces
- **PR:** #10499 (auto-merge enabled)
- **Paths:** src/shared/python/motion_matching/contact_mode_qualifier.py; tests/unit/motion_matching/test_contact_mode_qualifier_pf04.py
- **Started:** 2026-09-19
- **Last verified:** 2026-10-11 (`9cb9b951`; collated from verified GitHub merge metadata (PR #10499, 2026-09-21))
- **Shipped:** 2026-09-21
- **Summary:** Implemented ContactModeQualifier evaluating multi-sphere heel/toe support modes with clearance and velocity hysteresis, computing center of pressure (COP) and convex hull containment under arbitrary normal directions, evaluating slip velocity and friction saturation, and cross-checking allocated contact forces against constitutive Hunt-Crossley compliant models. Enforces separate linear force and moment residual budgets, rejects unphysical loads (> 5000 N or > 300 Nm), and generates parameter sensitivity reports.
- **Next step:** Shipped via PR #10499 on 2026-09-21 (merge SHA 9cb9b951622f0bd8ddcb1c12eb224ad6822e1579).
- **Evidence:** tests/unit/motion_matching/test_contact_mode_qualifier_pf04.py; src/shared/python/motion_matching/contact_mode_qualifier.py.

### DL-#10366 · MS-16: MuJoCo Native IK and MJ_Inverse Tracking

- **State:** shipped
- **Owner:** local
- **Issue:** #10366 (MS-16, epic #10363)
- **Branch:** fix/issue-10366-ms-16-mujoco-native-tools-marker-ik-on-m-cursor-composer-local
- **PR:** #10660
- **Paths:** src/engines/physics_engines/mujoco/python/ik_minimize.py; src/engines/physics_engines/mujoco/python/inverse_dynamics.py; src/shared/python/motion_matching/pipeline/reference.py; src/shared/python/motion_matching/pipeline/dynamics.py; src/shared/python/motion_matching/pipeline/cli.py; src/tools/motion_matching/pipeline.py; tests/unit/motion_matching/test_mujoco_ik_minimize.py; tests/unit/motion_matching/test_mujoco_mj_inverse.py; docs/development/matched_swing_program/README.md; tests/tools/matched_swing_browser/test_model.py
- **Started:** 2026-09-21
- **Last verified:** 2026-10-11 (`8e4ecee2`; collated from verified GitHub merge metadata (PR #10660, 2026-09-21))
- **Shipped:** 2026-09-21
- **Summary:** Added selectable `--ik-backend mujoco-minimize` (MuJoCo `minimize.least_squares` with LM warm start) and `--tracking mj-inverse` (plant KKT torques with native `mj_inverse` audit). Wired backends through pipeline plant, dynamics replay, CLI, Motion Matching tile, and receipt fields with DbC validation at API boundaries.
- **Next step:** Shipped via PR #10660 on 2026-09-21 (merge SHA 8e4ecee277d25cc305fcf8f4e37d333260e83a15).
- **Evidence:** docs/development/full_body_models/evidence/ground_support/anthro_driver_native_tools/receipt.json; tests/unit/motion_matching/test_mujoco_ik_minimize.py; tests/unit/motion_matching/test_mujoco_mj_inverse.py; tests/docs/test_matched_swing_status_freshness.py

### DL-#10361 · MS-90: Generic Capture Contract & 44-DOF Identifiability

- **State:** shipped
- **Owner:** local
- **Issue:** #10361 (MS-90, epic #10363)
- **Branch:** feat/10361-generic-capture-contract
- **PR:** #10633
- **Paths:** src/shared/python/motion_matching/tour_capture_contract.py; src/shared/python/motion_matching/identifiability.py; tests/unit/motion_matching/test_capture_contract_generic.py; tests/unit/motion_matching/test_identifiability.py; evidence/anthropometry/identifiability_driver.json; docs/user_guide/motion_matching/loading_targets.md
- **Started:** 2026-09-20
- **Last verified:** 2026-10-11 (`b68ed246`; collated from verified GitHub merge metadata (PR #10633, 2026-09-21))
- **Shipped:** 2026-09-21
- **Summary:** Implemented `CaptureContract` and `CaptureValidationReport` enabling validation and loading of arbitrary C3D captures without editing codebase source. Retained frozen tour captures as named instances of `CaptureContract`. Implemented `probe_spec_identifiability` and `probe_synthetic_chain_identifiability` performing linearised SVD identifiability analysis, detecting planted null directions and unobservable lower-body DOFs on the 44-DOF model, and demonstrating resolution to full rank via anthropometric prior regularization. Generated and committed `evidence/anthropometry/identifiability_driver.json`.
- **Next step:** Shipped via PR #10633 on 2026-09-21 (merge SHA b68ed246cc78e63c118c2d8aebebd3ec9d226b2a).
- **Evidence:** evidence/anthropometry/identifiability_driver.json; tests/unit/motion_matching/test_capture_contract_generic.py; tests/unit/motion_matching/test_identifiability.py.

### DL-#10348 · Simscape Topology + Full-Marker Terminal (MS-61)

- **State:** shipped
- **Owner:** local
- **Issue:** #10348 (MS-61, epic #10363)
- **Branch:** feat/issue-10348-ms61-simscape-topology
- **PR:** #10676
- **Paths:** src/shared/python/motion_matching/{simscape_topology.py,full_marker_terminal.py,tour_metrics.py,acceptance.py}; scripts/matlab/{materialize_ms61_topology_receipts.py,run_simscape_candidate.ps1}; docs/development/simscape_tour_matching/native_evidence/two_window_fit_9967_103/; docs/development/matched_swing_program/{GATES.md,README.md,WAVES.md}; docs/development/simscape_tour_matching/CHECKPOINTS.md
- **Started:** 2026-09-21
- **Last verified:** 2026-10-11 (`bc5a7ff9`; collated from verified GitHub merge metadata (PR #10676, 2026-09-22))
- **Shipped:** 2026-09-22
- **Summary:** Fail-closed 27-DOF topology classification (no independent neck), dual terminal disclosure (full + head cluster; body-excluding-head diagnostic only), acceptance/tour_metrics dual-terminal contracts, and run-103 scaffolding derived from run-102 without inventing native G1 success; repair linked to MS-104 (#10378).
- **Next step:** Shipped via PR #10676 on 2026-09-22 (merge SHA bc5a7ff90cfb0da3ed49a3ac3d20d1967ba267df). Preserved limitation: Scaffold only; do not represent scaffold as completed production behavior.
- **Evidence:** docs/development/simscape_tour_matching/native_evidence/two_window_fit_9967_103/{topology_report.json,terminal_breakdown.json,native_gate.json,runtime_license_receipt.json,parity_receipt.json,HANDOFF.md}; tests/unit/motion_matching/test_simscape_topology_ms61.py.

### DL-#10359 · Wire Video and Fit-Quality Report Export

- **State:** shipped
- **Owner:** claude
- **Issue:** #10359 (MS-86, epic #10363)
- **Branch:** feat/10359-export-video-report
- **PR:** #10632
- **Paths:** src/shared/python/motion_matching/export.py; src/shared/python/motion_matching/**main**.py; src/tools/matched_swing_browser/gui.py; src/config/launcher_manifest.json; src/config/models.yaml; tests/unit/motion_matching/test_export.py; tests/tools/matched_swing_browser/test_matched_swing_browser_gui.py; docs/development/matched_swing_program/evidence/reports/sample_fit_report.md
- **Started:** 2026-09-20
- **Last verified:** 2026-10-11 (`79ab1757`; collated from verified GitHub merge metadata (PR #10632, 2026-09-21))
- **Shipped:** 2026-09-21
- **Summary:** Implemented `export_video` and `export_report` with fail-closed DbC contracts, standardized metrics, acceptance gate verdicts, physical constraints, and full cryptographic provenance (#8820 / U3). Added CLI subcommands `export-video` and `export-report` to motion_matching module. Added "Export Video..." and "Export Report..." buttons to Results Browser GUI Actions card with file dialogs. Registered `video_export` and `report_export` capabilities in launcher manifest and models config. Emitted sample Markdown fit report in matched swing program evidence directory.
- **Next step:** Shipped via PR #10632 on 2026-09-21 (merge SHA 79ab1757c867d64419044468a607502c5bce6a56).
- **Evidence:** docs/development/matched_swing_program/evidence/reports/sample_fit_report.md; tests/unit/motion_matching/test_export.py; tests/tools/matched_swing_browser/test_matched_swing_browser_gui.py.

### DL-#8930 · Vectorize Rust Trajectory Post-Processing

- **State:** shipped
- **Owner:** claude
- **Issue:** #8930
- **Branch:** fix/8930-ball-flight-vectorized-rk4
- **PR:** #10648 (open)
- **Paths:** src/shared/python/physics/ball_simulator.py; tests/unit/physics/test_ball_simulator_post_process_vectorized_8930.py
- **Started:** 2026-09-21
- **Last verified:** 2026-10-11 (`f80c6193`; collated from verified GitHub merge metadata (PR #10648, 2026-09-21))
- **Shipped:** 2026-09-21
- **Summary:** `BallFlightSimulator._post_process_rust` called the scalar `_calculate_forces_single` path once per trajectory point instead of the existing vectorized `_calculate_forces_batch` path; now builds the `(3, N)` batch once and calls force calculation a single time per trajectory.
- **Next step:** Shipped via PR #10648 on 2026-09-21 (merge SHA f80c6193316231adefa5bdc7a560ba10abb3635c).

### DL-#10436 · Explore Feasible Force Null Spaces and Publish Torque-Distribution Tradeoffs

- **State:** shipped
- **Owner:** local
- **Issue:** #10436 (PF-06, epic #10430)
- **Branch:** feat/issue-10436-pf06-feasible-force-nullspace
- **PR:** #10504
- **Paths:** src/shared/python/motion_matching/force_nullspace.py; tests/unit/motion_matching/test_force_nullspace.py; tests/unit/motion_matching/test_force_nullspace_pf06.py
- **Started:** 2026-09-19
- **Last verified:** 2026-10-11 (`a5cec306`; collated from verified GitHub merge metadata (PR #10504, 2026-09-21))
- **Shipped:** 2026-09-21
- **Summary:** Extended force null space representation to support scaled SVD and column-pivoted QR decomposition with dynamic rank and contact mode reporting. Added `NullSpaceAnalysis` reporting condition number and machine-precision residuals (A N = 0, A x_p = b). Implemented `redistribute_trajectory` with physical rate penalties strictly invariant to basis sign changes. Implemented `explore_torque_tradeoffs` generating Pareto tradeoff alternatives across baseline minimum effort, conservative default, trail arm reduction sweeps (50%, 80%), hard-zero trail feasibility checks, relaxed minimum trail alternatives, grip squeeze minimization, and ground reaction regularization. Detailed diagnostics report per-joint torque, power, lead/trail effort, ground COP, grip wrench, and explicit SI units. Exported reproducible Pareto tradeoff tables to JSON and CSV. Selected conservative default with mechanical rationale without unfounded metabolic or injury claims.
- **Next step:** Shipped via PR #10504 on 2026-09-21 (merge SHA a5cec3063e050da0f6b153381ac04d453ffcdad0).
- **Evidence:** tests/unit/motion_matching/test_force_nullspace.py; tests/unit/motion_matching/test_force_nullspace_pf06.py.

### DL-#10376 · Complete Engine and Model Inventory With Runnable Model Packages (MS-102)

- **State:** shipped
- **Owner:** local
- **Issue:** #10376 (MS-102, epic #10363)
- **Branch:** feat/ms102-engine-model-inventory
- **PR:** #10677
- **Paths:** src/engines/model_inventory.py; src/config/engine_model_inventory.json; tests/unit/engines/test_model_inventory.py; docs/development/matched_swing_program/evidence/ms102/
- **Started:** 2026-09-21
- **Last verified:** 2026-10-11 (`00e32b56`; collated from verified GitHub merge metadata (PR #10677, 2026-09-22))
- **Shipped:** 2026-09-22
- **Summary:** Authority-derived engine/model inventory (models.yaml + capability matrix + ENGINE_TIERS) with dual-club flagship packages, immutable hashes, qualification harness (resolve/hash/load/FK/dynamics/viewer/save), and named repair blockers. Not a competing catalog. Simscape entries require MATLAB R2025b.
- **Next step:** Shipped via PR #10677 on 2026-09-22 (merge SHA 00e32b5646df5bd6506606886d9c0aa045857681).
- **Evidence:** docs/development/matched_swing_program/evidence/ms102/

### DL-#10344 · MyoSuite Golfer Scene: Pinned MyoSim Submodule + Dual-Grip Club Contacts (MS-51)

- **State:** shipped
- **Owner:** local
- **Issue:** #10344 (MS-51, epic #10363)
- **Branch:** fix/10344-ms51-myosuite-repair
- **PR:** #10685
- **Paths:** src/engines/physics_engines/myosuite/python/golfer_scene.py; coordinate_map_anthro.json; shared/models/myosuite/golf/body/; scripts/setup_myosuite_models.{ps1,sh}; src/engines/model_inventory.py; src/config/engine_model_inventory.json; docs/engines/myosuite.md; tests/unit/engines/myosuite/test_golfer_scene.py; tests/myosuite/test_golfer_scene_native.py
- **Started:** 2026-09-22
- **Last verified:** 2026-10-11 (`cfe7217b`; collated from verified GitHub merge metadata (PR #10685, 2026-09-22))
- **Shipped:** 2026-09-22
- **Summary:** Pinned myo_sim gitlink documented; bootstrap scripts; generated driver/iron golfer MJCF on myobody_simpleupper with dual-grip site welds and four foot contact markers; diagnostic 40-of-44 coordinate map; MS-102 inventory left repair after real probes.
- **Next step:** Shipped via PR #10685 on 2026-09-22 (merge SHA cfe7217bf41ab59f546624e22df4712f1e75bb87). Preserved limitation: Map coverage is 40/44; preserve the four missing mappings as follow-up.
- **Evidence:** shared/models/myosuite/golf/body/golfer*myobody_receipt.json; docs/development/matched_swing_program/evidence/ms102/myosuite*\*\_structural_receipt.json

### DL-#9479 · Consolidate Engine Meta-Tiles and Clarify Confusable Launcher Tile Names

- **State:** shipped
- **Owner:** claude
- **Issue:** #9479, #9480 (parent #9412; cluster #9410 Cluster B)
- **Branch:** fix/9479-9480-launcher-tiles
- **PR:** #10653
- **Paths:** src/config/launcher_manifest.json; src/config/models.yaml; src/launchers/workspace_navigation.py; scripts/check_launcher_logo_families.py; tests/config/launcher_manifest/test_engine_hub_consolidation_9479.py; tests/config/launcher_manifest/test_tile_name_clarity_9480.py; tests/launchers/test_workspace_navigation.py
- **Started:** 2026-09-21
- **Last verified:** 2026-10-11 (`c70555a5`; collated from verified GitHub merge metadata (PR #10653, 2026-09-22))
- **Shipped:** 2026-09-22
- **Summary:** Hide duplicate per-engine dashboards with documented reasons; reclassify three specialized tools from `physics_engine` to `simulation`; clarify confusable data/video tile pairs; move non-golf utilities to `dev_research`.
- **Next step:** Shipped via PR #10653 on 2026-09-22 (merge SHA c70555a5e1aed74c01ce1001dd2e28265fdcc0b7).
- **Evidence:** tests/config/launcher_manifest/test_engine_hub_consolidation_9479.py; tests/config/launcher_manifest/test_tile_name_clarity_9480.py; tests/launchers/test_workspace_navigation.py::TestWorkspaceMembership::test_non_golf_utilities_moved_out_of_primary_workflow.

### DL-#8883 · Video Analyzer Real GUI Replacing the Placeholder Label

- **State:** shipped
- **Owner:** local
- **Issue:** #8883 (related #8854, #10512)
- **Branch:** fix/8883-video-analyzer-gui
- **PR:** #10651, #10739
- **Paths:** src/tools/video_analyzer/analyzer.py; src/tools/video_analyzer/gui.py; src/launchers/external_tools_adapter.py; src/launchers/task_launch_truthfulness.py; tests/unit/test_video_analyzer_pipeline.py; tests/ui/tools/video_analyzer/; tests/launchers/test_simulation_guis.py; tests/launchers/test_task_launch_truthfulness.py
- **Started:** 2026-09-21
- **Last verified:** 2026-10-11 (`5dd695f8`; collated from verified GitHub merge metadata (PR #10739, 2026-09-23))
- **Shipped:** 2026-09-23
- **Summary:** Replaced the static `QLabel("Video Analyzer (GUI placeholder)")` with a real `MainWidget` (choose video, Analyze, report pane) wired to a new `SwingAnalyzer.analyze_video()` that runs MediaPipe pose estimation (via the existing `pose_estimation` registry) and feeds the already-tested head-stability math. Removed the launcher's dead sibling-repo import fallback (`video_analyzer.launch_pyqt6`, confirmed nonexistent by #8854) so the tile no longer depends on an external Tools provider; updated `task_launch_truthfulness`'s audit entry from `PROVIDER_REQUIRED` to `PRODUCTION_SOLVER` accordingly.
- **Next step:** Shipped via PR #10739 on 2026-09-23 (merge SHA 5dd695f89774a615af59ebc0e89b4f0a631049bf).
- **Evidence:** tests/unit/test_video_analyzer_pipeline.py; tests/ui/tools/video_analyzer/test_gui.py.

### DL-#10743 · Docs-Consistency Cross-Repo Path Exemption

- **State:** shipped
- **Owner:** claude (session `fleet-remediation-k`)
- **Issue:** #10743
- **Branch:** `fix/10743-docs-consistency-cross-repo`
- **PR:** #10744, #10752
- **Paths:** `scripts/check_agent_docs_consistency.py`, `tests/architecture/test_check_agent_docs_consistency.py`
- **Started:** 2026-09-22
- **Last verified:** 2026-10-11 (`2a9e7a4d`; collated from verified GitHub merge metadata (PR #10752, 2026-09-23))
- **Shipped:** 2026-09-23
- **Summary:** Exempt sibling-repo-qualified backticked paths (possessive or not) from the local-existence check so the fleet-managed deferred-validation block no longer fails `repo-structure-gates`; bare local paths stay strict.
- **Next step:** Shipped via PR #10752 on 2026-09-23 (merge SHA 2a9e7a4d30bf169cc666eb81093f2ebe770f8678).

### DL-#10751 · Fix CI Standard 'Deleted Python Test Files' Check False-Positives in Shallow Checkouts

- **State:** shipped
- **Owner:** antigravity
- **Issue:** #10751
- **Branch:** fix/10751-deleted-tests-shallow-checkout
- **PR:** #10762
- **Paths:** scripts/ci/check_deleted_test_files.py; tests/scripts/test_check_deleted_test_files.py; .github/workflows/ci-standard.yml; tests/ci/test_ci_infrastructure.py
- **Started:** 2026-09-23
- **Last verified:** 2026-10-11 (`e1192b7c`; collated from verified GitHub merge metadata (PR #10762, 2026-09-23))
- **Shipped:** 2026-09-23
- **Summary:** Compute deleted tests diff against merge-base instead of base ref tip to prevent tests added on main from being falsely reported as deleted in PRs; set fetch-depth: 0 on checkout in tests job; invoke standalone check_deleted_test_files.py with unit and regression test coverage and fallback-to-base support directly from the CI workflow.
- **Next step:** Shipped via PR #10762 on 2026-09-23 (merge SHA e1192b7c550cb4974570ba3a2b66d4c520b5a032).

### DL-#10597 · Tour Baselines Baseline Guide, Agent Runbook, and End-to-End Acceptance (TB-12)

- **State:** shipped
- **Owner:** antigravity
- **Issue:** #10597 (TB-12, parent #10584, program #10363)
- **Branch:** `feat/tb12-baseline-guide-acceptance-10597`
- **Paths:** `docs/plans/tour_baselines/baseline_guide.md`; `docs/plans/tour_baselines/agent_runbook.md`; `docs/plans/tour_baselines/final_acceptance_report.md`; `tests/acceptance/test_tour_baselines_journey.py`; `docs/plans/tour_baselines/README.md`; `docs/development/matched_swing_program/README.md`; `docs/development/DEVELOPMENT_LOG.md`; `SPEC.md`
- **Started:** 2026-09-24
- **Last verified:** 2026-10-11 (`fbebf5c4`; collated from verified GitHub merge metadata (PR #10814, 2026-09-24))
- **Shipped:** 2026-09-24
- **Summary:** Published concise Operator User Guide covering launcher navigation, driver vs. iron capture, complexity tradeoffs, loading/cloning, and physical error formulas; Agent Runbook with copyable clean-environment reproduction commands, hash verification, MATLAB R2025b requirements, and tamper detection; and Final Acceptance Report recording the completed two-capture coverage matrix, software integration sign-off, and governing issue links for outstanding full-body models. Authored comprehensive end-to-end acceptance test suite verifying the full user journey.
- **Next step:** Shipped via PR #10814 on 2026-09-24 (merge SHA fbebf5c4785bfffa83b1e3890c8ee07217e09698).
- **PR:** #10814

### DL-#10849 · Pendulum Inertia Hash Integration Parameter Digest & Dynamics Cache Refresh

- **State:** shipped
- **Owner:** antigravity
- **Issue:** #10849, #10850, #10851
- **PR:** #10860
- **Branch:** `fix/10849-10850-inertia-hash-and-spec-keys`
- **Paths:** `src/engines/pendulum_models/python/double_pendulum_model/physics/double_pendulum.py`, `src/engines/physics_engines/pendulum/python/motion_matching/adapters.py`, `src/engines/physics_engines/pendulum/python/motion_matching/qualification.py`, `tests/unit/tour_baselines/test_qualification.py`, `SPEC.md`, `AGENT_HANDOFF.md`, `docs/development/HANDOFF.md`, `docs/development/DEVELOPMENT_LOG.md`
- **Started:** 2026-09-24
- **Last verified:** 2026-10-11 (`15ca47c2`; collated from verified GitHub merge metadata (PR #10860, 2026-09-24))
- **Shipped:** 2026-09-24
- **Summary:** Updated DoublePendulumDynamics and qualification routines to derive the inertia digest directly from the cached properties consumed by simulation integration, initialize calibrated segment lengths before parameter caching, and key SPEC.md change log rows accurately (#10849, #10850, #10851).
- **Next step:** Shipped via PR #10860 on 2026-09-24 (merge SHA 15ca47c2ebfc7731972ff6a98f58ceb7074b9553).

### DL-#10883 · Restore Green Main: Jules Bolt Learning Title Case Compliance

- **State:** shipped
- **Owner:** local
- **Issue:** #10883
- **PR:** #10884
- **Branch:** `fix/10883-docs-governance-title-case`
- **Paths:** `.jules/bolt.md`, `tests/scripts/test_document_title_case.py`, `SPEC.md`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`
- **Started:** 2026-09-24
- **Last verified:** 2026-10-11 (`b15558ec`; collated from verified GitHub merge metadata (PR #10884, 2026-09-24))
- **Shipped:** 2026-09-24
- **Summary:** Backticked code tokens (`np.linalg.norm`) in .jules/bolt.md heading to restore green main against required docs-governance-gates and added unit regression test (#10883).
- **Next step:** Shipped via PR #10884 on 2026-09-24 (merge SHA b15558ec3cb230e0fa90efc3a702dbb7f15d4313).

### DL-#10921 · Require Both Desktop and Start Menu Destinations and Update Handoff Governance

- **State:** shipped
- **Owner:** local
- **Issue:** #10921 (companion #10918, #10919, #10920, #10924, #10925)
- **PR:** #10923
- **Branch:** `fix/10921-shortcuts-feedback`
- **Paths:** `src/launchers/desktop_shortcuts.py`, `tests/unit/launchers/test_desktop_shortcuts.py`, `SPEC.md`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`
- **Started:** 2026-09-24
- **Last verified:** 2026-10-11 (`d8f5b0a3`; collated from verified GitHub merge metadata (PR #10923, 2026-09-25))
- **Shipped:** 2026-09-25
- **Summary:** Enforce that shortcut installation requires both Desktop and Start Menu destinations to succeed, and synchronize canonical handoff documentation and SPEC change log (#10918, #10919, #10920, #10921, #10924, #10925).
- **Next step:** Shipped via PR #10923 on 2026-09-25 (merge SHA d8f5b0a39aef2823dac6520f1d11b8988f1b778c).

### DL-#10946 · Keep Multi-Muscle Contracts and Torque Identical on the Rust Backend

- **State:** shipped
- **Owner:** claude
- **Issue:** #10946
- **PR:** draft PR from `agy/ud-10946-rust-muscle-parity`, #10976
- **Branch:** `agy/ud-10946-rust-muscle-parity`
- **Paths:** `src/shared/python/biomechanics/multi_muscle.py`, `tests/unit/biomechanics/test_multi_muscle_rust_parity.py`
- **Started:** 2026-09-26
- **Last verified:** 2026-10-11 (`fee5b214`; collated from verified GitHub merge metadata (PR #10976, 2026-09-26))
- **Shipped:** 2026-09-26
- **Summary:** Python `HillMuscleModel`s were never forwarded to the Rust `MuscleGroup` (it summed zero muscles, so net torque was always 0.0) and the Rust path skipped the activation precondition. Muscles are now converted to their Rust equivalent (or the group drops to pure Python when a muscle has custom physics), activations are validated once before backend choice, the antagonist pair is rebuilt from the groups' current backends, and a parity test pins Rust == Python.
- **Next step:** Shipped via PR #10976 on 2026-09-26 (merge SHA fee5b214ec81674e8626049ac9ff47907f567f3a).

### DL-#10516 · Apply the Same Workspace Navigation to React and Tauri

- **State:** shipped
- **Shipped:** 2026-09-20
- **Owner:** local
- **Issue:** #10516 (ORG-06, epic #10508)
- **Branch:** feat/issue-10516-org06-react-workspace-navigation
- **PR:** #10540
- **Paths:** ui/src/types/workspaceNavigation.ts; ui/src/api/capabilityAdapter.ts; ui/src/components/layout/WorkspaceNavigation.tsx; ui/src/components/layout/WorkspaceNavigation.test.tsx; ui/src/pages/WorkspacePage.tsx; ui/src/components/simulation/LauncherDashboard.tsx; ui/src/App.tsx; ui/src/utils/routeTitles.ts; docs/development/DEVELOPMENT_LOG.md; docs/development/HANDOFF.md
- **Started:** 2026-09-19
- **Last verified:** 2026-10-11 (`944d798e`; collated from docs/development/archive/DEVELOPMENT_LOG_RECONCILIATION_AUDIT_20261011.json)
- **Summary:** Applied task-oriented workspace navigation across React and Tauri (ORG-06). Reused shared catalog metadata for the five primary workspaces (`Capture & Analyze`, `Model & Match`, `Shot & Course Lab`, `Optimize & Train`, `Results & Compare`) and secondary navigation (`Developer & Research`, `All Tools`, `Favorites`, `History`). Integrated `WorkspaceSidebar` and `WorkspaceBreadcrumb` within `WorkspaceShell` preserving browser history, bookmarkable task URLs (`/workspaces/:slug`), centralized route titles, and keyboard focus recovery. Created shared `resolveWorkspaceToolAction` capability adapter opening native tools under Tauri/desktop while providing actionable explanations and web alternatives for browser-only users.
- **Next step:** Merged in PR #10540 on 2026-09-20; see parent #10508 for remaining ORG work.
- **Evidence:** ui/src/components/layout/WorkspaceNavigation.test.tsx; ui/src/components/layout/WorkspaceNavigation.tsx.

### DL-#10515 · Build Task-Oriented Desktop Navigation Over Existing Embedded Tools

- **State:** shipped
- **Shipped:** 2026-09-20
- **Owner:** local
- **Issue:** #10515 (ORG-05, epic #10508)
- **Branch:** feat/issue-10515-org05-desktop-navigation
- **PR:** #10539
- **Paths:** src/launchers/workspace_navigation.py; src/launchers/launcher_layout_manager.py; src/launchers/\_launcher_navigation_ui.py; src/launchers/launcher_ui_setup.py; tests/launchers/test_workspace_navigation.py; docs/development/DEVELOPMENT_LOG.md; docs/development/HANDOFF.md
- **Started:** 2026-09-19
- **Last verified:** 2026-10-11 (`944d798e`; collated from docs/development/archive/DEVELOPMENT_LOG_RECONCILIATION_AUDIT_20261011.json)
- **Summary:** Built task-oriented desktop navigation over embedded tools (ORG-05). Added 5 primary task workspaces (`Capture & Analyze`, `Model & Match`, `Shot & Course Lab`, `Optimize & Train`, `Results & Compare`) and secondary navigation (`Developer & Research`, `All Tools`, `Favorites`, `History`). Integrated `ALIAS_MAP` layout migration into `LayoutManager.load_layout` preserving user custom tile scaling, view mode, and dock state. Enforced single-instance tool reuse in `dock_widget_as_tab` and `focus_or_open_tool_tab`. Provided accessible names, keyboard navigation, narrow-window scrolling via `QScrollArea`, return-to-workspace breadcrumbs (`WorkspaceBreadcrumbBar`), and actionable status explanations for missing/unconfigured capabilities (`explain_tool_status`).
- **Next step:** Merged in PR #10539 on 2026-09-20; see parent #10508 for remaining ORG work.
- **Evidence:** tests/launchers/test_workspace_navigation.py; src/launchers/workspace_navigation.py.

### DL-#10513 · Validate Every Browser, Tauri, and Native Launch Destination

- **State:** shipped
- **Shipped:** 2026-09-20
- **Owner:** local
- **Issue:** #10513 (ORG-04, epic #10508)
- **Branch:** feat/issue-10513-org04-launch-destinations
- **PR:** #10537
- **Paths:** ui/src/routes.ts; ui/src/api/launcherReachability.ts; ui/src/api/launcherReachability.test.tsx; ui/src/App.tsx; ui/src/api/webLaunch.ts; src/config/launcher_manifest_loader.py; src/shared/python/movement_optimizer/model_pack.yaml; tests/config/launcher_manifest/test_parity.py
- **Started:** 2026-09-19
- **Last verified:** 2026-10-11 (`944d798e`; collated from docs/development/archive/DEVELOPMENT_LOG_RECONCILIATION_AUDIT_20261011.json)
- **Summary:** Established canonical route table `KNOWN_APP_ROUTES` and `isKnownAppRoute` in React UI; implemented browser and Tauri reachability evaluation and matrix generator (`evaluateTileReachability`, `generateReachabilityMatrix`); hardened `resolveTileLaunchAction` to reject unmapped routes with honest blocked/unavailable states; removed invalid web_route from Movement Optimizer model packs and sanitized in `LauncherManifestLoader` so Movement Optimizer resolves cleanly to `native-window`; expanded `test_route_mode_routes_exist_in_react_router` to inspect all loaded tiles from `LauncherManifest.load()` and added `test_every_tile_destination_resolves_authoritatively`.
- **Next step:** Merged in PR #10537 on 2026-09-20; see parent #10508 for remaining ORG work.
- **Evidence:** ui/src/api/launcherReachability.test.tsx; tests/config/launcher_manifest/test_parity.py.

### DL-#10512 · Replace Misleading Launches With Real Tasks or Explicit Nonlaunchable States

- **State:** shipped
- **Shipped:** 2026-09-20
- **Owner:** local
- **Issue:** #10512 (ORG-03, epic #10508)
- **Branch:** feat/issue-10512-org03-launch-truthfulness
- **PR:** #10536
- **Paths:** src/config/launcher_manifest.json; src/config/models.yaml; src/launchers/external_tools_adapter.py; src/launchers/launcher_model_handlers.py; src/launchers/launcher_process_manager.py; src/launchers/task_launch_truthfulness.py; tests/launchers/test_simulation_guis.py; tests/launchers/test_task_launch_truthfulness.py; ui/public/capability-atlas/graph.json; ui/public/capability-atlas/index.html
- **Started:** 2026-09-19
- **Last verified:** 2026-10-11 (`944d798e`; collated from docs/development/archive/DEVELOPMENT_LOG_RECONCILIATION_AUDIT_20261011.json)
- **Summary:** Audited capability launches and replaced misleading launch actions. Simulator prototype marked inspection/demo-only and distinct from qualified solvers; FreeMoCap parametric CLI guarded against zero-argument headless launch with parameter validation and cancellation safety; Video Analyzer placeholder fallback replaced with explicit unavailable provider diagnostic window (\_UnavailableToolWindow); library-only components (swing_optimizer, injury_analysis, pinn_pure_rigid, pinn_hybrid) and dual-shell service previews (canonical_core_estimation, canonical_core_comparison) audited with truthful dispositions, status messages, and next actions; fixed process assignment to Windows job objects for mock/invalid pids; regenerated capability atlas with verified freshness.
- **Next step:** Merged in PR #10536 on 2026-09-20; see parent #10508 for remaining ORG work.
- **Evidence:** tests/launchers/test_task_launch_truthfulness.py; tests/launchers/test_simulation_guis.py; tests/launchers/test_launcher_process_manager.py; tests/scripts/test_capability_atlas.py.

### DL-#10511 · Separate Capability Identity, Maturity, Availability, and Qualification (ORG-02)

- **State:** shipped
- **Shipped:** 2026-09-20
- **Owner:** local
- **Issue:** #10511 (ORG-02, epic #10508)
- **Branch:** feat/issue-10511-org02-capability-state-contract
- **PR:** #10535
- **Paths:** src/config/capability_state.py; src/config/launcher_manifest_loader.py; src/config/models.yaml; tests/config/launcher_manifest/test_capability_state_contract.py; docs/development/DEVELOPMENT_LOG.md; docs/development/HANDOFF.md
- **Started:** 2026-09-19
- **Last verified:** 2026-10-11 (`944d798e`; collated from docs/development/archive/DEVELOPMENT_LOG_RECONCILIATION_AUDIT_20261011.json)
- **Summary:** Disentangled overloaded capability status into four orthogonal typed dimensions: capability identity, lifecycle maturity (prototype/experimental/stable/deprecated), per-surface availability (desktop, web, api, cli) with actionable reason and remediation when unavailable, and evidence-backed qualification conforming to #10351 engine matrix contract (exempt for non-engine tools). Enforced invariant that installed engines without a qualified receipt never serialize as release-ready. Resolved Simscape/Matlab Models and Simulator/Golf Simulation Suite display names from a single canonical authority across native registry and web manifest views. Added lazy probing cache keyed on runtime identity without synchronous engine imports. Preserved backward-compatible legacy status field for API consumers.
- **Next step:** Merged in PR #10535 on 2026-09-20; see parent #10508 for remaining ORG work.
- **Evidence:** tests/config/launcher_manifest/test_capability_state_contract.py; src/config/capability_state.py; src/config/launcher_manifest_loader.py.

### DL-#10510 · Baseline Every Capability and Preserve Tile, Layout, and Artifact Identity

- **State:** shipped
- **Shipped:** 2026-09-20
- **Owner:** local
- **Issue:** #10510 (ORG-01, epic #10508)
- **Branch:** feat/issue-10510-org01-capability-baseline
- **PR:** #10534
- **Paths:** src/config/capability_migration.py; src/config/capability_migration.json; scripts/generate_capability_baseline.py; docs/development/ORG01_CAPABILITY_BASELINE.md; tests/config/test_capability_migration_coverage.py; scripts/capability_atlas/render.py; docs/architecture/CAPABILITY_ATLAS.md
- **Started:** 2026-09-19
- **Last verified:** 2026-10-11 (`944d798e`; collated from docs/development/archive/DEVELOPMENT_LOG_RECONCILIATION_AUDIT_20261011.json)
- **Summary:** Built the canonical machine-checkable capability baseline inventory (`capability_migration.json`) and schema/validation engine (`capability_migration.py`) for Epic #10508. Enforced explicit contracts: IDs are unique, aliases are acyclic and resolve to retained targets, every capability has exactly one primary workspace domain, provider absence changes availability rather than identity, and preserved test fixtures retain golden byte hashes. Preserved ADR-0047 viewer identity and provider seam rulings. Created `scripts/generate_capability_baseline.py` producing `docs/development/ORG01_CAPABILITY_BASELINE.md` with freshness validation.
- **Next step:** Merged in PR #10534 on 2026-09-20; see parent #10508 for remaining ORG work.
- **Evidence:** tests/config/test_capability_migration_coverage.py; docs/development/ORG01_CAPABILITY_BASELINE.md; src/config/capability_migration.json.

### DL-#8932 · Debounce and Memoize Advanced Analysis Tab Refreshes

- **State:** shipped
- **Shipped:** 2026-09-24
- **Owner:** antigravity
- **Issue:** #8932
- **Branch:** fix/8932-analysis-tab-debounce-idle
- **Paths:** `src/shared/python/dashboard/_analysis_refresh.py`; `src/shared/python/dashboard/advanced_analysis.py`; `tests/unit/shared_python/test_analysis_tab_refresh.py`
- **Started:** 2026-09-22
- **Last verified:** 2026-10-11 (`944d798e`; collated from docs/development/archive/DEVELOPMENT_LOG_RECONCILIATION_AUDIT_20261011.json)
- **Summary:** SpectrogramTab/WaveletTab/PhasePlaneTab/CoherenceTab spinboxes go through `DebouncedRefresh` (150 ms); CoherenceTab, SpectrogramTab, and WaveletTab memoize expensive transforms in `BoundedResultCache`; SwingPlaneTab builds its axes once and swaps artists; all six analysis tabs use non-blocking `draw_idle()`.
- **PR:** #10892
- **Next step:** Merged in PR #10892 on 2026-09-24.

### DL-#11000 · Pass the Reynolds-Curve Base Cd to the Rust Ball-Flight Kernel

- **State:** shipped
- **Owner:** claude
- **Issue:** #11000
- **PR:** #11004 (merged 642729daa)
- **Branch:** `claude/ud-rust-drag-base-cd`
- **Paths:** `src/shared/python/physics/ball_simulator.py`, `tests/unit/physics/test_rust_drag_base_coefficient.py`, `tests/unit/physics/test_launch_conditions_units_7223.py`
- **Started:** 2026-09-26
- **Last verified:** 2026-09-26 — with upstream-physics 2.1.3 built from source: trackman + launch-units + new boundary test 25 passed; tests/unit/physics + ball-flight suites 1041 passed (remaining failures identical with and without the fix, or local-environment only); parity + rust_bindings ball/aero 41 passed.
- **Summary:** `BallFlightSimulator` handed the kernel `BallProperties.cd0` (0.21, the spin polynomial's constant) as the base of its Reynolds drag curve; it now passes `GOLF_BALL_DRAG_COEFFICIENT` (0.25), as the enhanced engine does. Driver carry 275.0 -> 248.3 yd, 7-iron 194.3 -> 179.8 yd, matching the enhanced engine within 0.3 %. The degrees-regression test asserts the launch contract's refusal. The exported parity fixture stays byte-pinned and untouched.
- **Next step:** None; shipped. UD#10997 consumes the fix in the Linux unit gate.

### DL-#9406 · Retire UpstreamDrift Copies of Tools-Owned Sidekick Modules

- **State:** shipped
- **Owner:** claude (agy executor)
- **Issue:** #9406
- **PR:** #10991
- **Branch:** `claude/ud-9406-sidekick-shadow-retire`
- **Paths:** `src/shared/python/sidekick/standalone/`, `src/shared/python/sidekick/persistence/`, `src/shared/python/sidekick/__main__.py`, `src/shared/python/sidekick/ui/tools_sidebar/default_tabs.py`, `src/launchers/embedded_tool_bootstrap.py`, `sidekick.spec`, `scripts/config/unit_gate_quarantine.json`
- **Started:** 2026-09-26
- **Last verified:** 2026-09-26 — Linux CI: `test_bootstrap_registers_all_first_party_tools` still fails there (11 first-party tools do not bootstrap headless), re-quarantined; divergence inventory regenerated for the retired `sidekick/__main__.py`. Earlier: 14 target quarantine IDs pass with `-n 6 --tools-mode vendored`; affected suites (sidekick, launcher, launchers, packaging, repo_hygiene, integration/sidekick, c3d_viewer) show no new failure against the same run on `b8c27a7d2`; contract passes (141 IDs).
- **Summary:** Delete the downstream `sidekick` standalone, persistence, `__main__` and default-tabs copies so they resolve from the pinned Tools tree; put the vendored/explicit Tools paths ahead of UpstreamDrift's own in the embedded-tool bootstrap; take the Windows icon from the pinned Tools assets; retarget three tests to the vendor API.
- **Next step:** Shipped in #10991.

### DL-#10944 · Bump `vendor/ud-tools` to Tools Main With K0 and K3a

- **State:** shipped
- **Owner:** antigravity
- **Issue:** #10944
- **PR:** #10945
- **Branch:** `agy/issue-10944`
- **Paths:** `Cargo.toml`, `requirements-tools.txt`, `src/config/impact_acceptance.json`, `src/shared/python/ai/`, `src/shared/python/tour_baselines/reconciliation.py`, `docs/shared_tools/divergence_inventory.v1.json`, `docs/shared_tools/divergence_inventory.md`, `docs/agent_context/`, `tests/unit/ai/test_knowledge_and_wizards.py`, `tests/unit/tour_baselines/test_reconciliation.py`, `vendor/ud-tools`
- **Started:** 2026-09-25
- **Last verified:** 2026-09-25 — 40/40 companion tests pass, 20/20 child-copy contract tests pass, 10/10 divergence inventory tests pass, 18/18 spec changelog tests pass, 18/18 seam/pin tests pass, 3/3 knowledge/wizard tests pass, 27/27 impact acceptance tests pass, 3/3 reconciliation tests pass, agent-context check clean.
- **Summary:** Advance vendor/ud-tools submodule gitlink to 95ed6b47857e9a47211ab1973d02b28beae718bc on Tools main incorporating K0 (Tools#5348) and K3a (Tools#5350). Synchronize child copy of src/shared/python/ai/ (knowledge package, wizards.py, base adapter, panel tools, RAG deprecation). Advance Cargo.toml, requirements-tools.txt, impact_acceptance.json, reconciliation.py, regenerate divergence inventory, and render agent context.
- **Next step:** Shipped in PR #10945 (commit 2d5830d18).

### DL-#10487 · Stable PyQt Desktop Shortcuts and Consistent Taskbar and Favicon Identity

- **State:** shipped
- **Owner:** antigravity
- **Issue:** #10487
- **PR:** #10909
- **Branch:** `fix/10487-desktop-shortcuts-identity`
- **Paths:** `src/launchers/app_identity.py`, `src/launchers/desktop_shortcuts.py`, `src/launchers/upstream_drift_launcher.py`, `src/launchers/upstream_drift_launcher_main.py`, `launch_upstream_drift.py`, `scripts/create_shortcut.ps1`, `scripts/create_golf_robot_shortcut.ps1`, `ui/index.html`, `ui/public/favicon.ico`, `docs/development/desktop_viewer_setup.md`, `tests/unit/launchers/test_app_identity.py`, `tests/unit/launchers/test_desktop_shortcuts.py`
- **Started:** 2026-09-24
- **Last verified:** 2026-09-24 (`73185d259`) — verified 15 unit tests pass in test_app_identity.py and test_desktop_shortcuts.py; verified favicon bitwise parity, canonical AUMID registration, and idempotent shortcut creation
- **Summary:** Implemented canonical AppUserModelID registration and icon resolution hierarchy in app_identity.py, idempotent Desktop and Start Menu shortcut manager in desktop_shortcuts.py with readback validation, synchronized web UI favicon with launcher assets, and added setup documentation (Fixes #10487).
- **Next step:** Shipped in PR #10909.

### DL-#8886 · Unified Display Units Policy and Cross-Tool Consistency

- **State:** shipped
- **Owner:** antigravity
- **Issue:** #8886
- **PR:** #10893
- **Branch:** `fix/8886-unify-tool-units`
- **Paths:** `src/shared/python/ui/units.py`, `src/shared/python/ui/preferences_dialog.py`, `src/tools/ball_flight_gui/gui.py`, `src/tools/swing_flight_pipeline/gui.py`, `src/tools/putting_green_gui/gui.py`, `docs/development/unit_policy.md`, `docs/development/embedding_a_tool.md`, `tests/unit/shared_python/test_units.py`, `tests/tools/ball_flight_gui/test_ball_flight_gui.py`, `tests/tools/swing_flight_pipeline/test_gui.py`, `tests/tools/putting_green_gui/test_putting_green_gui.py`
- **Started:** 2026-09-24
- **Last verified:** 2026-09-24 — verified 133 unit tests pass across units conversions, settings persistence, ball flight, swing pipeline, putting green, and pose studio
- **Summary:** Implemented unified display units module with UnitSystem enum, settings persistence in PreferencesDialog, dynamic unit switching in ball flight, swing pipeline, and putting green simulators, and published fleet unit policy documentation (Fixes #8886).
- **Next step:** Shipped to main in PR #10893.

### DL-#10842 · SPEC Change Log and Root Handoff Governance

- **State:** shipped
- **Owner:** codex
- **Issue:** #10842, #10843
- **PR:** #10853
- **Branch:** `docs/10842-10843-spec-and-handoff-governance`
- **Paths:** `SPEC.md`, `AGENT_HANDOFF.md`, `docs/development/HANDOFF.md`, `docs/development/DEVELOPMENT_LOG.md`
- **Started:** 2026-09-24
- **Last verified:** 2026-09-24 — verified spec changelog rows, title-case consistency, and root handoff length <= 150 lines
- **Summary:** Registered missing Tour Baselines exact capture (#10837) and fail-closed qualification (#10841) rows in SPEC.md, and synchronized root AGENT_HANDOFF.md with active state per repository guidelines.
- **Next step:** Shipped to main in PR #10853.

### DL-#10838 · Optimize GripContactModel Slip Margin Performance

- **State:** shipped
- **Owner:** jules
- **Issue:** #10838
- **PR:** #10838
- **Branch:** `bolt-optimize-grip-margin-18311144978976259102`
- **Paths:** `src/shared/python/physics/_grip_model.py`, `tests/unit/test_grip_contact_model.py`, `.jules/bolt.md`, `SPEC.md`, `docs/development/HANDOFF.md`, `docs/development/DEVELOPMENT_LOG.md`
- **Started:** 2026-09-24
- **Last verified:** 2026-09-24 — verified slip margin speedup with float-promoted dot product to prevent integer overflow (#10845, #10848)
- **Summary:** Optimized GripContactModel.check_slip_margin by replacing np.linalg.norm with float-promoted dot product magnitude, preventing integer overflow while providing ~2x performance speedup.
- **Next step:** Shipped to main in PR #10838.

### DL-#8866 · Remove Dead Skeleton Extractors Providers From Starting Pose Matcher

- **State:** shipped
- **Owner:** issue-remediator
- **Issue:** #8866
- **PR:** #10808
- **Branch:** `staff/issue-remediator-task-2852a7`
- **Paths:** `src/tools/starting_pose_matcher/skeleton_extractors/` (deleted); `tests/unit/tools/starting_pose_matcher/test_{drake,mujoco,opensim,pinocchio}_provider.py`, `test_observed_input_providers.py`, `test_provider_error_paths.py` (deleted); `tests/tools/starting_pose_matcher/test_{observed_extractors,physics_extractors_with_stubs}.py` (deleted); `scripts/config/full_src_mypy_baseline.json`, `scripts/config/suite_marker_baseline.json`, `scripts/ci/lod_baseline.txt`, `docs/development/opensim_tour_matching/EPIC_GOLF_MODEL.md`
- **Started:** 2026-09-24
- **Last verified:** 2026-09-24 by 388650ec3 — ruff check passes, ruff format no new diffs, file-size budget passes; 9 mypy baseline stubs, 89 suite-marker baseline nodes, and 3 lod entries removed for deleted files
- **Summary:** Deleted 6 per-engine skeleton extractors in `skeleton_extractors/` (~1,614 lines) that have no callers outside tests, plus their 8 test modules; pruned stale baseline rows.
- **Next step:** Shipped to main in PR #10808.

### DL-#10593 · Tour Baselines Bounded Fit Campaigns (TB-08)

- **State:** shipped
- **Owner:** antigravity
- **Issue:** #10593 (TB-08, parent #10584, program #10363)
- **PR:** https://github.com/D-sorganization/UpstreamDrift/pull/10775
- **Branch:** `feat/tb08-bounded-fit-campaigns-10593`
- **Paths:** `src/shared/python/tour_baselines/campaign.py`; `src/shared/python/tour_baselines/__init__.py`; `src/shared/python/motion_matching/tour_baselines.py`; `tests/unit/tour_baselines/test_fit_campaign.py`
- **Started:** 2026-09-23
- **Last verified:** 2026-09-23 — 7 passed in `test_fit_campaign.py`, 64 passed in full `tests/unit/tour_baselines/` suite, 20 passed in repo hygiene child-copy contract. Ruff check and format clean. Merged via PR #10775.
- **Summary:** Implemented `CampaignJobSpec`, `CampaignCandidate`, `CandidateRanking`, `rank_candidates`, `CampaignEvaluationRecord`, `CampaignManifest`, `CampaignResult`, `PilotBudget`, `GeneralizationDisclaimer`, and `FitCampaignService` enforcing deterministic Pareto ranking (feasible candidate beats infeasible lower-error candidate), immutable checkpointing, hash-checked resume (`IncompatibleResumeError`), diagnostic preservation on cancel/timeout without promotion, exact full-clock score verification, and holdout disclaimers.
- **Next step:** Landed on main. Proceed to TB-09 (#10594).

### DL-#10594 · Tour Baselines Independent Qualification (TB-09)

- **State:** shipped
- **Owner:** antigravity
- **Issue:** #10594 (TB-09, parent #10584, program #10363)
- **PR:** https://github.com/D-sorganization/UpstreamDrift/pull/10785
- **Branch:** `feat/tb09-independent-qualification-10594`
- **Paths:** `src/shared/python/tour_baselines/qualification.py`; `src/shared/python/tour_baselines/__init__.py`; `src/shared/python/tour_baselines/coverage.py`; `tests/unit/tour_baselines/test_qualification.py`; `tests/unit/tour_baselines/test_model_identities.py`
- **Started:** 2026-09-23
- **Last verified:** 2026-09-24 — 20 passed in `test_qualification.py`, 84 passed in full `tests/unit/tour_baselines/` suite. Merged via PR #10785.
- **Summary:** Implemented `IntegrityViolation`, `IntegrityReport`, `RolloutReconstructionResult`, `ConstraintEvaluationResult`, `RecomputedMetrics`, `EndpointCheckResult`, `ModelAdequacyDecomposition`, `compare_cross_complexity`, `RefinementSensitivityRecord`, `ForceIdentifiabilityDisclaimer`, `RosterVerdict`, `evaluate_full_roster_qualification`, and `ExpertSignoff` enforcing cryptographic hash integrity, forward dynamic rollout reconstruction from single (q0, v0), physical and geometric constraint evaluation, independent metric recomputation, 3-way model adequacy decomposition, cross-complexity comparison over common observation sets, refinement sensitivity under dt, force identifiability disclaimer, full 40-cell coverage matrix qualification verdicts with strict G3 reduced model exclusion, and auditable JSON export.
- **Next step:** Landed on main. Proceed to TB-10 (#10595).

### DL-#10595 · Tour Baselines Discovery, Portable Loading and Safe Model Presets (TB-10)

- **State:** shipped
- **Owner:** antigravity
- **Issue:** #10595 (TB-10, parent #10584, program #10363)
- **PR:** https://github.com/D-sorganization/UpstreamDrift/pull/10793
- **Branch:** `feat/tb10-baseline-discovery-presets-10595`
- **Paths:** `src/shared/python/tour_baselines/discovery.py`; `src/shared/python/tour_baselines/__init__.py`; `src/shared/python/tour_baselines/qualification.py`; `src/shared/python/motion_matching/ledger.py`; `tests/unit/tour_baselines/test_discovery.py`; `tests/unit/tour_baselines/test_qualification.py`; `docs/shared_tools/divergence_inventory.v1.json`; `docs/shared_tools/divergence_inventory.md`
- **Started:** 2026-09-23
- **Last verified:** 2026-09-24 — 14 passed in `test_discovery.py`, 20 passed in `test_qualification.py`, 98 passed in full `tests/unit/tour_baselines/` suite. Merged via PR #10793.
- **Summary:** Implemented `SafeModelPreset`, `IncompatiblePresetError`, `MissingDependencyError`, `BaselineNotFoundError`, `BaselineFilter`, `BaselineSummary`, `BaselineDetail`, `BaselineDiscoveryService`, `export_to_ledger_rows`, and headless CLI supporting catalog scanning across configurable search paths, multi-field filtering (model, club, horizon, qualification status), fail-closed preset compatibility checks (refusing topology mismatches and missing solver dependencies), fail-closed default preset selection (unverified packages cannot be auto-selected), safe session cloning preserving user workspace, portable export and clean-machine import with SHA-256 verification and dependency diagnostics, deterministic re-indexing, and result index ledger conversion. Also remediated bot review items #10786, #10787, #10789, #10790.
- **Next step:** Landed on main. Address follow-up bot reviews #10794 and #10795.

### DL-#10794 · Reconstruct Presets From Verified Arrays & Qualifiable Pendulum Baselines (#10794, #10795)

- **State:** shipped
- **Owner:** antigravity
- **Issue:** #10794, #10795
- **PR:** https://github.com/D-sorganization/UpstreamDrift/pull/10798
- **Branch:** `fix/review-feedback-10794-10795`
- **Paths:** `src/shared/python/tour_baselines/discovery.py`; `src/shared/python/tour_baselines/qualification.py`; `src/shared/python/tour_baselines/__init__.py`; `src/engines/physics_engines/pendulum/python/motion_matching/qualification.py`; `tests/unit/tour_baselines/test_discovery.py`; `tests/unit/tour_baselines/test_qualification.py`
- **Started:** 2026-09-23
- **Last verified:** 2026-09-24 — Merged cleanly in PR #10798 with 100% green CI (all 53 checks passing).
- **Summary:** Remediated bot review feedback on TB-10: reconstructed `SafeModelPreset` from verified archive array members and added `np.array_equal` check against manifest-embedded arrays to prevent manifest tampering (#10795); evaluated continuous torque into `tau`, recorded `time`, set `horizon="G1"`, and generated non-empty identity hashes in pendulum `_assemble_baseline_package`, and implemented `migrate_legacy_package` in `qualification.py` with auto-migration support in `IndependentBaselineQualifier.qualify()` (#10794).
- **Next step:** Landed on main. Proceed to TB-11 (#10596).

### DL-#10596 · Expose Tour Baselines in Motion Matching, Pendulum Tools and Replay (TB-11)

- **State:** shipped
- **Owner:** antigravity
- **Issue:** #10596 (TB-11, parent #10584, program #10363)
- **PR:** https://github.com/D-sorganization/UpstreamDrift/pull/10809
- **Branch:** `feat/tb11-expose-tour-baselines-10596`
- **Paths:** `src/tools/motion_matching/tour_baselines_presenter.py`; `src/tools/motion_matching/tour_baselines_widget.py`; `src/tools/motion_matching/gui.py`; `tests/unit/motion_matching/test_tour_baselines_presenter.py`; `tests/tools/motion_matching/test_motion_matching_gui.py`; `SPEC.md`
- **Started:** 2026-09-23
- **Last verified:** 2026-09-24 — Exact-capture filtering applied to presenter roster package flags (#10829) and comparison (#10826); 10 unit tests passing in tests/unit/motion_matching/test_tour_baselines_presenter.py.
- **Summary:** Implemented `TourBaselinesPresenter`, `TourBaselineDetailView`, `ModelItemView`, `WhereThisCameFromView`, `BaselineOpenResult`, `ModelComparisonReport`, `EvidenceInspectionReport`, and `ComputeBudgetView` exposing the canonical two-capture coverage matrix, accessible status badges with text labels and symbols without relying on color alone, visual semantics distinguishing 3D and projected 2D views and observed markers from simulated meshes, plain-language "Where This Came From" metadata linking raw hashes, preprocessing, geometry, fit configs, and scientific limitations (force identifiability and holdout disclaimers), and session actions (Open in Viewer, Clone for Experiment, Compare Models, Inspect Evidence, Reproduce). Enforced exact capture matching in roster package flags and detail views to eliminate cross-capture fallbacks. Integrated "Tour Baselines" tab into `MotionMatchingWidget` with interactive capture/model selectors, immediate metadata display, collapsible provenance panel, and action button dispatches.
- **Next step:** Landed on main. Proceed to TB-12 (#10597).

### DL-#10774 · Deferred Validation Project Projection

- **State:** shipped
- **Owner:** codex
- **Issue:** #10774; parents Repository_Management#1687 and Runner_Dashboard#1248
- **Branch:** `docs/deferred-project-projection`
- **PR:** https://github.com/D-sorganization/UpstreamDrift/pull/10776
- **Paths:** `docs/project/`, `docs/index.md`, `docs/README.md`, `docs/development/HANDOFF.md`, `SPEC.md`
- **Started:** 2026-09-23
- **Last verified:** 2026-09-23 (8c15de9dafdc0caf5d172c1aa97037ffd4655c74; merged #10776; six parked owner rows/links verified in post-reboot central UI/API receipt; original plans unchanged)
- **Summary:** Projects all six published external-validation owner plans without inventing evidence, resource approval or completion. Software and research authorities are preserved.
- **Next step:** Projection published; Board decisions remain pending. Enforcement continues in DL-#10783.

### DL-#10750 · Keep Test-Generated JSON Out of Committed Working Tree

- **State:** shipped
- **Owner:** antigravity
- **Issue:** #10750
- **Branch:** fix/10750-tests-in-place-json-mutation
- **PR:** #10761
- **Paths:** src/shared/python/motion_matching/club_only/fast_matching.py; tests/unit/motion_matching/test_club_fast_matching.py; src/engines/physics_engines/mujoco/python/humanoid_launcher.py; tests/unit/test_gui_coverage.py
- **Started:** 2026-09-23
- **Last verified:** 2026-09-23 (`008487870`)
- **Summary:** Thread optional evidence_dir through save_fast_match_evidence and point tests at tmp_path; support optional config_path and avoid unconditional save_config on HumanoidLauncher instantiation so tests never rewrite committed JSON artifacts.
- **Evidence:** PR #10761 merged cleanly to main at 008487870 with all CI checks passing.

### DL-#10591 · Constrained Upper-Body Golfer Baseline (TB-06)

- **State:** shipped
- **Owner:** codex
- **Issue:** #10591 (TB-06, parent #10584, program #10363)
- **Branch:** feat/10591-upper-body-capture (merged)
- **PR:** #10740 (`56ad6a8dcd13221699ec14442b13c1df6fc5729a`, merged)
- **Paths:** src/shared/python/pendulum_simulator/upper_body_replay.py; src/shared/python/pendulum_simulator/simulation_core.py; src/shared/python/motion_matching/bernstein_controls.py; src/engines/physics_engines/pendulum/python/motion_matching/adapters_golfer.py; src/engines/physics_engines/pendulum/python/motion_matching/torque_optimization_golfer.py; src/engines/physics_engines/pendulum/python/motion_matching/torque_optimization.py; src/engines/physics_engines/pendulum/python/motion_matching/club_pendulum_match.py; src/engines/physics_engines/pendulum/python/motion_matching/upper_body_capture.py; scripts/motion_capture/upper_body_planarity_receipt.py; src/shared/python/tour_baselines/coverage.py; docs/plans/tour_baselines/coverage_matrix.md; docs/plans/tour_baselines/evidence/tb06_driver_planarity_receipt.json; docs/plans/tour_baselines/evidence/tb06_iron_planarity_receipt.json; reports/matched_swing_ledger.json; tests/unit/engines/physics_engines/pendulum/test_golfer_fit.py; tests/unit/pendulum_simulator/test_upper_body_replay.py; tests/unit/motion_matching/test_bernstein_controls.py; AGENT_HANDOFF.md; docs/development/DEVELOPMENT_LOG.md; SPEC.md
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 — reproducible C3D preflight receipts preserve raw 360 Hz/654-frame Driver and 359 Hz/657-frame Iron clocks, shared 1 kHz evaluation grid, source hashes, and full coverage for six declared upper-body markers. Their 110.4/112.7 mm normal-RMSE lower bounds exceed the 55 mm profile ceiling, so both are rejected before unsupported planar attachment or torque fitting.
- **Summary:** The native closed-loop replay and bounded torque fitter remain traceable, but the actual Driver and Iron source campaigns are scientifically disqualified by the fixed-plane lower bound. The coverage matrix and matched-swing ledger retain the two receipts as rejected outcomes rather than leaving a pending campaign or fabricating calibration.
- **Next step:** Open a separately scoped spatial-topology issue only if a new model is authorized.

### DL-#10592 · Reconcile Existing Reference and Full-Body Results (TB-07)

- **State:** shipped
- **Owner:** codex
- **Issue:** #10592 (TB-07, parent #10584, program #10363)
- **PR:** #10730 (merged)
- **Paths:** src/shared/python/tour_baselines/coverage.py; src/shared/python/motion_matching/ledger.py; tests/unit/tour_baselines/test_coverage_matrix.py; tests/unit/motion_matching/test_ledger.py; docs/plans/tour_baselines/coverage_matrix.md; docs/development/matched_swing_program/README.md; reports/matched_swing_ledger.json; AGENT_HANDOFF.md; docs/development/DEVELOPMENT_LOG.md; SPEC.md
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 at `22b7fd7a534ddfbc0b1db943e6443fdc9ac7efb3` — merged after focused tests, Ruff, architecture/file-size budgets, and required PR checks passed.
- **Summary:** Reconcile tour-baseline reporting with immutable TB-04 receipt verdicts. The coverage matrix and unified run ledger preserve replay evidence while rejecting the two disqualified candidates; historical/reduced evidence remains distinct from full-body G1/G2/G3 qualification.

### DL-#8887 · Wire Per-Engine Joint Limits Into Pose Studio's JointPanel

- **State:** shipped
- **Owner:** claude
- **Issue:** #8887
- **Branch:** fix/8887-pose-studio-joint-limits
- **PR:** #10650
- **Paths:** src/shared/python/pose_interchange/live_kinematics.py; src/shared/python/pose_interchange/services/\_mock.py; src/shared/python/pose_interchange/services/drake.py; src/shared/python/pose_interchange/services/mujoco.py; src/shared/python/pose_interchange/services/myosuite.py; src/shared/python/pose_interchange/services/opensim.py; src/shared/python/pose_interchange/services/pinocchio.py; src/shared/python/pose_interchange/services/simscape.py; src/tools/pose_studio/controllers/engine_controller.py; src/tools/pose_studio/gui.py; src/tools/pose_studio/widgets/joint_panel.py; tests/tools/pose_studio/test_engine_controller_internals.py; tests/unit/tools/pose_studio/test_gui.py; tests/unit/tools/pose_studio/test_joint_panel.py
- **Started:** 2026-09-21
- **Last verified:** 2026-09-22 at c3773d62c — merged to main via PR #10650.
- **Summary:** `LiveKinematicsService.joint_limits()` extends the kinematics-service protocol (every engine service implements it, `{}` pending real anatomical data); `JointPanel.set_limits()`/`set_error()` re-range joints per engine and give visible feedback on a rejected edit; wired from `MainWidget` on init, engine switch, and angle-edit rejection/success.
- **Next step:** N/A — shipped via #10650.

### DL-#10379 · Reliable Motion-Matching Jobs, Recovery and Portable Results (MS-105)

- **State:** shipped
- **Owner:** local
- **Issue:** #10379 (MS-105, epic #10363; folded PF-08 #10438)
- **Branch:** feat/10379-ms105-jobs-recovery
- **PR:** #10704
- **Paths:** src/shared/python/motion_matching/jobs/; tests/unit/motion_matching/jobs/test_matching_jobs.py; docs/plans/matched_swing/evidence/ms105_jobs_recovery.json; docs/development/HANDOFF.md; docs/development/DEVELOPMENT_LOG.md; SPEC.md
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 at 901b2de5e — merged to main via PR #10704.
- **Summary:** Matching job contracts with atomic manifests/checkpoints, compatible resume, fault recovery, process-tree cancel, portable packages, both-shell progress/failure views, and PF-08 service budgets (`guarantee=false`). Reuses `#8880`/`async_action` and `managed_popen`; no second scheduler.
- **Next step:** N/A — shipped via #10704.

### DL-#10347 · Simscape R2025b Run Management (Run-102 Package)

- **State:** shipped
- **Owner:** local
- **Issue:** #10347 (MS-60, epic #10363)
- **Branch:** fix/issue-10347-ms60-run-management
- **PR:** #10669
- **Paths:** scripts/matlab/run_simscape_candidate.ps1; src/shared/python/motion_matching/simscape_run_manifest.py; src/shared/python/motion_matching/candidate_convert.py; src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/{write_run_manifest.m,export_candidate.m}; docs/development/simscape_tour_matching/{CHECKPOINTS.md,CHECKPOINTS_HISTORY.md}; docs/development/simscape_tour_matching/native_evidence/two_window_fit_9967_102/{candidate.npz,run_manifest.json,playback.gif}
- **Started:** 2026-09-21
- **Last verified:** 2026-09-21 (merged to main as #10669)
- **Summary:** Documented one R2025b-only scripted replay path with fail-closed run manifest (host, release, model/candidate/replay SHAs, wall-clock), converted returned-replay NPZ into MatchedSwingCandidate via DRY reuse of returned81 layout, and committed run-102 playback GIF in-tree.
- **Next step:** <host> second-person replay under 30 minutes when MS-61 Fit is scheduled.
- **Evidence:** docs/development/simscape_tour_matching/native_evidence/two_window_fit_9967_102/{candidate.npz,run_manifest.json,playback.gif,qualified_candidate_replay.json}; tests/unit/motion_matching/test_simscape_candidate_convert.py; docs/shared_tools/divergence_inventory.v1.json.

### DL-#10588 · Calibrate Swing Planes, Fixed Geometry and Feasible Initial States

- **State:** shipped
- **Owner:** local
- **Issue:** #10588 (TB-03, parent #10584, program #10363)
- **Branch:** feat/tb03-trajectory-fitting-10588
- **PR:** #10631
- **Paths:** src/shared/python/motion_matching/projection_2d.py; src/shared/python/tour_baselines/calibration.py; src/shared/python/tour_baselines/**init**.py; tests/unit/motion_matching/test_projection_2d.py; tests/unit/tour_baselines/test_tour_calibration.py; docs/plans/tour_baselines/plane_calibration_and_initial_states.md
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 (All 60 unit tests pass across tour baselines and projection_2d suites; architecture budget passes; DRY duplication gate passes; suite marker ratchet passes; ruff clean; mypy 0 errors across 15 files).
- **Summary:** Replaced naive z-drop in `projection_2d.py` with `CalibratedSwingPlane` implementing rigid SE(3) transform, orthonormal right-handed SO(3) basis, inclination, azimuth, and `GeometricProjectionResidual` reporting RMSE and max deviation. Implemented `estimate_swing_plane` fitting one rigid plane per declared capture window, handling degeneracy (collinear points, rank < 2) and reflections. Implemented `calibrate_fixed_geometry` calibrating positive bounded link lengths (L1, L2) with frozen nonidentifiable mass/inertia priors and Fisher sensitivity rank diagnostic. Implemented `map_initial_state_double_pendulum` mapping t0 observations to generalized coordinates (theta1, theta2) and velocities with gap validation and verified forward kinematics. Implemented `compute_moving_hub_power` tracking external trajectory, velocity, power, and integrated work for prescribed moving hubs.
- **Next step:** Shipped in PR #10631 (commit cfcc8dfbd). Proceeded to TB-04 (#10589).
- **Evidence:** tests/unit/motion_matching/test_projection_2d.py; tests/unit/tour_baselines/test_tour_calibration.py; docs/plans/tour_baselines/plane_calibration_and_initial_states.md.

### DL-#10621 · NM-06 Masked Trajectory-to-Control Proposals

- **State:** shipped
- **Owner:** local
- **Issue:** #10621 (epic #10603)
- **Branch:** feat/10621-nm06-masked-proposals
- **PR:** #10709
- **Paths:** src/shared/python/neural_motion/proposals/; src/shared/python/motion_matching/inverse/{**init**,masked_proposal,proposal_shared,proposal_training,regressor_training,basis_time,collapse}.py; src/shared/python/motion_matching/hybrid.py; tests/unit/neural_motion/test_masked_proposals_nm06.py; tests/unit/motion_matching/test_masked_proposals_nm06.py; tests/unit/motion_matching/test_inverse_regressor_training.py; docs/plans/neural_motion_matching/masked_proposals.md; docs/plans/neural_motion_matching/evidence/nm06_masked_proposals_receipt.json
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 — merged via PR #10709 (`08bcec302`; tip `05f712281`)
- **Summary:** neural_motion/proposals package binds task u_dim, masked conditioning, selection/mixture heads, observation+regularization training, hybrid fail-closed polish, and strict checkpoints; inverse proposal_shared consolidates duplicated contract logic; proposal_training and regressor_training stay under function-line/parameter budgets; inverse package lazily loads torch-backed cVAE/regressor exports for unit-lane collection.
- **Next step:** N/A — shipped; check NM-07 #10622 claim before any start (do not steal claim:antigravity).
- **Evidence:** docs/plans/neural_motion_matching/masked_proposals.md; docs/plans/neural_motion_matching/evidence/nm06_masked_proposals_receipt.json

### DL-#10620 · NM-05 Classical and Small Neural Dynamics Baselines

- **State:** shipped
- **Owner:** local
- **Issue:** #10620 (epic #10603)
- **Branch:** feat/issue-10620-nm05-baselines
- **PR:** #10701
- **Paths:** src/shared/python/neural_motion/baselines/; src/shared/python/neural_motion/episodes/splits.py; tests/unit/neural_motion/test_dynamics_baselines_nm05.py; docs/plans/neural_motion_matching/dynamics_baselines.md; docs/plans/neural_motion_matching/evidence/nm05_dynamics_baselines_receipt.json
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 — merged via PR #10701
- **Summary:** Pilot dynamics baselines over NM-03 episode store: `DynamicsBaselineTrainerConfig`, split helpers, `FamilySplitPlan.ids_for`; analytical 1/2-DOF fixture map, ridge, nearest neighbor, optional small MLP; trial-level splits, train-only normalizer digest, inverse conditioning, identity-leakage and unavailable-torque guards; validation checkpointing with test untouched; honest analytical-vs-MLP comparison on fixtures.
- **Next step:** Continue NM-06 (#10621) under frozen baselines and episode-store contracts.
- **Evidence:** docs/plans/neural_motion_matching/dynamics_baselines.md; docs/plans/neural_motion_matching/evidence/nm05_dynamics_baselines_receipt.json

### DL-#10619 · NM-04 Teacher Episodes and Active-Learning Candidates

- **State:** shipped
- **Owner:** local
- **Issue:** #10619 (epic #10603)
- **Branch:** local/nm-04-teacher-episodes
- **PR:** #10698
- **Paths:** src/shared/python/neural_motion/teachers/; src/shared/python/training/scheduler.py; src/shared/python/training/datasets.py; tests/unit/neural_motion/test_teacher_episodes_nm04.py; docs/plans/neural_motion_matching/teacher_episodes.md; docs/plans/neural_motion_matching/evidence/nm04_teacher_episodes_receipt.json
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 — merged to main via #10698 (`2142d380b`)
- **Summary:** Versioned teacher generation (`neural-teacher-episodes/1.0.0`) with near-baseline/stratified/low-discrepancy/random-torque paths, rejection ledger and quarantine, nested corpus stages with resume/duplicate-seed avoidance, and active acquisition (`neural-acquisition-log/1.0.0`) that cannot consume test labels. Reuses NM-03 EpisodeStore and NM-01 nested stage sizes; training scheduler admits teacher budgets via `neural_teacher_corpus_budget` and datasets register teacher corpus paths without all-RAM load.
- **Next step:** Continue NM-05 (#10620) under frozen teacher and episode-store contracts.
- **Evidence:** docs/plans/neural_motion_matching/teacher_episodes.md; docs/plans/neural_motion_matching/evidence/nm04_teacher_episodes_receipt.json.

### DL-#10618 · NM-03 Episode Storage Splits and Dataset Views

- **State:** shipped
- **Owner:** local
- **Issue:** #10618 (epic #10603)
- **Branch:** feat/10618-nm03-episode-storage
- **PR:** #10686
- **Paths:** src/shared/python/neural_motion/episodes/; src/shared/python/neural_motion/json_io.py; src/shared/python/neural_motion/experiment.py; src/shared/python/training/datasets.py; tests/unit/neural_motion/test_episode_store_nm03.py; docs/plans/neural_motion_matching/episode_storage.md; docs/plans/neural_motion_matching/evidence/nm03_episode_storage_receipt.json
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 — merged to main via #10686 (`800703cb`)
- **Summary:** Versioned `neural-episode-store/1.0.0` HDF5 shards with content hashes; family-level splits with held-out strata and real-data eval bucket; compact-1.0 adapter via `CompactArrayBundle` preserves 27/189 layout; thin task views and transform-keyed window cache; training registry registers corpus paths without all-RAM load.
- **Next step:** Continue NM-04 (#10619) under frozen episode-store contracts.
- **Evidence:** docs/plans/neural_motion_matching/episode_storage.md; docs/plans/neural_motion_matching/evidence/nm03_episode_storage_receipt.json.

### DL-#10617 · NM-02 Native Dataset Labels Complete and Semantically Correct

- **State:** shipped
- **Owner:** local
- **Issue:** #10617 (epic #10603)
- **Branch:** fix/10617-nm02-native-dataset-labels
- **PR:** #10679
- **Paths:** src/shared/python/data_io/dataset_generator/{core,channel_finalize,sim_buffers,sim_recording,models,labels,adapters,**init**}.py; src/shared/python/engine_core/mock_engine.py; tests/unit/data_io/test_dataset_labels_nm02.py; tests/unit/data_io/test_nm02_adapters.py; docs/plans/neural_motion_matching/dataset_labels.md; docs/plans/neural_motion_matching/evidence/nm02_first_wave_label_receipts.json; docs/shared_tools/divergence_inventory.v1.json
- **Started:** 2026-09-21
- **Last verified:** 2026-09-22 — merged to main via #10679 (`4bb10daa0`)
- **Summary:** Completes DatasetGenerator channel evidence (no zero-as-measurement), native vs interval accelerations, requested/applied controls, DoF layout and root-force gate, restore StateError, residual helper, and first-wave mock+ODE qualification receipts keyed to NM-01 pilot roster. No training or speed claim.
- **Next step:** Continue NM-03 (#10618) under frozen learning contracts.
- **Evidence:** docs/plans/neural_motion_matching/dataset_labels.md; docs/plans/neural_motion_matching/evidence/nm02_first_wave_label_receipts.json.

### DL-#10616 · NM-01 Freeze Learning Tasks Model Roster and Benefit Experiment

- **State:** shipped
- **Owner:** local
- **Issue:** #10616 (epic #10603)
- **Branch:** feat/nm01-freeze-learning-tasks
- **PR:** #10672
- **Paths:** src/shared/python/neural*motion/tasks.py; src/shared/python/neural_motion/roster.py; src/shared/python/neural_motion/experiment.py; src/shared/python/neural_motion/**init**.py; tests/unit/neural_motion/test_learning_tasks.py; tests/unit/neural_motion/test_model_roster.py; tests/unit/neural_motion/test_benefit_experiment.py; tests/unit/neural_motion/test_nm01_dbc_optimize.py; docs/plans/neural_motion_matching/learning_freeze.md; docs/plans/neural_motion_matching/evidence/nm01*\*.json
- **Started:** 2026-09-21
- **Last verified:** 2026-09-21 — merged to main via #10672 (`f2e625099`)
- **Summary:** Freezes typed forward/inverse/masked learning-task contracts (dimensions from TB-00 identities; inverse non-uniqueness policy required), a 20-model neural roster with full-body deferred pending benefit, and the benefit experiment (nested 100/500/2000 stages, three seeds, five baselines, all-phase latency including failures, 2× median/non-worse p95 gates, break-even None when savings ≤ 0). No training or speed claim.
- **Next step:** Continue NM-02 (#10617) under the frozen learning contracts.
- **Evidence:** docs/plans/neural_motion_matching/learning_freeze.md; docs/plans/neural_motion_matching/evidence/nm01_learning_tasks_pilot.json; docs/plans/neural_motion_matching/evidence/nm01_model_roster.json; docs/plans/neural_motion_matching/evidence/nm01_benefit_experiment_receipt.json.

### DL-#10615 · NM-00 Dataset Checkpoint and Training Claim Audit

- **State:** shipped
- **Owner:** local
- **Issue:** #10615 (epic #10603)
- **Branch:** fix/issue-10615-nm00-dataset-audit
- **PR:** #10668
- **Paths:** src/shared/python/neural_motion/; src/shared/python/motion_matching/surrogate/artifact_paths.py; src/shared/python/motion_matching/surrogate/nm00_audit.py; src/shared/python/motion_matching/surrogate/perstep/extract_dataset.py; tests/unit/neural_motion/; docs/plans/neural_motion_matching/artifact_audit.md; docs/plans/neural_motion_matching/evidence/; docs/shared_tools/divergence_inventory.v1.json
- **Started:** 2026-09-21
- **Last verified:** 2026-09-21 at 839b645dc (merged to main via #10668)
- **Summary:** Fail-closed inventory of neural corpora, default checkpoints and historical training claims with retain/repair/migrate/reject/quarantine dispositions and a per-model coverage matrix keyed to TB-00 identities. No native training or speed claim.
- **Next step:** Continue NM-01 (#10616) under the frozen audit inventory.
- **Evidence:** docs/plans/neural_motion_matching/artifact_audit.md; docs/plans/neural_motion_matching/evidence/nm00_artifact_audit_receipt.json; docs/plans/neural_motion_matching/evidence/nm00_coverage_matrix.json.

### DL-#10604 · CO-00 Freeze Club Workbook Identity

- **State:** shipped
- **Owner:** local
- **Issue:** #10604 (epic #10602)
- **Branch:** fix/issue-10604-co00-workbook-identity
- **PR:** #10667 (merged)
- **Paths:** src/shared/python/motion_matching/club_only/; src/shared/python/motion_matching/loaders/event_labels.py; src/shared/python/motion_matching/loaders/excel.py; src/engines/physics_engines/pinocchio/python/motion_training/club_trajectory_parser.py; tests/unit/motion_matching/test_club_workbook_identity.py; docs/plans/club_only_matching/evidence/club_workbook_identity.json
- **Started:** 2026-09-21
- **Last verified:** 2026-09-21 — merged via #10667 onto main as prerequisite for CO-01
- **Summary:** Freezes hash-verified workbook manifests, four-trial lineage (Filtering Experiments aliases TW_ProV1), centimetre unit authority with inches declaration retained, native 240 Hz impact-relative events, and orientation derivation policy. Shared event-label normalization and axis-component helper remove silent NaN/zero bugs across Excel and Pinocchio loaders.
- **Next step:** Continue on #10605 (CO-01) observation contracts.
- **Evidence:** docs/plans/club_only_matching/evidence/club_workbook_identity.json; tests/unit/motion_matching/test_club_workbook_identity.py.

### DL-#10605 · CO-01 Extend Canonical Club Observation Contracts and Calibration

- **State:** shipped
- **Owner:** local
- **Issue:** #10605 (epic #10602)
- **Branch:** fix/issue-10605-co01-club-observation
- **PR:** #10670 (merged)
- **Paths:** src/shared/python/motion_matching/club_only/observation.py; src/shared/python/motion_matching/club_only/adapters.py; src/shared/python/motion_matching/club_calibration.py; src/shared/python/motion_matching/target.py; tests/unit/motion_matching/test_club_observation_contracts.py; docs/plans/club_only_matching/evidence/club_observation_contracts.json; docs/shared_tools/divergence_inventory.v1.json; docs/shared_tools/divergence_inventory.md
- **Started:** 2026-09-21
- **Last verified:** 2026-09-21 — merged via #10670 onto main as prerequisite for CO-02
- **Summary:** Adds `ClubObservation` with measured/derived/unobserved component masks, dual mid-hands/face frames and orientations, native 240 Hz clock, uncertainty/derivation metadata, SO(3) residuals/interpolation, catalog-backed `club_calibration` (fixed tool-to-model SE(3), grip-face rigidity, mid-hands→butt-end only with explicit offset), and legacy `ClubTarget` adapters that refuse invented identity quats.
- **Next step:** Continue on #10606 (CO-02) plausibility priors and acceptance.
- **Evidence:** docs/plans/club_only_matching/evidence/club_observation_contracts.json; tests/unit/motion_matching/test_club_observation_contracts.py.

### DL-#10606 · CO-02 Define Golf Plausibility Priors, Ambiguity and Acceptance

- **State:** shipped
- **Owner:** local
- **Issue:** #10606 (epic #10602)
- **Branch:** feat/co02-golf-plausibility-priors
- **PR:** #10675 (merged)
- **Paths:** src/shared/python/motion_matching/club_only/priors.py; src/shared/python/motion_matching/club_only/profiles.py; src/shared/python/motion_matching/club_only/ambiguity.py; src/shared/python/motion_matching/club_only/acceptance.py; tests/unit/motion_matching/test_club_plausibility_acceptance.py; docs/plans/club_only_matching/evidence/club_plausibility_acceptance.json
- **Started:** 2026-09-21
- **Last verified:** 2026-09-21 — merged via #10675 onto main as prerequisite for CO-03
- **Summary:** Freezes per-roster observation/physical/plausibility profiles, named GolfPlausibilityPriors (not measured truth), ambiguity retention for distinct body hashes on identical club residuals, and club-only acceptance that keeps kinematic preview, torque replay, scientific, and product statuses separate while preserving full-body G3 gates.
- **Next step:** Continue on #10607 (CO-03) retrieval and constrained IK starting guesses.
- **Evidence:** docs/plans/club_only_matching/evidence/club_plausibility_acceptance.json; tests/unit/motion_matching/test_club_plausibility_acceptance.py.

### DL-#10607 · CO-03 Build Retrieval and Constrained IK Starting Guesses

- **State:** shipped
- **Owner:** local
- **Issue:** #10607 (epic #10602)
- **Branch:** feat/10607-co03-retrieval-constrained-ik
- **PR:** #10678 (merged)
- **Paths:** src/shared/python/motion_matching/club_only/hand_geometry.py; src/shared/python/motion_matching/club_only/retrieval.py; src/shared/python/motion_matching/club_only/constrained_ik.py; src/shared/python/motion_matching/club_only/seeds.py; tests/unit/motion_matching/test_club_starting_guesses.py; docs/plans/club_only_matching/evidence/club_starting_guesses.json; docs/shared_tools/divergence_inventory.v1.json
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 — merged via #10678 onto main as prerequisite for CO-04/CO-05
- **Summary:** Adds handedness-aware model hand-frame offsets, library retrieval with one rigid placement and native-clock preservation, constrained-IK seeds with distinct Pink/DLS capability records (unsupported constraints fail closed), and a geometry/profile-keyed seed cache. Four-trial retrieval-only and constrained-IK baselines are kinematic previews only.
- **Next step:** Continue on #10608 (CO-04) and #10609 (CO-05) in parallel on non-overlapping paths.
- **Evidence:** docs/plans/club_only_matching/evidence/club_starting_guesses.json; tests/unit/motion_matching/test_club_starting_guesses.py.

### DL-#10608 · CO-04 Match Club-Only Motion With Double and Triple Pendulums

- **State:** shipped
- **Owner:** local
- **Issue:** #10608 (epic #10602)
- **Branch:** feat/issue-10608-co04-pendulum-club-match
- **PR:** #10680 (merged)
- **Paths:** src/shared/python/motion_matching/club_only/{hub_accounting,match_errors,pendulum_match,replay_package}.py; src/engines/physics_engines/pendulum/python/motion_matching/{club_pendulum_match,club_match_matrix}.py; tests/unit/motion_matching/test_club_pendulum_match.py; docs/plans/club_only_matching/evidence/club_pendulum_match.json; docs/shared_tools/divergence_inventory.v1.json
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 — merged via #10680 onto main as prerequisite for CO-06
- **Summary:** Club-only double/triple pendulum matching consumes driven adapters, separates in-plane vs 3D errors and fixed-pivot vs prescribed moving-hub IDs with external-work accounting, warm-starts from valid CO-03 seeds, scores frame 0 before integrate, retains best of cold vs retrieval, and saves replay packages without inventing native G1 pass. Fit orchestration lives in the pendulum engine package so shared never imports engines.
- **Next step:** Continue on #10610 (CO-06).
- **Evidence:** docs/plans/club_only_matching/evidence/club_pendulum_match.json; tests/unit/motion_matching/test_club_pendulum_match.py.

### DL-#10609 · CO-05 Generate Plausible Upper-Body and Full-Body Candidates

- **State:** shipped
- **Owner:** local
- **Issue:** #10609 (epic #10602)
- **Branch:** feat/issue-10609-co05-plausible-body-candidates
- **PR:** #10681 (merged)
- **Paths:** src/shared/python/motion_matching/club_only/body_candidates.py; src/shared/python/motion_matching/club_only/topology_mapping.py; src/shared/python/motion_matching/club_only/nullspace_proposals.py; src/shared/python/motion_matching/club_only/observation.py; src/shared/python/motion_matching/club_only/profiles.py; src/shared/python/motion_matching/club_only/seeds.py; src/shared/python/motion_matching/club_only/**init**.py; tests/unit/motion_matching/test_club_body_candidates.py; docs/plans/club_only_matching/evidence/club_body_candidates.json; docs/plans/club_only_matching/TURNOVER.md; docs/development/monolith_refactor_register.md; docs/shared_tools/divergence_inventory.v1.json
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 — merged via #10681 onto main as prerequisite for CO-06
- **Summary:** Explicit reduced→body topology maps (no pelvis teleport / unlimited root / pasted club animation), local grip-Jacobian null-space proposals with closure reprojection, and a roster × trial candidate matrix with separated observation-fit, plausibility, contact/effort, and runtime lanes; missing-runtime cells stay unqualified with precise blockers.
- **Next step:** Continue on #10610 (CO-06).
- **Evidence:** docs/plans/club_only_matching/evidence/club_body_candidates.json; tests/unit/motion_matching/test_club_body_candidates.py.

### DL-#10610 · CO-06 Recover Feasible Controls and Independently Replay Candidates

- **State:** shipped
- **Owner:** local
- **Issue:** #10610 (epic #10602)
- **Branch:** feat/10610-co06-controls-replay
- **PR:** #10687 (merged)
- **Paths:** src/shared/python/motion_matching/club_only/control_replay.py; src/shared/python/motion_matching/club_only/native_g1_gates.py; src/shared/python/motion_matching/club_only/replay_package.py; src/shared/python/motion_matching/club_only/**init**.py; tests/unit/motion_matching/test_club_control_replay.py; docs/plans/club_only_matching/evidence/club_control_replay.json; docs/plans/club_only_matching/TURNOVER.md; docs/development/HANDOFF.md; docs/development/monolith_refactor_register.md; docs/shared_tools/divergence_inventory.v1.json; SPEC.md
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 at f191dd09d — squash-merged to main via PR #10687; native G1 remains blocked on software-contract fixtures only.
- **Summary:** Recover minimum-effort feasible controls from CO-04/CO-05 candidates, separate net torque / actuated / passive / reactions / root slack, independently open-loop replay from q0/v0 without measured-state resets, and retain named native G1 blockers on software-contract fixtures only.
- **Next step:** Dispatch CO-07+ per club-only epic dependency order; do not invent native G1 pass.
- **Evidence:** docs/plans/club_only_matching/evidence/club_control_replay.json; tests/unit/motion_matching/test_club_control_replay.py.

### DL-#10611 · CO-07 Optimize Fast Matching and Expose Candidate Diversity

- **State:** shipped
- **Owner:** local
- **Issue:** #10611 (epic #10602)
- **Branch:** feat/10611-co07-fast-matching
- **PR:** #10700 (merged)
- **Paths:** src/shared/python/motion_matching/club_only/fast_matching.py; src/shared/python/motion_matching/club_only/**init**.py; tests/unit/motion_matching/test_club_fast_matching.py; docs/plans/club_only_matching/evidence/club_fast_matching.json; docs/shared_tools/divergence_inventory.v1.json; docs/development/HANDOFF.md; docs/development/DEVELOPMENT_LOG.md; AGENT_HANDOFF.md; SPEC.md
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 at f9ece7f6a — squash-merged to main via PR #10700; native_g1_pass false on software-contract fixtures.
- **Summary:** Adds bounded fast club-only matching orchestration with immutable cache keys, resumable checkpoints, cold/retrieval/reduced-to-full starts, feasibility-first pruning and Pareto diversity, optional neural proposal slot without weights, quality-vs-time curves, and profiling that includes verification time. Public knobs collapse onto FastMatchOptions for architecture parameter budgets.
- **Next step:** Dispatch CO-08 (#10612) per club-only epic dependency order; do not invent native G1 pass.
- **Evidence:** docs/plans/club_only_matching/evidence/club_fast_matching.json; tests/unit/motion_matching/test_club_fast_matching.py.

### DL-#10612 · CO-08 Qualify the Club-Only Matrix and Plausibility Tradeoffs

- **State:** shipped
- **Owner:** local
- **Issue:** #10612 (epic #10602)
- **Branch:** feat/issue-10612-co08-matrix
- **PR:** #10703 (merged)
- **Paths:** src/shared/python/motion_matching/club_only/matrix_qualification.py; src/shared/python/motion_matching/club_only/profiles.py; src/shared/python/motion_matching/club_only/body_candidates.py; src/shared/python/motion_matching/fit_metrics.py; src/shared/python/motion_matching/acceptance.py; src/shared/python/motion_matching/plot_fit_quality_card.py; src/shared/python/motion_matching/club_only/**init**.py; tests/unit/motion_matching/test_club_matrix_qualification.py; tests/unit/motion_matching/test_club_plausibility_acceptance.py; docs/plans/club_only_matching/evidence/club_matrix_qualification.json; docs/plans/club_only_matching/TURNOVER.md; docs/development/HANDOFF.md; docs/development/DEVELOPMENT_LOG.md; docs/shared_tools/divergence_inventory.v1.json; AGENT_HANDOFF.md; SPEC.md
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 at 17a0ee033 — squash-merged to main via PR #10703; native_g1_pass false on software-contract fixtures.
- **Summary:** Independent matrix qualification over native observation times for four workbook trials × #10585 roster with frozen CO-02 gates, published failures, withheld-body experiment semantics, common-observable comparison across complexities, and fail-closed checks for tamper/leakage/phase/orientation/reset/geometry/false-native claims. Shared roster/matrix scope helpers live in club_only/profiles.py.
- **Next step:** N/A — shipped; continue CO-09 (#10613).
- **Evidence:** docs/plans/club_only_matching/evidence/club_matrix_qualification.json; tests/unit/motion_matching/test_club_matrix_qualification.py.

### DL-#10613 · CO-09 Integrate Club-Only Matching Into Existing UI and Results

- **State:** shipped
- **Owner:** local
- **Issue:** #10613 (epic #10602)
- **Branch:** feat/co09-club-only-ui-10613
- **PR:** #10711
- **Paths:** src/shared/python/motion_matching/club_only/ui_integration.py; src/shared/python/motion_matching/club_only/**init**.py; src/shared/python/workspace/results_browser.py; src/tools/motion_matching/pipeline.py; src/tools/motion_matching/gui.py; src/tools/tour_matching_viewer/core.py; src/tools/tour_matching_viewer/**init**.py; src/config/feature_parity.json; tests/unit/motion_matching/test_club_ui_integration.py; tests/unit/tools/test_tour_matching_viewer_core.py; tests/unit/workspace/test_results_browser.py; docs/plans/club_only_matching/evidence/club_ui_integration.json; docs/plans/club_only_matching/TURNOVER.md; docs/development/HANDOFF.md; docs/development/DEVELOPMENT_LOG.md; SPEC.md
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 at SELF — rematched onto post-CO-10 main; LOD fix via `ClubOnlyUiSession.preset_name()`; review-gap harden GREEN; native_g1_pass false
- **Summary:** Baseline UI integration shipped via #10711. Follow-up #10717 hardens workbook trial load, receipt-hashed ledger append/dedup, default JSON ResultsBrowser index, cancel/resume off-thread, and tour_matching_viewer observed/inferred compare without parallel frameworks.
- **Next step:** Confirm CI green on PR #10717 and squash-merge; then check NM-07 #10622 claim (do not steal).
- **Evidence:** docs/plans/club_only_matching/evidence/club_ui_integration.json; tests/unit/motion_matching/test_club_ui_integration.py.

### DL-#10614 · CO-10 Publish Reproduction Guide and Final Club-Only Turnover

- **State:** shipped
- **Owner:** local
- **Issue:** #10614 (epic #10602)
- **Branch:** feat/10614-co10-reproduction-turnover
- **PR:** #10718 (baseline) + #10720 (survivor; both merged); #10719 closed duplicate
- **Paths:** src/shared/python/motion_matching/club_only/reproduction.py; src/shared/python/motion_matching/club_only/**init**.py; tests/unit/motion_matching/test_club_reproduction_turnover.py; docs/plans/club_only_matching/REPRODUCTION_GUIDE.md; docs/plans/club_only_matching/evidence/club_reproduction_turnover.json; docs/plans/club_only_matching/TURNOVER.md; docs/development/matched_swing_program/README.md; docs/development/HANDOFF.md; docs/development/DEVELOPMENT_LOG.md; AGENT_HANDOFF.md; SPEC.md
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 — merged via PR #10720 (`abd35e66b`); baseline #10718; duplicate #10719 stays closed
- **Summary:** Baseline operator/reproduction turnover landed via #10718. Survivor #10720 rematched runnable `fast_preview_match`/`asset_paths` saved-job commands, architecture-budget helper split, and succession docs. Epic #10602 stays open; software-contract tests are not native evidence.
- **Next step:** N/A — shipped; finish CO-09 review-gap PR #10717; schedule desk-native Fit/G1 for unresolved matrix cells.
- **Evidence:** docs/plans/club_only_matching/evidence/club_reproduction_turnover.json; tests/unit/motion_matching/test_club_reproduction_turnover.py.

### DL-#10431 · Freeze Fast-Matching Evidence, Schemas and Negative Acceptance Fixtures

- **State:** shipped
- **Owner:** local
- **Issue:** #10431 (PF-01)
- **Branch:** feat/issue-10431-pf01-fast-matching-evidence-schemas-fixtures
- **PR:** #10495
- **Paths:** src/shared/python/motion_matching/acceptance.py; src/shared/python/motion_matching/candidate.py; src/shared/python/motion_matching/candidate_convert.py; src/shared/python/motion_matching/candidate_io.py; src/shared/python/motion_matching/contact_force_allocator.py; src/shared/python/motion_matching/swing_evaluator.py; tests/unit/motion_matching/test_acceptance.py; tests/unit/motion_matching/test_candidate.py; tests/unit/motion_matching/test_contact_force_allocator.py; tests/unit/motion_matching/test_swing_evaluator.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (46 unit tests pass across test_candidate, test_contact_force_allocator, test_swing_evaluator, test_acceptance; ruff clean, black clean).
- **Summary:** Preserved existing rejected driver/iron fast-matching receipts and candidate artifacts. Extended MatchedSwingCandidate schema, auxiliary blocks (root_forces, contact_modes, grip_wrench), metadata (solver_status, handedness, name_maps), and NPZ converter. Truthfully renamed AllocationObjective.MINIMUM_TRAIL_ARM with backwards-compatible 'trail_zero' parsing, added HARD_ZERO_TRAIL mode enforcing exact zero trail arm torques, enforced actuator bounds clipping post-solve, and evaluated Coulomb friction cone ratios. Updated SwingEvaluator to avoid fabricating impact phases without declared t_events, audit closure translation and rotation separately, and return NaN RMSE for empty marker populations. Added 7 negative acceptance fixtures in test_acceptance covering friction cone violation, torque bound overwrite, missing root histories, coordinate mismatch (44 vs 41), missing club coverage, truncated horizon, and synthetic engine false qualification.
- **Next step:** Landed in main via PR #10495.
- **Evidence:** tests/unit/motion_matching/test_candidate.py; tests/unit/motion_matching/test_contact_force_allocator.py; tests/unit/motion_matching/test_swing_evaluator.py; tests/unit/motion_matching/test_acceptance.py.

### DL-#10350 · Unified Cross-Engine Parity Report

- **State:** shipped
- **Owner:** claude
- **Issue:** #10350 (MS-70, epic #10363)
- **Branch:** feat/10350-unified-parity-report
- **PR:** #10582
- **Paths:** src/shared/python/motion_matching/parity_schema.py; src/shared/python/motion_matching/parity_report.py; src/shared/python/motion_matching/cross_engine_replay.py; src/shared/python/motion_matching/leaderboard.py; src/engines/CROSS_ENGINE_PARITY_SPEC.md; evidence/matched/driver_g1/parity_report.json; evidence/matched/driver_g1/parity_report.md; tests/unit/motion_matching/test_parity_report.py
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 at HEAD (6 unit tests pass in tests/unit/motion_matching/test_parity_report.py; 39 leaderboard tests pass in tests/unit/motion_matching/test_leaderboard.py; evidence JSON and Markdown generated; cross-engine parity spec synchronized; merged into main).
- **Summary:** Implemented `UnifiedParityReport` and `build_parity_report` running candidates across all available physics engines (MuJoCo, Drake, Pinocchio, OpenSim, Simscape, MyoSuite). Evaluates pointwise trajectory error, pointwise joint torque comparison with 1.0 N·m absolute floor, total mechanical work in Joules, contact forces, and wall-clock execution time. Categorizes cross-engine comparisons into explicit classes: same-model numerical, native-model observable, and experimental accuracy. Gracefully stamps native engines lacking local platform SDKs as unavailable with reasons without falsifying synthetic data. Extends `cross_engine_replay.py` and `leaderboard.py` to ingest unified parity reports. Synchronized Section 3 of `src/engines/CROSS_ENGINE_PARITY_SPEC.md`.
- **Next step:** Shipped in PR #10582.
- **Evidence:** evidence/matched/driver_g1/parity_report.json; evidence/matched/driver_g1/parity_report.md; tests/unit/motion_matching/test_parity_report.py.

### DL-#10587 · TB-02: Define Versioned Baseline Packages, Fit Metrics, and Qualification Profiles

- **State:** shipped
- **Owner:** local
- **Issue:** #10587 (parent #10584, program #10363)
- **Branch:** feat/tb02-baseline-packages-10587
- **PR:** #10630
- **Paths:** src/shared/python/tour_baselines/baseline_package.py; src/shared/python/tour_baselines/fit_metrics.py; src/shared/python/tour_baselines/qualification_profiles.py; src/shared/python/tour_baselines/**init**.py; src/shared/python/motion_matching/tour_baselines.py; src/shared/python/motion_matching/acceptance.py; tests/unit/tour_baselines/test_fit_metrics.py; tests/unit/tour_baselines/test_baseline_packages.py; tests/unit/tour_baselines/test_qualification_profiles.py; docs/plans/tour_baselines/baseline_packages.md; docs/plans/tour_baselines/qualification_profiles.md; docs/plans/tour_baselines/README.md; docs/plans/tour_baselines/evidence/synthetic_valid_baseline_package.json; docs/plans/tour_baselines/evidence/synthetic_invalid_baseline_package.json; SPEC.md; docs/development/DEVELOPMENT_LOG.md; AGENT_HANDOFF.md
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 at HEAD (50/50 unit tests pass in tests/unit/tour_baselines/ and 45/45 motion-matching tests pass; ruff check clean; ruff format clean; mypy 0 issues in 14 source files; check_architecture_budget passes with 0 violations; check_file_size_budget passes; check_lod clean with 0 violations).
- **Summary:** Established the canonical Baseline Package contract (`tour-baseline-package/1.0.0`) extending portable packaging (#10379, #10334). Implemented `BaselineIdentity` linking capture target, model topology, backend pin, fit mode, horizon, frame/plane conventions, measurement map version, fixed geometry/inertia hashes, q0/v0 hashes, controls hash, solver config, seed, budgets, ancestry, and environment hashes. Formulated `StatusBundle` separating solver convergence, kinematic accuracy, dynamic feasibility, scientific qualification, and product promotion into orthogonal statuses, enforcing that missing native replay forbids scientific qualification. Formulated physical 3D Euclidean marker RMSE, p95, max, per-marker, per-phase, endpoint, and impact errors, distinct from optimizer loss, bound to cryptographic landmark signatures. Froze numeric qualification profiles for authoritative full-body G1/G2/G3 and reduced educational models (planar driven pendulum, upper-body golfer, triple pendulum) with documented attainable-geometry rationale without relaxing full-body thresholds. Provided clean-machine export/import with array checksum validation.
- **Next step:** Shipped in PR #10630. Proceeded to TB-03 (#10588).
- **Evidence:** docs/plans/tour_baselines/evidence/synthetic_valid_baseline_package.json; docs/plans/tour_baselines/evidence/synthetic_invalid_baseline_package.json; docs/plans/tour_baselines/baseline_packages.md; docs/plans/tour_baselines/qualification_profiles.md; tests/unit/tour_baselines/test_fit_metrics.py; tests/unit/tour_baselines/test_baseline_packages.py; tests/unit/tour_baselines/test_qualification_profiles.py.

### DL-#10586 · TB-01: Audit Tour Targets, Marker Semantics, Events, and Provenance

- **State:** shipped
- **Owner:** local
- **Issue:** #10586 (parent #10584, program #10363)
- **Branch:** feat/tb01-tour-targets-audit-10586
- **PR:** #10601
- **Paths:** src/shared/python/motion_matching/tour_capture_contract.py; src/shared/python/tour_baselines/audit.py; src/shared/python/tour_baselines/measurement_map.py; src/shared/python/tour_baselines/events.py; src/shared/python/tour_baselines/provenance.py; src/shared/python/tour_baselines/canonical_targets.py; src/shared/python/tour_baselines/**init**.py; tests/unit/tour_baselines/test_measurement_map.py; tests/unit/tour_baselines/test_tour_events.py; tests/unit/tour_baselines/test_target_audit.py; tests/unit/tour_baselines/test_canonical_targets.py; docs/plans/tour_baselines/target_audit.md; docs/plans/tour_baselines/evidence/driver_target_audit_receipt.json; docs/plans/tour_baselines/evidence/iron_target_audit_receipt.json
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 at HEAD (35/35 unit tests pass in tests/unit/tour_baselines/ and tests/opensim/test_tour_capture_contract.py; ruff check clean; ruff format clean; black clean; mypy 0 issues in 18 source files; check_architecture_budget passes; check_file_size_budget passes; divergence inventory synchronized; merged into main via PR #10601).
- **Summary:** Implemented target audit, content-based SHA-256 verification (never path name alone), versioned measurement map (tour-measurement-map/1.0.0) distinguishing surface markers, inferred joint centers, and rigid cluster centroids while explicitly marking calibrated clubface orientation and contact points as UNAVAILABLE in raw C3D. Defined native-clock swing intervals (360.0 Hz driver vs 359.0 Hz iron) with trajectory-inferred events explicitly labeled as inferred. Recorded OpticalCapture optical provenance, separating shared player anatomy from capture-specific club geometry. Emitted reproducible receipts in docs/plans/tour_baselines/evidence/ and documented in target_audit.md.
- **Next step:** Shipped in PR #10601.
- **Evidence:** docs/plans/tour_baselines/evidence/driver_target_audit_receipt.json; docs/plans/tour_baselines/evidence/iron_target_audit_receipt.json; tests/unit/tour_baselines/test_target_audit.py; tests/unit/tour_baselines/test_measurement_map.py; tests/unit/tour_baselines/test_tour_events.py; tests/unit/tour_baselines/test_canonical_targets.py.

### DL-#10585 · TB-00: Freeze Model Identities, Ownership, and the Two-Capture Coverage Matrix

- **State:** shipped
- **Owner:** local
- **Issue:** #10585 (parent #10584, program #10363)
- **Branch:** feat/tb00-model-identities-10585
- **PR:** #10598
- **Paths:** src/shared/python/tour_baselines/models.py; src/shared/python/tour_baselines/registry.py; src/shared/python/tour_baselines/coverage.py; src/shared/python/tour_baselines/reconciliation.py; src/shared/python/tour_baselines/**init**.py; src/shared/python/motion_matching/tour_baselines.py; tests/unit/tour_baselines/test_model_identities.py; tests/unit/tour_baselines/test_coverage_matrix.py; tests/unit/tour_baselines/test_reconciliation.py; docs/plans/tour_baselines/README.md; docs/plans/tour_baselines/model_identities.md; docs/plans/tour_baselines/coverage_matrix.md; docs/plans/tour_baselines/reconciliation.md; docs/development/matched_swing_program/README.md
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 at HEAD (13/13 unit tests pass in tests/unit/tour_baselines/; ruff check clean; ruff format clean; mypy 0 issues in 5 source files; document title case 0 violations across 4 docs; file size budget clean; merged into main).
- **Summary:** Established the canonical Tour Baselines domain model and registry separating kinematic reconstruction models from torque-driven pendulums, upper-body golfer with closed kinematic loop, and flagship full-body engines. Evaluated upper-body golfer constraint Jacobian SVD proving rank 3 (5 independent DOFs). Implemented two-capture coverage matrix across Driver and 7-Iron, non-golf tool exclusions, and closed-issue reconciliation (#9914, #9921, #10003). Verified Tools pin at a9ed0e7c5c6905b1164082659051d6381068052d and published comprehensive documentation under docs/plans/tour_baselines/.
- **Next step:** Shipped in PR #10598.
- **Evidence:** tests/unit/tour_baselines/test_model_identities.py; tests/unit/tour_baselines/test_coverage_matrix.py; tests/unit/tour_baselines/test_reconciliation.py; docs/plans/tour_baselines/README.md.

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
- **Last verified:** 2026-09-19 at HEAD (22 unit tests pass in test_mujoco_candidate_replay.py under native MuJoCo 3.13.0 on <host>; full 307-frame forward replay generated; same-state FK error 2.51e-15 m; fail-closed G1 rejection recorded).
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
- **Last verified:** 2026-09-19 at HEAD (Full 654-frame OpenSim IK executed in 50s on <host>; receipt.json validated; 6/6 native tests pass; 7 unit tests pass; architecture budget clean; ruff/black clean).
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
