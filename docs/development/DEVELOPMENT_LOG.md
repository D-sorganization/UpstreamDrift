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

### DL-#11792 — Whole-Body Motion Matching Readiness

- **State:** in_progress
- **Owner:** codex
- **Issue:** #11792 (parent #11784; implementation children #12342, #12343 and #12353)
- **PR:** #12355 (draft passive-readiness work; #12353 no-probe publication pending; related readiness work in #12344)
- **Branch:** `feat/native-passive-observation-receipt-12342`; `feat/native-no-probe-12353-20261012`
- **Paths:** `src/engines/physics_engines/opensim/python/tour_matching/native_moco_preparation.py`, `src/engines/physics_engines/opensim/python/tour_matching/native_moco_runner.py`, `src/engines/physics_engines/opensim/python/tour_matching/native_passive_receipt.py`, `tests/opensim/test_native_passive_receipt.py`, `tests/unit/engines/opensim/test_native_path_optimizer_domains.py`, `tests/unit/engines/opensim/test_native_moco_probe_mode.py`, `manuals/upstreamdrift/chapters/24-native-path-work.qmd`, `manuals/upstreamdrift/chapters/33-native-passive-readiness.qmd`, `manuals/upstreamdrift/chapters/56-native-diagnostic-execution-contract.qmd`, `docs/development/feedback_controls/F07_PASSIVE_READINESS_TURNOVER.md`, `docs/development/feedback_controls/F07_NATIVE_NO_PROBE_PREPARATION_TURNOVER.md`
- **Started:** 2026-10-11
- **Last verified:** 2026-10-12 at `f170775f04ad0511217b3422758ede0266c0c61e`; combined focused SDK-free tests passed (58 passed, 102 skipped because OpenSim is unavailable in the QA environment). The pinned Development Log validator passed with the existing portfolio warning (21 in-progress entries exceed the golf cap of 8). No SDK or native execution was performed.
- **Summary:** Software work adds source-bound optimizer-domain checks and persists passive-readiness observations without changing their scientific digest or qualification meaning. The no-probe preparation option skips forward integration only when explicitly requested for prepare-only; the default remains probe-enabled. The current diagnostic requests forward finite differences; no native solve, optimum, learned replay, or physical qualification is established. All 17 registered model/variant rows remain unqualified.
- **Next step:** Complete one full pre-PR validation for the combined changes, then review a source-bound no-probe preparation result; solver readback and a full-clock replay remain required before any qualification claim.

### DL-#11719 · GCV-13 Slice 3: Decorative Address Ball in the MuJoCo, OpenSim and MyoSuite Native Export

- **State:** in_review
- **Owner:** claude
- **Issue:** #11719 (epic #11706)
- **Branch:** claude/gcv-13c-native-ball-backends
- **PR:** not created (stacked on #12114, which is stacked on #11992)
- **Paths:** src/tools/native_viewer_export/backends/\_ball.py, src/tools/native_viewer_export/backends/\_worker_job.py, src/tools/native_viewer_export/backends/\_subprocess.py, src/tools/native_viewer_export/backends/mujoco_native.py, src/tools/native_viewer_export/backends/mujoco_worker.py, src/tools/native_viewer_export/backends/myosuite_arena.py, src/tools/native_viewer_export/backends/myosuite_worker.py, src/tools/native_viewer_export/backends/opensim_simbody.py, src/tools/native_viewer_export/backends/opensim_worker.py, src/tools/native_viewer_export/README.md, src/config/feature_parity.json, tests/unit/tools/native_viewer_export/test_ball_native_backends.py
- **Started:** 2026-10-10
- **Last verified:** 2026-10-10 (`f621f12f`)
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

### DL-#11614 · Same-Input Parity LaTeX Research Reference (P-1 Pointwise Result)

- **State:** in_progress
- **Owner:** unassigned
- **Issue:** #11614
- **Branch:** docs/sip-9-parity-reference-11614
- **PR:** #11616, #11625
- **Paths:** see #11616
- **Started:** 2026-10-07
- **Last verified:** 2026-10-07 (`cf3e475f`; collated from changes/11614-document-full-swing-five-engine-same-inp.md)
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

### DL-#9507 · Cone-Mode Sparse Checkout in Sidekick Wheel Smoke Job

- **State:** in_review
- **Owner:** claude
- **Issue:** #9507
- **Branch:** `claude/issue-9507`
- **Paths:** `.github/workflows/package-standalone-sidekick.yml`, `tests/unit/packaging/test_standalone_sidekick_workflows.py`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-05 (`43d63ce9`; collated from changes/9507-fail-fast-with-an-explicit-incomplete-ch.md)
- **Summary:** `smoke-test-wheel` used non-cone sparse checkout, which leaves `core.sparseCheckout=true` in `.git/config` and poisons later jobs on reused self-hosted runners. Switched to cone mode on `tests/fixtures` and added a guard test over all workflows.
- **Next step:** Merge the PR and confirm no runner reports tracked files missing after a Sidekick packaging run.
- **PR:** #11522

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

- **State:** in_progress
- **Owner:** codex
- **Issue:** #11235; parent #11232
- **PR:** [#11240](https://github.com/D-sorganization/UpstreamDrift/pull/11240)
- **Branch:** `feat/necromatcher-native-fit-11235`
- **Paths:** `src/shared/python/motion_matching/historical_fit/**`, `src/shared/python/workspace/necromatcher_*`, `src/engines/physics_engines/mujoco/python/**`
- **Started:** 2026-10-01
- **Last verified:** 2026-10-02 at b92732e72579bfb72ba77d16f671f5f657a1b37c — 108 focused cases passed; subsequent native test refactor passed 34 cases; independent v7 audit checked 419 Tiger and 1499 Hogan poses. These receipts precede the b50e8fd91e branch merge.
- **Summary:** Source-bound native fits retain canonical Hermite coefficients and optional analytic interior constraints. Both v7 candidates remain rejected: joint-limit violations and grip/contact errors require hard feasibility constraints. [Turnover](necromatcher-turnover.md).
- **Next step:** Implement canonical whole-Hermite coordinate bounds with red-first overshoot tests.

### DL-#11246 · Necromatcher Source Video Overlays

- **State:** in_review
- **Owner:** codex
- **Issue:** #11246; parent #11232
- **PR:** [#11240](https://github.com/D-sorganization/UpstreamDrift/pull/11240)
- **Branch:** `feat/necromatcher-native-fit-11235`
- **Paths:** `src/shared/python/workspace/necromatcher_video*`, `src/tools/necromatcher/**`, `ui/src/components/necromatcher/**`
- **Started:** 2026-10-01
- **Last verified:** 2026-10-02 at b92732e72579bfb72ba77d16f671f5f657a1b37c — all 210 Tiger and 750 Hogan v7 output frames decoded with matching hashes; prior live web v6 jobs and downloaded packages verified.
- **Summary:** Shared owned jobs export source-clock native joint-tree wireframes, PNGs and verified ZIP packages through native and web controls. Export success remains separate from scientific acceptance. [Turnover](necromatcher-turnover.md).
- **Next step:** Revalidate export-job recall and download guards against the merged branch.

### DL-#11247 · Necromatcher Reproducible Methods Report

- **State:** in_progress
- **Owner:** codex
- **Issue:** #11247; parent #11232
- **PR:** [#11240](https://github.com/D-sorganization/UpstreamDrift/pull/11240)
- **Branch:** `feat/necromatcher-native-fit-11235`
- **Paths:** `docs/development/necromatcher-methods.tex`, `docs/development/historical_capture/*summary.json`, `docs/development/necromatcher-turnover.md`
- **Started:** 2026-10-01
- **Last verified:** 2026-10-02 at b92732e72579bfb72ba77d16f671f5f657a1b37c — 19-page PDF compiled with existing MiKTeX installer disabled; all pages visually reviewed and 60 Desktop artifact hashes verified.
- **Summary:** Polished methods report records equations, provenance, assumptions, actual interim failures and reproducibility procedures. Built-in compiler infrastructure remains unavailable, so that issue criterion remains open. [Turnover](necromatcher-turnover.md).
- **Next step:** Reconcile the built-in compilation criterion with the verified fallback receipt.

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

### DL-#11083 — Motion Matching Board Review

- **State:** shipped
- **Owner:** codex
- **Issue:** owner-requested documentation review; tracked by PR #11083
- **PR:** #11083
- **Branch:** `docs/motion-matching-board-review`
- **Paths:** `docs/development/2026-09-28-motion-matching-board-review.md`
- **Started:** 2026-09-28
- **Shipped:** 2026-09-29
- **Last verified:** 2026-10-12 at 64962e8d6b6263d5fb32bb1444712b28a9df4433 (remote merge verification only; no tests or native runs)
- **Summary:** Evidence-backed engine/model review, recent GS3DX improvements, 18 issue proposals, anatomical Home-budget options and historical-video readiness; documentation review only, with no implementation or qualification claim.
- **Next step:** The documentary review is shipped; resulting implementation proposals remain separately tracked.

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
- **PR:** draft PR from `agy/ud-9549-ruling-guard`
- **Branch:** `agy/ud-9549-ruling-guard`
- **Paths:** `tests/shared_contracts/test_impact_interval_provider.py`, `docs/development/DEVELOPMENT_LOG.md`
- **Started:** 2026-09-25
- **Last verified:** 2026-09-25 (`ff9fbee62`) — verified 11 contract tests pass in tests/shared_contracts/test_impact_interval_provider.py guarding owner interval-tab ruling
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

### DL-#10960 · Fabricated-Evidence Audit: Fail-Closed Remediation of Literal Receipts and Self-Scored Fits

- **State:** shipped
- **Owner:** claude (agy executors on DeskComputer and OG Laptop)
- **Issue:** #10960
- **PR:** #11017
- **Branch:** `claude/ud-consolidated2-20260926`
- **Paths:** `src/shared/python/motion_matching/provider.py`, `src/engines/physics_engines/*/python/motion_matching/provider.py`, `src/shared/python/neural_motion/surrogates/`, `src/shared/python/neural_motion/matrix/`, `src/shared/python/neural_motion/turnover/`, `src/shared/python/neural_motion/benchmark/runner.py`, `src/shared/python/neural_motion/inference/orchestration.py`, `src/shared/python/motion_matching/contact_identification.py`, `src/engines/physics_engines/opensim/python/tour_matching/full_swing_tracking.py`, `docs/plans/neural_motion_matching/evidence/`, `docs/development/full_body_models/evidence/contact_id/receipt.json`, `src/shared/python/motion_matching/parity_report.py`, `src/shared/python/motion_matching/parity_schema.py`, `src/shared/python/motion_matching/cross_engine_replay.py`, `src/shared/python/motion_matching/full_body_forward_dynamics.py`, `src/shared/python/motion_matching/candidate_session.py`, `docs/development/full_body_models/evidence/fb5_matching/`, `docs/development/full_body_models/evidence/fb6_parity/`, `evidence/matched/driver_g1/parity_report.json`, `docs/development/full_body_models/evidence/_gates.py`, `docs/development/full_body_models/evidence/fb3_drake/`, `docs/development/full_body_models/evidence/fb4_calibration/`, `evidence/matched/driver_g1_pinocchio/`, `src/tools/matched_swing_browser/model.py`, `src/engines/physics_engines/opensim/python/tour_matching/document_ik.py`, `docs/development/full_body_models/evidence/ground_support/anthro_driver_opensim/receipt.json`, `src/shared/python/motion_matching/matching_strategy.py`, `src/shared/python/motion_matching/ledger.py`, `reports/matched_swing_ledger.json`, `tests/fixtures/motion_matching_strategy.py`, `src/shared/python/motion_matching/club_only/`, `src/tools/motion_matching/gui.py`, `src/shared/python/motion_matching/acceptance.py`, `src/shared/python/motion_matching/kinematic_smoothing.py`, `src/shared/python/motion_matching/loaders/body_json.py`, `src/engines/physics_engines/drake/python/motion_matching/simulate.py`, `src/engines/physics_engines/pendulum/python/motion_matching/club_pendulum_match.py`, `src/shared/python/tour_baselines/calibration.py`, `src/shared/python/pose_interchange/pose_io.py`, `docs/plans/club_only_matching/evidence/club_body_candidates.json`
- **Started:** 2026-09-25
- **Shipped:** 2026-09-27
- **Last verified:** 2026-09-26 — consolidated branch on `71610a4ff`: motion_matching, neural_motion, opensim, tools/motion_matching, config and every changed test file: 3034 passed, 21 failed, all baseline or host-only (16 known motion_matching failures, OpenSim binding absent, Rajagopal asset absent, local 10k corpus present); ledger regenerated and fresh.
- **Summary:** Fail-closed software fixes span motion matching, neural-motion receipts, parity and rollout evidence, native-lane reporting, and UI seed guards. Missing data remains unmeasured or unverified; no synthetic evidence or model qualification is emitted.
- **Next step:** The fail-closed receipt correction is shipped. The checked receipts retain empty qualified/promoted sets and unmeasured status; no native or physical qualification is claimed.

### DL-#9415 · Repository Root Allowlist Check in Docs Governance

- **State:** in_review
- **Owner:** claude
- **Issue:** #9415
- **PR:** draft PR from `agy/ud-9415-root-allowlist`
- **Branch:** `agy/ud-9415-root-allowlist`
- **Paths:** `scripts/check_docs_governance.py`, `scripts/config/root_allowlist.json`, `docs/governance/DOCS_GOVERNANCE.md`, `tests/scripts/test_doc_governance_checks.py`, `tests/unit/scripts/test_check_docs_governance_root_allowlist.py`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`
- **Started:** 2026-09-26
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Committed root entries (`git ls-tree --name-only HEAD`) must match `scripts/config/root_allowlist.json`; unlisted and stale entries both fail, and a failed git call or malformed config fails closed. Closes the last software checkbox of #9415; the history rewrite, Jules workflows and SPEC cap remain owner decisions.
- **Next step:** CI green, review, merge.

### DL-#10783 · Enforce the Deferred Validation Catalog

- **State:** in_review
- **Owner:** codex
- **Issue:** #10783; full rollout Repository_Management#1687
- **PR:** #10784
- **Branch:** `chore/10783-deferred-catalog-guard`
- **Paths:** `shared_scripts/`, `tests/unit/repo_hygiene/test_deferred_catalog_hook.py`, `.pre-commit-config.yaml`, `docs/development/`, `AGENT_HANDOFF.md`, `README.md`, `SPEC.md`
- **Started:** 2026-09-23
- **Last verified:** 2026-09-23 (`df571686df`; five RED then GREEN controls; twenty combined tests; root Ruff/8162-file format, strict new-test typing, size/manual gates and actual commit/pre-push hooks pass; initialized exact existing Tools gitlink after retained setup failures)
- **Summary:** Install the exact central validator and always-run hook while preserving six published v1 plans, executable software and unavailable empirical acceptance.
- **Next step:** User-requested committed handoff: architecture correction passes hosted job 107433594917; add the appropriate unit-suite marker to the three new catalog test functions, validate and complete protected #10784/default-branch verification. Three canonical-function exceptions expire 2026-10-23.

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
- **PR:** #11516 (draft)
- **Paths:** src/api/routes/bunker_workbench.py; src/tools/bunker_shot_gui/input_ranges.py; src/tools/bunker_shot_gui/design.py; src/tools/bunker_shot_gui/widgets.py; src/tools/bunker_shot_gui/report.py; src/tools/bunker_shot_gui/render.py; src/config/feature_parity.json; tests/bunkershot3d/test_bunker_workbench_api.py
- **Started:** 2026-10-04
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** The headless workbench API enforces request ranges, handedness and source-stamped verdicts; predictive playability is refused or reported without ranking. UI and API share input-range authority.
- **Next step:** Review the draft PR and list the route in the U15 entry of `src/config/bunkershot3d_qualification.json`.

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
- **PR:** #10441
- **Paths:** src/config/impact_acceptance.json; tests/config/impact_acceptance/test_impact_acceptance_matrix.py; scripts/ci/verify_impact_explorer_bundle.py; tests/ci/test_verify_impact_explorer_bundle.py; tests/api/test_impact_explorer_mount.py; .github/workflows/ci-standard.yml; src/config/feature_parity.json; docs/development/impact_acceptance_matrix.md
- **Started:** 2026-09-18
- **Last verified:** 2026-09-18 at HEAD (SELF; matrix gate 26 passed; verifier + mount 14 passed; feature-parity/industrial-readiness gates 91 passed; real pinned bundle `62e8cdbf` verified locally: 76 assets, 5 JS, 404 on missing artifact)
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

### DL-#11556 · Ci(Security): Re-Vendor fork_pr_runner_guard.py and New fork_pr_guard_analysis.py From Repository_Management (RM#1996): Same-Repo if: Exemption and Sink-Based Head-Checkout Analysis.

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
