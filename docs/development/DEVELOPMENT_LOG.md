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

### DL-#14434 · RUN-4: Ud-Sim Reproduce Reruns a Stored Run From Its Model File With the Recorded Seed and Parameters, Compares Each Output Within an Absolute Tolerance and Reports Engine Version Mismatches; the Run Store Now Closes Copied Files and Reports Corrupt Records Cleanly

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14434
- **Branch:** brick/backlog-completion/14434-reproduce-run
- **PR:** #14520
- **Paths:** see #14520
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`4f5f12f7`; collated from changes/14434-reproduce-a-stored-run-with-a-drift-report.md)
- **Summary:** RUN-4: ud-sim reproduce reruns a stored run from its model file with the recorded seed and parameters, compares each output within an absolute tolerance and reports engine version mismatches; the run store now closes copied files and reports corrupt records cleanly
- **Next step:** Merge the PR, then RUN-5 #14435 (route the other engines through the shared emitter and add their reproducers).

### DL-#14433 · RUN-3: Content-Addressed Run Store (Run_Id = SHA-256 of the Canonical Manifest, Outputs Stored Once by Hash) With Ud-Sim Runs Add, List and Show; Store Root From --Store or $UD_RUN_STORE

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14433
- **Branch:** brick/backlog-completion/14433-run-store
- **PR:** #14517
- **Paths:** see #14517
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`f6703b27`; collated from changes/14433-content-addressed-run-store-and-lookup-cli.md)
- **Summary:** RUN-3: content-addressed run store (run_id = SHA-256 of the canonical manifest, outputs stored once by hash) with ud-sim runs add, list and show; store root from --store or $UD_RUN_STORE
- **Next step:** Merge the PR, then RUN-4 #14434 (reproduce a stored run by id with a drift report).

### DL-#14453 · Install Trivy From Checksum-Verified GitHub Releases in CI.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14453
- **Branch:** codex/14453-trivy-install
- **PR:** #14481
- **Paths:** see #14481
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`4082c388`; collated from changes/14453-install-trivy-from-checksum-verified-git.md)
- **Summary:** Install Trivy from checksum-verified GitHub releases in CI.
- **Next step:** Merge the pull request after required checks pass.

### DL-#14274 · Config/Secret Separation and Upgrade/Rollback Guarantees

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14274
- **Branch:** feat/14274-config-secrets
- **PR:** #14414
- **Paths:** see #14414
- **Started:** 2026-11-03
- **Last verified:** 2026-11-03 (`2e099ba2`; collated from changes/14274-config-secrets-and-upgrade-rollback.md)
- **Summary:** Config/secret separation and upgrade/rollback guarantees
- **Next step:** Merge the PR.

### DL-#14275 · \ Define and Implement Clean-Environment Install Smoke Test\

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14275
- **Branch:** feat/14275-install-smoke
- **PR:** #14402
- **Paths:** see #14402
- **Started:** 2026-11-03
- **Last verified:** 2026-11-03 (`8823f013`; collated from changes/14275-define-and-implement-clean-environment-i.md)
- **Summary:** \ Define and implement clean-environment install smoke test\
- **Next step:** \Merge the PR.\

### DL-#14273 · Versioned Releases With Verifiable Artifacts and SBOM

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14273
- **Branch:** feat/14273-release-sbom
- **PR:** #14413
- **Paths:** see #14413
- **Started:** 2026-11-03
- **Last verified:** 2026-11-03 (`90eadd46`; collated from changes/14273-versioned-releases-and-sbom.md)
- **Summary:** Versioned releases with verifiable artifacts and SBOM
- **Next step:** Merge the PR.

### DL-#14259 · Redesign Home With Pipeline and Truthful Engine Readiness

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14259
- **Branch:** feat/14259-redesign-home
- **PR:** #14410
- **Paths:** see #14410
- **Started:** 2026-11-03
- **Last verified:** 2026-11-03 (`956f7fc7`; collated from changes/14259-redesign-home.md)
- **Summary:** Redesign Home with pipeline and truthful engine readiness
- **Next step:** Merge PR via auto-squash

### DL-#14094 · Record E1 Review Evidence, Arithmetic Limits and Pinned-Source Corrections Without Closing the Epic

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14094
- **Branch:** chore/14094-e1-evidence-ledger
- **PR:** #14101
- **Paths:** see #14101
- **Started:** 2026-11-03
- **Last verified:** 2026-11-03 (`3e4c587a`; collated from changes/14094-record-e1-review-evidence-arithmetic-lim.md)
- **Summary:** Record E1 review evidence, arithmetic limits and pinned-source corrections without closing the epic
- **Next step:** Review the E1 documentation PR; retain E2-E8 evidence gates

### DL-#14337 · Document a Bounded Local-Only Grok Bot Simulation Caller Path

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14337
- **Branch:** codex/14337-local-grok-simulation-relay-plan
- **PR:** #14359
- **Paths:** docs/agents/grok-simulation-relay.md
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`b385be55`; collated from changes/14337-grok-simulation-relay.md)
- **Summary:** Document a bounded local-only Grok Bot simulation caller path
- **Next step:** Merge the PR.

### DL-#14272 · Record Evidence-Based Deployment Boundary

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14272
- **Branch:** codex/14272-deployment-boundary
- **PR:** #14353
- **Paths:** docs/deployment/14272-deployment-boundary.md
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`f3b7adf7`; collated from changes/14272-record-evidence-based-deployment-boundar.md)
- **Summary:** Record evidence-based deployment boundary
- **Next step:** Merge the PR.

### DL-#14266 · Define Mobile Workflows and Server-Side Runtime Boundary

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14266
- **Branch:** codex/14266-mobile-workflow-discovery
- **PR:** #14355
- **Paths:** docs/ux/14266-workflow-boundary.md
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`f6c5af05`; collated from changes/14266-define-mobile-workflows-and-server-side.md)
- **Summary:** Define mobile workflows and server-side runtime boundary
- **Next step:** Merge the PR.

### DL-#14290 · Gate JaxSim/Pinocchio Free-Body Parity on Strict Engine Outputs and a Rotated Inertial Oracle

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14290
- **Branch:** cursor/14290-parity-test-guard-2cee
- **PR:** #14343
- **Paths:** see #14343
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`0ee3ee11`; collated from changes/14290-rotated-free-body-oracle.md)
- **Summary:** Gate JaxSim/Pinocchio free-body parity on strict engine outputs and a rotated inertial oracle
- **Next step:** Merge after cross-engine equivalence lane executes three named cases with receipt artifact.

### DL-#14257 · Wire Dashboard Help and F1 to the Global HelpPanel With Focus-Safe Close and Mobile Layout

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14257
- **Branch:** codex/dashboard-help-panel-14257-20261102
- **PR:** #14331
- **Paths:** see #14331
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`7e37e215`; collated from changes/14257-dashboard-help-panel.md)
- **Summary:** Wire dashboard Help and F1 to the global HelpPanel with focus-safe close and mobile layout
- **Next step:** Finish the focused UI checks, open the linked PR, and verify protected CI

### DL-#14253 · Replace Undefined Tailwind Gray Shades With Supported Palette Utilities Across the Model Explorer and Workspace UI

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14253
- **Branch:** codex/fix-tailwind-gray-scale-14253-main
- **PR:** #14329
- **Paths:** see #14329
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`fde79256`; collated from changes/14253-replace-undefined-tailwind-gray-shades-w.md)
- **Summary:** Replace undefined Tailwind gray shades with supported palette utilities across the model explorer and workspace UI
- **Next step:** Capture matching desktop and phone after views, then obtain independent review before pushing.

### DL-#14250 · Use a Local CC0 HDR With Fallback Scene Lighting on All Three Affected Pages

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14250
- **Branch:** codex/terrain-hdr-offline-14250
- **PR:** #14325
- **Paths:** see #14325
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`5dcd3ea0`; collated from changes/14250-use-a-local-cc0-hdr-with-fallback-scene.md)
- **Summary:** Use a local CC0 HDR with fallback scene lighting on all three affected pages
- **Next step:** After the baseline PlateImportForm build issue is resolved, verify a clean Vite/Tauri asset output; then request independent review.

### DL-#14251 · Normalize Terrain Preset Loading and Preserve Concurrent API Errors.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14251
- **Branch:** codex/terrain-api-contract-14251
- **PR:** #14318
- **Paths:** see #14318
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`4ef03dc0`; collated from changes/14251-normalize-terrain-preset-loading-and-pre.md)
- **Summary:** Normalize terrain preset loading and preserve concurrent API errors.
- **Next step:** Complete independent review before pushing.

### DL-#14177 · Putting Optimizer and Misread-Cost MCP Tools

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14177
- **Branch:** claude/putting-tools-14177
- **PR:** #14203, #14220, #14247, #14245, #14233, #14242, #14236, #14238, #14237, #14218, #14230, #14231
- **Paths:** see #14203
- **Started:** 2026-11-01
- **Last verified:** 2026-11-02 (`ab77c162`; collated from changes/14177-sg-optimizer-putting-provider.md)
- **Summary:** Putting Optimizer and Misread-Cost MCP Tools
- **Next step:** Wire the putting tools into the PyQt6 app and API routes with a feature-parity entry

### DL-#14176 · Add Putting Reference Data and Validation Harness

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14176
- **Branch:** claude/putting-validation-14176
- **PR:** #14198, #14232
- **Paths:** see #14198
- **Started:** 2026-11-01
- **Last verified:** 2026-11-02 (`2d2257a7`; collated from changes/14176-project-steward-status-2026-11-01.md)
- **Summary:** Add Putting Reference Data and Validation Harness
- **Next step:** Calibrate presets once the batch roll and skill sampler merge

### DL-#14175 · Add Putting Analytics Variance Attribution and Error-Source Elimination

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14175
- **Branch:** claude/putting-variance-14175
- **PR:** #14196, #14401
- **Paths:** see #14196
- **Started:** 2026-11-01
- **Last verified:** 2026-11-03 (`edfad820`; collated from changes/14175-document-variance-attribution-study-inpu.md)
- **Summary:** Add Putting Analytics Variance Attribution and Error-Source Elimination
- **Next step:** Merge after 14188

### DL-#14174 · Add Putting Analytics Putt Library and Aggregate Report

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14174
- **Branch:** claude/putting-library-14174
- **PR:** #14197
- **Paths:** see #14197
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`bb30e42a`; collated from changes/14174-add-putting-analytics-putt-library-and-a.md)
- **Summary:** Add Putting Analytics Putt Library and Aggregate Report
- **Next step:** Merge after 14181

### DL-#14173 · Plan on the Perceived Green and Execute on the True Green

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14173
- **Branch:** claude/putting-perceived-green-14173
- **PR:** #14199, #14392
- **Paths:** see #14199
- **Started:** 2026-11-01
- **Last verified:** 2026-11-03 (`d6398046`; collated from changes/14173-report-the-expected-strokes-cost-of-gree.md)
- **Summary:** Plan on the Perceived Green and Execute on the True Green
- **Next step:** Add aspect misreads and per-class read-cost reporting over the putt library

### DL-#14171 · Add BatchRollModel PuttOutcomeModel Adapter

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14171
- **Branch:** claude/putting-roll-model-14171
- **PR:** #14195, #14194
- **Paths:** see #14195
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`2dd6c08d`; collated from changes/14171-add-vectorised-batch-roll-integrator.md)
- **Summary:** Add BatchRollModel PuttOutcomeModel Adapter
- **Next step:** Merge after 14181; then the BatchRollModel adapter PR

### DL-#14170 · Putting Analytics: Expected Putts and Aim/Pace Optimiser

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14170
- **Branch:** claude/project-thread-sfn9g4
- **PR:** #14181
- **Paths:** see #14181
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`2403e048`; collated from changes/14170-putting-analytics-expected-putts.md)
- **Summary:** Add putting_analytics: skill-conditioned expected-putts curve solved exactly, aim and pace optimiser for strokes-gained and must-make objectives, and strokes gained against both the player's own curve and a tour reference (epic #14168)
- **Next step:** Land #14169 skill profile and dispersion sampler, then wire a putting_dynamics outcome model (#14171)

### DL-#14169 · Putting Analytics: Skill Profile and Dispersion Sampler

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14169
- **Branch:** claude/putting-skill-sampler-14169
- **PR:** #14188
- **Paths:** see #14188
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`ab03503c`; collated from changes/14169-add-putting-analytics-skill-profile-stro.md)
- **Summary:** Add putting_analytics skill profile (stroke face, path, speed, strike and green-read dispersion) and a per-channel seeded Monte Carlo sampler whose single-channel ablations leave other draws identical (epic #14168)
- **Next step:** Wire the profile into a putting_dynamics PuttOutcomeModel (#14171)

### DL-#14146 · Docker Size Gates Measure Compressed Content Size via the Engine API Instead of .Size, Whose Meaning Changed Between Docker Engine 29.1.3 And 29.9.2

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14146
- **Branch:** brick/ci-toolchain/14146-docker-size-measure
- **PR:** #14153
- **Paths:** see #14153
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`06c2d8d5`; collated from changes/14146-docker-size-gates-measure-compressed-con.md)
- **Summary:** Docker size gates measure compressed content size via the Engine API instead of .Size, whose meaning changed between Docker Engine 29.1.3 and 29.9.2
- **Next step:** Merge after acceptance review

### DL-#13885 · Pure Two-Hand Grip-Margin Metric: Per-Pad Coulomb Margin, UNLOADED Floor, Slip Events, Grasp-Map Internal Wrench and GripMarginReport

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13885
- **Branch:** brick/issues-ud/13885-grip-margin-metric
- **PR:** #14148, #14152, #14347
- **Paths:** see #14148
- **Started:** 2026-10-31
- **Last verified:** 2026-11-02 (`be3b8bc5`; collated from changes/13885-grip-margin-swing-receipt.md)
- **Summary:** Pure two-hand grip-margin metric: per-pad Coulomb margin, UNLOADED floor, slip events, grasp-map internal wrench and GripMarginReport
- **Next step:** Merge after CI; closes scripted swing acceptance for #13885.

### DL-#12712 · NSX-83: Add the ClubheadSeries to RigidClubState Adapter and Constant-Twist Event Location Feeding the Shared Impact Outcome Map.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #12712
- **Branch:** claude/ud-12712-1
- **PR:** #14019
- **Paths:** see #14019
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`3a0aba29`; collated from changes/12712-nsx-83-add-the-clubheadseries-to-rigidcl.md)
- **Summary:** NSX-83: add the ClubheadSeries to RigidClubState adapter and constant-twist event location feeding the shared impact outcome map.
- **Next step:** Merge the PR; then wire NSX-90/91/92 to impact_outcome_map and add Tools launch-angle/spin-rate.

### DL-#14061 · Guard Xwininfo and Xwd Dependency Checks in OpenSim Visualizer Capture

- **State:** shipped
- **Shipped:** 2026-10-29
- **Owner:** unassigned
- **Issue:** #14061
- **Branch:** fix/14061-opensim-ci-xwininfo
- **PR:** #14062
- **Paths:** see #14062
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`4c39130c`; collated from changes/14061-opensim-xwininfo-guard.md)
- **Summary:** Guard xwininfo and xwd dependency checks in OpenSim visualizer capture

### DL-#13174 · HRG-D3: Render_Golfer_Realism_Showcase --V2-Spec Renders Labelled Address/Top/Impact Stills of the Full-Body-V2 Trunk Visuals; GL Stills Still Need a GL Host

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13174
- **Branch:** claude/ud-13174-1
- **PR:** #14035, #14212
- **Paths:** see #14035
- **Started:** 2026-10-29
- **Last verified:** 2026-11-02 (`99a4b0c5`; collated from changes/13174-attach-osmesa-mujoco-address-top-impact.md)
- **Summary:** HRG-D3: render_golfer_realism_showcase --v2-spec renders labelled address/top/impact stills of the full-body-v2 trunk visuals; GL stills still need a GL host
- **Next step:** Merge PR; close #13174.

### DL-#14053 · Correct 6X6 Mobility Scaling by Multiplying Angular Rows by L

- **State:** shipped
- **Shipped:** 2026-10-29
- **Owner:** unassigned
- **Issue:** #14053
- **Branch:** fix/14053-mobility-scales
- **PR:** #14058
- **Paths:** see #14058
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`d5705ebb`; collated from changes/14053-correct-6x6-mobility-scaling-by-multiply.md)
- **Summary:** Correct 6x6 mobility scaling by multiplying angular rows by L

### DL-#13971 · Drake Motion Matching: Catalog Finite-Difference Generalized Accelerations in Simulate_With_Coefficients (UD-36)

- **State:** shipped
- **Shipped:** 2026-10-28
- **Owner:** unassigned
- **Issue:** #13971
- **Branch:** test/13971-catalog-finite-difference-qdd-drake
- **PR:** #13982
- **Paths:** see #13982
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`934b44df`; collated from changes/13971-catalog-finite-difference-qdd-drake.md)
- **Summary:** Drake motion matching: catalog finite-difference generalized accelerations in simulate_with_coefficients (UD-36)

### DL-#13970 · Catalog Constant Static Friction Contract in Contact_Law_Kernel for Torsional_Moment_Kernel and Link Epic #13757 (UD-33)

- **State:** shipped
- **Shipped:** 2026-10-28
- **Owner:** unassigned
- **Issue:** #13970
- **Branch:** test/13970-catalog-torsional-moment-static-friction
- **PR:** #13980
- **Paths:** see #13980
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`eac54cbe`; collated from changes/13970-catalog-torsional-moment-static-friction.md)
- **Summary:** Catalog constant static friction contract in contact_law_kernel for torsional_moment_kernel and link Epic #13757 (UD-33)

### DL-#13972 · OpenSim Spine: Catalog Coordinate Unit-Speed Probing in Inverse_Dynamics _U_Indices (UD-39)

- **State:** shipped
- **Shipped:** 2026-10-28
- **Owner:** unassigned
- **Issue:** #13972
- **Branch:** test/13972-catalog-coordinate-unit-speed-probing
- **PR:** #13983
- **Paths:** see #13983
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`03f7e9b2`; collated from changes/13972-catalog-coordinate-unit-speed-probing.md)
- **Summary:** OpenSim spine: catalog coordinate unit-speed probing in inverse_dynamics _u_indices (UD-39)

### DL-#13969 · Clarify VelocityRepresentation Origin Convention in Docstrings, Documenting Evaluation at Base Reference Point / Center of Mass vs Featherstone World Origin O and Linking Epic #13789 (UD-13)

- **State:** shipped
- **Shipped:** 2026-10-28
- **Owner:** unassigned
- **Issue:** #13969
- **Branch:** doc/13969-clarify-inertial-spatial-velocity-origin
- **PR:** #13976
- **Paths:** see #13976
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`2ba8eb82`; collated from changes/13969-clarify-inertial-spatial-velocity-origin.md)
- **Summary:** Clarify VelocityRepresentation origin convention in docstrings, documenting evaluation at base reference point / center of mass vs Featherstone world origin O and linking Epic #13789 (UD-13)

### DL-#13967 · Enforce Engine Availability Assertion and Cite Simscape Baseline Prerequisite (UD-26)

- **State:** shipped
- **Shipped:** 2026-10-28
- **Owner:** unassigned
- **Issue:** #13967
- **Branch:** test/13967-cross-engine-parity-guards
- **PR:** #13973
- **Paths:** see #13973
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`7ec71954`; collated from changes/13967-cross-engine-parity-guards.md)
- **Summary:** Enforce engine availability assertion and cite Simscape baseline prerequisite (UD-26)

### DL-#13968 · Clarify and Assert Zero-Linear-Velocity Contract in Single_Floating_Body_H_G, Documenting Zero Linear Coriolis Bias Under Center-of-Mass Frame Conventions and Linking Epic #13789 (UD-14)

- **State:** shipped
- **Shipped:** 2026-10-28
- **Owner:** unassigned
- **Issue:** #13968
- **Branch:** doc/13968-single-floating-body-linear-velocity-contract
- **PR:** #13975
- **Paths:** see #13975
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`7296e223`; collated from changes/13968-single-floating-body-linear-velocity-contract.md)
- **Summary:** Clarify and assert zero-linear-velocity contract in single_floating_body_h_g, documenting zero linear Coriolis bias under center-of-mass frame conventions and linking Epic #13789 (UD-14)

### DL-#13962 · Require _Advance_to to Reach Target Time Within Substep Budget (UD-37)

- **State:** shipped
- **Shipped:** 2026-10-28
- **Owner:** unassigned
- **Issue:** #13962
- **Branch:** fix/13962-mujoco-advance-to-substep-budget
- **PR:** #13963
- **Paths:** see #13963
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`5e855a22`; collated from changes/13962-mujoco-advance-to-substep-budget.md)
- **Summary:** Require _advance_to to reach target time within substep budget (UD-37)

### DL-#13959 · Enforce Both Upper Bound and Zero Transverse Force in Check_Static_Equilibrium (UD-30)

- **State:** shipped
- **Shipped:** 2026-10-28
- **Owner:** unassigned
- **Issue:** #13959
- **Branch:** fix/13959-grip-static-equilibrium
- **PR:** #13960
- **Paths:** see #13960
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`15e55478`; collated from changes/13959-grip-static-equilibrium.md)
- **Summary:** Enforce both upper bound and zero transverse force in check_static_equilibrium (UD-30)

### DL-#13956 · Allow MacDonaldHanzelyModel and Constant-Coefficient Models to Use Declared Cl Without 0.155 Constant-Spin Clamp

- **State:** shipped
- **Shipped:** 2026-10-28
- **Owner:** unassigned
- **Issue:** #13956
- **Branch:** fix/13956-flight-models-lift-cap
- **PR:** #13958
- **Paths:** see #13958
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`eb8b7140`; collated from changes/13956-flight-models-lift-cap.md)
- **Summary:** Allow MacDonaldHanzelyModel and constant-coefficient models to use declared cl without 0.155 constant-spin clamp

### DL-#13953 · OpenSim: Correct SimTK Quaternion Ordering Documentation in Coord_Map (UD-12)

- **State:** shipped
- **Shipped:** 2026-10-28
- **Owner:** unassigned
- **Issue:** #13953
- **Branch:** fix/13953-opensim-simtk-quat-docs
- **PR:** #13954
- **Paths:** see #13954
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`f838b4e2`; collated from changes/13953-opensim-simtk-quat-docs.md)
- **Summary:** OpenSim: correct SimTK quaternion ordering documentation in coord_map (UD-12)

### DL-#13949 · Pinocchio Viz: Adjust 1-Based Impact_Idx to 0-Based Indexing in Error_Timecourse (UD-23)

- **State:** shipped
- **Shipped:** 2026-10-28
- **Owner:** unassigned
- **Issue:** #13949
- **Branch:** fix/13949-pinocchio-viz-impact-idx
- **PR:** #13950
- **Paths:** see #13950
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`2f369c5a`; collated from changes/13949-pinocchio-viz-impact-idx.md)
- **Summary:** Pinocchio viz: adjust 1-based impact_idx to 0-based indexing in error_timecourse (UD-23)

### DL-#13947 · Restore Docstrings in Motion_Matching Cost Term Helpers (UD-20)

- **State:** shipped
- **Shipped:** 2026-10-28
- **Owner:** unassigned
- **Issue:** #13947
- **Branch:** fix/13947-cost-term-docstrings
- **PR:** #13948
- **Paths:** see #13948
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`b5182571`; collated from changes/13947-cost-term-docstrings.md)
- **Summary:** Restore docstrings in motion_matching cost term helpers (UD-20)

### DL-#13869 · Architecture Budget: Full-Scan Mode With Checked-In Baseline Ratchet (UD-25)

- **State:** shipped
- **Shipped:** 2026-10-28
- **Owner:** unassigned
- **Issue:** #13869
- **Branch:** fix/13869-architecture-budget-baseline
- **PR:** #13934
- **Paths:** see #13934
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`d652ae11`; collated from changes/13869-architecture-budget-baseline.md)
- **Summary:** Architecture budget: full-scan mode with checked-in baseline ratchet (UD-25)

### DL-#13868 · Motion Matching: Architecture Guard and Canonical _Geodesic Helpers (UD-24)

- **State:** shipped
- **Shipped:** 2026-10-28
- **Owner:** unassigned
- **Issue:** #13868
- **Branch:** fix/13868-quaternion-helper-guard
- **PR:** #13933
- **Paths:** see #13933
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`d86cf2e5`; collated from changes/13868-quaternion-helper-guard.md)
- **Summary:** Motion matching: architecture guard and canonical _geodesic helpers (UD-24)

### DL-#13761 · HRG-HX4 Part 2: Bushing_Wrench Force_Limit_N Caps the Bushing Force Magnitude, Direction Kept, Moment Unchanged, Default None Unlimited and Bit-Identical (Option C, Python Bushing Law Only)

- **State:** shipped
- **Shipped:** 2026-10-29
- **Owner:** unassigned
- **Issue:** #13761
- **Branch:** claude/13761-bushing-force-cap
- **PR:** #13941, #13939, #14033
- **Paths:** see #13941
- **Started:** 2026-10-27
- **Last verified:** 2026-10-29 (`6c3b0bea`; collated from changes/13761-hrg-hx4-part-3-bushing-force-limit-wiring.md)
- **Summary:** HRG-HX4 part 2: bushing_wrench force_limit_n caps the bushing force magnitude, direction kept, moment unchanged, default None unlimited and bit-identical (option c, Python bushing law only)

### DL-#13758 · Add SO(3) Observability Evidence for Grip Fitting

- **State:** shipped
- **Shipped:** 2026-10-30
- **Owner:** unassigned
- **Issue:** #13758
- **Branch:** staff/issue-remediator-13758-4adf95
- **PR:** #13816, #14116
- **Paths:** see #13816
- **Started:** 2026-10-25
- **Last verified:** 2026-10-30 (`30025fe5`; collated from changes/13758-grip-neutral-prior-and-mirror-contract.md)
- **Summary:** Add SO(3) observability evidence for grip fitting

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

## Shipped (Last 90 Days)

### DL-#14518 · Run Release Presets Through the Run Store: Sim_Run.Preset_Run Records Which Files a Preset Wrote, Copies Them Into a Run Directory and Emits a Reproducible Run Manifest

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14518
- **Branch:** merged via #14524
- **PR:** #14524, #14534, #14528
- **Paths:** see #14524
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`725ae488`; collated from changes/14518-ud-sim-preset-run-stores-a-release-prese.md)
- **Summary:** Run release presets through the run store: sim_run.preset_run records which files a preset wrote, copies them into a run directory and emits a reproducible run manifest
- **Next step:** Shipped in PR #14524.

### DL-#14494 · ZVCF Is the Control Component M^-1 Tau (Constrained: P_C Tau With the Reaction Change); Gravity-Only Term Renamed Zero_Velocity_Zero_Torque_Acceleration With a One-Release Deprecation Path; Analysis Schema 4.0.0

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14494
- **Branch:** merged via #14515
- **PR:** #14515
- **Paths:** see #14515
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`8bf1b165`; collated from changes/14494-zvcf-is-the-control-component-m-1-tau-co.md)
- **Summary:** ZVCF is the control component M^-1 tau (constrained: P_c tau with the reaction change); gravity-only term renamed zero_velocity_zero_torque_acceleration with a one-release deprecation path; analysis schema 4.0.0
- **Next step:** Shipped in PR #14515.

### DL-#14493 · Superposition Residual for Forward Counterfactual Trajectories: ZTCF + ZVCF Matches the Base Only at the Fork; Integrated Sums Report Their Residual

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14493
- **Branch:** merged via #14510
- **PR:** #14510
- **Paths:** see #14510
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`17da8467`; collated from changes/14493-superposition-residual-for-forward-count.md)
- **Summary:** Superposition residual for forward counterfactual trajectories: ZTCF + ZVCF matches the base only at the fork; integrated sums report their residual
- **Next step:** Shipped in PR #14510.

### DL-#14492 · Drift_and_Control_Reactions: Closure Reactions Split Into Drift (Gravity, Velocity) and Control (Load Only) Parts That Sum to the Total, on the KKT Projector

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14492
- **Branch:** merged via #14512
- **PR:** #14512
- **Paths:** see #14512
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`df49a7a8`; collated from changes/14492-drift-and-control-reactions-closure-reac.md)
- **Summary:** drift_and_control_reactions: closure reactions split into drift (gravity, velocity) and control (load only) parts that sum to the total, on the KKT projector
- **Next step:** Shipped in PR #14512.

### DL-#14491 · Golden ZTCF/ZVCF Contract Tests: Drift + Control = Total With Gravity Counted Once, on the Analytical and MuJoCo Double Pendulum

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14491
- **Branch:** merged via #14508
- **PR:** #14508
- **Paths:** see #14508
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`fe014c2a`; collated from changes/14491-golden-tests-drift-control-total-gravity.md)
- **Summary:** Golden ZTCF/ZVCF contract tests: drift + control = total with gravity counted once, on the analytical and MuJoCo double pendulum
- **Next step:** Shipped in PR #14508.

### DL-#14473 · Putting: PyQt6 Tile Readouts and Panel Style Move Out of gui.py Into _Scene_Text and _Style (No Behaviour Change; Keeps gui.py Under the Monolith Threshold)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14473
- **Branch:** merged via #14533
- **PR:** #14533
- **Paths:** see #14533
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`b50c8bb8`; collated from changes/14473-putting-gui-split.md)
- **Summary:** Putting: PyQt6 tile readouts and panel style move out of gui.py into _scene_text and _style (no behaviour change; keeps gui.py under the monolith threshold)
- **Next step:** Shipped in PR #14533.

### DL-#14467 · Putting Analytics: Player-Specific Bellman Expected Putts on the Caller's Green (Optimal Pace per Distance, Shared Skill Roll Model) and One Putt's Strokes Gained Against It, on /Tools/Putting-Analytics/Strokes-Gained

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14467
- **Branch:** merged via #14527
- **PR:** #14527
- **Paths:** see #14527
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`f3e71c73`; collated from changes/14467-putting-sg-core.md)
- **Summary:** Putting analytics: player-specific Bellman expected putts on the caller's green (optimal pace per distance, shared skill roll model) and one putt's strokes gained against it, on /tools/putting-analytics/strokes-gained
- **Next step:** Shipped in PR #14527.

### DL-#14465 · Putting: Auto-Solve Read on the PyQt6 Tile (Core Solve_Aim Off the GUI Thread, Writes Pace and Aim Into the Controls) and One-Call Aim+Pace on /Read-Green

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14465
- **Branch:** merged via #14531
- **PR:** #14531
- **Paths:** see #14531
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`2cc0a7ac`; collated from changes/14465-putting-auto-solve.md)
- **Summary:** Putting: Auto-Solve Read on the PyQt6 tile (core solve_aim off the GUI thread, writes pace and aim into the controls) and one-call aim+pace on /read-green
- **Next step:** Shipped in PR #14531.

### DL-#14463 · Putting Desktop: Green Type Selector (Cross Slope or Any Core GreenSpec Preset), Seeded Undulation in mm or in, and a Green Seed

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14463
- **Branch:** merged via #14526
- **PR:** #14526, #14521
- **Paths:** see #14526
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`2bb758eb`; collated from changes/14463-putting-green-spec.md)
- **Summary:** Putting desktop: Green Type selector (cross slope or any core GreenSpec preset), seeded undulation in mm or in, and a green seed
- **Next step:** Shipped in PR #14526.

### DL-#14461 · Putting Core: Named Putter Balance Presets (Toe-Hang, Face-Balanced, Zero-Torque, Counterbalanced) Over PutterState; GET /Tools/Putting-Green/Putter-Presets Lists Them and /Simulate-3D Accepts Balance_Type

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14461
- **Branch:** merged via #14516
- **PR:** #14516, #14519, #14511
- **Paths:** see #14516
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`9a5d0d0c`; collated from changes/14461-putting-strike-core.md)
- **Summary:** Putting core: named putter balance presets (toe-hang, face-balanced, zero-torque, counterbalanced) over PutterState; GET /tools/putting-green/putter-presets lists them and /simulate-3d accepts balance_type
- **Next step:** Shipped in PR #14516.

### DL-#14459 · Putting Analytics: Dispersion Batch (Up to 2000 Putts) and 0-45 Ft Make-Spread Services and Routes on the Caller's Green, From the Shared Skill Roll Model

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14459
- **Branch:** merged via #14487
- **PR:** #14487, #14506, #14504, #14513
- **Paths:** see #14487
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`52f41a89`; collated from changes/14459-putting-variance-close.md)
- **Summary:** Putting analytics: dispersion batch (up to 2000 putts) and 0-45 ft make-spread services and routes on the caller's green, from the shared skill roll model
- **Next step:** Shipped in PR #14487.

### DL-#14458 · React Putting Page: Metric/Imperial Units Selector Mirroring units.py (SI Internally), Golden Parity With the Python Core

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14458
- **Branch:** merged via #14489
- **PR:** #14489
- **Paths:** see #14489
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`96702ab2`; collated from changes/14458-react-putting-page-metric-imperial-units.md)
- **Summary:** React putting page: Metric/Imperial units selector mirroring units.py (SI internally), golden parity with the Python core
- **Next step:** Shipped in PR #14489.

### DL-#14457 · Putting Green Simulator: Metric/Imperial Units Selector That Converts Controls and Metrics in Place and Persists the Shared Unit Preference

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14457
- **Branch:** merged via #14485
- **PR:** #14485
- **Paths:** see #14485
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`8eff3cca`; collated from changes/14457-putting-units-selector.md)
- **Summary:** Putting Green Simulator: Metric/Imperial units selector that converts controls and metrics in place and persists the shared unit preference
- **Next step:** Shipped in PR #14485.

### DL-#14456 · Putting Feature Parity Matrix: 13 Advanced Features X Core/API, PyQt6, React, Tauri, Studio With File Evidence; Registry Test Tracks 53 Gaps as Strict Xfails

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14456
- **Branch:** merged via #14483
- **PR:** #14483
- **Paths:** see #14483
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`a1af70c1`; collated from changes/14456-putting-feature-parity-matrix-13-advance.md)
- **Summary:** Putting feature parity matrix: 13 advanced features x core/API, PyQt6, React, Tauri, studio with file evidence; registry test tracks 53 gaps as strict xfails
- **Next step:** Shipped in PR #14483.

### DL-#14448 · Putting Green Simulator: Python -m Launch Entry Point With --Smoke (One Putt, Clean Exit) and a Clear Exit-2 Message When the Putting Core Is Missing

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14448
- **Branch:** merged via #14455
- **PR:** #14455
- **Paths:** see #14455
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`1ef9306e`; collated from changes/14448-putting-green-simulator-python-m-launch.md)
- **Summary:** Putting Green Simulator: python -m launch entry point with --smoke (one putt, clean exit) and a clear exit-2 message when the putting core is missing
- **Next step:** Shipped in PR #14455.

### DL-#14446 · Add Append-Only Model-Matching Metrics Trend File and Daily-Report Delta Snippet

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14446
- **Branch:** merged via #14507
- **PR:** #14507
- **Paths:** see #14507
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`56c5a326`; collated from changes/14446-add-append-only-model-matching-metrics-t.md)
- **Summary:** Add append-only model-matching metrics trend file and daily-report delta snippet
- **Next step:** Shipped in PR #14507.

### DL-#14445 · Add Headless Best-Model Render Script (SHA-256-Pinned MuJoCo and OpenSim OS9 Inputs, Per-Model Summaries)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14445
- **Branch:** merged via #14505
- **PR:** #14505
- **Paths:** see #14505
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`c20dadd9`; collated from changes/14445-add-headless-best-model-render-script-sh.md)
- **Summary:** Add headless best-model render script (SHA-256-pinned MuJoCo and OpenSim OS9 inputs, per-model summaries)
- **Next step:** Shipped in PR #14505.

### DL-#14444 · Add the OpenSim Segment Scaling-Check Command: Gates Scale Factors and Rigid Residuals on the #13991 Length-Plausibility Bounds and Writes Metrics JSON Plus a Run-Manifest-V1

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14444
- **Branch:** merged via #14509
- **PR:** #14509
- **Paths:** see #14509
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`5cb2bd25`; collated from changes/14444-add-the-opensim-segment-scaling-check-co.md)
- **Summary:** Add the OpenSim segment scaling-check command: gates scale factors and rigid residuals on the #13991 length-plausibility bounds and writes metrics JSON plus a run-manifest-v1
- **Next step:** Shipped in PR #14509.

### DL-#14443 · Add Per-Engine Model-Matching Status Report (Ledger, Generator, STATUS.md) Marking Unverified Numbers

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14443
- **Branch:** merged via #14484
- **PR:** #14484
- **Paths:** see #14484
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`837a97b4`; collated from changes/14443-add-per-engine-model-matching-status-rep.md)
- **Summary:** Add per-engine model-matching status report (ledger, generator, STATUS.md) marking unverified numbers
- **Next step:** Shipped in PR #14484.

### DL-#14440 · Additivity Charts for the Pinned WSCG Run: ZTCF, ZVCF, Their Sum and the Full Response per Lead-Arm Joint, With the Residual Against Its Declared Tolerance Band and Event Markers, Written as SVG and PNG on Demand (Scripts/Research/Proximal_Distal_Energy/wscg_ztcf_additivity_charts.py).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14440
- **Branch:** merged via #14529
- **PR:** #14529
- **Paths:** see #14529
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`84d8cbfd`; collated from changes/14440-additivity-charts-for-the-pinned-wscg-ru.md)
- **Summary:** Additivity charts for the pinned WSCG run: ZTCF, ZVCF, their sum and the full response per lead-arm joint, with the residual against its declared tolerance band and event markers, written as SVG and PNG on demand (scripts/research/proximal_distal_energy/wscg_ztcf_additivity_charts.py).
- **Next step:** Shipped in PR #14529.

### DL-#14439 · Render Force and Torque Vectors at Swing Events of a Manifest Run (Src/Sim_Run/vector_overlay.py), With Legend Lines That State Each Reference's Drawn Size, and Keep the Matplotlib Legend Text Inside Its Inset.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14439
- **Branch:** merged via #14525
- **PR:** #14525
- **Paths:** see #14525
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`f542bbef`; collated from changes/14439-render-force-and-torque-vectors-at-swing.md)
- **Summary:** Render force and torque vectors at swing events of a manifest run (src/sim_run/vector_overlay.py), with legend lines that state each reference's drawn size, and keep the Matplotlib legend text inside its inset.
- **Next step:** Shipped in PR #14525.

### DL-#14438 · Assert ZTCF + ZVCF = Full (Canonical ZVCF Control Component, Gravity Once, Pointwise Only) and the ZTCF Velocity Drift on the Pinned WSCG Model, With Saved Residual Series

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14438
- **Branch:** merged via #14523
- **PR:** #14523
- **Paths:** see #14523
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`4406bff4`; collated from changes/14438-assert-the-exact-ztcf-split-on-the-pinned.md)
- **Summary:** Assert ZTCF + ZVCF = full (canonical ZVCF control component, gravity once, pointwise only) and the ZTCF velocity drift on the pinned WSCG model, with saved residual series
- **Next step:** Shipped in PR #14523.

### DL-#14437 · Add the WSCG Best-Match MuJoCo Model Selection Record (Structural Feature Score, Runner-Up, Run-Manifest Pin)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14437
- **Branch:** merged via #14488
- **PR:** #14488
- **Paths:** see #14488
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`85bac31d`; collated from changes/14437-add-the-wscg-best-match-mujoco-model-sel.md)
- **Summary:** Add the WSCG best-match MuJoCo model selection record (structural feature score, runner-up, run-manifest pin)
- **Next step:** Shipped in PR #14488.

### DL-#14435 · OpenSim, Drake, Pinocchio and MyoSuite Runs Emit the Shared Run Manifest; Absent Engines Skip With a Reason; Reproduce Fails Closed on Git Revisions and Missing Parameters

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14435
- **Branch:** merged via #14522
- **PR:** #14522
- **Paths:** see #14522
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`c38dc0ff`; collated from changes/14435-opensim-drake-pinocchio-and-myosuite-run.md)
- **Summary:** OpenSim, Drake, Pinocchio and MyoSuite runs emit the shared run manifest; absent engines skip with a reason; reproduce fails closed on git revisions and missing parameters
- **Next step:** Shipped in PR #14522.

### DL-#14432 · Emit a Run Manifest From the MuJoCo Run Path, With Output Artifact Hashes and a Shared Engine-Neutral Emitter

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14432
- **Branch:** merged via #14514
- **PR:** #14514
- **Paths:** see #14514
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`3e0a2f69`; collated from changes/14432-emit-a-run-manifest-from-the-mujoco-run-p.md)
- **Summary:** Emit a run manifest from the MuJoCo run path, with output artifact hashes and a shared engine-neutral emitter
- **Next step:** Shipped in PR #14514.

### DL-#14431 · Add the Run-Manifest V1 Schema and Validator: Model Id and Immutable Revision, Parameters, Input-Data Digest, Seed, Engine and Version, UD Commit and Created_at; Mutable Refs and Missing Provenance Fail

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14431
- **Branch:** merged via #14454
- **PR:** #14454
- **Paths:** see #14454
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`5607ae01`; collated from changes/14431-add-the-run-manifest-v1-schema-and-valid.md)
- **Summary:** Add the run-manifest v1 schema and validator: model id and immutable revision, parameters, input-data digest, seed, engine and version, UD commit and created_at; mutable refs and missing provenance fail
- **Next step:** Shipped in PR #14454.

### DL-#2427 · Add Report-Only Pr-Body-Lint Pull_Request_Target Caller (RM Reusable Workflow) and Contract Test.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #2427
- **Branch:** merged via #14405
- **PR:** #14405
- **Paths:** see #14405
- **Started:** 2026-11-03
- **Last verified:** 2026-11-03 (`e8d924b0`; collated from changes/2427-add-report-only-pr-body-lint-pull-reques.md)
- **Summary:** Add report-only pr-body-lint pull_request_target caller (RM reusable workflow) and contract test.
- **Next step:** Shipped in PR #14405.

### DL-#14407 · Title-Case Gate Treats the Middle Dot as a Heading Segment Boundary, so Collated DL Headings Starting With an Article Pass

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14407
- **Branch:** merged via #14408
- **PR:** #14408
- **Paths:** see #14408
- **Started:** 2026-11-03
- **Last verified:** 2026-11-03 (`65945a42`; collated from changes/14407-title-case-gate-treats-the-middle-dot-as.md)
- **Summary:** Title-case gate treats the middle dot as a heading segment boundary, so collated DL headings starting with an article pass
- **Next step:** Shipped in PR #14408.

### DL-#14390 · Release Companion-Protected-Main Attestations Retry Once After a Transient OIDC Token Failure

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14390
- **Branch:** merged via #14391
- **PR:** #14391
- **Paths:** see #14391
- **Started:** 2026-11-03
- **Last verified:** 2026-11-03 (`b0a79625`; collated from changes/14390-release-companion-protected-main-attesta.md)
- **Summary:** Release companion-protected-main attestations retry once after a transient OIDC token failure
- **Next step:** Shipped in PR #14391.

### DL-#14283 · Playwright Mobile Layout Smoke With Ratchet Allowlist for Six Core Routes.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14283
- **Branch:** merged via #14393
- **PR:** #14393
- **Paths:** see #14393
- **Started:** 2026-11-03
- **Last verified:** 2026-11-03 (`917d30e6`; collated from changes/14283-playwright-mobile-layout-smoke-with-ratc.md)
- **Summary:** Playwright mobile layout smoke with ratchet allowlist for six core routes.
- **Next step:** Shipped in PR #14393.

### DL-#14235 · The Web App Has a Putting Analytics Page at /Tools/Putting-Analytics That Runs the Best-Aim-and-Pace, Misread-Cost and Putt-Class Library Studies Through the API and Shows the Desktop Tool's Rows; the Parity Entry Is Now Parity.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14235
- **Branch:** merged via #14394
- **PR:** #14394
- **Paths:** see #14394
- **Started:** 2026-11-03
- **Last verified:** 2026-11-03 (`4d988bb1`; collated from changes/14235-the-web-app-has-a-putting-analytics-page.md)
- **Summary:** The web app has a Putting Analytics page at /tools/putting-analytics that runs the best-aim-and-pace, misread-cost and putt-class library studies through the API and shows the desktop tool's rows; the parity entry is now parity.
- **Next step:** Shipped in PR #14394.

### DL-#14200 · Record Putting Studio Offline Bridge Evaluation and PyQt6 Parity Exemption

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14200
- **Branch:** merged via #14399
- **PR:** #14399
- **Paths:** see #14399
- **Started:** 2026-11-03
- **Last verified:** 2026-11-03 (`df56d857`; collated from changes/14200-record-putting-studio-offline-bridge-eva.md)
- **Summary:** Record putting studio offline bridge evaluation and PyQt6 parity exemption
- **Next step:** Shipped in PR #14399.

### DL-#14398 · Pin Ci-Standard Hosted Runner to Ubuntu-24.04 so the Ubuntu-Latest 26.04 Rollout Stops Ejecting Merge Groups (Migration Tracked In #14397)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14398
- **Branch:** merged via #14400
- **PR:** #14400
- **Paths:** see #14400
- **Started:** 2026-11-03
- **Last verified:** 2026-11-03 (`8e0c06d2`; collated from changes/14398-pin-ci-standard-hosted-runner-to-ubuntu.md)
- **Summary:** Pin ci-standard hosted runner to ubuntu-24.04 so the ubuntu-latest 26.04 rollout stops ejecting merge groups (migration tracked in #14397)
- **Next step:** Shipped in PR #14400.

### DL-#14252 · Add Theme Tokens Codemod Tool, Unit Tests, CI Ratchet in colorGuard.Test.Ts, and Migrate Settings Page to Semantic Tokens

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14252
- **Branch:** merged via #14371
- **PR:** #14371
- **Paths:** see #14371
- **Started:** 2026-11-03
- **Last verified:** 2026-11-03 (`b19b771f`; collated from changes/14252-add-theme-tokens-codemod-tool-unit-tests.md)
- **Summary:** Add theme tokens codemod tool, unit tests, CI ratchet in colorGuard.test.ts, and migrate Settings page to semantic tokens
- **Next step:** Shipped in PR #14371.

### DL-#14385 · Tests: Remove the Preload_Engines Workaround From the Hand-Consistency and Palm-Thumb Pad Parity Integration Suites; #14382's _Protect_Engine_Modules Fix Makes It Redundant.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14385
- **Branch:** merged via #14386
- **PR:** #14386
- **Paths:** see #14386
- **Started:** 2026-11-03
- **Last verified:** 2026-11-03 (`1152a0ce`; collated from changes/14385-remove-engine-preload.md)
- **Summary:** tests: remove the preload_engines workaround from the hand-consistency and palm-thumb pad parity integration suites; #14382's _protect_engine_modules fix makes it redundant.
- **Next step:** Shipped in PR #14386.

### DL-#14381 · Tests: _Protect_Engine_Modules Keeps Genuine Pinocchio/Pydrake Modules First Imported During a Test (Their Native Extensions Cannot Be Re-Imported Once Discarded); Stubs and Src.Engines Additions Are Still Discarded and Replaced Modules Restored.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14381
- **Branch:** merged via #14382
- **PR:** #14382
- **Paths:** see #14382
- **Started:** 2026-11-03
- **Last verified:** 2026-11-03 (`2bb19026`; collated from changes/14381-engine-module-reimport.md)
- **Summary:** tests: _protect_engine_modules keeps genuine pinocchio/pydrake modules first imported during a test (their native extensions cannot be re-imported once discarded); stubs and src.engines additions are still discarded and replaced modules restored.
- **Next step:** Shipped in PR #14382.

### DL-#13997 · Grip Hand Mesh Gains a Dorsum (Metacarpal Block) Riding on the Palm Out Past the Knuckles, Clear of the Grip; OpenSim Hand STLs and the GS3DX Grip Topology Regenerated (Slice 1 Of 3)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13997
- **Branch:** merged via #14373
- **PR:** #14373
- **Paths:** see #14373
- **Started:** 2026-11-03
- **Last verified:** 2026-11-03 (`5679b382`; collated from changes/13997-grip-hand-mesh-dorsum-shell.md)
- **Summary:** Grip hand mesh gains a dorsum (metacarpal block) riding on the palm out past the knuckles, clear of the grip; OpenSim hand STLs and the GS3DX grip topology regenerated (slice 1 of 3)
- **Next step:** Shipped in PR #14373.

### DL-#14376 · Fix CANONICAL_ENGINE_TESTS Path for Opensim in Test_Fit_Result_Contract and Add Pytestmark Unit.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14376
- **Branch:** merged via #14377
- **PR:** #14377
- **Paths:** see #14377
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`6555234c`; collated from changes/14376-canonical-engine-test-path.md)
- **Summary:** Fix CANONICAL_ENGINE_TESTS path for opensim in test_fit_result_contract and add pytestmark unit.
- **Next step:** Shipped in PR #14377.

### DL-#14268 · Mobile-Friendly Long-Running Job Status View

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14268
- **Branch:** agy/14268-mobile-job-status
- **PR:** #14366
- **Paths:** ui/src/components/layout/JobStatusView.tsx,ui/src/components/layout/JobStatusView.test.tsx,ui/src/pages/BunkerWorkbench.tsx,src/api/routes/bunker_workbench_jobs.py
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`e0ddf5f2`; collated from changes/14268-mobile-friendly-long-running-job-status-view.md)
- **Summary:** Mobile-friendly long-running job status view
- **Next step:** Shipped in PR #14366.

### DL-#14267 · Adapt Controls for Touch and Small Screens

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14267
- **Branch:** agy/14267-adapt-controls-touch-mobile
- **PR:** #14364
- **Paths:** ui/src/index.css,ui/src/components/ui/touchTarget.test.tsx
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`6f70ba78`; collated from changes/14267-adapt-controls-for-touch-and-small-scree.md)
- **Summary:** Adapt controls for touch and small screens
- **Next step:** Shipped in PR #14364.

### DL-#13764 · Cross-Engine Hand Mass and Length Consistency Test (HRG-HX7 Slice 1): MuJoCo, Drake, Pinocchio and OpenSim Hand Body Masses Match the Separated-Hand Spec; Visual-Only Hand Length Is Measured Where Exported and Reported Elsewhere

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13764
- **Branch:** merged via #14358
- **PR:** #14358, #14367
- **Paths:** see #14358
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`ad434661`; collated from changes/13764-cross-engine-hand-frame-test-hrg-hx7-sli.md)
- **Summary:** Cross-engine hand mass and length consistency test (HRG-HX7 slice 1): MuJoCo, Drake, Pinocchio and OpenSim hand body masses match the separated-hand spec; visual-only hand length is measured where exported and reported elsewhere
- **Next step:** Shipped in PR #14358.

### DL-#14260 · Shared ToolShell With Persistent Navigation, Breadcrumb, and Actions Slot for Tool Pages

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14260
- **Branch:** merged via #14352
- **PR:** #14352
- **Paths:** see #14352
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`91b0731c`; collated from changes/14260-shared-toolshell-with-persistent-navigat.md)
- **Summary:** Shared ToolShell with persistent navigation, breadcrumb, and actions slot for tool pages
- **Next step:** Shipped in PR #14352.

### DL-#14265 · Delete Drifted Local Themes.Json Copy and Consume Tools Theme Catalog With Legacy Id Migration

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14265
- **Branch:** merged via #14349
- **PR:** #14349
- **Paths:** see #14349
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`01303fb2`; collated from changes/14265-delete-drifted-local-themes-json-copy-an.md)
- **Summary:** Delete drifted local themes.json copy and consume Tools theme catalog with legacy id migration
- **Next step:** Shipped in PR #14349.

### DL-#14212 · Fast-Tier Unit Gate Runs Matched-Swing README Freshness Test When Ledger Changes.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14212
- **Branch:** merged via #14345
- **PR:** #14345
- **Paths:** see #14345
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`173e5ea7`; collated from changes/14212-fast-tier-unit-gate-runs-matched-swing-r.md)
- **Summary:** Fast-tier unit gate runs matched-swing README freshness test when ledger changes.
- **Next step:** Shipped in PR #14345.

### DL-#14261 · Workspace Cards Show Authored Tool Names and One-Line Summaries Instead of a Template Sentence, and Both Card Actions Share One Primary Button Style Labeled Open or Open in Window.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14261
- **Branch:** merged via #14342
- **PR:** #14342
- **Paths:** see #14342
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`60b23d44`; collated from changes/14261-workspace-cards-show-authored-tool-names.md)
- **Summary:** Workspace cards show authored tool names and one-line summaries instead of a template sentence, and both card actions share one primary button style labeled Open or Open in Window.
- **Next step:** Shipped in PR #14342.

### DL-#14258 · Web Launcher Resolves models.yaml Logo Paths and Hides ROC Tile When Bundle Absent.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14258
- **Branch:** merged via #14344
- **PR:** #14344
- **Paths:** see #14344
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`35f2dab8`; collated from changes/14258-web-launcher-resolves-models-yaml-logo-p.md)
- **Summary:** Web launcher resolves models.yaml logo paths and hides ROC tile when bundle absent.
- **Next step:** Shipped in PR #14344.

### DL-#14262 · Canonical-Core Estimation and Comparison Are Marked Planned: They Leave Home and the Results & Compare Workspace, Stay Reachable From Developer & Research, and the Desktop Chip Reads Planned Instead of Ready

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14262
- **Branch:** merged via #14339
- **PR:** #14339
- **Paths:** see #14339
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`e5f0d867`; collated from changes/14262-canonical-core-estimation-and-comparison.md)
- **Summary:** Canonical-core estimation and comparison are marked planned: they leave home and the Results & Compare workspace, stay reachable from Developer & Research, and the desktop chip reads Planned instead of Ready
- **Next step:** Shipped in PR #14339.

### DL-#14263 · First-Run Onboarding Is a Non-Blocking Docked Panel Instead of a Full-Screen Modal, so Home Content Is Usable Immediately

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14263
- **Branch:** merged via #14336
- **PR:** #14336
- **Paths:** see #14336
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`52a592cc`; collated from changes/14263-first-run-onboarding-is-a-non-blocking-d.md)
- **Summary:** First-run onboarding is a non-blocking docked panel instead of a full-screen modal, so home content is usable immediately
- **Next step:** Shipped in PR #14336.

### DL-#14256 · Non-JSON API Responses Raise a Typed ApiError and describeApiError Maps Them to Plain-Language Copy; Golf Simulator Shows 'Simulator API Not Reachable' With Retry and Explains Its Disabled Controls

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14256
- **Branch:** merged via #14332
- **PR:** #14332
- **Paths:** see #14332
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`f575ace5`; collated from changes/14256-non-json-api-responses-raise-a-typed-api.md)
- **Summary:** Non-JSON API responses raise a typed ApiError and describeApiError maps them to plain-language copy; Golf Simulator shows 'Simulator API not reachable' with Retry and explains its disabled controls
- **Next step:** Shipped in PR #14332.

### DL-#14255 · Keep the Dashboard Launch Footer Clear of the Floating Diagnostics Badge: The Footer Reserves DIAGNOSTICS_BADGE_CLEARANCE and the Badge Is Icon-Only Below Md

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14255
- **Branch:** merged via #14327
- **PR:** #14327
- **Paths:** see #14327
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`33b26dff`; collated from changes/14255-keep-the-dashboard-launch-footer-clear-o.md)
- **Summary:** Keep the dashboard launch footer clear of the floating diagnostics badge: the footer reserves DIAGNOSTICS_BADGE_CLEARANCE and the badge is icon-only below md
- **Next step:** Shipped in PR #14327.

### DL-#14321 · Rust_Adapter Joins the Stream Worker QThread After Its Event Loop Quits (Pair of Tools#5951)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14321
- **Branch:** merged via #14322
- **PR:** #14322
- **Paths:** see #14322
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`598ae366`; collated from changes/14321-rust-adapter-joins-the-stream-worker-qth.md)
- **Summary:** rust_adapter joins the stream worker QThread after its event loop quits (pair of Tools#5951)
- **Next step:** Shipped in PR #14322.

### DL-#14281 · Controls Get a 44Px Hit Area on Coarse Pointers (Touch-Target Utility, Primitives and a Native-Control Base Rule), and Arbitrary Text Sizes Under 12Px Are Raised to Text-Xs

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14281
- **Branch:** merged via #14324
- **PR:** #14324
- **Paths:** see #14324
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`7a1196d5`; collated from changes/14281-controls-get-a-44px-hit-area-on-coarse-p.md)
- **Summary:** Controls get a 44px hit area on coarse pointers (touch-target utility, primitives and a native-control base rule), and arbitrary text sizes under 12px are raised to text-xs
- **Next step:** Shipped in PR #14324.

### DL-#14282 · Wide Content on Lift Baseline, Analysis Tools, Terrain, Data Explorer and Golf Simulator Now Scrolls or Wraps at Phone Width Instead of Being Clipped

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14282
- **Branch:** merged via #14319
- **PR:** #14319
- **Paths:** see #14319
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`d92b4b1f`; collated from changes/14282-wide-content-on-lift-baseline-analysis-t.md)
- **Summary:** Wide content on Lift Baseline, Analysis Tools, Terrain, Data Explorer and Golf Simulator now scrolls or wraps at phone width instead of being clipped
- **Next step:** Shipped in PR #14319.

### DL-#14289 · Pinned Offline Embedding Tests Fail on Bad Retrieval and Unexpected Initialization

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14289
- **Branch:** merged via #14311
- **PR:** #14311
- **Paths:** see #14311
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`9ebf7a40`; collated from changes/14289-pinned-offline-local-embedding-tests.md)
- **Summary:** Pinned offline embedding tests fail on bad retrieval and unexpected initialization
- **Next step:** Shipped in PR #14311.

### DL-#14280 · Dashboard Header Wraps With Icon-Only Actions Below Md and the Task Workspaces Links Wrap Instead of Clipping, so Every Action Is Reachable at 390 And 768

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14280
- **Branch:** merged via #14315
- **PR:** #14315
- **Paths:** see #14315
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`f5df4a57`; collated from changes/14280-dashboard-header-wraps-with-icon-only-ac.md)
- **Summary:** Dashboard header wraps with icon-only actions below md and the Task Workspaces links wrap instead of clipping, so every action is reachable at 390 and 768
- **Next step:** Shipped in PR #14315.

### DL-#14310 · Stop the Deprecated-DynamicsProvider Scan From Timing Out and Killing an Xdist Worker: Only Files Containing the Name Are Parsed (42.8s to 2.3s).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14310
- **Branch:** merged via #14313
- **PR:** #14313
- **Paths:** see #14313
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`e928a82f`; collated from changes/14310-stop-the-deprecated-dynamicsprovider-sca.md)
- **Summary:** Stop the deprecated-DynamicsProvider scan from timing out and killing an xdist worker: only files containing the name are parsed (42.8s to 2.3s).
- **Next step:** Shipped in PR #14313.

### DL-#14309 · Conflict Disarm Now Calls the Public Fleet-Workflows Reusable Workflow Pinned to 367B182c Instead of an Inlined Sweep; Dry_Run and the PR 12634 Exclusion Are Unchanged, and a Contract Test Guards the Uses: Ref and the Full-SHA Pin

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14309
- **Branch:** merged via #14312
- **PR:** #14312
- **Paths:** see #14312
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`fd844a1c`; collated from changes/14309-conflict-disarm-now-calls-the-public-fle.md)
- **Summary:** Conflict Disarm now calls the public fleet-workflows reusable workflow pinned to 367b182c instead of an inlined sweep; dry_run and the PR 12634 exclusion are unchanged, and a contract test guards the uses: ref and the full-SHA pin
- **Next step:** Shipped in PR #14312.

### DL-#14279 · Simulation, Model Explorer and Character Builder Move Onto WorkspaceShell: Below Lg the 3D View Fills the Width and Side Panels Open as Drawers From 44Px Toggles; the Scene3D View Toolbar Wraps on Phones

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14279
- **Branch:** merged via #14306
- **PR:** #14306
- **Paths:** see #14306
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`43bfa59b`; collated from changes/14279-simulation-model-explorer-and-character.md)
- **Summary:** Simulation, Model Explorer and Character Builder move onto WorkspaceShell: below lg the 3D view fills the width and side panels open as drawers from 44px toggles; the Scene3D view toolbar wraps on phones
- **Next step:** Shipped in PR #14306.

### DL-#14297 · Pinocchio Inverse-Dynamics Tests Fail on a Broken Real Constructor

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14297
- **Branch:** merged via #14301
- **PR:** #14301
- **Paths:** see #14301
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`4ba3cd06`; collated from changes/14297-pinocchio-inverse-dynamics-tests-fail-on.md)
- **Summary:** Pinocchio inverse-dynamics tests fail on a broken real constructor
- **Next step:** Shipped in PR #14301.

### DL-#14296 · Pinocchio Kinematic Audit Fails on Missing URDF/Joint/Frame Instead of Skipping

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14296
- **Branch:** merged via #14300
- **PR:** #14300
- **Paths:** see #14300
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`13ebd18d`; collated from changes/14296-pinocchio-kinematic-audit-fails-on-missi.md)
- **Summary:** Pinocchio kinematic audit fails on missing URDF/joint/frame instead of skipping
- **Next step:** Shipped in PR #14300.

### DL-#14295 · Real ChatSessionManager Lifecycle, Signal and Fault Tests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14295
- **Branch:** merged via #14305
- **PR:** #14305
- **Paths:** see #14305
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`50699ca7`; collated from changes/14295-real-chatsessionmanager-lifecycle-signal.md)
- **Summary:** Real ChatSessionManager lifecycle, signal and fault tests
- **Next step:** Shipped in PR #14305.

### DL-#14294 · MuJoCo Dependency Tests Check Qpos Contract and Rendering API Symbols

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14294
- **Branch:** merged via #14304
- **PR:** #14304
- **Paths:** see #14304
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`584e76d2`; collated from changes/14294-mujoco-dependency-tests-check-qpos-contr.md)
- **Summary:** MuJoCo dependency tests check qpos contract and rendering API symbols
- **Next step:** Shipped in PR #14304.

### DL-#14288 · RAG Embedding Determinism Checked Against an Explicit Drift Tolerance

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14288
- **Branch:** merged via #14302
- **PR:** #14302
- **Paths:** see #14302
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`6e3b352a`; collated from changes/14288-rag-embedding-determinism-checked-agains.md)
- **Summary:** RAG embedding determinism checked against an explicit drift tolerance
- **Next step:** Shipped in PR #14302.

### DL-#14293 · Live-View Controller Tests Fail on Loader Import or C3D Parse Errors; Skip Only on Absent Fixture or Ezc3d

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14293
- **Branch:** merged via #14299
- **PR:** #14299
- **Paths:** see #14299
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`e5d646a6`; collated from changes/14293-live-view-controller-tests-fail-on-loade.md)
- **Summary:** Live-view controller tests fail on loader import or C3D parse errors; skip only on absent fixture or ezc3d
- **Next step:** Shipped in PR #14299.

### DL-#14292 · PCA/Kalman Tests Fail on Product Import Errors; Skip Only When the Pinned Tools Tree Is Absent

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14292
- **Branch:** merged via #14298
- **PR:** #14298
- **Paths:** see #14298
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`ffe93043`; collated from changes/14292-pca-kalman-tests-fail-on-product-import.md)
- **Summary:** PCA/Kalman tests fail on product import errors; skip only when the pinned Tools tree is absent
- **Next step:** Shipped in PR #14298.

### DL-#14254 · Add the 5 Missing Tool Routes to KNOWN_APP_ROUTES and a Vitest Guard That Fails When App.Tsx Routes and the Canonical List Diverge

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14254
- **Branch:** merged via #14287
- **PR:** #14287
- **Paths:** see #14287
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`01fdf38d`; collated from changes/14254-add-the-5-missing-tool-routes-to-known-a.md)
- **Summary:** Add the 5 missing tool routes to KNOWN_APP_ROUTES and a vitest guard that fails when App.tsx routes and the canonical list diverge
- **Next step:** Shipped in PR #14287.

### DL-#14227 · Pinocchio Swing Model Uses the Shared GRAVITY_M_S2 Constant, so Its RNEA Agrees With the CasADi Symbolic Model Again After #13799

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14227
- **Branch:** merged via #14228
- **PR:** #14228
- **Paths:** see #14228
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`6486357d`; collated from changes/14227-pinocchio-swing-model-uses-the-shared-gr.md)
- **Summary:** Pinocchio swing model uses the shared GRAVITY_M_S2 constant, so its RNEA agrees with the CasADi symbolic model again after #13799
- **Next step:** Shipped in PR #14228.

### DL-#14225 · Skills_Manager Reads Workflows Only From .agents/workflows (Mirrors Tools#5910, RM #2567); Vendor/Ud-Tools Bumped 0616724 -> 0B5a1eb so the Chat Child Copy Stays Byte-Identical to the Pin

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14225
- **Branch:** merged via #14239
- **PR:** #14239
- **Paths:** see #14239
- **Started:** 2026-11-02
- **Last verified:** 2026-11-02 (`d4ff4d3f`; collated from changes/14225-skills-manager-reads-workflows-only-from.md)
- **Summary:** skills_manager reads workflows only from .agents/workflows (mirrors Tools#5910, RM #2567); vendor/ud-tools bumped 0616724 -> 0b5a1eb so the chat child copy stays byte-identical to the pin
- **Next step:** Shipped in PR #14239.

### DL-#14214 · Conflict Disarm App Token Scoped to UpstreamDrift and Repository_Management With Contents Read, Pull-Requests Write and Issues Write

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14214
- **Branch:** merged via #14219
- **PR:** #14219
- **Paths:** see #14219
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`31539870`; collated from changes/14214-conflict-disarm-app-token-scoped-to-upst.md)
- **Summary:** Conflict Disarm App token scoped to UpstreamDrift and Repository_Management with contents read, pull-requests write and issues write
- **Next step:** Shipped in PR #14219.

### DL-#14216 · Development Log Reconciled: 51 Active Entries Whose PRs All Merged and Issues Closed Move to Shipped (75 to 24 Active), Unblocking the WIP Ceiling

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14216
- **Branch:** merged via #14217
- **PR:** #14217
- **Paths:** see #14217
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`c825eb47`; collated from changes/14216-development-log-reconciled-51-active-ent.md)
- **Summary:** Development log reconciled: 51 active entries whose PRs all merged and issues closed move to shipped (75 to 24 active), unblocking the WIP ceiling
- **Next step:** Shipped in PR #14217.

### DL-#14206 · Conflict Disarm Runs Its Job Inline (Reading RM's Conflict_Disarm Script via an App-Token Checkout) Instead of Calling an Internal Repository_Management Reusable Workflow, Which a Public Repo Cannot Do, so It No Longer Fails Every Run With 0 Jobs

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14206
- **Branch:** merged via #14207
- **PR:** #14207
- **Paths:** see #14207
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`8fe5f2b8`; collated from changes/14206-inline-conflict-disarm.md)
- **Summary:** Conflict Disarm runs its job inline (reading RM's conflict_disarm script via an App-token checkout) instead of calling an internal Repository_Management reusable workflow, which a public repo cannot do, so it no longer fails every run with 0 jobs
- **Next step:** Shipped in PR #14207.

### DL-#14201 · Guard the Removed Legacy .agent/ Tree and Fence the GAAI Ci-Watch-and-Fix Skill in Project Rules (RM #2567).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14201
- **Branch:** merged via #14211
- **PR:** #14211
- **Paths:** see #14211
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`8cf45c0a`; collated from changes/14201-agent-guard-fence.md)
- **Summary:** Guard the removed legacy .agent/ tree and fence the GAAI ci-watch-and-fix skill in project rules (RM #2567).
- **Next step:** Shipped in PR #14211.

### DL-#14193 · Putting_Dynamics: Resolve Roll Onset When Initial Slip Is Within One Step, Preventing Zero-Friction RK4 Chattering

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14193
- **Branch:** merged via #14208
- **PR:** #14208
- **Paths:** see #14208
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`3e644158`; collated from changes/14193-putting-dynamics-tiny-slip-roll-onset.md)
- **Summary:** putting_dynamics: resolve roll onset when initial slip is within one step, preventing zero-friction RK4 chattering
- **Next step:** Shipped in PR #14208.

### DL-#14180 · Docs: Root REVIEW.md With UpstreamDrift Review Rules (RM #2565 Rollout)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14180
- **Branch:** merged via #14182
- **PR:** #14182
- **Paths:** see #14182
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`b8b14446`; collated from changes/14180-review-md.md)
- **Summary:** docs: root REVIEW.md with UpstreamDrift review rules (RM #2565 rollout)
- **Next step:** Shipped in PR #14182.

### DL-#14178 · CI Apt Installer Bounds Each Apt-Get Attempt (APT_ATTEMPT_TIMEOUT, Default 420s) and Sets 30s Acquire Timeouts, so a Stalled Mirror Download Is Retried Instead of Holding a Merge-Queue Job to Its 150-Minute Timeout

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14178
- **Branch:** merged via #14179
- **PR:** #14179
- **Paths:** see #14179
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`06d8b53f`; collated from changes/14178-apt-attempt-timeout.md)
- **Summary:** CI apt installer bounds each apt-get attempt (APT_ATTEMPT_TIMEOUT, default 420s) and sets 30s acquire timeouts, so a stalled mirror download is retried instead of holding a merge-queue job to its 150-minute timeout
- **Next step:** Shipped in PR #14179.

### DL-#14172 · Add Putting Analytics Stochastic Green Sampler

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14172
- **Branch:** claude/putting-green-sampler-14172
- **PR:** #14192
- **Paths:** see #14192
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`01dc574b`; collated from changes/14172-add-putting-analytics-stochastic-green-s.md)
- **Summary:** Add Putting Analytics Stochastic Green Sampler
- **Next step:** Shipped in PR #14192.

### DL-#14164 · Docker Size Gates Tag Each Build per Run, Label It, and Always Remove the Run's Image Plus This Workflow's Dangling Images, so the Shared Runner Disk No Longer Fills

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14164
- **Branch:** merged via #14165
- **PR:** #14165
- **Paths:** see #14165
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`24281956`; collated from changes/14164-docker-size-gates-tag-each-build-per-run.md)
- **Summary:** Docker size gates tag each build per run, label it, and always remove the run's image plus this workflow's dangling images, so the shared runner disk no longer fills
- **Next step:** Shipped in PR #14165.

### DL-#14006 · HCM-5: Force and Mobility Matrices at Any Segment Intersection

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14006
- **Branch:** feat/14006-segment-intersection-mobility
- **PR:** #14205
- **Paths:** see #14205
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`3a575a27`; collated from changes/14006-segment-intersection-mobility.md)
- **Summary:** Implements declarative segment intersection extraction and point mobility evaluation across kinematics frames, adds condition number cutoff pruning to eliminate near-singular glyph clutter, and exposes --mobility-points, --max-mobility-points, and --mobility-condition-cutoff CLI flags in native viewer export.
- **Next step:** Shipped in PR #14205.

### DL-#14005 · Hand-Club Mobility Parity for Pinocchio: Label Kinematic Manipulability Analyzer, Route Native Viewer Export Through Pinocchio Capability Source, and Add TDD Parity Tests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14005
- **Branch:** merged via #14189
- **PR:** #14189
- **Paths:** see #14189
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`1946da6a`; collated from changes/14005-pinocchio-hand-mobility-parity.md)
- **Summary:** Hand-club mobility parity for Pinocchio: label kinematic manipulability analyzer, route native viewer export through Pinocchio capability source, and add TDD parity tests
- **Next step:** Shipped in PR #14189.

### DL-#13786 · Benchmark Harness Gains an Mjx-Warm-Start Arm Scored Only on Shared-Simulator Replay RMS With the L-BFGS-60 Iteration and Wall-Clock Budget; Stub-Only Until #13783-#13785

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13786
- **Branch:** merged via #14190
- **PR:** #14190
- **Paths:** see #14190
- **Started:** 2026-11-01
- **Last verified:** 2026-11-01 (`bb9214ba`; collated from changes/13786-warm-start-arm.md)
- **Summary:** Benchmark harness gains an mjx-warm-start arm scored only on shared-simulator replay RMS with the L-BFGS-60 iteration and wall-clock budget; stub-only until #13783-#13785
- **Next step:** Shipped in PR #14190.

### DL-#13560 · MuJoCo Grip Samples Read Zero-Slip Pairs Frictionless, Matching the Shared Law's Zero Friction at Rest; the Ten_Finger Hold Net_Moment_Com Xfail Is Removed

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13560
- **Branch:** merged via #14185
- **PR:** #14185, #14351
- **Paths:** see #14185
- **Started:** 2026-11-01
- **Last verified:** 2026-11-02 (`1377299a`; collated from changes/13560-opensim-net-moment-diagnosis.md)
- **Summary:** MuJoCo grip samples read zero-slip pairs frictionless, matching the shared law's zero friction at rest; the ten_finger hold net_moment_com xfail is removed
- **Next step:** Shipped in PR #14185.

### DL-#14141 · Characterize Putting_Dynamics Contracts With Invariant Tests and Postconditions (Putting V4 PR 1)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14141
- **Branch:** merged via #14145
- **PR:** #14145, #14149, #14155, #14159, #14160, #14163, #14166, #14167, #14184, #14187, #14202, #14191, #14209, #14210, #14213, #14215, #14221, #14286, #14249, #14277, #14278, #14284, #14285, #14224, #14226, #14229, #14234, #14240, #14241, #14243, #14244, #14246, #14248
- **Paths:** see #14145
- **Started:** 2026-10-31
- **Last verified:** 2026-11-02 (`a8b30b8f`; collated from changes/14141-putting-v4-step9a-mcp-tools.md)
- **Summary:** Characterize putting_dynamics contracts with invariant tests and postconditions (putting v4 PR 1)
- **Next step:** Shipped in PR #14145.

### DL-#13892 · Per-Hand Grip Radius: Contact_Topology(Grip_Geometry=...) Places Every Pad on GripGeometry.Radius_at(s) at Its Own Station, With per_Hand_Required_Squeeze_N (HRG-GD4 PR 1)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13892
- **Branch:** claude/13892-per-hand-grip-radius
- **PR:** #14140, #14144
- **Paths:** see #14140
- **Started:** 2026-10-31
- **Last verified:** 2026-10-31 (`c19da732`; collated from changes/13892-grip-size-sweep-cli-and-receipt-hand-len.md)
- **Summary:** Per-hand grip radius: contact_topology(grip_geometry=...) places every pad on GripGeometry.radius_at(s) at its own station, with per_hand_required_squeeze_n (HRG-GD4 PR 1)
- **Next step:** Shipped in PR #14144 (`c19da732`).

### DL-#14137 · Ci: OSV Monitor Uses the Runnable Osv-Scanner-Action Subpath; Findings No Longer Fail the Monitor Step

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14137
- **Branch:** merged via #14138
- **PR:** #14138
- **Paths:** see #14138
- **Started:** 2026-10-31
- **Last verified:** 2026-10-31 (`b82a8a74`; collated from changes/14137-ci-osv-monitor-uses-the-runnable-osv-sca.md)
- **Summary:** ci: OSV Monitor uses the runnable osv-scanner-action subpath; findings no longer fail the monitor step
- **Next step:** Shipped in PR #14138.

### DL-#14131 · Live Cross-Engine Address Knee Harness: Same Spec and Q Through Each Installed Engine's FK vs MuJoCo at 0.1 Deg.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14131
- **Branch:** merged via #14134
- **PR:** #14134
- **Paths:** see #14134
- **Started:** 2026-10-31
- **Last verified:** 2026-10-31 (`5a1cd349`; collated from changes/14131-address-knee-cross-engine-harness.md)
- **Summary:** Live cross-engine address knee harness: same spec and q through each installed engine's FK vs MuJoCo at 0.1 deg.
- **Next step:** Shipped in PR #14134.

### DL-#14132 · Install Spec Kit Antigravity Skills for All Speckit Gemini Command TOMLs.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14132
- **Branch:** merged via #14133
- **PR:** #14133
- **Paths:** see #14133
- **Started:** 2026-10-31
- **Last verified:** 2026-10-31 (`a021cbe8`; collated from changes/14132-install-speckit-agy-skills-layout.md)
- **Summary:** Install Spec Kit Antigravity skills for all speckit Gemini command TOMLs.
- **Next step:** Shipped in PR #14133.

### DL-#14122 · Install Pinocchio Runtime Library for Parity CI

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14122
- **Branch:** merged via #14126
- **PR:** #14126
- **Paths:** see #14126
- **Started:** 2026-10-30
- **Last verified:** 2026-10-30 (`e289527e`; collated from changes/14122-install-pinocchio-runtime-library-for-pa.md)
- **Summary:** Install Pinocchio runtime library for parity CI
- **Next step:** Shipped in PR #14126.

### DL-#12553 · Explain Unverified Torque-Capacity Proxies Without Changing Values or Verification Status.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12553
- **Branch:** merged via #14125
- **PR:** #14125
- **Paths:** see #14125
- **Started:** 2026-10-30
- **Last verified:** 2026-10-30 (`e66a0a20`; collated from changes/12553-explain-unverified-torque-capacity-proxi.md)
- **Summary:** Explain unverified torque-capacity proxies without changing values or verification status.
- **Next step:** Shipped in PR #14125.

### DL-#13998 · Numeric Hand-on-Club Acceptance Gates With Per-Engine Report and Render Contract

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13998
- **Branch:** agy-c/13998-hand-on-club-acceptance
- **PR:** #14117
- **Paths:** see #14117
- **Started:** 2026-10-30
- **Last verified:** 2026-10-30 (`e1bf681d`; collated from changes/13998-hand-on-club-acceptance-gates.md)
- **Summary:** Adds the shared numeric hand-on-club acceptance evaluator covering palm-to-grip gap (at most 3 mm), lead finger wrap (at least 240 deg), lead thumb and V azimuth prior conformity, trail-hand gap series (at most 5 mm), wrist-to-template attachment residual (at most 3 mm), per-hand wrench feasibility, and render contact-sheet specification across engines.
- **Next step:** Shipped in PR #14117.

### DL-#13996 · Attach Visual Grip Hands to Wrist Follower Frame and Scale Wrist-to-Grip Origins by Hand Length

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13996
- **Branch:** merged via #14119
- **PR:** #14119
- **Paths:** see #14119
- **Started:** 2026-10-30
- **Last verified:** 2026-10-30 (`a581aeb2`; collated from changes/13996-attach-visual-grip-hands-to-wrist-follow.md)
- **Summary:** Attach visual grip hands to wrist follower frame and scale wrist-to-grip origins by hand length
- **Next step:** Shipped in PR #14119.

### DL-#13886 · Add SAP-4B Drive-Mode Switch and Paired STM Sensitivity Receipt

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13886
- **Branch:** merged via #14112
- **PR:** #14112
- **Paths:** see #14112
- **Started:** 2026-10-30
- **Last verified:** 2026-10-30 (`bc186952`; collated from changes/13886-add-sap-4b-drive-mode-switch-and-paired.md)
- **Summary:** Add SAP-4b drive-mode switch and paired STM sensitivity receipt
- **Next step:** Shipped in PR #14112.

### DL-#14109 · Disarm Unit-Gate Hang Watchdog on Pytest_Unconfigure

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14109
- **Branch:** agy-c/14109-disarm-hang-watchdog
- **PR:** #14111
- **Paths:** see #14111
- **Started:** 2026-10-30
- **Last verified:** 2026-10-30 (`f6c2b3f8`; collated from changes/14109-disarm-hang-watchdog.md)
- **Summary:** Disarm the root conftest hang watchdog in pytest_unconfigure (ignore SIGUSR1, close the dump pipe, and terminate/reap the watchdog subprocess) and exit the watchdog immediately on stdin EOF so late SIGUSR1 signals during interpreter shutdown cannot kill pytest with exit code 138.
- **Next step:** Shipped in PR #14111.

### DL-#13190 · HRG-H2: Compliant Palm-Thumb Pad Reference Subsection

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13190
- **Branch:** docs/hrg-h2-palm-thumb-pad-reference-13190
- **PR:** #14110, #14379
- **Paths:** see #14110
- **Started:** 2026-10-30
- **Last verified:** 2026-11-03 (`2c7287cf`; collated from changes/13190-palm-thumb-pad-parity-across-mujoco-drak.md)
- **Summary:** Adds the HRG-H2 (hrg-h2-palm-thumb-pad-v1) compliant palm-thumb pad subsection to docs/research/human_realistic_golfer/human_realistic_golfer.tex documenting the separated_v2 sphere-capsule contact law, 2.0 mm hand-drift gate, preload sensitivity, timestep-halving impact convergence, and unit test verification.
- **Next step:** Shipped in PR #14110.

### DL-#14092 · Sim-Run: Stdio Simulation MCP Adapter

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14092
- **Branch:** feat/14092-sim-mcp
- **PR:** #14108
- **Paths:** see #14108
- **Started:** 2026-10-30
- **Last verified:** 2026-10-30 (`29d77f03`; collated from changes/14092-sim-mcp-adapter.md)
- **Summary:** Adds stdio Model Context Protocol (MCP) JSON-RPC 2.0 adapter exposing describe_engines, submit_run, get_run, and fetch_artifact tools over the headless simulation run service with Private Mode and auth checks at the service boundary.
- **Next step:** Shipped in PR #14108.

### DL-#14091 · Sim-Run: No-Qt Docker Smoke Test Running One Case via Docker/Ud-Sim

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14091
- **Branch:** feat/14091-no-qt-docker-smoke
- **PR:** #14107
- **Paths:** see #14107
- **Started:** 2026-10-30
- **Last verified:** 2026-10-30 (`819bc132`; collated from changes/14091-no-qt-docker-smoke-test.md)
- **Summary:** Adds CI smoke verification running a minimal case through the ud-sim CLI, asserting zero Qt imports in sys.modules and verified succeeded status with contract and falsification testing.
- **Next step:** Shipped in PR #14107.

### DL-#14090 · Sim-Run: Headless Ud-Sim Run JSON Batch CLI Over Run Service

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14090
- **Branch:** feat/14090-sim-run-cli
- **PR:** #14105
- **Paths:** see #14105
- **Started:** 2026-10-30
- **Last verified:** 2026-10-30 (`b17f1096`; collated from changes/14090-headless-ud-sim-run-cli.md)
- **Summary:** Implements ud-sim run batch CLI dispatching simulation cases through SimulationRunStore and SimulationWorkerPool, writing run records and output envelopes to configurable results root, returning non-zero exit on failure, with console script registration and CLI-API equivalence falsifying tests.
- **Next step:** Shipped in PR #14105.

### DL-#14089 · Sim-Run: Result Validation and Provenance Gate Before Succeeded

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14089
- **Branch:** feat/14089-sim-run-result-validation
- **PR:** #14103
- **Paths:** see #14103
- **Started:** 2026-10-30
- **Last verified:** 2026-10-30 (`8c2dc6fc`; collated from changes/14089-sim-run-validation-gate.md)
- **Summary:** Result envelope records repo SHA, engine version, and input sha256; numerical integrity gate recursively catches NaN and Inf in outputs; schema-invalid or NaN-emitting runs force failed state with structured diagnostics.
- **Next step:** Shipped in PR #14103.

### DL-#14088 · Sim-Run: Isolated Worker Subprocess With Timeout and Reaping

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14088
- **Branch:** feat/14088-sim-run-worker-process
- **PR:** #14100
- **Paths:** see #14100
- **Started:** 2026-10-30
- **Last verified:** 2026-10-30 (`cbe0e25b`; collated from changes/14088-sim-run-worker-subprocess.md)
- **Summary:** Runs execute in isolated worker subprocess outside event loop with wall-clock timeout and concurrency bound; worker reaped before cleanup; Linux rlimits applied and documented as verified only on Linux.
- **Next step:** Shipped in PR #14100.

### DL-#14087 · Sim-Run: Durable Run Store With Run IDs, Disk Quota and Visible Expiration

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14087
- **Branch:** feat/14087-sim-run-store
- **PR:** #14098
- **Paths:** see #14098
- **Started:** 2026-10-30
- **Last verified:** 2026-10-30 (`e1ee48de`; collated from changes/14087-sim-run-durable-store.md)
- **Summary:** Atomic per-run record under configurable results root; strict illegal lifecycle state transitions (e.g. succeeded->running raises); disk quota pruning marking pruned runs visibly expired.
- **Next step:** Shipped in PR #14098.

### DL-#14095 · Rotate SPEC.md Section 12 Change-Log Archive Wave 3

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14095
- **Branch:** fix/14095-spec-archive-wave-3
- **PR:** #14096
- **Paths:** see #14096
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`84d6430c`; collated from changes/14095-rotate-spec-md-wave3.md)
- **Summary:** Rotate the 369 oldest SPEC.md section 12 rows (dated up to 2026-10-24) verbatim into a third pinned block of docs/spec/CHANGELOG_ARCHIVE.md; SPEC.md 349,968 to 269,881 bytes under the unchanged 350,000-byte cap; the change-log gate reads every archived row block
- **Next step:** Shipped in PR #14096.

### DL-#14086 · Sim-Run: Run Request/Record/Result JSON Schemas Derived From API Contracts

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14086
- **Branch:** feat/14086-sim-run-schemas
- **PR:** #14093
- **Paths:** see #14093
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`acd9a864`; collated from changes/14086-sim-run-schemas.md)
- **Summary:** Versioned schemas for RunRequest, RunRecord, and ResultEnvelope exported from pydantic models
- **Next step:** Shipped in PR #14093.

### DL-#14052 · CAP-12 (#14038): Design-Manual Equations, HQ Renders and Docs for Capability Ellipsoids

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14052
- **Branch:** feat/14052-design-manual-capability-ellipsoids
- **PR:** #14084
- **Paths:** see #14084
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`08fde17c`; collated from changes/14052-design-manual-capability-ellipsoids.md)
- **Summary:** Design-manual Chapter 69 documenting interface capability ellipsoids, operational-space equations, Okabe-Ito styling, and governance integration
- **Next step:** Shipped in PR #14084.

### DL-#14050 · CAP-10 (#14038): Cross-Engine Parity Qualification for Interface Ellipsoids

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14050
- **Branch:** feat/14050-cross-engine-interface-ellipsoids-parity
- **PR:** #14083
- **Paths:** see #14083
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`7e316bbd`; collated from changes/14050-cross-engine-interface-ellipsoids-parity.md)
- **Summary:** Cross-engine parity test suite asserting matrices, ranks, eigenvalues (1%), axes (2°), and capacity ellipsoids across MuJoCo, Drake, and Pinocchio
- **Next step:** Shipped in PR #14083.

### DL-#14048 · CAP-8 (#14038): Drake Capability Adapter at the Hand-Club Interfaces

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14048
- **Branch:** feat/14048-drake-capability-adapter
- **PR:** #14075
- **Paths:** see #14075
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`751e22fe`; collated from changes/14048-drake-capability-adapter.md)
- **Summary:** Drake capability adapter emitting acceleration and force capacity ellipsoids per grip point
- **Next step:** Shipped in PR #14075.

### DL-#14045 · Capacity Wireframe Glyphs for MuJoCo, Meshcat, and Matplotlib Render Backends (CAP-5B)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14045
- **Branch:** merged via #14069
- **PR:** #14069
- **Paths:** see #14069
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`b7c1d88d`; collated from changes/14045-capacity-wireframe-glyphs-for-render-back.md)
- **Summary:** Capacity wireframe glyphs for MuJoCo, Meshcat, and matplotlib render backends (CAP-5b)
- **Next step:** Shipped in PR #14069.

### DL-#14042 · CAP-3 (#14038): Velocity Plumbing in Native-Viewer Exporter Path

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14042
- **Branch:** feat/14042-velocity-plumbing
- **PR:** #14078
- **Paths:** see #14078
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`eb7cc923`; collated from changes/14042-velocity-plumbing.md)
- **Summary:** Carry velocities v through SwingInput and expose interface bias force and point_jdot_v
- **Next step:** Shipped in PR #14078.

### DL-#14044 · CAP-5a (#14038): Ellipsoid Display Builder: Linear Percentile Scale, Legend Scale Bar, Clamp Styling, and Hue Rule

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14044
- **Branch:** feat/14044-ellipsoid-display-builder
- **PR:** #14073
- **Paths:** see #14073
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`6008e7b3`; collated from changes/14044-ellipsoid-display-builder.md)
- **Summary:** Linear radius scaling, legend scale bar formatting, dashed clamp styling, and Okabe-Ito yellow trail hue rule for capability ellipsoids
- **Next step:** Shipped in PR #14073.

### DL-#14049 · CAP-9 (#14038): Pinocchio Capability Adapter at the Hand-Club Interfaces

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14049
- **Branch:** feat/14049-pinocchio-capability-adapter
- **PR:** #14076
- **Paths:** see #14076
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`e0ede838`; collated from changes/14049-pinocchio-capability-adapter.md)
- **Summary:** Pinocchio capability adapter emitting acceleration and force capacity ellipsoids per grip point
- **Next step:** Shipped in PR #14076.

### DL-#13938 · Evaluate Glass Conductivity From Temperature Only; Drop Power-Density Heating in Default Model.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13938
- **Branch:** merged via #14065
- **PR:** #14065
- **Paths:** see #14065
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`0bbbe4e3`; collated from changes/13938-evaluate-glass-conductivity-from-tempera.md)
- **Summary:** Evaluate glass conductivity from temperature only; drop power-density heating in default model.
- **Next step:** Shipped in PR #14065.

### DL-#14043 · CAP-4 (#14038): MuJoCo Capability Adapter at the Hand-Club Interfaces

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14043
- **Branch:** feat/14043-mujoco-capability-adapter
- **PR:** #14071
- **Paths:** see #14071
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`a6bad706`; collated from changes/14043-mujoco-capability-adapter.md)
- **Summary:** MuJoCo capability adapter emitting acceleration and force capacity ellipsoids per grip point
- **Next step:** Shipped in PR #14071.

### DL-#14047 · CAP-7 (#14038): Dynamically Consistent Force-Capacity Core

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14047
- **Branch:** feat/14047-force-capacity-core
- **PR:** #14067
- **Paths:** see #14067
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`89f99a0a`; collated from changes/14047-cap-7-dynamically-consistent-force-capac.md)
- **Summary:** CAP-7: Dynamically consistent force-capacity core
- **Next step:** Shipped in PR #14067.

### DL-#14046 · CAP-6 (#14038): Exporter Option --Interface-Ellipsoid With a --Hand-Mobility Alias

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14046
- **Branch:** feat/14046-interface-ellipsoid-options
- **PR:** #14064
- **Paths:** see #14064
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`05491337`; collated from changes/14046-interface-ellipsoid-options.md)
- **Summary:** CAP-6: Exporter option --interface-ellipsoid with a --hand-mobility alias
- **Next step:** Shipped in PR #14064.

### DL-#14039 · Add Interface-Ellipsoid Kind/Provenance Contracts and Engine Parity Ledger (CAP-0)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14039
- **Branch:** merged via #14055
- **PR:** #14055
- **Paths:** see #14055
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`494cbba4`; collated from changes/14039-add-interface-ellipsoid-kind-provenance.md)
- **Summary:** Add interface-ellipsoid kind/provenance contracts and engine parity ledger (CAP-0)
- **Next step:** Shipped in PR #14055.

### DL-#14027 · Drake GUI No Longer Crashes After MeshCat Starts and Opens the V2 Meshed Golfer With a Drake-vs-MuJoCo Parity Check

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14027
- **Branch:** merged via #14028
- **PR:** #14028
- **Paths:** see #14028
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`1a7cee00`; collated from changes/14027-drake-gui-no-longer-crashes-after-meshca.md)
- **Summary:** Drake GUI no longer crashes after MeshCat starts and opens the v2 meshed golfer with a Drake-vs-MuJoCo parity check
- **Next step:** Shipped in PR #14028.

### DL-#14003 · EllipsoidGlyph + MuJoCo Ellipsoid Drawing for Native_Viewer_Export

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14003
- **Branch:** merged via #14022
- **PR:** #14022
- **Paths:** see #14022
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`30684dd5`; collated from changes/14003-ellipsoidglyph-mujoco-ellipsoid-drawing.md)
- **Summary:** EllipsoidGlyph + MuJoCo ellipsoid drawing for native_viewer_export
- **Next step:** Shipped in PR #14022.

### DL-#13779 · Fix(Motion-Matching): L-BFGS-B Stop Reason Is Fail-Closed (ABNORMAL Exits Are Error, Not Max_Iterations) and a Non-Finite Trial No Longer Ends the Run or Freezes Best-Iterate Tracking

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13779
- **Branch:** merged via #14013
- **PR:** #14013
- **Paths:** see #14013
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`939ec1d6`; collated from changes/13779-fix-motion-matching-l-bfgs-b-stop-reason.md)
- **Summary:** fix(motion-matching): L-BFGS-B stop reason is fail-closed (ABNORMAL exits are error, not max_iterations) and a non-finite trial no longer ends the run or freezes best-iterate tracking
- **Next step:** Shipped in PR #14013.

### DL-#13795 · Correct Cross_Engine_V1 Dependency Status: The CC-7/9/10 Draft PRs Were Closed Unmerged, No Comparison Has Been Run

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13795
- **Branch:** merged via #14017
- **PR:** #14017, #14018, #14162
- **Paths:** see #14017
- **Started:** 2026-10-29
- **Last verified:** 2026-11-01 (`002d5c3a`; collated from changes/13795-unified-perturbation-receipt.md)
- **Summary:** Correct cross_engine_v1 dependency status: the CC-7/9/10 draft PRs were closed unmerged, no comparison has been run
- **Next step:** Shipped in PR #14017.

### DL-#12684 · R4 Cross-Engine Address Knee Flexion Check Over the Committed Receipts (Scripts/address_knee_engine_check.py): Per-Engine Disagreement Over 0.1 Deg, Unavailable Engines With Reasons, Fail-Closed on Malformed Receipts

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12684
- **Branch:** merged via #14014
- **PR:** #14014
- **Paths:** see #14014
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`8aee5888`; collated from changes/12684-r4-cross-engine-address-knee-flexion-che.md)
- **Summary:** R4 cross-engine address knee flexion check over the committed receipts (scripts/address_knee_engine_check.py): per-engine disagreement over 0.1 deg, unavailable engines with reasons, fail-closed on malformed receipts
- **Next step:** Shipped in PR #14014.

### DL-#14040 · CAP-1: Acceleration_Capacity Ellipsoid and Drift_Acceleration in Operational_Space

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14040
- **Branch:** merged via #14056
- **PR:** #14056
- **Paths:** see #14056
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`bc8dad19`; collated from changes/14040-cap-1-acceleration-capacity-ellipsoid-an.md)
- **Summary:** CAP-1: acceleration_capacity ellipsoid and drift_acceleration in operational_space
- **Next step:** Shipped in PR #14056.

### DL-#14041 · CAP-2: Torque-Limit Provider Adapting CapacityVector to (B, Tau_Lo, Tau_Hi) With Provenance

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14041
- **Branch:** merged via #14054
- **PR:** #14054
- **Paths:** see #14054
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`cfef12fb`; collated from changes/14041-cap-2-torque-limit-provider-adapting-cap.md)
- **Summary:** CAP-2: torque-limit provider adapting CapacityVector to (B, tau_lo, tau_hi) with provenance
- **Next step:** Shipped in PR #14054.

### DL-#13785 · ML-WS-8: ProposalInitializer Protocol and --Neural-Mode Preview Warm-Start Hook Into MJX Knot Optimiser

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13785
- **Branch:** merged via #14034
- **PR:** #14034
- **Paths:** see #14034
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`a0ee91aa`; collated from changes/13785-ml-ws-8-neural-mode-preview-warm-start.md)
- **Summary:** ML-WS-8: ProposalInitializer protocol and --neural-mode preview warm-start hook into MJX knot optimiser
- **Next step:** Shipped in PR #14034.

### DL-#14031 · Bump Tornado 6.5.10 to 6.6 for CVE-2026-116577, CVE-2026-115927 and CVE-2026-116578, and Msgpack 1.2.1 to 1.2.3 in the Docker Images for CVE-2026-114613 (No Audit Waiver)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14031
- **Branch:** merged via #14032
- **PR:** #14032
- **Paths:** see #14032
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`292f8195`; collated from changes/14031-bump-tornado-6-5-10-to-6-6-for-cve-2026.md)
- **Summary:** Bump tornado 6.5.10 to 6.6 for CVE-2026-116577, CVE-2026-115927 and CVE-2026-116578, and msgpack 1.2.1 to 1.2.3 in the Docker images for CVE-2026-114613 (no audit waiver)
- **Next step:** Shipped in PR #14032.

### DL-#14010 · Catch TypeError and Probe Availability in Cross-Engine Builder and Degrade Uninitialized Engines to Stub

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14010
- **Branch:** merged via #14029
- **PR:** #14029
- **Paths:** see #14029
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`891d03fb`; collated from changes/14010-cross-engine-probe-unusable-engines.md)
- **Summary:** Catch TypeError and probe availability in cross-engine builder and degrade uninitialized engines to stub
- **Next step:** Shipped in PR #14029.

### DL-#14004 · MuJoCo Adapter for Per-Hand Constrained Mobility W = J A J^T and Effective Masses at Each Grip Point (HCM-3A)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14004
- **Branch:** merged via #14026
- **PR:** #14026, #14114
- **Paths:** see #14026
- **Started:** 2026-10-29
- **Last verified:** 2026-10-30 (`e68db5ea`; collated from changes/14004-per-hand-force-and-mobility-ellipsoids-a.md)
- **Summary:** MuJoCo adapter for per-hand constrained mobility W = J A J^T and effective masses at each grip point (HCM-3a)
- **Next step:** Shipped in PR #14026.

### DL-#14002 · Add Operational-Space Mobility Matrix, Velocity/Force Ellipsoids and Spatial Transport

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14002
- **Branch:** merged via #14020
- **PR:** #14020
- **Paths:** see #14020
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`42ea10f3`; collated from changes/14002-add-operational-space-mobility-matrix-ve.md)
- **Summary:** Add operational-space mobility matrix, velocity/force ellipsoids and spatial transport
- **Next step:** Shipped in PR #14020.

### DL-#13995 · Thread One ResolvedHandSpec (Grip Style, Strength, Handedness, Hand Length) Through Every Engine Emitter

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13995
- **Branch:** merged via #14021
- **PR:** #14021
- **Paths:** see #14021
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`78189f73`; collated from changes/13995-thread-one-resolvedhandspec-grip-style-s.md)
- **Summary:** Thread one ResolvedHandSpec (grip style, strength, handedness, hand length) through every engine emitter
- **Next step:** Shipped in PR #14021.

### DL-#13994 · Hands Close-Up Camera Tracks the Kinematic Grip Midpoint From the Club Pose, With the Overlay on or Off

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13994
- **Branch:** merged via #14001
- **PR:** #14001
- **Paths:** see #14001
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`77a17e4b`; collated from changes/13994-hands-close-up-camera-tracks-the-kinemat.md)
- **Summary:** Hands close-up camera tracks the kinematic grip midpoint from the club pose, with the overlay on or off
- **Next step:** Shipped in PR #14001.

### DL-#13993 · Taper the Shared Visual Forearm 20 Percent Toward the Wrist (Appearance Only; Inertias Unchanged)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13993
- **Branch:** merged via #14008
- **PR:** #14008
- **Paths:** see #14008
- **Started:** 2026-10-29
- **Last verified:** 2026-10-29 (`831b01a9`; collated from changes/13993-taper-the-shared-visual-forearm-20-perce.md)
- **Summary:** Taper the shared visual forearm 20 percent toward the wrist (appearance only; inertias unchanged)
- **Next step:** Shipped in PR #14008.

### DL-#13992 · Graft Pelvis_Frame_From_Spec Accepts HIP_MIRROR-Conjugated Hips

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13992
- **Branch:** merged via #13999
- **PR:** #13999
- **Paths:** see #13999
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`05ba1f99`; collated from changes/13992-graft-pelvis-frame-from-spec-accepts-hip.md)
- **Summary:** Graft pelvis_frame_from_spec accepts HIP_MIRROR-conjugated hips
- **Next step:** Shipped in PR #13999.

### DL-#13793 · Add Sensitivity Fields (FTLE, Sigma-Max, Agreement Horizon) to Pendulum Parity Receipts (SAP-4)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13793
- **Branch:** merged via #13988
- **PR:** #13988
- **Paths:** see #13988
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`fd233d9c`; collated from changes/13793-sensitivity-receipts.md)
- **Summary:** Add sensitivity fields (FTLE, sigma-max, agreement horizon) to pendulum parity receipts (SAP-4)
- **Next step:** Shipped in PR #13988.

### DL-#13945 · Pin Pytest <9.2.0 Across Repository Dependencies and CI Workflows

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13945
- **Branch:** merged via #13945
- **PR:** #13945
- **Paths:** see #13945
- **Started:** 2026-10-28
- **Last verified:** 2026-10-28 (`9aba9431`; collated from changes/13945-pin-pytest-below-9-2-0.md)
- **Summary:** Pin pytest <9.2.0 across repository dependencies and CI workflows
- **Next step:** Shipped in PR #13945.

### DL-#13852 · Add WSCG Masked-JSON Parity Test Scaffold for Budget-Split PRs.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13852
- **Branch:** merged via #13860
- **PR:** #13860, #13937, #13942, #13989
- **Paths:** see #13860
- **Started:** 2026-10-26
- **Last verified:** 2026-10-28 (`fed67279`; collated from changes/13852-wscg-case-split-3.md)
- **Summary:** Add WSCG masked-JSON parity test scaffold for budget-split PRs.
- **Next step:** Shipped in PR #13860.

### DL-#13045 · Feat(Lifting): Barbell Visual Geometry and Lift Video Camera Framing

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13045
- **Branch:** merged via #13823
- **PR:** #13823, #13845, #13835, #13833, #13832, #13944
- **Paths:** see #13823
- **Started:** 2026-10-26
- **Last verified:** 2026-10-28 (`01da1de0`; collated from changes/13045-osl4-contact-sheet-test.md)
- **Summary:** feat(lifting): barbell visual geometry and lift video camera framing
- **Next step:** Shipped in PR #13823.

### DL-#13176 · Qualify HRG Full-Body-V2 Mass and Gravity in Drake

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13176
- **Branch:** merged via #13820
- **PR:** #13820, #14015, #14127, #14357, #14363, #14369, #14372, #14375
- **Paths:** see #13820
- **Started:** 2026-10-25
- **Last verified:** 2026-11-02 (`0fd41583`; collated from changes/13176-full-body-v2-closeout.md)
- **Summary:** Qualify HRG full-body-v2 mass and gravity in Drake
- **Next step:** Shipped in PR #13820.

### DL-#13043 · Joint Reaction Loads for Deadlift, Snatch, and Clean and Jerk

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13043
- **Branch:** merged via #13521
- **PR:** #13521, #14183
- **Paths:** see #13521
- **Started:** 2026-10-25
- **Last verified:** 2026-11-01 (`fa421d71`; collated from changes/13043-bench-joint-reactions.md)
- **Summary:** joint reaction loads for deadlift, snatch, and clean and jerk
- **Next step:** Shipped in PR #13521.

### DL-#13175 · HRG-E4 Slice: Desktop and Web Anatomical Event Table View With a Shared Golden Fixture and a Feature_Parity Gap Entry (Web Fetch Awaits #13538)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13175
- **Branch:** merged via #13543
- **PR:** #13543, #14012, #14113, #14128, #14157
- **Paths:** see #13543
- **Started:** 2026-10-25
- **Last verified:** 2026-11-01 (`b3d2d1ed`; collated from changes/13175-desktop-event-table-tile.md)
- **Summary:** HRG-E4 slice: desktop and web anatomical event table view with a shared golden fixture and a feature_parity gap entry (web fetch awaits #13538)
- **Next step:** Shipped in PR #13543.

### DL-#12567 · Add a Zero-Side Consistency Shim for HARD_ZERO_TRAIL Allocation and Reject Algebraically Inconsistent Zero-Side Systems.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12567
- **Branch:** merged via #13606
- **PR:** #13606, #14016
- **Paths:** see #13606
- **Started:** 2026-10-25
- **Last verified:** 2026-10-29 (`2639f4cd`; collated from changes/12567-nsx24-frame-seam.md)
- **Summary:** Add a zero-side consistency shim for HARD_ZERO_TRAIL allocation and reject algebraically inconsistent zero-side systems.
- **Next step:** Shipped in PR #13606.

### DL-#13579 · Record #13579 Closure Evidence (Docs/Operations/Evidence/shipped-even-issues-20261024.md)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13579
- **Branch:** cursor/issue-closure-receipts-66dc
- **PR:** #13685, #13986
- **Paths:** see #13685
- **Started:** 2026-10-24
- **Last verified:** 2026-10-28 (`dd75bc61`; collated from changes/13579-engine-docker-cache-contract.md)
- **Summary:** Record #13579 closure evidence (docs/operations/evidence/shipped-even-issues-20261024.md)
- **Next step:** Shipped in PR #13685.

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
