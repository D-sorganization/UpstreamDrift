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

### DL-#14906 · Correct SHOW-A10 Evidence Provenance Manifest and Summary Attribution Without Regenerating Artifacts

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14906
- **Branch:** agy/ct/14906-show-a10-provenance-correction
- **PR:** #14920
- **Paths:** see #14920
- **Started:** 2026-11-09
- **Last verified:** 2026-11-09 (`94f794b0`; collated from changes/14906-show-a10-provenance-correction.md)
- **Summary:** Correct SHOW-A10 evidence provenance manifest and summary attribution without regenerating artifacts
- **Next step:** Merge the A10 provenance correction slice while keeping epic #14906 open.

### DL-#14397 · Canonical Float Rounding Before Hashing for Ubuntu 26.04

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14397
- **Branch:** agy/14397-ubuntu-2604-hash-drift
- **PR:** #14899
- **Paths:** see #14899
- **Started:** 2026-11-08
- **Last verified:** 2026-11-08 (`604503f6`; collated from changes/14397-ubuntu-2604-hash-drift.md)
- **Summary:** physics_spec_sha256 and the saved-comparison receipts compare floats at a canonical precision (12 significant digits, 1e-12 absolute) so BLAS/libm last-bit drift cannot move them (owner option A)
- **Next step:** Merge the PR.

### DL-#14896 · CI Checkout Depth Qualification and Transfer Budget Policy

- **State:** in_review
- **Owner:** fleet-orchestrator
- **Issue:** #14896
- **Branch:** agy/14896-ci-checkout-depth-contracts
- **PR:** #14915, #14916
- **Paths:** scripts/ci/qualify_checkout_depth.py, tests/ci/test_qualify_checkout_depth.py, docs/development/evidence/CI_CHECKOUT_DEPTH_AND_TRANSFER_BUDGET_POLICY.md
- **Started:** 2026-11-08
- **Last verified:** 2026-11-08 (`0ec24365`; collated from changes/14896-ci-checkout-depth-contracts.md)
- **Summary:** Qualify job-specific CI checkout depth, transfer bandwidth budgets, and sparse-state regression contracts
- **Next step:** Open PR from agy/14896-ci-checkout-depth-contracts; arm auto-merge.

### DL-#14895 · Source Asset Selective LFS Policy and Inventory Contracts

- **State:** in_review
- **Owner:** fleet-orchestrator
- **Issue:** #14895
- **Branch:** agy/14895-source-asset-lfs-policy
- **PR:** #14914
- **Paths:** scripts/assets/source_asset_inventory.py, tests/unit/assets/test_source_asset_inventory.py, docs/development/evidence/SOURCE_ASSET_LFS_POLICY.md
- **Started:** 2026-11-08
- **Last verified:** 2026-11-08 (`c6375e53`; collated from changes/14895-source-asset-lfs-policy.md)
- **Summary:** Selective Git LFS policy, source asset inventory contracts, and fail-closed hydration verification
- **Next step:** Open PR from agy/14895-source-asset-lfs-policy; arm auto-merge.

### DL-#14894 · Research Artifacts: Checksum-Bound Storage Contracts

- **State:** shipped
- **Owner:** fleet-orchestrator
- **Issue:** #14894
- **Branch:** agy/14894-research-artifact-storage
- **PR:** #14911, #14912
- **Paths:** scripts/research/shot_pattern_analysis/fetch_release_assets.py, tests/unit/research/test_fetch_release_assets.py, docs/development/evidence/RESEARCH_ARTIFACT_STORAGE_POLICY.md
- **Started:** 2026-11-08
- **Last verified:** 2026-11-08 (`781dcc0b`; collated from changes/14894-research-artifact-storage-contracts.md)
- **Summary:** Research artifacts: checksum-bound durable storage, fail-closed offline/integrity verification, and evidence preservation policy
- **Next step:** Shipped.

### DL-#13158 · OSL-19: Grip Closure and OpenSim Reach Receipt

- **State:** in_review
- **Owner:** fleet-orchestrator
- **Issue:** #13158
- **Branch:** agy/13158-osl19-grip-closure-reach
- **PR:** #14802, #14907
- **Paths:** src/shared/python/osl_scoping/osl19.py, tests/fixtures/lifting/osl_scoping/osl19_scoping_v1.json, tests/unit/lifting/osl_scoping/test_osl19_scoping_contract.py
- **Started:** 2026-11-06
- **Last verified:** 2026-11-08 (`9e90db40`; collated from changes/13158-osl-19-grip-closure-reach.md)
- **Summary:** OSL-19: grip closure contracts and OpenSim reach hook for back-squat bundles
- **Next step:** Open PR from agy/13158-osl19-grip-closure-reach; resolve right-hand reach solve under cited arm ranges.

### DL-#13131 · OSL-17: Back-Squat Bar Rack and Rest Window

- **State:** in_progress
- **Owner:** fleet-orchestrator
- **Issue:** #13131
- **Branch:** agy/13131-osl17-bar-rack-rest
- **PR:** #14802, #14903
- **Paths:** src/shared/python/lifting/back_rack.py, src/shared/python/osl_scoping/osl17.py, tests/unit/lifting/test_back_rack.py, tests/unit/lifting/osl_scoping/test_osl17_scoping_contract.py, docs/development/lifting/evidence/opensim_ik_id/receipts.json
- **Started:** 2026-11-06
- **Last verified:** 2026-11-08 (`ccd72956`; collated from changes/13131-osl-17-back-squat-bar-rack-and-rest.md)
- **Summary:** OSL-17: put back-squat bar on BackTop high-bar rack, verify lead-in settle and whole-window OpenSim Hicks and marker gates
- **Next step:** Open PR from agy/13131-osl17-bar-rack-rest; resolve floating-base regulation for re-track bound.

### DL-#13093 · OSL-15: Regenerate Source Back Squat With Bar Load

- **State:** in_review
- **Owner:** fleet-orchestrator
- **Issue:** #13093
- **Branch:** agy/13093-osl15-load-balance-squat
- **PR:** #14904
- **Paths:** src/shared/python/osl_scoping/osl15.py, tests/fixtures/lifting/osl_scoping/osl15_scoping_v1.json, tests/unit/lifting/osl_scoping/test_osl15_scoping_contract.py
- **Started:** 2026-11-06
- **Last verified:** 2026-11-08 (`7b629286`; collated from changes/13093-osl-15-load-balance-squat.md)
- **Summary:** OSL-15: verify bar-load balance in IK provenance (combined CoM constrained over mid-foot) on back-squat barbell and source bundles
- **Next step:** Merge PR #14904; regenerate deadlift/clean bundles to clear xfails.

### DL-#11746 · LIFT-6: OpenSim Lift Composition and Static Hold Contracts

- **State:** in_review
- **Owner:** fleet-orchestrator
- **Issue:** #11746
- **Branch:** agy/11746-lift6-composition-static-hold
- **PR:** #14805, #14910
- **Paths:** src/shared/python/lift_scoping/lift6.py, tests/fixtures/lifting/lift_scoping/lift6_epic_v1.json, tests/unit/lifting/lift_scoping/test_lift6_epic_contract.py
- **Started:** 2026-11-06
- **Last verified:** 2026-11-08 (`c3813358`; collated from changes/11746-lift-6-composition-static-hold.md)
- **Summary:** LIFT-6: OpenSim full-body lift composition, 80-muscle count, static hold, and kinetics qualification contracts
- **Next step:** Open PR from agy/11746-lift6-composition-static-hold; complete remaining host evidence gates.

### DL-#11745 · LIFT-5: Four-Engine Bar Path Parity Contract

- **State:** in_review
- **Owner:** fleet-orchestrator
- **Issue:** #11745
- **Branch:** agy/11745-lift5-four-engine-bar-path-parity
- **PR:** #14805, #14877, #14909
- **Paths:** src/shared/python/lift_scoping/lift5.py, tests/fixtures/lifting/lift_scoping/lift5_epic_v1.json, tests/unit/lifting/lift_scoping/test_lift5_epic_contract.py
- **Started:** 2026-11-06
- **Last verified:** 2026-11-08 (`960d2084`; collated from changes/11745-lift-5-four-engine-bar-path-parity.md)
- **Summary:** LIFT-5: four-engine bar path parity contract and tests covering MuJoCo, Drake, Pinocchio, and OpenSim
- **Next step:** Open PR from agy/11745-lift5-four-engine-bar-path-parity; optional parity lanes for remaining lifts.

### DL-#13991 · Docs: ADR-0060 OpenSim Scaling and Fit Plan (#13991)

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13991
- **Branch:** cursor/opensim-scaling-panel-plan-13991-94df
- **PR:** #14803
- **Paths:** see #14803
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`320872fc`; collated from changes/13991-docs-adr-0060-opensim-scaling-and-fit-pl.md)
- **Summary:** docs: ADR-0060 OpenSim scaling and fit plan (#13991)
- **Next step:** Panel records dissent on issue; promote ADR to Accepted when agreed.

### DL-#14419 · Avoid Merge_Group Changed-Paths Checkout Hang and Sync Admission Publisher

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14419
- **Branch:** fix/14419-merge-group-checkout-depth
- **PR:** #14420
- **Paths:** see #14420
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`d100e477`; collated from changes/14419-avoid-merge-group-changed-paths-checkout.md)
- **Summary:** Avoid merge_group changed-paths checkout hang and sync admission publisher
- **Next step:** Merge the PR.

### DL-#14417 · Sync the Admission Publisher Workflow and Provenance Digest With the Canonical Provider

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14417
- **Branch:** codex/14417-admission-publisher-sync
- **PR:** #14418
- **Paths:** see #14418
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`982d34d5`; collated from changes/14417-sync-the-admission-publisher-workflow-an.md)
- **Summary:** Sync the admission publisher workflow and provenance digest with the canonical provider
- **Next step:** Merge the PR after required checks pass.

### DL-#14429 · Avoid OpenSim Classify Checkout Hang With Shallow Fetch

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14429
- **Branch:** agy/14429-opensim-classify-shallow-checkout
- **PR:** #14430
- **Paths:** see #14430
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`a9657f01`; collated from changes/14429-avoid-opensim-classify-checkout-hang-wit.md)
- **Summary:** Avoid OpenSim classify checkout hang with shallow fetch
- **Next step:** Merge the PR.

### DL-#14535 · Size-Aware Development-Log Archiver: Archive Oldest Terminal Entries When the Log Nears the Byte Cap

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #14535
- **Branch:** claude-ct/14535-devlog-size-trigger
- **PR:** #14592
- **Paths:** see #14592
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`8c6f82c1`; collated from changes/14535-size-aware-development-log-archiver-arch.md)
- **Summary:** Size-aware development-log archiver: archive oldest terminal entries when the log nears the byte cap
- **Next step:** Merge the PR.

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
- **PR:** #14414, #14403
- **Paths:** see #14414
- **Started:** 2026-11-03
- **Last verified:** 2026-11-05 (`b281ba3e`; collated from changes/14274-root-dockerignore-secret-contract.md)
- **Summary:** Config/secret separation and upgrade/rollback guarantees
- **Next step:** Complete independent review and open the scoped CI contract PR.

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
- **PR:** #14198, #14232, #14827
- **Paths:** see #14198
- **Started:** 2026-11-01
- **Last verified:** 2026-11-07 (`4ef29833`; collated from changes/14176-putting-make-rate-calibration.md)
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

### DL-#14938 · \ Marker IK Exposes Convergence Status,Reports NO_MARKERS and NaN RMS for Empty Frames,and Routes LM With Finite Bounds to TRF\

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14938
- **Branch:** agy/14938-marker-ik-convergence
- **PR:** #14974
- **Paths:** see #14974
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`bb977b68`; collated from changes/14938-marker-ik-convergence.md)
- **Summary:** \ Marker IK exposes convergence status,reports NO_MARKERS and NaN RMS for empty frames,and routes LM with finite bounds to TRF\
- **Next step:** Shipped in PR #14974.

### DL-#14946 · Make Coverage Truth Explicit in CI Summary and Remove Dead Cov-Fail-Under Flags

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14946
- **Branch:** merged via #14972
- **PR:** #14972
- **Paths:** see #14972
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`57b4ffa7`; collated from changes/14946-coverage-truth-in-summary.md)
- **Summary:** Make coverage truth explicit in CI summary and remove dead cov-fail-under flags
- **Next step:** Shipped in PR #14972.

### DL-#14950 · Simulation WS Error Frame Handling and isPaused Reset on Stop (Part Of #14950).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14950
- **Branch:** merged via #14958
- **PR:** #14958
- **Paths:** see #14958
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`8053f6e4`; collated from changes/14950-simulation-ws-error-frame-handling-and-i.md)
- **Summary:** Simulation WS error frame handling and isPaused reset on stop (Part of #14950).
- **Next step:** Shipped in PR #14958.

### DL-#14945 · Make Architecture and File-Size Budget Expiry Evaluation Deterministic

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14945
- **Branch:** merged via #14954
- **PR:** #14954
- **Paths:** see #14954
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`268f76ed`; collated from changes/14945-deterministic-budget-expiry.md)
- **Summary:** Make architecture and file-size budget expiry evaluation deterministic
- **Next step:** Shipped in PR #14954.

### DL-#14944 · ZTCF: Guard Multi-Body Misuse of Compute_Ztcf_Forces; Double-Pendulum Regression (#14944, Part Of #14934)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14944
- **Branch:** merged via #14953
- **PR:** #14953
- **Paths:** see #14953
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`7625298e`; collated from changes/14944-ztcf-multibody-guard.md)
- **Summary:** ZTCF: guard multi-body misuse of compute_ztcf_forces; double-pendulum regression (#14944, Part of #14934)
- **Next step:** Shipped in PR #14953.

### DL-#14227 · Document CI Optional Stack Slice 1 (Pinocchio Gravity) and Add Runbook Contract Tests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14227
- **Branch:** merged via #14923
- **PR:** #14923
- **Paths:** see #14923
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`e946eb78`; collated from changes/14227-document-ci-optional-stack-slice-1-pinoc.md)
- **Summary:** Document CI Optional Stack slice 1 (Pinocchio gravity) and add runbook contract tests
- **Next step:** Shipped in PR #14923.

### DL-#13604 · NSX Salvage Tracker: Unit Receipts for Merged NSX-22/NSX-24 Rows on Main (#13604)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13604
- **Branch:** merged via #14930
- **PR:** #14930
- **Paths:** see #14930
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`2d4fde75`; collated from changes/13604-nsx-salvage-tracker-receipts.md)
- **Summary:** NSX salvage tracker: unit receipts for merged NSX-22/NSX-24 rows on main (#13604)
- **Next step:** Shipped in PR #14930.

### DL-#14860 · Redact Fleet Host and Home Literals Batch 4 Recovery

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14860
- **Branch:** merged via #14933
- **PR:** #14933
- **Paths:** see #14933
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`1cc97639`; collated from changes/14860-host-redact-04.md)
- **Summary:** Redact fleet host and home literals batch 4 recovery
- **Next step:** Shipped in PR #14933.

### DL-#14820 · UD Docs and CI Contract Tests Reaffirm PR Size Is Advisory Only (Dieter 2026-11-06)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14820
- **Branch:** merged via #14926
- **PR:** #14926
- **Paths:** see #14926
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`ee388e1a`; collated from changes/14820-ud-pr-size-advisory-review-contract.md)
- **Summary:** UD docs and CI contract tests reaffirm PR size is advisory only (Dieter 2026-11-06)
- **Next step:** Shipped in PR #14926.

### DL-#14928 · Align Realtime Shadow-Ledger Sunset_Date With Q4 Baseline and Validate Against UTC Date (#14928, #12046)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14928
- **Branch:** merged via #14931
- **PR:** #14931
- **Paths:** see #14931
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`a3e2f0f1`; collated from changes/14928-realtime-shadow-sunset.md)
- **Summary:** Align realtime shadow-ledger sunset_date with Q4 baseline and validate against UTC date (#14928, #12046)
- **Next step:** Shipped in PR #14931.

### DL-#12878 · Explorer Contract: Engine-Tagged Playback Archives and Per-Engine MP4 Distinctness Test (#12878 Workstream C)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12878
- **Branch:** merged via #14921
- **PR:** #14921
- **Paths:** see #14921
- **Started:** 2026-11-10
- **Last verified:** 2026-11-10 (`f6c43790`; collated from changes/12878-explorer-contract-engine-tagged-playback.md)
- **Summary:** Explorer contract: engine-tagged playback archives and per-engine MP4 distinctness test (#12878 workstream C)
- **Next step:** Shipped in PR #14921.

### DL-#14905 · Preserve #14733 Proposal Durable Change-History Evidence Across Normal Change-Fragment Collation and Log Archival

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14905
- **Branch:** merged via #14908
- **PR:** #14908
- **Paths:** see #14908
- **Started:** 2026-11-08
- **Last verified:** 2026-11-08 (`a89dd7dd`; collated from changes/14905-preserve-proposal-collation-evidence.md)
- **Summary:** Preserve #14733 proposal durable change-history evidence across normal change-fragment collation and log archival
- **Next step:** Shipped in PR #14908.

### DL-#14897 · Point Driver Motion-Matching Smoke to Canonical Capture-Fit Spec (Full_Body_Spec_Anthro_Driver_Capture.Json) and Align Anthro_Driver Receipt Base_Spec_File Metadata

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14897
- **Branch:** merged via #14901
- **PR:** #14901
- **Paths:** see #14901
- **Started:** 2026-11-08
- **Last verified:** 2026-11-08 (`1c3126cd`; collated from changes/14897-motion-matching-smoke-driver-spec.md)
- **Summary:** Point driver motion-matching smoke to canonical capture-fit spec (full_body_spec_anthro_driver_capture.json) and align anthro_driver receipt base_spec_file metadata
- **Next step:** Shipped in PR #14901.

### DL-#14795 · SHOW-A19: Re-Characterise Wrist and Scapula ROM From Evidence and Evaluate Option (C) Address Reachability

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14795
- **Branch:** merged via #14890
- **PR:** #14890
- **Paths:** see #14890
- **Started:** 2026-11-08
- **Last verified:** 2026-11-08 (`632c36c9`; collated from changes/14795-show-a19-wrist-scapula-rom.md)
- **Summary:** SHOW-A19: re-characterise wrist and scapula ROM from evidence and evaluate option (c) address reachability
- **Next step:** Shipped in PR #14890.

### DL-#14733 · Proposal (Write-Up Only): #13560 OpenSim Net-Moment Gap - Recorder Inconsistency Finding, Conditioning Math, Options

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14733
- **Branch:** merged via #14888
- **PR:** #14888
- **Paths:** see #14888
- **Started:** 2026-11-08
- **Last verified:** 2026-11-08 (`36155e93`; collated from changes/14733-opensim-net-moment-gap-proposal.md)
- **Summary:** Proposal (write-up only): #13560 OpenSim net-moment gap - recorder inconsistency finding, conditioning math, options
- **Next step:** Shipped in PR #14888.

### DL-#14500 · NP-07 Loop-Constraint Projection of Policy Output (Grip Loop, Diagnostic) (#14500)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14500
- **Branch:** merged via #14824
- **PR:** #14824
- **Paths:** see #14824
- **Started:** 2026-11-08
- **Last verified:** 2026-11-08 (`884bf8f5`; collated from changes/14500-np07-loop-projection.md)
- **Summary:** NP-07 loop-constraint projection of policy output (grip loop, diagnostic) (#14500)
- **Next step:** Shipped in PR #14824.

### DL-#14477 · NP-02 Demonstration Fixture and Behavior-Cloned 64X64 MLP Baseline (#14477)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14477
- **Branch:** merged via #14839
- **PR:** #14839
- **Paths:** see #14839
- **Started:** 2026-11-08
- **Last verified:** 2026-11-08 (`e01522fd`; collated from changes/14477-np02-bc-mlp-baseline.md)
- **Summary:** NP-02 demonstration fixture and behavior-cloned 64x64 MLP baseline (#14477)
- **Next step:** Shipped in PR #14839.

### DL-#14763 · SHOW-A18: ROM Diagnostic Decomposing A2 Trail-Wrist Excursions Against Pre-A2 Capture and G2 Options

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14763
- **Branch:** merged via #14886
- **PR:** #14886
- **Paths:** see #14886
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`38cfcf83`; collated from changes/14763-show-a18-rwinputy-range.md)
- **Summary:** SHOW-A18: ROM diagnostic decomposing A2 trail-wrist excursions against pre-A2 capture and G2 options
- **Next step:** Shipped in PR #14886.

### DL-#14796 · SHOW-A20: Refit Capture Club Pose per Trail-Grip Option Removing Copy-Grip Bias

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14796
- **Branch:** merged via #14883
- **PR:** #14883
- **Paths:** see #14883
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`3ee20382`; collated from changes/14796-show-a20-refit-club-pose.md)
- **Summary:** SHOW-A20: refit capture club pose per trail-grip option removing copy-grip bias
- **Next step:** Shipped in PR #14883.

### DL-#14797 · SHOW-A21: Evidence-Based Grip Guards (Dorsal Separation >= 90 Deg and Trail V-Offset Bound in [-75, 75] Deg Grounded in Biomechanical Literature and Tour Data)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14797
- **Branch:** merged via #14882
- **PR:** #14882
- **Paths:** see #14882
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`29f9fac0`; collated from changes/14797-evidence-grip-guards.md)
- **Summary:** SHOW-A21: evidence-based grip guards (dorsal separation >= 90 deg and trail V-offset bound in [-75, 75] deg grounded in biomechanical literature and tour data)
- **Next step:** Shipped in PR #14882.

### DL-#14735 · Key-Frame PNG Writer and Grip-Clock Inset Helper for Styled Renders (SHOW-A10)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14735
- **Branch:** merged via #14880
- **PR:** #14880, #14900
- **Paths:** see #14880
- **Started:** 2026-11-07
- **Last verified:** 2026-11-08 (`df2d7034`; collated from changes/14735-show-a10-evidence.md)
- **Summary:** Key-frame PNG writer and grip-clock inset helper for styled renders (SHOW-A10)
- **Next step:** Shipped in PR #14880.

### DL-#14734 · SHOW-A9: Refit Best-Match MuJoCo Driver on A7 Grip + A8 Trunk Spec; Record Candidate Receipt and Baseline Comparison

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14734
- **Branch:** merged via #14878
- **PR:** #14878
- **Paths:** see #14878
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`f304efdc`; collated from changes/14734-show-a9-mujoco-refit.md)
- **Summary:** SHOW-A9: Refit best-match MuJoCo driver on A7 grip + A8 trunk spec; record candidate receipt and baseline comparison
- **Next step:** Shipped in PR #14878.

### DL-#14738 · Fleet Grip Guard: PR-Tier Test That Every Model and Binding Reads 15° ± 1° Under A7 Convention (SHOW-A13, #14738)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14738
- **Branch:** merged via #14875
- **PR:** #14875
- **Paths:** see #14875
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`62a43192`; collated from changes/14738-show-a13-fleet-grip-guard.md)
- **Summary:** Fleet grip guard: PR-tier test that every model and binding reads 15° ± 1° under A7 convention (SHOW-A13, #14738)
- **Next step:** Shipped in PR #14875.

### DL-#14737 · Apply A7 Grip Roll Zero to Drake, Pinocchio, OpenSim and MyoSuite via the Shared Metric

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14737
- **Branch:** merged via #14873
- **PR:** #14873
- **Paths:** see #14873
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`6823740b`; collated from changes/14737-show-a12-cross-engine-grip.md)
- **Summary:** Apply A7 grip roll zero to Drake, Pinocchio, OpenSim and MyoSuite via the shared metric
- **Next step:** Shipped in PR #14873.

### DL-#14736 · SHOW-A11: Validate Active MuJoCo Spec and Grip Bindings Against A7 Grip Roll Convention (+15 Deg Neutral)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14736
- **Branch:** merged via #14869
- **PR:** #14869
- **Paths:** see #14869
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`910c29fa`; collated from changes/14736-show-a11-apply-a7-grip-convention.md)
- **Summary:** SHOW-A11: Validate active MuJoCo spec and grip bindings against A7 grip roll convention (+15 deg neutral)
- **Next step:** Shipped in PR #14869.

### DL-#14428 · Add GM-5 Posture, Translation, Club-at-Impact, and Han Et Al. (2019) GRF Pivoting Moment Metrics.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14428
- **Branch:** merged via #14832
- **PR:** #14832
- **Paths:** see #14832
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`56341fdd`; collated from changes/14428-gm5-posture-impact-metrics.md)
- **Summary:** Add GM-5 posture, translation, club-at-impact, and Han et al. (2019) GRF pivoting moment metrics.
- **Next step:** Shipped in PR #14832.

### DL-#14424 · Split Address-Relative and Target-Line/Lab-Frame Shoulder and Hip Turn Conventions With Declared Segment Proxies and Chu 2010 Event Reporting (GM-1C).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14424
- **Branch:** merged via #14863
- **PR:** #14863
- **Paths:** see #14863
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`19999b17`; collated from changes/14424-gm1c-shoulder-hip-turn-conventions.md)
- **Summary:** Split address-relative and target-line/lab-frame shoulder and hip turn conventions with declared segment proxies and Chu 2010 event reporting (GM-1c).
- **Next step:** Shipped in PR #14863.

### DL-#14731 · Re-Zero Grip Roll to the Owner's A7 Convention: Wrists on Top of Shaft, + CCW Toward Target

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14731
- **Branch:** merged via #14865
- **PR:** #14865
- **Paths:** see #14865
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`f01bfcee`; collated from changes/14731-grip-roll-zero-a7-convention.md)
- **Summary:** Re-zero grip roll to the owner's A7 convention: wrists on top of shaft, + CCW toward target
- **Next step:** Shipped in PR #14865.

### DL-#14786 · Optional-Stack CI Runs Loop_Nullspace Snapshot vs Spatial Fixture With Real Pinocchio

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14786
- **Branch:** merged via #14788
- **PR:** #14788
- **Paths:** see #14788
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`f88d28a2`; collated from changes/14786-optional-stack-ci-runs-loop-nullspace-sn.md)
- **Summary:** Optional-stack CI runs loop_nullspace snapshot vs spatial fixture with real pinocchio
- **Next step:** Shipped in PR #14788.

### DL-#14753 · SHOW-A17: Mirror the Right Arm in Full_Body_Spec_Anthro_Iron7 and Regenerate Its Two Ground-Support Receipts

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14753
- **Branch:** merged via #14862
- **PR:** #14862
- **Paths:** see #14862
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`b6e846e1`; collated from changes/14753-show-a17-mirror-the-right-arm-in-full-bo.md)
- **Summary:** SHOW-A17: mirror the right arm in full_body_spec_anthro_iron7 and regenerate its two ground-support receipts
- **Next step:** Shipped in PR #14862.

### DL-#14747 · Orient the Drawn Grip Hands From the Wrist Follower Frames: The Drawn Dorsum Shell Now Lies on the Wrist Dorsal Axis (Was 35 Deg Lead / 75 Deg Trail Off)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14747
- **Branch:** merged via #14858
- **PR:** #14858
- **Paths:** see #14858
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`e14357ac`; collated from changes/14747-orient-the-drawn-grip-hands-from-the-wri.md)
- **Summary:** Orient the drawn grip hands from the wrist follower frames: the drawn dorsum shell now lies on the wrist dorsal axis (was 35 deg lead / 75 deg trail off)
- **Next step:** Shipped in PR #14858.

### DL-#14732 · SHOW-A8: V2 Lumbar+Thoracic Trunk Into Best-Match Mujoco_Driver Model With Styled Render

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14732
- **Branch:** merged via #14855
- **PR:** #14855
- **Paths:** see #14855
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`6392e761`; collated from changes/14732-show-a8-v2-trunk-best-match.md)
- **Summary:** SHOW-A8: v2 lumbar+thoracic trunk into best-match mujoco_driver model with styled render
- **Next step:** Shipped in PR #14855.

### DL-#13882 · Catalog Exact Existing Regression Nodes and Merged Provenance for Six Resolved Board Findings

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13882
- **Branch:** merged via #14842
- **PR:** #14842
- **Paths:** see #14842
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`f46f90ed`; collated from changes/13882-regression-catalog.md)
- **Summary:** Catalog exact existing regression nodes and merged provenance for six resolved board findings
- **Next step:** Shipped in PR #14842.

### DL-#14751 · G2: Trail-Hand Grip Option (B15) With Explicit 15 Deg Alpha and Widened Trail V-Offset Bound

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14751
- **Branch:** merged via #14819
- **PR:** #14819
- **Paths:** see #14819
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`832f01c9`; collated from changes/14751-g2-trail-hand-grip-option-b15-with-expli.md)
- **Summary:** G2: trail-hand grip option (b15) with explicit 15 deg alpha and widened trail V-offset bound
- **Next step:** Shipped in PR #14819.

### DL-#12983 · Reload Pinocchio in OpenSim Dynamics Parity When Unit Mocks Left Stubs in Sys.Modules

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12983
- **Branch:** merged via #14843
- **PR:** #14843
- **Paths:** see #14843
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`1c58d8d3`; collated from changes/12983-reload-pinocchio-in-opensim-dynamics-par.md)
- **Summary:** Reload pinocchio in OpenSim dynamics parity when unit mocks left stubs in sys.modules
- **Next step:** Shipped in PR #14843.

### DL-#14743 · TableShotModel Reads the Dispersion Cell for the Shot's Lie and the Conditions' Wind/Slope Bins (CourseConditions Gains Wind/Slope, Default Calm/Flat); Absent Wind/Slope Bins Are Rejected Instead of Silently Using Calm/Flat.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14743
- **Branch:** merged via #14784
- **PR:** #14784
- **Paths:** see #14784
- **Started:** 2026-11-07
- **Last verified:** 2026-11-07 (`b85eb218`; collated from changes/14743-tableshotmodel-reads-the-dispersion-cell.md)
- **Summary:** TableShotModel reads the dispersion cell for the shot's lie and the conditions' wind/slope bins (CourseConditions gains wind/slope, default calm/flat); absent wind/slope bins are rejected instead of silently using calm/flat.
- **Next step:** Shipped in PR #14784.

### DL-#14816 · SHOW-B3b: Omit Ground_Model From Carry-Only Dispersion Tables and Deduplicate Runout Entry Points

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14816
- **Branch:** merged via #14833
- **PR:** #14833
- **Paths:** see #14833
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`a425584c`; collated from changes/14816-carry-table-provenance.md)
- **Summary:** SHOW-B3b: omit ground_model from carry-only dispersion tables and deduplicate runout entry points
- **Next step:** Shipped in PR #14833.

### DL-#14501 · NP-08: Offline SVD-Direction Ablation vs High-Frequency Jitter Benchmark (#14501)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14501
- **Branch:** merged via #14823
- **PR:** #14823
- **Paths:** see #14823
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`d4e3a4eb`; collated from changes/14501-np08-svd-jitter-ablation.md)
- **Summary:** NP-08: offline SVD-direction ablation vs high-frequency jitter benchmark (#14501)
- **Next step:** Shipped in PR #14823.

### DL-#14423 · Map X-Factor Variants to Literature Families and Add Publish Payload + Unit Tests (GM-1B).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14423
- **Branch:** merged via #14818
- **PR:** #14818
- **Paths:** see #14818
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`f0cfeb27`; collated from changes/14423-map-x-factor-variants-to-literature-fami.md)
- **Summary:** Map X-factor variants to literature families and add publish payload + unit tests (GM-1b).
- **Next step:** Shipped in PR #14818.

### DL-#13884 · Controller Synergy Audit Slice 1: ControllerAuditSpec, #13837 Trace Adapter and ControllerAuditReport (UCM, TNC Fail-Closed Below 30 Trials, Null-Space Gain Norm) With an Analytic Lyapunov Golden

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13884
- **Branch:** merged via #14768
- **PR:** #14768, #14856
- **Paths:** see #14768
- **Started:** 2026-11-06
- **Last verified:** 2026-11-07 (`2f7dab23`; collated from changes/13884-controller-synergy-audit-slice-2-minimum.md)
- **Summary:** Controller synergy audit slice 1: ControllerAuditSpec, #13837 trace adapter and ControllerAuditReport (UCM, TNC fail-closed below 30 trials, null-space gain norm) with an analytic Lyapunov golden
- **Next step:** Shipped in PR #14768.

### DL-#14812 · Rewrite Pose_Studio Test_Widgets Against Real Qt Widgets With Qapp Fixture (#14812)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14812
- **Branch:** merged via #14815
- **PR:** #14815
- **Paths:** see #14815
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`5d662cbb`; collated from changes/14812-pose-studio-test-widgets.md)
- **Summary:** rewrite pose_studio test_widgets against real Qt widgets with qapp fixture (#14812)
- **Next step:** Shipped in PR #14815.

### DL-#14810 · Pose_Studio Test_Gui Requests a Real QApplication via an Autouse Qapp Fixture, so It No Longer Crashes an Xdist Worker When No QApplication Exists Yet

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14810
- **Branch:** merged via #14811
- **PR:** #14811
- **Paths:** see #14811
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`02450ff9`; collated from changes/14810-pose-studio-test-gui-requests-a-real-qap.md)
- **Summary:** pose_studio test_gui requests a real QApplication via an autouse qapp fixture, so it no longer crashes an xdist worker when no QApplication exists yet
- **Next step:** Shipped in PR #14811.

### DL-#14807 · Drake UI Abort Regression Tests Request Qapp via an Autouse Fixture, so They No Longer Crash an Xdist Worker When No QApplication Exists Yet

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14807
- **Branch:** merged via #14808
- **PR:** #14808
- **Paths:** see #14808
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`26bb158a`; collated from changes/14807-drake-ui-abort-regression-tests-request.md)
- **Summary:** Drake UI abort regression tests request qapp via an autouse fixture, so they no longer crash an xdist worker when no QApplication exists yet
- **Next step:** Shipped in PR #14808.

### DL-#14767 · Leaderboard Freshness Gate Moved to Scripts/Ci/check_leaderboard_freshness.py; a PR That Changes No Motion_Matching Solver Input Passes Despite a Squash-Orphaned Git_Head, Solver Changes Still Need a Rerun.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14767
- **Branch:** merged via #14783
- **PR:** #14783
- **Paths:** see #14783
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`7031b228`; collated from changes/14767-leaderboard-freshness-gate-moved-to-scri.md)
- **Summary:** Leaderboard freshness gate moved to scripts/ci/check_leaderboard_freshness.py; a PR that changes no motion_matching solver input passes despite a squash-orphaned git_head, solver changes still need a rerun.
- **Next step:** Shipped in PR #14783.

### DL-#14756 · P5: Parity Report Candidate Resolution Fails Closed; Canonical Checksum/Schema/Manifest Failures Never Fall Through to Legacy Loading, Legacy Numeric Archives Detected Explicitly, No Allow_Pickle=True

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14756
- **Branch:** merged via #14770
- **PR:** #14770
- **Paths:** see #14770
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`3e727f85`; collated from changes/14756-p5-parity-report-candidate-resolution-fa.md)
- **Summary:** P5: parity report candidate resolution fails closed; canonical checksum/schema/manifest failures never fall through to legacy loading, legacy numeric archives detected explicitly, no allow_pickle=True
- **Next step:** Shipped in PR #14770.

### DL-#14744 · Strategy CLI: --Shot-Table / --Shot-Model Inject the Selected Shot Model Into the Personal Solve and Report Its Provenance in Text and HTML

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14744
- **Branch:** merged via #14793
- **PR:** #14793
- **Paths:** see #14793
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`87681255`; collated from changes/14744-strategy-cli-shot-table-shot-model-injec.md)
- **Summary:** strategy CLI: --shot-table / --shot-model inject the selected shot model into the personal solve and report its provenance in text and HTML
- **Next step:** Shipped in PR #14793.

### DL-#14727 · Commit the V1 Source Spec of the Trunk_Visuals Driver V2 Bundle and Pin the Generator Against the Committed Rollout on Its True Inputs (Step 1)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14727
- **Branch:** merged via #14785
- **PR:** #14785, #14898
- **Paths:** see #14785
- **Started:** 2026-11-06
- **Last verified:** 2026-11-08 (`4bd0ae50`; collated from changes/14727-drake-v2-reference.md)
- **Summary:** Commit the v1 source spec of the trunk_visuals driver v2 bundle and pin the generator against the committed rollout on its true inputs (step 1)
- **Next step:** Shipped in PR #14785.

### DL-#14633 · Re-Base HX Grip Alpha to Face-On Zero at Flagship Address; Preserve Fitted Euler Triples.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14633
- **Branch:** merged via #14639
- **PR:** #14639
- **Paths:** see #14639
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`25a0c0a6`; collated from changes/14633-re-base-hx-grip-alpha-to-face-on-zero-at.md)
- **Summary:** Re-base HX grip alpha to face-on zero at flagship address; preserve fitted Euler triples.
- **Next step:** Shipped in PR #14639.

### DL-#14621 · Flight Model Validity: Launch Spin Cap, Trajectory Guard, Pipeline Metadata

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14621
- **Branch:** merged via #14641
- **PR:** #14641
- **Paths:** see #14641
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`74f43930`; collated from changes/14621-flight-model-validity-launch-spin-cap-tr.md)
- **Summary:** Flight model validity: launch spin cap, trajectory guard, pipeline metadata
- **Next step:** Shipped in PR #14641.

### DL-#14503 · NP-10 Offline Weight Inspector and Lexical CI Guardrail (#14503)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14503
- **Branch:** merged via #14817
- **PR:** #14817
- **Paths:** see #14817
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`c0739cf2`; collated from changes/14503-np10-weight-inspector.md)
- **Summary:** NP-10 offline weight inspector and lexical CI guardrail (#14503)
- **Next step:** Shipped in PR #14817.

### DL-#14405 · Bump PR Body Lint Caller Pin to the Post-RM#2697 Reusable Workflow (RM#2427)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14405
- **Branch:** merged via #14791
- **PR:** #14791
- **Paths:** see #14791
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`2c971605`; collated from changes/14405-bump-pr-body-lint-caller-pin-to-the-post.md)
- **Summary:** Bump PR Body Lint caller pin to the post-RM#2697 reusable workflow (RM#2427)
- **Next step:** Shipped in PR #14791.

### DL-#12583 · NSX-55: Document DV-12583 as Present on Main, Add Catalog Regression Test, Fix Publication_Gate Checklist

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12583
- **Branch:** merged via #14806
- **PR:** #14806
- **Paths:** see #14806
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`c32c76dc`; collated from changes/12583-nsx55-dv-catalog-gate-and-publication-checklis.md)
- **Summary:** NSX-55: document DV-12583 as present on main, add catalog regression test, fix publication_gate checklist
- **Next step:** Shipped in PR #14806.

### DL-#14771 · Ground-Support Segment-Scale Tie-Break, Receipt Canonical Compare (#14771)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14771
- **Branch:** merged via #14777
- **PR:** #14777
- **Paths:** see #14777
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`d0705fe6`; collated from changes/14771-ground-support-segment-scale-tie-break-r.md)
- **Summary:** Ground-support segment-scale tie-break, receipt canonical compare (#14771)
- **Next step:** Shipped in PR #14777.

### DL-#14757 · NSX-13 Spatial-Fixture Test Double Builds the R11 Shaft Springs Like NativePinocchioModel (Rigid Club), so Test_Snapshot_vs_Spatial_Fixture Is Green Again

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14757
- **Branch:** merged via #14759
- **PR:** #14759
- **Paths:** see #14759
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`a2cba942`; collated from changes/14757-nsx-13-spatial-fixture-test-double-build.md)
- **Summary:** NSX-13 spatial-fixture test double builds the R11 shaft springs like NativePinocchioModel (rigid club), so test_snapshot_vs_spatial_fixture is green again
- **Next step:** Shipped in PR #14759.

### DL-#14746 · SHOW-G1: Mirror the Right-Arm Joint Axes in Driver V1/V2 (R Scapula, Shoulder, Forearm, Wrist) so Positive L and R Coordinates Give Mirrored Motion

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14746
- **Branch:** merged via #14762
- **PR:** #14762
- **Paths:** see #14762
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`a4765a14`; collated from changes/14746-show-g1-mirror-the-right-arm-joint-axes.md)
- **Summary:** SHOW-G1: mirror the right-arm joint axes in driver v1/v2 (R scapula, shoulder, forearm, wrist) so positive L and R coordinates give mirrored motion
- **Next step:** Shipped in PR #14762.

### DL-#14741 · Add PR Size Gate Workflow and Git Numstat Checker (500-Line Cap, Exemptions).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14741
- **Branch:** merged via #14774
- **PR:** #14774, #14820
- **Paths:** see #14774
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`f46eba70`; collated from changes/14741-pr-size-advisory-bdb9.md)
- **Summary:** Add PR size gate workflow and git numstat checker (500-line cap, exemptions).
- **Next step:** Shipped in PR #14774.

### DL-#14478 · NP-03/04 Toy MLP Rank-1 Edit and LoRA Comparator (#14478 #14479)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14478
- **Branch:** merged via #14778
- **PR:** #14778
- **Paths:** see #14778
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`716ce0bc`; collated from changes/14478-np-03-04-toy-mlp-rank-1-edit-and-lora-co.md)
- **Summary:** NP-03/04 toy MLP rank-1 edit and LoRA comparator (#14478 #14479)
- **Next step:** Shipped in PR #14778.

### DL-#14422 · Add Pydantic Golf Metric-Definitions Registry Schema (GM-1a) With Filter Descriptor.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14422
- **Branch:** merged via #14776
- **PR:** #14776
- **Paths:** see #14776
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`78676865`; collated from changes/14422-add-pydantic-golf-metric-definitions-reg.md)
- **Summary:** Add Pydantic golf metric-definitions registry schema (GM-1a) with filter descriptor.
- **Next step:** Shipped in PR #14776.

### DL-#12724 · S6 Catalogue Reason Now Names the Missing Arm on the NSX-97 UCM Split

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12724
- **Branch:** merged via #14761
- **PR:** #14761
- **Paths:** see #14761
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`104a43b6`; collated from changes/12724-s6-catalogue-reason-now-names-the-missin.md)
- **Summary:** S6 catalogue reason now names the missing arm on the NSX-97 UCM split
- **Next step:** Shipped in PR #14761.

### DL-#12721 · Xfail Placeholder for Anthro Noise-Aware Receipt (#12721)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12721
- **Branch:** merged via #14779
- **PR:** #14779
- **Paths:** see #14779
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`7bf79325`; collated from changes/12721-xfail-placeholder-for-anthro-noise-aware.md)
- **Summary:** xfail placeholder for anthro noise-aware receipt (#12721)
- **Next step:** Shipped in PR #14779.

### DL-#12719 · Event_Outcome_Rows Refuses a Near-Grazing Impact (Transversality Below GRAZING_TRANSVERSALITY) Instead of Returning Unreliable Linear Rows

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12719
- **Branch:** merged via #14782
- **PR:** #14782
- **Paths:** see #14782
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`05fb6000`; collated from changes/12719-event-outcome-rows-refuses-a-near-grazin.md)
- **Summary:** event_outcome_rows refuses a near-grazing impact (transversality below GRAZING_TRANSVERSALITY) instead of returning unreliable linear rows
- **Next step:** Shipped in PR #14782.

### DL-#14748 · Native_Viewer_Export CLI Explains a Foreign Shared Package Shadowing the Vendored Tools Tree

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14748
- **Branch:** merged via #14769
- **PR:** #14769, #14809
- **Paths:** see #14769
- **Started:** 2026-11-06
- **Last verified:** 2026-11-07 (`f111638e`; collated from changes/14748-native-viewer-export-bootstrap-imports-a.md)
- **Summary:** native_viewer_export CLI explains a foreign shared package shadowing the vendored Tools tree
- **Next step:** Shipped in PR #14769.

### DL-#14739 · Simscape Grip Docs State the SHOW-A7 Grip Zero (#14731); a MATLAB Test Checks the Gs3dx Topology Frames; the R2025b Address-Roll Check Is Cataloged as Deferred

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14739
- **Branch:** merged via #14772
- **PR:** #14772
- **Paths:** see #14772
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`90c8e841`; collated from changes/14739-simscape-grip-docs-state-the-show-a7-gri.md)
- **Summary:** Simscape grip docs state the SHOW-A7 grip zero (#14731); a MATLAB test checks the gs3dx topology frames; the R2025b address-roll check is cataloged as deferred
- **Next step:** Shipped in PR #14772.

### DL-#12737 · SPN-0B data_extraction.md: DOIs and Full Author Lists for Pearcy 1984, Fujii 2007, Wilke 1999 and Wu 2002 From Crossref; No Verified_From_Primary Flag Changed

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12737
- **Branch:** merged via #14760
- **PR:** #14760, #14924
- **Paths:** see #14760
- **Started:** 2026-11-06
- **Last verified:** 2026-11-10 (`ee2d7b2c`; collated from changes/12737-contract-test-bruno-b2-supplement-stays.md)
- **Summary:** SPN-0b data_extraction.md: DOIs and full author lists for Pearcy 1984, Fujii 2007, Wilke 1999 and Wu 2002 from Crossref; no verified_from_primary flag changed
- **Next step:** Shipped in PR #14760.

### DL-#12684 · Pin the R4 Capture-Mode Address Knee on the Canonical Receipts and Record Their Marker RMS Before/After (Acceptance 6)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12684
- **Branch:** merged via #14758
- **PR:** #14758
- **Paths:** see #14758
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`410d9f58`; collated from changes/12684-pin-the-r4-capture-mode-address-knee-on.md)
- **Summary:** Pin the R4 capture-mode address knee on the canonical receipts and record their marker RMS before/after (acceptance 6)
- **Next step:** Shipped in PR #14758.

### DL-#13044 · OSL-7 Predictive Deadlift Receipt Now Records the Tracked Comparison as Unavailable (Non-Converged Solve) via a No-Solve --Refresh-Comparison Mode; lift_models.tex Notes the Exploratory Status.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13044
- **Branch:** merged via #14766
- **PR:** #14766
- **Paths:** see #14766
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`8065fb0f`; collated from changes/13044-osl-7-predictive-deadlift-receipt-now-re.md)
- **Summary:** OSL-7 predictive deadlift receipt now records the tracked comparison as unavailable (non-converged solve) via a no-solve --refresh-comparison mode; lift_models.tex notes the exploratory status.
- **Next step:** Shipped in PR #14766.

### DL-#14749 · Fix Lod-Quality-Gate Regression: TableShotModel Asks DispersionTable.Club_Label Instead of Reaching Into Provenance (Law of Demeter)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14749
- **Branch:** merged via #14752
- **PR:** #14752
- **Paths:** see #14752
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`5a89fb4d`; collated from changes/14749-fix-lod-quality-gate-regression-tablesho.md)
- **Summary:** Fix lod-quality-gate regression: TableShotModel asks DispersionTable.club_label instead of reaching into provenance (Law of Demeter)
- **Next step:** Shipped in PR #14752.

### DL-#14703 · Putting Studio: Exact Imperial Ft Factor, a Parity Test for the Core Green Edge Clamp, and Locked Slope Sliders That Read the Core Green's Slope at the Ball

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14703
- **Branch:** merged via #14709
- **PR:** #14709
- **Paths:** see #14709
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`c179ec77`; collated from changes/14703-putting-studio-exact-imperial-ft-factor.md)
- **Summary:** Putting studio: exact imperial ft factor, a parity test for the core green edge clamp, and locked slope sliders that read the core green's slope at the ball
- **Next step:** Shipped in PR #14709.

### DL-#13779 · Knot_Gradient_Optimiser: Lbfgs_Minimise Records a Per-Evaluation Objective Versus Gradient Timing Split and the Projected-Gradient Norm (Per Evaluation and at SciPy's Final Point) so a Max_Iterations Stop Is Diagnosable; Results Are Pinned Bit-for-Bit

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13779
- **Branch:** merged via #14723
- **PR:** #14723, #14775
- **Paths:** see #14723
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`3d68d51d`; collated from changes/13779-mjx-l-bfgs-parameter-scale-scipy-ftol-gt.md)
- **Summary:** knot_gradient_optimiser: lbfgs_minimise records a per-evaluation objective versus gradient timing split and the projected-gradient norm (per evaluation and at SciPy's final point) so a max_iterations stop is diagnosable; results are pinned bit-for-bit
- **Next step:** Shipped in PR #14723.

### DL-#13057 · Lifting: The Four Strict Residual-Gate Xfails (Deadlift, Bench Press, Snatch, Clean and Jerk) Now Cite #13057, the Hip and Pelvis Rebuild That Owns the Residual Excess, Instead Of #13041

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13057
- **Branch:** merged via #14730
- **PR:** #14730, #14802
- **Paths:** see #14730
- **Started:** 2026-11-06
- **Last verified:** 2026-11-06 (`9569ef7e`; collated from changes/13057-osl-14-scoping-osl-scoping-osl14-fixture.md)
- **Summary:** lifting: the four strict residual-gate xfails (deadlift, bench press, snatch, clean and jerk) now cite #13057, the hip and pelvis rebuild that owns the residual excess, instead of #13041
- **Next step:** Shipped in PR #14730.

### DL-#14698 · Add the Lighthouse Mobile Baseline Receipt (Loopback Fixture Server and Egress Proxy, Report Validator, CI Steps on the Fleet Runner) as the Follow-Up to #14270; Scores Informational, No Threshold.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14698
- **Branch:** merged via #14717
- **PR:** #14717, #14813
- **Paths:** see #14717
- **Started:** 2026-11-05
- **Last verified:** 2026-11-06 (`a5fb78f4`; collated from changes/14698-ci-lighthouse-mobile-baseline-runs-in-it.md)
- **Summary:** Add the Lighthouse mobile baseline receipt (loopback fixture server and egress proxy, report validator, CI steps on the fleet runner) as the follow-up to #14270; scores informational, no threshold.
- **Next step:** Shipped in PR #14717.

### DL-#14610 · SHOW-B3b: Bounce/Roll Rest Positions (Sg_Runout_V1) on the B6 Slope Bins, CRN Lie Modifier for Dispersion Tables

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14610
- **Branch:** merged via #14708
- **PR:** #14708, #14726, #14642
- **Paths:** see #14708
- **Started:** 2026-11-05
- **Last verified:** 2026-11-06 (`ac859070`; collated from changes/14610-show-b3b-dispersion-ground-runout.md)
- **Summary:** SHOW-B3b: bounce/roll rest positions (sg_runout_v1) on the B6 slope bins, CRN lie modifier for dispersion tables
- **Next step:** Shipped in PR #14708.

### DL-#12769 · SPN-14a: Fixture_Peak_Spread_Table Serialises Law Drive Residuals and Lumbar Compression Spreads; the Invented-Fixture Table Is Committed as a CI Golden (Tests/Fixtures/Spine/Spn14a_Fixture_Peaks.Json)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12769
- **Branch:** merged via #14716
- **PR:** #14716
- **Paths:** see #14716
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`6bcba0c7`; collated from changes/12769-spn14a-fixture-golden.md)
- **Summary:** SPN-14a: fixture_peak_spread_table serialises law drive residuals and lumbar compression spreads; the invented-fixture table is committed as a CI golden (tests/fixtures/spine/spn14a_fixture_peaks.json)
- **Next step:** Shipped in PR #14716.

### DL-#14452 · Putting Parity Matrix: Units.React and Units.Tauri Move to Present, Citing the React Units Selector That #14458 Landed (PR #14489); No Putting Gaps Remain in the Matrix (Part Of #14452)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14452
- **Branch:** merged via #14712
- **PR:** #14712
- **Paths:** see #14712
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`5eee5589`; collated from changes/14452-units-matrix-cell.md)
- **Summary:** putting parity matrix: units.react and units.tauri move to present, citing the React units selector that #14458 landed (PR #14489); no putting gaps remain in the matrix (Part of #14452)
- **Next step:** Shipped in PR #14712.

### DL-#14702 · Putting_Dynamics: The Post-Impact Hop Reflects the Vertical Speed Relative to the Green, so a Downhill Putt Settles in Under 0.1 s Instead of Skipping for About 5 s; the Studio Long-Downhill Read Is Back in Core Parity

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14702
- **Branch:** merged via #14705
- **PR:** #14705
- **Paths:** see #14705
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`b76beaa7`; collated from changes/14702-downhill-hop.md)
- **Summary:** putting_dynamics: the post-impact hop reflects the vertical speed relative to the green, so a downhill putt settles in under 0.1 s instead of skipping for about 5 s; the studio long-downhill read is back in core parity
- **Next step:** Shipped in PR #14705.

### DL-#12691 · R11: Per-Engine Chain-Shaft Impact-Window Receipt and Overlay Now Cover Drake and Pinocchio, With an Engine-Free Gate

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12691
- **Branch:** merged via #14692
- **PR:** #14692, #14773
- **Paths:** see #14692
- **Started:** 2026-11-05
- **Last verified:** 2026-11-06 (`7e843383`; collated from changes/12691-manual-ch-67-quotes-the-five-engine-r11.md)
- **Summary:** R11: per-engine chain-shaft impact-window receipt and overlay now cover drake and pinocchio, with an engine-free gate
- **Next step:** Shipped in PR #14692.

### DL-#12567 · NSX-24: Unify ContactForceAllocator and MultiEngineTorqueAllocator on Allocate_Contact_Frame Conic Path

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12567
- **Branch:** merged via #14699
- **PR:** #14699
- **Paths:** see #14699
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`ef08fbcb`; collated from changes/12567-nsx-24-unify-contactforceallocator-and-m.md)
- **Summary:** NSX-24: unify ContactForceAllocator and MultiEngineTorqueAllocator on allocate_contact_frame conic path
- **Next step:** Shipped in PR #14699.

### DL-#14677 · Sg_Optimizer: Common Random Numbers for an Injected ShotOutcomeModel so the Scalar Backup Converges

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14677
- **Branch:** merged via #14688
- **PR:** #14688
- **Paths:** see #14688
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`fe666fa5`; collated from changes/14677-sg-optimizer-common-random-numbers-for-a.md)
- **Summary:** sg_optimizer: common random numbers for an injected ShotOutcomeModel so the scalar backup converges
- **Next step:** Shipped in PR #14688.

### DL-#14650 · Putting Studio: Batch Strokes Gained, the E(d) Chart and the Optimal Pace Read the Core's Solved Player E(d) From Core_Sg.Js (Service_Sg.Sg_Study, Tour-Average Player, Side-Slope Grid), Replacing the Hand-Tuned Comeback Fit

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14650
- **Branch:** merged via #14689
- **PR:** #14689, #14695
- **Paths:** see #14689
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`37294a10`; collated from changes/14650-studio-pace.md)
- **Summary:** Putting studio: batch strokes gained, the E(d) chart and the optimal pace read the core's solved player E(d) from core_sg.js (service_sg.sg_study, tour-average player, side-slope grid), replacing the hand-tuned comeback fit
- **Next step:** Shipped in PR #14689.

### DL-#12553 · NSX-04 Slice 1: Verified Subtalar Inversion/Eversion and MTP Flexion Torque Capacities (da Fonseca 2025, Saeki 2021); Hip and MTP-Extension Slots Keep ASSUMED With Stated Reasons

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12553
- **Branch:** merged via #14684
- **PR:** #14684, #14690, #14696, #14704, #14713, #14725, #14787, #14925
- **Paths:** see #14684
- **Started:** 2026-11-05
- **Last verified:** 2026-11-10 (`448c4a2c`; collated from changes/12553-nsx04-verification-status-inventory.md)
- **Summary:** NSX-04 slice 1: verified subtalar inversion/eversion and MTP flexion torque capacities (da Fonseca 2025, Saeki 2021); hip and MTP-extension slots keep ASSUMED with stated reasons
- **Next step:** Shipped in PR #14684.

### DL-#14649 · Putting Studio: Auto-Solve Runs solveAimCore, a Port of the Core Solve_Aim (Pace Bisection and Heading Secant) Golden-Pinned to the Core Read on the Core Greens, and the Batch Strikes at the Solved Head Speed While the Read Is Current

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14649
- **Branch:** merged via #14683
- **PR:** #14683, #14678
- **Paths:** see #14683
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`032b2de4`; collated from changes/14649-studio-roll.md)
- **Summary:** Putting studio: Auto-Solve runs solveAimCore, a port of the core solve_aim (pace bisection and heading secant) golden-pinned to the core read on the core greens, and the batch strikes at the solved head speed while the read is current
- **Next step:** Shipped in PR #14683.

### DL-#14572 · SHOW-B8: Strategy CLI and HTML View With Tour vs Personal Strokes, Shot Options and Dispersion Ellipses

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14572
- **Branch:** merged via #14659
- **PR:** #14659
- **Paths:** see #14659
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`8b83fe3b`; collated from changes/14572-show-b8-strategy-cli-and-view.md)
- **Summary:** SHOW-B8: strategy CLI and HTML view with tour vs personal strokes, shot options and dispersion ellipses
- **Next step:** Shipped in PR #14659.

### DL-#14551 · Neutral Address Grip: Flagship Driver Grip Set to the HX-Canonical Grip (Alpha -10.73, Psi 40, Gamma 0) so Both Hands Read 15 Deg Face-On (Was 98/108 Deg); Address and Swing Fixtures Re-Solved. Deferred to A6 (#14555): At Address the RW Wrist Reads -41.8 Deg (Range -40..25) and the Driver Swing Leaves the Wrist Ranges (RWInputY 691 Frames to -130 Deg, RWInputX 66 Frames, LWInputY 216 Frames); Numbers in Neutral_Grip.Json. Capture-Matching Receipts Keep Chaining to Full_Body_Spec_Anthro_Driver_Capture.Json (The Pre-A2 Fitted Grip, Fitted to the Capture and Not a Grip Recommendation): On the Neutral Grip the Seeds Receipt Fails the Gates (IK Closure 3.9 -> 76.8 mm vs 5 mm, IK RMS 29.5 -> 55.3 mm, Backswing Min Weight Fraction 0.0 BW, Address-to-Impact 64 mm vs 34 mm), the A6 Problem Statement

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14551
- **Branch:** merged via #14644
- **PR:** #14644
- **Paths:** see #14644
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`1b514e12`; collated from changes/14551-neutral-address-grip-15-deg.md)
- **Summary:** Neutral address grip: flagship driver grip set to the HX-canonical grip (alpha -10.73, psi 40, gamma 0) so both hands read 15 deg face-on (was 98/108 deg); address and swing fixtures re-solved. Deferred to A6 (#14555): at address the RW wrist reads -41.8 deg (range -40..25) and the driver swing leaves the wrist ranges (RWInputY 691 frames to -130 deg, RWInputX 66 frames, LWInputY 216 frames); numbers in neutral_grip.json. Capture-matching receipts keep chaining to full_body_spec_anthro_driver_capture.json (the pre-A2 fitted grip, fitted to the capture and not a grip recommendation): on the neutral grip the seeds receipt fails the gates (IK closure 3.9 -> 76.8 mm vs 5 mm, IK RMS 29.5 -> 55.3 mm, backswing min weight fraction 0.0 BW, address-to-impact 64 mm vs 34 mm), the A6 problem statement
- **Next step:** Shipped in PR #14644.

### DL-#14648 · Putting Studio: Green Type Selector for Every Core Green Preset (Undulating Included), Height Grids Generated From GreenSpec Into Core_Greens.Js and Read With the Core Bilinear Lookup

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14648
- **Branch:** merged via #14676
- **PR:** #14676
- **Paths:** see #14676
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`42b69064`; collated from changes/14648-studio-greens.md)
- **Summary:** Putting studio: Green Type selector for every core green preset (undulating included), height grids generated from GreenSpec into core_greens.js and read with the core bilinear lookup
- **Next step:** Shipped in PR #14676.

### DL-#14660 · SG Optimizer: Common Random Numbers and Pin-Relative Aims so HoleMDP Value Iteration Converges (Cap Is a Safety Bound Only)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14660
- **Branch:** merged via #14663
- **PR:** #14663
- **Paths:** see #14663
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`6780ce16`; collated from changes/14660-sg-mdp-common-random-numbers.md)
- **Summary:** SG optimizer: common random numbers and pin-relative aims so HoleMDP value iteration converges (cap is a safety bound only)
- **Next step:** Shipped in PR #14663.

### DL-#14449 · Cross-Frontend Putting Golden: Shared-Core Fixture, React/Tauri Replay Parity Tests and a Generated Parity Report

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14449
- **Branch:** merged via #14666
- **PR:** #14666
- **Paths:** see #14666
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`300f194e`; collated from changes/14449-cross-frontend-putting-golden-shared-cor.md)
- **Summary:** Cross-frontend putting golden: shared-core fixture, React/Tauri replay parity tests and a generated parity report
- **Next step:** Shipped in PR #14666.

### DL-#14647 · Putting Studio: The Skid-to-Roll Transition Ends Where the Core Simulate_Strike Skid Ends for the Ball Speed, and the Telemetry Shows the Core Launch Angle

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14647
- **Branch:** merged via #14662
- **PR:** #14662, #14667
- **Paths:** see #14662
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`33a3f33f`; collated from changes/14647-studio-heel-toe-strike.md)
- **Summary:** Putting studio: the skid-to-roll transition ends where the core simulate_strike skid ends for the ball speed, and the telemetry shows the core launch angle
- **Next step:** Shipped in PR #14662.

### DL-#14600 · Head-CG Offset (Opt-In) and Toe_Down_m Shaft Metric for the C4 Shaft State (Refs #14600)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14600
- **Branch:** merged via #14658
- **PR:** #14658, #14674
- **Paths:** see #14658
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`7ecb8022`; collated from changes/14600-shaft-axis-from-chain.md)
- **Summary:** Head-CG offset (opt-in) and toe_down_m shaft metric for the C4 shaft state (Refs #14600)
- **Next step:** Shipped in PR #14658.

### DL-#14496 · Clarify the Fixed-State ZTCF Drift and ZVCF Control Component Definitions

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14496
- **Branch:** merged via #14498
- **PR:** #14498
- **Paths:** see #14498
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`b9398531`; collated from changes/14496-clarify-the-fixed-state-ztcf-drift-and-z.md)
- **Summary:** Clarify the fixed-state ZTCF drift and ZVCF control component definitions
- **Next step:** Shipped in PR #14498.

### DL-#12688 · R9a: Commit Drake and Pinocchio MeshCat Appearance Stills and Evidence Test (#12688)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12688
- **Branch:** merged via #14395
- **PR:** #14395
- **Paths:** see #14395
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`ec12acb9`; collated from changes/12688-regenerate-matched-swing-ledger-after-r9.md)
- **Summary:** R9a: commit Drake and Pinocchio MeshCat appearance stills and evidence test (#12688)
- **Next step:** Shipped in PR #14395.

### DL-#14646 · Putting Studio: A Metric/Imperial Selector; Distance, Speed and Strike Labels Come From the Shared Units Table Generated Into Core_Units.Js

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14646
- **Branch:** merged via #14654
- **PR:** #14654, #14671
- **Paths:** see #14654
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`ea6d0a4d`; collated from changes/14646-units-api.md)
- **Summary:** Putting studio: a Metric/Imperial selector; distance, speed and strike labels come from the shared units table generated into core_units.js
- **Next step:** Shipped in PR #14654.

### DL-#14640 · Replace Wall-Clock Speed-Up Ratio in Test_Topology_Contact_Batched With a Deterministic Per-Pair Kernel Call Count

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14640
- **Branch:** merged via #14651
- **PR:** #14651
- **Paths:** see #14651
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`2c4bed1a`; collated from changes/14640-replace-wall-clock-speed-up-ratio-in-tes.md)
- **Summary:** Replace wall-clock speed-up ratio in test_topology_contact_batched with a deterministic per-pair kernel call count
- **Next step:** Shipped in PR #14651.

### DL-#14495 · Align MuJoCo Counterfactuals Zvcf() and the MATLAB ZVCF Comments With the Control-Component Contract

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14495
- **Branch:** merged via #14636
- **PR:** #14636
- **Paths:** see #14636
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`96bb31ae`; collated from changes/14495-align-mujoco-counterfactuals-zvcf-and-th.md)
- **Summary:** Align MuJoCo counterfactuals zvcf() and the MATLAB ZVCF comments with the control-component contract
- **Next step:** Shipped in PR #14636.

### DL-#14472 · React/Tauri Putting Page: Saved Scenarios and Comparison Over the Shared Core, Golden-Pinned (Part 1 Of #14472)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14472
- **Branch:** merged via #14637
- **PR:** #14637, #14656
- **Paths:** see #14637
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`b2c6da6c`; collated from changes/14472-react-tauri-putting-page-export-and-impo.md)
- **Summary:** React/Tauri putting page: saved scenarios and comparison over the shared core, golden-pinned (part 1 of #14472)
- **Next step:** Shipped in PR #14637.

### DL-#14630 · Putting: The Strokes Gained Chart Title Fits the Phone-Width Window and the Holed Check Box Reads on the Dark Controls Panel

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14630
- **Branch:** merged via #14645
- **PR:** #14645
- **Paths:** see #14645
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`43ee052d`; collated from changes/14630-sg-panel-phone-polish.md)
- **Summary:** Putting: the Strokes Gained chart title fits the phone-width window and the Holed check box reads on the dark controls panel
- **Next step:** Shipped in PR #14645.

### DL-#14570 · SHOW-B6: Slope/Stance Bins and Wind Bins in the Dispersion Tables (slope_wind_bins.py), Labelled Assumptions

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14570
- **Branch:** merged via #14628
- **PR:** #14628
- **Paths:** see #14628
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`1558c825`; collated from changes/14570-show-b6-slope-stance-bins-and-wind-bins.md)
- **Summary:** SHOW-B6: slope/stance bins and wind bins in the dispersion tables (slope_wind_bins.py), labelled assumptions
- **Next step:** Shipped in PR #14628.

### DL-#14563 · SHOW-C7 Flight Report and Measured-Swing Flight Card (Part Of #14563)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14563
- **Branch:** merged via #14620
- **PR:** #14620, #14670
- **Paths:** see #14620
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`a3a1276e`; collated from changes/14563-c7-slice-2-address-pose-stills-capture-f.md)
- **Summary:** SHOW-C7 flight report and measured-swing flight card (Part of #14563)
- **Next step:** Shipped in PR #14620.

### DL-#14632 · Drake IK Now Implements the Club-Sole Ground Term (Base Was a No-Op), Fixing the 126.7 mm Address Hover

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14632
- **Branch:** merged via #14634
- **PR:** #14634
- **Paths:** see #14634
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`d58c3d44`; collated from changes/14632-drake-ik-now-implements-the-club-sole-gr.md)
- **Summary:** Drake IK now implements the club-sole ground term (base was a no-op), fixing the 126.7 mm address hover
- **Next step:** Shipped in PR #14634.

### DL-#14568 · SHOW-B4: Physics Backend on Dispersion Tables, EB Shrinkage, Personalised vs Tour Strategy

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14568
- **Branch:** merged via #14623
- **PR:** #14623
- **Paths:** see #14623
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`9b8b8625`; collated from changes/14568-show-b4-physics-backend-on-dispersion-ta.md)
- **Summary:** SHOW-B4: physics backend on dispersion tables, EB shrinkage, personalised vs tour strategy
- **Next step:** Shipped in PR #14623.

### DL-#14617 · Putting Simulator: The Counter-Torque Panel Bounds the Gust Input by the Core's GUST_RANGE_NM Instead of the Preload Range

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14617
- **Branch:** merged via #14626
- **PR:** #14626
- **Paths:** see #14626
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`1a513ea5`; collated from changes/14617-torque-gust-spin-range.md)
- **Summary:** Putting simulator: the counter-torque panel bounds the gust input by the core's GUST_RANGE_NM instead of the preload range
- **Next step:** Shipped in PR #14626.

### DL-#14451 · Add Npm Run Tauri:Artifact: A Local Tauri Installer Named With Version and Commit, Plus a Sha256 File, Never Published

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14451
- **Branch:** merged via #14624
- **PR:** #14624
- **Paths:** see #14624
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`84d828e2`; collated from changes/14451-add-npm-run-tauri-artifact-a-local-tauri.md)
- **Summary:** Add npm run tauri:artifact: a local Tauri installer named with version and commit, plus a sha256 file, never published
- **Next step:** Shipped in PR #14624.

### DL-#14562 · SHOW-C6 Shot-Shape Presets, Ball-Flight Curvature Sign Tests and Swing-Variant Artefact With Run Identity

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14562
- **Branch:** merged via #14618
- **PR:** #14618, #14686
- **Paths:** see #14618
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`46a51da5`; collated from changes/14562-name-shot-presets-by-face-to-path-d30-is.md)
- **Summary:** SHOW-C6 shot-shape presets, ball-flight curvature sign tests and swing-variant artefact with run identity
- **Next step:** Shipped in PR #14618.

### DL-#14471 · Putting Analytics: Versioned Setup Document (The Studio's Schema_Version 4.0.0) With Export/Import (Studio Exports Load Their Green and Distance), Named Scenarios With Their Dispersion Result and a Comparison Against the First, as a Shared Service and /Scenario, /Scenario-Compare, /Setup-Export and /Setup-Import Routes

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14471
- **Branch:** merged via #14615
- **PR:** #14615, #14629
- **Paths:** see #14615
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`392a57d0`; collated from changes/14471-putting-scenarios-desktop.md)
- **Summary:** Putting analytics: versioned setup document (the studio's schema_version 4.0.0) with export/import (studio exports load their green and distance), named scenarios with their dispersion result and a comparison against the first, as a shared service and /scenario, /scenario-compare, /setup-export and /setup-import routes
- **Next step:** Shipped in PR #14615.

### DL-#14571 · SHOW-B7: Fit a SkillFamily From Launch-Monitor Shot Data, Shrunk Toward a Preset

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14571
- **Branch:** merged via #14616
- **PR:** #14616
- **Paths:** see #14616
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`f2deca76`; collated from changes/14571-show-b7-fit-a-skillfamily-from-launch-mo.md)
- **Summary:** SHOW-B7: fit a SkillFamily from launch-monitor shot data, shrunk toward a preset
- **Next step:** Shipped in PR #14616.

### DL-#14569 · SHOW-B5: Parametric Lie/Rough Impact-Condition Modifier (Physics/lie_interaction.py) With Labelled Assumptions

- **State:** shipped
- **Owner:** fleet-orchestrator
- **Issue:** #14569
- **Branch:** desk/show-neck/14569-lie-interaction
- **PR:** #14613, #14852
- **Paths:** see #14613
- **Started:** 2026-11-05
- **Last verified:** 2026-11-07 (`684daf6b`; collated from changes/14569-b5-the-lie-spin-factor-now-reaches-the-b.md)
- **Summary:** SHOW-B5: parametric lie/rough impact-condition modifier (physics/lie_interaction.py) with labelled assumptions
- **Next step:** Shipped in PR #14613.

### DL-#14470 · React Putting Simulator Gains the Counter-Torque Stabilization Explorer (Sliders, Putter Preset, Hand-Couple Chart, Impact Readouts) Over the Core /Counter-Torque Route

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14470
- **Branch:** merged via #14611
- **PR:** #14611
- **Paths:** see #14611
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`162c2aef`; collated from changes/14470-react-putting-simulator-gains-the-counte.md)
- **Summary:** React putting simulator gains the counter-torque stabilization explorer (sliders, putter preset, hand-couple chart, impact readouts) over the core /counter-torque route
- **Next step:** Shipped in PR #14611.

### DL-#14567 · SHOW-B3: Cached, Versioned MC Dispersion Tables (CRN) + Pipeline Landing Adapter

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14567
- **Branch:** merged via #14608
- **PR:** #14608
- **Paths:** see #14608
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`25f3520a`; collated from changes/14567-show-b3-cached-versioned-mc-dispersion-t.md)
- **Summary:** SHOW-B3: cached, versioned MC dispersion tables (CRN) + pipeline landing adapter
- **Next step:** Shipped in PR #14608.

### DL-#14561 · SHOW-C5 Speed Continuation 100-125 Mph With Pareto Front and Binding Report

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14561
- **Branch:** merged via #14606
- **PR:** #14606, #14673, #14681, #14721, #14722
- **Paths:** see #14606
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`c64cb10c`; collated from changes/14561-show-c5-ladder-binding-report-classes-ev.md)
- **Summary:** SHOW-C5 speed continuation 100-125 mph with Pareto front and binding report
- **Next step:** Shipped in PR #14606.

### DL-#14550 · Add Golf-Language Face-On Address Grip Roll Metric (0 Deg Vertical, + Strong) With Flagship Measurement and Overlay Render

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14550
- **Branch:** merged via #14601
- **PR:** #14601
- **Paths:** see #14601
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`986b3224`; collated from changes/14550-add-golf-language-face-on-address-grip-r.md)
- **Summary:** Add golf-language face-on address grip roll metric (0 deg vertical, + strong) with flagship measurement and overlay render
- **Next step:** Shipped in PR #14601.

### DL-#14560 · SHOW-C4: Shaft_State_Series Reports MuJoCo Chain-Shaft Tip Deflection, Lead/Droop, Deflection Rate and Strain Energy Through Impact, With R11-Window Evidence on the Best-Fit Driver

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14560
- **Branch:** merged via #14598
- **PR:** #14598
- **Paths:** see #14598
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`1760fe35`; collated from changes/14560-show-c4-shaft-state-series-reports-mujoc.md)
- **Summary:** SHOW-C4: shaft_state_series reports MuJoCo chain-shaft tip deflection, lead/droop, deflection rate and strain energy through impact, with R11-window evidence on the best-fit driver
- **Next step:** Shipped in PR #14598.

### DL-#14554 · SHOW-A5: Trunk Observability Report (Marker Jacobian SVD), ADR-0059 and Coupled-by-Default Showpiece Spec

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14554
- **Branch:** merged via #14603
- **PR:** #14603
- **Paths:** see #14603
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`513127ec`; collated from changes/14554-show-a5-trunk-observability-report-marke.md)
- **Summary:** SHOW-A5: trunk observability report (marker Jacobian SVD), ADR-0059 and coupled-by-default showpiece spec
- **Next step:** Shipped in PR #14603.

### DL-#14553 · SHOW-A4: Cited Neck ROM (neck_rom.py, Single Source for Range_of_Motion) Enforced as MJCF Joint Limits on NeckInput X/Y/Z

- **State:** shipped
- **Owner:** fleet-orchestrator
- **Issue:** #14553
- **Branch:** desk/show-neck/14553-neck-segment
- **PR:** #14578
- **Paths:** see #14578
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`00e1e9ff`; collated from changes/14553-show-a4-neck-rom.md)
- **Summary:** SHOW-A4: cited neck ROM (neck_rom.py, single source for range_of_motion) enforced as MJCF joint limits on NeckInput X/Y/Z
- **Next step:** Shipped in PR #14578.

### DL-#14595 · CI Standard Sets PIP_RETRIES=10 and PIP_DEFAULT_TIMEOUT=60 Workflow-Wide and Passes Pip-Audit --Timeout 60, so Transient PyPI Read Timeouts No Longer Fail CI

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14595
- **Branch:** merged via #14597
- **PR:** #14597
- **Paths:** see #14597
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`807605a6`; collated from changes/14595-ci-standard-sets-pip-retries-10-and-pip.md)
- **Summary:** CI Standard sets PIP_RETRIES=10 and PIP_DEFAULT_TIMEOUT=60 workflow-wide and passes pip-audit --timeout 60, so transient PyPI read timeouts no longer fail CI
- **Next step:** Shipped in PR #14597.

### DL-#14566 · SHOW-B2: SkillFamily Schema (Impact-Condition Mean/Cov + Provenance), Assumption-Marked Presets, Registry, PlayerProfile Link

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14566
- **Branch:** merged via #14590
- **PR:** #14590
- **Paths:** see #14590
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`8e6506c8`; collated from changes/14566-show-b2-skillfamily-schema-impact-condit.md)
- **Summary:** SHOW-B2: SkillFamily schema (impact-condition mean/cov + provenance), assumption-marked presets, registry, PlayerProfile link
- **Next step:** Shipped in PR #14590.

### DL-#14530 · Putting Green Simulator: The Ud.Putt_Scene/1 Export Writes and Reads Impact_Toe_m (Optional on Read, Default 0.0, Range-Checked Against IMPACT_TOE_RANGE_M), so a Toe or Heel Strike Round-Trips

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14530
- **Branch:** merged via #14594
- **PR:** #14594
- **Paths:** see #14594
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`eaa7b7ac`; collated from changes/14530-scene-export-impact-toe.md)
- **Summary:** Putting Green Simulator: the ud.putt_scene/1 export writes and reads impact_toe_m (optional on read, default 0.0, range-checked against IMPACT_TOE_RANGE_M), so a toe or heel strike round-trips
- **Next step:** Shipped in PR #14594.

### DL-#14450 · React Web UI: One Command (Npm Run Local) Builds From the Lockfile and Serves It Through the API From a Clean Checkout; CI Build Check Runs the Same Clean-Build Check; README Run Steps; a Test Bars Paid-API Hosts and Keys

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14450
- **Branch:** merged via #14593
- **PR:** #14593
- **Paths:** see #14593
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`69ae6718`; collated from changes/14450-react-web-ui-one-command-npm-run-local-b.md)
- **Summary:** React web UI: one command (npm run local) builds from the lockfile and serves it through the API from a clean checkout; CI Build Check runs the same clean-build check; README run steps; a test bars paid-API hosts and keys
- **Next step:** Shipped in PR #14593.

### DL-#14559 · SHOW-C3a: SplineTorqueControl Piecewise-Linear Torque Parameterisation

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14559
- **Branch:** merged via #14589
- **PR:** #14589, #14596, #14625, #14638, #14653, #14655, #14680, #14685, #14693
- **Paths:** see #14589
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`d865f3c3`; collated from changes/14559-show-c3c-d32-assumption-limited-vs-proba.md)
- **Summary:** SHOW-C3a: SplineTorqueControl piecewise-linear torque parameterisation
- **Next step:** Shipped in PR #14589.

### DL-#14552 · SHOW-A3: Optional Lumbar:Thoracic Regional-Split Coupling (Trunk_Coupling, Off by Default) for the Full-Body-V2 Trunk, MuJoCo Joint Equalities, and Trunk Joint-Tree/Mass/ROM Contract Tests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14552
- **Branch:** merged via #14577
- **PR:** #14577
- **Paths:** see #14577
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`0cecc8f7`; collated from changes/14552-show-a3-optional-lumbar-thoracic-regiona.md)
- **Summary:** SHOW-A3: optional lumbar:thoracic regional-split coupling (trunk_coupling, off by default) for the full-body-v2 trunk, MuJoCo joint equalities, and trunk joint-tree/mass/ROM contract tests
- **Next step:** Shipped in PR #14577.

### DL-#14466 · Putting API: POST /Solve-Aim-3D Solves the Pace and Line That Hole a /Simulate-3D Putt on That Route's Own Surface and Head (Core Solve_Aim, Like the Desktop Auto-Solve); the 3-D Request Models Move to Src/Api/Models/putting_green_requests.py

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14466
- **Branch:** merged via #14588
- **PR:** #14588, #14604
- **Paths:** see #14588
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`416c462c`; collated from changes/14466-react-putting-simulator-gains-the-deskto.md)
- **Summary:** Putting API: POST /solve-aim-3d solves the pace and line that hole a /simulate-3d putt on that route's own surface and head (core solve_aim, like the desktop Auto-Solve); the 3-D request models move to src/api/models/putting_green_requests.py
- **Next step:** Shipped in PR #14588.

### DL-#14441 · WSCG Update Manuscript Draft on the Best-Match MuJoCo ZTCF/ZVCF Run, Citing the Pinned Run_Id

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14441
- **Branch:** merged via #14584
- **PR:** #14584
- **Paths:** see #14584
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`c82e3803`; collated from changes/14441-wscg-update-manuscript-draft-on-the-best.md)
- **Summary:** WSCG update manuscript draft on the best-match MuJoCo ZTCF/ZVCF run, citing the pinned run_id
- **Next step:** Shipped in PR #14584.

### DL-#14565 · SHOW-B1: ShotOutcomeModel Protocol + Backend Registry; Tour Baseline Is the Default Quick_Sg Backend

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14565
- **Branch:** merged via #14581
- **PR:** #14581
- **Paths:** see #14581
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`950bedd8`; collated from changes/14565-show-b1-shotoutcomemodel-protocol-backen.md)
- **Summary:** SHOW-B1: ShotOutcomeModel protocol + backend registry; tour baseline is the default quick_sg backend
- **Next step:** Shipped in PR #14581.

### DL-#14558 · SHOW-C2: ImpactConditions + Evaluate_Rollout Facade for Optimiser Loops (Face, Path, AoA, Speed, Strike) Over Impact_Parameters

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14558
- **Branch:** merged via #14579
- **PR:** #14579
- **Paths:** see #14579
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`ddf355e4`; collated from changes/14558-show-c2-impactconditions-evaluate-rollou.md)
- **Summary:** SHOW-C2: ImpactConditions + evaluate_rollout facade for optimiser loops (face, path, AoA, speed, strike) over impact_parameters
- **Next step:** Shipped in PR #14579.

### DL-#14557 · SHOW-C1: ImpactTarget, TrustRegion and PhysiologicalEnvelope Contracts (Impact_Optimization Package; Face:Path 10:1; NSX-04 Torque Limits; Cited Velocity YAML)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14557
- **Branch:** merged via #14580
- **PR:** #14580, #14707
- **Paths:** see #14580
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`3e6b8f9b`; collated from changes/14557-source-skilled-male-peak-angular-velocit.md)
- **Summary:** SHOW-C1: ImpactTarget, TrustRegion and PhysiologicalEnvelope contracts (impact_optimization package; face:path 10:1; NSX-04 torque limits; cited velocity YAML)
- **Next step:** Shipped in PR #14580.

### DL-#14464 · Putting API: /Simulate-3D Accepts the Core Green Shape (Green: GreenShapeRequest, a Preset or Slope, Tier and Bumps) and Rolls on Its GreenSpec Height Field, Reporting That Spec's Grade and Aspect, and Strikes Along Aim_Heading_Deg (Was Always +X), so the React Page Can Send the Core Presets and an Aimed Putt

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14464
- **Branch:** merged via #14583
- **PR:** #14583, #14586
- **Paths:** see #14583
- **Started:** 2026-11-05
- **Last verified:** 2026-11-05 (`f2db11cc`; collated from changes/14464-react-putting-page-green-type-selector-o.md)
- **Summary:** Putting API: /simulate-3d accepts the core green shape (green: GreenShapeRequest, a preset or slope, tier and bumps) and rolls on its GreenSpec height field, reporting that spec's grade and aspect, and strikes along aim_heading_deg (was always +x), so the React page can send the core presets and an aimed putt
- **Next step:** Shipped in PR #14583.

### DL-#14474 · React Putting Page: Desktop Camera Views and Jumps to the Core Phase Marks

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14474
- **Branch:** merged via #14576
- **PR:** #14576
- **Paths:** see #14576
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`6ea810fa`; collated from changes/14474-react-putting-page-desktop-camera-views.md)
- **Summary:** React putting page: desktop camera views and jumps to the core phase marks
- **Next step:** Shipped in PR #14576.

### DL-#14469 · Putting Dynamics: Counter-Torque Stabilization Model (Two-Hand Couples, Off-Plane Preload, Noise, Gust, Lie, Putter) in the Shared Core, Reading the Shared Putter Presets, Drift-Gated Against the Studio

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14469
- **Branch:** merged via #14548
- **PR:** #14548, #14602, #14609
- **Paths:** see #14548
- **Started:** 2026-11-04
- **Last verified:** 2026-11-05 (`c39d2c75`; collated from changes/14469-putting-counter-torque-desktop.md)
- **Summary:** Putting dynamics: counter-torque stabilization model (two-hand couples, off-plane preload, noise, gust, lie, putter) in the shared core, reading the shared putter presets, drift-gated against the studio
- **Next step:** Shipped in PR #14548.

### DL-#14468 · React Putting Page: Strokes-Gained Readout and Expected-Putts Curve From the Shared Core's /Strokes-Gained

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14468
- **Branch:** merged via #14573
- **PR:** #14573
- **Paths:** see #14573
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`3f51085e`; collated from changes/14468-react-putting-page-strokes-gained-readou.md)
- **Summary:** React putting page: strokes-gained readout and expected-putts curve from the shared core's /strokes-gained
- **Next step:** Shipped in PR #14573.

### DL-#14462 · React Putting Page: Balance Type Selector Over the Core Putter Presets With Head Mass and MOI Inputs; Balance_Type React/Tauri Parity Cells Now Present

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14462
- **Branch:** merged via #14547
- **PR:** #14547
- **Paths:** see #14547
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`0f7f02e1`; collated from changes/14462-react-putting-page-balance-type-selector.md)
- **Summary:** React putting page: Balance Type selector over the core putter presets with head mass and MOI inputs; balance_type React/Tauri parity cells now present
- **Next step:** Shipped in PR #14547.

### DL-#14460 · Putting Page Groundwork: The Seven Core Noise Channels, the Simulator-to-Core Dispersion Arguments and the /Dispersion and /Make-Spread Client, Pinned to the Shared Core by a Golden File (PUTT-SHARE B-Web Part 1)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14460
- **Branch:** merged via #14532
- **PR:** #14532, #14540
- **Paths:** see #14532
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`411d89e5`; collated from changes/14460-react-putting-page-seven-core-noise-chan.md)
- **Summary:** Putting page groundwork: the seven core noise channels, the simulator-to-core dispersion arguments and the /dispersion and /make-spread client, pinned to the shared core by a golden file (PUTT-SHARE b-web part 1)
- **Next step:** Shipped in PR #14532.

### DL-#14518 · Run Release Presets Through the Run Store: Sim_Run.Preset_Run Records Which Files a Preset Wrote, Copies Them Into a Run Directory and Emits a Reproducible Run Manifest

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14518
- **Branch:** merged via #14524
- **PR:** #14524, #14534, #14528, #14539, #14543
- **Paths:** see #14524
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`b753c279`; collated from changes/14518-analytic-and-diagram-figures-in-the-prox.md)
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

### DL-#14473 · Putting: PyQt6 Tile Readouts and Panel Style Move Out of gui.py Into \_Scene_Text and \_Style (No Behaviour Change; Keeps gui.py Under the Monolith Threshold)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14473
- **Branch:** merged via #14533
- **PR:** #14533, #14537, #14546
- **Paths:** see #14533
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`c641ff70`; collated from changes/14473-putting-playback-desktop.md)
- **Summary:** Putting: PyQt6 tile readouts and panel style move out of gui.py into \_scene_text and \_style (no behaviour change; keeps gui.py under the monolith threshold)
- **Next step:** Shipped in PR #14533.

### DL-#14467 · Putting Analytics: Player-Specific Bellman Expected Putts on the Caller's Green (Optimal Pace per Distance, Shared Skill Roll Model) and One Putt's Strokes Gained Against It, on /Tools/Putting-Analytics/Strokes-Gained

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14467
- **Branch:** merged via #14527
- **PR:** #14527, #14542
- **Paths:** see #14527
- **Started:** 2026-11-04
- **Last verified:** 2026-11-04 (`4bf12081`; collated from changes/14467-putting-sg-desktop.md)
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

### DL-#13560 · MuJoCo Grip Samples Read Zero-Slip Pairs Frictionless, Matching the Shared Law's Zero Friction at Rest; the Ten_Finger Hold Net_Moment_Com Xfail Is Removed

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #13560
- **Branch:** merged via #14185
- **PR:** #14185, #14351, #14354, #14711, #14902
- **Paths:** see #14185
- **Started:** 2026-11-01
- **Last verified:** 2026-11-08 (`79dc26d9`; collated from changes/13560-grip-contact-preload-parity.md)
- **Summary:** MuJoCo grip samples read zero-slip pairs frictionless, matching the shared law's zero friction at rest; the ten_finger hold net_moment_com xfail is removed
- **Next step:** Shipped in PR #14185.

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
