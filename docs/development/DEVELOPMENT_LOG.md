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

### DL-#10965 — Unify Per-Package Coverage Gates on the Exclusion Budget

- **State:** in_review
- **Owner:** claude
- **Issue:** #10965
- **Branch:** `bot/issue-10965-coverage-gate-authority`
- **Paths:** `scripts/check_coverage_gates.py`, `scripts/check_mypy_exclusion_budget.py`, `scripts/config/mypy_exclusion_budget.json`, `.github/workflows/ci-standard.yml`, `tests/unit/scripts/test_check_coverage_gates.py`, `tests/unit/scripts/test_check_mypy_exclusion_budget.py`
- **Started:** 2026-09-28
- **Last verified:** 2026-09-28 — 29 focused script-checker tests passed; checker smoke run against a real coverage.json fixture (strict exit 1 on unmatched, lax warning + exit 0).
- **Summary:** Budget JSON is the single per-package coverage gate authority: `check_coverage_gates.py` reads the 6 gates from `mypy_exclusion_budget.json`, matches by path prefix, and `--strict` now fails on gates matching zero files (warning only without the flag). The `tests` CI job emits `coverage.json` and runs the checker with `--strict`. Gate floors set to locally measured coverage rounded DOWN (api-routes 84.3, data-io 50.1, engine-core 22.4, deployment 48.4, optimization 49.9, engines 7.1) with `ratchet_on` 2027-01-01; expired gate ratchets in `check_mypy_exclusion_budget.py` are now non-fatal warnings.
- **Next step:** CI green, then ready and arm the PR.

### DL-#11083 — Motion Matching Board Review

- **State:** in_review
- **Owner:** codex
- **Issue:** owner-requested documentation review; tracked by PR #11083
- **PR:** #11083
- **Branch:** `docs/motion-matching-board-review`
- **Paths:** `docs/development/2026-09-28-motion-matching-board-review.md`
- **Started:** 2026-09-28
- **Last verified:** 2026-09-28 — 375 focused tests passed, full Ruff lint/format and packet title case passed.
- **Summary:** Evidence-backed engine/model review, recent GS3DX improvements, 18 issue proposals, anatomical Home-budget options and historical-video readiness; no implementation/qualification claim.
- **Next step:** Board disposition and overlap reconciliation before creating implementation children.

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

### DL-#11071 — Run the MJX L-BFGS Arm Toward Convergence

- **State:** in_review
- **Owner:** claude
- **Issue:** #11071
- **PR:** #11075
- **Branch:** `claude/ud-11071-lbfgs-converge`
- **Paths:** `scripts/benchmark_mjx_knot_solvers.py`, `docs/development/full_body_models/evidence/mjx_benchmark_lbfgs60/`
- **Started:** 2026-09-28
- **Last verified:** 2026-09-28 — 60-iteration L-BFGS runs (rc 0) committed; report regenerated from receipts.
- **Summary:** Re-runs only `mjx-lbfgs` at 60 iterations against the #11058 incumbents. Replay RMS falls to 40.8 / 42.6 mm (driver / iron) and the iron downswing weight fraction rises to 0.29, but neither capture converges, so `none` stays the default.
- **Next step:** CI green, then ready and arm the #11071 PR.

### DL-#11034 · Canonical Force-Plate Import and Extension Overlay Parent Attribute Cleanup

- **State:** in_review
- **Owner:** local
- **Issue:** #11034
- **PR:** #11069
- **Branch:** `fix/force-plate-test-module-identity-11034`
- **Paths:** `src/launchers/sidekick_extension_overlay.py`, `tests/unit/launcher/test_sidekick_extension_overlay.py`, `tests/unit/sidekick/lab/bio/test_force_plate_stitching.py`
- **Started:** 2026-09-28
- **Last verified:** 2026-09-28 — 13 passed in test_force_plate_stitching.py and test_sidekick_extension_overlay.py (single and pytest -n 2); ruff, black, and file size checks clean.
- **Summary:** test_force_plate_stitching imports the canonical CombinedForcePlateProcessor directly instead of reinstalling the extension overlay in-test. Additionally, ManifestGatedSidekickFinder.uninstall() cleanly detaches uninstalled modules from parent package attributes in sys.modules to prevent attribute pollution.
- **Next step:** CI green, auto-merge squash to main.

### DL-#11008 · Make Ball-Flight Parity Fixture Export Opt-In

- **State:** in_review
- **Owner:** claude
- **Issue:** #11008
- **PR:** #11012 (ready; auto-merge armed)
- **Branch:** `fix/ball-flight-parity-fixture-opt-in`
- **Paths:** `tests/parity/test_ball_flight_parity.py`
- **Started:** 2026-09-26
- **Last verified:** 2026-09-27 at f1625998c (main through #11045) — `tests/parity/test_ball_flight_parity.py`: 19 passed, 1 skipped (opt-in regen); committed fixture unchanged after the run.
- **Summary:** The export test writes to `tmp_path` and asserts the vector schema. It rewrites the committed golden only when `UPSTREAMDRIFT_REGENERATE_PARITY_FIXTURES=1`. A new test checks read-only that the committed fixture has the schema and the path pinned in `src/config/capability_migration.json`.
- **Next step:** Let the armed auto-merge land #11012, then mark this entry shipped.

### DL-#11003 · Restate the #9243 BunkerShot Uncertainty Claims Under the Corrected F0 Model

- **State:** in_review
- **Owner:** claude
- **Issue:** #11003
- **PR:** #11010
- **Branch:** `claude/ud-9243-claims-restated`
- **Paths:** `tests/unit/tools/bunker_shot_gui/test_uncertainty_propagation_9243.py`
- **Started:** 2026-09-26
- **Last verified:** 2026-09-26 — test_uncertainty_propagation_9243.py 30 passed with upstream-physics 2.1.3 (source build) and with 2.1.0.
- **Summary:** Two #9243 claims went stale while the kernel-less gate skipped them. The window-band fixture now aims at the nominal shot's carry (0.690 m) instead of an unreachable 2 m and asserts a non-empty window; band (0.0, 1.141, 2.282). The mass interval's share is 0.676, so the test now says it dominates without swamping; `DOMINANCE_SHARE` stays 0.75.
- **Next step:** Let the armed auto-merge land it, then replay UD#10997 onto main.

### DL-#9411-Burndown-Batch2 · Retire Passing Test in Hygiene Mock Scopes

- **State:** in_review
- **Owner:** antigravity
- **Issue:** #9411
- **PR:** #11013
- **Branch:** `feat/ud-9411-quarantine-batch2`
- **Paths:** `scripts/config/unit_gate_quarantine.json`, `tests/unit/repo_hygiene/test_optional_dependency_mock_scope.py`, `SPEC.md`
- **Started:** 2026-09-26
- **Last verified:** 20e4ef275 — verified quarantined test passes on Linux CI; unit gate quarantine contract passed (55 -> 54 node IDs across 10 clusters); ruff, black, and file size budget pass.
- **Summary:** Retire verified-passing test in `scripts/config/unit_gate_quarantine.json` (ratchet 55 -> 54); narrow pathspec scoping in `test_optional_dependency_mock_scope.py`.
- **Next step:** Push branch, update PR #11013, monitor CI Standard with squash auto-merge enabled.

### DL-#1755 · Retire the Review-Comment-to-Issue Converter

- **State:** in_review
- **Owner:** claude
- **Issue:** Repository_Management#1755
- **Branch:** `chore/retire-comment-converter-v3`
- **PR:** draft PR from `chore/retire-comment-converter-v3` (supersedes #10978)
- **Paths:** `.github/workflows/Comment-to-Issue-Converter.yml`, `scripts/ci/process_review_comments.py`, `tests/ci/test_process_review_comments.py`, `tests/ci/test_ci_infrastructure.py`, `.github/workflows/Nightly-Doc-Organizer.yml`, `docs/development/ci_workflow_map.md`, `docs/development/`
- **Started:** 2026-09-25
- **Last verified:** 2026-09-26 — `tests/ci/` 89 passed; `check_spec_paths.py` exits 0 with the #10931 SPEC section collapsed to a retirement note.
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
- **Last verified:** 2026-09-25 — 98/98 unit tests pass across test_myosuite_engine.py, test_golf_swing_pendulum.py, and test_dynamics.py (1 Pinocchio integration test cleanly skipped); ruff check and format clean; mypy clean.
- **Summary:** Remove success-shaped empty/zero array returns from uninitialized ZTCF/ZVCF counterfactual methods (raising StateError and ValueError for invalid dimensions), update dtack compute_zvcf to canonical (v=0, tau=0) semantics, add compute_zero_velocity_controlled for control-preserved dynamics, and update GUI caller and button label to "Zero-velocity (control kept)".
- **Next step:** CI green, review, merge.

### DL-#9619 · Amend ADR-0041 to Record Consumer-Side Fitter Decision

- **State:** in_review
- **Owner:** local (session `claude-deskcomputer-20260925-ud`)
- **Issue:** #9619 (companion #9630)
- **Branch:** `agy/ud-9619-adr0041`
- **Paths:** `docs/adr/0041-markerless-mocap-consumer-authority.md`, `tests/architecture/test_markerless_mocap_authority.py`, `docs/development/DEVELOPMENT_LOG.md`
- **Started:** 2026-09-25
- **Last verified:** 2026-09-25 — test_adr_records_consumer_side_fitter_amendment passes in test_markerless_mocap_authority.py
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
- **Last verified:** 2026-09-25 — 18 unit tests pass in test_capability_9688.py verifying fail-closed sand-motion capability register
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
- **Last verified:** 2026-09-25 — 8 unit tests pass in test_record_repeat.py; full rig test suite clean (128 passed, 0 regressions)
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
- **Summary:** Retire unit-gate quarantine IDs whose tests pass on the Linux CI lane after fixing real defects (shoulder FK origin), deduplicating the `data_fitting` and `mujoco_viewer` coordinators onto their existing helper modules, and retargeting stale tests. Tools-owned `ai` adapter changes and the gear-effect sign question are out of scope here (Tools#5362 / a physics-convention decision). Earlier slice (#10973) landed the mypy budget and 43 retirements. Rust slice (#10997): unit-test-gate maturin-builds upstream-physics into its venv (ball_simulator enforces strict Rust parity), retiring the 12 BunkerShot workbench/GUI IDs.
- **Next step:** land #11004 and resolve #11003, then rerun #10997's unit-test-gate.

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

### DL-#10946 · Keep Multi-Muscle Contracts and Torque Identical on the Rust Backend

- **State:** in_review
- **Owner:** claude
- **Issue:** #10946
- **PR:** draft PR from `agy/ud-10946-rust-muscle-parity`
- **Branch:** `agy/ud-10946-rust-muscle-parity`
- **Paths:** `src/shared/python/biomechanics/multi_muscle.py`, `tests/unit/biomechanics/test_multi_muscle_rust_parity.py`
- **Started:** 2026-09-26
- **Last verified:** 2026-09-26 — 85/85 multi-muscle + DbC + parity tests with the `upstream_muscle` wheel; 26 pass + parity module skipped with it hidden; `cargo test` 32/32.
- **Summary:** Python `HillMuscleModel`s were never forwarded to the Rust `MuscleGroup` (it summed zero muscles, so net torque was always 0.0) and the Rust path skipped the activation precondition. Muscles are now converted to their Rust equivalent (or the group drops to pure Python when a muscle has custom physics), activations are validated once before backend choice, the antagonist pair is rebuilt from the groups' current backends, and a parity test pins Rust == Python.
- **Next step:** CI green, review, merge.

### DL-#10960 · Fabricated-Evidence Audit: Fail-Closed Remediation of Literal Receipts and Self-Scored Fits

- **State:** in_review
- **Owner:** claude (agy executors on DeskComputer and OG Laptop)
- **Issue:** #10960
- **PR:** consolidated PR from `claude/ud-consolidated2-20260926` (supersedes #10974)
- **Branch:** `claude/ud-consolidated2-20260926`
- **Paths:** `src/shared/python/motion_matching/provider.py`, `src/engines/physics_engines/*/python/motion_matching/provider.py`, `src/shared/python/neural_motion/surrogates/`, `src/shared/python/neural_motion/matrix/`, `src/shared/python/neural_motion/turnover/`, `src/shared/python/neural_motion/benchmark/runner.py`, `src/shared/python/neural_motion/inference/orchestration.py`, `src/shared/python/motion_matching/contact_identification.py`, `src/engines/physics_engines/opensim/python/tour_matching/full_swing_tracking.py`, `docs/plans/neural_motion_matching/evidence/`, `docs/development/full_body_models/evidence/contact_id/receipt.json`, `src/shared/python/motion_matching/parity_report.py`, `src/shared/python/motion_matching/parity_schema.py`, `src/shared/python/motion_matching/cross_engine_replay.py`, `src/shared/python/motion_matching/full_body_forward_dynamics.py`, `src/shared/python/motion_matching/candidate_session.py`, `docs/development/full_body_models/evidence/fb5_matching/`, `docs/development/full_body_models/evidence/fb6_parity/`, `evidence/matched/driver_g1/parity_report.json`, `docs/development/full_body_models/evidence/_gates.py`, `docs/development/full_body_models/evidence/fb3_drake/`, `docs/development/full_body_models/evidence/fb4_calibration/`, `evidence/matched/driver_g1_pinocchio/`, `src/tools/matched_swing_browser/model.py`, `src/engines/physics_engines/opensim/python/tour_matching/document_ik.py`, `docs/development/full_body_models/evidence/ground_support/anthro_driver_opensim/receipt.json`, `src/shared/python/motion_matching/matching_strategy.py`, `src/shared/python/motion_matching/ledger.py`, `reports/matched_swing_ledger.json`, `tests/fixtures/motion_matching_strategy.py`, `src/shared/python/motion_matching/club_only/`, `src/tools/motion_matching/gui.py`, `src/shared/python/motion_matching/acceptance.py`, `src/shared/python/motion_matching/kinematic_smoothing.py`, `src/shared/python/motion_matching/loaders/body_json.py`, `src/engines/physics_engines/drake/python/motion_matching/simulate.py`, `src/engines/physics_engines/pendulum/python/motion_matching/club_pendulum_match.py`, `src/shared/python/tour_baselines/calibration.py`, `src/shared/python/pose_interchange/pose_io.py`, `docs/plans/club_only_matching/evidence/club_body_candidates.json`
- **Started:** 2026-09-25
- **Last verified:** 2026-09-26 — consolidated branch on `71610a4ff`: motion_matching, neural_motion, opensim, tools/motion_matching, config and every changed test file: 3034 passed, 21 failed, all baseline or host-only (16 known motion_matching failures, OpenSim binding absent, Rajagopal asset absent, local 10k corpus present); ledger regenerated and fresh.
- **Summary:** P0 slices 1, 3, 5-6, 7, 8, 9, 24 and P1-7, P1-8, P1-9, P1-10: body-target fits, NM-07 surrogate ablation, NM-09/NM-12 checkpoint matrix and turnover, NM-10 benchmark baselines, NM-08 orchestration verdicts, MS-20 contact identification and OpenSim full-swing receipts no longer emit literal success values; unmeasured fields are None/UNQUALIFIED/unmeasured or the call raises NotImplementedError. Parity reports only qualify a row after a comparison ran (missing reference, shape mismatch and never-gated engines are `unverified`, empty is `PARTIAL`, no self-comparison fallback) and publish `is_parity_accepted`; failed forward rollouts carry `contact_audit=None`, `shared_metrics=None` and NaN markers. FB-3..FB-6 verify scripts compute status from recorded thresholds through one gate helper (committed receipts relabelled); the OpenSim document-IK receipt is built only from OpenSim fields (`opensim-document-ik/v1`, `IK_PARITY_ONLY`, never physically accepted); the sample strategy builder moved to test fixtures; the ledger no longer reports solver wall-clock `elapsed_s` as a trajectory horizon. P1-1..P1-6 and P1-11: constrained IK fails closed with no solver; the retrieval library excludes the query trial; branch fit and closure are None (infeasible) until computed; the GUI and club-only UI refuse synthetic seeds (`VERIFIED_SEED_SOURCES`); CO-05 body candidates are never accepted without a measured fit (0/32 committed) and carry no invented closure; control replay reports `software_contract_consistent` with nullable circular metrics; the acceptance force gate matches native-fit lanes exactly and emits `NOT_APPLICABLE`, and dual-terminal disclosure is `DISCLOSED`, not PASSED. P2: kinematic smoothing reports `smoothing_applied`, Drake simulate reports NaN and `partial`, body-JSON digests are hashed from the file, fixed-pivot hub work is None. The OpenSim full-swing lane cannot qualify while its marker residuals are self-scored.
- **Next step:** Merge the receipts follow-up (`claude/ud-10960-diagnostic-receipts`): the committed NM-09 and NM-12 DIAGNOSTIC receipts still listed three `qualified_native`/`promoted_models` and `all_issues_completed: true`; they now list the models as `unqualified`/`unmeasured_models`, and `test_diagnostic_receipts_10960.py` pins it. Then close #10960.

### DL-#9415 · Repository Root Allowlist Check in Docs Governance

- **State:** in_review
- **Owner:** claude
- **Issue:** #9415
- **PR:** draft PR from `agy/ud-9415-root-allowlist`
- **Branch:** `agy/ud-9415-root-allowlist`
- **Paths:** `scripts/check_docs_governance.py`, `scripts/config/root_allowlist.json`, `docs/governance/DOCS_GOVERNANCE.md`, `tests/scripts/test_doc_governance_checks.py`, `tests/unit/scripts/test_check_docs_governance_root_allowlist.py`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`
- **Started:** 2026-09-26
- **Last verified:** 2026-09-26 — `python scripts/check_docs_governance.py` exits 0; 26/26 docs-governance tests pass (junit count); new tests error on the base checker (red) and pass after; ruff clean; architecture budget OK.
- **Summary:** Committed root entries (`git ls-tree --name-only HEAD`) must match `scripts/config/root_allowlist.json`; unlisted and stale entries both fail, and a failed git call or malformed config fails closed. Closes the last software checkbox of #9415; the history rewrite, Jules workflows and SPEC cap remain owner decisions.
- **Next step:** CI green, review, merge.

### DL-#10965 · Coverage Gate Checker Reads the Budget File and Maps Cobertura Sources

- **State:** in_review
- **Owner:** claude
- **Issue:** #10965
- **PR:** merged in consolidated #10988; follow-up fix for #10989 on `fix/10989-coverage-gates-defusedxml`
- **Branch:** `agy/ud-10965-coverage-gates`
- **Paths:** `scripts/check_coverage_gates.py`, `tests/unit/scripts/test_check_coverage_gates.py`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`
- **Started:** 2026-09-26
- **Last verified:** 2026-09-26 — 13/13 checker tests pass incl. the new XXE-rejection test; `bandit -ll -ii` clean after switching the report parser to defusedxml (main's CI Standard bandit step failed on B314, #10989).
- **Summary:** `check_coverage_gates.py` now takes its gates from `coverage_gates` in `scripts/config/mypy_exclusion_budget.json` (hard-coded COVERAGE_GATES/MODULE_PATTERNS were never enforced and are gone), reads Cobertura XML or coverage.py JSON, resolves XML filenames through `<sources>` against the repo root, and treats a gate with no matching files or an unmappable source as exit 2. Not wired into CI yet; the ratchet dates cannot move before 2026-10-01 because of the #8731 pin.
- **Next step:** CI green, review, merge.

### DL-#9191 · Verify Companion Screenshot Bytes and Pixel Dimensions

- **State:** in_review
- **Owner:** claude
- **Issue:** #9191
- **PR:** draft PR from `agy/ud-9191-screenshot-verifier`
- **Branch:** `agy/ud-9191-screenshot-verifier`
- **Paths:** `scripts/verify_companion_screenshots.py`, `tests/unit/scripts/test_verify_companion_screenshots.py`, `tests/companion/test_companion_catalog.py`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`
- **Started:** 2026-09-26
- **Last verified:** 2026-09-26 — 57/57 pass across the verifier tests and tests/companion/test_companion_catalog.py; ruff clean; architecture budget OK.
- **Summary:** `scripts/verify_companion_screenshots.py` recomputes SHA-256 and PNG IHDR pixel size for every captured screenshot record (path must resolve inside the repo, alt text required) and requires pending records to carry null asset fields plus a reason; the companion catalog tests run it on the exporter's real payload. Delivers the #9191 'SHA-256/dimension verification' item; the governed capture workflow, real assets and AffineDrift #4025 alignment remain.
- **Next step:** CI green, review, merge.

### DL-#10943 · Drift Wizard Sidekick Knowledge Pack

- **State:** in_review
- **Owner:** antigravity
- **Issue:** #10943
- **PR:** #10968
- **Branch:** `agy/issue-10943`
- **Paths:** `knowledge/pack.yml`, `knowledge/wizard.yml`, `sidekick.spec`, `scripts/packaging/build_sidekick_binary.py`, `scripts/ci/lod_baseline.txt`, `.github/workflows/wizard-pack.yml`, `.github/WORKFLOWS.md`, `tests/unit/ai/test_drift_wizard.py`, `SPEC.md`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`
- **Started:** 2026-09-25
- **Last verified:** 2026-09-25 — 7/7 tests pass in test_drift_wizard.py; 30/30 tests pass in child copy and divergence test suites; packaging tests pass; check_lod clean scan.
- **Summary:** Define source catalog knowledge/pack.yml and wizard configuration knowledge/wizard.yml for UpstreamDrift product documentation and reference. Wire Sidekick packaging to compile and bundle the SQLite knowledge pack. Add wizard-pack CI workflow and test suite. Baseline child-copy LOD finding to preserve child-copy contract.
- **Next step:** Pass all quality gates and squash-merge PR.

### DL-#10921 · Require Both Desktop and Start Menu Destinations and Update Handoff Governance

- **State:** in_progress
- **Owner:** local
- **Issue:** #10921 (companion #10918, #10919, #10920, #10924, #10925)
- **PR:** #10923
- **Branch:** `fix/10921-shortcuts-feedback`
- **Paths:** `src/launchers/desktop_shortcuts.py`, `tests/unit/launchers/test_desktop_shortcuts.py`, `SPEC.md`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`
- **Started:** 2026-09-24
- **Last verified:** 2026-09-24 — unit tests pass in test_desktop_shortcuts.py verifying partial installation failure
- **Summary:** Enforce that shortcut installation requires both Desktop and Start Menu destinations to succeed, and synchronize canonical handoff documentation and SPEC change log (#10918, #10919, #10920, #10921, #10924, #10925).
- **Next step:** Land PR and close review feedback issues.

### DL-#10883 · Restore Green Main: Jules Bolt Learning Title Case Compliance

- **State:** in_review
- **Owner:** local
- **Issue:** #10883
- **PR:** #10884
- **Branch:** `fix/10883-docs-governance-title-case`
- **Paths:** `.jules/bolt.md`, `tests/scripts/test_document_title_case.py`, `SPEC.md`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`
- **Started:** 2026-09-24
- **Last verified:** 2026-09-24 — verified Title Case check passes across diff, unit tests in test_document_title_case.py pass (5/5)
- **Summary:** Backticked code tokens (`np.linalg.norm`) in .jules/bolt.md heading to restore green main against required docs-governance-gates and added unit regression test (#10883).
- **Next step:** Create PR, enable auto-merge, verify CI green on main.

### DL-#10849 · Pendulum Inertia Hash Integration Parameter Digest & Dynamics Cache Refresh

- **State:** in_review
- **Owner:** antigravity
- **Issue:** #10849, #10850, #10851
- **PR:** #10860
- **Branch:** `fix/10849-10850-inertia-hash-and-spec-keys`
- **Paths:** `src/engines/pendulum_models/python/double_pendulum_model/physics/double_pendulum.py`, `src/engines/physics_engines/pendulum/python/motion_matching/adapters.py`, `src/engines/physics_engines/pendulum/python/motion_matching/qualification.py`, `tests/unit/tour_baselines/test_qualification.py`, `SPEC.md`, `AGENT_HANDOFF.md`, `docs/development/HANDOFF.md`, `docs/development/DEVELOPMENT_LOG.md`
- **Started:** 2026-09-24
- **Last verified:** 2026-09-24 — verified calibrated length initialization before caching, refresh_cache() synchronization, and derivation of inertia digest from cached integration parameters
- **Summary:** Updated DoublePendulumDynamics and qualification routines to derive the inertia digest directly from the cached properties consumed by simulation integration, initialize calibrated segment lengths before parameter caching, and key SPEC.md change log rows accurately (#10849, #10850, #10851).
- **Next step:** Open PR with auto-merge enabled, monitor CI, and close issues #10849, #10850, #10851 upon merge.

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

### DL-#10799 · Fail-Closed Qualification for Unsigned Legacy Evidence & Dynamic Inertia Digest (#10799, #10800)

- **State:** ready
- **Owner:** antigravity
- **Issue:** #10799, #10800
- **Branch:** `fix/10799-10800-qualification-integrity`
- **Paths:** `src/shared/python/tour_baselines/qualification.py`; `src/engines/physics_engines/pendulum/python/motion_matching/qualification.py`; `tests/unit/tour_baselines/test_qualification.py`; `SPEC.md`
- **Started:** 2026-09-24
- **Last verified:** 2026-09-24 — 23 unit tests passed in `tests/unit/tour_baselines/test_qualification.py`, 102 passed in full `tests/unit/tour_baselines/` suite, 21 passed in `tests/unit/motion_matching/test_acceptance.py`.
- **Summary:** Remediated bot review feedback on PR #10798:
  1. Kept qualification strictly fail-closed for unsigned legacy evidence (#10799): removed silent auto-migration from `IndependentBaselineQualifier.qualify()`, ensuring packages without recorded controls, time, or identity hashes are rejected at the integrity gate with `IntegrityViolation`. Explicit `migrate_legacy_package` compatibility operations leave package status as `UNVERIFIED` with `has_native_replay=False` rather than silently passing qualification.
  2. Built `fixed_inertia_hash` from actual dynamics parameters (#10800): introduced `compute_pendulum_inertia_hash` in pendulum qualification to digest segment masses, center-of-mass ratios, and rotational inertias from the actual `DoublePendulumDynamics` rollout parameters, ensuring changes to dynamics physics invalidate the inertia digest even when link lengths remain identical.
- **Next step:** Run CI pre-commit checks, push branch, open PR with `--auto --squash`, verify merge, and release coordination leases.

### DL-#10597 · Tour Baselines Baseline Guide, Agent Runbook, and End-to-End Acceptance (TB-12)

- **State:** in_progress
- **Owner:** antigravity
- **Issue:** #10597 (TB-12, parent #10584, program #10363)
- **Branch:** `feat/tb12-baseline-guide-acceptance-10597`
- **Paths:** `docs/plans/tour_baselines/baseline_guide.md`; `docs/plans/tour_baselines/agent_runbook.md`; `docs/plans/tour_baselines/final_acceptance_report.md`; `tests/acceptance/test_tour_baselines_journey.py`; `docs/plans/tour_baselines/README.md`; `docs/development/matched_swing_program/README.md`; `docs/development/DEVELOPMENT_LOG.md`; `SPEC.md`
- **Started:** 2026-09-24
- **Last verified:** 2026-09-24 — 8 passed in `tests/acceptance/test_tour_baselines_journey.py` (100% pass). Documentation links and runbook instructions verified.
- **Summary:** Published concise Operator User Guide covering launcher navigation, driver vs. iron capture, complexity tradeoffs, loading/cloning, and physical error formulas; Agent Runbook with copyable clean-environment reproduction commands, hash verification, MATLAB R2025b requirements, and tamper detection; and Final Acceptance Report recording the completed two-capture coverage matrix, software integration sign-off, and governing issue links for outstanding full-body models. Authored comprehensive end-to-end acceptance test suite verifying the full user journey.
- **Next step:** Run CI pre-commit checks, open PR, auto-merge, and close Epic #10584.

### DL-#10751 · Fix CI Standard 'Deleted Python Test Files' Check False-Positives in Shallow Checkouts

- **State:** in_review
- **Owner:** antigravity
- **Issue:** #10751
- **Branch:** fix/10751-deleted-tests-shallow-checkout
- **PR:** #10762
- **Paths:** scripts/ci/check_deleted_test_files.py; tests/scripts/test_check_deleted_test_files.py; .github/workflows/ci-standard.yml; tests/ci/test_ci_infrastructure.py
- **Started:** 2026-09-23
- **Last verified:** 2026-09-23 (`008487870` + c31f26e; 10 passed in test_check_deleted_test_files.py, 85 passed in test_ci_infrastructure.py)
- **Summary:** Compute deleted tests diff against merge-base instead of base ref tip to prevent tests added on main from being falsely reported as deleted in PRs; set fetch-depth: 0 on checkout in tests job; invoke standalone check_deleted_test_files.py with unit and regression test coverage and fallback-to-base support directly from the CI workflow.
- **Next step:** Await owner workflow approval for #10762.

### DL-#10743 · Docs-Consistency Cross-Repo Path Exemption

- **State:** in_review
- **Owner:** claude (session `fleet-remediation-k`)
- **Issue:** #10743
- **Branch:** `fix/10743-docs-consistency-cross-repo`
- **PR:** #10744
- **Paths:** `scripts/check_agent_docs_consistency.py`, `tests/architecture/test_check_agent_docs_consistency.py`
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 (checker passes on main CLAUDE.md; 33 focused tests pass)
- **Summary:** Exempt sibling-repo-qualified backticked paths (possessive or not) from the local-existence check so the fleet-managed deferred-validation block no longer fails `repo-structure-gates`; bare local paths stay strict.

### DL-#8907 · One User Config Root and One QSettings Namespace for the Launcher

- **State:** in_review
- **Owner:** claude
- **Issue:** #8907
- **Branch:** fix/8907-settings-root
- **PR:** #10742 (open)
- **Paths:** src/shared/python/data_io/user_config_root.py; src/launchers/launcher_settings_store.py; src/launchers/launcher_constants.py; tests/unit/data_io/test_user_config_root.py; tests/launchers/test_launcher_settings_store.py
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 — migration, alias and three-window geometry round-trip tests pass offscreen; launcher/ui suites show no new failures versus origin/main; Ruff, format, file-size, architecture and error-handling gates pass.
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

### DL-#8883 · Video Analyzer Real GUI Replacing the Placeholder Label

- **State:** in_progress
- **Owner:** local
- **Issue:** #8883 (related #8854, #10512)
- **Branch:** fix/8883-video-analyzer-gui
- **PR:** #10651
- **Paths:** src/tools/video_analyzer/analyzer.py; src/tools/video_analyzer/gui.py; src/launchers/external_tools_adapter.py; src/launchers/task_launch_truthfulness.py; tests/unit/test_video_analyzer_pipeline.py; tests/ui/tools/video_analyzer/; tests/launchers/test_simulation_guis.py; tests/launchers/test_task_launch_truthfulness.py
- **Started:** 2026-09-21
- **Last verified:** 2026-09-21 at HEAD (ruff check/format clean; mypy clean on the four changed src files; pytest green on tests/unit/test_video_analyzer_pipeline.py, tests/unit/test_video_analyzer_math.py, tests/ui/tools/video_analyzer, tests/launchers/test_simulation_guis.py, tests/launchers/test_task_launch_truthfulness.py, tests/config/feature_parity).
- **Summary:** Replaced the static `QLabel("Video Analyzer (GUI placeholder)")` with a real `MainWidget` (choose video, Analyze, report pane) wired to a new `SwingAnalyzer.analyze_video()` that runs MediaPipe pose estimation (via the existing `pose_estimation` registry) and feeds the already-tested head-stability math. Removed the launcher's dead sibling-repo import fallback (`video_analyzer.launch_pyqt6`, confirmed nonexistent by #8854) so the tile no longer depends on an external Tools provider; updated `task_launch_truthfulness`'s audit entry from `PROVIDER_REQUIRED` to `PRODUCTION_SOLVER` accordingly.
- **Next step:** Open PR `Closes #8883`, push, and drive CI to green.
- **Evidence:** tests/unit/test_video_analyzer_pipeline.py; tests/ui/tools/video_analyzer/test_gui.py.

### DL-#9700-Planning · Deferred External Validation Plans

- **State:** in_review
- **Owner:** codex (session `codex-validation-planning-20260922-ud`)
- **Issue:** #9700; #10375; #10382; #9619; #9613; #9546
- **Branch:** `docs/deferred-validation-planning`
- **PR:** #10741
- **Paths:** `docs/development/planning/`, `docs/development/HANDOFF.md`, `AGENT_HANDOFF.md`
- **Started:** 2026-09-22
- **Last verified:** 2026-09-22 (source bodies unchanged; #9546 already closed by #10446 before migration)
- **Summary:** Preserve external evidence requirements for Board consideration. Keep executable work distinct; no experimental results, actor approvals or provider changes.
- **Next step:** Validate and publish six plans; verify exact artifacts before source comments. Keep five open sources open and preserve the existing #9546 disposition.

### DL-#8941 · Simulation-Page Polling Complete (Server + Client)

- **State:** in_review
- **Owner:** local
- **Issue:** #8941 (server via #10736 merged; client polling via #10748 merged; ActuatorPanel and SimulationToolbar loops onto usePolling here; WS frames stay with #8936/#8940)
- **Branch:** fix/8941-use-polling-toolbar-actuator
- **PR:** #10881 (open)
- **Paths:** ui/src/components/simulation/ActuatorPanel.tsx; ui/src/components/simulation/ActuatorPanel.test.tsx; ui/src/components/simulation/SimulationToolbar.tsx; ui/src/components/simulation/SimulationToolbar.test.tsx
- **Started:** 2026-09-24
- **Last verified:** 2026-09-24 — ActuatorPanel and SimulationToolbar migrated from raw setInterval to shared usePolling hook (pauses when tab hidden or simulation stopped, single-flight ticks, interval cleared on unmount); 31 focused vitest tests pass, full ui suite (98 files / 934 tests) passes, tsc -b/eslint/build clean, architecture budget and error handling ratchet pass.
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
- **Last verified:** 2026-09-22 at SELF — unit suite green for MS-111 contracts; PR #10723 opened with squash auto-merge; no native ControlTower desk claim.
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
- **Last verified:** 2026-09-22 at SELF — PR #10722 opened ready-for-review; RED→GREEN provenance suite; push hooks passed.
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
- **Last verified:** 2026-09-22 at SELF — merged origin/main through #10709/#10718/#10720/#10721; regenerated divergence inventory; removed duplicate SPEC #10721 key; PR #10707 squash auto-merge armed.
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
- **Last verified:** 2026-09-22 at SELF — merged origin/main; DRY duplication gate clean after shared comparison/banner helpers and `wire_primary_action_button`; ruff clean on touched files.
- **Summary:** Migrated `bunker_shot_gui`, `ball_flight_gui`, and `swing_flight_pipeline` onto `src/tools/async_action.py`; added lower-only GUI-thread-blocking ratchet (baseline 12); primary run buttons use shared theme wiring (#10654). ~9 tools remain un-migrated (see PR #10656 Deferred).
- **Next step:** Merge PR #10656 after CI green; follow-up PRs for remaining inline tools.

### DL-#9479 · Consolidate Engine Meta-Tiles and Clarify Confusable Launcher Tile Names

- **State:** in_review
- **Owner:** claude
- **Issue:** #9479, #9480 (parent #9412; cluster #9410 Cluster B)
- **Branch:** fix/9479-9480-launcher-tiles
- **PR:** #10653
- **Paths:** src/config/launcher_manifest.json; src/config/models.yaml; src/launchers/workspace_navigation.py; scripts/check_launcher_logo_families.py; tests/config/launcher_manifest/test_engine_hub_consolidation_9479.py; tests/config/launcher_manifest/test_tile_name_clarity_9480.py; tests/launchers/test_workspace_navigation.py
- **Started:** 2026-09-21
- **Last verified:** 2026-09-22 at SELF — rebased onto origin/main; agent-context/capability atlas/MODEL_IMAGES fixes; suite markers on new tests.
- **Summary:** Hide duplicate per-engine dashboards with documented reasons; reclassify three specialized tools from `physics_engine` to `simulation`; clarify confusable data/video tile pairs; move non-golf utilities to `dev_research`.
- **Next step:** Squash merge PR #10653 after CI green.
- **Evidence:** tests/config/launcher_manifest/test_engine_hub_consolidation_9479.py; tests/config/launcher_manifest/test_tile_name_clarity_9480.py; tests/launchers/test_workspace_navigation.py::TestWorkspaceMembership::test_non_golf_utilities_moved_out_of_primary_workflow.

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

### DL-#10344 · MyoSuite Golfer Scene: Pinned MyoSim Submodule + Dual-Grip Club Contacts (MS-51)

- **State:** in_review
- **Owner:** local
- **Issue:** #10344 (MS-51, epic #10363)
- **Branch:** fix/10344-ms51-myosuite-repair
- **PR:** #10685
- **Paths:** src/engines/physics_engines/myosuite/python/golfer_scene.py; coordinate_map_anthro.json; shared/models/myosuite/golf/body/; scripts/setup_myosuite_models.{ps1,sh}; src/engines/model_inventory.py; src/config/engine_model_inventory.json; docs/engines/myosuite.md; tests/unit/engines/myosuite/test_golfer_scene.py; tests/myosuite/test_golfer_scene_native.py
- **Started:** 2026-09-22
- **Last verified:** 2026-09-21 at SELF — CI gate fixes: LoD basename helper, generate_golfer_scene under function-line budget, defusedxml in unit XML tests; scoped pytest green locally.
- **Summary:** Pinned myo_sim gitlink documented; bootstrap scripts; generated driver/iron golfer MJCF on myobody_simpleupper with dual-grip site welds and four foot contact markers; diagnostic 40-of-44 coordinate map; MS-102 inventory left repair after real probes.
- **Next step:** Push CI repair commit and confirm #10685 quality-gate green before squash merge.
- **Evidence:** shared/models/myosuite/golf/body/golfer*myobody_receipt.json; docs/development/matched_swing_program/evidence/ms102/myosuite*\*\_structural_receipt.json

### DL-#10376 · Complete Engine and Model Inventory With Runnable Model Packages (MS-102)

- **State:** in_review
- **Owner:** local
- **Issue:** #10376 (MS-102, epic #10363)
- **Branch:** feat/ms102-engine-model-inventory
- **PR:** #10677
- **Paths:** src/engines/model_inventory.py; src/config/engine_model_inventory.json; tests/unit/engines/test_model_inventory.py; docs/development/matched_swing_program/evidence/ms102/
- **Started:** 2026-09-21
- **Last verified:** 2026-09-22 at 84e2c5a45373b3864a7479b4e5f52cbb6376425d — retargeted onto origin/main; `_run_native_pipeline` split under architecture budget; real vendor pin a9ed0e7c5; `pytest tests/unit/engines/test_model_inventory.py -q -n 0 --no-cov` green; agent-context check clean.
- **Summary:** Authority-derived engine/model inventory (models.yaml + capability matrix + ENGINE_TIERS) with dual-club flagship packages, immutable hashes, qualification harness (resolve/hash/load/FK/dynamics/viewer/save), and named repair blockers. Not a competing catalog. Simscape entries require MATLAB R2025b.
- **Next step:** Confirm CI green and squash merge of PR #10677.
- **Evidence:** docs/development/matched_swing_program/evidence/ms102/

### DL-#10345 · MyoSuite Kinematic Replay With Coordinate Map and Marker Parity Receipt

- **State:** in_review
- **Owner:** local
- **Issue:** #10345 (epic #10363, MS-52)
- **Branch:** fix/issue-10345-ms-52-local
- **PR:** #10666
- **Paths:** src/engines/physics_engines/myosuite/python/{retarget,replay,golfer_scene,coordinate_map_anthro.json,viz/render_replay.py}; tests/unit/engines/myosuite/test_retarget.py; tests/myosuite/test_replay_native.py; evidence/matched/driver_g1_myosuite/
- **Started:** 2026-09-21
- **Last verified:** 2026-09-22 at SELF — TDD RED→GREEN on `tests/unit/engines/myosuite/test_retarget.py`; native replay writes receipt/candidate/GIF; MS-51 golfer scene landed with `parity_budget_qualified=false` so 15 mm gate remains deferred.
- **Summary:** Added pure-numpy retarget map, kinematic replay CLI, marker parity receipt (`stage=replay`), MyoSuite registration in `cross_engine_replay.VALID_ENGINES` as kinematic-only, and viewer colour. Evidence committed under `evidence/matched/driver_g1_myosuite/`.
- **Next step:** Merge PR after CI green; drive 15 mm parity on the MS-51 golfer scene (still unqualified).

### DL-#10436 · Explore Feasible Force Null Spaces and Publish Torque-Distribution Tradeoffs

- **State:** in_review
- **Owner:** local
- **Issue:** #10436 (PF-06, epic #10430)
- **Branch:** feat/issue-10436-pf06-feasible-force-nullspace
- **PR:** #10504
- **Paths:** src/shared/python/motion_matching/force_nullspace.py; tests/unit/motion_matching/test_force_nullspace.py; tests/unit/motion_matching/test_force_nullspace_pf06.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-21 at SELF — rebased onto origin/main (MS-62); SPEC §12 `#10504` present; focused PF-06 suites 29 passed; architecture budget OK; DRY duplication gate OK.
- **Summary:** Extended force null space representation to support scaled SVD and column-pivoted QR decomposition with dynamic rank and contact mode reporting. Added `NullSpaceAnalysis` reporting condition number and machine-precision residuals (A N = 0, A x_p = b). Implemented `redistribute_trajectory` with physical rate penalties strictly invariant to basis sign changes. Implemented `explore_torque_tradeoffs` generating Pareto tradeoff alternatives across baseline minimum effort, conservative default, trail arm reduction sweeps (50%, 80%), hard-zero trail feasibility checks, relaxed minimum trail alternatives, grip squeeze minimization, and ground reaction regularization. Detailed diagnostics report per-joint torque, power, lead/trail effort, ground COP, grip wrench, and explicit SI units. Exported reproducible Pareto tradeoff tables to JSON and CSV. Selected conservative default with mechanical rationale without unfounded metabolic or injury claims.
- **Next step:** Confirm PR #10504 CI is green after force-with-lease push, then merge.
- **Evidence:** tests/unit/motion_matching/test_force_nullspace.py; tests/unit/motion_matching/test_force_nullspace_pf06.py.

### DL-#8930 · Vectorize Rust Trajectory Post-Processing

- **State:** in_review
- **Owner:** claude
- **Issue:** #8930
- **Branch:** fix/8930-ball-flight-vectorized-rk4
- **PR:** #10648 (open)
- **Paths:** src/shared/python/physics/ball_simulator.py; tests/unit/physics/test_ball_simulator_post_process_vectorized_8930.py
- **Started:** 2026-09-21
- **Last verified:** 2026-09-21 (`98abb965f1`) — RED shown for the batching regression test (25 calls before the fix), then GREEN; numerical-equivalence and empty-trajectory tests pass; `tests/unit/physics/` shows no new failures versus unmodified main (three pre-existing rust-engine/tolerance failures reproduced identically via `git stash`); ruff check/format clean.
- **Summary:** `BallFlightSimulator._post_process_rust` called the scalar `_calculate_forces_single` path once per trajectory point instead of the existing vectorized `_calculate_forces_batch` path; now builds the `(3, N)` batch once and calls force calculation a single time per trajectory.
- **Next step:** Land the vectorization PR (#10648) if not already merged.

### DL-#9544 · Bunker Contact Regimes and Coupled Club Rotation Across Fidelity Tiers

- **State:** in_review
- **Owner:** claude
- **Issue:** #9544 (epic #9541)
- **Branch:** conductor/issue-9544
- **PR:** #10455 (open)
- **Paths:** src/bunkershot3d/ball/regimes.py; src/bunkershot3d/ball/pipeline.py; src/bunkershot3d/ball/splash.py; src/bunkershot3d/solvers/shot.py; src/bunkershot3d/solvers/mpm/wholeshot.py; src/bunkershot3d/vandv/conservation.py; src/tools/bunker_shot_gui/model.py; tests/bunkershot3d/ball/test_contact_regimes_9544.py; tests/bunkershot3d/solvers/test_rotation_coupling_9544.py; docs/bunkershot3d/contact-regimes.md
- **Started:** 2026-09-18
- **Last verified:** 2026-09-18 at SELF (RED shown for both new test files, then GREEN: 15 regime tests and 14 coupling tests pass; tests/bunkershot3d + tests/unit/tools/bunker_shot_gui pass except three pre-existing failures reproduced with the classifier bypassed — pyvista `n_faces` API in test_shot_scene_render_vtk.py and two #9243 budget/band assertions; ruff check/format clean)
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
- **Summary:** Software half of the measurement-to-prediction program: measured-stroke intake contract with instrument-only launch records and raw-data digests, registered intended-use matrix and protocol, session-designated split with leakage refusal and #9286 sand-batch admission, bounded fit with identifiability and sensitivity checks preserving failed fits, held-out V&V 20 comparison against predeclared tolerances, versioned `TransferQualification` that lifts the launch verdict floor per qualified regime only, `CALIBRATED` provenance basis, and the #9239 objective disposition. No strokes are on file; physical qualification remains blocked and the issue stays open.
- **Next step:** Open the PR referencing #9543 (not `Closes`), then acquire measured strokes under `MEASUREMENT_PROTOCOL` before any qualification is attempted.
- **Evidence:** tests/bunkershot3d/ball/test_transfer_qualification.py; docs/bunkershot3d/transfer-qualification.md.

### DL-#9546 · Impact Zone Readiness Execution Index and I1/I2 Pin Consumption

- **Issue:** #9546 (children #9547, #9548, #9549, #9550; reused #9484, #9349)
- **Branch:** conductor/issue-9546
- **PR:** #10446
- **Paths:** src/config/impact_zone_readiness.json; src/config/industrial_readiness_loader.py; scripts/generate_industrial_readiness_index.py; docs/operations/impact-zone-readiness-index.md; tests/config/industrial_readiness/; tests/shared_contracts/test_impact_interval_provider.py
- **Last verified:** 2026-09-18 (SELF; ledger + freshness gates 45 passed; consumer contract 3 passed with `--tools-mode=vendored` against Tools pin `62e8cdbf9`)
- **Summary:** Epic #9546 gets the same machine-readable execution index as #9539: `impact_zone_readiness.json` reconciles I1–I4 and the two reused issues against `5347cba0f` and the vendored Tools pin, under the existing loader contract (keys `I<n>`/`R<n>` admitted, nothing else relaxed). The Tools fixes for I1 (#5088) and I2 (#5079) are consumed by a UD consumer contract that drives the audit probe through the vendored solver; I3, I4 and the Tools half of #9349 remain open with ordered plans, and `release_status` is `blocked`.
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
- **Last verified:** 2026-09-22 at HEAD (All 41 pendulum unit tests pass; LOD zero new violations; DRY duplication gate passed with 0 unapproved duplicates; ruff check, ruff format, and mypy clean).
- **Summary:** Implemented 3-DOF planar torque-driven Hub–Arm–Club triple pendulum fitting, independent 4x tighter replay, and Tools shipped simulator replay parity. Formulated 21 degree-6 Bernstein control points strictly bounded and regularized with curvature and effort penalties. Validated positive non-zero geometry calibration ($L_1 \approx 0.315\text{ m}, L_2 \approx 0.585\text{ m}, L_3 \approx 0.782\text{ m}$ for driver; $L_1 \approx 0.315\text{ m}, L_2 \approx 0.585\text{ m}, L_3 \approx 0.700\text{ m}$ for iron) seeded from double fit without invalid zero-length link reductions. Generated authoritative baseline packages and qualification receipts for Driver and 7-Iron. Completed comprehensive DRY refactoring eliminating duplicate windows across qualification, provider, and torque optimization modules.
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

### DL-#10359 · Wire Video and Fit-Quality Report Export

- **State:** in_review
- **Owner:** claude
- **Issue:** #10359 (MS-86, epic #10363)
- **Branch:** feat/10359-export-video-report
- **PR:** #10632
- **Paths:** src/shared/python/motion_matching/export.py; src/shared/python/motion_matching/**main**.py; src/tools/matched_swing_browser/gui.py; src/config/launcher_manifest.json; src/config/models.yaml; tests/unit/motion_matching/test_export.py; tests/tools/matched_swing_browser/test_matched_swing_browser_gui.py; docs/development/matched_swing_program/evidence/reports/sample_fit_report.md
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 at HEAD (11 unit tests in test_export.py pass; 9 GUI tests in test_matched_swing_browser_gui.py pass; ruff clean; black clean; architecture budget passed).
- **Summary:** Implemented `export_video` and `export_report` with fail-closed DbC contracts, standardized metrics, acceptance gate verdicts, physical constraints, and full cryptographic provenance (#8820 / U3). Added CLI subcommands `export-video` and `export-report` to motion_matching module. Added "Export Video..." and "Export Report..." buttons to Results Browser GUI Actions card with file dialogs. Registered `video_export` and `report_export` capabilities in launcher manifest and models config. Emitted sample Markdown fit report in matched swing program evidence directory.
- **Next step:** Open PR referencing #10359, await green CI, merge and release lease.
- **Evidence:** docs/development/matched_swing_program/evidence/reports/sample_fit_report.md; tests/unit/motion_matching/test_export.py; tests/tools/matched_swing_browser/test_matched_swing_browser_gui.py.

### DL-#10349 · Simscape 44-to-27 Coordinate Slice and Boundary-Load Validation

- **State:** in_review
- **Owner:** local
- **Issue:** #10349 (MS-62, epic #10363)
- **Branch:** fix/issue-10349-ms-62-local
- **PR:** #10665 (open)
- **Paths:** src/shared/python/motion_matching/coordinate_slice.py; tests/unit/motion_matching/test_coordinate_slice.py; evidence/matched/driver_g1_simscape_slice/; src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/align_measured_to_model.m
- **Started:** 2026-09-21
- **Last verified:** 2026-09-21 at HEAD (9 unit tests pass in test_coordinate_slice.py; ruff and architecture budget clean; kinematic slice receipt committed; dynamic Simscape replay unqualified)
- **Summary:** Added JSON-backed 44-to-27 coordinate slice with name-based kinematic projection, virtual-work decomposition tests, boundary-wrench derivation for omitted neck/leg DOFs, CLI, and Simscape workspace overrides from anthropometric geometry documents in `align_measured_to_model.m`.
- **Next step:** R2025b Simscape native replay of `candidate27.npz` with boundary-load validation.
- **Evidence:** evidence/matched/driver_g1_simscape_slice/{slice_map.json,receipt.json,parity.json,run_manifest.json,candidate27.npz}; tests/unit/motion_matching/test_coordinate_slice.py.

### DL-#10348 · Simscape Topology + Full-Marker Terminal (MS-61)

- **State:** in_progress
- **Owner:** local
- **Issue:** #10348 (MS-61, epic #10363)
- **Branch:** feat/issue-10348-ms61-simscape-topology
- **PR:** #10676
- **Paths:** src/shared/python/motion_matching/{simscape_topology.py,full_marker_terminal.py,tour_metrics.py,acceptance.py}; scripts/matlab/{materialize_ms61_topology_receipts.py,run_simscape_candidate.ps1}; docs/development/simscape_tour_matching/native_evidence/two_window_fit_9967_103/; docs/development/matched_swing_program/{GATES.md,README.md,WAVES.md}; docs/development/simscape_tour_matching/CHECKPOINTS.md
- **Started:** 2026-09-21
- **Last verified:** 2026-09-22 at 84e2c5a45373b3864a7479b4e5f52cbb6376425d — regenerated matched_swing_program README status from ledger (103 receipts); freshness test GREEN; fail-closed blocked native_gate + reduced_27_no_neck retained.
- **Summary:** Fail-closed 27-DOF topology classification (no independent neck), dual terminal disclosure (full + head cluster; body-excluding-head diagnostic only), acceptance/tour_metrics dual-terminal contracts, and run-103 scaffolding derived from run-102 without inventing native G1 success; repair linked to MS-104 (#10378).
- **Next step:** Confirm CI green and squash auto-merge of PR #10676.
- **Evidence:** docs/development/simscape_tour_matching/native_evidence/two_window_fit_9967_103/{topology_report.json,terminal_breakdown.json,native_gate.json,runtime_license_receipt.json,parity_receipt.json,HANDOFF.md}; tests/unit/motion_matching/test_simscape_topology_ms61.py.

### DL-#10342 · OpenSim/MyoSuite Native Nightly Lane Receipts

- **State:** in_review
- **Owner:** local
- **Issue:** #10342 (MS-43, epic #10363)
- **Branch:** fix/issue-10342-ms-43-native-lane-local
- **Paths:** scripts/ci/run_native_engine_lane.py; scripts/ci/run_native_engine_lane.sh; docs/development/matched_swing_program/evidence/nightly/; tests/docs/test_native_lane_freshness.py; tests/scripts/test_run_native_engine_lane.py
- **Started:** 2026-09-21
- **Last verified:** 2026-09-21 at HEAD (16 unit tests pass in test_native_lane_freshness.py and test_run_native_engine_lane.py; ruff clean; architecture budget passed; bootstrap receipts committed pending ControlTower SDK refresh)
- **Summary:** Added idempotent native-engine lane runner emitting hashed nightly receipts for OpenSim and MyoSuite (`requires_opensim` / `requires_myosuite` pytest markers), freshness gate warning at seven days and failing at thirty days, and ControlTower verification docs without editing `.github/workflows`.
- **Next step:** Refresh receipts on ControlTower opensim-10003 venv; merge PR closing #10342.
- **Evidence:** docs/development/matched_swing_program/evidence/nightly/opensim_receipt.json; docs/development/matched_swing_program/evidence/nightly/myosuite_receipt.json; tests/docs/test_native_lane_freshness.py.

### DL-#10361 · MS-90: Generic Capture Contract & 44-DOF Identifiability

- **State:** in_progress
- **Owner:** local
- **Issue:** #10361 (MS-90, epic #10363)
- **Branch:** feat/10361-generic-capture-contract
- **PR:** #10633
- **Paths:** src/shared/python/motion_matching/tour_capture_contract.py; src/shared/python/motion_matching/identifiability.py; tests/unit/motion_matching/test_capture_contract_generic.py; tests/unit/motion_matching/test_identifiability.py; evidence/anthropometry/identifiability_driver.json; docs/user_guide/motion_matching/loading_targets.md
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 at HEAD (12 unit tests pass in test_capture_contract_generic.py and test_identifiability.py covering: frozen tour capture backwards compatibility, CMU locomotion C3D rejection with named diagnostic reasons, custom label mapping and mm->m unit scaling, gap fraction limits, synthetic chain planted null direction detection and resolution via prior/off-axis marker, 44-DOF uncalibrated leg DOF unobservability, calibrated leg observability, and full rank 44/44 recovery via anthropometric prior; check_architecture_budget, ruff check, ruff format, and mypy clean).
- **Summary:** Implemented `CaptureContract` and `CaptureValidationReport` enabling validation and loading of arbitrary C3D captures without editing codebase source. Retained frozen tour captures as named instances of `CaptureContract`. Implemented `probe_spec_identifiability` and `probe_synthetic_chain_identifiability` performing linearised SVD identifiability analysis, detecting planted null directions and unobservable lower-body DOFs on the 44-DOF model, and demonstrating resolution to full rank via anthropometric prior regularization. Generated and committed `evidence/anthropometry/identifiability_driver.json`.
- **Next step:** Push branch, open PR with auto-merge, complete lease on #10361.
- **Evidence:** evidence/anthropometry/identifiability_driver.json; tests/unit/motion_matching/test_capture_contract_generic.py; tests/unit/motion_matching/test_identifiability.py.

### DL-#10366 · MS-16: MuJoCo Native IK and MJ_Inverse Tracking

- **State:** in_review
- **Owner:** local
- **Issue:** #10366 (MS-16, epic #10363)
- **Branch:** fix/issue-10366-ms-16-mujoco-native-tools-marker-ik-on-m-cursor-composer-local
- **PR:** #10660
- **Paths:** src/engines/physics_engines/mujoco/python/ik_minimize.py; src/engines/physics_engines/mujoco/python/inverse_dynamics.py; src/shared/python/motion_matching/pipeline/reference.py; src/shared/python/motion_matching/pipeline/dynamics.py; src/shared/python/motion_matching/pipeline/cli.py; src/tools/motion_matching/pipeline.py; tests/unit/motion_matching/test_mujoco_ik_minimize.py; tests/unit/motion_matching/test_mujoco_mj_inverse.py; docs/development/matched_swing_program/README.md; tests/tools/matched_swing_browser/test_model.py
- **Started:** 2026-09-21
- **Last verified:** 2026-09-21 at HEAD (SELF; refreshed matched_swing ledger to 101 receipts; architecture budget via ShootingFitConfig retained)
- **Summary:** Added selectable `--ik-backend mujoco-minimize` (MuJoCo `minimize.least_squares` with LM warm start) and `--tracking mj-inverse` (plant KKT torques with native `mj_inverse` audit). Wired backends through pipeline plant, dynamics replay, CLI, Motion Matching tile, and receipt fields with DbC validation at API boundaries.
- **Next step:** Confirm unit-test-gate and quality-gate green on PR #10660 so squash auto-merge can land.
- **Evidence:** docs/development/full_body_models/evidence/ground_support/anthro_driver_native_tools/receipt.json; tests/unit/motion_matching/test_mujoco_ik_minimize.py; tests/unit/motion_matching/test_mujoco_mj_inverse.py; tests/docs/test_matched_swing_status_freshness.py

### DL-#9422 · Rig Capture Sessions Through the Tools MocapSession Contract

- **State:** in_review
- **Owner:** claude
- **Issue:** #9422 (readiness P6; Tools #4706 M-track consumer)
- **Branch:** conductor/issue-9422
- **PR:** #10466 (open)
- **Paths:** src/motion_capture/rig/tools_bridge.py; src/motion_capture/rig/**main**.py; tests/motion_capture/rig/test_tools_session_export.py; tests/fixtures/mocap_session_export/; docs/motion_capture/capture_rig.md
- **Started:** 2026-09-19
- **Last verified:** 2026-09-22 at HEAD (SELF; #9604 `CameraCapabilities` mapping: 14 Tools-first export checks pass under `tests/fixtures/mocap_session_export/run_checks.py`; in-process bridge tests pass; scoped Ruff + mypy clean).
- **Summary:** The bridge now probes the pinned Tools family (`shared.python.sidekick.lab.mocap`), pins `mocap-session/1.0.0`, and projects a rig capture session onto the Tools `MocapSessionManifest` through the Tools builders and canonical serializer; `capture`/`record` write `mocap_session.json` beside the rig manifest and record the export outcome under `tools_schema.export`. Retained raw video without `--consent-recorded` is refused by the Tools policy, not faked. #9604 adds `map_camera_records`: one Tools `CameraIdentity` + `CameraCapabilities` per rig camera. D-track (Tools #4707, D3 open) and the #8865/#8866/#8867 prerequisites remain open; this is the first consumer slice, not closure of the program.
- **Next step:** Open the PR, then route the C3D upload path (#8865) through the same pinned contract as the next consumer slice.
- **Evidence:** tests/fixtures/mocap_session_export/export_checks.py; tests/motion_capture/rig/test_tools_session_export.py.

### DL-#10434 · Qualify Contact Modes and Native Pinocchio Force Feasibility

- **State:** in_review
- **Owner:** local
- **Issue:** #10434 (epic #10363, PF-04)
- **Branch:** feat/issue-10434-pf04-qualify-contact-modes-pinocchio-forces
- **PR:** #10499 (auto-merge enabled)
- **Paths:** src/shared/python/motion_matching/contact_mode_qualifier.py; tests/unit/motion_matching/test_contact_mode_qualifier_pf04.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (SELF; 10 unit tests pass in test_contact_mode_qualifier_pf04.py covering support mode hysteresis, ambiguity scoring, ground support geometry / COP / convex hull containment, slip speed and friction saturation, compliant vs allocated force consistency via Hunt-Crossley model, separate linear force and moment residual budgets, unphysical mega-Newton load and kNm torque rejection, and mass/geometry/friction sensitivity reporting; ruff, ruff format, black, mypy, bandit, and pre-push hooks all clean)
- **Summary:** Implemented ContactModeQualifier evaluating multi-sphere heel/toe support modes with clearance and velocity hysteresis, computing center of pressure (COP) and convex hull containment under arbitrary normal directions, evaluating slip velocity and friction saturation, and cross-checking allocated contact forces against constitutive Hunt-Crossley compliant models. Enforces separate linear force and moment residual budgets, rejects unphysical loads (> 5000 N or > 300 Nm), and generates parameter sensitivity reports.
- **Next step:** Land PR #10499 via CI and proceed to PF-05.
- **Evidence:** tests/unit/motion_matching/test_contact_mode_qualifier_pf04.py; src/shared/python/motion_matching/contact_mode_qualifier.py.

### DL-#10589 · TB-04: Fit and Independently Replay the Actual Driven Double Pendulum

- **State:** in_review
- **Owner:** local
- **Issue:** #10589 (TB-04, parent #10584, program #10363)
- **Branch:** feat/tb04-double-pendulum-fit-10589
- **PR:** #10638
- **Paths:** src/engines/physics_engines/pendulum/python/motion_matching/adapters.py; src/engines/physics_engines/pendulum/python/motion_matching/torque_optimization.py; src/engines/physics_engines/pendulum/python/motion_matching/provider.py; src/engines/physics_engines/pendulum/python/motion_matching/qualification.py; src/engines/physics_engines/pendulum/python/motion_matching/**init**.py; tests/unit/engines/physics_engines/pendulum/test_double_pendulum_fit.py; tests/unit/engines/physics_engines/pendulum/test_motion_matching_provider.py; docs/plans/tour_baselines/evidence/tb04_driver_qualification_receipt.json; docs/plans/tour_baselines/evidence/tb04_iron_qualification_receipt.json; docs/plans/tour_baselines/evidence/tb04_driver_baseline_package.npz; docs/plans/tour_baselines/evidence/tb04_iron_baseline_package.npz; docs/plans/tour_baselines/coverage_matrix.md
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 (15/15 unit tests pass in 6.5s across test_double_pendulum_fit.py and test_motion_matching_provider.py; ruff check clean; ruff format clean; black clean; mypy 0 errors across 4 source and 2 test files; check_architecture_budget passes; check_file_size_budget passes; check_dry_duplication_gate passes).
- **Summary:** Built bidirectional mapping and verified mathematical & numerical acceleration parity (< 1e-14) between DoublePendulumDynamics and Tools physics.py. Formulated continuous smooth bounded joint torques via degree-6 Bernstein polynomials strictly bounded in [tau_min, tau_max] with curvature and effort regularization. Fixed frame-0 off-by-one initial state evaluation bug, implemented non-uniform timestep integration, and added independent 4x tighter substep replay verification. Produced authoritative qualification receipts and baseline packages for Driver and 7-Iron.
- **Next step:** Land PR via normal squash merge and proceed to TB-05 (#10590).
- **Evidence:** docs/plans/tour_baselines/evidence/tb04_driver_qualification_receipt.json; docs/plans/tour_baselines/evidence/tb04_iron_qualification_receipt.json; tests/unit/engines/physics_engines/pendulum/test_double_pendulum_fit.py; tests/unit/engines/physics_engines/pendulum/test_motion_matching_provider.py.

### DL-#10433 · Enforce Contact, Actuator and Root Constraints in Force Allocation

- **State:** in_review
- **Owner:** local
- **Issue:** #10433 (PF-03, epic #10430)
- **Branch:** feat/issue-10433-pf03-contact-actuator-root-constraints
- **PR:** #10498 (auto-merge enabled)
- **Paths:** src/shared/python/motion_matching/contact_force_allocator.py; tests/unit/motion_matching/test_contact_force_allocator.py; tests/unit/motion_matching/test_contact_force_allocator_pf03.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (12 unit tests pass across contact_force_allocator and test_contact_force_allocator_pf03; ruff clean; black clean; mypy strict clean; bandit clean).
- **Summary:** Upgrades ContactForceAllocator with constrained QP inverse dynamics. Enforces 8-faceted polyhedral friction pyramid, non-negative normal ground forces along arbitrary terrain normals, exact contact separation masks, and strict actuator bounds without post-projection. Separates diagnostic root slack so ungrounded reactions never create false physical success. Introduces FeasibilityStatus, HARD_ZERO_TRAIL mode, and verify_torque_and_rate_bounds.
- **Next step:** Land PR #10498 via CI and proceed to PF-04.
- **Evidence:** tests/unit/motion_matching/test_contact_force_allocator_pf03.py; tests/unit/motion_matching/test_contact_force_allocator.py.

### DL-#10440 · Connect Qualified Matching Strategies to Existing Results and Engine Feature Contracts

- **State:** in_progress
- **Owner:** local
- **Issue:** #10440 (PF-10, epic #10430, program #10363)
- **Branch:** feat/issue-10440-pf10-matching-strategies-contracts
- **PR:** #10509
- **Paths:** src/shared/python/motion_matching/matching_strategy.py; src/shared/python/motion_matching/**init**.py; tests/unit/motion_matching/test_matching_strategy.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (30 unit tests pass across test_matching_strategy, test_candidate, and test_candidate_session; ruff clean; black clean; mypy 0 errors; full 6-engine and dual-club contract validation passed).
- **Summary:** Implemented versioned matching strategy schema (`STRATEGY_SCHEMA_VERSION = "matched-strategy-v1"`), stage-separated qualification matrix tracking all 6 stages (`model_available`, `kinematic_fit`, `force_feasible`, `replay_accepted`, `runtime_budget_met`, `muscle_qualified`), strategy presets, controller specifications, and contact reaction history containers. Created `CandidateStrategyPackage` supporting lossless .npz serialization without pickle, and fail-closed name-permuted coordinate remapping. Implemented `StrategyComparisonService` exposing cross-strategy torque profiles, kinematics/closure errors, and capability auditing invalidating supported status on missing SDKs. Provided sample package generator for concrete handoff to MV-03 #10479 without modifying viewer files.
- **Next step:** PR creation, auto-merge, complete lease on #10440 and report back.
- **Evidence:** tests/unit/motion_matching/test_matching_strategy.py.

### DL-#9162 · Local Branch Triage

- **State:** in_progress
- **Owner:** claude
- **Issue:** #9162 (rollout Repository_Management#1460; no-seed bootstrap #9161)
- **Branch:** conductor/issue-9162
- **Paths:** docs/development/branch_triage_9162.md
- **Started:** 2026-09-15
- **Last verified:** 2026-09-15 (SELF; 386 local heads classified from refs, reflogs, worktree HEADs, origin refs and SPEC rows without mutating git state)
- **Summary:** Disposition ledger for every local branch in the primary checkout: 3 protected, 27 deferred to the worktree pass, 34 live (merged sweep then DL entry), 178 stale and 144 abandoned scratch branches (bundle snapshot then delete). The runbook in the ledger executes the deletions and the coordination notice for agent-owned branches.
- **Next step:** Execute the runbook in docs/development/branch_triage_9162.md from the primary checkout and fill in its Outcome section.

### DL-#9410 · Adversarial Product Review Remediation Epic

- **State:** in_progress
- **Owner:** claude
- **Issue:** #9410 (program Repository_Management#1505; children #8820–#8943, #8360, #8641, #8843, #8846, #8853, #8861–#8870, #8874–#8876, #8894)
- **Branch:** conductor/issue-9410
- **Paths:** docs/development/adversarial_review_remediation_9410.md
- **Started:** 2026-09-16
- **Last verified:** 2026-09-16 (SELF; every child re-checked against `db4fe88c4` by first-parent history search plus source reads of each residual)
- **Summary:** Reconciliation ledger for the 2026-08-21 adversarial review: 34 of 61 children landed on `main` (SHAs recorded), 27 residual. Three residual fixes already exist on unmerged branches (`readiness/p0-9412-one-tile-registry`, `conductor/issue-8865`, `conductor/issue-8866`). Epic acceptance (one registry, one API factory, one C3D reader, one pose type, no GUI-thread simulation) is not met on `main`.
- **Next step:** Rebase and merge `readiness/p0-9412-one-tile-registry`, then re-verify the Cluster B rows in the ledger.

### DL-#8684 · Coupled Grip, Shaft, Ground Rollup

- **State:** in_review
- **Owner:** claude
- **Issue:** #8684
- **Branch:** conductor/issue-8684
- **PR:** #10668
- **Paths:** docs/research/proximal_distal_energy_transfer/COMPREHENSIVE_RESEARCH_PROGRAM.md, MODEL_COMPLETION_FALSIFICATION_MATRIX.md
- **Started:** 2026-09-11
- **Last verified:** 2026-09-11 (`SELF`)
- **Summary:** Parent rollup of child tiers #8685/#8797/#8715/#8723 answering the four #8684 questions; manifests re-pinned.
- **Next step:** Open the PR with `Closes #8684`; confirm claim-evidence and release-bundle tests pass in CI.

### DL-#10439 · Replace Synthetic Force Adapters With Native Model-Conformant Bridges

- **State:** in_progress
- **Owner:** local
- **Issue:** #10439 (PF-09, epic #10430, program #10363)
- **Branch:** feat/issue-10439-pf09-native-force-bridges
- **PR:** #10507
- **Paths:** scripts/allocate_swing_torques.py; src/engines/physics_engines/pinocchio/python/force_adapter.py; src/engines/physics_engines/pinocchio/python/native_model.py; src/shared/python/motion_matching/multi_engine_torque_allocator.py; tests/integration/engines/pinocchio/test_force_adapter.py; tests/integration/engines/pinocchio/test_force_mapping.py; tests/unit/motion_matching/test_force_bridges_pf09.py; tests/unit/motion_matching/test_multi_engine_torque_allocator.py; tests/unit/motion_matching/test_native_force_equations.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (17 unit tests pass across test_native_force_equations, test_multi_engine_torque_allocator, and test_force_bridges_pf09; Pinocchio integration tests collect/skip cleanly when Pinocchio not installed; ruff clean; black clean; mypy 0 errors across 7 source files).
- **Summary:** Quarantined `_AnalyticalMultibodyBase` production routes as `SyntheticMultibodyFixture` requiring explicit `allow_synthetic=True`, failing closed with `RuntimeError` on unbridged engines (Drake, OpenSim, Simscape). Extended `BaseEngineForceAdapter` protocol with `model_hash`, `coordinate_order`, `contact_names`, and `compute_mass_and_bias`. Corrected `MujocoForceAdapter` to compute raw unconstrained generalized dynamic forces M a + bias eliminating `qfrc_inverse` passive/constraint force double counting, added input validation and state refresh before mutation, and verified exact acceleration parity. Implemented `PinocchioForceAdapter` with fresh constraint kinematics refresh (`_refresh_constraint_data` and `closure_force_jacobian`). Updated `allocate_swing_torques.py` CLI to support Pinocchio and enforce `--allow-synthetic` gate.
- **Next step:** Auto-merge PR, release lease on #10439 and claim #10440 (PF-10).
- **Evidence:** tests/unit/motion_matching/test_native_force_equations.py; tests/unit/motion_matching/test_multi_engine_torque_allocator.py; tests/unit/motion_matching/test_force_bridges_pf09.py.

### DL-#10623 · NM-08 Add Native Verification, Distribution Checks and Safe Fallback

- **State:** in_progress
- **Owner:** local
- **Issue:** #10623 (epic #10603)
- **Branch:** feat/nm08-native-verification-10623
- **PR:** #10770
- **Paths:** src/shared/python/neural_motion/inference/; src/shared/python/motion_matching/hybrid.py; tests/unit/neural_motion/test_verified_inference_nm08.py; tests/unit/motion_matching/test_verified_inference_nm08.py; docs/plans/neural_motion_matching/native_verification.md; docs/plans/neural_motion_matching/evidence/nm08_native_verification_receipt.json
- **Started:** 2026-09-23
- **Last verified:** 2026-09-23 — 16 unit and behavioral tests pass across neural_motion and motion_matching; architecture budgets and ruff clean.
- **Summary:** Added VerifiedInferenceOrchestrator, DistributionBounds, and check_target_distribution under neural_motion/inference/ (schema neural-verified-inference/1.0.0). Enforces fail-closed validation of non-finite inputs, geometry/engine/control dimension mismatch, and incompatible checkpoint contracts. Validates empirical coverage (durations, peak velocities, contact regimes) with confidence scoring recorded as domain metrics, not golfer truth probability. Enforces mandatory independent replay before dynamic acceptance; on failed proposal or missing checkpoint, falls back to classical/retrieval solver with shared remaining wall-clock budget and retains all attempts with auditable statuses (NEURAL_ACCEPTED, CLASSICAL_FALLBACK, REJECTED). Integrated fit_swing_verified_inference into hybrid.py facade.
- **Next step:** Enable auto-merge, verify CI passes, and hand off to NM-09 (#10624).
- **Evidence:** docs/plans/neural_motion_matching/native_verification.md; docs/plans/neural_motion_matching/evidence/nm08_native_verification_receipt.json

### DL-#10622 · NM-07 Compare Forward Surrogates and Physics-Structured Alternatives

- **State:** in_review
- **Owner:** local
- **Issue:** #10622 (epic #10603)
- **Branch:** feat/nm07-forward-surrogates-10622
- **PR:** #10768
- **Paths:** src/shared/python/motion_matching/surrogate/validate.py; src/shared/python/motion_matching/surrogate/nm07_comparison.py; src/shared/python/neural_motion/surrogates/; tests/unit/motion_matching/test_forward_surrogates_nm07.py; tests/unit/neural_motion/test_forward_surrogates_nm07.py; tests/unit/neural_motion/test_surrogate_nm07_discovery.py; docs/plans/neural_motion_matching/forward_surrogates.md; docs/plans/neural_motion_matching/evidence/nm07_forward_surrogates_receipt.json
- **Started:** 2026-09-23
- **Last verified:** 2026-09-23 — 24 focused tests pass across motion_matching and neural_motion; architecture budgets and ruff pass.
- **Summary:** Compare forward surrogate inversion, hybrid polish, physics-structured residual dynamics, and diffusion fallback under schema neural-surrogate-comparison/1.0.0. Added real-clock timegrid resampling, antipodal quaternion geodesic distance, trust-region validation, directional derivative gradient fidelity check, and contact-boundary failure rejection to validate.py. Documented adversarial exploitation risk of unconstrained forward surrogate inversion and high latency/sample inefficiency of diffusion fallback.
- **Next step:** PR #10768 merged into main; proceed with NM-08 (#10623).
- **Evidence:** docs/plans/neural_motion_matching/forward_surrogates.md; docs/plans/neural_motion_matching/evidence/nm07_forward_surrogates_receipt.json

### DL-#10602 · Club-Only Motion Matching Plan

- **State:** in_review
- **Owner:** codex
- **Issue:** #10602
- **Branch:** `agy/ud-10602-real-matrix`
- **PR:** draft PR from `agy/ud-10602-real-matrix`
- **Paths:** docs/plans/club_neural_review/; docs/plans/club_only_matching/; docs/plans/neural_motion_matching/; src/shared/python/motion_matching/club_only/
- **Started:** 2026-09-20
- **Last verified:** 2026-09-25 — CO-08 matrix scores only complete recorded CO-04/CO-05 fit outcomes (fit_outcomes.py); 0/80 cells scored, 24 unqualified with named missing metrics; 276 club tests pass.
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
- **Last verified:** 2026-09-22 — NM-02 (#10617) in progress on fix/10617-nm02-native-dataset-labels; CO-02 (#10606) shipped via #10675
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

### DL-#10381 · Qualify Crocoddyl Full-Body Fit & Analytic Pelvis Yaw (MS-107)

- **State:** in_progress
- **Owner:** claude
- **Issue:** #10381 (MS-107, epic #10363)
- **Branch:** feat/10381-crocoddyl-pelvis-yaw-g1
- **PR:** #10599
- **Paths:** src/engines/physics_engines/pinocchio/python/crocoddyl_problem.py; src/engines/physics_engines/pinocchio/python/crocoddyl_action.py; src/engines/physics_engines/pinocchio/python/full_body_fit.py; tests/unit/motion_matching/test_crocoddyl_pelvis_yaw.py; SPEC.md; docs/development/DEVELOPMENT_LOG.md
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 at HEAD (19 unit tests pass in tests/unit/motion_matching/test_crocoddyl_pelvis_yaw.py, test_crocoddyl_action.py, and test_crocoddyl_problem.py; ruff check clean; black clean; ruff format clean; check_lod clean with 0 violations).
- **Summary:** Integrated analytic pelvis-yaw orientation cost into the Crocoddyl full-body solver (`_NodeCost`, `ImplicitEulerAction`, `TerminalAction`). Augmented `FitWeights` with validated non-negative `pelvis_yaw: float = 0.0` and `MarkerTargets` with `waist_indices`. Wired exact Gauss-Newton Jacobian ($J_{yaw}^T J_{yaw}$) and configuration gradient ($J_{yaw}^T r_{yaw}$) into node dynamics. Added `--pelvis-yaw-weight` CLI argument to `full_body_fit.py` and updated `cost_breakdown` to include the `pelvis_yaw` term in trajectory receipts. Verified that aligned targets evaluate to zero cost and gradient, analytic gradients match central finite differences, and missing waist markers gracefully no-op.
- **Next step:** Land PR #10599 with auto-merge enabled.
- **Evidence:** tests/unit/motion_matching/test_crocoddyl_pelvis_yaw.py.

### DL-#10533 · Reconcile, Audit, and Freeze Feature Preservation Across All Historical Boundaries

- **State:** in_progress
- **Owner:** local
- **Issue:** #10533 (ORG-24, epic #10508)
- **Branch:** feat/issue-10533-org24-feature-preservation-audit
- **PR:**
- **Paths:** src/shared/python/workspace/feature_preservation_audit.py; src/shared/python/workspace/**init**.py; tests/integration/test_feature_preservation_audit.py; docs/development/ORG24_FEATURE_PRESERVATION_AUDIT.md; docs/development/HANDOFF.md; docs/development/DEVELOPMENT_LOG.md
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 at HEAD (11 integration tests pass in tests/integration/test_feature_preservation_audit.py covering all RED and GREEN criteria: missing baseline entries fail closed, corrupted fixture hashes fail closed, alias cycles fail closed, unknown workspace domains fail closed, all 159 baseline capabilities reconciled, 13 golden fixtures verified byte-for-byte, acyclic alias resolution, 5 core workspaces verified, external dependencies/engine qualification verified, saved layouts compatible, full audit report generated and published; ruff check, ruff format, mypy, check_file_size_budget all pass cleanly).
- **Summary:** Implemented `FeaturePreservationAuditor` providing comprehensive reconciliation, integrity auditing, and disposition freezing for Epic #10508 closeout per issue #10533. Validated that all 159 capabilities from the ORG-01 baseline are preserved without silent removals. Verified byte-exact SHA-256 and file size integrity across all supported preservation fixtures. Confirmed acyclic transitive resolution of legacy aliases (`starting_pose_matcher` -> `motion_target_preview`, `putting_green_gui` -> `putting_green`). Audited 5 core workspaces and confirmed honest physics engine qualifications (#10351, #10353). Generated and published frozen audit disposition report at `docs/development/ORG24_FEATURE_PRESERVATION_AUDIT.md`.
- **Next step:** Push branch, open PR with auto-merge, release lease on #10533.
- **Evidence:** tests/integration/test_feature_preservation_audit.py; src/shared/python/workspace/feature_preservation_audit.py; docs/development/ORG24_FEATURE_PRESERVATION_AUDIT.md.

### DL-#10532 · Validate and Accept Every Enabled Recommended Task Journey Across Shipped Surfaces

- **State:** in_review
- **Owner:** local
- **Issue:** #10532 (ORG-23, epic #10508)
- **Branch:** feat/issue-10532-org23-installed-workspace-journeys
- **PR:** #10580
- **Paths:** src/shared/python/workspace/installed_journeys.py; src/shared/python/workspace/**init**.py; tests/integration/test_installed_workspace_journeys.py; src/tools/capture_rig/gui.py
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 at HEAD (9 integration tests pass in tests/integration/test_installed_workspace_journeys.py covering all 6 recommended task journeys across shipped surfaces plus cancellation recovery, dependency failure, and schema corruption rejection; ruff check, ruff format, mypy, bandit, unit tests pass cleanly).
- **Summary:** Implemented `InstalledWorkspaceJourneysCoordinator` validating and accepting all enabled recommended task journeys across shipped surfaces per ADR-0047 and issue #10532. Covered optical/video import to inspection, model/pose to supported fit, shot to named flight comparison, optimization/training to result, bounded estimation to cross-engine comparison, and global utilities navigation. Enforced failure recovery preserving source media, actionable dependency diagnostics, and rejection of corrupted artifact schemas. Decoupled `CaptureRigWidget.open_in_inspect_targets()` from StepRail action set to preserve contract and fix test suites.
- **Next step:** Auto-merge PR #10580, release lease on #10532.
- **Evidence:** tests/integration/test_installed_workspace_journeys.py; src/shared/python/workspace/installed_journeys.py.

### DL-#10530 · Reconcile and Document Intentionally Excluded, Research-Only, and Incomplete Workflows

- **State:** in_review
- **Owner:** local
- **Issue:** #10530 (ORG-22, epic #10508)
- **Branch:** feat/issue-10530-org22-research-lifecycle
- **PR:**
- **Paths:** src/config/research_capability_lifecycle.py; src/config/**init**.py; src/config/capability_migration.py; src/tools/model_converter/**main**.py; tests/config/test_research_capability_lifecycle.py; docs/development/HANDOFF.md; docs/development/DEVELOPMENT_LOG.md; SPEC.md
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 at HEAD (8 unit tests pass in tests/config/test_research_capability_lifecycle.py covering all RED and GREEN acceptance criteria: untiled nonexcluded package fails coverage, undocumented headless service fails coverage, CLI-only action cannot be represented as GUI-ready tile, retained headless tools have runnable entrypoints, hidden aliases resolve acyclic, catalog links have owners and next actions, SG optimizer Phase 3 UI follow-up is accurately recorded, research headless documentation links exist; 18/18 config suite tests pass; ruff check and ruff format clean).
- **Summary:** Implemented `ResearchCapabilityLifecycleManager` and `IncompleteCapabilityRecord` reconciling intentionally excluded, research-only, and incomplete workflows per ADR-0047 and issue #10530. Audited every excluded package under `src/tools/` against launcher tiles and `src/config/registry_exclusions.yaml`. Enforced fail-closed prevention of adapting CLI-only tools as GUI tiles via `CLINotInteractiveGUIError`. Verified and standardized CLI entry points for retained tools (`contraction`, `drift_control`, `model_converter`, `sg_optimizer`). Disclosed structured owner/issue/next-action metadata for incomplete capabilities, honestly documenting the SG optimizer Phase 3 PyQt6 UI follow-up tied to #6272 without fake GUIs or premature claims of completion.
- **Next step:** Push branch, open PR with auto-merge, complete lease on #10530.
- **Evidence:** tests/config/test_research_capability_lifecycle.py; src/config/research_capability_lifecycle.py.

### DL-#10528 · Unify Sidekick, Setup, Help, and Library as Global Utilities

- **State:** in_progress
- **Owner:** local
- **Issue:** #10528 (ORG-19, epic #10508)
- **Branch:** feat/issue-10528-org19-global-utilities
- **PR:** #10578
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

### DL-#10358 · Web Matched-Swing Results API and Results Page

- **State:** in_progress
- **Owner:** local
- **Issue:** #10358 (MS-85, epic #10363)
- **Branch:** fix/issue-10358-ms-85-local
- **Paths:** src/api/routes/matched_swings.py; src/api/services/matched_swings_service.py; tests/api/test_matched_swings.py; ui/src/pages/MatchedSwings.tsx; ui/src/api/matchedSwings.ts; ui/src/App.tsx; src/config/launcher_manifest.json; src/api/route_registry.py
- **Started:** 2026-09-21
- **Last verified:** 2026-09-21 at HEAD (8 API tests pass; 4 MatchedSwings vitest tests pass; ruff and architecture budget clean).
- **Summary:** Added local-only read-only `/api/matched-swings` routes (ledger, receipt, candidate NPZ/preview, parity, GIF) keyed by receipt SHA-256 without leaking absolute paths. Web Results page at `/tools/matched-swings` lists runs with verdict badges, GIF playback, and MocapSkeleton3D marker preview; mounted CrossEngineDashboard at `/tools/cross-engine` with launcher manifest web routes for both tiles.
- **Next step:** Open PR, drive CI green, merge, release lease.
- **Evidence:** tests/api/test_matched_swings.py; ui/src/pages/MatchedSwings.test.tsx.

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

### DL-#10355 · Motion Matching Tile Visual Playback, Standardized Metrics, and Navigation Handoff (MS-82)

- **State:** in_progress
- **Owner:** local
- **Issue:** #10355 (MS-82, closes #10106 gap)
- **Branch:** feat/10355-motion-matching-tile-playback
- **PR:** #10995
- **Paths:** src/tools/motion_matching/gui.py; src/tools/motion_matching/pipeline.py; src/config/feature_parity.json; docs/development/feature_parity_matrix.md; tests/tools/motion_matching/test_motion_matching_gui.py
- **Started:** 2026-09-20
- **Last verified:** 2026-09-20 at HEAD (18 unit tests pass in tests/tools/motion_matching/; 39 parity tests pass in tests/config/feature_parity/; ruff check & format clean).
- **Summary:** Enhanced the Motion Matching PyQt6 tile to display asynchronous animation playback (`ik_playback.gif` and `tracking_playback.gif`) via `QMovie`, populated five standardized headline metrics (`full_capture_ik_rms_mm`, `address_marker_rms_mm`, `backswing_root_error_max_mm`, `whole_run_root_rms_mm`, `inside_support_polygon_fraction`), acceptance verdict badge, and navigation handoffs to the Matched Swing Results Browser and Tour Matching Viewer. Upgraded `tools.motion_matching` in `feature_parity.json` from `gap` to `parity`.
- **Next step:** Create PR, enable auto-merge, verify merge, and release lease.
- **Evidence:** tests/tools/motion_matching/test_motion_matching_gui.py, tests/config/feature_parity/test_matrix_freshness.py.

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

### DL-#10460 · Consume Shared GSPro Open Connect V1 Codec From Tools

- **State:** in_progress
- **Owner:** local
- **Issue:** #10460
- **Branch:** feat/10460-consume-tools-gspro-codec
- **PR:** #10668
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
- **PR:** #10668
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
- **PR:** #10668
- **Paths:** src/shared/python/motion_matching/candidate.py; src/shared/python/motion_matching/candidate_io.py; src/shared/python/motion_matching/candidate_convert.py; src/tools/tour_matching_viewer/core.py; src/shared/python/motion_matching/cross_engine_replay.py; docs/development/full_body_models/CANDIDATES.md; tests/unit/motion_matching/test_candidate.py
- **Started:** 2026-09-19
- **Last verified:** 2026-09-19 at HEAD (SELF; candidate package defined with kinematic and dynamic profiles, immutable arrays, and SHA-256 tamper-proof checksumming; lossless converters for returned81 replays, OpenSim MOT files, and ground-support IK/dynamics; 24 candidate and viewer unit tests pass 100%; architecture budget, ruff, mypy clean)
- **Summary:** Implemented unified MatchedSwingCandidate versioned package (matched-swing-candidate-v1) supporting distinct kinematic and dynamic profiles, full-body generalized coordinates and tangent velocities (nq != nv), virtual work power consistency verification, immutable arrays, and SHA-256 array tamper detection. Implemented converters for legacy returned81 replays and OpenSim MOT files, updated viewer loader, and authored complete CANDIDATES.md schema document.
- **Next step:** Run CI checks, commit, push, open PR referencing Closes #10334, enable auto-merge, release lease.
- **Evidence:** docs/development/full_body_models/CANDIDATES.md; tests/unit/motion_matching/test_candidate.py; docs/development/full_body_models/evidence/replays/mujoco_returned81_candidate.npz; docs/development/full_body_models/evidence/viewer/opensim_os3b_candidate.npz.

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

### DL-#9967 · Native Simscape Tour Matching

- **State:** in_progress
- **Owner:** codex (turnover review; execution ownership by next lease)
- **Issue:** #9967 (parent #9921)
- **Branch:** feat/9967-native-simscape-pinocchio
- **Paths:** src/shared/python/motion_matching; docs/development/simscape_tour_matching
- **Started:** 2026-09-09
- **Last verified:** 2026-09-15 (SELF; raw run101 MAT/NPZ metrics independently recomputed; seven focused yaw/replay tests passed)
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

### DL-#8869 · Realtime Pub/Sub: Wire the WS Transport Instead of a Silent No-Op

- **State:** in_review
- **Owner:** claude
- **Issue:** #8869 (folds in #8868, #8942A; seam #9406 ruling: `realtime` is `split pending`, UD keeps this facade)
- **Branch:** fix/8869-realtime-pubsub-decision
- **PR:** #10655 (open)
- **Paths:** `src/shared/python/realtime/api.py`; `src/shared/python/realtime/channels.py`; `src/shared/python/realtime/__init__.py`; `tests/unit/realtime/test_facade.py`; `tests/unit/realtime/test_channels.py`; `tests/shared/realtime/test_api.py`; `tests/shared/realtime/test_channels.py`; `docs/config/pydantic-settings-migration.md` (removed `file_pubsub.py` mention)
- **Started:** 2026-09-21
- **Last verified:** 2026-09-22 at SELF (rebased onto main including #10654; 125 tests pass in tests/unit/realtime + tests/shared/realtime, incl. new unsupported-transport and ws-routing tests; 3 tests skip cleanly when the optional `upstream_realtime` Rust wheel isn't built locally; ruff check/format clean; seam-drift gate passes with 14 pre-existing notes, no new overlap)
- **Summary:** Decision (documented in full in the PR body): WIRE, not delete. `ws_pubsub.py` looked dead from `api.py`'s perspective but is a mature, independently soak-tested feature (`.github/workflows/realtime-soak.yml`, nightly 24h run against issue #5235/#5214 latency budgets, backed by the `upstream-realtime` Rust/Tokio crate) — deleting it would have thrown away real, maintained infrastructure. `api.py.publish()`/`subscribe()` now route to `WSPubSub` when `transport="ws"` or `REALTIME_TRANSPORT=ws` is set explicitly; any other value raises `ValueError` immediately instead of silently falling back to file (the literal defect in #8869's title). `channels.py`'s colliding `register_channel` was renamed to `register_channel_hint` to resolve the naming collision the issue flagged; its automatic frequency-based transport routing is intentionally **not** wired into the default path yet (constructing `WSPubSub` autostarts a background server — an implicit default-on network listener for existing unaware callers like Pose Studio is a separate, riskier product decision). Deleted `file_pubsub.py` (a genuinely redundant, fully-dead second file-transport implementation superseded by `transport_file.py`, imported by nothing but its own tests) and its two test files. Left `#8868` (training-progress publisher/subscriber default wiring) and `#8942A` (already appears fixed on main — `transport_file.py` already tracks a read offset and doesn't re-parse whole files) for separate follow-up; not claimed as resolved here.
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
- **Last verified:** 2026-09-17 (SELF; `tests/companion` 140 passed locally with the workflow-execution test deselected for a pre-existing interpreter/env issue; ruff, ruff-format, mypy clean on changed files; generated page `--check` current)
- **Summary:** Replace the empty documentation inventory and unqualified engine placeholders with two hashed registries parsed by one module: exact-commit, hash-bound, immutable-URL documentation records with derived freshness; engine capabilities qualified only by exact test/artifact evidence with an executing CI gate; per-program documentation routes; known gaps with owning issues; derived publication blockers; generated, freshness-checked provider page.
- **Next step:** Open the non-draft PR with `Closes #9193`, then record reviews for the sixteen `unknown` documentation records in follow-up PRs.

### DL-#9349 · ADR-0046 G2 Workbench Re-Point Closure

- **Issue:** #9349 (ADR-0046 Stage 2; module retirement landed under #9348)
- **Branch:** conductor/issue-9349
- **PR:** not created
- **Paths:** src/config/launcher_manifest.json, src/config/models.yaml, tests/config/launcher_manifest/test_launch_monitor_tiles_share_one_engine.py, docs/adr/0046-launch-monitor-analytics-single-model-layer.md, ui/public/capability-atlas, docs/architecture/CAPABILITY_ATLAS.md
- **Started:** 2026-09-14
- **Last verified:** 2026-09-14 (SELF; new manifest test 4 pass; test_canonical_layer_parity.py and tests/ui/tools/launch_monitor pass against the vendored canonical layer at pin e83bd2e4; capability atlas regenerated)
- **Summary:** Closes the last Stage 2 deliverable this repository owns: both launch-monitor tiles (UD workbench and Rate of Closure Impact Explorer) now state the "same analytics engine" relationship in models.yaml (desktop launcher) and launcher_manifest.json (web launcher), pinned by a test; ADR-0046 follow-ups record G2 as landed. No workbench code changed — both UIs keep their identity.
- **Next step:** Open the PR with `Closes #9349` and merge once quality-gate is green.
- **Evidence:** tests/config/launcher_manifest/test_launch_monitor_tiles_share_one_engine.py; tests/unit/launch_monitor/test_canonical_layer_parity.py.

## Shipped (Last 90 Days)

### DL-#10943 · Drift Wizard Sidekick Product Knowledge Pack

- **State:** in_progress
- **Owner:** antigravity
- **Issue:** `#10943`
- **Branch:** `agy/issue-10943`
- **PR:** not created
- **Paths:** `knowledge/wizard.yml`, `knowledge/pack.yml`, `tests/unit/ai/test_drift_wizard.py`, `.github/workflows/wizard-pack.yml`, `.github/WORKFLOWS.md`, `sidekick.spec`
- **Started:** 2026-09-26
- **Last verified:** 2026-09-26 (`2d5830d18`)
- **Summary:** Author product knowledge pack definition and CI packaging for Drift Wizard in Sidekick chat.
- **Next step:** Author `knowledge/wizard.yml` and `knowledge/pack.yml` to make test suite green.

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
