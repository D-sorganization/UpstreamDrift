# Development Log Archive — 2026

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
- **Next step:** DeskComputer second-person replay under 30 minutes when MS-61 Fit is scheduled.
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
