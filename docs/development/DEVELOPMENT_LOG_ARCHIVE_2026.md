# Development Log Archive — 2026

Finished entries (`shipped`, plus legacy `completed` entries whose PRs merged),
moved verbatim out of the live [development log](DEVELOPMENT_LOG.md) to keep it
under the validator's size ceiling. Entries are never edited here; the live log
remains the source of truth for work in flight.

### DL-#11058 — MJX Knot Optimiser Head-to-Head Benchmark and Promotion Decision

- **State:** shipped
- **Owner:** claude
- **Issue:** #11058
- **PR:** #11072
- **Branch:** `claude/ud-11058-benchmark`
- **Paths:** `src/shared/python/motion_matching/solver_benchmark.py`, `scripts/benchmark_mjx_knot_solvers.py`, `src/shared/python/motion_matching/knot_gradient_optimiser.py`, `src/engines/physics_engines/mujoco/python/motion_matching/mjx_knot_optimiser.py`, `src/shared/python/motion_matching/pipeline/`, `docs/development/full_body_models/evidence/mjx_benchmark/`
- **Started:** 2026-09-27
- **Last verified:** 2026-09-28 — full benchmark: 8 pipeline runs with rc 0; the report is regenerated from the committed receipts.
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
- **Last verified:** 2026-09-27 — the new test fails on `main` with the `NameError` and passes with the fix; a real `--shooting-fit 8` driver run completes.
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
- **Last verified:** 2026-09-27 — 29 passed in the MJX env; pipeline tests 76 passed, 10 skipped; a 3-iteration pipeline run improved marker RMS 0.0652 → 0.0503 m and the default run is unchanged apart from `elapsed_s`.
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
- **Last verified:** 2026-09-27 — helper tests 12 passed on MuJoCo 3.13; 1967 passed on MuJoCo 3.4 across the 200 test files that reference a touched module, with no failure that is not also on `main` or order-dependent.
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
- **Last verified:** 2026-09-27 — `anthro_driver_seeds` `--iterations 3` receipt history identical to the #11046 CLI (0.0 relative at every iteration); 29 passed in `~/.venv-mjx`; ruff and mypy clean.
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
- **Last verified:** 2026-09-27 — old vs rewired on `anthro_driver_seeds`: port check 65.16457 vs 65.16494 mm, iteration 3 50.249 vs 50.295 mm (9.2e-4 relative; float32 reordering differences grow per Adam step: 5.6e-6, 1.6e-5, 7.2e-5, 9.2e-4); CLI test passes in `~/.venv-mjx`.
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
- **Last verified:** 2026-09-27 — `tests/ci/` 201 passed, 1 skipped; the new exit-5 contract test is red on `origin/main` and green here.
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
- **Last verified:** 2026-09-27 — seven full pipeline regenerations; provenance-chain, handoff-number, ledger and status freshness tests pass.
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
- **Last verified:** 2026-09-27 — full pipeline `--static-seeds --fit-closure` 861.2 → 83.0 mm dynamics RMS (weight fraction max 50.0 → 3.72); `--static-seeds` 83.0 → 84.5 mm; 41 targeted tests pass.
- **Summary:** Tracked replays start from the reference velocity projected onto the weld (mass-weighted, the weld's inelastic impulse), because the acceleration-level KKT conserves any initial weld violation and the fitted closure's 187 mm/s start opened the grip and drove the controller through a truncated-SVD singularity.
- **Next step:** None — merged as PR #11047.

### DL-#11034 · Force-Plate Fixture Reports Module Identity on the Intermittent Failure

- **State:** shipped
- **Owner:** claude
- **Issue:** #11034 (refs; stays open for the root cause)
- **PR:** #11042
- **Branch:** `claude/ud-11034-force-plate-flake`
- **Paths:** `tests/unit/sidekick/lab/bio/test_force_plate_stitching.py`
- **Started:** 2026-09-27
- **Last verified:** 2026-09-27 — 5 passed locally; the report renders on a synthetic module; Linux `unit-test-gate` passed twice on the PR.
- **Summary:** The intermittent `AttributeError` on `shared.python.sidekick.lab.bio.force_plate_stitching` now fails with the resolved file, spec, module names, meta path and related module entries, so the next occurrence names the polluting state. Local reproduction attempts (single file, overlay-first, pairwise with every import-rewiring test) did not fail.
- **Next step:** None — merged as PR #11042.

### DL-#11039 · MJX Tracking Plant and Differentiable Rollout

- **State:** shipped
- **Owner:** claude (agy executor, Gemini 3.8 Flash)
- **Issue:** #11039 (epic #11006 package 2b)
- **PR:** #11040
- **Branch:** `claude/ud-11039-mjx-plant`
- **Paths:** `src/engines/physics_engines/mujoco/python/motion_matching/mjx_tracking_plant.py`, `tests/unit/engines/mujoco/test_mjx_tracking_plant.py`
- **Started:** 2026-09-27
- **Last verified:** 2026-09-27 — 11 passed in `~/.venv-mjx`; skipped cleanly in Python312; ruff, format and mypy clean on the module.
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
- **Last verified:** 2026-09-27 — 8 passed in `~/.venv-mjx` (with and without conftest); skipped cleanly in Python312; ruff, format and mypy clean.
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
- **Last verified:** 2026-09-27 — 16 new tests pass; the module imports with JAX absent; ruff, format and mypy clean.
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
- **Last verified:** 2026-09-27 — the 12 new tests pass; tests/unit/estimation plus tests/unit/motion_matching 2169 passed, and the 18 failures there fail identically on clean origin/main; ruff, format and mypy clean.
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
- **Last verified:** 2026-09-27 — tests/unit/neural_motion (except test_artifact_audit) 161 passed; motion_matching -k surrogate 128 passed; the 6 failures elsewhere in both runs fail identically on clean origin/main; ruff, format and mypy clean.
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
- **Last verified:** 2026-09-27 — tests/unit/estimation plus the wiring tests 81 passed, 1 skipped; motion_matching prefix/shooting/multi tests 73 passed; tests/motion_matching prefix and regularization 37 passed; architecture budget OK.
- **Summary:** `least_squares_parameter_uncertainty` gives the Gauss-Newton (Laplace) covariance of a `least_squares` optimum: σ̂² = 2·cost/(m − n_free); the requested block is the marginal taken from the full free-parameter covariance (Schur complement over nuisance variables such as shooting states); parameters held at an active bound get NaN; a rank-deficient Jacobian or m ≤ n_free returns no covariance at all rather than pinv errors. `PrefixStage.parameter_uncertainty` and `MultipleShootingFit.theta_uncertainty` carry it; the equality-constrained shooting path leaves it None, and a non-finite Jacobian yields None so a diverged fit still reports.
- **Next step:** None — merged as PR #11023.

### DL-#11014 · Consolidate the Swing Phase Detectors Into One Canonical Event Detector

- **State:** shipped
- **Owner:** claude (agy executor, Gemini 3.8 Flash)
- **Issue:** #11014
- **PR:** #11020
- **Branch:** `claude/ud-11014-swing-events`
- **Paths:** `src/shared/python/analysis/swing_events.py`, `src/motion_capture/reconstruct/analytics.py`, `src/shared/python/analysis/phase_detection.py`, `src/shared/python/data_io/swing_capture_import.py`, `src/shared/python/motion_matching/loaders/_align.py`, `tests/unit/analysis/test_swing_events.py`, `tests/unit/analysis/test_swing_event_parity_11014.py`
- **Started:** 2026-09-27
- **Last verified:** 2026-09-27 — analysis, reconstruct analytics, swing-capture import, loaders-align, DbC swing-phase and shared-biomechanics tests: 374 passed, existing tests unedited.
- **Summary:** New `swing_events.py` holds the single frame-rate-aware detector (`detect_swing_events`, quiet-window address, top within `max_downswing_s` before the peak, finish under `finish_fraction` or the quiet threshold) and `peak_speed_index`; `SwingEventFrames` enforces `0 <= address <= top <= peak <= finish` even with contracts OFF. `analytics.detect_events` delegates bit-identically; `PhaseDetectionMixin` derives fps from the median time step (keeps its 30 % finish threshold); `SwingCaptureImporter` uses the trajectory frame rate; `_align` uses `peak_speed_index`. The old `0.7 * impact` and `n_frames // 2` top searches depended on where the recording started and are gone. A parity test pins all four sites to the same peak/top/address on one fixture.
- **Next step:** None — merged as PR #11020; package 2 continues as #11024.

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

### DL-#9411-AI-Adapters · Converge Gemini and BitNet Adapters on Canonical Tools and Retire 5 Tests

- **State:** shipped
- **Owner:** antigravity
- **Issue:** #9411
- **PR:** #11005 (merged)
- **Branch:** `feat/ud-9411-ai-adapters-convergence`
- **Paths:** `src/shared/python/ai/adapters/gemini_adapter.py`, `src/shared/python/ai/adapters/bitnet_adapter.py`, `scripts/config/unit_gate_quarantine.json`, `SPEC.md`
- **Started:** 2026-09-26
- **Last verified:** e9f72ac1f — merged via squash auto-merge with zero administrative bypasses; 5 retired tests green; ratchet at 67.
- **Summary:** Converge Gemini and BitNet AI adapters on canonical Tools implementations; normalize seam imports; retire 5 verified-passing tests in `scripts/config/unit_gate_quarantine.json` (ratchet 72 -> 67).
- **Next step:** Shipped in PR #11005.

### DL-#9411-Burndown-Batch1 · Fix UI Module Monkeypatch Invariants and Retire 6 Quarantined Tests

- **State:** shipped
- **Owner:** antigravity
- **Issue:** #9411
- **PR:** #10999 (merged)
- **Branch:** `feat/ud-9411-quarantine-burndown-batch1`
- **Paths:** `scripts/config/unit_gate_quarantine.json`, `tests/unit/ui/dialogs/test_install_prompt.py`, `tests/unit/ui/test_window_icon.py`
- **Started:** 2026-09-26
- **Last verified:** 170481160 — merged via squash auto-merge with zero administrative bypasses; 6 retired tests green; ratchet at 72.
- **Summary:** Fix module monkeypatching and typing in `test_window_icon.py` and `test_install_prompt.py`; condense `test_install_prompt.py` to <= 500 LOC; retire 6 verified-passing tests in `scripts/config/unit_gate_quarantine.json` (ratchet 78 -> 72).

### DL-#9411-Pin · Bump `vendor/ud-tools` to Tools Main `3678409fc` and Retire 10 Quarantined Tests

- **State:** shipped
- **Owner:** claude
- **Issue:** #9411
- **PR:** #10995
- **Branch:** `claude/ud-9411-vendor-bump-safe-eval`
- **Paths:** `vendor/ud-tools`, `Cargo.toml`, `requirements-tools.txt`, `src/config/impact_acceptance.json`, `src/shared/python/tour_baselines/reconciliation.py`, `tests/unit/tour_baselines/test_reconciliation.py`, `scripts/config/unit_gate_quarantine.json`, `src/shared/python/`, `docs/shared_tools/`, `docs/agent_context/`
- **Started:** 2026-09-26
- **Last verified:** 2026-09-26 — companion (143 passed), reconciliation (3 passed), 10 un-quarantined tests passed; child-copy contract (20 passed), quarantine ratchet (145 IDs in 10 clusters), divergence inventory and agent-context clean.
- **Summary:** Advance the gitlink to Tools main 3678409fc (#5364 input contracts and #5361 safe_eval power bound) and align all pin strings. Converge child copies under `src/shared/python/` on canonical Tools and retire 10 quarantined tests in `scripts/config/unit_gate_quarantine.json` (ratchet 155 -> 145 node IDs).
- **Next step:** Shipped in #10995.

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

### DL-#8932 · Debounce and Memoize Advanced Analysis Tab Refreshes

- **State:** completed
- **Owner:** antigravity
- **Issue:** #8932
- **Branch:** fix/8932-analysis-tab-debounce-idle
- **Paths:** `src/shared/python/dashboard/_analysis_refresh.py`; `src/shared/python/dashboard/advanced_analysis.py`; `tests/unit/shared_python/test_analysis_tab_refresh.py`
- **Started:** 2026-09-22
- **Last verified:** 2026-09-24 — PhasePlaneTab and CoherenceTab spinboxes debounced with `DebouncedRefresh` (150 ms); CoherenceTab memoizes `compute_coherence` with `BoundedResultCache` on `analysis_cache_key`; all tabs in `advanced_analysis.py` (CorrelationTab, PhasePlaneTab, CoherenceTab, SwingPlaneTab, SpectrogramTab, WaveletTab) use `draw_idle()` with 0 blocking `canvas.draw()`; all 25 unit tests pass offscreen; ruff, black, mypy, DRY duplication gate clean.
- **Summary:** SpectrogramTab/WaveletTab/PhasePlaneTab/CoherenceTab spinboxes go through `DebouncedRefresh` (150 ms); CoherenceTab, SpectrogramTab, and WaveletTab memoize expensive transforms in `BoundedResultCache`; SwingPlaneTab builds its axes once and swaps artists; all six analysis tabs use non-blocking `draw_idle()`.

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
