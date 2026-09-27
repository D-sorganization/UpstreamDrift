# Current Handoff — MJX Tracking Plant and Differentiable Rollout (#11039)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11039-plant`
- Branch: `claude/ud-11039-mjx-plant` (baseline `origin/main`)
- Commit: `SELF`
- Pull request: see DL-#11039. Package 2b of epic #11006. Executed by agy (Gemini 3.8 Flash), reviewed line by line.
- Built: `src/shared/python/motion_matching/mjx_tracking_plant.py` (imports JAX and MJX; no `__init__` imports it): `TrackingPlantSpec` (explicit `root_vertical_index`, `validate_against_model` refuses equality constraints and out-of-range ids), computed-torque control over the reference, sphere-ground and grip-weld wrenches from `jax_contact`, substep/frame/rollout, and `initial_state` that seats the model at static penetration under `model.opt.gravity`.
- Orchestrator changes on review: `from_package` now requires `substeps` and `root_vertical_index` (the heuristics were removed) and raises unconditionally when the package declares a grip closure without `weld_gains`; `build_tracking_plant` works on a copy, so the caller's `model.opt.timestep` is untouched; `rollout` reuses `rollout_diagnostic`.
- Measured in `~/.venv-mjx` on the toy model: computed-torque residual 7.1e-15, marker RMS 0.28 mm over 20 frames without contact, rollout gradient against central differences 2.2e-9 relative.
- The evidence prototype is still untouched; package 3 benchmarks the plant on an exported `mjx_package.npz` and decides promotion.
- Next step: CI green, mark ready, arm.

---

# Past Handoff — Differentiable JAX Contact Law and Grip Weld (#11037)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11037-contact`
- Branch: `claude/ud-11037-jax-contact` (baseline `origin/main`)
- Commit: merged to `main` as PR #11038
- Pull request: #11038; entry DL-#11037. Package 2a of epic #11006. Executed by agy (Gemini 3.8 Flash), reviewed line by line.
- Built: `src/shared/python/motion_matching/jax_contact.py` (imports JAX; no `__init__` imports it): `sphere_ground_contact_jax` over the shared `ContactParameters`/`GroundPlane` with an explicit tangential-speed floor, and `weld_wrench_jax` with caller-supplied `WeldGains`.
- Measured parity against `contact_law.sphere_ground_contact` on 2000 seeded states: normal 0.0 N, friction 1.0e-11 N maximum difference (float64).
- The tests run only where JAX is installed: `~/.venv-mjx` from `scripts/setup_mjx_env.ps1` now exists on DeskComputer (jax 0.11.1, mujoco 3.13.0); the default env and CI skip them.
- Next step: none; merged as PR #11038.

---

# Past Handoff — JAX-Free Knot Basis and Adam Driver for the MJX Solver (#11032)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11032-knot`
- Branch: `claude/ud-11032-knot-adam` (baseline `origin/main`)
- Commit: merged to `main` as PR #11035
- Pull request: #11035; entry DL-#11032. Package 1 of epic #11006. Executed by agy (Gemini 3.8 Flash), reviewed line by line.
- Built: `src/shared/python/motion_matching/knot_gradient_optimiser.py` (numpy only): `knot_grid` (integer knot count, so no untouched trailing knot as the prototype's float `arange` could give), `knot_basis` (vectorised hat functions, refuses untouched knots), `horizon_knot_mask`, keyword-only `AdamSettings`, and `adam_minimise` over an array namespace `xp` with best-by-objective tracking and non-finite stops.
- Orchestrator rewrite on review: `AdamSettings` was a hand-parsed `*args` initialiser and is now a keyword-only frozen dataclass; `adam_minimise` copied nothing, so freezing `best_x` could freeze the caller's `x0` (now copied, with a test).
- The evidence prototype is untouched; package 2 (MJX plant adapter, needs `scripts/setup_mjx_env`) will reuse this driver.
- Next step: none; merged as PR #11035.

---

# Past Handoff — White-Jerk RTS Kinematic Smoother With Uncertainty (#11029)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11029-rts`
- Branch: `claude/ud-11029-rts-smoother` (baseline `origin/main`)
- Commit: merged to `main` as PR #11031
- Pull request: #11031; entry DL-#11029. Package 5 of epic #11007. Executed by agy (Gemini 3.8 Flash), reviewed line by line and replayed into this worktree.
- Built: `src/shared/python/estimation/kinematic_smoother.py` (exact white-jerk discretisation, vectorised Kalman filter plus RTS pass per coordinate, NaN = missing frame, log marginal likelihood, per-coordinate ML noise fit seeded from finite differences) and `smooth_reference_bayesian` in `motion_matching/pipeline/reference.py`. `smooth_reference` is unchanged and pinned bit-for-bit against `butter(4)` + `filtfilt`.
- Orchestrator hardening on review: the initial mean and covariance must be given together and in shape, infinities are refused (only NaN means missing), and result and noise arrays are read-only.
- Validation: see DL-#11029. The 18 local failures in `tests/unit/motion_matching` (event alignment, Rust parity/bench, surrogate training and others) fail identically on clean `origin/main`.
- Next step: none; merged as PR #11031.

---

# Past Handoff — Fitted Sparse Residual for the Physics-Structured Surrogate (#11024)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11024-residual`
- Branch: `claude/ud-11024-sparse-residual` (baseline `origin/main`)
- Commit: merged to `main` as PR #11025
- Pull request: #11025; entry DL-#11024. Package 2 of epic #11007. Executed by agy (Gemini 3.8 Flash), reviewed line by line and replayed into this worktree.
- Built: `src/shared/python/neural_motion/surrogates/sparse_residual.py` (numpy-only STLSQ, `CandidateLibrary`, `SparseResidualFit`); `PhysicsStructuredSurrogate` takes an optional fitted residual and refuses without one. The hand-chosen `0.02*tanh(0.5*prior)` residual is deleted, not kept as a fallback.
- Changed existing test: only `test_forward_surrogates_nm07.py::test_physics_structured_surrogate_prior_and_residual`, which asserted the invented constant; it now asserts the refusal and still checks the prior's shape.
- The STLSQ loop is its own helper `_stlsq` so `fit_sparse_residual` stays inside the 100-line architecture budget.
- CI gates fixed on replay: `setflags(write=False)` instead of the three-level `.flags.writeable` chain (lod-quality-gate), and the fail-closed refusal in `residual_correction` names its tracking issue on the same line (stub-introduction guard).
- Validation: see DL-#11024. Known local failures that also fail on clean `origin/main`: `test_artifact_identity_dbc`, `test_nm01_dbc_optimize` (1 each) and `test_swing_surrogate_training` (4).
- Next step: none; merged as PR #11025.

---

# Past Handoff — Parameter Covariance on the Shared Least-Squares Fits (#11021)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11021-covariance`
- Branch: `claude/ud-11021-fit-covariance` (baseline `origin/main`)
- Commit: merged to `main` as PR #11023
- Pull request: #11023; entry DL-#11021. Child of MS-101 #10375 (Board RM#1793). Executed by agy (Gemini 3.8 Flash), reviewed line by line; the orchestrator replaced the duplicated call-site guard with one tested helper (`fitted_uncertainty_or_none`).
- Built: `src/shared/python/estimation/fit_uncertainty.py`; prefix and multiple-shooting fits report parameter uncertainty.
- Validation: see DL-#11021. `residual_regularization.Array` is now a declared `TypeAlias`: the pre-push mypy env has no numpy, and the bare alias failed as `Array?` in any change to `multi_shooting_fit.py`.
- CI `repo-structure-gates` flagged `least_squares_parameter_uncertainty` at 182 lines (budget 100). It is now split into single-purpose helpers (`_validated_jacobian`, `_requested_indices`, `_free_mask`, `_rank_and_condition`, `_free_covariance`, `_marginal_statistics`) with unchanged behaviour: the 81 estimation tests and the wiring tests pass unmodified, and `check_architecture_budget.py` is OK.
- Next step: none; merged as PR #11023.

---

# Past Handoff — Canonical Swing Event Detector (#11014)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11014-events`
- Branch: `claude/ud-11014-swing-events` (baseline `origin/main`)
- Commit: merged to `main` as PR #11020
- Pull request: #11020; entry DL-#11014. Package 1 of epic #11007. Executed by agy (Gemini 3.8 Flash), reviewed line by line and replayed into this worktree.
- Built: `src/shared/python/analysis/swing_events.py` (one detector plus `peak_speed_index`); four call sites delegate to it; parity and contract tests added.
- Validation: see DL-#11014 (374 passed; plus 188 passed across reconstruct, statistical-analysis, advanced-analysis and swing-comparison consumers).
- Next step: none; merged as PR #11020. Package 2 is #11024.

---

# Past Handoff — Retract the NM-09/NM-12 DIAGNOSTIC Receipt Claims (#10960)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-10960-receipts`
- Branch: `claude/ud-10960-diagnostic-receipts` (baseline `origin/main`)
- Commit: `SELF`
- Pull request: #11017 (merged); entry DL-#10960. Executed by agy (Gemini 3.8 Flash), reviewed line by line.
- Built: the committed NM-09 checkpoint-matrix receipt lists no `qualified_native` models (the three move to `unqualified`) and its limitation says native ODE replay is not verified; the NM-12 turnover receipt promotes no models (the three move to `unmeasured_models`), `end_to_end_verification.status` is `not_verified` and `all_issues_completed` is false. New `tests/unit/neural_motion/test_diagnostic_receipts_10960.py` fails on the old receipts: a DIAGNOSTIC receipt may not qualify or promote anything, and no nested status except `validation.outcome` may read `passed`.
- Validation: `pytest tests/unit/neural_motion/ --deselect test_artifact_audit.py` -> 155 passed (the artifact audit times out locally on a 9.1 GB parquet on disk; CI runs it).
- Next step: CI green, mark ready, arm; then close #10960 with evidence (all code slices already on main).

---

# Past Handoff — Rust Kernel in the Linux Unit-Test Gate (#9411)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-gate-rust`
- Branch: `claude/ud-unit-gate-rust-kernel` (baseline `origin/main`)
- Commit: `SELF`
- Pull request: #10997 (draft), workflow change shipping alone; entry DL-#9411.
- Built: `ci-standard.yml` `unit-test-gate` installs the pinned Rust toolchain, caches the Cargo registry, maturin-builds only `rust_core/upstream-physics` into the unit-gate venv and runs the fail-closed `import_built_rust_wheels.py upstream_physics` probe before pytest; timeout 25 -> 35 min. The 12 BunkerShot workbench/GUI IDs leave `unit_gate_quarantine.json` (67 -> 55 on main after #11010) because `ball_simulator` raises without the kernel.
- Validation: `pytest tests/ci/test_unit_gate_rust_kernel.py tests/ci/test_ci_infrastructure.py tests/unit/repo_hygiene/test_hygiene_guards_run_in_ci.py` -> 111 passed, 1 skipped (new file fails 3/3 against the old workflow); `tests/ci/test_unit_gate_quarantine_contract.py` -> 11 passed; the 12 retired tests pass locally with the kernel installed. Linux proof is this PR's unit-test-gate run.
- First Linux run (0f01454cf): the kernel built and imported, and none of the 12 retired IDs failed. Six tests that the kernel un-skipped failed. Root `Cargo.lock` clutter came from this workflow and is fixed (the untracked lock is removed after the build). Rust vs enhanced carry and TrackMan windows = #11000, fixed in #11004. The degrees-regression test is updated in #11004. The two #9243 uncertainty claims = #11003 (tier:strong).
- Unblocked: #11004 and #11010 (#11003) merged; replayed onto main d716c698f.
- Next step: unit-test-gate green on the replay, then mark ready and arm via `automerge_guard.py`.

---

# Past Handoff — Restate the #9243 BunkerShot Uncertainty Claims (#11003)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11003`
- Branch: `claude/ud-9243-claims-restated` (baseline `origin/main`)
- Commit: `SELF`
- Pull request: #11010 (draft); entry DL-#11003. Unblocks UD#10997 (Rust kernel in the unit gate).
- Decision (tier:strong, owner delegated the judgement 2026-09-26): both failures are stale claims, not model defects. (1) The window-band fixture aimed at a fixed 2 m target; the corrected F0 model's grid carries are 0.05-0.66 m (nominal 0.690 m), so the window was empty on all three grids. The fixture now targets the nominal shot's own carry and asserts a non-empty window; the band measures (0.0, 1.141, 2.282) and the not-decorative claim holds. (2) The accelerated-mass share is 0.676 < `DOMINANCE_SHARE` 0.75: it dominates but does not swamp. The test now states that; the threshold is unchanged.
- Validation: `tests/unit/tools/bunker_shot_gui/test_uncertainty_propagation_9243.py` 30 passed with upstream-physics 2.1.3 built from source, and 30 passed with the user-site 2.1.0 wheel.
- Known: an absolute nominal carry of 0.69 m is short for a greenside splash shot; it sits inside the named uncalibrated transfer-efficiency gap (TestHonestyBoundary), not this PR.
- Next step: CI green, mark ready, arm, then replay #10997.

# Past Handoff — Rust Ball-Flight Kernel Base Cd (#11000)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-rust-drag`
- Branch: `claude/ud-rust-drag-base-cd` (baseline `origin/main`)
- Commit: `SELF`
- Pull request: #11004 (draft); entry DL-#11000. Found by UD#10997 (Rust kernel in the unit gate), which is blocked on this PR and on #11003.
- Built: `ball_simulator.py` passes `GOLF_BALL_DRAG_COEFFICIENT` (0.25) instead of `ball.cd0` (0.21) as the kernel's Reynolds-curve base Cd. New kernel-independent boundary test with a fake `upstream_physics`. `test_raw_degrees_would_collapse_carry` became `test_raw_degrees_are_refused_before_they_collapse_carry` (the launch contract now raises). The exported parity fixture `default_trajectory.json` is left untouched: `src/config/capability_migration.json` pins its byte hash (`test_green_fixture_byte_hashes_unchanged`).
- Validation: see DL-#11000. Local kernel = upstream-physics 2.1.3 built with maturin into a scratch venv; the user-site 2.1.0 wheel on this box is stale and gives different numbers.
- Known: `tests/parity/test_ball_flight_parity.py::test_export_reference_vectors` rewrites the committed, hash-pinned fixture on every run where the kernel is present (pre-existing; follow-up task filed). Restore it with `git checkout -- tests/parity_fixtures` after local runs.
- Next step: CI green, mark ready, arm, then rebase #10997 and fix its untracked root `Cargo.lock` (root-clutter gate).

# Past Handoff — #9411 Unit-Gate Quarantine Burn-Down (Slice 2) 2026-09-26

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/claude-ud-9411-probe`
- Branch: `claude/ud-9411-unquarantine-passing`
- Baseline commit: `b8c27a7d2` (origin/main, after #10988)
- Implementation commit: `SELF`
- Pull request: #10990 (draft)
- Governing issue: #9411 (DL-#9411)
- Lease session: `claude-deskcomputer-20260925-ud`

## Objective and Status

- Objective: retire unit-gate quarantine IDs whose tests pass for the right reason.
- Status: 54 IDs retired (155 → 101). The second Linux run re-quarantined the order-dependent
  `test_install_prompt` worker test and replaced a module `reload` in `test_synthesize_target` (it
  broke the pinocchio_golf facade identity test) with an AST check. The first push retired 90, but Linux CI failed 24 of
  them (GL-less mujoco, missing trimesh, bunker GUI carry, force-plate ownership guard) and the
  child-copy contract flagged the `ai/adapters` edits. `ai/adapters` and the gear-effect sign flip
  were reverted to main (the flip contradicted `test_impact_physics_value_assertions`: toe impact
  gives draw spin, so the two `test_impact_model` gear tests need a convention decision, not a
  sign change); their 11 dependent IDs and the 24 Linux failures are re-quarantined.
- Slices (agy on DeskComputer and OG Laptop, reviewed and trimmed by Claude):
  - validation-contracts: shoulder FK origin (the gear-effect sign flip was reverted).
  - signal-hygiene: launcher test drift.
  - ai-gui-adapters: lazy API routers; theme router prefix (the `ai` adapter shadow edits were
    reverted: Tools child copies may only be deleted, never edited).
  - data-fitting split: `data_fitting.py` 1081 → 479 LOC by removing classes duplicated in
    `_data_fitting_models.py` / `_data_fitting_solvers.py`. The agy `theta_optimal` alias was
    dropped; `test_init_dempster` asserts `coefficients` again (a #4273 blind rename).
  - mujoco-viewer split: coordinator duplicates removed. The agy `sys.modules` mock lookup was
    replaced by patching `_mujoco_viewer_backend` in `test_model_explorer_temp_file.py`.
  - sidekick-drift subset: Simscape C3D embed adapter clears its widget on cleanup; test-only
    retargets. Shadow deletions, the `ai` panel growth and the bootstrap path flip were held
    back for a dedicated #9406 slice.
- Reverted: edits to Tools-owned shadow files (`signal_toolkit` guards, `model_generation`
  gravity and positive-mass checks, `sidekick/data_processing`, `ai/gui/session_manager` and
  `assistant_panel`, `humanoid_character_builder`). Each grew the file's drift from vendor; the
  fixes belong in Tools. 11 IDs that only passed because of them are quarantined again.
- Rejected: deleting `tests/unit/test_safe_eval.py`. Its five failures are real (Tools#5360,
  fixed in Tools#5361); the IDs stay quarantined until the vendor pin moves.

## Validation

- Quarantine rerun after the revert: `pytest -n 6 -o addopts="" --tools-mode vendored <101
candidate IDs>` → 90 passed, 11 failed (re-quarantined).
- `scripts/ci/check_unit_gate_quarantine.py`: contract passed, 65 IDs in 10 clusters;
  `tests/ci/test_unit_gate_quarantine_contract.py`: 11 passed.
- Data-fitting tests (4 files): 72 passed. Viewer, embed-adapter, launcher and sidekick tests:
  234 passed, 4 failed (all 4 still quarantined).
- `ruff check` / `ruff format --check` on changed files: clean. 29 local repo-structure steps pass.

## Blockers and Risks

- Remaining 65 IDs: inertia API (#6995 design, 10), `safe_eval` (Tools#5360, 5), sidekick
  shadow retirement (#9406), signal_toolkit limits/core, `test_level` (`-O` contract level
  design), size budgets for Tools-owned files, Simscape `ezc3d`.
- Order-dependent failures in the remaining set come from `sys.modules` pollution, not the DbC
  level: `tests/unit/sidekick/standalone/test_session_store.py` makes later tests load the Tools
  copies (gravity 9.81, no guards). The #9406 shadow retirement removes that test's target.
- `check_architecture_budget.py` fails on unmodified main (`validate_inertia_tensor`, 101 lines).
- Rebase onto main after #10988: HANDOFF conflicts (take this file), DL/SPEC keep both rows.

## Next Steps

1. After #10988 merges, rebase, open the draft PR, add the SPEC row keyed by its number.
2. Get `quality-gate` green; mark ready; arm through `scripts/automerge_guard.py`.
3. Comment on #9411 with the remaining clusters and owners.

---

# Past Handoff — Bump `vendor/ud-tools` to Tools Main `3678409fc` (#9411)

- **Branch:** `claude/ud-9411-vendor-bump-safe-eval`; PR #10995, pairs with Tools#5364 (`UD-PAIR`).
- **Change:** gitlink 95ed6b478 → 3678409fc; `Cargo.toml`, `requirements-tools.txt`, `src/config/impact_acceptance.json`, `reconciliation.py` and its test aligned; converged child copies in `src/shared/python/` on canonical Tools; retired 10 quarantined tests in `scripts/config/unit_gate_quarantine.json` (ratchet 155 -> 145 node IDs); divergence inventory and agent context regenerated.
- **Validation:** companion (143 passed), reconciliation (3 passed), 10 un-quarantined tests passed, child-copy contract (20 passed), quarantine ratchet (145 IDs in 10 clusters), divergence inventory and `agent_context check` clean.
- **Next:** verify pre-push checks, push, run CI Standard, verify quality gate, and arm auto-merge.

---

# Past Handoff — Consolidated Bolt Norm Micro-Optimizations (#10983, #10984)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-bolt-consol`
- Branch: `claude/ud-bolt-consolidated-20260926` (baseline `origin/main`)
- Commit: `SELF`
- Pull request: #10993 (draft); supersedes #10983 and #10984 (both conflicted on their SPEC rows)
- Issue: none (Bolt performance PRs). No material development-log change — two behaviour-preserving single-expression rewrites.
- Built: `fit_pipeline._quality` computes landmark distances as `sqrt(einsum)` (#10984); `shot._rotation_increment` computes the angular-velocity norm once with `math.sqrt(np.vdot)` and reuses it for the axis (#10983). Inline comments shortened to the 88-column limit; the Bolt journal entry is kept with its date corrected to 2026-09-26.
- Validation: `pytest tests/motion_capture/test_reference_fit_pipeline.py tests/motion_capture/test_reference_fit_preview.py tests/bunkershot3d/solvers/test_shot.py` -> 45 passed; ruff check/format clean.
- Next step: CI green, mark ready, arm via `automerge_guard.py`, then close #10983 and #10984 as superseded.

---

# Past Handoff — Main Red on Bandit B314 in the Coverage Gate Checker (#10989)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-10989`
- Branch: `fix/10989-coverage-gates-defusedxml` (baseline `84ac37579`)
- Commit: `SELF`
- Pull request: #10994 (draft)
- Issue: #10989 (fleet-main-health: CI Standard red on main); development-log entry DL-#10965
- Cause: #10988 landed `scripts/check_coverage_gates.py` parsing Cobertura XML with `xml.etree.ElementTree.parse`; the push-lane full-tree `bandit -ll -ii` flags B314 and fails `security-scans`, which fails `quality-gate`.
- Fix: parse with `defusedxml.ElementTree` (a core dependency, the convention in `scripts/config/coverage_enforcer.py`); `Element` imported under `TYPE_CHECKING` for the annotation. New test `test_xml_entity_expansion_is_rejected` was red on stdlib ET and is green now.
- Validation: `pytest tests/unit/scripts/test_check_coverage_gates.py` -> 13 passed (new test carries `@pytest.mark.unit` for the suite-marker ratchet); `bandit -ll -ii scripts/check_coverage_gates.py` clean; ruff and mypy clean.
- Next step: CI green, mark ready, arm via `automerge_guard.py`; #10989 closes itself when main's next CI Standard run succeeds.

---

# Implementation Handoff — #9406 Sidekick Shadow Retirement 2026-09-26

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/claude-ud-9406-shadow`
- Branch: `claude/ud-9406-sidekick-shadow-retire`
- Baseline commit: `b8c27a7d2` (origin/main, after #10988)
- Implementation commit: `SELF`
- Pull request: #10991 (draft)
- Governing issue: #9406 (DL-#9406)
- Lease session: `claude-deskcomputer-20260925-ud`

## Objective and Status

- Objective: stop UpstreamDrift shipping stale copies of Tools-owned `sidekick` modules, so the
  quarantined tests that demand their absence pass.
- Status: agy (Gemini 3.8 Flash) slice on DeskComputer, reviewed and trimmed by Claude.
  - Deleted, now resolved from `vendor/ud-tools/src/shared/python/sidekick/`: `standalone/`
    (runner, window, session_store, preferences, onboarding), `persistence/`, `__main__.py`,
    `ui/tools_sidebar/default_tabs.py`.
  - `embedded_tool_bootstrap._bootstrap_python_paths`: vendored or explicit Tools paths now
    precede UpstreamDrift's own, as the bootstrap tests require.
  - `sidekick.spec`: Windows icon is the pinned `vendor/ud-tools/assets/tools_icon_hq.ico`.
  - Tests retargeted to the vendor API: pickle is no longer auto-detected (`test_data_io`), C3D
    export needs `C3D_ALLOW_ANY_EXPORT_PATH` (`test_c3d_reader`), and `test_cli` patches the
    launcher factory under every import prefix.
  - 14 quarantine IDs retired (155 → 141).
- Dropped from the agy output: an `architecture_budget.json` exception (the violation predates
  this branch) and an `ai/gui/assistant_panel.py` vendor copy (the backward-compat identity tests
  still fail because vendor imports through the `shared.python` prefix).

## Validation

- `pytest -n 6 -o addopts="" --tools-mode vendored <33 sidekick/packaging/bootstrap IDs>`:
  14 passed (all retired); the rest stay quarantined.
- `UNIT_GATE_QUARANTINE=1 pytest -n 6 --tools-mode vendored tests/unit/sidekick tests/unit/launcher
tests/launchers tests/unit/packaging tests/unit/repo_hygiene tests/integration/sidekick
tests/ui/c3d_viewer`: 67 failed / 2686 passed here vs 65 / 2687 on `b8c27a7d2`; the only new
  ID (`test_embed_adapter::test_create_main_widget_returns_qwidget`) passes alone twice
  (xdist order). The #9406 slice also fixed one main failure.
- `scripts/ci/check_unit_gate_quarantine.py`: contract passed, 141 IDs.

## Blockers and Risks

- The same launcher/launchers failures appear on main in this Windows environment.
- #9411 (quarantine burn-down) edits the same ledger; whichever lands second takes the union
  of removals.

## Next Steps

1. Open the draft PR, add the SPEC row keyed by its number, get `quality-gate` green.
2. Mark ready and arm through `scripts/automerge_guard.py`.
