# Current Handoff — Ball-Flight Parity Fixture Export Is Opt-In (#11008) 2026-09-26

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/claude-ud-parity-fixture-optin`
- Branch: `fix/ball-flight-parity-fixture-opt-in` (baseline `origin/main` 170481160)
- Implementation commit: `SELF`
- Pull request: #11012 (draft)
- Governing issue: #11008 (DL-#11008)

## Objective and Status

- Objective: stop `test_export_reference_vectors` from rewriting the committed golden
  `tests/parity_fixtures/ball_flight/default_trajectory.json` on every Rust-enabled run.
- Done: vector construction is in `build_default_trajectory_vectors()`, the schema is in
  `assert_vector_schema()`, and writing goes through `write_vectors()` (LF, round-trip
  checked). The export test now writes to `tmp_path` and asserts that the committed bytes
  are unchanged. `test_regenerate_committed_fixture` rewrites the committed file only when
  `UPSTREAMDRIFT_REGENERATE_PARITY_FIXTURES=1` (exactly `1`).
- New `TestParityFixtureContract` (not Rust-gated): checks the opt-in switch semantics,
  the committed fixture's schema, the fixture path pinned in
  `src/config/capability_migration.json`, and that malformed vectors are rejected.
  `pytestmark` became a class-level `@requires_rust` so the contract tests run in every lane.
- The fixture bytes and the path are unchanged, so the sha256 pin stays valid.

## Validation

- `python -m pytest tests/parity/test_ball_flight_parity.py -o addopts=""` (Python 3.12, Rust
  kernel available): 18 passed, 1 skipped (opt-in regen), 1 failed. The failure is
  `test_default_trajectory_physics` (max height 60.8 m > 60 m), and it fails the same way on
  unmodified main. `git status` stays clean after the run.
- With `UPSTREAMDRIFT_REGENERATE_PARITY_FIXTURES=1 ... -k regenerate`: the fixture is rewritten (reverted).
- `ruff check` / `ruff format --check` on the changed test: clean.

## Blockers and Risks

- The committed golden is stale against the current model. Regenerating it changes ball
  defaults (mass 0.0459→0.04593, cd1 0.05→0.25) and carry (187→243 m). #11004 owns that
  regeneration and the sha256/size pin update. This PR does not touch the fixture.

## Next Steps

1. Get `quality-gate` green on draft PR #11012.
2. Mark it ready and arm through `scripts/automerge_guard.py`.

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
