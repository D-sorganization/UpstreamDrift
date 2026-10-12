# 66-entry candidate source snapshot

Source basis: UpstreamDrift 944d798ed3748fc09e115825cf9db30b65d1f232; docs/development/DEVELOPMENT_LOG.md Git blob cfa0c48e1a47a620228bebfa326c4138d7e5912a; SHA-256 d9268879c0527199ae588dcf91a93a50556d1f29dac28924a658a2a43bca4bdb. Candidate and close/merge metadata source: dl-github-closure-audit.json; SHA-256 c0e349d78ce4d26287ef124287fc932337b625a21c3ace7c87deb4cce3fb44eb. Every block below is exact original UTF-8 text captured from that source; no canonical entries are changed by this planning artifact.

## #11589 source occurrence 1

```text
### DL-#11589 · Retire Stale Shared-Code Header Test

- **State:** in_review
- **Owner:** claude
- **Issue:** #11589
- **Branch:** chore/retire-unflagged-header-test-11589
- **PR:** #11590
- **Paths:** see #11590
- **Started:** 2026-10-06
- **Last verified:** 2026-10-06 (`a4031aa0`; collated from changes/11589-chore-tests-retire-stale-test-no-unflagg.md)
- **Summary:** chore(tests): retire stale test_no_unflagged_shared_code header test and its unit-gate quarantine entry
- **Next step:** Merge the PR; child-copy enforcement remains in test_tools_child_copy_contract.py.
```

## #11513 source occurrence 1

```text
### DL-#11513 · Point Contributors at Change Fragments in Post_Spec_Reminder

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #11513
- **Branch:** fix/11513-post-spec-reminder-fragments
- **PR:** #11519
- **Paths:** see #11519
- **Started:** 2026-10-05
- **Last verified:** 2026-10-05 (`739df607`; collated from changes/11513-point-contributors-at-change-fragments-i.md)
- **Summary:** Point contributors at change fragments in post_spec_reminder
- **Next step:** Merge the PR.
```

## #11190 source occurrence 1

```text
### DL-#11190 · Disable FastRestart for Solve_Starting_Pose and Resolve Model Path in Run_Leaderboard

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #11190
- **Branch:** fix/11190-motion-matching-fast-restart
- **PR:** #11518
- **Paths:** see #11518
- **Started:** 2026-10-05
- **Last verified:** 2026-10-05 (`d816c998`; collated from changes/11190-disable-fastrestart-for-solve-starting-p.md)
- **Summary:** Disable FastRestart for solve_starting_pose and resolve model path in run_leaderboard
- **Next step:** Merge the PR.
```

## #11233 source occurrence 1

```text
### DL-#11233 — Necromatcher Persistent Library

- **State:** in_progress
- **Owner:** codex
- **PR:** [#11237](https://github.com/D-sorganization/UpstreamDrift/pull/11237)
- **Issue:** #11233; parent #11232; player epics #11226 and #11229
- **Branch:** `feat/necromatcher-library-11233`
- **Paths:** `src/shared/python/workspace/necromatcher.py`, `src/shared/python/workspace/necromatcher_capture.py`, `src/api/routes/necromatcher.py`
- **Last verified:** 2026-10-01 — 43 workspace/API tests passed; scoped library mypy passed; four real captures imported and recalled. Final rerun passed.
- **Summary:** Reuse project/session spine, artifact hashes and canonical torque evaluation for immutable historical-player library; captures remain image observations and model bytes remain unqualified candidates. [Turnover](necromatcher-turnover.md).
```

## #11184 source occurrence 1

```text
### DL-#11184 — Restore the High-Severity UI Npm Audit Gate

- **State:** in_review
- **Owner:** codex
- **Issue:** #11184
- **Branch:** `fix/main-npm-audit-11184`
- **Paths:** `ui/package-lock.json`, `ui/src/test/dependencySecurityContract.test.ts`, `SPEC.md`
- **Started:** 2026-09-30
- **Last verified:** 2026-09-30 at `d3b3a27bea` — baseline audit reproduced two HIGH and two MODERATE advisories; after the lock-only patch `npm ci` passed, `npm audit --audit-level=high` passed with two MODERATE findings remaining, contract 2 passed, lint/type-check passed, UI tests 936 passed, and production build passed. Normal commit and push hooks passed.
- **Summary:** Raised only the compatible transitive resolutions for `brace-expansion` (5.0.9 → 5.0.12) and `undici` (8.10.0 → 8.11.2); added a lockfile security contract. Manifest dependencies, overrides, audit threshold, and the two moderate findings are unchanged.
- **Next step:** Root reviews draft PR #11187 and decides whether it is ready for merge.
```

## #11124 source occurrence 1

```text
### DL-#11124 — Anti-Phantom-Merge Path Extraction for Scripts, Workflows, and Parentheticals

- **State:** in_review
- **Owner:** local
- **Issue:** #11124
- **Branch:** `fix/phantom-guard-scripts-paths-11124`
- **Paths:** `scripts/ci/check_phantom_guard_paths.py`, `tests/scripts/test_check_phantom_guard_paths.py`
- **Started:** 2026-09-29
- **Last verified:** 2026-09-29 — 27 focused unit tests pass in `test_check_phantom_guard_paths.py`; ruff, black, architecture budget, and LOD clean.
- **Summary:** Fixes false positive in anti-phantom-merge Rule 3 by expanding `ISSUE_PATH_PATTERN` to recognize `scripts/` and `.github/workflows/` (and `.github/`) paths, adding `(?<![\w/.-])` boundary protection to reject non-path prefix substrings (e.g. `engines/api`), stripping trailing punctuation from extracted paths, and dropping parenthetical prose fragments lacking valid file extensions.
- **Next step:** Submit PR for review and release lease.
```

## #11107 source occurrence 1

```text
### DL-#11107 — Implement Simscape Continuous-Replay Qualification Harness

- **State:** in_review
- **Owner:** local
- **Issue:** #11107
- **Branch:** `feat/mmr-07-simscape-continuous-replay-11107`
- **Paths:** `src/shared/python/motion_matching/simscape_replay_harness.py`, `tests/unit/motion_matching/test_simscape_continuous_replay_harness.py`, `scripts/matlab/run_simscape_candidate.ps1`, `docs/development/simscape_tour_matching/native_evidence/two_window_fit_9967_102/simscape_replay_qualification.json`, `SPEC.md`
- **Started:** 2026-09-29
- **Last verified:** 2026-09-29 at 27c0d34967 — 22 focused harness tests passed (11 review-fix regressions demonstrated red pre-fix, green post-fix); `tests/unit/motion_matching` 1901 passed, 11 failed all pre-existing (`c3d_reader.load_c3d`, `bunkershot3d`, stability-matrix precondition — verified identical on a pristine baseline checkout); ruff check/format and mypy clean on touched modules.
- **Summary:** Added Simscape continuous-replay qualification harness with full-rate continuous trajectory validation, fail-closed contracts for missing samples, non-finite states, timestamp monotonicity, non-R2025b releases, motion prescription, and candidate hash mismatches. Structured qualification receipt schema records per-marker/phase channels, provenance, and evaluates against frozen G1 acceptance gates. Native candidate run-102 honestly rejected at terminal phase (40.3 mm vs 35 mm G1 ceiling). Review fixes (PR #11126 Codex P1/P2): metrics derive from the canonical `compute_replay_five_metrics()` (0.60 s early window, canonical club labels), pelvis yaw measured, unmeasured contact quantities disclosed as unavailable, elapsed-span/start validation, receipt loader null preservation, and `load_replay_evidence_inputs()` deriving digest/initial conditions/control identity from native evidence (run-102 receipt extended additively with `control_identity`, no measured value changed).
- **Next step:** CI green, PR review.
```

## #11098 source occurrence 1

```text
### DL-#11098 — Historical-Video Evidence Review Workflow

- **State:** in_review
- **Owner:** local
- **Issue:** #11098
- **Branch:** `feat/mmr-12-historical-video-review-11098`
- **Paths:** `src/shared/python/shadow_tracker/service.py`, `src/tools/shadow_tracker/gui.py`, `tests/unit/shadow_tracker/test_service.py`, `tests/tools/shadow_tracker/test_shadow_tracker_gui.py`
- **Started:** 2026-09-29
- **Last verified:** 2026-09-29 at 0c31f341a7 — 330 tests collected across `tests/unit/shadow_tracker/` and `tests/tools/shadow_tracker/`, all passing (review-fix regressions demonstrated red pre-fix, green post-fix); ruff check/format and mypy clean on touched modules.
- **Summary:** Preserves multi-shot frame isolation and revision lineage in `DefaultShadowTrackerService`, implements video ingestion with container PTS and slow-motion affine timing mappings, bounded decode limits with prompt cancellation, recoverable session state on corrupt media, and keyboard review scrubbing with honest automated fit refusal in `ShadowTrackerWidget`. Review fixes (PR #11127 Codex P1s): repeat video imports stage and validate scope/revision/asset ownership before any session mutation (atomic failure), no-mapping imports keep `physical_time_s=None` with the canonical unknown-time reason (PTS authority stays in `timing_mode`/`clock_evidence`), and the viewport renders unknown physical time safely instead of formatting `None`.
- **Next step:** Commit and submit PR; release agent lease.
```

## #11097 source occurrence 1

```text
### DL-#11097 — Repair Reduced-Model and Club-Only Product Claims (MMR-11)

- **State:** in_review
- **Owner:** local
- **Issue:** #11097
- **Branch:** `feat/mmr-11-reduced-model-claims-11097`
- **Paths:** `src/shared/python/tour_baselines/coverage.py`, `src/shared/python/tour_baselines/baseline_package.py`, `src/shared/python/tour_baselines/qualification.py`, `src/shared/python/motion_matching/provider.py`, `src/engines/physics_engines/pendulum/python/motion_matching/provider.py`, `src/shared/python/motion_matching/club_only/ui_integration.py`, `src/shared/python/motion_matching/club_only/matrix_qualification.py`, `tests/unit/motion_matching/test_fit_options_dbc.py`
- **Started:** 2026-09-29
- **Last verified:** 2026-09-29 at `46b804bc69` — scoped runs: `pytest tests/unit/motion_matching/test_fit_options_dbc.py tests/unit/engines/physics_engines/pendulum/test_motion_matching_provider.py` (18 passed), and `pytest tests/unit/tour_baselines/test_coverage_matrix.py tests/unit/tour_baselines/test_qualification.py tests/unit/tour_baselines/test_baseline_packages.py tests/unit/motion_matching/test_club_matrix_qualification.py tests/unit/motion_matching/test_club_ui_integration.py` (86 passed). New TDD tests observed RED pre-fix (planar-floor gate used the calibrated plane's stale residual; missing `max_marker_rmse_m` DbC validation, NaN ceiling silently disabled the gate) and GREEN post-fix; `ruff check` clean on changed files. Not run / not available: full project suite, mypy, CI gates, raw-to-package regeneration, native runs.
- **Summary:** Added code-level, fail-closed claims gates: four driven-triple pendulum receipts (`driver_amateur`, `driver_elite`, `iron_amateur`, `iron_elite`) disqualified at projection time (evidence files on disk unchanged); promoted-package hash and out-of-plane-residual integrity checks; club-only fresh continuous replay + labeled inferred posture; matrix all-complete claims fail closed on unresolved cells; pendulum planar-floor early rejection now driven by the CURRENT target's plane distances with a DbC-validated `max_marker_rmse_m`.
- **Open (per MMR-11 acceptance, not satisfied here):** raw-to-package reproduction/regeneration of the reported baselines, Board-selected required club-only cells, and native qualification runs. PR references (not "Closes") #11097 for these.
```

## #11102 source occurrence 1

```text
### DL-#11102 — Publish a Best-Candidate Viewer With Honest Residuals

- **State:** in_review (review fixes for Codex findings on PR #11132)
- **Owner:** local
- **Issue:** #11102
- **Branch:** `feat/mmr-16-best-candidate-viewer-11102`
- **Paths:** `src/tools/tour_matching_viewer/core.py`, `src/tools/tour_matching_viewer/gui.py`, `src/tools/matched_swing_browser/model.py`, `src/tools/matched_swing_browser/gui.py`, `tests/unit/tools/test_tour_matching_viewer_residuals.py`, `tests/unit/tools/test_matched_swing_browser_best_candidate.py`, `tests/unit/tools/test_tour_matching_viewer_combo.py`
- **Started:** 2026-09-29
- **Last verified:** e86a4d4f1c — scoped pytest with `/tmp/ud-venv-11132` (PyQt6 + mujoco, offscreen): red outcomes recorded pre-fix for every review finding, then `tests/unit/tools/test_matched_swing_browser_best_candidate.py tests/unit/tools/test_tour_matching_viewer_residuals.py` 21 passed, targeted viewer suites (core, playback, combo, forces, native_button, adapter) 29 passed; `ruff check` and `ruff format --check` clean on changed files. Project-wide suites, web/API surfaces and mypy were not run in this slice.
- **Summary:** Published best-candidate viewer with honest residuals (MMR-16): enforced raw observation immutability on ReplayData, added residual vector computation, board-ready export stills with complete metadata banners, drive mode filtering, comparable candidate ranking in ascending RMS error preserving rejected verdicts, worst-residual jump navigation, camera and appearance presets preserving invariant physics scores, and graceful missing-engine recovery. Review fixes: wired `rank_candidates` into the browser's real list-build path (best comparable candidate first, rejected rows preserved), pooled per-marker 3D distances for the receipt-convention RMS in the residual summary, rendered-frame and multi-candidate captions, excluded zero-valid-marker frames from global-worst selection and mean statistics, and forwarded the selected row's provenance and verdict into the viewer load path.
- **Remaining scope on #11102:** web/API surface parity, accessibility/native visual review and the remaining acceptance checkboxes; desktop PyQt fixes above do not close them.
- **Next step:** main-lane rebase/CI and frontier review of PR #11132.
```

## #11071 source occurrence 1

```text
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
```

## #11008 source occurrence 1

```text
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
```

## #10946 source occurrence 1

```text
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
```

## #9191 source occurrence 1

```text
### DL-#9191 · Govern Screenshot Schema, Capture Metadata, and Qualified Assets

- **State:** in_review
- **Owner:** local
- **Issue:** #9191
- **Branch:** `feat/comp-b3-screenshot-governance-9191`
- **Paths:** `docs/api/contracts/upstreamdrift-companion-screenshots-v1.schema.json`, `docs/api/contracts/upstreamdrift-companion-v1.schema.json`, `scripts/companion_screenshots.py`, `scripts/companion_catalog.py`, `scripts/config/companion_screenshots.v1.json`, `docs/screenshots/`, `tests/companion/test_companion_screenshots.py`, `tests/companion/test_companion_catalog.py`, `tests/companion/test_companion_publication.py`, `tests/fixtures/companion/current-v1.0.0.json`, `SPEC.md`, `docs/development/DEVELOPMENT_LOG.md`, `docs/development/HANDOFF.md`
- **Started:** 2026-09-29
- **Last verified:** 2026-09-29 — 165/165 pass across tests/companion and screenshot verification suites; ruff check clean; ruff format clean; prettier clean; architecture budget OK; DRY duplication gate OK; hardcoded style ratchet OK.
- **Summary:** Complete implementation of COMP-B3 screenshot governance (#9191): updated schemas with exact capture metadata (`capture_environment`, `source_commit`, pixels/viewport, and summary screenshot counts); implemented `scripts/companion_screenshots.py` with standard library PNG IHDR byte parsing, SHA-256 computation, registry loading/validation, and headless deterministic asset generation; generated 6 representative visual assets in `docs/screenshots/` (pendulum simulator desktop light/dark + mobile dark, project map, rate of closure, tour matching viewer); registered 74 visible programs in `scripts/config/companion_screenshots.v1.json` (6 captured, 70 pending with explicit reasons); wired screenshot registry and inventory into `scripts/companion_catalog.py` and publication bundle.
- **Next step:** Create PR and release lease.
```

## #10921 source occurrence 1

```text
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
```

## #10883 source occurrence 1

```text
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
```

## #10849 source occurrence 1

```text
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
```

## #10597 source occurrence 1

```text
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
```

## #10751 source occurrence 1

```text
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
```

## #10743 source occurrence 1

```text
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
```

## #8883 source occurrence 1

```text
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
```

## #9479 source occurrence 1

```text
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
```

## #10344 source occurrence 1

```text
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
```

## #10376 source occurrence 1

```text
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
```

## #10436 source occurrence 1

```text
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
```

## #8930 source occurrence 1

```text
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
```

## #10359 source occurrence 1

```text
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
```

## #10348 source occurrence 1

```text
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
```

## #10361 source occurrence 1

```text
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
```

## #10366 source occurrence 1

```text
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
```

## #10434 source occurrence 1

```text
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
```

## #10589 source occurrence 1

```text
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
```

## #10433 source occurrence 1

```text
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
```

## #10440 source occurrence 1

```text
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
```

## #8684 source occurrence 1

```text
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
```

## #10439 source occurrence 1

```text
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
```

## #10623 source occurrence 1

```text
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
```

## #10622 source occurrence 1

```text
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
```

## #10432 source occurrence 1

```text
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
```

## #10533 source occurrence 1

```text
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
```

## #10532 source occurrence 1

```text
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
```

## #10530 source occurrence 1

```text
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
```

## #10528 source occurrence 1

```text
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
```

## #10525 source occurrence 1

```text
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
```

## #10524 source occurrence 1

```text
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
```

## #10523 source occurrence 1

```text
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
```

## #10522 source occurrence 1

```text
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
```

## #10531 source occurrence 1

```text
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
```

## #10353 source occurrence 1

```text
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
```

## #10358 source occurrence 1

```text
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
```

## #10529 source occurrence 1

```text
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
```

## #10520 source occurrence 1

```text
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
```

## #10519 source occurrence 1

```text
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
```

## #10518 source occurrence 1

```text
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
```

## #10517 source occurrence 1

```text
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
```

## #10514 source occurrence 1

```text
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
```

## #10482 source occurrence 1

```text
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
```

## #10460 source occurrence 1

```text
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
```

## #10233 source occurrence 1

```text
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
```

## #10403 source occurrence 1

```text
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
```

## #10401 source occurrence 1

```text
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
```

## #10397 source occurrence 1

```text
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
```

## #10395 source occurrence 1

```text
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
```

## #10323 source occurrence 1

```text
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
```

## #10334 source occurrence 1

```text
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
```

## #8766 source occurrence 1

```text
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
```
