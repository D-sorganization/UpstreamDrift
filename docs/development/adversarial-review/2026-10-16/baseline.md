# Mechanical Baseline — UpstreamDrift (2026-10-16)

Every tracked Python file was scanned. This page is evidence for the review, not a finding list; the reviewer
decided which of these matter and wrote those up as findings.

## The Repo's Own Ruff Check

Command: `ruff check .`

```
warning: Invalid `# noqa` directive on src/tools/character_builder/gui.py:22: expected a comma-separated list of codes (e.g., `# noqa: F401, F841`).
warning: Invalid `# noqa` directive on src/tools/grip_wrench_plots/gui.py:63: expected a comma-separated list of codes (e.g., `# noqa: F401, F841`).
warning: Invalid `# noqa` directive on src/tools/ground_reaction_plots/gui.py:54: expected a comma-separated list of codes (e.g., `# noqa: F401, F841`).
warning: Invalid `# noqa` directive on src/tools/impact_parameters_panel/gui.py:147: expected a comma-separated list of codes (e.g., `# noqa: F401, F841`).
warning: Invalid `# noqa` directive on src/tools/necromatcher/gui.py:3: expected a comma-separated list of codes (e.g., `# noqa: F401, F841`).
10	LOG004 	log-exception-outside-except-handler
 6	FURB192	sorted-min-max
Found 16 errors.
No fixes available (16 hidden fixes can be enabled with the `--unsafe-fixes` option).
```

## Isolated Adversarial Ruff Rules

Rules the repo does not enable; counts over `src` and `scripts`.

| count | rule                                               |
| ----- | -------------------------------------------------- |
| 24173 | `TRY003` [ ] raise-vanilla-args                    |
| 4181  | `PLC0415` [ ] import-outside-top-level             |
| 3667  | `S101` [ ] assert                                  |
| 1424  | `RUF100` [*] unused-noqa                           |
| 1347  | `N806` [ ] non-lowercase-variable-in-function      |
| 903   | `PLR0917` [ ] too-many-positional-arguments        |
| 841   | `ARG005` [ ] unused-lambda-argument                |
| 631   | `F401` [-] unused-import                           |
| 598   | `T201` [ ] print                                   |
| 532   | `C901` [ ] complex-structure                       |
| 499   | `TRY004` [ ] type-check-without-type-error         |
| 408   | `ARG002` [ ] unused-method-argument                |
| 374   | `TRY400` [ ] error-instead-of-exception            |
| 358   | `ARG001` [ ] unused-function-argument              |
| 337   | `RUF001` [ ] ambiguous-unicode-character-string    |
| 331   | `RUF022` [-] unsorted-dunder-all                   |
| 313   | `RUF002` [ ] ambiguous-unicode-character-docstring |
| 313   | `TRY300` [ ] try-consider-else                     |
| 307   | `RUF046` [-] unnecessary-cast-to-int               |
| 278   | `PTH123` [ ] builtin-open                          |
| 278   | `PLR0912` [ ] too-many-branches                    |
| 257   | `E402` [ ] module-import-not-at-top-of-file        |
| 242   | `ERA001` [ ] commented-out-code                    |
| 241   | `N803` [ ] invalid-argument-name                   |
| 239   | `N802` [ ] invalid-function-name                   |

## Structure From the AST

- functions: 107,278; files > 800 lines: 252; functions > 80 lines: 1168; functions > 6 params: 60
- largest files:
  - 1979 lines `tests/ci/test_ci_infrastructure.py`
  - 1946 lines `tests/opensim/test_native_moco_ladder_harness.py`
  - 1939 lines `tests/opensim/native_moco_ladder_worker.py`
  - 1638 lines `src/tools/launch_monitor_analytics/gui.py`
  - 1589 lines `src/shared/python/motion_matching/full_body_forward_dynamics.py`
  - 1559 lines `tests/opensim/native_moco_ladder_harness.py`
  - 1488 lines `tests/unit/api/test_simulation_ws.py`
  - 1465 lines `src/shared/python/motion_matching/acceptance.py`
  - 1446 lines `tests/conftest.py`
  - 1421 lines `src/shared/python/motion_matching/pipeline/cli.py`
  - 1345 lines `tests/unit/engines/myosuite/test_native_direct_model_replay.py`
  - 1344 lines `tests/unit/shadow_tracker/test_silhouette_segmentation.py`
- longest functions:
  - 603 lines `docs/development/simscape_tour_matching/native_evidence/two_window_fit_9967_101/two_window_fit_101.py:97` `main`
  - 590 lines `docs/development/simscape_tour_matching/native_evidence/two_window_fit_9967_102/two_window_fit_102.py:101` `main`
  - 576 lines `docs/development/simscape_tour_matching/native_evidence/two_window_fit_9967_100/two_window_fit_100.py:93` `main`
  - 531 lines `src/shared/python/motion_matching/multi_shooting_fit.py:204` `fit_multiple_shooting`
  - 520 lines `scripts/match_pinocchio_c3d.py:53` `match_pinocchio_c3d`
  - 363 lines `src/shared/python/motion_matching/anthropometric_geometry.py:220` `build_upper_body`
  - 352 lines `src/shared/python/chat/_qt/ui_builder.py:106` `build_chat_dock_ui`
  - 335 lines `docs/development/simscape_tour_matching/native_evidence/reproduction/constrained_sextic_transition_fit.py:129` `main`
  - 323 lines `src/shared/python/motion_matching/contact_force_allocator.py:233` `allocate`
  - 307 lines `src/engines/physics_engines/pinocchio/python/full_body_fit.py:559` `run_fit`
  - 298 lines `scripts/research/proximal_distal_energy/run_hand_path_attribution_study.py:693` `_make_figures`
  - 296 lines `tests/opensim/test_native_mechanics_profile.py:232` `test_native_mechanics_profile_crosses_moco_export_and_fresh_replay`
  - 291 lines `tests/unit/engines/mujoco/test_native_full_body_replay.py:364` `test_bundle_builder_uses_t01_generalized_effort_and_complete_qv`
  - 268 lines `tests/soak/realtime/test_24h_soak.py:266` `test_realtime_ws_soak`
  - 262 lines `src/shared/python/motion_matching/pipeline/cli.py:166` `build_parser`

## Test-Suite Texture

Greps over `tests/` and `src/`.

- `pytest.skip` / `@pytest.mark.skip` sites: 923
- `MagicMock` / `patch(` sites: 5830
- literal `assert True` / `assert 1` lines: 49
  - `tests/bunkershot3d/solvers/mpm/test_verification.py:444`
  - `tests/bunkershot3d/solvers/test_rotation_coupling_9544.py:355`
  - `tests/bunkershot3d/solvers/test_rotation_coupling_9544.py:356`
  - `tests/bunkershot3d/sand/test_presets.py:95`
  - `tests/bunkershot3d/domain/test_value_objects.py:202`
  - `tests/bunkershot3d/domain/test_value_objects.py:290`
  - `tests/motion_matching/mujoco_mjcf/test_synthesize.py:96`
  - `tests/research/wave6_research/test_deformable_objects.py:496`
  - `tests/engines/opensim/test_opensim_synthesize.py:163`
  - `tests/heavy_integration/test_pinocchio_recovery.py:69`

## Bandit Findings at Medium and High Severity

| sev    | id   | text                                                                   | n   |
| ------ | ---- | ---------------------------------------------------------------------- | --- |
| MEDIUM | B608 | Possible SQL injection vector through string-based query construction. | 2   |
| MEDIUM | B604 | Function call with shell=True parameter identified, possible security  | 2   |

## Dangerous-Pattern Greps

Over `src` and `scripts`; many `exec(` hits are Qt `dialog.exec()` — the reviewer triaged these.

- `exec(`: 174
- `eval(`: 41
- `shell=True`: 9
- `md5`: 4
- `yaml.load(`: 1

## Radon: Functions at Cyclomatic Complexity D, E or F

Count at or above 21: 237

- `C 35:0 DataReader - D (21)`
- `M 30:4 SegmentLoadSeries.__post_init__ - D (25)`
- `C 34:0 MarkerBinding - D (29)`
- `M 57:4 MarkerBinding.__post_init__ - D (28)`
- `M 137:4 _SimRecordingMixin._record_dynamics_step - D (22)`
- `F 111:0 detect_gh_intent - D (28)`
- `F 102:0 cmd_convert - D (21)`
- `F 607:0 _check_contract_reasons - D (27)`
- `F 135:0 derive_visual_skeleton - E (35)`
- `M 206:4 UnifiedParityReport.render_markdown - D (23)`
- `F 220:0 build_upper_body - F (42)`
- `M 97:4 FullBodyPolynomialControl.from_spec - D (24)`
- `F 277:0 hip_rotation_zero - E (34)`
- `F 141:0 functional_hip_calibration - D (21)`
- `F 182:0 classify_receipt - F (45)`
- `M 511:4 DistributedFeedbackPolicy.request_for_step - D (24)`
- `M 114:4 LoopTuningProblem.__post_init__ - D (21)`
- `F 330:0 _validate_fit_inputs - D (21)`
- `F 299:0 _extract_report_provenance - D (27)`
- `F 120:0 _aggregate_group_runs - D (28)`
- `F 240:0 _check_capture_parity - D (22)`
- `F 204:0 fit_multiple_shooting - F (49)`
- `C 58:0 MultipleShootingOptions - D (29)`
- `M 116:4 MultipleShootingOptions.__post_init__ - D (28)`
- `F 17:0 solve_equality_least_squares - E (33)`

## Vulture Dead-Code Hits

At 80 % confidence or higher: 172 hits

- `src/api/auth/middleware.py:25: unused variable 'permission' (100% confidence)`
- `src/api/routes/theme.py:62: unused variable 'theme_id' (100% confidence)`
- `src/engines/common/simulation_control.py:262: unused variable 'body_c' (100% confidence)`
- `src/engines/physics_engines/drake/python/drake_physics_engine.py:44: unused import 'LoadModelDirectives' (90% confidence)`
- `src/engines/physics_engines/drake/python/drake_physics_engine.py:44: unused import 'ProcessModelDirectives' (90% confidence)`
- `src/engines/physics_engines/drake/python/motion_matching/fit_swing_autodiff.py:635: unused variable 'dtype_factory' (100% confidence)`
- `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/linkage_mechanisms/special_mechanisms.py:12: unused variable 'num_slots' (100% confidence)`
- `src/engines/physics_engines/opensim/python/tour_matching/document_ik.py:21: unused import 'ANTHRO_DOCUMENT_MARKER_BODIES' (90% confidence)`
- `src/engines/physics_engines/opensim/python/tour_matching/document_ik.py:21: unused import 'build_ik_task_set' (90% confidence)`
- `src/engines/physics_engines/opensim/python/tour_matching/muscle_qualification.py:605: unused variable 'short_replay_duration_s' (100% confidence)`
- `src/engines/physics_engines/opensim/python/tour_matching/registration.py:197: unused variable 'stance_left_labels' (100% confidence)`
- `src/engines/physics_engines/opensim/python/tour_matching/registration.py:198: unused variable 'stance_right_labels' (100% confidence)`
- `src/engines/physics_engines/pinocchio/python/dtack/viz/swing_dataset_viewer.py:112: unused variable 'show_acceleration' (100% confidence)`
- `src/engines/physics_engines/pinocchio/python/marker_kinematics.py:521: unused variable 'temporal_regularisation' (100% confidence)`
- `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/option3_inverse_nn/inverse_cvae.py:98: unused variable 'h_x' (100% confidence)`
- `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Simscape Multibody Data Plotters/Python Version/integrated_golf_gui_r0/golf_opengl_renderer.py:482: unused variable 'dynamics_data' (100% confidence)`
- `src/launchers/_launcher_navigation_ui.py:262: redundant if-condition (100% confidence)`
- `src/launchers/_launcher_navigation_ui.py:704: redundant if-condition (100% confidence)`
- `src/launchers/_launcher_top_bar_ui.py:581: unused variable 'sync_combo' (100% confidence)`
- `src/launchers/_launcher_top_bar_ui.py:601: redundant if-condition (100% confidence)`
- `src/launchers/_launcher_top_bar_ui.py:605: redundant if-condition (100% confidence)`
- `src/launchers/_launcher_top_bar_ui.py:607: redundant if-condition (100% confidence)`
- `src/launchers/_launcher_top_bar_ui.py:609: redundant if-condition (100% confidence)`
- `src/launchers/_launcher_top_bar_ui.py:616: redundant if-condition (100% confidence)`
- `src/launchers/_launcher_top_bar_ui.py:621: redundant if-condition (100% confidence)`
