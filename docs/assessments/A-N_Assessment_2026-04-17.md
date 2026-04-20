# A-N Assessment - UpstreamDrift - 2026-04-17

Run time: 2026-04-17T08:01:19.6221680Z UTC
Sync status: synced
Sync notes: local changes stashed before sync. Already up to date.
From https://github.com/D-sorganization/UpstreamDrift
 * branch                bolt/optimize-array-reductions-17265335108285615514 -> FETCH_HEAD

Overall grade: C (70/100)

## Coverage Notes
- Reviewed tracked first-party files from git ls-files, excluding cache, build, vendor, virtualenv, temp, and generated output directories.
- Reviewed 7042 tracked files, including 3505 code files, 1159 test files, 67 CI files, 43 config/build files, and 1014 docs/onboarding files.
- This is a read-only static assessment of committed files. TDD history and confirmed Law of Demeter semantics require commit-history review and deeper call-graph analysis; this report distinguishes those limits from confirmed file evidence.

## Category Grades
### A. Architecture and Boundaries: C (72/100)
Assesses source organization and boundary clarity from tracked first-party layout.
- Evidence: `7042 tracked first-party files`
- Evidence: `5042 files under source-like directories`

### B. Build and Dependency Management: B (84/100)
Assesses committed build, dependency, and tool configuration.
- Evidence: `Dockerfile`
- Evidence: `Dockerfile.heavy_test`
- Evidence: `Makefile`
- Evidence: `docker-compose.gpu.yml`
- Evidence: `docker-compose.yml`
- Evidence: `environment.yml`
- Evidence: `installer/windows/setup.py`
- Evidence: `package.json`
- Evidence: `pyproject.toml`
- Evidence: `rust_core/upstream-physics/pyproject.toml`

### C. Configuration and Environment Hygiene: C (78/100)
Checks whether runtime and developer configuration is explicit.
- Evidence: `Dockerfile`
- Evidence: `Dockerfile.heavy_test`
- Evidence: `Makefile`
- Evidence: `docker-compose.gpu.yml`
- Evidence: `docker-compose.yml`
- Evidence: `environment.yml`
- Evidence: `installer/windows/setup.py`
- Evidence: `package.json`
- Evidence: `pyproject.toml`
- Evidence: `rust_core/upstream-physics/pyproject.toml`

### D. Contracts, Types, and Domain Modeling: B (82/100)
Design by Contract evidence includes validation, assertions, typed models, explicit raised errors, and invariants.
- Evidence: `.gaai/core/scripts/backlog-scheduler.sh`
- Evidence: `build_hooks.py`
- Evidence: `docs/sphinx/_static/jquery.js`
- Evidence: `examples/02_parameter_sweeps.py`
- Evidence: `examples/aerodynamics_demo.py`
- Evidence: `examples/motion_training_demo.py`
- Evidence: `examples/topography_demo.py`
- Evidence: `installer/windows/packaging_profiles.py`
- Evidence: `installer/windows/setup_config.py`
- Evidence: `launch_golf_suite.py`

### E. Reliability and Error Handling: C (76/100)
Reliability is graded from test presence plus explicit validation/error-handling signals.
- Evidence: `.agent/skills/tests/SKILL.md`
- Evidence: `.claude/skills/tests/SKILL.md`
- Evidence: `docs/assessments/Assessment_C_Test_Coverage.md`
- Evidence: `docs/assessments/issues/issue_test_coverage.md`
- Evidence: `docs/development/TEST_AUDIT_REPORT.md`
- Evidence: `.gaai/core/scripts/backlog-scheduler.sh`
- Evidence: `build_hooks.py`
- Evidence: `docs/sphinx/_static/jquery.js`
- Evidence: `examples/02_parameter_sweeps.py`
- Evidence: `examples/aerodynamics_demo.py`

### F. Function, Module Size, and SRP: F (55/100)
Evaluates function size, script/module size, and single responsibility using static size signals.
- Evidence: `.gaai/core/scripts/backlog-scheduler.sh (509 lines)`
- Evidence: `.gaai/core/scripts/delivery-daemon.sh (1190 lines)`
- Evidence: `docs/sphinx/_static/jquery.js (5586 lines)`
- Evidence: `docs/sphinx/_static/searchtools.js (693 lines)`
- Evidence: `docs/sphinx/searchindex.js (950 lines)`
- Evidence: `rust_core/upstream-physics/src/aerodynamics.rs (684 lines)`
- Evidence: `scripts/create_assessment_issues.sh (915 lines)`
- Evidence: `.gaai/core/scripts/backlog-scheduler.sh (coarse avg 170 lines/definition)`
- Evidence: `.gaai/core/scripts/delivery-metrics.sh (coarse avg 215 lines/definition)`
- Evidence: `docs/sphinx/_static/_sphinx_javascript_frameworks_compat.js (coarse avg 137 lines/definition)`

### G. Testing and TDD Posture: B (82/100)
TDD history cannot be confirmed statically; grade reflects committed automated test posture.
- Evidence: `.agent/skills/tests/SKILL.md`
- Evidence: `.claude/skills/tests/SKILL.md`
- Evidence: `docs/assessments/Assessment_C_Test_Coverage.md`
- Evidence: `docs/assessments/issues/issue_test_coverage.md`
- Evidence: `docs/development/TEST_AUDIT_REPORT.md`
- Evidence: `docs/development/TEST_COVERAGE_REPORT.md`
- Evidence: `docs/development/heavy_test_coverage_issues.md`
- Evidence: `output/simulations/mujoco/test_20260108_084325.csv`
- Evidence: `output/simulations/mujoco/test_20260108_084442.csv`
- Evidence: `output/simulations/mujoco/test_20260108_084451.csv`
- Evidence: `output/simulations/mujoco/test_20260108_084500.csv`
- Evidence: `output/simulations/mujoco/test_20260108_084509.csv`

### H. CI/CD and Automation: C (78/100)
Checks for tracked CI/CD workflow files.
- Evidence: `.github/workflows/Bot-CI-Trigger.yml`
- Evidence: `.github/workflows/Code-Metrics.yml`
- Evidence: `.github/workflows/Comment-to-Issue-Converter.yml`
- Evidence: `.github/workflows/Jules-Archivist.yml`
- Evidence: `.github/workflows/Jules-Assessment-AutoFix.yml`
- Evidence: `.github/workflows/Jules-Assessment-Generator.yml`
- Evidence: `.github/workflows/Jules-Assessment-Remediator.yml`
- Evidence: `.github/workflows/Jules-Auto-Assign-Issues.yml`
- Evidence: `.github/workflows/Jules-Auto-Repair.yml`
- Evidence: `.github/workflows/Jules-Code-Quality-Fixer.yml`

### I. Security and Secret Hygiene: F (35/100)
Secret scan is regex-based; findings require manual confirmation.
- Evidence: `src/shared/python/ai/adapters/anthropic_adapter.py`
- Evidence: `src/shared/python/ai/adapters/openai_adapter.py`
- Evidence: `src/shared/python/ai/config.py`
- Evidence: `tests/api/test_auth_security.py`
- Evidence: `tests/unit/api/test_auth_models.py`
- Evidence: `tests/unit/api/test_security.py`
- Evidence: `tests/unit/shared_python/ai/adapters/test_gemini_adapter.py`
- Evidence: `tests/unit/test_api_security.py`
- Evidence: `tests/unit/test_issue_fixes_1777_1778_1779_1782.py`
- Evidence: `tests/unit/test_security_and_module_fixes.py`

### J. Documentation and Onboarding: B (82/100)
Checks docs, README, onboarding, and release documents.
- Evidence: `.Jules/bolt.md`
- Evidence: `.Jules/palette.md`
- Evidence: `.agent/skills/issues-10-sequential/SKILL.md`
- Evidence: `.agent/skills/issues-5-combined/SKILL.md`
- Evidence: `.agent/skills/lint/SKILL.md`
- Evidence: `.agent/skills/tests/SKILL.md`
- Evidence: `.agent/skills/update-issues/SKILL.md`
- Evidence: `.agent/workflows/issues-10-sequential.md`
- Evidence: `.agent/workflows/issues-5-combined.md`
- Evidence: `.agent/workflows/lint.md`
- Evidence: `.agent/workflows/tests.md`
- Evidence: `.agent/workflows/update-issues.md`

### K. Maintainability, DRY, and Duplication: F (55/100)
DRY is assessed through duplicate filename clusters and TODO/FIXME density as static heuristics.
- Evidence: `__main__ appears in 5 files`
- Evidence: `analyzer appears in 7 files`
- Evidence: `base appears in 6 files`
- Evidence: `code_quality_check appears in 7 files`
- Evidence: `codeissuesgui appears in 4 files`
- Evidence: `docs/sphinx/conf.py`
- Evidence: `docs/sphinx/searchindex.js`
- Evidence: `scripts/analyze_completist_data.py`
- Evidence: `scripts/generate_todo_fixme_register.py`
- Evidence: `scripts/pragmatic_programmer_review.py`

### L. API Surface and Law of Demeter: F (58/100)
Law of Demeter is approximated with deep member-chain hints; confirmed violations require semantic review.
- Evidence: `build_hooks.py`
- Evidence: `docs/sphinx/_static/_sphinx_javascript_frameworks_compat.js`
- Evidence: `docs/sphinx/_static/jquery.js`
- Evidence: `docs/sphinx/_static/js/badge_only.js`
- Evidence: `docs/sphinx/_static/js/theme.js`
- Evidence: `docs/sphinx/_static/js/versions.js`
- Evidence: `docs/sphinx/_static/searchtools.js`
- Evidence: `docs/sphinx/_static/sphinx_highlight.js`
- Evidence: `docs/sphinx/searchindex.js`
- Evidence: `examples/01_basic_simulation.py`

### M. Observability and Operability: C (74/100)
Checks for logging, metrics, monitoring, and operational artifacts.
- Evidence: `.gaai/core/scripts/delivery-metrics.sh`
- Evidence: `.gaai/core/skills/cross/success-metrics-evaluation/SKILL.md`
- Evidence: `.github/workflows/Code-Metrics.yml`
- Evidence: `.github/workflows/agent-metrics-dashboard.yml`
- Evidence: `docs/assessments/Assessment_L_Logging.md`
- Evidence: `docs/development/LOGGING_STYLE_GUIDE.md`
- Evidence: `docs/engineering/logging-policy.md`
- Evidence: `src/engines/Simscape_Multibody_Models/2D_Golf_Model/matlab/GolfSwingLogging.slx`
- Evidence: `src/engines/Simscape_Multibody_Models/2D_Golf_Model/python/src/logger_utils.py`
- Evidence: `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/scripts/post_processing/postprocess_golf_metrics.m`

### N. Governance, Licensing, and Release Hygiene: C (74/100)
Checks ownership, release, contribution, security, and license metadata.
- Evidence: `.gaai/core/skills/cross/security-audit/SKILL.md`
- Evidence: `.github/CODEOWNERS`
- Evidence: `.github/workflows/docker-security-scan.yml`
- Evidence: `CHANGELOG.md`
- Evidence: `CONTRIBUTING.md`
- Evidence: `LICENSE`
- Evidence: `SECURITY.md`
- Evidence: `docs/assessments/Assessment_F_Security.md`
- Evidence: `docs/assessments/issues/Issue_2027_Incomplete_Stub_in_security_py_330.md`
- Evidence: `docs/assessments/issues/Issue_2144_Incomplete_Stub_in_security_py_328.md`

## Explicit Engineering Practice Review
- TDD: Automated tests are present, but red-green-refactor history is not confirmable from static files.
- DRY: Duplicate responsibility clusters require review: __main__ appears in 5 files; analyzer appears in 7 files; base appears in 6 files; code_quality_check appears in 7 files; codeissuesgui appears in 4 files
- Design by Contract: Validation/contract signals were found in tracked code.
- Law of Demeter: Deep member-chain hints were found and should be semantically reviewed.
- Function size and SRP: Large modules or coarse long-definition signals were found.

## Key Risks
- Large modules/scripts reduce maintainability and SRP clarity.
- Potential hard-coded secret patterns require manual security review.
- Repeated filename clusters suggest possible duplicated responsibilities.
- Deep member-chain usage may indicate Law of Demeter pressure points.

## Prioritized Remediation Recommendations
1. Split the largest modules by responsibility and add characterization tests before refactoring.
2. Review duplicate filename/responsibility clusters and extract shared helpers only where behavior is truly repeated.
3. Review deep member chains and introduce boundary methods where object graph traversal leaks across modules.

## Actionable Issue Candidates
### Split oversized modules by responsibility
- Severity: medium
- Problem: Oversized files found: .gaai/core/scripts/backlog-scheduler.sh (509 lines); .gaai/core/scripts/delivery-daemon.sh (1190 lines); docs/sphinx/_static/jquery.js (5586 lines); docs/sphinx/_static/searchtools.js (693 lines); docs/sphinx/searchindex.js (950 lines); rust_core/upstream-physics/src/aerodynamics.rs (684 lines); scripts/create_assessment_issues.sh (915 lines); scripts/mypy_autofix_agent.py (755 lines); scripts/refactor_dry_orthogonality.py (538 lines); src/api/auth/security.py (510 lines); src/api/diagnostics.py (623 lines); src/api/local_server.py (770 lines); src/api/models/requests.py (512 lines); src/api/models/responses.py (660 lines); src/api/routes/analysis_tools.py (559 lines); src/api/routes/force_overlays.py (501 lines); src/api/routes/physics.py (621 lines); src/deployment/realtime/controller.py (519 lines); src/engines/Simscape_Multibody_Models/2D_Golf_Model/matlab/2D GUI/main_scripts/golf_swing_analysis_gui.m (4075 lines); src/engines/Simscape_Multibody_Models/2D_Golf_Model/matlab/2D GUI/scripts/performance_analysis_script.m (542 lines); src/engines/Simscape_Multibody_Models/2D_Golf_Model/matlab/2D GUI/visualization/GolfSwingVisualizer.m (1184 lines); src/engines/Simscape_Multibody_Models/2D_Golf_Model/matlab/2D GUI/visualization/InteractiveSignalPlotter.m (772 lines); src/engines/Simscape_Multibody_Models/2D_Golf_Model/matlab/2D GUI/visualization/SkeletonPlotter.m (893 lines); src/engines/Simscape_Multibody_Models/2D_Golf_Model/matlab/2D GUI/visualization/create_advanced_plot_viewer.m (544 lines); src/engines/Simscape_Multibody_Models/2D_Golf_Model/matlab/2D GUI/visualization/create_performance_monitor.m (543 lines); src/engines/Simscape_Multibody_Models/2D_Golf_Model/matlab/2D GUI/visualization/skeleton_plotter_wrapper.m (550 lines); src/engines/Simscape_Multibody_Models/2D_Golf_Model/matlab/Model Output/Scripts/SCRIPT_UpdateCalcsforImpulseandWork.m (601 lines); src/engines/Simscape_Multibody_Models/2D_Golf_Model/matlab/Scripts/SCRIPT_UpdateCalcsforImpulseandWork.m (601 lines); src/engines/Simscape_Multibody_Models/2D_Golf_Model/matlab/Skeleton Plotter/GolfSwingVisualizer.m (1184 lines); src/engines/Simscape_Multibody_Models/2D_Golf_Model/matlab_optimized/visualization/SkeletonPlotter.m (948 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/2D GUI/main_scripts/golf_swing_analysis_gui.m (4083 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/2D GUI/scripts/performance_analysis_script.m (539 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/2D GUI/visualization/GolfSwingVisualizer.m (1181 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/2D GUI/visualization/InteractiveSignalPlotter.m (772 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/2D GUI/visualization/SkeletonPlotter.m (880 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/2D GUI/visualization/create_advanced_plot_viewer.m (544 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/2D GUI/visualization/create_performance_monitor.m (543 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/2D GUI/visualization/skeleton_plotter_wrapper.m (550 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Motion Capture Plotter/motion_capture_plotter_visualization.py (731 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Simscape Multibody Data Plotters/Matlab Versions/SkeletonPlotter/GolfSwingVisualizer.m (1181 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Simscape Multibody Data Plotters/Matlab Versions/SkeletonPlotter/Older Revs/Safe_Copy_of_SkeletonPlotter.m (541 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Simscape Multibody Data Plotters/Matlab Versions/SkeletonPlotter/Older Revs/SkeletonPlotter.m (541 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Simscape Multibody Data Plotters/Python Version/golf_gui_r0/golf_camera_system.py (981 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Simscape Multibody Data Plotters/Python Version/golf_gui_r0/golf_main_application.py (806 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Simscape Multibody Data Plotters/Python Version/golf_gui_r0/golf_visualizer_renderer.py (728 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Simscape Multibody Data Plotters/Python Version/integrated_golf_gui_r0/golf_data_core.py (977 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Simscape Multibody Data Plotters/Python Version/integrated_golf_gui_r0/golf_gui_tabs.py (593 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Simscape Multibody Data Plotters/Python Version/integrated_golf_gui_r0/golf_opengl_renderer.py (990 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Simscape Multibody Data Plotters/Python Version/integrated_golf_gui_r0/golf_video_export.py (600 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Simscape Multibody Data Plotters/Python Version/integrated_golf_gui_r0/golf_wiffle_main.py (773 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Simscape Multibody Data Plotters/Python Version/integrated_golf_gui_r0/wiffle_data_loader.py (695 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/functions/dataset_generator/PostProcessingModule.m (1114 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/functions/dataset_generator/calculateWorkPowerAndGranularAngularImpulse3D.m (668 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/functions/dataset_generator/extractAllSignalsFromBus.m (860 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/functions/dataset_generator/performance_optimizer.m (566 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/scripts/MASTER_SCRIPT_ZTCF_ZVCF_PLOT_GENERATOR_3D.m (502 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/scripts/dataset_generator/Dataset_GUI.m (4850 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/scripts/dataset_generator/runSimulation.m (1307 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/scripts/plotting/PLOT_BASE_Plots.m (720 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/scripts/plotting/PLOT_DELTA_Plots.m (698 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/scripts/plotting/PLOT_ModelData_Plots.m (774 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/scripts/plotting/PLOT_ZTCF_Plots.m (719 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/scripts/runParallelSimulations_14cores.m (694 lines); src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/c3d_reader.py (896 lines); src/engines/common/physics.py (523 lines); src/engines/common/state.py (534 lines); src/engines/pendulum_models/python/double_pendulum_model/physics/double_pendulum.py (559 lines); src/engines/pendulum_models/python/double_pendulum_model/ui/double_pendulum_gui.py (883 lines); src/engines/pendulum_models/python/double_pendulum_model/ui/pendulum_pyqt_app.py (546 lines); src/engines/pendulum_models/python/double_pendulum_model/visualization/double_pendulum_web/app.js (539 lines); src/engines/physics_engines/drake/python/drake_physics_engine.py (622 lines); src/engines/physics_engines/drake/python/perturbation/analyzer.py (506 lines); src/engines/physics_engines/drake/python/src/drake_golf_model.py (811 lines); src/engines/physics_engines/drake/python/src/drake_gui_ui.py (529 lines); src/engines/physics_engines/drake/python/src/drake_ui_mixin.py (638 lines); src/engines/physics_engines/drake/python/src/drake_visualization_mixin.py (733 lines); src/engines/physics_engines/drake/python/src/pose_editor_tab.py (812 lines); src/engines/physics_engines/mujoco/docker/gui/deepmind_control_suite_MuJoCo_GUI.py (951 lines); src/engines/physics_engines/mujoco/docker/src/humanoid_golf/sim.py (681 lines); src/engines/physics_engines/mujoco/golf_swing_models_xml.py (1016 lines); src/engines/physics_engines/mujoco/head_models.py (520 lines); src/engines/physics_engines/mujoco/python/humanoid_launcher_ui.py (541 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/advanced_control.py (690 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/advanced_gui_methods.py (549 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/advanced_kinematics.py (600 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/biomechanics.py (707 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/examples_chaotic_pendulum.py (657 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/grip_modelling_tab.py (1035 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/gui/core/main_window.py (537 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/gui/tabs/controls_tab.py (1076 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/gui/tabs/humanoid_config_tab.py (740 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/gui/tabs/manipulation_tab.py (767 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/gui/tabs/physics_tab.py (506 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/gui/tabs/visualization_tab.py (850 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/interactive_manipulation.py (882 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/inverse_dynamics.py (946 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/motion_capture.py (752 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/motion_optimization.py (756 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/physics_engine.py (715 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/pinocchio_interface.py (613 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/polynomial_generator.py (649 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/power_flow.py (524 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/recording_library.py (750 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/rigid_body_dynamics/aba.py (511 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/sim_rendering_mixin.py (694 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/sim_widget.py (999 lines); src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/urdf_io.py (994 lines); src/engines/physics_engines/myosuite/python/muscle_analysis.py (563 lines); src/engines/physics_engines/myosuite/python/myosuite_physics_engine.py (923 lines); src/engines/physics_engines/myosuite/python/perturbation/analyzer.py (533 lines); src/engines/physics_engines/opensim/python/opensim_physics_engine.py (779 lines); src/engines/physics_engines/opensim/python/perturbation/analyzer.py (542 lines); src/engines/physics_engines/pendulum/python/golf_swing_physics_engine.py (512 lines); src/engines/physics_engines/pinocchio/data/rob_neal/ClubDataGUI_v2.m (564 lines); src/engines/physics_engines/pinocchio/python/dtack/gui/main_window.py (522 lines); src/engines/physics_engines/pinocchio/python/dtack/utils/urdf_exporter.py (526 lines); src/engines/physics_engines/pinocchio/python/motion_training/dual_hand_ik_solver.py (550 lines); src/engines/physics_engines/pinocchio/python/motion_training/motion_visualizer.py (696 lines); src/engines/physics_engines/pinocchio/python/pinocchio_golf/pose_editor_tab.py (787 lines); src/engines/physics_engines/putting_green/python/ball_roll_physics.py (589 lines); src/engines/physics_engines/putting_green/python/green_surface.py (817 lines); src/engines/physics_engines/putting_green/python/simulator.py (841 lines); src/launchers/cross_engine_dashboard.py (827 lines); src/launchers/golf_launcher.py (723 lines); src/launchers/launcher_diagnostics.py (833 lines); src/launchers/launcher_process_manager.py (700 lines); src/launchers/launcher_simulation.py (626 lines); src/launchers/launcher_ui_setup.py (541 lines); src/launchers/settings_dialog.py (559 lines); src/launchers/shot_tracer.py (514 lines); src/learning/imitation/learners.py (911 lines); src/learning/retargeting/retargeter.py (701 lines); src/learning/rl/manipulation_envs.py (553 lines); src/research/deformable/objects.py (677 lines); src/research/differentiable/engine.py (635 lines); src/research/mpc/controller.py (579 lines); src/research/multi_robot/coordination.py (617 lines); src/robotics/control/whole_body/wbc_controller.py (804 lines); src/robotics/locomotion/footstep_planner.py (611 lines); src/robotics/planning/collision/collision_checker.py (527 lines); src/robotics/planning/collision/geometric_primitives.py (751 lines); src/robotics/planning/motion/rrt_star.py (517 lines); src/shared/models/opensim/opensim-models/Copy_of_Tutorial_7_Set_up_OpenSim_Moco_in_Google_Colab.ipynb (813 lines); src/shared/python/ai/adapters/anthropic_adapter.py (537 lines); src/shared/python/ai/adapters/ollama_adapter.py (507 lines); src/shared/python/ai/adapters/openai_adapter.py (518 lines); src/shared/python/ai/education.py (674 lines); src/shared/python/ai/gui/assistant_panel.py (1020 lines); src/shared/python/ai/gui/settings_dialog.py (787 lines); src/shared/python/ai/sample_tools.py (616 lines); src/shared/python/ai/tool_registry.py (532 lines); src/shared/python/ai/types.py (517 lines); src/shared/python/ai/workflow_definitions.py (620 lines); src/shared/python/ai/workflow_engine.py (641 lines); src/shared/python/analysis/nonlinear_dynamics.py (662 lines); src/shared/python/analysis/reporting.py (534 lines); src/shared/python/cli_utils.py (625 lines); src/shared/python/club_data/club_data_tab.py (575 lines); src/shared/python/club_data/loader.py (663 lines); src/shared/python/club_data/targets.py (513 lines); src/shared/python/config/environment.py (581 lines); src/shared/python/config/handedness_support.py (517 lines); src/shared/python/config/model_pack_manifest.py (618 lines); src/shared/python/contracts.py (709 lines); src/shared/python/control_features_registry.py (518 lines); src/shared/python/control_interface.py (674 lines); src/shared/python/core/error_utils.py (709 lines); src/shared/python/core/type_utils.py (575 lines); src/shared/python/dashboard/advanced_analysis.py (743 lines); src/shared/python/dashboard/recorder.py (817 lines); src/shared/python/dashboard/widgets.py (951 lines); src/shared/python/data_io/dataset_generator/core.py (831 lines); src/shared/python/data_io/export.py (526 lines); src/shared/python/data_io/output_manager.py (880 lines); src/shared/python/data_io/swing_capture_import.py (739 lines); src/shared/python/data_processing/processor.py (603 lines); src/shared/python/engine_core/base_physics_engine.py (556 lines); src/shared/python/engine_core/checkpoint.py (647 lines); src/shared/python/engine_core/engine_probes.py (700 lines); src/shared/python/engine_core/interfaces.py (768 lines); src/shared/python/gui_launcher/launcher.py (677 lines); src/shared/python/gui_pkg/ellipsoid_visualization.py (633 lines); src/shared/python/gui_pkg/help_content.py (575 lines); src/shared/python/gui_pkg/help_system.py (774 lines); src/shared/python/gui_pkg/plot_generator.py (657 lines); src/shared/python/gui_pkg/video_pose_pipeline.py (618 lines); src/shared/python/gui_pkg/viewpoint_controls.py (626 lines); src/shared/python/humanoid_character_builder/core/anthropometry.py (629 lines); src/shared/python/humanoid_character_builder/core/segment_definitions.py (760 lines); src/shared/python/humanoid_character_builder/generators/mesh_generator_makehuman.py (708 lines); src/shared/python/humanoid_character_builder/generators/mesh_generator_smplx.py (714 lines); src/shared/python/humanoid_character_builder/interfaces/api.py (703 lines); src/shared/python/humanoid_character_builder/mesh/collision_generator.py (763 lines); src/shared/python/humanoid_character_builder/mesh/inertia_calculator.py (625 lines); src/shared/python/humanoid_character_builder/mesh/mesh_processor.py (812 lines); src/shared/python/injury/injury_risk.py (611 lines); src/shared/python/injury/joint_stress.py (515 lines); src/shared/python/injury/spinal_load_analysis.py (743 lines); src/shared/python/logging_pkg/logging_config.py (617 lines); src/shared/python/model_generation/api/rest_api.py (621 lines); src/shared/python/model_generation/builders/manual_builder.py (581 lines); src/shared/python/model_generation/builders/parametric_builder.py (665 lines); src/shared/python/model_generation/cli/main.py (824 lines); src/shared/python/model_generation/converters/mjcf_converter.py (632 lines); src/shared/python/model_generation/converters/simscape/mdl_parser.py (612 lines); src/shared/python/model_generation/converters/simscape/simscape_converter.py (823 lines); src/shared/python/model_generation/converters/urdf_parser.py (673 lines); src/shared/python/model_generation/core/physics_validation.py (570 lines); src/shared/python/model_generation/core/types.py (719 lines); src/shared/python/model_generation/core/validation.py (507 lines); src/shared/python/model_generation/editor/editor_modifications.py (782 lines); src/shared/python/model_generation/editor/frankenstein_editor.py (722 lines); src/shared/python/model_generation/editor/text_editor.py (1041 lines); src/shared/python/model_generation/explorer/model_explorer.py (717 lines); src/shared/python/model_generation/inertia/calculator.py (618 lines); src/shared/python/model_generation/library/model_library.py (882 lines); src/shared/python/model_generation/library/repository.py (529 lines); src/shared/python/optimization/swing_optimizer.py (900 lines); src/shared/python/pendulum_simulator/cross_engine_perturbation.py (506 lines); src/shared/python/pendulum_simulator/gui/analysis_tab.py (770 lines); src/shared/python/pendulum_simulator/gui/base_pendulum_widget.py (688 lines); src/shared/python/pendulum_simulator/gui/controls_widget.py (715 lines); src/shared/python/pendulum_simulator/gui/controls_widget_golfer.py (631 lines); src/shared/python/pendulum_simulator/gui/equations_popup_reference_content.py (643 lines); src/shared/python/pendulum_simulator/gui/golfer_pendulum_widget.py (755 lines); src/shared/python/pendulum_simulator/gui/main_window.py (622 lines); src/shared/python/pendulum_simulator/gui/optimization_widget.py (767 lines); src/shared/python/pendulum_simulator/gui/panel_builders.py (900 lines); src/shared/python/pendulum_simulator/gui/pendulum_widget.py (911 lines); src/shared/python/pendulum_simulator/gui/perturbation_panel.py (570 lines); src/shared/python/pendulum_simulator/gui/simulation_panel.py (956 lines); src/shared/python/pendulum_simulator/gui/swing_comparison_dialog.py (536 lines); src/shared/python/pendulum_simulator/gui/toolstrip_widget.py (872 lines); src/shared/python/pendulum_simulator/native_backend.py (762 lines); src/shared/python/pendulum_simulator/pendulum_perturbation_analyzer.py (610 lines); src/shared/python/pendulum_simulator/physics.py (725 lines); src/shared/python/pendulum_simulator/physics_golfer_jax.py (707 lines); src/shared/python/pendulum_simulator/physics_triple.py (617 lines); src/shared/python/perturbation/cross_engine_runner.py (547 lines); src/shared/python/physics/aerodynamics.py (1096 lines); src/shared/python/physics/ball_flight_physics.py (863 lines); src/shared/python/physics/flexible_shaft.py (1031 lines); src/shared/python/physics/flight_models.py (595 lines); src/shared/python/physics/grip_contact_model.py (820 lines); src/shared/python/physics/ground_reaction_forces.py (639 lines); src/shared/python/physics/terrain_engine.py (938 lines); src/shared/python/physics/terrain_mixin.py (501 lines); src/shared/python/physics/terrain_representation.py (1046 lines); src/shared/python/physics/topography.py (743 lines); src/shared/python/plot_engine/matplotlib_renderer.py (587 lines); src/shared/python/plot_theme/themes.py (596 lines); src/shared/python/plotting/core.py (528 lines); src/shared/python/plotting/renderers/coordination.py (945 lines); src/shared/python/plotting/renderers/kinematics.py (637 lines); src/shared/python/plotting/renderers/kinetics.py (794 lines); src/shared/python/pose_editor/library.py (662 lines); src/shared/python/pose_editor/widgets.py (810 lines); src/shared/python/signal_toolkit/calculus.py (635 lines); src/shared/python/signal_toolkit/core.py (596 lines); src/shared/python/signal_toolkit/filters.py (801 lines); src/shared/python/signal_toolkit/fitting.py (887 lines); src/shared/python/signal_toolkit/io.py (720 lines); src/shared/python/signal_toolkit/limits.py (521 lines); src/shared/python/signal_toolkit/noise.py (562 lines); src/shared/python/signal_toolkit/polynomial_generator.py (714 lines); src/shared/python/signal_toolkit/series.py (747 lines); src/shared/python/signal_toolkit/signal_processing.py (948 lines); src/shared/python/signal_toolkit/widget_processing.py (641 lines); src/shared/python/signal_toolkit/widget_ui.py (926 lines); src/shared/python/theme/colors.py (560 lines); src/shared/python/theme/dialogs/theme_manager_dialog.py (516 lines); src/shared/python/theme/style_constants.py (792 lines); src/shared/python/theme/stylesheets.py (615 lines); src/shared/python/theme/theme_manager.py (566 lines); src/shared/python/ui/qt/widgets/signal_toolkit_processing_mixin.py (800 lines); src/shared/python/ui/qt/widgets/signal_toolkit_ui_mixin.py (830 lines); src/shared/python/ui/simulation_gui_base.py (534 lines); src/shared/python/upstream_drift_tools/calculators/conversion/flow_rate_converter.py (701 lines); src/shared/python/upstream_drift_tools/calculators/conversion/service.py (894 lines); src/shared/python/upstream_drift_tools/calculators/conversion/tables.py (586 lines); src/shared/python/upstream_drift_tools/calculators/electrical/electrical_model.py (515 lines); src/shared/python/upstream_drift_tools/calculators/thermo/steam_engine.py (910 lines); src/shared/python/upstream_drift_tools/data_processing/core.py (667 lines); src/shared/python/upstream_drift_tools/lab/bio/c3d_reader.py (866 lines); src/shared/python/upstream_drift_tools/process_calculators/acid_gas_dewpoint_calculator.py (931 lines); src/shared/python/upstream_drift_tools/process_calculators/constants.py (616 lines); src/shared/python/upstream_drift_tools/process_calculators/optimization.py (542 lines); src/shared/python/upstream_drift_tools/process_calculators/pressure_drop_calculator/utils/gas_properties.py (946 lines); src/shared/python/upstream_drift_tools/process_calculators/psa_package/References/psa_stage_removal_sensitivity.ipynb (936 lines); src/shared/python/upstream_drift_tools/process_calculators/psa_package/psa_analysis.ipynb (1181 lines); src/shared/python/upstream_drift_tools/process_calculators/psa_package/psa_analysis_colab.ipynb (711 lines); src/shared/python/upstream_drift_tools/process_calculators/psa_package/psa_gui.py (1056 lines); src/shared/python/upstream_drift_tools/process_calculators/psa_package/psa_webapp.py (695 lines); src/shared/python/upstream_drift_tools/process_calculators/psa_package/test_psa_model.py (511 lines); src/shared/python/upstream_drift_tools/process_calculators/scrubber_calculator.py (803 lines); src/shared/python/upstream_drift_tools/process_calculators/syngas_compression_calculator.py (1162 lines); src/shared/python/upstream_drift_tools/process_calculators/syngas_water_calculator.py (727 lines); src/shared/python/upstream_drift_tools/process_calculators/wgs_reactor_calculator.py (741 lines); src/shared/python/upstream_drift_tools/ui/mixins/calculator_state_mixin.py (805 lines); src/shared/python/upstream_drift_tools/ui/widgets/data_processor_widget.py (605 lines); src/shared/python/upstream_drift_tools/ui/widgets/unit_converter_widget.py (565 lines); src/shared/python/upstream_drift_tools/utils/state_manager.py (570 lines); src/shared/python/validation_pkg/comparative_plotting.py (579 lines); src/shared/python/validation_pkg/data_fitting.py (1065 lines); src/tools/model_explorer/chain_manipulation.py (931 lines); src/tools/model_explorer/component_library.py (776 lines); src/tools/model_explorer/end_effector_manager.py (940 lines); src/tools/model_explorer/frankenstein_editor/editor.py (522 lines); src/tools/model_explorer/joint_manipulator.py (812 lines); src/tools/model_explorer/main_window.py (798 lines); src/tools/model_explorer/mesh_browser.py (802 lines); src/tools/model_explorer/model_library.py (1005 lines); src/tools/model_explorer/model_loader_dialog.py (1038 lines); src/tools/model_explorer/mujoco_viewer.py (1052 lines); src/tools/model_explorer/segment_panel.py (636 lines); src/tools/model_explorer/urdf_builder.py (689 lines); src/tools/model_explorer/urdf_code_editor.py (814 lines); src/tools/model_explorer/urdf_editor_window.py (577 lines); src/tools/model_explorer/visualization_widget.py (677 lines); src/tools/video_analyzer/analyzer.py (740 lines); src/unreal_integration/mesh_loader.py (851 lines); src/unreal_integration/skeleton_mapper.py (811 lines); src/unreal_integration/streaming.py (788 lines); src/unreal_integration/visualization.py (559 lines); src/unreal_integration/vr_interaction.py (578 lines); tests/api/test_phase3_api.py (674 lines); tests/api/test_phase4_api.py (822 lines); tests/config/test_launcher_manifest.py (640 lines); tests/cross_engine/test_mujoco_vs_pinocchio.py (521 lines); tests/heavy_integration/test_cross_engine_integration.py (518 lines); tests/heavy_integration/test_phase1_drake_integration.py (524 lines); tests/integration/putting_green/test_putting_simulation.py (563 lines); tests/integration/test_conservation_laws.py (538 lines); tests/integration/test_phase1_drake_integration.py (521 lines); tests/launchers/test_launcher_diagnostics.py (528 lines); tests/launchers/test_launcher_process_manager.py (626 lines); tests/shared/python/signal_toolkit/test_series.py (760 lines); tests/test_docker_integration.py (524 lines); tests/test_drag_drop_functionality.py (739 lines); tests/test_launcher_fixes.py (501 lines); tests/unit/analysis/test_analysis_comprehensive.py (548 lines); tests/unit/api/test_security.py (609 lines); tests/unit/api/test_tracing.py (513 lines); tests/unit/engines/putting_green/test_ball_roll_physics.py (522 lines); tests/unit/engines/putting_green/test_simulator.py (671 lines); tests/unit/optimization/test_optimization_comprehensive.py (599 lines); tests/unit/optimization/test_swing_bridge.py (522 lines); tests/unit/robotics/test_contact.py (529 lines); tests/unit/robotics/test_control.py (775 lines); tests/unit/robotics/test_locomotion.py (720 lines); tests/unit/robotics/test_planning_collision.py (781 lines); tests/unit/robotics/test_planning_motion.py (557 lines); tests/unit/shared_python/test_ai_workflow_engine.py (518 lines); tests/unit/signal_toolkit/test_signal_toolkit.py (800 lines); tests/unit/spatial_algebra/test_pose6dof_placement.py (646 lines); tests/unit/spatial_algebra/test_spatial_algebra_comprehensive.py (514 lines); tests/unit/test_activation_dynamics.py (577 lines); tests/unit/test_aerodynamics.py (974 lines); tests/unit/test_api_security.py (519 lines); tests/unit/test_ball_flight_physics.py (798 lines); tests/unit/test_cross_engine_perturbation_runner.py (539 lines); tests/unit/test_data_processing_core_extended.py (670 lines); tests/unit/test_energy_monitor.py (756 lines); tests/unit/test_enhanced_ball_flight.py (539 lines); tests/unit/test_grip_contact_model.py (605 lines); tests/unit/test_impact_model.py (804 lines); tests/unit/test_issue_fixes_1777_1778_1779_1782.py (532 lines); tests/unit/test_launcher_diagnostics.py (505 lines); tests/unit/test_model_registry.py (754 lines); tests/unit/test_muscle_analysis.py (580 lines); tests/unit/test_muscle_equilibrium.py (673 lines); tests/unit/test_myoconverter_integration.py (509 lines); tests/unit/test_pendulum_simulator_physics.py (762 lines); tests/unit/test_property_based_physics.py (743 lines); tests/unit/test_terrain.py (685 lines); tests/unit/test_third_party_integration_audit.py (872 lines); tests/unit/test_unreal_integration/test_data_models.py (669 lines); tests/unit/test_unreal_integration/test_mesh_loader.py (587 lines); tests/unit/test_unreal_integration/test_visualization_vr_backends.py (609 lines); tests/unit/tools/humanoid_character_builder/test_mesh_generators.py (640 lines); tests/unit/tools/model_generation/test_core_types.py (514 lines); tests/unit/tools/model_generation/test_dbc_decorators.py (557 lines); tests/unit/tools/model_generation/test_external_integration.py (515 lines); tests/unit/tools/model_generation/test_integration_roundtrip.py (685 lines); tests/unit/tools/model_generation/test_property_based.py (644 lines); tests/unit/tools/model_generation/test_security_fixes.py (519 lines); tests/unit/validation_pkg/test_validation_comprehensive.py (946 lines); ui/src/api/useSimulation.test.ts (803 lines); ui/src/integration/SimulationWorkflow.test.tsx (580 lines); ui/src/pages/DataExplorer.tsx (592 lines); ui/src/pages/MotionCapture.tsx (620 lines); ui/src/pages/PuttingGreen.tsx (715 lines)
- Evidence: Category F lists files over 500 lines or coarse long-definition signals.
- Impact: Large modules obscure ownership, complicate review, and weaken SRP.
- Proposed fix: Add characterization tests, then split cohesive responsibilities into smaller modules.
- Acceptance criteria: Largest files are reduced or justified; extracted modules have focused tests.
- Expectations: SRP, function size, module size, maintainability

### Review duplicated responsibility clusters
- Severity: medium
- Problem: Repeated filename clusters found: __main__ appears in 5 files; analyzer appears in 7 files; base appears in 6 files; code_quality_check appears in 7 files; codeissuesgui appears in 4 files
- Evidence: Category K duplicate-name clustering found repeated responsibility names.
- Impact: Potential duplicated logic increases maintenance cost and drift risk.
- Proposed fix: Review clusters, remove accidental duplication, and extract shared helpers where behavior is truly common.
- Acceptance criteria: Documented review of clusters; duplicated implementations are consolidated or justified.
- Expectations: DRY, maintainability, SRP

### Investigate potential hard-coded secret patterns
- Severity: high
- Problem: Potential secret-like assignments found in: src/shared/python/ai/adapters/anthropic_adapter.py; src/shared/python/ai/adapters/openai_adapter.py; src/shared/python/ai/config.py; tests/api/test_auth_security.py; tests/unit/api/test_auth_models.py; tests/unit/api/test_security.py; tests/unit/shared_python/ai/adapters/test_gemini_adapter.py; tests/unit/test_api_security.py; tests/unit/test_issue_fixes_1777_1778_1779_1782.py; tests/unit/test_security_and_module_fixes.py
- Evidence: Category I regex scan matched secret-like assignments.
- Impact: Hard-coded secrets can expose credentials and create security incidents.
- Proposed fix: Manually verify findings, rotate any exposed credentials, and move secrets to environment or secret management.
- Acceptance criteria: Secret scan is clean or findings are documented false positives; exposed credentials are rotated.
- Expectations: security, reliability

### Review deep object traversal hotspots
- Severity: medium
- Problem: Deep member-chain hints found in: build_hooks.py; docs/sphinx/_static/_sphinx_javascript_frameworks_compat.js; docs/sphinx/_static/jquery.js; docs/sphinx/_static/js/badge_only.js; docs/sphinx/_static/js/theme.js; docs/sphinx/_static/js/versions.js; docs/sphinx/_static/searchtools.js; docs/sphinx/_static/sphinx_highlight.js
- Evidence: Category L found repeated chains with three or more member hops.
- Impact: Law of Demeter pressure can make APIs brittle and increase coupling.
- Proposed fix: Review hotspots and introduce boundary methods or DTOs where callers traverse object graphs.
- Acceptance criteria: Hotspots are documented, simplified, or justified; tests cover any API boundary changes.
- Expectations: Law of Demeter, SRP, maintainability

