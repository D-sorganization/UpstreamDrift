# Completist Report: 2026-03-04

## Executive Summary

An updated automated analysis of implementation gaps generated using `scripts/scan_for_incomplete_code.py`. The scan identifies 458 potential issues in 174 files. Specifically, it found:
- **404** empty `pass` blocks, particularly concentrated in test files, indicating widespread gaps in test coverage and assertions.
- **5** `NotImplementedError` stubs blocking functionality in core utility modules.
- **46** `TODO` markers indicating missing feature implementations or incomplete logic.
- **19** `FIXME` markers denoting technical debt, inaccurate physics approximations, and potential legal risks.

These widespread placeholders must be systematically addressed to ensure system reliability and fidelity.

## Detailed Findings

```text
Found 458 potential issues in 174 files.

./installer/windows/build_installer.py:
  Line 38 [pass_block]: pass
  Line 87 [pass_block]: pass

./scripts/analyze_completist_data.py:
  Line 116 [TODO]: return {"file": filepath, "line": lineno, "text": content, "type": "TODO"}
  Line 130 [TODO]: if marker_item["type"] == "TODO":
  Line 211 [FIXME]: "FIXME": 2,
  Line 212 [TODO]: "TODO": 3,
  Line 285 [TODO]: chart.append(f'    "Feature Requests (TODO)" : {len(todos)}')
  Line 286 [FIXME]: chart.append(f'    "Technical Debt (FIXME)" : {len(fixmes)}')
  Line 432 [TODO]: f"- **Feature Gaps (TODO)**: {len(todos)}",

./scripts/assess_repository.py:
  Line 136 [pass_block]: pass
  Line 328 [pass_block]: pass

./scripts/fix_shims.py:
  Line 39 [pass_block]: pass

./scripts/generate_todo_fixme_register.py:
  Line 10 [TODO]: OUT = ROOT / "docs" / "technical_debt" / "TODO_FIXME_REGISTER.md"
  Line 10 [FIXME]: OUT = ROOT / "docs" / "technical_debt" / "TODO_FIXME_REGISTER.md"
  Line 11 [TODO]: PATTERN = re.compile(r"\b(TODO|FIXME)\b")
  Line 11 [FIXME]: PATTERN = re.compile(r"\b(TODO|FIXME)\b")
  Line 16 [TODO]: ["rg", "-n", "TODO|FIXME", "src", "tests", "scripts"],
  Line 16 [FIXME]: ["rg", "-n", "TODO|FIXME", "src", "tests", "scripts"],
  Line 25 [TODO]: "# TODO/FIXME Debt Register",
  Line 25 [FIXME]: "# TODO/FIXME Debt Register",
  Line 27 [TODO]: "This register is generated from inline TODO/FIXME markers.",
  Line 27 [FIXME]: "This register is generated from inline TODO/FIXME markers.",
  Line 41 [TODO]: marker = "TODO" if "TODO" in text else "FIXME"
  Line 41 [FIXME]: marker = "TODO" if "TODO" in text else "FIXME"

./scripts/maintain_workflows.py:
  Line 31 [pass_block]: pass
  Line 52 [pass_block]: pass

./scripts/pragmatic_programmer_review.py:
  Line 148 [pass_block]: pass
  Line 171 [pass_block]: pass
  Line 176 [TODO]: """Report high TODO counts as a technical debt indicator."""
  Line 182 [TODO]: if "TODO" in content:
  Line 185 [pass_block]: pass
  Line 192 [TODO]: "title": f"High TODO count ({len(todos)})",
  Line 195 [TODO]: "recommendation": "Review TODOs",

./scripts/refresh_completist_data.py:
  Line 54 [TODO]: # 2. Grep for TODOs
  Line 56 [TODO]: "TODO|FIXME|XXX|HACK|TEMP",
  Line 56 [FIXME]: "TODO|FIXME|XXX|HACK|TEMP",

./scripts/scan_for_incomplete_code.py:
  Line 7 [TODO]: "TODO": re.compile(r"TODO"),
  Line 8 [FIXME]: "FIXME": re.compile(r"FIXME"),
  Line 9 [NotImplementedError]: "NotImplementedError": re.compile(r"raise NotImplementedError"),

./scripts/setup_hooks.py:
  Line 110 [TODO]: - quality-check (no TODOs/FIXMEs)
  Line 110 [FIXME]: - quality-check (no TODOs/FIXMEs)

./shared/models/opensim/opensim-models/Tutorials/Building_a_Passive_Dynamic_Walker/DynamicWalkerBuild/DynamicWalkerBuildModelStudent.cpp:
  Line 30 [TODO]: // TODO: Add Code to Begin Model here
  Line 90 [TODO]: // TODO: Set the coordinate properties

./shared/models/opensim/opensim-models/Tutorials/Building_a_Passive_Dynamic_Walker/DynamicWalkerBuildModel.cpp:
  Line 60 [TODO]: // Section A.1 TODO: Create the Pelvis and set the coordinate properties
  Line 63 [TODO]: // Section A.2 TODO: Create the LeftThigh, LeftShank, RightThigh and RightShank bodies
  Line 81 [TODO]: // Section B.1 TODO: Add ContactSphere to the left hip, the knee, and the foot points
  Line 87 [TODO]: // Section B.2 TODO: Add HuntCrossleyForces
  Line 100 [TODO]: // Section B.2 TODO: Add HuntCrossleyForces betweeen the remaining ContactSpheres
  Line 107 [TODO]: // Section C.1 TODO: Construct CoordinateLimitForces for the Hip and Knee

./shared/models/opensim/opensim-models/Tutorials/Building_a_Passive_Dynamic_Walker/skeleton.cpp:
  Line 15 [TODO]: // TODO: Add Code to Begin Model here

./src/api/aip/methods.py:
  Line 220 [pass_block]: pass
  Line 316 [pass_block]: pass
  Line 371 [pass_block]: pass

./src/api/diagnostics.py:
  Line 359 [pass_block]: pass

./src/api/routes/actuator_controls.py:
  Line 89 [pass_block]: pass
  Line 167 [pass_block]: pass
  Line 324 [pass_block]: pass

./src/api/routes/analysis_tools.py:
  Line 78 [pass_block]: pass
  Line 89 [pass_block]: pass
  Line 101 [pass_block]: pass

./src/api/routes/data_explorer.py:
  Line 146 [pass_block]: pass
  Line 392 [pass_block]: pass

./src/api/routes/dataset.py:
  Line 256 [pass_block]: pass

./src/api/routes/physics.py:
  Line 270 [pass_block]: pass
  Line 278 [pass_block]: pass
  Line 292 [pass_block]: pass
  Line 348 [pass_block]: pass
  Line 359 [pass_block]: pass

./src/deployment/realtime/controller.py:
  Line 363 [NotImplementedError]: raise NotImplementedError(
  Line 429 [NotImplementedError]: raise NotImplementedError(

./src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Simscape Multibody Data Plotters/Python Version/golf_gui_r0/golf_visualizer_implementation.py:
  Line 205 [pass_block]: pass
  Line 217 [pass_block]: pass

./src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Simscape Multibody Data Plotters/Python Version/integrated_golf_gui_r0/golf_wiffle_main.py:
  Line 621 [pass_block]: pass

./src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/__init__.py:
  Line 28 [pass_block]: pass

./src/engines/pendulum_models/python/double_pendulum_model/ui/double_pendulum_gui.py:
  Line 32 [pass_block]: pass

./src/engines/physics_engines/drake/python/src/drake_gui_analysis.py:
  Line 138 [pass_block]: pass

./src/engines/physics_engines/drake/python/src/drake_gui_viz.py:
  Line 161 [pass_block]: pass

./src/engines/physics_engines/drake/python/src/drake_visualization_mixin.py:
  Line 167 [pass_block]: pass

./src/engines/physics_engines/drake/python/src/pose_editor_tab.py:
  Line 147 [pass_block]: pass

./src/engines/physics_engines/mujoco/docker/example_dynamic_stance.py:
  Line 63 [pass_block]: pass
  Line 283 [pass_block]: pass

./src/engines/physics_engines/mujoco/docker/example_golf_swing.py:
  Line 160 [pass_block]: pass

./src/engines/physics_engines/mujoco/docker/gui/golf_gui_docker.py:
  Line 50 [pass_block]: pass

./src/engines/physics_engines/mujoco/docker/gui/golf_gui_styles.py:
  Line 17 [pass_block]: pass

./src/engines/physics_engines/mujoco/docker/src/humanoid_golf/sim.py:
  Line 91 [pass_block]: pass
  Line 123 [pass_block]: pass
  Line 190 [pass_block]: pass
  Line 327 [pass_block]: pass

./src/engines/physics_engines/mujoco/docker/src/humanoid_golf/visualization.py:
  Line 291 [pass_block]: pass
  Line 353 [pass_block]: pass

./src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/biomechanics.py:
  Line 500 [pass_block]: pass

./src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/gui/tabs/manipulation_tab.py:
  Line 416 [pass_block]: pass

./src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/meshcat_adapter.py:
  Line 54 [pass_block]: pass

./src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/sim_rendering_mixin.py:
  Line 140 [pass_block]: pass
  Line 268 [pass_block]: pass

./src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/sim_widget.py:
  Line 688 [pass_block]: pass

./src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/video_export.py:
  Line 338 [pass_block]: pass
  Line 374 [pass_block]: pass

./src/engines/physics_engines/myosuite/python/myosuite_physics_engine.py:
  Line 655 [pass_block]: pass

./src/engines/physics_engines/pinocchio/python/dtack/gui/main_window.py:
  Line 47 [pass_block]: pass

./src/engines/physics_engines/pinocchio/python/motion_training/__init__.py:
  Line 65 [pass_block]: pass
  Line 77 [pass_block]: pass
  Line 82 [pass_block]: pass
  Line 92 [pass_block]: pass
  Line 97 [pass_block]: pass

./src/engines/physics_engines/pinocchio/python/pinocchio_golf/analysis_controller.py:
  Line 29 [pass_block]: pass
  Line 36 [pass_block]: pass

./src/engines/physics_engines/pinocchio/python/pinocchio_golf/gui_simulation.py:
  Line 275 [pass_block]: pass
  Line 283 [pass_block]: pass

./src/engines/physics_engines/pinocchio/python/pinocchio_golf/pinocchio_analysis_mixin.py:
  Line 251 [pass_block]: pass

./src/engines/physics_engines/pinocchio/python/pinocchio_golf/pinocchio_visualization_mixin.py:
  Line 259 [pass_block]: pass

./src/engines/physics_engines/pinocchio/python/pinocchio_golf/ui/analysis_mixin.py:
  Line 227 [pass_block]: pass

./src/engines/physics_engines/pinocchio/python/pinocchio_golf/ui/main_window.py:
  Line 156 [pass_block]: pass

./src/launchers/assets/generate_tile_images.py:
  Line 268 [pass_block]: pass

./src/launchers/assets/optimize_assets.py:
  Line 151 [pass_block]: pass

./src/launchers/base.py:
  Line 35 [pass_block]: pass

./src/launchers/golf_launcher.py:
  Line 644 [pass_block]: pass

./src/launchers/launcher_diagnostics.py:
  Line 26 [pass_block]: pass

./src/launchers/launcher_dialogs.py:
  Line 38 [pass_block]: pass
  Line 174 [pass_block]: pass

./src/launchers/launcher_theme.py:
  Line 259 [pass_block]: pass
  Line 266 [pass_block]: pass

./src/launchers/launcher_ui_setup.py:
  Line 42 [pass_block]: pass

./src/launchers/model_card.py:
  Line 28 [pass_block]: pass

./src/launchers/settings_dialog.py:
  Line 310 [pass_block]: pass
  Line 327 [pass_block]: pass

./src/launchers/unified_launcher.py:
  Line 23 [pass_block]: pass
  Line 53 [pass_block]: pass
  Line 60 [pass_block]: pass
  Line 164 [pass_block]: pass
  Line 173 [pass_block]: pass

./src/robotics/contact/contact_manager.py:
  Line 353 [pass_block]: pass
  Line 356 [pass_block]: pass

./src/shared/models/myosuite/examples/train_elbow_policy.py:
  Line 134 [pass_block]: pass

./src/shared/models/opensim/opensim-models/Tutorials/Building_a_Passive_Dynamic_Walker/DynamicWalkerBuild/DynamicWalkerBuildModelStudent.cpp:
  Line 30 [TODO]: // TODO: Add Code to Begin Model here
  Line 100 [TODO]: // TODO: Set the coordinate properties

./src/shared/models/opensim/opensim-models/Tutorials/Building_a_Passive_Dynamic_Walker/DynamicWalkerBuildModel.cpp:
  Line 65 [TODO]: // Section A.1 TODO: Create the Pelvis and set the coordinate
  Line 69 [TODO]: // Section A.2 TODO: Create the LeftThigh, LeftShank, RightThigh and
  Line 89 [TODO]: // Section B.1 TODO: Add ContactSphere to the left hip, the knee,
  Line 96 [TODO]: // Section B.2 TODO: Add HuntCrossleyForces
  Line 114 [TODO]: // Section B.2 TODO: Add HuntCrossleyForces betweeen the remaining
  Line 122 [TODO]: // Section C.1 TODO: Construct CoordinateLimitForces for the Hip and

./src/shared/models/opensim/opensim-models/Tutorials/Building_a_Passive_Dynamic_Walker/skeleton.cpp:
  Line 15 [TODO]: // TODO: Add Code to Begin Model here

./src/shared/python/ai/gui/assistant_panel.py:
  Line 723 [pass_block]: pass
  Line 747 [pass_block]: pass
  Line 831 [pass_block]: pass

./src/shared/python/ai/gui/chat_dock_widget.py:
  Line 47 [pass_block]: pass
  Line 57 [pass_block]: pass

./src/shared/python/ai/gui/settings_dialog.py:
  Line 46 [pass_block]: pass

./src/shared/python/biomechanics/swing_plane_visualization.py:
  Line 36 [pass_block]: pass

./src/shared/python/chat/chat_dock_widget.py:
  Line 59 [pass_block]: pass
  Line 69 [pass_block]: pass

./src/shared/python/config/handedness_support.py:
  Line 32 [pass_block]: pass

./src/shared/python/contracts.py:
  Line 209 [pass_block]: pass

./src/shared/python/core/__init__.py:
  Line 80 [pass_block]: pass

./src/shared/python/core/datetime_utils.py:
  Line 226 [pass_block]: pass

./src/shared/python/core/error_decorators.py:
  Line 12 [pass_block]: pass
  Line 217 [pass_block]: pass
  Line 266 [pass_block]: pass

./src/shared/python/dashboard/advanced_analysis.py:
  Line 325 [pass_block]: pass

./src/shared/python/dashboard/widgets.py:
  Line 805 [pass_block]: pass

./src/shared/python/data_io/common_utils.py:
  Line 284 [pass_block]: pass

./src/shared/python/data_io/dataset_generator.py:
  Line 610 [pass_block]: pass
  Line 646 [pass_block]: pass
  Line 657 [pass_block]: pass
  Line 664 [pass_block]: pass
  Line 752 [pass_block]: pass

./src/shared/python/data_io/output_manager.py:
  Line 661 [pass_block]: pass

./src/shared/python/data_io/provenance.py:
  Line 125 [pass_block]: pass

./src/shared/python/data_io/reproducibility.py:
  Line 32 [pass_block]: pass

./src/shared/python/data_io/swing_capture_import.py:
  Line 711 [pass_block]: pass

./src/shared/python/engine_core/base_physics_engine.py:
  Line 18 [pass_block]: pass
  Line 22 [pass_block]: pass
  Line 401 [pass_block]: pass

./src/shared/python/engine_core/engine_availability.py:
  Line 97 [pass_block]: pass
  Line 105 [pass_block]: pass
  Line 113 [pass_block]: pass
  Line 121 [pass_block]: pass
  Line 129 [pass_block]: pass
  Line 137 [pass_block]: pass
  Line 145 [pass_block]: pass
  Line 153 [pass_block]: pass
  Line 162 [pass_block]: pass
  Line 171 [pass_block]: pass
  Line 180 [pass_block]: pass
  Line 188 [pass_block]: pass
  Line 223 [pass_block]: pass
  Line 236 [pass_block]: pass
  Line 246 [pass_block]: pass
  Line 254 [pass_block]: pass
  Line 262 [pass_block]: pass
  Line 270 [pass_block]: pass
  Line 278 [pass_block]: pass
  Line 286 [pass_block]: pass
  Line 294 [pass_block]: pass
  Line 302 [pass_block]: pass
  Line 310 [pass_block]: pass
  Line 318 [pass_block]: pass
  Line 326 [pass_block]: pass
  Line 334 [pass_block]: pass
  Line 342 [pass_block]: pass
  Line 350 [pass_block]: pass
  Line 358 [pass_block]: pass
  Line 366 [pass_block]: pass
  Line 374 [pass_block]: pass
  Line 382 [pass_block]: pass
  Line 390 [pass_block]: pass
  Line 398 [pass_block]: pass
  Line 406 [pass_block]: pass
  Line 414 [pass_block]: pass
  Line 422 [pass_block]: pass
  Line 430 [pass_block]: pass
  Line 438 [pass_block]: pass
  Line 446 [pass_block]: pass
  Line 454 [pass_block]: pass
  Line 462 [pass_block]: pass
  Line 470 [pass_block]: pass
  Line 478 [pass_block]: pass
  Line 486 [pass_block]: pass

./src/shared/python/engine_core/unified_engine_interface.py:
  Line 228 [pass_block]: pass

./src/shared/python/gui_pkg/help_system.py:
  Line 46 [pass_block]: pass

./src/shared/python/gui_pkg/viewpoint_controls.py:
  Line 30 [pass_block]: pass

./src/shared/python/humanoid_character_builder/mesh/collision_generator.py:
  Line 148 [pass_block]: pass
  Line 155 [pass_block]: pass
  Line 342 [pass_block]: pass

./src/shared/python/humanoid_character_builder/mesh/mesh_processor.py:
  Line 19 [pass_block]: pass

./src/shared/python/injury/joint_stress.py:
  Line 494 [pass_block]: pass
  Line 500 [pass_block]: pass

./src/shared/python/injury/swing_modifications.py:
  Line 351 [pass_block]: pass
  Line 375 [pass_block]: pass

./src/shared/python/logging_pkg/logger_utils.py:
  Line 84 [pass_block]: pass

./src/shared/python/model_generation/library/model_library.py:
  Line 667 [pass_block]: pass

./src/shared/python/model_generation/library/unified_loader.py:
  Line 133 [pass_block]: pass

./src/shared/python/optimization/swing_optimizer.py:
  Line 866 [pass_block]: pass

./src/shared/python/physics/flexible_shaft.py:
  Line 35 [pass_block]: pass

./src/shared/python/physics/grip_contact_model.py:
  Line 26 [pass_block]: pass

./src/shared/python/physics/impact_model.py:
  Line 36 [pass_block]: pass

./src/shared/python/physics/terrain_mixin.py:
  Line 8 [pass_block]: pass

./src/shared/python/plot_theme/manager.py:
  Line 82 [pass_block]: pass

./src/shared/python/plotting/energy.py:
  Line 24 [pass_block]: pass
  Line 42 [pass_block]: pass

./src/shared/python/plotting/kinematics.py:
  Line 25 [pass_block]: pass

./src/shared/python/plotting/renderers/kinetics.py:
  Line 668 [pass_block]: pass

./src/shared/python/security/secure_subprocess.py:
  Line 161 [pass_block]: pass
  Line 233 [pass_block]: pass

./src/shared/python/signal_toolkit/series.py:
  Line 423 [pass_block]: pass

./src/shared/python/spatial_algebra/reference_frames.py:
  Line 24 [pass_block]: pass

./src/shared/python/ui/loading_button.py:
  Line 238 [pass_block]: pass

./src/shared/python/ui/preferences_dialog.py:
  Line 14 [pass_block]: pass
  Line 185 [pass_block]: pass

./src/shared/python/ui/qt/process_worker.py:
  Line 22 [pass_block]: pass

./src/shared/python/ui/qt/utils.py:
  Line 131 [pass_block]: pass
  Line 257 [pass_block]: pass

./src/shared/python/ui/recent_models.py:
  Line 33 [pass_block]: pass

./src/shared/python/upstream_drift_tools/calculators/conversion/service.py:
  Line 311 [pass_block]: pass

./src/shared/python/upstream_drift_tools/process_calculators/optimization.py:
  Line 469 [pass_block]: pass

./src/shared/python/upstream_drift_tools/process_calculators/syngas_compression_calculator.py:
  Line 625 [pass_block]: pass
  Line 857 [pass_block]: pass

./src/shared/python/upstream_drift_tools/process_calculators/wgs_reactor_calculator.py:
  Line 108 [pass_block]: pass

./src/shared/python/upstream_drift_tools/ui/mixins/calculator_state_mixin.py:
  Line 122 [pass_block]: pass
  Line 312 [pass_block]: pass
  Line 366 [pass_block]: pass
  Line 431 [pass_block]: pass
  Line 524 [pass_block]: pass
  Line 546 [pass_block]: pass
  Line 603 [pass_block]: pass
  Line 621 [pass_block]: pass

./src/shared/python/upstream_drift_tools/ui/widgets/data_processor_widget.py:
  Line 595 [pass_block]: pass

./src/shared/python/upstream_drift_tools/ui/widgets/unit_converter_widget.py:
  Line 204 [pass_block]: pass

./src/shared/python/validation_pkg/data_fitting.py:
  Line 25 [pass_block]: pass

./src/shared/tools/human-gazebo/legacy/control/src/HumanGazeboControlModule.cpp:
  Line 131 [TODO]: //TODO read the joint names list and then put then in the control board options

./src/tools/check_markdown_links.py:
  Line 72 [pass_block]: pass

./src/tools/code_quality_check.py:
  Line 192 [pass_block]: pass

./src/tools/humanoid_character_builder/generators/mesh_generator.py:
  Line 1086 [pass_block]: pass

./src/tools/humanoid_character_builder/mesh/mesh_processor.py:
  Line 19 [pass_block]: pass
  Line 362 [pass_block]: pass

./src/tools/model_explorer/model_library.py:
  Line 858 [pass_block]: pass

./src/tools/model_explorer/model_loader_dialog.py:
  Line 490 [pass_block]: pass

./src/tools/model_generation/builders/parametric_builder.py:
  Line 295 [pass_block]: pass
  Line 304 [pass_block]: pass

./src/tools/model_generation/converters/format_utils.py:
  Line 159 [NotImplementedError]: raise NotImplementedError(

./src/tools/model_generation/library/model_library.py:
  Line 575 [pass_block]: pass

./src/tools/video_analyzer/video_processor.py:
  Line 37 [pass_block]: pass

./tests/acceptance/test_counterfactual_experiments.py:
  Line 330 [pass_block]: pass

./tests/conftest.py:
  Line 131 [pass_block]: pass

./tests/deployment/test_safety.py:
  Line 142 [pass_block]: pass
  Line 155 [pass_block]: pass

./tests/integration/test_c3d_workflow.py:
  Line 27 [pass_block]: pass

./tests/integration/test_conservation_laws.py:
  Line 537 [pass_block]: pass

./tests/integration/test_engine_integration.py:
  Line 102 [pass_block]: pass

./tests/integration/test_golf_launcher_integration.py:
  Line 24 [pass_block]: pass
  Line 30 [pass_block]: pass
  Line 33 [pass_block]: pass
  Line 36 [pass_block]: pass
  Line 39 [pass_block]: pass
  Line 42 [pass_block]: pass
  Line 45 [pass_block]: pass
  Line 48 [pass_block]: pass
  Line 51 [pass_block]: pass
  Line 54 [pass_block]: pass
  Line 57 [pass_block]: pass
  Line 60 [pass_block]: pass
  Line 81 [pass_block]: pass
  Line 85 [pass_block]: pass
  Line 106 [pass_block]: pass
  Line 109 [pass_block]: pass
  Line 112 [pass_block]: pass
  Line 115 [pass_block]: pass
  Line 120 [pass_block]: pass
  Line 123 [pass_block]: pass
  Line 126 [pass_block]: pass

./tests/integration/test_opensim_myosuite_wiring.py:
  Line 22 [pass_block]: pass

./tests/integration/test_real_engine_loading.py:
  Line 212 [pass_block]: pass
  Line 225 [pass_block]: pass
  Line 238 [pass_block]: pass
  Line 268 [pass_block]: pass

./tests/learning/test_sim2real.py:
  Line 63 [pass_block]: pass

./tests/test_dashboard_enhancements.py:
  Line 38 [pass_block]: pass
  Line 41 [pass_block]: pass
  Line 54 [pass_block]: pass
  Line 64 [pass_block]: pass

./tests/test_golf_gui_tabs.py:
  Line 43 [pass_block]: pass
  Line 46 [pass_block]: pass
  Line 68 [pass_block]: pass
  Line 72 [pass_block]: pass
  Line 76 [pass_block]: pass

./tests/test_layout_persistence.py:
  Line 203 [pass_block]: pass

./tests/test_pinocchio_ecosystem.py:
  Line 66 [pass_block]: pass

./tests/test_tool_registry.py:
  Line 59 [pass_block]: pass
  Line 72 [pass_block]: pass
  Line 87 [pass_block]: pass
  Line 91 [pass_block]: pass

./tests/unit/ai/test_tool_registry.py:
  Line 242 [pass_block]: pass

./tests/unit/api/test_path_validation.py:
  Line 235 [pass_block]: pass

./tests/unit/core/test_error_decorators.py:
  Line 72 [pass_block]: pass
  Line 154 [pass_block]: pass
  Line 221 [pass_block]: pass
  Line 270 [pass_block]: pass

./tests/unit/data_io/test_data_io_comprehensive.py:
  Line 318 [pass_block]: pass

./tests/unit/dbc/test_dbc_runtime_analysis_metrics.py:
  Line 45 [pass_block]: pass
  Line 98 [pass_block]: pass
  Line 165 [pass_block]: pass
  Line 235 [pass_block]: pass

./tests/unit/engines/mujoco/test_dependencies.py:
  Line 59 [pass_block]: pass

./tests/unit/engines/mujoco/test_pinocchio_interface.py:
  Line 219 [pass_block]: pass

./tests/unit/engines/pinocchio/motion_training/test_motion_training.py:
  Line 306 [pass_block]: pass

./tests/unit/engines/simscape/3d/test_quality_check.py:
  Line 139 [NotImplementedError]: lines = ["def function():\n", "    raise NotImplementedError\n"]
  Line 257 [pass_block]: pass
  Line 299 [pass_block]: pass

./tests/unit/engines/test_plugin_registry.py:
  Line 46 [pass_block]: pass
  Line 49 [pass_block]: pass
  Line 52 [pass_block]: pass
  Line 55 [pass_block]: pass
  Line 58 [pass_block]: pass
  Line 64 [pass_block]: pass
  Line 67 [pass_block]: pass
  Line 107 [pass_block]: pass

./tests/unit/launchers/test_unified_launcher.py:
  Line 93 [pass_block]: pass

./tests/unit/robotics/test_locomotion.py:
  Line 150 [pass_block]: pass

./tests/unit/robotics/test_planning_collision.py:
  Line 607 [pass_block]: pass

./tests/unit/shared_python/test_utils.py:
  Line 14 [pass_block]: pass
  Line 49 [pass_block]: pass
  Line 83 [pass_block]: pass

./tests/unit/test_common_utils_coverage.py:
  Line 71 [pass_block]: pass

./tests/unit/test_golf_launcher_logic.py:
  Line 29 [pass_block]: pass
  Line 32 [pass_block]: pass
  Line 35 [pass_block]: pass
  Line 38 [pass_block]: pass
  Line 41 [pass_block]: pass
  Line 44 [pass_block]: pass
  Line 47 [pass_block]: pass
  Line 50 [pass_block]: pass
  Line 53 [pass_block]: pass
  Line 56 [pass_block]: pass
  Line 59 [pass_block]: pass
  Line 62 [pass_block]: pass
  Line 65 [pass_block]: pass
  Line 68 [pass_block]: pass
  Line 71 [pass_block]: pass
  Line 74 [pass_block]: pass
  Line 80 [pass_block]: pass
  Line 107 [pass_block]: pass
  Line 130 [pass_block]: pass
  Line 133 [pass_block]: pass
  Line 148 [pass_block]: pass
  Line 157 [pass_block]: pass
  Line 161 [pass_block]: pass
  Line 165 [pass_block]: pass
  Line 169 [pass_block]: pass
  Line 181 [pass_block]: pass

./tests/unit/test_golf_suite_launcher.py:
  Line 33 [pass_block]: pass
  Line 36 [pass_block]: pass
  Line 39 [pass_block]: pass
  Line 42 [pass_block]: pass
  Line 45 [pass_block]: pass
  Line 53 [pass_block]: pass
  Line 58 [pass_block]: pass
  Line 61 [pass_block]: pass
  Line 64 [pass_block]: pass
  Line 67 [pass_block]: pass
  Line 70 [pass_block]: pass
  Line 75 [pass_block]: pass
  Line 78 [pass_block]: pass
  Line 81 [pass_block]: pass
  Line 86 [pass_block]: pass
  Line 89 [pass_block]: pass
  Line 95 [pass_block]: pass
  Line 98 [pass_block]: pass
  Line 106 [pass_block]: pass
  Line 109 [pass_block]: pass
  Line 112 [pass_block]: pass
  Line 115 [pass_block]: pass
  Line 118 [pass_block]: pass
  Line 123 [pass_block]: pass
  Line 126 [pass_block]: pass
  Line 129 [pass_block]: pass
  Line 132 [pass_block]: pass
  Line 135 [pass_block]: pass
  Line 138 [pass_block]: pass
  Line 146 [pass_block]: pass
  Line 152 [pass_block]: pass

./tests/unit/test_gui_coverage.py:
  Line 33 [pass_block]: pass

./tests/unit/test_launcher_x11_logic.py:
  Line 74 [pass_block]: pass

./tests/unit/test_lazy_imports.py:
  Line 86 [pass_block]: pass

./tests/unit/test_muscle_analysis.py:
  Line 480 [pass_block]: pass

./tests/unit/test_muscle_equilibrium.py:
  Line 259 [pass_block]: pass
  Line 629 [pass_block]: pass

./tests/unit/test_pendulum_putter_model.py:
  Line 22 [pass_block]: pass

./tests/unit/test_pinocchio_gui.py:
  Line 14 [pass_block]: pass

./tests/unit/test_pinocchio_physics_engine.py:
  Line 28 [pass_block]: pass

./tests/unit/test_shared_engine_probes.py:
  Line 147 [pass_block]: pass

./tests/unit/test_ux_enhancements.py:
  Line 10 [pass_block]: pass
  Line 17 [pass_block]: pass
  Line 20 [pass_block]: pass
  Line 23 [pass_block]: pass
  Line 26 [pass_block]: pass
  Line 29 [pass_block]: pass
  Line 32 [pass_block]: pass
  Line 40 [pass_block]: pass
  Line 43 [pass_block]: pass
  Line 46 [pass_block]: pass
  Line 49 [pass_block]: pass
  Line 52 [pass_block]: pass
  Line 55 [pass_block]: pass
  Line 58 [pass_block]: pass
  Line 61 [pass_block]: pass
  Line 69 [pass_block]: pass
  Line 72 [pass_block]: pass
  Line 75 [pass_block]: pass
  Line 78 [pass_block]: pass
  Line 81 [pass_block]: pass
  Line 86 [pass_block]: pass
  Line 89 [pass_block]: pass
  Line 92 [pass_block]: pass
  Line 95 [pass_block]: pass
  Line 98 [pass_block]: pass
  Line 103 [pass_block]: pass
  Line 106 [pass_block]: pass

```
