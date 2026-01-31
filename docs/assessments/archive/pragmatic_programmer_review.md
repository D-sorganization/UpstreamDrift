# Pragmatic Programmer Review

**Repository**: Golf_Modeling_Suite
**Date**: 2026-01-24
**Files Analyzed**: 813

## Overall Score: 5.7/10

## Principle Scores

| Principle | Score | Weight | Status |
|-----------|-------|--------|--------|
| Don't Repeat Yourself | 0.0 | 2.0x | Critical |
| Orthogonality & Decoupling | 0.0 | 1.5x | Critical |
| Reversibility & Flexibility | 0.0 | 1.0x | Critical |
| Code Quality & Craftsmanship | 7.0 | 1.5x | Pass |
| Error Handling & Robustness | 8.0 | 2.0x | Pass |
| Testing & Validation | 10.0 | 2.0x | Pass |
| Documentation & Communication | 10.0 | 1.0x | Pass |
| Automation & Tooling | 10.0 | 1.5x | Pass |

## Issue Summary

- **Critical**: 0
- **Major**: 691
- **Minor**: 1

## Detailed Findings

### Don't Repeat Yourself

- [!] **Significant duplicate code block**
  - Found in 3 locations across 3 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\launch_golf_suite.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ball_flight_physics.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\start_api_server.py

- [!] **Significant duplicate code block**
  - Found in 2 locations across 2 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\setup_golf_suite.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\playground_experiments\humanoid_cm_demo.py

- [!] **Significant duplicate code block**
  - Found in 5 locations across 5 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\migrate_api_keys.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\setup_golf_suite.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\playground_experiments\humanoid_cm_demo.py

- [!] **Significant duplicate code block**
  - Found in 2 locations across 2 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\baseline_assessments.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\start_api_server.py

- [!] **Significant duplicate code block**
  - Found in 3 locations across 3 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\examples\01_basic_simulation.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\playground_experiments\humanoid_cm_demo.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\playground_experiments\mocapact_demo.py

- [!] **Significant duplicate code block**
  - Found in 3 locations across 3 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\examples\02_parameter_sweeps.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\golf_wiffle_main.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\test_improved_visualization.py

- [!] **Significant duplicate code block**
  - Found in 8 locations across 8 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\examples\03_injury_risk_tutorial.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\2D_Golf_Model\tools\code_quality_check.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\tools\code_quality_check.py

- [!] **Significant duplicate code block**
  - Found in 2 locations across 2 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\examples\03_injury_risk_tutorial.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\__main__.py

- [!] **Significant duplicate code block**
  - Found in 2 locations across 1 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\analyze_completist_data.py

- [!] **Significant duplicate code block**
  - Found in 4 locations across 4 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\assess_repository.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_issues_from_assessment.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\generate_assessment_summary.py

- [!] **Significant duplicate code block**
  - Found in 3 locations across 2 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\check_system_health.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\test_docker_venv.py

- [!] **Significant duplicate code block**
  - Found in 5 locations across 5 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_cropped_robot_icon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_favicon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_high_quality_favicon.py

- [!] **Significant duplicate code block**
  - Found in 5 locations across 5 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_cropped_robot_icon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_favicon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_high_quality_favicon.py

- [!] **Significant duplicate code block**
  - Found in 4 locations across 4 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_cropped_robot_icon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_favicon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_high_quality_favicon.py

- [!] **Significant duplicate code block**
  - Found in 5 locations across 5 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_cropped_robot_icon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_favicon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_high_quality_favicon.py

- [!] **Significant duplicate code block**
  - Found in 5 locations across 5 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_cropped_robot_icon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_favicon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_high_quality_favicon.py

- [!] **Significant duplicate code block**
  - Found in 5 locations across 5 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_cropped_robot_icon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_favicon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_high_quality_favicon.py

- [!] **Significant duplicate code block**
  - Found in 3 locations across 3 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_cropped_robot_icon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_high_quality_favicon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_windows_optimized_icon.py

- [!] **Significant duplicate code block**
  - Found in 3 locations across 3 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_cropped_robot_icon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_high_quality_favicon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_windows_optimized_icon.py

- [!] **Significant duplicate code block**
  - Found in 3 locations across 3 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_cropped_robot_icon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_high_quality_favicon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_windows_optimized_icon.py

- [!] **Significant duplicate code block**
  - Found in 3 locations across 3 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_cropped_robot_icon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_high_quality_favicon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_windows_optimized_icon.py

- [!] **Significant duplicate code block**
  - Found in 4 locations across 4 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_cropped_robot_icon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_favicon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_high_quality_favicon.py

- [!] **Significant duplicate code block**
  - Found in 3 locations across 3 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_favicon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_high_quality_favicon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_windows_optimized_icon.py

- [!] **Significant duplicate code block**
  - Found in 3 locations across 3 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_favicon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_high_quality_favicon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_windows_optimized_icon.py

- [!] **Significant duplicate code block**
  - Found in 3 locations across 3 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_favicon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_high_quality_favicon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_windows_optimized_icon.py

- [!] **Significant duplicate code block**
  - Found in 3 locations across 3 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_favicon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_high_quality_favicon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_windows_optimized_icon.py

- [!] **Significant duplicate code block**
  - Found in 3 locations across 3 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_favicon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_high_quality_favicon.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_windows_optimized_icon.py

- [!] **Significant duplicate code block**
  - Found in 2 locations across 2 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_issues_from_assessment.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\secure_subprocess.py

- [!] **Significant duplicate code block**
  - Found in 3 locations across 3 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_issues_from_assessment.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\generate_assessment_summary.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\pragmatic_programmer_review.py

- [!] **Significant duplicate code block**
  - Found in 3 locations across 3 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_issues_from_assessment.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\generate_assessment_summary.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\pragmatic_programmer_review.py

- [!] **Significant duplicate code block**
  - Found in 3 locations across 3 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\create_issues_from_assessment.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\generate_assessment_summary.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\pragmatic_programmer_review.py

- [!] **Significant duplicate code block**
  - Found in 11 locations across 9 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\fix_numpy_compatibility.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\drake_physics_engine.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\src\drake_gui_app.py

- [!] **Significant duplicate code block**
  - Found in 2 locations across 2 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\generate_assessment_summary.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\pragmatic_programmer_review.py

- [!] **Significant duplicate code block**
  - Found in 2 locations across 2 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\generate_assessment_summary.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\assessment\constants.py

- [!] **Significant duplicate code block**
  - Found in 2 locations across 2 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\generate_assessment_summary.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\assessment\constants.py

- [!] **Significant duplicate code block**
  - Found in 2 locations across 2 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\generate_assessment_summary.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\assessment\constants.py

- [!] **Significant duplicate code block**
  - Found in 2 locations across 2 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\generate_assessment_summary.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\assessment\constants.py

- [!] **Significant duplicate code block**
  - Found in 2 locations across 2 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\generate_assessment_summary.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\assessment\constants.py

- [!] **Significant duplicate code block**
  - Found in 2 locations across 2 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\generate_assessment_summary.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\assessment\constants.py

- [!] **Significant duplicate code block**
  - Found in 305 locations across 142 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\maintain_workflows.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\api\server.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\golf_gui_r0\golf_camera_system.py

- [!] **Significant duplicate code block**
  - Found in 9 locations across 9 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\maintain_workflows.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\docker\src\humanoid_golf\visualization.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\biomechanics.py

- [!] **Significant duplicate code block**
  - Found in 7 locations across 7 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\maintain_workflows.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\golf_gui_r0\golf_camera_system.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\golf_data_core.py

- [!] **Significant duplicate code block**
  - Found in 2 locations across 2 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\migrate_api_keys.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\verify_physics.py

- [!] **Significant duplicate code block**
  - Found in 2 locations across 2 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\migrate_api_keys.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\api\auth\dependencies.py

- [!] **Significant duplicate code block**
  - Found in 2 locations across 2 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\migrate_api_keys.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\api\auth\dependencies.py

- [!] **Significant duplicate code block**
  - Found in 2 locations across 2 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\migrate_api_keys.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\api\auth\dependencies.py

- [!] **Significant duplicate code block**
  - Found in 2 locations across 2 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\migrate_api_keys.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\api\auth\dependencies.py

- [!] **Significant duplicate code block**
  - Found in 2 locations across 2 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\migrate_api_keys.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\api\auth\dependencies.py

- [!] **Significant duplicate code block**
  - Found in 2 locations across 2 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\migrate_api_keys.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\api\auth\dependencies.py

- [!] **Significant duplicate code block**
  - Found in 2 locations across 2 files
  - Recommendation: Consolidate into a shared utility or base class
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\migrate_api_keys.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\api\auth\dependencies.py

### Orthogonality & Decoupling

- [!] **God function: main (62 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\examples\01_basic_simulation.py

- [!] **God function: main (66 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\examples\02_parameter_sweeps.py

- [!] **God function: run_tutorial (78 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\examples\03_injury_risk_tutorial.py

- [!] **God function: generate_report (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\analyze_completist_data.py

- [!] **God function: find_duplicates (70 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\check_duplicates.py

- [!] **God function: main (60 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\check_system_health.py

- [!] **God function: diagnose_and_fix_icons (62 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\diagnose_icon_quality.py

- [!] **God function: fix_numpy_compatibility (57 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\fix_numpy_compatibility.py

- [!] **God function: _build_markdown_report (60 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\generate_assessment_summary.py

- [!] **God function: generate_summary (72 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\generate_assessment_summary.py

- [!] **God function: migrate_api_keys (91 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\migrate_api_keys.py

- [!] **God function: main (97 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\migrate_api_keys.py

- [!] **God function: check_dry_violations (60 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\pragmatic_programmer_review.py

- [!] **God function: check_orthogonality (73 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\pragmatic_programmer_review.py

- [!] **God function: check_reversibility (59 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\pragmatic_programmer_review.py

- [!] **God function: check_quality (78 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\pragmatic_programmer_review.py

- [!] **God function: check_robustness (62 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\pragmatic_programmer_review.py

- [!] **God function: check_testing (66 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\pragmatic_programmer_review.py

- [!] **God function: check_documentation (67 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\pragmatic_programmer_review.py

- [!] **God function: check_automation (88 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\pragmatic_programmer_review.py

- [!] **God function: run_review (82 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\pragmatic_programmer_review.py

- [!] **God function: generate_markdown_report (69 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\pragmatic_programmer_review.py

- [!] **God function: create_github_issues (89 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\pragmatic_programmer_review.py

- [!] **God function: main (72 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\pragmatic_programmer_review.py

- [!] **God function: refactor_logging_imports (65 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\refactor_dry_orthogonality.py

- [!] **God function: refactor_path_patterns (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\refactor_dry_orthogonality.py

- [!] **God function: refactor_qapp_patterns (53 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\refactor_dry_orthogonality.py

- [!] **God function: refactor_array_validation (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\refactor_dry_orthogonality.py

- [!] **God function: refactor_path_resolution (59 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\refactor_dry_orthogonality.py

- [!] **God function: run_assessment (73 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\run_assessment.py

- [!] **God function: main (53 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\run_tests_in_docker.py

- [!] **God function: check_build_system (67 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\validate_phase1_upgrades.py

- [!] **God function: check_output_management (73 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\validate_phase1_upgrades.py

- [!] **God function: validate_shared_components (58 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\validate_suite.py

- [!] **God function: main (96 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\verify_installation.py

- [!] **God function: run_verification (110 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\verify_physics.py

- [!] **God function: test_mujoco_humanoid_command (65 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\test_docker_integration.py

- [!] **God function: test_drake_command (66 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\test_docker_integration.py

- [!] **God function: test_pinocchio_command (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\test_docker_integration.py

- [!] **God function: test_docker_launch_command_structure (53 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\test_launcher_fixes.py

- [!] **God function: test_layout_save_load_integration (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\test_layout_persistence.py

- [!] **God function: test_urdf_generation_structure (59 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\test_urdf_generator.py

- [!] **God function: test_jacobian_consistency (53 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\cross_engine\test_mujoco_vs_pinocchio.py

- [!] **God function: test_equation_of_motion_consistency (76 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\cross_engine\test_mujoco_vs_pinocchio.py

- [!] **God function: test_drift_control_superposition (59 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\integration\test_conservation_laws.py

- [!] **God function: test_work_equals_kinetic_energy_change (71 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\integration\test_conservation_laws.py

- [!] **God function: test_mujoco_ball_drop_energy_dissipation (56 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\integration\test_contact_cross_engine.py

- [!] **God function: test_forward_dynamics_trajectory_consistency (80 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\integration\test_cross_engine_consistency.py

- [!] **God function: test_acceleration_triangulation (92 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\integration\test_cross_engine_consistency.py

- [!] **God function: test_inverse_dynamics_agreement (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\integration\test_cross_engine_validation.py

- [!] **God function: test_jacobian_agreement (70 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\integration\test_cross_engine_validation.py

- [!] **God function: test_ztcf_counterfactual_agreement (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\integration\test_cross_engine_validation.py

- [!] **God function: test_zvcf_counterfactual_agreement (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\integration\test_cross_engine_validation.py

- [!] **God function: test_free_fall_energy_conservation (66 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\integration\test_energy_conservation.py

- [!] **God function: test_work_energy_theorem (80 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\integration\test_energy_conservation.py

- [!] **God function: test_power_balance (72 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\integration\test_energy_conservation.py

- [!] **God function: test_zero_gravity_angular_momentum_conservation (54 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\integration\test_energy_conservation.py

- [!] **God function: launcher_env (91 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\integration\test_golf_launcher_integration.py

- [!] **God function: simple_arm_model (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\integration\test_opensim_muscles.py

- [!] **God function: test_pinocchio_golfer_stability (54 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\physics_validation\test_complex_models.py

- [!] **God function: test_mujoco_ballistic_energy_conservation (61 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\physics_validation\test_energy_conservation.py

- [!] **God function: test_pinocchio_energy_check (73 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\physics_validation\test_energy_conservation.py

- [!] **God function: test_drake_energy_conservation (71 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\physics_validation\test_energy_conservation.py

- [!] **God function: test_mujoco_momentum_conservation (81 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\physics_validation\test_momentum_conservation.py

- [!] **God function: test_pinocchio_momentum_conservation (86 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\physics_validation\test_momentum_conservation.py

- [!] **God function: test_mujoco_pendulum_accuracy (79 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\physics_validation\test_pendulum_accuracy.py

- [!] **God function: test_drake_pendulum_accuracy (106 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\physics_validation\test_pendulum_accuracy.py

- [!] **God function: test_password_not_logged (54 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\unit\test_api_security.py

- [!] **God function: mock_pyqt (57 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\unit\test_golf_launcher_logic.py

- [!] **God function: test_empty_state_ux (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\unit\test_launcher_ux.py

- [!] **God function: test_mujoco_iaa_logic (76 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\unit\test_mujoco_induced_acceleration.py

- [!] **God function: mock_casadi (56 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\unit\test_optimize_arm.py

- [!] **God function: test_auto_path_resolution (61 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\unit\test_output_manager.py

- [!] **God function: test_export_to_urdf (82 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\unit\test_urdf_io.py

- [!] **God function: init_db (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\api\database.py

- [!] **God function: _validate_model_path (62 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\api\server.py

- [!] **God function: analyze_video (85 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\api\server.py

- [!] **God function: analyze_video_async (53 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\api\server.py

- [!] **God function: _process_video_background (60 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\api\server.py

- [!] **God function: run (62 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\launchers\docker_manager.py

- [!] **God function: __init__ (107 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\launchers\golf_launcher.py

- [!] **God function: _load_layout (58 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\launchers\golf_launcher.py

- [!] **God function: _center_window (56 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\launchers\golf_launcher.py

- [!] **God function: init_ui (175 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\launchers\golf_launcher.py

- [!] **God function: update_launch_button (53 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\launchers\golf_launcher.py

- [!] **God function: launch_simulation (60 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\launchers\golf_launcher.py

- [!] **God function: _launch_docker_container (85 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\launchers\golf_launcher.py

- [!] **God function: _setup_ui (124 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\launchers\golf_suite_launcher.py

- [!] **God function: main (68 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\launchers\shot_tracer.py

- [!] **God function: _create_controls_panel (131 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\launchers\shot_tracer.py

- [!] **God function: _create_visualization_panel (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\launchers\shot_tracer.py

- [!] **God function: drawContents (101 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\launchers\ui_components.py

- [!] **God function: setup_ui (70 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\launchers\ui_components.py

- [!] **God function: check_links (54 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\tools\check_markdown_links.py

- [!] **God function: main (61 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\tools\code_quality_check.py

- [!] **God function: _setup_menu_bar (100 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\tools\urdf_generator\main_window.py

- [!] **God function: download_human_model (70 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\tools\urdf_generator\model_library.py

- [!] **God function: _create_golf_club_urdf (139 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\tools\urdf_generator\model_library.py

- [!] **God function: _create_human_models_group (58 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\tools\urdf_generator\model_loader_dialog.py

- [!] **God function: convert (73 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\tools\urdf_generator\mujoco_viewer.py

- [!] **God function: _setup_ui (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\tools\urdf_generator\mujoco_viewer.py

- [!] **God function: _validate_urdf (55 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\tools\urdf_generator\mujoco_viewer.py

- [!] **God function: _setup_geometry_tab (77 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\tools\urdf_generator\segment_panel.py

- [!] **God function: _setup_physics_tab (69 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\tools\urdf_generator\segment_panel.py

- [!] **God function: _setup_joint_tab (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\tools\urdf_generator\segment_panel.py

- [!] **God function: _get_segment_data (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\tools\urdf_generator\segment_panel.py

- [!] **God function: _load_segment_data (53 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\tools\urdf_generator\segment_panel.py

- [!] **God function: _validate_physical_parameters (79 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\tools\urdf_generator\urdf_builder.py

- [!] **God function: mirror_for_handedness (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\tools\urdf_generator\urdf_builder.py

- [!] **God function: paintGL (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\tools\urdf_generator\visualization_widget.py

- [!] **God function: _compute_rk4_step (57 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ball_flight_physics.py

- [!] **God function: add_logging_args (58 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\cli_utils.py

- [!] **God function: add_output_args (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\cli_utils.py

- [!] **God function: align_signals (70 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\comparative_analysis.py

- [!] **God function: generate_comparison_report (54 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\comparative_analysis.py

- [!] **God function: plot_comparison (87 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\comparative_plotting.py

- [!] **God function: plot_phase_comparison (72 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\comparative_plotting.py

- [!] **God function: plot_coordination_comparison (79 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\comparative_plotting.py

- [!] **God function: plot_3d_trajectory_comparison (87 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\comparative_plotting.py

- [!] **God function: plot_dashboard (60 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\comparative_plotting.py

- [!] **God function: plot_bland_altman (82 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\comparative_plotting.py

- [!] **God function: setup_structured_logging (104 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\core.py

- [!] **God function: compare_states (73 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\cross_engine_validator.py

- [!] **God function: _log_result (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\cross_engine_validator.py

- [!] **God function: compare_torques_with_rms (74 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\cross_engine_validator.py

- [!] **God function: parse_timestamp (53 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\datetime_utils.py

- [!] **God function: format_duration (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\datetime_utils.py

- [!] **God function: compute_velocity_ellipsoid (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ellipsoid_visualization.py

- [!] **God function: compute_force_ellipsoid (67 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ellipsoid_visualization.py

- [!] **God function: generate_ellipsoid_mesh (79 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ellipsoid_visualization.py

- [!] **God function: check_and_warn (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\energy_monitor.py

- [!] **God function: __init__ (79 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\engine_manager.py

- [!] **God function: probe (84 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\engine_probes.py

- [!] **God function: probe (101 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\engine_probes.py

- [!] **God function: probe (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\engine_probes.py

- [!] **God function: probe (58 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\engine_probes.py

- [!] **God function: probe (59 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\engine_probes.py

- [!] **God function: get_env_int (56 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\environment.py

- [!] **God function: get_env_float (56 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\environment.py

- [!] **God function: validate_api_security (73 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\env_validator.py

- [!] **God function: validate_database_config (56 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\env_validator.py

- [!] **God function: validate_environment (93 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\env_validator.py

- [!] **God function: print_validation_report (63 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\env_validator.py

- [!] **God function: export_to_hdf5 (60 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\export.py

- [!] **God function: export_recording_all_formats (95 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\export.py

- [!] **God function: update_from_mujoco (74 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\grip_contact_model.py

- [!] **God function: extract_grf_from_contacts (58 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ground_reaction_forces.py

- [!] **God function: validate_grf_cross_engine (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ground_reaction_forces.py

- [!] **God function: compute_impulse_metrics (62 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ground_reaction_forces.py

- [!] **God function: analyze (62 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ground_reaction_forces.py

- [!] **God function: create_optimized_icon (69 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\image_utils.py

- [!] **God function: compute_gear_effect_spin (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\impact_model.py

- [!] **God function: validate_energy_balance (66 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\impact_model.py

- [!] **God function: solve (81 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\impact_model.py

- [!] **God function: solve (97 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\impact_model.py

- [!] **God function: save_yaml (56 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\io_utils.py

- [!] **God function: validate_model_against_dataset (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\kaggle_validation.py

- [!] **God function: compare_all_models_to_dataset (59 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\kaggle_validation.py

- [!] **God function: analyze (114 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\kinematic_sequence.py

- [!] **God function: setup_logging (99 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\logging_config.py

- [!] **God function: check_jacobian_conditioning (76 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\manipulability.py

- [!] **God function: fit_segment_pose (127 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\marker_mapping.py

- [!] **God function: plot_registration_diagnostics (77 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\marker_mapping.py

- [!] **God function: extract_synergies (86 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\muscle_analysis.py

- [!] **God function: solve_fiber_length (67 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\muscle_equilibrium.py

- [!] **God function: convert_osim_to_mujoco (101 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\myoconverter_integration.py

- [!] **God function: _validate_inputs (67 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\myoconverter_integration.py

- [!] **God function: _handle_conversion_error (57 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\myoconverter_integration.py

- [!] **God function: step (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\myosuite_adapter.py

- [!] **God function: save_simulation_results (120 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\output_manager.py

- [!] **God function: load_simulation_results (59 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\output_manager.py

- [!] **God function: export_analysis_report (57 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\output_manager.py

- [!] **God function: cleanup_old_files (60 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\output_manager.py

- [!] **God function: _load_default_parameters (188 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\physics_parameters.py

- [!] **God function: plot_angle_angle_diagram (72 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_coupling_angle (66 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_coordination_patterns (116 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_stability_metrics (85 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_dtw_alignment (56 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_correlation_sum (53 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_jerk_trajectory (65 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_lag_matrix (71 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_multiscale_entropy (60 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_club_head_trajectory (54 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_phase_diagram (67 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_frequency_analysis (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_spectrogram (67 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_summary_dashboard (131 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_kinematic_sequence (97 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_work_loop (76 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_x_factor_cycle (81 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_3d_phase_space (54 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_poincare_map_3d (148 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_lyapunov_exponent (101 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_phase_space_reconstruction (116 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_muscle_synergies (87 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_correlation_matrix (75 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_swing_plane (94 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_kinematic_sequence_bars (95 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_activation_heatmap (87 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_phase_space_density (55 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_grf_butterfly_diagram (87 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_angular_momentum_3d (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_stability_diagram (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_radar_chart (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_power_flow (56 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_joint_power_curves (69 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_induced_acceleration (153 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_club_induced_acceleration (83 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_counterfactual_comparison (73 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: _plot_counterfactual_dual (60 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_dynamic_correlation (73 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_3d_vector_field (72 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_local_stability (95 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_wavelet_scalogram (89 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_cross_wavelet (106 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_principal_component_analysis (84 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_joint_stiffness (67 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: plot_dynamic_stiffness (105 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\plotting_core.py

- [!] **God function: run (55 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\process_worker.py

- [!] **God function: capture (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\provenance.py

- [!] **God function: to_header_lines (61 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\provenance.py

- [!] **God function: fit_instantaneous_swing_plane (58 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\reference_frames.py

- [!] **God function: fit_functional_swing_plane (79 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\reference_frames.py

- [!] **God function: secure_popen (68 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\secure_subprocess.py

- [!] **God function: secure_run (73 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\secure_subprocess.py

- [!] **God function: _dtw_core (74 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\signal_processing.py

- [!] **God function: _dtw_path_core (86 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\signal_processing.py

- [!] **God function: compute_spectral_arc_length (84 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\signal_processing.py

- [!] **God function: compute_cwt (103 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\signal_processing.py

- [!] **God function: compute_time_shift (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\signal_processing.py

- [!] **God function: compute_dtw_distance (60 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\signal_processing.py

- [!] **God function: _generate_golf_club_urdf (94 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\standard_models.py

- [!] **God function: compute_coordination_metrics (88 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\statistical_analysis.py

- [!] **God function: generate_comprehensive_report (100 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\statistical_analysis.py

- [!] **God function: analyze_kinematic_sequence (66 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\statistical_analysis.py

- [!] **God function: compute_rolling_correlation (70 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\statistical_analysis.py

- [!] **God function: compute_local_divergence_rate (100 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\statistical_analysis.py

- [!] **God function: compute_swing_profile (121 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\statistical_analysis.py

- [!] **God function: compute_work_metrics (67 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\statistical_analysis.py

- [!] **God function: compute_joint_power_metrics (56 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\statistical_analysis.py

- [!] **God function: compute_impulse_metrics (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\statistical_analysis.py

- [!] **God function: compute_recurrence_matrix (88 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\statistical_analysis.py

- [!] **God function: compute_rqa_metrics (77 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\statistical_analysis.py

- [!] **God function: compute_correlation_dimension (54 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\statistical_analysis.py

- [!] **God function: estimate_lyapunov_exponent (124 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\statistical_analysis.py

- [!] **God function: compute_principal_component_analysis (69 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\statistical_analysis.py

- [!] **God function: compute_permutation_entropy (75 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\statistical_analysis.py

- [!] **God function: compute_joint_stiffness (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\statistical_analysis.py

- [!] **God function: compute_dynamic_stiffness (70 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\statistical_analysis.py

- [!] **God function: compute_fractal_dimension (53 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\statistical_analysis.py

- [!] **God function: compute_sample_entropy (60 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\statistical_analysis.py

- [!] **God function: compute_multiscale_entropy (57 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\statistical_analysis.py

- [!] **God function: compute_jerk_metrics (62 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\statistical_analysis.py

- [!] **God function: compute_lag_matrix (84 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\statistical_analysis.py

- [!] **God function: export_statistics_csv (113 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\statistical_analysis.py

- [!] **God function: kill_process_tree (53 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\subprocess_utils.py

- [!] **God function: compute_kinematic_similarity (56 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\swing_comparison.py

- [!] **God function: compare_peak_speeds (60 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\swing_comparison.py

- [!] **God function: export_scene_json (62 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\swing_plane_visualization.py

- [!] **God function: safe_float (63 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\type_utils.py

- [!] **God function: validate_physical_bounds (72 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\validation.py

- [!] **God function: validate_joint_state (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\validation_helpers.py

- [!] **God function: process_video (69 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\video_pose_pipeline.py

- [!] **God function: get_preset_camera_params (62 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\viewpoint_controls.py

- [!] **God function: _build_default_glossary (379 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\education.py

- [!] **God function: _register_data_tools (164 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\sample_tools.py

- [!] **God function: _register_analysis_tools (111 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\sample_tools.py

- [!] **God function: _register_education_tools (120 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\sample_tools.py

- [!] **God function: _register_validation_tools (117 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\sample_tools.py

- [!] **God function: get_marker_info (57 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\sample_tools.py

- [!] **God function: interpret_torques (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\sample_tools.py

- [!] **God function: create_first_analysis_workflow (127 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\workflow_engine.py

- [!] **God function: execute_next_step (118 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\workflow_engine.py

- [!] **God function: compute_grf_metrics (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\analysis\grf_metrics.py

- [!] **God function: detect_swing_phases (132 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\analysis\phase_detection.py

- [!] **God function: compute_stability_metrics (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\analysis\stability_metrics.py

- [!] **God function: __init__ (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\dashboard\advanced_analysis.py

- [!] **God function: update_plot (81 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\dashboard\advanced_analysis.py

- [!] **God function: update_plot (85 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\dashboard\advanced_analysis.py

- [!] **God function: update_plot (56 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\dashboard\advanced_analysis.py

- [!] **God function: __init__ (59 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\dashboard\advanced_analysis.py

- [!] **God function: update_plot (55 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\dashboard\advanced_analysis.py

- [!] **God function: record_step (138 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\dashboard\recorder.py

- [!] **God function: compute_analysis_post_hoc (79 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\dashboard\recorder.py

- [!] **God function: __init__ (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\dashboard\widgets.py

- [!] **God function: __init__ (184 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\dashboard\widgets.py

- [!] **God function: update_plot (170 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\dashboard\widgets.py

- [!] **God function: __init__ (58 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\dashboard\widgets.py

- [!] **God function: refresh_static_plot (123 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\dashboard\window.py

- [!] **God function: _score_spinal_risks (60 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\injury\injury_risk.py

- [!] **God function: _score_joint_risks (77 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\injury\injury_risk.py

- [!] **God function: _score_technique_risks (57 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\injury\injury_risk.py

- [!] **God function: _generate_recommendations (88 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\injury\injury_risk.py

- [!] **God function: analyze_all_joints (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\injury\joint_stress.py

- [!] **God function: analyze_wrist (55 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\injury\joint_stress.py

- [!] **God function: create_example_analysis (73 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\injury\spinal_load_analysis.py

- [!] **God function: analyze (74 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\injury\spinal_load_analysis.py

- [!] **God function: _compute_segment_loads (76 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\injury\spinal_load_analysis.py

- [!] **God function: _assess_risk (56 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\injury\spinal_load_analysis.py

- [!] **God function: get_recommendations (79 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\injury\spinal_load_analysis.py

- [!] **God function: recommend (116 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\injury\swing_modifications.py

- [!] **God function: optimize (92 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\optimization\swing_optimizer.py

- [!] **God function: estimate_from_image (60 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\pose_estimation\mediapipe_estimator.py

- [!] **God function: _apply_temporal_smoothing (54 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\pose_estimation\mediapipe_estimator.py

- [!] **God function: _keypoints_to_joint_angles (60 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\pose_estimation\mediapipe_estimator.py

- [!] **God function: init_ui (55 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\pose_estimation\openpose_gui.py

- [!] **God function: test_rigid_body_friction_spin (99 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\tests\test_impact_model.py

- [!] **God function: _setup_ui (59 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ui\recent_models.py

- [!] **God function: _setup_ui (132 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ui\shortcuts_overlay.py

- [!] **God function: main (150 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\optimization\examples\optimize_arm.py

- [!] **God function: stream_response (57 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\adapters\anthropic_adapter.py

- [!] **God function: _format_messages (70 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\adapters\anthropic_adapter.py

- [!] **God function: send_message (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\adapters\ollama_adapter.py

- [!] **God function: stream_response (53 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\adapters\ollama_adapter.py

- [!] **God function: validate_connection (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\adapters\ollama_adapter.py

- [!] **God function: stream_response (74 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\adapters\openai_adapter.py

- [!] **God function: _format_messages (60 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\adapters\openai_adapter.py

- [!] **God function: _parse_response (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\adapters\openai_adapter.py

- [!] **God function: _create_header (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\gui\assistant_panel.py

- [!] **God function: _create_input_area (59 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\gui\assistant_panel.py

- [!] **God function: apply_settings (59 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\gui\assistant_panel.py

- [!] **God function: _setup_ui (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\gui\settings_dialog.py

- [!] **God function: _create_provider_tab (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\gui\settings_dialog.py

- [!] **God function: build_simple_arm_model (115 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\models\opensim\examples\build_simple_arm_model.py

- [!] **God function: train_policy (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\models\myosuite\examples\train_elbow_policy.py

- [!] **God function: main (53 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\models\myosuite\examples\train_elbow_policy.py

- [!] **God function: test_show_status (54 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\launchers\tests\test_unified_launcher.py

- [!] **God function: main (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\scripts\quality-check.py

- [!] **God function: points_dataframe (129 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\python\src\c3d_reader.py

- [!] **God function: get_force_plate_channels (60 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\python\src\c3d_reader.py

- [!] **God function: force_plate_dataframe (127 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\python\src\c3d_reader.py

- [!] **God function: _export_dataframe (110 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\python\src\c3d_reader.py

- [!] **God function: test_c3d_viewer_open_file_ux (85 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\python\tests\test_c3d_viewer_ux.py

- [!] **God function: load_c3d_file (100 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\python\src\apps\services\c3d_loader.py

- [!] **God function: _update_time_series (55 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\python\src\apps\ui\tabs\force_plot_tab.py

- [!] **God function: _update_cop_trajectory (69 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\python\src\apps\ui\tabs\force_plot_tab.py

- [!] **God function: update_plot (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\python\src\apps\ui\tabs\marker_plot_tab.py

- [!] **God function: update_view (69 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\python\src\apps\ui\tabs\viewer_3d_tab.py

- [!] **God function: analyze_coordinate_system (228 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Motion Capture Plotter\analyze_coordinate_system.py

- [!] **God function: analyze_simscape_data (284 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Motion Capture Plotter\analyze_simscape_data.py

- [!] **God function: create_control_panel (199 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Motion Capture Plotter\Motion_Capture_Plotter.py

- [!] **God function: load_excel_file (114 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Motion Capture Plotter\Motion_Capture_Plotter.py

- [!] **God function: load_simscape_csv (119 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Motion Capture Plotter\Motion_Capture_Plotter.py

- [!] **God function: print_data_debug (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Motion Capture Plotter\Motion_Capture_Plotter.py

- [!] **God function: visualize_motion_capture_data (176 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Motion Capture Plotter\Motion_Capture_Plotter.py

- [!] **God function: visualize_simscape_data (284 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Motion Capture Plotter\Motion_Capture_Plotter.py

- [!] **God function: update_info_text (59 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Motion Capture Plotter\Motion_Capture_Plotter.py

- [!] **God function: on_mouse_move (55 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Motion Capture Plotter\Motion_Capture_Plotter.py

- [!] **God function: update_inertia (53 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\golf_gui_r0\golf_camera_system.py

- [!] **God function: _update_animation (56 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\golf_gui_r0\golf_camera_system.py

- [!] **God function: update_cinematic_camera (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\golf_gui_r0\golf_camera_system.py

- [!] **God function: main (68 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\golf_gui_r0\golf_main_application.py

- [!] **God function: load_data_files (66 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\golf_gui_r0\golf_main_application.py

- [!] **God function: _show_export_dialog (72 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\golf_gui_r0\golf_main_application.py

- [!] **God function: extract_frame_data (57 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\golf_gui_r0\golf_visualizer_implementation.py

- [!] **God function: __init__ (74 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\golf_gui_r0\golf_visualizer_implementation.py

- [!] **God function: _render_body_segments (69 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\golf_gui_r0\golf_visualizer_implementation.py

- [!] **God function: _render_vectors (54 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\golf_gui_r0\golf_visualizer_implementation.py

- [!] **God function: _apply_modern_style (86 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\golf_gui_r0\golf_visualizer_implementation.py

- [!] **God function: deep_analyze_matlab_file (75 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\detailed_data_analysis.py

- [!] **God function: extract_actual_data (57 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\detailed_data_analysis.py

- [!] **God function: main (54 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\detailed_data_analysis.py

- [!] **God function: test_data_loading_accuracy (73 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\final_robustness_test.py

- [!] **God function: test_data_consistency (59 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\final_robustness_test.py

- [!] **God function: generate_final_report (56 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\final_robustness_test.py

- [!] **God function: get_column_data (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\golf_data_core.py

- [!] **God function: create_arrow_mesh (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\golf_data_core.py

- [!] **God function: _apply_modern_style (139 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\golf_gui_application.py

- [!] **God function: calculate_inverse_dynamics (57 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\golf_inverse_dynamics.py

- [!] **God function: _render_ground (54 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\golf_opengl_renderer.py

- [!] **God function: _render_body_segments (125 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\golf_opengl_renderer.py

- [!] **God function: _render_cylinder_between_points (68 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\golf_opengl_renderer.py

- [!] **God function: _render_club (112 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\golf_opengl_renderer.py

- [!] **God function: export_video (71 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\golf_video_export.py

- [!] **God function: _start_ffmpeg_process (57 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\golf_video_export.py

- [!] **God function: _render_frame_to_buffer (61 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\golf_video_export.py

- [!] **God function: _setup_ui (78 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\golf_video_export.py

- [!] **God function: _start_export (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\golf_video_export.py

- [!] **God function: _create_data_panel (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\golf_wiffle_main.py

- [!] **God function: _create_analysis_panel (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\golf_wiffle_main.py

- [!] **God function: _apply_modern_style (53 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\golf_wiffle_main.py

- [!] **God function: create_dialog (92 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\gui_performance_options.py

- [!] **God function: main (61 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\simple_data_test.py

- [!] **God function: create_sample_data (68 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\test_improved_visualization.py

- [!] **God function: main (55 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\test_improved_visualization.py

- [!] **God function: main (57 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\test_signal_bus_compatibility.py

- [!] **God function: __post_init__ (71 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\wiffle_data_loader.py

- [!] **God function: _process_sheet_data (119 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\wiffle_data_loader.py

- [!] **God function: _create_body_part_estimates (70 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\src\apps\golf_gui\Simscape Multibody Data Plotters\Python Version\integrated_golf_gui_r0\wiffle_data_loader.py

- [!] **God function: test_updated_environment (77 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\add_defusedxml_to_robotics_env.py

- [!] **God function: test_qt_environment (53 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\add_qt_dependencies.py

- [!] **God function: test_specific_imports (54 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\docker_test_dependencies.py

- [!] **God function: generate_flexible_club_xml (145 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\head_models.py

- [!] **God function: generate_rigid_club_xml (68 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\head_models.py

- [!] **God function: test_docker_venv (132 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\test_docker_venv.py

- [!] **God function: main (62 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\pinocchio\scripts\quality_check.py

- [!] **God function: __init__ (108 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\pinocchio\python\pinocchio_golf\gui.py

- [!] **God function: _setup_ui (176 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\pinocchio\python\pinocchio_golf\gui.py

- [!] **God function: _setup_analysis_tab (58 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\pinocchio\python\pinocchio_golf\gui.py

- [!] **God function: _generate_plot (55 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\pinocchio\python\pinocchio_golf\gui.py

- [!] **God function: _plot_induced_accelerations (85 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\pinocchio\python\pinocchio_golf\gui.py

- [!] **God function: load_urdf (96 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\pinocchio\python\pinocchio_golf\gui.py

- [!] **God function: _game_loop (159 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\pinocchio\python\pinocchio_golf\gui.py

- [!] **God function: _draw_ellipsoids (63 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\pinocchio\python\pinocchio_golf\gui.py

- [!] **God function: _draw_induced_vectors (77 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\pinocchio\python\pinocchio_golf\gui.py

- [!] **God function: compute_components (65 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\pinocchio\python\pinocchio_golf\induced_acceleration.py

- [!] **God function: compute_metrics (72 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\pinocchio\python\pinocchio_golf\manipulability.py

- [!] **God function: test_ik_convergence (56 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\pinocchio\python\tests\test_ik_simple.py

- [!] **God function: solve_ik (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\pinocchio\python\dtack\backends\pink_backend.py

- [!] **God function: _run_counterfactual (65 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\pinocchio\python\dtack\gui\main_window.py

- [!] **God function: _generate_segment_urdf (59 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\pinocchio\python\dtack\utils\urdf_exporter.py

- [!] **God function: _generate_universal_joint (72 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\pinocchio\python\dtack\utils\urdf_exporter.py

- [!] **God function: _generate_gimbal_joint (92 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\pinocchio\python\dtack\utils\urdf_exporter.py

- [!] **God function: visualize_frame (73 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\pinocchio\python\dtack\viz\rob_neal_viewer.py

- [!] **God function: compute_zvcf (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\pendulum\python\pendulum_physics_engine.py

- [!] **God function: add_cylindrical_wrap (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\opensim\python\muscle_analysis.py

- [!] **God function: init_ui (55 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\opensim\python\opensim_gui.py

- [!] **God function: compute_jacobian (97 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\opensim\python\opensim_physics_engine.py

- [!] **God function: _rotation_difference (54 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\opensim\python\opensim_physics_engine.py

- [!] **God function: compute_ztcf (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\opensim\python\opensim_physics_engine.py

- [!] **God function: _load_opensim (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\opensim\python\opensim_golf\core.py

- [!] **God function: _run_opensim_simulation (89 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\opensim\python\opensim_golf\core.py

- [!] **God function: mock_opensim_env (55 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\opensim\python\tests\test_opensim_core.py

- [!] **God function: customize_model (89 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\docker\example_dynamic_stance.py

- [!] **God function: main (188 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\docker\example_dynamic_stance.py

- [!] **God function: main (72 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\docker\example_golf_swing.py

- [!] **God function: colorize_humanoid (69 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\docker\example_golf_swing.py

- [!] **God function: main (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\docker\measure_height.py

- [!] **God function: _launch_drake (58 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\golf_suite_launcher.py

- [!] **God function: setup_sim_tab (144 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\humanoid_launcher.py

- [!] **God function: setup_appearance_tab (70 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\humanoid_launcher.py

- [!] **God function: setup_equip_tab (67 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\humanoid_launcher.py

- [!] **God function: open_polynomial_generator (95 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\humanoid_launcher.py

- [!] **God function: get_simulation_command (88 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\humanoid_launcher.py

- [!] **God function: plot_induced_acceleration (89 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\humanoid_launcher.py

- [!] **God function: main (66 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\examples\example_screw_theory.py

- [!] **God function: _compute_hybrid_control (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\advanced_control.py

- [!] **God function: _compute_task_space_control (77 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\advanced_control.py

- [!] **God function: compute_operational_space_control (84 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\advanced_control.py

- [!] **God function: export_to_matlab (55 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\advanced_export.py

- [!] **God function: export_to_hdf5 (69 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\advanced_export.py

- [!] **God function: export_to_c3d (57 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\advanced_export.py

- [!] **God function: _export_to_c3d_ezc3d (73 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\advanced_export.py

- [!] **God function: export_recording_all_formats (71 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\advanced_export.py

- [!] **God function: create_matlab_script (152 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\advanced_export.py

- [!] **God function: show_advanced_plots_dialog (374 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\advanced_gui_methods.py

- [!] **God function: solve_inverse_kinematics (95 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\advanced_kinematics.py

- [!] **God function: extract_full_state (98 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\biomechanics.py

- [!] **God function: export_to_dict (103 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\biomechanics.py

- [!] **God function: __init__ (57 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\control_system.py

- [!] **God function: ztcf (86 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\counterfactuals.py

- [!] **God function: zvcf (83 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\counterfactuals.py

- [!] **God function: plot_counterfactual_comparison (63 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\counterfactuals.py

- [!] **God function: decompose (126 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\drift_control.py

- [!] **God function: plot_decomposition (62 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\drift_control.py

- [!] **God function: run_simulation (63 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\examples_chaotic_pendulum.py

- [!] **God function: plot_results (71 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\examples_chaotic_pendulum.py

- [!] **God function: example_6_sensitivity_to_initial_conditions (104 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\examples_chaotic_pendulum.py

- [!] **God function: example_constraint_forces (77 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\examples_joint_analysis.py

- [!] **God function: example_3_compute_kinematic_forces (63 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\examples_motion_capture.py

- [!] **God function: example_4_inverse_dynamics (78 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\examples_motion_capture.py

- [!] **God function: example_5_complete_analysis_pipeline (106 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\examples_motion_capture.py

- [!] **God function: example_6_swing_comparison (76 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\examples_motion_capture.py

- [!] **God function: __init__ (63 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\grip_modelling_tab.py

- [!] **God function: load_current_hand_model (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\grip_modelling_tab.py

- [!] **God function: _prepare_scene_xml (233 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\grip_modelling_tab.py

- [!] **God function: _add_joint_control_row (75 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\grip_modelling_tab.py

- [!] **God function: screen_to_ray (60 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\interactive_manipulation.py

- [!] **God function: pick_body (66 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\interactive_manipulation.py

- [!] **God function: drag_to (83 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\interactive_manipulation.py

- [!] **God function: _solve_ik_for_body (104 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\interactive_manipulation.py

- [!] **God function: export_inverse_dynamics_to_csv (98 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\inverse_dynamics.py

- [!] **God function: compute_torques_with_posture (76 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\inverse_dynamics.py

- [!] **God function: compute_induced_accelerations (105 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\inverse_dynamics.py

- [!] **God function: compute_end_effector_forces (57 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\inverse_dynamics.py

- [!] **God function: validate_solution (58 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\inverse_dynamics.py

- [!] **God function: analyze_captured_motion (61 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\inverse_dynamics.py

- [!] **God function: analyze_constraint_forces_over_time (57 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\joint_analysis.py

- [!] **God function: analyze_torque_transmission (53 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\joint_analysis.py

- [!] **God function: export_kinematic_forces_to_csv (77 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\kinematic_forces.py

- [!] **God function: __init__ (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\kinematic_forces.py

- [!] **God function: decompose_coriolis_forces (60 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\kinematic_forces.py

- [!] **God function: compute_club_head_apparent_forces (90 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\kinematic_forces.py

- [!] **God function: analyze_trajectory (62 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\kinematic_forces.py

- [!] **God function: compute_effective_mass (149 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\kinematic_forces.py

- [!] **God function: compute_metrics (106 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\manipulability.py

- [!] **God function: load_model_geometry (75 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\meshcat_adapter.py

- [!] **God function: draw_induced_vectors (62 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\meshcat_adapter.py

- [!] **God function: generate_flexible_club_xml (145 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\models.py

- [!] **God function: generate_rigid_club_xml (68 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\models.py

- [!] **God function: golf_swing_marker_set (54 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\motion_capture.py

- [!] **God function: _solve_frame_ik (77 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\motion_capture.py

- [!] **God function: optimize_trajectory (75 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\motion_optimization.py

- [!] **God function: _generate_initial_guess (53 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\motion_optimization.py

- [!] **God function: _simulate_trajectory (117 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\motion_optimization.py

- [!] **God function: set_shaft_properties (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\physics_engine.py

- [!] **God function: _compute_shaft_modes (59 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\physics_engine.py

- [!] **God function: compute_end_effector_jacobian (53 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\pinocchio_interface.py

- [!] **God function: update (55 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\playback_control.py

- [!] **God function: __init__ (94 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\polynomial_generator.py

- [!] **God function: _setup_ui (133 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\polynomial_generator.py

- [!] **God function: compute_power_flow (136 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\power_flow.py

- [!] **God function: compute_inter_segment_transfer (99 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\power_flow.py

- [!] **God function: plot_power_flow (65 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\power_flow.py

- [!] **God function: _save_file_to_library (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\recording_library.py

- [!] **God function: update_recording (55 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\recording_library.py

- [!] **God function: delete_recording (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\recording_library.py

- [!] **God function: search_recordings (68 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\recording_library.py

- [!] **God function: get_statistics (59 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\recording_library.py

- [!] **God function: plot_screw_axis_3d (65 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\screw_kinematics.py

- [!] **God function: compute_twist (53 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\screw_kinematics.py

- [!] **God function: compute_screw_axis (66 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\screw_kinematics.py

- [!] **God function: __init__ (117 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\sim_widget.py

- [!] **God function: _finalize_model_load (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\sim_widget.py

- [!] **God function: reset_state (54 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\sim_widget.py

- [!] **God function: _compute_model_bounds (67 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\sim_widget.py

- [!] **God function: compute_ellipsoids (63 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\sim_widget.py

- [!] **God function: _on_timer (91 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\sim_widget.py

- [!] **God function: _render_once (87 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\sim_widget.py

- [!] **God function: _add_manipulation_overlays (59 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\sim_widget.py

- [!] **God function: mousePressEvent (71 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\sim_widget.py

- [!] **God function: mouseMoveEvent (58 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\sim_widget.py

- [!] **God function: _add_frame_and_com_overlays (57 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\sim_widget.py

- [!] **God function: _create_joint (90 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\urdf_io.py

- [!] **God function: import_from_urdf (99 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\urdf_io.py

- [!] **God function: _build_mujoco_body (98 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\urdf_io.py

- [!] **God function: check_body_jacobian (55 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\verification.py

- [!] **God function: create_metrics_overlay (59 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\video_export.py

- [!] **God function: export_simulation_video (119 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\video_export.py

- [!] **God function: export_recording (76 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\video_export.py

- [!] **God function: __init__ (166 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\__main__.py

- [!] **God function: test_export_to_urdf (59 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\tests\test_urdf_io.py

- [!] **God function: generate_four_bar_linkage_xml (111 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\linkage_mechanisms\__init__.py

- [!] **God function: generate_slider_crank_xml (103 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\linkage_mechanisms\__init__.py

- [!] **God function: generate_scotch_yoke_xml (77 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\linkage_mechanisms\__init__.py

- [!] **God function: generate_geneva_mechanism_xml (72 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\linkage_mechanisms\__init__.py

- [!] **God function: generate_peaucellier_linkage_xml (75 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\linkage_mechanisms\__init__.py

- [!] **God function: generate_chebyshev_linkage_xml (85 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\linkage_mechanisms\__init__.py

- [!] **God function: generate_pantograph_xml (82 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\linkage_mechanisms\__init__.py

- [!] **God function: generate_delta_robot_xml (136 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\linkage_mechanisms\__init__.py

- [!] **God function: generate_five_bar_parallel_xml (95 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\linkage_mechanisms\__init__.py

- [!] **God function: generate_stewart_platform_xml (194 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\linkage_mechanisms\__init__.py

- [!] **God function: generate_watt_linkage_xml (83 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\linkage_mechanisms\__init__.py

- [!] **God function: generate_oldham_coupling_xml (79 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\linkage_mechanisms\__init__.py

- [!] **God function: aba (284 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\rigid_body_dynamics\aba.py

- [!] **God function: crba (159 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\rigid_body_dynamics\crba.py

- [!] **God function: compute_components (91 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\rigid_body_dynamics\induced_acceleration.py

- [!] **God function: compute_task_space_components (71 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\rigid_body_dynamics\induced_acceleration.py

- [!] **God function: rnea (216 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\rigid_body_dynamics\rnea.py

- [!] **God function: adjoint_transform (70 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\screw_theory\adjoint.py

- [!] **God function: exponential_map (76 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\screw_theory\exponential.py

- [!] **God function: logarithmic_map (91 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\screw_theory\exponential.py

- [!] **God function: screw_axis (59 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\screw_theory\screws.py

- [!] **God function: mci (57 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\spatial_algebra\inertia.py

- [!] **God function: jcalc (145 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\spatial_algebra\joints.py

- [!] **God function: crm (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\spatial_algebra\spatial_vectors.py

- [!] **God function: crf (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\spatial_algebra\spatial_vectors.py

- [!] **God function: cross_motion (56 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\spatial_algebra\spatial_vectors.py

- [!] **God function: cross_force (56 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\spatial_algebra\spatial_vectors.py

- [!] **God function: cross_motion_axis (64 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\spatial_algebra\spatial_vectors.py

- [!] **God function: xtrans (53 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\spatial_algebra\transforms.py

- [!] **God function: _load_stylesheet (117 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\gui\core\main_window.py

- [!] **God function: _create_status_bar (57 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\gui\core\main_window.py

- [!] **God function: _update_status_bar (57 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\gui\core\main_window.py

- [!] **God function: on_export_csv (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\gui\tabs\analysis_tab.py

- [!] **God function: _setup_ui (128 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\gui\tabs\controls_tab.py

- [!] **God function: _create_advanced_actuator_control (67 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\gui\tabs\controls_tab.py

- [!] **God function: _refresh_kinematic_controls (87 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\gui\tabs\controls_tab.py

- [!] **God function: __init__ (134 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\gui\tabs\controls_tab.py

- [!] **God function: _update_analysis (76 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\gui\tabs\manipulability_tab.py

- [!] **God function: _setup_ui (293 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\gui\tabs\manipulation_tab.py

- [!] **God function: _init_model_configs (166 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\gui\tabs\physics_tab.py

- [!] **God function: _setup_ui (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\gui\tabs\physics_tab.py

- [!] **God function: _setup_ui (117 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\gui\tabs\plotting_tab.py

- [!] **God function: on_generate_plot (229 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\gui\tabs\plotting_tab.py

- [!] **God function: _setup_ui (332 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\python\mujoco_humanoid_golf\gui\tabs\visualization_tab.py

- [!] **God function: __init__ (81 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\docker\gui\deepmind_control_suite_MuJoCo_GUI.py

- [!] **God function: setup_styles (115 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\docker\gui\deepmind_control_suite_MuJoCo_GUI.py

- [!] **God function: setup_sim_tab (292 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\docker\gui\deepmind_control_suite_MuJoCo_GUI.py

- [!] **God function: setup_appearance_tab (157 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\docker\gui\deepmind_control_suite_MuJoCo_GUI.py

- [!] **God function: setup_equip_tab (143 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\docker\gui\deepmind_control_suite_MuJoCo_GUI.py

- [!] **God function: rebuild_docker (100 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\docker\gui\deepmind_control_suite_MuJoCo_GUI.py

- [!] **God function: _run_docker_process (184 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\docker\gui\deepmind_control_suite_MuJoCo_GUI.py

- [!] **God function: run_update (78 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\docker\gui\deepmind_control_suite_MuJoCo_GUI.py

- [!] **God function: compute_induced_accelerations (85 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\docker\src\humanoid_golf\iaa_helper.py

- [!] **God function: run_simulation (238 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\docker\src\humanoid_golf\sim.py

- [!] **God function: load_humanoid_with_props (106 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\docker\src\humanoid_golf\utils.py

- [!] **God function: _attach_club (56 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\docker\src\humanoid_golf\utils.py

- [!] **God function: customize_visuals (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\docker\src\humanoid_golf\utils.py

- [!] **God function: add_visualization_overlays (62 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\mujoco\docker\src\humanoid_golf\visualization.py

- [!] **God function: compute_ztcf (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\drake_physics_engine.py

- [!] **God function: compute_zvcf (54 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\drake_physics_engine.py

- [!] **God function: optimize_trajectory (72 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\motion_optimization.py

- [!] **God function: main (62 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\scripts\quality_check.py

- [!] **God function: build_golf_swing_diagram (57 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\src\drake_golf_model.py

- [!] **God function: generate (305 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\src\drake_golf_model.py

- [!] **God function: _init_simulation (71 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\src\drake_gui_app.py

- [!] **God function: _setup_ui (250 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\src\drake_gui_app.py

- [!] **God function: _build_kinematic_controls (97 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\src\drake_gui_app.py

- [!] **God function: _game_loop (156 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\src\drake_gui_app.py

- [!] **God function: _update_vectors (113 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\src\drake_gui_app.py

- [!] **God function: _update_ellipsoids (79 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\src\drake_gui_app.py

- [!] **God function: _show_overlay_dialog (58 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\src\drake_gui_app.py

- [!] **God function: _show_induced_acceleration_plot (91 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\src\drake_gui_app.py

- [!] **God function: _show_counterfactuals_plot (55 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\src\drake_gui_app.py

- [!] **God function: _show_advanced_plots (66 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\src\drake_gui_app.py

- [!] **God function: _draw_ellipsoids (87 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\src\drake_gui_app.py

- [!] **God function: compute_components (99 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\src\induced_acceleration.py

- [!] **God function: compute_metrics (108 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\src\manipulability.py

- [!] **God function: mcI (74 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\src\spatial_algebra\inertia.py

- [!] **God function: jcalc (106 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\src\spatial_algebra\joints.py

- [!] **God function: crm (65 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\src\spatial_algebra\spatial_vectors.py

- [!] **God function: crf (66 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\src\spatial_algebra\spatial_vectors.py

- [!] **God function: spatial_cross (98 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\src\spatial_algebra\spatial_vectors.py

- [!] **God function: xtrans (56 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\src\spatial_algebra\transforms.py

- [!] **God function: inv_xtrans (54 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\physics_engines\drake\python\src\spatial_algebra\transforms.py

- [!] **God function: main (62 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\pendulum_models\scripts\quality_check.py

- [!] **God function: _calc_mass_matrix (77 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\pendulum_models\python\double_pendulum_model\physics\triple_pendulum.py

- [!] **God function: _calc_bias_vector (65 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\pendulum_models\python\double_pendulum_model\physics\triple_pendulum.py

- [!] **God function: step (56 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\pendulum_models\python\double_pendulum_model\physics\triple_pendulum.py

- [!] **God function: _setup_physical_parameters (54 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\pendulum_models\python\double_pendulum_model\ui\double_pendulum_gui.py

- [!] **God function: _setup_simulation_options (71 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\pendulum_models\python\double_pendulum_model\ui\double_pendulum_gui.py

- [!] **God function: _update_pendulum_immediately (87 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\pendulum_models\python\double_pendulum_model\ui\double_pendulum_gui.py

- [!] **God function: _calculate_3d_positions (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\pendulum_models\python\double_pendulum_model\ui\double_pendulum_gui.py

- [!] **God function: _draw_reference_lines (69 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\pendulum_models\python\double_pendulum_model\ui\double_pendulum_gui.py

- [!] **God function: _draw_segments (96 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\pendulum_models\python\double_pendulum_model\ui\double_pendulum_gui.py

- [!] **God function: _build_form (51 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\engines\pendulum_models\python\double_pendulum_model\ui\pendulum_pyqt_app.py

- [!] **God function: get_current_user_from_api_key (104 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\api\auth\dependencies.py

- [!] **God function: run_simulation (86 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\api\services\simulation_service.py

- [!] **God function: main (52 lines)**
  - Functions over 50 lines violate single responsibility
  - Recommendation: Break into smaller, focused functions
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\installer\windows\build_installer.py

- [!] **Excessive global state (15 globals)**
  - Global variables create hidden dependencies
  - Recommendation: Use dependency injection or encapsulation
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\api\server.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\launchers\golf_launcher.py

### Reversibility & Flexibility

- [!] **Hardcoded API key**
  - Configuration should be external, not hardcoded
  - Recommendation: Use environment variables or config files
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\test_ai_adapters.py

- [!] **Hardcoded password**
  - Configuration should be external, not hardcoded
  - Recommendation: Use environment variables or config files
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\tests\unit\test_api_security.py

- [!] **Hardcoded host**
  - Configuration should be external, not hardcoded
  - Recommendation: Use environment variables or config files
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\api\server.py

- [!] **Hardcoded API key**
  - Configuration should be external, not hardcoded
  - Recommendation: Use environment variables or config files
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\adapters\anthropic_adapter.py

- [!] **Hardcoded host**
  - Configuration should be external, not hardcoded
  - Recommendation: Use environment variables or config files
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\adapters\ollama_adapter.py

- [!] **Hardcoded API key**
  - Configuration should be external, not hardcoded
  - Recommendation: Use environment variables or config files
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\adapters\openai_adapter.py

- [!] **Hardcoded host**
  - Configuration should be external, not hardcoded
  - Recommendation: Use environment variables or config files
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\src\shared\python\ai\gui\settings_dialog.py

### Code Quality & Craftsmanship

- [!] **Known bugs: 2 fix-needed comments**
  - Fix markers indicate known problems
  - Recommendation: Fix or create issues for known bugs
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\analyze_completist_data.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\src\tools\matlab_utilities\scripts\matlab_quality_check.py

- [-] **Low type hint coverage**
  - Type hints improve code clarity and catch errors
  - Recommendation: Add type hints to function signatures

### Error Handling & Robustness

- [!] **Overly broad exception handling (502 found)**
  - Catching 'Exception' hides specific errors
  - Recommendation: Catch specific exception types
  - Files: C:\Users\diete\Repositories\Golf_Modeling_Suite\launch_golf_suite.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\setup_golf_suite.py, C:\Users\diete\Repositories\Golf_Modeling_Suite\scripts\apply_quick_fixes.py


---

*Generated by Pragmatic Programmer Review workflow*
*Based on "The Pragmatic Programmer" by David Thomas and Andrew Hunt*
