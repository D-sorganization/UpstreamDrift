# Capability Baseline Inventory: Analysis (Part 2/2)

<!-- Generated: python -m scripts.generate_capability_baseline -->

Part of the [ORG-01 baseline index](../ORG01_CAPABILITY_BASELINE.md).

| ID | Name | Entity Kind | Primary Workspace | Secondary Links | Authority | Lifecycle | Evidence | Acceptance Owner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `reporting.swing_report_dialog` | Swing Report Full/Custom plot selection and package export | `feature` | `analysis` | — | `core` | 🟢 active_feature | `src/api/routes/swing_report.py` | `tests/unit/api/test_routes_swing_report.py` |
| `shot_pattern_analysis` | Shot Pattern Analysis | `tile` | `analysis` | — | `core` | 🟢 active_feature | `src/tools/shot_pattern_analysis/__main__.py` | `tests/config/test_launcher_registry_parity.py` |
| `shot_tracer` | Shot Tracer | `tile` | `analysis` | — | `core` | 🟢 active_feature | `src/launchers/_shot_tracer_gui.py` | `tests/config/test_launcher_registry_parity.py` |
| `simulation.shot_tracer` | Shot Tracer / ball-flight visualization | `feature` | `analysis` | — | `core` | 🟢 active_feature | `src/launchers/_shot_tracer_gui.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
| `spine_viewer` | Spine Viewer | `tile` | `analysis` | — | `core` | 🟢 active_feature | `src/tools/spine_viewer/gui.py` | `tests/tools/spine_viewer/test_gui.py` |
| `tools.data_explorer` | Data Explorer (import/filter/visualize datasets) | `feature` | `analysis` | — | `core` | 🔵 planned | `src/api/routes/data_explorer.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
| `tools.launch_monitor_analytics` | Launch-monitor import, interdependency analysis, monitor comparison, dispersion, and longitudinal trends | `feature` | `analysis` | — | `core` | 🔵 planned | `src/tools/launch_monitor_analytics/gui.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
| `tools.matched_swing_browser` | Matched Swing Results Browser (browse and filter matched swing execution receipts) | `feature` | `analysis` | `simulation`, `capture` | `core` | 🟢 active_feature | `src/tools/matched_swing_browser/gui.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
| `tools.model_explorer` | Model Explorer (browse/select/build URDF-MJCF) | `feature` | `analysis` | — | `core` | 🔵 planned | `src/tools/model_explorer/launch_model_explorer.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
| `tools.model_explorer.frankenstein_assembly` | Frankenstein drag-and-drop assembly with typed attachment ports | `feature` | `analysis` | — | `core` | 🔵 planned | `src/tools/model_explorer/frankenstein_editor/assembly_panel.py` | `tests/unit/tools/model_explorer/test_assembly_panel_gui.py` |
| `tools.shot_pattern_analysis` | Shot Pattern Analysis Workbench | `feature` | `analysis` | — | `core` | 🟢 active_feature | `src/tools/shot_pattern_analysis/gui.py` | `tests/config/test_feature_parity_registry.py` |
| `tools.spine_viewer` | Segmental spine viewer | `feature` | `analysis` | — | `core` | 🟢 active_feature | `src/tools/spine_viewer/gui.py` | `tests/tools/spine_viewer/test_core.py` |
| `tools_calculator_hub` | Tools Calculator Suite | `cli_tool` | `analysis` | — | `tools` | 🟢 active_feature | `src/config/models.yaml` | `tests/config/test_launcher_registry_parity.py` |
| `tools_video_analyzer` | Tools Package: video_analyzer | `library` | `analysis` | `simulation` | `core` | 🟣 intentionally_headless | `src/tools/video_analyzer/__init__.py` | `tests/config/test_registry_exclusions.py` |
