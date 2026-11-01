# Capability Baseline Inventory: Analysis (Part 2/2)

<!-- Generated: python -m scripts.generate_capability_baseline -->

Part of the [ORG-01 baseline index](../ORG01_CAPABILITY_BASELINE.md).

| ID | Name | Entity Kind | Primary Workspace | Secondary Links | Authority | Lifecycle | Evidence | Acceptance Owner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `perturbation_analysis` | Perturbation Analysis | `tile` | `analysis` | — | `core` | 🟢 active_feature | `src/config/models.yaml` | `tests/config/test_launcher_registry_parity.py` |
| `putting_analytics` | Putting Analytics | `tile` | `analysis` | — | `core` | 🟢 active_feature | `src/tools/putting_analytics/__main__.py` | `tests/config/test_launcher_registry_parity.py` |
| `rate_of_closure` | Rate of Closure Impact Explorer | `tile` | `analysis` | — | `tools` | 🟢 active_feature | `src/config/models.yaml` | `tests/config/test_launcher_registry_parity.py` |
| `render.body_appearance` | Golfer body appearance with a visible head, face and neck in every engine render | `feature` | `analysis` | — | `core` | 🔵 planned | `src/shared/python/model_appearance/head.py` | `tests/unit/model_appearance/test_head_model.py` |
| `render.club_head_and_ball` | Realistic Club Head And Ball Rendering | `feature` | `analysis` | — | `core` | 🔵 planned | `src/shared/python/model_appearance/club_head_mesh.py` | `tests/unit/model_appearance/test_club_head_mesh.py` |
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
| `tools.model_explorer.results_playback` | Model Explorer results playback on the shared-appearance golfer | `feature` | `analysis` | — | `core` | 🔵 planned | `src/tools/model_explorer/results_playback_panel.py` | `tests/ui/explorer/test_results_playback_panel.py` |
| `tools.putting_analytics` | Putting analytics studies | `feature` | `analysis` | — | `core` | 🟢 active_feature | `src/tools/putting_analytics/gui.py` | `tests/config/feature_parity/test_registry.py` |
| `tools.shot_pattern_analysis` | Shot Pattern Analysis Workbench | `feature` | `analysis` | — | `core` | 🟢 active_feature | `src/tools/shot_pattern_analysis/gui.py` | `tests/config/test_feature_parity_registry.py` |
| `tools.spine_study` | Segmental spine study execution and model asset selection | `feature` | `analysis` | — | `core` | 🟢 active_feature | `src/api/routes/spine_viewer.py` | `tests/api/test_spine_viewer_route.py` |
| `tools.spine_viewer` | Segmental spine viewer | `feature` | `analysis` | — | `core` | 🟢 active_feature | `src/tools/spine_viewer/gui.py` | `tests/tools/spine_viewer/test_core.py` |
| `tools_calculator_hub` | Tools Calculator Suite | `cli_tool` | `analysis` | — | `tools` | 🟢 active_feature | `src/config/models.yaml` | `tests/config/test_launcher_registry_parity.py` |
| `tools_video_analyzer` | Tools Package: video_analyzer | `library` | `analysis` | `simulation` | `core` | 🟣 intentionally_headless | `src/tools/video_analyzer/__init__.py` | `tests/config/test_registry_exclusions.py` |
