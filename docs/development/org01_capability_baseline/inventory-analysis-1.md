# Capability Baseline Inventory: Analysis (Part 1/2)

<!-- Generated: python -m scripts.generate_capability_baseline -->

Part of the [ORG-01 baseline index](../ORG01_CAPABILITY_BASELINE.md).

| ID | Name | Entity Kind | Primary Workspace | Secondary Links | Authority | Lifecycle | Evidence | Acceptance Owner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `analysis.analysis_tools_api` | Analysis Tools REST endpoints (swing metrics, biomechanics) | `feature` | `analysis` | — | `core` | 🔵 planned | `src/api/routes/analysis_tools.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
| `analysis.counterfactuals` | ZTCF/ZVCF + induced-acceleration counterfactuals | `feature` | `analysis` | — | `core` | 🔵 planned | `src/shared/python/biomechanics/ztcf.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
| `analysis.cross_engine_robustness` | Cross-engine robustness dashboard (perturbation/CV) | `feature` | `analysis` | — | `core` | 🟢 active_feature | `src/launchers/cross_engine_dashboard.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
| `analysis.grip_wrench` | Per-hand grip wrench on the club emitted as GRIP overlay frames | `feature` | `analysis` | `simulation` | `core` | 🟢 active_feature | `src/tools/grip_wrench_plots/gui.py` | `tests/unit/grip_overlay/test_grip_glyphs.py` |
| `analysis.ground_reaction` | Ground reaction plots per foot and net | `feature` | `analysis` | `simulation` | `core` | 🟢 active_feature | `src/tools/ground_reaction_plots/gui.py` | `tests/unit/biomechanics/test_ground_reaction_plot_model.py` |
| `analysis.impact_parameters` | Impact parameters panel (target-relative launch-monitor card) | `feature` | `analysis` | `simulation` | `core` | 🟢 active_feature | `src/tools/impact_parameters_panel/gui.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
| `analysis.static_plots` | Static analysis plots (20+ plot types) | `feature` | `analysis` | — | `core` | 🟢 active_feature | `vendor/ud-tools/src/shared/python/plot_engine/pyqt6_widget.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
| `analysis_tools_api` | Analysis Tools | `service` | `analysis` | — | `tools` | 🟢 active_feature | `src/api/routes/analysis_tools.py` | `tests/config/test_launcher_registry_parity.py` |
| `biomech.exercise_injury_dashboards` | Exercise + injury-risk biomechanics dashboards | `feature` | `analysis` | — | `core` | ⚪ exempt | `src/launchers/exercise_dashboard.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
| `data_explorer` | Data Explorer | `tile` | `analysis` | — | `tools` | 🟢 active_feature | `src/config/models.yaml` | `tests/config/test_launcher_registry_parity.py` |
| `data_processor` | Data Processor | `tile` | `analysis` | — | `tools` | 🟢 active_feature | `src/config/models.yaml` | `tests/config/test_launcher_registry_parity.py` |
| `force_overlays` | Force Overlays | `tile` | `analysis` | — | `core` | 🟢 active_feature | `src/config/models.yaml` | `tests/config/test_launcher_registry_parity.py` |
| `grip_wrench_plots` | Grip Wrench Plots | `tile` | `analysis` | `simulation` | `core` | 🟢 active_feature | `src/tools/grip_wrench_plots/gui.py` | `tests/config/test_launcher_registry_parity.py` |
| `ground_reaction_plots` | Ground Reaction Plots | `tile` | `analysis` | `simulation` | `core` | 🟢 active_feature | `src/tools/ground_reaction_plots/gui.py` | `tests/config/test_launcher_registry_parity.py` |
| `impact_parameters` | Impact Parameters | `tile` | `analysis` | `simulation` | `core` | 🟢 active_feature | `src/tools/impact_parameters_panel/gui.py` | `tests/config/test_launcher_registry_parity.py` |
| `injury_analysis` | Injury Risk Analysis | `tile` | `analysis` | — | `core` | 🟢 active_feature | `src/shared/python/injury/injury_risk.py` | `tests/config/test_launcher_registry_parity.py` |
| `launch_monitor_analytics` | Launch Monitor Analytics | `tile` | `analysis` | — | `core` | 🟢 active_feature | `src/tools/launch_monitor_analytics/__main__.py` | `tests/config/test_launcher_registry_parity.py` |
| `launch_monitor_model` | Tools Package: launch_monitor_model | `library` | `analysis` | `simulation` | `core` | 🟣 intentionally_headless | `src/tools/launch_monitor_model/__init__.py` | `tests/config/test_registry_exclusions.py` |
| `lifting.five_lift_viewing` | Weightlifting viewing and analysis for the five lifts | `feature` | `analysis` | — | `core` | 🔵 planned | `src/launchers/exercise_dashboard.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
| `matched_swing_browser` | Matched Swing Results Browser | `tile` | `analysis` | `simulation`, `capture` | `core` | 🟢 active_feature | `src/tools/matched_swing_browser/gui.py` | `tests/config/test_launcher_registry_parity.py` |
| `model_explorer` | Model Explorer | `tile` | `analysis` | — | `core` | 🟢 active_feature | `src/tools/model_explorer/main_window.py` | `tests/config/test_launcher_registry_parity.py` |
| `perturbation_analysis` | Perturbation Analysis | `tile` | `analysis` | — | `core` | 🟢 active_feature | `src/config/models.yaml` | `tests/config/test_launcher_registry_parity.py` |
| `rate_of_closure` | Rate of Closure Impact Explorer | `tile` | `analysis` | — | `tools` | 🟢 active_feature | `src/config/models.yaml` | `tests/config/test_launcher_registry_parity.py` |
| `render.body_appearance` | Golfer body appearance with a visible head, face and neck in every engine render | `feature` | `analysis` | — | `core` | 🔵 planned | `src/shared/python/model_appearance/head.py` | `tests/unit/model_appearance/test_head_model.py` |
| `render.club_head_and_ball` | Realistic Club Head And Ball Rendering | `feature` | `analysis` | — | `core` | 🔵 planned | `src/shared/python/model_appearance/club_head_mesh.py` | `tests/unit/model_appearance/test_club_head_mesh.py` |
