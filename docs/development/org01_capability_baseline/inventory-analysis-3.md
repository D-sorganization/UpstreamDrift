# Capability Baseline Inventory: Analysis (Part 3/3)

<!-- Generated: python -m scripts.generate_capability_baseline -->

Part of the [ORG-01 baseline index](../ORG01_CAPABILITY_BASELINE.md).

| ID | Name | Entity Kind | Primary Workspace | Secondary Links | Authority | Lifecycle | Evidence | Acceptance Owner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `tools.spine_study` | Segmental spine study execution and model asset selection | `feature` | `analysis` | — | `core` | 🟢 active_feature | `src/api/routes/spine_viewer.py` | `tests/api/test_spine_viewer_route.py` |
| `tools.spine_viewer` | Segmental spine viewer | `feature` | `analysis` | — | `core` | 🟢 active_feature | `src/tools/spine_viewer/gui.py` | `tests/tools/spine_viewer/test_core.py` |
| `tools.swing_report_package` | Swing report package: one command regenerates a swing's videos, grids and phase sheets (CLI) | `feature` | `analysis` | `simulation` | `core` | 🟢 active_feature | `src/tools/swing_report_package/cli.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
| `tools_calculator_hub` | Tools Calculator Suite (Data Processor Entry) | `cli_tool` | `analysis` | — | `tools` | 🟢 active_feature | `src/config/models.yaml` | `tests/config/test_launcher_registry_parity.py` |
| `tools_video_analyzer` | Tools Package: video_analyzer | `library` | `analysis` | `simulation` | `core` | 🟣 intentionally_headless | `src/tools/video_analyzer/__init__.py` | `tests/config/test_registry_exclusions.py` |
