# Capability Baseline Inventory: Analysis (Part 3/3)

<!-- Generated: python -m scripts.generate_capability_baseline -->

Part of the [ORG-01 baseline index](../ORG01_CAPABILITY_BASELINE.md).

| ID | Name | Entity Kind | Primary Workspace | Secondary Links | Authority | Lifecycle | Evidence | Acceptance Owner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `tools.spine_viewer` | Segmental spine viewer | `feature` | `analysis` | — | `core` | 🟢 active_feature | `src/tools/spine_viewer/gui.py` | `tests/tools/spine_viewer/test_core.py` |
| `tools_calculator_hub` | Tools Calculator Suite (Data Processor Entry) | `cli_tool` | `analysis` | — | `tools` | 🟢 active_feature | `src/config/models.yaml` | `tests/config/test_launcher_registry_parity.py` |
| `tools_video_analyzer` | Tools Package: video_analyzer | `library` | `analysis` | `simulation` | `core` | 🟣 intentionally_headless | `src/tools/video_analyzer/__init__.py` | `tests/config/test_registry_exclusions.py` |
