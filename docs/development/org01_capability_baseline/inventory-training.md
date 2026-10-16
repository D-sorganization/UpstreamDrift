# Capability Baseline Inventory: Training

<!-- Generated: python -m scripts.generate_capability_baseline -->

Part of the [ORG-01 baseline index](../ORG01_CAPABILITY_BASELINE.md).

| ID | Name | Entity Kind | Primary Workspace | Secondary Links | Authority | Lifecycle | Evidence | Acceptance Owner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `character_builder` | Character Builder | `cli_tool` | `training` | — | `core` | 🟢 active_feature | `src/shared/python/model_generation/cli/main.py` | `tests/config/test_launcher_registry_parity.py` |
| `dataset_generator` | Dataset Generator | `cli_tool` | `training` | `analysis` | `core` | 🟢 active_feature | `src/config/models.yaml` | `tests/config/test_launcher_registry_parity.py` |
| `motion_matching` | Motion Matching | `tile` | `training` | — | `core` | 🟢 active_feature | `src/tools/motion_matching/__main__.py` | `tests/config/test_launcher_registry_parity.py` |
| `motion_pipeline` | Motion Pipeline | `service` | `training` | — | `core` | 🟢 active_feature | `src/shared/python/motion_pipeline/api.py` | `tests/config/test_launcher_registry_parity.py` |
| `movement_optimizer` | Movement Optimizer | `tile` | `training` | — | `movement_optimizer` | 🟢 active_feature | `src/movement_optimizer/__main__.py` | `tests/config/test_launcher_registry_parity.py` |
| `optimization.swing_optimizer` | Swing Optimizer (trajectory optimization GUI) | `feature` | `training` | — | `core` | ⚪ exempt | `src/shared/python/optimization/swing_optimizer.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
| `pid_generator` | P&ID Generator | `cli_tool` | `training` | — | `tools` | 🟢 active_feature | `src/config/models.yaml` | `tests/config/test_launcher_registry_parity.py` |
| `sg_optimizer` | Tools Package: sg_optimizer | `cli_tool` | `training` | `simulation` | `core` | 🔵 planned | `src/tools/sg_optimizer/__init__.py` | `tests/config/test_registry_exclusions.py` |
| `simulation.swing_objective_lab` | Swing Objective Lab — mechanism-vs-outcome downswing comparison | `feature` | `training` | — | `core` | 🟢 active_feature | `src/launchers/adapters/swing_objective_lab_embed.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
| `swing_flight_pipeline` | Swing → Flight Pipeline | `service` | `training` | `simulation` | `core` | 🟢 active_feature | `src/tools/swing_flight_pipeline/gui.py` | `tests/config/test_launcher_registry_parity.py` |
| `swing_objective_lab` | Swing Objective Lab | `tile` | `training` | `simulation` | `tools` | 🟢 active_feature | `src/config/models.yaml` | `tests/config/test_launcher_registry_parity.py` |
| `swing_optimizer` | Swing Optimizer | `tile` | `training` | `simulation` | `core` | 🟢 active_feature | `src/shared/python/optimization/swing_optimizer.py` | `tests/config/test_launcher_registry_parity.py` |
| `tools.character_builder` | Character Builder (humanoid URDF generation) | `feature` | `training` | — | `core` | 🔵 planned | `src/shared/python/model_generation/cli/main.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
| `tools.dataset_generator` | Swing dataset generation and import | `feature` | `training` | — | `core` | 🟢 active_feature | `src/api/routes/dataset.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
| `tools_movement_optimizer` | Movement Optimizer | `tile` | `training` | — | `tools` | 🟢 active_feature | `src/movement_optimizer/launch_pyqt6.py` | `tests/config/test_launcher_registry_parity.py` |
| `tour_matching_viewer` | Tour Matching Viewer | `tile` | `training` | — | `core` | 🟠 confirmed_prototype | `src/tools/tour_matching_viewer/__main__.py` | `tests/config/test_launcher_registry_parity.py` |
| `training_controller` | Training Controller | `tile` | `training` | — | `core` | 🟢 active_feature | `src/tools/training_controller/__main__.py` | `tests/config/test_launcher_registry_parity.py` |
