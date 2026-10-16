# Capability Baseline Inventory: Simulation (Part 1/4)

<!-- Generated: python -m scripts.generate_capability_baseline -->

Part of the [ORG-01 baseline index](../ORG01_CAPABILITY_BASELINE.md).

| ID | Name | Entity Kind | Primary Workspace | Secondary Links | Authority | Lifecycle | Evidence | Acceptance Owner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `actuator_controls` | Actuator Controls | `tile` | `simulation` | — | `core` | 🟢 active_feature | `src/config/models.yaml` | `tests/config/test_launcher_registry_parity.py` |
| `ball_flight_simulator` | Ball Flight Simulator | `tile` | `simulation` | — | `core` | 🟢 active_feature | `src/tools/ball_flight_gui/gui.py` | `tests/config/test_launcher_registry_parity.py` |
| `biomech_exercise` | Exercise Dashboard | `tile` | `simulation` | — | `core` | 🟢 active_feature | `src/launchers/exercise_dashboard.py` | `tests/config/test_launcher_registry_parity.py` |
| `biomech_gait` | Gait Model | `tile` | `simulation` | — | `core` | 🟢 active_feature | `virtual/biomech_exercise/biomech_gait` | `tests/config/test_launcher_registry_parity.py` |
| `biomech_sit_to_stand` | Sit-to-Stand Model | `tile` | `simulation` | — | `core` | 🟢 active_feature | `virtual/biomech_exercise/biomech_sit_to_stand` | `tests/config/test_launcher_registry_parity.py` |
| `canonical_core.workspaces` | Canonical-core estimation/comparison workspaces | `feature` | `simulation` | — | `core` | 🟢 active_feature | `src/tools/canonical_core/estimation.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
| `canonical_core_comparison` | Canonical-Core Comparison | `tile` | `simulation` | — | `core` | 🟢 active_feature | `src/tools/canonical_core/comparison.py` | `tests/config/test_launcher_registry_parity.py` |
| `canonical_core_estimation` | Canonical-Core Estimation | `tile` | `simulation` | — | `core` | 🟢 active_feature | `src/tools/canonical_core/estimation.py` | `tests/config/test_launcher_registry_parity.py` |
| `contraction` | Tools Package: contraction | `cli_tool` | `simulation` | `analysis` | `core` | 🟣 intentionally_headless | `src/tools/contraction/__init__.py` | `tests/config/test_registry_exclusions.py` |
| `cross_engine_dashboard` | Cross-Engine Dashboard | `tile` | `simulation` | — | `core` | 🟢 active_feature | `src/launchers/cross_engine_dashboard.py` | `tests/config/test_launcher_registry_parity.py` |
| `diagnostics.integrations_health` | Diagnostics + integrations-health panel | `feature` | `simulation` | — | `core` | 🟢 active_feature | `src/launchers/integrations_health_panel.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
| `drake.gui_force_overlay` | Drake GUI force/torque arrows and live tension/compression shading | `feature` | `simulation` | — | `core` | 🔵 planned | `src/engines/physics_engines/drake/python/src/drake_force_overlay.py` | `tests/unit/engines/drake/test_drake_gui_force_overlay.py` |
| `drake_dashboard` | Drake Dashboard | `model` | `simulation` | — | `drake` | 🟢 active_feature | `src/launchers/drake_dashboard.py` | `tests/config/test_launcher_registry_parity.py` |
| `drake_golf` | Drake | `model` | `simulation` | — | `drake` | 🟢 active_feature | `src/engines/physics_engines/drake/python/src/drake_gui_app.py` | `tests/config/test_launcher_registry_parity.py` |
| `drake_models-bench_press` | Bench Press | `model` | `simulation` | — | `drake` | 🟢 active_feature | `src/drake_models/exercises/bench_press` | `tests/config/test_launcher_registry_parity.py` |
| `drake_models-clean_and_jerk` | Clean And Jerk | `model` | `simulation` | — | `drake` | 🟢 active_feature | `src/drake_models/exercises/clean_and_jerk` | `tests/config/test_launcher_registry_parity.py` |
| `drake_models-deadlift` | Deadlift | `model` | `simulation` | — | `drake` | 🟢 active_feature | `src/drake_models/exercises/deadlift` | `tests/config/test_launcher_registry_parity.py` |
| `drake_models-gait` | Gait | `model` | `simulation` | — | `drake` | 🟢 active_feature | `src/drake_models/exercises/gait` | `tests/config/test_launcher_registry_parity.py` |
| `drake_models-sit_to_stand` | Sit To Stand | `model` | `simulation` | — | `drake` | 🟢 active_feature | `src/drake_models/exercises/sit_to_stand` | `tests/config/test_launcher_registry_parity.py` |
| `drake_models-snatch` | Snatch | `model` | `simulation` | — | `drake` | 🟢 active_feature | `src/drake_models/exercises/snatch` | `tests/config/test_launcher_registry_parity.py` |
| `drake_models-squat` | Squat | `model` | `simulation` | — | `drake` | 🟢 active_feature | `src/drake_models/exercises/squat` | `tests/config/test_launcher_registry_parity.py` |
| `drake_models_shared` | Drake Models | `model` | `simulation` | — | `drake` | 🟢 active_feature | `Drake_Models` | `tests/config/test_launcher_registry_parity.py` |
| `drift_control` | Tools Package: drift_control | `cli_tool` | `simulation` | `analysis` | `core` | 🟣 intentionally_headless | `src/tools/drift_control/__init__.py` | `tests/config/test_registry_exclusions.py` |
| `engines.dashboards` | Per-engine interactive dashboards (Drake/MuJoCo/Pinocchio) | `feature` | `simulation` | — | `core` | ⚪ exempt | `src/launchers/drake_dashboard.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
| `engines.load_and_simulate` | Engine load/probe + basic simulation loop | `feature` | `simulation` | — | `core` | 🟢 active_feature | `src/launchers/launcher_simulation.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
