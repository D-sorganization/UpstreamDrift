# Capability Baseline Inventory: Simulation (Part 3/4)

<!-- Generated: python -m scripts.generate_capability_baseline -->

Part of the [ORG-01 baseline index](../ORG01_CAPABILITY_BASELINE.md).

| ID | Name | Entity Kind | Primary Workspace | Secondary Links | Authority | Lifecycle | Evidence | Acceptance Owner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `offline_validation` | Tools Package: offline_validation | `library` | `simulation` | `analysis` | `core` | 🟣 intentionally_headless | `src/tools/offline_validation/__init__.py` | `tests/config/test_registry_exclusions.py` |
| `onboarding.about_version` | About/version info + onboarding | `feature` | `simulation` | — | `core` | 🔵 planned | `src/launchers/about_dialog.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
| `opensim_golf` | OpenSim | `model` | `simulation` | — | `opensim` | 🟢 active_feature | `src/engines/physics_engines/opensim/python/opensim_gui.py` | `tests/config/test_launcher_registry_parity.py` |
| `opensim_models-bench_press` | Bench Press | `model` | `simulation` | — | `opensim` | 🟢 active_feature | `src/opensim_models/exercises/bench_press` | `tests/config/test_launcher_registry_parity.py` |
| `opensim_models-clean_and_jerk` | Clean And Jerk | `model` | `simulation` | — | `opensim` | 🟢 active_feature | `src/opensim_models/exercises/clean_and_jerk` | `tests/config/test_launcher_registry_parity.py` |
| `opensim_models-deadlift` | Deadlift | `model` | `simulation` | — | `opensim` | 🟢 active_feature | `src/opensim_models/exercises/deadlift` | `tests/config/test_launcher_registry_parity.py` |
| `opensim_models-gait` | Gait | `model` | `simulation` | — | `opensim` | 🟢 active_feature | `src/opensim_models/exercises/gait` | `tests/config/test_launcher_registry_parity.py` |
| `opensim_models-sit_to_stand` | Sit To Stand | `model` | `simulation` | — | `opensim` | 🟢 active_feature | `src/opensim_models/exercises/sit_to_stand` | `tests/config/test_launcher_registry_parity.py` |
| `opensim_models-snatch` | Snatch | `model` | `simulation` | — | `opensim` | 🟢 active_feature | `src/opensim_models/exercises/snatch` | `tests/config/test_launcher_registry_parity.py` |
| `opensim_models-squat` | Squat | `model` | `simulation` | — | `opensim` | 🟢 active_feature | `src/opensim_models/exercises/squat` | `tests/config/test_launcher_registry_parity.py` |
| `opensim_models_shared` | OpenSim Models | `model` | `simulation` | — | `opensim` | 🟢 active_feature | `OpenSim_Models` | `tests/config/test_launcher_registry_parity.py` |
| `pendulum_simulator` | Pendulum Simulator | `tile` | `simulation` | — | `tools` | 🟢 active_feature | `src/config/models.yaml` | `tests/config/test_launcher_registry_parity.py` |
| `pinn_hybrid` | PINN Hybrid | `tile` | `simulation` | — | `core` | 🟠 confirmed_prototype | `virtual/physics_informed/pinn_hybrid` | `tests/config/test_launcher_registry_parity.py` |
| `pinn_pure_rigid` | Pure Rigid Body | `tile` | `simulation` | — | `core` | 🟠 confirmed_prototype | `virtual/physics_informed/pinn_pure_rigid` | `tests/config/test_launcher_registry_parity.py` |
| `pinocchio_dashboard` | Pinocchio Dashboard | `model` | `simulation` | — | `pinocchio` | 🟢 active_feature | `src/launchers/pinocchio_dashboard.py` | `tests/config/test_launcher_registry_parity.py` |
| `pinocchio_golf` | Pinocchio | `model` | `simulation` | — | `pinocchio` | 🟢 active_feature | `src/engines/physics_engines/pinocchio/python/pinocchio_golf/gui.py` | `tests/config/test_launcher_registry_parity.py` |
| `pinocchio_models-bench_press` | Bench Press | `model` | `simulation` | — | `pinocchio` | 🟢 active_feature | `src/pinocchio_models/exercises/bench_press` | `tests/config/test_launcher_registry_parity.py` |
| `pinocchio_models-clean_and_jerk` | Clean And Jerk | `model` | `simulation` | — | `pinocchio` | 🟢 active_feature | `src/pinocchio_models/exercises/clean_and_jerk` | `tests/config/test_launcher_registry_parity.py` |
| `pinocchio_models-deadlift` | Deadlift | `model` | `simulation` | — | `pinocchio` | 🟢 active_feature | `src/pinocchio_models/exercises/deadlift` | `tests/config/test_launcher_registry_parity.py` |
| `pinocchio_models-gait` | Gait | `model` | `simulation` | — | `pinocchio` | 🟢 active_feature | `src/pinocchio_models/exercises/gait` | `tests/config/test_launcher_registry_parity.py` |
| `pinocchio_models-sit_to_stand` | Sit To Stand | `model` | `simulation` | — | `pinocchio` | 🟢 active_feature | `src/pinocchio_models/exercises/sit_to_stand` | `tests/config/test_launcher_registry_parity.py` |
| `pinocchio_models-snatch` | Snatch | `model` | `simulation` | — | `pinocchio` | 🟢 active_feature | `src/pinocchio_models/exercises/snatch` | `tests/config/test_launcher_registry_parity.py` |
| `pinocchio_models-squat` | Squat | `model` | `simulation` | — | `pinocchio` | 🟢 active_feature | `src/pinocchio_models/exercises/squat` | `tests/config/test_launcher_registry_parity.py` |
| `pinocchio_models_shared` | Pinocchio Models | `model` | `simulation` | — | `pinocchio` | 🟢 active_feature | `Pinocchio_Models` | `tests/config/test_launcher_registry_parity.py` |
| `realtime_ws` | Realtime WebSocket | `service` | `simulation` | — | `core` | 🟢 active_feature | `src/config/models.yaml` | `tests/config/test_launcher_registry_parity.py` |
