# Capability Baseline Inventory: Putting

<!-- Generated: python -m scripts.generate_capability_baseline -->

Part of the [ORG-01 baseline index](../ORG01_CAPABILITY_BASELINE.md).

| ID | Name | Entity Kind | Primary Workspace | Secondary Links | Authority | Lifecycle | Evidence | Acceptance Owner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `bunkershot3d` | BunkerShot3D Designer Workbench | `tile` | `putting` | — | `core` | 🟢 active_feature | `src/tools/bunker_shot_gui/gui.py` | `tests/config/test_launcher_registry_parity.py` |
| `putting_green` | Putting Green | `tile` | `putting` | — | `core` | 🟢 active_feature | `src/tools/putting_green_gui/gui.py` | `tests/config/test_launcher_registry_parity.py` |
| `putting_green_gui` | Putting Green | `tile` | `putting` | — | `core` | 🟡 deprecated_alias | `src/tools/putting_green_gui/gui.py` | `tests/config/test_capability_migration_coverage.py` |
| `tools.bunkershot3d_workbench` | BunkerShot3D designer workbench (W2 sole parameters, W3 sand condition, F0 dynamic-RFT shot, W7 metrics, playability window, bounce utilisation, animated sole load field, 3-D shot animation through the ADR-0027 viewport, linked scalar traces with a validity band, F1 sand-field cross-sections, A/B comparison, validity verdict) | `feature` | `putting` | — | `core` | 🔵 planned | `src/tools/bunker_shot_gui/gui.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
| `tools.putting_green` | Putting green simulation | `feature` | `putting` | — | `core` | 🟢 active_feature | `src/engines/physics_engines/putting_green/python/simulator.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
| `tools.rate_of_closure` | Rate of Closure Impact Explorer (swing-impact-flight-putting simulation suite) | `feature` | `putting` | — | `core` | 🟢 active_feature | `vendor/ud-tools/src/rate_of_closure/launch_pyqt6.py` | `tests/config/feature_parity/test_feature_parity_registry.py` |
