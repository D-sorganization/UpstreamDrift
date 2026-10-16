# ORG-01: Baseline Capability Inventory & Entity Identity Preservation

<!-- Generated: python -m scripts.generate_capability_baseline -->

## Executive Summary

This document establishes the canonical baseline capability inventory for
**Epic #10508 (Integrated Workspace Reorganization)** and enforces entity identity
preservation across launcher tiles, provider models, parity features, excluded tool
packages, saved layouts, and golden test fixtures.

### Architectural Rulings and Boundaries
- **Explicit Identity Contracts**: Capability IDs are unique; every capability has exactly one primary workspace domain; secondary links are non-exclusive cross-domain associations.
- **Acyclic Alias Resolution**: Deprecated aliases resolve transitively and acyclically to retained active targets (`starting_pose_matcher` → `motion_target_preview`, `putting_green_gui` → `putting_green`).
- **Decoupled Provider Availability**: The absence of an external provider (MuJoCo, Drake, Pinocchio, OpenSim) alters runtime operational availability, never entity identity or workspace ownership.
- **Preservation of ADR-0047 Viewer Identity**: Preserves separate viewer identities (`shot_tracer`, `BallFlight`, `Impact Explorer`) communicating via shared trajectory record contracts without merging or destroying viewer tools.
- **Preservation of User Artifacts**: Golden fixtures and user-owned scripts are strictly preserved; no destructive file removals.

## Baseline Metrics

- **Total Cataloged Entries**: 184
- **Observed Launcher Tiles / Models**: 104 (61 base desktop models + 29 discovered provider models + 14 web catalog tiles)
- **Feature Parity Contracts**: 45
- **Excluded Tool Packages / Libraries**: 9
- **Preserved Golden Fixtures**: 15

### Workspace Distribution

| Workspace | Primary Capability Count | Description |
| :--- | :---: | :--- |
| `analysis` | 39 | Telemetry extraction, metric calculation, video/data analysis, and flight comparison |
| `capture` | 21 | Multi-camera mocap, pose estimation, marker tracking, and 3D reconstruction |
| `governance` | 13 | Configuration setup, project architecture mapping, sidekick docks, and registry admin |
| `putting` | 6 | Putting physics, green surface simulation, and ball rolling dynamics |
| `simulation` | 88 | Physics engines, multi-body kinematics, dynamics solvers, and forward simulation |
| `training` | 17 | Drills, objective laboratories, movement/swing optimization, and skill reinforcement |

## Representative Golden Preservation Fixtures

| Fixture Key | Kind | Relative Path | Size (bytes) | SHA-256 Hash | Support Status |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `capture.multi_camera_observations` | `capture` | `tests/fixtures/pose_estimation/canonical_observations_multi_camera.json` | 2431 | `2eca60f355db9a62...` | ✅ supported |
| `export.companion_current` | `export` | `tests/fixtures/companion/current-v1.0.0.json` | 1775 | `99b95f45494ba3ec...` | ✅ supported |
| `export.control_names_matlab` | `export` | `tests/fixtures/control_names_matlab.json` | 4248 | `94386ca037fe5dc5...` | ✅ supported |
| `flight.adr0046_cross_stack_session` | `flight` | `tests/fixtures/launch_monitor/adr0046_cross_stack_session_v1.json` | 83536 | `6328bd22493baaf2...` | ✅ supported |
| `flight.ball_flight_trajectory` | `flight` | `tests/parity_fixtures/ball_flight/default_trajectory.json` | 1228 | `cbbc7395e9750e7f...` | ✅ supported |
| `flight.flightscope_csv` | `flight` | `tests/fixtures/launch_monitor/flightscope.csv` | 298 | `ec83f5700f7ba706...` | ✅ supported |
| `flight.trackman_csv` | `flight` | `tests/fixtures/launch_monitor/trackman.csv` | 450 | `e8e344c9b0c08f83...` | ✅ supported |
| `model.double_pendulum_urdf` | `model` | `tests/fixtures/models/double_pendulum.urdf` | 4647 | `807f40e3feaf5955...` | ✅ supported |
| `model.simple_pendulum_urdf` | `model` | `tests/fixtures/models/simple_pendulum.urdf` | 4100 | `91b7eae580ba15eb...` | ✅ supported |
| `pose.simscape_offsets` | `pose` | `tests/fixtures/pose_io/simscape_offsets_golden.json` | 850 | `df3a27febfbeb182...` | ✅ supported |
| `pose.test_poses` | `pose` | `tests/fixtures/test_poses.json` | 1347 | `c8fb8ed96ecde54d...` | ✅ supported |
| `replay.opensim_reference` | `replay` | `tests/fixtures/opensim_simscape_equivalence/simscape_reference.npz` | 9406 | `097116cfa8dc3917...` | ✅ supported |
| `replay.simscape_baseline` | `replay` | `tests/fixtures/cross_engine_parity/simscape_canonical_baseline.npz` | 24392 | `143ad2d2b282f3b6...` | ✅ supported |
| `unsupported.companion_future_v2` | `unsupported` | `tests/fixtures/companion/rejected-future-v2.0.0.json` | 165 | `5261b978d2309d69...` | ❌ unsupported |
| `unsupported.companion_incompatible_v1` | `unsupported` | `tests/fixtures/companion/rejected-incompatible-v1.0.0.json` | 187 | `06f2dff60e5c1f01...` | ❌ unsupported |

## Legacy Alias Resolution Graph

| Deprecated Alias ID | Canonical Target ID | Lifecycle | Evidence | Acceptance Owner |
| :--- | :--- | :--- | :--- | :--- |
| `putting_green_gui` | `putting_green` | 🟡 deprecated_alias | `src/tools/putting_green_gui/gui.py` | `tests/config/test_capability_migration_coverage.py` |
| `starting_pose_matcher` | `motion_target_preview` | 🟡 deprecated_alias | `src/tools/starting_pose_matcher/__main__.py` | `tests/config/test_capability_migration_coverage.py` |

## Complete Capability Baseline Inventory

The full inventory table is split into linked parts by primary workspace.

| Part | Entries | First ID | Last ID |
| :--- | :---: | :--- | :--- |
| [Capability Baseline Inventory: Analysis (Part 1/2)](org01_capability_baseline/inventory-analysis-1.md) | 25 | `analysis.analysis_tools_api` | `render.club_head_and_ball` |
| [Capability Baseline Inventory: Analysis (Part 2/2)](org01_capability_baseline/inventory-analysis-2.md) | 14 | `reporting.swing_report_dialog` | `tools_video_analyzer` |
| [Capability Baseline Inventory: Capture](org01_capability_baseline/inventory-capture.md) | 21 | `c3d_viewer` | `video_processor` |
| [Capability Baseline Inventory: Governance](org01_capability_baseline/inventory-governance.md) | 13 | `aip` | `tools.terrain_engine` |
| [Capability Baseline Inventory: Putting](org01_capability_baseline/inventory-putting.md) | 6 | `bunkershot3d` | `tools.rate_of_closure` |
| [Capability Baseline Inventory: Simulation (Part 1/4)](org01_capability_baseline/inventory-simulation-1.md) | 25 | `actuator_controls` | `engines.load_and_simulate` |
| [Capability Baseline Inventory: Simulation (Part 2/4)](org01_capability_baseline/inventory-simulation-2.md) | 25 | `export.video_speed_variants` | `native_viewer_export` |
| [Capability Baseline Inventory: Simulation (Part 3/4)](org01_capability_baseline/inventory-simulation-3.md) | 25 | `offline_validation` | `realtime_ws` |
| [Capability Baseline Inventory: Simulation (Part 4/4)](org01_capability_baseline/inventory-simulation-4.md) | 13 | `robotics_module` | `visualization.force_color_controls` |
| [Capability Baseline Inventory: Training](org01_capability_baseline/inventory-training.md) | 17 | `character_builder` | `training_controller` |
