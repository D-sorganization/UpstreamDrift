# Feature Navigation Gap Analysis -- UpstreamDrift

## Executive Summary

The launcher manifest defines 15 tiles, but the codebase contains **27+ implemented features** that users cannot discover or reach from the tiled layout. Three sidebar category buttons (Biomechanics, Simulation, Motion Matching) filter to **zero tiles** -- a broken navigation experience. Several features have complete GUIs and handler registrations but are invisible in the manifest.

This analysis classifies every gap into three tiers:

- **Tile-worthy**: A standalone feature that users would explicitly seek. Needs its own manifest entry.
- **Internal library**: Consumed by other features. Should be documented in capabilities, not given a tile.
- **Borderline**: Has standalone value but could also be a sub-feature. Needs a judgment call.

---

## Current State

### Manifest Tiles (15 visible + 1 hidden)

| #   | Tile ID               | Category       | Route                  |
| --- | --------------------- | -------------- | ---------------------- |
| 1   | model_explorer        | tool           | /tools/model-explorer  |
| 2   | mujoco_unified        | physics_engine | --                     |
| 3   | drake_golf            | physics_engine | --                     |
| 4   | pinocchio_golf        | physics_engine | --                     |
| 5   | opensim_golf          | physics_engine | --                     |
| 6   | myosim_suite          | physics_engine | --                     |
| 7   | putting_green         | physics_engine | /tools/putting-green   |
| 8   | matlab_unified        | external       | --                     |
| 9   | motion_target_preview | tool           | --                     |
| 10  | motion_capture        | tool           | /tools/motion-capture  |
| 11  | video_analyzer        | tool           | /tools/video-analyzer  |
| 12  | video_processor       | tool           | /tools/video-processor |
| 13  | data_explorer         | tool           | /tools/data-explorer   |
| 14  | data_processor        | tool           | /tools/data-processor  |
| 15  | project_map           | tool           | --                     |
| --  | starting_pose_matcher | tool           | HIDDEN (legacy alias)  |

### Sidebar Categories

| Category            | Tile Count | Status                                      |
| ------------------- | ---------- | ------------------------------------------- |
| Home (All)          | 15         | OK                                          |
| Physics Engines     | 5          | OK                                          |
| **Biomechanics**    | **0**      | **EMPTY**                                   |
| **Simulation**      | **0**      | **EMPTY** (Putting Green is physics_engine) |
| **Motion Matching** | **0**      | **EMPTY**                                   |
| Motion Capture      | 1          | OK                                          |
| Tools & Data        | 8          | OK                                          |

### Tauri Routes

/ | /simulation | /tools/model-explorer | /tools/putting-green | /tools/video-analyzer | /tools/data-explorer | /tools/motion-capture | /chat

---

## Issue Index

### P0 -- Broken Navigation (3 issues)

| Issue | File                                | Summary                                            |
| ----- | ----------------------------------- | -------------------------------------------------- |
| #01   | `01_empty_sidebar_categories.md`    | Three sidebar categories show zero tiles           |
| #02   | `02_cross_engine_dashboard_tile.md` | Cross-Engine Dashboard (core feature F6) invisible |
| #03   | `03_exercise_dashboard_tile.md`     | Exercise Dashboard has handler but no tile         |

### P1 -- Core Features Invisible (8 issues)

| Issue | File                                   | Summary                                           |
| ----- | -------------------------------------- | ------------------------------------------------- |
| #04   | `04_shot_tracer_tile.md`               | Shot Tracer GUI unreachable from unified launcher |
| #05   | `05_pose_studio_tile.md`               | Pose Studio standalone GUI completely hidden      |
| #06   | `06_golf_simulation_suite_tile.md`     | Golf Simulation Suite has handler, no tile        |
| #07   | `07_engine_dashboards_unreachable.md`  | MuJoCo/Drake/Pinocchio dashboards unreachable     |
| #08   | `08_swing_optimization_tile.md`        | Swing Optimization (F11) no UI entry              |
| #09   | `09_injury_risk_analysis_tile.md`      | Injury Risk Analysis no UI entry                  |
| #10   | `10_terrain_api_no_tile.md`            | Terrain API (6 endpoints) no tile                 |
| #11   | `11_dataset_generation_no_tile.md`     | Dataset Generation API (9 endpoints) no tile      |
| #12   | `12_motion_matching_category_empty.md` | Motion Matching sidebar has zero tiles            |
| #13   | `13_analysis_tools_api_no_tile.md`     | Analysis Tools API (6 endpoints) no tile          |

### P2 -- Important but Less Urgent (7 issues)

| Issue | File                                       | Summary                                    |
| ----- | ------------------------------------------ | ------------------------------------------ |
| #14   | `14_chat_ai_sidekick_no_manifest_entry.md` | Chat/AI Sidekick no manifest entry         |
| #15   | `15_motion_pipeline_no_tile.md`            | Motion Pipeline only partially covered     |
| #16   | `16_perturbation_analysis_no_ui.md`        | Perturbation/robustness scoring no UI      |
| #17   | `17_character_builder_no_tile.md`          | URDF Character Builder no tile             |
| #18   | `18_pendulum_simulator_tile.md`            | Educational Pendulum Simulator unreachable |
| #30   | `30_putting_green_category_mismatch.md`    | Putting Green miscategorized               |
| #31   | `31_simulation_category_needs_tiles.md`    | Simulation category needs population       |

### P3 -- Niche / Advanced / Internal (7 issues)

| Issue | File                                        | Summary                             |
| ----- | ------------------------------------------- | ----------------------------------- |
| #19   | `19_force_overlays_api_no_tile.md`          | Internal: add to capabilities       |
| #20   | `20_realtime_websocket_no_tile.md`          | Internal: document as capability    |
| #21   | `21_aip_no_tile.md`                         | Internal: surface through Chat tile |
| #22   | `22_actuator_controls_api_no_tile.md`       | Internal: add to capabilities       |
| #23   | `23_bunkershot3d_no_tile.md`                | Tile-worthy but niche               |
| #24   | `24_unreal_integration_no_tile.md`          | Advanced: sub-feature of engines    |
| #25   | `25_robotics_module_no_presence.md`         | Internal: control scheme config     |
| #26   | `26_upstream_drift_tools_breadth_hidden.md` | Expand Data Processor description   |
| #27   | `27_programmatic_pid_hidden.md`             | Niche: may not warrant tile         |
| #28   | `28_dashboard_recorder_hidden.md`           | Internal: sub-feature of dashboards |
| #29   | `29_internal_libraries_no_tile_needed.md`   | Audit: correctly internal           |

---

## Classification: Tile-worthy vs. Internal Library

### Tile-worthy (need manifest entries)

| Feature                        | Proposed Category | Rationale                                 |
| ------------------------------ | ----------------- | ----------------------------------------- |
| Cross-Engine Dashboard         | simulation        | Core feature F6, complete GUI             |
| Exercise Dashboard             | biomechanics      | Complete GUI with handler                 |
| Shot Tracer                    | simulation        | Complete GUI, only in deprecated launcher |
| Pose Studio                    | tool              | Standalone GUI with main()                |
| Golf Simulation Suite          | simulation        | Has handler, no tile                      |
| Swing Optimization             | simulation        | Core feature F11                          |
| Injury Risk Analysis           | biomechanics      | Distinct analysis workflow                |
| Motion Matching (recategorize) | motion_matching   | Sidebar exists, needs tiles               |
| Terrain                        | tool              | 6 API endpoints                           |
| Dataset Generation             | tool              | 9 API endpoints                           |
| BunkerShot3D                   | physics_engine    | Complete simulation system                |
| Pendulum Simulator             | physics_engine    | Educational entry point                   |
| Chat/AI Sidekick               | tool              | Major feature, no manifest entry          |
| Character Builder              | tool              | Distinct workflow from Model Explorer     |

### Internal Library (document in capabilities, no tile)

| Module                | Consumed By               | Action                                 |
| --------------------- | ------------------------- | -------------------------------------- |
| Biomechanics library  | MuJoCo, OpenSim, MyoSuite | Add `biomechanics` capability tag      |
| Screw theory          | Physics engines           | Add `screw_theory` capability tag      |
| Spatial algebra       | Physics engines           | Add `spatial_algebra` capability tag   |
| Data I/O              | Data Explorer, export     | Add `data_io` capability tag           |
| Plot engine           | Dashboards                | Internal, no action needed             |
| Reporting             | Analysis/export           | Internal, no action needed             |
| Realtime transport    | Simulation WS             | Add `realtime` capability tag          |
| Dashboard recorder    | Engine dashboards         | Expose when dashboards are exposed     |
| Robotics control      | Drake, MuJoCo             | Add `control_schemes` capability tag   |
| Rust physics kernels  | Physics engines           | Document in engine descriptions        |
| Actuator controls API | Simulation page           | Add `actuator_controls` capability tag |
| Force overlays API    | Simulation page           | Add `force_overlays` capability tag    |
| AIP                   | Chat/Sidekick             | Add `ai_methods` capability tag        |
| Freemocap ingest      | Motion Capture            | Internal, no action needed             |
| Deployment modules    | Production                | Internal, no action needed             |
| Perturbation analysis | Cross-Engine Dashboard    | Surface in dashboard when exposed      |

---

## Recommended Manifest Changes

### New tiles to add

``json
[
{""id"": ""cross_engine"", ""name"": ""Cross-Engine Dashboard"", ""category"": ""simulation"", ""type"": ""special_app"", ""path"": ""src/launchers/cross_engine_dashboard.py"", ""capabilities"": [""cross_validation"", ""perturbation"", ""robustness_scoring""], ""order"": 20},
{""id"": ""biomech_exercise"", ""name"": ""Exercise Dashboard"", ""category"": ""biomechanics"", ""type"": ""biomech_exercise"", ""path"": ""src/launchers/exercise_dashboard.py"", ""capabilities"": [""injury_risk"", ""joint_stress"", ""swing_modification""], ""order"": 21},
{""id"": ""shot_tracer"", ""name"": ""Shot Tracer"", ""category"": ""simulation"", ""type"": ""special_app"", ""path"": ""src/launchers/\_shot_tracer_gui.py"", ""capabilities"": [""trajectory_visualization"", ""multi_model_comparison""], ""order"": 22},
{""id"": ""pose_studio"", ""name"": ""Pose Studio"", ""category"": ""tool"", ""type"": ""special_app"", ""path"": ""src/tools/pose_studio/**main**.py"", ""capabilities"": [""pose_editing"", ""retargeting""], ""order"": 23},
{""id"": ""golf_simulation_suite"", ""name"": ""Golf Simulation Suite"", ""category"": ""simulation"", ""type"": ""golf_simulation"", ""path"": ""src/tools/golf_simulation_suite/**main**.py"", ""capabilities"": [""full_simulation"", ""parameter_sweep""], ""order"": 24},
{""id"": ""swing_optimization"", ""name"": ""Swing Optimizer"", ""category"": ""simulation"", ""type"": ""special_app"", ""path"": ""src/shared/python/optimization/swing_optimizer.py"", ""capabilities"": [""trajectory_optimization"", ""constraint_solving""], ""order"": 25},
{""id"": ""injury_analysis"", ""name"": ""Injury Risk Analysis"", ""category"": ""biomechanics"", ""type"": ""special_app"", ""path"": ""src/shared/python/injury/injury_risk.py"", ""capabilities"": [""injury_risk"", ""joint_stress"", ""spinal_load""], ""order"": 26},
{""id"": ""terrain"", ""name"": ""Terrain Engine"", ""category"": ""tool"", ""type"": ""special_app"", ""path"": """", ""web_route"": ""/tools/terrain"", ""capabilities"": [""surface_modeling"", ""ball_physics"", ""topography""], ""order"": 27},
{""id"": ""dataset_generator"", ""name"": ""Dataset Generator"", ""category"": ""tool"", ""type"": ""special_app"", ""path"": """", ""web_route"": ""/tools/dataset"", ""capabilities"": [""dataset_generation"", ""swing_import"", ""parameter_sweep""], ""order"": 28},
{""id"": ""bunkershot3d"", ""name"": ""BunkerShot3D"", ""category"": ""physics_engine"", ""type"": ""special_app"", ""path"": ""src/bunkershot3d/"", ""capabilities"": [""sand_simulation"", ""mpm"", ""calibration""], ""order"": 29},
{""id"": ""pendulum"", ""name"": ""Pendulum Simulator"", ""category"": ""physics_engine"", ""type"": ""special_app"", ""path"": ""src/shared/python/pendulum_simulator/**main**.py"", ""capabilities"": [""educational"", ""2dof""], ""order"": 30},
{""id"": ""chat"", ""name"": ""AI Assistant"", ""category"": ""tool"", ""type"": ""special_app"", ""path"": """", ""web_route"": ""/chat"", ""capabilities"": [""ai_methods"", ""calculator"", ""workspace""], ""order"": 31},
{""id"": ""character_builder"", ""name"": ""Character Builder"", ""category"": ""tool"", ""type"": ""special_app"", ""path"": ""src/shared/python/model_generation/cli/main.py"", ""capabilities"": [""urdf_generation"", ""anthropometry"", ""mesh_generation""], ""order"": 32}
]

```

### Category changes to existing tiles

| Tile ID | Current Category | Proposed Category |
|---------|------------------|--------------------|
| putting_green | physics_engine | **simulation** |
| motion_target_preview | tool | **motion_matching** |

### Capability tag additions to existing tiles

| Tile ID | Capabilities to Add |
|---------|-------------------|
| mujoco_unified | biomechanics, screw_theory, spatial_algebra, actuator_controls, force_overlays, control_schemes, realtime |
| drake_golf | actuator_controls, force_overlays, control_schemes, realtime |
| pinocchio_golf | actuator_controls, force_overlays |
| opensim_golf | biomechanics, actuator_controls |
| data_processor | process_calculators |

---

## Files

All issue files are in: `reports/feature_navigation_gaps/`
```
