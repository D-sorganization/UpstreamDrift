# Feature Parity Matrix (PyQt6 ↔ Tauri/React)

<!-- AUTO-GENERATED — do not edit by hand. -->
<!-- Regenerate with: python -m scripts.generate_feature_parity_matrix -->

Generated from [`src/config/feature_parity.json`](../../src/config/feature_parity.json) (registry v1.0.0).
The PyQt6 desktop app is the canonical model; the web app must match
(epic #7462, registry mechanism #7445).

**Summary:** 16 parity · 11 gap · 13 exempt · 2 api_only (11 pending decision in #7460).

| Feature | Status | PyQt6 | API | Web | Tracking |
| --- | --- | --- | --- | --- | --- |
| `analysis.analysis_tools_api`<br>Analysis Tools REST endpoints (swing metrics, biomechanics) | 🔴 gap | — | `src/api/routes/analysis_tools.py` | `ui/src/pages/AnalysisTools.tsx` | #7448 |
| `analysis.counterfactuals`<br>ZTCF/ZVCF + induced-acceleration counterfactuals | 🔴 gap | `src/shared/python/biomechanics/ztcf.py` | — | — | #7450 |
| `analysis.cross_engine_robustness`<br>Cross-engine robustness dashboard (perturbation/CV) | ✅ parity | `src/launchers/cross_engine_dashboard.py` | `src/api/routes/cross_engine.py` | `ui/src/pages/CrossEngineDashboard.tsx` | — |
| `analysis.static_plots`<br>Static analysis plots (20+ plot types) | ✅ parity | `src/shared/python/plot_engine/pyqt6_widget.py` | `src/api/routes/analysis_plots.py` | `ui/src/components/analysis/PlotsSection.tsx` | — |
| `biomech.exercise_injury_dashboards`<br>Exercise + injury-risk biomechanics dashboards | ⚪ exempt | `src/launchers/exercise_dashboard.py` | — | — | Desktop biomechanics dashboards; desktop-only candidate pending #7460. — **pending decision (#7460)** |
| `canonical_core.workspaces`<br>Canonical-core estimation/comparison workspaces | ✅ parity | `src/tools/canonical_core/estimation.py` | — | `ui/src/pages/CanonicalCoreShell.tsx` | — |
| `chat.live_context`<br>Live app/engine context in chat | ✅ parity | `src/launchers/launcher_sidekick_sidebar.py` | `src/api/services/chat_app_context.py` | `ui/src/components/ui/ChatContextChip.tsx` | — |
| `chat.transport`<br>AI chat transport (message send/stream) | ✅ parity | `src/launchers/launcher_sidekick_sidebar.py` | `src/api/routes/chat_ws.py` | `ui/src/pages/Chat.tsx` | — |
| `diagnostics.integrations_health`<br>Diagnostics + integrations-health panel | ✅ parity | `src/launchers/integrations_health_panel.py` | `src/api/routes/diagnostics.py` | `ui/src/components/ui/DiagnosticsPanel.tsx` | — |
| `docs.document_library`<br>Document library / project map viewer | ⚪ exempt | `src/launchers/library_widget.py` | — | — | Desktop documentation browser; desktop-only candidate pending #7460. — **pending decision (#7460)** |
| `engines.dashboards`<br>Per-engine interactive dashboards (Drake/MuJoCo/Pinocchio) | ⚪ exempt | `src/launchers/drake_dashboard.py` | — | — | Experimental desktop dashboards; desktop-only candidate pending #7460. — **pending decision (#7460)** |
| `engines.load_and_simulate`<br>Engine load/probe + basic simulation loop | ✅ parity | `src/launchers/launcher_simulation.py` | `src/api/routes/engines.py` | `ui/src/pages/Simulation.tsx` | — |
| `export.recordings_downloads`<br>Export/recording parity (HDF5/MAT/C3D/CSV/video, persisted recordings) | 🔴 gap | `src/shared/python/data_io/export.py` | `src/api/routes/export.py` | — | #7451 |
| `launcher.docker_management`<br>Docker engine management dialog | ⚪ exempt | `src/launchers/docker_manager.py` | — | — | Manages the local Docker daemon from the desktop; desktop-only candidate pending #7460. — **pending decision (#7460)** |
| `launcher.embedded_tabs_docks`<br>Embedded tool host (tabs + docks) | ⚪ exempt | `src/launchers/embedded_host.py` | — | — | Desktop windowing composition is explicitly desktop-only per ADR-0028 (React shell uses routes, not embedded Qt docks). |
| `launcher.mcp_config`<br>MCP server configuration writer/preferences | ⚪ exempt | `src/launchers/mcp_config_writer.py` | — | — | Writes local MCP configuration files for desktop AI integrations; desktop-only candidate pending #7460. — **pending decision (#7460)** |
| `launcher.model_repositories`<br>Model repository folder shortcuts | ⚪ exempt | `src/launchers/launcher_model_handlers.py` | — | — | Desktop-only folder shortcuts into the vendored model repositories; the web app browses models through Model Explorer. |
| `launcher.tile_grid`<br>Launcher tile grid from shared manifest | ✅ parity | `src/launchers/embedded_tool_bootstrap.py` | `src/api/routes/launcher.py` | `ui/src/pages/Dashboard.tsx` | — |
| `launcher.tile_web_reachability`<br>Manifest tile web-reachability contract (route / native-window / unavailable) | 🔴 gap | `src/launchers/embedded_tool_bootstrap.py` | — | `ui/src/pages/Dashboard.tsx` | #7461 |
| `mocap.breadth`<br>Motion-capture breadth (C3D upload/playback, OpenPose source) | ✅ parity | `src/tools/freemocap_sidecar/run_freemocap.py` | `src/api/routes/motion_capture.py` | `ui/src/pages/MotionCapture.tsx` | — |
| `onboarding.about_version`<br>About/version info + onboarding | 🔴 gap | `src/launchers/about_dialog.py` | — | — | #7459 |
| `optimization.swing_optimizer`<br>Swing Optimizer (trajectory optimization GUI) | ⚪ exempt | `src/shared/python/optimization/swing_optimizer.py` | — | — | Desktop optimization GUI; desktop-only candidate pending #7460. — **pending decision (#7460)** |
| `platform.aip_protocol`<br>AI Protocol (AIP) structured method dispatch | 🔌 api_only | — | `src/api/routes/aip.py` | — | REST protocol consumed by both shells; no dedicated tile surface in either (issue #8861). |
| `settings.desktop_only_tabs`<br>Desktop-only settings tabs (MCP Servers, Processes, Startup/Docker, Layout, Performance) | ⚪ exempt | `src/launchers/settings_dialog.py` | — | — | Desktop-process management (MCP server processes, Docker startup, window layout, app zoom of native widgets) has no browser equivalent; awaiting the desktop-only exemption decision in issue #7460. — **pending decision (#7460)** |
| `settings.preferences`<br>Settings/preferences surface + persistence | ✅ parity | `src/launchers/settings_dialog.py` | `src/api/routes/settings.py` | `ui/src/pages/Settings.tsx` | — |
| `sidekick.terminal_repl_jupyter_skills`<br>Sidekick OS terminal / REPL / Jupyter / skills | ⚪ exempt | `src/launchers/launcher_sidekick_sidebar.py` | — | — | Desktop-native OS integration (terminal/REPL/Jupyter/skills) per ADR-0028; final disposition pending #7460. — **pending decision (#7460)** |
| `simulation.controls_wiring`<br>Web SimulationControls wiring (camera presets, recording toggle, trajectory export, force overlays, actuator controls) | 🔴 gap | `src/launchers/launcher_simulation.py` | — | `ui/src/components/simulation/SimulationControls.tsx` | #7452 |
| `simulation.golf_suite_batch`<br>Golf Simulation Suite (parameter sweeps, batch runs) | ⚪ exempt | `src/tools/golf_simulation_suite/__main__.py` | — | — | Desktop batch-simulation GUI; desktop-only candidate pending #7460. — **pending decision (#7460)** |
| `simulation.realtime_ws_stream`<br>Live simulation data over WebSocket pub-sub | 🔌 api_only | `src/launchers/launcher_simulation.py` | `src/api/routes/simulation_ws.py` | `ui/src/pages/Simulation.tsx` | WebSocket endpoint consumed by both shells; no dedicated tile surface in either (issue #8861). |
| `simulation.shot_tracer`<br>Shot Tracer / ball-flight visualization | ✅ parity | `src/launchers/_shot_tracer_gui.py` | `src/api/routes/ball_flight.py` | `ui/src/pages/BallFlight.tsx` | — |
| `simulation.swing_objective_lab`<br>Swing Objective Lab — mechanism-vs-outcome downswing comparison | ✅ parity | `src/launchers/adapters/swing_objective_lab_embed.py` | `src/api/routes/swing_objectives.py` | `ui/src/pages/SwingObjectiveLab.tsx` | — |
| `tools.bunkershot3d_workbench`<br>BunkerShot3D designer workbench (W2 sole parameters, W3 sand condition, F0 dynamic-RFT shot, W7 metrics, playability window, bounce utilisation, animated sole load field, 3-D shot animation through the ADR-0027 viewport, linked scalar traces with a validity band, F1 sand-field cross-sections, A/B comparison, validity verdict) | 🔴 gap | `src/tools/bunker_shot_gui/gui.py` | — | — | #8607 |
| `tools.character_builder`<br>Character Builder (humanoid URDF generation) | 🔴 gap | `src/shared/python/model_generation/cli/main.py` | `src/api/routes/character_builder.py` | `ui/src/pages/CharacterBuilder.tsx` | #7448 |
| `tools.data_explorer`<br>Data Explorer (import/filter/visualize datasets) | 🔴 gap | — | `src/api/routes/data_explorer.py` | `ui/src/pages/DataExplorer.tsx` | #7448 |
| `tools.dataset_generator`<br>Swing dataset generation and import | ✅ parity | — | `src/api/routes/dataset.py` | `ui/src/pages/DatasetGenerator.tsx` | — |
| `tools.launch_monitor_analytics`<br>Launch-monitor import, interdependency analysis, monitor comparison, dispersion, and longitudinal trends | 🔴 gap | `src/tools/launch_monitor_analytics/gui.py` | `src/api/routes/launch_monitor_analytics.py` | — | #8364 |
| `tools.matlab_suite`<br>MATLAB/Simscape model suite | ⚪ exempt | `src/launchers/matlab_suite_dialog.py` | — | — | Requires a local MATLAB installation; desktop-only candidate pending #7460. — **pending decision (#7460)** |
| `tools.model_explorer`<br>Model Explorer (browse/select/build URDF-MJCF) | 🔴 gap | `src/tools/model_explorer/launch_model_explorer.py` | `src/api/routes/model_explorer.py` | `ui/src/pages/ModelExplorer.tsx` | #7448 |
| `tools.pose_editing`<br>Pose Studio interactive pose editing | ⚪ exempt | `src/tools/pose_studio/__main__.py` | — | — | Interactive 3D pose editing; desktop-only candidate pending #7460. — **pending decision (#7460)** |
| `tools.putting_green`<br>Putting green simulation | ✅ parity | `src/engines/physics_engines/putting_green/python/simulator.py` | `src/api/routes/putting_green.py` | `ui/src/pages/PuttingGreen.tsx` | — |
| `tools.rate_of_closure`<br>Rate of Closure Impact Explorer (swing-impact-flight-putting simulation suite) | ✅ parity | `vendor/ud-tools/src/rate_of_closure/launch_pyqt6.py` | `src/api/local_server.py` | `ui/src/pages/ImpactExplorer.tsx` | Desktop tile launches the vendored PyQt app; /tools/impact-explorer embeds the vendored React build when present (built with --base=/impact-explorer-app/) and states how to build it when absent. |
| `tools.terrain_engine`<br>Terrain and topography configuration | ✅ parity | — | `src/api/routes/terrain.py` | `ui/src/pages/Terrain.tsx` | — |

## Launcher tile coverage

Tiles from `src/config/launcher_manifest.json` mapped to registry entries:

| Tile id | Feature |
| --- | --- |
| `actuator_controls` | `simulation.controls_wiring` |
| `aip` | `platform.aip_protocol` |
| `analysis_tools_api` | `analysis.analysis_tools_api` |
| `ball_flight_simulator` | `launcher.tile_web_reachability` |
| `biomech_exercise` | `biomech.exercise_injury_dashboards` |
| `biomech_gait` | `biomech.exercise_injury_dashboards` |
| `biomech_sit_to_stand` | `biomech.exercise_injury_dashboards` |
| `bunkershot3d` | `tools.bunkershot3d_workbench` |
| `c3d_viewer` | `launcher.tile_web_reachability` |
| `canonical_core_comparison` | `canonical_core.workspaces` |
| `canonical_core_estimation` | `canonical_core.workspaces` |
| `character_builder` | `tools.character_builder` |
| `chat_assistant` | `chat.transport` |
| `config_setup_wizard` | `launcher.tile_web_reachability` |
| `cross_engine_dashboard` | `analysis.cross_engine_robustness` |
| `data_explorer` | `tools.data_explorer` |
| `data_processor` | `launcher.tile_web_reachability` |
| `dataset_generator` | `tools.dataset_generator` |
| `drake_dashboard` | `engines.dashboards` |
| `drake_golf` | `engines.load_and_simulate` |
| `drake_models_shared` | `launcher.model_repositories` |
| `force_overlays` | `simulation.controls_wiring` |
| `golf_environment` | `launcher.tile_web_reachability` |
| `golf_simulation_suite` | `simulation.golf_suite_batch` |
| `injury_analysis` | `biomech.exercise_injury_dashboards` |
| `jaxsim_dashboard` | `engines.dashboards` |
| `launch_monitor_analytics` | `tools.launch_monitor_analytics` |
| `library_tool` | `launcher.tile_web_reachability` |
| `matlab_suite` | `tools.matlab_suite` |
| `mediapipe_analysis` | `launcher.tile_web_reachability` |
| `model_explorer` | `tools.model_explorer` |
| `motion_capture` | `mocap.breadth` |
| `motion_pipeline` | `mocap.breadth` |
| `motion_target_preview` | `mocap.breadth` |
| `movement_optimizer` | `launcher.tile_web_reachability` |
| `mujoco_dashboard` | `engines.dashboards` |
| `mujoco_models_shared` | `launcher.model_repositories` |
| `mujoco_unified` | `engines.load_and_simulate` |
| `myosim_suite` | `engines.load_and_simulate` |
| `openpose_analysis` | `launcher.tile_web_reachability` |
| `opensim_golf` | `engines.load_and_simulate` |
| `opensim_models_shared` | `launcher.model_repositories` |
| `pendulum_simulator` | `engines.load_and_simulate` |
| `perturbation_analysis` | `analysis.cross_engine_robustness` |
| `pid_generator` | `launcher.tile_web_reachability` |
| `pinn_hybrid` | `launcher.tile_web_reachability` |
| `pinn_pure_rigid` | `launcher.tile_web_reachability` |
| `pinocchio_dashboard` | `engines.dashboards` |
| `pinocchio_golf` | `engines.load_and_simulate` |
| `pinocchio_models_shared` | `launcher.model_repositories` |
| `pose_studio` | `tools.pose_editing` |
| `pose_subscriber_demo` | `launcher.tile_web_reachability` |
| `project_map` | `docs.document_library` |
| `putting_green` | `tools.putting_green` |
| `putting_green_gui` | `tools.putting_green` |
| `rate_of_closure` | `tools.rate_of_closure` |
| `realtime_ws` | `simulation.realtime_ws_stream` |
| `robotics_module` | `launcher.tile_web_reachability` |
| `shot_tracer` | `simulation.shot_tracer` |
| `sidekick` | `chat.transport` |
| `simulation_backends` | `analysis.counterfactuals` |
| `starting_pose_matcher` | `mocap.breadth` |
| `swing_flight_pipeline` | `launcher.tile_web_reachability` |
| `swing_objective_lab` | `simulation.swing_objective_lab` |
| `swing_optimizer` | `optimization.swing_optimizer` |
| `terrain_engine` | `tools.terrain_engine` |
| `tools_calculator_hub` | `launcher.tile_web_reachability` |
| `training_controller` | `launcher.tile_web_reachability` |
| `unreal_integration` | `launcher.tile_web_reachability` |
| `video_analyzer` | `launcher.tile_web_reachability` |
| `video_processor` | `launcher.tile_web_reachability` |
