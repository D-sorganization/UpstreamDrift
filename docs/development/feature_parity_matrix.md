# Feature Parity Matrix (PyQt6 ↔ Tauri/React)

<!-- AUTO-GENERATED — do not edit by hand. -->
<!-- Regenerate with: python -m scripts.generate_feature_parity_matrix -->

Generated from [`src/config/feature_parity.json`](../../src/config/feature_parity.json) (registry v1.0.0).
The PyQt6 desktop app is the canonical model; the web app must match
(epic #7462, registry mechanism #7445).

**Summary:** 37 parity · 7 gap · 16 exempt (12 pending decision in #7460).

| Feature | Status | PyQt6 | API | Web | Tracking |
| --- | --- | --- | --- | --- | --- |
| `analysis.analysis_tools_api`<br>Analysis Tools REST endpoints (swing metrics, biomechanics) | ✅ parity | — | `src/api/routes/analysis_tools.py` | `ui/src/pages/AnalysisTools.tsx` | — |
| `analysis.counterfactuals`<br>ZTCF/ZVCF + induced-acceleration counterfactuals | ✅ parity | `src/shared/python/biomechanics/ztcf.py` | `src/api/routes/analysis.py` | `ui/src/components/analysis/CounterfactualPanel.tsx` | — |
| `analysis.cross_engine_robustness`<br>Cross-engine robustness dashboard (perturbation/CV) | ✅ parity | `src/launchers/cross_engine_dashboard.py` | `src/api/routes/cross_engine.py` | `ui/src/pages/CrossEngineDashboard.tsx` | — |
| `analysis.grip_wrench`<br>Per-hand grip wrench on the club (weld multipliers and efc_force): overlay glyphs and force/couple plots | ✅ parity | `src/tools/grip_wrench_plots/gui.py` | `src/api/routes/analysis.py` | `ui/src/components/analysis/GripWrenchCharts.tsx` | — |
| `analysis.ground_reaction`<br>Ground reaction plots: per-foot and net force (N and body weights), vertical load share, CoP path, free moment and moment about CoM | ✅ parity | `src/tools/ground_reaction_plots/gui.py` | `src/api/routes/analysis.py` | `ui/src/components/analysis/GroundReactionCharts.tsx` | — |
| `analysis.impact_parameters`<br>Impact parameters panel (speed, attack angle, path, face, face-to-path, dynamic loft, spin loft) relative to a target line | ✅ parity | `src/tools/impact_parameters_panel/gui.py` | `src/api/routes/analysis.py` | `ui/src/components/analysis/ImpactParametersPanel.tsx` | — |
| `analysis.static_plots`<br>Static analysis plots (20+ plot types) | ✅ parity | `vendor/ud-tools/src/shared/python/plot_engine/pyqt6_widget.py` | `src/api/routes/analysis_plots.py` | `ui/src/components/analysis/PlotsSection.tsx` | — |
| `biomech.exercise_injury_dashboards`<br>Exercise + injury-risk biomechanics dashboards | ⚪ exempt | `src/launchers/exercise_dashboard.py` | — | — | Desktop biomechanics dashboards; desktop-only candidate pending #7460. — **pending decision (#7460)** |
| `canonical_core.workspaces`<br>Canonical-core estimation/comparison workspaces | ✅ parity | `src/tools/canonical_core/estimation.py` | — | `ui/src/pages/CanonicalCoreShell.tsx` | — |
| `chat.live_context`<br>Live app/engine context in chat | ✅ parity | `src/launchers/launcher_sidekick_sidebar.py` | `src/api/services/chat_app_context.py` | `ui/src/components/ui/ChatContextChip.tsx` | — |
| `chat.transport`<br>AI chat transport (message send/stream) | ✅ parity | `src/launchers/launcher_sidekick_sidebar.py` | `src/api/routes/chat_ws.py` | `ui/src/pages/Chat.tsx` | — |
| `diagnostics.integrations_health`<br>Diagnostics + integrations-health panel | ✅ parity | `src/launchers/integrations_health_panel.py` | `src/api/routes/diagnostics.py` | `ui/src/components/ui/DiagnosticsPanel.tsx` | — |
| `docs.document_library`<br>Document library / project map viewer | ⚪ exempt | `src/launchers/library_widget.py` | — | — | Desktop documentation browser; desktop-only candidate pending #7460. — **pending decision (#7460)** |
| `drake.gui_force_overlay`<br>Drake GUI force/torque arrows and live tension/compression shading | ✅ parity | `src/engines/physics_engines/drake/python/src/drake_force_overlay.py` | `src/api/routes/force_overlays.py` | `ui/src/components/visualization/ForceOverlay.tsx` | — |
| `engines.dashboards`<br>Per-engine interactive dashboards (Drake/MuJoCo/Pinocchio) | ⚪ exempt | `src/launchers/drake_dashboard.py` | — | — | Experimental desktop dashboards; desktop-only candidate pending #7460. — **pending decision (#7460)** |
| `engines.load_and_simulate`<br>Engine load/probe + basic simulation loop | ✅ parity | `src/launchers/launcher_simulation.py` | `src/api/routes/engines.py` | `ui/src/pages/Simulation.tsx` | — |
| `export.recordings_downloads`<br>Export/recording parity (HDF5/MAT/C3D/CSV/video, persisted recordings) | ✅ parity | `src/shared/python/data_io/export.py` | `src/api/routes/recordings.py` | `ui/src/components/simulation/RecordingsPanel.tsx` | — |
| `export.video_speed_variants`<br>Time-based native video export with full- and half-speed clips and an impact slow-motion clip | ⚪ exempt | `src/tools/native_viewer_export/cli.py` | — | — | Headless CLI batch export of mp4 clips (60 fps, 1x and 0.5x, optional 0.1x impact window) through each engine's desktop viewer; no web route renders or serves these clips, so there is no browser equivalent until one is added (GCV-14, #11720, epic #11706). |
| `launcher.docker_management`<br>Docker engine management dialog | ⚪ exempt | `src/launchers/docker_manager.py` | — | — | Manages the local Docker daemon from the desktop; desktop-only candidate pending #7460. — **pending decision (#7460)** |
| `launcher.embedded_tabs_docks`<br>Embedded tool host (tabs + docks) | ⚪ exempt | `src/launchers/embedded_host.py` | — | — | Desktop windowing composition is explicitly desktop-only per ADR-0028 (React shell uses routes, not embedded Qt docks). |
| `launcher.mcp_config`<br>MCP server configuration writer/preferences | ⚪ exempt | `src/launchers/mcp_config_writer.py` | — | — | Writes local MCP configuration files for desktop AI integrations; desktop-only candidate pending #7460. — **pending decision (#7460)** |
| `launcher.tile_grid`<br>Launcher tile grid from shared manifest | ✅ parity | `src/launchers/embedded_tool_bootstrap.py` | `src/api/routes/launcher.py` | `ui/src/pages/Dashboard.tsx` | — |
| `launcher.tile_web_reachability`<br>Manifest tile web-reachability contract (route / native-window / unavailable) | ✅ parity | `src/launchers/embedded_tool_bootstrap.py` | `src/api/routes/launcher.py` | `ui/src/pages/Dashboard.tsx` | — |
| `lifting.five_lift_viewing`<br>Weightlifting viewing and analysis for the five lifts (squat, deadlift, bench press, snatch, clean and jerk): lift selection, barbell, hand-force and GRF overlays, plots | 🔴 gap | `src/launchers/exercise_dashboard.py` | `src/api/routes/lifting.py` | — | #11748 |
| `mocap.breadth`<br>Motion-capture breadth (C3D upload/playback, OpenPose source) | ✅ parity | `src/tools/freemocap_sidecar/run_freemocap.py` | `src/api/routes/motion_capture.py` | `ui/src/pages/MotionCapture.tsx` | — |
| `mujoco.force_overlays`<br>MuJoCo GUI force/torque overlays drawn through the shared glyph renderers (native viewer and MeshCat) | ✅ parity | `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/sim_rendering_mixin.py` | `src/api/routes/force_overlays.py` | `ui/src/components/visualization/ForceOverlay.tsx` | — |
| `onboarding.about_version`<br>About/version info + onboarding | ✅ parity | `src/launchers/about_dialog.py` | `src/api/routes/about.py` | `ui/src/components/ui/AboutModal.tsx` | — |
| `opencap.session_import`<br>OpenCap session import action and OpenSim engine handoff | ✅ parity | `src/engines/physics_engines/opensim/python/opencap_import_action.py` | `src/api/routes/opencap.py` | `ui/src/components/opencap/OpenCapImportModal.tsx` | — |
| `optimization.swing_optimizer`<br>Swing Optimizer (trajectory optimization GUI) | ⚪ exempt | `src/shared/python/optimization/swing_optimizer.py` | — | — | Desktop optimization GUI; desktop-only candidate pending #7460. — **pending decision (#7460)** |
| `platform.aip_protocol`<br>AI Protocol (AIP) structured method dispatch | ✅ parity | — | `src/api/routes/aip.py` | — | — |
| `render.body_appearance`<br>Golfer body appearance with a visible head, face and neck in every engine render | ⚪ exempt | `src/shared/python/model_appearance/head.py` | — | — | Engine-render appearance layer (MuJoCo appearance and visual layers, Drake and Pinocchio MeshCat, MyoSuite, OpenSim native viewers) that produces mp4 and PNG artefacts and has no interactive PyQt6 surface; the web GolferModel.tsx head is a follow-up under #11718 (epic #11706). |
| `render.club_head_and_ball`<br>Realistic Club Head And Ball Rendering | 🔴 gap | `src/shared/python/model_appearance/club_head_mesh.py` | — | — | #11717 |
| `settings.desktop_only_tabs`<br>Desktop-only settings tabs (MCP Servers, Processes, Startup/Docker, Layout, Performance) | ⚪ exempt | `src/launchers/settings_dialog.py` | — | — | Desktop-process management (MCP server processes, Docker startup, window layout, app zoom of native widgets) has no browser equivalent; awaiting the desktop-only exemption decision in issue #7460. — **pending decision (#7460)** |
| `settings.preferences`<br>Settings/preferences surface + persistence | ✅ parity | `src/launchers/settings_dialog.py` | `src/api/routes/settings.py` | `ui/src/pages/Settings.tsx` | — |
| `sidekick.terminal_repl_jupyter_skills`<br>Sidekick OS terminal / REPL / Jupyter / skills | ⚪ exempt | `src/launchers/launcher_sidekick_sidebar.py` | — | — | Desktop-native OS integration (terminal/REPL/Jupyter/skills) per ADR-0028; final disposition pending #7460. — **pending decision (#7460)** |
| `simulation.controls_wiring`<br>Web SimulationControls wiring (camera presets, recording toggle, trajectory export, force overlays, actuator controls) | ✅ parity | `src/launchers/launcher_simulation.py` | `src/api/routes/force_overlays.py` | `ui/src/components/simulation/SimulationControls.tsx` | #7452 |
| `simulation.golf_suite_batch`<br>Golf Simulation Suite (parameter sweeps, batch runs) | ⚪ exempt | `src/tools/golf_simulation_suite/__main__.py` | — | — | Desktop batch-simulation GUI; desktop-only candidate pending #7460. — **pending decision (#7460)** |
| `simulation.realtime_ws_stream`<br>Live simulation data over WebSocket pub-sub | ✅ parity | `src/launchers/launcher_simulation.py` | `src/api/routes/simulation_ws.py` | `ui/src/pages/Simulation.tsx` | — |
| `simulation.shot_tracer`<br>Shot Tracer / ball-flight visualization | ✅ parity | `src/launchers/_shot_tracer_gui.py` | `src/api/routes/ball_flight.py` | `ui/src/pages/BallFlight.tsx` | — |
| `simulation.swing_objective_lab`<br>Swing Objective Lab — mechanism-vs-outcome downswing comparison | ✅ parity | `src/launchers/adapters/swing_objective_lab_embed.py` | `src/api/routes/swing_objectives.py` | `ui/src/pages/SwingObjectiveLab.tsx` | — |
| `tools.bunkershot3d_workbench`<br>BunkerShot3D designer workbench (W2 sole parameters, W3 sand condition, F0 dynamic-RFT shot, W7 metrics, playability window, bounce utilisation, animated sole load field, 3-D shot animation through the ADR-0027 viewport, linked scalar traces with a validity band, F1 sand-field cross-sections, A/B comparison, validity verdict) | 🔴 gap | `src/tools/bunker_shot_gui/gui.py` | `src/api/routes/bunker_workbench.py` | — | #9545 |
| `tools.capture_rig`<br>Capture Rig camera controller with the live view as the central pane and every control in a movable dock (step rail with the next action, settings tabs, action grid, log drawer) with saved layouts, themed header with status strip and workflow-grouped actions that reflow to fit a narrow pane, transport-style recorder (countdown, presets, REC readout, live view during the take), overlay player, camera-subset matching (variants), model-on-video overlays, provenance viewer and manual point annotation/editing; layout model with named layout presets, an interactive layout editor, live preview and playback composited through the chosen multiview layout, and composite multiview video export; non-destructive swing trim/crop and separate video export, visible capture library with durable swing notes, import/open/edit, archive/restore and storage management; saved coaching lines/arrows/circles/ellipses/rectangles with selection, resize, undo/redo, frame visibility and annotated still/video exports; reference comparison and overlay player force/torque vector arrow layers (FTO-25, #11310) | ⚪ exempt | `src/tools/capture_rig/__main__.py` | — | — | Drives local USB cameras through ffmpeg/DirectShow and reads multi-gigabyte recordings from disk (#9619); variants, overlays, provenance and annotation (#9790, #9791) and the multiview layout model/presets/editor and the layout-composited live preview and playback (#9810, #9811, #9812, #9813, #9814) run on the same local session trees; desktop-only candidate pending #7460. — **pending decision (#7460)** |
| `tools.character_builder`<br>Character Builder (spec-native character, presets, engine export) | ✅ parity | `src/tools/character_builder/gui.py` | `src/api/routes/character_builder.py` | `ui/src/pages/CharacterBuilder.tsx` | — |
| `tools.data_explorer`<br>Data Explorer (import/filter/visualize datasets) | ✅ parity | — | `src/api/routes/data_explorer.py` | `ui/src/pages/DataExplorer.tsx` | — |
| `tools.dataset_generator`<br>Swing dataset generation and import | ✅ parity | — | `src/api/routes/dataset.py` | `ui/src/pages/DatasetGenerator.tsx` | — |
| `tools.golf_simulator`<br>Golf Simulator capability-aware controls and replay submission | ✅ parity | `src/tools/golf_simulator/gui.py` | `src/api/routes/golf_simulator.py` | `ui/src/pages/GolfSimulator.tsx` | — |
| `tools.launch_monitor_analytics`<br>Launch-monitor import, interdependency analysis, monitor comparison, dispersion, and longitudinal trends | 🔴 gap | `src/tools/launch_monitor_analytics/gui.py` | `src/api/routes/launch_monitor_analytics.py` | — | #11987 |
| `tools.matched_swing_browser`<br>Matched Swing Results Browser | 🔴 gap | `src/tools/matched_swing_browser/gui.py` | — | — | #11987 |
| `tools.matlab_suite`<br>MATLAB/Simscape model suite | ⚪ exempt | `src/launchers/matlab_suite_dialog.py` | — | — | Requires a local MATLAB installation; desktop-only candidate pending #7460. — **pending decision (#7460)** |
| `tools.model_explorer`<br>Model Explorer (browse/select/build URDF-MJCF) | ✅ parity | `src/tools/model_explorer/launch_model_explorer.py` | `src/api/routes/model_explorer.py` | `ui/src/pages/ModelExplorer.tsx` | — |
| `tools.model_explorer.frankenstein_assembly`<br>Frankenstein drag-and-drop assembly with typed attachment ports | 🔴 gap | `src/tools/model_explorer/frankenstein_editor/assembly_panel.py` | `src/api/routes/model_explorer_assembly.py` | — | #11651 |
| `tools.motion_matching`<br>Motion Matching tour-average and club-only Excel matching | ✅ parity | `src/tools/motion_matching/gui.py` | — | — | — |
| `tools.native_viewer_export`<br>Native per-engine viewer video export (CLI) | ⚪ exempt | `src/tools/native_viewer_export/cli.py` | — | — | Headless command-line batch tool that drives each engine's own desktop viewer (MeshCat via headless Chromium, simbody-visualizer under xvfb, MuJoCo EGL); the artefacts are mp4 files, so there is no interactive browser equivalent (epic #11673). |
| `tools.necromatcher`<br>Historical Player Library And Source Review | ✅ parity | `src/tools/necromatcher/gui.py` | `src/api/routes/necromatcher.py` | `ui/src/pages/Necromatcher.tsx` | — |
| `tools.pose_editing`<br>Pose Studio interactive pose editing | ⚪ exempt | `src/tools/pose_studio/__main__.py` | — | — | Interactive 3D pose editing and shared scene-bound native reference points/planes (#9942); desktop-only candidate pending #7460. — **pending decision (#7460)** |
| `tools.putting_green`<br>Putting green simulation | ✅ parity | `src/engines/physics_engines/putting_green/python/simulator.py` | `src/api/routes/putting_green.py` | `ui/src/pages/PuttingGreen.tsx` | — |
| `tools.rate_of_closure`<br>Rate of Closure Impact Explorer (swing-impact-flight-putting simulation suite) | 🔴 gap | `vendor/ud-tools/src/rate_of_closure/launch_pyqt6.py` | `src/api/local_server.py` | `ui/src/pages/ImpactExplorer.tsx` | #11987 |
| `tools.terrain_engine`<br>Terrain and topography configuration | ✅ parity | — | `src/api/routes/terrain.py` | `ui/src/pages/Terrain.tsx` | — |
| `tools.video_analyzer`<br>Video Analyzer (pose tracking and force/torque overlay) | ✅ parity | `src/tools/video_analyzer/gui.py` | `src/api/routes/video_overlays.py` | `ui/src/pages/VideoAnalyzer.tsx` | — |
| `visualization.force_color_controls`<br>Shared Segment Force Color Controls | ✅ parity | `src/shared/python/body_part_viz/force_color_controls.py` | — | `ui/src/components/visualization/ForceColorControls.tsx` | — |

## Launcher Tile Coverage

Tiles from `src/config/launcher_manifest.json` mapped to registry entries:

| Tile id | Feature |
| --- | --- |
| `actuator_controls` | `simulation.controls_wiring` |
| `aip` | `platform.aip_protocol` |
| `analysis_tools_api` | `analysis.analysis_tools_api` |
| `biomech_exercise` | `biomech.exercise_injury_dashboards` |
| `bunkershot3d` | `tools.bunkershot3d_workbench` |
| `canonical_core_comparison` | `canonical_core.workspaces` |
| `canonical_core_estimation` | `canonical_core.workspaces` |
| `capture_rig` | `tools.capture_rig` |
| `character_builder` | `tools.character_builder` |
| `chat_assistant` | `chat.transport` |
| `cross_engine_dashboard` | `analysis.cross_engine_robustness` |
| `data_explorer` | `tools.data_explorer` |
| `data_processor` | `launcher.tile_web_reachability` |
| `dataset_generator` | `tools.dataset_generator` |
| `drake_dashboard` | `engines.dashboards` |
| `drake_golf` | `engines.load_and_simulate` |
| `force_overlays` | `simulation.controls_wiring` |
| `golf_simulation_suite` | `simulation.golf_suite_batch` |
| `golf_simulator` | `tools.golf_simulator` |
| `grip_wrench_plots` | `analysis.grip_wrench` |
| `ground_reaction_plots` | `analysis.ground_reaction` |
| `impact_parameters` | `analysis.impact_parameters` |
| `injury_analysis` | `biomech.exercise_injury_dashboards` |
| `launch_monitor_analytics` | `tools.launch_monitor_analytics` |
| `matched_swing_browser` | `tools.matched_swing_browser` |
| `matlab_suite` | `tools.matlab_suite` |
| `model_explorer` | `tools.model_explorer` |
| `motion_capture` | `mocap.breadth` |
| `motion_matching` | `tools.motion_matching` |
| `motion_pipeline` | `mocap.breadth` |
| `motion_target_preview` | `mocap.breadth` |
| `mujoco_dashboard` | `engines.dashboards` |
| `mujoco_unified` | `engines.load_and_simulate` |
| `myosim_suite` | `engines.load_and_simulate` |
| `necromatcher` | `tools.necromatcher` |
| `opensim_golf` | `engines.load_and_simulate` |
| `pendulum_simulator` | `engines.load_and_simulate` |
| `perturbation_analysis` | `analysis.cross_engine_robustness` |
| `pid_generator` | `launcher.tile_web_reachability` |
| `pinocchio_dashboard` | `engines.dashboards` |
| `pinocchio_golf` | `engines.load_and_simulate` |
| `pose_studio` | `tools.pose_editing` |
| `project_map` | `docs.document_library` |
| `putting_green` | `tools.putting_green` |
| `rate_of_closure` | `tools.rate_of_closure` |
| `realtime_ws` | `simulation.realtime_ws_stream` |
| `robotics_module` | `launcher.tile_web_reachability` |
| `shadow_tracker` | `mocap.breadth` |
| `shot_tracer` | `simulation.shot_tracer` |
| `simulation_backends` | `analysis.counterfactuals` |
| `starting_pose_matcher` | `mocap.breadth` |
| `swing_objective_lab` | `simulation.swing_objective_lab` |
| `swing_optimizer` | `optimization.swing_optimizer` |
| `terrain_engine` | `tools.terrain_engine` |
| `tools_calculator_hub` | `launcher.tile_web_reachability` |
| `unreal_integration` | `launcher.tile_web_reachability` |
| `video_analyzer` | `tools.video_analyzer` |
| `video_processor` | `launcher.tile_web_reachability` |
