---
title: MuJoCo
tile_id: mujoco_unified
status: complete
---

# MuJoCo

## Purpose

The MuJoCo tile is the suite's main biomechanical golf-swing workbench. You
pick a swing model (from a 2 degree-of-freedom pendulum up to a 28
degree-of-freedom biomechanical golfer), run it forward under MuJoCo, drive the
joints with torque sliders or pose sliders, watch the result in an embedded
viewer, and record the run for plotting and export. It is the tile to start with
if you do not have a reason to prefer another engine.

## Inputs

| Input | Unit / type | Notes |
| --- | --- | --- |
| Model selection | enumeration | Built-in MJCF models, chosen in the *Physics Models & Mechanisms* combo box. |
| Operating mode | `Dynamic (Torque Control)` or `Kinematic (Pose Adjustment)` | Dynamic integrates physics; kinematic sets joint positions directly. |
| Per-actuator torque command | N*m | One slider per actuator, range -100 N*m to +100 N*m, default 0 N*m; the slider label reads back in N*m. |
| Joint pose | rad (revolute) / m (prismatic) | Kinematic-mode sliders, limits taken from the loaded MJCF. |
| Loaded MJCF model | XML string | The built-in models are embedded XML constants in `models/` of the `mujoco_humanoid_golf` package. |
| Camera / overlay toggles | boolean | Force, torque, contact and reference-frame overlays. |

Built-in model sizes as declared in the tile's own model configuration
(`__main__.py` and `gui/tabs/physics_tab.py`): chaotic driven pendulum
(2 DOF), double pendulum (2 DOF), triple pendulum (3 DOF), upper body plus arms
(10 DOF), full body with legs (15 DOF), advanced biomechanical (28 DOF), plus
imported CMU humanoid and MyoSuite musculoskeletal models where those packages
are installed.

## Outputs

| Output | Unit / type | Notes |
| --- | --- | --- |
| Live 3D viewport | rendered frames | MuJoCo scene at the widget's configured frame rate. |
| Simulation time | s | Shown in the status bar. |
| Joint positions / velocities | rad, rad/s (revolute) | Recorded per frame while recording is armed. |
| Applied joint torques | N*m | Whatever the sliders or the active control scheme commanded. |
| Recorded frame count | count | Status readout while recording. |
| Exported recording | file | Written through the shared exporter; see `src/shared/python/data_io/export.py`. |
| Plots | figures | Live Analysis, Plotting and Analysis tabs. |

## Method

MuJoCo's own solver does the integration. The tile is a Qt front end: the
launcher embeds `MainWidget`
(`src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/gui/core/main_widget.py`),
which wraps `AdvancedGolfAnalysisWindow`
(`.../gui/core/main_window.py`) as a child widget. That window is built on the
shared `SimulationGUIBase` and carries the Physics, Controls, Visualization,
Analysis, Plotting, Live Analysis, Interactive Pose and Manipulability tabs, plus
the Golf Swing Analysis, Grip Modelling and Humanoid Config top-level tabs.

Model loading and stepping go through the engine adapter
`MuJoCoPhysicsEngine` in
`src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/physics_engine.py`,
as documented in [MuJoCo engine reference](../engines/mujoco.md). The engine
capability list for this tile is declared in `src/config/models.yaml` under
`mujoco_unified`.

## Limitations

- MuJoCo must be installed. `main()` exits with a non-zero status and an
  install hint if `import mujoco` fails; there is no demo or fallback mode.
- The torque sliders are a manual open-loop input. They are not a controller,
  a trajectory optimiser, or an inverse-dynamics fit to measured data.
- Slider torque range is a fixed -100 N*m to +100 N*m regardless of what the
  loaded model's actuators can physically deliver.
- The legacy simple window (`main_simple()`, `MainWindow` in `__main__.py`) is
  kept for backwards compatibility only. It is not what the launcher tile
  shows, and it exposes a smaller fixed model list.
- Model fidelity is whatever the built-in MJCF declares. Nothing here validates
  a model's segment masses or inertias against a real golfer.

## See Also
- [MuJoCo engine reference](../engines/mujoco.md)
- [Engine capabilities matrix](../engines/engine_capabilities.md)
- [Engine support tiers](../engines/support_tiers.md)
- [Engine selection guide](engine_selection.md)
- [Simulation controls](simulation_controls.md)
- [Visualization](visualization.md)
- [Analysis tools](analysis_tools.md)
- [MuJoCo Models library tile](mujoco_models_shared.md)
