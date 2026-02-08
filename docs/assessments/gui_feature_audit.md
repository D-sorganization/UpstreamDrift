# GUI Feature Audit: PyQt6 vs React Implementation Status

**Date:** 2026-02-08
**Scope:** All GUI features across PyQt6 and React frontends

## Feature Parity Summary

| Feature | PyQt6 | React |
|---|---|---|
| Force/Torque Visualization | Full | None |
| Joint/Actuator Sliders | Full | None |
| Model Explorer / Frankenstein Editor | Full | None (tile link only) |
| Character Builder | Full | None |
| Putting Green Simulation | Full engine | None |
| 6-DOF Model Manipulation | Full (IK-based drag) | Camera orbit only |
| Control Panels During Sim | Full (multi-tab) | Start/Pause/Stop only |
| 3D Visualization | MuJoCo native | Three.js basic golfer |

## 1. Forces & Torques On Screen

### PyQt6: IMPLEMENTED

- **MuJoCo VisualizationTab** (`visualization_tab.py:257-349`): Force & Torque Visualization group with joint torque vectors, constraint forces, induced acceleration, counterfactuals (ZTCF/ZVCF), contact forces, isolation to selected body, and adjustable scale sliders.
- **sim_widget.py:1141-1183**: `_add_force_torque_overlays()` renders arrows via OpenCV `arrowedLine` projected from 3D to screen space.
- **Drake GUI** (`drake_gui_app.py:859-864`): Show Forces / Show Torques checkboxes.

### React: NOT IMPLEMENTED

`Scene3D.tsx` has no force/torque visualization.

## 2. Changeable Joint Inputs

### PyQt6: IMPLEMENTED

- **ControlsTab** (`controls_tab.py`): Per-actuator sliders with control type selection (constant, polynomial, damping).
- **JointManipulator** (`joint_manipulator.py:149-748`): Full slider+spinbox per joint, auto-load from URDF, filter, reset, random pose, joint property editor.

### React: NOT IMPLEMENTED

`ParameterPanel.tsx` only has simulation-level parameters. No joint/actuator controls.

## 3. Model Explorer / Character Builder

### PyQt6: IMPLEMENTED

- **FrankensteinEditor** (`frankenstein_editor.py:566-1109`): Side-by-side URDF editor for component stealing between models. Copy, merge, swap, replace subtree, diff.
- **Humanoid Character Builder** (`humanoid_character_builder/`): Video game-style character customization with height/mass/build type, anthropometry, mesh generation, URDF export.
- **Model Explorer** (`model_explorer/main_window.py`): Dockable UI with segment panel, visualization, model library.

### React: NOT IMPLEMENTED

`LauncherDashboard.tsx` lists Model Explorer as a tile but contains no actual Model Explorer component.

## 4. Putting Green Simulation

### PyQt6 (backend): IMPLEMENTED

- **PuttingGreenSimulator** (`putting_green/python/simulator.py`): Complete physics engine (804 lines) with ball roll, turf properties, heightmaps, putter strokes, wind, hole detection, trajectory recording, practice mode, scatter analysis, aim assist, green reading.

### React: NOT IMPLEMENTED

No putting green GUI in React.

## 5. 6-DOF Model Movement

### PyQt6: IMPLEMENTED

- **InteractiveManipulation** (`interactive_manipulation.py`): Mouse picking via ray-casting, IK-based drag, body constraints (fixed/relative), pose library.

### React: PARTIAL (camera only)

`Scene3D.tsx` has `OrbitControls` for camera orbit/zoom/pan. No model body manipulation.

## 6. Interactive Control Panels During Simulation

### PyQt6: IMPLEMENTED

- **ControlsTab**: Play/Pause, actuator sliders, control types, polynomial generator, quick camera.
- **VisualizationTab**: Camera presets, force/torque toggles, trajectory recording.
- **Advanced GUI**: Multi-tab control panel docked alongside simulation viewport.

### React: MINIMAL

`SimulationControls.tsx`: Start/Pause/Resume/Stop buttons only. `ParameterPanel.tsx`: Duration/timestep/toggles only.

## Old/Incomplete Versions to Archive

| Path | Issue |
|---|---|
| `Simscape.../Python Version/golf_gui_r0/` | Old r0 GUI with 5 files, superseded by current MuJoCo GUI |
| `Simscape.../Python Version/integrated_golf_gui_r0/` | Old r0 integrated GUI with 15+ files |
| `2D_Golf_Model/matlab/Backup Folder/` | Explicitly named backup |
| `SkeletonPlotter/Older Revs/` | Explicitly labeled old revisions |
| `Model Output/Scripts/_Archived Scripts/` | Already labeled archived but in source tree |
| `3D_Golf_Model/golf_swing_dataset_20250907_bk` | Backup dataset |
| `2D_Golf_Model/matlab/Scripts/` vs `Model Output/Scripts/` | Near-duplicate directory structures |
