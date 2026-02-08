# React UI Parity Issues

**Date:** 2026-02-08
**Source Assessment:** `docs/assessments/gui_feature_audit.md`
**Scope:** All features implemented in PyQt6 GUIs that are missing or incomplete in the React/Tauri frontend (`ui/`)

Each section below is formatted as a standalone GitHub issue ready for creation.

---

## Issue 1: Add Force & Torque Vector Overlay to React 3D Scene

**Labels:** `enhancement`, `react-ui`, `visualization`, `high-priority`

### Description

The PyQt6 MuJoCo GUI (`visualization_tab.py:257-349`, `sim_widget.py:1141-1183`) renders real-time force and torque vectors as arrow overlays on the 3D simulation viewport. The React `Scene3D.tsx` has no equivalent. Users of the web UI cannot visualize the physics driving the simulation.

### Current State (PyQt6)

- **Joint torque vectors** with adjustable scale slider (`show_torques_cb`)
- **Constraint force vectors** with adjustable scale slider (`show_forces_cb`)
- **Induced acceleration vectors** (gravity, velocity, total, per-actuator) in magenta
- **Counterfactual overlays** (ZTCF accel, ZVCF torque) in yellow
- **Contact force arrows** (`show_contacts_cb`)
- **Isolate to selected body** filter
- Rendering via OpenCV `arrowedLine` projected from 3D world to screen

The Drake GUI (`drake_gui_app.py:859-864`) also has Show Forces / Show Torques toggles.

### Current State (React)

`Scene3D.tsx` renders a basic Three.js scene with a golfer model and club trajectory trail. Zero force/torque visualization exists.

### Acceptance Criteria

- [ ] `SimulationFrame` type in `client.ts` extended with `forces` and `torques` arrays from the WebSocket backend
- [ ] New `ForceArrows` Three.js component renders 3D arrow helpers at joint positions, colored by type (torque=cyan, constraint=red, contact=green)
- [ ] Toggle checkboxes in a new visualization panel: Show Torques, Show Forces, Show Contact Forces
- [ ] Scale slider controls arrow length
- [ ] Arrows update each frame from WebSocket data
- [ ] Isolate-to-body filter: click a body to show only its vectors

### Implementation Notes

- Use Three.js `ArrowHelper` or instanced mesh for performance
- Backend WS endpoint must stream `data.qfrc_actuator`, `data.qfrc_constraint`, and `data.cfrc_ext` per frame
- Reference: `sim_widget.py:_add_force_torque_overlays()` for projection logic

### Files to Create/Modify

- `ui/src/api/client.ts` — extend `SimulationFrame` interface
- `ui/src/components/visualization/ForceOverlay.tsx` — new component
- `ui/src/components/visualization/Scene3D.tsx` — integrate `ForceOverlay`
- `ui/src/components/visualization/VisualizationPanel.tsx` — new toggle/slider panel
- `ui/src/pages/Simulation.tsx` — wire panel into sidebar

---

## Issue 2: Add Joint/Actuator Control Sliders to React UI

**Labels:** `enhancement`, `react-ui`, `controls`, `high-priority`

### Description

The PyQt6 MuJoCo GUI (`controls_tab.py`) provides per-actuator sliders with selectable control types (constant, polynomial, damping). The Model Explorer (`joint_manipulator.py:149-748`) has a full `JointSliderWidget` per joint with spinbox, filter, reset, and random pose. The React UI has no joint or actuator control at all.

### Current State (PyQt6)

- Per-actuator slider rows with label, slider, and value display
- Control type dropdown per actuator: Constant, Polynomial, Damping
- Polynomial coefficient editor and `PolynomialGeneratorWidget`
- Simplified mode auto-triggers for models with >20 actuators
- JointManipulator: auto-populated from URDF, All/Movable/Fixed filter, Reset All, Random Pose

### Current State (React)

`ParameterPanel.tsx` has only simulation-level parameters (duration, timestep, GPU toggle). No per-joint or per-actuator controls exist.

### Acceptance Criteria

- [ ] New `JointControlPanel` component listing all actuators from the loaded engine
- [ ] Per-actuator slider (range: actuator limits) with numeric display
- [ ] Control type selector per actuator (constant / polynomial)
- [ ] "Reset All" button zeros all controls
- [ ] "Random Pose" button sets random valid joint positions
- [ ] Changes sent to backend via WebSocket `set_control` action
- [ ] Panel integrated into Simulation page left sidebar, collapsible
- [ ] Simplified mode when >20 actuators (grouped by body region)

### Implementation Notes

- Backend must expose actuator metadata (names, limits, types) via `/api/engines/{name}/actuators` REST endpoint
- Control updates sent as `{ action: 'set_control', actuator: 'name', value: N }` over existing WS
- Reference: `controls_tab.py:_setup_ui()` for layout, `control_system.py` for control types

### Files to Create/Modify

- `ui/src/api/client.ts` — add actuator metadata types, `set_control` WS action
- `ui/src/components/simulation/JointControlPanel.tsx` — new component
- `ui/src/components/simulation/ActuatorSlider.tsx` — reusable slider row
- `ui/src/pages/Simulation.tsx` — integrate panel into sidebar

---

## Issue 3: Add Model Explorer / Frankenstein Editor to React UI

**Labels:** `enhancement`, `react-ui`, `model-explorer`, `medium-priority`

### Description

The PyQt6 codebase has a full Model Explorer (`model_explorer/main_window.py`) with a Frankenstein Editor (`frankenstein_editor.py:566-1109`) for side-by-side URDF editing, component copying, and model merging. The React `LauncherDashboard.tsx` shows "Model Explorer" as a launcher tile but has no actual component behind it.

### Current State (PyQt6)

- `URDFGeneratorWindow`: Dockable panels, segment panel, visualization widget, model library
- `FrankensteinEditor`: Side-by-side URDF trees with context menus
  - Copy Selected Component, Copy Link Chain (recursive), Merge All
  - Swap Models, Replace Subtree, Show Diff dialog
- `ModelLibrary`: Preset human models (MuJoCo humanoid default)
- `SegmentPanel`: Add/edit/remove body segments with properties
- `VisualizationWidget`: 3D preview of assembled model

### Current State (React)

`LauncherDashboard.tsx` renders a tile with label "Model Explorer" and status chip. Clicking it does nothing meaningful — no editor, no URDF tree, no 3D preview.

### Acceptance Criteria

- [ ] New `/model-explorer` route and `ModelExplorerPage` component
- [ ] URDF tree viewer showing links, joints, materials hierarchy
- [ ] Side-by-side dual-tree layout for Frankenstein editing
- [ ] Context menu: Copy Component, Copy Chain, Merge, Replace Subtree
- [ ] 3D URDF preview using Three.js (meshes loaded from backend)
- [ ] Model library sidebar with preset models
- [ ] Export assembled URDF to file

### Implementation Notes

- URDF parsing can be done client-side with a lightweight XML parser or via backend `/api/models/{id}/urdf` endpoint
- Three.js URDF loader: `urdf-loader` npm package can render URDF+meshes
- Reference: `frankenstein_editor.py` for all edit operations, `urdf_builder.py` for URDF generation

### Files to Create/Modify

- `ui/src/pages/ModelExplorer.tsx` — new page
- `ui/src/components/model-explorer/URDFTreeView.tsx` — tree component
- `ui/src/components/model-explorer/FrankensteinEditor.tsx` — dual editor
- `ui/src/components/model-explorer/ModelLibrary.tsx` — preset sidebar
- `ui/src/components/model-explorer/URDFPreview3D.tsx` — Three.js URDF viewer
- `ui/src/api/models.ts` — model/URDF API client

---

## Issue 4: Add Character Builder to React UI

**Labels:** `enhancement`, `react-ui`, `model-explorer`, `medium-priority`

### Description

The standalone `humanoid_character_builder` module provides video game-style character customization with anthropometry-based body generation, mesh creation, and URDF export. No React equivalent exists.

### Current State (PyQt6/Python)

- `CharacterBuilder` class: accepts `BodyParameters` (height, mass, build type)
- `BodyParameters`: height_m, mass_kg, build_type ("athletic", "average", "heavy", etc.)
- `AppearanceParameters`: visual customization
- `AnthropometryData`: segment length/mass ratios from biomechanics literature
- `MeshGenerator`: generates STL meshes per segment
- URDF export with computed inertias (`InertiaMode.MESH_UNIFORM_DENSITY`)

### Current State (React)

Nothing. No character builder UI exists.

### Acceptance Criteria

- [ ] New `CharacterBuilder` component accessible from Model Explorer or as standalone page
- [ ] Sliders for height (1.50-2.10m), mass (40-150kg)
- [ ] Build type selector (dropdown or visual cards): Athletic, Average, Heavy, Slim
- [ ] Real-time 3D preview updating as parameters change
- [ ] Segment breakdown panel showing computed segment masses and lengths
- [ ] "Generate URDF" button that calls backend and returns downloadable file
- [ ] Optional: appearance parameters (colors/textures per segment)

### Implementation Notes

- Backend endpoint: `POST /api/character-builder/generate` with `BodyParameters` JSON body, returns URDF + mesh bundle
- Preview can use simple capsule/sphere geometries matching the anthropometry ratios without full mesh generation
- Reference: `humanoid_character_builder/__init__.py` for API, `core/body_parameters.py` for parameter types

### Files to Create/Modify

- `ui/src/pages/CharacterBuilder.tsx` or embed in ModelExplorer
- `ui/src/components/character-builder/BodyParameterControls.tsx` — sliders/selectors
- `ui/src/components/character-builder/CharacterPreview3D.tsx` — live 3D preview
- `ui/src/components/character-builder/SegmentBreakdown.tsx` — data table
- `ui/src/api/characterBuilder.ts` — API client

---

## Issue 5: Add Putting Green Simulation GUI to React UI

**Labels:** `enhancement`, `react-ui`, `putting-green`, `medium-priority`

### Description

A complete `PuttingGreenSimulator` physics engine exists (804 lines, `putting_green/python/simulator.py`) with ball roll physics, turf properties, wind effects, aim assist, and practice mode. No React GUI exists to interact with this engine.

### Current State (Python Backend)

- `PuttingGreenSimulator`: Full `PhysicsEngine` protocol implementation
- Ball roll with stimp rating, grass type, heightmap/topographical data
- Putter stroke simulation with speed, direction, face angle, attack angle
- Wind effects (configurable speed + direction)
- Practice mode with feedback ("hit firmer", "check aim line")
- Scatter analysis (Monte Carlo)
- Aim assist / green reading (break calculation, recommended speed)
- Trajectory recording and export

### Current State (React)

Nothing. The engine exists in the launcher manifest but has no dedicated GUI.

### Acceptance Criteria

- [ ] New `/putting-green` route and `PuttingGreenPage` component
- [ ] Top-down 2D green view (Canvas or SVG) showing: green surface, hole, ball, slope regions
- [ ] Click-to-place ball position
- [ ] Stroke controls: speed slider, direction (click to aim or drag angle), face angle
- [ ] "Putt" button triggers simulation, trajectory drawn as animated line
- [ ] Result display: distance from hole, holed/missed, total roll distance
- [ ] Wind controls: speed + direction selector
- [ ] Aim assist overlay: recommended aim point, break line
- [ ] Practice mode toggle with feedback messages
- [ ] Optional: 3D green view using heightmap

### Implementation Notes

- Backend: Putting green already implements `PhysicsEngine` protocol, so it should integrate with the existing WS simulation endpoint
- Green rendering: 2D `<canvas>` with gradient coloring for slope regions is simplest
- Reference: `simulator.py:simulate_putt()` for the core loop, `compute_aim_line()` for aim assist, `simulate_with_feedback()` for practice mode

### Files to Create/Modify

- `ui/src/pages/PuttingGreen.tsx` — new page
- `ui/src/components/putting-green/GreenCanvas.tsx` — 2D green renderer
- `ui/src/components/putting-green/StrokeControls.tsx` — input controls
- `ui/src/components/putting-green/WindControls.tsx` — wind settings
- `ui/src/components/putting-green/ResultsPanel.tsx` — outcome display
- `ui/src/api/puttingGreen.ts` — specialized API client

---

## Issue 6: Add 6-DOF Model Manipulation to React 3D Scene

**Labels:** `enhancement`, `react-ui`, `visualization`, `medium-priority`

### Description

The PyQt6 MuJoCo GUI (`interactive_manipulation.py`) provides full 6-DOF model manipulation with mouse ray-casting, IK-based drag, body constraints, and pose library. The React `Scene3D.tsx` only has camera `OrbitControls` — users cannot click, select, or drag model bodies.

### Current State (PyQt6)

- `MousePickingRay`: Ray-cast from screen click to select bodies in 3D
- `InteractiveManipulator`: IK-based drag to reposition selected body
- `BodyConstraint`: Fix body in space or relative to another body
- `StoredPose`: Save, load, interpolate between named poses
- Visual feedback: highlight selected body, show constraint markers

### Current State (React)

`Scene3D.tsx` uses `OrbitControls` for camera orbit/zoom/pan only. No object selection or manipulation.

### Acceptance Criteria

- [ ] Click on model body to select it (raycasting via Three.js `Raycaster`)
- [ ] Selected body highlighted with outline or color change
- [ ] TransformControls gizmo (translate + rotate) appears on selected body
- [ ] Gizmo drag sends updated position/orientation to backend
- [ ] Context menu on right-click: "Fix in Space", "Release Constraint", "Reset Pose"
- [ ] Pose library panel: Save Pose, Load Pose, named pose list
- [ ] Body info tooltip on hover (name, mass, joint type)

### Implementation Notes

- Three.js `TransformControls` from `@react-three/drei` provides the translate/rotate gizmo
- Raycasting: `useFrame` + `pointer` from R3F for hover detection, `onClick` for selection
- Backend: `POST /api/simulation/set_state` to apply position changes
- Reference: `interactive_manipulation.py:MousePickingRay` for ray-cast logic, `InteractiveManipulator` for IK drag

### Files to Create/Modify

- `ui/src/components/visualization/ModelInteraction.tsx` — click/drag/select logic
- `ui/src/components/visualization/TransformGizmo.tsx` — gizmo wrapper
- `ui/src/components/visualization/PoseLibrary.tsx` — save/load UI
- `ui/src/components/visualization/Scene3D.tsx` — integrate interaction
- `ui/src/api/client.ts` — add `set_state` and `set_constraint` WS actions

---

## Issue 7: Add Full Simulation Control Panel to React UI

**Labels:** `enhancement`, `react-ui`, `controls`, `high-priority`

### Description

The PyQt6 MuJoCo Advanced GUI has multi-tabbed, dockable control panels (ControlsTab, VisualizationTab) that are interactive during simulation. The React UI only has Start/Pause/Stop buttons and basic parameter inputs.

### Current State (PyQt6)

- **ControlsTab**: Play/Pause/Step/Reset, speed control, per-actuator sliders, control type selection, polynomial generator, quick camera buttons, timestep adjustment, model reload
- **VisualizationTab**: Camera presets (side/front/top/follow/down-the-line), azimuth/elevation/distance sliders, swing trajectory recording, force/torque toggles, background color, rendering quality
- **Advanced panels**: Analysis tab with biomechanics data, export controls

### Current State (React)

- `SimulationControls.tsx`: Start/Pause/Resume/Stop buttons only
- `ParameterPanel.tsx`: Duration, timestep dropdown, live analysis toggle, GPU toggle
- No camera controls, no trajectory recording, no speed control, no model management

### Acceptance Criteria

- [ ] Tabbed or accordion control panel in the left sidebar with sections:
  - **Simulation**: Play/Pause/Step/Reset, speed slider (0.1x-5x), timestep adjustment
  - **Camera**: Preset buttons (side/front/top/follow), azimuth/elevation/distance sliders
  - **Visualization**: Force/torque toggles (from Issue 1), trajectory trail toggle, background color
  - **Actuators**: Joint sliders (from Issue 2)
- [ ] Controls remain interactive while simulation is running
- [ ] "Step" button advances exactly one frame when paused
- [ ] Speed slider adjusts simulation playback rate without restarting
- [ ] Camera presets snap the OrbitControls to defined angles
- [ ] Trajectory trail toggle shows/hides the club trajectory line
- [ ] Panel state persists across page navigation (localStorage or URL params)

### Implementation Notes

- Camera preset positions can be defined as `{ azimuth, elevation, distance, target }` objects applied to OrbitControls via ref
- Speed control: send `{ action: 'set_speed', factor: N }` over WS
- Step control: send `{ action: 'step' }` over WS
- Reference: `controls_tab.py` for layout, `visualization_tab.py` for camera presets

### Files to Create/Modify

- `ui/src/components/simulation/ControlPanel.tsx` — tabbed container
- `ui/src/components/simulation/CameraControls.tsx` — presets + sliders
- `ui/src/components/simulation/SpeedControl.tsx` — playback speed
- `ui/src/components/simulation/SimulationControls.tsx` — extend with Step/Reset
- `ui/src/pages/Simulation.tsx` — replace current sidebar with ControlPanel

---

## Issue 8: Archive Old r0 GUI Versions and Backup Directories

**Labels:** `cleanup`, `tech-debt`, `low-priority`

### Description

Several old, incomplete, or explicitly-labeled backup directories remain in the main source tree. These duplicate functionality now living in the current engine GUIs and add confusion about which code is authoritative.

### Directories to Archive

| Current Path | Evidence |
|---|---|
| `Simscape.../Python Version/golf_gui_r0/` | Revision 0 — 5-file old GUI with `golf_main_application.py`, `golf_visualizer_implementation.py`. Superseded by current MuJoCo/Drake GUIs. |
| `Simscape.../Python Version/integrated_golf_gui_r0/` | Revision 0 — 15+ files including `golf_gui_application.py`, `golf_opengl_renderer.py`, `golf_inverse_dynamics.py`, test files. |
| `2D_Golf_Model/matlab/Backup Folder/` | Explicitly named "Backup Folder" containing `ModelInputsZVCF.mat`. |
| `SkeletonPlotter/Older Revs/` | Explicitly labeled "Older Revs". |
| `Model Output/Scripts/_Archived Scripts/` | Already labeled "Archived" but still in active source tree. |
| `3D_Golf_Model/golf_swing_dataset_20250907_bk` | `_bk` suffix = backup of dataset. |
| `2D_Golf_Model/matlab/Scripts/` vs `Model Output/Scripts/` | Near-duplicate directory trees with same subfolder names. |

### Acceptance Criteria

- [ ] All listed directories moved under `archive/` (project root) or `docs/archive/` with README explaining provenance
- [ ] No import paths or references in active code break after move (search for imports)
- [ ] Duplicated MATLAB Script directories consolidated (keep one, archive the other)
- [ ] `archive/README.md` updated with an entry per archived directory explaining what it was and when it was archived

### Implementation Notes

- Run `grep -r "golf_gui_r0\|integrated_golf_gui_r0\|Backup Folder\|Older Revs" --include="*.py" --include="*.m"` to verify no active imports
- These are mostly Python and MATLAB files; no React/TypeScript code references them
