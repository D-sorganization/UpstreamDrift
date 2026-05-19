# UpstreamDrift — Project Map

> **Complete guide to every feature, module, and integration in the UpstreamDrift platform.**
> Use this document to navigate the launcher tiles, understand engine capabilities, and discover the tools available.

---

## Launcher Overview

The UpstreamDrift Launcher is a PyQt6 desktop application that provides a **unified tile-based interface** for accessing all physics engines, analysis tools, and utilities in the platform.

### Quick Reference

| Shortcut              | Action                     |
| --------------------- | -------------------------- |
| `F1`                  | Help dialog                |
| `Ctrl+?`              | Keyboard shortcuts overlay |
| `Ctrl+,`              | Preferences                |
| `Ctrl+=` / `Ctrl+-`   | Zoom in / out              |
| `Ctrl+Q`              | Quit                       |
| Double-click tile     | Launch application         |
| Drag tile (edit mode) | Reorder tiles              |

### View Modes

| Mode        | Layout    | Description                        |
| ----------- | --------- | ---------------------------------- |
| Comfortable | 4 columns | Large tiles with full descriptions |
| Compact     | 6 columns | Medium tiles with descriptions     |
| Dense       | 8 columns | Small tiles, no descriptions       |
| List        | 1 column  | Full-width rows with descriptions  |

### Runtime Modes

| Mode               | Description                                                                                   |
| ------------------ | --------------------------------------------------------------------------------------------- |
| **Native Windows** | Default. Engines run directly on the host.                                                    |
| **Docker**         | Engines run inside the `upstream-drift:engine` Linux container. Full Drake/Pinocchio support. |
| **WSL2**           | Engines run in your WSL2 Ubuntu environment. Faster file I/O, easier debugging.               |

---

## Physics Engines

### Tier 1 — Supported (Required CI)

#### MuJoCo

- **Tile:** MuJoCo
- **Type:** `custom_humanoid`
- **Features:** Contact dynamics, 2-28 DOF models, flexible shafts, humanoid configuration, passive/active viewers
- **Entry:** `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/__main__.py`

### Tier 2 — Extended (Nightly Validation)

#### Drake

- **Tile:** Drake
- **Type:** `drake`
- **Features:** Trajectory optimization, MeshCat visualization, contact modeling, system analysis
- **Entry:** `src/engines/physics_engines/drake/python/src/drake_gui_app.py`

#### Pinocchio

- **Tile:** Pinocchio
- **Type:** `pinocchio`
- **Features:** High-performance rigid body dynamics, analytical derivatives (Jacobians), PINK IK, constrained systems
- **Entry:** `src/engines/physics_engines/pinocchio/python/pinocchio_golf/gui.py`

### Tier 3 — Experimental (Best-Effort)

#### OpenSim

- **Tile:** OpenSim
- **Type:** `opensim`
- **Features:** Musculoskeletal modeling, biomechanics validation
- **Entry:** `src/engines/physics_engines/opensim/python/opensim_gui.py`

#### MyoSuite

- **Tile:** MyoSuite
- **Type:** `myosim`
- **Features:** Muscle-actuated simulation, reinforcement learning environments
- **Entry:** `src/engines/physics_engines/myosuite/python/myosuite_physics_engine.py`

### Specialized Engines

#### Putting Green

- **Tile:** Putting Green
- **Type:** `putting_green`
- **Features:** Realistic putting physics, ball rolling dynamics, green contours
- **Entry:** `src/engines/physics_engines/putting_green/python/simulator.py`

---

## Analysis Tools

### Motion Capture Pipeline

#### C3D Viewer

- **Tile:** C3D Viewer
- **Features:** Visualize and analyze optical motion capture data (.c3d format)
- **Entry:** Searches multiple candidate paths (in-repo Simscape 3D, fleet vendor, legacy tools)

#### Motion-Match Preview

- **Tile:** Motion-Match Preview
- **Features:** Preview multi-source motion-capture targets (C3D, .mat, body markers, club/ball) on a shared time grid before driving a physics-engine model
- **Tags:** c3d, mocap, club, body, preview

#### OpenPose

- **Tile:** OpenPose
- **Features:** High-accuracy body pose estimation (Academic License)
- **Entry:** `src/shared/python/pose_estimation/openpose_gui.py`

#### MediaPipe

- **Tile:** MediaPipe
- **Features:** Fast pose estimation using Google MediaPipe (Apache 2.0)
- **Entry:** `src/shared/python/pose_estimation/mediapipe_gui.py`

### Visualization & Data

#### Video Analyzer

- **Tile:** Video Analyzer
- **Features:** Video-based motion analysis with pose estimation and tracking
- **Entry:** `src/tools/video_analyzer/launch_video_analyzer.py`

#### Data Explorer

- **Tile:** Data Explorer
- **Features:** Import, filter, and visualize simulation datasets
- **Entry:** `src/tools/data_explorer/data_explorer_app.py`

#### Model Explorer

- **Tile:** Model Explorer
- **Features:** Interactive URDF browser for humanoid, pendulum, and robotic models. Generate, import, and export URDFs. Frankenstein mode for combining models.
- **Entry:** `src/tools/model_explorer/launch_model_explorer.py`

### Cross-Engine Validation

#### Cross-Engine Dashboard

- **Access:** Tools menu or direct launch
- **Features:** Monte Carlo perturbation comparison across all engines, robustness scoring, CV analysis, trajectory overlay visualization
- **Entry:** `src/launchers/cross_engine_dashboard.py`

---

## MATLAB / Simscape Integration

#### Matlab Simscape Models

- **Tile:** Matlab Simscape Models
- **Type:** `matlab_suite`
- **Features:** Suite of MATLAB/Simscape models, dataset generators, and analysis tools. Opens the MATLAB Suite Dialog for sub-model selection.
- **Requirements:** MATLAB R2023a+ with Simulink and Simscape Multibody

---

## Launcher Features

### Theme System

Three built-in themes with live-switching:

- **Dark** (default) — Deep dark backgrounds with glassmorphism cards
- **Light** — Clean white backgrounds
- **High Contrast** — Accessibility-optimized high contrast

Custom Matplotlib plot themes can be configured via the Theme menu → Plot Theme submenu.

### AI Assistant

- Integrated chat panel (toggle via toolbar button)
- Supports Ollama (local/free), OpenAI, Anthropic, and Google Gemini
- RAG-powered codebase awareness for context-sensitive help
- Configurable expertise levels (Beginner → Expert)

### Layout Customization

- **Drag-and-drop** tile reordering (enable via Layout Lock toggle)
- **Tile visibility** management (Edit Tiles dialog)
- **Zoom slider** for tile size adjustment
- **Persistent layout** saved to `~/.golf_modeling_suite/launcher_layout.json`

### Toast Notifications

Non-intrusive overlay notifications for launch events, errors, and status updates. Types: success, error, warning, info.

### Process Management

- Centralized process manager tracks all launched simulations
- Process Output Console (dockable) shows real-time stdout/stderr
- Automatic cleanup on launcher exit

### Diagnostics

- **Settings → Diagnostics tab:** System health checks, engine availability, runtime state
- **Application Log viewer:** Recent app log entries
- **Process Output Log:** Recent subprocess output
- Re-run diagnostics on demand

### Docker Image Building

- **Settings → Configuration → Docker Image:** Build or rebuild the `upstream-drift:engine` container
- Target stage selection (all, mujoco, pinocchio, drake, base)
- Real-time build output streaming
- Cancel support

---

## Architecture Quick Reference

```
UpstreamDrift/
├── src/
│   ├── launchers/           # PyQt6 launcher application
│   │   ├── upstream_drift_launcher.py         # Main window (QMainWindow + mixins)
│   │   ├── launcher_ui_setup.py     # UI construction mixin
│   │   ├── launcher_theme.py        # Theme management mixin
│   │   ├── launcher_simulation.py   # Simulation launch mixin
│   │   ├── launcher_dialogs.py      # Dialog management mixin
│   │   ├── launcher_layout_manager.py # Grid layout persistence
│   │   ├── model_card.py            # Glassmorphic tile widgets
│   │   ├── startup.py               # Splash screen & async init
│   │   ├── settings_dialog.py       # Settings (Layout/Config/Diagnostics)
│   │   ├── cross_engine_dashboard.py # Cross-engine comparison UI
│   │   └── about_dialog.py          # Version info & about
│   ├── engines/
│   │   ├── physics_engines/         # Python physics engine implementations
│   │   ├── Simscape_Multibody_Models/ # MATLAB/Simulink models
│   │   └── pendulum_models/         # Educational pendulum models
│   ├── shared/                      # Cross-engine shared code
│   │   └── python/
│   │       ├── ai/                  # AI assistant integration
│   │       ├── engine_core/         # Engine manager & registry
│   │       ├── config/              # Model registry
│   │       ├── gui_pkg/             # Shared GUI widgets
│   │       ├── theme/               # Design tokens & style constants
│   │       └── ui/                  # Toast, shortcuts, preferences
│   ├── tools/                       # Analysis tools
│   │   ├── model_explorer/          # URDF browser & editor
│   │   ├── video_analyzer/          # Video-based analysis
│   │   └── data_explorer/           # Dataset visualization
│   ├── config/
│   │   └── models.yaml              # Model registry definition
│   └── api/                         # REST/WebSocket API server
├── ui/                              # Tauri/Vite web frontend (experimental)
├── rust_core/                       # Rust simulation kernel (PyO3)
└── docs/                            # Documentation hub
```

---

## Configuration Files

| File                 | Location                          | Purpose                                |
| -------------------- | --------------------------------- | -------------------------------------- |
| `models.yaml`        | `src/config/`                     | Model registry (all launcher tiles)    |
| `layout.json`        | `~/.golf_modeling_suite/`         | Persisted tile order & window geometry |
| `app_launch.log`     | `./` or `~/.golf_modeling_suite/` | Application log                        |
| `process_output.log` | `~/.golf_modeling_suite/`         | Subprocess output capture              |

---

## Getting Started

1. **Install:** `pip install -e ".[dev]"`
2. **Launch:** `python launch_golf_suite.py`
3. **Select a model** from the tile grid (MuJoCo recommended for first-time users)
4. **Double-click** to launch, or select and click the Launch button
5. Press **F1** for help or **Ctrl+?** for keyboard shortcuts

For detailed installation and engine setup, see the [User Guide](user_guide/README.md).
