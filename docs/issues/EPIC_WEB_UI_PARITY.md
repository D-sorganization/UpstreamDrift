# Epic: UpstreamDrift Web/Tauri to PyQt6 Feature Parity

## Overview

The `UpstreamDrift` PyQt6 desktop app contains an advanced, highly polished feature set and dynamic `ThemeManager`. The React/Tauri frontend currently lags behind in both theming consistency and core simulation controls. This epic organizes the detailed subtasks required to achieve 100% parity between the desktop and web versions.

## Subtasks

### 1. Unified Theme Bridging (Foundation)

- [x] **Status:** Completed/In-Progress
- [ ] **Objective:** Ensure the React UI dynamically mirrors the active PyQt6 theme.
- [ ] **Implementation Details:**
  - [x] Expose the `ThemeManager` via a FastAPI REST endpoint (`/api/v1/themes/active`).
  - [x] Fetch the active theme on React app mount and inject the colors as CSS custom properties (e.g., `--theme-bg`).
  - [x] Configure `tailwind.config.js` to map these variables to Tailwind utility classes (`bg-theme-bg`).

### 2. Component Theme Refactoring

- [x] **Status:** Completed
- [x] **Objective:** Strip hardcoded colors from the React UI and apply the dynamic theme tokens.
- [x] **Implementation Details:**
  - [x] Refactor `LauncherDashboard.tsx`, replacing `bg-gray-900` with `bg-theme-bg`, `text-blue-300` with `text-theme-accent`, etc.
  - [x] Refactor the Sidebar, Navigation, Simulation Controls, Actuator Panels, and all other components to use theme classes.
  - [x] Ensure all hover states utilize `--theme-button-hover` and focus rings utilize `--theme-focus`.

### 3. Add Force & Torque Vector Overlays to 3D Scene

- [x] **Status:** Completed
- [x] **Objective:** Replicate the PyQt6 physics visualization in the React `Scene3D.tsx`.
- [x] **Implementation Details:**
  - [x] Extend the WebSocket `SimulationFrame` payload to include `forces` and `torques` arrays.
  - [x] Implement a Three.js `ForceArrows` component to render 3D arrow helpers at joint positions.
  - [x] Add UI toggles for "Show Torques", "Show Forces", and a scale slider.

### 4. Implement Full Simulation Control Panel

- [x] **Status:** Completed
- [x] **Objective:** Bring the advanced dockable control panels from PyQt6 to the web UI.
- [x] **Implementation Details:**
  - [x] Build a tabbed control panel in the left sidebar for Simulation, Camera, Visualization, and Actuators.
  - [x] Implement a Playback Speed slider (0.1x - 5x) that communicates via WebSocket.
  - [x] Add camera preset buttons (Side, Front, Top, Follow).

### 5. Joint and Actuator Controls

- [x] **Status:** Completed
- [x] **Objective:** Allow per-actuator manipulation identical to the PyQt6 `controls_tab.py`.
- [x] **Implementation Details:**
  - [x] Create a `JointControlPanel` component that lists all actuators dynamically via REST API metadata.
  - [x] Provide a slider per actuator bounded by physical limits.
  - [x] Support "Reset All" and "Random Pose" actions.

### 6. Model Explorer / Frankenstein Editor

- [ ] **Objective:** Port the PyQt6 URDF Builder and Frankenstein Editor to the web.
- [ ] **Implementation Details:**
  - Implement a dual-pane URDF tree viewer for side-by-side editing.
  - Support drag-and-drop or context-menu operations to "Copy Subtree" or "Merge Models".
  - [x] Provide a live 3D URDF preview using `urdf-loader` for Three.js.
