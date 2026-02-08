# Dual GUI Architecture Review: PyQt6 + React/Tauri

## Executive Summary

**Is it practical?** Yes, but with significant caveats. This codebase is already
structured in a way that makes dual-GUI maintenance feasible — the FastAPI backend,
`PhysicsEngine` protocol, and `EngineManager` provide a clean separation between
business logic and presentation. However, the current PyQt6 GUIs embed substantial
domain logic (plotting, analysis orchestration, simulation control) directly in
widget code, which would need to be extracted into shared services before the two
frontends could truly share features.

---

## 1. Current Architecture Inventory

### 1.1 React/Tauri Frontend (`ui/`)

| Component | File | Function |
|-----------|------|----------|
| App shell | `ui/src/App.tsx` | BrowserRouter with Dashboard + Simulation routes |
| Dashboard | `ui/src/pages/Dashboard.tsx` | Launcher tile grid, navigates to simulation |
| Simulation | `ui/src/pages/Simulation.tsx` | Engine selector, parameter panel, 3D view, live plot |
| API client | `ui/src/api/client.ts` | WebSocket simulation streaming, REST engine queries |
| Engine hook | `ui/src/api/useEngineManager.ts` | Lazy engine load/unload via REST |
| Backend mgmt | `ui/src/api/backend.ts` | Tauri IPC to start/stop Python backend |
| 3D scene | `ui/src/components/visualization/Scene3D.tsx` | Three.js/R3F visualization |
| Live plot | `ui/src/components/analysis/LivePlot.tsx` | Recharts real-time plotting |
| Tauri shell | `ui/src-tauri/src/lib.rs` | Rust process manager for Python backend |

**Stack:** React 19, React Router 7, TanStack Query, Three.js (R3F), Recharts,
Tailwind CSS, Vite 7, Tauri 2 (Rust).

### 1.2 PyQt6 Desktop GUIs (Multiple)

There is no single PyQt6 "app" — there are **several independent PyQt6 GUIs**, each
tailored to a specific engine or tool:

| GUI | File | Scope |
|-----|------|-------|
| MuJoCo Analysis | `src/engines/.../mujoco/.../gui/core/main_window.py` | 9 tabs: Physics, Controls, Visualization, Analysis, Plotting, Live Analysis, Interactive Pose, Manipulability, Grip Modelling |
| Unified Dashboard | `src/shared/python/dashboard/window.py` | Generic: Control Panel, Live Plot, Plotting tab, Counterfactuals tab, Export tab |
| Drake GUI | `src/engines/.../drake/python/src/drake_gui_app.py` | Drake-specific simulation + MeshCat |
| Pinocchio GUI | `src/engines/.../pinocchio/python/dtack/gui/main_window.py` | Pinocchio-specific dynamics |
| Pendulum GUI | `src/engines/.../pendulum_models/.../ui/pendulum_pyqt_app.py` | Double/triple pendulum simulation |
| C3D Viewer | `src/engines/.../3D_Golf_Model/python/src/apps/c3d_viewer.py` | Motion capture data viewer |
| OpenSim GUI | `src/engines/.../opensim/python/opensim_gui.py` | OpenSim wrapper |
| Pose GUIs | `src/shared/python/pose_estimation/mediapipe_gui.py`, `openpose_gui.py` | Video pose estimation |
| AI Assistant | `src/shared/python/ai/gui/assistant_panel.py` | AI chat panel |
| Launcher | `src/launchers/golf_suite_launcher.py` | Main desktop launcher (tile grid) |

### 1.3 Shared Backend Infrastructure

| Layer | Key Files | Purpose |
|-------|-----------|---------|
| FastAPI server | `src/api/server.py` | REST + WebSocket API |
| Simulation routes | `src/api/routes/simulation.py` | POST /simulate, /simulate/async |
| WS streaming | `src/api/routes/simulation_ws.py` | Real-time WebSocket frame streaming |
| Engine routes | `src/api/routes/engines.py` | Engine discovery, load/unload |
| Analysis routes | `src/api/routes/analysis.py` | Biomechanical analysis |
| Video routes | `src/api/routes/video.py` | Video pose estimation |
| Export routes | `src/api/routes/export.py` | Data export |
| Request models | `src/api/models/requests.py` | Pydantic request schemas |
| Response models | `src/api/models/responses.py` | Pydantic response schemas |
| Engine Manager | `src/shared/python/engine_manager.py` | Engine discovery, loading, switching |
| PhysicsEngine | `src/shared/python/interfaces.py` | Abstract protocol all engines implement |
| Engine Registry | `src/shared/python/engine_registry.py` | Plugin-style engine registration |
| Launcher Manifest | `src/config/launcher_manifest.json` | Shared tile definitions for both UIs |
| GUI utils | `src/shared/python/gui_utils.py` | PyQt6 base classes, helpers |
| Shared UI widgets | `src/shared/python/ui/` | Reusable PyQt6 components |

### 1.4 Launch Modes (Already Dual)

The launcher (`launch_golf_suite.py`) already supports dual modes:

```
golf-suite              # Web UI (React/Tauri) — default
golf-suite --classic    # Classic PyQt6 desktop launcher
golf-suite --api-only   # API server only
golf-suite --engine X   # Direct engine launch
```

---

## 2. Gap Analysis: What Diverges Today

### 2.1 Feature Coverage Gap

| Feature | React UI | PyQt6 GUIs | Notes |
|---------|----------|------------|-------|
| Engine selection/loading | Yes (lazy) | Yes (tile launcher) | React is more mature |
| Real-time simulation | Yes (WebSocket) | Yes (QThread + timer) | Different transport |
| 3D visualization | Three.js/R3F | MuJoCo native / OpenGL | Fundamentally different renderers |
| Live plotting | Recharts (2D) | Matplotlib (2D) | Could share data format |
| Static analysis plots | Not yet | 20+ plot types | Major gap |
| Counterfactual analysis | Not yet | ZTCF/ZVCF, Induced Accel | Major gap |
| Data export | Not yet | CSV, MAT, HDF5, C3D | Major gap |
| Interactive pose editing | Not yet | Drag joints in MuJoCo | Hard to replicate |
| Grip modelling | Not yet | Full tab | Major gap |
| Camera controls | Basic orbit | Full preset system | Moderate gap |
| Video pose estimation | Via API only | Dedicated GUI | Different UX |
| Keyboard shortcuts | None | Full shortcut system | Easy to add |
| Recording/playback | Not yet | Full recorder + overlay | Moderate gap |

**Key finding:** The React UI covers ~30% of the PyQt6 feature set. The MuJoCo
`AdvancedGolfAnalysisWindow` alone has 9 specialized tabs vs the React UI's 2 pages.

### 2.2 Architectural Divergence

| Aspect | React UI | PyQt6 GUIs |
|--------|----------|------------|
| Data transport | HTTP/WebSocket to FastAPI | Direct Python object calls |
| Plotting | Recharts (client-side JS) | Matplotlib (server-side Python) |
| 3D rendering | Three.js (WebGL) | MuJoCo viewer (OpenGL) / VTK |
| State management | React hooks + TanStack Query | PyQt signals/slots |
| Layout system | Tailwind CSS flexbox | QSplitter + QTabWidget |
| Process model | Browser tab + API server | Single Python process |

---

## 3. Practical Strategy for Dual-GUI Maintenance

### 3.1 Recommended Architecture: Shared Service Layer

The key to maintaining both GUIs is to **extract all business logic into a shared
service layer** that both UIs consume — PyQt6 GUIs call services directly in-process,
while the React UI calls the same services via FastAPI endpoints.

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  React/Tauri │     │  FastAPI Server   │     │  PyQt6 Desktop   │
│    (Web UI)  │────>│  (REST + WS)     │     │  (Native UI)     │
└──────────────┘     └────────┬─────────┘     └────────┬─────────┘
                              │                        │
                              │ HTTP/WS                │ Direct Python
                              │                        │
                    ┌─────────▼────────────────────────▼──────────┐
                    │          Shared Service Layer                │
                    │  ┌──────────────────────────────────┐       │
                    │  │ SimulationService                 │       │
                    │  │ AnalysisService                   │       │
                    │  │ ExportService                     │       │
                    │  │ PlottingService (returns data)    │       │
                    │  │ RecordingService                  │       │
                    │  └──────────────────────────────────┘       │
                    │                                              │
                    │  ┌──────────────────────────────────┐       │
                    │  │ EngineManager + PhysicsEngine     │       │
                    │  │ Engine Registry + Loaders         │       │
                    │  └──────────────────────────────────┘       │
                    └─────────────────────────────────────────────┘
```

### 3.2 Concrete Steps

#### Step 1: Extract Analysis Logic from PyQt6 Widgets

Currently, `UnifiedDashboardWindow.refresh_static_plot()` (window.py:266-388)
contains 20+ plot type dispatch branches that call `GolfSwingPlotter` methods
directly. This logic should be extracted into an `AnalysisService` that returns
plot data (numpy arrays / dicts) rather than rendering directly to matplotlib.

**Before (PyQt6-coupled):**
```python
# In window.py — logic embedded in widget
def refresh_static_plot(self):
    if plot_type == "Joint Angles":
        self.plotter.plot_joint_angles(self.static_canvas.fig)
```

**After (shared service):**
```python
# In shared/python/services/analysis_service.py
class AnalysisOrchestrator:
    def get_plot_data(self, plot_type: str, recorder) -> PlotData:
        """Returns structured data, no rendering."""
        if plot_type == "joint_angles":
            return PlotData(series=recorder.get_time_series("joint_positions"))

# PyQt6 consumes:
data = self.analysis_orchestrator.get_plot_data("joint_angles", recorder)
self.plotter.render_to_canvas(data, self.static_canvas)

# React consumes (via API):
# GET /api/analysis/plot-data?type=joint_angles -> JSON
```

#### Step 2: Expose Missing Features via FastAPI

Add API routes for features currently only available in PyQt6:

| New Endpoint | Mirrors PyQt6 Feature |
|---|---|
| `GET /api/analysis/plot-data/{type}` | `UnifiedDashboardWindow.refresh_static_plot()` |
| `POST /api/analysis/counterfactual` | `compute_analysis()` in dashboard |
| `GET /api/export/formats` | `get_available_export_formats()` |
| `POST /api/export/{format}` | `export_data()` in dashboard |
| `POST /api/recording/start` | `recorder.start_recording()` |
| `POST /api/recording/stop` | `recorder.stop_recording()` |
| `GET /api/recording/frames` | `recorder.get_num_frames()` |
| `GET /api/engine/{type}/models` | Model configs from PhysicsTab |
| `POST /api/engine/{type}/camera` | Camera preset system |

#### Step 3: Shared Layout Specification

Use the existing `launcher_manifest.json` pattern to define UI layouts declaratively:

```json
{
  "simulation_page": {
    "layout": "sidebar-main-sidebar",
    "left_sidebar": {
      "sections": ["engine_selector", "parameter_panel", "simulation_controls"]
    },
    "main": {
      "sections": ["3d_visualization", "live_plot"]
    },
    "right_sidebar": {
      "sections": ["live_data"]
    }
  }
}
```

Both UIs read this manifest and render their respective implementations of each
section. The manifest ensures both UIs maintain the same logical structure.

#### Step 4: Shared Data Contracts (Already Partially Done)

The Pydantic models in `src/api/models/` already serve as the data contract.
Generate TypeScript types from them to keep both sides in sync:

```bash
# Generate TypeScript interfaces from Pydantic models
pip install pydantic-to-typescript
pydantic2ts --module src.api.models.requests --output ui/src/api/types.ts
pydantic2ts --module src.api.models.responses --output ui/src/api/types.ts --append
```

#### Step 5: Adopt a "Feature Flag" Approach for Gradual Parity

Track feature parity explicitly:

```python
# src/shared/python/feature_registry.py
FEATURES = {
    "simulation.basic": {"pyqt": True, "react": True},
    "simulation.recording": {"pyqt": True, "react": False},
    "analysis.joint_angles": {"pyqt": True, "react": False},
    "analysis.counterfactual": {"pyqt": True, "react": False},
    "export.csv": {"pyqt": True, "react": False},
    "visualization.3d": {"pyqt": True, "react": True},
    ...
}
```

---

## 4. What Works Well Today

1. **`PhysicsEngine` Protocol** (`src/shared/python/interfaces.py`) — All engines
   implement the same interface. Both GUIs can consume any engine identically.

2. **`EngineManager`** (`src/shared/python/engine_manager.py`) — Centralized engine
   lifecycle. Already used by both FastAPI and PyQt6 launchers.

3. **`launcher_manifest.json`** — Already declared as "single source of truth for
   all launcher tiles across PyQt and Tauri/React UIs." This pattern works and
   should be extended to other UI areas.

4. **FastAPI with Pydantic Models** — Clean typed contracts. The React UI already
   consumes these via `fetch()` and WebSocket.

5. **Dual Launch Modes** — `launch_golf_suite.py` already supports `--classic`
   (PyQt6) and default (web) modes.

---

## 5. Risks and Challenges

### 5.1 3D Visualization Parity

The PyQt6 MuJoCo GUI uses MuJoCo's native OpenGL renderer with per-frame access to
the simulation state, camera, and contact forces. The React UI uses Three.js with
R3F. These renderers are fundamentally different — achieving visual parity would
require either:

- Streaming rendered frames from the server (high bandwidth, high latency), or
- Reimplementing model loading and rendering in Three.js (significant effort), or
- Accepting that 3D views will look different (pragmatic choice)

**Recommendation:** Accept visual differences. Focus on **data parity** (same
numbers, same plots, same export), not pixel-perfect rendering parity.

### 5.2 Matplotlib vs Recharts

PyQt6 uses Matplotlib for 20+ sophisticated plot types including 3D plots, radar
charts, Poincare maps, and recurrence plots. Recharts cannot replicate all of these.
Options:

- **Server-side rendering:** Generate plot images (PNG/SVG) server-side with
  Matplotlib and serve them to React via API. Simplest path to parity.
- **Plotly.js:** Richer than Recharts, handles 3D plots, radar charts, heatmaps.
  Would cover ~80% of plot types.
- **Hybrid:** Use Recharts for simple live plots, Plotly for complex analysis plots,
  and server-rendered images for exotic plots (Poincare, recurrence).

### 5.3 Maintenance Burden

Maintaining two frontends means every new feature must be implemented twice. Mitigate
by:

- Making the service layer the **primary development target** — UI changes are just
  wiring.
- Using the layout manifest to enforce structural consistency.
- Running automated tests against the API (which both UIs consume).
- Accepting that PyQt6 may lead in specialized features (e.g., interactive pose
  editing) while React leads in accessibility and deployment.

### 5.4 Testing Strategy

| Layer | Testing Approach |
|-------|------------------|
| Shared services | Unit tests (pytest) — engine-agnostic |
| API endpoints | Integration tests (httpx/TestClient) |
| React UI | Vitest + Testing Library (already exists) |
| PyQt6 UI | pytest-qt (partially exists) |
| Cross-GUI parity | API contract tests (same inputs -> same outputs) |

---

## 6. Recommendation

**Phase 1 (Low effort, high value):** Extract analysis/plotting logic from PyQt6
widgets into shared services. Expose via new API endpoints. This benefits PyQt6 too
(cleaner code).

**Phase 2 (Medium effort):** Add server-side plot rendering endpoint. React UI gains
all 20+ plot types by displaying rendered images. PyQt6 can optionally use the same
endpoint or continue with direct Matplotlib.

**Phase 3 (Higher effort):** Build React equivalents of the most-used PyQt6 tabs
(Controls, Analysis, Export) using the shared API. Use Plotly.js for interactive
client-side plots where needed.

**Long-term posture:** The React/Tauri UI becomes the primary user-facing
application (cross-platform, web-deployable). The PyQt6 GUIs remain as
power-user/developer tools for engine-specific work that benefits from direct
in-process access (interactive pose editing, real-time MuJoCo rendering, MATLAB
integration).

---

## 7. Files to Modify (Phase 1)

| Action | File | Change |
|--------|------|--------|
| Extract | `src/shared/python/dashboard/window.py` | Move plot dispatch to new service |
| Create | `src/shared/python/services/plotting_orchestrator.py` | Plot data generation (no rendering) |
| Create | `src/api/routes/plots.py` | `GET /api/plots/{type}` endpoint |
| Extend | `src/api/routes/analysis.py` | Counterfactual analysis endpoint |
| Extend | `src/api/routes/export.py` | Multi-format export endpoint |
| Create | `ui/src/pages/Analysis.tsx` | New Analysis page in React |
| Create | `ui/src/components/analysis/PlotViewer.tsx` | Server-rendered plot display |
| Update | `ui/src/App.tsx` | Add `/analysis` route |
| Extend | `src/config/launcher_manifest.json` | Add layout specs for new pages |
