# Dual GUI Implementation Plan: Complete Issue Breakdown

This document defines every GitHub issue needed to implement the dual-GUI
architecture (PyQt6 + React/Tauri) with shared backend services. Issues are
organized into 4 phases with explicit dependencies. Each issue includes the
full body text ready to paste into GitHub.

## Dependency Graph

```
Phase 0: Foundation (must be done first)
  #0.1 Pydantic TypeScript codegen
  #0.2 Feature parity tracker
  #0.3 Shared layout manifest schema

Phase 1: Extract Shared Services (the critical path)
  #1.1 PlottingOrchestrator service        ← depends on nothing
  #1.2 RecordingService                    ← depends on nothing
  #1.3 CounterfactualAnalysisService       ← depends on nothing
  #1.4 ExportService (enhance existing)    ← depends on nothing
  #1.5 Refactor dashboard/window.py        ← depends on #1.1, #1.2, #1.3, #1.4
  #1.6 Refactor MuJoCo main_window.py      ← depends on #1.1, #1.2

Phase 2: API Endpoints
  #2.1 Plot data + rendering endpoints     ← depends on #1.1
  #2.2 Recording REST endpoints            ← depends on #1.2
  #2.3 Counterfactual analysis endpoint    ← depends on #1.3
  #2.4 Multi-format export endpoints       ← depends on #1.4
  #2.5 Model config / camera endpoints     ← depends on nothing
  #2.6 API contract test suite             ← depends on #2.1-#2.5

Phase 3: React UI Feature Parity
  #3.1 Analysis page + PlotViewer          ← depends on #2.1
  #3.2 Recording controls component        ← depends on #2.2
  #3.3 Export page                         ← depends on #2.4
  #3.4 Simulation controls enhancement     ← depends on #2.5
  #3.5 Keyboard shortcuts                  ← depends on nothing
  #3.6 Camera presets in 3D view           ← depends on #2.5
  #3.7 Plotly.js interactive plots         ← depends on #2.1

Phase 4: Polish & Parity
  #4.1 Cross-GUI integration tests         ← depends on Phase 2, Phase 3
  #4.2 Layout manifest for all pages       ← depends on #0.3, Phase 3
  #4.3 Documentation: dual-GUI dev guide   ← depends on Phase 1
```

---

## Phase 0: Foundation

---

### Issue #0.1 — Auto-generate TypeScript types from Pydantic models

**Labels:** `enhancement`, `dual-gui`, `phase-0`, `dx`

```markdown
## Summary

Set up automated TypeScript type generation from Python Pydantic models so
the React frontend always has accurate, up-to-date type definitions that
match the API contracts.

## Context

The React UI in `ui/src/api/client.ts` manually defines TypeScript interfaces
(`SimulationFrame`, `SimulationConfig`, `EngineStatus`) that duplicate the
Pydantic models in `src/api/models/requests.py` and `src/api/models/responses.py`.
When a field changes in Python, TypeScript silently becomes stale.

## Requirements

1. Add `pydantic2ts` (or `datamodel-code-generator`) to dev dependencies
2. Create a script `scripts/generate_ts_types.py` that:
   - Reads all Pydantic models from `src/api/models/requests.py` and `responses.py`
   - Generates `ui/src/api/generated-types.ts`
   - Includes a header comment warning not to edit manually
3. Add a `make types` target to the Makefile
4. Add a pre-commit hook or CI check that verifies generated types are up to date
5. Migrate existing manual interfaces in `client.ts` to import from `generated-types.ts`

## Files to create/modify

| Action | File |
|--------|------|
| Create | `scripts/generate_ts_types.py` |
| Create | `ui/src/api/generated-types.ts` (auto-generated) |
| Modify | `ui/src/api/client.ts` — import from generated types |
| Modify | `ui/src/api/useEngineManager.ts` — import from generated types |
| Modify | `Makefile` — add `types` target |
| Modify | `pyproject.toml` — add dev dependency |

## Acceptance criteria

- [ ] `make types` generates `ui/src/api/generated-types.ts` from Pydantic models
- [ ] React code imports types from generated file, not manual definitions
- [ ] CI fails if generated types are stale (optional but recommended)
- [ ] `SimulationRequest`, `SimulationResponse`, `EngineStatusResponse`, `AnalysisRequest`, `AnalysisResponse`, `VideoAnalysisRequest`, `VideoAnalysisResponse` all have TS equivalents
```

---

### Issue #0.2 — Create feature parity tracker

**Labels:** `enhancement`, `dual-gui`, `phase-0`, `tracking`

```markdown
## Summary

Create a machine-readable feature parity registry that tracks which features
are available in which UI (PyQt6 vs React), enabling CI to report parity
coverage and developers to identify what's missing.

## Requirements

1. Create `src/config/feature_parity.json` with this schema:
   ```json
   {
     "features": [
       {
         "id": "simulation.basic",
         "name": "Basic simulation (start/stop/pause)",
         "category": "simulation",
         "pyqt": true,
         "react": true,
         "api_endpoint": "/api/ws/simulate/{engine}",
         "notes": ""
       },
       {
         "id": "analysis.joint_angles_plot",
         "name": "Joint angles plot",
         "category": "analysis",
         "pyqt": true,
         "react": false,
         "api_endpoint": null,
         "notes": "Blocked on #2.1"
       }
     ]
   }
   ```
2. Populate with ALL features from the gap analysis table in
   `docs/architecture/dual-gui-architecture-review.md`
3. Create `scripts/report_feature_parity.py` that reads the JSON and outputs
   a summary table:
   ```
   Feature Parity Report
   =====================
   Total features: 42
   Both UIs:       12 (28%)
   PyQt6 only:     28 (67%)
   React only:      2 (5%)
   ```
4. Add as a CI reporting step (non-blocking)

## Feature categories to enumerate

- **Simulation**: basic, recording, playback, async, parameter sweep
- **Visualization**: 3D scene, camera presets, force vectors, contact forces, overlay
- **Analysis/Plotting**: all 20 plot types from `dashboard/window.py`, plus 5 counterfactual types
- **Export**: JSON, CSV, MAT, HDF5, C3D
- **Engine Management**: discover, load, unload, probe, switch, model selection
- **UI Chrome**: keyboard shortcuts, toast notifications, diagnostics panel, status bar

## Acceptance criteria

- [ ] `src/config/feature_parity.json` exists with >= 40 features enumerated
- [ ] `scripts/report_feature_parity.py` outputs clean summary
- [ ] Each feature has a clear `id`, `category`, and `pyqt`/`react` booleans
```

---

### Issue #0.3 — Extend launcher manifest with page layout specifications

**Labels:** `enhancement`, `dual-gui`, `phase-0`

```markdown
## Summary

Extend `src/config/launcher_manifest.json` (or create a sibling
`ui_layout_manifest.json`) to declaratively describe page layouts so both
PyQt6 and React UIs can read the same structural specification and render
their respective implementations.

## Context

`launcher_manifest.json` already serves as "single source of truth for all
launcher tiles across PyQt and Tauri/React UIs." This pattern should extend
to page layouts. The manifest should describe *what* sections appear on each
page and in *what arrangement*, not *how* to render them.

## Requirements

1. Add a `pages` key to the manifest (or create `src/config/ui_layout_manifest.json`):
   ```json
   {
     "pages": {
       "dashboard": {
         "layout": "grid",
         "sections": ["tile_grid", "status_bar"],
         "description": "Main launcher / tile selector"
       },
       "simulation": {
         "layout": "sidebar-main-sidebar",
         "left_sidebar": {
           "sections": [
             { "id": "engine_selector", "label": "Physics Engines" },
             { "id": "parameter_panel", "label": "Parameters" },
             { "id": "simulation_controls", "label": "Controls" }
           ]
         },
         "main": {
           "sections": [
             { "id": "3d_visualization", "label": "3D View", "flex": 3 },
             { "id": "live_plot", "label": "Live Analysis", "flex": 1 }
           ]
         },
         "right_sidebar": {
           "sections": [
             { "id": "live_data", "label": "Live Data" }
           ]
         }
       },
       "analysis": {
         "layout": "sidebar-main",
         "left_sidebar": {
           "sections": [
             { "id": "plot_type_selector", "label": "Plot Type" },
             { "id": "analysis_type_selector", "label": "Analysis" },
             { "id": "export_controls", "label": "Export" }
           ]
         },
         "main": {
           "sections": [
             { "id": "plot_canvas", "label": "Plot", "flex": 2 },
             { "id": "data_table", "label": "Data", "flex": 1 }
           ]
         }
       }
     }
   }
   ```
2. Create a Pydantic model `src/config/layout_models.py` for validation
3. Create a loader `src/config/layout_loader.py` used by both UIs
4. Create a React hook `ui/src/api/useLayoutManifest.ts` to fetch layout specs

## Acceptance criteria

- [ ] Layout manifest defines at least `dashboard`, `simulation`, `analysis` pages
- [ ] Pydantic model validates the manifest at startup
- [ ] PyQt6 launcher can read the manifest (even if not yet used for rendering)
- [ ] React hook can fetch and parse the manifest
```

---

## Phase 1: Extract Shared Services

The goal of Phase 1 is to extract business logic currently embedded in PyQt6
widget code into framework-agnostic Python services. These services accept
data and return structured results — no Qt, no Matplotlib rendering, no UI.

---

### Issue #1.1 — Create PlottingOrchestrator service

**Labels:** `enhancement`, `dual-gui`, `phase-1`, `backend`

```markdown
## Summary

Extract the plot-type dispatch logic from `UnifiedDashboardWindow.refresh_static_plot()`
into a standalone `PlottingOrchestrator` service that returns structured data (dicts/arrays),
not rendered figures. This is the single highest-value task in the dual-GUI plan.

## Context

`src/shared/python/dashboard/window.py:266-388` contains a 20-case dispatch
that calls `GolfSwingPlotter` methods, each of which renders directly to a
Matplotlib figure. This tightly couples plot data extraction to Matplotlib
rendering and PyQt6 widgets.

The existing `GolfSwingPlotter` (in `src/shared/python/plotting/core.py`)
already delegates to renderer subclasses (`KinematicsRenderer`,
`EnergyRenderer`, etc.) via a `DataManager`. The `DataManager` +
`RecorderInterface` in `src/shared/python/plotting/transforms.py` already
abstract data access. We need to add a layer that returns data as
serializable dicts, above the renderers.

## Requirements

1. Create `src/shared/python/services/__init__.py`
2. Create `src/shared/python/services/plotting_orchestrator.py`:
   ```python
   class PlotDataResult:
       """Structured plot data, JSON-serializable."""
       plot_type: str
       title: str
       series: list[PlotSeries]  # Each has x, y, label, color
       axes_labels: dict[str, str]  # {"x": "Time (s)", "y": "Angle (rad)"}
       metadata: dict[str, Any]

   class PlotSeries:
       x: list[float]
       y: list[float]
       label: str
       color: str | None = None
       style: str = "line"  # "line", "scatter", "bar", "heatmap"

   class PlottingOrchestrator:
       """Returns plot data without rendering. UI-framework-agnostic."""

       AVAILABLE_PLOT_TYPES: list[str]  # All 20 types

       def __init__(self, recorder: RecorderInterface):
           ...

       def get_available_types(self) -> list[str]:
           ...

       def get_plot_data(self, plot_type: str, **kwargs) -> PlotDataResult:
           """Main dispatch. Returns structured data for any plot type."""
           ...

       def render_to_image(
           self, plot_type: str, format: str = "png",
           width: int = 800, height: int = 600, dpi: int = 100,
           **kwargs
       ) -> bytes:
           """Convenience: render via Matplotlib and return image bytes."""
           ...
   ```
3. Implement `get_plot_data()` for all 20 plot types from `window.py:119-141`:
   - `joint_angles`, `joint_velocities`, `joint_torques` → time series
   - `energies`, `club_head_speed`, `angular_momentum` → time series
   - `power_flow`, `joint_power_curves`, `impulse_accumulation` → time series
   - `phase_diagram` → 2D scatter (angle vs velocity)
   - `poincare_map_3d` → 3D scatter
   - `lyapunov_exponent` → time series
   - `recurrence_plot` → 2D heatmap (uses `StatisticalAnalyzer`)
   - `stability_diagram`, `cop_trajectory` → 2D trajectory
   - `grf_butterfly_diagram` → 2D multi-series
   - `club_head_trajectory` → 3D trajectory
   - `kinematic_sequence_bars` → bar chart data
   - `swing_profile_radar` → radar chart data (uses `StatisticalAnalyzer`)
   - `summary_dashboard` → composite of multiple plots
4. Implement `render_to_image()` that creates a Matplotlib figure, calls
   the existing `GolfSwingPlotter` method, and returns PNG/SVG bytes
5. Define Pydantic models for all return types (for API serialization)
6. Write unit tests in `tests/shared/test_plotting_orchestrator.py`

## Key source files to reference

- `src/shared/python/dashboard/window.py:266-388` — dispatch logic to extract
- `src/shared/python/plotting/core.py` — `GolfSwingPlotter` and its renderers
- `src/shared/python/plotting/transforms.py` — `DataManager`, `RecorderInterface`
- `src/shared/python/statistical_analysis.py` — `StatisticalAnalyzer` (for radar, recurrence)
- `src/shared/python/dashboard/recorder.py` — `GenericPhysicsRecorder`

## Files to create/modify

| Action | File |
|--------|------|
| Create | `src/shared/python/services/__init__.py` |
| Create | `src/shared/python/services/plotting_orchestrator.py` |
| Create | `src/shared/python/services/models.py` (Pydantic data models) |
| Create | `tests/shared/test_plotting_orchestrator.py` |

## Acceptance criteria

- [ ] `PlottingOrchestrator.get_plot_data("joint_angles")` returns serializable `PlotDataResult`
- [ ] All 20 plot types from the dashboard are supported
- [ ] `render_to_image()` returns valid PNG bytes for all 20 types
- [ ] No PyQt6 or Matplotlib imports at module level (Matplotlib only in `render_to_image`)
- [ ] Unit tests cover at least 10 of the 20 plot types with mock recorder data
- [ ] `PlotDataResult` has a `.to_dict()` that produces JSON-safe output
```

---

### Issue #1.2 — Create RecordingService

**Labels:** `enhancement`, `dual-gui`, `phase-1`, `backend`

```markdown
## Summary

Create a `RecordingService` that wraps `GenericPhysicsRecorder` with a
stateless, session-based API suitable for consumption by both PyQt6 (direct)
and React (via REST).

## Context

`GenericPhysicsRecorder` in `src/shared/python/dashboard/recorder.py` is
currently instantiated and held by PyQt6 widgets. The React UI has no access
to recording functionality. We need a service layer that manages recorder
instances keyed by session/engine, so the API can expose recording controls.

## Requirements

1. Create `src/shared/python/services/recording_service.py`:
   ```python
   class RecordingSession:
       id: str
       engine_type: str
       recorder: GenericPhysicsRecorder
       started_at: datetime
       is_recording: bool
       frame_count: int

   class RecordingService:
       def create_session(self, engine: PhysicsEngine) -> RecordingSession:
           ...
       def start_recording(self, session_id: str) -> None:
           ...
       def stop_recording(self, session_id: str) -> None:
           ...
       def get_status(self, session_id: str) -> RecordingSession:
           ...
       def get_frame_count(self, session_id: str) -> int:
           ...
       def get_time_series(self, session_id: str, field: str) -> dict:
           """Returns {times: [...], values: [...]}"""
           ...
       def get_data_dict(self, session_id: str) -> dict:
           """Returns full recording data for export."""
           ...
       def cleanup_session(self, session_id: str) -> None:
           ...
   ```
2. Manage concurrent sessions with cleanup (TTL-based, like `TaskManager`)
3. Wire into FastAPI `app.state` at startup (alongside `SimulationService`)
4. Update `SimulationService` to create recording sessions during simulations
5. Write unit tests

## Files to create/modify

| Action | File |
|--------|------|
| Create | `src/shared/python/services/recording_service.py` |
| Modify | `src/api/server.py` — register service in `app.state` |
| Modify | `src/api/dependencies.py` — add `get_recording_service` |
| Create | `tests/shared/test_recording_service.py` |

## Acceptance criteria

- [ ] `RecordingService` can create, start, stop, and query sessions
- [ ] Sessions have TTL-based cleanup (configurable, default 1 hour)
- [ ] `get_time_series()` returns JSON-serializable data
- [ ] Unit tests cover full lifecycle (create → start → record → stop → query → cleanup)
```

---

### Issue #1.3 — Create CounterfactualAnalysisService

**Labels:** `enhancement`, `dual-gui`, `phase-1`, `backend`

```markdown
## Summary

Extract the post-hoc counterfactual analysis logic from `UnifiedDashboardWindow`
into a standalone service that can be called from either UI.

## Context

`window.py:390-411` (`compute_analysis`) calls `self.recorder.compute_analysis_post_hoc()`
and `window.py:413-442` (`show_analysis_plot`) dispatches 5 counterfactual analysis types:
- ZTCF vs ZVCF
- Induced Acceleration (Gravity)
- Induced Acceleration (Control)
- Club Induced Acceleration Breakdown
- Stability Metrics

The actual computation lives in `GenericPhysicsRecorder.compute_analysis_post_hoc()`
and the advanced analysis module at `src/shared/python/dashboard/advanced_analysis.py`.
This logic is already mostly UI-independent; we just need a clean service wrapper.

## Requirements

1. Create `src/shared/python/services/counterfactual_service.py`:
   ```python
   class CounterfactualResult:
       analysis_type: str
       computed_at: datetime
       data: dict[str, Any]  # Contains arrays for plotting
       metadata: dict[str, Any]

   class CounterfactualService:
       AVAILABLE_TYPES = [
           "ztcf_vs_zvcf",
           "induced_accel_gravity",
           "induced_accel_control",
           "club_induced_accel",
           "stability_metrics",
       ]

       def compute(self, recorder: GenericPhysicsRecorder) -> None:
           """Run all post-hoc analysis. Populates recorder internal buffers."""
           ...

       def get_result(
           self, recorder: GenericPhysicsRecorder, analysis_type: str,
           **kwargs
       ) -> CounterfactualResult:
           """Extract result for a specific analysis type."""
           ...
   ```
2. Pydantic models for the result types
3. Unit tests with mock recorder

## Files to create/modify

| Action | File |
|--------|------|
| Create | `src/shared/python/services/counterfactual_service.py` |
| Create | `tests/shared/test_counterfactual_service.py` |

## Acceptance criteria

- [ ] All 5 counterfactual analysis types return structured results
- [ ] Results are JSON-serializable
- [ ] No PyQt6 imports
- [ ] Tests cover all 5 types
```

---

### Issue #1.4 — Enhance ExportService for multi-format support

**Labels:** `enhancement`, `dual-gui`, `phase-1`, `backend`

```markdown
## Summary

Create an `ExportService` wrapping the existing `src/shared/python/export.py`
module with a clean API that both UIs can use, and that the FastAPI endpoint
can expose.

## Context

`src/shared/python/export.py` already has `export_to_matlab()`, `export_to_csv()`,
`export_to_hdf5()`, `export_to_c3d()`, `export_to_json()`, and
`get_available_export_formats()`. But the current API route (`src/api/routes/export.py`)
only supports exporting completed task results as JSON — it doesn't support the
multi-format file export that the PyQt6 dashboard has.

## Requirements

1. Create `src/shared/python/services/export_service.py`:
   ```python
   class ExportResult:
       format: str
       success: bool
       file_path: str | None  # For file exports
       data: bytes | None  # For in-memory exports
       error: str | None

   class ExportService:
       def get_available_formats(self) -> dict[str, dict]:
           """Returns format metadata (name, extension, available)."""
           ...

       def export_to_file(
           self, data: dict, format: str, output_path: str
       ) -> ExportResult:
           ...

       def export_to_bytes(
           self, data: dict, format: str
       ) -> ExportResult:
           """Export to in-memory bytes (for HTTP download responses)."""
           ...
   ```
2. Wrap existing functions from `src/shared/python/export.py`
3. Add `export_to_bytes()` variants for server-side generation (streaming downloads)
4. Unit tests

## Files to create/modify

| Action | File |
|--------|------|
| Create | `src/shared/python/services/export_service.py` |
| Create | `tests/shared/test_export_service.py` |

## Acceptance criteria

- [ ] All 5 formats (JSON, CSV, MAT, HDF5, C3D) supported
- [ ] `export_to_bytes()` returns bytes for HTTP streaming
- [ ] `get_available_formats()` reports which formats are available (based on installed deps)
- [ ] Graceful handling when scipy/h5py/ezc3d not installed
```

---

### Issue #1.5 — Refactor UnifiedDashboardWindow to use shared services

**Labels:** `refactor`, `dual-gui`, `phase-1`, `pyqt`

**Depends on:** #1.1, #1.2, #1.3, #1.4

```markdown
## Summary

Refactor `src/shared/python/dashboard/window.py` to use the new shared services
(`PlottingOrchestrator`, `RecordingService`, `CounterfactualService`, `ExportService`)
instead of inline business logic. This is the proof-of-concept that the shared service
layer works for PyQt6.

## Requirements

1. Replace the 20-case `refresh_static_plot()` dispatch with:
   ```python
   def refresh_static_plot(self):
       plot_type = self.plot_type_combo.currentText()
       # Get data from service
       result = self.plotting_service.get_plot_data(plot_type_id)
       # Render to canvas (still Matplotlib, but data comes from service)
       self._render_plot_result(result, self.static_canvas)
   ```
2. Replace `compute_analysis()` with `CounterfactualService.compute()`
3. Replace `show_analysis_plot()` with `CounterfactualService.get_result()`
4. Replace `export_data()` with `ExportService.export_to_file()`
5. Keep the PyQt6 rendering layer (canvas, signals, widgets) — only extract the logic
6. Verify all existing functionality still works via manual testing or existing tests

## Files to modify

| Action | File |
|--------|------|
| Modify | `src/shared/python/dashboard/window.py` |
| Verify | `src/shared/python/dashboard/tests/test_widgets.py` — still passes |

## Acceptance criteria

- [ ] `window.py` no longer contains plot dispatch logic (delegated to `PlottingOrchestrator`)
- [ ] `window.py` no longer directly calls `recorder.compute_analysis_post_hoc()` (delegated)
- [ ] `window.py` no longer directly calls `export_recording_all_formats()` (delegated)
- [ ] All 20 plot types still render correctly in the PyQt6 dashboard
- [ ] Existing tests pass
```

---

### Issue #1.6 — Refactor MuJoCo AdvancedGolfAnalysisWindow to use shared services

**Labels:** `refactor`, `dual-gui`, `phase-1`, `pyqt`

**Depends on:** #1.1, #1.2

```markdown
## Summary

Refactor the MuJoCo GUI at
`src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/gui/core/main_window.py`
to use the shared `PlottingOrchestrator` for its Live Analysis tab and any
shared plotting it does. This ensures the MuJoCo-specific GUI also benefits
from the service extraction.

## Context

The MuJoCo GUI (`AdvancedGolfAnalysisWindow`) is the most feature-rich PyQt6 app
with 9 tabs. Its `PlottingTab` and `AnalysisTab` have their own plotting logic
that partially overlaps with the dashboard's. The `LivePlotWidget` is already
shared from `src/shared/python/dashboard/widgets.py`.

## Requirements

1. Wire `PlottingOrchestrator` into the MuJoCo GUI's analysis/plotting tabs
2. Where the MuJoCo GUI has engine-specific plotting (e.g., MuJoCo-specific
   contact forces, actuator torques), keep those as engine-specific extensions
3. Ensure `LivePlotWidget` can optionally use `PlottingOrchestrator.get_plot_data()`
   for its data instead of direct recorder access

## Files to modify

| Action | File |
|--------|------|
| Modify | `src/engines/.../mujoco/.../gui/core/main_window.py` |
| Modify | `src/engines/.../mujoco/.../gui/tabs/plotting_tab.py` |
| Modify | `src/engines/.../mujoco/.../gui/tabs/analysis_tab.py` |

## Acceptance criteria

- [ ] MuJoCo GUI uses `PlottingOrchestrator` for shared plot types
- [ ] MuJoCo-specific plots remain in engine-specific code
- [ ] No regressions in MuJoCo GUI functionality
```

---

## Phase 2: API Endpoints

---

### Issue #2.1 — Plot data and server-side rendering API endpoints

**Labels:** `enhancement`, `dual-gui`, `phase-2`, `api`

**Depends on:** #1.1

```markdown
## Summary

Expose the `PlottingOrchestrator` via FastAPI endpoints so the React UI can
request plot data as JSON or rendered images.

## Requirements

1. Create `src/api/routes/plots.py` with:
   ```python
   router = APIRouter(prefix="/api/plots")

   @router.get("/types")
   async def list_plot_types() -> list[str]:
       """Return all available plot types."""

   @router.get("/data/{plot_type}")
   async def get_plot_data(
       plot_type: str,
       joint_idx: int = 0,
       session_id: str | None = None,
   ) -> PlotDataResult:
       """Return structured plot data as JSON."""

   @router.get("/render/{plot_type}")
   async def render_plot(
       plot_type: str,
       format: str = "png",  # png, svg
       width: int = 800,
       height: int = 600,
       dpi: int = 100,
       session_id: str | None = None,
   ) -> StreamingResponse:
       """Return rendered plot image."""
   ```
2. Register in `src/api/server.py`
3. Add Pydantic request/response models for query parameters
4. Integration tests using `httpx.AsyncClient` / `TestClient`

## Files to create/modify

| Action | File |
|--------|------|
| Create | `src/api/routes/plots.py` |
| Modify | `src/api/server.py` — register router |
| Modify | `src/api/models/requests.py` — add `PlotRequest` if needed |
| Modify | `src/api/models/responses.py` — add `PlotDataResponse` |
| Create | `tests/api/test_plots_routes.py` |

## Acceptance criteria

- [ ] `GET /api/plots/types` returns list of 20 plot types
- [ ] `GET /api/plots/data/joint_angles` returns JSON with `series`, `axes_labels`
- [ ] `GET /api/plots/render/joint_angles?format=png` returns valid PNG
- [ ] `GET /api/plots/render/joint_angles?format=svg` returns valid SVG
- [ ] Error handling for invalid plot types (404), no data available (400)
- [ ] Integration tests pass
```

---

### Issue #2.2 — Recording REST API endpoints

**Labels:** `enhancement`, `dual-gui`, `phase-2`, `api`

**Depends on:** #1.2

```markdown
## Summary

Expose `RecordingService` via REST endpoints so the React UI can control
recording during simulation.

## Requirements

1. Create `src/api/routes/recording.py`:
   ```python
   router = APIRouter(prefix="/api/recording")

   @router.post("/sessions")
   async def create_session(engine_type: str) -> RecordingSessionResponse

   @router.post("/sessions/{session_id}/start")
   async def start_recording(session_id: str) -> dict

   @router.post("/sessions/{session_id}/stop")
   async def stop_recording(session_id: str) -> dict

   @router.get("/sessions/{session_id}")
   async def get_session_status(session_id: str) -> RecordingSessionResponse

   @router.get("/sessions/{session_id}/data/{field}")
   async def get_time_series(session_id: str, field: str) -> dict

   @router.delete("/sessions/{session_id}")
   async def delete_session(session_id: str) -> dict
   ```
2. Register in server.py
3. Integration tests

## Files to create/modify

| Action | File |
|--------|------|
| Create | `src/api/routes/recording.py` |
| Modify | `src/api/server.py` |
| Create | `tests/api/test_recording_routes.py` |

## Acceptance criteria

- [ ] Full CRUD lifecycle for recording sessions works via REST
- [ ] Recording integrates with WebSocket simulation (frames get recorded)
- [ ] Session cleanup works (TTL and explicit delete)
```

---

### Issue #2.3 — Counterfactual analysis API endpoint

**Labels:** `enhancement`, `dual-gui`, `phase-2`, `api`

**Depends on:** #1.3

```markdown
## Summary

Add endpoints for triggering post-hoc counterfactual analysis and retrieving
results.

## Requirements

1. Add to `src/api/routes/analysis.py`:
   ```python
   @router.post("/analysis/counterfactual/compute")
   async def compute_counterfactual(session_id: str) -> dict

   @router.get("/analysis/counterfactual/{analysis_type}")
   async def get_counterfactual_result(
       analysis_type: str, session_id: str
   ) -> CounterfactualResult

   @router.get("/analysis/counterfactual/types")
   async def list_counterfactual_types() -> list[str]
   ```
2. Integration tests

## Acceptance criteria

- [ ] All 5 counterfactual types available via API
- [ ] Compute endpoint triggers analysis and returns status
- [ ] Result endpoint returns structured data
```

---

### Issue #2.4 — Multi-format export API endpoints

**Labels:** `enhancement`, `dual-gui`, `phase-2`, `api`

**Depends on:** #1.4

```markdown
## Summary

Replace the current limited export route with full multi-format export
endpoints that support streaming file downloads.

## Requirements

1. Enhance `src/api/routes/export.py`:
   ```python
   @router.get("/export/formats")
   async def list_export_formats() -> dict

   @router.post("/export/{format}")
   async def export_data(
       format: str,
       session_id: str,
   ) -> StreamingResponse:
       """Export recording data as downloadable file."""

   @router.post("/export/all")
   async def export_all_formats(
       session_id: str,
       formats: list[str] | None = None,
   ) -> dict:
       """Export to multiple formats, return download URLs."""
   ```
2. Use `StreamingResponse` with appropriate `Content-Type` and
   `Content-Disposition` headers for file downloads
3. Integration tests

## Files to modify

| Action | File |
|--------|------|
| Modify | `src/api/routes/export.py` — replace with new endpoints |
| Create | `tests/api/test_export_routes.py` |

## Acceptance criteria

- [ ] `GET /export/formats` returns available formats with availability flags
- [ ] `POST /export/csv` returns downloadable CSV file
- [ ] `POST /export/mat` returns downloadable .mat file (if scipy installed)
- [ ] Proper error when format dependency missing (h5py, ezc3d, scipy)
```

---

### Issue #2.5 — Model configuration and camera preset API endpoints

**Labels:** `enhancement`, `dual-gui`, `phase-2`, `api`

```markdown
## Summary

Add API endpoints for model configuration (listing available models per engine)
and camera presets, so the React UI can offer the same model selection and
camera controls as the PyQt6 GUIs.

## Requirements

1. Add to engine routes or create new route:
   ```python
   @router.get("/api/engines/{engine_type}/models")
   async def list_models(engine_type: str) -> list[ModelConfig]:
       """Return available model configurations for an engine."""

   @router.post("/api/engines/{engine_type}/camera")
   async def set_camera_preset(
       engine_type: str, preset: CameraPresetRequest
   ) -> dict:
       """Set camera to a named preset or custom values."""

   @router.get("/api/engines/{engine_type}/camera/presets")
   async def list_camera_presets(engine_type: str) -> list[CameraPreset]
   ```
2. Model configs should come from the same source as the MuJoCo PhysicsTab's
   `model_configs` list
3. Camera presets should match the MuJoCo GUI's preset system (side, front, top,
   follow, down-the-line)

## Acceptance criteria

- [ ] Model list endpoint returns engine-specific model configurations
- [ ] Camera preset endpoint returns named presets with azimuth/elevation/distance
- [ ] React UI can query and display model options
```

---

### Issue #2.6 — API contract test suite

**Labels:** `testing`, `dual-gui`, `phase-2`

**Depends on:** #2.1, #2.2, #2.3, #2.4, #2.5

```markdown
## Summary

Create a comprehensive API contract test suite that verifies all endpoints
return data matching their Pydantic response models. This test suite serves
as the single source of truth for both frontends.

## Requirements

1. Create `tests/api/test_contracts.py`:
   - Test every endpoint returns valid response model
   - Test error cases return proper HTTP status codes
   - Test that response shapes match what the React UI expects
2. Use `httpx.AsyncClient` with the FastAPI test client
3. Add to CI pipeline

## Acceptance criteria

- [ ] Every API endpoint has at least one contract test
- [ ] Tests verify response Pydantic model validation
- [ ] Tests run in CI on every PR
```

---

## Phase 3: React UI Feature Parity

---

### Issue #3.1 — Analysis page with PlotViewer component

**Labels:** `enhancement`, `dual-gui`, `phase-3`, `react`

**Depends on:** #2.1

```markdown
## Summary

Create a new Analysis page in the React UI that displays all 20+ plot types
from the backend, using both JSON data (for interactive plots) and server-
rendered images (for complex plots).

## Requirements

1. Create `ui/src/pages/Analysis.tsx`:
   - Left sidebar: plot type selector dropdown (fetched from `/api/plots/types`)
   - Left sidebar: analysis type selector (counterfactual types)
   - Main area: plot display
   - Right sidebar: data table showing underlying values
2. Create `ui/src/components/analysis/PlotViewer.tsx`:
   - For simple time-series plots: render with Recharts using JSON data from
     `/api/plots/data/{type}`
   - For complex plots (3D, radar, heatmap): display server-rendered image from
     `/api/plots/render/{type}?format=svg`
   - Loading state, error state
3. Add route to `App.tsx`: `<Route path="/analysis" element={<AnalysisPage />} />`
4. Add navigation link from Dashboard and Simulation pages

## Files to create/modify

| Action | File |
|--------|------|
| Create | `ui/src/pages/Analysis.tsx` |
| Create | `ui/src/components/analysis/PlotViewer.tsx` |
| Create | `ui/src/api/usePlots.ts` — hooks for plot data/rendering |
| Modify | `ui/src/App.tsx` — add route |
| Create | `ui/src/pages/Analysis.test.tsx` |

## Acceptance criteria

- [ ] Analysis page renders with plot type selector
- [ ] Selecting a plot type fetches and displays the plot
- [ ] Time-series plots render interactively with Recharts
- [ ] Complex plots (radar, 3D, heatmap) display as server-rendered SVG
- [ ] Loading and error states handled
- [ ] Vitest tests pass
```

---

### Issue #3.2 — Recording controls React component

**Labels:** `enhancement`, `dual-gui`, `phase-3`, `react`

**Depends on:** #2.2

```markdown
## Summary

Add recording controls to the React Simulation page so users can start/stop
recording and see frame count, mirroring the PyQt6 overlay's REC button.

## Requirements

1. Create `ui/src/components/simulation/RecordingControls.tsx`:
   - Record button (toggles start/stop)
   - Frame counter display
   - Recording duration display
   - Visual indicator (red dot when recording)
2. Create `ui/src/api/useRecording.ts` — hook wrapping recording REST API
3. Integrate into `Simulation.tsx` overlay area

## Files to create/modify

| Action | File |
|--------|------|
| Create | `ui/src/components/simulation/RecordingControls.tsx` |
| Create | `ui/src/api/useRecording.ts` |
| Modify | `ui/src/pages/Simulation.tsx` — add recording controls |
| Create | `ui/src/components/simulation/RecordingControls.test.tsx` |

## Acceptance criteria

- [ ] Record button starts/stops recording via REST API
- [ ] Frame count updates in real-time
- [ ] Red recording indicator visible when recording
- [ ] Works alongside WebSocket simulation streaming
```

---

### Issue #3.3 — Export page in React UI

**Labels:** `enhancement`, `dual-gui`, `phase-3`, `react`

**Depends on:** #2.4

```markdown
## Summary

Create an Export page (or dialog) in the React UI that shows available formats,
lets the user select formats, and triggers file downloads.

## Requirements

1. Create `ui/src/pages/Export.tsx` or `ui/src/components/simulation/ExportDialog.tsx`:
   - List available formats from `/api/export/formats` with availability status
   - Checkboxes to select formats
   - Export button that triggers download via `/api/export/{format}`
   - Handle file download via `<a download>` or `URL.createObjectURL`
2. Add route or modal trigger

## Acceptance criteria

- [ ] Shows available export formats with installed/not-installed status
- [ ] User can select formats and trigger export
- [ ] Files download correctly in the browser
- [ ] Unavailable formats are disabled with explanation
```

---

### Issue #3.4 — Enhanced simulation controls (model selection, parameters)

**Labels:** `enhancement`, `dual-gui`, `phase-3`, `react`

**Depends on:** #2.5

```markdown
## Summary

Enhance the React Simulation page's controls to support model selection per
engine (currently it only selects engine type, not specific models within an
engine) and richer parameter configuration.

## Requirements

1. Enhance `ParameterPanel.tsx`:
   - Fetch model configs from `/api/engines/{type}/models`
   - Model selector dropdown (e.g., "humanoid_golf", "simple_arm")
   - Engine-specific parameter groups
2. Enhance `SimulationControls.tsx`:
   - Reset button (currently missing)
   - Step-by-step mode (single step)

## Acceptance criteria

- [ ] Model selector shows available models for the loaded engine
- [ ] Selecting a model loads it via the API
- [ ] Reset and step buttons work
```

---

### Issue #3.5 — Keyboard shortcuts for React UI

**Labels:** `enhancement`, `dual-gui`, `phase-3`, `react`

```markdown
## Summary

Add keyboard shortcuts to the React UI matching the PyQt6 GUI's shortcuts:
Space (play/pause), R (reset), 1-5 (camera presets), O (toggle overlay),
H (help).

## Requirements

1. Create `ui/src/hooks/useKeyboardShortcuts.ts`
2. Register shortcuts globally in `App.tsx`
3. Show shortcut help (similar to PyQt6's help group)
4. Shortcuts configurable via layout manifest (future)

## Key mapping (from `main_window.py:230-278`)

| Key | Action |
|-----|--------|
| Space | Play/Pause simulation |
| R | Reset simulation |
| 1-5 | Camera presets (side, front, top, follow, down-the-line) |
| O | Toggle overlay |
| H | Toggle help panel |
| Escape | Stop simulation |

## Acceptance criteria

- [ ] All shortcuts from the mapping work
- [ ] Shortcuts don't fire when typing in input fields
- [ ] Help overlay shows available shortcuts
```

---

### Issue #3.6 — Camera presets in React 3D view

**Labels:** `enhancement`, `dual-gui`, `phase-3`, `react`

**Depends on:** #2.5

```markdown
## Summary

Add camera preset buttons and keyboard shortcuts to the React 3D visualization,
mirroring the PyQt6 MuJoCo GUI's camera system.

## Requirements

1. Fetch presets from `/api/engines/{type}/camera/presets`
2. Add preset buttons to the 3D view overlay
3. Wire keyboard shortcuts (1-5) from #3.5
4. Animate camera transitions (using R3F's camera controls)

## Acceptance criteria

- [ ] Camera preset buttons appear as overlay on 3D view
- [ ] Clicking a preset smoothly transitions the camera
- [ ] Keyboard shortcuts 1-5 trigger corresponding presets
```

---

### Issue #3.7 — Add Plotly.js for interactive analysis plots

**Labels:** `enhancement`, `dual-gui`, `phase-3`, `react`

**Depends on:** #2.1

```markdown
## Summary

Add Plotly.js (via react-plotly.js) for rendering interactive versions of the
complex plot types that Recharts can't handle: 3D scatter, radar, heatmaps,
and contour plots.

## Context

Recharts handles simple 2D time-series well but cannot render:
- 3D scatter plots (Poincare map, club head trajectory)
- Radar/spider charts (swing profile)
- Heatmaps (recurrence plot)
- Contour plots (GRF butterfly diagram)

Plotly.js handles all of these with built-in interactivity (zoom, rotate, hover).

## Requirements

1. Add `react-plotly.js` and `plotly.js` to `ui/package.json`
2. Create `ui/src/components/analysis/InteractivePlot.tsx`:
   - Accepts `PlotDataResult` from the API
   - Automatically selects appropriate Plotly trace type based on `series.style`
   - Line → `scatter` mode lines
   - Scatter → `scatter` mode markers
   - Bar → `bar`
   - Heatmap → `heatmap`
   - 3D scatter → `scatter3d`
   - Radar → `scatterpolar`
3. Integrate into `PlotViewer.tsx` as the primary renderer (fallback to
   server-rendered images for unsupported types)

## Acceptance criteria

- [ ] Plotly.js renders time-series plots with zoom/pan
- [ ] 3D plots render with rotation controls
- [ ] Radar chart renders for swing profile
- [ ] Heatmap renders for recurrence plot
- [ ] Bundle size impact documented (Plotly is large — consider partial import)
```

---

## Phase 4: Polish & Parity

---

### Issue #4.1 — Cross-GUI integration tests

**Labels:** `testing`, `dual-gui`, `phase-4`

**Depends on:** Phase 2, Phase 3

```markdown
## Summary

Create integration tests that verify both UIs produce equivalent results
for the same operations. These tests call the shared API and verify the
data contracts are respected.

## Requirements

1. Create `tests/integration/test_cross_gui_parity.py`:
   - For each plot type: request data via API, verify it matches what
     `PlottingOrchestrator` returns directly
   - For export: verify exported file format is correct
   - For recording: verify session lifecycle works end-to-end
2. Create `tests/integration/test_react_api_compatibility.py`:
   - Verify every endpoint returns data that matches TypeScript type expectations
   - Use snapshot testing for response shapes

## Acceptance criteria

- [ ] API responses match direct service calls for all plot types
- [ ] Export format correctness verified
- [ ] Tests run in CI
```

---

### Issue #4.2 — Complete layout manifest for all pages

**Labels:** `enhancement`, `dual-gui`, `phase-4`

**Depends on:** #0.3, Phase 3

```markdown
## Summary

Once all React pages exist, populate the layout manifest with complete
specifications for every page and verify both UIs respect the manifest.

## Requirements

1. Add layout specs for: Dashboard, Simulation, Analysis, Export
2. Add validation that both UIs render all sections declared in the manifest
3. Add a CI check that flags when a manifest section has no implementation
   in either UI

## Acceptance criteria

- [ ] Manifest covers all pages in both UIs
- [ ] CI validation step passes
```

---

### Issue #4.3 — Developer guide for dual-GUI development

**Labels:** `documentation`, `dual-gui`, `phase-4`

**Depends on:** Phase 1

```markdown
## Summary

Write a developer guide explaining how to add new features to both UIs
using the shared service layer pattern.

## Requirements

1. Create `docs/development/dual-gui-developer-guide.md`:
   - Architecture overview with diagram
   - Step-by-step: "How to add a new plot type"
     1. Add data extraction to `PlottingOrchestrator`
     2. Add API endpoint (or it auto-discovers)
     3. Add to PyQt6 plot selector
     4. Add to React plot selector
   - Step-by-step: "How to add a new analysis feature"
   - Conventions: naming, file locations, testing expectations
   - FAQ: "Why two UIs?", "Which should I develop first?", "How to test parity?"
2. Link from main `CONTRIBUTING.md`

## Acceptance criteria

- [ ] Guide covers the full workflow for adding features
- [ ] Includes code examples for each step
- [ ] Reviewed by at least one contributor
```

---

## Summary: Issue Count by Phase

| Phase | Issues | Effort Estimate |
|-------|--------|-----------------|
| Phase 0: Foundation | 3 | Small (setup/config) |
| Phase 1: Extract Services | 6 | **Large** (core refactoring) |
| Phase 2: API Endpoints | 6 | Medium (wiring) |
| Phase 3: React UI | 7 | Medium-Large (new UI code) |
| Phase 4: Polish | 3 | Medium (testing/docs) |
| **Total** | **25** | |

## Recommended Sprint Assignment

| Sprint | Issues | Goal |
|--------|--------|------|
| Sprint 1 | #0.1, #0.2, #0.3 | Foundation in place |
| Sprint 2 | #1.1, #1.2, #1.3, #1.4 | All services created |
| Sprint 3 | #1.5, #1.6, #2.1, #2.2 | PyQt6 refactored, first API endpoints |
| Sprint 4 | #2.3, #2.4, #2.5, #2.6 | All API endpoints live |
| Sprint 5 | #3.1, #3.2, #3.3 | Core React pages |
| Sprint 6 | #3.4, #3.5, #3.6, #3.7 | React UI enhancements |
| Sprint 7 | #4.1, #4.2, #4.3 | Testing, docs, polish |
